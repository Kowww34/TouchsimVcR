"""
Simple I/O and filtering utilities for TouchSim-style objects.
    save(obj, kind, path, format="auto")
    load(path, kind=None, format="auto")
    filter_response(resp,
                    classes=None,
                    regions=None,
                    ids=None,
                    logic="and")

Overview
--------
This module provides two main capabilities:

1) Serialization (save/load)
   - Save and load TouchSim-related objects (afferent populations, stimuli,
     and responses).
   - Supports both general-purpose (pickle) and compact (NumPy binary) formats.

2) Response filtering
   - Efficient sub-selection of afferents from a response dictionary based on
     class, region, or index.
   - Designed to be chainable and preserve schema consistency.

Supported object kinds
----------------------
- "affpop"
    Afferent population objects (e.g., TouchSim afferent sets).

- "stimulus"
    Stimulus objects (e.g., indentation or vibration inputs).

- "response"
    Simulation outputs containing spike trains and associated metadata.

Supported storage formats
-------------------------
- "pickle"
    General-purpose Python serialization.
- "binary" (.npz)
    Compact NumPy-based storage (currently optimized for responses).

Response Dictionary Schema
--------------------------
Binary-loaded responses (and filter inputs) follow this structure:

    resp = {
        "spikes":      list[np.ndarray],   # length N (one per afferent)
        "location":    np.ndarray (N, 2),
        "class_str":   np.ndarray (N,),    # e.g., "PC", "RA", "SA1"
        "idx":         np.ndarray (N,),    # afferent indices
        "region_str":  np.ndarray (N,)     # region labels
    }

Optional (if present):
        "spike_times":  np.ndarray (total_spikes,)
        "spike_indptr": np.ndarray (N+1,)   # CSR layout


Filtering Behavior
------------------
filter_response(...) selects afferents using boolean masks derived from:
Output preserves the same dictionary schema (chainable).

- classes : class labels ("PC", "RA", etc.)
- regions : region labels (supports prefix matching, e.g., "D2" → "D2*")
- ids     : afferent indices

Selection rules:
- Each selector can be:
    * scalar (single value)
    * iterable (list/set of values)
    * callable (predicate function)
    * None (select all)

- logic:
    * "and" → intersection of all filters (default)
    * "or"  → union of all filters



Typical Usage
-------------
    #save a response (compact)
    save(resp, kind="response", path="resp.npz", format="binary")

    #load it
    r = load("resp.npz", kind="response")

    #filter by class and region
    r_pc = filter_response(r, classes="PC", regions="D2")

    #chain filters
    r_subset = filter_response(r_pc, ids=[1, 2, 3])

"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray


#public type aliases
ObjectKind = Literal["affpop", "stimulus", "response"]
SaveFormat = Literal["auto", "pickle", "binary"]


#minimal protocols so editors/type checkers know what is expected
class _AfferentProtocol(Protocol):
    @property
    def idx(self) -> int: ...


class _AfferentPopProtocol(Protocol):
    @property
    def location(self) -> Any: ...
    @property
    def affclass(self) -> Any: ...
    @property
    def afferents(self) -> Any: ...
    @property
    def region(self) -> tuple[Any, Any]: ...


class _ResponseProtocol(Protocol):
    @property
    def spikes(self) -> Any: ...
    @property
    def aff(self) -> _AfferentPopProtocol: ...
    def __len__(self) -> int: ...


#========================
#public API
#========================
def save(
    obj: Any,
    kind: ObjectKind,
    path: str | Path,
    format: SaveFormat = "auto",
) -> Path:
    """
    Parameters
    ----------
    obj
        The object to save.

    kind
        What kind of object is being saved.
        Must be one of:
            - "affpop"
            - "stimulus"
            - "response"

    path
        Output file path.

    format
        Storage format:
            - "pickle" : save the full Python object with pickle
            - "binary" : save a compact NumPy-based representation

    Returns
    -------
    Path
        The resolved output path.

    Notes
    -----
    Current defaults:
    - responses -> binary
    - affpop/stimulus -> pickle

    Raises
    ------
    ValueError
        If `kind` or `format` is invalid, or if the requested binary format
        is not implemented for that object kind.
    """
    path = Path(path)

    if format == "auto":
        format = _default_format(kind)

    if format == "pickle":
        _save_pickle(obj, path)
        return path

    if format == "binary":
        if kind == "response":
            _save_response_binary(obj, path)
            return path
        if kind == "affpop":
            raise ValueError(
                "binary save for kind='affpop' is not implemented yet. "
                "Use format='pickle' for now."
            )
        if kind == "stimulus":
            raise ValueError(
                "binary save for kind='stimulus' is not implemented yet. "
                "Use format='pickle' for now."
            )

    raise ValueError(f"unsupported save request: kind={kind!r}, format={format!r}")


def load(
    path: str | Path,
    kind: ObjectKind | None = None,
    format: SaveFormat = "auto",
) -> Any:
    """
    Load an object from disk.

    Parameters
    ----------
    path
        File path to load from.

    kind
        Expected object kind. Required for some binary formats so the loader
        knows how to interpret the stored arrays. Can be omitted for pickle.

    format
        Storage format:
            - "auto"   : infer from file suffix when possible
            - "pickle" : load with pickle
            - "binary" : load compact NumPy-based representation

    Returns
    -------
    Any
        The loaded object.

    Notes
    -----
    Current behavior:
    - pickle returns the original object
    - binary response returns a plain dictionary with minimal fields

    Raises
    ------
    ValueError
        If the format cannot be inferred, or if the requested binary loader
        is not implemented.
    """
    path = Path(path)

    if format == "auto":
        format = _infer_format(path)

    if format == "pickle":
        return _load_pickle(path)

    if format == "binary":
        if kind is None:
            raise ValueError(
                "kind must be provided when loading binary data."
            )
        if kind == "response":
            return _load_response_binary(path)
        if kind == "affpop":
            raise ValueError(
                "binary load for kind='affpop' is not implemented yet."
            )
        if kind == "stimulus":
            raise ValueError(
                "binary load for kind='stimulus' is not implemented yet."
            )

    raise ValueError(f"unsupported load request: kind={kind!r}, format={format!r}")


#========================
#format helpers
#========================
def _default_format(kind: ObjectKind) -> SaveFormat:
    """
    Choose the default storage format for a given object kind.

    Current policy:
    - response -> binary
    - affpop   -> pickle
    - stimulus -> pickle
    """
    if kind == "response":
        return "binary"
    return "pickle"


def _infer_format(path: Path) -> SaveFormat:
    """
    Infer storage format from the file suffix.

    Rules
    -----
    - .pkl / .pickle -> pickle
    - .npz           -> binary

    Raises
    ------
    ValueError
        If the suffix is unknown.
    """
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return "pickle"
    if suffix == ".npz":
        return "binary"
    raise ValueError(
        f"could not infer format from suffix {suffix!r}. "
        "Please pass format='pickle' or format='binary'."
    )


#========================
#pickle backend
#========================
def _save_pickle(obj: Any, path: Path) -> None:
    """
    Save any Python object using pickle.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path) -> Any:
    """
    Load any Python object previously saved with pickle.
    """
    with path.open("rb") as f:
        return pickle.load(f)


#========================
#binary response backend
#========================
def _save_response_binary(resp: _ResponseProtocol, path: Path) -> None:
    """
    Save a response object to a compact `.npz` file.

    Stored arrays
    -------------
    spike_times : float32, shape (total_spikes,)
        Concatenated spike times for all afferents.

    spike_indptr : int64, shape (N+1,)
        CSR-style pointer array where spikes for afferent `i` live in:
            spike_times[spike_indptr[i] : spike_indptr[i+1]]

    location : float32, shape (N, 2)
        Afferent x/y locations.

    class_str : unicode, shape (N,)
        Afferent class labels such as PC, RA, SA1.

    idx : int32, shape (N,)
        Afferent indices.

    region_str : unicode, shape (N,)
        Region labels.

    Why CSR?
    --------
    A Python list of spike arrays is convenient in memory, but wasteful on disk.
    Flattening to `spike_times + spike_indptr` avoids pickling thousands of
    tiny arrays and is usually much more compact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(resp)

    indptr = np.zeros(n + 1, dtype=np.int64)
    chunks: list[NDArray[np.float32]] = []

    for i, spike_train in enumerate(resp.spikes):
        arr = np.asarray(spike_train, dtype=np.float32)
        indptr[i + 1] = indptr[i] + arr.size
        if arr.size:
            chunks.append(arr)

    spike_times = (
        np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
    )

    location = np.asarray(resp.aff.location, dtype=np.float32)
    class_str = np.asarray(list(resp.aff.affclass), dtype="U8")
    idx = np.asarray([a.idx for a in resp.aff.afferents], dtype=np.int32)

    region_tags, _ = resp.aff.region
    region_str = np.asarray(
        list(map(str, region_tags)) if region_tags is not None else [""] * n,
        dtype="U32",
    )

    np.savez_compressed(
        path,
        spike_times=spike_times,
        spike_indptr=indptr,
        location=location,
        class_str=class_str,
        idx=idx,
        region_str=region_str,
    )


def _load_response_binary(path: Path) -> dict[str, Any]:
    """
    Load a response saved by `_save_response_binary`.

    Returns
    -------
    dict
        Dictionary with keys:
            - "spikes"
            - "location"
            - "class_str"
            - "idx"
            - "region_str"

    Notes
    -----
    This function reconstructs the list-of-arrays representation for spikes
    because that is often the easiest structure to work with in analysis code.
    """
    with np.load(path, allow_pickle=False) as z:
        spike_times: NDArray[np.float32] = z["spike_times"]
        indptr: NDArray[np.int64] = z["spike_indptr"]

        spikes = [
            spike_times[indptr[i] : indptr[i + 1]]
            for i in range(len(indptr) - 1)
        ]

        return {
            "spikes": spikes,
            "location": z["location"],
            "class_str": z["class_str"],
            "idx": z["idx"],
            "region_str": z["region_str"],
        }



#========================
#Filtering helpers
#========================
def _region_mask(
    region_labels: Iterable[Any],
    selector: str | Iterable[str] | None,
) -> NDArray[np.bool_]:
    """
    Build boolean mask for region selection.

    - None: all True
    - "D2": matches any label starting with "D2" (prefix match)
    - "D2t_f" (contains "_"): exact match only
    - list/tuple of selectors: OR of each rule above
    """
    labels = np.asarray(list(region_labels), dtype=object)
    n = len(labels)

    if selector is None:
        return np.ones(n, dtype=bool)

    # Normalize to list of strings
    sels: list[str] = [selector] if isinstance(selector, str) else list(selector)

    mask = np.zeros(n, dtype=bool)

    for sel in sels:
        if sel is None:
            continue
        sel_str = str(sel)
        if "_" in sel_str:
            # Exact match
            mask |= labels == sel_str
        else:
            # Prefix match
            mask |= np.fromiter(
                (str(lbl).startswith(sel_str) for lbl in labels),
                dtype=bool,
                count=n,
            )
    return mask


def filter_response(
    resp: dict[str, Any],
    *,
    classes: StrSel = None,
    regions: StrSel = None,
    ids: IntSel = None,
    logic: Literal["and", "or"] = "and",
) -> dict[str, Any]:
    """
    Input: response dict
    Output: response dict with the same schema (chainable).
    - Subsets any field aligned to afferents (length N).
    - Keeps other fields unchanged.
    - If CSR ('spike_times' + 'spike_indptr') exists, rebuilds it for the subset.
    """
    if "spikes" not in resp:
        raise KeyError("Expected resp['spikes'] (list of per-afferent arrays).")

    N = len(resp["spikes"])
    for k in ("class_str", "region_str", "idx"):
        if k not in resp:
            raise KeyError(f"resp lacks '{k}' needed for filtering.")

    m_class  = _normalize_pred(resp["class_str"],  classes)
    #custom region matcher with D2 → all D2* behavior
    m_region = _region_mask(resp["region_str"], regions)
    #this line was missing in your erroring version:
    m_idx    = _normalize_pred(resp["idx"],       ids)

    mask = (m_class | m_region | m_idx) if logic == "or" else (m_class & m_region & m_idx)
    sel = np.flatnonzero(mask)

    out: dict[str, Any] = {}
    had_csr = ("spike_times" in resp) and ("spike_indptr" in resp)

    #subset afferent-aligned fields; copy others
    for k, v in resp.items():
        if k in ("spike_times", "spike_indptr"):
            continue  #rebuild later if present
        if k == "spikes":
            out[k] = [resp["spikes"][i] for i in sel]
        elif isinstance(v, np.ndarray) and v.shape[:1] == (N,):
            out[k] = v[sel]
        elif isinstance(v, list) and len(v) == N:
            out[k] = [v[i] for i in sel]
        else:
            out[k] = v  #passthrough

    #rebuild CSR for the subset if original had it
    if had_csr:
        sp_list = out["spikes"]
        lengths = np.fromiter((len(a) for a in sp_list), dtype=np.int64, count=len(sp_list))
        indptr = np.empty(len(sp_list) + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(lengths, out=indptr[1:])
        nonempty = [np.asarray(a, dtype=np.float32) for a in sp_list if len(a)]
        st = np.concatenate(nonempty) if nonempty else np.empty(0, dtype=np.float32)
        out["spike_times"] = st
        out["spike_indptr"] = indptr

    return out


# Thin, chainable wrappers (dict in -> dict out)
def class_filter(
    resp: dict[str, Any],
    cls: str | Iterable[str] | Callable[[str], bool],
) -> dict[str, Any]:
    """Filter response by afferent class."""
    return filter_response(resp, classes=cls)


def region_filter(
    resp: dict[str, Any],
    region: str | Iterable[str] | Callable[[str], bool],
) -> dict[str, Any]:
    """Filter response by region."""
    return filter_response(resp, regions=region)


def idx_filter(
    resp: dict[str, Any],
    ids: int | Iterable[int] | Callable[[int], bool],
) -> dict[str, Any]:
    """Filter response by afferent index."""
    return filter_response(resp, ids=ids)
