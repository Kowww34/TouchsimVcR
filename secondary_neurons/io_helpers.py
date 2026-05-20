"""Secondary-neuron patch I/O helpers."""

"""
Dictionary-first patching utilities for TouchSim responses.

Primary workflows
-----------------
Collapsed per-patch representation:
    resp = ts.load("pulse1.npz", kind="response")
    patch_resp, patch_shapes = ts.patch(resp)

Expanded afferent-level representation with patch IDs:
    resp = ts.load("pulse1.npz", kind="response")
    resp_with_patch, patch_shapes = ts.patch_expand(resp)
"""

import math
import re

import matplotlib.pyplot as plt
import numpy as np



def extract_spikes_by_global(master_npz_path, global_indices):
    """
    Extract spike times for specified afferents by global index.

    TouchSim stores spikes in CSR (Compressed Sparse Row) format:
    - spike_times: Concatenated spike times for all neurons
    - spike_indptr: Index pointers (length N+1)

    For neuron i, spikes are: spike_times[indptr[i]:indptr[i+1]]

    Parameters
    ----------
    master_npz_path : str
        Path to master NPZ file.
    global_indices : array-like
        Row indices of afferents to extract.

    Returns
    -------
    list of ndarray
        Spike time arrays for each requested afferent.
    """
    d = np.load(master_npz_path, allow_pickle=True)
    st = np.asarray(d["spike_times"], float)  # All spike times concatenated
    ip = np.asarray(d["spike_indptr"], int)    # Index pointers

    out = []
    for i in np.asarray(global_indices, int):
        a, b = ip[i], ip[i + 1]
        out.append(st[a:b].copy())

    return out

def patch_class_global_indices(patch_npz_path, class_name="PC"):
    """
    Get global indices for a specific afferent class within each patch.

    Parameters
    ----------
    patch_npz_path : str
        Path to patch NPZ file.
    class_name : str
        Afferent class ("PC", "RA", or "SA1").

    Returns
    -------
    centers : ndarray, shape (P, 2)
        Patch center coordinates.
    per_patch : dict
        Mapping: patch_id -> array of global indices for that class.
    """
    d = np.load(patch_npz_path, allow_pickle=True)
    centers = np.asarray(d["centers"], float)
    pid = np.asarray(d["patch_id_per_afferent"], int)
    glob = np.asarray(d["global_indices"], int)
    cstr = np.asarray(d["class_str"]).astype(str)
    cstr = np.where(cstr == "SA", "SA1", cstr)

    cname = "SA1" if class_name in ("SA", "SA1") else class_name
    mask = cstr == cname

    out = {}
    P = centers.shape[0]
    for p in range(P):
        m = mask & (pid == p)
        if m.any():
            out[p] = glob[m]

    return centers, out

def _to_float(x):
    """Convert array-like to scalar float, handling edge cases."""
    a = np.asarray(x)
    return float(a.ravel()[0]) if a.size else float("nan")

def patch_class_mean_firing_from_master(patch_npz_path, master_npz_path, class_name, mean_firing):
    """
    Compute mean firing rate per patch for a specific afferent class.

    For each patch:
    1. Find all afferents of the specified class
    2. Extract their spike trains
    3. Compute per-unit firing rate
    4. Average across units in the patch

    Parameters
    ----------
    patch_npz_path : str
        Path to patch NPZ file.
    master_npz_path : str
        Path to master NPZ file with spike data.
    class_name : str
        Afferent class ("PC", "RA", or "SA1").
    mean_firing : callable
        Function to compute firing rate. Signature: (spike_list) -> rates

    Returns
    -------
    DataFrame
        Columns: patch_id, n_units, patch_mean, x_center, y_center
    """
    centers, per_patch = patch_class_global_indices(patch_npz_path, class_name=class_name)

    rows = []
    for p in sorted(per_patch.keys()):
        gidx = per_patch[p]
        sp_list = extract_spikes_by_global(master_npz_path, gidx)

        # Compute per-unit rates, then average
        unit_rates = [_to_float(mean_firing([sp])) for sp in sp_list if len(sp)]
        patch_mean = float("nan") if len(unit_rates) == 0 else float(np.mean(unit_rates))

        rows.append(
            dict(
                patch_id=p,
                n_units=len(gidx),
                patch_mean=patch_mean,
                x_center=float(centers[p, 0]),
                y_center=float(centers[p, 1]),
            )
        )

    return pd.DataFrame(rows).sort_values("patch_id").reset_index(drop=True)

def global_class_mean_firing_for_regions(master_npz_path, class_name, region_whitelist, mean_firing):
    """
    Compute global mean firing rate for a class across specified regions.

    This computes the unpatched mean for comparison with patch-based means.

    Parameters
    ----------
    master_npz_path : str
        Path to master NPZ file.
    class_name : str
        Afferent class.
    region_whitelist : list of str
        Region names to include.
    mean_firing : callable
        Firing rate function.

    Returns
    -------
    float
        Mean firing rate across all matching afferents.
    """
    d = np.load(master_npz_path, allow_pickle=True)
    reg = np.asarray(d["region_str"]).astype(str)
    cls = np.asarray(d["class_str"]).astype(str)
    cls = np.where(cls == "SA", "SA1", cls)

    cname = "SA1" if class_name in ("SA", "SA1") else class_name

    mask = cls == cname
    if region_whitelist:
        rmask = np.zeros(reg.shape[0], bool)
        for r in region_whitelist:
            rmask |= reg == r
        mask &= rmask

    gidx = np.nonzero(mask)[0]
    sp_list = extract_spikes_by_global(master_npz_path, gidx)
    unit_rates = [_to_float(mean_firing([sp])) for sp in sp_list if len(sp)]

    return float(np.mean(unit_rates)) if unit_rates else float("nan")

