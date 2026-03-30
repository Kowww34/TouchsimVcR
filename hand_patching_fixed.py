"""
Hand Patching Tools for TouchSim Afferent Data
===============================================

This module provides utilities for spatial analysis of touch receptor data
from the TouchSim simulator. The main goal is to divide the hand surface
into hexagonal "patches" for region-based analysis.

Overview
--------
TouchSim simulates mechanoreceptor responses across the hand. The hand has
thousands of afferent nerve fibers (receptors) distributed across fingers
and palm. For population-level analysis, we group nearby receptors into
spatial patches.

Key Concepts
------------
1. **Afferent Types**: The hand has three main mechanoreceptor types:
   - PC (Pacinian Corpuscles): Sensitive to vibration (40-400 Hz)
   - RA (Rapidly Adapting): Sensitive to movement/flutter (5-40 Hz)
   - SA1 (Slowly Adapting Type 1): Sensitive to pressure/edges

2. **Hexagonal Patches**: We use hexagonal grids because:
   - Hexagons tessellate the plane efficiently
   - All neighboring patches are equidistant from the center
   - Better approximation of circular receptive fields than squares

3. **Convex Hull**: The boundary of each patch is computed as the convex
   hull of all afferents assigned to that patch.

Data Format
-----------
The TouchSim NPZ files contain:
- location: (N, 3) array of afferent positions [x, y, z] in mm
- class_str: (N,) array of afferent types ("PC", "RA", "SA1")
- region_str: (N,) array of anatomical regions ("D1d_t", "Pw2", etc.)
- spike_times: Concatenated spike times for all afferents
- spike_indptr: Index pointers (CSR format) to extract per-afferent spikes

Region naming convention:
- D1-D5: Digits (thumb to pinky)
- Pw1-Pw4: Palm wrist regions
- Pp1-Pp2: Palm proximal regions
- Suffixes: d=distal, m=medial, p=proximal, t=tip, f=finger pad

Author: [Your name]
Date: 2024
"""

import json
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# GEOMETRIC UTILITIES
# =============================================================================

def _show_df_simple(name, df, path=None):
    """Helper to display and optionally save a DataFrame."""
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"{name} -> saved to {path} (rows={len(df)})")
    try:
        from IPython.display import display
        print(name)
        display(df)
    except Exception:
        print(name)
        print(df.head(20).to_string(index=False))


def _convex_hull_safe(points):
    """
    Compute the convex hull of a set of 2D points.

    Uses Andrew's monotone chain algorithm, which is efficient O(n log n)
    and numerically stable.

    The convex hull is the smallest convex polygon that contains all points.
    Think of it as stretching a rubber band around all points.

    Parameters
    ----------
    points : array-like, shape (N, 2)
        2D coordinates of points.

    Returns
    -------
    hull : ndarray, shape (M, 2)
        Vertices of the convex hull in counter-clockwise order.
        Empty array if fewer than 3 points.
    area : float
        Area of the hull computed via the shoelace formula.

    Notes
    -----
    The shoelace formula computes area as:
        A = 0.5 * |sum_i (x_i * y_{i+1} - x_{i+1} * y_i)|

    This works because it sums the signed areas of triangles formed
    with the origin.
    """
    P = np.asarray(points, float)

    # Need at least 3 points to form a hull
    if P.ndim != 2 or P.shape[0] < 3:
        return np.empty((0, 2), float), 0.0

    # Sort points lexicographically (by x, then by y)
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def cross(o, a, b):
        """Cross product of vectors OA and OB (2D)."""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull (left to right)
    lower = []
    for p in P:
        # Remove points that make a right turn (not on convex hull)
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    # Build upper hull (right to left)
    upper = []
    for p in P[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    # Concatenate (remove duplicate endpoints)
    hull = np.array(lower[:-1] + upper[:-1], float)

    if hull.size == 0:
        return np.empty((0, 2), float), 0.0

    # Compute area via shoelace formula
    a = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        a += x1 * y2 - x2 * y1

    return hull, abs(a) * 0.5


def _point_in_convex_poly(pt, poly):
    """
    Test if a point lies inside a convex polygon.

    Uses the cross-product method: a point is inside if it's on the
    same side of all edges (all cross products have the same sign).

    Parameters
    ----------
    pt : tuple (x, y)
        Point to test.
    poly : ndarray, shape (M, 2)
        Vertices of convex polygon in order.

    Returns
    -------
    bool
        True if point is inside or on the boundary.
    """
    if poly.size == 0:
        return True

    x, y = pt
    sign = None

    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]

        # Cross product: (edge vector) x (point - vertex)
        cr = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

        if cr == 0:
            continue  # Point on edge

        s = cr > 0
        if sign is None:
            sign = s
        elif s != sign:
            return False  # Different sides = outside

    return True


def _build_hex_centers_inside_hull(hull, xmin, xmax, ymin, ymax, R):
    """
    Generate hexagonal grid centers that lie inside a convex hull.

    Hexagonal grids are created by offsetting every other row. The spacing
    is chosen so that hexagons tile the plane perfectly.

    For a hexagon with "radius" R (center to vertex distance):
    - Horizontal spacing: dx = 1.5 * R
    - Vertical spacing: dy = sqrt(3) * R
    - Odd rows offset by: dx/2 = 0.75 * R

    Parameters
    ----------
    hull : ndarray, shape (M, 2)
        Convex hull vertices.
    xmin, xmax, ymin, ymax : float
        Bounding box for the grid.
    R : float
        Hexagon "radius" (center to vertex).

    Returns
    -------
    centers : ndarray, shape (K, 2)
        Hexagon center coordinates inside the hull.
    dx, dy : float
        Horizontal and vertical spacing.
    """
    dx = 1.5 * R
    dy = math.sqrt(3.0) * R
    centers = []
    j = 0
    yy = ymin

    while yy <= ymax + 1e-9:
        # Offset odd rows for hexagonal packing
        row_offset = 0.75 * R if (j % 2) == 1 else 0.0
        xx = xmin + row_offset

        while xx <= xmax + 1e-9:
            c = np.array([xx, yy], float)
            # Only keep centers inside the hull
            if _point_in_convex_poly(c, hull):
                centers.append(c)
            xx += dx

        yy += dy
        j += 1

    # Fallback: at least one center at the centroid
    if not centers:
        centers = [np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0], float)]

    return np.array(centers, float), dx, dy


# =============================================================================
# MAIN PATCH BUILDER
# =============================================================================

def build_hex_patches_for_finger_safe_v2(
    npz_path="PC_optimized.npz",
    finger_prefix="D2",
    *,
    min_pc=10,
    alpha_margin_mm=2.0,
    region_whitelist=None,
    save_dir="data",
    summary_csv=True,
    dedup_by_idx=False,
    force_rebuild=False,
):
    """
    Build hexagonal patches for a finger or region group.

    This is the main function for spatial analysis. It:
    1. Loads afferent data from NPZ
    2. Filters to the specified region(s)
    3. Creates a hexagonal grid sized to have ~min_pc PCs per patch
    4. Assigns each afferent to its nearest patch center
    5. Merges sparse patches (< min_pc PCs) into neighbors
    6. Saves results to NPZ and CSV

    Algorithm
    ---------
    1. Compute PC density: rho_PC = N_PC / area
    2. Target patch area: A_target = min_pc / rho_PC
    3. Hexagon radius: R = sqrt(2 * A_target / (3 * sqrt(3)))
       (This comes from hexagon area = (3*sqrt(3)/2) * R^2)
    4. Generate hex grid, assign afferents to nearest center
    5. Iteratively merge patches with < min_pc PCs

    Parameters
    ----------
    npz_path : str
        Path to TouchSim NPZ file.
    finger_prefix : str
        Region prefix (e.g., "D2" for index finger, "Pw1" for palm).
    min_pc : int
        Minimum PC afferents per patch. Controls patch size.
    alpha_margin_mm : float
        Margin (mm) added around the hull for hex grid generation.
    region_whitelist : list of str, optional
        Explicit list of region names to include. If None, uses all
        regions starting with finger_prefix.
    save_dir : str
        Output directory for NPZ and CSV files.
    summary_csv : bool
        Whether to save a summary CSV with patch statistics.
    dedup_by_idx : bool
        Deprecated parameter, kept for API compatibility.
    force_rebuild : bool
        If False, reuse existing patch file if present.

    Returns
    -------
    dict
        {"npz": output_path, "summary_df": DataFrame}

    Output NPZ Fields
    -----------------
    - centers: (P, 2) patch center coordinates
    - patch_id_per_afferent: (N,) patch assignment for each afferent
    - global_indices: (N,) row indices in the master NPZ
    - x, y: (N,) afferent coordinates
    - class_str, region_str: (N,) afferent metadata
    - params: dict of construction parameters

    Example
    -------
    >>> res = build_hex_patches_for_finger_safe_v2(
    ...     "PC_optimized.npz",
    ...     finger_prefix="D2",
    ...     region_whitelist=["D2d_t", "D2m_f", "D2p_f"],
    ...     min_pc=10
    ... )
    >>> print(f"Created {res['npz']} with {len(res['summary_df'])} patches")
    """
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"patches_{finger_prefix}_minPC{min_pc}_v2.npz")
    csv_path = os.path.join(save_dir, f"patch_summary_{finger_prefix}_minPC{min_pc}_v2.csv")

    # ----- Reuse existing if available -----
    if not force_rebuild and os.path.exists(out_path):
        print(f"[{finger_prefix}] Using existing {out_path}")
        df = pd.read_csv(csv_path) if summary_csv and os.path.exists(csv_path) else None
        return {"npz": out_path, "summary_df": df}

    # ----- Load data -----
    d = np.load(npz_path, allow_pickle=True)
    required = {"location", "class_str", "region_str"}
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"NPZ missing keys: {missing}. Present: {list(d.keys())}")

    # Parse location array (may be object array or 2D float)
    loc = np.asarray(d["location"])
    if loc.ndim == 1 and loc.dtype == object:
        loc = np.vstack(loc).astype(float)
    loc = np.asarray(loc, float)

    cls = np.asarray(d["class_str"]).astype(str)
    reg = np.asarray(d["region_str"]).astype(str)

    # Normalize "SA" to "SA1" for consistency
    cls_norm = np.where(cls == "SA", "SA1", cls).astype(str)

    # ----- Build region mask -----
    if region_whitelist is not None:
        mask = np.zeros(len(reg), bool)
        for rsel in region_whitelist:
            mask |= reg == rsel
    else:
        mask = np.array([r.startswith(finger_prefix) for r in reg])

    if not mask.any():
        raise ValueError(
            f"No afferents for prefix '{finger_prefix}' (or whitelist={region_whitelist})."
        )

    # ----- Extract selected afferents -----
    pos = np.flatnonzero(mask)  # GLOBAL row indices in master NPZ
    L = loc[pos, :2].astype(float)  # (N, 2) x,y coordinates
    C = cls_norm[pos]  # Class labels
    R = reg[pos]  # Region labels
    G = pos.copy()  # Global indices
    is_pc = C == "PC"  # Boolean mask for PCs

    # Compute convex hull of selected region
    hull, area_hull = _convex_hull_safe(L)
    print(
        f"[{finger_prefix}] N={L.shape[0]} | PCs={int(is_pc.sum())} | "
        f"area~{area_hull:.2f} mm^2 | min_pc={min_pc}"
    )

    # ----- Handle edge cases -----
    if (L.shape[0] < 3) or (area_hull < 1e-6) or (is_pc.sum() == 0):
        # Too few points: create single patch at centroid
        centers = np.array([L.mean(axis=0)], float)
        assign = np.zeros(L.shape[0], int)
    else:
        # ----- Compute optimal hexagon size -----
        # PC density (PCs per mm^2)
        rho_pc = is_pc.sum() / max(area_hull, 1e-12)

        # Target patch area to achieve min_pc PCs
        A_target = max(min_pc, 1) / max(rho_pc, 1e-12)

        # Hexagon radius from area formula: A = (3*sqrt(3)/2) * R^2
        # Solving: R = sqrt(2*A / (3*sqrt(3)))
        Rhex = math.sqrt((2.0 * A_target) / (3.0 * math.sqrt(3.0)))

        # Bounding box with margin
        xmin = float(L[:, 0].min() - alpha_margin_mm)
        xmax = float(L[:, 0].max() + alpha_margin_mm)
        ymin = float(L[:, 1].min() - alpha_margin_mm)
        ymax = float(L[:, 1].max() + alpha_margin_mm)

        # Generate hex centers
        centers, dx, dy = _build_hex_centers_inside_hull(hull, xmin, xmax, ymin, ymax, Rhex)

        # ----- Assign afferents to nearest center -----
        # Compute squared distances: D2[i, j] = ||L[i] - centers[j]||^2
        D2 = ((L[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(D2, axis=1)  # (N,) patch assignment

        # ----- Merge sparse patches -----
        # Iteratively merge patches with < min_pc PCs into neighbors
        P = centers.shape[0]
        removed = np.zeros(P, bool)

        while True:
            # Count PCs and total afferents per patch
            pc_counts = np.bincount(assign, weights=is_pc.astype(int), minlength=P).astype(int)
            all_counts = np.bincount(assign, minlength=P).astype(int)

            # Find patches below threshold
            cand = np.where((pc_counts < min_pc) & (~removed))[0]
            if cand.size == 0:
                break  # All patches meet threshold

            # Pick the sparsest patch to remove
            w = cand[np.argsort(pc_counts[cand] + 1e-6 * all_counts[cand])][0]
            removed[w] = True

            # Reassign its afferents to nearest remaining patch
            keep_mask = ~removed
            idx_w = np.where(assign == w)[0]

            if idx_w.size:
                keep_idx = np.nonzero(keep_mask)[0]
                if keep_idx.size == 0:
                    removed[w] = False  # Can't remove last patch
                    break
                elif keep_idx.size == 1:
                    assign[idx_w] = keep_idx[0]
                else:
                    Ck = centers[keep_mask]
                    D2w = ((L[idx_w][:, None, :] - Ck[None, :, :]) ** 2).sum(axis=2)
                    jmin = np.argmin(D2w, axis=1)
                    assign[idx_w] = keep_idx[jmin]

            if keep_mask.sum() <= 1:
                break  # Keep at least one patch

        # Remove deleted patches and remap assignments
        keep = ~removed if removed.any() else np.ones(P, bool)
        centers = centers[keep]
        remap = {old: new for new, old in enumerate(np.nonzero(keep)[0])}
        assign = np.array([remap[a] for a in assign], int)

    # ----- Save results -----
    np.savez_compressed(
        out_path,
        centers=centers,
        patch_id_per_afferent=assign,
        global_indices=G,
        x=L[:, 0],
        y=L[:, 1],
        class_str=C,
        region_str=R,
        params=dict(
            min_pc=int(min_pc),
            finger_prefix=str(finger_prefix),
            region_whitelist=list(region_whitelist) if region_whitelist is not None else None,
            alpha_margin_mm=float(alpha_margin_mm),
        ),
    )

    # ----- Generate summary DataFrame -----
    rows = []
    for pid in range(centers.shape[0]):
        m = assign == pid
        rows.append(
            dict(
                finger=finger_prefix,
                patch_id=pid,
                x_center=float(centers[pid, 0]),
                y_center=float(centers[pid, 1]),
                n_total=int(m.sum()),
                PC=int((C[m] == "PC").sum()),
                RA=int((C[m] == "RA").sum()),
                SA1=int((C[m] == "SA1").sum()),
            )
        )
    df = pd.DataFrame(rows).sort_values("patch_id").reset_index(drop=True)

    if summary_csv:
        df.to_csv(csv_path, index=False)
        print("Saved summary ->", csv_path)

    print(f"Saved -> {out_path} | patches={centers.shape[0]}")
    return {"npz": out_path, "summary_df": df}


# =============================================================================
# WHOLE-HAND PATCH BUILDER
# =============================================================================

def build_hand_and_palm_patches(npz_path, *, min_pc=10, save_dir="data", summary_csv=False):
    """
    Build patches for all anatomical regions (digits + palm).

    This is a convenience function that automatically identifies all
    region groups in the data and creates patches for each.

    Region groups are identified by the leading alphanumeric prefix:
    - D1, D2, D3, D4, D5 (digits)
    - Pw1, Pw2, Pw3, Pw4 (palm wrist)
    - Pp1, Pp2 (palm proximal)

    Parameters
    ----------
    npz_path : str
        Path to master NPZ file.
    min_pc : int
        Minimum PC afferents per patch.
    save_dir : str
        Output directory.
    summary_csv : bool
        Whether to save per-region summary CSVs.

    Returns
    -------
    dict
        {"patch_files": list of NPZ paths, "groups": list of region group names}

    Example
    -------
    >>> res = build_hand_and_palm_patches("PC_optimized.npz", min_pc=10)
    >>> print(f"Created patches for {len(res['groups'])} regions")
    """
    d = np.load(npz_path, allow_pickle=True)
    all_regions = np.asarray(d["region_str"]).astype(str)

    # Extract unique region prefixes (e.g., "D1", "Pw2")
    keys = []
    for r in all_regions:
        m = re.match(r"^([A-Za-z]+[0-9]?)", r)
        if m:
            keys.append(m.group(1))
    groups = sorted(set(keys))

    # Build patches for each group
    patch_files = []
    for g in groups:
        whitelist = sorted({r for r in all_regions if r.startswith(g)})
        try:
            res = build_hex_patches_for_finger_safe_v2(
                npz_path=npz_path,
                finger_prefix=g,
                region_whitelist=whitelist,
                min_pc=min_pc,
                save_dir=save_dir,
                summary_csv=summary_csv,
            )
            patch_files.append(res["npz"])
        except Exception as e:
            print(f"[WARN] Skipping {g}: {e}")

    return {"patch_files": patch_files, "groups": groups}


# =============================================================================
# PHASE CACHE (for efficient repeated analysis)
# =============================================================================

def compute_or_load_pc_phase_cache(
    master_npz,
    *,
    t0=0.5,
    dt=1 / 5000.0,
    cache_dir="data/pc_phase_cache",
    batch=256,
    spike_phase_func=None,
):
    """
    Compute or load cached PC phase data for efficient analysis.

    Computing spike phase is expensive (O(N_neurons * N_timepoints)).
    This function caches the results in a memory-mapped file for reuse.

    The cache includes:
    - Phase values for all PCs at each time point
    - Firing rates
    - Mapping from global indices to cache rows

    Parameters
    ----------
    master_npz : str
        Path to master NPZ file.
    t0 : float
        Start time for phase computation (seconds).
    dt : float
        Time step (seconds). Default 1/5000 = 0.0002s = 5 kHz sampling.
    cache_dir : str
        Directory to store cache files.
    batch : int
        Batch size for phase computation (memory optimization).
    spike_phase_func : callable, optional
        Function to compute spike phase. Signature: (spike_list, t) -> (phi, fr)
        If None, imports from spk_phase module.

    Returns
    -------
    phi_mmap : memmap, shape (n_valid, n_times)
        Memory-mapped phase array (float32).
    t : ndarray
        Time vector.
    rowmap : dict
        Mapping: global_index -> row in phi_mmap.
    fr_all : ndarray
        Firing rates for each valid neuron.
    pc_gidx : ndarray
        Global indices of all PC afferents.

    Notes
    -----
    The cache is validated by checking master_npz path, t0, and dt.
    If parameters change, the cache is rebuilt.

    Memory-mapping allows working with large phase arrays without
    loading everything into RAM.
    """
    if spike_phase_func is None:
        try:
            from spk_phase import spike_phase
            spike_phase_func = spike_phase
        except ImportError:
            raise ImportError("spike_phase function not found; provide spike_phase_func argument")

    os.makedirs(cache_dir, exist_ok=True)

    # Cache file paths
    meta_path = os.path.join(cache_dir, "pc_phase_meta.json")
    phi_path = os.path.join(cache_dir, "pc_phase_phi_float32.dat")
    rowmap_path = os.path.join(cache_dir, "pc_phase_rowmap.npy")
    t_path = os.path.join(cache_dir, "pc_phase_t.npy")
    fr_path = os.path.join(cache_dir, "pc_phase_fr.npy")
    gid_path = os.path.join(cache_dir, "pc_global_indices.npy")

    # ----- Check for valid existing cache -----
    if all(os.path.exists(p) for p in [meta_path, phi_path, rowmap_path, t_path, fr_path, gid_path]):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if (
            meta.get("master_npz") == master_npz
            and abs(meta.get("t0", -1) - t0) < 1e-12
            and abs(meta.get("dt", -1) - dt) < 1e-12
        ):
            # Cache is valid - load and return
            t = np.load(t_path)
            n_valid, nt = int(meta["n_valid"]), int(meta["nt"])
            phi_mmap = np.memmap(phi_path, dtype=np.float32, mode="r", shape=(n_valid, nt))
            rowmap = np.load(rowmap_path, allow_pickle=True).item()
            fr_all = np.load(fr_path)
            pc_gidx = np.load(gid_path)
            return phi_mmap, t, rowmap, fr_all, pc_gidx

    # ----- Build fresh cache -----
    print("Building phase cache (this may take a few minutes)...")

    d = np.load(master_npz, allow_pickle=True)
    cls = np.asarray(d["class_str"]).astype(str)
    is_pc = cls == "PC"
    pc_gidx = np.nonzero(is_pc)[0].astype(int)

    # Extract spikes for ALL PCs
    PC_spikes_all = extract_spikes_by_global(master_npz, pc_gidx)

    # Determine global time range
    t_max = 0.0
    valid_mask = np.array([len(sp) >= 2 for sp in PC_spikes_all], dtype=bool)
    for sp in PC_spikes_all:
        if len(sp):
            t_max = max(t_max, float(sp[-1]))

    if t_max <= t0 + dt:
        raise RuntimeError("Insufficient spike support to build a time axis.")

    t = np.arange(t0, t_max, dt, dtype=float)
    nt = t.size

    # Only include neurons with >= 2 spikes (required for phase computation)
    valid_gidx = pc_gidx[valid_mask]
    n_valid = valid_gidx.size

    # Allocate memory-mapped file
    phi_mmap = np.memmap(phi_path, dtype=np.float32, mode="w+", shape=(n_valid, nt))
    fr_all = np.zeros(n_valid, dtype=np.float32)
    rowmap = {}

    # Process in batches to limit memory usage
    rows_written = 0
    gidx_to_dense = {int(g): i for i, g in enumerate(pc_gidx)}

    for start in range(0, n_valid, batch):
        stop = min(start + batch, n_valid)
        batch_g = valid_gidx[start:stop]
        batch_spikes = [PC_spikes_all[gidx_to_dense[int(g)]] for g in batch_g]

        # Compute phase
        phi_b, fr_b = spike_phase_func(batch_spikes, t)
        nb = phi_b.shape[0]

        # Store results
        phi_mmap[rows_written:rows_written + nb, :] = phi_b.astype(np.float32)
        fr_all[rows_written:rows_written + nb] = fr_b.astype(np.float32)

        # Build index mapping
        for j, g in enumerate(batch_g):
            rowmap[int(g)] = int(rows_written + j)

        rows_written += nb

    # ----- Save metadata -----
    meta = dict(master_npz=master_npz, t0=float(t0), dt=float(dt), n_valid=int(n_valid), nt=int(nt))
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    np.save(t_path, t)
    np.save(fr_path, fr_all)
    np.save(gid_path, pc_gidx)
    np.save(rowmap_path, rowmap, allow_pickle=True)
    phi_mmap.flush()

    # Reopen as read-only
    phi_mmap = np.memmap(phi_path, dtype=np.float32, mode="r", shape=(n_valid, nt))
    print(f"Phase cache built: {n_valid} neurons, {nt} time points")

    return phi_mmap, t, rowmap, fr_all, pc_gidx


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_patches(
    patch_npz_path,
    *,
    draw_hulls=True,
    annotate_centers=False,
    edge_color="k",
    edge_lw=1.6,
    s_aff=8,
    alpha_pts=0.7,
):
    """
    Visualize patches with afferents colored by type.

    Creates a scatter plot of all afferents, colored by class (PC, RA, SA1),
    with patch boundaries shown as convex hulls.

    Parameters
    ----------
    patch_npz_path : str
        Path to patch NPZ file.
    draw_hulls : bool
        Whether to draw patch boundary hulls.
    annotate_centers : bool
        Whether to label patch centers with IDs.
    edge_color : str
        Color for hull edges.
    edge_lw : float
        Line width for hull edges.
    s_aff : float
        Marker size for afferents.
    alpha_pts : float
        Transparency for afferent markers.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    d = np.load(patch_npz_path, allow_pickle=True)
    centers = d["centers"]
    pid = d["patch_id_per_afferent"].astype(int)
    x = np.asarray(d["x"], float)
    y = np.asarray(d["y"], float)
    cls = np.asarray(d["class_str"]).astype(str)
    cls = np.where(cls == "SA", "SA1", cls)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Scatter afferents by class
    for name, color, z in [("PC", "C0", 3), ("RA", "C1", 2), ("SA1", "C2", 1)]:
        m = cls == name
        if m.any():
            ax.scatter(x[m], y[m], s=s_aff, alpha=alpha_pts, label=name, zorder=z)

    # Draw patch hulls
    if draw_hulls:
        P = centers.shape[0]
        for p in range(P):
            m = pid == p
            if not m.any():
                continue
            hull, area = _convex_hull_safe(np.c_[x[m], y[m]])
            if hull.size:
                ax.plot(
                    np.r_[hull[:, 0], hull[0, 0]],
                    np.r_[hull[:, 1], hull[0, 1]],
                    color=edge_color,
                    lw=edge_lw,
                    alpha=0.9,
                    zorder=5,
                )

    # Mark centers
    ax.scatter(centers[:, 0], centers[:, 1], c="k", s=12, zorder=6)

    if annotate_centers:
        for i, (cx, cy) in enumerate(centers):
            ax.text(cx, cy, str(i), ha="center", va="center", fontsize=9, color="k", zorder=7)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.legend()
    ax.set_title(os.path.basename(patch_npz_path))
    plt.tight_layout()

    return fig, ax


def plot_hand_patches(patch_files, *, annotate_centers=False, edge_color="k", edge_lw=1.6, s_aff=6):
    """
    Visualize all patches across the entire hand.

    Parameters
    ----------
    patch_files : list of str
        Paths to patch NPZ files.
    annotate_centers : bool
        Whether to label patch centers.
    edge_color : str
        Color for hull edges.
    edge_lw : float
        Line width for hull edges.
    s_aff : float
        Marker size for afferents.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    for path in patch_files:
        d = np.load(path, allow_pickle=True)
        centers = d["centers"]
        pid = d["patch_id_per_afferent"].astype(int)
        x = np.asarray(d["x"], float)
        y = np.asarray(d["y"], float)

        # Draw region hull
        hull, area = _convex_hull_safe(np.c_[x, y])
        if hull.size:
            ax.plot(
                np.r_[hull[:, 0], hull[0, 0]],
                np.r_[hull[:, 1], hull[0, 1]],
                color=edge_color,
                lw=edge_lw,
                alpha=0.9,
                zorder=3,
            )

        ax.scatter(centers[:, 0], centers[:, 1], c="k", s=10, zorder=4)

        if annotate_centers:
            for i, (cx, cy) in enumerate(centers):
                ax.text(cx, cy, str(i), ha="center", va="center", fontsize=8, color="k", zorder=5)

        ax.scatter(x, y, s=s_aff, alpha=0.25, color="gray", zorder=1)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Hand patches")
    plt.tight_layout()

    return fig, ax


# =============================================================================
# SPIKE DATA EXTRACTION
# =============================================================================

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


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_hand_patches(npz_path, *, min_pc=10, save_dir="data"):
    """
    Build patches for digits D1-D5 only (excluding palm).

    Parameters
    ----------
    npz_path : str
        Path to master NPZ.
    min_pc : int
        Minimum PCs per patch.
    save_dir : str
        Output directory.

    Returns
    -------
    dict
        {"patch_files": list of NPZ paths}
    """
    d = np.load(npz_path, allow_pickle=True)
    all_regions = np.asarray(d["region_str"]).astype(str)

    # Find digit prefixes (D1-D5)
    fingers = sorted({r[:2] for r in all_regions if r and (r[0] == "D") and r[1].isdigit()})

    patch_files = []
    for f in fingers:
        mask_regions = sorted({r for r in all_regions if r.startswith(f)})
        try:
            res = build_hex_patches_for_finger_safe_v2(
                npz_path=npz_path,
                finger_prefix=f,
                region_whitelist=mask_regions,
                min_pc=min_pc,
                save_dir=save_dir,
                summary_csv=True,
            )
            patch_files.append(res["npz"])
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")

    return {"patch_files": patch_files}
