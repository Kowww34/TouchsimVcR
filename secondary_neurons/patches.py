"""Secondary-neuron hand patches and response pooling."""

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

def _normalize_class_labels(c):
    arr = np.asarray(c).astype(str)
    return np.where(arr == "SA", "SA1", arr)

def _class_count_vector(c):
    c = _normalize_class_labels(c)
    return np.array(
        [
            int(np.sum(c == "PC")),
            int(np.sum(c == "RA")),
            int(np.sum(c == "SA1")),
        ],
        dtype=int,
    )

def expand_patch_response(resp, patch_data):
    """
    Keep afferent-level response and append per-afferent patch labels.

    Adds:
        - patch_ID: np.ndarray (K,), globally unique IDs 1..N_patches
                    (0 means unassigned / failed group)
        - patch_group: np.ndarray (K,), group label per afferent
        - patch_local_id: np.ndarray (K,), local patch id within group

    Returns:
        resp_with_patch, patch_shapes
    """
    required = {"spikes", "location", "class_str", "region_str"}
    missing = required.difference(resp.keys())
    if missing:
        raise KeyError(f"response dictionary missing keys: {sorted(missing)}")

    k = len(resp["spikes"])
    patch_id = np.zeros(k, dtype=int)
    patch_group = np.full(k, "", dtype=object)
    patch_local_id = np.full(k, -1, dtype=int)

    centers_all = []
    hulls_all = []
    group_all = []
    local_id_all = []
    global_id_all = []

    next_global_patch_id = 1
    for g in patch_data.get("groups", []):
        pdata = patch_data.get("patches", {}).get(g)
        if pdata is None:
            continue
        centers = np.asarray(pdata["centers"], dtype=float)
        assign = np.asarray(pdata["patch_id_per_afferent"], dtype=int)
        global_indices = np.asarray(pdata["global_indices"], dtype=int)

        group_global_ids = {}
        for pid in range(centers.shape[0]):
            group_global_ids[int(pid)] = next_global_patch_id
            next_global_patch_id += 1

            m = assign == pid
            if not np.any(m):
                continue
            gidx = global_indices[m]
            pts = np.asarray(resp["location"], dtype=float)[gidx, :2]
            hull, _ = _convex_hull_safe(pts)

            patch_id[gidx] = group_global_ids[int(pid)]
            patch_group[gidx] = str(g)
            patch_local_id[gidx] = int(pid)

            centers_all.append(np.asarray(centers[pid, :2], dtype=float))
            hulls_all.append(hull if hull.size else np.empty((0, 2), dtype=float))
            group_all.append(str(g))
            local_id_all.append(int(pid))
            global_id_all.append(group_global_ids[int(pid)])

    # copy original response dict to preserve source fields
    resp_with_patch = dict(resp)
    resp_with_patch["patch_ID"] = patch_id
    resp_with_patch["patch_group"] = np.asarray(patch_group).astype(str)
    resp_with_patch["patch_local_id"] = patch_local_id

    if centers_all:
        centers_arr = np.vstack(centers_all).astype(float)
    else:
        centers_arr = np.empty((0, 2), dtype=float)

    patch_shapes = {
        "location": centers_arr,
        "hulls": hulls_all,
        "group": np.asarray(group_all).astype(str),
        "patch_local_id": np.asarray(local_id_all, dtype=int),
        "patch_ID": np.asarray(global_id_all, dtype=int),
    }
    return resp_with_patch, patch_shapes

def patch_finger(
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

def patch_hand(
    d,
    *,
    min_pc=10,
    alpha_margin_mm=2.0,
    include_groups=None,
    exclude_groups=None,
):
    """
    Build hexagonal afferent subpopulations directly from an in-memory
    TouchSim-style dictionary.

    This function replaces the file-based workflow. It does not load from disk
    and does not save to disk. Instead, it takes a dictionary already in memory,
    identifies anatomical region groups, builds hexagonal patches for each
    group, merges sparse patches when needed, and returns all results as Python
    objects that can be used directly in an analysis pipeline.

    Required dictionary keys
    ------------------------
    d must contain at least:
        - "location"   : array-like, shape (N,2) or (N,3)
        - "class_str"  : array-like, shape (N,)
        - "region_str" : array-like, shape (N,)

    Optional dictionary keys
    ------------------------
        - "idx"        : array-like, shape (N,)

    If "idx" is not present, the global row indices 0..N-1 are used instead.

    Grouping behavior
    -----------------
    Anatomical groups are inferred from the leading alphanumeric prefix of each
    region string. Examples:

        D1d_t  -> D1
        D2m_f  -> D2
        Pw1    -> Pw1
        Pp2    -> Pp2

    Patches are built independently within each group.

    Patch construction logic
    ------------------------
    For each anatomical group, this function:

    1. Selects afferents belonging to that group
    2. Computes the convex hull of their xy locations
    3. Estimates PC density within the hull
    4. Chooses a hexagon size so that each patch should contain roughly
       `min_pc` PC afferents
    5. Generates hexagonal centers inside the hull
    6. Assigns each afferent to its nearest hex center
    7. Merges sparse patches until each remaining patch has at least `min_pc`
       PCs, when possible

    Important note
    --------------
    Patch size is determined from PC density, not total afferent density.
    RA and SA1 afferents are assigned after the PC-based hex grid is built.

    Parameters
    ----------
    d : dict
        In-memory TouchSim-style dictionary containing afferent metadata.

    min_pc : int, default=10
        Minimum PC count per patch after merging.

    alpha_margin_mm : float, default=2.0
        Margin added to the bounding box before generating the hex grid.

    include_groups : iterable of str or None, default=None
        If provided, only these anatomical groups are patched.

    exclude_groups : iterable of str or None, default=None
        If provided, these anatomical groups are skipped.

    Returns
    -------
    out : dict
        Dictionary with the following structure:

        {
            "groups": list[str],
            "patches": {
                group_name: {
                    "group": str,
                    "centers": ndarray,               # shape (P,2)
                    "patch_id_per_afferent": ndarray, # shape (M,)
                    "global_indices": ndarray,        # shape (M,)
                    "x": ndarray,                     # shape (M,)
                    "y": ndarray,                     # shape (M,)
                    "class_str": ndarray,             # shape (M,)
                    "region_str": ndarray,            # shape (M,)
                    "idx": ndarray,                   # shape (M,)
                    "params": dict,
                    "summary_df": pandas.DataFrame,
                },
                ...
            },
            "summary_df": pandas.DataFrame,
            "failed_groups": {
                group_name: "error message",
                ...
            }
        }

    Interpretation of indices
    -------------------------
    For each group:
        - patch_id_per_afferent is local to that group
        - global_indices points back to rows of the original input dictionary

    This means patch IDs are not globally unique across the whole hand.

    Example
    -------
    >>> res = patch_hand(resp_dict, min_pc=10)
    >>> print(res["groups"])
    >>> print(res["summary_df"])
    >>> d2 = res["patches"]["D2"]
    >>> print(d2["centers"])
    >>> print(d2["patch_id_per_afferent"])
    """

    required = {"location", "class_str", "region_str"}
    missing = required.difference(d.keys())
    if missing:
        raise KeyError(f"dictionary missing required keys: {sorted(missing)}")

    loc = np.asarray(d["location"])
    if loc.ndim == 1 and loc.dtype == object:
        loc = np.vstack(loc).astype(float)
    loc = np.asarray(loc, float)

    if loc.ndim != 2 or loc.shape[1] < 2:
        raise ValueError("'location' must be an (N,2) or (N,3) array-like object")

    cls = np.asarray(d["class_str"]).astype(str)
    reg = np.asarray(d["region_str"]).astype(str)
    cls = np.where(cls == "SA", "SA1", cls)

    if "idx" in d:
        idx_all = np.asarray(d["idx"])
    else:
        idx_all = np.arange(len(cls), dtype=int)

    #local helper: convex hull with monotone chain
    def _convex_hull_safe(points):
        P = np.asarray(points, float)
        if P.ndim != 2 or P.shape[0] < 3:
            return np.empty((0, 2), float), 0.0

        P = P[np.lexsort((P[:, 1], P[:, 0]))]

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in P:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(tuple(p))

        upper = []
        for p in P[::-1]:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(tuple(p))

        hull = np.array(lower[:-1] + upper[:-1], float)
        if hull.size == 0:
            return np.empty((0, 2), float), 0.0

        a = 0.0
        for i in range(len(hull)):
            x1, y1 = hull[i]
            x2, y2 = hull[(i + 1) % len(hull)]
            a += x1 * y2 - x2 * y1

        return hull, abs(a) * 0.5

    #local helper: point in convex polygon
    def _point_in_convex_poly(pt, poly):
        if poly.size == 0:
            return True

        x, y = pt
        sign = None
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            cr = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

            if cr == 0:
                continue

            s = cr > 0
            if sign is None:
                sign = s
            elif s != sign:
                return False

        return True

    #local helper: hex centers inside hull
    def _build_hex_centers_inside_hull(hull, xmin, xmax, ymin, ymax, R):
        dx = 1.5 * R
        dy = math.sqrt(3.0) * R
        centers = []
        j = 0
        yy = ymin

        while yy <= ymax + 1e-9:
            row_offset = 0.75 * R if (j % 2) == 1 else 0.0
            xx = xmin + row_offset

            while xx <= xmax + 1e-9:
                c = np.array([xx, yy], float)
                if _point_in_convex_poly(c, hull):
                    centers.append(c)
                xx += dx

            yy += dy
            j += 1

        if not centers:
            centers = [np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0], float)]

        return np.array(centers, float)

    #infer anatomical groups
    keys = []
    for r in reg:
        m = re.match(r"^([A-Za-z]+[0-9]?)", r)
        if m:
            keys.append(m.group(1))
    groups = sorted(set(keys))

    if include_groups is not None:
        include_groups = set(include_groups)
        groups = [g for g in groups if g in include_groups]

    if exclude_groups is not None:
        exclude_groups = set(exclude_groups)
        groups = [g for g in groups if g not in exclude_groups]

    patches = {}
    summaries = []
    failed = {}

    for g in groups:
        whitelist = sorted({r for r in reg if r.startswith(g)})

        try:
            mask = np.zeros(len(reg), dtype=bool)
            for rsel in whitelist:
                mask |= (reg == rsel)

            if not mask.any():
                raise ValueError(f"no afferents found for group {g!r}")

            pos = np.flatnonzero(mask)
            L = loc[pos, :2].astype(float)
            C = cls[pos]
            R = reg[pos]
            IDX = idx_all[pos]
            G = pos.copy()
            is_pc = (C == "PC")

            hull, area_hull = _convex_hull_safe(L)

            if (L.shape[0] < 3) or (area_hull < 1e-12) or (is_pc.sum() == 0):
                centers = np.array([L.mean(axis=0)], dtype=float)
                assign = np.zeros(L.shape[0], dtype=int)
            else:
                rho_pc = is_pc.sum() / max(area_hull, 1e-12)
                A_target = max(min_pc, 1) / max(rho_pc, 1e-12)
                Rhex = math.sqrt((2.0 * A_target) / (3.0 * math.sqrt(3.0)))

                xmin = float(L[:, 0].min() - alpha_margin_mm)
                xmax = float(L[:, 0].max() + alpha_margin_mm)
                ymin = float(L[:, 1].min() - alpha_margin_mm)
                ymax = float(L[:, 1].max() + alpha_margin_mm)

                centers = _build_hex_centers_inside_hull(hull, xmin, xmax, ymin, ymax, Rhex)

                D2 = ((L[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                assign = np.argmin(D2, axis=1)

                P = centers.shape[0]
                removed = np.zeros(P, dtype=bool)

                while True:
                    pc_counts = np.bincount(assign, weights=is_pc.astype(int), minlength=P).astype(int)
                    all_counts = np.bincount(assign, minlength=P).astype(int)

                    cand = np.where((pc_counts < min_pc) & (~removed))[0]
                    if cand.size == 0:
                        break

                    w = cand[np.argsort(pc_counts[cand] + 1e-6 * all_counts[cand])][0]
                    removed[w] = True

                    keep_mask = ~removed
                    idx_w = np.where(assign == w)[0]

                    if idx_w.size:
                        keep_idx = np.nonzero(keep_mask)[0]

                        if keep_idx.size == 0:
                            removed[w] = False
                            break
                        elif keep_idx.size == 1:
                            assign[idx_w] = keep_idx[0]
                        else:
                            Ck = centers[keep_mask]
                            D2w = ((L[idx_w][:, None, :] - Ck[None, :, :]) ** 2).sum(axis=2)
                            jmin = np.argmin(D2w, axis=1)
                            assign[idx_w] = keep_idx[jmin]

                    if keep_mask.sum() <= 1:
                        break

                keep = ~removed if removed.any() else np.ones(P, dtype=bool)
                centers = centers[keep]
                remap = {old: new for new, old in enumerate(np.nonzero(keep)[0])}
                assign = np.array([remap[a] for a in assign], dtype=int)

            rows = []
            for pid in range(centers.shape[0]):
                m = (assign == pid)
                rows.append(
                    {
                        "group": g,
                        "patch_id": pid,
                        "x_center": float(centers[pid, 0]),
                        "y_center": float(centers[pid, 1]),
                        "n_total": int(m.sum()),
                        "PC": int((C[m] == "PC").sum()),
                        "RA": int((C[m] == "RA").sum()),
                        "SA1": int((C[m] == "SA1").sum()),
                    }
                )
            summary_df = pd.DataFrame(rows).sort_values("patch_id").reset_index(drop=True)

            res = {
                "group": g,
                "centers": centers,
                "patch_id_per_afferent": assign,
                "global_indices": G,
                "x": L[:, 0],
                "y": L[:, 1],
                "class_str": C,
                "region_str": R,
                "idx": IDX,
                "params": {
                    "min_pc": int(min_pc),
                    "group_prefix": str(g),
                    "region_whitelist": whitelist,
                    "alpha_margin_mm": float(alpha_margin_mm),
                },
                "summary_df": summary_df,
            }

            patches[g] = res
            summaries.append(summary_df)

        except Exception as e:
            failed[g] = str(e)

    summary_df = (
        pd.concat(summaries, ignore_index=True)
        if summaries else
        pd.DataFrame(columns=["group", "patch_id", "x_center", "y_center", "n_total", "PC", "RA", "SA1"])
    )

    return {
        "groups": groups,
        "patches": patches,
        "summary_df": summary_df,
        "failed_groups": failed,
    }

def collapse_patch_response(resp, patch_data):
    """
    Collapse afferent-level response dict to patch-level response dict.

    Parameters
    ----------
    resp : dict
        TouchSim response dictionary (from io.load), must contain at least:
        "spikes", "location", "class_str", "region_str". Optional: "idx".
    patch_data : dict
        Output from patch_hand(...).

    Returns
    -------
    patch_resp : dict
        {
            "spikes": list[np.ndarray],      # len = N_patches (pooled per patch)
            "location": np.ndarray (N, 2),   # patch centers
            "Class": np.ndarray (N, 3),      # columns: [PC, RA, SA1]
            "idx": list[np.ndarray],         # afferent idx values per patch
            "region": list[np.ndarray],      # unique region labels per patch
            "global_indices": list[np.ndarray],
            "group": np.ndarray (N,),        # group name per patch
            "patch_id": np.ndarray (N,),     # local patch id per group
        }
    patch_shapes : dict
        Geometry payload for plotting:
        {
            "location": np.ndarray (N, 2),
            "hulls": list[np.ndarray],       # each hull (M,2), may be empty
            "group": np.ndarray (N,),
            "patch_id": np.ndarray (N,),
        }
    """
    required_resp = {"spikes", "location", "class_str", "region_str"}
    missing = required_resp.difference(resp.keys())
    if missing:
        raise KeyError(f"response dictionary missing keys: {sorted(missing)}")

    spikes_all = resp["spikes"]
    loc_all = np.asarray(resp["location"], float)
    cls_all = _normalize_class_labels(resp["class_str"])
    reg_all = np.asarray(resp["region_str"]).astype(str)
    if "idx" in resp:
        idx_all = np.asarray(resp["idx"])
    else:
        idx_all = np.arange(len(spikes_all), dtype=int)

    patch_spikes = []
    patch_centers = []
    patch_class_counts = []
    patch_idx = []
    patch_region = []
    patch_gidx = []
    patch_group = []
    patch_pid = []
    patch_hulls = []

    for g in patch_data.get("groups", []):
        if g not in patch_data.get("patches", {}):
            continue

        pdata = patch_data["patches"][g]
        centers = np.asarray(pdata["centers"], float)
        assign = np.asarray(pdata["patch_id_per_afferent"], int)
        global_indices = np.asarray(pdata["global_indices"], int)

        if centers.ndim != 2 or centers.shape[1] < 2:
            continue

        for pid in range(centers.shape[0]):
            m = assign == pid
            if not np.any(m):
                continue

            gidx = global_indices[m]
            # pooled spikes for patch
            per_aff_sp = [np.asarray(spikes_all[i], float) for i in gidx]
            pooled = (
                np.sort(np.concatenate(per_aff_sp))
                if len(per_aff_sp)
                else np.empty(0, dtype=float)
            )

            cls_patch = cls_all[gidx]
            reg_patch = reg_all[gidx]
            idx_patch = np.asarray(idx_all[gidx])
            pts = np.asarray(loc_all[gidx, :2], float)
            hull, _ = _convex_hull_safe(pts)

            patch_spikes.append(pooled)
            patch_centers.append(np.asarray(centers[pid, :2], float))
            patch_class_counts.append(_class_count_vector(cls_patch))
            patch_idx.append(idx_patch)
            patch_region.append(np.unique(reg_patch))
            patch_gidx.append(gidx)
            patch_group.append(str(g))
            patch_pid.append(int(pid))
            patch_hulls.append(hull if hull.size else np.empty((0, 2), float))

    if patch_centers:
        centers_arr = np.vstack(patch_centers).astype(float)
        class_arr = np.vstack(patch_class_counts).astype(int)
        group_arr = np.asarray(patch_group).astype(str)
        pid_arr = np.asarray(patch_pid, int)
    else:
        centers_arr = np.empty((0, 2), float)
        class_arr = np.empty((0, 3), int)
        group_arr = np.empty((0,), dtype=str)
        pid_arr = np.empty((0,), dtype=int)

    patch_resp = {
        "spikes": patch_spikes,
        "location": centers_arr,
        "Class": class_arr,
        "idx": patch_idx,
        "region": patch_region,
        "global_indices": patch_gidx,
        "group": group_arr,
        "patch_id": pid_arr,
    }

    patch_shapes = {
        "location": centers_arr,
        "hulls": patch_hulls,
        "group": group_arr,
        "patch_id": pid_arr,
    }

    return patch_resp, patch_shapes

def patch(
    resp,
    *,
    min_pc=10,
    alpha_margin_mm=2.0,
    include_groups=None,
    exclude_groups=None,
):
    """
    Build patches and return collapsed patch-level response + plot shapes.

    This is the intended in-memory workflow entry point:
        resp = ts.load("pulse1.npz", kind="response")
        patch_resp, patch_shapes = ts.patch(resp)
    """
    patch_data = patch_hand(
        resp,
        min_pc=min_pc,
        alpha_margin_mm=alpha_margin_mm,
        include_groups=include_groups,
        exclude_groups=exclude_groups,
    )
    return collapse_patch_response(resp, patch_data)

def patch_expand(
    resp,
    *,
    min_pc=10,
    alpha_margin_mm=2.0,
    include_groups=None,
    exclude_groups=None,
):
    """
    End-to-end in-memory patch workflow (expanded afferent-level output).

    Example:
        resp = ts.load("pulse1.npz", kind="response")
        resp_with_patch, patch_shapes = ts.patch_expand(resp)
    """
    pdata = patch_hand(
        resp,
        min_pc=min_pc,
        alpha_margin_mm=alpha_margin_mm,
        include_groups=include_groups,
        exclude_groups=exclude_groups,
    )
    return expand_patch_response(resp, pdata)

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

