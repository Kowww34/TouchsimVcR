"""Secondary-neuron patch matplotlib plotting."""

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

def plot_patch_metric(
    patch_shapes,
    patch_metric,
    *,
    title="Patch metric",
    cbar_label="metric",
    cmap="viridis",
):
    """
    Plot scalar metric per patch using hull geometry returned by patch().
    """
    centers = np.asarray(patch_shapes["location"], float)
    hulls = patch_shapes["hulls"]
    vals = np.asarray(patch_metric, float)
    if vals.shape[0] != centers.shape[0]:
        raise ValueError(
            f"patch_metric length {vals.shape[0]} != number of patches {centers.shape[0]}"
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if vmin == vmax:
            vmin -= 1e-6
            vmax += 1e-6

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    for i, hull in enumerate(hulls):
        col = "lightgray" if not np.isfinite(vals[i]) else cmap_obj(norm(vals[i]))
        if np.asarray(hull).size and np.asarray(hull).shape[0] >= 3:
            h = np.asarray(hull, float)
            ax.fill(h[:, 0], h[:, 1], color=col, alpha=0.9, edgecolor="k", linewidth=0.8)
            ax.plot(
                np.r_[h[:, 0], h[0, 0]],
                np.r_[h[:, 1], h[0, 1]],
                color="k",
                linewidth=0.8,
            )
        else:
            ax.scatter(centers[i, 0], centers[i, 1], color=col, s=40)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax

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

