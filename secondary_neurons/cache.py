"""Secondary-neuron phase cache."""

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

