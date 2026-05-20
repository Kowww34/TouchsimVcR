#!/usr/bin/env python3
"""Peak memory comparison across transduction backends (identical stimulus params)."""

from __future__ import annotations

import gc
import os
import sys
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.expanduser("~"))

import touchsim as ts

BACKENDS = ("orig", "cutoff", "lag")


def d2_center():
    return np.asarray(
        ts.hand_surface.centers[ts.hand_surface.tag2idx("D2d_t")], dtype=float
    ).reshape(-1, 2)[0]


def build_sine(backend, len_s=0.15, radius=1.5, ppm=1.5, dens=0.25):
    center = d2_center()
    base = ts.stim_sine(
        freq=200.0,
        amp=0.02,
        len=len_s,
        loc=center,
        fs=5000.0,
        pin_radius=0.5,
        backend=backend,
    )
    pins = ts.shape_circle(radius=radius, pins_per_mm=ppm, center=center)
    stim = ts.stim_indent_shape(pins, base, pin_radius=0.5, backend=backend)
    aff = ts.affpop_hand(region="D2d_t", affclass=["PC"], density_multiplier=dens)
    return stim, aff


def peak_mb(fn):
    """Return (peak_traced_mb, result) for one call after gc."""
    gc.collect()
    tracemalloc.start()
    try:
        out = fn()
        _, peak = tracemalloc.get_traced_memory()
        return peak / (1024 * 1024), out
    finally:
        tracemalloc.stop()


def retained_mb(obj):
    """Shallow size of key stimulus/response arrays (MiB)."""
    if hasattr(obj, "_profile"):
        stim = obj
        prof = stim._profile.nbytes + stim._profiledyn.nbytes
        loc = stim.location.nbytes + stim.trace.nbytes
        return (prof + loc) / (1024 * 1024)
    if hasattr(obj, "spikes"):
        return sum(s.nbytes for s in obj.spikes) / (1024 * 1024)
    return 0.0


def run_case(label, len_s, radius, ppm, dens):
    print(f"\n=== {label} ===")
    rows = []
    for backend in BACKENDS:
        def build():
            return build_sine(backend, len_s, radius, ppm, dens)

        peak_build, (stim, aff) = peak_mb(build)

        def respond():
            return aff.response(stim)

        peak_resp, resp = peak_mb(respond)

        rows.append(
            (
                backend,
                len(stim),
                stim.trace.shape[1],
                len(aff),
                peak_build,
                peak_resp,
                retained_mb(stim),
                retained_mb(resp),
            )
        )
        del stim, aff, resp
        gc.collect()

    print(
        f"{'backend':<8} {'pins':>5} {'samples':>8} {'aff':>5} "
        f"{'peak_build':>11} {'peak_resp':>10} {'stim_ret':>9} {'resp_ret':>9}"
    )
    print("-" * 72)
    for row in rows:
        print(
            f"{row[0]:<8} {row[1]:5d} {row[2]:8d} {row[3]:5d} "
            f"{row[4]:9.2f} MiB {row[5]:8.2f} MiB {row[6]:7.2f} MiB {row[7]:7.2f} MiB"
        )

    orig = rows[0]
    lag = rows[2]
    print(
        f"\n  lag vs orig peak_build: {lag[4]/orig[4]:.2f}x   "
        f"peak_resp: {lag[5]/orig[5]:.2f}x"
    )


def main():
    print("Peak traced Python allocations (tracemalloc); one run per stage after gc.")
    run_case(
        "Light (smoke-test scale)",
        len_s=0.15,
        radius=1.5,
        ppm=1.5,
        dens=0.25,
    )
    run_case(
        "Heavy (60 pins, 1 s, full density)",
        len_s=1.0,
        radius=2.5,
        ppm=2.0,
        dens=1.0,
    )
    print("\nNotes:")
    print("  peak_build  = stimulus construction (skin_touch_profile dominates)")
    print("  peak_resp   = affpop.response (propagate + LIF)")
    print("  stim_ret    = nbytes of profile arrays kept on Stimulus")
    print("  lag uses joblib threads; peak may include short-lived per-pin workspaces")


if __name__ == "__main__":
    main()
