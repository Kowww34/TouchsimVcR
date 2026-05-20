#!/usr/bin/env python3
"""Compare runtime across transduction backends with identical stimulus parameters."""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.expanduser("~"))

import touchsim as ts
from touchsim.transduction import (
    check_pin_radius,
    circ_load_dyn_wave,
    circ_load_vert_stress,
    skin_touch_profile,
)

BACKENDS = ("orig", "cutoff", "lag")
REPEATS = 3


def d2_center():
    return np.asarray(
        ts.hand_surface.centers[ts.hand_surface.tag2idx("D2d_t")], dtype=float
    ).reshape(-1, 2)[0]


def build_sine_stim(backend: str):
    """Same multi-pin sine setup as test_transduction_backends."""
    center = d2_center()
    base = ts.stim_sine(
        freq=200.0,
        amp=0.02,
        len=0.15,
        loc=center,
        fs=5000.0,
        pin_radius=0.5,
        backend=backend,
    )
    pins = ts.shape_circle(radius=1.5, pins_per_mm=1.5, center=center)
    return ts.stim_indent_shape(pins, base, pin_radius=0.5, backend=backend)


def small_affpop():
    return ts.affpop_hand(region="D2d_t", affclass=["PC"], density_multiplier=0.25)


def bench(fn, repeats=REPEATS):
    for _ in range(1):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def main():
    affpop = small_affpop()
    n_aff = len(affpop)
    print(f"Afferent population: {n_aff} PC afferents (D2d_t, density_multiplier=0.25)")
    print(f"Timing: {REPEATS} repeats after 1 warmup\n")

    rows = []
    for backend in BACKENDS:
        stim = build_sine_stim(backend)
        n_pins = len(stim)
        n_samples = stim.trace.shape[1]

        def make_stim():
            build_sine_stim(backend)

        def profile_only():
            s = build_sine_stim(backend)
            r = check_pin_radius(s.location, s.pin_radius)
            if s.pin_radius > r:
                s.pin_radius = r
            skin_touch_profile(
                s.trace, s.location, s.fs, s.pin_radius, backend=backend
            )

        def propagate_only():
            s = build_sine_stim(backend)
            s.propagate(affpop)

        def full_response():
            s = build_sine_stim(backend)
            affpop.response(s)

        t_stim, _ = bench(make_stim)
        t_prof, _ = bench(profile_only)
        t_prop, _ = bench(propagate_only)
        t_resp, _ = bench(full_response)

        rows.append(
            (backend, n_pins, n_samples, t_stim, t_prof, t_prop, t_resp)
        )

    header = (
        f"{'backend':<8} {'pins':>5} {'samples':>8} "
        f"{'stim+solve':>12} {'profile':>10} {'propagate':>11} {'response':>10}"
    )
    print(header)
    print("-" * len(header))
    for backend, n_pins, n_samples, t_stim, t_prof, t_prop, t_resp in rows:
        print(
            f"{backend:<8} {n_pins:5d} {n_samples:8d} "
            f"{t_stim:10.3f}s {t_prof:9.3f}s {t_prop:10.3f}s {t_resp:9.3f}s"
        )

    print("\nNotes:")
    print("  stim+solve  = build stimulus (includes skin_touch_profile on create)")
    print("  profile     = skin_touch_profile only (re-built stim each repeat)")
    print("  propagate   = stim.propagate(affpop): static + dynamic stress, all afferents")
    print("  response    = full affpop.response (propagate + LIF for all afferents)")


if __name__ == "__main__":
    main()
