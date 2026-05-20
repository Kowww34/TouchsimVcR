#!/usr/bin/env python3
"""
Smoke-test all transduction backends with sine and VCR stimuli.

For each (backend, stimulus_type) pair, builds a small hand population, runs
``affpop.response(stim)``, and checks basic invariants. No save/load/plot per
combo.

One full round-trip (save, load, plot) is run for a single chosen case
(default: cutoff + stim_sine).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.expanduser("~"))

import holoviews as hv

import touchsim as ts
from touchsim import transduction
from touchsim.classes import Response
from touchsim.plotting import figsave, plot

hv.extension("matplotlib")

BACKENDS = ("orig", "cutoff", "lag")
STIM_BUILDERS = ("sine", "vcr_single_pulse")


def _d2_center():
    return np.asarray(
        ts.hand_surface.centers[ts.hand_surface.tag2idx("D2d_t")], dtype=float
    ).reshape(-1, 2)[0]


def _small_affpop():
    return ts.affpop_hand(region="D2d_t", affclass=["PC"], density_multiplier=0.25)


def build_stim(stim_type: str, transduction_name: str):
    """Return a short stimulus for smoke tests."""
    if stim_type == "sine":
        # Multi-pin indent: single-pin sine breaks lag ``add_delays`` (pin count mismatch).
        base = ts.stim_sine(
            freq=200.0,
            amp=0.02,
            len=0.15,
            loc=_d2_center(),
            fs=5000.0,
            pin_radius=0.5,
        )
        pins = ts.shape_circle(radius=1.5, pins_per_mm=1.5, center=_d2_center())
        return ts.stim_indent_shape(
            pins, base, pin_radius=0.5, transduction=transduction_name
        )
    if stim_type == "vcr_single_pulse":
        return ts.stim_vcr_single_pulse(
            freq=250.0,
            amp=0.1,
            active_finger="D2",
            burst_len=0.1,
            edge_pad_len=0.05,
            fingers=["D2", "D3"],
            indent_map={"D2": 0.0, "D3": 0.0},
            transduction=transduction_name,
        )
    raise ValueError(f"unknown stim_type: {stim_type!r}")


def smoke_run(backend: str, stim_type: str) -> int:
    """Run one backend × stimulus; return total spike count."""
    stim = build_stim(stim_type, backend)
    assert stim.transduction == backend, (
        f"expected transduction={backend!r}, got {stim.transduction!r}"
    )
    assert stim._profile is not None and stim._profiledyn is not None
    assert stim._profile.shape[0] > 0

    affpop = _small_affpop()
    resp = affpop.response(stim)
    assert len(resp) == len(affpop)
    n_spikes = int(np.sum([len(s) for s in resp.spikes]))
    assert n_spikes >= 0
    return n_spikes


def verify_save_load_plot(transduction_name: str = "cutoff", stim_type: str = "sine"):
    """Full I/O + HoloViews plot for one configuration."""
    affpop = _small_affpop()
    stim = build_stim(stim_type, transduction_name)
    resp = affpop.response(stim)
    n_before = int(np.sum([len(s) for s in resp.spikes]))

    with tempfile.TemporaryDirectory() as tmp:
        aff_path = os.path.join(tmp, "aff.pkl")
        stim_path = os.path.join(tmp, "stim.pkl")
        resp_path = os.path.join(tmp, "resp.npz")

        ts.save(affpop, "affpop", aff_path)
        ts.save(stim, "stimulus", stim_path)
        ts.save(resp, "response", resp_path)

        aff_loaded = ts.load(aff_path)
        stim_loaded = ts.load(stim_path)
        resp_loaded = ts.load(resp_path, kind="response")

        assert stim_loaded.transduction == transduction_name
        assert len(aff_loaded) == len(affpop)
        spikes_reload = resp_loaded["spikes"]
        n_after = int(np.sum([len(s) for s in spikes_reload]))
        assert n_after == n_before

        resp_obj = Response(aff_loaded, [stim_loaded], [spikes_reload])
        figsave(plot(aff_loaded), os.path.join(tmp, "affpop"))
        figsave(plot(stim_loaded), os.path.join(tmp, "stim"))
        figsave(plot(resp_obj, spatial=True), os.path.join(tmp, "resp"))

        for name in ("affpop", "stim", "resp"):
            assert os.path.isfile(os.path.join(tmp, f"{name}.png"))


class TestTransductionBackendsSmoke(unittest.TestCase):
    def test_all_backend_stim_combos(self):
        results = {}
        for backend in BACKENDS:
            self.assertIn(backend, transduction.BACKENDS)
            for stim_type in STIM_BUILDERS:
                with self.subTest(backend=backend, stim=stim_type):
                    n = smoke_run(backend, stim_type)
                    results[(backend, stim_type)] = n
        # At least one combo should produce spikes (sanity, not strict per-backend).
        self.assertTrue(any(n > 0 for n in results.values()), results)

    def test_save_load_plot_cutoff_sine(self):
        verify_save_load_plot("cutoff", "sine")


class TestTransductionBackendsVcrSequence(unittest.TestCase):
    """Full multi-finger stim_vcr; optional (--full from CLI)."""

    def test_vcr_sequence_cutoff(self):
        stim = ts.stim_vcr(
            freq=200.0,
            amp=0.05,
            burst_len=0.08,
            fingers=["D2", "D3"],
            order=["D2", "D3"],
            transduction="cutoff",
        )
        affpop = _small_affpop()
        resp = affpop.response(stim)
        self.assertEqual(len(resp), len(affpop))


def main():
    parser = argparse.ArgumentParser(description="Transduction backend smoke tests")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run stim_vcr (multi-finger sequence) smoke test",
    )
    parser.add_argument(
        "--pipeline-backend",
        default="cutoff",
        choices=BACKENDS,
        help="Backend for save/load/plot check",
    )
    parser.add_argument(
        "--pipeline-stim",
        default="sine",
        choices=STIM_BUILDERS,
        help="Stimulus type for save/load/plot check",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    failed = []
    for backend in BACKENDS:
        for stim_type in STIM_BUILDERS:
            label = f"{backend}/{stim_type}"
            try:
                n = smoke_run(backend, stim_type)
                if args.verbose:
                    print(f"OK  {label}  spikes={n}", flush=True)
            except Exception as exc:
                failed.append((label, exc))
                print(f"FAIL {label}: {exc}", flush=True)

    if args.full:
        label = "cutoff/vcr_sequence"
        try:
            stim = ts.stim_vcr(
                freq=200.0,
                amp=0.05,
                burst_len=0.08,
                fingers=["D2", "D3"],
                order=["D2", "D3"],
                transduction="cutoff",
            )
            resp = _small_affpop().response(stim)
            n = int(np.sum([len(s) for s in resp.spikes]))
            if args.verbose:
                print(f"OK  {label}  spikes={n}", flush=True)
        except Exception as exc:
            failed.append((label, exc))
            print(f"FAIL {label}: {exc}", flush=True)

    if failed:
        print(f"\n{len(failed)} smoke test(s) failed.", flush=True)
        raise SystemExit(1)

    print(f"\nAll {len(BACKENDS) * len(STIM_BUILDERS)} smoke tests passed.", flush=True)

    try:
        verify_save_load_plot(args.pipeline_backend, args.pipeline_stim)
        print(
            f"OK  save/load/plot  backend={args.pipeline_backend}  stim={args.pipeline_stim}",
            flush=True,
        )
    except Exception as exc:
        print(f"FAIL save/load/plot: {exc}", flush=True)
        raise SystemExit(1) from exc

    print("Done.", flush=True)


if __name__ == "__main__":
    if "--unittest" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--unittest"]
        unittest.main()
    else:
        main()
