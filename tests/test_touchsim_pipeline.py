#!/usr/bin/env python3
"""
Integration test: create affpop/stimulus/response, save, load, and plot.

Uses TouchSim HoloViews plotting (``touchsim.plotting.plot`` / ``figsave``),
not matplotlib.pyplot directly.
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
from touchsim.classes import Response
from touchsim.plotting import figsave, plot

hv.extension("matplotlib")


def _run_pipeline(transduction: str, out_dir: str) -> None:
    """Create, simulate, save, load, and plot one backend configuration."""
    affpop = ts.affpop_hand(region="D2d_t", affclass=["PC"], density_multiplier=0.3)
    center = ts.hand_surface.centers[ts.hand_surface.tag2idx("D2d_t")]
    stim = ts.stim_sine(
        freq=200.0,
        amp=0.02,
        len=0.2,
        loc=center,
        fs=5000.0,
        pin_radius=0.5,
        transduction=transduction,
    )
    resp = affpop.response(stim)

    aff_path = os.path.join(out_dir, f"aff_{transduction}.pkl")
    stim_path = os.path.join(out_dir, f"stim_{transduction}.pkl")
    resp_path = os.path.join(out_dir, f"resp_{transduction}.npz")

    ts.save(affpop, "affpop", aff_path)
    ts.save(stim, "stimulus", stim_path)
    ts.save(resp, "response", resp_path)

    aff_loaded = ts.load(aff_path)
    stim_loaded = ts.load(stim_path)
    resp_loaded = ts.load(resp_path, kind="response")

    assert stim_loaded.transduction == transduction
    assert len(aff_loaded) == len(affpop)
    assert len(resp_loaded["spikes"]) == len(affpop)

    spikes_reload = resp_loaded["spikes"]
    resp_obj = Response(aff_loaded, [stim_loaded], [spikes_reload])

    figsave(plot(aff_loaded), os.path.join(out_dir, f"aff_{transduction}"))
    figsave(plot(stim_loaded), os.path.join(out_dir, f"stim_{transduction}"))
    figsave(
        plot(resp_obj, spatial=True),
        os.path.join(out_dir, f"resp_{transduction}"),
    )

    n_spikes_orig = sum(len(s) for s in resp.spikes)
    n_spikes_load = sum(len(s) for s in spikes_reload)
    assert n_spikes_load == n_spikes_orig
    assert n_spikes_orig > 0


class TestTouchSimPipeline(unittest.TestCase):
    def test_cutoff_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            _run_pipeline("cutoff", tmp)

    def test_orig_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            _run_pipeline("orig", tmp)


class TestTouchSimPipelineFull(unittest.TestCase):
    """Lag backend is slower; run only with --full."""

    def test_lag_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            _run_pipeline("lag", tmp)


def main():
    parser = argparse.ArgumentParser(description="TouchSim pipeline integration test")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run lag-backend test (slower)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory for plots (default: temp dir)",
    )
    args = parser.parse_args()

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestTouchSimPipeline)
    if args.full:
        suite.addTests(loader.loadTestsFromTestCase(TestTouchSimPipelineFull))

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        _run_pipeline("cutoff", args.out_dir)
        _run_pipeline("orig", args.out_dir)
        if args.full:
            _run_pipeline("lag", args.out_dir)
        print(f"Wrote plots to {args.out_dir}")
        return

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
