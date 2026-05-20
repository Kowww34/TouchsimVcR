"""Basic spike-train metrics."""

import numpy as np


def mean_firing(zz):
    """
    Compute mean firing rate for each neuron from spike times.

    The firing rate is computed as:
        FR = (N_spikes - 1) / (t_last - t_first)

    Parameters
    ----------
    zz : list of 1D arrays
        Spike times (seconds), sorted ascending, one array per neuron.

    Returns
    -------
    ndarray, shape (len(zz),)
        Mean firing rate (Hz). Zero for neurons with fewer than 2 spikes.
    """
    out_fr = []
    for s in zz:
        s = np.asarray(s, float)
        if s.size > 1:
            dt_support = s[-1] - s[0]
        else:
            dt_support = 0
        if dt_support > 0:
            fr = (s.size - 1) / dt_support
        else:
            fr = 0
        out_fr.append(fr)
    return np.asarray(out_fr, float)
