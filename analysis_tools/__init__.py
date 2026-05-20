"""
Spike-train analysis: firing rates, phase, and synchrony indices.

Submodules
----------
util
    ``mean_firing``
phase
    ``spike_phase``, ``sync_index``, ``sync_index_all``,
    ``cross_neuron_sync_max``, ``cross_neuron_sync_nostim``
"""

from .util import mean_firing
from .phase import (
    cross_neuron_sync_max,
    cross_neuron_sync_nostim,
    spike_phase,
    sync_index,
    sync_index_all,
)

__all__ = [
    "mean_firing",
    "spike_phase",
    "sync_index",
    "sync_index_all",
    "cross_neuron_sync_max",
    "cross_neuron_sync_nostim",
]
