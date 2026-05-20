### TouchSim Extensions (VCR + Analysis Toolkit)

This repository builds on the original TouchSim framework and adds tools for vibrotactile coordinated reset (vCR), data handling, and analysis.

### Original TouchSim

This work is based on https://github.com/hsaal/touchsim

If you are not familiar with TouchSim, start there. This repository assumes you understand its core concepts such as afferents, stimuli, and responses.

What this repository adds

### vCR sequence generation

This repository implements structured vibrotactile coordinated reset stimulation. This includes multi pad burst sequences, phase shifted stimulation across spatial locations, and configurable frequency, amplitude, and timing structure. The goal is to enable controlled desynchronization experiments at the peripheral level.

### Saving and loading utilities

A simplified I O system is provided through two functions

save(obj, kind, path, format="auto")
load(path, kind=None, format="auto")

Responses can be stored as compact binary files using a CSR spike representation in npz format. Afferent populations and stimuli are currently stored using pickle. This approach improves disk speed, reduces file size for large simulations, and supports reproducible workflows.

### Response filtering

The function filter_response allows efficient sub selection of afferents from a response dictionary. Filtering can be performed using afferent class, region, or index. Region filtering supports prefix matching such as using D2 to select all D2 related regions. The function is chainable and preserves the dictionary structure of the response.

### Analysis tools

The repository includes tools for time resolved firing rate estimation, synchrony metrics such as the order parameter, spike/phase mapping, spatial grouping using nearest neighbor methods, and visualization of afferent activity across the hand. These tools are intended to bridge peripheral responses with downstream modeling and analysis.

### Static strain and distance bug fix

The original TouchSim implementation uses Euclidean distance to compute interactions between pins. This leads to non physical behavior where signals can propagate unrealistically across the hand, including between fingers that are not physically connected.

Two fixes are implemented.

1) The first fix introduces a propagation cutoff. Interactions beyond a specified distance are set to zero. This removes long range non physical coupling and enforces locality, although it introduces a hard cutoff. Dynamic pressure transmission, which was originally coded with delay is unaffected.

2) The second fix introduces a propagation delay proportional to distance. Static strain contributions are shifted in time according to tau equals distance divided by velocity. This changes the behavior from instantaneous coupling to wave like propagation and improves physical realism.

Mechanical backend is selected per stimulus: ``stim_sine(..., transduction='lag')``
(one of ``orig``, ``cutoff``, ``lag``). Defaults to ``cutoff``. Cutoff radii are
configured via ``touchsim.transduction.set_distance_cutoffs_mm``.

