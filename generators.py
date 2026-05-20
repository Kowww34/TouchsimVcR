"""
Factory functions for afferent populations and tactile stimuli.

All ``stim_*`` builders accept ``backend`` (default ``orig``) and forward it to
``Stimulus``. ``transduction`` is a backward-compatible alias. Use ``affpop_*``
helpers to place afferents on surfaces.
"""

import numpy as np
from scipy import signal
import random

from .classes import Afferent,AfferentPopulation,Stimulus
from .surface import Surface, hand_surface


def _make_stimulus(trace, location, fs, pin_radius, backend=None, transduction=None):
    """Build a Stimulus; ``backend`` (or alias ``transduction``) sets all pipeline steps."""
    kwargs = dict(
        trace=trace,
        location=location,
        fs=fs,
        pin_radius=pin_radius,
    )
    if backend is not None:
        kwargs["backend"] = backend
    if transduction is not None:
        kwargs["transduction"] = transduction
    return Stimulus(**kwargs)


def _backend_kwargs_from_args(args):
    """Extract ``backend`` / ``transduction`` from a stim_* ``**args`` dict."""
    kw = {}
    if "backend" in args:
        kw["backend"] = args.get("backend")
    if "transduction" in args:
        kw["transduction"] = args.get("transduction")
    return kw


default_params ={'dist':1.,
                 'max_extent':10.,
                 'affclass':Afferent.affclasses,
                 'idx':None,
                 'len':1.,
                 'loc':np.array([0, 0]),
                 'fs':5000.,
                 'ramp_len':0.05,
                 'ramp_type':'lin',
                 'pin_radius':2.5,
                 'pre_indent':0.,
                 'pad_len':0.,
                 'pins_per_mm':2}

def affpop_single_models(**args):
    """Returns AfferentPopulation containing all single neuron models.

    Kwargs:
        affclass: Single affclass or list, e.g. ['SA1','RA'] (default: all).
        args: All other kwargs will be passed on to Afferent constructor.

    Returns:
        AfferentPopulation object.
    """
    affclass = args.pop('affclass',default_params['affclass'])
    if type(affclass) is not list:
        affclass = [affclass]
    a = AfferentPopulation()
    for t in affclass:
        for i in range(Afferent.affparams.get(t).shape[0]):
            a.afferents.append(Afferent(t,idx=i,**args))
    return a


def affpop_linear(**args):
    """Generates afferents on a line extending from the origin along the 1st axis.

    Kwargs:
        dist (float): distance between neighboring afferent locations in mm
            (default 1.).
        max_extent (float): distance of farthest afferent in mm (default: 10.).
        affclass (str or list): Single affclass or list (default: ['SA1','RA','PC']).
        idx (int): Afferent model index; None (default) picks all available.
        args: All other kwargs will be passed on to Afferent constructor.

    Returns:
        AfferentPopulation object.
    """
    affclass = args.pop('affclass',default_params['affclass'])
    if type(affclass) is not list:
        affclass = [affclass]
    dist = args.pop('dist',default_params['dist'])
    max_extent = args.pop('max_extent',default_params['max_extent'])
    idx = args.pop('idx',default_params['idx'])

    locs = np.r_[0.:max_extent+dist:dist]

    a = AfferentPopulation()

    for l in np.nditer(locs):
        if idx is None:
            a_sub = affpop_single_models(location=np.array([l,0]),**args)
            a.afferents.extend(a_sub.afferents)
        else:
            for t in affclass:
                a.afferents.append(
                    Afferent(t,location=np.array([l,0]),idx=idx,**args))
    return a


def affpop_grid(**args):
    """Generates afferents on a 2D square grid centred on the origin.

    Kwargs:
        dist (float): distance between neighboring afferent locations in mm
            (default 1.).
        max_extent (float): length of square in mm (default: 10.).
        affclass (str or list): Single affclass or list (default: ['SA1','RA','PC']).
        idx (int): Afferent model index; None (default) picks all available.
        args: All other kwargs will be passed on to Afferent constructor.

    Returns:
        AfferentPopulation object.
    """
    affclass = args.pop('affclass',default_params['affclass'])
    if type(affclass) is not list:
        affclass = [affclass]
    dist = args.pop('dist',default_params['dist'])
    max_extent = args.pop('max_extent',default_params['max_extent'])
    idx = args.pop('idx',default_params['idx'])

    locs = np.r_[-max_extent/2:max_extent/2+dist:dist]

    a = AfferentPopulation()

    for l1 in np.nditer(locs):
        for l2 in np.nditer(locs):
            if idx is None:
                a_sub = affpop_single_models(
                    location=np.array([l1,l2]),**args)
                a.afferents.extend(a_sub.afferents)
            else:
                for t in affclass:
                    a.afferents.append(
                        Afferent(t,location=np.array([l1,l2]),idx=idx,**args))
    return a


def affpop_hand(**args):
    """Places receptors on the standard hand surface.

    Kwargs:
        affclass (str or list): Single affclass or list, (default: ['SA1','RA','PC']).
        region (str): identifies region(s) to populate, with None selecting
            all regions (default: None), e.g.
            'D2' for a full finger (digit 2), or
            'D2d' for a part (tip of digit 2).
        density (dictionary): Mapping from tags to densities
            (default: hand_surface.density).
        density_multiplier (float): Allows proportional scaling of densities
            (default: 1.).
        args: All other kwargs will be passed on to Afferent constructor.

    Returns:
        AfferentPopulation object.
    """
    return affpop_surface(surface=hand_surface,**args)

def affpop_random(**args):
    """Places receptors on a surface. Like affpop_hand(), but for arbitrary
    surfaces using the surface keyword.
    """
    affclass = args.pop('affclass',default_params['affclass'])
    surface = args.pop('surface',hand_surface)
    density = args.pop('density',surface.density)
    density_multiplier = args.pop('density_multiplier',1.)
    if type(affclass) is not list:
        affclass = [affclass]
    region = args.pop('region',None)
    seed = args.pop('seed',None)

    if seed is not None:
        random.seed(seed)

    idx = surface.tag2idx(region)

    afferents = list()
    for a in affclass:
        for i in idx:
            dens = density_multiplier*density[(a,i)]
            xy = surface.sample_uniform(i,density=dens,seed=seed)
            for l in range(xy.shape[0]):
                afferents.append(RandomAfferent(a,location=xy[l,:],**args))
    affpop = AfferentPopulation(surface=surface,*afferents)
    return affpop

def affpop_surface(**args):
    """Places receptors on a surface. Like affpop_hand(), but for arbitrary
    surfaces using the surface keyword.
    """
    affclass = args.pop('affclass',default_params['affclass'])
    surface = args.pop('surface',hand_surface)
    density = args.pop('density',surface.density)
    density_multiplier = args.pop('density_multiplier',1.)
    if type(affclass) is not list:
        affclass = [affclass]
    region = args.pop('region',None)
    seed = args.pop('seed',None)

    if seed is not None:
        random.seed(seed)

    idx = surface.tag2idx(region)

    afferents = list()
    for a in affclass:
        for i in idx:
            dens = density_multiplier*density[(a,i)]
            xy = surface.sample_uniform(i,density=dens,seed=seed)
            for l in range(xy.shape[0]):
                afferents.append(Afferent(a,location=xy[l,:],**args))
    affpop = AfferentPopulation(surface=surface,*afferents)
    return affpop

def stim_sine(**args):
    """Generates indenting complex sine stimulus.

    Kwargs:
        freq (float or list): vector of frequencies in Hz (default: 200.).
        amp (float or list): vector of amplitudes in mm (default: 0.02).
        phase (float or list): vector of phases in degrees (default: 0.).
        len (float): stimulus duration in s (default: 1.).
        loc (array): stimulus location in mm (default: [0, 0]).
        fs (float): sampling frequency in Hz (default 5000.).
        ramp_len (float): length of on and off ramps in s (default: 0.05).
        ramp_type (str): 'lin' or 'sin' (default: 'lin').
        pin_radius (float): radius of probe pin in mm (default: 0.5).
        pre_indent (float): static indentation throughout trial (default: 0.).
        pre_pad_len (float): duration of pre-stimulus zero-padding (default: 0.).
        post_pad_len (float): duration of post-stimulus zero-padding (default: 0.).
        backend (str): Mechanical backend ('orig', 'cutoff', 'lag'; default: 'orig').
            Alias: ``transduction``.

    Returns:
        Stimulus object.
    """
    freq = np.array(args.get('freq', 200.))
    amp = np.array(args.get('amp', .02 * np.ones(freq.shape)))
    phase = np.array(args.get('phase', np.zeros(freq.shape)))
    len = args.get('len', default_params['len'])
    loc = np.array(args.get('loc', default_params['loc']))
    fs = args.get('fs', default_params['fs'])
    ramp_len = args.get('ramp_len', default_params['ramp_len'])
    ramp_type = args.get('ramp_type', default_params['ramp_type'])
    pin_radius = args.get('pin_radius', default_params['pin_radius'])
    pre_indent = args.get('pre_indent', default_params['pre_indent'])
    pre_pad_len = args.get('pre_pad_len', default_params['pad_len'])
    post_pad_len = args.get('post_pad_len', default_params['pad_len'])

    trace = np.zeros(int(fs * len))
    for f, a, p in zip(np.nditer(freq), np.nditer(amp), np.nditer(phase)):
        trace += a * np.sin(p * np.pi / 180. \
                            + np.linspace(0., 2. * np.pi * f * len, int(fs * len)))

    apply_ramp(trace, ramp_len=ramp_len, fs=fs)

    if pre_pad_len > 0 or post_pad_len > 0:
        trace = apply_pad(trace, pre_pad_len=pre_pad_len, post_pad_len=post_pad_len, fs=fs)

    trace += pre_indent

    return _make_stimulus(
        trace=trace, location=loc, fs=fs, pin_radius=pin_radius,
        **_backend_kwargs_from_args(args),
    )


def stim_noise(**args):
    """Generates bandpass Gaussian white noise stimulus.

    Kwargs:
        freq (list): upper and lower bandpass frequencies in Hz (default: [100.,300.]).
        amp (float): amplitude (standard deviation of trace) in mm (default: 0.02).
        len (float): stimulus duration in s (default: 1.).
        loc (array): stimulus location in mm (default: [0, 0]).
        fs (float): sampling frequency in Hz (default 5000.).
        ramp_len (float): length of on and off ramps in s (default: 0.05).
        ramp_type (str): 'lin' or 'sin' (default: 'lin').
        pin_radius (float): radius of probe pin in mm (default: 0.5).
        pre_indent (float): static indentation throughout trial (default: 0.).
        pad_len (float): duration of stimulus zero-padding (default: 0.).
        seed (int): seed for random number generator (default: None).

    Returns:
        Stimulus object.
    """
    freq = args.get('freq',[100.,300.])
    amp = args.get('amp',.02)
    len = args.get('len',default_params['len'])
    loc = np.array(args.get('loc',default_params['loc']))
    fs = args.get('fs',default_params['fs'])
    ramp_len = args.get('ramp_len',default_params['ramp_len'])
    ramp_type = args.get('ramp_type',default_params['ramp_type'])
    pin_radius = args.get('pin_radius',default_params['pin_radius'])
    pre_indent = args.get('pre_indent',default_params['pre_indent'])
    pad_len = args.get('pad_len',default_params['pad_len'])
    seed = args.get('seed',None)

    if seed is not None:
        np.random.seed(seed)

    trace = np.random.randn(int(fs*len))

    bfilt,afilt = signal.butter(3,np.array(freq)/fs/2.,btype='bandpass')
    trace = signal.lfilter(bfilt,afilt,trace)

    trace = trace/np.std(trace)*amp

    apply_ramp(trace,ramp_len=ramp_len,fs=fs)
    if pad_len>0:
        trace = apply_pad(trace,pad_len=pad_len,fs=fs)
    trace += pre_indent

    return _make_stimulus(
        trace=trace, location=loc, fs=fs, pin_radius=pin_radius,
        **_backend_kwargs_from_args(args),
    )


def stim_impulse(**args):
    """Generates a short impulse to the skin.

    Kwargs:
        amp (float): amplitude of the pulse in mm (default: 0.03).
        len (float): pulse duration in s (default: 0.01).
        pad_len (float): duration of stimulus zero-padding (default: 0.045).
        loc (array): stimulus location in mm (default: [0, 0]).
        fs (float): sampling frequency in Hz (default 5000.).
        pin_radius (float): radius of probe pin in mm (default: 0.5).
        pre_indent (float): static indentation throughout trial (default: 0.).

    Returns:
        Stimulus object.
    """
    amp = args.get('amp',.03)
    len = args.get('len',0.01)
    loc = np.array(args.get('loc',default_params['loc']))
    fs = args.get('fs',default_params['fs'])
    pin_radius = args.get('pin_radius',default_params['pin_radius'])
    pre_indent = args.get('pre_indent',default_params['pre_indent'])
    pad_len = args.get('pad_len',default_params['pad_len'])

    trace = signal.windows.gaussian(int(fs*len),std=7) *\
        np.sin(np.linspace(-np.pi,np.pi,int(fs*len)))
    trace = trace/np.max(trace)*amp

    if pad_len>0:
        trace = apply_pad(trace,pad_len=pad_len,fs=fs)
    trace += pre_indent

    return _make_stimulus(
        trace=trace, location=loc, fs=fs, pin_radius=pin_radius,
        **_backend_kwargs_from_args(args),
    )


def stim_ramp(**args):
    """Generates ramp up / hold / ramp down indentation.

    Kwargs:
        amp (float): amplitude in mm (default: 1.).
        ramp_type (str): 'lin' or 'sin' (default: 'lin').
        len (float): total duration of stimulus in s (default: 1.).
        loc (array): stimulus location in mm (default: [0, 0]).
        fs (float): sampling frequency in Hz (default: 5000.).
        ramp_len (float): duration of on and off ramps in s (default 0.05).
        pin_radius (float): probe radius in mm (default: 0.5).
        pre_indent (float): static indentation throughout trial (default: 0.).
        pad_len (float): duration of stimulus zero-padding (default: 0.).

    Returns:
        Stimulus object.
    """
    amp = args.get('amp',1.)
    len = args.get('len',default_params['len'])
    loc = np.array(args.get('loc',default_params['loc']))
    fs = args.get('fs',default_params['fs'])
    ramp_len = args.get('ramp_len',default_params['ramp_len'])
    ramp_type = args.get('ramp_type',default_params['ramp_type'])
    pin_radius = args.get('pin_radius',default_params['pin_radius'])
    pre_indent = args.get('pre_indent',default_params['pre_indent'])
    pad_len = args.get('pad_len',default_params['pad_len'])

    trace = amp*np.ones(int(fs*len))
    apply_ramp(trace,ramp_len=ramp_len,fs=fs,ramp_type=ramp_type)
    if pad_len>0:
        trace = apply_pad(trace,pad_len=pad_len,fs=fs)
    trace += pre_indent

    return _make_stimulus(
        trace=trace, location=loc, fs=fs, pin_radius=pin_radius,
        **_backend_kwargs_from_args(args),
    )

def stim_vcr(**args):
    """Generates a vCR burst sequence across selectable fingers.

    Kwargs:
        freq (float): carrier frequency in Hz.
        amp (float): burst amplitude in mm.
        indent (float): baseline indentation in mm (default baseline for all active fingers).
        fs (float): sampling frequency in Hz (default: 5000).
        burst_len (float): burst duration in s (default: 0.100).
        cycle_len (float): reference cycle length in s for the baseline
            100 ms burst design (default: 0.667).
        base_burst_len (float): baseline burst duration used to derive pause
            length from cycle_len (default: 0.100). Pauses remain fixed when
            burst_len changes.
        delta (float): pulse window duration in s (default: 0.100).
        slope (float): tanh pulse slope (default: 1000).
        pin_radius (float): probe radius in mm (default: 0.5).
        fingers (list[str]): active fingers (default: ["D2","D3","D4","D5"]).
        order (None | "random" | list[int] | list[str]): firing order.
            - None / "random": random permutation of active fingers (default behavior).
            - list[int]: 0-based indices into `fingers`.
            - list[str]: finger tags from `fingers`, e.g. ["D2","D3","D4","D5"].
        indent_map (dict[str, float] | None): per-finger baseline indentation override.
            Example: {"D2": 0.5, "D3": 0.0, "D4": 0.0, "D5": 0.0}
        edge_pad_len: ignored; edge padding is fixed to inter-burst pause.

    Returns:
        Stimulus object.
    """
    freq = args.get('freq')
    amp = args.get('amp')
    indent = args.get('indent', 0.0)
    fs = args.get('fs', 5000)
    burst_len = args.get('burst_len', 0.100)
    cycle_len = args.get('cycle_len', 0.667)
    base_burst_len = args.get('base_burst_len', 0.100)
    # Match stim_burst behavior: default pulse window tracks burst_len.
    delta = args.get('delta', burst_len)
    slope = args.get('slope', 1000)
    pin_radius = args.get('pin_radius', default_params['pin_radius'])
    indenter_radius = args.get('indenter_radius', 2.5)
    pins_per_mm = args.get('pins_per_mm', 2.0)
    fingers = args.get('fingers', ['D2', 'D3', 'D4', 'D5'])
    order = args.get('order', None)
    indent_map = args.get('indent_map', None)
    edge_pad_len = args.get('edge_pad_len', None)
    if freq is None or amp is None:
        raise ValueError("stim_vcr requires 'freq' and 'amp'.")
    if type(fingers) is not list:
        fingers = list(fingers)
    fingers = [str(f).upper() for f in fingers]
    valid = {"D1", "D2", "D3", "D4", "D5"}
    if len(fingers) == 0:
        raise ValueError("stim_vcr requires at least one finger in `fingers`.")
    if len(set(fingers)) != len(fingers):
        raise ValueError("stim_vcr `fingers` entries must be unique.")
    if any(f not in valid for f in fingers):
        raise ValueError(f"stim_vcr `fingers` must be in {sorted(valid)}")

    # Resolve firing order as indices into `fingers`.
    n_fingers = len(fingers)
    if order is None or order == 'random':
        fire_idx = np.random.permutation(n_fingers)
    else:
        if type(order) is not list:
            order = list(order)
        if len(order) != n_fingers:
            raise ValueError("stim_vcr `order` must include each active finger exactly once.")
        if all(isinstance(o, (int, np.integer)) for o in order):
            fire_idx = np.asarray(order, dtype=int)
            if np.any(fire_idx < 0) or np.any(fire_idx >= n_fingers):
                raise ValueError("stim_vcr integer `order` entries out of range.")
            if len(np.unique(fire_idx)) != n_fingers:
                raise ValueError("stim_vcr integer `order` must be a permutation.")
        else:
            order_names = [str(o).upper() for o in order]
            if set(order_names) != set(fingers):
                raise ValueError("stim_vcr string `order` must be a permutation of `fingers`.")
            fire_idx = np.asarray([fingers.index(name) for name in order_names], dtype=int)

    # Per-finger baseline indentation; default to scalar `indent`.
    base_indent = np.full(n_fingers, float(indent), dtype=float)
    if indent_map is not None:
        if not isinstance(indent_map, dict):
            raise ValueError("stim_vcr `indent_map` must be a dict of {finger: indent_mm}.")
        for k, v in indent_map.items():
            kk = str(k).upper()
            if kk in fingers:
                base_indent[fingers.index(kk)] = float(v)

    dt = 1 / fs
    # Keep inter-pulse pause fixed from the baseline design:
    # pause = (cycle_len - n_fingers*base_burst_len) / n_fingers.
    # Increasing burst_len then increases total sequence duration while
    # preserving pause duration.
    inter_pad = (cycle_len - n_fingers * base_burst_len) / max(n_fingers, 1)
    if inter_pad < 0:
        raise ValueError("stim_vcr requires cycle_len >= len(fingers) * base_burst_len.")
    # Edge padding is fixed to inter-pulse pause.
    edge_pad = inter_pad
    w = 2 * np.pi * freq
    center_locs = np.array(
        [
            np.asarray(hand_surface.centers[hand_surface.tag2idx(f"{f}d")], dtype=float).reshape(-1, 2)[0]
            for f in fingers
        ],
        dtype=float,
    )
    t_max = n_fingers * burst_len + max(n_fingers - 1, 0) * inter_pad + 2 * edge_pad
    t = np.linspace(0.0, t_max, int(t_max / dt) + 1)
    stim_traces = np.tile(base_indent[:, None], (1, t.size))
    for j, fi in enumerate(fire_idx):
        t_start = edge_pad + j * (burst_len + inter_pad)
        t0 = t_start + burst_len / 2
        hf = 0.5 * (1 + np.cos(w * (t - t0)))
        pulse = (
            0.5
            * (np.tanh(slope * (t - t0 + delta / 2))
               - np.tanh(slope * (t - t0 - delta / 2)))
            * hf * amp
            + base_indent[fi]
        )
        mask = (t >= t_start) & (t <= (t_start + burst_len))
        stim_traces[fi, mask] = pulse[mask]
    # Expand each indenter center to a multi-pin circular contact by default.
    if pins_per_mm is not None and float(pins_per_mm) > 0 and float(indenter_radius) > 0:
        loc_blocks = []
        trace_blocks = []
        for fi in range(n_fingers):
            shp = shape_circle(radius=float(indenter_radius), pins_per_mm=float(pins_per_mm), center=center_locs[fi])
            xy = np.asarray(shp, float)[:, :2]
            loc_blocks.append(xy)
            trace_blocks.append(np.tile(stim_traces[fi:fi + 1, :], (xy.shape[0], 1)))
        locs = np.vstack(loc_blocks)
        stim_traces = np.vstack(trace_blocks)
    else:
        locs = center_locs

    return _make_stimulus(
        trace=stim_traces, location=locs, fs=fs, pin_radius=pin_radius,
        **_backend_kwargs_from_args(args),
    )


def stim_vcr_single_pulse(**args):
    """Generates a single-finger burst with other fingers held at static indent.

    This is a simplified, non-sequenced variant of stim_vcr:
    - one active finger receives a single burst
    - all listed fingers are present as indentors
    - non-active fingers stay at baseline indent throughout

    Kwargs:
        freq (float): carrier frequency in Hz.
        amp (float): burst amplitude in mm for active finger.
        active_finger (str): finger that receives the burst (default: "D2").
        fingers (list[str]): all fingers present as indentors
            (default: ["D2","D3","D4","D5"]).
        indent (float): default baseline indentation for all listed fingers (default: 0.0).
        indent_map (dict[str, float] | None): per-finger baseline overrides.
            Example: {"D2": 0.5, "D3": 0.1, "D4": 0.1, "D5": 0.1}
        fs (float): sampling frequency in Hz (default: 5000).
        burst_len (float): burst duration in s (default: 0.200).
        delta (float): tanh pulse window duration in s (default: burst_len).
        slope (float): tanh pulse slope (default: 1000).
        edge_pad_len (float): padding before and after burst in s (default: 0.100).
        pin_radius (float): probe radius in mm (default: 0.5).

    Returns:
        Stimulus object with multi-pin trace shape (N_fingers, T_samples).
    """
    freq = args.get('freq')
    amp = args.get('amp')
    active_finger = str(args.get('active_finger', 'D2')).upper()
    fingers = args.get('fingers', ['D2', 'D3', 'D4', 'D5'])
    indent = args.get('indent', 0.0)
    indent_map = args.get('indent_map', None)
    fs = args.get('fs', 5000)
    burst_len = args.get('burst_len', 0.200)
    delta = args.get('delta', burst_len)
    slope = args.get('slope', 1000)
    edge_pad = float(args.get('edge_pad_len', 0.100))
    pin_radius = args.get('pin_radius', 2.5)
    indenter_radius = args.get('indenter_radius', 2.5)
    pins_per_mm = args.get('pins_per_mm', 2.0)

    if freq is None or amp is None:
        raise ValueError("stim_vcr_single_pulse requires 'freq' and 'amp'.")
    if type(fingers) is not list:
        fingers = list(fingers)
    fingers = [str(f).upper() for f in fingers]
    valid = {"D1", "D2", "D3", "D4", "D5"}
    if len(fingers) == 0:
        raise ValueError("stim_vcr_single_pulse requires at least one finger in `fingers`.")
    if len(set(fingers)) != len(fingers):
        raise ValueError("stim_vcr_single_pulse `fingers` entries must be unique.")
    if any(f not in valid for f in fingers):
        raise ValueError(f"stim_vcr_single_pulse `fingers` must be in {sorted(valid)}")
    if active_finger not in fingers:
        raise ValueError("stim_vcr_single_pulse `active_finger` must be included in `fingers`.")
    if edge_pad < 0:
        raise ValueError("stim_vcr_single_pulse `edge_pad_len` must be >= 0.")

    n_fingers = len(fingers)
    dt = 1 / fs
    w = 2 * np.pi * freq
    t_max = burst_len + 2 * edge_pad
    t = np.linspace(0.0, t_max, int(t_max / dt) + 1)

    # Per-finger baseline indentation.
    base_indent = np.full(n_fingers, float(indent), dtype=float)
    if indent_map is not None:
        if not isinstance(indent_map, dict):
            raise ValueError("stim_vcr_single_pulse `indent_map` must be a dict of {finger: indent_mm}.")
        for k, v in indent_map.items():
            kk = str(k).upper()
            if kk in fingers:
                base_indent[fingers.index(kk)] = float(v)

    stim_traces = np.tile(base_indent[:, None], (1, t.size))
    active_idx = fingers.index(active_finger)
    t_start = edge_pad
    t0 = t_start + burst_len / 2
    hf = 0.5 * (1 + np.cos(w * (t - t0)))
    pulse = (
        0.5
        * (np.tanh(slope * (t - t0 + delta / 2))
           - np.tanh(slope * (t - t0 - delta / 2)))
        * hf * amp
        + base_indent[active_idx]
    )
    mask = (t >= t_start) & (t <= (t_start + burst_len))
    stim_traces[active_idx, mask] = pulse[mask]

    center_locs = np.array(
        [
            np.asarray(hand_surface.centers[hand_surface.tag2idx(f"{f}d")], dtype=float).reshape(-1, 2)[0]
            for f in fingers
        ],
        dtype=float,
    )

    # Expand each indenter center to a multi-pin circular contact by default.
    if pins_per_mm is not None and float(pins_per_mm) > 0 and float(indenter_radius) > 0:
        loc_blocks = []
        trace_blocks = []
        for fi in range(n_fingers):
            shp = shape_circle(radius=float(indenter_radius), pins_per_mm=float(pins_per_mm), center=center_locs[fi])
            xy = np.asarray(shp, float)[:, :2]
            loc_blocks.append(xy)
            trace_blocks.append(np.tile(stim_traces[fi:fi + 1, :], (xy.shape[0], 1)))
        locs = np.vstack(loc_blocks)
        stim_traces = np.vstack(trace_blocks)
    else:
        locs = center_locs

    return _make_stimulus(
        trace=stim_traces, location=locs, fs=fs, pin_radius=pin_radius,
        **_backend_kwargs_from_args(args),
    )


def stim_burst(**args):
    """Generates a single burst stimulus on D2.

    Kwargs:
        freq (float): carrier frequency in Hz.
        amp (float): burst amplitude in mm.
        indent (float): baseline indentation in mm.
        fs (float): sampling frequency in Hz (default: 5000).
        burst_len (float): burst duration in s (default: 0.100).
        cycle_len (float): total window length in s (default: 0.667/4).
        delta (float): pulse window duration in s (default: 0.100).
        slope (float): tanh pulse slope (default: 1000).
        loc (array): stimulus location in mm (default: D2d center).
        pin_radius (float): probe radius in mm (default: 0.5).

    Returns:
        Stimulus object.
    """
    freq = args.get('freq')
    amp = args.get('amp')
    indent = args.get('indent', 0.0)
    fs = args.get('fs', 5000)
    burst_len = args.get('burst_len', 0.100)
    cycle_len = args.get('cycle_len', 0.667 / 4.0)
    # By default, make the tanh window width match the requested burst length
    # so the high-amplitude vibration lasts exactly `burst_len` seconds.
    delta = args.get('delta', burst_len)
    slope = args.get('slope', 1000)
    pin_radius = args.get('pin_radius', default_params['pin_radius'])
    indenter_radius = args.get('indenter_radius', 2.5)
    pins_per_mm = args.get('pins_per_mm', 2.0)
    loc = np.array(args.get(
        'loc',
        hand_surface.centers[hand_surface.tag2idx('D2d')]
    ))
    if freq is None or amp is None:
        raise ValueError("stim_burst requires 'freq' and 'amp'.")
    dt = 1 / fs
    # Keep pre/post padding fixed to the baseline 100 ms burst design:
    # pad = (T - 0.100) / 2, with T = cycle_len.
    # Increasing burst_len only extends the active burst segment.
    base_burst_len = 0.100
    t_pad = (cycle_len - base_burst_len) / 2.0
    if t_pad < 0:
        raise ValueError(
            "cycle_len is too short for baseline padding rule: require cycle_len >= 0.100"
        )
    t_max = burst_len + 2 * t_pad
    w = 2 * np.pi * freq
    t = np.linspace(0.0, t_max, int(t_max / dt) + 1)
    t0 = t_pad + burst_len / 2
    hf = 0.5 * (1 + np.cos(w * (t - t0)))
    stim = (
        0.5
        * (np.tanh(slope * (t - t0 + delta / 2))
           - np.tanh(slope * (t - t0 - delta / 2)))
        * hf * amp
        + indent
    )
    # enforce exact baseline before and after burst window
    stim[(t < t_pad) | (t > (t_pad + burst_len))] = indent
    base = _make_stimulus(
        trace=stim, location=loc, fs=fs, pin_radius=pin_radius,
        **_backend_kwargs_from_args(args),
    )
    if pins_per_mm is not None and float(pins_per_mm) > 0 and float(indenter_radius) > 0:
        shp = shape_circle(radius=float(indenter_radius), pins_per_mm=float(pins_per_mm), center=np.asarray(loc).reshape(2))
        return stim_indent_shape(shp, base, pin_radius=pin_radius)
    return base


def stim_multiple_bursts(**args):
    """Generates a single synchronous burst delivered to two fingers.
    Kwargs:
        freq (float): carrier frequency in Hz.
        amp (float): burst amplitude in mm.
        indent (float): baseline indentation in mm.
        finger1 (str): first finger label in {'D1','D2','D3','D4','D5'}.
        finger2 (str): second finger label in {'D1','D2','D3','D4','D5'}.
        fs (float): sampling frequency in Hz (default: 5000).
        burst_len (float): burst duration in s (default: 0.100).
        cycle_len (float): total window length in s (default: 0.667/4).
        delta (float): pulse window duration in s (default: 0.100).
        slope (float): tanh pulse slope (default: 1000).
        pin_radius (float): probe radius in mm (default: 0.5).
    Returns:
        Stimulus object.
    """
    freq = args.get('freq')
    amp = args.get('amp')
    indent = args.get('indent', 0.0)
    finger1 = str(args.get('finger1', 'D2')).upper()
    finger2 = str(args.get('finger2', 'D5')).upper()
    fs = args.get('fs', 5000)
    burst_len = args.get('burst_len', 0.100)
    cycle_len = args.get('cycle_len', 0.667 / 4.0)
    delta = args.get('delta', 0.100)
    slope = args.get('slope', 1000)
    pin_radius = args.get('pin_radius', default_params['pin_radius'])
    if freq is None or amp is None:
        raise ValueError("stim_multiple_bursts requires 'freq' and 'amp'.")
    valid = {"D1", "D2", "D3", "D4", "D5"}
    if finger1 not in valid or finger2 not in valid:
        raise ValueError(f"finger1 and finger2 must be in {sorted(valid)}")
    dt = 1 / fs
    t_pad = cycle_len - burst_len
    t_max = burst_len + 2 * t_pad
    w = 2 * np.pi * freq
    t = np.linspace(0.0, t_max, int(t_max / dt) + 1)
    t0 = t_pad + burst_len / 2
    hf = 0.5 * (1 + np.cos(w * (t - t0)))
    stim = (
        0.5
        * (np.tanh(slope * (t - t0 + delta / 2))
           - np.tanh(slope * (t - t0 - delta / 2)))
        * hf * amp
        + indent
    )
    stim[(t < t_pad) | (t > (t_pad + burst_len))] = indent
    loc1 = hand_surface.centers[hand_surface.tag2idx(f"{finger1}d")]
    loc2 = hand_surface.centers[hand_surface.tag2idx(f"{finger2}d")]
    #single representative location to match Stimulus signature
    #using first finger for consistency
    return _make_stimulus(
        trace=stim, location=loc1, fs=fs, pin_radius=pin_radius,
        **_backend_kwargs_from_args(args),
    )

def stim_indent_shape(shape,trace,**args):
    """Applies indentation trace to several pins that make up a shape.

    Args:
        shape (2D array): pin positions making up object shape, e.g. shape_bar().
        trace (array or Stimulus):

    Kwargs:
        rectify (bool): Resets negative indentations to zero (default: True)
        pin_radius (float): probe radius in mm.
        fs (float): sampling frequency.

    Returns:
        Stimulus object.
     """
    if type(trace) is Stimulus:
        t = trace.trace[0:1]
        if 'fs' not in args:
            args['fs'] = trace.fs
        if 'pin_radius' not in args:
            args['pin_radius'] = trace.pin_radius
        if 'backend' not in args and 'transduction' not in args:
            args['backend'] = trace.backend
    else:
        t = np.reshape(np.atleast_2d(trace),(1,-1))

    t = np.tile(t,(shape.shape[0],1))

    if shape.shape[1]==3:
        t += shape[:,2:3]

    if args.pop('rectify',True):
        t[t<0] = 0

    fs = args.pop('fs', default_params['fs'])
    pin_radius = args.pop('pin_radius', default_params['pin_radius'])
    return _make_stimulus(
        trace=t,
        location=shape[:, 0:2],
        fs=fs,
        pin_radius=pin_radius,
        **_backend_kwargs_from_args(args),
    )


def shape_bar(**args):
    """Generates pin locations for a bar shape.

    Kwargs:
        width (float): bar width in mm (default: 1.).
        height (float): bar height in mm (default: 0.5).
        angle: bar angle in degrees (default: 0.).
        pins_per_mm (int): Pins per mm (default: 10).
        center (array): Location of stimulus center (default: [0.,0.]).
        hdiff (float): depth difference between center and edge (default: 0.).

    Returns:
        3D array of pin locations.
    """
    width = args.get('width',1.)
    height = args.get('height',.5)
    angle = np.deg2rad(args.get('angle',0.))
    pins_per_mm = args.get('pins_per_mm',default_params['pins_per_mm'])

    xy = np.mgrid[-width/2.:width/2.:width*pins_per_mm*1j,
        -height/2.:height/2.:height*pins_per_mm*1j]
    xy = xy.reshape(2,xy.shape[1]*xy.shape[2]).T
    xy = np.dot(np.array([[np.cos(angle),-np.sin(angle)],
        [np.sin(angle),np.cos(angle)]]),xy.T).T
    if 'hdiff' in args:
        d =  -args.get('hdiff')*(1/width*2*xy[:,0:1])**2
    else:
        d = np.zeros((xy.shape[0],1))
    d -= np.max(d)
    if 'center' in args:
        xy = xy + np.array(args.get('center'))
    xy = np.hstack((xy,d))
    return xy


def shape_circle(**args):
    """Generates pin locations for a circle.

    Kwargs:
        radius (float): circle radius in mm (default: 2.).
        pins_per_mm (int): Pins per mm (default: 10).
        curvature (float): Between 0 (flat) and 1 (sphere) (default: 0.).
        center (array): Location of stimulus center (default: [0.,0.]).

    Returns:
        3D array of pin locations.
    """

    radius = args.get('radius',2.)
    pins_per_mm = args.get('pins_per_mm',default_params['pins_per_mm'])

    xy = np.mgrid[-radius:radius:2*radius*pins_per_mm*1j,
        -radius:radius:2*radius*pins_per_mm*1j]
    xy = xy.reshape(2,xy.shape[1]*xy.shape[2]).T
    r = np.hypot(xy[:,0],xy[:,1])
    xy = xy[r<=radius]
    if 'hdiff' in args:
        r = r[r<=radius]
        d =  np.atleast_2d(-args.get('hdiff')*(1/radius*r)**2).T
    else:
        d = np.zeros((xy.shape[0],1))
    if 'center' in args:
        xy = xy + np.array(args.get('center'))
    xy = np.hstack((xy,d))
    return xy


def apply_ramp(trace,**args):
    """Applies on/off ramps to stimulus indentation trace.

    Args:
        trace (array): Indentation trace.

    Kwargs:
        ramp_len (float): length of on/off ramp in s or number of samples (if fs
            not set) (default: 0.05).
        fs (float): sampling frequency (default: None).
        ramp_type (str): 'lin' for linear, 'sin' for sine (default: 'lin').

    Returns:
        Nothing, original trace is modified in place.
    """
    ramp_len = args.get('ramp_len',.05)
    fs = args.get('fs',None)
    ramp_type = args.get('ramp_type','lin')
    if max(ramp_len,0.)==0.:
        return
    if fs is not None:
        ramp_len = round(ramp_len*fs)

    if ramp_type=='lin':    # apply ramp
        trace[:ramp_len] *= np.linspace(0,1,ramp_len)
        trace[-ramp_len:] *= np.linspace(1,0,ramp_len)
    elif ramp_type=='sin' or ramp_type=='sine':
        trace[:ramp_len] *= np.cos(np.linspace(np.pi,2.*np.pi,ramp_len))/2.+.5
        trace[-ramp_len:] *= np.cos(np.linspace(0.,np.pi,ramp_len))/2.+.5
    else:
        raise RuntimeError("ramp_type must be 'lin' or 'sin'")


def apply_pad(trace, **args):
    """Applies zero-padding to stimulus indentation trace.

    Args:
        trace (array): Indentation trace.

    Kwargs:
        pre_pad_len (float): length of padding before the trace in seconds or 
            number of samples (if fs not set) (default: 0.05).
        post_pad_len (float): length of padding after the trace in seconds or 
            number of samples (if fs not set) (default: 0.05).
        fs (float): sampling frequency (default: None).

    Returns:
        Padded trace (array).
    """
    pre_pad_len = args.get('pre_pad_len', 0.05)
    post_pad_len = args.get('post_pad_len', 0.05)
    fs = args.get('fs', None)

    if fs is not None:
        pre_pad_len = int(np.ceil(pre_pad_len * fs))
        post_pad_len = int(np.ceil(post_pad_len * fs))

    trace = np.concatenate((np.zeros(pre_pad_len), trace, np.zeros(post_pad_len)))
    return trace

