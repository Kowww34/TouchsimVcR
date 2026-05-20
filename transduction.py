"""
TouchSim mechanical transduction and afferent spiking.

Converts pin indentation traces to surface pressure profiles, propagates
static/dynamic stress to afferent locations (Sneddon 1946), and simulates
LIF spiking. Three backends are available:

- ``orig``: full surface-graph coupling (reference).
- ``cutoff``: distance-limited static coupling and stress.
- ``lag``: time-delayed static solve and stress accumulation.

Units: mm, seconds, Hz unless noted. See README for cutoff/lag rationale.
"""

import numpy as np
from scipy import interpolate, signal
from numba import guvectorize, float64, boolean
from typing import Optional

from .constants import ihbasis
from .surface import hand_surface

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKENDS = ("orig", "cutoff", "lag")

PROPAGATION_CUTOFF_DISTANCE_MM = 90.0
STRESS_ACCUMULATION_RCUT_MM = 90.0


def normalize_backend(name: str) -> str:
    """Normalize backend name (``delay`` → ``lag``) and validate. Call once on ``Stimulus`` creation."""
    key = str(name).lower()
    if key == "delay":
        key = "lag"
    if key not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {name!r}")
    return key


def set_distance_cutoffs_mm(
    propagation_cutoff_mm: Optional[float] = None,
    stress_rcut_mm: Optional[float] = None,
) -> None:
    """Configure default propagation cutoffs for the cutoff backend (mm)."""
    global PROPAGATION_CUTOFF_DISTANCE_MM, STRESS_ACCUMULATION_RCUT_MM
    if propagation_cutoff_mm is not None:
        PROPAGATION_CUTOFF_DISTANCE_MM = float(propagation_cutoff_mm)
    if stress_rcut_mm is not None:
        STRESS_ACCUMULATION_RCUT_MM = float(stress_rcut_mm)
    elif propagation_cutoff_mm is not None:
        STRESS_ACCUMULATION_RCUT_MM = float(propagation_cutoff_mm)


# ---------------------------------------------------------------------------
# Shared pipeline (block solve, dynamic wave, LIF)
# ---------------------------------------------------------------------------

def block_solve(Z0,D):
    nz = Z0!=0
    # do clever packing to speed up unique_rows
    if nz.shape[1]<128:
        packed = np.packbits(nz,axis=1)
    else:
        nz_ext = nz
        add = nz.shape[1] % 64
        if add>0:
            nz_ext = np.concatenate((nz,
                np.zeros((nz.shape[0],64-add),dtype=np.bool_)),axis=1)
        packed = np.packbits(nz_ext,axis=1).view(np.uint64)
    # find similar lines to solve the linear system
    u,ia,ic = np.unique(packed,axis=0,return_index=True,return_inverse=True)
    unz = nz[ia,:] # unique non-zeros elements
    P = np.zeros(Z0.shape)
    for ii in range(0,ia.size):
        lines = ic==ii    # lines of this block
        nZi = unz[ii,:]   # non-zeros elements
        ixgrid = np.ix_(lines,nZi)
        nZigrid = np.ix_(nZi,nZi)
        P[ixgrid] = np.linalg.solve(D[nZigrid],Z0[ixgrid].T).T
    return P


def circ_load_dyn_wave(dynProfile, Ploc, PRad, Rloc, Rdepth, sfreq, sur):
    """Dynamic wave deflection at receptors (Sneddon decay + propagation delay)."""
    dr = sur.distance(Ploc, Rloc)
    rdel = dr - PRad
    rdel[rdel < 0.0] = 0.0
    delay = np.atleast_2d(rdel / 8000.0)
    np.seterr(all="ignore")
    decay = 1.0 / PRad / np.pi * np.arcsin(PRad / dr)
    np.seterr(all="warn")
    decay[dr <= PRad] = 1.0 / 2.0 / PRad
    udyn = add_delays(delay.T, decay.T, dynProfile, sfreq)
    udyn = udyn.T
    udyn = udyn / (Rdepth**2)
    return udyn


@guvectorize(
    [(float64[:], float64[:], float64[:, :], float64[:], float64[:])],
    "(m),(m),(m,n),()->(n)",
    nopython=True,
    target="parallel",
)
def add_delays(delay, decay, dynProfile, sfreq, udyn):
    for i in range(udyn.shape[0]):
        udyn[i] = 0
    for jj in range(dynProfile.shape[0]):
        delay_idx = int(np.rint(delay[jj] * sfreq[0]))
        if delay_idx > 0:
            for i in range(delay_idx, dynProfile.shape[1]):
                udyn[i] += dynProfile[jj, i - delay_idx] * decay[jj]
        else:
            for i in range(dynProfile.shape[1]):
                udyn[i] += dynProfile[jj, i] * decay[jj]


def lif_neuron(aff, stimi, dstimi):
    """Simulate LIF spiking for an afferent population."""
    srate = 5000.0
    stimi = stimi.T
    dstimi = dstimi.T
    p = np.atleast_2d(aff.parameters)
    ih = np.dot(p[:, 10:12], ihbasis)
    uq, ia, ic = np.unique(
        np.atleast_2d(aff.gid), axis=0, return_index=True, return_inverse=True
    )
    for group_idx in range(uq.shape[0]):
        cutoff_hz = p[ia[group_idx], 0] * 4.0 / 1000.0
        b, a = signal.butter(3, cutoff_hz)
        in_group = ic == group_idx
        if uq[group_idx, 0] == 0:
            stimi[in_group] = signal.lfilter(b, a, stimi[in_group], axis=1)
        dstimi[in_group] = signal.lfilter(b, a, dstimi[in_group], axis=1)
    Iinj = weight_inputs(p, stimi, dstimi)
    Vmem = np.zeros(Iinj.shape)
    Zp = lif_sub(Vmem, Iinj, ih, p, aff.noisy)
    spikes = []
    for i in range(len(aff)):
        spike_indices = np.flatnonzero(Zp[i])
        spike_times_s = spike_indices / srate + p[i, 12] / 1000.0 + 1.0 / srate
        spikes.append(spike_times_s)
    return spikes


@guvectorize(
    [(float64[:], float64[:], float64[:], float64[:])],
    "(m),(n),(n)->(n)",
    nopython=True,
    target="parallel",
)
def weight_inputs(p, stimi, dstimi, Iinj):
    for i in range(stimi.shape[0]):
        if np.sign(stimi[i]) >= 0:
            Iinj[i] = p[1] * stimi[i]
        else:
            Iinj[i] = -p[2] * stimi[i]
        if np.sign(dstimi[i]) >= 0:
            Iinj[i] += p[3] * dstimi[i]
        else:
            Iinj[i] += -p[4] * dstimi[i]
        ddstimi = dstimi[min(i + 1, stimi.shape[0] - 1)] - dstimi[i]
        if np.sign(ddstimi) >= 0:
            Iinj[i] += p[5] * ddstimi
        else:
            Iinj[i] += -p[6] * ddstimi


@guvectorize(
    [(float64[:], float64[:], float64[:], float64[:], boolean[:], float64[:])],
    "(n),(n),(m),(o),()->(n)",
    nopython=True,
    target="parallel",
)
def lif_sub(Vmem, Iinj, ih, p, noisy, Zp):
    if noisy[0]:
        Iinj += p[8] * np.random.standard_normal(Iinj.shape)
    tau = p[9]
    if p[7] > 0.0:
        Iinj = p[7] * Iinj / (p[7] + np.abs(Iinj))
        Iinj[np.isnan(Iinj)] = 0.0
    nh = ih.size
    ih_counter = nh
    for ii in range(Vmem.size):
        if ih_counter == nh:
            Vmem[ii] = Vmem[ii - 1] + (-(Vmem[ii - 1]) / tau + Iinj[ii])
        else:
            Vmem[ii] = Vmem[ii - 1] + (-(Vmem[ii - 1]) / tau + Iinj[ii] + ih[ih_counter])
            ih_counter += 1
        if Vmem[ii] > 1.0 and ih_counter > 5:
            Zp[ii] = 1
            Vmem[ii] = 0.0
            ih_counter = 0
        else:
            Zp[ii] = 0


def check_pin_radius(loc, rad):
    """
    Maximum pin radius that avoids overlap between pins.

    Returns half the minimum center-to-center spacing, or ``rad`` for a single pin.
    """
    num_pins = loc.shape[0]
    if num_pins == 1:
        return rad
    if num_pins == 2:
        return np.linalg.norm(loc[0] - loc[1]) / 2.0
    dx = loc[:, 0][:, None] - loc[:, 0][None, :]
    dy = loc[:, 1][:, None] - loc[:, 1][None, :]
    pairwise_distances = np.sqrt(dx**2 + dy**2)
    pairwise_distances[pairwise_distances == 0] = np.nan
    return np.nanmin(pairwise_distances) / 2.0


def _pin_pairwise_distance(xy, sur):
    """Pairwise pin distances on the hand surface graph."""
    return sur.distance(xy, xy)


def _sneddon_compliance_matrix(R, ProbeRad, E=0.05, nu=0.4):
    """Flat-cylinder Sneddon (1946) compliance matrix between pins."""
    np.seterr(all="ignore")
    D = (1.0 - nu**2) / (np.pi * ProbeRad) * np.arcsin(ProbeRad / R) / E
    np.seterr(all="warn")
    D[R <= ProbeRad] = (1.0 - nu**2) / (2.0 * ProbeRad * E)
    return D


def skin_touch_profile(
    Z0,
    xy,
    samp_freq,
    ProbeRad,
    backend="orig",
    sur=hand_surface,
    v=8000.0,
    E=0.05,
    nu=0.4,
    prop_cutoff=None,
):
    """Static/dynamic pin pressures from indentation (``backend`` set on ``Stimulus``)."""
    Z0 = Z0.T
    R = _pin_pairwise_distance(xy, sur)
    D = _sneddon_compliance_matrix(R, ProbeRad, E=E, nu=nu)

    if backend == "cutoff":
        pc = PROPAGATION_CUTOFF_DISTANCE_MM if prop_cutoff is None else float(prop_cutoff)
        D[R > pc] = 0.0

    if backend == "lag":
        from joblib import Parallel, delayed

        time_steps, pins = Z0.shape
        dt = 1.0 / samp_freq
        shifts = np.rint((R / v) / dt).astype(np.int64)
        negative_z_mask = Z0 < 0
        absolute_z = np.abs(Z0)
        absolute_dtype = absolute_z.dtype

        def solve_one_i(i):
            Zi = np.empty((time_steps, pins), dtype=absolute_dtype)
            for j in range(pins):
                shift = int(shifts[i, j])
                src = absolute_z[:, j]
                if shift <= 0:
                    Zi[:, j] = src
                elif shift >= time_steps:
                    Zi[:, j] = src[0]
                else:
                    pad_val = src[0]
                    Zi[:shift, j] = pad_val
                    Zi[shift:, j] = src[:-shift]
            previous_Zi = np.zeros_like(Zi)
            temporary_p = np.zeros_like(Zi)
            count = 0
            while count == 0 or (temporary_p < 0).any():
                Zi[temporary_p < 0] = 0.0
                count += 1
                changed_rows = np.sum(Zi - previous_Zi, axis=1) != 0.0
                zloc = Zi[changed_rows, :]
                temporary_p[changed_rows, :] = block_solve(zloc, D)
                previous_Zi = Zi.copy()
            return temporary_p[:, i]

        p = np.column_stack(
            Parallel(n_jobs=-1, prefer="threads")(
                delayed(solve_one_i)(i) for i in range(pins)
            )
        )
        p[negative_z_mask] = -p[negative_z_mask]
        deflection = np.dot(p, D)
        if pins > 1:
            dp = (
                np.r_[deflection[1:, :], np.nan * np.ones((1, deflection.shape[1]))]
                - np.r_[np.nan * np.ones((1, deflection.shape[1])), deflection[0:-1, :]]
            ) / 2.0 * samp_freq
            dp[0, :] = dp[1, :]
            dp[-1, :] = dp[-2, :]
            p_dyn = np.linalg.lstsq(D, dp.T, rcond=None)[0]
        else:
            p_dyn = np.zeros(p.shape)
        return p, p_dyn

    # orig and cutoff: iterative contact-detection (static + Bycroft dynamic)
    shape = Z0.shape
    Z0_negative = Z0 < 0
    abs_Z0 = np.abs(Z0)
    p = np.zeros(shape)
    prev_Z0 = np.zeros(shape)
    count = 0
    while count == 0 or p[p < 0].size > 0:
        abs_Z0[p < 0] = 0.0
        count += 1
        changed = np.sum(abs_Z0 - prev_Z0, axis=1) != 0.0
        Z0_loc = abs_Z0[changed, :]
        p[changed, :] = block_solve(Z0_loc, D)
        prev_Z0 = abs_Z0.copy()

    p[Z0_negative] = -p[Z0_negative]
    deflection = np.dot(p, D)

    if shape[0] > 1:
        deflection_p = (
            np.r_[deflection[1:, :], np.nan * np.ones((1, deflection.shape[1]))]
            - np.r_[np.nan * np.ones((1, deflection.shape[1])), deflection[0:-1, :]]
        ) / 2.0 * samp_freq
        deflection_p[0, :] = deflection_p[1, :]
        deflection_p[-1, :] = deflection_p[-2, :]
        p_dyn = np.linalg.lstsq(D, deflection_p.T, rcond=None)[0]
    else:
        p_dyn = np.zeros(p.shape)
    return p, p_dyn


def _sneddon_stress_kernel(r, z, PRad):
    """Sneddon (1946) vertical-stress kernel K between pins and afferents (npin, nrec)."""
    xsi = z / PRad
    rho = r / PRad
    rr = np.sqrt(1.0 + xsi**2)
    R = np.sqrt((rho**2 + xsi**2 - 1.0) ** 2 + 4.0 * xsi**2)
    theta = np.arctan(1.0 / xsi)
    phi = np.arctan2(2.0 * xsi, (rho**2 + xsi**2 - 1.0))
    j01 = np.sin(phi / 2.0) / np.sqrt(R)
    j02 = rr * np.sin(1.5 * phi - theta) / (R**1.5)
    return j01 + xsi * j02


def circ_load_vert_stress(
    P,
    PLoc,
    PRad,
    AffLoc,
    AffDepth,
    backend="orig",
    sur=hand_surface,
    rcut=None,
):
    """Vertical stress at afferents from pin loads (``backend`` set on ``Stimulus``)."""
    AffDepth = np.atleast_2d(np.array(AffDepth))
    nsamp, npin = P.shape
    nrec = AffLoc.shape[0]

    xy_dist = sur.distance(AffLoc[:, :2], PLoc[:, :2]).T  # (npin, nrec)
    z = np.dot(np.ones((npin, 1)), AffDepth)
    r = np.sqrt(xy_dist**2 + z**2)
    kernel = _sneddon_stress_kernel(r, z, PRad)
    eps = P / (2.0 * PRad * PRad * np.pi)

    if backend == "cutoff":
        rc = STRESS_ACCUMULATION_RCUT_MM if rcut is None else float(rcut)
        kernel = kernel * (r <= rc)

    if backend == "lag":
        delay_samples = np.clip(np.rint(r / 1.0).astype(int), 0, nsamp)
        s_z = np.zeros((nsamp, nrec), dtype=eps.dtype)
        for p in range(npin):
            ep = eps[:, p]
            for j in range(nrec):
                kj = int(delay_samples[p, j])
                w = kernel[p, j]
                if kj <= 0:
                    s_z[:, j] += ep * w
                elif kj < nsamp:
                    s_z[:kj, j] += ep[:kj] * w
                    s_z[kj:, j] += ep[: nsamp - kj] * w
                else:
                    s_z[:, j] += ep * w
        return s_z

    return np.dot(eps, kernel)
