"""Spike phase and m:n phase-locking / synchrony indices."""

import numpy as np


def spike_phase(zz, t):
    """
    Compute continuous (unwrapped) spike phase for each neuron.

    Linearly interpolates phase between consecutive spikes (2*pi per ISI).
    Neurons with fewer than 2 spikes are omitted.

    Parameters
    ----------
    zz : list of 1D arrays
        Spike times per neuron.
    t : 1D array
        Times at which to evaluate phase [s].

    Returns
    -------
    phi : ndarray, shape (n_valid, len(t))
        Unwrapped phase [rad]; NaN outside spiking support.
    fr : ndarray, shape (n_valid,)
        Mean firing rate [Hz] over spiking support.
    """
    t = np.asarray(t, float)
    out_phi = []
    out_fr = []

    for s in zz:
        s = np.asarray(s, float)
        if s.size < 2:
            continue

        dt_support = s[-1] - s[0]
        if dt_support > 0:
            fr = (s.size - 1) / dt_support
        else:
            fr = np.nan

        idx = np.searchsorted(s, t, side="right") - 1
        valid = (idx >= 0) & (idx < s.size - 1)
        phi = np.full(t.shape, np.nan, dtype=float)

        isi = s[idx[valid] + 1] - s[idx[valid]]
        isi_safe = np.where(isi > 0, isi, np.nan)
        frac = (t[valid] - s[idx[valid]]) / isi_safe
        phi[valid] = 2 * np.pi * (idx[valid] + frac)

        out_phi.append(phi)
        out_fr.append(fr)

    if not out_phi:
        return np.empty((0, t.size), float), np.empty((0,), float)

    return np.vstack(out_phi), np.asarray(out_fr, float)
def sync_index(phi, theta, m_max=100, n_max=100):
    """
    Compute per-neuron synchrony index, maximizing over m:n phase locking.

    This function measures how well each neuron's firing is phase-locked
    to an external stimulus. It searches over different m:n ratios to find
    the best match.

    Theory
    ------
    The synchrony index (vector strength) for m:n locking is:

        R_{m:n} = |< exp(i * (n*theta - m*phi)) >|

    where <...> denotes time average, theta is stimulus phase, phi is neural
    phase, and i is the imaginary unit.

    - If m*phi = n*theta + constant, then R = 1 (perfect locking)
    - If phases are random, R -> 0

    We search over all (m,n) pairs up to (m_max, n_max) and return the
    maximum R value.

    Parameters
    ----------
    phi : ndarray, shape (n_neurons, n_times)
        Per-neuron unwrapped spike phases [radians].
    theta : ndarray, shape (n_times,)
        Unwrapped stimulus phase [radians].
    m_max, n_max : int
        Maximum values for m and n in the m:n locking search.

    Returns
    -------
    Rmax : ndarray, shape (n_neurons,)
        Maximum synchrony index for each neuron.
    m1 : ndarray of int, shape (n_neurons,)
        Optimal m value for each neuron (1-based).
    n1 : ndarray of int, shape (n_neurons,)
        Optimal n value for each neuron (1-based).
    """
    # Validate inputs
    try:
        m_max = int(m_max)
        n_max = int(n_max)
    except Exception:
        raise TypeError("m_max and n_max must be integer-like scalars.")
    if m_max < 1 or n_max < 1:
        raise ValueError("m_max and n_max must be >= 1.")

    theta_valid = np.isfinite(theta)
    n_aff, nt = phi.shape

    # ----- Precompute stimulus harmonics -----
    # E_theta[n, t] = exp(i * n * theta[t])
    # This is computed once and reused for all neurons
    ns = np.arange(1, n_max + 1, dtype=int)[:, None]  # (n_max, 1)
    Etheta = np.exp(1j * ns * theta[None, :])          # (n_max, nt)

    # Initialize output arrays
    Rmax = np.zeros(n_aff, dtype=float)
    m1 = np.ones(n_aff, dtype=int)
    n1 = np.ones(n_aff, dtype=int)

    ms = np.arange(1, m_max + 1, dtype=int)[:, None]   # (m_max, 1)

    # ----- Loop over neurons -----
    for i in range(n_aff):
        # Mask for valid time points (both phi and theta finite)
        mask = np.isfinite(phi[i]) & theta_valid
        count = int(mask.sum())

        if count == 0:
            Rmax[i] = 0.0
            m1[i] = 1
            n1[i] = 1
            continue

        # P[m, t] = exp(-i * m * phi[i, t])
        P = np.exp(-1j * ms * phi[i][None, :])  # (m_max, nt)
        P[:, ~mask] = 0.0  # Zero out invalid times

        # Z[n, m] = (1/count) * sum_t E_theta[n,t] * P[m,t]
        # This is the synchrony measure for each (m,n) pair
        Z = (Etheta @ P.T) / count  # (n_max, m_max)
        R = np.abs(Z)

        # Find maximum
        k = int(R.argmax())
        n_idx, m_idx = np.unravel_index(k, R.shape)
        Rmax[i] = R[n_idx, m_idx]
        n1[i] = n_idx + 1  # Convert to 1-based
        m1[i] = m_idx + 1

    return Rmax, m1, n1


# =============================================================================
# POPULATION SYNCHRONY INDEX
# =============================================================================

def sync_index_all(phi, theta, m_max=100, n_max=100):
    """
    Population-level synchrony index with equal neuron weighting.

    This function computes a single synchrony value for the entire neural
    population by averaging across neurons with equal weights.

    Parameters
    ----------
    phi : ndarray, shape (n_neurons, n_times)
        Unwrapped spike phases [radians], NaN where invalid.
    theta : ndarray, shape (n_times,)
        Unwrapped stimulus phase [radians].
    m_max, n_max : int
        Search bounds for m:n locking ratio.

    Returns
    -------
    Rmax : float
        Maximum population synchrony.
    m_best, n_best : int
        Optimal m and n (1-based).
    """
    phi = np.asarray(phi, float)
    theta = np.asarray(theta, float)
    n_aff, nt = phi.shape

    theta_valid = np.isfinite(theta)

    # Precompute stimulus harmonics
    ns = np.arange(1, n_max + 1)[:, None]
    Etheta = np.exp(1j * ns * theta[None, :])  # (n_max, nt)

    # Per-neuron masks & counts
    valid = np.isfinite(phi) & theta_valid[None, :]  # (n_aff, nt)
    counts = valid.sum(axis=1)  # (n_aff,)
    keep = counts > 0

    if not np.all(keep):
        valid = valid[keep]
        phi = phi[keep]
        counts = counts[keep]
        n_aff = keep.sum()
        if n_aff == 0:
            return 0.0, 1, 1

    inv_counts = 1.0 / counts  # (n_aff,)

    Rmax = 0.0
    m_best = n_best = 1

    # ----- Search over m values -----
    for m in range(1, m_max + 1):
        V = np.exp(-1j * m * phi)  # (n_aff, nt)
        V[~valid] = 0.0  # Zero out invalid times

        # Z(n,i) = (1/count_i) * sum_t Etheta[n,t] * V[i,t]
        Z = Etheta @ V.T  # (n_max, n_aff)
        Z *= inv_counts[None, :]  # Normalize per neuron

        # Population average (equal weights across neurons)
        Zpop = Z.mean(axis=1)  # (n_max,)
        Rn = np.abs(Zpop)

        k = int(np.argmax(Rn))
        if Rn[k] > Rmax:
            Rmax = float(Rn[k])
            n_best = k + 1
            m_best = m

    return Rmax, m_best, n_best


# =============================================================================
# CROSS-NEURON SYNCHRONY (WITH STIMULUS)
# =============================================================================

def cross_neuron_sync_max(phi, theta, m_max=100, n_max=100, min_neurons=2):
    """
    Cross-neuron synchrony index maximizing over (m,n) phase locking.

    This measures how synchronized neurons are WITH EACH OTHER, while also
    considering their relationship to an external stimulus phase.

    Method
    ------
    1. At each time t, compute the mean phase vector across neurons:
           Q(t) = (1/M_t) * sum_i exp(i * m * phi_i(t))
       where M_t is the number of valid neurons at time t.

    2. Compute the synchrony with stimulus phase:
           S_{m:n} = | (1/T) * sum_t exp(-i * n * theta(t)) * Q(t) |

    3. Search over (m, n) to find maximum S.

    This captures both:
    - Inter-neuron synchrony (through the mean phase vector Q)
    - Neural-stimulus synchrony (through the coupling with theta)

    Parameters
    ----------
    phi : ndarray, shape (n_neurons, n_times)
        Per-neuron unwrapped spike phases [radians]; NaN where undefined.
    theta : ndarray, shape (n_times,)
        Unwrapped stimulus phase [radians]; finite where valid.
    m_max, n_max : int
        Search bounds for m and n (positive integers).
    min_neurons : int
        Minimum number of valid neurons required at a time point to include it.

    Returns
    -------
    Smax : float
        Maximum cross-neuron synchrony value over (m,n).
    m_best, n_best : int
        Optimal m and n (1-based).
    """
    # ----- Input validation -----
    phi = np.asarray(phi, float)     # (n_aff, nt)
    theta = np.asarray(theta, float)  # (nt,)
    n_aff, nt = phi.shape

    m_max = int(m_max)
    n_max = int(n_max)
    if m_max < 1 or n_max < 1:
        raise ValueError("m_max and n_max must be >= 1")

    # ----- Build validity masks -----
    theta_ok = np.isfinite(theta)
    valid = np.isfinite(phi) & theta_ok[None, :]  # Per-neuron & theta validity
    n_valid_t = valid.sum(axis=0)  # Number of valid neurons at each time
    time_mask = n_valid_t >= int(min_neurons)  # Require enough neurons

    if not np.any(time_mask):
        return 0.0, 1, 1

    # ----- Keep only valid time samples -----
    theta_use = theta[time_mask]
    valid_use = valid[:, time_mask]
    phi_use = phi[:, time_mask]
    n_valid_use = n_valid_t[time_mask].astype(float)  # (T_use,)

    T_use = theta_use.size

    # ----- Precompute stimulus harmonics -----
    # Etheta_neg[n, t] = exp(-i * n * theta[t])
    ns = np.arange(1, n_max + 1)[:, None]                    # (n_max, 1)
    Etheta_neg = np.exp(-1j * ns * theta_use[None, :])       # (n_max, T_use)

    # ----- Search over m values -----
    Smax = 0.0
    m_best = n_best = 1

    for m in range(1, m_max + 1):
        # U[i,t] = exp(i * m * phi_i(t))
        U = np.exp(1j * m * phi_use)  # (n_aff, T_use)
        U[~valid_use] = 0.0  # Zero out invalid entries

        # Q[t] = across-neuron mean at each time
        sumU = U.sum(axis=0)  # (T_use,)
        Q = np.zeros(T_use, dtype=np.complex128)
        ok = n_valid_use > 0
        Q[ok] = sumU[ok] / n_valid_use[ok]

        # S_{m:n} = | (1/T_use) * sum_t Etheta_neg[n,t] * Q[t] |
        v = (Etheta_neg @ Q) / T_use  # (n_max,)
        Rn = np.abs(v)

        k = int(np.argmax(Rn))
        if Rn[k] > Smax:
            Smax = float(Rn[k])
            n_best = k + 1
            m_best = m

    return Smax, m_best, n_best


# =============================================================================
# CROSS-NEURON SYNCHRONY (WITHOUT STIMULUS)
# =============================================================================

def cross_neuron_sync_nostim(phi, min_neurons=2, m_max=50):
    """
    Cross-neuron synchrony WITHOUT external stimulus reference.

    This measures intrinsic synchrony between neurons, independent of any
    external driving signal. It answers: "Are the neurons synchronized with
    each other?"

    Method
    ------
    At each time t, compute the mean phase vector:
        Q(t) = (1/M_t) * sum_i exp(i * m * phi_i(t))

    The magnitude |Q(t)| measures how aligned the neurons' phases are:
    - |Q| = 1 means all neurons have the same phase (perfect sync)
    - |Q| = 0 means phases are uniformly distributed (no sync)

    The synchrony index is the time-average of |Q(t)|.

    Parameters
    ----------
    phi : ndarray, shape (n_neurons, n_times)
        Per-neuron unwrapped spike phases [radians]; NaN where undefined.
    min_neurons : int
        Minimum valid neurons per time point.
    m_max : int
        Search bound for phase multiplier m.

    Returns
    -------
    Sbest : float
        Best synchrony value (between 0 and 1).
    m_best : int
        Optimal m (1-based).

    Notes
    -----
    The parameter m allows detection of different types of synchrony:
    - m=1: Standard phase synchrony
    - m=2: Neurons fire at opposite phases but same frequency (anti-phase)
    - m>2: Higher-order phase relationships
    """
    phi = np.asarray(phi, float)  # (n_aff, nt)
    valid = np.isfinite(phi)
    n_aff, nt = phi.shape

    # Count valid neurons at each time
    n_valid_t = valid.sum(axis=0)
    time_mask = n_valid_t >= int(min_neurons)

    if not np.any(time_mask):
        return 0.0, 1

    # Extract valid time samples
    phi_use = phi[:, time_mask]
    valid_use = valid[:, time_mask]
    n_valid_use = n_valid_t[time_mask].astype(float)
    T_use = phi_use.shape[1]

    # ----- Search over m values -----
    Sbest, m_best = 0.0, 1

    for m in range(1, m_max + 1):
        # U[i,t] = exp(i * m * phi_i(t))
        U = np.exp(1j * m * phi_use)
        U[~valid_use] = 0.0

        # Mean phase vector at each time
        meanU = np.zeros(T_use, dtype=np.complex128)
        ok = n_valid_use > 0
        meanU[ok] = U[:, ok].sum(axis=0) / n_valid_use[ok]

        # Synchrony = time-average of |meanU|
        Sm = np.abs(meanU).mean()

        if Sm > Sbest:
            Sbest, m_best = float(Sm), m

    return Sbest, m_best
