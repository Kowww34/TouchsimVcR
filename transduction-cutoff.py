import numpy as np
from scipy import interpolate,signal
from numba import guvectorize,float64,boolean

from .constants import ihbasis
from .surface import hand_surface

def check_pin_radius(pin_xy, default_radius):
    """
    Compute the maximum allowable pin radius such that pins do not overlap.

    Parameters
    ----------
    pin_xy : (N, 2) ndarray
        Array of pin center locations in the xy-plane.
        Each row is [x, y] for one pin.

    default_radius : float
        Radius to return if only a single pin is present
        (no spacing constraint exists).

    Returns
    -------
    max_radius : float
        Maximum pin radius that avoids overlap between pins.
        Defined as half of the minimum center-to-center distance.
    """
    num_pins = pin_xy.shape[0]
    # If there is only one pin, spacing constraints do not apply
    if num_pins == 1:
        return default_radius
    # If there are exactly two pins, compute their separation directly
    if num_pins == 2:
        center_distance = np.linalg.norm(pin_xy[0] - pin_xy[1])
        return center_distance / 2.0
    # For three or more pins:
    # compute all pairwise center-to-center distances
    dx = pin_xy[:, 0][:, None] - pin_xy[:, 0][None, :]
    dy = pin_xy[:, 1][:, None] - pin_xy[:, 1][None, :]
    pairwise_distances = np.sqrt(dx**2 + dy**2)
    # Ignore self-distances on the diagonal
    pairwise_distances[pairwise_distances == 0] = np.nan
    # The limiting radius is half the minimum distance
    min_spacing = np.nanmin(pairwise_distances)
    return min_spacing / 2.0


def skin_touch_profile(
    S0,
    xy,
    samp_freq,
    ProbeRad,
    sur=hand_surface,
    v=8000.0,
    E=0.05,
    nu=0.4,
    prop_cutoff=50.0,  # (mm) static pressure propagation cutoff radius
):
    S0 = S0.T # hack, needs to be fixed
    s = S0.shape

    R = sur.distance(xy, xy) #pairwise distance matrix

    # flat cylinder indenter solution from (SNEDDON 1946):
    np.seterr(all="ignore")
    D = (1.0 - nu**2.0) / (np.pi * ProbeRad) * np.arcsin(ProbeRad / R) / E
    np.seterr(all="warn")
    D[R <= ProbeRad] = (1.0 - nu**2.0) / (2.0 * ProbeRad * E)

    # cutoff: no propagation beyond prop_cutoff (same units as xy/R, typically mm)
    if prop_cutoff is not None:
        D[R > prop_cutoff] = 0.0

    S0neg = S0 < 0
    absS0 = np.abs(S0)

    P = np.zeros(s)
    prevS0 = np.zeros(s)
    count = 0

    # iterative contact-detection algorithm
    while count == 0 or P[P < 0].size > 0:
        absS0[P < 0] = 0.0
        count += 1

        # only work on changed (and nonzeros) line
        diffl = np.sum(absS0 - prevS0, axis=1) != 0.0
        S0loc = absS0[diffl, :]
        P[diffl, :] = block_solve(S0loc, D)
        prevS0 = absS0.copy()

    # correct for the hack
    P[S0neg] = -P[S0neg]

    # actual skin profile under the pins
    S1 = np.dot(P, D)

    # time derivative of deflection profile
    # assumes same distribution of pressure as in static case
    # proposed by BYCROFT (1955) and confirmed by SCHMIDT (1981)
    if s[0] > 1:
        S1p = (np.r_[S1[1:, :], np.nan * np.ones((1, S1.shape[1]))]
               - np.r_[np.nan * np.ones((1, S1.shape[1])), S1[0:-1, :]]) / 2.0 * samp_freq
        S1p[0, :] = S1p[1, :]
        S1p[-1, :] = S1p[-2, :]

        Pdyn = np.linalg.lstsq(D, S1p.T, rcond=None)[0]
    else:
        Pdyn = np.zeros(P.shape)

    return P, Pdyn


def block_solve(S0,D):
    nz = S0!=0
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
    P = np.zeros(S0.shape)
    for ii in range(0,ia.size):
        lines = ic==ii    # lines of this block
        nzi = unz[ii,:]   # non-zeros elements
        ixgrid = np.ix_(lines,nzi)
        nzigrid = np.ix_(nzi,nzi)
        P[ixgrid] = np.linalg.solve(D[nzigrid],S0[ixgrid].T).T
    return P



def circ_load_vert_stress(P, PLoc, PRad, AffLoc, AffDepth, sur=hand_surface, rcut=50.0):
    """
    Compute the vertical (normal) stress at afferent locations caused by
    circular pressure loads applied at pin locations.
    This implements the axisymmetric elastic solution for a uniformly
    loaded circular area (Sneddon-type formulation), evaluated at
    subsurface receptor locations.
    Parameters
    ----------
    P : (N_time, N_pins) ndarray
        Pressure (or force density) applied at each pin as a function of time.
        Each row corresponds to one time sample.
        Each column corresponds to one pin.
    PLoc : (N_pins, 2) ndarray
        xy-locations of the pin centers (load application points).
    PRad : float
        Radius of the circular contact area for each pin (assumed identical).
    AffLoc : (N_receptors, 2) ndarray
        xy-locations of afferent (receptor) positions where stress is evaluated.
    AffDepth : float or array-like of length N_receptors
        Depth(s) of afferents below the surface (z > 0).
        Can be a scalar (same depth for all receptors) or per-receptor values.
    Returns
    -------
    s_z : (N_time, N_receptors) ndarray
        Vertical (z-direction) stress at each afferent location as a function
        of time, obtained by summing contributions from all pins.
    """
    AffDepth = np.atleast_2d(np.array(AffDepth))
    nsamp, npin = P.shape
    nrec = AffLoc.shape[0]

    xy_dist = hand_surface.distance(AffLoc[:, :2], PLoc[:, :2])  # (nrec,npin)
    xy_dist = xy_dist.T  # (npin,nrec)
    z = np.dot(np.ones((npin, 1)), AffDepth)  # (npin,nrec)
    r = np.sqrt(xy_dist**2 + z**2)  # (npin,nrec)
    keep = (r <= rcut)  # (npin,nrec)

    #sneddon1946kernel
    XSI = z / PRad
    RHO = r / PRad
    rr = np.sqrt(1.0 + XSI**2)
    R = np.sqrt((RHO**2 + XSI**2 - 1.0)**2 + 4.0 * XSI**2)
    theta = np.arctan(1.0 / XSI)
    phi = np.arctan2(2.0 * XSI, (RHO**2 + XSI**2 - 1.0))
    J01 = np.sin(phi / 2.0) / np.sqrt(R)
    J02 = rr * np.sin(1.5 * phi - theta) / (R**1.5)
    K = (J01 + XSI * J02)  # (npin,nrec)
    K = K * keep  #hardcutoff
    eps = P / (2.0 * PRad * PRad * np.pi)  # (nsamp,npin)
    s_z = np.dot(eps, K)  # (nsamp,nrec)
    return s_z

def circ_load_dyn_wave(dynProfile,Ploc,PRad,Rloc,Rdepth,sfreq,sur):
    """
    Compute a "dynamic wave" deflection signal at receptor locations produced by
    time-varying (dynamic) pin deflection/pressure profiles, using a simple
    propagation + attenuation model.
    This function does three main things:
      1) Compute horizontal distances from each pin to each receptor.
      2) Convert distance into a propagation delay (assuming constant wave speed).
      3) Apply a distance-based amplitude decay (Sneddon-style radial decay),
         then apply an additional depth decay ~ 1/z^2.
    Parameters
    ----------
    dynProfile : (N_pins, N_time) ndarray
        Dynamic profile per pin over time (e.g., "dynamic pressure" or "dynamic deflection"
        term). Each row corresponds to one pin; each column is a time sample.
    Ploc : (N_pins, 2) ndarray
        xy-locations of the pin centers (source locations).
    PRad : float
        Pin/contact radius (used for both the delay offset and Sneddon-style decay).
    Rloc : (N_receptors, 2) ndarray
        xy-locations of receptors (receiver locations).
    Rdepth : float or array-like
        Receptor depth(s) below the surface. Used as an additional attenuation:
            amplitude ∝ 1 / (depth^2)
        If this is per-receptor, it should broadcast correctly against the output.
    sfreq : float
        Sampling frequency (Hz). Used by add_delays to convert delays (seconds)
        into sample shifts.
    sur : object
        Surface/geometry helper that provides:
            sur.distance(Ploc, Rloc) -> distances
        Expected output shape is (N_pins, N_receptors) or something compatible.
    Returns
    -------
    udyn : (N_receptors, N_time) ndarray
        Dynamic deflection signal at each receptor over time after applying delays
        and attenuation and depth scaling.
    """
    nsamp = dynProfile.shape[1]
    npin = dynProfile.shape[0]
    dr = sur.distance(Ploc,Rloc)
    # delay (everything is synchronous under the probe)
    rdel = dr-PRad
    rdel[rdel<0.] = 0.
    delay = np.atleast_2d(rdel/8000.) # 8000 is the wave velocity in mm/s
    # decay (=skin deflection decay given by Sneddon 1946)
    np.seterr(all="ignore")
    decay = 1./PRad/np.pi*np.arcsin(PRad/dr)
    np.seterr(all="warn")
    decay[dr<=PRad] = 1./2./PRad
    udyn = add_delays(delay.T,decay.T,dynProfile,sfreq)
    udyn = udyn.T
    # z decay is 1/z^2
    udyn = udyn / (Rdepth**2)
    return udyn



@guvectorize([(float64[:],float64[:],float64[:,:],float64[:],float64[:])],
    '(m),(m),(m,n),()->(n)',nopython=True,target='parallel')
def add_delays(delay,decay,dynProfile,sfreq,udyn):
    for i in range(udyn.shape[0]):
        udyn[i] = 0
    for jj in range(dynProfile.shape[0]):
        delay_idx = int(np.rint(delay[jj]*sfreq[0]))
        if delay_idx>0:
            for i in range(delay_idx,dynProfile.shape[1]):
                udyn[i] += dynProfile[jj,i-delay_idx]*decay[jj]
        else:
            for i in range(dynProfile.shape[1]):
                udyn[i] += dynProfile[jj,i]*decay[jj]

def lif_neuron(aff, stimi, dstimi):
    """
    Simulate a leaky integrate-and-fire (LIF) neuron model for an afferent population.
    This function takes stimulus-drive signals (and their time-derivative drive),
    applies afferent-type-specific lowpass filtering, computes injected current,
    runs the LIF dynamics, then converts threshold crossings into spike times.

    Parameters
    ----------
    aff : object
        Afferent population-like object that supports:
          - len(aff): number of afferents
          - aff.parameters: array-like, per-afferent parameter matrix
          - aff.gid: array-like, group/type identifiers per afferent
          - aff.noisy: bool or array-like controlling noise in lif_sub
        Typical TouchSim-style "affpop" object.
        Expected parameter usage (by index; inferred from code):
          - p[:, 0]   : cutoff frequency parameter used for Butterworth filtering
          - p[:,10:12]: coefficients used to build post-spike current filter (ih)
          - p[:,12]   : spike time offset (ms) applied after simulation
    stimi : ndarray
        Stimulus-derived drive signal per afferent over time.
        This code transposes the input immediately, so the expected incoming shape
        is likely (N_time, N_aff) and it becomes (N_aff, N_time) internally.
        After transpose, indexing assumes:
            stimi[i, t] is the drive for afferent i at time sample t.
    dstimi : ndarray
        Time-derivative (or dynamic) drive signal per afferent over time.
        Same shape convention as stimi.
    Returns
    -------
    spikes : list of 1D ndarrays
        A list of length N_aff, where each entry is an array of spike times (seconds)
        for that afferent.
    """
    #fixed simulation sampling frequency (Hz)
    srate = 5000.0
    #normalize to internal shape (N_aff, N_time)
    stimi = stimi.T
    dstimi = dstimi.T
    #per-afferent parameter matrix (ensure 2D)
    p = np.atleast_2d(aff.parameters)
    #build post-spike current basis/filter per afferent
    #ihbasis is assumed to be a global basis matrix used by the model
    ih = np.dot(p[:, 10:12], ihbasis)
    #group afferents by gid so we only design filters once per group
    #uq: unique gids
    #ia: indices of first occurrence of each unique gid (for grabbing representative parameters)
    #ic: group assignment per afferent (maps each afferent -> index into uq)
    uq, ia, ic = np.unique(
        np.atleast_2d(aff.gid),
        axis=0,
        return_index=True,
        return_inverse=True
    )
    #apply group-specific lowpass filtering to stimulus drives
    for group_idx in range(uq.shape[0]):
        cutoff_hz = p[ia[group_idx], 0] * 4.0 / 1000.0
        b, a = signal.butter(3, cutoff_hz)
        #which afferents are in this group
        in_group = (ic == group_idx)
        #domain-specific behavior: group id 0 gets stimi filtered too
        if uq[group_idx, 0] == 0:
            stimi[in_group] = signal.lfilter(b, a, stimi[in_group], axis=1)
        #all groups get dstimi filtered
        dstimi[in_group] = signal.lfilter(b, a, dstimi[in_group], axis=1)
    #convert filtered drives into injected current for the LIF model
    Iinj = weight_inputs(p, stimi, dstimi)
    #run LIF dynamics
    Vmem = np.zeros(Iinj.shape)
    Sp = lif_sub(Vmem, Iinj, ih, p, aff.noisy)
    #convert spike raster to spike times (seconds)
    spikes = []
    for i in range(len(aff)):
        spike_indices = np.flatnonzero(Sp[i])
        spike_times_s = spike_indices / srate + p[i, 12] / 1000.0 + 1.0 / srate
        spikes.append(spike_times_s)
    return spikes


@guvectorize([(float64[:],float64[:],float64[:],float64[:])],
    '(m),(n),(n)->(n)',nopython=True,target='parallel')
def weight_inputs(p,stimi,dstimi,Iinj):
    for i in range(stimi.shape[0]):
        if np.sign(stimi[i])>=0:
            Iinj[i]  = p[1]*stimi[i]
        else:
            Iinj[i]  = -p[2]*stimi[i]

        if np.sign(dstimi[i])>=0:
            Iinj[i]  += p[3]*dstimi[i]
        else:
            Iinj[i]  += -p[4]*dstimi[i]

        ddstimi = (dstimi[min(i+1,stimi.shape[0]-1)]-dstimi[i])
        if np.sign(ddstimi)>=0:
            Iinj[i]  += p[5]*ddstimi
        else:
            Iinj[i]  += -p[6]*ddstimi

@guvectorize([(float64[:],float64[:],float64[:],float64[:],boolean[:],
    float64[:])],'(n),(n),(m),(o),()->(n)',nopython=True,target='parallel')
def lif_sub(Vmem,Iinj,ih,p,noisy,Sp):
    if noisy[0]:
        Iinj += p[8]*np.random.standard_normal(Iinj.shape)

    tau = p[9]
    if p[7]>0.:
        Iinj = p[7]*Iinj/(p[7]+np.abs(Iinj))
        Iinj[np.isnan(Iinj)] = 0.

    nh = ih.size
    ih_counter = nh
    for ii in range(Vmem.size):

        if ih_counter==nh:
            Vmem[ii] =  Vmem[ii-1] + (-(Vmem[ii-1])/tau + Iinj[ii])
        else:
            Vmem[ii] =  Vmem[ii-1] + (-(Vmem[ii-1])/tau + Iinj[ii] + ih[ih_counter])
            ih_counter += 1

        if Vmem[ii]>1. and ih_counter>5:
            Sp[ii] = 1
            Vmem[ii] = 0.
            ih_counter = 0
        else:
            Sp[ii] = 0
