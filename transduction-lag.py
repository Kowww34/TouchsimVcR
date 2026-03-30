import numpy as np

from scipy import interpolate,signal
from numba import guvectorize,float64,boolean

from .constants import ihbasis
from .surface import hand_surface

def check_pin_radius(loc,rad):
    if loc.shape[0]>1:
        if loc.shape[0]>=3:
            dx = loc[:,0:1] - loc[:,0:1].T
            dy = loc[:,1:2] - loc[:,1:2].T
            dist = np.sqrt(dx**2. + dy**2)
            dist[dist==0] = np.nan
            return np.nanmin(dist)/2.
        elif loc.shape[0]==2:
            return np.sqrt(np.sum((loc[0]-loc[1])**2))/2.
    else:
        return rad

def skin_touch_profile(Z, xy, samp_freq, ProbeRad, sur=hand_surface, v=8000.0, E = 0.05, nu = 0.4):
    Z = Z.T # hack, needs to be fixed
    time_steps, pins = Z.shape
    dt = 1.0/samp_freq

    R = sur.distance(xy,xy) #pairwise didstance matrix
    K = R/v # time delay matrix
    shifts = np.rint(K/dt).astype(np.int64) #shift matrix

    # flat cylinder indenter solution from (SNEDDON 1946):
    np.seterr(all="ignore")
    D_Kernel = (1.-nu**2.)/np.pi/ProbeRad * np.arcsin(ProbeRad/R)/E
    np.seterr(all="warn")
    D_Kernel[R<=ProbeRad] = (1.-nu**2.)/2./ProbeRad/E

    negativeZ_mask = Z < 0
    absoluteZ = np.abs(Z)

    P = np.zeros((time_steps, pins)) #pressures
    #masterZ[i,j,:] = Z[:,j] shifted by shift[i,j], padded with the first shifted value
    masterZ = np.empty((pins, pins, time_steps), dtype=Z.dtype)
    from joblib import Parallel, delayed
    def solve_one_i(i):
        Zi = np.empty((time_steps, pins), dtype=absoluteZ.dtype)
        for j in range(pins):
            s = int(shifts[i, j])
            src = absoluteZ[:, j]
            if s <= 0:
                Zi[:, j] = src
            elif s >= time_steps:
                Zi[:, j] = src[0]
            else:
                pad_val = src[0]
                Zi[:s, j] = pad_val
                Zi[s:, j] = src[:-s]
        previousZi = np.zeros_like(Zi)
        temporaryP = np.zeros_like(Zi)
        count = 0
        while count == 0 or (temporaryP < 0).any():
            Zi[temporaryP < 0] = 0.0
            count += 1
            changed_rows = (np.sum(Zi - previousZi, axis=1) != 0.0)
            Zloc = Zi[changed_rows, :]
            temporaryP[changed_rows, :] = block_solve(Zloc, D_Kernel)
            previousZi = Zi.copy()
        return temporaryP[:, i]
    P_cols = Parallel(n_jobs=-1, prefer="threads")(delayed(solve_one_i)(i) for i in range(pins))
    P = np.column_stack(P_cols)  # (time_steps, pins)


    # correct for the hack
    P[negativeZ_mask] = -P[negativeZ_mask]

    # actual skin profile under the pins
    D = np.dot(P,D_Kernel)

    # time derivative of deflection profile
    # assumes same distribution of pressure as in static case
    # proposed by BYCROFT (1955) and confirmed by SCHMIDT (1981)
    if pins>1:
        # compute time derivative
        Dp = (np.r_[D[1:,:], np.nan*np.ones((1,D.shape[1]))] \
            - np.r_[np.nan*np.ones((1,D.shape[1])), D[0:-1,:]]) / 2. * samp_freq;
        Dp[0,:] = Dp[1,:]
        Dp[-1,:] = Dp[-2,:]
        # linsolve
        #Pdyn = np.linalg.solve(D,S1p.T)
        #Pdyn = np.linalg.solve(D + 1e-6 * np.eye(D.shape[0]), S1p.T)
        Pdyn = np.linalg.lstsq(D_Kernel, Dp.T, rcond=None)[0]
    else:
        Pdyn = np.zeros(P.shape);
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

def circ_load_vert_stress(P,PLoc,PRad,AffLoc,AffDepth):
    AffDepth = np.atleast_2d(np.array(AffDepth))
    nsamp,npin = P.shape
    nrec = AffLoc.shape[0]
    xy_dist = hand_surface.distance(AffLoc[:, :2], PLoc[:, :2])  # (nrec,npin)
    xy_dist = xy_dist.T  # (npin,nrec)
    z = np.dot(np.ones((npin, 1)), AffDepth)  # (npin,nrec)
    r = np.sqrt(xy_dist**2 + z**2)  # (npin,nrec)
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
    eps = P / (2.0 * PRad * PRad * np.pi)  # (nsamp,npin)
    #delaysettings
    v = 1.0  #distanceunitsper_sample(adjusttounitsandfs)
    #computeintegerdelaysamplesk[p,j]=round(r[p,j]/v)
    k = np.rint(r / v).astype(int)  # (npin,nrec)
    k = np.clip(k, 0, nsamp)  #avoidneg/overflow
    #accumulatewithperpin-perreceptorshift
    s_z = np.zeros((nsamp, nrec), dtype=eps.dtype)

    for p in range(npin):
        ep = eps[:, p]  # (nsamp,)

        for j in range(nrec):
            kj = int(k[p, j])
            w = K[p, j]

            if kj <= 0:
                #no delay
                s_z[:, j] += ep * w

            elif kj < nsamp:
                #backfill with the original (undelayed) contribution for early times
                s_z[:kj, j] += ep[:kj] * w

                #then apply delay (shift right): use earlier ep values
                s_z[kj:, j] += ep[:nsamp - kj] * w

            else:
                #delay larger than signal length: nothing arrives; keep original undelayed contribution
                s_z[:, j] += ep * w

    return s_z

def circ_load_dyn_wave(dynProfile,Ploc,PRad,Rloc,Rdepth,sfreq,sur):
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

def lif_neuron(aff,stimi,dstimi):
    srate = 5000. # fixed sampling frequency

    stimi = stimi.T
    dstimi = dstimi.T

    p = np.atleast_2d(aff.parameters)

    # Make basis for post-spike current
    ih = np.dot(p[:,10:12],ihbasis)

    uq,ia,ic = np.unique(np.atleast_2d(aff.gid),axis=0,
        return_index=True,return_inverse=True)
    for i in range(uq.shape[0]):
        bfilt,afilt = signal.butter(3,p[ia[i],0]*4./1000.)
        if uq[i,0]==0:
            stimi[ic==i] = signal.lfilter(bfilt,afilt,stimi[ic==i],axis=1)
        dstimi[ic==i] = signal.lfilter(bfilt,afilt,dstimi[ic==i],axis=1)

    Iinj = weight_inputs(p,stimi,dstimi)

    Vmem = np.zeros(Iinj.shape)
    Sp = lif_sub(Vmem,Iinj,ih,p,aff.noisy)

    spikes = []
    for i in range(len(aff)):
        spikes.append(np.flatnonzero(Sp[i])/srate + p[i,12]/1000. + 1./srate)

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
