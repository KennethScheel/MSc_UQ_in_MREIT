# ========================================================================
# Created by:
# Felipe Uribe @ DTU compute
# ========================================================================
# Version 2021
# ========================================================================
import numpy as np
import scipy as sp
import scipy.stats as sps
# eps = np.finfo(float).eps



# ===================================================================
def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= 4*n
    
    # Optionally normalize
    if norm:
        acf /= acf[0]
    return acf

# =========================================================================
def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i



# =========================================================================
# =========================================================================
# =========================================================================
def iact(y): 
    if len(y.shape) == 1:
        y = y[:, np.newaxis]
    N, nx = y.shape
    tau = np.zeros(nx)
    m = np.zeros(nx)
    #
    x = np.fft.fft(y, axis=0)
    xr, xi = np.real(x), np.imag(x)
    xr = xr**2 + xi**2 
    xr[0, :] = 0
    xr = np.real(np.fft.fft(xr, axis=0))
    var = xr[0, :]/N/(N-1)
    for j in range(nx):
        if var[j] == 0:
            pass
        xr[:, j] = xr[:, j]/xr[0, j]
        summ = -1/3
        for i in range(N):
            summ += xr[i, j]-1/6
            if summ < 0:
                tau[j] = 2*(summ + i/6)
                m[j] = i
                break
    return tau#, m



# ===================================================================
# ===================================================================
# ===================================================================
def Geweke(X, A = 0.1, B = 0.5):
    # Geweke's MCMC convergence diagnostic:
    # Test for equality of the means of the first A% (default 10%)  
    # and last B (50%) of a Markov chain
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    Ns, _ = X.shape
    nA = int(np.floor(A*Ns))
    nB = int(Ns - np.floor(B*Ns))
    
    # extract sub chains
    X_A, X_B = X[:nA, :], X[nB:, :]
    
    # compute the mean
    mean_X_A, mean_X_B = X_A.mean(axis=0), X_B.mean(axis=0)

    # Spectral estimates for variance
    var_X_A, var_X_B = spectrum0(X_A), spectrum0(X_B)

    # params of the geweke
    z = (mean_X_A - mean_X_B) / (np.sqrt((var_X_A/nA) + (var_X_B/(Ns-nB+1))))
    p = 2*(1-sps.norm.cdf(abs(z)))
    
    return z, p

# ===================================================================
def spectrum0(x):
    # Spectral density at frequency zero
    # Marko Laine <marko.laine@fmi.fi>
    m, n = x.shape
    s = np.empty(n)
    for i in range(n):
        spec, _ = spectrum(x[:, i], m) # check this later: sp.signal.welch(x[:, i])[0]
        s[i] = spec[0]

    return s

# ===================================================================
def spectrum(x, nfft):
    # Power spectral density using Hanning window
    # Marko Laine <marko.laine@fmi.fi>
    n = len(x)
    nw = int(np.fix(nfft/4))
    noverlap = int(np.fix(nw/2))
    if (n < nw):
        x[nw], n = 0, nw
        
    # Hanning window
    idx = np.arange(1, nw+1, 1)
    w = 0.5*(1 - np.cos(2*np.pi*idx/(nw+1))) # check this later: np.hanning(nw)
    
    # estimate PSD
    k = int(np.fix((n-noverlap)/(nw-noverlap)))    # number of windows
    kmu = k*np.linalg.norm(w)**2                   # normalizing scale factor
    y = np.zeros(nfft)
    for _ in range(k):
        xw = w*x[idx-1]
        idx += (nw - noverlap)
        Xx = abs(np.fft.fft(xw, nfft))**2
        y += Xx
    y = y*(1/kmu) # normalize
    
    n2 = int(np.floor(nfft/2))
    idx2 = np.arange(0, n2, 1)
    y = y[idx2]
    f = 1/(n*idx2)

    return y, f