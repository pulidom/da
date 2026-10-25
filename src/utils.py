#!/usr/bin/python
''' 
 Mathematical codes
 Matrix algebra

 @author: Manuel Pulido
'''
import numpy as np
import scipy.linalg as sla

def RMSE(E):
  ''' Definida indep de x y t, de esta manera hay mas sensibilidad a los extremos '''
  return np.sqrt(np.mean(np.square(E)))

def spread(X):
  " X[nt,nx,npa] "
  return np.sqrt(np.mean(X.var(axis=-1, ddof=1)))

def tarmse(xdif,ind=None):
  " Time averaged state RMSE "
  if ind is None: ind=range(xdif.shape[0])
  return np.mean(np.sqrt(np.mean((xdif[ind,:])**2,0)))

def is_diagonal(M):
  " Check if the matrix is diagonal "
  return np.all(M == np.diag(np.diagonal(M)))
#  return np.allclose(M, np.diag(np.diagonal(M)))

def check_symmetric(A, rtol=1e-05, atol=1e-08):
    return np.allclose(A, A.T, rtol=rtol, atol=atol)

def mean(x,axis): # aesthetic reasons
  " Media que mantiene las dimensiones "
  return np.mean(x,axis=axis,keepdims=True)

#------------------------------------------------
# Covariance
#------------------------------------------------
def cova(X,Y):
  "Determine the sample cov of two fields dim(state,ens members)"

  [nx,nem]=X.shape

  xm=np.mean(X,1)
  ym=np.mean(Y,1)

  dX=X-np.reshape(xm,[nx,1])
  dY=Y-np.reshape(ym,[nx,1])

  covXY=np.dot(dX,dY.T)/(nem-1.)

  return covXY
  
def meanper(X):
  " Compute mean and perturbation X[nx,nem]"
  xm = X.mean(1, keepdims=True)#[...,np.newaxis]        
  return xm,X-xm
  
def PaWa(B):
    "B: (..., k, k) symmetric positive definite. Returns B^-1 and sqrt(k-1) B^(-1/2) for ETFK."
    s, V = np.linalg.eigh(B)
    k = B.shape[-1]
    Vt = V.swapaxes(-1, -2)
    Pa = (V / s[..., None, :]) @ Vt
    Wa = (V * (np.sqrt(k - 1) / np.sqrt(s))[..., None, :]) @ Vt
    return Pa, Wa
  
#--------------------------------------------
def likelihood(xf_t, y_t, H, sqR):
    "innovation log likelihood"
    
    ncy = xf_t.shape[2]

    xmean = np.mean(xf_t, 1)
    ny=np.size(sqR)

    l = 0
    for it in range(1,ncy,1):
        d = y_t[:, it] - H.dot(xmean[:, it])
        hxf = H.dot(xf_t[:, :, it])
        
        hxf = np.cov(hxf) + np.dot(sqR,sqR.T)
        if ny > 1:
            hxfinv = np.linalg.pinv(hxf)
            hxfdet = np.linalg.det(hxf)
        else: # better to define always as matrices
            hxfinv=1./hxf
            hxfdet = hxf

        d1= np.dot(hxfinv,d)
        l -= .5 * np.log(hxfdet)
        l -= .5 * np.dot(d.T,d1)

    return l

#------------------------------------------------------------------------------
def obs_points(Nx, Ny, mode='uniform', rng=None):
    """
    mode='uniform' : evenly spaced centers around the ring (deterministic)
    mode='random'  : N_y centers drawn without replacement (needs rng)
    """
    if mode == 'uniform':
        obs_indices = np.linspace(0, Nx, Ny, endpoint=False).astype(int)
    elif mode == 'random':
        obs_indices = np.sort(rng.choice(Nx, size=Ny, replace=False))
    else:
        raise ValueError(f'unknown mode {mode}')
    return obs_indices

#------------------------------------------------------------------------------
def H_linear(x):
    """ Linear operator."""
    return x

def H_log(x,  eps=1e-3):
    """
    Log observation operator: log(|footprint| + eps).
 
    eps=0 reproduces the strongly-nonlinear H(x) = log|x| benchmark of
    Poterjoy (2006), used as a standard stress test for EnKF-type filters
    on Lorenz models — including its singularity where footprint == 0.
    A small eps > 0 (the default) regularizes that singularity for
    numerically stable experiments (e.g. gradient-based training).
    """
    return np.log(np.abs(x) + eps)

def H_tanh(x, A=1.0, B=0.0):
    """Saturating radiance/brightness-temperature-style operator."""
    return A * np.tanh(x) + B

def H_log_quadratic(x):
    """Radar-backscatter-style operator: log(1 + footprint^2)."""
    return np.log1p(x ** 2)


def H_quartic(x, sigma=1.0):
    """Stefan-Boltzmann-style operator: sigma * footprint^4 (harsh nonlinearity)."""
    return sigma * x ** 4

def dH_linear(f):
    return np.ones_like(f)

def dH_log(f, eps=1e-3):
    return np.sign(f) / (np.abs(f) + eps)

def dH_tanh(f, A=1.0, B=0.0):
    return A * (1.0 - np.tanh(f) ** 2)

def dH_log_quadratic(f):
    return 2.0 * f / (1.0 + f ** 2)

def dH_quartic(f, sigma=1.0):
    return 4.0 * sigma * f ** 3

DH = {H_linear: dH_linear, H_log: dH_log, H_tanh: dH_tanh,
      H_log_quadratic: dH_log_quadratic, H_quartic: dH_quartic}
  
def correlated_R(W, alpha=0.5):
    """
    Construct a footprint-correlated observation error covariance matrix R.

    R = sigma^2 * [ (1 - alpha) * I + alpha * Wn @ Wn.T ]

    where Wn is W with each row normalized to unit L2 norm, so Wn @ Wn.T
    has unit diagonal and off-diagonal entries equal to the cosine
    similarity between footprint kernels (i.e. how much two observations'
    footprints overlap on the state).

    W       : (K, N) footprint/weight matrix, as returned by build_footprint_matrix.
    sigma   : marginal observation error std dev (same for all K obs).
    alpha   : in [0, 1]; 0 -> diagonal R (independent noise), 1 -> error
              correlation fully tracks footprint overlap. Intermediate
              values blend independent instrument noise with shared
              retrieval error.

    Returns a (K, K) symmetric positive-definite covariance matrix.
    """
    row_norms = np.linalg.norm(W, axis=1, keepdims=True)
    Wn = W / row_norms  # row-normalized: diag(Wn @ Wn.T) == 1

    K = W.shape[0]
    corr = Wn @ Wn.T
    R = ((1.0 - alpha) * np.eye(K) + alpha * corr)
    return R
