"""  Localized versions of the ensemble kalman filter
"""
import numpy as np
# my modules
import utils import meanper, PaWa

#--------------------------------------------------------
class LFILTER(FILTER):
    """Localized ensemble filter: LEnKF + LETKF.
    """
    def __init__(self, Fmdl, Obs, assmtd='letkf', finf=1.0, seed=0,
                 corrl=2.0,    # localization radius in grid points
                 loc_tag='GC', # 'GC', 'Gauss' or 'Step'
                 cutoff=1e-6):
        super().__init__(Fmdl, Obs, assmtd=assmtd, finf=finf, seed=seed)

        
        self.corrl, self.loc_tag, self.cutoff = corrl, loc_tag, cutoff
        self.Cloc = loc_matrix(Obs.Nx, Obs.obs_indices, corrl, loc_tag)   # (Nx, Ny)
        self.Coo  = self.Cloc[Obs.obs_indices]
        
        self.sqRinv = pinv(Obs.sqR) # letkf
        
        if assmtd == 'letkf':
            assert np.allclose(self.R, np.diag(np.diag(self.R))), \
                "letkf needs a diagonal R"
            
        rinv = 1.0 / np.diag(self.R)
        self.iobs = [np.flatnonzero(c > cutoff) for c in self.Cloc]
        self.wloc = [rinv[i] * c[i] for i, c in zip(self.iobs, self.Cloc)]

    #--------------------------------------------------------    
    def perobs2(self, Xf, y):
        """   Perturbed-obs EnKF with covariance localization.
              Xf [Nx,Nem], y [Ny]
        """
        
        nem = Xf.shape[1]; ny = y.shape[0]
        Hx = self.Hop(Xf)                          # (Ny, Nem)

        xfm, Xper = meanper(Xf)
        hxm, Yper = meanper(Hx)

        Pxy = (Xper @ Yper.T) / (nem - 1) * self.Cloc   # (Nx, Ny), tapered
        Pyy = (Yper @ Yper.T) / (nem - 1) * self.Coo    # (Ny, Ny), tapered
        S = Pyy + self.R

        K = np.linalg.solve(S, Pxy.T).T                 # Pxy S^-1, S symmetric
        Noise = self.sqR @ self.randn((ny, nem))
        
        return Xf + K @ (y[:, None] - Hx + Noise)

    #--------------------------------------------------------
    def letkf(self, Xf, y):
        """  LETKF (Hunt et al. 2007),  assumes R diagonal
             Xf: (nx, k) already inflated in asscy. 
             y: (ny,).
        """
    
        nx, k = Xf.shape
        xfm, Xp = meanper(Xf)
        yfm, Yp = meanper(self.Hop(Xf))
        dy = y - yfm[:, 0]
        I = (k - 1) * np.eye(k)

        Xa = np.empty_like(Xf)
        
        for ix in range(nx):
            iobs, w = self.iobs[ix], self.wloc[ix]
            Yi = Yp[iobs]                                  # (l, k)
            B = (Yi.T * w) @ Yi + I                        # Eq. (21), inverse of Pa
            Pa, Wa = PaWa(B)                           # Pa and Eq. (24)
            wbar = Pa @ (Yi.T @ (w * dy[iobs]))            # Eq. (20)
            Xa[ix] = xfm[ix] + Xp[ix] @ (wbar[:, None] + Wa)   # Eq. (25)
            
        return Xa
    
#--------------------------------------------------------
def dist2coeff(dists, radius, tag='GC'):
    """  Localization coefficients from distances (any array shape).
         radius is in grid points. 
    """
    
    coeffs = np.zeros(dists.shape)

    if tag == 'Gauss':
        coeffs = np.exp(-0.5 * (dists / radius) ** 2)
    elif tag == 'Step':
        coeffs[dists <= radius] = 1.0
    elif tag == 'GC':
        R = radius * 1.7386
        r = dists / R
        i1 = r <= 1
        i2 = (r > 1) & (r <= 2)
        r1, r2, r3 = r[i1], r[i1] ** 2, r[i1] ** 3
        coeffs[i1] = 1 + r2 * (-r3 / 4 + r2 / 2) + r3 * 5 / 8 - r2 * 5 / 3
        r1, r2, r3 = r[i2], r[i2] ** 2, r[i2] ** 3
        coeffs[i2] = (r2 * (r3 / 12 - r2 / 2) + r3 * 5 / 8 + r2 * 5 / 3
                      - r1 * 5 + 4 - (2 / 3) / r1)
    else:
        raise KeyError(f'Unknown localization tag: {tag}')
    return coeffs

#--------------------------------------------------------
def loc_matrix(Nx, obs_indices, radius, tag='GC'):
    """ Localization weights Cloc[ix, j] 
        between grid point ix and observation j,
        on a periodic 1D ring. Shape (Nx, Ny)."""
    
    d = np.abs(np.arange(Nx)[:, None] - np.asarray(obs_indices)[None, :])
    d = np.minimum(d, Nx - d)                    # periodic distance
    return dist2coeff(d, radius, tag)
