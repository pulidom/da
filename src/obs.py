import numpy as np
from utils import DH, H_linear

class OBS:
    """ Observations y = Hmap(x[obs_indices]) + noise, with Hmap acting elementwise.
        Hmap = H_linear, H_log, H_tanh, H_quartic, H_log_quadratic 
    """

    def __init__(self, R,  obs_indices, Hmap=H_linear, seed=22):
        
        self.rng = np.random.default_rng(seed)
        self.randn = self.rng.standard_normal

        self.obs_indices = np.atleast_1d(obs_indices)
        self.Ny = len(self.obs_indices)

        self.Hmap, self.dHmap = Hmap, DH[Hmap]
        self.obs_shape = (self.Ny,)

        self.R = np.atleast_2d(R)
        self.Rinv = np.linalg.inv(self.R)
        self.sqR = np.linalg.cholesky(self.R)        # R = sqR @ sqR.T

    def Hop(self, x):
        return self.Hmap(x[self.obs_indices])

    def Htop(self, x, dy):
        """H'(x)^T dy, linearized at x."""
        out = np.zeros(x.shape[0])
        out[self.obs_indices] = self.dHmap(x[self.obs_indices]) * dy
        return out

    def jacobian(self, x):
        """H'(x), shape (Ny, Nx), built row by row from the adjoint."""
        return np.array([self.Htop(x, e) for e in np.eye(self.Ny)])

    def generate_obs(self, x0_true, Ncy, Tmdl):
        """xt_t: (Ncy, Nx); y_t: (Ncy, *obs_shape)"""
        x1 = np.copy(x0_true)
        y_t = np.empty((Ncy, *self.obs_shape))
        xt_t = np.empty((Ncy, x0_true.shape[0]))

        for it in range(Ncy):
            yf = self.Hop(x1)
            y_t[it] = yf + self.randn(yf.shape) @ self.sqR.T   # works for (Ny,) and (Ntobs, Ny)
            xt_t[it] = x1
            x1 = Tmdl.integ(x1)

        return xt_t, y_t
    
    def generate_trueobs(self, xt0, ncy, Tmdl, dat_fname=None):
        
        if dat_fname and os.path.isfile(dat_fname):
            data = np.load(dat_fname)
            yo_t, xt_t = data["y"], data["xt"]
            
            if (yo_t.shape[0] >= ncy and yo_t.shape[1:] == self.obs_shape
                    and xt_t.shape[1] == Tmdl.nx):
                print("Load true and obs from", dat_fname)
                return xt_t[:ncy], yo_t[:ncy]

        print("Generate true and obs")
        xt_t, yo_t = self.generate_obs(xt0, ncy, Tmdl)
        
        if dat_fname:
            np.savez(dat_fname, y=yo_t, xt=xt_t)
            
        return xt_t, yo_t
        
    __call__ = generate_trueobs
    
#----------------------------------------------------------
class OBS4D(OBS):
    """Observations at Ntobs times inside the window, via the model Mdl."""

    def __init__(self, R,  obs_indices, Ntobs=1, Mdl=None, **kwargs):
        super().__init__(R,  obs_indices, **kwargs)
        
        self.Ntobs = Ntobs
        Mdl.set_ntobs(ntobs=Ntobs)
        self.Mdl = Mdl
        self.obs_shape = (Ntobs, self.Ny)

    def Hop(self, x0):
        x = self.Mdl.integ2obs(x0)                    # (Ntobs, Nx), stores Mdl.x_traj
        return self.Hmap(x[:, self.obs_indices])      # (Ntobs, Ny)

    def Htop(self, x0, dy):
        """Uses the trajectory from the last Hop(x0) call."""
        
        jt = self.Mdl.jtobs
        adx = np.zeros_like(x0)
        for ij in range(self.Ntobs, 0, -1):
            adx += super().Htop(self.Mdl.x_traj[jt[ij]], dy[ij - 1])
            adx = self.Mdl.backward(self.Mdl.x_traj, adx, jt[ij - 1], jt[ij])
            
        return adx

#----------------------------------------------------------
def weight_matrix(Nx, obs_indices, length_scale, tag='GC'):
    """Footprint matrix W (Ny, Nx) on a periodic ring: row j averages the state
    around obs_indices[j] with a taper of radius length_scale (grid points).
    Rows are normalized to sum 1. Reuses loc_matrix from the localization module."""
    W = loc_matrix(Nx, obs_indices, length_scale, tag).T     # (Ny, Nx)
    return W / W.sum(axis=1, keepdims=True)


#----------------------------------------------------------
class OBSW(OBS):
    """OBS with nonlocal observations: y = Hmap(W x) + noise.
    length_scale = 0 -> local observations (identical to OBS)."""

    def __init__(self, R,  obs_indices, Nx  = 0, length_scale=0, tag='GC', **kwargs):
        super().__init__(R, obs_indices, **kwargs)
        self.length_scale = length_scale
        self.W = (weight_matrix(Nx, self.obs_indices, length_scale, tag)
                  if length_scale > 0 else None)

    def Hop(self, x):
        if self.W is None:
            return super().Hop(x)
        return self.Hmap(self.W @ x)               # works for (Nx,) and (Nx, nem)

    def Htop(self, x, dy):
        """H'(x)^T dy = W^T (Hmap'(W x) * dy)."""
        if self.W is None:
            return super().Htop(x, dy)
        return self.W.T @ (self.dHmap(self.W @ x) * dy)
    
#----------------------------------------------------------
if __name__ == '__main__':
    
    import dyn
    Tmdl = dyn.L96()
    xt0, X0 = Tmdl.initialization()

    Ncy = 100
    Nx, Ny = 40, 20
    indobs = utils.obs_points(Nx, Ny)

    Obs = OBS4D(1.0 * np.eye(Ny), indobs , Ntobs=1, Mdl=Tmdl)

    xt_t, y_t = Obs(xt0, Ncy, Tmdl, dat_fname='../dat/test/trueobs.npz')   # note the .npz
    print('Observaciones:', Obs.obs_indices)
