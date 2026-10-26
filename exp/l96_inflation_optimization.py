import numpy as np
import sys;  sys.path.insert(0, '../src/'); 
from dyn import L96 as mdl
import obs as obs
import utils
from enkf import FILTER
from brute_force import robust_zoom

Nx = 40
Ny = 20
Mdl= mdl(Nx=Nx,expdir='../dat/l96_40v/',Nem=30)

xt0,X0 = Mdl.initialization()


#H=np.eye(Nx)
indobs = utils.obs_points(Nx, Ny)
Obs    = obs.OBS(0.5*np.eye(Ny),indobs,Tmdl=Mdl,Ncy=1200)
xt_t,y_t= Obs(xt0)
ispin = 200
# Reference values perfect model R=0.5
# Nem = 30
## Ny=20 ETKF finf=1.0125 RMSE:  0.1896904916746387
## Ny=20 perobs2 1.083 RMSE:  0.2757086485893622
## Ny=40 perobs2 1.05 RMSE:  0.15822656105698532
## Ny=40 ETKF 1.00416 RMSE:  0.12350096239685815

# Nem = 100  Ny=40 perobs2  p=1.015    0.1306
#            etkf    p=1.009    f=0.1304


def make_objective(seed=4, assmtd='etkf'):
    def f(finf):
        errs = []
        filt = FILTER(Mdl, Obs, assmtd=assmtd, finf=finf, seed=seed)
        try:
            Xa_t, _ = filt(X0, y_t)
        except np.linalg.LinAlgError:
            return np.inf                       # diverged: reject this finf
        r = utils.RMSE(xt_t[ispin:] - Xa_t[ispin:].mean(-1))
        if not np.isfinite(r):
            return np.inf
        return r
    return f

assmtd='etkf'
print('Method: ',assmtd)
finf_opt, cache = robust_zoom(make_objective(assmtd=assmtd), 1.0, 1.3, npts=10, nlevels=3)

# look at the whole curve: the U shape should be visible
for p in sorted(cache):
    print(f'{p:.4f}  {cache[p]:.4f}')
