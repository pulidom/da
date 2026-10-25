import numpy as np
import sys;  sys.path.insert(0, '../src/'); 
from dyn import L96 as mdl
import obs as obs
import utils
from enkf import FILTER

Nx = 40
Ny = 20
Mdl= mdl(nx=Nx,expdir='../dat/l96_40v/')

xt0,X0 = Mdl.initialization(nem=30)


#H=np.eye(Nx)
indobs = utils.obs_points(Nx, Ny)
Obs    = obs.OBS(0.5*np.eye(Ny),indobs)
xt_t,y_t= Obs(xt0,1200,Mdl)
ispin = 200
## level 3: [1.07, 1.089]  best p=1.077  f=0.26

def robust_zoom(f, lo, hi, npts=7, nlevels=4, max_expand=3, verbose=True):
    """Minimize a noisy scalar f(p) on [lo, hi] by successive grid refinement.
    - f may return np.inf/nan for failed runs (treated as inf)
    - evaluations are cached
    - if the best point is on the edge of the ORIGINAL range, the range is expanded
    Returns best p and a dict {p: f(p)} with every evaluation."""
    cache = {}

    def ev(p):
        key = round(float(p), 10)
        if key not in cache:
            v = f(p)
            cache[key] = v if np.isfinite(v) else np.inf
        return cache[key]

    lo0, hi0, nexp = lo, hi, 0
    for level in range(nlevels):
        ps = np.linspace(lo, hi, npts)
        fs = np.array([ev(p) for p in ps])
        if not np.isfinite(fs).any():
            raise RuntimeError(f'all runs failed in [{lo:.4g}, {hi:.4g}]')
        i = int(np.argmin(fs))
        if verbose:
            print(f'level {level}: [{lo:.4g}, {hi:.4g}]  best p={ps[i]:.4g}  f={fs[i]:.4g}')

        # optimum on the edge of the original range: move the window outward
        w = hi - lo
        if i == 0 and np.isclose(lo, lo0) and nexp < max_expand:
            lo, hi, lo0, nexp = lo - w, lo + w, lo - w, nexp + 1
            if verbose: print('   edge: expanding downward')
            continue
        if i == npts - 1 and np.isclose(hi, hi0) and nexp < max_expand:
            lo, hi, hi0, nexp = hi - w, hi + w, hi + w, nexp + 1
            if verbose: print('   edge: expanding upward')
            continue

        lo, hi = ps[max(i - 1, 0)], ps[min(i + 1, npts - 1)]
    return min(cache, key=cache.get), cache

def make_objective(seeds=(0, 1, 2, 3, 4)):
    def f(finf):
        errs = []
        for s in seeds:
            filt = FILTER(Mdl, Obs, assmtd='etkf', finf=finf, seed=s)
            try:
                Xa_t, _ = filt(X0, y_t)
            except np.linalg.LinAlgError:
                return np.inf                       # diverged: reject this finf
            r = utils.RMSE(xt_t[ispin:] - Xa_t[ispin:].mean(-1))
            if not np.isfinite(r):
                return np.inf
            errs.append(r)
        return np.mean(errs)
    return f

finf_opt, cache = robust_zoom(make_objective(), 1.0, 1.3, npts=6, nlevels=4)

# look at the whole curve: the U shape should be visible
for p in sorted(cache):
    print(f'{p:.4f}  {cache[p]:.4f}')
