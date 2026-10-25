""" Opmitiza un hyperparametro en distintos niveles dada la funcion objetivo
"""
import numpy as np
def brute_search(lo,hi,n_eval):
    finf=np.linspace(lo,hi,n_val)
    for i in range(n_eval):
        Filter=FILTER(Mdl,Obs,finf=finf[i],assmtd='etkf') #perobs2')#'etkf')
        Xa_t,Xf_t=Filter(X0,y_t)
        print(finf[i])
        err=xt_t-Xa_t.mean(-1)
        print('RMSE: ',utils.RMSE(err[200:]))


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
        #w = hi - lo
        #if i == 0 and np.isclose(lo, lo0) and nexp < max_expand:
        #    lo, hi, lo0, nexp = lo - w, lo + w, lo - w, nexp + 1
        #    if verbose: print('   edge: expanding downward')
        #    continue
        #if i == npts - 1 and np.isclose(hi, hi0) and nexp < max_expand:
        #    lo, hi, hi0, nexp = hi - w, hi + w, hi + w, nexp + 1
        #    if verbose: print('   edge: expanding upward')
        #    continue

        lo, hi = ps[max(i - 1, 0)], ps[min(i + 1, npts - 1)]
    return min(cache, key=cache.get), cache
