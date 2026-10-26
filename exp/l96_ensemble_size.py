import os
os.environ['OMP_NUM_THREADS'] = '1'  # one per worker
import numpy as np
import sys;  sys.path.insert(0, '../src/'); 
from dyn import L96 as mdl
import obs as obs
import utils
from enkf import FILTER
from brute_force import robust_zoom
from multiprocessing import Pool


nworkers=16
seed=22
Nx = 40
Ny = 20
Nems = [20, 30, 50, 100, 200, 300]
assmtd = 'etkf'
ispin = 200
Ncy = 2200
dtcy = 0.05
Rfac = 0.5
finf_max = [1.4,1.3,1.2,1.1,1.05,1.05]
expdir = f'../dat/l96_40v/inf_{assmtd}_dtcy{dtcy:.2f}_R{Rfac:.1f}/'


os.makedirs(expdir, exist_ok=True)

Mdl = mdl(Nx=Nx, dtcy = dtcy, expdir=expdir, Nem=Nems[-1])
xt0, X00 = Mdl.initialization()
 
Obs    = obs.OBS(R = Rfac, Nx=Nx, Ny=Ny, Tmdl=Mdl, Ncy=Ncy)
xt_t, y_t = Obs(xt0)
 

def make_objective(Mdl, X0, seed=4, assmtd='etkf'):
    def f(finf):
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
 
def run(Nem, fmax):
    X0=X00[:,:Nem]
    finf_opt, cache = robust_zoom(make_objective(Mdl, X0, assmtd=assmtd),
                                  1.0, fmax, npts=10, nlevels=3)
    
    # rerun the filter with the best inflation (same seed, same result)
    # and save its spread
    filt = FILTER(Mdl, Obs, assmtd=assmtd, finf=finf_opt, seed=seed)
    Xa_t, _ = filt(X0, y_t)
    rmse = utils.RMSE(xt_t[ispin:] - Xa_t[ispin:].mean(-1))
    spread = utils.spread(Xa_t[ispin:])
    return Nem, finf_opt, rmse, spread
 
 
if __name__ == '__main__':
 
    print('Method: ', assmtd)
    # largest ensembles first: they are the slowest, so this balances the load
    jobs = list(zip(Nems, finf_max))[::-1]
    with Pool(nworkers) as pool:
        out = pool.starmap(run, jobs, chunksize=1)
    results = {Nem: (finf_opt, rmse, spread) for Nem, finf_opt, rmse, spread in out}
 
#    for Nem in Nems:
#        finf_opt, cache = results[Nem]
#        print('\nNem =', Nem)
#        # look at the whole curve: the U shape should be visible
#        for p in sorted(cache):
#            print(f'{p:.4f}  {cache[p]:.4f}')
 
    print('\nSummary: ', assmtd)
    print('Nem    finf_opt   RMSE')
    for Nem in Nems:
        finf_opt, rmse, spread = results[Nem]
        print(f'{Nem:4d}   {finf_opt:.4f}    {rmse:.4f} {spread:.4f}')

    # same table in a text file
    with open(f'{expdir}results_{assmtd}.txt', 'w') as fo:
        fo.write('# Nem    finf_opt   RMSE     spread\n')
        for Nem in Nems:
            finf_opt, cache, spread = results[Nem]
            fo.write(f'{Nem:4d}   {finf_opt:.4f}    {rmse:.4f}   {spread:.4f}\n')
 
        
