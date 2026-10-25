"""  Hace un barrido en dtcy y en R para ver las dependencias de la no-gaussianidad
      Para cada caso optimiza el factor de inflacion

"""
import os
import numpy as np
import sys;  sys.path.insert(0, '../src/'); 
from dyn import L96 as mdl
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
from scipy.stats import skew, kurtosis
import utils
from enkf import FILTER
from brute_force import robust_zoom
from obs import OBS

def w2_gauss(x):                      # x: (nem,)
    q = norm.ppf((np.arange(1, len(x) + 1) - 0.5) / len(x))
    return np.sqrt(np.mean((np.sort(x) - x.mean() - x.std(ddof=1) * q) ** 2)) / x.std(ddof=1)

def nongauss_stats(Xf_t, y_t, Obs, ispin=100):
    sk, ku, w2, z_all = [], [], [], []
    for icy in range(ispin, Xf_t.shape[0]):
        X = Xf_t[icy]                                   # (Nx, Nem)
        sk.append(np.abs(skew(X, axis=1)).mean())
        ku.append(np.abs(kurtosis(X, axis=1)).mean())   # excess kurtosis
        w2.append(np.mean([w2_gauss(x) for x in X]))

        HX = Obs.Hop(X)                                 # (Ny, Nem)
        d = y_t[icy] - HX.mean(axis=1)
        s = np.sqrt(HX.var(axis=1, ddof=1) + np.diag(Obs.R))
        z_all.append(d / s)
    z = np.concatenate(z_all)
    return dict(skew=np.mean(sk), kurt=np.mean(ku), w2=np.mean(w2),
                z_mean=z.mean(), z_var=z.var(),
                z_skew=skew(z), z_kurt=kurtosis(z))

def plot_scan(res, keys=('rmse', 'ratio', 'finf', 'kurt', 'w2', 'z_var')):
    """Heatmaps of the scan results against (Rfac, dtcy)."""
    n = len(keys)
    fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(4.2 * ((n + 1) // 2), 7))
    for ax, k in zip(axs.ravel(), keys):
        im = ax.imshow(res[k], origin='lower', aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(res['Rfac']))); ax.set_xticklabels([f'{r:g}' for r in res['Rfac']])
        ax.set_yticks(range(len(res['dtcy']))); ax.set_yticklabels([f'{d:g}' for d in res['dtcy']])
        ax.set_xlabel('Rfac'); ax.set_ylabel('dtcy'); ax.set_title(k)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig

def ic_file(expdir, dtcy, nem, seed, nx):
    return os.path.join(expdir, f'ini_cond_dtcy{dtcy}_nem{nem}_seed{seed}_nx{nx}.npz')

def precompute_ics(dtcy_list, expdir, nem, seed, nx):
    os.makedirs(expdir, exist_ok=True)
    for dtcy in dtcy_list:
        fname = ic_file(expdir, dtcy, nem, seed, nx)
        if os.path.isfile(fname):
            continue
        Tmdl =mdl(dtcy=dtcy)
        xt0, X0 = Tmdl.gen_ICs(nem=nem, seed=seed)
        tmp = fname + f'.{os.getpid()}.tmp.npz'
        np.savez(tmp, xt0=xt0, X0=X0)
        os.replace(tmp, fname)
        
def run_cell(args):
    """One (dtcy, Rfac) cell. Must be a top-level function (picklable)."""
    (dtcy, Rfac, Nx, obs_indices, nem, ncy, ispin, seed,
     assmtd, finf_range, tune, finf_fixed, expdir) = args
    
    print(f'running dtcy={dtcy} Rfac={Rfac}', flush=True)
    
    Tmdl = mdl(dtcy = dtcy)                       # build the model INSIDE the worker
    #xt0, X0 = Tmdl.gen_ICs(nem=nem, seed=seed)
    data = np.load(ic_file(expdir, dtcy, nem, seed, Nx))
    xt0, X0 = data['xt0'], data['X0']
    Ny = len(obs_indices)
    obs = OBS(Rfac * np.eye(Ny), obs_indices, seed=seed)
    xt_t, y_t = obs.generate_obs(xt0, ncy, Tmdl)

    def run(finf):
        return FILTER(Tmdl, obs, assmtd=assmtd, finf=finf, seed=seed)(X0, y_t)

    def objective(finf):
        try:
            Xa_t, _ = run(finf)
        except np.linalg.LinAlgError:
            return np.inf
        r = np.sqrt(np.mean((xt_t[ispin:] - Xa_t[ispin:].mean(-1)) ** 2))
        return r if np.isfinite(r) else np.inf

    try:
        finf = (robust_zoom(objective, *finf_range, npts=5, nlevels=3, verbose=False)[0]
                if tune else finf_fixed)
        Xa_t, Xf_t = run(finf)
    except RuntimeError:                      # every finf failed
        return dict(dtcy=dtcy, Rfac=Rfac)     # leaves NaN in the arrays

    rmse = np.sqrt(np.mean((xt_t[ispin:] - Xa_t[ispin:].mean(-1)) ** 2))
    spread = np.sqrt(np.mean(Xa_t[ispin:].var(axis=-1, ddof=1)))
    res = dict(dtcy=dtcy, Rfac=Rfac, finf=finf, rmse=rmse, spread=spread,
               ratio=spread / rmse)
    res.update(nongauss_stats(Xf_t, y_t, obs, ispin=ispin))
    return res


def scan_parallel(dtcy_list, Rfac_list, Nx, obs_indices, expdir, nworkers=8, **kw):
    defaults = dict(nem=200, ncy=1200, ispin=200, seed=0, assmtd='etkf',
                    finf_range=(1.0, 1.3), tune=True, finf_fixed=1.05)
    defaults.update(kw)
    
    precompute_ics(dtcy_list, expdir, defaults['nem'], defaults['seed'], Nx)

    
    tasks = [(d, R, Nx, obs_indices, defaults['nem'], defaults['ncy'], defaults['ispin'],
              defaults['seed'], defaults['assmtd'], defaults['finf_range'],
              defaults['tune'], defaults['finf_fixed'], expdir)
             for d in dtcy_list for R in Rfac_list]

    with ProcessPoolExecutor(max_workers=nworkers) as ex:
        results = list(ex.map(run_cell, tasks))      # keeps task order

    keys = ['finf', 'rmse', 'spread', 'ratio', 'skew', 'kurt', 'w2',
            'z_mean', 'z_var', 'z_skew', 'z_kurt']
    out = {k: np.full((len(dtcy_list), len(Rfac_list)), np.nan) for k in keys}
    for n, r in enumerate(results):
        i, j = divmod(n, len(Rfac_list))
        for k in keys:
            if k in r:
                out[k][i, j] = r[k]
    out['dtcy'], out['Rfac'] = np.asarray(dtcy_list), np.asarray(Rfac_list)
    return out


if __name__ == '__main__':                     # REQUIRED with multiprocessing
    Nx = 40; Ny = 20
    idx =  utils.obs_points(Nx, Ny)
    res = scan_parallel([0.05, 0.1, 0.2, 0.4], [0.1, 0.5, 1.0, 1.5, 2.0],
                        Nx=Nx, obs_indices=idx, nworkers=4,expdir='../dat/l96_40v/ics')

    np.savez('../dat/l96_40v/scan_results.npz', **res)
    fig.plot_scan(res)
    fig.savefig('../tmp/nongauss.png')
