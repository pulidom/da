from scipy.stats import norm
from scipy.stats import skew, kurtosis
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


def scan_dtcy_rfac(Tmdl, Nx, obs_indices, dtcy_list, Rfac_list,
                   nem=200, ncy=500, ispin=100, seed=0, assmtd='etkf',
                   finf_range=(1.0, 1.3), tune=True, finf_fixed=1.05,
                   set_dtcy=None, verbose=True):
    """Scan the cycle length dtcy and the observation error variance Rfac.

    For each (dtcy, Rfac): regenerate truth and observations, tune finf
    (zoom search), run the filter, and compute RMSE, spread and
    non-Gaussianity statistics of the forecast ensemble.

    set_dtcy : function (Tmdl, dtcy) -> None that changes the cycle length.
               Default assumes Tmdl.kt = round(dtcy / Tmdl.dt); adapt it to your model.
    Returns a dict of arrays with shape (len(dtcy_list), len(Rfac_list)).
    """
    if set_dtcy is None:
        def set_dtcy(M, d):
            M.kt = max(1, int(round(d / M.dt)))

    Ny = len(obs_indices)
    keys = ['finf', 'rmse', 'spread', 'ratio', 'skew', 'kurt', 'w2',
            'z_mean', 'z_var', 'z_skew', 'z_kurt']
    out = {k: np.full((len(dtcy_list), len(Rfac_list)), np.nan) for k in keys}
    kt_orig = Tmdl.kt

    try:
        for i, dtcy in enumerate(dtcy_list):
            set_dtcy(Tmdl, dtcy)
            # compute_initial_conditions bypasses the file cache, which would be
            # stale because the climatology depends on the cycle length
            xt0, X0 = Tmdl.compute_initial_conditions(nem=nem, seed=seed)

            for j, Rfac in enumerate(Rfac_list):
                obs = OBS(Rfac * np.eye(Ny), Nx, obs_indices, seed=seed)
                xt_t, y_t = obs.generate_obs(xt0, ncy, Tmdl)

                def run(finf):
                    filt = FILTER(Tmdl, obs, assmtd=assmtd, finf=finf, seed=seed)
                    return filt(X0, y_t)

                def objective(finf):          # ETKF is deterministic: one run
                    try:
                        Xa_t, _ = run(finf)
                    except np.linalg.LinAlgError:
                        return np.inf
                    r = np.sqrt(np.mean((xt_t[ispin:] - Xa_t[ispin:].mean(-1)) ** 2))
                    return r if np.isfinite(r) else np.inf

                finf = (robust_zoom(objective, *finf_range, npts=5, nlevels=3,
                                    verbose=False)[0] if tune else finf_fixed)
                Xa_t, Xf_t = run(finf)

                err = xt_t[ispin:] - Xa_t[ispin:].mean(-1)
                rmse = np.sqrt(np.mean(err ** 2))
                spread = np.sqrt(np.mean(Xa_t[ispin:].var(axis=-1, ddof=1)))
                ng = nongauss_stats(Xf_t, y_t, obs, ispin=ispin)

                out['finf'][i, j] = finf
                out['rmse'][i, j] = rmse
                out['spread'][i, j] = spread
                out['ratio'][i, j] = spread / rmse
                for k in ('skew', 'kurt', 'w2', 'z_mean', 'z_var', 'z_skew', 'z_kurt'):
                    out[k][i, j] = ng[k]

                if verbose:
                    print(f'dtcy={dtcy:.3g} Rfac={Rfac:.3g}  finf={finf:.3f}  '
                          f'rmse={rmse:.3f}  spread/rmse={spread / rmse:.2f}  '
                          f'kurt={ng["kurt"]:.2f}  z_var={ng["z_var"]:.2f}')
    finally:
        Tmdl.kt = kt_orig                     # always restore the model

    out['dtcy'], out['Rfac'] = np.asarray(dtcy_list), np.asarray(Rfac_list)
    return out


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
res = scan_dtcy_rfac(Tmdl, Nx=40, obs_indices=idx,
                     dtcy_list=[0.05, 0.1, 0.2, 0.4],
                     Rfac_list=[0.1, 0.5, 1.0, 2.0],
                     nem=200, ncy=500, ispin=100)
plot_scan(res)
