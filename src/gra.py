import sys  
import subprocess
import string
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size' : 9})
matplotlib.rcParams.update({'legend.fontsize' : 'small'})
matplotlib.rcParams.update({'legend.frameon' : 0})
import matplotlib.pyplot as plt


def save_and_show(fig, fname):
    "Save the figure and display it (external viewer)."
    fig.savefig(fname)
    subprocess.run(['checkifrunning', fname], check=False)


def plot_panel(ax, jx, tspan, xa, *, xa2=None, xf=None, xf2=None,
               xt=None, y=None, indobs=None,
               xa_label='Ana', xa2_label='Ana 2', ylabel=None, ylim=None):
    """Time series of state variable jx.
    xa, xa2, xf, xf2 : ensembles (ncy, nx, nem); xt : (ncy, nx); y : (ncy, ny)."""

    # ensemble members (gray) and the first one labelled for the legend
    for ens, label in ((xa, 'Ens'), (xa2, None)):
        if ens is not None:
            lines = ax.plot(tspan, ens[:, jx, :], color='gray', alpha=0.4, lw=0.8)
            if label:
                lines[0].set_label(label)

    # ensemble means
    for ens, style, label in ((xf,  '-.', 'Frcst 1'),
                              (xf2, '-.', 'Frcst 2'),
                              (xa,  '-',  xa_label),
                              (xa2, '-',  xa2_label)):
        if ens is not None:
            ax.plot(tspan, ens[:, jx, :].mean(axis=-1), style, label=label)

    if xt is not None:
        ax.plot(tspan, xt[:, jx], '--', label='True')

    if y is not None:
        # indobs=None: all variables observed, in order
        indobs = np.arange(y.shape[1]) if indobs is None else np.asarray(indobs)
        iobs = np.flatnonzero(indobs == jx)
        if iobs.size:
            ax.plot(tspan, y[:, iobs[0]], '.', label='Obs')

    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel or f'x[{jx}]')
    if ylim is not None:
        ax.set_ylim(ylim)


def assm(variables, xa, *, filename=None, title='', ylabel=None, ylim=None,
         legend_panel=1, **panel_kw):
    """Panels with assimilation time series, one per variable (at most 3).
    xa : analysis ensemble (ncy, nx, nem).
    panel_kw : xa2, xf, xf2, xt, y, indobs, xa_label, xa2_label."""
    if len(variables) > 3:
        raise ValueError('At most 3 variables per figure; call twice')

    nv = len(variables)
    tspan = np.arange(xa.shape[0])

    fig, axs = plt.subplots(1, nv, figsize=(10, 3), squeeze=False,
                            constrained_layout=True)
    fig.suptitle(title)

    for k, (ax, jx) in enumerate(zip(axs[0], variables)):
        plot_panel(ax, jx, tspan, xa, ylabel=ylabel, ylim=ylim, **panel_kw)
        ax.text(0.92, 1.03, f'({string.ascii_lowercase[k]})',
                fontsize=14, transform=ax.transAxes)

    axs[0, min(legend_panel, nv - 1)].legend(loc='best')

    if filename is not None:
        save_and_show(fig, filename)
    plt.close(fig)
    
#--------------------------------------------------
def ylabelp(ylabel0,string):
    if ylabel0 is None: ylabel0=string
    return ylabel0

labelPanel=['(a)','(b)','(c)','(d)','(e)','(f)']    


def plot_hovmoller(x, dtcy, title="Lorenz-96"):
    nt, nx = x.shape
    time = np.arange(nt) * dtcy

    fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.imshow(
        x.T,
        aspect="auto",
        origin="lower",
        extent=[time[0], time[-1], 0, nx],
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Variable")
    ax.set_title(title)

    fig.colorbar(im, ax=ax, label="X")

    plt.tight_layout()
    plt.savefig(f'../tmp/{title}_hovmoller.png')
