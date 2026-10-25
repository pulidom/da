import sys;  sys.path.insert(0, '../src/'); 
import numpy as np
from dyn import L96
from dyn import L96_2scl
import gra

def integ_mdl(mdl,nt=1000):
    
    x0,X0 = mdl.initialization()
    x_t=np.zeros([nt,x0.shape[0]])
    x_t[0]=x0
    for it in range(1,nt):
        x_t[it]=l96(x_t[it-1])
    return x_t

nt = 1000
l96  = L96_2scl(F=10,Nx=40,Ny=400,dtcy = 0.05)
x_t=integ_mdl(l96,nt=nt)


gra.plot_hovmoller(x_t[500:1000,:40],0.05,filename='../tmp/hovmoller_l96_2scl.png')

def plot_variable(x1, dtcy, k=0,filename='../tmp/ts_l96_2scl.png' ):

    nt, nx = x1.shape
    time = np.arange(nt) * dtcy

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(time, x1[:, k])
    ax.plot(time, x1[:, 41])

    ax.set_xlabel("Time")
    ax.set_ylabel(rf"$X_{k}$")

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(f'../tmp/{title}_ts.png')
    
plot_variable(x_t[500:1000],0.05)
#
