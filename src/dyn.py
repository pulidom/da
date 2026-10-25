#!/usr/bin/python
"""
Initialization and integration of a dynamical system

Lorenz-63 
Lorenz-96 
Lorenz-92 2sl

[2025-01-09] Simplicacion de codigos. Uso herencia.
 
Author Manuel Pulido


 Reference:

Pulido M., G. Scheffler, J. Ruiz, M. Lucini and P. Tandeo, 2016: Estimation of the functional form of subgrid-scale schemes using ensemble-based data assimilation: a simple model experiment. Q. J.  Roy. Meteorol. Soc.,  142, 2974-2984.

http://doi.org/10.1002/qj.2879


"""
import sys; sys.path.insert(0, '../')
import os
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
#import utils
from time import time
#from numba import njit

#----------------------------------------------------------
class M: # parent dynamical model class
    def __init__(self,
                 dtcy  = 0.01, # output times
                 dt    = None, # integration time step
#                 num_scheme = rk4, # numerical integration scheme
                 expdir = None, # experiment dir set to save the initial conditions
                 ):
        self.dtcy=dtcy        
        self.kt= int(dtcy/dt) # number of model time steps between observations
                              #(outputs of the model)
        self.dt=dtcy/self.kt # exact integration time step
        self.__num_scheme= self.rk4 #num_scheme
        self.expdir = expdir
        
    def __call__(self,x):
        return self.integ(x)
    
    def integ(self,x):
        " Integrate in a dtcy time "
        mdl = self.__num_scheme 
        for it in range(self.kt):
            x = mdl( x )            
        return x

    def set_ntobs(self, ntobs = 1): # number of observations within the cycle
        self.jtobs=np.linspace(0, self.kt - 1, ntobs+1, dtype=int)
        self.ntobs=ntobs
        
    def integ2obs(self,x): 
        ''' Integrate state in a given set of times for 4Dvar '''
        
        mdl = self.__num_scheme
        jt  = self.jtobs
        ntobs = len(jt)-1
        Xobs = np.empty((ntobs, *x.shape))
        x_traj = np.empty((jt[-1] + 1, *x.shape)) #full time step trajectory 
        x_traj[0] = x
        for ij in range(1,ntobs+1):
            for it in range(jt[ij-1],jt[ij]):
                x = mdl( x )
                x_traj[it + 1] = x

            Xobs[ij-1] = x
        self.x_traj = x_traj # store for use of the adjoint
    
        return Xobs
    
    def rk4(self,x):
        " 4th order Runge Kutta "

        dx1 = self._mdl( x )
        dx2 = self._mdl( x + 0.5 * dx1 )
        dx3 = self._mdl( x + 0.5 * dx2 )
        dx4 = self._mdl( x + dx3 )

        return  x +  ( dx1 + 2.0 * (dx2 + dx3) + dx4 ) / 6.0
        
    def euler(self,x):
        return x + self._mdl( x )

    def gen_ICs(self, nt_spinup=7200, nt_corr=30, seed=25, nem=100,
                x0=0, x0std=1, lens=1, nt_samples=1000):
        """Spinup, climatology, and random selection of xt0 and X0."""

        
        print('Generating initial conditions')
        rng = np.random.default_rng(seed)

        # Spinup from a random initial condition
        x = x0 + x0std * rng.normal(0, 1, self.nx)
        kt_orig = self.kt
        
        self.kt = kt_orig * nt_spinup
        x = self.integ(x)
        
        # Climatology: samples separated by nt_corr steps (uncorrelated)
        self.kt = kt_orig * nt_corr
        print('Climatology')
        x_t = np.empty((nt_samples, self.nx))
        x_t[0] = x
        for it in range(1, nt_samples):
            x_t[it] = self.integ(x_t[it - 1])
            
        self.kt = kt_orig  

        rng = np.random.default_rng(seed + 1)
        self.nem = nem

        if lens == 1:
            print('Take initial condition from climatology')
            idx = rng.choice(nt_samples, nem + 1, replace=True)
            X0 = x_t[idx[:nem]].T
            xt0 = x_t[idx[nem]]
        elif lens == 0:
            print('Take initial ensemble randomly around true state')
            xt0 = x_t[rng.integers(nt_samples)]
            Pf0 = np.cov(x_t.T)                    # climatological covariance
            self.sqPf0 = 0.1 * np.linalg.cholesky(Pf0)
            X0 = xt0[:, None] + self.sqPf0 @ rng.normal(0, 1, (self.nx, nem))

        return xt0, X0
    

    def initialization(self, nt_spinup=7200, nt_corr=30, seed=25, nem=100,
                       x0=0, x0std=1, lens=1, nt_samples=1000):
        """Initial true state xt0 (nx,) and ensemble X0 (nx, nem), from cache if available."""
        fname = None if self.expdir is None else os.path.join(self.expdir, 'ini_cond.npz')

        if fname and os.path.isfile(fname):
            print('Read sample file from:', fname)
            data = np.load(fname)
            return data['xt0'], data['X0']

        xt0, X0 = self.gen_ICs(nt_spinup, nt_corr, seed, nem,
                               x0, x0std, lens, nt_samples)
        if fname:
            np.savez(fname, X0=X0, xt0=xt0)
        return xt0, X0


#----------------------------------------------------------
class L63(M):
    def __init__( self,dt=0.005, # integration time step
                  sigma=10, rho=28., beta=8/3., # model paramaters
                  **kwargs):
        super().__init__(  dt=dt, **kwargs )
        
        """ Initializes L63 params
              background parameters sigma=11.5,beta=2.87,rho=32
        """        
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.nx = 3 # state dimension
        
        
    def _mdl(self,x):
        " L63 model equations "

        dx0 = self.sigma * (x[1] - x[0])
        dx1 = x[0] * (self.rho - x[2]) - x[1]
        dx2 = x[0] * x[1] - self.beta * x[2]
        
        return self.dt * np.array([dx0,dx1,dx2])
    
#----------------------------------------------------------
class L96(M):
    '''  
    Lorenz-96 dynamical system 
    \[ d_t X_k = - X_{k-1} (X_{k-2} - X_{k+1} ) - X_k + F \]

    Uses list of indices to speed up calculations
    Works with ensembles (second dimension) 
    '''
    def __init__( self, dt=0.005, dtcy = 0.05, F=8., nx=40, **kwargs):
        super().__init__(  dt=dt, dtcy = dtcy, **kwargs)

        self.dt=dt
        self.nx=nx
        self.F = F 
        
        #indices for model scheme
        k = np.arange(nx)
        self.indm2 = (k - 2) % nx
        self.indm1 = (k - 1) % nx
        self.indp1 = (k + 1) % nx


    def _mdl(self,x):
        " L96 model equations Requires nx as first dim x[nx,npa/nem]"
        dx = (x[self.indp1] - x[self.indm2]) * x[self.indm1] - x + self.F
        return dx * self.dt


#----------------------------------------------------------
class L96_2scl(M):
    '''
    Two scale L96
    Uso los parametros de paper original pero con 40 variables en lugar de 36:
    F=10,nx=40, ny = 400, c=10, b=10, h=1, # L96 two scale
    dtcy=0.05,dt=0.005,   # integration parameters

    x = [ nx,] nem ]
    '''
    def __init__(self, F=10,nx=40, #model parameters
                 ny = 400, c=10, b=10, h=1, # L96 two scale
                 dtcy=0.05,dt=0.005, #integration parameters
                 **kwargs):
        super().__init__(dt=dt, dtcy=dtcy, **kwargs)

        self.nx = nx + ny # total number of variables for the state
        
        self.nxls = nx
        self.ny = ny
        self.nj  = ny//nx
        self.Fext = F
        self.c = c             # c,b,h small-scale parameters
        self.a = c*b           # 
        self.hint = h*c/b
        
        #indices for model scheme
        k = np.arange(nx)
        self.indm2 = (k - 2) % nx
        self.indm1 = (k - 1) % nx
        self.indp1 = (k + 1) % nx
            
        j = np.arange(ny)
        self.indm1y = (j - 1) % ny
        self.indp1y = (j + 1) % ny
        self.indp2y = (j + 2) % ny
        self.indx2y = j // self.nj
        

    def _mdl(self,x):
        " Two scale L96 model equations"

        xls = x[:self.nxls,...]
        xss = x[self.nxls:,...]

        sumxss = xss.reshape(self.nxls,self.nj,*xss.shape[1:]).sum(axis=1)
        
        self.F = self.Fext - self.hint  *  sumxss        
        dxls = self._mdl_ls(xls)
        
        self.Fls = self.hint * xls
        dxss = self._mdl_ss(xss)
        
        return np.concatenate((dxls,dxss))

    def _mdl_ls(self,x):
        " large scale model equations"
        dx = (x[self.indp1] - x[self.indm2]) * x[self.indm1] - x + self.F
        return dx * self.dt 
    
    def _mdl_ss(self,y):
        " small-scale model equations"
        dy = (self.a * (y[self.indm1y] - y[self.indp2y]) * y[self.indp1y] -
              self.c * y + self.Fls[self.indx2y])
        return dy * self.dt

    def diagnostic_forcing(self, x):
        ' Compute the small-scale forcing   for parameterization '
        xss = x[..., self.nxls:]

        sumxss = xss.reshape(*xss.shape[:-1],
            self.nxls,self.nj,).sum(axis=-1)

        return  - self.hint * sumxss
    
    def forcing_regression(self, x):
        """
        Linear regression of effective forcing against
        F = intercept + slope * X
        """
        xls = x[..., :self.nxls]
        F = self.diagnostic_forcing(x)
        
        return -np.polyfit(xls.ravel(), F.ravel(),1)
    
#----------------------------------------------------------
class L96lforcing(M):
    '''  
    Add a linear forcing to the L96 to mimic the two-scale L96
    '''
    def __init__(self, F=10,nx=40, #model parameters
                 ny = 400, c=10, b=10, h=1, # L96 two scale
                 dtcy=0.05,dt=0.005, #integration parameters
                 **kwargs):        
        super().__init__(dt=dt,dtcy=dtcy, **kwargs ) #dtcy=dtcy,dt=dt)

        if F == 10: # standard case with standard parameters
            a0 = 0.165
            a1 = 0.320
        else:
            nt=100000
            l96_2scl = L96_2scl(F=F,nx=nx,dtcy=dtcy,dt=dt,
                                ny=ny,c=c,b=b,h=h)
            x0,_ = l96_2scl.initialization()

            x_t=np.zeros([nt,x0.shape[0]])
            x_t[0]=x0
            for it in range(1,nt):
                x_t[it]=l96_2scl(x_t[it-1])

            a1,a0 =  l96_2scl.forcing_regression(x_t)
        
        self.dt=dt
        self.nx=nx
        self.F = F 
        
        #indices for model scheme
        k = np.arange(nx)
        self.indm2 = (k - 2) % nx
        self.indm1 = (k - 1) % nx
        self.indp1 = (k + 1) % nx

        self.Ft = self.F - a0
        self.lin = -1.0 - a1
        
    def _mdl(self,x):
        " L96 model equations with forcing x[nx,npa/nem]"
        dx = (x[self.indp1] - x[self.indm2]) * x[self.indm1] + self.lin * x + self.Ft 
        return dx * self.dt
#    def _mdl(self,x):
#        dx = super()._mdl(x)
#        dx += self._forceparam(x) * self.dt
#        return dx
#    
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

#----------------------------------------------------------
if __name__ == '__main__':

    
    #l96 = L96lforcing()
    #l96 = L96()
    l96 = L96_2scl()
    x0,X0 = l96.initialization()

    #
    print(x0.shape)
    
    t0 = time()
    nt=50000
    x_t=np.zeros([nt,x0.shape[0]])
    x_t[0]=x0

#    x_t=np.zeros([nt,X0.shape[0],X0.shape[1]])
#    x_t[0]=X0
    for it in range(1,nt):
        x_t[it]=l96(x_t[it-1])

    #print(x_t)
    #print(l96.forcing_regression(x_t))
             
    print('Tiempo:',time()-t0)
    
    #print(x_t.shape)
    plot_hovmoller(x_t[1000:1500,:40],0.05,title='L96-2scl')#'L96forcing')

    quit()
