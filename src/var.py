''' Assimilacion variacional:  3dvar, 4dvar y OI
'''
import sys; sys.path.insert(0, '../')
import os
import numpy as np
import numpy.random as rnd
from numpy.linalg import pinv
from numpy.linalg import cholesky as sqrtm
import scipy.optimize as so
from scipy.optimize import minimize

class DA:
    def __init__(self,Mdl,Obs,
                 #assmtd='oi',
                 ):

        self.Mdl=Mdl # modelo con el que vamos a hacer los pronosticos
        self.Obs=Obs # objeto de las observaciones
        #self.assmtd=eval('self.'+assmtd)
        
    #--------------------------------------------
    def assm(self,Xa,y_t):
        """
            assimilation cycles.
            Observations are expected at it=1 (xa is at it=0)
            Xa_t, Xf_t Outputs are all at it=1
        """

        [nx] = Xa.shape; [ncy, ny] = y_t.shape
        Xa_t,Xf_t=np.zeros((2,ncy,nx))

        print ('Assimilating ...')
        
        # assimilation cycles
        for icy in range(ncy):

            # Evolve forecast state/ensemble members
            Xf = self.Mdl(Xa)
            
            # Assimilate
            Xa = self.assmtd(Xf,y_t[icy])
            
            # save time dependent variables
            Xa_t[icy] = Xa
            Xf_t[icy] = Xf
            
        return Xa_t,Xf_t
    __call__=assm

class OI(DA):
    ''' Optimal interpolation  '''
    def __init__( self, *args,Pb= None,
                  **kwargs):
        super().__init__( *args,**kwargs ) # parameters for the parent DA class
        self.Pb=Pb
        
    def assmtd(self,xf,y):
        """ Method for optimal interpolation
         """
        H=self.Obs.H
        Pb= self.Pb
        innov=y-H @ xf
        
        # Classical Kalman gain
        K = (Pb @ H.T) @ pinv(H @ Pb @ H.T + self.Obs.R)
           
        return xf + K @ innov

class VAR4D:
    ''' Strong-constraint 4D-Var: one assimilation window per model cycle.

    Mdl.dtcy, Mdl.kt steps is the assimilation window. 
    The control variable xa is the state at the START of window, 
    Mdl.ntobs observations inside the window

    Shapes:  y_t: (ncy, ntobs, ny)      Xa_t, Xf_t: (ncy, nx)
      Xa_t[k] : analysis, state at the start of cycle k (same index as y_t[k])
      Xf_t[k] : Mdl(Xa_t[k]), state at the end of cycle k = background of cycle k+1
    '''
    def __init__(self, Mdl, Obs, Pb,
                 opt_mtd='grad_desc',
                 nobs_window=1,
                 ):

        self.Mdl = Mdl
        self.Obs = Obs

        self.invPb = pinv(Pb)
        self.opt_mtd = opt_mtd
        self.nobs_win = nobs_window

    def asscy(self, xb, y_t):
        """
            assimilation cycles.
        """
        nx, = xb.shape; ncy, ntobs, ny = y_t.shape
        Xa_t, Xf_t = np.zeros((2, ncy, nx))

        print('Assimilating ...')
        for icy in range(ncy):

            # Assimilate: analysis at the start of the cycle (window)
            xa = self.assmtd(xb, y_t[icy])
            Xa_t[icy] = xa

            # evolve the analysis through the cycle -> background of the next one
            xf = self.Mdl(xa)
            Xf_t[icy] = xf
            xb = xf

        return Xa_t, Xf_t

    def gradJ(self, xa):
        gradJback = self.invPb @ (xa - self.xb)
        innov = self.y - self.Obs.Hop(xa)
        gradJobs = -self.Obs.Htop(xa, innov @ self.Obs.invR)
        return gradJback + gradJobs
    
    def fnJ(self, xa):
        ''' Cost function '''

        Jback = (xa - self.xb).T @ (self.invPb @ (xa - self.xb))
        innov = self.y - self.Obs.Hop(xa)
        Jobs = np.sum((innov @ self.Obs.invR) * innov)
        J = 0.5 * (Jback + Jobs)
        
        return J

    def grad_desc(self, x0,       # initial  state
                  Jtol=1e-3,      # tolerancia para cortar las iteraciones
                  alfa=0.01,      # tasa de aprendizaje
                  max_iter=300,   # numero de iteracionees
                  ):

        xa = np.copy(x0)
        Jold = 1.e99
        J = self.fnJ(xa)
        i = 0

        while (i < max_iter) and (np.abs(J - Jold) > Jtol):
            xa = xa - alfa * self.gradJ(xa)
            Jold = J
            J = self.fnJ(xa)
            i += 1
        return xa

    def assmtd(self, xb, y):

        self.xb = xb
        self.y = y
        if self.opt_mtd == 'grad_desc':
            xa = self.grad_desc(xb)
        else:
            res = so.minimize(self.fnJ, xb,
                              method='L-BFGS-B',
                              jac=self.gradJ,
                              options={'maxiter': 30})
            xa = res['x']
        return xa
    __call__ = asscy
    
class VAR3D(DA):
    ''' Asimilación variacional 3D '''
    def __init__(self, *args, Pb=None, opt_mtd='bfgs', **kwargs):
        super().__init__(*args, **kwargs)
        self.invPb = inv(Pb)
        self.opt_mtd = opt_mtd
        

    def gradJ(self, xa):
        xa = np.asarray(xa).ravel()
        innov = self.y - self.Obs.Hop(xa)
        return self.invPb @ (xa - self.xb) - self.Obs.Htop(xa, self.Obs.invR @ innov)

    def fnJ(self, xa):
        xa = np.asarray(xa).ravel()
        innov = self.y - self.Obs.Hop(xa)
        dx = xa - self.xb
        return 0.5 * (dx @ self.invPb @ dx + innov @ self.Obs.invR @ innov)

    def grad_desc(self, x0, gtol=1e-6, alfa=0.02, max_iter=1000):
        xa = np.array(x0, dtype=float)
        for i in range(max_iter):
            g = self.gradJ(xa)
            if np.linalg.norm(g) < gtol:
                break
            xa = xa - alfa * g
        return xa

    def assmtd(self, xb, y):
        self.xb = np.asarray(xb).ravel()
        self.y = np.asarray(y).ravel()
        if self.opt_mtd == 'grad_desc':
            return self.grad_desc(self.xb)
        res = minimize(self.fnJ, self.xb, jac=self.gradJ,
                       method='BFGS', options={'gtol': 1e-8})
        return res.x
    
def nmc_method(Mdl,xt0,ncy=5000,cov_fac=0.05):
    """ NMC: Determine background covarince comparing two forecast times """
    def gen_traj(x0, Ncy, Tmdl):
        x1 = np.copy(x0)
        xt_t = np.empty((Ncy, x0.shape[0]))
        
        for ic in range(Ncy):
            xt_t[ic] = x1   
            x1 = Tmdl.integ(x1)             

        return xt_t
    xt_t = gen_traj(xt0,ncy,Mdl)
    Pf0 =  np.cov(xt_t.T) #climatology cov
    # NMC  predictions to 0.2 
    # first predictions
    sqPf0 = cov_fac * sqrtm(Pf0) 
    wrk = rnd.normal(0, 1, xt_t.shape)
    xa = xt_t + wrk @ sqPf0

    x_t = np.zeros( (3,) + xa.shape)
    x_t[0]= xa
    for it  in range(1,3):        
        x_t[it]=Mdl(x_t[it-1].T).T # los tiempos como un ensamble

    x6=x_t[1,1:]
    x12=x_t[2,:-1]
    Pf_nmc = np.cov((x12-x6).T)
    return Pf_nmc

#----------------------------------------------------------
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from src.dyn import L63b, L63
    from src.obs import OBS
    
    Mdl=L63b() # instancio la clase
    xt0,X0= Mdl.initialization(nem=1)

    Obs=OBS(Rdiag=True,Rfac=9, Ntobs=2,Nx=3,Ny=3, Mdl=Mdl)
    Pb = nmc_method(Mdl,xt0,cov_fac=0.1)
    #Pb = np.array([[ 1.99201541e+00,  7.16884049e-01,  3.18082236e-02],
    #   [ 7.16884049e-01,  1.99201541e+00, -1.71903899e-03],
    #   [ 3.18082236e-02, -1.71903899e-03,  1.37779395e+00]])

    ''' 
    El mejor resultado me esta dando con grad_desc y 300 iteraciones (mejor que 'L-BFGS-B')
    RMSE:  0.8195400579718487
    RMSE 0:  0.7168689690740385
    RMSE 1:  0.807753788465286
    RMSE 2:  0.9211788199151164
    '''
    Assm=VAR4D(Mdl,Obs, Pb=Pb, nobs_window=2, opt_mtd='s')#'grad_desc')
    xt_t, y_t = Obs(xt0,1000,Mdl)
    xa_t,xf_t = Assm(X0[:,0],y_t)
    
    
    print('RMSE: ',np.mean((xa_t-xt_t)**2)**.5)
    for i in range(3):
        print(f'RMSE {i}: ',np.mean((xa_t[:,i]-xt_t[:,i])**2)**.5)
    def plot(xa_t,y_t,xt_t,file='../tmp/4dvar1.png'):
        time=Mdl.dtcy * np.arange(xa_t.shape[0])
        xt_g=xt_t.T;xa_g=xa_t.T; y_g=y_t.T

        fig,ax = plt.subplots(3,1,figsize=(10, 5))
        ax[0].plot(time,xt_g[0],'k')
        ax[0].plot(time,xa_g[0])
        #ax[0].plot(time,y_g[0],'.')
        ax[0].set(ylabel=r'$x$')
        ax[1].plot(time,xt_g[1],'k')
        ax[1].plot(time,xa_g[1])
        #ax[1].plot(time,y_g[1],'.')
        ax[1].set(ylabel=r'$y$')
        ax[2].plot(time,xt_g[2],'k')
        ax[2].plot(time,xa_g[2])
        #ax[2].plot(time,y_g[2],'.')
        ax[2].set(ylabel=r'$z$')
        fig.tight_layout()
        plt.savefig(file)
        

    xa1_t=xa_t[0:400]
    y1_t=y_t[0:400]
    xt1_t=xt_t[0:400]
    plot(xa1_t,y1_t,xt1_t,file='../tmp/4dvar1.png')
    xa1_t=xa_t[400:]
    y1_t=y_t[400:]
    xt1_t=xt_t[400:]
    plot(xa1_t,y1_t,xt1_t,file='../tmp/4dvar2.png')
