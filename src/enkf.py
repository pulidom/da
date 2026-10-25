#!/usr/bin/python3
#
# Ensemble kalman filter 
#    includes perobs, perobs2, etkf, 
#  
# load python modules
import os
import numpy as np
from numpy.linalg import pinv

# my modules
import utils
from utils import meanper, PaWa

class FILTER:
    
    #- 1 -------------------------------------------------    
    def __init__(self, Fmdl,Obs,
                 assmtd='perobs',
                 finf=1.0,
                 seed = 22,
                 ):
        
        self.assmtd = getattr(self, assmtd)    
        self.Hop    = Obs.Hop
        self.Obs    = Obs
        self.R, self.sqR = Obs.R, Obs.sqR
        self.Rinv   = Obs.Rinv      # etkf
        
        self.rng = np.random.default_rng(seed)
        self.randn = self.rng.standard_normal
        
        self.integ  = Fmdl.integ
        
        self.finf =finf
        
        
    #- 2 -------------------------------------------------
    def asscy(self,X0,y_t):
        """
            assimilation cyle.
            Observations are expected at it=1 (xa is at it=0)
        """

        [Nx, Nem] = X0.shape; [Ncy,Ny] = y_t.shape

        # variable initializations
        Xa_t,Xf_t=np.zeros((2,Ncy,Nx,Nem))

        Xa_t[0]=X0 # a priori state at icy=0
        Xf_t[0]=X0

        Xa=np.copy(X0)

        
        # next assimilation cycles
        for icy in range(1,Ncy,1):

            # Evolve forecast ensemble members
            Xf = self.integ(Xa)
            
            Xf_t[icy]=Xf # without inflation
            
            if np.isfinite(y_t[icy]).all(): # NaN observations do not exist
                
               Xf = self.inflate(Xf)
               Xa = self.assmtd(Xf, y_t[icy])
                
            else: # if no obs, forecasts
                Xa=Xf
            
            # save time dependent variables
            Xa_t[icy]=Xa


        # End do it
        return Xa_t,Xf_t
    __call__=asscy

    #- 2 -------------------------------------------------
    def inflate(self, Xf):
        if self.finf > 1.0:
            xm = Xf.mean(1, keepdims=True)
            Xf = xm + self.finf * (Xf - xm)
        return Xf

        #- 2 -------------------------------------------------
    def perobs(self, Xf, yo):
        """Assimilate using classical EnKF with perturbed observations."""
        Nx, Nem = Xf.shape
        Ny = yo.shape[0]
        H   = self.Obs.jacobian(Xf.mean(axis=1))
        R, sqR = self.R, self.sqR

        Pfxx = np.cov(Xf)

        HPf = H @ Pfxx                     # (Ny, Nx)
        S = HPf @ H.T + R        # (Ny, Ny) innovation covariance

        # Classical Kalman gain
        # K = Pfxx @ H.T @ inv(S) = HPf.T @ inv(S)
        K = np.linalg.solve(S, HPf).T      # (Nx, Ny)

        Y = H @ Xf + sqR @ self.randn( (Ny, Nem) )
        innov = yo.reshape(Ny, 1) - Y      # (Ny, Nem)

        Xa = Xf + K @ innov
        return Xa
    
    def perobs2(self, Xf, y):
        """EnKF con observaciones perturbadas, usando solo Hop (H no lineal permitido).
        Xf [Nx,Nem], y [Ny]"""
        Nem = Xf.shape[1]; Ny = y.shape[0]
        Hx = self.Hop(Xf)                                   # (Ny, Nem)

        xfm, Xper = meanper(Xf)
        hxm, Yper = meanper(Hx)

        D = Yper @ Yper.T + self.R * (Nem - 1)              # (Ny, Ny), simétrica
        M = np.linalg.solve(D, Yper).T                      # Yper^T D^-1, (Nem, Ny)

        innov = y[:, None] - Hx + self.sqR @ self.randn((Ny, Nem))

        if Nem > 2 * Ny:                                    # ensamble grande
            return Xf + (Xper @ M) @ innov
        return Xf + Xper @ (M @ innov)
    

 
#    #- 2 -------------------------------------------------
#    def perobs2(self,Xf,y):
#        """ Observaciones perturbadas  EnKF
#             Evita calculo de la covarianza
#             Usa diferencias del operador observacional 
#                (suitable para nonlinear map)
#             Xf [Nx,Nem],y [Ny]
#             implicit: H, sqR, R
#             \[ X^a = X^f + X'  [ \mathcal{H}'(x) ]
#                 [ \mathcal{H}'(x)^\tau \mathcal{H}'(x) + R ]^{-1} 
#                 ( y- \mathcal{H}(x) ) \]
#        """
#
#        
#        nem=Xf.shape[1]; ny=y.shape[0]
#        Hx = self.Hop(Xf) # # Observational map Ny,Nem
#        
#        xfm,Xper = meanper(Xf)        
#        hxm,Yper = meanper(Hx)
#        R = self.sqR.T @ self.sqR
#
#        D = Yper @ Yper.T + R * (nem-1)
#
#        K = Xper @ mrdiv(Yper.T,D) 
#        Noise = self.sqR @ self.randn((ny, nem))
#        Xa = Xf + K @ (y[...,np.newaxis] - Hx + Noise)
#
#        # Alternativa colapsar al ny? primero
#        #Noise = self.sqR @ rnd.normal(0,1,[ny,nem])
#        #w = self.mrdiv(Yper.T,D) @ (y[...,np.newaxis] - Hx + Noise)
#        #Xa = Xf + Xper @ w
#        
#        return Xa
    
    #- 3 -------------------------------------------------
    def etkf(self,Xf,yo):
        " Assimilate using Hunt et al, etkf, square root  enkf"
        
        Nem = Xf.shape[1]
        Rinv= self.Rinv
        
        Yf = self.Hop(Xf) # Observational map

        # mean and perturbations
        xfm,Xper = meanper(Xf)        
        hxfm,Yper =meanper(Yf)

        C = Yper.T @ Rinv
        B = (Nem-1.)*np.eye(Nem)+ C @ Yper

        Pa , Wa = PaWa ( B ) # inverse and square root via EIG

        innov = yo[...,np.newaxis] - hxfm
        wa = Pa @ (C @ innov)

        Xa = xfm + Xper @ (wa+Wa)
        return Xa

#------------------------------------------------------
def mrdiv(b,A):
  """ Matrix right division  K= P Q^-1  --> K=mrdiv(P,Q)
  
  """
  return np.linalg.solve(A.T,b.T).T
    
if __name__=="__main__":
    
    from dyn import L96 as mdl
    import src.obs as obs
    
    Nx = 40
    Ny = 20
    Mdl= mdl(nx=Nx,expdir='../dat/l96_40v/')

    xt0,X0 = Mdl.initialization(nem=30)

    
    #H=np.eye(Nx)
    indobs = utils.obs_points(Nx, Ny)
    Obs    = obs.OBS(0.5*np.eye(Ny),indobs)
    xt_t,y_t= Obs(xt0,1200,Mdl)
    from time import time
    t0=time()
    finf=np.linspace(1.0,1.1,25)
    for i in range(25):
        Filter=FILTER(Mdl,Obs,finf=finf[i],assmtd='etkf')#'perobs2')#'etkf')
        Xa_t,Xf_t=Filter(X0,y_t)
        print(finf[i])
        err=xt_t-Xa_t.mean(-1)
        print('RMSE: ',utils.RMSE(err[200:]))
    print(time()-t0)
        
    #gra.assm([0,1,2],Xa_t,xt=xt_t,indobs=indobs)
