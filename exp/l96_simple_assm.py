import sys;  sys.path.insert(0, '../src/'); 
import numpy as np
import dyn
import obs
import gra
import enkf as fil
import utils

ncy    = 1200
ispin  = 200
expdir = '../dat/L96_40v/001/' #L96_40v/'

dtcy = 0.05
Rfac = 0.5
Nem  = 200
Ny = 20
m_kwargs = { 'dtcy':dtcy,'expdir':expdir,'Nem':Nem }
o_kwargs = {'R':Rfac*np.eye(Ny),'Nx':40,'Ny':Ny,
            'dat_fname':expdir+'trueobs',
            'obs_type':'uniform','Ncy':Ncy}
a_kwargs = { 'assmtd':'etkf','finf':1.05}

Mdl    = dyn.L96(**m_kwargs)
xt0,X0 = Mdl.initialization()


print('entra gen obs')

# Generate true state and observations
Obs = obs.OBS( Tmdl=Mdl, **o_kwargs)
xt_t, y_t = Obs( xt0 )


print('entra filtro')
Filter = fil.FILTER(Mdl,Obs, **a_kwargs)

# Assimilate 
Xa_t, Xf_t = Filter(X0,y_t)

rmse = utils.RMSE(xt_t[ispin:] - Xa_t[ispin:].mean(-1))

print('RMSE:',rmse)
#save
np.save(expdir+f'/anaL96per_ny{Ncy}_dtcy{dtcy}_R{Rfac}_inf{finf}_nem{Nem}.npz',
        xt=xt_t,y=y_t,xa=Xa_t,xf=Xf_t)

gra.plot_hovmoller(x_t[1000:1500,:40],0.05,filename='../tmp/hovmoller.png')
ini=800;end=1000
gra.assm([0,1,2],Xa_t[ini:end],y=y_t[ini:end], xt=xt_t[ini:end], indobs=Obs.obs_indices,
         filename='../tmp/assm_L96.png')
