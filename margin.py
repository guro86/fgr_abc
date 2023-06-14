#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:58:24 2022

@author: gustav
"""

#Import modules
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gamma, norm, lognorm 
from scipy.stats import norm, halfcauchy, uniform
import emcee
from corner import corner
from multiprocessing import Pool
from scipy.stats import multivariate_normal
import pandas as pd
from lib import margin
from lib import model_wrapper
import data
from lib.gp import gp_ensemble
from lib.transforms import range_tanh, logit

#%%

data_obj = data.dakota_data()
data_obj.process()

#Get the Xtrain and Xtest 
Xtrain = data_obj.Xtrain.values
Xtest = data_obj.Xtest.values

#measurements 
meas_v = data_obj.meas_v

trans = logit()
trans.fit(Xtrain)

#Create a gp
gp = gp_ensemble(
    Xtrain = trans.transform(Xtrain),
    ytrain = data_obj.ytrain.values,
    )   

#Fit the gp
gp.fit()

#%%

sns.pairplot(trans.transform(data_obj.Xtrain))

#%%
plt.plot(data_obj.ytest,
         gp.predict_fast(trans.transform(Xtest)),
         '+'
         )

#%%
        
rgp = model_wrapper(
    model=gp,
    pos = np.array([0,1,4]),
    default = np.array([0,0,1.8,-1.05,0])
    )
   
    
#%%


x0 = np.concatenate((
    np.array([-1.5,-.4,0.1]),
    np.ones(3)*.1,
    np.zeros(3)
    ))

ub = data_obj.Xtrain.quantile(q=0.99).values[rgp.pos]
lb = data_obj.Xtrain.quantile(q=0.01).values[rgp.pos]


#sd = np.array([0.12, 0.17, 0.67])
#sd = np.array([0.27894827, 1.03073904, 0.03689269])
sd = np.array([0.03534472, 0.13215198, 0.07650687])

#mu = np.array([0.6,1.25,0.67])
#mu = np.array([2.10743787, 3.231398  , 0.46852582])
mu = np.array([-1.54125814, -0.40837107,  0.12749811])

mu_prior = norm(loc=mu,scale=sd)
sd_prior = halfcauchy(scale=.1)

m = margin(
    d=3,model=rgp,meas_v=data_obj.meas_v,mu_prior=mu_prior,sd_prior=sd_prior,
    R=data_obj.meas_v * 0.01 + 0.01
    )

m.update(x0)

m.logp()

#%%

l = np.linspace(0,.4,2)
yerr = (m.cov_y+m.R**2)**.5
plt.errorbar(m.meas_v,m.pred,fmt='+',yerr=yerr)
plt.plot(l,l)

#%%
l = np.linspace(0,.4,2)

plt.plot(m.meas_v,m.pred,'o')
plt.plot(l,l)

plt.xlabel('Measured fission gas release [-]')
plt.ylabel('Predicted fission gas release [-]')

plt.xscale('log')
plt.yscale('log')

#%%

nwalkers = 18
ndim = 9

sampler = emcee.EnsembleSampler(
    nwalkers, 
    ndim, 
    m.update_and_return_logp,
    )

initial_state = np.random.randn(nwalkers,ndim)*0.01 + x0

sampler.run_mcmc(
    initial_state, 
    nsteps=2000,
    progress=True,
    )

#%%
corner(sampler.get_chain(flat=True),discard=500,dpi=300)

#%%
idxmax = sampler.get_log_prob(flat=True).argmax()
x = sampler.get_chain(flat=True)[idxmax]

m.update(x)

l = np.linspace(0,.4,2)
yerr = (m.cov_y+0.01**2)**.5
plt.errorbar(m.meas_v,m.pred,fmt='+',yerr=yerr)
plt.plot(l,l)

plt.xlabel('Measured fission gas release [-]')
plt.ylabel('Predicted fission gas release [-]')

#%%

corner(sampler.get_chain(flat=True),discard=500,dpi=300,truths=x)

#%%

chain = pd.DataFrame(
    sampler.get_chain(discard=500,flat=True)
    )

def sim(x):
    m.update(x.values)
    
    return pd.Series(
        multivariate_normal(m.mu,m.cov,allow_singular=True).rvs()
        )
   
    # return np.linalg.det(m.cov)

marg = chain.iloc[:].apply(sim,axis=1)

marg.columns = ['diff', 'gb_saturation', 'crack']

#%%

corner(
       marg,
       # range=[(0,10),(0,10),(0,10)]
)

#%%
Xnew = np.ones((len(marg),5))
Xnew[:] = rgp.default
Xnew[:,rgp.pos] = marg.values

#%%

marg_trans = \
    trans.inverse_transform(Xnew)[:,rgp.pos]

corner(
       marg_trans,
       # range=[(0,10),(0,10),(0,10)]
)

#%%

chain_y = rgp.predict_chain(marg)

#%%

idxmax = sampler.get_log_prob(flat=True).argmax()
x = sampler.get_chain(flat=True)[idxmax]

m.update(x)

q = np.quantile(chain_y,[0.05,0.95,.5],axis=0)

yerr = np.abs(q[-1] - q[:2])

l = np.linspace(0,.4,2)
plt.plot(l,l,'--')
plt.errorbar(data_obj.meas_v,q[-1,:],yerr=yerr,fmt='o',label='Monte Carlo')
#plt.plot(data_obj.meas_v,q[-2,:])

yerr = 1.64*(m.cov_y+0.01**2)**.5
plt.errorbar(m.meas_v,m.pred,fmt='+',yerr=yerr,label='Sandwich')

plt.legend()

plt.xlabel('Measured fission gas release [-]')
plt.ylabel('Predicted fission gas release [-]')

#%%


idxmax = sampler.get_log_prob(flat=True).argmax()
x = sampler.get_chain(flat=True)[idxmax]

m.update(x)

q = np.quantile(chain_y,[0.05,0.95,.5],axis=0)

yerr = chain_y.std(axis=0)

l = np.linspace(0,.4,2)
plt.plot(l,l,'--')
plt.errorbar(data_obj.meas_v,q[-1,:],capsize=5,
             yerr=yerr,fmt='o',label='Monte Carlo')
#plt.plot(data_obj.meas_v,q[-2,:])

yerr = (m.cov_y+0.01**2)**.5
plt.errorbar(m.meas_v,m.pred,fmt='o',yerr=yerr,label='Sandwich',
             capsize=5)

plt.legend()

plt.xlabel('Measured fission gas release [-]')
plt.ylabel('Predicted fission gas release [-]')

#%%

plt.plot(m.cov_y**.5,chain_y.std(axis=0),'o')

plt.ylabel('Monte Carlo std.')
plt.xlabel('Sandwich std.')

l = np.linspace(0,0.06,2)
plt.plot(l,l,'--')


#%%

idxmax = sampler.get_log_prob(flat=True).argmax()
x = sampler.get_chain(flat=True)[idxmax]

m.update(x)

q = np.quantile(chain_y,[0.05,0.95,.5],axis=0)

yerr = np.abs(q[-1] - q[:2])

l = np.linspace(0,.4,2)
plt.plot(l,l,'--')
plt.errorbar(data_obj.meas_v,q[-1,:],
             alpha = .2,
             yerr=yerr,fmt='o',label='Monte Carlo')

plt.plot(data_obj.meas_v,q[0,:],'o')

plt.plot(data_obj.meas_v,q[1,:],'o')

plt.legend()

plt.xlabel('Measured fission gas release [-]')
plt.ylabel('Predicted fission gas release [-]')


#%%
plt.plot(l,l,'--')
plt.plot(data_obj.meas_v,q[1],'o')