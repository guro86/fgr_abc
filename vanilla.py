#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:58:15 2022

@author: gustav
"""


import emcee 
from scipy.stats import norm, uniform
import pickle
import emcee 
import numpy as np
import seaborn as sns
from corner import corner
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool

#%%
with open('gp.p','rb') as f:
    gp, data_obj = pickle.load(f)

lb = data_obj.Xtrain.min()
ub = data_obj.Xtrain.max()


#%%
like = norm(loc=data_obj.meas_v,scale=0.01)
prior = uniform(loc = lb, scale = ub - lb)

#%%


def pred(params,pos,default):
    
    X = default
    X[:,pos] = params
    
    pred = gp.predict(X[None,:]) 
    
    return pred


def logp(params,pos,default):
    
    X = default
    X[pos] = params
    
    pred = gp.predict(X[None,:]) 
    
    logp = np.sum(prior.logpdf(X))
    logp += np.sum(like.logpdf(pred))
    
    return logp
    

#%%

default = np.ones(5)*.99

pos = np.array([0,1,4])

ndim = len(pos)
nwalkers = ndim*2

print(data_obj.Xtrain.columns[pos])

#%%

with Pool() as pool:
    
    sampler = emcee.EnsembleSampler(
        nwalkers = nwalkers, 
        ndim = ndim, 
        log_prob_fn = logp, 
        pool=pool,
        args=[pos,default]
        )
    
    nsteps = 2000
    
    initial_state = np.random.randn(nwalkers,ndim)  * 0.01 + .5
    
    sampler.run_mcmc(initial_state, nsteps, progress=True)


#%%

chain = pd.DataFrame(
    sampler.get_chain(flat=True,discard=500),
    columns = data_obj.Xtrain.columns[pos]
    )

corner(chain)

#%%

pred_all = pred(chain.values,pos,default)

#%%

idxmax = sampler.get_log_prob(flat=True).argmax()
be_reds = sampler.get_chain(flat=True)[idxmax]

be = np.ones(5)
be[pos] = be_reds


#%%
l= np.linspace(0,.4,2)
#plt.errorbar(data_obj.meas_v,pred_all.mean(axis=0),yerr=pred_all.std(axis=0),fmt='+')
plt.plot(l,l)
plt.plot(data_obj.meas_v,gp.predict(np.ones(5)[None,:]).flatten(),'o',
         label='Before calibration')
plt.plot(data_obj.meas_v,gp.predict(be[None,:]).flatten(),'+',
         label='After calibration'
         )

plt.legend()

#plt.xscale('log')
#plt.yscale('log')

plt.xlabel('Measured relative fission gas release[-]')
plt.ylabel('Predicted relative fission gas release[-]')

#%%

be_pred = gp.predict(be[None,:]).flatten()
res = be_pred - data_obj.meas_v

plt.hist(res)