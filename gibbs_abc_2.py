#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:30:56 2023

@author: robertgc
"""

import data
import numpy as np
from lib.gp import gp_ensemble
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.stats import norm, truncnorm, uniform
from tqdm import tqdm
import emcee
from corner import corner
from scipy.stats import mode

#%%

#load the data
data_obj = data.dakota_data()
data_obj.process()

#%%

#Get the Xtrain and Xtest 
Xtrain = data_obj.Xtrain.values
Xtest = data_obj.Xtest.values

#measurements 
meas_v = data_obj.meas_v

#%%

#Fit a scaler 
scaler = StandardScaler()
scaler.fit(Xtrain)

#Get scaled X data
Xtrain_scaled = pd.DataFrame(
    scaler.transform(Xtrain),
    columns=data_obj.Xtrain.columns
    )

#%%

#Just present a pairplot
sns.pairplot(
    pd.DataFrame(scaler.transform(data_obj.Xtrain))
    )

#%%

#Create a gp
gp = gp_ensemble(
    Xtrain = scaler.transform(Xtrain),
    ytrain = data_obj.ytrain.values,
    )   

#Fit the gp
gp.fit()

#%%
plt.plot(data_obj.ytest,
         gp.predict_fast(scaler.transform(Xtest)),
         '+'
         )

#%%

def pred_dim(X):
    
    #Make the predictions for all experiments
    pred_d = np.array(
        [gp.predict(X[i].reshape(-1,5)) for i,gp in enumerate(gp.gps)]
        )
    
    return pred_d

#%% Vanilla

nwalkers = 10
ndim = 5

like = norm(loc=meas_v,scale=0.001 + .1 * meas_v)

prior = uniform(
    loc = Xtrain_scaled.min(),
    scale = Xtrain_scaled.max() - Xtrain_scaled.min()
    )

def log_prob_fn(x):
    
    pred = gp.predict_fast(x[None,:]).flatten()
    
    logp = np.sum(
        like.logpdf(pred)
    )
    
    logp += np.sum(
        prior.logpdf(x)
        )
    
    if np.isnan(logp):
        logp = -np.inf
    
    return logp

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn)

initial_state = norm(loc=0,scale=0.1).rvs((nwalkers,ndim))
nsteps = 5000

state = sampler.run_mcmc(initial_state, nsteps,progress = True)

#%%

chain = pd.DataFrame(
    scaler.inverse_transform(sampler.get_chain(flat=True,discard=2000)),
    columns=data_obj.Xtrain.columns
    )

corner(chain)

#%%

chain = sampler.get_chain(flat=True,discard=2000)
logp = sampler.get_log_prob(flat=True,discard=2000)

be = chain[logp.argmax()]
print(be)

#%%

#Xmax and Xmin
Xmin = Xtrain_scaled.min().values
Xmax = Xtrain_scaled.max().values

#Number of experiments and dimensions
nexp = len(data_obj.meas_v)
ndim = Xtrain.shape[-1]

loc_dist = uniform(loc=Xmin,scale=Xmax-Xmin)
scale_dist = uniform(loc=np.zeros(5),scale=2)

loc = loc_dist.rvs(ndim)
# loc = be
scale = scale_dist.rvs(ndim)

a = (Xmin - loc) / scale
b = (Xmax - loc) / scale

#Dist type to start with
dist = truncnorm(a=a,b=b,loc=loc,scale=scale)

#Initialization of X
X = dist.rvs((nexp,ndim))

#Samples to test
nsamp = 500

#Measument noise
meas_unc = norm(loc=0,scale=0.001 + .1 * meas_v)

#%%

l = np.linspace(0,.4,2)

pred = gp.predict_fast(dist.rvs(1,5))

q = np.quantile(
    gp.predict_fast(dist.rvs((2000,5))),
    [0.01,0.99],
    axis=0
    )

yerr = np.abs(q-pred)

plt.errorbar(meas_v,pred,yerr=yerr,fmt='o')
plt.plot(meas_v,q[0],'o')


plt.plot(l,l)

plt.title('Prior')
plt.xlabel('Measured fission gas release [-]')
plt.ylabel('Predicted fission gas release [-]')


#%%

#Initialization of X
X = dist.rvs((nexp,ndim))

N = 1000

scales = np.empty((N,ndim)) 
locs = np.empty((N,ndim)) 

Xs = np.empty((N,nexp,ndim))

for i in tqdm(range(N)):
    
    #Canidate samples
    X_cand = dist.rvs((nsamp,ndim))
    
    #Candidated evaluation samples 
    #nexp x nsamp x ndim
    X_eval_cand = np.ones(nexp)[:,None,None] * X_cand
    
    #Base evaluation samples with out the dth column inserted
    #nexp x nsamp x ndim
    X_eval_base = np.ones(nsamp)[:,None] * X[:,None,:]
    
    for d in np.arange(ndim):
            
        #For dimension d, take base evaluation samples
        X_eval_d = X_eval_base
        
        #Insert candidates
        X_eval_d[:,:,d] = X_eval_cand[:,:,d]
        
        #Make the predictions for all experiments
        pred_d = np.array(
            [gp.predict(X_eval_d[i]) for i,gp in enumerate(gp.gps)]
            )
        
        #Measurement ucn
        pred_d_e = meas_unc.rvs((nsamp,nexp)).T
        
        #Get the idx where the distance is smallest
        idx = np.argmin(
            np.abs(pred_d + pred_d_e - meas_v[:,None]),
            axis=1
            )
        
        #Update X
        X[:,d] = X_cand[idx,d]
        

    scale_cand = scale_dist.rvs((nsamp,ndim))
    
    loc_cand = loc_dist.rvs((nsamp,ndim))
        
    a_cand = (Xmin[None,:] - loc) / scale_cand
    b_cand = (Xmax[None,:] - loc) / scale_cand
    
    X_cand = truncnorm(
        a = a_cand,
        b = b_cand,
        loc = loc_cand,
        scale = scale_cand
        ).rvs((nsamp,ndim))
    
    diff = np.abs(X_cand * np.ones(nexp)[:,None,None] - X[:,None,:])
    
    idx = np.argmin(diff.sum(axis=0),axis=0)
    
    scale = scale_cand[idx,np.arange(ndim)]
    loc = loc_cand[idx,np.arange(ndim)]
          
    
    a = (Xmin - loc) / scale
    b = (Xmax - loc) / scale
    
    dist = truncnorm(a=a,b=b,loc=loc,scale=scale)
    
    scales[i] = scale
    locs[i] = loc
    Xs[i] = X
    
#%%
l = np.linspace(0,.4,2)

plt.plot(meas_v,pred_dim(X),'o')
plt.plot(l,l)

#%%

l = np.linspace(0,.4,2)


a = (Xmin - locs) / scales
b = (Xmax - locs) / scales

X = truncnorm(a=a,b=b,scale=scales,loc=locs).rvs()

X = X[000:]

pred_be = gp.predict(X).mean(axis=0)

pred_be = gp.predict(
    np.median(X,axis=0)[None,:]
    ).flatten()

q = np.quantile(
    gp.predict(X),
    [0.05,0.95],
    axis=0
    )

yerr = np.abs(q-pred_be)


# plt.errorbar(meas_v ,pred_be,yerr=yerr,fmt='o')
plt.plot(meas_v,pred_be,'o')
# plt.plot(meas_v,q[-1],'o')
# plt.plot(meas_v,q[0],'o')

plt.plot(l,l)

plt.title('Posterior')
plt.xlabel('Measured fission gas release [-]')
plt.ylabel('Predicted fission gas release [-]')
