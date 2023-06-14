#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 23:25:28 2023

@author: gustav
"""

import emcee 
import numpy as np
import seaborn as sns
from scipy.stats import norm, uniform, truncnorm
import data
from lib.gp import gp_ensemble
import matplotlib.pyplot as plt
from corner import corner
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

#%%

data_obj = data.dakota_data()
data_obj.process()

#%%

scaler = StandardScaler()

Xtrain = data_obj.Xtrain.values
Xtest = data_obj.Xtest.values

ytrain = data_obj.ytrain.values
ytest = data_obj.ytest.values

scaler.fit(Xtrain)

Xtrain_scaled = scaler.transform(Xtrain)

meas_v = data_obj.meas_v

#%%

gp = gp_ensemble(
    Xtrain = scaler.transform(Xtrain),
    ytrain = data_obj.ytrain.values,
    )   

gp.fit()

#%%

plt.plot(ytest,gp.predict_fast(scaler.transform(Xtest)),'+')
plt.show()

#%%

ndim = Xtrain.shape[-1]
nexp = meas_v.shape[-1]

nsteps = 1000

#Initial locs and scales
locs = np.zeros(ndim)
scales = np.ones(ndim)

#Get the boudaries of the scaled parameters
Xtrain_scaled_min = Xtrain_scaled.min(axis=0)
Xtrain_scaled_max = Xtrain_scaled.max(axis=0)

#As and bs
a_s = (Xtrain_scaled_min - locs) / scales
b_s = (Xtrain_scaled_max - locs) / scales

#Struct dist
struct_dist = truncnorm(a=a_s,b=b_s,loc=locs,scale=scales)

#Random log numbers
log_us = np.log(uniform().rvs((nsteps,nexp,ndim)))

#Proposed steps
delta_X = norm().rvs((nsteps,nexp,ndim))

#All local parameters
Xs = np.zeros((nexp,ndim))

#Likelihood
like = norm(loc=meas_v,scale=0.01)

#Caclculate the current logp of the likelihoods 
logp_Xs = norm.logpdf(
    [gp.predict_i(Xs[i][None,:],i) for i in range(nexp)]
    ).flatten()

#Add the logp of the struct dist
logp_Xs += struct_dist.logpdf(Xs).sum(axis=1)

#Loop the number of steps
for i_step in range(nsteps):
    
    #Loop the experiments
    for i_exp in range(nexp):
        
        #Loop the dimensions
        for i_dim in range(ndim):
            
            #Get a log random number
            log_u = log_us[i_step,i_exp,i_dim]
            
            #Get X vector for current experiment
            X = Xs[i_exp]
            
            print(X)
            
            
            #Get x
            pass