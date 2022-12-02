#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 21:27:08 2022

@author: robertgc
"""

import numpy as np
from scipy.stats import norm, gamma, beta
import matplotlib.pyplot as plt 
import pickle 
from tqdm import tqdm
import pandas as pd
from corner import corner 

#%%
with open('gp.p','rb') as f:
    stored = pickle.load(f)
    
    gp = stored['gp']
    data_obj = stored['data_obj']
    trans = stored['trans']
    
#%%
    
#Numer of iterations sampler
N = 2000

#Numer of samples for eps 
N_eps = 59


mu_prior_scale = np.array([0.03534472, 0.13215198, 0.07650687])
mu_prior_loc = np.array([-1.54125814, -0.40837107,  0.12749811])

#Prior for mu 
mu_prior = norm(
    loc = mu_prior_loc,
    scale = mu_prior_scale
    )

#Prior for sigma
sd_prior = gamma(
    a = np.ones(3)*2,
    )

#Get the different models 
models = gp.gps

#Get the measurements
meas_v = data_obj.meas_v

#Structural distribution
struct_dist = norm()

#Numer of dimensions
dims = 3

#Hack for setting defaults
default = np.array([0,0,1.8,-1.05,0])
pos = np.array([0,1,4])

dims_full = 5

#%%

Xs = np.zeros((N,len(models),dims))

sds = np.ones((N,dims))
mus = np.zeros((N,dims))

sd = np.ones(3)
mu = np.zeros(3)

#Loop samples 
for i_sample in tqdm(range(N)):
    
    #Loop models 
    for i_model in range(len(models)):
        
        #Construct default array
        X_candidate_model = np.ones(N_eps)[:,None] * default
        
        #Add scaled and shifted samples
        X_candidate = struct_dist.rvs((N_eps,dims)) * sd \
            + mu
        
        #Non default values
        X_candidate_model[:,pos] = X_candidate
        
        #Transform for use in GP
        X_candidate_model = trans.inverse_transform(X_candidate_model)
        
        #Predict the N times and add meas_noise
        pred = models[i_model].predict(X_candidate_model) + \
            np.random.randn()*0.01
        
        #Calculated the distance 
        dist = (pred - meas_v[i_model])**2
        
        #Get the best one's idx
        idx_min = np.argmin(dist)
        
        Xs[i_sample,i_model,:] = X_candidate[idx_min]
     
    #sd and mu candidates
    sd_candidates = sd_prior.rvs((N_eps,3))
    mu_candidates = mu_prior.rvs((N_eps,3))
    
    #Simulated data
    X_simulated = struct_dist.rvs((N_eps,dims)) * sd_candidates + \
        mu_candidates
        
    dist = np.sum(
        (X_simulated - Xs[i_sample,i_model,:])**2
        ,axis=1)
    
    idx_min = np.argmin(dist)
    
    sds[i_sample,:] = sd_candidates[idx_min,:]
    mus[i_sample,:] = mu_candidates[idx_min,:]
    
    sd = sd_candidates[idx_min,:]
    mu = mu_candidates[idx_min,:]
    
    
#%%

Np = 2000

prior_samples = struct_dist.rvs((Np,3)) * sd_prior.rvs((Np,3))  \
    + mu_prior.rvs((Np,3))
    
corner(prior_samples)

#%%

posterior_samples = struct_dist.rvs((N,3)) * sds + mus
    
corner(posterior_samples)

#%%

hyper = pd.DataFrame(
    np.column_stack((mus,sds))
    )

corner(hyper.iloc[1000:])

#%%

corner(Xs[:,:,:])
#corner(prior_samples)