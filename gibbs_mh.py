#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 20:48:57 2023

@author: robertgc
"""

import emcee 
import numpy as np
import seaborn as sns
from scipy.stats import norm, uniform
import data
from lib.gp import gp_ensemble
import matplotlib.pyplot as plt
from corner import corner
from tqdm import tqdm

#%%

data_obj = data.dakota_data()
data_obj.process()

#%%

Xtrain = data_obj.Xtrain.values
Xtest = data_obj.Xtest.values

meas_v = data_obj.meas_v

#%%

gp = gp_ensemble(
    Xtrain = Xtrain,
    ytrain = data_obj.ytrain.values,
    )   

gp.fit()

#%%
class prob():
    
    def __init__(self,ndim=5,observed=meas_v):
        
        self.gps = gp.gps
        self.observed = observed
        
        self.mu = np.ones(ndim)
        self.sd = np.ones(ndim)
        
        self.ndim = ndim
        
        self.struct = norm(
            loc=self.mu,
            scale=self.sd
            )
        
        self.likes = [
            norm(loc=loc,scale=0.001) for loc in meas_v
            ]
        
        self.params = np.ones(
            (len(observed),ndim)
            )
        
        nwalkers = 1
        
        self.meas_v = meas_v
        
        moves = emcee.moves.GaussianMove(.1)
        
        self.samplers = [
            emcee.EnsembleSampler(
                nwalkers, 
                ndim, 
                self.logprob,
                moves=moves,
                args = (i,)
                )  
                for i in range(len(meas_v))
            ]
        
        self.x_like = uniform(
            loc = [.15,.1,.1,.13,.02],
            scale = [39.76156355,  9.8859954 ,  0.89630072,  9.84139848,  0.98665118]
            )
        
        self.mu_prior = uniform(
            loc = [.15,.1,.1,.13,.02],
            scale = [39.76156355,  9.8859954 ,  0.89630072,  9.84139848,  0.98665118]
            )
        
        self.sd_prior = uniform(
            loc = 1e-6 * np.ones(5),
            scale =  np.ones(5)
            )
        
        self.states = [
            np.random.randn(nwalkers,ndim) * .1 + 0.9
            for i in range(len(meas_v))]
        
        self.hyper_state = np.random.randn(nwalkers,2*ndim) * .1 + 0.09
        
        hyper_moves = emcee.moves.GaussianMove(.1)
        
        self.hyper_sampler = emcee.EnsembleSampler(
            nwalkers, 
            2*ndim, 
            self.logprob_hyp,
            moves=hyper_moves,
            )  
                
        
    def sample(self):
        
        samplers = self.samplers
        states = self.states
        
        hyper_state = self.hyper_state
        
        
        for i,sampler in enumerate(samplers):
            for state in sampler.sample(
                    states[i],
                    skip_initial_state_check=True
                    ):
                pass
        
            states[i] = state
            
        self.states = states
        
        for hyper_state in self.hyper_sampler.sample(
                hyper_state,
                skip_initial_state_check=True
                ):
            pass
        
        self.hyper_state = hyper_state
        
   
    def logprob_hyp(self,x):
        
        #Get dimensions and states 
        ndim = self.ndim 
        states = self.states
        
        sd_prior = self.sd_prior
        mu_prior = self.mu_prior
        
        #Get the params from the current states
        params = (np.array([state[0] for state in states]).squeeze())
        
        #Get hyper parameters from the arg
        mu = x[:ndim]
        sd = x[-ndim:]
        
        struct = norm(loc=mu,scale=sd)
    
        self.struct = struct
        
        #Calculate logp
        lp = np.sum(
            norm(loc=mu,scale=sd).logpdf(params)
            )
        
        lp += np.sum(
            sd_prior.logpdf(sd)
            )
        
        lp += np.sum(
            mu_prior.logpdf(mu)
            )
        
        if np.isnan(lp):
            lp = -np.inf
        
        #Return sum of logp
        return lp
        
        
    def logprob(self,x,i):
        
        x_like = self.x_like
                
        struct = self.struct
        
        like = self.likes[i]
        
        gp = self.gps[i]
        
        lp = np.sum(
            struct.logpdf(x)
            )
        
        pred = gp.predict(x[None,:])
        
        lp += like.logpdf(
            pred
            ).flatten()
        
        lp += np.sum(x_like.logpdf(
            x
            )
            ).flatten()
        
                
        if np.isnan(lp):
            lp = -np.inf
                
        return lp
    
    
p = prob()
p.logprob(np.ones(5),1)
for i in tqdm(range((2000))):
    p.sample()