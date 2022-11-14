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

#%%

#Load data and gaussian process
with open('gp.p','rb') as f:
    gp, data_obj = pickle.load(f)
    
#%%
    
class margin():
    
    def __init__(self,**kwargs):
        
        #Dimensions
        self.d = kwargs.get('d',3)
        
        #Eta parameter
        self.eta = kwargs.get('eta',10)
        
        #Cov, corr, p, sd, and mu
        self.cov = np.eye(self.d) 
        self.corr = np.eye(self.d) 
        self.p = np.ones(self.d)
        self.sd = np.ones(self.d)
        self.mu = np.ones(self.d) 
        
        self.R = kwargs.get('R',0.01)
        
        self.sd_prior = kwargs.get('sd_prior',None)
        self.mu_prior = kwargs.get('mu_prior',None)

        #lower indices of cholesky composition
        self.tril_indices = np.tril_indices(self.d,k=-1)
        
        #The forward model
        #Predicts values and derivatives
        self.model=kwargs.get('model')
        
        #The measurements
        self.meas_v = kwargs.get('meas_v')
        
        #Number of correlations 
        self.num_corr = int((self.d**2 - self.d)/2)
        
        #Total number of parameters
        self.total_parameters = 2*self.d + self.num_corr
        
        
    def lkj_prior(self):
        
        #Get correlation matrix
        corr = self.corr
        eta = self.eta 
        
        #Calculate the determinantn 
        det = np.linalg.det(corr)
        
        #If det is positive
        if det>0:
            
            #Calculate and return logp
            return (eta-1) * np.log(det)
        
        else:
            
            #Else return -np.inf
            return -np.inf
        
        
    def like(self):
        
        #Get propaged errors
        cov_y = self.cov_y
        
        #Predictions
        pred = self.pred
        
        #Measurements 
        meas_v = self.meas_v
        
        #Mesurement error/erros
        R = self.R
        
        #Calculate the error on the output in terms of sd
        scale = (cov_y + R**2)**.5
        
        #Scale the error 
        #res = (meas_v - pred)/scale
                
        #Return logp
        logp = np.sum(norm(loc=meas_v,scale=scale).logpdf(pred))
        
        #Just checking that the error when propagated is 
        #positive
        if cov_y.min() < 0: 
            return -np.inf
        else:
            return logp
    
    def logp(self):
        
        #Start with zero logp
        logp = 0
        
        #Add the logp of the lkj prior
        logp += self.lkj_prior()
        
        #If we have a sd prior, add
        if self.sd_prior:
            logp += np.sum(
                self.sd_prior.logpdf(self.sd)
                )
            
        #If we have a mu priro add
        if self.mu_prior:
            logp += np.sum(
                self.mu_prior.logpdf(self.mu)
                )
        
        #if the we have a finite logp add likelihood
        if np.isfinite(logp):
            logp += self.like()
        
        #If the likelihood is nan, set it to -inf
        if np.isnan(logp):
            #import pdb; pdb.set_trace()
            logp = -np.inf
        
        return logp
        
    def update_and_return_logp(self,x):
        
        self.update(x)
        
        return self.logp()
        
    def update(self,x):
        
        #Dimensions
        d = self.d
        
        #Get model
        model = self.model
        
        #Total parameters
        total_parameters = self.total_parameters

        #Lower indices
        tril_indices = self.tril_indices
        
        #Correlations matrix
        corr = self.corr
        
        #Number of correlations
        num_corr = self.num_corr
        
        if len(x) != total_parameters:
            raise ValueError('Unvalid lenght of x')
        
        mu = x[:d]
        sd = x[d:2*d]
        p = x[-num_corr:]

        corr[tril_indices] = p
        corr[tril_indices[::-1]] = p
        
        #Calculate the S matrix
        S = np.diag(sd)
        
        #Calculate Cov matrix
        cov = S @ corr @ S
        
        pred = model.predict(mu[None,:]).flatten()
        J = model.predict_der(mu[None,:])
        
        #Transpose derivatives
        JT = J.transpose(0,2,1)
        
        #Covy
        cov_y = (J @ cov @ JT).flatten()
        
        self.cov = cov
        self.corr = corr
        self.sd = sd
        self.mu = mu
        self.pred = pred 
        self.J = J
        self.cov_y = cov_y

#%%

class reduced_model():
    
    def __init__(self,**kwargs):
        
        self.default = kwargs.get(
            'default',
            np.array([1,1,1,1,1])*.99
            )
        
        self.pos = kwargs.get('pos',[0,1,2])
        self.model = kwargs.get('model',None)
    
    def predict(self,x):
        
        default = self.default
        pos = self.pos
        
        X = default
        
        X[pos] = x
        
        return self.model.predict(X[None,:])
    
    def predict_der(self,x):
        
        pos = self.pos 
        
        X = self.default
        X[self.pos] = x
        
        return self.model.predict_der(X[None,:])[:,:,pos]
    
    
        
rgp = reduced_model(
    model=gp,
    pos = np.array([0,1,4])
    )
   
    
#%%

x0 = np.concatenate((
    np.array([1,1,.5]),
    np.ones(3),
    np.zeros(3)
    ))

ub = data_obj.Xtrain.quantile(q=0.99).values[rgp.pos]
lb = data_obj.Xtrain.quantile(q=0.01).values[rgp.pos]

sd = np.array([0.25927605, 0.63275283, 0.03008127])
mu = np.array([1.26456655, 5.82954543, 0.58463441])

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

nwalkers = 18
ndim = 9


sampler = emcee.EnsembleSampler(
    nwalkers, ndim, m.update_and_return_logp
    )

initial_state = np.random.randn(nwalkers,ndim)*0.01 + x0
sampler.run_mcmc(initial_state, nsteps=2000,progress=True)


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
        multivariate_normal(m.mu,m.cov).rvs()
        )

marg = chain.iloc[:100].apply(sim,axis=1)

