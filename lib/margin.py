#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 21:50:16 2023

@author: gustav
"""

import numpy as np
from scipy.stats import norm


class model_wrapper():
    
    def __init__(self,**kwargs):
        
        self.default = kwargs.get(
            'default',
            np.zeros(5)
            )
        
        self.pos = kwargs.get('pos',[0,1,2])
        self.model = kwargs.get('model',None)
    
    def predict(self,x):
        
        default = self.default
        pos = self.pos
        
        X = default
        
        X[pos] = x
        
        return self.model.predict_fast(X[None,:])
    
    
    def predict_chain(self,X):
        
        default = self.default
        
        pos = self.pos
        
        Xnew = np.zeros((len(X),len(default)))
        
        Xnew[:,:] = default
        
        Xnew[:,pos] = X
        
        return self.model.predict_fast(Xnew)
    
    def predict_der(self,x):
        
        pos = self.pos 
        
        X = self.default
        X[self.pos] = x
        
        return self.model.predict_der_fast(X[None,:])[:,:,pos]
    
    def predict_pred_and_der(self,x):
    
        pos = self.pos 
        
        X = self.default
        X[self.pos] = x
        
        pred, der = self.model.predict_pred_and_der_fast(X[None,:])
        
        return pred, der[:,:,pos]
    

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
        
        # pred = model.predict(mu[None,:]).flatten()
        # J = model.predict_der(mu[None,:])
        
        pred, J = model.predict_pred_and_der(mu[None,:])
        
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