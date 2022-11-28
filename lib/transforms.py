#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:52:49 2022

@author: gustav
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


#%%

class range_tanh(BaseEstimator,TransformerMixin):
    
    def __init__(self,alpha=1,eps_range=1.05,**kwargs):
        
        self.Xmin = kwargs.get('Xmin')
        self.Xmax = kwargs.get('Xmax') 
        
        self.Xrange = kwargs.get('Xrange')
        self.X0 = kwargs.get('X0',0)
        
        self.alpha = alpha 
        self.eps_range = eps_range
    
    def fit(self,X,y=None):
        
        Xmin = X.min(axis=0)
        Xmax = X.max(axis=0)
                
        self.Xmin = Xmin
        self.Xmax = Xmax
        
        return self
        
    def transform(self,X,y=None):
        
        Xmin = self.Xmin
        Xmax = self.Xmax
        
        alpha = self.alpha 
        eps_range = self.eps_range
            
        Xrange = Xmax - Xmin
              
        Xp = (X-(Xmin+Xmax)/2.) * 2. / (Xrange*eps_range)
                
        Xtrans = np.arctanh(Xp) / alpha
        
        return Xtrans
        
    def inverse_transform(self,Xtrans,y=None):
        
        alpha = self.alpha 
        Xmin = self.Xmin
        Xmax = self.Xmax
        eps_range = self.eps_range
                
        Xrange = Xmax - Xmin
        
        return (Xmin + Xmax)/2 + \
            Xrange * eps_range * np.tanh(alpha * Xtrans) / 2
            
            
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt 
    
    trans = range_tanh(Xmax=np.array([40]),Xmin=np.array([1/40]),
                       alpha=1,eps=1.05,X0=1)
    
    X = np.linspace(1/40,40,100)
    Xtrans = trans.transform(X)
    X2 = trans.inverse_transform(Xtrans)
    
    plt.plot(X,Xtrans)
    