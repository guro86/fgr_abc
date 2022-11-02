#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 09:20:27 2022

@author: robertgc
"""

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

class gp_ensamble():
    
    def __init__(self,**kwargs):
        
        #Get the instances from kwargs
        self.Xtrain = kwargs.get('Xtrain',None)
        self.ytrain = kwargs.get('ytrain',None)
        
        #Gps and dimensions
        self.gps = None 
        self.dims = None
        
        self.use_lr_mean = kwargs.get('use_lr_mean',False)
        
        self.alpha = kwargs.get('alpha',1e-2)
        
        self.alpha_dist = uniform(loc=1e-10,scale=1e-1 - 1e-10)
        
        self.n_jobs_alpha = kwargs.get('n_jobs_alpha',4)
        
        self.use_cv_alpha = kwargs.get('use_cv_alpha',False)
        
        self.gps = []
        self.lr_means = []
        
    def predict(self,X,**kwargs):
        
        #Get return std kwarg
        return_std = kwargs.get('return_std',False)
        
        #Get gps
        gps = self.gps 
        
        use_lr_mean = self.use_lr_mean
                
        lr_means = self.lr_means
        
        if use_lr_mean:
            
            pred = np.column_stack([gp.predict(X) + lr.predict(X) for lr,gp \
                                    in zip(lr_means,gps)])
            
        else:
            
            pred = np.column_stack([gp.predict(X) for gp in gps])
        
        
        #If std is requested
        if return_std:
        
            #Stack std of all gps
            std = np.column_stack(
                [gp.predict(X,return_std=True)[-1] for gp in gps]
                )
            
            #Return pred and std
            return pred, std
        
        #If stds are requested, return pred only
        else:
            
            #Return pred
            return pred
    
    def predict_der(self,X):
        
        #Get all the individual gps
        gps = self.gps
        
        #Calculate and stack all derviatives 
        J = np.stack(
            [gp.predict_der(X).T for gp in gps]
            )
        
        #Return
        return J
    
    def sample_y(self,X):
        
        gps = self.gps 
        
        samples = np.column_stack([gp.sample_y(X) for gp in gps])
        
        return samples
    
    def fit(self):
        
        #Get training data 
        Xtrain = self.Xtrain
        ytrain = self.ytrain
        
        use_lr_mean = self.use_lr_mean
        use_cv_alpha = self.use_cv_alpha
        
        n_jobs_alpha = self.n_jobs_alpha
        
        #Calc dimensions of inpu
        dims = Xtrain.shape[-1]
        
        #Store dimensions 
        self.dims = dims
        
        alpha_dist = self.alpha_dist
        
        #A list of one gp per output dimension
        gps = [self._factory() for i in range(ytrain.shape[-1])]
       
        if use_lr_mean:
            lr_means = [LinearRegression() for i in range(ytrain.shape[-1])] 
       
        for i,gp in enumerate(gps): 
            
            ytrain_i = ytrain[:,i]
            
            if use_lr_mean:
                
                lr_means[i].fit(
                    Xtrain,ytrain_i
                    )
                
                ytrain_i = ytrain_i - lr_means[i].predict(Xtrain)
            
            
            if use_cv_alpha:
                
                print('Cross validating alpha for {}-th gp'.format(i))
                
                search = RandomizedSearchCV(
                    gp,
                    {'alpha':alpha_dist},
                    n_jobs = n_jobs_alpha
                    )
                
                search.fit(
                    Xtrain,
                    ytrain_i,
                    )
                
                gps[i] = search.best_estimator_
                
            else:
                
                gp.fit(
                    Xtrain,
                    ytrain_i
                    )
                

        if use_lr_mean:
            self.lr_means = lr_means
 
        #Store trained gps
        self.gps = gps

        
        
    #Internal factory function to create a local gaussian process
    def _factory(self):
        
        #Get fimenstions 
        dims = self.dims 
        
        #Get alpha
        alpha = self.alpha
        
        #Create kernel
        kernel = 1 * RBF(
            length_scale=np.ones(dims),
            length_scale_bounds=(1e-5,1e10)
            )
                
        #Create gp
        gp = my_gp(
            kernel = kernel,
            normalize_y=True,
            alpha = alpha,
            )
        
        #return gp
        return gp

class my_gp(GPR):
    
    #Function that calculates derivative
    #(give an RBF kernel)
    def predict_der(self,X):
        
        #std used in normalization
        y_train_std = self._y_train_std
        
        #Get kernel 
        kernel = self.kernel_
        
        #Get alpha vector
        alpha = self.alpha_ 
        
        #Get training data
        Xtrain = self.X_train_

        #Get length scale/scales
        l = self.kernel_.get_params()['k2__length_scale']
        
        #Fix dimensions if np array
        #i.e. l in the first dimension and 
        #append the last two
        if isinstance(l,np.ndarray):
            l = l[:,np.newaxis,np.newaxis]
            
        #Calculate diffs dimension for dimension
        #All combinations of diffs are calculated 
        #for all different dimensions (first axis)
        diff = Xtrain[np.newaxis,:].transpose(2,1,0) - \
            X[np.newaxis,:].transpose(2,0,1)
        
        #Use kernel and diffs to evaluate ks
        ks = kernel(Xtrain,X) * diff / l**2
        
        print(ks.shape)
                        
        #Calculate derivative 
        der = ks.transpose(0,2,1) @ alpha 
        
        #Undo normalization
        der *= y_train_std
        
        #Return derivative
        return der
