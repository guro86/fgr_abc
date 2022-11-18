#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 06:08:12 2022

@author: robertgc
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
from scipy.stats import norm, multivariate_normal


#%%
with open('gp.p','rb') as f:
    gp, data_obj = pickle.load(f)
    
    
mu = np.log(np.array([2,3,0.5]))
sd = np.array([1,.7,.1])

n = 2000

dist = norm(loc=mu,scale=sd)

Xs = dist.rvs((n,3))

covX = np.diag(sd) @ np.diag(sd)


X = np.ones((n,5))*.99

X[:,[0,1,-1]] = np.exp(Xs)

pred = gp.predict_fast(X)

print(X.max(axis=0))
print(X.min(axis=0))

mu_all = np.ones((1,5))*0
mu_all[:,[0,1,-1]] = mu

X_df = pd.DataFrame(X)
corner(X_df.iloc[:,[0,1,4]])

J = gp.predict_der_fast(mu_all)[:,:,[0,1,4]]

covy = (J @ covX @ J.transpose(0,2,1)).flatten()



#%%

pred_df = pd.DataFrame(pred)
be = gp.predict_fast(np.exp(mu_all))

yerr = np.std(pred,axis=0)
yerr = np.abs(pred_df.quantile(q=[0.05,0.95]) - be)


l = np.linspace(0,.4)

plt.errorbar(data_obj.meas_v,gp.predict_fast(np.exp(mu_all)),yerr=yerr,fmt='+')

# plt.errorbar(
#     data_obj.meas_v,gp.predict_fast(np.exp(mu_all)),yerr=(covy**.5),fmt='+',
#     alpha = .5
#     )

plt.errorbar(
    data_obj.meas_v,gp.predict_fast(np.exp(mu_all)),yerr=pred_df.std(),fmt='+',
    alpha = .5
    )

plt.plot(l,l)
print(pred)


#%%

(pred_df - be).iloc[:,20].hist()