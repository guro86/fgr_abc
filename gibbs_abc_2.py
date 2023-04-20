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
from scipy.stats import norm

#%%

data_obj = data.dakota_data()
data_obj.process()

#%%

Xtrain = data_obj.Xtrain.values
Xtest = data_obj.Xtest.values

#%%

scaler = StandardScaler()
scaler.fit(Xtrain)

Xtrain_scaled = pd.DataFrame(
    scaler.transform(Xtrain),
    columns=data_obj.Xtrain.columns
    )

#%%
sns.pairplot(
    pd.DataFrame(scaler.transform(data_obj.Xtrain))
    )

#%%

gp = gp_ensemble(
    Xtrain = scaler.transform(Xtrain),
    ytrain = data_obj.ytrain.values,
    )   

gp.fit()

#%%
plt.plot(data_obj.ytest,
         gp.predict_fast(scaler.transform(Xtest)),
         '+'
         )

#%%

dist = norm(loc=np.zeros(5),scale=.1)

X = dist.rvs((31,5))

M = 59


#%%

#Candidate samples from the prior M x 5
x_cand = dist.rvs((M,5))

#Samples to evaluate 31 x M x 5
X_eval = X[:,None,:] * np.ones(M)[:,None]

#%%

d = 0

x_cand_d = np.ones(31)[:,None] * x_cand[:,d]

X_eval_d = X_eval

X_eval_d[:,:,d] = x_cand_d


#%%

def test(x,y):

    y = int(y)
    
    return np.array(gp.predict_i(x,y))

vec = np.vectorize(test,signature='(j,k),(0)->(1)')

test = vec(X_eval_d,np.arange(31)[:,None])