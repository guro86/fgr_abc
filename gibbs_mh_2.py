#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 23:25:28 2023

@author: gustav
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