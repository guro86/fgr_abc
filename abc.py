#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 08:53:54 2022

@author: gustav
"""

import pickle
import matplotlib.pyplot as plt
import pyabc
import numpy as np
import seaborn as sns
from scipy.stats import gamma, norm, lognorm
import tempfile
import os

#%%

with open('gp.p','rb') as f:
    gp, data_obj = pickle.load(f)
    
#%%

pred = gp.predict(data_obj.Xtest.values)
plt.plot(data_obj.ytest,pred,'+')

#%%
res = gp.predict(np.ones(5)[None,:]).flatten() - data_obj.meas_v
plt.plot(data_obj.meas_v,res,'o')

#%%

#Test of sampling
gp.sample_y(data_obj.Xtest.values)


def model(params):
     
    X =  np.array([params[x] for x in data_obj.Xtrain.columns])
    
    y = gp.sample_y(X[None,:]).flatten() 
        
    e = np.random.randn(len((y))) * 0.01 
    
    return {'fgr': y + e}

#%%

lb = data_obj.Xtrain.min()
ub = data_obj.Xtrain.max()

limits = zip(lb,ub,data_obj.Xtrain.columns)

dists = {
    limit[-1]: pyabc.RV('uniform',limit[0],limit[1]) for limit in limits
    }

prior = pyabc.Distribution(
    **dists
    )

#%%

def distance(x, x0):
    return np.sum(np.abs(x["fgr"] - x0["fgr"]))


#%%
abc = pyabc.ABCSMC(model, prior, distance, population_size=1000)

db_path = os.path.join(tempfile.gettempdir(), "test.db")
observation = data_obj.meas_v
abc.new("sqlite:///" + db_path, {"fgr": observation})

#%%
history = abc.run(minimum_epsilon=0.1, max_nr_populations=10)

#%%
pyabc.visualization.plot_histogram_matrix(history, t=9)
plt.savefig('abc.png')