#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 22:22:22 2022

@author: gustav
"""


import data
from lib.gp import gp_ensemble
import matplotlib.pyplot as plt
import pickle
import numpy as np

#%%

data_obj = data.dakota_data()
data_obj.process()


gp = gp_ensemble(
    Xtrain = data_obj.Xtrain.values,
    ytrain = data_obj.ytrain.values,
    # use_cv_alpha = True,
    # n_jobs_alpha = 8
    )   

gp.fit()

#%%


Xtrain = data_obj.Xtrain
X = np.ones(5)[None,:]


#%%

gp.predict(data_obj.Xtest.values)

#%%

gp.predict_fast(data_obj.Xtest.values)

#%%

gp.predict_der(X)

#%%

gp.predict_der_fast(data_obj.Xtest.values).shape

#%%

%%timeit

gp.predict_fast(X).shape
gp.predict_der_fast(X).shape

#%%

%%timeit 

a,b = gp.predict_pred_and_der_fast(X)