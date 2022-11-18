#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 09:23:15 2022

@author: robertgc
"""

import data
from lib.gp import gp_ensemble
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, FunctionTransformer

#%%

data_obj = data.dakota_data()

data_obj.process()

print(data_obj.Xtrain)

#%%

gp = gp_ensemble(
    Xtrain = data_obj.Xtrain.values,
    ytrain = data_obj.ytrain.values,
    # use_cv_alpha=True,
    # n_jobs_alpha = 8
    )   

gp.fit()

#%%

pred = gp.predict(data_obj.Xtest.values)
        
plt.plot(data_obj.ytest,pred,'+')

#%%

pred = gp.predict_fast(
        data_obj.Xtest.values
        )
plt.plot(data_obj.ytest,pred,'+')

#%%

with open('gp.p','wb') as f:
    pickle.dump([gp,data_obj],f)