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
         'o'
         )

#%%
