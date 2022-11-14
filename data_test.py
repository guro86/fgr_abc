#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 09:23:15 2022

@author: robertgc
"""

import data
from lib.gp import gp_ensamble
import matplotlib.pyplot as plt
import pickle

#%%

data_obj = data.dakota_data()
data_obj.process()


gp = gp_ensamble(
    Xtrain = data_obj.Xtrain.values,
    ytrain = data_obj.ytrain.values,
#    use_cv_alpha=True
    )   

gp.fit()

#%%

pred = gp.predict(data_obj.Xtest.values)
plt.plot(data_obj.ytest,pred,'+')

#%%

gp.gps[0].predict(np.ones(5)[None,:])

#%%

with open('gp.p','wb') as f:
    pickle.dump([gp,data_obj],f)