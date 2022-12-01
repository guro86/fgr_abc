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
from lib.transforms import range_tanh

#%%

trans = range_tanh()

data_obj = data.dakota_data()

data_obj.process()

#%%
Xtrain_trans = trans.fit_transform(data_obj.Xtrain.values)
Xtest_trans = trans.transform(data_obj.Xtest.values)

Xtrain = data_obj.Xtrain.values
Xtest = data_obj.Xtest.values

#%%

gp = gp_ensemble(
    Xtrain = Xtrain,
    ytrain = data_obj.ytrain.values,
#    use_cv_alpha=True,
#    n_jobs_alpha = 8
    )   

gp.fit()

#%%
gp_trans = gp_ensemble(
    Xtrain = Xtrain_trans,
    ytrain = data_obj.ytrain.values,
#    use_cv_alpha=True,
#    n_jobs_alpha = 8
    )   

gp_trans.fit()

#%%

pred = gp.predict(Xtest)
        
plt.plot(data_obj.ytest,pred,'+')

#%%

pred = gp.predict_fast(
    Xtest
        )
plt.plot(data_obj.ytest,pred,'+')

#%%

pred = gp_trans.predict_fast(
    Xtest_trans
        )
plt.plot(data_obj.ytest,pred,'+')


#%%

with open('gp.p','wb') as f:
    pickle.dump(
         {
             'gp':gp,
             'gp_trans':gp_trans,
             'trans':trans,
             'data_obj':data_obj
             }
        ,f)