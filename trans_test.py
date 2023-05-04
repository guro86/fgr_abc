#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:16:22 2023

@author: robertgc
"""

import data 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from lib.transforms import range_tanh
import numpy as np
import matplotlib.pyplot as plt

data_obj = data.dakota_data()
data_obj.process()

ftrans3 = FunctionTransformer(func = lambda x: (x) / .5)

ftrans4 = FunctionTransformer(func = lambda x: (x) / 2)


trans = Pipeline(
    [
       ('scaler1',StandardScaler()),
     ('range_tanh',range_tanh()),
      ('scaler2',StandardScaler())
    ]
    )

trans2 = range_tanh()

trans3 = Pipeline(
    [
     ('range_tanh',range_tanh()),
      ('scaler2',ftrans3)
    ]
    )


trans4 = Pipeline(
    [
     ('range_tanh',range_tanh()),
      ('scaler2',ftrans4)
    ]
    )


trans.fit(
    data_obj.Xtrain.iloc[:,0].values.reshape(-1,1)
    )

trans2.fit(
    data_obj.Xtrain.iloc[:,0].values.reshape(-1,1)
    )

trans3.fit(
    data_obj.Xtrain.iloc[:,0].values.reshape(-1,1)
    )

trans4.fit(
    data_obj.Xtrain.iloc[:,0].values.reshape(-1,1)
    )


Xmin = data_obj.Xtrain.iloc[:,0].values.min()
Xmax = data_obj.Xtrain.iloc[:,0].values.max()

X = np.linspace(Xmin,Xmax,100).reshape(-1,1)

y = trans.transform(X)
y2 = trans2.transform(X)
y3 = trans3.transform(X)
y4 = trans4.transform(X)


plt.plot(X,y,label='Scaled')
plt.plot(X,y2,ls='--',label='Not scaled')
plt.plot(X,y3,label='Divided by .5')
plt.plot(X,y4,label='Divided by 2')

plt.legend()

plt.xlabel('Physical parameter')
plt.ylabel('Transformed parameter')