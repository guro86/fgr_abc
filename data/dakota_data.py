#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 09:25:56 2022

@author: robertgc
"""

import os 
import pandas as pd
from sklearn.model_selection import train_test_split

class dakota_data():

    #Constructor    
    def __init__(self,**kwargs):
        
        #Get path of this file
        self.path = os.path.dirname(__file__)
        
        #filenames 
        self.dakota_file = 'dakota_tabular.dat'
        
        self.meas_file = 'measurements.csv'
        
        #kwargs for data splitting
        self.train_test_split_kwargs = \
            kwargs.get('train_test_split_kwargs',{})
            
        self.n_xvars = 5
        
    def process(self):

        #Get base path
        path = self.path
        
        train_test_split_kwargs = self.train_test_split_kwargs

        #Get filenames 
        dakota_file = self.dakota_file
        meas_file = self.meas_file
        
        n_xvars = self.n_xvars
        
        #Generate file paths
        dakota_file_path = os.sep.join((path,dakota_file))
        meas_file_path = os.sep.join((path,meas_file))
        
        dakota_data = pd.read_csv(
            dakota_file_path,
            sep = '\s+'
            ).iloc[:,2:]

        #Read measurement data 
        meas = pd.read_csv(
            meas_file_path,
            index_col=0
            )
        
        X = dakota_data.iloc[:,:n_xvars]
        y = dakota_data.iloc[:,n_xvars:]
        
        
        self.meas = meas
        self.dakota_data = dakota_data
        
        
        # #Used kr-xe if not nan, else use xe
        # meas['fgr'] = meas['kr-xe'].fillna(meas['xe'])
        
        # #Reorder base on tu_data order
        # meas_v = meas.loc[tu_data.columns,'fgr'].values / 100

        
        #Spliting X and y samples in train and test data sets
        # Xtrain, Xtest, ytrain, ytest = train_test_split(
        #     samples,
        #     tu_data,
        #     **train_test_split_kwargs,
        #     #random_state = 100 #Fixing random state now, can be removed
        #     )
        
        
        #Store data 
        # self.Xtrain = Xtrain
        # self.Xtest = Xtest
        # self.ytrain = ytrain
        # self.ytest = ytest
        
        # self.meas_v = meas_v
        # self.meas = meas
