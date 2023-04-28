#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:46:56 2023

@author: gustav
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os 


class dakota_data():
    
    def __init__(self,**kwargs):
        
        #Get path of this file
        self.path = os.path.dirname(__file__)
        
        #filenames 
        self.tu_data_file = 'data.dat'
        self.meas_file = 'meas.dat'
        
        self.train_test_split_kwargs = \
            kwargs.get('train_test_split_kwargs',{})
        
    def process(self):

        #Get base path
        path = self.path
        
        train_test_split_kwargs = self.train_test_split_kwargs

        #Get filenames 
        tu_data_file = self.tu_data_file
        meas_file = self.meas_file
        
        #Generate file paths
        tu_data_file_path = os.sep.join((path,tu_data_file))
        meas_file_path = os.sep.join((path,meas_file))


        #Read tu data 
        dakota_data = pd.read_csv(
            tu_data_file_path,
            index_col=0,
            )
        

        #Read measurement data 
        meas = pd.read_csv(
            meas_file_path,
            index_col=0
            )
        
        calibration_parameters = dakota_data.filter(regex='cal').columns
        
        
        rods = meas.index
        
        # #Used kr-xe if not nan, else use xe
        # meas['fgr'] = meas['kr-xe'].fillna(meas['xe'])
        
        # #Reorder base on tu_data order
        meas_v = meas.loc[rods,'FGR_m'].values 
        
        # #Spliting X and y samples in train and test data sets
        train, test = train_test_split(
            dakota_data,
            **train_test_split_kwargs,
            #random_state = 100 #Fixing random state now, can be removed
            )

        
        
        # #Store data 
        self.Xtrain = train.loc[:,calibration_parameters]
        self.Xtest = test.loc[:,calibration_parameters]
        
        self.ytrain = train.loc[:,rods]
        self.ytest = test.loc[:,rods]
        
        
        self.meas_v = meas_v
        self.meas = meas
        
        # self.samples = samples 
        self.dakota_data = dakota_data
        
        
if __name__ == '__main__':
    
    d = dakota_data()
    d.process()
    
    print(d.Xtrain)