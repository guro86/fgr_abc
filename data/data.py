#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 23:14:17 2022

@author: robertgc
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os 


class data():
    
    def __init__(self,**kwargs):
        
        #Get path of this file
        self.path = os.path.dirname(__file__)
        
        #filenames 
        self.tu_data_file = 'tu_data.csv'
        self.samples_file = 'calibration_samples.csv'
        self.meas_file = 'measurements.csv'
        
        #potential transforms
        self.Xtransform = kwargs.get('Xtransform',None) 
        self.ytransform = kwargs.get('ytransform',None) 
        
        self.train_test_split_kwargs = \
            kwargs.get('train_test_split_kwargs',{})
        
    def process(self):

        #Get base path
        path = self.path
        
        train_test_split_kwargs = self.train_test_split_kwargs

        #Get filenames 
        tu_data_file = self.tu_data_file
        samples_file = self.samples_file       
        meas_file = self.meas_file
        
        #Generate file paths
        tu_data_file_path = os.sep.join((path,tu_data_file))
        samples_file_path = os.sep.join((path,samples_file))
        meas_file_path = os.sep.join((path,meas_file))

        #Get transforms 
        Xtransform = self.Xtransform
        ytransform = self.ytransform

        #Read tu data 
        tu_data = pd.read_csv(
            tu_data_file_path,
            index_col=0,
            )
        
        #Read samples data 
        samples = pd.read_csv(
            samples_file_path,
            index_col=0
            )
        
        #Read measurement data 
        meas = pd.read_csv(
            meas_file_path,
            index_col=0
            )
        
        #Used kr-xe if not nan, else use xe
        meas['fgr'] = meas['kr-xe'].fillna(meas['xe'])
        
        #Reorder base on tu_data order
        meas_v = meas.loc[tu_data.columns,'fgr'].values / 100
        
        #If transforms, apply them
        if Xtransform is not None:
            samples = Xtransform(samples)
        
        if ytransform is not None:
            tu_data = ytransform(tu_data)
        
        #Spliting X and y samples in train and test data sets
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            samples,
            tu_data,
            **train_test_split_kwargs,
            #random_state = 100 #Fixing random state now, can be removed
            )

        samples_tu_data = pd.concat((samples,tu_data),axis=1)
        
        
        #Store data 
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.ytest = ytest
        
        self.meas_v = meas_v
        self.meas = meas
        
        self.samples = samples 
        self.tu_data = tu_data
        self.samples_tu_data = samples_tu_data