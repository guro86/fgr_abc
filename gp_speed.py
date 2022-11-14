#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:03:15 2022

@author: robertgc
"""

from lib.gp import gp_ensamble
from data import dakota_data


data_obj = dakota_data()
data_obj.process()

data_obj.Xtrain