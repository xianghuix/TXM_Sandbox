#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:40:31 2020

@author: xiao
"""

SKLEARN_OPT = {"NEIGHBOR":{"KDE":{'algorithm':['auto', 'kd_tree', 'ball_tree'],
                                  'kernel':['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'],
                                  'metric':['euclidean'],
                                  'breadth_first':True,
                                  'bandwidth':1.0,
                                  'atol':0,
                                  'rtol':1e-8,
                                  'leaf_size':40}},
               "CLUSTER":{},
               "REGRES":{},
               "CLASSIF":{},
               "DECOMP":{},
               "PREPROC":{}}