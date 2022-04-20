#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:43:37 2020

@author: xiao
"""
XANES_ANA_METHOD = {'KDE':{'bandwidth': (None, ""),
                           'algorithm': ('kd_tree', ""),
                           'kernel': ('gaussian', ""),
                           'metric': ('euclidean', ""),
                           'atol': (1e-5, ""),
                           'rtol': (1e-5, ""),
                           'breadth_first': (True, ""),
                           'leaf_size': (40, ""),
                           'metric_params': (None, "")}}
