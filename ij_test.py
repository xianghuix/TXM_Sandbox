#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:31:35 2020

@author: xiao
"""

BG_subtract_class = jnius.autoclass('ij.plugin.filter.BackgroundSubtracter')
BG_subtractor     = BG_subtract_class()
ImagePlusClass    = jnius.autoclass('ij.ImagePlus')

# generate some random data
frame = np.zeros((1000,1000))
frame[250:750, 250:750] = 1.0

# convert frame to Image Plus object
imp = ij.py.to_java(frame)
imp = ij.dataset().create(imp)
imp = ij.convert().convert(imp, ImagePlusClass)

BG_subtractor.rollingBallBackground(imp.getProcessor(), 100, False, False, False, False, False)

res = ij.py.from_dataset(imp)