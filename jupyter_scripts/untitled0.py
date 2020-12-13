#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 10:25:28 2020

@author: xiao
"""

import h5py, tifffile
from pathlib import Path
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
import xanes_math as xm
import xanes_analysis as xa
xa = reload(xa)

aligned_4D_recon_file = "/NSLS2/xf18id1/users/2020Q1/YUAN_YANG_Proposal_305811/3D_trial_reg_scan_id_56600-56609_2020-03-13-17-48-29.h5"
raw_file_template = "/NSLS2/xf18id1/users/2020Q1/YUAN_YANG_Proposal_305811/fly_scan_id_{0}.h5"
out_fn = '/NSLS2/xf18id1/users/2020Q1/YUAN_YANG_Proposal_305811/test_4D_56000_56009.h5'

# if it is whiteline, set is_wl = True, otherwise is_wl = False
is_wl = True
# scan_id start and end
scan_id_s = 56000
scan_id_e = 56009

# if you want to make a mask with a recon, set make_mask = True
make_mask = True
# scan id for making mask; it is good to choose one after the edge
mask_scan_id = 56005
# set threshold for making the mask; you can pre-test out the threshold in imagej
mask_threshold = 0

# edge_offset from Co
edge_offset_2_Co = 8.333 - 7.709
# estimated edge energy
edge_eng = 7.709 + edge_offset_2_Co
# end poit of the pre-edge relative to the edge_eng in keV
pre_ee = -0.05
# start point of the post-edge relative to the edge_eng in keV
post_es = 0.1
# how many times of the edge jump magnitude should be compared to the pre-edge standard deviation
edge_jump_threshold = 4
# how much should the pre_edge be offset up for validating if the post edge trend is in a reasonable range
# this is a factor to the pre-edge deviation
pre_edge_threshold = 3
# define an energy range for 0.5 absorption postion fitting
ep_eng_s = 7.716 + edge_offset_2_Co
ep_eng_e = 7.731 + edge_offset_2_Co
# define an energy range for whiteline peak postion fitting
wl_eng_s = 7.725 + edge_offset_2_Co +0.001
wl_eng_e = 7.735 + edge_offset_2_Co - 0.003

# define an energy range for edge_pos display
ep_vmin = 7.720 + edge_offset_2_Co
ep_vmax = 7.731 + edge_offset_2_Co
# define an energy range for whiteline display
wl_vmin = 7.7285 + edge_offset_2_Co
wl_vmax = 7.7305 + edge_offset_2_Co
# define path and file name to save xanes analysis results; if you use the default path and name as below,
# you don't need to change anything. otherwise, give your full path and file name below.
#out_fn = os.path.join(str(Path(fn_template).parent), 'xanes_analysis_' + str(Path(fn_template).name)).format(scan_id)
#print(out_fn)

f = h5py.File(aligned_4D_recon_file, 'r')
imgs_shape = f['/registration_results/reg_results/registered_xanes3D'].shape
imgs = f['/registration_results/reg_results/registered_xanes3D'][:, 0, :, :]
eng = f['/trial_registration/trial_reg_parameters/eng_list'][:]
f.close()
xana = xa.xanes_analysis(imgs, eng, edge_eng, pre_ee=pre_ee, post_es=post_es,
                         edge_jump_threshold=edge_jump_threshold, pre_edge_threshold=pre_edge_threshold)
xanes3d = np.ndarray([imgs_shape[1], imgs_shape[2], imgs_shape[3]], dtype=np.float32)

f = h5py.File(aligned_4D_recon_file, 'r')
for ii in range(imgs_shape[1]):
    xana.spectrum[:] = imgs = f['/registration_results/reg_results/registered_xanes3D'][:, ii, :, :]
    if is_wl:
        xana.fit_whiteline(wl_eng_s, wl_eng_e)
        xanes3d[ii] = xana.whiteline_pos[:]
f.close()
print('done')