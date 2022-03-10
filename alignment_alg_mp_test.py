#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 16:51:32 2020

@author: xiao
"""
import h5py,  time
from TXM_Sandbox.TXM_Sandbox.utils.xanes_math import multi_resolutin_tv_reg
from multiprocess import freeze_support



if __name__ == '__main__':
    freeze_support()

    fn = '/run/media/xiao/Data/data/2D_xanes/2D_trial_reg_xanes_scan2_id_54311_2020-07-01-11-20-34.h5'
    pair_id = 1
    with h5py.File(fn, 'r') as f:
        img = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format(str(pair_id).zfill(3))][:]
        fixed_img = f['/trial_registration/trial_reg_results/{0}/trial_reg_fixed{0}'.format(str(pair_id).zfill(3))][:]
        
    print(time.asctime())
    #for ii in range(5):
    #    tv1_pxl, tv1_pxl_id, shift_list, shift = multi_resolutin_tv_reg(fixed_img, img, levs=4, wz=10, step=0.2)
    tv1_pxl, tv1_pxl_id, shift_list, shift = multi_resolutin_tv_reg(fixed_img, img, levs=4, wz=10, sp_wz=8, sp_step=0.5)
    print(time.asctime())