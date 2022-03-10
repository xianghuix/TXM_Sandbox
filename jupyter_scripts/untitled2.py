#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 22:17:22 2020

@author: xiao
"""
from importlib import reload
import h5py
import xanes_math as xm
import numpy as np
import xanes_analysis as xa
xa = reload(xa)
xm = reload(xm)

f = h5py.File('/NSLS2/xf18id1/users/2020Q1/ENYUAN_Proposal_305836/2D_trial_reg_multipos_2D_xanes_scan2_id_55374_repeat_00_pos_00_config_2020-04-17-21-01-37.h5', 'r+')
imgs = f['/registration_results/reg_results/registered_xanes2D'][:]
xanes2D_analysis_eng_list = f['/processed_XANES2D/proc_parameters/eng_list'][:]
xanes2D_analysis_edge_eng = f['/processed_XANES2D/proc_parameters/edge_eng'][()]
xanes2D_analysis_pre_edge_e = f['/processed_XANES2D/proc_parameters/pre_edge_e'][()]
xanes2D_analysis_post_edge_s = f['/processed_XANES2D/proc_parameters/post_edge_s'][()]
xanes2D_analysis_edge_jump_thres = f['/processed_XANES2D/proc_parameters/edge_jump_threshold'][()]
xanes2D_analysis_edge_offset_thres = f['/processed_XANES2D/proc_parameters/edge_offset_threshold'][()]
xanes2D_analysis_use_mask = f['/processed_XANES2D/proc_parameters/use_mask'][()]
xanes2D_analysis_type = f['/processed_XANES2D/proc_parameters/analysis_type'][()]
xanes2D_analysis_data_shape = f['/processed_XANES2D/proc_parameters/data_shape'][:]
xanes2D_analysis_edge_0p5_fit_s = f['/processed_XANES2D/proc_parameters/edge_0p5_fit_s'][()]
xanes2D_analysis_edge_0p5_fit_e = f['/processed_XANES2D/proc_parameters/edge_0p5_fit_e'][()]
xanes2D_analysis_wl_fit_eng_s = f['/processed_XANES2D/proc_parameters/wl_fit_eng_s'][()]
xanes2D_analysis_wl_fit_eng_e = f['/processed_XANES2D/proc_parameters/wl_fit_eng_e'][()]
xanes2D_analysis_norm_fitting_order = f['/processed_XANES2D/proc_parameters/normalized_fitting_order'][()]
xana = xa.xanes_analysis(imgs, xanes2D_analysis_eng_list, xanes2D_analysis_edge_eng, pre_ee=xanes2D_analysis_pre_edge_e, post_es=xanes2D_analysis_post_edge_s, edge_jump_threshold=xanes2D_analysis_edge_jump_thres, pre_edge_threshold=xanes2D_analysis_edge_offset_thres)
f.close()

wl_fit_eng_s = 8.343
wl_fit_eng_e = 8.357
wl_ee_idx = xm.index_of(xana.eng, wl_fit_eng_e)
wl_es_idx = xm.index_of(xana.eng, wl_fit_eng_s)

# xana.fit_whiteline(self, wl_fit_eng_s, wl_fit_eng_e, order=3)

a = xana.spectrum[wl_es_idx:wl_ee_idx, 150:460, 280:490]

wl_fit = xm.fit_polynd(np.arange(wl_es_idx, wl_ee_idx), a, 3)
us_idx = np.linspace(wl_es_idx, wl_ee_idx,
                  num=(wl_ee_idx - wl_es_idx + 1)*10)

for ii in range(1, len(wl_fit.shape)):
    us_idx = us_idx[:, np.newaxis]

# b = np.polyval(whiteline_fit, xana.eng[wl_es_idx:wl_ee_idx,
#                                        np.newaxis, np.newaxis])
b = np.polyval(wl_fit, np.arange(wl_es_idx, wl_ee_idx)[:, np.newaxis, np.newaxis])

c = np.polyval(wl_fit, us_idx)

wl_pos_us_idx = np.argmax(c, axis=0)

wl_pos = xm.index_lookup(wl_pos_us_idx, xana.eng[wl_es_idx:wl_ee_idx], us_ratio=10)

xana.fit_whiteline(wl_fit_eng_s,
                   wl_fit_eng_e,
                   order=3,
                   us_ratio=10,
                   flt_spec=True)

xana.cal_pre_edge_sd()
xana.cal_post_edge_sd()
xana.fit_pre_edge()
xana.fit_post_edge()
xana.normalize_xanes(xanes2D_analysis_edge_eng, order=xanes2D_analysis_norm_fitting_order)
xana.fit_edge_pos(8.34, 8.352, order=3, us_ratio=10)


plt.figure(1);plt.imshow(wl_pos)
plt.figure(2);plt.imshow(xana.wl_pos)
plt.figure(3);plt.imshow(xana.edge_pos)

# plt.figure(2);plt.imshow(xana.wl_pos[150:460, 280:490])
# plt.figure(3);plt.imshow(xana.edge_pos[150:460, 280:490])
















fitc = np.polynomial.polynomial.Polynomial.fit(xana.eng[wl_es_idx:wl_ee_idx], a[:,5,5], 2)

c0=0.8240827
c1=0.20904131
c2=-0.17660113
fitcurv = c0 + c1*xana.eng[wl_es_idx:wl_ee_idx] + c2*(xana.eng[wl_es_idx:wl_ee_idx])**2
plt.plot(fitcurv)


plt.plot(a[:, 497650])

plt.plot(a[:, 497700])

plt.plot(a[:, 497600])

plt.imshow(whiteline_pos)
