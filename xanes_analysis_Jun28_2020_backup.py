#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 12:46:23 2019

@author: xiao
"""

import numpy as np
import xanes_math as xm
from xanes_math import index_of
import xanes_spectra_filters as xf
#Simport xaens_normalization as xn
from scipy.signal import argrelmax
from scipy.ndimage import median_filter
import h5py, os, tifffile
from pathlib import Path
from copy import deepcopy
import multiprocess as mp

"""
    This class include xanes spectrum analyses. These analyses are based on 
    filtered xanes spectra. The basic functions include:
        1. curve fitting
        2. peak fitting
    The analyses in this package include:
        
        1. Subtracting pre_edge backgroud
        2. Normalizing spectrum
        3. Finding e0
        4. Calculating edge jump
        5. Calculating pre_edge and post_edge statistics
        6. Finding peak (whiteline)
        7, PCA analysis
        8. Linear Combinatiotn analysis
"""
class xanes_analysis():
    def __init__(self, spectrum, eng, preset_edge_eng,
                 pre_es=None, pre_ee=-50,
                 post_es=100, post_ee=None,
                 edge_jump_threshold=5,
                 pre_edge_threshold=1):
        self.spectrum = spectrum
        self.eng = eng
        self.preset_edge_eng = preset_edge_eng


        if pre_es is None:
            pre_es = eng[0] - preset_edge_eng
        self.pre_es = preset_edge_eng + pre_es
        self.pre_ee = preset_edge_eng + pre_ee
        self.pre_es_idx = xm.index_of(self.eng, self.pre_es)
        self.pre_ee_idx = xm.index_of(self.eng, self.pre_ee)

        self.post_es = preset_edge_eng + post_es
        if post_ee is None:
            post_ee = eng[-1] - preset_edge_eng
        self.post_ee = preset_edge_eng + post_ee
        self.post_es_idx = xm.index_of(self.eng, self.post_es)
        self.post_ee_idx = xm.index_of(self.eng, self.post_ee)

        self.edge_jump_threshold = edge_jump_threshold
        self.fitted_edge_threshold = pre_edge_threshold
        self.auto_e0 = [None]
        self.pre_edge_fit = None
        self.post_edge_fit = None
        self.edge_jump_mask = None
        self.fitted_edge_mask = None
        self.normalized_spectrum = None
        self.fitted_edge_flted_spec = None
        self.edge_jump_flted_spec = None
        self.removed_img_ids = []
        self.post_edge_fit_coef = []
        self.pre_edge_fit_coef = []
        self.pre_edge_sd_map = []
        self.post_edge_sd_map = []
        self.pre_edge_mean_map = []
        self.post_edge_mean_map = []
        self.pre_avg = []
        self.post_avg = []
        self.edge_jump_map = []
        # self.fitted_edge_threshold = []

        self.wl_pos_direct = np.array([None])
        self.direct_wl_ph = np.array([None])
        self.wl_pos = np.array([None])
        self.edge_pos = np.array([None])
        
        self.pre_edge_fitted = False
        self.post_edge_fitted = False
        self.spec_normalized = False
        self.spec_linear_fitted = False
        self.spec_pca_fitted = False

    def cal_pre_edge_sd(self):
        """
        return: ndarray, pre_edge_sd has dimension of spectrum.shape[1:]
        """
        self.pre_edge_sd_map = self.spectrum[self.pre_es_idx:self.pre_ee_idx, 
                                             :].std(axis=0)

    def cal_post_edge_sd(self):
        """
        return: ndarray, post_edge_sd has dimension of spectrum.shape[1:]
        """
        self.post_edge_sd_map = self.spectrum[self.post_es_idx:self.post_ee_idx, 
                                              :].std(axis=0)
        
    def cal_pre_edge_mean(self):
        """
        return: ndarray, pre_edge_sd has dimension of spectrum.shape[1:]
        """
        self.pre_edge_mean_map = self.spectrum[self.pre_es_idx:self.pre_ee_idx, 
                                               :].mean(axis=0)

    def cal_post_edge_mean(self):
        """
        return: ndarray, post_edge_sd has dimension of spectrum.shape[1:]
        """
        self.post_edge_mean_map = self.spectrum[self.post_es_idx:self.post_ee_idx, 
                                                :].mean(axis=0)

    def fit_pre_edge(self):
        """
        return: ndarray, pre_edge_fit has dimension of [2].append(list(spectrum.shape[1:]))
        """
        try:
            self.pre_edge_fit_coef = xm.fit_polynd(self.eng[self.pre_es_idx:self.pre_ee_idx],
                                                   self.spectrum[self.pre_es_idx:self.pre_ee_idx].\
                                                        reshape((self.pre_ee_idx-self.pre_es_idx, -1)), 1).\
                                                        reshape([2]+list(self.spectrum.shape[1:]))
            self.pre_edge_fitted = True                                            
            # if len(self.spectrum.shape) == 2:
            #     self.pre_edge_fit = (self.pre_edge_fit_coef * \
            #                         (np.vstack((self.eng, np.ones(self.eng.shape[0]))).T)[:, :, np.newaxis]).sum(axis=1)
            #     self.pre_edge_fitted = True
            # elif len(self.spectrum.shape) == 3:
            #     self.pre_edge_fit = (self.pre_edge_fit_coef * \
            #                         (np.vstack((self.eng, np.ones(self.eng.shape[0]))).T)[:, :, np.newaxis, np.newaxis]).sum(axis=1)
            #     self.pre_edge_fitted = True
            # else:
            #     self.pre_edge_fit = None
            #     self.pre_edge_fitted = False           
        except:
            self.pre_edge_fitted = False

    def fit_post_edge(self):
        """
        return: ndarray, post_edge_fit has dimension of [2].append(list(spectrum.shape[1:]))
        """
        try:
            self.post_edge_fit_coef = xm.fit_polynd(self.eng[self.post_es_idx:self.post_ee_idx],
                                                    self.spectrum[self.post_es_idx:self.post_ee_idx].\
                                                    reshape(self.post_ee_idx-self.post_es_idx, -1), 1).\
                                                    reshape([2]+list(self.spectrum.shape[1:]))
            self.post_edge_fitted = True                                        
            # if len(self.spectrum.shape) == 2:
            #     self.post_edge_fit = (self.post_edge_fit_coef * \
            #                          (np.vstack((self.eng, np.ones(self.eng.shape[0]))).T)[:, :, np.newaxis]).sum(axis=1)
            #     self.post_edge_fitted = True
            # elif len(self.spectrum.shape) == 3:
            #     self.post_edge_fit = (self.post_edge_fit_coef * \
            #                          (np.vstack((self.eng, np.ones(self.eng.shape[0]))).T)[:, :, np.newaxis, np.newaxis]).sum(axis=1)
            #     self.post_edge_fitted = True
            # else:
            #     self.post_edge_fit = None
            #     self.post_edge_fitted = False                                   
        except:
            self.post_edge_fitted = False
            
    def cal_edge_jump_map(self):
        """
        return: ndarray, edge_jump_map has dimension of spectrum.shape[1:]
        """
        self.edge_jump_map = (xf.cal_post_edge_fit(np.array([self.preset_edge_eng]), self.post_edge_fit_coef) - 
                              xf.cal_pre_edge_fit(np.array([self.preset_edge_eng]), self.pre_edge_fit_coef))
        
    def normalize_xanes(self, eng0, order=1, save_pre_post=False):
        """
        For 2D XANES, self.spectrum dimensions are [eng, 2D_space]
        For 3D XANES, self.spectrum dimensions are [eng, 3D_space]
        
        normalize_xanes includes two steps: 1) pre-edge background subtraction (order = 0)),
                                            2) post-edge fitted or average mu normalization (order = 1)
        return: ndarray, same dimension as spectrum.shape
        """
        eng = deepcopy(self.eng)
        for ii in range(1, self.pre_edge_fit_coef.ndim):
            eng = eng[:, np.newaxis]
        
        if order == 0:
            self.normalized_spectrum = (self.spectrum - self.pre_edge_mean_map)/\
                                       (self.post_edge_mean_map - self.pre_edge_mean_map)
            self.normalized_spectrum[np.isnan(self.normalized_spectrum)] = 0
            self.normalized_spectrum[np.isinf(self.normalized_spectrum)] = 0
            self.spec_normalized = True
        elif order == 1:
            if save_pre_post:
                self.pre_edge_fit = xf.cal_pre_edge_fit(self.eng, self.pre_edge_fit_coef)
                self.post_edge_fit = xf.cal_post_edge_fit(self.eng, self.post_edge_fit_coef)
                self.normalized_spectrum = (self.spectrum - self.pre_edge_fit)/(
                                            self.post_edge_fit -
                                            self.pre_edge_fit)
                self.normalized_spectrum[np.isnan(self.normalized_spectrum)] = 0
                self.normalized_spectrum[np.isinf(self.normalized_spectrum)] = 0
            else:
                self.normalized_spectrum = (self.spectrum -
                                            xf.cal_pre_edge_fit(self.eng, self.pre_edge_fit_coef))/(
                                            xf.cal_post_edge_fit(self.eng, self.post_edge_fit_coef) -
                                            xf.cal_pre_edge_fit(self.eng, self.pre_edge_fit_coef))                
                self.normalized_spectrum[np.isnan(self.normalized_spectrum)] = 0
                self.normalized_spectrum[np.isinf(self.normalized_spectrum)] = 0
            self.spec_normalized = True
        else:
            print('Order can only be "1" or "0".')
            self.spec_normalized = False

        # eng = deepcopy(self.eng)
        # for ii in range(1, self.pre_edge_fit_coef.ndim):
        #     eng = eng[:, np.newaxis]
        
        # if order == 0:
        #     self.normalized_spectrum = (self.spectrum - self.pre_edge_mean_map)/\
        #                                (self.post_edge_mean_map - self.pre_edge_mean_map)
        #     self.normalized_spectrum[np.isnan(self.normalized_spectrum)] = 0
        #     self.normalized_spectrum[np.isinf(self.normalized_spectrum)] = 0
        # elif order == 1:
        #     if save_pre_post:
        #         if len(self.spectrum.shape) == 3:
        #             self.pre_edge_fit = (self.pre_edge_fit_coef * \
        #                                 (np.vstack((self.eng, np.ones(self.eng.shape[0]))).T)[:, :, np.newaxis, np.newaxis]).sum(axis=1)
        #             self.post_edge_fit = (self.post_edge_fit_coef * \
        #                                  (np.vstack((self.eng, np.ones(self.eng.shape[0]))).T)[:, :, np.newaxis, np.newaxis]).sum(axis=1)
        #         elif len(self.spectrum.shape) == 4:
        #             self.pre_edge_fit = (self.pre_edge_fit_coef * \
        #                                 (np.vstack((self.eng, np.ones(self.eng.shape[0]))).T)[:, :, np.newaxis, np.newaxis, np.newaxis]).sum(axis=1)
        #             self.post_edge_fit = (self.post_edge_fit_coef * \
        #                                  (np.vstack((self.eng, np.ones(self.eng.shape[0]))).T)[:, :, np.newaxis, np.newaxis, np.newaxis]).sum(axis=1)
                        
        #         self.normalized_spectrum = (self.spectrum - self.pre_edge_fit)/(
        #                                     self.post_edge_fit -
        #                                     self.pre_edge_fit)
        #         self.normalized_spectrum[np.isnan(self.normalized_spectrum)] = 0
        #         self.normalized_spectrum[np.isinf(self.normalized_spectrum)] = 0
        #     else:
        #         if len(self.spectrum.shape) == 3:
        #             self.normalized_spectrum = (self.spectrum -
        #                                         (self.pre_edge_fit_coef * \
        #                                         (np.vstack((self.eng, np.ones(self.eng.shape[0]))).T)[:, :, np.newaxis, np.newaxis]).sum(axis=1))/(
        #                                         (self.post_edge_fit_coef * \
        #                                         (np.vstack((self.eng, np.ones(self.eng.shape[0]))).T)[:, :, np.newaxis, np.newaxis]).sum(axis=1) -
        #                                         (self.pre_edge_fit_coef * \
        #                                         (np.vstack((self.eng, np.ones(self.eng.shape[0]))).T)[:, :, np.newaxis, np.newaxis]).sum(axis=1))
        #         elif len(self.spectrum.shape) == 4:
        #             self.normalized_spectrum = (self.spectrum -
        #                                         (self.post_edge_fit_coef * \
        #                                         (np.vstack((self.eng, np.ones(self.eng.shape[0]))).T)[:, :, np.newaxis, np.newaxis, np.newaxis]).sum(axis=1))/(
        #                                         (self.post_edge_fit_coef * \
        #                                         (np.vstack((self.eng, np.ones(self.eng.shape[0]))).T)[:, :, np.newaxis, np.newaxis, np.newaxis]).sum(axis=1) -
        #                                         (self.pre_edge_fit_coef[0][np.newaxis, :]*eng + self.pre_edge_fit_coef[1][np.newaxis, :]))
        #         self.normalized_spectrum[np.isnan(self.normalized_spectrum)] = 0
        #         self.normalized_spectrum[np.isinf(self.normalized_spectrum)] = 0
        # else:
        #     print('Order can only be "1" or "0".')
        #     pass

    def create_edge_jump_filter(self, threshold):
        """
        return: ndarray, same dimension as spectrum.shape
        """
        self.edge_jump_threshold = threshold
        # print(np.array([self.preset_edge_eng]).shape,
        #       self.pre_edge_fit_coef.shape,
        #       self.post_edge_fit_coef.shape,
        #       self.pre_edge_sd_map,
        #       self.edge_jump_threshold.shape)
        self.edge_jump_mask = xf.edge_jump_filter(np.array([self.preset_edge_eng]), 
                                                  self.pre_edge_fit_coef, 
                                                  self.post_edge_fit_coef, 
                                                  self.pre_edge_sd_map, 
                                                  self.edge_jump_threshold)
        # self.edge_jump_mask = xf.edge_jump_filter(self.preset_edge_eng, self.pre_edge_fit_coef,
        #                                           self.post_edge_fit_coef, self.pre_edge_sd_map,
        #                                           self.edge_jump_threshold)

    def create_fitted_edge_filter(self, threshold):
        """
        return: ndarray, same dimension as spectrum.shape
        """
        self.fitted_edge_threshold = threshold
        self.fitted_edge_mask = xf.fitted_edges_filter(self.eng, 
                                                       self.pre_edge_fit_coef,
                                                       self.post_edge_fit_coef, 
                                                       self.pre_edge_sd_map,
                                                       self.fitted_edge_threshold)

    def apply_edge_jump_filter(self, threshold):
        """
        return: ndarray, same dimension as spectrum.shape
        """
        return self.spectrum * self.edge_jump_mask

    def apply_fitted_edge_filter(self, threshold):
        """
        return: ndarray, same dimension as spectrum.shape
        """
        return self.spectrum * self.fitted_edge_mask

    def set_external_mask(self, mask):
        self.external_mask = mask
        
    def apply_external_mask(self):
        return self.spectrum * self.mask

    def remove_sli(self, id_list):
        self.spectrum = np.delete(self.spectrum, id_list, axis=0)
        self.eng = np.delete(self.eng, id_list, axis=0)
        self.removed_img_ids = id_list
        self.pre_es_idx = xm.index_of(self.eng, self.pre_es)
        self.pre_ee_idx = xm.index_of(self.eng, self.pre_ee)
        self.post_es_idx = xm.index_of(self.eng, self.post_es)
        self.post_ee_idx = xm.index_of(self.eng, self.post_ee)

    def fit_whiteline_polynomial(self, peak_es, peak_ee, order=3, ufac=10, flt_spec=False):
        self.wl_es_idx = xm.index_of(self.eng, peak_es)
        self.wl_ee_idx = xm.index_of(self.eng, peak_ee)

        if type(flt_spec) == str:
            if flt_spec == 'True':
                flt_spec = True
            else:
                flt_spec = False
        if flt_spec:
            flt_kernal = np.int32(np.ones(len(self.spectrum.shape)))
            flt_kernal[0] = 3
            self.wl_fit = xm.fit_polynd(np.arange(self.wl_es_idx, self.wl_ee_idx),
                                                  median_filter(self.spectrum[self.wl_es_idx:self.wl_ee_idx, :], size=flt_kernal),
                                                  order)
        else:
            self.wl_fit = xm.fit_polynd(np.arange(self.wl_es_idx, self.wl_ee_idx),
                                        self.spectrum[self.wl_es_idx:self.wl_ee_idx, :], order)
        us_idx = np.linspace(self.wl_es_idx, self.wl_ee_idx,
                              num=(self.wl_ee_idx - self.wl_es_idx + 1)*ufac)
        for ii in range(1, len(self.wl_fit.shape)):
            us_idx = us_idx[:, np.newaxis]

        self.wl_pos_us_idx = np.argmax(np.polyval(self.wl_fit, us_idx), axis=0)
        self.wl_pos = xm.index_lookup(self.wl_pos_us_idx,
                                      self.eng[self.wl_es_idx:self.wl_ee_idx],
                                      ufac=ufac)[:]
        
    def fit_whiteline_scipy(self, peak_es, peak_ee, 
                      model='lorentzian', fvars=None, 
                      bnds=None, ftol=1e-7, xtol=1e-7, 
                      gtol=1e-7, jac='3-point', 
                      method='trf', ufac=100):    
        idx_s = xm.index_of(self.eng, peak_es)
        idx_e = xm.index_of(self.eng, peak_ee)
        if fvars is None:
            self.wl_fvars = (0.01, self.preset_edge_eng, 2)
        else:
            self.wl_fvars = fvars
        self.wl_model = model
        self.wl_bnds = bnds
        self.wl_ftol = ftol
        self.wl_xtol = xtol
        self.wl_gtol = gtol
        self.wl_jac = jac
        self.wl_method = method
        self.wl_peak_fit_coef = xm.fit_peak_scipy(self.eng[idx_s:idx_e], 
                                                  self.spectrum[idx_s:idx_e, :].reshape([idx_e-idx_s, -1]), 
                                                  self.wl_model, 
                                                  self.wl_fvars, 
                                                  bnds=self.wl_bnds,
                                                  ftol = self.wl_ftol, 
                                                  xtol = self.wl_xtol, 
                                                  gtol = self.wl_gtol,
                                                  jac = self.wl_jac, 
                                                  method = self.wl_method)
        # return self.wl_peak_fit_coef
        eng = np.linspace(self.eng[idx_s], self.eng[idx_e], (idx_e-idx_s)*ufac)
        self.wl_pos = np.array(xm.find_peak_map_scipy(self.wl_model, eng, 
                                                  self.wl_peak_fit_coef)).reshape(self.spectrum.shape[1:])
        
        # us_idx = np.linspace(idx_s, idx_e, num=(idx_e - idx_s + 1)*ufac)
        # # for ii in range(1, len(self.edge_fit.shape)):
        # #     us_idx = us_idx[:, np.newaxis]
            
        # self.edge_pos_us_idx = xm.find_deriv_peak_map_scipy(model, us_idx, self.edge_peak_fit_coef)
        # self.edge_pos = xm.index_lookup(self.edge_pos_us_idx,
        #                                 self.eng[idx_s:idx_e],
        #                                 ufac=ufac)[:]
        
        # if self.spec_normalized:
        #     self.edge_pos_0p5_us_idx = xm.find_edge_0p5_map_scipy(model, us_idx, self.edge_peak_fit_coef)
        #     self.edge_pos_0p5 = xm.index_lookup(self.edge_pos_0p5_us_idx,
        #                                         self.eng[idx_s:idx_e],
        #                                         ufac=ufac)[:]
        self.edge_pos = xm.find_deriv_peak_map_scipy(model, eng, self.edge_peak_fit_coef)

        if self.spec_normalized:
            self.edge_pos_0p5 = xm.find_edge_0p5_map_scipy(model, eng, self.edge_peak_fit_coef)
        

    def calc_direct_whiteline(self, peak_es, peak_ee):
        self.wl_es_idx = xm.index_of(self.eng, peak_es)
        self.wl_ee_idx = xm.index_of(self.eng, peak_ee)

        eng = self.eng[self.wl_es_idx:self.wl_ee_idx]
        for ii in range(1, len(self.spectrum.shape)):
            eng = eng[:, np.newaxis]

        self.wl_pos_direct = np.squeeze(eng[np.argmax(self.spectrum[self.wl_es_idx:self.wl_ee_idx, :], axis=0)])

    def calc_direct_whiteline_peak_height(self, peak_es, peak_ee):
        self.wl_es_idx = xm.index_of(self.eng, peak_es)
        self.wl_ee_idx = xm.index_of(self.eng, peak_ee)

        eng = self.eng[self.wl_es_idx:self.wl_ee_idx]
        for ii in range(1, len(self.spectrum.shape)):
            eng = eng[:, np.newaxis]
        self.direct_wl_ph = np.squeeze(np.max(self.spectrum[self.wl_es_idx:self.wl_ee_idx, :], axis=0))

    def calc_weighted_eng(self, eng_s):
        if self.direct_wl_ph[0] is None:
            print('Please calculate the whiteline peak height first. Quit!')
            return 1
        if (self.wl_pos_direct[0] is None) and (self.wl_pos[0] is None):
            print('Please calculate the whiteline energy first. Quit!')
            return 1
        if self.wl_pos[0] is not None:
            eng_e = self.wl_pos
        else:
            eng_e = self.wl_pos_direct
        eng = deepcopy(self.eng)
        for ii in range(0, len(eng_e.shape)):
            eng = eng[:, np.newaxis]

        a = (eng>=eng_s) & (eng<=eng_e+0.0002)
        b = (np.where(a, self.spectrum, 0) * eng * (np.roll(eng, -1, axis=0)-eng)).sum(axis=0)
        c = (np.where(a, self.spectrum, 0)* (np.roll(eng, -1, axis=0)-eng)).sum(axis=0)
        d = np.where(a, eng * (np.roll(eng, -1, axis=0)-eng), 0).sum(axis=0)
        e = (np.where(a, self.spectrum, 0) * np.abs(eng-eng_e) * (np.roll(eng, -1, axis=0)-eng)).sum(axis=0)

        self.centroid_of_eng = b/c
        self.centroid_of_eng_rel_wl = e/c
        self.weighted_atten = b/d
        self.weighted_eng = b/self.direct_wl_ph

    def fit_edge_polynomial(self, es, ee, order=3, ufac=10, flt_spec=False):
        idx_s = xm.index_of(self.eng, es)
        idx_e = xm.index_of(self.eng, ee)
        eng = deepcopy(self.eng[idx_s:idx_e])
        for ii in range(1, self.normalized_spectrum.ndim):
            eng = eng[:, np.newaxis]

        if type(flt_spec) == str:
            if flt_spec == 'True':
                flt_spec = True
            else:
                flt_spec = False
        if flt_spec or (flt_spec == 'True'):
            flt_kernal = np.int32(np.ones(len(self.spectrum.shape)))
            flt_kernal[0] = 3
            self.edge_fit = xm.fit_polynd(np.arange(idx_s, idx_e),
                                          median_filter(self.normalized_spectrum[idx_s:idx_e, :], size=flt_kernal), order)
        else:
            self.edge_fit = xm.fit_polynd(np.arange(idx_s, idx_e),
                                          self.normalized_spectrum[idx_s:idx_e, :], order)
        us_idx = np.linspace(idx_s, idx_e, num=(idx_e - idx_s + 1)*ufac)
        for ii in range(1, len(self.edge_fit.shape)):
            us_idx = us_idx[:, np.newaxis]

        self.edge_pos_0p5_us_idx = xm.index_of(np.polyval(self.edge_fit, us_idx), 0.5)
        self.edge_pos_0p5 = xm.index_lookup(self.edge_pos_0p5_us_idx,
                                        self.eng[idx_s:idx_e],
                                        ufac=ufac)[:]
        self.edge_pos_us_idx = np.argmax(np.gradient(np.polyval(self.edge_fit, us_idx), axis=0)/
                                             np.gradient(us_idx, axis=0), axis=0)
        self.edge_pos = xm.index_lookup(self.edge_pos_us_idx,
                                        self.eng[idx_s:idx_e],
                                        ufac=ufac)[:]
        
    def fit_edge_scipy(self, es, ee, 
                       model='lorentzian', fvars=None, 
                       bnds=None, ftol=1e-7, xtol=1e-7, 
                       gtol=1e-7, jac='3-point', 
                       method='trf', ufac=100):    
        idx_s = xm.index_of(self.eng, es)
        idx_e = xm.index_of(self.eng, ee)
        if fvars is None:
            self.edge_fvars = (0.01, self.preset_edge_eng, 2)
        else:
            self.edge_fvars = fvars
        self.edge_model = model
        self.edge_bnds = bnds
        self.edge_ftol = ftol
        self.edge_xtol = xtol
        self.edge_gtol = gtol
        self.edge_jac = jac
        self.edge_method = method
        self.edge_peak_fit_coef = xm.fit_peak_scipy(self.eng[idx_s:idx_e], 
                                                    self.spectrum[idx_s:idx_e, :].reshape([idx_e-idx_s, -1]), 
                                                    self.edge_model, 
                                                    self.edge_fvars, 
                                                    bnds=self.edge_bnds,
                                                    ftol = self.edge_ftol, 
                                                    xtol = self.edge_xtol, 
                                                    gtol = self.edge_gtol,
                                                    jac = self.edge_jac, 
                                                    method = self.edge_method)        
        us_idx = np.linspace(idx_s, idx_e, num=(idx_e - idx_s + 1)*ufac)
        # for ii in range(1, len(self.edge_peak_fit_coef)):
        #     us_idx = us_idx[:, np.newaxis]

        self.edge_pos_0p5_us_idx = xm.find_edge_0p5_map_scipy(model, us_idx, self.edge_peak_fit_coef)
        self.edge_pos_0p5 = xm.index_lookup(self.edge_pos_0p5_us_idx,
                                            self.eng[idx_s:idx_e],
                                            ufac=ufac)[:]
        self.edge_pos_us_idx = xm.find_deriv_peak_map_scipy(model, us_idx, self.edge_peak_fit_coef)
        self.edge_pos = xm.index_lookup(self.edge_pos_us_idx,
                                        self.eng[idx_s:idx_e],
                                        ufac=ufac)[:]
    
    ### adopted from larch
    def find_e0(self):
        """calculate :math:`E_0`, the energy threshold of absorption, or
        'edge energy', given :math:`\mu(E)`.
    
        :math:`E_0` is found as the point with maximum derivative with
        some checks to avoid spurious glitches.
    
        Arguments:
            energy (ndarray or group): array of x-ray energies, in eV, or group
            mu     (ndaarray or None): array of mu(E) values
            group  (group or None):    output group
            _larch (larch instance or None):  current larch session.
    
        Returns:
            float: Value of e0. If a group is provided, group.e0 will also be set.
    
        Notes:
            1. Supports :ref:`First Argument Group` convention, requiring group members `energy` and `mu`
            2. Supports :ref:`Set XAFS Group` convention within Larch or if `_larch` is set.
        """
        def _finde0(energy, mu):
            if len(energy.shape) > 1:
                energy = energy.squeeze()
            if len(mu.shape) > 1:
                mu = mu.squeeze()
        
            dmu = np.gradient(mu)/np.gradient(energy)
            # find points of high derivative
            dmu[np.where(~np.isfinite(dmu))] = -1.0
            nmin = max(3, int(len(dmu)*0.05))
            maxdmu = max(dmu[nmin:-nmin])
        
            high_deriv_pts = np.where(dmu >  maxdmu*0.1)[0]
            idmu_max, dmu_max = 0, 0
        
            for i in high_deriv_pts:
                if i < nmin or i > len(energy) - nmin:
                    continue
                if (dmu[i] > dmu_max and
                    (i+1 in high_deriv_pts) and
                    (i-1 in high_deriv_pts)):
                    idmu_max, dmu_max = i, dmu[i]
        
            return energy[idmu_max]
        ids = xm.index_of(self.eng, self.pre_ee)
        ide = xm.index_of(self.eng, self.post_es)
        eng = self.eng[ids:ide]
        dim = self.spectrum[ids:ide, ...].shape
        spec = self.spectrum[ids:ide, ...].reshape([dim[0], -1])
        n_cpu = os.cpu_count()
        with mp.Pool(n_cpu-1) as pool:
            rlt = pool.starmap(_finde0, [(eng, spec[:, ii]) for ii in np.int32(np.arange(spec.shape[1]))])
        pool.close()
        pool.join()
        self.auto_e0 = np.array(rlt).reshape(dim[1:])
        
    def update_with_auto_e0(self):
        self.find_e0()

        self.pre_ee = self.auto_e0 + (self.pre_ee-self.preset_edge_eng)
        self.pre_ee_idx = xm.index_of(self.eng, self.pre_ee)

        self.post_es = self.auto_e0 + (self.post_es-self.preset_edge_eng)
        self.post_es_idx = xm.index_of(self.eng, self.post_es)
        
    def save_results(self, filename, dtype='2D_XANES', **kwargs):
        filename = Path(filename)
        assert filename.suffix.lower() in ['.h5', '.hdf', '.hdf5'], \
               'unsupported file format...'
        if not filename.parent.exists():
            filename.parent.Path.mkdir(parents=True)

        if dtype == '2D_XANES':
            f = h5py.File(filename, 'a')
            if 'XANES_preprocessing' not in f:
                g1 = f.create_group('XANES_preprocessing')
            else:
                del f['XANES_preprocessing']
                g1 = f.create_group('XANES_preprocessing')

            g1.create_dataset('preset_edge_eng', data=self.preset_edge_eng)
            g1.create_dataset('removed_img_ids', data=self.removed_img_ids)
            g1.create_dataset('pre_edge_fit_coef', data=self.pre_edge_fit_coef)
            g1.create_dataset('post_edge_fit_coef', data=self.post_edge_fit_coef)
            g1.create_dataset('pre_edge_sd_map', data=self.pre_edge_sd_map)
            g1.create_dataset('post_edge_sd_map', data=self.post_edge_sd_map)
            g1.create_dataset('pre_edge_avg', data=self.pre_avg)
            g1.create_dataset('post_edge_avg', data=self.post_avg)
            g1.create_dataset('edge_jump_map', data=self.edge_jump_map)
            g11 = g1.create_group('fitting_configurations')
            g11.create_dataset('pre_edge_energy_range', data=[self.pre_es, self.pre_ee])
            g11.create_dataset('post_edge_energy_range', data=[self.post_es, self.post_ee])

            if 'XANES_filtering' not in f:
                g2 = f.create_group('XANES_filtering')
            else:
                del f['XANES_filtering']
                g2 = f.create_group('XANES_filtering')

            g21 = g2.create_group('edge_jump_filter')
            g21.create_dataset('edge_jump_threshold', data=self.edge_jump_threshold)
            g21.create_dataset('edge_jump_mask', data=self.edge_jump_mask.astype(np.int8))
            g22 = g2.create_group('fitted_edge_fitler')
            g22.create_dataset('fitted_edge_threshold', data=self.fitted_edge_threshold)
            g22.create_dataset('fitted_edge_mask', data=self.fitted_edge_mask.astype(np.int8))

            if 'XANES_results' not in f:
                g3 = f.create_group('XANES_results')
            else:
                del f['XANES_results']
                g3 = f.create_group('XANES_results')

            g3.create_dataset('eng', data=self.eng.astype(np.float32), dtype=np.float32)
            g3.create_dataset('normalized_spectrum', data=self.normalized_spectrum.astype(np.float32), dtype=np.float32)
            g3.create_dataset('edge_pos', data=self.auto_e0.astype(np.float32), dtype=np.float32)
            g3.create_dataset('edge_0.5_pos', data=self.edge_pos.astype(np.float32), dtype=np.float32)
            g3.create_dataset('whiteline_pos', data=self.wl_pos.astype(np.float32), dtype=np.float32)
            g3.create_dataset('whiteline_pos_direct', data=self.wl_pos_direct.astype(np.float32), dtype=np.float32)

            f.close()
        elif dtype == '3D_XANES':
            f = h5py.File(filename, 'a')
            if 'processed_XANES3D' not in f:
                g1 = f.create_group('processed_XANES3D')
            else:
                del f['processed_XANES3D']
                g1 = f.create_group('processed_XANES3D')

            g1.create_dataset('removed_img_ids',
                              data=self.removed_img_ids)
            g1.create_dataset('pre_edge_fit_coef',
                              data=self.pre_edge_fit_coef)
            g1.create_dataset('post_edge_fit_coef',
                              data=self.post_edge_fit_coef)
            g1.create_dataset('pre_edge_sd_map',
                              data=self.pre_edge_sd_map)
            g1.create_dataset('post_edge_sd_map',
                              data=self.post_edge_sd_map)
            g1.create_dataset('pre_edge_avg',
                              data=self.pre_avg)
            g1.create_dataset('post_edge_avg',
                              data=self.post_avg)
            g1.create_dataset('edge_jump_map',
                              data=self.edge_jump_map)
            g1.create_dataset('edge_jump_threshold',
                              data=self.edge_jump_threshold)
            g1.create_dataset('fitted_edge_threshold',
                              data=self.fitted_edge_threshold)
            g11 = g1.create_group('fitting_configurations')
            g11.create_dataset('pre_edge_energy_range',
                               data=[self.pre_es, self.pre_ee])
            g11.create_dataset('post_edge_energy_range',
                               data=[self.post_es, self.post_ee])

            if 'XANES_results' not in f:
                g3 = f.create_group('XANES_results')
            else:
                del f['XANES_results']
                g3 = f.create_group('XANES_results')

            g3.create_dataset('eng',
                              data=self.eng.astype(np.float32),
                              dtype=np.float32)
            for key, item in kwargs.items():
                g3.create_dataset(key,
                                  data=item.astype(np.float32),
                                  dtype=np.float32)
            f.close()
        else:
            print('Unrecognized data type!')
