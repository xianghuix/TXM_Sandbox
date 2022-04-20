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

"""
    This class include xanes spectrum analyses. These analyses are based on filtered
    xanes spectra. These analyses include:
        1. LC analysis
        2. PCA
        3. Curve fitting
        4. Peak fitting
        5.
"""
class xanes_analysis():
    def __init__(self, spectrum, eng, preset_edge_eng,
                 pre_es=None, pre_ee=-0.05,
                 post_es=0.1, post_ee=None,
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
        self.pre_edge_threshold = pre_edge_threshold
        self.pre_edge_sd = None
        self.post_edge_sd = None
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
        self.pre_avg = []
        self.post_avg = []
        self.edge_jump_map = []
        self.fitted_edge_threshold = []

        self.wl_pos_direct = np.array([None])
        self.direct_wl_ph = np.array([None])
        self.wl_pos = np.array([None])
        self.edge_eng_pos = np.array([None])

    def edge_jump(self):
        """
        return: ndarray, edge_jump_map has dimension of spectrum.shape[1:]
        """
        self.pre_avg = self.spectrum[self.pre_es_idx:self.pre_ee_idx].mean(axis=0)
        self.post_avg = self.spectrum[self.post_es_idx:self.post_ee_idx].mean(axis=0)
        self.edge_jump_map = np.abs(self.post_avg - self.pre_avg)

    def cal_pre_edge_sd(self):
        """
        return: ndarray, pre_edge_sd has dimension of spectrum.shape[1:]
        """
        self.pre_edge_sd_map = self.spectrum[self.pre_es_idx:self.pre_ee_idx, :].std(axis=0)

    def cal_post_edge_sd(self):
        """
        return: ndarray, post_edge_sd has dimension of spectrum.shape[1:]
        """
        self.post_edge_sd_map = self.spectrum[self.post_es_idx:self.post_ee_idx, :].std(axis=0)

    def fit_pre_edge(self):
        """
        return: ndarray, pre_edge_fit has dimension of [2].append(list(spectrum.shape[1:]))
        """
        self.pre_edge_fit_coef = xm.fit_polynd(self.eng[self.pre_es_idx:self.pre_ee_idx],
                                               self.spectrum[self.pre_es_idx:self.pre_ee_idx].\
                                                   reshape((self.pre_ee_idx-self.pre_es_idx, -1)), 1).\
                                                   reshape([2]+list(self.spectrum.shape[1:]))

    def fit_post_edge(self):
        """
        return: ndarray, post_edge_fit has dimension of [2].append(list(spectrum.shape[1:]))
        """
        self.post_edge_fit_coef = xm.fit_polynd(self.eng[self.post_es_idx:self.post_ee_idx],
                                                self.spectrum[self.post_es_idx:self.post_ee_idx].\
                                                    reshape(self.post_ee_idx-self.post_es_idx, -1), 1).\
                                                    reshape([2]+list(self.spectrum.shape[1:]))

    def create_edge_jump_filter(self, threshold):
        """
        return: ndarray, same dimension as spectrum.shape
        """
        self.edge_jump_threshold = threshold
        # self.edge_jump_mask = xf.edge_jump_filter(self.edge_jump_map, self.pre_edge_sd_map,
        #                                           self.edge_jump_threshold)
        self.edge_jump_mask = xf.edge_jump_filter(self.preset_edge_eng, self.pre_edge_fit_coef,
                                                  self.post_edge_fit_coef, self.pre_edge_sd_map,
                                                  self.edge_jump_threshold)

    def create_fitted_edge_filter(self, threshold):
        """
        return: ndarray, same dimension as spectrum.shape
        """
        self.fitted_edge_threshold = threshold
        self.fitted_edge_mask = xf.fitted_edges_filter(self.eng, self.pre_edge_fit_coef,
                                                       self.post_edge_fit_coef, self.pre_edge_sd_map,
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

    def remove_sli(self, id_list):
        self.spectrum = np.delete(self.spectrum, id_list, axis=0)
        self.eng = np.delete(self.eng, id_list, axis=0)
        self.removed_img_ids = id_list
        self.pre_es_idx = xm.index_of(self.eng, self.pre_es)
        self.pre_ee_idx = xm.index_of(self.eng, self.pre_ee)
        self.post_es_idx = xm.index_of(self.eng, self.post_es)
        self.post_ee_idx = xm.index_of(self.eng, self.post_ee)

    def normalize_xanes(self, eng0, order=0):
        """
        normalize_xanes includes two steps: 1) pre-edge background subtraction (order = 0)),
                                            2) post-edge fitted or average mu normalization (order = 1)
        return: ndarray, same dimension as spectrum.shape
        """
        eng = deepcopy(self.eng)
        for ii in range(1, self.pre_edge_fit_coef.ndim):
            eng = eng[:, np.newaxis]
        if order == 0:
            self.normalized_spectrum = (self.spectrum -
                                        (self.pre_edge_fit_coef[0][np.newaxis, :]*eng + self.pre_edge_fit_coef[1][np.newaxis, :]))/(
                                        self.post_edge_fit_coef[0][np.newaxis, :]*eng0 + self.post_edge_fit_coef[1][np.newaxis, :] -
                                        (self.pre_edge_fit_coef[0][np.newaxis, :]*eng + self.pre_edge_fit_coef[1][np.newaxis, :]))
            self.normalized_spectrum[np.isnan(self.normalized_spectrum)] = 0
            self.normalized_spectrum[np.isinf(self.normalized_spectrum)] = 0
        elif order == 1:
            self.normalized_spectrum = (self.spectrum -
                                        (self.pre_edge_fit_coef[0][np.newaxis, :]*eng + self.pre_edge_fit_coef[1][np.newaxis, :]))/(
                                        self.post_edge_fit_coef[0][np.newaxis, :]*eng + self.post_edge_fit_coef[1][np.newaxis, :] -
                                        (self.pre_edge_fit_coef[0][np.newaxis, :]*eng + self.pre_edge_fit_coef[1][np.newaxis, :]))
            self.normalized_spectrum[np.isnan(self.normalized_spectrum)] = 0
            self.normalized_spectrum[np.isinf(self.normalized_spectrum)] = 0
        else:
            print('Order can only be "1" or "0".')
            pass

    def fit_edge_pos(self, es, ee, order=3, us_ratio=10, flt_spec=False):
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
        if flt_spec | (flt_spec == 'True'):
            flt_kernal = np.int32(np.ones(len(self.spectrum.shape)))
            flt_kernal[0] = 3
            self.edge_fit = xm.fit_polynd(np.arange(idx_s, idx_e),
                                          median_filter(self.normalized_spectrum[idx_s:idx_e, :], size=flt_kernal), order)
        else:
            self.edge_fit = xm.fit_polynd(np.arange(idx_s, idx_e),
                                          self.normalized_spectrum[idx_s:idx_e, :], order)
        us_idx = np.linspace(idx_s, idx_e, num=(idx_e - idx_s + 1)*us_ratio)
        for ii in range(1, len(self.edge_fit.shape)):
            us_idx = us_idx[:, np.newaxis]

        self.edge_pos_us_idx = xm.index_of(np.polyval(self.edge_fit, us_idx), 0.5)
        self.edge_pos = xm.index_lookup(self.edge_pos_us_idx,
                                        self.eng[idx_s:idx_e],
                                        us_ratio=us_ratio)[:]

    def fit_whiteline(self, peak_es, peak_ee, order=3, us_ratio=10, flt_spec=False):
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
                             num=(self.wl_ee_idx - self.wl_es_idx + 1)*us_ratio)
        for ii in range(1, len(self.wl_fit.shape)):
            us_idx = us_idx[:, np.newaxis]

        self.wl_pos_us_idx = np.argmax(np.polyval(self.wl_fit, us_idx), axis=0)
        self.wl_pos = xm.index_lookup(self.wl_pos_us_idx,
                                      self.eng[self.wl_es_idx:self.wl_ee_idx],
                                      us_ratio=us_ratio)[:]

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
            g3.create_dataset('edge_eng_pos', data=self.edge_eng_pos.astype(np.float32), dtype=np.float32)
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
