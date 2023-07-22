#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 12:46:23 2019

@author: xiao
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter
from scipy.interpolate import splrep, splev
import h5py, os, time
from copy import deepcopy
import multiprocess as mp
from tomopy.util.mproc import distribute_jobs

from .misc import msgit
from . import xanes_math as xm

N_CPU = os.cpu_count()-1 if os.cpu_count()>1 else os.cpu_count()
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
        self.spec = spectrum
        self.eng = eng
        self.preset_edge_eng = preset_edge_eng
        self.model = {'edge': {}, 'wl': {}}
        self.lcf_use = False
        self.lcf_constr_use = True
        self.lcf_ref_spec = None
        self.lcf_ref = None
        self.lcf_fit = None
        self.lcf_model = {'constr': self.lcf_constr_use, 'ref': self.lcf_ref, 'rlt': None}

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

        self.edge_jump_thres = edge_jump_threshold
        self.fitted_edge_thres = pre_edge_threshold
        self.auto_e0 = [None]
        self.pre_edge_fit = None
        self.post_edge_fit = None
        self.edge_jump_mask = None
        self.fitted_edge_mask = None
        self.norm_spec = None
        self.fitted_edge_flted_spec = None
        self.edge_jump_flted_spec = None
        self.removed_img_ids = []
        self.post_edge_fit_rlt = np.empty([])
        self.pre_edge_fit_rlt = np.empty([])
        self.pre_edge_sd_map = np.empty([])
        self.post_edge_sd_map = np.empty([])
        self.pre_edge_mean_map = np.empty([])
        self.post_edge_mean_map = np.empty([])
        self.pre_avg = np.empty([])
        self.post_avg = np.empty([])
        self.edge_jump_map = np.empty([])
        self.centroid_of_eng = np.empty([])
        self.centroid_of_eng_rel_wl = np.empty([])
        self.weighted_atten = np.empty([])
        self.weighted_eng = np.empty([])
        self.edge0p5_pos_fit = np.empty([])
        self.edge_fit_rlt = np.empty([])
        self.wl_fit_rlt = np.empty([])

        self.wl_pos_dir = np.empty([])
        self.wl_ph_dir = np.empty([])
        self.wl_pos_fit = np.empty([])
        self.edge_pos = np.empty([])

        self.pre_edge_fitted = False
        self.post_edge_fitted = False
        self.spec_normalized = False
        self.spec_linear_fitted = False
        self.spec_pca_fitted = False

    def apply_edge_jump_filter(self):
        """
        return: ndarray, same dimension as spectrum.shape
        """
        return self.spec * self.edge_jump_mask

    def apply_external_mask(self):
        return self.spec * self.ext_mask

    def apply_fitted_edge_filter(self):
        """
        return: ndarray, same dimension as spectrum.shape
        """
        return self.spec * self.fitted_edge_mask

    def cal_pre_edge_sd(self):
        """
        return: ndarray, pre_edge_sd has dimension of spectrum.shape[1:]
        """
        self.pre_edge_sd_map = self.spec[self.pre_es_idx:self.pre_ee_idx,
                               :].std(axis=0)

    def cal_post_edge_sd(self):
        """
        return: ndarray, post_edge_sd has dimension of spectrum.shape[1:]
        """
        self.post_edge_sd_map = self.spec[self.post_es_idx:self.post_ee_idx,
                                :].std(axis=0)

    def cal_pre_edge_mean(self):
        """
        return: ndarray, pre_edge_sd has dimension of spectrum.shape[1:]
        """
        self.pre_edge_mean_map = self.spec[self.pre_es_idx:self.pre_ee_idx, :].mean(axis=0)

    def cal_post_edge_mean(self):
        """
        return: ndarray, post_edge_sd has dimension of spectrum.shape[1:]
        """
        self.post_edge_mean_map = self.spec[self.post_es_idx:self.post_ee_idx,
                                  :].mean(axis=0)

    def cal_fit(self, fit_coef, eng=None, reshape=True):
        if eng is None:
            eng = self.eng
        if not isinstance(eng, np.ndarray):
            eng = np.array(eng)
        fit = xm.eval_polynd(fit_coef, eng)
        if reshape:
            if eng.shape:
                return fit.reshape([eng.shape[0], *self.spec.shape[1:]])
            else:
                return fit.reshape([*self.spec.shape[1:]])
        else:
            return fit

    @msgit(wd=100, fill='+')
    def cal_pre_edge_fit(self, eng=None, reshape=True):
        """
        inputs:
            eng: 1D array-like; full energy list of the spectrum
            pre_edge_fit_coef: array-like; pixel-wise pre_edge fit coef
        ouputs:
            pre_edge_fit: array-like; pixel-wise line profile over eng
        """
        self.fit_pre_edge()
        return self.cal_fit(self.pre_edge_fit_rlt[0], eng=eng, reshape=reshape)

    @msgit(wd=100, fill='+')
    def cal_post_edge_fit(self, eng=None, reshape=True):
        """
        inputs:
            eng: 1D array-like; full energy list of the spectrum
            post_edge_fit_coef: array-like; pixel-wise post_edge fit coef
        ouputs:
            post_edge_fit: array-like; pixel-wise line profile over eng
        """
        self.fit_post_edge()
        return self.cal_fit(self.post_edge_fit_rlt[0], eng=eng, reshape=reshape)

    @msgit(wd=100, fill='+')
    def cal_edge_jump_map(self):
        """
        return: ndarray, edge_jump_map has dimension of spectrum.shape[1:]
        """
        self.edge_jump_map = (self.cal_post_edge_fit(np.array([self.preset_edge_eng])) -
                              self.cal_pre_edge_fit(np.array([self.preset_edge_eng])))

    def cal_drt_wl_peak_hgt(self, peak_es, peak_ee):
        self.wl_es_idx = xm.index_of(self.eng, peak_es)
        self.wl_ee_idx = xm.index_of(self.eng, peak_ee)

        eng = self.eng[self.wl_es_idx:self.wl_ee_idx]
        for ii in range(1, len(self.spec.shape)):
            eng = eng[:, np.newaxis]
        self.wl_ph_dir = np.squeeze(np.max(self.spec[self.wl_es_idx:self.wl_ee_idx, :], axis=0))

    def cal_wgt_eng(self, eng_s, eng_e=None):
        if (not self.wl_ph_dir.shape) and (not self.wl_ph_fit.shape):
            print('Please calculate the whiteline peak height first. Quit!')
            self.centroid_of_eng = np.empty([0])
            self.centroid_of_eng_rel_wl = np.empty([0])
            self.weighted_atten = np.empty([0])
            self.weighted_eng = np.empty([0])
            return
        if (not self.wl_pos_dir.shape) and (not self.wl_pos_fit.shape):
            print('Please calculate the whiteline energy first. Quit!')
            self.centroid_of_eng = np.empty([0])
            self.centroid_of_eng_rel_wl = np.empty([0])
            self.weighted_atten = np.empty([0])
            self.weighted_eng = np.empty([0])
            return
        if eng_e is None:
            if self.wl_pos_fit.shape:
                eng_e = self.wl_pos_fit
            else:
                eng_e = self.wl_pos_dir
        eng = np.squeeze(deepcopy(self.eng))
        for ii in range(0, len(self.spec.shape)-1):
            eng = eng[:, np.newaxis]
        
        if self.wl_ph_fit.shape:
            wl_ph = self.wl_ph_fit
        else:
            wl_ph = self.wl_ph_dir

        a = (eng >= eng_s) & (eng <= eng_e + 0.0002)
        b = (np.where(a, self.spec, 0) * eng * (np.roll(eng, -1, axis=0) - eng)).sum(axis=0)
        c = (np.where(a, self.spec, 0) * (np.roll(eng, -1, axis=0) - eng)).sum(axis=0)
        d = np.where(a, eng * (np.roll(eng, -1, axis=0) - eng), 0).sum(axis=0)
        e = (np.where(a, self.spec, 0) * np.abs(eng - eng_e) * (np.roll(eng, -1, axis=0) - eng)).sum(axis=0)

        self.centroid_of_eng = b / c
        self.centroid_of_eng_rel_wl = e / c
        self.weighted_atten = b / d
        self.weighted_eng = b / wl_ph

    @msgit(wd=100, fill='+')
    def create_edge_jump_filter(self, eng0=None):
        """
        return: ndarray, same dimension as spectrum.shape
        """
        if eng0 is None:
            eng0 = self.edge_eng
        self.edge_jump_mask = ((self.cal_post_edge_fit(eng0) -
                                self.cal_pre_edge_fit(eng0))
                               > self.edge_jump_thres * self.pre_edge_sd_map).astype(np.int8)

    @msgit(wd=100, fill='+')
    def create_fitted_edge_filter(self):
        """
        return: ndarray, same dimension as spectrum.shape
        """
        self.fitted_edge_mask = np.any((self.cal_post_edge_fit(self.eng) -
                                        self.cal_pre_edge_fit(self.eng))
                                       > self.fitted_edge_thres * self.pre_edge_sd_map, axis=0).astype(np.int8)

    @msgit(wd=100, fill='$')
    def find_edge_0p5(self, es, ee, optimizer='model', ufac=20):
        idx_s = xm.index_of(self.eng, es)
        idx_e = xm.index_of(self.eng, ee)
        us_eng = np.linspace(self.eng[idx_s], self.eng[idx_e],
                             num=(idx_e - idx_s + 1) * ufac) - self.model['edge']['eoff']
        if optimizer == 'model':
            self.edge0p5_pos_fit = xm.find_fit_val(self.model['edge']['model'],
                                                   np.array(self.model['edge']['fit_rlt'][0]),
                                                   us_eng, v=0.5).reshape(self.spec.shape[1:]) + \
                                   self.model['edge']['eoff']
        elif optimizer == 'direct':
            self.edge0p5_pos_dir = xm.lookup(self.eng[idx_s:idx_e],
                                             self.norm_spec[idx_s:idx_e].reshape(idx_e - idx_s, -1), 0.5). \
                reshape(self.spec.shape[1:])

    @msgit(wd=100, fill='$')
    def find_edge_50(self, ees, eee, pes, pee, optimizer='model', ufac=20):
        idx_es = xm.index_of(self.eng, ees)
        idx_ee = xm.index_of(self.eng, eee)
        idx_ps = xm.index_of(self.eng, pes)
        idx_pe = xm.index_of(self.eng, pee)
        us_e_eng = np.linspace(self.eng[idx_es], self.eng[idx_ee], num=(idx_ee - idx_es) * ufac + 1)
        us_p_eng = np.linspace(self.eng[idx_ps], self.eng[idx_pe], num=(idx_pe - idx_ps) * ufac + 1)
        if optimizer == 'both':
            print('both')
            self.edge50_pos_fit = xm.find_50_peak(self.model['edge']['model'], us_e_eng-self.model['edge']['eoff'],
                                                  np.array(self.model['edge']['fit_rlt'][0]),
                                                  self.model['wl']['model'], us_p_eng-self.model['wl']['eoff'],
                                                  np.array(self.model['wl']['fit_rlt'][0]), ftype=optimizer). \
                    reshape(self.spec.shape[1:]) + self.model['edge']['eoff']
        elif optimizer == 'wl':
            print('wl')
            self.edge50_pos_fit = xm.find_50_peak(us_e_eng, self.eng[idx_es:idx_ee+1],
                                                  self.norm_spec[idx_es:idx_ee+1].reshape([idx_ee-idx_es+1, -1]),
                                                  self.model['wl']['model'], us_p_eng-self.model['wl']['eoff'],
                                                  np.array(self.model['wl']['fit_rlt'][0]), ftype=optimizer). \
                    reshape(self.spec.shape[1:])
        elif optimizer == 'edge':
            print('edge')
            self.edge50_pos_fit = xm.find_50_peak(self.model['edge']['model'], us_e_eng-self.model['edge']['eoff'],
                                                  np.array(self.model['edge']['fit_rlt'][0]),
                                                  us_p_eng, self.eng[idx_ps:idx_pe+1],
                                                  self.norm_spec[idx_ps:idx_pe+1].reshape([idx_pe-idx_ps+1, -1]),
                                                  ftype=optimizer). \
                    reshape(self.spec.shape[1:]) + self.model['edge']['eoff']
        elif optimizer == 'none':
            print('none')
            self.edge50_pos_fit_none = xm.find_50_peak(us_e_eng, self.eng[idx_es:idx_ee+1],
                                                  self.norm_spec[idx_es:idx_ee+1].reshape([idx_ee-idx_es+1, -1]),
                                                  us_p_eng, self.eng[idx_ps:idx_pe + 1],
                                                  self.norm_spec[idx_ps:idx_pe+1].reshape([idx_pe-idx_ps+1, -1]),
                                                  ftype=optimizer). \
                    reshape(self.spec.shape[1:])
        elif optimizer == 'direct':
            print('dir')
            self.edge50_pos_dir = xm.lookup(self.eng[idx_es:idx_ee],
                                            self.norm_spec[idx_es:idx_ee].reshape([idx_ee-idx_es, -1]),
                                            0.5*self.norm_spec[idx_ps:idx_pe].reshape([idx_pe-idx_ps, -1]).max(axis=0)). \
                    reshape(self.spec.shape[1:])

    @msgit(wd=100, fill='$')
    def find_wl(self, es, ee, optimizer='model', ufac=20):
        idx_s = xm.index_of(self.eng, es)
        idx_e = xm.index_of(self.eng, ee)
        if optimizer == 'model':
            us_eng = np.linspace(self.eng[idx_s], self.eng[idx_e],
                                 num=(idx_e - idx_s) * ufac) - self.model['wl']['eoff']
            self.wl_pos_fit, self.wl_ph_fit = xm.find_fit_peak(self.model['wl']['model'],
                                               np.array(self.model['wl']['fit_rlt'][0]), us_eng) 
            self.wl_pos_fit += self.model['wl']['eoff']
            self.wl_pos_fit = self.wl_pos_fit.reshape(self.spec.shape[1:])
            self.wl_ph_fit = self.wl_ph_fit.reshape(self.spec.shape[1:])
            print(f"{self.wl_pos_fit.shape=}, {self.wl_ph_fit.shape=}") 
        elif optimizer == 'direct':
            if self.norm_spec is None:
                self.wl_pos_dir = xm.lookup(self.eng[idx_s:idx_e],
                                            self.spec[idx_s:idx_e].reshape(idx_e - idx_s, -1),
                                            self.spec[idx_s:idx_e].reshape(idx_e - idx_s, -1).max(axis=0)) \
                    .reshape(self.spec.shape[1:])
                self.wl_ph_dir = np.squeeze(np.max(self.spec[idx_s:idx_e, :], axis=0))
            else:
                self.wl_pos_dir = xm.lookup(self.eng[idx_s:idx_e],
                                            self.norm_spec[idx_s:idx_e].reshape(idx_e - idx_s, -1),
                                            self.norm_spec[idx_s:idx_e].reshape(idx_e - idx_s, -1).max(axis=0)) \
                    .reshape(self.spec.shape[1:])
                self.wl_ph_dir = np.squeeze(np.max(self.norm_spec[idx_s:idx_e, :], axis=0))

    @msgit(wd=100, fill='$')
    def find_edge_deriv(self, es, ee, optimizer='model', ufac=20):
        idx_s = xm.index_of(self.eng, es)
        idx_e = xm.index_of(self.eng, ee)
        us_eng = np.linspace(self.eng[idx_s], self.eng[idx_e],
                             num=(idx_e - idx_s) * ufac) - self.model['edge']['eoff']
        if optimizer == 'model':
            self.edge_pos_fit = xm.find_deriv_peak(self.model['edge']['model'],
                                                   np.array(self.model['edge']['fit_rlt'][0]), us_eng) \
                                    .reshape(self.spec.shape[1:]) + self.model['edge']['eoff']
        elif optimizer == 'direct':
            if self.norm_spec is None:
                self.edge_pos_dir = self.spec[idx_s:idx_e].max(axis=0)
            else:
                self.edge_pos_dir = self.norm_spec[idx_s:idx_e].max(axis=0)

    def _finde0(energy, mu):
        if len(energy.shape) > 1:
            energy = energy.squeeze()
        if len(mu.shape) > 1:
            mu = mu.squeeze()
        dmu = np.gradient(mu) / np.gradient(energy)
        # find points of high derivative
        dmu[np.where(~np.isfinite(dmu))] = -1.0
        nmin = max(3, int(len(dmu) * 0.05))
        maxdmu = max(dmu[nmin:-nmin])
        high_deriv_pts = np.where(dmu > maxdmu * 0.1)[0]
        idmu_max, dmu_max = 0, 0
        for i in high_deriv_pts:
            if i < nmin or i > len(energy) - nmin:
                continue
            if (dmu[i] > dmu_max and
                    (i + 1 in high_deriv_pts) and
                    (i - 1 in high_deriv_pts)):
                idmu_max, dmu_max = i, dmu[i]
        return energy[idmu_max]

    def find_edge(self):
        """
        ### adopted from larch
        calculate :math:`E_0`, the energy threshold of absorption, or
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
        ids = xm.index_of(self.eng, self.pre_ee)
        ide = xm.index_of(self.eng, self.post_es)
        eng = self.eng[ids:ide]
        dim = self.spec[ids:ide, ...].shape
        spec = self.spec[ids:ide, ...].reshape([dim[0], -1])
        with mp.Pool(N_CPU) as pool:
            rlt = pool.starmap(_finde0, [(eng, spec[:, ii]) for ii in np.int32(np.arange(spec.shape[1]))])
        pool.close()
        pool.join()
        self.edge_pos_dir = np.array(rlt).reshape(dim[1:])

    def find_edge_t(self):
        """calculate :math:`E_0`, the energy threshold of absorption, or
        'edge energy', given :math:`\mu(E)`.

        :math:`E_0` is found as the point with maximum derivative with
        some checks to avoid spurious glitches; using tomopy distribute_jobs
        for multiprocessing

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
        ids = xm.index_of(self.eng, self.pre_ee)
        ide = xm.index_of(self.eng, self.post_es)
        eng = self.eng[ids:ide]

        self.edge_pos_dir = np.ndarray(self.spec.shape[1:])
        distribute_jobs(self.spec[ids:ide, ...], _finde0, 0, kwargs={'energy': eng}, out=self.edge_pos_dir)

    def fit_pre_edge(self):
        """
        return: ndarray, pre_edge_fit has dimension of [2].append(list(spectrum.shape[1:]))
        """
        try:
            kernel = 5 * np.ones([len(self.spec.shape)])
            kernel[0] = 1
            self.pre_edge_fit_rlt = xm.fit_curv_polynd(self.eng[self.pre_es_idx:self.pre_ee_idx],
                                                       uniform_filter(self.spec[self.pre_es_idx:self.pre_ee_idx],
                                                                     size=kernel), 1)
            self.pre_edge_fitted = True
        except Exception as e:
            self.pre_edge_fitted = False
            print(str(e))
            print("pre_edge fitting went wrong")

    def fit_post_edge(self):
        """
        return: ndarray, post_edge_fit has dimension of [2].append(list(spectrum.shape[1:]))
        """
        try:
            kernel = 5*np.ones([len(self.spec.shape)])
            kernel[0] = 1
            self.post_edge_fit_rlt = xm.fit_curv_polynd(self.eng[self.post_es_idx:self.post_ee_idx],
                                                        uniform_filter(self.spec[self.post_es_idx:self.post_ee_idx],
                                                                      size=kernel), 1)
            self.post_edge_fitted = True
        except Exception as e:
            print(type(e))
            print(e.args)
            print(e)
            self.post_edge_fitted = False
            print("post_edge fitting went wrong")

    def fit_spec(self, es, ee, eoff, optimizer, flt_spec, on, *args, **kwargs):
        """
        Provide curve fitting api to that are based on numpy.ployfit and
        scipy.optimize.lsq.
        Parameters
        ----------
        es : flot
            energy starting point in eV.
        ee : flot
            energy end point in eV.
        eoff: float
            energy value offset in eV; for 'numpy' option, a meaningful value, e.g.,
            edge energy, is necessary to obtain good results
        optimizer : str
            'scipy': lsq based
            'numpy': numpy.polyfit based.
        flt_spec: boolean
            if median filtering the spectrum before fitting,
        on: str
            fitting on either 'raw' or 'norm'(alized) spectrum data
        *args: positional arguments to fitting functions
            'scipy':
                model: model function name
                fvars: model function's initial argument values
            'numpy':
                order: order of the polynomial fitting function
        **kwargs : keyword arguments to fitting functions
            'scipy': least_squares arguments
                bnds=None,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
                jac='3-point',
                method='trf'
            'numpy': NA

        Returns
        -------
        rlt: ndarray of object
            rlt[0]: coef of fitting curve.
            rlt[1:4]: residuals, rank, singular_values, rcond in numpy.polyfit case
            rlt[1:3]: cost, status, success in scipy.optimize.lsq case

        """
        idx_s = xm.index_of(self.eng, es)
        idx_e = xm.index_of(self.eng, ee)
        if flt_spec or (flt_spec == 'True'):
            flt_kernal = np.ones(len(self.spec.shape), dtype=np.int32)
            flt_kernal[0] = 3

            if on == "raw":
                if optimizer == "scipy":
                    fit_rlt = xm.fit_curv_scipy(self.eng[idx_s:idx_e] - eoff,
                                                median_filter(self.spec[idx_s:idx_e, :],
                                                              size=flt_kernal),
                                                *args, **kwargs)
                elif optimizer == "numpy":
                    fit_rlt = xm.fit_curv_polynd(self.eng[idx_s:idx_e] - eoff,
                                                 median_filter(self.spec[idx_s:idx_e, :],
                                                               size=flt_kernal),
                                                 *args)
            elif on == "norm":
                if optimizer == "scipy":
                    fit_rlt = xm.fit_curv_scipy(self.eng[idx_s:idx_e] - eoff,
                                                median_filter(self.norm_spec[idx_s:idx_e, :],
                                                              size=flt_kernal),
                                                *args, **kwargs)
                elif optimizer == "numpy":
                    fit_rlt = xm.fit_curv_polynd(self.eng[idx_s:idx_e] - eoff,
                                                 median_filter(self.norm_spec[idx_s:idx_e, :],
                                                               size=flt_kernal),
                                                 *args)
        else:
            if on == "raw":
                if optimizer == "scipy":
                    fit_rlt = xm.fit_curv_scipy(self.eng[idx_s:idx_e] - eoff,
                                                self.spec[idx_s:idx_e, :],
                                                *args, **kwargs)
                elif optimizer == "numpy":
                    fit_rlt = xm.fit_curv_polynd(self.eng[idx_s:idx_e] - eoff,
                                                 self.spec[idx_s:idx_e, :],
                                                 *args)
            elif on == "norm":
                if optimizer == "scipy":
                    fit_rlt = xm.fit_curv_scipy(self.eng[idx_s:idx_e] - eoff,
                                                self.norm_spec[idx_s:idx_e, :],
                                                *args, **kwargs)
                elif optimizer == "numpy":
                    fit_rlt = xm.fit_curv_polynd(self.eng[idx_s:idx_e] - eoff,
                                                 self.norm_spec[idx_s:idx_e, :],
                                                 *args)
        return fit_rlt

    def fit_edge(self, es, ee, eoff=None, optimizer='numpy',
                 flt_spec=False, on='norm', ftype='edge', order=3,
                 model='lorentzian', fvars=None,
                 bnds=None, ftol=1e-7, xtol=1e-7,
                 gtol=1e-7, jac='3-point',
                 method='trf'):
        if eoff is None:
            eoff = self.preset_edge_eng
        if optimizer == "numpy":
            self.model[ftype]['model'] = 'polynd'
            args = [order]
            kwargs = {}
        elif optimizer == "scipy":
            self.model[ftype]['model'] = model
            args = [model, fvars]
            kwargs = {'bnds': bnds, 'ftol': ftol, 'xtol': xtol, 'gtol': gtol,
                      'jac': jac, 'method': method}
        self.model[ftype]['eoff'] = eoff
        self.model[ftype]['fit_rlt'] = self.fit_spec(es, ee, eoff, optimizer, flt_spec, on,
                                                     *args, **kwargs)

    @msgit(wd=100, fill='$')
    def full_spec_preprocess(self, eng0, order=1, save_pre_post=False):
        e0_idx = xm.index_of(self.eng, eng0)
        pre = self.cal_pre_edge_fit(self.eng)
        post = self.cal_post_edge_fit(self.eng)

        self.edge_jump_mask = np.squeeze((post[e0_idx] - pre[e0_idx])
                                         > self.edge_jump_thres * self.pre_edge_sd_map).astype(np.int8)
        self.fitted_edge_mask = np.any((post - pre) > self.fitted_edge_thres * self.pre_edge_sd_map, axis=0).astype(
            np.int8)

        self.norm_spec = (self.spec - pre) / (self.cal_post_edge_fit(eng0) - self.cal_pre_edge_fit(eng0))
        self.norm_spec[np.isnan(self.norm_spec)] = 0
        self.norm_spec[np.isinf(self.norm_spec)] = 0

    def interp_ref_spec(self):
        self.lcf_ref_spec = np.ndarray([self.eng.shape[0], self.lcf_ref.index.size])
        for ii in range(self.lcf_ref.index.size):
            c = splrep(self.lcf_ref.iloc[ii]['eng'], self.lcf_ref.iloc[ii]['mu'])
            self.lcf_ref_spec[:, ii] = splev(self.eng, c)

    @msgit(wd=100, fill='+')
    def lcf(self):
        self.lcf_model['constr'] = self.lcf_constr_use
        self.lcf_model['ref'] = self.lcf_ref
        self.lcf_model['rlt'] = xm.lcf(self.lcf_ref_spec, self.norm_spec, constr=self.lcf_constr_use)
        self.lcf_fit = self.lcf_model['rlt'][0].reshape(self.lcf_model['rlt'][0].shape[0], *self.spec.shape[1:])
        self.lcf_fit_err = self.lcf_model['rlt'][1].reshape(*self.spec.shape[1:])

    @msgit(wd=100, fill='+')
    def normalize_xanes(self, eng0, order=1, save_pre_post=False):
        """
        For 2D XANES, self.spec dimensions are [eng, 2D_space]
        For 3D XANES, self.spec dimensions are [eng, 3D_space]

        normalize_xanes includes two steps: 1) pre-edge background subtraction (order = 0)),
                                            2) post-edge fitted or average mu normalization (order = 1)
        return: ndarray, same dimension as spectrum.shape
        """
        eng = deepcopy(self.eng)
        for ii in range(1, self.pre_edge_fit_rlt[0].ndim):
            eng = eng[:, np.newaxis]

        if order == 0:
            self.norm_spec = (self.spec - self.pre_edge_mean_map) / \
                             (self.post_edge_mean_map - self.pre_edge_mean_map)
            self.norm_spec[np.isnan(self.norm_spec)] = 0
            self.norm_spec[np.isinf(self.norm_spec)] = 0
            self.spec_normalized = True
        elif order == 1:
            if save_pre_post:
                e0_idx = xm.index_of(self.eng, eng0)
                self.pre_edge_fit = self.cal_pre_edge_fit(self.eng)
                self.post_edge_fit = self.cal_post_edge_fit([eng])
                self.norm_spec = (self.spec - self.pre_edge_fit) / (
                        self.post_edge_fit[e0_idx] - self.pre_edge_fit[e0_idx])
                self.norm_spec[np.isnan(self.norm_spec)] = 0
                self.norm_spec[np.isinf(self.norm_spec)] = 0
            else:
                self.norm_spec = (self.spec -
                                  self.cal_pre_edge_fit([eng0])) / (
                                         self.cal_post_edge_fit([eng0]) -
                                         self.cal_pre_edge_fit([eng0]))
                self.norm_spec[np.isnan(self.norm_spec)] = 0
                self.norm_spec[np.isinf(self.norm_spec)] = 0
            self.spec_normalized = True
        else:
            print('Order can only be "1" or "0".')
            self.spec_normalized = False

    def remove_sli(self, id_list):
        self.spec = np.delete(self.spec, id_list, axis=0)
        self.eng = np.delete(self.eng, id_list, axis=0)
        self.removed_img_ids = id_list
        self.pre_es_idx = xm.index_of(self.eng, self.pre_es)
        self.pre_ee_idx = xm.index_of(self.eng, self.pre_ee)
        self.post_es_idx = xm.index_of(self.eng, self.post_es)
        self.post_ee_idx = xm.index_of(self.eng, self.post_ee)

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
            g1.create_dataset('pre_edge_fit_coef', data=self.pre_edge_fit_rlt[0])
            g1.create_dataset('post_edge_fit_coef', data=self.post_edge_fit_rlt[0])
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
            g21.create_dataset('edge_jump_threshold', data=self.edge_jump_thres)
            g21.create_dataset('edge_jump_mask', data=self.edge_jump_mask.astype(np.int8))
            g22 = g2.create_group('fitted_edge_fitler')
            g22.create_dataset('fitted_edge_threshold', data=self.fitted_edge_thres)
            g22.create_dataset('fitted_edge_mask', data=self.fitted_edge_mask.astype(np.int8))

            if 'XANES_results' not in f:
                g3 = f.create_group('XANES_results')
            else:
                del f['XANES_results']
                g3 = f.create_group('XANES_results')

            g3.create_dataset('eng', data=self.eng.astype(np.float32), dtype=np.float32)
            g3.create_dataset('normalized_spectrum', data=self.norm_spec.astype(np.float32), dtype=np.float32)
            g3.create_dataset('edge_pos', data=self.auto_e0.astype(np.float32), dtype=np.float32)
            g3.create_dataset('edge_0.5_pos', data=self.edge_pos.astype(np.float32), dtype=np.float32)
            g3.create_dataset('whiteline_pos_fit', data=self.wl_pos_fit.astype(np.float32), dtype=np.float32)
            g3.create_dataset('whiteline_pos_direct', data=self.wl_pos_dir.astype(np.float32), dtype=np.float32)

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
                              data=self.pre_edge_fit_rlt[0])
            g1.create_dataset('post_edge_fit_coef',
                              data=self.post_edge_fit_rlt[0])
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
                              data=self.edge_jump_thres)
            g1.create_dataset('fitted_edge_threshold',
                              data=self.fitted_edge_thres)
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

    def set_external_mask(self, mask):
        self.ext_mask = mask

    def update_with_auto_e0(self):
        self.find_edge()

        self.pre_ee = self.auto_e0 + (self.pre_ee - self.preset_edge_eng)
        self.pre_ee_idx = xm.index_of(self.eng, self.pre_ee)

        self.post_es = self.auto_e0 + (self.post_es - self.preset_edge_eng)
        self.post_es_idx = xm.index_of(self.eng, self.post_es)
