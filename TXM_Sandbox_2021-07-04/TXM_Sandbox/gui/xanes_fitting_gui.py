#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:56:17 2020

@author: xiao
"""
import os, h5py, json, numpy as np, pandas as pd

from ipywidgets import widgets
from copy import deepcopy
import napari

from .gui_components import (NumpyArrayEncoder, get_handles, enable_disable_boxes,
                            gen_external_py_script, fiji_viewer_off, fiji_viewer_state,
                            fiji_viewer_on, scale_eng_list, SelectFilesButton, update_json_content)
import TXM_Sandbox.TXM_Sandbox.utils.xanes_math as xm
import TXM_Sandbox.TXM_Sandbox.utils.xanes_analysis as xa

inf = np.inf

napari.gui_qt()

PEAK_LINE_SHAPES = ['lorentzian', 'gaussian', 'voigt',
                    'pvoigt', 'moffat', 'pearson7',
                    'breit_wigner', 'damped_oscillator',
                    'dho', 'lognormal',
                    'students_t', 'expgaussian', 'donaich',
                    'skewed_gaussian','skewed_voigt',
                    'step', 'rectangle', 'parabolic',
                    'sine', 'expsine', 'split_lorentzian']

STEP_LINE_SHAPES = ['logistic', 'exponential',
                    'powerlaw', 'linear']

PEAK_FIT_PARAM_DICT = {"parabolic":{0: ["a", 1, "a: ampflitude in parabolic function"],
                                    1: ["b", 0, "b: center of parabolic function"],
                                    2: ["c", 1, "c: standard deviation of parabolic function"]},
                      "gaussian":{0: ["amp", 1, "amp: ampflitude in gaussian function"],
                                  1: ["cen", 0, "cen: center of gaussian function"],
                                  2: ["sig", 1, "sig: standard deviation of gaussian function"]},
                      "lorentzian":{0: ["amp", 1, "amp: ampflitude in lorentzian function"],
                                    1: ["cen", 0, "cen: center of lorentzian function"],
                                    2: ["sig", 1, "sig: standard deviation of lorentzian function"]},
                      "damped_oscillator":{0: ["amp", 1, "amp: ampflitude in damped_oscillator function"],
                                           1: ["cen", 0, "cen: center of damped_oscillator function"],
                                           2: ["sig", 1, "sig: standard deviation of damped_oscillator function"]},
                      "lognormal":{0: ["amp", 1, "amp: ampflitude in lognormal function"],
                                   1: ["cen", 0, "cen: center of lognormal function"],
                                   2: ["sig", 1, "sig: standard deviation of lognormal function"]},
                      "students_t":{0: ["amp", 1, "amp: ampflitude in students_t function"],
                                    1: ["cen", 0, "cen: center of students_t function"],
                                    2: ["sig", 1, "sig: standard deviation of students_t function"]},
                      "sine":{0: ["amp", 1, "amp: ampflitude in sine function"],
                              1: ["frq", 1, "frq: freqency in sine function"],
                              2: ["shft", 0, "shft: shift in sine function"]},
                      "voigt":{0: ["amp", 1, "amp: ampflitude in voigt function"],
                               1: ["cen", 0, "cen: center of voigt function"],
                               2: ["sig", 1, "sig: standard voigt of gaussian function"],
                               3: ["gamma", 0, "gamma: "]},
                      "split_lorentzian":{0: ["amp", 1, "amp: ampflitude in split_lorentzian function"],
                                          1: ["cen", 0, "cen: center of split_lorentzian function"],
                                          2: ["sig", 1, "sig: standard deviation of split_lorentzian function"],
                                          3: ["sigr", 1, "sigr: standard deviation of the right-hand side half in split_lorentzian function"]},
                      "pvoigt":{0: ["amp", 1, "amp: ampflitude in pvoigt function"],
                                1: ["cen", 0, "cen: center of pvoigt function"],
                                2: ["sig", 1, "sig: standard pvoigt of gaussian function"],
                                3: ["frac", 0, "frac: "]},
                      "moffat":{0: ["amp", 1, "amp: ampflitude in moffat function"],
                                1: ["cen", 0, "cen: center of moffat function"],
                                2: ["sig", 1, "sig: standard moffat of gaussian function"],
                                3: ["beta", 0, "beta: "]},
                      "pearson7": {0: ["amp", 1, "amp: ampflitude in pearson7 function"],
                                   1: ["cen", 0, "cen: center of pearson7 function"],
                                   2: ["sig", 1, "sig: standard pearson7 of gaussian function"],
                                   3: ["expo", 0, "expo: "]},
                      "breit_wigner":{0: ["amp", 1, "amp: ampflitude in breit_wigner function"],
                                      1: ["cen", 0, "cen: center of breit_wigner function"],
                                      2: ["sig", 1, "sig: standard breit_wigner of gaussian function"],
                                      3: ["q", 0, "q: "]},
                      "dho":{0: ["amp", 1, "amp: ampflitude in dho function"],
                             1: ["cen", 0, "cen: center of dho function"],
                             2: ["sig", 1, "sig: standard dho of gaussian function"],
                             3: ["gama", 1, "gama: "]},
                      "expgaussian":{0: ["amp", 1, "amp: ampflitude in expgaussian function"],
                                     1: ["cen", 0, "cen: center of expgaussian function"],
                                     2: ["sig", 1, "sig: standard expgaussian of gaussian function"],
                                     3: ["gama", 1, "gama: "]},
                      "donaich":{0: ["amp", 1, "amp: ampflitude in donaich function"],
                                 1: ["cen", 0, "cen: center of donaich function"],
                                 2: ["sig", 1, "sig: standard donaich of gaussian function"],
                                 3: ["gama", 0, "gama: "]},
                      "skewed_gaussian":{0: ["amp", 1, "amp: ampflitude in skewed_gaussian function"],
                                         1: ["cen", 0, "cen: center of skewed_gaussian function"],
                                         2: ["sig", 1, "sig: standard skewed_gaussian of gaussian function"],
                                         3: ["gama", 0, "gama: "]},
                      "expsine":{0: ["amp", 1, "amp: ampflitude in expsine function"],
                                 1: ["frq", 1, "frq:  width of expsine function"],
                                 2: ["shft", 0, "shft: center of gaussian function"],
                                 3: ["dec", 0, "dec: exponential decay factor"]},
                      "step":{0: ["amp", 1, "amp: ampflitude in step function"],
                              1: ["cen", 0, "cen: center of step function"],
                              2: ["sig", 1, "sig: standard step of gaussian function"],
                              5: ["form", "linear", "form: "]},
                      "skewed_voigt":{0: ["amp", 1, "amp: ampflitude in skewed_voigt function"],
                                      1: ["cen", 0, "cen: center of skewed_voigt function"],
                                      2: ["sig", 1, "sig: standard skewed_voigt of gaussian function"],
                                      3: ["gamma", 0, "gamma: "],
                                      4: ["skew", 0, "skew: "]},
                      "rectangle":{0: ["amp", 1, "amp: ampflitude in rectangle function"],
                                   1: ["cen1", 0, "cen1: center of rectangle function"],
                                   2: ["sig1", 1, "sig1: standard deviation of rectangle function"],
                                   3: ["cen2", 0, "cen2: center of rectangle function"],
                                   4: ["sig2", 1, "sig2: standard deviation of rectangle function"],
                                   5: ["form", "linear", "form: "]},}

PEAK_FIT_PARAM_BND_DICT = {"parabolic":{0: ["a", [-1e3, 1e3], "a: ampflitude in parabolic function"],
                                   1: ["b", [-1e3, 1e3], "b: center of parabolic function"],
                                   2: ["c", [-1e5, 1e5], "c: standard deviation of parabolic function"]},
                          "gaussian":{0: ["amp", [-10, 10], "amp: ampflitude in gaussian function"],
                                      1: ["cen", [-2, 2], "cen: center of gaussian function"],
                                      2: ["sig", [0, 1e3], "sig: standard deviation of gaussian function"]},
                          "lorentzian":{0: ["amp", [-10, 10], "amp: ampflitude in lorentzian function"],
                                        1: ["cen", [-2, 2], "cen: center of lorentzian function"],
                                        2: ["sig", [0, 1e3], "sig: standard deviation of lorentzian function"]},
                          "damped_oscillator":{0: ["amp", [-10, 10], "amp: ampflitude in damped_oscillator function"],
                                               1: ["cen", [-2, 2], "cen: center of damped_oscillator function"],
                                               2: ["sig", [0, 1e3], "sig: standard deviation of damped_oscillator function"]},
                          "lognormal":{0: ["amp", [-10, 10], "amp: ampflitude in lognormal function"],
                                       1: ["cen", [-2, 2], "cen: center of lognormal function"],
                                       2: ["sig", [0, 1e3], "sig: standard deviation of lognormal function"]},
                          "students_t":{0: ["amp", [-10, 10], "amp: ampflitude in students_t function"],
                                        1: ["cen", [-2, 2], "cen: center of students_t function"],
                                        2: ["sig", [0, 1e3], "sig: standard deviation of students_t function"]},
                          "sine":{0: ["amp", [-10, 10], "amp: ampflitude in sine function"],
                                  1: ["frq", [0, 1], "frq: freqency in sine function"],
                                  2: ["shft", [0, 1], "shft: shift in sine function"]},
                          "voigt":{0: ["amp", [-10, 10], "amp: ampflitude in voigt function"],
                                   1: ["cen", [-2, 2], "cen: center of voigt function"],
                                   2: ["sig", [0, 1e3], "sig: standard voigt of gaussian function"],
                                   3: ["gamma", [0, 1e3], "gamma: "]},
                          "split_lorentzian":{0: ["amp", [-10, 10], "amp: ampflitude in split_lorentzian function"],
                                              1: ["cen", [-2, 2], "cen: center of split_lorentzian function"],
                                              2: ["sig", [0, 1e3], "sig: standard deviation of split_lorentzian function"],
                                              3: ["sigr", [0, 1e3], "sigr: standard deviation of the right-hand side half in split_lorentzian function"]},
                          "pvoigt":{0: ["amp", [-10, 10], "amp: ampflitude in pvoigt function"],
                                    1: ["cen", [-2, 2], "cen: center of pvoigt function"],
                                    2: ["sig", [0, 1e3], "sig: standard pvoigt of gaussian function"],
                                    3: ["frac", [0, 1], "frac: "]},
                          "moffat":{0: ["amp", [-10, 10], "amp: ampflitude in moffat function"],
                                    1: ["cen", [-2, 2], "cen: center of moffat function"],
                                    2: ["sig", [0, 1e3], "sig: standard moffat of gaussian function"],
                                    3: ["beta", [-1e3, 1e3], "beta: "]},
                          "pearson7": {0: ["amp", [-10, 10], "amp: ampflitude in pearson7 function"],
                                       1: ["cen", [-2, 2], "cen: center of pearson7 function"],
                                       2: ["sig", [0, 1e3], "sig: standard pearson7 of gaussian function"],
                                       3: ["expo", [-1e2, 1e2], "expo: "]},
                          "breit_wigner":{0: ["amp", [-10, 10], "amp: ampflitude in breit_wigner function"],
                                          1: ["cen", [-2,2], "cen: center of breit_wigner function"],
                                          2: ["sig", [0, 1e3], "sig: standard breit_wigner of gaussian function"],
                                          3: ["q", [-10, 10], "q: "]},
                          "dho":{0: ["amp", [-10, 10], "amp: ampflitude in dho function"],
                                 1: ["cen", [-2, 2], "cen: center of dho function"],
                                 2: ["sig", [0, 1e3], "sig: standard dho of gaussian function"],
                                 3: ["gama", [-10, 10], "gama: "]},
                          "expgaussian":{0: ["amp", [-10, 10], "amp: ampflitude in expgaussian function"],
                                         1: ["cen", [-2, 2], "cen: center of expgaussian function"],
                                         2: ["sig", [0, 1e3], "sig: standard expgaussian of gaussian function"],
                                         3: ["gama", [-10, 10], "gama: "]},
                          "donaich":{0: ["amp", [-10, 10], "amp: ampflitude in donaich function"],
                                     1: ["cen", [-2, 2], "cen: center of donaich function"],
                                     2: ["sig", [0, 1e3], "sig: standard donaich of gaussian function"],
                                     3: ["gama", [-10, 10], "gama: "]},
                          "skewed_gaussian":{0: ["amp", [-10, 10], "amp: ampflitude in skewed_gaussian function"],
                                             1: ["cen", [-2, 2], "cen: center of skewed_gaussian function"],
                                             2: ["sig", [0, 1e3], "sig: standard skewed_gaussian of gaussian function"],
                                             3: ["gama", 0, "gama: "]},
                          "expsine":{0: ["amp", [-10, 10], "amp: ampflitude in expsine function"],
                                     1: ["frq", [0, 1], "frq: center of expsine function"],
                                     2: ["shft", [0, 1], "shft: standard expsine of gaussian function"],
                                     3: ["dec", [-10, 10], "dec: "]},
                          "step":{0: ["amp", [-10, 10], "amp: ampflitude in step function"],
                                  1: ["cen", [-2, 2], "cen: center of step function"],
                                  2: ["sig", [0, 1e3], "sig: standard step of gaussian function"]},
                          "skewed_voigt":{0: ["amp", [-10, 10], "amp: ampflitude in skewed_voigt function"],
                                          1: ["cen", [-2, 2], "cen: center of skewed_voigt function"],
                                          2: ["sig", [0, 1e3], "sig: standard skewed_voigt of gaussian function"],
                                          3: ["gamma", [0, 1e-3], "gamma: "],
                                          4: ["skew", [-10, 10], "skew: "]},
                          "rectangle":{0: ["amp", [-10, 10], "amp: ampflitude in rectangle function"],
                                       1: ["cen1", [-2, 2], "cen1: center of rectangle function"],
                                       2: ["sig1", [0, 1e3], "sig1: standard deviation of rectangle function"],
                                       3: ["cen2", [-2, 2], "cen2: center of rectangle function"],
                                       4: ["sig2", [0, 1e3], "sig2: standard deviation of rectangle function"]}}

EDGE_LINE_SHAPES = ['lorentzian', 'split_lorentzian',
                    'voigt', 'pvoigt', 'skewed_voigt',
                    'gaussian', 'skewed_gaussian', 'expgaussian',
                    'sine', 'expsine']

EDGE_FIT_PARAM_DICT = {"gaussian":{0: ["amp", 1, "amp: ampflitude in gaussian function"],
                                  1: ["cen", 0, "cen: center of gaussian function"],
                                  2: ["sig", 1, "sig: standard deviation of gaussian function"]},
                      "lorentzian":{0: ["amp", 1, "amp: ampflitude in lorentzian function"],
                                    1: ["cen", 0, "cen: center of lorentzian function"],
                                    2: ["sig", 1, "sig: standard deviation of lorentzian function"]},
                      "sine":{0: ["amp", 1, "amp: ampflitude in sine function"],
                              1: ["frq", 1, "frq: freqency in sine function"],
                              2: ["shft", 0, "shft: shift in sine function"]},
                      "voigt":{0: ["amp", 1, "amp: ampflitude in voigt function"],
                               1: ["cen", 0, "cen: center of voigt function"],
                               2: ["sig", 1, "sig: standard voigt of gaussian function"],
                               3: ["gamma", 0, "gamma: "]},
                      "split_lorentzian":{0: ["amp", 1, "amp: ampflitude in split_lorentzian function"],
                                          1: ["cen", 0, "cen: center of split_lorentzian function"],
                                          2: ["sig", 1, "sig: standard deviation of split_lorentzian function"],
                                          3: ["sigr", 1, "sigr: standard deviation of the right-hand side half in split_lorentzian function"]},
                      "pvoigt":{0: ["amp", 1, "amp: ampflitude in pvoigt function"],
                                1: ["cen", 0, "cen: center of pvoigt function"],
                                2: ["sig", 1, "sig: standard pvoigt of gaussian function"],
                                3: ["frac", 0, "frac: "]},
                      "expgaussian":{0: ["amp", 1, "amp: ampflitude in expgaussian function"],
                                     1: ["cen", 0, "cen: center of expgaussian function"],
                                     2: ["sig", 1, "sig: standard expgaussian of gaussian function"],
                                     3: ["gama", 1, "gama: "]},
                      "skewed_gaussian":{0: ["amp", 1, "amp: ampflitude in skewed_gaussian function"],
                                         1: ["cen", 0, "cen: center of skewed_gaussian function"],
                                         2: ["sig", 1, "sig: standard skewed_gaussian of gaussian function"],
                                         3: ["gama", 0, "gama: "]},
                      "expsine":{0: ["amp", 1, "amp: ampflitude in expsine function"],
                                 1: ["frq", 1, "frq:  width of expsine function"],
                                 2: ["shft", 0, "shft: center of gaussian function"],
                                 3: ["dec", 0, "dec: exponential decay factor"]},
                      "skewed_voigt":{0: ["amp", 1, "amp: ampflitude in skewed_voigt function"],
                                      1: ["cen", 0, "cen: center of skewed_voigt function"],
                                      2: ["sig", 1, "sig: standard skewed_voigt of gaussian function"],
                                      3: ["gamma", 0, "gamma: "],
                                      4: ["skew", 0, "skew: "]}}

EDGE_FIT_PARAM_BND_DICT = {"gaussian":{0: ["amp", [-10, 10], "amp: ampflitude in gaussian function"],
                                      1: ["cen", [0, 1], "cen: center of gaussian function"],
                                      2: ["sig", [0, 1e3], "sig: standard deviation of gaussian function"]},
                          "lorentzian":{0: ["amp", [-10, 10], "amp: ampflitude in lorentzian function"],
                                        1: ["cen", [0, 1], "cen: center of lorentzian function"],
                                        2: ["sig", [0, 1e3], "sig: standard deviation of lorentzian function"]},
                          "sine":{0: ["amp", [-10, 10], "amp: ampflitude in sine function"],
                                  1: ["frq", [0, 1], "frq: freqency in sine function"],
                                  2: ["shft", [0, 1], "shft: shift in sine function"]},
                          "voigt":{0: ["amp", [-10, 10], "amp: ampflitude in voigt function"],
                                   1: ["cen", [0, 1], "cen: center of voigt function"],
                                   2: ["sig", [0, 1e3], "sig: standard voigt of gaussian function"],
                                   3: ["gamma", [0, 1e3], "gamma: "]},
                          "split_lorentzian":{0: ["amp", [-10, 10], "amp: ampflitude in split_lorentzian function"],
                                              1: ["cen", [0, 1], "cen: center of split_lorentzian function"],
                                              2: ["sig", [0, 1e3], "sig: standard deviation of split_lorentzian function"],
                                              3: ["sigr", [0, 1e3], "sigr: standard deviation of the right-hand side half in split_lorentzian function"]},
                          "pvoigt":{0: ["amp", [-10, 10], "amp: ampflitude in pvoigt function"],
                                    1: ["cen", [0, 1], "cen: center of pvoigt function"],
                                    2: ["sig", [0, 1e3], "sig: standard pvoigt of gaussian function"],
                                    3: ["frac", [0, 1], "frac: "]},
                          "expgaussian":{0: ["amp", [-10, 10], "amp: ampflitude in expgaussian function"],
                                         1: ["cen", [0, 1], "cen: center of expgaussian function"],
                                         2: ["sig", [0, 1e3], "sig: standard expgaussian of gaussian function"],
                                         3: ["gama", [-10, 10], "gama: "]},
                          "skewed_gaussian":{0: ["amp", [-10, 10], "amp: ampflitude in skewed_gaussian function"],
                                             1: ["cen", [0, 1], "cen: center of skewed_gaussian function"],
                                             2: ["sig", [0, 1e3], "sig: standard skewed_gaussian of gaussian function"],
                                             3: ["gama", 0, "gama: "]},
                          "expsine":{0: ["amp", [-10, 10], "amp: ampflitude in expsine function"],
                                     1: ["frq", [0, 1], "frq: center of expsine function"],
                                     2: ["shft", [0, 1], "shft: standard expsine of gaussian function"],
                                     3: ["dec", [-10, 10], "dec: "]},
                          "skewed_voigt":{0: ["amp", [-10, 10], "amp: ampflitude in skewed_voigt function"],
                                          1: ["cen", [0, 1], "cen: center of skewed_voigt function"],
                                          2: ["sig", [0, 1e3], "sig: standard skewed_voigt of gaussian function"],
                                          3: ["gamma", [0, 1e-3], "gamma: "],
                                          4: ["skew", [-10, 10], "skew: "]}}

NUMPY_FIT_ORDER = [2, 3, 4]

FULL_SAVE_ITEM_OPTIONS = ['norm_spec',
                          'wl_pos_fit',
                          'wl_fit_err',
                          'wl_pos_dir',
                          'wl_peak_height_dir',
                          'centroid_of_eng',
                          'centroid_of_eng_relative_to_wl',
                          'weighted_attenuation',
                          'weighted_eng',
                          'edge50_pos_fit',
                          'edge50_pos_dir',
                          # 'edge0.5_pos_direct',
                          'edge_pos_fit',
                          'edge_fit_err',
                          'edge_pos_dir',
                          'edge_jump_filter',
                          'edge_offset_filter',
                          'pre_edge_sd',
                          'pre_edge_mean',
                          'post_edge_sd',
                          'post_edge_mean',
                          'pre_edge_fit_coef',
                          'post_edge_fit_coef',
                          'wl_fit_coef',
                          'edge_fit_coef',
                          'lcf_fit',
                          'lcf_fit_err']

FULL_SAVE_DEFAULT = ['',
                     'norm_spec',
                     'wl_pos_fit',
                     'wl_fit_err',
                     'wl_pos_dir',
                     'wl_peak_height_dir',
                     'centroid_of_eng',
                     'centroid_of_eng_relative_to_wl',
                     'weighted_attenuation',
                     'weighted_eng',
                     'edge50_pos_fit',
                     'edge50_pos_dir',
                     # 'edge0.5_pos_fit',
                     # 'edge0.5_pos_direct',
                     'edge_pos_fit',
                     'edge_fit_err',
                     'edge_pos_dir',
                     'edge_jump_filter',
                     'edge_offset_filter',
                     'pre_edge_sd',
                     'pre_edge_mean',
                     'post_edge_sd',
                     'post_edge_mean']

WL_SAVE_ITEM_OPTIONS = ['wl_pos_fit',
                        'wl_fit_err',
                        'wl_pos_dir',
                        'edge_pos_fit',
                        'edge_fit_err',
                        'edge_pos_dir',
                        'wl_fit_coef',
                        'edge_fit_coef']

WL_SAVE_DEFAULT= ['',
                  'wl_pos_fit',
                  'wl_fit_err',
                  'wl_pos_dir',
                  'edge_pos_fit',
                  'edge_fit_err',
                  'edge_pos_dir']

class xanes_fitting_gui():
    def __init__(self, parent_h, form_sz=[650, 740]):
        self.parent_h = parent_h
        self.global_h = parent_h.global_h
        self.hs = {}
        self.form_sz = form_sz
        self.fit_img_bin_fac = 1
        self.fit_wl_fit_use_param_bnd = False
        self.fit_wl_optimizer = 'numpy'
        self.fit_wl_fit_func = 2
        self.fit_edge_fit_use_param_bnd = False
        self.fit_edge_optimizer = 'numpy'
        self.fit_edge_fit_func = 3
        self.analysis_saving_items = set(FULL_SAVE_DEFAULT)
        self.parent_h.xanes_fit_type == 'full'
        self.xanes_fit_eng_config = False

        self.fit_flt_prev = False
        self.fit_flt_prev_sli = 0
        self.fit_flt_prev_configed = False
        self.fit_flt_prev_maskit = False
        self.fit_lcf = False
        self.fit_lcf_ref_set = False
        self.fit_lcf_constr = True
        self.fit_lcf_ref_num = 2
        self.fit_lcf_ref_spec = pd.DataFrame(columns=['fpath', 'fname', 'eng', 'mu'])

        self.pre_es_idx = None
        self.pre_ee_idx = None
        self.post_es_idx = None
        self.post_ee_idx = None
        self.fit_flt_prev_xana = None
        self.e0_idx = None
        self.pre = None
        self.post = None
        self.edge_jump_mask = None
        self.fitted_edge_mask = None

        self.fit_wl_pos = True
        self.fit_edge_pos = False
        self.fit_edge50_pos = True
        self.find_edge_pos_dir = False
        self.find_edge50_pos_dir = True
        self.fit_use_flt_spec = False

        if self.parent_h.gui_name == 'xanes3D':
            self.fn = self.parent_h.xanes_save_trial_reg_filename
            self.xanes_mask_external_command_name = os.path.join(os.path.abspath(os.path.curdir),
                                                                'xanes3D_mask_external_command.py')
        elif self.parent_h.gui_name == 'xanes2D':
            self.fn = self.parent_h.xanes_save_trial_reg_filename
            self.xanes_mask_external_command_name = os.path.join(os.path.abspath(os.path.curdir),
                                                                'xanes2D_mask_external_command.py')


    def build_gui(self):
        ## ## ## bin sub-tabs in each tab - analysis&display TAB in 3D_xanes TAB -- start
        ## ## ## ## define functional widgets each tab in each sub-tab - analysis box in analysis&display TAB -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.98*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3]_fitting_form'] = widgets.VBox()
        self.hs['L[0][x][3]_fitting_form'].layout = layout

        ## ## ## ## fit energy config box -- start
        layout = {'border': '3px solid #8855AA', 'width': f'{self.form_sz[1] - 98}px',
                  'height': f'{0.285 * (self.form_sz[0] - 128)}px'}
        self.hs['L[0][x][3][0]_fit_eng_config_box'] = widgets.VBox()
        self.hs['L[0][x][3][0]_fit_eng_config_box'].layout = layout

        ## ## ## ## ## label analysis box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-104}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][0][0]_fit_title_box'] = widgets.HBox()
        self.hs['L[0][x][3][0][0]_fit_title_box'].layout = layout
        # self.hs['L[0][x][3][0][0][0]_fit_title_text'] = widgets.Text(value='XANES Fitting', disabled=True)
        self.hs['L[0][x][3][0][0][0]_fit_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'XANES Fitting' + '</span>')
        layout = {'background-color':'white', 'color':'cyan', 'left':'41%'}
        self.hs['L[0][x][3][0][0][0]_fit_title_text'].layout = layout
        self.hs['L[0][x][3][0][0]_fit_title_box'].children = get_handles(self.hs, 'L[0][x][3][0][0]_fit_title_box', -1)
        ## ## ## ## ## label analysis box -- end

        ## ## ## ## ## define type of analysis and energy range -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{1*self.form_sz[1]-104}px', 'height':f'{0.14*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][0][1]_fit_energy_range_box'] = widgets.VBox()
        self.hs['L[0][x][3][0][1]_fit_energy_range_box'].layout = layout
        layout = {'border':'none', 'width':f'{1*self.form_sz[1]-110}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][0][1][0]_fit_energy_range_box1'] = widgets.HBox()
        self.hs['L[0][x][3][0][1][0]_fit_energy_range_box1'].layout = layout
        layout = {'width':'19%', 'height':'90%'}
        self.hs['L[0][x][3][0][1][0][0]_fit_eng_range_option_dropdown'] = widgets.Dropdown(description='analysis type',
                                                                                           description_tooltip='wl: find whiteline positions without doing background removal and normalization; edge0.5: find energy point where the normalized spectrum value equal to 0.5; full: doing regular xanes preprocessing',
                                                                                           options=['wl', 'full'],
                                                                                           value ='wl',
                                                                                           disabled=True)
        self.hs['L[0][x][3][0][1][0][0]_fit_eng_range_option_dropdown'].layout = layout
        # layout = {'width':'19%', 'height':'100%', 'top':'0%', 'visibility':'hidden'}
        layout = {'width':'19%', 'height':'90%'}
        self.hs['L[0][x][3][0][1][0][1]_fit_eng_range_edge_eng_text'] = widgets.BoundedFloatText(description='edge eng',
                                                                                                  description_tooltip='edge energy (keV)',
                                                                                                  value =0,
                                                                                                  min = 0,
                                                                                                  max = 50000,
                                                                                                  step=0.5,
                                                                                                  disabled=True)
        self.hs['L[0][x][3][0][1][0][1]_fit_eng_range_edge_eng_text'].layout = layout
        layout = {'width':'19%', 'height':'90%'}
        self.hs['L[0][x][3][0][1][0][2]_fit_eng_range_pre_edge_e_text'] = widgets.BoundedFloatText(description='pre edge e',
                                                                                                   description_tooltip='relative ending energy point (keV) of pre-edge from edge energy for background removal',
                                                                                                   value =-50,
                                                                                                   min = -500,
                                                                                                   max = 0,
                                                                                                   step=0.5,
                                                                                                   disabled=True)
        self.hs['L[0][x][3][0][1][0][2]_fit_eng_range_pre_edge_e_text'].layout = layout
        layout = {'width':'19%', 'height':'90%'}
        self.hs['L[0][x][3][0][1][0][3]_fit_eng_range_post_edge_s_text'] = widgets.BoundedFloatText(description='post edge s',
                                                                                                   description_tooltip='relative starting energy point (keV) of post-edge from edge energy for normalization',
                                                                                                   value =100,
                                                                                                   min = 0,
                                                                                                   max = 500,
                                                                                                   step=0.5,
                                                                                                   disabled=True)
        self.hs['L[0][x][3][0][1][0][3]_fit_eng_range_post_edge_s_text'].layout = layout
        layout = {'width':'19%', 'height':'90%'}
        # self.hs['L[0][x][3][0][1][0][4]_fit_filter_spec_checkbox'] = widgets.Checkbox(description='flt spec',
        #                                                                                    description_tooltip='relative starting energy point (keV) of post-edge from edge energy for normalization',
        #                                                                                    value = False,
        #                                                                                    disabled=True)
        # self.hs['L[0][x][3][0][1][0][4]_fit_filter_spec_checkbox'].layout = layout

        self.hs['L[0][x][3][0][1][0]_fit_energy_range_box1'].children = get_handles(self.hs, 'L[0][x][3][0][1][0]_fit_energy_range_box1', -1)
        self.hs['L[0][x][3][0][1][0][0]_fit_eng_range_option_dropdown'].observe(self.L0_x_3_0_1_0_0_fit_eng_range_option_dropdown, names='value')
        self.hs['L[0][x][3][0][1][0][1]_fit_eng_range_edge_eng_text'].observe(self.L0_x_3_0_1_0_1_fit_eng_range_edge_eng_text, names='value')
        self.hs['L[0][x][3][0][1][0][2]_fit_eng_range_pre_edge_e_text'].observe(self.L0_x_3_0_1_0_2_fit_eng_range_pre_edge_e_text, names='value')
        self.hs['L[0][x][3][0][1][0][3]_fit_eng_range_post_edge_s_text'].observe(self.L0_x_3_0_1_0_3_fit_eng_range_post_edge_s_text, names='value')
        # self.hs['L[0][x][3][0][1][0][4]_fit_filter_spec_checkbox'].observe(self.L0_x_3_0_1_0_4_analysis_filter_spec_checkbox, names='value')

        layout = {'border':'none', 'width':f'{1*self.form_sz[1]-110}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][0][1][1]_fit_energy_range_box2'] = widgets.HBox()
        self.hs['L[0][x][3][0][1][1]_fit_energy_range_box2'].layout = layout
        layout = {'width':'19%', 'height':'90%'}
        self.hs['L[0][x][3][0][1][1][0]_fit_eng_range_wl_fit_s_text'] = widgets.BoundedFloatText(description='wl eng s',
                                                                                                description_tooltip='absolute energy starting point (keV) for whiteline fitting. "wl eng s" and "wl eng e" shoudl be roughly symmetric about whiteline peak.',
                                                                                                value =0,
                                                                                                min = 0,
                                                                                                max = 50000,
                                                                                                step=0.5,
                                                                                                disabled=True)
        self.hs['L[0][x][3][0][1][1][0]_fit_eng_range_wl_fit_s_text'].layout = layout
        layout = {'width':'19%', 'height':'90%'}
        self.hs['L[0][x][3][0][1][1][1]_fit_eng_range_wl_fit_e_text'] = widgets.BoundedFloatText(description='wl eng e',
                                                                                                description_tooltip='absolute energy ending point (keV) for whiteline fitting. "wl eng s" and "wl eng e" shoudl be roughly symmetric about whiteline peak.',
                                                                                                value =0,
                                                                                                min = 0,
                                                                                                max = 50030,
                                                                                                step=0.5,
                                                                                                disabled=True)
        self.hs['L[0][x][3][0][1][1][1]_fit_eng_range_wl_fit_e_text'].layout = layout
        layout = {'width':'19%', 'height':'90%'}
        self.hs['L[0][x][3][0][1][1][2]_fit_eng_range_edge0.5_fit_s_text'] = widgets.BoundedFloatText(description='edge0.5 s',
                                                                                                     description_tooltip='absolute energy starting point (keV) for edge-0.5 fitting. "edge0.5 s" and "edge0.5 e" shoudl be close to the foot and the peak locations on the ascendent edge of the spectra.',
                                                                                                     value =0,
                                                                                                     min = 0,
                                                                                                     max = 50000,
                                                                                                     step=0.5,
                                                                                                     disabled=True)
        self.hs['L[0][x][3][0][1][1][2]_fit_eng_range_edge0.5_fit_s_text'].layout = layout
        layout = {'width':'19%', 'height':'90%'}
        self.hs['L[0][x][3][0][1][1][3]_fit_eng_range_edge0.5_fit_e_text'] = widgets.BoundedFloatText(description='edge0.5 e',
                                                                                                       description_tooltip='absolute energy starting point (keV) for edge-0.5 fitting. "edge0.5 s" and "edge0.5 e" shoudl be close to the foot and the peak locations on the ascendent edge of the spectra.',
                                                                                                       value =0,
                                                                                                       min = 0,
                                                                                                       max = 50030,
                                                                                                       step=0.5,
                                                                                                       disabled=True)
        self.hs['L[0][x][3][0][1][1][3]_fit_eng_range_edge0.5_fit_e_text'].layout = layout

        layout = {'width':'15%', 'height':'90%', 'left':'7%'}
        self.hs['L[0][x][3][0][1][1][4]_fit_energy_confirm_btn'] = widgets.Button(description='Confirm',
                                                                                     description_tooltip='Confirm energy range settings',
                                                                                     disabled=True)
        self.hs['L[0][x][3][0][1][1][4]_fit_energy_confirm_btn'].layout = layout
        self.hs['L[0][x][3][0][1][1][4]_fit_energy_confirm_btn'].style.button_color = 'darkviolet'

        self.hs['L[0][x][3][0][1][1]_fit_energy_range_box2'].children = get_handles(self.hs, 'L[0][x][3][0][1][1]_fit_energy_range_box2', -1)
        self.hs['L[0][x][3][0][1][1][0]_fit_eng_range_wl_fit_s_text'].observe(self.L0_x_3_0_1_1_0_fit_eng_range_wl_fit_s_text, names='value')
        self.hs['L[0][x][3][0][1][1][1]_fit_eng_range_wl_fit_e_text'].observe(self.L0_x_3_0_1_1_1_fit_eng_range_wl_fit_e_text, names='value')
        self.hs['L[0][x][3][0][1][1][2]_fit_eng_range_edge0.5_fit_s_text'].observe(self.L0_x_3_0_1_1_2_fit_eng_range_edge0p5_fit_s_text, names='value')
        self.hs['L[0][x][3][0][1][1][3]_fit_eng_range_edge0.5_fit_e_text'].observe(self.L0_x_3_0_1_1_3_fit_eng_range_edge0p5_fit_e_text, names='value')
        self.hs['L[0][x][3][0][1][1][4]_fit_energy_confirm_btn'].on_click(self.L0_x_3_0_1_0_4_fit_eng_range_confirm_btn)

        self.hs['L[0][x][3][0][1]_fit_energy_range_box'].children = get_handles(self.hs, 'L[0][x][3][0][1]_fit_energy_range_box', -1)
        ## ## ## ## ## define type of analysis and energy range -- end

        self.hs['L[0][x][3][0]_fit_eng_config_box'].children = get_handles(self.hs, 'L[0][x][3][0]_fit_eng_config_box', -1)
        ## ## ## ## fit energy config box -- end

        ## ## ## ## ## define fitting parameters -- start
        self.hs['L[0][x][3][1]_fit_item_box'] = widgets.Tab()

        ## ## ## ## ## ## define analysis options parameters -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{0.96*self.form_sz[1]-98}px', 'height':f'{0.49*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][1][0]_fit_config_box'] = widgets.HBox()
        self.hs['L[0][x][3][1][0]_fit_config_box'].layout = layout

        self.hs['L[0][x][3][1][0][0]_fit_config'] = widgets.GridspecLayout(8, 200,
                                                                          layout = {"border":"3px solid #FFCC00",
                                                                                    'width':f'{0.96*self.form_sz[1]-104}px',
                                                                                    "height":f"{0.48*(self.form_sz[0]-128)}px",
                                                                                    "align_items":"flex-start",
                                                                                    "justify_items":"flex-start"})

        self.hs['L[0][x][3][1][0][0]_fit_config'][0, :33] = widgets.Checkbox(description='fit wl',
                                                                            description_tooltip="fit whiteline if it is checked",
                                                                            value =True,
                                                                            layout = {'width':f'{0.25*self.form_sz[1]-104}px'},
                                                                            indent = False,
                                                                            disabled=False)
        self.hs['L[0][x][3][1][0][0][0]_fit_config_fit_wl_checkbox'] = self.hs['L[0][x][3][1][0][0]_fit_config'][0, :33]
        self.hs['L[0][x][3][1][0][0]_fit_config'][0, 33:66] = widgets.Checkbox(description='fit edge',
                                                                               description_tooltip='fit edge if it is checked',
                                                                               value =False,
                                                                               layout = {'width':f'{0.25*self.form_sz[1]-104}px'},
                                                                               indent = False,
                                                                               disabled=False)
        self.hs['L[0][x][3][1][0][0][1]_fit_config_fit_edge_checkbox'] = self.hs['L[0][x][3][1][0][0]_fit_config'][0, 33:66]
        self.hs['L[0][x][3][1][0][0]_fit_config'][0, 66:99] = widgets.Checkbox(description='fit 50% peak',
                                                                                description_tooltip='fit the position where chi is 50% of maximum if it is checked',
                                                                                value=True,
                                                                                layout={
                                                                                    'width': f'{0.25 * self.form_sz[1] - 104}px'},
                                                                                indent=False,
                                                                                disabled=False)
        self.hs['L[0][x][3][1][0][0][3]_fit_config_fit_50%_checkbox'] = self.hs['L[0][x][3][1][0][0]_fit_config'][0, 66:99]
        self.hs['L[0][x][3][1][0][0]_fit_config'][0, 99:132] = widgets.Checkbox(description='filter spec',
                                                                                description_tooltip='filter spec before fitting edge if it is checked',
                                                                                value =False,
                                                                                layout = {'width':f'{0.25*self.form_sz[1]-104}px'},
                                                                                indent = False,
                                                                                disabled=False)
        self.hs['L[0][x][3][1][0][0][2]_fit_config_flt_spec_checkbox'] = self.hs['L[0][x][3][1][0][0]_fit_config'][0, 99:132]
        self.hs['L[0][x][3][1][0][0]_fit_config'][0, 132:165] = widgets.Checkbox(description='edge direct',
                                                                                 description_tooltip='find edge by derivative on raw spectra if it is checked',
                                                                                 value =False,
                                                                                 layout = {'width':f'{0.25*self.form_sz[1]-104}px'},
                                                                                 indent = False,
                                                                                 disabled=False)
        self.hs['L[0][x][3][1][0][0][4]_fit_config_cal_edge_direct_checkbox'] = self.hs['L[0][x][3][1][0][0]_fit_config'][0, 132:165]
        self.hs['L[0][x][3][1][0][0]_fit_config'][0, 165:198] = widgets.Checkbox(description='50% peak direct',
                                                                                 description_tooltip='cal edge at 50% of maximum chi if it is checked',
                                                                                 value =False,
                                                                                 layout = {'width':f'{0.25*self.form_sz[1]-104}px'},
                                                                                 indent = False,
                                                                                 disabled=False)
        self.hs['L[0][x][3][1][0][0][5]_fit_config_cal_edge50_direct_checkbox'] = self.hs['L[0][x][3][1][0][0]_fit_config'][0, 165:198]

        self.hs['L[0][x][3][1][0][0]_fit_config'][1, :70] = widgets.FloatSlider(description='edge jump thres',
                                                                                description_tooltip='edge jump in unit of the standard deviation of the signal in energy range pre to the edge. larger threshold enforces more restrict data quality validation on the data',
                                                                                value =1,
                                                                                min = 0,
                                                                                max = 50,
                                                                                step=0.1,
                                                                                disabled=True)
        self.hs['L[0][x][3][1][0][1][0]_fit_config_edge_jump_thres_slider'] = self.hs['L[0][x][3][1][0][0]_fit_config'][1, :70]
        self.hs['L[0][x][3][1][0][0]_fit_config'][1, 70:140] = widgets.FloatSlider(description='offset thres',
                                                                                   description_tooltip='offset between pre-edge and post-edge in unit of the standard deviation of pre-edge. larger offser enforces more restrict data quality validation on the data',
                                                                                   value =1,
                                                                                   min = 0,
                                                                                   max = 50,
                                                                                   step=0.1,
                                                                                   disabled=True)
        self.hs['L[0][x][3][1][0][1][1]_fit_config_edge_offset_thres_slider'] = self.hs['L[0][x][3][1][0][0]_fit_config'][1, 70:140]
        self.hs['L[0][x][3][1][0][0]_fit_config'][1, 150:200] = widgets.BoundedIntText(description='binning fac',
                                                                                       description_tooltip='binning factor applied to the spec image before xanes analysis',
                                                                                       value=1,
                                                                                       min=1,
                                                                                       max=10,
                                                                                       step=1,
                                                                                       layout={'width':f'{0.19 * (self.form_sz[1] - 104)}px'},
                                                                                       disabled=True)
        self.hs['L[0][x][3][1][0][1][2]_fit_config_bin_fac_text'] = self.hs['L[0][x][3][1][0][0]_fit_config'][1, 150:200]
        # self.hs['L[0][x][3][1][0][1][2]_fit_config_bin_fac_text'].layout = {'width':f'{0.19 * (self.form_sz[1] - 104)}px'}

        self.hs['L[0][x][3][1][0][0]_fit_config'][2, :20] = widgets.Checkbox(description='mask preview',
                                                                            description_tooltip='preview edge jump and edge offset masks',
                                                                            value =0,
                                                                            indent=False,
                                                                            disabled=True)
        self.hs['L[0][x][3][1][0][2][0]_fit_config_flt_prev_checkbox'] = self.hs['L[0][x][3][1][0][0]_fit_config'][2, :20]

        self.hs['L[0][x][3][1][0][0]_fit_config'][3, :70] = widgets.IntSlider(description='xy slice',
                                                                                description_tooltip='Select a slice to generate edge_jump and edge_offset filters for preview',
                                                                                value =0,
                                                                                min=0,
                                                                                max=10,
                                                                                disabled=True)
        self.hs['L[0][x][3][1][0][3][0]_fit_config_flt_prev_sli_slider'] = self.hs['L[0][x][3][1][0][0]_fit_config'][3, :70]

        self.hs['L[0][x][3][1][0][0]_fit_config'][3, 80:90] = widgets.Button(description='calc masks',
                                                                              description_tooltip='Calculate and preview the edge_jump and edge_offset filters',
                                                                              layout = {'width':f'{0.1*(self.form_sz[1]-104)}px'},
                                                                              disabled=True)
        self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_calc_btn'] = self.hs['L[0][x][3][1][0][0]_fit_config'][3, 80:90]
        self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_calc_btn'].style.button_color = 'darkviolet'

        self.hs['L[0][x][3][1][0][0]_fit_config'][3, 110:120] = widgets.Button(description='EM it',
                                                                              description_tooltip='Calculate and preview the edge_jump and edge_offset filters',
                                                                              layout = {'width':f'{0.1*(self.form_sz[1]-104)}px'},
                                                                              disabled=True)
        self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_em_it_btn'] = self.hs['L[0][x][3][1][0][0]_fit_config'][3, 110:120]
        self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_em_it_btn'].style.button_color = 'darkviolet'

        self.hs['L[0][x][3][1][0][0]_fit_config'][3, 140:150] = widgets.Button(description='OM it',
                                                                              description_tooltip='Calculate and preview the edge_jump and edge_offset filters',
                                                                              layout = {'width':f'{0.1*(self.form_sz[1]-104)}px'},
                                                                              disabled=True)
        self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_om_it_btn'] = self.hs['L[0][x][3][1][0][0]_fit_config'][3, 140:150]
        self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_om_it_btn'].style.button_color = 'darkviolet'

        self.hs['L[0][x][3][1][0][0]_fit_config'][3, 170:180] = widgets.Button(description='EM&OM it',
                                                                              description_tooltip='Calculate and preview the edge_jump and edge_offset filters',
                                                                              layout = {'width':f'{0.1*(self.form_sz[1]-104)}px'},
                                                                              disabled=True)
        self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_emom_it_btn'] = self.hs['L[0][x][3][1][0][0]_fit_config'][3, 170:180]
        self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_emom_it_btn'].style.button_color = 'darkviolet'

        self.hs['L[0][x][3][1][0][0]_fit_config'][4, :40] = widgets.Checkbox(description='LCF',
                                                                             description_tooltip='Linear Combination Fitting',
                                                                             value=0,
                                                                             indent=False,
                                                                             disabled=True)
        self.hs['L[0][x][3][1][0][4][0]_fit_config_lcf_checkbox'] = self.hs['L[0][x][3][1][0][0]_fit_config'][4, :40]

        self.hs['L[0][x][3][1][0][0]_fit_config'][4, 40:80] = widgets.Checkbox(description='Constr',
                                                                               value=1,
                                                                               description_tooltip='Apply unity constraint in LCF',
                                                                               indent=False,
                                                                               disabled=True)
        self.hs['L[0][x][3][1][0][4][5]_fit_config_lcf_constr_checkbox'] = self.hs['L[0][x][3][1][0][0]_fit_config'][4, 40:80]

        self.hs['L[0][x][3][1][0][0]_fit_config'][5, :40] = widgets.BoundedIntText(description='# of spec',
                                                                                   description_tooltip='total # of reference spectra for Linear Combination Fitting',
                                                                                   value=2,
                                                                                   min=2,
                                                                                   max=5,
                                                                                   step=1,
                                                                                   indent=False,
                                                                                   layout={'width':f'{0.19 * (self.form_sz[1] - 104)}px'},
                                                                                   disabled=True)
        self.hs['L[0][x][3][1][0][4][1]_fit_config_num_spec_text'] = self.hs['L[0][x][3][1][0][0]_fit_config'][5, :40]

        self.hs['L[0][x][3][1][0][0]_fit_config'][6, 14:40] = SelectFilesButton(option='askopenfilename',
                                                                                open_filetypes=(('text files', '*.txt'),),
                                                                                layout={
                                                                                  'left':f'{0.05*(self.form_sz[1] - 104)}px',
                                                                                  'top':f'{0.24*(self.form_sz[0] - 128)}px',
                                                                                  'width':f'{0.05*(self.form_sz[1] - 104)}px'}
                                                                              )
        self.hs['L[0][x][3][1][0][4][2]_fit_config_sel_ref_btn'] = self.hs['L[0][x][3][1][0][0]_fit_config'][6, 14:40]
        self.hs['L[0][x][3][1][0][4][2]_fit_config_sel_ref_btn'].description = "Load Ref Spec"
        try:
            with open(self.global_h.GUI_cfg_file, 'r') as f:
                cfg = json.load(f)
                self.hs['L[0][x][3][1][0][4][2]_fit_config_sel_ref_btn'].initialdir = cfg['xanes_ref_d']
        except:
            with open(self.global_h.GUI_cfg_file, 'r') as f:
                cfg = json.load(f)
                self.hs['L[0][x][3][1][0][4][2]_fit_config_sel_ref_btn'].initialdir = os.path.abspath(os.path.curdir)

        self.hs['L[0][x][3][1][0][0]_fit_config'][5:8, 50:150] = \
            widgets.SelectMultiple(options=[],
                                   value=[],
                                   layout={"width": f"{0.5 * self.form_sz[1] - 98}px",
                                           "height": f"{0.15 * (self.form_sz[0] - 128)}px"},
                                   rows=10,
                                   description='Ref Specs',
                                   disabled=True)
        self.hs['L[0][x][3][1][0][4][3]_fit_config_ref_spec_list'] = self.hs['L[0][x][3][1][0][0]_fit_config'][5:8, 50:150]

        self.hs['L[0][x][3][1][0][0]_fit_config'][6, 170:190] = widgets.Button(layout={'width':f'{0.1*(self.form_sz[1]-104)}px'},
                                                                               description='Remove',
                                                                               disabled=True)
        self.hs['L[0][x][3][1][0][4][4]_fit_config_ref_rm_btn'] = self.hs['L[0][x][3][1][0][0]_fit_config'][6, 170:190]
        self.hs['L[0][x][3][1][0][4][4]_fit_config_ref_rm_btn'].style.button_color = 'darkviolet'

        self.hs['L[0][x][3][1][0][0][0]_fit_config_fit_wl_checkbox'].observe(self.L0_x_3_0_4_3_0_0_fit_config_fit_wl_checkbox_change, names='value')
        self.hs['L[0][x][3][1][0][0][1]_fit_config_fit_edge_checkbox'].observe(self.L0_x_3_0_4_3_0_1_fit_config_fit_edge_checkbox_change, names='value')
        self.hs['L[0][x][3][1][0][0][3]_fit_config_fit_50%_checkbox'].observe(self.L0_x_3_1_0_0_3_fit_config_fit_edge50_checkbox, names='value')
        self.hs['L[0][x][3][1][0][0][2]_fit_config_flt_spec_checkbox'].observe(self.L0_x_3_0_4_3_0_2_fit_config_flt_spec_checkbox_change, names='value')
        self.hs['L[0][x][3][1][0][0][4]_fit_config_cal_edge_direct_checkbox'].observe(self.L0_x_3_0_4_3_0_3_fit_config_cal_edge_direct_checkbox_change, names='value')
        self.hs['L[0][x][3][1][0][0][5]_fit_config_cal_edge50_direct_checkbox'].observe(self.L0_x_3_0_4_3_0_3_fit_config_cal_edge50_direct_checkbox_change, names='value')
        self.hs['L[0][x][3][1][0][1][0]_fit_config_edge_jump_thres_slider'].observe(self.L0_x_3_0_4_3_0_1_fit_config_edge_jump_thres_slider_change, names='value')
        self.hs['L[0][x][3][1][0][1][1]_fit_config_edge_offset_thres_slider'].observe(self.L0_x_3_0_4_3_0_2_fit_config_edge_offset_thres_slider_change, names='value')
        self.hs['L[0][x][3][1][0][1][2]_fit_config_bin_fac_text'].observe(self.L0_x_3_1_0_1_2_fit_config_bin_fac_text_chg, names='value')
        self.hs['L[0][x][3][1][0][2][0]_fit_config_flt_prev_checkbox'].observe(self.L0_x_3_0_4_3_2_0_fit_config_flt_prev_checkbox_change, names='value')
        self.hs['L[0][x][3][1][0][3][0]_fit_config_flt_prev_sli_slider'].observe(self.L0_x_3_0_4_3_3_0_fit_config_flt_prev_sli_slider_change, names='value')
        self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_calc_btn'].on_click(self.L0_x_3_0_4_3_3_1_fit_config_flt_prev_calc_btn_click)
        self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_em_it_btn'].on_click(self.L0_x_3_0_4_3_3_1_fit_config_flt_prev_em_it_btn_click)
        self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_om_it_btn'].on_click(self.L0_x_3_0_4_3_3_1_fit_config_flt_prev_om_it_btn_click)
        self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_emom_it_btn'].on_click(self.L0_x_3_0_4_3_3_1_fit_config_flt_prev_emom_it_btn_click)
        self.hs['L[0][x][3][1][0][4][0]_fit_config_lcf_checkbox'].observe(self.L0_x_3_1_0_4_0_fit_config_lcf_checkbox_chg, names='value')
        self.hs['L[0][x][3][1][0][4][5]_fit_config_lcf_constr_checkbox'].observe(self.L0_x_3_1_0_4_5_fit_config_lcf_constr_checkbox_chg, names='value')
        self.hs['L[0][x][3][1][0][4][1]_fit_config_num_spec_text'].observe(self.L0_x_3_1_0_4_1_fit_config_num_spec_text_chg, names='value')
        self.hs['L[0][x][3][1][0][4][2]_fit_config_sel_ref_btn'].on_click(self.L0_x_3_1_0_4_2_fit_config_sel_ref_btn_clk)
        self.hs['L[0][x][3][1][0][4][3]_fit_config_ref_spec_list'].observe(self.L0_x_3_1_0_4_3_fit_config_ref_spec_list_chg, names='value')
        self.hs['L[0][x][3][1][0][4][4]_fit_config_ref_rm_btn'].on_click(self.L0_x_3_1_0_4_4_fit_config_ref_rm_btn_clk)
        self.hs['L[0][x][3][1][0]_fit_config_box'].children = get_handles(self.hs, 'L[0][x][3][1][0]_fit_config_box', -1)
        ## ## ## ## ## ## define analysis options parameters -- end

        ## ## ## ## ## ## define wl fitting parameters -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{0.96*self.form_sz[1]-98}px', 'height':f'{0.49*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][1][1]_fit_fit_wl_param_box'] = widgets.HBox()
        self.hs['L[0][x][3][1][1]_fit_fit_wl_param_box'].layout = layout

        self.hs['L[0][x][3][1][1][0]_fit_wl'] = widgets.GridspecLayout(8, 200,
                                                                        layout = {"border":"3px solid #FFCC00",
                                                                                  'width':f'{0.96*self.form_sz[1]-98}px',
                                                                                  "height":f"{0.48*(self.form_sz[0]-128)}px",
                                                                                  "align_items":"flex-start",
                                                                                  "justify_items":"flex-start"})

        self.hs['L[0][x][3][1][1][0]_fit_wl'][0, :50] = widgets.Dropdown(description='optimizer',
                                                                         description_tooltip='use scipy.optimize or numpy.polyfit',
                                                                         options = ['scipy',
                                                                                    'numpy'],
                                                                         value ='numpy',
                                                                         layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                         disabled=True)
        self.hs['L[0][x][3][1][1][0][0]_fit_wl_optimizer'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][0, :50]
        self.hs['L[0][x][3][1][1][0]_fit_wl'][0, 50:100] = widgets.Dropdown(description='peak func',
                                                                            description_tooltip='peak fitting functions',
                                                                            options = [2, 3, 4],
                                                                            value =2,
                                                                            layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                            disabled=True)
        self.hs['L[0][x][3][1][1][0][1]_fit_wl_fit_func'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][0, 50:100]
        self.hs['L[0][x][3][1][1][0]_fit_wl'][0, 100:150] = widgets.Checkbox(description='para bnd',
                                                                             value = False,
                                                                             description_tooltip = "if set boundaries to the peak fitting function's parameters",
                                                                             layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                             disabled=True)
        self.hs['L[0][x][3][1][1][0][2]_fit_wl_fit_use_bnd'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][0, 100:150]

        self.hs['L[0][x][3][1][1][0]_fit_wl'][1, 0:50] = widgets.BoundedFloatText(description = 'p0',
                                                                                   value = 0,
                                                                                   min = -1e5,
                                                                                   max = 1e5,
                                                                                   description_tooltip = "fitting function variable 0",
                                                                                   layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                   disabled=True)
        self.hs['L[0][x][3][1][1][0][3]_fit_wl_p0'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][1, 0:50]
        self.hs['L[0][x][3][1][1][0]_fit_wl'][1, 50:100] = widgets.BoundedFloatText(description = 'p1',
                                                                                     value = 0,
                                                                                     min = -1e5,
                                                                                     max = 1e5,
                                                                                     description_tooltip = "fitting function variable 1",
                                                                                     layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                     disabled=True)
        self.hs['L[0][x][3][1][1][0][4]_fit_wl_p1'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][1, 50:100]
        self.hs['L[0][x][3][1][1][0]_fit_wl'][1, 100:150] = widgets.BoundedFloatText(description = 'p2',
                                                                                      value = 0,
                                                                                      min = -1e5,
                                                                                      max = 1e5,
                                                                                      description_tooltip = "fitting function variable 2",
                                                                                      layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                      disabled=True)
        self.hs['L[0][x][3][1][1][0][5]_fit_wl_p2'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][1, 100:150]
        self.hs['L[0][x][3][1][1][0]_fit_wl'][1, 150:200] = widgets.BoundedFloatText(description = 'p3',
                                                                                      value = 0,
                                                                                      min = -1e5,
                                                                                      max = 1e5,
                                                                                      description_tooltip = "fitting function variable 3",
                                                                                      layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                      disabled=True)
        self.hs['L[0][x][3][1][1][0][6]_fit_wl_p3'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][1, 150:200]

        self.hs['L[0][x][3][1][1][0]_fit_wl'][2, 0:50] = widgets.BoundedFloatText(description = 'p4',
                                                                                   value = 0,
                                                                                   min = -1e5,
                                                                                   max = 1e5,
                                                                                   description_tooltip = "fitting function variable 4",
                                                                                   layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                   disabled=True)
        self.hs['L[0][x][3][1][1][0][7]_fit_wl_p4'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][2, 0:50]
        self.hs['L[0][x][3][1][1][0]_fit_wl'][2, 50:100] = widgets.Dropdown(description = 'p5',
                                                                            value = 'linear',
                                                                            options = ['linear'],
                                                                            description_tooltip = "fitting function variable 5",
                                                                            layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                            disabled=True)
        self.hs['L[0][x][3][1][1][0][8]_fit_wl_p5'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][2, 50:100]

        self.hs['L[0][x][3][1][1][0]_fit_wl'][3, 0:50] = widgets.Dropdown(description = 'jac',
                                                                          value = '3-point',
                                                                          options = ['2-point', '3-point', 'cs'],
                                                                          description_tooltip = "",
                                                                          layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                          disabled=True)
        self.hs['L[0][x][3][1][1][0][9]_fit_wl_jac'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][3, 0:50]
        self.hs['L[0][x][3][1][1][0]_fit_wl'][3, 50:100] = widgets.Dropdown(description = 'method',
                                                                            value = 'trf',
                                                                            options = ['trf', 'dogbox', 'lm'],
                                                                            description_tooltip = "",
                                                                            layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                            disabled=True)
        self.hs['L[0][x][3][1][1][0][10]_fit_wl_method'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][3, 50:100]
        self.hs['L[0][x][3][1][1][0]_fit_wl'][3, 100:150] = widgets.BoundedFloatText(description = 'ftol',
                                                                                      value = 1e-7,
                                                                                      min = 0,
                                                                                      max = 1e-3,
                                                                                      description_tooltip = "function value change tolerance for terminating the optimization process",
                                                                                      layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                      disabled=True)
        self.hs['L[0][x][3][1][1][0][11]_fit_wl_ftol'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][3, 100:150]
        self.hs['L[0][x][3][1][1][0]_fit_wl'][3, 150:200] = widgets.BoundedFloatText(description = 'xtol',
                                                                                      value = 1e-7,
                                                                                      min = 0,
                                                                                      max = 1e-3,
                                                                                      description_tooltip = "function parameter change tolerance for terminating the optimization process",
                                                                                      layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                      disabled=True)
        self.hs['L[0][x][3][1][1][0][12]_fit_wl_xtol'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][3, 150:200]
        self.hs['L[0][x][3][1][1][0]_fit_wl'][4, 0:50] = widgets.BoundedFloatText(description = 'gtol',
                                                                                   value = 1e-7,
                                                                                   min = 0,
                                                                                   max = 1e-3,
                                                                                   description_tooltip = "function gradient change tolerance for terminating the optimization process",
                                                                                   layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                   disabled=True)
        self.hs['L[0][x][3][1][1][0][13]_fit_wl_gtol'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][4, 0:50]
        self.hs['L[0][x][3][1][1][0]_fit_wl'][4, 50:100] = widgets.BoundedIntText(description = 'ufac',
                                                                                   value = 50,
                                                                                   min = 1,
                                                                                   max = 100,
                                                                                   description_tooltip = "upsampling factor to energy points in peak fitting",
                                                                                   layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                   disabled=True)
        self.hs['L[0][x][3][1][1][0][14]_fit_wl_ufac'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][4, 50:100]

        self.hs['L[0][x][3][1][1][0]_fit_wl'][5, 0:50] = widgets.BoundedFloatText(description = 'p0 lb',
                                                                                   value = 0,
                                                                                   min = -1e5,
                                                                                   max = 1e5,
                                                                                   description_tooltip = "p0 lower bound",
                                                                                   layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                   disabled=True)
        self.hs['L[0][x][3][1][1][0][15]_fit_wl_p0_lb'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][5, 0:50]
        self.hs['L[0][x][3][1][1][0]_fit_wl'][5, 50:100] = widgets.BoundedFloatText(description = 'p0 ub',
                                                                                     value = 0,
                                                                                     min = -1e5,
                                                                                     max = 1e5,
                                                                                     description_tooltip = "p0 upper bound",
                                                                                     layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                     disabled=True)
        self.hs['L[0][x][3][1][1][0][16]_fit_wl_p0_ub'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][5, 50:100]
        self.hs['L[0][x][3][1][1][0]_fit_wl'][5, 100:150] = widgets.BoundedFloatText(description = 'p1 lb',
                                                                                      value = 0,
                                                                                      min = -1e5,
                                                                                      max = 1e5,
                                                                                      description_tooltip = "p1 lower bound",
                                                                                      layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                      disabled=True)
        self.hs['L[0][x][3][1][1][0][17]_fit_wl_p1_lb'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][5, 100:150]
        self.hs['L[0][x][3][1][1][0]_fit_wl'][5, 150:200] = widgets.BoundedFloatText(description = 'p1 ub',
                                                                                      value = 0,
                                                                                      min = -1e5,
                                                                                      max = 1e5,
                                                                                      description_tooltip = "p1 upper bound",
                                                                                      layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                      disabled=True)
        self.hs['L[0][x][3][1][1][0][18]_fit_wl_p1_ub'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][5, 150:200]

        self.hs['L[0][x][3][1][1][0]_fit_wl'][6, 0:50] = widgets.BoundedFloatText(description = 'p2 lb',
                                                                                   value = 0,
                                                                                   min = -1e5,
                                                                                   max = 1e5,
                                                                                   description_tooltip = "p2 lower bound",
                                                                                   layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                   disabled=True)
        self.hs['L[0][x][3][1][1][0][19]_fit_wl_p2_lb'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][6, 0:50]
        self.hs['L[0][x][3][1][1][0]_fit_wl'][6, 50:100] = widgets.BoundedFloatText(description = 'p3 ub',
                                                                                     value = 0,
                                                                                     min = -1e5,
                                                                                     max = 1e5,
                                                                                     description_tooltip = "p3 upper bound",
                                                                                     layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                     disabled=True)
        self.hs['L[0][x][3][1][1][0][20]_fit_wl_p2_ub'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][6, 50:100]
        self.hs['L[0][x][3][1][1][0]_fit_wl'][6, 100:150] = widgets.BoundedFloatText(description = 'p4 lb',
                                                                                      value = 0,
                                                                                      min = -1e5,
                                                                                      max = 1e5,
                                                                                      description_tooltip = "p4 lower bound",
                                                                                      layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                      disabled=True)
        self.hs['L[0][x][3][1][1][0][21]_fit_wl_p3_lb'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][6, 100:150]
        self.hs['L[0][x][3][1][1][0]_fit_wl'][6, 150:200] = widgets.BoundedFloatText(description = 'p4 ub',
                                                                                     value = 0,
                                                                                     min = -1e5,
                                                                                     max = 1e5,
                                                                                     description_tooltip = "p4 upper bound",
                                                                                     layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                     disabled=True)
        self.hs['L[0][x][3][1][1][0][22]_fit_wl_p3_ub'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][6, 150:200]

        self.hs['L[0][x][3][1][1][0]_fit_wl'][7, 0:50] = widgets.BoundedFloatText(description = 'p5 lb',
                                                                                   value = 0,
                                                                                   min = -1e5,
                                                                                   max = 1e5,
                                                                                   description_tooltip = "p5 lower bound",
                                                                                   layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                   disabled=True)
        self.hs['L[0][x][3][1][1][0][23]_fit_wl_p4_lb'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][7, 0:50]
        self.hs['L[0][x][3][1][1][0]_fit_wl'][7, 50:100] = widgets.BoundedFloatText(description = 'p5 ub',
                                                                                     value = 0,
                                                                                     min = -1e5,
                                                                                     max = 1e5,
                                                                                     description_tooltip = "p5 upper bound",
                                                                                     layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                     disabled=True)
        self.hs['L[0][x][3][1][1][0][24]_fit_wl_p4_ub'] = self.hs['L[0][x][3][1][1][0]_fit_wl'][7, 50:100]

        self.hs['L[0][x][3][1][1][0][0]_fit_wl_optimizer'].observe(self.L0_x_3_0_4_0_0_0_fit_wl_optimizer, names='value')
        self.hs['L[0][x][3][1][1][0][1]_fit_wl_fit_func'].observe(self.L0_x_3_0_4_0_0_1_fit_wl_fit_func, names='value')
        self.hs['L[0][x][3][1][1][0][2]_fit_wl_fit_use_bnd'].observe(self.L0_x_3_0_4_0_0_2_fit_wl_fit_bnd, names='value')

        self.hs['L[0][x][3][1][1]_fit_fit_wl_param_box'].children = get_handles(self.hs, 'L[0][x][3][1][1]_fit_fit_wl_param_box', -1)
        ## ## ## ## ## ## define wl fitting parameters -- end

        ## ## ## ## ## ## define edge fitting parameters -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{0.96*self.form_sz[1]-98}px', 'height':f'{0.49*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][1][2]_fit_fit_edge_param_box'] = widgets.HBox()
        self.hs['L[0][x][3][1][2]_fit_fit_edge_param_box'].layout = layout

        self.hs['L[0][x][3][1][2][0]_fit_edge'] = widgets.GridspecLayout(8, 200,
                                                                         layout = {"border":"3px solid #FFCC00",
                                                                                  'width':f'{0.96*self.form_sz[1]-98}px',
                                                                                  "height":f"{0.48*(self.form_sz[0]-128)}px",
                                                                                  "align_items":"flex-start",
                                                                                  "justify_items":"flex-start"})

        self.hs['L[0][x][3][1][2][0]_fit_edge'][0, 0:50] = widgets.Dropdown(description='optimizer',
                                                                            description_tooltip='use scipy.optimize or numpy.polyfit',
                                                                            options = ['scipy',
                                                                                       'numpy'],
                                                                            value ='numpy',
                                                                            layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                            disabled=True)
        self.hs['L[0][x][3][1][2][0][0]_fit_edge_optimizer'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][0, 0:50]
        self.hs['L[0][x][3][1][2][0]_fit_edge'][0, 50:100] = widgets.Dropdown(description='fit func',
                                                                                description_tooltip='edge fitting functions',
                                                                                options=[2, 3, 4],
                                                                                value=3,
                                                                                layout={'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                disabled=True)
        self.hs['L[0][x][3][1][2][0][1]_fit_edge_fit_func'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][0, 50:100]
        self.hs['L[0][x][3][1][2][0]_fit_edge'][0, 100:150] = widgets.Checkbox(description='para bnd',
                                                                                value = False,
                                                                                description_tooltip = "if set boundaries to the peak fitting function's parameters",
                                                                                layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                disabled=True)
        self.hs['L[0][x][3][1][2][0][2]_fit_edge_fit_use_bnd'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][0, 100:150]

        self.hs['L[0][x][3][1][2][0]_fit_edge'][1, 0:50] = widgets.BoundedFloatText(description = 'p0',
                                                                                    value = 0,
                                                                                    min = -1e5,
                                                                                    max = 1e5,
                                                                                    description_tooltip = "fitting function variable 0",
                                                                                    layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                    disabled=True)
        self.hs['L[0][x][3][1][2][0][3]_fit_edge_p0'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][1, 0:50]
        self.hs['L[0][x][3][1][2][0]_fit_edge'][1, 50:100] = widgets.BoundedFloatText(description = 'p1',
                                                                                        value = 0,
                                                                                        min = -1e5,
                                                                                        max = 1e5,
                                                                                        description_tooltip = "fitting function variable 1",
                                                                                        layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                        disabled=True)
        self.hs['L[0][x][3][1][2][0][4]_fit_edge_p1'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][1, 50:100]
        self.hs['L[0][x][3][1][2][0]_fit_edge'][1, 100:150] = widgets.BoundedFloatText(description = 'p2',
                                                                                       value = 0,
                                                                                       min = -1e5,
                                                                                       max = 1e5,
                                                                                       description_tooltip = "fitting function variable 2",
                                                                                       layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                       disabled=True)
        self.hs['L[0][x][3][1][2][0][5]_fit_edge_p2'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][1, 100:150]
        self.hs['L[0][x][3][1][2][0]_fit_edge'][1, 150:200] = widgets.BoundedFloatText(description = 'p3',
                                                                                        value = 0,
                                                                                        min = -1e5,
                                                                                        max = 1e5,
                                                                                        description_tooltip = "fitting function variable 3",
                                                                                        layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                        disabled=True)
        self.hs['L[0][x][3][1][2][0][6]_fit_edge_p3'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][1, 150:200]

        self.hs['L[0][x][3][1][2][0]_fit_edge'][2, 0:50] = widgets.BoundedFloatText(description = 'p4',
                                                                                    value = 0,
                                                                                    min = -1e5,
                                                                                    max = 1e5,
                                                                                    description_tooltip = "fitting function variable 4",
                                                                                    layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                    disabled=True)
        self.hs['L[0][x][3][1][2][0][7]_fit_edge_p4'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][2, 0:50]
        self.hs['L[0][x][3][1][2][0]_fit_edge'][2, 50:100] = widgets.Dropdown(description = 'p5',
                                                                              value = 'linear',
                                                                              options = ['linear'],
                                                                              description_tooltip = "fitting function variable 5",
                                                                              layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                              disabled=True)
        self.hs['L[0][x][3][1][2][0][8]_fit_edge_p5'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][2, 50:100]

        self.hs['L[0][x][3][1][2][0]_fit_edge'][3, 0:50] = widgets.Dropdown(description = 'jac',
                                                                            value = '3-point',
                                                                            options = ['2-point', '3-point', 'cs'],
                                                                            description_tooltip = "",
                                                                            layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                            disabled=True)
        self.hs['L[0][x][3][1][2][0][9]_fit_edge_jac'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][3, 0:50]
        self.hs['L[0][x][3][1][2][0]_fit_edge'][3, 50:100] = widgets.Dropdown(description = 'method',
                                                                              value = 'trf',
                                                                              options = ['trf', 'dogbox', 'lm'],
                                                                              description_tooltip = "",
                                                                              layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                              disabled=True)
        self.hs['L[0][x][3][1][2][0][10]_fit_edge_method'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][3, 50:100]
        self.hs['L[0][x][3][1][2][0]_fit_edge'][3, 100:150] = widgets.BoundedFloatText(description = 'ftol',
                                                                                       value = 1e-7,
                                                                                       min = 0,
                                                                                       max = 1e-3,
                                                                                       description_tooltip = "function value change tolerance for terminating the optimization process",
                                                                                       layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                       disabled=True)
        self.hs['L[0][x][3][1][2][0][11]_fit_edge_ftol'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][3, 100:150]
        self.hs['L[0][x][3][1][2][0]_fit_edge'][3, 150:200] = widgets.BoundedFloatText(description = 'xtol',
                                                                                       value = 1e-7,
                                                                                       min = 0,
                                                                                       max = 1e-3,
                                                                                       description_tooltip = "function parameter change tolerance for terminating the optimization process",
                                                                                       layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                       disabled=True)
        self.hs['L[0][x][3][1][2][0][12]_fit_edge_xtol'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][3, 150:200]
        self.hs['L[0][x][3][1][2][0]_fit_edge'][4, 0:50] = widgets.BoundedFloatText(description = 'gtol',
                                                                                    value = 1e-7,
                                                                                    min = 0,
                                                                                    max = 1e-3,
                                                                                    description_tooltip = "function gradient change tolerance for terminating the optimization process",
                                                                                    layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                    disabled=True)
        self.hs['L[0][x][3][1][2][0][13]_fit_edge_gtol'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][4, 0:50]
        self.hs['L[0][x][3][1][2][0]_fit_edge'][4, 50:100] = widgets.BoundedIntText(description = 'ufac',
                                                                                    value = 50,
                                                                                    min = 1,
                                                                                    max = 100,
                                                                                    description_tooltip = "upsampling factor to energy points in peak fitting",
                                                                                    layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                    disabled=True)
        self.hs['L[0][x][3][1][2][0][14]_fit_edge_ufac'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][4, 50:100]

        self.hs['L[0][x][3][1][2][0]_fit_edge'][5, 0:50] = widgets.BoundedFloatText(description = 'p0 lb',
                                                                                    value = 0,
                                                                                    min = -1e5,
                                                                                    max = 1e5,
                                                                                    description_tooltip = "p0 lower bound",
                                                                                    layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                    disabled=True)
        self.hs['L[0][x][3][1][2][0][15]_fit_edge_p0_lb'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][5, 0:50]
        self.hs['L[0][x][3][1][2][0]_fit_edge'][5, 50:100] = widgets.BoundedFloatText(description = 'p0 ub',
                                                                                      value = 0,
                                                                                      min = -1e5,
                                                                                      max = 1e5,
                                                                                      description_tooltip = "p0 upper bound",
                                                                                      layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                      disabled=True)
        self.hs['L[0][x][3][1][2][0][16]_fit_edge_p0_ub'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][5, 50:100]
        self.hs['L[0][x][3][1][2][0]_fit_edge'][5, 100:150] = widgets.BoundedFloatText(description = 'p1 lb',
                                                                                        value = 0,
                                                                                        min = -1e5,
                                                                                        max = 1e5,
                                                                                        description_tooltip = "p1 lower bound",
                                                                                        layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                        disabled=True)
        self.hs['L[0][x][3][1][2][0][17]_fit_edge_p1_lb'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][5, 100:150]
        self.hs['L[0][x][3][1][2][0]_fit_edge'][5, 150:200] = widgets.BoundedFloatText(description = 'p1 ub',
                                                                                       value = 0,
                                                                                       min = -1e5,
                                                                                       max = 1e5,
                                                                                       description_tooltip = "p1 upper bound",
                                                                                       layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                       disabled=True)
        self.hs['L[0][x][3][1][2][0][18]_fit_edge_p1_ub'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][5, 150:200]

        self.hs['L[0][x][3][1][2][0]_fit_edge'][6, 0:50] = widgets.BoundedFloatText(description = 'p2 lb',
                                                                                    value = 0,
                                                                                    min = -1e5,
                                                                                    max = 1e5,
                                                                                    description_tooltip = "p2 lower bound",
                                                                                    layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                    disabled=True)
        self.hs['L[0][x][3][1][2][0][19]_fit_edge_p2_lb'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][6, 0:50]
        self.hs['L[0][x][3][1][2][0]_fit_edge'][6, 50:100] = widgets.BoundedFloatText(description = 'p3 ub',
                                                                                      value = 0,
                                                                                      min = -1e5,
                                                                                      max = 1e5,
                                                                                      description_tooltip = "p3 upper bound",
                                                                                      layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                      disabled=True)
        self.hs['L[0][x][3][1][2][0][20]_fit_edge_p2_ub'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][6, 50:100]
        self.hs['L[0][x][3][1][2][0]_fit_edge'][6, 100:150] = widgets.BoundedFloatText(description = 'p4 lb',
                                                                                       value = 0,
                                                                                       min = -1e5,
                                                                                       max = 1e5,
                                                                                       description_tooltip = "p4 lower bound",
                                                                                       layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                       disabled=True)
        self.hs['L[0][x][3][1][2][0][21]_fit_edge_p3_lb'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][6, 100:150]
        self.hs['L[0][x][3][1][2][0]_fit_edge'][6, 150:200] = widgets.BoundedFloatText(description = 'p4 ub',
                                                                                       value = 0,
                                                                                       min = -1e5,
                                                                                       max = 1e5,
                                                                                       description_tooltip = "p4 upper bound",
                                                                                       layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                       disabled=True)
        self.hs['L[0][x][3][1][2][0][22]_fit_edge_p3_ub'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][6, 150:200]

        self.hs['L[0][x][3][1][2][0]_fit_edge'][7, 0:50] = widgets.BoundedFloatText(description = 'p5 lb',
                                                                                    value = 0,
                                                                                    min = -1e5,
                                                                                    max = 1e5,
                                                                                    description_tooltip = "p5 lower bound",
                                                                                    layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                    disabled=True)
        self.hs['L[0][x][3][1][2][0][23]_fit_edge_p4_lb'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][7, 0:50]
        self.hs['L[0][x][3][1][2][0]_fit_edge'][7, 50:100] = widgets.BoundedFloatText(description = 'p5 ub',
                                                                                      value = 0,
                                                                                      min = -1e5,
                                                                                      max = 1e5,
                                                                                      description_tooltip = "p5 upper bound",
                                                                                      layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                      disabled=True)
        self.hs['L[0][x][3][1][2][0][24]_fit_edge_p4_ub'] = self.hs['L[0][x][3][1][2][0]_fit_edge'][7, 50:100]

        # self.hs['L[0][x][3][1][2][0][3]_fit_edge_refine'].observe(self.L0_x_3_0_4_2_0_3_fit_edge_refine, names='value')
        self.hs['L[0][x][3][1][2][0][0]_fit_edge_optimizer'].observe(self.L0_x_3_0_4_2_0_0_fit_edge_optimizer, names='value')
        self.hs['L[0][x][3][1][2][0][1]_fit_edge_fit_func'].observe(self.L0_x_3_0_4_2_0_1_fit_edge_fit_func, names='value')
        self.hs['L[0][x][3][1][2][0][2]_fit_edge_fit_use_bnd'].observe(self.L0_x_3_0_4_2_0_2_fit_edge_fit_use_bnd, names='value')

        self.hs['L[0][x][3][1][2]_fit_fit_edge_param_box'].children = get_handles(self.hs, 'L[0][x][3][1][2]_fit_fit_edge_param_box', -1)
        ## ## ## ## ## ## define edge fitting parameters -- end

        ## ## ## ## ## ## define saving setting -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{0.96*self.form_sz[1]-98}px', 'height':f'{0.49*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][1][3]_fit_save_setting_box'] = widgets.HBox()
        self.hs['L[0][x][3][1][3]_fit_save_setting_box'].layout = layout
        self.hs['L[0][x][3][1][3][0]_fit_save_setting_gridlayout'] = \
            widgets.GridspecLayout(8, 200,
                                   layout = {"border":"3px solid #FFCC00",
                                             "width":f"{0.96*self.form_sz[1]-98}px",
                                             "height":f"{0.48*(self.form_sz[0]-128)}px",
                                             "align_items":"flex-start",
                                             "justify_items":"flex-start"})
        self.hs['L[0][x][3][1][3][0]_fit_save_setting_gridlayout'][0:6, 0:100] = \
            widgets.SelectMultiple(options=FULL_SAVE_ITEM_OPTIONS,
                                   value=['wl_fit_coef'],
                                   layout = {"width":f"{0.5*self.form_sz[1]-98}px",
                                             "height":f"{0.4*(self.form_sz[0]-128)}px"},
                                   rows=10,
                                   description='Aval Items',
                                   disabled=True)
        self.hs['L[0][x][3][1][3][0][0]_fit_save_setting_multiselection'] = \
            self.hs['L[0][x][3][1][3][0]_fit_save_setting_gridlayout'][0:6, 0:100]

        self.hs['L[0][x][3][1][3][0]_fit_save_setting_gridlayout'][0:6, 100:200] = \
            widgets.SelectMultiple(options=FULL_SAVE_DEFAULT,
                                   value=['wl_peak_height_dir',],
                                   layout = {"width":f"{0.5*self.form_sz[1]-98}px",
                                             "height":f"{0.4*(self.form_sz[0]-128)}px"},
                                   description='Selected:',
                                   disabled=True)
        self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'] = \
            self.hs['L[0][x][3][1][3][0]_fit_save_setting_gridlayout'][0:6, 100:200]

        self.hs['L[0][x][3][1][3][0]_fit_save_setting_gridlayout'][7:8, 40:60] = \
            widgets.Button(description = '==>',
                           description_tooltip = 'Select saving items',
                           disabled = True)
        self.hs['L[0][x][3][1][3][0][2]_fit_save_setting_add_btn'] = \
            self.hs['L[0][x][3][1][3][0]_fit_save_setting_gridlayout'][7:8, 40:60]
        self.hs['L[0][x][3][1][3][0][2]_fit_save_setting_add_btn'].style.button_color = \
            'darkviolet'
        self.hs['L[0][x][3][1][3][0]_fit_save_setting_gridlayout'][7:8, 140:160] = \
            widgets.Button(description = '<==',
                           description_tooltip = 'Remove saving items',
                           disabled = True)
        self.hs['L[0][x][3][1][3][0][3]_fit_save_setting_remove_btn'] = \
            self.hs['L[0][x][3][1][3][0]_fit_save_setting_gridlayout'][7:8, 140:160]
        self.hs['L[0][x][3][1][3][0][3]_fit_save_setting_remove_btn'].style.button_color = \
            'darkviolet'
        self.hs['L[0][x][3][1][3][0][2]_fit_save_setting_add_btn'].on_click(self.L0_x_3_0_4_1_0_2_fit_save_setting_add_btn_click)
        self.hs['L[0][x][3][1][3][0][3]_fit_save_setting_remove_btn'].on_click(self.L0_x_3_0_4_1_0_3_fit_save_setting_remove_btn_click)
        self.hs['L[0][x][3][1][3]_fit_save_setting_box'].children = \
            get_handles(self.hs, 'L[0][x][3][1][3]_fit_save_setting_box', -1)
        ## ## ## ## ## ## define saving setting -- end

        self.hs['L[0][x][3][1]_fit_item_box'].children = [self.hs['L[0][x][3][1][0]_fit_config_box'],
                                                          self.hs['L[0][x][3][1][1]_fit_fit_wl_param_box'],
                                                          self.hs['L[0][x][3][1][2]_fit_fit_edge_param_box'],
                                                          self.hs['L[0][x][3][1][3]_fit_save_setting_box']]
        self.hs['L[0][x][3][1]_fit_item_box'].set_title(0, 'config analysis')
        self.hs['L[0][x][3][1]_fit_item_box'].set_title(1, 'fit whiteline params')
        self.hs['L[0][x][3][1]_fit_item_box'].set_title(2, 'fit edge params')
        self.hs['L[0][x][3][1]_fit_item_box'].set_title(3, 'saving setting')
        ## ## ## ## ## define fitting parameters -- send

        ## ## ## ## ## run xanes analysis -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{1*self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][2]_fit_run_box'] = widgets.HBox()
        self.hs['L[0][x][3][2]_fit_run_box'].layout = layout
        layout = {'width':'85%', 'height':'90%'}
        self.hs['L[0][x][3][2][0]_fit_run_text'] = widgets.Text(description='please check your settings before run the analysis .. ',
                                                                disabled=True)
        self.hs['L[0][x][3][2][0]_fit_run_text'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][x][3][2][1]_fit_run_btn'] = widgets.Button(description='run',
                                                                    disabled=True)
        self.hs['L[0][x][3][2][1]_fit_run_btn'].layout = layout
        self.hs['L[0][x][3][2][1]_fit_run_btn'].style.button_color = 'darkviolet'
        self.hs['L[0][x][3][2]_fit_run_box'].children = get_handles(self.hs, 'L[0][x][3][2]_fit_run_box', -1)
        self.hs['L[0][x][3][2][1]_fit_run_btn'].on_click(self.L0_x_3_0_5_1_fit_run_btn)
        ## ## ## ## ## run xanes analysis -- end

        ## ## ## ## ## run analysis progress -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][3]_fit_progress_box'] = widgets.HBox()
        self.hs['L[0][x][3][3]_fit_progress_box'].layout = layout
        layout = {'width':'100%', 'height':'90%'}
        self.hs['L[0][x][3][3][0]_fit_run_progress_bar'] = widgets.IntProgress(value=0,
                                                                               min=0,
                                                                               max=10,
                                                                               step=1,
                                                                               description='Completing:',
                                                                               bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                                                               orientation='horizontal')
        self.hs['L[0][x][3][3][0]_fit_run_progress_bar'].layout = layout
        self.hs['L[0][x][3][3]_fit_progress_box'].children = get_handles(self.hs, 'L[0][x][3][3]_fit_progress_box', -1)
        ## ## ## ## ## run analysis progress -- end

        self.hs['L[0][x][3]_fitting_form'].children = get_handles(self.hs, 'L[0][x][3]_fitting_form', -1)
        ## ## ## ## define functional widgets each tab in each sub-tab - analysis box in analysis&display TAB -- end
        self.bundle_fit_var_handles()
        self.boxes_logic()


    def boxes_logic(self):
        def compound_logic():
            if self.fit_wl_pos:
                if self.fit_wl_optimizer == 'scipy':
                    for ii in self.fit_wl_fit_func_arg_handles:
                        ii.disabled = False
                    for ii in self.fit_wl_optimizer_arg_handles:
                        ii.disabled = False
                    if self.fit_wl_fit_use_param_bnd:
                        for ii in self.fit_wl_fit_func_bnd_handles:
                            ii.disabled = False
                    else:
                        for ii in self.fit_wl_fit_func_bnd_handles:
                            ii.disabled = True
                    self.hs['L[0][x][3][1][1][0][2]_fit_wl_fit_use_bnd'].disabled = False
                    self.hs['L[0][x][3][1][1][0][1]_fit_wl_fit_func'].options = PEAK_LINE_SHAPES
                    self.hs['L[0][x][3][1][1][0][1]_fit_wl_fit_func'].description = 'peak func'
                    self.hs['L[0][x][3][1][1][0][1]_fit_wl_fit_func'].description_tooltip = 'peak fitting functions'
                elif self.fit_wl_optimizer == 'numpy':
                    for ii in self.fit_wl_fit_func_arg_handles:
                        ii.disabled = True
                    for ii in self.fit_wl_optimizer_arg_handles:
                        ii.disabled = True
                    for ii in self.fit_wl_fit_func_bnd_handles:
                        ii.disabled = True
                    self.hs['L[0][x][3][1][2][0][14]_fit_edge_ufac'].disabled = False
                    self.hs['L[0][x][3][1][1][0][2]_fit_wl_fit_use_bnd'].disabled = True
                    self.hs['L[0][x][3][1][1][0][1]_fit_wl_fit_func'].options = NUMPY_FIT_ORDER
                    self.hs['L[0][x][3][1][1][0][1]_fit_wl_fit_func'].description = 'order'
                    self.hs['L[0][x][3][1][1][0][1]_fit_wl_fit_func'].description_tooltip = 'order of polynominal fitting function'
            if self.fit_edge_pos:
                if self.fit_edge_optimizer == 'scipy':
                    for ii in self.fit_edge_fit_func_arg_handles:
                        ii.disabled = False
                    for ii in self.fit_edge_optimizer_arg_handles:
                        ii.disabled = False
                    if self.fit_edge_fit_use_param_bnd:
                        for ii in self.fit_edge_fit_func_bnd_handles:
                            ii.disabled = False
                    else:
                        for ii in self.fit_edge_fit_func_bnd_handles:
                            ii.disabled = True
                    self.hs['L[0][x][3][1][2][0][2]_fit_edge_fit_use_bnd'].disabled = False
                    self.hs['L[0][x][3][1][2][0][1]_fit_edge_fit_func'].options = EDGE_LINE_SHAPES
                    self.hs['L[0][x][3][1][2][0][1]_fit_edge_fit_func'].description = 'peak func'
                    self.hs['L[0][x][3][1][2][0][1]_fit_edge_fit_func'].description_tooltip = 'peak fitting functions'
                elif self.fit_edge_optimizer == 'numpy':
                    for ii in self.fit_edge_fit_func_arg_handles:
                        ii.disabled = True
                    for ii in self.fit_edge_optimizer_arg_handles:
                        ii.disabled = True
                    for ii in self.fit_edge_fit_func_bnd_handles:
                        ii.disabled = True
                    self.hs['L[0][x][3][1][2][0][14]_fit_edge_ufac'].disabled = False
                    self.hs['L[0][x][3][1][2][0][2]_fit_edge_fit_use_bnd'].disabled = True
                    self.hs['L[0][x][3][1][2][0][1]_fit_edge_fit_func'].options = NUMPY_FIT_ORDER
                    self.hs['L[0][x][3][1][2][0][1]_fit_edge_fit_func'].description = 'order'
                    self.hs['L[0][x][3][1][2][0][1]_fit_edge_fit_func'].description_tooltip = 'order of polynominal fitting function'

            if self.fit_flt_prev:
                if self.parent_h.gui_name == 'xanes3D':
                    self.hs['L[0][x][3][1][0][3][0]_fit_config_flt_prev_sli_slider'].disabled = False
                else:
                    self.hs['L[0][x][3][1][0][3][0]_fit_config_flt_prev_sli_slider'].disabled = True
                self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_calc_btn'].disabled = False
                if self.fit_flt_prev_maskit:
                    self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_em_it_btn'].disabled = False
                    self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_om_it_btn'].disabled = False
                    self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_emom_it_btn'].disabled = False
                else:
                    self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_em_it_btn'].disabled = True
                    self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_om_it_btn'].disabled = True
                    self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_emom_it_btn'].disabled = True
            else:
                self.hs['L[0][x][3][1][0][3][0]_fit_config_flt_prev_sli_slider'].disabled = True
                self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_calc_btn'].disabled = True
                self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_em_it_btn'].disabled = True
                self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_om_it_btn'].disabled = True
                self.hs['L[0][x][3][1][0][3][1]_fit_config_flt_prev_emom_it_btn'].disabled = True

            if self.fit_lcf:
                boxes = ['L[0][x][3][1][0][4]_']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                if len(self.hs['L[0][x][3][1][0][4][3]_fit_config_ref_spec_list'].options) == self.fit_lcf_ref_num:
                    self.hs['L[0][x][3][1][0][4][2]_fit_config_sel_ref_btn'].disabled = True
                    self.fit_lcf_ref_set = True
                else:
                    self.hs['L[0][x][3][1][0][4][2]_fit_config_sel_ref_btn'].disabled = False
                    self.fit_lcf_ref_set = False
                if self.fit_lcf_ref_set:
                    self.hs['L[0][x][3][2][1]_fit_run_btn'].disabled = False
                else:
                    self.hs['L[0][x][3][2][1]_fit_run_btn'].disabled = True
            else:
                boxes = ['L[0][x][3][1][0][4]_']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                self.hs['L[0][x][3][1][0][4][0]_fit_config_lcf_checkbox'].disabled = False
                self.hs['L[0][x][3][2][1]_fit_run_btn'].disabled = False

            if self.fit_edge50_pos:
                self.hs['L[0][x][3][1][0][0][0]_fit_config_fit_wl_checkbox'].value = True
                self.hs['L[0][x][3][1][0][0][0]_fit_config_fit_wl_checkbox'].disabled = True
                # self.hs['L[0][x][3][1][0][0][1]_fit_config_fit_edge_checkbox'].value = True
                # self.hs['L[0][x][3][1][0][0][1]_fit_config_fit_edge_checkbox'].disabled = True
            else:
                self.hs['L[0][x][3][1][0][0][0]_fit_config_fit_wl_checkbox'].disabled = False
                # self.hs['L[0][x][3][1][0][0][1]_fit_config_fit_edge_checkbox'].disabled = False

        if not (self.parent_h.xanes_file_configured & self.parent_h.xanes_alignment_done):
            boxes = ['L[0][x][3][0][1]_fit_energy_range_box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            self.hs['L[0][x][3][1][0][0][0]_fit_config_fit_wl_checkbox'].value = 1
        else:
            boxes = ['L[0][x][3][0][1]_fit_energy_range_box']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
        if self.xanes_fit_eng_config:
            if self.parent_h.xanes_fit_type == 'wl':
                boxes = ['L[0][x][3][1]_fit_item_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            else:
                boxes = ['L[0][x][3][1]_fit_item_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            if self.fit_wl_pos:
                boxes = ['L[0][x][3][1][1]_fit_fit_wl_param_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            else:
                boxes = ['L[0][x][3][1][1]_fit_fit_wl_param_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            if self.fit_edge_pos:
                boxes = ['L[0][x][3][1][2]_fit_fit_edge_param_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            else:
                boxes = ['L[0][x][3][1][2]_fit_fit_edge_param_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            compound_logic()
        else:
            boxes = ['L[0][x][3][1]_fit_item_box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)


    def bundle_fit_var_handles(self):
        self.fit_wl_fit_func_arg_handles = [self.hs['L[0][x][3][1][1][0][3]_fit_wl_p0'],
                                              self.hs['L[0][x][3][1][1][0][4]_fit_wl_p1'],
                                              self.hs['L[0][x][3][1][1][0][5]_fit_wl_p2'],
                                              self.hs['L[0][x][3][1][1][0][6]_fit_wl_p3'],
                                              self.hs['L[0][x][3][1][1][0][7]_fit_wl_p4'],
                                              self.hs['L[0][x][3][1][1][0][8]_fit_wl_p5']]

        self.fit_wl_optimizer_arg_handles = [self.hs['L[0][x][3][1][1][0][9]_fit_wl_jac'],
                                               self.hs['L[0][x][3][1][1][0][10]_fit_wl_method'],
                                               self.hs['L[0][x][3][1][1][0][11]_fit_wl_ftol'],
                                               self.hs['L[0][x][3][1][1][0][12]_fit_wl_xtol'],
                                               self.hs['L[0][x][3][1][1][0][13]_fit_wl_gtol'],
                                               self.hs['L[0][x][3][1][1][0][14]_fit_wl_ufac']]

        self.fit_wl_fit_func_bnd_handles = [self.hs['L[0][x][3][1][1][0][15]_fit_wl_p0_lb'],
                                              self.hs['L[0][x][3][1][1][0][16]_fit_wl_p0_ub'],
                                              self.hs['L[0][x][3][1][1][0][17]_fit_wl_p1_lb'],
                                              self.hs['L[0][x][3][1][1][0][18]_fit_wl_p1_ub'],
                                              self.hs['L[0][x][3][1][1][0][19]_fit_wl_p2_lb'],
                                              self.hs['L[0][x][3][1][1][0][20]_fit_wl_p2_ub'],
                                              self.hs['L[0][x][3][1][1][0][21]_fit_wl_p3_lb'],
                                              self.hs['L[0][x][3][1][1][0][22]_fit_wl_p3_ub'],
                                              self.hs['L[0][x][3][1][1][0][23]_fit_wl_p4_lb'],
                                              self.hs['L[0][x][3][1][1][0][24]_fit_wl_p4_ub']]

        self.fit_edge_fit_func_arg_handles = [self.hs['L[0][x][3][1][2][0][3]_fit_edge_p0'],
                                              self.hs['L[0][x][3][1][2][0][4]_fit_edge_p1'],
                                              self.hs['L[0][x][3][1][2][0][5]_fit_edge_p2'],
                                              self.hs['L[0][x][3][1][2][0][6]_fit_edge_p3'],
                                              self.hs['L[0][x][3][1][2][0][7]_fit_edge_p4'],
                                              self.hs['L[0][x][3][1][2][0][8]_fit_edge_p5']]

        self.fit_edge_optimizer_arg_handles = [self.hs['L[0][x][3][1][2][0][9]_fit_edge_jac'],
                                               self.hs['L[0][x][3][1][2][0][10]_fit_edge_method'],
                                               self.hs['L[0][x][3][1][2][0][11]_fit_edge_ftol'],
                                               self.hs['L[0][x][3][1][2][0][12]_fit_edge_xtol'],
                                               self.hs['L[0][x][3][1][2][0][13]_fit_edge_gtol'],
                                               self.hs['L[0][x][3][1][2][0][14]_fit_edge_ufac']]

        self.fit_edge_fit_func_bnd_handles = [self.hs['L[0][x][3][1][2][0][15]_fit_edge_p0_lb'],
                                              self.hs['L[0][x][3][1][2][0][16]_fit_edge_p0_ub'],
                                              self.hs['L[0][x][3][1][2][0][17]_fit_edge_p1_lb'],
                                              self.hs['L[0][x][3][1][2][0][18]_fit_edge_p1_ub'],
                                              self.hs['L[0][x][3][1][2][0][19]_fit_edge_p2_lb'],
                                              self.hs['L[0][x][3][1][2][0][20]_fit_edge_p2_ub'],
                                              self.hs['L[0][x][3][1][2][0][21]_fit_edge_p3_lb'],
                                              self.hs['L[0][x][3][1][2][0][22]_fit_edge_p3_ub'],
                                              self.hs['L[0][x][3][1][2][0][23]_fit_edge_p4_lb'],
                                              self.hs['L[0][x][3][1][2][0][24]_fit_edge_p4_ub']]


    def fit_coef_num(self, ftype='wl_fit_coef'):
        if ftype == 'wl_fit_coef':
            if self.fit_wl_optimizer == 'scipy':
                return len(list(PEAK_FIT_PARAM_DICT[self.fit_wl_fit_func].keys()))
            elif self.fit_wl_optimizer == 'numpy':
                return self.hs['L[0][x][3][1][1][0][1]_fit_wl_fit_func'].value+1
        elif ftype == 'edge_fit_coef':
            if self.fit_edge_optimizer == 'scipy':
                return len(list(EDGE_FIT_PARAM_DICT[self.fit_edge_fit_func].keys()))
            elif self.fit_edge_optimizer == 'numpy':
                return self.hs['L[0][x][3][1][2][0][1]_fit_edge_fit_func'].value+1
        elif ftype == 'post_edge_fit_coef':
            return 2
        elif ftype == 'pre_edge_fit_coef':
            return 2


    def set_save_items(self):
        if self.parent_h.xanes_fit_type == 'wl':
            options = set(['wl_pos_dir'])
            selects = set(['', 'wl_pos_dir'])
            if self.fit_wl_pos:
                options.add('wl_pos_fit')
                selects.add('wl_pos_fit')
                for ii in ('wl_fit_coef', 'wl_fit_err'):
                    options.add(ii)
                selects.add('wl_fit_err')
            if self.fit_edge_pos:
                options.add('edge_pos_fit')
                selects.add('edge_pos_fit')
                for ii in ('edge_fit_coef', 'edge_fit_err'):
                    options.add(ii)
                selects.add('edge_fit_err')
            if self.find_edge_pos_dir:
                options.add('edge_pos_dir')
                selects.add('edge_pos_dir')
        elif self.parent_h.xanes_fit_type == 'full':
            options = set(['norm_spec',
                           'wl_pos_dir',
                           'wl_peak_height_dir',
                           'centroid_of_eng',
                           'centroid_of_eng_relative_to_wl',
                           'weighted_attenuation',
                           'weighted_eng',
                           'edge_jump_filter',
                           'edge_offset_filter',
                           'pre_edge_sd',
                           'pre_edge_mean',
                           'post_edge_sd',
                           'post_edge_mean',
                           'pre_edge_fit_coef',
                           'post_edge_fit_coef'])
            selects = set(['',
                           'norm_spec',
                           'wl_pos_dir',
                           'wl_peak_height_dir',
                           'centroid_of_eng',
                           'centroid_of_eng_relative_to_wl',
                           'weighted_attenuation',
                           'weighted_eng',
                           'edge_jump_filter',
                           'edge_offset_filter',
                           'pre_edge_sd',
                           'pre_edge_mean',
                           'post_edge_sd',
                           'post_edge_mean'])
            if self.fit_wl_pos:
                options.add('wl_pos_fit')
                selects.add('wl_pos_fit')
                for ii in ('wl_fit_coef', 'wl_fit_err'):
                    options.add(ii)
                selects.add('wl_fit_err')
            if self.fit_edge_pos:
                options.add('edge_pos_fit')
                selects.add('edge_pos_fit')
                for ii in ('edge_fit_coef', 'edge_fit_err'):
                    options.add(ii)
                selects.add('edge_fit_err')
            if self.fit_edge50_pos:
                options.add('edge50_pos_fit')
                selects.add('edge50_pos_fit')
            if self.find_edge_pos_dir:
                options.add('edge_pos_dir')
                selects.add('edge_pos_dir')
            if self.find_edge50_pos_dir:
                options.add('edge50_pos_dir')
                selects.add('edge50_pos_dir')

            if self.fit_lcf:
                options.add('lcf_fit')
                options.add('lcf_fit_err')
                selects.add('lcf_fit')
                selects.add('lcf_fit_err')
        self.hs['L[0][x][3][1][3][0][0]_fit_save_setting_multiselection'].value = ['wl_pos_dir']
        self.hs['L[0][x][3][1][3][0][0]_fit_save_setting_multiselection'].options = list(sorted(options))
        self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].value = ['']
        self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].options = list(sorted(selects))


    def set_xanes_analysis_eng_bounds(self):
        eng_list_len = self.parent_h.xanes_fit_eng_list.shape[0]
        if self.parent_h.xanes_fit_wl_fit_eng_e > self.parent_h.xanes_fit_eng_list.max():
            self.parent_h.xanes_fit_wl_fit_eng_e = \
                self.parent_h.xanes_fit_eng_list.max()
        elif self.parent_h.xanes_fit_wl_fit_eng_e > self.parent_h.xanes_fit_eng_list.min():
            self.parent_h.xanes_fit_wl_fit_eng_e = \
                self.parent_h.xanes_fit_eng_list[int(eng_list_len/2)]
        if self.parent_h.xanes_fit_wl_fit_eng_s < self.parent_h.xanes_fit_eng_list.min():
            self.parent_h.xanes_fit_wl_fit_eng_s = \
                self.parent_h.xanes_fit_eng_list.min()
        elif self.parent_h.xanes_fit_wl_fit_eng_s < self.parent_h.xanes_fit_eng_list.max():
            self.parent_h.xanes_fit_wl_fit_eng_s = \
                self.parent_h.xanes_fit_eng_list[int(eng_list_len/2)] - 1
        self.hs['L[0][x][3][0][1][1][1]_fit_eng_range_wl_fit_e_text'].min = 0
        self.hs['L[0][x][3][0][1][1][1]_fit_eng_range_wl_fit_e_text'].max = \
            self.parent_h.xanes_fit_eng_list.max()
        self.hs['L[0][x][3][0][1][1][1]_fit_eng_range_wl_fit_e_text'].min = \
            self.parent_h.xanes_fit_eng_list.min()
        self.hs['L[0][x][3][0][1][1][0]_fit_eng_range_wl_fit_s_text'].min = 0
        self.hs['L[0][x][3][0][1][1][0]_fit_eng_range_wl_fit_s_text'].max = \
            self.parent_h.xanes_fit_eng_list.max()
        self.hs['L[0][x][3][0][1][1][0]_fit_eng_range_wl_fit_s_text'].min = \
            self.parent_h.xanes_fit_eng_list.min()

        if self.hs['L[0][x][3][0][1][0][0]_fit_eng_range_option_dropdown'].value == 'full':
            if ((self.parent_h.xanes_fit_edge_eng > self.parent_h.xanes_fit_eng_list.max()) or
                self.parent_h.xanes_fit_edge_eng < self.parent_h.xanes_fit_eng_list.min()):
                self.parent_h.xanes_fit_edge_eng  = \
                    self.parent_h.xanes_fit_eng_list[int(eng_list_len/2)]
            if self.parent_h.xanes_fit_edge_0p5_fit_e > self.parent_h.xanes_fit_eng_list.max():
                self.parent_h.xanes_fit_edge_0p5_fit_e = \
                    self.parent_h.xanes_fit_eng_list.max()
            elif self.parent_h.xanes_fit_edge_0p5_fit_e < self.parent_h.xanes_fit_eng_list.min():
                self.parent_h.xanes_fit_edge_0p5_fit_e = \
                    self.parent_h.xanes_fit_eng_list[int(eng_list_len/2)]
            if self.parent_h.xanes_fit_edge_0p5_fit_s < self.parent_h.xanes_fit_eng_list.min():
                self.parent_h.xanes_fit_edge_0p5_fit_s = \
                    self.parent_h.xanes_fit_eng_list.min()
            elif self.parent_h.xanes_fit_edge_0p5_fit_s > self.parent_h.xanes_fit_eng_list.max():
                self.parent_h.xanes_fit_edge_0p5_fit_s = \
                    self.parent_h.xanes_fit_eng_list[int(eng_list_len/2)] - 1
            self.hs['L[0][x][3][0][1][0][1]_fit_eng_range_edge_eng_text'].min = 0
            self.hs['L[0][x][3][0][1][0][1]_fit_eng_range_edge_eng_text'].max = \
                self.parent_h.xanes_fit_eng_list.max()
            self.hs['L[0][x][3][0][1][0][1]_fit_eng_range_edge_eng_text'].min = \
                self.parent_h.xanes_fit_eng_list.min()

            self.hs['L[0][x][3][0][1][1][1]_fit_eng_range_wl_fit_e_text'].min = 0
            self.hs['L[0][x][3][0][1][1][1]_fit_eng_range_wl_fit_e_text'].max = \
                self.parent_h.xanes_fit_eng_list.max()
            self.hs['L[0][x][3][0][1][1][1]_fit_eng_range_wl_fit_e_text'].min = \
                self.parent_h.xanes_fit_eng_list.min()

            self.hs['L[0][x][3][0][1][1][0]_fit_eng_range_wl_fit_s_text'].min = 0
            self.hs['L[0][x][3][0][1][1][0]_fit_eng_range_wl_fit_s_text'].max = \
                self.parent_h.xanes_fit_eng_list.max()
            self.hs['L[0][x][3][0][1][1][0]_fit_eng_range_wl_fit_s_text'].min = \
                self.parent_h.xanes_fit_eng_list.min()

            self.hs['L[0][x][3][0][1][1][3]_fit_eng_range_edge0.5_fit_e_text'].min = 0
            self.hs['L[0][x][3][0][1][1][3]_fit_eng_range_edge0.5_fit_e_text'].max = \
                self.parent_h.xanes_fit_eng_list.max()
            self.hs['L[0][x][3][0][1][1][3]_fit_eng_range_edge0.5_fit_e_text'].min = \
                self.parent_h.xanes_fit_eng_list.min()

            self.hs['L[0][x][3][0][1][1][2]_fit_eng_range_edge0.5_fit_s_text'].min = 0
            self.hs['L[0][x][3][0][1][1][2]_fit_eng_range_edge0.5_fit_s_text'].max = \
                self.parent_h.xanes_fit_eng_list.max()
            self.hs['L[0][x][3][0][1][1][2]_fit_eng_range_edge0.5_fit_s_text'].min = \
                self.parent_h.xanes_fit_eng_list.min()


    def update_fit_params(self):
        self.parent_h.xanes_fit_type = self.hs['L[0][x][3][0][1][0][0]_fit_eng_range_option_dropdown'].value
        if self.parent_h.xanes_fit_type == 'wl':
            self.parent_h.xanes_fit_wl_fit_eng_s = self.hs['L[0][x][3][0][1][1][0]_fit_eng_range_wl_fit_s_text'].value
            self.parent_h.xanes_fit_wl_fit_eng_e = self.hs['L[0][x][3][0][1][1][1]_fit_eng_range_wl_fit_e_text'].value

            self.fit_wl_pos = self.hs['L[0][x][3][1][0][0][0]_fit_config_fit_wl_checkbox'].value
            self.fit_use_flt_spec = self.hs['L[0][x][3][1][0][0][2]_fit_config_flt_spec_checkbox'].value

            if self.fit_wl_pos:
                self.fit_wl_optimizer = self.hs['L[0][x][3][1][1][0][0]_fit_wl_optimizer'].value
                self.fit_wl_fit_func = self.hs['L[0][x][3][1][1][0][1]_fit_wl_fit_func'].value
            else:
                self.fit_wl_optimizer = None
                self.fit_wl_fit_func = None

            self.analysis_saving_items = set(self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].options)
            self.analysis_saving_items.remove('')
        elif self.parent_h.xanes_fit_type == 'full':
            self.parent_h.xanes_fit_edge_eng = self.hs['L[0][x][3][0][1][0][1]_fit_eng_range_edge_eng_text'].value
            self.parent_h.xanes_fit_pre_edge_e = self.hs['L[0][x][3][0][1][0][2]_fit_eng_range_pre_edge_e_text'].value
            self.parent_h.xanes_fit_post_edge_s = self.hs['L[0][x][3][0][1][0][3]_fit_eng_range_post_edge_s_text'].value
            self.parent_h.xanes_fit_wl_fit_eng_s = self.hs['L[0][x][3][0][1][1][0]_fit_eng_range_wl_fit_s_text'].value
            self.parent_h.xanes_fit_wl_fit_eng_e = self.hs['L[0][x][3][0][1][1][1]_fit_eng_range_wl_fit_e_text'].value
            self.parent_h.xanes_fit_edge_0p5_fit_s = self.hs['L[0][x][3][0][1][1][2]_fit_eng_range_edge0.5_fit_s_text'].value
            self.parent_h.xanes_fit_edge_0p5_fit_e = self.hs['L[0][x][3][0][1][1][3]_fit_eng_range_edge0.5_fit_e_text'].value

            self.fit_wl_pos = self.hs['L[0][x][3][1][0][0][0]_fit_config_fit_wl_checkbox'].value
            self.fit_edge_pos = self.hs['L[0][x][3][1][0][0][1]_fit_config_fit_edge_checkbox'].value
            self.fit_edge50_pos = self.hs['L[0][x][3][1][0][0][3]_fit_config_fit_50%_checkbox'].value
            self.find_edge_pos_dir = self.hs['L[0][x][3][1][0][0][4]_fit_config_cal_edge_direct_checkbox'].value
            self.find_edge50_pos_dir = self.hs['L[0][x][3][1][0][0][5]_fit_config_cal_edge50_direct_checkbox'].value
            self.fit_use_flt_spec = self.hs['L[0][x][3][1][0][0][2]_fit_config_flt_spec_checkbox'].value
            self.fit_flt_prev = self.hs['L[0][x][3][1][0][2][0]_fit_config_flt_prev_checkbox'].value

            if self.fit_wl_pos:
                self.fit_wl_optimizer = self.hs['L[0][x][3][1][1][0][0]_fit_wl_optimizer'].value
                self.fit_wl_fit_func = self.hs['L[0][x][3][1][1][0][1]_fit_wl_fit_func'].value
            else:
                self.fit_wl_optimizer = None
                self.fit_wl_fit_func = None
            if self.fit_edge_pos:
                self.fit_edge_optimizer = self.hs['L[0][x][3][1][2][0][0]_fit_edge_optimizer'].value
                self.fit_edge_fit_func = self.hs['L[0][x][3][1][2][0][1]_fit_edge_fit_func'].value
            else:
                self.fit_edge_optimizer = None
                self.fit_edge_fit_func = None

            self.analysis_saving_items = set(self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].options)
            self.analysis_saving_items.remove('')
        if self.parent_h.xanes_fit_spectrum is None:
            self.parent_h.xanes_fit_spectrum = np.ndarray(self.parent_h.xanes_fit_data_shape[1:], dtype=np.float32)
        if self.parent_h.gui_name == 'xanes3D':
            self.parent_h.update_xanes3D_config()
            json.dump(self.parent_h.xanes_config, open(self.parent_h.xanes_save_trial_reg_config_filename, 'w'), cls=NumpyArrayEncoder)
        elif self.parent_h.gui_name == 'xanes2D':
            self.parent_h.update_xanes2D_config()
            json.dump(self.parent_h.xanes_config, open(self.parent_h.xanes_file_save_trial_reg_config_filename, 'w'), cls=NumpyArrayEncoder)


    def L0_x_3_0_1_0_0_fit_eng_range_option_dropdown(self, a):
        self.parent_h.xanes_fit_eng_configured = False
        self.xanes_fit_eng_config = False

        self.hs['L[0][x][3][0][1][0][1]_fit_eng_range_edge_eng_text'].value = self.parent_h.xanes_fit_edge_eng
        self.hs['L[0][x][3][0][1][1][1]_fit_eng_range_wl_fit_e_text'].value = self.parent_h.xanes_fit_wl_fit_eng_e
        self.hs['L[0][x][3][0][1][1][0]_fit_eng_range_wl_fit_s_text'].value = self.parent_h.xanes_fit_wl_fit_eng_s
        self.hs['L[0][x][3][0][1][1][3]_fit_eng_range_edge0.5_fit_e_text'].value = self.parent_h.xanes_fit_edge_0p5_fit_e
        self.hs['L[0][x][3][0][1][1][2]_fit_eng_range_edge0.5_fit_s_text'].value = self.parent_h.xanes_fit_edge_0p5_fit_s
        self.hs['L[0][x][3][0][1][0][2]_fit_eng_range_pre_edge_e_text'].value = self.parent_h.xanes_fit_pre_edge_e
        self.hs['L[0][x][3][0][1][0][3]_fit_eng_range_post_edge_s_text'].value = self.parent_h.xanes_fit_post_edge_s
        self.set_xanes_analysis_eng_bounds()

        if a['owner'].value == 'wl':
            layout = {'width': '19%', 'height': '100%', 'top': '15%', 'visibility': 'hidden'}
            self.hs['L[0][x][3][0][1][0][1]_fit_eng_range_edge_eng_text'].layout = layout
            layout = {'width': '19%', 'height': '100%', 'top': '15%', 'visibility': 'hidden'}
            self.hs['L[0][x][3][0][1][0][2]_fit_eng_range_pre_edge_e_text'].layout = layout
            layout = {'width': '19%', 'height': '100%', 'top': '15%', 'visibility': 'hidden'}
            self.hs['L[0][x][3][0][1][0][3]_fit_eng_range_post_edge_s_text'].layout = layout
            layout = {'width': '19%', 'height': '100%', 'top': '25%', 'visibility': 'hidden'}
            self.hs['L[0][x][3][0][1][1][2]_fit_eng_range_edge0.5_fit_s_text'].layout = layout
            layout = {'width': '19%', 'height': '100%', 'top': '25%', 'visibility': 'hidden'}
            self.hs['L[0][x][3][0][1][1][3]_fit_eng_range_edge0.5_fit_e_text'].layout = layout

            # self.hs['L[0][x][3][1][0][0][0]_fit_config_fit_wl_checkbox'].value = True
            # self.hs['L[0][x][3][1][0][0][1]_fit_config_fit_edge_checkbox'].value = False
            # self.hs['L[0][x][3][1][0][0][3]_fit_config_fit_50%_checkbox'].value = False
            # self.hs['L[0][x][3][1][0][0][2]_fit_config_flt_spec_checkbox'].value = False
            # self.hs['L[0][x][3][1][0][0][4]_fit_config_cal_edge_direct_checkbox'].value = False
            # self.hs['L[0][x][3][1][0][0][5]_fit_config_cal_edge50_direct_checkbox'].value = False
            # self.hs['L[0][x][3][1][0][2][0]_fit_config_flt_prev_checkbox'].value = False
            
            # self.hs['L[0][x][3][1][0][0][4]_fit_config_cal_edge_direct_checkbox'].disabled = True
            # self.hs['L[0][x][3][1][0][0][1]_fit_config_fit_edge_checkbox'].disabled = True
            # self.hs['L[0][x][3][1][0][0][3]_fit_config_fit_50%_checkbox'].disabled = True
            # self.hs['L[0][x][3][1][0][0][5]_fit_config_cal_edge50_direct_checkbox'].disabled = True
            # self.hs['L[0][x][3][1][0][1][0]_fit_config_edge_jump_thres_slider'].disabled = True
            # self.hs['L[0][x][3][1][0][1][1]_fit_config_edge_offset_thres_slider'].disabled = True
            # self.hs['L[0][x][3][1][0][2][0]_fit_config_flt_prev_checkbox'].disabled = True
            # self.fit_flt_prev = False
        elif a['owner'].value == 'full':
            layout = {'width': '19%', 'height': '100%', 'top': '15%', 'visibility': 'visible'}
            self.hs['L[0][x][3][0][1][0][1]_fit_eng_range_edge_eng_text'].layout = layout
            layout = {'width': '19%', 'height': '100%', 'top': '15%', 'visibility': 'visible'}
            self.hs['L[0][x][3][0][1][0][2]_fit_eng_range_pre_edge_e_text'].layout = layout
            self.hs['L[0][x][3][0][1][0][2]_fit_eng_range_pre_edge_e_text'].max = 0
            self.hs['L[0][x][3][0][1][0][2]_fit_eng_range_pre_edge_e_text'].value = -50
            self.hs['L[0][x][3][0][1][0][2]_fit_eng_range_pre_edge_e_text'].min = -500
            layout = {'width': '19%', 'height': '100%', 'top': '15%', 'visibility': 'visible'}
            self.hs['L[0][x][3][0][1][0][3]_fit_eng_range_post_edge_s_text'].layout = layout
            self.hs['L[0][x][3][0][1][0][3]_fit_eng_range_post_edge_s_text'].min = 0
            self.hs['L[0][x][3][0][1][0][3]_fit_eng_range_post_edge_s_text'].value = 100
            self.hs['L[0][x][3][0][1][0][3]_fit_eng_range_post_edge_s_text'].max = 500
            layout = {'width': '19%', 'height': '100%', 'top': '25%', 'visibility': 'visible'}
            self.hs['L[0][x][3][0][1][1][2]_fit_eng_range_edge0.5_fit_s_text'].layout = layout
            layout = {'width': '19%', 'height': '100%', 'top': '25%', 'visibility': 'visible'}
            self.hs['L[0][x][3][0][1][1][3]_fit_eng_range_edge0.5_fit_e_text'].layout = layout

            # self.hs['L[0][x][3][1][0][0][0]_fit_config_fit_wl_checkbox'].value = True
            # self.hs['L[0][x][3][1][0][0][1]_fit_config_fit_edge_checkbox'].value = False
            # self.hs['L[0][x][3][1][0][0][3]_fit_config_fit_50%_checkbox'].value = True
            # self.hs['L[0][x][3][1][0][0][2]_fit_config_flt_spec_checkbox'].value = False
            # self.hs['L[0][x][3][1][0][0][4]_fit_config_cal_edge_direct_checkbox'].value = False
            # self.hs['L[0][x][3][1][0][0][5]_fit_config_cal_edge50_direct_checkbox'].value = False
            # self.hs['L[0][x][3][1][0][2][0]_fit_config_flt_prev_checkbox'].value = False
                        
            # self.hs['L[0][x][3][1][0][0][0]_fit_config_fit_wl_checkbox'].disabled = False
            # self.hs['L[0][x][3][1][0][0][4]_fit_config_cal_edge_direct_checkbox'].disabled = False
            # self.hs['L[0][x][3][1][0][0][1]_fit_config_fit_edge_checkbox'].disabled = False
            # self.hs['L[0][x][3][1][0][0][3]_fit_config_fit_50%_checkbox'].disabled = False
            # self.hs['L[0][x][3][1][0][0][5]_fit_config_cal_edge50_direct_checkbox'].disabled = False
            # self.hs['L[0][x][3][1][0][2][0]_fit_config_flt_prev_checkbox'].disabled = False
            # self.hs['L[0][x][3][1][0][1][0]_fit_config_edge_jump_thres_slider'].disabled = False
            # self.hs['L[0][x][3][1][0][1][1]_fit_config_edge_offset_thres_slider'].disabled = False
        self.hs['L[0][x][3][2][0]_fit_run_text'].value ='please check your settings before run the analysis ...'
        self.fit_flt_prev_configed = False
        self.boxes_logic()


    def L0_x_3_0_1_0_1_fit_eng_range_edge_eng_text(self, a):
        self.parent_h.xanes_fit_eng_configured = False
        self.xanes_fit_eng_config = False
        self.fit_flt_prev_configed = False
        self.parent_h.xanes_fit_edge_eng = self.hs['L[0][x][3][0][1][0][1]_fit_eng_range_edge_eng_text'].value
        self.hs['L[0][x][3][2][0]_fit_run_text'].description ='please check your settings before run the analysis ...'


    def L0_x_3_0_1_0_2_fit_eng_range_pre_edge_e_text(self, a):
        self.parent_h.xanes_fit_eng_configured = False
        self.xanes_fit_eng_config = False
        self.fit_flt_prev_configed = False
        self.parent_h.xanes_fit_pre_edge_e = self.hs['L[0][x][3][0][1][0][2]_fit_eng_range_pre_edge_e_text'].value
        self.hs['L[0][x][3][2][0]_fit_run_text'].description ='please check your settings before run the analysis ...'


    def L0_x_3_0_1_0_3_fit_eng_range_post_edge_s_text(self, a):
        self.parent_h.xanes_fit_eng_configured = False
        self.xanes_fit_eng_config = False
        self.fit_flt_prev_configed = False
        self.parent_h.xanes_fit_post_edge_s = self.hs['L[0][x][3][0][1][0][3]_fit_eng_range_post_edge_s_text'].value
        self.hs['L[0][x][3][2][0]_fit_run_text'].description ='please check your settings before run the analysis ...'


    def L0_x_3_0_1_1_0_fit_eng_range_wl_fit_s_text(self, a):
        self.parent_h.xanes_fit_eng_configured = False
        self.xanes_fit_eng_config = False
        self.hs['L[0][x][3][2][0]_fit_run_text'].value ='please check your settings before run the analysis ...'
        self.parent_h.xanes_fit_wl_fit_eng_s = self.hs['L[0][x][3][0][1][1][0]_fit_eng_range_wl_fit_s_text'].value


    def L0_x_3_0_1_1_1_fit_eng_range_wl_fit_e_text(self, a):
        self.parent_h.xanes_fit_eng_configured = False
        self.xanes_fit_eng_config = False
        self.hs['L[0][x][3][2][0]_fit_run_text'].value ='please check your settings before run the analysis ...'
        self.parent_h.xanes_fit_wl_fit_eng_e = self.hs['L[0][x][3][0][1][1][1]_fit_eng_range_wl_fit_e_text'].value


    def L0_x_3_0_1_1_2_fit_eng_range_edge0p5_fit_s_text(self, a):
        self.parent_h.xanes_fit_eng_configured = False
        self.xanes_fit_eng_config = False
        self.hs['L[0][x][3][2][0]_fit_run_text'].value ='please check your settings before run the analysis ...'
        self.parent_h.xanes_fit_edge_0p5_fit_s = self.hs['L[0][x][3][0][1][1][2]_fit_eng_range_edge0.5_fit_s_text'].value


    def L0_x_3_0_1_1_3_fit_eng_range_edge0p5_fit_e_text(self, a):
        self.parent_h.xanes_fit_eng_configured = False
        self.xanes_fit_eng_config = False
        self.hs['L[0][x][3][2][0]_fit_run_text'].value ='please check your settings before run the analysis ...'
        self.parent_h.xanes_fit_edge_0p5_fit_e = self.hs['L[0][x][3][0][1][1][3]_fit_eng_range_edge0.5_fit_e_text'].value


    def L0_x_3_0_1_0_4_fit_eng_range_confirm_btn(self, a):
        self.update_fit_params()
        self.parent_h.xanes_fit_type = self.hs['L[0][x][3][0][1][0][0]_fit_eng_range_option_dropdown'].value
        if self.parent_h.xanes_fit_type == 'wl':
            self.parent_h.xanes_fit_wl_fit_eng_s = self.hs['L[0][x][3][0][1][1][0]_fit_eng_range_wl_fit_s_text'].value
            self.parent_h.xanes_fit_wl_fit_eng_e = self.hs['L[0][x][3][0][1][1][1]_fit_eng_range_wl_fit_e_text'].value
            # self.hs['L[0][x][3][1][0][0][4]_fit_config_cal_edge_direct_checkbox'].disabled = True
            # self.hs['L[0][x][3][1][0][0][1]_fit_config_fit_edge_checkbox'].disabled = True
            # self.hs['L[0][x][3][1][0][0][3]_fit_config_fit_50%_checkbox'].disabled = True
            # self.hs['L[0][x][3][1][0][0][5]_fit_config_cal_edge50_direct_checkbox'].disabled = True
            # self.hs['L[0][x][3][1][0][1][0]_fit_config_edge_jump_thres_slider'].disabled = True
            # self.hs['L[0][x][3][1][0][1][1]_fit_config_edge_offset_thres_slider'].disabled = True
            # self.hs['L[0][x][3][1][0][2][0]_fit_config_flt_prev_checkbox'].disabled = True
            # self.hs['L[0][x][3][1][0][0][5]_fit_config_cal_edge50_direct_checkbox'].value = False
            # self.find_edge50_pos_dir = False

            self.hs['L[0][x][3][1][0][0][0]_fit_config_fit_wl_checkbox'].value = True
            self.hs['L[0][x][3][1][0][0][1]_fit_config_fit_edge_checkbox'].value = False
            self.hs['L[0][x][3][1][0][0][3]_fit_config_fit_50%_checkbox'].value = False
            self.hs['L[0][x][3][1][0][0][2]_fit_config_flt_spec_checkbox'].value = False
            self.hs['L[0][x][3][1][0][0][4]_fit_config_cal_edge_direct_checkbox'].value = False
            self.hs['L[0][x][3][1][0][0][5]_fit_config_cal_edge50_direct_checkbox'].value = False
            self.hs['L[0][x][3][1][0][2][0]_fit_config_flt_prev_checkbox'].value = False

            self.hs['L[0][x][3][1][3][0][0]_fit_save_setting_multiselection'].value = ['wl_pos_dir']
            self.hs['L[0][x][3][1][3][0][0]_fit_save_setting_multiselection'].options = WL_SAVE_ITEM_OPTIONS
            self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].value = ['']
            self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].options = WL_SAVE_DEFAULT
            self.analysis_saving_items = deepcopy(set(WL_SAVE_DEFAULT))
            self.analysis_saving_items.remove('')
        elif self.parent_h.xanes_fit_type == 'full':
            self.parent_h.xanes_fit_edge_eng = self.hs['L[0][x][3][0][1][0][1]_fit_eng_range_edge_eng_text'].value
            self.parent_h.xanes_fit_pre_edge_e = self.hs['L[0][x][3][0][1][0][2]_fit_eng_range_pre_edge_e_text'].value
            self.parent_h.xanes_fit_post_edge_s = self.hs['L[0][x][3][0][1][0][3]_fit_eng_range_post_edge_s_text'].value
            self.parent_h.xanes_fit_wl_fit_eng_s = self.hs['L[0][x][3][0][1][1][0]_fit_eng_range_wl_fit_s_text'].value
            self.parent_h.xanes_fit_wl_fit_eng_e = self.hs['L[0][x][3][0][1][1][1]_fit_eng_range_wl_fit_e_text'].value
            self.parent_h.xanes_fit_edge_0p5_fit_s = self.hs['L[0][x][3][0][1][1][2]_fit_eng_range_edge0.5_fit_s_text'].value
            self.parent_h.xanes_fit_edge_0p5_fit_e = self.hs['L[0][x][3][0][1][1][3]_fit_eng_range_edge0.5_fit_e_text'].value
            # self.hs['L[0][x][3][1][0][0][5]_fit_config_cal_edge50_direct_checkbox'].disabled = False
            # self.hs['L[0][x][3][1][0][0][5]_fit_config_cal_edge50_direct_checkbox'].value = True
            # self.find_edge50_pos_dir = True
            self.hs['L[0][x][3][1][0][0][0]_fit_config_fit_wl_checkbox'].value = True
            self.hs['L[0][x][3][1][0][0][1]_fit_config_fit_edge_checkbox'].value = False
            self.hs['L[0][x][3][1][0][0][3]_fit_config_fit_50%_checkbox'].value = True
            self.hs['L[0][x][3][1][0][0][2]_fit_config_flt_spec_checkbox'].value = False
            self.hs['L[0][x][3][1][0][0][4]_fit_config_cal_edge_direct_checkbox'].value = False
            self.hs['L[0][x][3][1][0][0][5]_fit_config_cal_edge50_direct_checkbox'].value = False
            self.hs['L[0][x][3][1][0][2][0]_fit_config_flt_prev_checkbox'].value = False

            self.hs['L[0][x][3][1][3][0][0]_fit_save_setting_multiselection'].value = ['wl_pos_dir']
            self.hs['L[0][x][3][1][3][0][0]_fit_save_setting_multiselection'].options = FULL_SAVE_ITEM_OPTIONS
            self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].value = ['']
            self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].options = FULL_SAVE_DEFAULT
            self.analysis_saving_items = deepcopy(set(FULL_SAVE_DEFAULT))
            self.analysis_saving_items.remove('')
        if self.parent_h.xanes_fit_spectrum is None:
            self.parent_h.xanes_fit_spectrum = np.ndarray(self.parent_h.xanes_fit_data_shape[1:], dtype=np.float32)
        if self.parent_h.gui_name == 'xanes3D':
            self.parent_h.update_xanes3D_config()
            json.dump(self.parent_h.xanes_config, open(self.parent_h.xanes_save_trial_reg_config_filename, 'w'), cls=NumpyArrayEncoder)
        elif self.parent_h.gui_name == 'xanes2D':
            self.parent_h.update_xanes2D_config()
            json.dump(self.parent_h.xanes_config, open(self.parent_h.xanes_file_save_trial_reg_config_filename, 'w'), cls=NumpyArrayEncoder)


        for ii in PEAK_LINE_SHAPES:
            PEAK_FIT_PARAM_DICT[ii][1][1] = (self.parent_h.xanes_fit_wl_fit_eng_s +
                                             self.parent_h.xanes_fit_wl_fit_eng_e)/2.
        boxes = ['L[0][x][3][1][0]_fit_config_box',
                 'L[0][x][3][0][4][1]_fit_save_setting_box',
                 'L[0][x][3][2]_fit_run_box']
        enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)

        cnt = 0
        for ii in self.fit_edge_fit_func_arg_handles:
            ii.disabled = True
            ii.description = 'p'+str(cnt)
            cnt += 1
        self.hs['L[0][x][3][1][1][0][0]_fit_wl_optimizer'].value = 'scipy'
        self.hs['L[0][x][3][1][1][0][0]_fit_wl_optimizer'].value = 'numpy'
        self.hs['L[0][x][3][1][1][0][1]_fit_wl_fit_func'].options = [2, 3, 4]
        self.hs['L[0][x][3][1][1][0][1]_fit_wl_fit_func'].value = 2
        # self.hs['L[0][x][3][1][1][0][0]_fit_wl_optimizer'].value = 'scipy'
        # self.hs['L[0][x][3][1][1][0][1]_fit_wl_fit_func'].value = 'lorentzian'
        self.hs['L[0][x][3][1][2][0][0]_fit_edge_optimizer'].value = 'numpy'
        self.hs['L[0][x][3][1][2][0][1]_fit_edge_fit_func'].options = [2, 3, 4]
        self.hs['L[0][x][3][1][2][0][1]_fit_edge_fit_func'].value = 3
        self.parent_h.xanes_fit_eng_configured = True
        self.xanes_fit_eng_config = True
        self.set_save_items()
        self.boxes_logic()


    def L0_x_3_0_4_3_0_0_fit_config_fit_wl_checkbox_change(self, a):
        if a['owner'].value:
            self.fit_wl_pos = True
            boxes = ['L[0][x][3][1][1]_fit_fit_wl_param_box']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            self.hs['L[0][x][3][1][1][0][0]_fit_wl_optimizer'].value = 'numpy'
            self.hs['L[0][x][3][1][1][0][1]_fit_wl_fit_func'].options = [2, 3, 4]
            self.hs['L[0][x][3][1][1][0][1]_fit_wl_fit_func'].value = 2
            # self.hs['L[0][x][3][1][1][0][0]_fit_wl_optimizer'].value = 'scipy'
            # self.hs['L[0][x][3][1][1][0][1]_fit_wl_fit_func'].value = 'lorentzian'
            self.hs['L[0][x][3][1][3][0][0]_fit_save_setting_multiselection'].value = ['wl_pos_dir']
            self.hs['L[0][x][3][1][3][0][0]_fit_save_setting_multiselection'].options = WL_SAVE_ITEM_OPTIONS
            self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].value = ['']
            self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].options = WL_SAVE_DEFAULT
            self.analysis_saving_items = deepcopy(set(WL_SAVE_DEFAULT))
            self.analysis_saving_items.remove('')
        else:
            self.fit_wl_pos = False
            boxes = ['L[0][x][3][1][1]_fit_fit_wl_param_box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        self.set_save_items()
        self.boxes_logic()


    def L0_x_3_0_4_3_0_1_fit_config_fit_edge_checkbox_change(self, a):
        if a['owner'].value:
            self.fit_edge_pos = True
            boxes = ['L[0][x][3][1][2]_fit_fit_edge_param_box']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            self.hs['L[0][x][3][1][2][0][0]_fit_edge_optimizer'].value = 'numpy'
            self.hs['L[0][x][3][1][2][0][1]_fit_edge_fit_func'].options = [2, 3, 4]
            self.hs['L[0][x][3][1][2][0][1]_fit_edge_fit_func'].value = 3
        else:
            self.fit_edge_pos = False
            boxes = ['L[0][x][3][1][2]_fit_fit_edge_param_box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        self.set_save_items()
        self.boxes_logic()

    def L0_x_3_1_0_0_3_fit_config_fit_edge50_checkbox(self, a):
        self.fit_edge50_pos =  a['owner'].value
        self.set_save_items()
        self.boxes_logic()


    def L0_x_3_0_4_3_0_2_fit_config_flt_spec_checkbox_change(self, a):
        self.fit_use_flt_spec = a['owner'].value
        self.set_save_items()


    def L0_x_3_0_4_3_0_3_fit_config_cal_edge_direct_checkbox_change(self, a):
        self.find_edge_pos_dir = a['owner'].value
        self.set_save_items()


    def L0_x_3_0_4_3_0_3_fit_config_cal_edge50_direct_checkbox_change(self, a):
        self.find_edge50_pos_dir = a['owner'].value
        self.set_save_items()


    def L0_x_3_0_4_3_0_1_fit_config_edge_jump_thres_slider_change(self, a):
        self.fit_flt_prev_maskit = False
        self.parent_h.xanes_fit_edge_jump_thres = a['owner'].value
        self.boxes_logic()


    def L0_x_3_0_4_3_0_2_fit_config_edge_offset_thres_slider_change(self, a):
        self.fit_flt_prev_maskit = False
        self.parent_h.xanes_fit_edge_offset_thres = a['owner'].value
        self.boxes_logic()


    def L0_x_3_1_0_1_2_fit_config_bin_fac_text_chg(self, a):
        self.fit_img_bin_fac = a['owner'].value
        self.boxes_logic()


    def L0_x_3_0_4_3_2_0_fit_config_flt_prev_checkbox_change(self, a):
        if a['owner'].value:
            self.fit_flt_prev = True
            if self.parent_h.gui_name == 'xanes3D':
                self.hs['L[0][x][3][1][0][3][0]_fit_config_flt_prev_sli_slider'].max = self.parent_h.xanes_fit_data_shape[2]-1
                self.hs['L[0][x][3][1][0][3][0]_fit_config_flt_prev_sli_slider'].disabled = False
                with h5py.File(self.parent_h.xanes_save_trial_reg_filename, 'r') as f:
                    self.spec = f['/registration_results/reg_results/registered_xanes3D'][:, :, self.fit_flt_prev_sli, :]
            else:
                self.hs['L[0][x][3][1][0][3][0]_fit_config_flt_prev_sli_slider'].disabled = True
                with h5py.File(self.parent_h.xanes_save_trial_reg_filename, 'r') as f:
                    self.spec = f['/registration_results/reg_results/registered_xanes2D'][:]
        else:
            self.fit_flt_prev = False
            self.hs['L[0][x][3][1][0][3][0]_fit_config_flt_prev_sli_slider'].disabled = True
        self.boxes_logic()


    def L0_x_3_0_4_3_3_0_fit_config_flt_prev_sli_slider_change(self, a):
        self.fit_flt_prev_sli = a['owner'].value
        if self.parent_h.gui_name == 'xanes3D':
            with h5py.File(self.parent_h.xanes_save_trial_reg_filename, 'r') as f:
                self.spec = f['/registration_results/reg_results/registered_xanes3D'][:, :, self.fit_flt_prev_sli, :]
            self.fit_flt_prev_configed = False


    def L0_x_3_0_4_3_3_1_fit_config_flt_prev_calc_btn_click(self, a):
        tmp_file = os.path.join(self.global_h.tmp_dir, 'xanes2D_tmp.h5')
        if self.fit_flt_prev_configed:
            with h5py.File(tmp_file, 'r') as f:
                self.e0_idx = f['/fitting/e0_idx'][:]
                self.pre = f['/fitting/pre'][:]
                self.post = f['/fitting/post'][:]
            self.edge_jump_mask = np.squeeze((self.post[self.e0_idx]-self.pre[self.e0_idx])
                                             > self.parent_h.xanes_fit_edge_jump_thres*self.fit_flt_prev_xana.pre_edge_sd_map).astype(np.int8)
            self.fitted_edge_mask = np.any((self.post-self.pre)
                                           > self.parent_h.xanes_fit_edge_offset_thres*self.fit_flt_prev_xana.pre_edge_sd_map, axis=0).astype(np.int8)
        else:
            self.pre_es_idx = xm.index_of(self.parent_h.xanes_fit_eng_list, self.parent_h.xanes_fit_eng_list[0])
            self.pre_ee_idx = xm.index_of(self.parent_h.xanes_fit_eng_list, self.parent_h.xanes_fit_pre_edge_e)
            self.post_es_idx = xm.index_of(self.parent_h.xanes_fit_eng_list, self.parent_h.xanes_fit_post_edge_s)
            self.post_ee_idx = xm.index_of(self.parent_h.xanes_fit_eng_list, self.parent_h.xanes_fit_eng_list[-1])
            # self.fit_flt_prev_xana = xa.xanes_analysis(self.spec,
            #                                            self.parent_h.xanes_fit_eng_list,
            #                                            self.parent_h.xanes_fit_edge_eng,
            #                                            pre_ee=self.parent_h.xanes_fit_pre_edge_e,
            #                                            post_es=self.parent_h.xanes_fit_post_edge_s,
            #                                            edge_jump_threshold=self.parent_h.xanes_fit_edge_jump_thres,
            #                                            pre_edge_threshold=self.parent_h.xanes_fit_edge_offset_thres)
            # self.fit_flt_prev_xana.cal_pre_edge_sd()
            # self.fit_flt_prev_xana.cal_post_edge_sd()
            # self.fit_flt_prev_xana.cal_pre_edge_mean()
            # self.fit_flt_prev_xana.cal_post_edge_mean()
            #
            # self.e0_idx = xm.index_of(self.fit_flt_prev_xana.eng,
            #                           self.fit_flt_prev_xana.preset_edge_eng)
            # self.pre = self.fit_flt_prev_xana.cal_pre_edge_fit()
            # self.post = self.fit_flt_prev_xana.cal_post_edge_fit()
            #
            # self.edge_jump_mask = np.squeeze((self.post[self.e0_idx]-self.pre[self.e0_idx])
            #                                  > self.fit_flt_prev_xana.edge_jump_thres*self.fit_flt_prev_xana.pre_edge_sd_map).astype(np.int8)
            # self.fitted_edge_mask = np.any((self.post-self.pre)
            #                                > self.fit_flt_prev_xana.fitted_edge_thres*self.fit_flt_prev_xana.pre_edge_sd_map, axis=0).astype(np.int8)
            # self.fit_flt_prev_configed = True

            ln = 0
            code = {}
            code[ln] = f"import TXM_Sandbox.TXM_Sandbox.utils.xanes_analysis as xa";ln += 1
            code[ln] = f"import TXM_Sandbox.TXM_Sandbox.utils.xanes_math as xm";ln += 1
            code[ln] = f"import numpy as np";ln += 1
            code[ln] = f"import h5py";ln += 1
            code[ln] = f"";ln += 1
            code[ln] = f"with h5py.File('{self.parent_h.xanes_save_trial_reg_filename}', 'r') as f:";ln += 1
            code[ln] = f"    if '{self.parent_h.gui_name}' == 'xanes3D':";ln += 1
            code[ln] = f"        spec = f['/registration_results/reg_results/registered_xanes3D'][:, :, {self.fit_flt_prev_sli}, :]";ln += 1
            code[ln] = f"    else:";ln += 1
            code[ln] = f"        spec = f['/registration_results/reg_results/registered_xanes2D'][:]";ln += 1
            code[ln] = f"    eng_list = f['/registration_results/reg_results/eng_list'][:]";ln += 1
            code[ln] = f"fit_flt_prev_xana = xa.xanes_analysis(spec, eng_list,\
                                                                {self.parent_h.xanes_fit_edge_eng},\
                                                                pre_ee={self.parent_h.xanes_fit_pre_edge_e},\
                                                                post_es={self.parent_h.xanes_fit_post_edge_s},\
                                                                edge_jump_threshold={self.parent_h.xanes_fit_edge_jump_thres},\
                                                                pre_edge_threshold={self.parent_h.xanes_fit_edge_offset_thres})";ln += 1
            code[ln] = f"fit_flt_prev_xana.cal_pre_edge_sd()";ln += 1
            code[ln] = f"fit_flt_prev_xana.cal_post_edge_sd()";ln += 1
            code[ln] = f"fit_flt_prev_xana.cal_pre_edge_mean()";ln += 1
            code[ln] = f"fit_flt_prev_xana.cal_post_edge_mean()";ln += 1
            code[ln] = f"e0_idx = xm.index_of(fit_flt_prev_xana.eng, fit_flt_prev_xana.preset_edge_eng)";ln += 1
            code[ln] = f"pre = fit_flt_prev_xana.cal_pre_edge_fit()";ln += 1
            code[ln] = f"post = fit_flt_prev_xana.cal_post_edge_fit()";ln += 1
            code[ln] = f"edge_jump_mask = np.squeeze((post[e0_idx] - pre[e0_idx]) > \
                                                     fit_flt_prev_xana.edge_jump_thres * \
                                                     fit_flt_prev_xana.pre_edge_sd_map).astype(np.int8)";ln += 1
            code[ln] = f"fitted_edge_mask = np.any((post - pre) >\
                                                    fit_flt_prev_xana.fitted_edge_thres * \
                                                    fit_flt_prev_xana.pre_edge_sd_map, axis=0).astype(np.int8)";ln += 1
            code[ln] = f"with h5py.File('{tmp_file}', 'a') as f:";ln += 1
            code[ln] = f"    if '/fitting' not in f:";ln += 1
            code[ln] = f"        g0 = f.create_group('fitting')";ln += 1
            code[ln] = f"        g0.create_dataset('e0_idx', data=e0_idx)";ln += 1
            code[ln] = f"        g0.create_dataset('pre', data=pre)";ln += 1
            code[ln] = f"        g0.create_dataset('post', data=post)";ln += 1
            code[ln] = f"        g0.create_dataset('edge_jump_mask', data=edge_jump_mask)";ln += 1
            code[ln] = f"        g0.create_dataset('fitted_edge_mask', data=fitted_edge_mask)";ln += 1
            code[ln] = f"    else:";ln += 1
            code[ln] = f"        del f['/fitting']";ln += 1
            code[ln] = f"        g0 = f.create_group('fitting')";ln += 1
            code[ln] = f"        g0.create_dataset('e0_idx', data=e0_idx)";ln += 1
            code[ln] = f"        g0.create_dataset('pre', data=pre)";ln += 1
            code[ln] = f"        g0.create_dataset('post', data=post)";ln += 1
            code[ln] = f"        g0.create_dataset('edge_jump_mask', data=edge_jump_mask)";ln += 1
            code[ln] = f"        g0.create_dataset('fitted_edge_mask', data=fitted_edge_mask)";ln += 1

            gen_external_py_script(self.xanes_mask_external_command_name, code)
            sig = os.system(f"python '{self.xanes_mask_external_command_name}'")
            if sig == 0:
                print('XANES2D mask is generated.')
                with h5py.File(tmp_file, 'r') as f:
                    self.e0_idx = f['/fitting/e0_idx'][()]
                    self.pre = f['/fitting/pre'][:]
                    self.post = f['/fitting/post'][:]

                    self.edge_jump_mask = f['/fitting/edge_jump_mask'][:]
                    self.fitted_edge_mask = f['/fitting/fitted_edge_mask'][:]
                self.fit_flt_prev_configed = True
                if self.parent_h.gui_name == 'xanes3D':
                    data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_fit_jump_flt_viewer')
                    if not viewer_state:
                        fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_fit_jump_flt_viewer')
                    self.global_h.xanes2D_fiji_windows['xanes3D_fit_jump_flt_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
                                                                                                         self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(self.edge_jump_mask)),
                                                                                                     self.global_h.ImagePlusClass))
                    self.global_h.ij.py.run_macro("""setMinAndMax(0, 1)""")

                    data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_fit_thres_flt_viewer')
                    if not viewer_state:
                        fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_fit_thres_flt_viewer')
                    self.global_h.xanes2D_fiji_windows['xanes3D_fit_thres_flt_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
                                                                                                          self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(self.fitted_edge_mask)),
                                                                                                      self.global_h.ImagePlusClass))
                    self.global_h.ij.py.run_macro("""setMinAndMax(0, 1)""")
                else:
                    data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_fit_jump_flt_viewer')
                    if not viewer_state:
                        fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_fit_jump_flt_viewer')
                    self.global_h.xanes2D_fiji_windows['xanes2D_fit_jump_flt_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
                                                                                                         self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(self.edge_jump_mask)),
                                                                                                     self.global_h.ImagePlusClass))
                    self.global_h.ij.py.run_macro("""setMinAndMax(0, 1)""")

                    data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_fit_thres_flt_viewer')
                    if not viewer_state:
                        fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_fit_thres_flt_viewer')
                    self.global_h.xanes2D_fiji_windows['xanes2D_fit_thres_flt_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
                                                                                                          self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(self.fitted_edge_mask)),
                                                                                                      self.global_h.ImagePlusClass))
                    self.global_h.ij.py.run_macro("""setMinAndMax(0, 1)""")
                self.fit_flt_prev_maskit = True
            else:
                print('Something wrong when generating XANES2D masks ...')
                self.fit_flt_prev_configed = False
                self.fit_flt_prev_maskit = False
        self.boxes_logic()


    def L0_x_3_0_4_3_3_1_fit_config_flt_prev_em_it_btn_click(self, a):
        if self.parent_h.gui_name == 'xanes3D':
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_fit_maskit_viewer')
            if not viewer_state:
                fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_fit_maskit_viewer')
            self.global_h.xanes2D_fiji_windows['xanes3D_fit_maskit_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
                                                                                                 self.global_h.ij.dataset().create(
                                                                                                     self.global_h.ij.py.to_java(self.edge_jump_mask*self.spec[int(self.spec.shape[0]/2)])),
                                                                                           self.global_h.ImagePlusClass))
            self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        else:
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_fit_maskit_viewer')
            if not viewer_state:
                fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_fit_maskit_viewer')
            self.global_h.xanes2D_fiji_windows['xanes2D_fit_maskit_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
                                                                                                 self.global_h.ij.dataset().create(
                                                                                                     self.global_h.ij.py.to_java(self.edge_jump_mask*self.spec[int(self.spec.shape[0]/2)])),
                                                                                             self.global_h.ImagePlusClass))
            self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")


    def L0_x_3_0_4_3_3_1_fit_config_flt_prev_om_it_btn_click(self, a):
        if self.parent_h.gui_name == 'xanes3D':
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_fit_maskit_viewer')
            if not viewer_state:
                fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_fit_maskit_viewer')
            self.global_h.xanes2D_fiji_windows['xanes3D_fit_maskit_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
                                                                                                 self.global_h.ij.dataset().create(
                                                                                                     self.global_h.ij.py.to_java(self.fitted_edge_mask*self.spec[int(self.spec.shape[0]/2)])),
                                                                                           self.global_h.ImagePlusClass))
            self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        else:
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_fit_maskit_viewer')
            if not viewer_state:
                fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_fit_maskit_viewer')
            self.global_h.xanes2D_fiji_windows['xanes2D_fit_maskit_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
                                                                                                 self.global_h.ij.dataset().create(
                                                                                                     self.global_h.ij.py.to_java(self.fitted_edge_mask*self.spec[int(self.spec.shape[0]/2)])),
                                                                                             self.global_h.ImagePlusClass))
            self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")


    def L0_x_3_0_4_3_3_1_fit_config_flt_prev_emom_it_btn_click(self, a):
        if self.parent_h.gui_name == 'xanes3D':
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_fit_maskit_viewer')
            if not viewer_state:
                fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_fit_maskit_viewer')
            self.global_h.xanes2D_fiji_windows['xanes3D_fit_maskit_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
                                                                                                 self.global_h.ij.dataset().create(
                                                                                                     self.global_h.ij.py.to_java(self.edge_jump_mask*self.fitted_edge_mask*self.spec[int(self.spec.shape[0]/2)])),
                                                                                           self.global_h.ImagePlusClass))
            self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        else:
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_fit_maskit_viewer')
            if not viewer_state:
                fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_fit_maskit_viewer')
            self.global_h.xanes2D_fiji_windows['xanes2D_fit_maskit_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
                                                                                                 self.global_h.ij.dataset().create(
                                                                                                     self.global_h.ij.py.to_java(self.edge_jump_mask*self.fitted_edge_mask*self.spec[int(self.spec.shape[0]/2)])),
                                                                                             self.global_h.ImagePlusClass))
            self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")


    def L0_x_3_0_3_0_fit_img_use_mask_checkbox(self, a):
        pass


    def L0_x_3_0_3_1_fit_img_mask_scan_id_slider(self, a):
        pass


    def L0_x_3_0_3_1_fit_img_mask_thres_slider(self, a):
        pass


    def L0_x_3_1_0_4_0_fit_config_lcf_checkbox_chg(self, a):
        if a['owner'].value:
            self.fit_lcf = True
            if not self.fit_lcf_ref_set:
                self.hs['L[0][x][3][2][1]_fit_run_btn'].disabled = True
        else:
            self.fit_lcf = False
            self.hs['L[0][x][3][2][1]_fit_run_btn'].disabled = False
        self.set_save_items()
        self.boxes_logic()


    def L0_x_3_1_0_4_5_fit_config_lcf_constr_checkbox_chg(self, a):
        self.fit_lcf_constr = a['owner'].value


    def L0_x_3_1_0_4_1_fit_config_num_spec_text_chg(self, a):
        self.fit_lcf_ref_num = a['owner'].value
        self.fit_lcf_ref_spec = pd.DataFrame(columns=['fpath', 'fname', 'eng', 'mu'])
        self.hs['L[0][x][3][1][0][4][3]_fit_config_ref_spec_list'].options = []
        self.hs['L[0][x][3][1][0][4][2]_fit_config_sel_ref_btn'].disabled = False
        self.boxes_logic()


    def L0_x_3_1_0_4_2_fit_config_sel_ref_btn_clk(self, a):
        if len(self.hs['L[0][x][3][1][0][4][3]_fit_config_ref_spec_list'].options) < self.fit_lcf_ref_num:
            if len(a.files[0]) != 0:
                spec = np.loadtxt(a.files[0])
                if spec[:, 0].max() < 50:
                    spec[:, 0] *= 1000
                df = pd.DataFrame([[os.path.dirname(a.files[0]), os.path.basename(a.files[0]), spec[:, 0], spec[:, 1]]],
                                  columns=['fpath', 'fname', 'eng', 'mu'])
                self.fit_lcf_ref_spec = self.fit_lcf_ref_spec.append(df, ignore_index=True)
                self.hs['L[0][x][3][1][0][4][3]_fit_config_ref_spec_list'].options = self.fit_lcf_ref_spec['fname']
                update_json_content(self.global_h.GUI_cfg_file, {'xanes_ref_d':os.path.dirname(os.path.abspath(a.files[0]))})
                self.hs['L[0][x][3][1][0][4][2]_fit_config_sel_ref_btn'].initialdir = os.path.dirname(
                    os.path.abspath(a.files[0]))
        self.boxes_logic()


    def L0_x_3_1_0_4_3_fit_config_ref_spec_list_chg(self, a):
        if len(self.hs['L[0][x][3][1][0][4][3]_fit_config_ref_spec_list'].options) < self.fit_lcf_ref_num:
            self.hs['L[0][x][3][1][0][4][2]_fit_config_sel_ref_btn'].disabled = False
        self.boxes_logic()


    def L0_x_3_1_0_4_4_fit_config_ref_rm_btn_clk(self, a):
        self.fit_lcf_ref_spec = \
            self.fit_lcf_ref_spec.drop(list(self.hs['L[0][x][3][1][0][4][3]_fit_config_ref_spec_list'].index)). \
                reset_index(drop=True)
        tem = list(self.hs['L[0][x][3][1][0][4][3]_fit_config_ref_spec_list'].options)
        for ii in self.hs['L[0][x][3][1][0][4][3]_fit_config_ref_spec_list'].index:
            del tem[ii]
        self.hs['L[0][x][3][1][0][4][3]_fit_config_ref_spec_list'].options = tem
        self.boxes_logic()


    def L0_x_3_0_4_0_0_0_fit_wl_optimizer(self, a):
        self.fit_wl_optimizer = a['owner'].value
        self.boxes_logic()


    def L0_x_3_0_4_0_0_1_fit_wl_fit_func(self, a):
        self.fit_wl_fit_func = a['owner'].value
        cnt = 0
        for ii in self.fit_wl_fit_func_arg_handles:
                ii.disabled = True
                ii.description = 'p'+str(cnt)
                cnt += 1
        if self.fit_wl_optimizer == 'scipy':
            for ii in sorted(PEAK_FIT_PARAM_DICT[self.fit_wl_fit_func].keys()):
                self.fit_wl_fit_func_arg_handles[ii].disabled = False
                self.fit_wl_fit_func_arg_handles[ii].description = \
                    PEAK_FIT_PARAM_DICT[self.fit_wl_fit_func][ii][0]
                self.fit_wl_fit_func_arg_handles[ii].value = \
                    PEAK_FIT_PARAM_DICT[self.fit_wl_fit_func][ii][1]
                self.fit_wl_fit_func_arg_handles[ii].description_tooltip = \
                    PEAK_FIT_PARAM_DICT[self.fit_wl_fit_func][ii][2]
        if self.fit_wl_fit_func == "rectangle":
            self.fit_wl_fit_func_arg_handles[5].options = \
                    ['linear', 'atan', 'erf', 'logisitic']
        if self.fit_wl_fit_func == "step":
            self.fit_wl_fit_func_arg_handles[5].options = \
                    ['linear', 'atan', 'erf', 'logisitic']


    def L0_x_3_0_4_0_0_2_fit_wl_fit_bnd(self, a):
        self.fit_wl_fit_use_param_bnd = a['owner'].value
        if self.fit_wl_fit_use_param_bnd:
            for ii in self.fit_wl_fit_func_bnd_handles:
                ii.disabled = False
            for ii in ["gaussian", "lorentzian", "damped_oscillator",
                       "lognormal", "students_t", "voigt",
                       "split_lorentzian", "pvoigt", "moffat", "pearson7",
                       "breit_wigner", "dho", "expgaussian", "donaich",
                       "skewed_gaussian", "step", "skewed_voigt"]:
                PEAK_FIT_PARAM_BND_DICT[ii][1][1] = (self.parent_h.xanes_fit_wl_fit_eng_s +
                                                     self.parent_h.xanes_fit_wl_fit_eng_e)/2.
        else:
            for ii in self.fit_wl_fit_func_bnd_handles:
                ii.disabled = True


    def L0_x_3_0_4_2_0_0_fit_edge_optimizer(self, a):
        self.fit_edge_optimizer = a['owner'].value
        self.boxes_logic()


    def L0_x_3_0_4_2_0_1_fit_edge_fit_func(self, a):
        self.fit_edge_fit_func = a['owner'].value
        cnt = 0
        for ii in self.fit_edge_fit_func_arg_handles:
                ii.disabled = True
                ii.description = 'p'+str(cnt)
                cnt += 1
        if self.fit_edge_optimizer == 'scipy':
            for ii in sorted(PEAK_FIT_PARAM_DICT[self.fit_edge_fit_func].keys()):
                self.fit_edge_fit_func_arg_handles[ii].disabled = False
                self.fit_edge_fit_func_arg_handles[ii].description = \
                    PEAK_FIT_PARAM_DICT[self.fit_edge_fit_func][ii][0]
                self.fit_edge_fit_func_arg_handles[ii].value = \
                    PEAK_FIT_PARAM_DICT[self.fit_edge_fit_func][ii][1]
                self.fit_edge_fit_func_arg_handles[ii].description_tooltip = \
                    PEAK_FIT_PARAM_DICT[self.fit_edge_fit_func][ii][2]
        if self.fit_edge_fit_func == "rectangle":
            self.fit_edge_fit_func_arg_handles[5].options = \
                    ['linear', 'atan', 'erf', 'logisitic']
        if self.fit_edge_fit_func == "step":
            self.fit_edge_fit_func_arg_handles[5].options = \
                    ['linear', 'atan', 'erf', 'logisitic']


    def L0_x_3_0_4_2_0_2_fit_edge_fit_use_bnd(self, a):
        self.fit_edge_fit_use_param_bnd = a['owner'].value
        if self.fit_edge_fit_use_param_bnd:
            for ii in self.fit_edge_fit_func_bnd_handles:
                ii.disabled = False
            for ii in ["gaussian", "lorentzian", "voigt",
                       "split_lorentzian", "pvoigt", "expgaussian",
                       "skewed_gaussian", "skewed_voigt"]:
                PEAK_FIT_PARAM_BND_DICT[ii][1][1] = (self.parent_h.xanes_fit_wl_fit_eng_s +
                                                     self.parent_h.xanes_fit_wl_fit_eng_e)/2.
        else:
            for ii in self.fit_edge_fit_func_bnd_handles:
                ii.disabled = True


    def L0_x_3_0_4_1_0_2_fit_save_setting_add_btn_click(self, a):
        self.analysis_saving_items = set(self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].options)
        for ii in self.hs['L[0][x][3][1][3][0][0]_fit_save_setting_multiselection'].value:
            self.analysis_saving_items.add(ii)
        self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].value = ['']
        tem = deepcopy(self.analysis_saving_items)
        tem.add('')
        self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].options = sorted(tem)
        self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].value = list('')


    def L0_x_3_0_4_1_0_3_fit_save_setting_remove_btn_click(self, a):
        self.analysis_saving_items = set(self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].options)
        for ii in self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].value:
            if ii != '':
                self.analysis_saving_items.remove(ii)
        self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].value = ['']
        tem = deepcopy(self.analysis_saving_items)
        tem.add('')
        self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].options = sorted(tem)
        self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].value = list('')


    def L0_x_3_0_5_1_fit_run_btn(self, a):
        try:
            fiji_viewer_off(self.parent_h.global_h, viewer_name='all')
        except:
            pass
        self.analysis_saving_items = set()
        for ii in self.hs['L[0][x][3][1][3][0][1]_fit_save_setting_selection_text'].options:
            if ii != '':
                self.analysis_saving_items.add(ii)
        self.update_fit_params()

        if self.fit_wl_pos:
            wl_fvars = []
            wl_bnds = []
            wl_params = {}
            wl_params['optimizer'] = self.fit_wl_optimizer
            wl_params['ftype'] = 'wl'
            if self.parent_h.xanes_fit_type == 'full':
                wl_params['on'] = 'norm'
            else:
                wl_params['on'] = 'raw'
            if self.fit_wl_optimizer == 'scipy':
                for ii in sorted(PEAK_FIT_PARAM_DICT[self.fit_wl_fit_func].keys()):
                    wl_fvars.append(self.fit_wl_fit_func_arg_handles[ii].value)
                if self.fit_wl_fit_func in ["sine", "expsine"]:
                    wl_fvars[2] = 0
                elif self.fit_wl_fit_func == "rectangle":
                    wl_fvars[1] = wl_fvars[3] = 0
                else:
                    wl_fvars[1] = 0
                if self.fit_wl_fit_use_param_bnd :
                    for ii in sorted(PEAK_FIT_PARAM_DICT[self.fit_wl_fit_func].keys()):
                        wl_bnds.append((self.fit_wl_fit_func_bnd_handles[2*ii].value,
                                        self.fit_wl_fit_func_bnd_handles[2*ii+1].value))
                else:
                    wl_bnds = [-inf, inf]
                wl_params['eoff'] = self.hs['L[0][x][3][1][1][0][4]_fit_wl_p1'].value
                wl_params['model'] = self.fit_wl_fit_func
                wl_params['fvars'] = wl_fvars
                wl_params['bnds'] = wl_bnds
                wl_params['jac']=self.fit_wl_optimizer_arg_handles[0].value
                wl_params['method']=self.fit_wl_optimizer_arg_handles[1].value
                wl_params['ftol']=self.fit_wl_optimizer_arg_handles[2].value
                wl_params['xtol']=self.fit_wl_optimizer_arg_handles[3].value
                wl_params['gtol']=self.fit_wl_optimizer_arg_handles[4].value
            elif self.fit_wl_optimizer == 'numpy':
                wl_params['eoff'] = (self.parent_h.xanes_fit_wl_fit_eng_s+self.parent_h.xanes_fit_wl_fit_eng_e)/2.
                wl_params['model'] = 'polynd'
                wl_order = self.hs['L[0][x][3][1][1][0][1]_fit_wl_fit_func'].value
                wl_params['order'] = wl_order
                wl_params['flt_spec'] = self.fit_use_flt_spec
            wl_ufac = self.fit_wl_optimizer_arg_handles[5].value
        else:
            wl_params = {}
            wl_ufac = 20

        if self.fit_edge_pos:
            edge_fvars = []
            edge_bnds = []
            edge_params = {}

            edge_params['eoff'] = self.parent_h.xanes_fit_edge_eng
            edge_params['optimizer'] = self.fit_edge_optimizer
            edge_params['ftype'] = 'edge'
            if self.parent_h.xanes_fit_type == 'full':
                edge_params['on'] = 'norm'
            else:
                edge_params['on'] = 'raw'
            if self.fit_edge_optimizer == 'scipy':
                for ii in sorted(EDGE_FIT_PARAM_DICT[self.fit_edge_fit_func].keys()):
                    edge_fvars.append(self.fit_edge_fit_func_arg_handles[ii].value)

                if self.fit_edge_fit_func in ["sine", "expsine"]:
                    edge_fvars[2] = 0
                elif self.fit_edge_fit_func == "rectangle":
                    edge_fvars[1] = edge_fvars[3] = 0
                else:
                    edge_fvars[1] = 0

                if self.fit_wl_fit_use_param_bnd :
                    for ii in sorted(EDGE_FIT_PARAM_DICT[self.fit_wdge_fit_func].keys()):
                        edge_bnds.append((self.fit_edge_fit_func_bnd_handles[2*ii].value,
                                     self.fit_edge_fit_func_bnd_handles[2*ii+1].value))
                else:
                    edge_bnds = [-inf, inf]

                edge_params['model'] = self.fit_edge_fit_func
                edge_params['fvars'] = edge_fvars
                edge_params['bnds'] = edge_bnds
                edge_params['jac']=self.fit_edge_optimizer_arg_handles[0].value
                edge_params['method']=self.fit_edge_optimizer_arg_handles[1].value
                edge_params['ftol']=self.fit_edge_optimizer_arg_handles[2].value
                edge_params['xtol']=self.fit_edge_optimizer_arg_handles[3].value
                edge_params['gtol']=self.fit_edge_optimizer_arg_handles[4].value
            elif self.fit_edge_optimizer == 'numpy':
                edge_params['model'] = 'polynd'
                edge_order = self.hs['L[0][x][3][1][2][0][1]_fit_edge_fit_func'].value
                edge_params['order'] = edge_order
                edge_params['flt_spec'] = self.fit_use_flt_spec
            edge_ufac = self.fit_edge_optimizer_arg_handles[5].value
        else:
            edge_params = {}
            edge_ufac = 20

        if self.fit_edge_pos and self.fit_wl_pos:
            edge50_optimizer = 'both'
        elif (not self.fit_edge_pos) and self.fit_wl_pos:
            edge50_optimizer = 'wl'
        elif self.fit_edge_pos and (not self.fit_wl_pos):
            edge50_optimizer = 'edge'
        else:
            edge50_optimizer = 'none'

            # edge_fit_item = ['edge50_pos_fit',
        #                  'edge_pos_fit']
        edge_fit_item = ['edge_pos_fit']
        wl_fit_item = ['wl_pos_fit']

        if self.parent_h.gui_name == 'xanes3D':
            with h5py.File(self.parent_h.xanes_save_trial_reg_filename, 'r+') as f:
                if 'processed_XANES3D' not in f:
                    g1 = f.create_group('processed_XANES3D')
                else:
                    del f['processed_XANES3D']
                    g1 = f.create_group('processed_XANES3D')
                g11 = g1.create_group('proc_parameters')

                g11.create_dataset('element', data=str(self.parent_h.xanes_element))
                g11.create_dataset('eng_list',
                                   data=scale_eng_list(self.parent_h.xanes_fit_eng_list).astype(np.float32))
                g11.create_dataset('edge_eng',
                                   data=self.parent_h.xanes_fit_edge_eng)
                g11.create_dataset('pre_edge_e',
                                   data=self.parent_h.xanes_fit_pre_edge_e)
                g11.create_dataset('post_edge_s',
                                   data=self.parent_h.xanes_fit_post_edge_s)
                g11.create_dataset('edge_jump_threshold',
                                   data=self.parent_h.xanes_fit_edge_jump_thres)
                g11.create_dataset('edge_offset_threshold',
                                   data=self.parent_h.xanes_fit_edge_offset_thres)
                g11.create_dataset('use_mask',
                                   data=str(self.parent_h.xanes_fit_use_mask))
                g11.create_dataset('analysis_type',
                                   data=self.parent_h.xanes_fit_type)
                g11.create_dataset('data_shape',
                                   data=self.parent_h.xanes_fit_data_shape)
                g11.create_dataset('edge50_fit_s',
                                   data=self.parent_h.xanes_fit_edge_0p5_fit_s)
                g11.create_dataset('edge50_fit_e',
                                   data=self.parent_h.xanes_fit_edge_0p5_fit_e)
                g11.create_dataset('wl_fit_eng_s',
                                   data=self.parent_h.xanes_fit_wl_fit_eng_s)
                g11.create_dataset('wl_fit_eng_e',
                                   data=self.parent_h.xanes_fit_wl_fit_eng_e)
                g11.create_dataset('pre_post_edge_norm_fit_order', data=1)
                g11.create_dataset('flt_spec', data=str(self.fit_use_flt_spec))
                g11.create_dataset('bin_fact', data=int(self.fit_img_bin_fac))

                g111 = g11.create_group('wl_fit method')
                g111.create_dataset('optimizer', data=str(wl_params['optimizer']))
                if wl_params['optimizer'] == 'scipy':
                    g111.create_dataset('method', data=str(wl_params['model']))
                    g1111 = g111.create_group('params')
                    g1111.create_dataset('eng_offset', data=wl_params['eoff'])
                    g1111.create_dataset('spec', data=str(wl_params['on']))
                    g1111.create_dataset('jac', data=str(wl_params['jac']))
                    g1111.create_dataset('method', data=str(wl_params['method']))
                    g1111.create_dataset('ftol', data=wl_params['ftol'])
                    g1111.create_dataset('xtol', data=wl_params['xtol'])
                    g1111.create_dataset('gtol', data=wl_params['gtol'])
                    g1111.create_dataset('fvars_init', data=wl_fvars)
                    if wl_bnds[0] is -inf:
                        g1111.create_dataset('bnds', data=str('None'))
                    else:
                        g1111.create_dataset('bnds', data=wl_bnds)
                else:
                    g111.create_dataset('method', data=str('polynd'))
                    g1111 = g111.create_group('params')
                    g1111.create_dataset('order', data=wl_params['order'])
                    g1111.create_dataset('median_flt', data=str(wl_params['flt_spec']))
                    g1111.create_dataset('eng_offset', data=wl_params['eoff'])
                    g1111.create_dataset('spec', data=str(wl_params['on']))

                g112 = g11.create_group('edge_fit method')
                if self.fit_edge_pos:
                    g112.create_dataset('optimizer',
                                        data=str(edge_params['optimizer']))
                    if edge_params['optimizer'] == 'scipy':
                        g112.create_dataset('method',
                                            data=str(edge_params['model']))
                        g1121 = g112.create_group('params')
                        g1121.create_dataset('eng_offset', data=edge_params['eoff'])
                        g1121.create_dataset('spec', data=str(edge_params['on']))
                        g1121.create_dataset('jac', data=str(edge_params['jac']))
                        g1121.create_dataset('method', data=str(edge_params['method']))
                        g1121.create_dataset('ftol', data=edge_params['ftol'])
                        g1121.create_dataset('xtol', data=edge_params['xtol'])
                        g1121.create_dataset('gtol', data=edge_params['gtol'])
                        g1121.create_dataset('fvars_init', data=edge_fvars)
                        if edge_bnds[0] is -inf:
                            g1121.create_dataset('bnds', data=str('None'))
                        else:
                            g1121.create_dataset('bnds', data=edge_bnds)
                    else:
                        g112.create_dataset('method', data=str('polynd'))
                        g1121 = g112.create_group('params')
                        g1121.create_dataset('order', data=edge_order)
                        g1121.create_dataset('median_flt', data=str(edge_params['flt_spec']))
                        g1121.create_dataset('eng_offset', data=edge_params['eoff'])
                        g1121.create_dataset('spec', data=str(edge_params['on']))
                g113 = g11.create_group('LCF')
                g113.create_dataset('use_lcf', data=str(self.fit_lcf))
                g113.create_dataset('use_constr', data=str(self.fit_lcf_constr))
            self.fit_lcf_ref_spec.to_hdf(self.parent_h.xanes_save_trial_reg_filename,
                                         '/processed_XANES3D/proc_parameters/LCF/ref', mode='a')

            code = {}
            ln = 0
            code[ln] = f"import os, h5py"; ln += 1
            code[ln] = f"import numpy as np"; ln += 1
            code[ln] = f"import pandas as pd";ln += 1
            code[ln] = f"from scipy.ndimage import zoom";ln += 1
            code[ln] = f"import TXM_Sandbox.TXM_Sandbox.utils.xanes_math as xm"; ln += 1
            code[ln] = f"import TXM_Sandbox.TXM_Sandbox.utils.xanes_analysis as xa"; ln += 1
            code[ln] = f"from TXM_Sandbox.TXM_Sandbox.utils.misc import str2bool";ln += 1
            code[ln] = f"from copy import deepcopy"; ln += 1
            code[ln] = f"inf = np.inf"; ln += 1
            code[ln] = f""; ln += 1
            code[ln] = f"with h5py.File('{self.parent_h.xanes_save_trial_reg_filename}', 'r+') as f:"; ln += 1
            code[ln] = f"    if {self.fit_img_bin_fac} == 1:";ln += 1
            code[ln] = f"        imgs = f['/registration_results/reg_results/registered_xanes3D'][:, 0, :, :]"; ln += 1
            code[ln] = f"    else:";ln += 1
            code[ln] = f"        imgs = zoom(f['/registration_results/reg_results/registered_xanes3D'][:, 0:{self.fit_img_bin_fac}, :, :], (1, 1/{self.fit_img_bin_fac}, 1/{self.fit_img_bin_fac}, 1/{self.fit_img_bin_fac}))";ln += 1
            code[ln] = f"    xanes3D_analysis_eng_list = f['/processed_XANES3D/proc_parameters/eng_list'][:]"; ln += 1
            code[ln] = f"    xanes3D_analysis_edge_eng = f['/processed_XANES3D/proc_parameters/edge_eng'][()]"; ln += 1
            code[ln] = f"    xanes3D_analysis_pre_edge_e = f['/processed_XANES3D/proc_parameters/pre_edge_e'][()]"; ln += 1
            code[ln] = f"    xanes3D_analysis_post_edge_s = f['/processed_XANES3D/proc_parameters/post_edge_s'][()]"; ln += 1
            code[ln] = f"    xanes3D_analysis_edge_jump_thres = f['/processed_XANES3D/proc_parameters/edge_jump_threshold'][()]"; ln += 1
            code[ln] = f"    xanes3D_analysis_edge_offset_thres = f['/processed_XANES3D/proc_parameters/edge_offset_threshold'][()]"; ln += 1
            code[ln] = f"    xanes3D_analysis_use_mask = f['/processed_XANES3D/proc_parameters/use_mask'][()].decode('utf-8')"; ln += 1
            code[ln] = f"    xanes3D_analysis_type = f['/processed_XANES3D/proc_parameters/analysis_type'][()].decode('utf-8')"; ln += 1
            code[ln] = f"    xanes3D_analysis_data_shape = np.round(f['/processed_XANES3D/proc_parameters/data_shape'][:]/{self.fit_img_bin_fac})"; ln += 1
            code[ln] = f"    xanes3D_analysis_edge50_fit_s = f['/processed_XANES3D/proc_parameters/edge50_fit_s'][()]"; ln += 1
            code[ln] = f"    xanes3D_analysis_edge50_fit_e = f['/processed_XANES3D/proc_parameters/edge50_fit_e'][()]"; ln += 1
            code[ln] = f"    xanes3D_analysis_wl_fit_eng_s = f['/processed_XANES3D/proc_parameters/wl_fit_eng_s'][()]"; ln += 1
            code[ln] = f"    xanes3D_analysis_wl_fit_eng_e = f['/processed_XANES3D/proc_parameters/wl_fit_eng_e'][()]"; ln += 1
            code[ln] = f"    xanes3D_analysis_edge_fit_order = f['/processed_XANES3D/proc_parameters/pre_post_edge_norm_fit_order'][()]"; ln += 1
            code[ln] = f"    xanes3D_analysis_use_flt_spec = f['/processed_XANES3D/proc_parameters/flt_spec'][()].decode('utf-8')"; ln += 1
            code[ln] = f"    xanes3D_analysis_lcf_use = str2bool(f['/processed_XANES3D/proc_parameters/LCF/use_lcf'][()])"; ln += 1
            code[ln] = f"    xanes3D_analysis_lcf_constr = str2bool(f['/processed_XANES3D/proc_parameters/LCF/use_constr'][()])"; ln += 1
            code[ln] = f"    xanes3D_analysis_lcf_ref = pd.read_hdf('{self.parent_h.xanes_save_trial_reg_filename}', '/processed_XANES3D/proc_parameters/LCF/ref')"; ln += 1
            code[ln] = f"    xana = xa.xanes_analysis(imgs, xanes3D_analysis_eng_list, xanes3D_analysis_edge_eng, pre_ee=xanes3D_analysis_pre_edge_e, post_es=xanes3D_analysis_post_edge_s, edge_jump_threshold=xanes3D_analysis_edge_jump_thres, pre_edge_threshold=xanes3D_analysis_edge_offset_thres)"; ln += 1
            code[ln] = f"    xana.lcf_use = xanes3D_analysis_lcf_use";ln += 1
            code[ln] = f"    xana.lcf_constr_use = xanes3D_analysis_lcf_constr";ln += 1
            code[ln] = f"    xana.lcf_ref = xanes3D_analysis_lcf_ref";ln += 1
            code[ln] = f"    if '/processed_XANES3D/proc_spectrum' in f:"; ln += 1
            code[ln] = f"        del f['/processed_XANES3D/proc_spectrum']"; ln += 1
            code[ln] = f"        g12 = f.create_group('/processed_XANES3D/proc_spectrum')"; ln += 1
            code[ln] = f"    else:"; ln += 1
            code[ln] = f"        g12 = f.create_group('/processed_XANES3D/proc_spectrum')"; ln += 1
            code[ln] = f""; ln += 1
            code[ln] = f"    if xanes3D_analysis_type == 'wl':"; ln += 1
            code[ln] = "        _g12 = {}"; ln += 1
            code[ln] = f"        for jj in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"            if jj == 'wl_fit_coef':"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='wl_fit_coef')}, *xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = f"            elif jj == 'edge_fit_coef':"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='edge_fit_coef')}, *xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = f"            elif jj in ['pre_edge_fit_coef', 'post_edge_fit_coef']:"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='pre_edge_fit_coef')}, *xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = f"            else:"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = f"        if {self.fit_img_bin_fac} == 1:";ln += 1
            code[ln] = f"            for ii in range(xanes3D_analysis_data_shape[1]):"; ln += 1
            code[ln] = f"                xana.spec[:] = f['/registration_results/reg_results/registered_xanes3D'][:, ii, :, :]"; ln += 1
            code[ln] = f"                if np.any(np.array([ii in {self.analysis_saving_items} for ii in {edge_fit_item}])):"; ln += 1
            code[ln] = f"                    xana.fit_edge(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, **{edge_params})"; ln += 1
            code[ln] = f"                if np.any(np.array([ii in {self.analysis_saving_items} for ii in {wl_fit_item}])):"; ln += 1
            code[ln] = f"                    xana.fit_edge(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, **{wl_params})"; ln += 1
            code[ln] = f"                if ('edge_pos_fit' in {self.analysis_saving_items}):"; ln += 1
            code[ln] = f"                    xana.cal_edge_deriv(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, optimizer='{edge_params['optimizer']}',ufac={edge_ufac})"; ln += 1
            code[ln] = f"                if 'edge_pos_dir' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.find_edge()"; ln += 1
            code[ln] = f"                if 'wl_pos_fit' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.find_wl(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='{wl_params['optimizer']}',ufac={wl_ufac})"; ln += 1
            code[ln] = f"                if 'wl_pos_dir' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.find_wl(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='direct',ufac={wl_ufac})"; ln += 1
            code[ln] = f"                for jj in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    if jj == 'wl_pos_fit':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.wl_pos_fit)[:]"; ln += 1
            code[ln] = f"                    if jj == 'wl_pos_dir':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.wl_pos_dir)[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge_pos_fit':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.edge_pos_fit)[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge_pos_dir':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.edge_pos_dir)[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge_fit_coef':"; ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.model['edge']['fit_rlt'][0].reshape(xana.model['edge']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge_fit_err':";ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.model['edge']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]";ln += 1
            code[ln] = f"                    if jj == 'wl_fit_coef':"; ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][0].reshape(xana.model['wl']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"                    if jj == 'wl_fit_err':";ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]";ln += 1
            code[ln] = f"                print(ii)"; ln += 1
            code[ln] = f"        else:";ln += 1
            code[ln] = f"            for ii in range(0, xanes3D_analysis_data_shape[1], {self.fit_img_bin_fac}):"; ln += 1
            code[ln] = f"                xana.spec[:] = zoom(f['/registration_results/reg_results/registered_xanes3D'][:, ii:ii+{self.fit_img_bin_fac}, :, :], (1, 1/{self.fit_img_bin_fac}, 1/{self.fit_img_bin_fac}, 1/{self.fit_img_bin_fac}))"; ln += 1
            code[ln] = f"                if np.any(np.array([ii in {self.analysis_saving_items} for ii in {edge_fit_item}])):"; ln += 1
            code[ln] = f"                    xana.fit_edge(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, **{edge_params})"; ln += 1
            code[ln] = f"                if np.any(np.array([ii in {self.analysis_saving_items} for ii in {wl_fit_item}])):"; ln += 1
            code[ln] = f"                    xana.fit_edge(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, **{wl_params})"; ln += 1
            code[ln] = f"                if ('edge_pos_fit' in {self.analysis_saving_items}):"; ln += 1
            code[ln] = f"                    xana.cal_edge_deriv(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, optimizer='{edge_params['optimizer']}',ufac={edge_ufac})"; ln += 1
            code[ln] = f"                if 'edge_pos_dir' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.find_edge()"; ln += 1
            code[ln] = f"                if 'wl_pos_fit' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.find_wl(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='{wl_params['optimizer']}',ufac={wl_ufac})"; ln += 1
            code[ln] = f"                if 'wl_pos_dir' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.find_wl(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='direct',ufac={wl_ufac})"; ln += 1
            code[ln] = f"                for jj in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    if jj == 'wl_pos_fit':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.wl_pos_fit)[:]"; ln += 1
            code[ln] = f"                    if jj == 'wl_pos_dir':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.wl_pos_dir)[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge_pos_fit':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.edge_pos_fit)[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge_pos_dir':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.edge_pos_dir)[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge_fit_coef':"; ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.model['edge']['fit_rlt'][0].reshape(xana.model['edge']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge_fit_err':";ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.model['edge']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]";ln += 1
            code[ln] = f"                    if jj == 'wl_fit_coef':"; ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][0].reshape(xana.model['wl']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"                    if jj == 'wl_fit_err':";ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]";ln += 1
            code[ln] = f"                print(ii)"; ln += 1
            code[ln] = f"    elif xanes3D_analysis_type == 'full':"; ln += 1
            code[ln] = "        _g12 = {}"; ln += 1
            code[ln] = f"        for jj in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"            if jj == 'wl_fit_coef':"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='wl_fit_coef')}, *xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = f"            elif jj == 'edge_fit_coef':"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='edge_fit_coef')}, *xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = f"            elif jj in ['pre_edge_fit_coef', 'post_edge_fit_coef']:"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='pre_edge_fit_coef')}, *xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = f"            elif jj == 'norm_spec':"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=(xanes3D_analysis_data_shape), dtype=np.float32)"; ln += 1
            code[ln] = f"            elif jj == 'lcf_fit':"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=([xana.lcf_ref.index.size, *xanes3D_analysis_data_shape[1:]]), dtype=np.float32)"; ln += 1
            code[ln] = f"            else:"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = f"        if {self.fit_img_bin_fac} == 1:";ln += 1
            code[ln] = f"            for ii in range(xanes3D_analysis_data_shape[1]):"; ln += 1
            code[ln] = f"                xana.spec[:] = f['/registration_results/reg_results/registered_xanes3D'][:, ii, :, :]"; ln += 1
            code[ln] = f"                xana.cal_pre_edge_sd()"; ln += 1
            code[ln] = f"                xana.cal_post_edge_sd()"; ln += 1
            code[ln] = f"                xana.cal_pre_edge_mean()"; ln += 1
            code[ln] = f"                xana.cal_post_edge_mean()"; ln += 1
            code[ln] = f"                xana.full_spec_preprocess(xanes3D_analysis_edge_eng, order=xanes3D_analysis_edge_fit_order, save_pre_post=True)"; ln += 1
            code[ln] = f"                if np.any(np.array([ii in {self.analysis_saving_items} for ii in {edge_fit_item}])):"; ln += 1
            code[ln] = f"                    xana.fit_edge(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, **{edge_params})"; ln += 1
            code[ln] = f"                if np.any(np.array([ii in {self.analysis_saving_items} for ii in {wl_fit_item}])):"; ln += 1
            code[ln] = f"                    xana.fit_edge(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, **{wl_params})"; ln += 1
            code[ln] = f"                if ('edge50_pos_fit' in {self.analysis_saving_items}):";ln += 1
            code[ln] = f"                    xana.find_edge_50(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='{edge50_optimizer}', ufac={edge_ufac})";ln += 1
            code[ln] = f"                if 'edge50_pos_dir' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.find_edge_50(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='none', ufac={edge_ufac})";ln += 1
            code[ln] = f"                if ('edge_pos_fit' in {self.analysis_saving_items}):"; ln += 1
            code[ln] = f"                    xana.find_edge_deriv(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, optimizer='model',ufac={edge_ufac})"; ln += 1
            code[ln] = f"                if 'edge_pos_dir' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.find_edge_deriv(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, optimizer='direct',ufac={edge_ufac})"; ln += 1
            code[ln] = f"                if 'wl_pos_fit' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.find_wl(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='model',ufac={wl_ufac})"; ln += 1
            code[ln] = f"                if 'wl_pos_dir' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.find_wl(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='direct',ufac={wl_ufac})"; ln += 1
            code[ln] = f"                if 'wl_peak_height_dir' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.cal_drt_wl_peak_hgt(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e)"; ln += 1
            code[ln] = f"                if ('centroid_of_eng' or 'centroid_of_eng_relative_to_wl' or 'weighted_attenuation' or 'weighted_eng') in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.cal_wgt_eng(xanes3D_analysis_pre_edge_e)"; ln += 1
            code[ln] = f"                if 'lcf_fit' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.interp_ref_spec()"; ln += 1
            code[ln] = f"                    xana.lcf()"; ln += 1
            code[ln] = f""; ln += 1
            code[ln] = f"                for jj in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    if jj == 'wl_pos_fit':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.wl_pos_fit)[:]"; ln += 1
            code[ln] = f"                    if jj == 'wl_pos_dir':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.wl_pos_dir)[:]"; ln += 1
            code[ln] = f"                    if jj == 'wl_peak_height_dir':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.direct_wl_ph)[:]"; ln += 1
            code[ln] = f"                    if jj == 'centroid_of_eng':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.centroid_of_eng)[:]"; ln += 1
            code[ln] = f"                    if jj == 'centroid_of_eng_relative_to_wl':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.centroid_of_eng_rel_wl)[:]"; ln += 1
            code[ln] = f"                    if jj == 'weighted_attenuation':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.weighted_atten)[:]"; ln += 1
            code[ln] = f"                    if jj == 'weighted_eng':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.weighted_eng)[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge50_pos_fit':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.edge50_pos_fit)[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge50_pos_dir':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.edge50_pos_fit_none)[:]"; ln += 1
            code[ln] = f"                    if (jj == 'edge_pos_fit'):"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.edge_pos_fit)[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge_pos_dir':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.edge_pos_dir)[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge_jump_filter':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.edge_jump_mask)[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge_offset_filter':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.fitted_edge_mask)[:]"; ln += 1
            code[ln] = f"                    if jj == 'pre_edge_sd':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.pre_edge_sd_map)[:]"; ln += 1
            code[ln] = f"                    if jj == 'post_edge_sd':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.post_edge_sd_map)[:]"; ln += 1
            code[ln] = f"                    if jj == 'pre_edge_mean':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.pre_edge_mean_map)[:]"; ln += 1
            code[ln] = f"                    if jj == 'post_edge_mean':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.post_edge_mean_map)[:]"; ln += 1
            code[ln] = f"                    if jj  == 'pre_edge_fit_coef':"; ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.pre_edge_fit_rlt[0].reshape(xana.pre_edge_fit_rlt[0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"                    if jj == 'post_edge_fit_coef':"; ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.post_edge_fit_rlt[0].reshape(xana.post_edge_fit_rlt[0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge_fit_coef':"; ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.model['edge']['fit_rlt'][0].reshape(xana.model['edge']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge_fit_err':";ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.model['edge']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]";ln += 1
            code[ln] = f"                    if jj == 'wl_fit_coef':"; ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][0].reshape(xana.model['wl']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"                    if jj == 'wl_fit_err':";ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]";ln += 1
            code[ln] = f"                    if jj == 'norm_spec':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.norm_spec)[:]"; ln += 1
            code[ln] = f"                    if jj == 'lcf_fit':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.lcf_fit)[:]"; ln += 1
            code[ln] = f"                    if jj == 'lcf_fit_err':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.lcf_fit_err)[:]"; ln += 1
            code[ln] = f"                print(ii)"; ln += 1
            code[ln] = f"        else:";ln += 1
            code[ln] = f"            for ii in range(0, xanes3D_analysis_data_shape[1], {self.fit_img_bin_fac}):"; ln += 1
            code[ln] = f"                xana.spec[:] = zoom(f['/registration_results/reg_results/registered_xanes3D'][:, ii:ii+{self.fit_img_bin_fac}, :, :], (1, 1/{self.fit_img_bin_fac}, 1/{self.fit_img_bin_fac}, 1/{self.fit_img_bin_fac}))"; ln += 1
            code[ln] = f"                xana.cal_pre_edge_sd()"; ln += 1
            code[ln] = f"                xana.cal_post_edge_sd()"; ln += 1
            code[ln] = f"                xana.cal_pre_edge_mean()"; ln += 1
            code[ln] = f"                xana.cal_post_edge_mean()"; ln += 1
            code[ln] = f"                xana.full_spec_preprocess(xanes3D_analysis_edge_eng, order=xanes3D_analysis_edge_fit_order, save_pre_post=True)"; ln += 1
            code[ln] = f"                if np.any(np.array([ii in {self.analysis_saving_items} for ii in {edge_fit_item}])):"; ln += 1
            code[ln] = f"                    xana.fit_edge(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, **{edge_params})"; ln += 1
            code[ln] = f"                if np.any(np.array([ii in {self.analysis_saving_items} for ii in {wl_fit_item}])):"; ln += 1
            code[ln] = f"                    xana.fit_edge(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, **{wl_params})"; ln += 1
            code[ln] = f"                if ('edge50_pos_fit' in {self.analysis_saving_items}):";ln += 1
            code[ln] = f"                    xana.find_edge_50(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='{edge50_optimizer}', ufac={edge_ufac})";ln += 1
            code[ln] = f"                if ('edge50_pos_dir' in {self.analysis_saving_items}):"; ln += 1
            code[ln] = f"                    xana.find_edge_50(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='none', ufac={edge_ufac})";ln += 1
            code[ln] = f"                if ('edge_pos_fit' in {self.analysis_saving_items}):"; ln += 1
            code[ln] = f"                    xana.find_edge_deriv(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, optimizer='model',ufac={edge_ufac})"; ln += 1
            code[ln] = f"                if 'edge_pos_dir' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.find_edge_deriv(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, optimizer='direct',ufac={edge_ufac})"; ln += 1
            code[ln] = f"                if 'wl_pos_fit' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.find_wl(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='model',ufac={wl_ufac})"; ln += 1
            code[ln] = f"                if 'wl_pos_dir' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.find_wl(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='direct',ufac={wl_ufac})"; ln += 1
            code[ln] = f"                if 'wl_peak_height_dir' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.cal_drt_wl_peak_hgt(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e)"; ln += 1
            code[ln] = f"                if ('centroid_of_eng' or 'centroid_of_eng_relative_to_wl' or 'weighted_attenuation' or 'weighted_eng') in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.cal_wgt_eng(xanes3D_analysis_pre_edge_e)"; ln += 1
            code[ln] = f"                if 'lcf_fit' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    xana.interp_ref_spec()"; ln += 1
            code[ln] = f"                    xana.lcf()"; ln += 1
            code[ln] = f""; ln += 1
            code[ln] = f"                for jj in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"                    if jj == 'wl_pos_fit':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.wl_pos_fit)[:]"; ln += 1
            code[ln] = f"                    if jj == 'wl_pos_dir':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.wl_pos_dir)[:]"; ln += 1
            code[ln] = f"                    if jj == 'wl_peak_height_dir':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.direct_wl_ph)[:]"; ln += 1
            code[ln] = f"                    if jj == 'centroid_of_eng':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.centroid_of_eng)[:]"; ln += 1
            code[ln] = f"                    if jj == 'centroid_of_eng_relative_to_wl':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.centroid_of_eng_rel_wl)[:]"; ln += 1
            code[ln] = f"                    if jj == 'weighted_attenuation':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.weighted_atten)[:]"; ln += 1
            code[ln] = f"                    if jj == 'weighted_eng':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.weighted_eng)[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge50_pos_fit':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.edge50_pos_fit)[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge50_pos_dir':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.edge50_pos_fit_none)[:]"; ln += 1
            code[ln] = f"                    if (jj == 'edge_pos_fit') and {self.fit_edge_pos}:"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.edge_pos_fit)[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge_pos_dir':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.edge_pos_dir)[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge_jump_filter':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.edge_jump_mask)[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge_offset_filter':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.fitted_edge_mask)[:]"; ln += 1
            code[ln] = f"                    if jj == 'pre_edge_sd':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.pre_edge_sd_map)[:]"; ln += 1
            code[ln] = f"                    if jj == 'post_edge_sd':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.post_edge_sd_map)[:]"; ln += 1
            code[ln] = f"                    if jj == 'pre_edge_mean':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.pre_edge_mean_map)[:]"; ln += 1
            code[ln] = f"                    if jj == 'post_edge_mean':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.post_edge_mean_map)[:]"; ln += 1
            code[ln] = f"                    if jj  == 'pre_edge_fit_coef':"; ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.pre_edge_fit_rlt[0].reshape(xana.pre_edge_fit_rlt[0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"                    if jj == 'post_edge_fit_coef':"; ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.post_edge_fit_rlt[0].reshape(xana.post_edge_fit_rlt[0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge_fit_coef':"; ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.model['edge']['fit_rlt'][0].reshape(xana.model['edge']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"                    if jj == 'edge_fit_err':";ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.model['edge']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]";ln += 1
            code[ln] = f"                    if jj == 'wl_fit_coef':"; ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][0].reshape(xana.model['wl']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"                    if jj == 'wl_fit_err':";ln += 1
            code[ln] = f"                        _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]";ln += 1
            code[ln] = f"                    if jj == 'norm_spec':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.norm_spec)[:]"; ln += 1
            code[ln] = f"                    if jj == 'lcf_fit':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.lcf_fit)[:]"; ln += 1
            code[ln] = f"                    if jj == 'lcf_fit_err':"; ln += 1
            code[ln] = f"                        _g12[jj][ii] = np.float32(xana.lcf_fit_err)[:]"; ln += 1
            code[ln] = f"                print(ii)"; ln += 1
            code[ln] = f"print('xanes3D analysis is done!')"; ln += 1

            gen_external_py_script(self.parent_h.xanes_fit_external_command_name, code)
            sig = os.system(f'python {self.parent_h.xanes_fit_external_command_name}')
            if sig == 0:
                self.hs['L[0][x][3][2][0]_fit_run_text'].value = 'XANES3D analysis is done ...'
            else:
                self.hs['L[0][x][3][2][0]_fit_run_text'].value = 'something wrong in analysis ...'
            self.parent_h.update_xanes3D_config()
        elif self.parent_h.gui_name == 'xanes2D':
            with h5py.File(self.parent_h.xanes_save_trial_reg_filename, 'r+') as f:
                if 'processed_XANES2D' not in f:
                    g1 = f.create_group('processed_XANES2D')
                else:
                    del f['processed_XANES2D']
                    g1 = f.create_group('processed_XANES2D')
                g11 = g1.create_group('proc_parameters')
                g11.create_dataset('eng_list', data=self.parent_h.xanes_fit_eng_list)
                g11.create_dataset('edge_eng', data=self.parent_h.xanes_fit_edge_eng)
                g11.create_dataset('pre_edge_e', data=self.parent_h.xanes_fit_pre_edge_e)
                g11.create_dataset('post_edge_s', data=self.parent_h.xanes_fit_post_edge_s)
                g11.create_dataset('edge_jump_threshold', data=self.parent_h.xanes_fit_edge_jump_thres)
                g11.create_dataset('edge_offset_threshold', data=self.parent_h.xanes_fit_edge_offset_thres)
                g11.create_dataset('use_mask', data=str(self.parent_h.xanes_fit_use_mask))
                g11.create_dataset('analysis_type', data=self.parent_h.xanes_fit_type)
                g11.create_dataset('data_shape', data=self.parent_h.xanes_fit_data_shape)
                g11.create_dataset('edge50_fit_s', data=self.parent_h.xanes_fit_edge_0p5_fit_s)
                g11.create_dataset('edge50_fit_e', data=self.parent_h.xanes_fit_edge_0p5_fit_e)
                g11.create_dataset('wl_fit_eng_s', data=self.parent_h.xanes_fit_wl_fit_eng_s)
                g11.create_dataset('wl_fit_eng_e', data=self.parent_h.xanes_fit_wl_fit_eng_e)
                g11.create_dataset('pre_post_edge_norm_fit_order', data=1)
                g11.create_dataset('flt_spec', data=str(self.fit_use_flt_spec))
                g11.create_dataset('bin_fact', data=int(self.fit_img_bin_fac))

                g111 = g11.create_group('wl_fit method')
                g111.create_dataset('optimizer', data=str(self.fit_wl_optimizer))
                if self.fit_wl_optimizer == 'scipy':
                    g111.create_dataset('method', data=str(self.fit_wl_fit_func))
                    g1111 = g111.create_group('params')
                    g1111.create_dataset('jac', data=str(wl_params['jac']))
                    g1111.create_dataset('method', data=str(wl_params['method']))
                    g1111.create_dataset('ftol', data=wl_params['ftol'])
                    g1111.create_dataset('xtol', data=wl_params['xtol'])
                    g1111.create_dataset('gtol', data=wl_params['gtol'])
                    g1111.create_dataset('fvars_init', data=wl_fvars)
                    if wl_bnds[0] is -inf:
                        g1111.create_dataset('bnds', data=str('None'))
                    else:
                        g1111.create_dataset('bnds', data=wl_bnds)
                else:
                    g111.create_dataset('method', data=str('polynimial'))
                    g1111 = g111.create_group('params')
                    g1111.create_dataset('order', data=wl_order)

                g112 = g11.create_group('edge_fit method')
                if self.fit_edge_pos:
                    g112.create_dataset('optimizer',
                                        data=str(self.fit_edge_optimizer))
                    if self.fit_edge_optimizer == 'scipy':
                        g112.create_dataset('method',
                                            data=str(self.fit_edge_fit_func))
                        g1121 = g112.create_group('params')
                        g1121.create_dataset('jac', data=str(edge_params['jac']))
                        g1121.create_dataset('method', data=str(edge_params['method']))
                        g1121.create_dataset('ftol', data=edge_params['ftol'])
                        g1121.create_dataset('xtol', data=edge_params['xtol'])
                        g1121.create_dataset('gtol', data=edge_params['gtol'])
                        g1121.create_dataset('fvars_init', data=edge_fvars)
                        if edge_bnds[0] is -inf:
                            g1121.create_dataset('bnds', data=str('None'))
                        else:
                            g1121.create_dataset('bnds', data=edge_bnds)
                    else:
                        g112.create_dataset('method', data=str('polynimial'))
                        g1121 = g112.create_group('params')
                        g1121.create_dataset('order', data=edge_order)
                g113 = g11.create_group('LCF')
                g113.create_dataset('use_lcf', data=str(self.fit_lcf))
                g113.create_dataset('use_constr', data=str(self.fit_lcf_constr))
            self.fit_lcf_ref_spec.to_hdf(self.parent_h.xanes_save_trial_reg_filename,
                                         '/processed_XANES2D/proc_parameters/LCF/ref', mode='a')
            code = {}
            ln = 0
            code[ln] = f"import os, h5py"; ln += 1
            code[ln] = f"import numpy as np"; ln += 1
            code[ln] = f"import pandas as pd";ln += 1
            code[ln] = f"from scipy.ndimage import zoom";ln += 1
            code[ln] = f"import TXM_Sandbox.TXM_Sandbox.utils.xanes_math as xm"; ln += 1
            code[ln] = f"import TXM_Sandbox.TXM_Sandbox.utils.xanes_analysis as xa"; ln += 1
            code[ln] = f"from TXM_Sandbox.TXM_Sandbox.utils.misc import str2bool";ln += 1
            code[ln] = f"from copy import deepcopy"; ln += 1
            code[ln] = f"inf = np.inf"; ln += 1
            # code[ln] = f"os.system('taskset -p 0xff %d' % os.getpid())"; ln += 1
            code[ln] = f""; ln += 1
            code[ln] = f"with h5py.File('{self.parent_h.xanes_save_trial_reg_filename}', 'r+') as f:"; ln += 1
            code[ln] = f"    if {self.fit_img_bin_fac} == 1:";ln += 1
            code[ln] = f"        imgs = f['/registration_results/reg_results/registered_xanes2D'][:]"; ln += 1
            code[ln] = f"    else:";ln += 1
            code[ln] = f"        imgs = zoom(f['/registration_results/reg_results/registered_xanes2D'][:], (1, {1/self.fit_img_bin_fac}, {1/self.fit_img_bin_fac}))";ln += 1
            code[ln] = f"    xanes2D_analysis_eng_list = f['/processed_XANES2D/proc_parameters/eng_list'][:]"; ln += 1
            code[ln] = f"    xanes2D_analysis_edge_eng = f['/processed_XANES2D/proc_parameters/edge_eng'][()]"; ln += 1
            code[ln] = f"    xanes2D_analysis_pre_edge_e = f['/processed_XANES2D/proc_parameters/pre_edge_e'][()]"; ln += 1
            code[ln] = f"    xanes2D_analysis_post_edge_s = f['/processed_XANES2D/proc_parameters/post_edge_s'][()]"; ln += 1
            code[ln] = f"    xanes2D_analysis_edge_jump_thres = f['/processed_XANES2D/proc_parameters/edge_jump_threshold'][()]"; ln += 1
            code[ln] = f"    xanes2D_analysis_edge_offset_thres = f['/processed_XANES2D/proc_parameters/edge_offset_threshold'][()]"; ln += 1
            code[ln] = f"    xanes2D_analysis_use_mask = f['/processed_XANES2D/proc_parameters/use_mask'][()].decode('utf-8')"; ln += 1
            code[ln] = f"    xanes2D_analysis_type = f['/processed_XANES2D/proc_parameters/analysis_type'][()].decode('utf-8')"; ln += 1
            code[ln] = f"    xanes2D_analysis_data_shape = imgs.shape"; ln += 1
            code[ln] = f"    xanes2D_analysis_edge50_fit_s = f['/processed_XANES2D/proc_parameters/edge50_fit_s'][()]"; ln += 1
            code[ln] = f"    xanes2D_analysis_edge50_fit_e = f['/processed_XANES2D/proc_parameters/edge50_fit_e'][()]"; ln += 1
            code[ln] = f"    xanes2D_analysis_wl_fit_eng_s = f['/processed_XANES2D/proc_parameters/wl_fit_eng_s'][()]"; ln += 1
            code[ln] = f"    xanes2D_analysis_wl_fit_eng_e = f['/processed_XANES2D/proc_parameters/wl_fit_eng_e'][()]"; ln += 1
            code[ln] = f"    xanes2D_analysis_use_flt_spec = f['/processed_XANES2D/proc_parameters/flt_spec'][()].decode('utf-8')"; ln += 1
            code[ln] = f"    xanes2D_analysis_edge_fit_order = f['/processed_XANES2D/proc_parameters/pre_post_edge_norm_fit_order'][()]"; ln += 1
            code[ln] = f"    xanes2D_analysis_lcf_use = str2bool(f['/processed_XANES2D/proc_parameters/LCF/use_lcf'][()])"; ln += 1
            code[ln] = f"    xanes2D_analysis_lcf_constr = str2bool(f['/processed_XANES2D/proc_parameters/LCF/use_constr'][()])"; ln += 1
            code[ln] = f"    xanes2D_analysis_lcf_ref = pd.read_hdf('{self.parent_h.xanes_save_trial_reg_filename}', '/processed_XANES2D/proc_parameters/LCF/ref')"; ln += 1
            code[ln] = f"    xana = xa.xanes_analysis(imgs, xanes2D_analysis_eng_list, xanes2D_analysis_edge_eng, pre_ee=xanes2D_analysis_pre_edge_e, post_es=xanes2D_analysis_post_edge_s, edge_jump_threshold=xanes2D_analysis_edge_jump_thres, pre_edge_threshold=xanes2D_analysis_edge_offset_thres)"; ln += 1
            code[ln] = f"    xana.lcf_use = xanes2D_analysis_lcf_use";ln += 1
            code[ln] = f"    xana.lcf_constr_use = xanes2D_analysis_lcf_constr";ln += 1
            code[ln] = f"    xana.lcf_ref = xanes2D_analysis_lcf_ref";ln += 1
            code[ln] = f"    if '/processed_XANES2D/proc_spectrum' in f:"; ln += 1
            code[ln] = f"        del f['/processed_XANES2D/proc_spectrum']"; ln += 1
            code[ln] = f"        g12 = f.create_group('/processed_XANES2D/proc_spectrum')"; ln += 1
            code[ln] = f"    else:"; ln += 1
            code[ln] = f"        g12 = f.create_group('/processed_XANES2D/proc_spectrum')"; ln += 1
            code[ln] = f""; ln += 1
            code[ln] = f"    if xanes2D_analysis_type == 'wl':"; ln += 1
            code[ln] = "        _g12 = {}"; ln += 1
            code[ln] = f"        for jj in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"            if jj == 'wl_fit_coef':"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='wl_fit_coef')}, *xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = f"            elif jj == 'edge_fit_coef':"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='edge_fit_coef')}, *xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = f"            else:"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = f"        if np.any(np.array([ii in {self.analysis_saving_items} for ii in {edge_fit_item}])):"; ln += 1
            code[ln] = f"            xana.fit_edge(xanes2D_analysis_edge50_fit_s, xanes2D_analysis_edge50_fit_e, **{edge_params})"; ln += 1
            code[ln] = f"        if np.any(np.array([ii in {self.analysis_saving_items} for ii in {wl_fit_item}])):"; ln += 1
            code[ln] = f"            xana.fit_edge(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, **{wl_params})"; ln += 1
            code[ln] = f"        if ('edge_pos_fit' in {self.analysis_saving_items}):"; ln += 1
            code[ln] = f"            xana.cal_edge_deriv(xanes2D_analysis_edge50_fit_s, xanes2D_analysis_edge50_fit_e, optimizer='model',ufac={edge_ufac})"; ln += 1
            code[ln] = f"        if 'edge_pos_dir' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"            xana.cal_edge_deriv(xanes2D_analysis_edge50_fit_s, xanes2D_analysis_edge50_fit_e, optimizer='direct',ufac={edge_ufac})"; ln += 1
            code[ln] = f"        if 'wl_pos_fit' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"            xana.find_wl(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, optimizer='model',ufac={wl_ufac})"; ln += 1
            code[ln] = f"        if 'wl_pos_dir' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"            xana.find_wl(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, optimizer='direct',ufac={wl_ufac})"; ln += 1
            code[ln] = f"        for jj in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"            if jj == 'wl_pos_fit':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.wl_pos_fit)[:]"; ln += 1
            code[ln] = f"            if jj == 'wl_pos_dir':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.wl_pos_dir)[:]"; ln += 1
            code[ln] = f"            if jj == 'edge_pos_fit':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.edge_pos_fit)[:]"; ln += 1
            code[ln] = f"            if jj == 'edge_pos_dir':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.edge_pos_dir)[:]"; ln += 1
            code[ln] = f"            if jj  == 'pre_edge_fit_coef':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.pre_edge_fit_rlt[0].reshape(xana.pre_edge_fit_rlt[0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"            if jj == 'post_edge_fit_coef':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.post_edge_fit_rlt[0].reshape(xana.post_edge_fit_rlt[0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"            if jj == 'edge_fit_coef':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.model['edge']['fit_rlt'][0].reshape(xana.model['edge']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"            if jj == 'edge_fit_err':";ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.model['edge']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]";ln += 1
            code[ln] = f"            if jj == 'wl_fit_coef':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][0].reshape(xana.model['wl']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"            if jj == 'wl_fit_err':";ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]";ln += 1
            code[ln] = f""; ln += 1
            code[ln] = f"    elif xanes2D_analysis_type == 'full':"; ln += 1
            code[ln] = "        _g12 = {}"; ln += 1
            code[ln] = f"        for jj in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"            if jj == 'wl_fit_coef':"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='wl_fit_coef')}, *xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = f"            elif jj == 'edge_fit_coef':"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='edge_fit_coef')}, *xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = f"            elif jj in ['pre_edge_fit_coef', 'post_edge_fit_coef']:"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='pre_edge_fit_coef')}, *xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = f"            elif jj == 'norm_spec':"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=(xanes2D_analysis_data_shape), dtype=np.float32)"; ln += 1
            code[ln] = f"            elif jj == 'lcf_fit':"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=([xana.lcf_ref.index.size, *xanes2D_analysis_data_shape[1:]]), dtype=np.float32)"; ln += 1
            code[ln] = f"            else:"; ln += 1
            code[ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = f"        xana.cal_pre_edge_sd()"; ln += 1
            code[ln] = f"        xana.cal_post_edge_sd()"; ln += 1
            code[ln] = f"        xana.cal_pre_edge_mean()"; ln += 1
            code[ln] = f"        xana.cal_post_edge_mean()"; ln += 1
            code[ln] = f"        xana.full_spec_preprocess(xanes2D_analysis_edge_eng, order=xanes2D_analysis_edge_fit_order, save_pre_post=True)"; ln += 1
            code[ln] = f"        if np.any(np.array([ii in {self.analysis_saving_items} for ii in {edge_fit_item}])):"; ln += 1
            code[ln] = f"            xana.fit_edge(xanes2D_analysis_edge50_fit_s, xanes2D_analysis_edge50_fit_e, **{edge_params})"; ln += 1
            code[ln] = f"        if np.any(np.array([ii in {self.analysis_saving_items} for ii in {wl_fit_item}])):"; ln += 1
            code[ln] = f"            xana.fit_edge(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, **{wl_params})"; ln += 1
            code[ln] = f"        if ('edge50_pos_fit' in {self.analysis_saving_items}):";ln += 1
            code[ln] = f"            xana.find_edge_50(xanes2D_analysis_edge50_fit_s, xanes2D_analysis_edge50_fit_e, xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, optimizer='{edge50_optimizer}', ufac={edge_ufac})";ln += 1
            code[ln] = f"        if 'edge50_pos_dir' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"            xana.find_edge_50(xanes2D_analysis_edge50_fit_s, xanes2D_analysis_edge50_fit_e, xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, optimizer='none', ufac={edge_ufac})";ln += 1
            code[ln] = f"        if ('edge_pos_fit' in {self.analysis_saving_items}):"; ln += 1
            code[ln] = f"            xana.find_edge_deriv(xanes2D_analysis_edge50_fit_s, xanes2D_analysis_edge50_fit_e, optimizer='model',ufac={edge_ufac})"; ln += 1
            code[ln] = f"        if 'edge_pos_dir' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"            xana.find_edge_deriv(xanes2D_analysis_edge50_fit_s, xanes2D_analysis_edge50_fit_e, optimizer='direct',ufac={edge_ufac})"; ln += 1
            code[ln] = f"        if 'wl_pos_fit' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"            xana.find_wl(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, optimizer='model',ufac={wl_ufac})"; ln += 1
            code[ln] = f"        if 'wl_pos_dir' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"            xana.find_wl(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, optimizer='direct',ufac={wl_ufac})"; ln += 1
            code[ln] = f"        if 'wl_peak_height_dir' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"            xana.cal_drt_wl_peak_hgt(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e)"; ln += 1
            code[ln] = f"        if ('centroid_of_eng' or 'centroid_of_eng_relative_to_wl' or 'weighted_attenuation' or 'weighted_eng') in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"            xana.cal_wgt_eng(xanes2D_analysis_pre_edge_e)"; ln += 1
            code[ln] = f"        if 'lcf_fit' in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"            xana.interp_ref_spec()"; ln += 1
            code[ln] = f"            xana.lcf()"; ln += 1
            # code[ln] = f"            print(xana.lcf_fit.shape)";ln += 1
            code[ln] = f"";ln += 1
            code[ln] = f"        for jj in {self.analysis_saving_items}:"; ln += 1
            code[ln] = f"            if jj == 'wl_pos_fit':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.wl_pos_fit)[:]"; ln += 1
            code[ln] = f"            if jj == 'wl_pos_dir':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.wl_pos_dir)[:]"; ln += 1
            code[ln] = f"            if jj == 'wl_peak_height_dir':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.direct_wl_ph)[:]"; ln += 1
            code[ln] = f"            if jj == 'centroid_of_eng':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.centroid_of_eng)[:]"; ln += 1
            code[ln] = f"            if jj == 'centroid_of_eng_relative_to_wl':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.centroid_of_eng_rel_wl)[:]"; ln += 1
            code[ln] = f"            if jj == 'weighted_attenuation':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.weighted_atten)[:]"; ln += 1
            code[ln] = f"            if jj == 'weighted_eng':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.weighted_eng)[:]"; ln += 1
            code[ln] = f"            if jj == 'edge50_pos_fit':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.edge50_pos_fit)[:]"; ln += 1
            code[ln] = f"            if jj == 'edge50_pos_dir':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.edge50_pos_fit_none)[:]"; ln += 1
            code[ln] = f"            if jj == 'edge_pos_fit':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.edge_pos_fit)[:]"; ln += 1
            code[ln] = f"            if jj == 'edge_pos_dir':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.edge_pos_dir)[:]"; ln += 1
            code[ln] = f"            if jj == 'edge_jump_filter':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.edge_jump_mask)[:]"; ln += 1
            code[ln] = f"            if jj == 'edge_offset_filter':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.fitted_edge_mask)[:]"; ln += 1
            code[ln] = f"            if jj == 'pre_edge_sd':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.pre_edge_sd_map)[:]"; ln += 1
            code[ln] = f"            if jj == 'post_edge_sd':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.post_edge_sd_map)[:]"; ln += 1
            code[ln] = f"            if jj == 'pre_edge_mean':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.pre_edge_mean_map)[:]"; ln += 1
            code[ln] = f"            if jj == 'post_edge_mean':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.post_edge_mean_map)[:]"; ln += 1
            code[ln] = f"            if jj  == 'pre_edge_fit_coef':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.pre_edge_fit_rlt[0].reshape(xana.pre_edge_fit_rlt[0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"            if jj == 'post_edge_fit_coef':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.post_edge_fit_rlt[0].reshape(xana.post_edge_fit_rlt[0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"            if jj == 'edge_fit_coef':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.model['edge']['fit_rlt'][0].reshape(xana.model['edge']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"            if jj == 'edge_fit_err':";ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.model['edge']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]";ln += 1
            code[ln] = f"            if jj == 'wl_fit_coef':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][0].reshape(xana.model['wl']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"; ln += 1
            code[ln] = f"            if jj == 'wl_fit_err':";ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]";ln += 1
            code[ln] = f"            if jj == 'norm_spec':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.norm_spec)[:]"; ln += 1
            code[ln] = f"            if jj == 'lcf_fit':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.lcf_fit)[:]"; ln += 1
            code[ln] = f"            if jj == 'lcf_fit_err':"; ln += 1
            code[ln] = f"                _g12[jj][:] = np.float32(xana.lcf_fit_err)[:]"; ln += 1
            code[ln] = f"print('xanes2D analysis is done!')"; ln += 1
            code[ln] = f""; ln += 1

            gen_external_py_script(self.parent_h.xanes_fit_external_command_name, code)
            sig = os.system(f'python {self.parent_h.xanes_fit_external_command_name}')
            # print(4)
            if sig == 0:
                self.hs['L[0][x][3][2][0]_fit_run_text'].value = 'XANES2D analysis is done ...'
            else:
                self.hs['L[0][x][3][2][0]_fit_run_text'].value = 'somthing wrong in analysis ...'
            self.parent_h.update_xanes2D_config()

        self.parent_h.boxes_logic()
