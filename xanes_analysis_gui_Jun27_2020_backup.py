#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:56:17 2020

@author: xiao
"""
from ipywidgets import widgets
# from IPython.display import display
# from fnmatch import fnmatch
import os, glob, h5py, json
import numpy as np
import time, gc
from scipy.ndimage import gaussian_filter
from scipy.ndimage import fourier_shift
from gui_components import (SelectFilesButton, NumpyArrayEncoder, 
                            get_handles, enable_disable_boxes, 
                            check_file_availability, get_raw_img_info,
                            gen_external_py_script, fiji_viewer_state, 
                            fiji_viewer_on, fiji_viewer_off,
                            read_config_from_reg_file, restart,
                            determine_element, determine_fitting_energy_range,
                            scale_eng_list)
# from jnius import autoclass
# self.global_h.ImagePlusClass = autoclass('ij.ImagePlus')
import napari
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
                                      1: ["cen", [0, 1], "cen: center of gaussian function"],
                                      2: ["sig", [0, 1e3], "sig: standard deviation of gaussian function"]},
                          "lorentzian":{0: ["amp", [-10, 10], "amp: ampflitude in lorentzian function"],
                                        1: ["cen", [0, 1], "cen: center of lorentzian function"],
                                        2: ["sig", [0, 1e3], "sig: standard deviation of lorentzian function"]},
                          "damped_oscillator":{0: ["amp", [-10, 10], "amp: ampflitude in damped_oscillator function"],
                                               1: ["cen", [0, 1], "cen: center of damped_oscillator function"],
                                               2: ["sig", [0, 1e3], "sig: standard deviation of damped_oscillator function"]},
                          "lognormal":{0: ["amp", [-10, 10], "amp: ampflitude in lognormal function"],
                                       1: ["cen", [0, 1], "cen: center of lognormal function"],
                                       2: ["sig", [0, 1e3], "sig: standard deviation of lognormal function"]},
                          "students_t":{0: ["amp", [-10, 10], "amp: ampflitude in students_t function"],
                                        1: ["cen", [0, 1], "cen: center of students_t function"],
                                        2: ["sig", [0, 1e3], "sig: standard deviation of students_t function"]},
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
                          "moffat":{0: ["amp", [-10, 10], "amp: ampflitude in moffat function"],
                                    1: ["cen", [0, 1], "cen: center of moffat function"],
                                    2: ["sig", [0, 1e3], "sig: standard moffat of gaussian function"],
                                    3: ["beta", [-1e3, 1e3], "beta: "]},
                          "pearson7": {0: ["amp", [-10, 10], "amp: ampflitude in pearson7 function"],
                                       1: ["cen", [0, 1], "cen: center of pearson7 function"],
                                       2: ["sig", [0, 1e3], "sig: standard pearson7 of gaussian function"],
                                       3: ["expo", [-1e2, 1e2], "expo: "]},
                          "breit_wigner":{0: ["amp", [-10, 10], "amp: ampflitude in breit_wigner function"],
                                          1: ["cen", [0, 1], "cen: center of breit_wigner function"],
                                          2: ["sig", [0, 1e3], "sig: standard breit_wigner of gaussian function"],
                                          3: ["q", [-10, 10], "q: "]},
                          "dho":{0: ["amp", [-10, 10], "amp: ampflitude in dho function"],
                                 1: ["cen", [0, 1], "cen: center of dho function"],
                                 2: ["sig", [0, 1e3], "sig: standard dho of gaussian function"],
                                 3: ["gama", [-10, 10], "gama: "]},
                          "expgaussian":{0: ["amp", [-10, 10], "amp: ampflitude in expgaussian function"],
                                         1: ["cen", [0, 1], "cen: center of expgaussian function"],
                                         2: ["sig", [0, 1e3], "sig: standard expgaussian of gaussian function"],
                                         3: ["gama", [-10, 10], "gama: "]},
                          "donaich":{0: ["amp", [-10, 10], "amp: ampflitude in donaich function"],
                                     1: ["cen", [0, 1], "cen: center of donaich function"],
                                     2: ["sig", [0, 1e3], "sig: standard donaich of gaussian function"],
                                     3: ["gama", [-10, 10], "gama: "]},
                          "skewed_gaussian":{0: ["amp", [-10, 10], "amp: ampflitude in skewed_gaussian function"],
                                             1: ["cen", [0, 1], "cen: center of skewed_gaussian function"],
                                             2: ["sig", [0, 1e3], "sig: standard skewed_gaussian of gaussian function"],
                                             3: ["gama", 0, "gama: "]},
                          "expsine":{0: ["amp", [-10, 10], "amp: ampflitude in expsine function"],
                                     1: ["frq", [0, 1], "frq: center of expsine function"],
                                     2: ["shft", [0, 1], "shft: standard expsine of gaussian function"],
                                     3: ["dec", [-10, 10], "dec: "]},
                          "step":{0: ["amp", [-10, 10], "amp: ampflitude in step function"],
                                  1: ["cen", [0, 1], "cen: center of step function"],
                                  2: ["sig", [0, 1e3], "sig: standard step of gaussian function"]},
                          "skewed_voigt":{0: ["amp", [-10, 10], "amp: ampflitude in skewed_voigt function"],
                                          1: ["cen", [0, 1], "cen: center of skewed_voigt function"],
                                          2: ["sig", [0, 1e3], "sig: standard skewed_voigt of gaussian function"],
                                          3: ["gamma", [0, 1e-3], "gamma: "],
                                          4: ["skew", [-10, 10], "skew: "]},
                          "rectangle":{0: ["amp", [-10, 10], "amp: ampflitude in rectangle function"],
                                       1: ["cen1", [0, 1], "cen1: center of rectangle function"],
                                       2: ["sig1", [0, 1e3], "sig1: standard deviation of rectangle function"],
                                       3: ["cen2", [0, 1], "cen2: center of rectangle function"],
                                       4: ["sig2", [0, 1e3], "sig2: standard deviation of rectangle function"]}}

NUMPY_FIT_ORDER = [2, 3, 4]

class xanes_analysis_gui():
    def __init__(self, parent_h, form_sz=[650, 740]):
        self.parent_h = parent_h
        self.hs = {}
        self.form_sz = form_sz
        self.analysis_fit_use_param_bnd = False
        self.analysis_optimizer = 'scipy'
        self.analysis_method = 'lorentzian'
        
        if self.parent_h.gui_name == 'xanes3D':
            self.fn = self.parent_h.xanes3D_save_trial_reg_filename
        elif self.parent_h.gui_name == 'xanes2D':
            self.fn = self.parent_h.xanes2D_save_trial_reg_filename
        
    def set_xanes_analysis_eng_bounds(self):
        eng_list_len = self.parent_h.xanes_analysis_eng_list.shape[0]
        if self.parent_h.xanes_analysis_wl_fit_eng_e > self.parent_h.xanes_analysis_eng_list.max():
            self.parent_h.xanes_analysis_wl_fit_eng_e = \
                self.parent_h.xanes_analysis_eng_list.max()
        elif self.parent_h.xanes_analysis_wl_fit_eng_e > self.parent_h.xanes_analysis_eng_list.min():
            self.parent_h.xanes_analysis_wl_fit_eng_e = \
                self.parent_h.xanes_analysis_eng_list[int(eng_list_len/2)]
        if self.parent_h.xanes_analysis_wl_fit_eng_s < self.parent_h.xanes_analysis_eng_list.min():
            self.parent_h.xanes_analysis_wl_fit_eng_s = \
                self.parent_h.xanes_analysis_eng_list.min()
        elif self.parent_h.xanes_analysis_wl_fit_eng_s < self.parent_h.xanes_analysis_eng_list.max():
            self.parent_h.xanes_analysis_wl_fit_eng_s = \
                self.parent_h.xanes_analysis_eng_list[int(eng_list_len/2)] - 1
        self.hs['L[0][x][3][0][1][1][1]_analysis_energy_range_wl_fit_e_text'].max = \
            self.parent_h.xanes_analysis_eng_list.max()
        self.hs['L[0][x][3][0][1][1][1]_analysis_energy_range_wl_fit_e_text'].min = \
            self.parent_h.xanes_analysis_eng_list.min()
        self.hs['L[0][x][3][0][1][1][0]_analysis_energy_range_wl_fit_s_text'].max = \
            self.parent_h.xanes_analysis_eng_list.max()
        self.hs['L[0][x][3][0][1][1][0]_analysis_energy_range_wl_fit_s_text'].min = \
            self.parent_h.xanes_analysis_eng_list.min()
            
        if self.hs['L[0][x][3][0][1][0][0]_analysis_energy_range_option_dropdown'].value == 'full':            
            if ((self.parent_h.xanes_analysis_edge_eng > self.parent_h.xanes_analysis_eng_list.max()) or
                self.parent_h.xanes_analysis_edge_eng < self.parent_h.xanes_analysis_eng_list.min()):
                self.parent_h.xanes_analysis_edge_eng  = \
                    self.parent_h.xanes_analysis_eng_list[int(eng_list_len/2)]
            if self.parent_h.xanes_analysis_edge_0p5_fit_e > self.parent_h.xanes_analysis_eng_list.max():
                self.parent_h.xanes_analysis_edge_0p5_fit_e = \
                    self.parent_h.xanes_analysis_eng_list.max()
            elif self.parent_h.xanes_analysis_edge_0p5_fit_e < self.parent_h.xanes_analysis_eng_list.min():
                self.parent_h.xanes_analysis_edge_0p5_fit_e = \
                    self.parent_h.xanes_analysis_eng_list[int(eng_list_len/2)]
            if self.parent_h.xanes_analysis_edge_0p5_fit_s < self.parent_h.xanes_analysis_eng_list.min():
                self.parent_h.xanes_analysis_edge_0p5_fit_s = \
                    self.parent_h.xanes_analysis_eng_list.min()
            elif self.parent_h.xanes_analysis_edge_0p5_fit_s > self.parent_h.xanes_analysis_eng_list.max():
                self.parent_h.xanes_analysis_edge_0p5_fit_s = \
                    self.parent_h.xanes_analysis_eng_list[int(eng_list_len/2)] - 1  
            self.hs['L[0][x][3][0][1][0][1]_analysis_energy_range_edge_eng_text'].max = \
                self.parent_h.xanes_analysis_eng_list.max()
            self.hs['L[0][x][3][0][1][0][1]_analysis_energy_range_edge_eng_text'].min = \
                self.parent_h.xanes_analysis_eng_list.min()
            self.hs['L[0][x][3][0][1][1][1]_analysis_energy_range_wl_fit_e_text'].max = \
                self.parent_h.xanes_analysis_eng_list.max()
            self.hs['L[0][x][3][0][1][1][1]_analysis_energy_range_wl_fit_e_text'].min = \
                self.parent_h.xanes_analysis_eng_list.min()
            self.hs['L[0][x][3][0][1][1][0]_analysis_energy_range_wl_fit_s_text'].max = \
                self.parent_h.xanes_analysis_eng_list.max()
            self.hs['L[0][x][3][0][1][1][0]_analysis_energy_range_wl_fit_s_text'].min = \
                self.parent_h.xanes_analysis_eng_list.min()
            self.hs['L[0][x][3][0][1][1][3]_analysis_energy_range_edge0.5_fit_e_text'].max = \
                self.parent_h.xanes_analysis_eng_list.max()
            self.hs['L[0][x][3][0][1][1][3]_analysis_energy_range_edge0.5_fit_e_text'].min = \
                self.parent_h.xanes_analysis_eng_list.min()
            self.hs['L[0][x][3][0][1][1][2]_analysis_energy_range_edge0.5_fit_s_text'].max = \
                self.parent_h.xanes_analysis_eng_list.max()
            self.hs['L[0][x][3][0][1][1][2]_analysis_energy_range_edge0.5_fit_s_text'].min = \
                self.parent_h.xanes_analysis_eng_list.min()
        
    def build_gui(self):
        ## ## ## bin sub-tabs in each tab - analysis&display TAB in 3D_xanes TAB -- start
        ## ## ## ## define functional widgets each tab in each sub-tab - analysis box in analysis&display TAB -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.98*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][0]_analysis_box'] = widgets.VBox()
        self.hs['L[0][x][3][0]_analysis_box'].layout = layout

        ## ## ## ## ## label analysis box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][0][0]_analysis_title_box'] = widgets.HBox()
        self.hs['L[0][x][3][0][0]_analysis_title_box'].layout = layout
        self.hs['L[0][x][3][0][0][0]_analysis_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Analyze 3D XANES' + '</span>')
        # self.hs['L[0][x][3][0][0][0]_analysis_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Analyze XANES3D' + '</span>')
        layout = {'background-color':'white', 'color':'cyan', 'left':'41%'}
        self.hs['L[0][x][3][0][0][0]_analysis_title_text'].layout = layout
        self.hs['L[0][x][3][0][0]_analysis_title_box'].children = get_handles(self.hs, 'L[0][x][3][0][0]_analysis_title_box', -1)
        ## ## ## ## ## label analysis box -- end

        ## ## ## ## ## define type of analysis and energy range -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{1*self.form_sz[1]-98}px', 'height':f'{0.14*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][0][1]_analysis_energy_range_box'] = widgets.VBox()
        self.hs['L[0][x][3][0][1]_analysis_energy_range_box'].layout = layout
        layout = {'border':'none', 'width':f'{1*self.form_sz[1]-106}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][0][1][0]_analysis_energy_range_box1'] = widgets.HBox()
        self.hs['L[0][x][3][0][1][0]_analysis_energy_range_box1'].layout = layout
        layout = {'width':'19%', 'height':'90%'}
        self.hs['L[0][x][3][0][1][0][0]_analysis_energy_range_option_dropdown'] = widgets.Dropdown(description='analysis type',
                                                                                                  description_tooltip='wl: find whiteline positions without doing background removal and normalization; edge0.5: find energy point where the normalized spectrum value equal to 0.5; full: doing regular xanes preprocessing',
                                                                                                  options=['wl', 'full'],
                                                                                                  value ='wl',
                                                                                                  disabled=True)
        self.hs['L[0][x][3][0][1][0][0]_analysis_energy_range_option_dropdown'].layout = layout
        # layout = {'width':'19%', 'height':'100%', 'top':'0%', 'visibility':'hidden'}
        layout = {'width':'19%', 'height':'90%'}
        self.hs['L[0][x][3][0][1][0][1]_analysis_energy_range_edge_eng_text'] = widgets.BoundedFloatText(description='edge eng',
                                                                                                      description_tooltip='edge energy (keV)',
                                                                                                      value =0,
                                                                                                      min = 0,
                                                                                                      max = 50000,
                                                                                                      step=0.5,
                                                                                                      disabled=True)
        self.hs['L[0][x][3][0][1][0][1]_analysis_energy_range_edge_eng_text'].layout = layout
        layout = {'width':'19%', 'height':'90%'}
        self.hs['L[0][x][3][0][1][0][2]_analysis_energy_range_pre_edge_e_text'] = widgets.BoundedFloatText(description='pre edge e',
                                                                                                   description_tooltip='relative ending energy point (keV) of pre-edge from edge energy for background removal',
                                                                                                   value =-50,
                                                                                                   min = -500,
                                                                                                   max = 0,
                                                                                                   step=0.5,
                                                                                                   disabled=True)
        self.hs['L[0][x][3][0][1][0][2]_analysis_energy_range_pre_edge_e_text'].layout = layout
        layout = {'width':'19%', 'height':'90%'}
        self.hs['L[0][x][3][0][1][0][3]_analysis_energy_range_post_edge_s_text'] = widgets.BoundedFloatText(description='post edge s',
                                                                                                   description_tooltip='relative starting energy point (keV) of post-edge from edge energy for normalization',
                                                                                                   value =0.1,
                                                                                                   min = 0,
                                                                                                   max = 500,
                                                                                                   step=0.5,
                                                                                                   disabled=True)
        self.hs['L[0][x][3][0][1][0][3]_analysis_energy_range_post_edge_s_text'].layout = layout
        layout = {'width':'19%', 'height':'90%'}
        self.hs['L[0][x][3][0][1][0][4]_analysis_filter_spec_checkbox'] = widgets.Checkbox(description='flt spec',
                                                                                           description_tooltip='relative starting energy point (keV) of post-edge from edge energy for normalization',
                                                                                           value = False,
                                                                                           disabled=True)
        self.hs['L[0][x][3][0][1][0][4]_analysis_filter_spec_checkbox'].layout = layout

        self.hs['L[0][x][3][0][1][0]_analysis_energy_range_box1'].children = get_handles(self.hs, 'L[0][x][3][0][1][0]_analysis_energy_range_box1', -1)
        self.hs['L[0][x][3][0][1][0][0]_analysis_energy_range_option_dropdown'].observe(self.L0_x_3_0_1_0_0_analysis_energy_range_option_dropdown, names='value')
        self.hs['L[0][x][3][0][1][0][1]_analysis_energy_range_edge_eng_text'].observe(self.L0_x_3_0_1_0_1_analysis_energy_range_edge_eng_text, names='value')
        self.hs['L[0][x][3][0][1][0][2]_analysis_energy_range_pre_edge_e_text'].observe(self.L0_x_3_0_1_0_2_analysis_energy_range_pre_edge_e_text, names='value')
        self.hs['L[0][x][3][0][1][0][3]_analysis_energy_range_post_edge_s_text'].observe(self.L0_x_3_0_1_0_3_analysis_energy_range_post_edge_s_text, names='value')
        self.hs['L[0][x][3][0][1][0][4]_analysis_filter_spec_checkbox'].observe(self.L0_x_3_0_1_0_4_analysis_filter_spec_checkbox, names='value')

        layout = {'border':'none', 'width':f'{1*self.form_sz[1]-106}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][0][1][1]_analysis_energy_range_box2'] = widgets.HBox()
        self.hs['L[0][x][3][0][1][1]_analysis_energy_range_box2'].layout = layout
        layout = {'width':'19%', 'height':'90%'}
        self.hs['L[0][x][3][0][1][1][0]_analysis_energy_range_wl_fit_s_text'] = widgets.BoundedFloatText(description='wl eng s',
                                                                                            description_tooltip='absolute energy starting point (keV) for whiteline fitting. "wl eng s" and "wl eng e" shoudl be roughly symmetric about whiteline peak.',
                                                                                            value =0,
                                                                                            min = 0,
                                                                                            max = 50000,
                                                                                            step=0.5,
                                                                                            disabled=True)
        self.hs['L[0][x][3][0][1][1][0]_analysis_energy_range_wl_fit_s_text'].layout = layout
        layout = {'width':'19%', 'height':'90%'}
        self.hs['L[0][x][3][0][1][1][1]_analysis_energy_range_wl_fit_e_text'] = widgets.BoundedFloatText(description='wl eng e',
                                                                                            description_tooltip='absolute energy ending point (keV) for whiteline fitting. "wl eng s" and "wl eng e" shoudl be roughly symmetric about whiteline peak.',
                                                                                            value =0,
                                                                                            min = 0,
                                                                                            max = 50030,
                                                                                            step=0.5,
                                                                                            disabled=True)
        self.hs['L[0][x][3][0][1][1][1]_analysis_energy_range_wl_fit_e_text'].layout = layout
        layout = {'width':'19%', 'height':'90%'}
        self.hs['L[0][x][3][0][1][1][2]_analysis_energy_range_edge0.5_fit_s_text'] = widgets.BoundedFloatText(description='edge0.5 s',
                                                                                                 description_tooltip='absolute energy starting point (keV) for edge-0.5 fitting. "edge0.5 s" and "edge0.5 e" shoudl be close to the foot and the peak locations on the ascendent edge of the spectra.',
                                                                                                 value =0,
                                                                                                 min = 0,
                                                                                                 max = 50000,
                                                                                                 step=0.5,
                                                                                                 disabled=True)
        self.hs['L[0][x][3][0][1][1][2]_analysis_energy_range_edge0.5_fit_s_text'].layout = layout
        layout = {'width':'19%', 'height':'90%'}
        self.hs['L[0][x][3][0][1][1][3]_analysis_energy_range_edge0.5_fit_e_text'] = widgets.BoundedFloatText(description='edge0.5 e',
                                                                                                   description_tooltip='absolute energy starting point (keV) for edge-0.5 fitting. "edge0.5 s" and "edge0.5 e" shoudl be close to the foot and the peak locations on the ascendent edge of the spectra.',
                                                                                                   value =0,
                                                                                                   min = 0,
                                                                                                   max = 50030,
                                                                                                   step=0.5,
                                                                                                   disabled=True)
        self.hs['L[0][x][3][0][1][1][3]_analysis_energy_range_edge0.5_fit_e_text'].layout = layout

        layout = {'width':'15%', 'height':'90%', 'left':'7%'}
        self.hs['L[0][x][3][0][1][1][4]_analysis_energy_confirm_button'] = widgets.Button(description='Confirm',
                                                                                             description_tooltip='Confirm energy range settings',
                                                                                             disabled=True)
        self.hs['L[0][x][3][0][1][1][4]_analysis_energy_confirm_button'].layout = layout
        self.hs['L[0][x][3][0][1][1][4]_analysis_energy_confirm_button'].style.button_color = 'darkviolet'

        self.hs['L[0][x][3][0][1][1]_analysis_energy_range_box2'].children = get_handles(self.hs, 'L[0][x][3][0][1][1]_analysis_energy_range_box2', -1)
        self.hs['L[0][x][3][0][1][1][0]_analysis_energy_range_wl_fit_s_text'].observe(self.L0_x_3_0_1_1_0_analysis_energy_range_wl_fit_s_text, names='value')
        self.hs['L[0][x][3][0][1][1][1]_analysis_energy_range_wl_fit_e_text'].observe(self.L0_x_3_0_1_1_1_analysis_energy_range_wl_fit_e_text, names='value')
        self.hs['L[0][x][3][0][1][1][2]_analysis_energy_range_edge0.5_fit_s_text'].observe(self.L0_x_3_0_1_1_2_analysis_energy_range_edge0p5_fit_s_text, names='value')
        self.hs['L[0][x][3][0][1][1][3]_analysis_energy_range_edge0.5_fit_e_text'].observe(self.L0_x_3_0_1_1_3_analysis_energy_range_edge0p5_fit_e_text, names='value')
        self.hs['L[0][x][3][0][1][1][4]_analysis_energy_confirm_button'].on_click(self.L0_x_3_0_1_0_4_analysis_energy_range_confirm_button)

        self.hs['L[0][x][3][0][1]_analysis_energy_range_box'].children = get_handles(self.hs, 'L[0][x][3][0][1]_analysis_energy_range_box', -1)
        ## ## ## ## ## define type of analysis and energy range -- end

        ## ## ## ## ## define energy filter related parameters -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{1*self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][0][2]_analysis_energy_filter_box'] = widgets.HBox()
        self.hs['L[0][x][3][0][2]_analysis_energy_filter_box'].layout = layout
        layout = {'width':'48%', 'height':'90%'}
        self.hs['L[0][x][3][0][2][0]_analysis_energy_filter_edge_jump_thres_slider'] = widgets.FloatSlider(description='edge jump thres',
                                                                                                              description_tooltip='edge jump in unit of the standard deviation of the signal in energy range pre to the edge. larger threshold enforces more restrict data quality validation on the data',
                                                                                                              value =1,
                                                                                                              min = 0,
                                                                                                              max = 10,
                                                                                                              step=0.1,
                                                                                                              disabled=True)
        self.hs['L[0][x][3][0][2][0]_analysis_energy_filter_edge_jump_thres_slider'].layout = layout
        layout = {'width':'48%', 'height':'90%'}
        self.hs['L[0][x][3][0][2][1]_analysis_energy_filter_edge_offset_slider'] = widgets.FloatSlider(description='edge offset',
                                                                                      description_tooltip='offset between pre-edge and post-edge in unit of the standard deviation of pre-edge. larger offser enforces more restrict data quality validation on the data',
                                                                                      value =1,
                                                                                      min = 0,
                                                                                      max = 10,
                                                                                      step=0.1,
                                                                                      disabled=True)
        self.hs['L[0][x][3][0][2][1]_analysis_energy_filter_edge_offset_slider'].layout = layout
        self.hs['L[0][x][3][0][2][0]_analysis_energy_filter_edge_jump_thres_slider'].observe(self.L0_x_3_0_2_0_analysis_energy_filter_edge_jump_thres_slider_change, names='value')
        self.hs['L[0][x][3][0][2][1]_analysis_energy_filter_edge_offset_slider'].observe(self.L0_x_3_0_2_1_analysis_energy_filter_edge_offset_slider_change, names='value')
        self.hs['L[0][x][3][0][2]_analysis_energy_filter_box'].children = get_handles(self.hs, 'L[0][x][3][0][2]_analysis_energy_filter_box', -1)
        ## ## ## ## ## define energy filter related parameters -- end

        ## ## ## ## ## define mask parameters -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{1*self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][0][3]_analysis_image_mask_box'] = widgets.HBox()
        self.hs['L[0][x][3][0][3]_analysis_image_mask_box'].layout = layout
        layout = {'width':'20%', 'height':'90%'}
        self.hs['L[0][x][3][0][3][0]_analysis_image_use_mask_checkbox'] = widgets.Checkbox(description='use mask',
                                                                                              description_tooltip='use a mask based on gray value threshold to define sample region',
                                                                                              value =False,
                                                                                              disabled=True)
        self.hs['L[0][x][3][0][3][0]_analysis_image_use_mask_checkbox'].layout = layout
        layout = {'width':'40%', 'height':'90%'}
        self.hs['L[0][x][3][0][3][1]_analysis_image_mask_scan_id_slider'] = widgets.IntSlider(description='mask scan id',
                                                                                                 description_tooltip='scan id with which the mask is made',
                                                                                                 value =1,
                                                                                                 min = 0,
                                                                                                 max = 10,
                                                                                                 disabled=True)
        self.hs['L[0][x][3][0][3][1]_analysis_image_mask_scan_id_slider'].layout = layout
        layout = {'width':'48%', 'height':'90%'}
        self.hs['L[0][x][3][0][3][1]_analysis_image_mask_thres_slider'] = widgets.FloatSlider(description='mask thres',
                                                                                            description_tooltip='threshold for making the mask',
                                                                                            value =0,
                                                                                            min = -1,
                                                                                            max = 1,
                                                                                            step = 0.00005,
                                                                                            readout_format='.5f',
                                                                                            disabled=True)
        self.hs['L[0][x][3][0][3][1]_analysis_image_mask_thres_slider'].layout = layout

        self.hs['L[0][x][3][0][3]_analysis_image_mask_box'].children = get_handles(self.hs, 'L[0][x][3][0][3]_analysis_image_mask_box', -1)
        self.hs['L[0][x][3][0][3][0]_analysis_image_use_mask_checkbox'].observe(self.L0_x_3_0_3_0_analysis_image_use_mask_checkbox, names='value')
        self.hs['L[0][x][3][0][3][1]_analysis_image_mask_scan_id_slider'].observe(self.L0_x_3_0_3_1_analysis_image_mask_scan_id_slider, names='value')
        self.hs['L[0][x][3][0][3][1]_analysis_image_mask_thres_slider'].observe(self.L0_x_3_0_3_1_analysis_image_mask_thres_slider, names='value')
        ## ## ## ## ## define mask parameters -- end
        
        ## ## ## ## ## define fitting parameters -- start
        self.hs['L[0][x][3][0][4]_analysis_tool_box'] = widgets.Tab()
        
        ## ## ## ## ## ## define fitting parameters -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{0.96*self.form_sz[1]-98}px', 'height':f'{0.49*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][0][4][0]_analysis_fit_params_box'] = widgets.HBox()
        self.hs['L[0][x][3][0][4][0]_analysis_fit_params_box'].layout = layout
        
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'] = widgets.GridspecLayout(8, 200,
                                                                                       layout = {"border":"3px solid #FFCC00",
                                                                                                 'width':f'{0.96*self.form_sz[1]-98}px',
                                                                                                 "height":f"{0.48*(self.form_sz[0]-128)}px",
                                                                                                 "align_items":"flex-start",
                                                                                                 "justify_items":"flex-start"})
        
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][0, :50] = widgets.Dropdown(description='optimizer',
                                                                                         description_tooltip='use scipy.optimize or numpy.polyfit',
                                                                                         options = ['scipy', 
                                                                                                    'numpy'],
                                                                                         value ='scipy',
                                                                                         layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                         disabled=True)
        self.hs['L[0][x][3][0][4][0][0][0]_analysis_gridlayout_optimizer'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][0, :50]
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][0, 50:100] = widgets.Dropdown(description='peak func',
                                                                                            description_tooltip='peak fitting functions',
                                                                                            options = PEAK_LINE_SHAPES,
                                                                                            value ='lorentzian',
                                                                                            layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                            disabled=True)
        self.hs['L[0][x][3][0][4][0][0][1]_analysis_gridlayout_fit_func'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][0, 50:100]
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][0, 100:150] = widgets.Checkbox(description='para bnd',
                                                                                             value = False,
                                                                                             description_tooltip = "if set boundaries to the peak fitting function's parameters",
                                                                                             layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                             disabled=True)
        self.hs['L[0][x][3][0][4][0][0][2]_analysis_gridlayout_fit_bnd'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][0, 100:150]
        
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][1, 0:50] = widgets.BoundedFloatText(description = 'p0',
                                                                                           value = 0,
                                                                                           min = -1e5,
                                                                                           max = 1e5,
                                                                                           description_tooltip = "fitting function variable 0",
                                                                                           layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                           disabled=True)
        self.hs['L[0][x][3][0][4][0][0][3]_analysis_gridlayout_p0'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][1, 0:50]
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][1, 50:100] = widgets.BoundedFloatText(description = 'p1',
                                                                                             value = 0,
                                                                                             min = -1e5,
                                                                                             max = 1e5,
                                                                                             description_tooltip = "fitting function variable 1",
                                                                                             layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                             disabled=True)
        self.hs['L[0][x][3][0][4][0][0][4]_analysis_gridlayout_p1'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][1, 50:100]
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][1, 100:150] = widgets.BoundedFloatText(description = 'p2',
                                                                                              value = 0,
                                                                                              min = -1e5,
                                                                                              max = 1e5,
                                                                                              description_tooltip = "fitting function variable 2",
                                                                                              layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                              disabled=True)
        self.hs['L[0][x][3][0][4][0][0][5]_analysis_gridlayout_p2'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][1, 100:150]
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][1, 150:200] = widgets.BoundedFloatText(description = 'p3',
                                                                                              value = 0,
                                                                                              min = -1e5,
                                                                                              max = 1e5,
                                                                                              description_tooltip = "fitting function variable 3",
                                                                                              layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                              disabled=True)
        self.hs['L[0][x][3][0][4][0][0][6]_analysis_gridlayout_p3'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][1, 150:200]
        
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][2, 0:50] = widgets.BoundedFloatText(description = 'p4',
                                                                                           value = 0,
                                                                                           min = -1e5,
                                                                                           max = 1e5,
                                                                                           description_tooltip = "fitting function variable 4",
                                                                                           layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                           disabled=True)
        self.hs['L[0][x][3][0][4][0][0][7]_analysis_gridlayout_p4'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][2, 0:50]
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][2, 50:100] = widgets.Dropdown(description = 'p5',
                                                                                            value = 'linear',
                                                                                            options = ['linear'],
                                                                                            description_tooltip = "fitting function variable 5",
                                                                                            layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                            disabled=True)
        self.hs['L[0][x][3][0][4][0][0][8]_analysis_gridlayout_p5'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][2, 50:100]
        
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][3, 0:50] = widgets.Dropdown(description = 'jac',
                                                                                          value = '3-point',
                                                                                          options = ['2-point', '3-point', 'cs'],
                                                                                          description_tooltip = "",
                                                                                          layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                          disabled=True)
        self.hs['L[0][x][3][0][4][0][0][9]_analysis_gridlayout_jac'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][3, 0:50]
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][3, 50:100] = widgets.Dropdown(description = 'method',
                                                                                            value = 'trf',
                                                                                            options = ['trf', 'dogbox', 'lm'],
                                                                                            description_tooltip = "",
                                                                                            layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                            disabled=True)
        self.hs['L[0][x][3][0][4][0][0][10]_analysis_gridlayout_method'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][3, 50:100]
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][3, 100:150] = widgets.BoundedFloatText(description = 'ftol',
                                                                                              value = 1e-7,
                                                                                              min = 0,
                                                                                              max = 1e-3,
                                                                                              description_tooltip = "function value change tolerance for terminating the optimization process",
                                                                                              layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                              disabled=True)
        self.hs['L[0][x][3][0][4][0][0][11]_analysis_gridlayout_ftol'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][3, 100:150]
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][3, 150:200] = widgets.BoundedFloatText(description = 'xtol',
                                                                                              value = 1e-7,
                                                                                              min = 0,
                                                                                              max = 1e-3,
                                                                                              description_tooltip = "function parameter change tolerance for terminating the optimization process",
                                                                                              layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                              disabled=True)
        self.hs['L[0][x][3][0][4][0][0][12]_analysis_gridlayout_xtol'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][3, 150:200]
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][4, 0:50] = widgets.BoundedFloatText(description = 'gtol',
                                                                                           value = 1e-7,
                                                                                           min = 0,
                                                                                           max = 1e-3,
                                                                                           description_tooltip = "function gradient change tolerance for terminating the optimization process",
                                                                                           layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                           disabled=True)
        self.hs['L[0][x][3][0][4][0][0][13]_analysis_gridlayout_gtol'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][4, 0:50]
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][4, 50:100] = widgets.BoundedIntText(description = 'ufac',
                                                                                           value = 100,
                                                                                           min = 1,
                                                                                           max = 100,
                                                                                           description_tooltip = "upsampling factor to energy points in peak fitting",
                                                                                           layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                           disabled=True)
        self.hs['L[0][x][3][0][4][0][0][14]_analysis_gridlayout_ufac'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][4, 50:100]
        
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][5, 0:50] = widgets.BoundedFloatText(description = 'p0 lb',
                                                                                           value = 0,
                                                                                           min = -1e5,
                                                                                           max = 1e5,
                                                                                           description_tooltip = "p0 lower bound",
                                                                                           layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                           disabled=True)
        self.hs['L[0][x][3][0][4][0][0][15]_analysis_gridlayout_p0_lb'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][5, 0:50]
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][5, 50:100] = widgets.BoundedFloatText(description = 'p0 ub',
                                                                                             value = 0,
                                                                                             min = -1e5,
                                                                                             max = 1e5,
                                                                                             description_tooltip = "p0 upper bound",
                                                                                             layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                             disabled=True)
        self.hs['L[0][x][3][0][4][0][0][16]_analysis_gridlayout_p0_ub'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][5, 50:100]
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][5, 100:150] = widgets.BoundedFloatText(description = 'p1 lb',
                                                                                              value = 0,
                                                                                              min = -1e5,
                                                                                              max = 1e5,
                                                                                              description_tooltip = "p1 lower bound",
                                                                                              layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                              disabled=True)
        self.hs['L[0][x][3][0][4][0][0][17]_analysis_gridlayout_p1_lb'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][5, 100:150]
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][5, 150:200] = widgets.BoundedFloatText(description = 'p1 ub',
                                                                                              value = 0,
                                                                                              min = -1e5,
                                                                                              max = 1e5,
                                                                                              description_tooltip = "p1 upper bound",
                                                                                              layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                              disabled=True)
        self.hs['L[0][x][3][0][4][0][0][18]_analysis_gridlayout_p1_ub'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][5, 150:200]
        
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][6, 0:50] = widgets.BoundedFloatText(description = 'p2 lb',
                                                                                           value = 0,
                                                                                           min = -1e5,
                                                                                           max = 1e5,
                                                                                           description_tooltip = "p2 lower bound",
                                                                                           layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                           disabled=True)
        self.hs['L[0][x][3][0][4][0][0][19]_analysis_gridlayout_p2_lb'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][6, 0:50]
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][6, 50:100] = widgets.BoundedFloatText(description = 'p3 ub',
                                                                                             value = 0,
                                                                                             min = -1e5,
                                                                                             max = 1e5,
                                                                                             description_tooltip = "p3 upper bound",
                                                                                             layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                             disabled=True)
        self.hs['L[0][x][3][0][4][0][0][20]_analysis_gridlayout_p2_ub'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][6, 50:100]
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][6, 100:150] = widgets.BoundedFloatText(description = 'p4 lb',
                                                                                              value = 0,
                                                                                              min = -1e5,
                                                                                              max = 1e5,
                                                                                              description_tooltip = "p4 lower bound",
                                                                                              layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                              disabled=True)
        self.hs['L[0][x][3][0][4][0][0][21]_analysis_gridlayout_p3_lb'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][6, 100:150]
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][6, 150:200] = widgets.BoundedFloatText(description = 'p4 ub',
                                                                                             value = 0,
                                                                                             min = -1e5,
                                                                                             max = 1e5,
                                                                                             description_tooltip = "p4 upper bound",
                                                                                             layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                             disabled=True)
        self.hs['L[0][x][3][0][4][0][0][22]_analysis_gridlayout_p3_ub'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][6, 150:200]
        
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][7, 0:50] = widgets.BoundedFloatText(description = 'p5 lb',
                                                                                           value = 0,
                                                                                           min = -1e5,
                                                                                           max = 1e5,
                                                                                           description_tooltip = "p5 lower bound",
                                                                                           layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                           disabled=True)
        self.hs['L[0][x][3][0][4][0][0][23]_analysis_gridlayout_p4_lb'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][7, 0:50]
        self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][7, 50:100] = widgets.BoundedFloatText(description = 'p5 ub',
                                                                                             value = 0,
                                                                                             min = -1e5,
                                                                                             max = 1e5,
                                                                                             description_tooltip = "p5 upper bound",
                                                                                             layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                                             disabled=True)
        self.hs['L[0][x][3][0][4][0][0][24]_analysis_gridlayout_p4_ub'] = self.hs['L[0][x][3][0][4][0][0]_analysis_gridlayout'][7, 50:100]
        
        self.hs['L[0][x][3][0][4][0][0][0]_analysis_gridlayout_optimizer'].observe(self.L0_x_3_0_4_0_0_0_analysis_gridlayout_optimizer, names='value')
        self.hs['L[0][x][3][0][4][0][0][1]_analysis_gridlayout_fit_func'].observe(self.L0_x_3_0_4_0_0_1_analysis_gridlayout_fit_func, names='value')
        self.hs['L[0][x][3][0][4][0][0][2]_analysis_gridlayout_fit_bnd'].observe(self.L0_x_3_0_4_0_0_2_analysis_gridlayout_fit_bnd, names='value')
        
        self.hs['L[0][x][3][0][4][0]_analysis_fit_params_box'].children = get_handles(self.hs, 'L[0][x][3][0][4][0]_analysis_fit_params_box', -1)
        ## ## ## ## ## ## define fitting parameters -- end
        
        ## ## ## ## ## ## define saving setting -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{0.96*self.form_sz[1]-98}px', 'height':f'{0.49*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][0][4][1]_analysis_save_setting_box'] = widgets.HBox()
        self.hs['L[0][x][3][0][4][1]_analysis_save_setting_box'].layout = layout
        layout = {'width':'20%', 'height':'90%'}
        ## ## ## ## ## ## define saving setting -- end
        
        self.hs['L[0][x][3][0][4]_analysis_tool_box'].children = [self.hs['L[0][x][3][0][4][0]_analysis_fit_params_box'],
                                                                  self.hs['L[0][x][3][0][4][1]_analysis_save_setting_box']]
        self.hs['L[0][x][3][0][4]_analysis_tool_box'].set_title(0, 'fitting params')
        self.hs['L[0][x][3][0][4]_analysis_tool_box'].set_title(1, 'saving setting')
        ## ## ## ## ## define fitting parameters -- send
        
        ## ## ## ## ## run xanes3D analysis -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{1*self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][0][5]_analysis_run_box'] = widgets.HBox()
        self.hs['L[0][x][3][0][5]_analysis_run_box'].layout = layout
        layout = {'width':'85%', 'height':'90%'}
        self.hs['L[0][x][3][0][5][0]_analysis_run_text'] = widgets.Text(description='please check your settings before run the analysis .. ',
                                                                        disabled=True)
        self.hs['L[0][x][3][0][5][0]_analysis_run_text'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][x][3][0][5][1]_analysis_run_button'] = widgets.Button(description='run',
                                                                            disabled=True)
        self.hs['L[0][x][3][0][5][1]_analysis_run_button'].layout = layout
        self.hs['L[0][x][3][0][5][1]_analysis_run_button'].style.button_color = 'darkviolet'
        self.hs['L[0][x][3][0][5]_analysis_run_box'].children = get_handles(self.hs, 'L[0][x][3][0][5]_analysis_run_box', -1)
        self.hs['L[0][x][3][0][5][1]_analysis_run_button'].on_click(self.L0_x_3_0_5_1_analysis_run_button)
        ## ## ## ## ## run xanes3D analysis -- end

        ## ## ## ## ## run analysis progress -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][3][0][6]_analysis_progress_box'] = widgets.HBox()
        self.hs['L[0][x][3][0][6]_analysis_progress_box'].layout = layout
        layout = {'width':'100%', 'height':'90%'}
        self.hs['L[0][x][3][0][6][0]_analysis_run_progress_bar'] = widgets.IntProgress(value=0,
                                                                                       min=0,
                                                                                       max=10,
                                                                                       step=1,
                                                                                       description='Completing:',
                                                                                       bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                                                                       orientation='horizontal')
        self.hs['L[0][x][3][0][6][0]_analysis_run_progress_bar'].layout = layout
        self.hs['L[0][x][3][0][6]_analysis_progress_box'].children = get_handles(self.hs, 'L[0][x][3][0][6]_analysis_progress_box', -1)
        ## ## ## ## ## run analysis progress -- end

        self.hs['L[0][x][3][0]_analysis_box'].children = get_handles(self.hs, 'L[0][x][3][0]_analysis_box', -1)
        ## ## ## ## define functional widgets each tab in each sub-tab - analysis box in analysis&display TAB -- end
        self.bundle_analysis_variable_handles()
        
    def bundle_analysis_variable_handles(self):
        self.analysis_fit_func_arg_handles = [self.hs['L[0][x][3][0][4][0][0][3]_analysis_gridlayout_p0'],
                                              self.hs['L[0][x][3][0][4][0][0][4]_analysis_gridlayout_p1'],
                                              self.hs['L[0][x][3][0][4][0][0][5]_analysis_gridlayout_p2'],
                                              self.hs['L[0][x][3][0][4][0][0][6]_analysis_gridlayout_p3'],
                                              self.hs['L[0][x][3][0][4][0][0][7]_analysis_gridlayout_p4'],
                                              self.hs['L[0][x][3][0][4][0][0][8]_analysis_gridlayout_p5']]
        
        self.analysis_optimizer_arg_handles = [self.hs['L[0][x][3][0][4][0][0][9]_analysis_gridlayout_jac'],
                                               self.hs['L[0][x][3][0][4][0][0][10]_analysis_gridlayout_method'],
                                               self.hs['L[0][x][3][0][4][0][0][11]_analysis_gridlayout_ftol'],
                                               self.hs['L[0][x][3][0][4][0][0][12]_analysis_gridlayout_xtol'],
                                               self.hs['L[0][x][3][0][4][0][0][13]_analysis_gridlayout_gtol'],
                                               self.hs['L[0][x][3][0][4][0][0][14]_analysis_gridlayout_ufac']]
        
        self.analysis_fit_func_bnd_handles = [self.hs['L[0][x][3][0][4][0][0][15]_analysis_gridlayout_p0_lb'],
                                              self.hs['L[0][x][3][0][4][0][0][16]_analysis_gridlayout_p0_ub'],
                                              self.hs['L[0][x][3][0][4][0][0][17]_analysis_gridlayout_p1_lb'],
                                              self.hs['L[0][x][3][0][4][0][0][18]_analysis_gridlayout_p1_ub'],
                                              self.hs['L[0][x][3][0][4][0][0][19]_analysis_gridlayout_p2_lb'],
                                              self.hs['L[0][x][3][0][4][0][0][20]_analysis_gridlayout_p2_ub'],
                                              self.hs['L[0][x][3][0][4][0][0][21]_analysis_gridlayout_p3_lb'],
                                              self.hs['L[0][x][3][0][4][0][0][22]_analysis_gridlayout_p3_ub'],
                                              self.hs['L[0][x][3][0][4][0][0][23]_analysis_gridlayout_p4_lb'],
                                              self.hs['L[0][x][3][0][4][0][0][24]_analysis_gridlayout_p4_ub']]
        
    def L0_x_3_0_1_0_0_analysis_energy_range_option_dropdown(self, a):
        self.parent_h.xanes_analysis_eng_configured = False
        self.parent_h.xanes_analysis_type = a['owner'].value
        self.hs['L[0][x][3][0][1][0][1]_analysis_energy_range_edge_eng_text'].value = self.parent_h.xanes_analysis_edge_eng
        
        self.hs['L[0][x][3][0][1][1][1]_analysis_energy_range_wl_fit_e_text'].value = self.parent_h.xanes_analysis_wl_fit_eng_e
        self.hs['L[0][x][3][0][1][1][0]_analysis_energy_range_wl_fit_s_text'].value = self.parent_h.xanes_analysis_wl_fit_eng_s
        self.hs['L[0][x][3][0][1][1][3]_analysis_energy_range_edge0.5_fit_e_text'].value = self.parent_h.xanes_analysis_edge_0p5_fit_e
        self.hs['L[0][x][3][0][1][1][2]_analysis_energy_range_edge0.5_fit_s_text'].value = self.parent_h.xanes_analysis_edge_0p5_fit_s
        self.hs['L[0][x][3][0][1][0][2]_analysis_energy_range_pre_edge_e_text'].value = self.parent_h.xanes_analysis_pre_edge_e
        self.hs['L[0][x][3][0][1][0][3]_analysis_energy_range_post_edge_s_text'].value = self.parent_h.xanes_analysis_post_edge_s
        
        self.set_xanes_analysis_eng_bounds()
        self.hs['L[0][x][3][0][4][0]_analysis_run_text'].value ='please check your settings before run the analysis ...'
        if a['owner'].value == 'wl':
            layout = {'width':'19%', 'height':'100%', 'top':'15%', 'visibility':'hidden'}
            self.hs['L[0][x][3][0][1][0][1]_analysis_energy_range_edge_eng_text'].layout = layout
            layout = {'width':'19%', 'height':'100%', 'top':'15%', 'visibility':'hidden'}
            self.hs['L[0][x][3][0][1][0][2]_analysis_energy_range_pre_edge_e_text'].layout = layout
            layout = {'width':'19%', 'height':'100%', 'top':'15%', 'visibility':'hidden'}
            self.hs['L[0][x][3][0][1][0][3]_analysis_energy_range_post_edge_s_text'].layout = layout
            layout = {'width':'19%', 'height':'100%', 'top':'25%', 'visibility':'hidden'}
            self.hs['L[0][x][3][0][1][1][2]_analysis_energy_range_edge0.5_fit_s_text'].layout = layout
            layout = {'width':'19%', 'height':'100%', 'top':'25%', 'visibility':'hidden'}
            self.hs['L[0][x][3][0][1][1][3]_analysis_energy_range_edge0.5_fit_e_text'].layout = layout
        elif a['owner'].value == 'full':
            layout = {'width':'19%', 'height':'100%', 'top':'15%', 'visibility':'visible'}
            self.hs['L[0][x][3][0][1][0][1]_analysis_energy_range_edge_eng_text'].layout = layout
            layout = {'width':'19%', 'height':'100%', 'top':'15%', 'visibility':'visible'}
            self.hs['L[0][x][3][0][1][0][2]_analysis_energy_range_pre_edge_e_text'].layout = layout
            self.hs['L[0][x][3][0][1][0][2]_analysis_energy_range_pre_edge_e_text'].max = 0
            self.hs['L[0][x][3][0][1][0][2]_analysis_energy_range_pre_edge_e_text'].value = -50
            self.hs['L[0][x][3][0][1][0][2]_analysis_energy_range_pre_edge_e_text'].min = -500
            layout = {'width':'19%', 'height':'100%', 'top':'15%', 'visibility':'visible'}
            self.hs['L[0][x][3][0][1][0][3]_analysis_energy_range_post_edge_s_text'].layout = layout
            self.hs['L[0][x][3][0][1][0][3]_analysis_energy_range_post_edge_s_text'].min = 0
            self.hs['L[0][x][3][0][1][0][3]_analysis_energy_range_post_edge_s_text'].value = 100
            self.hs['L[0][x][3][0][1][0][3]_analysis_energy_range_post_edge_s_text'].max = 500
            layout = {'width':'19%', 'height':'100%', 'top':'25%', 'visibility':'visible'}
            self.hs['L[0][x][3][0][1][1][2]_analysis_energy_range_edge0.5_fit_s_text'].layout = layout
            layout = {'width':'19%', 'height':'100%', 'top':'25%', 'visibility':'visible'}
            self.hs['L[0][x][3][0][1][1][3]_analysis_energy_range_edge0.5_fit_e_text'].layout = layout
        self.parent_h.boxes_logic()

    def L0_x_3_0_1_0_1_analysis_energy_range_edge_eng_text(self, a):
        self.parent_h.xanes_analysis_eng_configured = False
        self.parent_h.xanes_analysis_edge_eng = self.hs['L[0][x][3][0][1][0][1]_analysis_energy_range_edge_eng_text'].value
        self.hs['L[0][x][3][0][4][0]_analysis_run_text'].description ='please check your settings before run the analysis ...'
        # self.parent_h.boxes_logic()

    def L0_x_3_0_1_0_2_analysis_energy_range_pre_edge_e_text(self, a):
        self.parent_h.xanes_analysis_eng_configured = False
        self.parent_h.xanes_analysis_pre_edge_e = self.hs['L[0][x][3][0][1][0][2]_analysis_energy_range_pre_edge_e_text'].value
        self.hs['L[0][x][3][0][4][0]_analysis_run_text'].description ='please check your settings before run the analysis ...'
        self.parent_h.boxes_logic()

    def L0_x_3_0_1_0_3_analysis_energy_range_post_edge_s_text(self, a):
        self.parent_h.xanes_analysis_eng_configured = False
        self.parent_h.xanes_analysis_post_edge_s = self.hs['L[0][x][3][0][1][0][3]_analysis_energy_range_post_edge_s_text'].value
        self.hs['L[0][x][3][0][4][0]_analysis_run_text'].description ='please check your settings before run the analysis ...'
        self.parent_h.boxes_logic()

    def L0_x_3_0_1_0_4_analysis_filter_spec_checkbox(self, a):
        self.parent_h.xanes_analysis_eng_configured = False
        self.parent_h.xanes_analysis_use_flt_spec = a['owner'].value
        self.hs['L[0][x][3][0][4][0]_analysis_run_text'].description ='please check your settings before run the analysis ...'
        self.parent_h.boxes_logic()

    def L0_x_3_0_1_1_0_analysis_energy_range_wl_fit_s_text(self, a):
        self.parent_h.xanes_analysis_eng_configured = False
        # if a['owner'].value < self.xanes_analysis_edge_eng:
        #     a['owner'].value = self.xanes_analysis_edge_eng
        #     self.hs['L[0][x][3][0][4][0]_analysis_run_text'].value = 'The whiteline fitting energy starting point might be too low. Reset it to the edge energy.'
        # elif (a['owner'].value+0.005) > self.hs['L[0][x][3][0][1][1][1]_analysis_energy_range_wl_fit_e_text'].value:
        #     a['owner'].value = self.hs['L[0][x][3][0][1][1][1]_analysis_energy_range_wl_fit_e_text'].value - 0.005
        #     self.hs['L[0][x][3][0][4][0]_analysis_run_text'].value = 'The whiteline fitting energy starting point might be too high. Reset it to 0.005keV lower than whiteline fitting energy ending point.'
        # else:
        #     self.hs['L[0][x][3][0][4][0]_analysis_run_text'].value ='please check your settings before run the analysis ...'
        self.hs['L[0][x][3][0][4][0]_analysis_run_text'].value ='please check your settings before run the analysis ...'
        self.parent_h.xanes_analysis_wl_fit_eng_s = self.hs['L[0][x][3][0][1][1][0]_analysis_energy_range_wl_fit_s_text'].value
        self.parent_h.boxes_logic()

    def L0_x_3_0_1_1_1_analysis_energy_range_wl_fit_e_text(self, a):
        self.parent_h.xanes_analysis_eng_configured = False
        # if a['owner'].value < (self.hs['L[0][x][3][0][1][1][0]_analysis_energy_range_wl_fit_s_text'].value + 0.005):
        #     a['owner'].value = (self.hs['L[0][x][3][0][1][1][0]_analysis_energy_range_wl_fit_s_text'].value + 0.005)
        #     self.hs['L[0][x][3][0][4][0]_analysis_run_text'].value = 'The whiteline fitting energy ending point might be too low. Reset it to 0.005keV higher than whiteline fitting energy starting point.'
        # else:
        #     self.hs['L[0][x][3][0][4][0]_analysis_run_text'].value ='please check your settings before run the analysis ...'
        self.hs['L[0][x][3][0][4][0]_analysis_run_text'].value ='please check your settings before run the analysis ...'
        self.parent_h.xanes_analysis_wl_fit_eng_e = self.hs['L[0][x][3][0][1][1][1]_analysis_energy_range_wl_fit_e_text'].value
        self.parent_h.boxes_logic()

    def L0_x_3_0_1_1_2_analysis_energy_range_edge0p5_fit_s_text(self, a):
        self.parent_h.xanes_analysis_eng_configured = False
        # if a['owner'].value > self.xanes_analysis_edge_eng:
        #     a['owner'].value = self.xanes_analysis_edge_eng
        #     self.hs['L[0][x][3][0][4][0]_analysis_run_text'].value = 'The edge-0.5 fitting energy starting point might be too high. Reset it to edge energy.'
        # else:
        #     self.hs['L[0][x][3][0][4][0]_analysis_run_text'].value ='please check your settings before run the analysis ...'
        self.hs['L[0][x][3][0][4][0]_analysis_run_text'].value ='please check your settings before run the analysis ...'
        self.parent_h.xanes_analysis_edge_0p5_fit_s = self.hs['L[0][x][3][0][1][1][2]_analysis_energy_range_edge0.5_fit_s_text'].value
        self.parent_h.boxes_logic()

    def L0_x_3_0_1_1_3_analysis_energy_range_edge0p5_fit_e_text(self, a):
        self.parent_h.xanes_analysis_eng_configured = False
        # if a['owner'].value < self.xanes_analysis_edge_eng:
        #     a['owner'].value = self.xanes_analysis_edge_eng
        #     self.hs['L[0][x][3][0][4][0]_analysis_run_text'].value = 'The edge-0.5 fitting energy ending point might be too low. Reset it to edge energy.'
        # else:
        #     self.hs['L[0][x][3][0][4][0]_analysis_run_text'].value ='please check your settings before run the analysis ...'
        self.hs['L[0][x][3][0][4][0]_analysis_run_text'].value ='please check your settings before run the analysis ...'
        self.parent_h.xanes_analysis_edge_0p5_fit_e = self.hs['L[0][x][3][0][1][1][3]_analysis_energy_range_edge0.5_fit_e_text'].value
        self.parent_h.boxes_logic()

    def L0_x_3_0_1_0_4_analysis_energy_range_confirm_button(self, a):
        if self.parent_h.xanes_analysis_type == 'wl':
            self.parent_h.xanes_analysis_wl_fit_eng_s = self.hs['L[0][x][3][0][1][1][0]_analysis_energy_range_wl_fit_s_text'].value
            self.parent_h.xanes_analysis_wl_fit_eng_e = self.hs['L[0][x][3][0][1][1][1]_analysis_energy_range_wl_fit_e_text'].value
        elif self.parent_h.xanes_analysis_type == 'full':
            self.parent_h.xanes_analysis_edge_eng = self.hs['L[0][x][3][0][1][0][1]_analysis_energy_range_edge_eng_text'].value
            self.parent_h.xanes_analysis_pre_edge_e = self.hs['L[0][x][3][0][1][0][2]_analysis_energy_range_pre_edge_e_text'].value
            self.parent_h.xanes_analysis_post_edge_s = self.hs['L[0][x][3][0][1][0][3]_analysis_energy_range_post_edge_s_text'].value
            self.parent_h.xanes_analysis_wl_fit_eng_s = self.hs['L[0][x][3][0][1][1][0]_analysis_energy_range_wl_fit_s_text'].value
            self.parent_h.xanes_analysis_wl_fit_eng_e = self.hs['L[0][x][3][0][1][1][1]_analysis_energy_range_wl_fit_e_text'].value
            self.parent_h.xanes_analysis_edge_0p5_fit_s = self.hs['L[0][x][3][0][1][1][2]_analysis_energy_range_edge0.5_fit_s_text'].value
            self.parent_h.xanes_analysis_edge_0p5_fit_e = self.hs['L[0][x][3][0][1][1][3]_analysis_energy_range_edge0.5_fit_e_text'].value
        if self.parent_h.xanes_analysis_spectrum is None:
            self.parent_h.xanes_analysis_spectrum = np.ndarray(self.parent_h.xanes_analysis_data_shape[1:], dtype=np.float32)
        self.parent_h.xanes_analysis_use_flt_spec = self.hs['L[0][x][3][0][1][0][4]_analysis_filter_spec_checkbox'].value
        
        for ii in ["gaussian", "lorentzian", "damped_oscillator",
                   "logistic", "lognormal", "students_t", "voigt",
                   "split_lorentzian", "pvoigt", "moffat", "pearson7",
                   "breit_wigner", "dho", "expgaussian", "donaich",
                   "skewed_gaussian", "step", "skewed_voigt"]:
            PEAK_FIT_PARAM_BND_DICT[ii][1][1] = (self.parent_h.xanes_analysis_wl_fit_eng_s +
                                            self.parent_h.xanes_analysis_wl_fit_eng_e)/2.
        
        # PEAK_FIT_PARAM_BND_DICT_ = {"exponential":{0: ["amp", [1e-3, 10], "amp: ampflitude in exponential function"],
        #                              1: ["dec", 0, "dec: center of exponential function"]},
        #               "powerlaw":{0: ["amp", 1, "amp: ampflitude in powerlaw function"],
        #                           1: ["expo", 0, "expo: center of powerlaw function"]},
        #               "linear":{0: ["slp", 1, "slp: ampflitude in linear function"],
        #                         1: ["incp", 0, "incp: center of linear function"]},
        #               "parabolic":{0: ["a", 1, "a: ampflitude in parabolic function"],
        #                            1: ["b", 0, "b: center of parabolic function"],
        #                            2: ["c", 1, "c: standard deviation of parabolic function"]},
        #               "gaussian":{0: ["amp", 1, "amp: ampflitude in gaussian function"],
        #                           1: ["cen", 0, "cen: center of gaussian function"],
        #                           2: ["sig", 1, "sig: standard deviation of gaussian function"]},
        #               "lorentzian":{0: ["amp", 1, "amp: ampflitude in lorentzian function"],
        #                             1: ["cen", 0, "cen: center of lorentzian function"],
        #                             2: ["sig", 1, "sig: standard deviation of lorentzian function"]},
        #               "damped_oscillator":{0: ["amp", 1, "amp: ampflitude in damped_oscillator function"],
        #                                    1: ["cen", 0, "cen: center of damped_oscillator function"],
        #                                    2: ["sig", 1, "sig: standard deviation of damped_oscillator function"]},
        #               "logistic":{0: ["amp", 1, "amp: ampflitude in logistic function"],
        #                           1: ["cen", 0, "cen: center of logistic function"],
        #                           2: ["sig", 1, "sig: standard deviation of logistic function"]},
        #               "lognormal":{0: ["amp", 1, "amp: ampflitude in lognormal function"],
        #                            1: ["cen", 0, "cen: center of lognormal function"],
        #                            2: ["sig", 1, "sig: standard deviation of lognormal function"]},
        #               "students_t":{0: ["amp", 1, "amp: ampflitude in students_t function"],
        #                             1: ["cen", 0, "cen: center of students_t function"],
        #                             2: ["sig", 1, "sig: standard deviation of students_t function"]},
        #               "sine":{0: ["amp", 1, "amp: ampflitude in sine function"],
        #                       1: ["frq", 0, "frq: freqency in sine function"],
        #                       2: ["shft", 1, "shft: shift in sine function"]},
        #               "voigt":{0: ["amp", 1, "amp: ampflitude in voigt function"],
        #                        1: ["cen", 0, "cen: center of voigt function"],
        #                        2: ["sig", 1, "sig: standard voigt of gaussian function"],
        #                        3: ["gamma", 0, "gamma: "]},
        #               "split_lorentzian":{0: ["amp", 1, "amp: ampflitude in split_lorentzian function"],
        #                                   1: ["cen", 0, "cen: center of split_lorentzian function"],
        #                                   2: ["sig", 1, "sig: standard deviation of split_lorentzian function"],
        #                                   3: ["sigr", 1, "sigr: standard deviation of the right-hand side half in split_lorentzian function"]},
        #               "pvoigt":{0: ["amp", 1, "amp: ampflitude in pvoigt function"],
        #                         1: ["cen", 0, "cen: center of pvoigt function"],
        #                         2: ["sig", 1, "sig: standard pvoigt of gaussian function"],
        #                         3: ["frac", 0, "frac: "]},
        #               "moffat":{0: ["amp", 1, "amp: ampflitude in moffat function"],
        #                         1: ["cen", 0, "cen: center of moffat function"],
        #                         2: ["sig", 1, "sig: standard moffat of gaussian function"],
        #                         3: ["beta", 0, "beta: "]},
        #               "pearson7": {0: ["amp", 1, "amp: ampflitude in pearson7 function"],
        #                            1: ["cen", 0, "cen: center of pearson7 function"],
        #                            2: ["sig", 1, "sig: standard pearson7 of gaussian function"],
        #                            3: ["expo", 0, "expo: "]},
        #               "breit_wigner":{0: ["amp", 1, "amp: ampflitude in breit_wigner function"],
        #                               1: ["cen", 0, "cen: center of breit_wigner function"],
        #                               2: ["sig", 1, "sig: standard breit_wigner of gaussian function"],
        #                               3: ["q", 0, "q: "]},
        #               "dho":{0: ["amp", 1, "amp: ampflitude in dho function"],
        #                      1: ["cen", 0, "cen: center of dho function"],
        #                      2: ["sig", 1, "sig: standard dho of gaussian function"],
        #                      3: ["gama", 1, "gama: "]},
        #               "expgaussian":{0: ["amp", 1, "amp: ampflitude in expgaussian function"],
        #                              1: ["cen", 0, "cen: center of expgaussian function"],
        #                              2: ["sig", 1, "sig: standard expgaussian of gaussian function"],
        #                              3: ["gama", 1, "gama: "]},
        #               "donaich":{0: ["amp", 1, "amp: ampflitude in donaich function"],
        #                          1: ["cen", 0, "cen: center of donaich function"],
        #                          2: ["sig", 1, "sig: standard donaich of gaussian function"],
        #                          3: ["gama", 0, "gama: "]},
        #               "skewed_gaussian":{0: ["amp", 1, "amp: ampflitude in skewed_gaussian function"],
        #                                  1: ["cen", 0, "cen: center of skewed_gaussian function"],
        #                                  2: ["sig", 1, "sig: standard skewed_gaussian of gaussian function"],
        #                                  3: ["gama", 0, "gama: "]},
        #               "expsine":{0: ["amp", 1, "amp: ampflitude in expsine function"],
        #                          1: ["frq", 0, "frq: center of expsine function"],
        #                          2: ["shft", 1, "shft: standard expsine of gaussian function"],
        #                          3: ["dec", 0, "dec: "]},
        #               "step":{0: ["amp", 1, "amp: ampflitude in step function"],
        #                       1: ["cen", 0, "cen: center of step function"],
        #                       2: ["sig", 1, "sig: standard step of gaussian function"],
        #                       3: ["form", "linear", "form: "]},
        #               "skewed_voigt":{0: ["amp", 1, "amp: ampflitude in skewed_voigt function"],
        #                               1: ["cen", 0, "cen: center of skewed_voigt function"],
        #                               2: ["sig", 1, "sig: standard skewed_voigt of gaussian function"],
        #                               3: ["gamma", 0, "gamma: "],
        #                               4: ["skew", 0, "skew: "]},
        #               "rectangle":{0: ["amp", 1, "amp: ampflitude in rectangle function"],
        #                            1: ["cen1", 0, "cen1: center of rectangle function"],
        #                            2: ["sig1", 1, "sig1: standard deviation of rectangle function"],
        #                            3: ["cen2", 0, "cen2: center of rectangle function"],
        #                            4: ["sig2", 1, "sig2: standard deviation of rectangle function"],
        #                            5: ["form", "linear", "form: "]}}
        
        self.parent_h.xanes_analysis_eng_configured = True
        self.parent_h.boxes_logic()

    def L0_x_3_0_2_0_analysis_energy_filter_edge_jump_thres_slider_change(self, a):
        self.parent_h.xanes_analysis_edge_jump_thres = a['owner'].value
        self.parent_h.boxes_logic()

    def L0_x_3_0_2_1_analysis_energy_filter_edge_offset_slider_change(self, a):
        self.parent_h.xanes_analysis_edge_offset_thres = a['owner'].value
        self.parent_h.boxes_logic()

    def L0_x_3_0_3_0_analysis_image_use_mask_checkbox(self, a):
        pass
        # if a['owner'].value:
        #     self.parent_h.xanes_analysis_use_mask = True
        #     self.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'].value = 'x-y-z'
        #     # f = h5py.File(self.fn, 'r')
        #     with h5py.File(self.fn, 'r') as f:
        #         self.parent_h.xanes_aligned_data = 0
        #         self.parent_h.xanes_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][0, :, :, :]
        #     # f.close()
        #     self.hs['L[0][x][3][0][3][1]_analysis_image_mask_scan_id_slider'].max = self.xanes_scan_id_e
        #     self.hs['L[0][x][3][0][3][1]_analysis_image_mask_scan_id_slider'].value = self.xanes_scan_id_s
        #     self.hs['L[0][x][3][0][3][1]_analysis_image_mask_scan_id_slider'].min = self.xanes_scan_id_s
        #     if self.parent_h.xanes_analysis_mask == 1:
        #         self.parent_h.xanes_analysis_mask = (self.parent_h.xanes_aligned_data>self.parent_h.xanes_analysis_mask_thres).astype(np.int8)
        # else:
        #     self.parent_h.xanes_analysis_use_mask = False
        # self.parent_h.boxes_logic()

    def L0_x_3_0_3_1_analysis_image_mask_scan_id_slider(self, a):
        pass
        # self.parent_h.xanes_analysis_mask_scan_id = a['owner'].value
        # self.parent_h.xanes_analysis_mask_thres = self.hs['L[0][x][3][0][3][1]_analysis_image_mask_thres_slider'].value

        # data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_analysis_viewer')
        # if not viewer_state:
        #     fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_analysis_viewer')

        # # f = h5py.File(self.fn, 'r')
        # with h5py.File(self.fn, 'r') as f:
        #     self.parent_h.xanes_aligned_data[:] = f['/registration_results/reg_results/registered_xanes3D'][self.parent_h.xanes_analysis_mask_scan_id-self.xanes_scan_id_s, :, :, :]
        # # f.close()
        # self.parent_h.xanes_analysis_mask[:] = (self.parent_h.xanes_aligned_data>self.parent_h.xanes_analysis_mask_thres).astype(np.int8)[:]
        # self.parent_h.xanes_aligned_data[:] = (self.parent_h.xanes_aligned_data*self.parent_h.xanes_analysis_mask)[:]
        # self.parent_h.xanes_fiji_aligned_data = self.global_h.ij.convert().convert(self.global_h.ij.dataset().create(
        #     self.global_h.ij.py.to_java(self.parent_h.xanes_aligned_data)), self.global_h.ImagePlusClass)
        # self.global_h.xanes_fiji_windows['xanes3D_analysis_viewer']['ip'].setImage(self.parent_h.xanes_fiji_aligned_data)
        # self.global_h.xanes_fiji_windows['xanes3D_analysis_viewer']['ip'].show()
        # self.global_h.xanes_fiji_windows['xanes3D_analysis_viewer']['ip'].setTitle(f"{a['owner'].description} slice: {a['owner'].value}")
        # self.global_h.ij.py.run_macro("""run("Collect Garbage")""")
        # self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        # self.parent_h.boxes_logic()

    def L0_x_3_0_3_1_analysis_image_mask_thres_slider(self, a):
        pass
        # self.parent_h.xanes_analysis_mask_thres = a['owner'].value
        # self.parent_h.xanes_analysis_mask_scan_id = self.hs['L[0][x][3][0][3][1]_analysis_image_mask_scan_id_slider'].value

        # data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_analysis_viewer')
        # if not viewer_state:
        #     fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_analysis_viewer')

        # # f = h5py.File(self.fn, 'r')
        # with h5py.File(self.fn, 'r') as f:
        #     self.parent_h.xanes_aligned_data[:] = f['/registration_results/reg_results/registered_xanes3D'][self.parent_h.xanes_analysis_mask_scan_id-self.xanes_scan_id_s, :, :, :]
        # # f.close()
        # self.parent_h.xanes_analysis_mask[:] = (self.parent_h.xanes_aligned_data>self.parent_h.xanes_analysis_mask_thres).astype(np.int8)[:]
        # self.parent_h.xanes_aligned_data[:] = (self.parent_h.xanes_aligned_data*self.parent_h.xanes_analysis_mask)[:]
        # self.parent_h.xanes_fiji_aligned_data = self.global_h.ij.convert().convert(self.global_h.ij.dataset().create(
        #     self.global_h.ij.py.to_java(self.parent_h.xanes_aligned_data)), self.global_h.ImagePlusClass)
        # self.global_h.xanes_fiji_windows['xanes3D_analysis_viewer']['ip'].setImage(self.parent_h.xanes_fiji_aligned_data)
        # self.global_h.xanes_fiji_windows['xanes3D_analysis_viewer']['ip'].show()
        # self.global_h.xanes_fiji_windows['xanes3D_analysis_viewer']['ip'].setTitle(f"{a['owner'].description} slice: {a['owner'].value}")
        # self.global_h.ij.py.run_macro("""run("Collect Garbage")""")
        # self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        # self.parent_h.boxes_logic()
        
    def L0_x_3_0_4_0_0_0_analysis_gridlayout_optimizer(self, a):
        self.analysis_optimizer = a['owner'].value
        if self.analysis_optimizer == 'scipy':
            for ii in self.analysis_fit_func_arg_handles:
                ii.disabled = False
            for ii in self.analysis_optimizer_arg_handles:
                ii.disabled = False
            if self.analysis_fit_use_param_bnd:
                for ii in self.analysis_fit_func_bnd_handles:
                    ii.disabled = False
            else:
                for ii in self.analysis_fit_func_bnd_handles:
                    ii.disabled = True
            self.hs['L[0][x][3][0][4][0][0][2]_analysis_gridlayout_fit_bnd'].disabled = False
            self.hs['L[0][x][3][0][4][0][0][1]_analysis_gridlayout_fit_func'].options = PEAK_LINE_SHAPES
            self.hs['L[0][x][3][0][4][0][0][1]_analysis_gridlayout_fit_func'].description = 'peak func'
            self.hs['L[0][x][3][0][4][0][0][1]_analysis_gridlayout_fit_func'].description_tooltip = 'peak fitting functions'
        elif self.analysis_optimizer == 'numpy':
            for ii in self.analysis_fit_func_arg_handles:
                ii.disabled = True
            for ii in self.analysis_optimizer_arg_handles:
                ii.disabled = True
            for ii in self.analysis_fit_func_bnd_handles:
                ii.disabled = True
            self.hs['L[0][x][3][0][4][0][0][2]_analysis_gridlayout_fit_bnd'].disabled = True
            self.hs['L[0][x][3][0][4][0][0][1]_analysis_gridlayout_fit_func'].options = NUMPY_FIT_ORDER
            self.hs['L[0][x][3][0][4][0][0][1]_analysis_gridlayout_fit_func'].description = 'order'
            self.hs['L[0][x][3][0][4][0][0][1]_analysis_gridlayout_fit_func'].description_tooltip = 'order of polynominal fitting function'
                
    def L0_x_3_0_4_0_0_1_analysis_gridlayout_fit_func(self, a):
        self.analysis_method = a['owner'].value
        cnt = 0
        for ii in self.analysis_fit_func_arg_handles:
                ii.disabled = True
                ii.description = 'p'+str(cnt)
                cnt += 1
        if self.analysis_optimizer == 'scipy':
            for ii in sorted(PEAK_FIT_PARAM_DICT[self.analysis_method].keys()):
                self.analysis_fit_func_arg_handles[ii].disabled = False
                self.analysis_fit_func_arg_handles[ii].description = \
                    PEAK_FIT_PARAM_DICT[self.analysis_method][ii][0]
                self.analysis_fit_func_arg_handles[ii].value = \
                    PEAK_FIT_PARAM_DICT[self.analysis_method][ii][1]
                self.analysis_fit_func_arg_handles[ii].description_tooltip = \
                    PEAK_FIT_PARAM_DICT[self.analysis_method][ii][2]
        if self.analysis_method == "rectangle":
            self.analysis_fit_func_arg_handles[5].options = \
                    ['linear', 'atan', 'erf', 'logisitic']
        if self.analysis_method == "step":
            self.analysis_fit_func_arg_handles[5].options = \
                    ['linear', 'atan', 'erf', 'logisitic']
        
        # if self.analysis_method == "exponential":
        #     self.analysis_fit_func_arg_handles[0].description = 'amp'
        # elif self.analysis_method == "powerlaw":
        #     pass
        # elif self.analysis_method == "parabolic":
        #     pass
        # elif self.analysis_method == 'gaussian':
        #     pass
        # elif self.analysis_method == 'lorentzian':
        #     pass
        # elif self.analysis_method == "damped_oscillator":
        #     pass
        # elif self.analysis_method == "logistic":
        #     pass
        # elif self.analysis_method == "students_t":
        #     pass
        # elif self.analysis_method == "sine":
        #     pass
        # elif self.analysis_method == "voigt":
        #     pass
        # elif self.analysis_method == "split_lorentzian":
        #     pass
        # elif self.analysis_method == "pvoigt":
        #     pass
        # elif self.analysis_method == "moffat":
        #     pass
        # elif self.analysis_method == "pearson7":
        #     pass
        # elif self.analysis_method == "breit_wigner":
        #     pass
        # elif self.analysis_method == "dho":
        #     pass
        # elif self.analysis_method == "expgaussian":
        #     pass
        # elif self.analysis_method == "donaich":
        #     pass
        # elif self.analysis_method == "skewed_gaussian":
        #     pass
        # elif self.analysis_method == "expsine":
        #     pass
        # elif self.analysis_method == "step":
        #     pass
        # elif self.analysis_method == "skewed_voigt":
        #     pass
        # elif self.analysis_method == "rectangle":
        #     pass
    
    def L0_x_3_0_4_0_0_2_analysis_gridlayout_fit_bnd(self, a):
        self.analysis_fit_use_param_bnd = a['owner'].value
        if self.analysis_fit_use_param_bnd:
            for ii in self.analysis_fit_func_bnd_handles:
                ii.disabled = False
            for ii in ["gaussian", "lorentzian", "damped_oscillator",
                   "logistic", "lognormal", "students_t", "voigt",
                   "split_lorentzian", "pvoigt", "moffat", "pearson7",
                   "breit_wigner", "dho", "expgaussian", "donaich",
                   "skewed_gaussian", "step", "skewed_voigt"]:
                PEAK_FIT_PARAM_BND_DICT[ii][1][1] = (self.parent_h.xanes_analysis_wl_fit_eng_s +
                                                self.parent_h.xanes_analysis_wl_fit_eng_e)/2.
        else:
            for ii in self.analysis_fit_func_bnd_handles:
                ii.disabled = True

    def L0_x_3_0_5_1_analysis_run_button(self, a):
        try:
            fiji_viewer_off(self.parent_h.global_h, self, viewer_name='all')
        except:
            pass

        if self.analysis_fit_use_param_bnd :
            bnds = []
        else:
            bnds = None
        if self.parent_h.gui_name == 'xanes3D':
            with h5py.File(self.fn, 'r+') as f:
                if 'processed_XANES3D' not in f:
                    g1 = f.create_group('processed_XANES3D')
                else:
                    del f['processed_XANES3D']
                    g1 = f.create_group('processed_XANES3D')
                g11 = g1.create_group('proc_parameters')
                g11.create_dataset('element', data=str(self.parent_h.xanes_element))
                g11.create_dataset('eng_list', 
                                   data=scale_eng_list(self.parent_h.xanes_analysis_eng_list).astype(np.float32))
                g11.create_dataset('edge_eng', 
                                   data=self.parent_h.xanes_analysis_edge_eng)
                g11.create_dataset('pre_edge_e', 
                                   data=self.parent_h.xanes_analysis_pre_edge_e)
                g11.create_dataset('post_edge_s', 
                                   data=self.parent_h.xanes_analysis_post_edge_s)
                g11.create_dataset('edge_jump_threshold', 
                                   data=self.parent_h.xanes_analysis_edge_jump_thres)
                g11.create_dataset('edge_offset_threshold', 
                                   data=self.parent_h.xanes_analysis_edge_offset_thres)
                g11.create_dataset('use_mask', 
                                   data=str(self.parent_h.xanes_analysis_use_mask))
                g11.create_dataset('analysis_type', 
                                   data=self.parent_h.xanes_analysis_type)
                g11.create_dataset('data_shape', 
                                   data=self.parent_h.xanes_analysis_data_shape)
                g11.create_dataset('edge_0p5_fit_s', 
                                   data=self.parent_h.xanes_analysis_edge_0p5_fit_s)
                g11.create_dataset('edge_0p5_fit_e', 
                                   data=self.parent_h.xanes_analysis_edge_0p5_fit_e)
                g11.create_dataset('wl_fit_eng_s', 
                                   data=self.parent_h.xanes_analysis_wl_fit_eng_s)
                g11.create_dataset('wl_fit_eng_e', 
                                   data=self.parent_h.xanes_analysis_wl_fit_eng_e)
                g11.create_dataset('normalized_fitting_order', data=1)
                g11.create_dataset('flt_spec', 
                                   data=str(self.parent_h.xanes_analysis_use_flt_spec))
            
            code = {}
            ln = 0
            code[ln] = "import os, h5py"; ln+=1
            code[ln] = "import numpy as np"; ln += 1
            code[ln] = "import xanes_math as xm"; ln += 1
            code[ln] = "import xanes_analysis as xa"; ln += 1
            code[ln] = ""; ln += 1
            code[ln] = f"with h5py.File('{self.fn}', 'r+') as f:"; ln += 1
            code[ln] = "    imgs = f['/registration_results/reg_results/registered_xanes3D'][:, 0, :, :]"; ln += 1
            code[ln] = "    xanes3D_analysis_eng_list = f['/processed_XANES3D/proc_parameters/eng_list'][:]"; ln += 1
            code[ln] = "    xanes3D_analysis_edge_eng = f['/processed_XANES3D/proc_parameters/edge_eng'][()]"; ln += 1
            code[ln] = "    xanes3D_analysis_pre_edge_e = f['/processed_XANES3D/proc_parameters/pre_edge_e'][()]"; ln += 1
            code[ln] = "    xanes3D_analysis_post_edge_s = f['/processed_XANES3D/proc_parameters/post_edge_s'][()]"; ln += 1
            code[ln] = "    xanes3D_analysis_edge_jump_thres = f['/processed_XANES3D/proc_parameters/edge_jump_threshold'][()]"; ln += 1
            code[ln] = "    xanes3D_analysis_edge_offset_thres = f['/processed_XANES3D/proc_parameters/edge_offset_threshold'][()]"; ln += 1
            code[ln] = "    xanes3D_analysis_use_mask = f['/processed_XANES3D/proc_parameters/use_mask'][()]"; ln += 1
            code[ln] = "    xanes3D_analysis_type = f['/processed_XANES3D/proc_parameters/analysis_type'][()]"; ln += 1
            code[ln] = "    xanes3D_analysis_data_shape = f['/processed_XANES3D/proc_parameters/data_shape'][:]"; ln += 1
            code[ln] = "    xanes3D_analysis_edge_0p5_fit_s = f['/processed_XANES3D/proc_parameters/edge_0p5_fit_s'][()]"; ln += 1
            code[ln] = "    xanes3D_analysis_edge_0p5_fit_e = f['/processed_XANES3D/proc_parameters/edge_0p5_fit_e'][()]"; ln += 1
            code[ln] = "    xanes3D_analysis_wl_fit_eng_s = f['/processed_XANES3D/proc_parameters/wl_fit_eng_s'][()]"; ln += 1
            code[ln] = "    xanes3D_analysis_wl_fit_eng_e = f['/processed_XANES3D/proc_parameters/wl_fit_eng_e'][()]"; ln += 1
            code[ln] = "    xanes3D_analysis_norm_fitting_order = f['/processed_XANES3D/proc_parameters/normalized_fitting_order'][()]"; ln += 1
            code[ln] = "    xanes3D_analysis_use_flt_spec = f['/processed_XANES3D/proc_parameters/flt_spec'][()]"; ln += 1
            code[ln] = "    xana = xa.xanes_analysis(imgs, xanes3D_analysis_eng_list, xanes3D_analysis_edge_eng, pre_ee=xanes3D_analysis_pre_edge_e, post_es=xanes3D_analysis_post_edge_s, edge_jump_threshold=xanes3D_analysis_edge_jump_thres, pre_edge_threshold=xanes3D_analysis_edge_offset_thres)"; ln += 1
            code[ln] = "    if '/processed_XANES3D/proc_spectrum' in f:"; ln += 1
            code[ln] = "        del f['/processed_XANES3D/proc_spectrum']"; ln += 1
            code[ln] = "        g12 = f.create_group('/processed_XANES3D/proc_spectrum')"; ln += 1
            code[ln] = "    else:"; ln += 1
            code[ln] = "        g12 = f.create_group('/processed_XANES3D/proc_spectrum')"; ln += 1
            code[ln] = ""; ln += 1
            code[ln] = "    if xanes3D_analysis_type == 'wl':"; ln += 1
            code[ln] = "        g120 = g12.create_dataset('spectrum_whiteline_pos', shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = "        g121 = g12.create_dataset('spectrum_whiteline_pos_direct', shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = f"        g12.create_dataset('img_mask', data={np.int8(self.parent_h.xanes_analysis_mask)}, dtype=np.int8)"; ln += 1
            code[ln] = "        for ii in range(xanes3D_analysis_data_shape[1]):"; ln += 1
            code[ln] = "            imgs[:] = f['/registration_results/reg_results/registered_xanes3D'][:, ii, :, :]"; ln += 1
            code[ln] = "            xana.spectrum[:] = imgs[:]"; ln += 1
            code[ln] = "            xana.fit_whiteline(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, model='lorentzian', fvars=(0.5, xanes3D_analysis_wl_fit_eng_s, 1), bnds=None, ftol=1e-7, xtol=1e-7, gtol=1e-7, jac='3-point', method='trf', ufac=100)"; ln+=1
            code[ln] = "            xana.calc_direct_whiteline(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e)"; ln += 1
            code[ln] = "            g120[ii] = np.float32(xana.wl_pos)[:]"; ln += 1
            code[ln] = "            g121[ii] = np.float32(xana.wl_pos_direct)[:]"; ln += 1
            code[ln] = "            print(ii)"; ln += 1
            code[ln] = "    else:"; ln += 1
            code[ln] = "        g120 = g12.create_dataset('spectrum_whiteline_pos', shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = "        g121 = g12.create_dataset('spectrum_whiteline_pos_direct', shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = "        g122 = g12.create_dataset('spectrum_direct_whiteline_peak_height', shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = "        g123 = g12.create_dataset('spectrum_centroid_of_eng', shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = "        g124 = g12.create_dataset('spectrum_centroid_of_eng_relative_to_wl', shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = "        g125 = g12.create_dataset('spectrum_weighted_atten', shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = "        g126 = g12.create_dataset('spectrum_weighted_eng', shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = "        g127 = g12.create_dataset('spectrum_edge0.5_pos', shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = "        g128 = g12.create_dataset('spectrum_edge_jump_filter', shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = "        g129 = g12.create_dataset('spectrum_edge_offset_filter', shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln += 1
            code[ln] = "        g1210 = g12.create_dataset('spectrum_pre_edge_sd', shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = "        g1211 = g12.create_dataset('spectrum_pre_edge_mean', shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = "        g1212 = g12.create_dataset('spectrum_post_edge_sd', shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = "        g1213 = g12.create_dataset('spectrum_post_edge_mean', shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = f"        g12.create_dataset('img_mask', data={np.int8(self.parent_h.xanes_analysis_mask)}, dtype=np.int8)"; ln += 1
            code[ln] = "        for ii in range(xanes3D_analysis_data_shape[1]):"; ln += 1
            code[ln] = "            imgs[:] = f['/registration_results/reg_results/registered_xanes3D'][:, ii, :, :]"; ln += 1
            code[ln] = "            xana.spectrum[:] = imgs[:]"; ln += 1        
            code[ln] = "            xana.fit_pre_edge()"; ln+=1
            code[ln] = "            xana.fit_post_edge()"; ln+=1
            code[ln] = "            xana.cal_edge_jump_map()"; ln+=1
            code[ln] = "            xana.cal_pre_edge_sd()"; ln+=1
            code[ln] = "            xana.cal_post_edge_sd()"; ln+=1
            code[ln] = "            xana.cal_pre_edge_mean()"; ln+=1
            code[ln] = "            xana.cal_post_edge_mean()"; ln+=1
            code[ln] = "            xana.create_edge_jump_filter(xanes3D_analysis_edge_jump_thres)"; ln+=1
            code[ln] = "            xana.create_fitted_edge_filter(xanes3D_analysis_edge_offset_thres)"; ln+=1
            code[ln] = "            xana.normalize_xanes(xanes3D_analysis_edge_eng, order=xanes3D_analysis_norm_fitting_order)"; ln += 1
            code[ln] = "            xana.fit_edge_pos(xanes3D_analysis_edge_0p5_fit_s, xanes3D_analysis_edge_0p5_fit_e, order=3, flt_spec=xanes3D_analysis_use_flt_spec)"; ln += 1
            code[ln] = "            xana.fit_whiteline(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, model='lorentzian', fvars=(0.5, xanes3D_analysis_wl_fit_eng_s, 1), bnds=None, ftol=1e-7, xtol=1e-7, gtol=1e-7, jac='3-point', method='trf', ufac=100)"; ln+=1
            code[ln] = "            xana.calc_direct_whiteline_peak_height(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e)"; ln+=1
            code[ln] = "            xana.calc_direct_whiteline(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e)"; ln += 1
            code[ln] = "            xana.calc_weighted_eng(xanes3D_analysis_pre_edge_e)"; ln += 1
            code[ln] = "            g120[ii] = np.float32(xana.wl_pos)[:]"; ln += 1
            code[ln] = "            g121[ii] = np.float32(xana.wl_pos_direct)[:]"; ln += 1
            code[ln] = "            g122[ii] = np.float32(xana.direct_wl_ph)[:]"; ln += 1
            code[ln] = "            g123[ii] = np.float32(xana.centroid_of_eng)[:]"; ln += 1
            code[ln] = "            g124[ii] = np.float32(xana.centroid_of_eng_rel_wl)[:]"; ln += 1
            code[ln] = "            g125[ii] = np.float32(xana.weighted_atten)[:]"; ln += 1
            code[ln] = "            g126[ii] = np.float32(xana.weighted_eng)[:]"; ln += 1
            code[ln] = "            g127[ii] = np.float32(xana.edge_pos)[:]"; ln += 1
            code[ln] = "            g128[ii] = np.float32(xana.edge_jump_mask)[:]"; ln += 1
            code[ln] = "            g129[ii] = np.float32(xana.fitted_edge_mask)[:]"; ln += 1
            code[ln] = "            g1210[:] = np.float32(xana.pre_edge_sd_map)[:]"; ln+=1
            code[ln] = "            g1211[:] = np.float32(xana.post_edge_sd_map)[:]"; ln+=1
            code[ln] = "            g1212[:] = np.float32(xana.pre_edge_mean_map)[:]"; ln+=1
            code[ln] = "            g1213[:] = np.float32(xana.post_edge_mean_map)[:]"; ln+=1
            code[ln] = "            print(str(ii)+'/'+str(xanes3D_analysis_data_shape[1]))"; ln += 1
    
            gen_external_py_script(self.parent_h.xanes_external_command_name, code)
            sig = os.system(f'python {self.parent_h.xanes_external_command_name}')
            if sig == 0:
                self.hs['L[0][x][3][0][4][0]_analysis_run_text'].value = 'XANES3D analysis is done ...'
            else:
                self.hs['L[0][x][3][0][4][0]_analysis_run_text'].value = 'something wrong in analysis ...'
            self.parent_h.update_xanes3D_config()
        elif self.parent_h.gui_name == 'xanes2D':
            with h5py.File(self.xanes2D_file_save_trial_reg_filename, 'r+') as f:
                if 'processed_XANES2D' not in f:
                    g1 = f.create_group('processed_XANES2D')
                else:
                    del f['processed_XANES2D']
                    g1 = f.create_group('processed_XANES2D')
                g11 = g1.create_group('proc_parameters')
                g11.create_dataset('eng_list', data=self.xanes2D_analysis_eng_list*1000)
                g11.create_dataset('edge_eng', data=self.xanes2D_analysis_edge_eng)
                g11.create_dataset('pre_edge_e', data=self.xanes2D_analysis_pre_edge_e)
                g11.create_dataset('post_edge_s', data=self.xanes2D_analysis_post_edge_s)
                g11.create_dataset('edge_jump_threshold', data=self.xanes2D_analysis_edge_jump_thres)
                g11.create_dataset('edge_offset_threshold', data=self.xanes2D_analysis_edge_offset_thres)
                g11.create_dataset('use_mask', data=str(self.xanes2D_analysis_use_mask))
                g11.create_dataset('analysis_type', data=self.xanes2D_analysis_type)
                g11.create_dataset('data_shape', data=self.xanes2D_img_roi.shape)
                g11.create_dataset('edge_0p5_fit_s', data=self.xanes2D_analysis_edge_0p5_fit_s)
                g11.create_dataset('edge_0p5_fit_e', data=self.xanes2D_analysis_edge_0p5_fit_e)
                g11.create_dataset('wl_fit_eng_s', data=self.xanes2D_analysis_wl_fit_eng_s)
                g11.create_dataset('wl_fit_eng_e', data=self.xanes2D_analysis_wl_fit_eng_e)
                g11.create_dataset('normalized_fitting_order', data=0)
                g11.create_dataset('flt_spec', data=str(self.xanes2D_analysis_use_flt_spec))
            
            code = {}
            ln = 0
            code[ln] = "import os, h5py"; ln+=1
            code[ln] = "import numpy as np"; ln+=1
            code[ln] = "import xanes_math as xm"; ln+=1
            code[ln] = "import xanes_analysis as xa"; ln+=1
            code[ln] = ""; ln+=1
            code[ln] = f"with h5py.File('{self.fn}', 'r+') as f:"; ln+=1
            code[ln] = "    imgs = f['/registration_results/reg_results/registered_xanes2D'][:]"; ln+=1
            code[ln] = "    xanes2D_analysis_eng_list = f['/processed_XANES2D/proc_parameters/eng_list'][:]"; ln+=1
            code[ln] = "    xanes2D_analysis_edge_eng = f['/processed_XANES2D/proc_parameters/edge_eng'][()]"; ln+=1
            code[ln] = "    xanes2D_analysis_pre_edge_e = f['/processed_XANES2D/proc_parameters/pre_edge_e'][()]"; ln+=1
            code[ln] = "    xanes2D_analysis_post_edge_s = f['/processed_XANES2D/proc_parameters/post_edge_s'][()]"; ln+=1
            code[ln] = "    xanes2D_analysis_edge_jump_thres = f['/processed_XANES2D/proc_parameters/edge_jump_threshold'][()]"; ln+=1
            code[ln] = "    xanes2D_analysis_edge_offset_thres = f['/processed_XANES2D/proc_parameters/edge_offset_threshold'][()]"; ln+=1
            code[ln] = "    xanes2D_analysis_use_mask = f['/processed_XANES2D/proc_parameters/use_mask'][()]"; ln+=1
            code[ln] = "    xanes2D_analysis_type = f['/processed_XANES2D/proc_parameters/analysis_type'][()]"; ln+=1
            code[ln] = "    xanes2D_analysis_data_shape = f['/processed_XANES2D/proc_parameters/data_shape'][:]"; ln+=1
            code[ln] = "    xanes2D_analysis_edge_0p5_fit_s = f['/processed_XANES2D/proc_parameters/edge_0p5_fit_s'][()]"; ln+=1
            code[ln] = "    xanes2D_analysis_edge_0p5_fit_e = f['/processed_XANES2D/proc_parameters/edge_0p5_fit_e'][()]"; ln+=1
            code[ln] = "    xanes2D_analysis_wl_fit_eng_s = f['/processed_XANES2D/proc_parameters/wl_fit_eng_s'][()]"; ln+=1
            code[ln] = "    xanes2D_analysis_wl_fit_eng_e = f['/processed_XANES2D/proc_parameters/wl_fit_eng_e'][()]"; ln+=1
            code[ln] = "    xanes2D_analysis_use_flt_spec = f['/processed_XANES2D/proc_parameters/flt_spec'][()]"; ln+=1
            code[ln] = "    xanes2D_analysis_norm_fitting_order = f['/processed_XANES2D/proc_parameters/normalized_fitting_order'][()]"; ln+=1
            code[ln] = "    xana = xa.xanes_analysis(imgs, xanes2D_analysis_eng_list, xanes2D_analysis_edge_eng, pre_ee=xanes2D_analysis_pre_edge_e, post_es=xanes2D_analysis_post_edge_s, edge_jump_threshold=xanes2D_analysis_edge_jump_thres, pre_edge_threshold=xanes2D_analysis_edge_offset_thres)"; ln+=1
            code[ln] = "    if '/processed_XANES2D/proc_spectrum' in f:"; ln+=1
            code[ln] = "        del f['/processed_XANES2D/proc_spectrum']"; ln+=1
            code[ln] = "        g12 = f.create_group('/processed_XANES2D/proc_spectrum')"; ln+=1
            code[ln] = "    else:"; ln+=1
            code[ln] = "        g12 = f.create_group('/processed_XANES2D/proc_spectrum')"; ln+=1
            code[ln] = ""; ln+=1
            code[ln] = "    if xanes2D_analysis_type == 'wl':"; ln+=1
            code[ln] = "        g120 = g12.create_dataset('spectrum_whiteline_pos', shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = "        g121 = g12.create_dataset('spectrum_whiteline_pos_direct', shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = f"        g12.create_dataset('img_mask', data={np.int8(self.parent_h.xanes2D_analysis_mask)}, dtype=np.int8)"; ln+=1
            code[ln] = "        xana.fit_whiteline(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, model='lorentzian', fvars=(0.5, xanes2D_analysis_wl_fit_eng_s, 1), bnds=None, ftol=1e-7, xtol=1e-7, gtol=1e-7, jac='3-point', method='trf', ufac=100)"; ln+=1
            code[ln] = "        xana.calc_direct_whiteline(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e)"; ln+=1
            code[ln] = "        g120[0] = xana.wl_pos[:]"; ln+=1
            code[ln] = "        g121[0] = xana.wl_pos_direct[:]"; ln+=1
            code[ln] = ""; ln+=1
            code[ln] = ""; ln+=1
            code[ln] = "    else:"; ln+=1
            code[ln] = "        g120 = g12.create_dataset('spectrum_whiteline_pos', shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = "        g121 = g12.create_dataset('spectrum_whiteline_pos_direct', shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = "        g122 = g12.create_dataset('spectrum_direct_whiteline_peak_height', shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = "        g123 = g12.create_dataset('spectrum_centroid_of_eng', shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = "        g124 = g12.create_dataset('spectrum_centroid_of_eng_relative_to_wl', shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = "        g125 = g12.create_dataset('spectrum_weighted_atten', shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = "        g126 = g12.create_dataset('spectrum_weighted_eng', shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = "        g127 = g12.create_dataset('spectrum_edge0.5_pos', shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = "        g128 = g12.create_dataset('spectrum_edge_jump_filter', shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = "        g129 = g12.create_dataset('spectrum_edge_offset_filter', shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = "        g1210 = g12.create_dataset('spectrum_pre_edge_sd', shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = "        g1211 = g12.create_dataset('spectrum_pre_edge_mean', shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = "        g1212 = g12.create_dataset('spectrum_post_edge_sd', shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = "        g1213 = g12.create_dataset('spectrum_post_edge_mean', shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"; ln+=1
            code[ln] = f"        g12.create_dataset('img_mask', data={np.int8(self.parent_h.xanes2D_analysis_mask)}, dtype=np.int8)"; ln+=1
            code[ln] = "        xana.fit_pre_edge()"; ln+=1
            code[ln] = "        xana.fit_post_edge()"; ln+=1
            code[ln] = "        xana.cal_edge_jump_map()"; ln+=1
            code[ln] = "        xana.cal_pre_edge_sd()"; ln+=1
            code[ln] = "        xana.cal_post_edge_sd()"; ln+=1
            code[ln] = "        xana.cal_pre_edge_mean()"; ln+=1
            code[ln] = "        xana.cal_post_edge_mean()"; ln+=1
            code[ln] = "        xana.create_edge_jump_filter(xanes2D_analysis_edge_jump_thres)"; ln+=1
            code[ln] = "        xana.create_fitted_edge_filter(xanes2D_analysis_edge_offset_thres)"; ln+=1
            code[ln] = "        xana.normalize_xanes(xanes2D_analysis_edge_eng, order=xanes2D_analysis_norm_fitting_order)"; ln+=1
            code[ln] = "        xana.fit_edge_pos(xanes2D_analysis_edge_0p5_fit_s, xanes2D_analysis_edge_0p5_fit_e, order=3, flt_spec=xanes2D_analysis_use_flt_spec)"; ln+=1
            code[ln] = "        xana.fit_whiteline(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, model='lorentzian', fvars=(0.5, xanes2D_analysis_wl_fit_eng_s, 1), bnds=None, ftol=1e-7, xtol=1e-7, gtol=1e-7, jac='3-point', method='trf', ufac=100)"; ln+=1
            code[ln] = "        xana.calc_direct_whiteline(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e)"; ln+=1
            code[ln] = "        xana.calc_direct_whiteline_peak_height(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e)"; ln+=1
            code[ln] = "        xana.calc_weighted_eng(xanes2D_analysis_pre_edge_e)"; ln+=1
            code[ln] = "        g120[:] = np.float32(xana.wl_pos)[:]"; ln+=1
            code[ln] = "        g121[:] = np.float32(xana.wl_pos_direct)[:]"; ln+=1
            code[ln] = "        g122[:] = np.float32(xana.direct_wl_ph)[:]"; ln+=1
            code[ln] = "        g123[:] = np.float32(xana.centroid_of_eng)[:]"; ln+=1
            code[ln] = "        g124[:] = np.float32(xana.centroid_of_eng_rel_wl)[:]"; ln+=1
            code[ln] = "        g125[:] = np.float32(xana.weighted_atten)[:]"; ln+=1
            code[ln] = "        g126[:] = np.float32(xana.weighted_eng)[:]"; ln+=1
            code[ln] = "        g127[:] = np.float32(xana.edge_pos)[:]"; ln+=1
            code[ln] = "        g128[:] = np.float32(xana.edge_jump_mask)[:]"; ln+=1
            code[ln] = "        g129[:] = np.float32(xana.fitted_edge_mask)[:]"; ln+=1
            code[ln] = "        g1210[:] = np.float32(xana.pre_edge_sd_map)[:]"; ln+=1
            code[ln] = "        g1211[:] = np.float32(xana.post_edge_sd_map)[:]"; ln+=1
            code[ln] = "        g1212[:] = np.float32(xana.pre_edge_mean_map)[:]"; ln+=1
            code[ln] = "        g1213[:] = np.float32(xana.post_edge_mean_map)[:]"; ln+=1
            code[ln] = ""; ln+=1
    
            gen_external_py_script(self.parent_h.xanes2D_external_command_name, code)
            sig = os.system(f'python {self.parent_h.xanes2D_external_command_name}')
            print(4)
            if sig == 0:
                self.hs['L[0][x][3][0][4][0]_analysis_run_text'].value = 'XANES2D analysis is done ...'
            else:
                self.hs['L[0][x][3][0][4][0]_analysis_run_text'].value = 'somthing wrong in analysis ...'   
            self.parent_h.update_xanes2D_config()

        self.parent_h.boxes_logic()