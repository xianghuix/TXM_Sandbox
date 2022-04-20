#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:33:11 2020

@author: xiao
"""

import traitlets
from tkinter import Tk, filedialog
from ipywidgets import widgets, GridspecLayout
from IPython.display import display, HTML
from fnmatch import fnmatch
import os, functools, glob, tifffile, h5py, json
import numpy as np
import skimage.morphology as skm
import xanes_regtools as xr
import time, gc, shutil
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import fourier_shift
from json import JSONEncoder
import napari
from gui_components import NumpyArrayEncoder
from gui_components import SelectFilesButton
# from gui_components import get_handles, get_decendant
from gui_components import *
from collections import OrderedDict

FILTER_PARAM_DICT = OrderedDict({'phase retrieval': {0: ['filter', ['paganin', 'paganino', 'bronnikov', 'fba'], 'filter: filter type used in phase retrieval'],
                                                     1: ['pad', ['True', 'False'], 'pad: boolean, if pad the data before phase retrieval filtering'],
                                                     6: ['pixel_size', 6.5e-5, 'pixel_size: in cm unit'],
                                                     7: ['dist', 15.0, 'dist: sample-detector distance in cm'],
                                                     8: ['energy', 35.0, 'energy: x-ray energy in keV'],
                                                     9: ['alpha', 1e-2, 'alpha: beta/delta, wherr n = (1-delta + i*beta) is x-ray rafractive index in the sample']},

                                 'down sample data': {6: ['level', 0.5, 'level: data downsampling ratio; level=0.5 downsamples the projection images by a factor 2. Projection angles wont be downsampled']},

                                 'flatting bkg': {6: ['air', 30, 'air: number of pixels on the both sides of projection images where is sample free region. This region will be used to correct background nonunifromness']},

                                 'remove cupping': {6: ['cc', 0.5, 'cc: constant that is subtracted from the logrithm of the normalized images. This is for correcting the cup-like background in the case when the sample size is much larger than the image view']},

                                 'stripe_removal: vo': {6: ['snr', 3, 'snr: signal-to-noise ratio'],
                                                        7: ['la_size', 81, "la_size: large ring's width in pixel"],
                                                        8:["sm_size", 21, "sm_size: small ring's width in pixel"]},

                                 'stripe_removal: ti': {6: ['nblock', 1, 'nblock: '],
                                                        7: ['alpha', 5, 'alphs: ']},

                                 'stripe_removal: sf': {6: ['size', 31, 'size: ']},

                                 'stripe_removal: fw': {0: ['pad', ['True', 'False'], 'pad: boolean, if padding data before filtering'],
                                                        1: ['wname', ['db1', 'db2', 'db3', 'db5', 'sym2', 'sym6', 'haar', 'gaus1', 'gaus2', 'gaus3', 'gaus4'], 'wname: wavelet name'],
                                                        6: ['level', 6, 'level: how many of level of wavelet transforms'],
                                                        7: ['sigma', 2, 'sigma: sigam of gaussian filter in image Fourier space']},

                                 'denoise: wiener': {0: ['reg', ['None'], 'reg: '],
                                                     1: ['is_real', ['True', 'False'], 'is_real: '],
                                                     2: ['clip', ['True', 'False'], 'clip: '],
                                                     6: ['psf', 2, 'psf: '],
                                                     7: ['balance', 0.3, 'balance: ']},

                                 'denoise: unsupervised_wiener': {0: ['reg', ['None'], 'reg: '],
                                                                  1: ['is_real', ['True', 'False'], 'is_real: '],
                                                                  2: ['clip', ['True', 'False'], 'clip: '],
                                                                  6: ['psf', 2, 'psf: '],
                                                                  7: ['balance', 0.3, 'balance: ']},

                                 'denoise: denoise_nl_means': {0: ['multichannel', ['False', 'True'], 'multichannel: '],
                                                               1: ['fast_mode', ['True', 'False'], 'fast_mode: '],
                                                               6: ['patch_size', 5, 'patch_size: '],
                                                               7: ['patch_distance', 7, 'patch_distance: '],
                                                               8: ['h', 0.1, 'h: '],
                                                               9: ['sigma', 0.05, 'sigma: ']},

                                 'denoise: denoise_tv_bregman': {0: ['isotrophic', ['True', 'False'], 'isotrophic: '],
                                                                 6: ['weight', 1.0, 'weight: '],
                                                                 7: ['max_iter', 100, 'max_iter: '],
                                                                 8: ['eps', 0.001, 'eps: ']},

                                 'denoise: denoise_tv_chambolle': {0: ['multichannel', ['False', 'True'], 'multichannel: '],
                                                                   6: ['weight', 0.1, 'weight: '],
                                                                   7: ['max_iter', 100, 'max_iter: '],
                                                                   8: ['eps', 0.002, 'eps: ']},

                                 'denoise: denoise_bilateral': {0: ['win_size', ['None'], 'win_size'],
                                                                1: ['sigma_color', ['None'], 'sigma_color'],
                                                                2: ['multichannel', ['False', 'True'], 'multichannel: '],
                                                                3: ['mode', ['constant'], 'mode: '],
                                                                6: ['sigma_spatial', 1, 'sigma_spatial: '],
                                                                7: ['bins', 10000, 'bins: '],
                                                                8: ['cval', 0, 'cval: ']},

                                 'denoise: denoise_wavelet': {0: ['wavelet', ['db1', 'db2', 'db3', 'db5', 'sym2', 'sym6', 'haar', 'gaus1', 'gaus2', 'gaus3', 'gaus4'], 'wavelet: '],
                                                              1: ['mode', ['soft'], 'mode: '],
                                                              2: ['multichannel', ['False', 'True'], 'multichannel: '],
                                                              3: ['convert2ycbcr', ['False', 'True'], 'convert2ycbcr: '],
                                                              4: ['method', ['BayesShrink'], 'method: '],
                                                              6: ['sigma', 1, 'sigma: '],
                                                              7: ['wavelet_levels', 3, 'wavelet_levels: ']}})

ALG_PARAM_DICT = OrderedDict({'gridrec':{0: ['filter_name', ['parzen'], 'filter_name: filter that is used in frequency space']},
                              'sirt':{3: ['num_gridx', 1280, 'num_gridx: number of the reconstructed slice image along x direction'],
                                      4: ['num_gridy', 1280, 'num_gridy: number of the reconstructed slice image along y direction'],
                                      5: ['num_inter', 10, 'num_inter: number of reconstruction iterations']},
                              'tv':{3: ['num_gridx', 1280, 'num_gridx: number of the reconstructed slice image along x direction'],
                                    4: ['num_gridy', 1280, 'num_gridy: number of the reconstructed slice image along y direction'],
                                    5: ['num_inter', 10, 'num_inter: number of reconstruction iterations'],
                                    6: ['reg_par', 0.1, 'reg_par: relaxation factor in tv regulation']},
                              'mlem':{3: ['num_gridx', 1280, 'num_gridx: number of the reconstructed slice image along x direction'],
                                      4: ['num_gridy', 1280, 'num_gridy: number of the reconstructed slice image along y direction'],
                                      5: ['num_inter', 10, 'num_inter: number of reconstruction iterations']},
                              'astra':{0: ['method', 'EM_CUDA', 'method: astra reconstruction methods'],
                                       1: ['proj_type', ['cuda'], 'proj_type: projection calculation options used in astra'],
                                       2: ['extra_options', ['MinConstraint'], 'extra_options: extra constraints used in the reconstructions. you need to set p03 for a MinConstraint level'],
                                       3: ['extra_options_param', -0.1, 'extra_options_param: parameter used together with extra_options'],
                                       4: ['num_inter', 50, 'num_inter: number of reconstruction iterations']}})

class tomo_recon_gui():
    def __init__(self, form_sz=[650, 740]):
        self.hs = {}
        self.form_sz = form_sz

        self.tomo_recon_type = 'Trial Center'

        self.tomo_filepath_configured = False
        self.tomo_data_configured = False

        self.tomo_use_debug = False
        self.tomo_use_alt_flat = False
        self.tomo_use_alt_dark = False
        self.tomo_use_fake_flat = False
        self.tomo_use_fake_dark = False
        self.tomo_use_rm_zinger = False
        self.tomo_use_mask = False
        self.tomo_is_wedge = False
        self.tomo_wedge_ang_auto_det_thres = False

        self.tomo_left_box_selected_flt = 'phase retrieval'
        self.tomo_right_filter_dict = {0:{}}

    def build_gui(self):
        #################################################################################################################
        #                                                                                                               #
        #                                                    TOMO RECON                                                 #
        #                                                                                                               #
        #################################################################################################################
        ## ## ## define 2D_XANES_tabs layout -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-86}px', 'height':f'{self.form_sz[0]-128}px'}
        self.hs['L[0][0][0]_config_input_form'] = widgets.VBox()
        self.hs['L[0][0][1]_filter&recon_form'] = widgets.VBox()
        self.hs['L[0][0][2]_reg&review_form'] = widgets.VBox()
        self.hs['L[0][0][3]_analysis&display_form'] = widgets.VBox()
        self.hs['L[0][0][0]_config_input_form'].layout = layout
        self.hs['L[0][0][1]_filter&recon_form'].layout = layout
        self.hs['L[0][0][2]_reg&review_form'].layout = layout
        self.hs['L[0][0][3]_analysis&display_form'].layout = layout

        ## ## ## define boxes in config_input_form -- start
        ## ## ## ## define functional widget tabs in each sub-tab - configure file settings -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.42*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][0]_select_file&path_box'] = widgets.VBox()
        self.hs['L[0][0][0][0]_select_file&path_box'].layout = layout

        ## ## ## ## ## label configure file settings box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][0][0]_select_file&path_title_box'] = widgets.HBox()
        self.hs['L[0][0][0][0][0]_select_file&path_title_box'].layout = layout
        self.hs['L[0][0][0][0][0][0]_select_file&path_title'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Config Dirs & Files' + '</span>')
        layout = {'background-color':'white', 'color':'cyan', 'left':'39%'}
        self.hs['L[0][0][0][0][0][0]_select_file&path_title'].layout = layout
        self.hs['L[0][0][0][0][0]_select_file&path_title_box'].children = get_handles(self.hs, 'L[0][0][0][0][0]_select_file&path_title_box', -1)

        ## ## ## ## ## raw h5 top directory
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][0][1]_select_raw_box'] = widgets.HBox()
        self.hs['L[0][0][0][0][1]_select_raw_box'].layout = layout
        self.hs['L[0][0][0][0][1][1]_select_raw_h5_top_dir_text'] = widgets.Text(value='Choose raw h5 top dir ...', description='', disabled=True)
        layout = {'width':'66%', 'display':'inline_flex'}
        self.hs['L[0][0][0][0][1][1]_select_raw_h5_top_dir_text'].layout = layout
        self.hs['L[0][0][0][0][1][0]_select_raw_h5_top_dir_button'] = SelectFilesButton(option='askdirectory',
                                                                                        text_h=self.hs['L[0][0][0][0][1][1]_select_raw_h5_top_dir_text'])
        self.hs['L[0][0][0][0][1][0]_select_raw_h5_top_dir_button'].description = 'Raw Top Dir'
        self.hs['L[0][0][0][0][1][0]_select_raw_h5_top_dir_button'].description_tooltip = 'Select the top directory in which the raw h5 files are located.'
        layout = {'width':'15%'}
        self.hs['L[0][0][0][0][1][0]_select_raw_h5_top_dir_button'].layout = layout
        self.hs['L[0][0][0][0][1][0]_select_raw_h5_top_dir_button'].on_click(self.L0_0_0_0_1_0_select_raw_h5_top_dir_button_click)
        self.hs['L[0][0][0][0][1]_select_raw_box'].children = get_handles(self.hs, 'L[0][0][0][0][1]_select_raw_box', -1)

        ## ## ## ## ##  save recon directory
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][0][2]_select_save_recon_box'] = widgets.HBox()
        self.hs['L[0][0][0][0][2]_select_save_recon_box'].layout = layout
        self.hs['L[0][0][0][0][2][1]_select_save_recon_dir_text'] = widgets.Text(value='Select top directory where recon subdirectories are saved...', description='', disabled=True)
        layout = {'width':'66%', 'display':'inline_flex'}
        self.hs['L[0][0][0][0][2][1]_select_save_recon_dir_text'].layout = layout
        self.hs['L[0][0][0][0][2][0]_select_save_recon_dir_button'] = SelectFilesButton(option='askdirectory',
                                                                                        text_h=self.hs['L[0][0][0][0][2][1]_select_save_recon_dir_text'])
        self.hs['L[0][0][0][0][2][0]_select_save_recon_dir_button'].description = 'Save Rec File'
        self.hs['L[0][0][0][0][2][0]_select_save_recon_dir_button'].disabled = True
        layout = {'width':'15%'}
        self.hs['L[0][0][0][0][2][0]_select_save_recon_dir_button'].layout = layout
        self.hs['L[0][0][0][0][2][0]_select_save_recon_dir_button'].on_click(self.L0_0_0_0_2_0_select_save_recon_dir_button_click)
        self.hs['L[0][0][0][0][2]_select_save_recon_box'].children = get_handles(self.hs, 'L[0][0][0][0][2]_select_save_recon_box', -1)

        ## ## ## ## ##  save data_center directory
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][0][3]_select_save_data_center_box'] = widgets.HBox()
        self.hs['L[0][0][0][0][3]_select_save_data_center_box'].layout = layout
        self.hs['L[0][0][0][0][3][1]_select_save_data_center_dir_text'] = widgets.Text(value='Select top directory where data_center will be created...', description='', disabled=True)
        layout = {'width':'66%', 'display':'inline_flex'}
        self.hs['L[0][0][0][0][3][1]_select_save_data_center_dir_text'].layout = layout
        self.hs['L[0][0][0][0][3][0]_select_save_data_center_dir_button'] = SelectFilesButton(option='askdirectory',
                                                                                              text_h=self.hs['L[0][0][0][0][3][1]_select_save_data_center_dir_text'])
        self.hs['L[0][0][0][0][3][0]_select_save_data_center_dir_button'].description = 'Save Data_Center'
        layout = {'width':'15%'}
        self.hs['L[0][0][0][0][3][0]_select_save_data_center_dir_button'].layout = layout
        self.hs['L[0][0][0][0][3][0]_select_save_data_center_dir_button'].on_click(self.L0_0_0_0_3_0_select_save_data_center_dir_button_click)
        self.hs['L[0][0][0][0][3]_select_save_data_center_box'].children = get_handles(self.hs, 'L[0][0][0][0][3]_select_save_data_center_box', -1)

        ## ## ## ## ##  save debug directory
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][0][4]_select_save_debug_box'] = widgets.HBox()
        self.hs['L[0][0][0][0][4]_select_save_debug_box'].layout = layout
        self.hs['L[0][0][0][0][4][1]_select_save_debug_dir_text'] = widgets.Text(value='Debug is disabled...', description='', disabled=True)
        layout = {'width':'66%', 'display':'inline_flex'}
        self.hs['L[0][0][0][0][4][1]_select_save_debug_dir_text'].layout = layout
        self.hs['L[0][0][0][0][4][0]_select_save_debug_dir_button'] = SelectFilesButton(option='askdirectory',
                                                                                        text_h=self.hs['L[0][0][0][0][4][1]_select_save_debug_dir_text'])
        self.hs['L[0][0][0][0][4][0]_select_save_debug_dir_button'].description = 'Save Debug Dir'
        self.hs['L[0][0][0][0][4][0]_select_save_debug_dir_button'].disabled = True
        layout = {'width':'15%'}
        self.hs['L[0][0][0][0][4][0]_select_save_debug_dir_button'].layout = layout
        self.hs['L[0][0][0][0][4][2]_save_debug_checkbox'] = widgets.Checkbox(value=False,
                                                                              description='Save Debug',
                                                                              disabled=False,
                                                                              indent=False)
        layout = {'left':'1%','width':'13%', 'display':'inline_flex'}
        self.hs['L[0][0][0][0][4][2]_save_debug_checkbox'].layout = layout
        self.hs['L[0][0][0][0][4][0]_select_save_debug_dir_button'].on_click(self.L0_0_0_0_4_0_select_save_debug_dir_button_click)
        self.hs['L[0][0][0][0][4][2]_save_debug_checkbox'].observe(self.L0_0_0_0_4_2_save_debug_checkbox_change, names='value')
        self.hs['L[0][0][0][0][4]_select_save_debug_box'].children = get_handles(self.hs, 'L[0][0][0][0][4]_select_save_debug_box', -1)

        ## ## ## ## ## confirm file configuration
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][0][5]_select_file&path_title_comfirm_box'] = widgets.HBox()
        self.hs['L[0][0][0][0][5]_select_file&path_title_comfirm_box'].layout = layout
        self.hs['L[0][0][0][0][5][1]_confirm_file&path_text'] = widgets.Text(value='After setting directories, confirm to proceed ...', description='', disabled=True)
        layout = {'width':'66%'}
        self.hs['L[0][0][0][0][5][1]_confirm_file&path_text'].layout = layout
        self.hs['L[0][0][0][0][5][0]_confirm_file&path_button'] = widgets.Button(description='Confirm',
                                                                                 tooltip='Confirm: Confirm after you finish file configuration')
        self.hs['L[0][0][0][0][5][0]_confirm_file&path_button'].style.button_color = 'darkviolet'
        self.hs['L[0][0][0][0][5][0]_confirm_file&path_button'].on_click(self.L0_0_0_0_5_0_confirm_file_path_button_click)
        layout = {'width':'15%'}
        self.hs['L[0][0][0][0][5][0]_confirm_file&path_button'].layout = layout

        self.hs['L[0][0][0][0][5][2]_file_path_options_dropdown'] = widgets.Dropdown(value='Trial Center',
                                                                              options=['Trial Center',
                                                                                       'Vol Recon: Single',
                                                                                       'Vol Recon: Multi'],
                                                                              description='',
                                                                              description_tooltip='"Trial Center": doing trial recon on a single slice to find rotation center; "Vol Recon: Single": doing volume recon of a single scan dataset; "Vol Recon: Multi": doing volume recon of a series of  scan datasets.',
                                                                              disabled=False)
        layout = {'width':'15%', 'top':'0%'}
        self.hs['L[0][0][0][0][5][2]_file_path_options_dropdown'].layout = layout

        self.hs['L[0][0][0][0][5][2]_file_path_options_dropdown'].observe(self.L0_0_0_0_5_2_file_path_options_dropdown_change, names='value')
        self.hs['L[0][0][0][0][5]_select_file&path_title_comfirm_box'].children = get_handles(self.hs, 'L[0][0][0][0][5]_select_file&path_title_comfirm_box', -1)

        self.hs['L[0][0][0][0]_select_file&path_box'].children = get_handles(self.hs, 'L[0][0][0][0]_select_file&path_box', -1)
        ## ## ## ## bin widgets in hs['L[0][0][0][0]_select_file&path_box'] -- configure file settings -- end


        ## ## ## ## define widgets recon_options_box -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-94}px', 'height':f'{0.56*(self.form_sz[0]-110)}px'}
        self.hs['L[0][0][0][1]_data_tab'] = widgets.Tab()
        self.hs['L[0][0][0][1]_data_tab'].layout = layout

        ## ## ## ## ## define sub-tabs in data_tab -- start
        layout = {'border':'3px solid #FFCC00', 'left':'-8px', 'width':f'{self.form_sz[1]-114}px', 'top':'-8px', 'height':f'{0.49*(self.form_sz[0]-116)}px'}
        self.hs['L[0][0][0][1][0]_data_config_tab'] = widgets.VBox()
        self.hs['L[0][0][0][1][0]_data_config_tab'].layout = layout
        self.hs['L[0][0][0][1][1]_alg_config_tab'] = widgets.VBox()
        self.hs['L[0][0][0][1][1]_alg_config_tab'].layout = layout
        self.hs['L[0][0][0][1][2]_data_info_tab'] = widgets.VBox()
        self.hs['L[0][0][0][1][2]_data_info_tab'].layout = layout
        self.hs['L[0][0][0][1]_data_tab'].children = get_handles(self.hs, 'L[0][0][0][1]_data_tab', -1)
        self.hs['L[0][0][0][1]_data_tab'].set_title(0, 'Data Config')
        self.hs['L[0][0][0][1]_data_tab'].set_title(1, 'Algorithm Config')
        self.hs['L[0][0][0][1]_data_tab'].set_title(2, 'Data Info')
        ## ## ## ## ## define sub-tabs in data_tab -- end

        ## ## ## ## ## ## config data parameters in data_config_box in data_config_tab -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-120}px', 'height':f'{0.49*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][0][1]_data_config_box'] = widgets.VBox()
        self.hs['L[0][0][0][1][0][1]_data_config_box'].layout = layout

        ## ## ## ## ## ## ## config sub-boxes in data_config_box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-126}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][0][1][0]_recon_config_box0'] = widgets.HBox()
        self.hs['L[0][0][0][1][0][1][0]_recon_config_box0'].layout = layout
        self.hs['L[0][0][0][1][0][1][0][0]_scan_id_text'] = widgets.IntText(value=0, description='Scan id', disabled=True)
        layout = {'width':'24%', 'height':'90%'}
        self.hs['L[0][0][0][1][0][1][0][0]_scan_id_text'].layout = layout
        self.hs['L[0][0][0][1][0][1][0][1]_rot_cen_text'] = widgets.FloatText(value=1280.0, description='Center', disabled=True)
        layout = {'width':'24%', 'height':'90%'}
        self.hs['L[0][0][0][1][0][1][0][1]_rot_cen_text'].layout = layout
        self.hs['L[0][0][0][1][0][1][0][2]_cen_win_left_text'] = widgets.IntText(value=1240, description='Cen Win L', disabled=True)
        layout = {'width':'24%', 'height':'90%'}
        self.hs['L[0][0][0][1][0][1][0][2]_cen_win_left_text'].layout = layout
        self.hs['L[0][0][0][1][0][1][0][3]_cen_win_wz_text'] = widgets.IntText(value=80, description='Cen Win W', disabled=True)
        layout = {'width':'24%', 'height':'90%'}
        self.hs['L[0][0][0][1][0][1][0][3]_cen_win_wz_text'].layout = layout
        self.hs['L[0][0][0][1][0][1][0]_recon_config_box0'].children = get_handles(self.hs, 'L[0][0][0][1][0][1][0]_data_preprocessing_options_box0', -1)
        self.hs['L[0][0][0][1][0][1][0][0]_scan_id_text'].observe(self.L0_0_0_1_0_1_0_0_scan_id_text_change, names='value')
        self.hs['L[0][0][0][1][0][1][0][1]_rot_cen_text'].observe(self.L0_0_0_1_0_1_0_1_rot_cen_text_change, names='value')
        self.hs['L[0][0][0][1][0][1][0][2]_cen_win_left_text'].observe(self.L0_0_0_1_0_1_0_2_cen_win_left_text_change, names='value')
        self.hs['L[0][0][0][1][0][1][0][3]_cen_win_wz_text'].observe(self.L0_0_0_1_0_1_0_3_cen_win_wz_text_change, names='value')

        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-126}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][0][1][3]_chunk_config_box3'] = widgets.HBox()
        self.hs['L[0][0][0][1][0][1][3]_chunk_config_box3'].layout = layout
        self.hs['L[0][0][0][1][0][1][3][0]_sli_start_text'] = widgets.IntText(value=0, description='Sli Start', disabled=True)
        layout = {'width':'24%', 'height':'90%'}
        self.hs['L[0][0][0][1][0][1][3][0]_sli_start_text'].layout = layout
        self.hs['L[0][0][0][1][0][1][3][1]_sli_end_text'] = widgets.IntText(value=0, description='Sli End', disabled=True)
        layout = {'width':'24%', 'height':'90%'}
        self.hs['L[0][0][0][1][0][1][3][1]_sli_end_text'].layout = layout
        self.hs['L[0][0][0][1][0][1][3][2]_chunk_sz_text'] = widgets.IntText(value=200, description='Chunk Sz', disabled=True)
        layout = {'width':'24%', 'height':'90%'}
        self.hs['L[0][0][0][1][0][1][3][2]_chunk_sz_text'].layout = layout
        self.hs['L[0][0][0][1][0][1][3][3]_margin_sz_text'] = widgets.IntText(value=15, description='Margin Sz', disabled=True)
        layout = {'width':'24%', 'height':'90%'}
        self.hs['L[0][0][0][1][0][1][3][3]_margin_sz_text'].layout = layout
        self.hs['L[0][0][0][1][0][1][3]_chunk_config_box3'].children = get_handles(self.hs, 'L[0][0][0][1][0][1][3]_chunk_config_box3', -1)
        self.hs['L[0][0][0][1][0][1][3][0]_sli_start_text'].observe(self.L0_0_0_1_0_1_3_0_sli_start_text_change, names='value')
        self.hs['L[0][0][0][1][0][1][3][1]_sli_end_text'].observe(self.L0_0_0_1_0_1_3_1_sli_end_text_change, names='value')
        self.hs['L[0][0][0][1][0][1][3][2]_chunk_sz_text'].observe(self.L0_0_0_1_0_1_3_2_chunk_sz_text_change, names='value')
        self.hs['L[0][0][0][1][0][1][3][3]_margin_sz_text'].observe(self.L0_0_0_1_0_1_3_3_margin_sz_text_change, names='value')

        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-126}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][0][1][1]_alt_flat/dark_options_box1'] = widgets.HBox()
        self.hs['L[0][0][0][1][0][1][1]_alt_flat/dark_options_box1'].layout = layout
        layout = {'width':'20%'}
        self.hs['L[0][0][0][1][0][1][1][0]_use_alt_flat_checkbox'] = widgets.Checkbox(value=False, description='Alt Flat', disabled=True)
        self.hs['L[0][0][0][1][0][1][1][0]_use_alt_flat_checkbox'].layout = layout
        layout = {'left':'12.5%', 'width':'15%'}
        self.hs['L[0][0][0][1][0][1][1][1]_alt_flat_file_button'] = SelectFilesButton(option='askopenfilename',
                                                                                    **{'open_filetypes': (('h5 files', '*.h5'),)})
        self.hs['L[0][0][0][1][0][1][1][1]_alt_flat_file_button'].description = 'Alt Flat File'
        self.hs['L[0][0][0][1][0][1][1][1]_alt_flat_file_button'].layout = layout
        layout = {'left':'12.5%', 'width':'20%'}
        self.hs['L[0][0][0][1][0][1][1][2]_use_alt_dark_checkbox'] = widgets.Checkbox(value=False, description='Alt Dark', disabled=True)
        self.hs['L[0][0][0][1][0][1][1][2]_use_alt_dark_checkbox'].layout = layout
        layout = {'left':'25%', 'width':'15%'}
        self.hs['L[0][0][0][1][0][1][1][3]_alt_dark_file_button'] = SelectFilesButton(option='askopenfilename',
                                                                                    **{'open_filetypes': (('h5 files', '*.h5'),)})
        self.hs['L[0][0][0][1][0][1][1][3]_alt_dark_file_button'].description = 'Alt Dark File'
        self.hs['L[0][0][0][1][0][1][1][3]_alt_dark_file_button'].layout = layout
        self.hs['L[0][0][0][1][0][1][1]_alt_flat/dark_options_box1'].children = get_handles(self.hs, 'L[0][0][0][1][0][1][1]_alt_flat/dark_options_box1', -1)
        self.hs['L[0][0][0][1][0][1][1][0]_use_alt_flat_checkbox'].observe(self.L0_0_0_1_0_1_1_0_use_alt_flat_checkbox_change, names='value')
        self.hs['L[0][0][0][1][0][1][1][1]_alt_flat_file_button'].observe(self.L0_0_0_1_0_1_1_1_alt_flat_file_button_change, names='value')
        self.hs['L[0][0][0][1][0][1][1][2]_use_alt_dark_checkbox'].observe(self.L0_0_0_1_0_1_1_2_use_alt_dark_checkbox_change, names='value')
        self.hs['L[0][0][0][1][0][1][1][3]_alt_dark_file_button'].observe(self.L0_0_0_1_0_1_1_3_alt_dark_file_button_change, names='value')

        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-126}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][0][1][2]_fake_flat/dark_options_box2'] = widgets.HBox()
        self.hs['L[0][0][0][1][0][1][2]_fake_flat/dark_options_box2'].layout = layout
        layout = {'width':'24%'}
        self.hs['L[0][0][0][1][0][1][2][0]_use_fake_flat_checkbox'] = widgets.Checkbox(value=False,
                                                                                       description='Fake Flat',
                                                                                       disabled=True)
        self.hs['L[0][0][0][1][0][1][2][0]_use_fake_flat_checkbox'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][0][1][2][1]_fake_flat_val_text'] = widgets.FloatText(value=10000.0,
                                                                                    disabled=True)
        self.hs['L[0][0][0][1][0][1][2][1]_fake_flat_val_text'].layout = layout
        layout = {'width':'24%'}
        self.hs['L[0][0][0][1][0][1][2][2]_use_fake_dark_checkbox'] = widgets.Checkbox(value=False,
                                                                                       description='Fake Dark',
                                                                                       disabled=True)
        self.hs['L[0][0][0][1][0][1][2][2]_use_fake_dark_checkbox'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][0][1][2][3]_fake_dark_val_text'] = widgets.FloatText(value=100.0,
                                                                                    disabled=True)
        self.hs['L[0][0][0][1][0][1][2][3]_fake_dark_val_text'].layout = layout
        self.hs['L[0][0][0][1][0][1][2][0]_use_fake_flat_checkbox'].observe(self.L0_0_0_1_0_1_2_0_use_fake_flat_checkbox_change, names='value')
        self.hs['L[0][0][0][1][0][1][2][1]_fake_flat_val_text'].observe(self.L0_0_0_1_0_1_2_1_fake_flat_val_text_change, names='value')
        self.hs['L[0][0][0][1][0][1][2][2]_use_fake_dark_checkbox'].observe(self.L0_0_0_1_0_1_2_2_use_fake_dark_checkbox_change, names='value')
        self.hs['L[0][0][0][1][0][1][2][3]_fake_dark_val_text'].observe(self.L0_0_0_1_0_1_2_3_fake_dark_val_text_change, names='value')
        self.hs['L[0][0][0][1][0][1][2]_fake_flat/dark_options_box2'].children = get_handles(self.hs, 'L[0][0][0][1][0][1][2]_data_preprocessing_options_box2', -1)

        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-126}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][0][1][4]_misc_options_box4'] = widgets.HBox()
        self.hs['L[0][0][0][1][0][1][4]_misc_options_box4'].layout = layout
        layout = {'width':'24%'}
        self.hs['L[0][0][0][1][0][1][4][0]_rm_zinger_checkbox'] = widgets.Checkbox(value=False,
                                                                                   description='Rm Zinger',
                                                                                   disabled=True)
        self.hs['L[0][0][0][1][0][1][4][0]_rm_zinger_checkbox'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][0][1][4][1]_zinger_level_text'] = widgets.FloatText(value=500.0,
                                                                                disabled=True)
        self.hs['L[0][0][0][1][0][1][4][1]_zinger_level_text'].layout = layout
        layout = {'width':'24%'}
        self.hs['L[0][0][0][1][0][1][4][2]_use_mask_checkbox'] = widgets.Checkbox(value=False,
                                                                                  description='Use Mask',
                                                                                  disabled=True)
        self.hs['L[0][0][0][1][0][1][4][2]_use_mask_checkbox'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][0][1][4][3]_mask_ratio_text'] = widgets.FloatText(value=1,
                                                                                 min=0,
                                                                                 max=1,
                                                                                 step=0.05,
                                                                                 disabled=True)
        self.hs['L[0][0][0][1][0][1][4][3]_mask_ratio_text'].layout = layout
        self.hs['L[0][0][0][1][0][1][4][0]_rm_zinger_checkbox'].observe(self.L0_0_0_1_0_1_4_0_rm_zinger_checkbox_change, names='value')
        self.hs['L[0][0][0][1][0][1][4][1]_zinger_level_text'].observe(self.L0_0_0_1_0_1_4_1_zinger_level_text_change, names='value')
        self.hs['L[0][0][0][1][0][1][4][2]_use_mask_checkbox'].observe(self.L0_0_0_1_0_1_4_2_use_mask_checkbox_change, names='value')
        self.hs['L[0][0][0][1][0][1][4][3]_mask_ratio_text'].observe(self.L0_0_0_1_0_1_4_3_mask_ratio_text_change, names='value')
        self.hs['L[0][0][0][1][0][1][4]_misc_options_box4'].children = get_handles(self.hs, 'L[0][0][0][1][0][1][4]_misc_options_box4', -1)

        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-126}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][0][1][5]_wedge_options_box0'] = widgets.HBox()
        self.hs['L[0][0][0][1][0][1][5]_wedge_options_box0'].layout = layout
        layout = {'width':'20%'}
        self.hs['L[0][0][0][1][0][1][5][0]_is_wedge_checkbox'] = widgets.Checkbox(value=False,
                                                                               description='Is Wedge',
                                                                               disabled=True,
                                                                               indent=False)
        self.hs['L[0][0][0][1][0][1][5][0]_is_wedge_checkbox'].layout = layout
        layout = {'width':'20%'}
        self.hs['L[0][0][0][1][0][1][5][1]_blankat_dropdown'] = widgets.Dropdown(value=90,
                                                                              options=[0, 90],
                                                                              description='Blandk At',
                                                                              disabled=True)
        self.hs['L[0][0][0][1][0][1][5][1]_blankat_dropdown'].layout = layout
        layout = {'width':'20%'}
        self.hs['L[0][0][0][1][0][1][5][2]_missing_start_text'] = widgets.IntText(value=50,
                                                                               description='Miss S',
                                                                               disabled=True)
        self.hs['L[0][0][0][1][0][1][5][2]_missing_start_text'].layout = layout
        layout = {'width':'20%'}
        self.hs['L[0][0][0][1][0][1][5][3]_missing_end_text'] =  widgets.IntText(value=100,
                                                                              description='Miss E',
                                                                              disabled=True)
        self.hs['L[0][0][0][1][0][1][5][3]_missing_end_text'].layout = layout
        self.hs['L[0][0][0][1][0][1][5][0]_is_wedge_checkbox'].observe(self.L0_0_1_0_1_0_0_is_wedge_checkbox_change, names='value')
        self.hs['L[0][0][0][1][0][1][5][1]_blankat_dropdown'].observe(self.L0_0_1_0_1_0_1_blankat_dropdown_change, names='value')
        self.hs['L[0][0][0][1][0][1][5][2]_missing_start_text'].observe(self.L0_0_1_0_1_0_2_missing_start_text_change, names='value')
        self.hs['L[0][0][0][1][0][1][5][3]_missing_end_text'].observe(self.L0_0_1_0_1_0_3_missing_end_text_change, names='value')
        self.hs['L[0][0][0][1][0][1][5]_wedge_options_box0'].children = get_handles(self.hs, 'L[0][0][0][1][0][1][5]_wedge_options_box0', -1)

        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-126}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][0][1][6]_wedge_options_box1'] = widgets.HBox()
        self.hs['L[0][0][0][1][0][1][6]_wedge_options_box1'].layout = layout
        HTML('<style> .widget-hbox .widget-label { max-width:350ex; text-align:left} </style>')
        layout = {'width':'13%'}
        self.hs['L[0][0][0][1][0][1][6][0]_auto_detect_checkbox'] = widgets.Checkbox(value=True,
                                                                                  description='Auto Det',
                                                                                  disabled=True,
                                                                                  indent=False)
        self.hs['L[0][0][0][1][0][1][6][0]_auto_detect_checkbox'].layout = layout
        layout = {'width':'20%'}
        self.hs['L[0][0][0][1][0][1][6][1]_auto_thres_text'] = widgets.FloatText(value=500.0,
                                                                              description='Auto Thres',
                                                                              disabled=True)
        self.hs['L[0][0][0][1][0][1][6][1]_auto_thres_text'].layout = layout
        layout = {'width':'15%'}
        self.hs['L[0][0][0][1][0][1][6][2]_auto_ref_fn_button'] = SelectFilesButton(option='askopenfilename',
                                                                                 **{'open_filetypes': (('txt files', '*.txt'),)})
        self.hs['L[0][0][0][1][0][1][6][2]_auto_ref_fn_button'].layout = layout
        layout = {'width':'13%'}
        self.hs['L[0][0][0][1][0][1][6][3]_fiji_viewer_checkbox'] = widgets.Checkbox(value=False,
                                                                                  description='fiji view',
                                                                                  disabled=True,
                                                                                  indent=False)
        self.hs['L[0][0][0][1][0][1][6][3]_fiji_viewer_checkbox'].layout = layout
        layout = {'width':'39%'}
        self.hs['L[0][0][0][1][0][1][6][4]_sli_range_slider'] = widgets.IntSlider(value=0,
                                                                               description='sli',
                                                                               min=0,
                                                                               max=100,
                                                                               disabled=True,
                                                                               indent=False)
        self.hs['L[0][0][0][1][0][1][6][4]_sli_range_slider'].layout = layout
        self.hs['L[0][0][0][1][0][1][6][0]_auto_detect_checkbox'].observe(self.L0_0_1_0_1_1_0_auto_detect_checkbox_change, names='value')
        self.hs['L[0][0][0][1][0][1][6][1]_auto_thres_text'].observe(self.L0_0_1_0_1_1_1_auto_thres_text_change, names='value')
        self.hs['L[0][0][0][1][0][1][6][2]_auto_ref_fn_button'].on_click(self.L0_0_1_0_1_1_2_auto_ref_fn_button_change)
        self.hs['L[0][0][0][1][0][1][6][3]_fiji_viewer_checkbox'].observe(self.L0_0_1_0_1_1_3_fiji_viewer_checkbox_change, names='value')
        self.hs['L[0][0][0][1][0][1][6][4]_sli_range_slider'].observe(self.L0_0_1_0_1_1_4_sli_range_slider_change, names='value')
        self.hs['L[0][0][0][1][0][1][6]_wedge_options_box1'].children = get_handles(self.hs, 'L[0][0][0][1][0][1][6]_wedge_options_box1', -1)
        ## ## ## ## ## ## ## config sub-boxes in data_config_box -- end

        self.hs['L[0][0][0][1][0][1]_data_config_box'].children = get_handles(self.hs, 'L[0][0][0][1][0][1]_recon_options_box', -1)
        ## ## ## ## ## ## config data parameters in data_config_box in data_config_tab -- end

        self.hs['L[0][0][0][1][0]_data_config_tab'].children = get_handles(self.hs, 'L[0][0][0][1][0]_recon_options_box1', -1)
        ## ## ## ## ## config data_config_tab -- end



        ## ## ## ## ## ## config alg_config_box in alg_config tab -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-121}px', 'height':f'{0.21*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][1][0]_alg_config_box'] = widgets.VBox()
        self.hs['L[0][0][0][1][1][0]_alg_config_box'].layout = layout

        ## ## ## ## ## ## ## label alg_config_box -- start
        layout = {'justify-content':'center', 'align-items':'center','align-contents':'center','border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-127}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][1][0][0]_alg_config_title_box'] = widgets.HBox()
        self.hs['L[0][0][0][1][1][0][0]_alg_config_title_box'].layout = layout
        self.hs['L[0][0][0][1][1][0][0][0]_alg_config_title'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Wedge Data' + '</span>')
        layout = {'left':'43.25%', 'background-color':'white', 'color':'cyan'}
        self.hs['L[0][0][0][1][1][0][0][0]_alg_config_title'].layout = layout
        self.hs['L[0][0][0][1][1][0][0]_alg_config_title_box'].children = get_handles(self.hs, 'L[0][0][0][1][1][0][0]_alg_config_title_box', -1)
        ## ## ## ## ## ## ## label alg_config_box -- end

        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-127}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][1][0][1]_alg_options_box0'] = widgets.HBox()
        self.hs['L[0][0][0][1][1][0][1]_alg_options_box0'].layout = layout
        layout = {'width':'24%'}
        self.hs['L[0][0][0][1][1][0][1][0]_alg_options_dropdown'] = widgets.Dropdown(value='gridrec',
                                                                                  options=['gridrec', 'sirt', 'tv', 'mlem', 'astra'],
                                                                                  description='algs',
                                                                                  disabled=True)
        self.hs['L[0][0][0][1][1][0][1][0]_alg_options_dropdown'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][1][0][1][1]_alg_p00_dropdown'] = widgets.Dropdown(value='',
                                                                                  options=[''],
                                                                                  description='p00',
                                                                                  disabled=True)
        self.hs['L[0][0][0][1][1][0][1][1]_alg_p00_dropdown'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][1][0][1][2]_alg_p01_dropdown'] = widgets.Dropdown(value='',
                                                                                  options=[''],
                                                                                  description='p01',
                                                                                  disabled=True)
        self.hs['L[0][0][0][1][1][0][1][2]_alg_p01_dropdown'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][1][0][1][3]_alg_p02_dropdown'] = widgets.Dropdown(value='',
                                                                                options=[''],
                                                                                description='p02',
                                                                                disabled=True)
        self.hs['L[0][0][0][1][1][0][1][3]_alg_p02_dropdown'].layout = layout
        self.hs['L[0][0][0][1][1][0][1][0]_alg_options_dropdown'].observe(self.L0_0_0_1_1_0_1_0_alg_options_dropdown_change, names='value')
        self.hs['L[0][0][0][1][1][0][1][1]_alg_p00_dropdown'].observe(self.L0_0_0_1_1_0_1_1_alg_p00_dropdown_change, names='value')
        self.hs['L[0][0][0][1][1][0][1][2]_alg_p01_dropdown'].observe(self.L0_0_0_1_1_0_1_2_alg_p01_dropdown_change, names='value')
        self.hs['L[0][0][0][1][1][0][1][3]_alg_p02_dropdown'].observe(self.L0_0_0_1_1_0_1_3_alg_p02_dropdown_change, names='value')
        self.hs['L[0][0][0][1][1][0][1]_alg_options_box0'].children = get_handles(self.hs, 'L[0][0][0][1][1][0][1]_alg_options_box0', -1)


        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-127}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][1][0][2]_alg_options_box1'] = widgets.HBox()
        self.hs['L[0][0][0][1][1][0][2]_alg_options_box1'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][1][0][2][0]_alg_p03_text'] = widgets.FloatText(value=0,
                                                                              description='p03',
                                                                              disabled=True)
        self.hs['L[0][0][0][1][1][0][2][0]_alg_p03_text'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][1][0][2][1]_alg_p04_text'] = widgets.FloatText(value=0,
                                                                              description='p04',
                                                                              disabled=True)
        self.hs['L[0][0][0][1][1][0][2][1]_alg_p04_text'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][1][0][2][2]_alg_p05_text'] = widgets.FloatText(value=0,
                                                                              description='p05',
                                                                              disabled=True)
        self.hs['L[0][0][0][1][1][0][2][2]_alg_p05_text'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][1][0][2][3]_alg_p06_text'] = widgets.FloatText(value=0.0,
                                                                              description='p06',
                                                                              disabled=True)
        self.hs['L[0][0][0][1][1][0][2][3]_alg_p06_text'].layout = layout
        self.hs['L[0][0][0][1][1][0][2][0]_alg_p03_text'].observe(self.L0_0_0_1_1_0_2_0_alg_p03_text_change, names='value')
        self.hs['L[0][0][0][1][1][0][2][1]_alg_p04_text'].observe(self.L0_0_0_1_1_0_2_1_alg_p04_text_change, names='value')
        self.hs['L[0][0][0][1][1][0][2][2]_alg_p05_text'].observe(self.L0_0_0_1_1_0_2_2_alg_p05_text_change, names='value')
        self.hs['L[0][0][0][1][1][0][2][3]_alg_p06_text'].observe(self.L0_0_0_1_1_0_2_3_alg_p06_text_change, names='value')
        self.hs['L[0][0][0][1][1][0][2]_alg_options_box1'].children = get_handles(self.hs, 'L[0][0][0][1][1][0][2]_alg_options_box1', -1)

        self.hs['L[0][0][0][1][1][0]_alg_config_box'].children = get_handles(self.hs, 'L[0][0][0][1][1][0]_alg_config_box', -1)
        ## ## ## ## ## ## config alg_config_box in alg_config tab -- end

        self.hs['L[0][0][0][1][1]_alg_config_tab'].children = get_handles(self.hs, 'L[0][0][0][1][1]_alg_config_box', -1)
        ## ## ## ## ## define alg_config tab -- end

        self.hs['L[0][0][0]_config_input_form'].children = get_handles(self.hs, 'L[0][0][0]_config_input_form', -1)
        ## ## ## config config_input_form -- end


        ## ## ## ## config filter_config_box -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.7*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][1][0]_filter_config_box'] = widgets.VBox()
        self.hs['L[0][0][1][0]_filter_config_box'].layout = layout

        ## ## ## ## ## label filter_config_box -- start
        layout = {'justify-content':'center', 'align-items':'center','align-contents':'center','border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][1][0][0]_filter_config_title_box'] = widgets.HBox()
        self.hs['L[0][0][1][0][0]_filter_config_title_box'].layout = layout
        self.hs['L[0][0][1][0][0][0]_filter_config_title'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Filter Config' + '</span>')
        layout = {'left':'43%', 'background-color':'white', 'color':'cyan'}
        self.hs['L[0][0][1][0][0][0]_filter_config_title'].layout = layout
        self.hs['L[0][0][1][0][0]_filter_config_title_box'].children = get_handles(self.hs, 'L[0][0][1][0][0]_filter_config_title_box', -1)
        ## ## ## ## ## label filter_config_box -- end

        ## ## ## ## ## config filters with GridspecLayout-- start
        self.hs['L[0][0][1][0][1]_filter_config_box'] = GridspecLayout(2, 200,
                                                                       layout = {'border':'3px solid #FFCC00',
                                                                                 'height':f'{0.63*(self.form_sz[0]-128)}px',
                                                                                 'align_items':'flex-start',
                                                                                 'justify_items':'flex-start'})
        ## ## ## ## ## ## config filters: left-hand side box in GridspecLayout -- start
        self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100] = GridspecLayout(10, 20,
                                                                                grid_gap='8px',
                                                                                layout = {'border':'3px solid #FFCC00',
                                                                                          'height':f'{0.56*(self.form_sz[0]-128)}px',
                                                                                          'grid_row_gap':'8px',
                                                                                          'align_items':'flex-start',
                                                                                          'justify_items':'flex-start',
                                                                                          'grid_column_gap':'8px'})
        self.hs['L[0][0][1][0][1][0]_filter_config_left_box'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100]
        self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][0, :16] = widgets.Dropdown(value='phase retrieval',
                                                                                          options=['phase retrieval',
                                                                                                   'down sample data',
                                                                                                   'flatting bkg',
                                                                                                   'remove cupping',
                                                                                                   'stripe_removal: vo',
                                                                                                   'stripe_removal: ti',
                                                                                                   'stripe_removal: sf',
                                                                                                   'stripe_removal: fw',
                                                                                                   'denoise: wiener',
                                                                                                   'denoise: unsupervised_wiener',
                                                                                                   'denoise: denoise_nl_means',
                                                                                                   'denoise: denoise_tv_bregman',
                                                                                                   'denoise: denoise_tv_chambolle',
                                                                                                   'denoise: denoise_bilateral',
                                                                                                   'denoise: denoise_wavelet'],
                                                                                          description='Filter List',
                                                                                          indent=False,
                                                                                          disabled=True)
        self.hs['L[0][0][1][0][1][0][0]_filter_config_left_box_filter_list'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][0, :16]
        self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][0, 16:19] = widgets.Button(description='==>',
                                                                                          disabled=True,
                                                                                          layout = {'width':f'{int(1.5*(self.form_sz[1]-98)/20)}px'})
        self.hs['L[0][0][1][0][1][0][1]_filter_config_left_box_==>_button'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][0, 16:19]
        self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][0, 16:19].style.button_color = "#0000FF"
        for ii in range(3):
            for jj in range(2):
                self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][1+ii, jj*8:(jj+1)*8] = widgets.Dropdown(value='',
                                                                                                               options=[''],
                                                                                                               description='p'+str(ii*2+jj).zfill(2),
                                                                                                               disabled=True,
                                                                                                               layout = {'align_items':'flex-start',
                                                                                                                         'width':f'{int(7.5*(self.form_sz[1]-98)/40)}px'})
        for ii in range(3):
            for jj in range(2):
                self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][4+ii, jj*8:(jj+1)*8] = widgets.FloatText(value=0,
                                                                                                                description='p'+str((ii+3)*2+jj).zfill(2),
                                                                                                                disabled=True,
                                                                                                                layout = {'align_items':'flex-start',
                                                                                                                          'width':f'{int(7.5*(self.form_sz[1]-98)/40)}px'})
        self.hs['L[0][0][1][0][1][0][2]_filter_config_left_box_p00'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][1, 0:8]
        self.hs['L[0][0][1][0][1][0][3]_filter_config_left_box_p01'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][1, 8:16]
        self.hs['L[0][0][1][0][1][0][4]_filter_config_left_box_p02'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][2, 0:8]
        self.hs['L[0][0][1][0][1][0][5]_filter_config_left_box_p03'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][2, 8:16]
        self.hs['L[0][0][1][0][1][0][6]_filter_config_left_box_p04'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][3, 0:8]
        self.hs['L[0][0][1][0][1][0][7]_filter_config_left_box_p05'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][3, 8:16]
        self.hs['L[0][0][1][0][1][0][8]_filter_config_left_box_p06'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][4, 0:8]
        self.hs['L[0][0][1][0][1][0][9]_filter_config_left_box_p07'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][4, 8:16]
        self.hs['L[0][0][1][0][1][0][10]_filter_config_left_box_p08'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][5, 0:8]
        self.hs['L[0][0][1][0][1][0][11]_filter_config_left_box_p09'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][5, 8:16]
        self.hs['L[0][0][1][0][1][0][12]_filter_config_left_box_p10'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][6, 0:8]
        self.hs['L[0][0][1][0][1][0][13]_filter_config_left_box_p11'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][6, 8:16]


        self.hs['L[0][0][1][0][1]_filter_config_box'][0, :100][7:, :] = widgets.HTML(value= '<style>p{word-wrap: break-word}</style> <p>'+ 'Hover mouse over params for the description of the param for each filter.' +' </p>')
        self.hs['L[0][0][1][0][1][0][0]_filter_config_left_box_filter_list'].observe(self.L0_0_1_0_1_0_0_filter_config_box_left_box_filter_dropdown_change, names='value')
        self.hs['L[0][0][1][0][1][0][1]_filter_config_left_box_==>_button'].on_click(self.L0_0_1_0_1_0_1_filter_config_box_left_box_move_button_change)
        self.hs['L[0][0][1][0][1][0][2]_filter_config_left_box_p00'].observe(self.L0_0_1_0_1_0_2_filter_config_box_left_box_p00_change, names='value')
        self.hs['L[0][0][1][0][1][0][3]_filter_config_left_box_p01'].observe(self.L0_0_1_0_1_0_3_filter_config_box_left_box_p01_change, names='value')
        self.hs['L[0][0][1][0][1][0][4]_filter_config_left_box_p02'].observe(self.L0_0_1_0_1_0_4_filter_config_box_left_box_p02_change, names='value')
        self.hs['L[0][0][1][0][1][0][5]_filter_config_left_box_p03'].observe(self.L0_0_1_0_1_0_5_filter_config_box_left_box_p03_change, names='value')
        self.hs['L[0][0][1][0][1][0][6]_filter_config_left_box_p04'].observe(self.L0_0_1_0_1_0_6_filter_config_box_left_box_p04_change, names='value')
        self.hs['L[0][0][1][0][1][0][7]_filter_config_left_box_p05'].observe(self.L0_0_1_0_1_0_7_filter_config_box_left_box_p05_change, names='value')
        self.hs['L[0][0][1][0][1][0][8]_filter_config_left_box_p06'].observe(self.L0_0_1_0_1_0_8_filter_config_box_left_box_p06_change, names='value')
        self.hs['L[0][0][1][0][1][0][9]_filter_config_left_box_p07'].observe(self.L0_0_1_0_1_0_9_filter_config_box_left_box_p07_change, names='value')
        self.hs['L[0][0][1][0][1][0][10]_filter_config_left_box_p08'].observe(self.L0_0_1_0_1_0_10_filter_config_box_left_box_p08_change, names='value')
        self.hs['L[0][0][1][0][1][0][11]_filter_config_left_box_p09'].observe(self.L0_0_1_0_1_0_11_filter_config_box_left_box_p09_change, names='value')
        self.hs['L[0][0][1][0][1][0][12]_filter_config_left_box_p10'].observe(self.L0_0_1_0_1_0_12_filter_config_box_left_box_p10_change, names='value')
        self.hs['L[0][0][1][0][1][0][13]_filter_config_left_box_p11'].observe(self.L0_0_1_0_1_0_13_filter_config_box_left_box_p11_change, names='value')
        self.hs['L[0][0][1][0][1][0]_filter_config_left_box'].children = get_handles(self.hs, 'L[0][0][1][0][1][0]_filter_config_left_box', -1)
        ## ## ## ## ## ## config filters: left-hand side box in GridspecLayout -- end

        ## ## ## ## ## ## config filters: right-hand side box in GridspecLayout -- start
        self.hs['L[0][0][1][0][1]_filter_config_box'][0, 100:] = GridspecLayout(10, 10,
                                                                                grid_gap='8px',
                                                                                layout = {'border':'3px solid #FFCC00',
                                                                                          'height':f'{0.56*(self.form_sz[0]-128)}px'})
        self.hs['L[0][0][1][0][1][1]_filter_config_right_box'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, 100:]
        self.hs['L[0][0][1][0][1]_filter_config_box'][0, 100:][:, :9] = widgets.SelectMultiple(value=['None'],
                                                                                               options=['None'],
                                                                                               description='Filter Seq',
                                                                                               disabled=True,
                                                                                               layout={'height':f'{0.48*(self.form_sz[0]-128)}px'})
        self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, 100:][:, :9]
        self.hs['L[0][0][1][0][1]_filter_config_box'][0, 100:][1, 9] = widgets.Button(description='Move Up',
                                                                                      disabled=True,
                                                                                      layout={'width':f'{int(2*(self.form_sz[1]-98)/20)}px'})
        self.hs['L[0][0][1][0][1]_filter_config_box'][0, 100:][1, 9].style.button_color = "#0000FF"
        self.hs['L[0][0][1][0][1][1][1]_filter_config_right_box_mv_up_button'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, 100:][1, 9]
        self.hs['L[0][0][1][0][1]_filter_config_box'][0, 100:][2, 9] = widgets.Button(description='Move Dn',
                                                                                      disabled=True,
                                                                                      layout={'width':f'{int(2*(self.form_sz[1]-98)/20)}px'})
        self.hs['L[0][0][1][0][1]_filter_config_box'][0, 100:][2, 9].style.button_color = "#0000FF"
        self.hs['L[0][0][1][0][1][1][2]_filter_config_right_box_mv_dn_button'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, 100:][2, 9]
        self.hs['L[0][0][1][0][1]_filter_config_box'][0, 100:][3, 9] = widgets.Button(description='Remove',
                                                                                      disabled=True,
                                                                                      layout={'width':f'{int(2*(self.form_sz[1]-98)/20)}px'})
        self.hs['L[0][0][1][0][1]_filter_config_box'][0, 100:][3, 9].style.button_color = "#0000FF"
        self.hs['L[0][0][1][0][1][1][3]_filter_config_right_box_rm_button'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, 100:][3, 9]
        self.hs['L[0][0][1][0][1]_filter_config_box'][0, 100:][4, 9] = widgets.Button(description='Finish',
                                                                                      disabled=True,
                                                                                      layout={'width':f'{int(2*(self.form_sz[1]-98)/20)}px'})
        self.hs['L[0][0][1][0][1]_filter_config_box'][0, 100:][4, 9].style.button_color = "#0000FF"
        self.hs['L[0][0][1][0][1][1][4]_filter_config_right_box_finish_button'] = self.hs['L[0][0][1][0][1]_filter_config_box'][0, 100:][4, 9]
        self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].observe(self.L0_0_1_0_1_1_0_filter_config_box_right_box_selectmultiple_change, names='value')
        self.hs['L[0][0][1][0][1][1][1]_filter_config_right_box_mv_up_button'].on_click(self.L0_0_1_0_1_1_1_filter_config_box_right_box_move_up_button_change)
        self.hs['L[0][0][1][0][1][1][2]_filter_config_right_box_mv_dn_button'].on_click(self.L0_0_1_0_1_1_2_filter_config_box_right_box_move_dn_button_change)
        self.hs['L[0][0][1][0][1][1][3]_filter_config_right_box_rm_button'].on_click(self.L0_0_1_0_1_1_3_filter_config_box_right_box_remove_button_change)
        self.hs['L[0][0][1][0][1][1][4]_filter_config_right_box_finish_button'].on_click(self.L0_0_1_0_1_1_4_filter_config_box_right_box_finish_button_change)
        self.hs['L[0][0][1][0][1][1]_filter_config_right_box'].children = get_handles(self.hs, 'L[0][0][1][0][1][1]_filter_config_right_box', -1)
        ## ## ## ## ## ## config filters: right-hand side box in GridspecLayout -- end

        ## ## ## ## ## ## config confirm box in GridspecLayout -- start
        self.hs['L[0][0][1][0][1]_filter_config_box'][1, :140] = widgets.Text(value='Confirm to proceed after you finish data and algorithm configuration...',
                                                                              layout={'width':f'{int(0.696*(self.form_sz[1]-98))}px', 'height':'90%'},
                                                                              disabled=True)
        self.hs['L[0][0][1][0][1][2]_filter_config_box_confirm_text'] = self.hs['L[0][0][1][0][1]_filter_config_box'][1, :140]
        self.hs['L[0][0][1][0][1]_filter_config_box'][1, 141:171] = widgets.Button(description='Confirm',
                                                                                   disabled=True,
                                                                                   layout={'width':f'{int(0.15*(self.form_sz[1]-98))}px', 'height':'90%'})
        self.hs['L[0][0][1][0][1]_filter_config_box'][1, 141:171].style.button_color = 'darkviolet'
        self.hs['L[0][0][1][0][1][3]_filter_config_box_confirm_button'] = self.hs['L[0][0][1][0][1]_filter_config_box'][1, 141:171]
        self.hs['L[0][0][1][0][1][3]_filter_config_box_confirm_button'].observe(self.L0_0_1_0_1_3_filter_config_box_confirm_button, names='value')
        ## ## ## ## ## ## config confirm box in GridspecLayout -- end

        self.hs['L[0][0][1][0][1]_filter_config_box'].children = get_handles(self.hs, 'L[0][0][1][0][1]_filter_config_box', -1)
        ## ## ## ## ## config filters with GridspecLayout-- end

        self.hs['L[0][0][1][0]_filter_config_box'].children = get_handles(self.hs, 'L[0][0][1][0]_filter_config_box', -1)
        ## ## ## ## config  filter_config_box -- end



        ## ## ## ## config recon_box -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.14*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][1][1]_recon_box'] = widgets.VBox()
        self.hs['L[0][0][1][1]_recon_box'].layout = layout

        ## ## ## ## ## label recon_box -- start
        layout = {'justify-content':'center', 'align-items':'center','align-contents':'center','border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][1][1][0]_reecon_title_box'] = widgets.HBox()
        self.hs['L[0][0][1][1][0]_reecon_title_box'].layout = layout
        self.hs['L[0][0][1][1][0][0]_reecon_title'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'recon' + '</span>')
        layout = {'left':'46.5%', 'background-color':'white', 'color':'cyan'}
        self.hs['L[0][0][1][1][0][0]_reecon_title'].layout = layout
        self.hs['L[0][0][1][1][0]_reecon_title_box'].children = get_handles(self.hs, 'L[0][0][1][1][0]_reecon_title_box', -1)
        ## ## ## ## ## label recon_box -- end

        ## ## ## ## ## ## config widgets in recon_box -- start
        layout = {'justify-content':'center', 'align-items':'center','align-contents':'center','border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][1][1][1]_reecon_box'] = widgets.HBox()
        self.hs['L[0][0][1][1][1]_reecon_box'].layout = layout
        layout = {'width':'70%', 'height':'90%'}
        self.hs['L[0][0][1][1][1][0]_reecon_progress_bar'] = widgets.IntProgress(value=0,
                                                                                 min=0,
                                                                                 max=10,
                                                                                 step=1,
                                                                                 description='Completing:',
                                                                                 bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                                                                 orientation='horizontal')
        self.hs['L[0][0][1][1][1][0]_reecon_progress_bar'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][0][1][1][1][1]_reecon_button'] = widgets.Button(description='Recon',
                                                                            disabled=True)
        self.hs['L[0][0][1][1][1][1]_reecon_button'].style.button_color = 'darkviolet'
        self.hs['L[0][0][1][1][1][1]_reecon_button'].layout = layout
        self.hs['L[0][0][1][1][1]_reecon_box'].children = get_handles(self.hs, 'L[0][0][1][1][1]_reecon_box', -1)
        self.hs['L[0][0][1][1][1][1]_reecon_button'].on_click(self.L0_0_1_2_1_1_reecon_button_change)
        ## ## ## ## ## ## config widgets in recon_box -- end

        self.hs['L[0][0][1][1]_recon_box'].children = get_handles(self.hs, 'L[0][0][1][1]_recon_box', -1)
        ## ## ## ## config recon box -- end

        self.hs['L[0][0][1]_filter&recon_form'].children = get_handles(self.hs, 'L[0][0][1]_filter&recon_form', -1)
        ## ## ## define boxes in filter&recon_form -- end
        self.bundle_param_handles()

    def bundle_param_handles(self):
        self.flt_phs = [self.hs['L[0][0][1][0][1][0][2]_filter_config_left_box_p00'],
                           self.hs['L[0][0][1][0][1][0][3]_filter_config_left_box_p01'],
                           self.hs['L[0][0][1][0][1][0][4]_filter_config_left_box_p02'],
                           self.hs['L[0][0][1][0][1][0][5]_filter_config_left_box_p03'],
                           self.hs['L[0][0][1][0][1][0][6]_filter_config_left_box_p04'],
                           self.hs['L[0][0][1][0][1][0][7]_filter_config_left_box_p05'],
                           self.hs['L[0][0][1][0][1][0][8]_filter_config_left_box_p06'],
                           self.hs['L[0][0][1][0][1][0][9]_filter_config_left_box_p07'],
                           self.hs['L[0][0][1][0][1][0][10]_filter_config_left_box_p08'],
                           self.hs['L[0][0][1][0][1][0][11]_filter_config_left_box_p09'],
                           self.hs['L[0][0][1][0][1][0][12]_filter_config_left_box_p10'],
                           self.hs['L[0][0][1][0][1][0][13]_filter_config_left_box_p11']]
        self.alg_phs = [self.hs['L[0][0][0][1][1][0][1][1]_alg_p00_dropdown'],
                        self.hs['L[0][0][0][1][1][0][1][2]_alg_p01_dropdown'],
                        self.hs['L[0][0][0][1][1][0][1][3]_alg_p02_dropdown'],
                        self.hs['L[0][0][0][1][1][0][2][0]_alg_p03_text'],
                        self.hs['L[0][0][0][1][1][0][2][1]_alg_p04_text'],
                        self.hs['L[0][0][0][1][1][0][2][2]_alg_p05_text'],
                        self.hs['L[0][0][0][1][1][0][2][3]_alg_p06_text']]

    def restart(self):
        pass

    def boxes_logic(self):
        if self.tomo_recon_type == 'Trial Center':
            self.hs['L[0][0][0][0][4][2]_save_debug_checkbox'].disabled = False
        elif self.tomo_recon_type in ['Vol Recon: Single', 'Vol Recon: Multi']:
            self.tomo_use_debug = False
            self.hs['L[0][0][0][0][4][2]_save_debug_checkbox'].value = False
            self.hs['L[0][0][0][0][4][2]_save_debug_checkbox'].disabled = True

        if not self.tomo_filepath_configured:
            boxes = ['L[0][0][0][1]_data_tab',
                     'L[0][0][1]_filter&recon_form']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        elif (self.tomo_filepath_configured & (not self.tomo_data_configured)):
            boxes = ['L[0][0][0][1][0]_data_config_tab',
                     'L[0][0][0][1][1][0][1][0]_alg_options_dropdown',
                     'L[0][0][0][1][2]_data_info_tab',
                     'L[0][0][1][0][1][0][0]_filter_config_left_box_filter_list',
                     'L[0][0][1][0][1][0][1]_filter_config_left_box_==>_button',
                     'L[0][0][1][0][1][1]_filter_config_right_box',
                     'L[0][0][1][0][1][3]_filter_config_box_confirm_button']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            boxes = ['L[0][0][1][1]_recon_box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        elif (self.tomo_filepath_configured & self.tomo_data_configured):
            boxes = ['L[0][0][0][1][0]_data_config_tab',
                     'L[0][0][0][1][1][0][1][0]_alg_options_dropdown',
                     'L[0][0][0][1][2]_data_info_tab',
                     'L[0][0][1][0][1][0][0]_filter_config_left_box_filter_list',
                     'L[0][0][1][0][1][0][1]_filter_config_left_box_==>_button',
                     'L[0][0][1][0][1][1]_filter_config_right_box',
                     'L[0][0][1][1]_recon_box']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)

    def tomo_compound_logic(self):
        if self.tomo_recon_type == 'Trial Center':
            self.hs['L[0][0][0][1][0][1][0][2]_cen_win_left_text'].disabled = False
            self.hs['L[0][0][0][1][0][1][0][3]_cen_win_wz_text'].disabled = False
            self.hs['L[0][0][0][1][0][1][0][1]_rot_cen_text'].disabled = True
            self.hs['L[0][0][0][1][0][1][3][2]_chunk_sz_text'].disabled = True
            self.hs['L[0][0][0][1][0][1][3][3]_margin_sz_text'].disabled = True
        elif self.tomo_recon_type in ['Vol Recon: Single', 'Vol Recon: Multi']:
            self.hs['L[0][0][0][1][0][1][0][2]_cen_win_left_text'].disabled = True
            self.hs['L[0][0][0][1][0][1][0][3]_cen_win_wz_text'].disabled = True
            self.hs['L[0][0][0][1][0][1][0][1]_rot_cen_text'].disabled = False
            self.hs['L[0][0][0][1][0][1][3][2]_chunk_sz_text'].disabled = False
            self.hs['L[0][0][0][1][0][1][3][3]_margin_sz_text'].disabled = False

        if self.tomo_use_alt_flat:
            self.hs['L[0][0][0][1][0][1][1][1]_alt_flat_file_button'].disabled = False
        else:
            self.hs['L[0][0][0][1][0][1][1][1]_alt_flat_file_button'].disabled = True

        if self.tomo_use_fake_flat:
            self.hs['L[0][0][0][1][0][1][2][1]_fake_flat_val_text'].disabled = False
        else:
            self.hs['L[0][0][0][1][0][1][2][1]_fake_flat_val_text'].disabled = True

        if self.tomo_use_alt_dark:
            self.hs['L[0][0][0][1][0][1][1][3]_alt_dark_file_button'].disabled = False
        else:
            self.hs['L[0][0][0][1][0][1][1][3]_alt_dark_file_button'].disabled = True

        if self.tomo_use_fake_dark:
            self.hs['L[0][0][0][1][0][1][2][3]_fake_dark_val_text'].disabled = False
        else:
            self.hs['L[0][0][0][1][0][1][2][3]_fake_dark_val_text'].disabled = True

        if self.tomo_use_rm_zinger:
            self.hs['L[0][0][0][1][0][1][4][1]_zinger_level_text'].disabled = False
        else:
            self.hs['L[0][0][0][1][0][1][4][1]_zinger_level_text'].disabled = True

        if self.tomo_use_mask:
            self.hs['L[0][0][0][1][0][1][4][3]_mask_ratio_text'].disabled = False
        else:
            self.hs['L[0][0][0][1][0][1][4][3]_mask_ratio_text'].disabled = True

        if self.tomo_is_wedge:
            self.hs['L[0][0][0][1][0][1][6][0]_auto_detect_checkbox'].disabled = False
            if self.tomo_wedge_ang_auto_det:
                self.hs['L[0][0][0][1][0][1][5][1]_blankat_dropdown'].disabled = True
                self.hs['L[0][0][0][1][0][1][5][2]_missing_start_text'].disabled = True
                self.hs['L[0][0][0][1][0][1][5][3]_missing_end_text'].disabled = True
                self.hs['L[0][0][0][1][0][1][6][1]_auto_thres_text'].disabled = False
                self.hs['L[0][0][0][1][0][1][6][2]_auto_ref_fn_button'].disabled = False
            else:
                self.hs['L[0][0][0][1][0][1][5][1]_blankat_dropdown'].disabled = False
                self.hs['L[0][0][0][1][0][1][5][2]_missing_start_text'].disabled = False
                self.hs['L[0][0][0][1][0][1][5][3]_missing_end_text'].disabled = False
                self.hs['L[0][0][0][1][0][1][6][1]_auto_thres_text'].disabled = True
                self.hs['L[0][0][0][1][0][1][6][2]_auto_ref_fn_button'].disabled = True
        else:
            self.hs['L[0][0][0][1][0][1][6][0]_auto_detect_checkbox'].value = False
            self.hs['L[0][0][0][1][0][1][6][0]_auto_detect_checkbox'].disabled = True
            self.hs['L[0][0][0][1][0][1][5][1]_blankat_dropdown'].disabled = True
            self.hs['L[0][0][0][1][0][1][5][2]_missing_start_text'].disabled = True
            self.hs['L[0][0][0][1][0][1][5][3]_missing_end_text'].disabled = True
            self.hs['L[0][0][0][1][0][1][6][1]_auto_thres_text'].disabled = True
            self.hs['L[0][0][0][1][0][1][6][2]_auto_ref_fn_button'].disabled = True

    def reset_alg_param_widgets(self):
        for ii in range(3):
            self.alg_phs[ii].options = ''
            self.alg_phs[ii].description_tooltip = 'p'+str(ii).zfill(2)
        for ii in range(3, 7):
            self.alg_phs[ii].value = 0
            self.alg_phs[ii].description_tooltip = 'p'+str(ii).zfill(2)

    def set_alg_param_widgets(self):
        """
        h_set: list,
            the collection of the param handles
        enable_list: list
            an index list to a sub-set of h_set that to be enabled
        desc_list: list
            descriptions that are for each handle in enable_list
        """
        self.reset_alg_param_widgets()
        for h  in self.alg_phs:
            h.disabled = True
            layout = {'width':'23.5%', 'visibility':'hidden'}
            h.layout = layout
        alg = ALG_PARAM_DICT[self.tomo_selected_alg]
        for idx in alg.keys():
            self.alg_phs[idx].disabled = False
            layout = {'width':'23.5%', 'visibility':'visible'}
            self.alg_phs[idx].layout = layout
            if idx < 3:
                self.alg_phs[idx].options = alg[idx][1]
                self.alg_phs[idx].value = alg[idx][1][0]
            else:
                self.alg_phs[idx].value = alg[idx][1]
            self.alg_phs[idx].description_tooltip = alg[idx][2]
        # print(self.alg_phs[idx].description_tooltip)

    def read_alg_param_widgets(self):
        self.alg_param_dict = {}
        alg = ALG_PARAM_DICT[self.tomo_selected_alg]
        for idx in alg.keys():
            self.alg_param_dict[alg[idx][0]] = self.alg_phs[idx].value

    def reset_flt_param_widgets(self):
        for ii in range(6):
            self.flt_phs[ii].options = ''
            self.flt_phs[ii].description_tooltip = 'p'+str(ii).zfill(2)
        for ii in range(6, 12):
            self.flt_phs[ii].value = 0
            self.flt_phs[ii].description_tooltip = 'p'+str(ii).zfill(2)

    def set_flt_param_widgets(self):
        """
        h_set: list,
            the collection of the param handles
        enable_list: list
            an index list to a sub-set of h_set that to be enabled
        desc_list: list
            descriptions that are for each handle in enable_list
        """
        self.reset_flt_param_widgets()
        for h  in self.flt_phs:
            h.disabled = True
            # h.description = ''
            # layout = {'align_items':'flex-start',
            #           'width':f'{int(7.5*(self.form_sz[1]-98)/40)}px',
            #           'visibility':'hidden'}
            # h.layout = layout
        flt = FILTER_PARAM_DICT[self.tomo_left_box_selected_flt]
        for idx in flt.keys():
            self.flt_phs[idx].disabled = False
            # h.description = 'p'+str(idx).zfill(2)
            # layout = {'align_items':'flex-start',
            #           'width':f'{int(7.5*(self.form_sz[1]-98)/40)}px',
            #           'visibility':'visible'}
            # self.flt_phs[idx].layout = layout
            if idx < 6:
                self.flt_phs[idx].options = flt[idx][1]
                self.flt_phs[idx].value = flt[idx][1][0]
            else:
                self.flt_phs[idx].value = flt[idx][1]
            self.flt_phs[idx].description_tooltip = flt[idx][2]
        # print(self.flt_phs[idx].description_tooltip)

    def read_flt_param_widgets(self):
        self.filter_param_dict = {}
        flt = FILTER_PARAM_DICT[self.tomo_left_box_selected_flt]
        for idx in flt.keys():
            self.filter_param_dict[flt[idx][0]] = self.flt_phs[idx].value

    def L0_0_0_0_1_0_select_raw_h5_top_dir_button_click(self, a):
        self.restart()
        if len(a.files[0]) != 0:
            self.tomo_raw_data_top_dir = a.files[0]
            self.tomo_raw_data_top_dir_set = True
            self.hs['L[0][0][0][0][1][0]_select_raw_h5_top_dir_button'].initialdir = os.path.abspath(a.files[0])
            self.hs['L[0][0][0][0][2][0]_select_save_recon_dir_button'].initialdir = os.path.abspath(a.files[0])
            self.hs['L[0][0][0][0][3][0]_select_save_data_center_dir_button'].initialdir = os.path.abspath(a.files[0])
            self.hs['L[0][0][0][0][4][0]_select_save_debug_dir_button'].initialdir = os.path.abspath(a.files[0])
        else:
            self.tomo_raw_data_top_dir = None
            self.tomo_raw_data_top_dir_set = False
            self.hs['L[0][0][0][0][1][1]_select_raw_h5_top_dir_text'].value = 'Choose raw h5 top dir ...'
        self.tomo_filepath_configured = False
        self.hs['L[0][0][0][0][5][1]_confirm_file&path_text'].value='After setting directories, confirm to proceed ...'
        self.boxes_logic()

    def L0_0_0_0_2_0_select_save_recon_dir_button_click(self, a):
        self.restart()
        if not self.tomo_raw_data_top_dir_set:
            self.hs['L[0][0][0][0][5][1]_confirm_file&path_text'].value = 'Please specify raw h5 top directory first ...'
            self.hs['L[0][0][0][0][2][1]_select_save_recon_dir_text'].value = 'Choose top directory where recon subdirectories are saved...'
            self.tomo_recon_path_set = False
        else:
            if len(a.files[0]) != 0:
                self.tomo_recon_top_dir = a.files[0]
                self.tomo_recon_path_set = True
                self.hs['L[0][0][0][0][2][0]_select_save_recon_dir_button'].initialdir = os.path.abspath(a.files[0])
            else:
                self.tomo_recon_top_dir = None
                self.tomo_recon_path_set = False
                self.hs['L[0][0][0][0][2][1]_select_save_recon_dir_text'].value = 'Select top directory where recon subdirectories are saved...'
            self.hs['L[0][0][0][0][5][1]_confirm_file&path_text'].value='After setting directories, confirm to proceed ...'
        self.tomo_filepath_configured = False
        self.boxes_logic()

    def L0_0_0_0_3_0_select_save_data_center_dir_button_click(self, a):
        self.restart()
        if not self.tomo_raw_data_top_dir_set:
            self.hs['L[0][0][0][0][5][1]_confirm_file&path_text'].value = 'Please specify raw h5 top directory first ...'
            self.hs['L[0][0][0][0][3][1]_select_save_data_center_dir_text'].value='Select top directory where data_center will be created...'
            self.tomo_data_center_path_set = False
        else:
            if len(a.files[0]) != 0:
                self.tomo_data_center_top_dir = a.files[0]
                # print(self.tomo_data_center_path)
                self.tomo_data_center_path_set = True
                self.hs['L[0][0][0][0][3][0]_select_save_data_center_dir_button'].initialdir = os.path.abspath(a.files[0])
            else:
                self.tomo_data_center_top_dir = None
                self.tomo_data_center_path_set = False
                self.hs['L[0][0][0][0][3][1]_select_save_data_center_dir_text'].value='Select top directory where data_center will be created...'
            self.hs['L[0][0][0][0][5][1]_confirm_file&path_text'].value='After setting directories, confirm to proceed ...'
        self.tomo_filepath_configured = False
        self.boxes_logic()

    def L0_0_0_0_4_0_select_save_debug_dir_button_click(self, a):
        self.restart()
        if not self.tomo_raw_data_top_dir_set:
            self.hs['L[0][0][0][0][5][1]_confirm_file&path_text'].value = 'Please specify raw h5 top directory first ...'
            self.hs['L[0][0][0][0][4][1]_select_save_debug_dir_text'].value = 'Select top directory where debug dir will be created...'
            self.tomo_debug_path_set = False
        else:
            if len(a.files[0]) != 0:
                self.tomo_debug_top_dir = a.files[0]
                # print(self.tomo_debug_path)
                self.tomo_debug_path_set = True
                self.hs['L[0][0][0][0][4][0]_select_save_debug_dir_button'].initialdir = os.path.abspath(a.files[0])
            else:
                self.tomo_debug_top_dir = None
                self.tomo_debug_path_set = False
                self.hs['L[0][0][0][0][4][1]_select_save_debug_dir_text'].value = 'Select top directory where debug dir will be created...'
            self.hs['L[0][0][0][0][5][1]_confirm_file&path_text'].value='After setting directories, confirm to proceed ...'
        self.tomo_filepath_configured = False
        self.boxes_logic()

    def L0_0_0_0_4_2_save_debug_checkbox_change(self, a):
        self.restart()
        if a['owner'].value:
            self.tomo_use_debug = True
            self.hs['L[0][0][0][0][4][0]_select_save_debug_dir_button'].disabled = False
            self.hs['L[0][0][0][0][4][1]_select_save_debug_dir_text'].value = 'Select top directory where debug dir will be created...'
            self.hs['L[0][0][0][0][4][0]_select_save_debug_dir_button'].style.button_color = "orange"
        else:
            self.tomo_use_debug = False
            self.hs['L[0][0][0][0][4][0]_select_save_debug_dir_button'].disabled = True
            self.hs['L[0][0][0][0][4][1]_select_save_debug_dir_text'].value = 'Debug is disabled...'
            self.hs['L[0][0][0][0][4][0]_select_save_debug_dir_button'].style.button_color = "orange"
        self.tomo_filepath_configured = False
        self.boxes_logic()

    def L0_0_0_0_5_2_file_path_options_dropdown_change(self, a):
        self.restart()
        self.tomo_recon_type = a['owner'].value
        if self.tomo_recon_type == 'Trial Center':
            self.hs['L[0][0][0][0][3][0]_select_save_data_center_dir_button'].disabled = False
            self.hs['L[0][0][0][0][2][0]_select_save_recon_dir_button'].disabled = True
            self.hs['L[0][0][0][0][3][0]_select_save_data_center_dir_button'].style.button_color = "orange"
            self.hs['L[0][0][0][0][2][0]_select_save_recon_dir_button'].style.button_color = "orange"
        elif self.tomo_recon_type in ['Vol Recon: Single', 'Vol Recon: Multi']:
            self.hs['L[0][0][0][0][3][0]_select_save_data_center_dir_button'].disabled = True
            self.hs['L[0][0][0][0][2][0]_select_save_recon_dir_button'].disabled = False
            self.hs['L[0][0][0][0][3][0]_select_save_data_center_dir_button'].style.button_color = "orange"
            self.hs['L[0][0][0][0][2][0]_select_save_recon_dir_button'].style.button_color = "orange"
        self.tomo_filepath_configured = False
        self.boxes_logic()

    def L0_0_0_0_5_0_confirm_file_path_button_click(self, a):
        if self.tomo_recon_type == 'Trial Center':
            if (self.tomo_raw_data_top_dir_set & self.tomo_data_center_path_set):
                if self.tomo_use_debug:
                    if self.tomo_debug_path_set:
                        self.tomo_data_center_path = os.path.join(self.tomo_data_center_top_dir, 'data_center')
                        self.tomo_filepath_configured = True
                    else:
                        self.hs['L[0][0][0][0][5][1]_confirm_file&path_text'].value = 'You need to select the top directory where debug dir can be created...'
                        self.tomo_filepath_configured = False
                else:
                    self.tomo_data_center_path = os.path.join(self.tomo_data_center_top_dir, 'data_center')
                    self.tomo_filepath_configured = True
            else:
                self.hs['L[0][0][0][0][5][1]_confirm_file&path_text'].value = 'You need to select the top raw dir and top dir where debug dir can be created...'
                self.tomo_filepath_configured = False
        elif self.tomo_recon_type in ['Vol Recon: Single', 'Vol Recon: Multi']:
            if (self.tomo_raw_data_top_dir_set & self.tomo_recon_path_set):
                self.tomo_filepath_configured = True
            else:
                self.hs['L[0][0][0][0][5][1]_confirm_file&path_text'].value = 'You need to select the top raw dir and top dir where recon dir can be created...'
                self.tomo_filepath_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_0_0_scan_id_text_change(self, a):
        self.tomo_scan_id = list(a['owner'].value)
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_0_1_rot_cen_text_change(self, a):
        self.tomo_rot_cen = list(a['owner'].value)
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_0_2_cen_win_left_text_change(self, a):
        self.tomo_cen_win_s = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_0_3_cen_win_wz_text_change(self, a):
        self.tomo_cen_win_w = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_1_0_use_alt_flat_checkbox_change(self, a):
        self.tomo_use_alt_flat = a['owner'].value
        if self.tomo_use_alt_flat:
            self.hs['L[0][0][0][1][0][1][2][0]_use_fake_flat_checkbox'].value = False
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_1_1_alt_flat_file_button_change(self, a):
        if len(a.files[0]) != 0:
            self.tomo_alt_flat_file = a.files[0]
        else:
            self.tomo_alt_flat_file = None
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_1_2_use_alt_dark_checkbox_change(self, a):
        self.tomo_use_alt_dark = a['owner'].value
        if self.hs['L[0][0][0][1][0][1][1][2]_use_alt_dark_checkbox'].value:
            self.hs['L[0][0][0][1][0][1][2][2]_use_fake_dark_checkbox'].value = False
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_1_3_alt_dark_file_button_change(self, a):
        if len(a.files[0]) != 0:
            self.tomo_alt_dark_file = a.files[0]
        else:
            self.tomo_alt_dark_file = None
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_2_0_use_fake_flat_checkbox_change(self, a):
        self.tomo_use_fake_flat = a['owner'].value
        if self.hs['L[0][0][0][1][0][1][2][0]_use_fake_flat_checkbox'].value:
            self.hs['L[0][0][0][1][0][1][1][0]_use_alt_flat_checkbox'].value = False
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_2_1_fake_flat_val_text_change(self, a):
        self.tomo_fake_flat_val = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_2_2_use_fake_dark_checkbox_change(self, a):
        self.tomo_use_fake_dark = a['owner'].value
        if self.hs['L[0][0][0][1][0][1][2][2]_use_fake_dark_checkbox'].value:
            self.hs['L[0][0][0][1][0][1][1][2]_use_alt_dark_checkbox'].value = False
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_2_3_fake_dark_val_text_change(self, a):
        self.tomo_fake_dark_val = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_3_0_sli_start_text_change(self, a):
        self.tomo_sli_s = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_3_1_sli_end_text_change(self, a):
        self.tomo_sli_e = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_3_2_chunk_sz_text_change(self, a):
        self.tomo_chunk_sz = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_3_3_margin_sz_text_change(self, a):
        self.tomo_margin = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_4_0_rm_zinger_checkbox_change(self, a):
        self.tomo_use_rm_zinger = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_4_1_zinger_level_text_change(self, a):
        self.tomo_zinger_val = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_4_2_use_mask_checkbox_change(self, a):
        self.tomo_use_mask = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_4_3_mask_ratio_text_change(self, a):
        self.tomo_mask_ratio = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_1_0_1_0_alg_options_dropdown_change(self, a):
        self.tomo_selected_alg = a['owner'].value
        self.set_alg_param_widgets()
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_1_0_1_1_alg_p00_dropdown_change(self, a):
        self.tomo_alg_p01 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_1_0_1_2_alg_p01_dropdown_change(self, a):
        self.tomo_alg_p02 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_1_0_1_3_alg_p02_dropdown_change(self, a):
        self.tomo_alg_p03 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_1_0_2_0_alg_p03_text_change(self, a):
        self.tomo_alg_p04 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_1_0_2_1_alg_p04_text_change(self, a):
        self.tomo_alg_p05 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_1_0_2_2_alg_p05_text_change(self, a):
        self.tomo_alg_p06 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_1_0_2_3_alg_p06_text_change(self, a):
        self.tomo_alg_p07 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_0_0_is_wedge_checkbox_change(self, a):
        self.tomo_is_wedge = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_0_1_blankat_dropdown_change(self, a):
        self.tomo_wedge_blankat = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_0_2_missing_start_text_change(self, a):
        self.tomo_wedge_missing_s = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_0_3_missing_end_text_change(self, a):
        self.tomo_wedge_missing_e = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_1_0_auto_detect_checkbox_change(self, a):
        self.tomo_wedge_ang_auto_det = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_1_1_auto_thres_text_change(self, a):
        self.tomo_wedge_ang_auto_det_thres = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_1_2_auto_ref_fn_button_change(self, a):
        if len(a.files[0]) != 0:
            self.tomo_wedge_ang_auto_det_ref_fn = a['owner'].value
        else:
            self.tomo_wedge_ang_auto_det_ref_fn = None
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_1_3_fiji_viewer_checkbox_change(self, a):
        pass

    def L0_0_1_0_1_1_4_sli_range_slider_change(self, a):
        pass

    def L0_0_1_0_1_0_0_filter_config_box_left_box_filter_dropdown_change(self, a):
        self.tomo_left_box_selected_flt = a['owner'].value
        self.set_flt_param_widgets()
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_0_1_filter_config_box_left_box_move_button_change(self, a):
        # self.filter_param_dict = {}
        self.read_flt_param_widgets()
        if ((len(self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].options) == 1) &
            (self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].options[0] == 'None')):
            self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].options = [self.tomo_left_box_selected_flt,]
            self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].value = [self.tomo_left_box_selected_flt,]
            self.tomo_right_filter_dict[0] = {self.tomo_left_box_selected_flt:self.filter_param_dict}
        else:
            a = list(self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].options)
            a.append(self.tomo_left_box_selected_flt)
            self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].options = a
            self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].value = \
                self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].options
            idx = len(a) - 1
            self.tomo_right_filter_dict[idx] = {self.tomo_left_box_selected_flt:self.filter_param_dict}
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_0_2_filter_config_box_left_box_p00_change(self, a):
        self.tomo_left_box_selected_flt_p00 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_0_3_filter_config_box_left_box_p01_change(self, a):
        self.tomo_left_box_selected_flt_p01 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_0_4_filter_config_box_left_box_p02_change(self, a):
        self.tomo_left_box_selected_flt_p02 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_0_5_filter_config_box_left_box_p03_change(self, a):
        self.tomo_left_box_selected_flt_p03 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_0_6_filter_config_box_left_box_p04_change(self, a):
        self.tomo_left_box_selected_flt_p04 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_0_7_filter_config_box_left_box_p05_change(self, a):
        self.tomo_left_box_selected_flt_p05 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_0_8_filter_config_box_left_box_p06_change(self, a):
        self.tomo_left_box_selected_flt_p06 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_0_9_filter_config_box_left_box_p07_change(self, a):
        self.tomo_left_box_selected_flt_p07 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_0_10_filter_config_box_left_box_p08_change(self, a):
        self.tomo_left_box_selected_flt_p08 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_0_11_filter_config_box_left_box_p09_change(self, a):
        self.tomo_left_box_selected_flt_p09 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_0_12_filter_config_box_left_box_p10_change(self, a):
        self.tomo_left_box_selected_flt_p10 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_0_13_filter_config_box_left_box_p11_change(self, a):
        self.tomo_left_box_selected_flt_p11 = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_1_0_filter_config_box_right_box_selectmultiple_change(self, a):
        self.tomo_right_list_filter = a['owner'].value
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_1_1_filter_config_box_right_box_move_up_button_change(self, a):
        if (len(self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].options) == 1):
            pass
        else:
            a = np.array(self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].options)
            idxs = np.array(self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].index)
            cnt = 0
            for b in idxs:
                if b == 0:
                    idxs[cnt] = b
                else:
                    a[b], a[b-1] = a[b-1], a[b]
                    self.tomo_right_filter_dict[b], self.tomo_right_filter_dict[b-1] =\
                        self.tomo_right_filter_dict[b-1], self.tomo_right_filter_dict[b]
                    idxs[cnt] = b-1
                cnt += 1
            print(self.tomo_right_filter_dict)
            self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].options = list(a)
            self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].value = list(a[idxs])
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_1_2_filter_config_box_right_box_move_dn_button_change(self, a):
        if (len(self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].options) == 1):
            pass
        else:
            a = np.array(self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].options)
            idxs = np.array(self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].index)
            cnt = 0
            for b in idxs:
                if b == (len(a)-1):
                    idxs[cnt] = b
                else:
                    a[b], a[b+1] = a[b+1], a[b]
                    self.tomo_right_filter_dict[b], self.tomo_right_filter_dict[b+1] =\
                        self.tomo_right_filter_dict[b+1], self.tomo_right_filter_dict[b]
                    idxs[cnt] = b+1
                cnt += 1
            # print(a, idxs)
            print(self.tomo_right_filter_dict)
            self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].options = list(a)
            self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].value = list(a[idxs])
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_1_3_filter_config_box_right_box_remove_button_change(self, a):
        a = list(self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].options)
        idxs = list(self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].index)
        d = {}
        for b in idxs:
            del a[b]
            del self.tomo_right_filter_dict[b]
        if len(a)>0:
            self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].options = list(a)
            self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].value = [a[0],]
            cnt = 0
            for ii in sorted(self.tomo_right_filter_dict.keys()):
                d[cnt] = self.tomo_right_filter_dict[ii]
                cnt += 1
            self.tomo_right_filter_dict = d
        else:
            self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].options = ['None',]
            self.hs['L[0][0][1][0][1][1][0]_filter_config_right_box_selectmultiple'].value = ['None',]
            self.tomo_right_filter_dict = {0:{}}
        print(self.tomo_right_filter_dict)
        self.tomo_data_configured = False
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_1_4_filter_config_box_right_box_finish_button_change(self, a):
        pass

    def L0_0_1_0_1_3_filter_config_box_confirm_button(self, a):
        self.read_alg_param_widgets()

    def L0_0_1_2_1_1_reecon_button_change(self, a):
        pass











