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
from gui_components import get_handles, get_decendant

class tomo_recon_gui():
    def __init__(self, form_sz=[650, 740]):
        self.hs = {}
        self.form_sz = form_sz

        self.tomo_recon_type = 'Trial Center'
        self.tomo_use_debug = False

    def build_gui(self):
        #################################################################################################################
        #                                                                                                               #
        #                                                    TOMO RECON                                                 #
        #                                                                                                               #
        #################################################################################################################
        ## ## ## define 2D_XANES_tabs layout -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-86}px', 'height':f'{self.form_sz[0]-128}px'}
        self.hs['L[0][0][0]_config_input_form'] = widgets.VBox()
        self.hs['L[0][0][1]_data&filter_form'] = widgets.VBox()
        self.hs['L[0][0][2]_reg&review_form'] = widgets.VBox()
        self.hs['L[0][0][3]_analysis&display_form'] = widgets.VBox()
        self.hs['L[0][0][0]_config_input_form'].layout = layout
        self.hs['L[0][0][1]_data&filter_form'].layout = layout
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
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.56*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1]_recon_options_box'] = widgets.VBox()
        self.hs['L[0][0][0][1]_recon_options_box'].layout = layout
        ## ## ## ## ## label config_data box -- start
        layout = {'justify-content':'center', 'align-items':'center','align-contents':'center','border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][0]_recon_options_title_box'] = widgets.HBox()
        self.hs['L[0][0][0][1][0]_recon_options_title_box'].layout = layout
        # self.hs['L[0][0][0][1][0][0]_recon_options_title'] = widgets.Text(value='Config Files', disabled=True)
        self.hs['L[0][0][0][1][0][0]_recon_options_title'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Recon Options' + '</span>')
        layout = {'left':'41%', 'background-color':'white', 'color':'cyan'}
        self.hs['L[0][0][0][1][0][0]_recon_options_title'].layout = layout
        self.hs['L[0][0][0][1][0]_recon_options_title_box'].children = get_handles(self.hs, 'L[0][0][0][1][0]_config_data_title_box', -1)
        ## ## ## ## ## label config_data box -- end

        ## ## ## ## ## config data parameters -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.49*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][1]_recon_options_config_box'] = widgets.VBox()
        self.hs['L[0][0][0][1][1]_recon_options_config_box'].layout = layout
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-104}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][1][0]_recon_config_box0'] = widgets.HBox()
        self.hs['L[0][0][0][1][1][0]_recon_config_box0'].layout = layout
        self.hs['L[0][0][0][1][1][0][0]_scan_id_text'] = widgets.IntText(value=0, description='Scan id', disabled=True)
        layout = {'width':'24%', 'height':'90%'}
        self.hs['L[0][0][0][1][1][0][0]_scan_id_text'].layout = layout
        self.hs['L[0][0][0][1][1][0][1]_rot_cen_text'] = widgets.FloatText(value=1280.0, description='Center', disabled=True)
        layout = {'width':'24%', 'height':'90%'}
        self.hs['L[0][0][0][1][1][0][1]_rot_cen_text'].layout = layout
        self.hs['L[0][0][0][1][1][0][2]_cen_win_left_text'] = widgets.IntText(value=1240, description='Cen Win L', disabled=True)
        layout = {'width':'24%', 'height':'90%'}
        self.hs['L[0][0][0][1][1][0][2]_cen_win_left_text'].layout = layout
        self.hs['L[0][0][0][1][1][0][3]_cen_win_wz_text'] = widgets.IntText(value=80, description='Cen Win W', disabled=True)
        layout = {'width':'24%', 'height':'90%'}
        self.hs['L[0][0][0][1][1][0][3]_cen_win_wz_text'].layout = layout
        self.hs['L[0][0][0][1][1][0]_recon_config_box0'].children = get_handles(self.hs, 'L[0][0][0][1][1][0]_data_preprocessing_options_box0', -1)
        self.hs['L[0][0][0][1][1][0][0]_scan_id_text'].observe(self.L0_0_0_1_1_0_0_scan_id_text_change, names='value')
        self.hs['L[0][0][0][1][1][0][1]_rot_cen_text'].observe(self.L0_0_0_1_1_0_1_rot_cen_text_change, names='value')
        self.hs['L[0][0][0][1][1][0][2]_cen_win_left_text'].observe(self.L0_0_0_1_1_0_2_cen_win_left_text_change, names='value')
        self.hs['L[0][0][0][1][1][0][3]_cen_win_wz_text'].observe(self.L0_0_0_1_1_0_3_cen_win_wz_text_change, names='value')

        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-104}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][1][3]_chunk_config_box3'] = widgets.HBox()
        self.hs['L[0][0][0][1][1][3]_chunk_config_box3'].layout = layout
        self.hs['L[0][0][0][1][1][3][0]_sli_start_text'] = widgets.IntText(value=0, description='Sli Start', disabled=True)
        layout = {'width':'24%', 'height':'90%'}
        self.hs['L[0][0][0][1][1][3][0]_sli_start_text'].layout = layout
        self.hs['L[0][0][0][1][1][3][1]_sli_end_text'] = widgets.FloatText(value=0, description='Sli End', disabled=True)
        layout = {'width':'24%', 'height':'90%'}
        self.hs['L[0][0][0][1][1][3][1]_sli_end_text'].layout = layout
        self.hs['L[0][0][0][1][1][3][2]_chunk_sz_text'] = widgets.IntText(value=200, description='Chunk Sz', disabled=True)
        layout = {'width':'24%', 'height':'90%'}
        self.hs['L[0][0][0][1][1][3][2]_chunk_sz_text'].layout = layout
        self.hs['L[0][0][0][1][1][3][3]_margin_sz_text'] = widgets.IntText(value=15, description='Margin Sz', disabled=True)
        layout = {'width':'24%', 'height':'90%'}
        self.hs['L[0][0][0][1][1][3][3]_margin_sz_text'].layout = layout
        self.hs['L[0][0][0][1][1][3]_chunk_config_box3'].children = get_handles(self.hs, 'L[0][0][0][1][1][3]_chunk_config_box3', -1)
        self.hs['L[0][0][0][1][1][3][0]_sli_start_text'].observe(self.L0_0_0_1_1_3_0_sli_start_text_change, names='value')
        self.hs['L[0][0][0][1][1][3][1]_sli_end_text'].observe(self.L0_0_0_1_1_3_1_sli_end_text_change, names='value')
        self.hs['L[0][0][0][1][1][3][2]_chunk_sz_text'].observe(self.L0_0_0_1_1_3_2_chunk_sz_text_change, names='value')
        self.hs['L[0][0][0][1][1][3][3]_margin_sz_text'].observe(self.L0_0_0_1_1_3_3_margin_sz_text_change, names='value')

        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-104}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][1][1]_alt_flat/dark_options_box1'] = widgets.HBox()
        self.hs['L[0][0][0][1][1][1]_alt_flat/dark_options_box1'].layout = layout
        layout = {'width':'20%'}
        self.hs['L[0][0][0][1][1][1][0]_use_alt_flat_checkbox'] = widgets.Checkbox(value=False, description='Alt Flat', disabled=True)
        self.hs['L[0][0][0][1][1][1][0]_use_alt_flat_checkbox'].layout = layout
        layout = {'left':'12.5%', 'width':'15%'}
        self.hs['L[0][0][0][1][1][1][1]_alt_flat_file_button'] = SelectFilesButton(option='askopenfilename',
                                                                                   **{'open_filetypes': (('h5 files', '*.h5'),)})
        self.hs['L[0][0][0][1][1][1][1]_alt_flat_file_button'].description = 'Alt Flat File'
        self.hs['L[0][0][0][1][1][1][1]_alt_flat_file_button'].layout = layout
        layout = {'left':'12.5%', 'width':'20%'}
        self.hs['L[0][0][0][1][1][1][2]_use_alt_dark_checkbox'] = widgets.Checkbox(value=False, description='Alt Dark', disabled=True)
        self.hs['L[0][0][0][1][1][1][2]_use_alt_dark_checkbox'].layout = layout
        layout = {'left':'25%', 'width':'15%'}
        self.hs['L[0][0][0][1][1][1][3]_alt_dark_file_button'] = SelectFilesButton(option='askopenfilename',
                                                                                   **{'open_filetypes': (('h5 files', '*.h5'),)})
        self.hs['L[0][0][0][1][1][1][3]_alt_dark_file_button'].description = 'Alt Dark File'
        self.hs['L[0][0][0][1][1][1][3]_alt_dark_file_button'].layout = layout
        self.hs['L[0][0][0][1][1][1]_alt_flat/dark_options_box1'].children = get_handles(self.hs, 'L[0][0][0][1][1][1]_alt_flat/dark_options_box1', -1)
        self.hs['L[0][0][0][1][1][1][0]_use_alt_flat_checkbox'].observe(self.L0_0_0_1_1_1_0_use_alt_flat_checkbox_change, names='value')
        self.hs['L[0][0][0][1][1][1][1]_alt_flat_file_button'].observe(self.L0_0_0_1_1_1_1_alt_flat_file_button_change, names='value')
        self.hs['L[0][0][0][1][1][1][2]_use_alt_dark_checkbox'].observe(self.L0_0_0_1_1_1_2_use_alt_dark_checkbox_change, names='value')
        self.hs['L[0][0][0][1][1][1][3]_alt_dark_file_button'].observe(self.L0_0_0_1_1_1_3_alt_dark_file_button_change, names='value')

        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-104}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][1][2]_fake_flat/dark_options_box2'] = widgets.HBox()
        self.hs['L[0][0][0][1][1][2]_fake_flat/dark_options_box2'].layout = layout
        layout = {'width':'24%'}
        self.hs['L[0][0][0][1][1][2][0]_use_fake_flat_checkbox'] = widgets.Checkbox(value=False,
                                                                                    description='Fake Flat',
                                                                                    disabled=True)
        self.hs['L[0][0][0][1][1][2][0]_use_fake_flat_checkbox'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][1][2][1]_fake_flat_val_text'] = widgets.FloatText(value=10000.0,
                                                                                disabled=True)
        self.hs['L[0][0][0][1][1][2][1]_fake_flat_val_text'].layout = layout
        layout = {'width':'24%'}
        self.hs['L[0][0][0][1][1][2][2]_use_fake_dark_checkbox'] = widgets.Checkbox(value=False,
                                                                                    description='Fake Dark',
                                                                                    disabled=True)
        self.hs['L[0][0][0][1][1][2][2]_use_fake_dark_checkbox'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][1][2][3]_fake_dark_val_text'] = widgets.FloatText(value=100.0,
                                                                                disabled=True)
        self.hs['L[0][0][0][1][1][2][3]_fake_dark_val_text'].layout = layout
        self.hs['L[0][0][0][1][1][2][0]_use_fake_flat_checkbox'].observe(self.L0_0_0_1_1_2_0_use_fake_flat_checkbox_change, names='value')
        self.hs['L[0][0][0][1][1][2][1]_fake_flat_val_text'].observe(self.L0_0_0_1_1_2_1_fake_flat_val_text_change, names='value')
        self.hs['L[0][0][0][1][1][2][2]_use_fake_dark_checkbox'].observe(self.L0_0_0_1_1_2_2_use_fake_dark_checkbox_change, names='value')
        self.hs['L[0][0][0][1][1][2][3]_fake_dark_val_text'].observe(self.L0_0_0_1_1_2_3_fake_dark_val_text_change, names='value')
        self.hs['L[0][0][0][1][1][2]_fake_flat/dark_options_box2'].children = get_handles(self.hs, 'L[0][0][0][1][1][2]_data_preprocessing_options_box2', -1)

        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-104}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][1][4]_misc_options_box4'] = widgets.HBox()
        self.hs['L[0][0][0][1][1][4]_misc_options_box4'].layout = layout
        layout = {'width':'24%'}
        self.hs['L[0][0][0][1][1][4][0]_rm_zinger_checkbox'] = widgets.Checkbox(value=False,
                                                                                    description='Rm Zinger',
                                                                                    disabled=True)
        self.hs['L[0][0][0][1][1][4][0]_rm_zinger_checkbox'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][1][4][1]_zinger_level_text'] = widgets.FloatText(value=500.0,
                                                                                disabled=True)
        self.hs['L[0][0][0][1][1][4][1]_zinger_level_text'].layout = layout
        layout = {'width':'24%'}
        self.hs['L[0][0][0][1][1][4][2]_use_mask_checkbox'] = widgets.Checkbox(value=False,
                                                                                    description='Use Mask',
                                                                                    disabled=True)
        self.hs['L[0][0][0][1][1][4][2]_use_mask_checkbox'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][1][4][3]_mask_ratio_text'] = widgets.FloatText(value=1,
                                                                                disabled=True)
        self.hs['L[0][0][0][1][1][4][3]_mask_ratio_text'].layout = layout
        self.hs['L[0][0][0][1][1][4][0]_rm_zinger_checkbox'].observe(self.L0_0_0_1_1_4_0_rm_zinger_checkbox_change, names='value')
        self.hs['L[0][0][0][1][1][4][1]_zinger_level_text'].observe(self.L0_0_0_1_1_4_1_zinger_level_text_change, names='value')
        self.hs['L[0][0][0][1][1][4][2]_use_mask_checkbox'].observe(self.L0_0_0_1_1_4_2_use_mask_checkbox_change, names='value')
        self.hs['L[0][0][0][1][1][4][3]_mask_ratio_text'].observe(self.L0_0_0_1_1_4_3_mask_ratio_text_change, names='value')
        self.hs['L[0][0][0][1][1][4]_misc_options_box4'].children = get_handles(self.hs, 'L[0][0][0][1][1][4]_misc_options_box4', -1)

        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-104}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][1][5]_alg_options_box5'] = widgets.HBox()
        self.hs['L[0][0][0][1][1][5]_alg_options_box5'].layout = layout
        layout = {'width':'24%'}
        self.hs['L[0][0][0][1][1][5][0]_alg_options_dropdown'] = widgets.Dropdown(value='gridrec',
                                                                                  options=['gridrec', 'sirt', 'tv', 'mlem', 'astra'],
                                                                                  description='algs',
                                                                                  disabled=True)
        self.hs['L[0][0][0][1][1][5][0]_alg_options_dropdown'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][1][5][1]_alg_param1_dropdown'] = widgets.Dropdown(value='',
                                                                                 options=[''],
                                                                                 description='param1',
                                                                                 disabled=True)
        self.hs['L[0][0][0][1][1][5][1]_alg_param1_dropdown'].layout = layout
        layout = {'width':'24%'}
        self.hs['L[0][0][0][1][1][5][2]_alg_param2_dropdown'] = widgets.Dropdown(value='',
                                                                                 options=[''],
                                                                                 description='param2',
                                                                                 disabled=True)
        self.hs['L[0][0][0][1][1][5][2]_alg_param2_dropdown'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][1][5][3]_alg_param3_text'] = widgets.FloatText(value=0.0,
                                                                              description='param3',
                                                                              disabled=True)
        self.hs['L[0][0][0][1][1][5][3]_alg_param3_text'].layout = layout
        self.hs['L[0][0][0][1][1][5][0]_alg_options_dropdown'].observe(self.L0_0_0_1_1_5_0_alg_options_dropdown_change, names='value')
        self.hs['L[0][0][0][1][1][5][1]_alg_param1_dropdown'].observe(self.L0_0_0_1_1_5_1_alg_param1_dropdown_change, names='value')
        self.hs['L[0][0][0][1][1][5][2]_alg_param2_dropdown'].observe(self.L0_0_0_1_1_5_2_alg_param2_dropdown_change, names='value')
        self.hs['L[0][0][0][1][1][5][3]_alg_param3_text'].observe(self.L0_0_0_1_1_5_3_alg_param3_text_change, names='value')
        self.hs['L[0][0][0][1][1][5]_alg_options_box5'].children = get_handles(self.hs, 'L[0][0][0][1][1][5]_alg_options_box5', -1)

        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-104}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][0][1][1][6]_alg_options_box6'] = widgets.HBox()
        self.hs['L[0][0][0][1][1][6]_alg_options_box6'].layout = layout
        layout = {'width':'24%'}
        self.hs['L[0][0][0][1][1][6][0]_alg_param4_text'] = widgets.FloatText(value=0,
                                                                              description='param4',
                                                                              disabled=True)
        self.hs['L[0][0][0][1][1][6][0]_alg_param4_text'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][1][6][1]_alg_param5_text'] = widgets.FloatText(value=0,
                                                                              description='param5',
                                                                              disabled=True)
        self.hs['L[0][0][0][1][1][6][1]_alg_param5_text'].layout = layout
        layout = {'width':'24%'}
        self.hs['L[0][0][0][1][1][6][2]_alg_param6_text'] = widgets.FloatText(value=0,
                                                                              description='param6',
                                                                              disabled=True)
        self.hs['L[0][0][0][1][1][6][2]_alg_param6_text'].layout = layout
        layout = {'width':'23.5%'}
        self.hs['L[0][0][0][1][1][6][3]_alg_param7_text'] = widgets.FloatText(value=0.0,
                                                                              description='param7',
                                                                              disabled=True)
        self.hs['L[0][0][0][1][1][6][3]_alg_param7_text'].layout = layout
        self.hs['L[0][0][0][1][1][6][0]_alg_param4_text'].observe(self.L0_0_0_1_1_6_0_alg_param4_text_change, names='value')
        self.hs['L[0][0][0][1][1][6][1]_alg_param5_text'].observe(self.L0_0_0_1_1_6_1_alg_param5_text_change, names='value')
        self.hs['L[0][0][0][1][1][6][2]_alg_param6_text'].observe(self.L0_0_0_1_1_6_2_alg_param6_text_change, names='value')
        self.hs['L[0][0][0][1][1][6][3]_alg_param7_text'].observe(self.L0_0_0_1_1_6_3_alg_param7_text_change, names='value')
        self.hs['L[0][0][0][1][1][6]_alg_options_box6'].children = get_handles(self.hs, 'L[0][0][0][1][1][6]_alg_options_box6', -1)

        self.hs['L[0][0][0][1][1]_recon_options_config_box'].children = get_handles(self.hs, 'L[0][0][0][1][1]_recon_options_box', -1)
        ## ## ## ## ## config data parameters -- end

        self.hs['L[0][0][0][1]_recon_options_box'].children = get_handles(self.hs, 'L[0][0][0][1]_config_indices_box', -1)
        ## ## ## ## define widgets recon_options_box -- end

        self.hs['L[0][0][0]_config_input_form'].children = get_handles(self.hs, 'L[0][0][0]_config_input_form', -1)
        ## ## ## define boxes in config_input_form -- end


        ## ## ## define boxes in data&filter_form -- start
        ## ## ## ## define widgets wedge_data_config_box -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.21*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][1][0]_wedge_data_config_box'] = widgets.VBox()
        self.hs['L[0][0][1][0]_wedge_data_config_box'].layout = layout

        ## ## ## ## ## label wedge_data_config_box -- start
        layout = {'justify-content':'center', 'align-items':'center','align-contents':'center','border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][1][0][0]_wedge_data_config_title_box'] = widgets.HBox()
        self.hs['L[0][0][1][0][0]_wedge_data_config_title_box'].layout = layout
        self.hs['L[0][0][1][0][0][0]_wedge_data_config_title'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Wedge Data' + '</span>')
        layout = {'left':'43.25%', 'background-color':'white', 'color':'cyan'}
        self.hs['L[0][0][1][0][0][0]_wedge_data_config_title'].layout = layout
        self.hs['L[0][0][1][0][0]_wedge_data_config_title_box'].children = get_handles(self.hs, 'L[0][0][1][0][0]_wedge_data_config_title_box', -1)
        ## ## ## ## ## label wedge_data_config_box -- end

        ## ## ## ## ## config wedge_data -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.14*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][1][0][1]_wedge_data_config_box'] = widgets.VBox()
        self.hs['L[0][0][1][0][1]_wedge_data_config_box'].layout = layout

        ## ## ## ## ## ## config wedge_data box0 -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-104}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][1][0][1][0]_wedge_options_box0'] = widgets.HBox()
        self.hs['L[0][0][1][0][1][0]_wedge_options_box0'].layout = layout
        layout = {'width':'20%'}
        self.hs['L[0][0][1][0][1][0][0]_is_wedge_checkbox'] = widgets.Checkbox(value=False,
                                                                               description='Is Wedge',
                                                                               disabled=True,
                                                                               indent=False)
        self.hs['L[0][0][1][0][1][0][0]_is_wedge_checkbox'].layout = layout
        layout = {'width':'20%'}
        self.hs['L[0][0][1][0][1][0][1]_blankat_dropdown'] = widgets.Dropdown(value=90,
                                                                              options=[0, 90],
                                                                              description='Blandk At',
                                                                              disabled=True)
        self.hs['L[0][0][1][0][1][0][1]_blankat_dropdown'].layout = layout
        layout = {'width':'20%'}
        self.hs['L[0][0][1][0][1][0][2]_missing_start_text'] = widgets.IntText(value=0,
                                                                               description='Miss S',
                                                                               disabled=True)
        self.hs['L[0][0][1][0][1][0][2]_missing_start_text'].layout = layout
        layout = {'width':'20%'}
        self.hs['L[0][0][1][0][1][0][3]_missing_end_text'] =  widgets.IntText(value=0,
                                                                              description='Miss E',
                                                                              disabled=True)
        self.hs['L[0][0][1][0][1][0][3]_missing_end_text'].layout = layout
        self.hs['L[0][0][1][0][1][0][0]_is_wedge_checkbox'].observe(self.L0_0_1_0_1_0_0_is_wedge_checkbox_change, names='value')
        self.hs['L[0][0][1][0][1][0][1]_blankat_dropdown'].observe(self.L0_0_1_0_1_0_1_blankat_dropdown_change, names='value')
        self.hs['L[0][0][1][0][1][0][2]_missing_start_text'].observe(self.L0_0_1_0_1_0_2_missing_start_text_change, names='value')
        self.hs['L[0][0][1][0][1][0][3]_missing_end_text'].observe(self.L0_0_1_0_1_0_3_missing_end_text_change, names='value')
        self.hs['L[0][0][1][0][1][0]_wedge_options_box0'].children = get_handles(self.hs, 'L[0][0][1][0][1][0]_wedge_options_box0', -1)
        ## ## ## ## ## ## config wedge_data box0 -- end

        ## ## ## ## ## ## config wedge_data box1 -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-104}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][1][0][1][1]_wedge_options_box1'] = widgets.HBox()
        self.hs['L[0][0][1][0][1][1]_wedge_options_box1'].layout = layout
        HTML('<style> .widget-hbox .widget-label { max-width:350ex; text-align:left} </style>')
        layout = {'width':'13%'}
        self.hs['L[0][0][1][0][1][1][0]_auto_detect_checkbox'] = widgets.Checkbox(value=True,
                                                                                  description='Auto Det',
                                                                                  disabled=True,
                                                                                  indent=False)
        self.hs['L[0][0][1][0][1][1][0]_auto_detect_checkbox'].layout = layout
        layout = {'width':'20%'}
        self.hs['L[0][0][1][0][1][1][1]_auto_thres_text'] = widgets.FloatText(value=0.0,
                                                                              description='Auto Thres',
                                                                              disabled=True)
        self.hs['L[0][0][1][0][1][1][1]_auto_thres_text'].layout = layout
        layout = {'width':'15%'}
        self.hs['L[0][0][1][0][1][1][2]_auto_ref_fn_button'] = SelectFilesButton(option='askopenfilename',
                                                                                 **{'open_filetypes': (('txt files', '*.txt'),)})
        self.hs['L[0][0][1][0][1][1][2]_auto_ref_fn_button'].layout = layout
        layout = {'width':'13%'}
        self.hs['L[0][0][1][0][1][1][3]_fiji_viewer_checkbox'] = widgets.Checkbox(value=False,
                                                                                  description='fiji view',
                                                                                  disabled=True,
                                                                                  indent=False)
        self.hs['L[0][0][1][0][1][1][3]_fiji_viewer_checkbox'].layout = layout
        layout = {'width':'39%'}
        self.hs['L[0][0][1][0][1][1][4]_sli_range_slider'] = widgets.IntSlider(value=0,
                                                                               description='sli',
                                                                               min=0,
                                                                               max=100,
                                                                               disabled=True,
                                                                               indent=False)
        self.hs['L[0][0][1][0][1][1][4]_sli_range_slider'].layout = layout
        self.hs['L[0][0][1][0][1][1][0]_auto_detect_checkbox'].observe(self.L0_0_1_0_1_1_0_auto_detect_checkbox_change, names='value')
        self.hs['L[0][0][1][0][1][1][1]_auto_thres_text'].observe(self.L0_0_1_0_1_1_1_auto_thres_text_change, names='value')
        self.hs['L[0][0][1][0][1][1][2]_auto_ref_fn_button'].on_click(self.L0_0_1_0_1_1_2_auto_ref_fn_button_change)
        self.hs['L[0][0][1][0][1][1][3]_fiji_viewer_checkbox'].observe(self.L0_0_1_0_1_1_3_fiji_viewer_checkbox_change, names='value')
        self.hs['L[0][0][1][0][1][1][4]_sli_range_slider'].observe(self.L0_0_1_0_1_1_4_sli_range_slider_change, names='value')
        self.hs['L[0][0][1][0][1][1]_wedge_options_box1'].children = get_handles(self.hs, 'L[0][0][1][0][1][1]_wedge_options_box1', -1)
        ## ## ## ## ## ## config wedge_data box1 -- end

        self.hs['L[0][0][1][0][1]_wedge_data_config_box'].children = get_handles(self.hs, 'L[0][0][1][0][1]_wedge_data_config_box', -1)
        ## ## ## ## ## config wedge_data -- end

        self.hs['L[0][0][1][0]_wedge_data_config_box'].children = get_handles(self.hs, 'L[0][0][1][0]_wedge_data_config_box', -1)
        ## ## ## ## define widgets wedge_data_config_box -- end



        ## ## ## ## define filter_config_box -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.7*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][1][1]_filter_config_box'] = widgets.VBox()
        self.hs['L[0][0][1][1]_filter_config_box'].layout = layout

        ## ## ## ## ## label filter_config_box -- start
        layout = {'justify-content':'center', 'align-items':'center','align-contents':'center','border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][1][1][0]_filter_config_title_box'] = widgets.HBox()
        self.hs['L[0][0][1][1][0]_filter_config_title_box'].layout = layout
        self.hs['L[0][0][1][1][0][0]_filter_config_title'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Filter Config' + '</span>')
        layout = {'left':'43%', 'background-color':'white', 'color':'cyan'}
        self.hs['L[0][0][1][1][0][0]_filter_config_title'].layout = layout
        self.hs['L[0][0][1][1][0]_filter_config_title_box'].children = get_handles(self.hs, 'L[0][0][1][1][0]_filter_config_title_box', -1)
        ## ## ## ## ## label filter_config_box -- end

        ## ## ## ## ## config filters -- start
        self.hs['L[0][0][1][1][1]_filter_config_box'] = GridspecLayout(2, 200,
                                                                       layout = {'border':'3px solid #FFCC00',
                                                                                 'height':f'{0.63*(self.form_sz[0]-128)}px',
                                                                                 'align_items':'flex-start',
                                                                                 'justify_items':'flex-start'})
        ## ## ## ## ## ## config filters: left-hand side box -- start
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100] = GridspecLayout(10, 20,
                                                                                grid_gap='8px',
                                                                                layout = {'border':'3px solid #FFCC00',
                                                                                          'height':f'{0.52*(self.form_sz[0]-128)}px',
                                                                                          'grid_row_gap':'8px',
                                                                                          'align_items':'flex-start',
                                                                                          'justify_items':'flex-start',
                                                                                          'grid_column_gap':'8px'})
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][0, :16] = widgets.Dropdown(value='phase retrieval',
                                                                                          options=['phase retrieval',
                                                                                                   'stripe_removal: vo',
                                                                                                   'stripe_removal: fw'],
                                                                                          description='Filter List',
                                                                                          indent=False,
                                                                                          disabled=False)
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][0, 16:19] = widgets.Button(description='==>',
                                                                                          disabled=False,
                                                                                          layout = {'width':f'{int(1.5*(self.form_sz[1]-98)/20)}px'})
        for ii in range(3):
            for jj in range(2):
                self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][1+ii, jj*8:(jj+1)*8] = widgets.Dropdown(value='',
                                                                                                               options=[''],
                                                                                                               description='p'+str(ii*2+jj).zfill(2),
                                                                                                               disabled=False,
                                                                                                               layout = {'align_items':'flex-start',
                                                                                                                         'width':f'{int(7.5*(self.form_sz[1]-98)/40)}px'})
        for ii in range(3):
            for jj in range(2):
                self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][4+ii, jj*8:(jj+1)*8] = widgets.FloatText(value=0,
                                                                                                                description='p'+str((ii+3)*2+jj).zfill(2),
                                                                                                                disabled=False,
                                                                                                                layout = {'align_items':'flex-start',
                                                                                                                          'width':f'{int(7.5*(self.form_sz[1]-98)/40)}px'})
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][7:, :] = widgets.HTML(value= '<style>p{word-wrap: break-word}</style> <p>'+ 'Hover mouse over params for the description of the param for each filter.' +' </p>')
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][0, :16].observe(self.L0_0_1_1_1_filter_config_box_left_box_alg_dropdown_change, names='value')
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][0, 16:19].observe(self.L0_0_1_1_1_filter_config_box_left_box_move_button_change, names='value')
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][1, 0:8].observe(self.L0_0_1_1_1_filter_config_box_left_box_p00_change, names='value')
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][1, 8:16].observe(self.L0_0_1_1_1_filter_config_box_left_box_p01_change, names='value')
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][2, 0:8].observe(self.L0_0_1_1_1_filter_config_box_left_box_p02_change, names='value')
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][2, 8:16].observe(self.L0_0_1_1_1_filter_config_box_left_box_p03_change, names='value')
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][3, 0:8].observe(self.L0_0_1_1_1_filter_config_box_left_box_p04_change, names='value')
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][3, 8:16].observe(self.L0_0_1_1_1_filter_config_box_left_box_p05_change, names='value')
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][4, 0:8].observe(self.L0_0_1_1_1_filter_config_box_left_box_p06_change, names='value')
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][4, 8:16].observe(self.L0_0_1_1_1_filter_config_box_left_box_p07_change, names='value')
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][5, 0:8].observe(self.L0_0_1_1_1_filter_config_box_left_box_p08_change, names='value')
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][5, 8:16].observe(self.L0_0_1_1_1_filter_config_box_left_box_p09_change, names='value')
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][6, 0:8].observe(self.L0_0_1_1_1_filter_config_box_left_box_p10_change, names='value')
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, :100][6, 8:16].observe(self.L0_0_1_1_1_filter_config_box_left_box_p11_change, names='value')
        ## ## ## ## ## ## config filters: left-hand side box -- end

        ## ## ## ## ## ## config filters: right-hand side box -- start
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, 100:] = GridspecLayout(10, 10,
                                                                                grid_gap='8px',
                                                                                layout = {'border':'3px solid #FFCC00',
                                                                                          'height':f'{0.52*(self.form_sz[0]-128)}px'})
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, 100:][:, :9] = widgets.SelectMultiple(value=[''],
                                                                                               options=[''],
                                                                                               description='Filter Seq',
                                                                                               disabled=False,
                                                                                               layout={'height':f'{0.48*(self.form_sz[0]-128)}px'})
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, 100:][1, 9] = widgets.Button(description='Move Up',
                                                                                      disabled=False,
                                                                                      layout={'width':f'{int(2*(self.form_sz[1]-98)/20)}px'})
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, 100:][2, 9] = widgets.Button(description='Move Dn',
                                                                                      disabled=False,
                                                                                      layout={'width':f'{int(2*(self.form_sz[1]-98)/20)}px'})
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, 100:][3, 9] = widgets.Button(description='Remove',
                                                                                      disabled=False,
                                                                                      layout={'width':f'{int(2*(self.form_sz[1]-98)/20)}px'})
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, 100:][4, 9] = widgets.Button(description='Finish',
                                                                                      disabled=False,
                                                                                      layout={'width':f'{int(2*(self.form_sz[1]-98)/20)}px'})
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, 100:][:, :9].observe(self.L0_0_1_1_1_filter_config_box_right_box_selectmultiple_change, names='value')
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, 100:][1, 9].on_click(self.L0_0_1_1_1_filter_config_box_right_box_move_up_button_change)
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, 100:][2, 9].on_click(self.L0_0_1_1_1_filter_config_box_right_box_move_dn_button_change)
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, 100:][3, 9].on_click(self.L0_0_1_1_1_filter_config_box_right_box_remove_button_change)
        self.hs['L[0][0][1][1][1]_filter_config_box'][0, 100:][4, 9].on_click(self.L0_0_1_1_1_filter_config_box_right_box_finish_button_change)
        ## ## ## ## ## ## config filters: right-hand side box -- end

        ## ## ## ## ## ## config filters: confirm box -- start
        self.hs['L[0][0][1][1][1]_filter_config_box'][1, :140] = widgets.IntProgress(value=0,
                                                                                     min=0,
                                                                                     max=10,
                                                                                     step=1,
                                                                                     description='Completing:',
                                                                                     bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                                                                     orientation='horizontal',
                                                                                     layout={'width':f'{int(0.696*(self.form_sz[1]-98))}px', 'height':'90%'})
        self.hs['L[0][0][1][1][1]_filter_config_box'][1, 141:171] = widgets.Button(description='Finish',
                                                                                   disabled=False,
                                                                                   layout={'width':f'{int(0.15*(self.form_sz[1]-98))}px', 'height':'90%'})
        self.hs['L[0][0][1][1][1]_filter_config_box'][1, 141:171].style.button_color = 'darkviolet'
        ## ## ## ## ## ## config filters: confirm box -- start

        ## ## ## ## ## config filters -- start
        self.hs['L[0][0][1][1]_filter_config_box'].children = get_handles(self.hs, 'L[0][0][1][1]_filter_config_box', -1)
        ## ## ## ## define filter_config_box -- end



        ## ## ## ## define recon_box -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.14*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][1][2]_recon_box'] = widgets.VBox()
        self.hs['L[0][0][1][2]_recon_box'].layout = layout

        ## ## ## ## ## label recon_box -- start
        layout = {'justify-content':'center', 'align-items':'center','align-contents':'center','border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][1][2][0]_reecon_title_box'] = widgets.HBox()
        self.hs['L[0][0][1][2][0]_reecon_title_box'].layout = layout
        self.hs['L[0][0][1][2][0][0]_reecon_title'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'recon' + '</span>')
        layout = {'left':'46.5%', 'background-color':'white', 'color':'cyan'}
        self.hs['L[0][0][1][2][0][0]_reecon_title'].layout = layout
        self.hs['L[0][0][1][2][0]_reecon_title_box'].children = get_handles(self.hs, 'L[0][0][1][2][0]_reecon_title_box', -1)
        ## ## ## ## ## label recon_box -- end

        ## ## ## ## ## define recon_box -- start
        layout = {'justify-content':'center', 'align-items':'center','align-contents':'center','border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-128)}px'}
        self.hs['L[0][0][1][2][1]_reecon_box'] = widgets.HBox()
        self.hs['L[0][0][1][2][1]_reecon_box'].layout = layout
        layout = {'width':'70%', 'height':'90%'}
        self.hs['L[0][0][1][2][1][0]_reecon_progress_bar'] = widgets.IntProgress(value=0,
                                                                                 min=0,
                                                                                 max=10,
                                                                                 step=1,
                                                                                 description='Completing:',
                                                                                 bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                                                                 orientation='horizontal')
        self.hs['L[0][0][1][2][1][0]_reecon_progress_bar'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][0][1][2][1][1]_reecon_button'] = widgets.Button(description='Recon',
                                                                            disabled=True)
        self.hs['L[0][0][1][2][1][1]_reecon_button'].style.button_color = 'darkviolet'
        self.hs['L[0][0][1][2][1][1]_reecon_button'].layout = layout
        self.hs['L[0][0][1][2][1]_reecon_box'].children = get_handles(self.hs, 'L[0][0][1][2][1]_reecon_box', -1)
        self.hs['L[0][0][1][2][1][1]_reecon_button'].on_click(self.L0_0_1_2_1_1_reecon_button_change)
        ## ## ## ## ## define recon_box -- start

        self.hs['L[0][0][1][2]_recon_box'].children = get_handles(self.hs, 'L[0][0][1][2]_recon_box', -1)
        ## ## ## ## define recon box -- end

        self.hs['L[0][0][1]_data&filter_form'].children = get_handles(self.hs, 'L[0][0][1]_data&filter_form', -1)
        ## ## ## define boxes in data&filter_form -- end

    def restart(self):
        pass

    def boxes_logic(self):
        def tomo_compound_logic():
            if self.tomo_recon_type == 'Trial Center':
                self.hs['L[0][0][0][0][4][2]_save_debug_checkbox'].disabled = False
            elif self.tomo_recon_type in ['Vol Recon: Single', 'Vol Recon: Multi']:
                self.tomo_use_debug = False
                self.hs['L[0][0][0][0][4][2]_save_debug_checkbox'].value = False
                self.hs['L[0][0][0][0][4][2]_save_debug_checkbox'].disabled = True

        if self.tomo_filepath_configured:
            boxes = []

    def L0_0_0_0_1_0_select_raw_h5_top_dir_button_click(self, a):
        self.restart()
        if len(a.files[0]) != 0:
            self.tomo_raw_data_top_dir = a.files[0]
            # print(self.tomo_raw_data_top_dir)
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
                # print(self.tomo_recon_top_dir)
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
        elif self.tomo_recon_type == 'Vol Recon: Single':
            self.hs['L[0][0][0][0][3][0]_select_save_data_center_dir_button'].disabled = True
            self.hs['L[0][0][0][0][2][0]_select_save_recon_dir_button'].disabled = False
            self.hs['L[0][0][0][0][3][0]_select_save_data_center_dir_button'].style.button_color = "orange"
            self.hs['L[0][0][0][0][2][0]_select_save_recon_dir_button'].style.button_color = "orange"
        elif self.tomo_recon_type == 'Vol Recon: Multi':
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

    def L0_0_0_1_1_0_0_scan_id_text_change(self, a):
        pass

    def L0_0_0_1_1_0_1_rot_cen_text_change(self, a):
        pass

    def L0_0_0_1_1_0_2_cen_win_left_text_change(self, a):
        pass

    def L0_0_0_1_1_0_3_cen_win_wz_text_change(self, a):
        pass

    def L0_0_0_1_1_1_0_use_alt_flat_checkbox_change(self, a):
        pass

    def L0_0_0_1_1_1_1_alt_flat_file_button_change(self, a):
        pass

    def L0_0_0_1_1_1_2_use_alt_dark_checkbox_change(self, a):
        pass

    def L0_0_0_1_1_1_3_alt_dark_file_button_change(self, a):
        pass

    def L0_0_0_1_1_2_0_use_fake_flat_checkbox_change(self, a):
        pass

    def L0_0_0_1_1_2_1_fake_flat_val_text_change(self, a):
        pass

    def L0_0_0_1_1_2_2_use_fake_dark_checkbox_change(self, a):
        pass

    def L0_0_0_1_1_2_3_fake_dark_val_text_change(self, a):
        pass

    def L0_0_0_1_1_3_0_sli_start_text_change(self, a):
        pass

    def L0_0_0_1_1_3_1_sli_end_text_change(self, a):
        pass

    def L0_0_0_1_1_3_2_chunk_sz_text_change(self, a):
        pass

    def L0_0_0_1_1_3_3_margin_sz_text_change(self, a):
        pass

    def L0_0_0_1_1_4_0_rm_zinger_checkbox_change(self, a):
        pass

    def L0_0_0_1_1_4_1_zinger_level_text_change(self, a):
        pass

    def L0_0_0_1_1_4_2_use_mask_checkbox_change(self, a):
        pass

    def L0_0_0_1_1_4_3_mask_ratio_text_change(self, a):
        pass

    def L0_0_0_1_1_5_0_alg_options_dropdown_change(self, a):
        pass

    def L0_0_0_1_1_5_1_alg_param1_dropdown_change(self, a):
        pass

    def L0_0_0_1_1_5_2_alg_param2_dropdown_change(self, a):
        pass

    def L0_0_0_1_1_5_3_alg_param3_text_change(self, a):
        pass

    def L0_0_0_1_1_6_0_alg_param4_text_change(self, a):
        pass

    def L0_0_0_1_1_6_1_alg_param5_text_change(self, a):
        pass

    def L0_0_0_1_1_6_2_alg_param6_text_change(self, a):
        pass

    def L0_0_0_1_1_6_3_alg_param7_text_change(self, a):
        pass

    def L0_0_1_0_1_0_0_is_wedge_checkbox_change(self, a):
        pass

    def L0_0_1_0_1_0_1_blankat_dropdown_change(self, a):
        pass

    def L0_0_1_0_1_0_2_missing_start_text_change(self, a):
        pass

    def L0_0_1_0_1_0_3_missing_end_text_change(self, a):
        pass

    def L0_0_1_0_1_1_0_auto_detect_checkbox_change(self, a):
        pass

    def L0_0_1_0_1_1_1_auto_thres_text_change(self, a):
        pass

    def L0_0_1_0_1_1_2_auto_ref_fn_button_change(self, a):
        pass

    def L0_0_1_0_1_1_3_fiji_viewer_checkbox_change(self, a):
        pass

    def L0_0_1_0_1_1_4_sli_range_slider_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_left_box_alg_dropdown_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_left_box_move_button_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_left_box_p00_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_left_box_p01_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_left_box_p02_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_left_box_p03_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_left_box_p04_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_left_box_p05_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_left_box_p06_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_left_box_p07_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_left_box_p08_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_left_box_p09_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_left_box_p10_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_left_box_p11_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_right_box_selectmultiple_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_right_box_move_up_button_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_right_box_move_dn_button_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_right_box_remove_button_change(self, a):
        pass

    def L0_0_1_1_1_filter_config_box_right_box_finish_button_change(self, a):
        pass

    def L0_0_1_1_2_0_filter_config_confirm_button(self, a):
        pass

    def L0_0_1_2_1_1_reecon_button_change(self, a):
        pass











