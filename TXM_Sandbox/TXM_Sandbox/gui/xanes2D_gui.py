#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:05:59 2020

@author: xiao
"""
import os, h5py, json, time, gc, numpy as np

from ipywidgets import widgets
import skimage.morphology as skm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import fourier_shift
import napari

from ..utils import xanes_regtools as xr
from ..utils.io import data_reader, xanes2D_h5_reader
from . import xanes_fitting_gui as xfg
from . import xanes_analysis_gui as xag
from .gui_components import (SelectFilesButton, NumpyArrayEncoder, 
                            get_handles, enable_disable_boxes, 
                            fiji_viewer_state, restart,
                            fiji_viewer_on, fiji_viewer_off,
                            determine_element, determine_fitting_energy_range,
                            gen_external_py_script)

napari.gui_qt()

class xanes2D_tools_gui():
    def __init__(self, parent_h, form_sz=[650, 740]):
        self.gui_name = 'xanes2D'
        self.form_sz = form_sz
        self.global_h = parent_h              
        self.hs = {}
        
        if self.global_h.use_struc_h5_reader:
            self.reader = data_reader(xanes2D_h5_reader)
        else:
            pass
        
        self.xanes2D_raw_fn_temp = self.global_h.io_xanes2D_cfg['xanes2D_raw_fn_template']

        self.xanes2D_fit_external_command_name = os.path.join(os.path.abspath(os.path.curdir), 'xanes2D_fit_external_command.py')
        self.xanes2D_reg_external_command_name = os.path.join(os.path.abspath(os.path.curdir), 'xanes2D_reg_external_command.py')
        self.xanes2D_align_external_command_name = os.path.join(os.path.abspath(os.path.curdir), 'xanes2D_align_external_command.py')

        self.xanes2D_file_configured = False
        self.xanes2D_data_configured = False
        self.xanes2D_roi_configured = False
        self.xanes2D_reg_params_configured = False
        self.xanes2D_reg_done = False
        self.xanes2D_reg_review_done = False
        self.xanes2D_alignment_done = False
        self.xanes_analysis_eng_configured = False
        self.xanes2D_review_read_alignment_option = False

        self.xanes2D_file_raw_h5_set = False
        self.xanes2D_file_save_trial_set = False
        self.xanes2D_file_reg_file_set = False
        self.xanes2D_file_config_file_set = False
        self.xanes2D_config_alternative_flat_set = False
        self.xanes2D_config_raw_img_readed = False
        self.xanes2D_regparams_anchor_idx_set = False
        self.xanes2D_file_reg_file_readed = False
        self.xanes_analysis_eng_set = False

        self.xanes2D_config_is_raw = False
        self.xanes2D_config_is_refine = False
        self.xanes2D_config_img_scalar = 1
        self.xanes2D_config_use_smooth_flat = False
        self.xanes2D_config_smooth_flat_sigma = 0
        self.xanes2D_config_use_alternative_flat = False
        self.xanes2D_config_eng_list = None

        self.xanes2D_file_analysis_option = 'Do New Reg'
        self.xanes2D_file_raw_h5_filename = None
        self.xanes2D_save_trial_reg_filename = None
        self.xanes2D_file_save_trial_reg_config_filename = None
        self.xanes2D_config_alternative_flat_filename = None
        self.xanes2D_review_reg_best_match_filename = None

        self.xanes2D_reg_use_chunk = True
        self.xanes2D_reg_anchor_idx = 0
        self.xanes2D_reg_roi = [0, 10, 0, 10]
        self.xanes2D_reg_use_mask = False
        self.xanes2D_reg_mask = None
        self.xanes2D_reg_mask_dilation_width = 0
        self.xanes2D_reg_mask_thres = 0
        self.xanes2D_reg_use_smooth_img = False
        self.xanes2D_reg_smooth_img_sigma = 5
        self.xanes2D_reg_chunk_sz = None
        self.xanes2D_reg_method = None
        self.xanes2D_reg_ref_mode = None
        self.xanes2D_reg_mrtv_level = 4
        self.xanes2D_reg_mrtv_width = 10
        self.xanes2D_reg_mrtv_subpixel_step = 0.2

        self.xanes2D_visualization_auto_bc = False

        self.xanes2D_img = None
        self.xanes2D_img_roi = None
        self.xanes2D_review_aligned_img_original = None
        self.xanes2D_use_existing_reg_reviewed = False
        self.xanes2D_reg_file_readed = False
        self.xanes2D_reg_review_file = None
        self.xanes2D_review_aligned_img = None
        self.xanes2D_review_fixed_img = None
        self.xanes2D_review_bad_shift = False
        self.xanes2D_review_shift_dict = None
        self.xanes2D_manual_xshift = 0
        self.xanes2D_manual_yshift = 0
        self.xanes2D_review_shift_dict = {}

        self.xanes2D_file_analysis_option = 'Do New Reg'
        self.xanes2D_eng_id_s = 0
        self.xanes2D_eng_id_e = 1

        self.xanes_element = None
        self.xanes_analysis_eng_list = None
        self.xanes_analysis_type = 'wl'
        self.xanes_analysis_edge_eng = 0
        self.xanes_analysis_wl_fit_eng_s = 0
        self.xanes_analysis_wl_fit_eng_e = 0
        self.xanes_analysis_pre_edge_e = -50
        self.xanes_analysis_post_edge_s = 100
        self.xanes_analysis_edge_0p5_fit_s = 0
        self.xanes_analysis_edge_0p5_fit_e = 0
        self.xanes_analysis_spectrum = None
        self.xanes_analysis_use_mask = False
        self.xanes_analysis_mask_thres = None
        self.xanes_analysis_mask_img_id = None
        self.xanes_analysis_mask = 1
        self.xanes_analysis_edge_jump_thres = 1.0
        self.xanes_analysis_edge_offset_thres = 1.0

        self.xanes2D_config = {"filepath config":{"xanes2D_file_raw_h5_filename":self.xanes2D_file_raw_h5_filename,
                                                  "xanes_save_trial_reg_filename":self.xanes2D_save_trial_reg_filename,
                                                  "xanes2D_file_save_trial_reg_config_filename": self.xanes2D_file_save_trial_reg_config_filename,
                                                  "xanes2D_review_reg_best_match_filename":self.xanes2D_review_reg_best_match_filename,
                                                  "xanes2D_file_analysis_option":self.xanes2D_file_analysis_option,
                                                  "xanes2D_file_configured":self.xanes2D_file_configured,
                                                  "xanes2D_file_raw_h5_set":self.xanes2D_file_raw_h5_set,
                                                  "xanes2D_file_save_trial_set":self.xanes2D_file_save_trial_set,
                                                  "xanes2D_file_reg_file_set":self.xanes2D_file_reg_file_set,
                                                  "xanes2D_file_config_file_set":self.xanes2D_file_config_file_set,
                                                  "xanes2D_file_reg_file_readed":self.xanes2D_file_reg_file_readed},
                               "data_config":{"xanes2D_config_is_raw":self.xanes2D_config_is_raw,
                                              "xanes2D_config_is_refine":self.xanes2D_config_is_refine,
                                              "xanes2D_config_img_scalar":self.xanes2D_config_img_scalar,
                                              "xanes2D_config_use_alternative_flat":self.xanes2D_config_use_alternative_flat,
                                              "xanes2D_config_use_smooth_flat":self.xanes2D_config_use_smooth_flat,
                                              "xanes2D_config_smooth_flat_sigma":self.xanes2D_config_smooth_flat_sigma,
                                              "xanes2D_config_alternative_flat_filename":self.xanes2D_config_alternative_flat_filename,
                                              "xanes2D_config_alternative_flat_set":self.xanes2D_config_alternative_flat_set,
                                              "xanes2D_config_eng_points_range_s":self.xanes2D_eng_id_s,
                                              "xanes2D_config_eng_points_range_e":self.xanes2D_eng_id_e,
                                              "xanes2D_config_eng_s":0,
                                              "xanes2D_config_eng_e":0,
                                              "xanes2D_config_fiji_view_on":False,
                                              "xanes2D_config_img_num":0,
                                              "xanes2D_config_raw_img_readed":self.xanes2D_config_raw_img_readed,
                                              "xanes2D_config_norm_scale_text_min":0,
                                              "xanes2D_config_norm_scale_text_val":1,
                                              "xanes2D_config_norm_scale_text_max":10,
                                              "xanes2D_config_smooth_flat_sigma_text.min":0,
                                              "xanes2D_config_smooth_flat_sigma_text.val":0,
                                              "xanes2D_config_smooth_flat_sigma_text.max":30,
                                              "xanes2D_config_eng_points_range_slider_min":0,
                                              "xanes2D_config_eng_points_range_slider_val":0,
                                              "xanes2D_config_eng_points_range_slider_max":0,
                                              "xanes2D_config_eng_s_text_min":0,
                                              "xanes2D_config_eng_s_text_val":0,
                                              "xanes2D_config_eng_s_text_max":0,
                                              "xanes2D_config_eng_e_text_min":0,
                                              "xanes2D_config_eng_e_text_val":0,
                                              "xanes2D_config_eng_e_text_max":0,
                                              "xanes2D_config_fiji_eng_id_slider_min":0,
                                              "xanes2D_config_fiji_eng_id_slider_val":0,
                                              "xanes2D_config_fiji_eng_id_slider_max":0,
                                              "xanes2D_data_configured":self.xanes2D_data_configured},
                               "roi_config":{"2D_roi_x_slider_min":0,
                                             "2D_roi_x_slider_val":0,
                                             "2D_roi_x_slider_max":0,
                                             "2D_roi_y_slider_min":0,
                                             "2D_roi_y_slider_val":0,
                                             "2D_roi_y_slider_max":0,
                                             "2D_roi":list(self.xanes2D_reg_roi),
                                             "xanes2D_roi_configured":self.xanes2D_roi_configured},
                               "registration_config":{"xanes2D_regparams_fiji_viewer":True,
                                                      "xanes2D_regparams_use_chunk":True,
                                                      "xanes2D_regparams_anchor_id":0,
                                                      "xanes2D_regparams_use_mask":True,
                                                      "xanes2D_regparams_mask_thres":0,
                                                      "xanes2D_regparams_mask_dilation":0,
                                                      "xanes2D_regparams_use_smooth_img":False,
                                                      "xanes2D_regparams_smooth_sigma":5,
                                                      "xanes2D_regparams_chunk_sz":7,
                                                      "xanes2D_regparams_reg_method":"MPC",
                                                      "xanes2D_regparams_ref_mode":"single",
                                                      "xanes2D_regparams_anchor_id_slider_min":0,
                                                      "xanes2D_regparams_anchor_id_slider_val":0,
                                                      "xanes2D_regparams_anchor_id_slider_max":0,
                                                      "xanes2D_regparams_mask_thres_slider_min":0,
                                                      "xanes2D_regparams_mask_thres_slider_val":0,
                                                      "xanes2D_regparams_mask_thres_slider_max":0,
                                                      "xanes2D_regparams_mask_dilation_slider.min":0,
                                                      "xanes2D_regparams_mask_dilation_slider.val":0,
                                                      "xanes2D_regparams_mask_dilation_slider.max":0,
                                                      "xanes2D_regparams_smooth_sigma_text.min":0,
                                                      "xanes2D_regparams_smooth_sigma_text.val":0,
                                                      "xanes2D_regparams_smooth_sigma_text.max":0,
                                                      "xanes2D_regparams_chunk_sz_slider.min":0,
                                                      "xanes2D_regparams_chunk_sz_slider.val":0,
                                                      "xanes2D_regparams_chunk_sz_slider.max":0,
                                                      "xanes2D_regparams_reg_method_dropdown_options":("MPC", "PC", "SR"),
                                                      "xanes2D_regparams_ref_mode_dropdown_options":("single", "neighbour"),
                                                      "xanes2D_regparams_mrtv_level":4,
                                                      "xanes2D_regparams_mrtv_width":10,
                                                      "xanes2D_regparams_mrtv_subpixel_step":0.2,
                                                      "xanes2D_regparams_mrtv_subpixel_width":8,
                                                      "xanes2D_regparams_configured":self.xanes2D_reg_params_configured},
                               "run registration":{"xanes2D_reg_done":self.xanes2D_reg_done},
                               "review registration":{"xanes2D_review_use_existing_reg_reviewed":self.xanes2D_review_read_alignment_option,
                                                      "xanes2D_reviewed_reg_shift":self.xanes2D_review_shift_dict,
                                                      "reg_pair_slider_min":0,
                                                      "reg_pair_slider_val":0,
                                                      "reg_pair_slider_max":0,
                                                      "xshift_text_val":0,
                                                      "yshift_text_val":0,
                                                      "xanes2D_reg_review_done":self.xanes2D_reg_review_done},
                               "align 2D recon":{"xanes2D_alignment_done":self.xanes2D_alignment_done,
                                                 "xanes2D_analysis_edge_eng":self.xanes_analysis_edge_eng, 
                                                 "xanes2D_analysis_wl_fit_eng_s":self.xanes_analysis_wl_fit_eng_s,
                                                 "xanes2D_analysis_wl_fit_eng_e":self.xanes_analysis_wl_fit_eng_e,
                                                 "xanes2D_analysis_pre_edge_e":self.xanes_analysis_pre_edge_e,
                                                 "xanes2D_analysis_post_edge_s":self.xanes_analysis_post_edge_s, 
                                                 "xanes2D_analysis_edge_0p5_fit_s":self.xanes_analysis_edge_0p5_fit_s,
                                                 "xanes2D_analysis_edge_0p5_fit_e":self.xanes_analysis_edge_0p5_fit_e,
                                                 "xanes2D_analysis_type":self.xanes_analysis_type}
                               }
        self.xanes_fitting_gui_h = xfg.xanes_fitting_gui(self, form_sz=self.form_sz)
        self.xanes_fitting_gui_h.build_gui()  
        self.xanes_analysis_gui_h = xag.xanes_analysis_gui(self, form_sz=self.form_sz)
        self.xanes_analysis_gui_h.build_gui()
        
    def lock_message_text_boxes(self):
        boxes = ['L[0][1][0][0][1][1]_select_raw_h5_path_text',
                 'L[0][1][0][0][2][1]_select_save_trial_text',
                 'L[0][1][0][0][3][1]_confirm_file&path_text',
                 'L[0][1][0][1][2][1]_eng_s_text',
                 'L[0][1][0][1][2][2]_eng_e_text'
                 'L[0][1][0][1][4][1]_confirm_config_data_text',
                 'L[0][1][1][0][2][0]_confirm_roi_text',
                 'L[0][1][1][1][5][0]_confirm_reg_params_text',
                 'L[0][1][1][2][1][1]_run_reg_text',
                 'L[0][1][2][0][4][0]_confirm_review_results_text',
                 'L[0][1][2][1][1][0]_align_text',
                 'L[0][1][2][2][1][3]_visualization_eng_text']
        enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][5][0]_fit_run_text'].disabled=True

    def update_xanes2D_config(self):
        tem = {}
        for key, item in self.xanes2D_review_shift_dict.items():
            tem[key] = list(item)
        self.xanes2D_config = {"filepath config":{"xanes2D_file_raw_h5_filename":self.xanes2D_file_raw_h5_filename,
                                                  "xanes_save_trial_reg_filename":self.xanes2D_save_trial_reg_filename,
                                                  "xanes2D_file_save_trial_reg_config_filename": self.xanes2D_file_save_trial_reg_config_filename,
                                                  "xanes2D_review_reg_best_match_filename":self.xanes2D_review_reg_best_match_filename,
                                                  "xanes2D_file_analysis_option":self.xanes2D_file_analysis_option,
                                                  "xanes2D_file_configured":self.xanes2D_file_configured,
                                                  "xanes2D_file_raw_h5_set":self.xanes2D_file_raw_h5_set,
                                                  "xanes2D_file_save_trial_set":self.xanes2D_file_save_trial_set,
                                                  "xanes2D_file_reg_file_set":self.xanes2D_file_reg_file_set,
                                                  "xanes2D_file_config_file_set":self.xanes2D_file_config_file_set,
                                                  "xanes2D_file_reg_file_readed":self.xanes2D_file_reg_file_readed},
                               "data_config":{"xanes2D_config_is_raw":self.xanes2D_config_is_raw,
                                              "xanes2D_config_is_refine":self.xanes2D_config_is_refine,
                                              "xanes2D_config_img_scalar":self.xanes2D_config_img_scalar,
                                              "xanes2D_config_use_alternative_flat":self.xanes2D_config_use_alternative_flat,
                                              "xanes2D_config_use_smooth_flat":self.xanes2D_config_use_smooth_flat,
                                              "xanes2D_config_smooth_flat_sigma":self.xanes2D_config_smooth_flat_sigma,
                                              "xanes2D_config_alternative_flat_filename":self.xanes2D_config_alternative_flat_filename,
                                              "xanes2D_config_alternative_flat_set":self.xanes2D_config_alternative_flat_set,
                                              "xanes2D_config_eng_points_range_s":self.xanes2D_eng_id_s,
                                              "xanes2D_config_eng_points_range_e":self.xanes2D_eng_id_e,
                                              "xanes2D_config_eng_s":self.hs['L[0][1][0][1][2][1]_eng_s_text'].value,
                                              "xanes2D_config_eng_e":self.hs['L[0][1][0][1][2][2]_eng_e_text'].value,
                                              "xanes2D_config_fiji_view_on":self.hs['L[0][1][0][1][3][0]_fiji_raw_img_preview_checkbox'].value,
                                              "xanes2D_config_img_num":self.hs['L[0][1][0][1][3][2]_fiji_eng_id_slider'].value,
                                              "xanes2D_config_norm_scale_text_min":self.hs['L[0][1][0][1][1][0][2]_norm_scale_text'].min,
                                              "xanes2D_config_norm_scale_text_val":self.hs['L[0][1][0][1][1][0][2]_norm_scale_text'].value,
                                              "xanes2D_config_norm_scale_text_max":self.hs['L[0][1][0][1][1][0][2]_norm_scale_text'].max,
                                              "xanes2D_config_smooth_flat_sigma_text.min":self.hs['L[0][1][0][1][1][1][2]_smooth_flat_sigma_text'].min,
                                              "xanes2D_config_smooth_flat_sigma_text.val":self.hs['L[0][1][0][1][1][1][2]_smooth_flat_sigma_text'].value,
                                              "xanes2D_config_smooth_flat_sigma_text.max":self.hs['L[0][1][0][1][1][1][2]_smooth_flat_sigma_text'].max,
                                              "xanes2D_config_eng_points_range_slider_min":self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].min,
                                              "xanes2D_config_eng_points_range_slider_val":self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].value,
                                              "xanes2D_config_eng_points_range_slider_max":self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].max,
                                              "xanes2D_config_eng_s_text_min":self.hs['L[0][1][0][1][2][1]_eng_s_text'].min,
                                              "xanes2D_config_eng_s_text_val":self.hs['L[0][1][0][1][2][1]_eng_s_text'].value,
                                              "xanes2D_config_eng_s_text_max":self.hs['L[0][1][0][1][2][1]_eng_s_text'].max,
                                              "xanes2D_config_eng_e_text_min":self.hs['L[0][1][0][1][2][2]_eng_e_text'].min,
                                              "xanes2D_config_eng_e_text_val":self.hs['L[0][1][0][1][2][2]_eng_e_text'].value,
                                              "xanes2D_config_eng_e_text_max":self.hs['L[0][1][0][1][2][2]_eng_e_text'].max,
                                              "xanes2D_config_fiji_eng_id_slider_min":self.hs['L[0][1][0][1][3][2]_fiji_eng_id_slider'].min,
                                              "xanes2D_config_fiji_eng_id_slider_val":self.hs['L[0][1][0][1][3][2]_fiji_eng_id_slider'].value,
                                              "xanes2D_config_fiji_eng_id_slider_max":self.hs['L[0][1][0][1][3][2]_fiji_eng_id_slider'].max,
                                              "xanes2D_config_raw_img_readed":self.xanes2D_config_raw_img_readed,
                                              "xanes2D_data_configured":self.xanes2D_data_configured},
                               "roi_config":{"2D_roi_x_slider_min":self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].min,
                                             "2D_roi_x_slider_val":self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].value,
                                             "2D_roi_x_slider_max":self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].max,
                                             "2D_roi_y_slider_min":self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].min,
                                             "2D_roi_y_slider_val":self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].value,
                                             "2D_roi_y_slider_max":self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].max,
                                             "2D_roi":list(self.xanes2D_reg_roi),
                                             "xanes2D_roi_configured":self.xanes2D_roi_configured},
                               "registration_config":{"xanes2D_regparams_fiji_viewer":self.hs['L[0][1][1][1][1][0]_fiji_mask_viewer_checkbox'].value,
                                                      "xanes2D_regparams_use_chunk":self.hs['L[0][1][1][1][1][1]_use_chunk_checkbox'].value,
                                                      "xanes2D_regparams_anchor_id":self.hs['L[0][1][1][1][1][2]_anchor_id_slider'].value,
                                                      "xanes2D_regparams_use_mask":self.hs['L[0][1][1][1][2][0]_use_mask_checkbox'].value,
                                                      "xanes2D_regparams_mask_thres":self.hs['L[0][1][1][1][2][1]_mask_thres_slider'].value,
                                                      "xanes2D_regparams_mask_dilation":self.hs['L[0][1][1][1][2][2]_mask_dilation_slider'].value,
                                                      "xanes2D_regparams_use_smooth_img":self.hs['L[0][1][1][1][3][0]_use_smooth_checkbox'].value,
                                                      "xanes2D_regparams_smooth_sigma":self.hs['L[0][1][1][1][3][1]_smooth_sigma_text'].value,
                                                      "xanes2D_regparams_chunk_sz":self.hs['L[0][1][1][1][1][3]_chunk_sz_slider'].value,
                                                      "xanes2D_regparams_reg_method":self.hs['L[0][1][1][1][3][2]_reg_method_dropdown'].value,
                                                      "xanes2D_regparams_ref_mode":self.hs['L[0][1][1][1][3][3]_ref_mode_dropdown'].value,
                                                      "xanes2D_regparams_anchor_id_slider_min":self.hs['L[0][1][1][1][1][2]_anchor_id_slider'].min,
                                                      "xanes2D_regparams_anchor_id_slider_val":self.hs['L[0][1][1][1][1][2]_anchor_id_slider'].value,
                                                      "xanes2D_regparams_anchor_id_slider_max":self.hs['L[0][1][1][1][1][2]_anchor_id_slider'].max,
                                                      "xanes2D_regparams_mask_thres_slider_min":self.hs['L[0][1][1][1][2][1]_mask_thres_slider'].min,
                                                      "xanes2D_regparams_mask_thres_slider_val":self.hs['L[0][1][1][1][2][1]_mask_thres_slider'].value,
                                                      "xanes2D_regparams_mask_thres_slider_max":self.hs['L[0][1][1][1][2][1]_mask_thres_slider'].max,
                                                      "xanes2D_regparams_mask_dilation_slider.min":self.hs['L[0][1][1][1][2][2]_mask_dilation_slider'].min,
                                                      "xanes2D_regparams_mask_dilation_slider.val":self.hs['L[0][1][1][1][2][2]_mask_dilation_slider'].value,
                                                      "xanes2D_regparams_mask_dilation_slider.max":self.hs['L[0][1][1][1][2][2]_mask_dilation_slider'].max,
                                                      "xanes2D_regparams_smooth_sigma_text.min":self.hs['L[0][1][1][1][3][1]_smooth_sigma_text'].min,
                                                      "xanes2D_regparams_smooth_sigma_text.val":self.hs['L[0][1][1][1][3][1]_smooth_sigma_text'].value,
                                                      "xanes2D_regparams_smooth_sigma_text.max":self.hs['L[0][1][1][1][3][1]_smooth_sigma_text'].max,
                                                      "xanes2D_regparams_chunk_sz_slider.min":self.hs['L[0][1][1][1][1][3]_chunk_sz_slider'].min,
                                                      "xanes2D_regparams_chunk_sz_slider.val":self.hs['L[0][1][1][1][1][3]_chunk_sz_slider'].value,
                                                      "xanes2D_regparams_chunk_sz_slider.max":self.hs['L[0][1][1][1][1][3]_chunk_sz_slider'].max,
                                                      # "xanes2D_regparams_reg_method_dropdown_options":self.hs['L[0][1][1][1][3][2]_reg_method_dropdown'].options,
                                                      "xanes2D_regparams_ref_mode_dropdown_options":self.hs['L[0][1][1][1][3][3]_ref_mode_dropdown'].options,                                                      
                                                      "xanes2D_regparams_mrtv_level":self.hs['L[0][1][1][1][4][0]_mrtv_level_text'].value,
                                                      "xanes2D_regparams_mrtv_width":self.hs['L[0][1][1][1][4][1]_mrtv_width_text'].value,
                                                      "xanes2D_regparams_mrtv_subpixel_step":self.hs['L[0][1][1][1][4][3]_mrtv_subpixel_step_text'].value,
                                                      "xanes2D_regparams_mrtv_subpixel_width":self.hs['L[0][1][1][1][4][2]_mrtv_subpixel_wz_text'].value,
                                                      "xanes2D_regparams_configured":self.xanes2D_reg_params_configured},
                               "run registration":{"xanes2D_reg_done":self.xanes2D_reg_done},
                               "review registration":{"xanes2D_review_use_existing_reg_reviewed":self.hs['L[0][1][2][0][1][1]_read_alignment_checkbox'].value,
                                                      "xanes2D_reviewed_reg_shift":tem,
                                                      "reg_pair_slider_min":self.hs['L[0][1][2][0][2][0]_reg_pair_slider'].min,
                                                      "reg_pair_slider_val":self.hs['L[0][1][2][0][2][0]_reg_pair_slider'].value,
                                                      "reg_pair_slider_max":self.hs['L[0][1][2][0][2][0]_reg_pair_slider'].max,
                                                      "xshift_text_val":self.hs['L[0][1][2][0][3][0]_x_shift_text'].value,
                                                      "yshift_text_val":self.hs['L[0][1][2][0][3][1]_y_shift_text'].value,
                                                      "xanes2D_reg_review_done":self.xanes2D_reg_review_done},
                               "align 2D recon":{"xanes2D_alignment_done":self.xanes2D_alignment_done,
                                                 "xanes2D_analysis_edge_eng":self.xanes_analysis_edge_eng, 
                                                 "xanes2D_analysis_wl_fit_eng_s":self.xanes_analysis_wl_fit_eng_s,
                                                 "xanes2D_analysis_wl_fit_eng_e":self.xanes_analysis_wl_fit_eng_e,
                                                 "xanes2D_analysis_pre_edge_e":self.xanes_analysis_pre_edge_e,
                                                 "xanes2D_analysis_post_edge_s":self.xanes_analysis_post_edge_s, 
                                                 "xanes2D_analysis_edge_0p5_fit_s":self.xanes_analysis_edge_0p5_fit_s,
                                                 "xanes2D_analysis_edge_0p5_fit_e":self.xanes_analysis_edge_0p5_fit_e,
                                                 "xanes2D_analysis_type":self.xanes_analysis_type}
                               }

    def read_xanes2D_config(self):
        with open(self.xanes2D_file_save_trial_reg_config_filename_original, 'r') as f:
            self.xanes2D_config = json.load(f)

    def set_xanes2D_variables(self):
        self.xanes2D_file_raw_h5_filename = self.xanes2D_config["filepath config"]["xanes2D_file_raw_h5_filename"]
        self.xanes2D_save_trial_reg_filename = self.xanes2D_config["filepath config"]["xanes_save_trial_reg_filename"]
        self.xanes2D_review_reg_best_match_filename = self.xanes2D_config["filepath config"]["xanes2D_review_reg_best_match_filename"]
        self.xanes2D_file_raw_h5_set = self.xanes2D_config["filepath config"]["xanes2D_file_raw_h5_set"]
        self.xanes2D_file_save_trial_set = self.xanes2D_config["filepath config"]["xanes2D_file_save_trial_set"]
        self.xanes2D_file_reg_file_set = self.xanes2D_config["filepath config"]["xanes2D_file_reg_file_set"]
        self.xanes2D_file_config_file_set = self.xanes2D_config["filepath config"]["xanes2D_file_config_file_set"]
        self.xanes2D_file_configured = self.xanes2D_config["filepath config"]["xanes2D_file_configured"]

        self.xanes2D_config_is_raw = self.xanes2D_config["data_config"]["xanes2D_config_is_raw"]
        self.xanes2D_config_is_refine = self.xanes2D_config["data_config"]["xanes2D_config_is_refine"]
        self.xanes2D_config_img_scalar = self.xanes2D_config["data_config"]["xanes2D_config_img_scalar"]
        self.xanes2D_config_use_alternative_flat = self.xanes2D_config["data_config"]["xanes2D_config_use_alternative_flat"]
        self.xanes2D_config_use_smooth_flat = self.xanes2D_config["data_config"]["xanes2D_config_use_smooth_flat"]
        self.xanes2D_config_smooth_flat_sigma = self.xanes2D_config["data_config"]["xanes2D_config_smooth_flat_sigma"]
        self.xanes2D_config_alternative_flat_filename = self.xanes2D_config["data_config"]["xanes2D_config_alternative_flat_filename"]
        self.xanes2D_config_alternative_flat_set = self.xanes2D_config["data_config"]["xanes2D_config_alternative_flat_set"]
        self.xanes2D_eng_id_s = self.xanes2D_config["data_config"]["xanes2D_config_eng_points_range_s"]
        self.xanes2D_eng_id_e = self.xanes2D_config["data_config"]["xanes2D_config_eng_points_range_e"]
        self.xanes2D_config_raw_img_readed = self.xanes2D_config["data_config"]["xanes2D_config_raw_img_readed"]
        self.xanes2D_data_configured = self.xanes2D_config["data_config"]["xanes2D_data_configured"]

        self.xanes2D_reg_roi = self.xanes2D_config["roi_config"]["2D_roi"]
        self.xanes2D_roi_configured = self.xanes2D_config["roi_config"]["xanes2D_roi_configured"]
        self.xanes2D_reg_use_chunk = self.xanes2D_config["registration_config"]["xanes2D_regparams_use_chunk"]
        self.xanes2D_reg_anchor_idx = self.xanes2D_config["registration_config"]["xanes2D_regparams_anchor_id"]
        self.xanes2D_reg_use_mask = self.xanes2D_config["registration_config"]["xanes2D_regparams_use_mask"]
        self.xanes2D_reg_mask_thres = self.xanes2D_config["registration_config"]["xanes2D_regparams_mask_thres"]
        self.xanes2D_reg_mask_dilation_width = self.xanes2D_config["registration_config"]["xanes2D_regparams_mask_dilation"]
        self.xanes2D_reg_use_smooth_img = self.xanes2D_config["registration_config"]["xanes2D_regparams_use_smooth_img"]
        self.xanes2D_reg_smooth_img_sigma = self.xanes2D_config["registration_config"]["xanes2D_regparams_smooth_sigma"]
        self.xanes2D_reg_chunk_sz = self.xanes2D_config["registration_config"]["xanes2D_regparams_chunk_sz"]
        self.xanes2D_reg_method = self.xanes2D_config["registration_config"]["xanes2D_regparams_reg_method"]
        self.xanes2D_reg_ref_mode = self.xanes2D_config["registration_config"]["xanes2D_regparams_ref_mode"]
        
        self.xanes2D_reg_mrtv_level = self.xanes2D_config["registration_config"]["xanes2D_regparams_mrtv_level"]
        self.xanes2D_reg_mrtv_width = self.xanes2D_config["registration_config"]["xanes2D_regparams_mrtv_width"]
        self.xanes2D_reg_mrtv_subpixel_step = self.xanes2D_config["registration_config"]["xanes2D_regparams_mrtv_subpixel_step"]
        self.xanes2D_reg_mrtv_subpixel_wz = self.xanes2D_config["registration_config"]["xanes2D_regparams_mrtv_subpixel_width"]
        self.xanes2D_reg_params_configured = self.xanes2D_config["registration_config"]["xanes2D_regparams_configured"]

        self.xanes2D_reg_done = self.xanes2D_config["run registration"]["xanes2D_reg_done"]

        self.xanes2D_review_read_alignment_option = self.xanes2D_config["review registration"]["xanes2D_review_use_existing_reg_reviewed"]
        tem = self.xanes2D_config["review registration"]["xanes2D_reviewed_reg_shift"]
        self.xanes2D_review_shift_dict = {}
        for key, item in tem.items():
            self.xanes2D_review_shift_dict[key] = np.array(item)
        self.xanes2D_reg_review_done = self.xanes2D_config["review registration"]["xanes2D_reg_review_done"]

        self.xanes2D_alignment_done = self.xanes2D_config["align 2D recon"]["xanes2D_alignment_done"]
        self.xanes_analysis_edge_eng = self.xanes2D_config["align 2D recon"]["xanes2D_analysis_edge_eng"] 
        self.xanes_analysis_wl_fit_eng_s = self.xanes2D_config["align 2D recon"]["xanes2D_analysis_wl_fit_eng_s"]
        self.xanes_analysis_wl_fit_eng_e = self.xanes2D_config["align 2D recon"]["xanes2D_analysis_wl_fit_eng_e"]
        self.xanes_analysis_pre_edge_e = self.xanes2D_config["align 2D recon"]["xanes2D_analysis_pre_edge_e"]
        self.xanes_analysis_post_edge_s = self.xanes2D_config["align 2D recon"]["xanes2D_analysis_post_edge_s"]
        self.xanes_analysis_edge_0p5_fit_s = self.xanes2D_config["align 2D recon"]["xanes2D_analysis_edge_0p5_fit_s"]
        self.xanes_analysis_edge_0p5_fit_e = self.xanes2D_config["align 2D recon"]["xanes2D_analysis_edge_0p5_fit_e"]
        self.xanes_analysis_type = self.xanes2D_config["align 2D recon"]["xanes2D_analysis_type"]


    def set_xanes2D_handles(self):
        self.hs['L[0][1][0][1][1][0][0]_is_raw_checkbox'].value = self.xanes2D_config_is_raw
        self.hs['L[0][1][0][1][1][0][1]_is_refine_checkbox'].value = self.xanes2D_config_is_refine
        self.hs['L[0][1][0][1][1][0][3]_use_alternative_flat_checkbox'].value = self.xanes2D_config_use_alternative_flat
        self.hs['L[0][1][0][1][1][1][1]_smooth_flat_checkbox'].value = self.xanes2D_config_use_smooth_flat
        
        self.hs['L[0][1][0][1][1][0][2]_norm_scale_text'].value = self.xanes2D_config_img_scalar
        self.hs['L[0][1][0][1][1][1][2]_smooth_flat_sigma_text'].value = self.xanes2D_config_smooth_flat_sigma
        self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].max = self.xanes2D_config["data_config"]["xanes2D_config_eng_points_range_slider_max"]
        self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].min = self.xanes2D_config["data_config"]["xanes2D_config_eng_points_range_slider_min"]
        self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].value = self.xanes2D_config["data_config"]["xanes2D_config_eng_points_range_slider_val"]
        self.hs['L[0][1][0][1][2][1]_eng_s_text'].max = self.xanes2D_config["data_config"]["xanes2D_config_eng_s_text_max"]
        self.hs['L[0][1][0][1][2][1]_eng_s_text'].min = self.xanes2D_config["data_config"]["xanes2D_config_eng_s_text_min"]
        self.hs['L[0][1][0][1][2][1]_eng_s_text'].value = self.xanes2D_config["data_config"]["xanes2D_config_eng_s_text_val"]
        self.hs['L[0][1][0][1][2][2]_eng_e_text'].max = self.xanes2D_config["data_config"]["xanes2D_config_eng_e_text_max"]
        self.hs['L[0][1][0][1][2][2]_eng_e_text'].min = self.xanes2D_config["data_config"]["xanes2D_config_eng_e_text_min"]
        self.hs['L[0][1][0][1][2][2]_eng_e_text'].value = self.xanes2D_config["data_config"]["xanes2D_config_eng_e_text_val"]
        self.hs['L[0][1][0][1][3][2]_fiji_eng_id_slider'].max = self.xanes2D_config["data_config"]["xanes2D_config_fiji_eng_id_slider_max"]
        self.hs['L[0][1][0][1][3][2]_fiji_eng_id_slider'].min = self.xanes2D_config["data_config"]["xanes2D_config_fiji_eng_id_slider_min"]
        self.hs['L[0][1][0][1][3][2]_fiji_eng_id_slider'].value = self.xanes2D_config["data_config"]["xanes2D_config_fiji_eng_id_slider_val"]

        self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].max = self.xanes2D_config["roi_config"]["2D_roi_x_slider_max"]
        self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].min = self.xanes2D_config["roi_config"]["2D_roi_x_slider_min"]
        self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].value = self.xanes2D_config["roi_config"]["2D_roi_x_slider_val"]
        self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].max = self.xanes2D_config["roi_config"]["2D_roi_y_slider_max"]
        self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].min = self.xanes2D_config["roi_config"]["2D_roi_y_slider_min"]
        self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].value = self.xanes2D_config["roi_config"]["2D_roi_y_slider_val"]

        self.hs['L[0][1][1][1][1][2]_anchor_id_slider'].max = self.xanes2D_config["registration_config"]["xanes2D_regparams_anchor_id_slider_max"]
        self.hs['L[0][1][1][1][1][2]_anchor_id_slider'].min = self.xanes2D_config["registration_config"]["xanes2D_regparams_anchor_id_slider_min"]
        self.hs['L[0][1][1][1][1][2]_anchor_id_slider'].value = self.xanes2D_config["registration_config"]["xanes2D_regparams_anchor_id_slider_val"]
        self.hs['L[0][1][1][1][2][0]_use_mask_checkbox'].value = self.xanes2D_config["registration_config"]["xanes2D_regparams_use_mask"]
        self.hs['L[0][1][1][1][2][1]_mask_thres_slider'].max = self.xanes2D_config["registration_config"]["xanes2D_regparams_mask_thres_slider_max"]
        self.hs['L[0][1][1][1][2][1]_mask_thres_slider'].min = self.xanes2D_config["registration_config"]["xanes2D_regparams_mask_thres_slider_min"]
        self.hs['L[0][1][1][1][2][1]_mask_thres_slider'].value = self.xanes2D_config["registration_config"]["xanes2D_regparams_mask_thres_slider_val"]
        self.hs['L[0][1][1][1][2][2]_mask_dilation_slider'].max = self.xanes2D_config["registration_config"]["xanes2D_regparams_mask_dilation_slider.max"]
        self.hs['L[0][1][1][1][2][2]_mask_dilation_slider'].min = self.xanes2D_config["registration_config"]["xanes2D_regparams_mask_dilation_slider.min"]
        self.hs['L[0][1][1][1][2][2]_mask_dilation_slider'].value = self.xanes2D_config["registration_config"]["xanes2D_regparams_mask_dilation_slider.val"]
        self.hs['L[0][1][1][1][3][0]_use_smooth_checkbox'].value = self.xanes2D_config["registration_config"]["xanes2D_regparams_use_smooth_img"]
        self.hs['L[0][1][1][1][3][1]_smooth_sigma_text'].max = self.xanes2D_config["registration_config"]["xanes2D_regparams_smooth_sigma_text.max"]
        self.hs['L[0][1][1][1][3][1]_smooth_sigma_text'].min = self.xanes2D_config["registration_config"]["xanes2D_regparams_smooth_sigma_text.min"]
        self.hs['L[0][1][1][1][3][1]_smooth_sigma_text'].value = self.xanes2D_config["registration_config"]["xanes2D_regparams_smooth_sigma_text.val"]
        self.hs['L[0][1][1][1][1][3]_chunk_sz_slider'].max = self.xanes2D_config["registration_config"]["xanes2D_regparams_chunk_sz_slider.max"]
        self.hs['L[0][1][1][1][1][3]_chunk_sz_slider'].min = self.xanes2D_config["registration_config"]["xanes2D_regparams_chunk_sz_slider.min"]
        self.hs['L[0][1][1][1][1][3]_chunk_sz_slider'].value = self.xanes2D_config["registration_config"]["xanes2D_regparams_chunk_sz_slider.val"]
        # self.hs['L[0][1][1][1][3][2]_reg_method_dropdown'].options = self.xanes2D_config["registration_config"]["xanes2D_regparams_reg_method_dropdown_options"]
        self.hs['L[0][1][1][1][3][2]_reg_method_dropdown'].value = self.xanes2D_config["registration_config"]["xanes2D_regparams_reg_method"]
        self.hs['L[0][1][1][1][3][3]_ref_mode_dropdown'].options = self.xanes2D_config["registration_config"]["xanes2D_regparams_ref_mode_dropdown_options"]
        self.hs['L[0][1][1][1][3][3]_ref_mode_dropdown'].value = self.xanes2D_config["registration_config"]["xanes2D_regparams_ref_mode"]
        self.hs['L[0][1][1][1][4][0]_mrtv_level_text'].value = self.xanes2D_config["registration_config"]["xanes2D_regparams_mrtv_level"]
        self.hs['L[0][1][1][1][4][1]_mrtv_width_text'].value = self.xanes2D_config["registration_config"]["xanes2D_regparams_mrtv_width"]
        self.hs['L[0][1][1][1][4][2]_mrtv_subpixel_wz_text'].value = self.xanes2D_config["registration_config"]["xanes2D_regparams_mrtv_subpixel_width"]
        self.hs['L[0][1][1][1][4][3]_mrtv_subpixel_step_text'].value = self.xanes2D_config["registration_config"]["xanes2D_regparams_mrtv_subpixel_step"]

        self.hs['L[0][1][2][0][2][0]_reg_pair_slider'].max = self.xanes2D_config["review registration"]["reg_pair_slider_max"]
        self.hs['L[0][1][2][0][2][0]_reg_pair_slider'].min = self.xanes2D_config["review registration"]["reg_pair_slider_min"]
        self.hs['L[0][1][2][0][2][0]_reg_pair_slider'].value = self.xanes2D_config["review registration"]["reg_pair_slider_val"]
        self.hs['L[0][1][2][0][3][0]_x_shift_text'].value = self.xanes2D_config["review registration"]["xshift_text_val"]
        self.hs['L[0][1][2][0][3][1]_y_shift_text'].value = self.xanes2D_config["review registration"]["yshift_text_val"]
        
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][0][0]_fit_eng_range_option_dropdown'].value = \
            self.xanes2D_config["align 2D recon"]["xanes2D_analysis_type"]
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][0][1]_fit_eng_range_edge_eng_text'].value = \
            self.xanes2D_config["align 2D recon"]["xanes2D_analysis_edge_eng"] 
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][0][2]_fit_eng_range_pre_edge_e_text'].value = \
            self.xanes2D_config["align 2D recon"]["xanes2D_analysis_pre_edge_e"]
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][0][3]_fit_eng_range_post_edge_s_text'].value = \
            self.xanes2D_config["align 2D recon"]["xanes2D_analysis_post_edge_s"]
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][1][0]_fit_eng_range_wl_fit_s_text'].value = \
            self.xanes2D_config["align 2D recon"]["xanes2D_analysis_wl_fit_eng_s"]
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][1][1]_fit_eng_range_wl_fit_e_text'].value = \
            self.xanes2D_config["align 2D recon"]["xanes2D_analysis_wl_fit_eng_e"]
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][1][2]_fit_eng_range_edge0.5_fit_s_text'].value = \
            self.xanes2D_config["align 2D recon"]["xanes2D_analysis_edge_0p5_fit_s"]
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][1][3]_fit_eng_range_edge0.5_fit_e_text'].value = \
            self.xanes2D_config["align 2D recon"]["xanes2D_analysis_edge_0p5_fit_e"]
        
    def boxes_logic(self):
        def xanes2D_compound_logic():
            if self.xanes2D_file_analysis_option == 'Reg By Shift':
                if self.xanes2D_roi_configured:
                    boxes = ['L[0][1][2][0]_review_reg_results_box']
                    enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                    self.hs['L[0][1][2][0][1][1]_read_alignment_checkbox'].disabled = True
                    self.hs['L[0][1][2][0][1][0]_read_alignment_button'].disabled = False
                else:
                    boxes = ['L[0][1][2][0]_review_reg_results_box']
                    enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                if self.xanes2D_reg_file_readed:
                    self.hs['L[0][1][2][0][4][1]_confirm_review_results_button'].disabled = False
            else:
                if (self.xanes2D_reg_done and (not self.xanes2D_reg_file_readed) and
                    self.hs['L[0][1][2][0][1][1]_read_alignment_checkbox'].value):
                    boxes = ['L[0][1][2][0][2]_reg_pair_box',
                             'L[0][1][2][0][3]_correct_shift_box']
                    enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                    self.hs['L[0][1][2][0][1][0]_read_alignment_button'].disabled = False
                elif (self.xanes2D_reg_done and 
                      (not self.hs['L[0][1][2][0][1][1]_read_alignment_checkbox'].value)):
                    boxes = ['L[0][1][2][0][2]_reg_pair_box']
                    enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                    boxes = ['L[0][1][2][0][3]_correct_shift_box']
                    enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                    self.hs['L[0][1][2][0][1][0]_read_alignment_button'].disabled = True                    

                if not self.xanes2D_review_bad_shift:
                    boxes = ['L[0][1][2][0][3]_correct_shift_box']
                    enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                else:
                    boxes = ['L[0][1][2][0][3]_correct_shift_box']
                    enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                    
            if self.xanes2D_config_raw_img_readed:
                boxes = ['L[0][1][0][1][2]_eng_range_box',
                         'L[0][1][0][1][3]_fiji_box',
                         'L[0][1][0][1][4]_config_data_confirm_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            else:
                boxes = ['L[0][1][0][1][2]_eng_range_box',
                         'L[0][1][0][1][3]_fiji_box',
                         'L[0][1][0][1][4]_config_data_confirm_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                
            if self.hs['L[0][1][0][1][1][0][3]_use_alternative_flat_checkbox'].value:
                boxes = ['L[0][1][0][1][1][0][4]_alternative_flat_file_button']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            else:
                boxes = ['L[0][1][0][1][1][0][4]_alternative_flat_file_button']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                
            if self.hs['L[0][1][0][1][1][1][1]_smooth_flat_checkbox'].value:
                boxes = ['L[0][1][0][1][1][1][2]_smooth_flat_sigma_text']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            else:
                boxes = ['L[0][1][0][1][1][1][2]_smooth_flat_sigma_text']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                
            if (self.hs['L[0][1][0][1][1][0][0]_is_raw_checkbox'].value & 
                self.xanes2D_file_configured):
                self.hs['L[0][1][0][1][1][0][1]_is_refine_checkbox'].disabled = True
                self.hs['L[0][1][0][1][1][0][3]_use_alternative_flat_checkbox'].disabled = False
                self.hs['L[0][1][0][1][1][1][1]_smooth_flat_checkbox'].disabled = False
                self.hs['L[0][1][0][1][1][0][2]_norm_scale_text'].disabled = False
            else:
                self.hs['L[0][1][0][1][1][0][1]_is_refine_checkbox'].disabled = False
                self.hs['L[0][1][0][1][1][0][3]_use_alternative_flat_checkbox'].disabled = True
                self.hs['L[0][1][0][1][1][1][1]_smooth_flat_checkbox'].disabled = True
                self.hs['L[0][1][0][1][1][0][2]_norm_scale_text'].disabled = True
                
            if self.hs['L[0][1][1][1][1][1]_use_chunk_checkbox'].value:
                self.hs['L[0][1][1][1][1][2]_anchor_id_slider'].disabled = False
                self.hs['L[0][1][1][1][1][3]_chunk_sz_slider'].disabled = False
                self.xanes2D_regparams_anchor_idx_set = False
            else:
                self.hs['L[0][1][1][1][1][2]_anchor_id_slider'].disabled = True
                self.hs['L[0][1][1][1][1][3]_chunk_sz_slider'].disabled = True
                self.xanes2D_regparams_anchor_idx_set = False
            
            if self.hs['L[0][1][1][1][3][2]_reg_method_dropdown'].value in ['MPC', 'MPC+MRTV']:
                self.hs['L[0][1][1][1][2][0]_use_mask_checkbox'].value = 1
                self.hs['L[0][1][1][1][2][1]_mask_thres_slider'].disabled = False
                self.hs['L[0][1][1][1][2][2]_mask_dilation_slider'].disabled = False                
            elif self.hs['L[0][1][1][1][3][2]_reg_method_dropdown'].value in ['MRTV', 'PC', 'LS+MRTV']:
                self.hs['L[0][1][1][1][2][0]_use_mask_checkbox'].value = 0
                self.hs['L[0][1][1][1][2][1]_mask_thres_slider'].disabled = True
                self.hs['L[0][1][1][1][2][2]_mask_dilation_slider'].disabled = True
            elif self.hs['L[0][1][1][1][3][2]_reg_method_dropdown'].value == 'SR':
                self.hs['L[0][1][1][1][2][1]_mask_thres_slider'].disabled = False
                self.hs['L[0][1][1][1][2][2]_mask_dilation_slider'].disabled = False
                
                
            if self.hs['L[0][1][1][1][3][0]_use_smooth_checkbox'].value:
                self.hs['L[0][1][1][1][3][1]_smooth_sigma_text'].disabled = False
            else:
                self.hs['L[0][1][1][1][3][1]_smooth_sigma_text'].disabled = True
    
            if self.hs['L[0][1][1][1][3][2]_reg_method_dropdown'].value in ['MPC', 'SR', 'PC']:
                boxes = ['L[0][1][1][1][4]_mrtv_options_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            elif self.hs['L[0][1][1][1][3][2]_reg_method_dropdown'].value == 'MPC+MRTV':
                self.hs['L[0][1][1][1][4][0]_mrtv_level_text'].disabled = True
                self.hs['L[0][1][1][1][4][1]_mrtv_width_text'].disabled = True
                self.hs['L[0][1][1][1][4][2]_mrtv_subpixel_wz_text'].disabled = False
                self.hs['L[0][1][1][1][4][3]_mrtv_subpixel_step_text'].disabled = False
            elif self.hs['L[0][1][1][1][3][2]_reg_method_dropdown'].value == 'LS+MRTV':
                self.hs['L[0][1][1][1][4][0]_mrtv_level_text'].disabled = True
                self.hs['L[0][1][1][1][4][1]_mrtv_width_text'].disabled = False
                self.hs['L[0][1][1][1][4][1]_mrtv_width_text'].max = 300
                self.hs['L[0][1][1][1][4][1]_mrtv_width_text'].min = 100
                self.hs['L[0][1][1][1][4][2]_mrtv_subpixel_wz_text'].disabled = False
                self.hs['L[0][1][1][1][4][3]_mrtv_subpixel_step_text'].disabled = False
            elif self.hs['L[0][1][1][1][3][2]_reg_method_dropdown'].value == 'MRTV':                
                self.hs['L[0][1][1][1][4][1]_mrtv_width_text'].min = 1
                self.hs['L[0][1][1][1][4][1]_mrtv_width_text'].max = 20
                if self.hs['L[0][1][1][1][4][1]_mrtv_width_text'].value >= 20:
                    self.hs['L[0][1][1][1][4][1]_mrtv_width_text'].value = 10
                boxes = ['L[0][1][1][1][4]_mrtv_options_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                
            if self.hs['L[0][1][1][1][3][2]_reg_method_dropdown'].value in ['PC', 'MRTV', 'LS+MRTV']:
                boxes = ['L[0][1][1][1][2]_mask_options_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            elif self.hs['L[0][1][1][1][3][2]_reg_method_dropdown'].value in ['MPC', 'MPC+MRTV', 'SR']:
                boxes = ['L[0][1][1][1][2]_mask_options_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)

        if self.xanes2D_file_analysis_option in ['Do New Reg', 'Read Config File']:
            if not self.xanes2D_file_configured:
                boxes = ['L[0][1][0][1]_config_data_box',
                         'L[0][1][1][0]_2D_roi_box',
                         'L[0][1][1][1]_config_reg_params_box',
                         'L[0][1][1][2]_run_reg_box',
                         'L[0][1][2][0]_review_reg_results_box',
                         'L[0][1][2][1]_align_images_box',
                         'L[0][1][2][2]_visualize_images_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0]_analysis_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
            elif self.xanes2D_file_configured & (not self.xanes2D_data_configured):
                boxes = ['L[0][1][0][1]_config_data_box',
                         'L[0][1][1][0]_2D_roi_box',
                         'L[0][1][1][1]_config_reg_params_box',
                         'L[0][1][1][2]_run_reg_box',
                         'L[0][1][2][0]_review_reg_results_box',
                         'L[0][1][2][1]_align_images_box',
                         'L[0][1][2][2]_visualize_images_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0]_analysis_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
                boxes = ['L[0][1][0][1][1][0]_data_preprocessing_options_box0',
                         'L[0][1][0][1][1][1]_data_preprocessing_options_box1',
                         'L[0][1][0][1][1][2]_data_preprocessing_options_box2']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            elif ((self.xanes2D_file_configured & self.xanes2D_data_configured) &
                  (not self.xanes2D_roi_configured)):
                boxes = ['L[0][1][1][1]_config_reg_params_box',
                         'L[0][1][1][2]_run_reg_box',
                         'L[0][1][2][0]_review_reg_results_box',
                         'L[0][1][2][1]_align_images_box',
                         'L[0][1][2][2]_visualize_images_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0]_analysis_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
                boxes = ['L[0][1][0][1]_config_data_box',
                         'L[0][1][1][0]_2D_roi_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            elif ((self.xanes2D_file_configured & self.xanes2D_data_configured) &
                  self.xanes2D_roi_configured & (not self.xanes2D_reg_params_configured)):
                boxes = ['L[0][1][1][2]_run_reg_box',
                         'L[0][1][2][0]_review_reg_results_box',
                         'L[0][1][2][1]_align_images_box',
                         'L[0][1][2][2]_visualize_images_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0]_analysis_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
                boxes = ['L[0][1][0][1]_config_data_box',
                         'L[0][1][1][0]_2D_roi_box',
                         'L[0][1][1][1]_config_reg_params_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            elif ((self.xanes2D_file_configured & self.xanes2D_data_configured) &
                  (self.xanes2D_roi_configured & self.xanes2D_reg_params_configured) &
                  (not self.xanes2D_reg_done)):
                boxes = ['L[0][1][2][0]_review_reg_results_box',
                         'L[0][1][2][1]_align_images_box',
                         'L[0][1][2][2]_visualize_images_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0]_analysis_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
                boxes = ['L[0][1][0][1]_config_data_box',
                         'L[0][1][1][0]_2D_roi_box',
                         'L[0][1][1][1]_config_reg_params_box',
                         'L[0][1][1][2]_run_reg_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            elif ((self.xanes2D_file_configured & self.xanes2D_data_configured) &
                  (self.xanes2D_roi_configured & self.xanes2D_reg_params_configured) &
                  (self.xanes2D_reg_done & (not self.xanes2D_reg_review_done))):
                boxes = ['L[0][1][2][1]_align_images_box',
                         'L[0][1][2][2]_visualize_images_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0]_analysis_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
                boxes = ['L[0][1][0][1]_config_data_box',
                         'L[0][1][1][0]_2D_roi_box',
                         'L[0][1][1][1]_config_reg_params_box',
                         'L[0][1][1][2]_run_reg_box',
                         'L[0][1][2][0]_review_reg_results_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            elif ((self.xanes2D_file_configured & self.xanes2D_data_configured) &
                  (self.xanes2D_roi_configured & self.xanes2D_reg_params_configured) &
                  (self.xanes2D_reg_done & self.xanes2D_reg_review_done) &
                  (not self.xanes2D_alignment_done)):
                boxes = ['L[0][1][2][2]_visualize_images_box',]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0]_analysis_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
                boxes = ['L[0][1][0][1]_config_data_box',
                         'L[0][1][1][0]_2D_roi_box',
                         'L[0][1][1][1]_config_reg_params_box',
                         'L[0][1][1][2]_run_reg_box',
                         'L[0][1][2][0]_review_reg_results_box',
                         'L[0][1][2][1]_align_images_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            elif ((self.xanes2D_file_configured & self.xanes2D_data_configured) &
                  (self.xanes2D_roi_configured & self.xanes2D_reg_params_configured) &
                  (self.xanes2D_reg_done & self.xanes2D_reg_review_done) &
                  (self.xanes2D_alignment_done) & (not self.xanes_analysis_eng_configured)):
                boxes = ['L[0][1][0][1]_config_data_box',
                         'L[0][1][1][0]_2D_roi_box',
                         'L[0][1][1][1]_config_reg_params_box',
                         'L[0][1][1][2]_run_reg_box',
                         'L[0][1][2][0]_review_reg_results_box',
                         'L[0][1][2][1]_align_images_box',
                         'L[0][1][2][2]_visualize_images_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ['L[0][x][3][0][1]_analysis_energy_range_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=False, level=-1)
                boxes = ['L[0][x][3][0][4]_analysis_tool_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
            elif ((self.xanes2D_file_configured & self.xanes2D_data_configured) &
                  (self.xanes2D_roi_configured & self.xanes2D_reg_params_configured) &
                  (self.xanes2D_reg_done & self.xanes2D_reg_review_done) &
                  (self.xanes2D_alignment_done & self.xanes_analysis_eng_configured)):
                boxes = ['L[0][1][0][1]_config_data_box',
                         'L[0][1][1][0]_2D_roi_box',
                         'L[0][1][1][1]_config_reg_params_box',
                         'L[0][1][1][2]_run_reg_box',
                         'L[0][1][2][0]_review_reg_results_box',
                         'L[0][1][2][1]_align_images_box',
                         'L[0][1][2][2]_visualize_images_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ['L[0][x][3][0][1]_analysis_energy_range_box',
                          'L[0][x][3][0][4]_analysis_tool_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                      boxes, disabled=False, level=-1)
            xanes2D_compound_logic()
        elif self.xanes2D_file_analysis_option == 'Reg By Shift':
            if not self.xanes2D_file_configured:
                boxes = ['L[0][1][1][0]_2D_roi_box',
                         'L[0][1][2][0]_review_reg_results_box',
                         'L[0][1][2][1]_align_images_box',
                         'L[0][1][2][2]_visualize_images_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0]_analysis_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
            elif (self.xanes2D_file_configured & (not self.xanes2D_roi_configured)):
                boxes = ['L[0][1][1][0]_2D_roi_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ['L[0][1][2][0]_review_reg_results_box',
                         'L[0][1][2][1]_align_images_box',
                         'L[0][1][2][2]_visualize_images_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0]_analysis_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
            elif ((self.xanes2D_file_configured & self.xanes2D_roi_configured) &
                  (not self.xanes2D_reg_review_done)):
                boxes = ['L[0][1][1][0]_2D_roi_box',
                         'L[0][1][2][0][1]_read_alignment_file_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                self.hs['L[0][1][2][0][1][1]_read_alignment_checkbox'].value = True
                self.hs['L[0][1][2][0][1][1]_read_alignment_checkbox'].disabled = True
                self.hs['L[0][1][2][0][1][0]_read_alignment_button'].disabled = False
                
                boxes = ['L[0][1][2][1]_align_images_box',
                         'L[0][1][2][2]_visualize_images_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0]_analysis_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
            elif ((self.xanes2D_file_configured & self.xanes2D_roi_configured) & 
                  (self.xanes2D_reg_review_done & (not self.xanes2D_alignment_done))):
                boxes = ['L[0][1][1][0]_2D_roi_box',
                         'L[0][1][2][0][1]_read_alignment_file_box',
                         'L[0][1][2][1]_align_images_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                self.hs['L[0][1][2][0][1][1]_read_alignment_checkbox'].value = True
                self.hs['L[0][1][2][0][1][1]_read_alignment_checkbox'].disabled = True
                self.hs['L[0][1][2][0][1][0]_read_alignment_button'].disabled = False
                
                boxes = ['L[0][1][2][2]_visualize_images_box',]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0]_analysis_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)                
            elif ((self.xanes2D_file_configured & self.xanes2D_roi_configured) &
                  (self.xanes2D_reg_review_done & self.xanes2D_alignment_done) & 
                  (not self.xanes_analysis_eng_configured)):
                boxes = ['L[0][1][1][0]_2D_roi_box',
                         'L[0][1][2][0][1]_read_alignment_file_box',
                         'L[0][1][2][1]_align_images_box',
                         'L[0][1][2][2]_visualize_images_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                self.hs['L[0][1][2][0][1][1]_read_alignment_checkbox'].value = True
                self.hs['L[0][1][2][0][1][1]_read_alignment_checkbox'].disabled = True
                self.hs['L[0][1][2][0][1][0]_read_alignment_button'].disabled = False
                boxes = ['L[0][x][3][0][1]_analysis_energy_range_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=False, level=-1)
                boxes = ['L[0][x][3][0][4]_analysis_tool_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
            elif ((self.xanes2D_file_configured & self.xanes2D_roi_configured) &
                  (self.xanes2D_reg_review_done & self.xanes2D_alignment_done) & 
                  self.xanes_analysis_eng_configured):
                boxes = ['L[0][1][1][0]_2D_roi_box',
                         'L[0][1][2][0][1]_read_alignment_file_box',
                         'L[0][1][2][1]_align_images_box',
                         'L[0][1][2][2]_visualize_images_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                self.hs['L[0][1][2][0][1][1]_read_alignment_checkbox'].value = True
                self.hs['L[0][1][2][0][1][1]_read_alignment_checkbox'].disabled = True
                self.hs['L[0][1][2][0][1][0]_read_alignment_button'].disabled = False
                boxes = ['L[0][x][3][0][1]_analysis_energy_range_box',
                          'L[0][x][3][0][4]_analysis_tool_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                      boxes, disabled=False, level=-1)
            xanes2D_compound_logic()
            boxes = ['L[0][1][0][1]_config_data_box',
                     'L[0][1][1][1]_config_reg_params_box',
                     'L[0][1][1][2]_run_reg_box']            
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        elif self.xanes2D_file_analysis_option == 'Do Analysis':
            boxes = ['L[0][1][0][1]_config_data_box',
                     'L[0][1][1][0]_2D_roi_box',
                     'L[0][1][1][1]_config_reg_params_box',
                     'L[0][1][1][2]_run_reg_box',
                     'L[0][1][2][0]_review_reg_results_box',
                     'L[0][1][2][1]_align_images_box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            boxes = ['L[0][1][2][2]_visualize_images_box']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            boxes = ['L[0][x][3][0][1]_analysis_energy_range_box',
                     'L[0][x][3][0][4]_analysis_tool_box']
            enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                 boxes, disabled=False, level=-1)
            xanes2D_compound_logic()
        self.lock_message_text_boxes()
    
    def build_gui(self):
        """
        hs: widget handler sets; hs is a multi-layer structured dictionary. Layers
        are labeled as '0', '1', '2' in order of their hiearchical relations. In
        each layer, the item '0' is always the parent widget that hosts all other
        layers with keys '1' to 'n'. The keys are the name of the widget itmes used
        to identify them.

        Returns
        -------
        None.

        """
        #################################################################################################################
        #                                                                                                               #
        #                                                     2D XANES                                                  #
        #                                                                                                               #
        #################################################################################################################
        ## ## ## define 2D_XANES_tabs layout -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-86}px', 'height':f'{self.form_sz[0]-136}px'}
        self.hs['L[0][1][0]_config_input_form'] = widgets.VBox()
        self.hs['L[0][1][1]_reg_setting_form'] = widgets.VBox()
        self.hs['L[0][1][2]_reg&review_form'] = widgets.VBox()
        self.hs['L[0][1][3]_fitting_form'] = widgets.VBox()
        self.hs['L[0][1][4]_analysis_form'] = widgets.VBox()
        self.hs['L[0][1][0]_config_input_form'].layout = layout
        self.hs['L[0][1][1]_reg_setting_form'].layout = layout
        self.hs['L[0][1][2]_reg&review_form'].layout = layout
        self.hs['L[0][1][3]_fitting_form'].layout = layout
        self.hs['L[0][1][4]_analysis_form'].layout = layout

        ## ## ## ## define functional widget tabs in each sub-tab - configure file settings -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.25*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][0][0]_select_file&path_box'] = widgets.VBox()
        self.hs['L[0][1][0][0]_select_file&path_box'].layout = layout

        ## ## ## ## ## label configure file settings box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][0][0][0]_select_file&path_title_box'] = widgets.HBox()
        self.hs['L[0][1][0][0][0]_select_file&path_title_box'].layout = layout
        # self.hs['L[0][1][0][0][0][0]_select_file&path_title_box'] = widgets.Text(value='Config Files', disabled=True)
        self.hs['L[0][1][0][0][0][0]_select_file&path_title_box'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Config Dirs & Files' + '</span>')
        layout = {'background-color':'white', 'color':'cyan', 'left':'39%'}
        self.hs['L[0][1][0][0][0][0]_select_file&path_title_box'].layout = layout
        self.hs['L[0][1][0][0][0]_select_file&path_title_box'].children = get_handles(self.hs, 'L[0][1][0][0][0]_select_file&path_title_box', -1)

        ## ## ## ## ## raw h5 top directory
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][0][0][1]_select_raw_box'] = widgets.HBox()
        self.hs['L[0][1][0][0][1]_select_raw_box'].layout = layout
        self.hs['L[0][1][0][0][1][1]_select_raw_h5_path_text'] = widgets.Text(value='Choose raw h5 directory ...', description='', disabled=True)
        layout = {'width':'66%', 'display':'inline_flex'}
        self.hs['L[0][1][0][0][1][1]_select_raw_h5_path_text'].layout = layout
        self.hs['L[0][1][0][0][1][0]_select_raw_h5_path_button'] = SelectFilesButton(option='askopenfilename',
                                                                                     text_h=self.hs['L[0][1][0][0][1][1]_select_raw_h5_path_text'],
                                                                                     **{'open_filetypes': (('h5 files', '*.h5'),)})
        layout = {'width':'15%'}
        self.hs['L[0][1][0][0][1][0]_select_raw_h5_path_button'].layout = layout
        self.hs['L[0][1][0][0][1][0]_select_raw_h5_path_button'].description = 'XANES2D File'
        self.hs['L[0][1][0][0][1][0]_select_raw_h5_path_button'].on_click(self.L0_1_0_0_1_0_select_raw_h5_path_button_click)
        self.hs['L[0][1][0][0][1]_select_raw_box'].children = get_handles(self.hs, 'L[0][1][0][0][1]_select_raw_box', -1)

        ## ## ## ## ## trial save file
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][0][0][2]_select_save_trial_box'] = widgets.HBox()
        self.hs['L[0][1][0][0][2]_select_save_trial_box'].layout = layout
        self.hs['L[0][1][0][0][2][1]_select_save_trial_text'] = widgets.Text(value='Save trial registration as ...', description='', disabled=True)
        layout = {'width':'66%', 'display':'inline_flex'}
        self.hs['L[0][1][0][0][2][1]_select_save_trial_text'].layout = layout
        self.hs['L[0][1][0][0][2][0]_select_save_trial_button'] = SelectFilesButton(option='asksaveasfilename',
                                                                                    text_h=self.hs['L[0][1][0][0][2][1]_select_save_trial_text'],
                                                                                    **{'open_filetypes': (('h5 files', '*.h5'),),
                                                                                       'initialfile':'2D_trial_reg.h5'})
        self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].description = 'Save Reg File'
        layout = {'width':'15%'}
        self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].layout = layout
        self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].on_click(self.L0_1_0_0_3_0_select_save_trial_button_click)
        self.hs['L[0][1][0][0][2]_select_save_trial_box'].children = get_handles(self.hs, 'L[0][1][0][0][2]_select_save_trial_box', -1)

        ## ## ## ## ## confirm file configuration
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][0][0][3]_select_file&path_title_comfirm_box'] = widgets.HBox()
        self.hs['L[0][1][0][0][3]_select_file&path_title_comfirm_box'].layout = layout
        self.hs['L[0][1][0][0][3][1]_confirm_file&path_text'] = widgets.Text(value='Save trial registration, or go directly review registration ...', description='', disabled=True)
        layout = {'width':'66%'}
        self.hs['L[0][1][0][0][3][1]_confirm_file&path_text'].layout = layout
        self.hs['L[0][1][0][0][3][0]_confirm_file&path_button'] = widgets.Button(description='Confirm',
                                                                                 tooltip='Confirm: Confirm after you finish file configuration')
        self.hs['L[0][1][0][0][3][0]_confirm_file&path_button'].style.button_color = 'darkviolet'
        self.hs['L[0][1][0][0][3][0]_confirm_file&path_button'].on_click(self.L0_1_0_0_4_0_confirm_file_path_button_click)
        layout = {'width':'15%'}
        self.hs['L[0][1][0][0][3][0]_confirm_file&path_button'].layout = layout

        self.hs['L[0][1][0][0][3][2]_file_path_options_dropdown'] = widgets.Dropdown(value='Do New Reg',
                                                                              options=['Do New Reg',
                                                                                       'Read Config File',
                                                                                       'Reg By Shift',
                                                                                       'Do Analysis'],
                                                                              description='',
                                                                              description_tooltip='"Do New Reg": start registration and review results from beginning; "Read Reg File": if you have already done registraion and like to review the results; "Read Config File": if you like to resume analysis with an existing configuration.',
                                                                              disabled=False)
        layout = {'width':'15%', 'top':'0%'}
        self.hs['L[0][1][0][0][3][2]_file_path_options_dropdown'].layout = layout

        self.hs['L[0][1][0][0][3][2]_file_path_options_dropdown'].observe(self.L0_1_0_0_3_2_file_path_options_dropdown_change, names='value')
        self.hs['L[0][1][0][0][3]_select_file&path_title_comfirm_box'].children = get_handles(self.hs, 'L[0][1][0][0][3]_select_file&path_title_comfirm_box', -1)

        self.hs['L[0][1][0][0]_select_file&path_box'].children = get_handles(self.hs, 'L[0][1][0][0]_select_file&path_box', -1)
        ## ## ## ## bin widgets in hs['L[0][1][0][0]_select_file&path_box'] -- configure file settings -- end

        ## ## ## ## define functional widgets in each box in each sub-tab  - define data -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.43*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][0][1]_config_data_box'] = widgets.VBox()
        self.hs['L[0][1][0][1]_config_data_box'].layout = layout
        ## ## ## ## ## label config_data box
        layout = {'justify-content':'center', 'align-items':'center','align-contents':'center','border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][0][1][0]_config_data_title_box'] = widgets.HBox()
        self.hs['L[0][1][0][1][0]_config_data_title_box'].layout = layout
        self.hs['L[0][1][0][1][0][0]_config_data_title'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Config Data & Preprocessing' + '</span>')
        layout = {'left':'35%', 'background-color':'white', 'color':'cyan'}
        self.hs['L[0][1][0][1][0][0]_config_data_title'].layout = layout
        self.hs['L[0][1][0][1][0]_config_data_title_box'].children = get_handles(self.hs, 'L[0][1][0][1][0]_config_data_title_box', -1)

        ## ## ## ## ## data_preprocessing_options -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.6*0.23*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][0][1][1]_data_preprocessing_options_box'] = widgets.VBox()
        self.hs['L[0][1][0][1][1]_data_preprocessing_options_box'].layout = layout
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-104}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][0][1][1][0]_data_preprocessing_options_box0'] = widgets.HBox()
        self.hs['L[0][1][0][1][1][0]_data_preprocessing_options_box0'].layout = layout
        self.hs['L[0][1][0][1][1][0][0]_is_raw_checkbox'] = widgets.Checkbox(value=False, description='Is Raw', disabled=True,
                                                                             indent=False, description_tooltip='check it if the XANES data in the file is not normalized')
        layout = {'width':'20%'}
        self.hs['L[0][1][0][1][1][0][0]_is_raw_checkbox'].layout = layout
        self.hs['L[0][1][0][1][1][0][1]_is_refine_checkbox'] = widgets.Checkbox(value=False, description='Is Refine', disabled=True,
                                                                                indent=False, description_tooltip='checkit if the XANES data is already pre-aligned')
        layout = {'width':'20%'}
        self.hs['L[0][1][0][1][1][0][1]_is_refine_checkbox'].layout = layout
        self.hs['L[0][1][0][1][1][0][2]_norm_scale_text'] = widgets.BoundedFloatText(value=1,
                                                                                     min=1e-10,
                                                                                     max=100,
                                                                                     step=0.1,
                                                                                     description='Norm Scale', disabled=True,
                                                                                     description_tooltip='scale the XANES data with a factor if the normalized data is not in range [0, 1]')
        layout = {'width':'20%'}
        self.hs['L[0][1][0][1][1][0][2]_norm_scale_text'].layout = layout
        layout = {'width':'20%'}
        self.hs['L[0][1][0][1][1][0][3]_use_alternative_flat_checkbox'] = widgets.Checkbox(value=False, description='alt flat', disabled=True,
                                                                                           indent=False)
        self.hs['L[0][1][0][1][1][0][3]_use_alternative_flat_checkbox'].layout = layout
        
        layout = {'width':'15%'}
        self.hs['L[0][1][0][1][1][0][4]_alternative_flat_file_button'] = SelectFilesButton(option='askopenfilename',
                                                                                           **{'open_filetypes':(('h5 file', '*.h5'),)},
                                                                                           disabled=True)
        self.hs['L[0][1][0][1][1][0][4]_alternative_flat_file_button'].description = 'select alt flat'
        self.hs['L[0][1][0][1][1][0][4]_alternative_flat_file_button'].layout = layout
        self.hs['L[0][1][0][1][1][0][4]_alternative_flat_file_button'].disabled = True
        
        self.hs['L[0][1][0][1][1][0]_data_preprocessing_options_box0'].children = get_handles(self.hs, 'L[0][1][0][1][1][0]_data_preprocessing_options_box0', -1)
        self.hs['L[0][1][0][1][1][0][0]_is_raw_checkbox'].observe(self.L0_1_0_1_1_0_0_is_raw_checkbox_change, names='value')
        self.hs['L[0][1][0][1][1][0][1]_is_refine_checkbox'].observe(self.L0_1_0_1_1_0_1_is_refine_checkbox_change, names='value')
        self.hs['L[0][1][0][1][1][0][2]_norm_scale_text'].observe(self.L0_1_0_1_1_0_2_norm_scale_text_change, names='value')
        self.hs['L[0][1][0][1][1][0][3]_use_alternative_flat_checkbox'].observe(self.L0_1_0_1_1_0_3_use_alternative_flat_checkbox_change, names='value')
        self.hs['L[0][1][0][1][1][0][4]_alternative_flat_file_button'].on_click(self.L0_1_0_1_1_0_4_alternative_flat_file_button_click)

        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-104}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][0][1][1][1]_data_preprocessing_options_box1'] = widgets.HBox()
        self.hs['L[0][1][0][1][1][1]_data_preprocessing_options_box1'].layout = layout
        
        layout = {'width':'20%'}
        self.hs['L[0][1][0][1][1][1][1]_smooth_flat_checkbox'] = widgets.Checkbox(value=False, description='smooth flat', disabled=True,
                                                                                  indent=False, description_tooltip='smooth the flat image to reduce artifacts from background image')
        self.hs['L[0][1][0][1][1][1][1]_smooth_flat_checkbox'].layout = layout
        layout = {'width':'20%'}
        self.hs['L[0][1][0][1][1][1][2]_smooth_flat_sigma_text'] = widgets.BoundedFloatText(value=False,
                                                                                            min=0,
                                                                                            max=30,
                                                                                            step=0.1,
                                                                                            description='smooth sigma', disabled=True)
        self.hs['L[0][1][0][1][1][1][2]_smooth_flat_sigma_text'].layout = layout
        layout = {'left':'40.9%', 'width':'15%'}
        self.hs['L[0][1][0][1][1][1][3]_config_data_load_images_button'] = widgets.Button(description='Load Images', disabled=True)
        self.hs['L[0][1][0][1][1][1][3]_config_data_load_images_button'].layout = layout
        self.hs['L[0][1][0][1][1][1][3]_config_data_load_images_button'].style.button_color = 'darkviolet'
        
        self.hs['L[0][1][0][1][1][1][1]_smooth_flat_checkbox'].observe(self.L0_1_0_1_1_1_1_smooth_flat_checkbox_change, names='value')
        self.hs['L[0][1][0][1][1][1][2]_smooth_flat_sigma_text'].observe(self.L0_1_0_1_1_1_2_smooth_flat_sigma_text_change, names='value')
        self.hs['L[0][1][0][1][1][1][3]_config_data_load_images_button'].on_click(self.L0_1_0_1_1_1_3_config_data_load_images_button_click)
        self.hs['L[0][1][0][1][1][1]_data_preprocessing_options_box1'].children = get_handles(self.hs, 'L[0][1][0][1][1][1]_data_preprocessing_options_box1', -1)

        self.hs['L[0][1][0][1][1]_data_preprocessing_options_box'].children = get_handles(self.hs, 'L[0][1][0][1][1]_data_preprocessing_options_box', -1)
        ## ## ## ## ## data_preprocessing_options -- end


        ## ## ## ## ## define eng points -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][0][1][2]_eng_range_box'] = widgets.HBox()
        self.hs['L[0][1][0][1][2]_eng_range_box'].layout = layout
        self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'] = widgets.IntRangeSlider(value=0, description='eng points range', disabled=True)
        layout = {'width':'55%'}
        self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].layout = layout
        self.hs['L[0][1][0][1][2][1]_eng_s_text'] = widgets.BoundedFloatText(value=0,
                                                                             min=1e3,
                                                                             max=5e4,
                                                                             step=0.1,
                                                                             description='eng s (eV)', disabled=True)
        layout = {'width':'20%'}
        self.hs['L[0][1][0][1][2][1]_eng_s_text'].layout = layout
        self.hs['L[0][1][0][1][2][2]_eng_e_text'] = widgets.BoundedFloatText(value=0,
                                                                             min=1e3,
                                                                             max=5e4,
                                                                             step=0.1,
                                                                             description='eng e (eV)', disabled=True)
        layout = {'width':'20%'}
        self.hs['L[0][1][0][1][2][2]_eng_e_text'].layout = layout
        self.hs['L[0][1][0][1][2]_eng_range_box'].children = get_handles(self.hs, 'L[0][1][0][1][2]_eng_range_box', -1)
        self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].observe(self.L0_1_0_1_2_0_eng_points_range_slider_change, names='value')
        ## ## ## ## ## define eng points -- end

        ## ## ## ## ## fiji option -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][0][1][3]_fiji_box'] = widgets.HBox()
        self.hs['L[0][1][0][1][3]_fiji_box'].layout = layout
        self.hs['L[0][1][0][1][3][0]_fiji_raw_img_preview_checkbox'] = widgets.Checkbox(value=False, description='fiji view', disabled=True,
                                                                                        indent=False)
        layout = {'width':'20%'}
        self.hs['L[0][1][0][1][3][0]_fiji_raw_img_preview_checkbox'].layout = layout
        self.hs['L[0][1][0][1][3][2]_fiji_eng_id_slider'] = widgets.IntSlider(value=False, description='img #', disabled=True, min=0)
        layout = {'width':'60.6%'}
        self.hs['L[0][1][0][1][3][2]_fiji_eng_id_slider'].layout = layout
        self.hs['L[0][1][0][1][3][1]_fiji_close_button'] = widgets.Button(description='close all fiji viewers', disabled=True)
        layout = {'width':'15%'}
        self.hs['L[0][1][0][1][3][1]_fiji_close_button'].layout = layout
        self.hs['L[0][1][0][1][3]_fiji_box'].children = get_handles(self.hs, 'L[0][1][0][1][3]_fiji_box', -1)
        self.hs['L[0][1][0][1][3][0]_fiji_raw_img_preview_checkbox'].observe(self.L0_1_0_1_3_0_fiji_virtural_stack_preview_checkbox_change, names='value')
        self.hs['L[0][1][0][1][3][2]_fiji_eng_id_slider'].observe(self.L0_1_0_1_3_2_fiji_eng_id_slider, names='value')
        self.hs['L[0][1][0][1][3][1]_fiji_close_button'].on_click(self.L0_1_0_1_3_1_fiji_close_button_click)
        ## ## ## ## ## fiji option -- end

        ## ## ## ## ## confirm data configuration -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][0][1][4]_config_data_confirm_box'] = widgets.HBox()
        self.hs['L[0][1][0][1][4]_config_data_confirm_box'].layout = layout
        self.hs['L[0][1][0][1][4][1]_confirm_config_data_text'] = widgets.Text(value='Confirm setting once you are done ...', description='', disabled=True)
        layout = {'width':'81.0%'}
        self.hs['L[0][1][0][1][4][1]_confirm_config_data_text'].layout = layout
        self.hs['L[0][1][0][1][4][0]_confirm_config_data_button'] = widgets.Button(description='Confirm',
                                                                                description_tooltip='Confirm: Confirm after you finish file configuration',
                                                                                disabled=True)
        self.hs['L[0][1][0][1][4][0]_confirm_config_data_button'].style.button_color = 'darkviolet'
        layout = {'width':'15%'}
        self.hs['L[0][1][0][1][4][0]_confirm_config_data_button'].layout = layout
        self.hs['L[0][1][0][1][4]_config_data_confirm_box'].children = get_handles(self.hs, 'L[0][1][0][1][4]_config_data_confirm_box', -1)
        self.hs['L[0][1][0][1][4][0]_confirm_config_data_button'].on_click(self.L0_1_0_1_4_0_confirm_config_data_button_click)
        ## ## ## ## ## confirm data configuration -- end

        self.hs['L[0][1][0][1]_config_data_box'].children = get_handles(self.hs, 'L[0][1][0][1]_config_indices_box', -1)
        ## ## ## ## bin widgets in hs['L[0][1][0][1]_config_data_box']  - config data -- end
        self.hs['L[0][1][0]_config_input_form'].children = get_handles(self.hs, 'L[0][1][0]_config_input_form', -1)
        ## ## ## define 2D_XANES_tabs layout file&data configuration-- end


        ## ## ## define 2D_XANES_tabs layout - registration configuration --start
        ## ## ## ## define functional widgets each tab in each sub-tab - config roi -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.28*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][1][0]_2D_roi_box'] = widgets.VBox()
        self.hs['L[0][1][1][0]_2D_roi_box'].layout = layout
        ## ## ## ## ## label 2D_roi_title box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][1][0][0]_2D_roi_title_box'] = widgets.HBox()
        self.hs['L[0][1][1][0][0]_2D_roi_title_box'].layout = layout
        self.hs['L[0][1][1][0][0][0]_2D_roi_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Config 2D ROI' + '</span>')
        layout = {'justify-content':'center', 'background-color':'white', 'color':'cyan', 'left':'43%'}
        self.hs['L[0][1][1][0][0][0]_2D_roi_title_text'].layout = layout
        self.hs['L[0][1][1][0][0]_2D_roi_title_box'].children = get_handles(self.hs, 'L[0][1][1][0][0]_2D_roi_title_box', -1)
        ## ## ## ## ## label 2D_roi_title box -- end

        ## ## ## ## ## define roi -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.14*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][1][0][1]_2D_roi_define_box'] = widgets.VBox()
        self.hs['L[0][1][1][0][1]_2D_roi_define_box'].layout = layout
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-108}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'] = widgets.IntRangeSlider(value=[10, 40],
                                                                                min=1,
                                                                                max=50,
                                                                                step=1,
                                                                                description='x range:',
                                                                                disabled=True,
                                                                                continuous_update=False,
                                                                                orientation='horizontal',
                                                                                readout=True,
                                                                                readout_format='d')
        self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].layout = layout
        self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'] = widgets.IntRangeSlider(value=[10, 40],
                                                                                min=1,
                                                                                max=50,
                                                                                step=1,
                                                                                description='y range:',
                                                                                disabled=True,
                                                                                continuous_update=False,
                                                                                orientation='horizontal',
                                                                                readout=True,
                                                                                readout_format='d')
        self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].layout = layout
        self.hs['L[0][1][1][0][1]_2D_roi_define_box'].children = get_handles(self.hs, 'L[0][1][1][0][1]_2D_roi_define_box', -1)
        self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].observe(self.L0_1_1_0_1_0_2D_roi_x_slider_change, names='value')
        self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].observe(self.L0_1_1_0_1_1_2D_roi_y_slider_change, names='value')
        ## ## ## ## ## define roi -- end

        ## ## ## ## ## confirm roi -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][1][0][2]_2D_roi_confirm_box'] = widgets.HBox()
        self.hs['L[0][1][1][0][2]_2D_roi_confirm_box'].layout = layout
        layout = {'width':'70%', 'height':'90%'}
        self.hs['L[0][1][1][0][2][0]_confirm_roi_text'] = widgets.Text(description='',
                                                                   value='Please confirm after ROI is set ...',
                                                                   disabled=True)
        self.hs['L[0][1][1][0][2][0]_confirm_roi_text'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][1][1][0][2][1]_confirm_roi_button'] = widgets.Button(description='Confirm',
                                                                       description_tooltip='Confirm the roi once you define the ROI ...',
                                                                       disabled=True)
        self.hs['L[0][1][1][0][2][1]_confirm_roi_button'].style.button_color = 'darkviolet'
        self.hs['L[0][1][1][0][2][1]_confirm_roi_button'].layout = layout
        self.hs['L[0][1][1][0][2]_2D_roi_confirm_box'].children = get_handles(self.hs, 'L[0][1][1][0][2]_2D_roi_confirm_box', -1)
        self.hs['L[0][1][1][0][2][1]_confirm_roi_button'].on_click(self.L0_1_1_0_2_1_confirm_roi_button_click)
        ## ## ## ## ## confirm roi -- end

        self.hs['L[0][1][1][0]_2D_roi_box'].children = get_handles(self.hs, 'L[0][1][1][0]_2D_roi_box', -1)
        ## ## ## ## define functional widgets in each sub-tab - config roi -- end

        ## ## ## ## define functional widgets in each sub-tab - config registration -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.42*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][1][1]_config_reg_params_box'] = widgets.VBox()
        self.hs['L[0][1][1][1]_config_reg_params_box'].layout = layout

        ## ## ## ## ## label config_reg_params box --start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][1][1][0]_config_reg_params_title_box'] = widgets.HBox()
        self.hs['L[0][1][1][1][0]_config_reg_params_title_box'].layout = layout
        self.hs['L[0][1][1][1][0][0]_config_reg_params_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Config Reg Params' + '</span>')
        # self.hs['L[0][1][1][1][0][0]_config_reg_params_title_text'] = widgets.Text(value='Config Reg Params', disabled=True)
        layout = {'background-color':'white', 'color':'cyan', 'left':'40.5%'}
        self.hs['L[0][1][1][1][0][0]_config_reg_params_title_text'].layout = layout
        self.hs['L[0][1][1][1][0]_config_reg_params_title_box'].children = get_handles(self.hs, 'L[0][1][1][1][0]_config_reg_params_title_box', -1)
        ## ## ## ## ## label config_reg_params box --end

        ## ## ## ## ## fiji&anchor box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][1][1][1]_fiji&anchor_box'] = widgets.HBox()
        self.hs['L[0][1][1][1][1]_fiji&anchor_box'].layout = layout
        self.hs['L[0][1][1][1][1][0]_fiji_mask_viewer_checkbox'] = widgets.Checkbox(value=False,
                                                                        disabled=True,
                                                                        description='preview',
                                                                        indent=False)
        layout = {'width':'19%'}
        self.hs['L[0][1][1][1][1][0]_fiji_mask_viewer_checkbox'].layout = layout
        self.hs['L[0][1][1][1][1][1]_use_chunk_checkbox'] = widgets.Checkbox(value=True,
                                                                          disabled=True,
                                                                          description='use anchor',
                                                                          indent=False)
        layout = {'width':'19%'}
        self.hs['L[0][1][1][1][1][1]_use_chunk_checkbox'].layout = layout
        self.hs['L[0][1][1][1][1][2]_anchor_id_slider'] = widgets.IntSlider(value=1,
                                                                            min=1,
                                                                            disabled=True,
                                                                            description='anchor id')
        layout = {'width':'29%'}
        self.hs['L[0][1][1][1][1][2]_anchor_id_slider'].layout = layout
        self.hs['L[0][1][1][1][1][3]_chunk_sz_slider'] = widgets.IntSlider(value=7,
                                                                           min=1,
                                                                           disabled=True,
                                                                           description='chunk size')
        layout = {'width':'29%'}
        self.hs['L[0][1][1][1][1][3]_chunk_sz_slider'].layout = layout
        
        self.hs['L[0][1][1][1][1]_fiji&anchor_box'].children = get_handles(self.hs, 'L[0][1][1][1][1]_fiji&anchor_box', -1)
        self.hs['L[0][1][1][1][1][0]_fiji_mask_viewer_checkbox'].observe(self.L0_1_1_1_1_0_fiji_mask_viewer_checkbox_change, names='value')
        self.hs['L[0][1][1][1][1][1]_use_chunk_checkbox'].observe(self.L0_1_1_1_1_1_chunk_checkbox_change, names='value')
        self.hs['L[0][1][1][1][1][2]_anchor_id_slider'].observe(self.L0_1_1_1_1_2_anchor_id_slider_change, names='value')
        self.hs['L[0][1][1][1][1][3]_chunk_sz_slider'].observe(self.L0_1_1_1_1_3_chunk_sz_slider_change, names='value')
        ## ## ## ## ## fiji&anchor box -- end

        ## ## ## ## ## mask options box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][1][1][2]_mask_options_box'] = widgets.HBox()
        self.hs['L[0][1][1][1][2]_mask_options_box'].layout = layout
        self.hs['L[0][1][1][1][2][0]_use_mask_checkbox'] = widgets.Checkbox(value=False,
                                                                        disabled=True,
                                                                        description='use mask',
                                                                        indent=False)
        layout = {'width':'19%', 'flex-direction':'row'}
        self.hs['L[0][1][1][1][2][0]_use_mask_checkbox'].layout = layout
        self.hs['L[0][1][1][1][2][1]_mask_thres_slider'] = widgets.FloatSlider(value=False,
                                                                          disabled=True,
                                                                          description='mask thres',
                                                                          readout_format='.5f',
                                                                          min=-1.,
                                                                          max=1.,
                                                                          step=1e-5,
                                                                          indent=False)
        layout = {'width':'29%'}
        self.hs['L[0][1][1][1][2][1]_mask_thres_slider'].layout = layout
        self.hs['L[0][1][1][1][2][2]_mask_dilation_slider'] = widgets.IntSlider(value=False,
                                                                          disabled=True,
                                                                          description='mask dilation',
                                                                          min=0,
                                                                          max=30,
                                                                          step=1,
                                                                          indent=False)
        layout = {'width':'29%'}
        self.hs['L[0][1][1][1][2][2]_mask_dilation_slider'].layout = layout
        self.hs['L[0][1][1][1][2]_mask_options_box'].children = get_handles(self.hs, 'L[0][1][1][1][2]_mask_options_box', -1)
        self.hs['L[0][1][1][1][2][0]_use_mask_checkbox'].observe(self.L0_1_1_1_2_0_use_mask_checkbox_change, names='value')
        self.hs['L[0][1][1][1][2][1]_mask_thres_slider'].observe(self.L0_1_1_1_2_1_mask_thres_slider_change, names='value')
        self.hs['L[0][1][1][1][2][2]_mask_dilation_slider'].observe(self.L0_1_1_1_2_2_mask_dilation_slider_change, names='value')
        ## ## ## ## ## mask options box -- end

        ## ## ## ## ## smooth & chunk_size box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][1][1][3]_sli_search&chunk_size_box'] = widgets.HBox()
        self.hs['L[0][1][1][1][3]_sli_search&chunk_size_box'].layout = layout
        self.hs['L[0][1][1][1][3][0]_use_smooth_checkbox'] = widgets.Checkbox(value=False,
                                                                              disabled=True,
                                                                              description='smooth img',
                                                                              description_tooltip='enable option to smooth images before image registration',
                                                                              indent=False)
        layout = {'width':'19%'}
        self.hs['L[0][1][1][1][3][0]_use_smooth_checkbox'].layout = layout
        self.hs['L[0][1][1][1][3][1]_smooth_sigma_text'] = widgets.BoundedFloatText(value=5,
                                                                                    min=0,
                                                                                    max=30,
                                                                                    step=0.1,
                                                                                    disabled=True,
                                                                                    description_tooltip='kernel width for smoothing images',
                                                                                    description='smooth sig')
        layout = {'width':'19%'}
        self.hs['L[0][1][1][1][3][1]_smooth_sigma_text'].layout = layout
        self.hs['L[0][1][1][1][3][2]_reg_method_dropdown'] = widgets.Dropdown(value='MRTV',
                                                                              options=['MRTV', 'MPC', 'PC', 'SR', 'LS+MRTV', 'MPC+MRTV'],
                                                                              description='reg method',
                                                                              description_tooltip='reg method: MRTV: Multi-resolution Total Variantion, MPC: Masked Phase Correlation; PC: Phase Correlation; SR: StackReg; LS+MRTV: a hybrid TV minimization combining line search and multiresolution strategy; MPC+MRTV: a hybrid search algorihm with MPC as primary search followed by subpixel TV search',
                                                                              disabled=True)
        layout = {'width':'19%'}
        self.hs['L[0][1][1][1][3][2]_reg_method_dropdown'].layout = layout
        self.hs['L[0][1][1][1][3][3]_ref_mode_dropdown'] = widgets.Dropdown(value='single',
                                                                            options=['single', 'neighbor', 'average'],
                                                                            description='ref mode',
                                                                            description_tooltip='ref mode: single: two images with fixed id gap in two consecutive chunks are used for the registration between two chunks; neighbor: two neighbor images in two consecutive chunks are used for the registration between two chunks; average: deprecated',
                                                                            disabled=True)
        layout = {'width':'19%'}
        self.hs['L[0][1][1][1][3][3]_ref_mode_dropdown'].layout = layout
        
        self.hs['L[0][1][1][1][3]_sli_search&chunk_size_box'].children = get_handles(self.hs, 'L[0][1][1][1][3]_sli_search&chunk_size_box', -1)
        self.hs['L[0][1][1][1][3][0]_use_smooth_checkbox'].observe(self.L0_1_1_1_3_0_use_smooth_checkbox_change, names='value')
        self.hs['L[0][1][1][1][3][1]_smooth_sigma_text'].observe(self.L0_1_1_1_3_1_smooth_sigma_text_change, names='value')
        self.hs['L[0][1][1][1][3][2]_reg_method_dropdown'].observe(self.L0_1_1_1_3_2_reg_method_dropdown_change, names='value')
        self.hs['L[0][1][1][1][3][3]_ref_mode_dropdown'].observe(self.L0_1_1_1_3_3_ref_mode_dropdown_change, names='value')
        ## ## ## ## ## smooth & chunk_size box -- end

        ## ## ## ## ##  reg_options box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][1][1][4]_mrtv_options_box'] = widgets.HBox()
        self.hs['L[0][1][1][1][4]_mrtv_options_box'].layout = layout
        
        self.hs['L[0][1][1][1][4][0]_mrtv_level_text'] = widgets.BoundedIntText(value=5,
                                                                                min=1,
                                                                                max=10,
                                                                                step=1,
                                                                                description='level',
                                                                                description_tooltip='level: multi-resolution level',
                                                                                disabled=True)
        layout = {'width':'19%'}
        self.hs['L[0][1][1][1][4][0]_mrtv_level_text'].layout = layout
        
        self.hs['L[0][1][1][1][4][1]_mrtv_width_text'] = widgets.BoundedIntText(value=10,
                                                                                min=1,
                                                                                max=20,
                                                                                step=1,
                                                                                description='width',
                                                                                description_tooltip='width: multi-resolution searching width at each level (number of searching steps)',
                                                                                disabled=True)
        layout = {'width':'19%'}
        self.hs['L[0][1][1][1][4][1]_mrtv_width_text'].layout = layout
        
        self.hs['L[0][1][1][1][4][2]_mrtv_subpixel_wz_text'] = widgets.BoundedIntText(value=8,
                                                                                 min=2,
                                                                                 max=20,
                                                                                 step=0.1,
                                                                                 description='subpxl width',
                                                                                 description_tooltip='subpxl width: final sub-pixel searching width (number of searching steps)',
                                                                                 disabled=True)
        layout = {'width':'19%'}
        self.hs['L[0][1][1][1][4][2]_mrtv_subpixel_wz_text'].layout = layout
        
        self.hs['L[0][1][1][1][4][3]_mrtv_subpixel_step_text'] = widgets.BoundedFloatText(value=0.5,
                                                                                 min=0.1,
                                                                                 max=1,
                                                                                 step=0.1,
                                                                                 description='subpxl step',
                                                                                 description_tooltip='subpxl step: final sub-pixel searching step size',
                                                                                 disabled=True)
        layout = {'width':'19%'}
        self.hs['L[0][1][1][1][4][3]_mrtv_subpixel_step_text'].layout = layout
        
        self.hs['L[0][1][1][1][4]_mrtv_options_box'].children = get_handles(self.hs, 'L[0][1][1][1][4]_mrtv_options_box', -1)
        
        self.hs['L[0][1][1][1][4][0]_mrtv_level_text'].observe(self.L0_1_1_1_4_0_mrtv_level_text_change, names='value')
        self.hs['L[0][1][1][1][4][1]_mrtv_width_text'].observe(self.L0_1_1_1_4_1_mrtv_width_text_change, names='value')
        self.hs['L[0][1][1][1][4][2]_mrtv_subpixel_wz_text'].observe(self.L0_1_1_1_4_2_mrtv_subpixel_wz_text_change, names='value')
        self.hs['L[0][1][1][1][4][3]_mrtv_subpixel_step_text'].observe(self.L0_1_1_1_4_3_mrtv_subpixel_step_text_change, names='value')
        ## ## ## ## ##  reg_options box -- end

        ## ## ## ## ## confirm reg settings -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][1][1][5]_config_reg_params_confirm_box'] = widgets.HBox()
        self.hs['L[0][1][1][1][5]_config_reg_params_confirm_box'].layout = layout
        layout = {'width':'70%', 'height':'90%'}
        self.hs['L[0][1][1][1][5][0]_confirm_reg_params_text'] = widgets.Text(description='',
                                                                   value='Confirm the roi once you define the ROI ...',
                                                                   disabled=True)
        self.hs['L[0][1][1][1][5][0]_confirm_reg_params_text'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][1][1][1][5][1]_confirm_reg_params_button'] = widgets.Button(description='Confirm',
                                                                       description_tooltip='Confirm the roi once you define the ROI ...',
                                                                       disabled=True)
        self.hs['L[0][1][1][1][5][1]_confirm_reg_params_button'].style.button_color = 'darkviolet'
        self.hs['L[0][1][1][1][5][1]_confirm_reg_params_button'].layout = layout
        self.hs['L[0][1][1][1][5]_config_reg_params_confirm_box'].children = get_handles(self.hs, 'L[0][1][1][1][5]_config_reg_params_confirm_box', -1)
        self.hs['L[0][1][1][1][5][1]_confirm_reg_params_button'].on_click(self.L0_1_1_1_5_1_confirm_reg_params_button_click)
        ## ## ## ## ## confirm reg settings -- end

        self.hs['L[0][1][1][1]_config_reg_params_box'].children = get_handles(self.hs, 'L[0][1][1][1]_config_reg_params_box', -1)
        ## ## ## ## define functional widgets in each sub-tab - config registration -- end


        ## ## ## ## define functional widgets in each sub-tab - run registration -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.21*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][1][2]_run_reg_box'] = widgets.VBox()
        self.hs['L[0][1][1][2]_run_reg_box'].layout = layout
        ## ## ## ## ## label run_reg box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][1][2][0]_run_reg_title_box'] = widgets.HBox()
        self.hs['L[0][1][1][2][0]_run_reg_title_box'].layout = layout
        self.hs['L[0][1][1][2][0][0]_run_reg_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Run Registration' + '</span>')
        layout = {'background-color':'white', 'color':'cyan', 'left':'41%'}
        self.hs['L[0][1][1][2][0][0]_run_reg_title_text'].layout = layout
        self.hs['L[0][1][1][2][0]_run_reg_title_box'].children = get_handles(self.hs, 'L[0][1][1][2][0]_run_reg_title_box', -1)
        ## ## ## ## ## label run_reg box -- end

        ## ## ## ## ## run reg & status -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][1][2][1]_run_reg_confirm_box'] = widgets.HBox()
        self.hs['L[0][1][1][2][1]_run_reg_confirm_box'].layout = layout
        layout = {'width':'70%', 'height':'90%'}
        self.hs['L[0][1][1][2][1][1]_run_reg_text'] = widgets.Text(description='',
                                                                   value='run registration once you are ready ...',
                                                                   disabled=True)
        self.hs['L[0][1][1][2][1][1]_run_reg_text'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][1][1][2][1][0]_run_reg_button'] = widgets.Button(description='Run Reg',
                                                                       description_tooltip='run registration once you are ready ...',
                                                                       disabled=True)
        self.hs['L[0][1][1][2][1][0]_run_reg_button'].style.button_color = 'darkviolet'
        self.hs['L[0][1][1][2][1][0]_run_reg_button'].layout = layout
        self.hs['L[0][1][1][2][1]_run_reg_confirm_box'].children = get_handles(self.hs, 'L[0][1][1][2][1]_run_reg_confirm_box', -1)
        self.hs['L[0][1][1][2][1][0]_run_reg_button'].on_click(self.L0_1_1_2_1_0_run_reg_button_click)
        ## ## ## ## ## run reg progress
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][1][2][2]_run_reg_progress_box'] = widgets.HBox()
        self.hs['L[0][1][1][2][2]_run_reg_progress_box'].layout = layout
        layout = {'width':'100%', 'height':'90%'}
        self.hs['L[0][1][1][2][2][0]_run_reg_progress_bar'] = widgets.IntProgress(value=0,
                                                                                  min=0,
                                                                                  max=10,
                                                                                  step=1,
                                                                                  description='Completing:',
                                                                                  bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                                                                  orientation='horizontal')
        self.hs['L[0][1][1][2][2][0]_run_reg_progress_bar'].layout = layout
        self.hs['L[0][1][1][2][2]_run_reg_progress_box'].children = get_handles(self.hs, 'L[0][1][1][2][2]_run_reg_progress_box', -1)
        ## ## ## ## ## run reg & status -- start

        self.hs['L[0][1][1][2]_run_reg_box'].children = get_handles(self.hs, 'L[0][1][1][2]_run_reg_box', -1)
        ## ## ## ## define functional widgets in each sub-tab - run registration -- end

        self.hs['L[0][1][1]_reg_setting_form'].children = get_handles(self.hs, 'L[0][1][1]_reg_setting_form', -1)
        ## ## ## define 2D_XANES_tabs layout - config registration --end


        ## ## ## define 2D_XANES_tabs layout - review/correct reg & align data -- start
        ## ## ## ## define functional widgets in each sub-tab - review/correct reg -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.35*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][2][0]_review_reg_results_box'] = widgets.VBox()
        self.hs['L[0][1][2][0]_review_reg_results_box'].layout = layout
        ## ## ## ## ## label review_reg_results box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][2][0][0]_review_reg_results_title_box'] = widgets.HBox()
        self.hs['L[0][1][2][0][0]_review_reg_results_title_box'].layout = layout
        self.hs['L[0][1][2][0][0][0]_review_reg_results_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Review/Correct Reg Results' + '</span>')
        layout = {'background-color':'white', 'color':'cyan', 'left':'35.7%'}
        self.hs['L[0][1][2][0][0][0]_review_reg_results_title_text'].layout = layout
        self.hs['L[0][1][2][0][0]_review_reg_results_title_box'].children = get_handles(self.hs, 'L[0][1][2][0][0]_review_reg_results_title_box', -1)
        ## ## ## ## ## label review_reg_results box -- end

        ## ## ## ## ## read alignment file -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][2][0][1]_read_alignment_file_box'] = widgets.HBox()
        self.hs['L[0][1][2][0][1]_read_alignment_file_box'].layout = layout
        layout = {'width':'70%', 'height':'90%'}
        self.hs['L[0][1][2][0][1][1]_read_alignment_checkbox'] = widgets.Checkbox(description='read alignment',
                                                                                  value=False,
                                                                                  disabled=True)
        self.hs['L[0][1][2][0][1][1]_read_alignment_checkbox'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][1][2][0][1][0]_read_alignment_button'] = SelectFilesButton(option='askopenfilename')
        self.hs['L[0][1][2][0][1][0]_read_alignment_button'].disabled = True
        self.hs['L[0][1][2][0][1][0]_read_alignment_button'].layout = layout
        self.hs['L[0][1][2][0][1]_read_alignment_file_box'].children = get_handles(self.hs, 'L[0][1][2][0][1]_read_alignment_file_box', -1)
        self.hs['L[0][1][2][0][1][1]_read_alignment_checkbox'].observe(self.L0_1_2_0_1_1_read_alignment_checkbox_change, names='value')
        self.hs['L[0][1][2][0][1][0]_read_alignment_button'].on_click(self.L0_1_2_0_1_0_read_alignment_button_click)
        ## ## ## ## ## read alignment file -- end

        ## ## ## ## ## reg pair box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][2][0][2]_reg_pair_box'] = widgets.HBox()
        self.hs['L[0][1][2][0][2]_reg_pair_box'].layout = layout
        layout = {'width':'70%', 'height':'90%'}
        self.hs['L[0][1][2][0][2][0]_reg_pair_slider'] = widgets.IntSlider(value=False,
                                                                           disabled=True,
                                                                           description='reg pair #')
        self.hs['L[0][1][2][0][2][0]_reg_pair_slider'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][1][2][0][2][1]_reg_pair_bad_button'] = widgets.Button(disabled=True,
                                                                        description='Bad')
        self.hs['L[0][1][2][0][2][1]_reg_pair_bad_button'].layout = layout
        self.hs['L[0][1][2][0][2][1]_reg_pair_bad_button'].style.button_color = 'darkviolet'
        self.hs['L[0][1][2][0][2]_reg_pair_box'].children = get_handles(self.hs, 'L[0][1][2][0][2]_reg_pair_box', -1)
        self.hs['L[0][1][2][0][2][0]_reg_pair_slider'].observe(self.L0_1_2_0_2_0_reg_pair_slider_change, names='value')
        self.hs['L[0][1][2][0][2][1]_reg_pair_bad_button'].on_click(self.L0_1_2_0_2_1_reg_pair_bad_button_click)
        ## ## ## ## ## reg pair box -- end

        ## ## ## ## ## zshift box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][2][0][3]_correct_shift_box'] = widgets.HBox()
        self.hs['L[0][1][2][0][3]_correct_shift_box'].layout = layout
        layout = {'width':'30%', 'height':'90%'}
        self.hs['L[0][1][2][0][3][0]_x_shift_text'] = widgets.FloatText(value=0,
                                                                         disabled=True,
                                                                         min = -100,
                                                                         max = 100,
                                                                         step=0.5,
                                                                         description='x shift')
        self.hs['L[0][1][2][0][3][0]_x_shift_text'].layout = layout
        layout = {'width':'30%', 'height':'90%'}
        self.hs['L[0][1][2][0][3][1]_y_shift_text'] = widgets.FloatText(value=0,
                                                                         disabled=True,
                                                                         min = -100,
                                                                         max = 100,
                                                                         step=0.5,
                                                                         description='y shift')
        self.hs['L[0][1][2][0][3][1]_y_shift_text'].layout = layout
        layout = {'left':'9.5%', 'width':'15%', 'height':'90%'}
        self.hs['L[0][1][2][0][3][2]_record_button'] = widgets.Button(description='Record',
                                                                       description_tooltip='Record',
                                                                       disabled=True)
        self.hs['L[0][1][2][0][3][2]_record_button'].layout = layout
        self.hs['L[0][1][2][0][3][2]_record_button'].style.button_color = 'darkviolet'
        self.hs['L[0][1][2][0][3]_correct_shift_box'].children = get_handles(self.hs, 'L[0][1][2][0][3]_zshift_box', -1)
        self.hs['L[0][1][2][0][3][0]_x_shift_text'].observe(self.L0_1_2_0_3_0_x_shift_text_change, names='value')
        self.hs['L[0][1][2][0][3][1]_y_shift_text'].observe(self.L0_1_2_0_3_1_y_shift_text_change, names='value')
        self.hs['L[0][1][2][0][3][2]_record_button'].on_click(self.L0_1_2_0_3_2_record_button_click)
        ## ## ## ## ## zshift box -- end

        ## ## ## ## ## confirm review results box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][2][0][4]_review_reg_results_comfirm_box'] = widgets.HBox()
        self.hs['L[0][1][2][0][4]_review_reg_results_comfirm_box'].layout = layout
        layout = {'width':'70%', 'height':'90%', 'display':'inline_flex'}
        self.hs['L[0][1][2][0][4][0]_confirm_review_results_text'] = widgets.Text(description='',
                                                                   value='Confirm after you finish reg review ...',
                                                                   disabled=True)
        self.hs['L[0][1][2][0][4][0]_confirm_review_results_text'].layout = layout
        layout = {'left':'0.%', 'width':'15%', 'height':'90%'}
        self.hs['L[0][1][2][0][4][1]_confirm_review_results_button'] = widgets.Button(description='Confirm',
                                                                       description_tooltip='Confirm after you finish reg review ...',
                                                                       disabled=True)
        self.hs['L[0][1][2][0][4][1]_confirm_review_results_button'].style.button_color = 'darkviolet'
        self.hs['L[0][1][2][0][4][1]_confirm_review_results_button'].layout = layout
        self.hs['L[0][1][2][0][4]_review_reg_results_comfirm_box'].children = get_handles(self.hs, 'L[0][1][2][0][4]_review_reg_results_comfirm_box', -1)
        self.hs['L[0][1][2][0][4][1]_confirm_review_results_button'].on_click(self.L0_1_2_0_4_1_confirm_review_results_button_click)
        ## ## ## ## ## confirm review results box -- end

        self.hs['L[0][1][2][0]_review_reg_results_box'].children = get_handles(self.hs, 'L[0][1][2][0]_review_reg_results_box', -1)
        ## ## ## ## define functional widgets in each sub-tab - review/correct reg -- end

        ## ## ## ## define functional widgets in each sub-tab - align data -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.21*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][2][1]_align_images_box'] = widgets.VBox()
        self.hs['L[0][1][2][1]_align_images_box'].layout = layout
        ## ## ## ## ## label align_recon box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][2][1][0]_align_images_title_box'] = widgets.HBox()
        self.hs['L[0][1][2][1][0]_align_images_title_box'].layout = layout
        self.hs['L[0][1][2][1][0][0]_align_images_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Align Images' + '</span>')
        layout = {'background-color':'white', 'color':'cyan', 'left':'41%'}
        self.hs['L[0][1][2][1][0][0]_align_images_title_text'].layout = layout
        self.hs['L[0][1][2][1][0]_align_images_title_box'].children = get_handles(self.hs, 'L[0][1][2][1][0]_align_images_title_box', -1)
        ## ## ## ## ## label align_recon box -- end

        ## ## ## ## ## define run reg & status -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][2][1][1]_align_recon_comfirm_box'] = widgets.HBox()
        self.hs['L[0][1][2][1][1]_align_recon_comfirm_box'].layout = layout
        layout = {'width':'70%', 'height':'90%'}
        self.hs['L[0][1][2][1][1][0]_align_text'] = widgets.Text(description='',
                                                                   value='Confirm to proceed alignment ...',
                                                                   disabled=True)
        self.hs['L[0][1][2][1][1][0]_align_text'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][1][2][1][1][1]_align_button'] = widgets.Button(description='Align',
                                                                       description_tooltip='This will perform xanes2D alignment according to your configurations ...',
                                                                       disabled=True)
        self.hs['L[0][1][2][1][1][1]_align_button'].style.button_color = 'darkviolet'
        self.hs['L[0][1][2][1][1][1]_align_button'].layout = layout
        self.hs['L[0][1][2][1][1]_align_recon_comfirm_box'].children = get_handles(self.hs, 'L[0][1][2][1][1]_align_recon_comfirm_box', -1)
        self.hs['L[0][1][2][1][1][1]_align_button'].on_click(self.L0_1_2_1_1_1_align_button_click)
        ## ## ## ## ## define run reg & status -- end

        ## ## ## ## ## define run reg progress -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][2][1][2]_align_progress_box'] = widgets.HBox()
        self.hs['L[0][1][2][1][2]_align_progress_box'].layout = layout
        layout = {'width':'100%', 'height':'90%'}
        self.hs['L[0][1][2][1][2][0]_align_progress_bar'] = widgets.IntProgress(value=0,
                                                                                  min=0,
                                                                                  max=10,
                                                                                  step=1,
                                                                                  description='Completing:',
                                                                                  bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                                                                  orientation='horizontal')
        self.hs['L[0][1][2][1][2][0]_align_progress_bar'].layout = layout
        self.hs['L[0][1][2][1][2]_align_progress_box'].children = get_handles(self.hs, 'L[0][1][2][1][2]_align_progress_box', -1)
        ## ## ## ## ## define run reg progress -- end

        self.hs['L[0][1][2][1]_align_images_box'].children = get_handles(self.hs, 'L[0][1][2][1]_align_images_box', -1)
        ## ## ## ## define functional widgets in each sub-tab - align recon in register/review/shift TAB -- end

        ## ## ## define 2D_XANES_tabs layout - visualization&analysis reg & align data -- start
        ## ## ## ## define functional widgets in each sub-tab - visualizaton TAB -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.14*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][2][2]_visualize_images_box'] = widgets.VBox()
        self.hs['L[0][1][2][2]_visualize_images_box'].layout = layout
        ## ## ## ## ## define visualization title -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][2][2][0]_visualize_images_title_box'] = widgets.HBox()
        self.hs['L[0][1][2][2][0]_visualize_images_title_box'].layout = layout
        self.hs['L[0][1][2][2][0][0]_visualize_images_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Visualize Images' + '</span>')
        layout = {'background-color':'white', 'color':'cyan', 'left':'42%'}
        self.hs['L[0][1][2][2][0][0]_visualize_images_title_text'].layout = layout
        self.hs['L[0][1][2][2][0]_visualize_images_title_box'].children = get_handles(self.hs, 'L[0][1][2][2][0]_visualize_images_title_box', -1)
        ## ## ## ## ## define visualization title -- end

        ## ## ## ## ## define visualization&confirm -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][1][2][2][1]_visualization&comfirm_box'] = widgets.HBox()
        self.hs['L[0][1][2][2][1]_visualization&comfirm_box'].layout = layout
        layout = {'width':'60%', 'height':'90%'}
        self.hs['L[0][1][2][2][1][0]_visualization_slider'] = widgets.IntSlider(description='eng #',
                                                                                min=0,
                                                                                max=10,
                                                                                disabled=True)
        self.hs['L[0][1][2][2][1][0]_visualization_slider'].layout = layout
        layout = {'width':'10%', 'height':'90%'}
        self.hs['L[0][1][2][2][1][3]_visualization_eng_text'] = widgets.FloatText(description='',
                                                                                     value=0,
                                                                                     disabled=True)
        self.hs['L[0][1][2][2][1][3]_visualization_eng_text'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][1][2][2][1][2]_visualization_R&C_checkbox'] = widgets.Checkbox(description='Auto R&C',
                                                                                     value=True,
                                                                                     disabled=True)
        self.hs['L[0][1][2][2][1][2]_visualization_R&C_checkbox'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][1][2][2][1][1]_spec_in_roi_button'] = widgets.Button(description='spec in roi',
                                                                       description_tooltip='This will perform xanes2D alignment according to your configurations ...',
                                                                       disabled=True)
        self.hs['L[0][1][2][2][1][1]_spec_in_roi_button'].style.button_color = 'darkviolet'
        self.hs['L[0][1][2][2][1][1]_spec_in_roi_button'].layout = layout
        self.hs['L[0][1][2][2][1]_visualization&comfirm_box'].children = get_handles(self.hs, 'L[0][1][2][2][1]_visualization&comfirm_box', -1)
        self.hs['L[0][1][2][2][1][0]_visualization_slider'].observe(self.L0_1_2_2_1_0_visualization_slider_change, names='value')
        self.hs['L[0][1][2][2][1][2]_visualization_R&C_checkbox'].observe(self.L0_1_2_2_1_2_visualization_RC_checkbox_change, names='value')
        self.hs['L[0][1][2][2][1][1]_spec_in_roi_button'].on_click(self.L0_1_2_2_1_1_spec_in_roi_button_click)
        ## ## ## ## ## define visualization&confirm -- end
        self.hs['L[0][1][2][2]_visualize_images_box'].children = get_handles(self.hs, 'L[0][1][2][2]_visualize_images_box', -1)
        ## ## ## ## define functional widgets in each sub-tab - visualizaton TAB -- end
        self.hs['L[0][1][2]_reg&review_form'].children = get_handles(self.hs, 'L[0][1][2]_reg&review_form', -1)
        ## ## ## define 2D_XANES_tabs layout - review/correct reg & align data -- end

        self.hs['L[0][1][3]_fitting_form'].children = [self.xanes_fitting_gui_h.hs['L[0][x][3][0]_fitting_box']]
        ## ## ## define 2D_XANES_tabs layout - visualization&analysis reg & align data -- end
        
        self.hs['L[0][1][4]_analysis_form'].children = [self.xanes_analysis_gui_h.hs['L[0][x][4][0]_ana_box']]
        ## ## ## define 2D_XANES_tabs layout - analysis box -- end
        
        self.hs['L[0][1][0][0][1][0]_select_raw_h5_path_button'].initialdir = self.global_h.cwd
        self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].initialdir = self.global_h.cwd
        self.hs['L[0][1][0][1][1][0][4]_alternative_flat_file_button'].initialdir = self.global_h.cwd
        self.hs['L[0][1][2][0][1][0]_read_alignment_button'].initialdir = self.global_h.cwd
            
    def L0_1_0_0_1_0_select_raw_h5_path_button_click(self, a):
        if len(a.files[0]) != 0:
            self.xanes2D_file_raw_h5_filename = os.path.abspath(a.files[0])
            self.hs['L[0][1][0][0][1][0]_select_raw_h5_path_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
            self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
            self.hs['L[0][1][0][1][1][0][4]_alternative_flat_file_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
            with open(self.global_h.GUI_cfg_file, 'w') as f:
                json.dump({'cwd':os.path.abspath(a.files[0])},f)
            self.xanes2D_file_raw_h5_set = True
        else:
            self.hs['L[0][1][0][0][1][1]_select_raw_h5_path_text'].value = 'Select raw h5 file ...'
            self.xanes2D_file_raw_h5_set = False
        self.xanes2D_file_configured = False
        self.hs['L[0][1][0][0][3][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        self.boxes_logic()

    def L0_1_0_0_3_0_select_save_trial_button_click(self, a):
        if self.xanes2D_file_analysis_option == 'Do New Reg':
            if len(a.files[0]) != 0:
                self.xanes2D_save_trial_reg_filename_template = a.files[0]
                self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
                with open(self.global_h.GUI_cfg_file, 'w') as f:
                    json.dump({'cwd':os.path.dirname(os.path.abspath(a.files[0]))},f)
                self.xanes2D_file_save_trial_set = True
                self.xanes2D_file_reg_file_set = False
                self.xanes2D_file_config_file_set = False
            else:
                self.hs['L[0][1][0][0][2][1]_select_save_trial_text'].value='Save trial registration as ...'
                self.xanes2D_file_save_trial_set = False
                self.xanes2D_file_reg_file_set = False
                self.xanes2D_file_config_file_set = False
            self.xanes2D_file_configured = False
            self.hs['L[0][1][0][0][3][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        elif self.xanes2D_file_analysis_option == 'Read Config File':
            if len(a.files[0]) != 0:
                self.xanes2D_file_save_trial_reg_config_filename_original = a.files[0]
                b = ''
                t = (time.strptime(time.asctime()))
                for ii in range(6):
                    b +=str(t[ii]).zfill(2)+'-'
                self.xanes2D_file_save_trial_reg_config_filename = os.path.abspath(a.files[0]).split('config')[0]+'config_'+b.strip('-')+'.json'
                self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
                self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].initialfile = os.path.basename(a.files[0])
                self.hs['L[0][1][0][0][2][1]_select_save_trial_text'].value='Configuration File is Read ...'
                with open(self.global_h.GUI_cfg_file, 'w') as f:
                    json.dump({'cwd':os.path.dirname(os.path.abspath(a.files[0]))},f)
                self.xanes2D_file_save_trial_set = False
                self.xanes2D_file_reg_file_set = False
                self.xanes2D_file_config_file_set = True
            else:
                self.hs['L[0][1][0][0][2][1]_select_save_trial_text'].value='Save Existing Configuration File ...'
                self.xanes2D_file_save_trial_set = False
                self.xanes2D_file_reg_file_set = False
                self.xanes2D_file_config_file_set = False
            self.xanes2D_file_configured = False
            self.hs['L[0][1][0][0][3][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        elif self.xanes2D_file_analysis_option == 'Reg By Shift':
            if len(a.files[0]) != 0:
                self.xanes2D_file_save_trial_reg_config_filename_original = a.files[0]
                b = ''
                t = (time.strptime(time.asctime()))
                for ii in range(6):
                    b +=str(t[ii]).zfill(2)+'-'
                self.xanes2D_file_save_trial_reg_config_filename = os.path.abspath(a.files[0]).split('config')[0]+'config_'+b.strip('-')+'.json'
                self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
                self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].initialfile = os.path.basename(a.files[0])
                self.hs['L[0][1][0][0][2][1]_select_save_trial_text'].value='Configuration File is Read ...'
                with open(self.global_h.GUI_cfg_file, 'w') as f:
                    json.dump({'cwd':os.path.dirname(os.path.abspath(a.files[0]))},f)
                self.xanes2D_file_save_trial_set = False
                self.xanes2D_file_reg_file_set = False
                self.xanes2D_file_config_file_set = True
            else:
                self.hs['L[0][1][0][0][2][1]_select_save_trial_text'].value='Save Existing Configuration File ...'
                self.xanes2D_file_save_trial_set = False
                self.xanes2D_file_reg_file_set = False
                self.xanes2D_file_config_file_set = False
            self.xanes2D_file_configured = False
            self.hs['L[0][1][0][0][3][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        elif self.xanes2D_file_analysis_option == 'Do Analysis':
            if len(a.files[0]) != 0:
                self.xanes2D_save_trial_reg_filename = os.path.abspath(a.files[0])
                b = ''
                t = (time.strptime(time.asctime()))
                for ii in range(6):
                    b +=str(t[ii]).zfill(2)+'-'
                template = ''
                for ii in os.path.abspath(a.files[0]).split('_')[:-2]:
                    template += (ii+'_')
                self.xanes2D_file_save_trial_reg_config_filename = template+'config_'+b.strip('-')+'.json'
                self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
                self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].initialfile = os.path.basename(a.files[0])
                with open(self.global_h.GUI_cfg_file, 'w') as f:
                    json.dump({'cwd':os.path.dirname(os.path.abspath(a.files[0]))},f)
                self.hs['L[0][1][0][0][2][1]_select_save_trial_text'].value='Existing Registration File is Read ...'
                self.xanes2D_file_save_trial_set = False
                self.xanes2D_file_reg_file_set = True
                self.xanes2D_file_config_file_set = False
            else:
                self.hs['L[0][1][0][0][2][1]_select_save_trial_text'].value='Read Existing Registration File ...'
                self.xanes2D_file_save_trial_set = False
                self.xanes2D_file_reg_file_set = False
                self.xanes2D_file_config_file_set = False
            self.xanes2D_file_configured = False
            self.hs['L[0][1][0][0][3][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        self.boxes_logic()

    def L0_1_0_0_3_2_file_path_options_dropdown_change(self,a ):
        restart(self, dtype='2D_XANES')
        self.xanes2D_file_analysis_option = a['owner'].value
        self.xanes2D_file_configured = False
        if self.xanes2D_file_analysis_option == 'Do New Reg':
            self.hs['L[0][1][0][0][1][0]_select_raw_h5_path_button'].disabled = False
            self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].option = 'asksaveasfilename'
            self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].description = 'Save Reg File'
            self.hs['L[0][1][0][0][2][1]_select_save_trial_text'].value='Save trial registration as ...'
            self.hs['L[0][1][2][0][1][1]_read_alignment_checkbox'].value = False
        elif self.xanes2D_file_analysis_option == 'Read Config File':
            self.hs['L[0][1][0][0][1][0]_select_raw_h5_path_button'].disabled = True
            self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].option = 'askopenfilename'
            self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].description = 'Read Config'
            self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].open_filetypes = (('json files', '*.json'), ('text files', '*.txt'))
            self.hs['L[0][1][0][0][2][1]_select_save_trial_text'].value='Save Existing Configuration File ...'
        elif self.xanes2D_file_analysis_option == 'Reg By Shift':
            self.hs['L[0][1][0][0][1][0]_select_raw_h5_path_button'].disabled = True
            self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].option = 'askopenfilename'
            self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].description = 'Read Config'
            self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].open_filetypes = (('json files', '*.json'), ('text files', '*.txt'))
            self.hs['L[0][1][0][0][2][1]_select_save_trial_text'].value='Save Existing Configuration File ...'
        elif self.xanes2D_file_analysis_option == 'Do Analysis':
            self.hs['L[0][1][0][0][1][0]_select_raw_h5_path_button'].disabled = True
            self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].option = 'askopenfilename'
            self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].description = 'Read Reg File'
            self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].open_filetypes = (('h5 files', '*.h5'),)
            self.hs['L[0][1][0][0][2][1]_select_save_trial_text'].value='Read Existing Registration File ...'
        self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].icon = "square-o"
        self.hs['L[0][1][0][0][2][0]_select_save_trial_button'].style.button_color = "orange"
        self.hs['L[0][1][2][0][4][0]_confirm_review_results_text'].value = 'Please comfirm your change ...'
        self.boxes_logic()

    def L0_1_0_0_4_0_confirm_file_path_button_click(self, a):
        if self.xanes2D_file_analysis_option == 'Do New Reg':
            if not self.xanes2D_file_raw_h5_set:
                self.hs['L[0][1][0][0][3][1]_confirm_file&path_text'].value = 'Please specifiy raw h5 file ...'
                self.xanes2D_file_configured = False
            elif not self.xanes2D_file_save_trial_set:
                self.hs['L[0][1][0][0][3][1]_confirm_file&path_text'].value = 'Please specifiy where to save trial reg result ...'
                self.xanes2D_file_configured = False
            else:
                b = ''
                t = (time.strptime(time.asctime()))
                for ii in range(6):
                    b +=str(t[ii]).zfill(2)+'-'
                self.xanes2D_save_trial_reg_filename = os.path.join(os.path.dirname(os.path.abspath(self.xanes2D_save_trial_reg_filename_template)),
                                                                         os.path.basename(os.path.abspath(self.xanes2D_save_trial_reg_filename_template)).split('.')[0]+
                                                                         '_'+os.path.basename(self.xanes2D_file_raw_h5_filename).split('.')[0]+'_'+b.strip('-')+'.h5')
                self.xanes2D_file_save_trial_reg_config_filename = os.path.join(os.path.dirname(os.path.abspath(self.xanes2D_save_trial_reg_filename_template)),
                                                                                os.path.basename(os.path.abspath(self.xanes2D_save_trial_reg_filename_template)).split('.')[0]+
                                                                                '_'+os.path.basename(self.xanes2D_file_raw_h5_filename).split('.')[0]+'_config_'+b.strip('-')+'.json')

                self.xanes2D_config_eng_list = self.reader(self.xanes2D_file_raw_h5_filename, 
                                                           dtype='eng', sli=[None],
                                                           cfg=self.global_h.io_xanes2D_cfg)
                if self.xanes2D_config_eng_list.max() < 70:
                    self.xanes2D_config_eng_list *= 1000
                    
                self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].value = (0,1)
                self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].min = 0
                self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].max = self.xanes2D_config_eng_list.shape[0]-1
                self.hs['L[0][1][0][1][2][1]_eng_s_text'].value = self.xanes2D_config_eng_list[0]
                self.hs['L[0][1][0][1][2][2]_eng_e_text'].value = self.xanes2D_config_eng_list[1]

                self.xanes2D_eng_id_s = self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].value[0]
                self.xanes2D_eng_id_e = self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].value[1]
                self.hs['L[0][1][0][0][3][1]_confirm_file&path_text'].value = 'XANES2D file config is done ...'
                self.update_xanes2D_config()
                self.xanes2D_review_reg_best_match_filename = os.path.splitext(self.xanes2D_file_save_trial_reg_config_filename)[0].replace('config', 'reg_best_match') + '.json'
                self.xanes2D_file_configured = True
        elif self.xanes2D_file_analysis_option == 'Read Config File':
            if not self.xanes2D_file_config_file_set:
                self.hs['L[0][1][0][0][3][1]_confirm_file&path_text'].value = 'Please specifiy the configuration file to be read ...'
                self.xanes2D_file_configured = False
            else:
                self.read_xanes2D_config()
                self.set_xanes2D_variables()
                
                self.xanes2D_config_eng_list = self.reader(self.xanes2D_file_raw_h5_filename, 
                                                           dtype='eng', sli=[None],
                                                           cfg=self.global_h.io_xanes2D_cfg)
                if self.xanes2D_config_eng_list.max() < 70:
                    self.xanes2D_config_eng_list *= 1000
                    
                if self.xanes2D_config_is_raw:
                    if self.xanes2D_config_use_alternative_flat:
                        if self.xanes2D_config_alternative_flat_set:
                            self.xanes2D_img = -np.log((self.reader(self.xanes2D_file_raw_h5_filename, dtype='data', sli=[None], cfg=self.global_h.io_xanes2D_cfg) -
                                                        self.reader(self.xanes2D_file_raw_h5_filename, dtype='dark', sli=[None], cfg=self.global_h.io_xanes2D_cfg))/
                                                       (self.reader(self.xanes2D_config_alternative_flat_filename, dtype='flat', sli=[None], cfg=self.global_h.io_xanes2D_cfg) -
                                                        self.reader(self.xanes2D_config_alternative_flat_filename, dtype='dark', sli=[None], cfg=self.global_h.io_xanes2D_cfg)))
                            self.xanes2D_img[np.isinf(self.xanes2D_img)] = 0
                            self.xanes2D_img[np.isnan(self.xanes2D_img)] = 0
                            self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].value = (0, self.xanes2D_img.shape[0]-1)
                            self.xanes2D_config_raw_img_readed = True
                        else:
                            self.xanes2D_config_raw_img_readed = False
                    else:
                        self.xanes2D_img = -np.log((self.reader(self.xanes2D_file_raw_h5_filename, dtype='data', sli=[None], cfg=self.global_h.io_xanes2D_cfg) -
                                                    self.reader(self.xanes2D_file_raw_h5_filename, dtype='dark', sli=[None], cfg=self.global_h.io_xanes2D_cfg))/
                                                   (self.reader(self.xanes2D_file_raw_h5_filename, dtype='flat', sli=[None], cfg=self.global_h.io_xanes2D_cfg) -
                                                    self.reader(self.xanes2D_file_raw_h5_filename, dtype='dark', sli=[None], cfg=self.global_h.io_xanes2D_cfg)))
                        self.xanes2D_img[np.isinf(self.xanes2D_img)] = 0
                        self.xanes2D_img[np.isnan(self.xanes2D_img)] = 0
                        self.xanes2D_config_raw_img_readed = True                        
                elif self.xanes2D_config_is_refine:
                    with h5py.File(self.xanes2D_save_trial_reg_filename, 'r') as f:
                        self.xanes2D_img  = f['/registration_results/reg_results/registered_xanes2D'][:]
                    self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].value = (0, self.xanes2D_img.shape[0]-1)
                    self.xanes2D_config_raw_img_readed = True
                else:
                    self.xanes2D_img = -np.log(self.reader(self.xanes2D_file_raw_h5_filename, dtype='data', sli=[None], cfg=self.global_h.io_xanes2D_cfg))
                    self.xanes2D_img[np.isinf(self.xanes2D_img)] = 0
                    self.xanes2D_img[np.isnan(self.xanes2D_img)] = 0
                    self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].value = (0, self.xanes2D_img.shape[0]-1)
                    self.xanes2D_config_raw_img_readed = True

                if self.xanes2D_data_configured:
                    self.xanes_analysis_eng_list = self.reader(self.xanes2D_file_raw_h5_filename, 
                                dtype='eng', sli=[None],
                                cfg=self.global_h.io_xanes2D_cfg)[self.xanes2D_eng_id_s:self.xanes2D_eng_id_e+1]
                    if self.xanes_analysis_eng_list.max() < 70:
                        self.xanes_analysis_eng_list *= 1000
                
                if self.xanes2D_roi_configured:
                    self.xanes2D_img_roi = self.xanes2D_img[self.xanes2D_eng_id_s:self.xanes2D_eng_id_e+1,
                                                            self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                            self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]]

                if self.xanes2D_reg_done:
                    with h5py.File(self.xanes2D_save_trial_reg_filename, 'r') as f:
                        self.xanes2D_review_aligned_img_original = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format(str(0).zfill(3))][:]
                        self.xanes2D_review_aligned_img = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format(str(0).zfill(3))][:]
                        self.xanes2D_review_fixed_img = f['/trial_registration/trial_reg_results/{0}/trial_reg_fixed{0}'.format(str(0).zfill(3))][:]
                    self.xanes2D_review_shift_dict = {}
                self.xanes2D_review_reg_best_match_filename = os.path.splitext(self.xanes2D_file_save_trial_reg_config_filename)[0].replace('config', 'reg_best_match') + '.json'
                self.xanes2D_file_configured = True
                                
                self.hs['L[0][1][1][1][1][0]_fiji_mask_viewer_checkbox'].value = False
                print(self.xanes2D_img_roi.shape)

                if self.xanes2D_alignment_done:
                    with h5py.File(self.xanes2D_save_trial_reg_filename, 'r') as f:
                        self.xanes_analysis_data_shape = f['/registration_results/reg_results/registered_xanes2D'].shape
                        self.xanes2D_img_roi = f['/registration_results/reg_results/registered_xanes2D'][:]
                        self.xanes_analysis_eng_list = f['/registration_results/reg_results/eng_list'][:]

                    self.hs['L[0][1][2][2][1][0]_visualization_slider'].min = 0
                    self.hs['L[0][1][2][2][1][0]_visualization_slider'].max = self.xanes_analysis_eng_list.shape[0]-1

                    self.xanes2D_review_reg_best_match_filename = os.path.splitext(self.xanes2D_file_save_trial_reg_config_filename)[0].replace('config', 'reg_best_match') + '.json'
                    self.xanes_element = determine_element(self.xanes_analysis_eng_list)
                    tem = determine_fitting_energy_range(self.xanes_element)
                    self.xanes_analysis_edge_eng = tem[0] 
                    self.xanes_analysis_wl_fit_eng_s = tem[1]
                    self.xanes_analysis_wl_fit_eng_e = tem[2]
                    self.xanes_analysis_pre_edge_e = tem[3]
                    self.xanes_analysis_post_edge_s = tem[4] 
                    self.xanes_analysis_edge_0p5_fit_s = tem[5]
                    self.xanes_analysis_edge_0p5_fit_e = tem[6]
                    self.xanes_analysis_type = 'full'
                    self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][0][0]_fit_eng_range_option_dropdown'].value = 'full'                
                self.set_xanes2D_handles()
                self.set_xanes2D_variables()
                fiji_viewer_off(self.global_h, self,viewer_name='all')
        elif self.xanes2D_file_analysis_option == 'Reg By Shift':
            if not self.xanes2D_file_config_file_set:
                self.hs['L[0][1][0][0][3][1]_confirm_file&path_text'].value = 'Please specifiy the configuration file to be read ...'
                self.xanes2D_file_configured = False
            else:
                self.read_xanes2D_config()
                self.set_xanes2D_variables()                
                if self.xanes2D_reg_review_done:
                    self.xanes2D_config_eng_list = self.reader(self.xanes2D_file_raw_h5_filename, 
                                                               dtype='eng', sli=[None],
                                                               cfg=self.global_h.io_xanes2D_cfg)
                    if self.xanes2D_config_eng_list.max() < 70:
                        self.xanes2D_config_eng_list *= 1000
                        
                    if self.xanes2D_config_is_raw:
                        if self.xanes2D_config_use_alternative_flat:
                            if self.xanes2D_config_alternative_flat_set:
                                self.xanes2D_img = -np.log((self.reader(self.xanes2D_file_raw_h5_filename, dtype='data', sli=[None], cfg=self.global_h.io_xanes2D_cfg) -
                                                            self.reader(self.xanes2D_file_raw_h5_filename, dtype='dark', sli=[None], cfg=self.global_h.io_xanes2D_cfg))/
                                                           (self.reader(self.xanes2D_config_alternative_flat_filename, dtype='flat', sli=[None], cfg=self.global_h.io_xanes2D_cfg) -
                                                            self.reader(self.xanes2D_config_alternative_flat_filename, dtype='dark', sli=[None], cfg=self.global_h.io_xanes2D_cfg)))
                                self.xanes2D_img[np.isinf(self.xanes2D_img)] = 0
                                self.xanes2D_img[np.isnan(self.xanes2D_img)] = 0
                                self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].value = (0, self.xanes2D_img.shape[0]-1)
                                self.xanes2D_config_raw_img_readed = True
                            else:
                                self.xanes2D_config_raw_img_readed = False
                        else:
                            self.xanes2D_img = -np.log((self.reader(self.xanes2D_file_raw_h5_filename, dtype='data', sli=[None], cfg=self.global_h.io_xanes2D_cfg) -
                                                        self.reader(self.xanes2D_file_raw_h5_filename, dtype='dark', sli=[None], cfg=self.global_h.io_xanes2D_cfg))/
                                                       (self.reader(self.xanes2D_file_raw_h5_filename, dtype='flat', sli=[None], cfg=self.global_h.io_xanes2D_cfg) -
                                                        self.reader(self.xanes2D_file_raw_h5_filename, dtype='dark', sli=[None], cfg=self.global_h.io_xanes2D_cfg)))
                            self.xanes2D_img[np.isinf(self.xanes2D_img)] = 0
                            self.xanes2D_img[np.isnan(self.xanes2D_img)] = 0
                            self.xanes2D_config_raw_img_readed = True                        
                    elif self.xanes2D_config_is_refine:
                        with h5py.File(self.xanes2D_save_trial_reg_filename, 'r') as f:
                            self.xanes2D_img  = f['/registration_results/reg_results/registered_xanes2D'][:]
                        self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].value = (0, self.xanes2D_img.shape[0]-1)
                        self.xanes2D_config_raw_img_readed = True
                    else:
                        self.xanes2D_img = -np.log(self.reader(self.xanes2D_file_raw_h5_filename, dtype='data', sli=[None], cfg=self.global_h.io_xanes2D_cfg))
                        self.xanes2D_img[np.isinf(self.xanes2D_img)] = 0
                        self.xanes2D_img[np.isnan(self.xanes2D_img)] = 0
                        self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].value = (0, self.xanes2D_img.shape[0]-1)
                        self.xanes2D_config_raw_img_readed = True
    
                    if self.xanes2D_data_configured:
                        self.xanes_analysis_eng_list = self.reader(self.xanes2D_file_raw_h5_filename, 
                                    dtype='eng', sli=[None],
                                    cfg=self.global_h.io_xanes2D_cfg)[self.xanes2D_eng_id_s:self.xanes2D_eng_id_e+1]
                        if self.xanes_analysis_eng_list.max() < 70:
                            self.xanes_analysis_eng_list *= 1000
                    
                    if self.xanes2D_roi_configured:
                        self.xanes2D_img_roi = self.xanes2D_img[self.xanes2D_eng_id_s:self.xanes2D_eng_id_e+1,
                                                                self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]]
    
                    if self.xanes2D_reg_done:
                        with h5py.File(self.xanes2D_save_trial_reg_filename, 'r') as f:
                            self.xanes2D_review_aligned_img_original = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format(str(0).zfill(3))][:]
                            self.xanes2D_review_aligned_img = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format(str(0).zfill(3))][:]
                            self.xanes2D_review_fixed_img = f['/trial_registration/trial_reg_results/{0}/trial_reg_fixed{0}'.format(str(0).zfill(3))][:]
                        self.xanes2D_review_shift_dict = {}
                    self.xanes2D_review_reg_best_match_filename = os.path.splitext(self.xanes2D_file_save_trial_reg_config_filename)[0].replace('config', 'reg_best_match') + '.json'
                    self.xanes2D_file_configured = True
                                     
                    self.set_xanes2D_handles()
                    self.set_xanes2D_variables()
                    self.xanes2D_data_configured = False
                    self.xanes2D_roi_configured = False
                    self.xanes2D_reg_review_done = False
                    self.xanes2D_alignment_done = False
                    self.xanes2D_file_analysis_option = 'Reg By Shift'
                    fiji_viewer_off(self.global_h, self,viewer_name='all')
                    self.hs['L[0][1][0][0][3][1]_confirm_file&path_text'].value = 'XANES2D file config is done ...'
                else:
                    self.xanes2D_file_configured = False
                    self.hs['L[0][1][0][0][3][1]_confirm_file&path_text'].value = 'To use this option, a config up to reg review is needed ...'
        elif self.xanes2D_file_analysis_option == 'Do Analysis':
            if not self.xanes2D_file_reg_file_set:
                self.hs['L[0][1][0][0][3][1]_confirm_file&path_text'].value = 'Please specifiy the aligned data to be read...'
                self.xanes2D_file_configured = False
                self.xanes2D_reg_review_done = False
                self.xanes2D_data_configured = False
                self.xanes2D_roi_configured = False
                self.xanes2D_reg_params_configured = False
                self.xanes2D_reg_done = False
                self.xanes2D_alignment_done = False
                self.xanes_analysis_eng_configured = False
            else:
                self.xanes2D_file_configured = True
                self.xanes2D_reg_review_done = False
                self.xanes2D_data_configured = False
                self.xanes2D_roi_configured = False
                self.xanes2D_reg_params_configured = False
                self.xanes2D_reg_done = False
                self.xanes2D_alignment_done = True
                self.xanes_analysis_eng_configured = False
                with h5py.File(self.xanes2D_save_trial_reg_filename, 'r') as f:
                    self.xanes_analysis_data_shape = f['/registration_results/reg_results/registered_xanes2D'].shape
                    self.xanes2D_img_roi = f['/registration_results/reg_results/registered_xanes2D'][:]
                    self.xanes_analysis_eng_list = f['/registration_results/reg_results/eng_list'][:]

                data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_analysis_viewer')
                if not viewer_state:
                    fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_analysis_viewer')
                self.global_h.xanes2D_fiji_windows['xanes2D_analysis_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
                    self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(self.xanes2D_img_roi)), self.global_h.ImagePlusClass))
                self.global_h.xanes2D_fiji_windows['xanes2D_analysis_viewer']['ip'].setSlice(0)
                self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")

                self.hs['L[0][1][2][2][1][0]_visualization_slider'].min = 0
                self.hs['L[0][1][2][2][1][0]_visualization_slider'].max = self.xanes_analysis_eng_list.shape[0]-1
                self.xanes2D_review_reg_best_match_filename = os.path.splitext(self.xanes2D_file_save_trial_reg_config_filename)[0].replace('config', 'reg_best_match') + '.json'
                self.xanes_element = determine_element(self.xanes_analysis_eng_list)
                tem = determine_fitting_energy_range(self.xanes_element)
                self.xanes_analysis_edge_eng = tem[0] 
                self.xanes_analysis_wl_fit_eng_s = tem[1]
                self.xanes_analysis_wl_fit_eng_e = tem[2]
                self.xanes_analysis_pre_edge_e = tem[3]
                self.xanes_analysis_post_edge_s = tem[4] 
                self.xanes_analysis_edge_0p5_fit_s = tem[5]
                self.xanes_analysis_edge_0p5_fit_e = tem[6]
                self.xanes_analysis_type = 'full'
                self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][0][0]_fit_eng_range_option_dropdown'].value = 'full'
        self.boxes_logic()

    def L0_1_0_1_1_0_1_is_refine_checkbox_change(self, a):
        self.xanes2D_config_is_refine = a['owner'].value
        self.xanes2D_config_raw_img_readed = False
        self.boxes_logic()

    def L0_1_0_1_1_0_0_is_raw_checkbox_change(self, a):
        self.xanes2D_config_is_raw = a['owner'].value
        self.xanes2D_config_raw_img_readed = False
        self.boxes_logic()

    def L0_1_0_1_1_0_2_norm_scale_text_change(self, a):
        self.xanes2D_config_img_scalar = a['owner'].value
        self.xanes2D_config_raw_img_readed = False
        self.boxes_logic()

    def L0_1_0_1_1_0_3_use_alternative_flat_checkbox_change(self, a):
        self.xanes2D_config_use_alternative_flat = a['owner'].value
        self.xanes2D_config_raw_img_readed = False
        self.boxes_logic()

    def L0_1_0_1_1_1_1_smooth_flat_checkbox_change(self, a):
        self.xanes2D_config_use_smooth_flat = a['owner'].value
        self.xanes2D_config_raw_img_readed = False
        self.boxes_logic()

    def L0_1_0_1_1_1_2_smooth_flat_sigma_text_change(self, a):
        self.xanes2D_config_smooth_flat_sigma = a['owner'].value
        self.xanes2D_config_raw_img_readed = False
        self.boxes_logic()

    def L0_1_0_1_1_0_4_alternative_flat_file_button_click(self, a):
        if len(a.files) != 0:
            self.xanes2D_config_alternative_flat_filename = a.files[0]
            self.hs['L[0][1][0][1][1][0][4]_alternative_flat_file_button'].description = 'selected'
            self.xanes2D_config_alternative_flat_set = True
        else:
            self.hs['L[0][1][0][1][1][0][4]_alternative_flat_file_button'].description = 'select alt flat'
            self.xanes2D_config_alternative_flat_set = False
        self.xanes2D_config_raw_img_readed = False
        self.boxes_logic()

    def L0_1_0_1_1_1_3_config_data_load_images_button_click(self, a):
        self.xanes2D_config_is_raw = self.hs['L[0][1][0][1][1][0][0]_is_raw_checkbox'].value
        self.xanes2D_config_is_refine = self.hs['L[0][1][0][1][1][0][1]_is_refine_checkbox'].value
        self.xanes2D_config_use_smooth_flat = self.hs['L[0][1][0][1][1][1][1]_smooth_flat_checkbox'].value
        self.xanes2D_config_use_alternative_flat = self.hs['L[0][1][0][1][1][0][3]_use_alternative_flat_checkbox'].value
        self.xanes2D_config_img_scalar = self.hs['L[0][1][0][1][1][0][2]_norm_scale_text'].value
        self.xanes2D_config_smooth_flat_sigma = self.hs['L[0][1][0][1][1][1][2]_smooth_flat_sigma_text'].value
        if self.xanes2D_config_is_raw:
            if self.xanes2D_config_use_alternative_flat:
                if self.xanes2D_config_alternative_flat_set:
                    self.xanes2D_img = -np.log((self.reader(self.xanes2D_file_raw_h5_filename, dtype='data', sli=[None], cfg=self.global_h.io_xanes2D_cfg) -
                                                self.reader(self.xanes2D_file_raw_h5_filename, dtype='dark', sli=[None], cfg=self.global_h.io_xanes2D_cfg))/
                                               (self.reader(self.xanes2D_config_alternative_flat_filename, dtype='flat', sli=[None], cfg=self.global_h.io_xanes2D_cfg) -
                                                self.reader(self.xanes2D_config_alternative_flat_filename, dtype='dark', sli=[None], cfg=self.global_h.io_xanes2D_cfg)))
                    self.xanes2D_img[np.isinf(self.xanes2D_img)] = 0
                    self.xanes2D_img[np.isnan(self.xanes2D_img)] = 0
                    self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].value = (0, self.xanes2D_img.shape[0]-1)
                    self.xanes2D_config_raw_img_readed = True
                else:
                    self.xanes2D_config_raw_img_readed = False
            else:
                self.xanes2D_img = -np.log((self.reader(self.xanes2D_file_raw_h5_filename, dtype='data', sli=[None], cfg=self.global_h.io_xanes2D_cfg) -
                                            self.reader(self.xanes2D_file_raw_h5_filename, dtype='dark', sli=[None], cfg=self.global_h.io_xanes2D_cfg))/
                                           (self.reader(self.xanes2D_file_raw_h5_filename, dtype='flat', sli=[None], cfg=self.global_h.io_xanes2D_cfg) -
                                            self.reader(self.xanes2D_file_raw_h5_filename, dtype='dark', sli=[None], cfg=self.global_h.io_xanes2D_cfg)))
                self.xanes2D_img[np.isinf(self.xanes2D_img)] = 0
                self.xanes2D_img[np.isnan(self.xanes2D_img)] = 0
                self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].value = (0, self.xanes2D_img.shape[0]-1)
                self.xanes2D_config_raw_img_readed = True
        elif self.xanes2D_config_is_refine:
            with h5py.File(self.xanes2D_file_raw_h5_filename, 'r') as f:
                self.xanes2D_img  = f['/registration_results/reg_results/registered_xanes2D'][:]
            self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].value = (0, self.xanes2D_img.shape[0]-1)
            self.xanes2D_config_raw_img_readed = True
        else:
            self.xanes2D_img = -np.log(self.reader(self.xanes2D_file_raw_h5_filename, dtype='data', sli=[None], cfg=self.global_h.io_xanes2D_cfg))
            self.xanes2D_img[np.isinf(self.xanes2D_img)] = 0
            self.xanes2D_img[np.isnan(self.xanes2D_img)] = 0
            self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].value = (0, self.xanes2D_img.shape[0]-1)
            self.xanes2D_config_raw_img_readed = True
        self.boxes_logic()

    def L0_1_0_1_2_0_eng_points_range_slider_change(self, a):
        self.xanes2D_eng_id_s = a['owner'].value[0]
        self.xanes2D_eng_id_e = a['owner'].value[1]
        self.hs['L[0][1][0][1][2][1]_eng_s_text'].value = self.xanes2D_config_eng_list[a['owner'].value[0]]
        self.hs['L[0][1][0][1][2][2]_eng_e_text'].value = self.xanes2D_config_eng_list[a['owner'].value[1]]
        self.hs['L[0][1][0][1][3][2]_fiji_eng_id_slider'].value = 0
        self.hs['L[0][1][0][1][3][2]_fiji_eng_id_slider'].min = 0
        self.hs['L[0][1][0][1][3][2]_fiji_eng_id_slider'].max = a['owner'].value[1] - a['owner'].value[0]
        self.boxes_logic()

    def L0_1_0_1_3_0_fiji_virtural_stack_preview_checkbox_change(self, a):
        if a['owner'].value:
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_raw_img_viewer')
            if not viewer_state:
                fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_raw_img_viewer')
        else:
            fiji_viewer_off(self.global_h, self, viewer_name='xanes2D_raw_img_viewer')
        self.boxes_logic()

    def L0_1_0_1_3_2_fiji_eng_id_slider(self, a):
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_raw_img_viewer')
        if not viewer_state:
            fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_raw_img_viewer')
        self.hs['L[0][1][0][1][3][0]_fiji_raw_img_preview_checkbox'].value = True
        if not data_state:
            self.hs['L[0][1][0][1][4][1]_confirm_config_roi_text'].value = 'xanes2D_img is not defined yet.'
        else:
            self.global_h.xanes2D_fiji_windows['xanes2D_raw_img_viewer']['ip'].setSlice(a['owner'].value+self.xanes2D_eng_id_s)
            self.hs['L[0][1][0][1][2][1]_eng_s_text'].value = self.xanes2D_config_eng_list[a['owner'].value+self.xanes2D_eng_id_s]
        # self.boxes_logic()

    def L0_1_0_1_3_1_fiji_close_button_click(self, a):
        fiji_viewer_off(self.global_h, self, viewer_name='all')
        self.boxes_logic()

    def L0_1_0_1_4_0_confirm_config_data_button_click(self, a):
        self.xanes2D_config_is_raw = self.hs['L[0][1][0][1][1][0][0]_is_raw_checkbox'].value
        self.xanes2D_config_is_refine = self.hs['L[0][1][0][1][1][0][1]_is_refine_checkbox'].value
        self.xanes2D_config_use_smooth_flat = self.hs['L[0][1][0][1][1][1][1]_smooth_flat_checkbox'].value
        self.xanes2D_config_use_alternative_flat = self.hs['L[0][1][0][1][1][0][3]_use_alternative_flat_checkbox'].value

        self.xanes2D_config_img_scalar = self.hs['L[0][1][0][1][1][0][2]_norm_scale_text'].value
        self.xanes2D_config_smooth_flat_sigma = self.hs['L[0][1][0][1][1][1][2]_smooth_flat_sigma_text'].value
        self.xanes2D_eng_id_s = self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].value[0]
        self.xanes2D_eng_id_e = self.hs['L[0][1][0][1][2][0]_eng_points_range_slider'].value[1]

        self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].max = self.xanes2D_img.shape[2]-100
        self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].min = 100
        self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].value = (100, self.xanes2D_img.shape[2]-100)

        self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].max = self.xanes2D_img.shape[1]-100
        self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].min = 100
        self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].value = (100, self.xanes2D_img.shape[1]-100)

        self.update_xanes2D_config()
        json.dump(self.xanes2D_config, open(self.xanes2D_file_save_trial_reg_config_filename, 'w'), cls=NumpyArrayEncoder)
        self.xanes2D_data_configured = True
        
        self.xanes_analysis_eng_list = self.reader(self.xanes2D_file_raw_h5_filename, 
                                            dtype='eng', sli=[[self.xanes2D_eng_id_s, self.xanes2D_eng_id_e+1]],
                                            cfg=self.global_h.io_xanes2D_cfg)
        if self.xanes_analysis_eng_list.max() < 70:
            self.xanes_analysis_eng_list *= 1000
        self.boxes_logic()

    def L0_1_1_0_1_0_2D_roi_x_slider_change(self, a):
        self.xanes2D_roi_configured = False
        if a['owner'].value[0] < 20:
            a['owner'].value[0] = 20
        if a['owner'].value[1] > (self.xanes2D_img.shape[2]-20):
            a['owner'].value[1] = self.xanes2D_img.shape[2]-20
        self.xanes2D_reg_roi[2] = a['owner'].value[0]
        self.xanes2D_reg_roi[3] = a['owner'].value[1]
        self.hs['L[0][1][1][0][2][0]_confirm_roi_text'].value = 'Please confirm after ROI is set'

        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_raw_img_viewer')
        if (not data_state) | (not viewer_state):
            fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_raw_img_viewer')
            self.hs['L[0][1][0][1][3][0]_fiji_raw_img_preview_checkbox'].value = True
        self.global_h.xanes2D_fiji_windows['xanes2D_raw_img_viewer']['ip'].setRoi(self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].value[0],
                                           self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].value[0],
                                           self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].value[1]-self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].value[0],
                                           self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].value[1]-self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].value[0])
        # self.boxes_logic()

    def L0_1_1_0_1_1_2D_roi_y_slider_change(self, a):
        self.xanes2D_roi_configured = False
        if a['owner'].value[0] < 20:
            a['owner'].value[0] = 20
        if a['owner'].value[1] > (self.xanes2D_img.shape[1]-20):
            a['owner'].value[1] = self.xanes2D_img.shape[1]-20
        self.xanes2D_reg_roi[0] = a['owner'].value[0]
        self.xanes2D_reg_roi[1] = a['owner'].value[1]
        self.hs['L[0][1][1][0][2][0]_confirm_roi_text'].value = 'Please confirm after ROI is set'

        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_raw_img_viewer')
        if (not data_state) | (not viewer_state):
            fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_raw_img_viewer')
            self.hs['L[0][1][0][1][3][0]_fiji_raw_img_preview_checkbox'].value = True
        self.global_h.xanes2D_fiji_windows['xanes2D_raw_img_viewer']['ip'].setRoi(self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].value[0],
                                           self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].value[0],
                                           self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].value[1]-self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].value[0],
                                           self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].value[1]-self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].value[0])
        # self.boxes_logic()

    def L0_1_1_0_2_1_confirm_roi_button_click(self, a):
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_raw_img_viewer')
        if viewer_state:
            fiji_viewer_off(self.global_h, self, viewer_name='xanes2D_raw_img_viewer')
        self.hs['L[0][1][0][1][3][0]_fiji_raw_img_preview_checkbox'].value = False

        if self.xanes2D_roi_configured:
            pass
        else:
            self.xanes2D_reg_roi[0] = self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].value[0]
            self.xanes2D_reg_roi[1] = self.hs['L[0][1][1][0][1][1]_2D_roi_y_slider'].value[1]
            self.xanes2D_reg_roi[2] = self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].value[0]
            self.xanes2D_reg_roi[3] = self.hs['L[0][1][1][0][1][0]_2D_roi_x_slider'].value[1]
            self.hs['L[0][1][1][1][1][2]_anchor_id_slider'].max = self.xanes2D_eng_id_e - self.xanes2D_eng_id_s
            self.hs['L[0][1][1][1][1][2]_anchor_id_slider'].value = 1
            self.hs['L[0][1][1][1][1][2]_anchor_id_slider'].min = 1
            self.xanes2D_reg_anchor_idx = self.xanes2D_eng_id_s
            self.hs['L[0][1][1][0][2][0]_confirm_roi_text'].value = 'ROI is set'
            del(self.xanes2D_img_roi)
            gc.collect()
            self.xanes2D_img_roi = self.xanes2D_img[self.xanes2D_eng_id_s:self.xanes2D_eng_id_e+1,
                                                    self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                    self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]]
            self.xanes2D_reg_mask = (self.xanes2D_img[0,
                                                      self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                      self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]] >
                                     0).astype(np.int8)
            self.update_xanes2D_config()
            json.dump(self.xanes2D_config, open(self.xanes2D_file_save_trial_reg_config_filename, 'w'), cls=NumpyArrayEncoder)
            self.xanes2D_roi_configured = True
        self.boxes_logic()

    def L0_1_1_1_1_0_fiji_mask_viewer_checkbox_change(self, a):
        if a['owner'].value:
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_mask_viewer')
            if not viewer_state:
                fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_mask_viewer')
        else:
            fiji_viewer_off(self.global_h, self, viewer_name='xanes2D_mask_viewer')
        self.boxes_logic()

    def L0_1_1_1_1_1_chunk_checkbox_change(self, a):
        self.xanes2D_reg_use_chunk = a['owner'].value
        self.xanes2D_reg_params_configured = False
        self.boxes_logic()

    def L0_1_1_1_1_2_anchor_id_slider_change(self, a):
        self.xanes2D_reg_anchor_idx = a['owner'].value+self.xanes2D_eng_id_s
        if not self.hs['L[0][1][1][1][1][0]_fiji_mask_viewer_checkbox'].value:
            self.hs['L[0][1][1][1][1][0]_fiji_mask_viewer_checkbox'].value = True
        else:
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_mask_viewer')
            if not viewer_state:
                fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_mask_viewer')
        self.global_h.xanes2D_fiji_windows['xanes2D_mask_viewer']['ip'].setSlice(self.xanes2D_reg_anchor_idx-self.xanes2D_eng_id_s)
        self.xanes2D_reg_params_configured = False
        self.xanes2D_regparams_anchor_idx_set = True
        # self.boxes_logic()

    def L0_1_1_1_2_0_use_mask_checkbox_change(self, a):
        self.xanes2D_reg_use_mask = a['owner'].value
        if self.xanes2D_reg_use_mask:
            self.xanes2D_reg_mask = (self.xanes2D_img[self.xanes2D_reg_anchor_idx,
                                                      self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                      self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]] >
                                     self.xanes2D_reg_use_mask).astype(np.int8)
        self.xanes2D_reg_params_configured = False
        self.boxes_logic()

    def L0_1_1_1_2_1_mask_thres_slider_change(self, a):
        self.xanes2D_reg_mask_thres = a['owner'].value
        self.hs['L[0][1][1][1][1][2]_anchor_id_slider'].disabled = True
        if self.xanes2D_reg_mask is None:
            self.xanes2D_reg_mask = (self.xanes2D_img[self.xanes2D_reg_anchor_idx,
                                                      self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                      self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]] >
                                      self.xanes2D_reg_use_mask).astype(np.int8)
        if not self.hs['L[0][1][1][1][1][0]_fiji_mask_viewer_checkbox'].value:
            self.hs['L[0][1][1][1][1][0]_fiji_mask_viewer_checkbox'].value = True
        else:
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_mask_viewer')
            if not viewer_state:
                fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_mask_viewer')
        if self.xanes2D_reg_use_smooth_img:
            if self.xanes2D_reg_mask_dilation_width > 0:
                self.xanes2D_reg_mask[:] = skm.binary_dilation(((gaussian_filter(self.xanes2D_img[self.xanes2D_reg_anchor_idx,
                                                                                                  self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                                                  self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]], self.xanes2D_reg_smooth_img_sigma) >
                                                                 self.xanes2D_reg_mask_thres).astype(np.int8)),
                                                               np.ones([self.xanes2D_reg_mask_dilation_width,
                                                                        self.xanes2D_reg_mask_dilation_width]))[:]
            else:
                self.xanes2D_reg_mask[:] = ((gaussian_filter(self.xanes2D_img[self.xanes2D_reg_anchor_idx,
                                                                              self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                              self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]], self.xanes2D_reg_smooth_img_sigma) >
                                             self.xanes2D_reg_mask_thres).astype(np.int8))[:]
            self.global_h.xanes2D_fiji_windows['xanes2D_mask_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
                self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(gaussian_filter(self.xanes2D_img[:,
                                                                                                               self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                                                               self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]],
                                                                                              self.xanes2D_reg_smooth_img_sigma)*self.xanes2D_reg_mask)),
                self.global_h.ImagePlusClass))
        else:
            if self.xanes2D_reg_mask_dilation_width > 0:
                self.xanes2D_reg_mask[:] = skm.binary_dilation(((self.xanes2D_img[self.xanes2D_reg_anchor_idx,
                                                                                  self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                                  self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]] >
                                                                 self.xanes2D_reg_mask_thres).astype(np.int8)),
                                                               np.ones([self.xanes2D_reg_mask_dilation_width,
                                                                        self.xanes2D_reg_mask_dilation_width]))[:]
            else:
                self.xanes2D_reg_mask[:] = ((self.xanes2D_img[self.xanes2D_reg_anchor_idx,
                                                              self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                              self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]] >
                                             self.xanes2D_reg_mask_thres).astype(np.int8))[:]
            self.global_h.xanes2D_fiji_windows['xanes2D_mask_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
                self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(self.xanes2D_img[:,
                                                                                               self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                                               self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]]*self.xanes2D_reg_mask)),
                self.global_h.ImagePlusClass))
        self.global_h.xanes2D_fiji_windows['xanes2D_mask_viewer']['ip'].setSlice(self.xanes2D_reg_anchor_idx)
        self.xanes2D_reg_params_configured = False

    def L0_1_1_1_2_2_mask_dilation_slider_change(self, a):
        self.xanes2D_reg_mask_dilation_width = a['owner'].value
        self.hs['L[0][1][1][1][1][2]_anchor_id_slider'].disabled = True
        if not self.hs['L[0][1][1][1][1][0]_fiji_mask_viewer_checkbox'].value:
            self.hs['L[0][1][1][1][1][0]_fiji_mask_viewer_checkbox'].value = True
        else:
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_mask_viewer')
            if not viewer_state:
                fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_mask_viewer')
        if self.xanes2D_reg_use_smooth_img:
            if self.xanes2D_reg_mask_dilation_width > 0:
                self.xanes2D_reg_mask[:] = skm.binary_dilation(((gaussian_filter(self.xanes2D_img[self.xanes2D_reg_anchor_idx,
                                                                                                  self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                                                  self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]], self.xanes2D_reg_smooth_img_sigma) >
                                                                 self.xanes2D_reg_mask_thres).astype(np.int8)),
                                                               np.ones([self.xanes2D_reg_mask_dilation_width,
                                                                        self.xanes2D_reg_mask_dilation_width]))[:]
            else:
                self.xanes2D_reg_mask[:] = ((gaussian_filter(self.xanes2D_img[self.xanes2D_reg_anchor_idx,
                                                                              self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                              self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]], self.xanes2D_reg_smooth_img_sigma) >
                                             self.xanes2D_reg_mask_thres).astype(np.int8))[:]
            self.global_h.xanes2D_fiji_windows['xanes2D_mask_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
                self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(gaussian_filter(self.xanes2D_img[:,
                                                                                                               self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                                                               self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]],
                                                                                              self.xanes2D_reg_smooth_img_sigma)*self.xanes2D_reg_mask)),
                self.global_h.ImagePlusClass))
        else:
            if self.xanes2D_reg_mask_dilation_width > 0:
                self.xanes2D_reg_mask[:] = skm.binary_dilation(((self.xanes2D_img[self.xanes2D_reg_anchor_idx,
                                                                                  self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                                  self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]] >
                                                                 self.xanes2D_reg_mask_thres).astype(np.int8)),
                                                               np.ones([self.xanes2D_reg_mask_dilation_width,
                                                                        self.xanes2D_reg_mask_dilation_width]))[:]
            else:
                self.xanes2D_reg_mask[:] = ((self.xanes2D_img[self.xanes2D_reg_anchor_idx,
                                                              self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                              self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]] >
                                             self.xanes2D_reg_mask_thres).astype(np.int8))[:]
            self.global_h.xanes2D_fiji_windows['xanes2D_mask_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
                self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(self.xanes2D_img[:,
                                                                                               self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                                               self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]]*
                                                                              self.xanes2D_reg_mask)),
                self.global_h.ImagePlusClass))
        self.global_h.xanes2D_fiji_windows['xanes2D_mask_viewer']['ip'].setSlice(self.xanes2D_reg_anchor_idx)
        self.xanes2D_reg_params_configured = False
        # self.boxes_logic()

    def L0_1_1_1_3_0_use_smooth_checkbox_change(self, a):
        self.xanes2D_reg_use_smooth_img = a['owner'].value
        self.xanes2D_reg_params_configured = False
        self.boxes_logic()

    def L0_1_1_1_3_1_smooth_sigma_text_change(self, a):
        self.xanes2D_reg_smooth_img_sigma = a['owner'].value
        if self.xanes2D_reg_use_mask:
            self.xanes2D_reg_mask = (self.xanes2D_img[self.xanes2D_reg_anchor_idx,
                                                      self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                      self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]] >
                                     self.xanes2D_reg_use_mask).astype(np.int8)
        if not self.hs['L[0][1][1][1][1][0]_fiji_mask_viewer_checkbox'].value:
            self.hs['L[0][1][1][1][1][0]_fiji_mask_viewer_checkbox'].value = True
        else:
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_mask_viewer')
            if not viewer_state:
                fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_mask_viewer')
        if self.xanes2D_reg_mask_dilation_width > 0:
            self.xanes2D_reg_mask[:] = skm.binary_dilation(((gaussian_filter(self.xanes2D_img[self.xanes2D_reg_anchor_idx,
                                                                                              self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                                              self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]], self.xanes2D_reg_smooth_img_sigma) >
                                                             self.xanes2D_reg_mask_thres).astype(np.int8)),
                                                           np.ones([self.xanes2D_reg_mask_dilation_width,
                                                                    self.xanes2D_reg_mask_dilation_width]))[:]
        else:
            self.xanes2D_reg_mask[:] = ((gaussian_filter(self.xanes2D_img[self.xanes2D_reg_anchor_idx,
                                                                          self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                          self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]], self.xanes2D_reg_smooth_img_sigma) >
                                         self.xanes2D_reg_mask_thres).astype(np.int8))[:]
        self.global_h.xanes2D_fiji_windows['xanes2D_mask_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
            self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(gaussian_filter(self.xanes2D_img[:,
                                                                                                           self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                                                           self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]],
                                                                                          self.xanes2D_reg_smooth_img_sigma)*self.xanes2D_reg_mask)),
            self.global_h.ImagePlusClass))
        self.global_h.xanes2D_fiji_windows['xanes2D_mask_viewer']['ip'].setSlice(self.xanes2D_reg_anchor_idx)
        self.xanes2D_reg_params_configured = False
        # self.boxes_logic()

    def L0_1_1_1_1_3_chunk_sz_slider_change(self, a):
        self.xanes2D_reg_chunk_sz = a['owner'].value
        self.xanes2D_reg_params_configured = False
        # self.boxes_logic()

    def L0_1_1_1_3_2_reg_method_dropdown_change(self, a):
        self.xanes2D_reg_method = a['owner'].value
        self.xanes2D_reg_params_configured = False
        self.boxes_logic()

    def L0_1_1_1_3_3_ref_mode_dropdown_change(self, a):
        self.xanes2D_reg_ref_mode = a['owner'].value
        self.xanes2D_reg_params_configured = False
        self.boxes_logic()
        
    def L0_1_1_1_4_0_mrtv_level_text_change(self, a):
        self.xanes2D_reg_mrtv_level = a['owner'].value
        self.xanes2D_reg_params_configured = False
        self.boxes_logic()
    
    def L0_1_1_1_4_1_mrtv_width_text_change(self, a):
        self.xanes2D_reg_mrtv_width = a['owner'].value
        self.xanes2D_reg_params_configured = False
        self.boxes_logic()
    
    def L0_1_1_1_4_2_mrtv_subpixel_wz_text_change(self, a):
        self.xanes2D_reg_mrtv_subpixel_wz = a['owner'].value
        self.xanes2D_reg_params_configured = False
        self.boxes_logic()
    
    def L0_1_1_1_4_3_mrtv_subpixel_step_text_change(self, a):
        self.xanes2D_reg_mrtv_subpixel_step = a['owner'].value
        self.xanes2D_reg_params_configured = False
        self.boxes_logic()

    def L0_1_1_1_5_1_confirm_reg_params_button_click(self, a):
        if self.xanes2D_reg_params_configured:
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_mask_viewer')
            if viewer_state:
                fiji_viewer_off(self.global_h, self, viewer_name='xanes2D_mask_viewer')
        self.hs['L[0][1][1][1][1][0]_fiji_mask_viewer_checkbox'].value = False

        self.xanes2D_reg_use_chunk = self.hs['L[0][1][1][1][1][1]_use_chunk_checkbox'].value
        self.xanes2D_reg_anchor_idx = self.hs['L[0][1][1][1][1][2]_anchor_id_slider'].value + self.xanes2D_eng_id_s
        self.xanes2D_reg_use_mask = self.hs['L[0][1][1][1][2][0]_use_mask_checkbox'].value
        self.xanes2D_reg_mask_thres = self.hs['L[0][1][1][1][2][1]_mask_thres_slider'].value
        self.xanes2D_reg_mask_dilation_width = self.hs['L[0][1][1][1][2][2]_mask_dilation_slider'].value
        self.xanes2D_reg_use_smooth_img = self.hs['L[0][1][1][1][3][0]_use_smooth_checkbox'].value
        self.xanes2D_reg_smooth_img_sigma = self.hs['L[0][1][1][1][3][1]_smooth_sigma_text'].value
        self.xanes2D_reg_chunk_sz = self.hs['L[0][1][1][1][1][3]_chunk_sz_slider'].value
        self.xanes2D_reg_method = self.hs['L[0][1][1][1][3][2]_reg_method_dropdown'].value
        self.xanes2D_reg_ref_mode = self.hs['L[0][1][1][1][3][3]_ref_mode_dropdown'].value        
        self.xanes2D_reg_mrtv_level = self.hs['L[0][1][1][1][4][0]_mrtv_level_text'].value
        self.xanes2D_reg_mrtv_width = self.hs['L[0][1][1][1][4][1]_mrtv_width_text'].value
        self.xanes2D_reg_mrtv_subpixel_step = self.hs['L[0][1][1][1][4][3]_mrtv_subpixel_step_text'].value
        self.xanes2D_reg_mrtv_subpixel_wz = self.hs['L[0][1][1][1][4][2]_mrtv_subpixel_wz_text'].value
        if self.xanes2D_reg_use_mask:
            if self.xanes2D_reg_use_smooth_img:
                self.xanes2D_img_roi[:] = gaussian_filter(self.xanes2D_img[self.xanes2D_eng_id_s:self.xanes2D_eng_id_e+1,
                                                                           self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                           self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]],
                                                          self.xanes2D_reg_smooth_img_sigma)[:]
                if self.xanes2D_reg_mask_dilation_width > 0:
                    self.xanes2D_reg_mask[:] = skm.binary_dilation(((gaussian_filter(self.xanes2D_img[self.xanes2D_reg_anchor_idx,
                                                                                                      self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                                                      self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]],
                                                                                     self.xanes2D_reg_smooth_img_sigma) >
                                                                     self.xanes2D_reg_mask_thres).astype(np.int8)),
                                                                   np.ones([self.xanes2D_reg_mask_dilation_width,
                                                                            self.xanes2D_reg_mask_dilation_width]))[:]
                else:
                    self.xanes2D_reg_mask[:] = ((gaussian_filter(self.xanes2D_img[self.xanes2D_reg_anchor_idx,
                                                                                  self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                                  self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]],
                                                                 self.xanes2D_reg_smooth_img_sigma) >
                                                 self.xanes2D_reg_mask_thres).astype(np.int8))[:]
            else:
                self.xanes2D_img_roi[:] = self.xanes2D_img[self.xanes2D_eng_id_s:self.xanes2D_eng_id_e+1,
                                                           self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                           self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]]
                if self.xanes2D_reg_mask_dilation_width > 0:
                    self.xanes2D_reg_mask[:] = skm.binary_dilation(((self.xanes2D_img[self.xanes2D_reg_anchor_idx,
                                                                                      self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                                      self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]]>
                                                                     self.xanes2D_reg_mask_thres).astype(np.int8)),
                                                                   np.ones([self.xanes2D_reg_mask_dilation_width,
                                                                            self.xanes2D_reg_mask_dilation_width]))[:]
                else:
                    self.xanes2D_reg_mask[:] = ((self.xanes2D_img[self.xanes2D_reg_anchor_idx,
                                                                  self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                  self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]] >
                                                 self.xanes2D_reg_mask_thres).astype(np.int8))[:]
        else:
            if self.xanes2D_reg_use_smooth_img:
                self.xanes2D_img_roi[:] = gaussian_filter(self.xanes2D_img[self.xanes2D_eng_id_s:self.xanes2D_eng_id_e+1,
                                                                           self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                                           self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]],
                                                          self.xanes2D_reg_smooth_img_sigma)[:]
            else:
                self.xanes2D_img_roi[:] = self.xanes2D_img[self.xanes2D_eng_id_s:self.xanes2D_eng_id_e+1,
                                                           self.xanes2D_reg_roi[0]:self.xanes2D_reg_roi[1],
                                                           self.xanes2D_reg_roi[2]:self.xanes2D_reg_roi[3]]
        self.update_xanes2D_config()
        json.dump(self.xanes2D_config, open(self.xanes2D_file_save_trial_reg_config_filename, 'w'), cls=NumpyArrayEncoder)
        self.xanes2D_reg_params_configured = True
        self.boxes_logic()

    def L0_1_1_2_1_0_run_reg_button_click(self, a):
        tmp_file = os.path.join(self.global_h.tmp_dir, 'xanes2D_tmp.h5')
        with h5py.File(tmp_file, 'w') as f:
            f.create_dataset('analysis_eng_list', data=self.xanes_analysis_eng_list.astype(np.float32))
            f.create_dataset('xanes2D_img', data=self.xanes2D_img.astype(np.float32))
            if self.xanes2D_reg_mask is not None:
                f.create_dataset('xanes2D_reg_mask', data=self.xanes2D_reg_mask.astype(np.float32))
            else:
                f.create_dataset('xanes2D_reg_mask', data=np.array([0]))
        code = {}
        ln = 0
        
        code[ln] = f"import os"; ln+=1
        code[ln] = f"from TXM_Sandbox.TXM_Sandbox.utils import xanes_regtools as xr"; ln+=1
        code[ln] = f"from multiprocessing import freeze_support"; ln+=1
        code[ln] = f"if __name__ == '__main__':"; ln+=1
        code[ln] = f"    freeze_support()"; ln+=1
        code[ln] = f"    reg = xr.regtools(dtype='2D_XANES', method='{self.xanes2D_reg_method}', mode='TRANSLATION')"; ln+=1
        code[ln] = f"    reg.set_xanes2D_raw_filename('{self.xanes2D_file_raw_h5_filename}')"; ln+=1
        kwargs = {'raw_h5_filename':self.xanes2D_file_raw_h5_filename, 'config_filename':self.xanes2D_file_save_trial_reg_config_filename}
        code[ln] = f"    reg.set_raw_data_info(**{kwargs})"; ln+=1
        code[ln] = f"    reg.set_method('{self.xanes2D_reg_method}')"; ln+=1
        code[ln] = f"    reg.set_ref_mode('{self.xanes2D_reg_ref_mode}')"; ln+=1
        code[ln] = f"    reg.set_roi({self.xanes2D_reg_roi})"; ln+=1
        code[ln] = f"    reg.set_indices({self.xanes2D_eng_id_s}, {self.xanes2D_eng_id_e+1}, {self.xanes2D_reg_anchor_idx})"; ln+=1
        code[ln] = f"    reg.set_xanes2D_tmp_filename('{tmp_file}')"; ln+=1
        code[ln] = f"    reg.read_xanes2D_tmp_file(mode='reg')"; ln+=1       
        code[ln] = f"    reg.set_reg_options(use_mask={self.xanes2D_reg_use_mask}, mask_thres={self.xanes2D_reg_mask_thres},\
                     use_chunk={self.xanes2D_reg_use_chunk}, chunk_sz={self.xanes2D_reg_chunk_sz},\
                     use_smooth_img={self.xanes2D_reg_use_smooth_img}, smooth_sigma={self.xanes2D_reg_smooth_img_sigma},\
                     mrtv_level={self.xanes2D_reg_mrtv_level}, mrtv_width={self.xanes2D_reg_mrtv_width}, \
                     mrtv_sp_wz={self.xanes2D_reg_mrtv_subpixel_wz}, mrtv_sp_step={self.xanes2D_reg_mrtv_subpixel_step})"; ln+=1
        code[ln] = f"    reg.set_saving(os.path.dirname('{self.xanes2D_save_trial_reg_filename}'), \
                     fn=os.path.basename('{self.xanes2D_save_trial_reg_filename}'))"; ln+=1
        code[ln] = f"    reg.compose_dicts()"; ln+=1
        code[ln] = f"    reg.reg_xanes2D_chunk()"; ln+=1
  
        gen_external_py_script(self.xanes2D_reg_external_command_name, code)
        sig = os.system(f'python {self.xanes2D_reg_external_command_name}')
 
        print(sig)
        if sig == 0:
            self.hs['L[0][1][1][2][1][1]_run_reg_text'].value = 'XANES2D registration is done'
            self.xanes2D_review_aligned_img_original = np.ndarray(self.xanes2D_img_roi[0].shape)
            self.xanes2D_review_aligned_img = np.ndarray(self.xanes2D_img_roi[0].shape)
            self.xanes2D_review_fixed_img = np.ndarray(self.xanes2D_img_roi[0].shape)
            self.xanes2D_review_shift_dict = {}
            with h5py.File(self.xanes2D_save_trial_reg_filename, 'r') as f:
                self.xanes2D_alignment_pairs = f['/trial_registration/trial_reg_parameters/alignment_pairs'][:]
                for ii in range(self.xanes2D_alignment_pairs.shape[0]):
                    self.xanes2D_review_shift_dict["{}".format(ii)] = f['/trial_registration/trial_reg_results/{0}/shift{0}'.format(str(ii).zfill(3))][:]
                    print(ii)
            self.hs['L[0][1][2][0][2][0]_reg_pair_slider'].min = 0
            self.hs['L[0][1][2][0][2][0]_reg_pair_slider'].max = self.xanes2D_alignment_pairs.shape[0]-1
    
            self.xanes2D_reg_done = True
            self.xanes2D_review_reg_best_match_filename = os.path.splitext(self.xanes2D_file_save_trial_reg_config_filename)[0].replace('config', 'reg_best_match') + '.json'
            self.update_xanes2D_config()
            json.dump(self.xanes2D_config, open(self.xanes2D_file_save_trial_reg_config_filename, 'w'), cls=NumpyArrayEncoder)
        else:
            self.hs['L[0][1][1][2][1][1]_run_reg_text'].value = 'Something went wrong during XANES2D registration'        
        self.boxes_logic()

    def L0_1_2_0_1_1_read_alignment_checkbox_change(self, a):
        self.xanes2D_use_existing_reg_reviewed = a['owner'].value
        self.xanes2D_reg_review_done = False
        self.boxes_logic()

    def L0_1_2_0_1_0_read_alignment_button_click(self, a):
        if len(a.files[0]) != 0:
            try:
                self.xanes2D_reg_review_file = os.path.abspath(a.files[0])
                if os.path.splitext(self.xanes2D_reg_review_file)[1] == '.json':
                    self.xanes2D_review_shift_dict = json.load(open(self.xanes2D_reg_review_file, 'r'))
                else:
                    self.xanes2D_review_shift_dict = np.float32(np.genfromtxt(self.xanes2D_reg_review_file))
                for ii in self.xanes2D_review_shift_dict:
                    self.xanes2D_review_shift_dict[ii] = np.float32(np.array(self.xanes2D_review_shift_dict[ii]))
                self.xanes2D_reg_file_readed = True
            except:
                self.xanes2D_reg_file_readed = False
        else:
            self.xanes2D_reg_file_readed = False
        self.xanes2D_reg_review_done = False
        self.boxes_logic()

    def L0_1_2_0_2_0_reg_pair_slider_change(self, a):
        self.xanes2D_review_alignment_pair_id = a['owner'].value
        with h5py.File(self.xanes2D_save_trial_reg_filename, 'r') as f:
            self.xanes2D_review_aligned_img_original[:] = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format(str(self.xanes2D_review_alignment_pair_id).zfill(3))][:]
            self.xanes2D_review_fixed_img[:] = f['/trial_registration/trial_reg_results/{0}/trial_reg_fixed{0}'.format(str(self.xanes2D_review_alignment_pair_id).zfill(3))][:]
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_review_viewer')
        if not viewer_state:
            fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_review_viewer')

        self.global_h.xanes2D_fiji_windows['xanes2D_review_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
            self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(self.xanes2D_review_aligned_img_original-
                                                                          self.xanes2D_review_fixed_img)),
            self.global_h.ImagePlusClass))
        self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        self.xanes2D_review_bad_shift = False
        # self.boxes_logic()

    def L0_1_2_0_2_1_reg_pair_bad_button_click(self, a):
        self.xanes2D_manual_xshift = 0
        self.xanes2D_manual_yshift = 0
        self.xanes2D_review_bad_shift = True
        self.boxes_logic()

    def L0_1_2_0_3_0_x_shift_text_change(self, a):
        self.xanes2D_manual_xshift = a['owner'].value
        self.xanes2D_review_aligned_img[:] = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(self.xanes2D_review_aligned_img_original),
                                                                                [self.xanes2D_manual_yshift, self.xanes2D_manual_xshift])))[:]
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_review_viewer')
        if not viewer_state:
            fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_review_viewer')

        self.global_h.xanes2D_fiji_windows['xanes2D_review_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
            self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(self.xanes2D_review_aligned_img-
                                                                          self.xanes2D_review_fixed_img)),
            self.global_h.ImagePlusClass))
        self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        # self.boxes_logic()

    def L0_1_2_0_3_1_y_shift_text_change(self, a):
        self.xanes2D_manual_yshift = a['owner'].value
        self.xanes2D_review_aligned_img[:] = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(self.xanes2D_review_aligned_img_original),
                                                                                [self.xanes2D_manual_yshift, self.xanes2D_manual_xshift])))[:]
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_review_viewer')
        if not viewer_state:
            fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_review_viewer')

        self.global_h.xanes2D_fiji_windows['xanes2D_review_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
            self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(self.xanes2D_review_aligned_img-
                                                                          self.xanes2D_review_fixed_img)),
            self.global_h.ImagePlusClass))
        self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        # self.boxes_logic()

    def L0_1_2_0_3_2_record_button_click(self, a):
        with h5py.File(self.xanes2D_save_trial_reg_filename, 'r') as f:
            shift = f['/trial_registration/trial_reg_results/{0}/shift{0}'.format(str(self.xanes2D_review_alignment_pair_id).zfill(3))][:]
        if self.xanes2D_reg_method.upper() == 'SR':
            shift[0, 2] += self.xanes2D_manual_yshift 
            shift[1, 2] += self.xanes2D_manual_xshift 
            self.xanes2D_review_shift_dict["{}".format(self.xanes2D_review_alignment_pair_id)] = np.array(shift)
        else:
            self.xanes2D_review_shift_dict["{}".format(self.xanes2D_review_alignment_pair_id)] = np.array([shift[0]+self.xanes2D_manual_yshift,
                                                                                                           shift[1]+self.xanes2D_manual_xshift])
        self.hs['L[0][1][2][0][3][0]_x_shift_text'].value = 0
        self.hs['L[0][1][2][0][3][1]_y_shift_text'].value = 0
        json.dump(self.xanes2D_review_shift_dict, open(self.xanes2D_review_reg_best_match_filename, 'w'), cls=NumpyArrayEncoder)
        self.xanes2D_review_bad_shift = False
        self.boxes_logic()

    def L0_1_2_0_4_1_confirm_review_results_button_click(self, a):
        if len(self.xanes2D_review_shift_dict) != (self.hs['L[0][1][2][0][2][0]_reg_pair_slider'].max+1):
            self.hs['L[0][1][2][0][4][0]_confirm_review_results_text'].value = 'reg review is not completed yet ...'
            idx = []
            offset = []
            for ii in sorted(self.xanes3D_review_shift_dict.keys()):
                offset.append(self.xanes3D_review_shift_dict[ii][0])
                idx.append(int(ii))
            plt.figure(1)
            plt.plot(idx, offset, 'b+')
            plt.xticks(np.arange(0, len(idx)+1, 5))
            plt.grid()
            self.xanes2D_reg_review_done = False
        else:
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_review_viewer')
            if viewer_state:
                fiji_viewer_off(self.global_h, self, viewer_name='xanes2D_review_viewer')
            self.xanes2D_reg_review_done = True
            self.update_xanes2D_config()
            json.dump(self.xanes2D_config, open(self.xanes2D_file_save_trial_reg_config_filename, 'w'), cls=NumpyArrayEncoder)
        self.boxes_logic()

    def L0_1_2_1_1_1_align_button_click(self, a):
        tmp_file = os.path.join(self.global_h.tmp_dir, 'xanes2D_tmp.h5')
        tmp_dict = {}
        for key in self.xanes2D_review_shift_dict.keys():
            tmp_dict[key] = tuple(self.xanes2D_review_shift_dict[key])
        code = {}
        ln = 0
        code[ln] = f"import TXM_Sandbox.TXM_Sandbox.utils.xanes_regtools as xr"; ln+=1
        code[ln] = f"reg = xr.regtools(dtype='2D_XANES', method='MPC', mode='TRANSLATION')"; ln+=1
        code[ln] = f"reg.set_roi({self.xanes2D_reg_roi})"; ln+=1
        code[ln] = f"reg.set_indices({self.xanes2D_eng_id_s}, {self.xanes2D_eng_id_e+1}, {self.xanes2D_reg_anchor_idx})"; ln+=1
        code[ln] = f"reg.set_xanes2D_tmp_filename('{tmp_file}')"; ln+=1
        code[ln] = f"reg.read_xanes2D_tmp_file(mode='align')"; ln+=1
        code[ln] = f"reg.apply_xanes2D_chunk_shift({tmp_dict}, \
                     trialfn='{self.xanes2D_save_trial_reg_filename}', \
                     savefn='{self.xanes2D_save_trial_reg_filename}')"; ln+=1
                
        gen_external_py_script(self.xanes2D_align_external_command_name, code)
        sig = os.system(f"python '{self.xanes2D_align_external_command_name}'")
        if sig == 0:
            self.hs['L[0][1][2][1][1][0]_align_text'].value = 'XANES2D alignment is done ...'
        else:
            self.hs['L[0][1][2][1][1][0]_align_text'].value = 'Something wrong during XANES2D alignment'

        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_analysis_viewer')
        if not viewer_state:
            fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_analysis_viewer')
        self.global_h.xanes2D_fiji_windows['xanes2D_analysis_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
            self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(self.xanes2D_img_roi)), self.global_h.ImagePlusClass))
        self.global_h.xanes2D_fiji_windows['xanes2D_analysis_viewer']['ip'].setSlice(0)
        self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")

        with h5py.File(self.xanes2D_save_trial_reg_filename, 'r') as f:
            self.xanes_analysis_data_shape = f['/registration_results/reg_results/registered_xanes2D'].shape
            self.xanes_analysis_eng_list = f['/registration_results/reg_results/eng_list'][:]
            if self.xanes2D_img_roi is None:
                self.xanes2D_img_roi = f['/registration_results/reg_results/registered_xanes2D'][:]
            else:
                self.xanes2D_img_roi = f['/registration_results/reg_results/registered_xanes2D'][:]

        self.hs['L[0][1][2][2][1][0]_visualization_slider'].min = 0
        self.hs['L[0][1][2][2][1][0]_visualization_slider'].max = self.xanes_analysis_eng_list.shape[0]-1

        self.xanes_element = determine_element(self.xanes_analysis_eng_list)
        tem = determine_fitting_energy_range(self.xanes_element)
        self.xanes_analysis_edge_eng = tem[0] 
        self.xanes_analysis_wl_fit_eng_s = tem[1]
        self.xanes_analysis_wl_fit_eng_e = tem[2]
        self.xanes_analysis_pre_edge_e = tem[3]
        self.xanes_analysis_post_edge_s = tem[4] 
        self.xanes_analysis_edge_0p5_fit_s = tem[5]
        self.xanes_analysis_edge_0p5_fit_e = tem[6]
        self.xanes_analysis_type = 'full'
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][0][0]_fit_eng_range_option_dropdown'].value = 'full'

        self.xanes2D_alignment_done = True
        self.update_xanes2D_config()
        json.dump(self.xanes2D_config, open(self.xanes2D_file_save_trial_reg_config_filename, 'w'), cls=NumpyArrayEncoder)
        self.boxes_logic()

    def L0_1_2_2_1_0_visualization_slider_change(self, a):
        self.hs['L[0][1][2][2][1][3]_visualization_eng_text'].value = self.xanes_analysis_eng_list[a['owner'].value]
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_analysis_viewer')
        if not viewer_state:
            fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_analysis_viewer')
        self.global_h.xanes2D_fiji_windows['xanes2D_analysis_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
            self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(self.xanes2D_img_roi)), self.global_h.ImagePlusClass))
        self.global_h.xanes2D_fiji_windows['xanes2D_analysis_viewer']['ip'].setSlice(a['owner'].value)
        if self.xanes2D_visualization_auto_bc:
            self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")

    def L0_1_2_2_1_2_visualization_RC_checkbox_change(self, a):
        self.xanes2D_visualization_auto_bc = a['owner'].value
        self.boxes_logic()

    def L0_1_2_2_1_1_spec_in_roi_button_click(self, a):
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes2D_analysis_viewer')
        if not viewer_state:
            fiji_viewer_on(self.global_h, self, viewer_name='xanes2D_analysis_viewer')
        width = self.global_h.xanes2D_fiji_windows['xanes2D_analysis_viewer']['ip'].width
        height = self.global_h.xanes2D_fiji_windows['xanes2D_analysis_viewer']['ip'].height
        roi = [int((width-10)/2), int((height-10)/2), 10, 10]
        self.global_h.xanes2D_fiji_windows['xanes2D_analysis_viewer']['ip'].setRoi(roi[0], roi[1], roi[2], roi[3])
        self.global_h.ij.py.run_macro("""run("Plot Z-axis Profile")""")
        self.global_h.ij.py.run_macro("""Plot.setStyle(0, "black,none,1.0,Connected Circles")""")
        self.global_h.xanes2D_fiji_windows['analysis_viewer_z_plot_viewer']['ip'] = self.global_h.WindowManager.getCurrentImage()
        self.global_h.xanes2D_fiji_windows['analysis_viewer_z_plot_viewer']['fiji_id'] = self.global_h.WindowManager.getIDList()[-1]
        self.boxes_logic()
