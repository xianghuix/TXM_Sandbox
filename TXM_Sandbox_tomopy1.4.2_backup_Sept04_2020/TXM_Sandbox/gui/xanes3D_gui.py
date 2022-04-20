#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:04:50 2020

@author: xiao
"""
import os, glob, h5py, json, time, numpy as np

import traitlets
from ipywidgets import widgets
import skimage.morphology as skm
import matplotlib.pyplot as plt
from scipy.ndimage import fourier_shift
import napari

from ..utils import xanes_regtools as xr
from ..utils.io import data_reader, xanes3D_h5_reader
from . import xanes_fitting_gui as xfg
from . import xanes_analysis_gui as xag
from .gui_components import (SelectFilesButton, NumpyArrayEncoder, 
                            get_handles, enable_disable_boxes, 
                            gen_external_py_script, fiji_viewer_state, 
                            fiji_viewer_on, fiji_viewer_off,
                            read_config_from_reg_file, restart,
                            determine_element, determine_fitting_energy_range)

napari.gui_qt()

class xanes3D_tools_gui():
    def __init__(self, parent_h, form_sz=[650, 740]):
        self.gui_name = 'xanes3D'
        self.form_sz = form_sz
        self.global_h = parent_h
        self.hs = {}
        
        if self.global_h.use_struc_h5_reader:
            self.reader = data_reader(xanes3D_h5_reader)
        else:
            pass
        
        self.xanes3D_raw_fn_temp = self.global_h.io_xanes3D_cfg['tomo_raw_fn_template']
        self.xanes3D_recon_dir_temp = self.global_h.io_xanes3D_cfg['xanes3D_recon_dir_template']
        self.xanes3D_recon_fn_temp = self.global_h.io_xanes3D_cfg['xanes3D_recon_fn_template']

        self.xanes_external_command_name = os.path.join(os.path.abspath(os.path.curdir), 'xanes3D_external_command.py')

        self.xanes3D_filepath_configured = False
        self.xanes3D_indices_configured = False
        self.xanes3D_roi_configured = False
        self.xanes3D_reg_params_configured = False
        self.xanes3D_reg_done = False
        self.xanes3D_reg_review_done = False
        self.xanes3D_alignment_done = False
        self.xanes3D_use_existing_reg_file = False
        self.xanes3D_use_existing_reg_reviewed = False
        self.xanes3D_reg_review_file = None
        self.xanes3D_reg_use_chunk = True
        self.xanes3D_reg_use_mask = True
        self.xanes3D_reg_use_smooth_img = False

        self.xanes3D_raw_h5_path_set = False
        self.xanes3D_recon_path_set = False
        self.xanes3D_save_trial_set = False
        self.xanes3D_scan_id_set = False
        self.xanes3D_reg_file_set = False
        self.xanes3D_config_file_set = False
        self.xanes3D_fixed_scan_id_set = False
        self.xanes3D_fixed_sli_id_set = False
        self.xanes3D_reg_file_readed = False
        self.xanes_analysis_eng_configured = False

        self.xanes3D_review_shift_dict = {}
        self.xanes3D_reg_mask_dilation_width = 0
        self.xanes3D_reg_mask_thres = 0
        self.xanes3D_img_roi = None
        self.xanes3D_roi = [0, 10, 0, 10, 0, 10]
        self.xanes3D_mask = None
        self.xanes3D_aligned_data = None
        self.xanes_analysis_slice = 0
        self.xanes3D_raw_fn_temp = self.global_h.io_xanes3D_cfg['tomo_raw_fn_template']
        self.xanes3D_raw_3D_h5_top_dir = None
        self.xanes3D_recon_3D_top_dir = None
        self.xanes3D_save_trial_reg_filename = None
        self.xanes3D_save_trial_reg_config_filename = None
        self.xanes3D_save_trial_reg_config_filename_original = None
        self.xanes3D_raw_3D_h5_temp = None
        self.xanes3D_available_raw_ids = None
        self.xanes3D_recon_3D_tiff_temp = None
        self.xanes3D_recon_3D_dir_temp = None
        self.xanes3D_reg_best_match_filename = None
        self.xanes3D_available_recon_ids = None
        self.xanes3D_available_sli_file_ids = None
        self.xanes_analysis_option = 'Do New Reg'
        self.xanes3D_fixed_scan_id = None
        self.xanes3D_scan_id_s = None
        self.xanes3D_scan_id_e = None
        self.xanes3D_fixed_sli_id = None
        self.xanes3D_reg_sli_search_half_width = None
        self.xanes3D_reg_chunk_sz = None
        self.xanes3D_reg_smooth_sigma = 0
        self.xanes3D_reg_method = None
        self.xanes3D_reg_ref_mode = None
        self.xanes3D_review_bad_shift = False
        self.xanes3D_visualization_viewer_option = 'fiji'
        self.xanes_analysis_view_option = 'x-y-E'
        self.xanes_element = None
        self.xanes_analysis_type = 'wl'
        self.xanes_analysis_edge_eng = None
        self.xanes_analysis_wl_fit_eng_s = None
        self.xanes_analysis_wl_fit_eng_e = None
        self.xanes_analysis_pre_edge_e = None
        self.xanes_analysis_post_edge_s = None
        self.xanes_analysis_edge_0p5_fit_s = None
        self.xanes_analysis_edge_0p5_fit_e = None
        self.xanes_analysis_spectrum = None
        self.xanes_analysis_use_mask = False
        self.xanes_analysis_mask_thres = None
        self.xanes_analysis_mask_scan_id = None
        self.xanes_analysis_mask = 1
        self.xanes_analysis_edge_jump_thres = 1.0
        self.xanes_analysis_edge_offset_thres =1.0
        self.xanes_analysis_use_flt_spec = False

        self.xanes3D_config = {"filepath config":{"xanes3D_raw_3D_h5_top_dir":self.xanes3D_raw_3D_h5_top_dir,
                                                  "xanes3D_recon_3D_top_dir":self.xanes3D_recon_3D_top_dir,
                                                  "xanes_save_trial_reg_filename":self.xanes3D_save_trial_reg_filename,
                                                  "xanes3D_save_trial_reg_config_filename":self.xanes3D_save_trial_reg_config_filename,
                                                  "xanes3D_save_trial_reg_config_filename_original":self.xanes3D_save_trial_reg_config_filename_original,
                                                  "xanes3D_raw_3D_h5_temp":self.xanes3D_raw_3D_h5_temp,
                                                  "xanes3D_recon_3D_tiff_temp":self.xanes3D_recon_3D_tiff_temp,
                                                  "xanes3D_recon_3D_dir_temp":self.xanes3D_recon_3D_dir_temp,
                                                  "xanes3D_reg_best_match_filename":self.xanes3D_reg_best_match_filename,
                                                  "xanes3D_analysis_option":self.xanes_analysis_option,
                                                  "xanes3D_filepath_configured":self.xanes3D_filepath_configured,
                                                  "xanes3D_raw_h5_path_set":self.xanes3D_raw_h5_path_set,
                                                  "xanes3D_recon_path_set":self.xanes3D_recon_path_set,
                                                  "xanes3D_save_trial_set":self.xanes3D_save_trial_set
                                                  },
		       "indeices config":{"select_scan_id_start_text_min":0,
                                  "select_scan_id_start_text_val":0,
                                  "select_scan_id_start_text_max":0,
                                  "select_scan_id_end_text_min":0,
                                  "select_scan_id_end_text_val":0,
                                  "select_scan_id_end_text_max":0,
                                  "fixed_scan_id_slider_min":0,
                                  "fixed_scan_id_slider_val":0,
                                  "fixed_scan_id_slider_max":0,
                                  "fixed_sli_id_slider_min":0,
                                  "fixed_sli_id_slider_val":0,
                                  "fixed_sli_id_slider_max":0,
                                  "xanes3D_fixed_scan_id":self.xanes3D_fixed_scan_id,
                                  "xanes3D_scan_id_s":self.xanes3D_scan_id_s,
                                  "xanes3D_scan_id_e":self.xanes3D_scan_id_e,
                                  "xanes3D_fixed_sli_id":self.xanes3D_fixed_sli_id,
                                  "xanes3D_scan_id_set":self.xanes3D_scan_id_set,
                                  "xanes3D_fixed_scan_id_set":self.xanes3D_fixed_scan_id_set,
                                  "xanes3D_fixed_sli_id_set":self.xanes3D_fixed_sli_id_set,
                                  "xanes3D_indices_configured":self.xanes3D_indices_configured
                                  },
		       "roi config":{"3D_roi_x_slider_min":0,
                             "3D_roi_x_slider_val":0,
                             "3D_roi_x_slider_max":0,
                             "3D_roi_y_slider_min":0,
                             "3D_roi_y_slider_val":0,
                             "3D_roi_y_slider_max":0,
                             "3D_roi_z_slider_min":0,
                             "3D_roi_z_slider_val":0,
                             "3D_roi_z_slider_max":0,
                             "3D_roi":list(self.xanes3D_roi),
                             "xanes3D_roi_configured":self.xanes3D_roi_configured
                             },
		       "registration config":{"xanes3D_reg_use_chunk":self.xanes3D_reg_use_chunk,
                                      "xanes3D_reg_use_mask":self.xanes3D_reg_use_mask,
                                      "xanes3D_reg_mask_thres":self.xanes3D_reg_mask_thres,
                                      "xanes3D_reg_mask_dilation_width":self.xanes3D_reg_mask_dilation_width,
                                      "xanes3D_reg_use_smooth_img":self.xanes3D_reg_use_smooth_img,
                                      "xanes3D_reg_smooth_sigma":self.xanes3D_reg_smooth_sigma,
                                      "xanes3D_reg_sli_search_half_width":self.xanes3D_reg_sli_search_half_width,
                                      "xanes3D_reg_chunk_sz":self.xanes3D_reg_chunk_sz,
                                      "xanes3D_reg_method":self.xanes3D_reg_method,
                                      "xanes3D_reg_ref_mode":self.xanes3D_reg_ref_mode,
                                      "mask_thres_slider_min":0,
                                      "mask_thres_slider_val":0,
                                      "mask_thres_slider_max":0,
                                      "mask_dilation_slider_min":0,
                                      "mask_dilation_slider_val":0,
                                      "mask_dilation_slider_max":0,
                                      "sli_search_slider_min":0,
                                      "sli_search_slider_val":0,
                                      "sli_search_slider_max":0,
                                      "chunk_sz_slider_min":0,
                                      "chunk_sz_slider_val":0,
                                      "chunk_sz_slider_max":0,
                                      "xanes3D_reg_params_configured":self.xanes3D_reg_params_configured
                                      },
		       "run registration":{"xanes3D_reg_done":self.xanes3D_reg_done
                                   },
		       "review registration":{"xanes3D_use_existing_reg_reviewed":self.xanes3D_use_existing_reg_reviewed,
                                      "xanes3D_reg_review_file":self.xanes3D_reg_review_file,
                                      "xanes3D_reg_review_done":self.xanes3D_reg_review_done,
                                      "read_alignment_checkbox":False,
                                      "reg_pair_slider_min":0,
                                      "reg_pair_slider_val":0,
                                      "reg_pair_slider_max":0,
                                      "zshift_slider_min":0,
                                      "zshift_slider_val":0,
                                      "zshift_slider_max":0,
                                      "best_match_text":0,
                                      "alignment_best_match":self.xanes3D_review_shift_dict
                                      },
               "align 3D recon":{"xanes3D_alignment_done":self.xanes3D_alignment_done,
                                 "xanes3D_analysis_edge_eng":self.xanes_analysis_edge_eng, 
                                 "xanes3D_analysis_wl_fit_eng_s":self.xanes_analysis_wl_fit_eng_s,
                                 "xanes3D_analysis_wl_fit_eng_e":self.xanes_analysis_wl_fit_eng_e,
                                 "xanes3D_analysis_pre_edge_e":self.xanes_analysis_pre_edge_e,
                                 "xanes3D_analysis_post_edge_s":self.xanes_analysis_post_edge_s, 
                                 "xanes3D_analysis_edge_0p5_fit_s":self.xanes_analysis_edge_0p5_fit_s,
                                 "xanes3D_analysis_edge_0p5_fit_e":self.xanes_analysis_edge_0p5_fit_e,
                                 "xanes3D_analysis_type":self.xanes_analysis_type
                                 }
		       }
        self.xanes_fitting_gui_h = xfg.xanes_fitting_gui(self, form_sz=self.form_sz)
        self.xanes_fitting_gui_h.build_gui()
        self.xanes_analysis_gui_h = xag.xanes_analysis_gui(self, form_sz=self.form_sz)
        self.xanes_analysis_gui_h.build_gui()

    def lock_message_text_boxes(self):
        boxes = ['L[0][2][0][0][1][1]_select_raw_h5_path_text',
                 'L[0][2][0][0][2][1]_select_recon_path_text',
                 'L[0][2][0][0][3][1]_select_save_trial_text',
                 'L[0][2][0][0][4][1]_confirm_file&path_text',
                 'L[0][2][0][1][4][1]_confirm_config_data_text',
                 'L[0][2][1][0][2][0]_confirm_roi_text',
                 'L[0][2][1][1][5][0]_confirm_reg_params_text',
                 'L[0][2][1][2][1][1]_run_reg_text',
                 'L[0][2][2][0][4][0]_confirm_review_results_text',
                 'L[0][2][2][1][2][0]_align_text',
                 'L[0][2][2][2][2][0]_visualize_spec_view_text',
                 'L[0][2][2][2][1][1]_visualize_alignment_eng_text',
                 'L[0][2][2][2][2][0]_visualize_spec_view_text']
        enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][5][0]_fit_run_text'].disabled=True




    def update_xanes3D_config(self):
        if self.xanes_analysis_option == 'Do New Reg':
            # self.xanes3D_save_trial_reg_config_filename = self.xanes3D_save_trial_reg_config_filename_template.format(self.xanes3D_scan_id_s, self.xanes3D_scan_id_e)
            pass
        self.xanes3D_config = {"filepath config":{"xanes3D_raw_3D_h5_top_dir":self.xanes3D_raw_3D_h5_top_dir,
                                                  "xanes3D_recon_3D_top_dir":self.xanes3D_recon_3D_top_dir,
                                                   "xanes_save_trial_reg_filename":self.xanes3D_save_trial_reg_filename,
                                                  "xanes3D_raw_3D_h5_temp":self.xanes3D_raw_3D_h5_temp,
                                                  "xanes3D_recon_3D_tiff_temp":self.xanes3D_recon_3D_tiff_temp,
                                                  "xanes3D_recon_3D_dir_temp":self.xanes3D_recon_3D_dir_temp,
                                                  "xanes3D_reg_best_match_filename":self.xanes3D_reg_best_match_filename,
                                                  "xanes3D_analysis_option":self.xanes_analysis_option,
                                                  "xanes3D_filepath_configured":self.xanes3D_filepath_configured,
                                                  "xanes3D_raw_h5_path_set":self.xanes3D_raw_h5_path_set,
                                                  "xanes3D_recon_path_set":self.xanes3D_recon_path_set,
                                                  "xanes3D_save_trial_set":self.xanes3D_save_trial_set
                                                  },
		       "indeices config":{"select_scan_id_start_text_min":self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].min,
                                  "select_scan_id_start_text_val":self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].value,
                                  "select_scan_id_start_text_max":self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].max,
                                  "select_scan_id_end_text_min":self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].min,
                                  "select_scan_id_end_text_val":self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].value,
                                  "select_scan_id_end_text_max":self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].max,
                                  "fixed_scan_id_slider_min":self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].min,
                                  "fixed_scan_id_slider_val":self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].value,
                                  "fixed_scan_id_slider_max":self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].max,
                                  "fixed_sli_id_slider_min":self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min,
                                  "fixed_sli_id_slider_val":self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].value,
                                  "fixed_sli_id_slider_max":self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].max,
                                  "xanes3D_fixed_scan_id":int(self.xanes3D_fixed_scan_id),
                                  "xanes3D_scan_id_s":int(self.xanes3D_scan_id_s),
                                  "xanes3D_scan_id_e":int(self.xanes3D_scan_id_e),
                                  "xanes3D_fixed_sli_id":int(self.xanes3D_fixed_sli_id),
                                  "xanes3D_scan_id_set":self.xanes3D_scan_id_set,
                                  "xanes3D_fixed_scan_id_set":self.xanes3D_fixed_scan_id_set,
                                  "xanes3D_fixed_sli_id_set":self.xanes3D_fixed_sli_id_set,
                                  "xanes3D_indices_configured":self.xanes3D_indices_configured
                                  },
		       "roi config":{"3D_roi_x_slider_min":self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].min,
                             "3D_roi_x_slider_val":self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value,
                             "3D_roi_x_slider_max":self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].max,
                             "3D_roi_y_slider_min":self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].min,
                             "3D_roi_y_slider_val":self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value,
                             "3D_roi_y_slider_max":self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].max,
                             "3D_roi_z_slider_min":self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].min,
                             "3D_roi_z_slider_val":self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].value,
                             "3D_roi_z_slider_max":self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].max,
                             "3D_roi":list(self.xanes3D_roi),
                             "xanes3D_roi_configured":self.xanes3D_roi_configured
                             },
		       "registration config":{"xanes3D_reg_use_chunk":self.xanes3D_reg_use_chunk,
                                      "xanes3D_reg_use_mask":self.xanes3D_reg_use_mask,
                                      "xanes3D_reg_mask_thres":self.xanes3D_reg_mask_thres,
                                      "xanes3D_reg_mask_dilation_width":self.xanes3D_reg_mask_dilation_width,
                                      "xanes3D_reg_use_smooth_img":self.xanes3D_reg_use_smooth_img,
                                      "xanes3D_reg_smooth_sigma":self.xanes3D_reg_smooth_sigma,
                                      "xanes3D_reg_sli_search_half_width":self.xanes3D_reg_sli_search_half_width,
                                      "xanes3D_reg_chunk_sz":self.xanes3D_reg_chunk_sz,
                                      "xanes3D_reg_method":self.xanes3D_reg_method,
                                      "xanes3D_reg_ref_mode":self.xanes3D_reg_ref_mode,
                                      "mask_thres_slider_min":self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].min,
                                      "mask_thres_slider_val":self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].value,
                                      "mask_thres_slider_max":self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].max,
                                      "mask_dilation_slider_min":self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].min,
                                      "mask_dilation_slider_val":self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].value,
                                      "mask_dilation_slider_max":self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].max,
                                      "sli_search_slider_min":self.hs['L[0][2][1][1][3][0]_sli_search_slider'].min,
                                      "sli_search_slider_val":self.hs['L[0][2][1][1][3][0]_sli_search_slider'].value,
                                      "sli_search_slider_max":self.hs['L[0][2][1][1][3][0]_sli_search_slider'].max,
                                      "chunk_sz_slider_min":self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].min,
                                      "chunk_sz_slider_val":self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].value,
                                      "chunk_sz_slider_max":self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].max,
                                      "xanes3D_reg_params_configured":self.xanes3D_reg_params_configured
                                      },
		       "run registration":{"xanes3D_reg_done":self.xanes3D_reg_done
                                   },
		       "review registration":{"xanes3D_use_existing_reg_reviewed":self.xanes3D_use_existing_reg_reviewed,
                                      "xanes3D_reg_review_file":self.xanes3D_reg_review_file,
                                      "xanes3D_reg_review_done":self.xanes3D_reg_review_done,
                                      "read_alignment_checkbox":self.hs['L[0][2][2][0][1][1]_read_alignment_checkbox'].value,
                                      "reg_pair_slider_min":self.hs['L[0][2][2][0][2][0]_reg_pair_slider'].min,
                                      "reg_pair_slider_val":self.hs['L[0][2][2][0][2][0]_reg_pair_slider'].value,
                                      "reg_pair_slider_max":self.hs['L[0][2][2][0][2][0]_reg_pair_slider'].max,
                                      "zshift_slider_min":self.hs['L[0][2][2][0][3][0]_zshift_slider'].min,
                                      "zshift_slider_val":self.hs['L[0][2][2][0][3][0]_zshift_slider'].value,
                                      "zshift_slider_max":self.hs['L[0][2][2][0][3][0]_zshift_slider'].max,
                                      # "best_match_text":self.hs['L[0][2][2][0][3][1]_best_match_text'].value,
                                      "alignment_best_match":self.xanes3D_review_shift_dict
                                      },
               "align 3D recon":{"xanes3D_alignment_done":self.xanes3D_alignment_done,
                                 "xanes3D_analysis_edge_eng":self.xanes_analysis_edge_eng, 
                                 "xanes3D_analysis_wl_fit_eng_s":self.xanes_analysis_wl_fit_eng_s,
                                 "xanes3D_analysis_wl_fit_eng_e":self.xanes_analysis_wl_fit_eng_e,
                                 "xanes3D_analysis_pre_edge_e":self.xanes_analysis_pre_edge_e,
                                 "xanes3D_analysis_post_edge_s":self.xanes_analysis_post_edge_s, 
                                 "xanes3D_analysis_edge_0p5_fit_s":self.xanes_analysis_edge_0p5_fit_s,
                                 "xanes3D_analysis_edge_0p5_fit_e":self.xanes_analysis_edge_0p5_fit_e,
                                 "xanes3D_analysis_type":self.xanes_analysis_type
                                 }
		       }

    def read_xanes3D_config(self):
        with open(self.xanes3D_save_trial_reg_config_filename_original, 'r') as f:
            self.xanes3D_config = json.load(f)

    def set_xanes3D_variables(self):
        self.xanes3D_raw_3D_h5_top_dir = self.xanes3D_config["filepath config"]["xanes3D_raw_3D_h5_top_dir"]
        self.xanes3D_recon_3D_top_dir = self.xanes3D_config["filepath config"]["xanes3D_recon_3D_top_dir"]
        self.xanes3D_save_trial_reg_filename = self.xanes3D_config["filepath config"]["xanes_save_trial_reg_filename"]
        self.xanes3D_raw_3D_h5_temp = self.xanes3D_config["filepath config"]["xanes3D_raw_3D_h5_temp"]
        self.xanes3D_recon_3D_tiff_temp = self.xanes3D_config["filepath config"]["xanes3D_recon_3D_tiff_temp"]
        self.xanes3D_recon_3D_dir_temp = self.xanes3D_config["filepath config"]["xanes3D_recon_3D_dir_temp"]
        self.xanes3D_reg_best_match_filename = self.xanes3D_config["filepath config"]["xanes3D_reg_best_match_filename"]
        self.xanes_analysis_option = self.xanes3D_config["filepath config"]["xanes3D_analysis_option"]
        self.xanes3D_filepath_configured = self.xanes3D_config["filepath config"]["xanes3D_filepath_configured"]
        self.xanes3D_raw_h5_path_set = self.xanes3D_config["filepath config"]["xanes3D_raw_h5_path_set"]
        self.xanes3D_recon_path_set = self.xanes3D_config["filepath config"]["xanes3D_recon_path_set"]
        self.xanes3D_save_trial_set = self.xanes3D_config["filepath config"]["xanes3D_save_trial_set"]

        b = glob.glob(os.path.join(self.xanes3D_raw_3D_h5_top_dir, 'fly*.h5'))
        self.xanes3D_available_raw_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])
        b = glob.glob(os.path.join(self.xanes3D_recon_3D_top_dir, 'recon_fly*'))
        self.xanes3D_available_recon_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])

        self.xanes3D_fixed_scan_id = self.xanes3D_config["indeices config"]["xanes3D_fixed_scan_id"]
        self.xanes3D_scan_id_s = self.xanes3D_config["indeices config"]["xanes3D_scan_id_s"]
        self.xanes3D_scan_id_e = self.xanes3D_config["indeices config"]["xanes3D_scan_id_e"]
        self.xanes3D_fixed_sli_id = self.xanes3D_config["indeices config"]["xanes3D_fixed_sli_id"]
        self.xanes3D_scan_id_set = self.xanes3D_config["indeices config"]["xanes3D_scan_id_set"]
        self.xanes3D_fixed_scan_id_set = self.xanes3D_config["indeices config"]["xanes3D_fixed_scan_id_set"]
        self.xanes3D_fixed_sli_id_set = self.xanes3D_config["indeices config"]["xanes3D_fixed_sli_id_set"]
        self.xanes3D_indices_configured = self.xanes3D_config["indeices config"]["xanes3D_indices_configured"]

        self.xanes3D_roi = self.xanes3D_config["roi config"]["3D_roi"]
        self.xanes3D_roi_configured = self.xanes3D_config["roi config"]["xanes3D_roi_configured"]

        self.xanes3D_reg_use_chunk = self.xanes3D_config["registration config"]["xanes3D_reg_use_chunk"]
        self.xanes3D_reg_use_mask = self.xanes3D_config["registration config"]["xanes3D_reg_use_mask"]
        self.xanes3D_reg_mask_thres = self.xanes3D_config["registration config"]["xanes3D_reg_mask_thres"]
        self.xanes3D_reg_mask_dilation_width = self.xanes3D_config["registration config"]["xanes3D_reg_mask_dilation_width"]
        self.xanes3D_reg_use_smooth_img = self.xanes3D_config["registration config"]["xanes3D_reg_use_smooth_img"]
        self.xanes3D_reg_smooth_sigma = self.xanes3D_config["registration config"]["xanes3D_reg_smooth_sigma"]
        self.xanes3D_reg_sli_search_half_width = self.xanes3D_config["registration config"]["xanes3D_reg_sli_search_half_width"]
        self.xanes3D_reg_chunk_sz = self.xanes3D_config["registration config"]["xanes3D_reg_chunk_sz"]
        self.xanes3D_reg_method = self.xanes3D_config["registration config"]["xanes3D_reg_method"]
        self.xanes3D_reg_ref_mode = self.xanes3D_config["registration config"]["xanes3D_reg_ref_mode"]
        self.xanes3D_reg_params_configured = self.xanes3D_config["registration config"]["xanes3D_reg_params_configured"]

        self.xanes3D_reg_done = self.xanes3D_config["run registration"]["xanes3D_reg_done"]

        self.xanes3D_use_existing_reg_reviewed = self.xanes3D_config["review registration"]["xanes3D_use_existing_reg_reviewed"]
        self.xanes3D_reg_review_file = self.xanes3D_config["review registration"]["xanes3D_reg_review_file"]
        self.xanes3D_reg_review_done = self.xanes3D_config["review registration"]["xanes3D_reg_review_done"]
        self.xanes3D_review_shift_dict = self.xanes3D_config["review registration"]["alignment_best_match"]

        self.xanes3D_alignment_done = self.xanes3D_config["align 3D recon"]["xanes3D_alignment_done"]
        self.xanes_analysis_edge_eng = self.xanes3D_config["align 3D recon"]["xanes3D_analysis_edge_eng"] 
        self.xanes_analysis_wl_fit_eng_s = self.xanes3D_config["align 3D recon"]["xanes3D_analysis_wl_fit_eng_s"]
        self.xanes_analysis_wl_fit_eng_e = self.xanes3D_config["align 3D recon"]["xanes3D_analysis_wl_fit_eng_e"]
        self.xanes_analysis_pre_edge_e = self.xanes3D_config["align 3D recon"]["xanes3D_analysis_pre_edge_e"]
        self.xanes_analysis_post_edge_s = self.xanes3D_config["align 3D recon"]["xanes3D_analysis_post_edge_s"]
        self.xanes_analysis_edge_0p5_fit_s = self.xanes3D_config["align 3D recon"]["xanes3D_analysis_edge_0p5_fit_s"]
        self.xanes_analysis_edge_0p5_fit_e = self.xanes3D_config["align 3D recon"]["xanes3D_analysis_edge_0p5_fit_e"]
        self.xanes_analysis_type = self.xanes3D_config["align 3D recon"]["xanes3D_analysis_type"]

        self.boxes_logic()

    def set_xanes3D_handles(self):
        self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].max = self.xanes3D_config["indeices config"]["select_scan_id_start_text_max"]
        self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].max = self.xanes3D_config["indeices config"]["select_scan_id_end_text_max"]
        self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].value = self.xanes3D_config["indeices config"]["select_scan_id_start_text_val"]
        self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].value = self.xanes3D_config["indeices config"]["select_scan_id_end_text_val"]
        self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].min = self.xanes3D_config["indeices config"]["select_scan_id_start_text_min"]
        self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].min = self.xanes3D_config["indeices config"]["select_scan_id_end_text_min"]

        self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].max = self.xanes3D_config["indeices config"]["fixed_scan_id_slider_max"]
        self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].min = self.xanes3D_config["indeices config"]["fixed_scan_id_slider_min"]
        self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].value = self.xanes3D_config["indeices config"]["fixed_scan_id_slider_val"]
        self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].max = self.xanes3D_config["indeices config"]["fixed_sli_id_slider_max"]
        self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min = self.xanes3D_config["indeices config"]["fixed_sli_id_slider_min"]
        self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].value = self.xanes3D_config["indeices config"]["fixed_sli_id_slider_val"]

        self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].max = self.xanes3D_config["roi config"]["3D_roi_x_slider_max"]
        self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].min = self.xanes3D_config["roi config"]["3D_roi_x_slider_min"]
        self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value = self.xanes3D_config["roi config"]["3D_roi_x_slider_val"]
        self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].max = self.xanes3D_config["roi config"]["3D_roi_y_slider_max"]
        self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].min = self.xanes3D_config["roi config"]["3D_roi_y_slider_min"]
        self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value = self.xanes3D_config["roi config"]["3D_roi_y_slider_val"]
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].max = self.xanes3D_config["roi config"]["3D_roi_z_slider_max"]
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].min = self.xanes3D_config["roi config"]["3D_roi_z_slider_min"]
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].value = self.xanes3D_config["roi config"]["3D_roi_z_slider_val"]

        self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].max = self.xanes3D_config["registration config"]["mask_thres_slider_max"]
        self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].min = self.xanes3D_config["registration config"]["mask_thres_slider_min"]
        self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].value = self.xanes3D_config["registration config"]["mask_thres_slider_val"]
        self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].max = self.xanes3D_config["registration config"]["mask_dilation_slider_max"]
        self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].min = self.xanes3D_config["registration config"]["mask_dilation_slider_min"]
        self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].value = self.xanes3D_config["registration config"]["mask_dilation_slider_val"]
        self.hs['L[0][2][1][1][3][0]_sli_search_slider'].max = self.xanes3D_config["registration config"]["sli_search_slider_max"]
        self.hs['L[0][2][1][1][3][0]_sli_search_slider'].min = self.xanes3D_config["registration config"]["sli_search_slider_min"]
        self.hs['L[0][2][1][1][3][0]_sli_search_slider'].value = self.xanes3D_config["registration config"]["sli_search_slider_val"]
        self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].max = self.xanes3D_config["registration config"]["chunk_sz_slider_max"]
        self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].min = self.xanes3D_config["registration config"]["chunk_sz_slider_min"]
        self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].value = self.xanes3D_config["registration config"]["chunk_sz_slider_val"]

        self.hs['L[0][2][2][0][1][1]_read_alignment_checkbox'].value = self.xanes3D_config["review registration"]["read_alignment_checkbox"]
        self.hs['L[0][2][2][0][2][0]_reg_pair_slider'].max = self.xanes3D_config["review registration"]["reg_pair_slider_max"]
        self.hs['L[0][2][2][0][2][0]_reg_pair_slider'].min = self.xanes3D_config["review registration"]["reg_pair_slider_min"]
        self.hs['L[0][2][2][0][2][0]_reg_pair_slider'].value = self.xanes3D_config["review registration"]["reg_pair_slider_val"]
        self.hs['L[0][2][2][0][3][0]_zshift_slider'].max = self.xanes3D_config["review registration"]["zshift_slider_max"]
        self.hs['L[0][2][2][0][3][0]_zshift_slider'].value = self.xanes3D_config["review registration"]["zshift_slider_val"]
        # self.hs['L[0][2][2][0][3][1]_best_match_text'].value = self.xanes3D_config["review registration"]["best_match_text"]
        
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][0][0]_fit_eng_range_option_dropdown'].value = \
            self.xanes3D_config["align 3D recon"]["xanes3D_analysis_type"]
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][0][1]_fit_eng_range_edge_eng_text'].value = \
            self.xanes3D_config["align 3D recon"]["xanes3D_analysis_edge_eng"] 
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][0][2]_fit_eng_range_pre_edge_e_text'].value = \
            self.xanes3D_config["align 3D recon"]["xanes3D_analysis_pre_edge_e"]
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][0][3]_fit_eng_range_post_edge_s_text'].value = \
            self.xanes3D_config["align 3D recon"]["xanes3D_analysis_post_edge_s"]
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][1][0]_fit_eng_range_wl_fit_s_text'].value = \
            self.xanes3D_config["align 3D recon"]["xanes3D_analysis_wl_fit_eng_s"]
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][1][1]_fit_eng_range_wl_fit_e_text'].value = \
            self.xanes3D_config["align 3D recon"]["xanes3D_analysis_wl_fit_eng_e"]
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][1][2]_fit_eng_range_edge0.5_fit_s_text'].value = \
            self.xanes3D_config["align 3D recon"]["xanes3D_analysis_edge_0p5_fit_s"]
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][1][3]_fit_eng_range_edge0.5_fit_e_text'].value = \
            self.xanes3D_config["align 3D recon"]["xanes3D_analysis_edge_0p5_fit_e"]

        self.boxes_logic()

    def boxes_logic(self):
        def xanes3D_compound_logic():
            if self.xanes3D_reg_done:
                if (self.hs['L[0][2][2][0][1][1]_read_alignment_checkbox'].value &
                    (not self.xanes3D_reg_file_readed)):
                    boxes = ['L[0][2][2][0][2]_reg_pair_box',
                             'L[0][2][2][0][3]_zshift_box',
                             'L[0][2][2][0][5]_review_manual_shift_box']
                    enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                    self.hs['L[0][2][2][0][1][0]_read_alignment_button'].disabled = False
                else:
                    self.hs['L[0][2][2][0][1][0]_read_alignment_button'].disabled = True
                    if self.xanes3D_review_bad_shift:
                        boxes = ['L[0][2][2][0][5]_review_manual_shift_box']
                        enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                        boxes = ['L[0][2][2][0][2]_reg_pair_box',
                                 'L[0][2][2][0][3]_zshift_box']
                        enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                    else:
                        boxes = ['L[0][2][2][0][2]_reg_pair_box',
                                 'L[0][2][2][0][3]_zshift_box']
                        enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                        boxes = ['L[0][2][2][0][5]_review_manual_shift_box']
                        enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            else:
                boxes = ['L[0][2][2][0]_review_reg_results_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            # if self.xanes_analysis_use_mask:
                # self.xanes_fitting_gui_h.hs['L[0][x][3][0][3][1]_analysis_image_mask_scan_id_slider'].disabled = False
                # self.xanes_fitting_gui_h.hs['L[0][x][3][0][3][1]_analysis_image_mask_thres_slider'].disabled = False
            # else:
                # self.xanes_fitting_gui_h.hs['L[0][x][3][0][3][1]_analysis_image_mask_scan_id_slider'].disabled = True
                # self.xanes_fitting_gui_h.hs['L[0][x][3][0][3][1]_analysis_image_mask_thres_slider'].disabled = True
            if self.xanes_analysis_eng_configured:
                self.xanes_fitting_gui_h.hs['L[0][x][3][0][5][1]_fit_run_button'].disabled = False
            else:
                self.xanes_fitting_gui_h.hs['L[0][x][3][0][5][1]_fit_run_button'].disabled = True
            if self.xanes3D_reg_review_done:
                if self.hs['L[0][2][2][1][1][0]_align_recon_optional_slice_checkbox'].value:
                    boxes = ['L[0][2][2][1][1][1]_align_recon_optional_slice_start_text',
                            'L[0][2][2][1][1][2]_align_recon_optional_slice_range_slider',
                            'L[0][2][2][1][1][3]_align_recon_optional_slice_end_text']
                    enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                else:
                    boxes = ['L[0][2][2][1][1][1]_align_recon_optional_slice_start_text',
                            'L[0][2][2][1][1][2]_align_recon_optional_slice_range_slider',
                            'L[0][2][2][1][1][3]_align_recon_optional_slice_end_text']
                    enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            else:
                boxes = ['L[0][2][2][1][1][1]_align_recon_optional_slice_start_text',
                        'L[0][2][2][1][1][2]_align_recon_optional_slice_range_slider',
                        'L[0][2][2][1][1][3]_align_recon_optional_slice_end_text']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            if self.xanes3D_alignment_done | (self.xanes_analysis_option == 'Do Analysis'):
                if self.xanes3D_visualization_viewer_option == 'fiji':
                    boxes = ['L[0][2][2][2][1]_visualize_view_alignment_box',
                             'L[0][2][2][2][2]_visualize_spec_view_box']
                    enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                elif self.xanes3D_visualization_viewer_option == 'napari':
                    boxes = ['L[0][2][2][2][1]_visualize_view_alignment_box',
                             'L[0][2][2][2][2]_visualize_spec_view_box']
                    enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)

        if self.xanes_analysis_option in ['Do New Reg', 'Read Config File']:
            if not self.xanes3D_filepath_configured:
                boxes = ['L[0][2][0][1]_config_data_box',
                         'L[0][2][1][0]_3D_roi_box',
                         'L[0][2][1][1]_config_reg_params_box',
                         'L[0][2][1][2]_run_reg_box',
                         'L[0][2][2][0]_review_reg_results_box',
                         'L[0][2][2][1]_align_recon_box',
                         'L[0][2][2][2]_visualize_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0]_analysis_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
                self.lock_message_text_boxes()
                xanes3D_compound_logic()

            elif (self.xanes3D_filepath_configured & (not self.xanes3D_indices_configured)):
                if not self.xanes3D_scan_id_set:
                    boxes = ['L[0][2][0][1]_config_data_box',
                             'L[0][2][1][0]_3D_roi_box',
                             'L[0][2][1][1]_config_reg_params_box',
                             'L[0][2][1][2]_run_reg_box',
                             'L[0][2][2][0]_review_reg_results_box',
                             'L[0][2][2][1]_align_recon_box',
                             'L[0][2][2][2]_visualize_box']
                    enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                    boxes = ['L[0][2][0][1][1]_scan_id_range_box']
                    enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                    boxes = ['L[0][x][3][0]_analysis_box']
                    enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                         boxes, disabled=True, level=-1)
                    self.lock_message_text_boxes()
                    xanes3D_compound_logic()
                else:
                    boxes = ['L[0][2][1][0]_3D_roi_box',
                             'L[0][2][1][1]_config_reg_params_box',
                             'L[0][2][1][2]_run_reg_box',
                             'L[0][2][2][0]_review_reg_results_box',
                             'L[0][2][2][1]_align_recon_box',
                             'L[0][2][2][2]_visualize_box']
                    enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                    boxes = ['L[0][2][0][1]_config_data_box']
                    enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                    boxes = ['L[0][x][3][0]_analysis_box']
                    enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                         boxes, disabled=True, level=-1)
                    self.lock_message_text_boxes()
                    xanes3D_compound_logic()
            elif ((self.xanes3D_filepath_configured & self.xanes3D_indices_configured) &
                  (not self.xanes3D_roi_configured)):
                boxes = ['L[0][2][0][1]_config_data_box',
                         'L[0][2][1][0]_3D_roi_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ['L[0][2][1][1]_config_reg_params_box',
                         'L[0][2][1][2]_run_reg_box',
                         'L[0][2][2][0]_review_reg_results_box',
                         'L[0][2][2][1]_align_recon_box',
                         'L[0][2][2][2]_visualize_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0]_analysis_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
                self.lock_message_text_boxes()
                xanes3D_compound_logic()
            elif ((self.xanes3D_filepath_configured & self.xanes3D_indices_configured) &
                  (self.xanes3D_roi_configured & (not self.xanes3D_reg_params_configured))):
                boxes = ['L[0][2][0][1]_config_data_box',
                         'L[0][2][1][0]_3D_roi_box',
                         'L[0][2][1][1]_config_reg_params_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ['L[0][2][1][2]_run_reg_box',
                         'L[0][2][2][0]_review_reg_results_box',
                         'L[0][2][2][1]_align_recon_box',
                         'L[0][2][2][2]_visualize_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0]_analysis_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
                data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_mask_viewer')
                if not viewer_state:
                    self.global_h.xanes3D_fiji_windows['xanes3D_mask_viewer']['fiji_id'] = None
                    self.global_h.xanes3D_fiji_windows['xanes3D_mask_viewer']['ip'] = None
                    boxes = ['L[0][2][1][1][2]_mask_options_box']
                    enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                else:
                    if self.xanes3D_reg_use_mask:
                        boxes = ['L[0][2][1][1][2][2]_mask_dilation_slider',
                                 'L[0][2][1][1][2][1]_mask_thres_slider']
                        enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                        self.hs['L[0][2][1][1][4][0]_reg_method_dropdown'].options = ['MPC', 'PC', 'SR']
                    else:
                        boxes = ['L[0][2][1][1][2][2]_mask_dilation_slider',
                                 'L[0][2][1][1][2][1]_mask_thres_slider']
                        enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                        self.hs['L[0][2][1][1][4][0]_reg_method_dropdown'].options = ['PC', 'SR']
                self.lock_message_text_boxes()
                xanes3D_compound_logic()
            elif ((self.xanes3D_filepath_configured & self.xanes3D_indices_configured) &
                  (self.xanes3D_roi_configured & self.xanes3D_reg_params_configured) &
                  (not self.xanes3D_reg_done)):
                boxes = ['L[0][2][0][1]_config_data_box',
                         'L[0][2][1][0]_3D_roi_box',
                         'L[0][2][1][1]_config_reg_params_box',
                         'L[0][2][1][2]_run_reg_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ['L[0][2][2][1]_align_recon_box',
                         'L[0][2][2][0]_review_reg_results_box',
                         'L[0][2][2][2]_visualize_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0]_analysis_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
                self.lock_message_text_boxes()
                xanes3D_compound_logic()
            elif ((self.xanes3D_filepath_configured & self.xanes3D_indices_configured) &
                  (self.xanes3D_roi_configured & self.xanes3D_reg_params_configured) &
                  (self.xanes3D_reg_done & (not self.xanes3D_reg_review_done))):
                boxes = ['L[0][2][0][1]_config_data_box',
                         'L[0][2][1][0]_3D_roi_box',
                         'L[0][2][1][1]_config_reg_params_box',
                         'L[0][2][1][2]_run_reg_box',
                         'L[0][2][2][0]_review_reg_results_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ['L[0][2][2][1]_align_recon_box',
                         'L[0][2][2][2]_visualize_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0]_analysis_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
                self.lock_message_text_boxes()
                xanes3D_compound_logic()
            elif ((self.xanes3D_filepath_configured & self.xanes3D_indices_configured) &
                  (self.xanes3D_roi_configured & self.xanes3D_reg_params_configured) &
                  (self.xanes3D_reg_done & self.xanes3D_reg_review_done) &
                  (not self.xanes3D_alignment_done)):
                boxes = ['L[0][2][0][1]_config_data_box',
                         'L[0][2][1][0]_3D_roi_box',
                         'L[0][2][1][1]_config_reg_params_box',
                         'L[0][2][1][2]_run_reg_box',
                         'L[0][2][2][0]_review_reg_results_box',
                         'L[0][2][2][1]_align_recon_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ['L[0][2][2][2]_visualize_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0]_analysis_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
                self.lock_message_text_boxes()
                xanes3D_compound_logic()
            elif ((self.xanes3D_filepath_configured & self.xanes3D_indices_configured) &
                  (self.xanes3D_roi_configured & self.xanes3D_reg_params_configured) &
                  (self.xanes3D_reg_done & self.xanes3D_reg_review_done) &
                  (self.xanes3D_alignment_done & (not self.xanes_analysis_eng_configured))):
                boxes = ['L[0][2][0][1]_config_data_box',
                         'L[0][2][1][0]_3D_roi_box',
                         'L[0][2][1][1]_config_reg_params_box',
                         'L[0][2][1][2]_run_reg_box',
                         'L[0][2][2][0]_review_reg_results_box',
                         'L[0][2][2][1]_align_recon_box',
                         'L[0][2][2][2]_visualize_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ['L[0][x][3][0][1]_analysis_energy_range_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                      boxes, disabled=False, level=-1)
                boxes = ['L[0][x][3][0][4]_analysis_tool_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
                self.lock_message_text_boxes()
                xanes3D_compound_logic()
            elif ((self.xanes3D_filepath_configured & self.xanes3D_indices_configured) &
                  (self.xanes3D_roi_configured & self.xanes3D_reg_params_configured) &
                  (self.xanes3D_reg_done & self.xanes3D_reg_review_done) &
                  (self.xanes3D_alignment_done & self.xanes_analysis_eng_configured)):
                boxes = ['L[0][2][0][1]_config_data_box',
                         'L[0][2][1][0]_3D_roi_box',
                         'L[0][2][1][1]_config_reg_params_box',
                         'L[0][2][1][2]_run_reg_box',
                         'L[0][2][2][0]_review_reg_results_box',
                         'L[0][2][2][1]_align_recon_box',
                         'L[0][2][2][2]_visualize_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ['L[0][x][3][0][1]_analysis_energy_range_box',
                          'L[0][x][3][0][4]_analysis_tool_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                      boxes, disabled=False, level=-1)
                self.lock_message_text_boxes()
                xanes3D_compound_logic()
        elif self.xanes_analysis_option == 'Read Reg File':
            if (self.xanes3D_filepath_configured & self.xanes3D_reg_done &
                (not self.xanes3D_reg_review_done)):
                boxes = ['L[0][2][0][1]_config_data_box',
                         'L[0][2][1][0]_3D_roi_box',
                         'L[0][2][1][1]_config_reg_params_box',
                         'L[0][2][1][2]_run_reg_box',
                         'L[0][2][2][0]_review_reg_results_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ['L[0][2][2][1]_align_recon_box',
                         'L[0][2][2][2]_visualize_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0][1]_analysis_energy_range_box',
                         'L[0][x][3][0][4]_analysis_tool_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
                self.lock_message_text_boxes()
                xanes3D_compound_logic()
            elif ((self.xanes3D_filepath_configured & self.xanes3D_reg_done) &
                  (self.xanes3D_reg_review_done & (not self.xanes3D_alignment_done))):
                boxes = ['L[0][2][0][1]_config_data_box',
                         'L[0][2][1][0]_3D_roi_box',
                         'L[0][2][1][1]_config_reg_params_box',
                         'L[0][2][1][2]_run_reg_box',
                         'L[0][2][2][0]_review_reg_results_box',
                         'L[0][2][2][1]_align_recon_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ['L[0][2][2][2]_visualize_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0][1]_analysis_energy_range_box',
                          'L[0][x][3][0][4]_analysis_tool_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
                self.lock_message_text_boxes()
                xanes3D_compound_logic()
            elif ((self.xanes3D_filepath_configured & self.xanes3D_reg_done) &
                  (self.xanes3D_reg_review_done & self.xanes3D_alignment_done)):
                boxes = ['L[0][2][0][1]_config_data_box',
                         'L[0][2][1][0]_3D_roi_box',
                         'L[0][2][1][1]_config_reg_params_box',
                         'L[0][2][1][2]_run_reg_box',
                         'L[0][2][2][0]_review_reg_results_box',
                         'L[0][2][2][1]_align_recon_box',
                         'L[0][2][2][2]_visualize_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                # boxes = ['L[0][x][3][0]_analysis_box']
                boxes = ['L[0][x][3][0][1]_analysis_energy_range_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=False, level=-1)
                boxes = ['L[0][x][3][0][4]_analysis_tool_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
                self.lock_message_text_boxes()
                xanes3D_compound_logic()
        elif self.xanes_analysis_option == 'Do Analysis':
            if self.xanes3D_reg_file_set & self.xanes3D_filepath_configured:
                boxes = ['L[0][2][0][1]_config_data_box',
                         'L[0][2][1][0]_3D_roi_box',
                         'L[0][2][1][1]_config_reg_params_box',
                         'L[0][2][1][2]_run_reg_box',
                         'L[0][2][2][0]_review_reg_results_box',
                         'L[0][2][2][1]_align_recon_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][2][2][2]_visualize_box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ['L[0][x][3][0][1]_analysis_energy_range_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=False, level=-1)
                boxes = ['L[0][x][3][0][4]_analysis_tool_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
                self.lock_message_text_boxes()
                xanes3D_compound_logic()
            else:
                boxes = ['L[0][2][0][1]_config_data_box',
                         'L[0][2][1][0]_3D_roi_box',
                         'L[0][2][1][1]_config_reg_params_box',
                         'L[0][2][1][2]_run_reg_box',
                         'L[0][2][2][0]_review_reg_results_box',
                         'L[0][2][2][1]_align_recon_box',
                         'L[0][2][2][2]_visualize_box']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['L[0][x][3][0][1]_analysis_energy_range_box',
                          'L[0][x][3][0][4]_analysis_tool_box']
                enable_disable_boxes(self.xanes_fitting_gui_h.hs, 
                                     boxes, disabled=True, level=-1)
                self.lock_message_text_boxes()
                xanes3D_compound_logic()
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please specify and confirm the aligned xanes3D file ...'
        self.hs['L[0][2][2][0][3][0]_zshift_slider'].disabled = True
        self.hs['L[0][2][2][0][3][1]_best_match_text'] = True

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
        #                                                     3D XANES                                                  #
        #                                                                                                               #
        #################################################################################################################
        ## ## ## define functional tabs for each sub-tab -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-86}px', 'height':f'{self.form_sz[0]-136}px'}
        self.hs['L[0][2][0]_config_input_form'] = widgets.VBox()
        self.hs['L[0][2][1]_reg_setting_form'] = widgets.VBox()
        self.hs['L[0][2][2]_reg&review_form'] = widgets.VBox()
        self.hs['L[0][2][3]_fitting_form'] = widgets.VBox()
        self.hs['L[0][2][4]_analysis_form'] = widgets.VBox()
        self.hs['L[0][2][0]_config_input_form'].layout = layout
        self.hs['L[0][2][1]_reg_setting_form'].layout = layout
        self.hs['L[0][2][2]_reg&review_form'].layout = layout
        self.hs['L[0][2][3]_fitting_form'].layout = layout
        self.hs['L[0][2][4]_analysis_form'].layout = layout



        ## ## ## ## define functional widget tabs in each sub-tab - configure file settings -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.32*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][0][0]_select_file&path_box'] = widgets.VBox()
        self.hs['L[0][2][0][0]_select_file&path_box'].layout = layout

        ## ## ## ## ## label configure file settings box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][0][0][0]_select_file&path_title_box'] = widgets.HBox()
        self.hs['L[0][2][0][0][0]_select_file&path_title_box'].layout = layout
        # self.hs['L[0][2][0][0][0][0]_select_file&path_title_box'] = widgets.Text(value='Config Files', disabled=True)
        self.hs['L[0][2][0][0][0][0]_select_file&path_title_box'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Config Dirs & Files' + '</span>')
        layout = {'background-color':'white', 'color':'cyan', 'left':'39%'}
        self.hs['L[0][2][0][0][0][0]_select_file&path_title_box'].layout = layout
        self.hs['L[0][2][0][0][0]_select_file&path_title_box'].children = get_handles(self.hs, 'L[0][2][0][0][0]_select_file&path_title_box', -1)

        ## ## ## ## ## raw h5 top directory
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][0][0][1]_select_raw_box'] = widgets.HBox()
        self.hs['L[0][2][0][0][1]_select_raw_box'].layout = layout
        self.hs['L[0][2][0][0][1][1]_select_raw_h5_path_text'] = widgets.Text(value='Choose raw h5 directory ...', description='', disabled=True)
        layout = {'width':'66%', 'display':'inline_flex'}
        self.hs['L[0][2][0][0][1][1]_select_raw_h5_path_text'].layout = layout
        self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'] = SelectFilesButton(option='askdirectory',
                                                                                    text_h=self.hs['L[0][2][0][0][1][1]_select_raw_h5_path_text'])
        layout = {'width':'15%'}
        self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].layout = layout
        self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].description = "Raw h5 Dir"
        self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].on_click(self.L0_2_0_0_1_0_select_raw_h5_path_button_click)
        self.hs['L[0][2][0][0][1]_select_raw_box'].children = get_handles(self.hs, 'L[0][2][0][0][1]_select_raw_box', -1)

        ## ## ## ## ## recon top directory
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][0][0][2]_select_recon_box'] = widgets.HBox()
        self.hs['L[0][2][0][0][2]_select_recon_box'].layout = layout
        self.hs['L[0][2][0][0][2][1]_select_recon_path_text'] = widgets.Text(value='Choose recon top directory ...', description='', disabled=True)
        layout = {'width':'66%', 'display':'inline_flex'}
        self.hs['L[0][2][0][0][2][1]_select_recon_path_text'].layout = layout
        self.hs['L[0][2][0][0][2][0]_select_recon_path_button'] = SelectFilesButton(option='askdirectory',
                                                                                    text_h=self.hs['L[0][2][0][0][2][1]_select_recon_path_text'])
        layout = {'width':'15%'}
        self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].layout = layout
        self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].description = "Recon Top Dir"
        self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].on_click(self.L0_2_0_0_2_0_select_recon_path_button_click)
        self.hs['L[0][2][0][0][2]_select_recon_box'].children = get_handles(self.hs, 'L[0][2][0][0][2]_select_recon_box', -1)

        ## ## ## ## ## trial save file
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][0][0][3]_select_save_trial_box'] = widgets.HBox()
        self.hs['L[0][2][0][0][3]_select_save_trial_box'].layout = layout
        self.hs['L[0][2][0][0][3][1]_select_save_trial_text'] = widgets.Text(value='Save trial registration as ...', description='', disabled=True)
        layout = {'width':'66%', 'display':'inline_flex'}
        self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].layout = layout
        self.hs['L[0][2][0][0][3][0]_select_save_trial_button'] = SelectFilesButton(option='asksaveasfilename',
                                                                                    text_h=self.hs['L[0][2][0][0][3][1]_select_save_trial_text'])
        self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].description = 'Save Reg File'
        layout = {'width':'15%'}
        self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].layout = layout
        self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].on_click(self.L0_2_0_0_3_0_select_save_trial_button_click)
        self.hs['L[0][2][0][0][3]_select_save_trial_box'].children = get_handles(self.hs, 'L[0][2][0][0][3]_select_save_trial_box', -1)

        ## ## ## ## ## confirm file configuration
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][0][0][4]_select_file&path_title_comfirm_box'] = widgets.HBox()
        self.hs['L[0][2][0][0][4]_select_file&path_title_comfirm_box'].layout = layout
        self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'] = widgets.Text(value='Save trial registration, or go directly review registration ...', description='', disabled=True)
        layout = {'width':'66%'}
        self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].layout = layout
        self.hs['L[0][2][0][0][4][0]_confirm_file&path_button'] = widgets.Button(description='Confirm',
                                                                                 tooltip='Confirm: Confirm after you finish file configuration')
        self.hs['L[0][2][0][0][4][0]_confirm_file&path_button'].style.button_color = 'darkviolet'
        self.hs['L[0][2][0][0][4][0]_confirm_file&path_button'].on_click(self.L0_2_0_0_4_0_confirm_file_path_button_click)
        layout = {'width':'15%'}
        self.hs['L[0][2][0][0][4][0]_confirm_file&path_button'].layout = layout

        self.hs['L[0][2][0][0][4][2]_file_path_options_dropdown'] = widgets.Dropdown(value='Do New Reg',
                                                                              options=['Do New Reg',
                                                                                       'Read Reg File',
                                                                                       'Read Config File',
                                                                                       'Do Analysis'],
                                                                              description='',
                                                                              description_tooltip='"Do New Reg": start registration and review results from beginning; "Read Reg File": if you have already done registraion and like to review the results; "Read Config File": if you like to resume analysis with an existing configuration.',
                                                                              disabled=False)
        layout = {'width':'15%', 'top':'0%'}
        self.hs['L[0][2][0][0][4][2]_file_path_options_dropdown'].layout = layout

        self.hs['L[0][2][0][0][4][2]_file_path_options_dropdown'].observe(self.L0_2_0_0_4_2_file_path_options_dropdown_change, names='value')
        self.hs['L[0][2][0][0][4]_select_file&path_title_comfirm_box'].children = get_handles(self.hs, 'L[0][2][0][0][4]_select_file&path_title_comfirm_box', -1)

        self.hs['L[0][2][0][0]_select_file&path_box'].children = get_handles(self.hs, 'L[0][2][0][0]_select_file&path_box', -1)
        ## ## ## ## bin widgets in hs['L[0][2][0][0]_select_file&path_box'] -- configure file settings -- end



        ## ## ## ## define functional widgets each tab in each sub-tab  - define indices -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.32*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][0][1]_config_data_box'] = widgets.VBox()
        self.hs['L[0][2][0][1]_config_data_box'].layout = layout
        ## ## ## ## ## label define indices box
        layout = {'justify-content':'center', 'align-items':'center','align-contents':'center','border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][0][1][0]_config_data_title_box'] = widgets.HBox()
        self.hs['L[0][2][0][1][0]_config_data_title_box'].layout = layout
        # self.hs['L[0][2][0][1][0][0]_config_data_title'] = widgets.Text(value='Config Files', disabled=True)
        self.hs['L[0][2][0][1][0][0]_config_data_title'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Config Scan & Slice Indices' + '</span>')
        layout = {'left':'35%', 'background-color':'white', 'color':'cyan'}
        self.hs['L[0][2][0][1][0][0]_config_data_title'].layout = layout
        self.hs['L[0][2][0][1][0]_config_data_title_box'].children = get_handles(self.hs, 'L[0][2][0][1][0]_select_file&path_title_box', -1)

        ## ## ## ## ## scan id range
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][0][1][1]_scan_id_range_box'] = widgets.HBox()
        self.hs['L[0][2][0][1][1]_scan_id_range_box'].layout = layout
        self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'] = widgets.BoundedIntText(value=0, description='scan_id start', disabled=True)
        layout = {'width':'40%'}
        self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].layout = layout
        self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'] = widgets.BoundedIntText(value=0, description='scan_id end', disabled=True)
        layout = {'width':'40%'}
        self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].layout = layout
        self.hs['L[0][2][0][1][1]_scan_id_range_box'].children = get_handles(self.hs, 'L[0][2][0][1][1]_scan_id_range_box', -1)
        self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].observe(self.L0_2_0_1_1_0_select_scan_id_start_text_change, names='value')
        self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].observe(self.L0_2_0_1_1_1_select_scan_id_end_text_change, names='value')

        ## ## ## ## ## fixed scan and slice ids
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][0][1][2]_fixed_id_box'] = widgets.HBox()
        self.hs['L[0][2][0][1][2]_fixed_id_box'].layout = layout
        self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'] = widgets.IntSlider(value=0, description='fixed scan id', disabled=True)
        layout = {'width':'40%'}
        self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].layout = layout
        self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'] = widgets.IntSlider(value=0, description='fixed sli id', disabled=True)
        layout = {'width':'40%'}
        self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].layout = layout
        self.hs['L[0][2][0][1][2]_fixed_id_box'].children = get_handles(self.hs, 'L[0][2][0][1][2]_fixed_id_box', -1)
        self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].observe(self.L0_2_0_1_2_0_fixed_scan_id_slider_change, names='value')
        self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].observe(self.L0_2_0_1_2_1_fixed_sli_id_slider_change, names='value')

        ## ## ## ## ## fiji option
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][0][1][3]_fiji_box'] = widgets.HBox()
        self.hs['L[0][2][0][1][3]_fiji_box'].layout = layout
        self.hs['L[0][2][0][1][3][0]_fiji_virtural_stack_preview_checkbox'] = widgets.Checkbox(value=False, description='fiji view', disabled=True)
        layout = {'width':'40%'}
        self.hs['L[0][2][0][1][3][0]_fiji_virtural_stack_preview_checkbox'].layout = layout
        self.hs['L[0][2][0][1][3][1]_fiji_close_button'] = widgets.Button(description='close all fiji viewers', disabled=True)
        layout = {'width':'40%'}
        self.hs['L[0][2][0][1][3][1]_fiji_close_button'].layout = layout
        self.hs['L[0][2][0][1][3]_fiji_box'].children = get_handles(self.hs, 'L[0][2][0][1][3]_fiji_box', -1)
        self.hs['L[0][2][0][1][3][0]_fiji_virtural_stack_preview_checkbox'].observe(self.L0_2_0_1_3_0_fiji_virtural_stack_preview_checkbox_change, names='value')
        self.hs['L[0][2][0][1][3][1]_fiji_close_button'].on_click(self.L0_2_0_1_3_1_fiji_close_button_click)

        ## ## ## ## ## confirm indices configuration
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][0][1][4]_config_data_confirm_box'] = widgets.HBox()
        self.hs['L[0][2][0][1][4]_config_data_confirm_box'].layout = layout
        self.hs['L[0][2][0][1][4][1]_confirm_config_data_text'] = widgets.Text(value='Confirm setting once you are done ...', description='', disabled=True)
        layout = {'width':'66%'}
        self.hs['L[0][2][0][1][4][1]_confirm_config_data_text'].layout = layout
        self.hs['L[0][2][0][1][4][0]_confirm_config_data_button'] = widgets.Button(description='Confirm',
                                                                                description_tooltip='Confirm: Confirm after you finish file configuration',
                                                                                disabled=True)
        self.hs['L[0][2][0][1][4][0]_confirm_config_data_button'].style.button_color = 'darkviolet'
        layout = {'width':'15%'}
        self.hs['L[0][2][0][1][4][0]_confirm_config_data_button'].layout = layout
        self.hs['L[0][2][0][1][4]_config_data_confirm_box'].children = get_handles(self.hs, 'L[0][2][0][1][4]_config_roi_confirm_box', -1)
        self.hs['L[0][2][0][1][4][0]_confirm_config_data_button'].on_click(self.L0_2_0_1_4_0_confirm_config_data_button_click)

        self.hs['L[0][2][0][1]_config_data_box'].children = get_handles(self.hs, 'L[0][2][0][1]_config_data_box', -1)
        ## ## ## ## bin widgets in hs['L[0][2][0][0]_select_file&path_box']  - define indices -- end

        self.hs['L[0][2][0]_config_input_form'].children = get_handles(self.hs, 'L[0][2][0]_config_input_form', -1)
        ## ## ## bin boxes in hs['L[0][2][0]_config_input_form'] -- end



        ## ## ## ## define functional widgets each tab in each sub-tab - config roi -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.35*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][1][0]_3D_roi_box'] = widgets.VBox()
        self.hs['L[0][2][1][0]_3D_roi_box'].layout = layout
        ## ## ## ## ## label 3D_roi_title box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][1][0][0]_3D_roi_title_box'] = widgets.HBox()
        self.hs['L[0][2][1][0][0]_3D_roi_title_box'].layout = layout
        self.hs['L[0][2][1][0][0][0]_3D_roi_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Config 3D ROI' + '</span>')
        # self.hs['L[0][2][1][0][0][0]_3D_roi_title_text'] = widgets.Text(value='Config 3D ROI', disabled=True)
        layout = {'justify-content':'center', 'background-color':'white', 'color':'cyan', 'left':'43%', 'height':'90%'}
        self.hs['L[0][2][1][0][0][0]_3D_roi_title_text'].layout = layout
        self.hs['L[0][2][1][0][0]_3D_roi_title_box'].children = get_handles(self.hs, 'L[0][2][1][0][0]_3D_roi_title_box', -1)

        ## ## ## ## ## define roi
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.21*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][1][0][1]_3D_roi_define_box'] = widgets.VBox()
        self.hs['L[0][2][1][0][1]_3D_roi_define_box'].layout = layout
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-108}px', 'height':f'{0.21*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'] = widgets.IntRangeSlider(value=[10, 40],
                                                                                min=1,
                                                                                max=50,
                                                                                step=1,
                                                                                description='x range:',
                                                                                disabled=True,
                                                                                continuous_update=False,
                                                                                orientation='horizontal',
                                                                                readout=True,
                                                                                readout_format='d')
        self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].layout = layout
        self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'] = widgets.IntRangeSlider(value=[10, 40],
                                                                                min=1,
                                                                                max=50,
                                                                                step=1,
                                                                                description='y range:',
                                                                                disabled=True,
                                                                                continuous_update=False,
                                                                                orientation='horizontal',
                                                                                readout=True,
                                                                                readout_format='d')
        self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].layout = layout
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'] = widgets.IntRangeSlider(value=[10, 40],
                                                                                min=1,
                                                                                max=50,
                                                                                step=1,
                                                                                description='z range:',
                                                                                disabled=True,
                                                                                continuous_update=False,
                                                                                orientation='horizontal',
                                                                                readout=True,
                                                                                readout_format='d')
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].layout = layout
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].add_traits(mylower=traitlets.traitlets.Any(self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].lower))
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].add_traits(myupper=traitlets.traitlets.Any(self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].upper))
        self.hs['L[0][2][1][0][1]_3D_roi_define_box'].children = get_handles(self.hs, 'L[0][2][1][0][1]_3D_roi_define_box', -1)
        self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].observe(self.L0_2_1_0_1_0_3D_roi_x_slider_change, names='value')
        self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].observe(self.L0_2_1_0_1_1_3D_roi_y_slider_change, names='value')
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].observe(self.L0_2_1_0_1_2_3D_roi_z_slider_change, names='value')
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].observe(self.L0_2_1_0_1_2_3D_roi_z_slider_lower_change, names='mylower')
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].observe(self.L0_2_1_0_1_2_3D_roi_z_slider_upper_change, names='myupper')

        ## ## ## ## ## confirm roi
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][1][0][2]_3D_roi_confirm_box'] = widgets.HBox()
        self.hs['L[0][2][1][0][2]_3D_roi_confirm_box'].layout = layout
        layout = {'width':'70%', 'height':'90%'}
        self.hs['L[0][2][1][0][2][0]_confirm_roi_text'] = widgets.Text(description='',
                                                                   value='Please confirm after ROI is set ...',
                                                                   disabled=True)
        self.hs['L[0][2][1][0][2][0]_confirm_roi_text'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][2][1][0][2][1]_confirm_roi_button'] = widgets.Button(description='Confirm',
                                                                       description_tooltip='Confirm the roi once you define the ROI ...',
                                                                       disabled=True)
        self.hs['L[0][2][1][0][2][1]_confirm_roi_button'].style.button_color = 'darkviolet'
        self.hs['L[0][2][1][0][2][1]_confirm_roi_button'].layout = layout
        self.hs['L[0][2][1][0][2]_3D_roi_confirm_box'].children = get_handles(self.hs, 'L[0][2][1][0][2]_3D_roi_confirm_box', -1)
        self.hs['L[0][2][1][0][2][1]_confirm_roi_button'].on_click(self.L0_2_1_0_2_1_confirm_roi_button_click)

        self.hs['L[0][2][1][0]_3D_roi_box'].children = get_handles(self.hs, 'L[0][2][1][0]_3D_roi_box', -1)
        ## ## ## ## define functional widgets each tab in each sub-tab - config roi -- end

        ## ## ## ## define functional widgets each tab in each sub-tab - config registration -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.42*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][1][1]_config_reg_params_box'] = widgets.VBox()
        self.hs['L[0][2][1][1]_config_reg_params_box'].layout = layout
        ## ## ## ## ## label config_reg_params box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][1][1][0]_config_reg_params_title_box'] = widgets.HBox()
        self.hs['L[0][2][1][1][0]_config_reg_params_title_box'].layout = layout
        self.hs['L[0][2][1][1][0][0]_config_reg_params_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Config Reg Params' + '</span>')
        # self.hs['L[0][2][1][1][0][0]_config_reg_params_title_text'] = widgets.Text(value='Config Reg Params', disabled=True)
        layout = {'background-color':'white', 'color':'cyan', 'left':'40.5%', 'height':'90%'}
        self.hs['L[0][2][1][1][0][0]_config_reg_params_title_text'].layout = layout
        self.hs['L[0][2][1][1][0]_config_reg_params_title_box'].children = get_handles(self.hs, 'L[0][2][1][1][0]_config_reg_params_title_box', -1)

        ## ## ## ## ## fiji&anchor box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][1][1][1]_fiji&anchor_box'] = widgets.HBox()
        self.hs['L[0][2][1][1][1]_fiji&anchor_box'].layout = layout
        self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'] = widgets.Checkbox(value=False,
                                                                        disabled=True,
                                                                        description='preview mask in fiji')
        layout = {'width':'40%', 'height':'90%'}
        self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].layout = layout
        self.hs['L[0][2][1][1][1][1]_chunk_checkbox'] = widgets.Checkbox(value=True,
                                                                          disabled=True,
                                                                          description='use chunk')
        layout = {'width':'40%', 'height':'90%'}
        self.hs['L[0][2][1][1][1][1]_chunk_checkbox'].layout = layout
        self.hs['L[0][2][1][1][1]_fiji&anchor_box'].children = get_handles(self.hs, 'L[0][2][1][1][1]_fiji&anchor_box', -1)
        self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].observe(self.L0_2_1_1_1_0_fiji_mask_viewer_checkbox_change, names='value')
        self.hs['L[0][2][1][1][1][1]_chunk_checkbox'].observe(self.L0_2_1_1_1_1_chunk_checkbox_change, names='value')

        ## ## ## ## ## mask options box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][1][1][2]_mask_options_box'] = widgets.HBox()
        self.hs['L[0][2][1][1][2]_mask_options_box'].layout = layout
        self.hs['L[0][2][1][1][2][0]_use_mask_checkbox'] = widgets.Checkbox(value=True,
                                                                        disabled=True,
                                                                        description='use mask',
                                                                        display='flex',)
        layout = {'width':'15%', 'flex-direction':'row', 'height':'90%'}
        self.hs['L[0][2][1][1][2][0]_use_mask_checkbox'].layout = layout
        self.hs['L[0][2][1][1][2][1]_mask_thres_slider'] = widgets.FloatSlider(value=False,
                                                                          disabled=True,
                                                                          description='mask thres',
                                                                          readout_format='.5f',
                                                                          min=-1.,
                                                                          max=1.,
                                                                          step=1e-5)
        layout = {'width':'40%', 'left':'2.5%', 'height':'90%'}
        self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].layout = layout
        self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'] = widgets.IntSlider(value=False,
                                                                          disabled=True,
                                                                          description='mask dilation',
                                                                          min=0,
                                                                          max=30,
                                                                          step=1)
        layout = {'width':'40%', 'left':'2.5%', 'height':'90%'}
        self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].layout = layout
        self.hs['L[0][2][1][1][2]_mask_options_box'].children = get_handles(self.hs, 'L[0][2][1][1][2]_mask_options_box', -1)
        self.hs['L[0][2][1][1][2][0]_use_mask_checkbox'].observe(self.L0_2_1_1_2_0_use_mask_checkbox_change, names='value')
        self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].observe(self.L0_2_1_1_2_1_mask_thres_slider_change, names='value')
        self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].observe(self.L0_2_1_1_2_2_mask_dilation_slider_change, names='value')

        ## ## ## ## ## sli_search & chunk_size box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][1][1][3]_sli_search&chunk_size_box'] = widgets.HBox()
        self.hs['L[0][2][1][1][3]_sli_search&chunk_size_box'].layout = layout
        self.hs['L[0][2][1][1][3][0]_sli_search_slider'] = widgets.IntSlider(value=10,
                                                                             disabled=True,
                                                                             description='z search half width')
        layout = {'width':'40%', 'height':'90%'}
        self.hs['L[0][2][1][1][3][0]_sli_search_slider'].layout = layout
        self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'] = widgets.IntSlider(value=7,
                                                                           disabled=True,
                                                                           description='chunk size')
        layout = {'width':'40%', 'height':'90%'}
        self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].layout = layout
        self.hs['L[0][2][1][1][3]_sli_search&chunk_size_box'].children = get_handles(self.hs, 'L[0][2][1][1][3]_sli_search&chunk_size_box', -1)
        self.hs['L[0][2][1][1][3][0]_sli_search_slider'].observe(self.L0_2_1_1_3_0_sli_search_slider_change, names='value')
        self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].observe(self.L0_2_1_1_3_1_chunk_sz_slider_change, names='value')

        ## ## ## ## ##  reg_options box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][1][1][4]_reg_options_box'] = widgets.HBox()
        self.hs['L[0][2][1][1][4]_reg_options_box'].layout = layout
        self.hs['L[0][2][1][1][4][0]_reg_method_dropdown'] = widgets.Dropdown(value='MPC',
                                                                              options=['MPC', 'PC', 'SR'],
                                                                              description='reg method',
                                                                              description_tooltip='reg method: MPC: Masked Phase Correlation; PC: Phase Correlation; SR: StackReg',
                                                                              disabled=True)
        layout = {'width':'40%', 'height':'90%'}
        self.hs['L[0][2][1][1][4][0]_reg_method_dropdown'].layout = layout
        self.hs['L[0][2][1][1][4][1]_ref_mode_dropdown'] = widgets.Dropdown(value='single',
                                                                            options=['single', 'neighbor', 'average'],
                                                                            description='ref mode',
                                                                            description_tooltip='ref mode: single: two images with fixed id gap in two consecutive chunks are used for the registration between two chunks; neighbor: two neighbor images in two consecutive chunks are used for the registration between two chunks; average: deprecated',
                                                                            disabled=True)
        layout = {'width':'40%', 'height':'90%'}
        self.hs['L[0][2][1][1][4][1]_ref_mode_dropdown'].layout = layout
        self.hs['L[0][2][1][1][4]_reg_options_box'].children = get_handles(self.hs, 'L[0][2][1][1][4]_reg_options_box', -1)
        self.hs['L[0][2][1][1][4][0]_reg_method_dropdown'].observe(self.L0_2_1_1_4_0_reg_method_dropdown_change, names='value')
        self.hs['L[0][2][1][1][4][1]_ref_mode_dropdown'].observe(self.L0_2_1_1_4_1_ref_mode_dropdown_change, names='value')

        ## ## ## ## ## confirm reg settings -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][1][1][5]_config_reg_params_confirm_box'] = widgets.HBox()
        self.hs['L[0][2][1][1][5]_config_reg_params_confirm_box'].layout = layout
        layout = {'width':'70%', 'height':'90%'}
        self.hs['L[0][2][1][1][5][0]_confirm_reg_params_text'] = widgets.Text(description='',
                                                                   value='Confirm the roi once you define the ROI ...',
                                                                   disabled=True)
        self.hs['L[0][2][1][1][5][0]_confirm_reg_params_text'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][2][1][1][5][1]_confirm_reg_params_button'] = widgets.Button(description='Confirm',
                                                                       description_tooltip='Confirm the roi once you define the ROI ...',
                                                                       disabled=True)
        self.hs['L[0][2][1][1][5][1]_confirm_reg_params_button'].style.button_color = 'darkviolet'
        self.hs['L[0][2][1][1][5][1]_confirm_reg_params_button'].layout = layout
        self.hs['L[0][2][1][1][5]_config_reg_params_confirm_box'].children = get_handles(self.hs, 'L[0][2][1][1][5]_config_reg_params_confirm_box', -1)
        self.hs['L[0][2][1][1][5][1]_confirm_reg_params_button'].on_click(self.L0_2_1_1_5_1_confirm_reg_params_button_click)
        ## ## ## ## ## confirm reg settings -- end        
        self.hs['L[0][2][1][1]_config_reg_params_box'].children = get_handles(self.hs, 'L[0][2][1][1]_config_reg_params_box', -1)
        ## ## ## ## define functional widgets each tab in each sub-tab - config registration -- end

        ## ## ## ## define functional widgets each tab in each sub-tab - register in register/review/shift TAB -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.21*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][1][2]_run_reg_box'] = widgets.VBox()
        self.hs['L[0][2][1][2]_run_reg_box'].layout = layout
        ## ## ## ## ## label run_reg box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][1][2][0]_run_reg_title_box'] = widgets.HBox()
        self.hs['L[0][2][1][2][0]_run_reg_title_box'].layout = layout
        self.hs['L[0][2][1][2][0][0]_run_reg_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Run Registration' + '</span>')
        # self.hs['L[0][2][1][2][0][0]_run_reg_title_text'] = widgets.Text(value='Run Registration', disabled=True)
        layout = {'background-color':'white', 'color':'cyan', 'left':'41%'}
        self.hs['L[0][2][1][2][0][0]_run_reg_title_text'].layout = layout
        self.hs['L[0][2][1][2][0]_run_reg_title_box'].children = get_handles(self.hs, 'L[0][2][1][2][0]_run_reg_title_box', -1)

        ## ## ## ## ## run reg & status
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][1][2][1]_run_reg_confirm_box'] = widgets.HBox()
        self.hs['L[0][2][1][2][1]_run_reg_confirm_box'].layout = layout
        layout = {'width':'70%', 'height':'90%'}
        self.hs['L[0][2][1][2][1][1]_run_reg_text'] = widgets.Text(description='',
                                                                   value='run registration once you are ready ...',
                                                                   disabled=True)
        self.hs['L[0][2][1][2][1][1]_run_reg_text'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][2][1][2][1][0]_run_reg_button'] = widgets.Button(description='Run Reg',
                                                                       description_tooltip='run registration once you are ready ...',
                                                                       disabled=True)
        self.hs['L[0][2][1][2][1][0]_run_reg_button'].style.button_color = 'darkviolet'
        self.hs['L[0][2][1][2][1][0]_run_reg_button'].layout = layout
        self.hs['L[0][2][1][2][1]_run_reg_confirm_box'].children = get_handles(self.hs, 'L[0][2][1][2][1]_run_reg_confirm_box', -1)
        self.hs['L[0][2][1][2][1][0]_run_reg_button'].on_click(self.L0_2_1_2_1_0_run_reg_button_click)
        ## ## ## ## ## run reg progress
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][1][2][2]_run_reg_progress_box'] = widgets.HBox()
        self.hs['L[0][2][1][2][2]_run_reg_progress_box'].layout = layout
        layout = {'width':'100%', 'height':'90%'}
        self.hs['L[0][2][1][2][2][0]_run_reg_progress_bar'] = widgets.IntProgress(value=0,
                                                                                  min=0,
                                                                                  max=10,
                                                                                  step=1,
                                                                                  description='Completing:',
                                                                                  bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                                                                  orientation='horizontal')
        self.hs['L[0][2][1][2][2][0]_run_reg_progress_bar'].layout = layout
        self.hs['L[0][2][1][2][2]_run_reg_progress_box'].children = get_handles(self.hs, 'L[0][2][1][2][2]_run_reg_progress_box', -1)

        self.hs['L[0][2][1][2]_run_reg_box'].children = get_handles(self.hs, 'L[0][2][1][2]_run_reg_box', -1)
        ## ## ## ## define functional widgets each tab in each sub-tab - register in register/review/shift TAB -- end
        
        self.hs['L[0][2][1]_reg_setting_form'].children = get_handles(self.hs, 'L[0][2][1]_reg_setting_form', -1)
        ## ## ## bin boxes in hs['L[0][2][1]_reg_setting_form'] -- end
        
        
        ## ## ## ## define functional widgets each tab in each sub-tab - review in in register/review/shift TAB -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.35*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][2][0]_review_reg_results_box'] = widgets.VBox()
        self.hs['L[0][2][2][0]_review_reg_results_box'].layout = layout
        ## ## ## ## ## label the box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][2][0][0]_review_reg_results_title_box'] = widgets.HBox()
        self.hs['L[0][2][2][0][0]_review_reg_results_title_box'].layout = layout
        self.hs['L[0][2][2][0][0][0]_review_reg_results_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Review Registration Results' + '</span>')
        # self.hs['L[0][2][2][0][0][0]_review_reg_results_title_text'] = widgets.Text(value='Review Registration Results', disabled=True)
        layout = {'background-color':'white', 'color':'cyan', 'left':'35.7%', 'height':'90%'}
        self.hs['L[0][2][2][0][0][0]_review_reg_results_title_text'].layout = layout
        self.hs['L[0][2][2][0][0]_review_reg_results_title_box'].children = get_handles(self.hs, 'L[0][2][2][0][0]_review_reg_results_title_box', -1)

        ## ## ## ## ## read alignment file
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][2][0][1]_read_alignment_file_box'] = widgets.HBox()
        self.hs['L[0][2][2][0][1]_read_alignment_file_box'].layout = layout
        layout = {'width':'70%', 'height':'90%'}
        self.hs['L[0][2][2][0][1][1]_read_alignment_checkbox'] = widgets.Checkbox(description='read alignment',
                                                                                  value=False,
                                                                                  disabled=True)
        self.hs['L[0][2][2][0][1][1]_read_alignment_checkbox'].layout = layout
        layout = {'width':'30%', 'height':'90%'}
        self.hs['L[0][2][2][0][1][0]_read_alignment_button'] = SelectFilesButton(option='askopenfilename')
        self.hs['L[0][2][2][0][1][0]_read_alignment_button'].layout = layout
        self.hs['L[0][2][2][0][1][0]_read_alignment_button'].disabled = True
        self.hs['L[0][2][2][0][1]_read_alignment_file_box'].children = get_handles(self.hs, 'L[0][2][2][0][1]_read_alignment_file_box', -1)
        self.hs['L[0][2][2][0][1][1]_read_alignment_checkbox'].observe(self.L0_2_2_0_1_0_read_alignment_checkbox_change, names='value')
        self.hs['L[0][2][2][0][1][0]_read_alignment_button'].on_click(self.L0_2_2_0_1_1_read_alignment_button_click)

        ## ## ## ## ## reg pair box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][2][0][2]_reg_pair_box'] = widgets.HBox()
        self.hs['L[0][2][2][0][2]_reg_pair_box'].layout = layout
        layout = {'width':'70%', 'height':'90%'}
        self.hs['L[0][2][2][0][2][0]_reg_pair_slider'] = widgets.IntSlider(value=False,
                                                                           disabled=True,
                                                                           description='reg pair #')
        self.hs['L[0][2][2][0][2][0]_reg_pair_slider'].layout = layout
        self.hs['L[0][2][2][0][2]_reg_pair_box'].children = get_handles(self.hs, 'L[0][2][2][0][2]_reg_pair_box', -1)
        self.hs['L[0][2][2][0][2][0]_reg_pair_slider'].observe(self.L0_2_2_0_2_0_reg_pair_slider_change, names='value')

        ## ## ## ## ## zshift box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][2][0][3]_zshift_box'] = widgets.HBox()
        self.hs['L[0][2][2][0][3]_zshift_box'].layout = layout
        layout = {'width':'36%', 'height':'90%'}
        self.hs['L[0][2][2][0][3][0]_zshift_slider'] = widgets.IntSlider(value=False,
                                                                         disabled=True,
                                                                         min = 1,
                                                                         description='z shift')
        self.hs['L[0][2][2][0][3][0]_zshift_slider'].layout = layout
        layout = {'width':'36%', 'height':'90%'}
        self.hs['L[0][2][2][0][3][1]_best_match_text'] = widgets.IntText(value=0,
                                                                      disabled=True,
                                                                      description='Best Match')
        self.hs['L[0][2][2][0][3][1]_best_match_text'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][2][2][0][3][2]_record_button'] = widgets.Button(description='Record',
                                                                       description_tooltip='Record',
                                                                       disabled=True)
        self.hs['L[0][2][2][0][3][2]_record_button'].layout = layout
        self.hs['L[0][2][2][0][3][2]_record_button'].style.button_color = 'darkviolet'
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][2][2][0][3][3]_reg_pair_bad_button'] = widgets.Button(description='Bad',
                                                                       description_tooltip='Bad reg',
                                                                       disabled=True)
        self.hs['L[0][2][2][0][3][3]_reg_pair_bad_button'].layout = layout
        self.hs['L[0][2][2][0][3][3]_reg_pair_bad_button'].style.button_color = 'darkviolet'
        self.hs['L[0][2][2][0][3]_zshift_box'].children = get_handles(self.hs, 'L[0][2][2][0][3]_zshift_box', -1)
        self.hs['L[0][2][2][0][3][0]_zshift_slider'].observe(self.L0_2_2_0_3_0_zshift_slider_change, names='value')
        self.hs['L[0][2][2][0][3][1]_best_match_text'].observe(self.L0_2_2_0_3_1_best_match_text_change, names='value')
        self.hs['L[0][2][2][0][3][2]_record_button'].on_click(self.L0_2_2_0_3_2_record_button_click)
        self.hs['L[0][2][2][0][3][3]_reg_pair_bad_button'].on_click(self.L0_2_2_0_3_3_reg_pair_bad_button)
        # widgets.jslink((self.hs['L[0][2][2][0][3][0]_zshift_slider'], 'value'),
        #                 (self.hs['L[0][2][2][0][3][1]_best_match_text'], 'value'))

        ## ## ## ## ## manual shift box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][2][0][5]_review_manual_shift_box'] = widgets.HBox()
        self.hs['L[0][2][2][0][5]_review_manual_shift_box'].layout = layout
        layout = {'width':'28%', 'height':'90%'}
        self.hs['L[0][2][2][0][5][0]_review_manual_x_shift_text'] = widgets.FloatText(value=0,
                                                                                        disabled=True,
                                                                                        min = -100,
                                                                                        max = 100,
                                                                                        step = 0.5,
                                                                                        description='x shift')
        self.hs['L[0][2][2][0][5][0]_review_manual_x_shift_text'].layout = layout
        layout = {'width':'28%', 'height':'90%'}
        self.hs['L[0][2][2][0][5][1]_review_manual_y_shift_text'] = widgets.FloatText(value=0,
                                                                                        disabled=True,
                                                                                        min = -100,
                                                                                        max = 100,
                                                                                        step = 0.5,
                                                                                        description='y shift')
        self.hs['L[0][2][2][0][5][1]_review_manual_y_shift_text'].layout = layout
        layout = {'width':'28%', 'height':'90%'}
        self.hs['L[0][2][2][0][5][2]_review_manual_z_shift_text'] = widgets.IntText(value=0,
                                                                                      disabled=True,
                                                                                      min = 1,
                                                                                      max = 100,
                                                                                      step = 1,
                                                                                      description='z shift')
        self.hs['L[0][2][2][0][5][2]_review_manual_z_shift_text'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][2][2][0][5][3]_review_manual_shift_record_button'] = widgets.Button(disabled=True,
                                                                                          description='Record')
        self.hs['L[0][2][2][0][5][3]_review_manual_shift_record_button'].layout = layout
        self.hs['L[0][2][2][0][5][3]_review_manual_shift_record_button'].style.button_color = 'darkviolet'
        self.hs['L[0][2][2][0][5]_review_manual_shift_box'].children = get_handles(self.hs, 'L[0][2][2][0][5]_review_manual_shift_box', -1)
        self.hs['L[0][2][2][0][5][0]_review_manual_x_shift_text'].observe(self.L0_2_2_0_5_0_review_manual_x_shift_text, names='value')
        self.hs['L[0][2][2][0][5][1]_review_manual_y_shift_text'].observe(self.L0_2_2_0_5_1_review_manual_y_shift_text, names='value')
        self.hs['L[0][2][2][0][5][2]_review_manual_z_shift_text'].observe(self.L0_2_2_0_5_2_review_manual_z_shift_text, names='value')
        self.hs['L[0][2][2][0][5][3]_review_manual_shift_record_button'].on_click(self.L0_2_2_0_5_3_review_manual_shift_record_button)
        ## ## ## ## ## manual shift box -- end

        ## ## ## ## ## confirm review results box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][2][0][4]_review_reg_results_comfirm_box'] = widgets.HBox()
        self.hs['L[0][2][2][0][4]_review_reg_results_comfirm_box'].layout = layout
        layout = {'width':'70%', 'height':'90%', 'display':'inline_flex'}
        self.hs['L[0][2][2][0][4][0]_confirm_review_results_text'] = widgets.Text(description='',
                                                                   value='Confirm after you finish reg review ...',
                                                                   disabled=True)
        self.hs['L[0][2][2][0][4][0]_confirm_review_results_text'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][2][2][0][4][1]_confirm_review_results_button'] = widgets.Button(description='Confirm',
                                                                       description_tooltip='Confirm after you finish reg review ...',
                                                                       disabled=True)
        self.hs['L[0][2][2][0][4][1]_confirm_review_results_button'].style.button_color = 'darkviolet'
        self.hs['L[0][2][2][0][4][1]_confirm_review_results_button'].layout = layout
        self.hs['L[0][2][2][0][4]_review_reg_results_comfirm_box'].children = get_handles(self.hs, 'L[0][2][2][0][4]_review_reg_results_comfirm_box', -1)
        self.hs['L[0][2][2][0][4][1]_confirm_review_results_button'].on_click(self.L0_2_2_0_4_1_confirm_review_results_button_click)

        self.hs['L[0][2][2][0]_review_reg_results_box'].children = get_handles(self.hs, 'L[0][2][2][0]_review_reg_results_box', -1)
        ## ## ## ## define functional widgets each tab in each sub-tab - review in in register/review/shift TAB-- end

        ## ## ## ## define functional widgets each tab in each sub-tab - align recon in register/review/shift TAB -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.28*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][2][1]_align_recon_box'] = widgets.VBox()
        self.hs['L[0][2][2][1]_align_recon_box'].layout = layout
        ## ## ## ## ## label the box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][2][1][0]_align_recon_title_box'] = widgets.HBox()
        self.hs['L[0][2][2][1][0]_align_recon_title_box'].layout = layout
        self.hs['L[0][2][2][1][0][0]_align_recon_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Align 3D Recon' + '</span>')
        # self.hs['L[0][2][2][1][0][0]_align_recon_title_text'] = widgets.Text(value='Align 3D Recon', disabled=True)
        layout = {'background-color':'white', 'color':'cyan', 'left':'41%'}
        self.hs['L[0][2][2][1][0][0]_align_recon_title_text'].layout = layout
        self.hs['L[0][2][2][1][0]_align_recon_title_box'].children = get_handles(self.hs, 'L[0][2][2][1][0]_align_recon_title_box', -1)

        ## ## ## ## ## define slice region if it is necessary
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][2][1][1]_align_recon_optional_slice_region_box'] = widgets.HBox()
        self.hs['L[0][2][2][1][1]_align_recon_optional_slice_region_box'].layout = layout
        layout = {'width':'20%', 'height':'90%'}
        self.hs['L[0][2][2][1][1][0]_align_recon_optional_slice_checkbox'] = widgets.Checkbox(description='new z range',
                                                                                              description_tooltip='check this on if you like to adjust z slice range for alignment',
                                                                                              value =False,
                                                                                              disabled=True)
        self.hs['L[0][2][2][1][1][0]_align_recon_optional_slice_checkbox'].layout = layout
        layout = {'width':'10%', 'height':'90%'}
        self.hs['L[0][2][2][1][1][1]_align_recon_optional_slice_start_text'] = widgets.BoundedIntText(description='',
                                                                                          description_tooltip='In the case of reading and reviewing a registration file, you need to define slice start and end.',
                                                                                          value = 0,
                                                                                          min = 0,
                                                                                          max = 10,
                                                                                          disabled=True)
        self.hs['L[0][2][2][1][1][1]_align_recon_optional_slice_start_text'].layout = layout
        layout = {'width':'60%', 'height':'90%'}
        self.hs['L[0][2][2][1][1][2]_align_recon_optional_slice_range_slider'] = widgets.IntRangeSlider(description='z range',
                                                                                          description_tooltip='In the case of reading and reviewing a registration file, you need to define slice start and end.',
                                                                                          value = 0,
                                                                                          min = 0,
                                                                                          max = 10,
                                                                                          disabled=True)
        self.hs['L[0][2][2][1][1][2]_align_recon_optional_slice_range_slider'].layout = layout
        layout = {'width':'10%', 'height':'90%'}
        self.hs['L[0][2][2][1][1][3]_align_recon_optional_slice_end_text'] = widgets.BoundedIntText(description='',
                                                                                          description_tooltip='In the case of reading and reviewing a registration file, you need to define slice start and end.',
                                                                                          value = 0,
                                                                                          min = 0,
                                                                                          max = 10,
                                                                                          disabled=True)
        self.hs['L[0][2][2][1][1][3]_align_recon_optional_slice_end_text'].layout = layout
        self.hs['L[0][2][2][1][1]_align_recon_optional_slice_region_box'].children = get_handles(self.hs, 'L[0][2][2][1][1]_align_recon_optional_slice_region_box', -1)
        self.hs['L[0][2][2][1][1][0]_align_recon_optional_slice_checkbox'].observe(self.L0_2_2_1_1_0_align_recon_optional_slice_checkbox, names='value')
        self.hs['L[0][2][2][1][1][1]_align_recon_optional_slice_start_text'].observe(self.L0_2_2_1_1_1_align_recon_optional_slice_start_text, names='value')
        self.hs['L[0][2][2][1][1][3]_align_recon_optional_slice_end_text'].observe(self.L0_2_2_1_1_3_align_recon_optional_slice_end_text, names='value')
        self.hs['L[0][2][2][1][1][2]_align_recon_optional_slice_range_slider'].observe(self.L0_2_2_1_1_2_align_recon_optional_slice_range_slider, names='value')

        ## ## ## ## ## run reg & status
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][2][1][2]_align_recon_comfirm_box'] = widgets.HBox()
        self.hs['L[0][2][2][1][2]_align_recon_comfirm_box'].layout = layout
        layout = {'width':'70%', 'height':'90%'}
        self.hs['L[0][2][2][1][2][0]_align_text'] = widgets.Text(description='',
                                                                   value='Confirm to proceed alignment ...',
                                                                   disabled=True)
        self.hs['L[0][2][2][1][2][0]_align_text'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][2][2][1][2][1]_align_button'] = widgets.Button(description='Align',
                                                                       description_tooltip='This will perform xanes3D alignment according to your configurations ...',
                                                                       disabled=True)
        self.hs['L[0][2][2][1][2][1]_align_button'].style.button_color = 'darkviolet'
        self.hs['L[0][2][2][1][2][1]_align_button'].layout = layout
        self.hs['L[0][2][2][1][2]_align_recon_comfirm_box'].children = get_handles(self.hs, 'L[0][2][2][1][2]_align_recon_comfirm_box', -1)
        self.hs['L[0][2][2][1][2][1]_align_button'].on_click(self.L0_2_2_1_2_1_align_button_click)

        ## ## ## ## ## run reg progress
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][2][1][3]_align_progress_box'] = widgets.HBox()
        self.hs['L[0][2][2][1][3]_align_progress_box'].layout = layout
        layout = {'width':'100%', 'height':'90%'}
        self.hs['L[0][2][2][1][3][0]_align_progress_bar'] = widgets.IntProgress(value=0,
                                                                                  min=0,
                                                                                  max=10,
                                                                                  step=1,
                                                                                  description='Completing:',
                                                                                  bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                                                                  orientation='horizontal')
        self.hs['L[0][2][2][1][3][0]_align_progress_bar'].layout = layout
        self.hs['L[0][2][2][1][3]_align_progress_box'].children = get_handles(self.hs, 'L[0][2][2][1][3]_align_progress_box', -1)


        self.hs['L[0][2][2][1]_align_recon_box'].children = get_handles(self.hs, 'L[0][2][2][1]_align_recon_box', -1)
        ## ## ## ## define functional widgets each tab in each sub-tab - align recon in register/review/shift TAB -- end
        
        ## ## ## ## define functional widgets each tab in each sub-tab - visualziation box in analysis&display TAB -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.28*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][2][2]_visualize_box'] = widgets.VBox()
        self.hs['L[0][2][2][2]_visualize_box'].layout = layout

        ## ## ## ## ## label visualize box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][2][2][0]_visualize_title_box'] = widgets.HBox()
        self.hs['L[0][2][2][2][0]_visualize_title_box'].layout = layout
        self.hs['L[0][2][2][2][0][0]_visualize_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Visualize XANES3D' + '</span>')
        # self.hs['L[0][2][2][2][0][0]_visualize_title_text'] = widgets.Text(value='Visualize XANES3D', disabled=True)
        layout = {'background-color':'white', 'color':'cyan', 'left':'41%'}
        self.hs['L[0][2][2][2][0][0]_visualize_title_text'].layout = layout
        self.hs['L[0][2][2][2][0]_visualize_title_box'].children = get_handles(self.hs, 'L[0][2][2][2][0]_visualize_title_box', -1)
        ## ## ## ## ## label visualize box -- end

        ## ## ## ## ## visualization option box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{1*self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][2][2][3]_visualize_viewer_option_box'] = widgets.HBox()
        self.hs['L[0][2][2][2][3]_visualize_viewer_option_box'].layout = layout
        layout = {'width':'80%', 'height':'90%'}
        self.hs['L[0][2][2][2][3][0]_visualize_viewer_option_togglebutton'] = widgets.ToggleButtons(description='viewer options',
                                                                                        description_tooltip='napari: provides better image preview functions; fiji: provides quick spectrum inspection functions',
                                                                                        options=['fiji', 'napari'],
                                                                                        value ='fiji',
                                                                                        disabled=True)
        self.hs['L[0][2][2][2][3][0]_visualize_viewer_option_togglebutton'].layout = layout
        layout = {'width':'20%', 'height':'90%'}
        self.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'] = widgets.Dropdown(description='view option',
                                                                                                  description_tooltip='dimensions are defined as: E: energy dimension; x-y: slice lateral plane; z: dimension normal to slice plane',
                                                                                                  options=['x-y-E', 'y-z-E', 'z-x-E', 'x-y-z'],
                                                                                                  value ='x-y-E',
                                                                                                  disabled=True)
        self.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'].layout = layout
        self.hs['L[0][2][2][2][3][0]_visualize_viewer_option_togglebutton'].observe(self.L0_2_2_2_3_0_visualize_viewer_option_togglebutton_change, names='value')
        self.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'].observe(self.L0_2_2_2_3_1_visualize_view_alignment_option_dropdown, names='value')
        self.hs['L[0][2][2][2][3]_visualize_viewer_option_box'].children = get_handles(self.hs, 'L[0][2][2][2][3]_visualize_viewer_option_box', -1)
        ## ## ## ## ## visualization option box -- end

        ## ## ## ## ## define slice region and view slice cuts -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{1*self.form_sz[1]-98}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'}
        self.hs['L[0][2][2][2][1]_visualize_view_alignment_box'] = widgets.HBox()
        self.hs['L[0][2][2][2][1]_visualize_view_alignment_box'].layout = layout

        layout = {'width':'40%', 'height':'90%'}
        self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'] = widgets.IntSlider(description='z',
                                                                                               description_tooltip='Select one slice in the fourth dimension',
                                                                                               value =0,
                                                                                               min = 0,
                                                                                               disabled=True)
        self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].layout = layout
        layout = {'width':'40%', 'height':'90%'}
        self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'] = widgets.IntSlider(description='E',
                                                                                                disabled=True)
        self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].layout = layout
        layout = {'width':'20%', 'height':'90%'}
        self.hs['L[0][2][2][2][1][1]_visualize_alignment_eng_text'] = widgets.FloatText(value=0,
                                                                                             description='E',
                                                                                             disabled=True)
        self.hs['L[0][2][2][2][1][1]_visualize_alignment_eng_text'].layout = layout
        self.hs['L[0][2][2][2][1]_visualize_view_alignment_box'].children = get_handles(self.hs, 'L[0][2][2][2][1]_visualize_view_alignment_box', -1)
        self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].observe(self.L0_2_2_2_1_2_visualize_view_alignment_4th_dim_slider, names='value')
        self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].observe(self.L0_2_2_2_1_3_visualize_view_alignment_slice_slider, names='value')
        ## ## ## ## ## define slice region and view slice cuts -- end

        ## ## ## ## ## basic spectroscopic visualization -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{1*self.form_sz[1]-98}px',
                  'height':f'{0.07*(self.form_sz[0]-136)}px',
                  'layout':'center'}
        self.hs['L[0][2][2][2][2]_visualize_spec_view_box'] = widgets.HBox()
        self.hs['L[0][2][2][2][2]_visualize_spec_view_box'].layout = layout
        layout = {'width':'70%', 'height':'90%'}
        self.hs['L[0][2][2][2][2][0]_visualize_spec_view_text'] = widgets.Text(description='',
                                                                               value='visualize spectrum in roi ...',
                                                                               disabled=True)
        self.hs['L[0][2][2][2][2][0]_visualize_spec_view_text'].layout = layout
        layout = {'width':'15%', 'height':'90%', 'left':'-8%'}
        self.hs['L[0][2][2][2][2][1]_visualize_spec_view_mem_monitor_checkbox'] = widgets.Checkbox(description='mem use',
                                                                                                  description_tooltip='Check on this to monitor memmory usage',
                                                                                                  value=False,
                                                                                                  disabled=True)
        self.hs['L[0][2][2][2][2][1]_visualize_spec_view_mem_monitor_checkbox'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][2][2][2][2][2]_visualize_spec_view_in_roi_button'] = widgets.Button(description='spec in roi',
                                                                                         description_tooltip='adjust the roi size and drag roi over in the particles',
                                                                                         disabled=True)
        self.hs['L[0][2][2][2][2][2]_visualize_spec_view_in_roi_button'].layout = layout
        self.hs['L[0][2][2][2][2][2]_visualize_spec_view_in_roi_button'].style.button_color = 'darkviolet'
        self.hs['L[0][2][2][2][2]_visualize_spec_view_box'].children = get_handles(self.hs, 'L[0][2][2][2][2]_visualize_spec_view_box', -1)
        self.hs['L[0][2][2][2][2][1]_visualize_spec_view_mem_monitor_checkbox'].observe(self.L0_2_2_2_2_1_visualize_spec_view_mem_monitor_checkbox, names='value')
        self.hs['L[0][2][2][2][2][2]_visualize_spec_view_in_roi_button'].on_click(self.L0_2_2_2_2_2_visualize_spec_view_in_roi_button)
        ## ## ## ## ## basic spectroscopic visualization -- end

        self.hs['L[0][2][2][2]_visualize_box'].children = get_handles(self.hs, 'L[0][2][2][2]_visualize_box', -1)
        ## ## ## ## define functional widgets each tab in each sub-tab - visualziation box in analysis&display TAB -- end

        self.hs['L[0][2][2]_reg&review_form'].children = get_handles(self.hs, 'L[0][2][2]_reg&review_form', -1)
        ## ## ## bin sub-tabs in each tab - reg&review TAB in 3D_xanes TAB -- end

        # self.hs['L[0][2][3]_fitting_form'].children = get_handles(self.hs, 'L[0][2][3]_fitting_form', -1)
        self.hs['L[0][2][3]_fitting_form'].children = [self.xanes_fitting_gui_h.hs['L[0][x][3][0]_fitting_box']]
        ## ## ## bin sub-tabs in each tab - analysis&display TAB in 3D_xanes TAB -- end
        
        self.hs['L[0][2][4]_analysis_form'].children = [self.xanes_analysis_gui_h.hs['L[0][x][4][0]_ana_box']]
        ## ## ## define 2D_XANES_tabs layout - analysis box -- end
        
        self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].initialdir = self.global_h.cwd
        self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].initialdir = self.global_h.cwd
        self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialdir = self.global_h.cwd
        self.hs['L[0][2][2][0][1][0]_read_alignment_button'].initialdir = self.global_h.cwd


    def L0_2_0_0_1_0_select_raw_h5_path_button_click(self, a):
        # restart(self, dtype='3D_XANES')
        if len(a.files[0]) != 0:
            self.xanes3D_raw_3D_h5_top_dir = os.path.abspath(a.files[0])
            self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].initialdir = os.path.abspath(a.files[0])
            self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].initialdir = os.path.abspath(a.files[0])
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialdir = os.path.abspath(a.files[0])
            self.hs['L[0][2][2][0][1][0]_read_alignment_button'].initialdir = os.path.abspath(a.files[0])
            self.xanes3D_raw_3D_h5_temp = os.path.join(self.xanes3D_raw_3D_h5_top_dir, 'fly_scan_id_{}.h5')
            with open(self.global_h.GUI_cfg_file, 'w') as f:
                json.dump({'cwd':os.path.abspath(a.files[0])},f)
            self.xanes3D_raw_h5_path_set = True
        else:
            self.hs['L[0][2][0][0][1][1]_select_raw_h5_path_text'].value = 'Choose raw h5 directory ...'
            self.xanes3D_raw_h5_path_set = False
        self.xanes3D_filepath_configured = False
        self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        self.boxes_logic()

    def L0_2_0_0_2_0_select_recon_path_button_click(self, a):
        # restart(self, dtype='3D_XANES')
        if not self.xanes3D_raw_h5_path_set:
            self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'You need to specify raw h5 top directory first ...'
            self.hs['L[0][2][0][0][2][1]_select_recon_path_text'].value = 'Choose recon top directory ...'
            self.xanes3D_recon_path_set = False
            self.xanes3D_filepath_configured = False
        else:
            if len(a.files[0]) != 0:
                self.xanes3D_recon_3D_top_dir = os.path.abspath(a.files[0])
                # self.xanes3D_recon_3D_dir_temp = os.path.join(self.xanes3D_recon_3D_top_dir,
                #                                   'recon_fly_scan_id_{0}')                
                # self.xanes3D_recon_3D_tiff_temp = os.path.join(self.xanes3D_recon_3D_top_dir,
                #                                                'recon_fly_scan_id_{0}',
                #                                                'recon_fly_scan_id_{0}_{1}.tiff')
                
                self.xanes3D_recon_3D_dir_temp = os.path.join(self.xanes3D_recon_3D_top_dir,
                                                              self.xanes3D_recon_dir_temp)
                self.xanes3D_recon_3D_tiff_temp = os.path.join(self.xanes3D_recon_3D_top_dir,
                                                               self.xanes3D_recon_dir_temp,
                                                               self.xanes3D_recon_fn_temp)
                
                self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].initialdir = os.path.abspath(a.files[0])
                with open(self.global_h.GUI_cfg_file, 'w') as f:
                    json.dump({'cwd':os.path.abspath(a.files[0])},f)
                self.xanes3D_recon_path_set = True
            else:
                self.hs['L[0][2][0][0][2][1]_select_recon_path_text'].value = 'Choose recon top directory ...'
                self.xanes3D_recon_path_set = False
            self.xanes3D_filepath_configured = False
            self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        self.boxes_logic()

    def L0_2_0_0_3_0_select_save_trial_button_click(self, a):
        # restart(self, dtype='3D_XANES')
        if self.xanes_analysis_option == 'Do New Reg':
            if len(a.files[0]) != 0:
                b = ''
                t = (time.strptime(time.asctime()))
                for ii in range(6):
                    b +=str(t[ii]).zfill(2)+'-'
                self.xanes3D_save_trial_reg_filename_template = os.path.abspath(a.files[0]).split('.')[0]+'_scan_id_{0}-{1}_'+b.strip('-')+'.h5'
                self.xanes3D_save_trial_reg_config_filename_template = os.path.abspath(a.files[0]).split('.')[0]+'_scan_id_{0}-{1}_config_'+b.strip('-')+'.json'
                self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
                self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialfile = os.path.basename(a.files[0])
                self.hs['L[0][2][2][0][1][0]_read_alignment_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
                with open(self.global_h.GUI_cfg_file, 'w') as f:
                    json.dump({'cwd':os.path.dirname(os.path.abspath(a.files[0]))},f)
                self.xanes3D_save_trial_set = True
                self.xanes3D_reg_file_set = False
                self.xanes3D_config_file_set = False
            else:
                self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Save trial registration as ...'
                self.xanes3D_save_trial_set = False
                self.xanes3D_reg_file_set = False
                self.xanes3D_config_file_set = False
            self.xanes3D_filepath_configured = False
            self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        elif self.xanes_analysis_option == 'Read Reg File':
            if len(a.files[0]) != 0:
                self.xanes3D_save_trial_reg_filename = os.path.abspath(a.files[0])
                b = ''
                t = (time.strptime(time.asctime()))
                for ii in range(6):
                    b +=str(t[ii]).zfill(2)+'-'
                self.xanes3D_save_trial_reg_config_filename = os.path.abspath(a.files[0]).split('.')[0]+'_config_'+b.strip('-')+'.json'
                self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
                self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialfile = os.path.basename(a.files[0])
                self.hs['L[0][2][2][0][1][0]_read_alignment_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
                with open(self.global_h.GUI_cfg_file, 'w') as f:
                    json.dump({'cwd':os.path.dirname(os.path.abspath(a.files[0]))},f)
                self.xanes3D_save_trial_set = False
                self.xanes3D_reg_file_set = True
                self.xanes3D_config_file_set = False
            else:
                self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Read Existing Registration File ...'
                self.xanes3D_save_trial_set = False
                self.xanes3D_reg_file_set = False
                self.xanes3D_config_file_set = False
            self.xanes3D_filepath_configured = False
            self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        elif self.xanes_analysis_option == 'Read Config File':
            if len(a.files[0]) != 0:
                self.xanes3D_save_trial_reg_config_filename_original = os.path.abspath(a.files[0])
                b = ''
                t = (time.strptime(time.asctime()))
                for ii in range(6):
                    b +=str(t[ii]).zfill(2)+'-'
                self.xanes3D_save_trial_reg_config_filename = os.path.abspath(a.files[0]).split('config')[0]+'config_'+b.strip('-')+'.json'
                self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
                self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialfile = os.path.basename(a.files[0])
                self.hs['L[0][2][2][0][1][0]_read_alignment_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
                with open(self.global_h.GUI_cfg_file, 'w') as f:
                    json.dump({'cwd':os.path.dirname(os.path.abspath(a.files[0]))},f)
                self.xanes3D_save_trial_set = False
                self.xanes3D_reg_file_set = False
                self.xanes3D_config_file_set = True
            else:
                self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Save Existing Configuration File ...'
                self.xanes3D_save_trial_set = False
                self.xanes3D_reg_file_set = False
                self.xanes3D_config_file_set = False
            self.xanes3D_filepath_configured = False
            self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        elif self.xanes_analysis_option == 'Do Analysis':
            if len(a.files[0]) != 0:
                self.xanes3D_save_trial_reg_filename = os.path.abspath(a.files[0])
                b = ''
                t = (time.strptime(time.asctime()))
                for ii in range(6):
                    b +=str(t[ii]).zfill(2)+'-'
                self.xanes3D_save_trial_reg_config_filename = os.path.abspath(a.files[0]).split('.')[0]+'_config_'+b.strip('-')+'.json'
                self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
                self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialfile = os.path.basename(a.files[0])
                self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Existing Registration File is Read ...'
                with open(self.global_h.GUI_cfg_file, 'w') as f:
                    json.dump({'cwd':os.path.dirname(os.path.abspath(a.files[0]))},f)
                self.xanes3D_save_trial_set = False
                self.xanes3D_reg_file_set = True
                self.xanes3D_config_file_set = False
            else:
                self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Read Existing Registration File ...'
                self.xanes3D_save_trial_set = False
                self.xanes3D_reg_file_set = False
                self.xanes3D_config_file_set = False
            self.xanes3D_filepath_configured = False
            self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        self.boxes_logic()

    def L0_2_0_0_4_2_file_path_options_dropdown_change(self, a):
        restart(self, dtype='3D_XANES')
        self.xanes_analysis_option = a['owner'].value
        self.xanes3D_filepath_configured = False
        if self.xanes_analysis_option == 'Do New Reg':
            self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].disabled = False
            self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].disabled = False
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].option = 'asksaveasfilename'
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].description = 'Save Reg File'
            self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Save trial registration as ...'
            self.xanes3D_save_trial_set = False
            self.xanes3D_reg_file_set = False
            self.xanes3D_config_file_set = False
            self.hs['L[0][2][2][0][1][1]_read_alignment_checkbox'].value = False
        elif self.xanes_analysis_option == 'Read Reg File':
            self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].disabled = True
            self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].disabled = True
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].option = 'askopenfilename'
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].description = 'Read Reg File'
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].open_filetypes = (('h5 files', '*.h5'),)
            self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Read Existing Registration File ...'
            self.xanes3D_save_trial_set = False
            self.xanes3D_reg_file_set = False
            self.xanes3D_config_file_set = False
            self.hs['L[0][2][2][0][1][0]_read_alignment_button'].text_h = self.hs['L[0][2][2][0][4][0]_confirm_review_results_text']
        elif self.xanes_analysis_option == 'Read Config File':
            self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].disabled = True
            self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].disabled = True
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].option = 'askopenfilename'
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].description = 'Read Config'
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].open_filetypes = (('json files', '*.json'), ('text files', '*.txt'))
            self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Save Existing Configuration File ...'
            self.xanes3D_save_trial_set = False
            self.xanes3D_reg_file_set = False
            self.xanes3D_config_file_set = False
        elif self.xanes_analysis_option == 'Do Analysis':
            self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].disabled = True
            self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].disabled = True
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].option = 'askopenfilename'
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].description = 'Read Reg File'
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].open_filetypes = (('h5 files', '*.h5'),)
            self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Read Existing Registration File ...'
        self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].icon = "square-o"
        self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].style.button_color = "orange"
        self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        self.boxes_logic()

    def L0_2_0_0_4_0_confirm_file_path_button_click(self, a):
        if self.xanes_analysis_option == 'Do New Reg':
            if not self.xanes3D_raw_h5_path_set:
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please specifiy raw h5 file location ...'
                self.xanes3D_filepath_configured = False
            elif not self.xanes3D_recon_path_set:
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please specifiy recon top directory location ...'
                self.xanes3D_filepath_configured = False
            elif not self.xanes3D_save_trial_set:
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please specifiy where to save trial reg result ...'
                self.xanes3D_filepath_configured = False
            else:
                b = glob.glob(os.path.join(self.xanes3D_raw_3D_h5_top_dir, 'fly*.h5'))
                self.xanes3D_available_raw_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])
                b = glob.glob(os.path.join(self.xanes3D_recon_3D_top_dir, 'recon_fly*'))
                self.xanes3D_available_recon_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])
                # b = glob.glob(os.path.join(self.xanes3D_recon_3D_dir_temp.format(self.xanes3D_fixed_scan_id),
                #                            f'recon_fly_scan_id_{self.xanes3D_fixed_scan_id}_*.tiff'))
                # self.xanes3D_available_sli_file_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])

                self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].max = max(self.xanes3D_available_raw_ids)
                self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].max = max(self.xanes3D_available_raw_ids)
                self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].value = min(self.xanes3D_available_raw_ids)
                self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].value = min(self.xanes3D_available_raw_ids)
                self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].min = min(self.xanes3D_available_raw_ids)
                self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].min = min(self.xanes3D_available_raw_ids)
                self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].max = min(self.xanes3D_available_raw_ids)
                self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].min = min(self.xanes3D_available_raw_ids)
                self.xanes3D_fixed_scan_id = self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].value
                self.xanes3D_fixed_sli_id = self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].value
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'XANES3D file config is done ...'
                # self.update_xanes3D_config()
                # print(self.xanes3D_save_trial_reg_config_filename)
                # self.xanes3D_reg_best_match_filename = os.path.splitext(self.xanes3D_save_trial_reg_config_filename)[0].replace('config', 'reg_best_match') + '.json'
                self.xanes3D_filepath_configured = True
        elif self.xanes_analysis_option == 'Read Reg File':
            if not self.xanes3D_reg_file_set:
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please specifiy where to read trial reg result ...'
                self.xanes3D_filepath_configured = False
            else:
                read_config_from_reg_file(self, dtype='3D_XANES')
                self.xanes3D_review_aligned_img = np.ndarray(self.trial_reg[0].shape)

                b = glob.glob(os.path.join(self.xanes3D_raw_3D_h5_top_dir, 'fly*.h5'))
                self.xanes3D_available_raw_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])
                b = glob.glob(os.path.join(self.xanes3D_recon_3D_top_dir, 'recon_fly*'))
                self.xanes3D_available_recon_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])
                b = glob.glob(self.xanes3D_recon_3D_tiff_temp.format(self.xanes3D_fixed_scan_id, '*'))
                self.xanes3D_available_sli_file_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])

                self.xanes3D_alignment_pair_id = 0
                self.hs['L[0][2][2][0][2][0]_reg_pair_slider'].max = self.xanes3D_alignment_pairs.shape[0]-1
                self.hs['L[0][2][2][0][2][0]_reg_pair_slider'].value = 0
                self.hs['L[0][2][2][0][2][0]_reg_pair_slider'].min = 0
                self.hs['L[0][2][2][0][3][0]_zshift_slider'].max = self.xanes3D_reg_sli_search_half_width*2
                self.hs['L[0][2][2][0][3][0]_zshift_slider'].value = 1
                self.hs['L[0][2][2][0][3][0]_zshift_slider'].min = 1
                self.hs['L[0][2][2][1][1][0]_align_recon_optional_slice_checkbox'].value = False
                self.hs['L[0][2][2][1][1][2]_align_recon_optional_slice_range_slider'].max = max(self.xanes3D_available_sli_file_ids)
                self.hs['L[0][2][2][1][1][2]_align_recon_optional_slice_range_slider'].value = (self.xanes3D_roi[4],
                                                                                                self.xanes3D_roi[5])
                self.hs['L[0][2][2][1][1][2]_align_recon_optional_slice_range_slider'].min = min(self.xanes3D_available_sli_file_ids)
                self.hs['L[0][2][2][1][1][3]_align_recon_optional_slice_end_text'].max = max(self.xanes3D_available_sli_file_ids)
                self.hs['L[0][2][2][1][1][3]_align_recon_optional_slice_end_text'].value = max(self.xanes3D_available_sli_file_ids)
                self.hs['L[0][2][2][1][1][3]_align_recon_optional_slice_end_text'].min = min(self.xanes3D_available_sli_file_ids)
                self.hs['L[0][2][2][1][1][1]_align_recon_optional_slice_start_text'].max = max(self.xanes3D_available_sli_file_ids)
                self.hs['L[0][2][2][1][1][1]_align_recon_optional_slice_start_text'].value = min(self.xanes3D_available_sli_file_ids)
                self.hs['L[0][2][2][1][1][1]_align_recon_optional_slice_start_text'].min = min(self.xanes3D_available_sli_file_ids)

                fiji_viewer_off(self.global_h, self, viewer_name='all')
                self.xanes3D_review_shift_dict={}
                self.xanes3D_reg_best_match_filename = os.path.splitext(self.xanes3D_save_trial_reg_config_filename)[0].replace('config', 'reg_best_match') + '.json'
                self.xanes3D_recon_path_set = True
                self.xanes3D_filepath_configured = True
                self.xanes3D_reg_done = True
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'XANES3D file config is done ...'
        elif self.xanes_analysis_option == 'Read Config File':
            if not self.xanes3D_config_file_set:
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please specifiy where to read the configuration file ...'
                self.xanes3D_filepath_configured = False
            else:
                self.read_xanes3D_config()
                self.set_xanes3D_variables()

                if self.xanes3D_roi_configured:
                    self.xanes3D_img_roi = np.ndarray([self.xanes3D_roi[5]-self.xanes3D_roi[4]+1,
                                                       self.xanes3D_roi[1]-self.xanes3D_roi[0],
                                                       self.xanes3D_roi[3]-self.xanes3D_roi[2]])
                    self.xanes3D_mask = np.ndarray([self.xanes3D_roi[1]-self.xanes3D_roi[0],
                                                    self.xanes3D_roi[3]-self.xanes3D_roi[2]])

                if self.xanes3D_reg_done:
                    with h5py.File(self.xanes3D_save_trial_reg_filename, 'r') as f:
                        self.trial_reg = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format('000')][:]
                        self.trial_reg_fixed = f['/trial_registration/trial_reg_results/{0}/trial_fixed_img{0}'.format('000')][:]
                        self.xanes3D_review_aligned_img = np.ndarray(self.trial_reg[0].shape)
                # self.set_xanes3D_handles()
                # self.set_xanes3D_variables()
                self.xanes3D_recon_path_set = True
                self.xanes_analysis_option = 'Read Config File'

                self.xanes3D_reg_best_match_filename = os.path.splitext(self.xanes3D_save_trial_reg_config_filename)[0].replace('config', 'reg_best_match') + '.json'
                self.xanes3D_filepath_configured = True

                if self.xanes3D_alignment_done:
                    with h5py.File(self.xanes3D_save_trial_reg_filename, 'r') as f:
                        self.xanes_analysis_data_shape = f['/registration_results/reg_results/registered_xanes3D'].shape
                        self.xanes_analysis_eng_list = f['/trial_registration/trial_reg_parameters/eng_list'][:]
                        self.xanes3D_scan_id_s = f['/trial_registration/trial_reg_parameters/scan_ids'][0]
                        self.xanes3D_scan_id_e = f['/trial_registration/trial_reg_parameters/scan_ids'][-1]
                        self.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][0, :, :, :]
                    self.xanes3D_reg_best_match_filename = os.path.splitext(self.xanes3D_save_trial_reg_config_filename)[0].replace('config', 'reg_best_match') + '.json'
                    self.xanes_element = determine_element(self.xanes_analysis_eng_list)
                    tem = determine_fitting_energy_range(self.xanes_element)
                    self.xanes_analysis_edge_eng = tem[0] 
                    self.xanes_analysis_wl_fit_eng_s = tem[1]
                    self.xanes_analysis_wl_fit_eng_e = tem[2]
                    self.xanes_analysis_pre_edge_e = tem[3]
                    self.xanes_analysis_post_edge_s = tem[4] 
                    self.xanes_analysis_edge_0p5_fit_s = tem[5]
                    self.xanes_analysis_edge_0p5_fit_e = tem[6]
                    
                    self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].value = 0
                    self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].value = 0
                    self.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'].value = 'x-y-E'
                    self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].description = 'E'
                    self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].max = self.xanes_analysis_data_shape[0]-1
                    self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].min = 0
                    self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].description = 'z'
                    self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].max = self.xanes_analysis_data_shape[1]-1
                    self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].min = 0
                    self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'XANES3D file config is done ...'
                    self.xanes_analysis_type = 'full'
                    self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][0][0]_fit_eng_range_option_dropdown'].value = 'full'

                self.set_xanes3D_handles()
                self.set_xanes3D_variables()
                fiji_viewer_off(self.global_h, self,viewer_name='all')
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'XANES3D file config is done ...'
        elif self.xanes_analysis_option == 'Do Analysis':
            if not self.xanes3D_reg_file_set:
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please specifiy where to read the aligned data ...'
                self.xanes3D_filepath_configured = False
                self.xanes3D_indices_configured = False
                self.xanes3D_roi_configured = False
                self.xanes3D_reg_params_configured = False
                self.xanes3D_reg_done = False
                self.xanes3D_alignment_done = False
            else:
                self.xanes3D_filepath_configured = True
                self.xanes3D_indices_configured = False
                self.xanes3D_roi_configured = False
                self.xanes3D_reg_params_configured = False
                self.xanes3D_reg_done = False
                self.xanes3D_alignment_done = False
                with h5py.File(self.xanes3D_save_trial_reg_filename, 'r') as f:
                    self.xanes_analysis_data_shape = f['/registration_results/reg_results/registered_xanes3D'].shape
                    self.xanes_analysis_eng_list = f['/trial_registration/trial_reg_parameters/eng_list'][:]
                    self.xanes3D_scan_id_s = f['/trial_registration/trial_reg_parameters/scan_ids'][0]
                    self.xanes3D_scan_id_e = f['/trial_registration/trial_reg_parameters/scan_ids'][-1]
                    self.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][0, :, :, :]
                self.xanes3D_reg_best_match_filename = os.path.splitext(self.xanes3D_save_trial_reg_config_filename)[0].replace('config', 'reg_best_match') + '.json'
                self.xanes_element = determine_element(self.xanes_analysis_eng_list)
                tem = determine_fitting_energy_range(self.xanes_element)
                self.xanes_analysis_edge_eng = tem[0] 
                self.xanes_analysis_wl_fit_eng_s = tem[1]
                self.xanes_analysis_wl_fit_eng_e = tem[2]
                self.xanes_analysis_pre_edge_e = tem[3]
                self.xanes_analysis_post_edge_s = tem[4] 
                self.xanes_analysis_edge_0p5_fit_s = tem[5]
                self.xanes_analysis_edge_0p5_fit_e = tem[6]
                    
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].value = 0
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].value = 0
                self.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'].value = 'x-y-E'
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].description = 'E'
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].max = self.xanes_analysis_data_shape[0]-1
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].min = 0
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].description = 'z'
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].max = self.xanes_analysis_data_shape[1]-1
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].min = 0
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'XANES3D file config is done ...'
                self.xanes_analysis_type = 'full'
                self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][0][0]_fit_eng_range_option_dropdown'].value = 'full'
        # self.xanes3D_raw_fn_temp = self.global_h.io_xanes3D_cfg['structured_h5_reader']['tomo_raw_fn_template']
        self.boxes_logic()

    def L0_2_0_1_1_0_select_scan_id_start_text_change(self, a):
        if os.path.exists(self.xanes3D_raw_3D_h5_temp.format(a['owner'].value)):
            if os.path.exists(self.xanes3D_recon_3D_dir_temp.format(a['owner'].value)):
                if (a['owner'].value in self.xanes3D_available_raw_ids) and (a['owner'].value in self.xanes3D_available_recon_ids):
                    self.xanes3D_scan_id_s = a['owner'].value
                    self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].min = a['owner'].value
                    self.xanes3D_scan_id_set = True
                    self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].max = self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].value
                    self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].min = a['owner'].value
                    self.hs['L[0][2][0][1][4][1]_confirm_config_data_text'].value = 'scan_id_s are changed ...'
                else:
                    self.xanes3D_scan_id_set = False
            else:
                print(f"fly_scan_id_{a['owner'].value} is not reconstructed yet.")
        self.xanes3D_indices_configured = False
        self.boxes_logic()

    def L0_2_0_1_1_1_select_scan_id_end_text_change(self, a):
        ids = np.arange(self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].value, a['owner'].value)
        if (not set(ids).issubset(set(self.xanes3D_available_recon_ids))) or (not set(ids).issubset(set(self.xanes3D_available_raw_ids))):
            self.hs['L[0][2][0][1][4][1]_confirm_config_data_text'].value = 'The set index range is out of either the available raw or recon dataset ranges ...'
        else:
            self.xanes3D_scan_id_e = a['owner'].value
            self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].max = a['owner'].value
            self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].min = self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].value
        self.xanes3D_indices_configured = False
        self.boxes_logic()

    def L0_2_0_1_2_0_fixed_scan_id_slider_change(self, a):
        self.xanes3D_fixed_scan_id = a['owner'].value
        # b = glob.glob(os.path.join(self.xanes3D_recon_3D_dir_temp.format(self.xanes3D_fixed_scan_id),
        #                             f'recon_fly_scan_id_{self.xanes3D_fixed_scan_id}_*.tiff'))
        b = glob.glob(os.path.join(self.xanes3D_recon_3D_dir_temp.format(self.xanes3D_fixed_scan_id),
                                   self.xanes3D_recon_fn_temp.format(self.xanes3D_fixed_scan_id, '*')))
        self.xanes3D_available_sli_file_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])

        if (self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].value in (self.xanes3D_available_sli_file_ids)):
            self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].max = max(self.xanes3D_available_sli_file_ids)
            self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min = min(self.xanes3D_available_sli_file_ids)
        else:
            self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].max = max(self.xanes3D_available_sli_file_ids)
            self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].value = min(self.xanes3D_available_sli_file_ids)
            self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min = min(self.xanes3D_available_sli_file_ids)

        if self.hs['L[0][2][0][1][3][0]_fiji_virtural_stack_preview_checkbox'].value:
            fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_virtural_stack_preview_viewer')
        self.xanes3D_indices_configured = False
        self.boxes_logic()

    def L0_2_0_1_2_1_fixed_sli_id_slider_change(self, a):
        if self.hs['L[0][2][0][1][3][0]_fiji_virtural_stack_preview_checkbox'].value:
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_virtural_stack_preview_viewer')
            if viewer_state:
                self.global_h.xanes3D_fiji_windows['xanes3D_virtural_stack_preview_viewer']['ip'].setSlice(a['owner'].value-a['owner'].min+1)
                self.xanes3D_fixed_sli_id = a['owner'].value
                self.hs['L[0][2][0][1][4][1]_confirm_config_data_text'].value = 'fixed slice id is changed ...'
            else:
                self.hs['L[0][2][0][1][4][1]_confirm_config_data_text'].value = 'Please turn on fiji previewer first ...'
        self.xanes3D_indices_configured = False
        # self.boxes_logic()

    def L0_2_0_1_3_0_fiji_virtural_stack_preview_checkbox_change(self, a):
        if a['owner'].value:
            fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_virtural_stack_preview_viewer')
        else:
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_virtural_stack_preview_viewer')
            if viewer_state:
                self.global_h.xanes3D_fiji_windows['xanes3D_virtural_stack_preview_viewer']['ip'].close()
                self.global_h.xanes3D_fiji_windows['xanes3D_virtural_stack_preview_viewer']['ip'] = None
                self.global_h.xanes3D_fiji_windows['xanes3D_virtural_stack_preview_viewer']['fiji_id'] = None
            self.xanes3D_indices_configured = False
            self.boxes_logic()

    def L0_2_0_1_3_1_fiji_close_button_click(self, a):
        if self.hs['L[0][2][0][1][3][0]_fiji_virtural_stack_preview_checkbox'].value:
            self.hs['L[0][2][0][1][3][0]_fiji_virtural_stack_preview_checkbox'].value = False
        if self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value:
            self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value = False
        try:
            for ii in (self.global_h.WindowManager.getIDList()):
                self.global_h.WindowManager.getImage(ii).close()
        except:
            pass
        self.global_h.xanes3D_fiji_windows['xanes3D_virtural_stack_preview_viewer']['ip'] = None
        self.global_h.xanes3D_fiji_windows['xanes3D_virtural_stack_preview_viewer']['fiji_id'] = None
        self.global_h.xanes3D_fiji_windows['xanes3D_mask_viewer']['ip'] = None
        self.global_h.xanes3D_fiji_windows['xanes3D_mask_viewer']['fiji_id'] = None
        self.global_h.xanes3D_fiji_windows['xanes3D_analysis_viewer']['ip'] = None
        self.global_h.xanes3D_fiji_windows['xanes3D_analysis_viewer']['fiji_id'] = None
        self.global_h.xanes3D_fiji_windows['analysis_viewer_z_plot_viewer']['ip'] = None
        self.global_h.xanes3D_fiji_windows['analysis_viewer_z_plot_viewer']['fiji_id'] = None
        self.xanes3D_indices_configured = False
        self.boxes_logic()

    def L0_2_0_1_4_0_confirm_config_data_button_click(self, a):
        if self.xanes_analysis_option == 'Do New Reg':
            self.xanes3D_save_trial_reg_config_filename = self.xanes3D_save_trial_reg_config_filename_template.format(self.xanes3D_scan_id_s, self.xanes3D_scan_id_e)
            self.xanes3D_reg_best_match_filename = os.path.splitext(self.xanes3D_save_trial_reg_config_filename)[0].replace('config', 'reg_best_match') + '.json'
        if not self.xanes3D_indices_configured:
            self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].max = self.global_h.xanes3D_fiji_windows['xanes3D_virtural_stack_preview_viewer']['ip'].width
            self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].min = 0
            self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].max = self.global_h.xanes3D_fiji_windows['xanes3D_virtural_stack_preview_viewer']['ip'].height
            self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].min = 0
            self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].max = self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].max
            self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].min = self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min

            self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].upper = self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].max
            self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].lower = self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min

            self.xanes3D_scan_id_s = self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'] .value
            self.xanes3D_scan_id_e = self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].value
            self.xanes3D_fixed_sli_id = self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].value
            self.xanes3D_fixed_scan_id = self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].value
            self.hs['L[0][2][0][1][4][1]_confirm_config_data_text'].value = 'Indices configuration is done ...'
            self.hs['L[0][2][1][0][2][0]_confirm_roi_text'].value = 'Please confirm after ROI is set'
            if self.xanes_analysis_option == 'Do New Reg':
                self.xanes3D_save_trial_reg_filename = self.xanes3D_save_trial_reg_filename_template.format(self.xanes3D_scan_id_s, self.xanes3D_scan_id_e)
            self.xanes3D_indices_configured = True
            self.update_xanes3D_config()
            json.dump(self.xanes3D_config, open(self.xanes3D_save_trial_reg_config_filename, 'w'), cls=NumpyArrayEncoder)
        self.xanes3D_eng_list = self.reader(os.path.join(self.xanes3D_raw_3D_h5_top_dir, self.xanes3D_raw_fn_temp), 
                                            dtype='eng', sli=[[self.xanes3D_scan_id_s, self.xanes3D_scan_id_e+1]],
                                            cfg=self.io_xanes3D_cfg)
        if self.xanes3D_eng_list.max() < 70:
            self.xanes3D_eng_list *= 1000
        self.boxes_logic()

    def L0_2_1_0_1_0_3D_roi_x_slider_change(self, a):
        self.xanes3D_roi_configured = False
        self.boxes_logic()
        self.hs['L[0][2][1][0][2][0]_confirm_roi_text'].value = 'Please confirm after ROI is set'
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_virtural_stack_preview_viewer')
        if (not data_state) | (not viewer_state):
            fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_virtural_stack_preview_viewer')
        self.global_h.xanes3D_fiji_windows['xanes3D_virtural_stack_preview_viewer']['ip'].setRoi(self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value[0],
                                           self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value[0],
                                           self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value[1]-self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value[0],
                                           self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value[1]-self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value[0])

    def L0_2_1_0_1_1_3D_roi_y_slider_change(self, a):
        self.xanes3D_roi_configured = False
        self.boxes_logic()
        self.hs['L[0][2][1][0][2][0]_confirm_roi_text'].value = 'Please confirm after ROI is set'
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_virtural_stack_preview_viewer')
        if (not data_state) | (not viewer_state):
            fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_virtural_stack_preview_viewer')
        self.global_h.xanes3D_fiji_windows['xanes3D_virtural_stack_preview_viewer']['ip'].setRoi(self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value[0],
                                                                                self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value[0],
                                           self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value[1]-self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value[0],
                                           self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value[1]-self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value[0])

    def L0_2_1_0_1_2_3D_roi_z_slider_change(self, a):
        self.xanes3D_roi_configured = False
        if a['owner'].upper < self.xanes3D_fixed_sli_id:
            a['owner'].upper = self.xanes3D_fixed_sli_id
        if a['owner'].lower > self.xanes3D_fixed_sli_id:
            a['owner'].lower = self.xanes3D_fixed_sli_id
        a['owner'].mylower = a['owner'].lower
        a['owner'].myupper = a['owner'].upper
        self.hs['L[0][2][1][0][2][0]_confirm_roi_text'].value = 'Please confirm after ROI is set'

    def L0_2_1_0_1_2_3D_roi_z_slider_lower_change(self, a):
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_virtural_stack_preview_viewer')
        if (not data_state) | (not viewer_state):
            fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_virtural_stack_preview_viewer')
        self.global_h.xanes3D_fiji_windows['xanes3D_virtural_stack_preview_viewer']['ip'].setSlice(self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].value[0]-self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].min+1)
        self.boxes_logic()

    def L0_2_1_0_1_2_3D_roi_z_slider_upper_change(self, a):
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_virtural_stack_preview_viewer')
        if (not data_state) | (not viewer_state):
            fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_virtural_stack_preview_viewer')
        self.global_h.xanes3D_fiji_windows['xanes3D_virtural_stack_preview_viewer']['ip'].setSlice(self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].value[1]-self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].min+1)
        self.boxes_logic()

    def L0_2_1_0_2_1_confirm_roi_button_click(self, a):
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_virtural_stack_preview_viewer')
        if viewer_state:
            fiji_viewer_off(self.global_h, self,viewer_name='xanes3D_virtural_stack_preview_viewer')
        self.hs['L[0][2][0][1][3][0]_fiji_virtural_stack_preview_checkbox'].value = False
        self.xanes3D_indices_configured = True

        if self.xanes3D_roi_configured:
            pass
        else:
            self.xanes3D_roi = [self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value[0],
                                self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value[1],
                                self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value[0],
                                self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value[1],
                                self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].value[0],
                                self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].value[1]]

            self.xanes3D_img_roi = np.ndarray([self.xanes3D_roi[5]-self.xanes3D_roi[4]+1,
                                               self.xanes3D_roi[1]-self.xanes3D_roi[0],
                                               self.xanes3D_roi[3]-self.xanes3D_roi[2]])
            self.xanes3D_mask = np.ndarray([self.xanes3D_roi[1]-self.xanes3D_roi[0],
                                            self.xanes3D_roi[3]-self.xanes3D_roi[2]])

            self.hs['L[0][2][1][0][2][0]_confirm_roi_text'].value = 'ROI configuration is done ...'
            self.hs['L[0][2][1][1][3][0]_sli_search_slider'].max = min(abs(self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].value[1]-self.xanes3D_fixed_sli_id),
                                                                       abs(self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].value[0]-self.xanes3D_fixed_sli_id))
            self.xanes3D_roi_configured = True
        self.update_xanes3D_config()
        json.dump(self.xanes3D_config, open(self.xanes3D_save_trial_reg_config_filename, 'w'), cls=NumpyArrayEncoder)
        self.boxes_logic()

    def L0_2_1_1_1_0_fiji_mask_viewer_checkbox_change(self, a):
        if a['owner'].value:
            fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_mask_viewer')
        else:
            try:
                self.global_h.xanes3D_fiji_windows['xanes3D_mask_viewer']['ip'].close()
            except:
                pass
            self.global_h.xanes3D_fiji_windows['xanes3D_mask_viewer']['ip'] = None
            self.global_h.xanes3D_fiji_windows['xanes3D_mask_viewer']['fiji_id'] = None
        self.boxes_logic()

    def L0_2_1_1_1_1_chunk_checkbox_change(self, a):
        self.xanes3D_reg_params_configured = False
        if a['owner'].value:
            self.xanes3D_reg_use_chunk = True
        else:
            self.xanes3D_reg_use_chunk = False
        self.boxes_logic()

    def L0_2_1_1_2_0_use_mask_checkbox_change(self, a):
        self.xanes3D_reg_params_configured = False
        if self.hs['L[0][2][1][1][2][0]_use_mask_checkbox'].value:
            self.xanes3D_reg_use_mask = True
        else:
            self.xanes3D_reg_use_mask = False
        self.boxes_logic()

    def L0_2_1_1_2_1_mask_thres_slider_change(self, a):
        self.xanes3D_reg_params_configured = False
        self.xanes3D_reg_mask_thres = a['owner'].value
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_mask_viewer')
        if ((not data_state) |
            (not viewer_state)):
            self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value = False
            self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value = True

        if self.xanes3D_reg_mask_dilation_width == 0:
            self.xanes3D_mask[:] = (self.xanes3D_img_roi[self.xanes3D_fixed_sli_id-self.xanes3D_roi[4]]>self.xanes3D_reg_mask_thres).astype(np.uint8)[:]
        else:
            self.xanes3D_mask[:] = skm.binary_dilation((self.xanes3D_img_roi[self.xanes3D_fixed_sli_id-self.xanes3D_roi[4]]>self.xanes3D_reg_mask_thres).astype(np.uint8),
                                                       np.ones([self.xanes3D_reg_mask_dilation_width,
                                                                self.xanes3D_reg_mask_dilation_width])).astype(np.uint8)[:]
        self.global_h.xanes3D_fiji_windows['xanes3D_mask_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
            self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(self.xanes3D_img_roi*self.xanes3D_mask)), self.global_h.ImagePlusClass))
        self.global_h.xanes3D_fiji_windows['xanes3D_mask_viewer']['ip'].setSlice(self.xanes3D_fixed_sli_id-self.xanes3D_roi[4])
        self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        # self.boxes_logic()

    def L0_2_1_1_2_2_mask_dilation_slider_change(self, a):
        self.xanes3D_reg_params_configured = False
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_mask_viewer')
        if (not data_state) | (not viewer_state):
            self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value = False
            self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value = True
        self.xanes3D_reg_mask_dilation_width = a['owner'].value
        if self.xanes3D_reg_mask_dilation_width == 0:
            self.xanes3D_mask[:] = (self.xanes3D_img_roi[self.xanes3D_fixed_sli_id-self.xanes3D_roi[4]]>self.xanes3D_reg_mask_thres).astype(np.uint8)[:]
        else:
            self.xanes3D_mask[:] = skm.binary_dilation((self.xanes3D_img_roi[self.xanes3D_fixed_sli_id-self.xanes3D_roi[4]]>self.xanes3D_reg_mask_thres).astype(np.uint8),
                                                       np.ones([self.xanes3D_reg_mask_dilation_width,
                                                                self.xanes3D_reg_mask_dilation_width])).astype(np.uint8)[:]
        self.global_h.xanes3D_fiji_windows['xanes3D_mask_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
            self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(self.xanes3D_img_roi*self.xanes3D_mask)), self.global_h.ImagePlusClass))
        self.global_h.xanes3D_fiji_windows['xanes3D_mask_viewer']['ip'].setSlice(self.xanes3D_fixed_sli_id-self.xanes3D_roi[4])
        self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        # self.boxes_logic()

    def L0_2_1_1_3_0_sli_search_slider_change(self, a):
        self.xanes3D_reg_params_configured = False
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_mask_viewer')
        if (not data_state) | (not viewer_state):
            self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value = False
            self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value = True
        self.global_h.xanes3D_fiji_windows['xanes3D_mask_viewer']['ip'].setSlice(a['owner'].value)
        # self.boxes_logic()

    def L0_2_1_1_3_1_chunk_sz_slider_change(self, a):
        self.xanes3D_reg_params_configured = False
        self.boxes_logic()

    def L0_2_1_1_4_0_reg_method_dropdown_change(self, a):
        self.xanes3D_reg_params_configured = False
        self.boxes_logic()

    def L0_2_1_1_4_1_ref_mode_dropdown_change(self, a):
        self.xanes3D_reg_params_configured = False
        self.boxes_logic()

    def L0_2_1_1_5_1_confirm_reg_params_button_click(self, a):
        fiji_viewer_off(self.global_h, self, viewer_name='xanes3D_mask_viewer')
        self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value = False
        self.xanes3D_reg_sli_search_half_width = self.hs['L[0][2][1][1][3][0]_sli_search_slider'].value
        self.xanes3D_reg_chunk_sz = self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].value
        self.xanes3D_reg_method = self.hs['L[0][2][1][1][4][0]_reg_method_dropdown'].value
        self.xanes3D_reg_ref_mode = self.hs['L[0][2][1][1][4][1]_ref_mode_dropdown'].value
        self.xanes3D_reg_mask_dilation_width = self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].value
        self.xanes3D_reg_mask_thres = self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].value
        self.xanes3D_reg_params_configured = self.hs['L[0][2][1][1][1][1]_chunk_checkbox'].value
        self.hs['L[0][2][1][1][5][0]_confirm_reg_params_text'].value = 'registration parameters are set ...'
        self.xanes3D_reg_params_configured = True
        self.update_xanes3D_config()
        json.dump(self.xanes3D_config, open(self.xanes3D_save_trial_reg_config_filename, 'w'), cls=NumpyArrayEncoder)
        self.boxes_logic()

    def L0_2_1_2_1_0_run_reg_button_click(self, a):
        reg = xr.regtools(dtype='3D_XANES', method=self.xanes3D_reg_method, mode='TRANSLATION')
        reg.set_raw_data_info(**{'raw_h5_top_dir':self.xanes3D_raw_3D_h5_top_dir,
                                 'recon_top_dir':self.xanes3D_recon_3D_top_dir})
        reg.set_method(self.xanes3D_reg_method)
        reg.set_ref_mode(self.xanes3D_reg_ref_mode)
        reg.set_eng_list(self.xanes3D_eng_list)
        reg.set_xanes3D_raw_h5_top_dir(self.xanes3D_raw_3D_h5_top_dir)
        reg.set_indices(self.xanes3D_scan_id_s, self.xanes3D_scan_id_e, self.xanes3D_fixed_scan_id)
        reg.set_reg_options(use_mask=self.xanes3D_reg_use_mask, mask_thres=self.xanes3D_reg_mask_thres,
                        use_chunk=self.xanes3D_reg_use_chunk, chunk_sz=self.xanes3D_reg_chunk_sz,
                        use_smooth_img=self.xanes3D_reg_use_smooth_img, smooth_sigma=self.xanes3D_reg_smooth_sigma)
        reg.set_roi(self.xanes3D_roi)
        reg.set_mask(self.xanes3D_mask)

        reg.set_xanes3D_recon_path_template(self.xanes3D_recon_3D_tiff_temp)
        reg.set_saving(os.path.dirname(self.xanes3D_save_trial_reg_filename),
                       fn=os.path.basename(self.xanes3D_save_trial_reg_filename))

        reg.xanes3D_sli_search_half_range = self.xanes3D_reg_sli_search_half_width
        reg.xanes3D_recon_fixed_sli = self.xanes3D_fixed_sli_id
        reg.compose_dicts()
        reg.reg_xanes3D_chunk()
        
        with h5py.File(self.xanes3D_save_trial_reg_filename, 'r') as f:
            self.trial_reg = np.ndarray(f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format(str(0).zfill(3))].shape)
            self.trial_reg_fixed = np.ndarray(f['/trial_registration/trial_reg_results/{0}/trial_fixed_img{0}'.format(str(0).zfill(3))].shape)
            self.xanes3D_alignment_pairs = f['/trial_registration/trial_reg_parameters/alignment_pairs'][:]
            self.hs['L[0][2][2][0][2][0]_reg_pair_slider'].max = self.xanes3D_alignment_pairs.shape[0]-1
            self.hs['L[0][2][2][0][2][0]_reg_pair_slider'].min = 0
        self.hs['L[0][2][2][0][3][0]_zshift_slider'].max = self.xanes3D_reg_sli_search_half_width*2
        self.hs['L[0][2][2][0][3][0]_zshift_slider'].value = 1
        self.hs['L[0][2][2][0][3][0]_zshift_slider'].min = 1
        self.hs['L[0][2][2][0][5][2]_review_manual_z_shift_text'].max = self.xanes3D_reg_sli_search_half_width*2

        self.xanes3D_review_aligned_img = np.ndarray(self.trial_reg[0].shape)
        self.xanes3D_review_shift_dict = {}
        self.xanes3D_review_bad_shift = False
        self.xanes3D_reg_done = True
        self.xanes3D_reg_best_match_filename = os.path.splitext(self.xanes3D_save_trial_reg_config_filename)[0].replace('config', 'reg_best_match') + '.json'
        self.update_xanes3D_config()
        json.dump(self.xanes3D_config, open(self.xanes3D_save_trial_reg_config_filename, 'w'), cls=NumpyArrayEncoder)
        self.boxes_logic()

    def L0_2_2_0_1_0_read_alignment_checkbox_change(self, a):
        self.xanes3D_use_existing_reg_reviewed = a['owner'].value
        self.xanes3D_reg_review_done = False
        self.boxes_logic()

    def L0_2_2_0_1_1_read_alignment_button_click(self, a):
        if len(a.files[0]) != 0:
            self.xanes3D_reg_file_readed = True
            self.xanes3D_reg_review_file = os.path.abspath(a.files[0])
            if os.path.splitext(self.xanes3D_reg_review_file)[1] == '.json':
                self.xanes3D_review_shift_dict = json.load(open(self.xanes3D_reg_review_file, 'r'))
            else:
                self.xanes3D_review_shift_dict = np.float32(np.genfromtxt(self.xanes3D_reg_review_file))
            for ii in self.xanes3D_review_shift_dict:
                self.xanes3D_review_shift_dict[ii] = np.float32(np.array(self.xanes3D_review_shift_dict[ii]))
        else:
            self.xanes3D_reg_file_readed = False
        self.xanes3D_reg_review_done = False
        self.boxes_logic()

    def L0_2_2_0_2_0_reg_pair_slider_change(self, a):
        self.xanes3D_alignment_pair_id = a['owner'].value
        with h5py.File(self.xanes3D_save_trial_reg_filename, 'r') as f:
            self.trial_reg[:] = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format(str(self.xanes3D_alignment_pair_id).zfill(3))][:]
            self.trial_reg_fixed[:] = f['/trial_registration/trial_reg_results/{0}/trial_fixed_img{0}'.format(str(self.xanes3D_alignment_pair_id).zfill(3))][:]
        fiji_viewer_off(self.global_h, self, viewer_name='xanes3D_review_viewer')
        self.global_h.ij.py.run_macro("""call("java.lang.System.gc")""")
        fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_review_viewer')
        self.global_h.xanes3D_fiji_windows['xanes3D_review_viewer']['ip'].setTitle('reg pair: '+str(self.xanes3D_alignment_pair_id).zfill(3))
        self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")

        self.xanes3D_review_bad_shift = False
        self.xanes3D_reg_review_done = False
        # self.boxes_logic()

    def L0_2_2_0_3_0_zshift_slider_change(self, a):
        # fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_review_viewer')
        # sliid = a['owner'].value
        # self.global_h.xanes3D_fiji_windows['xanes3D_review_viewer']['ip'].setSlice(sliid)
        # self.xanes3D_reg_review_done = False
        # self.boxes_logic()
        pass

    def L0_2_2_0_3_1_best_match_text_change(self, a):
        # self.xanes3D_reg_review_done = False
        # self.boxes_logic()
        pass

    def L0_2_2_0_3_2_record_button_click(self, a):
        with h5py.File(self.xanes3D_save_trial_reg_filename, 'r') as f:
            shift = f['/trial_registration/trial_reg_results/{0}/shift{0}'.format(str(self.xanes3D_alignment_pair_id).zfill(3))][:]

        best_match = self.global_h.xanes3D_fiji_windows['xanes3D_review_viewer']['ip'].currentSlice - 1
        self.xanes3D_review_shift_dict[str(self.xanes3D_alignment_pair_id)] = np.array([best_match - self.xanes3D_reg_sli_search_half_width,
                                                                                        shift[best_match][0],
                                                                                        shift[best_match][1]])
        self.hs['L[0][2][2][0][4][0]_confirm_review_results_text'].value = str(self.xanes3D_review_shift_dict)
        json.dump(self.xanes3D_review_shift_dict, open(self.xanes3D_reg_best_match_filename, 'w'), cls=NumpyArrayEncoder)
        self.xanes3D_reg_review_done = False
        self.boxes_logic()

    def L0_2_2_0_3_3_reg_pair_bad_button(self, a):
        self.xanes3D_review_bad_shift = True
        self.xanes3D_manual_xshift = 0
        self.xanes3D_manual_yshift = 0
        # self.xanes3D_manual_zshift = self.hs['L[0][2][2][0][3][1]_best_match_text'].value
        # self.hs['L[0][2][2][0][5][2]_review_manual_z_shift_text'].value = self.hs['L[0][2][2][0][3][1]_best_match_text'].value
        self.xanes3D_manual_zshift = self.global_h.xanes3D_fiji_windows['xanes3D_review_viewer']['ip'].currentSlice
        fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_review_manual_viewer')
        self.hs['L[0][2][2][0][5][2]_review_manual_z_shift_text'].value = self.global_h.xanes3D_fiji_windows['xanes3D_review_viewer']['ip'].currentSlice
        self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        self.xanes3D_reg_review_done = False
        self.boxes_logic()

    def L0_2_2_0_5_0_review_manual_x_shift_text(self, a):
        self.xanes3D_manual_xshift = self.hs['L[0][2][2][0][5][0]_review_manual_x_shift_text'].value
        self.xanes3D_manual_yshift = self.hs['L[0][2][2][0][5][1]_review_manual_y_shift_text'].value
        self.xanes3D_manual_zshift = self.hs['L[0][2][2][0][5][2]_review_manual_z_shift_text'].value
        xanes3D_review_aligned_img = (np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(self.trial_reg[self.xanes3D_manual_zshift-1]),
                                                                                    [self.xanes3D_manual_yshift, self.xanes3D_manual_xshift]))))

        # fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_review_manual_viewer')
        self.global_h.xanes3D_fiji_windows['xanes3D_review_manual_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
            self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(xanes3D_review_aligned_img-self.trial_reg_fixed)),
            self.global_h.ImagePlusClass))
        # time.sleep(0.2)
        self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        # self.boxes_logic()

    def L0_2_2_0_5_1_review_manual_y_shift_text(self, a):
        self.xanes3D_manual_xshift = self.hs['L[0][2][2][0][5][0]_review_manual_x_shift_text'].value
        self.xanes3D_manual_yshift = self.hs['L[0][2][2][0][5][1]_review_manual_y_shift_text'].value
        self.xanes3D_manual_zshift = self.hs['L[0][2][2][0][5][2]_review_manual_z_shift_text'].value
        xanes3D_review_aligned_img = (np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(self.trial_reg[self.xanes3D_manual_zshift-1]),
                                                                                [self.xanes3D_manual_yshift, self.xanes3D_manual_xshift]))))

        # fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_review_manual_viewer')
        self.global_h.xanes3D_fiji_windows['xanes3D_review_manual_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
            self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(xanes3D_review_aligned_img-self.trial_reg_fixed)),
            self.global_h.ImagePlusClass))
        # time.sleep(0.2)
        self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        # self.boxes_logic()

    def L0_2_2_0_5_2_review_manual_z_shift_text(self, a):
        self.xanes3D_manual_xshift = self.hs['L[0][2][2][0][5][0]_review_manual_x_shift_text'].value
        self.xanes3D_manual_yshift = self.hs['L[0][2][2][0][5][1]_review_manual_y_shift_text'].value
        self.xanes3D_manual_zshift = self.hs['L[0][2][2][0][5][2]_review_manual_z_shift_text'].value
        xanes3D_review_aligned_img = (np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(self.trial_reg[self.xanes3D_manual_zshift-1]),
                                                                                [self.xanes3D_manual_yshift, self.xanes3D_manual_xshift]))))

        # fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_review_manual_viewer')
        self.global_h.xanes3D_fiji_windows['xanes3D_review_manual_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
            self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(xanes3D_review_aligned_img-self.trial_reg_fixed)),
            self.global_h.ImagePlusClass))
        # time.sleep(0.2)
        self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        # self.boxes_logic()

    def L0_2_2_0_5_3_review_manual_shift_record_button(self, a):
        """
        temporarily this wont work with SR for now.
        """
        self.xanes3D_review_bad_shift = False
        with h5py.File(self.xanes3D_save_trial_reg_filename, 'r') as f:
            shift = f['/trial_registration/trial_reg_results/{0}/shift{0}'.format(str(self.xanes3D_alignment_pair_id).zfill(3))][:]

        best_match = self.xanes3D_manual_zshift - 1
        self.xanes3D_review_shift_dict["{}".format(self.xanes3D_alignment_pair_id)] = np.array([best_match - self.xanes3D_reg_sli_search_half_width,
                                                                                                self.xanes3D_manual_yshift+shift[best_match, 0],
                                                                                                self.xanes3D_manual_xshift+shift[best_match, 1]])
        self.hs['L[0][2][2][0][5][0]_review_manual_x_shift_text'].value = 0
        self.hs['L[0][2][2][0][5][1]_review_manual_y_shift_text'].value = 0
        self.hs['L[0][2][2][0][5][2]_review_manual_z_shift_text'].value = 1
        fiji_viewer_off(self.global_h, self, viewer_name='xanes3D_review_manual_viewer')
        json.dump(self.xanes3D_review_shift_dict, open(self.xanes3D_reg_best_match_filename, 'w'), cls=NumpyArrayEncoder)
        self.xanes3D_reg_review_done = False
        self.boxes_logic()

    def L0_2_2_0_4_1_confirm_review_results_button_click(self, a):
        if len(self.xanes3D_review_shift_dict) != (self.hs['L[0][2][2][0][2][0]_reg_pair_slider'].max+1):
            self.hs['L[0][2][2][0][4][0]_confirm_review_results_text'].value = 'reg review is not completed yet ...'
            idx = []
            offset = []
            for ii in sorted(self.xanes3D_review_shift_dict.keys()):
                offset.append(self.xanes3D_review_shift_dict[ii][0])
                idx.append(int(ii))
            plt.figure(1)
            plt.plot(idx, offset, 'b+')
            plt.xticks(np.arange(0, len(idx)+1, 5))
            plt.grid()
            self.xanes3D_reg_review_done = False
        else:
            fiji_viewer_off(self.global_h, self, viewer_name='xanes3D_review_manual_viewer')
            fiji_viewer_off(self.global_h, self, viewer_name='xanes3D_review_viewer')
            self.hs['L[0][2][2][0][4][0]_confirm_review_results_text'].value = 'reg review is done ...'
            self.xanes3D_reg_review_done = True
            self.update_xanes3D_config()
            json.dump(self.xanes3D_config, open(self.xanes3D_save_trial_reg_config_filename, 'w'), cls=NumpyArrayEncoder)
        self.boxes_logic()

    def L0_2_2_1_1_0_align_recon_optional_slice_checkbox(self, a):
        boxes = ['align_recon_optional_slice_start_text',
                 'align_recon_optional_slice_range_slider',
                 'align_recon_optional_slice_end_text']
        if a['owner'].value:
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
        else:
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)

    def L0_2_2_1_1_1_align_recon_optional_slice_start_text(self, a):
        if a['owner'].value <= self.hs['L[0][2][2][1][1][2]_align_recon_optional_slice_range_slider'].upper:
            self.hs['L[0][2][2][1][1][2]_align_recon_optional_slice_range_slider'].lower = a['owner'].value
        else:
            a['owner'].value = self.hs['L[0][2][2][1][1][2]_align_recon_optional_slice_range_slider'].upper

    def L0_2_2_1_1_3_align_recon_optional_slice_end_text(self, a):
        if a['owner'].value >= self.hs['L[0][2][2][1][1][2]_align_recon_optional_slice_range_slider'].lower:
            self.hs['L[0][2][2][1][1][2]_align_recon_optional_slice_range_slider'].upper = a['owner'].value
        else:
            a['owner'].value = self.hs['L[0][2][2][1][1][2]_align_recon_optional_slice_range_slider'].lower

    def L0_2_2_1_1_2_align_recon_optional_slice_range_slider(self, a):
        self.hs['L[0][2][2][1][1][1]_align_recon_optional_slice_start_text'].value = a['owner'].lower
        self.hs['L[0][2][2][1][1][3]_align_recon_optional_slice_end_text'].value = a['owner'].upper
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_virtural_stack_preview_viewer')
        if (not data_state) | (not viewer_state):
            fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_virtural_stack_preview_viewer')
        self.global_h.xanes3D_fiji_windows['xanes3D_virtural_stack_preview_viewer']['ip'].setSlice(a['owner'].lower - a['owner'].min + 1)
        self.global_h.xanes3D_fiji_windows['xanes3D_virtural_stack_preview_viewer']['ip'].setRoi(int(self.xanes3D_roi[2]), int(self.xanes3D_roi[0]),
                                                                                int(self.xanes3D_roi[3]-self.xanes3D_roi[2]),
                                                                                int(self.xanes3D_roi[1]-self.xanes3D_roi[0]))

    def L0_2_2_1_2_1_align_button_click(self, a):
        if self.hs['L[0][2][2][1][1][0]_align_recon_optional_slice_checkbox'].value:
            self.xanes3D_roi[4] = self.hs['L[0][2][2][1][1][2]_align_recon_optional_slice_range_slider'].lower
            self.xanes3D_roi[5] = self.hs['L[0][2][2][1][1][2]_align_recon_optional_slice_range_slider'].upper
        with h5py.File(self.xanes3D_save_trial_reg_filename, 'r') as f:
            recon_path_template = str(f['/trial_registration/data_directory_info/recon_path_template'][()])
            
        code = {}
        ln = 0
        code[ln] = "import xanes_regtools as xr"; ln+=1
        code[ln] = "reg = xr.regtools(dtype='3D_XANES', mode='TRANSLATION')"; ln+=1
        code[ln] = f"reg.set_xanes3D_recon_path_template('{recon_path_template}')"; ln+=1
        code[ln] = f"reg.set_roi({self.xanes3D_roi})"; ln+=1
        code[ln] = f"reg.apply_xanes3D_chunk_shift({self.xanes3D_review_shift_dict}, {self.xanes3D_roi[4]}, {self.xanes3D_roi[5]}, trialfn='{self.xanes3D_save_trial_reg_filename}', savefn='{self.xanes3D_save_trial_reg_filename}')"; ln+=1
        
        gen_external_py_script(self.xanes_external_command_name, code)
        sig = os.system(f"python '{self.xanes_external_command_name}'")
        if sig == 0:
            self.hs['L[0][2][2][1][2][0]_align_text'].value = 'XANES3D alignment is done ...'
        else:
            self.hs['L[0][2][2][1][2][0]_align_text'].value = 'something wrong in XANES3D alignment ...'

        self.update_xanes3D_config()
        json.dump(self.xanes3D_config, open(self.xanes3D_save_trial_reg_config_filename, 'w'), cls=NumpyArrayEncoder)
        self.boxes_logic()
        self.xanes3D_alignment_done = True

        with h5py.File(self.xanes3D_save_trial_reg_filename, 'r') as f:
            self.xanes_analysis_data_shape = f['/registration_results/reg_results/registered_xanes3D'].shape
            self.xanes_analysis_eng_list = f['/trial_registration/trial_reg_parameters/eng_list'][:]
            self.xanes3D_scan_id_s = f['/trial_registration/trial_reg_parameters/scan_ids'][0]
            self.xanes3D_scan_id_e = f['/trial_registration/trial_reg_parameters/scan_ids'][-1]

        # self.xanes3D_reg_best_match_filename = os.path.splitext(self.xanes3D_save_trial_reg_config_filename)[0].replace('config', 'reg_best_match') + '.json'
        self.xanes_element = determine_element(self.xanes_analysis_eng_list)
        tem = determine_fitting_energy_range(self.xanes_element)
        self.xanes_analysis_edge_eng = tem[0] 
        self.xanes_analysis_wl_fit_eng_s = tem[1]
        self.xanes_analysis_wl_fit_eng_e = tem[2]
        self.xanes_analysis_pre_edge_e = tem[3]
        self.xanes_analysis_post_edge_s = tem[4] 
        self.xanes_analysis_edge_0p5_fit_s = tem[5]
        self.xanes_analysis_edge_0p5_fit_e = tem[6]
                    
        self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].value = 0
        self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].value = 0
        self.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'].value = 'x-y-E'
        self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].description = 'E'
        self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].max = self.xanes_analysis_data_shape[0]-1
        self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].min = 0
        self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].description = 'z'
        self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].max = self.xanes_analysis_data_shape[1]-1
        self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].min = 0
        self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'XANES3D file config is done ...'
        self.xanes_analysis_type = 'full'
        self.xanes_fitting_gui_h.hs['L[0][x][3][0][1][0][0]_fit_eng_range_option_dropdown'].value = 'full'

        self.boxes_logic()

    def L0_2_2_2_3_0_visualize_viewer_option_togglebutton_change(self, a):
        self.xanes3D_visualization_viewer_option = a['owner'].value
        if self.xanes3D_visualization_viewer_option == 'napari':
            with h5py.File(self.xanes3D_save_trial_reg_filename, 'r') as f:
                self.xanes3D_aligned_data = 0
                self.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][:]
            self.viewer = napari.view_image(self.xanes3D_aligned_data)
        self.boxes_logic()

    def L0_2_2_2_3_1_visualize_view_alignment_option_dropdown(self, a):
        self.xanes_analysis_view_option = a['owner'].value

        with h5py.File(self.xanes3D_save_trial_reg_filename, 'r') as f:
            if self.xanes_analysis_view_option == 'x-y-E':
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].value = 0
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].description = 'E'
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].max = self.xanes_analysis_data_shape[0]-1
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].min = 0
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].value = 0
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].description = 'z'
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].max = self.xanes_analysis_data_shape[1]-1
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].min = 0
                self.xanes3D_aligned_data = 0
                self.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][0, :, :, :]
            elif self.xanes_analysis_view_option == 'y-z-E':
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].value = 0
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].description = 'E'
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].max = self.xanes_analysis_data_shape[0]-1
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].min = 0
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].value = 0
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].description = 'x'
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].max = self.xanes_analysis_data_shape[3]-1
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].min = 0
                self.xanes3D_aligned_data = 0
                self.xanes3D_aligned_data = np.swapaxes(f['/registration_results/reg_results/registered_xanes3D'][0, :, :, :], 0, 2)
            elif self.xanes_analysis_view_option == 'z-x-E':
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].value = 0
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].description = 'E'
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].max = self.xanes_analysis_data_shape[0]-1
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].min = 0
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].value = 0
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].description = 'y'
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].max = self.xanes_analysis_data_shape[2]-1
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].min = 0
                self.xanes3D_aligned_data = 0
                self.xanes3D_aligned_data = np.swapaxes(f['/registration_results/reg_results/registered_xanes3D'][0, :, :, :], 0, 1)
            elif self.xanes_analysis_view_option == 'x-y-z':
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].value = 0
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].description = 'z'
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].max = self.xanes_analysis_data_shape[1]-1
                self.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].min = 0
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].value = 0
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].description = 'E'
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].max = self.xanes_analysis_data_shape[0]-1
                self.hs['L[0][2][2][2][1][2]_visualize_view_alignment_4th_dim_slider'].min = 0
                self.xanes3D_aligned_data = 0
                self.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][:, 0, :, :]
        self.xanes_analysis_view_option_previous = self.xanes_analysis_view_option
        self.boxes_logic()

    def L0_2_2_2_1_3_visualize_view_alignment_slice_slider(self, a):
        fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_analysis_viewer')
        # self.boxes_logic()

    def L0_2_2_2_1_2_visualize_view_alignment_4th_dim_slider(self, a):
        self.xanes_analysis_slice = a['owner'].value
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_analysis_viewer')

        if not viewer_state:
            fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_analysis_viewer')
        with h5py.File(self.xanes3D_save_trial_reg_filename, 'r') as f:
            if self.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'].value == 'x-y-E':
                self.xanes3D_aligned_data = 0
                self.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][:, self.xanes_analysis_slice, :, :]
                self.hs['L[0][2][2][2][1][1]_visualize_alignment_eng_text'].value = self.xanes_analysis_eng_list[self.xanes_analysis_slice]
            elif self.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'].value == 'y-z-E':
                self.xanes3D_aligned_data = 0
                self.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][:, :, :, self.xanes_analysis_slice]
                self.hs['L[0][2][2][2][1][1]_visualize_alignment_eng_text'].value = self.xanes_analysis_eng_list[self.xanes_analysis_slice]
            elif self.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'].value == 'z-x-E':
                self.xanes3D_aligned_data = 0
                self.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][:, :, self.xanes_analysis_slice, :]
                self.hs['L[0][2][2][2][1][1]_visualize_alignment_eng_text'].value = self.xanes_analysis_eng_list[self.xanes_analysis_slice]
            elif self.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'].value == 'x-y-z':
                self.xanes3D_aligned_data = 0
                self.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][self.xanes_analysis_slice, :, :, :]
        self.xanes3D_fiji_aligned_data = self.global_h.ij.convert().convert(self.global_h.ij.dataset().create(
            self.global_h.ij.py.to_java(self.xanes3D_aligned_data)), self.global_h.ImagePlusClass)
        self.global_h.xanes3D_fiji_windows['xanes3D_analysis_viewer']['ip'].setImage(self.xanes3D_fiji_aligned_data)
        self.global_h.xanes3D_fiji_windows['xanes3D_analysis_viewer']['ip'].show()
        self.global_h.xanes3D_fiji_windows['xanes3D_analysis_viewer']['ip'].setTitle(f"{a['owner'].description} slice: {a['owner'].value}")
        self.global_h.ij.py.run_macro("""run("Collect Garbage")""")
        self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        # self.boxes_logic()

    def L0_2_2_2_2_1_visualize_spec_view_mem_monitor_checkbox(self, a):
        if a['owner'].value:
            self.global_h.ij.py.run_macro("""run("Monitor Memory...")""")

    def L0_2_2_2_2_2_visualize_spec_view_in_roi_button(self, a):
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='xanes3D_analysis_viewer')
        if not viewer_state:
            fiji_viewer_on(self.global_h, self, viewer_name='xanes3D_analysis_viewer')
        width = self.global_h.xanes3D_fiji_windows['xanes3D_analysis_viewer']['ip'].width
        height = self.global_h.xanes3D_fiji_windows['xanes3D_analysis_viewer']['ip'].height
        roi = [int((width-10)/2), int((height-10)/2), 10, 10]
        self.global_h.xanes3D_fiji_windows['xanes3D_analysis_viewer']['ip'].setRoi(roi[0], roi[1], roi[2], roi[3])
        self.global_h.ij.py.run_macro("""run("Plot Z-axis Profile")""")
        self.global_h.ij.py.run_macro("""Plot.setStyle(0, "black,none,1.0,Connected Circles")""")
        self.global_h.xanes3D_fiji_windows['analysis_viewer_z_plot_viewer']['ip'] = self.global_h.WindowManager.getCurrentImage()
        self.global_h.xanes3D_fiji_windows['analysis_viewer_z_plot_viewer']['fiji_id'] = self.global_h.WindowManager.getIDList()[-1]
        self.hs['L[0][2][2][2][2][0]_visualize_spec_view_text'].value = 'drag the roi box to check the spectrum at different locations ...'
        self.boxes_logic()

