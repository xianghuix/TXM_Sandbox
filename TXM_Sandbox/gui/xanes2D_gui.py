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
from .gui_components import (
    SelectFilesButton,
    NumpyArrayEncoder,
    get_handles,
    enable_disable_boxes,
    fiji_viewer_state,
    restart,
    fiji_viewer_on,
    fiji_viewer_off,
    determine_element,
    determine_fitting_energy_range,
    gen_external_py_script,
    update_json_content,
)

napari.gui_qt()


class xanes2D_tools_gui:

    def __init__(self, parent_h, form_sz=[650, 740]):
        self.gui_name = "xanes2D"
        self.form_sz = form_sz
        self.global_h = parent_h
        self.hs = {}

        if self.global_h.io_xanes2D_cfg["use_h5_reader"]:
            self.reader = data_reader(xanes2D_h5_reader)
        else:
            from ..external.user_io import user_xanes2D_reader

            self.reader = data_reader(user_xanes2D_reader)

        self.xanes_raw_fn_temp = self.global_h.io_xanes2D_cfg[
            "xanes2D_raw_fn_template"]

        self.xanes_fit_external_command_name = os.path.join(
            self.global_h.script_dir, "xanes2D_fit_external_command.py")
        self.xanes_reg_external_command_name = os.path.join(
            self.global_h.script_dir, "xanes2D_reg_external_command.py")
        self.xanes_align_external_command_name = os.path.join(
            self.global_h.script_dir, "xanes2D_align_external_command.py")

        self.xanes_file_configured = False
        self.xanes_data_configured = False
        self.xanes_roi_configured = False
        self.xanes_reg_params_configured = False
        self.xanes_reg_done = False
        self.xanes_reg_review_done = False
        self.xanes_alignment_done = False
        self.xanes_fit_eng_configured = False
        self.xanes_review_read_alignment_option = False

        self.xanes_file_raw_h5_set = False
        self.xanes_file_save_trial_set = False
        self.xanes_file_reg_file_set = False
        self.xanes_file_config_file_set = False
        self.xanes_config_alternative_flat_set = False
        self.xanes_config_raw_img_readed = False
        self.xanes_regparams_anchor_idx_set = False
        self.xanes_file_reg_file_readed = False
        self.xanes_fit_eng_set = False

        self.xanes_config_is_raw = True
        self.xanes_config_img_scalar = 1
        self.xanes_config_use_smooth_flat = False
        self.xanes_config_smooth_flat_sigma = 0
        self.xanes_config_use_alternative_flat = False
        self.xanes_config_eng_list = None

        self.xanes_file_analysis_option = "Do New Reg"
        self.xanes_file_raw_h5_filename = None
        self.xanes_save_trial_reg_filename = None
        self.xanes_file_save_trial_reg_config_filename = None
        self.xanes_config_alternative_flat_filename = None
        self.xanes_review_reg_best_match_filename = None

        self.xanes_reg_use_chunk = True
        self.xanes_reg_anchor_idx = 0
        self.xanes_reg_roi = [0, 10, 0, 10]
        self.xanes_reg_use_mask = False
        self.xanes_reg_mask = None
        self.xanes_reg_mask_dilation_width = 0
        self.xanes_reg_mask_thres = 0
        self.xanes_reg_use_smooth_img = False
        self.xanes_reg_smooth_img_sigma = 5
        self.xanes_reg_chunk_sz = None
        self.xanes_reg_method = None
        self.xanes_reg_ref_mode = None
        self.xanes_reg_mrtv_level = 4
        self.xanes_reg_mrtv_width = 10
        self.xanes_reg_mrtv_subpixel_kernel = 0.2

        self.xanes_visualization_auto_bc = False

        self.xanes_img = None
        self.xanes_img_roi = None
        self.xanes_review_aligned_img_original = None
        self.xanes_use_existing_reg_reviewed = False
        self.xanes_reg_file_readed = False
        self.xanes_reg_review_file = None
        self.xanes_review_aligned_img = None
        self.xanes_review_fixed_img = None
        self.xanes_review_bad_shift = False
        self.xanes_review_shift_dict = None
        self.xanes_manual_xshift = 0
        self.xanes_manual_yshift = 0
        self.xanes_review_shift_dict = {}

        self.xanes_file_analysis_option = "Do New Reg"
        self.xanes_eng_id_s = 0
        self.xanes_eng_id_e = 1

        self.xanes_element = None
        self.xanes_fit_eng_list = None
        self.xanes_fit_type = "wl"
        self.xanes_fit_edge_eng = 0
        self.xanes_fit_wl_fit_eng_s = 0
        self.xanes_fit_wl_fit_eng_e = 0
        self.xanes_fit_pre_edge_e = -50
        self.xanes_fit_post_edge_s = 100
        self.xanes_fit_edge_0p5_fit_s = 0
        self.xanes_fit_edge_0p5_fit_e = 0
        self.xanes_fit_spectrum = None
        self.xanes_fit_use_mask = False
        self.xanes_fit_mask_thres = None
        self.xanes_fit_mask_img_id = None
        self.xanes_fit_mask = 1
        self.xanes_fit_edge_jump_thres = 1.0
        self.xanes_fit_edge_offset_thres = 1.0

        self.xanes_config = {
            "filepath config": {
                "xanes2D_file_raw_h5_filename":
                self.xanes_file_raw_h5_filename,
                "xanes_save_trial_reg_filename":
                self.xanes_save_trial_reg_filename,
                "xanes2D_file_save_trial_reg_config_filename":
                self.xanes_file_save_trial_reg_config_filename,
                "xanes2D_review_reg_best_match_filename":
                self.xanes_review_reg_best_match_filename,
                "xanes2D_file_analysis_option":
                self.xanes_file_analysis_option,
                "xanes2D_file_configured": self.xanes_file_configured,
                "xanes2D_file_raw_h5_set": self.xanes_file_raw_h5_set,
                "xanes2D_file_save_trial_set": self.xanes_file_save_trial_set,
                "xanes2D_file_reg_file_set": self.xanes_file_reg_file_set,
                "xanes2D_file_config_file_set":
                self.xanes_file_config_file_set,
                "xanes2D_file_reg_file_readed":
                self.xanes_file_reg_file_readed,
            },
            "data_config": {
                "xanes2D_config_is_raw": self.xanes_config_is_raw,
                "xanes2D_config_img_scalar": self.xanes_config_img_scalar,
                "xanes2D_config_use_alternative_flat":
                self.xanes_config_use_alternative_flat,
                "xanes2D_config_use_smooth_flat":
                self.xanes_config_use_smooth_flat,
                "xanes2D_config_smooth_flat_sigma":
                self.xanes_config_smooth_flat_sigma,
                "xanes2D_config_alternative_flat_filename":
                self.xanes_config_alternative_flat_filename,
                "xanes2D_config_alternative_flat_set":
                self.xanes_config_alternative_flat_set,
                "xanes2D_config_eng_points_range_s": self.xanes_eng_id_s,
                "xanes2D_config_eng_points_range_e": self.xanes_eng_id_e,
                "xanes2D_config_eng_s": 0,
                "xanes2D_config_eng_e": 0,
                "xanes2D_config_fiji_view_on": False,
                "xanes2D_config_img_num": 0,
                "xanes2D_config_raw_img_readed":
                self.xanes_config_raw_img_readed,
                "xanes2D_config_norm_scale_text_min": 0,
                "xanes2D_config_norm_scale_text_val": 1,
                "xanes2D_config_norm_scale_text_max": 10,
                "xanes2D_config_smooth_flat_sigma_text.min": 0,
                "xanes2D_config_smooth_flat_sigma_text.val": 0,
                "xanes2D_config_smooth_flat_sigma_text.max": 30,
                "xanes2D_config_eng_points_range_slider_min": 0,
                "xanes2D_config_eng_points_range_slider_val": 0,
                "xanes2D_config_eng_points_range_slider_max": 0,
                "xanes2D_config_eng_s_text_min": 0,
                "xanes2D_config_eng_s_text_val": 0,
                "xanes2D_config_eng_s_text_max": 0,
                "xanes2D_config_eng_e_text_min": 0,
                "xanes2D_config_eng_e_text_val": 0,
                "xanes2D_config_eng_e_text_max": 0,
                "xanes2D_config_fiji_eng_id_slider_min": 0,
                "xanes2D_config_fiji_eng_id_slider_val": 0,
                "xanes2D_config_fiji_eng_id_slider_max": 0,
                "xanes2D_data_configured": self.xanes_data_configured,
            },
            "roi_config": {
                "2D_roi_x_slider_min": 0,
                "2D_roi_x_slider_val": 0,
                "2D_roi_x_slider_max": 0,
                "2D_roi_y_slider_min": 0,
                "2D_roi_y_slider_val": 0,
                "2D_roi_y_slider_max": 0,
                "2D_roi": list(self.xanes_reg_roi),
                "xanes2D_roi_configured": self.xanes_roi_configured,
            },
            "registration_config": {
                "xanes2D_regparams_fiji_viewer":
                True,
                "xanes2D_regparams_use_chunk":
                True,
                "xanes2D_regparams_anchor_id":
                0,
                "xanes2D_regparams_use_mask":
                True,
                "xanes2D_regparams_mask_thres":
                0,
                "xanes2D_regparams_mask_dilation":
                0,
                "xanes2D_regparams_use_smooth_img":
                False,
                "xanes2D_regparams_smooth_sigma":
                5,
                "xanes2D_regparams_chunk_sz":
                7,
                "xanes2D_regparams_reg_method":
                "MPC",
                "xanes2D_regparams_ref_mode":
                "single",
                "xanes2D_regparams_anchor_id_slider_min":
                0,
                "xanes2D_regparams_anchor_id_slider_val":
                0,
                "xanes2D_regparams_anchor_id_slider_max":
                0,
                "xanes2D_regparams_mask_thres_slider_min":
                0,
                "xanes2D_regparams_mask_thres_slider_val":
                0,
                "xanes2D_regparams_mask_thres_slider_max":
                0,
                "xanes2D_regparams_mask_dilation_slider.min":
                0,
                "xanes2D_regparams_mask_dilation_slider.val":
                0,
                "xanes2D_regparams_mask_dilation_slider.max":
                0,
                "xanes2D_regparams_smooth_sigma_text.min":
                0,
                "xanes2D_regparams_smooth_sigma_text.val":
                0,
                "xanes2D_regparams_smooth_sigma_text.max":
                0,
                "xanes2D_regparams_chunk_sz_slider.min":
                0,
                "xanes2D_regparams_chunk_sz_slider.val":
                0,
                "xanes2D_regparams_chunk_sz_slider.max":
                0,
                "xanes2D_regparams_reg_method_dropdown_options":
                ("MPC", "PC", "SR"),
                "xanes2D_regparams_ref_mode_dropdown_options":
                ("single", "neighbour"),
                "xanes2D_regparams_mrtv_level":
                4,
                "xanes2D_regparams_mrtv_width":
                10,
                "xanes2D_regparams_mrtv_subpixel_kernel":
                3,
                "xanes2D_regparams_mrtv_subpixel_width":
                8,
                "xanes2D_regparams_configured":
                self.xanes_reg_params_configured,
            },
            "run registration": {
                "xanes2D_reg_done": self.xanes_reg_done
            },
            "review registration": {
                "xanes2D_review_use_existing_reg_reviewed":
                self.xanes_review_read_alignment_option,
                "xanes2D_reviewed_reg_shift": self.xanes_review_shift_dict,
                "reg_pair_slider_min": 0,
                "reg_pair_slider_val": 0,
                "reg_pair_slider_max": 0,
                "xshift_text_val": 0,
                "yshift_text_val": 0,
                "xanes2D_reg_review_done": self.xanes_reg_review_done,
            },
            "align 2D recon": {
                "xanes2D_alignment_done": self.xanes_alignment_done,
                "xanes2D_analysis_edge_eng": self.xanes_fit_edge_eng,
                "xanes2D_analysis_wl_fit_eng_s": self.xanes_fit_wl_fit_eng_s,
                "xanes2D_analysis_wl_fit_eng_e": self.xanes_fit_wl_fit_eng_e,
                "xanes2D_analysis_pre_edge_e": self.xanes_fit_pre_edge_e,
                "xanes2D_analysis_post_edge_s": self.xanes_fit_post_edge_s,
                "xanes2D_analysis_edge_0p5_fit_s":
                self.xanes_fit_edge_0p5_fit_s,
                "xanes2D_analysis_edge_0p5_fit_e":
                self.xanes_fit_edge_0p5_fit_e,
                "xanes2D_analysis_type": self.xanes_fit_type,
            },
        }
        self.xanes_fit_gui_h = xfg.xanes_fitting_gui(self,
                                                     form_sz=self.form_sz)
        self.xanes_fit_gui_h.build_gui()
        self.xanes_ana_gui_h = xag.xanes_analysis_gui(self,
                                                      form_sz=self.form_sz)
        self.xanes_ana_gui_h.build_gui()

    def lock_message_text_boxes(self):
        boxes = [
            "SelRawH5Path text",
            "SelSaveTrial text",
            "CfmFile&Path text",
            "EngStart text",
            "EngEnd text",
            "CfmConfigData text",
            "CfmRoi text",
            "CfmRegParams text",
            "RunReg text",
            "CfmRevRlt text",
            "AlignImg text",
            "VisEng text",
        ]
        enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        self.xanes_fit_gui_h.hs["FitRun text"].disabled = True

    def update_xanes2D_config(self):
        tem = {}
        for key, item in self.xanes_review_shift_dict.items():
            tem[key] = list(item)
        self.xanes_config = {
            "filepath config": {
                "xanes2D_file_raw_h5_filename":
                self.xanes_file_raw_h5_filename,
                "xanes_save_trial_reg_filename":
                self.xanes_save_trial_reg_filename,
                "xanes2D_file_save_trial_reg_config_filename":
                self.xanes_file_save_trial_reg_config_filename,
                "xanes2D_review_reg_best_match_filename":
                self.xanes_review_reg_best_match_filename,
                "xanes2D_file_analysis_option":
                self.xanes_file_analysis_option,
                "xanes2D_file_configured": self.xanes_file_configured,
                "xanes2D_file_raw_h5_set": self.xanes_file_raw_h5_set,
                "xanes2D_file_save_trial_set": self.xanes_file_save_trial_set,
                "xanes2D_file_reg_file_set": self.xanes_file_reg_file_set,
                "xanes2D_file_config_file_set":
                self.xanes_file_config_file_set,
                "xanes2D_file_reg_file_readed":
                self.xanes_file_reg_file_readed,
            },
            "data_config": {
                "xanes2D_config_is_raw":
                self.xanes_config_is_raw,
                "xanes2D_config_img_scalar":
                self.xanes_config_img_scalar,
                "xanes2D_config_use_alternative_flat":
                self.xanes_config_use_alternative_flat,
                "xanes2D_config_use_smooth_flat":
                self.xanes_config_use_smooth_flat,
                "xanes2D_config_smooth_flat_sigma":
                self.xanes_config_smooth_flat_sigma,
                "xanes2D_config_alternative_flat_filename":
                self.xanes_config_alternative_flat_filename,
                "xanes2D_config_alternative_flat_set":
                self.xanes_config_alternative_flat_set,
                "xanes2D_config_eng_points_range_s":
                self.xanes_eng_id_s,
                "xanes2D_config_eng_points_range_e":
                self.xanes_eng_id_e,
                "xanes2D_config_eng_s":
                self.hs["EngStart text"].value,
                "xanes2D_config_eng_e":
                self.hs["EngEnd text"].value,
                "xanes2D_config_fiji_view_on":
                self.hs["FijiRawImgPrev chbx"].value,
                "xanes2D_config_img_num":
                self.hs["FijiEngId sldr"].value,
                "xanes2D_config_norm_scale_text_min":
                self.hs["NormScale text"].min,
                "xanes2D_config_norm_scale_text_val":
                self.hs["NormScale text"].value,
                "xanes2D_config_norm_scale_text_max":
                self.hs["NormScale text"].max,
                "xanes2D_config_eng_points_range_slider_min":
                self.hs["EngPointsRange sldr"].min,
                "xanes2D_config_eng_points_range_slider_val":
                self.hs["EngPointsRange sldr"].value,
                "xanes2D_config_eng_points_range_slider_max":
                self.hs["EngPointsRange sldr"].max,
                "xanes2D_config_eng_s_text_min":
                self.hs["EngStart text"].min,
                "xanes2D_config_eng_s_text_val":
                self.hs["EngStart text"].value,
                "xanes2D_config_eng_s_text_max":
                self.hs["EngStart text"].max,
                "xanes2D_config_eng_e_text_min":
                self.hs["EngEnd text"].min,
                "xanes2D_config_eng_e_text_val":
                self.hs["EngEnd text"].value,
                "xanes2D_config_eng_e_text_max":
                self.hs["EngEnd text"].max,
                "xanes2D_config_fiji_eng_id_slider_min":
                self.hs["FijiEngId sldr"].min,
                "xanes2D_config_fiji_eng_id_slider_val":
                self.hs["FijiEngId sldr"].value,
                "xanes2D_config_fiji_eng_id_slider_max":
                self.hs["FijiEngId sldr"].max,
                "xanes2D_config_raw_img_readed":
                self.xanes_config_raw_img_readed,
                "xanes2D_data_configured":
                self.xanes_data_configured,
            },
            "roi_config": {
                "2D_roi_x_slider_min": self.hs["2DRoiX sldr"].min,
                "2D_roi_x_slider_val": self.hs["2DRoiX sldr"].value,
                "2D_roi_x_slider_max": self.hs["2DRoiX sldr"].max,
                "2D_roi_y_slider_min": self.hs["2DRoiY sldr"].min,
                "2D_roi_y_slider_val": self.hs["2DRoiY sldr"].value,
                "2D_roi_y_slider_max": self.hs["2DRoiY sldr"].max,
                "2D_roi": list(self.xanes_reg_roi),
                "xanes2D_roi_configured": self.xanes_roi_configured,
            },
            "registration_config": {
                "xanes2D_regparams_fiji_viewer":
                self.hs["FijiMaskViewer chbx"].value,
                "xanes2D_regparams_use_chunk":
                self.hs["UseChunk chbx"].value,
                "xanes2D_regparams_anchor_id":
                self.hs["AnchorId sldr"].value,
                "xanes2D_regparams_use_mask":
                self.hs["UseMask chbx"].value,
                "xanes2D_regparams_mask_thres":
                self.hs["MaskThres sldr"].value,
                "xanes2D_regparams_mask_dilation":
                self.hs["MaskDilation sldr"].value,
                "xanes2D_regparams_use_smooth_img":
                self.hs["UseSmooth chbx"].value,
                "xanes2D_regparams_smooth_sigma":
                self.hs["SmoothSigma text"].value,
                "xanes2D_regparams_chunk_sz":
                self.hs["ChunkSz sldr"].value,
                "xanes2D_regparams_reg_method":
                self.hs["RegMethod drpdn"].value,
                "xanes2D_regparams_ref_mode":
                self.hs["RefMode drpdn"].value,
                "xanes2D_regparams_anchor_id_slider_min":
                self.hs["AnchorId sldr"].min,
                "xanes2D_regparams_anchor_id_slider_val":
                self.hs["AnchorId sldr"].value,
                "xanes2D_regparams_anchor_id_slider_max":
                self.hs["AnchorId sldr"].max,
                "xanes2D_regparams_mask_thres_slider_min":
                self.hs["MaskThres sldr"].min,
                "xanes2D_regparams_mask_thres_slider_val":
                self.hs["MaskThres sldr"].value,
                "xanes2D_regparams_mask_thres_slider_max":
                self.hs["MaskThres sldr"].max,
                "xanes2D_regparams_mask_dilation_slider.min":
                self.hs["MaskDilation sldr"].min,
                "xanes2D_regparams_mask_dilation_slider.val":
                self.hs["MaskDilation sldr"].value,
                "xanes2D_regparams_mask_dilation_slider.max":
                self.hs["MaskDilation sldr"].max,
                "xanes2D_regparams_smooth_sigma_text.min":
                self.hs["SmoothSigma text"].min,
                "xanes2D_regparams_smooth_sigma_text.val":
                self.hs["SmoothSigma text"].value,
                "xanes2D_regparams_smooth_sigma_text.max":
                self.hs["SmoothSigma text"].max,
                "xanes2D_regparams_chunk_sz_slider.min":
                self.hs["ChunkSz sldr"].min,
                "xanes2D_regparams_chunk_sz_slider.val":
                self.hs["ChunkSz sldr"].value,
                "xanes2D_regparams_chunk_sz_slider.max":
                self.hs["ChunkSz sldr"].max,
                # "xanes2D_regparams_reg_method_dropdown_options":self.hs['RegMethod drpdn'].options,
                "xanes2D_regparams_ref_mode_dropdown_options":
                self.hs["RefMode drpdn"].options,
                "xanes2D_regparams_mrtv_level":
                self.hs["MrtvLevel text"].value,
                "xanes2D_regparams_mrtv_width":
                self.hs["MrtvWz text"].value,
                "xanes2D_regparams_mrtv_subpixel_kernel":
                self.hs["MrtvSubpixelKernel text"].value,
                "xanes2D_regparams_mrtv_subpixel_width":
                self.hs["MrtvSubpixelWz text"].value,
                "xanes2D_regparams_configured":
                self.xanes_reg_params_configured,
            },
            "run registration": {
                "xanes2D_reg_done": self.xanes_reg_done
            },
            "review registration": {
                "xanes2D_review_use_existing_reg_reviewed":
                self.hs["ReadAlign chbx"].value,
                "xanes2D_reviewed_reg_shift":
                tem,
                "reg_pair_slider_min":
                self.hs["RegPair sldr"].min,
                "reg_pair_slider_val":
                self.hs["RegPair sldr"].value,
                "reg_pair_slider_max":
                self.hs["RegPair sldr"].max,
                "xshift_text_val":
                self.hs["XShift text"].value,
                "yshift_text_val":
                self.hs["YShift text"].value,
                "xanes2D_reg_review_done":
                self.xanes_reg_review_done,
            },
            "align 2D recon": {
                "xanes2D_alignment_done": self.xanes_alignment_done,
                "xanes2D_analysis_edge_eng": self.xanes_fit_edge_eng,
                "xanes2D_analysis_wl_fit_eng_s": self.xanes_fit_wl_fit_eng_s,
                "xanes2D_analysis_wl_fit_eng_e": self.xanes_fit_wl_fit_eng_e,
                "xanes2D_analysis_pre_edge_e": self.xanes_fit_pre_edge_e,
                "xanes2D_analysis_post_edge_s": self.xanes_fit_post_edge_s,
                "xanes2D_analysis_edge_0p5_fit_s":
                self.xanes_fit_edge_0p5_fit_s,
                "xanes2D_analysis_edge_0p5_fit_e":
                self.xanes_fit_edge_0p5_fit_e,
                "xanes2D_analysis_type": self.xanes_fit_type,
            },
        }

    def read_xanes2D_config(self):
        with open(self.xanes_file_save_trial_reg_config_filename_original,
                  "r") as f:
            self.xanes_config = json.load(f)

    def set_xanes2D_variables(self):
        self.xanes_file_raw_h5_filename = self.xanes_config["filepath config"][
            "xanes2D_file_raw_h5_filename"]
        self.xanes_save_trial_reg_filename = self.xanes_config[
            "filepath config"]["xanes_save_trial_reg_filename"]
        self.xanes_review_reg_best_match_filename = self.xanes_config[
            "filepath config"]["xanes2D_review_reg_best_match_filename"]
        self.xanes_file_raw_h5_set = self.xanes_config["filepath config"][
            "xanes2D_file_raw_h5_set"]
        self.xanes_file_save_trial_set = self.xanes_config["filepath config"][
            "xanes2D_file_save_trial_set"]
        self.xanes_file_reg_file_set = self.xanes_config["filepath config"][
            "xanes2D_file_reg_file_set"]
        self.xanes_file_config_file_set = self.xanes_config["filepath config"][
            "xanes2D_file_config_file_set"]
        self.xanes_file_configured = self.xanes_config["filepath config"][
            "xanes2D_file_configured"]

        self.xanes_config_is_raw = self.xanes_config["data_config"][
            "xanes2D_config_is_raw"]
        self.xanes_config_img_scalar = self.xanes_config["data_config"][
            "xanes2D_config_img_scalar"]
        self.xanes_config_use_alternative_flat = self.xanes_config[
            "data_config"]["xanes2D_config_use_alternative_flat"]
        self.xanes_config_use_smooth_flat = self.xanes_config["data_config"][
            "xanes2D_config_use_smooth_flat"]
        self.xanes_config_smooth_flat_sigma = self.xanes_config["data_config"][
            "xanes2D_config_smooth_flat_sigma"]
        self.xanes_config_alternative_flat_filename = self.xanes_config[
            "data_config"]["xanes2D_config_alternative_flat_filename"]
        self.xanes_config_alternative_flat_set = self.xanes_config[
            "data_config"]["xanes2D_config_alternative_flat_set"]
        self.xanes_eng_id_s = self.xanes_config["data_config"][
            "xanes2D_config_eng_points_range_s"]
        self.xanes_eng_id_e = self.xanes_config["data_config"][
            "xanes2D_config_eng_points_range_e"]
        self.xanes_config_raw_img_readed = self.xanes_config["data_config"][
            "xanes2D_config_raw_img_readed"]
        self.xanes_data_configured = self.xanes_config["data_config"][
            "xanes2D_data_configured"]

        self.xanes_reg_roi = self.xanes_config["roi_config"]["2D_roi"]
        self.xanes_roi_configured = self.xanes_config["roi_config"][
            "xanes2D_roi_configured"]
        self.xanes_reg_use_chunk = self.xanes_config["registration_config"][
            "xanes2D_regparams_use_chunk"]
        self.xanes_reg_anchor_idx = self.xanes_config["registration_config"][
            "xanes2D_regparams_anchor_id"]
        self.xanes_reg_use_mask = self.xanes_config["registration_config"][
            "xanes2D_regparams_use_mask"]
        self.xanes_reg_mask_thres = self.xanes_config["registration_config"][
            "xanes2D_regparams_mask_thres"]
        self.xanes_reg_mask_dilation_width = self.xanes_config[
            "registration_config"]["xanes2D_regparams_mask_dilation"]
        self.xanes_reg_use_smooth_img = self.xanes_config[
            "registration_config"]["xanes2D_regparams_use_smooth_img"]
        self.xanes_reg_smooth_img_sigma = self.xanes_config[
            "registration_config"]["xanes2D_regparams_smooth_sigma"]
        self.xanes_reg_chunk_sz = self.xanes_config["registration_config"][
            "xanes2D_regparams_chunk_sz"]
        self.xanes_reg_method = self.xanes_config["registration_config"][
            "xanes2D_regparams_reg_method"]
        self.xanes_reg_ref_mode = self.xanes_config["registration_config"][
            "xanes2D_regparams_ref_mode"]

        self.xanes_reg_mrtv_level = self.xanes_config["registration_config"][
            "xanes2D_regparams_mrtv_level"]
        self.xanes_reg_mrtv_width = self.xanes_config["registration_config"][
            "xanes2D_regparams_mrtv_width"]
        self.xanes_reg_mrtv_subpixel_kernel = self.xanes_config[
            "registration_config"]["xanes2D_regparams_mrtv_subpixel_kernel"]
        self.xanes_reg_mrtv_subpixel_wz = self.xanes_config[
            "registration_config"]["xanes2D_regparams_mrtv_subpixel_width"]
        self.xanes_reg_params_configured = self.xanes_config[
            "registration_config"]["xanes2D_regparams_configured"]

        self.xanes_reg_done = self.xanes_config["run registration"][
            "xanes2D_reg_done"]

        self.xanes_review_read_alignment_option = self.xanes_config[
            "review registration"]["xanes2D_review_use_existing_reg_reviewed"]
        tem = self.xanes_config["review registration"][
            "xanes2D_reviewed_reg_shift"]
        self.xanes_review_shift_dict = {}
        for key, item in tem.items():
            self.xanes_review_shift_dict[key] = np.array(item)
        self.xanes_reg_review_done = self.xanes_config["review registration"][
            "xanes2D_reg_review_done"]

        self.xanes_alignment_done = self.xanes_config["align 2D recon"][
            "xanes2D_alignment_done"]
        self.xanes_fit_edge_eng = self.xanes_config["align 2D recon"][
            "xanes2D_analysis_edge_eng"]
        self.xanes_fit_wl_fit_eng_s = self.xanes_config["align 2D recon"][
            "xanes2D_analysis_wl_fit_eng_s"]
        self.xanes_fit_wl_fit_eng_e = self.xanes_config["align 2D recon"][
            "xanes2D_analysis_wl_fit_eng_e"]
        self.xanes_fit_pre_edge_e = self.xanes_config["align 2D recon"][
            "xanes2D_analysis_pre_edge_e"]
        self.xanes_fit_post_edge_s = self.xanes_config["align 2D recon"][
            "xanes2D_analysis_post_edge_s"]
        self.xanes_fit_edge_0p5_fit_s = self.xanes_config["align 2D recon"][
            "xanes2D_analysis_edge_0p5_fit_s"]
        self.xanes_fit_edge_0p5_fit_e = self.xanes_config["align 2D recon"][
            "xanes2D_analysis_edge_0p5_fit_e"]
        self.xanes_fit_type = self.xanes_config["align 2D recon"][
            "xanes2D_analysis_type"]

    def set_xanes2D_handles(self):
        self.hs["IsRaw chbx"].value = self.xanes_config_is_raw

        self.hs["NormScale text"].value = self.xanes_config_img_scalar
        self.hs["EngPointsRange sldr"].max = self.xanes_config["data_config"][
            "xanes2D_config_eng_points_range_slider_max"]
        self.hs["EngPointsRange sldr"].min = self.xanes_config["data_config"][
            "xanes2D_config_eng_points_range_slider_min"]
        self.hs["EngPointsRange sldr"].value = self.xanes_config[
            "data_config"]["xanes2D_config_eng_points_range_slider_val"]
        self.hs["EngStart text"].max = self.xanes_config["data_config"][
            "xanes2D_config_eng_s_text_max"]
        self.hs["EngStart text"].min = self.xanes_config["data_config"][
            "xanes2D_config_eng_s_text_min"]
        self.hs["EngStart text"].value = self.xanes_config["data_config"][
            "xanes2D_config_eng_s_text_val"]
        self.hs["EngEnd text"].max = self.xanes_config["data_config"][
            "xanes2D_config_eng_e_text_max"]
        self.hs["EngEnd text"].min = self.xanes_config["data_config"][
            "xanes2D_config_eng_e_text_min"]
        self.hs["EngEnd text"].value = self.xanes_config["data_config"][
            "xanes2D_config_eng_e_text_val"]
        self.hs["FijiEngId sldr"].max = self.xanes_config["data_config"][
            "xanes2D_config_fiji_eng_id_slider_max"]
        self.hs["FijiEngId sldr"].min = self.xanes_config["data_config"][
            "xanes2D_config_fiji_eng_id_slider_min"]
        self.hs["FijiEngId sldr"].value = self.xanes_config["data_config"][
            "xanes2D_config_fiji_eng_id_slider_val"]

        self.hs["2DRoiX sldr"].max = self.xanes_config["roi_config"][
            "2D_roi_x_slider_max"]
        self.hs["2DRoiX sldr"].min = self.xanes_config["roi_config"][
            "2D_roi_x_slider_min"]
        self.hs["2DRoiX sldr"].value = self.xanes_config["roi_config"][
            "2D_roi_x_slider_val"]
        self.hs["2DRoiY sldr"].max = self.xanes_config["roi_config"][
            "2D_roi_y_slider_max"]
        self.hs["2DRoiY sldr"].min = self.xanes_config["roi_config"][
            "2D_roi_y_slider_min"]
        self.hs["2DRoiY sldr"].value = self.xanes_config["roi_config"][
            "2D_roi_y_slider_val"]

        self.hs["AnchorId sldr"].max = self.xanes_config[
            "registration_config"]["xanes2D_regparams_anchor_id_slider_max"]
        self.hs["AnchorId sldr"].min = self.xanes_config[
            "registration_config"]["xanes2D_regparams_anchor_id_slider_min"]
        self.hs["AnchorId sldr"].value = self.xanes_config[
            "registration_config"]["xanes2D_regparams_anchor_id_slider_val"]
        self.hs["UseMask chbx"].value = self.xanes_config[
            "registration_config"]["xanes2D_regparams_use_mask"]
        self.hs["MaskThres sldr"].max = self.xanes_config[
            "registration_config"]["xanes2D_regparams_mask_thres_slider_max"]
        self.hs["MaskThres sldr"].min = self.xanes_config[
            "registration_config"]["xanes2D_regparams_mask_thres_slider_min"]
        self.hs["MaskThres sldr"].value = self.xanes_config[
            "registration_config"]["xanes2D_regparams_mask_thres_slider_val"]
        self.hs["MaskDilation sldr"].max = self.xanes_config[
            "registration_config"][
                "xanes2D_regparams_mask_dilation_slider.max"]
        self.hs["MaskDilation sldr"].min = self.xanes_config[
            "registration_config"][
                "xanes2D_regparams_mask_dilation_slider.min"]
        self.hs["MaskDilation sldr"].value = self.xanes_config[
            "registration_config"][
                "xanes2D_regparams_mask_dilation_slider.val"]
        self.hs["UseSmooth chbx"].value = self.xanes_config[
            "registration_config"]["xanes2D_regparams_use_smooth_img"]
        self.hs["SmoothSigma text"].max = self.xanes_config[
            "registration_config"]["xanes2D_regparams_smooth_sigma_text.max"]
        self.hs["SmoothSigma text"].min = self.xanes_config[
            "registration_config"]["xanes2D_regparams_smooth_sigma_text.min"]
        self.hs["SmoothSigma text"].value = self.xanes_config[
            "registration_config"]["xanes2D_regparams_smooth_sigma_text.val"]
        self.hs["ChunkSz sldr"].max = self.xanes_config["registration_config"][
            "xanes2D_regparams_chunk_sz_slider.max"]
        self.hs["ChunkSz sldr"].min = self.xanes_config["registration_config"][
            "xanes2D_regparams_chunk_sz_slider.min"]
        self.hs["ChunkSz sldr"].value = self.xanes_config[
            "registration_config"]["xanes2D_regparams_chunk_sz_slider.val"]
        self.hs["RegMethod drpdn"].value = self.xanes_config[
            "registration_config"]["xanes2D_regparams_reg_method"]
        self.hs["RefMode drpdn"].options = self.xanes_config[
            "registration_config"][
                "xanes2D_regparams_ref_mode_dropdown_options"]
        self.hs["RefMode drpdn"].value = self.xanes_config[
            "registration_config"]["xanes2D_regparams_ref_mode"]
        self.hs["MrtvLevel text"].value = self.xanes_config[
            "registration_config"]["xanes2D_regparams_mrtv_level"]
        self.hs["MrtvWz text"].value = self.xanes_config[
            "registration_config"]["xanes2D_regparams_mrtv_width"]
        self.hs["MrtvSubpixelWz text"].value = self.xanes_config[
            "registration_config"]["xanes2D_regparams_mrtv_subpixel_width"]
        self.hs["MrtvSubpixelKernel text"].value = self.xanes_config[
            "registration_config"]["xanes2D_regparams_mrtv_subpixel_kernel"]

        self.hs["RegPair sldr"].max = self.xanes_config["review registration"][
            "reg_pair_slider_max"]
        self.hs["RegPair sldr"].min = self.xanes_config["review registration"][
            "reg_pair_slider_min"]
        self.hs["RegPair sldr"].value = self.xanes_config[
            "review registration"]["reg_pair_slider_val"]
        self.hs["XShift text"].value = self.xanes_config[
            "review registration"]["xshift_text_val"]
        self.hs["YShift text"].value = self.xanes_config[
            "review registration"]["yshift_text_val"]

        self.xanes_fit_gui_h.hs[
            "FitEngRagOptn drpdn"].value = self.xanes_config["align 2D recon"][
                "xanes2D_analysis_type"]
        self.xanes_fit_gui_h.hs[
            "FitEngRagEdgeEng text"].value = self.xanes_config[
                "align 2D recon"]["xanes2D_analysis_edge_eng"]
        self.xanes_fit_gui_h.hs[
            "FitEngRagPreEdgeEnd text"].value = self.xanes_config[
                "align 2D recon"]["xanes2D_analysis_pre_edge_e"]
        self.xanes_fit_gui_h.hs[
            "FitEngRagPostEdgeStr text"].value = self.xanes_config[
                "align 2D recon"]["xanes2D_analysis_post_edge_s"]
        self.xanes_fit_gui_h.hs[
            "FitEngRagWlFitStr text"].value = self.xanes_config[
                "align 2D recon"]["xanes2D_analysis_wl_fit_eng_s"]
        self.xanes_fit_gui_h.hs[
            "FitEngRagWlFitEnd text"].value = self.xanes_config[
                "align 2D recon"]["xanes2D_analysis_wl_fit_eng_e"]
        self.xanes_fit_gui_h.hs[
            "FitEngRagEdge0.5Str text"].value = self.xanes_config[
                "align 2D recon"]["xanes2D_analysis_edge_0p5_fit_s"]
        self.xanes_fit_gui_h.hs[
            "FitEngRagEdge0.5End text"].value = self.xanes_config[
                "align 2D recon"]["xanes2D_analysis_edge_0p5_fit_e"]

    def boxes_logic(self):

        def xanes2D_compound_logic():
            if self.xanes_file_analysis_option == "Reg By Shift":
                if self.xanes_roi_configured:
                    boxes = ["RevRegRlt box"]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=True,
                                         level=-1)
                    self.hs["ReadAlign chbx"].disabled = True
                    self.hs["ReadAlign btn"].disabled = False
                else:
                    boxes = ["RevRegRlt box"]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=True,
                                         level=-1)
                if self.xanes_reg_file_readed:
                    self.hs["CfmRevRlt btn"].disabled = False
            else:
                if (self.xanes_reg_done and (not self.xanes_reg_file_readed)
                        and self.hs["ReadAlign chbx"].value):
                    boxes = ["RegPair box", "CorrSht box"]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=True,
                                         level=-1)
                    self.hs["ReadAlign btn"].disabled = False
                elif self.xanes_reg_done and (
                        not self.hs["ReadAlign chbx"].value):
                    boxes = ["RegPair box"]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=False,
                                         level=-1)
                    boxes = ["CorrSht box"]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=True,
                                         level=-1)
                    self.hs["ReadAlign btn"].disabled = True

                if not self.xanes_review_bad_shift:
                    boxes = ["CorrSht box"]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=True,
                                         level=-1)
                else:
                    boxes = ["CorrSht box"]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=False,
                                         level=-1)

            if self.xanes_config_raw_img_readed:
                boxes = ["EngRange box", "Fiji box", "ConfigDataCfm box"]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            else:
                boxes = ["EngRange box", "Fiji box", "ConfigDataCfm box"]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)

            if self.hs["IsRaw chbx"].value & self.xanes_file_configured:
                self.hs["NormScale text"].disabled = False
            else:
                self.hs["NormScale text"].disabled = True

            if self.hs["UseChunk chbx"].value:
                self.hs["AnchorId sldr"].disabled = False
                self.hs["ChunkSz sldr"].disabled = False
                self.xanes_regparams_anchor_idx_set = False
            else:
                self.hs["AnchorId sldr"].disabled = True
                self.hs["ChunkSz sldr"].disabled = True
                self.xanes_regparams_anchor_idx_set = False

            if self.hs["RegMethod drpdn"].value in ["MPC", "MPC+MRTV"]:
                self.hs["UseMask chbx"].value = 1
                self.hs["MaskThres sldr"].disabled = False
                self.hs["MaskDilation sldr"].disabled = False
            elif self.hs["RegMethod drpdn"].value in ["MRTV", "PC", "LS+MRTV"]:
                self.hs["UseMask chbx"].value = 0
                self.hs["MaskThres sldr"].disabled = True
                self.hs["MaskDilation sldr"].disabled = True
            elif self.hs["RegMethod drpdn"].value == "SR":
                self.hs["MaskThres sldr"].disabled = False
                self.hs["MaskDilation sldr"].disabled = False

            if self.hs["UseSmooth chbx"].value:
                self.hs["SmoothSigma text"].disabled = False
            else:
                self.hs["SmoothSigma text"].disabled = True

            if self.hs["RegMethod drpdn"].value in ["MPC", "SR", "PC"]:
                boxes = ["MrtvOptions box"]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            elif self.hs["RegMethod drpdn"].value == "MPC+MRTV":
                self.hs["MrtvLevel text"].disabled = True
                self.hs["MrtvWz text"].disabled = True
                self.hs["MrtvSubpixelWz text"].disabled = False
                self.hs["MrtvSubpixelKernel text"].disabled = False
            elif self.hs["RegMethod drpdn"].value == "LS+MRTV":
                self.hs["MrtvLevel text"].disabled = True
                self.hs["MrtvWz text"].disabled = False
                self.hs["MrtvWz text"].max = 300
                self.hs["MrtvWz text"].min = 100
                self.hs["MrtvSubpixelWz text"].disabled = False
                self.hs["MrtvSubpixelKernel text"].disabled = False
            elif self.hs["RegMethod drpdn"].value == "MRTV":
                self.hs["MrtvWz text"].min = 1
                self.hs["MrtvWz text"].max = 20
                if self.hs["MrtvWz text"].value >= 20:
                    self.hs["MrtvWz text"].value = 10
                boxes = ["MrtvOptions box"]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)

            if self.hs["RegMethod drpdn"].value in ["PC", "MRTV", "LS+MRTV"]:
                boxes = ["MaskOptions box"]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            elif self.hs["RegMethod drpdn"].value in ["MPC", "MPC+MRTV", "SR"]:
                boxes = ["MaskOptions box"]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)

            if self.hs["MrtvSubpixelSrch drpdn"].value == "analytical":
                self.hs["MrtvSubpixelWz text"].disabled = True
            else:
                self.hs["MrtvSubpixelWz text"].disabled = False

        if self.xanes_file_analysis_option in [
                "Do New Reg", "Read Config File"
        ]:
            if not self.xanes_file_configured:
                boxes = [
                    "ConfigData box",
                    "2DRoi box",
                    "ConfigRegParams box",
                    "RunReg box",
                    "RevRegRlt box",
                    "AlignImg box",
                    "VisImg box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            elif self.xanes_file_configured & (not self.xanes_data_configured):
                boxes = [
                    "ConfigData box",
                    "2DRoi box",
                    "ConfigRegParams box",
                    "RunReg box",
                    "RevRegRlt box",
                    "AlignImg box",
                    "VisImg box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
                boxes = ["DataPrepOptions box"]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            elif (self.xanes_file_configured & self.xanes_data_configured) & (
                    not self.xanes_roi_configured):
                boxes = [
                    "ConfigRegParams box",
                    "RunReg box",
                    "RevRegRlt box",
                    "AlignImg box",
                    "VisImg box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
                boxes = ["ConfigData box", "2DRoi box"]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            elif ((self.xanes_file_configured & self.xanes_data_configured)
                  & self.xanes_roi_configured
                  & (not self.xanes_reg_params_configured)):
                boxes = [
                    "RunReg box", "RevRegRlt box", "AlignImg box", "VisImg box"
                ]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
                boxes = ["ConfigData box", "2DRoi box", "ConfigRegParams box"]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            elif (
                (self.xanes_file_configured & self.xanes_data_configured)
                    &
                (self.xanes_roi_configured & self.xanes_reg_params_configured)
                    & (not self.xanes_reg_done)):
                boxes = ["RevRegRlt box", "AlignImg box", "VisImg box"]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
                boxes = [
                    "ConfigData box",
                    "2DRoi box",
                    "ConfigRegParams box",
                    "RunReg box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            elif (
                (self.xanes_file_configured & self.xanes_data_configured)
                    &
                (self.xanes_roi_configured & self.xanes_reg_params_configured)
                    & (self.xanes_reg_done &
                       (not self.xanes_reg_review_done))):
                boxes = ["AlignImg box", "VisImg box"]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
                boxes = [
                    "ConfigData box",
                    "2DRoi box",
                    "ConfigRegParams box",
                    "RunReg box",
                    "RevRegRlt box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            elif (
                (self.xanes_file_configured & self.xanes_data_configured)
                    &
                (self.xanes_roi_configured & self.xanes_reg_params_configured)
                    & (self.xanes_reg_done & self.xanes_reg_review_done)
                    & (not self.xanes_alignment_done)):
                boxes = [
                    "VisImg box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
                boxes = [
                    "ConfigData box",
                    "2DRoi box",
                    "ConfigRegParams box",
                    "RunReg box",
                    "RevRegRlt box",
                    "AlignImg box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            elif (
                (self.xanes_file_configured & self.xanes_data_configured)
                    &
                (self.xanes_roi_configured & self.xanes_reg_params_configured)
                    & (self.xanes_reg_done & self.xanes_reg_review_done)
                    & (self.xanes_alignment_done)
                    & (not self.xanes_fit_eng_configured)):
                boxes = [
                    "ConfigData box",
                    "2DRoi box",
                    "ConfigRegParams box",
                    "RunReg box",
                    "RevRegRlt box",
                    "AlignImg box",
                    "VisImg box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ["FitEngRag box"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=False,
                                     level=-1)
                boxes = ["FitItem tab"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            elif (
                (self.xanes_file_configured & self.xanes_data_configured)
                    &
                (self.xanes_roi_configured & self.xanes_reg_params_configured)
                    & (self.xanes_reg_done & self.xanes_reg_review_done)
                    &
                (self.xanes_alignment_done & self.xanes_fit_eng_configured)):
                boxes = [
                    "ConfigData box",
                    "2DRoi box",
                    "ConfigRegParams box",
                    "RunReg box",
                    "RevRegRlt box",
                    "AlignImg box",
                    "VisImg box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ["FitEngRag box", "FitItem tab"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=False,
                                     level=-1)
            xanes2D_compound_logic()
        elif self.xanes_file_analysis_option == "Reg By Shift":
            if not self.xanes_file_configured:
                boxes = [
                    "2DRoi box", "RevRegRlt box", "AlignImg box", "VisImg box"
                ]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            elif self.xanes_file_configured & (not self.xanes_roi_configured):
                boxes = ["2DRoi box"]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ["RevRegRlt box", "AlignImg box", "VisImg box"]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            elif (self.xanes_file_configured & self.xanes_roi_configured) & (
                    not self.xanes_reg_review_done):
                boxes = ["2DRoi box", "ReadAlignFile box"]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                self.hs["ReadAlign chbx"].value = True
                self.hs["ReadAlign chbx"].disabled = True
                self.hs["ReadAlign btn"].disabled = False

                boxes = ["AlignImg box", "VisImg box"]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            elif (self.xanes_file_configured & self.xanes_roi_configured) & (
                    self.xanes_reg_review_done &
                (not self.xanes_alignment_done)):
                boxes = ["2DRoi box", "ReadAlignFile box", "AlignImg box"]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                self.hs["ReadAlign chbx"].value = True
                self.hs["ReadAlign chbx"].disabled = True
                self.hs["ReadAlign btn"].disabled = False

                boxes = ["VisImg box"]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            elif ((self.xanes_file_configured & self.xanes_roi_configured)
                  & (self.xanes_reg_review_done & self.xanes_alignment_done)
                  & (not self.xanes_fit_eng_configured)):
                boxes = [
                    "2DRoi box", "ReadAlignFile box", "AlignImg box",
                    "VisImg box"
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                self.hs["ReadAlign chbx"].value = True
                self.hs["ReadAlign chbx"].disabled = True
                self.hs["ReadAlign btn"].disabled = False
                boxes = ["FitEngRag box"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=False,
                                     level=-1)
                boxes = ["FitItem tab"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            elif ((self.xanes_file_configured & self.xanes_roi_configured)
                  & (self.xanes_reg_review_done & self.xanes_alignment_done)
                  & self.xanes_fit_eng_configured):
                boxes = [
                    "2DRoi box", "ReadAlignFile box", "AlignImg box",
                    "VisImg box"
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                self.hs["ReadAlign chbx"].value = True
                self.hs["ReadAlign chbx"].disabled = True
                self.hs["ReadAlign btn"].disabled = False
                boxes = ["FitEngRag box"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=False,
                                     level=-1)
                boxes = ["FitItem tab"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            xanes2D_compound_logic()
            boxes = ["ConfigData box", "ConfigRegParams box", "RunReg box"]
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        elif self.xanes_file_analysis_option == "Do Analysis":
            boxes = [
                "ConfigData box",
                "2DRoi box",
                "ConfigRegParams box",
                "RunReg box",
                "RevRegRlt box",
                "AlignImg box",
            ]
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            boxes = ["VisImg box"]
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            boxes = ["FitEngRag box"]
            enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                 boxes,
                                 disabled=False,
                                 level=-1)
            boxes = ["FitItem tab"]
            if self.xanes_fit_eng_configured:
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=False,
                                     level=-1)
            else:
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
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
        layout = {
            "border": "3px solid #FFCC00",
            "width": "100%",
            "height": "100%"
        }
        self.hs["Config&Input form"] = widgets.VBox()
        self.hs["RegSetting form"] = widgets.VBox()
        self.hs["Reg&Rev form"] = widgets.VBox()
        self.hs["Fitting form"] = widgets.VBox()
        self.hs["Analysis form"] = widgets.VBox()
        self.hs["Config&Input form"].layout = layout
        self.hs["RegSetting form"].layout = layout
        self.hs["Reg&Rev form"].layout = layout
        self.hs["Fitting form"].layout = layout
        self.hs["Analysis form"].layout = layout

        ## ## ## ## define functional widget tabs in each sub-tab - configure file settings -- start
        base_wz_os = 92
        ex_ws_os = 6
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.28 * (self.form_sz[0] - 136)}px",
        }
        self.hs["SelFile&Path box"] = widgets.VBox()
        self.hs["SelFile&Path box"].layout = layout

        ## ## ## ## ## label configure file settings box
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["SelFile&PathTitle box"] = widgets.HBox()
        self.hs["SelFile&PathTitle box"].layout = layout
        self.hs["SelFile&PathTitle label"] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + "Config Dirs & Files" + "</span>")
        layout = {"background-color": "white", "color": "cyan", "left": "39%"}
        self.hs["SelFile&PathTitle label"].layout = layout

        self.hs["SelFile&PathTitle box"].children = [
            self.hs["SelFile&PathTitle label"]
        ]

        ## ## ## ## ## raw h5 top directory
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["SelRaw box"] = widgets.HBox()
        self.hs["SelRaw box"].layout = layout
        self.hs["SelRawH5Path text"] = widgets.Text(
            value="Choose raw h5 directory ...", description="", disabled=True)
        layout = {"width": "66%", "display": "inline_flex"}
        self.hs["SelRawH5Path text"].layout = layout
        self.hs["SelRawH5Path btn"] = SelectFilesButton(
            option="askopenfilename",
            text_h=self.hs["SelRawH5Path text"],
            **{"open_filetypes": (("h5 files", "*.h5"), )},
        )
        layout = {"width": "15%"}
        self.hs["SelRawH5Path btn"].layout = layout
        self.hs["SelRawH5Path btn"].description = "XANES2D File"

        self.hs["SelRawH5Path btn"].on_click(self.SelRawH5Path_btn_clk)
        self.hs["SelRaw box"].children = [
            self.hs["SelRawH5Path text"],
            self.hs["SelRawH5Path btn"],
        ]

        ## ## ## ## ## trial save file
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["SelSaveTrial box"] = widgets.HBox()
        self.hs["SelSaveTrial box"].layout = layout
        self.hs["SelSaveTrial text"] = widgets.Text(
            value="Save trial registration as ...",
            description="",
            disabled=True)
        layout = {"width": "66%", "display": "inline_flex"}
        self.hs["SelSaveTrial text"].layout = layout
        self.hs["SelSaveTrial btn"] = SelectFilesButton(
            option="asksaveasfilename",
            text_h=self.hs["SelSaveTrial text"],
            **{
                "open_filetypes": (("h5 files", "*.h5"), ),
                "initialfile": "2D_trial_reg.h5",
            },
        )
        self.hs["SelSaveTrial btn"].description = "Save Reg File"
        layout = {"width": "15%"}
        self.hs["SelSaveTrial btn"].layout = layout

        self.hs["SelSaveTrial btn"].on_click(self.SelSaveTrial_btn_clk)
        self.hs["SelSaveTrial box"].children = [
            self.hs["SelSaveTrial text"],
            self.hs["SelSaveTrial btn"],
        ]

        ## ## ## ## ## confirm file configuration
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["SelFile&PathComfirm box"] = widgets.HBox()
        self.hs["SelFile&PathComfirm box"].layout = layout
        self.hs["CfmFile&Path text"] = widgets.Text(
            value=
            "Save trial registration, or go directly review registration ...",
            description="",
            disabled=True,
        )
        layout = {"width": "66%"}
        self.hs["CfmFile&Path text"].layout = layout
        self.hs["CfmFile&Path btn"] = widgets.Button(
            description="Confirm",
            tooltip="Confirm: Confirm after you finish file configuration",
        )
        self.hs["CfmFile&Path btn"].style.button_color = "darkviolet"
        self.hs["CfmFile&Path btn"].on_click(self.CfmFile_Path_btn_clk)
        layout = {"width": "15%"}
        self.hs["CfmFile&Path btn"].layout = layout

        self.hs["FilePathOptions drpdn"] = widgets.Dropdown(
            value="Do New Reg",
            options=[
                "Do New Reg", "Read Config File", "Reg By Shift", "Do Analysis"
            ],
            description="",
            description_tooltip=
            '"Do New Reg": start registration and review results from beginning; "Read Reg File": if you have already done registraion and like to review the results; "Read Config File": if you like to resume analysis with an existing configuration.',
            disabled=False,
        )
        layout = {"width": "15%"}
        self.hs["FilePathOptions drpdn"].layout = layout

        self.hs["FilePathOptions drpdn"].observe(
            self.FilePathOptions_drpdn_chg, names="value")
        self.hs["SelFile&PathComfirm box"].children = [
            self.hs["CfmFile&Path text"],
            self.hs["CfmFile&Path btn"],
            self.hs["FilePathOptions drpdn"],
        ]

        self.hs["SelFile&Path box"].children = [
            self.hs["SelFile&PathTitle box"],
            self.hs["SelRaw box"],
            self.hs["SelSaveTrial box"],
            self.hs["SelFile&PathComfirm box"],
        ]
        ## ## ## ## boundle widgets in hs['SelFile&Path box'] -- configure file settings -- end

        ## ## ## ## define functional widgets in each box in each sub-tab  - define data -- start
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.55 * (self.form_sz[0] - 136)}px",
        }
        self.hs["DataConfig&Info tab"] = widgets.Tab()
        self.hs["DataConfig&Info tab"].layout = layout

        base_wz_os = 136
        ex_ws_os = 6
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.43 * (self.form_sz[0] - 136)}px",
        }
        self.hs["ConfigData box"] = widgets.VBox()
        self.hs["ConfigData box"].layout = layout

        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["DataPrepOptions box"] = widgets.HBox()
        self.hs["DataPrepOptions box"].layout = layout

        self.hs["IsRaw chbx"] = widgets.Checkbox(
            value=True,
            description="Is Raw",
            disabled=True,
            indent=False,
            description_tooltip=
            "check it if the XANES data in the file is not normalized",
        )
        layout = {"width": "20%"}
        self.hs["IsRaw chbx"].layout = layout
        layout = {"width": "20%"}
        self.hs["NormScale text"] = widgets.BoundedFloatText(
            value=1,
            min=1e-10,
            max=100,
            step=0.1,
            description="Norm Scale",
            disabled=True,
            description_tooltip=
            "scale the XANES data with a factor if the normalized data is not in range [0, 1]",
        )
        self.hs["NormScale text"].layout = layout
        layout = {"left": "40.9%", "width": "15%"}
        self.hs["ConfigDataLoadImg btn"] = widgets.Button(
            description="Load Images", disabled=True)
        self.hs["ConfigDataLoadImg btn"].layout = layout
        self.hs["ConfigDataLoadImg btn"].style.button_color = "darkviolet"

        self.hs["IsRaw chbx"].observe(self.IsRaw_chbx_change, names="value")
        self.hs["NormScale text"].observe(self.NormScale_text_chg,
                                          names="value")
        self.hs["ConfigDataLoadImg btn"].on_click(
            self.ConfigDataLoadImg_btn_clk)
        self.hs["DataPrepOptions box"].children = [
            self.hs["IsRaw chbx"],
            self.hs["NormScale text"],
            self.hs["ConfigDataLoadImg btn"],
        ]
        ## ## ## ## ## data_preprocessing_options -- end

        ## ## ## ## ## define eng points -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["EngRange box"] = widgets.HBox()
        self.hs["EngRange box"].layout = layout
        self.hs["EngPointsRange sldr"] = widgets.IntRangeSlider(
            value=[0, 1],
            step=1,
            description="eng points range",
            disabled=True)
        layout = {"width": "55%"}
        self.hs["EngPointsRange sldr"].layout = layout
        self.hs["EngStart text"] = widgets.BoundedFloatText(
            value=0,
            min=1e3,
            max=5e4,
            step=0.1,
            description="eng s (eV)",
            disabled=True)
        layout = {"width": "20%"}
        self.hs["EngStart text"].layout = layout
        self.hs["EngEnd text"] = widgets.BoundedFloatText(
            value=0,
            min=1e3,
            max=5e4,
            step=0.1,
            description="eng e (eV)",
            disabled=True)
        layout = {"width": "20%"}
        self.hs["EngEnd text"].layout = layout

        self.hs["EngPointsRange sldr"].observe(self.EngPointsRange_slider_chg,
                                               names="value")
        self.hs["EngRange box"].children = [
            self.hs["EngPointsRange sldr"],
            self.hs["EngStart text"],
            self.hs["EngEnd text"],
        ]
        ## ## ## ## ## define eng points -- end

        ## ## ## ## ## fiji option -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["Fiji box"] = widgets.HBox()
        self.hs["Fiji box"].layout = layout
        self.hs["FijiRawImgPrev chbx"] = widgets.Checkbox(
            value=False, description="fiji view", disabled=True, indent=False)
        layout = {"width": "20%"}
        self.hs["FijiRawImgPrev chbx"].layout = layout
        self.hs["FijiEngId sldr"] = widgets.IntSlider(value=False,
                                                      description="img #",
                                                      disabled=True,
                                                      min=0)
        layout = {"width": "60.6%"}
        self.hs["FijiEngId sldr"].layout = layout
        self.hs["FijiClose btn"] = widgets.Button(
            description="close all fiji viewers", disabled=True)
        layout = {"width": "15%"}
        self.hs["FijiClose btn"].layout = layout

        self.hs["FijiRawImgPrev chbx"].observe(self.FijiRawImgPrev_chbx_chg,
                                               names="value")
        self.hs["FijiEngId sldr"].observe(self.FijiEngId_sldr_chg,
                                          names="value")
        self.hs["FijiClose btn"].on_click(self.FijiClose_btn_clk)
        self.hs["Fiji box"].children = [
            self.hs["FijiRawImgPrev chbx"],
            self.hs["FijiEngId sldr"],
            self.hs["FijiClose btn"],
        ]
        ## ## ## ## ## fiji option -- end

        ## ## ## ## ## confirm data configuration -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["ConfigDataCfm box"] = widgets.HBox()
        self.hs["ConfigDataCfm box"].layout = layout
        self.hs["CfmConfigData text"] = widgets.Text(
            value="Cfm setting once you are done ...",
            description="",
            disabled=True)
        layout = {"width": "81.0%"}
        self.hs["CfmConfigData text"].layout = layout
        self.hs["CfmConfigData btn"] = widgets.Button(
            description="Confirm",
            description_tooltip=
            "Confirm: Confirm after you finish file configuration",
            disabled=True,
        )
        self.hs["CfmConfigData btn"].style.button_color = "darkviolet"
        layout = {"width": "15%"}
        self.hs["CfmConfigData btn"].layout = layout

        self.hs["CfmConfigData btn"].on_click(self.CfmConfigData_btn_clk)
        self.hs["ConfigDataCfm box"].children = [
            self.hs["CfmConfigData text"],
            self.hs["CfmConfigData btn"],
        ]
        ## ## ## ## ## confirm data configuration -- end
        self.hs["ConfigData box"].children = [
            self.hs["DataPrepOptions box"],
            self.hs["EngRange box"],
            self.hs["Fiji box"],
            self.hs["ConfigDataCfm box"],
        ]
        ## ## ## ## boundle widgets in hs['ConfigData box']  - config data -- end

        ## ## ## ## data metadata display -- start
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.43 * (self.form_sz[0] - 136)}px",
        }
        self.hs["MetaData box"] = widgets.VBox()
        self.hs["MetaData box"].layout = layout

        layout = {"width": "90%", "height": "90%"}
        self.hs["DataInfo text"] = widgets.Textarea(
            value="Data Info",
            placeholder="Data Info",
            description="Data Info",
            disabled=True,
        )
        self.hs["DataInfo text"].layout = layout

        self.hs["MetaData box"].children = [self.hs["DataInfo text"]]
        ## ## ## ## data metadata display -- end

        self.hs["DataConfig&Info tab"].children = [
            self.hs["ConfigData box"],
            self.hs["MetaData box"],
        ]
        self.hs["DataConfig&Info tab"].set_title(0, "Config Data")
        self.hs["DataConfig&Info tab"].set_title(1, "Data Info")
        ## ## ## Data Config Tab -- end

        self.hs["Config&Input form"].children = [
            self.hs["SelFile&Path box"],
            self.hs["DataConfig&Info tab"],
        ]
        ## ## ## define 2D_XANES_tabs layout file&data configuration-- end

        ## ## ## define 2D_XANES_tabs layout - registration configuration --start
        ## ## ## ## define functional widgets each tab in each sub-tab - config roi -- start
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.28 * (self.form_sz[0] - 136)}px",
        }
        self.hs["2DRoi box"] = widgets.VBox()
        self.hs["2DRoi box"].layout = layout
        base_wz_os = 92
        ex_ws_os = 6
        ## ## ## ## ## label 2D_roi_title box -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["2DRoiTitle box"] = widgets.HBox()
        self.hs["2DRoiTitle box"].layout = layout
        self.hs["2DRoiTitle text"] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + "Config 2D ROI" + "</span>")
        layout = {
            "justify-content": "center",
            "background-color": "white",
            "color": "cyan",
            "left": "43%",
        }
        self.hs["2DRoiTitle text"].layout = layout
        self.hs["2DRoiTitle box"].children = [self.hs["2DRoiTitle text"]]
        ## ## ## ## ## label 2D_roi_title box -- end

        ## ## ## ## ## define roi -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["2DRoiDefine box"] = widgets.VBox()
        self.hs["2DRoiDefine box"].layout = layout
        self.hs["2DRoiX sldr"] = widgets.IntRangeSlider(
            value=[10, 40],
            min=1,
            max=50,
            step=1,
            description="x range:",
            disabled=True,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        layout = {"width": "auto"}
        self.hs["2DRoiX sldr"].layout = layout
        self.hs["2DRoiY sldr"] = widgets.IntRangeSlider(
            value=[10, 40],
            min=1,
            max=50,
            step=1,
            description="y range:",
            disabled=True,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        self.hs["2DRoiY sldr"].layout = layout

        self.hs["2DRoiX sldr"].observe(self.Roi2DX_sldr_chg, names="value")
        self.hs["2DRoiY sldr"].observe(self.Roi2DY_sldr_chg, names="value")
        self.hs["2DRoiDefine box"].children = [
            self.hs["2DRoiX sldr"],
            self.hs["2DRoiY sldr"],
        ]
        ## ## ## ## ## define roi -- end

        ## ## ## ## ## confirm roi -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["2DRoiCfm box"] = widgets.HBox()
        self.hs["2DRoiCfm box"].layout = layout
        layout = {"width": "70%"}
        self.hs["CfmRoi text"] = widgets.Text(
            description="",
            value="Please confirm after ROI is set ...",
            disabled=True)
        self.hs["CfmRoi text"].layout = layout
        layout = {"width": "15%"}
        self.hs["CfmRoi btn"] = widgets.Button(
            description="Confirm",
            description_tooltip="Confirm the roi once you define the ROI ...",
            disabled=True,
        )
        self.hs["CfmRoi btn"].style.button_color = "darkviolet"
        self.hs["CfmRoi btn"].layout = layout

        self.hs["CfmRoi btn"].on_click(self.CfmRoi_btn_clk)
        self.hs["2DRoiCfm box"].children = [
            self.hs["CfmRoi text"],
            self.hs["CfmRoi btn"],
        ]
        ## ## ## ## ## confirm roi -- end

        self.hs["2DRoi box"].children = [
            self.hs["2DRoiTitle box"],
            self.hs["2DRoiDefine box"],
            self.hs["2DRoiCfm box"],
        ]
        ## ## ## ## define functional widgets in each sub-tab - config roi -- end

        ## ## ## ## define functional widgets in each sub-tab - config registration -- start
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.42 * (self.form_sz[0] - 136)}px",
        }
        self.hs["ConfigRegParams box"] = widgets.VBox()
        self.hs["ConfigRegParams box"].layout = layout

        ## ## ## ## ## label config_reg_params box --start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["ConfigRegParamsTitle box"] = widgets.HBox()
        self.hs["ConfigRegParamsTitle box"].layout = layout
        self.hs["ConfigRegParamsTitle text"] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + "Config Reg Params" + "</span>")
        layout = {
            "background-color": "white",
            "color": "cyan",
            "left": "40.5%"
        }
        self.hs["ConfigRegParamsTitle text"].layout = layout

        self.hs["ConfigRegParamsTitle box"].children = [
            self.hs["ConfigRegParamsTitle text"]
        ]
        ## ## ## ## ## label config_reg_params box --end

        ## ## ## ## ## fiji&anchor box -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["Fiji&Anchor box"] = widgets.HBox()
        self.hs["Fiji&Anchor box"].layout = layout
        self.hs["FijiMaskViewer chbx"] = widgets.Checkbox(
            value=False, disabled=True, description="preview", indent=False)
        layout = {"width": "19%"}
        self.hs["FijiMaskViewer chbx"].layout = layout
        self.hs["UseChunk chbx"] = widgets.Checkbox(value=True,
                                                    disabled=True,
                                                    description="use anchor",
                                                    indent=False)
        layout = {"width": "19%"}
        self.hs["UseChunk chbx"].layout = layout
        self.hs["AnchorId sldr"] = widgets.IntSlider(value=1,
                                                     min=1,
                                                     disabled=True,
                                                     description="anchor id")
        layout = {"width": "29%"}
        self.hs["AnchorId sldr"].layout = layout
        self.hs["ChunkSz sldr"] = widgets.IntSlider(value=7,
                                                    min=1,
                                                    disabled=True,
                                                    description="chunk size")
        layout = {"width": "29%"}
        self.hs["ChunkSz sldr"].layout = layout

        self.hs["FijiMaskViewer chbx"].observe(self.FijiMaskViewer_chbx_chg,
                                               names="value")
        self.hs["UseChunk chbx"].observe(self.UseChunk_chbx_chg, names="value")
        self.hs["AnchorId sldr"].observe(self.AnchorId_sldr_chg, names="value")
        self.hs["ChunkSz sldr"].observe(self.ChunkSz_sldr_chg, names="value")
        self.hs["Fiji&Anchor box"].children = [
            self.hs["FijiMaskViewer chbx"],
            self.hs["UseChunk chbx"],
            self.hs["AnchorId sldr"],
            self.hs["ChunkSz sldr"],
        ]
        ## ## ## ## ## fiji&anchor box -- end

        ## ## ## ## ## mask options box -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["MaskOptions box"] = widgets.HBox()
        self.hs["MaskOptions box"].layout = layout
        self.hs["UseMask chbx"] = widgets.Checkbox(value=False,
                                                   disabled=True,
                                                   description="use mask",
                                                   indent=False)
        layout = {"width": "19%", "flex-direction": "row"}
        self.hs["UseMask chbx"].layout = layout
        self.hs["MaskThres sldr"] = widgets.FloatSlider(
            value=False,
            disabled=True,
            description="mask thres",
            readout_format=".5f",
            min=-1.0,
            max=1.0,
            step=1e-5,
            indent=False,
        )
        layout = {"width": "29%"}
        self.hs["MaskThres sldr"].layout = layout
        self.hs["MaskDilation sldr"] = widgets.IntSlider(
            value=False,
            disabled=True,
            description="mask dilation",
            min=0,
            max=30,
            step=1,
            indent=False,
        )
        layout = {"width": "29%"}
        self.hs["MaskDilation sldr"].layout = layout

        self.hs["UseMask chbx"].observe(self.UseMask_chbx_chg, names="value")
        self.hs["MaskThres sldr"].observe(self.MaskThres_sldr_chg,
                                          names="value")
        self.hs["MaskDilation sldr"].observe(self.MaskDilation_sldr_chg,
                                             names="value")
        self.hs["MaskOptions box"].children = [
            self.hs["UseMask chbx"],
            self.hs["MaskThres sldr"],
            self.hs["MaskDilation sldr"],
        ]
        ## ## ## ## ## mask options box -- end

        ## ## ## ## ## smooth & chunk_size box -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["SliSrch&ChunkSz box"] = widgets.HBox()
        self.hs["SliSrch&ChunkSz box"].layout = layout
        self.hs["UseSmooth chbx"] = widgets.Checkbox(
            value=False,
            disabled=True,
            description="smooth img",
            description_tooltip=
            "enable option to smooth images before image registration",
            indent=False,
        )
        layout = {"width": "19%"}
        self.hs["UseSmooth chbx"].layout = layout
        self.hs["SmoothSigma text"] = widgets.BoundedFloatText(
            value=5,
            min=0,
            max=30,
            step=0.1,
            disabled=True,
            description_tooltip="kernel width for smoothing images",
            description="smooth sig",
        )
        layout = {"width": "19%"}
        self.hs["SmoothSigma text"].layout = layout
        self.hs["RegMethod drpdn"] = widgets.Dropdown(
            value="MRTV",
            options=["MRTV", "MPC", "PC", "SR", "LS+MRTV", "MPC+MRTV"],
            description="reg method",
            description_tooltip=
            "reg method: MRTV: Multi-resolution Total Variantion, MPC: Masked Phase Correlation; PC: Phase Correlation; SR: StackReg; LS+MRTV: a hybrid TV minimization combining line search and multiresolution strategy; MPC+MRTV: a hybrid search algorihm with MPC as primary search followed by subpixel TV search",
            disabled=True,
        )
        layout = {"width": "19%"}
        self.hs["RegMethod drpdn"].layout = layout
        self.hs["RefMode drpdn"] = widgets.Dropdown(
            value="single",
            options=["single", "neighbor", "average"],
            description="ref mode",
            description_tooltip=
            "ref mode: single: two images with fixed id gap in two consecutive chunks are used for the registration between two chunks; neighbor: two neighbor images in two consecutive chunks are used for the registration between two chunks; average: deprecated",
            disabled=True,
        )
        layout = {"width": "19%"}
        self.hs["RefMode drpdn"].layout = layout

        self.hs["UseSmooth chbx"].observe(self.UseSmooth_chbx_chg,
                                          names="value")
        self.hs["SmoothSigma text"].observe(self.SmoothSigma_text_chg,
                                            names="value")
        self.hs["RegMethod drpdn"].observe(self.RegMethod_drpdn_chg,
                                           names="value")
        self.hs["RefMode drpdn"].observe(self.RefMode_drpdn_chg, names="value")
        self.hs["SliSrch&ChunkSz box"].children = [
            self.hs["UseSmooth chbx"],
            self.hs["SmoothSigma text"],
            self.hs["RegMethod drpdn"],
            self.hs["RefMode drpdn"],
        ]
        ## ## ## ## ## smooth & chunk_size box -- end

        ## ## ## ## ##  reg_options box -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["MrtvOptions box"] = widgets.HBox()
        self.hs["MrtvOptions box"].layout = layout

        self.hs["MrtvLevel text"] = widgets.BoundedIntText(
            value=5,
            min=1,
            max=10,
            step=1,
            description="level",
            description_tooltip="level: multi-resolution level",
            disabled=True,
        )
        layout = {"width": "19%"}
        self.hs["MrtvLevel text"].layout = layout

        self.hs["MrtvWz text"] = widgets.BoundedIntText(
            value=10,
            min=1,
            max=20,
            step=1,
            description="width",
            description_tooltip=
            "width: multi-resolution searching width at each level (number of searching steps)",
            disabled=True,
        )
        layout = {"width": "19%"}
        self.hs["MrtvWz text"].layout = layout

        self.hs["MrtvSubpixelSrch drpdn"] = widgets.Dropdown(
            value="analytical",
            options=["analytical", "fitting"],
            description="subpxl srch",
            description_tooltip="subpxl srch: subpixel TV minization option",
            disabled=True,
        )
        layout = {"width": "19%"}
        self.hs["MrtvSubpixelSrch drpdn"].layout = layout

        self.hs["MrtvSubpixelWz text"] = widgets.BoundedIntText(
            value=3,
            min=2,
            max=20,
            step=0.1,
            description="subpxl wz",
            description_tooltip="subpxl wz: final sub-pixel fitting points",
            disabled=True,
        )
        layout = {"width": "19%"}
        self.hs["MrtvSubpixelWz text"].layout = layout

        self.hs["MrtvSubpixelKernel text"] = widgets.BoundedIntText(
            value=3,
            min=2,
            max=20,
            step=1,
            description="kernel wz",
            description_tooltip=
            "us factor: upsampling factor for subpixel search",
            disabled=True,
        )
        layout = {"width": "19%"}
        self.hs["MrtvSubpixelKernel text"].layout = layout

        self.hs["MrtvLevel text"].observe(self.MrtvLevel_text_chg,
                                          names="value")
        self.hs["MrtvWz text"].observe(self.MrtvWz_text_chg, names="value")
        self.hs["MrtvSubpixelWz text"].observe(self.MrtvSubpixelWz_text_chg,
                                               names="value")
        self.hs["MrtvSubpixelKernel text"].observe(
            self.MrtvSubpixelKernel_text_chg, names="value")
        self.hs["MrtvSubpixelSrch drpdn"].observe(
            self.MrtvSubpixelSrch_drpdn_chg, names="value")
        self.hs["MrtvOptions box"].children = [
            self.hs["MrtvLevel text"],
            self.hs["MrtvWz text"],
            self.hs["MrtvSubpixelWz text"],
            self.hs["MrtvSubpixelKernel text"],
            self.hs["MrtvSubpixelSrch drpdn"],
        ]
        ## ## ## ## ##  reg_options box -- end

        ## ## ## ## ## confirm reg settings -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["ConfigRegParamsCfm box"] = widgets.HBox()
        self.hs["ConfigRegParamsCfm box"].layout = layout
        layout = {"width": "70%"}
        self.hs["CfmRegParams text"] = widgets.Text(
            description="",
            value="Confirm the roi once you define the ROI ...",
            disabled=True,
        )
        self.hs["CfmRegParams text"].layout = layout
        layout = {"width": "15%"}
        self.hs["CfmRegParams btn"] = widgets.Button(
            description="Confirm",
            description_tooltip="Confirm the roi once you define the ROI ...",
            disabled=True,
        )
        self.hs["CfmRegParams btn"].style.button_color = "darkviolet"
        self.hs["CfmRegParams btn"].layout = layout

        self.hs["CfmRegParams btn"].on_click(self.CfmRegParams_btn_clk)
        self.hs["ConfigRegParamsCfm box"].children = [
            self.hs["CfmRegParams text"],
            self.hs["CfmRegParams btn"],
        ]
        ## ## ## ## ## confirm reg settings -- end

        self.hs["ConfigRegParams box"].children = [
            self.hs["ConfigRegParamsTitle box"],
            self.hs["Fiji&Anchor box"],
            self.hs["MaskOptions box"],
            self.hs["SliSrch&ChunkSz box"],
            self.hs["MrtvOptions box"],
            self.hs["ConfigRegParamsCfm box"],
        ]
        ## ## ## ## define functional widgets in each sub-tab - config registration -- end

        ## ## ## ## define functional widgets in each sub-tab - run registration -- start
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.21 * (self.form_sz[0] - 136)}px",
        }
        self.hs["RunReg box"] = widgets.VBox()
        self.hs["RunReg box"].layout = layout

        ## ## ## ## ## label run_reg box -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["RunRegTitle box"] = widgets.HBox()
        self.hs["RunRegTitle box"].layout = layout
        self.hs["RunRegTitle text"] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + "Run Registration" + "</span>")
        layout = {"background-color": "white", "color": "cyan", "left": "41%"}
        self.hs["RunRegTitle text"].layout = layout
        self.hs["RunRegTitle box"].children = [self.hs["RunRegTitle text"]]
        ## ## ## ## ## label run_reg box -- end

        ## ## ## ## ## run reg & status -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["RunRegCfm box"] = widgets.HBox()
        self.hs["RunRegCfm box"].layout = layout
        layout = {"width": "70%"}
        self.hs["RunReg text"] = widgets.Text(
            description="",
            value="run registration once you are ready ...",
            disabled=True,
        )
        self.hs["RunReg text"].layout = layout
        layout = {"width": "15%"}
        self.hs["RunReg btn"] = widgets.Button(
            description="Run Reg",
            description_tooltip="run registration once you are ready ...",
            disabled=True,
        )
        self.hs["RunReg btn"].style.button_color = "darkviolet"
        self.hs["RunReg btn"].layout = layout

        self.hs["RunReg btn"].on_click(self.RunReg_btn_clk)
        self.hs["RunRegCfm box"].children = [
            self.hs["RunReg text"],
            self.hs["RunReg btn"],
        ]

        ## ## ## ## ## run reg progress
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["RunRegProgress box"] = widgets.HBox()
        self.hs["RunRegProgress box"].layout = layout
        layout = {"width": "100%"}
        self.hs["RunRegProgress bar"] = widgets.IntProgress(
            value=0,
            min=0,
            max=10,
            step=1,
            description="Completing:",
            bar_style="info",
            orientation="horizontal",
        )
        self.hs["RunRegProgress bar"].layout = layout

        self.hs["RunRegProgress box"].children = [
            self.hs["RunRegProgress bar"]
        ]
        ## ## ## ## ## run reg & status -- start

        self.hs["RunReg box"].children = [
            self.hs["RunRegTitle box"],
            self.hs["RunRegCfm box"],
            self.hs["RunRegProgress box"],
        ]
        ## ## ## ## define functional widgets in each sub-tab - run registration -- end

        self.hs["RegSetting form"].children = [
            self.hs["2DRoi box"],
            self.hs["ConfigRegParams box"],
            self.hs["RunReg box"],
        ]
        ## ## ## define 2D_XANES_tabs layout - config registration --end

        ## ## ## define 2D_XANES_tabs layout - review/correct reg & align data -- start
        ## ## ## ## define functional widgets in each sub-tab - review/correct reg -- start
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.35 * (self.form_sz[0] - 136)}px",
        }
        self.hs["RevRegRlt box"] = widgets.VBox()
        self.hs["RevRegRlt box"].layout = layout

        ## ## ## ## ## label review_reg_results box -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["RevRegRltTitle box"] = widgets.HBox()
        self.hs["RevRegRltTitle box"].layout = layout
        self.hs["RevRegRltTitle text"] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + "Review/Correct Reg Results" + "</span>")
        layout = {
            "background-color": "white",
            "color": "cyan",
            "left": "35.7%"
        }
        self.hs["RevRegRltTitle text"].layout = layout
        self.hs["RevRegRltTitle box"].children = [
            self.hs["RevRegRltTitle text"]
        ]
        ## ## ## ## ## label review_reg_results box -- end

        ## ## ## ## ## read alignment file -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["ReadAlignFile box"] = widgets.HBox()
        self.hs["ReadAlignFile box"].layout = layout
        layout = {"width": "70%"}
        self.hs["ReadAlign chbx"] = widgets.Checkbox(
            description="read alignment", value=False, disabled=True)
        self.hs["ReadAlign chbx"].layout = layout
        layout = {"width": "15%"}
        self.hs["ReadAlign btn"] = SelectFilesButton(option="askopenfilename")
        self.hs["ReadAlign btn"].disabled = True
        self.hs["ReadAlign btn"].layout = layout

        self.hs["ReadAlign chbx"].observe(self.ReadAlign_chbx_chg,
                                          names="value")
        self.hs["ReadAlign btn"].on_click(self.ReadAlign_btn_clk)
        self.hs["ReadAlignFile box"].children = [
            self.hs["ReadAlign chbx"],
            self.hs["ReadAlign btn"],
        ]
        ## ## ## ## ## read alignment file -- end

        ## ## ## ## ## reg pair box -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["RegPair box"] = widgets.HBox()
        self.hs["RegPair box"].layout = layout
        layout = {"width": "70%"}
        self.hs["RegPair sldr"] = widgets.IntSlider(value=False,
                                                    disabled=True,
                                                    description="reg pair #")
        self.hs["RegPair sldr"].layout = layout
        layout = {"width": "15%"}
        self.hs["RegPairBad btn"] = widgets.Button(disabled=True,
                                                   description="Bad")
        self.hs["RegPairBad btn"].layout = layout
        self.hs["RegPairBad btn"].style.button_color = "darkviolet"

        self.hs["RegPair sldr"].observe(self.RegPair_sldr_chg, names="value")
        self.hs["RegPairBad btn"].on_click(self.RegPairBad_btn_clk)
        self.hs["RegPair box"].children = [
            self.hs["RegPair sldr"],
            self.hs["RegPairBad btn"],
        ]
        ## ## ## ## ## reg pair box -- end

        ## ## ## ## ## zshift box -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["CorrSht box"] = widgets.HBox()
        self.hs["CorrSht box"].layout = layout
        layout = {"width": "30%"}
        self.hs["XShift text"] = widgets.FloatText(value=0,
                                                   disabled=True,
                                                   min=-100,
                                                   max=100,
                                                   step=0.5,
                                                   description="x shift")
        self.hs["XShift text"].layout = layout
        layout = {"width": "30%"}
        self.hs["YShift text"] = widgets.FloatText(value=0,
                                                   disabled=True,
                                                   min=-100,
                                                   max=100,
                                                   step=0.5,
                                                   description="y shift")
        self.hs["YShift text"].layout = layout
        layout = {"left": "9.5%", "width": "15%"}
        self.hs["Record btn"] = widgets.Button(description="Record",
                                               description_tooltip="Record",
                                               disabled=True)
        self.hs["Record btn"].layout = layout
        self.hs["Record btn"].style.button_color = "darkviolet"

        self.hs["XShift text"].observe(self.XShift_text_chg, names="value")
        self.hs["YShift text"].observe(self.YShift_text_chg, names="value")
        self.hs["Record btn"].on_click(self.Record_btn_clk)
        self.hs["CorrSht box"].children = [
            self.hs["XShift text"],
            self.hs["YShift text"],
            self.hs["Record btn"],
        ]
        ## ## ## ## ## zshift box -- end

        ## ## ## ## ## confirm review results box -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["RevRegRltCfm box"] = widgets.HBox()
        self.hs["RevRegRltCfm box"].layout = layout
        layout = {"width": "70%", "display": "inline_flex"}
        self.hs["CfmRevRlt text"] = widgets.Text(
            description="",
            value="Confirm after you finish reg review ...",
            disabled=True,
        )
        self.hs["CfmRevRlt text"].layout = layout
        layout = {"left": "0.%", "width": "15%"}
        self.hs["CfmRevRlt btn"] = widgets.Button(
            description="Confirm",
            description_tooltip="Confirm after you finish reg review ...",
            disabled=True,
        )
        self.hs["CfmRevRlt btn"].style.button_color = "darkviolet"
        self.hs["CfmRevRlt btn"].layout = layout

        self.hs["CfmRevRlt btn"].on_click(self.CfmRevRlt_btn_clk)
        self.hs["RevRegRltCfm box"].children = [
            self.hs["CfmRevRlt text"],
            self.hs["CfmRevRlt btn"],
        ]
        ## ## ## ## ## confirm review results box -- end

        self.hs["RevRegRlt box"].children = [
            self.hs["RevRegRltTitle box"],
            self.hs["ReadAlignFile box"],
            self.hs["RegPair box"],
            self.hs["CorrSht box"],
            self.hs["RevRegRltCfm box"],
        ]
        ## ## ## ## define functional widgets in each sub-tab - review/correct reg -- end

        ## ## ## ## define functional widgets in each sub-tab - align data -- start
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.21 * (self.form_sz[0] - 136)}px",
        }
        self.hs["AlignImg box"] = widgets.VBox()
        self.hs["AlignImg box"].layout = layout

        ## ## ## ## ## label align_recon box -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["AlignImgTitle box"] = widgets.HBox()
        self.hs["AlignImgTitle box"].layout = layout
        self.hs["AlignImgTitle text"] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + "Align Images" + "</span>")
        layout = {"background-color": "white", "color": "cyan", "left": "41%"}
        self.hs["AlignImgTitle text"].layout = layout

        self.hs["AlignImgTitle box"].children = [self.hs["AlignImgTitle text"]]
        ## ## ## ## ## label align_recon box -- end

        ## ## ## ## ## define run reg & status -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["AlignImgCfm box"] = widgets.HBox()
        self.hs["AlignImgCfm box"].layout = layout
        layout = {"width": "70%"}
        self.hs["AlignImg text"] = widgets.Text(
            description="",
            value="Confirm to proceed alignment ...",
            disabled=True)
        self.hs["AlignImg text"].layout = layout
        layout = {"width": "15%"}
        self.hs["AlignImg btn"] = widgets.Button(
            description="Align",
            description_tooltip=
            "This will perform xanes2D alignment according to your configurations ...",
            disabled=True,
        )
        self.hs["AlignImg btn"].style.button_color = "darkviolet"
        self.hs["AlignImg btn"].layout = layout

        self.hs["AlignImg btn"].on_click(self.AlignImg_btn_clk)
        self.hs["AlignImgCfm box"].children = [
            self.hs["AlignImg text"],
            self.hs["AlignImg btn"],
        ]
        ## ## ## ## ## define run reg & status -- end

        ## ## ## ## ## define run reg progress -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["AlignPrgs box"] = widgets.HBox()
        self.hs["AlignPrgs box"].layout = layout
        layout = {"width": "100%"}
        self.hs["AlignPrgs bar"] = widgets.IntProgress(
            value=0,
            min=0,
            max=10,
            step=1,
            description="Completing:",
            bar_style="info",
            orientation="horizontal",
        )
        self.hs["AlignPrgs bar"].layout = layout
        self.hs["AlignPrgs box"].children = [self.hs["AlignPrgs bar"]]
        ## ## ## ## ## define run reg progress -- end

        self.hs["AlignImg box"].children = [
            self.hs["AlignImgTitle box"],
            self.hs["AlignImgCfm box"],
            self.hs["AlignPrgs box"],
        ]
        ## ## ## ## define functional widgets in each sub-tab - align recon in register/review/shift TAB -- end

        ## ## ## define 2D_XANES_tabs layout - visualization&analysis reg & align data -- start
        ## ## ## ## define functional widgets in each sub-tab - visualizaton TAB -- start
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.14 * (self.form_sz[0] - 136)}px",
        }
        self.hs["VisImg box"] = widgets.VBox()
        self.hs["VisImg box"].layout = layout

        ## ## ## ## ## define visualization title -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["VisImgTitle box"] = widgets.HBox()
        self.hs["VisImgTitle box"].layout = layout
        self.hs["VisImgTitle text"] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + "Visualize Images" + "</span>")
        layout = {"background-color": "white", "color": "cyan", "left": "42%"}
        self.hs["VisImgTitle text"].layout = layout
        self.hs["VisImgTitle box"].children = [self.hs["VisImgTitle text"]]
        ## ## ## ## ## define visualization title -- end

        ## ## ## ## ## define visualization&confirm -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["Vis&Cfm box"] = widgets.HBox()
        self.hs["Vis&Cfm box"].layout = layout
        layout = {"width": "60%"}
        self.hs["Vis sldr"] = widgets.IntSlider(description="eng #",
                                                min=1,
                                                max=10,
                                                disabled=True)
        self.hs["Vis sldr"].layout = layout
        layout = {"width": "10%"}
        self.hs["VisEng text"] = widgets.FloatText(description="",
                                                   value=0,
                                                   disabled=True)
        self.hs["VisEng text"].layout = layout
        layout = {"width": "15%"}
        self.hs["VisR&C chbx"] = widgets.Checkbox(description="Auto R&C",
                                                  value=True,
                                                  disabled=True)
        self.hs["VisR&C chbx"].layout = layout
        layout = {"width": "15%"}
        self.hs["SpecInRoi btn"] = widgets.Button(
            description="spec in roi",
            description_tooltip=
            "This will perform xanes2D alignment according to your configurations ...",
            disabled=True,
        )
        self.hs["SpecInRoi btn"].style.button_color = "darkviolet"
        self.hs["SpecInRoi btn"].layout = layout

        self.hs["Vis sldr"].observe(self.Vis_sldr_chg, names="value")
        self.hs["VisR&C chbx"].observe(self.VisRC_chbx_chg, names="value")
        self.hs["SpecInRoi btn"].on_click(self.SpecInRoi_btn_clk)
        self.hs["Vis&Cfm box"].children = [
            self.hs["Vis sldr"],
            self.hs["VisEng text"],
            self.hs["VisR&C chbx"],
            self.hs["SpecInRoi btn"],
        ]
        ## ## ## ## ## define visualization&confirm -- end

        self.hs["VisImg box"].children = [
            self.hs["VisImgTitle box"],
            self.hs["Vis&Cfm box"],
        ]

        ## ## ## ## define functional widgets in each sub-tab - visualizaton TAB -- end
        self.hs["Reg&Rev form"].children = [
            self.hs["RevRegRlt box"],
            self.hs["AlignImg box"],
            self.hs["VisImg box"],
        ]
        ## ## ## define 2D_XANES_tabs layout - review/correct reg & align data -- end

        self.hs["Fitting form"].children = [
            self.xanes_fit_gui_h.hs["Fitting form"]
        ]
        ## ## ## define 2D_XANES_tabs layout - visualization&analysis reg & align data -- end

        self.hs["Analysis form"].children = [
            self.xanes_ana_gui_h.hs["Ana form"]
        ]
        ## ## ## define 2D_XANES_tabs layout - analysis box -- end

        self.hs["SelRawH5Path btn"].initialdir = self.global_h.cwd
        self.hs["SelSaveTrial btn"].initialdir = self.global_h.cwd
        self.hs["ReadAlign btn"].initialdir = self.global_h.cwd

    def SelRawH5Path_btn_clk(self, a):
        if len(a.files[0]) != 0:
            self.xanes_file_raw_h5_filename = os.path.abspath(a.files[0])
            self.hs["SelRawH5Path btn"].initialdir = os.path.dirname(
                os.path.abspath(a.files[0]))
            self.hs["SelSaveTrial btn"].initialdir = os.path.dirname(
                os.path.abspath(a.files[0]))
            update_json_content(
                self.global_h.GUI_cfg_file,
                {"cwd": os.path.dirname(os.path.abspath(a.files[0]))},
            )
            self.global_h.cwd = os.path.dirname(os.path.abspath(a.files[0]))
            self.xanes_file_raw_h5_set = True
        else:
            self.hs["SelRawH5Path text"].value = "Select raw h5 file ..."
            self.xanes_file_raw_h5_set = False
        self.xanes_file_configured = False
        self.hs["CfmFile&Path text"].value = "Please comfirm your change ..."
        self.boxes_logic()

    def SelSaveTrial_btn_clk(self, a):
        if self.xanes_file_analysis_option == "Do New Reg":
            if len(a.files[0]) != 0:
                self.xanes_save_trial_reg_filename_template = a.files[0]
                self.hs["SelSaveTrial btn"].initialdir = os.path.dirname(
                    os.path.abspath(a.files[0]))
                update_json_content(
                    self.global_h.GUI_cfg_file,
                    {"cwd": os.path.dirname(os.path.abspath(a.files[0]))},
                )
                self.global_h.cwd = os.path.dirname(os.path.abspath(
                    a.files[0]))
                self.xanes_file_save_trial_set = True
                self.xanes_file_reg_file_set = False
                self.xanes_file_config_file_set = False
            else:
                self.hs[
                    "SelSaveTrial text"].value = "Save trial registration as ..."
                self.xanes_file_save_trial_set = False
                self.xanes_file_reg_file_set = False
                self.xanes_file_config_file_set = False
            self.xanes_file_configured = False
            self.hs[
                "CfmFile&Path text"].value = "Please comfirm your change ..."
        elif self.xanes_file_analysis_option == "Read Config File":
            if len(a.files[0]) != 0:
                self.xanes_file_save_trial_reg_config_filename_original = a.files[
                    0]
                b = ""
                t = time.strptime(time.asctime())
                for ii in range(6):
                    b += str(t[ii]).zfill(2) + "-"
                self.xanes_file_save_trial_reg_config_filename = (
                    os.path.abspath(a.files[0]).split("config")[0] +
                    "config_" + b.strip("-") + ".json")
                self.hs["SelSaveTrial btn"].initialdir = os.path.dirname(
                    os.path.abspath(a.files[0]))
                self.hs["SelSaveTrial btn"].initialfile = os.path.basename(
                    a.files[0])
                self.hs["SelSaveTrial text"].value = (
                    f"{self.xanes_file_save_trial_reg_config_filename} is Read ..."
                )
                update_json_content(
                    self.global_h.GUI_cfg_file,
                    {"cwd": os.path.dirname(os.path.abspath(a.files[0]))},
                )
                self.xanes_file_save_trial_set = False
                self.xanes_file_reg_file_set = False
                self.xanes_file_config_file_set = True
            else:
                self.hs[
                    "SelSaveTrial text"].value = "Save Existing Configuration File ..."
                self.xanes_file_save_trial_set = False
                self.xanes_file_reg_file_set = False
                self.xanes_file_config_file_set = False
            self.xanes_file_configured = False
            self.hs[
                "CfmFile&Path text"].value = "Please comfirm your change ..."
        elif self.xanes_file_analysis_option == "Reg By Shift":
            if len(a.files[0]) != 0:
                self.xanes_file_save_trial_reg_config_filename_original = a.files[
                    0]
                b = ""
                t = time.strptime(time.asctime())
                for ii in range(6):
                    b += str(t[ii]).zfill(2) + "-"
                self.xanes_file_save_trial_reg_config_filename = (
                    os.path.abspath(a.files[0]).split("config")[0] +
                    "config_" + b.strip("-") + ".json")
                self.hs["SelSaveTrial btn"].initialdir = os.path.dirname(
                    os.path.abspath(a.files[0]))
                self.hs["SelSaveTrial btn"].initialfile = os.path.basename(
                    a.files[0])
                self.hs["SelSaveTrial text"].value = (
                    f"{self.xanes_file_save_trial_reg_config_filename} is Read ..."
                )
                update_json_content(
                    self.global_h.GUI_cfg_file,
                    {"cwd": os.path.dirname(os.path.abspath(a.files[0]))},
                )
                self.xanes_file_save_trial_set = False
                self.xanes_file_reg_file_set = False
                self.xanes_file_config_file_set = True
            else:
                self.hs[
                    "SelSaveTrial text"].value = "Save Existing Configuration File ..."
                self.xanes_file_save_trial_set = False
                self.xanes_file_reg_file_set = False
                self.xanes_file_config_file_set = False
            self.xanes_file_configured = False
            self.hs[
                "CfmFile&Path text"].value = "Please comfirm your change ..."
        elif self.xanes_file_analysis_option == "Do Analysis":
            self.hs["SelSaveTrial btn"].style.button_color = "lightgreen"
            if len(a.files[0]) != 0:
                self.xanes_save_trial_reg_filename = os.path.abspath(
                    a.files[0])
                b = ""
                t = time.strptime(time.asctime())
                for ii in range(6):
                    b += str(t[ii]).zfill(2) + "-"
                template = ""
                for ii in os.path.abspath(a.files[0]).split("_")[:-2]:
                    template += ii + "_"
                self.xanes_file_save_trial_reg_config_filename = (
                    template + "config_" + b.strip("-") + ".json")
                self.hs["SelSaveTrial btn"].initialdir = os.path.dirname(
                    os.path.abspath(a.files[0]))
                self.hs["SelSaveTrial btn"].initialfile = os.path.basename(
                    a.files[0])
                update_json_content(
                    self.global_h.GUI_cfg_file,
                    {"cwd": os.path.dirname(os.path.abspath(a.files[0]))},
                )
                self.hs["SelSaveTrial text"].value = (
                    f"{self.xanes_file_save_trial_reg_config_filename} is Read ..."
                )
                self.xanes_file_save_trial_set = False
                self.xanes_file_reg_file_set = True
                self.xanes_file_config_file_set = False
            else:
                self.hs[
                    "SelSaveTrial text"].value = "Read Existing Registration File ..."
                self.xanes_file_save_trial_set = False
                self.xanes_file_reg_file_set = False
                self.xanes_file_config_file_set = False
            self.xanes_file_configured = False
            self.hs[
                "CfmFile&Path text"].value = "Please comfirm your change ..."
        self.boxes_logic()

    def FilePathOptions_drpdn_chg(self, a):
        restart(self, dtype="2D_XANES")
        restart(self.xanes_fit_gui_h, dtype="XANES_FITTING")
        self.xanes_file_analysis_option = a["owner"].value
        self.xanes_file_configured = False
        if self.xanes_file_analysis_option == "Do New Reg":
            self.hs["SelRawH5Path btn"].disabled = False
            self.hs["SelSaveTrial btn"].option = "asksaveasfilename"
            self.hs["SelSaveTrial btn"].description = "Save Reg File"
            self.hs[
                "SelSaveTrial text"].value = "Save trial registration as ..."
            self.hs["ReadAlign chbx"].value = False
        elif self.xanes_file_analysis_option == "Read Config File":
            self.hs["SelRawH5Path btn"].disabled = True
            self.hs["SelSaveTrial btn"].option = "askopenfilename"
            self.hs["SelSaveTrial btn"].description = "Read Config"
            self.hs["SelSaveTrial btn"].open_filetypes = (
                ("json files", "*.json"),
                ("text files", "*.txt"),
            )
            self.hs[
                "SelSaveTrial text"].value = "Save Existing Configuration File ..."
        elif self.xanes_file_analysis_option == "Reg By Shift":
            self.hs["SelRawH5Path btn"].disabled = True
            self.hs["SelSaveTrial btn"].option = "askopenfilename"
            self.hs["SelSaveTrial btn"].description = "Read Config"
            self.hs["SelSaveTrial btn"].open_filetypes = (
                ("json files", "*.json"),
                ("text files", "*.txt"),
            )
            self.hs[
                "SelSaveTrial text"].value = "Save Existing Configuration File ..."
        elif self.xanes_file_analysis_option == "Do Analysis":
            self.hs["SelRawH5Path btn"].disabled = True
            self.hs["SelSaveTrial btn"].option = "askopenfilename"
            self.hs["SelSaveTrial btn"].description = "Read Reg File"
            self.hs["SelSaveTrial btn"].open_filetypes = (("h5 files",
                                                           "*.h5"), )
            self.hs[
                "SelSaveTrial text"].value = "Read Existing Registration File ..."
        self.hs["SelSaveTrial btn"].icon = "square-o"
        self.hs["SelSaveTrial btn"].style.button_color = "orange"
        self.hs["CfmRevRlt text"].value = "Please comfirm your change ..."
        self.boxes_logic()

    def CfmFile_Path_btn_clk(self, a):
        if self.xanes_file_analysis_option == "Do New Reg":
            if not self.xanes_file_raw_h5_set:
                self.hs[
                    "CfmFile&Path text"].value = "Please specifiy raw h5 file ..."
                self.xanes_file_configured = False
            elif not self.xanes_file_save_trial_set:
                self.hs[
                    "CfmFile&Path text"].value = "Please specifiy where to save trial reg result ..."
                self.xanes_file_configured = False
            else:
                b = ""
                t = time.strptime(time.asctime())
                for ii in range(6):
                    b += str(t[ii]).zfill(2) + "-"
                self.xanes_save_trial_reg_filename = os.path.join(
                    os.path.dirname(
                        os.path.abspath(
                            self.xanes_save_trial_reg_filename_template)),
                    os.path.basename(
                        os.path.abspath(
                            self.xanes_save_trial_reg_filename_template)
                    ).split(".")[0] + "_" + os.path.basename(
                        self.xanes_file_raw_h5_filename).split(".")[0] + "_" +
                    b.strip("-") + ".h5",
                )
                self.xanes_file_save_trial_reg_config_filename = os.path.join(
                    os.path.dirname(
                        os.path.abspath(
                            self.xanes_save_trial_reg_filename_template)),
                    os.path.basename(
                        os.path.abspath(
                            self.xanes_save_trial_reg_filename_template)
                    ).split(".")[0] + "_" + os.path.basename(
                        self.xanes_file_raw_h5_filename).split(".")[0] +
                    "_config_" + b.strip("-") + ".json",
                )

                self.xanes_config_eng_list = self.reader(
                    self.xanes_file_raw_h5_filename,
                    dtype="eng",
                    sli=[None],
                    cfg=self.global_h.io_xanes2D_cfg,
                )
                if self.xanes_config_eng_list.max() < 70:
                    self.xanes_config_eng_list *= 1000

                self.hs["EngPointsRange sldr"].value = [0, 1]
                self.hs["EngPointsRange sldr"].min = 0
                self.hs["EngPointsRange sldr"].max = (
                    self.xanes_config_eng_list.shape[0] - 1)
                self.hs["EngStart text"].value = self.xanes_config_eng_list[0]
                self.hs["EngEnd text"].value = self.xanes_config_eng_list[1]

                self.xanes_eng_id_s = self.hs["EngPointsRange sldr"].value[0]
                self.xanes_eng_id_e = self.hs["EngPointsRange sldr"].value[1]
                self.hs[
                    "CfmFile&Path text"].value = "XANES2D file config is done ..."
                self.update_xanes2D_config()
                self.xanes_review_reg_best_match_filename = (os.path.splitext(
                    self.xanes_file_save_trial_reg_config_filename)[0].replace(
                        "config", "reg_best_match") + ".json")
                self.xanes_file_configured = True
        elif self.xanes_file_analysis_option == "Read Config File":
            if not self.xanes_file_config_file_set:
                self.hs[
                    "CfmFile&Path text"].value = "Please specifiy the configuration file to be read ..."
                self.xanes_file_configured = False
            else:
                self.read_xanes2D_config()
                self.set_xanes2D_variables()

                self.xanes_config_eng_list = self.reader(
                    self.xanes_file_raw_h5_filename,
                    dtype="eng",
                    sli=[None],
                    cfg=self.global_h.io_xanes2D_cfg,
                )
                if self.xanes_config_eng_list.max() < 70:
                    self.xanes_config_eng_list *= 1000

                if self.xanes_config_is_raw:
                    if self.xanes_config_use_alternative_flat:
                        if self.xanes_config_alternative_flat_set:
                            self.xanes_img = -np.log((self.reader(
                                self.xanes_file_raw_h5_filename,
                                dtype="data",
                                sli=[None],
                                cfg=self.global_h.io_xanes2D_cfg,
                            ) - self.reader(
                                self.xanes_file_raw_h5_filename,
                                dtype="dark",
                                sli=[None],
                                cfg=self.global_h.io_xanes2D_cfg,
                            )) / (self.reader(
                                self.xanes_config_alternative_flat_filename,
                                dtype="flat",
                                sli=[None],
                                cfg=self.global_h.io_xanes2D_cfg,
                            ) - self.reader(
                                self.xanes_config_alternative_flat_filename,
                                dtype="dark",
                                sli=[None],
                                cfg=self.global_h.io_xanes2D_cfg,
                            )))
                            self.xanes_img[np.isinf(self.xanes_img)] = 0
                            self.xanes_img[np.isnan(self.xanes_img)] = 0
                            self.hs["EngPointsRange sldr"].value = [
                                0,
                                self.xanes_img.shape[0] - 1,
                            ]
                            self.xanes_config_raw_img_readed = True
                        else:
                            self.xanes_config_raw_img_readed = False
                    else:
                        self.xanes_img = -np.log((self.reader(
                            self.xanes_file_raw_h5_filename,
                            dtype="data",
                            sli=[None],
                            cfg=self.global_h.io_xanes2D_cfg,
                        ) - self.reader(
                            self.xanes_file_raw_h5_filename,
                            dtype="dark",
                            sli=[None],
                            cfg=self.global_h.io_xanes2D_cfg,
                        )) / (self.reader(
                            self.xanes_file_raw_h5_filename,
                            dtype="flat",
                            sli=[None],
                            cfg=self.global_h.io_xanes2D_cfg,
                        ) - self.reader(
                            self.xanes_file_raw_h5_filename,
                            dtype="dark",
                            sli=[None],
                            cfg=self.global_h.io_xanes2D_cfg,
                        )))
                        self.xanes_img[np.isinf(self.xanes_img)] = 0
                        self.xanes_img[np.isnan(self.xanes_img)] = 0
                        self.xanes_config_raw_img_readed = True
                else:
                    self.xanes_img = -np.log(
                        self.reader(
                            self.xanes_file_raw_h5_filename,
                            dtype="data",
                            sli=[None],
                            cfg=self.global_h.io_xanes2D_cfg,
                        ))
                    self.xanes_img[np.isinf(self.xanes_img)] = 0
                    self.xanes_img[np.isnan(self.xanes_img)] = 0
                    self.hs["EngPointsRange sldr"].value = [
                        0,
                        self.xanes_img.shape[0] - 1,
                    ]
                    self.xanes_config_raw_img_readed = True

                if self.xanes_data_configured:
                    self.xanes_fit_eng_list = self.reader(
                        self.xanes_file_raw_h5_filename,
                        dtype="eng",
                        sli=[None],
                        cfg=self.global_h.io_xanes2D_cfg,
                    )[self.xanes_eng_id_s:self.xanes_eng_id_e + 1]
                    if self.xanes_fit_eng_list.max() < 70:
                        self.xanes_fit_eng_list *= 1000

                if self.xanes_roi_configured:
                    self.xanes_img_roi = self.xanes_img[
                        self.xanes_eng_id_s:self.xanes_eng_id_e + 1,
                        self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                        self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ]

                if self.xanes_reg_done:
                    with h5py.File(self.xanes_save_trial_reg_filename,
                                   "r") as f:
                        self.xanes_review_aligned_img_original = f[
                            "/trial_registration/trial_reg_results/{0}/trial_reg_img{0}"
                            .format(str(0).zfill(3))][:]
                        self.xanes_review_aligned_img = f[
                            "/trial_registration/trial_reg_results/{0}/trial_reg_img{0}"
                            .format(str(0).zfill(3))][:]
                        self.xanes_review_fixed_img = f[
                            "/trial_registration/trial_reg_results/{0}/trial_reg_fixed{0}"
                            .format(str(0).zfill(3))][:]
                    self.xanes_review_shift_dict = {}
                self.xanes_review_reg_best_match_filename = (os.path.splitext(
                    self.xanes_file_save_trial_reg_config_filename)[0].replace(
                        "config", "reg_best_match") + ".json")
                self.xanes_file_configured = True

                self.hs["FijiMaskViewer chbx"].value = False

                if self.xanes_alignment_done:
                    with h5py.File(self.xanes_save_trial_reg_filename,
                                   "r") as f:
                        self.xanes_fit_data_shape = f[
                            "/registration_results/reg_results/registered_xanes2D"].shape
                        self.xanes_img_roi = f[
                            "/registration_results/reg_results/registered_xanes2D"][:]
                        self.xanes_fit_eng_list = f[
                            "/registration_results/reg_results/eng_list"][:]

                    self.hs["Vis sldr"].min = 1
                    self.hs["Vis sldr"].max = self.xanes_fit_eng_list.shape[0]

                    self.xanes_review_reg_best_match_filename = (
                        os.path.splitext(
                            self.xanes_file_save_trial_reg_config_filename)
                        [0].replace("config", "reg_best_match") + ".json")
                    self.xanes_element = determine_element(
                        self.xanes_fit_eng_list)
                    tem = determine_fitting_energy_range(self.xanes_element)
                    self.xanes_fit_edge_eng = tem[0]
                    self.xanes_fit_wl_fit_eng_s = tem[1]
                    self.xanes_fit_wl_fit_eng_e = tem[2]
                    self.xanes_fit_pre_edge_e = tem[3]
                    self.xanes_fit_post_edge_s = tem[4]
                    self.xanes_fit_edge_0p5_fit_s = tem[5]
                    self.xanes_fit_edge_0p5_fit_e = tem[6]
                    self.xanes_fit_gui_h.hs["FitEngRagOptn drpdn"].value = "wl"
                    self.xanes_fit_gui_h.hs[
                        "FitEngRagOptn drpdn"].value = "full"
                    if (self.xanes_fit_eng_list.min() >
                        (self.xanes_fit_edge_eng - 50)) and (
                            self.xanes_fit_eng_list.max() <
                            (self.xanes_fit_edge_eng + 50)):
                        self.xanes_fit_gui_h.hs[
                            "FitEngRagOptn drpdn"].value = "wl"
                        self.xanes_fit_gui_h.hs[
                            "FitEngRagOptn drpdn"].disabled = True
                        self.xanes_fit_type = "wl"
                    else:
                        self.xanes_fit_gui_h.hs[
                            "FitEngRagOptn drpdn"].value = "full"
                        self.xanes_fit_gui_h.hs[
                            "FitEngRagOptn drpdn"].disabled = False
                        self.xanes_fit_type = "full"
                self.set_xanes2D_handles()
                self.set_xanes2D_variables()
                fiji_viewer_off(self.global_h, self, viewer_name="all")
        elif self.xanes_file_analysis_option == "Reg By Shift":
            if not self.xanes_file_config_file_set:
                self.hs[
                    "CfmFile&Path text"].value = "Please specifiy the configuration file to be read ..."
                self.xanes_file_configured = False
            else:
                self.read_xanes2D_config()
                self.set_xanes2D_variables()
                if self.xanes_reg_review_done:
                    self.xanes_config_eng_list = self.reader(
                        self.xanes_file_raw_h5_filename,
                        dtype="eng",
                        sli=[None],
                        cfg=self.global_h.io_xanes2D_cfg,
                    )
                    if self.xanes_config_eng_list.max() < 70:
                        self.xanes_config_eng_list *= 1000

                    if self.xanes_config_is_raw:
                        if self.xanes_config_use_alternative_flat:
                            if self.xanes_config_alternative_flat_set:
                                self.xanes_img = -np.log((self.reader(
                                    self.xanes_file_raw_h5_filename,
                                    dtype="data",
                                    sli=[None],
                                    cfg=self.global_h.io_xanes2D_cfg,
                                ) - self.reader(
                                    self.xanes_file_raw_h5_filename,
                                    dtype="dark",
                                    sli=[None],
                                    cfg=self.global_h.io_xanes2D_cfg,
                                )) / (self.reader(
                                    self.xanes_config_alternative_flat_filename,
                                    dtype="flat",
                                    sli=[None],
                                    cfg=self.global_h.io_xanes2D_cfg,
                                ) - self.reader(
                                    self.xanes_config_alternative_flat_filename,
                                    dtype="dark",
                                    sli=[None],
                                    cfg=self.global_h.io_xanes2D_cfg,
                                )))
                                self.xanes_img[np.isinf(self.xanes_img)] = 0
                                self.xanes_img[np.isnan(self.xanes_img)] = 0
                                self.hs["EngPointsRange sldr"].value = [
                                    0,
                                    self.xanes_img.shape[0] - 1,
                                ]
                                self.xanes_config_raw_img_readed = True
                            else:
                                self.xanes_config_raw_img_readed = False
                        else:
                            self.xanes_img = -np.log((self.reader(
                                self.xanes_file_raw_h5_filename,
                                dtype="data",
                                sli=[None],
                                cfg=self.global_h.io_xanes2D_cfg,
                            ) - self.reader(
                                self.xanes_file_raw_h5_filename,
                                dtype="dark",
                                sli=[None],
                                cfg=self.global_h.io_xanes2D_cfg,
                            )) / (self.reader(
                                self.xanes_file_raw_h5_filename,
                                dtype="flat",
                                sli=[None],
                                cfg=self.global_h.io_xanes2D_cfg,
                            ) - self.reader(
                                self.xanes_file_raw_h5_filename,
                                dtype="dark",
                                sli=[None],
                                cfg=self.global_h.io_xanes2D_cfg,
                            )))
                            self.xanes_img[np.isinf(self.xanes_img)] = 0
                            self.xanes_img[np.isnan(self.xanes_img)] = 0
                            self.xanes_config_raw_img_readed = True
                    else:
                        self.xanes_img = -np.log(
                            self.reader(
                                self.xanes_file_raw_h5_filename,
                                dtype="data",
                                sli=[None],
                                cfg=self.global_h.io_xanes2D_cfg,
                            ))
                        self.xanes_img[np.isinf(self.xanes_img)] = 0
                        self.xanes_img[np.isnan(self.xanes_img)] = 0
                        self.hs["EngPointsRange sldr"].value = [
                            0,
                            self.xanes_img.shape[0] - 1,
                        ]
                        self.xanes_config_raw_img_readed = True

                    if self.xanes_data_configured:
                        self.xanes_fit_eng_list = self.reader(
                            self.xanes_file_raw_h5_filename,
                            dtype="eng",
                            sli=[None],
                            cfg=self.global_h.io_xanes2D_cfg,
                        )[self.xanes_eng_id_s:self.xanes_eng_id_e + 1]
                        if self.xanes_fit_eng_list.max() < 70:
                            self.xanes_fit_eng_list *= 1000

                    if self.xanes_roi_configured:
                        self.xanes_img_roi = self.xanes_img[
                            self.xanes_eng_id_s:self.xanes_eng_id_e + 1,
                            self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                            self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ]

                    if self.xanes_reg_done:
                        with h5py.File(self.xanes_save_trial_reg_filename,
                                       "r") as f:
                            self.xanes_review_aligned_img_original = f[
                                "/trial_registration/trial_reg_results/{0}/trial_reg_img{0}"
                                .format(str(0).zfill(3))][:]
                            self.xanes_review_aligned_img = f[
                                "/trial_registration/trial_reg_results/{0}/trial_reg_img{0}"
                                .format(str(0).zfill(3))][:]
                            self.xanes_review_fixed_img = f[
                                "/trial_registration/trial_reg_results/{0}/trial_reg_fixed{0}"
                                .format(str(0).zfill(3))][:]
                        self.xanes_review_shift_dict = {}
                    self.xanes_review_reg_best_match_filename = (
                        os.path.splitext(
                            self.xanes_file_save_trial_reg_config_filename)
                        [0].replace("config", "reg_best_match") + ".json")
                    self.xanes_file_configured = True

                    self.set_xanes2D_handles()
                    self.set_xanes2D_variables()
                    self.xanes_data_configured = False
                    self.xanes_roi_configured = False
                    self.xanes_reg_review_done = False
                    self.xanes_alignment_done = False
                    self.xanes_file_analysis_option = "Reg By Shift"
                    fiji_viewer_off(self.global_h, self, viewer_name="all")
                    self.hs[
                        "CfmFile&Path text"].value = "XANES2D file config is done ..."
                else:
                    self.xanes_file_configured = False
                    self.hs["CfmFile&Path text"].value = (
                        "To use this option, a config up to reg review is needed ..."
                    )
        elif self.xanes_file_analysis_option == "Do Analysis":
            if not self.xanes_file_reg_file_set:
                self.hs[
                    "CfmFile&Path text"].value = "Please specifiy the aligned data to be read..."
                self.xanes_file_configured = False
                self.xanes_reg_review_done = False
                self.xanes_data_configured = False
                self.xanes_roi_configured = False
                self.xanes_reg_params_configured = False
                self.xanes_reg_done = False
                self.xanes_alignment_done = False
                self.xanes_fit_eng_configured = False
            else:
                self.xanes_file_configured = True
                self.xanes_reg_review_done = False
                self.xanes_data_configured = False
                self.xanes_roi_configured = False
                self.xanes_reg_params_configured = False
                self.xanes_reg_done = False
                self.xanes_alignment_done = True
                self.xanes_fit_eng_configured = False
                with h5py.File(self.xanes_save_trial_reg_filename, "r") as f:
                    self.xanes_fit_data_shape = f[
                        "/registration_results/reg_results/registered_xanes2D"].shape
                    self.xanes_img_roi = f[
                        "/registration_results/reg_results/registered_xanes2D"][:]
                    self.xanes_fit_eng_list = f[
                        "/registration_results/reg_results/eng_list"][:]
                    if self.xanes_fit_eng_list.max() < 70:
                        self.xanes_fit_eng_list *= 1000

                data_state, viewer_state = fiji_viewer_state(
                    self.global_h, self, viewer_name="xanes2D_analysis_viewer")
                if not viewer_state:
                    fiji_viewer_on(self.global_h,
                                   self,
                                   viewer_name="xanes2D_analysis_viewer")
                self.global_h.xanes2D_fiji_windows["xanes2D_analysis_viewer"][
                    "ip"].setImage(self.global_h.ij.convert().convert(
                        self.global_h.ij.dataset().create(
                            self.global_h.ij.py.to_java(self.xanes_img_roi)),
                        self.global_h.ImagePlusClass,
                    ))
                self.global_h.xanes2D_fiji_windows["xanes2D_analysis_viewer"][
                    "ip"].setSlice(0)
                self.global_h.ij.py.run_macro(
                    """run("Enhance Contrast", "saturated=0.35")""")

                self.hs["Vis sldr"].min = 1
                self.hs["Vis sldr"].max = self.xanes_fit_eng_list.shape[0]
                self.xanes_review_reg_best_match_filename = (os.path.splitext(
                    self.xanes_file_save_trial_reg_config_filename)[0].replace(
                        "config", "reg_best_match") + ".json")
                self.xanes_element = determine_element(self.xanes_fit_eng_list)
                tem = determine_fitting_energy_range(self.xanes_element)
                self.xanes_fit_edge_eng = tem[0]
                self.xanes_fit_wl_fit_eng_s = tem[1]
                self.xanes_fit_wl_fit_eng_e = tem[2]
                self.xanes_fit_pre_edge_e = tem[3]
                self.xanes_fit_post_edge_s = tem[4]
                self.xanes_fit_edge_0p5_fit_s = tem[5]
                self.xanes_fit_edge_0p5_fit_e = tem[6]
                self.xanes_fit_gui_h.hs["FitEngRagOptn drpdn"].value = "wl"
                self.xanes_fit_gui_h.hs["FitEngRagOptn drpdn"].value = "full"
                if (self.xanes_fit_eng_list.min() >
                    (self.xanes_fit_edge_eng - 50)) and (
                        self.xanes_fit_eng_list.max() <
                        (self.xanes_fit_edge_eng + 50)):
                    self.xanes_fit_gui_h.hs["FitEngRagOptn drpdn"].value = "wl"
                    self.xanes_fit_gui_h.hs[
                        "FitEngRagOptn drpdn"].disabled = True
                    self.xanes_fit_type = "wl"
                else:
                    self.xanes_fit_gui_h.hs[
                        "FitEngRagOptn drpdn"].value = "full"
                    self.xanes_fit_gui_h.hs[
                        "FitEngRagOptn drpdn"].disabled = False
                    self.xanes_fit_type = "full"
        self.boxes_logic()

    def IsRaw_chbx_change(self, a):
        self.xanes_config_is_raw = a["owner"].value
        self.xanes_config_raw_img_readed = False
        self.boxes_logic()

    def NormScale_text_chg(self, a):
        self.xanes_config_img_scalar = a["owner"].value
        self.xanes_config_raw_img_readed = False
        self.boxes_logic()

    def ConfigDataLoadImg_btn_clk(self, a):
        self.xanes_config_is_raw = self.hs["IsRaw chbx"].value
        self.xanes_config_img_scalar = self.hs["NormScale text"].value
        if self.xanes_config_is_raw:
            if self.xanes_config_use_alternative_flat:
                if self.xanes_config_alternative_flat_set:
                    self.xanes_img = -np.log((self.reader(
                        self.xanes_file_raw_h5_filename,
                        dtype="data",
                        sli=[None],
                        cfg=self.global_h.io_xanes2D_cfg,
                    ) - self.reader(
                        self.xanes_file_raw_h5_filename,
                        dtype="dark",
                        sli=[None],
                        cfg=self.global_h.io_xanes2D_cfg,
                    )) / (self.reader(
                        self.xanes_config_alternative_flat_filename,
                        dtype="flat",
                        sli=[None],
                        cfg=self.global_h.io_xanes2D_cfg,
                    ) - self.reader(
                        self.xanes_config_alternative_flat_filename,
                        dtype="dark",
                        sli=[None],
                        cfg=self.global_h.io_xanes2D_cfg,
                    )))
                    self.xanes_img[np.isinf(self.xanes_img)] = 0
                    self.xanes_img[np.isnan(self.xanes_img)] = 0
                    self.hs["EngPointsRange sldr"].value = [
                        0,
                        self.xanes_img.shape[0] - 1,
                    ]
                    self.xanes_config_raw_img_readed = True
                else:
                    self.xanes_config_raw_img_readed = False
            else:
                self.xanes_img = -np.log((self.reader(
                    self.xanes_file_raw_h5_filename,
                    dtype="data",
                    sli=[None],
                    cfg=self.global_h.io_xanes2D_cfg,
                ) - self.reader(
                    self.xanes_file_raw_h5_filename,
                    dtype="dark",
                    sli=[None],
                    cfg=self.global_h.io_xanes2D_cfg,
                )) / (self.reader(
                    self.xanes_file_raw_h5_filename,
                    dtype="flat",
                    sli=[None],
                    cfg=self.global_h.io_xanes2D_cfg,
                ) - self.reader(
                    self.xanes_file_raw_h5_filename,
                    dtype="dark",
                    sli=[None],
                    cfg=self.global_h.io_xanes2D_cfg,
                )))
                self.xanes_img[np.isinf(self.xanes_img)] = 0
                self.xanes_img[np.isnan(self.xanes_img)] = 0
                self.hs["EngPointsRange sldr"].value = [
                    0, self.xanes_img.shape[0] - 1
                ]
                self.xanes_config_raw_img_readed = True
        else:
            self.xanes_img = -np.log(
                self.reader(
                    self.xanes_file_raw_h5_filename,
                    dtype="data",
                    sli=[None],
                    cfg=self.global_h.io_xanes2D_cfg,
                ))
            self.xanes_img[np.isinf(self.xanes_img)] = 0
            self.xanes_img[np.isnan(self.xanes_img)] = 0
            self.hs["EngPointsRange sldr"].value = [
                0, self.xanes_img.shape[0] - 1
            ]
            self.xanes_config_raw_img_readed = True
        self.boxes_logic()

    def EngPointsRange_slider_chg(self, a):
        self.xanes_eng_id_s = a["owner"].value[0]
        self.xanes_eng_id_e = a["owner"].value[1]
        self.hs["EngStart text"].value = self.xanes_config_eng_list[
            self.xanes_eng_id_s]
        self.hs["EngEnd text"].value = self.xanes_config_eng_list[
            self.xanes_eng_id_e]
        self.hs["FijiEngId sldr"].value = 0
        self.hs["FijiEngId sldr"].min = 0
        self.hs[
            "FijiEngId sldr"].max = a["owner"].value[1] - a["owner"].value[0]
        self.boxes_logic()

    def FijiRawImgPrev_chbx_chg(self, a):
        if a["owner"].value:
            data_state, viewer_state = fiji_viewer_state(
                self.global_h, self, viewer_name="xanes2D_raw_img_viewer")
            if not viewer_state:
                fiji_viewer_on(self.global_h,
                               self,
                               viewer_name="xanes2D_raw_img_viewer")
        else:
            fiji_viewer_off(self.global_h,
                            self,
                            viewer_name="xanes2D_raw_img_viewer")
        self.boxes_logic()

    def FijiEngId_sldr_chg(self, a):
        data_state, viewer_state = fiji_viewer_state(
            self.global_h, self, viewer_name="xanes2D_raw_img_viewer")
        if not viewer_state:
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes2D_raw_img_viewer")
        self.hs["FijiRawImgPrev chbx"].value = True
        if not data_state:
            self.hs[
                "CfmConfigData text"].value = "xanes2D_img is not loaded yet."
        else:
            self.global_h.xanes2D_fiji_windows["xanes2D_raw_img_viewer"][
                "ip"].setSlice(a["owner"].value + self.xanes_eng_id_s)
            self.hs["EngStart text"].value = self.xanes_config_eng_list[
                a["owner"].value + self.xanes_eng_id_s]

    def FijiClose_btn_clk(self, a):
        fiji_viewer_off(self.global_h, self, viewer_name="all")
        self.boxes_logic()

    def CfmConfigData_btn_clk(self, a):
        self.xanes_config_is_raw = self.hs["IsRaw chbx"].value

        self.xanes_config_img_scalar = self.hs["NormScale text"].value
        self.xanes_eng_id_s = self.hs["EngPointsRange sldr"].value[0]
        self.xanes_eng_id_e = self.hs["EngPointsRange sldr"].value[1]

        self.hs["2DRoiX sldr"].max = self.xanes_img.shape[2] - 100
        self.hs["2DRoiX sldr"].min = 100
        self.hs["2DRoiX sldr"].value = (100, self.xanes_img.shape[2] - 100)

        self.hs["2DRoiY sldr"].max = self.xanes_img.shape[1] - 100
        self.hs["2DRoiY sldr"].min = 100
        self.hs["2DRoiY sldr"].value = (100, self.xanes_img.shape[1] - 100)

        self.update_xanes2D_config()
        json.dump(
            self.xanes_config,
            open(self.xanes_file_save_trial_reg_config_filename, "w"),
            cls=NumpyArrayEncoder,
        )
        self.xanes_data_configured = True

        self.xanes_fit_eng_list = self.reader(
            self.xanes_file_raw_h5_filename,
            dtype="eng",
            sli=[[self.xanes_eng_id_s, self.xanes_eng_id_e + 1]],
            cfg=self.global_h.io_xanes2D_cfg,
        )
        if self.xanes_fit_eng_list.max() < 70:
            self.xanes_fit_eng_list *= 1000
        self.boxes_logic()

    def Roi2DX_sldr_chg(self, a):
        self.xanes_roi_configured = False
        if a["owner"].value[0] < 20:
            a["owner"].value[0] = 20
        if a["owner"].value[1] > (self.xanes_img.shape[2] - 20):
            a["owner"].value[1] = self.xanes_img.shape[2] - 20
        self.xanes_reg_roi[2] = a["owner"].value[0]
        self.xanes_reg_roi[3] = a["owner"].value[1]
        self.hs["CfmRoi text"].value = "Please confirm after ROI is set"

        data_state, viewer_state = fiji_viewer_state(
            self.global_h, self, viewer_name="xanes2D_raw_img_viewer")
        if (not data_state) | (not viewer_state):
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes2D_raw_img_viewer")
            self.hs["FijiRawImgPrev chbx"].value = True
        self.global_h.xanes2D_fiji_windows["xanes2D_raw_img_viewer"][
            "ip"].setRoi(
                self.hs["2DRoiX sldr"].value[0],
                self.hs["2DRoiY sldr"].value[0],
                self.hs["2DRoiX sldr"].value[1] -
                self.hs["2DRoiX sldr"].value[0],
                self.hs["2DRoiY sldr"].value[1] -
                self.hs["2DRoiY sldr"].value[0],
            )

    def Roi2DY_sldr_chg(self, a):
        self.xanes_roi_configured = False
        if a["owner"].value[0] < 20:
            a["owner"].value[0] = 20
        if a["owner"].value[1] > (self.xanes_img.shape[1] - 20):
            a["owner"].value[1] = self.xanes_img.shape[1] - 20
        self.xanes_reg_roi[0] = a["owner"].value[0]
        self.xanes_reg_roi[1] = a["owner"].value[1]
        self.hs["CfmRoi text"].value = "Please confirm after ROI is set"

        data_state, viewer_state = fiji_viewer_state(
            self.global_h, self, viewer_name="xanes2D_raw_img_viewer")
        if (not data_state) | (not viewer_state):
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes2D_raw_img_viewer")
            self.hs["FijiRawImgPrev chbx"].value = True
        self.global_h.xanes2D_fiji_windows["xanes2D_raw_img_viewer"][
            "ip"].setRoi(
                self.hs["2DRoiX sldr"].value[0],
                self.hs["2DRoiY sldr"].value[0],
                self.hs["2DRoiX sldr"].value[1] -
                self.hs["2DRoiX sldr"].value[0],
                self.hs["2DRoiY sldr"].value[1] -
                self.hs["2DRoiY sldr"].value[0],
            )

    def CfmRoi_btn_clk(self, a):
        data_state, viewer_state = fiji_viewer_state(
            self.global_h, self, viewer_name="xanes2D_raw_img_viewer")
        if viewer_state:
            fiji_viewer_off(self.global_h,
                            self,
                            viewer_name="xanes2D_raw_img_viewer")
        self.hs["FijiRawImgPrev chbx"].value = False

        if self.xanes_roi_configured:
            pass
        else:
            self.xanes_reg_roi[0] = self.hs["2DRoiY sldr"].value[0]
            self.xanes_reg_roi[1] = self.hs["2DRoiY sldr"].value[1]
            self.xanes_reg_roi[2] = self.hs["2DRoiX sldr"].value[0]
            self.xanes_reg_roi[3] = self.hs["2DRoiX sldr"].value[1]
            self.hs[
                "AnchorId sldr"].max = self.xanes_eng_id_e - self.xanes_eng_id_s
            self.hs["AnchorId sldr"].value = max(
                int((self.xanes_eng_id_e - self.xanes_eng_id_s) / 2), 1)
            self.hs["AnchorId sldr"].min = 1
            self.xanes_reg_anchor_idx = self.xanes_eng_id_s
            self.hs["CfmRoi text"].value = "ROI is set"
            del self.xanes_img_roi
            gc.collect()
            self.xanes_img_roi = self.xanes_img[
                self.xanes_eng_id_s:self.xanes_eng_id_e + 1,
                self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ]
            self.xanes_reg_mask = (
                self.xanes_img[0, self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                               self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ] >
                0).astype(np.int8)
            self.update_xanes2D_config()
            json.dump(
                self.xanes_config,
                open(self.xanes_file_save_trial_reg_config_filename, "w"),
                cls=NumpyArrayEncoder,
            )
            self.xanes_roi_configured = True
        self.boxes_logic()

    def FijiMaskViewer_chbx_chg(self, a):
        if a["owner"].value:
            data_state, viewer_state = fiji_viewer_state(
                self.global_h, self, viewer_name="xanes2D_mask_viewer")
            if not viewer_state:
                fiji_viewer_on(self.global_h,
                               self,
                               viewer_name="xanes2D_mask_viewer")
        else:
            fiji_viewer_off(self.global_h,
                            self,
                            viewer_name="xanes2D_mask_viewer")
        self.boxes_logic()

    def UseChunk_chbx_chg(self, a):
        self.xanes_reg_use_chunk = a["owner"].value
        self.xanes_reg_params_configured = False
        self.boxes_logic()

    def AnchorId_sldr_chg(self, a):
        self.xanes_reg_anchor_idx = a["owner"].value + self.xanes_eng_id_s
        if not self.hs["FijiMaskViewer chbx"].value:
            self.hs["FijiMaskViewer chbx"].value = True
        else:
            data_state, viewer_state = fiji_viewer_state(
                self.global_h, self, viewer_name="xanes2D_mask_viewer")
            if not viewer_state:
                fiji_viewer_on(self.global_h,
                               self,
                               viewer_name="xanes2D_mask_viewer")
        self.global_h.xanes2D_fiji_windows["xanes2D_mask_viewer"][
            "ip"].setSlice(self.xanes_reg_anchor_idx - self.xanes_eng_id_s)
        self.xanes_reg_params_configured = False
        self.xanes_regparams_anchor_idx_set = True

    def UseMask_chbx_chg(self, a):
        self.xanes_reg_use_mask = a["owner"].value
        if self.xanes_reg_use_mask:
            self.xanes_reg_mask = (
                self.xanes_img[self.xanes_reg_anchor_idx,
                               self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                               self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ] >
                self.xanes_reg_use_mask).astype(np.int8)
        self.xanes_reg_params_configured = False
        self.boxes_logic()

    def MaskThres_sldr_chg(self, a):
        self.xanes_reg_mask_thres = a["owner"].value
        self.hs["AnchorId sldr"].disabled = True
        if self.xanes_reg_mask is None:
            self.xanes_reg_mask = (
                self.xanes_img[self.xanes_reg_anchor_idx,
                               self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                               self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ] >
                self.xanes_reg_use_mask).astype(np.int8)
        if not self.hs["FijiMaskViewer chbx"].value:
            self.hs["FijiMaskViewer chbx"].value = True
        else:
            data_state, viewer_state = fiji_viewer_state(
                self.global_h, self, viewer_name="xanes2D_mask_viewer")
            if not viewer_state:
                fiji_viewer_on(self.global_h,
                               self,
                               viewer_name="xanes2D_mask_viewer")
        if self.xanes_reg_use_smooth_img:
            if self.xanes_reg_mask_dilation_width > 0:
                self.xanes_reg_mask[:] = skm.binary_dilation(
                    ((gaussian_filter(
                        self.xanes_img[
                            self.xanes_reg_anchor_idx,
                            self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                            self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ],
                        self.xanes_reg_smooth_img_sigma,
                    ) > self.xanes_reg_mask_thres).astype(np.int8)),
                    np.ones([
                        self.xanes_reg_mask_dilation_width,
                        self.xanes_reg_mask_dilation_width,
                    ]),
                )[:]
            else:
                self.xanes_reg_mask[:] = ((gaussian_filter(
                    self.xanes_img[
                        self.xanes_reg_anchor_idx,
                        self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                        self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ],
                    self.xanes_reg_smooth_img_sigma,
                ) > self.xanes_reg_mask_thres).astype(np.int8))[:]
            self.global_h.xanes2D_fiji_windows["xanes2D_mask_viewer"][
                "ip"].setImage(self.global_h.ij.convert().convert(
                    self.global_h.ij.dataset().create(
                        self.global_h.ij.py.to_java(
                            gaussian_filter(
                                self.xanes_img[:, self.xanes_reg_roi[0]:self.
                                               xanes_reg_roi[1],
                                               self.xanes_reg_roi[2]:self.
                                               xanes_reg_roi[3], ],
                                self.xanes_reg_smooth_img_sigma,
                            ) * self.xanes_reg_mask)),
                    self.global_h.ImagePlusClass,
                ))
        else:
            if self.xanes_reg_mask_dilation_width > 0:
                self.xanes_reg_mask[:] = skm.binary_dilation(
                    ((self.xanes_img[
                        self.xanes_reg_anchor_idx,
                        self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                        self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ] >
                      self.xanes_reg_mask_thres).astype(np.int8)),
                    np.ones([
                        self.xanes_reg_mask_dilation_width,
                        self.xanes_reg_mask_dilation_width,
                    ]),
                )[:]
            else:
                self.xanes_reg_mask[:] = ((self.xanes_img[
                    self.xanes_reg_anchor_idx,
                    self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                    self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ] >
                                           self.xanes_reg_mask_thres).astype(
                                               np.int8))[:]
            self.global_h.xanes2D_fiji_windows["xanes2D_mask_viewer"][
                "ip"].setImage(self.global_h.ij.convert().convert(
                    self.global_h.ij.dataset().create(
                        self.global_h.ij.py.to_java(
                            self.xanes_img[:, self.xanes_reg_roi[0]:self.
                                           xanes_reg_roi[1],
                                           self.xanes_reg_roi[2]:self.
                                           xanes_reg_roi[3], ] *
                            self.xanes_reg_mask)),
                    self.global_h.ImagePlusClass,
                ))
        self.global_h.xanes2D_fiji_windows["xanes2D_mask_viewer"][
            "ip"].setSlice(self.xanes_reg_anchor_idx)
        self.xanes_reg_params_configured = False

    def MaskDilation_sldr_chg(self, a):
        self.xanes_reg_mask_dilation_width = a["owner"].value
        self.hs["AnchorId sldr"].disabled = True
        if not self.hs["FijiMaskViewer chbx"].value:
            self.hs["FijiMaskViewer chbx"].value = True
        else:
            data_state, viewer_state = fiji_viewer_state(
                self.global_h, self, viewer_name="xanes2D_mask_viewer")
            if not viewer_state:
                fiji_viewer_on(self.global_h,
                               self,
                               viewer_name="xanes2D_mask_viewer")
        if self.xanes_reg_use_smooth_img:
            if self.xanes_reg_mask_dilation_width > 0:
                self.xanes_reg_mask[:] = skm.binary_dilation(
                    ((gaussian_filter(
                        self.xanes_img[
                            self.xanes_reg_anchor_idx,
                            self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                            self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ],
                        self.xanes_reg_smooth_img_sigma,
                    ) > self.xanes_reg_mask_thres).astype(np.int8)),
                    np.ones([
                        self.xanes_reg_mask_dilation_width,
                        self.xanes_reg_mask_dilation_width,
                    ]),
                )[:]
            else:
                self.xanes_reg_mask[:] = ((gaussian_filter(
                    self.xanes_img[
                        self.xanes_reg_anchor_idx,
                        self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                        self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ],
                    self.xanes_reg_smooth_img_sigma,
                ) > self.xanes_reg_mask_thres).astype(np.int8))[:]
            self.global_h.xanes2D_fiji_windows["xanes2D_mask_viewer"][
                "ip"].setImage(self.global_h.ij.convert().convert(
                    self.global_h.ij.dataset().create(
                        self.global_h.ij.py.to_java(
                            gaussian_filter(
                                self.xanes_img[:, self.xanes_reg_roi[0]:self.
                                               xanes_reg_roi[1],
                                               self.xanes_reg_roi[2]:self.
                                               xanes_reg_roi[3], ],
                                self.xanes_reg_smooth_img_sigma,
                            ) * self.xanes_reg_mask)),
                    self.global_h.ImagePlusClass,
                ))
        else:
            if self.xanes_reg_mask_dilation_width > 0:
                self.xanes_reg_mask[:] = skm.binary_dilation(
                    ((self.xanes_img[
                        self.xanes_reg_anchor_idx,
                        self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                        self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ] >
                      self.xanes_reg_mask_thres).astype(np.int8)),
                    np.ones([
                        self.xanes_reg_mask_dilation_width,
                        self.xanes_reg_mask_dilation_width,
                    ]),
                )[:]
            else:
                self.xanes_reg_mask[:] = ((self.xanes_img[
                    self.xanes_reg_anchor_idx,
                    self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                    self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ] >
                                           self.xanes_reg_mask_thres).astype(
                                               np.int8))[:]
            self.global_h.xanes2D_fiji_windows["xanes2D_mask_viewer"][
                "ip"].setImage(self.global_h.ij.convert().convert(
                    self.global_h.ij.dataset().create(
                        self.global_h.ij.py.to_java(
                            self.xanes_img[:, self.xanes_reg_roi[0]:self.
                                           xanes_reg_roi[1],
                                           self.xanes_reg_roi[2]:self.
                                           xanes_reg_roi[3], ] *
                            self.xanes_reg_mask)),
                    self.global_h.ImagePlusClass,
                ))
        self.global_h.xanes2D_fiji_windows["xanes2D_mask_viewer"][
            "ip"].setSlice(self.xanes_reg_anchor_idx)
        self.xanes_reg_params_configured = False

    def UseSmooth_chbx_chg(self, a):
        self.xanes_reg_use_smooth_img = a["owner"].value
        self.xanes_reg_params_configured = False
        self.boxes_logic()

    def SmoothSigma_text_chg(self, a):
        self.xanes_reg_smooth_img_sigma = a["owner"].value
        if self.xanes_reg_use_mask:
            self.xanes_reg_mask = (
                self.xanes_img[self.xanes_reg_anchor_idx,
                               self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                               self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ] >
                self.xanes_reg_use_mask).astype(np.int8)
        if not self.hs["FijiMaskViewer chbx"].value:
            self.hs["FijiMaskViewer chbx"].value = True
        else:
            data_state, viewer_state = fiji_viewer_state(
                self.global_h, self, viewer_name="xanes2D_mask_viewer")
            if not viewer_state:
                fiji_viewer_on(self.global_h,
                               self,
                               viewer_name="xanes2D_mask_viewer")
        if self.xanes_reg_mask_dilation_width > 0:
            self.xanes_reg_mask[:] = skm.binary_dilation(
                ((gaussian_filter(
                    self.xanes_img[
                        self.xanes_reg_anchor_idx,
                        self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                        self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ],
                    self.xanes_reg_smooth_img_sigma,
                ) > self.xanes_reg_mask_thres).astype(np.int8)),
                np.ones([
                    self.xanes_reg_mask_dilation_width,
                    self.xanes_reg_mask_dilation_width,
                ]),
            )[:]
        else:
            self.xanes_reg_mask[:] = ((gaussian_filter(
                self.xanes_img[self.xanes_reg_anchor_idx,
                               self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                               self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ],
                self.xanes_reg_smooth_img_sigma,
            ) > self.xanes_reg_mask_thres).astype(np.int8))[:]
        self.global_h.xanes2D_fiji_windows["xanes2D_mask_viewer"][
            "ip"].setImage(self.global_h.ij.convert().convert(
                self.global_h.ij.dataset().create(
                    self.global_h.ij.py.to_java(
                        gaussian_filter(
                            self.xanes_img[:, self.xanes_reg_roi[0]:self.
                                           xanes_reg_roi[1],
                                           self.xanes_reg_roi[2]:self.
                                           xanes_reg_roi[3], ],
                            self.xanes_reg_smooth_img_sigma,
                        ) * self.xanes_reg_mask)),
                self.global_h.ImagePlusClass,
            ))
        self.global_h.xanes2D_fiji_windows["xanes2D_mask_viewer"][
            "ip"].setSlice(self.xanes_reg_anchor_idx)
        self.xanes_reg_params_configured = False

    def ChunkSz_sldr_chg(self, a):
        self.xanes_reg_chunk_sz = a["owner"].value
        self.xanes_reg_params_configured = False

    def RegMethod_drpdn_chg(self, a):
        self.xanes_reg_method = a["owner"].value
        self.xanes_reg_params_configured = False
        self.boxes_logic()

    def RefMode_drpdn_chg(self, a):
        self.xanes_reg_ref_mode = a["owner"].value
        self.xanes_reg_params_configured = False
        self.boxes_logic()

    def MrtvLevel_text_chg(self, a):
        self.xanes_reg_mrtv_level = a["owner"].value
        self.xanes_reg_params_configured = False
        self.boxes_logic()

    def MrtvWz_text_chg(self, a):
        self.xanes_reg_mrtv_width = a["owner"].value
        self.xanes_reg_params_configured = False
        self.boxes_logic()

    def MrtvSubpixelWz_text_chg(self, a):
        self.xanes_reg_mrtv_subpixel_wz = a["owner"].value
        self.xanes_reg_params_configured = False
        self.boxes_logic()

    def MrtvSubpixelKernel_text_chg(self, a):
        self.xanes_reg_mrtv_subpixel_kernel = a["owner"].value
        self.xanes_reg_params_configured = False
        self.boxes_logic()

    def MrtvSubpixelSrch_drpdn_chg(self, a):
        if a["owner"].value == "analytical":
            self.hs["MrtvSubpixelWz text"].value = 3
        else:
            self.hs["MrtvSubpixelWz text"].value = 5
        self.xanes_reg_params_configured = False
        self.boxes_logic()

    def CfmRegParams_btn_clk(self, a):
        if self.xanes_reg_params_configured:
            data_state, viewer_state = fiji_viewer_state(
                self.global_h, self, viewer_name="xanes2D_mask_viewer")
            if viewer_state:
                fiji_viewer_off(self.global_h,
                                self,
                                viewer_name="xanes2D_mask_viewer")
        self.hs["FijiMaskViewer chbx"].value = False

        self.xanes_reg_use_chunk = self.hs["UseChunk chbx"].value
        self.xanes_reg_anchor_idx = self.hs[
            "AnchorId sldr"].value + self.xanes_eng_id_s
        self.xanes_reg_use_mask = self.hs["UseMask chbx"].value
        self.xanes_reg_mask_thres = self.hs["MaskThres sldr"].value
        self.xanes_reg_mask_dilation_width = self.hs["MaskDilation sldr"].value
        self.xanes_reg_use_smooth_img = self.hs["UseSmooth chbx"].value
        self.xanes_reg_smooth_img_sigma = self.hs["SmoothSigma text"].value
        self.xanes_reg_chunk_sz = self.hs["ChunkSz sldr"].value
        self.xanes_reg_method = self.hs["RegMethod drpdn"].value
        self.xanes_reg_ref_mode = self.hs["RefMode drpdn"].value
        self.xanes_reg_mrtv_level = self.hs["MrtvLevel text"].value
        self.xanes_reg_mrtv_width = self.hs["MrtvWz text"].value
        self.xanes_reg_mrtv_subpixel_kernel = self.hs[
            "MrtvSubpixelKernel text"].value
        self.xanes_reg_mrtv_subpixel_wz = self.hs["MrtvSubpixelWz text"].value
        if self.hs["MrtvSubpixelSrch drpdn"].value == "analytical":
            self.xanes_reg_mrtv_subpixel_srch_option = "ana"
        else:
            self.xanes_reg_mrtv_subpixel_srch_option = "fit"

        if self.xanes_reg_use_mask:
            if self.xanes_reg_use_smooth_img:
                self.xanes_img_roi[:] = gaussian_filter(
                    self.xanes_img[
                        self.xanes_eng_id_s:self.xanes_eng_id_e + 1,
                        self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                        self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ],
                    self.xanes_reg_smooth_img_sigma,
                )[:]
                if self.xanes_reg_mask_dilation_width > 0:
                    self.xanes_reg_mask[:] = skm.binary_dilation(
                        ((gaussian_filter(
                            self.xanes_img[
                                self.xanes_reg_anchor_idx,
                                self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                                self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ],
                            self.xanes_reg_smooth_img_sigma,
                        ) > self.xanes_reg_mask_thres).astype(np.int8)),
                        np.ones([
                            self.xanes_reg_mask_dilation_width,
                            self.xanes_reg_mask_dilation_width,
                        ]),
                    )[:]
                else:
                    self.xanes_reg_mask[:] = ((gaussian_filter(
                        self.xanes_img[
                            self.xanes_reg_anchor_idx,
                            self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                            self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ],
                        self.xanes_reg_smooth_img_sigma,
                    ) > self.xanes_reg_mask_thres).astype(np.int8))[:]
            else:
                self.xanes_img_roi[:] = self.xanes_img[
                    self.xanes_eng_id_s:self.xanes_eng_id_e + 1,
                    self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                    self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ]
                if self.xanes_reg_mask_dilation_width > 0:
                    self.xanes_reg_mask[:] = skm.binary_dilation(
                        ((self.xanes_img[
                            self.xanes_reg_anchor_idx,
                            self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                            self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ] >
                          self.xanes_reg_mask_thres).astype(np.int8)),
                        np.ones([
                            self.xanes_reg_mask_dilation_width,
                            self.xanes_reg_mask_dilation_width,
                        ]),
                    )[:]
                else:
                    self.xanes_reg_mask[:] = (
                        (self.xanes_img[
                            self.xanes_reg_anchor_idx,
                            self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                            self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ] >
                         self.xanes_reg_mask_thres).astype(np.int8))[:]
        else:
            if self.xanes_reg_use_smooth_img:
                self.xanes_img_roi[:] = gaussian_filter(
                    self.xanes_img[
                        self.xanes_eng_id_s:self.xanes_eng_id_e + 1,
                        self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                        self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ],
                    self.xanes_reg_smooth_img_sigma,
                )[:]
            else:
                self.xanes_img_roi[:] = self.xanes_img[
                    self.xanes_eng_id_s:self.xanes_eng_id_e + 1,
                    self.xanes_reg_roi[0]:self.xanes_reg_roi[1],
                    self.xanes_reg_roi[2]:self.xanes_reg_roi[3], ]
        self.update_xanes2D_config()
        json.dump(
            self.xanes_config,
            open(self.xanes_file_save_trial_reg_config_filename, "w"),
            cls=NumpyArrayEncoder,
        )
        self.xanes_reg_params_configured = True
        self.boxes_logic()

    def RunReg_btn_clk(self, a):
        tmp_file = os.path.join(self.global_h.tmp_dir, "xanes2D_tmp.h5")
        with h5py.File(tmp_file, "w") as f:
            f.create_dataset("analysis_eng_list",
                             data=self.xanes_fit_eng_list.astype(np.float32))
            f.create_dataset("xanes2D_img",
                             data=self.xanes_img.astype(np.float32))
            if self.xanes_reg_mask is not None:
                f.create_dataset("xanes2D_reg_mask",
                                 data=self.xanes_reg_mask.astype(np.float32))
            else:
                f.create_dataset("xanes2D_reg_mask", data=np.array([0]))
        code = {}
        ln = 0

        code[ln] = f"import os"
        ln += 1
        code[ln] = f"from TXM_Sandbox.utils import xanes_regtools as xr"
        ln += 1
        code[ln] = f"from multiprocessing import freeze_support"
        ln += 1
        code[ln] = f"if __name__ == '__main__':"
        ln += 1
        code[ln] = f"    freeze_support()"
        ln += 1
        code[
            ln] = f"    reg = xr.regtools(dtype='2D_XANES', method='{self.xanes_reg_method}', mode='TRANSLATION')"
        ln += 1
        code[
            ln] = f"    reg.set_xanes2D_raw_filename('{self.xanes_file_raw_h5_filename}')"
        ln += 1
        kwargs = {
            "raw_h5_filename": self.xanes_file_raw_h5_filename,
            "config_filename": self.xanes_file_save_trial_reg_config_filename,
        }
        code[ln] = f"    reg.set_raw_data_info(**{kwargs})"
        ln += 1
        code[ln] = f"    reg.set_method('{self.xanes_reg_method}')"
        ln += 1
        code[ln] = f"    reg.set_ref_mode('{self.xanes_reg_ref_mode}')"
        ln += 1
        code[ln] = f"    reg.set_roi({self.xanes_reg_roi})"
        ln += 1
        code[
            ln] = f"    reg.set_indices({self.xanes_eng_id_s}, {self.xanes_eng_id_e + 1}, {self.xanes_reg_anchor_idx})"
        ln += 1
        code[ln] = f"    reg.set_xanes2D_tmp_filename('{tmp_file}')"
        ln += 1
        code[ln] = f"    reg.read_xanes2D_tmp_file(mode='reg')"
        ln += 1
        code[
            ln] = f"    reg.set_reg_options(use_mask={self.xanes_reg_use_mask}, mask_thres={self.xanes_reg_mask_thres},\
                     use_chunk={self.xanes_reg_use_chunk}, chunk_sz={self.xanes_reg_chunk_sz},\
                     use_smooth_img={self.xanes_reg_use_smooth_img}, smooth_sigma={self.xanes_reg_smooth_img_sigma},\
                     mrtv_level={self.xanes_reg_mrtv_level}, mrtv_width={self.xanes_reg_mrtv_width}, \
                     mrtv_sp_wz={self.xanes_reg_mrtv_subpixel_wz}, mrtv_sp_kernel={self.xanes_reg_mrtv_subpixel_kernel})"

        ln += 1
        code[
            ln] = f"    reg.set_saving(os.path.dirname('{self.xanes_save_trial_reg_filename}'), \
                     fn=os.path.basename('{self.xanes_save_trial_reg_filename}'))"

        ln += 1
        code[ln] = f"    reg.compose_dicts()"
        ln += 1
        code[ln] = f"    reg.reg_xanes2D_chunk()"
        ln += 1

        gen_external_py_script(self.xanes_reg_external_command_name, code)
        sig = os.system(f"python {self.xanes_reg_external_command_name}")

        print(sig)
        if sig == 0:
            self.hs["RunReg text"].value = "XANES2D registration is done"
            self.xanes_review_aligned_img_original = np.ndarray(
                self.xanes_img_roi[0].shape)
            self.xanes_review_aligned_img = np.ndarray(
                self.xanes_img_roi[0].shape)
            self.xanes_review_fixed_img = np.ndarray(
                self.xanes_img_roi[0].shape)
            self.xanes_review_shift_dict = {}
            with h5py.File(self.xanes_save_trial_reg_filename, "r") as f:
                self.xanes_alignment_pairs = f[
                    "/trial_registration/trial_reg_parameters/alignment_pairs"][:]
                for ii in range(self.xanes_alignment_pairs.shape[0]):
                    self.xanes_review_shift_dict["{}".format(ii)] = f[
                        "/trial_registration/trial_reg_results/{0}/shift{0}".
                        format(str(ii).zfill(3))][:]
                    print(ii)
            self.hs["RegPair sldr"].min = 0
            self.hs[
                "RegPair sldr"].max = self.xanes_alignment_pairs.shape[0] - 1

            self.xanes_reg_done = True
            self.xanes_review_reg_best_match_filename = (os.path.splitext(
                self.xanes_file_save_trial_reg_config_filename)[0].replace(
                    "config", "reg_best_match") + ".json")
            self.update_xanes2D_config()
            json.dump(
                self.xanes_config,
                open(self.xanes_file_save_trial_reg_config_filename, "w"),
                cls=NumpyArrayEncoder,
            )
            self.hs["RegPair sldr"].value = 0
        else:
            self.hs[
                "RunReg text"].value = "Something went wrong during XANES2D registration"
            self.xanes_reg_done = False
        self.boxes_logic()

    def ReadAlign_chbx_chg(self, a):
        self.xanes_use_existing_reg_reviewed = a["owner"].value
        self.xanes_reg_review_done = False
        self.boxes_logic()

    def ReadAlign_btn_clk(self, a):
        if len(a.files[0]) != 0:
            try:
                self.xanes_reg_review_file = os.path.abspath(a.files[0])
                if os.path.splitext(self.xanes_reg_review_file)[1] == ".json":
                    self.xanes_review_shift_dict = json.load(
                        open(self.xanes_reg_review_file, "r"))
                else:
                    self.xanes_review_shift_dict = np.float32(
                        np.genfromtxt(self.xanes_reg_review_file))
                for ii in self.xanes_review_shift_dict:
                    self.xanes_review_shift_dict[ii] = np.float32(
                        np.array(self.xanes_review_shift_dict[ii]))
                self.xanes_reg_file_readed = True
            except:
                self.xanes_reg_file_readed = False
        else:
            self.xanes_reg_file_readed = False
        self.xanes_reg_review_done = False
        self.boxes_logic()

    def RegPair_sldr_chg(self, a):
        self.xanes_review_alignment_pair_id = a["owner"].value
        with h5py.File(self.xanes_save_trial_reg_filename, "r") as f:
            self.xanes_review_aligned_img_original[:] = f[
                "/trial_registration/trial_reg_results/{0}/trial_reg_img{0}".
                format(str(self.xanes_review_alignment_pair_id).zfill(3))][:]
            self.xanes_review_fixed_img[:] = f[
                "/trial_registration/trial_reg_results/{0}/trial_reg_fixed{0}".
                format(str(self.xanes_review_alignment_pair_id).zfill(3))][:]
        data_state, viewer_state = fiji_viewer_state(
            self.global_h, self, viewer_name="xanes2D_review_viewer")
        if not viewer_state:
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes2D_review_viewer")

        self.global_h.xanes2D_fiji_windows["xanes2D_review_viewer"][
            "ip"].setImage(self.global_h.ij.convert().convert(
                self.global_h.ij.dataset().create(
                    self.global_h.ij.py.to_java(
                        self.xanes_review_aligned_img_original -
                        self.xanes_review_fixed_img)),
                self.global_h.ImagePlusClass,
            ))
        self.global_h.ij.py.run_macro(
            """run("Enhance Contrast", "saturated=0.35")""")
        self.xanes_review_bad_shift = False

    def RegPairBad_btn_clk(self, a):
        self.xanes_manual_xshift = 0
        self.xanes_manual_yshift = 0
        self.xanes_review_bad_shift = True
        self.boxes_logic()

    def XShift_text_chg(self, a):
        self.xanes_manual_xshift = a["owner"].value
        self.xanes_review_aligned_img[:] = np.real(
            np.fft.ifftn(
                fourier_shift(
                    np.fft.fftn(self.xanes_review_aligned_img_original),
                    [self.xanes_manual_yshift, self.xanes_manual_xshift],
                )))[:]
        data_state, viewer_state = fiji_viewer_state(
            self.global_h, self, viewer_name="xanes2D_review_viewer")
        if not viewer_state:
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes2D_review_viewer")

        self.global_h.xanes2D_fiji_windows["xanes2D_review_viewer"][
            "ip"].setImage(self.global_h.ij.convert().convert(
                self.global_h.ij.dataset().create(
                    self.global_h.ij.py.to_java(self.xanes_review_aligned_img -
                                                self.xanes_review_fixed_img)),
                self.global_h.ImagePlusClass,
            ))
        self.global_h.ij.py.run_macro(
            """run("Enhance Contrast", "saturated=0.35")""")

    def YShift_text_chg(self, a):
        self.xanes_manual_yshift = a["owner"].value
        self.xanes_review_aligned_img[:] = np.real(
            np.fft.ifftn(
                fourier_shift(
                    np.fft.fftn(self.xanes_review_aligned_img_original),
                    [self.xanes_manual_yshift, self.xanes_manual_xshift],
                )))[:]
        data_state, viewer_state = fiji_viewer_state(
            self.global_h, self, viewer_name="xanes2D_review_viewer")
        if not viewer_state:
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes2D_review_viewer")

        self.global_h.xanes2D_fiji_windows["xanes2D_review_viewer"][
            "ip"].setImage(self.global_h.ij.convert().convert(
                self.global_h.ij.dataset().create(
                    self.global_h.ij.py.to_java(self.xanes_review_aligned_img -
                                                self.xanes_review_fixed_img)),
                self.global_h.ImagePlusClass,
            ))
        self.global_h.ij.py.run_macro(
            """run("Enhance Contrast", "saturated=0.35")""")

    def Record_btn_clk(self, a):
        with h5py.File(self.xanes_save_trial_reg_filename, "r") as f:
            shift = f[
                "/trial_registration/trial_reg_results/{0}/shift{0}".format(
                    str(self.xanes_review_alignment_pair_id).zfill(3))][:]
        if self.xanes_reg_method.upper() == "SR":
            shift[0, 2] += self.xanes_manual_yshift
            shift[1, 2] += self.xanes_manual_xshift
            self.xanes_review_shift_dict["{}".format(
                self.xanes_review_alignment_pair_id)] = np.array(shift)
        else:
            self.xanes_review_shift_dict["{}".format(
                self.xanes_review_alignment_pair_id)] = np.array([
                    shift[0] + self.xanes_manual_yshift,
                    shift[1] + self.xanes_manual_xshift,
                ])
        self.hs["XShift text"].value = 0
        self.hs["YShift text"].value = 0
        json.dump(
            self.xanes_review_shift_dict,
            open(self.xanes_review_reg_best_match_filename, "w"),
            cls=NumpyArrayEncoder,
        )
        self.xanes_review_bad_shift = False
        self.boxes_logic()

    def CfmRevRlt_btn_clk(self, a):
        if len(self.xanes_review_shift_dict) != (self.hs["RegPair sldr"].max +
                                                 1):
            self.hs[
                "CfmRevRlt text"].value = "reg review is not completed yet ..."
            idx = []
            offset = []
            for ii in sorted(self.xanes_review_shift_dict.keys()):
                offset.append(self.xanes_review_shift_dict[ii][0])
                idx.append(int(ii))
            plt.figure(1)
            plt.plot(idx, offset, "b+")
            plt.xticks(np.arange(0, len(idx) + 1, 5))
            plt.grid()
            self.xanes_reg_review_done = False
        else:
            data_state, viewer_state = fiji_viewer_state(
                self.global_h, self, viewer_name="xanes2D_review_viewer")
            if viewer_state:
                fiji_viewer_off(self.global_h,
                                self,
                                viewer_name="xanes2D_review_viewer")
            self.hs[
                "CfmRevRlt text"].value = "reg review is not completed yet ..."
            self.xanes_reg_review_done = True
            self.update_xanes2D_config()
            json.dump(
                self.xanes_config,
                open(self.xanes_file_save_trial_reg_config_filename, "w"),
                cls=NumpyArrayEncoder,
            )
            json.dump(
                self.xanes_review_shift_dict,
                open(self.xanes_review_reg_best_match_filename, "w"),
                cls=NumpyArrayEncoder,
            )
        self.boxes_logic()

    def AlignImg_btn_clk(self, a):
        tmp_file = os.path.join(self.global_h.tmp_dir, "xanes2D_tmp.h5")
        tmp_dict = {}
        for key in self.xanes_review_shift_dict.keys():
            tmp_dict[key] = tuple(self.xanes_review_shift_dict[key])
        code = {}
        ln = 0
        code[ln] = f"import TXM_Sandbox.utils.xanes_regtools as xr"
        ln += 1
        code[
            ln] = f"reg = xr.regtools(dtype='2D_XANES', method='{self.xanes_reg_method}', mode='TRANSLATION')"
        ln += 1
        code[ln] = f"reg.set_roi({self.xanes_reg_roi})"
        ln += 1
        code[
            ln] = f"reg.set_indices({self.xanes_eng_id_s}, {self.xanes_eng_id_e + 1}, {self.xanes_reg_anchor_idx})"
        ln += 1
        code[ln] = f"reg.set_xanes2D_tmp_filename('{tmp_file}')"
        ln += 1
        code[ln] = f"reg.read_xanes2D_tmp_file(mode='align')"
        ln += 1
        code[ln] = f"reg.apply_xanes2D_chunk_shift({tmp_dict}, \
                     trialfn='{self.xanes_save_trial_reg_filename}', \
                     savefn='{self.xanes_save_trial_reg_filename}')"

        ln += 1

        gen_external_py_script(self.xanes_align_external_command_name, code)
        sig = os.system(f"python '{self.xanes_align_external_command_name}'")
        if sig == 0:
            self.hs["AlignImg text"].value = "XANES2D alignment is done ..."
        else:
            self.hs[
                "AlignImg text"].value = "Something wrong during XANES2D alignment"

        data_state, viewer_state = fiji_viewer_state(
            self.global_h, self, viewer_name="xanes2D_analysis_viewer")
        if not viewer_state:
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes2D_analysis_viewer")
        self.global_h.xanes2D_fiji_windows["xanes2D_analysis_viewer"][
            "ip"].setImage(self.global_h.ij.convert().convert(
                self.global_h.ij.dataset().create(
                    self.global_h.ij.py.to_java(self.xanes_img_roi)),
                self.global_h.ImagePlusClass,
            ))
        self.global_h.xanes2D_fiji_windows["xanes2D_analysis_viewer"][
            "ip"].setSlice(0)
        self.global_h.ij.py.run_macro(
            """run("Enhance Contrast", "saturated=0.35")""")

        with h5py.File(self.xanes_save_trial_reg_filename, "r") as f:
            self.xanes_fit_data_shape = f[
                "/registration_results/reg_results/registered_xanes2D"].shape
            self.xanes_fit_eng_list = f[
                "/registration_results/reg_results/eng_list"][:]
            if self.xanes_img_roi is None:
                self.xanes_img_roi = f[
                    "/registration_results/reg_results/registered_xanes2D"][:]
            else:
                self.xanes_img_roi = f[
                    "/registration_results/reg_results/registered_xanes2D"][:]

        self.hs["Vis sldr"].min = 1
        self.hs["Vis sldr"].max = self.xanes_fit_eng_list.shape[0]
        self.hs["Vis sldr"].value = 2

        self.xanes_element = determine_element(self.xanes_fit_eng_list)
        tem = determine_fitting_energy_range(self.xanes_element)
        self.xanes_fit_edge_eng = tem[0]
        self.xanes_fit_wl_fit_eng_s = tem[1]
        self.xanes_fit_wl_fit_eng_e = tem[2]
        self.xanes_fit_pre_edge_e = tem[3]
        self.xanes_fit_post_edge_s = tem[4]
        self.xanes_fit_edge_0p5_fit_s = tem[5]
        self.xanes_fit_edge_0p5_fit_e = tem[6]
        self.xanes_fit_gui_h.hs["FitEngRagOptn drpdn"].value = "wl"
        self.xanes_fit_gui_h.hs["FitEngRagOptn drpdn"].value = "full"
        if (self.xanes_fit_eng_list.min() >
            (self.xanes_fit_edge_eng - 50)) and (
                self.xanes_fit_eng_list.max() <
                (self.xanes_fit_edge_eng + 50)):
            self.xanes_fit_gui_h.hs["FitEngRagOptn drpdn"].value = "wl"
            self.xanes_fit_gui_h.hs["FitEngRagOptn drpdn"].disabled = True
            self.xanes_fit_type = "wl"
        else:
            self.xanes_fit_gui_h.hs["FitEngRagOptn drpdn"].value = "full"
            self.xanes_fit_gui_h.hs["FitEngRagOptn drpdn"].disabled = False
            self.xanes_fit_type = "full"

        self.xanes_alignment_done = True
        self.update_xanes2D_config()
        json.dump(
            self.xanes_config,
            open(self.xanes_file_save_trial_reg_config_filename, "w"),
            cls=NumpyArrayEncoder,
        )
        self.boxes_logic()

    def Vis_sldr_chg(self, a):
        self.hs["VisEng text"].value = self.xanes_fit_eng_list[a["owner"].value
                                                               - 1]
        data_state, viewer_state = fiji_viewer_state(
            self.global_h, self, viewer_name="xanes2D_analysis_viewer")
        if not viewer_state:
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes2D_analysis_viewer")
        self.global_h.xanes2D_fiji_windows["xanes2D_analysis_viewer"][
            "ip"].setImage(self.global_h.ij.convert().convert(
                self.global_h.ij.dataset().create(
                    self.global_h.ij.py.to_java(self.xanes_img_roi)),
                self.global_h.ImagePlusClass,
            ))
        self.global_h.xanes2D_fiji_windows["xanes2D_analysis_viewer"][
            "ip"].setSlice(a["owner"].value)
        if self.xanes_visualization_auto_bc:
            self.global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")

    def VisRC_chbx_chg(self, a):
        self.xanes_visualization_auto_bc = a["owner"].value
        self.boxes_logic()

    def SpecInRoi_btn_clk(self, a):
        data_state, viewer_state = fiji_viewer_state(
            self.global_h, self, viewer_name="xanes2D_analysis_viewer")
        if not viewer_state:
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes2D_analysis_viewer")
        width = self.global_h.xanes2D_fiji_windows["xanes2D_analysis_viewer"][
            "ip"].getWidth()
        height = self.global_h.xanes2D_fiji_windows["xanes2D_analysis_viewer"][
            "ip"].getHeight()
        roi = [int((width - 10) / 2), int((height - 10) / 2), 10, 10]
        self.global_h.xanes2D_fiji_windows["xanes2D_analysis_viewer"][
            "ip"].setRoi(roi[0], roi[1], roi[2], roi[3])
        self.global_h.ij.py.run_macro("""run("Plot Z-axis Profile")""")
        self.global_h.xanes2D_fiji_windows["analysis_viewer_z_plot_viewer"][
            "ip"] = self.global_h.WindowManager.getCurrentImage()
        self.global_h.xanes2D_fiji_windows["analysis_viewer_z_plot_viewer"][
            "fiji_id"] = self.global_h.WindowManager.getIDList()[-1]
        self.boxes_logic()
