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

from ..utils.io import data_reader, xanes3D_h5_reader
from . import xanes_fitting_gui as xfg
from . import xanes_analysis_gui as xag
from .gui_components import (
    SelectFilesButton,
    NumpyArrayEncoder,
    enable_disable_boxes,
    gen_external_py_script,
    fiji_viewer_state,
    fiji_viewer_on,
    fiji_viewer_off,
    update_json_content,
    read_config_from_reg_file,
    restart,
    determine_element,
    determine_fitting_energy_range,
)

napari.gui_qt()


class xanes3D_tools_gui:

    def __init__(self, parent_h, form_sz=[650, 740]):
        self.gui_name = "xanes3D"
        self.form_sz = form_sz
        self.global_h = parent_h
        self.hs = {}

        if self.global_h.io_xanes3D_cfg["use_h5_reader"]:
            self.reader = data_reader(xanes3D_h5_reader)
        else:
            from ..external.user_io import user_xanes3D_reader

            self.reader = data_reader(user_xanes3D_reader)

        self.xanes_raw_fn_temp = self.global_h.io_xanes3D_cfg[
            "tomo_raw_fn_template"]
        self.xanes_recon_dir_temp = self.global_h.io_xanes3D_cfg[
            "xanes3D_recon_dir_template"]
        self.xanes_recon_fn_temp = self.global_h.io_xanes3D_cfg[
            "xanes3D_recon_fn_template"]

        self.xanes_fit_external_command_name = os.path.join(
            self.global_h.script_dir, "xanes3D_fit_external_command.py")
        self.xanes_reg_external_command_name = os.path.join(
            self.global_h.script_dir, "xanes3D_reg_external_command.py")
        self.xanes_align_external_command_name = os.path.join(
            self.global_h.script_dir, "xanes3D_align_external_command.py")

        self.xanes_file_configured = False
        self.xanes_indices_configured = False
        self.xanes_roi_configured = False
        self.xanes_reg_params_configured = False
        self.xanes_reg_done = False
        self.xanes_reg_review_done = False
        self.xanes_alignment_done = False
        self.xanes_use_existing_reg_file = False
        self.xanes_use_existing_reg_reviewed = False
        self.xanes_reg_review_file = None
        self.xanes_reg_use_chunk = True
        self.xanes_reg_use_mask = False
        self.xanes_reg_use_smooth_img = False

        self.xanes_raw_h5_path_set = False
        self.xanes_recon_path_set = False
        self.xanes_save_trial_set = False
        self.xanes_scan_id_set = False
        self.xanes_reg_file_set = False
        self.xanes_config_file_set = False
        self.xanes_fixed_scan_id_set = False
        self.xanes_fixed_sli_id_set = False
        self.xanes_reg_file_readed = False
        self.xanes_fit_eng_configured = False

        self.xanes_review_shift_dict = {}
        self.xanes_reg_mask_dilation_width = 0
        self.xanes_reg_mask_thres = 0
        self.xanes_reg_mrtv_level = 4
        self.xanes_reg_mrtv_width = 10
        self.xanes_reg_mrtv_subpixel_wz = 8
        self.xanes_reg_mrtv_subpixel_kernel = 0.2
        self.xanes_reg_mrtv_subpixel_srch_option = "ana"
        self.xanes_img_roi = None
        self.xanes_roi = [0, 10, 0, 10, 0, 10]
        self.xanes_reg_mask = None
        self.xanes_aligned_data = None
        self.xanes_fit_4th_dim_idx = 0
        self.xanes_raw_fn_temp = self.global_h.io_xanes3D_cfg[
            "tomo_raw_fn_template"]
        self.xanes_raw_3D_h5_top_dir = None
        self.xanes_recon_3D_top_dir = None
        self.xanes_save_trial_reg_filename = None
        self.xanes_save_trial_reg_config_filename = None
        self.xanes_save_trial_reg_config_filename_original = None
        self.xanes_raw_3D_h5_temp = None
        self.xanes_available_raw_ids = None
        self.xanes_recon_3D_tiff_temp = None
        self.xanes_recon_3D_dir_temp = None
        self.xanes_reg_best_match_filename = None
        # self.xanes_available_recon_ids = None
        self.xanes_available_sli_file_ids = None
        self.xanes_fit_option = "Do New Reg"
        self.xanes_fixed_scan_id = None
        self.xanes_scan_id_s = None
        self.xanes_scan_id_e = None
        self.xanes_fixed_sli_id = None
        self.xanes_reg_sli_search_half_width = None
        self.xanes_reg_chunk_sz = None
        self.xanes_reg_smooth_sigma = 0
        self.xanes_reg_method = None
        self.xanes_reg_ref_mode = None
        self.xanes_review_bad_shift = False
        self.xanes_visualization_viewer_option = "fiji"
        self.xanes_fit_view_option = "x-y-E"
        self.xanes_element = None
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
        self.xanes_fit_mask_scan_id = None
        self.xanes_fit_mask = 1
        self.xanes_fit_edge_jump_thres = 1.0
        self.xanes_fit_edge_offset_thres = 1.0
        self.xanes_fit_use_flt_spec = False

        self.xanes_config = {
            "filepath config": {
                "xanes3D_raw_3D_h5_top_dir": self.xanes_raw_3D_h5_top_dir,
                "xanes3D_recon_3D_top_dir": self.xanes_recon_3D_top_dir,
                "xanes_save_trial_reg_filename":
                self.xanes_save_trial_reg_filename,
                "xanes3D_save_trial_reg_config_filename":
                self.xanes_save_trial_reg_config_filename,
                "xanes3D_save_trial_reg_config_filename_original":
                self.xanes_save_trial_reg_config_filename_original,
                "xanes3D_raw_3D_h5_temp": self.xanes_raw_3D_h5_temp,
                "xanes3D_recon_3D_tiff_temp": self.xanes_recon_3D_tiff_temp,
                "xanes3D_recon_3D_dir_temp": self.xanes_recon_3D_dir_temp,
                "xanes3D_reg_best_match_filename":
                self.xanes_reg_best_match_filename,
                "xanes3D_analysis_option": self.xanes_fit_option,
                "xanes3D_filepath_configured": self.xanes_file_configured,
                "xanes3D_raw_h5_path_set": self.xanes_raw_h5_path_set,
                "xanes3D_recon_path_set": self.xanes_recon_path_set,
                "xanes3D_save_trial_set": self.xanes_save_trial_set,
            },
            "indeices config": {
                "select_scan_id_start_text_min": 0,
                "select_scan_id_start_text_val": 0,
                "select_scan_id_start_text_max": 0,
                "select_scan_id_end_text_min": 0,
                "select_scan_id_end_text_val": 0,
                "select_scan_id_end_text_max": 0,
                "fixed_scan_id_slider_min": 0,
                "fixed_scan_id_slider_val": 0,
                "fixed_scan_id_slider_max": 0,
                "fixed_sli_id_slider_min": 0,
                "fixed_sli_id_slider_val": 0,
                "fixed_sli_id_slider_max": 0,
                "xanes3D_fixed_scan_id": self.xanes_fixed_scan_id,
                "xanes3D_scan_id_s": self.xanes_scan_id_s,
                "xanes3D_scan_id_e": self.xanes_scan_id_e,
                "xanes3D_fixed_sli_id": self.xanes_fixed_sli_id,
                "xanes3D_scan_id_set": self.xanes_scan_id_set,
                "xanes3D_fixed_scan_id_set": self.xanes_fixed_scan_id_set,
                "xanes3D_fixed_sli_id_set": self.xanes_fixed_sli_id_set,
                "xanes3D_indices_configured": self.xanes_indices_configured,
            },
            "roi config": {
                "3D_roi_x_slider_min": 0,
                "3D_roi_x_slider_val": 0,
                "3D_roi_x_slider_max": 0,
                "3D_roi_y_slider_min": 0,
                "3D_roi_y_slider_val": 0,
                "3D_roi_y_slider_max": 0,
                "3D_roi_z_slider_min": 0,
                "3D_roi_z_slider_val": 0,
                "3D_roi_z_slider_max": 0,
                "3D_roi": list(self.xanes_roi),
                "xanes3D_roi_configured": self.xanes_roi_configured,
            },
            "registration config": {
                "xanes3D_reg_use_chunk": self.xanes_reg_use_chunk,
                "xanes3D_reg_use_mask": self.xanes_reg_use_mask,
                "xanes3D_reg_mask_thres": self.xanes_reg_mask_thres,
                "xanes3D_reg_mask_dilation_width":
                self.xanes_reg_mask_dilation_width,
                "xanes3D_reg_use_smooth_img": self.xanes_reg_use_smooth_img,
                "xanes3D_reg_smooth_sigma": self.xanes_reg_smooth_sigma,
                "xanes3D_reg_sli_search_half_width":
                self.xanes_reg_sli_search_half_width,
                "xanes3D_reg_chunk_sz": self.xanes_reg_chunk_sz,
                "xanes3D_reg_method": self.xanes_reg_method,
                "xanes3D_reg_ref_mode": self.xanes_reg_ref_mode,
                "xanes3D_reg_mrtv_level": self.xanes_reg_mrtv_level,
                "xanes3D_reg_mrtv_width": self.xanes_reg_mrtv_width,
                "xanes3D_reg_mrtv_subpixel_kernel":
                self.xanes_reg_mrtv_subpixel_kernel,
                "xanes3D_reg_mrtv_subpixel_width":
                self.xanes_reg_mrtv_subpixel_wz,
                "mask_thres_slider_min": 0,
                "mask_thres_slider_val": 0,
                "mask_thres_slider_max": 0,
                "mask_dilation_slider_min": 0,
                "mask_dilation_slider_val": 0,
                "mask_dilation_slider_max": 0,
                "sli_search_slider_min": 0,
                "sli_search_slider_val": 0,
                "sli_search_slider_max": 0,
                "chunk_sz_slider_min": 0,
                "chunk_sz_slider_val": 0,
                "chunk_sz_slider_max": 0,
                "xanes3D_reg_params_configured":
                self.xanes_reg_params_configured,
            },
            "run registration": {
                "xanes3D_reg_done": self.xanes_reg_done
            },
            "review registration": {
                "xanes3D_use_existing_reg_reviewed":
                self.xanes_use_existing_reg_reviewed,
                "xanes3D_reg_review_file": self.xanes_reg_review_file,
                "xanes3D_reg_review_done": self.xanes_reg_review_done,
                "read_alignment_checkbox": False,
                "reg_pair_slider_min": 0,
                "reg_pair_slider_val": 0,
                "reg_pair_slider_max": 0,
                "zshift_slider_min": 0,
                "zshift_slider_val": 0,
                "zshift_slider_max": 0,
                "best_match_text": 0,
                "alignment_best_match": self.xanes_review_shift_dict,
            },
            "align 3D recon": {
                "xanes3D_alignment_done": self.xanes_alignment_done,
                "xanes3D_analysis_edge_eng": self.xanes_fit_edge_eng,
                "xanes3D_analysis_wl_fit_eng_s": self.xanes_fit_wl_fit_eng_s,
                "xanes3D_analysis_wl_fit_eng_e": self.xanes_fit_wl_fit_eng_e,
                "xanes3D_analysis_pre_edge_e": self.xanes_fit_pre_edge_e,
                "xanes3D_analysis_post_edge_s": self.xanes_fit_post_edge_s,
                "xanes3D_analysis_edge_0p5_fit_s":
                self.xanes_fit_edge_0p5_fit_s,
                "xanes3D_analysis_edge_0p5_fit_e":
                self.xanes_fit_edge_0p5_fit_e,
                "xanes3D_analysis_type": self.xanes_fit_type,
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
            "SelReconPath text",
            "SelSavTrial text",
            "SelFile&PathCfm text",
            "ConfigDataCfm text",
            "3DRoiCfm text",
            "ConfigRegParamsCfm text",
            "RunRegCfm text",
            "RevRegRltCfm text",
            "AlignReconCfm text",
            "VisSpecView text",
            "VisImgViewAlignEng text",
            "VisSpecView text",
        ]
        enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        self.xanes_fit_gui_h.hs["FitRun text"].disabled = True

    def update_xanes3D_config(self):
        self.xanes_config = {
            "filepath config":
            dict(
                xanes3D_raw_3D_h5_top_dir=self.xanes_raw_3D_h5_top_dir,
                xanes3D_recon_3D_top_dir=self.xanes_recon_3D_top_dir,
                xanes_save_trial_reg_filename=self.
                xanes_save_trial_reg_filename,
                xanes3D_raw_3D_h5_temp=self.xanes_raw_3D_h5_temp,
                xanes3D_recon_3D_tiff_temp=self.xanes_recon_3D_tiff_temp,
                xanes3D_recon_3D_dir_temp=self.xanes_recon_3D_dir_temp,
                xanes3D_reg_best_match_filename=self.
                xanes_reg_best_match_filename,
                xanes3D_analysis_option=self.xanes_fit_option,
                xanes3D_filepath_configured=self.xanes_file_configured,
                xanes3D_raw_h5_path_set=self.xanes_raw_h5_path_set,
                xanes3D_recon_path_set=self.xanes_recon_path_set,
                xanes3D_save_trial_set=self.xanes_save_trial_set,
            ),
            "indeices config":
            dict(
                select_scan_id_start_text_options=self.
                hs["SelScanIdStart drpdn"].options,
                select_scan_id_start_text_val=self.hs["SelScanIdStart drpdn"].
                value,
                select_scan_id_end_text_options=self.hs["SelScanIdEnd drpdn"].
                options,
                select_scan_id_end_text_val=self.hs["SelScanIdEnd drpdn"].
                value,
                fixed_scan_id_slider_options=self.hs["FixedScanId drpdn"].
                options,
                fixed_scan_id_slider_val=self.hs["FixedScanId drpdn"].value,
                fixed_sli_id_slider_min=self.hs["FixedSliId sldr"].min,
                fixed_sli_id_slider_val=self.hs["FixedSliId sldr"].value,
                fixed_sli_id_slider_max=self.hs["FixedSliId sldr"].max,
                xanes3D_fixed_scan_id=int(self.xanes_fixed_scan_id),
                xanes3D_avail_ds_ids=self.xanes_available_raw_ids,
                xanes3D_scan_id_s=int(self.xanes_scan_id_s),
                xanes3D_scan_id_e=int(self.xanes_scan_id_e),
                xanes3D_fixed_sli_id=int(self.xanes_fixed_sli_id),
                xanes3D_scan_id_set=self.xanes_scan_id_set,
                xanes3D_fixed_scan_id_set=self.xanes_fixed_scan_id_set,
                xanes3D_fixed_sli_id_set=self.xanes_fixed_sli_id_set,
                xanes3D_indices_configured=self.xanes_indices_configured,
            ),
            "roi config": {
                "3D_roi_x_slider_min": self.hs["3DRoiX sldr"].min,
                "3D_roi_x_slider_val": self.hs["3DRoiX sldr"].value,
                "3D_roi_x_slider_max": self.hs["3DRoiX sldr"].max,
                "3D_roi_y_slider_min": self.hs["3DRoiY sldr"].min,
                "3D_roi_y_slider_val": self.hs["3DRoiY sldr"].value,
                "3D_roi_y_slider_max": self.hs["3DRoiY sldr"].max,
                "3D_roi_z_slider_min": self.hs["3DRoiZ sldr"].min,
                "3D_roi_z_slider_val": self.hs["3DRoiZ sldr"].value,
                "3D_roi_z_slider_max": self.hs["3DRoiZ sldr"].max,
                "3D_roi": list(self.xanes_roi),
                "xanes3D_roi_configured": self.xanes_roi_configured,
            },
            "registration config":
            dict(
                xanes3D_reg_use_chunk=self.hs["ChunkSz chbx"].value,
                xanes3D_reg_use_mask=self.hs["UseMask chbx"].value,
                xanes3D_reg_mask_thres=self.hs["MaskThres sldr"].value,
                xanes3D_reg_mask_dilation_width=self.hs["MaskDilation sldr"].
                value,
                xanes3D_reg_use_smooth_img=self.hs["UseMask chbx"].value,
                xanes3D_reg_smooth_sigma=self.xanes_reg_smooth_sigma,
                xanes3D_reg_sli_search_half_width=self.
                xanes_reg_sli_search_half_width,
                xanes3D_reg_chunk_sz=self.xanes_reg_chunk_sz,
                xanes3D_reg_method=self.xanes_reg_method,
                xanes3D_reg_ref_mode=self.xanes_reg_ref_mode,
                xanes3D_reg_mrtv_level=self.xanes_reg_mrtv_level,
                xanes3D_reg_mrtv_width=self.xanes_reg_mrtv_width,
                xanes3D_reg_mrtv_subpixel_kernel=self.
                xanes_reg_mrtv_subpixel_kernel,
                xanes3D_reg_mrtv_subpixel_width=self.
                xanes_reg_mrtv_subpixel_wz,
                mask_thres_slider_min=self.hs["MaskThres sldr"].min,
                mask_thres_slider_val=self.hs["MaskThres sldr"].value,
                mask_thres_slider_max=self.hs["MaskThres sldr"].max,
                mask_dilation_slider_min=self.hs["MaskDilation sldr"].min,
                mask_dilation_slider_val=self.hs["MaskDilation sldr"].value,
                mask_dilation_slider_max=self.hs["MaskDilation sldr"].max,
                sli_search_slider_min=self.hs["SliSrch sldr"].min,
                sli_search_slider_val=self.hs["SliSrch sldr"].value,
                sli_search_slider_max=self.hs["SliSrch sldr"].max,
                chunk_sz_slider_min=self.hs["ChunkSz sldr"].min,
                chunk_sz_slider_val=self.hs["ChunkSz sldr"].value,
                chunk_sz_slider_max=self.hs["ChunkSz sldr"].max,
                xanes3D_reg_params_configured=self.xanes_reg_params_configured,
            ),
            "run registration":
            dict(xanes3D_reg_done=self.xanes_reg_done),
            "review registration":
            dict(
                xanes3D_use_existing_reg_reviewed=self.
                xanes_use_existing_reg_reviewed,
                xanes3D_reg_review_file=self.xanes_reg_review_file,
                xanes3D_reg_review_done=self.xanes_reg_review_done,
                read_alignment_checkbox=self.hs["ReadAlign chbx"].value,
                reg_pair_slider_min=self.hs["RegPair sldr"].min,
                reg_pair_slider_val=self.hs["RegPair sldr"].value,
                reg_pair_slider_max=self.hs["RegPair sldr"].max,
                alignment_best_match=self.xanes_review_shift_dict,
            ),
            "align 3D recon":
            dict(
                xanes3D_alignment_done=self.xanes_alignment_done,
                xanes3D_analysis_edge_eng=self.xanes_fit_edge_eng,
                xanes3D_analysis_wl_fit_eng_s=self.xanes_fit_wl_fit_eng_s,
                xanes3D_analysis_wl_fit_eng_e=self.xanes_fit_wl_fit_eng_e,
                xanes3D_analysis_pre_edge_e=self.xanes_fit_pre_edge_e,
                xanes3D_analysis_post_edge_s=self.xanes_fit_post_edge_s,
                xanes3D_analysis_edge_0p5_fit_s=self.xanes_fit_edge_0p5_fit_s,
                xanes3D_analysis_edge_0p5_fit_e=self.xanes_fit_edge_0p5_fit_e,
                xanes3D_analysis_type=self.xanes_fit_type,
            ),
        }

    def read_xanes3D_config(self):
        with open(self.xanes_save_trial_reg_config_filename_original,
                  "r") as f:
            self.xanes_config = json.load(f)

    def set_xanes3D_variables(self):
        self.xanes_raw_3D_h5_top_dir = self.xanes_config["filepath config"][
            "xanes3D_raw_3D_h5_top_dir"]
        self.xanes_recon_3D_top_dir = self.xanes_config["filepath config"][
            "xanes3D_recon_3D_top_dir"]
        self.xanes_save_trial_reg_filename = self.xanes_config[
            "filepath config"]["xanes_save_trial_reg_filename"]
        self.xanes_raw_3D_h5_temp = self.xanes_config["filepath config"][
            "xanes3D_raw_3D_h5_temp"]
        self.xanes_recon_3D_tiff_temp = self.xanes_config["filepath config"][
            "xanes3D_recon_3D_tiff_temp"]
        self.xanes_recon_3D_dir_temp = self.xanes_config["filepath config"][
            "xanes3D_recon_3D_dir_temp"]
        self.xanes_reg_best_match_filename = self.xanes_config[
            "filepath config"]["xanes3D_reg_best_match_filename"]
        self.xanes_fit_option = self.xanes_config["filepath config"][
            "xanes3D_analysis_option"]
        self.xanes_file_configured = self.xanes_config["filepath config"][
            "xanes3D_filepath_configured"]
        self.xanes_raw_h5_path_set = self.xanes_config["filepath config"][
            "xanes3D_raw_h5_path_set"]
        self.xanes_recon_path_set = self.xanes_config["filepath config"][
            "xanes3D_recon_path_set"]
        self.xanes_save_trial_set = self.xanes_config["filepath config"][
            "xanes3D_save_trial_set"]

        self.xanes_fixed_scan_id = self.xanes_config["indeices config"][
            "xanes3D_fixed_scan_id"]
        self.xanes_available_raw_ids = self.xanes_config["indeices config"][
            "xanes3D_avail_ds_ids"]
        self.xanes_scan_id_s = self.xanes_config["indeices config"][
            "xanes3D_scan_id_s"]
        self.xanes_scan_id_e = self.xanes_config["indeices config"][
            "xanes3D_scan_id_e"]
        self.xanes_fixed_sli_id = self.xanes_config["indeices config"][
            "xanes3D_fixed_sli_id"]
        self.xanes_scan_id_set = self.xanes_config["indeices config"][
            "xanes3D_scan_id_set"]
        self.xanes_fixed_scan_id_set = self.xanes_config["indeices config"][
            "xanes3D_fixed_scan_id_set"]
        self.xanes_fixed_sli_id_set = self.xanes_config["indeices config"][
            "xanes3D_fixed_sli_id_set"]
        self.xanes_indices_configured = self.xanes_config["indeices config"][
            "xanes3D_indices_configured"]
        b = glob.glob(
            self.xanes_recon_3D_tiff_temp.format(
                self.xanes_available_raw_ids[self.xanes_fixed_scan_id], "*"))
        self.xanes_available_sli_file_ids = sorted([
            int(os.path.basename(ii).split(".")[0].split("_")[-1]) for ii in b
        ])

        self.xanes_roi = self.xanes_config["roi config"]["3D_roi"]
        self.xanes_roi_configured = self.xanes_config["roi config"][
            "xanes3D_roi_configured"]

        self.xanes_reg_use_chunk = self.xanes_config["registration config"][
            "xanes3D_reg_use_chunk"]
        self.xanes_reg_use_mask = self.xanes_config["registration config"][
            "xanes3D_reg_use_mask"]
        self.xanes_reg_mask_thres = self.xanes_config["registration config"][
            "xanes3D_reg_mask_thres"]
        self.xanes_reg_mask_dilation_width = self.xanes_config[
            "registration config"]["xanes3D_reg_mask_dilation_width"]
        self.xanes_reg_use_smooth_img = self.xanes_config[
            "registration config"]["xanes3D_reg_use_smooth_img"]
        self.xanes_reg_smooth_sigma = self.xanes_config["registration config"][
            "xanes3D_reg_smooth_sigma"]
        self.xanes_reg_sli_search_half_width = self.xanes_config[
            "registration config"]["xanes3D_reg_sli_search_half_width"]
        self.xanes_reg_chunk_sz = self.xanes_config["registration config"][
            "xanes3D_reg_chunk_sz"]

        self.xanes_reg_method = self.xanes_config["registration config"][
            "xanes3D_reg_method"]
        self.xanes_reg_ref_mode = self.xanes_config["registration config"][
            "xanes3D_reg_ref_mode"]
        self.xanes_reg_mrtv_level = self.xanes_config["registration config"][
            "xanes3D_reg_mrtv_level"]
        self.xanes_reg_mrtv_width = self.xanes_config["registration config"][
            "xanes3D_reg_mrtv_width"]
        self.xanes_reg_mrtv_subpixel_kernel = self.xanes_config[
            "registration config"]["xanes3D_reg_mrtv_subpixel_kernel"]
        self.xanes_reg_mrtv_subpixel_wz = self.xanes_config[
            "registration config"]["xanes3D_reg_mrtv_subpixel_width"]

        self.xanes_reg_params_configured = self.xanes_config[
            "registration config"]["xanes3D_reg_params_configured"]

        self.xanes_reg_done = self.xanes_config["run registration"][
            "xanes3D_reg_done"]

        self.xanes_use_existing_reg_reviewed = self.xanes_config[
            "review registration"]["xanes3D_use_existing_reg_reviewed"]
        self.xanes_reg_review_file = self.xanes_config["review registration"][
            "xanes3D_reg_review_file"]
        self.xanes_reg_review_done = self.xanes_config["review registration"][
            "xanes3D_reg_review_done"]
        self.xanes_review_shift_dict = self.xanes_config[
            "review registration"]["alignment_best_match"]

        self.xanes_alignment_done = self.xanes_config["align 3D recon"][
            "xanes3D_alignment_done"]
        self.xanes_fit_edge_eng = self.xanes_config["align 3D recon"][
            "xanes3D_analysis_edge_eng"]
        self.xanes_fit_wl_fit_eng_s = self.xanes_config["align 3D recon"][
            "xanes3D_analysis_wl_fit_eng_s"]
        self.xanes_fit_wl_fit_eng_e = self.xanes_config["align 3D recon"][
            "xanes3D_analysis_wl_fit_eng_e"]
        self.xanes_fit_pre_edge_e = self.xanes_config["align 3D recon"][
            "xanes3D_analysis_pre_edge_e"]
        self.xanes_fit_post_edge_s = self.xanes_config["align 3D recon"][
            "xanes3D_analysis_post_edge_s"]
        self.xanes_fit_edge_0p5_fit_s = self.xanes_config["align 3D recon"][
            "xanes3D_analysis_edge_0p5_fit_s"]
        self.xanes_fit_edge_0p5_fit_e = self.xanes_config["align 3D recon"][
            "xanes3D_analysis_edge_0p5_fit_e"]
        self.xanes_fit_type = self.xanes_config["align 3D recon"][
            "xanes3D_analysis_type"]

        self.boxes_logic()

    def set_xanes3D_handles(self):
        self.hs["SelScanIdStart drpdn"].options = self.xanes_config[
            "indeices config"]["select_scan_id_start_text_options"]
        self.hs["SelScanIdEnd drpdn"].options = self.xanes_config[
            "indeices config"]["select_scan_id_end_text_options"]
        self.hs["SelScanIdStart drpdn"].value = self.xanes_config[
            "indeices config"]["select_scan_id_start_text_val"]
        self.hs["SelScanIdEnd drpdn"].value = self.xanes_config[
            "indeices config"]["select_scan_id_end_text_val"]

        self.hs["FixedScanId drpdn"].options = self.xanes_config[
            "indeices config"]["fixed_scan_id_slider_options"]
        self.hs["FixedScanId drpdn"].value = self.xanes_config[
            "indeices config"]["fixed_scan_id_slider_val"]
        self.hs["FixedSliId sldr"].max = self.xanes_config["indeices config"][
            "fixed_sli_id_slider_max"]
        self.hs["FixedSliId sldr"].min = self.xanes_config["indeices config"][
            "fixed_sli_id_slider_min"]
        self.hs["FixedSliId sldr"].value = self.xanes_config[
            "indeices config"]["fixed_sli_id_slider_val"]

        self.hs["3DRoiX sldr"].max = self.xanes_config["roi config"][
            "3D_roi_x_slider_max"]
        self.hs["3DRoiX sldr"].min = self.xanes_config["roi config"][
            "3D_roi_x_slider_min"]
        self.hs["3DRoiX sldr"].value = self.xanes_config["roi config"][
            "3D_roi_x_slider_val"]
        self.hs["3DRoiY sldr"].max = self.xanes_config["roi config"][
            "3D_roi_y_slider_max"]
        self.hs["3DRoiY sldr"].min = self.xanes_config["roi config"][
            "3D_roi_y_slider_min"]
        self.hs["3DRoiY sldr"].value = self.xanes_config["roi config"][
            "3D_roi_y_slider_val"]
        self.hs["3DRoiZ sldr"].max = self.xanes_config["roi config"][
            "3D_roi_z_slider_max"]
        self.hs["3DRoiZ sldr"].min = self.xanes_config["roi config"][
            "3D_roi_z_slider_min"]
        self.hs["3DRoiZ sldr"].value = self.xanes_config["roi config"][
            "3D_roi_z_slider_val"]

        self.hs["MaskThres sldr"].max = self.xanes_config[
            "registration config"]["mask_thres_slider_max"]
        self.hs["MaskThres sldr"].min = self.xanes_config[
            "registration config"]["mask_thres_slider_min"]
        self.hs["MaskThres sldr"].value = self.xanes_config[
            "registration config"]["mask_thres_slider_val"]
        self.hs["MaskDilation sldr"].max = self.xanes_config[
            "registration config"]["mask_dilation_slider_max"]
        self.hs["MaskDilation sldr"].min = self.xanes_config[
            "registration config"]["mask_dilation_slider_min"]
        self.hs["MaskDilation sldr"].value = self.xanes_config[
            "registration config"]["mask_dilation_slider_val"]
        self.hs["SliSrch sldr"].max = self.xanes_config["registration config"][
            "sli_search_slider_max"]
        self.hs["SliSrch sldr"].min = self.xanes_config["registration config"][
            "sli_search_slider_min"]
        self.hs["SliSrch sldr"].value = self.xanes_config[
            "registration config"]["sli_search_slider_val"]
        self.hs["ChunkSz sldr"].max = self.xanes_config["registration config"][
            "chunk_sz_slider_max"]
        self.hs["ChunkSz sldr"].min = self.xanes_config["registration config"][
            "chunk_sz_slider_min"]
        self.hs["ChunkSz sldr"].value = self.xanes_config[
            "registration config"]["chunk_sz_slider_val"]

        self.hs["RegMethod drpdn"].value = self.xanes_config[
            "registration config"]["xanes3D_reg_method"]
        self.hs["RefMode drpdn"].value = self.xanes_config[
            "registration config"]["xanes3D_reg_ref_mode"]
        self.hs["MRTVLevel text"].value = self.xanes_config[
            "registration config"]["xanes3D_reg_mrtv_level"]
        self.hs["MRTVWz text"].value = self.xanes_config[
            "registration config"]["xanes3D_reg_mrtv_width"]
        self.hs["MRTVSubpixelWz text"].value = self.xanes_config[
            "registration config"]["xanes3D_reg_mrtv_subpixel_width"]
        self.hs["MRTVSubpixelKernel text"].value = self.xanes_config[
            "registration config"]["xanes3D_reg_mrtv_subpixel_kernel"]

        self.hs["ReadAlign chbx"].value = self.xanes_config[
            "review registration"]["read_alignment_checkbox"]
        self.hs["RegPair sldr"].max = self.xanes_config["review registration"][
            "reg_pair_slider_max"]
        self.hs["RegPair sldr"].min = self.xanes_config["review registration"][
            "reg_pair_slider_min"]
        self.hs["RegPair sldr"].value = self.xanes_config[
            "review registration"]["reg_pair_slider_val"]

        self.xanes_fit_gui_h.hs[
            "FitEngRagOptn drpdn"].value = self.xanes_config["align 3D recon"][
                "xanes3D_analysis_type"]
        self.xanes_fit_gui_h.hs[
            "FitEngRagEdgeEng text"].value = self.xanes_config[
                "align 3D recon"]["xanes3D_analysis_edge_eng"]
        self.xanes_fit_gui_h.hs[
            "FitEngRagPreEdgeEnd text"].value = self.xanes_config[
                "align 3D recon"]["xanes3D_analysis_pre_edge_e"]
        self.xanes_fit_gui_h.hs[
            "FitEngRagPostEdgeStr text"].value = self.xanes_config[
                "align 3D recon"]["xanes3D_analysis_post_edge_s"]
        self.xanes_fit_gui_h.hs[
            "FitEngRagWlFitStr text"].value = self.xanes_config[
                "align 3D recon"]["xanes3D_analysis_wl_fit_eng_s"]
        self.xanes_fit_gui_h.hs[
            "FitEngRagWlFitEnd text"].value = self.xanes_config[
                "align 3D recon"]["xanes3D_analysis_wl_fit_eng_e"]
        self.xanes_fit_gui_h.hs[
            "FitEngRagEdge0.5Str text"].value = self.xanes_config[
                "align 3D recon"]["xanes3D_analysis_edge_0p5_fit_s"]
        self.xanes_fit_gui_h.hs[
            "FitEngRagEdge0.5End text"].value = self.xanes_config[
                "align 3D recon"]["xanes3D_analysis_edge_0p5_fit_e"]

        self.boxes_logic()

    def boxes_logic(self):

        def xanes3D_compound_logic():
            if self.xanes_fit_option == "Reg By Shift":
                if self.xanes_roi_configured:
                    boxes = ["RevRegRlt box"]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=True,
                                         level=-1)
                    self.hs["ReadAlign chbx"].value = True
                    self.hs["ReadAlign btn"].disabled = False
                else:
                    boxes = ["RevRegRlt box"]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=True,
                                         level=-1)
                if self.xanes_reg_file_readed:
                    self.hs["RevRegRltCfm btn"].disabled = False
            else:
                if self.xanes_reg_done:
                    if self.hs["ReadAlign chbx"].value & (
                            not self.xanes_reg_file_readed):
                        boxes = ["RegPair box", "CorrShft box"]
                        enable_disable_boxes(self.hs,
                                             boxes,
                                             disabled=True,
                                             level=-1)
                        self.hs["ReadAlign btn"].disabled = False
                    else:
                        self.hs["ReadAlign btn"].disabled = True
                        if self.xanes_review_bad_shift:
                            boxes = ["CorrShft box"]
                            enable_disable_boxes(self.hs,
                                                 boxes,
                                                 disabled=False,
                                                 level=-1)
                            boxes = ["RegPair box"]
                            enable_disable_boxes(self.hs,
                                                 boxes,
                                                 disabled=True,
                                                 level=-1)
                        else:
                            boxes = ["RegPair box"]
                            enable_disable_boxes(self.hs,
                                                 boxes,
                                                 disabled=False,
                                                 level=-1)
                            boxes = ["CorrShft box"]
                            enable_disable_boxes(self.hs,
                                                 boxes,
                                                 disabled=True,
                                                 level=-1)
                else:
                    boxes = ["RevRegRlt box"]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=True,
                                         level=-1)

            if self.xanes_fit_eng_configured:
                self.xanes_fit_gui_h.hs["FitRun btn"].disabled = False
            else:
                self.xanes_fit_gui_h.hs["FitRun btn"].disabled = True

            if self.xanes_reg_review_done:
                if self.hs["AlignReconOptnSli chbx"].value:
                    boxes = [
                        "AlignReconOptnSliStart text",
                        "AlignReconOptnSliRange sldr",
                        "AlignReconOptnSliEnd text",
                    ]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=False,
                                         level=-1)
                else:
                    boxes = [
                        "AlignReconOptnSliStart text",
                        "AlignReconOptnSliRange sldr",
                        "AlignReconOptnSliEnd text",
                    ]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=True,
                                         level=-1)
            else:
                boxes = [
                    "AlignReconOptnSliStart text",
                    "AlignReconOptnSliRange sldr",
                    "AlignReconOptnSliEnd text",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)

            if self.xanes_alignment_done | (self.xanes_fit_option
                                            == "Do Analysis"):
                if self.xanes_visualization_viewer_option == "fiji":
                    boxes = ["VisImgViewAlign box", "VisSpecView box"]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=False,
                                         level=-1)
                elif self.xanes_visualization_viewer_option == "napari":
                    boxes = ["VisImgViewAlign box", "VisSpecView box"]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=True,
                                         level=-1)

            if self.hs["ChunkSz chbx"].value:
                self.hs["ChunkSz sldr"].disabled = False
                self.xanes_regparams_anchor_idx_set = False
            else:
                self.hs["ChunkSz sldr"].disabled = True
                self.xanes_regparams_anchor_idx_set = False

            if self.hs["RegMethod drpdn"].value in ["MPC", "MPC+MRTV"]:
                self.hs["UseMask chbx"].value = 1
                self.hs["MaskDilation sldr"].disabled = False
                self.hs["MaskThres sldr"].disabled = False
            elif self.hs["RegMethod drpdn"].value in ["MRTV", "PC", "LS_MRTV"]:
                self.hs["UseMask chbx"].value = 0
                self.hs["MaskDilation sldr"].disabled = True
                self.hs["MaskThres sldr"].disabled = True
            elif self.hs["RegMethod drpdn"].value == "SR":
                self.hs["MaskDilation sldr"].disabled = False
                self.hs["MaskThres sldr"].disabled = False

            if self.hs["RegMethod drpdn"].value in ["MPC", "SR", "PC"]:
                boxes = ["MRTVOptn box"]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            elif self.hs["RegMethod drpdn"].value == "MPC+MRTV":
                self.hs["MRTVLevel text"].value = 1
                self.hs["MRTVLevel text"].disabled = True
                self.hs["MRTVWz text"].disabled = True
                self.hs["MRTVSubpixelWz text"].disabled = False
                self.hs["MRTVSubpixelKernel text"].disabled = False
            elif self.hs["RegMethod drpdn"].value == "LS+MRTV":
                self.hs["MRTVLevel text"].value = 2
                self.hs["MRTVLevel text"].disabled = True
                self.hs["MRTVWz text"].disabled = False
                self.hs["MRTVWz text"].max = 300
                self.hs["MRTVWz text"].value = 100
                self.hs["MRTVSubpixelWz text"].disabled = False
                self.hs["MRTVSubpixelKernel text"].disabled = False
            elif self.hs["RegMethod drpdn"].value == "MRTV":
                if self.hs["MRTVWz text"].value > 20:
                    self.hs["MRTVWz text"].value = 10
                self.hs["MRTVWz text"].max = 20
                boxes = ["MRTVOptn box"]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)

            if self.hs["RegMethod drpdn"].value in ["PC", "MRTV", "LS+MRTV"]:
                boxes = ["MaskOptn box"]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            elif self.hs["RegMethod drpdn"].value in ["MPC", "MPC+MRTV", "SR"]:
                boxes = ["MaskOptn box"]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)

            if self.hs["UseMask chbx"].value == 1:
                self.hs["MaskThres sldr"].disabled = False
                self.hs["MaskDilation sldr"].disabled = False
            else:
                self.hs["MaskThres sldr"].disabled = True
                self.hs["MaskDilation sldr"].disabled = True

            if self.hs["MRTVSubpixelSrch drpdn"].value == "analytical":
                self.hs["MRTVSubpixelWz text"].disabled = True
            else:
                self.hs["MRTVSubpixelWz text"].disabled = False

        if self.xanes_fit_option in ["Do New Reg", "Read Config File"]:
            if not self.xanes_file_configured:
                boxes = [
                    "ConfigData box",
                    "3DRoi box",
                    "ConfigRegParams box",
                    "RunReg box",
                    "RevRegRlt box",
                    "AlignRecon box",
                    "VisImg box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            elif self.xanes_file_configured & (
                    not self.xanes_indices_configured):
                if not self.xanes_scan_id_set:
                    boxes = [
                        "ConfigData box",
                        "3DRoi box",
                        "ConfigRegParams box",
                        "RunReg box",
                        "RevRegRlt box",
                        "AlignRecon box",
                        "VisImg box",
                    ]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=True,
                                         level=-1)
                    boxes = ["ScanIdRange box"]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=False,
                                         level=-1)
                    boxes = ["Fitting form"]
                    enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                         boxes,
                                         disabled=True,
                                         level=-1)
                else:
                    boxes = [
                        "3DRoi box",
                        "ConfigRegParams box",
                        "RunReg box",
                        "RevRegRlt box",
                        "AlignRecon box",
                        "VisImg box",
                    ]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=True,
                                         level=-1)
                    boxes = ["ConfigData box"]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=False,
                                         level=-1)
                    boxes = ["Fitting form"]
                    enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                         boxes,
                                         disabled=True,
                                         level=-1)
            elif (self.xanes_file_configured & self.xanes_indices_configured
                  ) & (not self.xanes_roi_configured):
                boxes = ["ConfigData box", "3DRoi box"]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = [
                    "ConfigRegParams box",
                    "RunReg box",
                    "RevRegRlt box",
                    "AlignRecon box",
                    "VisImg box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            elif (self.xanes_file_configured & self.xanes_indices_configured
                  ) & (self.xanes_roi_configured &
                       (not self.xanes_reg_params_configured)):
                boxes = [
                    "ConfigData box",
                    "3DRoi box",
                    "ConfigRegParams box",
                    "SliSrch sldr",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = [
                    "RunReg box", "RevRegRlt box", "AlignRecon box",
                    "VisImg box"
                ]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
                data_state, viewer_state = fiji_viewer_state(
                    self.global_h, self, viewer_name="xanes3D_mask_viewer")
                if not viewer_state:
                    self.global_h.xanes3D_fiji_windows["xanes3D_mask_viewer"][
                        "fiji_id"] = None
                    self.global_h.xanes3D_fiji_windows["xanes3D_mask_viewer"][
                        "ip"] = None
                    boxes = ["MaskOptn box"]
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=False,
                                         level=-1)
            elif (
                (self.xanes_file_configured & self.xanes_indices_configured)
                    &
                (self.xanes_roi_configured & self.xanes_reg_params_configured)
                    & (not self.xanes_reg_done)):
                boxes = [
                    "ConfigData box",
                    "3DRoi box",
                    "ConfigRegParams box",
                    "RunReg box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ["AlignRecon box", "RevRegRlt box", "VisImg box"]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            elif (
                (self.xanes_file_configured & self.xanes_indices_configured)
                    &
                (self.xanes_roi_configured & self.xanes_reg_params_configured)
                    & (self.xanes_reg_done &
                       (not self.xanes_reg_review_done))):
                boxes = [
                    "ConfigData box",
                    "3DRoi box",
                    "ConfigRegParams box",
                    "RunReg box",
                    "RevRegRlt box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ["AlignRecon box", "VisImg box"]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            elif (
                (self.xanes_file_configured & self.xanes_indices_configured)
                    &
                (self.xanes_roi_configured & self.xanes_reg_params_configured)
                    & (self.xanes_reg_done & self.xanes_reg_review_done)
                    & (not self.xanes_alignment_done)):
                boxes = [
                    "ConfigData box",
                    "3DRoi box",
                    "ConfigRegParams box",
                    "RunReg box",
                    "RevRegRlt box",
                    "AlignRecon box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ["VisImg box"]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            elif (
                (self.xanes_file_configured & self.xanes_indices_configured)
                    &
                (self.xanes_roi_configured & self.xanes_reg_params_configured)
                    & (self.xanes_reg_done & self.xanes_reg_review_done)
                    & (self.xanes_alignment_done &
                       (self.xanes_element is None))):
                boxes = [
                    "ConfigData box",
                    "3DRoi box",
                    "ConfigRegParams box",
                    "RunReg box",
                    "RevRegRlt box",
                    "AlignRecon box",
                    "VisImg box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            elif (
                (self.xanes_file_configured & self.xanes_indices_configured)
                    &
                (self.xanes_roi_configured & self.xanes_reg_params_configured)
                    & (self.xanes_reg_done & self.xanes_reg_review_done)
                    & (self.xanes_alignment_done &
                       (not self.xanes_fit_eng_configured))):
                boxes = [
                    "ConfigData box",
                    "3DRoi box",
                    "ConfigRegParams box",
                    "RunReg box",
                    "RevRegRlt box",
                    "AlignRecon box",
                    "VisImg box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ["FitEngRag box"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=False,
                                     level=-1)
                boxes = ["FitItemConfig box"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
                self.lock_message_text_boxes()
                xanes3D_compound_logic()
            elif (
                (self.xanes_file_configured & self.xanes_indices_configured)
                    &
                (self.xanes_roi_configured & self.xanes_reg_params_configured)
                    & (self.xanes_reg_done & self.xanes_reg_review_done)
                    &
                (self.xanes_alignment_done & self.xanes_fit_eng_configured)):
                boxes = [
                    "ConfigData box",
                    "3DRoi box",
                    "ConfigRegParams box",
                    "RunReg box",
                    "RevRegRlt box",
                    "AlignRecon box",
                    "VisImg box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ["FitEngRag box", "FitItemConfig box"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=False,
                                     level=-1)
        elif self.xanes_fit_option == "Reg By Shift":
            boxes = ["ConfigData box", "ConfigRegParams box", "RunReg box"]
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            self.hs["ReadAlign chbx"].value = True
            self.hs["ReadAlign chbx"].disabled = True
            if not self.xanes_file_configured:
                boxes = [
                    "3DRoi box", "AlignRecon box", "VisImg box",
                    "RevRegRlt box"
                ]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            elif self.xanes_file_configured & (not self.xanes_roi_configured):
                boxes = ["3DRoi box"]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ["VisImg box", "AlignRecon box", "RevRegRlt box"]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            elif (self.xanes_file_configured & self.xanes_roi_configured) & (
                    not self.xanes_reg_review_done):
                boxes = ["3DRoi box", "ReadAlignFile box"]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                self.hs["ReadAlign chbx"].value = True
                self.hs["ReadAlign chbx"].disabled = True
                self.hs["RevRegRltCfm btn"].disabled = False

                boxes = ["VisImg box", "AlignRecon box"]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            elif (self.xanes_file_configured & self.xanes_roi_configured) & (
                    self.xanes_reg_review_done &
                (not self.xanes_alignment_done)):
                boxes = ["3DRoi box", "AlignRecon box", "ReadAlignFile box"]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                self.hs["ReadAlign chbx"].value = True
                self.hs["ReadAlign chbx"].disabled = True
                self.hs["RevRegRltCfm btn"].disabled = False

                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
                boxes = ["VisImg box"]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            elif ((self.xanes_file_configured & self.xanes_roi_configured)
                  & (self.xanes_reg_review_done & self.xanes_alignment_done)
                  & (not self.xanes_fit_eng_configured)):
                boxes = [
                    "3DRoi box",
                    "AlignRecon box",
                    "VisImg box",
                    "ReadAlignFile box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                self.hs["ReadAlign chbx"].value = True
                self.hs["ReadAlign chbx"].disabled = True
                self.hs["RevRegRltCfm btn"].disabled = False

                boxes = ["FitEngRag box"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=False,
                                     level=-1)
                boxes = ["FitItemConfig box"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            elif ((self.xanes_file_configured & self.xanes_roi_configured)
                  & (self.xanes_reg_review_done & self.xanes_alignment_done)
                  & self.xanes_fit_eng_configured):
                boxes = [
                    "3DRoi box",
                    "AlignRecon box",
                    "VisImg box",
                    "ReadAlignFile box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                self.hs["ReadAlign chbx"].value = True
                self.hs["ReadAlign chbx"].disabled = True
                self.hs["RevRegRltCfm btn"].disabled = False

                boxes = ["Fitting form"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=False,
                                     level=-1)
        elif self.xanes_fit_option == "Do Analysis":
            if self.xanes_reg_file_set & self.xanes_file_configured:
                boxes = [
                    "ConfigData box",
                    "3DRoi box",
                    "ConfigRegParams box",
                    "RunReg box",
                    "RevRegRlt box",
                    "AlignRecon box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["VisImg box"]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                boxes = ["FitEngRag box"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=False,
                                     level=-1)
                boxes = ["FitItemConfig box"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
                boxes = ["FitItemConfig box"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
            else:
                boxes = [
                    "ConfigData box",
                    "3DRoi box",
                    "ConfigRegParams box",
                    "RunReg box",
                    "RevRegRlt box",
                    "AlignRecon box",
                    "VisImg box",
                ]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ["FitEngRag box", "FitItemConfig box"]
                enable_disable_boxes(self.xanes_fit_gui_h.hs,
                                     boxes,
                                     disabled=True,
                                     level=-1)
                self.hs[
                    "SelFile&PathCfm text"].value = "Please specify and confirm the aligned xanes3D file ..."
        self.lock_message_text_boxes()
        xanes3D_compound_logic()

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
            "height": f"{0.35*(self.form_sz[0]-136)}px",
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
            option="askdirectory", text_h=self.hs["SelRawH5Path text"])
        layout = {"width": "15%"}
        self.hs["SelRawH5Path btn"].layout = layout
        self.hs["SelRawH5Path btn"].description = "Raw h5 Dir"
        self.hs["SelRawH5Path btn"].on_click(self.SelRawH5Path_btn_clk)
        self.hs["SelRaw box"].children = [
            self.hs["SelRawH5Path text"],
            self.hs["SelRawH5Path btn"],
        ]

        ## ## ## ## ## recon top directory
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["SelRecon box"] = widgets.HBox()
        self.hs["SelRecon box"].layout = layout
        self.hs["SelReconPath text"] = widgets.Text(
            value="Choose recon top directory ...",
            description="",
            disabled=True)
        layout = {"width": "66%", "display": "inline_flex"}
        self.hs["SelReconPath text"].layout = layout
        self.hs["SelReconPath btn"] = SelectFilesButton(
            option="askdirectory", text_h=self.hs["SelReconPath text"])
        layout = {"width": "15%"}
        self.hs["SelReconPath btn"].layout = layout
        self.hs["SelReconPath btn"].description = "Recon Top Dir"
        self.hs["SelReconPath btn"].on_click(self.SelReconPath_btn_clk)
        self.hs["SelRecon box"].children = [
            self.hs["SelReconPath text"],
            self.hs["SelReconPath btn"],
        ]

        ## ## ## ## ## trial save file
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["SelSavTrial box"] = widgets.HBox()
        self.hs["SelSavTrial box"].layout = layout
        self.hs["SelSavTrial text"] = widgets.Text(
            value="Save trial registration as ...",
            description="",
            disabled=True)
        layout = {"width": "66%", "display": "inline_flex"}
        self.hs["SelSavTrial text"].layout = layout
        self.hs["SelSavTrial btn"] = SelectFilesButton(
            option="asksaveasfilename", text_h=self.hs["SelSavTrial text"])
        self.hs["SelSavTrial btn"].description = "Save Reg File"
        layout = {"width": "15%"}
        self.hs["SelSavTrial btn"].layout = layout
        self.hs["SelSavTrial btn"].on_click(self.SelSavTrial_btn_clk)
        self.hs["SelSavTrial box"].children = [
            self.hs["SelSavTrial text"],
            self.hs["SelSavTrial btn"],
        ]

        ## ## ## ## ## confirm file configuration
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["SelFile&PathCfm box"] = widgets.HBox()
        self.hs["SelFile&PathCfm box"].layout = layout
        self.hs["SelFile&PathCfm text"] = widgets.Text(
            value=
            "Save trial registration, or go directly review registration ...",
            description="",
            disabled=True,
        )
        layout = {"width": "66%"}
        self.hs["SelFile&PathCfm text"].layout = layout
        self.hs["SelFile&PathCfm btn"] = widgets.Button(
            description="Confirm",
            tooltip="Confirm: Confirm after you finish file configuration",
        )
        self.hs["SelFile&PathCfm btn"].style.button_color = "darkviolet"
        self.hs["SelFile&PathCfm btn"].on_click(self.SelFilePathCfm_btn_clk)
        layout = {"width": "15%"}
        self.hs["SelFile&PathCfm btn"].layout = layout

        self.hs["File&PathOptn drpdn"] = widgets.Dropdown(
            value="Do New Reg",
            options=[
                "Do New Reg",
                # 'Read Reg File',
                "Read Config File",
                "Reg By Shift",
                "Do Analysis",
            ],
            description="",
            disabled=False,
            description_tooltip=
            '"Do New Reg": start registration and review results from beginning; "Read Config File": if you like to resume analysis with an existing configuration.',
        )
        layout = {"width": "15%", "top": "0%"}
        self.hs["File&PathOptn drpdn"].layout = layout

        self.hs["File&PathOptn drpdn"].observe(self.FilePathOptn_drpdn_chg,
                                               names="value")
        self.hs["SelFile&PathCfm box"].children = [
            self.hs["SelFile&PathCfm text"],
            self.hs["SelFile&PathCfm btn"],
            self.hs["File&PathOptn drpdn"],
        ]

        self.hs["SelFile&Path box"].children = [
            self.hs["SelFile&PathTitle box"],
            self.hs["SelRaw box"],
            self.hs["SelRecon box"],
            self.hs["SelSavTrial box"],
            self.hs["SelFile&PathCfm box"],
        ]
        ## ## ## ## bin widgets in hs['SelFile&Path box'] -- configure file settings -- end

        ## ## ## ## define functional widgets each tab in each sub-tab  - define indices -- start
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.55*(self.form_sz[0]-136)}px",
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

        ## ## ## ## ## label define indices box
        layout = {
            "justify-content": "center",
            "align-items": "center",
            "align-contents": "center",
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto",
        }
        self.hs["ConfigDataTitle box"] = widgets.HBox()
        self.hs["ConfigDataTitle box"].layout = layout
        self.hs["ConfigDataTitle label"] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + "Config Scan & Slice Indices" + "</span>")
        layout = {"left": "35%", "background-color": "white", "color": "cyan"}
        self.hs["ConfigDataTitle label"].layout = layout
        self.hs["ConfigDataTitle box"].children = [
            self.hs["ConfigDataTitle label"]
        ]

        ## ## ## ## ## scan id range
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["ScanIdRange box"] = widgets.HBox()
        self.hs["ScanIdRange box"].layout = layout
        self.hs["SelScanIdStart drpdn"] = widgets.Dropdown(
            value=0, options=[0], description="scan_id start", disabled=True)
        layout = {"width": "40%"}
        self.hs["SelScanIdStart drpdn"].layout = layout
        self.hs["SelScanIdEnd drpdn"] = widgets.Dropdown(
            value=0, options=[0], description="scan_id end", disabled=True)
        layout = {"width": "40%"}
        self.hs["SelScanIdEnd drpdn"].layout = layout

        self.hs["SelScanIdStart drpdn"].observe(self.SelScanIdStart_drpdn_chg,
                                                names="value")
        self.hs["SelScanIdEnd drpdn"].observe(self.SelScanIdEnd_drpdn_chg,
                                              names="value")
        self.hs["ScanIdRange box"].children = [
            self.hs["SelScanIdStart drpdn"],
            self.hs["SelScanIdEnd drpdn"],
        ]

        ## ## ## ## ## fixed scan and slice ids
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["FixedScanId box"] = widgets.HBox()
        self.hs["FixedScanId box"].layout = layout
        self.hs["FixedScanId drpdn"] = widgets.Dropdown(
            value=0, options=[0], description="fixed scan id", disabled=True)
        layout = {"width": "40%"}
        self.hs["FixedScanId drpdn"].layout = layout
        self.hs["FixedSliId sldr"] = widgets.IntSlider(
            value=0, description="fixed sli id", disabled=True)
        layout = {"width": "40%"}
        self.hs["FixedSliId sldr"].layout = layout

        self.hs["FixedScanId drpdn"].observe(self.FixedScanId_drpdn_chg,
                                             names="value")
        self.hs["FixedSliId sldr"].observe(self.FixedSliId_sldr_chg,
                                           names="value")
        self.hs["FixedScanId box"].children = [
            self.hs["FixedScanId drpdn"],
            self.hs["FixedSliId sldr"],
        ]

        ## ## ## ## ## fiji option
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["Fiji box"] = widgets.HBox()
        self.hs["Fiji box"].layout = layout
        self.hs["FijiRawImgPrev chbx"] = widgets.Checkbox(
            value=False, description="fiji view", disabled=True)
        layout = {"width": "40%"}
        self.hs["FijiRawImgPrev chbx"].layout = layout
        self.hs["FijiClose btn"] = widgets.Button(
            description="close all fiji viewers", disabled=True)
        layout = {"width": "40%"}
        self.hs["FijiClose btn"].layout = layout

        self.hs["FijiRawImgPrev chbx"].observe(self.FijiRawImgPrev_chbx_chg,
                                               names="value")
        self.hs["FijiClose btn"].on_click(self.FijiClose_btn_clk)
        self.hs["Fiji box"].children = [
            self.hs["FijiRawImgPrev chbx"],
            self.hs["FijiClose btn"],
        ]

        ## ## ## ## ## confirm indices configuration
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["ConfigDataCfm box"] = widgets.HBox()
        self.hs["ConfigDataCfm box"].layout = layout
        self.hs["ConfigDataCfm text"] = widgets.Text(
            value="Confirm setting once you are done ...",
            description="",
            disabled=True)
        layout = {"width": "66%"}
        self.hs["ConfigDataCfm text"].layout = layout
        self.hs["ConfigDataCfm btn"] = widgets.Button(
            description="Confirm",
            disabled=True,
            description_tooltip=
            "Confirm: Confirm after you finish file configuration",
        )
        self.hs["ConfigDataCfm btn"].style.button_color = "darkviolet"
        layout = {"width": "15%"}
        self.hs["ConfigDataCfm btn"].layout = layout

        self.hs["ConfigDataCfm btn"].on_click(self.ConfigDataCfm_btn_clk)
        self.hs["ConfigDataCfm box"].children = [
            self.hs["ConfigDataCfm text"],
            self.hs["ConfigDataCfm btn"],
        ]

        self.hs["ConfigData box"].children = [
            self.hs["ConfigDataTitle box"],
            self.hs["ScanIdRange box"],
            self.hs["FixedScanId box"],
            self.hs["Fiji box"],
            self.hs["ConfigDataCfm box"],
        ]
        ## ## ## ## bin widgets in hs['ConfigData box'] -- end

        ## ## ## ## define widgets in hs['MetaData box'] -- start
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
        ## ## ## ## bundle widgets in hs['MetaData box'] -- end

        self.hs["DataConfig&Info tab"].children = [
            self.hs["ConfigData box"],
            self.hs["MetaData box"],
        ]
        self.hs["DataConfig&Info tab"].set_title(0, "Config Data")
        self.hs["DataConfig&Info tab"].set_title(1, "Data Info")
        ## ## ## ## bundle widgets in hs['DataConfig&Info tab'] -- end
        self.hs["Config&Input form"].children = [
            self.hs["SelFile&Path box"],
            self.hs["DataConfig&Info tab"],
        ]

        ## ## ## bundle boxes in hs['Config&Input form'] -- end

        ## ## ## ## define functional widgets each tab in each sub-tab - config roi -- start
        base_wz_os = 92
        ex_ws_os = 6
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.35*(self.form_sz[0]-136)}px",
        }
        self.hs["3DRoi box"] = widgets.VBox()
        self.hs["3DRoi box"].layout = layout

        ## ## ## ## ## label 3D_roi_title box
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["3DRoiTitle box"] = widgets.HBox()
        self.hs["3DRoiTitle box"].layout = layout
        self.hs["3DRoiTitle text"] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + "Config 3D ROI" + "</span>")
        layout = {
            "justify-content": "center",
            "background-color": "white",
            "color": "cyan",
            "left": "43%",
        }
        self.hs["3DRoiTitle text"].layout = layout
        self.hs["3DRoiTitle box"].children = [self.hs["3DRoiTitle text"]]

        ## ## ## ## ## define roi
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": f"{0.21*(self.form_sz[0]-136)}px",
        }
        self.hs["3DRoiDefine box"] = widgets.VBox()
        self.hs["3DRoiDefine box"].layout = layout
        layout = {"border": "3px solid #FFCC00", "width": "auto"}
        self.hs["3DRoiX sldr"] = widgets.IntRangeSlider(
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
        self.hs["3DRoiX sldr"].layout = layout
        self.hs["3DRoiY sldr"] = widgets.IntRangeSlider(
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
        self.hs["3DRoiY sldr"].layout = layout
        self.hs["3DRoiZ sldr"] = widgets.IntRangeSlider(
            value=[10, 40],
            min=1,
            max=50,
            step=1,
            description="z range:",
            disabled=True,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        self.hs["3DRoiZ sldr"].layout = layout
        self.hs["3DRoiZ sldr"].add_traits(
            mylower=traitlets.traitlets.Any(self.hs["3DRoiZ sldr"].lower))
        self.hs["3DRoiZ sldr"].add_traits(
            myupper=traitlets.traitlets.Any(self.hs["3DRoiZ sldr"].upper))

        self.hs["3DRoiX sldr"].observe(self.RoiX3D_sldr_chg, names="value")
        self.hs["3DRoiY sldr"].observe(self.RoiY3D_sldr_chg, names="value")
        self.hs["3DRoiZ sldr"].observe(self.RoiZ3D_val_sldr_chg, names="value")
        self.hs["3DRoiZ sldr"].observe(self.RoiZ3D_lwr_sldr_chg,
                                       names="mylower")
        self.hs["3DRoiZ sldr"].observe(self.RoiZ3D_upr_sldr_chg,
                                       names="myupper")
        self.hs["3DRoiDefine box"].children = [
            self.hs["3DRoiX sldr"],
            self.hs["3DRoiY sldr"],
            self.hs["3DRoiZ sldr"],
        ]

        ## ## ## ## ## confirm roi
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["3DRoiCfm box"] = widgets.HBox()
        self.hs["3DRoiCfm box"].layout = layout
        layout = {"width": "70%"}
        self.hs["3DRoiCfm text"] = widgets.Text(
            description="",
            value="Please confirm after ROI is set ...",
            disabled=True)
        self.hs["3DRoiCfm text"].layout = layout
        layout = {"width": "15%"}
        self.hs["3DRoiCfm btn"] = widgets.Button(
            description="Confirm",
            disabled=True,
            description_tooltip="Confirm the roi once you define the ROI ...",
        )
        self.hs["3DRoiCfm btn"].style.button_color = "darkviolet"
        self.hs["3DRoiCfm btn"].layout = layout

        self.hs["3DRoiCfm btn"].on_click(self.Roi3DCfm_btn_clk)
        self.hs["3DRoiCfm box"].children = [
            self.hs["3DRoiCfm text"],
            self.hs["3DRoiCfm btn"],
        ]

        self.hs["3DRoi box"].children = [
            self.hs["3DRoiTitle box"],
            self.hs["3DRoiDefine box"],
            self.hs["3DRoiCfm box"],
        ]
        ## ## ## ## define functional widgets each tab in each sub-tab - config roi -- end

        ## ## ## ## define functional widgets each tab in each sub-tab - config registration -- start
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.42*(self.form_sz[0]-136)}px",
        }
        self.hs["ConfigRegParams box"] = widgets.VBox()
        self.hs["ConfigRegParams box"].layout = layout

        ## ## ## ## ## label config_reg_params box
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

        ## ## ## ## ## fiji&anchor box
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
        self.hs["ChunkSz chbx"] = widgets.Checkbox(value=True,
                                                   disabled=True,
                                                   description="use chunk")
        layout = {"width": "19%"}
        self.hs["ChunkSz chbx"].layout = layout
        self.hs["ChunkSz sldr"] = widgets.IntSlider(value=7,
                                                    disabled=True,
                                                    description="chunk size")
        layout = {"width": "29%"}
        self.hs["ChunkSz sldr"].layout = layout
        self.hs["SliSrch sldr"] = widgets.IntSlider(
            value=10, disabled=True, description="z search half width")
        layout = {"width": "29%"}
        self.hs["SliSrch sldr"].layout = layout

        self.hs["FijiMaskViewer chbx"].observe(self.FijiMaskViewer_chbx_chg,
                                               names="value")
        self.hs["ChunkSz chbx"].observe(self.ChunkSz_chbx_chg, names="value")
        self.hs["ChunkSz sldr"].observe(self.ChunkSz_sldr_chg, names="value")
        self.hs["SliSrch sldr"].observe(self.SliSrch_sldr_chg, names="value")
        self.hs["Fiji&Anchor box"].children = [
            self.hs["FijiMaskViewer chbx"],
            self.hs["ChunkSz chbx"],
            self.hs["ChunkSz sldr"],
            self.hs["SliSrch sldr"],
        ]

        ## ## ## ## ## mask options box
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["MaskOptn box"] = widgets.HBox()
        self.hs["MaskOptn box"].layout = layout
        self.hs["UseMask chbx"] = widgets.Checkbox(
            value=False,
            disabled=True,
            description="use mask",
            display="flex",
            indent=False,
        )
        layout = {"width": "15%", "flex-direction": "row"}
        self.hs["UseMask chbx"].layout = layout
        self.hs["MaskThres sldr"] = widgets.FloatSlider(
            value=False,
            disabled=True,
            description="mask thres",
            readout_format=".5f",
            min=-1.0,
            max=10.0,
            step=1e-5,
        )
        layout = {"width": "40%", "left": "2.5%"}
        self.hs["MaskThres sldr"].layout = layout
        self.hs["MaskDilation sldr"] = widgets.IntSlider(
            value=False,
            disabled=True,
            description="mask dilation",
            min=0,
            max=30,
            step=1,
        )
        layout = {"width": "40%", "left": "2.5%"}
        self.hs["MaskDilation sldr"].layout = layout

        self.hs["UseMask chbx"].observe(self.UseMask_chbx_chg, names="value")
        self.hs["MaskThres sldr"].observe(self.MaskThres_sldr_chg,
                                          names="value")
        self.hs["MaskDilation sldr"].observe(self.MaskDilation_sldr_chg,
                                             names="value")
        self.hs["MaskOptn box"].children = [
            self.hs["UseMask chbx"],
            self.hs["MaskThres sldr"],
            self.hs["MaskDilation sldr"],
        ]

        ## ## ## ## ## sli_search & chunk_size box
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["RegOptn box"] = widgets.HBox()
        self.hs["RegOptn box"].layout = layout
        self.hs["RegMethod drpdn"] = widgets.Dropdown(
            value="MRTV",
            description="reg method",
            disabled=True,
            options=["MRTV", "MPC", "PC", "SR", "LS_MRTV", "MPC_MRTV"],
            description_tooltip=
            "reg method: MPC: Masked Phase Correlation; PC: Phase Correlation; SR: StackReg",
        )
        layout = {"width": "19%"}
        self.hs["RegMethod drpdn"].layout = layout
        self.hs["RefMode drpdn"] = widgets.Dropdown(
            value="single",
            options=["single", "neighbor", "average"],
            description="ref mode",
            disabled=True,
            description_tooltip=
            "ref mode: single: two images with fixed id gap in two consecutive chunks are used for the registration between two chunks; neighbor: two neighbor images in two consecutive chunks are used for the registration between two chunks; average: deprecated",
        )
        layout = {"width": "19%"}
        self.hs["RefMode drpdn"].layout = layout

        self.hs["RegMethod drpdn"].observe(self.RegMethod_drpdn_chg,
                                           names="value")
        self.hs["RefMode drpdn"].observe(self.RefMode_drpdn_chg, names="value")
        self.hs["RegOptn box"].children = [
            self.hs["RegMethod drpdn"],
            self.hs["RefMode drpdn"],
        ]

        ## ## ## ## ##  reg_options box
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["MRTVOptn box"] = widgets.HBox()
        self.hs["MRTVOptn box"].layout = layout

        self.hs["MRTVLevel text"] = widgets.BoundedIntText(
            value=5,
            min=1,
            max=10,
            step=1,
            description="level",
            description_tooltip="level: multi-resolution level",
            disabled=True,
        )
        layout = {"width": "19%"}
        self.hs["MRTVLevel text"].layout = layout

        self.hs["MRTVWz text"] = widgets.BoundedIntText(
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
        self.hs["MRTVWz text"].layout = layout

        self.hs["MRTVSubpixelSrch drpdn"] = widgets.Dropdown(
            value="analytical",
            disabled=True,
            options=["analytical", "fitting"],
            description="subpxl srch",
            description_tooltip="subpxl srch: subpixel TV minization option",
        )
        layout = {"width": "19%"}
        self.hs["MRTVSubpixelSrch drpdn"].layout = layout

        self.hs["MRTVSubpixelWz text"] = widgets.BoundedIntText(
            value=3,
            min=2,
            max=20,
            step=0.1,
            description="subpxl wz",
            disabled=True,
            description_tooltip="subpxl wz: final sub-pixel fitting points",
        )
        layout = {"width": "19%"}
        self.hs["MRTVSubpixelWz text"].layout = layout

        self.hs["MRTVSubpixelKernel text"] = widgets.BoundedIntText(
            value=3,
            min=2,
            max=20,
            step=1,
            description="kernel wz",
            disabled=True,
            description_tooltip=
            "kernel wz: Gaussian blurring width before TV minimization",
        )
        layout = {"width": "19%"}
        self.hs["MRTVSubpixelKernel text"].layout = layout

        self.hs["MRTVLevel text"].observe(self.MRTVLevel_text_chg,
                                          names="value")
        self.hs["MRTVWz text"].observe(self.MRTVWz_text_chg, names="value")
        self.hs["MRTVSubpixelWz text"].observe(self.MRTVSubpixelWz_text_chg,
                                               names="value")
        self.hs["MRTVSubpixelKernel text"].observe(
            self.MRTVSubpixelKernel_text_chg, names="value")
        self.hs["MRTVSubpixelSrch drpdn"].observe(
            self.MRTVSubpixelSrch_drpdn_chg, names="value")
        self.hs["MRTVOptn box"].children = [
            self.hs["MRTVLevel text"],
            self.hs["MRTVWz text"],
            self.hs["MRTVSubpixelSrch drpdn"],
            self.hs["MRTVSubpixelWz text"],
            self.hs["MRTVSubpixelKernel text"],
        ]

        ## ## ## ## ## confirm reg settings -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["ConfigRegParamsCfm box"] = widgets.HBox()
        self.hs["ConfigRegParamsCfm box"].layout = layout
        layout = {"width": "70%"}
        self.hs["ConfigRegParamsCfm text"] = widgets.Text(
            description="",
            disabled=True,
            value="Confirm the roi once you define the ROI ...",
        )
        self.hs["ConfigRegParamsCfm text"].layout = layout
        layout = {"width": "15%"}
        self.hs["ConfigRegParamsCfm btn"] = widgets.Button(
            description="Confirm",
            disabled=True,
            description_tooltip="Confirm the roi once you define the ROI ...",
        )
        self.hs["ConfigRegParamsCfm btn"].style.button_color = "darkviolet"
        self.hs["ConfigRegParamsCfm btn"].layout = layout

        self.hs["ConfigRegParamsCfm btn"].on_click(
            self.ConfigRegParamsCfm_btn_clk)
        self.hs["ConfigRegParamsCfm box"].children = [
            self.hs["ConfigRegParamsCfm text"],
            self.hs["ConfigRegParamsCfm btn"],
        ]
        ## ## ## ## ## confirm reg settings -- end
        self.hs["ConfigRegParams box"].children = [
            self.hs["ConfigRegParamsTitle box"],
            self.hs["Fiji&Anchor box"],
            self.hs["MaskOptn box"],
            self.hs["RegOptn box"],
            self.hs["MRTVOptn box"],
            self.hs["ConfigRegParamsCfm box"],
        ]
        ## ## ## ## define functional widgets each tab in each sub-tab - config registration -- end

        ## ## ## ## define functional widgets each tab in each sub-tab - register in register/review/shift TAB -- start
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.21*(self.form_sz[0]-136)}px",
        }
        self.hs["RunReg box"] = widgets.VBox()
        self.hs["RunReg box"].layout = layout

        ## ## ## ## ## label run_reg box
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

        ## ## ## ## ## run reg & status
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["RunRegCfm box"] = widgets.HBox()
        self.hs["RunRegCfm box"].layout = layout
        layout = {"width": "70%"}
        self.hs["RunRegCfm text"] = widgets.Text(
            description="",
            disabled=True,
            value="run registration once you are ready ...",
        )
        self.hs["RunRegCfm text"].layout = layout
        layout = {"width": "15%"}
        self.hs["RunRegCfm btn"] = widgets.Button(
            description="Run Reg",
            disabled=True,
            description_tooltip="run registration once you are ready ...",
        )
        self.hs["RunRegCfm btn"].style.button_color = "darkviolet"
        self.hs["RunRegCfm btn"].layout = layout

        self.hs["RunRegCfm btn"].on_click(self.RunRegCfm_btn_clk)
        self.hs["RunRegCfm box"].children = [
            self.hs["RunRegCfm text"],
            self.hs["RunRegCfm btn"],
        ]

        ## ## ## ## ## run reg progress
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["RunRegPrgr box"] = widgets.HBox()
        self.hs["RunRegPrgr box"].layout = layout
        layout = {"width": "100%"}
        self.hs["RunRegPrgr bar"] = widgets.IntProgress(
            value=0,
            min=0,
            max=10,
            step=1,
            description="Completing:",
            bar_style="info",  # 'success', 'info', 'warning', 'danger' or ''
            orientation="horizontal",
        )
        self.hs["RunRegPrgr bar"].layout = layout
        self.hs["RunRegPrgr box"].children = [self.hs["RunRegPrgr bar"]]

        self.hs["RunReg box"].children = [
            self.hs["RunRegTitle box"],
            self.hs["RunRegCfm box"],
            self.hs["RunRegPrgr box"],
        ]
        ## ## ## ## define functional widgets each tab in each sub-tab - register in register/review/shift TAB -- end

        self.hs["RegSetting form"].children = [
            self.hs["3DRoi box"],
            self.hs["ConfigRegParams box"],
            self.hs["RunReg box"],
        ]
        ## ## ## bin boxes in hs['RegSetting form'] -- end

        ## ## ## ## define functional widgets each tab in each sub-tab - review in in register/review/shift TAB -- start
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.35*(self.form_sz[0]-136)}px",
        }
        self.hs["RevRegRlt box"] = widgets.VBox()
        self.hs["RevRegRlt box"].layout = layout

        ## ## ## ## ## label the box
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["RevRegRltTitle box"] = widgets.HBox()
        self.hs["RevRegRltTitle box"].layout = layout
        self.hs["RevRegRltTitle text"] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + "Review Registration Results" + "</span>")
        layout = {
            "background-color": "white",
            "color": "cyan",
            "left": "35.7%"
        }
        self.hs["RevRegRltTitle text"].layout = layout
        self.hs["RevRegRltTitle box"].children = [
            self.hs["RevRegRltTitle text"]
        ]

        ## ## ## ## ## read alignment file
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["ReadAlignFile box"] = widgets.HBox()
        self.hs["ReadAlignFile box"].layout = layout
        layout = {"width": "85%"}
        self.hs["ReadAlign chbx"] = widgets.Checkbox(
            description="read alignment", value=False, disabled=True)
        self.hs["ReadAlign chbx"].layout = layout
        layout = {"width": "15%"}
        self.hs["ReadAlign btn"] = SelectFilesButton(option="askopenfilename")
        self.hs["ReadAlign btn"].layout = layout
        self.hs["ReadAlign btn"].disabled = True

        self.hs["ReadAlign chbx"].observe(self.ReadAlign_chbx_chg,
                                          names="value")
        self.hs["ReadAlign btn"].on_click(self.ReadAlign_btn_clk)
        self.hs["ReadAlignFile box"].children = [
            self.hs["ReadAlign chbx"],
            self.hs["ReadAlign btn"],
        ]

        ## ## ## ## ## reg pair box
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["RegPair box"] = widgets.HBox()
        self.hs["RegPair box"].layout = layout
        layout = {"width": "85%"}
        self.hs["RegPair sldr"] = widgets.IntSlider(value=False,
                                                    disabled=True,
                                                    description="reg pair #")
        self.hs["RegPair sldr"].layout = layout

        layout = {"width": "15%"}
        self.hs["RegPairBad btn"] = widgets.Button(
            description="Bad", description_tooltip="Bad reg", disabled=True)
        self.hs["RegPairBad btn"].layout = layout
        self.hs["RegPairBad btn"].style.button_color = "darkviolet"

        self.hs["RegPair sldr"].observe(self.RegPair_sldr_chg, names="value")
        self.hs["RegPairBad btn"].on_click(self.RegPairBad_btn_clk)
        self.hs["RegPair box"].children = [
            self.hs["RegPair sldr"],
            self.hs["RegPairBad btn"],
        ]

        ## ## ## ## ## manual shift box -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["CorrShft box"] = widgets.HBox()
        self.hs["CorrShft box"].layout = layout
        layout = {"width": "28%"}
        self.hs["CorrXShift text"] = widgets.FloatText(value=0,
                                                       disabled=True,
                                                       min=-100,
                                                       max=100,
                                                       step=0.5,
                                                       description="x shift")
        self.hs["CorrXShift text"].layout = layout
        layout = {"width": "28%"}
        self.hs["CorrYShift text"] = widgets.FloatText(value=0,
                                                       disabled=True,
                                                       min=-100,
                                                       max=100,
                                                       step=0.5,
                                                       description="y shift")
        self.hs["CorrYShift text"].layout = layout
        layout = {"width": "27.5%"}
        self.hs["CorrZShift text"] = widgets.IntText(value=0,
                                                     disabled=True,
                                                     min=1,
                                                     max=100,
                                                     step=1,
                                                     description="z shift")
        self.hs["CorrZShift text"].layout = layout
        layout = {"width": "15%"}
        self.hs["CorrShftRecord btn"] = widgets.Button(disabled=True,
                                                       description="Record")
        self.hs["CorrShftRecord btn"].layout = layout
        self.hs["CorrShftRecord btn"].style.button_color = "darkviolet"

        self.hs["CorrXShift text"].observe(self.CorrXShift_text_chg,
                                           names="value")
        self.hs["CorrYShift text"].observe(self.CorrYShift_text_chg,
                                           names="value")
        self.hs["CorrZShift text"].observe(self.CorrZShift_text_chg,
                                           names="value")
        self.hs["CorrShftRecord btn"].on_click(self.CorrShftRecord_btn_clk)
        self.hs["CorrShft box"].children = [
            self.hs["CorrXShift text"],
            self.hs["CorrYShift text"],
            self.hs["CorrZShift text"],
            self.hs["CorrShftRecord btn"],
        ]
        ## ## ## ## ## manual shift box -- end

        ## ## ## ## ## confirm review results box
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["RevRegRltCfm box"] = widgets.HBox()
        self.hs["RevRegRltCfm box"].layout = layout
        layout = {"width": "85%", "display": "inline_flex"}
        self.hs["RevRegRltCfm text"] = widgets.Text(
            description="",
            disabled=True,
            value="Confirm after you finish reg review ...",
        )
        self.hs["RevRegRltCfm text"].layout = layout
        layout = {"width": "15%"}
        self.hs["RevRegRltCfm btn"] = widgets.Button(
            description="Confirm",
            disabled=True,
            description_tooltip="Confirm after you finish reg review ...",
        )
        self.hs["RevRegRltCfm btn"].style.button_color = "darkviolet"
        self.hs["RevRegRltCfm btn"].layout = layout

        self.hs["RevRegRltCfm btn"].on_click(self.RevRegRltCfm_btn_clk)
        self.hs["RevRegRltCfm box"].children = [
            self.hs["RevRegRltCfm text"],
            self.hs["RevRegRltCfm btn"],
        ]

        self.hs["RevRegRlt box"].children = [
            self.hs["RevRegRltTitle box"],
            self.hs["ReadAlignFile box"],
            self.hs["RegPair box"],
            self.hs["CorrShft box"],
            self.hs["RevRegRltCfm box"],
        ]
        ## ## ## ## define functional widgets each tab in each sub-tab - review in in register/review/shift TAB-- end

        ## ## ## ## define functional widgets each tab in each sub-tab - align recon in register/review/shift TAB -- start
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.28*(self.form_sz[0]-136)}px",
        }
        self.hs["AlignRecon box"] = widgets.VBox()
        self.hs["AlignRecon box"].layout = layout

        ## ## ## ## ## label the box
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["AlignReconTitle box"] = widgets.HBox()
        self.hs["AlignReconTitle box"].layout = layout
        self.hs["AlignReconTitle text"] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + "Align 3D Recon" + "</span>")
        layout = {"background-color": "white", "color": "cyan", "left": "41%"}
        self.hs["AlignReconTitle text"].layout = layout
        self.hs["AlignReconTitle box"].children = [
            self.hs["AlignReconTitle text"]
        ]

        ## ## ## ## ## define slice region if it is necessary
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["AlignReconOptnSliRegion box"] = widgets.HBox()
        self.hs["AlignReconOptnSliRegion box"].layout = layout
        layout = {"width": "20%"}
        self.hs["AlignReconOptnSli chbx"] = widgets.Checkbox(
            description="new z range",
            value=False,
            disabled=True,
            description_tooltip=
            "check this on if you like to adjust z slice range for alignment",
        )
        self.hs["AlignReconOptnSli chbx"].layout = layout
        layout = {"width": "10%"}
        self.hs["AlignReconOptnSliStart text"] = widgets.BoundedIntText(
            description="",
            value=0,
            min=0,
            max=10,
            disabled=True,
            description_tooltip=
            "In the case of reading and reviewing a registration file, you need to define slice start and end.",
        )
        self.hs["AlignReconOptnSliStart text"].layout = layout
        layout = {"width": "60%"}
        self.hs["AlignReconOptnSliRange sldr"] = widgets.IntRangeSlider(
            description="z range",
            value=[0, 1],
            min=0,
            max=10,
            disabled=True,
            description_tooltip=
            "In the case of reading and reviewing a registration file, you need to define slice start and end.",
        )
        self.hs["AlignReconOptnSliRange sldr"].layout = layout
        layout = {"width": "10%"}
        self.hs["AlignReconOptnSliEnd text"] = widgets.BoundedIntText(
            description="",
            value=0,
            min=0,
            max=10,
            disabled=True,
            description_tooltip=
            "In the case of reading and reviewing a registration file, you need to define slice start and end.",
        )
        self.hs["AlignReconOptnSliEnd text"].layout = layout

        self.hs["AlignReconOptnSli chbx"].observe(
            self.AlignReconOptnSli_chbx_chg, names="value")
        self.hs["AlignReconOptnSliStart text"].observe(
            self.AlignReconOptnSliStart_text_chg, names="value")
        self.hs["AlignReconOptnSliEnd text"].observe(
            self.AlignReconOptnSliEnd_text_chg, names="value")
        self.hs["AlignReconOptnSliRange sldr"].observe(
            self.AlignReconOptnSliRange_sldr_chg, names="value")
        self.hs["AlignReconOptnSliRegion box"].children = [
            self.hs["AlignReconOptnSli chbx"],
            self.hs["AlignReconOptnSliStart text"],
            self.hs["AlignReconOptnSliRange sldr"],
            self.hs["AlignReconOptnSliEnd text"],
        ]

        ## ## ## ## ## run reg & status
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["AlignReconCfm box"] = widgets.HBox()
        self.hs["AlignReconCfm box"].layout = layout
        layout = {"width": "85%"}
        self.hs["AlignReconCfm text"] = widgets.Text(
            description="",
            disabled=True,
            value="Confirm to proceed alignment ...")
        self.hs["AlignReconCfm text"].layout = layout
        layout = {"width": "15%"}
        self.hs["AlignReconCfm btn"] = widgets.Button(
            description="Align",
            disabled=True,
            description_tooltip=
            "This will perform xanes3D alignment according to your configurations ...",
        )
        self.hs["AlignReconCfm btn"].style.button_color = "darkviolet"
        self.hs["AlignReconCfm btn"].layout = layout

        self.hs["AlignReconCfm btn"].on_click(self.AlignReconCfm_btn_clk)
        self.hs["AlignReconCfm box"].children = [
            self.hs["AlignReconCfm text"],
            self.hs["AlignReconCfm btn"],
        ]

        ## ## ## ## ## run reg progress
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["AlignReconPrgr box"] = widgets.HBox()
        self.hs["AlignReconPrgr box"].layout = layout
        layout = {"width": "100%"}
        self.hs["AlignReconPrgr bar"] = widgets.IntProgress(
            value=0,
            min=0,
            max=10,
            step=1,
            description="Completing:",
            orientation="horizontal",
            bar_style="info",
        )  # 'success', 'info', 'warning', 'danger' or ''
        self.hs["AlignReconPrgr bar"].layout = layout

        self.hs["AlignReconPrgr box"].children = [
            self.hs["AlignReconPrgr bar"]
        ]

        self.hs["AlignRecon box"].children = [
            self.hs["AlignReconTitle box"],
            self.hs["AlignReconOptnSliRegion box"],
            self.hs["AlignReconCfm box"],
            self.hs["AlignReconPrgr box"],
        ]
        ## ## ## ## define functional widgets each tab in each sub-tab - align recon in register/review/shift TAB -- end

        ## ## ## ## define functional widgets each tab in each sub-tab - visualziation box in analysis&display TAB -- start
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.28*(self.form_sz[0]-136)}px",
        }
        self.hs["VisImg box"] = widgets.VBox()
        self.hs["VisImg box"].layout = layout

        ## ## ## ## ## label visualize box -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["VisImgTitle box"] = widgets.HBox()
        self.hs["VisImgTitle box"].layout = layout
        self.hs["VisImgTitle text"] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + "Visualize XANES3D" + "</span>")
        layout = {"background-color": "white", "color": "cyan", "left": "38%"}
        self.hs["VisImgTitle text"].layout = layout

        self.hs["VisImgTitle box"].children = [self.hs["VisImgTitle text"]]
        ## ## ## ## ## label visualize box -- end

        ## ## ## ## ## visualization option box -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["VisImgViewerOptn box"] = widgets.HBox()
        self.hs["VisImgViewerOptn box"].layout = layout
        layout = {"width": "80%"}
        self.hs["VisImgViewerOptn tgbtn"] = widgets.ToggleButtons(
            description="viewer options",
            disabled=True,
            description_tooltip=
            "napari: provides better image preview functions; fiji: provides quick spectrum inspection functions",
            options=["fiji", "napari"],
            value="fiji",
        )
        self.hs["VisImgViewerOptn tgbtn"].layout = layout
        layout = {"width": "20%"}
        self.hs["VisImgViewAlignOptn drpdn"] = widgets.Dropdown(
            description="view option",
            description_tooltip=
            "dimensions are defined as: E: energy dimension; x-y: slice lateral plane; z: dimension normal to slice plane",
            options=["x-y-E", "y-z-E", "z-x-E", "x-y-z"],
            value="x-y-E",
            disabled=True,
        )
        self.hs["VisImgViewAlignOptn drpdn"].layout = layout

        self.hs["VisImgViewerOptn tgbtn"].observe(
            self.VisImgViewerOptn_tgbtn_clk, names="value")
        self.hs["VisImgViewAlignOptn drpdn"].observe(
            self.VisImgViewAlignOptn_drpdn_clk, names="value")
        self.hs["VisImgViewerOptn box"].children = [
            self.hs["VisImgViewerOptn tgbtn"],
            self.hs["VisImgViewAlignOptn drpdn"],
        ]
        ## ## ## ## ## visualization option box -- end

        ## ## ## ## ## define slice region and view slice cuts -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto"
        }
        self.hs["VisImgViewAlign box"] = widgets.HBox()
        self.hs["VisImgViewAlign box"].layout = layout

        layout = {"width": "35%"}
        self.hs["VisImgViewAlign4thDim sldr"] = widgets.IntSlider(
            description="z",
            value=0,
            min=0,
            disabled=True,
            description_tooltip="Select one slice in the fourth dimension",
        )
        self.hs["VisImgViewAlign4thDim sldr"].layout = layout
        layout = {"width": "35%"}
        self.hs["VisImgViewAlignSli sldr"] = widgets.IntSlider(description="E",
                                                               disabled=True)
        self.hs["VisImgViewAlignSli sldr"].layout = layout
        layout = {"width": "20%"}
        self.hs["VisImgViewAlignEng text"] = widgets.FloatText(value=0,
                                                               description="E",
                                                               disabled=True)
        self.hs["VisImgViewAlignEng text"].layout = layout

        self.hs["VisImgViewAlign4thDim sldr"].observe(
            self.VisImgViewAlign4thDim_sldr_chg, names="value")
        self.hs["VisImgViewAlignSli sldr"].observe(
            self.VisImgViewAlignSli_sldr_chg, names="value")
        self.hs["VisImgViewAlign box"].children = [
            self.hs["VisImgViewAlign4thDim sldr"],
            self.hs["VisImgViewAlignSli sldr"],
            self.hs["VisImgViewAlignEng text"],
        ]
        ## ## ## ## ## define slice region and view slice cuts -- end

        ## ## ## ## ## basic spectroscopic visualization -- start
        layout = {"border": "3px solid #FFCC00", "layout": "center"}
        self.hs["VisSpecView box"] = widgets.HBox()
        self.hs["VisSpecView box"].layout = layout
        layout = {"width": "70%"}
        self.hs["VisSpecView text"] = widgets.Text(
            description="",
            disabled=True,
            value="visualize spectrum in roi ...")
        self.hs["VisSpecView text"].layout = layout
        layout = {"width": "15%", "left": "-8%"}
        self.hs["VisSpecViewMemMnt chbx"] = widgets.Checkbox(
            description="mem use",
            value=False,
            disabled=True,
            description_tooltip="Check on this to monitor memmory usage",
        )
        self.hs["VisSpecViewMemMnt chbx"].layout = layout
        layout = {"width": "15%"}
        self.hs["VisSpecViewInRoi btn"] = widgets.Button(
            description="spec in roi",
            disabled=True,
            description_tooltip=
            "adjust the roi size and drag roi over in the particles",
        )
        self.hs["VisSpecViewInRoi btn"].layout = layout
        self.hs["VisSpecViewInRoi btn"].style.button_color = "darkviolet"

        self.hs["VisSpecViewMemMnt chbx"].observe(
            self.VisSpecViewMemMnt_chbx_chg, names="value")
        self.hs["VisSpecViewInRoi btn"].on_click(self.VisSpecViewInRoi_btn_clk)
        self.hs["VisSpecView box"].children = [
            self.hs["VisSpecViewMemMnt chbx"],
            self.hs["VisSpecViewInRoi btn"],
        ]
        ## ## ## ## ## basic spectroscopic visualization -- end

        self.hs["VisImg box"].children = [
            self.hs["VisImgTitle box"],
            self.hs["VisImgViewerOptn box"],
            self.hs["VisImgViewAlign box"],
            self.hs["VisSpecView box"],
        ]
        ## ## ## ## define functional widgets each tab in each sub-tab - visualziation box in analysis&display TAB -- end

        self.hs["Reg&Rev form"].children = [
            self.hs["RevRegRlt box"],
            self.hs["AlignRecon box"],
            self.hs["VisImg box"],
        ]
        ## ## ## bin sub-tabs in each tab - reg&review TAB in 3D_xanes TAB -- end

        self.hs["Fitting form"].children = [
            self.xanes_fit_gui_h.hs["Fitting form"]
        ]
        ## ## ## bin sub-tabs in each tab - analysis&display TAB in 3D_xanes TAB -- end

        self.hs["Analysis form"].children = [
            self.xanes_ana_gui_h.hs["Ana form"]
        ]
        ## ## ## define 2D_XANES_tabs layout - analysis box -- end

        self.hs["SelRawH5Path btn"].initialdir = self.global_h.cwd
        self.hs["SelReconPath btn"].initialdir = self.global_h.cwd
        self.hs["SelSavTrial btn"].initialdir = self.global_h.cwd
        self.hs["ReadAlign btn"].initialdir = self.global_h.cwd

    def SelRawH5Path_btn_clk(self, a):
        if len(a.files[0]) != 0:
            self.xanes_raw_3D_h5_top_dir = os.path.abspath(a.files[0])
            self.hs["SelRawH5Path btn"].initialdir = os.path.abspath(
                a.files[0])
            self.hs["SelReconPath btn"].initialdir = os.path.abspath(
                a.files[0])
            self.hs["SelSavTrial btn"].initialdir = os.path.abspath(a.files[0])
            self.hs["ReadAlign btn"].initialdir = os.path.abspath(a.files[0])
            self.xanes_raw_3D_h5_temp = os.path.join(
                self.xanes_raw_3D_h5_top_dir, self.xanes_raw_fn_temp)
            update_json_content(
                self.global_h.GUI_cfg_file,
                {"cwd": os.path.dirname(os.path.abspath(a.files[0]))},
            )
            self.global_h.cwd = os.path.dirname(os.path.abspath(a.files[0]))
            self.xanes_raw_h5_path_set = True
        else:
            self.hs["SelRawH5Path text"].value = "Choose raw h5 directory ..."
            self.xanes_raw_h5_path_set = False
        self.xanes_file_configured = False
        self.hs[
            "SelFile&PathCfm text"].value = "Please comfirm your change ..."
        self.boxes_logic()

    def SelReconPath_btn_clk(self, a):
        if not self.xanes_raw_h5_path_set:
            self.hs[
                "SelFile&PathCfm text"].value = "You need to specify raw h5 top directory first ..."
            self.hs[
                "SelReconPath text"].value = "Choose recon top directory ..."
            self.xanes_recon_path_set = False
            self.xanes_file_configured = False
        else:
            if len(a.files[0]) != 0:
                self.xanes_recon_3D_top_dir = os.path.abspath(a.files[0])
                self.xanes_recon_3D_dir_temp = os.path.join(
                    self.xanes_recon_3D_top_dir, self.xanes_recon_dir_temp)
                self.xanes_recon_3D_tiff_temp = os.path.join(
                    self.xanes_recon_3D_top_dir,
                    self.xanes_recon_dir_temp,
                    self.xanes_recon_fn_temp,
                )

                self.hs["SelReconPath btn"].initialdir = os.path.abspath(
                    a.files[0])
                update_json_content(self.global_h.GUI_cfg_file,
                                    {"cwd": os.path.abspath(a.files[0])})
                self.global_h.cwd = os.path.os.path.abspath(a.files[0])
                self.xanes_recon_path_set = True
            else:
                self.hs[
                    "SelReconPath text"].value = "Choose recon top directory ..."
                self.xanes_recon_path_set = False
            self.xanes_file_configured = False
            self.hs[
                "SelFile&PathCfm text"].value = "Please comfirm your change ..."
        self.boxes_logic()

    def SelSavTrial_btn_clk(self, a):
        if self.xanes_fit_option == "Do New Reg":
            if len(a.files[0]) != 0:
                b = ""
                t = time.strptime(time.asctime())
                for ii in range(6):
                    b += str(t[ii]).zfill(2) + "-"
                self.xanes_save_trial_reg_filename_template = (
                    os.path.basename(os.path.abspath(a.files[0])).split(".")[0]
                    + f'_{self.xanes_raw_fn_temp.split("_")[1]}' +
                    "_id_{0}-{1}_" + b.strip("-") + ".h5")
                self.xanes_save_trial_reg_config_filename_template = (
                    os.path.basename(os.path.abspath(a.files[0])).split(".")[0]
                    + f'_{self.xanes_raw_fn_temp.split("_")[1]}' +
                    "_id_{0}-{1}_config_" + b.strip("-") + ".json")
                self.hs["SelSavTrial btn"].initialdir = os.path.dirname(
                    os.path.abspath(a.files[0]))
                self.hs["SelSavTrial btn"].initialfile = os.path.basename(
                    a.files[0])
                self.hs["ReadAlign btn"].initialdir = os.path.dirname(
                    os.path.abspath(a.files[0]))
                update_json_content(
                    self.global_h.GUI_cfg_file,
                    {"cwd": os.path.dirname(os.path.abspath(a.files[0]))},
                )
                self.global_h.cwd = os.path.dirname(os.path.abspath(
                    a.files[0]))
                self.xanes_save_trial_set = True
                self.xanes_reg_file_set = False
                self.xanes_config_file_set = False
            else:
                self.hs[
                    "SelSavTrial text"].value = "Save trial registration as ..."
                self.xanes_save_trial_set = False
                self.xanes_reg_file_set = False
                self.xanes_config_file_set = False
            self.xanes_file_configured = False
            self.hs[
                "SelFile&PathCfm text"].value = "Please comfirm your change ..."
        elif self.xanes_fit_option == "Read Config File":
            if len(a.files[0]) != 0:
                self.xanes_save_trial_reg_config_filename_original = os.path.abspath(
                    a.files[0])
                b = ""
                t = time.strptime(time.asctime())
                for ii in range(6):
                    b += str(t[ii]).zfill(2) + "-"
                self.xanes_save_trial_reg_config_filename = (
                    os.path.abspath(a.files[0]).split("config")[0] +
                    "config_" + b.strip("-") + ".json")
                self.hs["SelSavTrial btn"].initialdir = os.path.dirname(
                    os.path.abspath(a.files[0]))
                self.hs["SelSavTrial btn"].initialfile = os.path.basename(
                    a.files[0])
                self.hs["ReadAlign btn"].initialdir = os.path.dirname(
                    os.path.abspath(a.files[0]))
                update_json_content(
                    self.global_h.GUI_cfg_file,
                    {"cwd": os.path.dirname(os.path.abspath(a.files[0]))},
                )
                self.xanes_save_trial_set = False
                self.xanes_reg_file_set = False
                self.xanes_config_file_set = True
            else:
                self.hs[
                    "SelSavTrial text"].value = "Save Existing Configuration File ..."
                self.xanes_save_trial_set = False
                self.xanes_reg_file_set = False
                self.xanes_config_file_set = False
            self.xanes_file_configured = False
            self.hs[
                "SelFile&PathCfm text"].value = "Please comfirm your change ..."
        elif self.xanes_fit_option == "Reg By Shift":
            if len(a.files[0]) != 0:
                self.xanes_save_trial_reg_config_filename_original = os.path.abspath(
                    a.files[0])
                b = ""
                t = time.strptime(time.asctime())
                for ii in range(6):
                    b += str(t[ii]).zfill(2) + "-"
                self.xanes_save_trial_reg_config_filename = (
                    os.path.abspath(a.files[0]).split("config")[0] +
                    "config_" + b.strip("-") + ".json")
                self.hs["SelSavTrial btn"].initialdir = os.path.dirname(
                    os.path.abspath(a.files[0]))
                self.hs["SelSavTrial btn"].initialfile = os.path.basename(
                    a.files[0])
                self.hs["ReadAlign btn"].initialdir = os.path.dirname(
                    os.path.abspath(a.files[0]))
                update_json_content(
                    self.global_h.GUI_cfg_file,
                    {"cwd": os.path.dirname(os.path.abspath(a.files[0]))},
                )
                self.xanes_save_trial_set = False
                self.xanes_reg_file_set = False
                self.xanes_config_file_set = True
            else:
                self.hs[
                    "SelSavTrial text"].value = "Save Existing Configuration File ..."
                self.xanes_save_trial_set = False
                self.xanes_reg_file_set = False
                self.xanes_config_file_set = False
            self.xanes_file_configured = False
            self.hs[
                "SelFile&PathCfm text"].value = "Please comfirm your change ..."
        elif self.xanes_fit_option == "Do Analysis":
            if len(a.files[0]) != 0:
                self.xanes_save_trial_reg_filename = os.path.abspath(
                    a.files[0])
                b = ""
                t = time.strptime(time.asctime())
                for ii in range(6):
                    b += str(t[ii]).zfill(2) + "-"
                self.xanes_save_trial_reg_config_filename = (os.path.basename(
                    os.path.abspath(a.files[0])).split(".")[0] + "_config_" +
                                                             b.strip("-") +
                                                             ".json")
                self.hs["SelSavTrial btn"].initialdir = os.path.dirname(
                    os.path.abspath(a.files[0]))
                self.hs["SelSavTrial btn"].initialfile = os.path.basename(
                    a.files[0])
                self.hs[
                    "SelSavTrial text"].value = "Existing Registration File is Read ..."
                update_json_content(
                    self.global_h.GUI_cfg_file,
                    {"cwd": os.path.dirname(os.path.abspath(a.files[0]))},
                )
                self.xanes_save_trial_set = False
                self.xanes_reg_file_set = True
                self.xanes_config_file_set = False
            else:
                self.hs[
                    "SelSavTrial text"].value = "Read Existing Registration File ..."
                self.xanes_save_trial_set = False
                self.xanes_reg_file_set = False
                self.xanes_config_file_set = False
            self.xanes_file_configured = False
            self.hs[
                "SelFile&PathCfm text"].value = "Please comfirm your change ..."
        self.boxes_logic()

    def FilePathOptn_drpdn_chg(self, a):
        restart(self, dtype="3D_XANES")
        self.xanes_fit_option = a["owner"].value
        self.xanes_file_configured = False
        if self.xanes_fit_option == "Do New Reg":
            self.hs["SelRawH5Path btn"].disabled = False
            self.hs["SelReconPath btn"].disabled = False
            self.hs["SelSavTrial btn"].option = "asksaveasfilename"
            self.hs["SelSavTrial btn"].description = "Save Reg File"
            self.hs[
                "SelSavTrial text"].value = "Save trial registration as ..."
            self.xanes_save_trial_set = False
            self.xanes_reg_file_set = False
            self.xanes_config_file_set = False
            self.hs["ReadAlign chbx"].value = False
        elif self.xanes_fit_option == "Read Config File":
            self.hs["SelRawH5Path btn"].disabled = True
            self.hs["SelReconPath btn"].disabled = True
            self.hs["SelSavTrial btn"].option = "askopenfilename"
            self.hs["SelSavTrial btn"].description = "Read Config"
            self.hs["SelSavTrial btn"].open_filetypes = (
                ("json files", "*.json"),
                ("text files", "*.txt"),
            )
            self.hs[
                "SelSavTrial text"].value = "Save Existing Configuration File ..."
            self.xanes_save_trial_set = False
            self.xanes_reg_file_set = False
            self.xanes_config_file_set = False
        elif self.xanes_fit_option == "Reg By Shift":
            self.hs["SelRawH5Path btn"].disabled = True
            self.hs["SelReconPath btn"].disabled = True
            self.hs["SelSavTrial btn"].option = "askopenfilename"
            self.hs["SelSavTrial btn"].description = "Read Config"
            self.hs["SelSavTrial btn"].open_filetypes = (
                ("json files", "*.json"),
                ("text files", "*.txt"),
            )
            self.hs[
                "SelSavTrial text"].value = "Save Existing Configuration File ..."
            self.xanes_save_trial_set = False
            self.xanes_reg_file_set = False
            self.xanes_config_file_set = False
        elif self.xanes_fit_option == "Do Analysis":
            self.hs["SelRawH5Path btn"].disabled = True
            self.hs["SelReconPath btn"].disabled = True
            self.hs["SelSavTrial btn"].option = "askopenfilename"
            self.hs["SelSavTrial btn"].description = "Do Analysis"
            self.hs["SelSavTrial btn"].open_filetypes = (("h5 files",
                                                          "*.h5"), )
            self.hs[
                "SelSavTrial text"].value = "Read Existing Registration File ..."
        self.hs["SelSavTrial btn"].icon = "square-o"
        self.hs["SelSavTrial btn"].style.button_color = "orange"
        self.hs[
            "SelFile&PathCfm text"].value = "Please comfirm your change ..."
        self.boxes_logic()

    def SelFilePathCfm_btn_clk(self, a):
        if self.xanes_fit_option == "Do New Reg":
            if not self.xanes_raw_h5_path_set:
                self.hs[
                    "SelFile&PathCfm text"].value = "Please specifiy raw h5 file location ..."
                self.xanes_file_configured = False
            elif not self.xanes_recon_path_set:
                self.hs[
                    "SelFile&PathCfm text"].value = "Please specifiy recon top directory location ..."
                self.xanes_file_configured = False
            elif not self.xanes_save_trial_set:
                self.hs[
                    "SelFile&PathCfm text"].value = "Please specifiy where to save trial reg result ..."
                self.xanes_file_configured = False
            else:
                b = glob.glob(
                    os.path.join(
                        self.xanes_raw_3D_h5_top_dir,
                        self.xanes_raw_3D_h5_temp.split("{")[0] + "*.h5",
                    ))
                ran = set([
                    int(os.path.basename(ii).split(".")[0].split("_")[-1])
                    for ii in b
                ])
                b = glob.glob(
                    os.path.join(
                        self.xanes_recon_3D_top_dir,
                        self.xanes_recon_3D_dir_temp.split("{")[0] + "*",
                    ))
                ren = set([
                    int(os.path.basename(ii).split(".")[0].split("_")[-1])
                    for ii in b
                ])
                n = sorted(list(ran & ren))

                if n:
                    # self.xanes_available_recon_ids = self.xanes_available_raw_ids = n
                    self.xanes_available_raw_ids = n
                    self.hs[
                        "SelScanIdStart drpdn"].options = self.xanes_available_raw_ids
                    self.hs[
                        "SelScanIdStart drpdn"].value = self.xanes_available_raw_ids[
                            0]
                    self.hs[
                        "SelScanIdEnd drpdn"].options = self.xanes_available_raw_ids
                    self.hs[
                        "SelScanIdEnd drpdn"].value = self.xanes_available_raw_ids[
                            -1]
                    self.hs[
                        "FixedScanId drpdn"].options = self.xanes_available_raw_ids
                    self.hs[
                        "FixedScanId drpdn"].value = self.xanes_available_raw_ids[
                            0]
                    self.xanes_scan_id_s = 0
                    self.xanes_scan_id_e = len(
                        self.xanes_available_raw_ids) - 1

                    self.xanes_fixed_scan_id = self.xanes_available_raw_ids[0]
                    self.xanes_fixed_sli_id = self.hs["FixedSliId sldr"].value
                    self.hs[
                        "SelFile&PathCfm text"].value = "XANES3D file config is done ..."
                    self.xanes_file_configured = True
                else:
                    self.hs[
                        "SelFile&PathCfm text"].value = "No valid datasets (both raw and recon data) in the directory..."
                    self.xanes_file_configured = False
        elif self.xanes_fit_option == "Read Config File":
            if not self.xanes_config_file_set:
                self.hs[
                    "SelFile&PathCfm text"].value = "Please specifiy where to read the configuration file ..."
                self.xanes_file_configured = False
            else:
                self.read_xanes3D_config()
                self.set_xanes3D_variables()
                if self.xanes_roi_configured:
                    self.xanes_img_roi = np.ndarray([
                        self.xanes_roi[5] - self.xanes_roi[4] + 1,
                        self.xanes_roi[1] - self.xanes_roi[0],
                        self.xanes_roi[3] - self.xanes_roi[2],
                    ])
                    self.xanes_reg_mask = np.ndarray([
                        self.xanes_roi[1] - self.xanes_roi[0],
                        self.xanes_roi[3] - self.xanes_roi[2],
                    ])

                if self.xanes_reg_done:
                    with h5py.File(self.xanes_save_trial_reg_filename,
                                   "r") as f:
                        self.trial_reg = f[
                            "/trial_registration/trial_reg_results/{0}/trial_reg_img{0}"
                            .format("000")][:]
                        self.trial_reg_fixed = f[
                            "/trial_registration/trial_reg_results/{0}/trial_fixed_img{0}"
                            .format("000")][:]
                        self.xanes_review_aligned_img = np.ndarray(
                            self.trial_reg[0].shape)
                self.xanes_recon_path_set = True
                self.xanes_fit_option = "Read Config File"

                self.xanes_reg_best_match_filename = (os.path.splitext(
                    self.xanes_save_trial_reg_config_filename)[0].replace(
                        "config", "reg_best_match") + ".json")
                self.xanes_file_configured = True

                self.xanes_eng_list = self.reader(
                    os.path.join(self.xanes_raw_3D_h5_top_dir,
                                 self.xanes_raw_fn_temp),
                    self.xanes_available_raw_ids[self.xanes_scan_id_s:self.
                                                 xanes_scan_id_e + 1],
                    dtype="eng",
                    cfg=self.global_h.io_xanes3D_cfg,
                )
                if self.xanes_eng_list.max() < 70:
                    self.xanes_eng_list *= 1000

                if self.xanes_alignment_done:
                    with h5py.File(self.xanes_save_trial_reg_filename,
                                   "r") as f:
                        self.xanes_fit_data_shape = f[
                            "/registration_results/reg_results/registered_xanes3D"].shape
                        self.xanes_fit_eng_list = f[
                            "/trial_registration/trial_reg_parameters/eng_list"][:]
                        self.xanes_scan_id_s = f[
                            "/trial_registration/trial_reg_parameters/scan_ids"][
                                0]
                        self.xanes_scan_id_e = f[
                            "/trial_registration/trial_reg_parameters/scan_ids"][
                                -1]
                        self.xanes_aligned_data = f[
                            "/registration_results/reg_results/registered_xanes3D"][
                                0, :, :, :]
                    self.xanes_reg_best_match_filename = (os.path.splitext(
                        self.xanes_save_trial_reg_config_filename)[0].replace(
                            "config", "reg_best_match") + ".json")
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

                    self.hs["VisImgViewAlignSli sldr"].value = 1
                    self.hs["VisImgViewAlign4thDim sldr"].value = 0
                    self.hs["VisImgViewAlignOptn drpdn"].value = "x-y-E"
                    self.hs["VisImgViewAlignSli sldr"].description = "E"
                    self.hs[
                        "VisImgViewAlignSli sldr"].max = self.xanes_fit_data_shape[
                            0]
                    self.hs["VisImgViewAlignSli sldr"].min = 1
                    self.hs["VisImgViewAlign4thDim sldr"].description = "z"
                    self.hs["VisImgViewAlign4thDim sldr"].max = (
                        self.xanes_fit_data_shape[1] - 1)
                    self.hs["VisImgViewAlign4thDim sldr"].min = 0
                    self.hs[
                        "SelFile&PathCfm text"].value = "XANES3D file config is done ..."
                    self.xanes_fit_type = "full"
                    self.xanes_fit_gui_h.hs[
                        "FitEngRagOptn drpdn"].value = "full"

                self.set_xanes3D_handles()
                self.set_xanes3D_variables()
                self.xanes_fit_option = "Read Config File"
                fiji_viewer_off(self.global_h, self, viewer_name="all")
                self.hs[
                    "SelFile&PathCfm text"].value = "XANES3D file config is done ..."
        elif self.xanes_fit_option == "Reg By Shift":
            if not self.xanes_config_file_set:
                self.hs[
                    "SelFile&PathCfm text"].value = "Please specifiy where to read the configuration file ..."
                self.xanes_file_configured = False
            else:
                self.read_xanes3D_config()
                self.set_xanes3D_variables()
                if self.xanes_reg_review_done:
                    if self.xanes_roi_configured:
                        self.xanes_img_roi = np.ndarray([
                            self.xanes_roi[5] - self.xanes_roi[4] + 1,
                            self.xanes_roi[1] - self.xanes_roi[0],
                            self.xanes_roi[3] - self.xanes_roi[2],
                        ])
                        self.xanes_reg_mask = np.ndarray([
                            self.xanes_roi[1] - self.xanes_roi[0],
                            self.xanes_roi[3] - self.xanes_roi[2],
                        ])

                    if self.xanes_reg_done:
                        with h5py.File(self.xanes_save_trial_reg_filename,
                                       "r") as f:
                            self.trial_reg = f[
                                "/trial_registration/trial_reg_results/{0}/trial_reg_img{0}"
                                .format("000")][:]
                            self.trial_reg_fixed = f[
                                "/trial_registration/trial_reg_results/{0}/trial_fixed_img{0}"
                                .format("000")][:]
                            self.xanes_review_aligned_img = np.ndarray(
                                self.trial_reg[0].shape)
                    self.xanes_recon_path_set = True
                    self.xanes_fit_option = "Read Config File"

                    self.xanes_reg_best_match_filename = (os.path.splitext(
                        self.xanes_save_trial_reg_config_filename)[0].replace(
                            "config", "reg_best_match") + ".json")
                    self.xanes_file_configured = True

                    self.xanes_eng_list = self.reader(
                        os.path.join(self.xanes_raw_3D_h5_top_dir,
                                     self.xanes_raw_fn_temp),
                        self.xanes_available_raw_ids[self.xanes_scan_id_s:self.
                                                     xanes_scan_id_e + 1],
                        dtype="eng",
                        cfg=self.global_h.io_xanes3D_cfg,
                    )
                    if self.xanes_eng_list.max() < 70:
                        self.xanes_eng_list *= 1000

                    self.set_xanes3D_handles()
                    self.set_xanes3D_variables()
                    self.xanes_fit_option = "Reg By Shift"
                    self.xanes_roi_configured = False
                    self.xanes_reg_review_done = False
                    self.xanes_alignment_done = False
                    self.xanes_fit_option = "Reg By Shift"
                    fiji_viewer_off(self.global_h, self, viewer_name="all")
                    self.hs[
                        "SelFile&PathCfm text"].value = "XANES3D file config is done ..."
                else:
                    self.hs["SelFile&PathCfm text"].value = (
                        "To use this option, a config up to reg review is needed ..."
                    )
        elif self.xanes_fit_option == "Do Analysis":
            if not self.xanes_reg_file_set:
                self.hs[
                    "SelFile&PathCfm text"].value = "Please specifiy where to read the aligned data ..."
                self.xanes_file_configured = False
                self.xanes_indices_configured = False
                self.xanes_roi_configured = False
                self.xanes_reg_params_configured = False
                self.xanes_reg_done = False
                self.xanes_alignment_done = False
            else:
                self.xanes_fit_option = "Do Analysis"
                self.xanes_file_configured = True
                self.xanes_indices_configured = False
                self.xanes_roi_configured = False
                self.xanes_reg_params_configured = False
                self.xanes_reg_done = False
                self.xanes_alignment_done = True
                with h5py.File(self.xanes_save_trial_reg_filename, "r") as f:
                    self.xanes_fit_data_shape = f[
                        "/registration_results/reg_results/registered_xanes3D"].shape
                    self.xanes_fit_eng_list = f[
                        "/trial_registration/trial_reg_parameters/eng_list"][:]
                    self.xanes_scan_id_s = f[
                        "/trial_registration/trial_reg_parameters/scan_ids"][0]
                    self.xanes_scan_id_e = f[
                        "/trial_registration/trial_reg_parameters/scan_ids"][
                            -1]
                    self.xanes_fixed_scan_id = f[
                        "/trial_registration/trial_reg_parameters/fixed_scan_id"][
                            ()]
                    self.xanes_fixed_sli_id = f[
                        "/trial_registration/trial_reg_parameters/fixed_slice"][
                            ()]
                    self.xanes_aligned_data = f[
                        "/registration_results/reg_results/registered_xanes3D"][
                            0, :, :, :]
                self.xanes_reg_best_match_filename = (os.path.splitext(
                    self.xanes_save_trial_reg_config_filename)[0].replace(
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

                self.hs["VisImgViewAlignSli sldr"].value = 1
                self.hs["VisImgViewAlign4thDim sldr"].value = 0
                self.hs["VisImgViewAlignOptn drpdn"].value = "x-y-E"
                self.hs["VisImgViewAlignSli sldr"].description = "E"
                self.hs[
                    "VisImgViewAlignSli sldr"].max = self.xanes_fit_data_shape[
                        0]
                self.hs["VisImgViewAlignSli sldr"].min = 1
                self.hs["VisImgViewAlign4thDim sldr"].description = "z"
                self.hs["VisImgViewAlign4thDim sldr"].max = (
                    self.xanes_fit_data_shape[1] - 1)
                self.hs["VisImgViewAlign4thDim sldr"].min = 0
                self.hs[
                    "SelFile&PathCfm text"].value = "XANES3D file config is done ..."
                self.xanes_fit_type = "full"
                self.xanes_fit_gui_h.hs["FitEngRagOptn drpdn"].value = "full"
        self.boxes_logic()

    def SelScanIdStart_drpdn_chg(self, a):
        self.xanes_scan_id_s = self.xanes_available_raw_ids.index(
            a["owner"].value)
        self.hs["SelScanIdEnd drpdn"].options = self.xanes_available_raw_ids[
            self.xanes_scan_id_s:]
        self.hs["SelScanIdEnd drpdn"].value = self.hs[
            "SelScanIdEnd drpdn"].options[-1]
        self.hs["FixedScanId drpdn"].options = self.hs[
            "SelScanIdEnd drpdn"].options
        self.hs["FixedScanId drpdn"].value = self.hs[
            "FixedScanId drpdn"].options[0]
        self.xanes_scan_id_set = True
        self.hs["ConfigDataCfm text"].value = "scan_id_s are changed ..."
        self.xanes_indices_configured = False
        self.boxes_logic()

    def SelScanIdEnd_drpdn_chg(self, a):
        self.xanes_scan_id_e = self.xanes_available_raw_ids.index(
            a["owner"].value)
        self.hs["FixedScanId drpdn"].options = self.xanes_available_raw_ids[
            self.xanes_scan_id_s:self.xanes_scan_id_e + 1]
        self.hs["FixedScanId drpdn"].value = self.hs[
            "FixedScanId drpdn"].options[0]
        self.xanes_indices_configured = False
        self.boxes_logic()

    def FixedScanId_drpdn_chg(self, a):
        self.xanes_fixed_scan_id = self.xanes_available_raw_ids.index(
            a["owner"].value)
        b = glob.glob(
            os.path.join(
                self.xanes_recon_3D_dir_temp.format(a["owner"].value),
                self.xanes_recon_fn_temp.format(a["owner"].value, "*"),
            ))
        self.xanes_available_sli_file_ids = sorted([
            int(os.path.basename(ii).split(".")[0].split("_")[-1]) for ii in b
        ])

        if self.hs["FixedSliId sldr"].value in (
                self.xanes_available_sli_file_ids):
            self.hs["FixedSliId sldr"].max = max(
                self.xanes_available_sli_file_ids)
            self.hs["FixedSliId sldr"].min = min(
                self.xanes_available_sli_file_ids)
        else:
            self.hs["FixedSliId sldr"].max = max(
                self.xanes_available_sli_file_ids)
            self.hs["FixedSliId sldr"].value = min(
                self.xanes_available_sli_file_ids)
            self.hs["FixedSliId sldr"].min = min(
                self.xanes_available_sli_file_ids)

        if self.hs["FijiRawImgPrev chbx"].value:
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes3D_virtural_stack_preview_viewer")
        self.xanes_indices_configured = False
        self.boxes_logic()

    def FixedSliId_sldr_chg(self, a):
        if self.hs["FijiRawImgPrev chbx"].value:
            data_state, viewer_state = fiji_viewer_state(
                self.global_h,
                self,
                viewer_name="xanes3D_virtural_stack_preview_viewer")
            if viewer_state:
                self.global_h.xanes3D_fiji_windows[
                    "xanes3D_virtural_stack_preview_viewer"]["ip"].setSlice(
                        a["owner"].value - a["owner"].min + 1)
                self.xanes_fixed_sli_id = a["owner"].value
                self.hs[
                    "ConfigDataCfm text"].value = "fixed slice id is changed ..."
            else:
                self.hs[
                    "ConfigDataCfm text"].value = "Please turn on fiji previewer first ..."
        self.xanes_indices_configured = False

    def FijiRawImgPrev_chbx_chg(self, a):
        if a["owner"].value:
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes3D_virtural_stack_preview_viewer")
        else:
            data_state, viewer_state = fiji_viewer_state(
                self.global_h,
                self,
                viewer_name="xanes3D_virtural_stack_preview_viewer")
            if viewer_state:
                self.global_h.xanes3D_fiji_windows[
                    "xanes3D_virtural_stack_preview_viewer"]["ip"].close()
                self.global_h.xanes3D_fiji_windows[
                    "xanes3D_virtural_stack_preview_viewer"]["ip"] = None
                self.global_h.xanes3D_fiji_windows[
                    "xanes3D_virtural_stack_preview_viewer"]["fiji_id"] = None
            self.xanes_indices_configured = False
            self.boxes_logic()

    def FijiClose_btn_clk(self, a):
        if self.hs["FijiRawImgPrev chbx"].value:
            self.hs["FijiRawImgPrev chbx"].value = False
        if self.hs["FijiMaskViewer chbx"].value:
            self.hs["FijiMaskViewer chbx"].value = False
        try:
            for ii in self.global_h.WindowManager.getIDList():
                self.global_h.WindowManager.getImage(ii).close()
        except:
            pass
        self.global_h.xanes3D_fiji_windows[
            "xanes3D_virtural_stack_preview_viewer"]["ip"] = None
        self.global_h.xanes3D_fiji_windows[
            "xanes3D_virtural_stack_preview_viewer"]["fiji_id"] = None
        self.global_h.xanes3D_fiji_windows["xanes3D_mask_viewer"]["ip"] = None
        self.global_h.xanes3D_fiji_windows["xanes3D_mask_viewer"][
            "fiji_id"] = None
        self.global_h.xanes3D_fiji_windows["xanes3D_analysis_viewer"][
            "ip"] = None
        self.global_h.xanes3D_fiji_windows["xanes3D_analysis_viewer"][
            "fiji_id"] = None
        self.global_h.xanes3D_fiji_windows["analysis_viewer_z_plot_viewer"][
            "ip"] = None
        self.global_h.xanes3D_fiji_windows["analysis_viewer_z_plot_viewer"][
            "fiji_id"] = None
        self.xanes_indices_configured = False
        self.boxes_logic()

    def ConfigDataCfm_btn_clk(self, a):
        if self.xanes_fit_option == "Do New Reg":
            self.xanes_save_trial_reg_filename = os.path.join(
                self.xanes_raw_3D_h5_top_dir,
                self.xanes_save_trial_reg_filename_template.format(
                    self.xanes_available_raw_ids[self.xanes_scan_id_s],
                    self.xanes_available_raw_ids[self.xanes_scan_id_e],
                ),
            )
            self.xanes_save_trial_reg_config_filename = os.path.join(
                self.xanes_raw_3D_h5_top_dir,
                self.xanes_save_trial_reg_config_filename_template.format(
                    self.xanes_available_raw_ids[self.xanes_scan_id_s],
                    self.xanes_available_raw_ids[self.xanes_scan_id_e],
                ),
            )
            self.xanes_reg_best_match_filename = (os.path.splitext(
                self.xanes_save_trial_reg_config_filename)[0].replace(
                    "config", "reg_best_match") + ".json")
        if not self.xanes_indices_configured:
            self.hs["3DRoiX sldr"].max = self.global_h.xanes3D_fiji_windows[
                "xanes3D_virtural_stack_preview_viewer"]["ip"].getWidth()
            self.hs["3DRoiX sldr"].min = 0
            self.hs["3DRoiY sldr"].max = self.global_h.xanes3D_fiji_windows[
                "xanes3D_virtural_stack_preview_viewer"]["ip"].getHeight()
            self.hs["3DRoiY sldr"].min = 0
            self.hs["3DRoiZ sldr"].max = self.hs["FixedSliId sldr"].max
            self.hs["3DRoiZ sldr"].min = self.hs["FixedSliId sldr"].min

            self.hs["3DRoiZ sldr"].upper = self.hs["FixedSliId sldr"].max
            self.hs["3DRoiZ sldr"].lower = self.hs["FixedSliId sldr"].min

            self.xanes_scan_id_s = self.xanes_available_raw_ids.index(
                self.hs["SelScanIdStart drpdn"].value)
            self.xanes_scan_id_e = self.xanes_available_raw_ids.index(
                self.hs["SelScanIdEnd drpdn"].value)
            self.xanes_fixed_scan_id = self.xanes_available_raw_ids.index(
                self.hs["FixedScanId drpdn"].value)
            # self.xanes_fixed_scan_id = self.hs['FixedScanId drpdn'].value
            self.xanes_fixed_sli_id = self.hs["FixedSliId sldr"].value
            self.hs[
                "ConfigDataCfm text"].value = "Indices configuration is done ..."
            self.hs["3DRoiCfm text"].value = "Please confirm after ROI is set"
            self.hs[
                "RegPair sldr"].max = self.xanes_scan_id_e - self.xanes_scan_id_s
            self.xanes_indices_configured = True
            self.update_xanes3D_config()
            json.dump(
                self.xanes_config,
                open(self.xanes_save_trial_reg_config_filename, "w"),
                cls=NumpyArrayEncoder,
            )
            self.xanes_eng_list = self.reader(
                os.path.join(self.xanes_raw_3D_h5_top_dir,
                             self.xanes_raw_fn_temp),
                self.xanes_available_raw_ids[self.xanes_scan_id_s:self.
                                             xanes_scan_id_e + 1],
                dtype="eng",
                cfg=self.global_h.io_xanes3D_cfg,
            )
            if self.xanes_eng_list.max() < 70:
                self.xanes_eng_list *= 1000
        self.boxes_logic()

    def RoiX3D_sldr_chg(self, a):
        self.xanes_roi_configured = False
        self.boxes_logic()
        self.hs["3DRoiCfm text"].value = "Please confirm after ROI is set"
        data_state, viewer_state = fiji_viewer_state(
            self.global_h,
            self,
            viewer_name="xanes3D_virtural_stack_preview_viewer")
        if (not data_state) | (not viewer_state):
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes3D_virtural_stack_preview_viewer")
        self.global_h.xanes3D_fiji_windows[
            "xanes3D_virtural_stack_preview_viewer"]["ip"].setRoi(
                self.hs["3DRoiX sldr"].value[0],
                self.hs["3DRoiY sldr"].value[0],
                self.hs["3DRoiX sldr"].value[1] -
                self.hs["3DRoiX sldr"].value[0],
                self.hs["3DRoiY sldr"].value[1] -
                self.hs["3DRoiY sldr"].value[0],
            )

    def RoiY3D_sldr_chg(self, a):
        self.xanes_roi_configured = False
        self.boxes_logic()
        self.hs["3DRoiCfm text"].value = "Please confirm after ROI is set"
        data_state, viewer_state = fiji_viewer_state(
            self.global_h,
            self,
            viewer_name="xanes3D_virtural_stack_preview_viewer")
        if (not data_state) | (not viewer_state):
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes3D_virtural_stack_preview_viewer")
        self.global_h.xanes3D_fiji_windows[
            "xanes3D_virtural_stack_preview_viewer"]["ip"].setRoi(
                self.hs["3DRoiX sldr"].value[0],
                self.hs["3DRoiY sldr"].value[0],
                self.hs["3DRoiX sldr"].value[1] -
                self.hs["3DRoiX sldr"].value[0],
                self.hs["3DRoiY sldr"].value[1] -
                self.hs["3DRoiY sldr"].value[0],
            )

    def RoiZ3D_val_sldr_chg(self, a):
        self.xanes_roi_configured = False
        if a["owner"].upper < self.xanes_fixed_sli_id:
            a["owner"].upper = self.xanes_fixed_sli_id
        if a["owner"].lower > self.xanes_fixed_sli_id:
            a["owner"].lower = self.xanes_fixed_sli_id
        a["owner"].mylower = a["owner"].lower
        a["owner"].myupper = a["owner"].upper
        self.hs["3DRoiCfm text"].value = "Please confirm after ROI is set"

    def RoiZ3D_lwr_sldr_chg(self, a):
        data_state, viewer_state = fiji_viewer_state(
            self.global_h,
            self,
            viewer_name="xanes3D_virtural_stack_preview_viewer")
        if (not data_state) | (not viewer_state):
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes3D_virtural_stack_preview_viewer")
        self.global_h.xanes3D_fiji_windows[
            "xanes3D_virtural_stack_preview_viewer"]["ip"].setSlice(
                self.hs["3DRoiZ sldr"].value[0] - self.hs["3DRoiZ sldr"].min +
                1)
        self.boxes_logic()

    def RoiZ3D_upr_sldr_chg(self, a):
        data_state, viewer_state = fiji_viewer_state(
            self.global_h,
            self,
            viewer_name="xanes3D_virtural_stack_preview_viewer")
        if (not data_state) | (not viewer_state):
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes3D_virtural_stack_preview_viewer")
        self.global_h.xanes3D_fiji_windows[
            "xanes3D_virtural_stack_preview_viewer"]["ip"].setSlice(
                self.hs["3DRoiZ sldr"].value[1] - self.hs["3DRoiZ sldr"].min +
                1)
        self.boxes_logic()

    def Roi3DCfm_btn_clk(self, a):
        data_state, viewer_state = fiji_viewer_state(
            self.global_h,
            self,
            viewer_name="xanes3D_virtural_stack_preview_viewer")
        if viewer_state:
            fiji_viewer_off(
                self.global_h,
                self,
                viewer_name="xanes3D_virtural_stack_preview_viewer")
        self.hs["FijiRawImgPrev chbx"].value = False
        self.xanes_indices_configured = True

        if not self.xanes_roi_configured:
            self.xanes_roi = [
                self.hs["3DRoiY sldr"].value[0],
                self.hs["3DRoiY sldr"].value[1],
                self.hs["3DRoiX sldr"].value[0],
                self.hs["3DRoiX sldr"].value[1],
                self.hs["3DRoiZ sldr"].value[0],
                self.hs["3DRoiZ sldr"].value[1],
            ]

            self.xanes_img_roi = np.ndarray([
                self.xanes_roi[5] - self.xanes_roi[4] + 1,
                self.xanes_roi[1] - self.xanes_roi[0],
                self.xanes_roi[3] - self.xanes_roi[2],
            ])
            self.xanes_reg_mask = np.ndarray([
                self.xanes_roi[1] - self.xanes_roi[0],
                self.xanes_roi[3] - self.xanes_roi[2],
            ])

            self.hs["3DRoiCfm text"].value = "ROI configuration is done ..."
            self.hs["SliSrch sldr"].max = min(
                abs(self.hs["3DRoiZ sldr"].value[1] - self.xanes_fixed_sli_id),
                abs(self.hs["3DRoiZ sldr"].value[0] - self.xanes_fixed_sli_id),
            )
            self.xanes_roi_configured = True
        self.update_xanes3D_config()
        json.dump(
            self.xanes_config,
            open(self.xanes_save_trial_reg_config_filename, "w"),
            cls=NumpyArrayEncoder,
        )
        self.boxes_logic()

    def FijiMaskViewer_chbx_chg(self, a):
        if a["owner"].value:
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes3D_mask_viewer")
        else:
            try:
                self.global_h.xanes3D_fiji_windows["xanes3D_mask_viewer"][
                    "ip"].close()
            except:
                pass
            self.global_h.xanes3D_fiji_windows["xanes3D_mask_viewer"][
                "ip"] = None
            self.global_h.xanes3D_fiji_windows["xanes3D_mask_viewer"][
                "fiji_id"] = None
        self.boxes_logic()

    def ChunkSz_chbx_chg(self, a):
        self.xanes_reg_params_configured = False
        if a["owner"].value:
            self.xanes_reg_use_chunk = True
        else:
            self.xanes_reg_use_chunk = False
        self.boxes_logic()

    def UseMask_chbx_chg(self, a):
        self.xanes_reg_params_configured = False
        if self.hs["UseMask chbx"].value:
            self.xanes_reg_use_mask = True
        else:
            self.xanes_reg_use_mask = False
        self.boxes_logic()

    def MaskThres_sldr_chg(self, a):
        if self.xanes_reg_use_mask:
            self.xanes_reg_params_configured = False
            self.xanes_reg_mask_thres = a["owner"].value
            data_state, viewer_state = fiji_viewer_state(
                self.global_h, self, viewer_name="xanes3D_mask_viewer")
            if (not data_state) | (not viewer_state):
                self.hs["FijiMaskViewer chbx"].value = False
                self.hs["FijiMaskViewer chbx"].value = True

            if self.xanes_reg_mask_dilation_width == 0:
                self.xanes_reg_mask[:] = (
                    self.xanes_img_roi[self.xanes_fixed_sli_id -
                                       self.xanes_roi[4]] >
                    self.xanes_reg_mask_thres).astype(np.uint8)[:]
            else:
                self.xanes_reg_mask[:] = skm.binary_dilation(
                    (self.xanes_img_roi[self.xanes_fixed_sli_id -
                                        self.xanes_roi[4]] >
                     self.xanes_reg_mask_thres).astype(np.uint8),
                    np.ones([
                        self.xanes_reg_mask_dilation_width,
                        self.xanes_reg_mask_dilation_width,
                    ]),
                ).astype(np.uint8)[:]
            self.global_h.xanes3D_fiji_windows["xanes3D_mask_viewer"][
                "ip"].setImage(self.global_h.ij.convert().convert(
                    self.global_h.ij.dataset().create(
                        self.global_h.ij.py.to_java(self.xanes_img_roi *
                                                    self.xanes_reg_mask)),
                    self.global_h.ImagePlusClass,
                ))
            self.global_h.xanes3D_fiji_windows["xanes3D_mask_viewer"][
                "ip"].setSlice(self.xanes_fixed_sli_id - self.xanes_roi[4])
            self.global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")

    def MaskDilation_sldr_chg(self, a):
        if self.xanes_reg_use_mask:
            self.xanes_reg_params_configured = False
            data_state, viewer_state = fiji_viewer_state(
                self.global_h, self, viewer_name="xanes3D_mask_viewer")
            if (not data_state) | (not viewer_state):
                self.hs["FijiMaskViewer chbx"].value = False
                self.hs["FijiMaskViewer chbx"].value = True
            self.xanes_reg_mask_dilation_width = a["owner"].value
            if self.xanes_reg_mask_dilation_width == 0:
                self.xanes_reg_mask[:] = (
                    self.xanes_img_roi[self.xanes_fixed_sli_id -
                                       self.xanes_roi[4]] >
                    self.xanes_reg_mask_thres).astype(np.uint8)[:]
            else:
                self.xanes_reg_mask[:] = skm.binary_dilation(
                    (self.xanes_img_roi[self.xanes_fixed_sli_id -
                                        self.xanes_roi[4]] >
                     self.xanes_reg_mask_thres).astype(np.uint8),
                    np.ones([
                        self.xanes_reg_mask_dilation_width,
                        self.xanes_reg_mask_dilation_width,
                    ]),
                ).astype(np.uint8)[:]
            self.global_h.xanes3D_fiji_windows["xanes3D_mask_viewer"][
                "ip"].setImage(self.global_h.ij.convert().convert(
                    self.global_h.ij.dataset().create(
                        self.global_h.ij.py.to_java(self.xanes_img_roi *
                                                    self.xanes_reg_mask)),
                    self.global_h.ImagePlusClass,
                ))
            self.global_h.xanes3D_fiji_windows["xanes3D_mask_viewer"][
                "ip"].setSlice(self.xanes_fixed_sli_id - self.xanes_roi[4])
            self.global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")

    def SliSrch_sldr_chg(self, a):
        self.xanes_reg_params_configured = False
        data_state, viewer_state = fiji_viewer_state(
            self.global_h, self, viewer_name="xanes3D_mask_viewer")
        if (not data_state) | (not viewer_state):
            self.hs["FijiMaskViewer chbx"].value = False
            self.hs["FijiMaskViewer chbx"].value = True
        self.global_h.xanes3D_fiji_windows["xanes3D_mask_viewer"][
            "ip"].setSlice(a["owner"].value)

    def ChunkSz_sldr_chg(self, a):
        self.xanes_reg_params_configured = False
        self.boxes_logic()

    def RegMethod_drpdn_chg(self, a):
        self.xanes_reg_params_configured = False
        self.boxes_logic()

    def RefMode_drpdn_chg(self, a):
        self.xanes_reg_params_configured = False
        self.boxes_logic()

    def MRTVLevel_text_chg(self, a):
        self.xanes_reg_params_configured = False
        self.boxes_logic()

    def MRTVWz_text_chg(self, a):
        self.xanes_reg_params_configured = False
        self.boxes_logic()

    def MRTVSubpixelWz_text_chg(self, a):
        self.xanes_reg_params_configured = False
        self.boxes_logic()

    def MRTVSubpixelKernel_text_chg(self, a):
        self.xanes_reg_params_configured = False
        self.boxes_logic()

    def MRTVSubpixelSrch_drpdn_chg(self, a):
        if a["owner"].value == "analytical":
            self.hs["MRTVSubpixelWz text"].value = 3
        else:
            self.hs["MRTVSubpixelWz text"].value = 5
        self.xanes_reg_params_configured = False
        self.boxes_logic()

    def ConfigRegParamsCfm_btn_clk(self, a):
        fiji_viewer_off(self.global_h, self, viewer_name="xanes3D_mask_viewer")
        self.hs["FijiMaskViewer chbx"].value = False
        self.xanes_reg_sli_search_half_width = self.hs["SliSrch sldr"].value
        self.xanes_reg_chunk_sz = self.hs["ChunkSz sldr"].value
        self.xanes_reg_method = self.hs["RegMethod drpdn"].value
        self.xanes_reg_ref_mode = self.hs["RefMode drpdn"].value
        self.xanes_reg_mask_dilation_width = self.hs["MaskDilation sldr"].value
        self.xanes_reg_mask_thres = self.hs["MaskThres sldr"].value
        self.xanes_reg_mrtv_level = self.hs["MRTVLevel text"].value
        self.xanes_reg_mrtv_width = self.hs["MRTVWz text"].value
        self.xanes_reg_mrtv_subpixel_wz = self.hs["MRTVSubpixelWz text"].value
        self.xanes_reg_mrtv_subpixel_kernel = self.hs[
            "MRTVSubpixelKernel text"].value
        if self.hs["MRTVSubpixelSrch drpdn"].value == "analytical":
            self.xanes_reg_mrtv_subpixel_srch_option = "ana"
        else:
            self.xanes_reg_mrtv_subpixel_srch_option = "fit"

        self.xanes_reg_params_configured = self.hs["ChunkSz chbx"].value
        self.hs[
            "ConfigRegParamsCfm text"].value = "registration parameters are set ..."
        self.xanes_reg_params_configured = True
        self.update_xanes3D_config()
        json.dump(
            self.xanes_config,
            open(self.xanes_save_trial_reg_config_filename, "w"),
            cls=NumpyArrayEncoder,
        )
        self.boxes_logic()

    def RunRegCfm_btn_clk(self, a):
        tmp_file = os.path.join(self.global_h.tmp_dir, "xanes3D_tmp.h5")
        with h5py.File(tmp_file, "w") as f:
            f.create_dataset("analysis_eng_list",
                             data=self.xanes_eng_list.astype(np.float32))
            if self.xanes_reg_mask is not None:
                f.create_dataset("xanes3D_reg_mask",
                                 data=self.xanes_reg_mask.astype(np.float32))
            else:
                f.create_dataset("xanes3D_reg_mask", data=np.array([0]))
        code = {}
        ln = 0

        code[ln] = f"import os"
        ln += 1
        code[ln] = f"from TXM_Sandbox.utils import xanes_regtools as xr"
        ln += 1
        code[
            ln] = f"reg = xr.regtools(dtype='3D_XANES', method='{self.xanes_reg_method}', mode='TRANSLATION')"
        ln += 1
        code[ln] = f"from multiprocessing import freeze_support"
        ln += 1
        code[ln] = f"if __name__ == '__main__':"
        ln += 1
        code[ln] = f"    freeze_support()"
        ln += 1
        kwargs = {
            "raw_h5_top_dir": self.xanes_raw_3D_h5_top_dir,
            "recon_top_dir": self.xanes_recon_3D_top_dir,
        }
        code[ln] = f"    reg.set_raw_data_info(**{kwargs})"
        ln += 1
        code[ln] = f"    reg.set_method('{self.xanes_reg_method}')"
        ln += 1
        code[ln] = f"    reg.set_ref_mode('{self.xanes_reg_ref_mode}')"
        ln += 1
        code[ln] = f"    reg.set_xanes3D_tmp_filename('{tmp_file}')"
        ln += 1
        code[ln] = f"    reg.read_xanes3D_tmp_file()"
        ln += 1
        code[
            ln] = f"    reg.set_xanes3D_raw_h5_top_dir('{self.xanes_raw_3D_h5_top_dir}')"
        ln += 1
        code[
            ln] = f"    reg.set_indices(0, {self.xanes_scan_id_e-self.xanes_scan_id_s+1}, {self.xanes_fixed_scan_id-self.xanes_scan_id_s})"
        ln += 1
        code[
            ln] = f"    reg.set_xanes3D_scan_ids({self.xanes_available_raw_ids[self.xanes_scan_id_s:self.xanes_scan_id_e+1]})"
        ln += 1
        code[
            ln] = f"    reg.set_reg_options(use_mask={self.xanes_reg_use_mask}, mask_thres={self.xanes_reg_mask_thres},\
                     use_chunk={self.xanes_reg_use_chunk}, chunk_sz={self.xanes_reg_chunk_sz},\
                     use_smooth_img={self.xanes_reg_use_smooth_img}, smooth_sigma={self.xanes_reg_smooth_sigma},\
                     mrtv_level={self.xanes_reg_mrtv_level}, mrtv_width={self.xanes_reg_mrtv_width}, \
                     mrtv_sp_wz={self.xanes_reg_mrtv_subpixel_wz}, mrtv_sp_kernel={self.xanes_reg_mrtv_subpixel_kernel})"

        ln += 1
        code[ln] = f"    reg.set_roi({self.xanes_roi})"
        ln += 1
        code[
            ln] = f"    reg.set_xanes3D_recon_path_template('{self.xanes_recon_3D_tiff_temp}')"
        ln += 1
        code[
            ln] = f"    reg.set_saving(os.path.dirname('{self.xanes_save_trial_reg_filename}'), \
                     fn=os.path.basename('{self.xanes_save_trial_reg_filename}'))"

        ln += 1
        code[
            ln] = f"    reg.xanes3D_sli_search_half_range = {self.xanes_reg_sli_search_half_width}"
        ln += 1
        code[
            ln] = f"    reg.xanes3D_recon_fixed_sli = {self.xanes_fixed_sli_id}"
        ln += 1
        code[ln] = f"    reg.compose_dicts()"
        ln += 1
        code[ln] = f"    reg.reg_xanes3D_chunk()"
        ln += 1

        gen_external_py_script(self.xanes_reg_external_command_name, code)
        sig = os.system(f"python {self.xanes_reg_external_command_name}")

        print(sig)
        if sig == 0:
            with h5py.File(self.xanes_save_trial_reg_filename, "r") as f:
                self.trial_reg = np.ndarray(f[
                    "/trial_registration/trial_reg_results/{0}/trial_reg_img{0}"
                    .format(str(0).zfill(3))].shape)
                self.trial_reg_fixed = np.ndarray(f[
                    "/trial_registration/trial_reg_results/{0}/trial_fixed_img{0}"
                    .format(str(0).zfill(3))].shape)
                self.xanes_alignment_pairs = f[
                    "/trial_registration/trial_reg_parameters/alignment_pairs"][:]
                self.hs["RegPair sldr"].max = self.xanes_alignment_pairs.shape[
                    0] - 1
                self.hs["RegPair sldr"].min = 0
            self.hs[
                "CorrZShift text"].max = self.xanes_reg_sli_search_half_width * 2

            self.xanes_review_aligned_img = np.ndarray(self.trial_reg[0].shape)
            self.xanes_review_shift_dict = {}
            self.xanes_review_bad_shift = False
            self.xanes_reg_done = True
            self.xanes_reg_best_match_filename = (os.path.splitext(
                self.xanes_save_trial_reg_config_filename)[0].replace(
                    "config", "reg_best_match") + ".json")
            self.update_xanes3D_config()
            json.dump(
                self.xanes_config,
                open(self.xanes_save_trial_reg_config_filename, "w"),
                cls=NumpyArrayEncoder,
            )
            self.hs["RunRegCfm text"].value = "XANES3D registration is done"
        else:
            self.hs[
                "RunRegCfm text"].value = "Something went wrong during XANES3D registration"
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
        self.boxes_logic()

    def RegPair_sldr_chg(self, a):
        self.xanes_alignment_pair_id = a["owner"].value
        with h5py.File(self.xanes_save_trial_reg_filename, "r") as f:
            self.trial_reg[:] = f[
                "/trial_registration/trial_reg_results/{0}/trial_reg_img{0}".
                format(str(self.xanes_alignment_pair_id).zfill(3))][:]
            self.trial_reg_fixed[:] = f[
                "/trial_registration/trial_reg_results/{0}/trial_fixed_img{0}".
                format(str(self.xanes_alignment_pair_id).zfill(3))][:]
            shift = f[
                "/trial_registration/trial_reg_results/{0}/shift{0}".format(
                    str(self.xanes_alignment_pair_id).zfill(3))][:]
            if self.xanes_reg_method == "MRTV":
                best_match = f[
                    "/trial_registration/trial_reg_results/{0}/mrtv_best_shift_id{0}"
                    .format(str(self.xanes_alignment_pair_id).zfill(3))][()]
            else:
                best_match = 0

        self.xanes_review_shift_dict[str(
            self.xanes_alignment_pair_id)] = np.array([
                best_match - self.xanes_reg_sli_search_half_width,
                shift[best_match][0],
                shift[best_match][1],
            ])
        fiji_viewer_off(self.global_h,
                        self,
                        viewer_name="xanes3D_review_viewer")
        self.global_h.ij.py.run_macro("""call("java.lang.System.gc")""")
        fiji_viewer_on(self.global_h,
                       self,
                       viewer_name="xanes3D_review_viewer")
        self.global_h.xanes3D_fiji_windows["xanes3D_review_viewer"][
            "ip"].setTitle("reg pair: " +
                           str(self.xanes_alignment_pair_id).zfill(3))
        self.global_h.xanes3D_fiji_windows["xanes3D_review_viewer"][
            "ip"].setSlice(best_match + 1)
        self.global_h.ij.py.run_macro(
            """run("Enhance Contrast", "saturated=0.35")""")

        self.hs["RevRegRltCfm text"].value = str(self.xanes_review_shift_dict)
        json.dump(
            self.xanes_review_shift_dict,
            open(self.xanes_reg_best_match_filename, "w"),
            cls=NumpyArrayEncoder,
        )

        self.xanes_review_bad_shift = False
        self.xanes_reg_review_done = False

    def RegPairBad_btn_clk(self, a):
        self.xanes_review_bad_shift = True
        self.xanes_manual_xshift = 0
        self.xanes_manual_yshift = 0
        self.xanes_manual_zshift = self.global_h.xanes3D_fiji_windows[
            "xanes3D_review_viewer"]["ip"].getCurrentSlice()
        fiji_viewer_on(self.global_h,
                       self,
                       viewer_name="xanes3D_review_manual_viewer")
        self.hs["CorrZShift text"].value = self.xanes_manual_zshift
        self.global_h.ij.py.run_macro(
            """run("Enhance Contrast", "saturated=0.35")""")
        self.xanes_reg_review_done = False
        self.boxes_logic()

    def CorrXShift_text_chg(self, a):
        self.xanes_manual_xshift = self.hs["CorrXShift text"].value
        self.xanes_manual_yshift = self.hs["CorrYShift text"].value
        self.xanes_manual_zshift = self.hs["CorrZShift text"].value
        xanes3D_review_aligned_img = np.real(
            np.fft.ifftn(
                fourier_shift(
                    np.fft.fftn(self.trial_reg[self.xanes_manual_zshift - 1]),
                    [self.xanes_manual_yshift, self.xanes_manual_xshift],
                )))

        self.global_h.xanes3D_fiji_windows["xanes3D_review_manual_viewer"][
            "ip"].setImage(self.global_h.ij.convert().convert(
                self.global_h.ij.dataset().create(
                    self.global_h.ij.py.to_java(xanes3D_review_aligned_img -
                                                self.trial_reg_fixed)),
                self.global_h.ImagePlusClass,
            ))
        self.global_h.ij.py.run_macro(
            """run("Enhance Contrast", "saturated=0.35")""")

    def CorrYShift_text_chg(self, a):
        self.xanes_manual_xshift = self.hs["CorrXShift text"].value
        self.xanes_manual_yshift = self.hs["CorrYShift text"].value
        self.xanes_manual_zshift = self.hs["CorrZShift text"].value
        xanes3D_review_aligned_img = np.real(
            np.fft.ifftn(
                fourier_shift(
                    np.fft.fftn(self.trial_reg[self.xanes_manual_zshift - 1]),
                    [self.xanes_manual_yshift, self.xanes_manual_xshift],
                )))

        self.global_h.xanes3D_fiji_windows["xanes3D_review_manual_viewer"][
            "ip"].setImage(self.global_h.ij.convert().convert(
                self.global_h.ij.dataset().create(
                    self.global_h.ij.py.to_java(xanes3D_review_aligned_img -
                                                self.trial_reg_fixed)),
                self.global_h.ImagePlusClass,
            ))
        self.global_h.ij.py.run_macro(
            """run("Enhance Contrast", "saturated=0.35")""")

    def CorrZShift_text_chg(self, a):
        self.xanes_manual_xshift = self.hs["CorrXShift text"].value
        self.xanes_manual_yshift = self.hs["CorrYShift text"].value
        self.xanes_manual_zshift = self.hs["CorrZShift text"].value
        xanes3D_review_aligned_img = np.real(
            np.fft.ifftn(
                fourier_shift(
                    np.fft.fftn(self.trial_reg[self.xanes_manual_zshift - 1]),
                    [self.xanes_manual_yshift, self.xanes_manual_xshift],
                )))

        self.global_h.xanes3D_fiji_windows["xanes3D_review_manual_viewer"][
            "ip"].setImage(self.global_h.ij.convert().convert(
                self.global_h.ij.dataset().create(
                    self.global_h.ij.py.to_java(xanes3D_review_aligned_img -
                                                self.trial_reg_fixed)),
                self.global_h.ImagePlusClass,
            ))
        self.global_h.ij.py.run_macro(
            """run("Enhance Contrast", "saturated=0.35")""")

    def CorrShftRecord_btn_clk(self, a):
        """
        temporarily this wont work with SR for now.
        """
        self.xanes_review_bad_shift = False
        with h5py.File(self.xanes_save_trial_reg_filename, "r") as f:
            shift = f[
                "/trial_registration/trial_reg_results/{0}/shift{0}".format(
                    str(self.xanes_alignment_pair_id).zfill(3))][:]
        best_match = self.xanes_manual_zshift - 1
        self.xanes_review_shift_dict["{}".format(
            self.xanes_alignment_pair_id)] = np.array([
                best_match - self.xanes_reg_sli_search_half_width,
                self.xanes_manual_yshift + shift[best_match, 0],
                self.xanes_manual_xshift + shift[best_match, 1],
            ])
        self.hs["CorrXShift text"].value = 0
        self.hs["CorrYShift text"].value = 0
        self.hs["CorrZShift text"].value = 1
        fiji_viewer_off(self.global_h,
                        self,
                        viewer_name="xanes3D_review_manual_viewer")
        json.dump(
            self.xanes_review_shift_dict,
            open(self.xanes_reg_best_match_filename, "w"),
            cls=NumpyArrayEncoder,
        )
        self.xanes_reg_review_done = False
        self.boxes_logic()

    def RevRegRltCfm_btn_clk(self, a):
        if len(self.xanes_review_shift_dict) != (self.hs["RegPair sldr"].max +
                                                 1):
            self.hs[
                "RevRegRltCfm text"].value = "reg review is not completed yet ..."
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
            fiji_viewer_off(self.global_h,
                            self,
                            viewer_name="xanes3D_review_manual_viewer")
            fiji_viewer_off(self.global_h,
                            self,
                            viewer_name="xanes3D_review_viewer")
            self.hs["RevRegRltCfm text"].value = "reg review is done ..."
            self.xanes_reg_review_done = True
            self.update_xanes3D_config()
            json.dump(
                self.xanes_config,
                open(self.xanes_save_trial_reg_config_filename, "w"),
                cls=NumpyArrayEncoder,
            )
        self.boxes_logic()

    def AlignReconOptnSli_chbx_chg(self, a):
        boxes = [
            "align_recon_optional_slice_start_text",
            "align_recon_optional_slice_range_slider",
            "align_recon_optional_slice_end_text",
        ]
        if a["owner"].value:
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
        else:
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)

    def AlignReconOptnSliStart_text_chg(self, a):
        if a["owner"].value <= self.hs["AlignReconOptnSliRange sldr"].upper:
            self.hs["AlignReconOptnSliRange sldr"].lower = a["owner"].value
        else:
            a["owner"].value = self.hs["AlignReconOptnSliRange sldr"].upper

    def AlignReconOptnSliEnd_text_chg(self, a):
        if a["owner"].value >= self.hs["AlignReconOptnSliRange sldr"].lower:
            self.hs["AlignReconOptnSliRange sldr"].upper = a["owner"].value
        else:
            a["owner"].value = self.hs["AlignReconOptnSliRange sldr"].lower

    def AlignReconOptnSliRange_sldr_chg(self, a):
        self.hs["AlignReconOptnSliStart text"].value = a["owner"].lower
        self.hs["AlignReconOptnSliEnd text"].value = a["owner"].upper
        data_state, viewer_state = fiji_viewer_state(
            self.global_h,
            self,
            viewer_name="xanes3D_virtural_stack_preview_viewer")
        if (not data_state) | (not viewer_state):
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes3D_virtural_stack_preview_viewer")
        self.global_h.xanes3D_fiji_windows[
            "xanes3D_virtural_stack_preview_viewer"]["ip"].setSlice(
                a["owner"].lower - a["owner"].min + 1)
        self.global_h.xanes3D_fiji_windows[
            "xanes3D_virtural_stack_preview_viewer"]["ip"].setRoi(
                int(self.xanes_roi[2]),
                int(self.xanes_roi[0]),
                int(self.xanes_roi[3] - self.xanes_roi[2]),
                int(self.xanes_roi[1] - self.xanes_roi[0]),
            )

    def AlignReconCfm_btn_clk(self, a):
        if self.hs["AlignReconOptnSli chbx"].value:
            self.xanes_roi[4] = self.hs["AlignReconOptnSliRange sldr"].lower
            self.xanes_roi[5] = self.hs["AlignReconOptnSliRange sldr"].upper
        tmp_dict = {}
        for key in self.xanes_review_shift_dict.keys():
            tmp_dict[key] = tuple(self.xanes_review_shift_dict[key])

        code = {}
        ln = 0
        code[ln] = "import TXM_Sandbox.utils.xanes_regtools as xr"
        ln += 1
        code[ln] = "reg = xr.regtools(dtype='3D_XANES', mode='TRANSLATION')"
        ln += 1
        code[
            ln] = f"reg.set_xanes3D_recon_path_template('{self.xanes_recon_3D_tiff_temp}')"
        ln += 1
        code[ln] = f"reg.set_roi({self.xanes_roi})"
        ln += 1
        code[
            ln] = f"reg.apply_xanes3D_chunk_shift({tmp_dict}, {self.xanes_roi[4]}, {self.xanes_roi[5]}, trialfn='{self.xanes_save_trial_reg_filename}', savefn='{self.xanes_save_trial_reg_filename}')"
        ln += 1

        gen_external_py_script(self.xanes_align_external_command_name, code)
        sig = os.system(f"python '{self.xanes_align_external_command_name}'")
        if sig == 0:
            self.hs[
                "AlignReconCfm text"].value = "XANES3D alignment is done ..."
            self.update_xanes3D_config()
            json.dump(
                self.xanes_config,
                open(self.xanes_save_trial_reg_config_filename, "w"),
                cls=NumpyArrayEncoder,
            )
            self.boxes_logic()
            self.xanes_alignment_done = True

            with h5py.File(self.xanes_save_trial_reg_filename, "r") as f:
                self.xanes_fit_data_shape = f[
                    "/registration_results/reg_results/registered_xanes3D"].shape
                self.xanes_fit_eng_list = f[
                    "/trial_registration/trial_reg_parameters/eng_list"][:]
                self.xanes_scan_id_s = f[
                    "/trial_registration/trial_reg_parameters/scan_ids"][0]
                self.xanes_scan_id_e = f[
                    "/trial_registration/trial_reg_parameters/scan_ids"][-1]

            self.xanes_element = determine_element(self.xanes_fit_eng_list)
            if self.xanes_element is None:
                print(
                    "Cannot determine the element edge. Maybe there is not enough number of energy points. Skip XANES fitting"
                )
            else:
                tem = determine_fitting_energy_range(self.xanes_element)
                self.xanes_fit_edge_eng = tem[0]
                self.xanes_fit_wl_fit_eng_s = tem[1]
                self.xanes_fit_wl_fit_eng_e = tem[2]
                self.xanes_fit_pre_edge_e = tem[3]
                self.xanes_fit_post_edge_s = tem[4]
                self.xanes_fit_edge_0p5_fit_s = tem[5]
                self.xanes_fit_edge_0p5_fit_e = tem[6]
                self.xanes_fit_type = "full"
                self.xanes_fit_gui_h.hs["FitEngRagOptn drpdn"].value = "full"

            self.hs["VisImgViewAlignSli sldr"].value = 1
            self.hs["VisImgViewAlign4thDim sldr"].value = 0
            self.hs["VisImgViewAlignOptn drpdn"].value = "x-y-E"
            self.hs["VisImgViewAlignSli sldr"].description = "E"
            self.hs["VisImgViewAlignSli sldr"].max = self.xanes_fit_data_shape[
                0]
            self.hs["VisImgViewAlignSli sldr"].min = 1
            self.hs["VisImgViewAlign4thDim sldr"].description = "z"
            self.hs[
                "VisImgViewAlign4thDim sldr"].max = self.xanes_fit_data_shape[
                    1] - 1
            self.hs["VisImgViewAlign4thDim sldr"].min = 0
            self.hs[
                "SelFile&PathCfm text"].value = "XANES3D file config is done ..."
        else:
            self.hs[
                "AlignReconCfm text"].value = "something wrong in XANES3D alignment ..."
        self.boxes_logic()

    def VisImgViewerOptn_tgbtn_clk(self, a):
        self.xanes_visualization_viewer_option = a["owner"].value
        if self.xanes_visualization_viewer_option == "napari":
            with h5py.File(self.xanes_save_trial_reg_filename, "r") as f:
                self.xanes_aligned_data = 0
                self.xanes_aligned_data = f[
                    "/registration_results/reg_results/registered_xanes3D"][:]
            self.viewer = napari.view_image(self.xanes_aligned_data)
        self.boxes_logic()

    def VisImgViewAlignOptn_drpdn_clk(self, a):
        """
        image data on the disk is save in format [E, z, y, x]
        """
        self.xanes_fit_view_option = a["owner"].value
        with h5py.File(self.xanes_save_trial_reg_filename, "r") as f:
            if self.xanes_fit_view_option == "x-y-E":
                self.hs["VisImgViewAlignSli sldr"].value = 1
                self.hs["VisImgViewAlignSli sldr"].description = "E"
                self.hs[
                    "VisImgViewAlignSli sldr"].max = self.xanes_fit_data_shape[
                        0]
                self.hs["VisImgViewAlignSli sldr"].min = 1
                self.hs["VisImgViewAlign4thDim sldr"].value = 0
                self.hs["VisImgViewAlign4thDim sldr"].description = "z"
                self.hs["VisImgViewAlign4thDim sldr"].max = (
                    self.xanes_fit_data_shape[1] - 1)
                self.hs["VisImgViewAlign4thDim sldr"].min = 0
                self.xanes_aligned_data = 0
                self.xanes_aligned_data = f[
                    "/registration_results/reg_results/registered_xanes3D"][:,
                                                                            0, :, :]
            elif self.xanes_fit_view_option == "y-z-E":
                self.hs["VisImgViewAlignSli sldr"].value = 1
                self.hs["VisImgViewAlignSli sldr"].description = "E"
                self.hs[
                    "VisImgViewAlignSli sldr"].max = self.xanes_fit_data_shape[
                        0]
                self.hs["VisImgViewAlignSli sldr"].min = 1
                self.hs["VisImgViewAlign4thDim sldr"].value = 0
                self.hs["VisImgViewAlign4thDim sldr"].description = "x"
                self.hs["VisImgViewAlign4thDim sldr"].max = (
                    self.xanes_fit_data_shape[3] - 1)
                self.hs["VisImgViewAlign4thDim sldr"].min = 0
                self.xanes_aligned_data = 0
                self.xanes_aligned_data = f[
                    "/registration_results/reg_results/registered_xanes3D"][:, :, :,
                                                                            0]
            elif self.xanes_fit_view_option == "z-x-E":
                self.hs["VisImgViewAlignSli sldr"].value = 1
                self.hs["VisImgViewAlignSli sldr"].description = "E"
                self.hs[
                    "VisImgViewAlignSli sldr"].max = self.xanes_fit_data_shape[
                        0]
                self.hs["VisImgViewAlignSli sldr"].min = 1
                self.hs["VisImgViewAlign4thDim sldr"].value = 0
                self.hs["VisImgViewAlign4thDim sldr"].description = "y"
                self.hs["VisImgViewAlign4thDim sldr"].max = (
                    self.xanes_fit_data_shape[2] - 1)
                self.hs["VisImgViewAlign4thDim sldr"].min = 0
                self.xanes_aligned_data = 0
                self.xanes_aligned_data = f[
                    "/registration_results/reg_results/registered_xanes3D"][:, :,
                                                                            0, :]
            elif self.xanes_fit_view_option == "x-y-z":
                self.hs["VisImgViewAlignSli sldr"].value = 1
                self.hs["VisImgViewAlignSli sldr"].description = "z"
                self.hs[
                    "VisImgViewAlignSli sldr"].max = self.xanes_fit_data_shape[
                        1]
                self.hs["VisImgViewAlignSli sldr"].min = 1
                self.hs["VisImgViewAlign4thDim sldr"].value = 0
                self.hs["VisImgViewAlign4thDim sldr"].description = "E"
                self.hs["VisImgViewAlign4thDim sldr"].max = (
                    self.xanes_fit_data_shape[0] - 1)
                self.hs["VisImgViewAlign4thDim sldr"].min = 0
                self.xanes_aligned_data = 0
                self.xanes_aligned_data = f[
                    "/registration_results/reg_results/registered_xanes3D"][
                        0, :, :, :]
        self.xanes_fit_view_option_previous = self.xanes_fit_view_option
        self.boxes_logic()

    def VisImgViewAlignSli_sldr_chg(self, a):
        fiji_viewer_on(self.global_h,
                       self,
                       viewer_name="xanes3D_analysis_viewer")
        self.hs["VisImgViewAlignEng text"].value = round(
            self.xanes_fit_eng_list[a["owner"].value - 1], 1)

    def VisImgViewAlign4thDim_sldr_chg(self, a):
        self.xanes_fit_4th_dim_idx = a["owner"].value
        with h5py.File(self.xanes_save_trial_reg_filename, "r") as f:
            if self.hs["VisImgViewAlignOptn drpdn"].value == "x-y-E":
                self.xanes_aligned_data = 0
                self.xanes_aligned_data = f[
                    "/registration_results/reg_results/registered_xanes3D"][:,
                                                                            self
                                                                            .
                                                                            xanes_fit_4th_dim_idx, :, :]
            elif self.hs["VisImgViewAlignOptn drpdn"].value == "y-z-E":
                self.xanes_aligned_data = 0
                self.xanes_aligned_data = f[
                    "/registration_results/reg_results/registered_xanes3D"][:, :, :,
                                                                            self
                                                                            .
                                                                            xanes_fit_4th_dim_idx]
            elif self.hs["VisImgViewAlignOptn drpdn"].value == "z-x-E":
                self.xanes_aligned_data = 0
                self.xanes_aligned_data = f[
                    "/registration_results/reg_results/registered_xanes3D"][:, :,
                                                                            self
                                                                            .
                                                                            xanes_fit_4th_dim_idx, :]
            elif self.hs["VisImgViewAlignOptn drpdn"].value == "x-y-z":
                self.xanes_aligned_data = 0
                self.xanes_aligned_data = f[
                    "/registration_results/reg_results/registered_xanes3D"][
                        self.xanes_fit_4th_dim_idx, :, :, :]
        data_state, viewer_state = fiji_viewer_state(
            self.global_h, self, viewer_name="xanes3D_analysis_viewer")
        if not viewer_state:
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes3D_analysis_viewer")
        self.xanes_fiji_aligned_data = self.global_h.ij.convert().convert(
            self.global_h.ij.dataset().create(
                self.global_h.ij.py.to_java(self.xanes_aligned_data)),
            self.global_h.ImagePlusClass,
        )
        self.global_h.xanes3D_fiji_windows["xanes3D_analysis_viewer"][
            "ip"].setImage(self.xanes_fiji_aligned_data)
        self.global_h.xanes3D_fiji_windows["xanes3D_analysis_viewer"][
            "ip"].show()
        self.global_h.xanes3D_fiji_windows["xanes3D_analysis_viewer"][
            "ip"].setTitle(
                f"{a['owner'].description} slice: {a['owner'].value}")
        self.global_h.ij.py.run_macro("""run("Collect Garbage")""")
        self.global_h.ij.py.run_macro(
            """run("Enhance Contrast", "saturated=0.35")""")

    def VisSpecViewMemMnt_chbx_chg(self, a):
        if a["owner"].value:
            self.global_h.ij.py.run_macro("""run("Monitor Memory...")""")

    def VisSpecViewInRoi_btn_clk(self, a):
        data_state, viewer_state = fiji_viewer_state(
            self.global_h, self, viewer_name="xanes3D_analysis_viewer")
        if not viewer_state:
            fiji_viewer_on(self.global_h,
                           self,
                           viewer_name="xanes3D_analysis_viewer")
        width = self.global_h.xanes3D_fiji_windows["xanes3D_analysis_viewer"][
            "ip"].getWidth()
        height = self.global_h.xanes3D_fiji_windows["xanes3D_analysis_viewer"][
            "ip"].getHeight()
        roi = [int((width - 10) / 2), int((height - 10) / 2), 10, 10]
        self.global_h.xanes3D_fiji_windows["xanes3D_analysis_viewer"][
            "ip"].setRoi(roi[0], roi[1], roi[2], roi[3])
        self.global_h.ij.py.run_macro("""run("Plot Z-axis Profile")""")
        self.global_h.ij.py.run_macro(
            """Plot.setStyle(0, "black,none,1.0,Connected Circles")""")
        self.global_h.xanes3D_fiji_windows["analysis_viewer_z_plot_viewer"][
            "ip"] = self.global_h.WindowManager.getCurrentImage()
        self.global_h.xanes3D_fiji_windows["analysis_viewer_z_plot_viewer"][
            "fiji_id"] = self.global_h.WindowManager.getIDList()[-1]
        self.hs[
            "VisSpecView text"].value = "drag the roi box to check the spectrum at different locations ..."
        self.boxes_logic()
