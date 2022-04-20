#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:33:11 2020

@author: xiao
"""

import os, h5py, json, time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from ipywidgets import widgets, GridspecLayout
from collections import OrderedDict

from .gui_components import (SelectFilesButton, get_handles, 
                            enable_disable_boxes, check_file_availability, 
                            get_raw_img_info, restart, fiji_viewer_state,
                            fiji_viewer_on, fiji_viewer_off, create_widget)
from ..utils.tomo_recon_tools import (FILTERLIST, TOMO_RECON_PARAM_DICT, 
                                      run_engine, read_data)
from ..utils.io import (data_reader, tomo_h5_reader,
                        data_info, tomo_h5_info)


FILTER_PARAM_DICT = OrderedDict({"phase retrieval": {0: ["filter", ["paganin", "bronnikov"], "filter: filter type used in phase retrieval"],
                                                     1: ["pad", ["True", "False"], "pad: boolean, if pad the data before phase retrieval filtering"],
                                                     6: ["pixel_size", 6.5e-5, "pixel_size: in cm unit"],
                                                     7: ["dist", 15.0, "dist: sample-detector distance in cm"],
                                                     8: ["energy", 35.0, "energy: x-ray energy in keV"],
                                                     9: ["alpha", 1e-2, "alpha: beta/delta, wherr n = (1-delta + i*beta) is x-ray rafractive index in the sample"]},

                                 "flatting bkg": {6: ["air", 30, "air: number of pixels on the both sides of projection images where is sample free region. This region will be used to correct background nonunifromness"]},

                                 "remove cupping": {6: ["cc", 0.5, "cc: constant that is subtracted from the logrithm of the normalized images. This is for correcting the cup-like background in the case when the sample size is much larger than the image view"]},

                                 "stripe_removal: vo": {6: ["snr", 3, "snr: signal-to-noise ratio"],
                                                        7: ["la_size", 81, "la_size: large ring's width in pixel"],
                                                        8:["sm_size", 21, "sm_size: small ring's width in pixel"]},

                                 "stripe_removal: ti": {6: ["nblock", 1, "nblock: "],
                                                        7: ["alpha", 5, "alpha: "]},

                                 "stripe_removal: sf": {6: ["size", 31, "size: "]},

                                 "stripe_removal: fw": {0: ["pad", ["True", "False"], "pad: boolean, if padding data before filtering"],
                                                        1: ["wname", ["db5", "db1", "db2", "db3", "sym2", "sym6", "haar", "gaus1", "gaus2", "gaus3", "gaus4"], "wname: wavelet name"],
                                                        6: ["level", 6, "level: how many of level of wavelet transforms"],
                                                        7: ["sigma", 2, "sigma: sigam of gaussian filter in image Fourier space"]},

                                 "denoise: wiener": {0: ["reg", ["None"], "reg: The regularisation operator. The Laplacian by default. It can be an impulse response or a transfer function, as for the psf"],
                                                     1: ["is_real", ["True", "False"], "is_real: "],
                                                     2: ["clip", ["True", "False"], "clip: True by default. If true, pixel values of the result above 1 or under -1 are thresholded for skimage pipeline compatibility"],
                                                     6: ["psf", 2, "psf: The impulse response (input image’s space) or the transfer function (Fourier space). Both are accepted. The transfer function is automatically recognized as being complex (np.iscomplexobj(psf))"],
                                                     7: ["balance", 0.3, "balance: "]},

                                 "denoise: unsupervised_wiener": {0: ["reg", ["None"], "reg: The regularisation operator. The Laplacian by default. It can be an impulse response or a transfer function, as for the psf. Shape constraint is the same as for the psf parameter"],
                                                                  1: ["is_real", ["True", "False"], "is_real: True by default. Specify if psf and reg are provided with hermitian hypothesis, that is only half of the frequency plane is provided (due to the redundancy of Fourier transform of real signal). It’s apply only if psf and/or reg are provided as transfer function. For the hermitian property see uft module or np.fft.rfftn"],
                                                                  2: ["clip", ["True", "False"], "clip: True by default. If True, pixel values of the result above 1 or under -1 are thresholded for skimage pipeline compatibility"],
                                                                  3: ["user_params", ["None"], "user_params: Dictionary of parameters for the Gibbs sampler. See below"],
                                                                  6: ["psf", 2, "psf: Point Spread Function. This is assumed to be the impulse response (input image space) if the data-type is real, or the transfer function (Fourier space) if the data-type is complex. There is no constraints on the shape of the impulse response. The transfer function must be of shape (M, N) if is_real is True, (M, N // 2 + 1) otherwise (see np.fft.rfftn)"]},

                                 "denoise: denoise_nl_means": {0: ["multichannel", ["False", "True"], "multichannel: Whether the last axis of the image is to be interpreted as multiple channels or another spatial dimension"],
                                                               1: ["fast_mode", ["True", "False"], "fast_mode: If True (default value), a fast version of the non-local means algorithm is used. If False, the original version of non-local means is used. See the Notes section for more details about the algorithms"],
                                                               6: ["patch_size", 5, "patch_size: Size of patches used for denoising"],
                                                               7: ["patch_distance", 7, "patch_distance: Maximal distance in pixels where to search patches used for denoising"],
                                                               8: ["h", 0.1, "h: Cut-off distance (in gray levels). The higher h, the more permissive one is in accepting patches. A higher h results in a smoother image, at the expense of blurring features. For a Gaussian noise of standard deviation sigma, a rule of thumb is to choose the value of h to be sigma of slightly less"],
                                                               9: ["sigma", 0.05, "sigma: The standard deviation of the (Gaussian) noise. If provided, a more robust computation of patch weights is computed that takes the expected noise variance into account (see Notes below)"]},

                                 "denoise: denoise_tv_bregman": {0: ["multichannel", ["False", "True"], "multichannel: Apply total-variation denoising separately for each channel. This option should be true for color images, otherwise the denoising is also applied in the channels dimension"],
                                                                 1: ["isotrophic", ["True", "False"], "isotrophic: Switch between isotropic and anisotropic TV denoisin"],
                                                                 6: ["weight", 1.0, "weight: Denoising weight. The smaller the weight, the more denoising (at the expense of less similarity to the input). The regularization parameter lambda is chosen as 2 * weight"],
                                                                 7: ["max_iter", 100, "max_iter: Maximal number of iterations used for the optimization"],
                                                                 8: ["eps", 0.001, "eps: Relative difference of the value of the cost function that determines the stop criterion."]},

                                 "denoise: denoise_tv_chambolle": {0: ["multichannel", ["False", "True"], "multichannel: Apply total-variation denoising separately for each channel. This option should be true for color images, otherwise the denoising is also applied in the channels dimension"],
                                                                   6: ["weight", 0.1, "weight: Denoising weight. The greater weight, the more denoising (at the expense of fidelity to input)."],
                                                                   7: ["n_iter_max", 100, "n_iter_max: Maximal number of iterations used for the optimization"],
                                                                   8: ["eps", 0.002, "eps: Relative difference of the value of the cost function that determines the stop criterion"]},

                                 "denoise: denoise_bilateral": {0: ["win_size", ["None"], "win_size: Window size for filtering. If win_size is not specified, it is calculated as max(5, 2 * ceil(3 * sigma_spatial) + 1)"],
                                                                1: ["sigma_color", ["None"], "sigma_color: Standard deviation for grayvalue/color distance (radiometric similarity). A larger value results in averaging of pixels with larger radiometric differences. Note, that the image will be converted using the img_as_float function and thus the standard deviation is in respect to the range [0, 1]. If the value is None the standard deviation of the image will be used"],
                                                                2: ["multichannel", ["False", "True"], "multichannel: Whether the last axis of the image is to be interpreted as multiple channels or another spatial dimension"],
                                                                3: ["mode", ["constant", "edge", "symmetric", "reflect", "wrap"], "mode: How to handle values outside the image borders. See numpy.pad for detail"],
                                                                6: ["sigma_spatial", 1, "sigma_spatial: Standard deviation for range distance. A larger value results in averaging of pixels with larger spatial difference"],
                                                                7: ["bins", 10000, "bins: Number of discrete values for Gaussian weights of color filtering. A larger value results in improved accuracy"],
                                                                8: ["cval", 0, "cval: Used in conjunction with mode ‘constant’, the value outside the image boundaries"]},

                                 "denoise: denoise_wavelet": {0: ["wavelet", ["db1", "db2", "db3", "db5", "sym2", "sym6", "haar", "gaus1", "gaus2", "gaus3", "gaus4"], "wavelet: The type of wavelet to perform and can be any of the options pywt.wavelist outputs"],
                                                              1: ["mode", ["soft"], "mode: An optional argument to choose the type of denoising performed. It noted that choosing soft thresholding given additive noise finds the best approximation of the original image"],
                                                              2: ["multichannel", ["False", "True"], "multichannel: Apply wavelet denoising separately for each channel (where channels correspond to the final axis of the array)"],
                                                              3: ["convert2ycbcr", ["False", "True"], "convert2ycbcr: If True and multichannel True, do the wavelet denoising in the YCbCr colorspace instead of the RGB color space. This typically results in better performance for RGB images"],
                                                              4: ["method", ["BayesShrink"], "method: Thresholding method to be used. The currently supported methods are 'BayesShrink' [1] and 'VisuShrink' [2]. Defaults to 'BayesShrink'"],
                                                              6: ["sigma", 1, "sigma: The noise standard deviation used when computing the wavelet detail coefficient threshold(s). When None (default), the noise standard deviation is estimated via the method in [2]"],
                                                              7: ["wavelet_levels", 3, "wavelet_levels: The number of wavelet decomposition levels to use. The default is three less than the maximum number of possible decomposition levels"]}})

ALG_PARAM_DICT = OrderedDict({"gridrec":{0: ["filter_name", ["parzen", 'shepp', 'cosine', 'hann', 'hamming', 'ramlak', 'butterworth', 'none'], "filter_name: filter that is used in frequency space"]},
                              "sirt":{3: ["num_gridx", 1280, "num_gridx: number of the reconstructed slice image along x direction"],
                                      4: ["num_gridy", 1280, "num_gridy: number of the reconstructed slice image along y direction"],
                                      5: ["num_inter", 10, "num_inter: number of reconstruction iterations"]},
                              "tv":{3: ["num_gridx", 1280, "num_gridx: number of the reconstructed slice image along x direction"],
                                    4: ["num_gridy", 1280, "num_gridy: number of the reconstructed slice image along y direction"],
                                    5: ["num_inter", 10, "num_inter: number of reconstruction iterations"],
                                    6: ["reg_par", 0.1, "reg_par: relaxation factor in tv regulation"]},
                              "mlem":{3: ["num_gridx", 1280, "num_gridx: number of the reconstructed slice image along x direction"],
                                      4: ["num_gridy", 1280, "num_gridy: number of the reconstructed slice image along y direction"],
                                      5: ["num_inter", 10, "num_inter: number of reconstruction iterations"]},
                              "astra":{0: ["method", ["EM_CUDA"], "method: astra reconstruction methods"],
                                       1: ["proj_type", ["cuda"], "proj_type: projection calculation options used in astra"],
                                       2: ["extra_options", ["MinConstraint"], "extra_options: extra constraints used in the reconstructions. you need to set p03 for a MinConstraint level"],
                                       3: ["extra_options_param", -0.1, "extra_options_param: parameter used together with extra_options"],
                                       4: ["num_inter", 50, "num_inter: number of reconstruction iterations"]}})

class tomo_recon_gui():
    def __init__(self, parent_h, form_sz=[650, 740]):
        self.hs = {}
        self.form_sz = form_sz
        self.global_h = parent_h
        
        if self.global_h.use_struc_h5_reader:
            self.reader = data_reader(tomo_h5_reader)
            self.info_reader = data_info(tomo_h5_info)
        else:
            pass
        
        self.tomo_raw_fn_temp = self.global_h.io_tomo_cfg['tomo_raw_fn_template']

        self.tomo_raw_data_top_dir_set = False
        self.tomo_recon_path_set = False
        self.tomo_data_center_path_set = False
        self.tomo_debug_path_set = False
        self.recon_finish = -1

        self.tomo_filepath_configured = False
        self.tomo_data_configured = False

        self.tomo_left_box_selected_flt = "phase retrieval"
        self.tomo_selected_alg = "gridrec"

        self.tomo_recon_param_dict = TOMO_RECON_PARAM_DICT

        self.tomo_raw_data_top_dir = None
        self.tomo_raw_data_file_template = None
        self.tomo_data_center_path = None
        self.tomo_recon_top_dir = None
        self.tomo_debug_top_dir = None
        self.tomo_cen_list_file = None
        self.tomo_alt_flat_file = None
        self.tomo_alt_dark_file = None
        self.tomo_wedge_ang_auto_det_ref_fn = None

        self.tomo_recon_type = "Trial Center"
        self.tomo_use_debug = False
        self.tomo_use_alt_flat = False
        self.tomo_use_alt_dark = False
        self.tomo_use_fake_flat = False
        self.tomo_use_fake_dark = False
        self.tomo_use_rm_zinger = False
        self.tomo_use_mask = True
        self.tomo_use_read_config = True
        self.tomo_use_downsample = False
        self.tomo_is_wedge = False
        self.tomo_use_wedge_ang_auto_det = False

        self.tomo_right_filter_dict = OrderedDict({0:{}})

        self.raw_proj_0 = None
        self.raw_proj_180 = None
        self.raw_proj = None
        self.load_raw_in_mem = False
        self.raw_is_in_mem = False
        self.tomo_cen_dict = {}
        self.tomo_trial_cen_dict_fn = None
        self.tomo_scan_id = 0
        self.tomo_ds_ratio = 1
        self.tomo_rot_cen = 1280
        self.tomo_cen_win_s = 1240
        self.tomo_cen_win_w = 80
        self.tomo_fake_flat_val = 1e4
        self.tomo_fake_dark_val = 100
        self.tomo_sli_s = 1280
        self.tomo_sli_e = 1300
        self.tomo_col_s = 0
        self.tomo_col_e = 100
        self.tomo_chunk_sz = 200
        self.tomo_margin = 15
        self.tomo_zinger_val = 500
        self.tomo_mask_ratio = 1
        self.tomo_wedge_missing_s = 500
        self.tomo_wedge_missing_e = 600
        self.tomo_wedge_auto_ref_col_s = 0
        self.tomo_wedge_auto_ref_col_e = 10
        self.tomo_wedge_ang_auto_det_thres = 0.1

        self.alg_param_dict = OrderedDict({})

    def build_gui(self):
        #################################################################################################################
        #                                                                                                               #
        #                                                    TOMO RECON                                                 #
        #                                                                                                               #
        #################################################################################################################
        ## ## ## define 2D_XANES_tabs layout -- start
        layout = {"border":"3px solid #FFCC00", "width":f"{self.form_sz[1]-86}px", "height":f"{self.form_sz[0]-136}px"}
        self.hs["L[0][0][0]_config_input_form"] = widgets.VBox()
        self.hs["L[0][0][1]_filter&recon_form"] = widgets.Tab()
        self.hs["L[0][0][2]_reg&review_form"] = widgets.VBox()
        self.hs["L[0][0][3]_analysis&display_form"] = widgets.VBox()
        self.hs["L[0][0][0]_config_input_form"].layout = layout
        self.hs["L[0][0][1]_filter&recon_form"].layout = layout
        self.hs["L[0][0][2]_reg&review_form"].layout = layout
        self.hs["L[0][0][3]_analysis&display_form"].layout = layout

        ## ## ## define boxes in config_input_form -- start
        ## ## ## ## define functional widget tabs in each sub-tab - configure file settings -- start
        layout = {"border":"3px solid #8855AA", "width":f"{self.form_sz[1]-92}px", "height":f"{0.42*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][0]_select_file&path_box"] = widgets.VBox()
        self.hs["L[0][0][0][0]_select_file&path_box"].layout = layout

        ## ## ## ## ## label configure file settings box
        layout = {"border":"3px solid #FFCC00", "width":f"{self.form_sz[1]-98}px", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][0][0]_select_file&path_title_box"] = widgets.HBox()
        self.hs["L[0][0][0][0][0]_select_file&path_title_box"].layout = layout
        # self.hs["L[0][0][0][0][0][0]_select_file&path_title"] = widgets.Text(value="Config Dirs & Files", disabled=True)
        self.hs["L[0][0][0][0][0][0]_select_file&path_title"] = widgets.HTML("<span style='color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);'>" + "Config Dirs & Files" + "</span>")
        layout = {"background-color":"white", "color":"cyan", "left":"39%"}
        self.hs["L[0][0][0][0][0][0]_select_file&path_title"].layout = layout
        self.hs["L[0][0][0][0][0]_select_file&path_title_box"].children = get_handles(self.hs, "L[0][0][0][0][0]_select_file&path_title_box", -1)

        ## ## ## ## ## raw h5 top directory
        layout = {"border":"3px solid #FFCC00", "width":f"{self.form_sz[1]-98}px", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][0][1]_select_raw_box"] = widgets.HBox()
        self.hs["L[0][0][0][0][1]_select_raw_box"].layout = layout
        self.hs["L[0][0][0][0][1][1]_select_raw_h5_top_dir_text"] = widgets.Text(value="Choose raw h5 top dir ...", description="", disabled=True)
        layout = {"width":"66%", "display":"inline_flex"}
        self.hs["L[0][0][0][0][1][1]_select_raw_h5_top_dir_text"].layout = layout
        self.hs["L[0][0][0][0][1][0]_select_raw_h5_top_dir_button"] = SelectFilesButton(option="askdirectory",
                                                                                        text_h=self.hs["L[0][0][0][0][1][1]_select_raw_h5_top_dir_text"])
        self.hs["L[0][0][0][0][1][0]_select_raw_h5_top_dir_button"].description = "Raw Top Dir"
        self.hs["L[0][0][0][0][1][0]_select_raw_h5_top_dir_button"].description_tooltip = "Select the top directory in which the raw h5 files are located."
        layout = {"width":"15%"}
        self.hs["L[0][0][0][0][1][0]_select_raw_h5_top_dir_button"].layout = layout
        self.hs["L[0][0][0][0][1][0]_select_raw_h5_top_dir_button"].on_click(self.L0_0_0_0_1_0_select_raw_h5_top_dir_button_click)
        self.hs["L[0][0][0][0][1]_select_raw_box"].children = get_handles(self.hs, "L[0][0][0][0][1]_select_raw_box", -1)

        ## ## ## ## ##  save recon directory
        layout = {"border":"3px solid #FFCC00", "width":f"{self.form_sz[1]-98}px", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][0][2]_select_save_recon_box"] = widgets.HBox()
        self.hs["L[0][0][0][0][2]_select_save_recon_box"].layout = layout
        self.hs["L[0][0][0][0][2][1]_select_save_recon_dir_text"] = widgets.Text(value="Select top directory where recon subdirectories are saved...", description="", disabled=True)
        layout = {"width":"66%", "display":"inline_flex"}
        self.hs["L[0][0][0][0][2][1]_select_save_recon_dir_text"].layout = layout
        self.hs["L[0][0][0][0][2][0]_select_save_recon_dir_button"] = SelectFilesButton(option="askdirectory",
                                                                                        text_h=self.hs["L[0][0][0][0][2][1]_select_save_recon_dir_text"])
        self.hs["L[0][0][0][0][2][0]_select_save_recon_dir_button"].description = "Save Rec File"
        self.hs["L[0][0][0][0][2][0]_select_save_recon_dir_button"].disabled = True
        layout = {"width":"15%"}
        self.hs["L[0][0][0][0][2][0]_select_save_recon_dir_button"].layout = layout
        self.hs["L[0][0][0][0][2][0]_select_save_recon_dir_button"].on_click(self.L0_0_0_0_2_0_select_save_recon_dir_button_click)
        self.hs["L[0][0][0][0][2]_select_save_recon_box"].children = get_handles(self.hs, "L[0][0][0][0][2]_select_save_recon_box", -1)

        ## ## ## ## ##  save data_center directory
        layout = {"border":"3px solid #FFCC00", "width":f"{self.form_sz[1]-98}px", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][0][3]_select_save_data_center_box"] = widgets.HBox()
        self.hs["L[0][0][0][0][3]_select_save_data_center_box"].layout = layout
        self.hs["L[0][0][0][0][3][1]_select_save_data_center_dir_text"] = widgets.Text(value="Select top directory where data_center will be created...", description="", disabled=True)
        layout = {"width":"66%", "display":"inline_flex"}
        self.hs["L[0][0][0][0][3][1]_select_save_data_center_dir_text"].layout = layout
        self.hs["L[0][0][0][0][3][0]_select_save_data_center_dir_button"] = SelectFilesButton(option="askdirectory",
                                                                                              text_h=self.hs["L[0][0][0][0][3][1]_select_save_data_center_dir_text"])
        self.hs["L[0][0][0][0][3][0]_select_save_data_center_dir_button"].description = "Save Data_Center"
        layout = {"width":"15%"}
        self.hs["L[0][0][0][0][3][0]_select_save_data_center_dir_button"].layout = layout
        self.hs["L[0][0][0][0][3][0]_select_save_data_center_dir_button"].on_click(self.L0_0_0_0_3_0_select_save_data_center_dir_button_click)
        self.hs["L[0][0][0][0][3]_select_save_data_center_box"].children = get_handles(self.hs, "L[0][0][0][0][3]_select_save_data_center_box", -1)

        ## ## ## ## ##  save debug directory
        layout = {"border":"3px solid #FFCC00", "width":f"{self.form_sz[1]-98}px", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][0][4]_select_save_debug_box"] = widgets.HBox()
        self.hs["L[0][0][0][0][4]_select_save_debug_box"].layout = layout
        self.hs["L[0][0][0][0][4][1]_select_save_debug_dir_text"] = widgets.Text(value="Debug is disabled...", description="", disabled=True)
        layout = {"width":"66%", "display":"inline_flex"}
        self.hs["L[0][0][0][0][4][1]_select_save_debug_dir_text"].layout = layout
        self.hs["L[0][0][0][0][4][0]_select_save_debug_dir_button"] = SelectFilesButton(option="askdirectory",
                                                                                        text_h=self.hs["L[0][0][0][0][4][1]_select_save_debug_dir_text"])
        self.hs["L[0][0][0][0][4][0]_select_save_debug_dir_button"].description = "Save Debug Dir"
        self.hs["L[0][0][0][0][4][0]_select_save_debug_dir_button"].disabled = True
        layout = {"width":"15%"}
        self.hs["L[0][0][0][0][4][0]_select_save_debug_dir_button"].layout = layout
        self.hs["L[0][0][0][0][4][2]_save_debug_checkbox"] = widgets.Checkbox(value=False,
                                                                              description="Save Debug",
                                                                              disabled=False,
                                                                              indent=False)
        layout = {"left":"1%","width":"13%", "display":"inline_flex"}
        self.hs["L[0][0][0][0][4][2]_save_debug_checkbox"].layout = layout
        self.hs["L[0][0][0][0][4][0]_select_save_debug_dir_button"].on_click(self.L0_0_0_0_4_0_select_save_debug_dir_button_click)
        self.hs["L[0][0][0][0][4][2]_save_debug_checkbox"].observe(self.L0_0_0_0_4_2_save_debug_checkbox_change, names="value")
        self.hs["L[0][0][0][0][4]_select_save_debug_box"].children = get_handles(self.hs, "L[0][0][0][0][4]_select_save_debug_box", -1)

        ## ## ## ## ## confirm file configuration
        layout = {"border":"3px solid #FFCC00", "width":f"{self.form_sz[1]-98}px", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][0][5]_select_file&path_title_comfirm_box"] = widgets.HBox()
        self.hs["L[0][0][0][0][5]_select_file&path_title_comfirm_box"].layout = layout
        self.hs["L[0][0][0][0][5][1]_confirm_file&path_text"] = widgets.Text(value="After setting directories, confirm to proceed ...", description="", disabled=True)
        layout = {"width":"66%"}
        self.hs["L[0][0][0][0][5][1]_confirm_file&path_text"].layout = layout
        self.hs["L[0][0][0][0][5][0]_confirm_file&path_button"] = widgets.Button(description="Confirm",
                                                                                 tooltip="Confirm: Confirm after you finish file configuration")
        self.hs["L[0][0][0][0][5][0]_confirm_file&path_button"].style.button_color = "darkviolet"
        self.hs["L[0][0][0][0][5][0]_confirm_file&path_button"].on_click(self.L0_0_0_0_5_0_confirm_file_path_button_click)
        layout = {"width":"15%"}
        self.hs["L[0][0][0][0][5][0]_confirm_file&path_button"].layout = layout

        self.hs["L[0][0][0][0][5][2]_file_path_options_dropdown"] = widgets.Dropdown(value="Trial Center",
                                                                                     options=["Trial Center",
                                                                                              "Vol Recon: Single",
                                                                                              "Vol Recon: Multi"],
                                                                                     description="",
                                                                                     description_tooltip="'Trial Center': doing trial recon on a single slice to find rotation center; 'Vol Recon: Single': doing volume recon of a single scan dataset; 'Vol Recon: Multi': doing volume recon of a series of  scan datasets.",
                                                                                     disabled=False)
        layout = {"width":"15%", "top":"0%"}
        self.hs["L[0][0][0][0][5][2]_file_path_options_dropdown"].layout = layout

        self.hs["L[0][0][0][0][5][2]_file_path_options_dropdown"].observe(self.L0_0_0_0_5_2_file_path_options_dropdown_change, names="value")
        self.hs["L[0][0][0][0][5]_select_file&path_title_comfirm_box"].children = get_handles(self.hs, "L[0][0][0][0][5]_select_file&path_title_comfirm_box", -1)

        self.hs["L[0][0][0][0]_select_file&path_box"].children = get_handles(self.hs, "L[0][0][0][0]_select_file&path_box", -1)
        ## ## ## ## bin widgets in hs["L[0][0][0][0]_select_file&path_box"] -- configure file settings -- end


        ## ## ## ## define widgets recon_options_box -- start
        layout = {"border":"3px solid #8855AA", "width":f"{self.form_sz[1]-94}px", "height":f"{0.56*(self.form_sz[0]-110)}px"}
        self.hs["L[0][0][0][1]_data_tab"] = widgets.Tab()
        self.hs["L[0][0][0][1]_data_tab"].layout = layout

        ## ## ## ## ## define sub-tabs in data_tab -- start
        layout = {"border":"3px solid #FFCC00", "left":"-8px", "width":f"{self.form_sz[1]-114}px", "top":"-8px", "height":f"{0.49*(self.form_sz[0]-116)}px"}
        self.hs["L[0][0][0][1][0]_data_config_tab"] = widgets.VBox()
        self.hs["L[0][0][0][1][0]_data_config_tab"].layout = layout
        self.hs["L[0][0][0][1][1]_alg_config_tab"] = widgets.VBox()
        self.hs["L[0][0][0][1][1]_alg_config_tab"].layout = layout
        self.hs["L[0][0][0][1][2]_data_info_tab"] = widgets.VBox()
        self.hs["L[0][0][0][1][2]_data_info_tab"].layout = layout
        self.hs["L[0][0][0][1][3]_data_preview_tab"] = widgets.VBox()
        self.hs["L[0][0][0][1][3]_data_preview_tab"].layout = layout
        self.hs["L[0][0][0][1]_data_tab"].children = get_handles(self.hs, "L[0][0][0][1]_data_tab", -1)
        self.hs["L[0][0][0][1]_data_tab"].set_title(0, "Data Config")
        self.hs["L[0][0][0][1]_data_tab"].set_title(1, "Algorithm Config")
        self.hs["L[0][0][0][1]_data_tab"].set_title(2, "Data Info")
        self.hs["L[0][0][0][1]_data_tab"].set_title(3, "Data Preview")
        ## ## ## ## ## define sub-tabs in data_tab -- end

        ## ## ## ## ## ## config data parameters in data_config_box in data_config_tab -- start
        layout = {"border":"3px solid #FFCC00", "width":f"{self.form_sz[1]-120}px", "height":f"{0.49*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][1][0][1]_data_config_box"] = widgets.VBox()
        self.hs["L[0][0][0][1][0][1]_data_config_box"].layout = layout

        ## ## ## ## ## ## ## config sub-boxes in data_config_box -- start
        # layout = {"border":"3px solid #FFCC00", "width":f"{self.form_sz[1]-126}px", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        layout = {"border":"3px solid #FFCC00", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][1][0][1][0]_recon_config_box0"] = widgets.HBox()
        self.hs["L[0][0][0][1][0][1][0]_recon_config_box0"].layout = layout
        # self.hs["L[0][0][0][1][0][1][0][0]_scan_id_text"] = widgets.IntText(value=0, description="Scan id", disabled=True)
        self.hs["L[0][0][0][1][0][1][0][0]_scan_id_dropdown"] = widgets.Dropdown(value=0, options=[0], description="Scan id", disabled=True)
        layout = {"width":"19%"}
        self.hs["L[0][0][0][1][0][1][0][0]_scan_id_dropdown"].layout = layout
        self.hs["L[0][0][0][1][0][1][0][1]_rot_cen_text"] = widgets.BoundedFloatText(value=1280.0, min=0, max=2500, description="Center", disabled=True)
        layout = {"width":"19%"}
        self.hs["L[0][0][0][1][0][1][0][1]_rot_cen_text"].layout = layout
        self.hs["L[0][0][0][1][0][1][0][2]_cen_win_left_text"] = widgets.BoundedIntText(value=1240, min=0, max=2500, description="Cen Win L", disabled=True)
        layout = {"width":"19%"}
        self.hs["L[0][0][0][1][0][1][0][2]_cen_win_left_text"].layout = layout
        self.hs["L[0][0][0][1][0][1][0][3]_cen_win_wz_text"] = widgets.BoundedIntText(value=80, min=1, max=200, description="Cen Win W", disabled=True)
        layout = {"width":"19%"}
        self.hs["L[0][0][0][1][0][1][0][3]_cen_win_wz_text"].layout = layout

        self.hs["L[0][0][0][1][0][1][0][4]_cen_list_button"] = SelectFilesButton(option="askopenfilename",
                                                                                 **{"open_filetypes": (("txt files", "*.txt"),("json files", "*.json"))})
        layout = {"width":"15%", "height":"85%", "visibility":"hidden"}
        self.hs["L[0][0][0][1][0][1][0][4]_cen_list_button"].layout = layout
        self.hs["L[0][0][0][1][0][1][0][4]_cen_list_button"].disabled = True
        
        self.hs["L[0][0][0][1][0][1][0][5]_use_config_checkbox"] = widgets.Checkbox(value=False, 
                                                                                    description="Use", 
                                                                                    description_tooltip='Use configuration read from the file',
                                                                                    disabled=True,
                                                                                    layout={"width":"7%", "visibility":"hidden"},
                                                                                    indent=False)

        self.hs["L[0][0][0][1][0][1][0]_recon_config_box0"].children = get_handles(self.hs, "L[0][0][0][1][0][1][0]_data_preprocessing_options_box0", -1)
        self.hs["L[0][0][0][1][0][1][0][0]_scan_id_dropdown"].observe(self.L0_0_0_1_0_1_0_0_scan_id_dropdown_change, names="value")
        self.hs["L[0][0][0][1][0][1][0][1]_rot_cen_text"].observe(self.L0_0_0_1_0_1_0_1_rot_cen_text_change, names="value")
        self.hs["L[0][0][0][1][0][1][0][2]_cen_win_left_text"].observe(self.L0_0_0_1_0_1_0_2_cen_win_left_text_change, names="value")
        self.hs["L[0][0][0][1][0][1][0][3]_cen_win_wz_text"].observe(self.L0_0_0_1_0_1_0_3_cen_win_wz_text_change, names="value")
        self.hs["L[0][0][0][1][0][1][0][4]_cen_list_button"].on_click(self.L0_0_0_1_0_1_0_4_cen_list_button_click)
        self.hs["L[0][0][0][1][0][1][0][5]_use_config_checkbox"].observe(self.L0_0_0_1_0_1_0_5_use_config_checkbox_change, names='value')

        layout = {"border":"3px solid #FFCC00", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][1][0][1][3]_roi_config_box3"] = widgets.HBox()
        self.hs["L[0][0][0][1][0][1][3]_roi_config_box3"].layout = layout
        self.hs["L[0][0][0][1][0][1][3][0]_sli_start_text"] = widgets.BoundedIntText(value=1280, min=0, max=2100, description="Sli Start", disabled=True)
        layout = {"width":"19%"}
        self.hs["L[0][0][0][1][0][1][3][0]_sli_start_text"].layout = layout
        self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"] = widgets.BoundedIntText(value=1300, min=0, max=2200, description="Sli End", disabled=True)
        layout = {"width":"19%"}
        self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].layout = layout
        self.hs["L[0][0][0][1][0][1][3][2]_col_start_text"] = widgets.BoundedIntText(value=0, min=0, max=400, description="Col Start", disabled=True)
        layout = {"width":"19%"}
        self.hs["L[0][0][0][1][0][1][3][2]_col_start_text"].layout = layout
        self.hs["L[0][0][0][1][0][1][3][3]_col_end_text"] = widgets.BoundedIntText(value=10, min=0, max=400, description="Col_End", disabled=True)
        layout = {"width":"19%"}
        self.hs["L[0][0][0][1][0][1][3][3]_col_end_text"].layout = layout
        self.hs["L[0][0][0][1][0][1][3][4]_downsample_ratio_text"] = widgets.BoundedFloatText(value=1, description="Down Sam R", min=0.1, max=1.0, step=0.1, disabled=True)
        layout = {"width":"19%"}
        self.hs["L[0][0][0][1][0][1][3][4]_downsample_ratio_text"].layout = layout
        
        self.hs["L[0][0][0][1][0][1][3]_roi_config_box3"].children = get_handles(self.hs, "L[0][0][0][1][0][1][3]_roi_config_box3", -1)
        self.hs["L[0][0][0][1][0][1][3][0]_sli_start_text"].observe(self.L0_0_0_1_0_1_3_0_sli_start_text_change, names="value")
        self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].observe(self.L0_0_0_1_0_1_3_1_sli_end_text_change, names="value")
        self.hs["L[0][0][0][1][0][1][3][2]_col_start_text"].observe(self.L0_0_0_1_0_1_3_2_col_start_text_change, names="value")
        self.hs["L[0][0][0][1][0][1][3][3]_col_end_text"].observe(self.L0_0_0_1_0_1_3_3_col_end_change, names="value")
        self.hs["L[0][0][0][1][0][1][3][4]_downsample_ratio_text"].observe(self.L0_0_0_1_0_1_3_4_downsample_ratio_text_change, names='value')

        layout = {"border":"3px solid #FFCC00", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][1][0][1][1]_alt_flat/dark_options_box1"] = widgets.HBox()
        self.hs["L[0][0][0][1][0][1][1]_alt_flat/dark_options_box1"].layout = layout
        layout = {"width":"24%"}
        self.hs["L[0][0][0][1][0][1][1][0]_use_alt_flat_checkbox"] = widgets.Checkbox(value=False, 
                                                                                      description="Alt Flat", 
                                                                                      disabled=True,
                                                                                      indent=False)
        self.hs["L[0][0][0][1][0][1][1][0]_use_alt_flat_checkbox"].layout = layout
        layout = {"width":"15%"}
        self.hs["L[0][0][0][1][0][1][1][1]_alt_flat_file_button"] = SelectFilesButton(option="askopenfilename",
                                                                                    **{"open_filetypes": (("h5 files", "*.h5"),)})
        self.hs["L[0][0][0][1][0][1][1][1]_alt_flat_file_button"].description = "Alt Flat File"
        self.hs["L[0][0][0][1][0][1][1][1]_alt_flat_file_button"].disabled = True
        self.hs["L[0][0][0][1][0][1][1][1]_alt_flat_file_button"].layout = layout
        layout = {'left':'9%', "width":"24%"}
        self.hs["L[0][0][0][1][0][1][1][2]_use_alt_dark_checkbox"] = widgets.Checkbox(value=False, 
                                                                                      description="Alt Dark", 
                                                                                      disabled=True,
                                                                                      indent=False)
        self.hs["L[0][0][0][1][0][1][1][2]_use_alt_dark_checkbox"].layout = layout
        layout = {'left':'9%', "width":"15%"}
        self.hs["L[0][0][0][1][0][1][1][3]_alt_dark_file_button"] = SelectFilesButton(option="askopenfilename",
                                                                                    **{"open_filetypes": (("h5 files", "*.h5"),)})
        self.hs["L[0][0][0][1][0][1][1][3]_alt_dark_file_button"].description = "Alt Dark File"
        self.hs["L[0][0][0][1][0][1][1][3]_alt_dark_file_button"].disabled = True
        self.hs["L[0][0][0][1][0][1][1][3]_alt_dark_file_button"].layout = layout
        
        self.hs["L[0][0][0][1][0][1][1]_alt_flat/dark_options_box1"].children = get_handles(self.hs, "L[0][0][0][1][0][1][1]_alt_flat/dark_options_box1", -1)
        self.hs["L[0][0][0][1][0][1][1][0]_use_alt_flat_checkbox"].observe(self.L0_0_0_1_0_1_1_0_use_alt_flat_checkbox_change, names="value")
        self.hs["L[0][0][0][1][0][1][1][1]_alt_flat_file_button"].observe(self.L0_0_0_1_0_1_1_1_alt_flat_file_button_change, names="value")
        self.hs["L[0][0][0][1][0][1][1][2]_use_alt_dark_checkbox"].observe(self.L0_0_0_1_0_1_1_2_use_alt_dark_checkbox_change, names="value")
        self.hs["L[0][0][0][1][0][1][1][3]_alt_dark_file_button"].observe(self.L0_0_0_1_0_1_1_3_alt_dark_file_button_change, names="value")

        layout = {"border":"3px solid #FFCC00", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][1][0][1][2]_fake_flat/dark_options_box2"] = widgets.HBox()
        self.hs["L[0][0][0][1][0][1][2]_fake_flat/dark_options_box2"].layout = layout
        layout = {"width":"19%"}
        self.hs["L[0][0][0][1][0][1][2][0]_use_fake_flat_checkbox"] = widgets.Checkbox(value=False,
                                                                                       description="Fake Flat",
                                                                                       disabled=True,
                                                                                       indent=False)
        self.hs["L[0][0][0][1][0][1][2][0]_use_fake_flat_checkbox"].layout = layout
        layout = {"width":"19%"}
        self.hs["L[0][0][0][1][0][1][2][1]_fake_flat_val_text"] = widgets.BoundedFloatText(value=10000.0,
                                                                                           description="Flat Val",
                                                                                            min=100,
                                                                                            max=65000,
                                                                                            disabled=True)
        self.hs["L[0][0][0][1][0][1][2][1]_fake_flat_val_text"].layout = layout
        layout = {"left":"10%", "width":"19%"}
        self.hs["L[0][0][0][1][0][1][2][2]_use_fake_dark_checkbox"] = widgets.Checkbox(value=False,
                                                                                       description="Fake Dark",
                                                                                       disabled=True,
                                                                                       indent=False)
        self.hs["L[0][0][0][1][0][1][2][2]_use_fake_dark_checkbox"].layout = layout
        layout = {"left":"19%", "width":"19%"}
        self.hs["L[0][0][0][1][0][1][2][3]_fake_dark_val_text"] = widgets.BoundedFloatText(value=100.0,
                                                                                           description="Dark Val",
                                                                                           min=0,
                                                                                           max=500,
                                                                                           disabled=True)
        self.hs["L[0][0][0][1][0][1][2][3]_fake_dark_val_text"].layout = layout
        
        self.hs["L[0][0][0][1][0][1][2][0]_use_fake_flat_checkbox"].observe(self.L0_0_0_1_0_1_2_0_use_fake_flat_checkbox_change, names="value")
        self.hs["L[0][0][0][1][0][1][2][1]_fake_flat_val_text"].observe(self.L0_0_0_1_0_1_2_1_fake_flat_val_text_change, names="value")
        self.hs["L[0][0][0][1][0][1][2][2]_use_fake_dark_checkbox"].observe(self.L0_0_0_1_0_1_2_2_use_fake_dark_checkbox_change, names="value")
        self.hs["L[0][0][0][1][0][1][2][3]_fake_dark_val_text"].observe(self.L0_0_0_1_0_1_2_3_fake_dark_val_text_change, names="value")
        self.hs["L[0][0][0][1][0][1][2]_fake_flat/dark_options_box2"].children = get_handles(self.hs, "L[0][0][0][1][0][1][2]_data_preprocessing_options_box2", -1)

        layout = {"border":"3px solid #FFCC00", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][1][0][1][4]_misc_options_box4"] = widgets.HBox()
        self.hs["L[0][0][0][1][0][1][4]_misc_options_box4"].layout = layout
        layout = {"width":"19%"}
        self.hs["L[0][0][0][1][0][1][4][0]_rm_zinger_checkbox"] = widgets.Checkbox(value=False,
                                                                                   description="Rm Zinger",
                                                                                   disabled=True,
                                                                                   indent=False)
        self.hs["L[0][0][0][1][0][1][4][0]_rm_zinger_checkbox"].layout = layout
        layout = {"width":"19%"}
        self.hs["L[0][0][0][1][0][1][4][1]_zinger_level_text"] = widgets.BoundedFloatText(value=500.0,
                                                                                          description='Zinger Lev',
                                                                                          min=10,
                                                                                          max=1000,
                                                                                          disabled=True)
        self.hs["L[0][0][0][1][0][1][4][1]_zinger_level_text"].layout = layout
        layout = {"left":"10%", "width":"19%"}
        self.hs["L[0][0][0][1][0][1][4][2]_use_mask_checkbox"] = widgets.Checkbox(value=True,
                                                                                  description="Use Mask",
                                                                                  disabled=True,
                                                                                  indent=False)
        self.hs["L[0][0][0][1][0][1][4][2]_use_mask_checkbox"].layout = layout
        layout = {"left":"19%", "width":"19%"}
        self.hs["L[0][0][0][1][0][1][4][3]_mask_ratio_text"] = widgets.BoundedFloatText(value=1,
                                                                                        description='Mask R',
                                                                                         min=0,
                                                                                         max=1,
                                                                                         step=0.05,
                                                                                         disabled=True)
        self.hs["L[0][0][0][1][0][1][4][3]_mask_ratio_text"].layout = layout
        
        self.hs["L[0][0][0][1][0][1][4][0]_rm_zinger_checkbox"].observe(self.L0_0_0_1_0_1_4_0_rm_zinger_checkbox_change, names="value")
        self.hs["L[0][0][0][1][0][1][4][1]_zinger_level_text"].observe(self.L0_0_0_1_0_1_4_1_zinger_level_text_change, names="value")
        self.hs["L[0][0][0][1][0][1][4][2]_use_mask_checkbox"].observe(self.L0_0_0_1_0_1_4_2_use_mask_checkbox_change, names="value")
        self.hs["L[0][0][0][1][0][1][4][3]_mask_ratio_text"].observe(self.L0_0_0_1_0_1_4_3_mask_ratio_text_change, names="value")
        self.hs["L[0][0][0][1][0][1][4]_misc_options_box4"].children = get_handles(self.hs, "L[0][0][0][1][0][1][4]_misc_options_box4", -1)

        layout = {"border":"3px solid #FFCC00", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][1][0][1][5]_wedge_options_box0"] = widgets.HBox()
        self.hs["L[0][0][0][1][0][1][5]_wedge_options_box0"].layout = layout
        layout = {"width":"10%"}
        self.hs["L[0][0][0][1][0][1][5][0]_is_wedge_checkbox"] = widgets.Checkbox(value=False,
                                                                                  description="Is Wedge",
                                                                                  disabled=True,
                                                                                  indent=False)
        self.hs["L[0][0][0][1][0][1][5][0]_is_wedge_checkbox"].layout = layout
        layout = {"width":"20%"}
        self.hs["L[0][0][0][1][0][1][5][2]_missing_start_text"] = widgets.BoundedIntText(value=500,
                                                                                         min=0,
                                                                                         max=5000,
                                                                                         description="Miss S",
                                                                                         disabled=True,)
        self.hs["L[0][0][0][1][0][1][5][2]_missing_start_text"].layout = layout
        layout = {"width":"20%"}
        self.hs["L[0][0][0][1][0][1][5][3]_missing_end_text"] =  widgets.BoundedIntText(value=600,
                                                                                        min=0,
                                                                                        max=5000,
                                                                                        description="Miss E",
                                                                                        disabled=True)
        self.hs["L[0][0][0][1][0][1][5][3]_missing_end_text"].layout = layout
        layout = {"width":"20%"}
        self.hs["L[0][0][0][1][0][1][5][4]_auto_detect_checkbox"] = widgets.Checkbox(value=True,
                                                                                     description="Auto Det",
                                                                                     disabled=True,
                                                                                     indent=True)
        self.hs["L[0][0][0][1][0][1][5][4]_auto_detect_checkbox"].layout = layout
        layout = {"left":"5%", "width":"20%"}
        self.hs["L[0][0][0][1][0][1][5][5]_auto_thres_text"] = widgets.BoundedFloatText(value=200,
                                                                                        min=0,
                                                                                        max=1000,
                                                                                        description="Auto Thres",
                                                                                        disabled=True)
        self.hs["L[0][0][0][1][0][1][5][5]_auto_thres_text"].layout = layout
        
        self.hs["L[0][0][0][1][0][1][5][0]_is_wedge_checkbox"].observe(self.L0_0_1_0_0_1_0_0_is_wedge_checkbox_change, names="value")
        # self.hs["L[0][0][0][1][0][1][5][1]_blankat_dropdown"].observe(self.L0_0_1_0_0_1_0_1_blankat_dropdown_change, names="value")
        self.hs["L[0][0][0][1][0][1][5][2]_missing_start_text"].observe(self.L0_0_1_0_0_1_0_2_missing_start_text_change, names="value")
        self.hs["L[0][0][0][1][0][1][5][3]_missing_end_text"].observe(self.L0_0_1_0_0_1_0_3_missing_end_text_change, names="value")
        self.hs["L[0][0][0][1][0][1][5][4]_auto_detect_checkbox"].observe(self.L0_0_0_1_0_1_5_4_auto_detect_checkbox_change, names="value")
        self.hs["L[0][0][0][1][0][1][5][5]_auto_thres_text"].observe(self.L0_0_0_1_0_1_5_5_auto_thres_text_change, names="value")
        self.hs["L[0][0][0][1][0][1][5]_wedge_options_box0"].children = get_handles(self.hs, "L[0][0][0][1][0][1][5]_wedge_options_box0", -1)

        layout = {"border":"3px solid #FFCC00", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][1][0][1][6]_wedge_options_box1"] = widgets.HBox()
        self.hs["L[0][0][0][1][0][1][6]_wedge_options_box1"].layout = layout
        
        layout = {"width":"15%"}
        self.hs["L[0][0][0][1][0][1][6][2]_auto_ref_fn_button"] = SelectFilesButton(option="askopenfilename",
                                                                                    **{"open_filetypes": (("h5 files", "*.h5"),)})
        self.hs["L[0][0][0][1][0][1][6][2]_auto_ref_fn_button"].layout = layout
        self.hs["L[0][0][0][1][0][1][6][2]_auto_ref_fn_button"].disabled = True
        self.hs["L[0][0][0][1][0][1][6][3]_auto_ref_sli_slider"] = create_widget('IntSlider', {'width':'40%'}, **{'description':'slice #', 'min':0, 'max':10, 'value':0, 'disabled':True})
        
        self.hs["L[0][0][0][1][0][1][6][4]_auto_ref_col_start_text"] = widgets.BoundedIntText(value=0, min=0, max=400, description="W Col_Start", disabled=True)
        layout = {"left":"2.5%", "width":"19%"}
        self.hs["L[0][0][0][1][0][1][6][4]_auto_ref_col_start_text"].layout = layout
        self.hs["L[0][0][0][1][0][1][6][5]_auto_ref_col_end_text"] = widgets.BoundedIntText(value=10, min=1, max=401, description="W Col_End", disabled=True)
        layout = {"left":"2.5%", "width":"19%"}
        self.hs["L[0][0][0][1][0][1][6][5]_auto_ref_col_end_text"].layout = layout
        
        self.hs["L[0][0][0][1][0][1][6][2]_auto_ref_fn_button"].on_click(self.L0_0_1_0_0_1_1_2_auto_ref_fn_button_change)
        self.hs["L[0][0][0][1][0][1][6][3]_auto_ref_sli_slider"].observe(self.L0_0_0_1_0_1_6_3_auto_ref_sli_slider_change, names='value')
        self.hs["L[0][0][0][1][0][1][6][4]_auto_ref_col_start_text"].observe(self.L0_0_0_1_0_1_6_4_auto_ref_col_start_text_change, names='value')
        self.hs["L[0][0][0][1][0][1][6][5]_auto_ref_col_end_text"].observe(self.L0_0_0_1_0_1_6_5_auto_ref_col_end_text_change, names='value')
        self.hs["L[0][0][0][1][0][1][6]_wedge_options_box1"].children = get_handles(self.hs, "L[0][0][0][1][0][1][6]_wedge_options_box1", -1)
        ## ## ## ## ## ## ## config sub-boxes in data_config_box -- end

        self.hs["L[0][0][0][1][0][1]_data_config_box"].children = get_handles(self.hs, "L[0][0][0][1][0][1]_recon_options_box", -1)
        ## ## ## ## ## ## config data parameters in data_config_box in data_config_tab -- end

        self.hs["L[0][0][0][1][0]_data_config_tab"].children = get_handles(self.hs, "L[0][0][0][1][0]_recon_options_box1", -1)
        ## ## ## ## ## config data_config_tab -- end



        ## ## ## ## ## ## config alg_config_box in alg_config tab -- start
        layout = {"border":"3px solid #8855AA", "width":f"{self.form_sz[1]-121}px", "height":f"{0.21*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][1][1][0]_alg_config_box"] = widgets.VBox()
        self.hs["L[0][0][0][1][1][0]_alg_config_box"].layout = layout

        # ## ## ## ## ## ## ## label alg_config_box -- start
        # layout = {"justify-content":"center", "align-items":"center","align-contents":"center","border":"3px solid #FFCC00", "width":f"{self.form_sz[1]-127}px", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        # self.hs["L[0][0][0][1][1][0][0]_alg_config_title_box"] = widgets.HBox()
        # self.hs["L[0][0][0][1][1][0][0]_alg_config_title_box"].layout = layout
        # self.hs["L[0][0][0][1][1][0][0][0]_alg_config_title"] = widgets.HTML("<span style='color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);'>" + "Alg Config" + "</span>")
        # layout = {"left":"43.25%", "background-color":"white", "color":"cyan"}
        # self.hs["L[0][0][0][1][1][0][0][0]_alg_config_title"].layout = layout
        # self.hs["L[0][0][0][1][1][0][0]_alg_config_title_box"].children = get_handles(self.hs, "L[0][0][0][1][1][0][0]_alg_config_title_box", -1)
        # ## ## ## ## ## ## ## label alg_config_box -- end

        layout = {"border":"3px solid #FFCC00", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][1][1][0][1]_alg_options_box0"] = widgets.HBox()
        self.hs["L[0][0][0][1][1][0][1]_alg_options_box0"].layout = layout
        layout = {"width":"24%"}
        self.hs["L[0][0][0][1][1][0][1][0]_alg_options_dropdown"] = widgets.Dropdown(value="gridrec",
                                                                                  options=["gridrec", "sirt", "tv", "mlem", "astra"],
                                                                                  description="algs",
                                                                                  disabled=True)
        self.hs["L[0][0][0][1][1][0][1][0]_alg_options_dropdown"].layout = layout
        layout = {"width":"23.5%"}
        self.hs["L[0][0][0][1][1][0][1][1]_alg_p00_dropdown"] = widgets.Dropdown(value="",
                                                                                  options=[""],
                                                                                  description="p00",
                                                                                  disabled=True)
        self.hs["L[0][0][0][1][1][0][1][1]_alg_p00_dropdown"].layout = layout
        layout = {"width":"23.5%"}
        self.hs["L[0][0][0][1][1][0][1][2]_alg_p01_dropdown"] = widgets.Dropdown(value="",
                                                                                  options=[""],
                                                                                  description="p01",
                                                                                  disabled=True)
        self.hs["L[0][0][0][1][1][0][1][2]_alg_p01_dropdown"].layout = layout
        layout = {"width":"23.5%"}
        self.hs["L[0][0][0][1][1][0][1][3]_alg_p02_dropdown"] = widgets.Dropdown(value="",
                                                                                options=[""],
                                                                                description="p02",
                                                                                disabled=True)
        self.hs["L[0][0][0][1][1][0][1][3]_alg_p02_dropdown"].layout = layout
        self.hs["L[0][0][0][1][1][0][1][0]_alg_options_dropdown"].observe(self.L0_0_0_1_1_0_1_0_alg_options_dropdown_change, names="value")
        self.hs["L[0][0][0][1][1][0][1][1]_alg_p00_dropdown"].observe(self.L0_0_0_1_1_0_1_1_alg_p00_dropdown_change, names="value")
        self.hs["L[0][0][0][1][1][0][1][2]_alg_p01_dropdown"].observe(self.L0_0_0_1_1_0_1_2_alg_p01_dropdown_change, names="value")
        self.hs["L[0][0][0][1][1][0][1][3]_alg_p02_dropdown"].observe(self.L0_0_0_1_1_0_1_3_alg_p02_dropdown_change, names="value")
        self.hs["L[0][0][0][1][1][0][1]_alg_options_box0"].children = get_handles(self.hs, "L[0][0][0][1][1][0][1]_alg_options_box0", -1)

        layout = {"border":"3px solid #FFCC00", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][1][1][0][2]_alg_options_box1"] = widgets.HBox()
        self.hs["L[0][0][0][1][1][0][2]_alg_options_box1"].layout = layout
        layout = {"width":"23.5%"}
        self.hs["L[0][0][0][1][1][0][2][0]_alg_p03_text"] = widgets.FloatText(value=0,
                                                                              description="p03",
                                                                              disabled=True)
        self.hs["L[0][0][0][1][1][0][2][0]_alg_p03_text"].layout = layout
        layout = {"width":"23.5%"}
        self.hs["L[0][0][0][1][1][0][2][1]_alg_p04_text"] = widgets.FloatText(value=0,
                                                                              description="p04",
                                                                              disabled=True)
        self.hs["L[0][0][0][1][1][0][2][1]_alg_p04_text"].layout = layout
        layout = {"width":"23.5%"}
        self.hs["L[0][0][0][1][1][0][2][2]_alg_p05_text"] = widgets.FloatText(value=0,
                                                                              description="p05",
                                                                              disabled=True)
        self.hs["L[0][0][0][1][1][0][2][2]_alg_p05_text"].layout = layout
        layout = {"width":"23.5%"}
        self.hs["L[0][0][0][1][1][0][2][3]_alg_p06_text"] = widgets.FloatText(value=0.0,
                                                                              description="p06",
                                                                              disabled=True)
        self.hs["L[0][0][0][1][1][0][2][3]_alg_p06_text"].layout = layout
        self.hs["L[0][0][0][1][1][0][2][0]_alg_p03_text"].observe(self.L0_0_0_1_1_0_2_0_alg_p03_text_change, names="value")
        self.hs["L[0][0][0][1][1][0][2][1]_alg_p04_text"].observe(self.L0_0_0_1_1_0_2_1_alg_p04_text_change, names="value")
        self.hs["L[0][0][0][1][1][0][2][2]_alg_p05_text"].observe(self.L0_0_0_1_1_0_2_2_alg_p05_text_change, names="value")
        self.hs["L[0][0][0][1][1][0][2][3]_alg_p06_text"].observe(self.L0_0_0_1_1_0_2_3_alg_p06_text_change, names="value")
        self.hs["L[0][0][0][1][1][0][2]_alg_options_box1"].children = get_handles(self.hs, "L[0][0][0][1][1][0][2]_alg_options_box1", -1)

        self.hs["L[0][0][0][1][1][0]_alg_config_box"].children = get_handles(self.hs, "L[0][0][0][1][1][0]_alg_config_box", -1)
        ## ## ## ## ## ## config alg_config_box in alg_config tab -- end

        self.hs["L[0][0][0][1][1]_alg_config_tab"].children = get_handles(self.hs, "L[0][0][0][1][1]_alg_config_box", -1)
        ## ## ## ## ## define alg_config tab -- end
        
        ## ## ## ## ## define data info tab -- start
        ## ## ## ## ## ## define data info box -- start
        layout = {"border":"3px solid #FFCC00", "height":f"{0.52*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][1][2][0]_data_info_box"] = widgets.HBox()
        self.hs["L[0][0][0][1][2][0]_data_info_box"].layout = layout
        layout = {"width":"95%", "height":"90%"}
        self.hs["L[0][0][0][1][2][0][0]_data_info_text"] = widgets.Textarea(value='Data Info',
                                                                            placeholder='Data Info',
                                                                            description='Data Info',
                                                                            disabled=True)
        self.hs["L[0][0][0][1][2][0][0]_data_info_text"].layout = layout
        self.hs["L[0][0][0][1][2][0]_data_info_box"].children = get_handles(self.hs, "L[0][0][0][1][2][0]_data_info_box", -1)
        ## ## ## ## ## ## define data info box -- end
        self.hs["L[0][0][0][1][2]_data_info_tab"].children = get_handles(self.hs, "L[0][0][0][1][2]_data_info_tab", -1)
        ## ## ## ## ## define data info tab -- end
        
        ## ## ## ## ## define data_preview tab -- start
        ## ## ## ## ## ## define data_preview_box -- start
        layout = {"border":"3px solid #8855AA", "width":f"{self.form_sz[1]-121}px", "height":f"{0.21*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][1][3][0]_data_preview_box"] = widgets.VBox()
        self.hs["L[0][0][0][1][3][0]_data_preview_box"].layout = layout
        
        ## ## ## ## ## ## ## define functional box widgets in data_prevew_box3 -- start
        layout = {"border":"3px solid #FFCC00", "width":f"{self.form_sz[1]-127}px", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][1][3][0][3]_data_preview_box3"] = widgets.HBox()
        self.hs["L[0][0][0][1][3][0][3]_data_preview_box3"].layout = layout
        layout = {"width":"50%"}
        self.hs["L[0][0][0][1][3][0][3][0]_raw_proj_slider"] = widgets.IntSlider(value=0,
                                                                                         description="proj",
                                                                                         description_tooltip = 'offset the image at 180 deg to overlap with the image at 0 deg. rotation center can be determined from the offset.',
                                                                                         min=-100,
                                                                                         max=100,
                                                                                         disabled=True,
                                                                                         indent=False)
        self.hs["L[0][0][0][1][3][0][3][0]_raw_proj_slider"].layout = layout   
        layout = {"width":"20%"}
        self.hs["L[0][0][0][1][3][0][3][1]_raw_proj_all_in_mem_checkbox"] = widgets.Checkbox(description="read in mem",
                                                                                             description_tooltip = 'Optional read entire raw proj dataset into memory for display',
                                                                                             value=False,
                                                                                             disabled=True,
                                                                                             indent=False)
        self.hs["L[0][0][0][1][3][0][3][1]_raw_proj_all_in_mem_checkbox"].layout = layout 
        layout = {"width":"20%"}
        self.hs["L[0][0][0][1][3][0][3][2]_raw_proj_viewer_close_button"] = widgets.Button(description="Close/Confirm",
                                                                                    description_tooltip = 'Optional confrimation of roi (slices and columns) definition',
                                                                                    disabled=True)
        self.hs["L[0][0][0][1][3][0][3][2]_raw_proj_viewer_close_button"].layout = layout 
        self.hs["L[0][0][0][1][3][0][3][2]_raw_proj_viewer_close_button"].style.button_color = 'darkviolet'
        
        self.hs["L[0][0][0][1][3][0][3][0]_raw_proj_slider"].observe(self.L0_0_0_1_3_0_3_0_raw_proj_slider_change, names="value") 
        self.hs["L[0][0][0][1][3][0][3][1]_raw_proj_all_in_mem_checkbox"].observe(self.L0_0_0_1_3_0_3_1_raw_proj_all_in_mem_checkbox_click, names='value')
        self.hs["L[0][0][0][1][3][0][3][2]_raw_proj_viewer_close_button"].on_click(self.L0_0_0_1_3_0_3_2_raw_proj_viewer_close_button_click)
        self.hs["L[0][0][0][1][3][0][3]_data_preview_box3"].children = get_handles(self.hs, "L[0][0][0][1][3][0][3]_data_preview_box3", -1)
        ## ## ## ## ## ## ## define functional box widgets in data_prevew_box3 -- end

        ## ## ## ## ## ## ## define functional box widgets in data_prevew_box0 -- start
        layout = {"border":"3px solid #FFCC00", "width":f"{self.form_sz[1]-127}px", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][1][3][0][0]_data_preview_box0"] = widgets.HBox()
        self.hs["L[0][0][0][1][3][0][0]_data_preview_box0"].layout = layout
        layout = {"width":"50%"}
        self.hs["L[0][0][0][1][3][0][0][0]_cen_offset_range_slider"] = widgets.IntSlider(value=0,
                                                                                         description="offset",
                                                                                         description_tooltip = 'offset the image at 180 deg to overlap with the image at 0 deg. rotation center can be determined from the offset.',
                                                                                         min=-100,
                                                                                         max=100,
                                                                                         disabled=True,
                                                                                         indent=False)
        self.hs["L[0][0][0][1][3][0][0][0]_cen_offset_range_slider"].layout = layout   
        layout = {"width":"20%"}
        self.hs["L[0][0][0][1][3][0][0][1]_cen_offset_confirm_button"] = widgets.Button(description="Confirm",
                                                                                    description_tooltip = 'Optional confrimation of the rough center',
                                                                                    disabled=True)
        self.hs["L[0][0][0][1][3][0][0][1]_cen_offset_confirm_button"].layout = layout 
        self.hs["L[0][0][0][1][3][0][0][1]_cen_offset_confirm_button"].style.button_color = 'darkviolet'
        layout = {"width":"20%"}
        self.hs["L[0][0][0][1][3][0][0][2]_cen_viewer_close_button"] = widgets.Button(description="Close",
                                                                                    description_tooltip = 'Optional close the viewer window',
                                                                                    disabled=True)
        self.hs["L[0][0][0][1][3][0][0][2]_cen_viewer_close_button"].layout = layout 
        self.hs["L[0][0][0][1][3][0][0][2]_cen_viewer_close_button"].style.button_color = 'darkviolet'
        
        self.hs["L[0][0][0][1][3][0][0][0]_cen_offset_range_slider"].observe(self.L0_0_0_1_3_0_0_0_cen_offset_range_slider_change, names="value") 
        self.hs["L[0][0][0][1][3][0][0][1]_cen_offset_confirm_button"].on_click(self.L0_0_0_1_3_0_0_1_cen_offset_confirm_button_click)
        self.hs["L[0][0][0][1][3][0][0][2]_cen_viewer_close_button"].on_click(self.L0_0_0_1_3_0_0_2_cen_viewer_close_button_click)
        self.hs["L[0][0][0][1][3][0][0]_data_preview_box0"].children = get_handles(self.hs, "L[0][0][0][1][3][0][0]_data_preview_box0", -1)
        ## ## ## ## ## ## ## define functional box widgets in data_prevew_box0 -- end
        
        ## ## ## ## ## ## ## define functional box widgets in data_prevew_box1 -- start
        layout = {"border":"3px solid #FFCC00", "width":f"{self.form_sz[1]-127}px", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][1][3][0][1]_data_preview_box1"] = widgets.HBox()
        self.hs["L[0][0][0][1][3][0][1]_data_preview_box1"].layout = layout
        layout = {"width":"50%"}
        self.hs["L[0][0][0][1][3][0][1][0]_trial_cen_slider"] = widgets.IntSlider(value=0,
                                                                                         description="trial cen",
                                                                                         description_tooltip = 'offset the image at 180 deg to overlap with the image at 0 deg. rotation center can be determined from the offset.',
                                                                                         min=-100,
                                                                                         max=100,
                                                                                         disabled=True,
                                                                                         indent=False)
        self.hs["L[0][0][0][1][3][0][1][0]_trial_cen_slider"].layout = layout   
        layout = {"width":"20%"}
        self.hs["L[0][0][0][1][3][0][1][1]_trial_cen_confirm_button"] = widgets.Button(description="Confirm",
                                                                                    description_tooltip = 'Optional confrimation of the rough center',
                                                                                    disabled=True)
        self.hs["L[0][0][0][1][3][0][1][1]_trial_cen_confirm_button"].layout = layout 
        self.hs["L[0][0][0][1][3][0][1][1]_trial_cen_confirm_button"].style.button_color = 'darkviolet'
        layout = {"width":"20%"}
        self.hs["L[0][0][0][1][3][0][1][2]_trial_cen_viewer_close_button"] = widgets.Button(description="Close",
                                                                                    description_tooltip = 'Optional close the viewer window',
                                                                                    disabled=True)
        self.hs["L[0][0][0][1][3][0][1][2]_trial_cen_viewer_close_button"].layout = layout 
        self.hs["L[0][0][0][1][3][0][1][2]_trial_cen_viewer_close_button"].style.button_color = 'darkviolet'
        
        self.hs["L[0][0][0][1][3][0][1][0]_trial_cen_slider"].observe(self.L0_0_0_1_3_0_1_0_trial_cen_slider_change, names="value") 
        self.hs["L[0][0][0][1][3][0][1][1]_trial_cen_confirm_button"].on_click(self.L0_0_0_1_3_0_1_1_trial_cen_confirm_button_click)
        self.hs["L[0][0][0][1][3][0][1][2]_trial_cen_viewer_close_button"].on_click(self.L0_0_0_1_3_0_1_2_trial_cen_viewer_close_button_click)
        self.hs["L[0][0][0][1][3][0][1]_data_preview_box1"].children = get_handles(self.hs, "L[0][0][0][1][3][0][1]_data_preview_box1", -1)
        ## ## ## ## ## ## ## define functional box widgets in data_prevew_box1 -- end
        
        ## ## ## ## ## ## ## define functional box widgets in data_prevew_box2 -- start
        layout = {"border":"3px solid #FFCC00", "width":f"{self.form_sz[1]-127}px", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][0][1][3][0][2]_data_preview_box2"] = widgets.HBox()
        self.hs["L[0][0][0][1][3][0][2]_data_preview_box2"].layout = layout
        layout = {"width":"50%"}
        self.hs["L[0][0][0][1][3][0][2][0]_vol_sli_slider"] = widgets.IntSlider(value=0,
                                                                                         description="sli",
                                                                                         description_tooltip = 'offset the image at 180 deg to overlap with the image at 0 deg. rotation center can be determined from the offset.',
                                                                                         min=-100,
                                                                                         max=100,
                                                                                         disabled=True,
                                                                                         indent=False)
        self.hs["L[0][0][0][1][3][0][2][0]_vol_sli_slider"].layout = layout   
        layout = {"width":"20%"}
        self.hs["L[0][0][0][1][3][0][2][1]_vol_sli_viewer_close_button"] = widgets.Button(description="Close",
                                                                                    description_tooltip = 'Optional confrimation of the rough center',
                                                                                    disabled=True)
        self.hs["L[0][0][0][1][3][0][2][1]_vol_sli_viewer_close_button"].layout = layout 
        self.hs["L[0][0][0][1][3][0][2][1]_vol_sli_viewer_close_button"].style.button_color = 'darkviolet'
        
        self.hs["L[0][0][0][1][3][0][2][0]_vol_sli_slider"].observe(self.L0_0_0_1_3_0_2_0_vol_sli_slider_change, names="value") 
        self.hs["L[0][0][0][1][3][0][2][1]_vol_sli_viewer_close_button"].on_click(self.L0_0_0_1_3_0_2_1_vol_sli_viewer_close_button_click)
        self.hs["L[0][0][0][1][3][0][2]_data_preview_box2"].children = get_handles(self.hs, "L[0][0][0][1][3][0][2]_data_preview_box2", -1)
        ## ## ## ## ## ## ## define functional box widgets in data_prevew_box2 -- end
        
        self.hs["L[0][0][0][1][3][0]_data_preview_box"].children = get_handles(self.hs, "L[0][0][0][1][3][0]_data_preview_box", -1)
        ## ## ## ## ## ## define data_preview_box-- end
        self.hs["L[0][0][0][1][3]_data_preview_tab"].children = get_handles(self.hs, "L[0][0][0][1][3]_data_preview_tab", -1)
        ## ## ## ## ## define data_preview tab -- start


        self.hs["L[0][0][0]_config_input_form"].children = get_handles(self.hs, "L[0][0][0]_config_input_form", -1)
        ## ## ## config config_input_form -- end



        ## ## ## ## config filter&recon tab -- start
        layout = {"border":"3px solid #8855AA", "width":f"{self.form_sz[1]-122}px", "height":f"{0.88*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][1][0]_filter&recon_tab"] = widgets.VBox()
        self.hs["L[0][0][1][0]_filter&recon_tab"].layout = layout

        ## ## ## ## ## config filter_config_box -- start
        layout = {"border":"3px solid #8855AA", "height":f"{0.79*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][1][0][0]_filter_config_box"] = widgets.VBox()
        self.hs["L[0][0][1][0][0]_filter_config_box"].layout = layout
        
        ## ## ## ## ## ## label recon_box -- start
        grid_recon_chunk = GridspecLayout(2, 100, layout = {"border":"3px solid #FFCC00",
                                                            "height":f"{0.14*(self.form_sz[0]-136)}px",
                                                            "width":f"{self.hs['L[0][0][1][0][0]_filter_config_box'].layout.width}",
                                                            "grid_row_gap":"4px",
                                                            "grid_column_gap":"8px",
                                                            "align_items":"flex-start",
                                                            "justify_items":"flex-start"})
        self.hs["L[0][0][1][0][0][2]_recon_chunk_box"] = grid_recon_chunk
        
        grid_recon_chunk[0, 30:70] = widgets.HTML("<span style='color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);'>" + "Chunk&Margin" + "</span>")
        self.hs["L[0][0][1][0][0][2][0]_recon_title"] = grid_recon_chunk[0, 30:70]
        layout = {"left":f"{0.41*(int(self.hs['L[0][0][1][0]_filter&recon_tab'].layout.width.strip('px'))-6)}px", 
                  "background-color":"white", "color":"cyan", "justify_items":"center"}
        # layout = {"left":f"{int(0.405*(self.form_sz[1]-122-6))}px", "background-color":"white", "color":"cyan"}
        self.hs["L[0][0][1][0][0][2][0]_recon_title"].layout = layout
        
        grid_recon_chunk[1, :10] = widgets.BoundedIntText(description='Chunk Sz', disabled=True, 
                                                        min=1, max=2000,
                                                        layout={'width':f'{int(0.20*(self.form_sz[1]-136))}px', 'height':'90%'},
                                                        # layout={'width':'60%', 'height':'90%'},
                                                        value=200, description_tooltip='how many slices will be loaded into memory for reconstruction each time')
        self.hs["L[0][0][1][0][0][2][1]_recon_chunk_sz_text"] = grid_recon_chunk[1, :10]
        grid_recon_chunk[1, 10:20] = widgets.BoundedIntText(description='Margin Sz', disabled=True, 
                                                        min=0, max=50,
                                                        layout={'width':f'{int(0.20*(self.form_sz[1]-136))}px', 'height':'90%'},
                                                        # layout={'width':'60%', 'height':'90%'},
                                                        value=15, description_tooltip='how many slices will be loaded into memory for reconstruction each time')
        self.hs["L[0][0][1][0][0][2][2]_recon_margin_sz_text"] = grid_recon_chunk[1, 10:20]

        self.hs["L[0][0][1][0][0][2][1]_recon_chunk_sz_text"].observe(self.L0_0_1_0_0_2_1_recon_chunk_sz_text_change, names="value")
        self.hs["L[0][0][1][0][0][2][2]_recon_margin_sz_text"].observe(self.L0_0_1_0_0_2_2_recon_margin_sz_text_change, names="value")
        self.hs["L[0][0][1][0][0][2]_recon_chunk_box"].children = get_handles(self.hs, "L[0][0][1][0][0][2]_recon_chunk_box", -1)
        ## ## ## ## ## ## label recon_box -- end

        ## ## ## ## ## ## label filter_config_box -- start
        layout = {"justify-content":"center", "align-items":"center","align-contents":"center","border":"3px solid #FFCC00", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][1][0][0][0]_filter_config_title_box"] = widgets.HBox()
        self.hs["L[0][0][1][0][0][0]_filter_config_title_box"].layout = layout
        self.hs["L[0][0][1][0][0][0][0]_filter_config_title"] = widgets.HTML("<span style='color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);'>" + "Filter Config" + "</span>")
        layout = {"left":"43%", "background-color":"white", "color":"cyan"}
        self.hs["L[0][0][1][0][0][0][0]_filter_config_title"].layout = layout
        self.hs["L[0][0][1][0][0][0]_filter_config_title_box"].children = get_handles(self.hs, "L[0][0][1][0][0][0]_filter_config_title_box", -1)
        ## ## ## ## ## ## label filter_config_box -- end

        ## ## ## ## ## ## config filters with GridspecLayout-- start
        self.hs["L[0][0][1][0][0][1]_filter_config_box"] = GridspecLayout(2, 200, layout = {"border":"3px solid #FFCC00",
                                                                                 "height":f"{0.57*(self.form_sz[0]-136)}px",
                                                                                 "align_items":"flex-start",
                                                                                 "justify_items":"flex-start"})
        ## ## ## ## ## ## ## config filters: left-hand side box in GridspecLayout -- start
        self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100] = GridspecLayout(10, 20,
                                                                                grid_gap="8px",
                                                                                layout = {"border":"3px solid #FFCC00",
                                                                                          "height":f"{0.50*(self.form_sz[0]-136)}px",
                                                                                          "grid_row_gap":"8px",
                                                                                          "align_items":"flex-start",
                                                                                          "justify_items":"flex-start",
                                                                                          "grid_column_gap":"8px"})
        self.hs["L[0][0][1][0][0][1][0]_filter_config_left_box"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100]
        self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][0, :16] = widgets.Dropdown(value="phase retrieval",
                                                                                          options=FILTERLIST,
                                                                                          description="Filter List",
                                                                                          indent=False,
                                                                                          disabled=True)
        self.hs["L[0][0][1][0][0][1][0][0]_filter_config_left_box_filter_list"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][0, :16]
        self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][0, 16:19] = widgets.Button(description="==>",
                                                                                          disabled=True,
                                                                                          layout = {"width":f"{int(1.5*(self.form_sz[1]-98)/20)}px"})
        self.hs["L[0][0][1][0][0][1][0][1]_filter_config_left_box_==>_button"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][0, 16:19]
        self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][0, 16:19].style.button_color = "#0000FF"
        for ii in range(3):
            for jj in range(2):
                self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][1+ii, jj*8:(jj+1)*8] = widgets.Dropdown(value="",
                                                                                                               options=[""],
                                                                                                               description="p"+str(ii*2+jj).zfill(2),
                                                                                                               disabled=True,
                                                                                                               layout = {"align_items":"flex-start",
                                                                                                                         "width":f"{int(7.5*(self.form_sz[1]-98)/40)}px"})
        for ii in range(3):
            for jj in range(2):
                self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][4+ii, jj*8:(jj+1)*8] = widgets.FloatText(value=0,
                                                                                                                description="p"+str((ii+3)*2+jj).zfill(2),
                                                                                                                disabled=True,
                                                                                                                layout = {"align_items":"flex-start",
                                                                                                                          "width":f"{int(7.5*(self.form_sz[1]-98)/40)}px"})
        self.hs["L[0][0][1][0][0][1][0][2]_filter_config_left_box_p00"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][1, 0:8]
        self.hs["L[0][0][1][0][0][1][0][3]_filter_config_left_box_p01"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][1, 8:16]
        self.hs["L[0][0][1][0][0][1][0][4]_filter_config_left_box_p02"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][2, 0:8]
        self.hs["L[0][0][1][0][0][1][0][5]_filter_config_left_box_p03"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][2, 8:16]
        self.hs["L[0][0][1][0][0][1][0][6]_filter_config_left_box_p04"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][3, 0:8]
        self.hs["L[0][0][1][0][0][1][0][7]_filter_config_left_box_p05"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][3, 8:16]
        self.hs["L[0][0][1][0][0][1][0][8]_filter_config_left_box_p06"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][4, 0:8]
        self.hs["L[0][0][1][0][0][1][0][9]_filter_config_left_box_p07"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][4, 8:16]
        self.hs["L[0][0][1][0][0][1][0][10]_filter_config_left_box_p08"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][5, 0:8]
        self.hs["L[0][0][1][0][0][1][0][11]_filter_config_left_box_p09"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][5, 8:16]
        self.hs["L[0][0][1][0][0][1][0][12]_filter_config_left_box_p10"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][6, 0:8]
        self.hs["L[0][0][1][0][0][1][0][13]_filter_config_left_box_p11"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][6, 8:16]


        self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][7:, :] = widgets.HTML(value= "<style>p{word-wrap: break-word}</style> <p>"+ "Hover mouse over params for the description of the param for each filter." +" </p>")
        # self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, :100][7:, :] = widgets.Text(value= "Hover mouse over params for the description of the param for each filter.", disabled=True)
        self.hs["L[0][0][1][0][0][1][0][0]_filter_config_left_box_filter_list"].observe(self.L0_0_1_0_0_1_0_0_filter_config_box_left_box_filter_dropdown_change, names="value")
        self.hs["L[0][0][1][0][0][1][0][1]_filter_config_left_box_==>_button"].on_click(self.L0_0_1_0_0_1_0_1_filter_config_box_left_box_move_button_change)
        self.hs["L[0][0][1][0][0][1][0][2]_filter_config_left_box_p00"].observe(self.L0_0_1_0_0_1_0_2_filter_config_box_left_box_p00_change, names="value")
        self.hs["L[0][0][1][0][0][1][0][3]_filter_config_left_box_p01"].observe(self.L0_0_1_0_0_1_0_3_filter_config_box_left_box_p01_change, names="value")
        self.hs["L[0][0][1][0][0][1][0][4]_filter_config_left_box_p02"].observe(self.L0_0_1_0_0_1_0_4_filter_config_box_left_box_p02_change, names="value")
        self.hs["L[0][0][1][0][0][1][0][5]_filter_config_left_box_p03"].observe(self.L0_0_1_0_0_1_0_5_filter_config_box_left_box_p03_change, names="value")
        self.hs["L[0][0][1][0][0][1][0][6]_filter_config_left_box_p04"].observe(self.L0_0_1_0_0_1_0_6_filter_config_box_left_box_p04_change, names="value")
        self.hs["L[0][0][1][0][0][1][0][7]_filter_config_left_box_p05"].observe(self.L0_0_1_0_0_1_0_7_filter_config_box_left_box_p05_change, names="value")
        self.hs["L[0][0][1][0][0][1][0][8]_filter_config_left_box_p06"].observe(self.L0_0_1_0_0_1_0_8_filter_config_box_left_box_p06_change, names="value")
        self.hs["L[0][0][1][0][0][1][0][9]_filter_config_left_box_p07"].observe(self.L0_0_1_0_0_1_0_9_filter_config_box_left_box_p07_change, names="value")
        self.hs["L[0][0][1][0][0][1][0][10]_filter_config_left_box_p08"].observe(self.L0_0_1_0_0_1_0_10_filter_config_box_left_box_p08_change, names="value")
        self.hs["L[0][0][1][0][0][1][0][11]_filter_config_left_box_p09"].observe(self.L0_0_1_0_0_1_0_11_filter_config_box_left_box_p09_change, names="value")
        self.hs["L[0][0][1][0][0][1][0][12]_filter_config_left_box_p10"].observe(self.L0_0_1_0_0_1_0_12_filter_config_box_left_box_p10_change, names="value")
        self.hs["L[0][0][1][0][0][1][0][13]_filter_config_left_box_p11"].observe(self.L0_0_1_0_0_1_0_13_filter_config_box_left_box_p11_change, names="value")
        self.hs["L[0][0][1][0][0][1][0]_filter_config_left_box"].children = get_handles(self.hs, "L[0][0][1][0][0][1][0]_filter_config_left_box", -1)
        ## ## ## ## ## ## ## config filters: left-hand side box in GridspecLayout -- end

        ## ## ## ## ## ## ## config filters: right-hand side box in GridspecLayout -- start
        self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, 100:] = GridspecLayout(10, 10,
                                                                                grid_gap="8px",
                                                                                layout = {"border":"3px solid #FFCC00",
                                                                                          "height":f"{0.50*(self.form_sz[0]-136)}px"})
        self.hs["L[0][0][1][0][0][1][1]_filter_config_right_box"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, 100:]
        self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, 100:][:, :9] = widgets.SelectMultiple(value=["None"],
                                                                                               options=["None"],
                                                                                               description="Filter Seq",
                                                                                               disabled=True,
                                                                                               layout={"height":f"{0.48*(self.form_sz[0]-136)}px"})
        self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, 100:][:, :9]
        self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, 100:][1, 9] = widgets.Button(description="Move Up",
                                                                                      disabled=True,
                                                                                      layout={"width":f"{int(2*(self.form_sz[1]-98)/20)}px"})
        self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, 100:][1, 9].style.button_color = "#0000FF"
        self.hs["L[0][0][1][0][0][1][1][1]_filter_config_right_box_mv_up_button"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, 100:][1, 9]
        self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, 100:][2, 9] = widgets.Button(description="Move Dn",
                                                                                      disabled=True,
                                                                                      layout={"width":f"{int(2*(self.form_sz[1]-98)/20)}px"})
        self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, 100:][2, 9].style.button_color = "#0000FF"
        self.hs["L[0][0][1][0][0][1][1][2]_filter_config_right_box_mv_dn_button"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, 100:][2, 9]
        self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, 100:][3, 9] = widgets.Button(description="Remove",
                                                                                      disabled=True,
                                                                                      layout={"width":f"{int(2*(self.form_sz[1]-98)/20)}px"})
        self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, 100:][3, 9].style.button_color = "#0000FF"
        self.hs["L[0][0][1][0][0][1][1][3]_filter_config_right_box_rm_button"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, 100:][3, 9]
        self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, 100:][4, 9] = widgets.Button(description="Finish",
                                                                                      disabled=True,
                                                                                      layout={"width":f"{int(2*(self.form_sz[1]-98)/20)}px"})
        self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, 100:][4, 9].style.button_color = "#0000FF"
        self.hs["L[0][0][1][0][0][1][1][4]_filter_config_right_box_finish_button"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][0, 100:][4, 9]
        self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].observe(self.L0_0_1_0_0_1_1_0_filter_config_box_right_box_selectmultiple_change, names="value")
        self.hs["L[0][0][1][0][0][1][1][1]_filter_config_right_box_mv_up_button"].on_click(self.L0_0_1_0_0_1_1_1_filter_config_box_right_box_move_up_button_change)
        self.hs["L[0][0][1][0][0][1][1][2]_filter_config_right_box_mv_dn_button"].on_click(self.L0_0_1_0_0_1_1_2_filter_config_box_right_box_move_dn_button_change)
        self.hs["L[0][0][1][0][0][1][1][3]_filter_config_right_box_rm_button"].on_click(self.L0_0_1_0_0_1_1_3_filter_config_box_right_box_remove_button_change)
        self.hs["L[0][0][1][0][0][1][1][4]_filter_config_right_box_finish_button"].on_click(self.L0_0_1_0_0_1_1_4_filter_config_box_right_box_finish_button_change)
        self.hs["L[0][0][1][0][0][1][1]_filter_config_right_box"].children = get_handles(self.hs, "L[0][0][1][0][0][1][1]_filter_config_right_box", -1)
        ## ## ## ## ## ## ## config filters: right-hand side box in GridspecLayout -- end

        ## ## ## ## ## ## ## config confirm box in GridspecLayout -- start
        self.hs["L[0][0][1][0][0][1]_filter_config_box"][1, :140] = widgets.Text(value="Confirm to proceed after you finish data and algorithm configuration...",
                                                                              layout={"width":f"{int(0.696*(self.form_sz[1]-136))}px", "height":"90%"},
                                                                              disabled=True)
        self.hs["L[0][0][1][0][0][1][2]_filter_config_box_confirm_text"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][1, :140]
        self.hs["L[0][0][1][0][0][1]_filter_config_box"][1, 141:171] = widgets.Button(description="Confirm",
                                                                                   disabled=True,
                                                                                   layout={"width":f"{int(0.15*(self.form_sz[1]-136))}px", "height":"90%"})
        self.hs["L[0][0][1][0][0][1]_filter_config_box"][1, 141:171].style.button_color = "darkviolet"
        self.hs["L[0][0][1][0][0][1][3]_filter_config_box_confirm_button"] = self.hs["L[0][0][1][0][0][1]_filter_config_box"][1, 141:171]
        self.hs["L[0][0][1][0][0][1][3]_filter_config_box_confirm_button"].on_click(self.L0_0_1_0_0_1_3_filter_config_box_confirm_button)
        ## ## ## ## ## ## ## config confirm box in GridspecLayout -- end

        self.hs["L[0][0][1][0][0][1]_filter_config_box"].children = get_handles(self.hs, "L[0][0][1][0][0][1]_filter_config_box", -1)
        ## ## ## ## ## ## config filters with GridspecLayout-- end

        self.hs["L[0][0][1][0][0]_filter_config_box"].children = get_handles(self.hs, "L[0][0][1][0][0]_filter_config_box", -1)
        ## ## ## ## ## config  filter_config_box -- end



        ## ## ## ## ## config recon_box -- start
        layout = {"border":"3px solid #8855AA", "height":f"{0.08*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][1][0][1]_recon_box"] = widgets.VBox()
        self.hs["L[0][0][1][0][1]_recon_box"].layout = layout

        ## ## ## ## ## ## ## config widgets in recon_box -- start
        layout = {"justify-content":"center", "align-items":"center","align-contents":"center","border":"3px solid #FFCC00", "height":f"{0.07*(self.form_sz[0]-136)}px"}
        self.hs["L[0][0][1][0][1][1]_recon_box"] = widgets.HBox()
        self.hs["L[0][0][1][0][1][1]_recon_box"].layout = layout
        layout = {"width":"70%", "height":"90%"}
        self.hs["L[0][0][1][0][1][1][0]_recon_progress_bar"] = widgets.IntProgress(value=0,
                                                                                 min=0,
                                                                                 max=10,
                                                                                 step=1,
                                                                                 description="Completing:",
                                                                                 bar_style="info", # "success", "info", "warning", "danger" or ""
                                                                                 orientation="horizontal")
        self.hs["L[0][0][1][0][1][1][0]_recon_progress_bar"].layout = layout
        layout = {"width":"15%", "height":"90%"}
        self.hs["L[0][0][1][0][1][1][1]_recon_button"] = widgets.Button(description="Recon",
                                                                            disabled=True)
        self.hs["L[0][0][1][0][1][1][1]_recon_button"].style.button_color = "darkviolet"
        self.hs["L[0][0][1][0][1][1][1]_recon_button"].layout = layout
        self.hs["L[0][0][1][0][1][1]_recon_box"].children = get_handles(self.hs, "L[0][0][1][0][1][1]_recon_box", -1)
        self.hs["L[0][0][1][0][1][1][1]_recon_button"].on_click(self.L0_0_1_0_1_1_1_recon_button_click)
        ## ## ## ## ## ## ## config widgets in recon_box -- end

        self.hs["L[0][0][1][0][1]_recon_box"].children = get_handles(self.hs, "L[0][0][1][0][1]_recon_box", -1)
        ## ## ## ## ## config recon box -- end

        self.hs["L[0][0][1][0]_filter&recon_tab"].children = get_handles(self.hs, "L[0][0][1][0]_filter&recon_tab", -1)
        ## ## ## ## config filter&recon tab -- end

        ## ## ## ## config filter&recon tab -- start
        layout = {"border":"3px solid #8855AA", "width":f"{self.form_sz[1]-136}px", "height":f"{0.84*(self.form_sz[0]-122)}px"}
        self.hs["L[0][0][1][1]_recon_config_summary_tab"] = widgets.VBox()
        self.hs["L[0][0][1][1]_recon_config_summary_tab"].layout = layout
        ## ## ## ## config filter&recon tab -- end

        self.hs["L[0][0][1]_filter&recon_form"].children = get_handles(self.hs, "L[0][0][1]_filter&recon_form", -1)
        self.hs["L[0][0][1]_filter&recon_form"].set_title(0, "Filter Config")
        self.hs["L[0][0][1]_filter&recon_form"].set_title(1, "Recon Config Summary")
        ## ## ## define boxes in filter&recon_form -- end
        self.bundle_param_handles()
        
        self.hs["L[0][0][0][0][1][0]_select_raw_h5_top_dir_button"].initialdir = self.global_h.cwd
        self.hs["L[0][0][0][0][2][0]_select_save_recon_dir_button"].initialdir = self.global_h.cwd
        self.hs["L[0][0][0][0][3][0]_select_save_data_center_dir_button"].initialdir = self.global_h.cwd
        self.hs["L[0][0][0][0][4][0]_select_save_debug_dir_button"].initialdir = self.global_h.cwd
        self.hs["L[0][0][0][1][0][1][1][1]_alt_flat_file_button"].initialdir = self.global_h.cwd
        self.hs["L[0][0][0][1][0][1][1][3]_alt_dark_file_button"].initialdir = self.global_h.cwd

    def bundle_param_handles(self):
        self.flt_phs = [self.hs["L[0][0][1][0][0][1][0][2]_filter_config_left_box_p00"],
                        self.hs["L[0][0][1][0][0][1][0][3]_filter_config_left_box_p01"],
                        self.hs["L[0][0][1][0][0][1][0][4]_filter_config_left_box_p02"],
                        self.hs["L[0][0][1][0][0][1][0][5]_filter_config_left_box_p03"],
                        self.hs["L[0][0][1][0][0][1][0][6]_filter_config_left_box_p04"],
                        self.hs["L[0][0][1][0][0][1][0][7]_filter_config_left_box_p05"],
                        self.hs["L[0][0][1][0][0][1][0][8]_filter_config_left_box_p06"],
                        self.hs["L[0][0][1][0][0][1][0][9]_filter_config_left_box_p07"],
                        self.hs["L[0][0][1][0][0][1][0][10]_filter_config_left_box_p08"],
                        self.hs["L[0][0][1][0][0][1][0][11]_filter_config_left_box_p09"],
                        self.hs["L[0][0][1][0][0][1][0][12]_filter_config_left_box_p10"],
                        self.hs["L[0][0][1][0][0][1][0][13]_filter_config_left_box_p11"]]
        self.alg_phs = [self.hs["L[0][0][0][1][1][0][1][1]_alg_p00_dropdown"],
                        self.hs["L[0][0][0][1][1][0][1][2]_alg_p01_dropdown"],
                        self.hs["L[0][0][0][1][1][0][1][3]_alg_p02_dropdown"],
                        self.hs["L[0][0][0][1][1][0][2][0]_alg_p03_text"],
                        self.hs["L[0][0][0][1][1][0][2][1]_alg_p04_text"],
                        self.hs["L[0][0][0][1][1][0][2][2]_alg_p05_text"],
                        self.hs["L[0][0][0][1][1][0][2][3]_alg_p06_text"]]

    def reset_config(self):
        self.hs["L[0][0][0][1][1][0][1][0]_alg_options_dropdown"].value = "gridrec"
        self.tomo_selected_alg = "gridrec"
        self.set_alg_param_widgets()
        self.hs["L[0][0][1][0][0][1][0][0]_filter_config_left_box_filter_list"].value = "phase retrieval"
        self.tomo_left_box_selected_flt = "phase retrieval"
        self.hs['L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple'].options = ['None']
        self.hs['L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple'].value = ['None']
        self.tomo_right_filter_dict = {0:{}}
        self.set_flt_param_widgets()

    def lock_message_text_boxes(self):
        boxes = ["L[0][0][0][1][2][0][0]_data_info_text"]
        enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        
    def boxes_logic(self):
        if self.tomo_recon_type == "Trial Center":
            self.hs["L[0][0][0][0][4][2]_save_debug_checkbox"].disabled = False
        elif self.tomo_recon_type in ["Vol Recon: Single", "Vol Recon: Multi"]:
            self.tomo_use_debug = False
            self.hs["L[0][0][0][0][4][2]_save_debug_checkbox"].value = False
            self.hs["L[0][0][0][0][4][2]_save_debug_checkbox"].disabled = True

        if not self.tomo_filepath_configured:
            boxes = ["L[0][0][0][1]_data_tab",
                     "L[0][0][1]_filter&recon_form"]
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            self.lock_message_text_boxes()
        elif (self.tomo_filepath_configured & (not self.tomo_data_configured)):
            boxes = ["L[0][0][0][1][0]_data_config_tab",
                     "L[0][0][0][1][1][0][1][0]_alg_options_dropdown",
                     "L[0][0][0][1][2]_data_info_tab",
                     "L[0][0][0][1][3][0][0]_data_preview_box0",
                     "L[0][0][0][1][3][0][3]_data_preview_box3",
                     "L[0][0][1][0][0][2]_recon_chunk_box",
                     "L[0][0][1][0][0][1][0][0]_filter_config_left_box_filter_list",
                     "L[0][0][1][0][0][1][0][1]_filter_config_left_box_==>_button",
                     "L[0][0][1][0][0][1][1]_filter_config_right_box",
                     "L[0][0][1][0][0][1][3]_filter_config_box_confirm_button"]
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            boxes = ["L[0][0][1][0][1]_recon_box",
                     "L[0][0][0][1][3][0][1]_data_preview_box1",
                     "L[0][0][0][1][3][0][2]_data_preview_box2"]
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            self.lock_message_text_boxes()
        elif ((self.tomo_filepath_configured & self.tomo_data_configured) &
              (self.recon_finish == -1)):
            boxes = ["L[0][0][0][1][0]_data_config_tab",
                     "L[0][0][0][1][1][0][1][0]_alg_options_dropdown",
                     "L[0][0][0][1][2]_data_info_tab",
                     "L[0][0][0][1][3][0][0]_data_preview_box0",
                     "L[0][0][0][1][3][0][3]_data_preview_box3",
                     "L[0][0][1][0][0][2]_recon_chunk_box",
                     "L[0][0][1][0][0][1][0][0]_filter_config_left_box_filter_list",
                     "L[0][0][1][0][0][1][0][1]_filter_config_left_box_==>_button",
                     "L[0][0][1][0][0][1][1]_filter_config_right_box",
                     "L[0][0][1][0][1]_recon_box"]
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            boxes = ["L[0][0][0][1][3][0][1]_data_preview_box1",
                     "L[0][0][0][1][3][0][2]_data_preview_box2"]
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            self.lock_message_text_boxes()
        elif ((self.tomo_filepath_configured & self.tomo_data_configured) &
              (self.recon_finish == 0) & (self.tomo_recon_type=="Trial Center")):
            boxes = ["L[0][0][0][1][0]_data_config_tab",
                     "L[0][0][0][1][1][0][1][0]_alg_options_dropdown",
                     "L[0][0][0][1][2]_data_info_tab",
                     "L[0][0][0][1][3][0][0]_data_preview_box0",
                     "L[0][0][0][1][3][0][1]_data_preview_box1",
                     "L[0][0][0][1][3][0][3]_data_preview_box3",
                     "L[0][0][1][0][0][2]_recon_chunk_box",
                     "L[0][0][1][0][0][1][0][0]_filter_config_left_box_filter_list",
                     "L[0][0][1][0][0][1][0][1]_filter_config_left_box_==>_button",
                     "L[0][0][1][0][0][1][1]_filter_config_right_box",
                     "L[0][0][1][0][1]_recon_box"]
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            boxes = ["L[0][0][0][1][3][0][2]_data_preview_box2"]
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            self.lock_message_text_boxes()
        elif ((self.tomo_filepath_configured & self.tomo_data_configured) &
              (self.recon_finish == 0) & (self.tomo_recon_type=="Vol Recon: Single")):
            boxes = ["L[0][0][0][1][0]_data_config_tab",
                     "L[0][0][0][1][1][0][1][0]_alg_options_dropdown",
                     "L[0][0][0][1][2]_data_info_tab",
                     "L[0][0][0][1][3][0][0]_data_preview_box0",
                     "L[0][0][0][1][3][0][2]_data_preview_box2",
                     "L[0][0][0][1][3][0][3]_data_preview_box3",
                     "L[0][0][1][0][0][2]_recon_chunk_box",
                     "L[0][0][1][0][0][1][0][0]_filter_config_left_box_filter_list",
                     "L[0][0][1][0][0][1][0][1]_filter_config_left_box_==>_button",
                     "L[0][0][1][0][0][1][1]_filter_config_right_box",
                     "L[0][0][1][0][1]_recon_box"]
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            boxes = ["L[0][0][0][1][3][0][1]_data_preview_box1"]
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            self.lock_message_text_boxes()
            

    def tomo_compound_logic(self):
        if self.tomo_recon_type == "Trial Center":
            if self.tomo_raw_data_top_dir_set & self.tomo_data_center_path_set:
                self.hs["L[0][0][0][1][0][1][0][0]_scan_id_dropdown"].disabled = False
                self.hs["L[0][0][0][1][0][1][0][1]_rot_cen_text"].disabled = True
                self.hs["L[0][0][0][1][0][1][0][2]_cen_win_left_text"].disabled = False
                self.hs["L[0][0][0][1][0][1][0][3]_cen_win_wz_text"].disabled = False
                self.hs["L[0][0][1][0][0][2][1]_recon_chunk_sz_text"].disabled = True
                self.hs["L[0][0][1][0][0][2][2]_recon_margin_sz_text"].disabled = True
                self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].disabled = True
        elif self.tomo_recon_type == "Vol Recon: Single":
            if self.tomo_raw_data_top_dir_set & self.tomo_recon_path_set:
                self.hs["L[0][0][0][1][0][1][0][0]_scan_id_dropdown"].disabled = False
                self.hs["L[0][0][0][1][0][1][0][1]_rot_cen_text"].disabled = False
                self.hs["L[0][0][0][1][0][1][0][2]_cen_win_left_text"].disabled = True
                self.hs["L[0][0][0][1][0][1][0][3]_cen_win_wz_text"].disabled = True
                self.hs["L[0][0][1][0][0][2][1]_recon_chunk_sz_text"].disabled = False
                self.hs["L[0][0][1][0][0][2][2]_recon_margin_sz_text"].disabled = False
                self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].disabled = False
        elif self.tomo_recon_type == "Vol Recon: Multi":
            if self.tomo_raw_data_top_dir_set & self.tomo_recon_path_set:
                self.hs["L[0][0][0][1][0][1][0][0]_scan_id_dropdown"].disabled = True
                self.hs["L[0][0][0][1][0][1][0][1]_rot_cen_text"].disabled = True
                self.hs["L[0][0][0][1][0][1][0][2]_cen_win_left_text"].disabled = True
                self.hs["L[0][0][0][1][0][1][0][3]_cen_win_wz_text"].disabled = True
                self.hs["L[0][0][1][0][0][2][1]_recon_chunk_sz_text"].disabled = False
                self.hs["L[0][0][1][0][0][2][2]_recon_margin_sz_text"].disabled = False
                self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].disabled = False

        if self.tomo_filepath_configured: 
            if self.tomo_use_alt_flat:
                self.hs["L[0][0][0][1][0][1][1][1]_alt_flat_file_button"].disabled = False
            else:
                self.hs["L[0][0][0][1][0][1][1][1]_alt_flat_file_button"].disabled = True
    
            if self.tomo_use_fake_flat:
                self.hs["L[0][0][0][1][0][1][2][1]_fake_flat_val_text"].disabled = False
            else:
                self.hs["L[0][0][0][1][0][1][2][1]_fake_flat_val_text"].disabled = True
    
            if self.tomo_use_alt_dark:
                self.hs["L[0][0][0][1][0][1][1][3]_alt_dark_file_button"].disabled = False
            else:
                self.hs["L[0][0][0][1][0][1][1][3]_alt_dark_file_button"].disabled = True
    
            if self.tomo_use_fake_dark:
                self.hs["L[0][0][0][1][0][1][2][3]_fake_dark_val_text"].disabled = False
            else:
                self.hs["L[0][0][0][1][0][1][2][3]_fake_dark_val_text"].disabled = True
    
            if self.tomo_use_rm_zinger:
                self.hs["L[0][0][0][1][0][1][4][1]_zinger_level_text"].disabled = False
            else:
                self.hs["L[0][0][0][1][0][1][4][1]_zinger_level_text"].disabled = True
    
            if self.tomo_use_mask:
                self.hs["L[0][0][0][1][0][1][4][3]_mask_ratio_text"].disabled = False
            else:
                self.hs["L[0][0][0][1][0][1][4][3]_mask_ratio_text"].disabled = True
    
            if self.tomo_is_wedge:
                self.hs["L[0][0][0][1][0][1][5][4]_auto_detect_checkbox"].disabled = False
                if self.tomo_use_wedge_ang_auto_det:
                    # self.hs["L[0][0][0][1][0][1][5][1]_blankat_dropdown"].disabled = True
                    self.hs["L[0][0][0][1][0][1][5][2]_missing_start_text"].disabled = True
                    self.hs["L[0][0][0][1][0][1][5][3]_missing_end_text"].disabled = True
                    self.hs["L[0][0][0][1][0][1][5][5]_auto_thres_text"].disabled = False
                    self.hs["L[0][0][0][1][0][1][6][2]_auto_ref_fn_button"].disabled = False
                    if self.tomo_wedge_ang_auto_det_ref_fn is not None:
                        self.hs["L[0][0][0][1][0][1][6][3]_auto_ref_sli_slider"].disabled = False
                    else:
                        self.hs["L[0][0][0][1][0][1][6][3]_auto_ref_sli_slider"].disabled = True
                else:
                    # self.hs["L[0][0][0][1][0][1][5][1]_blankat_dropdown"].disabled = False
                    self.hs["L[0][0][0][1][0][1][5][2]_missing_start_text"].disabled = False
                    self.hs["L[0][0][0][1][0][1][5][3]_missing_end_text"].disabled = False
                    self.hs["L[0][0][0][1][0][1][5][5]_auto_thres_text"].disabled = True
                    self.hs["L[0][0][0][1][0][1][6][2]_auto_ref_fn_button"].disabled = True
                    self.hs["L[0][0][0][1][0][1][6][3]_auto_ref_sli_slider"].disabled = True
            else:
                self.hs["L[0][0][0][1][0][1][5][4]_auto_detect_checkbox"].value = False
                self.hs["L[0][0][0][1][0][1][5][4]_auto_detect_checkbox"].disabled = True
                # self.hs["L[0][0][0][1][0][1][5][1]_blankat_dropdown"].disabled = True
                self.hs["L[0][0][0][1][0][1][5][2]_missing_start_text"].disabled = True
                self.hs["L[0][0][0][1][0][1][5][3]_missing_end_text"].disabled = True
                self.hs["L[0][0][0][1][0][1][5][5]_auto_thres_text"].disabled = True
                self.hs["L[0][0][0][1][0][1][6][2]_auto_ref_fn_button"].disabled = True

    def set_rec_param_from_config(self, recon_param_dict):        
        self.tomo_raw_data_top_dir = recon_param_dict["file_params"]["raw_data_top_dir"]
        self.tomo_data_center_path = recon_param_dict["file_params"]["data_center_dir"]
        self.tomo_recon_top_dir = recon_param_dict["file_params"]["recon_top_dir"]
        self.tomo_debug_top_dir = recon_param_dict["file_params"]["debug_top_dir"]
        self.tomo_cen_list_file = recon_param_dict["file_params"]["cen_list_file"]
        self.tomo_alt_flat_file = recon_param_dict["file_params"]["alt_flat_file"]
        self.tomo_alt_dark_file = recon_param_dict["file_params"]["alt_dark_file"]
        self.tomo_wedge_ang_auto_det_ref_fn = recon_param_dict["file_params"]["wedge_ang_auto_det_ref_fn"]
        self.global_h.io_tomo_cfg = recon_param_dict['file_params']['io_confg']
        self.tomo_recon_type = recon_param_dict["recon_config"]["recon_type"]
        self.tomo_use_debug = recon_param_dict["recon_config"]["use_debug"]
        
        self.tomo_use_alt_flat = recon_param_dict["recon_config"]["use_alt_flat"]
        self.tomo_use_alt_dark = recon_param_dict["recon_config"]["use_alt_dark"]
        self.tomo_use_fake_flat = recon_param_dict["recon_config"]["use_fake_flat"]
        self.tomo_use_fake_dark = recon_param_dict["recon_config"]["use_fake_dark"]
        self.tomo_use_rm_zinger = recon_param_dict["recon_config"]["use_rm_zinger"]
        self.tomo_use_mask = recon_param_dict["recon_config"]["use_mask"]
        self.tomo_use_wedge_ang_auto_det = recon_param_dict["recon_config"]["use_wedge_ang_auto_det"]
        self.tomo_is_wedge = recon_param_dict["recon_config"]["is_wedge"]
        self.tomo_use_read_config = recon_param_dict["recon_config"]["use_config_file"]
        
        self.tomo_right_filter_dict = recon_param_dict["flt_params"]
        self.tomo_scan_id = recon_param_dict["data_params"]["scan_id"]
        self.tomo_ds_ratio = recon_param_dict["data_params"]["downsample"]
        self.tomo_rot_cen = recon_param_dict["data_params"]["rot_cen"]
        self.tomo_cen_win_s = recon_param_dict["data_params"]["cen_win_s"]
        self.tomo_cen_win_w = recon_param_dict["data_params"]["cen_win_w"]
        self.tomo_fake_flat_val = recon_param_dict["data_params"]["fake_flat_val"]
        self.tomo_fake_dark_val = recon_param_dict["data_params"]["fake_dark_val"]
        self.tomo_sli_s = recon_param_dict["data_params"]["sli_s"]
        self.tomo_sli_e = recon_param_dict["data_params"]["sli_e"]
        self.tomo_col_s = recon_param_dict["data_params"]["col_s"]
        self.tomo_col_e = recon_param_dict["data_params"]["col_e"]
        self.tomo_chunk_sz = recon_param_dict["data_params"]["chunk_sz"]
        self.tomo_margin = recon_param_dict["data_params"]["margin"]
        self.tomo_zinger_val = recon_param_dict["data_params"]["zinger_val"]
        self.tomo_mask_ratio = recon_param_dict["data_params"]["mask_ratio"]
        self.tomo_wedge_missing_s = recon_param_dict["data_params"]["wedge_missing_s"]
        self.tomo_wedge_missing_e = recon_param_dict["data_params"]["wedge_missing_e"]
        self.tomo_wedge_auto_ref_col_s = recon_param_dict["data_params"]["wedge_col_s"]
        self.tomo_wedge_auto_ref_col_e = recon_param_dict["data_params"]["wedge_col_e"]
        self.tomo_wedge_ang_auto_det_thres = recon_param_dict["data_params"]["wedge_ang_auto_det_thres"]
        self.tomo_selected_alg = recon_param_dict["alg_params"]['algorithm']
        self.alg_param_dict = recon_param_dict["alg_params"]['params']
        
    def set_config_from_rec_param(self):
        self.tomo_recon_param_dict["file_params"]["raw_data_top_dir"] = self.tomo_raw_data_top_dir
        self.tomo_recon_param_dict["file_params"]["data_center_dir"]=self.tomo_data_center_path
        self.tomo_recon_param_dict["file_params"]["recon_top_dir"]=self.tomo_recon_top_dir
        self.tomo_recon_param_dict["file_params"]["debug_top_dir"]=self.tomo_debug_top_dir
        self.tomo_recon_param_dict["file_params"]["cen_list_file"]=self.tomo_cen_list_file
        self.tomo_recon_param_dict["file_params"]["alt_flat_file"]=self.tomo_alt_flat_file
        self.tomo_recon_param_dict["file_params"]["alt_dark_file"]=self.tomo_alt_dark_file
        self.tomo_recon_param_dict["file_params"]["wedge_ang_auto_det_ref_fn"]=self.tomo_wedge_ang_auto_det_ref_fn
        self.tomo_recon_param_dict['file_params']['io_confg'] = self.global_h.io_tomo_cfg
        self.tomo_recon_param_dict['file_params']['reader'] = self.reader
        self.tomo_recon_param_dict['file_params']['info_reader'] = self.info_reader
        self.tomo_recon_param_dict['file_params']['use_struc_h5_reader'] = self.global_h.use_struc_h5_reader
        self.tomo_recon_param_dict["recon_config"]["recon_type"]=self.tomo_recon_type
        self.tomo_recon_param_dict["recon_config"]["use_debug"]=self.tomo_use_debug
        self.tomo_recon_param_dict["recon_config"]["use_alt_flat"]=self.tomo_use_alt_flat
        self.tomo_recon_param_dict["recon_config"]["use_alt_dark"]=self.tomo_use_alt_dark
        self.tomo_recon_param_dict["recon_config"]["use_fake_flat"]=self.tomo_use_fake_flat
        self.tomo_recon_param_dict["recon_config"]["use_fake_dark"]=self.tomo_use_fake_dark
        self.tomo_recon_param_dict["recon_config"]["use_rm_zinger"]=self.tomo_use_rm_zinger
        self.tomo_recon_param_dict["recon_config"]["use_mask"]=self.tomo_use_mask
        self.tomo_recon_param_dict["recon_config"]["use_wedge_ang_auto_det"]=self.tomo_use_wedge_ang_auto_det
        self.tomo_recon_param_dict["recon_config"]["is_wedge"]=self.tomo_is_wedge
        self.tomo_recon_param_dict["recon_config"]["use_config_file"] = self.tomo_use_read_config
        self.tomo_recon_param_dict["flt_params"]=self.tomo_right_filter_dict
        self.tomo_recon_param_dict["data_params"]["scan_id"]=self.tomo_scan_id
        self.tomo_recon_param_dict["data_params"]["downsample"]=self.tomo_ds_ratio
        self.tomo_recon_param_dict["data_params"]["rot_cen"]=self.tomo_rot_cen
        self.tomo_recon_param_dict["data_params"]["cen_win_s"]=self.tomo_cen_win_s
        self.tomo_recon_param_dict["data_params"]["cen_win_w"]=self.tomo_cen_win_w
        self.tomo_recon_param_dict["data_params"]["fake_flat_val"]=self.tomo_fake_flat_val
        self.tomo_recon_param_dict["data_params"]["fake_dark_val"]=self.tomo_fake_dark_val
        self.tomo_recon_param_dict["data_params"]["sli_s"]=self.tomo_sli_s
        self.tomo_recon_param_dict["data_params"]["sli_e"]=self.tomo_sli_e
        self.tomo_recon_param_dict["data_params"]["col_s"]=self.tomo_col_s
        self.tomo_recon_param_dict["data_params"]["col_e"]=self.tomo_col_e
        self.tomo_recon_param_dict["data_params"]["chunk_sz"]=self.tomo_chunk_sz
        self.tomo_recon_param_dict["data_params"]["margin"]=self.tomo_margin
        self.tomo_recon_param_dict["data_params"]["zinger_val"]=self.tomo_zinger_val
        self.tomo_recon_param_dict["data_params"]["mask_ratio"]=self.tomo_mask_ratio
        self.tomo_recon_param_dict["data_params"]["wedge_missing_s"]=self.tomo_wedge_missing_s
        self.tomo_recon_param_dict["data_params"]["wedge_missing_e"]=self.tomo_wedge_missing_e
        self.tomo_recon_param_dict["data_params"]["wedge_col_s"]=self.tomo_wedge_auto_ref_col_s
        self.tomo_recon_param_dict["data_params"]["wedge_col_e"]=self.tomo_wedge_auto_ref_col_e
        self.tomo_recon_param_dict["data_params"]["wedge_ang_auto_det_thres"]=self.tomo_wedge_ang_auto_det_thres
        self.tomo_recon_param_dict["alg_params"] = {'algorithm':self.tomo_selected_alg,
                                                    'params':self.alg_param_dict}
        
    def set_widgets_from_rec_param(self, recon_param_dict):        
        self.hs["L[0][0][0][1][0][1][2][0]_use_fake_flat_checkbox"].value = self.tomo_use_alt_flat
        self.hs["L[0][0][0][1][0][1][1][2]_use_alt_dark_checkbox"].value = self.tomo_use_alt_dark
        self.hs["L[0][0][0][1][0][1][2][0]_use_fake_flat_checkbox"].value = self.tomo_use_fake_flat 
        self.hs["L[0][0][0][1][0][1][2][2]_use_fake_dark_checkbox"].value = self.tomo_use_fake_dark 
        self.hs["L[0][0][0][1][0][1][4][0]_rm_zinger_checkbox"].value = self.tomo_use_rm_zinger 
        self.hs["L[0][0][0][1][0][1][4][2]_use_mask_checkbox"].value = self.tomo_use_mask 
        self.hs["L[0][0][0][1][0][1][5][4]_auto_detect_checkbox"].value = self.tomo_use_wedge_ang_auto_det 
        self.hs["L[0][0][0][1][0][1][5][0]_is_wedge_checkbox"].value = self.tomo_is_wedge 
        self.hs["L[0][0][0][1][0][1][0][5]_use_config_checkbox"].value = self.tomo_use_read_config

        a = []
        for ii in sorted(self.tomo_right_filter_dict.keys()):
            a.append(self.tomo_right_filter_dict[ii]['filter_name'])
        self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].options = a
        self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].value = a[-1]
        self.hs["L[0][0][1][0][0][1][0][0]_filter_config_left_box_filter_list"].value = a[-1]
        self.tomo_left_box_selected_flt = a[-1]
        self.set_flt_param_widgets(par_dict=self.tomo_right_filter_dict[ii]['params'])
        
        self.hs["L[0][0][0][1][1][0][1][0]_alg_options_dropdown"].value = self.tomo_selected_alg
        self.set_alg_param_widgets(par_dict=self.alg_param_dict)
        
        self.hs["L[0][0][0][1][0][1][0][0]_scan_id_dropdown"].value = self.tomo_scan_id
        self.hs["L[0][0][0][1][0][1][3][4]_downsample_ratio_text"].value = self.tomo_ds_ratio 
        self.hs["L[0][0][0][1][0][1][0][1]_rot_cen_text"].value = self.tomo_rot_cen 
        self.hs["L[0][0][0][1][0][1][0][2]_cen_win_left_text"].value = self.tomo_cen_win_s 
        self.hs["L[0][0][0][1][0][1][0][3]_cen_win_wz_text"].value = self.tomo_cen_win_w 
        self.hs["L[0][0][0][1][0][1][2][1]_fake_flat_val_text"].value = self.tomo_fake_flat_val 
        self.hs["L[0][0][0][1][0][1][2][3]_fake_dark_val_text"].value = self.tomo_fake_dark_val 
        self.hs["L[0][0][0][1][0][1][3][0]_sli_start_text"].value = self.tomo_sli_s 
        self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].value = self.tomo_sli_e 
        self.hs["L[0][0][0][1][0][1][3][2]_col_start_text"].value = self.tomo_col_s 
        self.hs["L[0][0][0][1][0][1][3][3]_col_end_text"].value = self.tomo_col_e 
        self.hs["L[0][0][1][0][0][2][1]_recon_chunk_sz_text"].value = self.tomo_chunk_sz 
        self.hs["L[0][0][1][0][0][2][2]_recon_margin_sz_text"].value = self.tomo_margin 
        self.hs["L[0][0][0][1][0][1][4][1]_zinger_level_text"].value = self.tomo_zinger_val 
        self.hs["L[0][0][0][1][0][1][4][3]_mask_ratio_text"].value = self.tomo_mask_ratio 
        self.hs["L[0][0][0][1][0][1][5][2]_missing_start_text"].value = self.tomo_wedge_missing_s 
        self.hs["L[0][0][0][1][0][1][5][3]_missing_end_text"].value = self.tomo_wedge_missing_e 
        self.hs["L[0][0][0][1][0][1][6][4]_auto_ref_col_start_text"].value = self.tomo_wedge_auto_ref_col_s
        self.hs["L[0][0][0][1][0][1][6][5]_auto_ref_col_end_text"].value = self.tomo_wedge_auto_ref_col_e
        self.hs["L[0][0][0][1][0][1][5][5]_auto_thres_text"].value = self.tomo_wedge_ang_auto_det_thres 
        
    def set_rec_param_from_widgets(self):
        self.tomo_use_alt_flat = self.hs["L[0][0][0][1][0][1][2][0]_use_fake_flat_checkbox"].value
        self.tomo_use_alt_dark = self.hs["L[0][0][0][1][0][1][1][2]_use_alt_dark_checkbox"].value
        self.tomo_use_fake_flat = self.hs["L[0][0][0][1][0][1][2][0]_use_fake_flat_checkbox"].value
        self.tomo_use_fake_dark = self.hs["L[0][0][0][1][0][1][2][2]_use_fake_dark_checkbox"].value
        self.tomo_use_rm_zinger = self.hs["L[0][0][0][1][0][1][4][0]_rm_zinger_checkbox"].value
        self.tomo_use_mask = self.hs["L[0][0][0][1][0][1][4][2]_use_mask_checkbox"].value
        self.tomo_use_wedge_ang_auto_det = self.hs["L[0][0][0][1][0][1][5][4]_auto_detect_checkbox"].value
        self.tomo_is_wedge = self.hs["L[0][0][0][1][0][1][5][0]_is_wedge_checkbox"].value
        self.tomo_use_read_config = self.hs["L[0][0][0][1][0][1][0][5]_use_config_checkbox"].value

        a = list(self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].options)
        d = {}
        if len(a)>0:
            cnt = 0
            for ii in sorted(self.tomo_right_filter_dict.keys()):
                d[cnt] = self.tomo_right_filter_dict[ii]
                cnt += 1
            self.tomo_right_filter_dict = d
        else:
            self.tomo_right_filter_dict = {0:{}}
        
        self.tomo_scan_id = self.hs["L[0][0][0][1][0][1][0][0]_scan_id_dropdown"].value
        self.tomo_ds_ratio = self.hs["L[0][0][0][1][0][1][3][4]_downsample_ratio_text"].value
        self.tomo_rot_cen = self.hs["L[0][0][0][1][0][1][0][1]_rot_cen_text"].value
        self.tomo_cen_win_s = self.hs["L[0][0][0][1][0][1][0][2]_cen_win_left_text"].value
        self.tomo_cen_win_w = self.hs["L[0][0][0][1][0][1][0][3]_cen_win_wz_text"].value
        self.tomo_fake_flat_val = self.hs["L[0][0][0][1][0][1][2][1]_fake_flat_val_text"].value
        self.tomo_fake_dark_val = self.hs["L[0][0][0][1][0][1][2][3]_fake_dark_val_text"].value
        self.tomo_sli_s = self.hs["L[0][0][0][1][0][1][3][0]_sli_start_text"].value
        self.tomo_sli_e = self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].value
        self.tomo_col_s = self.hs["L[0][0][0][1][0][1][3][2]_col_start_text"].value
        self.tomo_col_e = self.hs["L[0][0][0][1][0][1][3][3]_col_end_text"].value
        self.tomo_chunk_sz = self.hs["L[0][0][1][0][0][2][1]_recon_chunk_sz_text"].value
        self.tomo_margin = self.hs["L[0][0][1][0][0][2][2]_recon_margin_sz_text"].value
        self.tomo_zinger_val = self.hs["L[0][0][0][1][0][1][4][1]_zinger_level_text"].value
        self.tomo_mask_ratio = self.hs["L[0][0][0][1][0][1][4][3]_mask_ratio_text"].value
        self.tomo_wedge_missing_s = self.hs["L[0][0][0][1][0][1][5][2]_missing_start_text"].value
        self.tomo_wedge_missing_e = self.hs["L[0][0][0][1][0][1][5][3]_missing_end_text"].value
        self.tomo_wedge_auto_ref_col_s = self.hs["L[0][0][0][1][0][1][6][4]_auto_ref_col_start_text"].value
        self.tomo_wedge_auto_ref_col_e = self.hs["L[0][0][0][1][0][1][6][5]_auto_ref_col_end_text"].value
        self.tomo_wedge_ang_auto_det_thres = self.hs["L[0][0][0][1][0][1][5][5]_auto_thres_text"].value
    
    def reset_alg_param_widgets(self):
        for ii in range(3):
            self.alg_phs[ii].options = ""
            self.alg_phs[ii].description_tooltip = "p"+str(ii).zfill(2)
        for ii in range(3, 7):
            self.alg_phs[ii].value = 0
            self.alg_phs[ii].description_tooltip = "p"+str(ii).zfill(2)

    def set_alg_param_widgets(self, par_dict=None):
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
            layout = {"width":"23.5%", "visibility":"hidden"}
            h.layout = layout
        alg = ALG_PARAM_DICT[self.tomo_selected_alg]
        for idx in alg.keys():
            self.alg_phs[idx].disabled = False
            layout = {"width":"23.5%", "visibility":"visible"}
            self.alg_phs[idx].layout = layout
            if idx < 3:
                self.alg_phs[idx].options = alg[idx][1]
                if par_dict is None:
                    self.alg_phs[idx].value = alg[idx][1][0]
                else:
                    self.alg_phs[idx].value = par_dict[alg[idx][0]]
            else:
                if par_dict is None:
                    self.alg_phs[idx].value = alg[idx][1]
                else:
                    self.alg_phs[idx].value = par_dict[alg[idx][0]]
            self.alg_phs[idx].description_tooltip = alg[idx][2]

    def read_alg_param_widgets(self):
        self.alg_param_dict = OrderedDict({})
        alg = ALG_PARAM_DICT[self.tomo_selected_alg]
        for idx in alg.keys():
            self.alg_param_dict[alg[idx][0]] = self.alg_phs[idx].value
            
    def reset_flt_param_widgets(self):
        for ii in range(6):
            self.flt_phs[ii].options = ""
            self.flt_phs[ii].description_tooltip = "p"+str(ii).zfill(2)
        for ii in range(6, 12):
            self.flt_phs[ii].value = 0
            self.flt_phs[ii].description_tooltip = "p"+str(ii).zfill(2)

    def set_flt_param_widgets(self, par_dict=None):
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
        flt = FILTER_PARAM_DICT[self.tomo_left_box_selected_flt]
        for idx in flt.keys():
            self.flt_phs[idx].disabled = False
            if idx < 6:
                self.flt_phs[idx].options = flt[idx][1]
                if par_dict is None:
                    self.flt_phs[idx].value = flt[idx][1][0]
                else:
                    self.flt_phs[idx].value = par_dict[flt[idx][0]]
            else:
                if par_dict is None:
                    self.flt_phs[idx].value = flt[idx][1]
                else:
                    self.flt_phs[idx].value = par_dict[flt[idx][0]]
            self.flt_phs[idx].description_tooltip = flt[idx][2]

    def read_flt_param_widgets(self):
        self.flt_param_dict = OrderedDict({})
        flt = FILTER_PARAM_DICT[self.tomo_left_box_selected_flt]
        for idx in flt.keys():
            self.flt_param_dict[flt[idx][0]] = self.flt_phs[idx].value

    def L0_0_0_0_1_0_select_raw_h5_top_dir_button_click(self, a):
        self.reset_config()
        if len(a.files[0]) != 0:
            self.tomo_raw_data_top_dir = a.files[0]
            self.tomo_raw_data_file_template = os.path.join(self.tomo_raw_data_top_dir, self.tomo_raw_fn_temp)
            b = ''
            t = (time.strptime(time.asctime()))
            for ii in range(6):
                b +=str(t[ii]).zfill(2)+'-'
            self.tomo_trial_cen_dict_fn = os.path.join(self.tomo_raw_data_top_dir, 'trial_cen_dict_{}.json'.format(b))
            self.tomo_recon_dict_fn = os.path.join(self.tomo_raw_data_top_dir, 'recon_dict_{}.json'.format(b))
            self.tomo_raw_data_top_dir_set = True
            self.hs["L[0][0][0][0][1][0]_select_raw_h5_top_dir_button"].initialdir = os.path.abspath(a.files[0])
            self.hs["L[0][0][0][0][2][0]_select_save_recon_dir_button"].initialdir = os.path.abspath(a.files[0])
            self.hs["L[0][0][0][0][3][0]_select_save_data_center_dir_button"].initialdir = os.path.abspath(a.files[0])
            self.hs["L[0][0][0][0][4][0]_select_save_debug_dir_button"].initialdir = os.path.abspath(a.files[0])
            self.hs['L[0][0][0][1][0][1][0][4]_cen_list_button'].initialdir = os.path.abspath(a.files[0])
            self.hs['L[0][0][0][1][0][1][1][1]_alt_flat_file_button'].initialdir = os.path.abspath(a.files[0])
            self.hs['L[0][0][0][1][0][1][1][3]_alt_dark_file_button'].initialdir = os.path.abspath(a.files[0])
            self.hs['L[0][0][0][1][0][1][6][2]_auto_ref_fn_button'].initialdir = os.path.abspath(a.files[0])
            with open(self.global_h.GUI_cfg_file, 'w') as f:
                json.dump({'cwd':os.path.abspath(a.files[0])},f)
        else:
            self.tomo_raw_data_top_dir = None
            self.tomo_raw_data_top_dir_set = False
            self.hs["L[0][0][0][0][1][1]_select_raw_h5_top_dir_text"].value = "Choose raw h5 top dir ..."
        self.tomo_filepath_configured = False
        self.recon_finish = -1
        self.hs["L[0][0][0][0][5][1]_confirm_file&path_text"].value="After setting directories, confirm to proceed ..."
        self.boxes_logic()

    def L0_0_0_0_2_0_select_save_recon_dir_button_click(self, a):
        self.reset_config()
        if not self.tomo_raw_data_top_dir_set:
            self.hs["L[0][0][0][0][5][1]_confirm_file&path_text"].value = "Please specify raw h5 top directory first ..."
            self.hs["L[0][0][0][0][2][1]_select_save_recon_dir_text"].value = "Choose top directory where recon subdirectories are saved..."
            self.tomo_recon_path_set = False
        else:
            if len(a.files[0]) != 0:
                self.tomo_recon_top_dir = a.files[0]
                self.tomo_recon_path_set = True
                self.hs["L[0][0][0][0][2][0]_select_save_recon_dir_button"].initialdir = os.path.abspath(a.files[0])
                with open(self.global_h.GUI_cfg_file, 'w') as f:
                    json.dump({'cwd':os.path.abspath(a.files[0])},f)
            else:
                self.tomo_recon_top_dir = None
                self.tomo_recon_path_set = False
                self.hs["L[0][0][0][0][2][1]_select_save_recon_dir_text"].value = "Select top directory where recon subdirectories are saved..."
            self.hs["L[0][0][0][0][5][1]_confirm_file&path_text"].value="After setting directories, confirm to proceed ..."
        self.tomo_filepath_configured = False
        self.recon_finish = -1
        self.boxes_logic()

    def L0_0_0_0_3_0_select_save_data_center_dir_button_click(self, a):
        self.reset_config()
        if not self.tomo_raw_data_top_dir_set:
            self.hs["L[0][0][0][0][5][1]_confirm_file&path_text"].value = "Please specify raw h5 top directory first ..."
            self.hs["L[0][0][0][0][3][1]_select_save_data_center_dir_text"].value="Select top directory where data_center will be created..."
            self.tomo_data_center_path_set = False
        else:
            if len(a.files[0]) != 0:
                self.tomo_data_center_top_dir = a.files[0]
                self.tomo_data_center_path_set = True
                self.hs["L[0][0][0][0][3][0]_select_save_data_center_dir_button"].initialdir = os.path.abspath(a.files[0])
                with open(self.global_h.GUI_cfg_file, 'w') as f:
                    json.dump({'cwd':os.path.abspath(a.files[0])},f)
            else:
                self.tomo_data_center_top_dir = None
                self.tomo_data_center_path_set = False
                self.hs["L[0][0][0][0][3][1]_select_save_data_center_dir_text"].value="Select top directory where data_center will be created..."
            self.hs["L[0][0][0][0][5][1]_confirm_file&path_text"].value="After setting directories, confirm to proceed ..."
        self.tomo_filepath_configured = False
        self.recon_finish = -1
        self.boxes_logic()

    def L0_0_0_0_4_0_select_save_debug_dir_button_click(self, a):
        self.reset_config()
        if not self.tomo_raw_data_top_dir_set:
            self.hs["L[0][0][0][0][5][1]_confirm_file&path_text"].value = "Please specify raw h5 top directory first ..."
            self.hs["L[0][0][0][0][4][1]_select_save_debug_dir_text"].value = "Select top directory where debug dir will be created..."
            self.tomo_debug_path_set = False
        else:
            if len(a.files[0]) != 0:
                self.tomo_debug_top_dir = a.files[0]
                self.tomo_debug_path_set = True
                self.hs["L[0][0][0][0][4][0]_select_save_debug_dir_button"].initialdir = os.path.abspath(a.files[0])
                with open(self.global_h.GUI_cfg_file, 'w') as f:
                    json.dump({'cwd':os.path.abspath(a.files[0])},f)
            else:
                self.tomo_debug_top_dir = None
                self.tomo_debug_path_set = False
                self.hs["L[0][0][0][0][4][1]_select_save_debug_dir_text"].value = "Select top directory where debug dir will be created..."
            self.hs["L[0][0][0][0][5][1]_confirm_file&path_text"].value="After setting directories, confirm to proceed ..."
        self.tomo_filepath_configured = False
        self.recon_finish = -1
        self.boxes_logic()

    def L0_0_0_0_4_2_save_debug_checkbox_change(self, a):
        self.reset_config()
        if a["owner"].value:
            self.tomo_use_debug = True
            self.hs["L[0][0][0][0][4][0]_select_save_debug_dir_button"].disabled = False
            self.hs["L[0][0][0][0][4][1]_select_save_debug_dir_text"].value = "Select top directory where debug dir will be created..."
            self.hs["L[0][0][0][0][4][0]_select_save_debug_dir_button"].style.button_color = "orange"
        else:
            self.tomo_use_debug = False
            self.hs["L[0][0][0][0][4][0]_select_save_debug_dir_button"].disabled = True
            self.hs["L[0][0][0][0][4][1]_select_save_debug_dir_text"].value = "Debug is disabled..."
            self.hs["L[0][0][0][0][4][0]_select_save_debug_dir_button"].style.button_color = "orange"
        self.tomo_filepath_configured = False
        self.recon_finish = -1
        self.boxes_logic()

    def L0_0_0_0_5_2_file_path_options_dropdown_change(self, a):
        restart(self, dtype='TOMO')
        self.reset_config()
        self.tomo_recon_type = a["owner"].value
        if self.tomo_recon_type == "Trial Center":
            layout = {"width":"15%", "height":"85%", "visibility":"hidden"}
            self.hs["L[0][0][0][1][0][1][0][4]_cen_list_button"].layout = layout
            layout = {"width":"7%", "visibility":"hidden"}
            self.hs["L[0][0][0][1][0][1][0][5]_use_config_checkbox"].layout = layout
            layout = {"width":"19%", "visibility":"visible"}
            self.hs["L[0][0][0][1][0][1][0][2]_cen_win_left_text"].layout = layout
            self.hs["L[0][0][0][1][0][1][0][3]_cen_win_wz_text"].layout = layout

            self.hs["L[0][0][0][0][3][0]_select_save_data_center_dir_button"].disabled = False
            self.hs["L[0][0][0][0][2][0]_select_save_recon_dir_button"].disabled = True
            self.hs["L[0][0][0][0][3][0]_select_save_data_center_dir_button"].style.button_color = "orange"
            self.hs["L[0][0][0][0][2][0]_select_save_recon_dir_button"].style.button_color = "orange"
            
            self.hs["L[0][0][0][1][0][1][0][5]_use_config_checkbox"].value = False
            self.tomo_use_read_config = False
        elif self.tomo_recon_type in ["Vol Recon: Single", "Vol Recon: Multi"]:
            self.hs["L[0][0][0][0][3][0]_select_save_data_center_dir_button"].disabled = True
            self.hs["L[0][0][0][0][2][0]_select_save_recon_dir_button"].disabled = False
            self.hs["L[0][0][0][0][3][0]_select_save_data_center_dir_button"].style.button_color = "orange"
            self.hs["L[0][0][0][0][2][0]_select_save_recon_dir_button"].style.button_color = "orange"
            # if self.tomo_recon_type == "Vol Recon: Multi":
            layout = {"width":"15%", "height":"85%", "visibility":"visible"}
            self.hs["L[0][0][0][1][0][1][0][4]_cen_list_button"].layout = layout
            layout = {"width":"7%", "visibility":"visible"}
            self.hs["L[0][0][0][1][0][1][0][5]_use_config_checkbox"].layout = layout
            layout = {"width":"19%", "visibility":"hidden"}
            self.hs["L[0][0][0][1][0][1][0][2]_cen_win_left_text"].layout = layout
            self.hs["L[0][0][0][1][0][1][0][3]_cen_win_wz_text"].layout = layout
            
            self.hs["L[0][0][0][1][0][1][0][5]_use_config_checkbox"].value = True
            self.tomo_use_read_config = True
        self.tomo_filepath_configured = False
        self.recon_finish = -1
        self.boxes_logic()

    def L0_0_0_0_5_0_confirm_file_path_button_click(self, a):        
        if self.tomo_recon_type == "Trial Center":
            if (self.tomo_raw_data_top_dir_set & self.tomo_data_center_path_set):
                self.tomo_available_raw_idx = check_file_availability(self.tomo_raw_data_top_dir, scan_id=None, signature=self.global_h.io_tomo_cfg['tomo_raw_fn_template'], return_idx=True)
                if len(self.tomo_available_raw_idx) == 0:
                    self.tomo_filepath_configured = False
                    return
                else:
                    self.hs["L[0][0][0][1][0][1][0][0]_scan_id_dropdown"].options = self.tomo_available_raw_idx
                    self.hs["L[0][0][0][1][0][1][0][0]_scan_id_dropdown"].value = self.tomo_available_raw_idx[0] 
                    if self.tomo_use_debug:
                        if self.tomo_debug_path_set:
                            self.tomo_data_center_path = os.path.join(self.tomo_data_center_top_dir, "data_center")
                            self.tomo_filepath_configured = True
                            self.set_alg_param_widgets()
                            self.set_flt_param_widgets()
                        else:
                            self.hs["L[0][0][0][0][5][1]_confirm_file&path_text"].value = "You need to select the top directory to create debug dir..."
                            self.tomo_filepath_configured = False
                    else:
                        self.tomo_data_center_path = os.path.join(self.tomo_data_center_top_dir, "data_center")
                        self.tomo_filepath_configured = True
                        self.set_alg_param_widgets()
                        self.set_flt_param_widgets()
            else:
                self.hs["L[0][0][0][0][5][1]_confirm_file&path_text"].value = "You need to select the top raw dir and top dir where debug dir can be created..."
                self.tomo_filepath_configured = False
        elif self.tomo_recon_type in ["Vol Recon: Single", "Vol Recon: Multi"]:
            if (self.tomo_raw_data_top_dir_set & self.tomo_recon_path_set):
                self.tomo_available_raw_idx = check_file_availability(self.tomo_raw_data_top_dir, scan_id=None, signature=self.global_h.io_tomo_cfg['tomo_raw_fn_template'], return_idx=True)
                if len(self.tomo_available_raw_idx) == 0:
                    self.tomo_filepath_configured = False
                    return
                else:
                    self.hs["L[0][0][0][1][0][1][0][0]_scan_id_dropdown"].options = self.tomo_available_raw_idx
                    self.hs["L[0][0][0][1][0][1][0][0]_scan_id_dropdown"].value = self.tomo_available_raw_idx[0]                   
                    self.tomo_filepath_configured = True
                    self.set_alg_param_widgets()
                    self.set_flt_param_widgets()
            else:
                self.hs["L[0][0][0][0][5][1]_confirm_file&path_text"].value = "You need to select the top raw dir and top dir where recon dir can be created..."
                self.tomo_filepath_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_0_0_scan_id_dropdown_change(self, a):
        self.tomo_scan_id = a["owner"].value
        self.data_info = get_raw_img_info(self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                                          self.global_h.io_tomo_cfg, scan_type='tomo')
        info = ""
        for key, item in self.data_info.items():
            info = info + str(key) + ':' + str(item) + '\n'
        self.hs["L[0][0][0][1][2][0][0]_data_info_text"].value = info
        if self.data_info is not None:
            self.hs["L[0][0][0][1][0][1][0][2]_cen_win_left_text"].value = 0
            self.hs["L[0][0][0][1][0][1][0][2]_cen_win_left_text"].max = self.data_info["img_dim"][2] - self.hs['L[0][0][0][1][0][1][0][3]_cen_win_wz_text'].value - 1
            self.hs["L[0][0][0][1][0][1][0][2]_cen_win_left_text"].value = int(self.data_info["img_dim"][2]/2) - 40
            if self.tomo_recon_type == 'Trial Center':
                self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].value = 1
                self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].max = self.data_info["img_dim"][1] - 1
                self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].value = int(self.data_info["img_dim"][1]/2) + 10
                self.hs["L[0][0][0][1][0][1][3][0]_sli_start_text"].value = 0
                self.hs["L[0][0][0][1][0][1][3][0]_sli_start_text"].max = self.data_info["img_dim"][1] - 2
                self.hs["L[0][0][0][1][0][1][3][0]_sli_start_text"].value = int(self.data_info["img_dim"][1]/2) - 10
                self.hs["L[0][0][0][1][0][1][3][2]_col_start_text"].value = 0
                self.hs["L[0][0][0][1][0][1][3][2]_col_start_text"].max = self.data_info["img_dim"][2]-1
                self.hs["L[0][0][0][1][0][1][3][3]_col_end_text"].value = 0
                self.hs["L[0][0][0][1][0][1][3][3]_col_end_text"].max = self.data_info["img_dim"][2]-1
                self.hs["L[0][0][0][1][0][1][3][3]_col_end_text"].value = self.data_info["img_dim"][2]-1
                self.hs["L[0][0][0][1][0][1][6][4]_auto_ref_col_start_text"].value = 0
                self.hs["L[0][0][0][1][0][1][6][4]_auto_ref_col_start_text"].max = self.data_info["img_dim"][2]-1
                self.hs["L[0][0][0][1][0][1][6][5]_auto_ref_col_end_text"].value = 0
                self.hs["L[0][0][0][1][0][1][6][5]_auto_ref_col_end_text"].max = self.data_info["img_dim"][2]-1
                self.hs["L[0][0][0][1][0][1][6][5]_auto_ref_col_end_text"].value = self.data_info["img_dim"][2]-1
            else:
                self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].value = 1
                self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].max = self.data_info["img_dim"][1] - 1
                self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].value = self.data_info["img_dim"][1] - 1
                self.hs["L[0][0][0][1][0][1][3][0]_sli_start_text"].value = 0
                self.hs["L[0][0][0][1][0][1][3][0]_sli_start_text"].max = self.data_info["img_dim"][1] - 2
                self.hs["L[0][0][0][1][0][1][3][0]_sli_start_text"].value = 0
                self.hs["L[0][0][0][1][0][1][3][2]_col_start_text"].value = 0
                self.hs["L[0][0][0][1][0][1][3][2]_col_start_text"].max = self.data_info["img_dim"][2]-1
                self.hs["L[0][0][0][1][0][1][3][3]_col_end_text"].value = 0
                self.hs["L[0][0][0][1][0][1][3][3]_col_end_text"].max = self.data_info["img_dim"][2]-1
                self.hs["L[0][0][0][1][0][1][3][3]_col_end_text"].value = self.data_info["img_dim"][2]-1
                self.hs["L[0][0][0][1][0][1][6][4]_auto_ref_col_start_text"].value = 0
                self.hs["L[0][0][0][1][0][1][6][4]_auto_ref_col_start_text"].max = self.data_info["img_dim"][2]-1
                self.hs["L[0][0][0][1][0][1][6][5]_auto_ref_col_end_text"].value = 0
                self.hs["L[0][0][0][1][0][1][6][5]_auto_ref_col_end_text"].max = self.data_info["img_dim"][2]-1
                self.hs["L[0][0][0][1][0][1][6][5]_auto_ref_col_end_text"].value = self.data_info["img_dim"][2]-1
            self.hs["L[0][0][0][1][3][0][3][0]_raw_proj_slider"].value = 0
            self.hs["L[0][0][0][1][3][0][3][0]_raw_proj_slider"].min = 0
            self.hs["L[0][0][0][1][3][0][3][0]_raw_proj_slider"].max = self.data_info["img_dim"][0] - 1
            self.hs["L[0][0][0][1][3][0][3][1]_raw_proj_all_in_mem_checkbox"].value = False
            self.raw_proj = np.ndarray([self.data_info["img_dim"][1], self.data_info["img_dim"][2]], 
                                       dtype=np.float32)
            self.raw_proj_0 = np.ndarray([self.data_info["img_dim"][1], self.data_info["img_dim"][2]], 
                                          dtype=np.float32)
            self.raw_proj_180 = np.ndarray([self.data_info["img_dim"][1], self.data_info["img_dim"][2]], 
                                          dtype=np.float32)
            self.raw_is_in_mem = False
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_0_1_rot_cen_text_change(self, a):
        self.tomo_rot_cen = [a["owner"].value]
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_0_2_cen_win_left_text_change(self, a):
        self.data_info = get_raw_img_info(self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                                          self.global_h.io_tomo_cfg, scan_type='tomo')
        if a["owner"].value > (self.data_info["img_dim"][2] - self.hs['L[0][0][0][1][0][1][0][3]_cen_win_wz_text'].value - 1):
            a["owner"].value = (self.data_info["img_dim"][2] - self.hs['L[0][0][0][1][0][1][0][3]_cen_win_wz_text'].value - 1)
        self.tomo_cen_win_s = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_0_3_cen_win_wz_text_change(self, a):
        self.data_info = get_raw_img_info(self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                                          self.global_h.io_tomo_cfg, scan_type='tomo')
        if a["owner"].value > (self.data_info["img_dim"][2] - self.hs["L[0][0][0][1][0][1][0][2]_cen_win_left_text"].value - 1):
            a["owner"].value = (self.data_info["img_dim"][2] - self.hs["L[0][0][0][1][0][1][0][2]_cen_win_left_text"].value - 1)
        self.tomo_cen_win_w = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_0_4_cen_list_button_click(self, a):
        if len(a.files[0]) != 0:
            self.tomo_cen_list_file = a.files[0]
            if self.tomo_use_read_config & (self.tomo_recon_type=="Vol Recon: Single"):
                with open(self.tomo_cen_list_file, 'r') as f:
                    recon_param_dict = json.load(f)
                    self.set_rec_param_from_config(recon_param_dict)
                    self.set_widgets_from_rec_param(recon_param_dict)
        else:
            self.tomo_cen_list_file = None
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()
        
    def L0_0_0_1_0_1_0_5_use_config_checkbox_change(self, a):
        self.tomo_use_read_config = a['owner'].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()        

    def L0_0_0_1_0_1_1_0_use_alt_flat_checkbox_change(self, a):
        self.tomo_use_alt_flat = a["owner"].value
        if self.tomo_use_alt_flat:
            self.hs["L[0][0][0][1][0][1][2][0]_use_fake_flat_checkbox"].value = False
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_1_1_alt_flat_file_button_change(self, a):
        if len(a.files[0]) != 0:
            self.tomo_alt_flat_file = a.files[0]
        else:
            self.tomo_alt_flat_file = None
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_1_2_use_alt_dark_checkbox_change(self, a):
        self.tomo_use_alt_dark = a["owner"].value
        if self.hs["L[0][0][0][1][0][1][1][2]_use_alt_dark_checkbox"].value:
            self.hs["L[0][0][0][1][0][1][2][2]_use_fake_dark_checkbox"].value = False
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_1_3_alt_dark_file_button_change(self, a):
        if len(a.files[0]) != 0:
            self.tomo_alt_dark_file = a.files[0]
        else:
            self.tomo_alt_dark_file = None
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_2_0_use_fake_flat_checkbox_change(self, a):
        self.tomo_use_fake_flat = a["owner"].value
        if self.hs["L[0][0][0][1][0][1][2][0]_use_fake_flat_checkbox"].value:
            self.hs["L[0][0][0][1][0][1][1][0]_use_alt_flat_checkbox"].value = False
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_2_1_fake_flat_val_text_change(self, a):
        self.tomo_fake_flat_val = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_2_2_use_fake_dark_checkbox_change(self, a):
        self.tomo_use_fake_dark = a["owner"].value
        if self.hs["L[0][0][0][1][0][1][2][2]_use_fake_dark_checkbox"].value:
            self.hs["L[0][0][0][1][0][1][1][2]_use_alt_dark_checkbox"].value = False
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_2_3_fake_dark_val_text_change(self, a):
        self.tomo_fake_dark_val = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_3_0_sli_start_text_change(self, a):
        if self.tomo_recon_type == 'Trial Center':
            if a["owner"].value>=self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].value:
                self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].value = a["owner"].value+20
        else:
            if a["owner"].value>=self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].value:
                a["owner"].value = self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].value - 1
        self.tomo_sli_s = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_3_1_sli_end_text_change(self, a):
        if a["owner"].value<=self.hs["L[0][0][0][1][0][1][3][0]_sli_start_text"].value:
            a["owner"].value = self.hs["L[0][0][0][1][0][1][3][0]_sli_start_text"].value + 1
        self.tomo_sli_e = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()
        
    def L0_0_0_1_0_1_3_2_col_start_text_change(self, a):
        if a["owner"].value>=self.hs["L[0][0][0][1][0][1][3][3]_col_end_text"].value:
            a["owner"].value = self.hs["L[0][0][0][1][0][1][3][3]_col_end_text"].value - 1
        self.tomo_col_s = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()
    
    def L0_0_0_1_0_1_3_3_col_end_change(self, a):
        if a["owner"].value<=self.hs["L[0][0][0][1][0][1][3][2]_col_start_text"].value:
            a["owner"].value = self.hs["L[0][0][0][1][0][1][3][2]_col_start_text"].value + 1
        self.tomo_col_e = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_3_4_downsample_ratio_text_change(self, a):
        self.tomo_ds_ratio = a['owner'].value
        if self.tomo_ds_ratio == 1:
            self.tomo_use_downsample = False
        else:
            self.tomo_use_downsample = True
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_4_0_rm_zinger_checkbox_change(self, a):
        self.tomo_use_rm_zinger = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_4_1_zinger_level_text_change(self, a):
        self.tomo_zinger_val = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_4_2_use_mask_checkbox_change(self, a):
        self.tomo_use_mask = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_4_3_mask_ratio_text_change(self, a):
        self.tomo_mask_ratio = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_1_0_1_0_alg_options_dropdown_change(self, a):
        self.tomo_selected_alg = a["owner"].value
        self.set_alg_param_widgets()
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_1_0_1_1_alg_p00_dropdown_change(self, a):
        self.tomo_alg_p01 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_1_0_1_2_alg_p01_dropdown_change(self, a):
        self.tomo_alg_p02 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_1_0_1_3_alg_p02_dropdown_change(self, a):
        self.tomo_alg_p03 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_1_0_2_0_alg_p03_text_change(self, a):
        self.tomo_alg_p04 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_1_0_2_1_alg_p04_text_change(self, a):
        self.tomo_alg_p05 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_1_0_2_2_alg_p05_text_change(self, a):
        self.tomo_alg_p06 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_1_0_2_3_alg_p06_text_change(self, a):
        self.tomo_alg_p07 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_0_0_is_wedge_checkbox_change(self, a):
        self.tomo_is_wedge = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_0_2_missing_start_text_change(self, a):
        self.tomo_wedge_missing_s = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_0_3_missing_end_text_change(self, a):
        self.tomo_wedge_missing_e = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_5_4_auto_detect_checkbox_change(self, a):
        self.tomo_use_wedge_ang_auto_det = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_0_1_0_1_5_5_auto_thres_text_change(self, a):
        self.tomo_wedge_ang_auto_det_thres = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_1_2_auto_ref_fn_button_change(self, a):
        if len(a.files[0]) != 0:
            self.tomo_wedge_ang_auto_det_ref_fn = a.files[0]
            self.wedge_eva_data, _, _, _ = read_data(self.reader, self.tomo_wedge_ang_auto_det_ref_fn, 
                                                     self.global_h.io_tomo_cfg, 
                                                     sli_start=self.tomo_sli_s, sli_end=self.tomo_sli_e,
                                                     col_start=self.tomo_wedge_auto_ref_col_s, col_end=self.tomo_wedge_auto_ref_col_e,
                                                     ds_use=self.tomo_use_downsample, ds_level=self.tomo_ds_ratio, 
                                                     mean_axis=2)
            self.hs["L[0][0][0][1][0][1][6][3]_auto_ref_sli_slider"].value = 0
            self.hs["L[0][0][0][1][0][1][6][3]_auto_ref_sli_slider"].max = self.wedge_eva_data.shape[1] - 1 
        else:
            self.tomo_wedge_ang_auto_det_ref_fn = None
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()
        
    def L0_0_0_1_0_1_6_3_auto_ref_sli_slider_change(self, a):
        plt.figure(0)
        plt.plot(self.wedge_eva_data[:, a['owner'].value])
        plt.plot(np.arange(self.wedge_eva_data.shape[0]), np.ones(self.wedge_eva_data.shape[0])*self.tomo_wedge_ang_auto_det_thres)
        plt.show()
        self.boxes_logic()
        self.tomo_compound_logic()
        
    def L0_0_0_1_0_1_6_4_auto_ref_col_start_text_change(self, a):
        if a['owner'].value >= self.hs["L[0][0][0][1][0][1][6][5]_auto_ref_col_end_text"].value:
            a['owner'].value = self.hs["L[0][0][0][1][0][1][6][5]_auto_ref_col_end_text"].value - 1
        self.tomo_wedge_auto_ref_col_s = a['owner'].value
    
    def L0_0_0_1_0_1_6_5_auto_ref_col_end_text_change(self, a):
        if a['owner'].value <= self.hs["L[0][0][0][1][0][1][6][4]_auto_ref_col_start_text"].value:
            a['owner'].value = self.hs["L[0][0][0][1][0][1][6][4]_auto_ref_col_start_text"].value + 1
        self.tomo_wedge_auto_ref_col_e = a['owner'].value
        
    def L0_0_0_1_3_0_3_0_raw_proj_slider_change(self, a):
        idx = a['owner'].value
        if self.load_raw_in_mem:
            if not self.raw_is_in_mem:
                self.raw = ((self.reader(self.tomo_raw_data_file_template.format(self.tomo_scan_id), dtype='data', sli=[None, None, None], cfg=self.global_h.io_tomo_cfg) - 
                             self.reader(self.tomo_raw_data_file_template.format(self.tomo_scan_id), dtype='dark', sli=[None, None, None], cfg=self.global_h.io_tomo_cfg).mean(axis=0)) /
                            (self.reader(self.tomo_raw_data_file_template.format(self.tomo_scan_id), dtype='flat', sli=[None, None, None], cfg=self.global_h.io_tomo_cfg).mean(axis=0) - 
                             self.reader(self.tomo_raw_data_file_template.format(self.tomo_scan_id), dtype='dark', sli=[None, None, None], cfg=self.global_h.io_tomo_cfg).mean(axis=0))).astype(np.float32)
                self.raw[:] = np.where(self.raw<0, 1, self.raw)[:]
                self.raw[np.isinf(self.raw)] = 1
                self.raw_is_in_mem = True
                
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='tomo_raw_img_viewer')
            if (not data_state) | (not viewer_state):
                fiji_viewer_on(self.global_h, self, viewer_name='tomo_raw_img_viewer')
                self.global_h.tomo_fiji_windows['tomo_raw_img_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
                    self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(self.raw)), self.global_h.ImagePlusClass))
            self.global_h.tomo_fiji_windows['tomo_raw_img_viewer']['ip'].setSlice(idx)
            self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
            self.global_h.tomo_fiji_windows['tomo_raw_img_viewer']['ip'].setRoi(self.tomo_col_s, self.tomo_sli_s, self.tomo_col_e-self.tomo_col_s, self.tomo_sli_e-self.tomo_sli_s)
        else:
            self.raw_proj[:] = ((self.reader(self.tomo_raw_data_file_template.format(self.tomo_scan_id), dtype='data', sli=[[idx, idx+1], None, None], cfg=self.global_h.io_tomo_cfg) -
                                 self.reader(self.tomo_raw_data_file_template.format(self.tomo_scan_id), dtype='dark', sli=[None, None, None], cfg=self.global_h.io_tomo_cfg).mean(axis=0)) /
                                (self.reader(self.tomo_raw_data_file_template.format(self.tomo_scan_id), dtype='flat', sli=[None, None, None], cfg=self.global_h.io_tomo_cfg).mean(axis=0) - 
                                 self.reader(self.tomo_raw_data_file_template.format(self.tomo_scan_id), dtype='dark', sli=[None, None, None], cfg=self.global_h.io_tomo_cfg).mean(axis=0))).astype(np.float32)
            self.raw_proj[:] = np.where(self.raw_proj<0, 1, self.raw_proj)[:]
            self.raw_proj[np.isinf(self.raw_proj)] = 1
                
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='tomo_raw_img_viewer')
            if (not data_state) | (not viewer_state):
                fiji_viewer_on(self.global_h, self, viewer_name='tomo_raw_img_viewer')
            self.global_h.tomo_fiji_windows['tomo_raw_img_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
                self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(self.raw_proj)), self.global_h.ImagePlusClass))
            self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
            self.global_h.tomo_fiji_windows['tomo_raw_img_viewer']['ip'].setRoi(self.tomo_col_s, self.tomo_sli_s, self.tomo_col_e-self.tomo_col_s, self.tomo_sli_e-self.tomo_sli_s)
    
    def L0_0_0_1_3_0_3_1_raw_proj_all_in_mem_checkbox_click(self, a):
        self.load_raw_in_mem = a['owner'].value
        fiji_viewer_off(self.global_h, self, viewer_name='tomo_raw_img_viewer')
    
    def L0_0_0_1_3_0_3_2_raw_proj_viewer_close_button_click(self, a):
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='tomo_raw_img_viewer')
        if viewer_state is not None:
            self.set_rec_param_from_widgets()
            x = self.global_h.tomo_fiji_windows['tomo_raw_img_viewer']['ip'].getRoi().getFloatPolygon().xpoints
            y = self.global_h.tomo_fiji_windows['tomo_raw_img_viewer']['ip'].getRoi().getFloatPolygon().ypoints
            # self.hs["L[0][0][0][1][0][1][3][0]_sli_start_text"].value = int(y[0])
            # self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].value = int(y[2])
            # self.hs["L[0][0][0][1][0][1][3][2]_col_start_text"].value = int(x[0])
            # self.hs["L[0][0][0][1][0][1][3][3]_col_end_text"].value = int(x[2])
            # self.hs["L[0][0][0][1][0][1][0][2]_cen_win_left_text"].value = int((x[2]+x[0])/2 - 40)            
            self.hs["L[0][0][0][1][0][1][6][4]_auto_ref_col_start_text"].value = int(x[0])
            self.hs["L[0][0][0][1][0][1][6][5]_auto_ref_col_end_text"].value = int(x[2])        
        fiji_viewer_off(self.global_h, self, viewer_name='tomo_raw_img_viewer')

    def L0_0_0_1_3_0_0_0_cen_offset_range_slider_change(self, a):
        self.raw_proj_0[:] = ((self.reader(self.tomo_raw_data_file_template.format(self.tomo_scan_id), dtype='data', sli=[[0, 1], None, None], cfg=self.global_h.io_tomo_cfg) - 
                               self.reader(self.tomo_raw_data_file_template.format(self.tomo_scan_id), dtype='dark', sli=[None, None, None], cfg=self.global_h.io_tomo_cfg).mean(axis=0)) /
                              (self.reader(self.tomo_raw_data_file_template.format(self.tomo_scan_id), dtype='flat', sli=[None, None, None], cfg=self.global_h.io_tomo_cfg).mean(axis=0) - 
                               self.reader(self.tomo_raw_data_file_template.format(self.tomo_scan_id), dtype='dark', sli=[None, None, None], cfg=self.global_h.io_tomo_cfg).mean(axis=0))).astype(np.float32)
        self.raw_proj_0[:] = np.where(self.raw_proj_0<0, 1, self.raw_proj_0)[:]
        self.raw_proj_0[np.isinf(self.raw_proj_0)] = 1
        self.raw_proj_180[:] = ((self.reader(self.tomo_raw_data_file_template.format(self.tomo_scan_id), dtype='data', sli=[-1, None, None], cfg=self.global_h.io_tomo_cfg) - 
                                 self.reader(self.tomo_raw_data_file_template.format(self.tomo_scan_id), dtype='dark', sli=[None, None, None], cfg=self.global_h.io_tomo_cfg).mean(axis=0)) /
                                (self.reader(self.tomo_raw_data_file_template.format(self.tomo_scan_id), dtype='flat', sli=[None, None, None], cfg=self.global_h.io_tomo_cfg).mean(axis=0) - 
                                 self.reader(self.tomo_raw_data_file_template.format(self.tomo_scan_id), dtype='dark', sli=[None, None, None], cfg=self.global_h.io_tomo_cfg).mean(axis=0))).astype(np.float32)
        self.raw_proj_180[:] = np.where(self.raw_proj_180<0, 1, self.raw_proj_180)[:]
        self.raw_proj_180[np.isinf(self.raw_proj_180)] = 1
        
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='tomo_0&180_viewer')
        if (not data_state) | (not viewer_state):
            fiji_viewer_on(self.global_h, self, viewer_name='tomo_0&180_viewer')
        self.global_h.tomo_fiji_windows['tomo_0&180_viewer']['ip'].setImage(self.global_h.ij.convert().convert(
            self.global_h.ij.dataset().create(self.global_h.ij.py.to_java(np.roll(self.raw_proj_180[:, ::-1], a['owner'].value, axis=1) - self.raw_proj_0)),
            self.global_h.ImagePlusClass))
        self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
    
    def L0_0_0_1_3_0_0_1_cen_offset_confirm_button_click(self, a):
        self.manual_cen = self.hs["L[0][0][0][1][3][0][0][0]_cen_offset_range_slider"].value/2.
        self.hs["L[0][0][0][1][0][1][0][1]_rot_cen_text"].value = self.manual_cen + self.data_info["img_dim"][2]/2
        self.hs["L[0][0][0][1][0][1][0][2]_cen_win_left_text"].value = self.hs["L[0][0][0][1][0][1][0][1]_rot_cen_text"].value - 10
        self.hs["L[0][0][0][1][0][1][0][3]_cen_win_wz_text"].value = 20
    
    def L0_0_0_1_3_0_0_2_cen_viewer_close_button_click(self, a):
        fiji_viewer_off(self.global_h, self, viewer_name='tomo_0&180_viewer')
    
    def L0_0_0_1_3_0_1_0_trial_cen_slider_change(self, a):
        data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='tomo_cen_review_viewer')
        if (not data_state) | (not viewer_state):
            fiji_viewer_on(self.global_h, self, viewer_name='tomo_cen_review_viewer')            
        self.global_h.tomo_fiji_windows['tomo_cen_review_viewer']['ip'].setSlice(a['owner'].value)
        self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""") 
    
    def L0_0_0_1_3_0_1_1_trial_cen_confirm_button_click(self, a):
        self.trial_cen = self.hs["L[0][0][0][1][0][1][0][2]_cen_win_left_text"].value + \
            (self.global_h.tomo_fiji_windows['tomo_cen_review_viewer']['ip'].getZ()-1)*0.5
        self.hs["L[0][0][0][1][0][1][0][1]_rot_cen_text"].value = self.trial_cen
        
        self.read_alg_param_widgets()
        self.set_rec_param_from_widgets()
        self.set_config_from_rec_param()

        try:
            with open(self.tomo_trial_cen_dict_fn, 'w+') as f:
                tem = json.load(f)
                tem[str(self.tomo_recon_param_dict["data_params"]["scan_id"])] = self.tomo_recon_param_dict
                tem[str(self.tomo_recon_param_dict["data_params"]["scan_id"])]['file_params']['io_confg']['customized_reader']['user_tomo_reader'] = ""
                tem[str(self.tomo_recon_param_dict["data_params"]["scan_id"])]['file_params']['reader'] = ""
                tem[str(self.tomo_recon_param_dict["data_params"]["scan_id"])]['file_params']['info_reader'] = ""
                json.dump(tem, f)
        except:
            tem = {}
            tem[str(self.tomo_recon_param_dict["data_params"]["scan_id"])] = self.tomo_recon_param_dict
            tem[str(self.tomo_recon_param_dict["data_params"]["scan_id"])]['file_params']['io_confg']['customized_reader']['user_tomo_reader'] = ""
            tem[str(self.tomo_recon_param_dict["data_params"]["scan_id"])]['file_params']['reader'] = ""
            tem[str(self.tomo_recon_param_dict["data_params"]["scan_id"])]['file_params']['info_reader'] = ""
            with open(self.tomo_trial_cen_dict_fn, 'w') as f:
                json.dump(tem, f)
               
    def L0_0_0_1_3_0_1_2_trial_cen_viewer_close_button_click(self, a):
        fiji_viewer_off(self.global_h, self, viewer_name='tomo_cen_review_viewer')  
    
    def L0_0_0_1_3_0_2_0_vol_sli_slider_change(self, a):
        pass
    
    def L0_0_0_1_3_0_2_1_vol_sli_viewer_close_button_click(self, a):
        pass
    
    def L0_0_1_0_0_2_1_recon_chunk_sz_text_change(self, a):
        if a['owner'].value < self.hs['L[0][0][1][0][0][2][2]_recon_margin_sz_text'].value*2:
            a['owner'].value = self.hs['L[0][0][1][0][0][2][2]_recon_margin_sz_text'].value*2 + 1
        self.tomo_chunk_sz = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_2_2_recon_margin_sz_text_change(self, a):
        if 2*a['owner'].value > self.hs['L[0][0][1][0][0][2][1]_recon_chunk_sz_text']:
            a['owner'].value = int(self.hs['L[0][0][1][0][0][2][1]_recon_chunk_sz_text']/2) - 1
        self.tomo_margin = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_0_0_filter_config_box_left_box_filter_dropdown_change(self, a):
        self.tomo_left_box_selected_flt = a["owner"].value
        self.set_flt_param_widgets()
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_0_1_filter_config_box_left_box_move_button_change(self, a):
        self.read_flt_param_widgets()
        if ((len(self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].options) == 1) &
            (self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].options[0] == "None")):
            self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].options = [self.tomo_left_box_selected_flt,]
            self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].value = [self.tomo_left_box_selected_flt,]
            self.tomo_right_filter_dict[0] = {'filter_name':self.tomo_left_box_selected_flt, 'params':self.flt_param_dict}
        else:
            a = list(self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].options)
            a.append(self.tomo_left_box_selected_flt)
            self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].options = a
            self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].value = \
                self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].options
            idx = len(a) - 1
            self.tomo_right_filter_dict[idx] = {'filter_name':self.tomo_left_box_selected_flt, 'params':self.flt_param_dict}
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_0_2_filter_config_box_left_box_p00_change(self, a):
        self.tomo_left_box_selected_flt_p00 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_0_3_filter_config_box_left_box_p01_change(self, a):
        self.tomo_left_box_selected_flt_p01 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_0_4_filter_config_box_left_box_p02_change(self, a):
        self.tomo_left_box_selected_flt_p02 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_0_5_filter_config_box_left_box_p03_change(self, a):
        self.tomo_left_box_selected_flt_p03 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_0_6_filter_config_box_left_box_p04_change(self, a):
        self.tomo_left_box_selected_flt_p04 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_0_7_filter_config_box_left_box_p05_change(self, a):
        self.tomo_left_box_selected_flt_p05 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_0_8_filter_config_box_left_box_p06_change(self, a):
        self.tomo_left_box_selected_flt_p06 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_0_9_filter_config_box_left_box_p07_change(self, a):
        self.tomo_left_box_selected_flt_p07 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_0_10_filter_config_box_left_box_p08_change(self, a):
        self.tomo_left_box_selected_flt_p08 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_0_11_filter_config_box_left_box_p09_change(self, a):
        self.tomo_left_box_selected_flt_p09 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_0_12_filter_config_box_left_box_p10_change(self, a):
        self.tomo_left_box_selected_flt_p10 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_0_13_filter_config_box_left_box_p11_change(self, a):
        self.tomo_left_box_selected_flt_p11 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_1_0_filter_config_box_right_box_selectmultiple_change(self, a):
        self.tomo_right_list_filter = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_1_1_filter_config_box_right_box_move_up_button_change(self, a):
        if (len(self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].options) == 1):
            pass
        else:
            a = np.array(self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].options)
            idxs = np.array(self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].index)
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
            self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].options = list(a)
            self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].value = list(a[idxs])
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_1_2_filter_config_box_right_box_move_dn_button_change(self, a):
        if (len(self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].options) == 1):
            pass
        else:
            a = np.array(self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].options)
            idxs = np.array(self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].index)
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
            self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].options = list(a)
            self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].value = list(a[idxs])
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_1_3_filter_config_box_right_box_remove_button_change(self, a):
        a = list(self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].options)
        idxs = list(self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].index)
        print(idxs)
        d = {}
        for b in sorted(idxs, reverse=True):
            print(b)
            del a[b]
            del self.tomo_right_filter_dict[b]
        if len(a)>0:
            self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].options = list(a)
            self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].value = [a[0],]
            cnt = 0
            for ii in sorted(self.tomo_right_filter_dict.keys()):
                d[cnt] = self.tomo_right_filter_dict[ii]
                cnt += 1
            self.tomo_right_filter_dict = d
        else:
            self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].options = ["None",]
            self.hs["L[0][0][1][0][0][1][1][0]_filter_config_right_box_selectmultiple"].value = ["None",]
            self.tomo_right_filter_dict = {0:{}}
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_0_1_1_4_filter_config_box_right_box_finish_button_change(self, a):
        pass

    def L0_0_1_0_0_1_3_filter_config_box_confirm_button(self, a):
        """
        enforce self.tomo_recon_param_dict has same structure as that in tomo_recon_tools
        """
        self.read_alg_param_widgets()
        self.set_rec_param_from_widgets()
        self.set_config_from_rec_param()

        self.tomo_data_configured = True
        self.boxes_logic()
        self.tomo_compound_logic()

    def L0_0_1_0_1_1_1_recon_button_click(self, a):
        self.recon_finish = run_engine(**self.tomo_recon_param_dict)
        if (self.tomo_recon_type == "Trial Center") & (self.recon_finish == 0):
            boxes = ["L[0][0][0][1][3][0][1]_data_preview_box1"]
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            boxes = ["L[0][0][0][1][3][0][2]_data_preview_box2"]
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            self.hs["L[0][0][0][1][3][0][1][0]_trial_cen_slider"].value = 0
            self.hs["L[0][0][0][1][3][0][1][0]_trial_cen_slider"].min = 0
            self.hs["L[0][0][0][1][3][0][1][0]_trial_cen_slider"].max = int(2*self.tomo_cen_win_w - 1)           
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='tomo_cen_review_viewer')
            if (not data_state) | (not viewer_state):
                fiji_viewer_on(self.global_h, self, viewer_name='tomo_cen_review_viewer') 
            self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")            
        elif (self.tomo_recon_type == "Vol Recon: Single") & (self.recon_finish == 0):
            with open(self.tomo_recon_dict_fn, 'w') as f:
                tem = deepcopy(self.tomo_recon_param_dict)
                tem['file_params']['reader'] = ''
                tem['file_params']['info_reader'] = ''
                json.dump(tem, f)
            boxes = ["L[0][0][0][1][3][0][1]_data_preview_box1"]
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            boxes = ["L[0][0][0][1][3][0][2]_data_preview_box2"]
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            self.hs["L[0][0][0][1][3][0][2][0]_vol_sli_slider"].value = 0
            self.hs["L[0][0][0][1][3][0][2][0]_vol_sli_slider"].min = 0
            self.hs["L[0][0][0][1][3][0][2][0]_vol_sli_slider"].max = int(self.hs["L[0][0][0][1][0][1][3][1]_sli_end_text"].value -
                                                                          self.hs["L[0][0][0][1][0][1][3][0]_sli_start_text"].value -
                                                                          self.hs["L[0][0][1][0][0][2][2]_recon_margin_sz_text"].value)               
            data_state, viewer_state = fiji_viewer_state(self.global_h, self, viewer_name='tomo_recon_viewer')
            if (not data_state) | (not viewer_state):
                fiji_viewer_on(self.global_h, self, viewer_name='tomo_recon_viewer') 
            self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")  
        elif self.recon_finish == -1:
            print('something runs wrong during the reconstruction.')
            
        











