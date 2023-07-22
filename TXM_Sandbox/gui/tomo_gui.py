#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:33:11 2020

@author: xiao
"""

import os, h5py, json, time
import numpy as np
import matplotlib.pyplot as plt
import napari
from copy import deepcopy

from ipywidgets import widgets, GridspecLayout
from collections import OrderedDict

from .gui_components import (
    SelectFilesButton,
    get_handles,
    msgbox,
    enable_disable_boxes,
    check_file_availability,
    get_raw_img_info,
    restart,
    fiji_viewer_state,
    fiji_viewer_on,
    fiji_viewer_off,
    create_widget,
    gen_external_py_script,
    update_json_content,
)
from ..utils.tomo_recon_tools import (
    FILTERLIST,
    TOMO_RECON_PARAM_DICT,
    run_engine,
    read_data,
    normalize,
)
from ..utils.io import data_reader, tomo_h5_reader, data_info, tomo_h5_info

FILTER_PARAM_DICT = dict(
    OrderedDict(
        {
            "phase retrieval": {
                0: [
                    "filter",
                    ["paganin", "bronnikov"],
                    "filter: filter type used in phase retrieval",
                ],
                1: [
                    "pad",
                    ["True", "False"],
                    "pad: boolean, if pad the data before phase retrieval filtering",
                ],
                6: ["pixel_size", 6.5e-5, "pixel_size: in cm unit"],
                7: ["dist", 15.0, "dist: sample-detector distance in cm"],
                8: ["energy", 35.0, "energy: x-ray energy in keV"],
                9: [
                    "alpha",
                    1e-2,
                    "alpha: beta/delta, wherr n = (1-delta + i*beta) is x-ray rafractive index in the sample",
                ],
            },
            "flatting bkg": {
                6: [
                    "air",
                    30,
                    "air: number of pixels on the both sides of projection images where is sample free region. This region will be used to correct background nonunifromness",
                ]
            },
            "remove cupping": {
                6: [
                    "cc",
                    0.5,
                    "cc: constant that is subtracted from the logrithm of the normalized images. This is for correcting the cup-like background in the case when the sample size is much larger than the image view",
                ]
            },
            "stripe_removal: vo": {
                6: ["snr", 4, "snr: signal-to-noise ratio"],
                7: ["la_size", 81, "la_size: large ring's width in pixel"],
                8: ["sm_size", 21, "sm_size: small ring's width in pixel"],
            },
            "stripe_removal: ti": {
                6: ["nblock", 1, "nblock: "],
                7: ["alpha", 5, "alpha: "],
            },
            "stripe_removal: sf": {6: ["size", 31, "size: "]},
            "stripe_removal: fw": {
                0: [
                    "pad",
                    ["True", "False"],
                    "pad: boolean, if padding data before filtering",
                ],
                1: [
                    "wname",
                    [
                        "db5",
                        "db1",
                        "db2",
                        "db3",
                        "sym2",
                        "sym6",
                        "haar",
                        "gaus1",
                        "gaus2",
                        "gaus3",
                        "gaus4",
                    ],
                    "wname: wavelet name",
                ],
                6: ["level", 6, "level: how many of level of wavelet transforms"],
                7: [
                    "sigma",
                    2,
                    "sigma: sigam of gaussian filter in image Fourier space",
                ],
            },
            "denoise: median": {
                6: ["size angle", 1, "median kernel size along rotation angle axis"],
                7: [
                    "size y",
                    5,
                    "median kernel size along projection image vertical axis",
                ],
                8: [
                    "size x",
                    5,
                    "median kernel size along projection image horizontal axis",
                ],
            },
            "denoise: wiener": {
                0: [
                    "reg",
                    ["None"],
                    "reg: The regularisation operator. The Laplacian by default. It can be an impulse response or a transfer function, as for the psf",
                ],
                1: ["is_real", ["True", "False"], "is_real: "],
                2: [
                    "clip",
                    ["True", "False"],
                    "clip: True by default. If true, pixel values of the result above 1 or under -1 are thresholded for skimage pipeline compatibility",
                ],
                6: [
                    "psf",
                    2,
                    "psf: The impulse response (input image’s space) or the transfer function (Fourier space). Both are accepted. The transfer function is automatically recognized as being complex (np.iscomplexobj(psf))",
                ],
                7: ["balance", 0.3, "balance: "],
            },
            "denoise: unsupervised_wiener": {
                0: [
                    "reg",
                    ["None"],
                    "reg: The regularisation operator. The Laplacian by default. It can be an impulse response or a transfer function, as for the psf. Shape constraint is the same as for the psf parameter",
                ],
                1: [
                    "is_real",
                    ["True", "False"],
                    "is_real: True by default. Specify if psf and reg are provided with hermitian hypothesis, that is only half of the frequency plane is provided (due to the redundancy of Fourier transform of real signal). It’s apply only if psf and/or reg are provided as transfer function. For the hermitian property see uft module or np.fft.rfftn",
                ],
                2: [
                    "clip",
                    ["True", "False"],
                    "clip: True by default. If True, pixel values of the result above 1 or under -1 are thresholded for skimage pipeline compatibility",
                ],
                3: [
                    "user_params",
                    ["None"],
                    "user_params: Dictionary of parameters for the Gibbs sampler. See below",
                ],
                6: [
                    "psf",
                    2,
                    "psf: Point Spread Function. This is assumed to be the impulse response (input image space) if the data-type is real, or the transfer function (Fourier space) if the data-type is complex. There is no constraints on the shape of the impulse response. The transfer function must be of shape (M, N) if is_real is True, (M, N // 2 + 1) otherwise (see np.fft.rfftn)",
                ],
            },
            "denoise: denoise_nl_means": {
                0: [
                    "multichannel",
                    ["False", "True"],
                    "multichannel: Whether the last axis of the image is to be interpreted as multiple channels or another spatial dimension",
                ],
                1: [
                    "fast_mode",
                    ["True", "False"],
                    "fast_mode: If True (default value), a fast version of the non-local means algorithm is used. If False, the original version of non-local means is used. See the Notes section for more details about the algorithms",
                ],
                6: ["patch_size", 5, "patch_size: Size of patches used for denoising"],
                7: [
                    "patch_distance",
                    7,
                    "patch_distance: Maximal distance in pixels where to search patches used for denoising",
                ],
                8: [
                    "h",
                    0.1,
                    "h: Cut-off distance (in gray levels). The higher h, the more permissive one is in accepting patches. A higher h results in a smoother image, at the expense of blurring features. For a Gaussian noise of standard deviation sigma, a rule of thumb is to choose the value of h to be sigma of slightly less",
                ],
                9: [
                    "sigma",
                    0.05,
                    "sigma: The standard deviation of the (Gaussian) noise. If provided, a more robust computation of patch weights is computed that takes the expected noise variance into account (see Notes below)",
                ],
            },
            "denoise: denoise_tv_bregman": {
                0: [
                    "multichannel",
                    ["False", "True"],
                    "multichannel: Apply total-variation denoising separately for each channel. This option should be true for color images, otherwise the denoising is also applied in the channels dimension",
                ],
                1: [
                    "isotrophic",
                    ["True", "False"],
                    "isotrophic: Switch between isotropic and anisotropic TV denoisin",
                ],
                6: [
                    "weight",
                    1.0,
                    "weight: Denoising weight. The smaller the weight, the more denoising (at the expense of less similarity to the input). The regularization parameter lambda is chosen as 2 * weight",
                ],
                7: [
                    "max_iter",
                    100,
                    "max_iter: Maximal number of iterations used for the optimization",
                ],
                8: [
                    "eps",
                    0.001,
                    "eps: Relative difference of the value of the cost function that determines the stop criterion.",
                ],
            },
            "denoise: denoise_tv_chambolle": {
                0: [
                    "multichannel",
                    ["False", "True"],
                    "multichannel: Apply total-variation denoising separately for each channel. This option should be true for color images, otherwise the denoising is also applied in the channels dimension",
                ],
                6: [
                    "weight",
                    0.1,
                    "weight: Denoising weight. The greater weight, the more denoising (at the expense of fidelity to input).",
                ],
                7: [
                    "n_iter_max",
                    100,
                    "n_iter_max: Maximal number of iterations used for the optimization",
                ],
                8: [
                    "eps",
                    0.002,
                    "eps: Relative difference of the value of the cost function that determines the stop criterion",
                ],
            },
            "denoise: denoise_bilateral": {
                0: [
                    "win_size",
                    ["None"],
                    "win_size: Window size for filtering. If win_size is not specified, it is calculated as max(5, 2 * ceil(3 * sigma_spatial) + 1)",
                ],
                1: [
                    "sigma_color",
                    ["None"],
                    "sigma_color: Standard deviation for grayvalue/color distance (radiometric similarity). A larger value results in averaging of pixels with larger radiometric differences. Note, that the image will be converted using the img_as_float function and thus the standard deviation is in respect to the range [0, 1]. If the value is None the standard deviation of the image will be used",
                ],
                2: [
                    "multichannel",
                    ["False", "True"],
                    "multichannel: Whether the last axis of the image is to be interpreted as multiple channels or another spatial dimension",
                ],
                3: [
                    "mode",
                    ["constant", "edge", "symmetric", "reflect", "wrap"],
                    "mode: How to handle values outside the image borders. See numpy.pad for detail",
                ],
                6: [
                    "sigma_spatial",
                    1,
                    "sigma_spatial: Standard deviation for range distance. A larger value results in averaging of pixels with larger spatial difference",
                ],
                7: [
                    "bins",
                    10000,
                    "bins: Number of discrete values for Gaussian weights of color filtering. A larger value results in improved accuracy",
                ],
                8: [
                    "cval",
                    0,
                    "cval: Used in conjunction with mode ‘constant’, the value outside the image boundaries",
                ],
            },
            "denoise: denoise_wavelet": {
                0: [
                    "wavelet",
                    [
                        "db1",
                        "db2",
                        "db3",
                        "db5",
                        "sym2",
                        "sym6",
                        "haar",
                        "gaus1",
                        "gaus2",
                        "gaus3",
                        "gaus4",
                    ],
                    "wavelet: The type of wavelet to perform and can be any of the options pywt.wavelist outputs",
                ],
                1: [
                    "mode",
                    ["soft"],
                    "mode: An optional argument to choose the type of denoising performed. It noted that choosing soft thresholding given additive noise finds the best approximation of the original image",
                ],
                2: [
                    "multichannel",
                    ["False", "True"],
                    "multichannel: Apply wavelet denoising separately for each channel (where channels correspond to the final axis of the array)",
                ],
                3: [
                    "convert2ycbcr",
                    ["False", "True"],
                    "convert2ycbcr: If True and multichannel True, do the wavelet denoising in the YCbCr colorspace instead of the RGB color space. This typically results in better performance for RGB images",
                ],
                4: [
                    "method",
                    ["BayesShrink"],
                    "method: Thresholding method to be used. The currently supported methods are 'BayesShrink' [1] and 'VisuShrink' [2]. Defaults to 'BayesShrink'",
                ],
                6: [
                    "sigma",
                    1,
                    "sigma: The noise standard deviation used when computing the wavelet detail coefficient threshold(s). When None (default), the noise standard deviation is estimated via the method in [2]",
                ],
                7: [
                    "wavelet_levels",
                    3,
                    "wavelet_levels: The number of wavelet decomposition levels to use. The default is three less than the maximum number of possible decomposition levels",
                ],
            },
        }
    )
)

ALG_PARAM_DICT = dict(
    OrderedDict(
        {
            "gridrec": {
                0: [
                    "filter_name",
                    [
                        "parzen",
                        "shepp",
                        "cosine",
                        "hann",
                        "hamming",
                        "ramlak",
                        "butterworth",
                        "none",
                    ],
                    "filter_name: filter that is used in frequency space",
                    str,
                ]
            },
            "sirt": {
                3: [
                    "num_gridx",
                    1280,
                    "num_gridx: number of the reconstructed slice image along x direction",
                    int,
                ],
                4: [
                    "num_gridy",
                    1280,
                    "num_gridy: number of the reconstructed slice image along y direction",
                    int,
                ],
                5: [
                    "num_iter",
                    10,
                    "num_iter: number of reconstruction iterations",
                    int,
                ],
            },
            "tv": {
                3: [
                    "num_gridx",
                    1280,
                    "num_gridx: number of the reconstructed slice image along x direction",
                    int,
                ],
                4: [
                    "num_gridy",
                    1280,
                    "num_gridy: number of the reconstructed slice image along y direction",
                    int,
                ],
                5: [
                    "num_iter",
                    10,
                    "num_iter: number of reconstruction iterations",
                    int,
                ],
                6: ["reg_par", 0.1, "reg_par: relaxation factor in tv regulation", int],
            },
            "mlem": {
                3: [
                    "num_gridx",
                    1280,
                    "num_gridx: number of the reconstructed slice image along x direction",
                    int,
                ],
                4: [
                    "num_gridy",
                    1280,
                    "num_gridy: number of the reconstructed slice image along y direction",
                    int,
                ],
                5: [
                    "num_iter",
                    10,
                    "num_iter: number of reconstruction iterations",
                    int,
                ],
            },
            "astra": {
                0: ["method", ["EM_CUDA"], "method: astra reconstruction methods", str],
                1: [
                    "proj_type",
                    ["cuda"],
                    "proj_type: projection calculation options used in astra",
                    str,
                ],
                2: [
                    "extra_options",
                    ["MinConstraint"],
                    "extra_options: extra constraints used in the reconstructions. you need to set p03 for a MinConstraint level",
                    str,
                ],
                3: [
                    "extra_options_param",
                    -0.1,
                    "extra_options_param: parameter used together with extra_options",
                    np.float32,
                ],
                4: [
                    "num_iter",
                    50,
                    "num_iter: number of reconstruction iterations",
                    int,
                ],
            },
        }
    )
)


class tomo_recon_gui:
    def __init__(self, parent_h, form_sz=[650, 740]):
        self.hs = {}
        self.form_sz = form_sz
        self.global_h = parent_h

        if self.global_h.io_tomo_cfg["use_h5_reader"]:
            self.reader = data_reader(tomo_h5_reader)
            self.info_reader = data_info(tomo_h5_info)
        else:
            from ..external.user_io import user_tomo_reader, user_tomo_info_reader

            self.reader = data_reader(user_tomo_reader)
            self.info_reader = data_info(user_tomo_info_reader)

        self.tomo_recon_external_command_name = os.path.join(
            os.path.abspath(os.path.curdir), "tomo_recon_external_command.py"
        )
        self.tomo_msg_on = False

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

        self.tomo_recon_type = "Trial Cent"
        self.tomo_use_debug = False
        self.tomo_use_alt_flat = False
        self.tomo_use_alt_dark = False
        self.tomo_use_fake_flat = False
        self.tomo_use_fake_dark = False
        self.tomo_use_blur_flat = False
        self.tomo_use_rm_zinger = False
        self.tomo_use_mask = True
        self.tomo_use_read_config = True
        self.tomo_use_downsample = False
        self.tomo_is_wedge = False
        self.tomo_use_wedge_ang_auto_det = False
        # self.tomo_read_config = False

        self.tomo_right_filter_dict = {0: {}}

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
        self.tomo_fake_flat_roi = None
        self.tomo_sli_s = 1280
        self.tomo_sli_e = 1300
        self.tomo_col_s = 0
        self.tomo_col_e = 100
        self.tomo_chunk_sz = 200
        self.tomo_margin = 15
        self.tomo_flat_blur_kernel = 1
        self.tomo_zinger_val = 500
        self.tomo_mask_ratio = 1
        self.tomo_wedge_missing_s = 500
        self.tomo_wedge_missing_e = 600
        self.tomo_wedge_auto_ref_col_s = 0
        self.tomo_wedge_auto_ref_col_e = 10
        self.tomo_wedge_ang_auto_det_thres = 0.1
        self.data_info = {}
        self.napari_viewer = None

        self.alg_param_dict = {}

    def build_gui(self):
        #################################################################################################################
        #                                                                                                               #
        #                                                    TOMO RECON                                                 #
        #                                                                                                               #
        #################################################################################################################
        ## ## ## define 2D_XANES_tabs layout -- start
        layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": f"{self.form_sz[0] - 136}px",
        }
        self.hs["Config&Input form"] = widgets.VBox()
        self.hs["Filter&Recon tab"] = widgets.Tab()
        self.hs["Reg&Rev form"] = widgets.VBox()
        self.hs["Analysis&Disp form"] = widgets.VBox()

        self.hs["Config&Input form"].layout = layout
        self.hs["Filter&Recon tab"].layout = {
            "border": "3px solid #FFCC00",
            "width": "auto",
            "height": "auto",
        }
        self.hs["Reg&Rev form"].layout = layout
        self.hs["Analysis&Disp form"].layout = layout

        ## ## ## define boxes in config_input_form -- start
        ## ## ## ## define functional widget tabs in each sub-tab - configure file settings -- start
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.35 * (self.form_sz[0] - 136)}px",
        }
        self.hs["SelFile&Path box"] = widgets.VBox()
        self.hs["SelFile&Path box"].layout = layout

        ## ## ## ## ## label configure file settings box
        layout = {"border": "3px solid #FFCC00", "width": "auto", "height": "auto"}
        self.hs["SelFile&PathTitle box"] = widgets.HBox()
        self.hs["SelFile&PathTitle box"].layout = layout
        self.hs["SelFile&PathTitle label"] = widgets.HTML(
            "<span style='color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);'>"
            + "Config Dirs & Files"
            + "</span>"
        )
        layout = {"background-color": "white", "color": "cyan", "left": "39%"}
        self.hs["SelFile&PathTitle label"].layout = layout
        self.hs["SelFile&PathTitle box"].children = [self.hs["SelFile&PathTitle label"]]

        ## ## ## ## ## raw h5 top directory
        layout = {"border": "3px solid #FFCC00", "width": "auto", "height": "auto"}
        self.hs["SelRaw box"] = widgets.HBox()
        self.hs["SelRaw box"].layout = layout
        self.hs["SelRawH5TopDir text"] = widgets.Text(
            value="Choose raw h5 top dir ...", description="", disabled=True
        )
        layout = {"width": "66%", "display": "inline_flex"}
        self.hs["SelRawH5TopDir text"].layout = layout
        self.hs["SelRawH5TopDir btn"] = SelectFilesButton(
            option="askdirectory", text_h=self.hs["SelRawH5TopDir text"]
        )
        self.hs["SelRawH5TopDir btn"].description = "Raw Top Dir"
        self.hs[
            "SelRawH5TopDir btn"
        ].description_tooltip = (
            "Select the top directory in which the raw h5 files are located."
        )
        layout = {"width": "15%"}
        self.hs["SelRawH5TopDir btn"].layout = layout
        self.hs["SelRawH5TopDir btn"].on_click(self.SelRawH5TopDir_btn_clk)
        self.hs["SelRaw box"].children = [
            self.hs["SelRawH5TopDir text"],
            self.hs["SelRawH5TopDir btn"],
        ]

        ## ## ## ## ##  save recon directory
        layout = {"border": "3px solid #FFCC00", "width": "auto", "height": "auto"}
        self.hs["SelSavRecon box"] = widgets.HBox()
        self.hs["SelSavRecon box"].layout = layout
        self.hs["SelSavReconDir text"] = widgets.Text(
            value="Select top directory where data_center directory will be created...",
            description="",
            disabled=True,
        )
        layout = {"width": "66%", "display": "inline_flex"}
        self.hs["SelSavReconDir text"].layout = layout
        self.hs["SelSavReconDir btn"] = SelectFilesButton(
            option="askdirectory", text_h=self.hs["SelSavReconDir text"]
        )
        self.hs["SelSavReconDir btn"].description = "Save Rec File"
        self.hs["SelSavReconDir btn"].disabled = False
        layout = {"width": "15%"}
        self.hs["SelSavReconDir btn"].layout = layout
        self.hs["SelSavReconDir btn"].on_click(self.SelSavReconDir_btn_clk)
        self.hs["SelSavRecon box"].children = [
            self.hs["SelSavReconDir text"],
            self.hs["SelSavReconDir btn"],
        ]

        ## ## ## ## ##  save debug directory
        layout = {"border": "3px solid #FFCC00", "width": "auto", "height": "auto"}
        self.hs["SelSavDebug box"] = widgets.HBox()
        self.hs["SelSavDebug box"].layout = layout
        self.hs["SelSavDebugDir text"] = widgets.Text(
            value="Debug is disabled...", description="", disabled=True
        )
        layout = {"width": "66%", "display": "inline_flex"}
        self.hs["SelSavDebugDir text"].layout = layout
        self.hs["SelSavDebugDir btn"] = SelectFilesButton(
            option="askdirectory", text_h=self.hs["SelSavDebugDir text"]
        )
        self.hs["SelSavDebugDir btn"].description = "Save Debug Dir"
        self.hs["SelSavDebugDir btn"].disabled = True
        layout = {"width": "15%"}
        self.hs["SelSavDebugDir btn"].layout = layout
        self.hs["SavDebug chbx"] = widgets.Checkbox(
            value=False, description="Save Debug", disabled=False, indent=False
        )
        layout = {"left": "1%", "width": "13%", "display": "inline_flex"}
        self.hs["SavDebug chbx"].layout = layout
        self.hs["SelSavDebugDir btn"].on_click(self.SelSavDebugDir_btn_clk)
        self.hs["SavDebug chbx"].observe(self.SavDebug_chbx_chg, names="value")
        self.hs["SelSavDebug box"].children = [
            self.hs["SelSavDebugDir text"],
            self.hs["SelSavDebugDir btn"],
            self.hs["SavDebug chbx"],
        ]

        ## ## ## ## ## confirm file configuration
        layout = {"border": "3px solid #FFCC00", "width": "auto", "height": "auto"}
        self.hs["SelFile&PathCfm box"] = widgets.HBox()
        self.hs["SelFile&PathCfm box"].layout = layout
        self.hs["SelFile&PathCfm text"] = widgets.Text(
            value="After setting directories, confirm to proceed ...",
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
            value="Trial Cent",
            options=["Trial Cent", "Vol Recon"],
            description="",
            disabled=False,
            description_tooltip="'Trial Cent': doing trial recon on a single slice to find rotation center; 'Vol Recon': doing volume recon of a series of  scan datasets.",
        )
        layout = {"width": "15%", "top": "0%"}
        self.hs["File&PathOptn drpdn"].layout = layout

        self.hs["File&PathOptn drpdn"].observe(
            self.FilePathOptn_drpdn_chg, names="value"
        )
        self.hs["SelFile&PathCfm box"].children = [
            self.hs["SelFile&PathCfm text"],
            self.hs["SelFile&PathCfm btn"],
            self.hs["File&PathOptn drpdn"],
        ]

        self.hs["SelFile&Path box"].children = [
            self.hs["SelFile&PathTitle box"],
            self.hs["SelRaw box"],
            self.hs["SelSavRecon box"],
            self.hs["SelSavDebug box"],
            self.hs["SelFile&PathCfm box"],
        ]
        ## ## ## ## bin widgets in hs["SelFile&Path box"] -- configure file settings -- end

        ## ## ## ## define widgets recon_options_box -- start
        layout = {"border": "3px solid #8855AA", "width": "auto", "height": "auto"}
        self.hs["Data tab"] = widgets.Tab()
        self.hs["Data tab"].layout = layout

        ## ## ## ## ## define sub-tabs in data_tab -- start
        layout = {"border": "3px solid #FFCC00", "width": "auto", "height": "auto"}
        self.hs["DataConfig tab"] = widgets.VBox()
        self.hs["DataConfig tab"].layout = layout
        self.hs["AlgConfig tab"] = widgets.VBox()
        self.hs["AlgConfig tab"].layout = layout
        self.hs["DataInfo tab"] = widgets.VBox()
        self.hs["DataInfo tab"].layout = layout
        self.hs["DataPrev tab"] = widgets.VBox()
        self.hs["DataPrev tab"].layout = layout
        self.hs["VolRecon tab"] = widgets.VBox()
        self.hs["VolRecon tab"].layout = layout
        self.hs["Data tab"].children = [
            self.hs["DataConfig tab"],
            self.hs["AlgConfig tab"],
            self.hs["DataInfo tab"],
            self.hs["DataPrev tab"],
            self.hs["VolRecon tab"],
        ]
        self.hs["Data tab"].set_title(0, "Data Config")
        self.hs["Data tab"].set_title(1, "Alg Config")
        self.hs["Data tab"].set_title(2, "Data Info")
        self.hs["Data tab"].set_title(3, "Data Preview")
        self.hs["Data tab"].set_title(4, "View Recon")
        ## ## ## ## ## define sub-tabs in data_tab -- end

        ## ## ## ## ## ## config data parameters in data_config_box in data_config_tab -- start
        layout = {"border": "3px solid #FFCC00", "width": "auto", "height": "auto"}
        self.hs["DataConfig box"] = widgets.VBox()
        self.hs["DataConfig box"].layout = layout

        ## ## ## ## ## ## ## config sub-boxes in data_config_box -- start
        layout = {"border": "3px solid #FFCC00", "height": "auto"}
        self.hs["ReconConfig box"] = widgets.HBox()
        self.hs["ReconConfig box"].layout = layout
        self.hs["ScanId drpdn"] = widgets.Dropdown(
            value=0, options=[0], description="Scan id", disabled=True
        )
        layout = {"width": "19%"}
        self.hs["ScanId drpdn"].layout = layout
        self.hs["RotCen text"] = widgets.BoundedFloatText(
            value=1280.0, min=0, max=2500, description="Center", disabled=True
        )
        layout = {"width": "19%"}
        self.hs["RotCen text"].layout = layout
        self.hs["CenWinLeft text"] = widgets.BoundedIntText(
            value=1240, min=0, max=2500, description="Cen Win L", disabled=True,
            tooltip="Center search window starting position relative to the image left handside edge."
        )
        layout = {"width": "19%"}
        self.hs["CenWinLeft text"].layout = layout
        self.hs["CenWinWz text"] = widgets.BoundedIntText(
            value=80, min=1, max=200, description="Cen Win W", disabled=True,
            tooltip="Center search window width"
        )
        layout = {"width": "19%"}
        self.hs["CenWinWz text"].layout = layout

        self.hs["ReadConfig_btn"] = SelectFilesButton(
            option="askopenfilename", **{"open_filetypes": (("json files", "*.json"),)}
        )
        layout = {"width": "15%", "height": "85%", "visibility": "hidden"}
        self.hs["ReadConfig_btn"].layout = layout
        self.hs["ReadConfig_btn"].disabled = True

        self.hs["UseConfig chbx"] = widgets.Checkbox(
            value=False,
            description="Use",
            description_tooltip="Use configuration read from the file",
            disabled=True,
            indent=False,
            layout={"width": "7%", "visibility": "hidden"},
        )

        self.hs["ScanId drpdn"].observe(self.ScanId_drpdn_chg, names="value")
        self.hs["RotCen text"].observe(self.RotCen_text_chg, names="value")
        self.hs["CenWinLeft text"].observe(self.CenWinLeft_text_chg, names="value")
        self.hs["CenWinWz text"].observe(self.CenWinWz_text_chg, names="value")
        self.hs["ReadConfig_btn"].on_click(self.ReadConfig_btn_clk)
        self.hs["UseConfig chbx"].observe(self.UseConfig_chbx_chg, names="value")
        self.hs["ReconConfig box"].children = [
            self.hs["ScanId drpdn"],
            self.hs["RotCen text"],
            self.hs["CenWinLeft text"],
            self.hs["CenWinWz text"],
            self.hs["ReadConfig_btn"],
            self.hs["UseConfig chbx"],
        ]

        layout = {"border": "3px solid #FFCC00", "height": "auto"}
        self.hs["RoiConfig box"] = widgets.HBox()
        self.hs["RoiConfig box"].layout = layout
        self.hs["RoiSliStart text"] = widgets.BoundedIntText(
            value=1280, min=0, max=2100, description="Sli Start", disabled=True
        )
        layout = {"width": "19%"}
        self.hs["RoiSliStart text"].layout = layout
        self.hs["RoiSliEnd text"] = widgets.BoundedIntText(
            value=1300, min=0, max=2200, description="Sli End", disabled=True
        )
        layout = {"width": "19%"}
        self.hs["RoiSliEnd text"].layout = layout
        self.hs["RoiColStart text"] = widgets.BoundedIntText(
            value=0, min=0, max=400, description="Col Start", disabled=True
        )
        layout = {"width": "19%"}
        self.hs["RoiColStart text"].layout = layout
        self.hs["RoiColEnd text"] = widgets.BoundedIntText(
            value=10, min=0, max=400, description="Col_End", disabled=True
        )
        layout = {"width": "19%"}
        self.hs["RoiColEnd text"].layout = layout
        self.hs["DnSampRat text"] = widgets.BoundedFloatText(
            value=1, description="Down Sam R", min=0.1, max=1.0, step=0.1, disabled=True
        )
        layout = {"width": "19%"}
        self.hs["DnSampRat text"].layout = layout

        self.hs["RoiSliStart text"].observe(self.RoiSliStart_text_chg, names="value")
        self.hs["RoiSliEnd text"].observe(self.RoiSliEnd_text_chg, names="value")
        self.hs["RoiColStart text"].observe(self.RoiColStart_text_chg, names="value")
        self.hs["RoiColEnd text"].observe(self.RoiColEnd_text_chg, names="value")
        self.hs["DnSampRat text"].observe(self.DnSampRat_text_chg, names="value")
        self.hs["RoiConfig box"].children = [
            self.hs["RoiSliStart text"],
            self.hs["RoiSliEnd text"],
            self.hs["RoiColStart text"],
            self.hs["RoiColEnd text"],
            self.hs["DnSampRat text"],
        ]

        layout = {"border": "3px solid #FFCC00", "height": "auto"}
        self.hs["AltFlatDarkOptn box"] = widgets.HBox()
        self.hs["AltFlatDarkOptn box"].layout = layout
        layout = {"width": "24%"}
        self.hs["UseAltFlat chbx"] = widgets.Checkbox(
            value=False, description="Alt Flat", disabled=True, indent=False
        )
        self.hs["UseAltFlat chbx"].layout = layout
        layout = {"width": "15%"}
        self.hs["AltFlatFile btn"] = SelectFilesButton(
            option="askopenfilename", **{"open_filetypes": (("h5 files", "*.h5"),)}
        )
        self.hs["AltFlatFile btn"].description = "Alt Flat File"
        self.hs["AltFlatFile btn"].disabled = True
        self.hs["AltFlatFile btn"].layout = layout
        layout = {"left": "9%", "width": "24%"}
        self.hs["UseAltDark chbx"] = widgets.Checkbox(
            value=False, description="Alt Dark", disabled=True, indent=False
        )
        self.hs["UseAltDark chbx"].layout = layout
        layout = {"left": "9%", "width": "15%"}
        self.hs["AltDarkFile btn"] = SelectFilesButton(
            option="askopenfilename", **{"open_filetypes": (("h5 files", "*.h5"),)}
        )
        self.hs["AltDarkFile btn"].description = "Alt Dark File"
        self.hs["AltDarkFile btn"].disabled = True
        self.hs["AltDarkFile btn"].layout = layout

        self.hs["UseAltFlat chbx"].observe(self.UseAltFlat_chbx_chg, names="value")
        self.hs["AltFlatFile btn"].observe(self.AltFlatFile_btn_clk, names="value")
        self.hs["UseAltDark chbx"].observe(self.UseAltDark_chbx_chg, names="value")
        self.hs["AltDarkFile btn"].observe(self.AltDarkFile_btn_clk, names="value")
        self.hs["AltFlatDarkOptn box"].children = [
            self.hs["UseAltFlat chbx"],
            self.hs["AltFlatFile btn"],
            self.hs["UseAltDark chbx"],
            self.hs["AltDarkFile btn"],
        ]

        layout = {"border": "3px solid #FFCC00", "height": "auto"}
        self.hs["FakeFlatDarkOptn box"] = widgets.HBox()
        self.hs["FakeFlatDarkOptn box"].layout = layout
        layout = {"width": "19%"}
        self.hs["UseFakeFlat chbx"] = widgets.Checkbox(
            value=False, description="Fake Flat", disabled=True, indent=False
        )
        self.hs["UseFakeFlat chbx"].layout = layout
        layout = {"width": "19%"}
        self.hs["FakeFlatVal text"] = widgets.BoundedFloatText(
            value=10000.0, description="Flat Val", min=100, max=65000, disabled=True
        )
        self.hs["FakeFlatVal text"].layout = layout
        layout = {"left": "10%", "width": "19%"}
        self.hs["UseFakeDark chbx"] = widgets.Checkbox(
            value=False, description="Fake Dark", disabled=True, indent=False
        )
        self.hs["UseFakeDark chbx"].layout = layout
        layout = {"left": "19%", "width": "19%"}
        self.hs["FakeDarkVal text"] = widgets.BoundedFloatText(
            value=100.0, description="Dark Val", min=0, max=500, disabled=True
        )
        self.hs["FakeDarkVal text"].layout = layout

        self.hs["UseFakeFlat chbx"].observe(self.UseFakeFlat_chbx_chg, names="value")
        self.hs["FakeFlatVal text"].observe(self.FakeFlatVal_text_chg, names="value")
        self.hs["UseFakeDark chbx"].observe(self.UseFakeDark_chbx_chg, names="value")
        self.hs["FakeDarkVal text"].observe(self.FakeDarkVal_text_chg, names="value")
        self.hs["FakeFlatDarkOptn box"].children = [
            self.hs["UseFakeFlat chbx"],
            self.hs["FakeFlatVal text"],
            self.hs["UseFakeDark chbx"],
            self.hs["FakeDarkVal text"],
        ]

        layout = {"border": "3px solid #FFCC00", "height": "auto"}
        self.hs["MiscOptn box"] = widgets.HBox()
        self.hs["MiscOptn box"].layout = layout
        layout = {"width": "10%"}
        self.hs["UseBlurFlat chbx"] = widgets.Checkbox(
            value=False, description="Blur Flat", disabled=True, indent=False
        )
        self.hs["UseBlurFlat chbx"].layout = layout
        layout = {"width": "17%"}
        self.hs["BlurKern text"] = widgets.BoundedIntText(
            value=20, description="Blur Kernel", min=2, max=200, disabled=True
        )
        self.hs["BlurKern text"].layout = layout
        layout = {"left": "5%", "width": "10%"}
        self.hs["UseRmZinger chbx"] = widgets.Checkbox(
            value=False, description="Rm Zinger", disabled=True, indent=False
        )
        self.hs["UseRmZinger chbx"].layout = layout
        layout = {"left": "5%", "width": "17%"}
        self.hs["ZingerLevel text"] = widgets.BoundedFloatText(
            value=500.0, description="Zinger Lev", min=10, max=1000, disabled=True
        )
        self.hs["ZingerLevel text"].layout = layout
        layout = {"left": "10%", "width": "10%"}
        self.hs["UseMask chbx"] = widgets.Checkbox(
            value=True, description="Use Mask", disabled=True, indent=False
        )
        self.hs["UseMask chbx"].layout = layout
        layout = {"left": "10%", "width": "17%"}
        self.hs["MaskRat text"] = widgets.BoundedFloatText(
            value=1, description="Mask R", min=0, max=1, step=0.05, disabled=True
        )
        self.hs["MaskRat text"].layout = layout

        self.hs["UseRmZinger chbx"].observe(self.UseRmZinger_chbx_chg, names="value")
        self.hs["ZingerLevel text"].observe(self.ZingerLevel_text_chg, names="value")
        self.hs["UseMask chbx"].observe(self.UseMask_chbx_chg, names="value")
        self.hs["MaskRat text"].observe(self.MaskRat_text_chg, names="value")
        self.hs["UseBlurFlat chbx"].observe(self.BlurFlat_chbx_chg, names="value")
        self.hs["BlurKern text"].observe(self.BlurKern_text_chg, names="value")
        self.hs["MiscOptn box"].children = [
            self.hs["UseBlurFlat chbx"],
            self.hs["BlurKern text"],
            self.hs["UseRmZinger chbx"],
            self.hs["ZingerLevel text"],
            self.hs["UseMask chbx"],
            self.hs["MaskRat text"],
        ]

        layout = {"border": "3px solid #FFCC00", "height": "auto"}
        self.hs["WedgeOptn box"] = widgets.HBox()
        self.hs["WedgeOptn box"].layout = layout
        layout = {"width": "10%"}
        self.hs["IsWedge chbx"] = widgets.Checkbox(
            value=False, description="Is Wedge", disabled=True, indent=False
        )
        self.hs["IsWedge chbx"].layout = layout
        layout = {"width": "20%"}
        self.hs["MissIdxStart text"] = widgets.BoundedIntText(
            value=500,
            min=0,
            max=5000,
            description="Miss S",
            disabled=True,
        )
        self.hs["MissIdxStart text"].layout = layout
        layout = {"width": "20%"}
        self.hs["MissIdxEnd text"] = widgets.BoundedIntText(
            value=600, min=0, max=5000, description="Miss E", disabled=True
        )
        self.hs["MissIdxEnd text"].layout = layout
        layout = {"width": "20%"}
        self.hs["AutoDet chbx"] = widgets.Checkbox(
            value=True, description="Auto Det", disabled=True, indent=True
        )
        self.hs["AutoDet chbx"].layout = layout
        layout = {"left": "5%", "width": "20%"}
        self.hs["AutoThres text"] = widgets.BoundedFloatText(
            value=0.2, min=0, max=1, description="Auto Thres", disabled=True
        )
        self.hs["AutoThres text"].layout = layout

        self.hs["IsWedge chbx"].observe(self.IsWedge_chbx_chg, names="value")
        self.hs["MissIdxStart text"].observe(self.MissIdxStart_text_chg, names="value")
        self.hs["MissIdxEnd text"].observe(self.MissIdxEnd_text_chg, names="value")
        self.hs["AutoDet chbx"].observe(self.AutoDet_chbx_chg, names="value")
        self.hs["AutoThres text"].observe(self.AutoThres_text_chg, names="value")
        self.hs["WedgeOptn box"].children = [
            self.hs["IsWedge chbx"],
            self.hs["MissIdxStart text"],
            self.hs["MissIdxEnd text"],
            self.hs["AutoDet chbx"],
            self.hs["AutoThres text"],
        ]

        layout = {"border": "3px solid #FFCC00", "height": "auto"}
        self.hs["WedgeRef box"] = widgets.HBox()
        self.hs["WedgeRef box"].layout = layout

        layout = {"width": "15%"}
        self.hs["AutoRefFn btn"] = SelectFilesButton(
            option="askopenfilename", **{"open_filetypes": (("h5 files", "*.h5"),)}
        )
        self.hs["AutoRefFn btn"].layout = layout
        self.hs["AutoRefFn btn"].disabled = True

        layout = {"width": "40%"}
        self.hs["AutoRefSli sldr"] = widgets.IntSlider(
            description="slice #",
            min=0,
            max=10,
            value=0,
            disabled=True,
            continuous_update=False,
        )
        self.hs["AutoRefSli sldr"].layout = layout

        self.hs["AutoRefColStart text"] = widgets.BoundedIntText(
            value=0, min=0, max=400, description="W Col_Start", disabled=True
        )
        layout = {"left": "2.5%", "width": "19%"}
        self.hs["AutoRefColStart text"].layout = layout
        self.hs["AutoRefColEnd text"] = widgets.BoundedIntText(
            value=10, min=1, max=401, description="W Col_End", disabled=True
        )
        layout = {"left": "2.5%", "width": "19%"}
        self.hs["AutoRefColEnd text"].layout = layout

        self.hs["AutoRefFn btn"].on_click(self.AutoRefFn_btn_clk)
        self.hs["AutoRefSli sldr"].observe(self.AutoRefSli_sldr_chg, names="value")
        self.hs["AutoRefColStart text"].observe(
            self.AutoRefColStart_text_chg, names="value"
        )
        self.hs["AutoRefColEnd text"].observe(
            self.AutoRefColEnd_text_chg, names="value"
        )
        self.hs["WedgeRef box"].children = [
            self.hs["AutoRefFn btn"],
            self.hs["AutoRefSli sldr"],
            self.hs["AutoRefColStart text"],
            self.hs["AutoRefColEnd text"],
        ]
        ## ## ## ## ## ## ## config sub-boxes in data_config_box -- end

        self.hs["DataConfig box"].children = [
            self.hs["ReconConfig box"],
            self.hs["RoiConfig box"],
            self.hs["AltFlatDarkOptn box"],
            self.hs["FakeFlatDarkOptn box"],
            self.hs["MiscOptn box"],
            self.hs["WedgeOptn box"],
            self.hs["WedgeRef box"],
        ]
        ## ## ## ## ## ## config data parameters in data_config_box in data_config_tab -- end

        self.hs["DataConfig tab"].children = [self.hs["DataConfig box"]]
        ## ## ## ## ## config data_config_tab -- end

        ## ## ## ## ## ## config alg_config_box in alg_config tab -- start
        layout = {
            "border": "3px solid #8855AA",
            "width": "auto",
            "height": f"{0.21 * (self.form_sz[0] - 136)}px",
        }
        self.hs["AlgConfig box"] = widgets.VBox()
        self.hs["AlgConfig box"].layout = layout

        layout = {"border": "3px solid #FFCC00", "height": "auto"}
        self.hs["AlgOptn box0"] = widgets.HBox()
        self.hs["AlgOptn box0"].layout = layout
        layout = {"width": "24%"}
        self.hs["AlgOptn drpdn"] = widgets.Dropdown(
            value="gridrec",
            options=["gridrec", "sirt", "tv", "mlem", "astra"],
            description="algs",
            disabled=True,
        )
        self.hs["AlgOptn drpdn"].layout = layout
        layout = {"width": "23.5%"}
        self.hs["AlgPar00 drpdn"] = widgets.Dropdown(
            value="", options=[""], description="p00", disabled=True
        )
        self.hs["AlgPar00 drpdn"].layout = layout
        layout = {"width": "23.5%"}
        self.hs["AlgPar01 drpdn"] = widgets.Dropdown(
            value="", options=[""], description="p01", disabled=True
        )
        self.hs["AlgPar01 drpdn"].layout = layout
        layout = {"width": "23.5%"}
        self.hs["AlgPar02 drpdn"] = widgets.Dropdown(
            value="", options=[""], description="p02", disabled=True
        )
        self.hs["AlgPar02 drpdn"].layout = layout

        self.hs["AlgOptn drpdn"].observe(self.AlgOptn_drpdn_chg, names="value")
        self.hs["AlgPar00 drpdn"].observe(self.AlgPar00_drpdn_chg, names="value")
        self.hs["AlgPar01 drpdn"].observe(self.AlgPar01_drpdn_chg, names="value")
        self.hs["AlgPar02 drpdn"].observe(self.AlgPar02_drpdn_chg, names="value")
        self.hs["AlgOptn box0"].children = [
            self.hs["AlgOptn drpdn"],
            self.hs["AlgPar00 drpdn"],
            self.hs["AlgPar01 drpdn"],
            self.hs["AlgPar02 drpdn"],
        ]

        layout = {"border": "3px solid #FFCC00", "height": "auto"}
        self.hs["AlgOptn box1"] = widgets.HBox()
        self.hs["AlgOptn box1"].layout = layout
        layout = {"width": "23.5%"}
        self.hs["AlgPar03 text"] = widgets.FloatText(
            value=0, description="p03", disabled=True
        )
        self.hs["AlgPar03 text"].layout = layout
        layout = {"width": "23.5%"}
        self.hs["AlgPar04 text"] = widgets.FloatText(
            value=0, description="p04", disabled=True
        )
        self.hs["AlgPar04 text"].layout = layout
        layout = {"width": "23.5%"}
        self.hs["AlgPar05 text"] = widgets.FloatText(
            value=0, description="p05", disabled=True
        )
        self.hs["AlgPar05 text"].layout = layout
        layout = {"width": "23.5%"}
        self.hs["AlgPar06 text"] = widgets.FloatText(
            value=0.0, description="p06", disabled=True
        )
        self.hs["AlgPar06 text"].layout = layout

        self.hs["AlgPar03 text"].observe(self.AlgPar03_text_chg, names="value")
        self.hs["AlgPar04 text"].observe(self.AlgPar04_text_chg, names="value")
        self.hs["AlgPar05 text"].observe(self.AlgPar05_text_chg, names="value")
        self.hs["AlgPar06 text"].observe(self.AlgPar06_text_chg, names="value")
        self.hs["AlgOptn box1"].children = [
            self.hs["AlgPar03 text"],
            self.hs["AlgPar04 text"],
            self.hs["AlgPar05 text"],
            self.hs["AlgPar06 text"],
        ]

        self.hs["AlgConfig box"].children = [
            self.hs["AlgOptn box0"],
            self.hs["AlgOptn box1"],
        ]
        ## ## ## ## ## ## config alg_config_box in alg_config tab -- end

        self.hs["AlgConfig tab"].children = [self.hs["AlgConfig box"]]
        ## ## ## ## ## define alg_config tab -- end

        ## ## ## ## ## define data info tab -- start
        ## ## ## ## ## ## define data info box -- start
        layout = {
            "border": "3px solid #FFCC00",
            "height": f"{0.42 * (self.form_sz[0] - 136)}px",
        }
        self.hs["DataInfo box"] = widgets.HBox()
        self.hs["DataInfo box"].layout = layout
        layout = {"width": "90%", "height": "90%"}
        self.hs["DataInfo text"] = widgets.Textarea(
            value="Data Info",
            placeholder="Data Info",
            description="Data Info",
            disabled=True,
        )
        self.hs["DataInfo text"].layout = layout
        self.hs["DataInfo box"].children = [self.hs["DataInfo text"]]
        ## ## ## ## ## ## define data info box -- end
        self.hs["DataInfo tab"].children = [self.hs["DataInfo box"]]
        ## ## ## ## ## define data info tab -- end

        ## ## ## ## ## define data_preview tab -- start
        ## ## ## ## ## ## define data_preview_box -- start
        layout = {"border": "3px solid #8855AA", "width": "auto", "height": "auto"}
        self.hs["DataPrev box"] = widgets.VBox()
        self.hs["DataPrev box"].layout = layout

        ## ## ## ## ## ## ## define functional box widgets in data_prevew_box3 -- start
        layout = {"border": "3px solid #FFCC00", "width": "auto", "height": "auto"}
        self.hs["ProjPrev box"] = widgets.HBox()
        self.hs["ProjPrev box"].layout = layout
        layout = {"width": "50%", "height": "auto"}
        self.hs["RawProj sldr"] = widgets.IntSlider(
            value=0,
            description="proj",
            description_tooltip="offset the image at 180 deg to overlap with the image at 0 deg. rotation center can be determined from the offset.",
            min=-100,
            max=100,
            disabled=True,
            indent=False,
            continuous_update=False,
        )
        self.hs["RawProj sldr"].layout = layout
        layout = {"width": "20%", "height": "auto"}
        self.hs["RawProjInMem chbx"] = widgets.Checkbox(
            description="read in mem",
            description_tooltip="Optional read entire raw proj dataset into memory for display",
            value=False,
            disabled=True,
            indent=False,
        )
        self.hs["RawProjInMem chbx"].layout = layout
        layout = {"width": "20%", "height": "auto"}
        self.hs["RawProjViewerClose btn"] = widgets.Button(
            description="Close/Confirm",
            description_tooltip="Optional confrimation of roi (slices and columns) definition",
            disabled=True,
        )
        self.hs["RawProjViewerClose btn"].layout = layout
        self.hs["RawProjViewerClose btn"].style.button_color = "darkviolet"

        self.hs["RawProj sldr"].observe(self.RawProj_sldr_chg, names="value")
        self.hs["RawProjInMem chbx"].observe(self.RawProjInMem_chbx_chg, names="value")
        self.hs["RawProjViewerClose btn"].on_click(self.RawProjViewerClose_btn_clk)
        self.hs["ProjPrev box"].children = [
            self.hs["RawProj sldr"],
            self.hs["RawProjInMem chbx"],
            self.hs["RawProjViewerClose btn"],
        ]
        ## ## ## ## ## ## ## define functional box widgets in data_prevew_box3 -- end

        ## ## ## ## ## ## ## define functional box widgets in data_prevew_box0 -- start
        layout = {"border": "3px solid #FFCC00", "width": "auto", "height": "auto"}
        self.hs["CenPrev box"] = widgets.HBox()
        self.hs["CenPrev box"].layout = layout
        layout = {"width": "50%", "height": "auto"}
        self.hs["CenOffsetRange sldr"] = widgets.IntSlider(
            value=0,
            description="offset",
            description_tooltip="offset the image at 180 deg to overlap with the image at 0 deg. rotation center can be determined from the offset.",
            min=-100,
            max=100,
            disabled=True,
            indent=False,
            continuous_update=False,
        )
        self.hs["CenOffsetRange sldr"].layout = layout
        layout = {"width": "20%", "height": "auto"}
        self.hs["CenOffsetCfm btn"] = widgets.Button(
            description="Confirm",
            description_tooltip="Optional confrimation of the rough center",
            disabled=True,
        )
        self.hs["CenOffsetCfm btn"].layout = layout
        self.hs["CenOffsetCfm btn"].style.button_color = "darkviolet"
        layout = {"width": "20%", "height": "auto"}
        self.hs["CenViewerClose btn"] = widgets.Button(
            description="Close",
            description_tooltip="Optional close the viewer window",
            disabled=True,
        )
        self.hs["CenViewerClose btn"].layout = layout
        self.hs["CenViewerClose btn"].style.button_color = "darkviolet"

        self.hs["CenOffsetRange sldr"].observe(
            self.CenOffsetRange_sldr_chg, names="value"
        )
        self.hs["CenOffsetCfm btn"].on_click(self.CenOffsetCfm_btn_clk)
        self.hs["CenViewerClose btn"].on_click(self.CenViewerClose_btn_clk)
        self.hs["CenPrev box"].children = [
            self.hs["CenOffsetRange sldr"],
            self.hs["CenOffsetCfm btn"],
            self.hs["CenViewerClose btn"],
        ]
        ## ## ## ## ## ## ## define functional box widgets in data_prevew_box0 -- end

        ## ## ## ## ## ## ## define functional box widgets in data_prevew_box1 -- start
        layout = {"border": "3px solid #FFCC00", "width": "auto", "height": "auto"}
        self.hs["TrialCenPrev box"] = widgets.HBox()
        self.hs["TrialCenPrev box"].layout = layout
        layout = {"width": "50%", "height": "auto"}
        self.hs["TrialCenPrev sldr"] = widgets.IntSlider(
            value=0,
            description="trial cen",
            description_tooltip="offset the image at 180 deg to overlap with the image at 0 deg. rotation center can be determined from the offset.",
            min=-100,
            max=100,
            disabled=True,
            indent=False,
            continuous_update=False,
        )
        self.hs["TrialCenPrev sldr"].layout = layout
        layout = {"width": "20%", "height": "auto"}
        self.hs["TrialCenCfm btn"] = widgets.Button(
            description="Confirm",
            description_tooltip="Optional confrimation of the rough center",
            disabled=True,
        )
        self.hs["TrialCenCfm btn"].layout = layout
        self.hs["TrialCenCfm btn"].style.button_color = "darkviolet"
        layout = {"width": "20%", "height": "auto"}
        self.hs["TrialCenViewerClose btn"] = widgets.Button(
            description="Close",
            description_tooltip="Optional close the viewer window",
            disabled=True,
        )
        self.hs["TrialCenViewerClose btn"].layout = layout
        self.hs["TrialCenViewerClose btn"].style.button_color = "darkviolet"

        self.hs["TrialCenPrev sldr"].observe(self.TrialCenPrev_sldr_chg, names="value")
        self.hs["TrialCenCfm btn"].on_click(self.TrialCenCfm_btn_clk)
        self.hs["TrialCenViewerClose btn"].on_click(self.TrialCenViewerClose_btn_clk)
        self.hs["TrialCenPrev box"].children = [
            self.hs["TrialCenPrev sldr"],
            self.hs["TrialCenCfm btn"],
            self.hs["TrialCenViewerClose btn"],
        ]
        ## ## ## ## ## ## ## define functional box widgets in data_prevew_box1 -- end

        # ## ## ## ## ## ## ## define functional box widgets in data_prevew_box2 -- start
        self.hs["DataPrev box"].children = [
            self.hs["ProjPrev box"],
            self.hs["CenPrev box"],
            self.hs["TrialCenPrev box"],
        ]
        ## ## ## ## ## ## define data_preview_box-- end

        self.hs["DataPrev tab"].children = [self.hs["DataPrev box"]]
        ## ## ## ## ## define data_preview tab -- end

        ## ## ## ## ## define VolView tab -- start
        ## ## ## ## ## ## define VolView_box -- start
        layout = {"border": "3px solid #8855AA", "width": "auto", "height": "auto"}
        self.hs["VolRecon box"] = widgets.VBox()
        self.hs["VolRecon box"].layout = layout

        ## ## ## ## ## ## ## define vol viewer box -- start
        layout = {"border": "3px solid #FFCC00", "width": "auto", "height": "auto"}
        self.hs["VolViewOpt box"] = widgets.HBox()
        self.hs["VolViewOpt box"].layout = layout
        layout = {"width": "60%", "height": "auto"}
        self.hs["VolViewOpt tgbtn"] = widgets.ToggleButtons(
            description="viewer options",
            disabled=True,
            description_tooltip="napari: provides 3D visualization; fiji: provides better slice visualization",
            options=["fiji", "napari"],
            value="fiji",
        )
        self.hs["VolViewOpt tgbtn"].layout = layout

        self.hs["VolViewOpt tgbtn"].observe(self.VolViewOpt_tgbtn_chg, names="value")
        self.hs["VolViewOpt box"].children = [
            self.hs["VolViewOpt tgbtn"],
        ]
        ## ## ## ## ## ## ## define vol viewer box -- end

        self.hs["VolRecon box"].children = [self.hs["VolViewOpt box"]]
        ## ## ## ## ## ## define VolView_box -- end
        self.hs["VolRecon tab"].children = [self.hs["VolRecon box"]]
        ## ## ## ## ## define VolView tab -- end

        self.hs["Config&Input form"].children = [
            self.hs["SelFile&Path box"],
            self.hs["Data tab"],
        ]
        ## ## ## config config_input_form -- end

        ## ## ## ## config filter&recon tab -- start
        layout = {"border": "3px solid #8855AA", "width": "auto", "height": "auto"}
        self.hs["Filter&Recon form"] = widgets.VBox()
        self.hs["Filter&Recon form"].layout = layout

        ## ## ## ## ## config filter_config_box -- start
        layout = {"border": "3px solid #8855AA", "height": "auto"}
        self.hs["Filter&Recon box"] = widgets.VBox()
        self.hs["Filter&Recon box"].layout = layout

        ## ## ## ## ## ## label recon_box -- start
        grid_recon_chunk = GridspecLayout(
            2,
            100,
            layout={
                "border": "3px solid #FFCC00",
                "height": "auto",
                "width": "auto",
                "grid_row_gap": "4px",
                "grid_column_gap": "8px",
                "align_items": "flex-start",
                "justify_items": "flex-start",
            },
        )
        self.hs["ReconChunk box"] = grid_recon_chunk

        grid_recon_chunk[0, 30:70] = widgets.HTML(
            "<span style='color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);'>"
            + "Chunk&Margin"
            + "</span>"
        )
        self.hs["ChunkMargTitle label"] = grid_recon_chunk[0, 30:70]
        layout = {
            "width": "auto",
            "height": "auto",
            "background-color": "white",
            "color": "cyan",
            "justify_items": "center",
        }
        self.hs["ChunkMargTitle label"].layout = layout

        grid_recon_chunk[1, :10] = widgets.BoundedIntText(
            description="Chunk Sz",
            disabled=True,
            min=1,
            max=2000,
            layout={"width": "auto", "height": "auto"},
            value=200,
            description_tooltip="how many slices will be loaded into memory for reconstruction each time",
        )
        self.hs["ReconChunkSz text"] = grid_recon_chunk[1, :10]

        grid_recon_chunk[1, 10:20] = widgets.BoundedIntText(
            description="Margin Sz",
            disabled=True,
            min=0,
            max=50,
            layout={"width": "auto", "height": "auto"},
            value=15,
            description_tooltip="how many slices will be loaded into memory for reconstruction each time",
        )
        self.hs["ReconMargSz text"] = grid_recon_chunk[1, 10:20]

        self.hs["ReconChunkSz text"].observe(self.ReconChunkSz_text_chg, names="value")
        self.hs["ReconMargSz text"].observe(self.ReconMargSz_text_chg, names="value")
        self.hs["ReconChunk box"].children = [
            self.hs["ChunkMargTitle label"],
            self.hs["ReconChunkSz text"],
            self.hs["ReconMargSz text"],
        ]
        ## ## ## ## ## ## label recon_box -- end

        ## ## ## ## ## ## label filter_config_box -- start
        layout = {
            "justify-content": "center",
            "align-items": "center",
            "align-contents": "center",
            "border": "3px solid #FFCC00",
            "height": "auto",
        }
        self.hs["FilterConfigTitle box"] = widgets.HBox()
        self.hs["FilterConfigTitle box"].layout = layout
        self.hs["FilterConfigTitle label"] = widgets.HTML(
            "<span style='color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);'>"
            + "Filter Config"
            + "</span>"
        )
        layout = {"left": "43%", "background-color": "white", "color": "cyan"}
        self.hs["FilterConfigTitle label"].layout = layout
        self.hs["FilterConfigTitle box"].children = [self.hs["FilterConfigTitle label"]]
        ## ## ## ## ## ## label filter_config_box -- end

        ## ## ## ## ## ## config filters with GridspecLayout-- start
        FilterConfigGrid = GridspecLayout(
            8,
            200,
            layout={
                "border": "3px solid #FFCC00",
                "width": "auto",
                "height": "auto",
                "align_items": "flex-start",
                "justify_items": "flex-start",
            },
        )

        ## ## ## ## ## ## ## config filters: left-hand side box in GridspecLayout -- start
        FilterConfigGrid[:7, :100] = GridspecLayout(
            10,
            20,
            grid_gap="8px",
            layout={
                "border": "3px solid #FFCC00",
                "width": "100%",
                "height": "100%",
                "grid_row_gap": "8px",
                "align_items": "flex-start",
                "justify_items": "flex-start",
                "grid_column_gap": "8px",
            },
        )
        self.hs["FilterConfigLeft box"] = FilterConfigGrid[:7, :100]

        FilterConfigGrid[:7, :100][0, :16] = widgets.Dropdown(
            value="phase retrieval",
            layout={"width": "auto"},
            options=FILTERLIST,
            description="Filter List",
            indent=False,
            disabled=True,
        )
        self.hs["FilterConfigLeftFltList drpdn"] = FilterConfigGrid[:7, :100][0, :16]

        FilterConfigGrid[:7, :100][0, 16:19] = widgets.Button(
            description="==>", disabled=True, layout={"width": "auto"}
        )
        self.hs["FilterConfigLeftAddTo btn"] = FilterConfigGrid[:7, :100][0, 16:19]
        FilterConfigGrid[:7, :100][0, 16:19].style.button_color = "#0000FF"
        for ii in range(3):
            for jj in range(2):
                FilterConfigGrid[:7, :100][
                    1 + ii, jj * 8 : (jj + 1) * 8
                ] = widgets.Dropdown(
                    value="",
                    options=[""],
                    description="p" + str(ii * 2 + jj).zfill(2),
                    disabled=True,
                    layout={"width": "90%"},
                )
        for ii in range(3):
            for jj in range(2):
                FilterConfigGrid[:7, :100][
                    4 + ii, jj * 8 : (jj + 1) * 8
                ] = widgets.FloatText(
                    value=0,
                    disabled=True,
                    description="p" + str((ii + 3) * 2 + jj).zfill(2),
                    layout={"width": "90%"},
                )

        self.hs["FilterConfigLeftPar00 text"] = FilterConfigGrid[:7, :100][1, 0:8]
        self.hs["FilterConfigLeftPar01 text"] = FilterConfigGrid[:7, :100][1, 8:16]
        self.hs["FilterConfigLeftPar02 text"] = FilterConfigGrid[:7, :100][2, 0:8]
        self.hs["FilterConfigLeftPar03 text"] = FilterConfigGrid[:7, :100][2, 8:16]
        self.hs["FilterConfigLeftPar04 text"] = FilterConfigGrid[:7, :100][3, 0:8]
        self.hs["FilterConfigLeftPar05 text"] = FilterConfigGrid[:7, :100][3, 8:16]
        self.hs["FilterConfigLeftPar06 text"] = FilterConfigGrid[:7, :100][4, 0:8]
        self.hs["FilterConfigLeftPar07 text"] = FilterConfigGrid[:7, :100][4, 8:16]
        self.hs["FilterConfigLeftPar08 text"] = FilterConfigGrid[:7, :100][5, 0:8]
        self.hs["FilterConfigLeftPar09 text"] = FilterConfigGrid[:7, :100][5, 8:16]
        self.hs["FilterConfigLeftPar10 text"] = FilterConfigGrid[:7, :100][6, 0:8]
        self.hs["FilterConfigLeftPar11 text"] = FilterConfigGrid[:7, :100][6, 8:16]

        FilterConfigGrid[:7, :100][7:, :] = widgets.HTML(
            value="<style>p{word-wrap: break-word}</style> <p>"
            + "Hover mouse over params for the description of the param for each filter."
            + " </p>"
        )
        self.hs["FilterConfigLeftFltList drpdn"].observe(
            self.FilterConfigLeftFltList_drpdn_chg, names="value"
        )
        self.hs["FilterConfigLeftAddTo btn"].on_click(
            self.FilterConfigLeftAddTo_btn_clk
        )
        self.hs["FilterConfigLeftPar00 text"].observe(
            self.FilterConfigLeftPar00_text_chg, names="value"
        )
        self.hs["FilterConfigLeftPar01 text"].observe(
            self.FilterConfigLeftPar01_text_chg, names="value"
        )
        self.hs["FilterConfigLeftPar02 text"].observe(
            self.FilterConfigLeftPar02_text_chg, names="value"
        )
        self.hs["FilterConfigLeftPar03 text"].observe(
            self.FilterConfigLeftPar03_text_chg, names="value"
        )
        self.hs["FilterConfigLeftPar04 text"].observe(
            self.FilterConfigLeftPar04_text_chg, names="value"
        )
        self.hs["FilterConfigLeftPar05 text"].observe(
            self.FilterConfigLeftPar05_text_chg, names="value"
        )
        self.hs["FilterConfigLeftPar06 text"].observe(
            self.FilterConfigLeftPar06_text_chg, names="value"
        )
        self.hs["FilterConfigLeftPar07 text"].observe(
            self.FilterConfigLeftPar07_text_chg, names="value"
        )
        self.hs["FilterConfigLeftPar08 text"].observe(
            self.FilterConfigLeftPar08_text_chg, names="value"
        )
        self.hs["FilterConfigLeftPar09 text"].observe(
            self.FilterConfigLeftPar09_text_chg, names="value"
        )
        self.hs["FilterConfigLeftPar10 text"].observe(
            self.FilterConfigLeftPar10_text_chg, names="value"
        )
        self.hs["FilterConfigLeftPar11 text"].observe(
            self.FilterConfigLeftPar11_text_chg, names="value"
        )
        self.hs["FilterConfigLeft box"].children = [
            self.hs["FilterConfigLeftFltList drpdn"],
            self.hs["FilterConfigLeftAddTo btn"],
            self.hs["FilterConfigLeftPar00 text"],
            self.hs["FilterConfigLeftPar01 text"],
            self.hs["FilterConfigLeftPar02 text"],
            self.hs["FilterConfigLeftPar03 text"],
            self.hs["FilterConfigLeftPar04 text"],
            self.hs["FilterConfigLeftPar05 text"],
            self.hs["FilterConfigLeftPar06 text"],
            self.hs["FilterConfigLeftPar07 text"],
            self.hs["FilterConfigLeftPar08 text"],
            self.hs["FilterConfigLeftPar09 text"],
            self.hs["FilterConfigLeftPar10 text"],
            self.hs["FilterConfigLeftPar11 text"],
        ]
        ## ## ## ## ## ## ## config filters: left-hand side box in GridspecLayout -- end

        ## ## ## ## ## ## ## config filters: right-hand side box in GridspecLayout -- start
        FilterConfigGrid[:7, 100:] = GridspecLayout(
            10,
            10,
            grid_gap="8px",
            layout={"border": "3px solid #FFCC00", "width": "100%", "height": "100%"},
        )
        self.hs["FilterConfigRight box"] = FilterConfigGrid[:7, 100:]

        FilterConfigGrid[:7, 100:][:7, :8] = widgets.SelectMultiple(
            value=["None"],
            options=["None"],
            description="Filter Seq",
            disabled=True,
            layout={"height": "100%"},
        )
        self.hs["FilterConfigRightFlt mulsel"] = FilterConfigGrid[:7, 100:][:7, :8]

        FilterConfigGrid[:7, 100:][1, 9] = widgets.Button(
            description="Move Up", disabled=True, layout={"width": "auto"}
        )
        FilterConfigGrid[:7, 100:][1, 9].style.button_color = "#0000FF"
        self.hs["FilterConfigRightMvUp btn"] = FilterConfigGrid[:7, 100:][1, 9]

        FilterConfigGrid[:7, 100:][2, 9] = widgets.Button(
            description="Move Dn", disabled=True, layout={"width": "auto"}
        )
        FilterConfigGrid[:7, 100:][2, 9].style.button_color = "#0000FF"
        self.hs["FilterConfigRightMvDn btn"] = FilterConfigGrid[:7, 100:][2, 9]

        FilterConfigGrid[:7, 100:][3, 9] = widgets.Button(
            description="Remove",
            disabled=True,
            layout={"width": f"{int(2 * (self.form_sz[1] - 98) / 20)}px"},
        )
        FilterConfigGrid[:7, 100:][3, 9].style.button_color = "#0000FF"
        self.hs["FilterConfigRightRm btn"] = FilterConfigGrid[:7, 100:][3, 9]

        FilterConfigGrid[:7, 100:][4, 9] = widgets.Button(
            description="Finish", disabled=True, layout={"width": "auto"}
        )
        FilterConfigGrid[:7, 100:][4, 9].style.button_color = "#0000FF"
        self.hs["FilterConfigRightFnsh btn"] = FilterConfigGrid[:7, 100:][4, 9]

        self.hs["FilterConfigRightFlt mulsel"].observe(
            self.FilterConfigRightFlt_mulsel_chg, names="value"
        )
        self.hs["FilterConfigRightMvUp btn"].on_click(
            self.FilterConfigRightMvUp_btn_clk
        )
        self.hs["FilterConfigRightMvDn btn"].on_click(
            self.FilterConfigRightMvDn_btn_clk
        )
        self.hs["FilterConfigRightRm btn"].on_click(self.FilterConfigRightRm_btn_clk)
        self.hs["FilterConfigRightFnsh btn"].on_click(
            self.FilterConfigRightFnsh_btn_clk
        )
        self.hs["FilterConfigRight box"].children = [
            self.hs["FilterConfigRightFlt mulsel"],
            self.hs["FilterConfigRightMvUp btn"],
            self.hs["FilterConfigRightMvDn btn"],
            self.hs["FilterConfigRightRm btn"],
            self.hs["FilterConfigRightFnsh btn"],
        ]
        ## ## ## ## ## ## ## config filters: right-hand side box in GridspecLayout -- end

        ## ## ## ## ## ## ## config confirm box in GridspecLayout -- start
        FilterConfigGrid[7, :141] = widgets.Text(
            value="Confirm to proceed after you finish data and algorithm configuration...",
            layout={"top": "20%", "width": "100%", "height": "auto"},
            disabled=True,
        )
        self.hs["FilterConfigCfm text"] = FilterConfigGrid[7, :141]

        FilterConfigGrid[7, 142:172] = widgets.Button(
            description="Confirm",
            disabled=True,
            layout={"top": "20%", "width": "100%", "height": "auto"},
        )
        FilterConfigGrid[7, 142:171].style.button_color = "darkviolet"
        self.hs["FilterConfigCfm btn"] = FilterConfigGrid[7, 142:172]
        self.hs["FilterConfigCfm btn"].on_click(self.FilterConfigCfm_btn_clk)
        ## ## ## ## ## ## ## config confirm box in GridspecLayout -- end

        self.hs["FilterConfig box"] = FilterConfigGrid
        self.hs["FilterConfig box"].children = [
            self.hs["FilterConfigLeft box"],
            self.hs["FilterConfigRight box"],
            self.hs["FilterConfigCfm text"],
            self.hs["FilterConfigCfm btn"],
        ]
        ## ## ## ## ## ## config filters with GridspecLayout-- end

        self.hs["Filter&Recon box"].children = [
            self.hs["ReconChunk box"],
            self.hs["FilterConfig box"],
        ]
        ## ## ## ## ## config  filter_config_box -- end

        ## ## ## ## ## config recon_box -- start
        layout = {"border": "3px solid #8855AA", "height": "auto"}
        self.hs["Recon box"] = widgets.VBox()
        self.hs["Recon box"].layout = layout

        ## ## ## ## ## ## ## config widgets in recon_box -- start
        layout = {
            "justify-content": "center",
            "align-items": "center",
            "align-contents": "center",
            "border": "3px solid #FFCC00",
            "height": "auto",
        }
        self.hs["ReconDo box"] = widgets.HBox()
        self.hs["ReconDo box"].layout = layout
        layout = {"width": "70%", "height": "auto"}
        self.hs["ReconPrgr bar"] = widgets.IntProgress(
            value=0,
            min=0,
            max=10,
            step=1,
            description="Completing:",
            bar_style="info",  # "success", "info", "warning", "danger" or ""
            orientation="horizontal",
        )
        self.hs["ReconPrgr bar"].layout = layout
        layout = {"width": "15%", "height": "auto"}
        self.hs["Recon btn"] = widgets.Button(description="Recon", disabled=True)
        self.hs["Recon btn"].style.button_color = "darkviolet"
        self.hs["Recon btn"].layout = layout

        self.hs["Recon btn"].on_click(self.Recon_btn_clk)
        self.hs["ReconDo box"].children = [
            self.hs["ReconPrgr bar"],
            self.hs["Recon btn"],
        ]
        ## ## ## ## ## ## ## config widgets in recon_box -- end

        self.hs["Recon box"].children = [
            self.hs["Filter&Recon box"],
            self.hs["ReconDo box"],
        ]
        ## ## ## ## ## config recon box -- end

        self.hs["Filter&Recon form"].children = [
            self.hs["Filter&Recon box"],
            self.hs["ReconDo box"],
        ]
        ## ## ## ## config filter&recon tab -- end

        ## ## ## ## config filter&recon tab -- start
        layout = {"border": "3px solid #8855AA", "width": "auto", "height": "auto"}
        self.hs["ReconConfigSumm form"] = widgets.VBox()
        self.hs["ReconConfigSumm form"].layout = layout

        ## ## ## ## ## config filter&recon text -- start
        layout = {"width": "95%", "height": "90%"}
        self.hs["ReconConfigSumm text"] = widgets.Textarea(
            value="Recon Config Info",
            placeholder="Recon Config Info",
            description="Recon Config Info",
            disabled=True,
        )
        self.hs["ReconConfigSumm text"].layout = layout
        ## ## ## ## ## config filter&recon text -- start
        self.hs["ReconConfigSumm form"].children = [self.hs["ReconConfigSumm text"]]
        ## ## ## ## config filter&recon tab -- end

        self.hs["Filter&Recon tab"].children = [
            self.hs["Filter&Recon form"],
            self.hs["ReconConfigSumm form"],
        ]
        self.hs["Filter&Recon tab"].set_title(0, "Filter Config")
        self.hs["Filter&Recon tab"].set_title(1, "Recon Config Summary")
        ## ## ## define boxes in filter&recon_form -- end
        self.bundle_param_handles()

        self.hs["SelRawH5TopDir btn"].initialdir = self.global_h.cwd
        self.hs["SelSavReconDir btn"].initialdir = self.global_h.cwd
        self.hs["SelSavDebugDir btn"].initialdir = self.global_h.cwd
        self.hs["AltFlatFile btn"].initialdir = self.global_h.cwd
        self.hs["AltDarkFile btn"].initialdir = self.global_h.cwd

    def bundle_param_handles(self):
        self.flt_phs = [
            self.hs["FilterConfigLeftPar00 text"],
            self.hs["FilterConfigLeftPar01 text"],
            self.hs["FilterConfigLeftPar02 text"],
            self.hs["FilterConfigLeftPar03 text"],
            self.hs["FilterConfigLeftPar04 text"],
            self.hs["FilterConfigLeftPar05 text"],
            self.hs["FilterConfigLeftPar06 text"],
            self.hs["FilterConfigLeftPar07 text"],
            self.hs["FilterConfigLeftPar08 text"],
            self.hs["FilterConfigLeftPar09 text"],
            self.hs["FilterConfigLeftPar10 text"],
            self.hs["FilterConfigLeftPar11 text"],
        ]
        self.alg_phs = [
            self.hs["AlgPar00 drpdn"],
            self.hs["AlgPar01 drpdn"],
            self.hs["AlgPar02 drpdn"],
            self.hs["AlgPar03 text"],
            self.hs["AlgPar04 text"],
            self.hs["AlgPar05 text"],
            self.hs["AlgPar06 text"],
        ]

    def reset_config(self):
        self.hs["AlgOptn drpdn"].value = "gridrec"
        self.tomo_selected_alg = "gridrec"
        self.set_alg_param_widgets()
        self.hs["FilterConfigLeftFltList drpdn"].value = "phase retrieval"
        self.tomo_left_box_selected_flt = "phase retrieval"
        self.hs["FilterConfigRightFlt mulsel"].options = ["None"]
        self.hs["FilterConfigRightFlt mulsel"].value = ["None"]
        self.tomo_right_filter_dict = {0: {}}
        self.set_flt_param_widgets()

    def lock_message_text_boxes(self):
        boxes = ["DataInfo text"]
        enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)

    def boxes_logic(self):
        if self.tomo_recon_type == "Trial Cent":
            self.hs["SavDebug chbx"].disabled = False
        elif self.tomo_recon_type == "Vol Recon":
            self.tomo_use_debug = False
            self.hs["SavDebug chbx"].value = False
            self.hs["SavDebug chbx"].disabled = True

        if not self.tomo_filepath_configured:
            boxes = ["Data tab", "Filter&Recon tab"]
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            self.lock_message_text_boxes()
        elif self.tomo_filepath_configured and (not self.tomo_data_configured):
            boxes = [
                "DataConfig tab",
                "AlgOptn drpdn",
                "DataInfo tab",
                "CenPrev box",
                "ProjPrev box",
                "ReconChunk box",
                "FilterConfigLeftFltList drpdn",
                "FilterConfigLeftAddTo btn",
                "FilterConfigRight box",
                "FilterConfigCfm btn",
            ]
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            boxes = ["ReconDo box", "TrialCenPrev box"]
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            self.lock_message_text_boxes()
        elif (self.tomo_filepath_configured & self.tomo_data_configured) & (
            self.recon_finish == -1
        ):
            boxes = [
                "DataConfig tab",
                "AlgOptn drpdn",
                "DataInfo tab",
                "CenPrev box",
                "ProjPrev box",
                "ReconChunk box",
                "FilterConfigLeftFltList drpdn",
                "FilterConfigLeftAddTo btn",
                "FilterConfigRight box",
                "Recon box",
            ]
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            boxes = ["TrialCenPrev box"]
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            self.lock_message_text_boxes()
        elif (
            (self.tomo_filepath_configured & self.tomo_data_configured)
            & (self.recon_finish == 0)
            & (self.tomo_recon_type == "Trial Cent")
        ):
            boxes = [
                "DataConfig tab",
                "AlgOptn drpdn",
                "DataInfo tab",
                "CenPrev box",
                "TrialCenPrev box",
                "ProjPrev box",
                "ReconChunk box",
                "FilterConfigLeftFltList drpdn",
                "FilterConfigLeftAddTo btn",
                "FilterConfigRight box",
                "Recon box",
            ]
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            self.lock_message_text_boxes()

    def cal_set_srch_win(self):
        if not self.data_info:
            self.data_info = get_raw_img_info(
                self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                self.global_h.io_tomo_cfg,
                scan_type="tomo",
            )
            info = ""
            for key, item in self.data_info.items():
                info = info + str(key) + ":" + str(item) + "\n"
            self.hs["DataInfo text"].value = info
            if self.data_info:
                if self.hs["CenWinLeft text"].value >= (
                    self.data_info["img_dim"][2] - self.hs["CenWinWz text"].value - 1
                ):
                    self.hs["CenWinLeft text"].value = 0
                    self.hs["CenWinLeft text"].max = (
                        self.data_info["img_dim"][2]
                        - self.hs["CenWinWz text"].value
                        - 1
                    )
                    self.hs["CenWinLeft text"].value = (
                        int(self.data_info["img_dim"][2] / 2) - 40
                    )
                else:
                    self.hs["CenWinLeft text"].max = (
                        self.data_info["img_dim"][2]
                        - self.hs["CenWinWz text"].value
                        - 1
                    )
        else:
            di = get_raw_img_info(
                self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                self.global_h.io_tomo_cfg,
                scan_type="tomo",
            )
            if di:
                info = ""
                for key, item in di.items():
                    info = info + str(key) + ":" + str(item) + "\n"
                self.hs["DataInfo text"].value = info
                if (not self.data_info) or (
                    list(self.data_info["img_dim"][1:]) != list(di["img_dim"][1:])
                ):
                    self.data_info = di
                    if self.hs["CenWinLeft text"].value >= (
                        self.data_info["img_dim"][2]
                        - self.hs["CenWinWz text"].value
                        - 1
                    ):
                        self.hs["CenWinLeft text"].value = 0
                        self.hs["CenWinLeft text"].max = (
                            self.data_info["img_dim"][2]
                            - self.hs["CenWinWz text"].value
                            - 1
                        )
                    else:
                        self.hs["CenWinLeft text"].max = (
                            self.data_info["img_dim"][2]
                            - self.hs["CenWinWz text"].value
                            - 1
                        )
                    self.hs["CenWinLeft text"].value = (
                        int(self.data_info["img_dim"][2] / 2) - 40
                    )
                else:
                    self.data_info = di
            else:
                self.hs["SelFile&PathCfm text"].value = "Cannot open the file..."

    def tomo_compound_logic(self):
        self.hs["RotCen text"].disabled = True
        if self.tomo_recon_type == "Trial Cent":
            if self.tomo_raw_data_top_dir_set & self.tomo_data_center_path_set:
                self.hs["ScanId drpdn"].disabled = False               
                self.hs["CenWinLeft text"].disabled = False
                self.hs["CenWinWz text"].disabled = False
                self.hs["ReconChunkSz text"].disabled = True
                self.hs["ReconMargSz text"].disabled = True
                self.hs["RoiSliEnd text"].disabled = True
        elif self.tomo_recon_type == "Vol Recon":
            if self.tomo_raw_data_top_dir_set & self.tomo_recon_path_set:
                self.hs["ScanId drpdn"].disabled = True
                self.hs["CenWinLeft text"].disabled = True
                self.hs["CenWinWz text"].disabled = True
                self.hs["ReconChunkSz text"].disabled = False
                self.hs["ReconMargSz text"].disabled = False
                self.hs["RoiSliStart text"].disabled = False
                self.hs["RoiSliEnd text"].disabled = False
                self.hs["UseConfig chbx"].value = True
                self.hs["UseConfig chbx"].disabled = True
                self.hs["RoiColStart text"].disabled = True
                self.hs["RoiColEnd text"].disabled = True
            self.tomo_use_read_config = True

        if self.tomo_filepath_configured:
            if self.tomo_use_alt_flat:
                self.hs["AltFlatFile btn"].disabled = False
            else:
                self.hs["AltFlatFile btn"].disabled = True

            if self.tomo_use_fake_flat:
                self.hs["FakeFlatVal text"].disabled = False
            else:
                self.hs["FakeFlatVal text"].disabled = True

            if self.tomo_use_alt_dark:
                self.hs["AltDarkFile btn"].disabled = False
            else:
                self.hs["AltDarkFile btn"].disabled = True

            if self.tomo_use_fake_dark:
                self.hs["FakeDarkVal text"].disabled = False
            else:
                self.hs["FakeDarkVal text"].disabled = True

            if self.tomo_use_blur_flat:
                self.hs["BlurKern text"].disabled = False
            else:
                self.hs["BlurKern text"].disabled = True

            if self.tomo_use_rm_zinger:
                self.hs["ZingerLevel text"].disabled = False
            else:
                self.hs["ZingerLevel text"].disabled = True

            if self.tomo_use_mask:
                self.hs["MaskRat text"].disabled = False
            else:
                self.hs["MaskRat text"].disabled = True

            if self.tomo_is_wedge:
                self.hs["AutoDet chbx"].disabled = False
                if self.tomo_use_wedge_ang_auto_det:
                    self.hs["MissIdxStart text"].disabled = True
                    self.hs["MissIdxEnd text"].disabled = True
                    self.hs["AutoThres text"].disabled = False
                    self.hs["AutoRefFn btn"].disabled = False
                    if self.tomo_wedge_ang_auto_det_ref_fn is not None:
                        self.hs["AutoRefSli sldr"].disabled = False
                    else:
                        self.hs["AutoRefSli sldr"].disabled = True
                else:
                    self.hs["MissIdxStart text"].disabled = False
                    self.hs["MissIdxEnd text"].disabled = False
                    self.hs["AutoThres text"].disabled = True
                    self.hs["AutoRefFn btn"].disabled = True
                    self.hs["AutoRefSli sldr"].disabled = True
            else:
                self.hs["AutoDet chbx"].value = False
                self.hs["AutoDet chbx"].disabled = True
                self.hs["MissIdxStart text"].disabled = True
                self.hs["MissIdxEnd text"].disabled = True
                self.hs["AutoThres text"].disabled = True
                self.hs["AutoRefFn btn"].disabled = True

    def set_rec_params_from_rec_dict(self, recon_param_dict):
        self.tomo_raw_data_top_dir = recon_param_dict["file_params"]["raw_data_top_dir"]
        self.tomo_data_center_path = recon_param_dict["file_params"]["data_center_dir"]
        self.tomo_recon_top_dir = recon_param_dict["file_params"]["recon_top_dir"]
        self.tomo_debug_top_dir = recon_param_dict["file_params"]["debug_top_dir"]
        self.tomo_alt_flat_file = recon_param_dict["file_params"]["alt_flat_file"]
        self.tomo_alt_dark_file = recon_param_dict["file_params"]["alt_dark_file"]
        self.tomo_wedge_ang_auto_det_ref_fn = recon_param_dict["file_params"][
            "wedge_ang_auto_det_ref_fn"
        ]
        self.global_h.io_tomo_cfg = recon_param_dict["file_params"]["io_confg"]
        self.tomo_use_debug = recon_param_dict["recon_config"]["use_debug"]

        self.tomo_use_alt_flat = recon_param_dict["recon_config"]["use_alt_flat"]
        self.tomo_use_alt_dark = recon_param_dict["recon_config"]["use_alt_dark"]
        self.tomo_use_fake_flat = recon_param_dict["recon_config"]["use_fake_flat"]
        self.tomo_use_fake_dark = recon_param_dict["recon_config"]["use_fake_dark"]
        if "use_flat_blur" in recon_param_dict["recon_config"]:
            self.tomo_use_blur_flat = recon_param_dict["recon_config"]["use_flat_blur"]
        else:
            self.tomo_use_blur_flat = False
        self.tomo_use_rm_zinger = recon_param_dict["recon_config"]["use_rm_zinger"]
        self.tomo_use_mask = recon_param_dict["recon_config"]["use_mask"]
        self.tomo_use_wedge_ang_auto_det = recon_param_dict["recon_config"][
            "use_wedge_ang_auto_det"
        ]
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
        if "blur_kernel" in recon_param_dict["data_params"]:
            self.tomo_flat_blur_kernel = recon_param_dict["data_params"]["blur_kernel"]
        else:
            self.tomo_flat_blur_kernel = 1
        self.tomo_zinger_val = recon_param_dict["data_params"]["zinger_val"]
        self.tomo_mask_ratio = recon_param_dict["data_params"]["mask_ratio"]
        self.tomo_wedge_missing_s = recon_param_dict["data_params"]["wedge_missing_s"]
        self.tomo_wedge_missing_e = recon_param_dict["data_params"]["wedge_missing_e"]
        self.tomo_wedge_auto_ref_col_s = recon_param_dict["data_params"]["wedge_col_s"]
        self.tomo_wedge_auto_ref_col_e = recon_param_dict["data_params"]["wedge_col_e"]
        self.tomo_wedge_ang_auto_det_thres = recon_param_dict["data_params"][
            "wedge_ang_auto_det_thres"
        ]
        self.tomo_selected_alg = recon_param_dict["alg_params"]["algorithm"]
        self.alg_param_dict = recon_param_dict["alg_params"]["params"]

    def set_rec_dict_from_rec_params(self):
        self.tomo_recon_param_dict["file_params"][
            "raw_data_top_dir"
        ] = self.tomo_raw_data_top_dir
        self.tomo_recon_param_dict["file_params"][
            "data_center_dir"
        ] = self.tomo_data_center_path
        self.tomo_recon_param_dict["file_params"][
            "recon_top_dir"
        ] = self.tomo_recon_top_dir
        self.tomo_recon_param_dict["file_params"][
            "debug_top_dir"
        ] = self.tomo_debug_top_dir
        self.tomo_recon_param_dict["file_params"][
            "cen_list_file"
        ] = self.tomo_cen_list_file
        self.tomo_recon_param_dict["file_params"][
            "alt_flat_file"
        ] = self.tomo_alt_flat_file
        self.tomo_recon_param_dict["file_params"][
            "alt_dark_file"
        ] = self.tomo_alt_dark_file
        self.tomo_recon_param_dict["file_params"][
            "wedge_ang_auto_det_ref_fn"
        ] = self.tomo_wedge_ang_auto_det_ref_fn
        self.tomo_recon_param_dict["file_params"][
            "io_confg"
        ] = self.global_h.io_tomo_cfg
        self.tomo_recon_param_dict["file_params"][
            "use_struc_h5_reader"
        ] = self.global_h.io_tomo_cfg["use_h5_reader"]
        self.tomo_recon_param_dict["recon_config"]["recon_type"] = self.tomo_recon_type
        self.tomo_recon_param_dict["recon_config"]["use_debug"] = self.tomo_use_debug
        self.tomo_recon_param_dict["recon_config"][
            "use_alt_flat"
        ] = self.tomo_use_alt_flat
        self.tomo_recon_param_dict["recon_config"][
            "use_alt_dark"
        ] = self.tomo_use_alt_dark
        self.tomo_recon_param_dict["recon_config"][
            "use_fake_flat"
        ] = self.tomo_use_fake_flat
        self.tomo_recon_param_dict["recon_config"][
            "use_fake_dark"
        ] = self.tomo_use_fake_dark
        self.tomo_recon_param_dict["recon_config"][
            "use_flat_blur"
        ] = self.tomo_use_blur_flat
        self.tomo_recon_param_dict["recon_config"][
            "use_rm_zinger"
        ] = self.tomo_use_rm_zinger
        self.tomo_recon_param_dict["recon_config"]["use_mask"] = self.tomo_use_mask
        self.tomo_recon_param_dict["recon_config"][
            "use_wedge_ang_auto_det"
        ] = self.tomo_use_wedge_ang_auto_det
        self.tomo_recon_param_dict["recon_config"]["is_wedge"] = self.tomo_is_wedge
        self.tomo_recon_param_dict["recon_config"][
            "use_config_file"
        ] = self.tomo_use_read_config
        self.tomo_recon_param_dict["flt_params"] = self.tomo_right_filter_dict
        self.tomo_recon_param_dict["data_params"]["scan_id"] = self.tomo_scan_id
        self.tomo_recon_param_dict["data_params"]["downsample"] = self.tomo_ds_ratio
        self.tomo_recon_param_dict["data_params"]["rot_cen"] = self.tomo_rot_cen
        self.tomo_recon_param_dict["data_params"]["cen_win_s"] = self.tomo_cen_win_s
        self.tomo_recon_param_dict["data_params"]["cen_win_w"] = self.tomo_cen_win_w
        self.tomo_recon_param_dict["data_params"][
            "fake_flat_val"
        ] = self.tomo_fake_flat_val
        self.tomo_recon_param_dict["data_params"][
            "fake_dark_val"
        ] = self.tomo_fake_dark_val
        self.tomo_recon_param_dict["data_params"]["fake_flat_roi"] = None
        self.tomo_recon_param_dict["data_params"]["sli_s"] = self.tomo_sli_s
        self.tomo_recon_param_dict["data_params"]["sli_e"] = self.tomo_sli_e
        self.tomo_recon_param_dict["data_params"]["col_s"] = self.tomo_col_s
        self.tomo_recon_param_dict["data_params"]["col_e"] = self.tomo_col_e
        self.tomo_recon_param_dict["data_params"]["chunk_sz"] = self.tomo_chunk_sz
        self.tomo_recon_param_dict["data_params"]["margin"] = self.tomo_margin
        self.tomo_recon_param_dict["data_params"][
            "blur_kernel"
        ] = self.tomo_flat_blur_kernel
        self.tomo_recon_param_dict["data_params"]["zinger_val"] = self.tomo_zinger_val
        self.tomo_recon_param_dict["data_params"]["mask_ratio"] = self.tomo_mask_ratio
        self.tomo_recon_param_dict["data_params"][
            "wedge_missing_s"
        ] = self.tomo_wedge_missing_s
        self.tomo_recon_param_dict["data_params"][
            "wedge_missing_e"
        ] = self.tomo_wedge_missing_e
        self.tomo_recon_param_dict["data_params"][
            "wedge_col_s"
        ] = self.tomo_wedge_auto_ref_col_s
        self.tomo_recon_param_dict["data_params"][
            "wedge_col_e"
        ] = self.tomo_wedge_auto_ref_col_e
        self.tomo_recon_param_dict["data_params"][
            "wedge_ang_auto_det_thres"
        ] = self.tomo_wedge_ang_auto_det_thres
        self.tomo_recon_param_dict["alg_params"] = {
            "algorithm": self.tomo_selected_alg,
            "params": self.alg_param_dict,
        }

    def set_widgets_from_rec_params(self, recon_param_dict):
        self.hs["UseAltFlat chbx"].value = self.tomo_use_alt_flat
        if self.tomo_use_alt_flat & (self.tomo_alt_flat_file is not None):
            self.hs["AltFlatFile btn"].files = [self.tomo_alt_flat_file]
            self.hs["AltFlatFile btn"].style.button_color = "lightgreen"
        else:
            self.hs["AltFlatFile btn"].files = []
            self.hs["AltFlatFile btn"].style.button_color = "orange"
        self.hs["UseAltDark chbx"].value = self.tomo_use_alt_dark
        if self.tomo_use_alt_dark & (self.tomo_alt_dark_file is not None):
            self.hs["AltDarkFile btn"].files = [self.tomo_alt_dark_file]
            self.hs["AltDarkFile btn"].style.button_color = "lightgreen"
        else:
            self.hs["AltDarkFile btn"].files = []
            self.hs["AltDarkFile btn"].style.button_color = "orange"
        self.hs["UseFakeFlat chbx"].value = self.tomo_use_fake_flat
        self.hs["UseFakeDark chbx"].value = self.tomo_use_fake_dark
        self.hs["UseBlurFlat chbx"].value = self.tomo_use_blur_flat
        self.hs["UseRmZinger chbx"].value = self.tomo_use_rm_zinger
        self.hs["UseMask chbx"].value = self.tomo_use_mask
        self.hs["AutoDet chbx"].value = self.tomo_use_wedge_ang_auto_det
        self.hs["IsWedge chbx"].value = self.tomo_is_wedge

        a = []
        for ii in sorted(self.tomo_right_filter_dict.keys()):
            a.append(self.tomo_right_filter_dict[ii]["filter_name"])
        self.hs["FilterConfigRightFlt mulsel"].options = a
        self.hs["FilterConfigRightFlt mulsel"].value = (a[0],)
        self.hs["FilterConfigLeftFltList drpdn"].value = a[0]
        self.tomo_left_box_selected_flt = a[0]
        self.set_flt_param_widgets(par_dict=self.tomo_right_filter_dict["0"]["params"])

        self.hs["AlgOptn drpdn"].value = self.tomo_selected_alg
        self.set_alg_param_widgets(par_dict=self.alg_param_dict)

        self.hs["ScanId drpdn"].value = self.tomo_scan_id
        self.hs["RoiColStart text"].value = self.tomo_col_s
        self.hs["RoiColEnd text"].value = self.tomo_col_e
        self.hs["DnSampRat text"].value = self.tomo_ds_ratio
        self.hs["RotCen text"].value = self.tomo_rot_cen
        self.hs["CenWinLeft text"].value = self.tomo_cen_win_s
        self.hs["CenWinWz text"].value = self.tomo_cen_win_w
        self.hs["FakeFlatVal text"].value = self.tomo_fake_flat_val
        self.hs["FakeDarkVal text"].value = self.tomo_fake_dark_val
        self.hs["ReconChunkSz text"].value = self.tomo_chunk_sz
        self.hs["ReconMargSz text"].value = self.tomo_margin
        self.hs["BlurKern text"].value = self.tomo_flat_blur_kernel
        self.hs["ZingerLevel text"].value = self.tomo_zinger_val
        if (
            self.tomo_use_wedge_ang_auto_det
            & self.tomo_is_wedge
            & (self.tomo_wedge_ang_auto_det_ref_fn is not None)
        ):
            self.hs["AutoRefFn btn"].files = [self.tomo_wedge_ang_auto_det_ref_fn]
            self.hs["AutoRefFn btn"].style.button_color = "lightgreen"
        else:
            self.hs["AutoRefFn btn"].files = []
            self.hs["AutoRefFn btn"].style.button_color = "orange"
        self.hs["MaskRat text"].value = self.tomo_mask_ratio
        self.hs["MissIdxStart text"].value = self.tomo_wedge_missing_s
        self.hs["MissIdxEnd text"].value = self.tomo_wedge_missing_e
        self.hs["AutoRefColStart text"].value = self.tomo_wedge_auto_ref_col_s
        self.hs["AutoRefColEnd text"].value = self.tomo_wedge_auto_ref_col_e
        self.hs["AutoThres text"].value = self.tomo_wedge_ang_auto_det_thres

    def set_rec_params_from_widgets(self):
        self.tomo_use_alt_flat = self.hs["UseAltFlat chbx"].value
        self.tomo_use_alt_dark = self.hs["UseAltDark chbx"].value
        self.tomo_use_fake_flat = self.hs["UseFakeFlat chbx"].value
        self.tomo_use_fake_dark = self.hs["UseFakeDark chbx"].value
        self.tomo_use_blur_flat = self.hs["UseBlurFlat chbx"].value
        self.tomo_use_rm_zinger = self.hs["UseRmZinger chbx"].value
        self.tomo_use_mask = self.hs["UseMask chbx"].value
        self.tomo_use_wedge_ang_auto_det = self.hs["AutoDet chbx"].value
        self.tomo_is_wedge = self.hs["IsWedge chbx"].value
        self.tomo_use_read_config = self.hs["UseConfig chbx"].value

        a = list(self.hs["FilterConfigRightFlt mulsel"].options)
        d = {}
        if len(a) > 0:
            cnt = 0
            for ii in sorted(self.tomo_right_filter_dict.keys()):
                d[cnt] = self.tomo_right_filter_dict[ii]
                cnt += 1
            self.tomo_right_filter_dict = d
        else:
            self.tomo_right_filter_dict = {0: {}}

        self.tomo_scan_id = self.hs["ScanId drpdn"].value
        self.tomo_ds_ratio = self.hs["DnSampRat text"].value
        self.tomo_rot_cen = self.hs["RotCen text"].value
        self.tomo_cen_win_s = self.hs["CenWinLeft text"].value
        self.tomo_cen_win_w = self.hs["CenWinWz text"].value
        self.tomo_fake_flat_val = self.hs["FakeFlatVal text"].value
        self.tomo_fake_dark_val = self.hs["FakeDarkVal text"].value
        if not self.hs["AltFlatFile btn"].files:
            self.tomo_alt_flat_file = None
        else:
            self.tomo_alt_flat_file = self.hs["AltFlatFile btn"].files[0]
        if not self.hs["AltDarkFile btn"].files:
            self.tomo_alt_dark_file = None
        else:
            self.tomo_alt_dark_file = self.hs["AltDarkFile btn"].files[0]
        self.tomo_sli_s = self.hs["RoiSliStart text"].value
        self.tomo_sli_e = self.hs["RoiSliEnd text"].value
        self.tomo_col_s = self.hs["RoiColStart text"].value
        self.tomo_col_e = self.hs["RoiColEnd text"].value
        self.tomo_chunk_sz = self.hs["ReconChunkSz text"].value
        self.tomo_margin = self.hs["ReconMargSz text"].value
        self.tomo_flat_blur_kernel = self.hs["BlurKern text"].value
        self.tomo_zinger_val = self.hs["ZingerLevel text"].value
        self.tomo_mask_ratio = self.hs["MaskRat text"].value
        self.tomo_wedge_missing_s = self.hs["MissIdxStart text"].value
        self.tomo_wedge_missing_e = self.hs["MissIdxEnd text"].value
        self.tomo_wedge_auto_ref_col_s = self.hs["AutoRefColStart text"].value
        self.tomo_wedge_auto_ref_col_e = self.hs["AutoRefColEnd text"].value
        self.tomo_wedge_ang_auto_det_thres = self.hs["AutoThres text"].value

    def reset_alg_param_widgets(self):
        for ii in range(3):
            self.alg_phs[ii].options = ""
            self.alg_phs[ii].description_tooltip = "p" + str(ii).zfill(2)
        for ii in range(3, 7):
            self.alg_phs[ii].value = 0
            self.alg_phs[ii].description_tooltip = "p" + str(ii).zfill(2)

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
        for h in self.alg_phs:
            h.disabled = True
            layout = {"width": "23.5%", "visibility": "hidden"}
            h.layout = layout
        alg = ALG_PARAM_DICT[self.tomo_selected_alg]
        for idx in alg.keys():
            self.alg_phs[idx].disabled = False
            layout = {"width": "23.5%", "visibility": "visible"}
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
        self.alg_param_dict = {}
        alg = ALG_PARAM_DICT[self.tomo_selected_alg]
        for idx in alg.keys():
            self.alg_param_dict[alg[idx][0]] = alg[idx][-1](self.alg_phs[idx].value)
        self.alg_param_dict = dict(OrderedDict(self.alg_param_dict))

    def reset_flt_param_widgets(self):
        for ii in range(6):
            self.flt_phs[ii].options = ""
            self.flt_phs[ii].description_tooltip = "p" + str(ii).zfill(2)
        for ii in range(6, 12):
            self.flt_phs[ii].value = 0
            self.flt_phs[ii].description_tooltip = "p" + str(ii).zfill(2)

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
        for h in self.flt_phs:
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
        self.flt_param_dict = {}
        flt = FILTER_PARAM_DICT[self.tomo_left_box_selected_flt]
        for idx in flt.keys():
            self.flt_param_dict[flt[idx][0]] = self.flt_phs[idx].value
        self.flt_param_dict = dict(OrderedDict(self.flt_param_dict))

    def read_config(self):
        if os.path.basename(self.tomo_cen_list_file).split(".")[-1] == "json":
            with open(self.tomo_cen_list_file, "r") as f:
                tem = dict(OrderedDict(json.load(f)))
            return tem
        else:
            print("json is the only allowed configuration file type.")
            return None

    def SelRawH5TopDir_btn_clk(self, a):
        self.reset_config()
        if len(a.files[0]) != 0:
            self.tomo_raw_data_top_dir = a.files[0]
            self.tomo_recon_top_dir = a.files[0]
            self.tomo_raw_data_file_template = os.path.join(
                self.tomo_raw_data_top_dir,
                self.global_h.io_tomo_cfg["tomo_raw_fn_template"],
            )
            b = ""
            t = time.strptime(time.asctime())
            for ii in range(6):
                b += str(t[ii]).zfill(2) + "-"
            self.tomo_trial_cen_dict_fn = os.path.join(
                self.tomo_raw_data_top_dir, "trial_cen_dict_{}.json".format(b)
            )
            self.tomo_recon_dict_fn = os.path.join(
                self.tomo_raw_data_top_dir, "recon_dict_{}.json".format(b)
            )
            self.tomo_raw_data_top_dir_set = True
            self.tomo_recon_path_set = True
            self.hs["SelRawH5TopDir btn"].initialdir = os.path.abspath(a.files[0])
            self.hs["SelSavReconDir btn"].initialdir = os.path.abspath(a.files[0])
            self.hs["SelSavDebugDir btn"].initialdir = os.path.abspath(a.files[0])
            self.hs["ReadConfig_btn"].initialdir = os.path.abspath(a.files[0])
            self.hs["AltFlatFile btn"].initialdir = os.path.abspath(a.files[0])
            self.hs["AltDarkFile btn"].initialdir = os.path.abspath(a.files[0])
            self.hs["AutoRefFn btn"].initialdir = os.path.abspath(a.files[0])
            update_json_content(
                self.global_h.GUI_cfg_file, {"cwd": os.path.abspath(a.files[0])}
            )
            self.global_h.cwd = os.path.os.path.abspath(a.files[0])
        else:
            self.tomo_raw_data_top_dir = None
            self.tomo_raw_data_top_dir_set = False
            self.tomo_recon_top_dir = None
            self.tomo_recon_path_set = False
            self.hs["SelRawH5TopDir text"].value = "Choose raw h5 top dir ..."
        self.tomo_filepath_configured = False
        self.recon_finish = -1
        self.hs[
            "SelFile&PathCfm text"
        ].value = "After setting directories, confirm to proceed ..."
        self.boxes_logic()

    def SelSavReconDir_btn_clk(self, a):
        self.reset_config()
        if not self.tomo_raw_data_top_dir_set:
            self.hs[
                "SelFile&PathCfm text"
            ].value = "Please specify raw h5 top directory first ..."
            self.hs[
                "SelSavReconDir text"
            ].value = "Choose top directory where recon subdirectories are saved..."
            self.tomo_recon_path_set = False
        else:
            if len(a.files[0]) != 0:
                self.tomo_recon_top_dir = a.files[0]
                self.tomo_recon_path_set = True
                # self.hs["SelSavReconDir btn"].initialdir = os.path.abspath(a.files[0])
                update_json_content(
                    self.global_h.GUI_cfg_file, {"cwd": os.path.abspath(a.files[0])}
                )
                self.global_h.cwd = os.path.os.path.abspath(a.files[0])
            else:
                self.tomo_recon_top_dir = None
                self.tomo_recon_path_set = False
                self.hs[
                    "SelSavReconDir text"
                ].value = "Select top directory where recon subdirectories are saved..."
            self.hs[
                "SelFile&PathCfm text"
            ].value = "After setting directories, confirm to proceed ..."
        self.tomo_filepath_configured = False
        self.recon_finish = -1
        self.boxes_logic()

    def SelSavDebugDir_btn_clk(self, a):
        self.reset_config()
        if not self.tomo_raw_data_top_dir_set:
            self.hs[
                "SelFile&PathCfm text"
            ].value = "Please specify raw h5 top directory first ..."
            self.hs[
                "SelSavDebugDir text"
            ].value = "Select top directory where debug dir will be created..."
            self.tomo_debug_path_set = False
        else:
            if len(a.files[0]) != 0:
                self.tomo_debug_top_dir = a.files[0]
                self.tomo_debug_path_set = True
                # self.hs["SelSavDebugDir btn"].initialdir = os.path.abspath(a.files[0])
                update_json_content(
                    self.global_h.GUI_cfg_file, {"cwd": os.path.abspath(a.files[0])}
                )
                self.global_h.cwd = os.path.os.path.abspath(a.files[0])
            else:
                self.tomo_debug_top_dir = None
                self.tomo_debug_path_set = False
                self.hs[
                    "SelSavDebugDir text"
                ].value = "Select top directory where debug dir will be created..."
            self.hs[
                "SelFile&PathCfm text"
            ].value = "After setting directories, confirm to proceed ..."
        self.tomo_filepath_configured = False
        self.recon_finish = -1
        self.boxes_logic()

    def SavDebug_chbx_chg(self, a):
        self.reset_config()
        if a["owner"].value:
            self.tomo_use_debug = True
            self.hs["SelSavDebugDir btn"].disabled = False
            self.hs[
                "SelSavDebugDir text"
            ].value = "Select top directory where debug dir will be created..."
            self.hs["SelSavDebugDir btn"].style.button_color = "orange"
        else:
            self.tomo_use_debug = False
            self.hs["SelSavDebugDir btn"].disabled = True
            self.hs["SelSavDebugDir text"].value = "Debug is disabled..."
            self.hs["SelSavDebugDir btn"].style.button_color = "orange"
        self.tomo_filepath_configured = False
        self.recon_finish = -1
        self.boxes_logic()

    def FilePathOptn_drpdn_chg(self, a):
        restart(self, dtype="TOMO")
        self.reset_config()
        self.tomo_recon_type = a["owner"].value
        if self.tomo_recon_type == "Trial Cent":
            layout = {"width": "15%", "height": "85%", "visibility": "hidden"}
            self.hs["ReadConfig_btn"].layout = layout
            layout = {"width": "7%", "visibility": "hidden"}
            self.hs["UseConfig chbx"].layout = layout
            layout = {"width": "19%", "visibility": "visible"}
            self.hs["CenWinLeft text"].layout = layout
            self.hs["CenWinWz text"].layout = layout

            self.hs["SelSavReconDir btn"].disabled = True
            self.hs["SelSavReconDir btn"].style.button_color = "orange"

            self.hs["SelSavReconDir btn"].disabled = False
            self.hs["SelSavReconDir btn"].style.button_color = "orange"
            self.hs[
                "SelSavReconDir text"
            ].value = (
                "Select top directory where data_center directory will be created..."
            )

            self.hs["UseConfig chbx"].value = False
            self.tomo_use_read_config = False
        elif self.tomo_recon_type == "Vol Recon":
            self.hs["SelSavReconDir btn"].disabled = False
            self.hs["SelSavReconDir btn"].style.button_color = "orange"
            self.hs[
                "SelSavReconDir text"
            ].value = (
                "Select top directory where recon subdirectories will be created..."
            )
            layout = {"width": "15%", "height": "85%", "visibility": "visible"}
            self.hs["ReadConfig_btn"].layout = layout
            layout = {"width": "7%", "visibility": "visible"}
            self.hs["UseConfig chbx"].layout = layout
            layout = {"width": "19%", "visibility": "hidden"}
            self.hs["CenWinLeft text"].layout = layout
            self.hs["CenWinWz text"].layout = layout

            self.hs["UseConfig chbx"].value = True
            self.tomo_use_read_config = True
        self.tomo_filepath_configured = False
        self.recon_finish = -1
        self.boxes_logic()

    def SelFilePathCfm_btn_clk(self, a):
        if self.tomo_recon_type == "Trial Cent":
            if self.tomo_raw_data_top_dir_set & self.tomo_recon_path_set:
                self.tomo_available_raw_idx = check_file_availability(
                    self.tomo_raw_data_top_dir,
                    scan_id=None,
                    signature=self.global_h.io_tomo_cfg["tomo_raw_fn_template"],
                    return_idx=True,
                )
                if len(self.tomo_available_raw_idx) == 0:
                    self.tomo_filepath_configured = False
                    return
                else:
                    self.hs["ScanId drpdn"].options = self.tomo_available_raw_idx
                    self.tomo_scan_id = self.tomo_available_raw_idx[0]
                    self.hs["ScanId drpdn"].value = self.tomo_scan_id
                    self.cal_set_srch_win()
                    if self.tomo_use_debug:
                        if self.tomo_debug_path_set:
                            self.tomo_data_center_path = os.path.join(
                                self.tomo_recon_top_dir, "data_center"
                            )
                            self.tomo_filepath_configured = True
                            self.set_alg_param_widgets()
                            self.set_flt_param_widgets()
                        else:
                            self.hs[
                                "SelFile&PathCfm text"
                            ].value = "You need to select the top directory to create debug dir..."
                            self.tomo_filepath_configured = False
                    else:
                        self.tomo_data_center_path = os.path.join(
                            self.tomo_recon_top_dir, "data_center"
                        )
                        self.tomo_filepath_configured = True
                        self.set_alg_param_widgets()
                        self.set_flt_param_widgets()
            else:
                self.hs[
                    "SelFile&PathCfm text"
                ].value = "You need to select the top raw dir and top dir where debug dir can be created..."
                self.tomo_filepath_configured = False
        elif self.tomo_recon_type == "Vol Recon":
            if self.tomo_raw_data_top_dir_set & self.tomo_recon_path_set:
                self.tomo_available_raw_idx = check_file_availability(
                    self.tomo_raw_data_top_dir,
                    scan_id=None,
                    signature=self.global_h.io_tomo_cfg["tomo_raw_fn_template"],
                    return_idx=True,
                )
                if len(self.tomo_available_raw_idx) == 0:
                    print(
                        'No data file available in {self.tomo_raw_data_top_dir} with filename pattern of {self.global_h.io_tomo_cfg["tomo_raw_fn_template"]}'
                    )
                    self.tomo_filepath_configured = False
                    return
                else:
                    self.hs["ScanId drpdn"].options = self.tomo_available_raw_idx
                    self.hs["ScanId drpdn"].value = self.tomo_available_raw_idx[0]
                    self.tomo_filepath_configured = True
                    self.set_alg_param_widgets()
                    self.set_flt_param_widgets()
            else:
                self.hs[
                    "SelFile&PathCfm text"
                ].value = "You need to select the top raw dir and top dir where recon dir can be created..."
                self.tomo_filepath_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def ScanId_drpdn_chg(self, a):
        self.tomo_scan_id = a["owner"].value
        self.cal_set_srch_win()
        if self.data_info:
            if self.tomo_recon_type == "Trial Cent":
                self.hs["RoiSliEnd text"].value = 1
                self.hs["RoiSliEnd text"].max = self.data_info["img_dim"][1] - 1
                self.hs["RoiSliEnd text"].value = (
                    int(self.data_info["img_dim"][1] / 2) + 10
                )
                self.hs["RoiSliStart text"].value = 0
                self.hs["RoiSliStart text"].max = self.data_info["img_dim"][1] - 2
                self.hs["RoiSliStart text"].value = (
                    int(self.data_info["img_dim"][1] / 2) - 10
                )
                self.hs["RoiColStart text"].value = 0
                self.hs["RoiColStart text"].max = self.data_info["img_dim"][2] - 1
                self.hs["RoiColEnd text"].value = 0
                self.hs["RoiColEnd text"].max = self.data_info["img_dim"][2] - 1
                self.hs["RoiColEnd text"].value = self.data_info["img_dim"][2] - 1
                self.hs["AutoRefColStart text"].value = 0
                self.hs["AutoRefColStart text"].max = self.data_info["img_dim"][2] - 1
                self.hs["AutoRefColEnd text"].value = 0
                self.hs["AutoRefColEnd text"].max = self.data_info["img_dim"][2] - 1
                self.hs["AutoRefColEnd text"].value = self.data_info["img_dim"][2] - 1
            elif self.tomo_recon_type == "Vol Recon":
                if self.hs["RoiSliEnd text"].value > (self.data_info["img_dim"][1] - 1):
                    self.hs["RoiSliEnd text"].value = self.data_info["img_dim"][1] - 1
                self.hs["RoiSliEnd text"].max = self.data_info["img_dim"][1] - 1
                if self.hs["RoiSliStart text"].value > (
                    self.data_info["img_dim"][1] - 2
                ):
                    self.hs["RoiSliStart text"].value = self.data_info["img_dim"][1] - 2
                self.hs["RoiSliStart text"].max = self.data_info["img_dim"][1] - 2
                if self.hs["RoiColStart text"].value > (
                    self.data_info["img_dim"][2] - 2
                ):
                    self.hs["RoiColStart text"].value = self.data_info["img_dim"][2] - 2
                self.hs["RoiColStart text"].max = self.data_info["img_dim"][2] - 1
                if self.hs["RoiColEnd text"].value > (self.data_info["img_dim"][2] - 1):
                    self.hs["RoiColEnd text"].value = self.data_info["img_dim"][2] - 1
                self.hs["RoiColEnd text"].max = self.data_info["img_dim"][2] - 1

                self.hs["AutoRefColStart text"].value = 0
                self.hs["AutoRefColStart text"].max = self.data_info["img_dim"][2] - 1
                self.hs["AutoRefColEnd text"].value = 0
                self.hs["AutoRefColEnd text"].max = self.data_info["img_dim"][2] - 1
                self.hs["AutoRefColEnd text"].value = self.data_info["img_dim"][2] - 1
            self.hs["RawProj sldr"].value = 0
            self.hs["RawProj sldr"].min = 0
            self.hs["RawProj sldr"].max = self.data_info["img_dim"][0] - 1
            self.hs["RawProjInMem chbx"].value = False
            self.raw_proj = np.ndarray(
                [self.data_info["img_dim"][1], self.data_info["img_dim"][2]],
                dtype=np.float32,
            )
            self.raw_proj_0 = np.ndarray(
                [self.data_info["img_dim"][1], self.data_info["img_dim"][2]],
                dtype=np.float32,
            )
            self.raw_proj_180 = np.ndarray(
                [self.data_info["img_dim"][1], self.data_info["img_dim"][2]],
                dtype=np.float32,
            )
            self.raw_is_in_mem = False
        else:
            print(
                "Cannot read metadata from the available files with the pre-defined Tomo Data Info Configuration."
            )
            self.tomo_filepath_configured = False
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def RotCen_text_chg(self, a):
        self.tomo_rot_cen = [a["owner"].value]
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def CenWinLeft_text_chg(self, a):
        if a["owner"].value > (
            self.data_info["img_dim"][2] - self.hs["CenWinWz text"].value - 1
        ):
            a["owner"].value = (
                self.data_info["img_dim"][2] - self.hs["CenWinWz text"].value - 1
            )
        self.tomo_cen_win_s = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def CenWinWz_text_chg(self, a):
        if a["owner"].value > (
            self.data_info["img_dim"][2] - self.hs["CenWinLeft text"].value - 1
        ):
            a["owner"].value = (
                self.data_info["img_dim"][2] - self.hs["CenWinLeft text"].value - 1
            )
        self.tomo_cen_win_w = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def ReadConfig_btn_clk(self, a):
        if self.tomo_use_read_config and (a.files[0] is not None):
            self.tomo_cen_list_file = a.files[0]
            tem = self.read_config()
            if tem is not None:
                key = next(iter(tem.keys()))
                recon_param_dict = tem[key]
                self.set_rec_params_from_rec_dict(recon_param_dict)
                print("1:", self.tomo_recon_param_dict)
                self.set_widgets_from_rec_params(recon_param_dict)
                self.hs["RoiSliStart text"].value = 0
                self.hs["RoiSliEnd text"].value = self.data_info["img_dim"][1] - 1
                self.set_rec_params_from_widgets()
                self.set_rec_dict_from_rec_params()
            else:
                print("Fail to read the configuration file.")
                self.tomo_cen_list_file = None
                return
        else:
            self.tomo_cen_list_file = None
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def UseConfig_chbx_chg(self, a):
        self.tomo_use_read_config = a["owner"].value
        if not self.tomo_use_read_config:
            self.tomo_cen_list_file = None
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def UseAltFlat_chbx_chg(self, a):
        self.tomo_use_alt_flat = a["owner"].value
        if self.tomo_use_alt_flat:
            self.hs["UseFakeFlat chbx"].value = False
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def AltFlatFile_btn_clk(self, a):
        if len(a.files[0]) != 0:
            self.tomo_alt_flat_file = a.files[0]
        else:
            self.tomo_alt_flat_file = None
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def UseAltDark_chbx_chg(self, a):
        self.tomo_use_alt_dark = a["owner"].value
        if self.tomo_use_alt_dark:
            self.hs["UseFakeDark chbx"].value = False
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def AltDarkFile_btn_clk(self, a):
        if len(a.files[0]) != 0:
            self.tomo_alt_dark_file = a.files[0]
        else:
            self.tomo_alt_dark_file = None
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def UseFakeFlat_chbx_chg(self, a):
        self.tomo_use_fake_flat = a["owner"].value
        if self.tomo_use_fake_flat:
            self.hs["UseAltFlat chbx"].value = False
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FakeFlatVal_text_chg(self, a):
        self.tomo_fake_flat_val = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def UseFakeDark_chbx_chg(self, a):
        self.tomo_use_fake_dark = a["owner"].value
        if self.tomo_use_fake_dark:
            self.hs["UseAltDark chbx"].value = False
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FakeDarkVal_text_chg(self, a):
        self.tomo_fake_dark_val = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def RoiSliStart_text_chg(self, a):
        if self.tomo_recon_type == "Trial Cent":
            if (a["owner"].value + 20) > self.hs["RoiSliEnd text"].max:
                self.hs["RoiSliEnd text"].value = self.hs["RoiSliEnd text"].max
            else:
                self.hs["RoiSliEnd text"].value = a["owner"].value + 20
        else:
            if a["owner"].value >= self.hs["RoiSliEnd text"].value:
                a["owner"].value = self.hs["RoiSliEnd text"].value - 1
        self.tomo_sli_s = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def RoiSliEnd_text_chg(self, a):
        if a["owner"].value <= self.hs["RoiSliStart text"].value:
            a["owner"].value = self.hs["RoiSliStart text"].value + 1
        self.tomo_sli_e = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def RoiColStart_text_chg(self, a):
        if a["owner"].value >= self.hs["RoiColEnd text"].value:
            a["owner"].value = self.hs["RoiColEnd text"].value - 1
        self.tomo_col_s = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def RoiColEnd_text_chg(self, a):
        if a["owner"].value <= self.hs["RoiColStart text"].value:
            a["owner"].value = self.hs["RoiColStart text"].value + 1
        self.tomo_col_e = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def DnSampRat_text_chg(self, a):
        self.tomo_ds_ratio = a["owner"].value
        if self.tomo_ds_ratio == 1:
            self.tomo_use_downsample = False
        else:
            self.tomo_use_downsample = True
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def BlurFlat_chbx_chg(self, a):
        self.tomo_use_blur_flat = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def BlurKern_text_chg(self, a):
        self.tomo_flat_blur_kernel = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def UseRmZinger_chbx_chg(self, a):
        self.tomo_use_rm_zinger = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def ZingerLevel_text_chg(self, a):
        self.tomo_zinger_val = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def UseMask_chbx_chg(self, a):
        self.tomo_use_mask = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def MaskRat_text_chg(self, a):
        self.tomo_mask_ratio = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def AlgOptn_drpdn_chg(self, a):
        self.tomo_selected_alg = a["owner"].value
        self.set_alg_param_widgets()
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def AlgPar00_drpdn_chg(self, a):
        self.tomo_alg_p01 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def AlgPar01_drpdn_chg(self, a):
        self.tomo_alg_p02 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def AlgPar02_drpdn_chg(self, a):
        self.tomo_alg_p03 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def AlgPar03_text_chg(self, a):
        self.tomo_alg_p04 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def AlgPar04_text_chg(self, a):
        self.tomo_alg_p05 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def AlgPar05_text_chg(self, a):
        self.tomo_alg_p06 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def AlgPar06_text_chg(self, a):
        self.tomo_alg_p07 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def IsWedge_chbx_chg(self, a):
        self.tomo_is_wedge = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def MissIdxStart_text_chg(self, a):
        self.tomo_wedge_missing_s = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def MissIdxEnd_text_chg(self, a):
        self.tomo_wedge_missing_e = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def AutoDet_chbx_chg(self, a):
        self.tomo_use_wedge_ang_auto_det = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def AutoThres_text_chg(self, a):
        self.tomo_wedge_ang_auto_det_thres = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def AutoRefFn_btn_clk(self, a):
        if len(a.files[0]) != 0:
            self.tomo_wedge_ang_auto_det_ref_fn = a.files[0]
            self.set_rec_dict_from_rec_params()
            cfg = deepcopy(self.tomo_recon_param_dict)
            cfg["file_params"][
                "wedge_ang_auto_det_ref_fn"
            ] = self.tomo_wedge_ang_auto_det_ref_fn
            cfg["file_params"]["reader"] = self.reader
            cfg["file_params"]["info_reader"] = data_info(tomo_h5_info)
            cfg["recon_config"]["use_ds"] = (
                True if self.tomo_use_downsample == 1 else False
            )
            self.wedge_eva_data, flat, dark, _ = read_data(
                self.reader,
                self.tomo_wedge_ang_auto_det_ref_fn,
                cfg,
                sli_start=self.tomo_sli_s,
                sli_end=self.tomo_sli_e,
                col_start=self.tomo_wedge_auto_ref_col_s,
                col_end=self.tomo_wedge_auto_ref_col_e,
                ds_use=self.tomo_use_downsample,
                ds_level=self.tomo_ds_ratio,
                mean_axis=2,
            )
            self.hs["AutoRefSli sldr"].value = 0
            self.hs["AutoRefSli sldr"].max = self.wedge_eva_data.shape[1] - 1
        else:
            self.tomo_wedge_ang_auto_det_ref_fn = None
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def AutoRefSli_sldr_chg(self, a):
        plt.figure(0)
        plt.plot(self.wedge_eva_data[:, a["owner"].value])
        plt.plot(
            np.arange(self.wedge_eva_data.shape[0]),
            np.ones(self.wedge_eva_data.shape[0]) * self.tomo_wedge_ang_auto_det_thres,
        )
        plt.show()
        self.boxes_logic()
        self.tomo_compound_logic()

    def AutoRefColStart_text_chg(self, a):
        if a["owner"].value >= self.hs["AutoRefColEnd text"].value:
            a["owner"].value = self.hs["AutoRefColEnd text"].value - 1
        self.tomo_wedge_auto_ref_col_s = a["owner"].value

    def AutoRefColEnd_text_chg(self, a):
        if a["owner"].value <= self.hs["AutoRefColStart text"].value:
            a["owner"].value = self.hs["AutoRefColStart text"].value + 1
        self.tomo_wedge_auto_ref_col_e = a["owner"].value

    def RawProj_sldr_chg(self, a):
        idx = a["owner"].value
        if self.load_raw_in_mem:
            if not self.raw_is_in_mem:
                self.raw = (
                    (
                        self.reader(
                            self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                            dtype="data",
                            sli=[None, None, None],
                            cfg=self.global_h.io_tomo_cfg,
                        )
                        - self.reader(
                            self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                            dtype="dark",
                            sli=[None, None, None],
                            cfg=self.global_h.io_tomo_cfg,
                        ).mean(axis=0)
                    )
                    / (
                        self.reader(
                            self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                            dtype="flat",
                            sli=[None, None, None],
                            cfg=self.global_h.io_tomo_cfg,
                        ).mean(axis=0)
                        - self.reader(
                            self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                            dtype="dark",
                            sli=[None, None, None],
                            cfg=self.global_h.io_tomo_cfg,
                        ).mean(axis=0)
                    )
                ).astype(np.float32)
                self.raw[:] = np.where(self.raw < 0, 1, self.raw)[:]
                self.raw[np.isinf(self.raw)] = 1
                self.raw_is_in_mem = True

            data_state, viewer_state = fiji_viewer_state(
                self.global_h, self, viewer_name="tomo_raw_img_viewer"
            )
            if (not data_state) | (not viewer_state):
                fiji_viewer_on(self.global_h, self, viewer_name="tomo_raw_img_viewer")
                self.global_h.tomo_fiji_windows["tomo_raw_img_viewer"]["ip"].setImage(
                    self.global_h.ij.convert().convert(
                        self.global_h.ij.dataset().create(
                            self.global_h.ij.py.to_java(self.raw)
                        ),
                        self.global_h.ImagePlusClass,
                    )
                )
            self.global_h.tomo_fiji_windows["tomo_raw_img_viewer"]["ip"].setSlice(idx)
            self.global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")"""
            )
            self.global_h.tomo_fiji_windows["tomo_raw_img_viewer"]["ip"].setRoi(
                self.tomo_col_s,
                self.tomo_sli_s,
                self.tomo_col_e - self.tomo_col_s,
                self.tomo_sli_e - self.tomo_sli_s,
            )
        else:
            self.raw_proj[:] = (
                (
                    self.reader(
                        self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                        dtype="data",
                        sli=[[idx, idx + 1], None, None],
                        cfg=self.global_h.io_tomo_cfg,
                    )
                    - self.reader(
                        self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                        dtype="dark",
                        sli=[None, None, None],
                        cfg=self.global_h.io_tomo_cfg,
                    ).mean(axis=0)
                )
                / (
                    self.reader(
                        self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                        dtype="flat",
                        sli=[None, None, None],
                        cfg=self.global_h.io_tomo_cfg,
                    ).mean(axis=0)
                    - self.reader(
                        self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                        dtype="dark",
                        sli=[None, None, None],
                        cfg=self.global_h.io_tomo_cfg,
                    ).mean(axis=0)
                )
            ).astype(np.float32)
            self.raw_proj[:] = np.where(self.raw_proj < 0, 1, self.raw_proj)[:]
            self.raw_proj[np.isinf(self.raw_proj)] = 1

            data_state, viewer_state = fiji_viewer_state(
                self.global_h, self, viewer_name="tomo_raw_img_viewer"
            )
            if (not data_state) | (not viewer_state):
                fiji_viewer_on(self.global_h, self, viewer_name="tomo_raw_img_viewer")
            self.global_h.tomo_fiji_windows["tomo_raw_img_viewer"]["ip"].setImage(
                self.global_h.ij.convert().convert(
                    self.global_h.ij.dataset().create(
                        self.global_h.ij.py.to_java(self.raw_proj)
                    ),
                    self.global_h.ImagePlusClass,
                )
            )
            self.global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")"""
            )
            self.global_h.tomo_fiji_windows["tomo_raw_img_viewer"]["ip"].setRoi(
                self.tomo_col_s,
                self.tomo_sli_s,
                self.tomo_col_e - self.tomo_col_s,
                self.tomo_sli_e - self.tomo_sli_s,
            )

    def RawProjInMem_chbx_chg(self, a):
        self.load_raw_in_mem = a["owner"].value
        fiji_viewer_off(self.global_h, self, viewer_name="tomo_raw_img_viewer")

    def RawProjViewerClose_btn_clk(self, a):
        data_state, viewer_state = fiji_viewer_state(
            self.global_h, self, viewer_name="tomo_raw_img_viewer"
        )
        if viewer_state is not None:
            self.set_rec_params_from_widgets()
            x = (
                self.global_h.tomo_fiji_windows["tomo_raw_img_viewer"]["ip"]
                .getRoi()
                .getFloatPolygon()
                .xpoints
            )
            y = (
                self.global_h.tomo_fiji_windows["tomo_raw_img_viewer"]["ip"]
                .getRoi()
                .getFloatPolygon()
                .ypoints
            )
            self.hs["AutoRefColStart text"].value = int(x[0])
            self.hs["AutoRefColEnd text"].value = int(x[2])
        fiji_viewer_off(self.global_h, self, viewer_name="tomo_raw_img_viewer")

    def CenOffsetRange_sldr_chg(self, a):
        self.raw_proj_0[:] = (
            (
                self.reader(
                    self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                    dtype="data",
                    sli=[[0, 1], None, None],
                    cfg=self.global_h.io_tomo_cfg,
                )
                - self.reader(
                    self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                    dtype="dark",
                    sli=[None, None, None],
                    cfg=self.global_h.io_tomo_cfg,
                ).mean(axis=0)
            )
            / (
                self.reader(
                    self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                    dtype="flat",
                    sli=[None, None, None],
                    cfg=self.global_h.io_tomo_cfg,
                ).mean(axis=0)
                - self.reader(
                    self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                    dtype="dark",
                    sli=[None, None, None],
                    cfg=self.global_h.io_tomo_cfg,
                ).mean(axis=0)
            )
        ).astype(np.float32)
        self.raw_proj_0[:] = np.where(self.raw_proj_0 < 0, 1, self.raw_proj_0)[:]
        self.raw_proj_0[np.isinf(self.raw_proj_0)] = 1
        self.raw_proj_180[:] = (
            (
                self.reader(
                    self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                    dtype="data",
                    sli=[[-2, -1], None, None],
                    cfg=self.global_h.io_tomo_cfg,
                )
                - self.reader(
                    self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                    dtype="dark",
                    sli=[None, None, None],
                    cfg=self.global_h.io_tomo_cfg,
                ).mean(axis=0)
            )
            / (
                self.reader(
                    self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                    dtype="flat",
                    sli=[None, None, None],
                    cfg=self.global_h.io_tomo_cfg,
                ).mean(axis=0)
                - self.reader(
                    self.tomo_raw_data_file_template.format(self.tomo_scan_id),
                    dtype="dark",
                    sli=[None, None, None],
                    cfg=self.global_h.io_tomo_cfg,
                ).mean(axis=0)
            )
        ).astype(np.float32)
        self.raw_proj_180[:] = np.where(self.raw_proj_180 < 0, 1, self.raw_proj_180)[:]
        self.raw_proj_180[np.isinf(self.raw_proj_180)] = 1

        data_state, viewer_state = fiji_viewer_state(
            self.global_h, self, viewer_name="tomo_0&180_viewer"
        )
        if (not data_state) | (not viewer_state):
            fiji_viewer_on(self.global_h, self, viewer_name="tomo_0&180_viewer")
        self.global_h.tomo_fiji_windows["tomo_0&180_viewer"]["ip"].setImage(
            self.global_h.ij.convert().convert(
                self.global_h.ij.dataset().create(
                    self.global_h.ij.py.to_java(
                        np.roll(self.raw_proj_180[:, ::-1], a["owner"].value, axis=1)
                        - self.raw_proj_0
                    )
                ),
                self.global_h.ImagePlusClass,
            )
        )
        self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")

    def CenOffsetCfm_btn_clk(self, a):
        self.manual_cen = self.hs["CenOffsetRange sldr"].value / 2.0
        self.hs["RotCen text"].value = (
            self.manual_cen + self.data_info["img_dim"][2] / 2
        )
        self.hs["CenWinLeft text"].value = self.hs["RotCen text"].value - 10
        self.hs["CenWinWz text"].value = 80
        fiji_viewer_off(self.global_h, self, viewer_name="tomo_0&180_viewer")

    def CenViewerClose_btn_clk(self, a):
        fiji_viewer_off(self.global_h, self, viewer_name="tomo_0&180_viewer")

    def TrialCenPrev_sldr_chg(self, a):
        data_state, viewer_state = fiji_viewer_state(
            self.global_h, self, viewer_name="tomo_cen_review_viewer"
        )
        if (not data_state) | (not viewer_state):
            fiji_viewer_on(self.global_h, self, viewer_name="tomo_cen_review_viewer")
        self.global_h.tomo_fiji_windows["tomo_cen_review_viewer"]["ip"].setSlice(
            a["owner"].value
        )
        self.global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")

    def TrialCenCfm_btn_clk(self, a):
        self.trial_cen = (
            self.hs["CenWinLeft text"].value
            + (
                self.global_h.tomo_fiji_windows["tomo_cen_review_viewer"]["ip"].getZ()
                - 1
            )
            * 0.5
        )
        self.hs["RotCen text"].value = self.trial_cen

        self.read_alg_param_widgets()
        self.set_rec_params_from_widgets()
        self.set_rec_dict_from_rec_params()

        try:
            with open(self.tomo_trial_cen_dict_fn, "r") as f:
                tem = json.load(f)
            with open(self.tomo_trial_cen_dict_fn, "w") as f:
                tem[
                    str(self.tomo_recon_param_dict["data_params"]["scan_id"])
                ] = self.tomo_recon_param_dict
                tem[str(self.tomo_recon_param_dict["data_params"]["scan_id"])][
                    "file_params"
                ]["io_confg"]["customized_reader"]["user_tomo_reader"] = ""
                tem[str(self.tomo_recon_param_dict["data_params"]["scan_id"])][
                    "file_params"
                ]["reader"] = ""
                tem[str(self.tomo_recon_param_dict["data_params"]["scan_id"])][
                    "file_params"
                ]["info_reader"] = ""
                self.tomo_recon_param_dict["recon_config"]["recon_type"] = "Vol Recon"
                json.dump(tem, f, indent=4, separators=(",", ": "))
        except:
            tem = {}
            tem[
                str(self.tomo_recon_param_dict["data_params"]["scan_id"])
            ] = self.tomo_recon_param_dict
            tem[str(self.tomo_recon_param_dict["data_params"]["scan_id"])][
                "file_params"
            ]["io_confg"]["customized_reader"]["user_tomo_reader"] = ""
            tem[str(self.tomo_recon_param_dict["data_params"]["scan_id"])][
                "file_params"
            ]["reader"] = ""
            tem[str(self.tomo_recon_param_dict["data_params"]["scan_id"])][
                "file_params"
            ]["info_reader"] = ""
            self.tomo_recon_param_dict["recon_config"]["recon_type"] = "Vol Recon"
            with open(self.tomo_trial_cen_dict_fn, "w") as f:
                json.dump(tem, f, indent=4, separators=(",", ": "))
        fiji_viewer_off(self.global_h, self, viewer_name="tomo_cen_review_viewer")

    def TrialCenViewerClose_btn_clk(self, a):
        fiji_viewer_off(self.global_h, self, viewer_name="tomo_cen_review_viewer")

    def VolViewOpt_tgbtn_chg(self, a):
        pass

    def VolSliViewerClose_btn_clk(self, a):
        pass

    def ReconChunkSz_text_chg(self, a):
        if a["owner"].value < self.hs["ReconMargSz text"].value * 2:
            a["owner"].value = self.hs["ReconMargSz text"].value * 2 + 1
        self.tomo_chunk_sz = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def ReconMargSz_text_chg(self, a):
        if 2 * a["owner"].value > self.hs["ReconChunkSz text"].value:
            a["owner"].value = int(self.hs["ReconChunkSz text"].value / 2) - 1
        self.tomo_margin = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigLeftFltList_drpdn_chg(self, a):
        self.tomo_left_box_selected_flt = a["owner"].value
        self.set_flt_param_widgets()
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigLeftAddTo_btn_clk(self, a):
        self.read_flt_param_widgets()
        if (len(self.hs["FilterConfigRightFlt mulsel"].options) == 1) & (
            self.hs["FilterConfigRightFlt mulsel"].options[0] == "None"
        ):
            self.hs["FilterConfigRightFlt mulsel"].options = [
                self.tomo_left_box_selected_flt,
            ]
            self.hs["FilterConfigRightFlt mulsel"].value = [
                self.tomo_left_box_selected_flt,
            ]
            self.tomo_right_filter_dict[0] = {
                "filter_name": self.tomo_left_box_selected_flt,
                "params": self.flt_param_dict,
            }
        else:
            a = list(self.hs["FilterConfigRightFlt mulsel"].options)
            a.append(self.tomo_left_box_selected_flt)
            self.hs["FilterConfigRightFlt mulsel"].options = a
            self.hs["FilterConfigRightFlt mulsel"].value = self.hs[
                "FilterConfigRightFlt mulsel"
            ].options
            idx = len(a) - 1
            self.tomo_right_filter_dict[idx] = {
                "filter_name": self.tomo_left_box_selected_flt,
                "params": self.flt_param_dict,
            }
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigLeftPar00_text_chg(self, a):
        self.tomo_left_box_selected_flt_p00 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigLeftPar01_text_chg(self, a):
        self.tomo_left_box_selected_flt_p01 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigLeftPar02_text_chg(self, a):
        self.tomo_left_box_selected_flt_p02 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigLeftPar03_text_chg(self, a):
        self.tomo_left_box_selected_flt_p03 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigLeftPar04_text_chg(self, a):
        self.tomo_left_box_selected_flt_p04 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigLeftPar05_text_chg(self, a):
        self.tomo_left_box_selected_flt_p05 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigLeftPar06_text_chg(self, a):
        self.tomo_left_box_selected_flt_p06 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigLeftPar07_text_chg(self, a):
        self.tomo_left_box_selected_flt_p07 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigLeftPar08_text_chg(self, a):
        self.tomo_left_box_selected_flt_p08 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigLeftPar09_text_chg(self, a):
        self.tomo_left_box_selected_flt_p09 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigLeftPar10_text_chg(self, a):
        self.tomo_left_box_selected_flt_p10 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigLeftPar11_text_chg(self, a):
        self.tomo_left_box_selected_flt_p11 = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigRightFlt_mulsel_chg(self, a):
        self.tomo_right_list_filter = a["owner"].value
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigRightMvUp_btn_clk(self, a):
        if len(self.hs["FilterConfigRightFlt mulsel"].options) == 1:
            pass
        else:
            a = np.array(self.hs["FilterConfigRightFlt mulsel"].options)
            idxs = np.array(self.hs["FilterConfigRightFlt mulsel"].index)
            cnt = 0
            for b in idxs:
                if b == 0:
                    idxs[cnt] = b
                else:
                    a[b], a[b - 1] = a[b - 1], a[b]
                    (
                        self.tomo_right_filter_dict[b],
                        self.tomo_right_filter_dict[b - 1],
                    ) = (
                        self.tomo_right_filter_dict[b - 1],
                        self.tomo_right_filter_dict[b],
                    )
                    idxs[cnt] = b - 1
                cnt += 1
            self.hs["FilterConfigRightFlt mulsel"].options = list(a)
            self.hs["FilterConfigRightFlt mulsel"].value = list(a[idxs])
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigRightMvDn_btn_clk(self, a):
        if len(self.hs["FilterConfigRightFlt mulsel"].options) == 1:
            pass
        else:
            a = np.array(self.hs["FilterConfigRightFlt mulsel"].options)
            idxs = np.array(self.hs["FilterConfigRightFlt mulsel"].index)
            cnt = 0
            for b in idxs:
                if b == (len(a) - 1):
                    idxs[cnt] = b
                else:
                    a[b], a[b + 1] = a[b + 1], a[b]
                    (
                        self.tomo_right_filter_dict[b],
                        self.tomo_right_filter_dict[b + 1],
                    ) = (
                        self.tomo_right_filter_dict[b + 1],
                        self.tomo_right_filter_dict[b],
                    )
                    idxs[cnt] = b + 1
                cnt += 1
            self.hs["FilterConfigRightFlt mulsel"].options = list(a)
            self.hs["FilterConfigRightFlt mulsel"].value = list(a[idxs])
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigRightRm_btn_clk(self, a):
        a = list(self.hs["FilterConfigRightFlt mulsel"].options)
        idxs = list(self.hs["FilterConfigRightFlt mulsel"].index)
        d = {}
        for b in sorted(idxs, reverse=True):
            del a[b]
            del self.tomo_right_filter_dict[b]
        if len(a) > 0:
            self.hs["FilterConfigRightFlt mulsel"].options = list(a)
            self.hs["FilterConfigRightFlt mulsel"].value = [
                a[0],
            ]
            cnt = 0
            for ii in sorted(self.tomo_right_filter_dict.keys()):
                d[cnt] = self.tomo_right_filter_dict[ii]
                cnt += 1
            self.tomo_right_filter_dict = d
        else:
            self.hs["FilterConfigRightFlt mulsel"].options = [
                "None",
            ]
            self.hs["FilterConfigRightFlt mulsel"].value = [
                "None",
            ]
            self.tomo_right_filter_dict = {0: {}}
        self.tomo_data_configured = False
        self.recon_finish = -1
        self.boxes_logic()
        self.tomo_compound_logic()

    def FilterConfigRightFnsh_btn_clk(self, a):
        pass

    def FilterConfigCfm_btn_clk(self, a):
        """
        enforce self.tomo_recon_param_dict has same structure as that in tomo_recon_tools
        """
        self.read_alg_param_widgets()
        self.set_rec_params_from_widgets()
        self.set_rec_dict_from_rec_params()

        self.tomo_data_configured = True
        if self.tomo_recon_type == "Trial Cent":
            info = ""
            for key, val in self.tomo_recon_param_dict.items():
                info = info + str(key) + ":" + str(val) + "\n\n"
            self.hs["ReconConfigSumm text"].value = info
        else:
            info = (
                "file_params:{} \n\n"
                + "recon_config:{} \n\n"
                + "flt_params:{} \n\n"
                + "data_params:{} \n\n"
                + "alg_params:{}"
            )
            self.hs["ReconConfigSumm text"].value = info
        self.boxes_logic()
        self.tomo_compound_logic()

    def Recon_btn_clk(self, a):
        if self.tomo_recon_type == "Trial Cent":
            fiji_viewer_off(self.global_h, self, viewer_name="tomo_cen_review_viewer")
            code = {}
            ln = 0
            code[ln] = f"from collections import OrderedDict"
            ln += 1
            code[ln] = f"from TXM_Sandbox.utils.tomo_recon_tools import run_engine"
            ln += 1
            code[
                ln
            ] = f"from TXM_Sandbox.utils.io import data_reader, tomo_h5_reader, data_info, tomo_h5_info"
            ln += 1
            code[ln] = f"params = {self.tomo_recon_param_dict}"
            ln += 1
            code[ln] = f"if {self.global_h.io_tomo_cfg['use_h5_reader']}:"
            ln += 1
            code[
                ln
            ] = f"    params['file_params']['reader'] = data_reader(tomo_h5_reader)"
            ln += 1
            code[
                ln
            ] = f"    params['file_params']['info_reader'] = data_info(tomo_h5_info)"
            ln += 1
            code[ln] = f"else:"
            ln += 1
            code[
                ln
            ] = f"    from TXM_Sandbox.external.user_io import user_tomo_reader, user_tomo_info_reader"
            ln += 1
            code[ln] = f"    self.reader = data_reader(user_tomo_reader)"
            ln += 1
            code[ln] = f"    self.info_reader = data_info(user_tomo_info_reader)"
            ln += 1
            code[ln] = f"run_engine(**params)"
            ln += 1
            gen_external_py_script(self.tomo_recon_external_command_name, code)
            sig = os.system(f"python {self.tomo_recon_external_command_name}")
            if sig == 0:
                boxes = ["TrialCenPrev box"]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                self.hs["TrialCenPrev sldr"].value = 0
                self.hs["TrialCenPrev sldr"].min = 0
                self.hs["TrialCenPrev sldr"].max = int(2 * self.tomo_cen_win_w - 1)
                data_state, viewer_state = fiji_viewer_state(
                    self.global_h, self, viewer_name="tomo_cen_review_viewer"
                )
                if (not data_state) | (not viewer_state):
                    fiji_viewer_on(
                        self.global_h, self, viewer_name="tomo_cen_review_viewer"
                    )
                self.global_h.ij.py.run_macro(
                    """run("Enhance Contrast", "saturated=0.35")"""
                )
            else:
                print("Something runs wrong during the reconstruction.")
        elif self.tomo_recon_type == "Vol Recon":
            if self.tomo_use_read_config and (self.tomo_cen_list_file is not None):
                tem = self.read_config()
                if tem is not None:
                    for key in tem.keys():
                        recon_param_dict = tem[key]
                        self.set_rec_params_from_rec_dict(recon_param_dict)
                        self.set_widgets_from_rec_params(recon_param_dict)
                        self.hs["UseConfig chbx"].value = True
                        self.hs["UseConfig chbx"].disabled = True
                        self.set_rec_params_from_widgets()
                        self.set_rec_dict_from_rec_params()

                        code = {}
                        ln = 0
                        code[ln] = f"from collections import OrderedDict"
                        ln += 1
                        code[
                            ln
                        ] = f"from TXM_Sandbox.utils.tomo_recon_tools import run_engine"
                        ln += 1
                        code[
                            ln
                        ] = f"from TXM_Sandbox.utils.io import data_reader, tomo_h5_reader, data_info, tomo_h5_info"
                        ln += 1
                        code[ln] = f"params = {self.tomo_recon_param_dict}"
                        ln += 1
                        code[
                            ln
                        ] = f"params['file_params']['reader'] = data_reader(tomo_h5_reader)"
                        ln += 1
                        code[
                            ln
                        ] = f"params['file_params']['info_reader'] = data_info(tomo_h5_info)"
                        ln += 1
                        code[ln] = f"run_engine(**params)"
                        ln += 1
                        gen_external_py_script(
                            self.tomo_recon_external_command_name, code
                        )
                        sig = os.system(
                            f"python {self.tomo_recon_external_command_name}"
                        )

                        if sig == 0:
                            print(f"Reconstruction of {key} is done.")
                        elif sig == -1:
                            print("Something runs wrong during the reconstruction.")
                            return
                else:
                    print("Fail to read the configuration file.")
                    return
            else:
                print('A configuration file is needed for "Vol Recon".')
