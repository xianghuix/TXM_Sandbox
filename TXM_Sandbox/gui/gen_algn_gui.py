#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:56:17 2020

@author: xiao
"""
import os, h5py, json, time, gc, numpy as np, shutil

from ipywidgets import widgets
from pathlib import Path

import skimage.morphology as skm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import fourier_shift
import napari

from ..utils import xanes_regtools as xr
from ..utils.io import (data_reader, tif_reader, tif_seq_reader, h5_reader,
                        data_writer, tif_writer, tif_seq_writer, h5_writer)
from .gui_components import (SelectFilesButton, NumpyArrayEncoder, get_handles,
                             enable_disable_boxes, fiji_viewer_state, restart,
                             fiji_viewer_on, fiji_viewer_off,
                             gen_external_py_script, update_json_content)

napari.gui_qt()


class gen_algn_gui():

    def __init__(self, parent_h, form_sz=[650, 740]):
        self.gui_name = 'misc'
        self.form_sz = form_sz
        self.global_h = parent_h
        self.hs = {}

        self.reader = None
        self.misc_algn_ext_cmd_fn = os.path.join(
            os.path.abspath(os.path.curdir), 'misc_align_external_command.py')

        self.misc_io_cfg_type = 'manual'
        self.misc_io_in_ftype = '2Ds tif'  # in ['2D tif', '2Ds tif', '3D tif', '2D h5', '3D h5']
        self.misc_io_tgt_ftype = self.misc_io_in_ftype
        self.misc_io_tgt_fn = None
        self.misc_io_tgt_fn_is_temp = False
        self.misc_io_tgt_fn_id_s = None
        self.misc_io_tgt_fn_id_e = None
        self.misc_io_tgt_flat_fn = None
        self.misc_io_tgt_dark_fn = None
        self.misc_io_tgt_is_raw = False
        self.misc_io_tgt_3D_tif_flat_id = 0
        self.misc_io_tgt_3D_tif_dark_id = -1
        self.misc_io_tgt_3D_slcn_dim = 0
        self.misc_io_tgt_3D_fxd_sli = None
        self.misc_io_tgt_h5_data_path = 'img'
        self.misc_io_tgt_h5_flat_path = 'img_bkg'
        self.misc_io_tgt_h5_dark_path = 'img_dark'

        self.misc_io_src_ftype = self.misc_io_in_ftype
        self.misc_io_src_fn = None
        self.misc_io_src_fn_temp = None
        self.misc_io_src_fn_is_temp = True
        self.misc_io_src_is_raw = False
        self.misc_io_src_fn_id_s = None
        self.misc_io_src_fn_id_e = None
        self.misc_io_src_sli_id_s = None
        self.misc_io_src_sli_id_e = None
        self.misc_io_src_flat_fn = None
        self.misc_io_src_dark_fn = None
        self.misc_io_src_3D_tif_flat_id = 0
        self.misc_io_src_3D_tif_dark_id = -1
        self.misc_io_src_3D_slcn_dim = 0
        self.misc_io_src_3D_srch_half_wz = 10
        self.misc_io_src_h5_data_path = 'img'
        self.misc_io_src_h5_flat_path = 'img_bkg'
        self.misc_io_src_h5_dark_path = 'img_dark'

        self.misc_io_cfg = None
        self.misc_io_cfg_fn = None
        self.misc_io_out_dtype = self.misc_io_in_ftype  # set according to self.misc_io__dtype
        self.misc_io_algn_dtype = 'h5'
        self.misc_io_sav_algn_fn = None
        self.misc_io_reader_cfg = {}
        self.misc_io_writer_cfg = {}
        self.misc_io_man_src_fn_set = False
        self.misc_io_man_tgt_fn_set = False
        self.misc_io_man_sav_fn_set = False
        self.misc_io_cfg_fn_set = False
        self.misc_io_cfg_set = False

        self.misc_dat_scla = 1
        self.misc_dat_use_smth_flat = False
        self.misc_dat_smth_flat_sig = 0
        self.misc_dat_use_alt_flat = False
        self.misc_dat_set = False

        self.misc_sav_trl_reg_cfg_fn = None
        self.misc_rev_best_mtch_fn = None

        self.misc_roi_xoff = None
        self.misc_roi_yoff = None
        self.misc_roi_zoff = None
        self.misc_roi_x = None
        self.misc_roi_y = None
        self.misc_roi_z = None
        self.misc_roi_set = False

        self.misc_reg_use_chnk = True
        self.misc_reg_anch_idx = 0
        self.misc_reg_roi = [0, 10, 0, 10]
        self.misc_reg_use_mask = False
        self.misc_reg_mask = None
        self.misc_reg_mask_dltn_wd = 0
        self.misc_reg_mask_thres = 0
        self.misc_reg_use_smth_img = False
        self.misc_reg_smth_img_sig = 5
        self.misc_reg_chnk_sz = None
        self.misc_reg_mthd = None
        self.misc_reg_ref_mode = None
        self.misc_reg_mrtv_lev = 4
        self.misc_reg_mrtv_wd = 10
        self.misc_reg_mrtv_subpxl_ker = 0.2
        self.misc_reg_parm_set = False
        self.misc_reg_done = False

        self.misc_rev_algn_img = None
        self.misc_rev_fxd_img = None
        self.misc_rev_bad_shft = False
        self.misc_rev_man_xshft = 0
        self.misc_rev_man_yshft = 0
        self.misc_rev_shft_dict = {}
        self.misc_rev_done = False
        self.misc_algn_done = False

    def build_gui(self):
        base_wz_os = 92
        ex_ws_os = 6

        ## define main form
        layout = {
            'border': '3px solid #FFCC00',
            'width': f'{self.form_sz[1] - 86}px',
            'height': f'{self.form_sz[0] - 136}px'
        }
        self.hs['GenImgAlign form'] = widgets.VBox(layout=layout)

        ## ## define accordion fields
        self.hs['GenAlgnImg acc'] = widgets.Accordion(
            titles=('Config Input/Output Files', ),
            layout={
                'width': 'auto',
                'border': '3px solid #8855AA',
                'align-content': 'center',
                'align-items': 'center',
                'justify-content': 'center'
            })

        ## ## ## define widgets in self.hs['GenAlgnImg acc']
        self.hs['AlgnCfgFns box'] = widgets.VBox(
            layout={
                'width': 'auto',
                'height': f'{0.7*(self.form_sz[0] - 136)}px'
            })

        self.hs['AlgnCfgFns acc'] = widgets.Accordion(
            titles=('Config Input/Output Files', ),
            layout={
                'width': 'auto',
                'align-content': 'center',
                'align-items': 'center',
                'justify-content': 'center'
            })

        self.hs['AlgnCfgFns box'].children = [self.hs['AlgnCfgFns acc']]

        ## ## ## ## define widgets in self.hs['AlgnCfgFns acc']
        self.hs['AlgnCfgManDatCfg box'] = widgets.GridBox(
            layout={
                'border': '3px solid #FFCC00',
                'width': 'auto',
                'height': f'{0.56*(self.form_sz[0] - 136)}px',
                'grid_template_columns': 'auto',
                'grid_template_rows': 'auto auto auto auto auto auto auto auto'
            })

        ## ## ## ## ## input title box
        self.hs['AlgnCfgInDatTtl box'] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0] - 136)}px',
                'grid_template_columns': 'auto',
                'grid_template_rows': 'auto'
            })
        self.hs['AlgnCfgInDatTtl lbl'] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + 'Config Input' + '</span>')
        self.hs['AlgnCfgInDatTtl lbl'].layout = {
            'background-color': 'white',
            'color': 'cyan',
            'width': 'auto',
            'height': 'auto'
        }
        self.hs['AlgnCfgInDatTtl box'].children = [
            self.hs['AlgnCfgInDatTtl lbl']
        ]

        ## ## ## ## ## select input data type
        self.hs['AlgnCfgInDatTyp box'] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0] - 136)}px',
                'grid_template_columns': 'auto',
                'grid_template_rows': 'auto'
            })

        self.hs['AlgnCfgInDatTyp Btn'] = widgets.RadioButtons(
            options=['2D tif', '2Ds tif', '3D tif', '2D h5', '3D h5'],
            value='2Ds tif',
            layout={
                'width': 'auto',
                'height': 'auto'
            },
            orientation='horizontal',
            description='Data Type')
        self.hs['AlgnCfgInDatTyp Btn'].observe(self.AlgnCfgInDatTyp_radbtn_chg,
                                               names='value')
        self.hs['AlgnCfgInDatTyp box'].children = [
            self.hs['AlgnCfgInDatTyp Btn']
        ]

        ## ## ## ## ## select source image directory
        self.hs['AlgnCfgInSrcImg box'] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0] - 136)}px',
                'grid_template_columns': '80% auto',
                'grid_template_rows': 'auto',
                'grid_gap': '2px 2px'
            })

        self.hs['AlgnCfgInSrcPath txt'] = widgets.Text(
            value='Choose Source Image Directory ...',
            description='',
            disabled=True,
            layout={
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnCfgInSrcPath btn'] = SelectFilesButton(
            option='askopenfilename',
            text_h=self.hs['AlgnCfgInSrcPath txt'],
            **{'open_filetypes': (('tif files', ['*.tif', '*.tiff']), )})
        self.hs[
            'AlgnCfgInSrcPath btn'].description = 'Choose Source Image Directory'
        self.hs['AlgnCfgInSrcPath btn'].on_click(self.AlgnCfgInSrcPath_btn_clk)

        self.hs['AlgnCfgInSrcImg box'].children = [
            self.hs['AlgnCfgInSrcPath txt'], self.hs['AlgnCfgInSrcPath btn']
        ]

        ## ## ## ## ## select target image directory
        self.hs['AlgnCfgInTgtImg box'] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0] - 136)}px',
                'grid_template_columns': '80% auto',
                'grid_template_rows': 'auto',
                'grid_gap': '2px 2px'
            })
        self.hs['AlgnCfgInTgtPath txt'] = widgets.Text(
            value='Choose Target Image Directory ...',
            description='',
            disabled=True,
            layout={
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnCfgInTgtPath btn'] = SelectFilesButton(
            option='askopenfilename',
            text_h=self.hs['AlgnCfgInTgtPath txt'],
            **{'open_filetypes': (('tif files', ['*.tif', '*.tiff']), )})
        self.hs[
            'AlgnCfgInTgtPath btn'].description = 'Choose Target File Directory'
        self.hs['AlgnCfgInTgtPath btn'].on_click(self.AlgnCfgInTgtPath_btn_clk)

        self.hs['AlgnCfgInTgtImg box'].children = [
            self.hs['AlgnCfgInTgtPath txt'], self.hs['AlgnCfgInTgtPath btn']
        ]

        ## ## ## ## ## output title box
        self.hs['AlgnCfgOutDatTtl box'] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0] - 136)}px',
                'grid_template_columns': 'auto',
                'grid_template_rows': 'auto'
            })
        self.hs['AlgnCfgOutDatTtl lbl'] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + 'Config Output' + '</span>')
        self.hs['AlgnCfgOutDatTtl lbl'].layout = {
            'background-color': 'white',
            'color': 'cyan',
            'width': 'auto',
            'height': 'auto'
        }
        self.hs['AlgnCfgOutDatTtl box'].children = [
            self.hs['AlgnCfgOutDatTtl lbl']
        ]

        ## ## ## ## ## output data type
        self.hs['AlgnCfgOutDatTyp box'] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0] - 136)}px',
                'grid_template_columns': 'auto',
                'grid_template_rows': 'auto'
            })

        self.hs['AlgnCfgOutDatTyp Btn'] = widgets.RadioButtons(
            options=['2Ds tif', '3D tif', '3D h5'],
            value='2Ds tif',
            layout={
                'width': 'auto',
                'height': 'auto'
            },
            orientation='horizontal',
            description='Data Type')
        self.hs['AlgnCfgOutDatTyp Btn'].observe(
            self.AlgnCfgOutDatTyp_radbtn_chg, names='value')
        self.hs['AlgnCfgOutDatTyp box'].children = [
            self.hs['AlgnCfgOutDatTyp Btn']
        ]

        ## ## ## ## ## select saved image directory
        self.hs['AlgnCfgOutFn box'] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0] - 136)}px',
                'grid_template_columns': '80% auto',
                'grid_template_rows': 'auto',
                'grid_gap': '2px 2px'
            })
        self.hs['AlgnCfgOutPath txt'] = widgets.Text(
            value='Choose Saving Image Directory ...',
            description='',
            disabled=True,
            layout={
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnCfgOutPath btn'] = SelectFilesButton(
            option='asksaveasfilename',
            text_h=self.hs['AlgnCfgOutPath txt'],
            **{'filetypes': (('tif files', ['*.tif', '*.tiff']), )},
            **{'open_filetypes': (('tif files', ['*.tif', '*.tiff']), )})
        self.hs[
            'AlgnCfgOutPath btn'].description = 'Choose Saving Image Directory'
        self.hs['AlgnCfgOutPath btn'].on_click(self.AlgnCfgOutPath_btn_clk)

        self.hs['AlgnCfgOutFn box'].children = [
            self.hs['AlgnCfgOutPath txt'], self.hs['AlgnCfgOutPath btn']
        ]

        self.hs['AlgnCfgCfmManFile&Path box'] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0] - 136)}px',
                'grid_template_columns': '80% auto',
                'grid_template_rows': 'auto',
                'grid_gap': '2px 2px'
            })

        self.hs['AlgnCfgCfmManFile&Path txt'] = widgets.Text(
            value='Confirm after Files/Directories are Chosen ...',
            description='',
            disabled=True,
            layout={
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnCfgCfmManFile&Path btn'] = widgets.Button(
            description='Confirm',
            tooltip='Confirm: Confirm after you finish file configuration')
        self.hs['AlgnCfgCfmManFile&Path btn'].style.button_color = 'darkviolet'
        self.hs['AlgnCfgCfmManFile&Path btn'].on_click(
            self.AlgnCfgCfmManFilePath_btn_clk)

        self.hs['AlgnCfgCfmManFile&Path box'].children = [
            self.hs['AlgnCfgCfmManFile&Path txt'],
            self.hs['AlgnCfgCfmManFile&Path btn']
        ]

        self.hs['AlgnCfgManDatCfg box'].children = [
            self.hs['AlgnCfgInDatTtl box'], self.hs['AlgnCfgInDatTyp box'],
            self.hs['AlgnCfgInTgtImg box'], self.hs['AlgnCfgInSrcImg box'],
            self.hs['AlgnCfgOutDatTtl box'], self.hs['AlgnCfgOutDatTyp box'],
            self.hs['AlgnCfgOutFn box'], self.hs['AlgnCfgCfmManFile&Path box']
        ]

        ## ## ## ## define widgets in self.hs['AlgnCfgRdDatCfg box']
        self.hs['AlgnCfgRdDatCfg box'] = widgets.GridBox(
            layout={
                'border': '3px solid #FFCC00',
                'width': 'auto',
                'height': f'{0.16*(self.form_sz[0]-136)}px',
                'grid_template_columns': 'auto',
                'grid_template_rows': 'auto auto',
                'grid_gap': '2px 2px'
            })

        self.hs['AlgnCfgCfgFn box'] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0] - 136)}px',
                'grid_template_columns': '80% auto',
                'grid_template_rows': 'auto',
                'grid_gap': '2px 2px'
            })
        self.hs['AlgnCfgCfgFn txt'] = widgets.Text(
            value='Specify Config File ...',
            description='',
            disabled=False,
            layout={
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnCfgCfgFn btn'] = SelectFilesButton(
            option='askopenfilename',
            text_h=self.hs['AlgnCfgCfgFn txt'],
            **{'open_filetypes': (('json file', '*.json'), )})
        self.hs['AlgnCfgCfgFn btn'].description = 'Specify Config File'
        self.hs['AlgnCfgCfgFn btn'].on_click(self.AlgnCfgCfgFn_btn_clk)

        self.hs['AlgnCfgCfgFn box'].children = [
            self.hs['AlgnCfgCfgFn txt'], self.hs['AlgnCfgCfgFn btn']
        ]

        self.hs['AlgnCfgCfmCfgFn box'] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0]-136)}px',
                'grid_template_columns': '80% auto',
                'grid_template_rows': 'auto',
                'grid_gap': '2px 2px'
            })

        self.hs['AlgnCfgCfmCfgFn txt'] = widgets.Text(
            value='Confirm to Proceed ...',
            description='',
            disabled=True,
            layout={
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnCfgCfmCfgFn btn'] = widgets.Button(
            description='Confirm', tooltip='Confirm: Confirm to Proceed')
        self.hs['AlgnCfgCfmCfgFn btn'].style.button_color = 'darkviolet'
        self.hs['AlgnCfgCfmCfgFn btn'].on_click(self.AlgnCfgCfmCfgFn_btn_clk)

        self.hs['AlgnCfgCfmCfgFn box'].children = [
            self.hs['AlgnCfgCfmCfgFn txt'], self.hs['AlgnCfgCfmCfgFn btn']
        ]
        self.hs['AlgnCfgRdDatCfg box'].children = [
            self.hs['AlgnCfgCfgFn box'], self.hs['AlgnCfgCfmCfgFn box']
        ]

        ## ## ## ## define widgets in self.hs['AlgnCfgFns acc']
        self.hs['AlgnCfgFnHlp txt'] = widgets.Textarea(
            value="'Manually Config Input/Output Files' option is" +
            " useful to stitch two sets of images." +
            " 'Read Config File' option is useful to stitch" +
            " more than two image sets. You can use" +
            " 'Manually Config Input/Output Files' option to" +
            " generate a configuration file that can be used as a" +
            " template for stitching more than two image sets.",
            description='Info:',
            disabled=True,
            layout={
                'border': '3px solid #FFCC00',
                'width': 'auto',
                'height': f'{0.35*(self.form_sz[0] - 136)}px'
            })

        self.hs['AlgnCfgFns acc'].children = [
            self.hs['AlgnCfgManDatCfg box'], self.hs['AlgnCfgRdDatCfg box'],
            self.hs['AlgnCfgFnHlp txt']
        ]

        # per https://github.com/jupyter-widgets/ipywidgets/issues/2790, accordion title can only be set
        # via set_title in jupyterlab 7.5. It should be set directly when accordion is created after jupyterlab 8.
        self.hs['AlgnCfgFns acc'].set_title(
            0, 'Manually Config Input/Output Files')
        self.hs['AlgnCfgFns acc'].set_title(1, 'Read Config File')
        self.hs['AlgnCfgFns acc'].set_title(2, 'Help Information')
        self.hs['AlgnCfgFns acc'].selected_index = 2

        ## ## ## ## ## config image ROI
        ## ## ## ## ## ## Image ROI Box
        self.hs['AlgnROI box'] = widgets.VBox(
            layout={
                'width': 'auto',
                'height': f'{0.49 * (self.form_sz[0] - 136)}px'
            })

        ## ## ## ## ## ## Define Predefined Offset between Source and Target Images
        self.hs['AlgnROIPreOShftTtl lbl'] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + 'Preset Offset' + '</span>',
            layout={
                'background-color': 'white',
                'color': 'cyan',
                'height': f'{0.07 * (self.form_sz[0] - 136)}px',
                'width': 'auto',
                'flex_shrink': 1,
                'align-content': 'center',
                'align-items': 'center',
                'justify-content': 'center'
            })
        self.hs['AlgnROIPreShft box'] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': f'{0.07 * (self.form_sz[0] - 136)}px',
                'grid_template_columns': '30% 30% 30%',
                'grid_template_rows': 'auto',
                'grid_gap': '2px 2px'
            })
        self.hs['AlgnROIPreAx0Shft txt'] = widgets.IntText(
            value=0,
            min=-10,
            max=10,
            description='Axis 0 offset',
            disabled=True,
            layout={
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnROIPreAx1Shft txt'] = widgets.IntText(
            value=0,
            min=-10,
            max=10,
            description='Axis 1 offset',
            disabled=True,
            layout={
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnROIPreAx2Shft txt'] = widgets.IntText(
            value=0,
            min=-10,
            max=10,
            description='Axis 2 offset',
            disabled=True,
            layout={
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnROIPreAx0Shft txt'].observe(
            self.AlgnROIPreAx0Shft_txt_chg, names='value')
        self.hs['AlgnROIPreAx1Shft txt'].observe(
            self.AlgnROIPreAx1Shft_txt_chg, names='value')
        self.hs['AlgnROIPreAx2Shft txt'].observe(
            self.AlgnROIPreAx2Shft_txt_chg, names='value')
        self.hs['AlgnROIPreShft box'].children = [
            self.hs['AlgnROIPreAx0Shft txt'], self.hs['AlgnROIPreAx1Shft txt'],
            self.hs['AlgnROIPreAx2Shft txt']
        ]

        ## ## ## ## ## ## Define ROI in Source Image
        self.hs['AlgnROISrcROI box'] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': f'{0.28 * (self.form_sz[0] - 136)}px',
                'grid_template_columns': 'auto',
                'grid_template_rows': 'auto auto auto',
                'grid_gap': '2px 2px'
            })

        self.hs['AlgnROISrcROI lbl'] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + 'Define ROI in Source Image' + '</span>',
            layout={
                'background-color': 'white',
                'color': 'cyan',
                'height': f'{0.07 * (self.form_sz[0] - 136)}px',
                'width': 'auto',
                'flex_shrink': 1,
                'align-content': 'center',
                'align-items': 'center',
                'justify-content': 'center'
            })

        self.hs['AlgnROISrcROIAx0 sldr'] = widgets.IntRangeSlider(
            value=[10, 40],
            min=1,
            max=50,
            step=1,
            description='Axis 0 range:',
            disabled=True,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout={
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnROISrcROIAx1 sldr'] = widgets.IntRangeSlider(
            value=[10, 40],
            min=1,
            max=50,
            step=1,
            description='Axis 1 range:',
            disabled=True,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout={
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnROISrcROIAx2 sldr'] = widgets.IntRangeSlider(
            value=[10, 40],
            min=1,
            max=50,
            step=1,
            description='Axis 2 range:',
            disabled=True,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout={
                'width': 'auto',
                'height': 'auto'
            })

        self.hs['AlgnROISrcROIAx0 sldr'].observe(
            self.AlgnROISrcROIAx0_sldr_chg, names='value')
        self.hs['AlgnROISrcROIAx1 sldr'].observe(
            self.AlgnROISrcROIAx1_sldr_chg, names='value')
        self.hs['AlgnROISrcROIAx2 sldr'].observe(
            self.AlgnROISrcROIAx2_sldr_chg, names='value')

        self.hs['AlgnROISrcROI box'].children = [
            self.hs['AlgnROISrcROI lbl'], self.hs['AlgnROISrcROIAx0 sldr'],
            self.hs['AlgnROISrcROIAx1 sldr'], self.hs['AlgnROISrcROIAx2 sldr']
        ]

        self.hs['AlgnROICfm box'] = widgets.GridBox(
            layout={
                'height': f'{0.07 * (self.form_sz[0] - 136)}px',
                'width': 'auto',
                'grid_template_columns': '80% auto'
            })

        self.hs['AlgnROICfm txt'] = widgets.Text(value='Confirm ROI Config',
                                                 description='',
                                                 disabled=True,
                                                 layout={
                                                     'width': 'auto',
                                                     'height': 'auto'
                                                 })
        self.hs['AlgnROICfm btn'] = widgets.Button(
            description='Confirm',
            disabled=True,
            tooltip='Confirm: Confirm ROI Config')
        self.hs['AlgnROICfm btn'].style.button_color = 'darkviolet'
        self.hs['AlgnROICfm btn'].on_click(self.AlgnROICfm_btn_clk)

        self.hs['AlgnROICfm box'].children = [
            self.hs['AlgnROICfm txt'], self.hs['AlgnROICfm btn']
        ]

        self.hs['AlgnROI box'].children = [
            self.hs['AlgnROIPreOShftTtl lbl'], self.hs['AlgnROIPreShft box'],
            self.hs['AlgnROISrcROI box'], self.hs['AlgnROICfm box']
        ]

        ## ## ## ## config registration algorithm -- start
        layout = {'width': 'auto', 'height': f'{0.35*(self.form_sz[0]-136)}px'}
        self.hs['AlgnRegPar box'] = widgets.VBox()
        self.hs['AlgnRegPar box'].layout = layout

        ## ## ## ## ## fiji&anchor box
        self.hs['AlgnRegFiji&Anch box'] = widgets.GridBox(
            layout={
                'border': '3px solid #FFCC00',
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0]-136)}px',
                'grid_template_columns': 'auto auto auto auto',
                'grid_template_rows': 'auto'
            })

        self.hs['AlgnRegFijiMskVwr chbx'] = widgets.Checkbox(
            value=False,
            disabled=True,
            description='preview',
            indent=False,
            layout={
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnRegChnkSz chbx'] = widgets.Checkbox(
            value=True,
            disabled=True,
            description='use chunk',
            indent=False,
            layout={
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnRegChnkSz sldr'] = widgets.IntSlider(
            value=7,
            disabled=True,
            description='chunk size',
            layout={
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnRegSliSrch sldr'] = widgets.IntSlider(
            value=10,
            disabled=True,
            description='z search half width',
            layout={
                'width': 'auto',
                'height': 'auto'
            })

        self.hs['AlgnRegFijiMskVwr chbx'].observe(
            self.AlgnRegFijiMaskViewer_chbx_chg, names='value')
        self.hs['AlgnRegChnkSz chbx'].observe(self.AlgnRegChnkSz_chbx_chg,
                                              names='value')
        self.hs['AlgnRegChnkSz sldr'].observe(self.AlgnRegChnkSz_sldr_chg,
                                              names='value')
        self.hs['AlgnRegSliSrch sldr'].observe(self.AlgnRegSliSrch_sldr_chg,
                                               names='value')
        self.hs['AlgnRegFiji&Anch box'].children = [
            self.hs['AlgnRegFijiMskVwr chbx'], self.hs['AlgnRegChnkSz chbx'],
            self.hs['AlgnRegChnkSz sldr'], self.hs['AlgnRegSliSrch sldr']
        ]

        ## ## ## ## ## mask options box
        self.hs['AlgnRegMskOptn box'] = widgets.GridBox(
            layout={
                'border': '3px solid #FFCC00',
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0]-136)}px',
                'grid_template_columns': '20% 40% 40%',
                'grid_template_rows': 'auto'
            })
        self.hs['AlgnRegUsMsk chbx'] = widgets.Checkbox(value=False,
                                                        disabled=True,
                                                        description='use mask',
                                                        display='flex',
                                                        indent=False,
                                                        layout={
                                                            'width': 'auto',
                                                            'height': 'auto'
                                                        })
        self.hs['AlgnRegMskThrs sldr'] = widgets.FloatSlider(
            value=False,
            disabled=True,
            description='mask thres',
            readout_format='.5f',
            min=-1.,
            max=10.,
            step=1e-5,
            layout={
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnRegMskDltn sldr'] = widgets.IntSlider(
            value=False,
            disabled=True,
            description='mask dilation',
            min=0,
            max=30,
            step=1,
            layout={
                'width': 'auto',
                'height': 'auto'
            })

        self.hs['AlgnRegUsMsk chbx'].observe(self.AlgnRegUsMsk_chbx_chg,
                                             names='value')
        self.hs['AlgnRegMskThrs sldr'].observe(self.AlgnRegMskThrs_sldr_chg,
                                               names='value')
        self.hs['AlgnRegMskDltn sldr'].observe(self.AlgnRegMskDltn_sldr_chg,
                                               names='value')
        self.hs['AlgnRegMskOptn box'].children = [
            self.hs['AlgnRegUsMsk chbx'], self.hs['AlgnRegMskThrs sldr'],
            self.hs['AlgnRegMskDltn sldr']
        ]

        ## ## ## ## ## sli_search & chunk_size box
        self.hs['AlgnRegRegOptn box'] = widgets.GridBox(
            layout={
                'border': '3px solid #FFCC00',
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0]-136)}px',
                'grid_template_columns': 'auto auto',
                'grid_template_rows': 'auto'
            })

        self.hs['AlgnRegRegMthd drpdn'] = widgets.Dropdown(
            value='MRTV',
            description='reg method',
            disabled=True,
            options=['MRTV', 'MPC', 'PC', 'SR', 'LS_MRTV', 'MPC_MRTV'],
            description_tooltip=
            'reg method: MPC: Masked Phase Correlation; PC: Phase Correlation; SR: StackReg',
            layout={
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnRegRefMod drpdn'] = widgets.Dropdown(
            value='single',
            options=['single', 'neighbor', 'average'],
            description='ref mode',
            disabled=True,
            description_tooltip=
            'ref mode: single: two images with fixed id gap in two consecutive chunks are used for the registration between two chunks; neighbor: two neighbor images in two consecutive chunks are used for the registration between two chunks; average: deprecated',
            layout={
                'width': 'auto',
                'height': 'auto'
            })

        self.hs['AlgnRegRegMthd drpdn'].observe(self.AlgnRegRegMthd_drpdn_chg,
                                                names='value')
        self.hs['AlgnRegRefMod drpdn'].observe(self.AlgnRegRefMod_drpdn_chg,
                                               names='value')
        self.hs['AlgnRegRegOptn box'].children = [
            self.hs['AlgnRegRegMthd drpdn'], self.hs['AlgnRegRefMod drpdn']
        ]

        ## ## ## ## ##  reg_options box
        self.hs['AlgnRegMRTVOptn box'] = widgets.GridBox(
            layout={
                'border': '3px solid #FFCC00',
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0]-136)}px',
                'grid_template_columns': 'auto auto auto auto auto',
                'grid_template_rows': 'auto'
            })

        self.hs['AlgnRegMRTVLv txt'] = widgets.BoundedIntText(
            value=5,
            min=1,
            max=10,
            step=1,
            description='level',
            description_tooltip='level: multi-resolution level',
            disabled=True,
            layout={
                'width': 'auto',
                'height': 'auto'
            })

        self.hs['AlgnRegMRTVWz txt'] = widgets.BoundedIntText(
            value=10,
            min=1,
            max=20,
            step=1,
            description='width',
            description_tooltip=
            'width: multi-resolution searching width at each level (number of searching steps)',
            disabled=True,
            layout={
                'width': 'auto',
                'height': 'auto'
            })

        self.hs['AlgnRegMRTVSubpxlSrch drpdn'] = widgets.Dropdown(
            value='analytical',
            disabled=True,
            options=['analytical', 'fitting'],
            description='subpxl srch',
            description_tooltip='subpxl srch: subpixel TV minization option',
            layout={
                'width': 'auto',
                'height': 'auto'
            })

        self.hs['AlgnRegMRTVSubpxlWz txt'] = widgets.BoundedIntText(
            value=3,
            min=2,
            max=20,
            step=0.1,
            description='subpxl wz',
            disabled=True,
            description_tooltip='subpxl wz: final sub-pixel fitting points',
            layout={
                'width': 'auto',
                'height': 'auto'
            })

        self.hs['AlgnRegMRTVSubpxlKrn txt'] = widgets.BoundedIntText(
            value=3,
            min=2,
            max=20,
            step=1,
            description='kernel wz',
            disabled=True,
            description_tooltip=
            'kernel wz: Gaussian blurring width before TV minimization',
            layout={
                'width': 'auto',
                'height': 'auto'
            })

        self.hs['AlgnRegMRTVLv txt'].observe(self.AlgnRegMRTVLv_text_chg,
                                             names='value')
        self.hs['AlgnRegMRTVWz txt'].observe(self.AlgnRegMRTVWz_txt_chg,
                                             names='value')
        self.hs['AlgnRegMRTVSubpxlWz txt'].observe(
            self.AlgnRegMRTVSubpxlWz_txt_chg, names='value')
        self.hs['AlgnRegMRTVSubpxlKrn txt'].observe(
            self.AlgnRegMRTVSubpxlKrn_txt_chg, names='value')
        self.hs['AlgnRegMRTVSubpxlSrch drpdn'].observe(
            self.AlgnRegMRTVSubpxlSrch_drpdn_chg, names='value')
        self.hs['AlgnRegMRTVOptn box'].children = [
            self.hs['AlgnRegMRTVLv txt'], self.hs['AlgnRegMRTVWz txt'],
            self.hs['AlgnRegMRTVSubpxlSrch drpdn'],
            self.hs['AlgnRegMRTVSubpxlWz txt'],
            self.hs['AlgnRegMRTVSubpxlKrn txt']
        ]

        ## ## ## ## ## run registration box
        self.hs['AlgnRegRunCfm box'] = widgets.GridBox(
            layout={
                'border': '3px solid #FFCC00',
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0]-136)}px',
                'grid_template_columns': '80% auto',
                'grid_template_rows': 'auto'
            })

        self.hs['AlgnRegRunCfm txt'] = widgets.Text(
            description='',
            disabled=True,
            value='Start Registration ...',
            layout={
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnRegRunCfm btn'] = widgets.Button(
            description='Run',
            disabled=True,
            description_tooltip='Run Registration')
        self.hs['AlgnRegRunCfm btn'].style.button_color = 'darkviolet'

        self.hs['AlgnRegRunCfm btn'].on_click(self.AlgnRegRunCfm_btn_clk)
        self.hs['AlgnRegRunCfm box'].children = [
            self.hs['AlgnRegRunCfm txt'], self.hs['AlgnRegRunCfm btn']
        ]

        self.hs['AlgnRegPar box'].children = [
            self.hs['AlgnRegFiji&Anch box'], self.hs['AlgnRegMskOptn box'],
            self.hs['AlgnRegRegOptn box'], self.hs['AlgnRegMRTVOptn box'],
            self.hs['AlgnRegRunCfm box']
        ]

        ## ## ## ## review registration result box
        self.hs['AlgnRevReg box'] = widgets.VBox(
            layout={
                'width': 'auto',
                'height': f'{0.28*(self.form_sz[0]-136)}px'
            })

        ## ## ## ## ## title the box
        self.hs['AlgnRevRegTtl box'] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0]-136)}px',
                'grid_template_columns': 'auto',
                'grid_template_rows': 'auto'
            })
        self.hs['AlgnRevRegTtl txt'] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + 'Review Registration Results' + '</span>',
            layout={
                'background-color': 'white',
                'color': 'cyan',
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnRevRegTtl box'].children = [self.hs['AlgnRevRegTtl txt']]

        ## ## ## ## ## reg pair box
        self.hs['AlgnRevRegPair box'] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0]-136)}px',
                'grid_template_columns': '80% auto',
                'grid_template_rows': 'auto',
                'grid_gap': '0px 2px'
            })
        self.hs['AlgnRevRegPair sldr'] = widgets.IntSlider(
            value=False,
            disabled=True,
            description='reg pair #',
            layout={
                'width': 'auto',
                'height': 'auto'
            })

        self.hs['AlgnRevRegPairBad btn'] = widgets.Button(
            description='Bad', description_tooltip='Bad reg', disabled=True)
        self.hs['AlgnRevRegPairBad btn'].style.button_color = 'darkviolet'

        self.hs['AlgnRevRegPair sldr'].observe(self.AlgnRevRegPair_sldr_chg,
                                               names='value')
        self.hs['AlgnRevRegPairBad btn'].on_click(
            self.AlgnRevRegPairBad_btn_clk)
        self.hs['AlgnRevRegPair box'].children = [
            self.hs['AlgnRevRegPair sldr'], self.hs['AlgnRevRegPairBad btn']
        ]

        ## ## ## ## ## manual shift box -- start
        self.hs['AlgnRevCorrShft box'] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0]-136)}px',
                'grid_template_columns': '30% 30% 30%',
                'grid_template_rows': 'auto',
                'grid_gap': '2px 2px'
            })
        self.hs['AlgnRevCorrXShft txt'] = widgets.FloatText(
            value=0,
            disabled=True,
            min=-100,
            max=100,
            step=0.5,
            description='x shift',
            indent=False,
            layout={'width': '100%'})
        self.hs['AlgnRevCorrYShft txt'] = widgets.FloatText(
            value=0,
            disabled=True,
            min=-100,
            max=100,
            step=0.5,
            description='y shift',
            indent=False,
            layout={'width': '100%'})
        self.hs['AlgnRevCorrZShft txt'] = widgets.IntText(
            value=0,
            disabled=True,
            min=1,
            max=100,
            step=1,
            description='z shift',
            indent=False,
            layout={'width': '100%'})

        self.hs['AlgnRevCorrXShft txt'].observe(self.AlgnRevCorrXShft_txt_chg,
                                                names='value')
        self.hs['AlgnRevCorrYShft txt'].observe(self.AlgnRevCorrYShft_txt_chg,
                                                names='value')
        self.hs['AlgnRevCorrZShft txt'].observe(self.AlgnRevCorrZShft_txt_chg,
                                                names='value')
        self.hs['AlgnRevCorrShft box'].children = [
            self.hs['AlgnRevCorrXShft txt'], self.hs['AlgnRevCorrYShft txt'],
            self.hs['AlgnRevCorrZShft txt']
        ]

        self.hs['AlgnRevCfmCorrShft box'] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0]-136)}px',
                'grid_template_columns': '80% auto',
                'grid_template_rows': 'auto',
                'grid_gap': '0px 2px'
            })

        self.hs['AlgnRevCorrShftRec btn'] = widgets.Button(
            disabled=True,
            description='Record',
        )
        self.hs['AlgnRevCorrShftRec btn'].style.button_color = 'darkviolet'
        self.hs['AlgnRevCorrShftRec btn'].on_click(
            self.AlgnRevCorrShftRec_btn_clk)
        self.hs['AlgnRevCfmCorrShft box'].children = [
            self.hs['AlgnRevCorrShft box'], self.hs['AlgnRevCorrShftRec btn']
        ]
        ## ## ## ## ## manual shift box -- end

        ## ## ## ## ## confirm review results box
        self.hs['AlgnRevRegCfm box'] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': f'{0.07*(self.form_sz[0]-136)}px',
                'grid_template_columns': '80% auto',
                'grid_template_rows': 'auto',
                'grid_gap': '0px 2px'
            })
        self.hs['AlgnRevRegCfm txt'] = widgets.Text(
            description='',
            disabled=True,
            value='Confirm after you finish reg review ...',
            layout={
                'width': 'auto',
                'height': 'auto'
            })
        self.hs['AlgnRevCfmRev&ProcAlgn btn'] = widgets.Button(
            description='Confirm',
            disabled=True,
            description_tooltip='Confirm after you finish reg review ...')
        self.hs['AlgnRevCfmRev&ProcAlgn btn'].style.button_color = 'darkviolet'

        self.hs['AlgnRevCfmRev&ProcAlgn btn'].on_click(
            self.AlgnRevCfmRevProcAlgn_btn_clk)
        self.hs['AlgnRevRegCfm box'].children = [
            self.hs['AlgnRevRegCfm txt'], self.hs['AlgnRevCfmRev&ProcAlgn btn']
        ]

        self.hs['AlgnRevReg box'].children = [
            self.hs['AlgnRevRegTtl box'], self.hs['AlgnRevRegPair box'],
            self.hs['AlgnRevCfmCorrShft box'], self.hs['AlgnRevRegCfm box']
        ]

        ## ## Integrate widgets
        self.hs['GenAlgnImg acc'].children = [
            self.hs['AlgnCfgFns box'], self.hs['AlgnROI box'],
            self.hs['AlgnRegPar box'], self.hs['AlgnRevReg box']
        ]
        # per https://github.com/jupyter-widgets/ipywidgets/issues/2790, accordion title can only be set
        # via set_title in jupyterlab 7.5. It should be set directly when accordion is created after jupyterlab 8.
        self.hs['GenAlgnImg acc'].set_title(0, 'Config Input/Output Files')
        self.hs['GenAlgnImg acc'].set_title(1, 'Config Image ROI')
        self.hs['GenAlgnImg acc'].set_title(2,
                                            'Config Registration Parameters')
        self.hs['GenAlgnImg acc'].set_title(
            3, 'Review Registration Result & Proceed Alignment')
        self.hs['GenAlgnImg acc'].selected_index = None

        self.hs['GenImgAlign form'].children = [self.hs['GenAlgnImg acc']]

        self.hs['AlgnCfgInTgtPath btn'].initialdir = self.global_h.cwd
        self.hs['AlgnCfgInSrcPath btn'].initialdir = self.global_h.cwd
        self.hs['AlgnCfgOutPath btn'].initialdir = self.global_h.cwd
        self.hs['AlgnCfgCfgFn btn'].initialdir = self.global_h.cwd

    def boxes_logic(self):
        if self.misc_io_cfg_fn_set:
            self.misc_io_man_tgt_fn_set = False
            self.misc_io_man_src_fn_set = False
            self.misc_io_man_sav_fn_set = False
        if not self.misc_io_man_tgt_fn_set:
            self.hs['AlgnCfgInTgtPath btn'].style.button_color = 'orange'
            self.hs['AlgnCfgInTgtPath btn'].description = 'Select File'
            self.hs[
                'SelInTgtPath text'].value = 'Choose Source Image Directory ...'
            self.misc_io_tgt_fn = None
        if not self.misc_io_man_src_fn_set:
            self.hs['AlgnCfgInSrcPath btn'].style.button_color = 'orange'
            self.hs['AlgnCfgInSrcPath btn'].description = 'Select File'
            self.hs[
                'SelInSrcPath text'].value = 'Choose Target Image Directory ...'
            self.misc_io_src_fn = None
        if not self.misc_io_man_sav_fn_set:
            self.hs['AlgnCfgOutPath btn'].style.button_color = 'orange'
            self.hs['AlgnCfgOutPath btn'].description = 'Select File'
            self.hs[
                'SelOutPath text'].value = 'Choose Saving Image Directory ...'
            self.misc_io_sav_algn_fn = None
        # if self.misc_io_cfg_set:
        #     enable_disable_boxes(self.hs, ['ConfigImgROI acc'],
        #                          disabled=False,
        #                          level=-1)  # 'ImgROI box'
        # else:
        #     enable_disable_boxes(self.hs, ['ConfigImgROI acc'],
        #                          disabled=True,
        #                          level=-1)  # 'ImgROI box'

    def read_io_cfg(self, fn):
        with open(fn, 'r') as f:
            self.misc_io_cfg = json.load(f)
        self.misc_io_cfg_type = 'cfg file'
        self.misc_io_tgt_ftype = self.misc_io_cfg['in tgt data']['ftype']
        self.misc_io_tgt_fn = self.misc_io_cfg['in tgt data']['data fn']
        self.misc_io_tgt_fn_is_temp = self.misc_io_cfg['in tgt data'][
            'is data fn temp']
        self.misc_io_tgt_is_raw = self.misc_io_cfg['in tgt data']['is raw']
        if self.misc_io_tgt_ftype == '2D tif':
            self.misc_io_tgt_flat_fn = self.misc_io_cfg['in tgt data']['tif'][
                '2D']['flat fn']
            self.misc_io_tgt_dark_fn = self.misc_io_cfg['in tgt data']['tif'][
                '2D']['dark fn']
        elif self.misc_io_tgt_ftype == '2Ds tif':
            self.misc_io_tgt_flat_fn = self.misc_io_cfg['in tgt data']['tif'][
                '2Ds']['flat fn']
            self.misc_io_tgt_dark_fn = self.misc_io_cfg['in tgt data']['tif'][
                '2Ds']['dark fn']
        elif self.misc_io_tgt_ftype == '3D tif':
            self.misc_io_tgt_flat_fn = self.misc_io_cfg['in tgt data']['tif'][
                '3D']['flat fn']
            self.misc_io_tgt_dark_fn = self.misc_io_cfg['in tgt data']['tif'][
                '3D']['dark fn']
            self.misc_io_tgt_3D_tif_flat_id = self.misc_io_cfg['in tgt data'][
                'tif']['3D']['flat sli idx']
            self.misc_io_tgt_3D_tif_dark_id = self.misc_io_cfg['in tgt data'][
                'tif']['3D']['dark sli idx']
            self.misc_io_tgt_3D_slcn_dim = self.misc_io_cfg['in tgt data'][
                'tif']['3D']['slicing dim']
            self.misc_io_tgt_3D_fxd_sli = self.misc_io_cfg['in tgt data'][
                'tif']['3D']['fixed sli']
        elif self.misc_io_tgt_ftype == '2D h5':
            self.misc_io_tgt_flat_fn = self.misc_io_cfg['in tgt data']['h5'][
                '2D']['flat fn']
            self.misc_io_tgt_dark_fn = self.misc_io_cfg['in tgt data']['h5'][
                '2D']['dark fn']
            self.misc_io_tgt_h5_data_path = self.misc_io_cfg['in tgt data'][
                'h5']['2D']['data path']
            self.misc_io_tgt_h5_flat_path = self.misc_io_cfg['in tgt data'][
                'h5']['2D']['flat path']
            self.misc_io_tgt_h5_dark_path = self.misc_io_cfg['in tgt data'][
                'h5']['2D']['dark path']
        elif self.misc_io_tgt_ftype == '3D h5':
            self.misc_io_tgt_flat_fn = self.misc_io_cfg['in tgt data']['h5'][
                '3D']['flat fn']
            self.misc_io_tgt_dark_fn = self.misc_io_cfg['in tgt data']['h5'][
                '3D']['dark fn']
            self.misc_io_tgt_h5_data_path = self.misc_io_cfg['in tgt data'][
                'h5']['3D']['data path']
            self.misc_io_tgt_h5_flat_path = self.misc_io_cfg['in tgt data'][
                'h5']['3D']['flat path']
            self.misc_io_tgt_h5_dark_path = self.misc_io_cfg['in tgt data'][
                'h5']['3D']['dark path']
            self.misc_io_tgt_3D_slcn_dim = self.misc_io_cfg['in tgt data'][
                'h5']['3D']['slicing dim']
            self.misc_io_tgt_3D_fxd_sli = self.misc_io_cfg['in tgt data'][
                'h5']['3D']['fixed sli']

        self.misc_io_src_ftype = self.misc_io_cfg['in src data']['ftype']
        self.misc_io_src_fn_temp = self.misc_io_cfg['in src data']['data fn']
        self.misc_io_src_fn_is_temp = self.misc_io_cfg['in src data'][
            'is data fn temp']
        self.misc_io_src_is_raw = self.misc_io_cfg['in src data']['is raw']
        self.misc_io_src_fn_id_s = self.misc_io_cfg['in src data'][
            'data fn id start']
        self.misc_io_src_fn_id_e = self.misc_io_cfg['in src data'][
            'data fn id end']
        if self.misc_io_src_ftype == '2D tif':
            self.misc_io_src_flat_fn = self.misc_io_cfg['in src data']['tif'][
                '2D']['flat fn']
            self.misc_io_src_dark_fn = self.misc_io_cfg['in src data']['tif'][
                '2D']['dark fn']
        elif self.misc_io_src_ftype == '2Ds tif':
            self.misc_io_src_flat_fn = self.misc_io_cfg['in src data']['tif'][
                '2Ds']['flat fn']
            self.misc_io_src_dark_fn = self.misc_io_cfg['in src data']['tif'][
                '2Ds']['dark fn']
            self.misc_io_src_sli_id_s = self.misc_io_cfg['in src data']['tif'][
                '2Ds']['sli start']
            self.misc_io_src_sli_id_e = self.misc_io_cfg['in src data']['tif'][
                '2Ds']['sli end']
            self.misc_io_src_3D_srch_half_wz = self.misc_io_cfg['in src data'][
                'tif']['2Ds']['srch sli half width']
        elif self.misc_io_src_ftype == '3D tif':
            self.misc_io_src_flat_fn = self.misc_io_cfg['in src data']['tif'][
                '3D']['flat fn']
            self.misc_io_src_dark_fn = self.misc_io_cfg['in src data']['tif'][
                '3D']['dark fn']
            self.misc_io_src_3D_tif_flat_id = self.misc_io_cfg['in src data'][
                'tif']['3D']['flat sli idx']
            self.misc_io_src_3D_tif_dark_id = self.misc_io_cfg['in src data'][
                'tif']['3D']['dark sli idx']
            self.misc_io_src_3D_slcn_dim = self.misc_io_cfg['in src data'][
                'tif']['3D']['slicing dim']
            self.misc_io_src_3D_srch_half_wz = self.misc_io_cfg['in src data'][
                'tif']['3D']['srch sli half width']
        elif self.misc_io_src_ftype == '2D h5':
            self.misc_io_src_flat_fn = self.misc_io_cfg['in src data']['h5'][
                '2D']['flat fn']
            self.misc_io_src_dark_fn = self.misc_io_cfg['in src data']['h5'][
                '2D']['dark fn']
            self.misc_io_src_h5_data_path = self.misc_io_cfg['in src data'][
                'h5']['2D']['data path']
            self.misc_io_src_h5_flat_path = self.misc_io_cfg['in src data'][
                'h5']['2D']['flat path']
            self.misc_io_src_h5_dark_path = self.misc_io_cfg['in src data'][
                'h5']['2D']['dark path']
        elif self.misc_io_src_ftype == '3D h5':
            self.misc_io_src_flat_fn = self.misc_io_cfg['in src data']['h5'][
                '3D']['flat fn']
            self.misc_io_src_dark_fn = self.misc_io_cfg['in src data']['h5'][
                '3D']['dark fn']
            self.misc_io_src_h5_data_path = self.misc_io_cfg['in src data'][
                'h5']['3D']['data path']
            self.misc_io_src_h5_flat_path = self.misc_io_cfg['in src data'][
                'h5']['3D']['flat path']
            self.misc_io_src_h5_dark_path = self.misc_io_cfg['in src data'][
                'h5']['3D']['dark path']
            self.misc_io_src_3D_slcn_dim = self.misc_io_cfg['in src data'][
                'h5']['3D']['slicing dim']
            self.misc_io_src_3D_srch_half_wz = self.misc_io_cfg['in src data'][
                'h5']['3D']['srch sli half width']

        self.misc_io_algn_dtype = self.misc_io_cfg['out algn data']['ftype']
        self.misc_io_sav_algn_fn = self.misc_io_cfg['out algn data']['out fn']
        self.misc_io_cfg_fn = self.misc_io_cfg['cfg fn']['cfg fn']

    def preset_roi_layout(self):
        self.misc_io_src_fn_temp = None
        self.misc_io_src_is_raw = False
        self.misc_io_src_fn_id_s = None
        self.misc_io_src_fn_id_e = None
        self.misc_io_src_flat_fn = None
        self.misc_io_src_dark_fn = None
        self.misc_io_src_3D_tif_flat_id = 0
        self.misc_io_src_3D_tif_dark_id = -1
        self.misc_io_src_3D_slcn_dim = 0
        self.misc_io_src_3D_srch_half_wz = 10
        self.misc_io_src_h5_data_path = 'img'
        self.misc_io_src_h5_flat_path = 'img_bkg'
        self.misc_io_src_h5_dark_path = 'img_dark'
        if self.misc_io_src_ftype == '2Ds tif':
            # fns = glob.glob(os.path.baseame(self.misc_io_src_fn_temp.replace('{0}', '*').replace('{1}', '*')),
            #                 root_dir=os.path.dirname(self.misc_io_src_fn_temp))
            # im_id = fns[int(len(fns)/2)].split('.')[0].split('_')[-1]
            fn_id = int(
                (self.misc_io_src_fn_id_s + self.misc_io_src_fn_id_e) / 2)
            sli_id = int(
                (self.misc_io_src_sli_id_s + self.misc_io_src_sli_id_e) / 2)
            self.misc_io_src_fn = self.misc_io_src_fn_temp.format(
                fn_id, sli_id)
            self.AlgnROIPreAx0Shft_txt_chg.disabled = False
            self.AlgnROISrcROIAx0_sldr_chg.disabled = False
        else:
            fn_id = int(
                (self.misc_io_src_fn_id_s + self.misc_io_src_fn_id_e) / 2)
            self.misc_io_src_fn = self.misc_io_src_fn_temp.format(fn_id)
            if self.misc_io_src_ftype in ['3D tif', '3D h5']:
                self.AlgnROIPreAx0Shft_txt_chg.disabled = False
                self.AlgnROISrcROIAx0_sldr_chg.disabled = False
            else:
                self.AlgnROIPreAx0Shft_txt_chg.disabled = True
                self.AlgnROISrcROIAx0_sldr_chg.disabled = True

    def AlgnCfgInDatTyp_radbtn_chg(self, a):
        self.misc_io_in_ftype = a['owner'].value
        self.misc_io_tgt_ftype = self.misc_io_in_ftype
        self.misc_io_src_ftype = self.misc_io_in_ftype
        if self.misc_io_in_ftype in ['2D tif', '2D h5']:
            self.hs['AlgnCfgOutDatTyp Btn'].options = ['2D tif', '2D h5']
        else:
            self.hs['AlgnCfgOutDatTyp Btn'].options = [
                '2Ds tif', '3D tif', '3D h5'
            ]
        self.hs['AlgnCfgOutDatTyp Btn'].value = self.misc_io_in_ftype

        if 'tif' in self.misc_io_in_ftype:
            self.hs['AlgnCfgInSrcPath btn'].open_filetypes = (('tif files', [
                '*.tif', '*.tiff'
            ]), )
            self.hs['AlgnCfgInTgtPath btn'].open_filetypes = (('tif files', [
                '*.tif', '*.tiff'
            ]), )
            self.hs['AlgnCfgOutPath btn'].filetypes = (['*.tif', '*.tiff'], )
            self.hs['AlgnCfgOutPath btn'].open_filetypes = (('tif files', [
                '*.tif', '*.tiff'
            ]), )
            self.hs['AlgnCfgOutPath btn'].initialfile = 'aligned_image.tif'
        else:
            self.hs['AlgnCfgInSrcPath btn'].open_filetypes = (('h5 files', [
                '*.h5', '*.hdf', '*.hdf5'
            ]), )
            self.hs['AlgnCfgInTgtPath btn'].open_filetypes = (('h5 files', [
                '*.h5', '*.hdf', '*.hdf5'
            ]), )
            self.hs['AlgnCfgOutPath btn'].filetypes = ([
                '*.h5', '*.hdf', '*.hdf5'
            ], )
            self.hs['AlgnCfgOutPath btn'].open_filetypes = (('h5 files', [
                '*.h5', '*.hdf', '*.hdf5'
            ]), )
            self.hs['AlgnCfgOutPath btn'].initialfile = 'aligned_image.h5'

        if self.misc_io_in_ftype in ['2D tif', '3D tif']:
            self.io_img_reader = data_reader(tif_reader)
            self.misc_io_reader_cfg = {}
        elif self.misc_io_in_ftype in ['2D h5', '3D h5']:
            self.io_img_reader = data_reader(h5_reader)
            self.misc_io_reader_cfg = {'ds_path': 'img'}
        else:
            self.io_img_reader = data_reader(tifs_reader)
            self.misc_io_reader_cfg = {'scan_id': None}
        self.misc_io_man_tgt_fn_set = False
        self.misc_io_man_src_fn_set = False
        self.misc_io_cfg_fn_set = False
        self.misc_io_cfg_set = False
        self.boxes_logic()

    def AlgnCfgInTgtPath_btn_clk(self, a):
        if len(a.files[0]) != 0:
            self.misc_io_tgt_fn = a.files[0]
            self.misc_io_man_tgt_fn_set = True
            # self.hs['AlgnCfgInTgtPath btn'].initialdir = os.path.abspath(
            #     os.path.dirname(a.files[0]))
            update_json_content(
                self.global_h.GUI_cfg_file,
                {'cwd': os.path.abspath(os.path.dirname(a.files[0]))})
        else:
            self.misc_io_tgt_fn = None
            self.misc_io_man_tgt_fn_set = False
        self.misc_io_cfg_fn_set = False
        self.misc_io_cfg_set = False
        self.boxes_logic()

    def AlgnCfgInSrcPath_btn_clk(self, a):
        if len(a.files[0]) != 0:
            self.misc_io_src_fn = a.files[0]
            self.misc_io_man_src_fn_set = True
            # self.hs['AlgnCfgInSrcPath btn'].initialdir = os.path.abspath(
            #     os.path.dirname(a.files[0]))
            update_json_content(
                self.global_h.GUI_cfg_file,
                {'cwd': os.path.abspath(os.path.dirname(a.files[0]))})
        else:
            self.misc_io_src_fn = None
            self.misc_io_man_src_fn_set = False
        self.misc_io_cfg_fn_set = False
        self.misc_io_cfg_set = False
        self.boxes_logic()

    def AlgnCfgOutDatTyp_radbtn_chg(self, a):
        self.misc_io_out_dtype = self.hs['AlgnCfgOutDatTyp Btn'].value
        if 'tif' in self.misc_io_out_dtype:
            self.hs['AlgnCfgOutPath btn'].filetypes = (('tif files',
                                                        ['*.tif', '*.tiff']), )
            self.hs['AlgnCfgOutPath btn'].open_filetypes = (('tif files', [
                '*.tif', '*.tiff'
            ]), )
            self.hs['AlgnCfgOutPath btn'].initialfile = 'aligned_image.tif'
        else:
            self.hs['AlgnCfgOutPath btn'].filetypes = (('h5 files', [
                '*.h5', '*.hdf', '*.hdf5'
            ]), )
            self.hs['AlgnCfgOutPath btn'].open_filetypes = (('h5 files', [
                '*.h5', '*.hdf', '*.hdf5'
            ]), )
            self.hs['AlgnCfgOutPath btn'].initialfile = 'aligned_image.h5'

        if self.misc_io_out_dtype in ['2D tif', '3D tif']:
            self.io_sav_writer = data_writer(tif_writer)
            self.misc_io_writer_cfg = {}
        elif self.misc_io_out_dtype in ['2D h5', '3D h5']:
            self.io_sav_writer = data_writer(h5_writer)
            self.misc_io_writer_cfg['ds_path'] = 'img'
        else:
            self.io_sav_writer = data_writer(tif_seq_writer)
            self.misc_io_writer_cfg['scan_id='] = None
            self.misc_io_writer_cfg['idx_s'] = 0
            self.misc_io_writer_cfg['idx_dim'] = 0
        self.misc_io_man_sav_fn_set = False
        self.misc_io_cfg_fn_set = False
        self.misc_io_cfg_set = False
        self.boxes_logic()

    def AlgnCfgOutPath_btn_clk(self, a):
        if len(a.files[0]) != 0:
            self.misc_io_sav_algn_fn = a.files[0]
            self.misc_io_man_sav_fn_set = True
            # self.hs['AlgnCfgOutPath btn'].initialdir = os.path.abspath(
            #     os.path.dirname(a.files[0]))
            update_json_content(
                self.global_h.GUI_cfg_file,
                {'cwd': os.path.abspath(os.path.dirname(a.files[0]))})
        else:
            self.misc_io_sav_algn_fn = None
            self.misc_io_man_sav_fn_set = False
        self.misc_io_cfg_fn_set = False
        self.misc_io_cfg_set = False
        self.boxes_logic()

    def AlgnCfgCfmManFilePath_btn_clk(self, a):
        self.misc_io_cfg_type = "manual"
        if self.misc_io_in_ftype in ['3D tif', '3D h5']:
            self.misc_io_tgt_3D_slcn_dim = 0
            self.misc_io_src_3D_slicing_dim = 0
        self.misc_io_tgt_is_raw = True
        self.misc_io_src_is_raw = True
        self.misc_dat_scla = 1
        self.misc_dat_use_smth_flat = False
        self.misc_dat_smth_flat_sig = 0
        self.misc_dat_use_alt_flat = False

        self.misc_io_alt_flat_fn = None

        dirname = os.path.dirname(self.misc_io_src_fn)
        if os.path.exists(os.path.join(dirname, 'aligned')):
            shutil.rmtree(os.path.join(dirname, 'aligned'))
        os.mkdir(os.path.join(dirname, 'aligned'))
        ext = os.path.basename(self.misc_io_src_fn).split('.')[-1]
        stem = ''.join(os.path.basename(self.misc_io_src_fn).split('.')[:-1])
        if self.misc_io_in_ftype == '2Ds tif':
            s = '_'.join(stem.split('_')[:-2]) + '_'
            self.misc_io_src_fn_temp = os.path.join(dirname,
                                                    (s + '{0}_{1}.' + ext))
            self.misc_io_src_fn_id_s = stem.split('_')[-2]
            self.misc_io_src_fn_id_e = stem.split('_')[-2]
            self.misc_io_src_sli_id_s = stem.split('_')[-1]
            self.misc_io_src_sli_id_e = stem.split('_')[-1]
        else:
            s = '_'.join(stem.split('_')[:-1]) + '_'
            self.misc_io_src_fn_temp = os.path.join(dirname,
                                                    (s + '{0}.' + ext))
            self.misc_io_src_fn_id_s = stem.split('_')[-1]
            self.misc_io_src_fn_id_e = stem.split('_')[-1]
        t = '-'.join(time.asctime().replace(':', '-').split(' '))
        self.misc_sav_algn_fn = os.path.join(dirname, 'aligned',
                                             ('aligned_' + s + t + '{}.h5'))
        self.misc_io_cfg_fn = os.path.join(dirname,
                                           s + 'io_config_' + t + '.json')
        self.misc_sav_trl_reg_cfg_fn = os.path.join(
            dirname, 'trial_reg_' + s + 'config_' + t + '.json')
        self.misc_rev_best_mtch_fn = os.path.join(
            dirname, 'trial_reg_' + s + 'best_match_' + t + '.json')

        self.misc_io_cfg = {'Note 1' : 'cfg type == manual is good for generating a config template that can be used '+\
                                       'in a more complicated scenario',
                            'Note 2' : 'With 2D/3D tif input file formats, it is suitable for registering two '+\
                                       'individual images. If the file names are distinguished by an index on the '+\
                                       'end, it is also possible to register a series of images to one reference '+\
                                       'image. The option with 2D/3D h5 file formats is similar to 2D/3D tif option. '+\
                                       'The difference between them is that you need to specify dataset path for h5. '+\
                                       'In both cases, you can specify flat and dark images if the input data is raw. '+\
                                       '2Ds tif input file format option is for registering two sets of 3D images '+\
                                       'in 2D maner. You will need to specify one slice image from the reference image '+\
                                       'then have the slice images in a range in the second image compare to the '+\
                                       'reference slice image. It is supposed that there are two indices in the file '+\
                                       'names, one is for distinguishing the two image series, and another is for '+\
                                       'indexing slice images in each image series.',
                            'cfg type' : self.misc_io_cfg_type,
                            'in tgt data' : {
                                'ftype' : self.misc_io_tgt_ftype,
                                'data fn' : self.misc_io_tgt_fn,
                                'is data fn temp': self.misc_io_tgt_fn_is_temp,
                                'data fn id start' : self.misc_io_tgt_fn_id_s,
                                'data fn id end' : self.misc_io_tgt_fn_id_e,
                                'is raw' : False,
                                'tif' : {
                                    '2D' : {'flat fn': None,
                                            'dark fn': None},
                                    '3D' : {'flat fn': None,
                                            'dark fn': None,
                                            'flat sli idx': 0,
                                            'dark sli idx': -1,
                                            'slicing dim': 0,
                                            'fixed sli': 0},
                                    '2Ds' : {'flat fn': None,
                                             'dark fn': None}
                                         },
                                'h5' : {
                                    '2D' : {
                                        'flat fn': self.misc_io_tgt_fn if self.misc_io_tgt_ftype == '2D h5' else None,
                                        'dark fn': self.misc_io_tgt_fn if self.misc_io_tgt_ftype == '2D h5' else None,
                                        'data path': '/img',
                                        'flat path': '/img_bkg',
                                        'dark path': '/img_dark'},
                                    '3D' : {
                                        'flat fn': self.misc_io_tgt_fn if self.misc_io_tgt_ftype == '3D h5' else None,
                                        'dark fn': self.misc_io_tgt_fn if self.misc_io_tgt_ftype == '3D h5' else None,
                                        'data path': '/img',
                                        'flat path': '/img_bkg',
                                        'dark path': '/img_dark',
                                        'slicing dim': 0,
                                        'fixed sli': 0}
                                    }
                                },
                            'in src data' : {
                                'ftype' : self.misc_io_src_ftype,
                                'is raw' : False,
                                'data fn' : self.misc_io_src_fn_temp,
                                'is data fn temp': self.misc_io_src_fn_is_temp,
                                'data fn id start' : self.misc_io_src_fn_id_s,
                                'data fn id end' : self.misc_io_src_fn_id_e,
                                'tif' : {
                                    '2D' : {'flat fn': None,
                                            'dark fn': None},
                                    '3D' : {'flat fn': None,
                                            'dark fn': None,
                                            'flat sli idx': 0,
                                            'dark sli idx': -1,
                                            'slicing dim': 0,
                                            'srch sli half width': 10},
                                    '2Ds' : {'flat fn': None,
                                             'dark fn': None,
                                             'sli start': self.misc_io_src_sli_id_s,
                                             'sli end': self.misc_io_src_sli_id_e,
                                             'srch sli half width': 10}
                                    },
                                'h5' : {
                                    '2D' : {
                                        'flat fn': self.misc_io_src_fn if self.misc_io_tgt_ftype == '2D h5' else None,
                                        'dark fn': self.misc_io_src_fn if self.misc_io_tgt_ftype == '2D h5' else None,
                                        'data path': '/img',
                                        'flat path': '/img_bkg',
                                        'dark path': '/img_dark'},
                                    '3D' : {
                                        'flat fn': self.misc_io_src_fn if self.misc_io_tgt_ftype == '3D h5' else None,
                                        'dark fn': self.misc_io_src_fn if self.misc_io_tgt_ftype == '3D h5' else None,
                                        'data path': '/img',
                                        'flat path': '/img_bkg',
                                        'dark path': '/img_dark',
                                        'slicing dim': 0,
                                        'srch sli half width': 10}
                                }
                            },
                            'out algn data' : {
                                'ftype' : 'h5',
                                'out fn' : self.misc_io_sav_algn_fn
                                },
                            'cfg fn' : {
                                'cfg fn' : self.misc_io_cfg_fn
                                }
                            }
        try:
            with open(self.misc_io_cfg_fn, 'w') as f:
                json.dump(self.misc_io_cfg,
                          f,
                          indent=4,
                          separators=(',', ': '))
            self.misc_io_cfg_set = True
        except:
            self.misc_io_cfg_set = False
            print(
                "Cannot save the configuration file to the specified location."
            )
        self.misc_io_cfg_fn_set = False
        self.misc_dat_set = False
        self.boxes_logic()

    def AlgnCfgCfgFn_btn_clk(self, a):
        if len(a.files[0]) != 0:
            self.misc_io_cfg_fn = a.files[0]
            self.misc_io_cfg_fn_set = True
            update_json_content(
                self.global_h.GUI_cfg_file,
                {'cwd': os.path.abspath(os.path.dirname(a.files[0]))})
        else:
            self.misc_io_cfg_fn = None
            self.misc_io_cfg_fn_set = False
        self.misc_io_cfg_set = False
        self.boxes_logic()

    def AlgnCfgCfmCfgFn_btn_clk(self, a):
        if self.misc_io_cfg_fn_set:
            try:
                self.read_io_cfg(self.misc_io_cfg_fn)
                self.misc_io_cfg['cfg type'] = 'cfg file'
                self.hs[
                    'SelCfgFn text'].value = "The Specified Config File is Read ..."
                self.misc_io_cfg_set = True
            except Exception as e:
                print(e.__repr__())
                self.hs[
                    'SelCfgFn text'].value = "Cannot Read the Specified Config File. Please Specify a Valid Config File ..."
                self.misc_io_cfg_set = False
        else:
            self.hs[
                'SelCfgFn text'].value = 'No Config File is Specified. Please Specify a Config File ...'
            self.misc_io_cfg_set = False
        self.misc_dat_set = False
        self.boxes_logic()

    def AlgnROIPreAx0Shft_txt_chg(self, a):
        pass

    def AlgnROIPreAx1Shft_txt_chg(self, a):
        pass

    def AlgnROIPreAx2Shft_txt_chg(self, a):
        pass

    def AlgnROISrcROIAx0_sldr_chg(self, a):
        pass

    def AlgnROISrcROIAx1_sldr_chg(self, a):
        pass

    def AlgnROISrcROIAx2_sldr_chg(self, a):
        pass

    def AlgnROICfm_btn_clk(self, a):
        pass

    def AlgnRegFijiMaskViewer_chbx_chg(self, a):
        pass

    def AlgnRegChnkSz_chbx_chg(self, a):
        pass

    def AlgnRegChnkSz_sldr_chg(self, a):
        pass

    def AlgnRegSliSrch_sldr_chg(self, a):
        pass

    def AlgnRegUsMsk_chbx_chg(self, a):
        pass

    def AlgnRegMskThrs_sldr_chg(self, a):
        pass

    def AlgnRegMskDltn_sldr_chg(self, a):
        pass

    def AlgnRegRegMthd_drpdn_chg(self, a):
        pass

    def AlgnRegRefMod_drpdn_chg(self, a):
        pass

    def AlgnRegMRTVLv_text_chg(self, a):
        pass

    def AlgnRegMRTVWz_txt_chg(self, a):
        pass

    def AlgnRegMRTVSubpxlWz_txt_chg(self, a):
        pass

    def AlgnRegMRTVSubpxlKrn_txt_chg(self, a):
        pass

    def AlgnRegMRTVSubpxlSrch_drpdn_chg(self, a):
        pass

    def AlgnRegRunCfm_btn_clk(self, a):
        pass

    def AlgnRevRegPair_sldr_chg(self, a):
        pass

    def AlgnRevRegPairBad_btn_clk(self, a):
        pass

    def AlgnRevCorrXShft_txt_chg(self, a):
        pass

    def AlgnRevCorrYShft_txt_chg(self, a):
        pass

    def AlgnRevCorrZShft_txt_chg(self, a):
        pass

    def AlgnRevCorrShftRec_btn_clk(self, a):
        pass

    def AlgnRevCfmRevProcAlgn_btn_clk(self, a):
        pass
