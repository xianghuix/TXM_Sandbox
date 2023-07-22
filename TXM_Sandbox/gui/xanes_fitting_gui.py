#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:56:17 2020

@author: xiao
"""
import os, h5py, json, numpy as np, pandas as pd

from ipywidgets import widgets
from copy import deepcopy
import napari

from .gui_components import (NumpyArrayEncoder, get_handles,
                             enable_disable_boxes, gen_external_py_script,
                             fiji_viewer_off, fiji_viewer_state,
                             fiji_viewer_on, scale_eng_list, SelectFilesButton,
                             update_json_content)
from ..utils import xanes_math as xm
from ..utils import xanes_analysis as xa
from ..dicts import customized_struct_dict as dat_dict

inf = np.inf

napari.gui_qt()

NUMPY_FIT_ORDER = [2, 3, 4]


class xanes_fitting_gui():

    def __init__(self, parent_h, form_sz=[650, 740]):
        self.parent_h = parent_h
        self.global_h = parent_h.global_h
        self.hs = {}
        self.form_sz = form_sz
        self.fit_img_bin_fac = 1
        self.fit_wl_fit_use_param_bnd = False
        self.fit_wl_optimizer = 'numpy'
        self.fit_wl_fit_func = 2
        self.fit_edge_fit_use_param_bnd = False
        self.fit_edge_optimizer = 'numpy'
        self.fit_edge_fit_func = 3
        self.analysis_saving_items = set(dat_dict.XANES_FULL_SAVE_DEFAULT)
        self.parent_h.xanes_fit_type == 'full'
        self.xanes_fit_eng_config = False

        self.fit_mask_prev = False
        self.fit_flt_prev_sli = 0
        self.fit_flt_prev_configed = False
        self.fit_flt_prev_maskit = False
        self.fit_lcf = False
        self.fit_lcf_ref_set = False
        self.fit_lcf_constr = True
        self.fit_lcf_ref_num = 2
        self.fit_lcf_ref_spec = pd.DataFrame(
            columns=['fpath', 'fname', 'eng', 'mu'])

        self.pre_es_idx = None
        self.pre_ee_idx = None
        self.post_es_idx = None
        self.post_ee_idx = None
        self.fit_flt_prev_xana = None
        self.e0_idx = None
        self.pre = None
        self.post = None
        self.edge_jump_mask = None
        self.fitted_edge_mask = None

        self.fit_wl_pos = True
        # self.fit_edge_pos = False
        self.fit_edge50_pos = True
        self.find_wl_pos_dir = False
        self.find_edge50_pos_dir = True
        self.fit_use_flt_spec = False

        if self.parent_h.gui_name == 'xanes3D':
            self.fn = self.parent_h.xanes_save_trial_reg_filename
            self.xanes_mask_external_command_name = os.path.join(
                self.global_h.script_dir, 'xanes3D_mask_external_command.py')
        elif self.parent_h.gui_name == 'xanes2D':
            self.fn = self.parent_h.xanes_save_trial_reg_filename
            self.xanes_mask_external_command_name = os.path.join(
                self.global_h.script_dir, 'xanes2D_mask_external_command.py')

    def build_gui(self):
        ## ## ## bundle sub-tabs in each tab - analysis&display TAB in 3D_xanes TAB -- start
        ## ## ## ## define functional widgets each tab in each sub-tab - analysis box in analysis&display TAB -- start
        layout = {
            'border': '3px solid #8855AA',
            'width': 'auto',
            'height': 'auto'
        }
        self.hs['Fitting form'] = widgets.VBox()
        self.hs['Fitting form'].layout = layout

        ## ## ## ## fit energy config box -- start
        layout = {
            'border': '3px solid #8855AA',
            'width': 'auto',
            'height': 'auto'
        }
        self.hs['FitEngConfig box'] = widgets.VBox()
        self.hs['FitEngConfig box'].layout = layout

        ## ## ## ## ## label analysis box -- start
        layout = {
            'border': '3px solid #FFCC00',
            'width': 'auto',
            'height': 'auto'
        }
        self.hs['FitTitle box'] = widgets.HBox()
        self.hs['FitTitle box'].layout = layout
        self.hs['FitTitle label'] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + 'XANES Fitting' + '</span>')
        layout = {
            'background-color': 'white',
            'color': 'cyan',
            'left': '41%',
            'width': 'auto'
        }
        self.hs['FitTitle label'].layout = layout
        self.hs['FitTitle box'].children = [self.hs['FitTitle label']]
        ## ## ## ## ## label analysis box -- end

        ## ## ## ## ## define type of analysis and energy range -- start
        layout = {
            'border': '3px solid #FFCC00',
            'width': 'auto',
            'height': 'auto'
        }
        self.hs['FitEngRag box'] = widgets.VBox()
        self.hs['FitEngRag box'].layout = layout
        layout = {'border': 'none', 'width': 'auto', 'height': 'auto'}
        self.hs['FitEngRagRow0 box'] = widgets.HBox()
        self.hs['FitEngRagRow0 box'].layout = layout
        layout = {'width': '19%', 'height': '90%'}
        self.hs['FitEngRagOptn drpdn'] = widgets.Dropdown(
            description='analysis type',
            description_tooltip=
            'wl: find whiteline positions without doing background removal and normalization; edge0.5: find energy point where the normalized spectrum value equal to 0.5; full: doing regular xanes preprocessing',
            options=['wl', 'full'],
            value='wl',
            disabled=True)
        self.hs['FitEngRagOptn drpdn'].layout = layout
        layout = {'width': '19%', 'height': '90%'}
        self.hs['FitEngRagEdgeEng text'] = widgets.BoundedFloatText(
            description='edge eng',
            description_tooltip='edge energy (keV)',
            value=0,
            min=0,
            max=50000,
            step=0.5,
            disabled=True)
        self.hs['FitEngRagEdgeEng text'].layout = layout
        layout = {'width': '19%', 'height': '90%'}
        self.hs['FitEngRagPreEdgeEnd text'] = widgets.BoundedFloatText(
            description='pre edge e',
            description_tooltip=
            'relative ending energy point (keV) of pre-edge from edge energy for background removal',
            value=-50,
            min=-500,
            max=0,
            step=0.5,
            disabled=True)
        self.hs['FitEngRagPreEdgeEnd text'].layout = layout
        layout = {'width': '19%', 'height': '90%'}
        self.hs['FitEngRagPostEdgeStr text'] = widgets.BoundedFloatText(
            description='post edge s',
            description_tooltip=
            'relative starting energy point (keV) of post-edge from edge energy for normalization',
            value=100,
            min=0,
            max=500,
            step=0.5,
            disabled=True)
        self.hs['FitEngRagPostEdgeStr text'].layout = layout
        layout = {'width': '19%', 'height': '90%'}

        self.hs['FitEngRagOptn drpdn'].observe(self.fit_eng_rag_optn_drpdn,
                                               names='value')
        self.hs['FitEngRagEdgeEng text'].observe(
            self.fit_eng_rag_edge_eng_text_chg, names='value')
        self.hs['FitEngRagPreEdgeEnd text'].observe(
            self.fit_eng_rag_pre_edge_end_text_chg, names='value')
        self.hs['FitEngRagPostEdgeStr text'].observe(
            self.fit_eng_rag_post_edge_str_text_chg, names='value')
        self.hs['FitEngRagRow0 box'].children = [
            self.hs['FitEngRagOptn drpdn'], self.hs['FitEngRagEdgeEng text'],
            self.hs['FitEngRagPreEdgeEnd text'],
            self.hs['FitEngRagPostEdgeStr text']
        ]

        layout = {
            'border': 'none',
            'width': f'{1*self.form_sz[1]-110}px',
            'height': f'{0.07*(self.form_sz[0]-128)}px'
        }
        self.hs['FitEngRagRow1 box'] = widgets.HBox()
        self.hs['FitEngRagRow1 box'].layout = layout
        layout = {'width': '19%', 'height': '90%'}
        self.hs['FitEngRagWlFitStr text'] = widgets.BoundedFloatText(
            description='wl eng s',
            description_tooltip=
            'absolute energy starting point (keV) for whiteline fitting. "wl eng s" and "wl eng e" shoudl be roughly symmetric about whiteline peak.',
            value=0,
            min=0,
            max=50000,
            step=0.5,
            disabled=True)
        self.hs['FitEngRagWlFitStr text'].layout = layout
        layout = {'width': '19%', 'height': '90%'}
        self.hs['FitEngRagWlFitEnd text'] = widgets.BoundedFloatText(
            description='wl eng e',
            description_tooltip=
            'absolute energy ending point (keV) for whiteline fitting. "wl eng s" and "wl eng e" shoudl be roughly symmetric about whiteline peak.',
            value=0,
            min=0,
            max=50030,
            step=0.5,
            disabled=True)
        self.hs['FitEngRagWlFitEnd text'].layout = layout
        layout = {'width': '19%', 'height': '90%'}
        self.hs['FitEngRagEdge0.5Str text'] = widgets.BoundedFloatText(
            description='edge0.5 s',
            description_tooltip=
            'absolute energy starting point (keV) for edge-0.5 fitting. "edge0.5 s" and "edge0.5 e" shoudl be close to the foot and the peak locations on the ascendent edge of the spectra.',
            value=0,
            min=0,
            max=50000,
            step=0.5,
            disabled=True)
        self.hs['FitEngRagEdge0.5Str text'].layout = layout
        layout = {'width': '19%', 'height': '90%'}
        self.hs['FitEngRagEdge0.5End text'] = widgets.BoundedFloatText(
            description='edge0.5 e',
            description_tooltip=
            'absolute energy starting point (keV) for edge-0.5 fitting. "edge0.5 s" and "edge0.5 e" shoudl be close to the foot and the peak locations on the ascendent edge of the spectra.',
            value=0,
            min=0,
            max=50030,
            step=0.5,
            disabled=True)
        self.hs['FitEngRagEdge0.5End text'].layout = layout

        layout = {'width': '15%', 'height': '70%', 'left': '7%'}
        self.hs['FitEngCmf btn'] = widgets.Button(
            description='Confirm',
            disabled=True,
            description_tooltip='Confirm energy range settings')
        self.hs['FitEngCmf btn'].layout = layout
        self.hs['FitEngCmf btn'].style.button_color = 'darkviolet'

        self.hs['FitEngRagWlFitStr text'].observe(
            self.fit_eng_rag_wl_fit_str_text_chg, names='value')
        self.hs['FitEngRagWlFitEnd text'].observe(
            self.fit_eng_rag_wl_fit_end_text_chg, names='value')
        self.hs['FitEngRagEdge0.5Str text'].observe(
            self.fit_eng_rag_edge50_fit_str_text_chg, names='value')
        self.hs['FitEngRagEdge0.5End text'].observe(
            self.fit_eng_rag_edge50_fit_end_text_chg, names='value')
        self.hs['FitEngCmf btn'].on_click(self.fit_eng_rag_cmf_btn_clk)
        self.hs['FitEngRagRow1 box'].children = [
            self.hs['FitEngRagWlFitStr text'],
            self.hs['FitEngRagWlFitEnd text'],
            self.hs['FitEngRagEdge0.5Str text'],
            self.hs['FitEngRagEdge0.5End text'], self.hs['FitEngCmf btn']
        ]

        self.hs['FitEngRag box'].children = [
            self.hs['FitEngRagRow0 box'], self.hs['FitEngRagRow1 box']
        ]
        ## ## ## ## ## define type of analysis and energy range -- end

        self.hs['FitEngConfig box'].children = [
            self.hs['FitTitle box'], self.hs['FitEngRag box']
        ]
        ## ## ## ## fit energy config box -- end

        ## ## ## ## ## define fitting parameters -- start
        self.hs['FitItem tab'] = widgets.Tab()
        self.hs['FitItem tab'].layout = {
            'width': 'auto',
            'height': f'{0.61*(self.form_sz[0]-128)}px'
        }

        ## ## ## ## ## ## define analysis options parameters -- start
        layout = {
            'border': '3px solid #FFCC00',
            'width': 'auto',
            'height': 'auto'
        }
        self.hs['FitItemConfig box'] = widgets.HBox()
        self.hs['FitItemConfig box'].layout = layout

        fit_pars_GridspecLayout = widgets.GridspecLayout(
            8,
            200,
            layout={
                "border": "3px solid #FFCC00",
                'width': '100%',
                "height": '100%'
            })

        fit_pars_GridspecLayout[0, :39] = widgets.Checkbox(
            description='fit wl',
            description_tooltip="fit whiteline if it is checked",
            value=True,
            indent=False,
            disabled=False,
            layout={'width': '95%'})
        self.hs['FitItemConfigFitWl chbx'] = fit_pars_GridspecLayout[0, :39]

        fit_pars_GridspecLayout[0, 39:78] = widgets.Checkbox(
            description='fit edge 50%',
            description_tooltip=
            'fit the position where normalized mu is 50% of maximum if it is checked',
            value=True,
            indent=False,
            disabled=False,
            layout={'width': '95%'})
        self.hs['FitItemConfigFitEdge50% chbx'] = fit_pars_GridspecLayout[
            0, 39:78]

        fit_pars_GridspecLayout[0, 78:117] = widgets.Checkbox(
            description='filter spec',
            description_tooltip=
            'filter spec before fitting edge if it is checked',
            value=False,
            indent=False,
            disabled=False,
            layout={'width': '95%'})
        self.hs['FitItemConfigFitSpec chbx'] = fit_pars_GridspecLayout[0,
                                                                       78:117]

        fit_pars_GridspecLayout[0, 117:156] = widgets.Checkbox(
            description='wl direct',
            description_tooltip=
            'find white-line positions directly from measured spectra',
            value=False,
            indent=False,
            disabled=False,
            layout={'width': '95%'})
        self.hs['FitItemConfigCalWlDir chbx'] = fit_pars_GridspecLayout[
            0, 117:156]

        fit_pars_GridspecLayout[0, 156:195] = widgets.Checkbox(
            description='edge 50% direct',
            description_tooltip=
            'find edge positions at which the normalized mu equal to 50% of their peak values',
            value=False,
            indent=False,
            disabled=False,
            layout={'width': '95%'})
        self.hs['FitItemConfigCalEdge50%Dir chbx'] = fit_pars_GridspecLayout[
            0, 156:195]

        fit_pars_GridspecLayout[1, :70] = widgets.FloatSlider(
            description='edge jump thres',
            description_tooltip=
            'edge jump in unit of the standard deviation of the signal in energy range pre to the edge. larger threshold enforces more restrict data quality validation on the data',
            value=1,
            min=0,
            max=50,
            step=0.1,
            disabled=True,
            layout={'width': '95%'})
        self.hs['FitItemConfigEdgeJumpThres sldr'] = fit_pars_GridspecLayout[
            1, :70]

        fit_pars_GridspecLayout[1, 70:140] = widgets.FloatSlider(
            description='offset thres',
            description_tooltip=
            'offset between pre-edge and post-edge in unit of the standard deviation of pre-edge. larger offser enforces more restrict data quality validation on the data',
            value=1,
            min=0,
            max=50,
            step=0.1,
            disabled=True,
            layout={'width': '95%'})
        self.hs['FitItemConfigEdgeOfstThres sldr'] = fit_pars_GridspecLayout[
            1, 70:140]

        fit_pars_GridspecLayout[1, 150:200] = widgets.BoundedIntText(
            description='binning fac',
            description_tooltip=
            'binning factor applied to the spec image before xanes analysis',
            value=1,
            min=1,
            max=10,
            step=1,
            disabled=True,
            layout={'width': '95%'})
        self.hs['FitItemConfigBinFac text'] = fit_pars_GridspecLayout[1,
                                                                      150:200]

        fit_pars_GridspecLayout[2, :20] = widgets.Checkbox(
            description='mask preview',
            description_tooltip='preview edge jump and edge offset masks',
            value=0,
            indent=False,
            disabled=True,
            layout={'width': '95%'})
        self.hs['FitItemConfigFitPrvw chbx'] = fit_pars_GridspecLayout[2, :20]

        fit_pars_GridspecLayout[3, :70] = widgets.IntSlider(
            description='xy slice',
            description_tooltip=
            'Select a slice to generate edge_jump and edge_offset filters for preview',
            value=0,
            min=0,
            max=10,
            disabled=True,
            layout={'width': '95%'})
        self.hs['FitItemConfigFitPrvwSli sldr'] = fit_pars_GridspecLayout[
            3, :70]

        fit_pars_GridspecLayout[3, 80:100] = widgets.Button(
            description='calc masks',
            disabled=True,
            description_tooltip=
            'Calculate and preview the edge_jump and edge_offset filters',
            layout={
                'width': '95%',
                'height': '70%'
            })
        self.hs['FitItemConfigFitPrvwCal btn'] = fit_pars_GridspecLayout[
            3, 80:100]
        self.hs[
            'FitItemConfigFitPrvwCal btn'].style.button_color = 'darkviolet'

        fit_pars_GridspecLayout[3, 110:130] = widgets.Button(
            description='EM it',
            disabled=True,
            description_tooltip=
            'Calculate and preview the edge_jump and edge_offset filters',
            layout={
                'width': '95%',
                'height': '70%'
            })
        self.hs['FitItemConfigFitPrvwEMIt btn'] = fit_pars_GridspecLayout[
            3, 110:130]
        self.hs[
            'FitItemConfigFitPrvwEMIt btn'].style.button_color = 'darkviolet'

        fit_pars_GridspecLayout[3, 140:160] = widgets.Button(
            description='OM it',
            disabled=True,
            description_tooltip=
            'Calculate and preview the edge_jump and edge_offset filters',
            layout={
                'width': '95%',
                'height': '70%'
            })
        self.hs['FitItemConfigFitPrvwOMIt btn'] = fit_pars_GridspecLayout[
            3, 140:160]
        self.hs[
            'FitItemConfigFitPrvwOMIt btn'].style.button_color = 'darkviolet'

        fit_pars_GridspecLayout[3, 170:190] = widgets.Button(
            description='EM&OM it',
            disabled=True,
            description_tooltip=
            'Calculate and preview the edge_jump and edge_offset filters',
            layout={
                'width': '95%',
                'height': '70%'
            })
        self.hs['FitItemConfigFitPrvwEMOMIt btn'] = fit_pars_GridspecLayout[
            3, 170:190]
        self.hs[
            'FitItemConfigFitPrvwEMOMIt btn'].style.button_color = 'darkviolet'

        fit_pars_GridspecLayout[4, :40] = widgets.Checkbox(
            description='LCF',
            description_tooltip='Linear Combination Fitting',
            value=0,
            indent=False,
            disabled=True,
            layout={
                'width': '95%',
                "height": "95%"
            })
        self.hs['FitItemConfigLCF chbx'] = fit_pars_GridspecLayout[4, :40]

        fit_pars_GridspecLayout[4, 40:80] = widgets.Checkbox(
            description='Constr',
            value=1,
            description_tooltip='Apply unity constraint in LCF',
            indent=False,
            disabled=True,
            layout={
                'width': '95%',
                "height": "95%"
            })
        self.hs['FitItemConfigLCFCnst chbx'] = fit_pars_GridspecLayout[4,
                                                                       40:80]

        fit_pars_GridspecLayout[5, :40] = widgets.BoundedIntText(
            description='# of spec',
            disabled=True,
            description_tooltip=
            'total # of reference spectra for Linear Combination Fitting',
            value=2,
            min=2,
            max=5,
            step=1,
            indent=False,
            layout={
                'width': '95%',
                "height": "95%"
            })
        self.hs['FitItemConfigLCFNumSpec text'] = fit_pars_GridspecLayout[
            5, :40]

        fit_pars_GridspecLayout[6, 14:40] = SelectFilesButton(
            option='askopenfilename',
            open_filetypes=(('text files', '*.txt'), ),
            layout={
                'width': '95%',
                "height": "95%"
            })
        self.hs['FitItemConfigLCFSelRef btn'] = fit_pars_GridspecLayout[6,
                                                                        14:40]
        self.hs['FitItemConfigLCFSelRef btn'].description = "Load Ref Spec"
        try:
            with open(self.global_h.GUI_cfg_file, 'r') as f:
                cfg = json.load(f)
                self.hs['FitItemConfigLCFSelRef btn'].initialdir = cfg[
                    'xanes_ref_d']
        except:
            self.hs['FitItemConfigLCFSelRef btn'].initialdir = os.path.abspath(
                os.path.curdir)

        fit_pars_GridspecLayout[5:8, 50:150] = \
            widgets.SelectMultiple(options=[], value=[],
                                   layout={"width": "95%", "height": "95%"},
                                   rows=10, description='Ref Specs', disabled=True)
        self.hs['FitItemConfigLCFRef list'] = fit_pars_GridspecLayout[5:8,
                                                                      50:150]

        fit_pars_GridspecLayout[6,
                                170:190] = widgets.Button(layout={
                                    'width': "95%",
                                    "height": "70%"
                                },
                                                          description='Remove',
                                                          disabled=True)
        self.hs['FitItemConfigLCFRmRef btn'] = fit_pars_GridspecLayout[6,
                                                                       170:190]
        self.hs['FitItemConfigLCFRmRef btn'].style.button_color = 'darkviolet'

        self.hs['FitItemConfigFitWl chbx'].observe(
            self.fit_config_fit_wl_chbx_chg, names='value')
        self.hs['FitItemConfigFitEdge50% chbx'].observe(
            self.fit_config_fit_edge50_chbx_chg, names='value')
        self.hs['FitItemConfigFitSpec chbx'].observe(
            self.fit_config_flt_spec_chbx_chg, names='value')
        self.hs['FitItemConfigCalWlDir chbx'].observe(
            self.fit_config_cal_wl_dir_chbx_chg, names='value')
        self.hs['FitItemConfigCalEdge50%Dir chbx'].observe(
            self.fit_config_cal_edge50_dir_chbx_chg, names='value')
        self.hs['FitItemConfigEdgeJumpThres sldr'].observe(
            self.fit_config_fit_edge_jump_thres_sldr_chg, names='value')
        self.hs['FitItemConfigEdgeOfstThres sldr'].observe(
            self.fit_config_fit_edge_ofst_thres_sldr_chg, names='value')
        self.hs['FitItemConfigBinFac text'].observe(
            self.fit_config_bin_fac_text_chg, names='value')
        self.hs['FitItemConfigFitPrvw chbx'].observe(
            self.fit_config_mask_prvw_chbx_chg, names='value')
        self.hs['FitItemConfigFitPrvwSli sldr'].observe(
            self.fit_config_flt_prvw_sli_sldr_chg, names='value')
        self.hs['FitItemConfigFitPrvwCal btn'].on_click(
            self.fit_config_flt_prvw_calc_btn_clk)
        self.hs['FitItemConfigFitPrvwEMIt btn'].on_click(
            self.fit_config_flt_prvw_em_it_btn_clk)
        self.hs['FitItemConfigFitPrvwOMIt btn'].on_click(
            self.fit_config_flt_prvw_om_it_btn_clk)
        self.hs['FitItemConfigFitPrvwEMOMIt btn'].on_click(
            self.fit_config_flt_prvw_emom_it_btn_clk)
        self.hs['FitItemConfigLCF chbx'].observe(self.fit_config_lcf_chbx_chg,
                                                 names='value')
        self.hs['FitItemConfigLCFCnst chbx'].observe(
            self.fit_config_lcf_cnst_chbx_chg, names='value')
        self.hs['FitItemConfigLCFNumSpec text'].observe(
            self.fit_config_num_spec_text_chg, names='value')
        self.hs['FitItemConfigLCFSelRef btn'].on_click(
            self.fit_config_sel_ref_btn_clk)
        self.hs['FitItemConfigLCFRef list'].observe(
            self.fit_config_ref_spec_list_chg, names='value')
        self.hs['FitItemConfigLCFRmRef btn'].on_click(
            self.fit_config_ref_rm_btn_clk)

        self.hs['FitItemConfig box'] = fit_pars_GridspecLayout
        self.hs['FitItemConfig box'].children = [
            self.hs['FitItemConfigFitWl chbx'],
            self.hs['FitItemConfigFitEdge50% chbx'],
            self.hs['FitItemConfigFitSpec chbx'],
            self.hs['FitItemConfigCalWlDir chbx'],
            self.hs['FitItemConfigCalEdge50%Dir chbx'],
            self.hs['FitItemConfigEdgeJumpThres sldr'],
            self.hs['FitItemConfigEdgeOfstThres sldr'],
            self.hs['FitItemConfigBinFac text'],
            self.hs['FitItemConfigFitPrvw chbx'],
            self.hs['FitItemConfigFitPrvwSli sldr'],
            self.hs['FitItemConfigFitPrvwCal btn'],
            self.hs['FitItemConfigFitPrvwEMIt btn'],
            self.hs['FitItemConfigFitPrvwOMIt btn'],
            self.hs['FitItemConfigFitPrvwEMOMIt btn'],
            self.hs['FitItemConfigLCF chbx'],
            self.hs['FitItemConfigLCFCnst chbx'],
            self.hs['FitItemConfigLCFNumSpec text'],
            self.hs['FitItemConfigLCFSelRef btn'],
            self.hs['FitItemConfigLCFRef list'],
            self.hs['FitItemConfigLCFRmRef btn']
        ]
        ## ## ## ## ## ## define analysis options parameters -- end

        ## ## ## ## ## ## define wl fitting parameters -- start
        layout = {
            'border': '3px solid #FFCC00',
            'width': 'auto',
            'height': 'auto'
        }
        self.hs['FitConfigWlPars box'] = widgets.HBox()
        self.hs['FitConfigWlPars box'].layout = layout

        fit_wl_pars_GridspecLayout = widgets.GridspecLayout(
            8,
            200,
            layout={
                "border": "3px solid #FFCC00",
                'width': '100%',
                'height': '100%'
            })

        fit_wl_pars_GridspecLayout[0, :50] = widgets.Dropdown(
            description='optimizer',
            value='numpy',
            isabled=True,
            description_tooltip='use scipy.optimize or numpy.polyfit',
            options=['scipy', 'numpy'],
            layout={
                'width': '95%',
                'height': '95%'
            })
        self.hs['FitConfigWlOptmzr drpdn'] = fit_wl_pars_GridspecLayout[0, :50]
        fit_wl_pars_GridspecLayout[0, 50:100] = widgets.Dropdown(
            description='peak func',
            disabled=True,
            description_tooltip='peak fitting functions',
            options=[2, 3, 4],
            value=2,
            layout={
                'width': '95%',
                'height': '95%'
            })
        self.hs['FitConfigWlFunc drpdn'] = fit_wl_pars_GridspecLayout[0,
                                                                      50:100]
        fit_wl_pars_GridspecLayout[0, 100:150] = widgets.Checkbox(
            description='para bnd',
            value=False,
            disabled=True,
            description_tooltip=
            "if set boundaries to the peak fitting function's parameters",
            layout={
                'width': '95%',
                'height': '95%'
            })
        self.hs['FitConfigWlFitUseBnd chbx'] = fit_wl_pars_GridspecLayout[
            0, 100:150]

        fit_wl_pars_GridspecLayout[1, 0:50] = widgets.BoundedFloatText(
            description='p0',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="fitting function variable 0",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitPars0 text'] = fit_wl_pars_GridspecLayout[1,
                                                                         0:50]
        fit_wl_pars_GridspecLayout[1, 50:100] = widgets.BoundedFloatText(
            description='p1',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="fitting function variable 1",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitPars1 text'] = fit_wl_pars_GridspecLayout[
            1, 50:100]
        fit_wl_pars_GridspecLayout[1, 100:150] = widgets.BoundedFloatText(
            description='p2',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="fitting function variable 2",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitPars2 text'] = fit_wl_pars_GridspecLayout[
            1, 100:150]
        fit_wl_pars_GridspecLayout[1, 150:200] = widgets.BoundedFloatText(
            description='p3',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="fitting function variable 3",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitPars3 text'] = fit_wl_pars_GridspecLayout[
            1, 150:200]

        fit_wl_pars_GridspecLayout[2, 0:50] = widgets.BoundedFloatText(
            description='p4',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="fitting function variable 4",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitPars4 text'] = fit_wl_pars_GridspecLayout[2,
                                                                         0:50]
        fit_wl_pars_GridspecLayout[2, 50:100] = widgets.Dropdown(
            description='p5',
            value='linear',
            options=['linear'],
            description_tooltip="fitting function variable 5",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitPars5 text'] = fit_wl_pars_GridspecLayout[
            2, 50:100]

        fit_wl_pars_GridspecLayout[3, 0:50] = widgets.Dropdown(
            description='jac',
            value='3-point',
            disabled=True,
            options=['2-point', '3-point', 'cs'],
            description_tooltip="",
            layout={
                'width': '95%',
                'height': '95%'
            })
        self.hs['FitConfigWlFitJac drpdn'] = fit_wl_pars_GridspecLayout[3,
                                                                        0:50]
        fit_wl_pars_GridspecLayout[3, 50:100] = widgets.Dropdown(
            description='method',
            value='trf',
            disabled=True,
            options=['trf', 'dogbox', 'lm'],
            description_tooltip="",
            layout={
                'width': '95%',
                'height': '95%'
            })
        self.hs['FitConfigWlFitMeth drpdn'] = fit_wl_pars_GridspecLayout[
            3, 50:100]
        fit_wl_pars_GridspecLayout[3, 100:150] = widgets.BoundedFloatText(
            description='ftol',
            value=1e-7,
            min=0,
            max=1e-3,
            description_tooltip=
            "function value change tolerance for terminating the optimization process",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitFtol text'] = fit_wl_pars_GridspecLayout[
            3, 100:150]
        fit_wl_pars_GridspecLayout[3, 150:200] = widgets.BoundedFloatText(
            description='xtol',
            value=1e-7,
            min=0,
            max=1e-3,
            description_tooltip=
            "function parameter change tolerance for terminating the optimization process",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitXtol text'] = fit_wl_pars_GridspecLayout[
            3, 150:200]
        fit_wl_pars_GridspecLayout[4, 0:50] = widgets.BoundedFloatText(
            description='gtol',
            value=1e-7,
            min=0,
            max=1e-3,
            description_tooltip=
            "function gradient change tolerance for terminating the optimization process",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitGtol text'] = fit_wl_pars_GridspecLayout[4,
                                                                        0:50]
        fit_wl_pars_GridspecLayout[4, 50:100] = widgets.BoundedIntText(
            description='ufac',
            value=50,
            min=1,
            max=100,
            description_tooltip=
            "upsampling factor to energy points in peak fitting",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitUfac text'] = fit_wl_pars_GridspecLayout[4,
                                                                        50:100]

        fit_wl_pars_GridspecLayout[5, 0:50] = widgets.BoundedFloatText(
            description='p0 lb',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p0 lower bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitPars0Lb text'] = fit_wl_pars_GridspecLayout[
            5, 0:50]
        fit_wl_pars_GridspecLayout[5, 50:100] = widgets.BoundedFloatText(
            description='p0 ub',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p0 upper bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitPars0Ub text'] = fit_wl_pars_GridspecLayout[
            5, 50:100]
        fit_wl_pars_GridspecLayout[5, 100:150] = widgets.BoundedFloatText(
            description='p1 lb',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p1 lower bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitPars1Lb text'] = fit_wl_pars_GridspecLayout[
            5, 100:150]
        fit_wl_pars_GridspecLayout[5, 150:200] = widgets.BoundedFloatText(
            description='p1 ub',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p1 upper bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitPars1Ub text'] = fit_wl_pars_GridspecLayout[
            5, 150:200]

        fit_wl_pars_GridspecLayout[6, 0:50] = widgets.BoundedFloatText(
            description='p2 lb',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p2 lower bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitPars2Lb text'] = fit_wl_pars_GridspecLayout[
            6, 0:50]
        fit_wl_pars_GridspecLayout[6, 50:100] = widgets.BoundedFloatText(
            description='p3 ub',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p3 upper bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitPars2Ub text'] = fit_wl_pars_GridspecLayout[
            6, 50:100]
        fit_wl_pars_GridspecLayout[6, 100:150] = widgets.BoundedFloatText(
            description='p4 lb',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p4 lower bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitPars3Lb text'] = fit_wl_pars_GridspecLayout[
            6, 100:150]
        fit_wl_pars_GridspecLayout[6, 150:200] = widgets.BoundedFloatText(
            description='p4 ub',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p4 upper bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitPars3Ub text'] = fit_wl_pars_GridspecLayout[
            6, 150:200]

        fit_wl_pars_GridspecLayout[7, 0:50] = widgets.BoundedFloatText(
            description='p5 lb',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p5 lower bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitPars4Lb text'] = fit_wl_pars_GridspecLayout[
            7, 0:50]
        fit_wl_pars_GridspecLayout[7, 50:100] = widgets.BoundedFloatText(
            description='p5 ub',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p5 upper bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigWlFitPars4Ub text'] = fit_wl_pars_GridspecLayout[
            7, 50:100]

        self.hs['FitConfigWlOptmzr drpdn'].observe(
            self.fit_config_wl_optmzr_drpdn_chg, names='value')
        self.hs['FitConfigWlFunc drpdn'].observe(
            self.fit_config_wl_func_drpdn_chg, names='value')
        self.hs['FitConfigWlFitUseBnd chbx'].observe(
            self.fit_config_wl_fit_bnd_chbx_chg, names='value')

        self.hs['FitConfigWlPars box'] = fit_wl_pars_GridspecLayout
        self.hs['FitConfigWlPars box'].children = [
            self.hs['FitConfigWlOptmzr drpdn'],
            self.hs['FitConfigWlFunc drpdn'],
            self.hs['FitConfigWlFitUseBnd chbx'],
            self.hs['FitConfigWlFitPars0 text'],
            self.hs['FitConfigWlFitPars1 text'],
            self.hs['FitConfigWlFitPars2 text'],
            self.hs['FitConfigWlFitPars3 text'],
            self.hs['FitConfigWlFitPars4 text'],
            self.hs['FitConfigWlFitPars5 text'],
            self.hs['FitConfigWlFitJac drpdn'],
            self.hs['FitConfigWlFitMeth drpdn'],
            self.hs['FitConfigWlFitFtol text'],
            self.hs['FitConfigWlFitXtol text'],
            self.hs['FitConfigWlFitGtol text'],
            self.hs['FitConfigWlFitUfac text'],
            self.hs['FitConfigWlFitPars0Lb text'],
            self.hs['FitConfigWlFitPars0Ub text'],
            self.hs['FitConfigWlFitPars1Lb text'],
            self.hs['FitConfigWlFitPars1Ub text'],
            self.hs['FitConfigWlFitPars2Lb text'],
            self.hs['FitConfigWlFitPars2Ub text'],
            self.hs['FitConfigWlFitPars3Lb text'],
            self.hs['FitConfigWlFitPars3Ub text'],
            self.hs['FitConfigWlFitPars4Lb text'],
            self.hs['FitConfigWlFitPars4Ub text']
        ]
        ## ## ## ## ## ## define wl fitting parameters -- end

        ## ## ## ## ## ## define edge fitting parameters -- start
        # layout = {'border':'3px solid #FFCC00', 'width':f'{0.96*self.form_sz[1]-98}px', 'height':f'{0.49*(self.form_sz[0]-128)}px'}
        layout = {
            'border': '3px solid #FFCC00',
            'width': 'auto',
            'height': 'auto'
        }
        self.hs['FitConfigEdgePars box'] = widgets.HBox()
        self.hs['FitConfigEdgePars box'].layout = layout

        fit_edge_pars_GridspecLayout = widgets.GridspecLayout(
            8,
            200,
            layout={
                "border": "3px solid #FFCC00",
                'width': '100%',
                'height': '100%'
            })

        fit_edge_pars_GridspecLayout[0, 0:50] = widgets.Dropdown(
            description='optimizer',
            value='numpy',
            disabled=True,
            description_tooltip='use scipy.optimize or numpy.polyfit',
            options=['scipy', 'numpy'],
            layout={
                'width': '95%',
                'height': '95%'
            })
        self.hs['FitConfigEdgeOptmzr drpdn'] = fit_edge_pars_GridspecLayout[
            0, 0:50]
        fit_edge_pars_GridspecLayout[0, 50:100] = widgets.Dropdown(
            description='fit func',
            disabled=True,
            description_tooltip='edge fitting functions',
            options=[2, 3, 4],
            value=3,
            layout={
                'width': '95%',
                'height': '95%'
            })
        self.hs['FitConfigEdgeFunc drpdn'] = fit_edge_pars_GridspecLayout[
            0, 50:100]
        fit_edge_pars_GridspecLayout[0, 100:150] = widgets.Checkbox(
            description='para bnd',
            value=False,
            disabled=True,
            description_tooltip=
            "if set boundaries to the peak fitting function's parameters",
            layout={
                'width': '95%',
                'height': '95%'
            })
        self.hs['FitConfigEdgeFitUseBnd chbx'] = fit_edge_pars_GridspecLayout[
            0, 100:150]

        fit_edge_pars_GridspecLayout[1, 0:50] = widgets.BoundedFloatText(
            description='p0',
            value=0,
            min=-1e5,
            ma=1e5,
            description_tooltip="fitting function variable 0",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitPars0 text'] = fit_edge_pars_GridspecLayout[
            1, 0:50]
        fit_edge_pars_GridspecLayout[1, 50:100] = widgets.BoundedFloatText(
            description='p1',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="fitting function variable 1",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitPars1 text'] = fit_edge_pars_GridspecLayout[
            1, 50:100]
        fit_edge_pars_GridspecLayout[1, 100:150] = widgets.BoundedFloatText(
            description='p2',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="fitting function variable 2",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitPars2 text'] = fit_edge_pars_GridspecLayout[
            1, 100:150]
        fit_edge_pars_GridspecLayout[1, 150:200] = widgets.BoundedFloatText(
            description='p3',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="fitting function variable 3",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitPars3 text'] = fit_edge_pars_GridspecLayout[
            1, 150:200]

        fit_edge_pars_GridspecLayout[2, 0:50] = widgets.BoundedFloatText(
            description='p4',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="fitting function variable 4",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitPars4 text'] = fit_edge_pars_GridspecLayout[
            2, 0:50]
        fit_edge_pars_GridspecLayout[2, 50:100] = widgets.Dropdown(
            description='p5',
            value='linear',
            options=['linear'],
            description_tooltip="fitting function variable 5",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitPars5 text'] = fit_edge_pars_GridspecLayout[
            2, 50:100]

        fit_edge_pars_GridspecLayout[3, 0:50] = widgets.Dropdown(
            description='jac',
            value='3-point',
            options=['2-point', '3-point', 'cs'],
            description_tooltip="",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitJac drpdn'] = fit_edge_pars_GridspecLayout[
            3, 0:50]
        fit_edge_pars_GridspecLayout[3, 50:100] = widgets.Dropdown(
            description='method',
            value='trf',
            disabled=True,
            options=['trf', 'dogbox', 'lm'],
            description_tooltip="",
            layout={
                'width': '95%',
                'height': '95%'
            })
        self.hs['FitConfigEdgeFitMeth drpdn'] = fit_edge_pars_GridspecLayout[
            3, 50:100]
        fit_edge_pars_GridspecLayout[3, 100:150] = widgets.BoundedFloatText(
            description='ftol',
            value=1e-7,
            min=0,
            max=1e-3,
            description_tooltip=
            "function value change tolerance for terminating the optimization process",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitFtol text'] = fit_edge_pars_GridspecLayout[
            3, 100:150]
        fit_edge_pars_GridspecLayout[3, 150:200] = widgets.BoundedFloatText(
            description='xtol',
            value=1e-7,
            min=0,
            max=1e-3,
            description_tooltip=
            "function parameter change tolerance for terminating the optimization process",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitXtol text'] = fit_edge_pars_GridspecLayout[
            3, 150:200]
        fit_edge_pars_GridspecLayout[4, 0:50] = widgets.BoundedFloatText(
            description='gtol',
            value=1e-7,
            min=0,
            max=1e-3,
            description_tooltip=
            "function gradient change tolerance for terminating the optimization process",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitGtol text'] = fit_edge_pars_GridspecLayout[
            4, 0:50]
        fit_edge_pars_GridspecLayout[4, 50:100] = widgets.BoundedIntText(
            description='ufac',
            value=50,
            min=1,
            max=100,
            description_tooltip=
            "upsampling factor to energy points in peak fitting",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitUfac text'] = fit_edge_pars_GridspecLayout[
            4, 50:100]

        fit_edge_pars_GridspecLayout[5, 0:50] = widgets.BoundedFloatText(
            description='p0 lb',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p0 lower bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitPars0Lb text'] = fit_edge_pars_GridspecLayout[
            5, 0:50]
        fit_edge_pars_GridspecLayout[5, 50:100] = widgets.BoundedFloatText(
            description='p0 ub',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p0 upper bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitPars0Ub text'] = fit_edge_pars_GridspecLayout[
            5, 50:100]
        fit_edge_pars_GridspecLayout[5, 100:150] = widgets.BoundedFloatText(
            description='p1 lb',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p1 lower bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitPars1Lb text'] = fit_edge_pars_GridspecLayout[
            5, 100:150]
        fit_edge_pars_GridspecLayout[5, 150:200] = widgets.BoundedFloatText(
            description='p1 ub',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p1 upper bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitPars1Ub text'] = fit_edge_pars_GridspecLayout[
            5, 150:200]

        fit_edge_pars_GridspecLayout[6, 0:50] = widgets.BoundedFloatText(
            description='p2 lb',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p2 lower bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitPars2Lb text'] = fit_edge_pars_GridspecLayout[
            6, 0:50]
        fit_edge_pars_GridspecLayout[6, 50:100] = widgets.BoundedFloatText(
            description='p3 ub',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p3 upper bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitPars2Ub text'] = fit_edge_pars_GridspecLayout[
            6, 50:100]
        fit_edge_pars_GridspecLayout[6, 100:150] = widgets.BoundedFloatText(
            description='p4 lb',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p4 lower bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitPars3Lb text'] = fit_edge_pars_GridspecLayout[
            6, 100:150]
        fit_edge_pars_GridspecLayout[6, 150:200] = widgets.BoundedFloatText(
            description='p4 ub',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p4 upper bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitPars3Ub text'] = fit_edge_pars_GridspecLayout[
            6, 150:200]

        fit_edge_pars_GridspecLayout[7, 0:50] = widgets.BoundedFloatText(
            description='p5 lb',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p5 lower bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitPars4Lb text'] = fit_edge_pars_GridspecLayout[
            7, 0:50]
        fit_edge_pars_GridspecLayout[7, 50:100] = widgets.BoundedFloatText(
            description='p5 ub',
            value=0,
            min=-1e5,
            max=1e5,
            description_tooltip="p5 upper bound",
            layout={
                'width': '95%',
                'height': '95%'
            },
            disabled=True)
        self.hs['FitConfigEdgeFitPars4Ub text'] = fit_edge_pars_GridspecLayout[
            7, 50:100]

        self.hs['FitConfigEdgeOptmzr drpdn'].observe(
            self.fit_config_edge_optmzr_drpdn_chg, names='value')
        self.hs['FitConfigEdgeFunc drpdn'].observe(
            self.fit_config_edge_func_drpdn_chg, names='value')
        self.hs['FitConfigEdgeFitUseBnd chbx'].observe(
            self.fit_config_edge_fit_bnd_chbx_chg, names='value')

        self.hs['FitConfigEdgePars box'] = fit_edge_pars_GridspecLayout
        self.hs['FitConfigEdgePars box'].children = [
            self.hs['FitConfigEdgeOptmzr drpdn'],
            self.hs['FitConfigEdgeFunc drpdn'],
            self.hs['FitConfigEdgeFitUseBnd chbx'],
            self.hs['FitConfigEdgeFitPars0 text'],
            self.hs['FitConfigEdgeFitPars1 text'],
            self.hs['FitConfigEdgeFitPars2 text'],
            self.hs['FitConfigEdgeFitPars3 text'],
            self.hs['FitConfigEdgeFitPars4 text'],
            self.hs['FitConfigEdgeFitPars5 text'],
            self.hs['FitConfigEdgeFitJac drpdn'],
            self.hs['FitConfigEdgeFitMeth drpdn'],
            self.hs['FitConfigEdgeFitFtol text'],
            self.hs['FitConfigEdgeFitXtol text'],
            self.hs['FitConfigEdgeFitGtol text'],
            self.hs['FitConfigEdgeFitUfac text'],
            self.hs['FitConfigEdgeFitPars0Lb text'],
            self.hs['FitConfigEdgeFitPars0Ub text'],
            self.hs['FitConfigEdgeFitPars1Lb text'],
            self.hs['FitConfigEdgeFitPars1Ub text'],
            self.hs['FitConfigEdgeFitPars2Lb text'],
            self.hs['FitConfigEdgeFitPars2Ub text'],
            self.hs['FitConfigEdgeFitPars3Lb text'],
            self.hs['FitConfigEdgeFitPars3Ub text'],
            self.hs['FitConfigEdgeFitPars4Lb text'],
            self.hs['FitConfigEdgeFitPars4Ub text']
        ]
        ## ## ## ## ## ## define edge fitting parameters -- end

        ## ## ## ## ## ## define saving setting -- start
        # layout = {'border':'3px solid #FFCC00', 'width':f'{0.96*self.form_sz[1]-98}px', 'height':f'{0.49*(self.form_sz[0]-128)}px'}
        layout = {
            'border': '3px solid #FFCC00',
            'width': 'auto',
            'height': 'auto'
        }
        self.hs['FitSavSetting box'] = widgets.HBox()
        self.hs['FitSavSetting box'].layout = layout
        fit_sav_setting_GridLayout = widgets.GridspecLayout(
            8,
            200,
            layout={
                "border": "3px solid #FFCC00",
                'width': '100%',
                'height': '100%'
            })
        fit_sav_setting_GridLayout[0:6, 0:100] = widgets.SelectMultiple(
            options=dat_dict.XANES_FULL_SAVE_ITEM_OPTIONS,
            value=['wl_fit_coef'],
            layout={
                'width': '95%',
                'height': '95%'
            },
            rows=10,
            description='Aval Items',
            disabled=True)
        self.hs['FitSavSettingOpt mulsel'] = fit_sav_setting_GridLayout[0:6,
                                                                     0:100]

        fit_sav_setting_GridLayout[0:6, 100:200] = widgets.SelectMultiple(
            options=dat_dict.XANES_FULL_SAVE_DEFAULT,
            value=[
                '',
            ],
            layout={
                'width': '95%',
                'height': '95%'
            },
            description='Selected:',
            disabled=True)
        self.hs['FitSavSettingSel mulsel'] = fit_sav_setting_GridLayout[
            0:6, 100:200]

        fit_sav_setting_GridLayout[7:8, 40:60] = widgets.Button(
            description='==>',
            disabled=True,
            description_tooltip='Select saving items')
        self.hs['FitSavSettingAdd btn'] = \
            fit_sav_setting_GridLayout[7:8, 40:60]
        self.hs['FitSavSettingAdd btn'].style.button_color = \
            'darkviolet'
        fit_sav_setting_GridLayout[7:8, 140:160] = widgets.Button(
            description='<==',
            disabled=True,
            description_tooltip='Remove saving items')
        self.hs['FitSavSettingRm btn'] = fit_sav_setting_GridLayout[7:8,
                                                                    140:160]
        self.hs['FitSavSettingRm btn'].style.button_color = 'darkviolet'
        self.hs['FitSavSettingAdd btn'].on_click(
            self.fit_sav_setting_add_btn_clk)
        self.hs['FitSavSettingRm btn'].on_click(
            self.fit_sav_setting_rm_btn_clk)

        self.hs['FitSavSetting box'] = fit_sav_setting_GridLayout
        self.hs['FitSavSetting box'].children = [
            self.hs['FitSavSettingOpt mulsel'],
            self.hs['FitSavSettingSel mulsel'],
            self.hs['FitSavSettingAdd btn'], self.hs['FitSavSettingRm btn']
        ]
        ## ## ## ## ## ## define saving setting -- end

        self.hs['FitItem tab'].children = [
            self.hs['FitItemConfig box'], self.hs['FitConfigWlPars box'],
            self.hs['FitConfigEdgePars box'], self.hs['FitSavSetting box']
        ]
        self.hs['FitItem tab'].set_title(0, 'config analysis')
        self.hs['FitItem tab'].set_title(1, 'fit whiteline params')
        self.hs['FitItem tab'].set_title(2, 'fit edge params')
        self.hs['FitItem tab'].set_title(3, 'saving setting')
        ## ## ## ## ## define fitting parameters -- send

        ## ## ## ## ## run xanes analysis -- start
        layout = {
            'border': '3px solid #FFCC00',
            'width': f'{1*self.form_sz[1]-98}px',
            'height': f'{0.07*(self.form_sz[0]-128)}px'
        }
        self.hs['FitRun box'] = widgets.HBox()
        self.hs['FitRun box'].layout = layout
        layout = {'width': '85%', 'height': '90%'}
        self.hs['FitRun text'] = widgets.Text(
            description=
            'please check your settings before run the analysis .. ',
            disabled=True)
        self.hs['FitRun text'].layout = layout
        layout = {'width': '15%', 'height': '90%'}
        self.hs['FitRun btn'] = widgets.Button(description='run',
                                               disabled=True)
        self.hs['FitRun btn'].layout = layout
        self.hs['FitRun btn'].style.button_color = 'darkviolet'

        self.hs['FitRun btn'].on_click(self.fit_run_btn_clk)

        self.hs['FitRun box'].children = [
            self.hs['FitRun text'], self.hs['FitRun btn']
        ]
        ## ## ## ## ## run xanes analysis -- end

        ## ## ## ## ## run analysis progress -- start
        layout = {
            'border': '3px solid #FFCC00',
            'width': f'{self.form_sz[1]-98}px',
            'height': f'{0.07*(self.form_sz[0]-128)}px'
        }
        self.hs['FitPrgr box'] = widgets.HBox()
        self.hs['FitPrgr box'].layout = layout
        layout = {'width': '100%', 'height': '90%'}
        self.hs['FitPrgr bar'] = widgets.IntProgress(
            value=0,
            min=0,
            max=10,
            step=1,
            description='Completing:',
            bar_style='info',  # 'success', 'info', 'warning', 'danger' or ''
            orientation='horizontal')
        self.hs['FitPrgr bar'].layout = layout
        self.hs['FitPrgr box'].children = [self.hs['FitPrgr bar']]
        ## ## ## ## ## run analysis progress -- end

        self.hs['Fitting form'].children = [
            self.hs['FitEngConfig box'], self.hs['FitItem tab'],
            self.hs['FitRun box'], self.hs['FitPrgr box']
        ]
        ## ## ## ## define functional widgets each tab in each sub-tab - analysis box in analysis&display TAB -- end
        self.bundle_fit_var_handles()
        self.boxes_logic()

    def boxes_logic(self):

        def compound_logic():
            if self.fit_wl_pos:
                if self.fit_wl_optimizer == 'scipy':
                    for ii in self.fit_wl_fit_func_arg_handles:
                        ii.disabled = False
                    for ii in self.fit_wl_optimizer_arg_handles:
                        ii.disabled = False
                    if self.fit_wl_fit_use_param_bnd:
                        for ii in self.fit_wl_fit_func_bnd_handles:
                            ii.disabled = False
                    else:
                        for ii in self.fit_wl_fit_func_bnd_handles:
                            ii.disabled = True
                    self.hs['FitConfigWlFitUseBnd chbx'].disabled = False
                    self.hs['FitConfigWlFunc drpdn'].options = dat_dict.XANES_PEAK_LINE_SHAPES
                    self.hs['FitConfigWlFunc drpdn'].description = 'peak func'
                    self.hs[
                        'FitConfigWlFunc drpdn'].description_tooltip = 'peak fitting functions'
                elif self.fit_wl_optimizer == 'numpy':
                    for ii in self.fit_wl_fit_func_arg_handles:
                        ii.disabled = True
                    for ii in self.fit_wl_optimizer_arg_handles:
                        ii.disabled = True
                    for ii in self.fit_wl_fit_func_bnd_handles:
                        ii.disabled = True
                    self.hs['FitConfigEdgeFitUfac text'].disabled = False
                    self.hs['FitConfigWlFitUseBnd chbx'].disabled = True
                    self.hs['FitConfigWlFunc drpdn'].options = NUMPY_FIT_ORDER
                    self.hs['FitConfigWlFunc drpdn'].description = 'order'
                    self.hs[
                        'FitConfigWlFunc drpdn'].description_tooltip = 'order of polynominal fitting function'

            if self.fit_mask_prev:
                if self.parent_h.gui_name == 'xanes3D':
                    self.hs['FitItemConfigFitPrvwSli sldr'].disabled = False
                else:
                    self.hs['FitItemConfigFitPrvwSli sldr'].disabled = True
                self.hs['FitItemConfigFitPrvwCal btn'].disabled = False
                if self.fit_flt_prev_maskit:
                    self.hs['FitItemConfigFitPrvwEMIt btn'].disabled = False
                    self.hs['FitItemConfigFitPrvwOMIt btn'].disabled = False
                    self.hs['FitItemConfigFitPrvwEMOMIt btn'].disabled = False
                else:
                    self.hs['FitItemConfigFitPrvwEMIt btn'].disabled = True
                    self.hs['FitItemConfigFitPrvwOMIt btn'].disabled = True
                    self.hs['FitItemConfigFitPrvwEMOMIt btn'].disabled = True
            else:
                self.hs['FitItemConfigFitPrvwSli sldr'].disabled = True
                self.hs['FitItemConfigFitPrvwCal btn'].disabled = True
                self.hs['FitItemConfigFitPrvwEMIt btn'].disabled = True
                self.hs['FitItemConfigFitPrvwOMIt btn'].disabled = True
                self.hs['FitItemConfigFitPrvwEMOMIt btn'].disabled = True

            if self.fit_lcf:
                boxes = [
                    'FitItemConfigLCF chbx', 'FitItemConfigLCFCnst chbx',
                    'FitItemConfigLCFNumSpec text',
                    'FitItemConfigLCFSelRef btn', 'FitItemConfigLCFRef list',
                    'FitItemConfigLCFRmRef btn'
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                if len(self.hs['FitItemConfigLCFRef list'].options
                       ) == self.fit_lcf_ref_num:
                    self.hs['FitItemConfigLCFSelRef btn'].disabled = True
                    self.fit_lcf_ref_set = True
                else:
                    self.hs['FitItemConfigLCFSelRef btn'].disabled = False
                    self.fit_lcf_ref_set = False
                if self.fit_lcf_ref_set:
                    self.hs['FitRun btn'].disabled = False
                else:
                    self.hs['FitRun btn'].disabled = True
            else:
                boxes = [
                    'FitItemConfigLCF chbx', 'FitItemConfigLCFCnst chbx',
                    'FitItemConfigLCFNumSpec text',
                    'FitItemConfigLCFSelRef btn', 'FitItemConfigLCFRef list',
                    'FitItemConfigLCFRmRef btn'
                ]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                self.hs['FitItemConfigLCF chbx'].disabled = False
                self.hs['FitRun btn'].disabled = False

            if self.fit_edge50_pos:
                if self.fit_edge_optimizer == 'scipy':
                    for ii in self.fit_edge_fit_func_arg_handles:
                        ii.disabled = False
                    for ii in self.fit_edge_optimizer_arg_handles:
                        ii.disabled = False
                    if self.fit_edge_fit_use_param_bnd:
                        for ii in self.fit_edge_fit_func_bnd_handles:
                            ii.disabled = False
                    else:
                        for ii in self.fit_edge_fit_func_bnd_handles:
                            ii.disabled = True
                    self.hs['FitConfigEdgeFitUseBnd chbx'].disabled = False
                    self.hs[
                        'FitConfigEdgeFunc drpdn'].options = dat_dict.XANES_EDGE_LINE_SHAPES
                    self.hs[
                        'FitConfigEdgeFunc drpdn'].description = 'peak func'
                    self.hs[
                        'FitConfigEdgeFunc drpdn'].description_tooltip = 'peak fitting functions'
                elif self.fit_edge_optimizer == 'numpy':
                    for ii in self.fit_edge_fit_func_arg_handles:
                        ii.disabled = True
                    for ii in self.fit_edge_optimizer_arg_handles:
                        ii.disabled = True
                    for ii in self.fit_edge_fit_func_bnd_handles:
                        ii.disabled = True
                    self.hs['FitConfigEdgeFitUfac text'].disabled = False
                    self.hs['FitConfigEdgeFitUseBnd chbx'].disabled = True
                    self.hs[
                        'FitConfigEdgeFunc drpdn'].options = NUMPY_FIT_ORDER
                    self.hs['FitConfigEdgeFunc drpdn'].description = 'order'
                    self.hs[
                        'FitConfigEdgeFunc drpdn'].description_tooltip = 'order of polynominal fitting function'

            if (not self.hs['FitItemConfigCalWlDir chbx'].value) and (
                    not self.hs['FitItemConfigFitWl chbx'].value):
                self.hs['FitItemConfigFitWl chbx'].value = True

        if not (self.parent_h.xanes_file_configured
                & self.parent_h.xanes_alignment_done):
            boxes = ['FitEngRag box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        else:
            if self.parent_h.xanes_fit_type == 'full':
                boxes = ['FitEngRag box']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            else:
                boxes = [
                    'FitEngRagEdgeEng text', 'FitEngRagPostEdgeStr text',
                    'FitEngRagPreEdgeEnd text', 'FitEngRagEdge0.5Str text',
                    'FitEngRagEdge0.5End text'
                ]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['FitEngRagWlFitStr text', 'FitEngRagWlFitEnd text']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            if self.xanes_fit_eng_config:
                if self.parent_h.xanes_fit_type == 'wl':
                    boxes = ['FitItem tab']
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=True,
                                         level=-1)
                    boxes = ['FitSavSetting box']
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=False,
                                         level=-1)
                else:
                    boxes = ['FitItem tab']
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=False,
                                         level=-1)
                if self.fit_wl_pos:
                    boxes = ['FitConfigWlPars box']
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=False,
                                         level=-1)
                else:
                    boxes = ['FitConfigWlPars box']
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=True,
                                         level=-1)
                if self.fit_edge50_pos:
                    boxes = ['FitConfigEdgePars box']
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=False,
                                         level=-1)
                else:
                    boxes = ['FitConfigEdgePars box']
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=True,
                                         level=-1)
                compound_logic()
            else:
                boxes = ['FitItem tab']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)

    def bundle_fit_var_handles(self):
        self.fit_wl_fit_func_arg_handles = [
            self.hs['FitConfigWlFitPars0 text'],
            self.hs['FitConfigWlFitPars1 text'],
            self.hs['FitConfigWlFitPars2 text'],
            self.hs['FitConfigWlFitPars3 text'],
            self.hs['FitConfigWlFitPars4 text'],
            self.hs['FitConfigWlFitPars5 text']
        ]

        self.fit_wl_optimizer_arg_handles = [
            self.hs['FitConfigWlFitJac drpdn'],
            self.hs['FitConfigWlFitMeth drpdn'],
            self.hs['FitConfigWlFitFtol text'],
            self.hs['FitConfigWlFitXtol text'],
            self.hs['FitConfigWlFitGtol text'],
            self.hs['FitConfigWlFitUfac text']
        ]

        self.fit_wl_fit_func_bnd_handles = [
            self.hs['FitConfigWlFitPars0Lb text'],
            self.hs['FitConfigWlFitPars0Ub text'],
            self.hs['FitConfigWlFitPars1Lb text'],
            self.hs['FitConfigWlFitPars1Ub text'],
            self.hs['FitConfigWlFitPars2Lb text'],
            self.hs['FitConfigWlFitPars2Ub text'],
            self.hs['FitConfigWlFitPars3Lb text'],
            self.hs['FitConfigWlFitPars3Ub text'],
            self.hs['FitConfigWlFitPars4Lb text'],
            self.hs['FitConfigWlFitPars4Ub text']
        ]

        self.fit_edge_fit_func_arg_handles = [
            self.hs['FitConfigEdgeFitPars0 text'],
            self.hs['FitConfigEdgeFitPars1 text'],
            self.hs['FitConfigEdgeFitPars2 text'],
            self.hs['FitConfigEdgeFitPars3 text'],
            self.hs['FitConfigEdgeFitPars4 text'],
            self.hs['FitConfigEdgeFitPars5 text']
        ]

        self.fit_edge_optimizer_arg_handles = [
            self.hs['FitConfigEdgeFitJac drpdn'],
            self.hs['FitConfigEdgeFitMeth drpdn'],
            self.hs['FitConfigEdgeFitFtol text'],
            self.hs['FitConfigEdgeFitXtol text'],
            self.hs['FitConfigEdgeFitGtol text'],
            self.hs['FitConfigEdgeFitUfac text']
        ]

        self.fit_edge_fit_func_bnd_handles = [
            self.hs['FitConfigEdgeFitPars0Lb text'],
            self.hs['FitConfigEdgeFitPars0Ub text'],
            self.hs['FitConfigEdgeFitPars1Lb text'],
            self.hs['FitConfigEdgeFitPars1Ub text'],
            self.hs['FitConfigEdgeFitPars2Lb text'],
            self.hs['FitConfigEdgeFitPars2Ub text'],
            self.hs['FitConfigEdgeFitPars3Lb text'],
            self.hs['FitConfigEdgeFitPars3Ub text'],
            self.hs['FitConfigEdgeFitPars4Lb text'],
            self.hs['FitConfigEdgeFitPars4Ub text']
        ]

    def fit_coef_num(self, ftype='wl_fit_coef'):
        if ftype == 'wl_fit_coef':
            if self.fit_wl_optimizer == 'scipy':
                return len(
                    list(dat_dict.XANES_PEAK_FIT_PARAM_DICT[self.fit_wl_fit_func].keys()))
            elif self.fit_wl_optimizer == 'numpy':
                return self.hs['FitConfigWlFunc drpdn'].value + 1
        elif ftype == 'edge_fit_coef':
            if self.fit_edge_optimizer == 'scipy':
                return len(
                    list(dat_dict.XANES_EDGE_FIT_PARAM_DICT[self.fit_edge_fit_func].keys()))
            elif self.fit_edge_optimizer == 'numpy':
                return self.hs['FitConfigEdgeFunc drpdn'].value + 1
        elif ftype == 'post_edge_fit_coef':
            return 2
        elif ftype == 'pre_edge_fit_coef':
            return 2

    def set_save_items(self):
        if self.parent_h.xanes_fit_type == 'wl':
            tem1 = set(deepcopy(sorted(dat_dict.XANES_WL_SAVE_ITEM_OPTIONS)))
            tem2 = set(deepcopy(sorted(dat_dict.XANES_WL_SAVE_DEFAULT)))
            if not self.fit_wl_pos:
                [
                    tem1.remove(ii)
                    for ii in ['wl_pos_fit', 'wl_fit_coef', 'wl_fit_err']
                    if ii in self.hs['FitSavSettingOpt mulsel'].options
                ]
                [
                    tem2.remove(ii)
                    for ii in ['wl_pos_fit', 'wl_fit_coef', 'wl_fit_err']
                    if ii in self.hs['FitSavSettingSel mulsel'].options
                ]
            if not self.find_wl_pos_dir:
                [
                    tem1.remove(ii) for ii in ['wl_pos_dir']
                    if ii in self.hs['FitSavSettingOpt mulsel'].options
                ]
                [
                    tem2.remove(ii) for ii in ['wl_pos_dir']
                    if ii in self.hs['FitSavSettingSel mulsel'].options
                ]
            self.hs['FitSavSettingOpt mulsel'].options = list(sorted(tem1))
            self.hs['FitSavSettingSel mulsel'].options = list(sorted(tem2))
            self.hs['FitSavSettingOpt mulsel'].value = ['centroid_of_eng']
            self.hs['FitSavSettingSel mulsel'].value = ['']
        elif self.parent_h.xanes_fit_type == 'full':
            tem1 = set(deepcopy(sorted(dat_dict.XANES_FULL_SAVE_ITEM_OPTIONS)))
            tem2 = set(deepcopy(sorted(dat_dict.XANES_FULL_SAVE_DEFAULT)))
            if not self.fit_wl_pos:
                [
                    tem1.remove(ii)
                    for ii in ['wl_pos_fit', 'wl_fit_coef', 'wl_fit_err']
                    if ii in self.hs['FitSavSettingOpt mulsel'].options
                ]
                [
                    tem2.remove(ii)
                    for ii in ['wl_pos_fit', 'wl_fit_coef', 'wl_fit_err']
                    if ii in self.hs['FitSavSettingSel mulsel'].options
                ]
            if not self.fit_edge50_pos:
                [
                    tem1.remove(ii) for ii in [
                        'edge50_pos_fit', 'edge_pos_fit', 'edge_fit_coef',
                        'edge_fit_err'
                    ] if ii in self.hs['FitSavSettingOpt mulsel'].options
                ]
                [
                    tem2.remove(ii) for ii in [
                        'edge50_pos_fit', 'edge_pos_fit', 'edge_fit_coef',
                        'edge_fit_err'
                    ] if ii in self.hs['FitSavSettingSel mulsel'].options
                ]
            if not self.find_wl_pos_dir:
                [
                    tem1.remove(ii) for ii in ['wl_pos_dir']
                    if ii in self.hs['FitSavSettingOpt mulsel'].options
                ]
                [
                    tem2.remove(ii) for ii in ['wl_pos_dir']
                    if ii in self.hs['FitSavSettingSel mulsel'].options
                ]
            if self.find_edge50_pos_dir:
                [
                    tem1.remove(ii) for ii in ['edge50_pos_dir']
                    if ii in self.hs['FitSavSettingOpt mulsel'].options
                ]
                [
                    tem2.remove(ii) for ii in ['edge50_pos_dir']
                    if ii in self.hs['FitSavSettingSel mulsel'].options
                ]
            if self.fit_lcf:
                [
                    tem1.remove(ii) for ii in ['lcf_fit', 'lcf_fit_err']
                    if ii in self.hs['FitSavSettingOpt mulsel'].options
                ]
                [
                    tem2.remove(ii) for ii in ['lcf_fit', 'lcf_fit_err']
                    if ii in self.hs['FitSavSettingSel mulsel'].options
                ]
            self.hs['FitSavSettingOpt mulsel'].options = list(sorted(tem1))
            self.hs['FitSavSettingSel mulsel'].options = list(sorted(tem2))
            self.hs['FitSavSettingOpt mulsel'].value = ['centroid_of_eng']
            self.hs['FitSavSettingSel mulsel'].value = ['']

    def set_xanes_analysis_eng_bounds(self):
        eng_list_len = self.parent_h.xanes_fit_eng_list.shape[0]
        if self.parent_h.xanes_fit_wl_fit_eng_e > self.parent_h.xanes_fit_eng_list.max(
        ):
            self.parent_h.xanes_fit_wl_fit_eng_e = self.parent_h.xanes_fit_eng_list.max(
            )
        elif self.parent_h.xanes_fit_wl_fit_eng_e > self.parent_h.xanes_fit_eng_list.min(
        ):
            self.parent_h.xanes_fit_wl_fit_eng_e = self.parent_h.xanes_fit_eng_list[
                int(eng_list_len / 2)]
        if self.parent_h.xanes_fit_wl_fit_eng_s < self.parent_h.xanes_fit_eng_list.min(
        ):
            self.parent_h.xanes_fit_wl_fit_eng_s = self.parent_h.xanes_fit_eng_list.min(
            )
        elif self.parent_h.xanes_fit_wl_fit_eng_s < self.parent_h.xanes_fit_eng_list.max(
        ):
            self.parent_h.xanes_fit_wl_fit_eng_s = self.parent_h.xanes_fit_eng_list[
                int(eng_list_len / 2)] - 1
        self.hs['FitEngRagWlFitEnd text'].min = 0
        self.hs[
            'FitEngRagWlFitEnd text'].max = self.parent_h.xanes_fit_eng_list.max(
            )
        self.hs[
            'FitEngRagWlFitEnd text'].min = self.parent_h.xanes_fit_eng_list.min(
            )
        self.hs['FitEngRagWlFitStr text'].min = 0
        self.hs[
            'FitEngRagWlFitStr text'].max = self.parent_h.xanes_fit_eng_list.max(
            )
        self.hs[
            'FitEngRagWlFitStr text'].min = self.parent_h.xanes_fit_eng_list.min(
            )

        if self.hs['FitEngRagOptn drpdn'].value == 'full':
            if ((self.parent_h.xanes_fit_edge_eng >
                 self.parent_h.xanes_fit_eng_list.max())
                    or (self.parent_h.xanes_fit_edge_eng <
                        self.parent_h.xanes_fit_eng_list.min())):
                self.parent_h.xanes_fit_edge_eng = self.parent_h.xanes_fit_eng_list[
                    int(eng_list_len / 2)]
            if self.parent_h.xanes_fit_edge_0p5_fit_e > self.parent_h.xanes_fit_eng_list.max(
            ):
                self.parent_h.xanes_fit_edge_0p5_fit_e = self.parent_h.xanes_fit_eng_list.max(
                )
            elif self.parent_h.xanes_fit_edge_0p5_fit_e < self.parent_h.xanes_fit_eng_list.min(
            ):
                self.parent_h.xanes_fit_edge_0p5_fit_e = self.parent_h.xanes_fit_eng_list[
                    int(eng_list_len / 2)]
            if self.parent_h.xanes_fit_edge_0p5_fit_s < self.parent_h.xanes_fit_eng_list.min(
            ):
                self.parent_h.xanes_fit_edge_0p5_fit_s = self.parent_h.xanes_fit_eng_list.min(
                )
            elif self.parent_h.xanes_fit_edge_0p5_fit_s > self.parent_h.xanes_fit_eng_list.max(
            ):
                self.parent_h.xanes_fit_edge_0p5_fit_s = self.parent_h.xanes_fit_eng_list[
                    int(eng_list_len / 2)] - 1
            self.hs['FitEngRagEdgeEng text'].min = 0
            self.hs[
                'FitEngRagEdgeEng text'].max = self.parent_h.xanes_fit_eng_list.max(
                )
            self.hs[
                'FitEngRagEdgeEng text'].min = self.parent_h.xanes_fit_eng_list.min(
                )

            self.hs['FitEngRagWlFitEnd text'].min = 0
            self.hs[
                'FitEngRagWlFitEnd text'].max = self.parent_h.xanes_fit_eng_list.max(
                )
            self.hs[
                'FitEngRagWlFitEnd text'].min = self.parent_h.xanes_fit_eng_list.min(
                )

            self.hs['FitEngRagWlFitStr text'].min = 0
            self.hs[
                'FitEngRagWlFitStr text'].max = self.parent_h.xanes_fit_eng_list.max(
                )
            self.hs[
                'FitEngRagWlFitStr text'].min = self.parent_h.xanes_fit_eng_list.min(
                )

            self.hs['FitEngRagEdge0.5End text'].min = 0
            self.hs[
                'FitEngRagEdge0.5End text'].max = self.parent_h.xanes_fit_eng_list.max(
                )
            self.hs[
                'FitEngRagEdge0.5End text'].min = self.parent_h.xanes_fit_eng_list.min(
                )

            self.hs['FitEngRagEdge0.5Str text'].min = 0
            self.hs[
                'FitEngRagEdge0.5Str text'].max = self.parent_h.xanes_fit_eng_list.max(
                )
            self.hs[
                'FitEngRagEdge0.5Str text'].min = self.parent_h.xanes_fit_eng_list.min(
                )

    def update_fit_params(self):
        self.parent_h.xanes_fit_type = self.hs['FitEngRagOptn drpdn'].value
        if self.parent_h.xanes_fit_type == 'wl':
            self.parent_h.xanes_fit_wl_fit_eng_s = self.hs[
                'FitEngRagWlFitStr text'].value
            self.parent_h.xanes_fit_wl_fit_eng_e = self.hs[
                'FitEngRagWlFitEnd text'].value

            self.fit_wl_pos = self.hs['FitItemConfigFitWl chbx'].value
            self.fit_use_flt_spec = self.hs['FitItemConfigFitSpec chbx'].value

            if self.fit_wl_pos:
                self.fit_wl_optimizer = self.hs[
                    'FitConfigWlOptmzr drpdn'].value
                self.fit_wl_fit_func = self.hs['FitConfigWlFunc drpdn'].value
            else:
                self.fit_wl_optimizer = None
                self.fit_wl_fit_func = None

            self.analysis_saving_items = set(
                self.hs['FitSavSettingSel mulsel'].options)
            self.analysis_saving_items.remove('')
        elif self.parent_h.xanes_fit_type == 'full':
            self.parent_h.xanes_fit_edge_eng = self.hs[
                'FitEngRagEdgeEng text'].value
            self.parent_h.xanes_fit_pre_edge_e = self.hs[
                'FitEngRagPreEdgeEnd text'].value
            self.parent_h.xanes_fit_post_edge_s = self.hs[
                'FitEngRagPostEdgeStr text'].value
            self.parent_h.xanes_fit_wl_fit_eng_s = self.hs[
                'FitEngRagWlFitStr text'].value
            self.parent_h.xanes_fit_wl_fit_eng_e = self.hs[
                'FitEngRagWlFitEnd text'].value
            self.parent_h.xanes_fit_edge_0p5_fit_s = self.hs[
                'FitEngRagEdge0.5Str text'].value
            self.parent_h.xanes_fit_edge_0p5_fit_e = self.hs[
                'FitEngRagEdge0.5End text'].value

            self.fit_wl_pos = self.hs['FitItemConfigFitWl chbx'].value
            self.fit_edge50_pos = self.hs['FitItemConfigFitEdge50% chbx'].value
            self.find_wl_pos_dir = self.hs['FitItemConfigCalWlDir chbx'].value
            self.find_edge50_pos_dir = self.hs[
                'FitItemConfigCalEdge50%Dir chbx'].value
            self.fit_use_flt_spec = self.hs['FitItemConfigFitSpec chbx'].value
            self.fit_mask_prev = self.hs['FitItemConfigFitPrvw chbx'].value

            if self.fit_wl_pos:
                self.fit_wl_optimizer = self.hs[
                    'FitConfigWlOptmzr drpdn'].value
                self.fit_wl_fit_func = self.hs['FitConfigWlFunc drpdn'].value
            else:
                self.fit_wl_optimizer = None
                self.fit_wl_fit_func = None
            if self.fit_edge50_pos:
                self.fit_edge_optimizer = self.hs[
                    'FitConfigEdgeOptmzr drpdn'].value
                self.fit_edge_fit_func = self.hs[
                    'FitConfigEdgeFunc drpdn'].value
            else:
                self.fit_edge_optimizer = None
                self.fit_edge_fit_func = None

            self.analysis_saving_items = set(
                self.hs['FitSavSettingSel mulsel'].options)
            self.analysis_saving_items.remove('')
        if self.parent_h.xanes_fit_spectrum is None:
            self.parent_h.xanes_fit_spectrum = np.ndarray(
                self.parent_h.xanes_fit_data_shape[1:], dtype=np.float32)
        if self.parent_h.gui_name == 'xanes3D':
            self.parent_h.update_xanes3D_config()
            json.dump(self.parent_h.xanes_config,
                      open(self.parent_h.xanes_save_trial_reg_config_filename,
                           'w'),
                      cls=NumpyArrayEncoder,
                      indent=4,
                      separators=(',', ': '))
        elif self.parent_h.gui_name == 'xanes2D':
            self.parent_h.update_xanes2D_config()
            json.dump(
                self.parent_h.xanes_config,
                open(self.parent_h.xanes_file_save_trial_reg_config_filename,
                     'w'),
                cls=NumpyArrayEncoder,
                indent=4,
                separators=(',', ': '))

    def fit_eng_rag_optn_drpdn(self, a):
        self.parent_h.xanes_fit_eng_configured = False
        self.xanes_fit_eng_config = False
        self.parent_h.xanes_fit_type = a['owner'].value
        self.hs[
            'FitEngRagEdgeEng text'].value = self.parent_h.xanes_fit_edge_eng
        self.hs[
            'FitEngRagWlFitEnd text'].value = self.parent_h.xanes_fit_wl_fit_eng_e
        self.hs[
            'FitEngRagWlFitStr text'].value = self.parent_h.xanes_fit_wl_fit_eng_s
        self.hs[
            'FitEngRagEdge0.5End text'].value = self.parent_h.xanes_fit_edge_0p5_fit_e
        self.hs[
            'FitEngRagEdge0.5Str text'].value = self.parent_h.xanes_fit_edge_0p5_fit_s
        self.hs[
            'FitEngRagPreEdgeEnd text'].value = self.parent_h.xanes_fit_pre_edge_e
        self.hs[
            'FitEngRagPostEdgeStr text'].value = self.parent_h.xanes_fit_post_edge_s
        self.set_xanes_analysis_eng_bounds()

        self.hs[
            'FitRun text'].value = 'please check your settings before run the analysis ...'
        self.fit_flt_prev_configed = False
        self.boxes_logic()

    def fit_eng_rag_edge_eng_text_chg(self, a):
        self.parent_h.xanes_fit_eng_configured = False
        self.xanes_fit_eng_config = False
        self.fit_flt_prev_configed = False
        self.parent_h.xanes_fit_edge_eng = self.hs[
            'FitEngRagEdgeEng text'].value
        self.hs[
            'FitRun text'].description = 'please check your settings before run the analysis ...'

    def fit_eng_rag_pre_edge_end_text_chg(self, a):
        self.parent_h.xanes_fit_eng_configured = False
        self.xanes_fit_eng_config = False
        self.fit_flt_prev_configed = False
        self.parent_h.xanes_fit_pre_edge_e = self.hs[
            'FitEngRagPreEdgeEnd text'].value
        self.hs[
            'FitRun text'].description = 'please check your settings before run the analysis ...'

    def fit_eng_rag_post_edge_str_text_chg(self, a):
        self.parent_h.xanes_fit_eng_configured = False
        self.xanes_fit_eng_config = False
        self.fit_flt_prev_configed = False
        self.parent_h.xanes_fit_post_edge_s = self.hs[
            'FitEngRagPostEdgeStr text'].value
        self.hs[
            'FitRun text'].description = 'please check your settings before run the analysis ...'

    def fit_eng_rag_wl_fit_str_text_chg(self, a):
        self.parent_h.xanes_fit_eng_configured = False
        self.xanes_fit_eng_config = False
        self.hs[
            'FitRun text'].value = 'please check your settings before run the analysis ...'
        self.parent_h.xanes_fit_wl_fit_eng_s = self.hs[
            'FitEngRagWlFitStr text'].value

    def fit_eng_rag_wl_fit_end_text_chg(self, a):
        self.parent_h.xanes_fit_eng_configured = False
        self.xanes_fit_eng_config = False
        self.hs[
            'FitRun text'].value = 'please check your settings before run the analysis ...'
        self.parent_h.xanes_fit_wl_fit_eng_e = self.hs[
            'FitEngRagWlFitEnd text'].value

    def fit_eng_rag_edge50_fit_str_text_chg(self, a):
        self.parent_h.xanes_fit_eng_configured = False
        self.xanes_fit_eng_config = False
        self.hs[
            'FitRun text'].value = 'please check your settings before run the analysis ...'
        self.parent_h.xanes_fit_edge_0p5_fit_s = self.hs[
            'FitEngRagEdge0.5Str text'].value

    def fit_eng_rag_edge50_fit_end_text_chg(self, a):
        self.parent_h.xanes_fit_eng_configured = False
        self.xanes_fit_eng_config = False
        self.hs[
            'FitRun text'].value = 'please check your settings before run the analysis ...'
        self.parent_h.xanes_fit_edge_0p5_fit_e = self.hs[
            'FitEngRagEdge0.5End text'].value

    def fit_eng_rag_cmf_btn_clk(self, a):
        self.update_fit_params()
        self.parent_h.xanes_fit_type = self.hs['FitEngRagOptn drpdn'].value
        if self.parent_h.xanes_fit_type == 'wl':
            self.parent_h.xanes_fit_wl_fit_eng_s = self.hs[
                'FitEngRagWlFitStr text'].value
            self.parent_h.xanes_fit_wl_fit_eng_e = self.hs[
                'FitEngRagWlFitEnd text'].value

            self.hs['FitItemConfigFitWl chbx'].value = True
            self.hs['FitItemConfigFitEdge50% chbx'].value = False
            self.hs['FitItemConfigFitSpec chbx'].value = False
            self.hs['FitItemConfigCalWlDir chbx'].value = False
            self.hs['FitItemConfigCalEdge50%Dir chbx'].value = False
            self.hs['FitItemConfigFitPrvw chbx'].value = False

            self.hs['FitSavSettingOpt mulsel'].options = dat_dict.XANES_WL_SAVE_ITEM_OPTIONS
            self.hs['FitSavSettingOpt mulsel'].value = ['wl_pos_fit']
            self.hs['FitSavSettingSel mulsel'].options = dat_dict.XANES_WL_SAVE_DEFAULT
            self.hs['FitSavSettingSel mulsel'].value = ['']
            self.analysis_saving_items = deepcopy(dat_dict.XANES_WL_SAVE_DEFAULT)
            self.analysis_saving_items.remove('')
        elif self.parent_h.xanes_fit_type == 'full':
            self.parent_h.xanes_fit_edge_eng = self.hs[
                'FitEngRagEdgeEng text'].value
            self.parent_h.xanes_fit_pre_edge_e = self.hs[
                'FitEngRagPreEdgeEnd text'].value
            self.parent_h.xanes_fit_post_edge_s = self.hs[
                'FitEngRagPostEdgeStr text'].value
            self.parent_h.xanes_fit_wl_fit_eng_s = self.hs[
                'FitEngRagWlFitStr text'].value
            self.parent_h.xanes_fit_wl_fit_eng_e = self.hs[
                'FitEngRagWlFitEnd text'].value
            self.parent_h.xanes_fit_edge_0p5_fit_s = self.hs[
                'FitEngRagEdge0.5Str text'].value
            self.parent_h.xanes_fit_edge_0p5_fit_e = self.hs[
                'FitEngRagEdge0.5End text'].value
            self.hs['FitItemConfigFitWl chbx'].value = True
            self.hs['FitItemConfigFitEdge50% chbx'].value = True
            self.hs['FitItemConfigFitSpec chbx'].value = False
            self.hs['FitItemConfigCalWlDir chbx'].value = False
            self.hs['FitItemConfigCalEdge50%Dir chbx'].value = False
            self.hs['FitItemConfigFitPrvw chbx'].value = False

            self.hs['FitSavSettingOpt mulsel'].options = dat_dict.XANES_FULL_SAVE_ITEM_OPTIONS
            self.hs['FitSavSettingOpt mulsel'].value = ['centroid_of_eng']
            self.hs['FitSavSettingSel mulsel'].options = dat_dict.XANES_FULL_SAVE_DEFAULT
            self.hs['FitSavSettingSel mulsel'].value = ['']
            self.analysis_saving_items = deepcopy(dat_dict.XANES_FULL_SAVE_DEFAULT)
            self.analysis_saving_items.remove('')
        if self.parent_h.xanes_fit_spectrum is None:
            self.parent_h.xanes_fit_spectrum = np.ndarray(
                self.parent_h.xanes_fit_data_shape[1:], dtype=np.float32)
        if self.parent_h.gui_name == 'xanes3D':
            self.parent_h.update_xanes3D_config()
            json.dump(self.parent_h.xanes_config,
                      open(self.parent_h.xanes_save_trial_reg_config_filename,
                           'w'),
                      cls=NumpyArrayEncoder,
                      indent=4,
                      separators=(',', ': '))
        elif self.parent_h.gui_name == 'xanes2D':
            self.parent_h.update_xanes2D_config()
            json.dump(
                self.parent_h.xanes_config,
                open(self.parent_h.xanes_file_save_trial_reg_config_filename,
                     'w'),
                cls=NumpyArrayEncoder,
                indent=4,
                separators=(',', ': '))

        for ii in dat_dict.XANES_PEAK_LINE_SHAPES:
            dat_dict.XANES_PEAK_FIT_PARAM_DICT[ii][1][1] = (
                self.parent_h.xanes_fit_wl_fit_eng_s +
                self.parent_h.xanes_fit_wl_fit_eng_e) / 2.
        boxes = ['FitItemConfig box', 'FitSavSetting box', 'FitRun box']
        enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)

        cnt = 0
        for ii in self.fit_edge_fit_func_arg_handles:
            ii.disabled = True
            ii.description = 'p' + str(cnt)
            cnt += 1
        self.hs['FitConfigWlOptmzr drpdn'].value = 'scipy'
        self.hs['FitConfigWlOptmzr drpdn'].value = 'numpy'
        self.hs['FitConfigWlFunc drpdn'].options = [2, 3, 4]
        self.hs['FitConfigWlFunc drpdn'].value = 2
        self.hs['FitConfigEdgeOptmzr drpdn'].value = 'numpy'
        self.hs['FitConfigEdgeFunc drpdn'].options = [2, 3, 4]
        self.hs['FitConfigEdgeFunc drpdn'].value = 3
        self.parent_h.xanes_fit_eng_configured = True
        self.xanes_fit_eng_config = True
        self.boxes_logic()
        self.set_save_items()

    def fit_config_fit_wl_chbx_chg(self, a):
        if a['owner'].value:
            a['owner'].value = True
            self.fit_wl_pos = True
            boxes = ['FitConfigWlPars box']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            self.hs['FitConfigWlOptmzr drpdn'].value = 'numpy'
            self.hs['FitConfigWlFunc drpdn'].options = [2, 3, 4]
            self.hs['FitConfigWlFunc drpdn'].value = 2
        else:
            self.fit_wl_pos = False
            boxes = ['FitConfigWlPars box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        self.boxes_logic()
        self.set_save_items()

    def fit_config_fit_edge50_chbx_chg(self, a):
        self.fit_edge50_pos = a['owner'].value
        self.boxes_logic()
        self.set_save_items()

    def fit_config_flt_spec_chbx_chg(self, a):
        self.fit_use_flt_spec = a['owner'].value
        self.set_save_items()

    def fit_config_cal_wl_dir_chbx_chg(self, a):
        self.find_wl_pos_dir = a['owner'].value
        self.set_save_items()
        self.boxes_logic()

    def fit_config_cal_edge50_dir_chbx_chg(self, a):
        self.find_edge50_pos_dir = a['owner'].value
        self.set_save_items()

    def fit_config_fit_edge_jump_thres_sldr_chg(self, a):
        self.fit_flt_prev_maskit = False
        self.parent_h.xanes_fit_edge_jump_thres = a['owner'].value
        self.boxes_logic()

    def fit_config_fit_edge_ofst_thres_sldr_chg(self, a):
        self.fit_flt_prev_maskit = False
        self.parent_h.xanes_fit_edge_offset_thres = a['owner'].value
        self.boxes_logic()

    def fit_config_bin_fac_text_chg(self, a):
        self.fit_img_bin_fac = a['owner'].value
        self.boxes_logic()

    def fit_config_mask_prvw_chbx_chg(self, a):
        if a['owner'].value:
            self.fit_mask_prev = True
            if self.parent_h.gui_name == 'xanes3D':
                self.hs[
                    'FitItemConfigFitPrvwSli sldr'].max = self.parent_h.xanes_fit_data_shape[
                        2] - 1
                self.hs['FitItemConfigFitPrvwSli sldr'].disabled = False
                with h5py.File(self.parent_h.xanes_save_trial_reg_filename,
                               'r') as f:
                    self.spec = f[
                        '/registration_results/reg_results/registered_xanes3D'][:, :,
                                                                                self
                                                                                .
                                                                                fit_flt_prev_sli, :]
            else:
                self.hs['FitItemConfigFitPrvwSli sldr'].disabled = True
                with h5py.File(self.parent_h.xanes_save_trial_reg_filename,
                               'r') as f:
                    self.spec = f[
                        '/registration_results/reg_results/registered_xanes2D'][:]
        else:
            self.fit_mask_prev = False
            self.hs['FitItemConfigFitPrvwSli sldr'].disabled = True
        self.boxes_logic()

    def fit_config_flt_prvw_sli_sldr_chg(self, a):
        self.fit_flt_prev_sli = a['owner'].value
        if self.parent_h.gui_name == 'xanes3D':
            with h5py.File(self.parent_h.xanes_save_trial_reg_filename,
                           'r') as f:
                self.spec = f[
                    '/registration_results/reg_results/registered_xanes3D'][:, :,
                                                                            self.fit_flt_prev_sli, :]
            self.fit_flt_prev_configed = False

    def fit_config_flt_prvw_calc_btn_clk(self, a):
        tmp_file = os.path.join(self.global_h.tmp_dir, 'xanes2D_tmp.h5')
        if self.fit_flt_prev_configed:
            with h5py.File(tmp_file, 'r') as f:
                self.e0_idx = f['/fitting/e0_idx'][:]
                self.pre = f['/fitting/pre'][:]
                self.post = f['/fitting/post'][:]
            self.edge_jump_mask = np.squeeze(
                (self.post[self.e0_idx] - self.pre[self.e0_idx]
                 ) > self.parent_h.xanes_fit_edge_jump_thres *
                self.fit_flt_prev_xana.pre_edge_sd_map).astype(np.int8)
            self.fitted_edge_mask = np.any(
                (self.post - self.pre) >
                self.parent_h.xanes_fit_edge_offset_thres *
                self.fit_flt_prev_xana.pre_edge_sd_map,
                axis=0).astype(np.int8)
        else:
            self.pre_es_idx = xm.index_of(self.parent_h.xanes_fit_eng_list,
                                          self.parent_h.xanes_fit_eng_list[0])
            self.pre_ee_idx = xm.index_of(self.parent_h.xanes_fit_eng_list,
                                          self.parent_h.xanes_fit_pre_edge_e)
            self.post_es_idx = xm.index_of(self.parent_h.xanes_fit_eng_list,
                                           self.parent_h.xanes_fit_post_edge_s)
            self.post_ee_idx = xm.index_of(
                self.parent_h.xanes_fit_eng_list,
                self.parent_h.xanes_fit_eng_list[-1])

            ln = 0
            code = {}
            code[
                ln] = f"import TXM_Sandbox.utils.xanes_analysis as xa"
            ln += 1
            code[
                ln] = f"import TXM_Sandbox.utils.xanes_math as xm"
            ln += 1
            code[
                ln] = f"import numpy as np"
            ln += 1
            code[
                ln] = f"import h5py"
            ln += 1
            code[
                ln] = f""
            ln += 1
            code[
                ln] = f"with h5py.File('{self.parent_h.xanes_save_trial_reg_filename}', 'r') as f:"
            ln += 1
            code[
                ln] = f"    if '{self.parent_h.gui_name}' == 'xanes3D':"
            ln += 1
            code[
                ln] = f"        spec = f['/registration_results/reg_results/registered_xanes3D'][:, :, {self.fit_flt_prev_sli}, :]"
            ln += 1
            code[
                ln] = f"    else:"
            ln += 1
            code[
                ln] = f"        spec = f['/registration_results/reg_results/registered_xanes2D'][:]"
            ln += 1
            code[
                ln] = f"    eng_list = f['/registration_results/reg_results/eng_list'][:]"
            ln += 1
            code[
                ln] = f"fit_flt_prev_xana = xa.xanes_analysis(spec, eng_list,\
                                                                {self.parent_h.xanes_fit_edge_eng},\
                                                                pre_ee={self.parent_h.xanes_fit_pre_edge_e},\
                                                                post_es={self.parent_h.xanes_fit_post_edge_s},\
                                                                edge_jump_threshold={self.parent_h.xanes_fit_edge_jump_thres},\
                                                                pre_edge_threshold={self.parent_h.xanes_fit_edge_offset_thres})"

            ln += 1
            code[
                ln] = f"fit_flt_prev_xana.cal_pre_edge_sd()"
            ln += 1
            code[
                ln] = f"fit_flt_prev_xana.cal_post_edge_sd()"
            ln += 1
            code[
                ln] = f"fit_flt_prev_xana.cal_pre_edge_mean()"
            ln += 1
            code[
                ln] = f"fit_flt_prev_xana.cal_post_edge_mean()"
            ln += 1
            code[
                ln] = f"e0_idx = xm.index_of(fit_flt_prev_xana.eng, fit_flt_prev_xana.preset_edge_eng)"
            ln += 1
            code[
                ln] = f"pre = fit_flt_prev_xana.cal_pre_edge_fit()"
            ln += 1
            code[
                ln] = f"post = fit_flt_prev_xana.cal_post_edge_fit()"
            ln += 1
            code[
                ln] = f"edge_jump_mask = np.squeeze((post[e0_idx] - pre[e0_idx]) > \
                                                     fit_flt_prev_xana.edge_jump_thres * \
                                                     fit_flt_prev_xana.pre_edge_sd_map).astype(np.int8)"

            ln += 1
            code[
                ln] = f"fitted_edge_mask = np.any((post - pre) >\
                                                    fit_flt_prev_xana.fitted_edge_thres * \
                                                    fit_flt_prev_xana.pre_edge_sd_map, axis=0).astype(np.int8)"

            ln += 1
            code[
                ln] = f"with h5py.File('{tmp_file}', 'a') as f:"
            ln += 1
            code[
                ln] = f"    if '/fitting' not in f:"
            ln += 1
            code[
                ln] = f"        g0 = f.create_group('fitting')"
            ln += 1
            code[
                ln] = f"        g0.create_dataset('e0_idx', data=e0_idx)"
            ln += 1
            code[
                ln] = f"        g0.create_dataset('pre', data=pre)"
            ln += 1
            code[
                ln] = f"        g0.create_dataset('post', data=post)"
            ln += 1
            code[
                ln] = f"        g0.create_dataset('edge_jump_mask', data=edge_jump_mask)"
            ln += 1
            code[
                ln] = f"        g0.create_dataset('fitted_edge_mask', data=fitted_edge_mask)"
            ln += 1
            code[
                ln] = f"    else:"
            ln += 1
            code[
                ln] = f"        del f['/fitting']"
            ln += 1
            code[
                ln] = f"        g0 = f.create_group('fitting')"
            ln += 1
            code[
                ln] = f"        g0.create_dataset('e0_idx', data=e0_idx)"
            ln += 1
            code[
                ln] = f"        g0.create_dataset('pre', data=pre)"
            ln += 1
            code[
                ln] = f"        g0.create_dataset('post', data=post)"
            ln += 1
            code[
                ln] = f"        g0.create_dataset('edge_jump_mask', data=edge_jump_mask)"
            ln += 1
            code[
                ln] = f"        g0.create_dataset('fitted_edge_mask', data=fitted_edge_mask)"
            ln += 1

            gen_external_py_script(self.xanes_mask_external_command_name, code)
            sig = os.system(
                f"python '{self.xanes_mask_external_command_name}'")
            if sig == 0:
                print('XANES2D mask is generated.')
                with h5py.File(tmp_file, 'r') as f:
                    self.e0_idx = f['/fitting/e0_idx'][()]
                    self.pre = f['/fitting/pre'][:]
                    self.post = f['/fitting/post'][:]

                    self.edge_jump_mask = f['/fitting/edge_jump_mask'][:]
                    self.fitted_edge_mask = f['/fitting/fitted_edge_mask'][:]
                self.fit_flt_prev_configed = True
                if self.parent_h.gui_name == 'xanes3D':
                    data_state, viewer_state = fiji_viewer_state(
                        self.global_h,
                        self,
                        viewer_name='xanes3D_fit_jump_flt_viewer')
                    if not viewer_state:
                        fiji_viewer_on(
                            self.global_h,
                            self,
                            viewer_name='xanes3D_fit_jump_flt_viewer')
                    self.global_h.xanes2D_fiji_windows[
                        'xanes3D_fit_jump_flt_viewer']['ip'].setImage(
                            self.global_h.ij.convert().convert(
                                self.global_h.ij.dataset().create(
                                    self.global_h.ij.py.to_java(
                                        self.edge_jump_mask)),
                                self.global_h.ImagePlusClass))
                    self.global_h.ij.py.run_macro("""setMinAndMax(0, 1)""")

                    data_state, viewer_state = fiji_viewer_state(
                        self.global_h,
                        self,
                        viewer_name='xanes3D_fit_thres_flt_viewer')
                    if not viewer_state:
                        fiji_viewer_on(
                            self.global_h,
                            self,
                            viewer_name='xanes3D_fit_thres_flt_viewer')
                    self.global_h.xanes2D_fiji_windows[
                        'xanes3D_fit_thres_flt_viewer']['ip'].setImage(
                            self.global_h.ij.convert().convert(
                                self.global_h.ij.dataset().create(
                                    self.global_h.ij.py.to_java(
                                        self.fitted_edge_mask)),
                                self.global_h.ImagePlusClass))
                    self.global_h.ij.py.run_macro("""setMinAndMax(0, 1)""")
                else:
                    data_state, viewer_state = fiji_viewer_state(
                        self.global_h,
                        self,
                        viewer_name='xanes2D_fit_jump_flt_viewer')
                    if not viewer_state:
                        fiji_viewer_on(
                            self.global_h,
                            self,
                            viewer_name='xanes2D_fit_jump_flt_viewer')
                    self.global_h.xanes2D_fiji_windows[
                        'xanes2D_fit_jump_flt_viewer']['ip'].setImage(
                            self.global_h.ij.convert().convert(
                                self.global_h.ij.dataset().create(
                                    self.global_h.ij.py.to_java(
                                        self.edge_jump_mask)),
                                self.global_h.ImagePlusClass))
                    self.global_h.ij.py.run_macro("""setMinAndMax(0, 1)""")

                    data_state, viewer_state = fiji_viewer_state(
                        self.global_h,
                        self,
                        viewer_name='xanes2D_fit_thres_flt_viewer')
                    if not viewer_state:
                        fiji_viewer_on(
                            self.global_h,
                            self,
                            viewer_name='xanes2D_fit_thres_flt_viewer')
                    self.global_h.xanes2D_fiji_windows[
                        'xanes2D_fit_thres_flt_viewer']['ip'].setImage(
                            self.global_h.ij.convert().convert(
                                self.global_h.ij.dataset().create(
                                    self.global_h.ij.py.to_java(
                                        self.fitted_edge_mask)),
                                self.global_h.ImagePlusClass))
                    self.global_h.ij.py.run_macro("""setMinAndMax(0, 1)""")
                self.fit_flt_prev_maskit = True
            else:
                print('Something wrong when generating XANES2D masks ...')
                self.fit_flt_prev_configed = False
                self.fit_flt_prev_maskit = False
        self.boxes_logic()

    def fit_config_flt_prvw_em_it_btn_clk(self, a):
        if self.parent_h.gui_name == 'xanes3D':
            data_state, viewer_state = fiji_viewer_state(
                self.global_h, self, viewer_name='xanes3D_fit_maskit_viewer')
            if not viewer_state:
                fiji_viewer_on(self.global_h,
                               self,
                               viewer_name='xanes3D_fit_maskit_viewer')
            self.global_h.xanes2D_fiji_windows['xanes3D_fit_maskit_viewer'][
                'ip'].setImage(self.global_h.ij.convert().convert(
                    self.global_h.ij.dataset().create(
                        self.global_h.ij.py.to_java(
                            self.edge_jump_mask *
                            self.spec[int(self.spec.shape[0] / 2)])),
                    self.global_h.ImagePlusClass))
            self.global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")
        else:
            data_state, viewer_state = fiji_viewer_state(
                self.global_h, self, viewer_name='xanes2D_fit_maskit_viewer')
            if not viewer_state:
                fiji_viewer_on(self.global_h,
                               self,
                               viewer_name='xanes2D_fit_maskit_viewer')
            self.global_h.xanes2D_fiji_windows['xanes2D_fit_maskit_viewer'][
                'ip'].setImage(self.global_h.ij.convert().convert(
                    self.global_h.ij.dataset().create(
                        self.global_h.ij.py.to_java(
                            self.edge_jump_mask *
                            self.spec[int(self.spec.shape[0] / 2)])),
                    self.global_h.ImagePlusClass))
            self.global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")

    def fit_config_flt_prvw_om_it_btn_clk(self, a):
        if self.parent_h.gui_name == 'xanes3D':
            data_state, viewer_state = fiji_viewer_state(
                self.global_h, self, viewer_name='xanes3D_fit_maskit_viewer')
            if not viewer_state:
                fiji_viewer_on(self.global_h,
                               self,
                               viewer_name='xanes3D_fit_maskit_viewer')
            self.global_h.xanes2D_fiji_windows['xanes3D_fit_maskit_viewer'][
                'ip'].setImage(self.global_h.ij.convert().convert(
                    self.global_h.ij.dataset().create(
                        self.global_h.ij.py.to_java(
                            self.fitted_edge_mask *
                            self.spec[int(self.spec.shape[0] / 2)])),
                    self.global_h.ImagePlusClass))
            self.global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")
        else:
            data_state, viewer_state = fiji_viewer_state(
                self.global_h, self, viewer_name='xanes2D_fit_maskit_viewer')
            if not viewer_state:
                fiji_viewer_on(self.global_h,
                               self,
                               viewer_name='xanes2D_fit_maskit_viewer')
            self.global_h.xanes2D_fiji_windows['xanes2D_fit_maskit_viewer'][
                'ip'].setImage(self.global_h.ij.convert().convert(
                    self.global_h.ij.dataset().create(
                        self.global_h.ij.py.to_java(
                            self.fitted_edge_mask *
                            self.spec[int(self.spec.shape[0] / 2)])),
                    self.global_h.ImagePlusClass))
            self.global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")

    def fit_config_flt_prvw_emom_it_btn_clk(self, a):
        if self.parent_h.gui_name == 'xanes3D':
            data_state, viewer_state = fiji_viewer_state(
                self.global_h, self, viewer_name='xanes3D_fit_maskit_viewer')
            if not viewer_state:
                fiji_viewer_on(self.global_h,
                               self,
                               viewer_name='xanes3D_fit_maskit_viewer')
            self.global_h.xanes2D_fiji_windows['xanes3D_fit_maskit_viewer'][
                'ip'].setImage(self.global_h.ij.convert().convert(
                    self.global_h.ij.dataset().create(
                        self.global_h.ij.py.to_java(
                            self.edge_jump_mask * self.fitted_edge_mask *
                            self.spec[int(self.spec.shape[0] / 2)])),
                    self.global_h.ImagePlusClass))
            self.global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")
        else:
            data_state, viewer_state = fiji_viewer_state(
                self.global_h, self, viewer_name='xanes2D_fit_maskit_viewer')
            if not viewer_state:
                fiji_viewer_on(self.global_h,
                               self,
                               viewer_name='xanes2D_fit_maskit_viewer')
            self.global_h.xanes2D_fiji_windows['xanes2D_fit_maskit_viewer'][
                'ip'].setImage(self.global_h.ij.convert().convert(
                    self.global_h.ij.dataset().create(
                        self.global_h.ij.py.to_java(
                            self.edge_jump_mask * self.fitted_edge_mask *
                            self.spec[int(self.spec.shape[0] / 2)])),
                    self.global_h.ImagePlusClass))
            self.global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")

    def fit_config_lcf_chbx_chg(self, a):
        if a['owner'].value:
            self.fit_lcf = True
            if not self.fit_lcf_ref_set:
                self.hs['FitRun btn'].disabled = True
        else:
            self.fit_lcf = False
            self.hs['FitRun btn'].disabled = False
        self.set_save_items()
        self.boxes_logic()

    def fit_config_lcf_cnst_chbx_chg(self, a):
        self.fit_lcf_constr = a['owner'].value

    def fit_config_num_spec_text_chg(self, a):
        self.fit_lcf_ref_num = a['owner'].value
        self.fit_lcf_ref_spec = pd.DataFrame(
            columns=['fpath', 'fname', 'eng', 'mu'])
        self.hs['FitItemConfigLCFRef list'].options = []
        self.hs['FitItemConfigLCFSelRef btn'].disabled = False
        self.boxes_logic()

    def fit_config_sel_ref_btn_clk(self, a):
        if len(self.hs['FitItemConfigLCFRef list'].options
               ) < self.fit_lcf_ref_num:
            if len(a.files[0]) != 0:
                spec = np.loadtxt(a.files[0])
                if spec[:, 0].max() < 50:
                    spec[:, 0] *= 1000
                df = pd.DataFrame([[
                    os.path.dirname(a.files[0]),
                    os.path.basename(a.files[0]), spec[:, 0], spec[:, 1]
                ]],
                                  columns=['fpath', 'fname', 'eng', 'mu'])
                self.fit_lcf_ref_spec = self.fit_lcf_ref_spec.append(
                    df, ignore_index=True)
                self.hs[
                    'FitItemConfigLCFRef list'].options = self.fit_lcf_ref_spec[
                        'fname']
                update_json_content(self.global_h.GUI_cfg_file, {
                    'xanes_ref_d':
                    os.path.dirname(os.path.abspath(a.files[0]))
                })
                self.hs[
                    'FitItemConfigLCFSelRef btn'].initialdir = os.path.dirname(
                        os.path.abspath(a.files[0]))
        self.boxes_logic()

    def fit_config_ref_spec_list_chg(self, a):
        if len(self.hs['FitItemConfigLCFRef list'].options
               ) < self.fit_lcf_ref_num:
            self.hs['FitItemConfigLCFSelRef btn'].disabled = False
        self.boxes_logic()

    def fit_config_ref_rm_btn_clk(self, a):
        self.fit_lcf_ref_spec = \
            self.fit_lcf_ref_spec.drop(list(self.hs['FitItemConfigLCFRef list'].index)).reset_index(drop=True)
        tem = list(self.hs['FitItemConfigLCFRef list'].options)
        for ii in self.hs['FitItemConfigLCFRef list'].index:
            del tem[ii]
        self.hs['FitItemConfigLCFRef list'].options = tem
        self.boxes_logic()

    def fit_config_wl_optmzr_drpdn_chg(self, a):
        self.fit_wl_optimizer = a['owner'].value
        self.boxes_logic()

    def fit_config_wl_func_drpdn_chg(self, a):
        self.fit_wl_fit_func = a['owner'].value
        cnt = 0
        for ii in self.fit_wl_fit_func_arg_handles:
            ii.disabled = True
            ii.description = 'p' + str(cnt)
            cnt += 1
        if self.fit_wl_optimizer == 'scipy':
            for ii in sorted(dat_dict.XANES_PEAK_FIT_PARAM_DICT[self.fit_wl_fit_func].keys()):
                self.fit_wl_fit_func_arg_handles[ii].disabled = False
                self.fit_wl_fit_func_arg_handles[
                    ii].description = dat_dict.XANES_PEAK_FIT_PARAM_DICT[
                        self.fit_wl_fit_func][ii][0]
                self.fit_wl_fit_func_arg_handles[
                    ii].value = dat_dict.XANES_PEAK_FIT_PARAM_DICT[
                        self.fit_wl_fit_func][ii][1]
                self.fit_wl_fit_func_arg_handles[
                    ii].description_tooltip = dat_dict.XANES_PEAK_FIT_PARAM_DICT[
                        self.fit_wl_fit_func][ii][2]
        if self.fit_wl_fit_func == "rectangle":
            self.fit_wl_fit_func_arg_handles[5].options = [
                'linear', 'atan', 'erf', 'logisitic'
            ]
        if self.fit_wl_fit_func == "step":
            self.fit_wl_fit_func_arg_handles[5].options = [
                'linear', 'atan', 'erf', 'logisitic'
            ]

    def fit_config_wl_fit_bnd_chbx_chg(self, a):
        self.fit_wl_fit_use_param_bnd = a['owner'].value
        if self.fit_wl_fit_use_param_bnd:
            for ii in self.fit_wl_fit_func_bnd_handles:
                ii.disabled = False
            for ii in [
                    "gaussian", "lorentzian", "damped_oscillator", "lognormal",
                    "students_t", "voigt", "split_lorentzian", "pvoigt",
                    "moffat", "pearson7", "breit_wigner", "dho", "expgaussian",
                    "donaich", "skewed_gaussian", "step", "skewed_voigt"
            ]:
                dat_dict.XANES_PEAK_FIT_PARAM_BND_DICT[ii][1][1] = (
                    self.parent_h.xanes_fit_wl_fit_eng_s +
                    self.parent_h.xanes_fit_wl_fit_eng_e) / 2.
        else:
            for ii in self.fit_wl_fit_func_bnd_handles:
                ii.disabled = True

    def fit_config_edge_optmzr_drpdn_chg(self, a):
        self.fit_edge_optimizer = a['owner'].value
        self.boxes_logic()

    def fit_config_edge_func_drpdn_chg(self, a):
        self.fit_edge_fit_func = a['owner'].value
        cnt = 0
        for ii in self.fit_edge_fit_func_arg_handles:
            ii.disabled = True
            ii.description = 'p' + str(cnt)
            cnt += 1
        if self.fit_edge_optimizer == 'scipy':
            for ii in sorted(
                    dat_dict.XANES_PEAK_FIT_PARAM_DICT[self.fit_edge_fit_func].keys()):
                self.fit_edge_fit_func_arg_handles[ii].disabled = False
                self.fit_edge_fit_func_arg_handles[
                    ii].description = dat_dict.XANES_PEAK_FIT_PARAM_DICT[
                        self.fit_edge_fit_func][ii][0]
                self.fit_edge_fit_func_arg_handles[
                    ii].value = dat_dict.XANES_PEAK_FIT_PARAM_DICT[
                        self.fit_edge_fit_func][ii][1]
                self.fit_edge_fit_func_arg_handles[
                    ii].description_tooltip = dat_dict.XANES_PEAK_FIT_PARAM_DICT[
                        self.fit_edge_fit_func][ii][2]
        if self.fit_edge_fit_func == "rectangle":
            self.fit_edge_fit_func_arg_handles[5].options = [
                'linear', 'atan', 'erf', 'logisitic'
            ]
        if self.fit_edge_fit_func == "step":
            self.fit_edge_fit_func_arg_handles[5].options = [
                'linear', 'atan', 'erf', 'logisitic'
            ]

    def fit_config_edge_fit_bnd_chbx_chg(self, a):
        self.fit_edge_fit_use_param_bnd = a['owner'].value
        if self.fit_edge_fit_use_param_bnd:
            for ii in self.fit_edge_fit_func_bnd_handles:
                ii.disabled = False
            for ii in [
                    "gaussian", "lorentzian", "voigt", "split_lorentzian",
                    "pvoigt", "expgaussian", "skewed_gaussian", "skewed_voigt"
            ]:
                dat_dict.XANES_PEAK_FIT_PARAM_BND_DICT[ii][1][1] = (
                    self.parent_h.xanes_fit_wl_fit_eng_s +
                    self.parent_h.xanes_fit_wl_fit_eng_e) / 2.
        else:
            for ii in self.fit_edge_fit_func_bnd_handles:
                ii.disabled = True

    def fit_sav_setting_add_btn_clk(self, a):
        tem = set(self.hs['FitSavSettingSel mulsel'].options)
        [tem.add(ii) for ii in self.hs['FitSavSettingOpt mulsel'].value]
        tem.add('')
        self.hs['FitSavSettingSel mulsel'].options = list(sorted(tem))
        self.hs['FitSavSettingSel mulsel'].value = ['']
        self.analysis_saving_items = list(tem)

    def fit_sav_setting_rm_btn_clk(self, a):
        tem = set(self.hs['FitSavSettingSel mulsel'].options)
        [
            tem.remove(ii) for ii in self.hs['FitSavSettingSel mulsel'].value
            if ii != ''
        ]
        tem.add('')
        self.hs['FitSavSettingSel mulsel'].options = list(sorted(tem))
        self.hs['FitSavSettingSel mulsel'].value = ['']
        self.analysis_saving_items = list(tem)

    def fit_run_btn_clk(self, a):
        try:
            fiji_viewer_off(self.parent_h.global_h, viewer_name='all')
        except:
            pass
        self.analysis_saving_items = set()
        for ii in self.hs['FitSavSettingSel mulsel'].options:
            if ii != '':
                self.analysis_saving_items.add(ii)
        self.update_fit_params()

        if self.fit_wl_pos:
            wl_fvars = []
            wl_bnds = []
            wl_params = {}
            wl_params['optimizer'] = self.fit_wl_optimizer
            wl_params['ftype'] = 'wl'
            if self.parent_h.xanes_fit_type == 'full':
                wl_params['on'] = 'norm'
            else:
                wl_params['on'] = 'raw'
            if self.fit_wl_optimizer == 'scipy':
                for ii in sorted(
                        dat_dict.XANES_PEAK_FIT_PARAM_DICT[self.fit_wl_fit_func].keys()):
                    wl_fvars.append(self.fit_wl_fit_func_arg_handles[ii].value)
                if self.fit_wl_fit_func in ["sine", "expsine"]:
                    wl_fvars[2] = 0
                elif self.fit_wl_fit_func == "rectangle":
                    wl_fvars[1] = wl_fvars[3] = 0
                else:
                    wl_fvars[1] = 0
                if self.fit_wl_fit_use_param_bnd:
                    for ii in sorted(
                            dat_dict.XANES_PEAK_FIT_PARAM_DICT[self.fit_wl_fit_func].keys()):
                        wl_bnds.append(
                            (self.fit_wl_fit_func_bnd_handles[2 * ii].value,
                             self.fit_wl_fit_func_bnd_handles[2 * ii +
                                                              1].value))
                else:
                    wl_bnds = [-inf, inf]
                wl_params['eoff'] = self.hs['FitConfigWlFitPars1 text'].value
                wl_params['model'] = self.fit_wl_fit_func
                wl_params['fvars'] = wl_fvars
                wl_params['bnds'] = wl_bnds
                wl_params['jac'] = self.fit_wl_optimizer_arg_handles[0].value
                wl_params['method'] = self.fit_wl_optimizer_arg_handles[
                    1].value
                wl_params['ftol'] = self.fit_wl_optimizer_arg_handles[2].value
                wl_params['xtol'] = self.fit_wl_optimizer_arg_handles[3].value
                wl_params['gtol'] = self.fit_wl_optimizer_arg_handles[4].value
            elif self.fit_wl_optimizer == 'numpy':
                wl_params['eoff'] = (self.parent_h.xanes_fit_wl_fit_eng_s +
                                     self.parent_h.xanes_fit_wl_fit_eng_e) / 2.
                wl_params['model'] = 'polynd'
                wl_order = self.hs['FitConfigWlFunc drpdn'].value
                wl_params['order'] = wl_order
                wl_params['flt_spec'] = self.fit_use_flt_spec
            wl_ufac = self.fit_wl_optimizer_arg_handles[5].value
        else:
            wl_params = {}
            wl_ufac = 20

        if self.fit_edge50_pos:
            edge_fvars = []
            edge_bnds = []
            edge_params = {}
            edge_params['eoff'] = self.parent_h.xanes_fit_edge_eng
            edge_params['optimizer'] = self.fit_edge_optimizer
            edge_params['ftype'] = 'edge'
            if self.parent_h.xanes_fit_type == 'full':
                edge_params['on'] = 'norm'
            else:
                edge_params['on'] = 'raw'
            if self.fit_edge_optimizer == 'scipy':
                for ii in sorted(
                        dat_dict.XANES_EDGE_FIT_PARAM_DICT[self.fit_edge_fit_func].keys()):
                    edge_fvars.append(
                        self.fit_edge_fit_func_arg_handles[ii].value)

                if self.fit_edge_fit_func in ["sine", "expsine"]:
                    edge_fvars[2] = 0
                elif self.fit_edge_fit_func == "rectangle":
                    edge_fvars[1] = edge_fvars[3] = 0
                else:
                    edge_fvars[1] = 0

                if self.fit_wl_fit_use_param_bnd:
                    for ii in sorted(dat_dict.XANES_EDGE_FIT_PARAM_DICT[
                            self.fit_wdge_fit_func].keys()):
                        edge_bnds.append(
                            (self.fit_edge_fit_func_bnd_handles[2 * ii].value,
                             self.fit_edge_fit_func_bnd_handles[2 * ii +
                                                                1].value))
                else:
                    edge_bnds = [-inf, inf]

                edge_params['model'] = self.fit_edge_fit_func
                edge_params['fvars'] = edge_fvars
                edge_params['bnds'] = edge_bnds
                edge_params['jac'] = self.fit_edge_optimizer_arg_handles[
                    0].value
                edge_params['method'] = self.fit_edge_optimizer_arg_handles[
                    1].value
                edge_params['ftol'] = self.fit_edge_optimizer_arg_handles[
                    2].value
                edge_params['xtol'] = self.fit_edge_optimizer_arg_handles[
                    3].value
                edge_params['gtol'] = self.fit_edge_optimizer_arg_handles[
                    4].value
            elif self.fit_edge_optimizer == 'numpy':
                edge_params['model'] = 'polynd'
                edge_order = self.hs['FitConfigEdgeFunc drpdn'].value
                edge_params['order'] = edge_order
                edge_params['flt_spec'] = self.fit_use_flt_spec
            edge_ufac = self.fit_edge_optimizer_arg_handles[5].value
        else:
            edge_params = {}
            edge_ufac = 20

        edge50_optimizer = ''
        if self.fit_edge50_pos and self.fit_wl_pos:
            edge50_optimizer = 'both'
        elif self.fit_edge50_pos and (not self.fit_wl_pos):
            edge50_optimizer = 'edge'

        edge_fit_item = ['edge50_pos_fit']
        wl_fit_item = ['wl_pos_fit']

        if self.parent_h.gui_name == 'xanes3D':
            with h5py.File(self.parent_h.xanes_save_trial_reg_filename,
                           'r+') as f:
                if 'processed_XANES3D' not in f:
                    g1 = f.create_group('processed_XANES3D')
                else:
                    del f['processed_XANES3D']
                    g1 = f.create_group('processed_XANES3D')
                g11 = g1.create_group('proc_parameters')

                g11.create_dataset('element',
                                   data=str(self.parent_h.xanes_element))
                g11.create_dataset(
                    'eng_list',
                    data=scale_eng_list(
                        self.parent_h.xanes_fit_eng_list).astype(np.float32))
                g11.create_dataset('edge_eng',
                                   data=self.parent_h.xanes_fit_edge_eng)
                g11.create_dataset('pre_edge_e',
                                   data=self.parent_h.xanes_fit_pre_edge_e)
                g11.create_dataset('post_edge_s',
                                   data=self.parent_h.xanes_fit_post_edge_s)
                g11.create_dataset(
                    'edge_jump_threshold',
                    data=self.parent_h.xanes_fit_edge_jump_thres)
                g11.create_dataset(
                    'edge_offset_threshold',
                    data=self.parent_h.xanes_fit_edge_offset_thres)
                g11.create_dataset('use_mask',
                                   data=str(self.parent_h.xanes_fit_use_mask))
                g11.create_dataset('analysis_type',
                                   data=self.parent_h.xanes_fit_type)
                g11.create_dataset('data_shape',
                                   data=self.parent_h.xanes_fit_data_shape)
                g11.create_dataset('edge50_fit_s',
                                   data=self.parent_h.xanes_fit_edge_0p5_fit_s)
                g11.create_dataset('edge50_fit_e',
                                   data=self.parent_h.xanes_fit_edge_0p5_fit_e)
                g11.create_dataset('wl_fit_eng_s',
                                   data=self.parent_h.xanes_fit_wl_fit_eng_s)
                g11.create_dataset('wl_fit_eng_e',
                                   data=self.parent_h.xanes_fit_wl_fit_eng_e)
                g11.create_dataset('pre_post_edge_norm_fit_order', data=1)
                g11.create_dataset('flt_spec', data=str(self.fit_use_flt_spec))
                g11.create_dataset('bin_fact', data=int(self.fit_img_bin_fac))

                g111 = g11.create_group('wl_fit method')
                if self.fit_wl_pos:
                    g111.create_dataset('optimizer',
                                        data=str(wl_params['optimizer']))
                    if wl_params['optimizer'] == 'scipy':
                        g111.create_dataset('method',
                                            data=str(wl_params['model']))
                        g1111 = g111.create_group('params')
                        g1111.create_dataset('eng_offset',
                                             data=wl_params['eoff'])
                        g1111.create_dataset('spec', data=str(wl_params['on']))
                        g1111.create_dataset('jac', data=str(wl_params['jac']))
                        g1111.create_dataset('method',
                                             data=str(wl_params['method']))
                        g1111.create_dataset('ftol', data=wl_params['ftol'])
                        g1111.create_dataset('xtol', data=wl_params['xtol'])
                        g1111.create_dataset('gtol', data=wl_params['gtol'])
                        g1111.create_dataset('fvars_init', data=wl_fvars)
                        if wl_bnds[0] is -inf:
                            g1111.create_dataset('bnds', data=str('None'))
                        else:
                            g1111.create_dataset('bnds', data=wl_bnds)
                    else:
                        g111.create_dataset('method', data=str('polynd'))
                        g1111 = g111.create_group('params')
                        g1111.create_dataset('order', data=wl_params['order'])
                        g1111.create_dataset('median_flt',
                                             data=str(wl_params['flt_spec']))
                        g1111.create_dataset('eng_offset',
                                             data=wl_params['eoff'])
                        g1111.create_dataset('spec', data=str(wl_params['on']))

                g112 = g11.create_group('edge_fit method')
                if self.fit_edge50_pos:
                    g112.create_dataset('optimizer',
                                        data=str(edge_params['optimizer']))
                    if edge_params['optimizer'] == 'scipy':
                        g112.create_dataset('method',
                                            data=str(edge_params['model']))
                        g1121 = g112.create_group('params')
                        g1121.create_dataset('eng_offset',
                                             data=edge_params['eoff'])
                        g1121.create_dataset('spec',
                                             data=str(edge_params['on']))
                        g1121.create_dataset('jac',
                                             data=str(edge_params['jac']))
                        g1121.create_dataset('method',
                                             data=str(edge_params['method']))
                        g1121.create_dataset('ftol', data=edge_params['ftol'])
                        g1121.create_dataset('xtol', data=edge_params['xtol'])
                        g1121.create_dataset('gtol', data=edge_params['gtol'])
                        g1121.create_dataset('fvars_init', data=edge_fvars)
                        if edge_bnds[0] is -inf:
                            g1121.create_dataset('bnds', data=str('None'))
                        else:
                            g1121.create_dataset('bnds', data=edge_bnds)
                    else:
                        g112.create_dataset('method', data=str('polynd'))
                        g1121 = g112.create_group('params')
                        g1121.create_dataset('order', data=edge_order)
                        g1121.create_dataset('median_flt',
                                             data=str(edge_params['flt_spec']))
                        g1121.create_dataset('eng_offset',
                                             data=edge_params['eoff'])
                        g1121.create_dataset('spec',
                                             data=str(edge_params['on']))
                g113 = g11.create_group('LCF')
                g113.create_dataset('use_lcf', data=str(self.fit_lcf))
                g113.create_dataset('use_constr',
                                    data=str(self.fit_lcf_constr))
            self.fit_lcf_ref_spec.to_hdf(
                self.parent_h.xanes_save_trial_reg_filename,
                '/processed_XANES3D/proc_parameters/LCF/ref',
                mode='a')

            code = {}
            ln = 0
            code[
                ln] = f"import os, h5py"
            ln += 1
            code[
                ln] = f"import numpy as np"
            ln += 1
            code[
                ln] = f"import pandas as pd"
            ln += 1
            code[
                ln] = f"from scipy.ndimage import zoom"
            ln += 1
            code[
                ln] = f"import TXM_Sandbox.utils.xanes_math as xm"
            ln += 1
            code[
                ln] = f"import TXM_Sandbox.utils.xanes_analysis as xa"
            ln += 1
            code[
                ln] = f"from TXM_Sandbox.utils.misc import str2bool"
            ln += 1
            code[
                ln] = f"from copy import deepcopy"
            ln += 1
            code[
                ln] = f"inf = np.inf"
            ln += 1
            code[
                ln] = f""
            ln += 1
            code[
                ln] = f"with h5py.File('{self.parent_h.xanes_save_trial_reg_filename}', 'r+') as f:"
            ln += 1
            code[
                ln] = f"    if {self.fit_img_bin_fac} == 1:"
            ln += 1
            code[
                ln] = f"        imgs = f['/registration_results/reg_results/registered_xanes3D'][:, 0, :, :]"
            ln += 1
            code[
                ln] = f"    else:"
            ln += 1
            code[
                ln] = f"        imgs = zoom(f['/registration_results/reg_results/registered_xanes3D'][:, 0:{self.fit_img_bin_fac}, :, :], (1, 1/{self.fit_img_bin_fac}, 1/{self.fit_img_bin_fac}, 1/{self.fit_img_bin_fac}))"
            ln += 1
            code[
                ln] = f"    xanes3D_analysis_eng_list = f['/processed_XANES3D/proc_parameters/eng_list'][:]"
            ln += 1
            code[
                ln] = f"    xanes3D_analysis_edge_eng = f['/processed_XANES3D/proc_parameters/edge_eng'][()]"
            ln += 1
            code[
                ln] = f"    xanes3D_analysis_pre_edge_e = f['/processed_XANES3D/proc_parameters/pre_edge_e'][()]"
            ln += 1
            code[
                ln] = f"    xanes3D_analysis_post_edge_s = f['/processed_XANES3D/proc_parameters/post_edge_s'][()]"
            ln += 1
            code[
                ln] = f"    xanes3D_analysis_edge_jump_thres = f['/processed_XANES3D/proc_parameters/edge_jump_threshold'][()]"
            ln += 1
            code[
                ln] = f"    xanes3D_analysis_edge_offset_thres = f['/processed_XANES3D/proc_parameters/edge_offset_threshold'][()]"
            ln += 1
            code[
                ln] = f"    xanes3D_analysis_use_mask = f['/processed_XANES3D/proc_parameters/use_mask'][()].decode('utf-8')"
            ln += 1
            code[
                ln] = f"    xanes3D_analysis_type = f['/processed_XANES3D/proc_parameters/analysis_type'][()].decode('utf-8')"
            ln += 1
            code[
                ln] = f"    xanes3D_analysis_data_shape = np.int32(np.round(f['/processed_XANES3D/proc_parameters/data_shape'][:]/{self.fit_img_bin_fac}))"
            ln += 1
            code[
                ln] = f"    xanes3D_analysis_edge50_fit_s = f['/processed_XANES3D/proc_parameters/edge50_fit_s'][()]"
            ln += 1
            code[
                ln] = f"    xanes3D_analysis_edge50_fit_e = f['/processed_XANES3D/proc_parameters/edge50_fit_e'][()]"
            ln += 1
            code[
                ln] = f"    xanes3D_analysis_wl_fit_eng_s = f['/processed_XANES3D/proc_parameters/wl_fit_eng_s'][()]"
            ln += 1
            code[
                ln] = f"    xanes3D_analysis_wl_fit_eng_e = f['/processed_XANES3D/proc_parameters/wl_fit_eng_e'][()]"
            ln += 1
            code[
                ln] = f"    xanes3D_analysis_edge_fit_order = f['/processed_XANES3D/proc_parameters/pre_post_edge_norm_fit_order'][()]"
            ln += 1
            code[
                ln] = f"    xanes3D_analysis_use_flt_spec = f['/processed_XANES3D/proc_parameters/flt_spec'][()].decode('utf-8')"
            ln += 1
            code[
                ln] = f"    xanes3D_analysis_lcf_use = str2bool(f['/processed_XANES3D/proc_parameters/LCF/use_lcf'][()])"
            ln += 1
            code[
                ln] = f"    xanes3D_analysis_lcf_constr = str2bool(f['/processed_XANES3D/proc_parameters/LCF/use_constr'][()])"
            ln += 1
            code[
                ln] = f"    xanes3D_analysis_lcf_ref = pd.read_hdf('{self.parent_h.xanes_save_trial_reg_filename}', '/processed_XANES3D/proc_parameters/LCF/ref')"
            ln += 1
            code[
                ln] = f"    xana = xa.xanes_analysis(imgs, xanes3D_analysis_eng_list, xanes3D_analysis_edge_eng, pre_ee=xanes3D_analysis_pre_edge_e, post_es=xanes3D_analysis_post_edge_s, edge_jump_threshold=xanes3D_analysis_edge_jump_thres, pre_edge_threshold=xanes3D_analysis_edge_offset_thres)"
            ln += 1
            code[
                ln] = f"    xana.lcf_use = xanes3D_analysis_lcf_use"
            ln += 1
            code[
                ln] = f"    xana.lcf_constr_use = xanes3D_analysis_lcf_constr"
            ln += 1
            code[
                ln] = f"    xana.lcf_ref = xanes3D_analysis_lcf_ref"
            ln += 1
            code[
                ln] = f"    if '/processed_XANES3D/proc_spectrum' in f:"
            ln += 1
            code[
                ln] = f"        del f['/processed_XANES3D/proc_spectrum']"
            ln += 1
            code[
                ln] = f"        g12 = f.create_group('/processed_XANES3D/proc_spectrum')"
            ln += 1
            code[
                ln] = f"    else:"
            ln += 1
            code[
                ln] = f"        g12 = f.create_group('/processed_XANES3D/proc_spectrum')"
            ln += 1
            code[
                ln] = f""
            ln += 1
            code[
                ln] = f"    if xanes3D_analysis_type == 'wl':"
            ln += 1
            code[
                ln] = "        _g12 = {}"
            ln += 1
            code[
                ln] = f"        for jj in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"            if jj == 'wl_fit_coef':"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='wl_fit_coef')}, *xanes3D_analysis_data_shape[1:]), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"            elif jj == 'edge_fit_coef':"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='edge_fit_coef')}, *xanes3D_analysis_data_shape[1:]), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"            elif jj in ['pre_edge_fit_coef', 'post_edge_fit_coef']:"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='pre_edge_fit_coef')}, *xanes3D_analysis_data_shape[1:]), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"            else:"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"        if {self.fit_img_bin_fac} == 1:"
            ln += 1
            code[
                ln] = f"            for ii in range(xanes3D_analysis_data_shape[1]):"
            ln += 1
            code[
                ln] = f"                xana.spec[:] = f['/registration_results/reg_results/registered_xanes3D'][:, ii, :, :]"
            ln += 1
            code[
                ln] = f"                if np.any(np.array([ii in {self.analysis_saving_items} for ii in {wl_fit_item}])):"
            ln += 1
            code[
                ln] = f"                    xana.fit_edge(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, **{wl_params})"
            ln += 1
            code[
                ln] = f"                if 'wl_pos_fit' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"                    xana.find_wl(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='model',ufac={wl_ufac})"
            ln += 1
            code[
                ln] = f"                if 'wl_pos_dir' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"                    xana.find_wl(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='direct',ufac={wl_ufac})"
            ln += 1
            code[
                ln] = f"                if any([ii in {self.analysis_saving_items} for ii in ['centroid_of_eng', 'centroid_of_eng_relative_to_wl', 'weighted_attenuation', 'weighted_eng']]):"
            ln += 1
            code[
                ln] = f"                    xana.cal_wgt_eng(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e)"
            ln += 1
            code[
                ln] = f"                for jj in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"                    if jj == 'wl_pos_fit':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.wl_pos_fit)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'wl_pos_dir':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.wl_pos_dir)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'wl_fit_coef':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][0].reshape(xana.model['wl']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'wl_fit_err':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'centroid_of_eng':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.centroid_of_eng)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'centroid_of_eng_relative_to_wl':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.centroid_of_eng_rel_wl)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'weighted_attenuation':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.weighted_atten)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'weighted_eng':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.weighted_eng)[:]"
            ln += 1
            code[
                ln] = f"                print(ii)"
            ln += 1
            code[
                ln] = f"        else:"
            ln += 1
            code[
                ln] = f"            for ii in range(0, xanes3D_analysis_data_shape[1], {self.fit_img_bin_fac}):"
            ln += 1
            code[
                ln] = f"                xana.spec[:] = zoom(f['/registration_results/reg_results/registered_xanes3D'][:, ii:ii+{self.fit_img_bin_fac}, :, :], (1, 1/{self.fit_img_bin_fac}, 1/{self.fit_img_bin_fac}, 1/{self.fit_img_bin_fac}))"
            ln += 1
            code[
                ln] = f"                if np.any(np.array([ii in {self.analysis_saving_items} for ii in {wl_fit_item}])):"
            ln += 1
            code[
                ln] = f"                    xana.fit_edge(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, **{wl_params})"
            ln += 1
            code[
                ln] = f"                if 'wl_pos_fit' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"                    xana.find_wl(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='model',ufac={wl_ufac})"
            ln += 1
            code[
                ln] = f"                if 'wl_pos_dir' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"                    xana.find_wl(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='direct',ufac={wl_ufac})"
            ln += 1
            code[
                ln] = f"                if any([ii in {self.analysis_saving_items} for ii in ['centroid_of_eng', 'centroid_of_eng_relative_to_wl', 'weighted_attenuation', 'weighted_eng']]):"
            ln += 1
            code[
                ln] = f"                    xana.cal_wgt_eng(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e)"
            ln += 1
            code[
                ln] = f"                for jj in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"                    if jj == 'wl_pos_fit':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.wl_pos_fit)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'wl_pos_dir':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.wl_pos_dir)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'wl_fit_coef':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][0].reshape(xana.model['wl']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'wl_fit_err':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'centroid_of_eng':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.centroid_of_eng)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'centroid_of_eng_relative_to_wl':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.centroid_of_eng_rel_wl)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'weighted_attenuation':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.weighted_atten)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'weighted_eng':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.weighted_eng)[:]"
            ln += 1
            code[
                ln] = f"                print(ii)"
            ln += 1
            code[
                ln] = f"    elif xanes3D_analysis_type == 'full':"
            ln += 1
            code[
                ln] = "        _g12 = {}"
            ln += 1
            code[
                ln] = f"        for jj in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"            if jj == 'wl_fit_coef':"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='wl_fit_coef')}, *xanes3D_analysis_data_shape[1:]), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"            elif jj == 'edge_fit_coef':"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='edge_fit_coef')}, *xanes3D_analysis_data_shape[1:]), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"            elif jj in ['pre_edge_fit_coef', 'post_edge_fit_coef']:"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='pre_edge_fit_coef')}, *xanes3D_analysis_data_shape[1:]), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"            elif jj == 'norm_spec':"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=(xanes3D_analysis_data_shape), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"            elif jj == 'lcf_fit':"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=([xana.lcf_ref.index.size, *xanes3D_analysis_data_shape[1:]]), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"            else:"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"        if {self.fit_img_bin_fac} == 1:"
            ln += 1
            code[
                ln] = f"            for ii in range(xanes3D_analysis_data_shape[1]):"
            ln += 1
            code[
                ln] = f"                xana.spec[:] = f['/registration_results/reg_results/registered_xanes3D'][:, ii, :, :]"
            ln += 1
            code[
                ln] = f"                xana.cal_pre_edge_sd()"
            ln += 1
            code[
                ln] = f"                xana.cal_post_edge_sd()"
            ln += 1
            code[
                ln] = f"                xana.cal_pre_edge_mean()"
            ln += 1
            code[
                ln] = f"                xana.cal_post_edge_mean()"
            ln += 1
            code[
                ln] = f"                xana.full_spec_preprocess(xanes3D_analysis_edge_eng, order=xanes3D_analysis_edge_fit_order, save_pre_post=True)"
            ln += 1
            code[
                ln] = f"                if np.any(np.array([ii in {self.analysis_saving_items} for ii in {edge_fit_item}])):"
            ln += 1
            code[
                ln] = f"                    xana.fit_edge(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, **{edge_params})"
            ln += 1
            code[
                ln] = f"                if np.any(np.array([ii in {self.analysis_saving_items} for ii in {wl_fit_item}])):"
            ln += 1
            code[
                ln] = f"                    xana.fit_edge(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, **{wl_params})"
            ln += 1
            code[
                ln] = f"                if ('edge50_pos_fit' in {self.analysis_saving_items}):"
            ln += 1
            code[
                ln] = f"                    xana.find_edge_50(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='{edge50_optimizer}', ufac={edge_ufac})"
            ln += 1
            code[
                ln] = f"                if 'edge50_pos_dir' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"                    xana.find_edge_50(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='none', ufac={edge_ufac})"
            ln += 1
            code[
                ln] = f"                if ('edge_pos_fit' in {self.analysis_saving_items}):"
            ln += 1
            code[
                ln] = f"                    xana.find_edge_deriv(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, optimizer='model',ufac={edge_ufac})"
            ln += 1
            code[
                ln] = f"                if 'edge_pos_dir' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"                    xana.find_edge_deriv(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, optimizer='direct',ufac={edge_ufac})"
            ln += 1
            code[
                ln] = f"                if 'wl_pos_fit' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"                    xana.find_wl(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='model',ufac={wl_ufac})"
            ln += 1
            code[
                ln] = f"                if 'wl_pos_dir' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"                    xana.find_wl(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='direct',ufac={wl_ufac})"
            ln += 1
            code[
                ln] = f"                if any([ii in {self.analysis_saving_items} for ii in ['centroid_of_eng', 'centroid_of_eng_relative_to_wl', 'weighted_attenuation', 'weighted_eng']]):"
            ln += 1
            code[
                ln] = f"                    xana.cal_wgt_eng(xanes3D_analysis_pre_edge_e + xanes3D_analysis_edge_eng, xanes3D_analysis_wl_fit_eng_e)"
            ln += 1
            code[
                ln] = f"                if 'lcf_fit' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"                    xana.interp_ref_spec()"
            ln += 1
            code[
                ln] = f"                    xana.lcf()"
            ln += 1
            code[
                ln] = f""
            ln += 1
            code[
                ln] = f"                for jj in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"                    if jj == 'wl_pos_fit':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.wl_pos_fit)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'wl_pos_dir':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.wl_pos_dir)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'wl_peak_height_dir':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.direct_wl_ph)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'centroid_of_eng':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.centroid_of_eng)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'centroid_of_eng_relative_to_wl':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.centroid_of_eng_rel_wl)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'weighted_attenuation':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.weighted_atten)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'weighted_eng':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.weighted_eng)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'edge50_pos_fit':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.edge50_pos_fit)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'edge50_pos_dir':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.edge50_pos_fit_none)[:]"
            ln += 1
            code[
                ln] = f"                    if (jj == 'edge_pos_fit'):"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.edge_pos_fit)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'edge_pos_dir':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.edge_pos_dir)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'edge_jump_filter':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.edge_jump_mask)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'edge_offset_filter':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.fitted_edge_mask)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'pre_edge_sd':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.pre_edge_sd_map)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'post_edge_sd':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.post_edge_sd_map)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'pre_edge_mean':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.pre_edge_mean_map)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'post_edge_mean':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.post_edge_mean_map)[:]"
            ln += 1
            code[
                ln] = f"                    if jj  == 'pre_edge_fit_coef':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][:] = np.float32(xana.pre_edge_fit_rlt[0].reshape(xana.pre_edge_fit_rlt[0].shape[0], *xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'post_edge_fit_coef':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][:] = np.float32(xana.post_edge_fit_rlt[0].reshape(xana.post_edge_fit_rlt[0].shape[0], *xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'edge_fit_coef':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][:] = np.float32(xana.model['edge']['fit_rlt'][0].reshape(xana.model['edge']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'edge_fit_err':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][:] = np.float32(xana.model['edge']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'wl_fit_coef':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][0].reshape(xana.model['wl']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'wl_fit_err':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'norm_spec':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][:, ii, :, :] = np.float32(xana.norm_spec)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'lcf_fit':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.lcf_fit)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'lcf_fit_err':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.lcf_fit_err)[:]"
            ln += 1
            code[
                ln] = f"                print(ii)"
            ln += 1
            code[
                ln] = f"        else:"
            ln += 1
            code[
                ln] = f"            for ii in range(0, xanes3D_analysis_data_shape[1], {self.fit_img_bin_fac}):"
            ln += 1
            code[
                ln] = f"                xana.spec[:] = zoom(f['/registration_results/reg_results/registered_xanes3D'][:, ii:ii+{self.fit_img_bin_fac}, :, :], (1, 1/{self.fit_img_bin_fac}, 1/{self.fit_img_bin_fac}, 1/{self.fit_img_bin_fac}))"
            ln += 1
            code[
                ln] = f"                xana.cal_pre_edge_sd()"
            ln += 1
            code[
                ln] = f"                xana.cal_post_edge_sd()"
            ln += 1
            code[
                ln] = f"                xana.cal_pre_edge_mean()"
            ln += 1
            code[
                ln] = f"                xana.cal_post_edge_mean()"
            ln += 1
            code[
                ln] = f"                xana.full_spec_preprocess(xanes3D_analysis_edge_eng, order=xanes3D_analysis_edge_fit_order, save_pre_post=True)"
            ln += 1
            code[
                ln] = f"                if np.any(np.array([ii in {self.analysis_saving_items} for ii in {edge_fit_item}])):"
            ln += 1
            code[
                ln] = f"                    xana.fit_edge(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, **{edge_params})"
            ln += 1
            code[
                ln] = f"                if np.any(np.array([ii in {self.analysis_saving_items} for ii in {wl_fit_item}])):"
            ln += 1
            code[
                ln] = f"                    xana.fit_edge(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, **{wl_params})"
            ln += 1
            code[
                ln] = f"                if ('edge50_pos_fit' in {self.analysis_saving_items}):"
            ln += 1
            code[
                ln] = f"                    xana.find_edge_50(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='{edge50_optimizer}', ufac={edge_ufac})"
            ln += 1
            code[
                ln] = f"                if ('edge50_pos_dir' in {self.analysis_saving_items}):"
            ln += 1
            code[
                ln] = f"                    xana.find_edge_50(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='none', ufac={edge_ufac})"
            ln += 1
            code[
                ln] = f"                if ('edge_pos_fit' in {self.analysis_saving_items}):"
            ln += 1
            code[
                ln] = f"                    xana.find_edge_deriv(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, optimizer='model',ufac={edge_ufac})"
            ln += 1
            code[
                ln] = f"                if 'edge_pos_dir' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"                    xana.find_edge_deriv(xanes3D_analysis_edge50_fit_s, xanes3D_analysis_edge50_fit_e, optimizer='direct',ufac={edge_ufac})"
            ln += 1
            code[
                ln] = f"                if 'wl_pos_fit' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"                    xana.find_wl(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='model',ufac={wl_ufac})"
            ln += 1
            code[
                ln] = f"                if 'wl_pos_dir' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"                    xana.find_wl(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, optimizer='direct',ufac={wl_ufac})"
            ln += 1
            code[
                ln] = f"                if any([ii in {self.analysis_saving_items} for ii in ['centroid_of_eng', 'centroid_of_eng_relative_to_wl', 'weighted_attenuation', 'weighted_eng']]):"
            ln += 1
            code[
                ln] = f"                    xana.cal_wgt_eng(xanes3D_analysis_pre_edge_e + xanes3D_analysis_edge_eng, xanes3D_analysis_wl_fit_eng_e)"
            ln += 1
            code[
                ln] = f"                if 'lcf_fit' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"                    xana.interp_ref_spec()"
            ln += 1
            code[
                ln] = f"                    xana.lcf()"
            ln += 1
            code[
                ln] = f""
            ln += 1
            code[
                ln] = f"                for jj in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"                    if jj == 'wl_pos_fit':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.wl_pos_fit)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'wl_pos_dir':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.wl_pos_dir)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'wl_peak_height_dir':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.direct_wl_ph)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'centroid_of_eng':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.centroid_of_eng)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'centroid_of_eng_relative_to_wl':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.centroid_of_eng_rel_wl)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'weighted_attenuation':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.weighted_atten)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'weighted_eng':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.weighted_eng)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'edge50_pos_fit':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.edge50_pos_fit)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'edge50_pos_dir':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.edge50_pos_fit_none)[:]"
            ln += 1
            code[
                ln] = f"                    if (jj == 'edge_pos_fit'):"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.edge_pos_fit)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'edge_pos_dir':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.edge_pos_dir)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'edge_jump_filter':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.edge_jump_mask)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'edge_offset_filter':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.fitted_edge_mask)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'pre_edge_sd':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.pre_edge_sd_map)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'post_edge_sd':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.post_edge_sd_map)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'pre_edge_mean':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.pre_edge_mean_map)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'post_edge_mean':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.post_edge_mean_map)[:]"
            ln += 1
            code[
                ln] = f"                    if jj  == 'pre_edge_fit_coef':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][:] = np.float32(xana.pre_edge_fit_rlt[0].reshape(xana.pre_edge_fit_rlt[0].shape[0], *xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'post_edge_fit_coef':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][:] = np.float32(xana.post_edge_fit_rlt[0].reshape(xana.post_edge_fit_rlt[0].shape[0], *xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'edge_fit_coef':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][:] = np.float32(xana.model['edge']['fit_rlt'][0].reshape(xana.model['edge']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'edge_fit_err':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][:] = np.float32(xana.model['edge']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'wl_fit_coef':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][0].reshape(xana.model['wl']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'wl_fit_err':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'norm_spec':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][:, ii, :, :] = np.float32(xana.norm_spec)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'lcf_fit':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.lcf_fit)[:]"
            ln += 1
            code[
                ln] = f"                    if jj == 'lcf_fit_err':"
            ln += 1
            code[
                ln] = f"                        _g12[jj][ii] = np.float32(xana.lcf_fit_err)[:]"
            ln += 1
            code[
                ln] = f"                print(ii)"
            ln += 1
            code[
                ln] = f"print('xanes3D analysis is done!')"
            ln += 1

            gen_external_py_script(
                self.parent_h.xanes_fit_external_command_name, code)
            sig = os.system(
                f'python {self.parent_h.xanes_fit_external_command_name}')
            if sig == 0:
                self.hs['FitRun text'].value = 'XANES3D analysis is done ...'
            else:
                self.hs[
                    'FitRun text'].value = 'something wrong in analysis ...'
            self.parent_h.update_xanes3D_config()
        elif self.parent_h.gui_name == 'xanes2D':
            with h5py.File(self.parent_h.xanes_save_trial_reg_filename,
                           'r+') as f:
                if 'processed_XANES2D' not in f:
                    g1 = f.create_group('processed_XANES2D')
                else:
                    del f['processed_XANES2D']
                    g1 = f.create_group('processed_XANES2D')
                g11 = g1.create_group('proc_parameters')
                g11.create_dataset('eng_list',
                                   data=self.parent_h.xanes_fit_eng_list)
                g11.create_dataset('edge_eng',
                                   data=self.parent_h.xanes_fit_edge_eng)
                g11.create_dataset('pre_edge_e',
                                   data=self.parent_h.xanes_fit_pre_edge_e)
                g11.create_dataset('post_edge_s',
                                   data=self.parent_h.xanes_fit_post_edge_s)
                g11.create_dataset(
                    'edge_jump_threshold',
                    data=self.parent_h.xanes_fit_edge_jump_thres)
                g11.create_dataset(
                    'edge_offset_threshold',
                    data=self.parent_h.xanes_fit_edge_offset_thres)
                g11.create_dataset('use_mask',
                                   data=str(self.parent_h.xanes_fit_use_mask))
                g11.create_dataset('analysis_type',
                                   data=self.parent_h.xanes_fit_type)
                g11.create_dataset('data_shape',
                                   data=self.parent_h.xanes_fit_data_shape)
                g11.create_dataset('edge50_fit_s',
                                   data=self.parent_h.xanes_fit_edge_0p5_fit_s)
                g11.create_dataset('edge50_fit_e',
                                   data=self.parent_h.xanes_fit_edge_0p5_fit_e)
                g11.create_dataset('wl_fit_eng_s',
                                   data=self.parent_h.xanes_fit_wl_fit_eng_s)
                g11.create_dataset('wl_fit_eng_e',
                                   data=self.parent_h.xanes_fit_wl_fit_eng_e)
                g11.create_dataset('pre_post_edge_norm_fit_order', data=1)
                g11.create_dataset('flt_spec', data=str(self.fit_use_flt_spec))
                g11.create_dataset('bin_fact', data=int(self.fit_img_bin_fac))

                g111 = g11.create_group('wl_fit method')
                if self.fit_wl_pos:
                    g111.create_dataset('optimizer',
                                        data=str(self.fit_wl_optimizer))
                    if self.fit_wl_optimizer == 'scipy':
                        g111.create_dataset('method',
                                            data=str(self.fit_wl_fit_func))
                        g1111 = g111.create_group('params')
                        g1111.create_dataset('jac', data=str(wl_params['jac']))
                        g1111.create_dataset('method',
                                             data=str(wl_params['method']))
                        g1111.create_dataset('ftol', data=wl_params['ftol'])
                        g1111.create_dataset('xtol', data=wl_params['xtol'])
                        g1111.create_dataset('gtol', data=wl_params['gtol'])
                        g1111.create_dataset('fvars_init', data=wl_fvars)
                        if wl_bnds[0] is -inf:
                            g1111.create_dataset('bnds', data=str('None'))
                        else:
                            g1111.create_dataset('bnds', data=wl_bnds)
                    else:
                        g111.create_dataset('method', data=str('polynimial'))
                        g1111 = g111.create_group('params')
                        g1111.create_dataset('order', data=wl_order)

                g112 = g11.create_group('edge_fit method')
                if self.fit_edge50_pos:
                    g112.create_dataset('optimizer',
                                        data=str(self.fit_edge_optimizer))
                    if self.fit_edge_optimizer == 'scipy':
                        g112.create_dataset('method',
                                            data=str(self.fit_edge_fit_func))
                        g1121 = g112.create_group('params')
                        g1121.create_dataset('jac',
                                             data=str(edge_params['jac']))
                        g1121.create_dataset('method',
                                             data=str(edge_params['method']))
                        g1121.create_dataset('ftol', data=edge_params['ftol'])
                        g1121.create_dataset('xtol', data=edge_params['xtol'])
                        g1121.create_dataset('gtol', data=edge_params['gtol'])
                        g1121.create_dataset('fvars_init', data=edge_fvars)
                        if edge_bnds[0] is -inf:
                            g1121.create_dataset('bnds', data=str('None'))
                        else:
                            g1121.create_dataset('bnds', data=edge_bnds)
                    else:
                        g112.create_dataset('method', data=str('polynimial'))
                        g1121 = g112.create_group('params')
                        g1121.create_dataset('order', data=edge_order)
                g113 = g11.create_group('LCF')
                g113.create_dataset('use_lcf', data=str(self.fit_lcf))
                g113.create_dataset('use_constr',
                                    data=str(self.fit_lcf_constr))
            self.fit_lcf_ref_spec.to_hdf(
                self.parent_h.xanes_save_trial_reg_filename,
                '/processed_XANES2D/proc_parameters/LCF/ref',
                mode='a')
            code = {}
            ln = 0
            code[
                ln] = f"import os, h5py"
            ln += 1
            code[
                ln] = f"import numpy as np"
            ln += 1
            code[
                ln] = f"import pandas as pd"
            ln += 1
            code[
                ln] = f"from scipy.ndimage import zoom"
            ln += 1
            code[
                ln] = f"import TXM_Sandbox.utils.xanes_math as xm"
            ln += 1
            code[
                ln] = f"import TXM_Sandbox.utils.xanes_analysis as xa"
            ln += 1
            code[
                ln] = f"from TXM_Sandbox.utils.misc import str2bool"
            ln += 1
            code[
                ln] = f"from copy import deepcopy"
            ln += 1
            code[
                ln] = f"inf = np.inf"
            ln += 1
            code[
                ln] = f""
            ln += 1
            code[
                ln] = f"with h5py.File('{self.parent_h.xanes_save_trial_reg_filename}', 'r+') as f:"
            ln += 1
            code[
                ln] = f"    if {self.fit_img_bin_fac} == 1:"
            ln += 1
            code[
                ln] = f"        imgs = f['/registration_results/reg_results/registered_xanes2D'][:]"
            ln += 1
            code[
                ln] = f"    else:"
            ln += 1
            code[
                ln] = f"        imgs = zoom(f['/registration_results/reg_results/registered_xanes2D'][:], (1, {1/self.fit_img_bin_fac}, {1/self.fit_img_bin_fac}))"
            ln += 1
            code[
                ln] = f"    xanes2D_analysis_eng_list = f['/processed_XANES2D/proc_parameters/eng_list'][:]"
            ln += 1
            code[
                ln] = f"    xanes2D_analysis_edge_eng = f['/processed_XANES2D/proc_parameters/edge_eng'][()]"
            ln += 1
            code[
                ln] = f"    xanes2D_analysis_pre_edge_e = f['/processed_XANES2D/proc_parameters/pre_edge_e'][()]"
            ln += 1
            code[
                ln] = f"    xanes2D_analysis_post_edge_s = f['/processed_XANES2D/proc_parameters/post_edge_s'][()]"
            ln += 1
            code[
                ln] = f"    xanes2D_analysis_edge_jump_thres = f['/processed_XANES2D/proc_parameters/edge_jump_threshold'][()]"
            ln += 1
            code[
                ln] = f"    xanes2D_analysis_edge_offset_thres = f['/processed_XANES2D/proc_parameters/edge_offset_threshold'][()]"
            ln += 1
            code[
                ln] = f"    xanes2D_analysis_use_mask = f['/processed_XANES2D/proc_parameters/use_mask'][()].decode('utf-8')"
            ln += 1
            code[
                ln] = f"    xanes2D_analysis_type = f['/processed_XANES2D/proc_parameters/analysis_type'][()].decode('utf-8')"
            ln += 1
            code[
                ln] = f"    xanes2D_analysis_data_shape = imgs.shape"
            ln += 1
            code[
                ln] = f"    xanes2D_analysis_edge50_fit_s = f['/processed_XANES2D/proc_parameters/edge50_fit_s'][()]"
            ln += 1
            code[
                ln] = f"    xanes2D_analysis_edge50_fit_e = f['/processed_XANES2D/proc_parameters/edge50_fit_e'][()]"
            ln += 1
            code[
                ln] = f"    xanes2D_analysis_wl_fit_eng_s = f['/processed_XANES2D/proc_parameters/wl_fit_eng_s'][()]"
            ln += 1
            code[
                ln] = f"    xanes2D_analysis_wl_fit_eng_e = f['/processed_XANES2D/proc_parameters/wl_fit_eng_e'][()]"
            ln += 1
            code[
                ln] = f"    xanes2D_analysis_use_flt_spec = f['/processed_XANES2D/proc_parameters/flt_spec'][()].decode('utf-8')"
            ln += 1
            code[
                ln] = f"    xanes2D_analysis_edge_fit_order = f['/processed_XANES2D/proc_parameters/pre_post_edge_norm_fit_order'][()]"
            ln += 1
            code[
                ln] = f"    xanes2D_analysis_lcf_use = str2bool(f['/processed_XANES2D/proc_parameters/LCF/use_lcf'][()])"
            ln += 1
            code[
                ln] = f"    xanes2D_analysis_lcf_constr = str2bool(f['/processed_XANES2D/proc_parameters/LCF/use_constr'][()])"
            ln += 1
            code[
                ln] = f"    xanes2D_analysis_lcf_ref = pd.read_hdf('{self.parent_h.xanes_save_trial_reg_filename}', '/processed_XANES2D/proc_parameters/LCF/ref')"
            ln += 1
            code[
                ln] = f"    xana = xa.xanes_analysis(imgs, xanes2D_analysis_eng_list, xanes2D_analysis_edge_eng, pre_ee=xanes2D_analysis_pre_edge_e, post_es=xanes2D_analysis_post_edge_s, edge_jump_threshold=xanes2D_analysis_edge_jump_thres, pre_edge_threshold=xanes2D_analysis_edge_offset_thres)"
            ln += 1
            code[
                ln] = f"    xana.lcf_use = xanes2D_analysis_lcf_use"
            ln += 1
            code[
                ln] = f"    xana.lcf_constr_use = xanes2D_analysis_lcf_constr"
            ln += 1
            code[
                ln] = f"    xana.lcf_ref = xanes2D_analysis_lcf_ref"
            ln += 1
            code[
                ln] = f"    if '/processed_XANES2D/proc_spectrum' in f:"
            ln += 1
            code[
                ln] = f"        del f['/processed_XANES2D/proc_spectrum']"
            ln += 1
            code[
                ln] = f"        g12 = f.create_group('/processed_XANES2D/proc_spectrum')"
            ln += 1
            code[
                ln] = f"    else:"
            ln += 1
            code[
                ln] = f"        g12 = f.create_group('/processed_XANES2D/proc_spectrum')"
            ln += 1
            code[
                ln] = f""
            ln += 1
            code[
                ln] = f"    if xanes2D_analysis_type == 'wl':"
            ln += 1
            code[
                ln] = "        _g12 = {}"
            ln += 1
            code[
                ln] = f"        for jj in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"            if jj == 'wl_fit_coef':"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='wl_fit_coef')}, *xanes2D_analysis_data_shape[1:]), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"            elif jj == 'edge_fit_coef':"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='edge_fit_coef')}, *xanes2D_analysis_data_shape[1:]), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"            else:"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"        if np.any(np.array([ii in {self.analysis_saving_items} for ii in {wl_fit_item}])):"
            ln += 1
            code[
                ln] = f"            xana.fit_edge(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, **{wl_params})"
            ln += 1
            code[
                ln] = f"        if 'wl_pos_fit' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"            xana.find_wl(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, optimizer='model',ufac={wl_ufac})"
            ln += 1
            code[
                ln] = f"        if 'wl_pos_dir' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"            xana.find_wl(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, optimizer='direct',ufac={wl_ufac})"
            ln += 1
            code[
                ln] = f"        if any([ii in {self.analysis_saving_items} for ii in ['centroid_of_eng', 'centroid_of_eng_relative_to_wl', 'weighted_attenuation', 'weighted_eng']]):"
            ln += 1
            code[
                ln] = f"            xana.cal_wgt_eng(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e)"
            ln += 1
            code[
                ln] = f"        for jj in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"            if jj == 'wl_pos_fit':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.wl_pos_fit)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'wl_pos_dir':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.wl_pos_dir)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'wl_fit_coef':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][0].reshape(xana.model['wl']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'wl_fit_err':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'centroid_of_eng':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.centroid_of_eng)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'centroid_of_eng_relative_to_wl':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.centroid_of_eng_rel_wl)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'weighted_attenuation':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.weighted_atten)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'weighted_eng':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.weighted_eng)[:]"
            ln += 1
            code[
                ln] = f""
            ln += 1
            code[
                ln] = f"    elif xanes2D_analysis_type == 'full':"
            ln += 1
            code[
                ln] = "        _g12 = {}"
            ln += 1
            code[
                ln] = f"        for jj in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"            if jj == 'wl_fit_coef':"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='wl_fit_coef')}, *xanes2D_analysis_data_shape[1:]), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"            elif jj == 'edge_fit_coef':"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='edge_fit_coef')}, *xanes2D_analysis_data_shape[1:]), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"            elif jj in ['pre_edge_fit_coef', 'post_edge_fit_coef']:"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=({self.fit_coef_num(ftype='pre_edge_fit_coef')}, *xanes2D_analysis_data_shape[1:]), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"            elif jj == 'norm_spec':"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=(xanes2D_analysis_data_shape), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"            elif jj == 'lcf_fit':"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=([xana.lcf_ref.index.size, *xanes2D_analysis_data_shape[1:]]), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"            else:"
            ln += 1
            code[
                ln] = f"                _g12[jj] = g12.create_dataset(jj, shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)"
            ln += 1
            code[
                ln] = f"        xana.cal_pre_edge_sd()"
            ln += 1
            code[
                ln] = f"        xana.cal_post_edge_sd()"
            ln += 1
            code[
                ln] = f"        xana.cal_pre_edge_mean()"
            ln += 1
            code[
                ln] = f"        xana.cal_post_edge_mean()"
            ln += 1
            code[
                ln] = f"        xana.full_spec_preprocess(xanes2D_analysis_edge_eng, order=xanes2D_analysis_edge_fit_order, save_pre_post=True)"
            ln += 1
            code[
                ln] = f"        if np.any(np.array([ii in {self.analysis_saving_items} for ii in {edge_fit_item}])):"
            ln += 1
            code[
                ln] = f"            xana.fit_edge(xanes2D_analysis_edge50_fit_s, xanes2D_analysis_edge50_fit_e, **{edge_params})"
            ln += 1
            code[
                ln] = f"        if np.any(np.array([ii in {self.analysis_saving_items} for ii in {wl_fit_item}])):"
            ln += 1
            code[
                ln] = f"            xana.fit_edge(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, **{wl_params})"
            ln += 1
            code[
                ln] = f"        if ('edge50_pos_fit' in {self.analysis_saving_items}):"
            ln += 1
            code[
                ln] = f"            xana.find_edge_50(xanes2D_analysis_edge50_fit_s, xanes2D_analysis_edge50_fit_e, xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, optimizer='{edge50_optimizer}', ufac={edge_ufac})"
            ln += 1
            code[
                ln] = f"        if 'edge50_pos_dir' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"            xana.find_edge_50(xanes2D_analysis_edge50_fit_s, xanes2D_analysis_edge50_fit_e, xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, optimizer='none', ufac={edge_ufac})"
            ln += 1
            code[
                ln] = f"        if ('edge_pos_fit' in {self.analysis_saving_items}):"
            ln += 1
            code[
                ln] = f"            xana.find_edge_deriv(xanes2D_analysis_edge50_fit_s, xanes2D_analysis_edge50_fit_e, optimizer='model',ufac={edge_ufac})"
            ln += 1
            code[
                ln] = f"        if 'edge_pos_dir' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"            xana.find_edge_deriv(xanes2D_analysis_edge50_fit_s, xanes2D_analysis_edge50_fit_e, optimizer='direct',ufac={edge_ufac})"
            ln += 1
            code[
                ln] = f"        if 'wl_pos_fit' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"            xana.find_wl(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, optimizer='model',ufac={wl_ufac})"
            ln += 1
            code[
                ln] = f"        if 'wl_pos_dir' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"            xana.find_wl(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, optimizer='direct',ufac={wl_ufac})"
            ln += 1
            code[
                ln] = f"        if any([ii in {self.analysis_saving_items} for ii in ['centroid_of_eng', 'centroid_of_eng_relative_to_wl', 'weighted_attenuation', 'weighted_eng']]):"
            ln += 1
            code[
                ln] = f"            xana.cal_wgt_eng(xanes2D_analysis_pre_edge_e + xanes2D_analysis_edge_eng, xanes2D_analysis_wl_fit_eng_e)"
            ln += 1
            code[
                ln] = f"        if 'lcf_fit' in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"            xana.interp_ref_spec()"
            ln += 1
            code[
                ln] = f"            xana.lcf()"
            ln += 1
            code[
                ln] = f""
            ln += 1
            code[
                ln] = f"        for jj in {self.analysis_saving_items}:"
            ln += 1
            code[
                ln] = f"            if jj == 'wl_pos_fit':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.wl_pos_fit)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'wl_pos_dir':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.wl_pos_dir)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'wl_peak_height_dir':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.direct_wl_ph)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'centroid_of_eng':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.centroid_of_eng)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'centroid_of_eng_relative_to_wl':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.centroid_of_eng_rel_wl)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'weighted_attenuation':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.weighted_atten)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'weighted_eng':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.weighted_eng)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'edge50_pos_fit':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.edge50_pos_fit)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'edge50_pos_dir':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.edge50_pos_fit_none)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'edge_pos_fit':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.edge_pos_fit)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'edge_pos_dir':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.edge_pos_dir)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'edge_jump_filter':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.edge_jump_mask)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'edge_offset_filter':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.fitted_edge_mask)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'pre_edge_sd':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.pre_edge_sd_map)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'post_edge_sd':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.post_edge_sd_map)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'pre_edge_mean':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.pre_edge_mean_map)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'post_edge_mean':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.post_edge_mean_map)[:]"
            ln += 1
            code[
                ln] = f"            if jj  == 'pre_edge_fit_coef':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.pre_edge_fit_rlt[0].reshape(xana.pre_edge_fit_rlt[0].shape[0], *xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'post_edge_fit_coef':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.post_edge_fit_rlt[0].reshape(xana.post_edge_fit_rlt[0].shape[0], *xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'edge_fit_coef':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.model['edge']['fit_rlt'][0].reshape(xana.model['edge']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'edge_fit_err':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.model['edge']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'wl_fit_coef':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][0].reshape(xana.model['wl']['fit_rlt'][0].shape[0], *xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'wl_fit_err':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.model['wl']['fit_rlt'][1].reshape(*xana.spec.shape[1:]))[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'norm_spec':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.norm_spec)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'lcf_fit':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.lcf_fit)[:]"
            ln += 1
            code[
                ln] = f"            if jj == 'lcf_fit_err':"
            ln += 1
            code[
                ln] = f"                _g12[jj][:] = np.float32(xana.lcf_fit_err)[:]"
            ln += 1
            code[
                ln] = f"print('xanes2D analysis is done!')"
            ln += 1
            code[
                ln] = f""
            ln += 1

            gen_external_py_script(
                self.parent_h.xanes_fit_external_command_name, code)
            sig = os.system(
                f'python {self.parent_h.xanes_fit_external_command_name}')
            if sig == 0:
                self.hs['FitRun text'].value = 'XANES2D analysis is done ...'
            else:
                self.hs['FitRun text'].value = 'somthing wrong in analysis ...'
            self.parent_h.update_xanes2D_config()
        self.parent_h.boxes_logic()
