#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:56:17 2020

@author: xiao
"""
import h5py
import numpy as np

from ipywidgets import widgets
from pathlib import Path

import napari

from ..utils.io import (h5_lazy_reader, tif_writer, tif_seq_writer, raw_writer,
                        raw_seq_writer, asc_writer)
from .gui_components import (SelectFilesButton, NumpyArrayEncoder,
                             enable_disable_boxes, fiji_viewer_state,
                             fiji_viewer_on, fiji_viewer_off,
                             update_json_content, update_global_cwd)
from ..dicts import customized_struct_dict as dat_dict

napari.gui_qt()


class conv_data_gui():

    def __init__(self, parent_h, form_sz=[650, 740]):
        self.gui_name = 'misc'
        self.form_sz = form_sz
        self.global_h = parent_h
        self.hs = {}

        self.reader = None
        # self.misc_algn_ext_cmd_fn = os.path.join(
        #     os.path.abspath(os.path.curdir), 'misc_align_external_command.py')
        self.misc_algn_ext_cmd_fn = str(Path.resolve(
            Path.cwd()).joinpath('misc_align_external_command.py'))

        self.in_fn_set = False
        self.in_dat_cfmd = False
        self.in_dat_type_chgd = False
        self.out_fn_set = False

        self.in_dat_type = '2D XANES rlt'
        self.in_dat_fn = None
        self.in_dat_items = []
        self.in_dat_slcd_item = ''
        self.in_dat_path_in_h5 = ''

        self.out_dat_type = '3D tiff'
        self.out_fn_path = None
        self.out_fn_fnt = None  # for 2D img sequence
        self.out_dat_id_s = 0
        self.out_dat_id_dgt = 5

    def build_gui(self):
        base_wz_os = 92
        ex_ws_os = 6

        ## define Tabs
        layout = {
            'border': '3px solid #FFCC00',
            'width': f'{self.form_sz[1] - 86}px',
            'height': f'{self.form_sz[0] - 136}px'
        }
        self.hs['ConvertData form'] = widgets.VBox(layout=layout)

        ## ## ## define functional field for ConvertData Form
        ## ## ## ## define functional field for ConvData Accor
        self.hs['ConvData acc'] = widgets.Accordion(
            titles=('Select Data to be Converted',
                    'Define the Format to be Converted to'),
            layout={
                'width': 'auto',
                'border': '3px solid #8855AA',
                'align-content': 'center',
                'align-items': 'center',
                'justify-content': 'center'
            })

        self.hs['ConvDatInCfg box'] = widgets.VBox(
            layout={
                'width': 'auto',
                'height': f'{0.65*(self.form_sz[0] - 136)}px'
            })

        conv_in_data_GridspecLayout = widgets.GridspecLayout(14,
                                                             100,
                                                             layout={
                                                                 'width':
                                                                 'auto',
                                                                 'height':
                                                                 'auto'
                                                             })

        ## ## ## ## ## Title of "Input Data Config"
        conv_in_data_GridspecLayout[0, 40:70] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': '100%',
                'grid_template_columns': '80% auto',
                'grid_template_rows': 'auto',
                'grid_gap': '0px 2px'
            })
        self.hs['ConvDatInTypTtl box'] = conv_in_data_GridspecLayout[0, 40:70]
        self.hs['ConvDatInTypTtl txt'] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + 'Input Data Config' + '</span>',
            layout={'height': '90%'})
        self.hs['ConvDatInTypTtl box'].children = [
            self.hs['ConvDatInTypTtl txt']
        ]

        ## ## ## ## ## Input Data Type
        conv_in_data_GridspecLayout[1, :] = widgets.Label('Input Data Type')
        self.hs['ConvDatInTyp lbl'] = conv_in_data_GridspecLayout[1, :]
        conv_in_data_GridspecLayout[2, :] = widgets.GridBox(
            layout={
                'width': '100%',
                'height': '100%',
                'grid_template_columns': '80% auto',
                'grid_template_rows': 'auto',
                'grid_gap': '0px 2px'
            })
        self.hs['ConvDatInTyp box'] = conv_in_data_GridspecLayout[2, :]
        self.hs['ConvDatInTyp btn'] = widgets.RadioButtons(
            options=['Tomo raw', '2D XANES rlt', '3D XANES rlt'],
            value='2D XANES rlt',
            layout={
                'width': 'auto',
                'height': '90%'
            },
            orientation='horizontal')

        self.hs['ConvDatInTyp box'].children = [self.hs['ConvDatInTyp btn']]

        ## ## ## ## ## Input Data Filename
        conv_in_data_GridspecLayout[3, :] = widgets.HBox(
            layout={
                'width': '100%',
                'height': '100%',
                'grid_template_columns': '80% auto',
                'grid_template_rows': 'auto',
                'grid_gap': '0px 2px',
            })
        self.hs['ConvDatInFn box'] = conv_in_data_GridspecLayout[3, :]

        self.hs['ConvDatInFn txt'] = widgets.Text(
            value='Select a data file ...',
            disabled=True,
            layout={
                'width': '80%',
                'height': '90%'
            })

        self.hs['ConvDatInFn btn'] = SelectFilesButton(
            option='askopenfilename',
            text_h=self.hs['ConvDatInFn txt'],
            **{'open_filetypes': (('h5 files', ['*.h5', '*.hdf']), )},
            layout={
                'left': '1%',
                'width': '18%',
                'height': '90%'
            })
        self.hs['ConvDatInFn btn'].description = 'Select File'
        self.hs['ConvDatInFn btn'].style.button_color = 'orange'

        self.hs['ConvDatInFn box'].children = [
            self.hs['ConvDatInFn txt'], self.hs['ConvDatInFn btn']
        ]

        conv_in_data_GridspecLayout[4:14, :] = widgets.VBox(
            layout={
                'width': '100%',
                'height': '100%',
                'border': '1px solid #FFCC11',
            })
        self.hs['ConvDatIn box'] = conv_in_data_GridspecLayout[4:14, :]

        self.hs['ConvDatInItm box'] = widgets.HBox(layout={
            'width': '100%',
            'height': '88%'
        })
        self.hs['ConvDatInItm sel'] = widgets.Select(options=[''],
                                                     layout={
                                                         'left': '3%',
                                                         'width': '30%',
                                                         'height': '98%'
                                                     },
                                                     disabled=True)
        self.hs['ConvDatInItmInfo txt'] = widgets.Textarea(value='',
                                                           layout={
                                                               'left': '3%',
                                                               'width': '61%',
                                                               'height': '98%'
                                                           },
                                                           disabled=True)
        self.hs['ConvDatInItm box'].children = [
            self.hs['ConvDatInItm sel'], self.hs['ConvDatInItmInfo txt']
        ]
        self.hs['ConvDatInItmCmf btn'] = widgets.Button(description='Confirm',
                                                        layout={
                                                            'top': '1%',
                                                            'left': '41%',
                                                            'width': '18%',
                                                            'height': '10%'
                                                        },
                                                        disabled=True)
        self.hs['ConvDatInItmCmf btn'].style.button_color = 'darkviolet'

        self.hs['ConvDatInTyp btn'].observe(self.ConvDatInType_radbtn_chg,
                                            names='value')
        self.hs['ConvDatInFn btn'].on_click(self.ConvDatInFn_btn_clk)
        self.hs['ConvDatInItm sel'].observe(self.ConvDatInItms_sel_chg,
                                            names='value')
        self.hs['ConvDatInItmCmf btn'].on_click(self.ConvDatInItmCmf_btn_clk)

        self.hs['ConvDatIn box'].children = [
            self.hs['ConvDatInItm box'], self.hs['ConvDatInItmCmf btn']
        ]

        ## ## ## ## define functional field for ConvDataOut
        self.hs['ConvDatOutCfg box'] = widgets.VBox(
            layout={
                'width': 'auto',
                'height': f'{0.45*(self.form_sz[0] - 136)}px'
            })

        conv_out_data_GridspecLayout = widgets.GridspecLayout(8,
                                                              100,
                                                              layout={
                                                                  'width':
                                                                  'auto',
                                                                  'height':
                                                                  'auto'
                                                              })

        ## ## ## ## Title of "Output Data Config"
        conv_out_data_GridspecLayout[0, 40:70] = widgets.GridBox(
            layout={
                'width': 'auto',
                'height': '100%',
                'grid_template_columns': '80% auto',
                'grid_template_rows': 'auto',
                'grid_gap': '0px 2px'
            })
        self.hs['ConvDatOutTypTtl box'] = conv_out_data_GridspecLayout[0,
                                                                       40:70]
        self.hs['ConvDatOutTypTtl txt'] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + 'Output Data Config' + '</span>',
            layout={'height': '90%'})
        self.hs['ConvDatOutTypTtl box'].children = [
            self.hs['ConvDatOutTypTtl txt']
        ]

        ## ## ## ## Output Data Type
        conv_out_data_GridspecLayout[1, :] = widgets.Label('Output Data Type')
        self.hs['ConvDatOutTyp lbl'] = conv_out_data_GridspecLayout[1, :]
        conv_out_data_GridspecLayout[2, :] = widgets.GridBox(layout={
            'width': 'auto',
            'height': '100%',
        })
        self.hs['ConvDatOutTyp box'] = conv_out_data_GridspecLayout[2, :]
        self.hs['ConvDatOutTyp btn'] = widgets.RadioButtons(
            options=['2D tiff img seq', '2D raw img seq', '3D tiff', '3D raw'],
            value='3D tiff',
            layout={
                'width': 'auto',
                'height': '90%',
                'grid_template_columns': '20% 20% 20% 20%',
                'grid_template_rows': 'auto',
                'grid_gap': '0px 2px'
            },
            orientation='horizontal',
            disabled=True)

        self.hs['ConvDatOutTyp box'].children = [self.hs['ConvDatOutTyp btn']]

        ## ## ## ## Output Data Filename
        conv_out_data_GridspecLayout[3, :] = widgets.HBox(
            layout={
                'width': '100%',
                'height': '100%',
                'grid_template_columns': '80% auto',
                'grid_template_rows': 'auto',
                'grid_gap': '0px 2px',
            })
        self.hs['ConvDatOutFn box'] = conv_out_data_GridspecLayout[3, :]

        self.hs['ConvDatOutFn txt'] = widgets.Text(value='Save as ...',
                                                   disabled=True,
                                                   layout={
                                                       'width': '80%',
                                                       'height': '90%'
                                                   })
        self.hs['ConvDatOutFn btn'] = SelectFilesButton(
            option='asksaveasfilename',
            text_h=self.hs['ConvDatOutFn txt'],
            **{'save_filetypes': (('tiff files', '*.tiff'),)},
            layout={
                'left': '1%',
                'width': '18%',
                'height': '90%'
            })
        self.hs['ConvDatOutFn btn'].description = 'Save as ...'
        self.hs['ConvDatOutFn btn'].style.button_color = 'orange'
        self.hs['ConvDatOutFn btn'].disabled = True

        self.hs['ConvDatOutFn box'].children = [
            self.hs['ConvDatOutFn txt'], self.hs['ConvDatOutFn btn']
        ]

        ## ## ## ## Title of "Output Data Config" for 2D Image Sequence
        conv_out_data_GridspecLayout[4,
                                     38:78] = widgets.GridBox(layout={
                                         'width': 'auto',
                                         'height': 'auto'
                                     })
        self.hs['ConvDatOut2DImgSeqTtl box'] = conv_out_data_GridspecLayout[
            4, 38:78]
        self.hs['ConvDatOut2DImgSeqTtl txt'] = widgets.HTML(
            '<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">'
            + 'For 2D Image Sequence' + '</span>')
        self.hs['ConvDatOut2DImgSeqTtl box'].children = [
            self.hs['ConvDatOut2DImgSeqTtl txt']
        ]

        conv_out_data_GridspecLayout[5, :] = widgets.HBox(layout={
            'width': '100%',
            'height': '100%',
            # 'border': '1px solid #FFCC11',
        })
        self.hs['ConvDatOut2DCfg box'] = conv_out_data_GridspecLayout[5, :]

        self.hs['ConvDatOut2DFnTmp txt'] = widgets.Text(
            description='Name',
            value='',
            layout={
                'width': '40%',
                'height': 'auto'
            },
            tooltip='File base name template',
            disabled=True)
        self.hs['ConvDatOut2DIdSta txt'] = widgets.IntText(
            description='Start At',
            min=0,
            layout={
                'width': '20%',
                'height': 'auto'
            },
            tooltip='Starting index for saving the image sequence',
            disabled=True)
        self.hs['ConvDatOut2DIdDigt txt'] = widgets.IntText(
            description='Digits',
            value=5,
            min=1,
            max=8,
            layout={
                'width': '20%',
                'height': 'auto'
            },
            tooltip=
            'Digits of the index in naming individual image file in the sequence',
            disabled=True)
        self.hs['ConvDatOut2DCfg box'].children = [
            self.hs['ConvDatOut2DFnTmp txt'], self.hs['ConvDatOut2DIdSta txt'],
            self.hs['ConvDatOut2DIdDigt txt']
        ]

        conv_out_data_GridspecLayout[7, :] = widgets.Button(
            description='Convert',
            layout={
                'top': '1%',
                'left': '41%',
                'width': '18%',
                'height': 'auto'
            },
            disabled=True)
        self.hs['ConvDatOutItmCnv btn'] = conv_out_data_GridspecLayout[7, :]
        self.hs['ConvDatOutItmCnv btn'].style.button_color = 'darkviolet'

        # self.hs['ConvDatOutTyp btn'].observe(self.ConvDatOutTyp_radbtn_chg,
        #                                      names='value')
        self.hs['ConvDatOutFn btn'].on_click(self.ConvDatOutFn_btn_clk)
        self.hs['ConvDatOutTyp btn'].on_trait_change(
            self.ConvDatOutTyp_radbtn_chg, name='value')
        # self.hs['ConvDatOutFn btn'].on_trait_change(self.ConvDatOutFn_btn_clk,
        #                                             name='files')

        self.hs['ConvDatOut2DFnTmp txt'].observe(
            self.ConvDatOut2DFnTmp_txt_chg, names='value')
        self.hs['ConvDatOut2DIdSta txt'].observe(
            self.ConvDatOut2DIdSta_txt_chg, names='value')
        self.hs['ConvDatOut2DIdDigt txt'].observe(
            self.ConvDatOut2DIdDigt_txt_chg, names='value')
        self.hs['ConvDatOutItmCnv btn'].on_click(self.ConvDatOutItmCnv_btn_clk)

        ## ## ## Integrate Widgets
        self.hs['ConvDatInCfg box'].children = [conv_in_data_GridspecLayout]
        self.hs['ConvDatOutCfg box'].children = [conv_out_data_GridspecLayout]
        self.hs['ConvData acc'].children = [
            self.hs['ConvDatInCfg box'], self.hs['ConvDatOutCfg box']
        ]

        # per https://github.com/jupyter-widgets/ipywidgets/issues/2790, accordion title can only be set
        # via set_title in jupyterlab 7.5. It should be set directly when accordion is created after jupyterlab 8.
        self.hs['ConvData acc'].set_title(0, 'Select Data to be Converted')
        self.hs['ConvData acc'].set_title(
            1, 'Define the Format to be Converted to')
        self.hs['ConvData acc'].selected_index = None
        self.hs['ConvertData form'].children = [self.hs['ConvData acc']]

        self.hs['ConvDatInFn btn'].initialdir = self.global_h.cwd
        self.hs['ConvDatOutFn btn'].initialdir = self.global_h.cwd

    def lock_text_boxes(self):
        boxes = ['ConvDatInFn txt', 'ConvDatInItmInfo txt', 'ConvDatOutFn txt']
        enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)

    def boxes_logic(self):
        if self.in_dat_type_chgd:
            self.hs[
                'ConvDatInFn txt'].value = 'File does not match the selected data type or exportable datasets'
            self.hs['ConvDatInFn btn'].style.button_color = 'orange'
            self.hs['ConvDatInItm sel'].options = []
            self.hs['ConvDatInItmInfo txt'].value = ''
            self.hs['ConvDatOutFn txt'].value = ''
            self.hs['ConvDatOutFn btn'].style.button_color = 'orange'
        if self.in_fn_set:
            boxes = ['ConvDatInItm sel', 'ConvDatInItmCmf btn']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
        else:
            boxes = ['ConvDatInItm sel', 'ConvDatInItmCmf btn']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        if self.in_dat_cfmd:
            boxes = ['ConvDatOutCfg box']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
        else:
            boxes = ['ConvDatOutCfg box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        if ('seq' in self.out_dat_type) and self.in_dat_cfmd:
            boxes = ['ConvDatOut2DCfg box']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
        else:
            boxes = ['ConvDatOut2DCfg box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        if self.out_fn_set:
            self.hs['ConvDatOutItmCnv btn'].disabled = False
        else:
            self.hs['ConvDatOutFn btn'].style.button_color = 'orange'
            self.hs['ConvDatOutFn txt'].value = ''
            self.hs['ConvDatOutItmCnv btn'].disabled = True
        self.lock_text_boxes()

    def ConvDatInType_radbtn_chg(self, a):
        self.in_dat_type = a['owner'].value
        self.in_dat_type_chgd = True
        self.in_fn_set = False
        self.in_dat_cfmd = False
        self.boxes_logic()

    def ConvDatInFn_btn_clk(self, a):
        self.in_dat_items = []
        if len(a.files[0]) != 0:
            self.in_dat_fn = a.files[0]
            with h5py.File(self.in_dat_fn, 'r') as f:
                keys = list(f.keys())
                if self.in_dat_type == 'Tomo raw':
                    if 'img_tomo' in keys:
                        self.in_dat_items = ['img_bkg', 'img_dark', 'img_tomo']
                elif self.in_dat_type == '2D XANES rlt':
                    if ('processed_XANES2D' in keys):
                        self.in_dat_items = list(
                            f['/processed_XANES2D/proc_spectrum'].keys())
                        if 'gen_masks' in f['/processed_XANES2D'].keys():
                            self.in_dat_items += list(
                                f['/processed_XANES2D/gen_masks'].keys())
                    if 'registration_results' in keys:
                        try:
                            if ('registered_xanes2D'
                                    in f['/registration_results/reg_results'].
                                    keys()):
                                self.in_dat_items += ['eng_list', 'registered_xanes2D']
                        except:
                            pass
                else:
                    if ('processed_XANES3D' in keys):
                        self.in_dat_items = list(
                            f['/processed_XANES3D/proc_spectrum'].keys())
                        if 'gen_masks' in f['/processed_XANES3D'].keys():
                            self.in_dat_items += list(
                                f['/processed_XANES3D/gen_masks'].keys())
                    if 'registration_results' in keys:
                        try:
                            if ('registered_xanes3D'
                                    in f['/registration_results/reg_results'].
                                    keys()):
                                self.in_dat_items += ['eng_list', 'registered_xanes3D']
                        except:
                            pass

            if len(self.in_dat_items) == 0:
                self.in_fn_set = False
            else:
                self.hs['ConvDatInFn txt'].value = self.in_dat_fn
                self.in_dat_items = sorted(self.in_dat_items)
                self.in_dat_type_chgd = False
                self.hs['ConvDatInItm sel'].options = self.in_dat_items
                self.hs['ConvDatInItm sel'].index = 0
                self.global_h.cwd = str(Path.resolve(
                    Path(self.in_dat_fn).parent))
                update_global_cwd(self.global_h, self.global_h.cwd)
                self.in_fn_set = True
        else:
            self.in_fn_set = False
        self.in_dat_cfmd = False
        self.out_fn_set = False
        self.boxes_logic()

    def ConvDatInItms_sel_chg(self, a):
        self.in_dat_slcd_item = a['owner'].value
        if (self.in_dat_type == 'Tomo raw') and self.in_dat_slcd_item:
            with h5py.File(self.in_dat_fn, 'r') as f:
                self.in_dat_item_sz = f[dat_dict.TOMO_H5_ITEM_DICT[
                    self.in_dat_slcd_item]['path']].shape
            self.hs['ConvDatInItmInfo txt'].value = dat_dict.TOMO_H5_ITEM_DICT[
                self.in_dat_slcd_item][
                    'description'] + f"\nshape: {self.in_dat_item_sz}" + f"\ndtype: {dat_dict.TOMO_H5_ITEM_DICT[self.in_dat_slcd_item]['dtype']}"
            self.in_dat_path_in_h5 = dat_dict.TOMO_H5_ITEM_DICT[
                self.in_dat_slcd_item]['path']
        elif (self.in_dat_type == '2D XANES rlt') and self.in_dat_slcd_item:
            if 'mk' in self.in_dat_slcd_item:
                with h5py.File(self.in_dat_fn, 'r') as f:
                    self.in_dat_item_sz = f[
                        dat_dict.XANES2D_ANA_ITEM_DICT['mask']['path'].format(
                            self.in_dat_slcd_item)].shape
                self.hs[
                    'ConvDatInItmInfo txt'].value = dat_dict.XANES2D_ANA_ITEM_DICT[
                        'mask'][
                            'description'] + f"\nshape: {self.in_dat_item_sz}" + f"\ndtype: {dat_dict.XANES2D_ANA_ITEM_DICT['mask']['dtype']}"
                self.in_dat_path_in_h5 = dat_dict.XANES2D_ANA_ITEM_DICT[
                    'mask']['path'].format(self.in_dat_slcd_item)
            else:
                with h5py.File(self.in_dat_fn, 'r') as f:
                    self.in_dat_item_sz = f[dat_dict.XANES2D_ANA_ITEM_DICT[
                        self.in_dat_slcd_item]['path']].shape
                self.hs[
                    'ConvDatInItmInfo txt'].value = dat_dict.XANES2D_ANA_ITEM_DICT[
                        self.in_dat_slcd_item][
                            'description'] + f"\nshape: {self.in_dat_item_sz}" + f"\ndtype: {dat_dict.XANES2D_ANA_ITEM_DICT[self.in_dat_slcd_item]['dtype']}"
                self.in_dat_path_in_h5 = dat_dict.XANES2D_ANA_ITEM_DICT[
                    self.in_dat_slcd_item]['path']
        elif (self.in_dat_type == '3D XANES rlt') and self.in_dat_slcd_item:
            if 'mk' in self.in_dat_slcd_item:
                with h5py.File(self.in_dat_fn, 'r') as f:
                    self.in_dat_item_sz = f[
                        dat_dict.XANES3D_ANA_ITEM_DICT['mask']['path'].format(
                            self.in_dat_slcd_item)].shape
                self.hs[
                    'ConvDatInItmInfo txt'].value = dat_dict.XANES3D_ANA_ITEM_DICT[
                        'mask'][
                            'description'] + f"\nshape: {self.in_dat_item_sz}" + f"\ndtype: {dat_dict.XANES3D_ANA_ITEM_DICT['mask']['dtype']}"
                self.in_dat_path_in_h5 = dat_dict.XANES3D_ANA_ITEM_DICT[
                    'mask']['path'].format(self.in_dat_slcd_item)
            else:
                with h5py.File(self.in_dat_fn, 'r') as f:
                    self.in_dat_item_sz = f[dat_dict.XANES3D_ANA_ITEM_DICT[
                        self.in_dat_slcd_item]['path']].shape
                self.hs[
                    'ConvDatInItmInfo txt'].value = dat_dict.XANES3D_ANA_ITEM_DICT[
                        self.in_dat_slcd_item][
                            'description'] + f"\nshape: {self.in_dat_item_sz}" + f"\ndtype: {dat_dict.XANES3D_ANA_ITEM_DICT[self.in_dat_slcd_item]['dtype']}"
                self.in_dat_path_in_h5 = dat_dict.XANES3D_ANA_ITEM_DICT[
                    self.in_dat_slcd_item]['path']
        self.in_dat_cfmd = False
        self.out_fn_set = False
        self.boxes_logic()

    def ConvDatInItmCmf_btn_clk(self, a):
        ifn = Path.resolve(Path(self.in_dat_fn))
        self.out_fn_path = ifn.parent.joinpath(
            ifn.stem + '_export/', self.in_dat_slcd_item)
        if not self.out_fn_path.exists():
            Path.mkdir(self.out_fn_path, parents=True)
        self.out_fn_path = str(self.out_fn_path)
        self.out_fn_fnt = self.in_dat_slcd_item + '.tiff'

        if len(self.in_dat_item_sz) == 1:
            self.hs['ConvDatOutTyp btn'].options = ['ascii']
            self.hs['ConvDatOutTyp btn'].value = 'ascii'
            self.out_dat_type = 'ascii'
            self.hs['ConvDatOutFn btn'].save_filetypes = (('text files',
                                                           '*.asc'), )
            self.hs['ConvDatOutFn btn'].defaultextension = '.asc'
        elif len(self.in_dat_item_sz) == 2:
            self.hs['ConvDatOutTyp btn'].options = ['2D tiff', '2D raw']
            self.hs['ConvDatOutTyp btn'].value = '2D tiff'
            self.out_dat_type = '2D tiff'
            self.hs['ConvDatOutFn btn'].save_filetypes = (('tiff files',
                                                           '*.tiff'), )
            self.hs['ConvDatOutFn btn'].defaultextension = '.tiff'
        elif len(self.in_dat_item_sz) == 3:
            self.hs['ConvDatOutTyp btn'].options = [
                '2D tiff img seq', '2D raw img seq', '3D tiff', '3D raw'
            ]
            self.hs['ConvDatOutTyp btn'].value = '3D tiff'
            self.out_dat_type = '3D tiff'
            self.hs['ConvDatOutFn btn'].save_filetypes = (('tiff files',
                                                           '*.tiff'), )
            self.hs['ConvDatOutFn btn'].defaultextension = '.tiff'
        elif len(self.in_dat_item_sz) == 4:
            self.hs['ConvDatOutTyp btn'].options = [
                '3D tiff img seq', '3D raw img seq', '4D tiff', '4D raw'
            ]
            self.hs['ConvDatOutTyp btn'].value = '4D tiff'
            self.out_dat_type = '4D tiff'
            self.hs['ConvDatOutFn btn'].save_filetypes = (('tiff files', '*.tiff'),)
            self.hs['ConvDatOutFn btn'].defaultextension = '.tiff'
        self.hs['ConvDatOutFn btn'].initialdir = self.out_fn_path
        self.hs['ConvDatOutFn btn'].initialfile = self.in_dat_slcd_item
        self.in_dat_cfmd = True
        self.out_fn_set = False
        self.boxes_logic()

    def ConvDatOutTyp_radbtn_chg(self, a):
        self.out_dat_type = self.hs['ConvDatOutTyp btn'].value
        if 'seq' in self.out_dat_type:
            self.hs['ConvDatOutFn btn'].option = 'askdirectory'
            self.hs['ConvDatOutFn btn'].description = 'Select dir'
            if 'tiff' in self.out_dat_type:
                self.hs['ConvDatOutFn btn'].defaultextension = '.tiff'
            else:
                self.hs['ConvDatOutFn btn'].defaultextension = '.raw'
            self.hs['ConvDatOut2DFnTmp txt'].value = ''
            self.hs['ConvDatOut2DFnTmp txt'].value = str(Path(self.out_fn_fnt).stem).strip('_{}')
        else:
            self.hs['ConvDatOutFn btn'].option = 'asksaveasfilename'
            self.hs['ConvDatOutFn btn'].description = 'Save as'
            if 'tiff' in self.out_dat_type:
                self.hs['ConvDatOutFn btn'].save_filetypes = (('tiff files', '*.tiff'),)
                self.hs['ConvDatOutFn btn'].defaultextension = '.tiff'
            elif 'raw' in self.out_dat_type:
                self.hs['ConvDatOutFn btn'].save_filetypes = (('raw files', '*.raw'),)
                self.hs['ConvDatOutFn btn'].defaultextension = '.raw'
            elif 'ascii' in self.out_dat_type:
                self.hs['ConvDatOutFn btn'].save_filetypes = (('text files',
                                                               '*.asc'), )
                self.hs['ConvDatOutFn btn'].defaultextension = '.asc'
        self.out_fn_set = False
        self.boxes_logic()

    def ConvDatOutFn_btn_clk(self, a):
        if len(self.hs['ConvDatOutFn btn'].files[0]) == 0:
            self.out_fn_set = False
        else:
            if 'seq' in self.out_dat_type:
                self.out_fn_path = str(Path.resolve(Path(self.hs['ConvDatOutFn btn'].files[0])))
                self.hs['ConvDatOut2DFnTmp txt'].value = str(Path(self.out_fn_fnt).stem).strip('_{}')
            else:
                self.out_fn_path = str(Path.resolve(Path(self.hs['ConvDatOutFn btn'].files[0]).parent))
                self.out_fn_fnt = Path(self.hs['ConvDatOutFn btn'].files[0]).name
            self.out_fn_set = True
        self.boxes_logic()

    def ConvDatOut2DFnTmp_txt_chg(self, a):
        self.boxes_logic()

    def ConvDatOut2DIdSta_txt_chg(self, a):
        self.out_dat_id_s = a['owner'].value
        self.boxes_logic()

    def ConvDatOut2DIdDigt_txt_chg(self, a):
        self.out_dat_id_dgt = a['owner'].value
        self.boxes_logic()

    def ConvDatOutItmCnv_btn_clk(self, a):
        dlyd_arr = h5_lazy_reader(self.in_dat_fn, self.in_dat_path_in_h5,
                                  np.s_[:])
        if 'seq' in self.out_dat_type:
            if 'tiff' in self.out_dat_type:
                self.out_fn_fnt = self.hs[
                    'ConvDatOut2DFnTmp txt'].value + '_{}.tiff'
                fn = str(Path(self.out_fn_path).joinpath(self.out_fn_fnt))
                tif_seq_writer(fn,
                               dlyd_arr.compute(),
                               ids=self.out_dat_id_s,
                               digit=self.out_dat_id_dgt)
            else:
                self.out_fn_fnt = self.hs[
                    'ConvDatOut2DFnTmp txt'].value + '_{}.raw'
                fn = str(Path(self.out_fn_path).joinpath(self.out_fn_fnt))
                raw_seq_writer(fn,
                               dlyd_arr.compute(),
                               ids=self.out_dat_id_s,
                               digit=self.out_dat_id_dgt)
        else:
            fn = str(Path(self.out_fn_path).joinpath(self.out_fn_fnt))
            if 'tiff' in self.out_dat_type:
                tif_writer(fn, dlyd_arr.compute())
            elif 'raw' in self.out_dat_type:
                raw_writer(fn, dlyd_arr.compute())
            elif 'asc' in self.out_dat_type:
                asc_writer(fn, dlyd_arr.compute())
        self.boxes_logic()
