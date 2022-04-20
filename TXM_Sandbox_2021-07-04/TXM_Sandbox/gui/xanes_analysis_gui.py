#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 22:58:16 2020

@author: xiao
"""

from ipywidgets import widgets
from copy import deepcopy
import os, h5py, json
import numpy as np

import napari
napari.gui_qt()

from .gui_components import (SelectFilesButton, NumpyArrayEncoder, get_handles, 
                             enable_disable_boxes, 
                             gen_external_py_script, fiji_viewer_off, scale_eng_list)
from ..dicts.xanes_analysis_dict import XANES_ANA_METHOD

class xanes_analysis_gui():
    def __init__(self, parent_h, form_sz=[650, 740]):
        self.parent_h = parent_h
        self.hs = {}
        self.form_sz = form_sz    
        
        self.ana_fn_seled = False
        self.ana_data_existed = False
        self.ana_data_configed = False
        
        self.ana_proc_path_in_h5 = None
        self.ana_proc_data_list = [""]
        self.ana_data = None
        
    def lock_message_text_boxes(self):
        boxes = ['L[0][x][4][0][0][0][0][3]_ana_data_setup_info_txt']
        enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        
    def boxes_logic(self):
        if (not self.ana_fn_seled):
            boxes = ['L[0][x][4][0][0][0][0][1]_ana_data_setup_item_list',
                     'L[0][x][4][0][0][0][0][2]_ana_data_setup_read_btn',
                     'L[0][x][4][0][2][0]_ana_config_box',
                     'L[0][x][4][0][3][0]_ana_data_disp_box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        elif (self.ana_fn_seled & (not self.ana_data_existed)):
            boxes = ['L[0][x][4][0][0][0][0][1]_ana_data_setup_item_list',
                     'L[0][x][4][0][0][0][0][2]_ana_data_setup_read_btn',
                     'L[0][x][4][0][2][0]_ana_config_box',
                     'L[0][x][4][0][3][0]_ana_data_disp_box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        elif ((self.ana_fn_seled & self.ana_data_existed) & 
              (not self.ana_data_configed)):
            boxes = ['L[0][x][4][0][2][0]_ana_config_box',
                     'L[0][x][4][0][3][0]_ana_data_disp_box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            boxes = ['L[0][x][4][0][0][0][0][1]_ana_data_setup_item_list',
                     'L[0][x][4][0][0][0][0][2]_ana_data_setup_read_btn']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
        elif ((self.ana_fn_seled & self.ana_data_existed) & 
              self.ana_data_configed):
            boxes = ['L[0][x][4][0][0][0][0][1]_ana_data_setup_item_list',
                     'L[0][x][4][0][0][0][0][2]_ana_data_setup_read_btn',
                     'L[0][x][4][0][2][0]_ana_config_box',
                     'L[0][x][4][0][3][0]_ana_data_disp_box']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
        
    def build_gui(self):
        self.hs['L[0][x][4][0]_ana_box'] = widgets.Tab()
        # layout = {'border':'3px solid #FFCC00', 'width':f'{0.96*self.form_sz[1]-98}px', 'height':f'{0.92*(self.form_sz[0]-128)}px'}
        # self.hs['L[0][x][4][0]_ana_box'].layout = layout
        self.hs['L[0][x][4][0][0]_ana_data'] = widgets.VBox()
        self.hs['L[0][x][4][0][1]_ana_preproc'] = widgets.VBox()
        self.hs['L[0][x][4][0][2]_ana_config'] = widgets.VBox()
        self.hs['L[0][x][4][0][3]_ana_disp'] = widgets.VBox()
        self.hs['L[0][x][4][0]_ana_box'].children = [self.hs['L[0][x][4][0][0]_ana_data'],
                                                     self.hs['L[0][x][4][0][1]_ana_preproc'],
                                                     self.hs['L[0][x][4][0][2]_ana_config'],
                                                     self.hs['L[0][x][4][0][3]_ana_disp']]
        layout = {'border':'3px solid #FFCC00', 'width':f'{0.96*self.form_sz[1]-98}px', 'height':f'{0.89*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][4][0][0]_ana_data'].layout = layout
        self.hs['L[0][x][4][0][2]_ana_config'].layout = layout
        self.hs['L[0][x][4][0][3]_ana_disp'].layout = layout
        self.hs['L[0][x][4][0]_ana_box'].set_title(0, 'Data Setup')
        self.hs['L[0][x][4][0]_ana_box'].set_title(1, 'Data PreProc')
        self.hs['L[0][x][4][0]_ana_box'].set_title(2, 'Ana Config')
        self.hs['L[0][x][4][0]_ana_box'].set_title(3, 'Ana Disp')
        
        ## ## ## ## ## data setup -- start
        self.hs['L[0][x][4][0][0][0]_data_setup_box'] = widgets.HBox()
        
        ## ## ## ## ## ## config parameters -- start
        self.hs['L[0][x][4][0][0][0][0]_data_setup'] = widgets.GridspecLayout(16, 200,
                                                                               layout = {"border":"3px solid #FFCC00",
                                                                                          "width":f"{0.96*self.form_sz[1]-98}px",
                                                                                         "height":f"{0.88*(self.form_sz[0]-128)}px",
                                                                                         "align_items":"flex-start",
                                                                                         "justify_items":"flex-start"})           
        self.hs['L[0][x][4][0][0][0][0]_data_setup'][4, 5:25] = SelectFilesButton(option='askopenfilename',
                                                                                  description_tip="Open h5 file in which the processed XANES data is available",
                                                                                  **{'open_filetypes': (('h5 files', '*.h5'),)},
                                                                                  initialdir=self.parent_h.global_h.cwd)
        self.hs['L[0][x][4][0][0][0][0][0]_ana_data_setup_open_btn'] = self.hs['L[0][x][4][0][0][0][0]_data_setup'][4, 5:25]
        self.hs['L[0][x][4][0][0][0][0][0]_ana_data_setup_open_btn'].style.button_color = 'darkviolet'
        self.hs['L[0][x][4][0][0][0][0][0]_ana_data_setup_open_btn'].disabled = True
        self.hs['L[0][x][4][0][0][0][0]_data_setup'][:8, 30:150] = \
            widgets.Select(options=[''],
                           value='',
                           rows=8,
                           description='item:',
                           disabled=True,
                           layout={'width':'100%', 'height':'100%'})
        self.hs['L[0][x][4][0][0][0][0][1]_ana_data_setup_item_list'] = self.hs['L[0][x][4][0][0][0][0]_data_setup'][:8, 30:150]
        self.hs['L[0][x][4][0][0][0][0]_data_setup'][4, 160:180] = widgets.Button(description="Read Item",
                                                               description_tip="Read in the selected item",
                                                               disabled = True) 
        self.hs['L[0][x][4][0][0][0][0][2]_ana_data_setup_read_btn'] = self.hs['L[0][x][4][0][0][0][0]_data_setup'][4, 160:180]
        self.hs['L[0][x][4][0][0][0][0][2]_ana_data_setup_read_btn'].style.button_color = 'darkviolet'
        self.hs['L[0][x][4][0][0][0][0]_data_setup'][9:, :] = \
            widgets.Textarea(value='Data Info',
                             placeholder='Data Info',
                             description='Data Info',
                             disabled=True,
                             layout={'width':'90%', 'height':'95%'})
        self.hs['L[0][x][4][0][0][0][0][3]_ana_data_setup_info_txt'] = self.hs['L[0][x][4][0][0][0][0]_data_setup'][9:, :]
        
        self.hs['L[0][x][4][0][0][0][0][0]_ana_data_setup_open_btn'].on_click(self.L0_x_4_0_0_0_0_0_ana_data_setup_open_btn_click)
        self.hs['L[0][x][4][0][0][0][0][1]_ana_data_setup_item_list'].observe(self.L0_x_4_0_0_0_0_1_ana_data_setup_item_list_change, names='value')
        self.hs['L[0][x][4][0][0][0][0][2]_ana_data_setup_read_btn'].on_click(self.L0_x_4_0_0_0_0_2_ana_data_setup_read_btn_click)
        self.hs['L[0][x][4][0][0][0]_data_setup_box'].children = get_handles(self.hs, 'L[0][x][4][0][0][0]_data_setup_box', -1)
        ## ## ## ## ## ## config parameters -- end
        
        self.hs['L[0][x][4][0][0]_ana_data'].children = get_handles(self.hs, 'L[0][x][4][0][0]_ana_data', -1)
        ## ## ## ## ## data setup -- end
        
        ## ## ## ## ##  config analysis -- start
        self.hs['L[0][x][4][0][2][0]_ana_config_box'] = widgets.HBox()
        
        ## ## ## ## ## ## config parameters -- start
        self.hs['L[0][x][4][0][2][0][0]_ana_config'] = widgets.GridspecLayout(16, 200,
                                                                            layout = {"border":"3px solid #FFCC00",
                                                                                      'width':f'{0.96*self.form_sz[1]-98}px',
                                                                                      "height":f"{0.88*(self.form_sz[0]-128)}px",
                                                                                      "align_items":"flex-start",
                                                                                      "justify_items":"flex-start"})
        self.hs['L[0][x][4][0][2][0][0]_ana_config'][0, 0:50] = widgets.Dropdown(description='ana type',
                                                                            description_tooltip='choose analysis to apply to the selected data',
                                                                            options = ['Preproc', 
                                                                                       'Decomp', 
                                                                                       'Classif',
                                                                                       'Regres',
                                                                                       'Cluster',
                                                                                       'Neighbor'],
                                                                            value ='Neighbor',
                                                                            disabled=True,
                                                                            layout={'width':f'{0.3*self.form_sz[1]-98}px'})
        self.hs['L[0][x][4][0][2][0][0][0]_ana_type_drpdn'] = self.hs['L[0][x][4][0][2][0][0]_ana_config'][0, 0:50]
        self.hs['L[0][x][4][0][2][0][0]_ana_config'][0, 50:100] = widgets.Dropdown(description='ana method',
                                                                            description_tooltip='choose analysis to apply to the selected data',
                                                                            options = ['KDE', 'PCA', 'KNN'],
                                                                            value ='KDE',
                                                                            disabled=True,
                                                                            layout={'width':f'{0.3*self.form_sz[1]-98}px'})
        self.hs['L[0][x][4][0][2][0][0][1]_ana_method_drpdn'] = self.hs['L[0][x][4][0][2][0][0]_ana_config'][0, 50:100]
        
        for ii in range(2):
            for jj in range(4):
                self.hs['L[0][x][4][0][2][0][0]_ana_config'][ii+1, jj*50:(jj+1)*50] = \
                    widgets.Dropdown(description = 'p{0}'.format(str(ii*4+jj).zfill(2)),
                                    value = 'linear',
                                    options = ['linear'],
                                    description_tooltip = "analysis function variable 0",
                                    layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                    disabled=True)
                self.hs['L[0][x][4][0][2][0][0][{0}]_ana_config_p{1}'.format(1+ii*4+jj, ii*4+jj)] = \
                    self.hs['L[0][x][4][0][2][0][0]_ana_config'][ii+1, jj*50:(jj+1)*50]
        
        for ii in range(2, 4):
            for jj in range(4):
                self.hs['L[0][x][4][0][2][0][0]_ana_config'][ii+1, jj*50:(jj+1)*50] = \
                    widgets.BoundedFloatText(description = 'p{0}'.format(str(ii*4+jj).zfill(2)),
                                            value = 0,
                                            min = -1e5,
                                            max = 1e5,
                                            description_tooltip = "analysis function variable 0",
                                            layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                            disabled=True)
                self.hs['L[0][x][4][0][2][0][0][{0}]_ana_config_p{1}'.format(1+ii*4+jj, ii*4+jj)] = \
                    self.hs['L[0][x][4][0][2][0][0]_ana_config'][ii+1, jj*50:(jj+1)*50]
        
        self.hs['L[0][x][4][0][2][0][0]_ana_config'][6, 10:190] = \
            widgets.IntProgress(value=0,
                                min=0,
                                max=100,
                                step=1,
                                description='Completing:',
                                bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                orientation='horizontal', 
                                indent=False,
                                layout={'width':'100%', 'height':'90%'})
        self.hs['L[0][x][4][0][2][0][0][17]_ana_config_progr_bar'] = \
            self.hs['L[0][x][4][0][2][0][0]_ana_config'][6, 10:190]
            
        self.hs['L[0][x][4][0][2][0][0]_ana_config'][7, 90:110] = \
            widgets.Button(description='Compute',
                           description_tip='Perform the analysis',
                           disabled=True)
        self.hs['L[0][x][4][0][2][0][0][18]_ana_config_compute_btn'] = \
            self.hs['L[0][x][4][0][2][0][0]_ana_config'][7, 90:110]
        self.hs['L[0][x][4][0][2][0][0][18]_ana_config_compute_btn'].style.button_color = 'darkviolet'
        
        self.hs['L[0][x][4][0][2][0][0][0]_ana_type_drpdn'].observe(self.L0_x_4_0_2_0_0_0_ana_type_drpdn_change, names='value')
        self.hs['L[0][x][4][0][2][0][0][1]_ana_method_drpdn'].observe(self.L0_x_4_0_2_0_0_1_ana_method_drpdn_change, names='value')
        self.hs['L[0][x][4][0][2][0][0][1]_ana_config_p0'].observe(self.L0_x_4_0_2_0_0_1_ana_config_p0_change, names='value')
        self.hs['L[0][x][4][0][2][0][0][2]_ana_config_p1'].observe(self.L0_x_4_0_2_0_0_2_ana_config_p1_change, names='value')
        self.hs['L[0][x][4][0][2][0][0][3]_ana_config_p2'].observe(self.L0_x_4_0_2_0_0_3_ana_config_p2_change, names='value')
        self.hs['L[0][x][4][0][2][0][0][4]_ana_config_p3'].observe(self.L0_x_4_0_2_0_0_4_ana_config_p3_change, names='value')
        self.hs['L[0][x][4][0][2][0][0][5]_ana_config_p4'].observe(self.L0_x_4_0_2_0_0_5_ana_config_p4_change, names='value')
        self.hs['L[0][x][4][0][2][0][0][6]_ana_config_p5'].observe(self.L0_x_4_0_2_0_0_6_ana_config_p5_change, names='value')
        self.hs['L[0][x][4][0][2][0][0][7]_ana_config_p6'].observe(self.L0_x_4_0_2_0_0_7_ana_config_p6_change, names='value')
        self.hs['L[0][x][4][0][2][0][0][8]_ana_config_p7'].observe(self.L0_x_4_0_2_0_0_8_ana_config_p7_change, names='value')
        self.hs['L[0][x][4][0][2][0][0][18]_ana_config_compute_btn'].on_click(self.L0_x_4_0_2_0_0_18_ana_config_compute_btn_click)
        self.hs['L[0][x][4][0][2][0]_ana_config_box'].children = get_handles(self.hs, 'L[0][x][4][0][2][0]_ana_config_box', -1)
        ## ## ## ## ## ## config parameters -- end
        
        self.hs['L[0][x][4][0][2]_ana_config'].children = get_handles(self.hs, 'L[0][x][4][0][2]_ana_config', -1)
        ## ## ## ## ## config analysis -- end
        
        ## ## ## ## ##  result display -- start
        self.hs['L[0][x][4][0][3][0]_ana_data_disp_box'] = widgets.HBox()
        ## ## ## ## ##  result display -- end
        
    def L0_x_4_0_0_0_0_0_ana_data_setup_open_btn_click(self, a):
        if len(a.files[0]) != 0:
            self.ana_fn_seled = True
            self.ana_fn = os.path.abspath(a.files[0])
            with h5py.File(self.ana_fn, 'r') as f:
                for key in f['/']:
                    if 'processed_' in key:
                        self.ana_proc_path_in_h5 = key + '/proc_spectrum'
                        self.ana_data_existed = True
                        break
                    else:
                        self.ana_data_existed = False
                if self.ana_data_existed:
                    self.ana_proc_data_list = []
                    for key in f[self.ana_proc_path_in_h5]:
                        self.ana_proc_data_list.append(key)
                    self.hs['L[0][x][4][0][0][0][0][1]_ana_data_setup_item_list'].options = \
                        self.ana_proc_data_list
                    # self.hs['L[0][x][4][0][0][0][0][1]_ana_data_setup_item_list'].value = \
                    #     self.ana_proc_data_list[0]
                else:
                    self.ana_proc_data_list = [""]
                    # self.hs['L[0][x][4][0][0][0][0][1]_ana_data_setup_item_list'].value = \
                    #     self.ana_proc_data_list[0]            
        else:
            self.ana_fn_seled = False
            self.ana_proc_data_list = [""]
            self.hs['L[0][x][4][0][0][0][0][1]_ana_data_setup_item_list'].options = \
                self.ana_proc_data_list
            self.hs['L[0][x][4][0][0][0][0][1]_ana_data_setup_item_list'].value = \
                self.ana_proc_data_list[0]  
        self.boxes_logic()
        self.lock_message_text_boxes()
        
    def L0_x_4_0_0_0_0_1_ana_data_setup_item_list_change(self, a):
        with h5py.File(self.ana_fn, 'r') as f:
            try:
                shape = f[self.ana_proc_path_in_h5+'/'+
                          self.hs['L[0][x][4][0][0][0][0][1]_ana_data_setup_item_list'].value].shape
            except:
                shape = None
            try:
                dtype = f[self.ana_proc_path_in_h5+'/'+
                          self.hs['L[0][x][4][0][0][0][0][1]_ana_data_setup_item_list'].value].dtype
            except:
                dtype = None
            try:
                dmin = np.min(f[self.ana_proc_path_in_h5+'/'+
                                self.hs['L[0][x][4][0][0][0][0][1]_ana_data_setup_item_list'].value])
            except:
                dmin = None
            try:
                dmax = np.max(f[self.ana_proc_path_in_h5+'/'+
                          self.hs['L[0][x][4][0][0][0][0][1]_ana_data_setup_item_list'].value])
            except:
                dmax = None
        self.hs['L[0][x][4][0][0][0][0][3]_ana_data_setup_info_txt'].value = 'shape: ' + str(shape) + '\n' +\
                                                                         'dtype: ' + str(dtype) + '\n' +\
                                                                         'data min: ' +str(dmin) + '\n' +\
                                                                         'data max: ' +str(dmax) 
        self.boxes_logic()
        self.lock_message_text_boxes()
    
    def L0_x_4_0_0_0_0_2_ana_data_setup_read_btn_click(self, a):
        with h5py.File(self.ana_fn, 'r') as f:
            self.ana_data = f[self.ana_proc_path_in_h5+'/'+
                              self.hs['L[0][x][4][0][0][0][0][1]_ana_data_setup_item_list'].value][:]
            self.ana_data_configed = True
        self.boxes_logic()
        self.lock_message_text_boxes()
    
    def L0_x_4_0_2_0_0_0_ana_type_drpdn_change(self, a):
        pass
    
    def L0_x_4_0_2_0_0_1_ana_method_drpdn_change(self, a):
        pass
    
    def L0_x_4_0_2_0_0_1_ana_config_p0_change(self, a):
        pass
    
    def L0_x_4_0_2_0_0_2_ana_config_p1_change(self, a):
        pass
    
    def L0_x_4_0_2_0_0_3_ana_config_p2_change(self, a):
        pass
    
    def L0_x_4_0_2_0_0_4_ana_config_p3_change(self, a):
        pass
    
    def L0_x_4_0_2_0_0_5_ana_config_p4_change(self, a):
        pass
    
    def L0_x_4_0_2_0_0_6_ana_config_p5_change(self, a):
        pass
    
    def L0_x_4_0_2_0_0_7_ana_config_p6_change(self, a):
        pass
    
    def L0_x_4_0_2_0_0_8_ana_config_p7_change(self, a):
        pass
    
    def L0_x_4_0_2_0_0_18_ana_config_compute_btn_click(self, a):
        pass
        
        
        