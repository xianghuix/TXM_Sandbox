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

from .gui_components import (SelectFilesButton, NumpyArrayEncoder, get_handles, enable_disable_boxes, 
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
        boxes = ['AnaDataSetupInfo text']
        enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        
    def boxes_logic(self):
        if (not self.ana_fn_seled):
            boxes = ['AnaDataSetupItem list',
                     'AnaDataSetupRead btn',
                     'AnaConfig box',
                     'AnaDataDisp box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        elif (self.ana_fn_seled & (not self.ana_data_existed)):
            boxes = ['AnaDataSetupItem list',
                     'AnaDataSetupRead btn',
                     'AnaConfig box',
                     'AnaDataDisp box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        elif ((self.ana_fn_seled & self.ana_data_existed) & 
              (not self.ana_data_configed)):
            boxes = ['AnaConfig box',
                     'AnaDataDisp box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            boxes = ['AnaDataSetupItem list',
                     'AnaDataSetupRead btn']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
        elif ((self.ana_fn_seled & self.ana_data_existed) & 
              self.ana_data_configed):
            boxes = ['AnaDataSetupItem list',
                     'AnaDataSetupRead btn',
                     'AnaConfig box',
                     'AnaDataDisp box']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
        
    def build_gui(self):
        self.hs['Ana form'] = widgets.Tab()
        self.hs['AnaData tab'] = widgets.VBox()
        self.hs['AnaPrep tab'] = widgets.VBox()
        self.hs['AnaConfig tab'] = widgets.VBox()
        self.hs['AnaDisp tab'] = widgets.VBox()
        self.hs['Ana form'].children = [self.hs['AnaData tab'],
                                        self.hs['AnaPrep tab'],
                                        self.hs['AnaConfig tab'],
                                        self.hs['AnaDisp tab']]
        layout = {'border':'3px solid #FFCC00', 'width':f'{0.96*self.form_sz[1]-98}px', 
                  'height':f'{0.89*(self.form_sz[0]-128)}px'}
        self.hs['AnaData tab'].layout = layout
        self.hs['AnaConfig tab'].layout = layout
        self.hs['AnaDisp tab'].layout = layout
        self.hs['Ana form'].set_title(0, 'Data Setup')
        self.hs['Ana form'].set_title(1, 'Data PreProc')
        self.hs['Ana form'].set_title(2, 'Ana Config')
        self.hs['Ana form'].set_title(3, 'Ana Disp')
        
        ## ## ## ## ## data setup -- start
        self.hs['DataSetup box'] = widgets.HBox()
        
        ## ## ## ## ## ## config parameters -- start
        data_setup_GridSpecLayout = widgets.GridspecLayout(16, 200,
                                                           layout={"border":"3px solid #FFCC00",
                                                                   "width":f"{0.96*self.form_sz[1]-98}px",
                                                                   "height":f"{0.88*(self.form_sz[0]-128)}px",
                                                                   "align_items":"flex-start",
                                                                   "justify_items":"flex-start"})           
        data_setup_GridSpecLayout[4, 5:25] = SelectFilesButton(option='askopenfilename',
                                                               description_tip="Open h5 file in which the processed XANES data is available",
                                                               **{'open_filetypes': (('h5 files', '*.h5'),)},
                                                               initialdir=self.parent_h.global_h.cwd)
        self.hs['AnaDataSetupOpen btn'] = data_setup_GridSpecLayout[4, 5:25]
        self.hs['AnaDataSetupOpen btn'].style.button_color = 'darkviolet'
        self.hs['AnaDataSetupOpen btn'].disabled = True
        data_setup_GridSpecLayout[:8, 30:150] = widgets.Select(options=[''],
                                                               value='',
                                                               rows=8,
                                                               description='item:',
                                                               disabled=True,
                                                               layout={'width':'100%', 'height':'100%'})
        self.hs['AnaDataSetupItem list'] = data_setup_GridSpecLayout[:8, 30:150]
        data_setup_GridSpecLayout[4, 160:180] = widgets.Button(description="Read Item",
                                                               description_tip="Read in the selected item",
                                                               disabled = True) 
        self.hs['AnaDataSetupRead btn'] = data_setup_GridSpecLayout[4, 160:180]
        self.hs['AnaDataSetupRead btn'].style.button_color = 'darkviolet'
        data_setup_GridSpecLayout[9:, :] = widgets.Textarea(value='Data Info',
                                                            placeholder='Data Info',
                                                            description='Data Info',
                                                            disabled=True,
                                                            layout={'width':'90%', 'height':'95%'})
        self.hs['AnaDataSetupInfo text'] = data_setup_GridSpecLayout[9:, :]
        
        self.hs['AnaDataSetupOpen btn'].on_click(self.ana_data_setup_open_btn_clk)
        self.hs['AnaDataSetupItem list'].observe(self.ana_data_setup_item_list_chg, names='value')
        self.hs['AnaDataSetupRead btn'].on_click(self.ana_data_setup_read_btn_clk)

        self.hs['DataSetup box'] = data_setup_GridSpecLayout
        self.hs['DataSetup box'].children = [self.hs['AnaDataSetupOpen btn'],
                                             self.hs['AnaDataSetupItem list'],
                                             self.hs['AnaDataSetupOpen btn'],
                                             self.hs['AnaDataSetupInfo text']]
        ## ## ## ## ## ## config parameters -- end
        
        self.hs['AnaData tab'].children = [self.hs['DataSetup box']]
        ## ## ## ## ## data setup -- end
        
        ## ## ## ## ##  config analysis -- start
        self.hs['AnaConfig box'] = widgets.HBox()
        
        ## ## ## ## ## ## config parameters -- start
        ana_setup_GridSpecLayout = widgets.GridspecLayout(16, 200,
                                                          layout = {"border":"3px solid #FFCC00",
                                                                    'width':f'{0.96*self.form_sz[1]-98}px',
                                                                    "height":f"{0.88*(self.form_sz[0]-128)}px",
                                                                    "align_items":"flex-start",
                                                                    "justify_items":"flex-start"})
        ana_setup_GridSpecLayout[0, 0:50] = widgets.Dropdown(description='ana type',
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
        self.hs['AnaType drpdn'] = ana_setup_GridSpecLayout[0, 0:50]
        ana_setup_GridSpecLayout[0, 50:100] = widgets.Dropdown(description='ana method',
                                                                            description_tooltip='choose analysis to apply to the selected data',
                                                                            options = ['KDE', 'PCA', 'KNN'],
                                                                            value ='KDE',
                                                                            disabled=True,
                                                                            layout={'width':f'{0.3*self.form_sz[1]-98}px'})
        self.hs['AnaMethod drpdn'] = ana_setup_GridSpecLayout[0, 50:100]
        
        for ii in range(2):
            for jj in range(4):
                ana_setup_GridSpecLayout[ii+1, jj*50:(jj+1)*50] = \
                    widgets.Dropdown(description = 'p{0}'.format(str(ii*4+jj).zfill(2)),
                                     value = 'linear',
                                     options = ['linear'],
                                     description_tooltip = "analysis function variable 0",
                                     layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                     disabled=True)
                self.hs['AnaConfigPars{1}'.format(1+ii*4+jj, ii*4+jj)] = \
                    ana_setup_GridSpecLayout[ii+1, jj*50:(jj+1)*50]
        
        for ii in range(2, 4):
            for jj in range(4):
                ana_setup_GridSpecLayout[ii+1, jj*50:(jj+1)*50] = \
                    widgets.BoundedFloatText(description = 'p{0}'.format(str(ii*4+jj).zfill(2)),
                                             value = 0,
                                             min = -1e5,
                                             max = 1e5,
                                             description_tooltip = "analysis function variable 0",
                                             layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                             disabled=True)
                self.hs['AnaConfigPars{1}'.format(1+ii*4+jj, ii*4+jj)] = \
                    ana_setup_GridSpecLayout[ii+1, jj*50:(jj+1)*50]
        
        ana_setup_GridSpecLayout[6, 10:190] = \
            widgets.IntProgress(value=0,
                                min=0,
                                max=100,
                                step=1,
                                description='Completing:',
                                bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                orientation='horizontal', 
                                indent=False,
                                layout={'width':'100%', 'height':'90%'})
        self.hs['AnaConfigPrgr bar'] = \
            ana_setup_GridSpecLayout[6, 10:190]
            
        ana_setup_GridSpecLayout[7, 90:110] = \
            widgets.Button(description='Compute',
                           description_tip='Perform the analysis',
                           disabled=True)
        self.hs['AnaConfigCmpt btn'] = \
            ana_setup_GridSpecLayout[7, 90:110]
        self.hs['AnaConfigCmpt btn'].style.button_color = 'darkviolet'
        
        self.hs['AnaType drpdn'].observe(self.ana_type_drpdn_chg, names='value')
        self.hs['AnaMethod drpdn'].observe(self.ana_meth_drpdn_chg, names='value')
        self.hs['AnaConfigPars0'].observe(self.ana_config_p0_chg, names='value')
        self.hs['AnaConfigPars1'].observe(self.ana_config_p1_chg, names='value')
        self.hs['AnaConfigPars2'].observe(self.ana_config_p2_chg, names='value')
        self.hs['AnaConfigPars3'].observe(self.ana_config_p3_chg, names='value')
        self.hs['AnaConfigPars4'].observe(self.ana_config_p4_chg, names='value')
        self.hs['AnaConfigPars5'].observe(self.ana_config_p5_chg, names='value')
        self.hs['AnaConfigPars6'].observe(self.ana_config_p6_chg, names='value')
        self.hs['AnaConfigPars7'].observe(self.ana_config_p7_chg, names='value')
        self.hs['AnaConfigCmpt btn'].on_click(self.ana_config_cmpt_btn_clk)
        self.hs['AnaConfig box'] = ana_setup_GridSpecLayout
        self.hs['AnaConfig box'].children = [self.hs['AnaType drpdn'],
                                             self.hs['AnaMethod drpdn'],
                                             self.hs['AnaConfigPars0'],
                                             self.hs['AnaConfigPars1'],
                                             self.hs['AnaConfigPars2'],
                                             self.hs['AnaConfigPars3'],
                                             self.hs['AnaConfigPars4'],
                                             self.hs['AnaConfigPars5'],
                                             self.hs['AnaConfigPars6'],
                                             self.hs['AnaConfigPars7'],
                                             self.hs['AnaConfigCmpt btn']]
        ## ## ## ## ## ## config parameters -- end
        
        self.hs['AnaConfig tab'].children = [self.hs['AnaConfig box']]
        ## ## ## ## ## config analysis -- end
        
        ## ## ## ## ##  result display -- start
        self.hs['AnaDataDisp box'] = widgets.HBox()
        ## ## ## ## ##  result display -- end
        
    def ana_data_setup_open_btn_clk(self, a):
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
                    self.hs['AnaDataSetupItem list'].options = \
                        self.ana_proc_data_list
                else:
                    self.ana_proc_data_list = [""]
        else:
            self.ana_fn_seled = False
            self.ana_proc_data_list = [""]
            self.hs['AnaDataSetupItem list'].options = \
                self.ana_proc_data_list
            self.hs['AnaDataSetupItem list'].value = \
                self.ana_proc_data_list[0]  
        self.boxes_logic()
        self.lock_message_text_boxes()
        
    def ana_data_setup_item_list_chg(self, a):
        with h5py.File(self.ana_fn, 'r') as f:
            try:
                shape = f[self.ana_proc_path_in_h5+'/'+
                          self.hs['AnaDataSetupItem list'].value].shape
            except:
                shape = None
            try:
                dtype = f[self.ana_proc_path_in_h5+'/'+
                          self.hs['AnaDataSetupItem list'].value].dtype
            except:
                dtype = None
            try:
                dmin = np.min(f[self.ana_proc_path_in_h5+'/'+
                                self.hs['AnaDataSetupItem list'].value])
            except:
                dmin = None
            try:
                dmax = np.max(f[self.ana_proc_path_in_h5+'/'+
                          self.hs['AnaDataSetupItem list'].value])
            except:
                dmax = None
        self.hs['AnaDataSetupInfo text'].value = 'shape: ' + str(shape) + '\n' +\
                                                                         'dtype: ' + str(dtype) + '\n' +\
                                                                         'data min: ' +str(dmin) + '\n' +\
                                                                         'data max: ' +str(dmax) 
        self.boxes_logic()
        self.lock_message_text_boxes()
    
    def ana_data_setup_read_btn_clk(self, a):
        with h5py.File(self.ana_fn, 'r') as f:
            self.ana_data = f[self.ana_proc_path_in_h5+'/'+
                              self.hs['AnaDataSetupItem list'].value][:]
            self.ana_data_configed = True
        self.boxes_logic()
        self.lock_message_text_boxes()
    
    def ana_type_drpdn_chg(self, a):
        pass
    
    def ana_meth_drpdn_chg(self, a):
        pass
    
    def ana_config_p0_chg(self, a):
        pass
    
    def ana_config_p1_chg(self, a):
        pass
    
    def ana_config_p2_chg(self, a):
        pass
    
    def ana_config_p3_chg(self, a):
        pass
    
    def ana_config_p4_chg(self, a):
        pass
    
    def ana_config_p5_chg(self, a):
        pass
    
    def ana_config_p6_chg(self, a):
        pass
    
    def ana_config_p7_chg(self, a):
        pass
    
    def ana_config_cmpt_btn_clk(self, a):
        pass
        
        
        