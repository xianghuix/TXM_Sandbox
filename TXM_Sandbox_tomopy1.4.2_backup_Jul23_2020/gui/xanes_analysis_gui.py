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
from .gui_components import (NumpyArrayEncoder, get_handles, enable_disable_boxes, 
                            gen_external_py_script, fiji_viewer_off, scale_eng_list)
import napari
napari.gui_qt()

class xanes_analysis_gui():
    def __init__(self, parent_h, form_sz=[650, 740]):
        self.parent_h = parent_h
        self.hs = {}
        self.form_sz = form_sz     
        
    def build_gui(self):
        self.hs['L[0][x][4][0]_ana_box'] = widgets.Tab()
        # layout = {'border':'3px solid #FFCC00', 'width':f'{0.96*self.form_sz[1]-98}px', 'height':f'{0.92*(self.form_sz[0]-128)}px'}
        # self.hs['L[0][x][4][0]_ana_box'].layout = layout
        self.hs['L[0][x][4][0][0]_ana_data'] = widgets.VBox()
        self.hs['L[0][x][4][0][1]_ana_config'] = widgets.VBox()
        self.hs['L[0][x][4][0][2]_ana_disp'] = widgets.VBox()
        self.hs['L[0][x][4][0]_ana_box'].children = [self.hs['L[0][x][4][0][0]_ana_data'],
                                                     self.hs['L[0][x][4][0][1]_ana_config'],
                                                     self.hs['L[0][x][4][0][2]_ana_disp']]
        layout = {'border':'3px solid #FFCC00', 'width':f'{0.96*self.form_sz[1]-98}px', 'height':f'{0.89*(self.form_sz[0]-128)}px'}
        self.hs['L[0][x][4][0][0]_ana_data'].layout = layout
        self.hs['L[0][x][4][0][1]_ana_config'].layout = layout
        self.hs['L[0][x][4][0][2]_ana_disp'].layout = layout
        self.hs['L[0][x][4][0]_ana_box'].set_title(0, 'Data Setup')
        self.hs['L[0][x][4][0]_ana_box'].set_title(1, 'Ana Config')
        self.hs['L[0][x][4][0]_ana_box'].set_title(2, 'Ana Disp')
        
        ## ## ## ## data setup -- start
        
        ## ## ## ## data setup -- end
        
        ## ## ## ## ##  config analysis -- start
        self.hs['L[0][x][4][0][1][0]_ana_config_box'] = widgets.HBox()
        
        ## ## ## ## ## ## config parameters -- start
        self.hs['L[0][x][4][0][1][0][0]_ana_config'] = widgets.GridspecLayout(16, 200,
                                                                            layout = {"border":"3px solid #FFCC00",
                                                                                      'width':f'{0.96*self.form_sz[1]-98}px',
                                                                                      "height":f"{0.88*(self.form_sz[0]-128)}px",
                                                                                      "align_items":"flex-start",
                                                                                      "justify_items":"flex-start"})
        self.hs['L[0][x][4][0][1][0][0]_ana_config'][0, 0:50] = widgets.Dropdown(description='ana type',
                                                                            description_tooltip='choose analysis to apply to the selected data',
                                                                            options = ['kernel density', 'PCA', 'KNN'],
                                                                            value ='kernel density',
                                                                            layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                                                            disabled=True)
        self.hs['L[0][x][4][0][1][0][0][0]_ana_method'] = self.hs['L[0][x][4][0][1][0][0]_ana_config'][0, 0:50]
        
        for ii in range(2):
            for jj in range(4):
                self.hs['L[0][x][4][0][1][0][0]_ana_config'][ii+1, jj*50:(jj+1)*50] = \
                    widgets.Dropdown(description = 'p{0}'.format(str(ii*4+jj).zfill(2)),
                                    value = 'linear',
                                    options = ['linear'],
                                    description_tooltip = "analysis function variable 0",
                                    layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                    disabled=True)
                self.hs['L[0][x][3][0][4][0][0][{0}]_ana_config_p{1}'.format(1+ii*4+jj, ii*4+jj)] = \
                    self.hs['L[0][x][4][0][1][0][0]_ana_config'][ii+1, jj*50:(jj+1)*50]
        
        for ii in range(2, 4):
            for jj in range(4):
                self.hs['L[0][x][4][0][1][0][0]_ana_config'][ii+1, jj*50:(jj+1)*50] = \
                    widgets.BoundedFloatText(description = 'p{0}'.format(str(ii*4+jj).zfill(2)),
                                            value = 0,
                                            min = -1e5,
                                            max = 1e5,
                                            description_tooltip = "analysis function variable 0",
                                            layout = {'width':f'{0.3*self.form_sz[1]-98}px'},
                                            disabled=True)
                self.hs['L[0][x][4][0][1][0][0][{0}]_ana_config_p{1}'.format(1+ii*4+jj, ii*4+jj)] = \
                    self.hs['L[0][x][4][0][1][0][0]_ana_config'][ii+1, jj*50:(jj+1)*50]
        
        self.hs['L[0][x][4][0][1][0][0]_ana_config'][6, 10:190] = \
            widgets.IntProgress(value=0,
                                min=0,
                                max=10,
                                step=1,
                                description='Completing:',
                                bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                orientation='horizontal')
        self.hs['L[0][x][4][0][1][0][0][17]_ana_config_progr_bar'] = \
            self.hs['L[0][x][4][0][1][0][0]_ana_config'][6, 10:190]
            
        self.hs['L[0][x][4][0][1][0][0]_ana_config'][7, 90:110] = \
            widgets.Button(description='Compute',
                           description_tip='Perform the analysis')
        self.hs['L[0][x][4][0][1][0][0][17]_ana_config_compute_btn'] = \
            self.hs['L[0][x][4][0][1][0][0]_ana_config'][7, 90:110]
        
        self.hs['L[0][x][4][0][1][0]_ana_config_box'].children = get_handles(self.hs, 'L[0][x][4][0][1][0]_ana_config_box', -1)
        ## ## ## ## ## ## config parameters -- end
        
        self.hs['L[0][x][4][0][1]_ana_config'].children = get_handles(self.hs, 'L[0][x][4][0][1]_ana_config', -1)
        ## ## ## ## ## config analysis -- end
        
        ## ## ## ## ##  result display -- start
        
        ## ## ## ## ##  result display -- end
        
        
        
        