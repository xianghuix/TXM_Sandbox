#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:19:17 2020

@author: xiao
"""
from importlib import reload
from ipywidgets import widgets
from IPython.display import display

import os, json, imp
import tomo_recon_gui as trg
trg = reload(trg)
import xanes2D_regtools_gui as x2drg
x2drg = reload(x2drg)
import xanes3D_regtools_gui as x3drg
x3drg = reload(x3drg)
from gui_components import (SelectFilesButton, NumpyArrayEncoder, 
                            get_handles, enable_disable_boxes, 
                            check_file_availability, get_raw_img_info,
                            gen_external_py_script, fiji_viewer_state, 
                            fiji_viewer_on, fiji_viewer_off,
                            read_config_from_reg_file, restart,
                            determine_element, determine_fitting_energy_range)
import napari
import imagej

# try:
ij = imagej.init('/home/xiao/software/Fiji.app', headless=False)
ijui = ij.ui()
ijui.showUI()

from jnius import autoclass
WindowManager = autoclass('ij.WindowManager')
ImagePlusClass = autoclass('ij.ImagePlus')
# except:
#     print('fiji is already up running!')
ij.py.run_macro("""run("Brightness/Contrast...");""")
napari.gui_qt()

try:
    tem = os.path.dirname(os.path.abspath(imp.find_module('xanes_analysis_gui')[1]))
    with open(os.path.join(tem, 'analysis_tool_gui_cfg.json')) as f:
        CFG = json.load(f)
        CWD = CFG['cwd']
except:
    CWD = os.path.abspath(os.path.curdir)

class xanes_regtools_gui():
    def __init__(self, form_sz=[650, 740]):
        self.ij = ij
        self.ijui = ijui
        self.WindowManager = WindowManager
        self.ImagePlusClass = ImagePlusClass
        self.hs = {}
        self.form_sz = form_sz
        self.cwd = CWD
        tem = os.path.dirname(os.path.abspath(imp.find_module('xanes_analysis_gui')[1]))
        self.GUI_cfg_file = os.path.join(tem, 'analysis_tool_gui_cfg.json')

        self.xanes2D_external_command_name = os.path.join(os.path.abspath(os.path.curdir), 'xanes2D_external_command.py')
        self.xanes3D_external_command_name = os.path.join(os.path.abspath(os.path.curdir), 'xanes3D_external_command.py')

        self.xanes3D_fiji_windows = {'xanes3D_virtural_stack_preview_viewer':{'ip':None,
                                                                      'fiji_id':None},
                                     'xanes3D_mask_viewer':{'ip':None,
                                                    'fiji_id':None},
                                     'xanes3D_review_viewer':{'ip':None,
                                                    'fiji_id':None},
                                     'xanes3D_review_manual_viewer':{'ip':None,
                                                    'fiji_id':None},
                                     'xanes3D_analysis_viewer':{'ip':None,
                                                        'fiji_id':None},
                                     'analysis_viewer_z_plot_viewer':{'ip':None,
                                                                      'fiji_id':None}}
        self.xanes2D_fiji_windows = {'xanes2D_raw_img_viewer':{'ip':None,
                                                    'fiji_id':None},
                                     'xanes2D_mask_viewer':{'ip':None,
                                                    'fiji_id':None},
                                     'xanes2D_review_viewer':{'ip':None,
                                                        'fiji_id':None},
                                     'xanes2D_analysis_viewer':{'ip':None,
                                                        'fiji_id':None},
                                     'analysis_viewer_z_plot_viewer':{'ip':None,
                                                                      'fiji_id':None}}
        self.tomo_fiji_windows = {'tomo_raw_img_viewer':{'ip':None,
                                                         'fiji_id':None},
                                  'tomo_0&180_viewer':{'ip':None,
                                                       'fiji_id':None},
                                  'tomo_cen_review_viewer':{'ip':None,
                                                            'fiji_id':None},
                                  'tomo_recon_viewer':{'ip':None,
                                                       'fiji_id':None}}

    def gui_layout(self):
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
        #                                                     Global Form                                               #
        #                                                                                                               #
        #################################################################################################################
        # define top tab form
        layout = {'border':'5px solid #00FF00', 'width':f'{self.form_sz[1]}px', 'height':f'{self.form_sz[0]}px'}
        self.hs['L[0]_top_tab_form'] = widgets.Tab()
        self.hs['L[0]_top_tab_form'].layout = layout

        ## ## define, organize, and name sub-tabs
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-46}px', 'height':f'{self.form_sz[0]-70}px'}
        self.hs['L[0][0]_tomo_recon_tabs'] = widgets.Tab()
        self.hs['L[0][1]_2D_xanes_tabs'] =  widgets.Tab()
        self.hs['L[0][2]_3D_xanes_tabs'] =  widgets.Tab()
        self.hs['L[0][0]_tomo_recon_tabs'].layout = layout
        self.hs['L[0][1]_2D_xanes_tabs'].layout = layout
        self.hs['L[0][2]_3D_xanes_tabs'].layout = layout

        self.hs['L[0]_top_tab_form'].children = get_handles(self.hs, 'L[0]_top_tab_form', -1)
        self.hs['L[0]_top_tab_form'].set_title(0, 'TOMO RECON')
        self.hs['L[0]_top_tab_form'].set_title(1, '2D XANES')
        self.hs['L[0]_top_tab_form'].set_title(2, '3D XANES')

        #################################################################################################################
        #                                                                                                               #
        #                                                     3D XANES                                                  #
        #                                                                                                               #
        #################################################################################################################
        
        self.xanes3D_gui = x3drg.xanes3D_regtools_gui(self, form_sz=self.form_sz)
        self.xanes3D_gui.build_gui()
        self.hs['L[0][2]_3D_xanes_tabs'].children = [self.xanes3D_gui.hs['L[0][2][0]_config_input_form'], 
                                                     self.xanes3D_gui.hs['L[0][2][1]_reg_setting_form'],
                                                     self.xanes3D_gui.hs['L[0][2][2]_reg&review_form'],
                                                     self.xanes3D_gui.hs['L[0][2][3]_analysis&display_form']]
        self.hs['L[0][2]_3D_xanes_tabs'].set_title(0, 'File Configurations')
        self.hs['L[0][2]_3D_xanes_tabs'].set_title(1, 'Registration Settings')
        self.hs['L[0][2]_3D_xanes_tabs'].set_title(2, 'Registration & Reviews')
        self.hs['L[0][2]_3D_xanes_tabs'].set_title(3, 'Analysis & Display')

        #################################################################################################################
        #                                                                                                               #
        #                                                     2D XANES                                                  #
        #                                                                                                               #
        #################################################################################################################
        self.xanes2D_gui = x2drg.xanes2D_regtools_gui(self, form_sz=self.form_sz)
        self.xanes2D_gui.build_gui()

        self.hs['L[0][1]_2D_xanes_tabs'].children = [self.xanes2D_gui.hs['L[0][1][0]_config_input_form'],
                                                     self.xanes2D_gui.hs['L[0][1][1]_reg_setting_form'], 
                                                     self.xanes2D_gui.hs['L[0][1][2]_reg&review_form'], 
                                                     self.xanes2D_gui.hs['L[0][1][3]_analysis&display_form']]
        self.hs['L[0][1]_2D_xanes_tabs'].set_title(0, 'File Configurations')
        self.hs['L[0][1]_2D_xanes_tabs'].set_title(1, 'Registration Settings')
        self.hs['L[0][1]_2D_xanes_tabs'].set_title(2, 'Registration & Reviews')
        self.hs['L[0][1]_2D_xanes_tabs'].set_title(3, 'Analysis & Display')

        #################################################################################################################
        #                                                                                                               #
        #                                                    TOMO RECON                                                 #
        #                                                                                                               #
        #################################################################################################################
        self.tomo_recon_gui = trg.tomo_recon_gui(self, form_sz=self.form_sz)
        self.tomo_recon_gui.build_gui()

        self.hs['L[0][0]_tomo_recon_tabs'].children = [self.tomo_recon_gui.hs['L[0][0][0]_config_input_form'],
                                                       self.tomo_recon_gui.hs['L[0][0][1]_filter&recon_form']]
        self.hs['L[0][0]_tomo_recon_tabs'].set_title(0, 'File Configuration')
        self.hs['L[0][0]_tomo_recon_tabs'].set_title(1, 'Recon')

        display(self.hs['L[0]_top_tab_form'])


    
