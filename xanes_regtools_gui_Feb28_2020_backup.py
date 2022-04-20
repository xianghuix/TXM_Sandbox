#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:19:17 2020

@author: xiao
"""

from ipywidgets import widgets
from ipyfilechooser import FileChooser
from IPython.display import display
import os, functools, glob, tifffile, h5py, json
import numpy as np
import skimage.morphology as skm
import xanes_regtools as xr

import imagej

#try:
ij = imagej.init('/home/xiao/software/Fiji.app', headless=False)
ijui = ij.ui()
ijui.showUI()

from jnius import autoclass
WindowManager = autoclass('ij.WindowManager')
ImagePlusClass = autoclass('ij.ImagePlus')
#except:
#    print('fiji is already up running!')
    
ij.py.run_macro("""run("Brightness/Contrast...");""")


class xanes_regtools_gui():
    def __init__(self, reg_params):
        self.hs = {}
        self.fc_2D_params_configured = False
        self.scan_id_2D_params_configured = False
        self.reg_2D_params_configured = False
        self.roi_2D_params_configured = False
        self.read_alignment_option = False
        self.aligned_2D = False
        self.reg_3D_done = False
        self.raw_viewer_ip = None
        self.mask_viewer_ip = None
        
        self.fc_3D_params_configured = False
        self.scan_id_3D_params_configured = False
        self.reg_params_3D_params_configured = False
        self.roi_3D_params_configured = False
        self.aligned_3D = False
        
        self.img = None
        self.raw_3D_h5_top_dir = None
        self.recon_3D_top_dir = None
        self.trial_reg_3D_save_file = None
        self.alignment_file = None
        reg_params['3D_roi'] = None
        reg_params['3D_scan_id_s'] = None 
        reg_params['3D_scan_id_e'] = None
        reg_params['3D_chunk_sz'] = None
        reg_params['3D_fixed_scan_id'] = None
        reg_params['3D_fixed_sli_id'] = None
        reg_params['3D_moving_sli_search_half_range'] = None
        reg_params['3D_use_mask'] = None
        reg_params['3D_mask_thres'] = None
        reg_params['3D_mask_dilation'] = None
        reg_params['3D_use_anchor'] = None
        reg_params['3D_reg_method'] = None
        reg_params['3D_ref_mode'] = None
        reg_params['3D_trial_reg'] = None
        
        
    
    def gui_layout(self, reg_params):
        self.hs['form_global'] = widgets.Box()
        self.hs['form_global'].layout = {'border':'5px solid #00FF00', 'width':'750px', 'height':'650px'}
        
        self.hs['Top_tabs'] = widgets.Tab()
        self.hs['3D_tabs'] = widgets.Tab()
        self.hs['2D_tabs'] = widgets.Tab()
        self.hs['Top_tabs'].children = [self.hs['2D_tabs'],
                                        self.hs['3D_tabs']]
        self.hs['Top_tabs'].set_title(0, '2D XANES')
        self.hs['Top_tabs'].set_title(1, '3D XANES')
        
        ##### 3D tab0: File configuration -- START
        self.hs['3D_tab0_form'] = widgets.VBox(display='flex')
        self.hs['3D_tab0_form'].layout = {'border':'3px solid #FFCC00', 'width':'655px', 'height': '570px'}
                                  
        tab0_3D_file_setup_box = widgets.VBox(flex_flow='column')
        tab0_3D_file_setup_box.layout = {'border':'3px solid #FFCC00', 'width':'650px', 'height': '350px'}
        label1 = widgets.HTML('<span style="background-color:rgb(135,206,250);">' + 'Choose 3D_raw_h5_top_dir' + '</span>')
        label2 = widgets.HTML('<span style="background-color:rgb(135,206,250);">' + 'Choose 3D_recon_top_dir' + '</span>')
        label3 = widgets.HTML('<span style="background-color:rgb(135,206,250);">' + '3D_trial_reg_save_file' + '</span>')
        self.hs["3D_tab0_fc_raw_dir"] = FileChooser(os.path.curdir, display='flex')
#        self.hs["3D_tab0_fc_raw_dir"].selected_path = None
        self.hs["3D_tab0_fc_raw_dir"].default_path = '/NSLS2/xf18id1/users/2020Q1/YUAN_YANG_Proposal_305811/'
        self.hs["3D_tab0_fc_recon_dir"] = FileChooser(os.path.curdir, description_tip='3D_recon_top_dir', display='flex')
        self.hs["3D_tab0_fc_recon_dir"].default_path = '/NSLS2/xf18id1/users/2020Q1/YUAN_YANG_Proposal_305811/'
        self.hs["3D_tab0_fc_trial_reg_file"] = FileChooser(os.path.curdir, description_tip='trial_reg_file...', display='flex') 
        self.hs["3D_tab0_fc_trial_reg_file"].default_path = '/NSLS2/xf18id1/users/2020Q1/YUAN_YANG_Proposal_305811/'
        self.hs["3D_tab0_fc_trial_reg_file"].default_filename = '3D_trial_reg.h5'
        
        box_button_confirm_option = widgets.VBox()
        self.hs['3D_tab0_fc_option_checkbox'] = widgets.Checkbox(description='Registation is completed',
                                                                 description_tooltip='Check this option if you have already done registration, and only need review the results and align the 3D datasets',
                                                                 value=False)
        box_button_confirm = widgets.HBox()
        self.hs["3D_tab0_fc_confirm_button"] = widgets.Button(description='Confirm', description_tooltip='Confirm File Configurations')        
        self.hs['3D_tab0_fc_confirm_button_state'] = widgets.Text(value='Please set directories ...', description='Status', disabled=True)
        self.hs['3D_tab0_fc_confirm_button_state'].layout = {'flex_flow':'left', 'width':'505px'}
        self.hs["3D_tab0_fc_confirm_button"].on_click(functools.partial(self.tab0_3D_fc_confirm_button, rp_=reg_params))
        box_button_confirm.children = [self.hs["3D_tab0_fc_confirm_button"],
                                       self.hs['3D_tab0_fc_confirm_button_state']]
        box_button_confirm.layout = {'align_items':'center'} 
        box_button_confirm_option.children = [self.hs['3D_tab0_fc_option_checkbox'],
                                              box_button_confirm]
        tab0_3D_file_setup_box.children = [label1, self.hs["3D_tab0_fc_raw_dir"],
                                   label2, self.hs["3D_tab0_fc_recon_dir"],
                                   label3, self.hs["3D_tab0_fc_trial_reg_file"],
                                   box_button_confirm_option]
        
        tab0_3D_scan_id_setup_box = widgets.HBox()
        self.hs["3D_tab0_scan_id_s"] = widgets.IntText(description='scan_id start',
                                                          display='flex', disabled=True)
        self.hs["3D_tab0_scan_id_s"].layout = {'width':'200px'}
        self.hs["3D_tab0_scan_id_e"] = widgets.IntText(description='scan_id end',
                                                       description_tooltip='you need to press Enter to enable 3D_fixed_scan_id slider below',
                                                          display='flex', disabled=True)
        self.hs["3D_tab0_scan_id_e"].layout = {'width':'200px'}                
        self.hs["3D_tab0_fixed_scan_id"] = widgets.IntSlider(description='3D_fixed_scan_id', disabled=True)
        self.hs["3D_tab0_fixed_sli_id"] = widgets.IntSlider(description='fixed_sli_id', disabled=True)
        tab0_3D_fiji_box = widgets.HBox()
        self.hs["3D_tab0_fixed_sli_fiji_checkbox"] = widgets.Checkbox(value=False, description='Fiji', disabled=True)
        self.hs['3D_tab0_close_all_fiji_viewers_button'] = widgets.Button(description='close all fiji viewers',
                                                                   description_tooltip='close all fiji viewers associated with the current session on the desktop. Fiji will be still on.',
                                                                   disabled=True)
        tab0_3D_fiji_box.children = [self.hs["3D_tab0_fixed_sli_fiji_checkbox"],
                                     self.hs['3D_tab0_close_all_fiji_viewers_button']]
        
        def tab0_3D_fc_raw_dir_changed(self, a):
            self.hs["3D_tab0_fc_raw_dir"].default_path = os.path.abspath(self.hs["3D_tab0_fc_raw_dir"].selected_path)
            self.hs["3D_tab0_fc_recon_dir"].default_path = os.path.abspath(self.hs["3D_tab0_fc_raw_dir"].selected_path)
            self.hs["3D_tab0_fc_trial_reg_file"].default_path = os.path.abspath(self.hs["3D_tab0_fc_raw_dir"].selected_path)
            
        def tab0_3D_scan_id_s_change(a):  
            if os.path.exists(self.raw_3D_h5_temp.format(str(self.hs["3D_tab0_scan_id_s"].value))):
                if os.path.exists(self.recon_3D_top_dir.format(str(self.hs["3D_tab0_scan_id_s"].value))):
                    if self.hs["3D_tab0_scan_id_e"].value < self.hs["3D_tab0_scan_id_s"].value:
                        self.hs["3D_tab0_scan_id_e"].value = self.hs["3D_tab0_scan_id_s"].value
                    self.hs["3D_tab0_fixed_scan_id"].disabled = False
                    self.hs["3D_tab0_fixed_scan_id"].max = self.hs["3D_tab0_scan_id_e"].value
                    self.hs["3D_tab0_fixed_scan_id"].min = self.hs["3D_tab0_scan_id_s"].value                    
                    fn = sorted(glob.glob(os.path.join(self.recon_3D_top_dir.format(self.hs["3D_tab0_scan_id_s"].value), '*.tiff')))[0]
                    self.img = tifffile.imread(fn)
                    self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'scan ids are changed ...'
                else:
                    self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'specified scan_id starting number does not exist in recon top dir ...'
                    self.hs["3D_tab0_fixed_scan_id"].disabled = True
                    self.hs["3D_tab0_fixed_sli_id"].disabled = True
                    self.hs["3D_tab0_fixed_sli_fiji_checkbox"].disabled = True
#                    self.hs["3D_tab0_scan_id_confirm_button"].disabled = True
            else:
                self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'raw h5 file does not exist in the raw top dir ...'
                self.hs["3D_tab0_fixed_scan_id"].disabled = True
                self.hs["3D_tab0_fixed_sli_id"].disabled = True
                self.hs["3D_tab0_fixed_sli_fiji_checkbox"].disabled = True
            
        def tab0_3D_scan_id_e_change(a):
            if os.path.exists(self.raw_3D_h5_temp.format(str(self.hs["3D_tab0_scan_id_e"].value))):
                if os.path.exists(self.recon_3D_top_dir.format(str(self.hs["3D_tab0_scan_id_e"].value))):
                    if self.hs["3D_tab0_scan_id_e"].value < self.hs["3D_tab0_scan_id_s"].value:
                        self.hs["3D_tab0_scan_id_s"].value = self.hs["3D_tab0_scan_id_e"].value
                    self.hs["3D_tab0_fixed_scan_id"].disabled = False
                    self.hs["3D_tab0_fixed_scan_id"].max = self.hs["3D_tab0_scan_id_e"].value
                    self.hs["3D_tab0_fixed_scan_id"].min = self.hs["3D_tab0_scan_id_s"].value                    
                    self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'scan ids are changed ...'
                else:
                    self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'specified scan_id starting number does not exist in recon top dir ...'
                    self.hs["3D_tab0_fixed_scan_id"].disabled = True
                    self.hs["3D_tab0_fixed_sli_id"].disabled = True
                    self.hs["3D_tab0_fixed_sli_fiji_checkbox"].disabled = True
#                    self.hs["3D_tab0_scan_id_confirm_button"].disabled = True
            else:
                self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'raw h5 file does not exist in the raw top dir ...'
                self.hs["3D_tab0_fixed_scan_id"].disabled = True
                self.hs["3D_tab0_fixed_sli_id"].disabled = True

        def tab0_3D_fixed_scan_id_slider_change(a):
            if os.path.exists(self.recon_3D_top_dir.format(str(self.hs["3D_tab0_fixed_scan_id"].value))):
                self.hs["3D_tab0_fixed_sli_id"].disabled = False
                self.hs["3D_tab0_fixed_sli_fiji_checkbox"].disabled = False
                self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'fixed scan id is changed ...'
                file_list = sorted(glob.glob(os.path.join(self.recon_3D_top_dir.format(self.hs["3D_tab0_fixed_scan_id"].value), '*.tiff')))
                self.hs["3D_tab0_fixed_sli_id"].max = int(file_list[-1].split('.')[0].split('_')[-1])
                self.hs["3D_tab0_fixed_sli_id"].min = int(file_list[0].split('.')[0].split('_')[-1])                
#                self.hs['3D_tab0_scan_id_confirm_button_state'].value = '{}'.format(str(self.hs["3D_tab0_fixed_scan_id"].value))
            else:
                self.hs["3D_tab0_fixed_scan_id"].disabled = True
                self.hs["3D_tab0_fixed_sli_id"].disabled = True
                self.hs["3D_tab0_fixed_sli_fiji_checkbox"].disabled = True
                self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'recon with fixed scan id {} is changed ...'.format(str(self.hs["3D_tab0_fixed_scan_id"].value))
            
        def tab0_3D_fixed_sli_slider_change(a):
            if self.hs["3D_tab0_fixed_sli_fiji_checkbox"].value:
                self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'fixed slice id is changed ...'
                self.raw_viewer_ip.setSlice(self.hs["3D_tab0_fixed_sli_id"].value-self.hs["3D_tab0_fixed_sli_id"].min+1)
            self.fixed_sli_3D_id = self.hs["3D_tab0_fixed_sli_id"].value
            
        def tab0_3D_fixed_sli_fiji_checked(a):
            if self.hs["3D_tab0_fixed_sli_fiji_checkbox"].value:
                self.fn0 = self.recon_3D_tiff_temp.format(self.hs["3D_tab0_fixed_scan_id"].value,
                                             str(self.hs["3D_tab0_fixed_sli_id"].min).zfill(5))
                args = {'directory':self.fn0, 'start':1}
                ij.py.run_plugin(" Open VirtualStack", args)
                self.raw_viewer_ip = WindowManager.getCurrentImage()
                self.raw_viewer_ip.setSlice(self.fixed_sli_3D_id--self.hs["3D_tab0_fixed_sli_id"].min+1) 
                self.hs['3D_tab0_close_all_fiji_viewers_button'].disabled=False                
            else:
                self.raw_viewer_ip.close()
                self.raw_viewer_ip = None
                self.hs['3D_tab0_close_all_fiji_viewers_button'].disabled=True 
                self.hs['3D_tab1_roi_x_range_slider'].disabled = True
                self.hs['3D_tab1_roi_y_range_slider'].disabled = True
                self.hs['3D_tab1_roi_z_range_slider'].disabled = True
                self.hs['3D_tab1_set_roi_button'].disabled = True
                
        def tab0_3D_close_all_fiji_viewers(a):
            for ii in (WindowManager.getIDList()):
                WindowManager.getImage(ii).close()
            self.hs["3D_tab0_fixed_sli_fiji_checkbox"].value = False
            self.hs['3D_tab1_roi_x_range_slider'].disabled = True
            self.hs['3D_tab1_roi_y_range_slider'].disabled = True
            self.hs['3D_tab1_roi_z_range_slider'].disabled = True
            self.hs['3D_tab1_set_roi_button'].disabled = True
            
        def tab0_3D_fc_option_checkbox(a):
            if self.hs['3D_tab0_fc_option_checkbox'].value:
                self.reg_3D_done = True
            else:
                self.reg_3D_done = False

        self.hs["3D_tab0_fc_raw_dir"].observe(tab0_3D_fc_raw_dir_changed, names='selected')                
        self.hs["3D_tab0_scan_id_s"].observe(tab0_3D_scan_id_s_change, names='value')
        self.hs["3D_tab0_scan_id_e"].observe(tab0_3D_scan_id_e_change, names='value')      
        self.hs["3D_tab0_fixed_sli_fiji_checkbox"].observe(tab0_3D_fixed_sli_fiji_checked, names='value')
        self.hs["3D_tab0_fixed_sli_id"].observe(tab0_3D_fixed_sli_slider_change, names='value')
        self.hs["3D_tab0_fixed_scan_id"].observe(tab0_3D_fixed_scan_id_slider_change, names='value')
        self.hs['3D_tab0_close_all_fiji_viewers_button'].on_click(tab0_3D_close_all_fiji_viewers)
        self.hs['3D_tab0_fc_option_checkbox'].observe(tab0_3D_fc_option_checkbox, names='value')
        
        box_button_confirm2 = widgets.HBox(display='flex')
        self.hs["3D_tab0_scan_id_confirm_button"] = widgets.Button(description='Confirm',
                                                                   description_tooltip='Confirm scan_id Configurations',
                                                                   disabled=True)        
        self.hs['3D_tab0_scan_id_confirm_button_state'] = widgets.Text(value='Please set scan ids ...', description='Status', disabled=True)
        self.hs['3D_tab0_scan_id_confirm_button_state'].layout = {'flex_flow':'left', 'width':'505px'}
        box_button_confirm2.children = [self.hs["3D_tab0_scan_id_confirm_button"],
                                        self.hs['3D_tab0_scan_id_confirm_button_state']]
        
        self.hs["3D_tab0_scan_id_confirm_button"].on_click(functools.partial(self.tab0_3D_scan_id_confirm_button, rp_=reg_params))
        
        tab0_3D_scan_id_setup_box.children = [self.hs["3D_tab0_scan_id_s"],
                                       self.hs["3D_tab0_scan_id_e"]]
        self.hs['3D_tab0_form'].children=[tab0_3D_file_setup_box,
                                  tab0_3D_scan_id_setup_box,
                                  self.hs["3D_tab0_fixed_scan_id"],
                                  self.hs["3D_tab0_fixed_sli_id"],
                                  tab0_3D_fiji_box,
                                  box_button_confirm2]
        ##### 3D tab0: File configuration -- START -- END
        
        ##### 3D tab1: configure registration parameters --START
        self.hs['3D_tab1_form'] = widgets.VBox()
        self.hs['3D_tab1_form'].layout = {'border': '3px solid #FFCC00', 'width': '655px', 'height': '570px'}
        tab1_3D_roi_box = widgets.VBox()
        tab1_3D_roi_box.layout = {'border': '3px solid #FFCC00', 'width': '650px', 'height': '200px'}
        label4 = widgets.HTML('<span style="background-color:rgb(135,206,250);">' + 'define 3D_roi' + '</span>')
        label4.layout = {'left':'270px', 'width':'200px'}
        self.hs['3D_tab1_roi_x_range_slider'] = widgets.IntRangeSlider(value=[10, 40],
                                                        min=1,
                                                        max=50,
                                                        step=1,
                                                        description='x range:',
                                                        disabled=True,
                                                        continuous_update=False,
                                                        orientation='horizontal',
                                                        readout=True,
                                                        readout_format='d')
        self.hs['3D_tab1_roi_x_range_slider'].layout = {'width': '600px'}
        
        self.hs['3D_tab1_roi_y_range_slider'] = widgets.IntRangeSlider(value=[10, 40],
                                                        min=1,
                                                        max=50,
                                                        step=1,
                                                        description='y range:',
                                                        disabled=True,
                                                        continuous_update=False,
                                                        orientation='horizontal',
                                                        readout=True,
                                                        readout_format='d')
        self.hs['3D_tab1_roi_y_range_slider'].layout = {'width': '600px'}
        
        self.hs['3D_tab1_roi_z_range_slider'] = widgets.IntRangeSlider(value=[10, 40],
                                                        min=1,
                                                        max=50,
                                                        step=1,
                                                        description='z range:',
                                                        disabled=True,
                                                        continuous_update=False,
                                                        orientation='horizontal',
                                                        readout=True,
                                                        readout_format='d')
        self.hs['3D_tab1_roi_z_range_slider'].layout = {'width': '600px'}
        tab1_3D_set_roi_box = widgets.HBox()
        tab1_3D_set_roi_box.layout = {'top':'20px'}
        self.hs['3D_tab1_set_roi_button'] = widgets.Button(description='set roi', disabled=True)
        self.hs['3D_tab1_set_roi_button_state'] = widgets.Text(value='Please have your fiji stack viewer on ...', description='Status', disabled=True)
        self.hs['3D_tab1_set_roi_button_state'].layout = {'flex_flow':'left', 'width':'505px'}
        tab1_3D_set_roi_box.children = [self.hs['3D_tab1_set_roi_button'],
                                         self.hs['3D_tab1_set_roi_button_state']]
        tab1_3D_roi_box.children = [label4,
                                    self.hs['3D_tab1_roi_x_range_slider'],
                                    self.hs['3D_tab1_roi_y_range_slider'],
                                    self.hs['3D_tab1_roi_z_range_slider'],
                                    tab1_3D_set_roi_box]
        
        def tab1_3D_roi_x_range_slider_change(a):
            self.raw_viewer_ip.setRoi(self.hs['3D_tab1_roi_x_range_slider'].value[0],
                                  self.hs['3D_tab1_roi_y_range_slider'].value[0],
                                  self.hs['3D_tab1_roi_x_range_slider'].value[1]-self.hs['3D_tab1_roi_x_range_slider'].value[0],
                                  self.hs['3D_tab1_roi_y_range_slider'].value[1]-self.hs['3D_tab1_roi_y_range_slider'].value[0])
        
        def tab1_3D_roi_y_range_slider_change(a):
            self.raw_viewer_ip.setRoi(self.hs['3D_tab1_roi_x_range_slider'].value[0],
                                  self.hs['3D_tab1_roi_y_range_slider'].value[0],
                                  self.hs['3D_tab1_roi_x_range_slider'].value[1]-self.hs['3D_tab1_roi_x_range_slider'].value[0],
                                  self.hs['3D_tab1_roi_y_range_slider'].value[1]-self.hs['3D_tab1_roi_y_range_slider'].value[0])
        
        def tab1_3D_roi_z_range_slider_change(a):
#            self.hs["3D_tab0_fixed_sli_id"].value = self.hs['3D_tab1_roi_z_range_slider'].value[0]-self.hs['3D_tab1_roi_z_range_slider'].min+1
#            self.hs["3D_tab0_fixed_sli_id"].value = self.hs['3D_tab1_roi_z_range_slider'].value[0]
            self.raw_viewer_ip.setSlice(self.hs['3D_tab1_roi_z_range_slider'].value[0]-self.hs['3D_tab1_roi_z_range_slider'].min+1)
        
        

        self.hs['3D_tab1_roi_x_range_slider'].observe(tab1_3D_roi_x_range_slider_change)
        self.hs['3D_tab1_roi_y_range_slider'].observe(tab1_3D_roi_y_range_slider_change)
        self.hs['3D_tab1_roi_z_range_slider'].observe(tab1_3D_roi_z_range_slider_change)
        self.hs['3D_tab1_set_roi_button'].on_click(self.tab1_3D_set_roi_button)

        label5 = widgets.HTML('<span style="background-color:rgb(135,206,250);">' + 'configure registration parameters' + '</span>')
        label5.layout = {'left':'220px', 'width':'200px'}        
        tab1_3D_mask_box1 = widgets.HBox()
        self.hs['3D_use_mask'] = widgets.Checkbox(value=False,
                                                  description='use mask',
                                                  description_tooltip='Use Mask: only use mask region for registration',
                                                  disabled=True)
        self.hs['3D_use_anchor'] = widgets.Checkbox(value=False,
                                                  description='use anchor',
                                                  description_tooltip='Use Anchor: registration will be done relative to the anchored scan',
                                                  disabled=True)        
        tab1_3D_mask_box1.children = [self.hs['3D_use_mask'],
                                      self.hs['3D_use_anchor']]
        
        tab1_3D_mask_box2 = widgets.HBox()
        self.hs['3D_mask_thres'] = widgets.FloatSlider(value=0.0003,
                                                       description='mask thres',
                                                       description_tooltip='Mask Thres: set threshold for generating a mask',
                                                       min=-1,
                                                       max=1,
                                                       step=0.0002,
                                                       readout_format='.4f',
                                                       disabled=True)        
        self.hs['3D_mask_dilation_width'] = widgets.IntSlider(value=5,
                                                              description='mask dilation width',
                                                              description_tooltip='Mask Dilation Width: dilate mask large enough to conver a sample region',
                                                              min=0,
                                                              max=50,
                                                              disabled=True)
        tab1_3D_mask_box2.children = [self.hs['3D_mask_thres'],
                                     self.hs['3D_mask_dilation_width']]
        
        self.hs['3D_tab1_mask_preview_fiji_checkbox'] = widgets.Checkbox(value=False,
                                                                         description='preview mask in Fiji',
                                                                         disabled=True)
        self.hs['3D_tab1_mask_preview_fiji_checkbox'].layout = {'left': '170px'}
        
        tab1_3D_chunk_box = widgets.HBox()
        self.hs['moving_sli_search_half_range'] = widgets.IntSlider(value=20,
                                                                    description='slice search half range',
                                                                    description_tooltip='Slice Search Half Range: matching of corresponding slices between different scans will be in a range [-half_range, half_range] relative to the fixed_sli in fixed_scan.',
                                                                    disabled=True)
        self.hs['3D_chunk_sz'] = widgets.IntSlider(description='chunk size',
                                                   description_tooltip='Chunk Size: the size for registering images chunk by chunk',
                                                   value=7,
                                                   min=1,
                                                   max=(self.hs["3D_tab0_fixed_sli_id"].max - self.hs["3D_tab0_fixed_sli_id"].min),
                                                   disabled=True)
        tab1_3D_chunk_box.children = [self.hs['moving_sli_search_half_range'],
                                      self.hs['3D_chunk_sz']]

        tab1_3D_reg_method_box = widgets.HBox()
        self.hs['3D_reg_method'] = widgets.Dropdown(value='MPC',
                                                    options=['MPC', 'PC', 'SR'],
                                                    description='reg method',
                                                    description_tooltip='Reg Method: MPC: Masked Phase Correlation; PC: Phase Correlation; SR: StackReg',
                                                    disabled=True)
        self.hs['3D_ref_mode'] = widgets.Dropdown(value='single',
                                                  options=['single', 'neighbor', 'average'],
                                                  description='reference mode',
                                                  description_tooltip='Reference Mode: single: two images with fixed id gap in two consecutive chunks are used for the registration between two chunks; neighbor: two neighbor images in two consecutive chunks are used for the registration between two chunks; average: deprecated',
                                                  disabled=True)  
        tab1_3D_reg_method_box.children = [self.hs['3D_reg_method'],
                                           self.hs['3D_ref_mode']]

        tab1_3D_set_reg_config_box = widgets.HBox()        
        self.hs['3D_tab1_reg_config_button'] = widgets.Button(description='set reg config', disabled=True)
        self.hs['3D_tab1_reg_config_button_state'] = widgets.Text(value='Please configure your registration parameters ...',
                                                                  description='Status', disabled=True)
        self.hs['3D_tab1_reg_config_button_state'].layout = {'flex_flow':'left', 'width':'505px'}
        tab1_3D_set_reg_config_box.children = [self.hs['3D_tab1_reg_config_button'],
                                               self.hs['3D_tab1_reg_config_button_state']]
        
        def tab1_3D_use_mask_checkbox(a):
            if self.hs['3D_use_mask'].value:
                self.mask = np.ndarray([self.roi[1]-self.roi[0],
                                       self.roi[3]-self.roi[2]])
                self.hs['3D_mask_thres'].disabled = False
                self.hs['3D_mask_dilation_width'].disabled = False
                self.hs['3D_reg_method'].options = ['MPC', 'PC', 'SR']
            else:
                self.hs['3D_reg_method'].options = ['PC', 'SR']
                self.hs['3D_mask_thres'].disabled = True
                self.hs['3D_mask_dilation_width'].disabled = True
                
        def tab1_3D_mask_fiji_preview_checked(a):
            if self.hs['3D_tab1_mask_preview_fiji_checkbox'].value:  
                self.img = np.ndarray([self.scan_3D_id_e-self.scan_3D_id_s+1,
                                       self.roi[1]-self.roi[0],
                                       self.roi[3]-self.roi[2]])
                cnt = 0
                for ii in range(self.scan_3D_id_s, self.scan_3D_id_e+1):
                    fn = self.recon_3D_tiff_temp.format(ii, str(self.fixed_sli_3D_id).zfill(5))
                    self.img[cnt, ...] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                    cnt += 1
                    
                ijui.show(ij.py.to_java(self.img))
                self.mask_viewer_ip = WindowManager.getCurrentImage()
                self.mask_viewer_ip.setTitle('mask preview')
            else:
                self.mask_viewer_ip.close()
                self.mask_viewer_ip = None
        
        def tab1_3D_mask_dialation_slider(a):
            self.mask[:] = skm.binary_dilation((self.img[self.fixed_scan_3D_id-self.scan_3D_id_s]>self.hs['3D_mask_thres'].value).astype(np.uint8),
                                        np.ones([self.hs['3D_mask_dilation_width'].value,
                                                 self.hs['3D_mask_dilation_width'].value])).astype(np.uint8)[:]
            self.mask_viewer_ip.setImage(ij.convert().convert(ij.dataset().create(ij.py.to_java(self.img*self.mask)), ImagePlusClass))
            ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
            
        ### below is the tested code in jupyter -- start
        # ## the above implementation is based on this piece of tested code
        # ijui.show(ij.py.to_java(gui.img))
        # ip = WindowManager.getCurrentImage()
        #            
        # imp = ij.py.to_java(gui.img)
        # imp = ij.dataset().create(imp)
        # imp = ij.convert().convert(imp, ImagePlus)
        # ip.setImage(imp)
        # ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        # ## below is the tested code in jupyter -- end
        
        def tab1_3D_mask_threshold_slider(a):
            if self.mask_viewer_ip is None:
                self.hs['3D_tab1_reg_config_button_state'].value = 'Please enable "Fiji Preview" first ...'
                self.hs['3D_tab1_mask_preview_fiji_checkbox'].value = True
            self.mask[:] = skm.binary_dilation((self.img[self.fixed_scan_3D_id-self.scan_3D_id_s]>self.hs['3D_mask_thres'].value).astype(np.uint8),
                                        np.ones([self.hs['3D_mask_dilation_width'].value,
                                                 self.hs['3D_mask_dilation_width'].value])).astype(np.uint8)[:]
            self.mask_viewer_ip.setImage(ij.convert().convert(ij.dataset().create(ij.py.to_java(self.img*self.mask)), ImagePlusClass))
            ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")

        self.hs['3D_tab1_mask_preview_fiji_checkbox'].observe(tab1_3D_mask_fiji_preview_checked, names='value')
        self.hs['3D_mask_dilation_width'].observe(tab1_3D_mask_dialation_slider, names='value') 
        self.hs['3D_mask_thres'].observe(tab1_3D_mask_threshold_slider, names='value') 
        self.hs['3D_use_mask'].observe(tab1_3D_use_mask_checkbox, names='value') 
        self.hs['3D_tab1_reg_config_button'].on_click(self.tab1_3D_set_reg_config_button)
        
        self.hs['3D_tab1_form'].children = [tab1_3D_roi_box,
                                            label5,
                                            tab1_3D_mask_box1,
                                            tab1_3D_mask_box2,
                                            self.hs['3D_tab1_mask_preview_fiji_checkbox'],
                                            tab1_3D_chunk_box,
                                            tab1_3D_reg_method_box,
                                            tab1_3D_set_reg_config_box]
        ##### 3D tab1: configure registration parameters -- END
        
        ##### 3D tab2: registration and result validation -- START
        self.hs['3D_tab2_form'] = widgets.VBox()
        self.hs['3D_tab2_form'].layout = {'border': '3px solid #FFCC00', 'width': '655px', 'height': '570px'}
               
        label6 = widgets.HTML('<span style="background-color:rgb(135,206,250);">' + 'run registrations' + '</span>')
        label6.layout = {'left':'270px', 'width':'200px'}
        tab2_3D_run_registration_box = widgets.VBox()
        tab2_3D_run_registration_box.layout = {'border':'3px solid #FFCC00', 'width':'650px', 'height': '120px'}
        tab2_3D_run_registration_subbox = widgets.HBox()
        self.hs['3D_tab2_run_registration_button'] = widgets.Button(description='start registration',
                                                                    description_tooltip='start registration with given configuration',
                                                                    disabled=True)
        self.hs['3D_tab2_run_registration_state'] = widgets.Text(description='reg state',
                                                                 disabled = True,
                                                                 value='please configure registration first ...')
        self.hs['3D_tab2_run_registration_state'].layout = {'flex_flow':'left', 'width':'505px'}
        tab2_3D_run_registration_subbox.children = [self.hs['3D_tab2_run_registration_button'],
                                                    self.hs['3D_tab2_run_registration_state']]
        self.hs['3D_tab2_run_registration_progress'] = widgets.IntProgress(value=0,
                                                                           min=0,
                                                                           max=10,
                                                                           step=1,
                                                                           description='Completing:',
                                                                           bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                                                           orientation='horizontal')
        self.hs['3D_tab2_run_registration_progress'].layout = {'width':'640px'}
        tab2_3D_run_registration_box.children = [label6,
                                                 tab2_3D_run_registration_subbox,
                                                 self.hs['3D_tab2_run_registration_progress']]
        self.hs['3D_tab2_run_registration_button'].on_click(self.tab2_3D_run_registration_button)

        label7 = widgets.HTML('<span style="background-color:rgb(135,206,250);">' + 'review registration results' + '</span>')
        label7.layout = {'left':'245px', 'width':'200px'}
        tab2_3D_review_registration_box = widgets.VBox()
        tab2_3D_review_registration_box.layout = {'border':'3px solid #FFCC00', 'width':'650px', 'height': '210px'}  
        tab2_3D_review_registration_subbox0 = widgets.HBox()
        self.hs['3D_tab2_read_user_specified_reg_results_checkbox'] = widgets.Checkbox(description='read alignment',
                                                                                       description_tooltip='Check this option if user has already identify the best match for each alignment pair',
                                                                                       value=False,
                                                                                       disabled=True)
        self.hs["3D_tab2_read_alignment_dir"] = FileChooser(os.path.curdir, description_tip='alignment file', display='flex', disabled=True)
        self.hs["3D_tab2_read_alignment_dir"].default_path = '/media'
        tab2_3D_review_registration_subbox0.children = [self.hs['3D_tab2_read_user_specified_reg_results_checkbox'],
                                                        self.hs["3D_tab2_read_alignment_dir"]]
        self.hs['3D_tab2_select_reg_pair_slider'] = widgets.IntSlider(description='reg pair #',
                                                                      description_tooltip='select one registration pair and find the best match',
                                                                      min=0,
                                                                      max=10,
                                                                      disabled=True)
        self.hs['3D_tab2_select_reg_pair_slider'].layout = {'width': '500px', 'left':'70px', 'align_items':'stretch'}
        tab2_3D_review_registration_subbox1 = widgets.HBox()        
        self.hs['3D_tab2_reg_sli_search_range_slider'] = widgets.IntSlider(description='z shift',
                                                                    description_tooltip='slide this bar, find the best match, and record the corresponding index.',
                                                                    min=1,
                                                                    max=10,
                                                                    disabled=True)
        self.hs['3D_tab2_reg_sli_search_range_slider'].layout = {'width': '400px'}
        self.hs['3D_tab2_reg_sli_best_match'] = widgets.IntText(description='best match idx',
                                                                description_tooltip='type the index of the best match here. If there is not best match, type "-1" here.',
                                                                disabled=True)
        self.hs['3D_tab2_reg_sli_best_match'].layout = {'width': '200px'}    
        tab2_3D_review_registration_subbox1.children = [self.hs['3D_tab2_reg_sli_search_range_slider'],
                                                       self.hs['3D_tab2_reg_sli_best_match']]
        self.hs['3D_tab2_align_sli_best_match_record_button'] = widgets.Button(description='record',
                                                                               description_tooltip='record the best match index for the given alignment pair',
                                                                               disabled=True)
        self.hs['3D_tab2_align_sli_best_match_record_button'].layout = {'left':'240px'}
        tab2_3D_review_registration_subbox2 = widgets.HBox() 
        self.hs['3D_tab2_reg_review_finish_button'] = widgets.Button(description='Finish Review',
                                                                     description_tooltip='press this button to finish reviewing the results, and move to the next step',
                                                                     disabled=True)
        self.hs['3D_tab2_reg_review_state'] = widgets.Text(description='review status',
                                                           disabled=True,
                                                           value='please run the registration first ...')
        self.hs['3D_tab2_reg_review_state'].layout = {'width':'400px'}
        tab2_3D_review_registration_subbox2.children = [self.hs['3D_tab2_reg_review_finish_button'],
                                                        self.hs['3D_tab2_reg_review_state']]
        tab2_3D_review_registration_box.children = [label7,
                                                    tab2_3D_review_registration_subbox0,
                                                    self.hs['3D_tab2_select_reg_pair_slider'],
                                                    tab2_3D_review_registration_subbox1,
                                                    self.hs['3D_tab2_align_sli_best_match_record_button'],
                                                    tab2_3D_review_registration_subbox2] 
        
        def tab2_3D_read_user_specified_reg_results_checkbox_change(a):
            if self.hs['3D_tab2_read_user_specified_reg_results_checkbox'].value:
                self.hs['3D_tab2_run_registration_progress'].disabled = True
                self.hs['3D_tab2_select_reg_pair_slider'].disabled = True
                self.hs['3D_tab2_reg_sli_search_range_slider'].disabled = True
                self.hs['3D_tab2_reg_sli_best_match'].disabled = True
                self.hs['3D_tab2_align_sli_best_match_record_button'].disabled = True
                self.hs["3D_tab2_read_alignment_dir"].disabled = False
                self.read_alignment_option = True
            else:
                self.hs['3D_tab2_run_registration_progress'].disabled = False
                self.hs['3D_tab2_select_reg_pair_slider'].disabled = False
                self.hs['3D_tab2_reg_sli_search_range_slider'].disabled = False
                self.hs['3D_tab2_reg_sli_best_match'].disabled = False
                self.hs['3D_tab2_align_sli_best_match_record_button'].disabled = False
                self.hs["3D_tab2_read_alignment_dir"].disabled = True
                self.read_alignment_option = False
                
        def tab2_3D_select_reg_pair_slider_change(a):
            self.alignment_pair_id = self.hs['3D_tab2_select_reg_pair_slider'].value
            fn = self.trial_reg_3D_save_file
            f = h5py.File(fn, 'r')
            self.trial_reg[:] = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format(str(self.alignment_pair_id).zfill(3))][:]
            f.close()
            
            if self.mask_viewer_ip is None:
                ijui.show(ij.py.to_java(self.trial_reg))
                self.mask_viewer_ip = WindowManager.getCurrentImage()
                ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
                self.mask_viewer_ip.setTitle(str(self.alignment_pair_id).zfill(3))
            else:
                self.mask_viewer_ip.setImage(ij.convert().convert(ij.dataset().create(ij.py.to_java(self.trial_reg)), ImagePlusClass))
                self.mask_viewer_ip.show()
                ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
                self.mask_viewer_ip.setTitle(str(self.alignment_pair_id).zfill(3))  
        def tab2_3D_reg_sli_search_range_slider_change(a):
            if self.mask_viewer_ip is None:
                self.hs['3D_tab2_reg_review_state'].value = 'Please slide "reg pair #" to open a viewer ...'
            else:
                self.mask_viewer_ip.setSlice(self.hs['3D_tab2_reg_sli_search_range_slider'].value)
        self.hs['3D_tab2_select_reg_pair_slider'].observe(tab2_3D_select_reg_pair_slider_change, names='value')
        self.hs['3D_tab2_reg_sli_search_range_slider'].observe(tab2_3D_reg_sli_search_range_slider_change, names='value')
        self.hs['3D_tab2_reg_review_finish_button'].on_click(self.tab2_3D_reg_review_finish_button)
        self.hs['3D_tab2_align_sli_best_match_record_button'].on_click(self.tab2_3D_reg_review_sli_best_match_record_button)
        self.hs['3D_tab2_read_user_specified_reg_results_checkbox'].observe(tab2_3D_read_user_specified_reg_results_checkbox_change, names='value')
        
        label8 = widgets.HTML('<span style="background-color:rgb(135,206,250);">' + 'Align 3D recons' + '</span>')
        label8.layout = {'left':'270px', 'width':'200px'}
        tab2_3D_align_recon_box = widgets.VBox()
        tab2_3D_align_recon_box.layout = {'border':'3px solid #FFCC00', 'width':'650px', 'height': '120px'}
        tab2_3D_align_recon_subbox1 = widgets.HBox()
        self.hs['3D_tab2_align_recon_button'] = widgets.Button(description='align recons',
                                                                    description_tooltip='align recon dataset according to user specified best trial results',
                                                                    disabled=True)
        self.hs['3D_tab2_align_recon_state'] = widgets.Text(description='reg state',
                                                                 disabled = True,
                                                                 value='please review the registration results, and confirm ...')
        self.hs['3D_tab2_align_recon_state'].layout = {'flex_flow':'left', 'width':'505px'}
        tab2_3D_align_recon_subbox1.children = [self.hs['3D_tab2_align_recon_button'],
                                                    self.hs['3D_tab2_align_recon_state']]
        self.hs['3D_tab2_align_recon_progress'] = widgets.IntProgress(value=0,
                                                                           min=0,
                                                                           max=10,
                                                                           step=1,
                                                                           description='Completing:',
                                                                           bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                                                           orientation='horizontal')
        self.hs['3D_tab2_align_recon_progress'].layout = {'width':'640px'}
        tab2_3D_align_recon_box.children = [label8,
                                            tab2_3D_align_recon_subbox1,
                                            self.hs['3D_tab2_align_recon_progress']]
        self.hs['3D_tab2_align_recon_button'].on_click(self.tab2_3D_align_recon_button)
                                       
                
        self.hs['3D_tab2_form'].children = [tab2_3D_run_registration_box,
                                            tab2_3D_review_registration_box,
                                            tab2_3D_align_recon_box]
        ##### 3D tab2: registration and result validation -- END
        
        self.hs['3D_tabs'].children = [self.hs['3D_tab0_form'],
                                       self.hs['3D_tab1_form'],
                                       self.hs['3D_tab2_form']]
        self.hs['form_global'].children=[self.hs['Top_tabs']]
        self.hs['3D_tabs'].set_title(0, 'File Configurations')
        self.hs['3D_tabs'].set_title(1, 'Registration Settings')
        self.hs['3D_tabs'].set_title(2, 'Registration & reviews')
        display(self.hs['form_global'])
        
    def tab0_3D_fc_confirm_button(self, a, rp_={}):
        if self.reg_3D_done:
            self.hs['3D_tab2_reg_review_finish_button'].disabled=False
            self.hs['3D_tab2_run_registration_progress'].disabled = False
            self.hs['3D_tab2_select_reg_pair_slider'].disabled = False
            self.hs['3D_tab2_reg_sli_search_range_slider'].disabled = False
            self.hs['3D_tab2_reg_sli_best_match'].disabled = False
            self.hs['3D_tab2_align_sli_best_match_record_button'].disabled = False
            self.hs['3D_tab2_read_user_specified_reg_results_checkbox'].disabled = False
            self.hs["3D_tab2_read_alignment_dir"].disabled = False
            
            self.alignment_best_match = {}
            
            self.hs['3D_tab0_fc_confirm_button_state'].value = 'Configuration is done!'
            self.raw_3D_h5_top_dir = os.path.abspath(self.hs["3D_tab0_fc_raw_dir"].selected_path)
            self.recon_3D_top_dir = os.path.abspath(self.hs["3D_tab0_fc_recon_dir"].selected_path)
            self.trial_reg_3D_save_file = os.path.abspath(self.hs["3D_tab0_fc_trial_reg_file"].selected)
            self.recon_3D_top_dir = os.path.join(self.recon_3D_top_dir,
                                              'recon_fly_scan_id_{0}')
            self.recon_3D_tiff_temp = os.path.join(self.recon_3D_top_dir,
                                                'recon_fly_scan_id_{0}_{1}.tiff')
            self.raw_3D_h5_temp = os.path.join(self.raw_3D_h5_top_dir,
                                            'fly_scan_id_{}.h5')
        
            f = h5py.File(self.trial_reg_3D_save_file, 'r')
            self.trial_reg = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format('0'.zfill(3))][:]
            self.hs['3D_tab2_select_reg_pair_slider'].max = f['/trial_registration/trial_reg_parameters/alignment_pairs'].shape[0]-1
            self.hs['3D_tab2_select_reg_pair_slider'].min = 0            
            self.hs['3D_tab2_reg_sli_search_range_slider'].max = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format('0'.zfill(3))].shape[0]
            self.hs['3D_tab2_reg_sli_search_range_slider'].min = 1
            f.close()            
        else:
            self.hs['3D_tab2_reg_review_finish_button'].disabled=True
            self.hs['3D_tab2_run_registration_progress'].disabled = True
            self.hs['3D_tab2_select_reg_pair_slider'].disabled = True
            self.hs['3D_tab2_reg_sli_search_range_slider'].disabled = True
            self.hs['3D_tab2_reg_sli_best_match'].disabled = True
            self.hs['3D_tab2_align_sli_best_match_record_button'].disabled = True
            self.hs['3D_tab2_read_user_specified_reg_results_checkbox'].disabled = True
            self.hs["3D_tab2_read_alignment_dir"].disabled = True
            if self.hs["3D_tab0_fc_raw_dir"].selected_path:
                if self.hs["3D_tab0_fc_recon_dir"].selected_path:
                    if self.hs["3D_tab0_fc_trial_reg_file"].selected_filename:
                        self.hs['3D_tab0_fc_confirm_button_state'].value = 'Configuration is done!'
                        self.raw_3D_h5_top_dir = os.path.abspath(self.hs["3D_tab0_fc_raw_dir"].selected_path)
                        self.recon_3D_top_dir = os.path.abspath(self.hs["3D_tab0_fc_recon_dir"].selected_path)
                        self.trial_reg_3D_save_file = os.path.abspath(self.hs["3D_tab0_fc_trial_reg_file"].selected)
                        self.recon_3D_top_dir = os.path.join(self.recon_3D_top_dir,
                                                             'recon_fly_scan_id_{0}')
                        self.recon_3D_tiff_temp = os.path.join(self.recon_3D_top_dir,
                                                            'recon_fly_scan_id_{0}_{1}.tiff')
                        self.raw_3D_h5_temp = os.path.join(self.raw_3D_h5_top_dir,
                                                           'fly_scan_id_{}.h5')
                        self.hs["3D_tab0_scan_id_s"].disabled = False
                        self.hs["3D_tab0_scan_id_e"].disabled = False
                        self.hs["3D_tab0_scan_id_confirm_button"].disabled = False
                        self.fc_3D_params_configured = True
                    else:
                        self.hs['3D_tab0_fc_confirm_button_state'].value = "trial_reg_file is not defined!"
                else:
                    self.hs['3D_tab0_fc_confirm_button_state'].value = "recon_dir is not defined!"
            else:
                self.hs['3D_tab0_fc_confirm_button_state'].value = "raw_dir is not defined!"
                
    def tab0_3D_scan_id_confirm_button(self, a, rp_={}):  
        if self.hs["3D_tab0_fixed_scan_id"].disabled: 
            self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'fixed scan id is not configured yet!'
        else:
            if self.hs["3D_tab0_fixed_sli_id"].disabled: 
                self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'fixed slice id is not configured yet!'
            else:
                self.scan_3D_id_s = self.hs["3D_tab0_scan_id_s"].value
                self.scan_3D_id_e = self.hs["3D_tab0_scan_id_e"].value
                self.fixed_scan_3D_id = self.hs["3D_tab0_fixed_scan_id"].value
                self.fixed_sli_3D_id = self.hs["3D_tab0_fixed_sli_id"].value
                self.hs['3D_tab1_roi_x_range_slider'].disabled = False
                self.hs['3D_tab1_roi_x_range_slider'].min = 0
                self.hs['3D_tab1_roi_x_range_slider'].max = self.raw_viewer_ip.width
                self.hs['3D_tab1_roi_y_range_slider'].disabled = False
                self.hs['3D_tab1_roi_y_range_slider'].min = 0
                self.hs['3D_tab1_roi_y_range_slider'].max = self.raw_viewer_ip.height
                self.hs['3D_tab1_roi_z_range_slider'].disabled = False
                self.hs['3D_tab1_roi_z_range_slider'].max = self.hs["3D_tab0_fixed_sli_id"].max
                self.hs['3D_tab1_roi_z_range_slider'].min = self.hs["3D_tab0_fixed_sli_id"].min
                self.hs['3D_tab1_roi_z_range_slider'].value = [self.hs["3D_tab0_fixed_sli_id"].value, self.hs["3D_tab0_fixed_sli_id"].max]
                self.hs['3D_tab1_set_roi_button'].disabled = False
                self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'scan ids are configured!'
                self.scan_id_3D_params_configured = True    
                        
    def tab1_3D_set_roi_button(self, a, rp_={}):
        self.roi = [self.hs['3D_tab1_roi_y_range_slider'].value[0],
                    self.hs['3D_tab1_roi_y_range_slider'].value[1],
                    self.hs['3D_tab1_roi_x_range_slider'].value[0],
                    self.hs['3D_tab1_roi_x_range_slider'].value[1],
                    self.hs['3D_tab1_roi_z_range_slider'].value[0],
                    self.hs['3D_tab1_roi_z_range_slider'].value[1]]
        self.alignment_sli_start = self.hs['3D_tab1_roi_z_range_slider'].value[0]
        self.alignment_sli_end = self.hs['3D_tab1_roi_z_range_slider'].value[1]
        
        self.hs['3D_use_mask'].disabled = False
        self.hs['3D_use_anchor'].disabled = False
        self.hs['3D_tab1_mask_preview_fiji_checkbox'].disabled = False
        self.hs['moving_sli_search_half_range'].disabled = False
        self.hs['3D_chunk_sz'].disabled = False
        self.hs['3D_reg_method'].disabled = False
        self.hs['3D_ref_mode'].disabled = False
        self.hs['3D_tab1_reg_config_button'].disabled = False
        self.roi_3D_params_configured = True
        
    def tab1_3D_set_reg_config_button(self, a, rp_={}):
        self.use_mask = self.hs['3D_use_mask'].value
        self.use_anchor = self.hs['3D_use_anchor'].value
        self.mask_thres = self.hs['3D_mask_thres'].value
        self.mask_dilation = self.hs['3D_mask_dilation_width'].value
        self.reg_method = self.hs['3D_reg_method'].value
        self.ref_mode = self.hs['3D_ref_mode'].value
        self.moving_sli_search_half_range = self.hs['moving_sli_search_half_range'].value
        self.chunk_sz = self.hs['3D_chunk_sz'].value
        self.reg_3D_params_configured = True
        self.hs['3D_tab2_run_registration_button'].disabled = False
        self.hs['3D_tab2_run_registration_progress'].disabled = False
        self.hs['3D_tab1_reg_config_button_state'].value = 'Registration parameter configuration is completed!'
        
    def tab2_3D_run_registration_button(self, a, rp_={}):
        self.hs['3D_tab2_select_reg_pair_slider'].disabled = False
        self.hs['3D_tab2_reg_sli_search_range_slider'].disabled = False
        self.hs['3D_tab2_reg_sli_best_match'].disabled = False
        self.hs['3D_tab2_reg_review_finish_button'].disabled = False
        self.hs['3D_tab2_align_sli_best_match_record_button'].disabled = False
        self.alignment_best_match = {}
        
        reg = xr.regtools(dtype='3D_XANES', method=self.reg_method, mode='TRANSLATION')
        reg.set_method(self.reg_method)
        reg.set_ref_mode(self.ref_mode)
        reg.cal_set_anchor(self.scan_3D_id_s, self.scan_3D_id_e, self.fixed_scan_3D_id, raw_h5_top_dir=self.raw_3D_h5_top_dir)
        reg.set_chunk_sz(self.chunk_sz)
        reg.set_roi(self.roi)
        if self.use_mask:
            reg.use_mask = True
            reg.set_mask(self.mask)
        ffn = self.recon_3D_tiff_temp.format(self.fixed_scan_3D_id,
                                             str(self.fixed_sli_3D_id).zfill(5))
        reg.set_fixed_data(tifffile.imread(ffn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]])
        reg.set_3D_recon_path_template(self.recon_3D_tiff_temp)
        reg.set_saving(save_path=os.path.dirname(self.trial_reg_3D_save_file),
                       fn=os.path.basename(self.trial_reg_3D_save_file))
        reg.xanes3D_sli_search_half_range = self.moving_sli_search_half_range
        reg.xanes3D_recon_fixed_sli = self.fixed_sli_3D_id
        reg.reg_xanes3D_chunk()
        f = h5py.File(self.trial_reg_3D_save_file, 'r')
        self.trial_reg = np.ndarray(f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format(str(0).zfill(3))].shape)
        self.hs['3D_tab2_select_reg_pair_slider'].max = f['/trial_registration/trial_reg_parameters/alignment_pairs'].shape[0]-1
        self.hs['3D_tab2_select_reg_pair_slider'].min = 0 
        f.close()       
        self.hs['3D_tab2_reg_sli_search_range_slider'].max = self.moving_sli_search_half_range*2
        self.hs['3D_tab2_reg_sli_search_range_slider'].min = 1

        
    def tab2_3D_reg_review_finish_button(self, a, rp_={}):
        if self.read_alignment_option:
            self.alignment_file = self.hs["3D_tab2_read_alignment_dir"].selected
            try:
                if self.alignment_file.split('.')[-1] == 'json':
                    self.alignment_best_match = json.load(open(self.alignment_file, 'r'))
                else:
                    self.alignment_best_match = np.genfromtxt(self.alignment_file)
                self.hs['3D_tab2_align_recon_button'].disabled = False
                self.hs['3D_tab2_align_recon_progress'].disabled = False
            except:
                self.hs['3D_tab2_reg_review_state'].value = 'The specified alignment file does not exist!'
        else:
            f = h5py.File(self.trial_reg_3D_save_file, 'r')
            scan_id_s = f['/trial_registration/trial_reg_parameters/scan_ids'][0]
            scan_id_e = f['/trial_registration/trial_reg_parameters/scan_ids'][-1] 
            json.dump(self.alignment_best_match, open(self.trial_reg_3D_save_file.split('.')[0]+f'_{scan_id_s}-{scan_id_e}_zshift.json', 'w'))
            self.hs['3D_tab2_align_recon_button'].disabled = False
            self.hs['3D_tab2_align_recon_progress'].disabled = False
            
#        self.user_specfied_shift = np.array(self.alignment_best_match)
        
    def tab2_3D_reg_review_sli_best_match_record_button(self, a):
        self.alignment_pair = self.hs['3D_tab2_select_reg_pair_slider'].value
        self.alignment_best_match[str(self.alignment_pair)] = self.hs['3D_tab2_reg_sli_best_match'].value-1
        
    def tab2_3D_align_recon_button(self, a, rp_={}):
        f = h5py.File(self.trial_reg_3D_save_file, 'r')
        recon_top_dir = f['/trial_registration/data_directory_info/recon_top_dir'][()] 
        recon_path_template = f['/trial_registration/data_directory_info/recon_path_template'][()] 
        roi = f['/trial_registration/trial_reg_parameters/slice_roi'][:] 
        reg_method = f['/trial_registration/trial_reg_parameters/reg_method'][()].lower()
        ref_mode = f['/trial_registration/trial_reg_parameters/reg_ref_mode'][()].lower()
        scan_id_s = f['/trial_registration/trial_reg_parameters/scan_ids'][0]
        scan_id_e = f['/trial_registration/trial_reg_parameters/scan_ids'][-1]
        fixed_scan_id = f['/trial_registration/trial_reg_parameters/fixed_scan_id'][()]
        chunk_sz = f['/trial_registration/trial_reg_parameters/chunk_sz'][()]
        eng_list = f['/trial_registration/trial_reg_parameters/eng_list'][:]
        f.close()
        
        reg = xr.regtools(dtype='3D_XANES', method=reg_method, mode='TRANSLATION')
        reg.set_method(reg_method)
        reg.set_ref_mode(ref_mode)
        reg.cal_set_anchor(scan_id_s, scan_id_e, fixed_scan_id)
        reg.eng_list = eng_list
        
        reg.set_chunk_sz(chunk_sz)
        reg.set_roi(roi)
        reg.set_3D_recon_path_template(recon_path_template)
        reg.set_saving(save_path=os.path.dirname(self.trial_reg_3D_save_file), fn=os.path.basename(self.trial_reg_3D_save_file))
        reg.apply_xanes3D_chunk_shift(self.alignment_best_match,
                                      roi[4],
                                      roi[5],
                                      trialfn=self.trial_reg_3D_save_file,
                                      savefn=self.trial_reg_3D_save_file)
        
        
        
        

        
        
        