#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:19:17 2020

@author: xiao
"""

from ipywidgets import widgets
from ipyfilechooser import FileChooser
from IPython.display import display
import os, functools, glob, tifffile

import imagej

#try:
ij = imagej.init('/home/xiao/software/Fiji.app', headless=False)
ijui = ij.ui()
ijui.showUI()

from jnius import autoclass
WindowManager = autoclass('ij.WindowManager')
#except:
#    print('fiji is already up running!')
    
ij.py.run_macro("""run("Brightness/Contrast...");""")


class xanes_regtools_gui():
    def __init__(self, reg_params):
        self.hs = {}
        self.fc_configured = False
        self.scan_id_configured = False
        self.img = None
        reg_params['raw_h5_top_dir'] = None
        reg_params['recon_top_dir'] = None
        reg_params['trial_reg_save_file'] = None
        reg_params['roi'] = None
        reg_params['scan_id_s'] = None 
        reg_params['scan_id_e'] = None
        reg_params['chunk_sz'] = None
        reg_params['fixed_scan_id'] = None
        reg_params['fixed_sli'] = None
        reg_params['moving_sli_search_half_range'] = None
        reg_params['use_mask'] = None
        reg_params['mask_thres'] = None
        reg_params['mask_dilation'] = None
        reg_params['use_anchor'] = None
        reg_params['reg_method'] = None
        reg_params['ref_mode'] = None
        reg_params['trial_reg'] = None
        
    
    def gui_layout(self, reg_params):
        self.hs['form_global'] = widgets.Box()
        self.hs['form_global'].layout = {'border':'5px solid #00FF00', 'width':'700px', 'height':'650px'}
        
        self.hs['top_tabs'] = widgets.Tab()
        self.hs['3D_tabs'] = widgets.Tab()
        self.hs['2D_tabs'] = widgets.Tab()
        self.hs['top_tabs'].children = [self.hs['2D_tabs'],
                                        self.hs['3D_tabs']]
        self.hs['top_tabs'].set_title(0, '2D')
        self.hs['top_tabs'].set_title(1, '3D')
        
        ##### 3D tab0: File configuration
        form_3D_tab0 = widgets.VBox(display='flex')
        form_3D_tab0.layout = {'border':'3px solid #FFCC00', 'width':'655px', 'height': '570px'}
                                  
        box_file_setup = widgets.VBox(flex_flow='column')
        box_file_setup.layout = {'border':'3px solid #FFCC00', 'width':'650px', 'height': '350px'}
        label1 = widgets.HTML('<span style="background-color:rgb(135,206,250);">' + 'Choose raw_h5_top_dir' + '</span>')
        label2 = widgets.HTML('<span style="background-color:rgb(135,206,250);">' + 'Choose recon_top_dir' + '</span>')
        label3 = widgets.HTML('<span style="background-color:rgb(135,206,250);">' + 'trial_reg_save_file' + '</span>')
        self.hs["3D_tab0_fc_raw_dir"] = FileChooser(os.path.curdir, display='flex')
        self.hs["3D_tab0_fc_raw_dir"].default_path = '/media'
        self.hs["3D_tab0_fc_recon_dir"] = FileChooser(os.path.curdir, description_tip='recon_top_dir', display='flex')
        self.hs["3D_tab0_fc_recon_dir"].default_path = '/media'
        self.hs["3D_tab0_fc_trial_reg_file"] = FileChooser(os.path.curdir, description_tip='trial_reg_file...', display='flex') 
        self.hs["3D_tab0_fc_trial_reg_file"].default_path = '/media'
        self.hs["3D_tab0_fc_trial_reg_file"].default_filename = 'trial_reg.h5'
        self.hs["3D_tab0_fc_raw_dir"].on_trait_change(self.fc_raw_dir_changed, name='selected_path')
        
        box_button_confirm1 = widgets.HBox(display='flex')
        self.hs["3D_tab0_fc_confirm_button"] = widgets.Button(description='Confirm', description_tooltip='Confirm File Configurations')        
        self.hs['3D_tab0_fc_confirm_button_state'] = widgets.Text(value='Please set directories ...', description='Status', disabled=True)
        self.hs['3D_tab0_fc_confirm_button_state'].layout = {'flex_flow':'left', 'width':'505px'}
        self.hs["3D_tab0_fc_confirm_button"].on_click(functools.partial(self.on_click_fc_confirm_button, rp_=reg_params))
        box_button_confirm1.children = [self.hs["3D_tab0_fc_confirm_button"],
                                       self.hs['3D_tab0_fc_confirm_button_state']]
        box_button_confirm1.layout = {'align_items':'center'}        
        box_file_setup.children = [label1, self.hs["3D_tab0_fc_raw_dir"],
                                   label2, self.hs["3D_tab0_fc_recon_dir"],
                                   label3, self.hs["3D_tab0_fc_trial_reg_file"],
                                   box_button_confirm1]
        
        tab0_files_scan_id = widgets.HBox()
        self.hs["3D_tab0_scan_id_s"] = widgets.IntText(description='scan_id start',
                                                          display='flex', disabled=True)
        self.hs["3D_tab0_scan_id_s"].layout = {'width':'200px'}
        self.hs["3D_tab0_scan_id_e"] = widgets.IntText(description='scan_id end',
                                                          display='flex', disabled=True)
        self.hs["3D_tab0_scan_id_e"].layout = {'width':'200px'}                
        self.hs["3D_tab0_fixed_scan_id"] = widgets.IntSlider(description='fixed_scan_id', disabled=True)
        self.hs["3D_tab0_fixed_sli_id"] = widgets.IntSlider(description='fixed_sli_id', disabled=True)
        self.hs["3D_tab0_fixed_sli_fiji_checkbox"] = widgets.Checkbox(value=False, description='Fiji', disabled=True)
        def tab0_3D_scan_id_s_change(a):  
            if os.path.exists(self.raw_h5_temp.format(str(self.hs["3D_tab0_scan_id_s"].value))):
                if os.path.exists(self.recon_top_dir.format(str(self.hs["3D_tab0_scan_id_s"].value))):
                    if self.hs["3D_tab0_scan_id_e"].value < self.hs["3D_tab0_scan_id_s"].value:
                        self.hs["3D_tab0_scan_id_e"].value = self.hs["3D_tab0_scan_id_s"].value
                    self.hs["3D_tab0_fixed_scan_id"].disabled = False
                    self.hs["3D_tab0_fixed_scan_id"].min = self.hs["3D_tab0_scan_id_s"].value
                    fn = sorted(glob.glob(os.path.join(self.recon_top_dir.format(self.hs["3D_tab0_scan_id_s"].value), '*.tiff')))[0]
                    self.img = tifffile.imread(fn)
                    self.hs['3D_scan_id_confirm_button_state'].value = 'scan ids are changed ...'
                else:
                    self.hs['3D_scan_id_confirm_button_state'].value = 'specified scan_id starting number does not exist in recon top dir ...'
                    self.hs["3D_tab0_fixed_scan_id"].disabled = True
                    self.hs["3D_tab0_fixed_sli_id"].disabled = True
                    self.hs["3D_tab0_fixed_sli_fiji_checkbox"].disabled = True
#                    self.hs["scan_id_confirm_button"].disabled = True
            else:
                self.hs['3D_scan_id_confirm_button_state'].value = 'raw h5 file does not exist in the raw top dir ...'
                self.hs["3D_tab0_fixed_scan_id"].disabled = True
                self.hs["3D_tab0_fixed_sli_id"].disabled = True
                self.hs["3D_tab0_fixed_sli_fiji_checkbox"].disabled = True
            
        def tab0_3D_scan_id_e_change(a):
            if os.path.exists(self.raw_h5_temp.format(str(self.hs["3D_tab0_scan_id_e"].value))):
                if os.path.exists(self.recon_top_dir.format(str(self.hs["3D_tab0_scan_id_e"].value))):
                    if self.hs["3D_tab0_scan_id_e"].value < self.hs["3D_tab0_scan_id_s"].value:
                        self.hs["3D_tab0_scan_id_s"].value = self.hs["3D_tab0_scan_id_e"].value
                    self.hs["3D_tab0_fixed_scan_id"].disabled = False
                    self.hs["3D_tab0_fixed_scan_id"].max = self.hs["3D_tab0_scan_id_e"].value
                    self.hs['3D_scan_id_confirm_button_state'].value = 'scan ids are changed ...'
                else:
                    self.hs['3D_scan_id_confirm_button_state'].value = 'specified scan_id starting number does not exist in recon top dir ...'
                    self.hs["3D_tab0_fixed_scan_id"].disabled = True
                    self.hs["3D_tab0_fixed_sli_id"].disabled = True
                    self.hs["3D_tab0_fixed_sli_fiji_checkbox"].disabled = True
#                    self.hs["scan_id_confirm_button"].disabled = True
            else:
                self.hs['3D_scan_id_confirm_button_state'].value = 'raw h5 file does not exist in the raw top dir ...'
                self.hs["3D_tab0_fixed_scan_id"].disabled = True
                self.hs["3D_tab0_fixed_sli_id"].disabled = True

        def tab0_3D_fixed_scan_id_slider_change(a):
            if os.path.exists(self.recon_top_dir.format(str(self.hs["3D_tab0_fixed_scan_id"].value))):
                self.hs["3D_tab0_fixed_sli_id"].disabled = False
                self.hs["3D_tab0_fixed_sli_fiji_checkbox"].disabled = False
                self.hs['3D_scan_id_confirm_button_state'].value = 'fixed scan id is changed ...'
                file_list = sorted(glob.glob(os.path.join(self.recon_top_dir.format(self.hs["3D_tab0_fixed_scan_id"].value), '*.tiff')))
                self.hs["3D_tab0_fixed_sli_id"].min = int(file_list[0].split('.')[0].split('_')[-1])
                self.hs["3D_tab0_fixed_sli_id"].max = int(file_list[-1].split('.')[0].split('_')[-1])
#                self.hs['3D_scan_id_confirm_button_state'].value = '{}'.format(str(self.hs["3D_tab0_fixed_scan_id"].value))
            else:
                self.hs["3D_tab0_fixed_scan_id"].disabled = True
                self.hs["3D_tab0_fixed_sli_id"].disabled = True
                self.hs["3D_tab0_fixed_sli_fiji_checkbox"].disabled = True
                self.hs['3D_scan_id_confirm_button_state'].value = 'recon with fixed scan id {} is changed ...'.format(str(self.hs["3D_tab0_fixed_scan_id"].value))
            
        def tab0_3D_fixed_sli_slider_change(a):
            if self.hs["3D_tab0_fixed_sli_fiji_checkbox"].value:
                self.hs['3D_scan_id_confirm_button_state'].value = 'fixed slice id is changed ...'
                self.viewer_ip.setSlice(self.hs["3D_tab0_fixed_sli_id"].value)
            self.fixed_sli_id = self.hs["3D_tab0_fixed_sli_id"].value
            
        def tab0_3D_fixed_sli_fiji_checked(a):
            if self.hs["3D_tab0_fixed_sli_fiji_checkbox"].value:
                self.fn0 = self.recon_tiff_temp.format(self.hs["3D_tab0_fixed_scan_id"].value,
                                             str(self.hs["3D_tab0_fixed_sli_id"].min).zfill(5))
                args = {'directory':self.fn0, 'start':1}
                ij.py.run_plugin(" Open VirtualStack", args)
                self.viewer_ip = WindowManager.getCurrentImage()
            else:
                self.viewer_ip.close()
                self.viewer_ip = None
            
        self.hs["3D_tab0_scan_id_s"].observe(tab0_3D_scan_id_s_change, names='value')
        self.hs["3D_tab0_scan_id_e"].observe(tab0_3D_scan_id_e_change, names='value')      
        self.hs["3D_tab0_fixed_sli_fiji_checkbox"].observe(tab0_3D_fixed_sli_fiji_checked, names='value')
        self.hs["3D_tab0_fixed_sli_id"].observe(tab0_3D_fixed_sli_slider_change, names='value')
        self.hs["3D_tab0_fixed_scan_id"].observe(tab0_3D_fixed_scan_id_slider_change, names='value')
        
        box_button_confirm2 = widgets.HBox(display='flex')
        self.hs["scan_id_confirm_button"] = widgets.Button(description='Confirm',
                                                           description_tooltip='Confirm scan_id Configurations')        
        self.hs['3D_scan_id_confirm_button_state'] = widgets.Text(value='Please set scan ids ...', description='Status', disabled=True)
        self.hs['3D_scan_id_confirm_button_state'].layout = {'flex_flow':'left', 'width':'505px'}
        box_button_confirm2.children = [self.hs["scan_id_confirm_button"],
                                        self.hs['3D_scan_id_confirm_button_state']]
        
        self.hs["scan_id_confirm_button"].on_click(functools.partial(self.on_click_scan_id_confirm_button, rp_=reg_params))
        
        tab0_files_scan_id.children = [self.hs["3D_tab0_scan_id_s"],
                                       self.hs["3D_tab0_scan_id_e"]]
        form_3D_tab0.children=[box_file_setup,
                                  tab0_files_scan_id,
                                  self.hs["3D_tab0_fixed_scan_id"],
                                  self.hs["3D_tab0_fixed_sli_id"],
                                  self.hs["3D_tab0_fixed_sli_fiji_checkbox"],
                                  box_button_confirm2]
        
        ##### 3D tab1: configure registration parameters
        form_3D_tab1 = widgets.Box()
        form_3D_tab1.layout = {'border': '3px solid #FFCC00', 'width': '655px', 'height': '570px'}
        box_roi = widgets.VBox()
        box_roi.layout = {'border': '3px solid #FFCC00', 'width': '650px', 'height': '120px'}
        label4 = widgets.HTML('<span style="background-color:rgb(135,206,250);">' + 'define ROI' + '</span>')
        self.hs['3D_tab1_roi_x_range'] = widgets.IntRangeSlider(value=[5, 7],
                                                        min=0,
                                                        max=10,
                                                        step=1,
                                                        description='x range:',
                                                        disabled=True,
                                                        continuous_update=False,
                                                        orientation='horizontal',
                                                        readout=True,
                                                        readout_format='d')
        self.hs['3D_tab1_roi_x_range'].layout = {'width': '600px'}
        self.hs['3D_tab1_roi_y_range'] = widgets.IntRangeSlider(value=[5, 7],
                                                        min=0,
                                                        max=10,
                                                        step=1,
                                                        description='y range:',
                                                        disabled=True,
                                                        continuous_update=False,
                                                        orientation='horizontal',
                                                        readout=True,
                                                        readout_format='d')
        self.hs['3D_tab1_roi_y_range'].layout = {'width': '600px'}

        box_roi.children = [label4, self.hs['3D_tab1_roi_x_range'], self.hs['3D_tab1_roi_x_range']]
        form_3D_tab1.children = [box_roi]
        
        self.hs['3D_tabs'].children = [form_3D_tab0,
                                       form_3D_tab1]
        self.hs['form_global'].children=[self.hs['top_tabs']]
        self.hs['3D_tabs'].set_title(0, 'File Configurations')
        self.hs['3D_tabs'].set_title(1, 'Image Settings')
        display(self.hs['form_global'])
        
    def on_click_fc_confirm_button(self, a, rp_={}):
        if self.hs["3D_tab0_fc_raw_dir"].selected_path:
            if self.hs["3D_tab0_fc_recon_dir"].selected_path:
                if self.hs["3D_tab0_fc_trial_reg_file"].selected_filename:
                    self.hs['3D_tab0_fc_confirm_button_state'].value = 'Configuration is done!'
                    rp_['raw_h5_top_dir'] = self.hs["3D_tab0_fc_raw_dir"].selected_path
                    rp_['recon_top_dir'] = self.hs["3D_tab0_fc_recon_dir"].selected_path
                    rp_['trial_reg_save_file'] = self.hs["3D_tab0_fc_trial_reg_file"].selected
                    self.recon_top_dir = os.path.join(rp_['recon_top_dir'],
                                                      'recon_fly_scan_id_{0}')
                    self.recon_tiff_temp = os.path.join(rp_['recon_top_dir'],
                                                        'recon_fly_scan_id_{0}',
                                                        'recon_fly_scan_id_{0}_{1}.tiff')
                    self.raw_h5_temp = os.path.join(rp_['raw_h5_top_dir'],
                                                    'fly_scan_id_{}.h5')
                    self.hs["3D_tab0_scan_id_s"].disabled = False
                    self.hs["3D_tab0_scan_id_e"].disabled = False
                    self.fc_configured = True
                else:
                    self.hs['3D_tab0_fc_confirm_button_state'].value = "trial_reg_file is not defined!"
            else:
                self.hs['3D_tab0_fc_confirm_button_state'].value = "recon_dir is not defined!"
        else:
            self.hs['3D_tab0_fc_confirm_button_state'].value = "raw_dir is not defined!"
                
    def on_click_scan_id_confirm_button(self, a, rp_={}):  
        if self.hs["3D_tab0_fixed_scan_id"].disabled: 
            self.hs['3D_scan_id_confirm_button_state'].value = 'fixed scan id is not configured yet!'
        else:
            if self.hs["3D_tab0_fixed_sli_id"].disabled: 
                self.hs['3D_scan_id_confirm_button_state'].value = 'fixed slice id is not configured yet!'
            else:
                rp_['scan_id_s'] = self.hs["3D_tab0_scan_id_s"].value
                rp_['scan_id_e'] = self.hs["3D_tab0_scan_id_e"].value
                rp_['fixed_scan_id'] = self.hs["3D_tab0_fixed_scan_id"].value
                rp_['fixed_sli'] = self.hs["3D_tab0_fixed_sli_id"].value
                self.hs['3D_scan_id_confirm_button_state'].value = 'scan ids are configured!'
                self.scan_id_configured = True
                
    def fc_raw_dir_changed(self, a):
        self.hs["3D_tab0_fc_raw_dir"].default_path = self.hs["3D_tab0_fc_raw_dir"].selected_path
        self.hs["3D_tab0_fc_recon_dir"].default_path = self.hs["3D_tab0_fc_raw_dir"].selected_path
        self.hs["3D_tab0_fc_trial_reg_file"].default_path = self.hs["3D_tab0_fc_raw_dir"].selected_path
        
        
        
        