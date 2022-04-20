#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:19:17 2020

@author: xiao
"""

import traitlets
from tkinter import Tk, filedialog
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
    

# class SelectFilesButton(widgets.Button):
#     """A file widget that leverages tkinter.filedialog."""

#     def __init__(self, option='askopenfilename',
#                  layout={}, value='Select File ...'):
#         """
        

#         Parameters
#         ----------
#         option : TYPE, optional
#             Type of browser in ['askopenfilename', 'askdirectory', 'asksaveasfilename'] 
#             The default is 'askopenfilename'.

#         Returns
#         -------
#         None.

#         """
#         super().__init__()
#         # Build a box
#         self.box = widgets.HBox()
#         self.box.layout = layout
#         # Add the selected_files trait
#         self.add_traits(files=traitlets.traitlets.List())
#         # Create the button.
#         self.option = option
#         if self.option == 'askopenfilename':
#             self.description = "Select File"            
#         elif self.option == 'asksaveasfilename':
#             self.description = "Save As File"
#         elif self.option == 'askdirectory':
#             self.description = "Choose Dir"
#         self.icon = "square-o"
#         self.style.button_color = "orange"
        
#         # Create a status bar        
#         self.status = widgets.Text(disabled=True, value=value)
        
#         # Set on click behavior.        
#         self.on_click(self.select_files)

#     @staticmethod
#     def select_files(b):
#         """Generate instance of tkinter.filedialog.

#         Parameters
#         ----------
#         b : obj:
#             An instance of ipywidgets.widgets.Button 
#         """
#         b.box.children = [b, b.status]
#         if b.option == 'askopenfilename':
#             files = filedialog.askopenfilename()
#         elif b.option == 'askdirectory':
#             files = filedialog.askdirectory()
#         elif b.option == 'asksaveasfilename':
#             files = filedialog.asksaveasfilename()
        
#         if len(files) == 0:
#             b.files = ['']
#             if b.option == 'askopenfilename':
#                 b.description = "Select File"
#             elif b.option == 'asksaveasfilename':
#                 b.description = "Save As File"
#             elif b.option == 'askdirectory':
#                 b.description = "Choose Dir"
#             b.icon = "square-o"
#             b.style.button_color = "orange"
#         else:
#             b.files = [files]
#             if b.option == 'askopenfilename':
#                 b.description = "File Selected"
#             elif b.option == 'asksaveasfilename':
#                 b.description = "Filename Chosen"
#             elif b.option == 'askdirectory':
#                 b.description = "Dir Selected"
#             b.icon = "check-square-o"
#             b.style.button_color = "lightgreen"
#         display(b.box)

class SelectFilesButton(widgets.Button):
    """A file widget that leverages tkinter.filedialog."""

    def __init__(self, option='askopenfilename', text_h=None):
        """
        

        Parameters
        ----------
        option : TYPE, optional
            Type of browser in ['askopenfilename', 'askdirectory', 'asksaveasfilename'] 
            The default is 'askopenfilename'.

        Returns
        -------
        None.

        """
        super().__init__()
        # Add the selected_files trait
        self.add_traits(files=traitlets.traitlets.List())
        # Create the button.
        self.text_h = text_h
        self.option = option
        if self.option == 'askopenfilename':
            self.description = "Select File"            
        elif self.option == 'asksaveasfilename':
            self.description = "Save As File"
        elif self.option == 'askdirectory':
            self.description = "Choose Dir"
        self.icon = "square-o"
        self.style.button_color = "orange"
        
        # define default directory/file options
        self.initialdir = '/NSLS2/xf18id1/users/2020Q1/YUAN_YANG_Proposal_305811'
        self.ext = '*.h5'
        self.initialfile = '3D_trial_reg.h5'
        self.open_filetypes = (('text files', '*.txt'), ('json files', '*.json'))
        # self.save_filetypes = (('hdf5 files', '*.h5'))
        # Set on click behavior.        
        self.on_click(self.select_files)

    @staticmethod
    def select_files(b):
        """Generate instance of tkinter.filedialog.

        Parameters
        ----------
        b : obj:
            An instance of ipywidgets.widgets.Button 
        """
        # Create Tk root
        root = Tk()
        # Hide the main window
        root.withdraw()
        # Raise the root to the top of all windows.
        root.call('wm', 'attributes', '.', '-topmost', True)
        # List of selected fileswill be set to b.value
        
        if b.option == 'askopenfilename':
            files = filedialog.askopenfilename(initialdir=b.initialdir,
                                               defaultextension='*.json',
                                               filetypes=b.open_filetypes)
        elif b.option == 'askdirectory':
            files = filedialog.askdirectory(initialdir=b.initialdir)
        elif b.option == 'asksaveasfilename':
            files = filedialog.asksaveasfilename(initialdir=b.initialdir,
                                                 initialfile=b.initialfile,
                                                 defaultextension='*.h5')
        
        if b.text_h is None:
            if len(files) == 0:
                b.files = ['']
                if b.option == 'askopenfilename':
                    b.description = "Select File"
                elif b.option == 'asksaveasfilename':
                    b.description = "Save As File"
                elif b.option == 'askdirectory':
                    b.description = "Choose Dir"
                b.icon = "square-o"
                b.style.button_color = "orange"
            else:
                b.files = [files]
                if b.option == 'askopenfilename':
                    b.description = "File Selected"
                elif b.option == 'asksaveasfilename':
                    b.description = "Filename Chosen"
                elif b.option == 'askdirectory':
                    b.description = "Dir Selected"
                b.icon = "check-square-o"
                b.style.button_color = "lightgreen"
        else:
            if len(files) == 0:
                b.files = ['']
                if b.option == 'askopenfilename':
                    b.description = "Select File"
                    b.text_h.value = 'select file ...'
                elif b.option == 'asksaveasfilename':
                    b.description = "Save As File"
                    b.text_h.value = 'select a path and specify a file name ...'
                elif b.option == 'askdirectory':
                    b.description = "Choose Dir"
                    b.text_h.value = 'select a directory ...'
                b.icon = "square-o"
                b.style.button_color = "orange"
            else:
                b.files = [files]
                if b.option == 'askopenfilename':
                    b.description = "File Selected"
                elif b.option == 'asksaveasfilename':
                    b.description = "Filename Chosen"
                elif b.option == 'askdirectory':
                    b.description = "Dir Selected"
                b.icon = "check-square-o"
                b.style.button_color = "lightgreen"
                b.text_h.value = os.path.abspath(b.files[0])
        
# my_button = SelectFilesButton(option='askdirectory')
# my_button


class xanes_regtools_gui():
    def __init__(self, form_sz=[650, 740]):
        self.hs = {}
        self.form_sz = form_sz
        
        self.xanes2D_filepath_configured = False
        self.xanes2D_indices_configured = False
        self.xanes2D_roi_configured = False
        self.xanes2D_reg_params_configured = False
        self.xanes2D_reg_done = False
        self.xanes2D_reg_review_done = False
        self.xanes2D_alignment_done = False        
        self.xanes2D_read_alignment_option = False


        # self.xanes3D_fiji_mask_viewer_ip = None
        self.xanes3D_fiji_windows = {'virtural_stack_preview_viewer':{'ip':None,
                                                                      'fiji_id':None},
                                     'mask_viewer':{'ip':None,
                                                    'fiji_id':None}}
        
        self.xanes3D_filepath_configured = False
        self.xanes3D_indices_configured = False
        self.xanes3D_roi_configured = False
        self.xanes3D_reg_params_configured = False
        self.xanes3D_reg_done = False
        self.xanes3D_reg_review_done = False
        self.xanes3D_alignment_done = False
        self.xanes3D_use_existing_reg_file = False
        self.xanes3D_use_existing_reg_review = False
        self.xanes3D_reg_use_anchor = True
        self.xanes3D_reg_use_mask = True
        self.xanes3D_reg_review_ready = False
        
        self.xanes3D_img = None
        self.xanes3D_raw_h5_path_set = False
        self.xanes3D_recon_path_set = False
        self.xanes3D_save_trial_set = False
        self.xanes3D_scan_id_set = False
        self.xanes3D_fixed_scan_id_set = False
        self.xanes3D_fixed_sli_id_set = False
        self.xanes3D_reg_file_read = False
        
        self.alignment_best_match = {}
        self.xanes3D_reg_mask_dilation_width = 0
        self.xanes3D_reg_mask_thres = 0

        
    def get_handles(self, handle_dict_name, n):
        """

        Parameters
        ----------
        handle_dict_name : string
            a handle's name in the handleset dictionary.
        n : int
            total number of children under the item with handle_dict_name.

        Returns
        -------
        a : widget handles
            children widget handles under handle_dict_name.

        """
        a = []
        jj = 0
        idx = handle_dict_name.split('_')[0]
        if n == -1:
            for ii in self.hs.keys():
                for jj in range(15):
                    if f'{idx}[{jj}]_' in ii:
                            a.append(self.hs[ii])
        else:
            for ii in self.hs.keys():
                for jj in range(n):
                    if f'{idx}[{jj}]_' in ii:
                        a.append(self.hs[ii])
        return a

    
    def get_decendant(self, handle_dict_name, level=-1):        
        for ii in self.hs.keys():
            if handle_dict_name in ii:
                self_handle = [self.hs[ii]]
        children_handles = []
        children_handles.append(self_handle)
        temp_handles = []
        actual_level = 0
        bottom = False
        
        if level == -1:
            while not bottom:
                for child in self_handle:
                    try:
                        children_handles.append(child.children)
                        for jj in range(len(child.children)):
                            temp_handles.append(child.children[jj])
                        actual_level += 1
                    except Exception as e:
                        # print(str(e))
                        bottom = True
                self_handle = temp_handles
                temp_handles = []                
        else:
            for ii in range(level):
                for child in self_handle:
                    try:
                        children_handles.append(child.children)
                        for jj in range(len(child.children)):
                            temp_handles.append(child.children[jj])
                        actual_level = ii+1
                    except Exception as e:
                        # print(str(e))
                        bottom = True
                self_handle = temp_handles
                temp_handles = [] 
        return children_handles, actual_level
    
    def fiji_viewer_on(self, viewer_name='virtural_stack_preview_viewer'):
        if viewer_name == 'virtural_stack_preview_viewer':
            if WindowManager.getIDList() is None:
                self.fn0 = self.xanes3D_recon_3D_tiff_temp.format(self.xanes3D_fixed_scan_id,
                                                                  str(min(self.xanes3D_available_recon_ids)).zfill(5))
                args = {'directory':self.fn0, 'start':1}
                ij.py.run_plugin(" Open VirtualStack", args)
                self.xanes3D_fiji_windows['virtural_stack_preview_viewer'] = {'ip':WindowManager.getCurrentImage(),
                                                                              'fiji_id':WindowManager.getIDList()[-1]}              
            elif not self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['fiji_id'] in WindowManager.getIDList():
                self.fn0 = self.xanes3D_recon_3D_tiff_temp.format(self.xanes3D_fixed_scan_id,
                                                                  str(min(self.xanes3D_available_recon_ids)).zfill(5))
                args = {'directory':self.fn0, 'start':1}
                ij.py.run_plugin(" Open VirtualStack", args)
                self.xanes3D_fiji_windows['virtural_stack_preview_viewer'] = {'ip':WindowManager.getCurrentImage(),
                                                                              'fiji_id':WindowManager.getIDList()[-1]}
        elif viewer_name == 'mask_viewer':
            if self.xanes3D_img is None:
                self.xanes3D_img = np.ndarray([self.xanes3D_roi[5]-self.xanes3D_roi[4]+1,
                                               self.xanes3D_roi[1]-self.xanes3D_roi[0],
                                               self.xanes3D_roi[3]-self.xanes3D_roi[2]])
                self.xanes3D_mask = np.ndarray([self.xanes3D_roi[1]-self.xanes3D_roi[0],
                                                self.xanes3D_roi[3]-self.xanes3D_roi[2]])
            elif self.xanes3D_img.shape != (self.xanes3D_roi[5]-self.xanes3D_roi[4]+1,
                                            self.xanes3D_roi[1]-self.xanes3D_roi[0],
                                            self.xanes3D_roi[3]-self.xanes3D_roi[2]):
                self.xanes3D_img = np.ndarray([self.xanes3D_roi[5]-self.xanes3D_roi[4]+1,
                                               self.xanes3D_roi[1]-self.xanes3D_roi[0],
                                               self.xanes3D_roi[3]-self.xanes3D_roi[2]])
                self.xanes3D_mask = np.ndarray([self.xanes3D_roi[1]-self.xanes3D_roi[0],
                                                self.xanes3D_roi[3]-self.xanes3D_roi[2]])
            if WindowManager.getIDList() is None:
                cnt = 0
                for ii in range(self.xanes3D_roi[4], self.xanes3D_roi[5]+1):
                    fn = self.xanes3D_recon_3D_tiff_temp.format(self.xanes3D_fixed_scan_id, str(ii).zfill(5))
                    self.xanes3D_img[cnt, ...] = tifffile.imread(fn)[self.xanes3D_roi[0]:self.xanes3D_roi[1],
                                                                     self.xanes3D_roi[2]:self.xanes3D_roi[3]]
                    cnt += 1
                    
                ijui.show(ij.py.to_java(self.xanes3D_img))
                self.xanes3D_fiji_windows['mask_viewer']['ip'] = WindowManager.getCurrentImage()
                self.xanes3D_fiji_windows['mask_viewer']['ip'].setSlice(self.xanes3D_fixed_sli_id-self.xanes3D_roi[4])
                self.xanes3D_fiji_windows['mask_viewer']['fiji_id'] = WindowManager.getIDList()[-1]
                self.xanes3D_fiji_windows['mask_viewer']['ip'].setTitle('mask preview')
            elif not self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['fiji_id'] in WindowManager.getIDList():
                cnt = 0
                for ii in range(self.xanes3D_roi[4], self.xanes3D_roi[5]+1):
                    fn = self.xanes3D_recon_3D_tiff_temp.format(self.xanes3D_fixed_scan_id, str(ii).zfill(5))
                    self.xanes3D_img[cnt, ...] = tifffile.imread(fn)[self.xanes3D_roi[0]:self.xanes3D_roi[1],
                                                                     self.xanes3D_roi[2]:self.xanes3D_roi[3]]
                    cnt += 1
                    
                ijui.show(ij.py.to_java(self.xanes3D_img))
                self.xanes3D_fiji_windows['mask_viewer']['ip'] = WindowManager.getCurrentImage()
                self.xanes3D_fiji_windows['mask_viewer']['ip'].setSlice(self.xanes3D_fixed_sli_id-self.xanes3D_roi[4])
                self.xanes3D_fiji_windows['mask_viewer']['fiji_id'] = WindowManager.getIDList()[-1]
                self.xanes3D_fiji_windows['mask_viewer']['ip'].setTitle('mask preview')
                
    def fiji_viewer_state(self, viewer_name='virtural_stack_preview_viewer'):
        if viewer_name == 'virtural_stack_preview_viewer':
            if (not self.xanes3D_recon_3D_tiff_temp) | (not self.xanes3D_fixed_scan_id) | (not self.xanes3D_available_recon_ids):
                data_state = False
            else:
                data_state = True
            if WindowManager.getIDList() is None:
                viewer_state =  False              
            elif not self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['fiji_id'] in WindowManager.getIDList():
                viewer_state =  False
            else:
                viewer_state =  True
                
        elif viewer_name == 'mask_viewer':
            if self.xanes3D_img is None:
                data_state = False
            elif self.xanes3D_img.shape != (self.xanes3D_roi[5]-self.xanes3D_roi[4]+1,
                                            self.xanes3D_roi[1]-self.xanes3D_roi[0],
                                            self.xanes3D_roi[3]-self.xanes3D_roi[2]):
                data_state = False
            if WindowManager.getIDList() is None:
                viewer_state = False
            elif not self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['fiji_id'] in WindowManager.getIDList():
                viewer_state = False
            else:
                viewer_state = False
        else:
            print('Unrecognized viewer name')
            data_state = False
            viewer_state = False
        return data_state, viewer_state
    
    def enable_disable_boxes(self, boxes, disabled=True, level=-1):
        for box in boxes:
            child_handles, level = self.get_decendant(box, level=level)
            # print(box, child_handles, level)
            for ii in child_handles:
                for child in ii:
                    try:
                        child.disabled = disabled
                    except Exception as e:
                        # print(str(e))
                        pass
                    
    def boxes_logic(self):
        if not self.xanes3D_filepath_configured:
            boxes = ['config_indices_box',
                     '3D_roi_box',
                     'config_reg_params_box',
                     'run_reg_box',
                     'review_reg_results_box',
                     'align_recon_box']
            self.enable_disable_boxes(boxes, disabled=True, level=-1)
        elif ((self.xanes3D_filepath_configured & self.xanes3D_use_existing_reg_file) 
               & (not self.xanes3D_reg_review_done)):
            boxes = ['review_reg_results_box']
            self.enable_disable_boxes(boxes, disabled=False, level=-1)
            self.hs['L[0][2][2][1][4][0]_confirm_review_results_text'].disabled = True
            if self.xanes3D_use_existing_reg_review:
                boxes = ['reg_pair_box',
                         'zshift_box']
                self.enable_disable_boxes(boxes, disabled=True, level=-1)
            else:
                boxes = ['reg_pair_box']
                self.enable_disable_boxes(boxes, disabled=False, level=-1)
                if self.xanes3D_reg_review_ready:
                    boxes = ['zshift_box']
                    self.enable_disable_boxes(boxes, disabled=False, level=-1)
                else:
                    boxes = ['zshift_box']
                    self.enable_disable_boxes(boxes, disabled=True, level=-1)
                
            boxes = ['config_indices_box',
                     '3D_roi_box',
                     'config_reg_params_box',
                     'run_reg_box',
                     'align_recon_box']
            self.enable_disable_boxes(boxes, disabled=True, level=-1)
        elif ((self.xanes3D_filepath_configured & self.xanes3D_use_existing_reg_file) 
               & (self.xanes3D_reg_review_done)):
            boxes = ['config_indices_box',
                     '3D_roi_box',
                     'config_reg_params_box',
                     'run_reg_box']
            self.enable_disable_boxes(boxes, disabled=True, level=-1)
            if not self.xanes3D_roi_configured:
                boxes = ['review_reg_results_box',
                         'align_recon_box']
                self.enable_disable_boxes(boxes, disabled=False, level=-1)
            else:
                boxes = ['review_reg_results_box',
                         'align_recon_box']
                self.enable_disable_boxes(boxes, disabled=False, level=-1)
                boxes = ['align_recon_optional_slice_region_box']
                self.enable_disable_boxes(boxes, disabled=True, level=-1)
        elif ((self.xanes3D_filepath_configured & (not self.xanes3D_use_existing_reg_file))
               & (not self.xanes3D_indices_configured)):
            if not self.xanes3D_scan_id_set:
                boxes = ['config_indices_box',
                         '3D_roi_box',
                         'config_reg_params_box',
                         'run_reg_box',
                         'review_reg_results_box',
                         'align_recon_box']
                self.enable_disable_boxes(boxes, disabled=True, level=-1)
                boxes = ['scan_id_range_box']
                self.enable_disable_boxes(boxes, disabled=False, level=-1)                
            # elif not self.xanes3D_fixed_scan_id_set:
            #     boxes = ['config_indices_box',
            #              '3D_roi_box',
            #              'config_reg_params_box',
            #              'run_reg_box',
            #              'review_reg_results_box',
            #              'align_recon_box']
            #     self.enable_disable_boxes(boxes, disabled=True, level=-1)
            #     boxes = ['scan_id_range_box',
            #              'fixed_id_box']
            #     self.enable_disable_boxes(boxes, disabled=False, level=-1)
            else:
                boxes = ['3D_roi_box',
                         'config_reg_params_box',
                         'run_reg_box',
                         'review_reg_results_box',
                         'align_recon_box']
                self.enable_disable_boxes(boxes, disabled=True, level=-1)
                boxes = ['config_indices_box']
                self.enable_disable_boxes(boxes, disabled=False, level=-1)
        elif ((self.xanes3D_filepath_configured & (not self.xanes3D_use_existing_reg_file)
               & self.xanes3D_indices_configured) & (not self.xanes3D_roi_configured)):
            boxes = ['config_reg_params_box',
                    'run_reg_box',
                    'review_reg_results_box',
                    'align_recon_box']
            self.enable_disable_boxes(boxes, disabled=True, level=-1)
            boxes = ['3D_roi_box']
            self.enable_disable_boxes(boxes, disabled=False, level=-1)
            self.hs['L[0][2][1][0][2][0]_confirm_roi_text'].disabled = True
        elif ((self.xanes3D_filepath_configured & (not self.xanes3D_use_existing_reg_file)
               & self.xanes3D_indices_configured & self.xanes3D_roi_configured) & 
              (not self.xanes3D_reg_params_configured)):
            boxes = ['run_reg_box',
                    'review_reg_results_box',
                    'align_recon_box']
            self.enable_disable_boxes(boxes, disabled=True, level=-1)
            boxes = ['config_reg_params_box']
            self.enable_disable_boxes(boxes, disabled=False, level=-1)
            self.hs['L[0][2][1][1][5][0]_confirm_reg_params_text'].disabled=True
            if self.xanes3D_fiji_windows['mask_viewer']['fiji_id'] not in WindowManager.getIDList():
                self.xanes3D_fiji_windows['mask_viewer']['fiji_id'] = None
                self.xanes3D_fiji_windows['mask_viewer']['ip'] = None
                boxes = ['mask_options_box']
                self.enable_disable_boxes(boxes, disabled=False, level=-1)
            else:
                if self.xanes3D_reg_use_mask:
                    boxes = ['mask_dilation_slider',
                             'mask_thres_slider']
                    self.enable_disable_boxes(boxes, disabled=False, level=-1)
                    self.hs['L[0][2][1][1][4][0]_reg_method_dropdown'].options = ['MPC', 'PC', 'SR']                    
                else:
                    boxes = ['mask_dilation_slider',
                             'mask_thres_slider']
                    self.enable_disable_boxes(boxes, disabled=True, level=-1)
                    self.hs['L[0][2][1][1][4][0]_reg_method_dropdown'].options = ['PC', 'SR']
        elif ((self.xanes3D_filepath_configured & (not self.xanes3D_use_existing_reg_file)
               & self.xanes3D_indices_configured & self.xanes3D_roi_configured & 
              self.xanes3D_reg_params_configured) & (not self.xanes3D_reg_done)):
            boxes = ['run_reg_box']
            self.enable_disable_boxes(boxes, disabled=False, level=-1) 
            self.hs['L[0][2][2][0][1][1]_run_reg_text'].disabled = True
        elif ((self.xanes3D_filepath_configured & (not self.xanes3D_use_existing_reg_file)
               & self.xanes3D_indices_configured & self.xanes3D_roi_configured & 
              self.xanes3D_reg_params_configured & self.xanes3D_reg_done) &
              (not self.xanes3D_reg_review_done)):
            boxes = ['review_reg_results_box']
            self.enable_disable_boxes(boxes, disabled=False, level=-1)
            self.hs['L[0][2][2][1][4][0]_confirm_review_results_text'].disabled = True
            boxes = ['align_recon_box']
            self.enable_disable_boxes(boxes, disabled=True, level=-1)
            if self.hs['L[0][2][2][1][1][1]_read_alignment_checkbox'].value:
                boxes = ['reg_pair_box',
                         'zshift_box']
                self.enable_disable_boxes(boxes, disabled=True, level=-1)
            else:
                boxes = ['reg_pair_box',
                         'zshift_box']
                self.enable_disable_boxes(boxes, disabled=False, level=-1)
        elif ((self.xanes3D_filepath_configured & (not self.xanes3D_use_existing_reg_file)
               & self.xanes3D_indices_configured & self.xanes3D_roi_configured & 
              self.xanes3D_reg_params_configured & self.xanes3D_reg_done &
              self.xanes3D_reg_review_done)):
            boxes = ['align_recon_title_box']
            self.enable_disable_boxes(boxes, disabled=False, level=-1)
            self.hs['L[0][2][2][2][1][0]_align_text'].disabled = True
                    
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
 
        self.hs['L[0]_top_tab_form'].children = self.get_handles('L[0]_top_tab_form', -1)       
        self.hs['L[0]_top_tab_form'].set_title(0, 'TOMO RECON')
        self.hs['L[0]_top_tab_form'].set_title(1, '2D XANES')
        self.hs['L[0]_top_tab_form'].set_title(2, '3D XANES')
        
        ## ## ## define functional tabs for each sub-tab -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-86}px', 'height':f'{self.form_sz[0]-128}px'}
        self.hs['L[0][2][0]_config_input_form'] = widgets.VBox()
        self.hs['L[0][2][1]_reg_setting_form'] = widgets.VBox() 
        self.hs['L[0][2][2]_reg&review_form'] = widgets.VBox()               
        self.hs['L[0][2][0]_config_input_form'].layout = layout
        self.hs['L[0][2][1]_reg_setting_form'].layout = layout
        self.hs['L[0][2][2]_reg&review_form'].layout = layout


        
        ## ## ## ## define functional widgets each tab in each sub-tab - configure file settings -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.32*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][0][0]_select_file&path_box'] = widgets.VBox()
        self.hs['L[0][2][0][0]_select_file&path_box'].layout = layout
        
        ## ## ## ## ## label configure file settings box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.32*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][0][0][0]_select_file&path_title_box'] = widgets.HBox()
        self.hs['L[0][2][0][0][0]_select_file&path_title_box'].layout = layout  
        self.hs['L[0][2][0][0][0][0]_select_file&path_title_box'] = widgets.Text(value='Config Files', disabled=True)
        self.hs['L[0][2][0][0][0][0]_select_file&path_title_box'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Config Dirs & Files' + '</span>')
        layout = {'background-color':'white', 'color':'cyan', 'left':'39%'}        
        self.hs['L[0][2][0][0][0][0]_select_file&path_title_box'].layout = layout
        self.hs['L[0][2][0][0][0]_select_file&path_title_box'].children = self.get_handles('L[0][2][0][0][0]_select_file&path_title_box', -1)
                
        ## ## ## ## ## raw h5 top directory
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.32*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][0][0][1]_select_raw_box'] = widgets.HBox()
        self.hs['L[0][2][0][0][1]_select_raw_box'].layout = layout        
        self.hs['L[0][2][0][0][1][1]_select_raw_h5_path_text'] = widgets.Text(value='Choose raw h5 directory ...', description='', disabled=True)
        layout = {'width':'66%'}
        self.hs['L[0][2][0][0][1][1]_select_raw_h5_path_text'].layout = layout
        self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'] = SelectFilesButton(option='askdirectory', 
                                                                                    text_h=self.hs['L[0][2][0][0][1][1]_select_raw_h5_path_text'])
        layout = {'width':'15%'}
        self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].layout = layout
        self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].on_click(self.L0_2_0_0_1_0_select_raw_h5_path_button_click)
        self.hs['L[0][2][0][0][1]_select_raw_box'].children = self.get_handles('L[0][2][0][0][1]_select_raw_box', -1) 
        
        ## ## ## ## ## recon top directory
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.32*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][0][0][2]_select_recon_box'] = widgets.HBox()
        self.hs['L[0][2][0][0][2]_select_recon_box'].layout = layout        
        self.hs['L[0][2][0][0][2][1]_select_recon_path_text'] = widgets.Text(value='--Choose recon top directory ...', description='', disabled=True)
        layout = {'width':'66%'}
        self.hs['L[0][2][0][0][2][1]_select_recon_path_text'].layout = layout
        self.hs['L[0][2][0][0][2][0]_select_recon_path_button'] = SelectFilesButton(option='askdirectory', 
                                                                                    text_h=self.hs['L[0][2][0][0][2][1]_select_recon_path_text'])
        layout = {'width':'15%'}
        self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].layout = layout
        self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].on_click(self.L0_2_0_0_2_0_select_recon_path_button_click)
        self.hs['L[0][2][0][0][2]_select_recon_box'].children = self.get_handles('L[0][2][0][0][2]_select_recon_box', -1) 
        
        ## ## ## ## ## trial save file
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.32*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][0][0][3]_select_save_trial_box'] = widgets.HBox()
        self.hs['L[0][2][0][0][3]_select_save_trial_box'].layout = layout        
        self.hs['L[0][2][0][0][3][1]_select_save_trial_text'] = widgets.Text(value='Save trial registration as ...', description='', disabled=True)
        layout = {'width':'66%'}
        self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].layout = layout
        self.hs['L[0][2][0][0][3][0]_select_save_trial_button'] = SelectFilesButton(option='asksaveasfilename', 
                                                                                    text_h=self.hs['L[0][2][0][0][3][1]_select_save_trial_text'])
        layout = {'width':'15%'}
        self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].layout = layout
        self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].on_click(self.L0_2_0_0_3_0_select_save_trial_button_click)
        self.hs['L[0][2][0][0][3]_select_save_trial_box'].children = self.get_handles('L[0][2][0][0][3]_select_save_trial_box', -1) 
        
        ## ## ## ## ## confirm file configuration
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.32*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][0][0][4]_select_file&path_title_comfirm_box'] = widgets.HBox()
        self.hs['L[0][2][0][0][4]_select_file&path_title_comfirm_box'].layout = layout        
        self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'] = widgets.Text(value='Save trial registration, or go directly review registration ...', description='', disabled=True)
        layout = {'width':'66%'}
        self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].layout = layout
        self.hs['L[0][2][0][0][4][0]_confirm_file&path_button'] = widgets.Button(description='Confirm', 
                                                                                description_tooltip='Confirm: Confirm after you finish file configuration')
        self.hs['L[0][2][0][0][4][0]_confirm_file&path_button'].on_click(self.L0_2_0_0_4_0_confirm_file_path_button_click)
        layout = {'width':'15%'}
        self.hs['L[0][2][0][0][4][0]_confirm_file&path_button'].layout = layout
        self.hs['L[0][2][0][0][4][2]_review_reg_result_checkbox'] = widgets.Checkbox(value=False, description='',
                                                                                    description_tooltip='Check this if registration is completed, and you want to review the registration results directly.')
        layout = {'width':'15%', 'top':'20%'}
        self.hs['L[0][2][0][0][4][2]_review_reg_result_checkbox'].layout = layout
        self.hs['L[0][2][0][0][4][2]_review_reg_result_checkbox'].observe(self.L0_2_0_0_4_2_review_reg_result_checkbox_change, names='value')
        self.hs['L[0][2][0][0][4]_select_file&path_title_comfirm_box'].children = self.get_handles('L[0][2][0][0][4]_select_file&path_title_comfirm_box', -1) 
        
        self.hs['L[0][2][0][0]_select_file&path_box'].children = self.get_handles('L[0][2][0][0]_select_file&path_box', -1) 
        ## ## ## ## bin widgets in hs['L[0][2][0][0]_select_file&path_box'] -- configure file settings -- end
 
       
        
        ## ## ## ## define functional widgets each tab in each sub-tab  - define indices -- start    
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.32*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][0][1]_config_indices_box'] = widgets.VBox()
        self.hs['L[0][2][0][1]_config_indices_box'].layout = layout
        ## ## ## ## ## label define indices box
        layout = {'justify-content':'center', 'align-items':'center','align-contents':'center','border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.32*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][0][1][0]_config_indices_title_box'] = widgets.HBox()
        self.hs['L[0][2][0][1][0]_config_indices_title_box'].layout = layout  
        self.hs['L[0][2][0][1][0][0]_config_indices_title'] = widgets.Text(value='Config Files', disabled=True)
        self.hs['L[0][2][0][1][0][0]_config_indices_title'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Config Scan & Slice Indices' + '</span>')
        layout = {'left':'35%', 'background-color':'white', 'color':'cyan'}        
        self.hs['L[0][2][0][1][0][0]_config_indices_title'].layout = layout
        self.hs['L[0][2][0][1][0]_config_indices_title_box'].children = self.get_handles('L[0][2][0][1][0]_select_file&path_title_box', -1)

        ## ## ## ## ## scan id range
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.32*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][0][1][1]_scan_id_range_box'] = widgets.HBox()
        self.hs['L[0][2][0][1][1]_scan_id_range_box'].layout = layout        
        self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'] = widgets.BoundedIntText(value=0, description='scan_id start', disabled=True)
        layout = {'width':'40%'}
        self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].layout = layout
        self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'] = widgets.BoundedIntText(value=0, description='scan_id end', disabled=True)
        layout = {'width':'40%'}
        self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].layout = layout
        self.hs['L[0][2][0][1][1]_scan_id_range_box'].children = self.get_handles('L[0][2][0][1][1]_scan_id_range_box', -1) 
        self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].observe(self.L0_2_0_1_1_0_select_scan_id_start_text_change, names='value')
        self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].observe(self.L0_2_0_1_1_1_select_scan_id_end_text_change, names='value')

        ## ## ## ## ## fixed scan and slice ids
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.32*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][0][1][2]_fixed_id_box'] = widgets.HBox()
        self.hs['L[0][2][0][1][2]_fixed_id_box'].layout = layout        
        self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'] = widgets.IntSlider(value=0, description='fixed scan id', disabled=True)
        layout = {'width':'40%'}
        self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].layout = layout
        self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'] = widgets.IntSlider(value=0, description='fixed sli id', disabled=True)
        layout = {'width':'40%'}
        self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].layout = layout
        self.hs['L[0][2][0][1][2]_fixed_id_box'].children = self.get_handles('L[0][2][0][1][2]_fixed_id_box', -1) 
        self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].observe(self.L0_2_0_1_2_0_fixed_scan_id_slider_change, names='value')
        self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].observe(self.L0_2_0_1_2_1_fixed_sli_id_slider_change, names='value')
        
        ## ## ## ## ## fiji option
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.32*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][0][1][3]_fiji_box'] = widgets.HBox()
        self.hs['L[0][2][0][1][3]_fiji_box'].layout = layout        
        self.hs['L[0][2][0][1][3][0]_fiji_virtural_stack_preview_checkbox'] = widgets.Checkbox(value=False, description='fiji view', disabled=True)
        layout = {'width':'40%'}
        self.hs['L[0][2][0][1][3][0]_fiji_virtural_stack_preview_checkbox'].layout = layout
        self.hs['L[0][2][0][1][3][1]_fiji_close_button'] = widgets.Button(description='close all fiji viewers', disabled=True)
        layout = {'width':'40%'}
        self.hs['L[0][2][0][1][3][1]_fiji_close_button'].layout = layout
        self.hs['L[0][2][0][1][3]_fiji_box'].children = self.get_handles('L[0][2][0][1][3]_fiji_box', -1) 
        self.hs['L[0][2][0][1][3][0]_fiji_virtural_stack_preview_checkbox'].observe(self.L0_2_0_1_3_0_fiji_virtural_stack_preview_checkbox_change, names='value')
        self.hs['L[0][2][0][1][3][1]_fiji_close_button'].on_click(self.L0_2_0_1_3_1_fiji_close_button_click)
        
        ## ## ## ## ## confirm indices configuration
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.32*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][0][1][4]_config_indices_confirm_box'] = widgets.HBox()
        self.hs['L[0][2][0][1][4]_config_indices_confirm_box'].layout = layout        
        self.hs['L[0][2][0][1][4][1]_confirm_config_indices_text'] = widgets.Text(value='Confirm setting once you are done ...', description='', disabled=True)
        layout = {'width':'66%'}
        self.hs['L[0][2][0][1][4][1]_confirm_config_indices_text'].layout = layout
        self.hs['L[0][2][0][1][4][0]_confirm_config_indices_button'] = widgets.Button(description='Confirm', 
                                                                                description_tooltip='Confirm: Confirm after you finish file configuration')
        layout = {'width':'15%'}
        self.hs['L[0][2][0][1][4][0]_confirm_config_indices_button'].layout = layout
        self.hs['L[0][2][0][1][4]_config_indices_confirm_box'].children = self.get_handles('L[0][2][0][1][4]_config_indices_confirm_box', -1) 
        self.hs['L[0][2][0][1][4][0]_confirm_config_indices_button'].on_click(self.L0_2_0_1_4_0_confirm_config_indices_button_click)
        
        self.hs['L[0][2][0][1]_config_indices_box'].children = self.get_handles('L[0][2][0][1]_config_indices_box', -1) 
        ## ## ## ## bin widgets in hs['L[0][2][0][0]_select_file&path_box']  - define indices -- end
        
        self.hs['L[0][2][0]_config_input_form'].children = self.get_handles('L[0][2][0]_config_input_form', -1)
        ## ## ## bin boxes in hs['L[0][2][0]_config_input_form'] -- end
        
        
        
        ## ## ## ## define functional widgets each tab in each sub-tab - config roi -- start        
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.4*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][1][0]_3D_roi_box'] = widgets.VBox()
        self.hs['L[0][2][1][0]_3D_roi_box'].layout = layout
        ## ## ## ## ## label 3D_roi_title box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}        
        self.hs['L[0][2][1][0][0]_3D_roi_title_box'] = widgets.HBox()
        self.hs['L[0][2][1][0][0]_3D_roi_title_box'].layout = layout 
        self.hs['L[0][2][1][0][0][0]_3D_roi_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Config 3D ROI' + '</span>')
        layout = {'justify-content':'center', 'background-color':'white', 'color':'cyan', 'left':'43%'}        
        self.hs['L[0][2][1][0][0][0]_3D_roi_title_text'].layout = layout
        self.hs['L[0][2][1][0][0]_3D_roi_title_box'].children = self.get_handles('L[0][2][1][0][0]_3D_roi_title_box', -1)
                
        ## ## ## ## ## define roi
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.6*0.4*(self.form_sz[0]-128)}px'}        
        self.hs['L[0][2][1][0][1]_3D_roi_define_box'] = widgets.VBox()
        self.hs['L[0][2][1][0][1]_3D_roi_define_box'].layout = layout 
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-108}px', 'height':f'{0.6*0.4*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'] = widgets.IntRangeSlider(value=[10, 40],
                                                                                min=1,
                                                                                max=50,
                                                                                step=1,
                                                                                description='x range:',
                                                                                disabled=True,
                                                                                continuous_update=False,
                                                                                orientation='horizontal',
                                                                                readout=True,
                                                                                readout_format='d')
        self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].layout = layout   
        self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'] = widgets.IntRangeSlider(value=[10, 40],
                                                                                min=1,
                                                                                max=50,
                                                                                step=1,
                                                                                description='y range:',
                                                                                disabled=True,
                                                                                continuous_update=False,
                                                                                orientation='horizontal',
                                                                                readout=True,
                                                                                readout_format='d')
        self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].layout = layout  
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'] = widgets.IntRangeSlider(value=[10, 40],
                                                                                min=1,
                                                                                max=50,
                                                                                step=1,
                                                                                description='z range:',
                                                                                disabled=True,
                                                                                continuous_update=False,
                                                                                orientation='horizontal',
                                                                                readout=True,
                                                                                readout_format='d')
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].layout = layout  
        self.hs['L[0][2][1][0][1]_3D_roi_define_box'].children = self.get_handles('L[0][2][1][0][1]_3D_roi_define_box', -1)
        self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].observe(self.L0_2_1_0_1_0_3D_roi_x_slider_change, names='value')
        self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].observe(self.L0_2_1_0_1_1_3D_roi_y_slider_change, names='value')
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].observe(self.L0_2_1_0_1_2_3D_roi_z_slider_change, names='value')
        
        ## ## ## ## ## confirm roi
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}          
        self.hs['L[0][2][1][0][2]_3D_roi_confirm_box'] = widgets.HBox()
        self.hs['L[0][2][1][0][2]_3D_roi_confirm_box'].layout = layout 
        layout = {'top': '5px','width':'70%', 'height':'100%'}
        self.hs['L[0][2][1][0][2][0]_confirm_roi_text'] = widgets.Text(description='',
                                                                   value='Please confirm after ROI is set ...',
                                                                   disabled=True)
        self.hs['L[0][2][1][0][2][0]_confirm_roi_text'].layout = layout 
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][2][1][0][2][1]_confirm_roi_button'] = widgets.Button(description='Confirm',
                                                                       description_tooltip='Confirm the roi once you define the ROI ...',
                                                                       disabled=True)
        self.hs['L[0][2][1][0][2][1]_confirm_roi_button'].layout = layout 
        self.hs['L[0][2][1][0][2]_3D_roi_confirm_box'].children = self.get_handles('L[0][2][1][0][2]_3D_roi_confirm_box', -1)
        self.hs['L[0][2][1][0][2][1]_confirm_roi_button'].on_click(self.L0_2_1_0_2_1_confirm_roi_button_click)
        
        self.hs['L[0][2][1][0]_3D_roi_box'].children = self.get_handles('L[0][2][1][0]_3D_roi_box', -1)
        ## ## ## ## define functional widgets each tab in each sub-tab - config roi -- end
               
        ## ## ## ## define functional widgets each tab in each sub-tab - config registration -- start        
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.4*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][1][1]_config_reg_params_box'] = widgets.VBox()
        self.hs['L[0][2][1][1]_config_reg_params_box'].layout = layout
        ## ## ## ## ## label config_reg_params box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}        
        self.hs['L[0][2][1][1][0]_config_reg_params_title_box'] = widgets.HBox()
        self.hs['L[0][2][1][1][0]_config_reg_params_title_box'].layout = layout 
        self.hs['L[0][2][1][1][0][0]_config_reg_params_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Config Reg Params' + '</span>')
        layout = {'background-color':'white', 'color':'cyan', 'left':'40.5%'}        
        self.hs['L[0][2][1][1][0][0]_config_reg_params_title_text'].layout = layout
        self.hs['L[0][2][1][1][0]_config_reg_params_title_box'].children = self.get_handles('L[0][2][1][1][0]_config_reg_params_title_box', -1)

        ## ## ## ## ## fiji&anchor box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}        
        self.hs['L[0][2][1][1][1]_fiji&anchor_box'] = widgets.HBox()
        self.hs['L[0][2][1][1][1]_fiji&anchor_box'].layout = layout 
        self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'] = widgets.Checkbox(value=False,
                                                                        disabled=True,
                                                                        description='preview mask in fiji')
        layout = {'width':'40%'}        
        self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].layout = layout
        self.hs['L[0][2][1][1][1][1]_anchor_checkbox'] = widgets.Checkbox(value=True,
                                                                          disabled=True,
                                                                          description='use anchor')
        layout = {'width':'40%'}        
        self.hs['L[0][2][1][1][1][1]_anchor_checkbox'].layout = layout
        self.hs['L[0][2][1][1][1]_fiji&anchor_box'].children = self.get_handles('L[0][2][1][1][1]_fiji&anchor_box', -1)
        self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].observe(self.L0_2_1_1_1_0_fiji_mask_viewer_checkbox_change, names='value')
        self.hs['L[0][2][1][1][1][1]_anchor_checkbox'].observe(self.L0_2_1_1_1_1_anchor_checkbox_change, names='value')

        ## ## ## ## ## mask options box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}        
        self.hs['L[0][2][1][1][2]_mask_options_box'] = widgets.HBox()
        self.hs['L[0][2][1][1][2]_mask_options_box'].layout = layout 
        self.hs['L[0][2][1][1][2][0]_use_mask_checkbox'] = widgets.Checkbox(value=True,
                                                                        disabled=True,
                                                                        description='use mask',
                                                                        display='flex',)
        layout = {'width':'15%', 'flex-direction':'row'}        
        self.hs['L[0][2][1][1][2][0]_use_mask_checkbox'].layout = layout
        self.hs['L[0][2][1][1][2][1]_mask_thres_slider'] = widgets.FloatSlider(value=False,
                                                                          disabled=True,
                                                                          description='mask thres',
                                                                          readout_format='.5f',
                                                                          min=-1.,
                                                                          max=1.,
                                                                          step=1e-5)
        layout = {'width':'40%', 'left':'2.5%'}        
        self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].layout = layout
        self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'] = widgets.IntSlider(value=False,
                                                                          disabled=True,
                                                                          description='mask dilation',
                                                                          min=0,
                                                                          max=30,
                                                                          step=1)
        layout = {'width':'40%', 'left':'2.5%'}        
        self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].layout = layout
        self.hs['L[0][2][1][1][2]_mask_options_box'].children = self.get_handles('L[0][2][1][1][2]_mask_options_box', -1)
        self.hs['L[0][2][1][1][2][0]_use_mask_checkbox'].observe(self.L0_2_1_1_2_0_use_mask_checkbox_change, names='value')
        self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].observe(self.L0_2_1_1_2_1_mask_thres_slider_change, names='value')
        self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].observe(self.L0_2_1_1_2_2_mask_dilation_slider_change, names='value')

        ## ## ## ## ## sli_search & chunk_size box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}        
        self.hs['L[0][2][1][1][3]_sli_search&chunk_size_box'] = widgets.HBox()
        self.hs['L[0][2][1][1][3]_sli_search&chunk_size_box'].layout = layout 
        self.hs['L[0][2][1][1][3][0]_sli_search_slider'] = widgets.IntSlider(value=10,
                                                                             disabled=True,
                                                                             description='z search half width')
        layout = {'width':'40%'}        
        self.hs['L[0][2][1][1][3][0]_sli_search_slider'].layout = layout
        self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'] = widgets.IntSlider(value=7,
                                                                           disabled=True,
                                                                           description='chunk size')
        layout = {'width':'40%'}        
        self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].layout = layout
        self.hs['L[0][2][1][1][3]_sli_search&chunk_size_box'].children = self.get_handles('L[0][2][1][1][3]_sli_search&chunk_size_box', -1)
        self.hs['L[0][2][1][1][3][0]_sli_search_slider'].observe(self.L0_2_1_1_3_0_sli_search_slider_change, names='value')
        self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].observe(self.L0_2_1_1_3_1_chunk_sz_slider_change, names='value')

        ## ## ## ## ##  reg_options box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}        
        self.hs['L[0][2][1][1][4]_reg_options_box'] = widgets.HBox()
        self.hs['L[0][2][1][1][4]_reg_options_box'].layout = layout 
        self.hs['L[0][2][1][1][4][0]_reg_method_dropdown'] = widgets.Dropdown(value='MPC',
                                                                              options=['MPC', 'PC', 'SR'],
                                                                              description='reg method',
                                                                              description_tooltip='reg method: MPC: Masked Phase Correlation; PC: Phase Correlation; SR: StackReg',
                                                                              disabled=True)
        layout = {'width':'40%'}        
        self.hs['L[0][2][1][1][4][0]_reg_method_dropdown'].layout = layout
        self.hs['L[0][2][1][1][4][1]_ref_mode_dropdown'] = widgets.Dropdown(value='single',
                                                                            options=['single', 'neighbor', 'average'],
                                                                            description='ref mode',
                                                                            description_tooltip='ref mode: single: two images with fixed id gap in two consecutive chunks are used for the registration between two chunks; neighbor: two neighbor images in two consecutive chunks are used for the registration between two chunks; average: deprecated',
                                                                            disabled=True)
        layout = {'width':'40%'}        
        self.hs['L[0][2][1][1][4][1]_ref_mode_dropdown'].layout = layout
        self.hs['L[0][2][1][1][4]_reg_options_box'].children = self.get_handles('L[0][2][1][1][4]_reg_options_box', -1)
        self.hs['L[0][2][1][1][4][0]_reg_method_dropdown'].observe(self.L0_2_1_1_4_0_reg_method_dropdown_change, names='value')
        self.hs['L[0][2][1][1][4][1]_ref_mode_dropdown'].observe(self.L0_2_1_1_4_1_ref_mode_dropdown_change, names='value')

        ## ## ## ## ## confirm reg settings
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}          
        self.hs['L[0][2][1][1][5]_config_reg_params_confirm_box'] = widgets.HBox()
        self.hs['L[0][2][1][1][5]_config_reg_params_confirm_box'].layout = layout 
        layout = {'top': '5px','width':'70%', 'height':'100%'}
        self.hs['L[0][2][1][1][5][0]_confirm_reg_params_text'] = widgets.Text(description='',
                                                                   value='Confirm the roi once you define the ROI ...',
                                                                   disabled=True)
        self.hs['L[0][2][1][1][5][0]_confirm_reg_params_text'].layout = layout 
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][2][1][1][5][1]_confirm_reg_params_button'] = widgets.Button(description='Confirm',
                                                                       description_tooltip='Confirm the roi once you define the ROI ...',
                                                                       disabled=True)
        self.hs['L[0][2][1][1][5][1]_confirm_reg_params_button'].layout = layout 
        self.hs['L[0][2][1][1][5]_config_reg_params_confirm_box'].children = self.get_handles('L[0][2][1][1][5]_config_reg_params_confirm_box', -1)
        self.hs['L[0][2][1][1][5][1]_confirm_reg_params_button'].on_click(self.L0_2_1_1_5_1_confirm_reg_params_button_click)
        ## ## ## ## define functional widgets each tab in each sub-tab - config registration -- end

        self.hs['L[0][2][1][1]_config_reg_params_box'].children = self.get_handles('L[0][2][1][1]_config_reg_params_box', -1)
        self.hs['L[0][2][1]_reg_setting_form'].children = self.get_handles('L[0][2][1]_reg_setting_form', -1)
        ## ## ## bin boxes in hs['L[0][2][1]_reg_setting_form'] -- end

        ## ## ## ## define functional widgets each tab in each sub-tab - register in register/review/shift TAB -- start        
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.25*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][2][0]_run_reg_box'] = widgets.VBox()
        self.hs['L[0][2][2][0]_run_reg_box'].layout = layout
        ## ## ## ## ## label run_reg box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}        
        self.hs['L[0][2][2][0][0]_run_reg_title_box'] = widgets.HBox()
        self.hs['L[0][2][2][0][0]_run_reg_title_box'].layout = layout 
        self.hs['L[0][2][2][0][0][0]_run_reg_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Run Registration' + '</span>')
        layout = {'background-color':'white', 'color':'cyan', 'left':'41%'}        
        self.hs['L[0][2][2][0][0][0]_run_reg_title_text'].layout = layout
        self.hs['L[0][2][2][0][0]_run_reg_title_box'].children = self.get_handles('L[0][2][2][0][0]_run_reg_title_box', -1)

        ## ## ## ## ## run reg & status
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}          
        self.hs['L[0][2][2][0][1]_run_reg_confirm_box'] = widgets.HBox()
        self.hs['L[0][2][2][0][1]_run_reg_confirm_box'].layout = layout 
        layout = {'top': '5px','width':'70%', 'height':'100%'}
        self.hs['L[0][2][2][0][1][1]_run_reg_text'] = widgets.Text(description='',
                                                                   value='run registration once you are ready ...',
                                                                   disabled=True)
        self.hs['L[0][2][2][0][1][1]_run_reg_text'].layout = layout 
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][2][2][0][1][0]_run_reg_button'] = widgets.Button(description='Run Reg',
                                                                       description_tooltip='Confirm the roi once you define the ROI ...',
                                                                       disabled=True)
        self.hs['L[0][2][2][0][1][0]_run_reg_button'].layout = layout 
        self.hs['L[0][2][2][0][1]_run_reg_confirm_box'].children = self.get_handles('L[0][2][2][0][1]_run_reg_confirm_box', -1)
        self.hs['L[0][2][2][0][1][0]_run_reg_button'].on_click(self.L0_2_2_0_1_0_run_reg_button_click)
        ## ## ## ## ## run reg progress
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}          
        self.hs['L[0][2][2][0][2]_run_reg_progress_box'] = widgets.HBox()
        self.hs['L[0][2][2][0][2]_run_reg_progress_box'].layout = layout 
        layout = {'top': '5px','width':'100%', 'height':'100%'}
        self.hs['L[0][2][2][0][2][0]_run_reg_progress_bar'] = widgets.IntProgress(value=0,
                                                                                  min=0,
                                                                                  max=10,
                                                                                  step=1,
                                                                                  description='Completing:',
                                                                                  bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                                                                  orientation='horizontal')
        self.hs['L[0][2][2][0][2][0]_run_reg_progress_bar'].layout = layout 
        self.hs['L[0][2][2][0][2]_run_reg_progress_box'].children = self.get_handles('L[0][2][2][0][2]_run_reg_progress_box', -1)

        self.hs['L[0][2][2][0]_run_reg_box'].children = self.get_handles('L[0][2][2][0]_run_reg_box', -1)
        ## ## ## ## define functional widgets each tab in each sub-tab - register in register/review/shift TAB -- end
        
        ## ## ## ## define functional widgets each tab in each sub-tab - review in in register/review/shift TAB -- start        
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.4*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][2][1]_review_reg_results_box'] = widgets.VBox()
        self.hs['L[0][2][2][1]_review_reg_results_box'].layout = layout
        ## ## ## ## ## label the box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}        
        self.hs['L[0][2][2][1][0]_review_reg_results_title_box'] = widgets.HBox()
        self.hs['L[0][2][2][1][0]_review_reg_results_title_box'].layout = layout 
        self.hs['L[0][2][2][1][0][0]_review_reg_results_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Review Registration Results' + '</span>')
        layout = {'background-color':'white', 'color':'cyan', 'left':'35.7%'}        
        self.hs['L[0][2][2][1][0][0]_review_reg_results_title_text'].layout = layout
        self.hs['L[0][2][2][1][0]_review_reg_results_title_box'].children = self.get_handles('L[0][2][2][1][0]_review_reg_results_title_box', -1)

        ## ## ## ## ## read alignment file
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}          
        self.hs['L[0][2][2][1][1]_read_alignment_file_box'] = widgets.HBox()
        self.hs['L[0][2][2][1][1]_read_alignment_file_box'].layout = layout 
        layout = {'top': '5px','width':'70%', 'height':'100%'}
        self.hs['L[0][2][2][1][1][1]_read_alignment_checkbox'] = widgets.Checkbox(description='read alignment',
                                                                                  value=False,
                                                                                  disabled=True)
        self.hs['L[0][2][2][1][1][1]_read_alignment_checkbox'].layout = layout 
        layout = {'width':'30%', 'height':'90%'}
        self.hs['L[0][2][2][1][1][0]_read_alignment_button'] = SelectFilesButton(option='askopenfilename')
        self.hs['L[0][2][2][1][1][0]_read_alignment_button'].layout = layout 
        self.hs['L[0][2][2][1][1]_read_alignment_file_box'].children = self.get_handles('L[0][2][2][1][1]_read_alignment_file_box', -1)
        self.hs['L[0][2][2][1][1][1]_read_alignment_checkbox'].observe(self.L0_2_2_1_1_0_read_alignment_checkbox_change, names='value')
        self.hs['L[0][2][2][1][1][0]_read_alignment_button'].on_click(self.L0_2_2_1_1_1_read_alignment_button_click)
        
        ## ## ## ## ## reg pair box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}          
        self.hs['L[0][2][2][1][2]_reg_pair_box'] = widgets.HBox()
        self.hs['L[0][2][2][1][2]_reg_pair_box'].layout = layout 
        layout = {'top': '5px','width':'70%', 'height':'100%'}
        self.hs['L[0][2][2][1][2][0]_reg_pair_slider'] = widgets.IntSlider(value=False,
                                                                           disabled=True,
                                                                           description='reg pair #')
        self.hs['L[0][2][2][1][2][0]_reg_pair_slider'].layout = layout 
        self.hs['L[0][2][2][1][2]_reg_pair_box'].children = self.get_handles('L[0][2][2][1][2]_reg_pair_box', -1)
        self.hs['L[0][2][2][1][2][0]_reg_pair_slider'].observe(self.L0_2_2_1_2_0_reg_pair_slider_change, names='value')
        
        ## ## ## ## ## zshift box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}          
        self.hs['L[0][2][2][1][3]_zshift_box'] = widgets.HBox()
        self.hs['L[0][2][2][1][3]_zshift_box'].layout = layout 
        layout = {'top': '5px','width':'70%', 'height':'100%'}
        self.hs['L[0][2][2][1][3][0]_zshift_slider'] = widgets.IntSlider(value=False,
                                                                         disabled=True,
                                                                         description='z shift')
        self.hs['L[0][2][2][1][3][0]_zshift_slider'].layout = layout 
        layout = {'top': '5px','width':'70%', 'height':'100%'}
        self.hs['L[0][2][2][1][3][1]_best_match_text'] = widgets.IntText(value=0,
                                                                      disabled=True,
                                                                      description='Best Match')
        self.hs['L[0][2][2][1][3][1]_best_match_text'].layout = layout 
        layout = {'width':'30%', 'height':'90%'}
        self.hs['L[0][2][2][1][3][2]_record_button'] = widgets.Button(description='Record',
                                                                       description_tooltip='Record',
                                                                       disabled=True)
        self.hs['L[0][2][2][1][3][2]_record_button'].layout = layout 
        self.hs['L[0][2][2][1][3]_zshift_box'].children = self.get_handles('L[0][2][2][1][3]_zshift_box', -1)
        self.hs['L[0][2][2][1][3][0]_zshift_slider'].observe(self.L0_2_2_1_3_0_zshift_slider_change, names='value')
        self.hs['L[0][2][2][1][3][1]_best_match_text'].observe(self.L0_2_2_1_3_1_best_match_text_change, names='value')
        self.hs['L[0][2][2][1][3][2]_record_button'].on_click(self.L0_2_2_1_3_2_record_button_click)
        widgets.jslink((self.hs['L[0][2][2][1][3][0]_zshift_slider'], 'value'),
                       (self.hs['L[0][2][2][1][3][1]_best_match_text'], 'value'))
        
        ## ## ## ## ## confirm review results box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}          
        self.hs['L[0][2][2][1][4]_review_reg_results_comfirm_box'] = widgets.HBox()
        self.hs['L[0][2][2][1][4]_review_reg_results_comfirm_box'].layout = layout 
        layout = {'top': '5px','width':'70%', 'height':'100%'}
        self.hs['L[0][2][2][1][4][0]_confirm_review_results_text'] = widgets.Text(description='',
                                                                   value='Confirm the roi once you define the ROI ...',
                                                                   disabled=True)
        self.hs['L[0][2][2][1][4][0]_confirm_review_results_text'].layout = layout 
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][2][2][1][4][1]_confirm_review_results_button'] = widgets.Button(description='Confirm',
                                                                       description_tooltip='Confirm the roi once you define the ROI ...',
                                                                       disabled=True)
        self.hs['L[0][2][2][1][4][1]_confirm_review_results_button'].layout = layout 
        self.hs['L[0][2][2][1][4]_review_reg_results_comfirm_box'].children = self.get_handles('L[0][2][2][1][4]_review_reg_results_comfirm_box', -1)
        self.hs['L[0][2][2][1][4][1]_confirm_review_results_button'].on_click(self.L0_2_2_1_4_1_confirm_review_results_button_click)

        self.hs['L[0][2][2][1]_review_reg_results_box'].children = self.get_handles('L[0][2][2][1]_review_reg_results_box', -1)
        ## ## ## ## define functional widgets each tab in each sub-tab - review in in register/review/shift TAB-- end
        
        ## ## ## ## define functional widgets each tab in each sub-tab - align recon in register/review/shift TAB -- start        
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.25*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][2][2]_align_recon_box'] = widgets.VBox()
        self.hs['L[0][2][2][2]_align_recon_box'].layout = layout
        ## ## ## ## ## label the box
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}        
        self.hs['L[0][2][2][2][0]_align_recon_title_box'] = widgets.HBox()
        self.hs['L[0][2][2][2][0]_align_recon_title_box'].layout = layout 
        self.hs['L[0][2][2][2][0][0]_align_recon_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Align 3D Recon' + '</span>')
        layout = {'background-color':'white', 'color':'cyan', 'left':'41%'}        
        self.hs['L[0][2][2][2][0][0]_align_recon_title_text'].layout = layout
        self.hs['L[0][2][2][2][0]_align_recon_title_box'].children = self.get_handles('L[0][2][2][2][0]_align_recon_title_box', -1)

        ## ## ## ## ## define slice region if it is necessary
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}          
        self.hs['L[0][2][2][2][1]_align_recon_optional_slice_region_box'] = widgets.HBox()
        self.hs['L[0][2][2][2][1]_align_recon_optional_slice_region_box'].layout = layout 
        layout = {'width':'15%', 'height':'100%'}
        self.hs['L[0][2][2][2][1][0]_align_recon_optional_slice_start_text'] = widgets.BoundedIntText(description='',
                                                                                          description_tooltip='In the case of reading and reviewing a registration file, you need to define slice start and end.',
                                                                                          value = 0,
                                                                                          min = 0,
                                                                                          max = 10,
                                                                                          disabled=True)
        self.hs['L[0][2][2][2][1][0]_align_recon_optional_slice_start_text'].layout = layout 
        layout = {'width':'60%', 'height':'100%'}
        self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_range_slider'] = widgets.IntRangeSlider(description='z range',
                                                                                          description_tooltip='In the case of reading and reviewing a registration file, you need to define slice start and end.',
                                                                                          value = 0,
                                                                                          min = 0,
                                                                                          max = 10,
                                                                                          disabled=True)
        self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_range_slider'].layout = layout
        layout = {'width':'15%', 'height':'100%'}
        self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_end_text'] = widgets.BoundedIntText(description='',
                                                                                          description_tooltip='In the case of reading and reviewing a registration file, you need to define slice start and end.',
                                                                                          value = 0,
                                                                                          min = 0,
                                                                                          max = 10,
                                                                                          disabled=True)
        self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_end_text'].layout = layout
        self.hs['L[0][2][2][2][1]_align_recon_optional_slice_region_box'].children = self.get_handles('L[0][2][2][2][1]_align_recon_optional_slice_region_box', -1)
        self.hs['L[0][2][2][2][1][0]_align_recon_optional_slice_start_text'].observe(self.L0_2_2_2_1_0_align_recon_optional_slice_start_text, names='value')
        self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_end_text'].observe(self.L0_2_2_2_1_2_align_recon_optional_slice_end_text, names='value')
        self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_range_slider'].observe(self.L0_2_2_2_1_1_align_recon_optional_slice_range_slider, names='value')
        
        ## ## ## ## ## run reg & status
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}          
        self.hs['L[0][2][2][2][2]_align_recon_comfirm_box'] = widgets.HBox()
        self.hs['L[0][2][2][2][2]_align_recon_comfirm_box'].layout = layout 
        layout = {'top': '5px','width':'70%', 'height':'100%'}
        self.hs['L[0][2][2][2][2][0]_align_text'] = widgets.Text(description='',
                                                                   value='Confirm the roi once you define the ROI ...',
                                                                   disabled=True)
        self.hs['L[0][2][2][2][2][0]_align_text'].layout = layout 
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][2][2][2][2][1]_align_button'] = widgets.Button(description='Align',
                                                                       description_tooltip='This will perform xanes3D alignment according to your configurations ...',
                                                                       disabled=True)
        self.hs['L[0][2][2][2][2][1]_align_button'].layout = layout 
        self.hs['L[0][2][2][2][2]_align_recon_comfirm_box'].children = self.get_handles('L[0][2][2][2][2]_align_recon_comfirm_box', -1)
        self.hs['L[0][2][2][2][2][1]_align_button'].on_click(self.L0_2_2_2_2_1_align_button_click)
        
        ## ## ## ## ## run reg progress
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}          
        self.hs['L[0][2][2][2][3]_align_progress_box'] = widgets.HBox()
        self.hs['L[0][2][2][2][3]_align_progress_box'].layout = layout 
        layout = {'top': '5px','width':'100%', 'height':'100%'}
        self.hs['L[0][2][2][2][3][0]_align_progress_bar'] = widgets.IntProgress(value=0,
                                                                                  min=0,
                                                                                  max=10,
                                                                                  step=1,
                                                                                  description='Completing:',
                                                                                  bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                                                                  orientation='horizontal')
        self.hs['L[0][2][2][2][3][0]_align_progress_bar'].layout = layout 
        self.hs['L[0][2][2][2][3]_align_progress_box'].children = self.get_handles('L[0][2][2][2][3]_align_progress_box', -1)
        

        self.hs['L[0][2][2][2]_align_recon_box'].children = self.get_handles('L[0][2][2][2]_align_recon_box', -1)
        ## ## ## ## define functional widgets each tab in each sub-tab - align recon in register/review/shift TAB -- end
        
        
        
        self.hs['L[0][2][2]_reg&review_form'].children = self.get_handles('L[0][2][2]_reg&review_form', -1)
        ## ## ## bin sub-tabs in each tab - reg&review TAB in 3D_xanes TAB -- end
        


        
        self.hs['L[0][2]_3D_xanes_tabs'].children = self.get_handles('L[0][2]_3D_xanes_tabs', -1)  
        self.hs['L[0][2]_3D_xanes_tabs'].set_title(0, 'File Configurations')
        self.hs['L[0][2]_3D_xanes_tabs'].set_title(1, 'Registration Settings')
        self.hs['L[0][2]_3D_xanes_tabs'].set_title(2, 'Registration & reviews') 
        ## ## bin forms in hs['L[0][2]_3D_xanes_tabs']
            
        display(self.hs['L[0]_top_tab_form'])
        
        
    def L0_2_0_0_1_0_select_raw_h5_path_button_click(self, a):
        if len(a.files[0]) != 0:
            self.xanes3D_raw_3D_h5_top_dir = os.path.abspath(a.files[0])
            self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].initialdir = os.path.abspath(a.files[0])
            self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].initialdir = os.path.abspath(a.files[0])
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialdir = os.path.abspath(a.files[0])
            self.xanes3D_raw_3D_h5_temp = os.path.join(self.xanes3D_raw_3D_h5_top_dir, 'fly_scan_id_{}.h5')
            b = glob.glob(os.path.join(self.xanes3D_raw_3D_h5_top_dir, 'fly*.h5'))
            self.xanes3D_available_raw_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])
            self.xanes3D_raw_h5_path_set = True
        else:
            self.xanes3D_raw_h5_path_set = False
        self.xanes3D_filepath_configured = False
        self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        self.boxes_logic()
        
    def L0_2_0_0_2_0_select_recon_path_button_click(self, a):
        if len(a.files[0]) != 0:
            self.xanes3D_recon_3D_top_dir = os.path.abspath(a.files[0])
            self.xanes3D_recon_3D_dir_temp = os.path.join(self.xanes3D_recon_3D_top_dir,
                                              'recon_fly_scan_id_{0}')
            self.xanes3D_recon_3D_tiff_temp = os.path.join(self.xanes3D_recon_3D_top_dir,
                                                           'recon_fly_scan_id_{0}',
                                                           'recon_fly_scan_id_{0}_{1}.tiff')
            self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].initialdir = os.path.abspath(a.files[0])
            b = glob.glob(os.path.join(self.xanes3D_recon_3D_top_dir, 'recon_fly*'))
            self.xanes3D_available_recon_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])
            self.xanes3D_recon_path_set = True
        else:
            self.xanes3D_recon_path_set = False
        self.xanes3D_filepath_configured = False
        self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        self.boxes_logic()
        
    def L0_2_0_0_3_0_select_save_trial_button_click(self, a):
        if len(a.files[0]) != 0:
            self.xanes3D_save_trial_reg_filename = os.path.abspath(a.files[0])
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialfile = os.path.basename(a.files[0])
            self.xanes3D_save_trial_set = True
        else:
            self.xanes3D_save_trial_set = False
        self.xanes3D_filepath_configured = False
        self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        self.boxes_logic()
            
    def L0_2_0_0_4_2_review_reg_result_checkbox_change(self, a):
        self.xanes3D_filepath_configured = False
        if self.hs['L[0][2][0][0][4][2]_review_reg_result_checkbox'].value:
            self.xanes3D_use_existing_reg_file = True
            self.hs['L[0][2][2][1][1][0]_read_alignment_button'].text_h = self.hs['L[0][2][2][1][4][0]_confirm_review_results_text']
        else:
            self.xanes3D_use_existing_reg_file = False
            self.hs['L[0][2][2][1][1][1]_read_alignment_checkbox'].value = False
        self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        self.boxes_logic()
            
    def L0_2_0_0_4_0_confirm_file_path_button_click(self, a):
        if self.xanes3D_use_existing_reg_file:
            if not self.xanes3D_save_trial_set:
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please specifiy where to read trial reg result ...'
                self.xanes3D_filepath_configured = False
                self.xanes3D_reg_review_ready = False
            else:
                try:
                    f = h5py.File(self.xanes3D_save_trial_reg_filename, 'r')
                    self.trial_reg = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format('000')][:]
                    self.hs['L[0][2][2][1][2][0]_reg_pair_slider'].max = f['/trial_registration/trial_reg_parameters/alignment_pairs'][:].shape[0] - 1
                    self.hs['L[0][2][2][1][2][0]_reg_pair_slider'].min = 0
                    self.hs['L[0][2][2][1][3][0]_zshift_slider'].max = self.trial_reg.shape[0] - 1
                    self.hs['L[0][2][2][1][3][0]_zshift_slider'].min = 0
                    self.xanes3D_recon_3D_tiff_temp = f['/trial_registration/data_directory_info/recon_path_template'][()]
                    self.xanes3D_fixed_scan_id = f['/trial_registration/trial_reg_parameters/fixed_scan_id'][()]
                    self.slice_roi = f['/trial_registration/trial_reg_parameters/slice_roi'][:]
                    f.close()
                
                    b = glob.glob(os.path.join(os.path.dirname(self.xanes3D_recon_3D_tiff_temp.format(self.xanes3D_fixed_scan_id, 0)), 'recon_fly*'))
                    self.xanes3D_available_recon_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])
                    self.xanes3D_recon_path_set = True
                    self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_range_slider'].max = max(self.xanes3D_available_recon_ids)
                    self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_range_slider'].value = (min(self.xanes3D_available_recon_ids),
                                                                                                    min(self.xanes3D_available_recon_ids))
                    self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_range_slider'].min = min(self.xanes3D_available_recon_ids)
                    self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_end_text'].max = max(self.xanes3D_available_recon_ids)
                    self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_end_text'].value = min(self.xanes3D_available_recon_ids)
                    self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_end_text'].min = min(self.xanes3D_available_recon_ids)
                    self.hs['L[0][2][2][2][1][0]_align_recon_optional_slice_start_text'].max = max(self.xanes3D_available_recon_ids)
                    self.hs['L[0][2][2][2][1][0]_align_recon_optional_slice_start_text'].value = min(self.xanes3D_available_recon_ids)
                    self.hs['L[0][2][2][2][1][0]_align_recon_optional_slice_start_text'].min = min(self.xanes3D_available_recon_ids)
                    
                    self.xanes3D_filepath_configured = True
                    self.xanes3D_reg_review_ready = False
                    self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'XANES3D file config is done ...'    
                except:
                    self.xanes3D_filepath_configured = False
                    self.xanes3D_reg_review_ready = False
                    self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Cannot open the specified file. Please check if it exists or is locked ...'                                 
        else:
            if not self.xanes3D_raw_h5_path_set:
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please specifiy raw h5 file location ...'
                self.xanes3D_filepath_configured = False
            elif not self.xanes3D_recon_path_set:
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please specifiy recon top directory location ...'
                self.xanes3D_filepath_configured = False
            elif not self.xanes3D_save_trial_set:
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please specifiy where to save trial reg result ...'
                self.xanes3D_filepath_configured = False
            else:
                self.xanes3D_filepath_configured = True
                self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].max = max(self.xanes3D_available_raw_ids)
                self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].max = max(self.xanes3D_available_raw_ids)
                self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].value = min(self.xanes3D_available_raw_ids) 
                self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].value = min(self.xanes3D_available_raw_ids)
                self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].min = min(self.xanes3D_available_raw_ids)
                self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].min = min(self.xanes3D_available_raw_ids)
                self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].max = max(self.xanes3D_available_raw_ids)
                self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].min = min(self.xanes3D_available_raw_ids)
                self.xanes3D_fixed_scan_id = self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].value
                self.xanes3D_fixed_sli_id = self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].value  
        self.boxes_logic()
    
    def L0_2_0_1_1_0_select_scan_id_start_text_change(self, a):
        if os.path.exists(self.xanes3D_raw_3D_h5_temp.format(a['owner'].value)):
            if os.path.exists(self.xanes3D_recon_3D_dir_temp.format(a['owner'].value)):
                if (a['owner'].value in self.xanes3D_available_raw_ids) and (a['owner'].value in self.xanes3D_available_recon_ids):
                    self.xanes3D_scan_id_s = a['owner'].value
                    self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].min = a['owner'].value
                    self.xanes3D_scan_id_set = True
                    self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].min = a['owner'].value
                    self.hs['L[0][2][0][1][4][1]_confirm_config_indices_text'].value = 'scan_id_s are changed ...'
                else:
                    self.xanes3D_scan_id_set = False
        self.xanes3D_indices_configured = False
        self.boxes_logic()
                    
    def L0_2_0_1_1_1_select_scan_id_end_text_change(self, a):
        ids = np.arange(self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].value, a['owner'].value)
        if (not set(ids).issubset(set(self.xanes3D_available_recon_ids))) or (not set(ids).issubset(set(self.xanes3D_available_raw_ids))):
            self.hs['L[0][2][0][1][4][1]_confirm_config_indices_text'].value = 'The set index range is out of either the available raw or recon dataset ranges ...'
        else:
            self.xanes3D_scan_id_e = a['owner'].value
            self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].max = a['owner'].value
        self.xanes3D_indices_configured = False
        self.boxes_logic()
                
    def L0_2_0_1_2_0_fixed_scan_id_slider_change(self, a):
        self.xanes3D_fixed_scan_id = a['owner'].value
        b = glob.glob(os.path.join(self.xanes3D_recon_3D_dir_temp.format(self.xanes3D_fixed_scan_id),
                                   f'recon_fly_scan_id_{self.xanes3D_fixed_scan_id}_*.tiff'))
        self.xanes3D_available_recon_file_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])
        
        if ((self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].value>min(self.xanes3D_available_recon_file_ids)) &
            (self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].value<max(self.xanes3D_available_recon_file_ids))):
            self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].max = max(self.xanes3D_available_recon_file_ids)
            self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min = min(self.xanes3D_available_recon_file_ids)
        else:
            self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].max = max(self.xanes3D_available_recon_file_ids)
            self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].value = min(self.xanes3D_available_recon_file_ids)
            self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min = min(self.xanes3D_available_recon_file_ids)

        if self.hs['L[0][2][0][1][3][0]_fiji_virtural_stack_preview_checkbox'].value:
            self.fiji_viewer_on(viewer_name='virtural_stack_preview_viewer')
            # if self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['fiji_id'] in WindowManager.getIDList():                
            #     self.fn0 = self.xanes3D_recon_3D_tiff_temp.format(self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].value,
            #                                                           str(self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min).zfill(5))
            #     args = {'directory':self.fn0, 'start':1}
            #     ij.py.run_plugin(" Open VirtualStack", args)
            #     self.xanes3D_fiji_windows['virtural_stack_preview_viewer'] = {'ip':WindowManager.getCurrentImage(),
            #                                                                   'fiji_id':WindowManager.getIDList()[-1]}
            #     self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setSlice(self.xanes3D_fixed_sli_id - self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min + 1) 
            # else:
            #     self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].close()
            #     self.fn0 = self.xanes3D_recon_3D_tiff_temp.format(self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].value,
            #                                                           str(self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min).zfill(5))
            #     args = {'directory':self.fn0, 'start':1}
            #     ij.py.run_plugin(" Open VirtualStack", args)
            #     self.xanes3D_fiji_windows['virtural_stack_preview_viewer'] = {'ip':WindowManager.getCurrentImage(),
            #                                                                   'fiji_id':WindowManager.getIDList()[-1]}
            #     self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setSlice(self.xanes3D_fixed_sli_id - self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min + 1) 
        self.xanes3D_indices_configured = False
        self.boxes_logic()
    
    def L0_2_0_1_2_1_fixed_sli_id_slider_change(self, a):
        if self.hs['L[0][2][0][1][3][0]_fiji_virtural_stack_preview_checkbox'].value:
            if self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['fiji_id'] in WindowManager.getIDList():
                self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setSlice(a['owner'].value-a['owner'].min+1)
                self.xanes3D_fixed_sli_id = a['owner'].value
                self.hs['L[0][2][0][1][4][1]_confirm_config_indices_text'].value = 'fixed slice id is changed ...'
            else:
                self.hs['L[0][2][0][1][4][1]_confirm_config_indices_text'].value = 'Please turn on fiji previewer first ...'
        self.xanes3D_indices_configured = False
        self.boxes_logic()
            
    def L0_2_0_1_3_0_fiji_virtural_stack_preview_checkbox_change(self, a):
        if a['owner'].value:
            self.fiji_viewer_on(viewer_name='virtural_stack_preview_viewer')
            # self.fn0 = self.xanes3D_recon_3D_tiff_temp.format(self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].value,
            #                                                   str(self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min).zfill(5))
            # args = {'directory':self.fn0, 'start':1}
            # ij.py.run_plugin(" Open VirtualStack", args)
            # self.xanes3D_fiji_windows['virtural_stack_preview_viewer'] = {'ip':WindowManager.getCurrentImage(),
            #                                                               'fiji_id':WindowManager.getIDList()[-1]}
            # self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setSlice(self.xanes3D_fixed_sli_id - self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min + 1)              
        else:
            fiji_id_list = WindowManager.getIDList()
            if self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['fiji_id'] in fiji_id_list:
                self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].close()
                self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'] = None
                self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['fiji_id'] = None
            self.xanes3D_indices_configured = False
            self.boxes_logic()
    
    def L0_2_0_1_3_1_fiji_close_button_click(self, a):
        if self.hs['L[0][2][0][1][3][0]_fiji_virtural_stack_preview_checkbox'].value:
            self.hs['L[0][2][0][1][3][0]_fiji_virtural_stack_preview_checkbox'].value = False
        if self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value:
            self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value = False
        try:
            for ii in (WindowManager.getIDList()):
                WindowManager.getImage(ii).close()
        except:
            pass
        self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'] = None
        self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['fiji_id'] = None
        self.xanes3D_fiji_windows['mask_viewer']['ip'] = None
        self.xanes3D_fiji_windows['mask_viewer']['fiji_id'] = None
        self.xanes3D_indices_configured = False
        self.boxes_logic()
    
    def L0_2_0_1_4_0_confirm_config_indices_button_click(self, a):
        if WindowManager.getIDList() is None:
            self.hs['L[0][2][0][1][4][1]_confirm_config_indices_text'].value = 'Please turn on fiji viewer to proceed ...' 
            self.xanes3D_indices_configured = False
        elif self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['fiji_id'] in WindowManager.getIDList():
            self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].max = self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].width
            self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].min = 0
            self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].max = self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].height
            self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].min = 0
            self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].max = self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].max
            self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].min = self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min
            
            try:
                self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].upper = self.xanes3D_fixed_sli_id + 50
            except:
                self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].upper = self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].max
            try:
                self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].lower = self.xanes3D_fixed_sli_id
            except:
                self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].lower = self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].min
            
            self.xanes3D_scan_id_s = self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'] .value
            self.xanes3D_scan_id_e = self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].value
            self.xanes3D_fixed_sli_id = self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].value
            self.xanes3D_fixed_scan_id = self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].value
            self.hs['L[0][2][0][1][4][1]_confirm_config_indices_text'].value = 'Indices configuration is done ...'
            self.hs['L[0][2][1][0][2][0]_confirm_roi_text'].value = 'Please confirm after ROI is set'
            self.xanes3D_indices_configured = True
        else:
            self.hs['L[0][2][0][1][4][1]_confirm_config_indices_text'].value = 'Please turn on fiji viewer to proceed ...' 
            self.xanes3D_indices_configured = False
        self.boxes_logic()
    
    def L0_2_1_0_1_0_3D_roi_x_slider_change(self, a):
        self.xanes3D_roi_configured = False
        self.boxes_logic()
        self.hs['L[0][2][1][0][2][0]_confirm_roi_text'].value = 'Please confirm after ROI is set'
        data_state, viewer_state = self.fiji_viewer_state(viewer_name='virtural_stack_preview_viewer')
        if (not data_state) | (not viewer_state):
            self.fiji_viewer_on(viewer_name='virtural_stack_preview_viewer')
        self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setRoi(self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value[0],
                                           self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value[0],
                                           self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value[1]-self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value[0],
                                           self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value[1]-self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value[0])
    
    def L0_2_1_0_1_1_3D_roi_y_slider_change(self, a):
        self.xanes3D_roi_configured = False
        self.boxes_logic()
        self.hs['L[0][2][1][0][2][0]_confirm_roi_text'].value = 'Please confirm after ROI is set'
        data_state, viewer_state = self.fiji_viewer_state(viewer_name='virtural_stack_preview_viewer')
        if (not data_state) | (not viewer_state):
            self.fiji_viewer_on(viewer_name='virtural_stack_preview_viewer')
        self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setRoi(self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value[0],
                                                                                self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value[0],
                                           self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value[1]-self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value[0],
                                           self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value[1]-self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value[0])
    
    def L0_2_1_0_1_2_3D_roi_z_slider_change(self, a):
        self.xanes3D_roi_configured = False
        if a['owner'].upper < self.xanes3D_fixed_sli_id:
            a['owner'].upper = self.xanes3D_fixed_sli_id
        if a['owner'].lower > self.xanes3D_fixed_sli_id:
            a['owner'].lower = self.xanes3D_fixed_sli_id
        self.boxes_logic()
        self.hs['L[0][2][1][0][2][0]_confirm_roi_text'].value = 'Please confirm after ROI is set'
        data_state, viewer_state = self.fiji_viewer_state(viewer_name='virtural_stack_preview_viewer')
        if (not data_state) | (not viewer_state):
            self.fiji_viewer_on(viewer_name='virtural_stack_preview_viewer')
        self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setSlice(self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].value[0]-self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].min+1)
    
    def L0_2_1_0_2_1_confirm_roi_button_click(self, a):
        self.xanes3D_roi_configured = True
        self.xanes3D_roi = [self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value[0],
                            self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value[1],
                            self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value[0],
                            self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value[1],
                            self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].value[0],
                            self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].value[1]]
        self.hs['L[0][2][1][0][2][0]_confirm_roi_text'].valuee = 'ROI configuration is done ...'
        self.hs['L[0][2][1][1][3][0]_sli_search_slider'].max = min(abs(self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].value[1]-self.xanes3D_fixed_sli_id),
                                                                   abs(self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].value[0]-self.xanes3D_fixed_sli_id))        
        self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value = False
        self.boxes_logic()
    
    def L0_2_1_1_1_0_fiji_mask_viewer_checkbox_change(self, a):
        if a['owner'].value:
            self.fiji_viewer_on(viewer_name='mask_viewer')
            # if self.xanes3D_img is None:
            #     self.xanes3D_img = np.ndarray([self.xanes3D_roi[5]-self.xanes3D_roi[4]+1,
            #                                    self.xanes3D_roi[1]-self.xanes3D_roi[0],
            #                                    self.xanes3D_roi[3]-self.xanes3D_roi[2]])
            #     self.xanes3D_mask = np.ndarray([self.xanes3D_roi[1]-self.xanes3D_roi[0],
            #                                     self.xanes3D_roi[3]-self.xanes3D_roi[2]])
            # elif self.xanes3D_img.shape != (self.xanes3D_roi[5]-self.xanes3D_roi[4]+1,
            #                                 self.xanes3D_roi[1]-self.xanes3D_roi[0],
            #                                 self.xanes3D_roi[3]-self.xanes3D_roi[2]):
            #     self.xanes3D_img = np.ndarray([self.xanes3D_roi[5]-self.xanes3D_roi[4]+1,
            #                                    self.xanes3D_roi[1]-self.xanes3D_roi[0],
            #                                    self.xanes3D_roi[3]-self.xanes3D_roi[2]])
            #     self.xanes3D_mask = np.ndarray([self.xanes3D_roi[1]-self.xanes3D_roi[0],
            #                                     self.xanes3D_roi[3]-self.xanes3D_roi[2]])
                
            # cnt = 0
            # for ii in range(self.xanes3D_roi[4], self.xanes3D_roi[5]+1):
            #     fn = self.xanes3D_recon_3D_tiff_temp.format(self.xanes3D_fixed_scan_id, str(ii).zfill(5))
            #     self.xanes3D_img[cnt, ...] = tifffile.imread(fn)[self.xanes3D_roi[0]:self.xanes3D_roi[1],
            #                                                      self.xanes3D_roi[2]:self.xanes3D_roi[3]]
            #     cnt += 1
                
            # ijui.show(ij.py.to_java(self.xanes3D_img))
            # self.xanes3D_fiji_windows['mask_viewer']['ip'] = WindowManager.getCurrentImage()
            # self.xanes3D_fiji_windows['mask_viewer']['ip'].setSlice(self.xanes3D_fixed_sli_id-self.xanes3D_roi[4])
            # self.xanes3D_fiji_windows['mask_viewer']['fiji_id'] = WindowManager.getIDList()[-1]
            # self.xanes3D_fiji_windows['mask_viewer']['ip'].setTitle('mask preview')
        else:
            try:
                self.xanes3D_fiji_windows['mask_viewer']['ip'].close()
            except:
                pass
            self.xanes3D_fiji_windows['mask_viewer']['ip'] = None
            self.xanes3D_fiji_windows['mask_viewer']['fiji_id'] = None
        self.boxes_logic()
    
    def L0_2_1_1_1_1_anchor_checkbox_change(self, a):
        self.xanes3D_reg_params_configured = False
        if a['owner'].value:
            self.xanes3D_reg_use_anchor = True
        else:
            self.xanes3D_reg_use_anchor = False
        self.boxes_logic()
    
    def L0_2_1_1_2_0_use_mask_checkbox_change(self, a):
        self.xanes3D_reg_params_configured = False
        if self.hs['L[0][2][1][1][2][0]_use_mask_checkbox'].value:
        # if a['owner'].value:
            self.xanes3D_reg_use_mask = True
        else:
            self.xanes3D_reg_use_mask = False
        self.boxes_logic()
    
    def L0_2_1_1_2_1_mask_thres_slider_change(self, a):
        self.xanes3D_reg_params_configured = False
        self.xanes3D_reg_mask_thres = a['owner'].value
        data_state, viewer_state = self.fiji_viewer_state(viewer_name='mask_viewer')
        if (not data_state) | (not viewer_state):
            self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value = False
            self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value = True  

        if self.xanes3D_reg_mask_dilation_width == 0:
            self.xanes3D_mask[:] = (self.xanes3D_img[self.xanes3D_fixed_sli_id-self.xanes3D_roi[4]]>self.xanes3D_reg_mask_thres).astype(np.uint8)[:]
        else:
            self.xanes3D_mask[:] = skm.binary_dilation((self.xanes3D_img[self.xanes3D_fixed_sli_id-self.xanes3D_roi[4]]>self.xanes3D_reg_mask_thres).astype(np.uint8),
                                                       np.ones([self.xanes3D_reg_mask_dilation_width,
                                                                self.xanes3D_reg_mask_dilation_width])).astype(np.uint8)[:]
        self.xanes3D_fiji_windows['mask_viewer']['ip'].setImage(ij.convert().convert(ij.dataset().create(ij.py.to_java(self.xanes3D_img*self.xanes3D_mask)), ImagePlusClass))
        self.xanes3D_fiji_windows['mask_viewer']['ip'].setSlice(self.xanes3D_fixed_sli_id-self.xanes3D_roi[4])
        ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")                
        self.boxes_logic()
    
    def L0_2_1_1_2_2_mask_dilation_slider_change(self, a):
        self.xanes3D_reg_params_configured = False
        data_state, viewer_state = self.fiji_viewer_state(viewer_name='mask_viewer')
        if (not data_state) | (not viewer_state):
            self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value = False
            self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value = True  
        self.xanes3D_reg_mask_dilation_width = a['owner'].value
        if self.xanes3D_reg_mask_dilation_width == 0:
            self.xanes3D_mask[:] = (self.xanes3D_img[self.xanes3D_fixed_sli_id-self.xanes3D_roi[4]]>self.xanes3D_reg_mask_thres).astype(np.uint8)[:]
        else:
            self.xanes3D_mask[:] = skm.binary_dilation((self.xanes3D_img[self.xanes3D_fixed_sli_id-self.xanes3D_roi[4]]>self.xanes3D_reg_mask_thres).astype(np.uint8),
                                                       np.ones([self.xanes3D_reg_mask_dilation_width,
                                                                self.xanes3D_reg_mask_dilation_width])).astype(np.uint8)[:]
        self.xanes3D_fiji_windows['mask_viewer']['ip'].setImage(ij.convert().convert(ij.dataset().create(ij.py.to_java(self.xanes3D_img*self.xanes3D_mask)), ImagePlusClass))
        self.xanes3D_fiji_windows['mask_viewer']['ip'].setSlice(self.xanes3D_fixed_sli_id-self.xanes3D_roi[4])
        ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")                
        self.boxes_logic()
    
    def L0_2_1_1_3_0_sli_search_slider_change(self, a):
        self.xanes3D_reg_params_configured = False
        data_state, viewer_state = self.fiji_viewer_state(viewer_name='mask_viewer')
        if (not data_state) | (not viewer_state):
            self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value = False
            self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value = True  
        self.xanes3D_fiji_windows['mask_viewer']['ip'].setSlice(a['owner'].value)        
        # self.xanes3D_reg_sli_search_half_width = a['owner'].value
        self.boxes_logic()
    
    def L0_2_1_1_3_1_chunk_sz_slider_change(self, a):
        self.xanes3D_reg_params_configured = False
        # self.xanes3D_reg_chunk_sz = a['owner'].value
        self.boxes_logic()
    
    def L0_2_1_1_4_0_reg_method_dropdown_change(self, a):
        self.xanes3D_reg_params_configured = False
        # self.xanes3D_reg_method = a['owner'].value
        self.boxes_logic()
    
    def L0_2_1_1_4_1_ref_mode_dropdown_change(self, a):
        self.xanes3D_reg_params_configured = False
        # self.xanes3D_reg_ref_mode = a['owner'].value
        self.boxes_logic()
    
    def L0_2_1_1_5_1_confirm_reg_params_button_click(self, a):
        self.xanes3D_reg_sli_search_half_width = self.hs['L[0][2][1][1][3][0]_sli_search_slider'].value
        self.xanes3D_reg_chunk_sz = self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].value
        self.xanes3D_reg_method = self.hs['L[0][2][1][1][4][0]_reg_method_dropdown'].value
        self.xanes3D_reg_ref_mode = self.hs['L[0][2][1][1][4][1]_ref_mode_dropdown'].value
        self.xanes3D_reg_mask_dilation_width = self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].value
        self.xanes3D_reg_mask_thres = self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].value
        self.xanes3D_reg_params_configured = self.hs['L[0][2][1][1][1][1]_anchor_checkbox'].value
        self.xanes3D_reg_params_configured = True
        self.boxes_logic()
    
    def L0_2_2_0_1_0_run_reg_button_click(self, a):
        reg = xr.regtools(dtype='3D_XANES', method=self.xanes3D_reg_method, mode='TRANSLATION')
        reg.set_method(self.xanes3D_reg_method)
        reg.set_ref_mode(self.xanes3D_reg_ref_mode)
        reg.cal_set_anchor(self.xanes3D_scan_id_s, self.xanes3D_scan_id_e, self.xanes3D_fixed_scan_id, raw_h5_top_dir=self.xanes3D_raw_3D_h5_top_dir)
        reg.set_chunk_sz(self.xanes3D_reg_chunk_sz)
        reg.set_roi(self.xanes3D_roi)
        if self.xanes3D_reg_use_mask:
            reg.use_mask = True
            reg.set_mask(self.xanes3D_mask)
        ffn = self.xanes3D_recon_3D_tiff_temp.format(self.xanes3D_fixed_scan_id,
                                                     str(self.xanes3D_fixed_sli_id).zfill(5))
        reg.set_fixed_data(tifffile.imread(ffn)[self.xanes3D_roi[0]:self.xanes3D_roi[1], 
                                                self.xanes3D_roi[2]:self.xanes3D_roi[3]])
        reg.set_3D_recon_path_template(self.xanes3D_recon_3D_tiff_temp)
        reg.set_saving(save_path=os.path.dirname(self.xanes3D_save_trial_reg_filename),
                        fn=os.path.basename(self.xanes3D_save_trial_reg_filename))
        
        reg.xanes3D_sli_search_half_range = self.xanes3D_reg_sli_search_half_width
        reg.xanes3D_recon_fixed_sli = self.xanes3D_fixed_sli_id
        reg.reg_xanes3D_chunk()
        
        f = h5py.File(self.xanes3D_save_trial_reg_filename, 'r')
        self.trial_reg = np.ndarray(f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format(str(0).zfill(3))].shape)
        self.hs['L[0][2][2][1][2][0]_reg_pair_slider'].max = f['/trial_registration/trial_reg_parameters/alignment_pairs'].shape[0]-1
        self.hs['L[0][2][2][1][2][0]_reg_pair_slider'].min = 0 
        f.close()       
        self.hs['L[0][2][2][1][3][0]_zshift_slider'].max = self.xanes3D_reg_sli_search_half_width*2-1
        self.hs['L[0][2][2][1][3][0]_zshift_slider'].min = 0
        
        self.xanes3D_reg_done = True        
        self.boxes_logic()
    
    def L0_2_2_1_1_0_read_alignment_checkbox_change(self, a):
        if a['owner'].value:
            self.xanes3D_use_existing_reg_review = True
        else:
            self.xanes3D_use_existing_reg_review = False
        self.xanes3D_reg_review_done = False        
        self.boxes_logic()
    
    def L0_2_2_1_1_1_read_alignment_button_click(self, a):
        if len(a.files[0]) != 0:
            self.xanes3D_reg_file_read = True
            self.xanes3D_reg_review_file = os.path.abspath(a.files[0])
            if os.path.splitext(self.xanes3D_reg_review_file)[1] == '.json':
                self.alignment_best_match = json.load(open(self.xanes3D_reg_review_file, 'r'))
            else:
                self.alignment_best_match = np.genfromtxt(self.xanes3D_reg_review_file)
        else:
            self.xanes3D_reg_file_read = False
        self.xanes3D_reg_review_done = False
        self.boxes_logic()
    
    def L0_2_2_1_2_0_reg_pair_slider_change(self, a):
        self.xanes3D_alignment_pair_id = a['owner'].value
        fn = self.xanes3D_save_trial_reg_filename
        f = h5py.File(fn, 'r')
        self.trial_reg[:] = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format(str(self.xanes3D_alignment_pair_id).zfill(3))][:]
        f.close() 
        if (WindowManager.getIDList() is None) :
            ijui.show(ij.py.to_java(self.trial_reg))
            self.xanes3D_fiji_windows['mask_viewer']['ip'] = WindowManager.getCurrentImage()
            self.xanes3D_fiji_windows['mask_viewer']['fiji_id'] = WindowManager.getIDList()[-1]
            ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
            self.xanes3D_fiji_windows['mask_viewer']['ip'].setTitle('reg pair: '+str(self.xanes3D_alignment_pair_id).zfill(3)) 
        elif (self.xanes3D_fiji_windows['mask_viewer']['fiji_id'] not in WindowManager.getIDList()):
            ijui.show(ij.py.to_java(self.trial_reg))
            self.xanes3D_fiji_windows['mask_viewer']['ip'] = WindowManager.getCurrentImage()
            self.xanes3D_fiji_windows['mask_viewer']['fiji_id'] = WindowManager.getIDList()[-1]
            ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
            self.xanes3D_fiji_windows['mask_viewer']['ip'].setTitle('reg pair: '+str(self.xanes3D_alignment_pair_id).zfill(3)) 
        else:            
            self.xanes3D_fiji_windows['mask_viewer']['ip'].setImage(ij.convert().convert(ij.dataset().create(ij.py.to_java(self.trial_reg)), ImagePlusClass))
            self.xanes3D_fiji_windows['mask_viewer']['ip'].show()
            ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
            self.xanes3D_fiji_windows['mask_viewer']['ip'].setTitle('reg pair: '+str(self.xanes3D_alignment_pair_id).zfill(3))  
            self.xanes3D_fiji_windows['mask_viewer']['ip'].setTitle('reg pair: '+str(self.xanes3D_alignment_pair_id).zfill(3))        
        self.xanes3D_reg_review_ready = True
        self.xanes3D_reg_review_done = False
        self.boxes_logic()
    
    def L0_2_2_1_3_0_zshift_slider_change(self, a):
        if (WindowManager.getIDList() is None):
            self.hs['L[0][2][2][1][4][0]_confirm_review_results_text'].value = 'Please slide "reg pair #" to open a viewer ...'
        elif (self.xanes3D_fiji_windows['mask_viewer']['fiji_id'] not in WindowManager.getIDList()):
            self.hs['L[0][2][2][1][4][0]_confirm_review_results_text'].value = 'Please slide "reg pair #" to open a viewer ...'
        else:
            self.xanes3D_fiji_windows['mask_viewer']['ip'].setSlice(self.hs['L[0][2][2][1][3][0]_zshift_slider'].value)
            self.hs['L[0][2][2][1][3][1]_best_match_text'].value = a['owner'].value
        self.xanes3D_reg_review_done = False
        self.boxes_logic()
        
    def L0_2_2_1_3_1_best_match_text_change(self, a):
        self.xanes3D_reg_review_done = False
        self.xanes3D_alignment_pair_id = a['owner'].value
        # self.hs['L[0][2][2][1][2][0]_reg_pair_slider'].value = a.value
        self.boxes_logic()
    
    def L0_2_2_1_3_2_record_button_click(self, a):
        self.alignment_best_match[str(self.xanes3D_alignment_pair_id)] = self.hs['L[0][2][2][1][3][1]_best_match_text'].value
        self.hs['L[0][2][2][1][4][0]_confirm_review_results_text'].value = self.alignment_best_match
        self.xanes3D_reg_review_done = False
        self.boxes_logic()
    
    def L0_2_2_1_4_1_confirm_review_results_button_click(self, a):
        if len(self.alignment_best_match) != (self.hs['L[0][2][2][1][2][0]_reg_pair_slider'].max+1):
            self.hs['L[0][2][2][1][4][0]_confirm_review_results_text'].value = 'reg review is not completed yet ...'
            self.xanes3D_reg_review_done = False
        else:
            self.xanes3D_reg_review_done = True  
            data_state, viewer_state = self.fiji_viewer_state(viewer_name='virtural_stack_preview_viewer')
            if (not data_state) | (not viewer_state):
                self.fiji_viewer_on(viewer_name='virtural_stack_preview_viewer')
            self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setSlice(1) 
            # if WindowManager.getIDList() is None:
            #     self.fn0 = self.xanes3D_recon_3D_tiff_temp.format(self.xanes3D_fixed_scan_id,
            #                                                       str(min(self.xanes3D_available_recon_ids)).zfill(5))
            #     args = {'directory':self.fn0, 'start':1}
            #     ij.py.run_plugin(" Open VirtualStack", args)
            #     self.xanes3D_fiji_windows['virtural_stack_preview_viewer'] = {'ip':WindowManager.getCurrentImage(),
            #                                                                   'fiji_id':WindowManager.getIDList()[-1]}
            #     self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setSlice(1)              
            # elif not self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['fiji_id'] in WindowManager.getIDList():
            #     self.fn0 = self.xanes3D_recon_3D_tiff_temp.format(self.xanes3D_fixed_scan_id,
            #                                                       str(min(self.xanes3D_available_recon_ids)).zfill(5))
            #     args = {'directory':self.fn0, 'start':1}
            #     ij.py.run_plugin(" Open VirtualStack", args)
            #     self.xanes3D_fiji_windows['virtural_stack_preview_viewer'] = {'ip':WindowManager.getCurrentImage(),
            #                                                                   'fiji_id':WindowManager.getIDList()[-1]}
            #     self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setSlice(1)
            # else:
            #     self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].close()
            #     self.fn0 = self.xanes3D_recon_3D_tiff_temp.format(self.xanes3D_fixed_scan_id,
            #                                                       str(min(self.xanes3D_available_recon_ids)).zfill(5))
            #     args = {'directory':self.fn0, 'start':1}
            #     ij.py.run_plugin(" Open VirtualStack", args)
            #     self.xanes3D_fiji_windows['virtural_stack_preview_viewer'] = {'ip':WindowManager.getCurrentImage(),
            #                                                                   'fiji_id':WindowManager.getIDList()[-1]}
            #     self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setSlice(1) 
            
        self.boxes_logic()
    
    def L0_2_2_2_1_0_align_recon_optional_slice_start_text(self, a):
        if a['owner'].value <= self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_range_slider'].upper:
            self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_range_slider'].lower = a['owner'].value
        else:
            a['owner'].value = self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_range_slider'].upper
  
    def L0_2_2_2_1_2_align_recon_optional_slice_end_text(self, a):
        if a['owner'].value >= self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_range_slider'].lower:
            self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_range_slider'].upper = a['owner'].value
        else:
            a['owner'].value = self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_range_slider'].lower
    
    def L0_2_2_2_1_1_align_recon_optional_slice_range_slider(self, a):
        self.hs['L[0][2][2][2][1][0]_align_recon_optional_slice_start_text'].value = a['owner'].lower
        self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_end_text'].value = a['owner'].upper
        data_state, viewer_state = self.fiji_viewer_state(viewer_name='virtural_stack_preview_viewer')
        if (not data_state) | (not viewer_state):
            self.fiji_viewer_on(viewer_name='virtural_stack_preview_viewer')
        # if WindowManager.getIDList() is None:
        #     self.fn0 = self.xanes3D_recon_3D_tiff_temp.format(self.xanes3D_fixed_scan_id,
        #                                                       str(min(self.xanes3D_available_recon_ids)).zfill(5))
        #     args = {'directory':self.fn0, 'start':1}
        #     ij.py.run_plugin(" Open VirtualStack", args)
        #     self.xanes3D_fiji_windows['virtural_stack_preview_viewer'] = {'ip':WindowManager.getCurrentImage(),
        #                                                                   'fiji_id':WindowManager.getIDList()[-1]}              
        # elif not self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['fiji_id'] in WindowManager.getIDList():
        #     self.fn0 = self.xanes3D_recon_3D_tiff_temp.format(self.xanes3D_fixed_scan_id,
        #                                                       str(min(self.xanes3D_available_recon_ids)).zfill(5))
        #     args = {'directory':self.fn0, 'start':1}
        #     ij.py.run_plugin(" Open VirtualStack", args)
        #     self.xanes3D_fiji_windows['virtural_stack_preview_viewer'] = {'ip':WindowManager.getCurrentImage(),
        #                                                                   'fiji_id':WindowManager.getIDList()[-1]}
        self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setSlice(a['owner'].lower - a['owner'].min + 1)
        self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setRoi(int(self.slice_roi[2]), int(self.slice_roi[0]),
                                                                                int(self.slice_roi[3]-self.slice_roi[2]),
                                                                                int(self.slice_roi[1]-self.slice_roi[0]))
    
    def L0_2_2_2_2_1_align_button_click(self, a):
        self.xanes3D_align_sli_start = self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_range_slider'].lower
        self.xanes3D_align_sli_end = self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_range_slider'].upper
        f = h5py.File(self.xanes3D_save_trial_reg_filename, 'r')
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
        reg.set_saving(save_path=os.path.dirname(self.xanes3D_save_trial_reg_filename), fn=os.path.basename(self.xanes3D_save_trial_reg_filename))
        reg.apply_xanes3D_chunk_shift(self.alignment_best_match,
                                      self.xanes3D_align_sli_start,
                                      self.xanes3D_align_sli_end,
                                      trialfn=self.xanes3D_save_trial_reg_filename,
                                      savefn=self.xanes3D_save_trial_reg_filename)
        self.xanes3D_alignment_done = True
        self.boxes_logic()
                
#     def tab0_3D_scan_id_s_change(a):  
#         if os.path.exists(self.xanes3D_raw_3D_h5_temp.format(str(self.hs["3D_tab0_scan_id_s"].value))):
#             if os.path.exists(self.xanes3D_recon_3D_top_dir.format(str(self.hs["3D_tab0_scan_id_s"].value))):
#                 if self.hs["3D_tab0_scan_id_e"].value < self.hs["3D_tab0_scan_id_s"].value:
#                     self.hs["3D_tab0_scan_id_e"].value = self.hs["3D_tab0_scan_id_s"].value
#                 self.hs["3D_tab0_fixed_scan_id"].disabled = False
#                 self.hs["3D_tab0_fixed_scan_id"].max = self.hs["3D_tab0_scan_id_e"].value
#                 self.hs["3D_tab0_fixed_scan_id"].min = self.hs["3D_tab0_scan_id_s"].value                    
#                 fn = sorted(glob.glob(os.path.join(self.xanes3D_recon_3D_top_dir.format(self.hs["3D_tab0_scan_id_s"].value), '*.tiff')))[0]
#                 self.xanes3D_img = tifffile.imread(fn)
#                 self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'scan ids are changed ...'
#             else:
#                 self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'specified scan_id starting number does not exist in recon top dir ...'
#                 self.hs["3D_tab0_fixed_scan_id"].disabled = True
#                 self.hs["3D_tab0_fixed_sli_id"].disabled = True
#                 self.hs["3D_tab0_fixed_sli_fiji_checkbox"].disabled = True
# #                    self.hs["3D_tab0_scan_id_confirm_button"].disabled = True
#         else:
#             self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'raw h5 file does not exist in the raw top dir ...'
#             self.hs["3D_tab0_fixed_scan_id"].disabled = True
#             self.hs["3D_tab0_fixed_sli_id"].disabled = True
#             self.hs["3D_tab0_fixed_sli_fiji_checkbox"].disabled = True
        
#     def tab0_3D_scan_id_e_change(a):
#         if os.path.exists(self.xanes3D_raw_3D_h5_temp.format(str(self.hs["3D_tab0_scan_id_e"].value))):
#             if os.path.exists(self.xanes3D_recon_3D_top_dir.format(str(self.hs["3D_tab0_scan_id_e"].value))):
#                 if self.hs["3D_tab0_scan_id_e"].value < self.hs["3D_tab0_scan_id_s"].value:
#                     self.hs["3D_tab0_scan_id_s"].value = self.hs["3D_tab0_scan_id_e"].value
#                 self.hs["3D_tab0_fixed_scan_id"].disabled = False
#                 self.hs["3D_tab0_fixed_scan_id"].max = self.hs["3D_tab0_scan_id_e"].value
#                 self.hs["3D_tab0_fixed_scan_id"].min = self.hs["3D_tab0_scan_id_s"].value                    
#                 self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'scan ids are changed ...'
#             else:
#                 self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'specified scan_id starting number does not exist in recon top dir ...'
#                 self.hs["3D_tab0_fixed_scan_id"].disabled = True
#                 self.hs["3D_tab0_fixed_sli_id"].disabled = True
#                 self.hs["3D_tab0_fixed_sli_fiji_checkbox"].disabled = True
# #                    self.hs["3D_tab0_scan_id_confirm_button"].disabled = True
#         else:
#             self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'raw h5 file does not exist in the raw top dir ...'
#             self.hs["3D_tab0_fixed_scan_id"].disabled = True
#             self.hs["3D_tab0_fixed_sli_id"].disabled = True

#     def tab0_3D_fixed_scan_id_slider_change(a):
#         if os.path.exists(self.xanes3D_recon_3D_top_dir.format(str(self.hs["3D_tab0_fixed_scan_id"].value))):
#             self.hs["3D_tab0_fixed_sli_id"].disabled = False
#             self.hs["3D_tab0_fixed_sli_fiji_checkbox"].disabled = False
#             self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'fixed scan id is changed ...'
#             file_list = sorted(glob.glob(os.path.join(self.xanes3D_recon_3D_top_dir.format(self.hs["3D_tab0_fixed_scan_id"].value), '*.tiff')))
#             self.hs["3D_tab0_fixed_sli_id"].max = int(file_list[-1].split('.')[0].split('_')[-1])
#             self.hs["3D_tab0_fixed_sli_id"].min = int(file_list[0].split('.')[0].split('_')[-1])                
# #                self.hs['3D_tab0_scan_id_confirm_button_state'].value = '{}'.format(str(self.hs["3D_tab0_fixed_scan_id"].value))
#         else:
#             self.hs["3D_tab0_fixed_scan_id"].disabled = True
#             self.hs["3D_tab0_fixed_sli_id"].disabled = True
#             self.hs["3D_tab0_fixed_sli_fiji_checkbox"].disabled = True
#             self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'recon with fixed scan id {} is changed ...'.format(str(self.hs["3D_tab0_fixed_scan_id"].value))
        
#     def tab0_3D_fixed_sli_slider_change(a):
#         if self.hs["3D_tab0_fixed_sli_fiji_checkbox"].value:
#             self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'fixed slice id is changed ...'
#             self.raw_viewer_ip.setSlice(self.hs["3D_tab0_fixed_sli_id"].value-self.hs["3D_tab0_fixed_sli_id"].min+1)
#         self.fixed_sli_3D_id = self.hs["3D_tab0_fixed_sli_id"].value
        
#     def tab0_3D_fixed_sli_fiji_checked(a):
#         if self.hs["3D_tab0_fixed_sli_fiji_checkbox"].value:
#             self.fn0 = self.recon_3D_tiff_temp.format(self.hs["3D_tab0_fixed_scan_id"].value,
#                                           str(self.hs["3D_tab0_fixed_sli_id"].min).zfill(5))
#             args = {'directory':self.fn0, 'start':1}
#             ij.py.run_plugin(" Open VirtualStack", args)
#             self.raw_viewer_ip = WindowManager.getCurrentImage()
#             self.raw_viewer_ip.setSlice(self.fixed_sli_3D_id--self.hs["3D_tab0_fixed_sli_id"].min+1) 
#             self.hs['3D_tab0_close_all_fiji_viewers_button'].disabled=False                
#         else:
#             self.raw_viewer_ip.close()
#             self.raw_viewer_ip = None
#             self.hs['3D_tab0_close_all_fiji_viewers_button'].disabled=True 
#             self.hs['3D_tab1_roi_x_range_slider'].disabled = True
#             self.hs['3D_tab1_roi_y_range_slider'].disabled = True
#             self.hs['3D_tab1_roi_z_range_slider'].disabled = True
#             self.hs['3D_tab1_set_roi_button'].disabled = True
            
#     def tab0_3D_close_all_fiji_viewers(a):
#         for ii in (WindowManager.getIDList()):
#             WindowManager.getImage(ii).close()
#         self.hs["3D_tab0_fixed_sli_fiji_checkbox"].value = False
#         self.hs['3D_tab1_roi_x_range_slider'].disabled = True
#         self.hs['3D_tab1_roi_y_range_slider'].disabled = True
#         self.hs['3D_tab1_roi_z_range_slider'].disabled = True
#         self.hs['3D_tab1_set_roi_button'].disabled = True
        
#     def tab0_3D_fc_option_checkbox(a):
#         if self.hs['3D_tab0_fc_option_checkbox'].value:
#             self.xanes3D_use_existing_reg_file = True
#         else:
#             self.xanes3D_use_existing_reg_file = False


    
#     def tab1_3D_roi_x_range_slider_change(a):
#         self.raw_viewer_ip.setRoi(self.hs['3D_tab1_roi_x_range_slider'].value[0],
#                               self.hs['3D_tab1_roi_y_range_slider'].value[0],
#                               self.hs['3D_tab1_roi_x_range_slider'].value[1]-self.hs['3D_tab1_roi_x_range_slider'].value[0],
#                               self.hs['3D_tab1_roi_y_range_slider'].value[1]-self.hs['3D_tab1_roi_y_range_slider'].value[0])
    
#     def tab1_3D_roi_y_range_slider_change(a):
#         self.raw_viewer_ip.setRoi(self.hs['3D_tab1_roi_x_range_slider'].value[0],
#                               self.hs['3D_tab1_roi_y_range_slider'].value[0],
#                               self.hs['3D_tab1_roi_x_range_slider'].value[1]-self.hs['3D_tab1_roi_x_range_slider'].value[0],
#                               self.hs['3D_tab1_roi_y_range_slider'].value[1]-self.hs['3D_tab1_roi_y_range_slider'].value[0])
    
#     def tab1_3D_roi_z_range_slider_change(a):
# #            self.hs["3D_tab0_fixed_sli_id"].value = self.hs['3D_tab1_roi_z_range_slider'].value[0]-self.hs['3D_tab1_roi_z_range_slider'].min+1
# #            self.hs["3D_tab0_fixed_sli_id"].value = self.hs['3D_tab1_roi_z_range_slider'].value[0]
#         self.raw_viewer_ip.setSlice(self.hs['3D_tab1_roi_z_range_slider'].value[0]-self.hs['3D_tab1_roi_z_range_slider'].min+1)
    

    
#     def tab1_3D_use_mask_checkbox(a):
#         if self.hs['3D_use_mask'].value:
#             self.mask = np.ndarray([self.roi[1]-self.roi[0],
#                                     self.roi[3]-self.roi[2]])
#             self.hs['3D_mask_thres'].disabled = False
#             self.hs['3D_mask_dilation_width'].disabled = False
#             self.hs['3D_reg_method'].options = ['MPC', 'PC', 'SR']
#         else:
#             self.hs['3D_reg_method'].options = ['PC', 'SR']
#             self.hs['3D_mask_thres'].disabled = True
#             self.hs['3D_mask_dilation_width'].disabled = True
            
#     def tab1_3D_mask_fiji_preview_checked(a):
#         if self.hs['3D_tab1_mask_preview_fiji_checkbox'].value:  
#             self.xanes3D_img = np.ndarray([self.scan_3D_id_e-self.scan_3D_id_s+1,
#                                     self.roi[1]-self.roi[0],
#                                     self.roi[3]-self.roi[2]])
#             cnt = 0
#             for ii in range(self.scan_3D_id_s, self.scan_3D_id_e+1):
#                 fn = self.recon_3D_tiff_temp.format(ii, str(self.fixed_sli_3D_id).zfill(5))
#                 self.xanes3D_img[cnt, ...] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
#                 cnt += 1
                
#             ijui.show(ij.py.to_java(self.xanes3D_img))
#             self.xanes3D_fiji_mask_viewer_ip = WindowManager.getCurrentImage()
#             self.xanes3D_fiji_mask_viewer_ip.setTitle('mask preview')
#         else:
#             self.xanes3D_fiji_mask_viewer_ip.close()
#             self.xanes3D_fiji_mask_viewer_ip = None
    
#     def tab1_3D_mask_dialation_slider(a):
#         self.mask[:] = skm.binary_dilation((self.xanes3D_img[self.fixed_scan_3D_id-self.scan_3D_id_s]>self.hs['3D_mask_thres'].value).astype(np.uint8),
#                                     np.ones([self.hs['3D_mask_dilation_width'].value,
#                                               self.hs['3D_mask_dilation_width'].value])).astype(np.uint8)[:]
#         self.xanes3D_fiji_mask_viewer_ip.setImage(ij.convert().convert(ij.dataset().create(ij.py.to_java(self.xanes3D_img*self.mask)), ImagePlusClass))
#         ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        
    
#     def tab1_3D_mask_threshold_slider(a):
#         if self.xanes3D_fiji_mask_viewer_ip is None:
#             self.hs['3D_tab1_reg_config_button_state'].value = 'Please enable "Fiji Preview" first ...'
#             self.hs['3D_tab1_mask_preview_fiji_checkbox'].value = True
#         self.mask[:] = skm.binary_dilation((self.xanes3D_img[self.fixed_scan_3D_id-self.scan_3D_id_s]>self.hs['3D_mask_thres'].value).astype(np.uint8),
#                                     np.ones([self.hs['3D_mask_dilation_width'].value,
#                                               self.hs['3D_mask_dilation_width'].value])).astype(np.uint8)[:]
#         self.xanes3D_fiji_mask_viewer_ip.setImage(ij.convert().convert(ij.dataset().create(ij.py.to_java(self.xanes3D_img*self.mask)), ImagePlusClass))
#         ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")


    
#     def tab2_3D_read_user_specified_reg_results_checkbox_change(a):
#         if self.hs['3D_tab2_read_user_specified_reg_results_checkbox'].value:
#             self.hs['3D_tab2_run_registration_progress'].disabled = True
#             self.hs['3D_tab2_select_reg_pair_slider'].disabled = True
#             self.hs['3D_tab2_reg_sli_search_range_slider'].disabled = True
#             self.hs['3D_tab2_reg_sli_best_match'].disabled = True
#             self.hs['3D_tab2_align_sli_best_match_record_button'].disabled = True
#             self.hs["3D_tab2_read_alignment_dir"].disabled = False
#             self.xanes3D_use_existing_reg_file = True
#         else:
#             self.hs['3D_tab2_run_registration_progress'].disabled = False
#             self.hs['3D_tab2_select_reg_pair_slider'].disabled = False
#             self.hs['3D_tab2_reg_sli_search_range_slider'].disabled = False
#             self.hs['3D_tab2_reg_sli_best_match'].disabled = False
#             self.hs['3D_tab2_align_sli_best_match_record_button'].disabled = False
#             self.hs["3D_tab2_read_alignment_dir"].disabled = True
#             self.xanes3D_use_existing_reg_file = False
            
#     def tab2_3D_select_reg_pair_slider_change(a):
#         self.xanes3D_alignment_pair_id = self.hs['3D_tab2_select_reg_pair_slider'].value
#         fn = self.trial_reg_3D_save_file
#         f = h5py.File(fn, 'r')
#         self.trial_reg[:] = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format(str(self.xanes3D_alignment_pair_id).zfill(3))][:]
#         f.close()
        
#         if self.xanes3D_fiji_mask_viewer_ip is None:
#             ijui.show(ij.py.to_java(self.trial_reg))
#             self.xanes3D_fiji_mask_viewer_ip = WindowManager.getCurrentImage()
#             ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
#             self.xanes3D_fiji_mask_viewer_ip.setTitle(str(self.xanes3D_alignment_pair_id).zfill(3))
#         else:
#             self.xanes3D_fiji_mask_viewer_ip.setImage(ij.convert().convert(ij.dataset().create(ij.py.to_java(self.trial_reg)), ImagePlusClass))
#             self.xanes3D_fiji_mask_viewer_ip.show()
#             ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
#             self.xanes3D_fiji_mask_viewer_ip.setTitle(str(self.xanes3D_alignment_pair_id).zfill(3))  
#     def tab2_3D_reg_sli_search_range_slider_change(a):
#             if self.xanes3D_fiji_mask_viewer_ip is None:
#                 self.hs['3D_tab2_reg_review_state'].value = 'Please slide "reg pair #" to open a viewer ...'
#             else:
#                 self.xanes3D_fiji_mask_viewer_ip.setSlice(self.hs['3D_tab2_reg_sli_search_range_slider'].value)

        
#     def xanes3D_tab0_fc_confirm_button(self, a):
#         if self.xanes3D_use_existing_reg_file:
#             self.hs['3D_tab2_reg_review_finish_button'].disabled=False
#             self.hs['3D_tab2_run_registration_progress'].disabled = False
#             self.hs['3D_tab2_select_reg_pair_slider'].disabled = False
#             self.hs['3D_tab2_reg_sli_search_range_slider'].disabled = False
#             self.hs['3D_tab2_reg_sli_best_match'].disabled = False
#             self.hs['3D_tab2_align_sli_best_match_record_button'].disabled = False
#             self.hs['3D_tab2_read_user_specified_reg_results_checkbox'].disabled = False
#             self.hs["3D_tab2_read_alignment_dir"].disabled = False
            
#             self.alignment_best_match = {}
            
#             self.hs['3D_tab0_fc_confirm_button_state'].value = 'Configuration is done!'
#             self.xanes3D_raw_3D_h5_top_dir = os.path.abspath(self.hs["3D_tab0_fc_raw_dir"].selected_path)
#             self.xanes3D_recon_3D_top_dir = os.path.abspath(self.hs["3D_tab0_fc_recon_dir"].selected_path)
#             self.trial_reg_3D_save_file = os.path.abspath(self.hs["3D_tab0_fc_trial_reg_file"].selected)
#             self.xanes3D_recon_3D_top_dir = os.path.join(self.xanes3D_recon_3D_top_dir,
#                                               'recon_fly_scan_id_{0}')
#             self.recon_3D_tiff_temp = os.path.join(self.xanes3D_recon_3D_top_dir,
#                                                 'recon_fly_scan_id_{0}_{1}.tiff')
#             self.xanes3D_raw_3D_h5_temp = os.path.join(self.xanes3D_raw_3D_h5_top_dir,
#                                             'fly_scan_id_{}.h5')
        
#             f = h5py.File(self.trial_reg_3D_save_file, 'r')
#             self.trial_reg = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format('0'.zfill(3))][:]
#             self.hs['3D_tab2_select_reg_pair_slider'].max = f['/trial_registration/trial_reg_parameters/alignment_pairs'].shape[0]-1
#             self.hs['3D_tab2_select_reg_pair_slider'].min = 0            
#             self.hs['3D_tab2_reg_sli_search_range_slider'].max = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format('0'.zfill(3))].shape[0]
#             self.hs['3D_tab2_reg_sli_search_range_slider'].min = 1
#             f.close()            
#         else:
#             self.hs['3D_tab2_reg_review_finish_button'].disabled=True
#             self.hs['3D_tab2_run_registration_progress'].disabled = True
#             self.hs['3D_tab2_select_reg_pair_slider'].disabled = True
#             self.hs['3D_tab2_reg_sli_search_range_slider'].disabled = True
#             self.hs['3D_tab2_reg_sli_best_match'].disabled = True
#             self.hs['3D_tab2_align_sli_best_match_record_button'].disabled = True
#             self.hs['3D_tab2_read_user_specified_reg_results_checkbox'].disabled = True
#             self.hs["3D_tab2_read_alignment_dir"].disabled = True
#             if self.hs["3D_tab0_fc_raw_dir"].selected_path:
#                 if self.hs["3D_tab0_fc_recon_dir"].selected_path:
#                     if self.hs["3D_tab0_fc_trial_reg_file"].selected_filename:
#                         self.hs['3D_tab0_fc_confirm_button_state'].value = 'Configuration is done!'
#                         self.xanes3D_raw_3D_h5_top_dir = os.path.abspath(self.hs["3D_tab0_fc_raw_dir"].selected_path)
#                         self.xanes3D_recon_3D_top_dir = os.path.abspath(self.hs["3D_tab0_fc_recon_dir"].selected_path)
#                         self.trial_reg_3D_save_file = os.path.abspath(self.hs["3D_tab0_fc_trial_reg_file"].selected)
#                         self.xanes3D_recon_3D_top_dir = os.path.join(self.xanes3D_recon_3D_top_dir,
#                                                               'recon_fly_scan_id_{0}')
#                         self.recon_3D_tiff_temp = os.path.join(self.xanes3D_recon_3D_top_dir,
#                                                             'recon_fly_scan_id_{0}_{1}.tiff')
#                         self.xanes3D_raw_3D_h5_temp = os.path.join(self.xanes3D_raw_3D_h5_top_dir,
#                                                             'fly_scan_id_{}.h5')
#                         self.hs["3D_tab0_scan_id_s"].disabled = False
#                         self.hs["3D_tab0_scan_id_e"].disabled = False
#                         self.hs["3D_tab0_scan_id_confirm_button"].disabled = False
#                         self.xanes3D_filepath_configured = True
#                     else:
#                         self.hs['3D_tab0_fc_confirm_button_state'].value = "trial_reg_file is not defined!"
#                 else:
#                     self.hs['3D_tab0_fc_confirm_button_state'].value = "recon_dir is not defined!"
#             else:
#                 self.hs['3D_tab0_fc_confirm_button_state'].value = "raw_dir is not defined!"
                
#     def tab0_3D_scan_id_confirm_button(self, a, rp_={}):  
#         if self.hs["3D_tab0_fixed_scan_id"].disabled: 
#             self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'fixed scan id is not configured yet!'
#         else:
#             if self.hs["3D_tab0_fixed_sli_id"].disabled: 
#                 self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'fixed slice id is not configured yet!'
#             else:
#                 self.scan_3D_id_s = self.hs["3D_tab0_scan_id_s"].value
#                 self.scan_3D_id_e = self.hs["3D_tab0_scan_id_e"].value
#                 self.fixed_scan_3D_id = self.hs["3D_tab0_fixed_scan_id"].value
#                 self.fixed_sli_3D_id = self.hs["3D_tab0_fixed_sli_id"].value
#                 self.hs['3D_tab1_roi_x_range_slider'].disabled = False
#                 self.hs['3D_tab1_roi_x_range_slider'].min = 0
#                 self.hs['3D_tab1_roi_x_range_slider'].max = self.raw_viewer_ip.width
#                 self.hs['3D_tab1_roi_y_range_slider'].disabled = False
#                 self.hs['3D_tab1_roi_y_range_slider'].min = 0
#                 self.hs['3D_tab1_roi_y_range_slider'].max = self.raw_viewer_ip.height
#                 self.hs['3D_tab1_roi_z_range_slider'].disabled = False
#                 self.hs['3D_tab1_roi_z_range_slider'].max = self.hs["3D_tab0_fixed_sli_id"].max
#                 self.hs['3D_tab1_roi_z_range_slider'].min = self.hs["3D_tab0_fixed_sli_id"].min
#                 self.hs['3D_tab1_roi_z_range_slider'].value = [self.hs["3D_tab0_fixed_sli_id"].value, self.hs["3D_tab0_fixed_sli_id"].max]
#                 self.hs['3D_tab1_set_roi_button'].disabled = False
#                 self.hs['3D_tab0_scan_id_confirm_button_state'].value = 'scan ids are configured!'
#                 self.xanes3D_indices_configured = True    
                        
#     def tab1_3D_set_roi_button(self, a, rp_={}):
#         self.roi = [self.hs['3D_tab1_roi_y_range_slider'].value[0],
#                     self.hs['3D_tab1_roi_y_range_slider'].value[1],
#                     self.hs['3D_tab1_roi_x_range_slider'].value[0],
#                     self.hs['3D_tab1_roi_x_range_slider'].value[1],
#                     self.hs['3D_tab1_roi_z_range_slider'].value[0],
#                     self.hs['3D_tab1_roi_z_range_slider'].value[1]]
#         self.alignment_sli_start = self.hs['3D_tab1_roi_z_range_slider'].value[0]
#         self.alignment_sli_end = self.hs['3D_tab1_roi_z_range_slider'].value[1]
        
#         self.hs['3D_use_mask'].disabled = False
#         self.hs['3D_use_anchor'].disabled = False
#         self.hs['3D_tab1_mask_preview_fiji_checkbox'].disabled = False
#         self.hs['moving_sli_search_half_range'].disabled = False
#         self.hs['3D_chunk_sz'].disabled = False
#         self.hs['3D_reg_method'].disabled = False
#         self.hs['3D_ref_mode'].disabled = False
#         self.hs['3D_tab1_reg_config_button'].disabled = False
#         self.roi_3D_params_configured = True
        
#     def tab1_3D_set_reg_config_button(self, a, rp_={}):
#         self.use_mask = self.hs['3D_use_mask'].value
#         self.use_anchor = self.hs['3D_use_anchor'].value
#         self.mask_thres = self.hs['3D_mask_thres'].value
#         self.mask_dilation = self.hs['3D_mask_dilation_width'].value
#         self.reg_method = self.hs['3D_reg_method'].value
#         self.ref_mode = self.hs['3D_ref_mode'].value
#         self.moving_sli_search_half_range = self.hs['moving_sli_search_half_range'].value
#         self.chunk_sz = self.hs['3D_chunk_sz'].value
#         self.reg_3D_params_configured = True
#         self.hs['3D_tab2_run_registration_button'].disabled = False
#         self.hs['3D_tab2_run_registration_progress'].disabled = False
#         self.hs['3D_tab1_reg_config_button_state'].value = 'Registration parameter configuration is completed!'
        
#     def tab2_3D_run_registration_button(self, a, rp_={}):
#         self.hs['3D_tab2_select_reg_pair_slider'].disabled = False
#         self.hs['3D_tab2_reg_sli_search_range_slider'].disabled = False
#         self.hs['3D_tab2_reg_sli_best_match'].disabled = False
#         self.hs['3D_tab2_reg_review_finish_button'].disabled = False
#         self.hs['3D_tab2_align_sli_best_match_record_button'].disabled = False
#         self.alignment_best_match = {}
        
#         reg = xr.regtools(dtype='3D_XANES', method=self.reg_method, mode='TRANSLATION')
#         reg.set_method(self.reg_method)
#         reg.set_ref_mode(self.ref_mode)
#         reg.cal_set_anchor(self.scan_3D_id_s, self.scan_3D_id_e, self.fixed_scan_3D_id, raw_h5_top_dir=self.xanes3D_raw_3D_h5_top_dir)
#         reg.set_chunk_sz(self.chunk_sz)
#         reg.set_roi(self.roi)
#         if self.use_mask:
#             reg.use_mask = True
#             reg.set_mask(self.mask)
#         ffn = self.recon_3D_tiff_temp.format(self.fixed_scan_3D_id,
#                                               str(self.fixed_sli_3D_id).zfill(5))
#         reg.set_fixed_data(tifffile.imread(ffn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]])
#         reg.set_3D_recon_path_template(self.recon_3D_tiff_temp)
#         reg.set_saving(save_path=os.path.dirname(self.trial_reg_3D_save_file),
#                         fn=os.path.basename(self.trial_reg_3D_save_file))
#         reg.xanes3D_sli_search_half_range = self.moving_sli_search_half_range
#         reg.xanes3D_recon_fixed_sli = self.fixed_sli_3D_id
#         reg.reg_xanes3D_chunk()
#         f = h5py.File(self.trial_reg_3D_save_file, 'r')
#         self.trial_reg = np.ndarray(f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format(str(0).zfill(3))].shape)
#         self.hs['3D_tab2_select_reg_pair_slider'].max = f['/trial_registration/trial_reg_parameters/alignment_pairs'].shape[0]-1
#         self.hs['3D_tab2_select_reg_pair_slider'].min = 0 
#         f.close()       
#         self.hs['3D_tab2_reg_sli_search_range_slider'].max = self.moving_sli_search_half_range*2
#         self.hs['3D_tab2_reg_sli_search_range_slider'].min = 1

        
#     def tab2_3D_reg_review_finish_button(self, a, rp_={}):
#         if self.xanes3D_use_existing_reg_file:
#             self.xanes3D_alignment_file = self.hs["3D_tab2_read_alignment_dir"].selected
#             try:
#                 if self.xanes3D_alignment_file.split('.')[-1] == 'json':
#                     self.alignment_best_match = json.load(open(self.xanes3D_alignment_file, 'r'))
#                 else:
#                     self.alignment_best_match = np.genfromtxt(self.xanes3D_alignment_file)
#                 self.hs['3D_tab2_align_recon_button'].disabled = False
#                 self.hs['3D_tab2_align_recon_progress'].disabled = False
#             except:
#                 self.hs['3D_tab2_reg_review_state'].value = 'The specified alignment file does not exist!'
#         else:
#             f = h5py.File(self.trial_reg_3D_save_file, 'r')
#             scan_id_s = f['/trial_registration/trial_reg_parameters/scan_ids'][0]
#             scan_id_e = f['/trial_registration/trial_reg_parameters/scan_ids'][-1] 
#             json.dump(self.alignment_best_match, open(self.trial_reg_3D_save_file.split('.')[0]+f'_{scan_id_s}-{scan_id_e}_zshift.json', 'w'))
#             self.hs['3D_tab2_align_recon_button'].disabled = False
#             self.hs['3D_tab2_align_recon_progress'].disabled = False
            
        
#     def tab2_3D_reg_review_sli_best_match_record_button(self, a):
#         self.alignment_pair = self.hs['3D_tab2_select_reg_pair_slider'].value
#         self.alignment_best_match[str(self.alignment_pair)] = self.hs['3D_tab2_reg_sli_best_match'].value-1
        
    # def tab2_3D_align_recon_button(self, a, rp_={}):
    #     f = h5py.File(self.trial_reg_3D_save_file, 'r')
    #     recon_top_dir = f['/trial_registration/data_directory_info/recon_top_dir'][()] 
    #     recon_path_template = f['/trial_registration/data_directory_info/recon_path_template'][()] 
    #     roi = f['/trial_registration/trial_reg_parameters/slice_roi'][:] 
    #     reg_method = f['/trial_registration/trial_reg_parameters/reg_method'][()].lower()
    #     ref_mode = f['/trial_registration/trial_reg_parameters/reg_ref_mode'][()].lower()
    #     scan_id_s = f['/trial_registration/trial_reg_parameters/scan_ids'][0]
    #     scan_id_e = f['/trial_registration/trial_reg_parameters/scan_ids'][-1]
    #     fixed_scan_id = f['/trial_registration/trial_reg_parameters/fixed_scan_id'][()]
    #     chunk_sz = f['/trial_registration/trial_reg_parameters/chunk_sz'][()]
    #     eng_list = f['/trial_registration/trial_reg_parameters/eng_list'][:]
    #     f.close()
        
    #     reg = xr.regtools(dtype='3D_XANES', method=reg_method, mode='TRANSLATION')
    #     reg.set_method(reg_method)
    #     reg.set_ref_mode(ref_mode)
    #     reg.cal_set_anchor(scan_id_s, scan_id_e, fixed_scan_id)
    #     reg.eng_list = eng_list
        
    #     reg.set_chunk_sz(chunk_sz)
    #     reg.set_roi(roi)
    #     reg.set_3D_recon_path_template(recon_path_template)
    #     reg.set_saving(save_path=os.path.dirname(self.trial_reg_3D_save_file), fn=os.path.basename(self.trial_reg_3D_save_file))
    #     reg.apply_xanes3D_chunk_shift(self.alignment_best_match,
    #                                   roi[4],
    #                                   roi[5],
    #                                   trialfn=self.trial_reg_3D_save_file,
    #                                   savefn=self.trial_reg_3D_save_file)
        
        
        
        

        
        
        