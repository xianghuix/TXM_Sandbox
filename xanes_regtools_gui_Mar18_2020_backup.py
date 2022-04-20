#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:19:17 2020

@author: xiao
"""

import traitlets
from tkinter import Tk, filedialog
from ipywidgets import widgets
from IPython.display import display
from fnmatch import fnmatch
import os, functools, glob, tifffile, h5py, json
import numpy as np
import skimage.morphology as skm
import xanes_regtools as xr
import xanes_math as xm
import xanes_analysis as xa
import time, gc

import imagej

try:
    ij = imagej.init('/home/xiao/software/Fiji.app', headless=False)
    ijui = ij.ui()
    ijui.showUI()

    from jnius import autoclass
    WindowManager = autoclass('ij.WindowManager')
    ImagePlusClass = autoclass('ij.ImagePlus')
except:
    print('fiji is already up running!')

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
        self.open_filetypes = (('json files', '*.json'), ('text files', '*.txt'))
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
        self.xanes3D_analysis_eng_configured = False
        self.xanes2D_read_alignment_option = False

        self.xanes3D_fiji_windows = {'virtural_stack_preview_viewer':{'ip':None,
                                                                      'fiji_id':None},
                                     'mask_viewer':{'ip':None,
                                                    'fiji_id':None},
                                     'analysis_viewer':{'ip':None,
                                                        'fiji_id':None},
                                     'analysis_viewer_z_plot_viewer':{'ip':None,
                                                                      'fiji_id':None}}

        self.xanes3D_filepath_configured = False
        self.xanes3D_indices_configured = False
        self.xanes3D_roi_configured = False
        self.xanes3D_reg_params_configured = False
        self.xanes3D_reg_done = False
        self.xanes3D_reg_review_done = False
        self.xanes3D_alignment_done = False
        self.xanes3D_use_existing_reg_file = False
        self.xanes3D_use_existing_reg_reviewed = False
        self.xanes3D_reg_review_file = None
        self.xanes3D_reg_use_anchor = True
        self.xanes3D_reg_use_mask = True
        self.xanes3D_reg_review_ready = False

        self.xanes3D_raw_h5_path_set = False
        self.xanes3D_recon_path_set = False
        self.xanes3D_save_trial_set = False
        self.xanes3D_scan_id_set = False
        self.xanes3D_reg_file_set = False
        self.xanes3D_config_file_set = False
        self.xanes3D_fixed_scan_id_set = False
        self.xanes3D_fixed_sli_id_set = False
        self.xanes3D_reg_file_readed = False

        self.xanes3D_alignment_best_match = {}
        self.xanes3D_reg_mask_dilation_width = 0
        self.xanes3D_reg_mask_thres = 0
        self.xanes3D_fixed_sli_id = 0
        self.xanes3D_img = None
        self.xanes3D_roi = [0, 10, 0, 10, 0, 10]
        self.xanes3D_mask = None
        self.xanes3D_aligned_data = None
        self.xanes3D_analysis_slice = 0
        self.xanes3D_raw_3D_h5_top_dir = None
        self.xanes3D_recon_3D_top_dir = None
        self.xanes3D_save_trial_reg_filename = None
        self.xanes3D_save_trial_reg_config_filename = None
        self.xanes3D_save_trial_reg_config_filename_original = None
        self.xanes3D_raw_3D_h5_temp = None
        self.xanes3D_available_raw_ids = None
        self.xanes3D_recon_3D_tiff_temp = None
        self.xanes3D_recon_3D_dir_temp = None
        self.xanes3D_available_recon_ids = None
        self.xanes3D_available_recon_file_ids = None
        self.xanes3D_analysis_option = 'Do New Reg'
        self.xanes3D_fixed_scan_id = None
        self.xanes3D_scan_id_s = None
        self.xanes3D_scan_id_e = None
        self.xanes3D_fixed_sli_id = None
        self.xanes3D_reg_sli_search_half_width = None
        self.xanes3D_reg_chunk_sz = None
        self.xanes3D_reg_method = None
        self.xanes3D_reg_ref_mode = None
        self.xanes3D_analysis_use_mask = False
        self.xanes3D_analysis_mask_thres = None
        self.xanes3D_analysis_mask = None

        self.xanes3D_config = {"filepath config":{"xanes3D_raw_3D_h5_top_dir":self.xanes3D_raw_3D_h5_top_dir,
                                                  "xanes3D_recon_3D_top_dir":self.xanes3D_recon_3D_top_dir,
                                                  "xanes3D_save_trial_reg_filename":self.xanes3D_save_trial_reg_filename,
                                                  "xanes3D_save_trial_reg_config_filename":self.xanes3D_save_trial_reg_config_filename,
                                                  "xanes3D_save_trial_reg_config_filename_original":self.xanes3D_save_trial_reg_config_filename_original,
                                                  "xanes3D_raw_3D_h5_temp":self.xanes3D_raw_3D_h5_temp,
                                                  "xanes3D_recon_3D_tiff_temp":self.xanes3D_recon_3D_tiff_temp,
                                                  "xanes3D_recon_3D_dir_temp":self.xanes3D_recon_3D_dir_temp,
                                                  "xanes3D_analysis_option":self.xanes3D_analysis_option,
                                                  "xanes3D_filepath_configured":self.xanes3D_filepath_configured,
                                                  "xanes3D_raw_h5_path_set":self.xanes3D_raw_h5_path_set,
                                                  "xanes3D_recon_path_set":self.xanes3D_recon_path_set,
                                                  "xanes3D_save_trial_set":self.xanes3D_save_trial_set
                                                  },
		       "indeices config":{"select_scan_id_start_text_min":0,
                                  "select_scan_id_start_text_val":0,
                                  "select_scan_id_start_text_max":0,
                                  "select_scan_id_end_text_min":0,
                                  "select_scan_id_end_text_val":0,
                                  "select_scan_id_end_text_max":0,
                                  "fixed_scan_id_slider_min":0,
                                  "fixed_scan_id_slider_val":0,
                                  "fixed_scan_id_slider_max":0,
                                  "fixed_sli_id_slider_min":0,
                                  "fixed_sli_id_slider_val":0,
                                  "fixed_sli_id_slider_max":0,
                                  "xanes3D_fixed_scan_id":self.xanes3D_fixed_scan_id,
                                  "xanes3D_scan_id_s":self.xanes3D_scan_id_s,
                                  "xanes3D_scan_id_e":self.xanes3D_scan_id_e,
                                  "xanes3D_fixed_sli_id":self.xanes3D_fixed_sli_id,
                                  "xanes3D_scan_id_set":self.xanes3D_scan_id_set,
                                  "xanes3D_fixed_scan_id_set":self.xanes3D_fixed_scan_id_set,
                                  "xanes3D_fixed_sli_id_set":self.xanes3D_fixed_sli_id_set,
                                  "xanes3D_indices_configured":self.xanes3D_indices_configured
                                  },
		       "roi config":{"3D_roi_x_slider_min":0,
                             "3D_roi_x_slider_val":0,
                             "3D_roi_x_slider_max":0,
                             "3D_roi_y_slider_min":0,
                             "3D_roi_y_slider_val":0,
                             "3D_roi_y_slider_max":0,
                             "3D_roi_z_slider_min":0,
                             "3D_roi_z_slider_val":0,
                             "3D_roi_z_slider_max":0,
                             "3D_roi":self.xanes3D_roi,
                             "xanes3D_roi_configured":self.xanes3D_roi_configured
                             },
		       "registration config":{"xanes3D_reg_use_anchor":self.xanes3D_reg_use_anchor,
                                      "xanes3D_reg_use_mask":self.xanes3D_reg_use_mask,
                                      "xanes3D_reg_mask_thres":self.xanes3D_reg_mask_thres,
                                      "xanes3D_reg_mask_dilation_width":self.xanes3D_reg_mask_dilation_width,
                                      "xanes3D_reg_sli_search_half_width":self.xanes3D_reg_sli_search_half_width,
                                      "xanes3D_reg_chunk_sz":self.xanes3D_reg_chunk_sz,
                                      "xanes3D_reg_method":self.xanes3D_reg_method,
                                      "xanes3D_reg_ref_mode":self.xanes3D_reg_ref_mode,
                                      "mask_thres_slider_min":0,
                                      "mask_thres_slider_val":0,
                                      "mask_thres_slider_max":0,
                                      "mask_dilation_slider_min":0,
                                      "mask_dilation_slider_val":0,
                                      "mask_dilation_slider_max":0,
                                      "sli_search_slider_min":0,
                                      "sli_search_slider_val":0,
                                      "sli_search_slider_max":0,
                                      "chunk_sz_slider_min":0,
                                      "chunk_sz_slider_val":0,
                                      "chunk_sz_slider_max":0,
                                      "xanes3D_reg_params_configured":self.xanes3D_reg_params_configured
                                      },
		       "run registration":{"xanes3D_reg_done":self.xanes3D_reg_done
                                   },
		       "review registration":{"xanes3D_use_existing_reg_reviewed":self.xanes3D_use_existing_reg_reviewed,
                                      "xanes3D_reg_review_file":self.xanes3D_reg_review_file,
                                      "xanes3D_reg_review_ready":self.xanes3D_reg_review_ready,
                                      "xanes3D_reg_review_done":self.xanes3D_reg_review_done,
                                      "read_alignment_checkbox":False,
                                      "reg_pair_slider_min":0,
                                      "reg_pair_slider_val":0,
                                      "reg_pair_slider_max":0,
                                      "zshift_slider_min":0,
                                      "zshift_slider_val":0,
                                      "zshift_slider_max":0,
                                      "best_match_text":0,
                                      "alignment_best_match":self.xanes3D_alignment_best_match
                                      }
		       }

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


    # def get_decendant(self, handle_dict_name, level=-1):
    #     for ii in self.hs.keys():
    #         if handle_dict_name in ii:
    #             self_handle = [self.hs[ii]]
    #     children_handles = []
    #     children_handles.append(self_handle)
    #     temp_handles = []
    #     actual_level = 0
    #     bottom = False

    #     if level == -1:
    #         while not bottom:
    #             for child in self_handle:
    #                 try:
    #                     children_handles.append(child.children)
    #                     for jj in range(len(child.children)):
    #                         temp_handles.append(child.children[jj])
    #                     actual_level += 1
    #                 except Exception as e:
    #                     # print(str(e))
    #                     bottom = True
    #             self_handle = temp_handles
    #             temp_handles = []
    #     else:
    #         for ii in range(level):
    #             for child in self_handle:
    #                 try:
    #                     children_handles.append(child.children)
    #                     for jj in range(len(child.children)):
    #                         temp_handles.append(child.children[jj])
    #                     actual_level = ii+1
    #                 except Exception as e:
    #                     # print(str(e))
    #                     bottom = True
    #             self_handle = temp_handles
    #             temp_handles = []
    #     return children_handles, actual_level

    def get_decendant(self, handle_dict_name, level=-1):
        for ii in self.hs.keys():
            if handle_dict_name in ii:
                self_handle = self.hs[ii]
                self_handle_name = ii
        children_handles = []
        children_handles.append(self_handle)
        parent_handle_label = self_handle_name.split('_')[0]
        if level == -1:
            for ii in self.hs.keys():
                if parent_handle_label in ii:
                    children_handles.append(self.hs[ii])
            actual_level = -1
        else:
            try:
                actual_level = 0
                for ii in range(level):
                    for jj in self.hs.keys():
                        if fnmatch(jj, parent_handle_label+ii*'[*]'):
                            children_handles.append(self.hs[jj])
                    actual_level += 1
            except:
                pass
        return children_handles, actual_level

    def fiji_viewer_on(self, viewer_name='virtural_stack_preview_viewer'):
        if viewer_name == 'virtural_stack_preview_viewer':
            data_state, viewer_state = self.fiji_viewer_state(viewer_name='virtural_stack_preview_viewer')
            if not viewer_state:
                self.fn0 = self.xanes3D_recon_3D_tiff_temp.format(self.xanes3D_fixed_scan_id,
                                                                  str(min(self.xanes3D_available_recon_file_ids)).zfill(5))
                args = {'directory':self.fn0, 'start':1}
                ij.py.run_plugin(" Open VirtualStack", args)
                ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
                self.xanes3D_fiji_windows['virtural_stack_preview_viewer'] = {'ip':WindowManager.getCurrentImage(),
                                                                              'fiji_id':WindowManager.getIDList()[-1]}
                self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setSlice(self.xanes3D_fixed_sli_id-
                                                                                          self.xanes3D_available_recon_file_ids[0])
        elif viewer_name == 'mask_viewer':
            data_state, viewer_state = self.fiji_viewer_state(viewer_name='mask_viewer')
            if not data_state:
                self.xanes3D_img = np.ndarray([self.xanes3D_roi[5]-self.xanes3D_roi[4]+1,
                                               self.xanes3D_roi[1]-self.xanes3D_roi[0],
                                               self.xanes3D_roi[3]-self.xanes3D_roi[2]])
                self.xanes3D_mask = np.ndarray([self.xanes3D_roi[1]-self.xanes3D_roi[0],
                                                self.xanes3D_roi[3]-self.xanes3D_roi[2]])
            if not viewer_state:
                cnt = 0
                for ii in range(self.xanes3D_roi[4], self.xanes3D_roi[5]+1):
                    fn = self.xanes3D_recon_3D_tiff_temp.format(self.xanes3D_fixed_scan_id, str(ii).zfill(5))
                    self.xanes3D_img[cnt, ...] = tifffile.imread(fn)[self.xanes3D_roi[0]:self.xanes3D_roi[1],
                                                                     self.xanes3D_roi[2]:self.xanes3D_roi[3]]
                    cnt += 1

                ijui.show(ij.py.to_java(self.xanes3D_img))
                ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
                self.xanes3D_fiji_windows['mask_viewer']['ip'] = WindowManager.getCurrentImage()
                self.xanes3D_fiji_windows['mask_viewer']['fiji_id'] = WindowManager.getIDList()[-1]
                self.xanes3D_fiji_windows['mask_viewer']['ip'].setSlice(self.xanes3D_fixed_sli_id-self.xanes3D_roi[4])
                self.xanes3D_fiji_windows['mask_viewer']['ip'].setTitle('mask preview')
            else:
                self.xanes3D_fiji_windows['mask_viewer']['ip'].setSlice(self.xanes3D_fixed_sli_id-self.xanes3D_roi[4])
        elif viewer_name == 'analysis_viewer':
            data_state, viewer_state = self.fiji_viewer_state(viewer_name='analysis_viewer')
            if not data_state:
                f = h5py.File(self.xanes3D_save_trial_reg_filename, 'r')
                if self.hs['L[0][2][3][0][1][0]_visualize_view_alignment_option_dropdown'].value == 'x-y-E':
                    self.xanes3D_aligned_data = 0
                    self.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][:, self.xanes3D_analysis_slice, :, :]
                elif self.hs['L[0][2][3][0][1][0]_visualize_view_alignment_option_dropdown'].value == 'y-z-E':
                    self.xanes3D_aligned_data = 0
                    self.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][:, :, :, self.xanes3D_analysis_slice]
                elif self.hs['L[0][2][3][0][1][0]_visualize_view_alignment_option_dropdown'].value == 'z-x-E':
                    self.xanes3D_aligned_data = 0
                    self.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][:, :, self.xanes3D_analysis_slice, :]
                elif self.hs['L[0][2][3][0][1][0]_visualize_view_alignment_option_dropdown'].value == 'x-y-z':
                    self.xanes3D_aligned_data = 0
                    self.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][self.xanes3D_analysis_slice, :, :, :]
                f.close()
            if not viewer_state:
                # ij.py.run_macro("""run("Monitor Memory...")""")
                ijui.show(ij.py.to_java(self.xanes3D_aligned_data))
                ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
                self.xanes3D_fiji_windows['analysis_viewer']['ip'] = WindowManager.getCurrentImage()
                self.xanes3D_fiji_windows['analysis_viewer']['fiji_id'] = WindowManager.getIDList()[-1]
                self.xanes3D_fiji_windows['analysis_viewer']['ip'].setSlice(self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].value)
                self.xanes3D_fiji_windows['analysis_viewer']['ip'].setTitle('xanes3D slice view')
            else:
                self.xanes3D_fiji_windows['analysis_viewer']['ip'].setSlice(self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].value)

    def fiji_viewer_state(self, viewer_name='virtural_stack_preview_viewer'):
        if viewer_name == 'virtural_stack_preview_viewer':
            if ((not self.xanes3D_recon_3D_tiff_temp) |
                (not self.xanes3D_fixed_scan_id) |
                (not self.xanes3D_available_recon_file_ids)):
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
            if ((self.xanes3D_img is None) |
                (self.xanes3D_mask is None)):
                data_state = False
            elif ((self.xanes3D_img.shape != (self.xanes3D_roi[5]-self.xanes3D_roi[4]+1,
                                            self.xanes3D_roi[1]-self.xanes3D_roi[0],
                                            self.xanes3D_roi[3]-self.xanes3D_roi[2])) |
                  (self.xanes3D_mask.shape != (self.xanes3D_roi[1]-self.xanes3D_roi[0],
                                                self.xanes3D_roi[3]-self.xanes3D_roi[2]))):
                data_state = False
            else:
                data_state = True

            if WindowManager.getIDList() is None:
                viewer_state = False
            elif not self.xanes3D_fiji_windows['mask_viewer']['fiji_id'] in WindowManager.getIDList():
                viewer_state = False
            else:
                viewer_state = True
        elif viewer_name == 'analysis_viewer':
            if (self.xanes3D_aligned_data is None):
                data_state = False
            else:
                f = h5py.File(self.xanes3D_save_trial_reg_filename, 'r')
                data_shape = f['/registration_results/reg_results/registered_xanes3D'].shape
                if self.hs['L[0][2][3][0][1][0]_visualize_view_alignment_option_dropdown'].value == 'x-y-E':
                    if (data_shape[0], data_shape[2], data_shape[3]) != self.xanes3D_aligned_data.shape:
                        data_state = False
                    else:
                        data_state = True
                elif self.hs['L[0][2][3][0][1][0]_visualize_view_alignment_option_dropdown'].value == 'y-z-E':
                    if (data_shape[0], data_shape[1], data_shape[2]) != self.xanes3D_aligned_data.shape:
                        data_state = False
                    else:
                        data_state = True
                elif self.hs['L[0][2][3][0][1][0]_visualize_view_alignment_option_dropdown'].value == 'z-x-E':
                    if (data_shape[0], data_shape[1], data_shape[3]) != self.xanes3D_aligned_data.shape:
                        data_state = False
                    else:
                        data_state = True
                elif self.hs['L[0][2][3][0][1][0]_visualize_view_alignment_option_dropdown'].value == 'x-y-z':
                    if (data_shape[1], data_shape[2], data_shape[3]) != self.xanes3D_aligned_data.shape:
                        data_state = False
                    else:
                        data_state = True
                else:
                    data_state = True
                f.close()
            if WindowManager.getIDList() is None:
                viewer_state = False
            elif not self.xanes3D_fiji_windows['analysis_viewer']['fiji_id'] in WindowManager.getIDList():
                viewer_state = False
            else:
                viewer_state = True
        else:
            print('Unrecognized viewer name')
            data_state = False
            viewer_state = False
        return data_state, viewer_state

    def enable_disable_boxes(self, boxes, disabled=True, level=-1):
        for box in boxes:
            child_handles, level = self.get_decendant(box, level=level)
            for child in child_handles:
                try:
                    child.disabled = disabled
                except Exception as e:
                    # print(str(e))
                    pass
        # for box in boxes:
        #     child_handles, level = self.get_decendant(box, level=level)
        #     for ii in child_handles:
        #         for child in ii:
        #             try:
        #                 child.disabled = disabled
        #             except Exception as e:
        #                 # print(str(e))
        #                 pass

    def lock_message_text_boxes(self):
        boxes = ['confirm_config_indices_text',
                 'confirm_roi_text',
                 'confirm_reg_params_text',
                 'run_reg_text',
                 'confirm_review_results_text',
                 'align_text',
                 'visualize_spec_view_text',
                 'analysis_run_text']
        self.enable_disable_boxes(boxes, disabled=True, level=-1)

    def update_xanes3D_config(self):
        if self.xanes3D_analysis_option == 'Do New Reg':
            self.xanes3D_save_trial_reg_config_filename = self.xanes3D_save_trial_reg_config_filename_template.format(self.xanes3D_scan_id_s, self.xanes3D_scan_id_e)
        self.xanes3D_config = {"filepath config":{"xanes3D_raw_3D_h5_top_dir":self.xanes3D_raw_3D_h5_top_dir,
                                                  "xanes3D_recon_3D_top_dir":self.xanes3D_recon_3D_top_dir,
                                                   "xanes3D_save_trial_reg_filename":self.xanes3D_save_trial_reg_filename,
                                                  # "xanes3D_save_trial_reg_config_filename":self.xanes3D_save_trial_reg_config_filename,
                                                  # "xanes3D_save_trial_reg_config_filename_original":self.xanes3D_save_trial_reg_config_filename_original,
                                                  "xanes3D_raw_3D_h5_temp":self.xanes3D_raw_3D_h5_temp,
                                                  "xanes3D_recon_3D_tiff_temp":self.xanes3D_recon_3D_tiff_temp,
                                                  "xanes3D_recon_3D_dir_temp":self.xanes3D_recon_3D_dir_temp,
                                                  "xanes3D_analysis_option":self.xanes3D_analysis_option,
                                                  "xanes3D_filepath_configured":self.xanes3D_filepath_configured,
                                                  "xanes3D_raw_h5_path_set":self.xanes3D_raw_h5_path_set,
                                                  "xanes3D_recon_path_set":self.xanes3D_recon_path_set,
                                                  "xanes3D_save_trial_set":self.xanes3D_save_trial_set
                                                  },
		       "indeices config":{"select_scan_id_start_text_min":self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].min,
                                  "select_scan_id_start_text_val":self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].value,
                                  "select_scan_id_start_text_max":self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].max,
                                  "select_scan_id_end_text_min":self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].min,
                                  "select_scan_id_end_text_val":self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].value,
                                  "select_scan_id_end_text_max":self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].max,
                                  "fixed_scan_id_slider_min":self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].min,
                                  "fixed_scan_id_slider_val":self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].value,
                                  "fixed_scan_id_slider_max":self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].max,
                                  "fixed_sli_id_slider_min":self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min,
                                  "fixed_sli_id_slider_val":self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].value,
                                  "fixed_sli_id_slider_max":self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].max,
                                  "xanes3D_fixed_scan_id":int(self.xanes3D_fixed_scan_id),
                                  "xanes3D_scan_id_s":int(self.xanes3D_scan_id_s),
                                  "xanes3D_scan_id_e":int(self.xanes3D_scan_id_e),
                                  "xanes3D_fixed_sli_id":int(self.xanes3D_fixed_sli_id),
                                  "xanes3D_scan_id_set":self.xanes3D_scan_id_set,
                                  "xanes3D_fixed_scan_id_set":self.xanes3D_fixed_scan_id_set,
                                  "xanes3D_fixed_sli_id_set":self.xanes3D_fixed_sli_id_set,
                                  "xanes3D_indices_configured":self.xanes3D_indices_configured
                                  },
		       "roi config":{"3D_roi_x_slider_min":self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].min,
                             "3D_roi_x_slider_val":self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value,
                             "3D_roi_x_slider_max":self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].max,
                             "3D_roi_y_slider_min":self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].min,
                             "3D_roi_y_slider_val":self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value,
                             "3D_roi_y_slider_max":self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].max,
                             "3D_roi_z_slider_min":self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].min,
                             "3D_roi_z_slider_val":self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].value,
                             "3D_roi_z_slider_max":self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].max,
                             "3D_roi":self.xanes3D_roi,
                             "xanes3D_roi_configured":self.xanes3D_roi_configured
                             },
		       "registration config":{"xanes3D_reg_use_anchor":self.xanes3D_reg_use_anchor,
                                      "xanes3D_reg_use_mask":self.xanes3D_reg_use_mask,
                                      "xanes3D_reg_mask_thres":self.xanes3D_reg_mask_thres,
                                      "xanes3D_reg_mask_dilation_width":self.xanes3D_reg_mask_dilation_width,
                                      "xanes3D_reg_sli_search_half_width":self.xanes3D_reg_sli_search_half_width,
                                      "xanes3D_reg_chunk_sz":self.xanes3D_reg_chunk_sz,
                                      "xanes3D_reg_method":self.xanes3D_reg_method,
                                      "xanes3D_reg_ref_mode":self.xanes3D_reg_ref_mode,
                                      "mask_thres_slider_min":self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].min,
                                      "mask_thres_slider_val":self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].value,
                                      "mask_thres_slider_max":self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].max,
                                      "mask_dilation_slider_min":self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].min,
                                      "mask_dilation_slider_val":self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].value,
                                      "mask_dilation_slider_max":self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].max,
                                      "sli_search_slider_min":self.hs['L[0][2][1][1][3][0]_sli_search_slider'].min,
                                      "sli_search_slider_val":self.hs['L[0][2][1][1][3][0]_sli_search_slider'].value,
                                      "sli_search_slider_max":self.hs['L[0][2][1][1][3][0]_sli_search_slider'].max,
                                      "chunk_sz_slider_min":self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].min,
                                      "chunk_sz_slider_val":self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].value,
                                      "chunk_sz_slider_max":self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].max,
                                      "xanes3D_reg_params_configured":self.xanes3D_reg_params_configured
                                      },
		       "run registration":{"xanes3D_reg_done":self.xanes3D_reg_done
                                   },
		       "review registration":{"xanes3D_use_existing_reg_reviewed":self.xanes3D_use_existing_reg_reviewed,
                                      "xanes3D_reg_review_file":self.xanes3D_reg_review_file,
                                      "xanes3D_reg_review_ready":self.xanes3D_reg_review_ready,
                                      "xanes3D_reg_review_done":self.xanes3D_reg_review_done,
                                      "read_alignment_checkbox":self.hs['L[0][2][2][1][1][1]_read_alignment_checkbox'].value,
                                      "reg_pair_slider_min":self.hs['L[0][2][2][1][2][0]_reg_pair_slider'].min,
                                      "reg_pair_slider_val":self.hs['L[0][2][2][1][2][0]_reg_pair_slider'].value,
                                      "reg_pair_slider_max":self.hs['L[0][2][2][1][2][0]_reg_pair_slider'].max,
                                      "zshift_slider_min":self.hs['L[0][2][2][1][3][0]_zshift_slider'].min,
                                      "zshift_slider_val":self.hs['L[0][2][2][1][3][0]_zshift_slider'].value,
                                      "zshift_slider_max":self.hs['L[0][2][2][1][3][0]_zshift_slider'].max,
                                      "best_match_text":self.hs['L[0][2][2][1][3][1]_best_match_text'].value,
                                      "alignment_best_match":self.xanes3D_alignment_best_match
                                      }
		       }

    def read_xanes3D_config(self):
        with open(self.xanes3D_save_trial_reg_config_filename_original, 'r') as f:
            self.xanes3D_config = json.load(f)

    def set_xanes3D_variables(self):
        self.xanes3D_raw_3D_h5_top_dir = self.xanes3D_config["filepath config"]["xanes3D_raw_3D_h5_top_dir"]
        self.xanes3D_recon_3D_top_dir = self.xanes3D_config["filepath config"]["xanes3D_recon_3D_top_dir"]
        self.xanes3D_save_trial_reg_filename = self.xanes3D_config["filepath config"]["xanes3D_save_trial_reg_filename"]
        # self.xanes3D_save_trial_reg_config_filename = self.xanes3D_config["filepath config"]["xanes3D_save_trial_reg_config_filename"]
        # self.xanes3D_save_trial_reg_config_filename_original = self.xanes3D_config["filepath config"]["xanes3D_save_trial_reg_config_filename_original"]
        self.xanes3D_raw_3D_h5_temp = self.xanes3D_config["filepath config"]["xanes3D_raw_3D_h5_temp"]
        self.xanes3D_recon_3D_tiff_temp = self.xanes3D_config["filepath config"]["xanes3D_recon_3D_tiff_temp"]
        self.xanes3D_recon_3D_dir_temp = self.xanes3D_config["filepath config"]["xanes3D_recon_3D_dir_temp"]
        self.xanes3D_analysis_option = self.xanes3D_config["filepath config"]["xanes3D_analysis_option"]
        self.xanes3D_filepath_configured = self.xanes3D_config["filepath config"]["xanes3D_filepath_configured"]
        self.xanes3D_raw_h5_path_set = self.xanes3D_config["filepath config"]["xanes3D_raw_h5_path_set"]
        self.xanes3D_recon_path_set = self.xanes3D_config["filepath config"]["xanes3D_recon_path_set"]
        self.xanes3D_save_trial_set = self.xanes3D_config["filepath config"]["xanes3D_save_trial_set"]

        b = glob.glob(os.path.join(self.xanes3D_raw_3D_h5_top_dir, 'fly*.h5'))
        self.xanes3D_available_raw_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])
        b = glob.glob(os.path.join(self.xanes3D_recon_3D_top_dir, 'recon_fly*'))
        self.xanes3D_available_recon_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])

        self.xanes3D_fixed_scan_id = self.xanes3D_config["indeices config"]["xanes3D_fixed_scan_id"]
        self.xanes3D_scan_id_s = self.xanes3D_config["indeices config"]["xanes3D_scan_id_s"]
        self.xanes3D_scan_id_e = self.xanes3D_config["indeices config"]["xanes3D_scan_id_e"]
        self.xanes3D_fixed_sli_id = self.xanes3D_config["indeices config"]["xanes3D_fixed_sli_id"]
        self.xanes3D_scan_id_set = self.xanes3D_config["indeices config"]["xanes3D_scan_id_set"]
        self.xanes3D_fixed_scan_id_set = self.xanes3D_config["indeices config"]["xanes3D_fixed_scan_id_set"]
        self.xanes3D_fixed_sli_id_set = self.xanes3D_config["indeices config"]["xanes3D_fixed_sli_id_set"]
        self.xanes3D_indices_configured = self.xanes3D_config["indeices config"]["xanes3D_indices_configured"]

        self.xanes3D_roi = self.xanes3D_config["roi config"]["3D_roi"]
        self.xanes3D_roi_configured = self.xanes3D_config["roi config"]["xanes3D_roi_configured"]

        self.xanes3D_reg_use_anchor = self.xanes3D_config["registration config"]["xanes3D_reg_use_anchor"]
        self.xanes3D_reg_use_mask = self.xanes3D_config["registration config"]["xanes3D_reg_use_mask"]
        self.xanes3D_reg_mask_thres = self.xanes3D_config["registration config"]["xanes3D_reg_mask_thres"]
        self.xanes3D_reg_mask_dilation_width = self.xanes3D_config["registration config"]["xanes3D_reg_mask_dilation_width"]
        self.xanes3D_reg_sli_search_half_width = self.xanes3D_config["registration config"]["xanes3D_reg_sli_search_half_width"]
        self.xanes3D_reg_chunk_sz = self.xanes3D_config["registration config"]["xanes3D_reg_chunk_sz"]
        self.xanes3D_reg_method = self.xanes3D_config["registration config"]["xanes3D_reg_method"]
        self.xanes3D_reg_ref_mode = self.xanes3D_config["registration config"]["xanes3D_reg_ref_mode"]
        self.xanes3D_reg_params_configured = self.xanes3D_config["registration config"]["xanes3D_reg_params_configured"]

        self.xanes3D_reg_done = self.xanes3D_config["run registration"]["xanes3D_reg_done"]

        self.xanes3D_use_existing_reg_reviewed = self.xanes3D_config["review registration"]["xanes3D_use_existing_reg_reviewed"]
        self.xanes3D_reg_review_file = self.xanes3D_config["review registration"]["xanes3D_reg_review_file"]
        self.xanes3D_reg_review_ready = self.xanes3D_config["review registration"]["xanes3D_reg_review_ready"]
        self.xanes3D_reg_review_done = self.xanes3D_config["review registration"]["xanes3D_reg_review_done"]
        self.xanes3D_alignment_best_match = self.xanes3D_config["review registration"]["alignment_best_match"]

        self.boxes_logic()

    def set_xanes3D_handles(self):
        self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].max = self.xanes3D_config["indeices config"]["select_scan_id_start_text_max"]
        self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].max = self.xanes3D_config["indeices config"]["select_scan_id_end_text_max"]
        self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].value = self.xanes3D_config["indeices config"]["select_scan_id_start_text_val"]
        self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].value = self.xanes3D_config["indeices config"]["select_scan_id_end_text_val"]
        self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'].min = self.xanes3D_config["indeices config"]["select_scan_id_start_text_min"]
        self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].min = self.xanes3D_config["indeices config"]["select_scan_id_end_text_min"]

        self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].max = self.xanes3D_config["indeices config"]["fixed_scan_id_slider_max"]
        self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].min = self.xanes3D_config["indeices config"]["fixed_scan_id_slider_min"]
        self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].value = self.xanes3D_config["indeices config"]["fixed_scan_id_slider_val"]
        self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].max = self.xanes3D_config["indeices config"]["fixed_sli_id_slider_max"]
        self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min = self.xanes3D_config["indeices config"]["fixed_sli_id_slider_min"]
        self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].value = self.xanes3D_config["indeices config"]["fixed_sli_id_slider_val"]

        self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].max = self.xanes3D_config["roi config"]["3D_roi_x_slider_max"]
        self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].min = self.xanes3D_config["roi config"]["3D_roi_x_slider_min"]
        self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].value = self.xanes3D_config["roi config"]["3D_roi_x_slider_val"]
        self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].max = self.xanes3D_config["roi config"]["3D_roi_y_slider_max"]
        self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].min = self.xanes3D_config["roi config"]["3D_roi_y_slider_min"]
        self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].value = self.xanes3D_config["roi config"]["3D_roi_y_slider_val"]
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].max = self.xanes3D_config["roi config"]["3D_roi_z_slider_max"]
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].min = self.xanes3D_config["roi config"]["3D_roi_z_slider_min"]
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].value = self.xanes3D_config["roi config"]["3D_roi_z_slider_val"]

        self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].max = self.xanes3D_config["registration config"]["mask_thres_slider_max"]
        self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].min = self.xanes3D_config["registration config"]["mask_thres_slider_min"]
        self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].value = self.xanes3D_config["registration config"]["mask_thres_slider_val"]
        self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].max = self.xanes3D_config["registration config"]["mask_dilation_slider_max"]
        self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].min = self.xanes3D_config["registration config"]["mask_dilation_slider_min"]
        self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].value = self.xanes3D_config["registration config"]["mask_dilation_slider_val"]
        self.hs['L[0][2][1][1][3][0]_sli_search_slider'].max = self.xanes3D_config["registration config"]["sli_search_slider_max"]
        self.hs['L[0][2][1][1][3][0]_sli_search_slider'].min = self.xanes3D_config["registration config"]["sli_search_slider_min"]
        self.hs['L[0][2][1][1][3][0]_sli_search_slider'].value = self.xanes3D_config["registration config"]["sli_search_slider_val"]
        self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].max = self.xanes3D_config["registration config"]["chunk_sz_slider_max"]
        self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].min = self.xanes3D_config["registration config"]["chunk_sz_slider_min"]
        self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].value = self.xanes3D_config["registration config"]["chunk_sz_slider_val"]

        self.hs['L[0][2][2][1][1][1]_read_alignment_checkbox'].value = self.xanes3D_config["review registration"]["read_alignment_checkbox"]
        self.hs['L[0][2][2][1][2][0]_reg_pair_slider'].max = self.xanes3D_config["review registration"]["reg_pair_slider_max"]
        self.hs['L[0][2][2][1][2][0]_reg_pair_slider'].min = self.xanes3D_config["review registration"]["reg_pair_slider_min"]
        self.hs['L[0][2][2][1][2][0]_reg_pair_slider'].value = self.xanes3D_config["review registration"]["reg_pair_slider_val"]
        self.hs['L[0][2][2][1][3][0]_zshift_slider'].max = self.xanes3D_config["review registration"]["zshift_slider_max"]
        # self.hs['L[0][2][2][1][3][0]_zshift_slider'].min = self.xanes3D_config["review registration"]["zshift_slider_min"]
        self.hs['L[0][2][2][1][3][0]_zshift_slider'].value = self.xanes3D_config["review registration"]["zshift_slider_val"]
        self.hs['L[0][2][2][1][3][1]_best_match_text'].value = self.xanes3D_config["review registration"]["best_match_text"]

        self.boxes_logic()


    def boxes_logic(self):
        if self.xanes3D_analysis_option in ['Do New Reg', 'Read Config File']:
            if not self.xanes3D_filepath_configured:
                boxes = ['config_indices_box',
                         '3D_roi_box',
                         'config_reg_params_box',
                         'run_reg_box',
                         'review_reg_results_box',
                         'align_recon_box']
                self.enable_disable_boxes(boxes, disabled=True, level=-1)
                self.lock_message_text_boxes()

            elif (self.xanes3D_filepath_configured & (not self.xanes3D_indices_configured)):
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
                    self.lock_message_text_boxes()
                else:
                    boxes = ['3D_roi_box',
                             'config_reg_params_box',
                             'run_reg_box',
                             'review_reg_results_box',
                             'align_recon_box']
                    self.enable_disable_boxes(boxes, disabled=True, level=-1)
                    boxes = ['config_indices_box']
                    self.enable_disable_boxes(boxes, disabled=False, level=-1)
                    self.lock_message_text_boxes()
            elif ((self.xanes3D_filepath_configured & self.xanes3D_indices_configured) &
                  (not self.xanes3D_roi_configured)):
                boxes = ['config_reg_params_box',
                        'run_reg_box',
                        'review_reg_results_box',
                        'align_recon_box']
                self.enable_disable_boxes(boxes, disabled=True, level=-1)
                boxes = ['config_indices_box',
                         '3D_roi_box']
                self.enable_disable_boxes(boxes, disabled=False, level=-1)
                self.lock_message_text_boxes()
            elif ((self.xanes3D_filepath_configured & self.xanes3D_indices_configured) &
                  (self.xanes3D_roi_configured & (not self.xanes3D_reg_params_configured))):
                boxes = ['run_reg_box',
                        'review_reg_results_box',
                        'align_recon_box']
                self.enable_disable_boxes(boxes, disabled=True, level=-1)
                boxes = ['config_indices_box',
                         '3D_roi_box',
                         'config_reg_params_box']
                self.enable_disable_boxes(boxes, disabled=False, level=-1)
                data_state, viewer_state = self.fiji_viewer_state(viewer_name='mask_viewer')
                if not viewer_state:
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
                self.lock_message_text_boxes()
            elif ((self.xanes3D_filepath_configured & self.xanes3D_indices_configured) &
                  (self.xanes3D_roi_configured & self.xanes3D_reg_params_configured) &
                  (not self.xanes3D_reg_done)):
                boxes = ['config_indices_box',
                         '3D_roi_box',
                         'config_reg_params_box',
                         'run_reg_box']
                self.enable_disable_boxes(boxes, disabled=False, level=-1)
                boxes = ['align_recon_box',
                         'review_reg_results_box']
                self.enable_disable_boxes(boxes, disabled=True, level=-1)
                self.lock_message_text_boxes()
            elif ((self.xanes3D_filepath_configured & self.xanes3D_indices_configured) &
                  (self.xanes3D_roi_configured & self.xanes3D_reg_params_configured) &
                  (self.xanes3D_reg_done & (not self.xanes3D_reg_review_done))):
                boxes = ['config_indices_box',
                         '3D_roi_box',
                         'config_reg_params_box',
                         'run_reg_box',
                         'review_reg_results_box']
                self.enable_disable_boxes(boxes, disabled=False, level=-1)
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
                self.lock_message_text_boxes()
            elif ((self.xanes3D_filepath_configured & self.xanes3D_indices_configured) &
                  (self.xanes3D_roi_configured & self.xanes3D_reg_params_configured) &
                  (self.xanes3D_reg_done & self.xanes3D_reg_review_done)):
                boxes = ['config_indices_box',
                         '3D_roi_box',
                         'config_reg_params_box',
                         'run_reg_box',
                         'review_reg_results_box',
                         'align_recon_box']
                self.enable_disable_boxes(boxes, disabled=False, level=-1)
                boxes = ['align_recon_optional_slice_start_text',
                         'align_recon_optional_slice_range_slider',
                         'align_recon_optional_slice_end_text']
                if self.hs['L[0][2][2][2][1][0]_align_recon_optional_slice_checkbox'].value:
                    self.enable_disable_boxes(boxes, disabled=False, level=-1)
                else:
                    self.enable_disable_boxes(boxes, disabled=True, level=-1)
                self.lock_message_text_boxes()
        elif self.xanes3D_analysis_option == 'Read Reg File':
            if (self.xanes3D_filepath_configured & (not self.xanes3D_reg_review_done)):
                boxes = ['review_reg_results_box']
                self.enable_disable_boxes(boxes, disabled=False, level=-1)
                self.hs['L[0][2][2][1][4][0]_confirm_review_results_text'].disabled = True
                if self.hs['L[0][2][2][1][1][1]_read_alignment_checkbox'].value:
                    boxes = ['_read_alignment_button']
                    self.enable_disable_boxes(boxes, disabled=False, level=-1)
                else:
                    boxes = ['_read_alignment_button']
                    self.enable_disable_boxes(boxes, disabled=True, level=-1)
                if self.xanes3D_use_existing_reg_reviewed:
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
                self.lock_message_text_boxes()
            elif (self.xanes3D_filepath_configured & (self.xanes3D_reg_review_done)):
                boxes = ['config_indices_box',
                         '3D_roi_box',
                         'config_reg_params_box',
                         'run_reg_box']
                self.enable_disable_boxes(boxes, disabled=True, level=-1)

                boxes = ['review_reg_results_box',
                         'align_recon_box']
                self.enable_disable_boxes(boxes, disabled=False, level=-1)
                boxes = ['align_recon_optional_slice_start_text',
                         'align_recon_optional_slice_range_slider',
                         'align_recon_optional_slice_end_text']
                if self.hs['L[0][2][2][2][1][0]_align_recon_optional_slice_checkbox'].value:
                    self.enable_disable_boxes(boxes, disabled=False, level=-1)
                else:
                    self.enable_disable_boxes(boxes, disabled=True, level=-1)
                self.lock_message_text_boxes()
        elif self.xanes3D_analysis_option == 'Do Analysis':
            if self.xanes3D_reg_file_set & self.xanes3D_filepath_configured:
                boxes = ['config_indices_box',
                         '3D_roi_box',
                         'config_reg_params_box',
                         'run_reg_box',
                         'review_reg_results_box',
                         'align_recon_box',
                         'analysis&display_form']
                self.enable_disable_boxes(boxes, disabled=True, level=-1)
                boxes = ['visualize_box',
                         'analysis_box']
                self.enable_disable_boxes(boxes, disabled=False, level=-1)
                if self.xanes3D_analysis_use_mask:
                    self.hs['L[0][2][3][1][3][1]_analysis_image_mask_scan_id_slider'].disabled = False
                    self.hs['L[0][2][3][1][3][1]_analysis_image_mask_thres_slider'].disabled = False
                    boxes = ['visualize_box']
                    self.enable_disable_boxes(boxes, disabled=True, level=-1)
                else:
                    self.hs['L[0][2][3][1][3][1]_analysis_image_mask_scan_id_slider'].disabled = True
                    self.hs['L[0][2][3][1][3][1]_analysis_image_mask_thres_slider'].disabled = True
                    boxes = ['visualize_box']
                    self.enable_disable_boxes(boxes, disabled=False, level=-1)
                if self.xanes3D_analysis_eng_configured:
                    self.hs['L[0][2][3][1][4][1]_analysis_run_button'].disabled = False
                else:
                    self.hs['L[0][2][3][1][4][1]_analysis_run_button'].disabled = True
                self.lock_message_text_boxes()
            else:
                boxes = ['config_indices_box',
                         '3D_roi_box',
                         'config_reg_params_box',
                         'run_reg_box',
                         'review_reg_results_box',
                         'align_recon_box',
                         'analysis&display_form']
                self.enable_disable_boxes(boxes, disabled=True, level=-1)
                self.lock_message_text_boxes()
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please specify and confirm the aligned xanes3D file ...'


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
        self.hs['L[0][2][3]_analysis&display_form'] = widgets.VBox()
        self.hs['L[0][2][0]_config_input_form'].layout = layout
        self.hs['L[0][2][1]_reg_setting_form'].layout = layout
        self.hs['L[0][2][2]_reg&review_form'].layout = layout
        self.hs['L[0][2][3]_analysis&display_form'].layout = layout



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
        self.hs['L[0][2][0][0][2][1]_select_recon_path_text'] = widgets.Text(value='Choose recon top directory ...', description='', disabled=True)
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
        self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].description = 'Save Reg File'
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
                                                                                 tooltip='Confirm: Confirm after you finish file configuration')
        self.hs['L[0][2][0][0][4][0]_confirm_file&path_button'].style.button_color = 'darkviolet'
        self.hs['L[0][2][0][0][4][0]_confirm_file&path_button'].on_click(self.L0_2_0_0_4_0_confirm_file_path_button_click)
        layout = {'width':'15%'}
        self.hs['L[0][2][0][0][4][0]_confirm_file&path_button'].layout = layout

        self.hs['L[0][2][0][0][4][2]_file_path_options_dropdown'] = widgets.Dropdown(value='Do New Reg',
                                                                              options=['Do New Reg',
                                                                                       'Read Reg File',
                                                                                       'Read Config File',
                                                                                       'Do Analysis'],
                                                                              description='',
                                                                              description_tooltip='"Do New Reg": start registration and review results from beginning; "Read Reg File": if you have already done registraion and like to review the results; "Read Config File": if you like to resume analysis with an existing configuration.',
                                                                              disabled=False)
        layout = {'width':'15%', 'top':'0%'}
        self.hs['L[0][2][0][0][4][2]_file_path_options_dropdown'].layout = layout

        self.hs['L[0][2][0][0][4][2]_file_path_options_dropdown'].observe(self.L0_2_0_0_4_2_file_path_options_dropdown, names='value')
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
        self.hs['L[0][2][0][1][4][0]_confirm_config_indices_button'].style.button_color = 'darkviolet'
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
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].add_traits(mylower=traitlets.traitlets.Any(self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].lower))
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].add_traits(myupper=traitlets.traitlets.Any(self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].upper))
        self.hs['L[0][2][1][0][1]_3D_roi_define_box'].children = self.get_handles('L[0][2][1][0][1]_3D_roi_define_box', -1)
        self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].observe(self.L0_2_1_0_1_0_3D_roi_x_slider_change, names='value')
        self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].observe(self.L0_2_1_0_1_1_3D_roi_y_slider_change, names='value')
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].observe(self.L0_2_1_0_1_2_3D_roi_z_slider_change, names='value')
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].observe(self.L0_2_1_0_1_2_3D_roi_z_slider_lower_change, names='mylower')
        self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].observe(self.L0_2_1_0_1_2_3D_roi_z_slider_upper_change, names='myupper')

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
        self.hs['L[0][2][1][0][2][1]_confirm_roi_button'].style.button_color = 'darkviolet'
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
        self.hs['L[0][2][1][1][5][1]_confirm_reg_params_button'].style.button_color = 'darkviolet'
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
                                                                       description_tooltip='run registration once you are ready ...',
                                                                       disabled=True)
        self.hs['L[0][2][2][0][1][0]_run_reg_button'].style.button_color = 'darkviolet'
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
                                                                         min = 1,
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
                                                                   value='Confirm after you finish reg review ...',
                                                                   disabled=True)
        self.hs['L[0][2][2][1][4][0]_confirm_review_results_text'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][2][2][1][4][1]_confirm_review_results_button'] = widgets.Button(description='Confirm',
                                                                       description_tooltip='Confirm after you finish reg review ...',
                                                                       disabled=True)
        self.hs['L[0][2][2][1][4][1]_confirm_review_results_button'].style.button_color = 'darkviolet'
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
        layout = {'width':'20%', 'height':'100%'}
        self.hs['L[0][2][2][2][1][0]_align_recon_optional_slice_checkbox'] = widgets.Checkbox(description='new z range',
                                                                                              description_tooltip='check this on if you like to adjust z slice range for alignment',
                                                                                              value =False,
                                                                                              disabled=True)
        self.hs['L[0][2][2][2][1][0]_align_recon_optional_slice_checkbox'].layout = layout
        layout = {'width':'10%', 'height':'100%'}
        self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_start_text'] = widgets.BoundedIntText(description='',
                                                                                          description_tooltip='In the case of reading and reviewing a registration file, you need to define slice start and end.',
                                                                                          value = 0,
                                                                                          min = 0,
                                                                                          max = 10,
                                                                                          disabled=True)
        self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_start_text'].layout = layout
        layout = {'width':'60%', 'height':'100%'}
        self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_range_slider'] = widgets.IntRangeSlider(description='z range',
                                                                                          description_tooltip='In the case of reading and reviewing a registration file, you need to define slice start and end.',
                                                                                          value = 0,
                                                                                          min = 0,
                                                                                          max = 10,
                                                                                          disabled=True)
        self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_range_slider'].layout = layout
        layout = {'width':'10%', 'height':'100%'}
        self.hs['L[0][2][2][2][1][3]_align_recon_optional_slice_end_text'] = widgets.BoundedIntText(description='',
                                                                                          description_tooltip='In the case of reading and reviewing a registration file, you need to define slice start and end.',
                                                                                          value = 0,
                                                                                          min = 0,
                                                                                          max = 10,
                                                                                          disabled=True)
        self.hs['L[0][2][2][2][1][3]_align_recon_optional_slice_end_text'].layout = layout
        self.hs['L[0][2][2][2][1]_align_recon_optional_slice_region_box'].children = self.get_handles('L[0][2][2][2][1]_align_recon_optional_slice_region_box', -1)
        self.hs['L[0][2][2][2][1][0]_align_recon_optional_slice_checkbox'].observe(self.L0_2_2_2_1_0_align_recon_optional_slice_checkbox, names='value')
        self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_start_text'].observe(self.L0_2_2_2_1_1_align_recon_optional_slice_start_text, names='value')
        self.hs['L[0][2][2][2][1][3]_align_recon_optional_slice_end_text'].observe(self.L0_2_2_2_1_3_align_recon_optional_slice_end_text, names='value')
        self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_range_slider'].observe(self.L0_2_2_2_1_2_align_recon_optional_slice_range_slider, names='value')

        ## ## ## ## ## run reg & status
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][2][2][2]_align_recon_comfirm_box'] = widgets.HBox()
        self.hs['L[0][2][2][2][2]_align_recon_comfirm_box'].layout = layout
        layout = {'top': '5px','width':'70%', 'height':'100%'}
        self.hs['L[0][2][2][2][2][0]_align_text'] = widgets.Text(description='',
                                                                   value='Confirm to proceed alignment ...',
                                                                   disabled=True)
        self.hs['L[0][2][2][2][2][0]_align_text'].layout = layout
        layout = {'width':'15%', 'height':'90%'}
        self.hs['L[0][2][2][2][2][1]_align_button'] = widgets.Button(description='Align',
                                                                       description_tooltip='This will perform xanes3D alignment according to your configurations ...',
                                                                       disabled=True)
        self.hs['L[0][2][2][2][2][1]_align_button'].style.button_color = 'darkviolet'
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


        ## ## ## bin sub-tabs in each tab - analysis&display TAB in 3D_xanes TAB -- start

        ## ## ## ## define functional widgets each tab in each sub-tab - visualziation box in analysis&display TAB -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.25*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][3][0]_visualize_box'] = widgets.VBox()
        self.hs['L[0][2][3][0]_visualize_box'].layout = layout

        ## ## ## ## ## label visualize box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][3][0][0]_visualize_title_box'] = widgets.HBox()
        self.hs['L[0][2][3][0][0]_visualize_title_box'].layout = layout
        self.hs['L[0][2][3][0][0][0]_visualize_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Visualize XANES3D' + '</span>')
        layout = {'background-color':'white', 'color':'cyan', 'left':'41%'}
        self.hs['L[0][2][3][0][0][0]_visualize_title_text'].layout = layout
        self.hs['L[0][2][3][0][0]_visualize_title_box'].children = self.get_handles('L[0][2][3][0][0]_visualize_title_box', -1)
        ## ## ## ## ## label visualize box -- end

        ## ## ## ## ## define slice region and view slice cuts -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{1*self.form_sz[1]-98}px', 'height':f'{0.2*0.4*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][3][0][1]_visualize_view_alignment_box'] = widgets.HBox()
        self.hs['L[0][2][3][0][1]_visualize_view_alignment_box'].layout = layout
        layout = {'width':'20%', 'height':'100%'}
        self.hs['L[0][2][3][0][1][0]_visualize_view_alignment_option_dropdown'] = widgets.Dropdown(description='view option',
                                                                                                  description_tooltip='dimensions are defined as: E: energy dimension; x-y: slice lateral plane; z: dimension normal to slice plane',
                                                                                                  options=['x-y-E', 'y-z-E', 'z-x-E', 'x-y-z'],
                                                                                                  value ='x-y-E',
                                                                                                  disabled=True)
        self.hs['L[0][2][3][0][1][0]_visualize_view_alignment_option_dropdown'].layout = layout
        layout = {'width':'40%', 'height':'100%'}
        self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'] = widgets.IntSlider(description='z',
                                                                                               description_tooltip='Select one slice in the fourth dimension',
                                                                                               value =0,
                                                                                               min = 0,
                                                                                               disabled=True)
        self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'].layout = layout
        layout = {'width':'40%', 'height':'100%'}
        self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'] = widgets.IntSlider(description='E',
                                                                                                disabled=True)
        self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].layout = layout
        self.hs['L[0][2][3][0][1]_visualize_view_alignment_box'].children = self.get_handles('L[0][2][3][0][1]_visualize_view_alignment_box', -1)
        self.hs['L[0][2][3][0][1][0]_visualize_view_alignment_option_dropdown'].observe(self.L0_2_3_0_1_0_visualize_view_alignment_option_dropdown, names='value')
        self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'].observe(self.L0_2_3_0_1_2_visualize_view_alignment_4th_dim_slider, names='value')
        self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].observe(self.L0_2_3_0_1_3_visualize_view_alignment_slice_slider, names='value')
        ## ## ## ## ## define slice region and view slice cuts -- end

        ## ## ## ## ## basic spectroscopic visualization -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{1*self.form_sz[1]-98}px',
                  'height':f'{0.2*0.4*(self.form_sz[0]-128)}px',
                  'layout':'center'}
        self.hs['L[0][2][3][0][2]_visualize_spec_view_box'] = widgets.HBox()
        self.hs['L[0][2][3][0][2]_visualize_spec_view_box'].layout = layout
        layout = {'top': '5px','width':'70%', 'height':'100%'}
        self.hs['L[0][2][3][0][2][0]_visualize_spec_view_text'] = widgets.Text(description='',
                                                                               value='visualize spectrum in roi ...',
                                                                               disabled=True)
        self.hs['L[0][2][3][0][2][0]_visualize_spec_view_text'].layout = layout
        layout = {'width':'15%', 'height':'100%', 'left':'-8%', 'top':'20%'}
        self.hs['L[0][2][3][0][2][1]_visualize_spec_view_mem_monitor_checkbox'] = widgets.Checkbox(description='mem use',
                                                                                                  description_tooltip='Check on this to monitor memmory usage',
                                                                                                  value=False,
                                                                                                  disabled=True)
        self.hs['L[0][2][3][0][2][1]_visualize_spec_view_mem_monitor_checkbox'].layout = layout
        layout = {'width':'15%', 'height':'100%'}
        self.hs['L[0][2][3][0][2][2]_visualize_spec_view_in_roi_button'] = widgets.Button(description='spec in roi',
                                                                                         description_tooltip='adjust the roi size and drag roi over in the particles',
                                                                                         disabled=True)
        self.hs['L[0][2][3][0][2][2]_visualize_spec_view_in_roi_button'].layout = layout
        self.hs['L[0][2][3][0][2][2]_visualize_spec_view_in_roi_button'].style.button_color = 'darkviolet'
        self.hs['L[0][2][3][0][2]_visualize_spec_view_box'].children = self.get_handles('L[0][2][3][0][2]_visualize_spec_view_box', -1)
        self.hs['L[0][2][3][0][2][1]_visualize_spec_view_mem_monitor_checkbox'].observe(self.L0_2_3_0_2_1_visualize_spec_view_mem_monitor_checkbox, names='value')
        self.hs['L[0][2][3][0][2][2]_visualize_spec_view_in_roi_button'].on_click(self.L0_2_3_0_2_2_visualize_spec_view_in_roi_button)
        ## ## ## ## ## basic spectroscopic visualization -- end

        self.hs['L[0][2][3][0]_visualize_box'].children = self.get_handles('L[0][2][3][0]_visualize_box', -1)
        ## ## ## ## define functional widgets each tab in each sub-tab - visualziation box in analysis&display TAB -- end

        ## ## ## ## define functional widgets each tab in each sub-tab - analysis box in analysis&display TAB -- start
        layout = {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.49*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][3][1]_analysis_box'] = widgets.VBox()
        self.hs['L[0][2][3][1]_analysis_box'].layout = layout

        ## ## ## ## ## label analysis box -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{1./6.*0.49*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][3][1][0]_analysis_title_box'] = widgets.HBox()
        self.hs['L[0][2][3][1][0]_analysis_title_box'].layout = layout
        self.hs['L[0][2][3][1][0][0]_analysis_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Analyze 3D XANES' + '</span>')
        # self.hs['L[0][2][3][1][0][0]_analysis_title_text'] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; background-color:rgb(135,206,250);">' + 'Analyze XANES3D' + '</span>')
        layout = {'background-color':'white', 'color':'cyan', 'left':'41%'}
        self.hs['L[0][2][3][1][0][0]_analysis_title_text'].layout = layout
        self.hs['L[0][2][3][1][0]_analysis_title_box'].children = self.get_handles('L[0][2][3][0][0]_visualize_title_box', -1)
        ## ## ## ## ## label analysis box -- end

        ## ## ## ## ## define type of analysis and energy range -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{1*self.form_sz[1]-98}px', 'height':f'{1./6.*0.49*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][3][1][1]_analysis_energy_range_box'] = widgets.HBox()
        self.hs['L[0][2][3][1][1]_analysis_energy_range_box'].layout = layout
        layout = {'width':'19%', 'height':'100%'}
        self.hs['L[0][2][3][1][1][0]_analysis_energy_range_option_dropdown'] = widgets.Dropdown(description='analysis type',
                                                                                                  description_tooltip='wl: find whiteline positions without doing background removal and normalization; edge0.5: find energy point where the normalized spectrum value equal to 0.5; full: doing regular xanes preprocessing',
                                                                                                  options=['wl', 'edge0.5', 'full'],
                                                                                                  value ='wl',
                                                                                                  disabled=True)
        self.hs['L[0][2][3][1][1][0]_analysis_energy_range_option_dropdown'].layout = layout
        # layout = {'width':'19%', 'height':'100%', 'top':'0%', 'visibility':'hidden'}
        layout = {'width':'19%', 'height':'100%', 'top':'15%'}
        self.hs['L[0][2][3][1][1][1]_analysis_energy_range_edge_eng_text'] = widgets.BoundedFloatText(description='edge eng',
                                                                                                      description_tooltip='edge energy (keV)',
                                                                                                      value =0,
                                                                                                      min = 0,
                                                                                                      step=0.0005,
                                                                                                      disabled=True)
        self.hs['L[0][2][3][1][1][1]_analysis_energy_range_edge_eng_text'].layout = layout
        layout = {'width':'19%', 'height':'100%', 'top':'15%'}
        self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'] = widgets.BoundedFloatText(description='wl_eng_s',
                                                                                                   description_tooltip='relative ending energy point (keV) of pre-edge from edge energy for background removal',
                                                                                                   value =0,
                                                                                                   min = 0,
                                                                                                   step=0.0005,
                                                                                                   disabled=True)
        self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].layout = layout
        layout = {'width':'19%', 'height':'100%', 'top':'15%'}
        self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'] = widgets.BoundedFloatText(description='wl_eng_e',
                                                                                                   description_tooltip='relative starting energy point (keV) of post-edge from edge energy for normalization',
                                                                                                   value =0,
                                                                                                   min = 0,
                                                                                                   step=0.0005,
                                                                                                   disabled=True)
        self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].layout = layout
        layout = {'width':'15%', 'height':'100%', 'top':'0%', 'left':'2.5%'}
        self.hs['L[0][2][3][1][1][4]_analysis_energy_range_confirm_button'] = widgets.Button(description='Confirm',
                                                                                             description_tooltip='Confirm energy range settings',
                                                                                             disabled=True)
        self.hs['L[0][2][3][1][1][4]_analysis_energy_range_confirm_button'].layout = layout
        self.hs['L[0][2][3][1][1][4]_analysis_energy_range_confirm_button'].style.button_color = 'darkviolet'

        self.hs['L[0][2][3][1][1]_analysis_energy_range_box'].children = self.get_handles('L[0][2][3][1][1]_analysis_energy_range_box', -1)
        self.hs['L[0][2][3][1][1][0]_analysis_energy_range_option_dropdown'].observe(self.L0_2_3_1_1_0_analysis_energy_range_option_dropdown, names='value')
        self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].observe(self.L0_2_3_1_1_2_analysis_energy_range_eng_s_text, names='value')
        self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].observe(self.L0_2_3_1_1_3_analysis_energy_range_eng_e_text, names='value')
        self.hs['L[0][2][3][1][1][4]_analysis_energy_range_confirm_button'].on_click(self.L0_2_3_1_1_4_analysis_energy_range_confirm_button)
        ## ## ## ## ## define type of analysis and energy range -- end

        ## ## ## ## ## define energy filter related parameters -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{1*self.form_sz[1]-98}px', 'height':f'{1./6.*0.49*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][3][1][2]_analysis_energy_filter_box'] = widgets.HBox()
        self.hs['L[0][2][3][1][2]_analysis_energy_filter_box'].layout = layout
        layout = {'width':'48%', 'height':'100%', 'top':'0%'}
        self.hs['L[0][2][3][1][2][0]_analysis_energy_filter_edge_jump_thres_slider'] = widgets.FloatSlider(description='edge jump thres',
                                                                                                              description_tooltip='edge jump in unit of the standard deviation of the signal in energy range pre to the edge. larger threshold enforces more restrict data quality validation on the data',
                                                                                                              value =1,
                                                                                                              min = 0,
                                                                                                              max = 10,
                                                                                                              step=0.1,
                                                                                                              disabled=True)
        self.hs['L[0][2][3][1][2][0]_analysis_energy_filter_edge_jump_thres_slider'].layout = layout
        layout = {'width':'48%', 'height':'100%', 'top':'0%'}
        self.hs['L[0][2][3][1][2][1]_analysis_energy_filter_edge_offset_slider'] = widgets.FloatSlider(description='edge offset',
                                                                                      description_tooltip='offset between pre-edge and post-edge in unit of the standard deviation of pre-edge. larger offser enforces more restrict data quality validation on the data',
                                                                                      value =1,
                                                                                      min = 0,
                                                                                      max = 10,
                                                                                      step=0.1,
                                                                                      disabled=True)
        self.hs['L[0][2][3][1][2][1]_analysis_energy_filter_edge_offset_slider'].layout = layout

        self.hs['L[0][2][3][1][2]_analysis_energy_filter_box'].children = self.get_handles('L[0][2][3][1][2]_analysis_energy_filter_box', -1)
        ## ## ## ## ## define energy filter related parameters -- end

        ## ## ## ## ## define mask parameters -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{1*self.form_sz[1]-98}px', 'height':f'{1./6.*0.49*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][3][1][3]_analysis_image_mask_box'] = widgets.HBox()
        self.hs['L[0][2][3][1][3]_analysis_image_mask_box'].layout = layout
        layout = {'width':'20%', 'height':'100%', 'top':'20%'}
        self.hs['L[0][2][3][1][3][0]_analysis_image_use_mask_checkbox'] = widgets.Checkbox(description='use mask',
                                                                                              description_tooltip='use a mask based on gray value threshold to define sample region',
                                                                                              value =False,
                                                                                              disabled=True)
        self.hs['L[0][2][3][1][3][0]_analysis_image_use_mask_checkbox'].layout = layout
        layout = {'width':'40%', 'height':'100%', 'top':'0%'}
        self.hs['L[0][2][3][1][3][1]_analysis_image_mask_scan_id_slider'] = widgets.IntSlider(description='mask scan id',
                                                                                                 description_tooltip='scan id with which the mask is made',
                                                                                                 value =1,
                                                                                                 min = 0,
                                                                                                 max = 10,
                                                                                                 disabled=True)
        self.hs['L[0][2][3][1][3][1]_analysis_image_mask_scan_id_slider'].layout = layout
        layout = {'width':'48%', 'height':'100%', 'top':'0%'}
        self.hs['L[0][2][3][1][3][1]_analysis_image_mask_thres_slider'] = widgets.FloatSlider(description='mask thres',
                                                                                            description_tooltip='threshold for making the mask',
                                                                                            value =0,
                                                                                            min = -1,
                                                                                            max = 1,
                                                                                            step = 0.00005,
                                                                                            readout_format='.5f',
                                                                                            disabled=True)
        self.hs['L[0][2][3][1][3][1]_analysis_image_mask_thres_slider'].layout = layout

        self.hs['L[0][2][3][1][3]_analysis_image_mask_box'].children = self.get_handles('L[0][2][3][1][3]_analysis_image_mask_box', -1)
        self.hs['L[0][2][3][1][3][0]_analysis_image_use_mask_checkbox'].observe(self.L0_2_3_1_3_0_analysis_image_use_mask_checkbox, names='value')
        self.hs['L[0][2][3][1][3][1]_analysis_image_mask_scan_id_slider'].observe(self.L0_2_3_1_3_1_analysis_image_mask_scan_id_slider, names='value')
        self.hs['L[0][2][3][1][3][1]_analysis_image_mask_thres_slider'].observe(self.L0_2_3_1_3_1_analysis_image_mask_thres_slider, names='value')
        ## ## ## ## ## define mask parameters -- end

        ## ## ## ## ## run xanes3D analysis -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{1*self.form_sz[1]-98}px', 'height':f'{1./6.*0.49*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][3][1][4]_analysis_run_box'] = widgets.HBox()
        self.hs['L[0][2][3][1][4]_analysis_run_box'].layout = layout
        layout = {'width':'70%', 'height':'100%', 'top':'10%'}
        self.hs['L[0][2][3][1][4][0]_analysis_run_text'] = widgets.Text(description='please check your settings before run the analysis .. ',
                                                                        disabled=True)
        self.hs['L[0][2][3][1][4][0]_analysis_run_text'].layout = layout
        layout = {'width':'15%', 'height':'90%', 'top':'0%'}
        self.hs['L[0][2][3][1][4][1]_analysis_run_button'] = widgets.Button(description='run',
                                                                            disabled=True)
        self.hs['L[0][2][3][1][4][1]_analysis_run_button'].layout = layout
        self.hs['L[0][2][3][1][4][1]_analysis_run_button'].style.button_color = 'darkviolet'
        self.hs['L[0][2][3][1][4]_analysis_run_box'].children = self.get_handles('L[0][2][3][1][4]_analysis_run_box', -1)
        self.hs['L[0][2][3][1][4][1]_analysis_run_button'].on_click(self.L0_2_3_1_4_1_analysis_run_button)
        ## ## ## ## ## run xanes3D analysis -- end

        ## ## ## ## ## run analysis progress -- start
        layout = {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-98}px', 'height':f'{1./6.*0.49*(self.form_sz[0]-128)}px'}
        self.hs['L[0][2][3][1][5]_analysis_progress_box'] = widgets.HBox()
        self.hs['L[0][2][3][1][5]_analysis_progress_box'].layout = layout
        layout = {'top': '-2px','width':'100%', 'height':'100%'}
        self.hs['L[0][2][3][1][5][0]_analysis_run_progress_bar'] = widgets.IntProgress(value=0,
                                                                                       min=0,
                                                                                       max=10,
                                                                                       step=1,
                                                                                       description='Completing:',
                                                                                       bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                                                                       orientation='horizontal')
        self.hs['L[0][2][3][1][5][0]_analysis_run_progress_bar'].layout = layout
        self.hs['L[0][2][3][1][5]_analysis_progress_box'].children = self.get_handles('L[0][2][3][1][5]_analysis_progress_box', -1)
        ## ## ## ## ## run analysis progress -- end

        self.hs['L[0][2][3][1]_analysis_box'].children = self.get_handles('L[0][2][3][1]_analysis_box', -1)
        ## ## ## ## define functional widgets each tab in each sub-tab - analysis box in analysis&display TAB -- end

        self.hs['L[0][2][3]_analysis&display_form'].children = self.get_handles('L[0][2][3]_analysis&display_form', -1)
        ## ## ## bin sub-tabs in each tab - analysis&display TAB in 3D_xanes TAB -- end

        self.hs['L[0][2]_3D_xanes_tabs'].children = self.get_handles('L[0][2]_3D_xanes_tabs', -1)
        self.hs['L[0][2]_3D_xanes_tabs'].set_title(0, 'File Configurations')
        self.hs['L[0][2]_3D_xanes_tabs'].set_title(1, 'Registration Settings')
        self.hs['L[0][2]_3D_xanes_tabs'].set_title(2, 'Registration & Reviews')
        self.hs['L[0][2]_3D_xanes_tabs'].set_title(3, 'Analysis & Display')
        ## ## bin forms in hs['L[0][2]_3D_xanes_tabs']

        display(self.hs['L[0]_top_tab_form'])


    def L0_2_0_0_1_0_select_raw_h5_path_button_click(self, a):
        if len(a.files[0]) != 0:
            self.xanes3D_raw_3D_h5_top_dir = os.path.abspath(a.files[0])
            self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].initialdir = os.path.abspath(a.files[0])
            self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].initialdir = os.path.abspath(a.files[0])
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialdir = os.path.abspath(a.files[0])
            self.xanes3D_raw_3D_h5_temp = os.path.join(self.xanes3D_raw_3D_h5_top_dir, 'fly_scan_id_{}.h5')
            self.xanes3D_raw_h5_path_set = True
        else:
            self.hs['L[0][2][0][0][1][1]_select_raw_h5_path_text'].value = 'Choose raw h5 directory ...'
            self.xanes3D_raw_h5_path_set = False
        self.xanes3D_filepath_configured = False
        self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        self.boxes_logic()

    def L0_2_0_0_2_0_select_recon_path_button_click(self, a):
        if not self.xanes3D_raw_h5_path_set:
            self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'You need to specify raw h5 top directory first ...'
            self.hs['L[0][2][0][0][2][1]_select_recon_path_text'].value = 'Choose recon top directory ...'
            self.xanes3D_recon_path_set = False
            self.xanes3D_filepath_configured = False
        else:
            if len(a.files[0]) != 0:
                self.xanes3D_recon_3D_top_dir = os.path.abspath(a.files[0])
                self.xanes3D_recon_3D_dir_temp = os.path.join(self.xanes3D_recon_3D_top_dir,
                                                  'recon_fly_scan_id_{0}')
                self.xanes3D_recon_3D_tiff_temp = os.path.join(self.xanes3D_recon_3D_top_dir,
                                                               'recon_fly_scan_id_{0}',
                                                               'recon_fly_scan_id_{0}_{1}.tiff')
                self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].initialdir = os.path.abspath(a.files[0])
                self.xanes3D_recon_path_set = True
            else:
                self.hs['L[0][2][0][0][2][1]_select_recon_path_text'].value = 'Choose recon top directory ...'
                self.xanes3D_recon_path_set = False
            self.xanes3D_filepath_configured = False
            self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        self.boxes_logic()

    def L0_2_0_0_3_0_select_save_trial_button_click(self, a):
        if self.xanes3D_analysis_option == 'Do New Reg':
            if len(a.files[0]) != 0:
                b = ''
                t = (time.strptime(time.asctime()))
                for ii in range(6):
                    b +=str(t[ii]).zfill(2)+'-'
                self.xanes3D_save_trial_reg_filename_template = os.path.abspath(a.files[0]).split('.')[0]+'_scan_id_{0}-{1}_'+b.strip('-')+'.h5'
                self.xanes3D_save_trial_reg_config_filename_template = os.path.abspath(a.files[0]).split('.')[0]+'_scan_id_{0}-{1}_config_'+b.strip('-')+'.json'
                self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
                self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialfile = os.path.basename(a.files[0])
                self.xanes3D_save_trial_set = True
                self.xanes3D_reg_file_set = False
                self.xanes3D_config_file_set = False
            else:
                self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Save trial registration as ...'
                self.xanes3D_save_trial_set = False
                self.xanes3D_reg_file_set = False
                self.xanes3D_config_file_set = False
            self.xanes3D_filepath_configured = False
            self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        elif self.xanes3D_analysis_option == 'Read Reg File':
            if len(a.files[0]) != 0:
                self.xanes3D_save_trial_reg_filename = os.path.abspath(a.files[0])
                b = ''
                t = (time.strptime(time.asctime()))
                for ii in range(6):
                    b +=str(t[ii]).zfill(2)+'-'
                self.xanes3D_save_trial_reg_config_filename = os.path.abspath(a.files[0]).split('.')[0]+'_config_'+b.strip('-')+'.json'
                self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
                self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialfile = os.path.basename(a.files[0])
                self.xanes3D_save_trial_set = False
                self.xanes3D_reg_file_set = True
                self.xanes3D_config_file_set = False
            else:
                self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Read Existing Registration File ...'
                self.xanes3D_save_trial_set = False
                self.xanes3D_reg_file_set = False
                self.xanes3D_config_file_set = False
            self.xanes3D_filepath_configured = False
            self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        elif self.xanes3D_analysis_option == 'Read Config File':
            if len(a.files[0]) != 0:
                self.xanes3D_save_trial_reg_config_filename_original = os.path.abspath(a.files[0])
                b = ''
                t = (time.strptime(time.asctime()))
                for ii in range(6):
                    b +=str(t[ii]).zfill(2)+'-'
                self.xanes3D_save_trial_reg_config_filename = os.path.abspath(a.files[0]).split('config')[0]+'config_'+b.strip('-')+'.json'
                self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
                self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialfile = os.path.basename(a.files[0])
                self.xanes3D_save_trial_set = False
                self.xanes3D_reg_file_set = False
                self.xanes3D_config_file_set = True
            else:
                self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Save Existing Configuration File ...'
                self.xanes3D_save_trial_set = False
                self.xanes3D_reg_file_set = False
                self.xanes3D_config_file_set = False
            self.xanes3D_filepath_configured = False
            self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        elif self.xanes3D_analysis_option == 'Do Analysis':
            if len(a.files[0]) != 0:
                self.xanes3D_save_trial_reg_filename = os.path.abspath(a.files[0])
                b = ''
                t = (time.strptime(time.asctime()))
                for ii in range(6):
                    b +=str(t[ii]).zfill(2)+'-'
                self.xanes3D_save_trial_reg_config_filename = os.path.abspath(a.files[0]).split('.')[0]+'_config_'+b.strip('-')+'.json'
                self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialdir = os.path.dirname(os.path.abspath(a.files[0]))
                self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialfile = os.path.basename(a.files[0])
                self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Existing Registration File is Read ...'
                self.xanes3D_save_trial_set = False
                self.xanes3D_reg_file_set = True
                self.xanes3D_config_file_set = False
            else:
                self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Read Existing Registration File ...'
                self.xanes3D_save_trial_set = False
                self.xanes3D_reg_file_set = False
                self.xanes3D_config_file_set = False
            self.xanes3D_filepath_configured = False
            self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        self.boxes_logic()

    def L0_2_0_0_4_2_file_path_options_dropdown(self, a):
        self.xanes3D_analysis_option = a['owner'].value
        self.xanes3D_filepath_configured = False
        if self.xanes3D_analysis_option == 'Do New Reg':
            self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].disabled = False
            self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].disabled = False
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].option = 'asksaveasfilename'
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].description = 'Save Reg File'
            self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Save trial registration as ...'
            self.xanes3D_save_trial_set = False
            self.xanes3D_reg_file_set = False
            self.xanes3D_config_file_set = False
            self.hs['L[0][2][2][1][1][1]_read_alignment_checkbox'].value = False
        elif self.xanes3D_analysis_option == 'Read Reg File':
            self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].disabled = True
            self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].disabled = True
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].option = 'askopenfilename'
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].description = 'Read Reg File'
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].open_filetypes = (('h5 files', '*.h5'),)
            self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Read Existing Registration File ...'
            self.xanes3D_save_trial_set = False
            self.xanes3D_reg_file_set = False
            self.xanes3D_config_file_set = False
            self.hs['L[0][2][2][1][1][0]_read_alignment_button'].text_h = self.hs['L[0][2][2][1][4][0]_confirm_review_results_text']
        elif self.xanes3D_analysis_option == 'Read Config File':
            self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].disabled = True
            self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].disabled = True
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].option = 'askopenfilename'
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].description = 'Read Config'
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].open_filetypes = (('json files', '*.json'), ('text files', '*.txt'))
            self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Save Existing Configuration File ...'
            self.xanes3D_save_trial_set = False
            self.xanes3D_reg_file_set = False
            self.xanes3D_config_file_set = False
        elif self.xanes3D_analysis_option == 'Do Analysis':
            self.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].disabled = True
            self.hs['L[0][2][0][0][2][0]_select_recon_path_button'].disabled = True
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].option = 'askopenfilename'
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].description = 'Read Reg File'
            self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].open_filetypes = (('h5 files', '*.h5'),)
            self.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Read Existing Registration File ...'
        self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].icon = "square-o"
        self.hs['L[0][2][0][0][3][0]_select_save_trial_button'].style.button_color = "orange"
        self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please comfirm your change ...'
        self.boxes_logic()

    def L0_2_0_0_4_0_confirm_file_path_button_click(self, a):
        if self.xanes3D_analysis_option == 'Do New Reg':
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
                b = glob.glob(os.path.join(self.xanes3D_raw_3D_h5_top_dir, 'fly*.h5'))
                self.xanes3D_available_raw_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])
                b = glob.glob(os.path.join(self.xanes3D_recon_3D_top_dir, 'recon_fly*'))
                self.xanes3D_available_recon_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])

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
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'XANES3D file config is done ...'
                self.update_xanes3D_config()
                self.xanes3D_filepath_configured = True
        elif self.xanes3D_analysis_option == 'Read Reg File':
            if not self.xanes3D_reg_file_set:
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please specifiy where to read trial reg result ...'
                self.xanes3D_filepath_configured = False
                self.xanes3D_reg_review_ready = False
            else:
                # try:
                f = h5py.File(self.xanes3D_save_trial_reg_filename, 'r')
                self.trial_reg = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format('000')][:]
                self.hs['L[0][2][2][1][2][0]_reg_pair_slider'].max = f['/trial_registration/trial_reg_parameters/alignment_pairs'][:].shape[0] - 1
                self.hs['L[0][2][2][1][2][0]_reg_pair_slider'].min = 0
                self.hs['L[0][2][2][1][3][0]_zshift_slider'].max = self.trial_reg.shape[0] - 1
                # self.hs['L[0][2][2][1][3][0]_zshift_slider'].min = 1
                self.xanes3D_recon_3D_tiff_temp = f['/trial_registration/data_directory_info/recon_path_template'][()]
                self.xanes3D_fixed_scan_id = f['/trial_registration/trial_reg_parameters/fixed_scan_id'][()]
                self.xanes3D_fixed_sli_id = f['/trial_registration/trial_reg_parameters/fixed_slice'][()]
                self.xanes3D_roi = f['/trial_registration/trial_reg_parameters/slice_roi'][:]
                f.close()

                b = glob.glob(os.path.join(self.xanes3D_raw_3D_h5_top_dir, 'fly*.h5'))
                self.xanes3D_available_raw_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])
                b = glob.glob(os.path.join(self.xanes3D_recon_3D_top_dir, 'recon_fly*'))
                self.xanes3D_available_recon_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])
                b = glob.glob(os.path.join(self.xanes3D_recon_3D_dir_temp.format(self.xanes3D_fixed_scan_id),
                                           f'recon_fly_scan_id_{self.xanes3D_fixed_scan_id}_*.tiff'))
                self.xanes3D_available_recon_file_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])

                self.xanes3D_recon_path_set = True
                self.hs['L[0][2][2][2][1][0]_align_recon_optional_slice_checkbox'].value = False
                self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_range_slider'].max = max(self.xanes3D_available_recon_file_ids)
                self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_range_slider'].value = (self.xanes3D_roi[4],
                                                                                                self.xanes3D_roi[5])
                self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_range_slider'].min = min(self.xanes3D_available_recon_file_ids)
                self.hs['L[0][2][2][2][1][3]_align_recon_optional_slice_end_text'].max = max(self.xanes3D_available_recon_file_ids)
                self.hs['L[0][2][2][2][1][3]_align_recon_optional_slice_end_text'].value = max(self.xanes3D_available_recon_file_ids)
                self.hs['L[0][2][2][2][1][3]_align_recon_optional_slice_end_text'].min = min(self.xanes3D_available_recon_file_ids)
                self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_start_text'].max = max(self.xanes3D_available_recon_file_ids)
                self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_start_text'].value = min(self.xanes3D_available_recon_file_ids)
                self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_start_text'].min = min(self.xanes3D_available_recon_file_ids)

                self.xanes3D_alignment_best_match={}
                self.xanes3D_filepath_configured = True
                self.xanes3D_reg_review_ready = False
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'XANES3D file config is done ...'
                # except:
                #     self.xanes3D_filepath_configured = False
                #     self.xanes3D_reg_review_ready = False
                #     self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Cannot open the specified file. Please check if it exists or is locked ...'
        elif self.xanes3D_analysis_option == 'Read Config File':
            if not self.xanes3D_config_file_set:
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please specifiy where to read the configuration file ...'
                self.xanes3D_filepath_configured = False
            else:
                self.read_xanes3D_config()
                self.set_xanes3D_variables()
                self.xanes3D_analysis_option = 'Read Config File'

                f = h5py.File(self.xanes3D_save_trial_reg_filename, 'r')
                self.trial_reg = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format('000')][:]
                self.xanes3D_recon_3D_tiff_temp = f['/trial_registration/data_directory_info/recon_path_template'][()]
                self.xanes3D_fixed_scan_id = f['/trial_registration/trial_reg_parameters/fixed_scan_id'][()]
                self.xanes3D_fixed_sli_id = f['/trial_registration/trial_reg_parameters/fixed_slice'][()]
                self.xanes3D_roi = f['/trial_registration/trial_reg_parameters/slice_roi'][:]
                f.close()

                b = glob.glob(os.path.join(self.xanes3D_raw_3D_h5_top_dir, 'fly*.h5'))
                self.xanes3D_available_raw_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])
                b = glob.glob(os.path.join(self.xanes3D_recon_3D_top_dir, 'recon_fly*'))
                self.xanes3D_available_recon_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])
                b = glob.glob(os.path.join(self.xanes3D_recon_3D_dir_temp.format(self.xanes3D_fixed_scan_id),
                                           f'recon_fly_scan_id_{self.xanes3D_fixed_scan_id}_*.tiff'))
                self.xanes3D_available_recon_file_ids = sorted([int(ii.split('.')[0].split('_')[-1]) for ii in b])

                self.xanes3D_recon_path_set = True
                self.hs['L[0][2][2][2][1][0]_align_recon_optional_slice_checkbox'].value = False
                self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_range_slider'].max = max(self.xanes3D_available_recon_file_ids)
                self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_range_slider'].value = (self.xanes3D_roi[4],
                                                                                                self.xanes3D_roi[5])
                self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_range_slider'].min = min(self.xanes3D_available_recon_file_ids)
                self.hs['L[0][2][2][2][1][3]_align_recon_optional_slice_end_text'].max = max(self.xanes3D_available_recon_file_ids)
                self.hs['L[0][2][2][2][1][3]_align_recon_optional_slice_end_text'].value = max(self.xanes3D_available_recon_file_ids)
                self.hs['L[0][2][2][2][1][3]_align_recon_optional_slice_end_text'].min = min(self.xanes3D_available_recon_file_ids)
                self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_start_text'].max = max(self.xanes3D_available_recon_file_ids)
                self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_start_text'].value = min(self.xanes3D_available_recon_file_ids)
                self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_start_text'].min = min(self.xanes3D_available_recon_file_ids)

                self.set_xanes3D_handles()
                self.set_xanes3D_variables()
                self.xanes3D_analysis_option = 'Read Config File'

                self.xanes3D_filepath_configured = True
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'XANES3D file config is done ...'
        elif self.xanes3D_analysis_option == 'Do Analysis':
            if not self.xanes3D_reg_file_set:
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'Please specifiy where to read the aligned data ...'
                self.xanes3D_filepath_configured = False
                self.xanes3D_reg_review_ready = False
                self.xanes3D_indices_configured = False
                self.xanes3D_roi_configured = False
                self.xanes3D_reg_params_configured = False
                self.xanes3D_reg_done = False
                self.xanes3D_alignment_done = False
            else:
                self.xanes3D_filepath_configured = True
                self.xanes3D_reg_review_ready = False
                self.xanes3D_indices_configured = False
                self.xanes3D_roi_configured = False
                self.xanes3D_reg_params_configured = False
                self.xanes3D_reg_done = False
                self.xanes3D_alignment_done = False
                f = h5py.File(self.xanes3D_save_trial_reg_filename, 'r')
                self.xanes3D_analysis_data_shape = f['/registration_results/reg_results/registered_xanes3D'].shape
                self.xanes3D_analysis_eng_list = f['/registration_results/trial_reg_parameters/eng_list'][:]
                f.close()
                self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].description = 'E'
                self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].max = self.xanes3D_analysis_data_shape[0]
                self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].value = 0
                self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].min = 0
                self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'].description = 'z'
                self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'].max = self.xanes3D_analysis_data_shape[1]
                self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'].value = 0
                self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'].min = 0
                self.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value = 'XANES3D file config is done ...'
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
        data_state, viewer_state = self.fiji_viewer_state(viewer_name='virtural_stack_preview_viewer')
        if not viewer_state:
            self.hs['L[0][2][0][1][4][1]_confirm_config_indices_text'].value = 'Please turn on fiji viewer to proceed ...'
            self.xanes3D_indices_configured = False
        else:
            self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].max = self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].width
            self.hs['L[0][2][1][0][1][0]_3D_roi_x_slider'].min = 0
            self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].max = self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].height
            self.hs['L[0][2][1][0][1][1]_3D_roi_y_slider'].min = 0
            self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].max = self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].max
            self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].min = self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min

            self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].upper = self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].max
            self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].lower = self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].min

            self.xanes3D_scan_id_s = self.hs['L[0][2][0][1][1][0]_select_scan_id_start_text'] .value
            self.xanes3D_scan_id_e = self.hs['L[0][2][0][1][1][1]_select_scan_id_end_text'].value
            self.xanes3D_fixed_sli_id = self.hs['L[0][2][0][1][2][1]_fixed_sli_id_slider'].value
            self.xanes3D_fixed_scan_id = self.hs['L[0][2][0][1][2][0]_fixed_scan_id_slider'].value
            self.hs['L[0][2][0][1][4][1]_confirm_config_indices_text'].value = 'Indices configuration is done ...'
            self.hs['L[0][2][1][0][2][0]_confirm_roi_text'].value = 'Please confirm after ROI is set'
            self.xanes3D_save_trial_reg_filename = self.xanes3D_save_trial_reg_filename_template.format(self.xanes3D_scan_id_s, self.xanes3D_scan_id_e)
            self.xanes3D_indices_configured = True
            self.update_xanes3D_config()
            json.dump(self.xanes3D_config, open(self.xanes3D_save_trial_reg_config_filename, 'w'))
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
        a['owner'].mylower = a['owner'].lower
        a['owner'].myupper = a['owner'].upper
        self.hs['L[0][2][1][0][2][0]_confirm_roi_text'].value = 'Please confirm after ROI is set'

    def L0_2_1_0_1_2_3D_roi_z_slider_lower_change(self, a):
        data_state, viewer_state = self.fiji_viewer_state(viewer_name='virtural_stack_preview_viewer')
        if (not data_state) | (not viewer_state):
            self.fiji_viewer_on(viewer_name='virtural_stack_preview_viewer')
        self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setSlice(self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].value[0]-self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].min+1)
        self.boxes_logic()

    def L0_2_1_0_1_2_3D_roi_z_slider_upper_change(self, a):
        data_state, viewer_state = self.fiji_viewer_state(viewer_name='virtural_stack_preview_viewer')
        if (not data_state) | (not viewer_state):
            self.fiji_viewer_on(viewer_name='virtural_stack_preview_viewer')
        self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setSlice(self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].value[1]-self.hs['L[0][2][1][0][1][2]_3D_roi_z_slider'].min+1)
        self.boxes_logic()

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
        self.update_xanes3D_config()
        json.dump(self.xanes3D_config, open(self.xanes3D_save_trial_reg_config_filename, 'w'))
        self.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value = False
        self.boxes_logic()

    def L0_2_1_1_1_0_fiji_mask_viewer_checkbox_change(self, a):
        if a['owner'].value:
            self.fiji_viewer_on(viewer_name='mask_viewer')
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
            self.xanes3D_reg_use_mask = True
        else:
            self.xanes3D_reg_use_mask = False
        self.boxes_logic()

    def L0_2_1_1_2_1_mask_thres_slider_change(self, a):
        self.xanes3D_reg_params_configured = False
        self.xanes3D_reg_mask_thres = a['owner'].value
        data_state, viewer_state = self.fiji_viewer_state(viewer_name='mask_viewer')
        if ((not data_state) |
            (not viewer_state)):
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
        self.boxes_logic()

    def L0_2_1_1_3_1_chunk_sz_slider_change(self, a):
        self.xanes3D_reg_params_configured = False
        self.boxes_logic()

    def L0_2_1_1_4_0_reg_method_dropdown_change(self, a):
        self.xanes3D_reg_params_configured = False
        self.boxes_logic()

    def L0_2_1_1_4_1_ref_mode_dropdown_change(self, a):
        self.xanes3D_reg_params_configured = False
        self.boxes_logic()

    def L0_2_1_1_5_1_confirm_reg_params_button_click(self, a):
        self.xanes3D_reg_sli_search_half_width = self.hs['L[0][2][1][1][3][0]_sli_search_slider'].value
        self.xanes3D_reg_chunk_sz = self.hs['L[0][2][1][1][3][1]_chunk_sz_slider'].value
        self.xanes3D_reg_method = self.hs['L[0][2][1][1][4][0]_reg_method_dropdown'].value
        self.xanes3D_reg_ref_mode = self.hs['L[0][2][1][1][4][1]_ref_mode_dropdown'].value
        self.xanes3D_reg_mask_dilation_width = self.hs['L[0][2][1][1][2][2]_mask_dilation_slider'].value
        self.xanes3D_reg_mask_thres = self.hs['L[0][2][1][1][2][1]_mask_thres_slider'].value
        self.xanes3D_reg_params_configured = self.hs['L[0][2][1][1][1][1]_anchor_checkbox'].value
        self.hs['L[0][2][1][1][5][0]_confirm_reg_params_text'].value = 'registration parameters are set ...'
        self.xanes3D_reg_params_configured = True
        self.update_xanes3D_config()
        json.dump(self.xanes3D_config, open(self.xanes3D_save_trial_reg_config_filename, 'w'))
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
        # self.hs['L[0][2][2][1][3][0]_zshift_slider'].min = 1

        self.xanes3D_reg_done = True
        self.update_xanes3D_config()
        json.dump(self.xanes3D_config, open(self.xanes3D_save_trial_reg_config_filename, 'w'))
        self.boxes_logic()

    def L0_2_2_1_1_0_read_alignment_checkbox_change(self, a):
        if a['owner'].value:
            self.xanes3D_use_existing_reg_reviewed = True
        else:
            self.xanes3D_use_existing_reg_reviewed = False
        self.xanes3D_reg_review_done = False
        self.boxes_logic()

    def L0_2_2_1_1_1_read_alignment_button_click(self, a):
        if len(a.files[0]) != 0:
            self.xanes3D_reg_file_readed = True
            self.xanes3D_reg_review_file = os.path.abspath(a.files[0])
            if os.path.splitext(self.xanes3D_reg_review_file)[1] == '.json':
                self.xanes3D_alignment_best_match = json.load(open(self.xanes3D_reg_review_file, 'r'))
            else:
                self.xanes3D_alignment_best_match = np.genfromtxt(self.xanes3D_reg_review_file)
        else:
            self.xanes3D_reg_file_readed = False
        self.xanes3D_reg_review_done = False
        self.boxes_logic()

    def L0_2_2_1_2_0_reg_pair_slider_change(self, a):
        self.xanes3D_alignment_pair_id = a['owner'].value
        fn = self.xanes3D_save_trial_reg_filename
        f = h5py.File(fn, 'r')
        self.trial_reg[:] = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format(str(self.xanes3D_alignment_pair_id).zfill(3))][:]
        f.close()
        data_state, viewer_state = self.fiji_viewer_state(viewer_name='mask_viewer')
        if viewer_state:
            self.xanes3D_fiji_windows['mask_viewer']['ip'].setImage(ij.convert().convert(ij.dataset().create(ij.py.to_java(self.trial_reg)), ImagePlusClass))
            self.xanes3D_fiji_windows['mask_viewer']['ip'].show()
            ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
            self.xanes3D_fiji_windows['mask_viewer']['ip'].setTitle('reg pair: '+str(self.xanes3D_alignment_pair_id).zfill(3))
        else:
            ijui.show(ij.py.to_java(self.trial_reg))
            self.xanes3D_fiji_windows['mask_viewer']['ip'] = WindowManager.getCurrentImage()
            self.xanes3D_fiji_windows['mask_viewer']['fiji_id'] = WindowManager.getIDList()[-1]
            ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
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
        self.boxes_logic()

    def L0_2_2_1_3_2_record_button_click(self, a):
        self.xanes3D_alignment_best_match[str(self.xanes3D_alignment_pair_id)] = self.hs['L[0][2][2][1][3][1]_best_match_text'].value
        self.hs['L[0][2][2][1][4][0]_confirm_review_results_text'].value = str(self.xanes3D_alignment_best_match)
        self.xanes3D_reg_review_done = False
        self.boxes_logic()

    def L0_2_2_1_4_1_confirm_review_results_button_click(self, a):
        if len(self.xanes3D_alignment_best_match) != (self.hs['L[0][2][2][1][2][0]_reg_pair_slider'].max+1):
            self.hs['L[0][2][2][1][4][0]_confirm_review_results_text'].value = 'reg review is not completed yet ...'
            self.xanes3D_reg_review_done = False
        else:
            data_state, viewer_state = self.fiji_viewer_state(viewer_name='mask_viewer')
            if viewer_state:
                self.xanes3D_fiji_windows['mask_viewer']['ip'].close()
            self.hs['L[0][2][2][1][4][0]_confirm_review_results_text'].value = 'reg review is done ...'
            self.xanes3D_reg_review_done = True
            data_state, viewer_state = self.fiji_viewer_state(viewer_name='virtural_stack_preview_viewer')
            if (not data_state) | (not viewer_state):
                self.fiji_viewer_on(viewer_name='virtural_stack_preview_viewer')
            self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setSlice(1)
        self.update_xanes3D_config()
        json.dump(self.xanes3D_config, open(self.xanes3D_save_trial_reg_config_filename, 'w'))
        self.boxes_logic()

    def L0_2_2_2_1_0_align_recon_optional_slice_checkbox(self, a):
        boxes = ['align_recon_optional_slice_start_text',
                 'align_recon_optional_slice_range_slider',
                 'align_recon_optional_slice_end_text']
        if a['owner'].value:
            self.enable_disable_boxes(boxes, disabled=False, level=-1)
        else:
            self.enable_disable_boxes(boxes, disabled=True, level=-1)

    def L0_2_2_2_1_1_align_recon_optional_slice_start_text(self, a):
        if a['owner'].value <= self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_range_slider'].upper:
            self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_range_slider'].lower = a['owner'].value
        else:
            a['owner'].value = self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_range_slider'].upper

    def L0_2_2_2_1_3_align_recon_optional_slice_end_text(self, a):
        if a['owner'].value >= self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_range_slider'].lower:
            self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_range_slider'].upper = a['owner'].value
        else:
            a['owner'].value = self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_range_slider'].lower

    def L0_2_2_2_1_2_align_recon_optional_slice_range_slider(self, a):
        self.hs['L[0][2][2][2][1][1]_align_recon_optional_slice_start_text'].value = a['owner'].lower
        self.hs['L[0][2][2][2][1][3]_align_recon_optional_slice_end_text'].value = a['owner'].upper
        data_state, viewer_state = self.fiji_viewer_state(viewer_name='virtural_stack_preview_viewer')
        if (not data_state) | (not viewer_state):
            self.fiji_viewer_on(viewer_name='virtural_stack_preview_viewer')
        self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setSlice(a['owner'].lower - a['owner'].min + 1)
        self.xanes3D_fiji_windows['virtural_stack_preview_viewer']['ip'].setRoi(int(self.xanes3D_roi[2]), int(self.xanes3D_roi[0]),
                                                                                int(self.xanes3D_roi[3]-self.xanes3D_roi[2]),
                                                                                int(self.xanes3D_roi[1]-self.xanes3D_roi[0]))

    def L0_2_2_2_2_1_align_button_click(self, a):
        if self.hs['L[0][2][2][2][1][0]_align_recon_optional_slice_checkbox'].value:
            self.xanes3D_roi[4] = self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_range_slider'].lower
            self.xanes3D_roi[5] = self.hs['L[0][2][2][2][1][2]_align_recon_optional_slice_range_slider'].upper
        f = h5py.File(self.xanes3D_save_trial_reg_filename, 'r')
        recon_top_dir = f['/trial_registration/data_directory_info/recon_top_dir'][()]
        recon_path_template = f['/trial_registration/data_directory_info/recon_path_template'][()]
        # roi = f['/trial_registration/trial_reg_parameters/slice_roi'][:]
        reg_method = f['/trial_registration/trial_reg_parameters/reg_method'][()].lower()
        ref_mode = f['/trial_registration/trial_reg_parameters/reg_ref_mode'][()].lower()
        scan_id_s = f['/trial_registration/trial_reg_parameters/scan_ids'][0]
        scan_id_e = f['/trial_registration/trial_reg_parameters/scan_ids'][-1]
        fixed_scan_id = f['/trial_registration/trial_reg_parameters/fixed_scan_id'][()]
        chunk_sz = f['/trial_registration/trial_reg_parameters/chunk_sz'][()]
        eng_list = f['/trial_registration/trial_reg_parameters/eng_list'][:]
        f.close()
        roi = self.xanes3D_roi

        reg = xr.regtools(dtype='3D_XANES', method=reg_method, mode='TRANSLATION')
        reg.set_method(reg_method)
        reg.set_ref_mode(ref_mode)
        reg.cal_set_anchor(scan_id_s, scan_id_e, fixed_scan_id)
        reg.eng_list = eng_list

        reg.set_chunk_sz(chunk_sz)
        reg.set_roi(roi)
        reg.set_3D_recon_path_template(recon_path_template)
        reg.set_saving(save_path=os.path.dirname(self.xanes3D_save_trial_reg_filename),
                       fn=os.path.basename(self.xanes3D_save_trial_reg_filename))
        reg.apply_xanes3D_chunk_shift(self.xanes3D_alignment_best_match,
                                      roi[4],
                                      roi[5],
                                      trialfn=self.xanes3D_save_trial_reg_filename,
                                      savefn=self.xanes3D_save_trial_reg_filename)
        self.hs['L[0][2][2][2][2][0]_align_text'].value = 'alignment is done ...'
        self.update_xanes3D_config()
        json.dump(self.xanes3D_config, open(self.xanes3D_save_trial_reg_config_filename, 'w'))
        self.boxes_logic()
        self.xanes3D_alignment_done = True
        self.boxes_logic()

    def L0_2_3_0_1_0_visualize_view_alignment_option_dropdown(self, a):
        self.xanes3D_analysis_view_option = a['owner'].value
        if self.xanes3D_analysis_view_option == 'x-y-E':
            self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].description = 'E'
            self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].max = self.xanes3D_analysis_data_shape[0]
            self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].min = 0
            self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'].description = 'z'
            self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'].max = self.xanes3D_analysis_data_shape[1]
            self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'].min = 0
        elif self.xanes3D_analysis_view_option == 'y-z-E':
            self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].description = 'E'
            self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].max = self.xanes3D_analysis_data_shape[0]
            self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].min = 0
            self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'].description = 'x'
            self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'].max = self.xanes3D_analysis_data_shape[1]
            self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'].min = 0
        elif self.xanes3D_analysis_view_option == 'z-x-E':
            self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].description = 'E'
            self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].max = self.xanes3D_analysis_data_shape[0]
            self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].min = 0
            self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'].description = 'y'
            self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'].max = self.xanes3D_analysis_data_shape[1]
            self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'].min = 0
        elif self.xanes3D_analysis_view_option == 'x-y-z':
            self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].description = 'z'
            self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].max = self.xanes3D_analysis_data_shape[0]
            self.hs['L[0][2][3][0][1][3]_visualize_view_alignment_slice_slider'].min = 0
            self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'].description = 'E'
            self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'].max = self.xanes3D_analysis_data_shape[1]
            self.hs['L[0][2][3][0][1][2]_visualize_view_alignment_4th_dim_slider'].min = 0
        self.xanes3D_analysis_view_option_previous = self.xanes3D_analysis_view_option
        self.boxes_logic()

    def L0_2_3_0_1_3_visualize_view_alignment_slice_slider(self, a):
        self.fiji_viewer_on(viewer_name='analysis_viewer')
        self.boxes_logic()

    def L0_2_3_0_1_2_visualize_view_alignment_4th_dim_slider(self, a):
        self.xanes3D_analysis_slice = a['owner'].value
        data_state, viewer_state = self.fiji_viewer_state(viewer_name='analysis_viewer')

        if not viewer_state:
            self.fiji_viewer_on(viewer_name='analysis_viewer')
        f = h5py.File(self.xanes3D_save_trial_reg_filename, 'r')
        if self.hs['L[0][2][3][0][1][0]_visualize_view_alignment_option_dropdown'].value == 'x-y-E':
            self.xanes3D_aligned_data = 0
            self.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][:, self.xanes3D_analysis_slice, :, :]
        elif self.hs['L[0][2][3][0][1][0]_visualize_view_alignment_option_dropdown'].value == 'y-z-E':
            self.xanes3D_aligned_data = 0
            self.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][:, :, :, self.xanes3D_analysis_slice]
        elif self.hs['L[0][2][3][0][1][0]_visualize_view_alignment_option_dropdown'].value == 'z-x-E':
            self.xanes3D_aligned_data = 0
            self.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][:, :, self.xanes3D_analysis_slice, :]
        elif self.hs['L[0][2][3][0][1][0]_visualize_view_alignment_option_dropdown'].value == 'x-y-z':
            self.xanes3D_aligned_data = 0
            self.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][self.xanes3D_analysis_slice, :, :, :]
        f.close()
        self.xanes3D_fiji_aligned_data = ij.convert().convert(ij.dataset().create(ij.py.to_java(self.xanes3D_aligned_data)), ImagePlusClass)
        self.xanes3D_fiji_windows['analysis_viewer']['ip'].setImage(self.xanes3D_fiji_aligned_data)
        self.xanes3D_fiji_windows['analysis_viewer']['ip'].show()
        self.xanes3D_fiji_windows['analysis_viewer']['ip'].setTitle(f"{a['owner'].description} slice: {a['owner'].value}")
        ij.py.run_macro("""run("Collect Garbage")""")
        ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        self.boxes_logic()

    def L0_2_3_0_2_1_visualize_spec_view_mem_monitor_checkbox(self, a):
        if a['owner'].value:
            ij.py.run_macro("""run("Monitor Memory...")""")

    def L0_2_3_0_2_2_visualize_spec_view_in_roi_button(self, a):
        data_state, viewer_state = self.fiji_viewer_state(viewer_name='analysis_viewer')
        if not viewer_state:
            self.fiji_viewer_on(viewer_name='analysis_viewer')
        width = self.xanes3D_fiji_windows['analysis_viewer']['ip'].width
        height = self.xanes3D_fiji_windows['analysis_viewer']['ip'].height
        roi = [int((width-10)/2), int((height-10)/2), 10, 10]
        self.xanes3D_fiji_windows['analysis_viewer']['ip'].setRoi(roi[0], roi[1], roi[2], roi[3])
        ij.py.run_macro("""run("Plot Z-axis Profile")""")
        ij.py.run_macro("""Plot.setStyle(0, "black,none,1.0,Connected Circles")""")
        self.xanes3D_fiji_windows['analysis_viewer_z_plot_viewer']['ip'] = WindowManager.getCurrentImage()
        self.xanes3D_fiji_windows['analysis_viewer_z_plot_viewer']['fiji_id'] = WindowManager.getIDList()[-1]
        self.hs['L[0][2][3][0][2][0]_visualize_spec_view_text'].value = 'drag the roi box to check the spectrum at different locations ...'
        self.boxes_logic()

    def L0_2_3_1_1_2_analysis_energy_range_eng_s_text(self, a):
        self.xanes3D_analysis_eng_configured = False
        self.hs['L[0][2][3][1][4][0]_analysis_run_text'].description ='please check your settings before run the analysis .. '
        if a['owner'].value in ['wl', 'edge0.5']:
            if self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].value<a['owner'].value:
                self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].value = a['owner'].value
        elif a['owner'].value == 'full':
            pass
        self.boxes_logic()

    def L0_2_3_1_1_3_analysis_energy_range_eng_e_text(self, a):
        self.xanes3D_analysis_eng_configured = False
        self.hs['L[0][2][3][1][4][0]_analysis_run_text'].description ='please check your settings before run the analysis .. '
        if a['owner'].value in ['wl', 'edge0.5']:
            if self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].value>a['owner'].value:
                self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].value = a['owner'].value
        elif a['owner'].value == 'full':
            pass
        self.boxes_logic()

    def L0_2_3_1_1_0_analysis_energy_range_option_dropdown(self, a):
        self.xanes3D_analysis_eng_configured = False
        self.hs['L[0][2][3][1][4][0]_analysis_run_text'].description ='please check your settings before run the analysis .. '
        if a['owner'].value == 'wl':
            self.xanes3D_analysis_type = 'wl'
            self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].value = 0
            self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].min = 0
            self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].max = 20
            self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].description = 'wl fit eng s'
            self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].description_tooltip = 'absolute starting energy point (keV) for wl analysis. together with "wl fit eng e", it defines energy range for fitting. "fit eng s" and "fit eng e" should be roughly symmetric to whiteline peak; for "edge0.5" option, "fit eng s" and "fit eng e" should be in the range of ascendant edge range'
            self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].value = 0
            self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].max = 20
            self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].min = 0
            self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].description = 'wl fit eng e'
            self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].description_tooltip = 'absolute ending energy point (keV) for wl analysis. together with "wl fit eng s", it defines energy range for fitting. "fit eng s" and "fit eng e" should be roughly symmetric to whiteline peak; for "edge0.5" option, "fit eng s" and "fit eng e" should be in the range of ascendant edge range'
        elif a['owner'].value == 'full':
            self.xanes3D_analysis_type = 'full'
            self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].min = -1
            self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].value = -0.05
            self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].max = 0
            self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].description = 'pre edge e'
            self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].description_tooltip = 'relative ending energy point (keV) of pre-edge from edge energy for background removal'
            self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].value = 0.1
            self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].max = 1
            self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].min = 0
            self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].description = 'post edge s'
            self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].description_tooltip = 'relative starting energy point (keV) of post-edge from edge energy for normalization'
        elif a['owner'].value == 'edge0.5':
            self.xanes3D_analysis_type = 'edge0.5'
            self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].value = 0
            self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].min = 0
            self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].max = 20
            self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].description = '0.5-fit eng s'
            self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text'].description_tooltip = 'absolute starting energy point (keV) for edge-0.5 analysis. together with "0.5-fit eng e", it defines energy range for fitting. "0.5-fit eng s" and "0.5-fit eng e" should be in the range of ascendant edge range'
            self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].value = 0
            self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].max = 20
            self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].min = 0
            self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].description = '0.5-fit eng e'
            self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text'].description_tooltip = 'absolute ending energy point (keV) for edge-0.5 analysis. together with "0.5-fit eng s", it defines energy range for fitting. "0.5-fit eng s" and "0.5-fit eng e" should be in the range of ascendant edge range'
        self.boxes_logic()

    def L0_2_3_1_1_4_analysis_energy_range_confirm_button(self, a):
        if a['owner'].value == 'wl':
            self.xanes3D_analysis_wl_fit_eng_s = self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text']
            self.xanes3D_analysis_wl_fit_eng_e = self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text']
            self.xanes3D_analysis_eng_configured = True
        elif a['owner'].value == 'full':
            self.xanes3D_analysis_pre_edge_e = self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text']
            self.xanes3D_analysis_post_edge_s = self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text']
            self.xanes3D_analysis_eng_configured = True
        elif a['owner'].value == 'edge0.5':
            self.xanes3D_analysis_edge_0p5_fit_s = self.hs['L[0][2][3][1][1][2]_analysis_energy_range_eng_s_text']
            self.xanes3D_analysis_edge_0p5_fit_e = self.hs['L[0][2][3][1][1][3]_analysis_energy_range_eng_e_text']
            self.xanes3D_analysis_eng_configured = True

    def L0_2_3_1_3_0_analysis_image_use_mask_checkbox(self, a):
        if a['owner'].value:
            self.xanes3D_analysis_use_mask = True
            self.hs['L[0][2][3][0][1][0]_visualize_view_alignment_option_dropdown'].value = 'x-y-z'
            f = h5py.File(self.xanes3D_save_trial_reg_filename, 'r')
            self.xanes3D_scan_id_s = f['/trial_registration/trial_reg_parameters/scan_ids'][0]
            self.xanes3D_scan_id_e = f['/trial_registration/trial_reg_parameters/scan_ids'][-1]
            self.xanes3D_aligned_data = 0
            self.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][0, :, :, :]
            f.close()
            self.xanes3D_analysis_mask = np.ndarray(self.xanes3D_aligned_data.shape, dtype=np.int8)
            self.hs['L[0][2][3][1][3][1]_analysis_image_mask_scan_id_slider'].max = self.xanes3D_scan_id_e
            self.hs['L[0][2][3][1][3][1]_analysis_image_mask_scan_id_slider'].value = self.xanes3D_scan_id_s
            self.hs['L[0][2][3][1][3][1]_analysis_image_mask_scan_id_slider'].min = self.xanes3D_scan_id_s
        else:
            self.xanes3D_analysis_use_mask = False
        self.boxes_logic()

    def L0_2_3_1_3_1_analysis_image_mask_scan_id_slider(self, a):
        self.xanes3D_analysis_mask_scan_id = a['owner'].value
        self.xanes3D_analysis_mask_thres = self.hs['L[0][2][3][1][3][1]_analysis_image_mask_thres_slider'].value

        data_state, viewer_state = self.fiji_viewer_state(viewer_name='analysis_viewer')
        if not viewer_state:
            self.fiji_viewer_on(viewer_name='analysis_viewer')

        f = h5py.File(self.xanes3D_save_trial_reg_filename, 'r')
        self.xanes3D_aligned_data[:] = f['/registration_results/reg_results/registered_xanes3D'][self.xanes3D_analysis_mask_scan_id-self.xanes3D_scan_id_s, :, :, :]
        f.close()
        self.xanes3D_analysis_mask[:] = (self.xanes3D_aligned_data>self.xanes3D_analysis_mask_thres).astype(np.int8)[:]
        self.xanes3D_aligned_data[:] = (self.xanes3D_aligned_data*self.xanes3D_analysis_mask)[:]
        self.xanes3D_fiji_aligned_data = ij.convert().convert(ij.dataset().create(ij.py.to_java(self.xanes3D_aligned_data)), ImagePlusClass)
        self.xanes3D_fiji_windows['analysis_viewer']['ip'].setImage(self.xanes3D_fiji_aligned_data)
        self.xanes3D_fiji_windows['analysis_viewer']['ip'].show()
        self.xanes3D_fiji_windows['analysis_viewer']['ip'].setTitle(f"{a['owner'].description} slice: {a['owner'].value}")
        ij.py.run_macro("""run("Collect Garbage")""")
        ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        self.boxes_logic()

    def L0_2_3_1_3_1_analysis_image_mask_thres_slider(self, a):
        self.xanes3D_analysis_mask_thres = a['owner'].value
        self.xanes3D_analysis_mask_scan_id = self.hs['L[0][2][3][1][3][1]_analysis_image_mask_scan_id_slider'].value

        data_state, viewer_state = self.fiji_viewer_state(viewer_name='analysis_viewer')
        if not viewer_state:
            self.fiji_viewer_on(viewer_name='analysis_viewer')

        f = h5py.File(self.xanes3D_save_trial_reg_filename, 'r')
        self.xanes3D_aligned_data[:] = f['/registration_results/reg_results/registered_xanes3D'][self.xanes3D_analysis_mask_scan_id-self.xanes3D_scan_id_s, :, :, :]
        f.close()
        self.xanes3D_analysis_mask[:] = (self.xanes3D_aligned_data>self.xanes3D_analysis_mask_thres).astype(np.int8)[:]
        self.xanes3D_aligned_data[:] = (self.xanes3D_aligned_data*self.xanes3D_analysis_mask)[:]
        self.xanes3D_fiji_aligned_data = ij.convert().convert(ij.dataset().create(ij.py.to_java(self.xanes3D_aligned_data)), ImagePlusClass)
        self.xanes3D_fiji_windows['analysis_viewer']['ip'].setImage(self.xanes3D_fiji_aligned_data)
        self.xanes3D_fiji_windows['analysis_viewer']['ip'].show()
        self.xanes3D_fiji_windows['analysis_viewer']['ip'].setTitle(f"{a['owner'].description} slice: {a['owner'].value}")
        ij.py.run_macro("""run("Collect Garbage")""")
        ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
        self.boxes_logic()

    def L0_2_3_1_4_1_analysis_run_button(self, a):
        f = h5py.File(self.xanes3D_save_trial_reg_filename, 'r+')
        imgs = f['/registration_results/reg_results/registered_xanes3D'][:, 0, :, :]
        self.xanes3D_analysis_spectrum = np.ndarray([self.xanes3D_analysis_data_shape[1:]], dtype=np.float32)
        xana = xa.xanes_analysis(imgs,
                                 self.xanes3D_analysis_eng_list,
                                 self.xanes3D_analysis_edge_eng,
                                 pre_ee=self.xanes3D_analysis_pre_edge_e,
                                 post_es=self.xanes3D_analysis_post_edge_s,
                                 edge_jump_threshold=self.xanes3D_analysis_edge_jump_thres,
                                 pre_edge_threshold=self.xanes3D_analysis_edge_offset_thres)
        if self.xanes3D_analysis_type == 'wl':
            for ii in range(self.xanes3D_data_shape[0]):
                imgs[:] = imgs = f['/registration_results/reg_results/registered_xanes3D'][:, ii, :, :]
                xana.fit_whiteline(self.xanes3D_analysis_wl_eng_s, self.xanes3D_analysis_wl_eng_e)
                self.xanes3D_analysis_spectrum[ii] = xana.whiteline_pos[:]*self.xanes3D_analysis_mask[ii]
                print(ii)
        elif self.xanes3D_analysis_type in ['full', 'edge0.5']:
            for ii in range(self.xanes3D_data_shape[0]):
                xana.edge_jump()
                xana.cal_pre_edge_sd()
                xana.cal_post_edge_sd()
                xana.fit_pre_edge()
                xana.fit_post_edge()
                xana.create_edge_jump_filter(self.xanes3D_analysis_edge_jump_thres)
                xana.create_fitted_edge_filter(self.xanes3D_analysis_edge_offset_thres)
                xana.normalize_xanes(self.xanes3D_analysis_edge_eng, order=0)
                xana.fit_edge_pos(self.xanes3D_analysis_ep_eng_s, self.xanes3D_analysis_ep_eng_e)
                xana.fit_whiteline(self.xanes3D_analysis_wl_eng_s, self.xanes3D_analysis_wl_eng_r)
                # xanes3d[0, ii] = xana.normalized_spectrum[:]*mask[ii]
                # xanes3d[1, ii] = xana.edge_eng_pos[:]*mask[ii]
                # xanes3d[2, ii] = xana.whiteline_pos[:]*mask[ii]
                print(ii)
        f.close()
        self.boxes_logic()


        # fn = []
        # eng = []
        # for ii in range(scan_id_s, scan_id_e):
        # #scan_id = 12455
        #     fn.append(sorted(list(Path(recon_path_template.format(ii)).glob('*.*'))))
        #     f = h5py.File(raw_file_template.format(ii), 'r')
        #     eng.append(np.array(f['X_eng']))
        #     f.close()

        # fn = np.array(fn)
        # eng = np.array(eng)
        # print(fn.shape, eng.shape)
        # print(eng)

        # ny, nx = tifffile.imread(str(fn[0, 0])).shape
        # imgs = np.ndarray([fn.shape[0], ny, nx])

        # if make_mask:
        #     mask = np.ndarray([fn.shape[1], ny, nx])

        # if is_wl:
        #     xanes3d = np.ndarray([fn.shape[1], ny, nx])
        # else:
        #     xanes3d = np.ndarray([3, fn.shape[1], ny, nx])

        # xana = xa.xanes_analysis(imgs, eng, edge_eng, pre_ee=pre_ee, post_es=post_es,
        #                          edge_jump_threshold=edge_jump_threshold, pre_edge_threshold=pre_edge_threshold)

        # print('Your results will be saved in the file', '\"'+out_fn+'\"')


        # if make_mask:
        #     idx = xm.index_of(np.arange(scan_id_s, scan_id_e), mask_scan_id)
        #     for ii in range(fn.shape[1]):
        #         imgs[idx] = tifffile.imread(str(fn[idx, ii]))
        #         mask[ii] = (imgs[idx]>mask_threshold).astype(np.uint8)[:]

        # for ii in range(fn.shape[1]):
        #     for jj in range(fn.shape[0]):
        #         imgs[jj] = tifffile.imread(str(fn[jj, ii]))

        #     xana.spectrum[:] = imgs[:]
        #     print(xana.spectrum.shape)
        #     if is_wl:
        #         xana.fit_whiteline(wl_eng_s, wl_eng_e)
        #         xanes3d[ii] = xana.whiteline_pos[:]*mask[ii]
        #     else:
        #         xana.edge_jump()
        #         xana.cal_pre_edge_sd()
        #         xana.cal_post_edge_sd()
        #         xana.fit_pre_edge()
        #         xana.fit_post_edge()
        #         # functions work but still need to improve member variable archietecture design
        #         xana.create_edge_jump_filter(edge_jump_threshold)
        #         xana.create_fitted_edge_filter(pre_edge_threshold)
        #         xana.normalize_xanes(edge_eng, order=0)
        #         xana.fit_edge_pos(ep_eng_s, ep_eng_e)
        #         xana.fit_whiteline(wl_eng_s, wl_eng_e)
        #         xanes3d[0, ii] = xana.normalized_spectrum[:]*mask[ii]
        #         xanes3d[1, ii] = xana.edge_eng_pos[:]*mask[ii]
        #         xanes3d[2, ii] = xana.whiteline_pos[:]*mask[ii]
        #     print(ii)

        # if is_wl:
        #     xana.save_results(out_fn, dtype='3D_XANES', **{'whiteline': xanes3d})
        # else:
        #     xana.save_results(out_fn, dtype='3D_XANES', **{'normalized_spectrum': xanes3d[0, :],
        #                                                    'endg_eng_pos': xanes3d[1, :],
        #                                                    'whiteline_pos': xanes3d[2, :],
        #                                                    'mask': mask})

        # print('Your results will be saved in the file', '\"'+out_fn+'\"')
        # print('\033[04m\033[01m\033[34m     Section 4 finished     ')








