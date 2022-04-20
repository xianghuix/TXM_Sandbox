#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:37:47 2020

@author: xiao
"""
import traitlets
from tkinter import Tk, filedialog
from ipywidgets import widgets
from fnmatch import fnmatch
import os, glob, fnmatch, h5py
import numpy as np
from json import JSONEncoder
import tifffile
from tomo_recon_tools import TOMO_RECON_PARAM_DICT
    
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return JSONEncoder.default(self, obj)
            # return super(NumpyArrayEncoder, self).default(obj)

class SelectFilesButton(widgets.Button):
    """A file widget that leverages tkinter.filedialog."""

    def __init__(gui_h, option='askopenfilename',
                 text_h=None, **kwargs):
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
        gui_h.add_traits(files=traitlets.traitlets.List())
        # Create the button.
        gui_h.text_h = text_h
        gui_h.option = option
        if gui_h.option == 'askopenfilename':
            gui_h.description = "Select File"
        elif gui_h.option == 'asksaveasfilename':
            gui_h.description = "Save As File"
        elif gui_h.option == 'askdirectory':
            gui_h.description = "Choose Dir"
        gui_h.icon = "square-o"
        gui_h.style.button_color = "orange"

        # define default directory/file options
        if 'initialdir' in kwargs.keys():
            gui_h.initialdir = kwargs['initialdir']
        else:
            gui_h.initialdir = '/NSLS2/xf18id1/users/2020Q1/YUAN_YANG_Proposal_305811'
        if 'ext' in kwargs.keys():
            gui_h.ext = kwargs['ext']
        else:
            gui_h.ext = '*.h5'
        if 'initialfile' in kwargs.keys():
            gui_h.initialfile = kwargs['initialfile']
        else:
            gui_h.initialfile = '3D_trial_reg.h5'
        if 'open_filetypes' in kwargs.keys():
            gui_h.open_filetypes = kwargs['open_filetypes']
        else:
            gui_h.open_filetypes = (('json files', '*.json'), ('text files', '*.txt'))
        # gui_h.save_filetypes = (('hdf5 files', '*.h5'))
        # Set on click behavior.
        gui_h.on_click(gui_h.select_files)

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

def get_handles(hs, handle_dict_name, n):
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
        for ii in hs.keys():
            for jj in range(15):
                if f'{idx}[{jj}]_' in ii:
                        a.append(hs[ii])
    else:
        for ii in hs.keys():
            for jj in range(n):
                if f'{idx}[{jj}]_' in ii:
                    a.append(hs[ii])
    return a

# def get_decendant(hs, handle_dict_name, level=-1):
#     for ii in hs.keys():
#         if handle_dict_name in ii:
#             gui_h_handle = hs[ii]
#             gui_h_handle_name = ii
#     children_handles = []
#     children_handles.append(gui_h_handle)
#     parent_handle_label = gui_h_handle_name.split('_')[0]
#     if level == -1:
#         for ii in hs.keys():
#             if parent_handle_label in ii:
#                 children_handles.append(hs[ii])
#         actual_level = -1
#     else:
#         try:
#             actual_level = 0
#             for ii in range(level):
#                 for jj in hs.keys():
#                     if fnmatch(jj, parent_handle_label+ii*'[*]'):
#                         children_handles.append(hs[jj])
#                 actual_level += 1
#         except:
#             pass
#     return children_handles, actual_level

def get_decendant(hs, parent_handle_name, level=-1):
    parent_handle_name = parent_handle_name.split('_')[0]
    children_handles = []
    if level == -1:
        for ii in hs.keys():
            if parent_handle_name in ii:
                children_handles.append(hs[ii])
        actual_level = -1
    else:
        try:
            actual_level = 0
            for ii in range(level):
                for jj in hs.keys():
                    if fnmatch(jj, parent_handle_name+ii*'[*]'):
                        children_handles.append(hs[jj])
                actual_level += 1
        except:
            pass
    return children_handles, actual_level

def enable_disable_boxes(hs, boxes, disabled=True, level=-1):
    # boxes = list(boxes)
    for box in boxes:
        child_handles, level = get_decendant(hs, box, level=level)
        for child in child_handles:
            try:
                child.disabled = disabled
            except Exception as e:
                pass

# def gen_external_py_script(fn, **kwargs):
#     with open(fn, 'w') as f:
#         for ii in range(1, 1+len(kwargs)):
#             f.writelines(kwargs[str(ii).zfill(3)])
#             f.writelines('\n')

def check_file_availability(raw_h5_dir, scan_id=None, 
                            signature='', return_idx=False):
    if scan_id is None:
        data_files = glob.glob(os.path.join(raw_h5_dir, 
                                            '*{}*.h5'.format(signature)))
    else:
        data_files = glob.glob(os.path.join(raw_h5_dir, 
                                            '*{0}*{1}.h5'.format(signature, scan_id)))

    if len(data_files) == 0:
        return []
    else:
        if return_idx:
            ids = []
            for fn in sorted(data_files):
                ids.append(os.path.basename(fn).split('.')[-2].split('_')[-1])
            return ids
        else:
            fns = []
            for fn in sorted(data_files):
                fns.append(os.path.basename(fn))
            return fns

def get_raw_img_info(fn):
    if 'fly_scan' in os.path.basename(fn):
        scan_type = 'fly_scan'
    elif 'xanes' in os.path.basename(fn):
        scan_type = 'xanes'
    else:
        scan_type = 'undefined'

    data_info = {}
    try:
        # f = h5py.File(fn, 'r')
        with h5py.File(fn, 'r') as f:
            if scan_type == 'fly_scan':
                data_info['img_dim'] = f['/img_tomo'].shape
                data_info['theta_len'] = f['/angle'].shape[0]
                data_info['theta_min'] = np.min(f['/angle'])
                data_info['theta_max'] = np.max(f['/angle'])
                data_info['eng'] = np.max(f['/X_eng'])
            elif scan_type == 'xanes':
                data_info['img_dim'] = f['/img_xanes'].shape
                data_info['eng'] = np.max(f['/X_eng'])
            else:
                print('wrong data file')
        # f.close()
    except:
        data_info = None
    return data_info

def determine_element(eng_list):
    eng_list = scale_eng_list(eng_list)
    if ((eng_list.min()<9.669e3) & (eng_list.max()>9.669e3)):
        return 'Zn'
    elif ((eng_list.min()<8.995e3) & (eng_list.max()>8.995e3)):
        return 'Cu'
    elif ((eng_list.min()<8.353e3) & (eng_list.max()>8.353e3)):
        return 'Ni'
    elif ((eng_list.min()<7.729e3) & (eng_list.max()>7.729e3)):
        return 'Co'
    elif ((eng_list.min()<7.136e3) & (eng_list.max()>7.136e3)):
        return 'Fe'
    elif ((eng_list.min()<6.561e3) & (eng_list.max()>6.561e3)):
        return 'Mn'
    elif ((eng_list.min()<6.009e3) & (eng_list.max()>6.009e3)):
        return 'Cr'
    elif ((eng_list.min()<5.495e3) & (eng_list.max()>5.495e3)):
        return 'V'
    elif ((eng_list.min()<4.984e3) & (eng_list.max()>4.984e3)):
        return 'Ti'

def determine_fitting_energy_range(xanes_element):
    if xanes_element == 'Zn':
        xanes_analysis_edge_eng = 9.661e3
        xanes_analysis_wl_fit_eng_s = 9.664e3
        xanes_analysis_wl_fit_eng_e = 9.673e3
        xanes_analysis_pre_edge_e = 9.611e3
        xanes_analysis_post_edge_s = 9.761e3
        xanes_analysis_edge_0p5_fit_s = 9.659e3
        xanes_analysis_edge_0p5_fit_e = 9.669e3
    elif xanes_element == 'Cu':
        xanes_analysis_edge_eng = 8.989e3
        xanes_analysis_wl_fit_eng_s = 8.990e3
        xanes_analysis_wl_fit_eng_e = 9.000e3
        xanes_analysis_pre_edge_e = 8.939e3
        xanes_analysis_post_edge_s = 9.089e3
        xanes_analysis_edge_0p5_fit_s = 8.976e3
        xanes_analysis_edge_0p5_fit_e = 8.996e3
    elif xanes_element == 'Ni':
        xanes_analysis_edge_eng = 8.347e3
        xanes_analysis_wl_fit_eng_s = 8.342e3
        xanes_analysis_wl_fit_eng_e = 8.355e3
        xanes_analysis_pre_edge_e = 8.297e3
        xanes_analysis_post_edge_s = 8.447e3
        xanes_analysis_edge_0p5_fit_s = 8.340e3
        xanes_analysis_edge_0p5_fit_e = 8.352e3
    elif xanes_element == 'Co':
        xanes_analysis_edge_eng = 7.724e3
        xanes_analysis_wl_fit_eng_s = 7.725e3
        xanes_analysis_wl_fit_eng_e = 7.733e3
        xanes_analysis_pre_edge_e = 7.674e3
        xanes_analysis_post_edge_s = 7.824e3
        xanes_analysis_edge_0p5_fit_s = 7.717e3
        xanes_analysis_edge_0p5_fit_e = 7.728e3
    elif xanes_element == 'Fe':
        xanes_analysis_edge_eng = 7.126e3
        xanes_analysis_wl_fit_eng_s = 7.128e3
        xanes_analysis_wl_fit_eng_e = 7.144e3
        xanes_analysis_pre_edge_e = 7.076e3
        xanes_analysis_post_edge_s = 7.226e3
        xanes_analysis_edge_0p5_fit_s = 7.116e3
        xanes_analysis_edge_0p5_fit_e = 7.134e3
    elif xanes_element == 'Mn':
        xanes_analysis_edge_eng = 6.556e3
        xanes_analysis_wl_fit_eng_s = 6.556e3
        xanes_analysis_wl_fit_eng_e = 6.565e3
        xanes_analysis_pre_edge_e = 6.506e3
        xanes_analysis_post_edge_s = 6.656e3
        xanes_analysis_edge_0p5_fit_s = 6.547e3
        xanes_analysis_edge_0p5_fit_e = 6.560e3
    elif xanes_element == 'Cr':
        xanes_analysis_edge_eng = 6.002e3
        xanes_analysis_wl_fit_eng_s = 6.005e3
        xanes_analysis_wl_fit_eng_e = 6.012e3
        xanes_analysis_pre_edge_e = 5.952e3
        xanes_analysis_post_edge_s = 6.102e3
        xanes_analysis_edge_0p5_fit_s = 5.998e3
        xanes_analysis_edge_0p5_fit_e = 6.011e3
    elif xanes_element == 'V':
        xanes_analysis_edge_eng = 5.483e3
        xanes_analysis_wl_fit_eng_s = 5.490e3
        xanes_analysis_wl_fit_eng_e = 5.499e3
        xanes_analysis_pre_edge_e = 5.433e3
        xanes_analysis_post_edge_s = 5.583e3
        xanes_analysis_edge_0p5_fit_s = 5.474e3
        xanes_analysis_edge_0p5_fit_e = 5.487e3
    elif xanes_element == 'Ti':
        xanes_analysis_edge_eng = 4.979e3
        xanes_analysis_wl_fit_eng_s = 4.981e3
        xanes_analysis_wl_fit_eng_e = 4.986e3
        xanes_analysis_pre_edge_e = 4.929e3
        xanes_analysis_post_edge_s = 5.079e3
        xanes_analysis_edge_0p5_fit_s = 4.973e3
        xanes_analysis_edge_0p5_fit_e = 4.984e3
    return (xanes_analysis_edge_eng, xanes_analysis_wl_fit_eng_s, 
            xanes_analysis_wl_fit_eng_e, xanes_analysis_pre_edge_e,
            xanes_analysis_post_edge_s, xanes_analysis_edge_0p5_fit_s,
            xanes_analysis_edge_0p5_fit_e)

def scale_eng_list(eng_list):
    eng_list = np.array(eng_list)
    if eng_list.max()<100:
        eng_list *= 1000
    return eng_list
    
def gen_external_py_script(filename, code):
    with open(filename, 'w') as f:
        for ii in range(len(code.keys())):
            f.writelines(code[ii]+'\n')
            
def fiji_viewer_off(global_h, gui_h, viewer_name='all'):
    try:
        if viewer_name == 'all':
            for ii in global_h.WindowManager.getIDList():
                global_h.WindowManager.getImage(ii).close()
                for jj in global_h.xanes2D_fiji_windows:
                    global_h.xanes2D_fiji_windows[jj]['ip'] = None
                    global_h.xanes2D_fiji_windows[jj]['fiji_id'] = None
                for jj in global_h.xanes3D_fiji_windows:
                    global_h.xanes3D_fiji_windows[jj]['ip'] = None
                    global_h.xanes3D_fiji_windows[jj]['fiji_id'] = None
        else:
            data_state, viewer_state = fiji_viewer_state(global_h, gui_h, 
                                                         viewer_name=viewer_name)
            if viewer_state:
                if viewer_name.split('_')[0] == 'xanes2D':
                    global_h.xanes2D_fiji_windows[viewer_name]['ip'].close()
                    global_h.xanes2D_fiji_windows[viewer_name]['ip'] = None
                    global_h.xanes2D_fiji_windows[viewer_name]['fiji_id'] = None
                else:
                    global_h.xanes3D_fiji_windows[viewer_name]['ip'].close()
                    global_h.xanes3D_fiji_windows[viewer_name]['ip'] = None
                    global_h.xanes3D_fiji_windows[viewer_name]['fiji_id'] = None
    except:
        pass

def fiji_viewer_on(global_h, gui_h, viewer_name='xanes2D_raw_img_viewer'):
    if viewer_name == 'xanes3D_virtural_stack_preview_viewer':
        data_state, viewer_state = fiji_viewer_state(global_h, gui_h, 
                                                     viewer_name='xanes3D_virtural_stack_preview_viewer')
        if not viewer_state:
            gui_h.fn0 = gui_h.xanes3D_recon_3D_tiff_temp.format(gui_h.xanes3D_fixed_scan_id,
                                                              str(min(gui_h.xanes3D_available_sli_file_ids)).zfill(5))
            args = {'directory':gui_h.fn0, 'start':1}
            global_h.ij.py.run_plugin(" Open VirtualStack", args)
            global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
            global_h.xanes3D_fiji_windows['xanes3D_virtural_stack_preview_viewer'] = {'ip':global_h.WindowManager.getCurrentImage(),
                                                                          'fiji_id':global_h.WindowManager.getIDList()[-1]}
            global_h.xanes3D_fiji_windows['xanes3D_virtural_stack_preview_viewer']['ip'].setSlice(gui_h.xanes3D_fixed_sli_id-
                                                                                      gui_h.xanes3D_available_sli_file_ids[0])
    elif viewer_name == 'xanes3D_mask_viewer':
        data_state, viewer_state = fiji_viewer_state(global_h, gui_h, 
                                                     viewer_name='xanes3D_mask_viewer')
        if not viewer_state:
            cnt = 0
            for ii in range(gui_h.xanes3D_roi[4], gui_h.xanes3D_roi[5]+1):
                fn = gui_h.xanes3D_recon_3D_tiff_temp.format(gui_h.xanes3D_fixed_scan_id, str(ii).zfill(5))
                gui_h.xanes3D_img_roi[cnt, ...] = tifffile.imread(fn)[gui_h.xanes3D_roi[0]:gui_h.xanes3D_roi[1],
                                                                 gui_h.xanes3D_roi[2]:gui_h.xanes3D_roi[3]]
                cnt += 1

            global_h.ijui.show(global_h.ij.py.to_java(gui_h.xanes3D_img_roi))
            global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
            global_h.xanes3D_fiji_windows['xanes3D_mask_viewer']['ip'] = global_h.WindowManager.getCurrentImage()
            global_h.xanes3D_fiji_windows['xanes3D_mask_viewer']['fiji_id'] = global_h.WindowManager.getIDList()[-1]
            global_h.xanes3D_fiji_windows['xanes3D_mask_viewer']['ip'].setSlice(gui_h.xanes3D_fixed_sli_id-gui_h.xanes3D_roi[4])
            global_h.xanes3D_fiji_windows['xanes3D_mask_viewer']['ip'].setTitle('mask preview')
        else:
            global_h.xanes3D_fiji_windows['xanes3D_mask_viewer']['ip'].setSlice(gui_h.xanes3D_fixed_sli_id-gui_h.xanes3D_roi[4])
    elif viewer_name == 'xanes3D_review_viewer':
        data_state, viewer_state = fiji_viewer_state(global_h, gui_h, 
                                                     viewer_name='xanes3D_review_viewer')
        if not viewer_state:
            global_h.ijui.show(global_h.ij.py.to_java(gui_h.trial_reg - gui_h.trial_reg_fixed))
            global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
            global_h.xanes3D_fiji_windows['xanes3D_review_viewer']['ip'] = global_h.WindowManager.getCurrentImage()
            global_h.xanes3D_fiji_windows['xanes3D_review_viewer']['fiji_id'] = global_h.WindowManager.getIDList()[-1]
            global_h.xanes3D_fiji_windows['xanes3D_review_viewer']['ip'].setSlice(0)
            global_h.xanes3D_fiji_windows['xanes3D_review_viewer']['ip'].setTitle('reg review')
    elif viewer_name == 'xanes3D_review_manual_viewer':
        data_state, viewer_state = fiji_viewer_state(global_h, gui_h, 
                                                     viewer_name='xanes3D_review_manual_viewer')
        if not viewer_state:
            global_h.ijui.show(global_h.ij.py.to_java(gui_h.xanes3D_review_aligned_img - gui_h.trial_reg_fixed))
            global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
            global_h.xanes3D_fiji_windows['xanes3D_review_manual_viewer']['ip'] = global_h.WindowManager.getCurrentImage()
            global_h.xanes3D_fiji_windows['xanes3D_review_manual_viewer']['fiji_id'] = global_h.WindowManager.getIDList()[-1]
            global_h.xanes3D_fiji_windows['xanes3D_review_manual_viewer']['ip'].setSlice(0)
            global_h.xanes3D_fiji_windows['xanes3D_review_manual_viewer']['ip'].setTitle('reg manual review')
    elif viewer_name == 'xanes3D_analysis_viewer':
        data_state, viewer_state = fiji_viewer_state(global_h, gui_h, 
                                                     viewer_name='xanes3D_analysis_viewer')
        if not data_state:
            # f = h5py.File(gui_h.xanes_save_trial_reg_filename, 'r')
            with h5py.File(gui_h.xanes3D_save_trial_reg_filename, 'r') as f:
                if gui_h.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'].value == 'x-y-E':
                    gui_h.xanes3D_aligned_data = 0
                    gui_h.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][:, gui_h.xanes3D_analysis_slice, :, :]
                elif gui_h.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'].value == 'y-z-E':
                    gui_h.xanes3D_aligned_data = 0
                    gui_h.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][:, :, :, gui_h.xanes3D_analysis_slice]
                elif gui_h.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'].value == 'z-x-E':
                    gui_h.xanes3D_aligned_data = 0
                    gui_h.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][:, :, gui_h.xanes3D_analysis_slice, :]
                elif gui_h.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'].value == 'x-y-z':
                    gui_h.xanes3D_aligned_data = 0
                    gui_h.xanes3D_aligned_data = f['/registration_results/reg_results/registered_xanes3D'][gui_h.xanes3D_analysis_slice, :, :, :]
            # f.close()
        if not viewer_state:
            # ij.py.run_macro("""run("Monitor Memory...")""")
            global_h.ijui.show(global_h.ij.py.to_java(gui_h.xanes3D_aligned_data))
            global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
            global_h.xanes3D_fiji_windows['xanes3D_analysis_viewer']['ip'] = global_h.WindowManager.getCurrentImage()
            global_h.xanes3D_fiji_windows['xanes3D_analysis_viewer']['fiji_id'] = global_h.WindowManager.getIDList()[-1]
            global_h.xanes3D_fiji_windows['xanes3D_analysis_viewer']['ip'].setSlice(gui_h.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].value)
            global_h.xanes3D_fiji_windows['xanes3D_analysis_viewer']['ip'].setTitle('xanes3D slice view')
        else:
            global_h.xanes3D_fiji_windows['xanes3D_analysis_viewer']['ip'].setSlice(gui_h.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].value)
            gui_h.hs['L[0][2][2][2][1][1]_visualize_alignment_eng_text'].value = gui_h.xanes_analysis_eng_list[gui_h.hs['L[0][2][2][2][1][3]_visualize_view_alignment_slice_slider'].value]
    elif viewer_name == 'xanes2D_raw_img_viewer':
        data_state, viewer_state = fiji_viewer_state(global_h, gui_h, 
                                                     viewer_name='xanes2D_raw_img_viewer')
        if not viewer_state:
            global_h.ijui.show(global_h.ij.py.to_java(gui_h.xanes2D_img))
            global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
            global_h.xanes2D_fiji_windows['xanes2D_raw_img_viewer']['ip'] = global_h.WindowManager.getCurrentImage()
            global_h.xanes2D_fiji_windows['xanes2D_raw_img_viewer']['fiji_id'] = global_h.WindowManager.getIDList()[-1]
            global_h.xanes2D_fiji_windows['xanes2D_raw_img_viewer']['ip'].setSlice(0)
            global_h.xanes2D_fiji_windows['xanes2D_raw_img_viewer']['ip'].setTitle('raw image preview')
    elif viewer_name == 'xanes2D_mask_viewer':
        data_state, viewer_state = fiji_viewer_state(global_h, gui_h, 
                                                     viewer_name='xanes2D_mask_viewer')
        if not viewer_state:
            global_h.ijui.show(global_h.ij.py.to_java(gui_h.xanes2D_img[gui_h.xanes2D_eng_id_s:gui_h.xanes2D_eng_id_e+1,
                                                     gui_h.xanes2D_reg_roi[0]:gui_h.xanes2D_reg_roi[1],
                                                     gui_h.xanes2D_reg_roi[2]:gui_h.xanes2D_reg_roi[3]]))
            global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
            global_h.xanes2D_fiji_windows['xanes2D_mask_viewer']['ip'] = global_h.WindowManager.getCurrentImage()
            global_h.xanes2D_fiji_windows['xanes2D_mask_viewer']['fiji_id'] = global_h.WindowManager.getIDList()[-1]
            global_h.xanes2D_fiji_windows['xanes2D_mask_viewer']['ip'].setSlice(0)
            global_h.xanes2D_fiji_windows['xanes2D_mask_viewer']['ip'].setTitle('roi&mask preview')
    elif viewer_name == 'xanes2D_review_viewer':
        data_state, viewer_state = fiji_viewer_state(global_h, gui_h, 
                                                     viewer_name='xanes2D_review_viewer')
        if not viewer_state:
            global_h.ijui.show(global_h.ij.py.to_java(gui_h.xanes2D_review_aligned_img - gui_h.xanes2D_review_fixed_img))
            global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
            global_h.xanes2D_fiji_windows['xanes2D_review_viewer']['ip'] = global_h.WindowManager.getCurrentImage()
            global_h.xanes2D_fiji_windows['xanes2D_review_viewer']['fiji_id'] = global_h.WindowManager.getIDList()[-1]
            global_h.xanes2D_fiji_windows['xanes2D_review_viewer']['ip'].setSlice(0)
            global_h.xanes2D_fiji_windows['xanes2D_review_viewer']['ip'].setTitle('reg review')
    elif viewer_name == 'xanes2D_analysis_viewer':
        data_state, viewer_state = fiji_viewer_state(global_h, gui_h, 
                                                     viewer_name='xanes2D_analysis_viewer')
        if not viewer_state:
            global_h.ijui.show(global_h.ij.py.to_java(gui_h.xanes2D_img_roi))
            global_h.ij.py.run_macro("""run("Enhance Contrast", "saturated=0.35")""")
            global_h.xanes2D_fiji_windows['xanes2D_analysis_viewer']['ip'] = global_h.WindowManager.getCurrentImage()
            global_h.xanes2D_fiji_windows['xanes2D_analysis_viewer']['fiji_id'] = global_h.WindowManager.getIDList()[-1]
            global_h.xanes2D_fiji_windows['xanes2D_analysis_viewer']['ip'].setSlice(0)
            global_h.xanes2D_fiji_windows['xanes2D_analysis_viewer']['ip'].setTitle('visualize reg')

def fiji_viewer_state(global_h, gui_h, viewer_name='xanes3D_virtural_stack_preview_viewer'):
    if viewer_name == 'xanes3D_virtural_stack_preview_viewer':
        if ((not gui_h.xanes3D_recon_3D_tiff_temp) |
            (not gui_h.xanes3D_fixed_scan_id) |
            (not gui_h.xanes3D_available_sli_file_ids)):
            data_state = False

        else:
            data_state = True
        if global_h.WindowManager.getIDList() is None:
            viewer_state =  False
        elif not global_h.xanes3D_fiji_windows['xanes3D_virtural_stack_preview_viewer']['fiji_id'] in global_h.WindowManager.getIDList():
            viewer_state =  False
        else:
            viewer_state =  True
    elif viewer_name == 'xanes3D_mask_viewer':
        if (gui_h.xanes3D_mask is None):
            data_state = False
        elif (gui_h.xanes3D_mask.shape != (gui_h.xanes3D_roi[1]-gui_h.xanes3D_roi[0],
                                            gui_h.xanes3D_roi[3]-gui_h.xanes3D_roi[2])):
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif not global_h.xanes3D_fiji_windows['xanes3D_mask_viewer']['fiji_id'] in global_h.WindowManager.getIDList():
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == 'xanes3D_review_viewer':
        if gui_h.xanes3D_img_roi is None:
            data_state = False
        elif (gui_h.xanes3D_img_roi.shape != (gui_h.xanes3D_roi[5]-gui_h.xanes3D_roi[4]+1,
                                             gui_h.xanes3D_roi[1]-gui_h.xanes3D_roi[0],
                                             gui_h.xanes3D_roi[3]-gui_h.xanes3D_roi[2])):
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif not global_h.xanes3D_fiji_windows['xanes3D_review_viewer']['fiji_id'] in global_h.WindowManager.getIDList():
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == 'xanes3D_review_manual_viewer':
        if gui_h.xanes3D_review_aligned_img is None:
            data_state = False
        elif (gui_h.xanes3D_review_aligned_img != (gui_h.xanes3D_roi[1]-gui_h.xanes3D_roi[0],
                                                   gui_h.xanes3D_roi[3]-gui_h.xanes3D_roi[2])):
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif not global_h.xanes3D_fiji_windows['xanes3D_review_manual_viewer']['fiji_id'] in global_h.WindowManager.getIDList():
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == 'xanes3D_analysis_viewer':
        if (gui_h.xanes3D_aligned_data is None):
            data_state = False
        else:
            # f = h5py.File(gui_h.xanes_save_trial_reg_filename, 'r')
            with h5py.File(gui_h.xanes3D_save_trial_reg_filename, 'r') as f:
                data_shape = f['/registration_results/reg_results/registered_xanes3D'].shape
                if gui_h.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'].value == 'x-y-E':
                    if (data_shape[0], data_shape[2], data_shape[3]) != gui_h.xanes3D_aligned_data.shape:
                        data_state = False
                    else:
                        data_state = True
                elif gui_h.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'].value == 'y-z-E':
                    if (data_shape[0], data_shape[1], data_shape[2]) != gui_h.xanes3D_aligned_data.shape:
                        data_state = False
                    else:
                        data_state = True
                elif gui_h.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'].value == 'z-x-E':
                    if (data_shape[0], data_shape[1], data_shape[3]) != gui_h.xanes3D_aligned_data.shape:
                        data_state = False
                    else:
                        data_state = True
                elif gui_h.hs['L[0][2][2][2][3][1]_visualize_view_alignment_option_dropdown'].value == 'x-y-z':
                    if (data_shape[1], data_shape[2], data_shape[3]) != gui_h.xanes3D_aligned_data.shape:
                        data_state = False
                    else:
                        data_state = True
                else:
                    data_state = True
            # f.close()
        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif not global_h.xanes3D_fiji_windows['xanes3D_analysis_viewer']['fiji_id'] in global_h.WindowManager.getIDList():
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == 'xanes2D_raw_img_viewer':
        if (gui_h.xanes2D_img is None):
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif not global_h.xanes2D_fiji_windows['xanes2D_raw_img_viewer']['fiji_id'] in global_h.WindowManager.getIDList():
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == 'xanes2D_mask_viewer':
        if (gui_h.xanes2D_img is None):
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif not global_h.xanes2D_fiji_windows['xanes2D_mask_viewer']['fiji_id'] in global_h.WindowManager.getIDList():
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == 'xanes2D_review_viewer':
        if (gui_h.xanes2D_review_aligned_img is None):
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif not global_h.xanes2D_fiji_windows['xanes2D_review_viewer']['fiji_id'] in global_h.WindowManager.getIDList():
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == 'xanes2D_analysis_viewer':
        if (gui_h.xanes2D_img_roi is None):
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif not global_h.xanes2D_fiji_windows['xanes2D_analysis_viewer']['fiji_id'] in global_h.WindowManager.getIDList():
            viewer_state = False
        else:
            viewer_state = True
    else:
        print('Unrecognized viewer name')
        data_state = False
        viewer_state = False
    return data_state, viewer_state

def restart(gui_h, dtype='2D_XANES'):
    if dtype == '2D_XANES':
        gui_h.hs['L[0][1][0][0][1][1]_select_raw_h5_path_text'].value = 'Choose raw h5 directory ...'
        gui_h.hs['L[0][1][0][0][2][1]_select_save_trial_text'].value = 'Save trial registration as ...'
        gui_h.hs['L[0][1][0][0][3][1]_confirm_file&path_text'].value = 'Save trial registration, or go directly review registration ...'
        gui_h.hs['L[0][1][0][0][1][0]_select_raw_h5_path_button'].description = 'XANES2D File'
        gui_h.hs['L[0][1][0][0][2][0]_select_save_trial_button'].description = 'Save Reg File'
        gui_h.hs['L[0][1][0][0][1][0]_select_raw_h5_path_button'].style.button_color = "orange"
        gui_h.hs['L[0][1][0][0][2][0]_select_save_trial_button'].style.button_color = "orange"
        gui_h.hs['L[0][1][0][1][3][0]_fiji_raw_img_preview_checkbox'].value = False
        gui_h.hs['L[0][1][1][1][1][0]_fiji_mask_viewer_checkbox'].value = False
        gui_h.hs['L[0][1][0][1][3][2]_fiji_eng_id_slider'].value = 0
        gui_h.hs['L[0][1][2][0][2][0]_reg_pair_slider'].value = 0
        fiji_viewer_off(gui_h.global_h, gui_h, viewer_name='all')
        
        gui_h.xanes2D_file_configured = False
        gui_h.xanes2D_data_configured = False
        gui_h.xanes2D_roi_configured = False
        gui_h.xanes2D_reg_params_configured = False
        gui_h.xanes2D_reg_done = False
        gui_h.xanes2D_reg_review_done = False
        gui_h.xanes2D_alignment_done = False
        gui_h.xanes2D_analysis_eng_configured = False
        gui_h.xanes2D_review_read_alignment_option = False

        gui_h.xanes2D_config_alternative_flat_set = False
        gui_h.xanes2D_config_raw_img_readed = False
        gui_h.xanes2D_regparams_anchor_idx_set = False
        gui_h.xanes2D_file_reg_file_readed = False
        gui_h.xanes2D_analysis_eng_set = False

        gui_h.xanes2D_config_is_raw = False
        gui_h.xanes2D_config_is_refine = False
        gui_h.xanes2D_config_img_scalar = 1
        gui_h.xanes2D_config_use_smooth_flat = False
        gui_h.xanes2D_config_smooth_flat_sigma = 0
        gui_h.xanes2D_config_use_alternative_flat = False
        gui_h.xanes2D_config_eng_list = None

        gui_h.xanes2D_config_alternative_flat_filename = None
        gui_h.xanes2D_review_reg_best_match_filename = None

        gui_h.xanes2D_reg_use_chunk = True
        gui_h.xanes2D_reg_anchor_idx = 0
        gui_h.xanes2D_reg_roi = [0, 10, 0, 10]
        gui_h.xanes2D_reg_use_mask = True
        gui_h.xanes2D_reg_mask = None
        gui_h.xanes2D_reg_mask_dilation_width = 0
        gui_h.xanes2D_reg_mask_thres = 0
        gui_h.xanes2D_reg_use_smooth_img = False
        gui_h.xanes2D_reg_smooth_img_sigma = 5
        gui_h.xanes2D_reg_chunk_sz = None
        gui_h.xanes2D_reg_method = None
        gui_h.xanes2D_reg_ref_mode = None

        gui_h.xanes2D_visualization_auto_bc = False

        gui_h.xanes2D_img = None
        gui_h.xanes2D_img_roi = None
        gui_h.xanes2D_review_aligned_img_original = None
        gui_h.xanes2D_review_aligned_img = None
        gui_h.xanes2D_review_fixed_img = None
        gui_h.xanes2D_review_bad_shift = False
        gui_h.xanes2D_manual_xshift = 0
        gui_h.xanes2D_manual_yshift = 0
        gui_h.xanes2D_review_shift_dict = {}

        gui_h.xanes2D_eng_id_s = 0
        gui_h.xanes2D_eng_id_e = 1

        gui_h.xanes_element = None
        gui_h.xanes2D_analysis_eng_list = None
        gui_h.xanes2D_analysis_type = 'wl'
        gui_h.xanes2D_analysis_edge_eng = None
        gui_h.xanes2D_analysis_wl_fit_eng_s = None
        gui_h.xanes2D_analysis_wl_fit_eng_e = None
        gui_h.xanes2D_analysis_pre_edge_e = None
        gui_h.xanes2D_analysis_post_edge_s = None
        gui_h.xanes2D_analysis_edge_0p5_fit_s = None
        gui_h.xanes2D_analysis_edge_0p5_fit_e = None
        gui_h.xanes2D_analysis_spectrum = None
        gui_h.xanes2D_analysis_use_mask = False
        gui_h.xanes2D_analysis_mask_thres = None
        gui_h.xanes2D_analysis_mask_img_id = None
        gui_h.xanes2D_analysis_mask = 1
        gui_h.xanes2D_analysis_edge_jump_thres = 1.0
        gui_h.xanes2D_analysis_edge_offset_thres = 1.0
        gui_h.hs['L[0][1][0][0][2][0]_select_save_trial_button'].initialfile = '2D_trial_reg.h5'
        
        gui_h.global_h.xanes2D_fiji_windows = {'xanes2D_raw_img_viewer':{'ip':None,
                                                                         'fiji_id':None},
                                               'xanes2D_mask_viewer':{'ip':None,
                                                                      'fiji_id':None},
                                               'xanes2D_review_viewer':{'ip':None,
                                                                        'fiji_id':None},
                                               'xanes2D_analysis_viewer':{'ip':None,
                                                                          'fiji_id':None},
                                               'analysis_viewer_z_plot_viewer':{'ip':None,
                                                                                'fiji_id':None}}
    elif dtype == '3D_XANES':
        gui_h.hs['L[0][2][0][1][3][0]_fiji_virtural_stack_preview_checkbox'].value = False
        gui_h.hs['L[0][2][1][1][1][0]_fiji_mask_viewer_checkbox'].value = False
        gui_h.hs['L[0][2][0][0][1][1]_select_raw_h5_path_text'].value = 'Choose raw h5 directory ...'
        gui_h.hs['L[0][2][0][0][2][1]_select_recon_path_text'].value='Choose recon top directory ...'
        gui_h.hs['L[0][2][0][0][3][1]_select_save_trial_text'].value='Save trial registration as ...'
        gui_h.hs['L[0][2][0][0][4][1]_confirm_file&path_text'].value='Save trial registration, or go directly review registration ...'
        gui_h.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].description = "Raw h5 Dir"
        gui_h.hs['L[0][2][0][0][2][0]_select_recon_path_button'].description = "Recon Top Dir"
        gui_h.hs['L[0][2][0][0][3][0]_select_save_trial_button'].description = "Save Reg File"       
        gui_h.hs['L[0][2][0][0][1][0]_select_raw_h5_path_button'].style.button_color = "orange"
        gui_h.hs['L[0][2][0][0][2][0]_select_recon_path_button'].style.button_color = "orange"
        gui_h.hs['L[0][2][0][0][3][0]_select_save_trial_button'].style.button_color = "orange"  
        fiji_viewer_off(gui_h.global_h, gui_h, viewer_name='all')
        
        gui_h.xanes3D_filepath_configured = False
        gui_h.xanes3D_indices_configured = False
        gui_h.xanes3D_roi_configured = False
        gui_h.xanes3D_reg_params_configured = False
        gui_h.xanes3D_reg_done = False
        gui_h.xanes3D_reg_review_done = False
        gui_h.xanes3D_alignment_done = False
        gui_h.xanes3D_use_existing_reg_file = False
        gui_h.xanes3D_use_existing_reg_reviewed = False
        gui_h.xanes3D_reg_review_file = None
        gui_h.xanes3D_reg_use_chunk = True
        gui_h.xanes3D_reg_use_mask = True
        gui_h.xanes3D_reg_use_smooth_img = False

        gui_h.xanes3D_raw_h5_path_set = False
        gui_h.xanes3D_recon_path_set = False
        gui_h.xanes3D_save_trial_set = False
        gui_h.xanes3D_scan_id_set = False
        gui_h.xanes3D_reg_file_set = False
        gui_h.xanes3D_config_file_set = False
        gui_h.xanes3D_fixed_scan_id_set = False
        gui_h.xanes3D_fixed_sli_id_set = False
        gui_h.xanes3D_reg_file_readed = False
        gui_h.xanes3D_analysis_eng_configured = False

        gui_h.xanes3D_review_shift_dict = {}
        gui_h.xanes3D_reg_mask_dilation_width = 0
        gui_h.xanes3D_reg_mask_thres = 0
        gui_h.xanes3D_img_roi = None
        gui_h.xanes3D_roi = [0, 10, 0, 10, 0, 10]
        gui_h.xanes3D_mask = None
        gui_h.xanes3D_aligned_data = None
        gui_h.xanes3D_analysis_slice = 0
        gui_h.xanes3D_raw_3D_h5_top_dir = None
        gui_h.xanes3D_recon_3D_top_dir = None
        gui_h.xanes3D_save_trial_reg_filename = None
        gui_h.xanes3D_save_trial_reg_config_filename = None
        gui_h.xanes3D_save_trial_reg_config_filename_original = None
        gui_h.xanes3D_raw_3D_h5_temp = None
        gui_h.xanes3D_available_raw_ids = None
        gui_h.xanes3D_recon_3D_tiff_temp = None
        gui_h.xanes3D_recon_3D_dir_temp = None
        gui_h.xanes3D_reg_best_match_filename = None
        gui_h.xanes3D_available_recon_ids = None
        gui_h.xanes3D_available_sli_file_ids = None
        gui_h.xanes3D_fixed_scan_id = None
        gui_h.xanes3D_scan_id_s = None
        gui_h.xanes3D_scan_id_e = None
        gui_h.xanes3D_fixed_sli_id = None
        gui_h.xanes3D_reg_sli_search_half_width = None
        gui_h.xanes3D_reg_chunk_sz = None
        gui_h.xanes3D_reg_smooth_sigma = 0
        gui_h.xanes3D_reg_method = None
        gui_h.xanes3D_reg_ref_mode = None
        gui_h.xanes3D_review_bad_shift = False
        gui_h.xanes3D_visualization_viewer_option = 'fiji'
        gui_h.xanes3D_analysis_view_option = 'x-y-E'
        gui_h.xanes_element = None
        gui_h.xanes3D_analysis_type = 'wl'
        gui_h.xanes3D_analysis_edge_eng = None
        gui_h.xanes3D_analysis_wl_fit_eng_s = None
        gui_h.xanes3D_analysis_wl_fit_eng_e = None
        gui_h.xanes3D_analysis_pre_edge_e = None
        gui_h.xanes3D_analysis_post_edge_s = None
        gui_h.xanes3D_analysis_edge_0p5_fit_s = None
        gui_h.xanes3D_analysis_edge_0p5_fit_e = None
        gui_h.xanes3D_analysis_spectrum = None
        gui_h.xanes3D_analysis_use_mask = False
        gui_h.xanes3D_analysis_mask_thres = None
        gui_h.xanes3D_analysis_mask_scan_id = None
        gui_h.xanes3D_analysis_mask = 1
        gui_h.xanes3D_analysis_edge_jump_thres = 1.0
        gui_h.xanes3D_analysis_edge_offset_thres =1.0
        gui_h.xanes3D_analysis_use_flt_spec = False
        gui_h.hs['L[0][2][0][0][3][0]_select_save_trial_button'].initialfile = '3D_trial_reg.h5'
        
        gui_h.global_h.xanes3D_fiji_windows = {'xanes3D_virtural_stack_preview_viewer':{'ip':None,
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
    elif dtype == 'TOMO':
        gui_h.hs["L[0][0][0][0][1][1]_select_raw_h5_top_dir_text"].value="Choose raw h5 top dir ..."
        gui_h.hs["L[0][0][0][0][2][1]_select_save_recon_dir_text"].value="Select top directory where recon subdirectories are saved..."
        gui_h.hs["L[0][0][0][0][3][1]_select_save_data_center_dir_text"].value="Select top directory where data_center will be created..."
        gui_h.hs["L[0][0][0][0][4][1]_select_save_debug_dir_text"].value="Debug is disabled..."
        gui_h.hs["L[0][0][0][0][5][1]_confirm_file&path_text"].value="After setting directories, confirm to proceed ..."
        gui_h.hs["L[0][0][0][0][1][0]_select_raw_h5_top_dir_button"].description = "Raw Top Dir"
        gui_h.hs["L[0][0][0][0][2][0]_select_save_recon_dir_button"].description = "Save Rec File"
        gui_h.hs["L[0][0][0][0][3][0]_select_save_data_center_dir_button"].description = "Save Data_Center"
        gui_h.hs["L[0][0][0][0][4][0]_select_save_debug_dir_button"].description = "Save Debug Dir"
        gui_h.hs["L[0][0][0][0][4][2]_save_debug_checkbox"].value = False
        gui_h.hs["L[0][0][0][0][1][0]_select_raw_h5_top_dir_button"].style.button_color = "orange"
        gui_h.hs["L[0][0][0][0][2][0]_select_save_recon_dir_button"].style.button_color = "orange"
        gui_h.hs["L[0][0][0][0][3][0]_select_save_data_center_dir_button"].style.button_color = "orange"
        gui_h.hs["L[0][0][0][0][4][0]_select_save_debug_dir_button"].style.button_color = "orange"
        gui_h.tomo_raw_data_top_dir_set = False
        gui_h.tomo_recon_path_set = False
        gui_h.tomo_data_center_path_set = False
        gui_h.tomo_debug_path_set = False

        gui_h.tomo_filepath_configured = False
        gui_h.tomo_data_configured = False

        gui_h.tomo_left_box_selected_flt = "phase retrieval"
        gui_h.tomo_selected_alg = "gridrec"

        gui_h.tomo_recon_param_dict = TOMO_RECON_PARAM_DICT

        gui_h.tomo_raw_data_top_dir = None
        gui_h.tomo_raw_data_file_template = None
        gui_h.tomo_data_center_path = None
        gui_h.tomo_recon_top_dir = None
        gui_h.tomo_debug_top_dir = None
        gui_h.tomo_cen_list_file = None
        gui_h.tomo_alt_flat_file = None
        gui_h.tomo_alt_dark_file = None
        gui_h.tomo_wedge_ang_auto_det_ref_fn = None

        gui_h.tomo_recon_type = "Trial Center"
        gui_h.tomo_use_debug = False
        gui_h.tomo_use_alt_flat = False
        gui_h.tomo_use_alt_dark = False
        gui_h.tomo_use_fake_flat = False
        gui_h.tomo_use_fake_dark = False
        gui_h.tomo_use_rm_zinger = False
        gui_h.tomo_use_mask = True
        gui_h.tomo_is_wedge = False
        gui_h.tomo_use_wedge_ang_auto_det = False

        gui_h.tomo_right_filter_dict = {0:{}}

        gui_h.tomo_scan_id = 0
        gui_h.tomo_ds_ratio=1
        gui_h.tomo_rot_cen = 1280
        gui_h.tomo_cen_win_s = 1240
        gui_h.tomo_cen_win_w = 80
        gui_h.tomo_fake_flat_val = 1e4
        gui_h.tomo_fake_dark_val = 100
        gui_h.tomo_sli_s = 1280
        gui_h.tomo_sli_e = 1300
        gui_h.tomo_chunk_sz = 200
        gui_h.tomo_margin = 15
        gui_h.tomo_zinger_val = 500
        gui_h.tomo_mask_ratio = 1
        gui_h.tomo_wedge_blankat = 90
        gui_h.tomo_wedge_missing_s = 500
        gui_h.tomo_wedge_missing_e = 600
        gui_h.tomo_wedge_ang_auto_det_thres = 500

        gui_h.alg_param_dict = {}
    
def read_config_from_reg_file(gui_h, dtype='2D_XANES'):
    if dtype == '2D_XANES':
        pass
    elif dtype == '3D_XANES':
        # f = h5py.File(gui_h.xanes_save_trial_reg_filename, 'r')
        with h5py.File(gui_h.xanes3D_save_trial_reg_filename, 'r') as f:
            gui_h.xanes3D_recon_3D_tiff_temp = f['/trial_registration/data_directory_info/recon_path_template'][()]
            gui_h.xanes3D_raw_3D_h5_top_dir = f['/trial_registration/data_directory_info/raw_h5_top_dir'][()]
            gui_h.xanes3D_recon_3D_top_dir = f['/trial_registration/data_directory_info/recon_top_dir'][()]
            gui_h.xanes3D_fixed_scan_id = int(f['/trial_registration/trial_reg_parameters/fixed_scan_id'][()])
            gui_h.xanes3D_fixed_sli_id = int(f['/trial_registration/trial_reg_parameters/fixed_slice'][()])
            gui_h.xanes3D_roi = f['/trial_registration/trial_reg_parameters/slice_roi'][:].tolist()
            gui_h.xanes3D_scan_id_s = int(f['/trial_registration/trial_reg_parameters/scan_ids'][0])
            gui_h.xanes3D_scan_id_e = int(f['/trial_registration/trial_reg_parameters/scan_ids'][-1])
            gui_h.xanes3D_reg_sli_search_half_width = int(f['/trial_registration/trial_reg_parameters/sli_search_half_range'][()])
            gui_h.xanes3D_reg_method = f['/trial_registration/trial_reg_parameters/reg_method'][()]
            gui_h.xanes3D_reg_ref_mode = f['/trial_registration/trial_reg_parameters/reg_ref_mode'][()]
            gui_h.xanes3D_reg_use_smooth_img = bool(f['/trial_registration/trial_reg_parameters/use_smooth_img'][()])
            gui_h.xanes3D_reg_smooth_sigma = int(f['/trial_registration/trial_reg_parameters/img_smooth_sigma'][()])
            gui_h.xanes3D_reg_use_chunk = bool(f['/trial_registration/trial_reg_parameters/use_chunk'][()])
            gui_h.xanes3D_reg_chunk_sz = int(f['/trial_registration/trial_reg_parameters/chunk_sz'][()])
            gui_h.xanes3D_reg_use_mask = bool(f['/trial_registration/trial_reg_parameters/use_mask'][()])
            gui_h.xanes3D_alignment_pairs = f['/trial_registration/trial_reg_parameters/alignment_pairs'][:]
            gui_h.trial_reg = f['/trial_registration/trial_reg_results/{0}/trial_reg_img{0}'.format('000')][:]
            gui_h.trial_reg_fixed = f['/trial_registration/trial_reg_results/{0}/trial_fixed_img{0}'.format('000')][:]
            gui_h.xanes3D_review_aligned_img = np.ndarray(gui_h.trial_reg[0].shape)
        # f.close()
    else:
        pass    