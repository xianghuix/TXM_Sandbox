#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:37:47 2020

@author: xiao
"""
import os, glob, h5py, numpy as np
from pathlib import Path

import traitlets
from tkinter import Tk, filedialog, Toplevel, Label, Button
from ipywidgets import widgets
from fnmatch import fnmatch
from json import JSONEncoder
import tifffile, json
from datetime import datetime
import types

from ..utils.tomo_recon_tools import TOMO_RECON_PARAM_DICT, rm_redundant
from ..dicts.config_dict import (
    IO_TOMO_CFG_DEFAULT,
    IO_XANES2D_CFG_DEFAULT,
    IO_XANES3D_CFG_DEFAULT,
)
from ..dicts import customized_struct_dict as dat_dict


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


class SelectFilesButton(widgets.Button):
    """A file widget that leverages tkinter.filedialog."""

    def __init__(gui_h, option="askopenfilename", text_h=None, **kwargs):
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
        if gui_h.option == "askopenfilename":
            gui_h.description = "Select File"
        elif gui_h.option == "asksaveasfilename":
            gui_h.description = "Save As File"
        elif gui_h.option == "askdirectory":
            gui_h.description = "Choose Dir"
        gui_h.icon = "square-o"
        gui_h.style.button_color = "orange"

        # define default directory/file options
        if "initialdir" in kwargs.keys():
            gui_h.initialdir = kwargs["initialdir"]
        else:
            gui_h.initialdir = str(Path.resolve(Path.cwd()))
        if "ext" in kwargs.keys():
            gui_h.ext = kwargs["ext"]
        else:
            gui_h.ext = "*.h5"
        if "initialfile" in kwargs.keys():
            gui_h.initialfile = kwargs["initialfile"]
        else:
            gui_h.initialfile = "3D_trial_reg.h5"
        if "open_filetypes" in kwargs.keys():
            gui_h.open_filetypes = kwargs["open_filetypes"]
        else:
            gui_h.open_filetypes = (("json files", "*.json"), ("text files",
                                                               "*.txt"))
        if "save_filetypes" in kwargs.keys():
            gui_h.save_filetypes = kwargs["save_filetypes"]
        else:
            gui_h.save_filetypes = (("h5 files", ["*.h5", "*.hdf"]),)
        if "defaultextension" in kwargs.keys():
            gui_h.defaultextension = kwargs["defaultextension"]
        else:
            gui_h.defaultextension = '.h5'
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
        root.call("wm", "attributes", ".", "-topmost", True)
        root.attributes("-topmost", True)
        # List of selected fileswill be set to b.value

        # if Path(b.initialfile).suffix not in b.save_filetypes

        if b.option == "askopenfilename":
            files = filedialog.askopenfilename(
                initialdir=b.initialdir,
                defaultextension="*.json",
                filetypes=b.open_filetypes,
            )
        elif b.option == "askdirectory":
            files = filedialog.askdirectory(initialdir=b.initialdir)
        elif b.option == "asksaveasfilename":
            files = filedialog.asksaveasfilename(
                initialdir=b.initialdir,
                initialfile=b.initialfile,
                filetypes=b.save_filetypes,
                defaultextension=b.defaultextension,
            )

        if b.text_h is None:
            if len(files) == 0:
                b.files = [""]
                if b.option == "askopenfilename":
                    b.description = "Select File"
                elif b.option == "asksaveasfilename":
                    b.description = "Save As File"
                elif b.option == "askdirectory":
                    b.description = "Choose Dir"
                b.icon = "square-o"
                b.style.button_color = "orange"
            else:
                b.files = [files]
                if b.option == "askopenfilename":
                    b.description = "File Selected"
                    b.initialdir = str(Path.resolve(Path(files).parent))
                    b.initialfile = Path(files).stem
                elif b.option == "asksaveasfilename":
                    b.description = "Filename Chosen"
                    b.initialdir = str(Path.resolve(Path(files).parent))
                    b.initialfile = Path(files).stem
                elif b.option == "askdirectory":
                    b.description = "Dir Selected"
                    b.initialdir = str(Path.resolve(Path(files)))
                b.icon = "check-square-o"
                b.style.button_color = "lightgreen"
        else:
            if len(files) == 0:
                b.files = [""]
                if b.option == "askopenfilename":
                    b.description = "Select File"
                    b.text_h.value = "select file ..."
                elif b.option == "asksaveasfilename":
                    b.description = "Save As File"
                    b.text_h.value = "select a path and specify a file name ..."
                elif b.option == "askdirectory":
                    b.description = "Choose Dir"
                    b.text_h.value = "select a directory ..."
                b.icon = "square-o"
                b.style.button_color = "orange"
            else:
                b.files = [files]
                if b.option == "askopenfilename":
                    b.description = "File Selected"
                    b.initialdir = str(Path.resolve(Path(files).parent))
                    b.initialfile = Path(files).stem
                elif b.option == "asksaveasfilename":
                    b.description = "Filename Chosen"
                    b.initialdir = str(Path.resolve(Path(files).parent))
                    b.initialfile = Path(files).stem
                elif b.option == "askdirectory":
                    b.description = "Dir Selected"
                    b.initialdir = Path.resolve(Path(files))
                b.icon = "check-square-o"
                b.style.button_color = "lightgreen"
                b.text_h.value = str(Path.resolve(Path(b.files[0])))


class msgbox:
    closed = False

    @classmethod
    def messagebox(cls, msg=""):
        cls.closed = False
        root = Tk()
        root.withdraw()

        def on_close():
            root.destroy()
            cls.closed = True
            print(cls.closed)

        toplevel = Toplevel(root)
        toplevel.protocol("WM_DELETE_WINDOW", on_close)

        toplevel.title("Message")
        toplevel.geometry("300x100")
        toplevel.wm_resizable(True, False)

        l1 = Label(toplevel, image="::tk::icons::information")
        l1.grid(row=0, column=0, pady=(7, 0), padx=(10, 30), sticky="e")
        l2 = Label(toplevel, text=msg)
        l2.grid(row=0, column=1, columnspan=3, pady=(7, 10), sticky="w")

        b1 = Button(toplevel, text="Close", command=on_close, width=10)
        b1.grid(row=1, column=1, padx=(2, 35), sticky="e")

        toplevel.mainloop()


def update_global_cwd(global_h, new_cwd):
    global_h.cwd = new_cwd
    update_json_content(
        global_h.GUI_cfg_file,
        {"cwd": new_cwd},
    )


def check_file_availability(raw_h5_dir,
                            scan_id=None,
                            signature="",
                            return_idx=False):
    if scan_id is None:
        # data_files = glob.glob(os.path.join(raw_h5_dir, signature.format("*")))
        data_files = glob.glob(
            str(Path(raw_h5_dir).joinpath(signature.format("*"))))
    else:
        # data_files = glob.glob(
        #     os.path.join(raw_h5_dir, signature.format(scan_id)))
        data_files = glob.glob(
            str(Path(raw_h5_dir).joinpath(Path(signature.format(scan_id)))))

    if len(data_files) == 0:
        return []
    else:
        if return_idx:
            ids = []
            for fn in sorted(data_files):
                # ids.append(os.path.basename(fn).split(".")[-2].split("_")[-1])
                ids.append(Path(fn).name.split(".")[-2].split("_")[-1])
            return ids
        else:
            fns = []
            for fn in sorted(data_files):
                # fns.append(os.path.basename(fn))
                fns.append(Path(fn).name)
            return fns


def create_widget(wtype, layout, **kwargs):
    if wtype == "VBox":
        hs = widgets.VBox(**kwargs)
        hs.layout = layout
    elif wtype == "HBox":
        hs = widgets.HBox(**kwargs)
        hs.layout = layout
    elif wtype == "SelectFilesButton":
        hs = SelectFilesButton(**kwargs)
        hs.layout = layout
    elif wtype == "Button":
        hs = widgets.Button(**kwargs)
        hs.layout = layout
    elif wtype == "Dropdown":
        hs = widgets.Dropdown(**kwargs)
        hs.layout = layout
    elif wtype == "BoundedIntText":
        hs = widgets.BoundedIntText(**kwargs)
        hs.layout = layout
    elif wtype == "BoundedFloatText":
        hs = widgets.BoundedFloatText(**kwargs)
        hs.layout = layout
    elif wtype == "IntRangeSlider":
        hs = widgets.IntRangeSlider(**kwargs)
        hs.layout = layout
    elif wtype == "Checkbox":
        hs = widgets.Checkbox(**kwargs)
        hs.layout = layout
    elif wtype == "FloatSlider":
        hs = widgets.FloatSlider(**kwargs)
        hs.layout = layout
    elif wtype == "IntSlider":
        hs = widgets.IntSlider(**kwargs)
        hs.layout = layout
    elif wtype == "IntProgress":
        hs = widgets.IntProgress(**kwargs)
        hs.layout = layout
    elif wtype == "Text":
        hs = widgets.Text(**kwargs)
        hs.layout = layout
    elif wtype == "IntText":
        hs = widgets.IntText(**kwargs)
        hs.layout = layout
    elif wtype == "FloatText":
        hs = widgets.FloatText(**kwargs)
        hs.layout = layout
    elif wtype == "ToggleButtons":
        hs = widgets.ToggleButtons(**kwargs)
        hs.layout = layout
    elif wtype == "SelectMultiple":
        hs = widgets.SelectMultiple(**kwargs)
        hs.layout = layout
    return hs


def determine_element(eng_list):
    eng_list = scale_eng_list(eng_list)
    if (eng_list.min() < 9.669e3) & (eng_list.max() > 9.669e3):
        return "Zn"
    elif (eng_list.min() < 8.995e3) & (eng_list.max() > 8.995e3):
        return "Cu"
    elif (eng_list.min() < 8.353e3) & (eng_list.max() > 8.353e3):
        return "Ni"
    elif (eng_list.min() < 7.729e3) & (eng_list.max() > 7.729e3):
        return "Co"
    elif (eng_list.min() < 7.136e3) & (eng_list.max() > 7.136e3):
        return "Fe"
    elif (eng_list.min() < 6.561e3) & (eng_list.max() > 6.561e3):
        return "Mn"
    elif (eng_list.min() < 6.009e3) & (eng_list.max() > 6.009e3):
        return "Cr"
    elif (eng_list.min() < 5.495e3) & (eng_list.max() > 5.495e3):
        return "V"
    elif (eng_list.min() < 4.984e3) & (eng_list.max() > 4.984e3):
        return "Ti"
    else:
        return None


def determine_fitting_energy_range(xanes_element):
    if xanes_element.upper() == "ZN":
        xanes_analysis_edge_eng = 9.661e3
        xanes_analysis_wl_fit_eng_s = 9.664e3
        xanes_analysis_wl_fit_eng_e = 9.673e3
        xanes_analysis_pre_edge_e = -50
        xanes_analysis_post_edge_s = 100
        xanes_analysis_edge_0p5_fit_s = 9.659e3
        xanes_analysis_edge_0p5_fit_e = 9.669e3
    elif xanes_element.upper() == "CU":
        xanes_analysis_edge_eng = 8.989e3
        xanes_analysis_wl_fit_eng_s = 8.990e3
        xanes_analysis_wl_fit_eng_e = 9.000e3
        xanes_analysis_pre_edge_e = -50
        xanes_analysis_post_edge_s = 100
        xanes_analysis_edge_0p5_fit_s = 8.976e3
        xanes_analysis_edge_0p5_fit_e = 8.996e3
    elif xanes_element.upper() == "NI":
        xanes_analysis_edge_eng = 8.347e3
        xanes_analysis_wl_fit_eng_s = 8.347e3
        xanes_analysis_wl_fit_eng_e = 8.354e3
        xanes_analysis_pre_edge_e = -50
        xanes_analysis_post_edge_s = 100
        xanes_analysis_edge_0p5_fit_s = 8.339e3
        xanes_analysis_edge_0p5_fit_e = 8.348e3
    elif xanes_element.upper() == "CO":
        xanes_analysis_edge_eng = 7.724e3
        xanes_analysis_wl_fit_eng_s = 7.724e3
        xanes_analysis_wl_fit_eng_e = 7.730e3
        xanes_analysis_pre_edge_e = -50
        xanes_analysis_post_edge_s = 100
        xanes_analysis_edge_0p5_fit_s = 7.717e3
        xanes_analysis_edge_0p5_fit_e = 7.726e3
    elif xanes_element.upper() == "FE":
        xanes_analysis_edge_eng = 7.126e3
        xanes_analysis_wl_fit_eng_s = 7.128e3
        xanes_analysis_wl_fit_eng_e = 7.144e3
        xanes_analysis_pre_edge_e = -50
        xanes_analysis_post_edge_s = 100
        xanes_analysis_edge_0p5_fit_s = 7.116e3
        xanes_analysis_edge_0p5_fit_e = 7.134e3
    elif xanes_element.upper() == "MN":
        xanes_analysis_edge_eng = 6.556e3
        xanes_analysis_wl_fit_eng_s = 6.556e3
        xanes_analysis_wl_fit_eng_e = 6.565e3
        xanes_analysis_pre_edge_e = -50
        xanes_analysis_post_edge_s = 100
        xanes_analysis_edge_0p5_fit_s = 6.547e3
        xanes_analysis_edge_0p5_fit_e = 6.560e3
    elif xanes_element.upper() == "CR":
        xanes_analysis_edge_eng = 6.002e3
        xanes_analysis_wl_fit_eng_s = 6.005e3
        xanes_analysis_wl_fit_eng_e = 6.012e3
        xanes_analysis_pre_edge_e = -50
        xanes_analysis_post_edge_s = 100
        xanes_analysis_edge_0p5_fit_s = 5.998e3
        xanes_analysis_edge_0p5_fit_e = 6.011e3
    elif xanes_element.upper() == "V":
        xanes_analysis_edge_eng = 5.483e3
        xanes_analysis_wl_fit_eng_s = 5.490e3
        xanes_analysis_wl_fit_eng_e = 5.499e3
        xanes_analysis_pre_edge_e = -50
        xanes_analysis_post_edge_s = 100
        xanes_analysis_edge_0p5_fit_s = 5.474e3
        xanes_analysis_edge_0p5_fit_e = 5.487e3
    elif xanes_element.upper() == "TI":
        xanes_analysis_edge_eng = 4.979e3
        xanes_analysis_wl_fit_eng_s = 4.981e3
        xanes_analysis_wl_fit_eng_e = 4.986e3
        xanes_analysis_pre_edge_e = -50
        xanes_analysis_post_edge_s = 100
        xanes_analysis_edge_0p5_fit_s = 4.973e3
        xanes_analysis_edge_0p5_fit_e = 4.984e3
    return (
        xanes_analysis_edge_eng,
        xanes_analysis_wl_fit_eng_s,
        xanes_analysis_wl_fit_eng_e,
        xanes_analysis_pre_edge_e,
        xanes_analysis_post_edge_s,
        xanes_analysis_edge_0p5_fit_s,
        xanes_analysis_edge_0p5_fit_e,
    ) or None


def enable_disable_boxes(hs, boxes, disabled=True, level=-1):
    if level < 0:
        level = 1000000
    for box in boxes:
        child_handles = get_decendant(hs[box], level=level)
        for child in child_handles:
            try:
                child.disabled = disabled
            except Exception as e:
                print(box, child)
                print(e)


def get_fiji_win_by_viewer_name(global_h, viewer_name):
    if "xanes2D" in viewer_name:
        return global_h.xanes2D_fiji_windows
    elif "xanes3D" in viewer_name:
        return global_h.xanes3D_fiji_windows
    elif "tomo" in viewer_name:
        return global_h.tomo_fiji_windows
    elif "xanes_data" in viewer_name:
        return global_h.xanes_ana_fiji_windows
    elif "xanes_pp" in viewer_name:
        return global_h.xanes_ana_fiji_windows


def fiji_viewer_off(global_h, gui_h=None, viewer_name="all"):
    try:
        if (viewer_name == "all") and (global_h.WindowManager.getIDList()
                                       is not None):
            for ii in global_h.WindowManager.getIDList():
                global_h.WindowManager.getImage(ii).close()
                for jj in global_h.xanes2D_fiji_windows.keys():
                    global_h.xanes2D_fiji_windows[jj]["ip"] = None
                    global_h.xanes2D_fiji_windows[jj]["fiji_id"] = None
                for jj in global_h.xanes3D_fiji_windows.keys():
                    global_h.xanes3D_fiji_windows[jj]["ip"] = None
                    global_h.xanes3D_fiji_windows[jj]["fiji_id"] = None
                for jj in global_h.tomo_fiji_windows.keys():
                    global_h.tomo_fiji_windows[jj]["ip"] = None
                    global_h.tomo_fiji_windows[jj]["fiji_id"] = None
                for jj in global_h.xanes_ana_fiji_windows.keys():
                    global_h.xanes_ana_fiji_windows[jj]["ip"] = None
                    global_h.xanes_ana_fiji_windows[jj]["fiji_id"] = None
        else:
            win = get_fiji_win_by_viewer_name(global_h, viewer_name)
            data_state, viewer_state = fiji_viewer_state(
                global_h, gui_h, viewer_name=viewer_name)
            if viewer_state:
                win[viewer_name]["ip"].close()
                win[viewer_name]["ip"] = None
                win[viewer_name]["fiji_id"] = None
    except:
        print("something wrong during closing", viewer_name)


def fiji_viewer_on(global_h,
                   gui_h,
                   viewer_name="xanes2D_raw_img_viewer",
                   win_name=None,
                   data=None,
                   idx=0):
    win = get_fiji_win_by_viewer_name(global_h, viewer_name)
    data_state, viewer_state = fiji_viewer_state(global_h,
                                                 gui_h,
                                                 viewer_name=viewer_name)
    if viewer_name == "xanes3D_virtural_stack_preview_viewer":
        if not viewer_state:
            gui_h.fn0 = gui_h.xanes_recon_3D_tiff_temp.format(
                gui_h.xanes_available_raw_ids[gui_h.xanes_fixed_scan_id -
                                              gui_h.xanes_scan_id_s],
                str(min(gui_h.xanes_available_sli_file_ids)).zfill(5),
            )
            args = {"directory": gui_h.fn0, "start": 1}
            global_h.ij.py.run_plugin(" Open VirtualStack", args)
            global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")
            global_h.WindowManager.getCurrentImage().setSlice(
                gui_h.xanes_fixed_sli_id -
                gui_h.xanes_available_sli_file_ids[0])
    elif viewer_name == "xanes3D_mask_viewer":
        if not viewer_state:
            cnt = 0
            for ii in range(gui_h.xanes_roi[4], gui_h.xanes_roi[5] + 1):
                fn = gui_h.xanes_recon_3D_tiff_temp.format(
                    gui_h.xanes_available_raw_ids[gui_h.xanes_fixed_scan_id -
                                                  gui_h.xanes_scan_id_s],
                    str(ii).zfill(5),
                )
                gui_h.xanes_img_roi[cnt, ...] = tifffile.imread(
                    fn)[gui_h.xanes_roi[0]:gui_h.xanes_roi[1],
                        gui_h.xanes_roi[2]:gui_h.xanes_roi[3], ]
                cnt += 1

            global_h.ijui.show(global_h.ij.py.to_java(gui_h.xanes_img_roi))
            global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")
            global_h.WindowManager.getCurrentImage().setSlice(
                gui_h.xanes_fixed_sli_id - gui_h.xanes_roi[4])
        else:
            win["xanes3D_mask_viewer"]["ip"].setSlice(
                gui_h.xanes_fixed_sli_id - gui_h.xanes_roi[4])
    elif viewer_name == "xanes3D_review_viewer":
        if not viewer_state:
            global_h.ijui.show(
                global_h.ij.py.to_java(gui_h.trial_reg -
                                       gui_h.trial_reg_fixed))
            global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")
            global_h.WindowManager.getCurrentImage().setSlice(0)
    elif viewer_name == "xanes3D_review_manual_viewer":
        if not viewer_state:
            global_h.ijui.show(
                global_h.ij.py.to_java(gui_h.xanes_review_aligned_img -
                                       gui_h.trial_reg_fixed))
            global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")
            global_h.WindowManager.getCurrentImage().setSlice(0)
    elif viewer_name == "xanes3D_analysis_viewer":
        if not data_state:
            with h5py.File(gui_h.xanes_save_trial_reg_filename, "r") as f:
                if gui_h.hs["VisImgViewAlignOptn drpdn"].value == "x-y-E":
                    gui_h.xanes_aligned_data = 0
                    gui_h.xanes_aligned_data = f[
                        "/registration_results/reg_results/registered_xanes3D"][:, gui_h.hs[
                            "VisImgViewAlign4thDim sldr"].value, :, :]
                elif gui_h.hs["VisImgViewAlignOptn drpdn"].value == "y-z-E":
                    gui_h.xanes_aligned_data = 0
                    gui_h.xanes_aligned_data = f[
                        "/registration_results/reg_results/registered_xanes3D"][:, :, :, gui_h.hs[
                            "VisImgViewAlign4thDim sldr"].value]
                elif gui_h.hs["VisImgViewAlignOptn drpdn"].value == "z-x-E":
                    gui_h.xanes_aligned_data = 0
                    gui_h.xanes_aligned_data = f[
                        "/registration_results/reg_results/registered_xanes3D"][:, :, gui_h.hs[
                            "VisImgViewAlign4thDim sldr"].value, :]
                elif gui_h.hs["VisImgViewAlignOptn drpdn"].value == "x-y-z":
                    gui_h.xanes_aligned_data = 0
                    gui_h.xanes_aligned_data = f[
                        "/registration_results/reg_results/registered_xanes3D"][
                            gui_h.hs["VisImgViewAlign4thDim sldr"].
                            value, :, :, :]
        if not viewer_state:
            global_h.ijui.show(global_h.ij.py.to_java(
                gui_h.xanes_aligned_data))
            global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")
            global_h.WindowManager.getCurrentImage().setSlice(
                gui_h.hs["VisImgViewAlignSli sldr"].value)
        else:
            win[viewer_name]["ip"].setSlice(
                gui_h.hs["VisImgViewAlignSli sldr"].value)
            gui_h.hs[
                "VisImgViewAlignEng text"].value = gui_h.xanes_fit_eng_list[
                    gui_h.hs["VisImgViewAlignSli sldr"].value - 1]
    elif viewer_name == "xanes3D_fit_jump_flt_viewer":
        if data_state:
            if not viewer_state:
                global_h.ijui.show(global_h.ij.py.to_java(
                    gui_h.edge_jump_mask))
                global_h.ij.py.run_macro("""setMinAndMax(0, 1)""")
    elif viewer_name == "xanes3D_fit_thres_flt_viewer":
        if data_state:
            if not viewer_state:
                global_h.ijui.show(
                    global_h.ij.py.to_java(gui_h.fitted_edge_mask))
                global_h.ij.py.run_macro("""setMinAndMax(0, 1)""")
    elif viewer_name == "xanes3D_fit_maskit_viewer":
        if data_state:
            if not viewer_state:
                global_h.ijui.show(
                    global_h.ij.py.to_java(
                        gui_h.edge_jump_mask *
                        gui_h.spec[int(gui_h.spec.shape[0] / 2)]))
                global_h.ij.py.run_macro(
                    """run("Enhance Contrast", "saturated=0.35")""")
    elif viewer_name == "xanes2D_raw_img_viewer":
        if not viewer_state:
            global_h.ijui.show(global_h.ij.py.to_java(gui_h.xanes_img))
            global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")
            global_h.WindowManager.getCurrentImage().setSlice(0)
    elif viewer_name == "xanes2D_mask_viewer":
        if not viewer_state:
            global_h.ijui.show(
                global_h.ij.py.to_java(gui_h.xanes_img[
                    gui_h.xanes_eng_id_s:gui_h.xanes_eng_id_e + 1,
                    gui_h.xanes_reg_roi[0]:gui_h.xanes_reg_roi[1],
                    gui_h.xanes_reg_roi[2]:gui_h.xanes_reg_roi[3], ]))
            global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")
            global_h.WindowManager.getCurrentImage().setSlice(0)
    elif viewer_name == "xanes2D_review_viewer":
        if not viewer_state:
            global_h.ijui.show(
                global_h.ij.py.to_java(gui_h.xanes_review_aligned_img -
                                       gui_h.xanes_review_fixed_img))
            global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")
            global_h.WindowManager.getCurrentImage().setSlice(0)
    elif viewer_name == "xanes2D_analysis_viewer":
        if not viewer_state:
            global_h.ijui.show(global_h.ij.py.to_java(gui_h.xanes_img_roi))
            global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")
            global_h.WindowManager.getCurrentImage().setSlice(0)
    elif viewer_name == "xanes2D_fit_jump_flt_viewer":
        if data_state:
            if not viewer_state:
                global_h.ijui.show(global_h.ij.py.to_java(
                    gui_h.edge_jump_mask))
                global_h.ij.py.run_macro("""setMinAndMax(0, 1)""")
    elif viewer_name == "xanes2D_fit_thres_flt_viewer":
        if data_state:
            if not viewer_state:
                global_h.ijui.show(
                    global_h.ij.py.to_java(gui_h.fitted_edge_mask))
                global_h.ij.py.run_macro("""setMinAndMax(0, 1)""")
    elif viewer_name == "xanes2D_fit_maskit_viewer":
        if data_state:
            if not viewer_state:
                global_h.ijui.show(
                    global_h.ij.py.to_java(
                        gui_h.edge_jump_mask *
                        gui_h.spec[int(gui_h.spec.shape[0] / 2)]))
                global_h.ij.py.run_macro(
                    """run("Enhance Contrast", "saturated=0.35")""")
    elif viewer_name == "tomo_raw_img_viewer":
        if not viewer_state:
            global_h.ijui.show(global_h.ij.py.to_java(gui_h.raw_proj))
            global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")
            global_h.WindowManager.getCurrentImage().setSlice(0)
    elif viewer_name == "tomo_0&180_viewer":
        if not viewer_state:
            global_h.ijui.show(global_h.ij.py.to_java(gui_h.raw_proj_0))
            global_h.ij.py.run_macro(
                """run("Enhance Contrast", "saturated=0.35")""")
            global_h.WindowManager.getCurrentImage().setSlice(0)
    elif viewer_name == "tomo_cen_review_viewer":
        if not viewer_state:
            args = {"directory": gui_h.tomo_data_center_path, "start": 1}
            global_h.ij.py.run_plugin(" Open VirtualStack", args)
            global_h.WindowManager.getCurrentImage().setSlice(0)
    elif viewer_name == "tomo_recon_viewer":
        if not viewer_state:
            # recon_dir = os.path.join(
            #     gui_h.tomo_recon_top_dir,
            #     "recon_fly_scan_id_{}".format(gui_h.tomo_scan_id[-1]),
            # )
            recon_dir = str(
                Path(gui_h.tomo_recon_top_dir).joinpath(
                    Path("recon_fly_scan_id_{}".format(
                        gui_h.tomo_scan_id[-1]))))

            args = {"directory": recon_dir, "start": 1}
            global_h.ij.py.run_plugin(" Open VirtualStack", args)
            global_h.WindowManager.getCurrentImage().setSlice(0)
    elif viewer_name == "xanes_pp_data_prev_viewer":
        if data_state:
            if not viewer_state:
                global_h.ijui.show(
                    global_h.ij.py.to_java(gui_h.ana_data[
                        gui_h.hs["AnaSelVars sel"].value].compute()))
                global_h.ij.py.run_macro(
                    """run("Enhance Contrast", "saturated=0.35")""")
    elif (viewer_name == "xanes_pp_mask_prev_viewer") and (data is not None):
        global_h.ijui.show(global_h.ij.py.to_java(data.compute()))
        global_h.ij.py.run_macro(
            """run("Enhance Contrast", "saturated=0.35")""")
    win[viewer_name]["ip"] = global_h.WindowManager.getCurrentImage()
    win[viewer_name]["fiji_id"] = global_h.WindowManager.getIDList()[-1]
    if win_name is not None:
        win[viewer_name]["ip"].setTitle(win_name)
    else:
        win[viewer_name]["ip"].setTitle(viewer_name)


def fiji_viewer_state(global_h,
                      gui_h,
                      viewer_name="xanes3D_virtural_stack_preview_viewer"):
    if viewer_name == "xanes3D_virtural_stack_preview_viewer":
        if ((not gui_h.xanes_recon_3D_tiff_temp)
                | (not gui_h.xanes_fixed_scan_id)
                | (not gui_h.xanes_available_sli_file_ids)):
            data_state = False

        else:
            data_state = True
        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif (not global_h.xanes3D_fiji_windows[viewer_name]["fiji_id"]
              in global_h.WindowManager.getIDList()):
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == "xanes3D_mask_viewer":
        if gui_h.xanes_reg_mask is None:
            data_state = False
        elif gui_h.xanes_reg_mask.shape != (
                gui_h.xanes_roi[1] - gui_h.xanes_roi[0],
                gui_h.xanes_roi[3] - gui_h.xanes_roi[2],
        ):
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif (not global_h.xanes3D_fiji_windows[viewer_name]["fiji_id"]
              in global_h.WindowManager.getIDList()):
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == "xanes3D_review_viewer":
        if gui_h.xanes_img_roi is None:
            data_state = False
        elif gui_h.xanes_img_roi.shape != (
                gui_h.xanes_roi[5] - gui_h.xanes_roi[4] + 1,
                gui_h.xanes_roi[1] - gui_h.xanes_roi[0],
                gui_h.xanes_roi[3] - gui_h.xanes_roi[2],
        ):
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif (not global_h.xanes3D_fiji_windows[viewer_name]["fiji_id"]
              in global_h.WindowManager.getIDList()):
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == "xanes3D_review_manual_viewer":
        if gui_h.xanes_review_aligned_img is None:
            data_state = False
        elif gui_h.xanes_review_aligned_img != (
                gui_h.xanes_roi[1] - gui_h.xanes_roi[0],
                gui_h.xanes_roi[3] - gui_h.xanes_roi[2],
        ):
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif (not global_h.xanes3D_fiji_windows[viewer_name]["fiji_id"]
              in global_h.WindowManager.getIDList()):
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == "xanes3D_analysis_viewer":
        if gui_h.xanes_aligned_data is None:
            data_state = False
        else:
            with h5py.File(gui_h.xanes_save_trial_reg_filename, "r") as f:
                data_shape = f[
                    "/registration_results/reg_results/registered_xanes3D"].shape
                if gui_h.hs["VisImgViewAlignOptn drpdn"].value == "x-y-E":
                    if (
                            data_shape[0],
                            data_shape[2],
                            data_shape[3],
                    ) != gui_h.xanes_aligned_data.shape:
                        data_state = False
                    else:
                        data_state = True
                elif gui_h.hs["VisImgViewAlignOptn drpdn"].value == "y-z-E":
                    if (
                            data_shape[0],
                            data_shape[1],
                            data_shape[2],
                    ) != gui_h.xanes_aligned_data.shape:
                        data_state = False
                    else:
                        data_state = True
                elif gui_h.hs["VisImgViewAlignOptn drpdn"].value == "z-x-E":
                    if (
                            data_shape[0],
                            data_shape[1],
                            data_shape[3],
                    ) != gui_h.xanes_aligned_data.shape:
                        data_state = False
                    else:
                        data_state = True
                elif gui_h.hs["VisImgViewAlignOptn drpdn"].value == "x-y-z":
                    if (
                            data_shape[1],
                            data_shape[2],
                            data_shape[3],
                    ) != gui_h.xanes_aligned_data.shape:
                        data_state = False
                    else:
                        data_state = True
                else:
                    data_state = True
        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif (not global_h.xanes3D_fiji_windows[viewer_name]["fiji_id"]
              in global_h.WindowManager.getIDList()):
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == "xanes3D_fit_jump_flt_viewer":
        if gui_h.edge_jump_mask is None:
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif (not global_h.xanes3D_fiji_windows[viewer_name]["fiji_id"]
              in global_h.WindowManager.getIDList()):
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == "xanes3D_fit_thres_flt_viewer":
        if gui_h.fitted_edge_mask is None:
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif (not global_h.xanes3D_fiji_windows[viewer_name]["fiji_id"]
              in global_h.WindowManager.getIDList()):
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == "xanes3D_fit_maskit_viewer":
        if gui_h.fit_flt_prev_maskit:
            data_state = True
        else:
            data_state = False

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif (not global_h.xanes3D_fiji_windows[viewer_name]["fiji_id"]
              in global_h.WindowManager.getIDList()):
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == "xanes2D_raw_img_viewer":
        if gui_h.xanes_img is None:
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif (not global_h.xanes2D_fiji_windows[viewer_name]["fiji_id"]
              in global_h.WindowManager.getIDList()):
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == "xanes2D_mask_viewer":
        if gui_h.xanes_img is None:
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif (not global_h.xanes2D_fiji_windows[viewer_name]["fiji_id"]
              in global_h.WindowManager.getIDList()):
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == "xanes2D_review_viewer":
        if gui_h.xanes_review_aligned_img is None:
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif (not global_h.xanes2D_fiji_windows[viewer_name]["fiji_id"]
              in global_h.WindowManager.getIDList()):
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == "xanes2D_analysis_viewer":
        if gui_h.xanes_img_roi is None:
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif (not global_h.xanes2D_fiji_windows[viewer_name]["fiji_id"]
              in global_h.WindowManager.getIDList()):
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == "xanes2D_fit_jump_flt_viewer":
        if gui_h.edge_jump_mask is None:
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif (not global_h.xanes2D_fiji_windows[viewer_name]["fiji_id"]
              in global_h.WindowManager.getIDList()):
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == "xanes2D_fit_thres_flt_viewer":
        if gui_h.fitted_edge_mask is None:
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif (not global_h.xanes2D_fiji_windows[viewer_name]["fiji_id"]
              in global_h.WindowManager.getIDList()):
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == "xanes2D_fit_maskit_viewer":
        if gui_h.fit_flt_prev_maskit:
            data_state = True
        else:
            data_state = False

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif (not global_h.xanes2D_fiji_windows[viewer_name]["fiji_id"]
              in global_h.WindowManager.getIDList()):
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name in ["tomo_raw_img_viewer", "tomo_0&180_viewer"]:
        if gui_h.tomo_scan_id is None:
            data_state = False
        else:
            data_state = True
        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif (not global_h.tomo_fiji_windows[viewer_name]["fiji_id"]
              in global_h.WindowManager.getIDList()):
            viewer_state = False
        else:
            viewer_state = True
    elif viewer_name == "tomo_cen_review_viewer":
        if gui_h.recon_finish & (gui_h.tomo_recon_type == "Trial Cent"):
            data_state = True
        else:
            data_state = False
        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif (not global_h.tomo_fiji_windows[viewer_name]["fiji_id"]
              in global_h.WindowManager.getIDList()):
            viewer_state = False
        else:
            viewer_state = True
    elif (viewer_name
          == "xanes_pp_data_prev_viewer") or (viewer_name
                                              == "xanes_pp_mask_prev_viewer"):
        if not gui_h.ana_data:
            data_state = False
        else:
            data_state = True

        if global_h.WindowManager.getIDList() is None:
            viewer_state = False
        elif (not global_h.xanes_ana_fiji_windows[viewer_name]["fiji_id"]
              in global_h.WindowManager.getIDList()):
            viewer_state = False
        else:
            viewer_state = True
    else:
        print("Unrecognized viewer name")
        data_state = False
        viewer_state = False
    return data_state, viewer_state


def gen_external_py_script(filename, code):
    with open(filename, "w") as f:
        for ii in range(len(code.keys())):
            f.writelines(code[ii] + "\n")


def get_handles(hs, key):

    def get_vals(dictionary, key):
        for k, v in dictionary.items():
            if k == key:
                yield v
            elif isinstance(v, dict):
                for rlt in get_vals(v, key):
                    yield rlt
            elif isinstance(v, list):
                for d in v:
                    for rlt in get_vals(d, key):
                        yield rlt

    tem = []

    def get_keys(dictionary):

        def _get_keys(dictionary):
            for key, val in dictionary.items():
                yield key
                if isinstance(val, dict):
                    yield _get_keys(val)

        def _degen(gen):
            for ii in gen:
                if not isinstance(ii, types.GeneratorType):
                    tem.append(ii)
                else:
                    _degen(ii)
            return tem

        return _degen(_get_keys(dictionary))

    return get_keys(next(get_vals(hs, key)))


def get_decendant(box, level=-1):
    if level < 0:
        level = 1000000
    tem = []
    cnt = [0]

    def _get_decendant(box, level):
        if hasattr(box, "children"):
            cnt[0] += 1
            if cnt[0] < level:
                for ii in box.children:
                    _get_decendant(ii, level - cnt[0])
        else:
            tem.append(box)
        return tem

    return _get_decendant(box, level=level)


def get_raw_img_info(fn, cfg, scan_type="tomo"):
    data_info = {}
    try:
        with h5py.File(fn, "r") as f:
            if scan_type == "tomo":
                ang = f[cfg["structured_h5_reader"]["io_data_info"]
                        ["item01_path"]][:]
                idx = rm_redundant(ang)
                data_info["theta_len"] = ang[idx].shape[0]
                data_info["theta_min"] = np.min(ang[idx])
                data_info["theta_max"] = np.max(ang[idx])
                data_info["img_dim"] = list(
                    f[cfg["structured_h5_reader"]["io_data_info"]
                      ["item00_path"]].shape)
                data_info["img_dim"][0] = data_info["theta_len"]
                for ii in range(2, 7):
                    key = cfg["structured_h5_reader"]["io_data_info"][
                        f"item{str(ii).zfill(2)}_path"]
                    if key in f:
                        # data_info[os.path.basename(key)] = f[key][()]
                        data_info[Path(key).name] = f[key][()]
            elif scan_type == "xanes2D":
                data_info["img_dim"] = f[cfg["structured_h5_reader"]
                                         ["io_data_info"]["item00_path"]].shape
                data_info["eng_len"] = f[cfg["structured_h5_reader"][
                    "io_data_info"]["item01_path"]].shape[0]
                data_info["eng_min"] = np.min(
                    f[cfg["structured_h5_reader"]["io_data_info"]
                      ["item01_path"]])
                data_info["eng_max"] = np.max(
                    f[cfg["structured_h5_reader"]["io_data_info"]
                      ["item01_path"]])
                for ii in range(2, 6):
                    key = cfg["structured_h5_reader"]["io_data_info"][
                        f"item{str(ii).zfill(2)}_path"]
                    if key in f:
                        # data_info[os.path.basename(key)] = f[key][()]
                        data_info[Path(key).name] = f[key][()]
                data_info["magnification"] = f[cfg["structured_h5_reader"][
                    "io_data_info"]["item01_path"]][()]
                data_info["pixel size"] = f[cfg["structured_h5_reader"]
                                            ["io_data_info"]["item02_path"]][(
                                            )]
                data_info["note"] = f[cfg["structured_h5_reader"]
                                      ["io_data_info"]["item04_path"]][()]
                data_info["scan time"] = datetime.fromtimestamp(
                    f[cfg["structured_h5_reader"]["io_data_info"]
                      ["item05_path"]][()]).strftime("%m/%d/%Y, %H:%M:%S")
            elif scan_type == "xanes3D":
                data_info["img_dim"] = f[cfg["structured_h5_reader"]
                                         ["io_data_info"]["item00_path"]].shape
                data_info["theta_len"] = f[cfg["structured_h5_reader"][
                    "io_data_info"]["item01_path"]].shape[0]
                data_info["theta_min"] = np.min(
                    f[cfg["structured_h5_reader"]["io_data_info"]
                      ["item01_path"]])
                data_info["theta_max"] = np.max(
                    f[cfg["structured_h5_reader"]["io_data_info"]
                      ["item01_path"]])
                for ii in range(2, 7):
                    key = cfg["structured_h5_reader"]["io_data_info"][
                        f"item{str(ii).zfill(2)}_path"]
                    if key in f:
                        # data_info[os.path.basename(key)] = f[key][()]
                        data_info[Path(key).name] = f[key][()]
    except Exception as err:
        print(str(err))
        data_info = {}
    return data_info


def load_io_config(main_gui_h):
    try:
        f = open(main_gui_h.io_data_struc_tomo_cfg_file, "r")
        io_tomo_cfg = json.load(f)
        set_io_config(main_gui_h.io_config_gui, io_tomo_cfg, cfg_type="tomo")
    except:
        set_io_config(main_gui_h.io_config_gui, IO_TOMO_CFG_DEFAULT)
    try:
        f = open(main_gui_h.io_data_struc_xanes2D_cfg_file, "r")
        io_xanes2D_cfg = json.load(f)
        set_io_config(main_gui_h.io_config_gui,
                      io_xanes2D_cfg,
                      cfg_type="xanes2D")
    except:
        set_io_config(main_gui_h.io_config_gui,
                      IO_XANES2D_CFG_DEFAULT,
                      cfg_type="xanes2D")
    try:
        f = open(main_gui_h.io_data_struc_xanes3D_cfg_file, "r")
        io_xanes3D_cfg = json.load(f)
        set_io_config(main_gui_h.io_config_gui,
                      io_xanes3D_cfg,
                      cfg_type="xanes3D")
    except:
        set_io_config(main_gui_h.io_config_gui,
                      IO_XANES3D_CFG_DEFAULT,
                      cfg_type="xanes3D")


def read_config_from_reg_file(gui_h, dtype="2D_XANES"):
    if dtype == "2D_XANES":
        pass
    elif dtype == "3D_XANES":
        with h5py.File(gui_h.xanes_save_trial_reg_filename, "r") as f:
            gui_h.xanes_recon_3D_tiff_temp = f[
                "/trial_registration/data_directory_info/recon_path_template"][
                    ()]
            gui_h.xanes_raw_3D_h5_top_dir = f[
                "/trial_registration/data_directory_info/raw_h5_top_dir"][()]
            gui_h.xanes_recon_3D_top_dir = f[
                "/trial_registration/data_directory_info/recon_top_dir"][()]
            gui_h.xanes_fixed_scan_id = int(
                f["/trial_registration/trial_reg_parameters/fixed_scan_id"][(
                )])
            gui_h.xanes_fixed_sli_id = int(
                f["/trial_registration/trial_reg_parameters/fixed_slice"][()])
            gui_h.xanes_roi = f[
                "/trial_registration/trial_reg_parameters/slice_roi"][:].tolist(
                )
            gui_h.xanes_scan_id_s = int(
                f["/trial_registration/trial_reg_parameters/scan_ids"][0])
            gui_h.xanes_scan_id_e = int(
                f["/trial_registration/trial_reg_parameters/scan_ids"][-1])
            gui_h.xanes_reg_sli_search_half_width = int(f[
                "/trial_registration/trial_reg_parameters/sli_search_half_range"]
                                                        [()])
            gui_h.xanes_reg_method = f[
                "/trial_registration/trial_reg_parameters/reg_method"][()]
            gui_h.xanes_reg_ref_mode = f[
                "/trial_registration/trial_reg_parameters/reg_ref_mode"][()]
            gui_h.xanes_reg_use_smooth_img = bool(
                f["/trial_registration/trial_reg_parameters/use_smooth_img"][(
                )])
            gui_h.xanes_reg_smooth_sigma = int(
                f["/trial_registration/trial_reg_parameters/img_smooth_sigma"][
                    ()])
            gui_h.xanes_reg_use_chunk = bool(
                f["/trial_registration/trial_reg_parameters/use_chunk"][()])
            gui_h.xanes_reg_chunk_sz = int(
                f["/trial_registration/trial_reg_parameters/chunk_sz"][()])
            gui_h.xanes_reg_use_mask = bool(
                f["/trial_registration/trial_reg_parameters/use_mask"][()])
            gui_h.xanes_alignment_pairs = f[
                "/trial_registration/trial_reg_parameters/alignment_pairs"][:]
            gui_h.trial_reg = f[
                "/trial_registration/trial_reg_results/{0}/trial_reg_img{0}".
                format("000")][:]
            gui_h.trial_reg_fixed = f[
                "/trial_registration/trial_reg_results/{0}/trial_fixed_img{0}".
                format("000")][:]
            gui_h.xanes_review_aligned_img = np.ndarray(
                gui_h.trial_reg[0].shape)


def restart(gui_h, dtype="2D_XANES"):
    if dtype == "2D_XANES":
        gui_h.hs["SelRawH5Path text"].value = "Choose raw h5 directory ..."
        gui_h.hs["SelSaveTrial text"].value = "Save trial registration as ..."
        gui_h.hs[
            "CfmFile&Path text"].value = "Save trial registration, or go directly review registration ..."
        gui_h.hs["SelRawH5Path btn"].description = "XANES2D File"
        gui_h.hs["SelSaveTrial btn"].description = "Save Reg File"
        gui_h.hs["SelRawH5Path btn"].style.button_color = "orange"
        gui_h.hs["SelSaveTrial btn"].style.button_color = "orange"
        gui_h.hs["FijiRawImgPrev chbx"].value = False
        gui_h.hs["FijiMaskViewer chbx"].value = False
        gui_h.hs["FijiEngId sldr"].value = 0
        gui_h.hs["RegPair sldr"].value = 0
        fiji_viewer_off(gui_h.global_h, gui_h, viewer_name="all")

        gui_h.xanes_file_configured = False
        gui_h.xanes_data_configured = False
        gui_h.xanes_roi_configured = False
        gui_h.xanes_reg_params_configured = False
        gui_h.xanes_reg_done = False
        gui_h.xanes_reg_review_done = False
        gui_h.xanes_alignment_done = False
        gui_h.xanes_fit_eng_configured = False
        gui_h.xanes_review_read_alignment_option = False

        gui_h.xanes_file_raw_h5_set = False
        gui_h.xanes_file_save_trial_set = False
        gui_h.xanes_file_reg_file_set = False
        gui_h.xanes_file_config_file_set = False
        gui_h.xanes_config_alternative_flat_set = False
        gui_h.xanes_config_raw_img_readed = False
        gui_h.xanes_regparams_anchor_idx_set = False
        gui_h.xanes_file_reg_file_readed = False
        gui_h.xanes_fit_eng_set = False

        gui_h.xanes_config_is_raw = True
        gui_h.xanes_config_is_refine = False
        gui_h.xanes_config_img_scalar = 1
        gui_h.xanes_config_use_smooth_flat = False
        gui_h.xanes_config_smooth_flat_sigma = 0
        gui_h.xanes_config_use_alternative_flat = False
        gui_h.xanes_config_eng_list = None

        gui_h.xanes_config_alternative_flat_filename = None
        gui_h.xanes_review_reg_best_match_filename = None

        gui_h.xanes_reg_use_chunk = True
        gui_h.xanes_reg_anchor_idx = 0
        gui_h.xanes_reg_roi = [0, 10, 0, 10]
        gui_h.xanes_reg_use_mask = True
        gui_h.xanes_reg_mask = None
        gui_h.xanes_reg_mask_dilation_width = 0
        gui_h.xanes_reg_mask_thres = 0
        gui_h.xanes_reg_use_smooth_img = False
        gui_h.xanes_reg_smooth_img_sigma = 5
        gui_h.xanes_reg_chunk_sz = None
        gui_h.xanes_reg_method = None
        gui_h.xanes_reg_ref_mode = None
        gui_h.xanes_reg_mrtv_level = 4
        gui_h.xanes_reg_mrtv_width = 10
        gui_h.xanes_reg_mrtv_subpixel_step = 0.2

        gui_h.xanes_visualization_auto_bc = False

        gui_h.xanes_img = None
        gui_h.xanes_img_roi = None
        gui_h.xanes_review_aligned_img_original = None
        gui_h.xanes_review_aligned_img = None
        gui_h.xanes_review_fixed_img = None
        gui_h.xanes_review_bad_shift = False
        gui_h.xanes_manual_xshift = 0
        gui_h.xanes_manual_yshift = 0
        gui_h.xanes_review_shift_dict = {}

        gui_h.xanes_eng_id_s = 0
        gui_h.xanes_eng_id_e = 1

        gui_h.xanes_element = None
        gui_h.xanes_fit_eng_list = None
        gui_h.xanes_fit_type = "wl"
        gui_h.xanes_fit_edge_eng = 0
        gui_h.xanes_fit_wl_fit_eng_s = 0
        gui_h.xanes_fit_wl_fit_eng_e = 0
        gui_h.xanes_fit_pre_edge_e = -50
        gui_h.xanes_fit_post_edge_s = 100
        gui_h.xanes_fit_edge_0p5_fit_s = 0
        gui_h.xanes_fit_edge_0p5_fit_e = 0
        gui_h.xanes_fit_spectrum = None
        gui_h.xanes_fit_use_mask = False
        gui_h.xanes_fit_mask_thres = None
        gui_h.xanes_fit_mask_img_id = None
        gui_h.xanes_fit_mask = 1
        gui_h.xanes_fit_edge_jump_thres = 1.0
        gui_h.xanes_fit_edge_offset_thres = 1.0
        gui_h.hs["SelSaveTrial btn"].initialfile = "2D_trial_reg.h5"

        gui_h.xanes_fit_gui_h.fit_flt_prev = False
        gui_h.xanes_fit_gui_h.fit_flt_prev_sli = 0
        gui_h.xanes_fit_gui_h.fit_flt_prev_configed = False
        gui_h.xanes_fit_gui_h.fit_flt_prev_maskit = False
        gui_h.xanes_fit_gui_h.hs["FitItemConfigEdgeJumpThres sldr"].value = 1
        gui_h.xanes_fit_gui_h.hs["FitItemConfigEdgeOfstThres sldr"].value = 1
        gui_h.xanes_fit_gui_h.hs["FitItemConfigFitPrvw chbx"].value = 0

        gui_h.global_h.xanes2D_fiji_windows = {
            "xanes2D_raw_img_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "xanes2D_mask_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "xanes2D_review_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "xanes2D_analysis_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "analysis_viewer_z_plot_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "xanes2D_fit_jump_flt_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "xanes2D_fit_thres_flt_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "xanes2D_fit_maskit_viewer": {
                "ip": None,
                "fiji_id": None
            },
        }
    elif dtype == "3D_XANES":
        gui_h.hs["FijiRawImgPrev chbx"].value = False
        gui_h.hs["FijiMaskViewer chbx"].value = False
        gui_h.hs["SelRawH5Path text"].value = "Choose raw h5 directory ..."
        gui_h.hs["SelReconPath text"].value = "Choose recon top directory ..."
        gui_h.hs["SelSavTrial box"].value = "Save trial registration as ..."
        gui_h.hs[
            "SelFile&PathCfm text"].value = "Save trial registration, or go directly review registration ..."
        gui_h.hs["SelRawH5Path btn"].description = "Raw h5 Dir"
        gui_h.hs["SelReconPath btn"].description = "Recon Top Dir"
        gui_h.hs["SelSavTrial btn"].description = "Save Reg File"
        gui_h.hs["SelRawH5Path btn"].style.button_color = "orange"
        gui_h.hs["SelReconPath btn"].style.button_color = "orange"
        gui_h.hs["SelSavTrial btn"].style.button_color = "orange"
        fiji_viewer_off(gui_h.global_h, gui_h, viewer_name="all")

        gui_h.xanes_filepath_configured = False
        gui_h.xanes_indices_configured = False
        gui_h.xanes_roi_configured = False
        gui_h.xanes_reg_params_configured = False
        gui_h.xanes_reg_done = False
        gui_h.xanes_reg_review_done = False
        gui_h.xanes_alignment_done = False
        gui_h.xanes_use_existing_reg_file = False
        gui_h.xanes_use_existing_reg_reviewed = False
        gui_h.xanes_reg_review_file = None
        gui_h.xanes_reg_use_chunk = True
        gui_h.xanes_reg_use_mask = True
        gui_h.xanes_reg_use_smooth_img = False

        gui_h.xanes_raw_h5_path_set = False
        gui_h.xanes_recon_path_set = False
        gui_h.xanes_save_trial_set = False
        gui_h.xanes_scan_id_set = False
        gui_h.xanes_reg_file_set = False
        gui_h.xanes_config_file_set = False
        gui_h.xanes_fixed_scan_id_set = False
        gui_h.xanes_fixed_sli_id_set = False
        gui_h.xanes_reg_file_readed = False
        gui_h.xanes_fit_eng_configured = False

        gui_h.xanes_review_shift_dict = {}
        gui_h.xanes_reg_mask_dilation_width = 0
        gui_h.xanes_reg_mask_thres = 0
        gui_h.xanes_img_roi = None
        gui_h.xanes_roi = [0, 10, 0, 10, 0, 10]
        gui_h.xanes_reg_mask = None
        gui_h.xanes_aligned_data = None
        gui_h.xanes_fit_slice = 0
        gui_h.xanes_raw_3D_h5_top_dir = None
        gui_h.xanes_recon_3D_top_dir = None
        gui_h.xanes_save_trial_reg_filename = None
        gui_h.xanes_save_trial_reg_config_filename = None
        gui_h.xanes_save_trial_reg_config_filename_original = None
        gui_h.xanes_raw_3D_h5_temp = None
        gui_h.xanes_available_raw_ids = None
        gui_h.xanes_recon_3D_tiff_temp = None
        gui_h.xanes_recon_3D_dir_temp = None
        gui_h.xanes_reg_best_match_filename = None
        gui_h.xanes_available_recon_ids = None
        gui_h.xanes_available_sli_file_ids = None
        gui_h.xanes_fixed_scan_id = None
        gui_h.xanes_scan_id_s = None
        gui_h.xanes_scan_id_e = None
        gui_h.xanes_fixed_sli_id = None
        gui_h.xanes_reg_sli_search_half_width = None
        gui_h.xanes_reg_chunk_sz = None
        gui_h.xanes_reg_smooth_sigma = 0
        gui_h.xanes_reg_method = None
        gui_h.xanes_reg_ref_mode = None
        gui_h.xanes_review_bad_shift = False
        gui_h.xanes_visualization_viewer_option = "fiji"
        gui_h.xanes_fit_view_option = "x-y-E"
        gui_h.xanes_element = None
        gui_h.xanes_fit_type = "wl"
        gui_h.xanes_fit_edge_eng = 0
        gui_h.xanes_fit_wl_fit_eng_s = 0
        gui_h.xanes_fit_wl_fit_eng_e = 0
        gui_h.xanes_fit_pre_edge_e = -50
        gui_h.xanes_fit_post_edge_s = 100
        gui_h.xanes_fit_edge_0p5_fit_s = 0
        gui_h.xanes_fit_edge_0p5_fit_e = 0
        gui_h.xanes_fit_spectrum = None
        gui_h.xanes_fit_use_mask = False
        gui_h.xanes_fit_mask_thres = None
        gui_h.xanes_fit_mask_scan_id = None
        gui_h.xanes_fit_mask = 1
        gui_h.xanes_fit_edge_jump_thres = 1.0
        gui_h.xanes_fit_edge_offset_thres = 1.0
        gui_h.xanes_fit_use_flt_spec = False
        gui_h.hs["SelSavTrial btn"].initialfile = "3D_trial_reg.h5"

        gui_h.xanes_fit_gui_h.fit_flt_prev = False
        gui_h.xanes_fit_gui_h.fit_flt_prev_sli = 0
        gui_h.xanes_fit_gui_h.fit_flt_prev_configed = False
        gui_h.xanes_fit_gui_h.fit_flt_prev_maskit = False
        gui_h.xanes_fit_gui_h.hs["FitItemConfigEdgeJumpThres sldr"].value = 1
        gui_h.xanes_fit_gui_h.hs["FitItemConfigEdgeOfstThres sldr"].value = 1
        gui_h.xanes_fit_gui_h.hs["FitItemConfigFitPrvw chbx"].value = 0

        gui_h.global_h.xanes3D_fiji_windows = {
            "xanes3D_virtural_stack_preview_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "xanes3D_mask_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "xanes3D_review_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "xanes3D_review_manual_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "xanes3D_analysis_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "analysis_viewer_z_plot_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "xanes3D_fit_jump_flt_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "xanes3D_fit_thres_flt_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "xanes3D_fit_maskit_viewer": {
                "ip": None,
                "fiji_id": None
            },
        }
    elif dtype == "TOMO":
        gui_h.hs["SelRawH5TopDir text"].value = "Choose raw h5 top dir ..."
        gui_h.hs[
            "SelSavReconDir text"].value = "Select top directory where recon subdirectories are saved..."
        gui_h.hs["SelSavDebugDir text"].value = "Debug is disabled..."
        gui_h.hs[
            "SelFile&PathCfm text"].value = "After setting directories, confirm to proceed ..."
        gui_h.hs["SelRawH5TopDir btn"].description = "Raw Top Dir"
        gui_h.hs["SelSavReconDir btn"].description = "Save Rec File"
        gui_h.hs["SelSavDebugDir btn"].description = "Save Debug Dir"
        gui_h.hs["SavDebug chbx"].value = False
        gui_h.hs["SelRawH5TopDir btn"].style.button_color = "orange"
        gui_h.hs["SelSavReconDir btn"].style.button_color = "orange"
        gui_h.hs["SelSavDebugDir btn"].style.button_color = "orange"
        gui_h.tomo_raw_data_top_dir_set = False
        gui_h.tomo_recon_path_set = False
        gui_h.tomo_data_center_path_set = False
        gui_h.tomo_debug_path_set = False

        gui_h.tomo_filepath_configured = False
        gui_h.tomo_data_configured = False

        gui_h.raw_proj_0 = None
        gui_h.raw_proj_180 = None

        gui_h.tomo_left_box_selected_flt = "phase retrieval"
        gui_h.tomo_selected_alg = "gridrec"

        gui_h.tomo_recon_param_dict = TOMO_RECON_PARAM_DICT

        gui_h.tomo_raw_data_top_dir = None
        gui_h.tomo_raw_data_file_template = None
        gui_h.tomo_recon_top_dir = None
        gui_h.tomo_debug_top_dir = None
        gui_h.tomo_cen_list_file = None
        gui_h.tomo_alt_flat_file = None
        gui_h.tomo_alt_dark_file = None
        gui_h.tomo_wedge_ang_auto_det_ref_fn = None

        gui_h.tomo_recon_type = "Trial Cent"
        gui_h.tomo_use_debug = False
        gui_h.tomo_use_alt_flat = False

        gui_h.hs["AltFlatFile btn"].style.button_color = "orange"
        gui_h.hs["AltFlatFile btn"].description = "Alt Flat File"
        gui_h.hs["AltDarkFile btn"].style.button_color = "orange"
        gui_h.hs["AltDarkFile btn"].description = "Alt Dark File"

        gui_h.hs["UseFakeFlat chbx"].value = False
        gui_h.hs["FakeFlatVal text"].value = 10000
        gui_h.hs["UseFakeDark chbx"].value = False
        gui_h.hs["FakeDarkVal text"].value = 100

        gui_h.hs["UseRmZinger chbx"].value = False
        gui_h.hs["ZingerLevel text"].value = 500
        gui_h.hs["UseMask chbx"].value = True
        gui_h.hs["MaskRat text"].value = 1

        gui_h.hs["AutoDet chbx"].value = False
        gui_h.hs["IsWedge chbx"].value = False
        gui_h.hs["AutoThres text"].value = 0.1
        gui_h.hs["AutoRefFn btn"].style.button_color = "orange"

        gui_h.tomo_scan_id = 0
        gui_h.tomo_ds_ratio = 1
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
    elif dtype == "XANES_FITTING":
        gui_h.fit_wl_fit_use_param_bnd = False
        gui_h.fit_wl_optimizer = "numpy"
        gui_h.fit_wl_fit_func = 2
        gui_h.fit_edge_fit_use_param_bnd = False
        gui_h.fit_edge_optimizer = "numpy"
        gui_h.fit_edge_fit_func = 3
        gui_h.analysis_saving_items = set(dat_dict.XANES_FULL_SAVE_DEFAULT)
        gui_h.parent_h.xanes_fit_type == "full"

        gui_h.fit_flt_prev = False
        gui_h.fit_flt_prev_sli = 0
        gui_h.fit_flt_prev_configed = False
        gui_h.fit_flt_prev_maskit = False

        gui_h.pre_es_idx = None
        gui_h.pre_ee_idx = None
        gui_h.post_es_idx = None
        gui_h.post_ee_idx = None
        gui_h.fit_flt_prev_xana = None
        gui_h.e0_idx = None
        gui_h.pre = None
        gui_h.post = None
        gui_h.edge_jump_mask = None
        gui_h.fitted_edge_mask = None

        gui_h.fit_fit_wl = True
        gui_h.fit_fit_edge = True
        gui_h.fit_find_edge = True
        gui_h.fit_find_edge0p5_dir = True
        gui_h.fit_use_flt_spec = False


def save_io_config(gui_h):
    io_data_structure_tomo = {}
    io_data_structure_tomo["use_h5_reader"] = gui_h.hs["IOOptnH5 chbx"].value
    io_data_structure_tomo["structured_h5_reader"] = {}
    io_data_structure_tomo["structured_h5_reader"]["io_data_structure"] = {}
    io_data_structure_tomo["structured_h5_reader"]["io_data_info"] = {}
    io_data_structure_tomo["structured_h5_reader"]["io_data_structure"][
        "data_path"] = gui_h.hs["IOTomoConfigDataImg text"].value
    io_data_structure_tomo["structured_h5_reader"]["io_data_structure"][
        "flat_path"] = gui_h.hs["IOTomoConfigDataFlat text"].value
    io_data_structure_tomo["structured_h5_reader"]["io_data_structure"][
        "dark_path"] = gui_h.hs["IOTomoConfigDataDark text"].value
    io_data_structure_tomo["structured_h5_reader"]["io_data_structure"][
        "theta_path"] = gui_h.hs["IOTomoConfigDataTheta text"].value

    io_data_structure_tomo["structured_h5_reader"]["io_data_info"][
        "item00_path"] = gui_h.hs["IOTomoConfigInfo0 text"].value
    io_data_structure_tomo["structured_h5_reader"]["io_data_info"][
        "item01_path"] = gui_h.hs["IOTomoConfigInfo1 text"].value
    io_data_structure_tomo["structured_h5_reader"]["io_data_info"][
        "item02_path"] = gui_h.hs["IOTomoConfigInfo2 text"].value
    io_data_structure_tomo["structured_h5_reader"]["io_data_info"][
        "item03_path"] = gui_h.hs["IOTomoConfigInfo3 text"].value

    io_data_structure_tomo["structured_h5_reader"]["io_data_info"][
        "item04_path"] = gui_h.hs["IOTomoConfigInfo4 text"].value
    io_data_structure_tomo["structured_h5_reader"]["io_data_info"][
        "item05_path"] = gui_h.hs["IOTomoConfigInfo5 text"].value
    io_data_structure_tomo["structured_h5_reader"]["io_data_info"][
        "item06_path"] = gui_h.hs["IOTomoConfigInfo6 text"].value
    io_data_structure_tomo["structured_h5_reader"]["io_data_info"][
        "item07_path"] = gui_h.hs["IOTomoConfigInfo7 text"].value

    io_data_structure_tomo["tomo_raw_fn_template"] = gui_h.hs[
        "FnTomoDefRawPatn text"].value

    io_data_structure_tomo["customized_reader"] = {}
    io_data_structure_tomo["customized_reader"]["user_tomo_reader"] = gui_h.hs[
        "IOSpecTomoRdr text"].value

    io_data_structure_xanes2D = {}
    io_data_structure_xanes2D["use_h5_reader"] = gui_h.hs[
        "IOOptnH5 chbx"].value
    io_data_structure_xanes2D["structured_h5_reader"] = {}
    io_data_structure_xanes2D["structured_h5_reader"]["io_data_structure"] = {}
    io_data_structure_xanes2D["structured_h5_reader"]["io_data_info"] = {}
    io_data_structure_xanes2D["structured_h5_reader"]["io_data_structure"][
        "data_path"] = gui_h.hs["IOXANES2DConfigDataImg text"].value
    io_data_structure_xanes2D["structured_h5_reader"]["io_data_structure"][
        "flat_path"] = gui_h.hs["IOXANES2DConfigDataFlat text"].value
    io_data_structure_xanes2D["structured_h5_reader"]["io_data_structure"][
        "dark_path"] = gui_h.hs["IOXANES2DConfigDataDark text"].value
    io_data_structure_xanes2D["structured_h5_reader"]["io_data_structure"][
        "eng_path"] = gui_h.hs["IOXANES2DConfigDataEng text"].value

    io_data_structure_xanes2D["structured_h5_reader"]["io_data_info"][
        "item00_path"] = gui_h.hs["IOXANES2DConfigInfo0 text"].value
    io_data_structure_xanes2D["structured_h5_reader"]["io_data_info"][
        "item01_path"] = gui_h.hs["IOXANES2DConfigInfo1 text"].value
    io_data_structure_xanes2D["structured_h5_reader"]["io_data_info"][
        "item02_path"] = gui_h.hs["IOXANES2DConfigInfo2 text"].value
    io_data_structure_xanes2D["structured_h5_reader"]["io_data_info"][
        "item03_path"] = gui_h.hs["IOXANES2DConfigInfo3 text"].value

    io_data_structure_xanes2D["structured_h5_reader"]["io_data_info"][
        "item04_path"] = gui_h.hs["IOXANES2DConfigInfo4 text"].value
    io_data_structure_xanes2D["structured_h5_reader"]["io_data_info"][
        "item05_path"] = gui_h.hs["IOXANES2DConfigInfo5 text"].value
    io_data_structure_xanes2D["structured_h5_reader"]["io_data_info"][
        "item06_path"] = gui_h.hs["IOXANES2DConfigInfo6 text"].value
    io_data_structure_xanes2D["structured_h5_reader"]["io_data_info"][
        "item07_path"] = gui_h.hs["IOXANES2DConfigInfo7 text"].value

    io_data_structure_xanes2D["xanes2D_raw_fn_template"] = gui_h.hs[
        "FnXANES2DDefRawPatn text"].value

    io_data_structure_xanes2D["customized_reader"] = {}
    io_data_structure_xanes2D["customized_reader"][
        "user_xanes2D_reader"] = gui_h.hs["IOSpecXANES2DRdr text"].value

    io_data_structure_xanes3D = {}
    io_data_structure_xanes3D["use_h5_reader"] = gui_h.hs[
        "IOOptnH5 chbx"].value
    io_data_structure_xanes3D["structured_h5_reader"] = {}
    io_data_structure_xanes3D["structured_h5_reader"]["io_data_structure"] = {}
    io_data_structure_xanes3D["structured_h5_reader"]["io_data_info"] = {}
    io_data_structure_xanes3D["structured_h5_reader"]["io_data_structure"][
        "data_path"] = gui_h.hs["IOXANES3DConfigDataImg text"].value
    io_data_structure_xanes3D["structured_h5_reader"]["io_data_structure"][
        "flat_path"] = gui_h.hs["IOXANES3DConfigDataFlat text"].value
    io_data_structure_xanes3D["structured_h5_reader"]["io_data_structure"][
        "dark_path"] = gui_h.hs["IOXANES3DConfigDataDark text"].value
    io_data_structure_xanes3D["structured_h5_reader"]["io_data_structure"][
        "eng_path"] = gui_h.hs["IOXANES3DConfigDataEng text"].value

    io_data_structure_xanes3D["structured_h5_reader"]["io_data_info"][
        "item00_path"] = gui_h.hs["IOXANES3DConfigInfo0 text"].value
    io_data_structure_xanes3D["structured_h5_reader"]["io_data_info"][
        "item01_path"] = gui_h.hs["IOXANES3DConfigInfo1 text"].value
    io_data_structure_xanes3D["structured_h5_reader"]["io_data_info"][
        "item02_path"] = gui_h.hs["IOXANES3DConfigInfo2 text"].value
    io_data_structure_xanes3D["structured_h5_reader"]["io_data_info"][
        "item03_path"] = gui_h.hs["IOXANES3DConfigInfo3 text"].value

    io_data_structure_xanes3D["structured_h5_reader"]["io_data_info"][
        "item04_path"] = gui_h.hs["IOXANES3DConfigInfo4 text"].value
    io_data_structure_xanes3D["structured_h5_reader"]["io_data_info"][
        "item05_path"] = gui_h.hs["IOXANES3DConfigInfo5 text"].value
    io_data_structure_xanes3D["structured_h5_reader"]["io_data_info"][
        "item06_path"] = gui_h.hs["IOXANES3DConfigInfo6 text"].value
    io_data_structure_xanes3D["structured_h5_reader"]["io_data_info"][
        "item07_path"] = gui_h.hs["IOXANES3DConfigInfo7 text"].value

    io_data_structure_xanes3D["tomo_raw_fn_template"] = gui_h.hs[
        "FnXANES3DDefRawPatn text"].value
    io_data_structure_xanes3D["xanes3D_recon_dir_template"] = gui_h.hs[
        "FnXANES3DDefReconDirPatn text"].value
    io_data_structure_xanes3D["xanes3D_recon_fn_template"] = gui_h.hs[
        "FnXANES3DDefReconFnPatn text"].value

    io_data_structure_xanes3D["customized_reader"] = {}
    io_data_structure_xanes3D["customized_reader"][
        "user_xanes3D_reader"] = gui_h.hs["IOSpecXANES3DRdr text"].value

    with open(gui_h.global_h.io_data_struc_tomo_cfg_file, "w") as f:
        json.dump(io_data_structure_tomo, f)
    with open(gui_h.global_h.io_data_struc_xanes2D_cfg_file, "w") as f:
        json.dump(io_data_structure_xanes2D, f)
    with open(gui_h.global_h.io_data_struc_xanes3D_cfg_file, "w") as f:
        json.dump(io_data_structure_xanes3D, f)


def scale_eng_list(eng_list):
    eng_list = np.array(eng_list)
    if eng_list.max() < 100:
        eng_list *= 1000
    return eng_list


def set_io_config(gui_h, cfg_dict, cfg_type="tomo"):
    if cfg_type == "tomo":
        gui_h.hs["IOTomoConfigDataImg text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_structure"]["data_path"]
        gui_h.hs["IOTomoConfigDataFlat text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_structure"]["flat_path"]
        gui_h.hs["IOTomoConfigDataDark text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_structure"]["dark_path"]
        gui_h.hs["IOTomoConfigDataTheta text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_structure"]["theta_path"]

        gui_h.hs["IOTomoConfigInfo0 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item00_path"]
        gui_h.hs["IOTomoConfigInfo1 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item01_path"]
        gui_h.hs["IOTomoConfigInfo2 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item02_path"]
        gui_h.hs["IOTomoConfigInfo3 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item03_path"]

        gui_h.hs["IOTomoConfigInfo4 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item04_path"]
        gui_h.hs["IOTomoConfigInfo5 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item05_path"]
        gui_h.hs["IOTomoConfigInfo6 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item06_path"]
        gui_h.hs["IOTomoConfigInfo7 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item07_path"]

        gui_h.hs["FnTomoDefRawPatn text"].value = cfg_dict[
            "tomo_raw_fn_template"]

        gui_h.hs["IOSpecTomoRdr text"].value = cfg_dict["customized_reader"][
            "user_tomo_reader"]
    elif cfg_type == "xanes2D":
        gui_h.hs["IOXANES2DConfigDataImg text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_structure"]["data_path"]
        gui_h.hs["IOXANES2DConfigDataFlat text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_structure"]["flat_path"]
        gui_h.hs["IOXANES2DConfigDataDark text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_structure"]["dark_path"]
        gui_h.hs["IOXANES2DConfigDataEng text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_structure"]["eng_path"]

        gui_h.hs["IOXANES2DConfigInfo0 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item00_path"]
        gui_h.hs["IOXANES2DConfigInfo1 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item01_path"]
        gui_h.hs["IOXANES2DConfigInfo2 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item02_path"]
        gui_h.hs["IOXANES2DConfigInfo3 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item03_path"]

        gui_h.hs["IOXANES2DConfigInfo4 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item04_path"]
        gui_h.hs["IOXANES2DConfigInfo5 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item05_path"]
        gui_h.hs["IOXANES2DConfigInfo6 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item06_path"]
        gui_h.hs["IOXANES2DConfigInfo7 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item07_path"]

        gui_h.hs["FnXANES2DDefRawPatn text"].value = cfg_dict[
            "xanes2D_raw_fn_template"]

        gui_h.hs["IOSpecXANES2DRdr text"].value = cfg_dict[
            "customized_reader"]["user_xanes2D_reader"]
    elif cfg_type == "xanes3D":
        gui_h.hs["IOXANES3DConfigDataImg text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_structure"]["data_path"]
        gui_h.hs["IOXANES3DConfigDataFlat text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_structure"]["flat_path"]
        gui_h.hs["IOXANES3DConfigDataDark text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_structure"]["dark_path"]
        gui_h.hs["IOXANES3DConfigDataEng text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_structure"]["eng_path"]

        gui_h.hs["IOXANES3DConfigInfo0 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item00_path"]
        gui_h.hs["IOXANES3DConfigInfo1 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item01_path"]
        gui_h.hs["IOXANES3DConfigInfo2 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item02_path"]
        gui_h.hs["IOXANES3DConfigInfo3 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item03_path"]

        gui_h.hs["IOXANES3DConfigInfo4 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item04_path"]
        gui_h.hs["IOXANES3DConfigInfo5 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item05_path"]
        gui_h.hs["IOXANES3DConfigInfo6 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item06_path"]
        gui_h.hs["IOXANES3DConfigInfo7 text"].value = cfg_dict[
            "structured_h5_reader"]["io_data_info"]["item07_path"]

        gui_h.hs["FnXANES3DDefRawPatn text"].value = cfg_dict[
            "tomo_raw_fn_template"]
        gui_h.hs["FnXANES3DDefReconDirPatn text"].value = cfg_dict[
            "xanes3D_recon_dir_template"]
        gui_h.hs["FnXANES3DDefReconFnPatn text"].value = cfg_dict[
            "xanes3D_recon_fn_template"]
        gui_h.hs["IOSpecXANES3DRdr text"].value = cfg_dict[
            "customized_reader"]["user_xanes3D_reader"]


def update_json_content(fn, new_item):
    # if os.path.exists(fn):
    fn = Path(fn)
    if fn.exists():
        with open(fn, "r") as f:
            tmp = json.load(f)
    else:
        tmp = {}
    for ii in new_item.keys():
        tmp[ii] = new_item[ii]
    with open(fn, "w") as f:
        json.dump(tmp, f)
