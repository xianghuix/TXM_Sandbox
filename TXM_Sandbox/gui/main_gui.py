#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:19:17 2020

@author: xiao
"""
import os, json, inspect
from pathlib import Path

from ipywidgets import widgets
from IPython.display import display
import napari
import imagej

from . import tomo_gui as trg
from . import xanes2D_gui as x2drg
from . import xanes3D_gui as x3drg
from . import misc_gui as misc
from . import io_config_gui as iocg
from .gui_components import get_handles, load_io_config, set_io_config
from ..dicts.config_dict import (
    IO_TOMO_CFG_DEFAULT,
    IO_XANES2D_CFG_DEFAULT,
    IO_XANES3D_CFG_DEFAULT,
)

napari.gui_qt()

try:
    pkg_dir = str(
        Path(os.path.dirname(os.path.abspath(inspect.getfile(trg)))).parent)
    with open(os.path.join(pkg_dir, "config",
                           "analysis_tool_gui_cfg.json")) as f:
        CFG = json.load(f)
        CWD = CFG["cwd"]
except:
    CWD = os.path.abspath(os.path.curdir)


class txm_gui:

    def __init__(self,
                 fiji_path=Path("/home/xiao/software/Fiji.app"),
                 form_sz=[650, 740]):
        self.script_dir = os.path.abspath(os.path.curdir)
        try:
            self.ij = imagej.init(fiji_path, mode=imagej.Mode.INTERACTIVE)
            self.ijui = self.ij.ui()
            self.ijui.showUI()
            self.ij.py.run_macro("""run("Brightness/Contrast...");""")
            if hasattr(imagej, "__version__"):
                # self.WindowManager = self.ij.py.window_manager()
                self.WindowManager = self.ij.WindowManager
                import scyjava as sj

                self.ImagePlusClass = sj.jimport("ij.ImagePlus")
            else:
                from jnius import autoclass

                self.WindowManager = autoclass("ij.WindowManager")
                self.ImagePlusClass = autoclass("ij.ImagePlus")
        except Exception as e:
            print(e)
        self.hs = {}
        self.form_sz = form_sz
        self.cwd = CWD
        # pkg_dir = str(Path(os.path.dirname(os.path.abspath(inspect.getfile(trg)))).parent)
        self.tmp_dir = os.path.join(pkg_dir, "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir, mode=0x777)
        self.GUI_cfg_file = os.path.join(pkg_dir, "config",
                                         "analysis_tool_gui_cfg.json")
        self.io_data_struc_tomo_cfg_file = os.path.join(
            pkg_dir, "config", "io_tomo_h5_data_structure.json")
        self.io_data_struc_xanes2D_cfg_file = os.path.join(
            pkg_dir, "config", "io_xanes2D_h5_data_structure.json")
        self.io_data_struc_xanes3D_cfg_file = os.path.join(
            pkg_dir, "config", "io_xanes3D_h5_data_structure.json")

        self.xanes2D_external_command_name = os.path.join(
            self.script_dir, "xanes2D_external_command.py")
        self.xanes3D_external_command_name = os.path.join(
            self.script_dir, "xanes3D_external_command.py")
        self.tomo_recon_external_command_name = os.path.join(
            self.script_dir, "tomo_recon_external_command.py")

        self.xanes3D_fiji_windows = {
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
        self.xanes2D_fiji_windows = {
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
        self.tomo_fiji_windows = {
            "tomo_raw_img_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "tomo_0&180_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "tomo_cen_review_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "tomo_recon_viewer": {
                "ip": None,
                "fiji_id": None
            },
        }
        self.xanes_ana_fiji_windows = {
            "xanes_pp_data_prev_viewer": {
                "ip": None,
                "fiji_id": None
            },
            "xanes_pp_mask_prev_viewer": {
                "ip": None,
                "fiji_id": None
            },
        }

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
        layout = {
            "border": "5px solid #00FF00",
            "width": f"{self.form_sz[1]}px",
            "height": f"{self.form_sz[0]}px",
        }
        self.hs["MainGUI form"] = widgets.Tab()
        self.hs["MainGUI form"].layout = layout

        ## ## define, organize, and name sub-tabs
        layout = {
            "border": "3px solid #FFCC00",
            "width": f"{self.form_sz[1]-46}px",
            "height": f"{self.form_sz[0]-72}px",
        }
        self.hs["TomoRecon tab"] = widgets.Tab()
        self.hs["XANES2D tab"] = widgets.Tab()
        self.hs["XANES3D tab"] = widgets.Tab()
        self.hs["MISC tab"] = widgets.Tab()
        self.hs["IOConfig tab"] = widgets.Tab()
        self.hs["TomoRecon tab"].layout = layout
        self.hs["XANES2D tab"].layout = layout
        self.hs["XANES3D tab"].layout = layout
        self.hs["MISC tab"].layout = layout
        self.hs["IOConfig tab"].layout = layout

        self.hs["MainGUI form"].children = [
            self.hs["TomoRecon tab"],
            self.hs["XANES2D tab"],
            self.hs["XANES3D tab"],
            self.hs["MISC tab"],
            self.hs["IOConfig tab"],
        ]
        self.hs["MainGUI form"].set_title(0, "TOMO RECON")
        self.hs["MainGUI form"].set_title(1, "2D XANES")
        self.hs["MainGUI form"].set_title(2, "3D XANES")
        self.hs["MainGUI form"].set_title(3, "MISC")
        self.hs["MainGUI form"].set_title(4, "IO CONFIG")

        #################################################################################################################
        #                                                                                                               #
        #                                                     IO CONFIG                                                 #
        #                                                                                                               #
        #################################################################################################################
        self.io_config_gui = iocg.io_config_gui(self, form_sz=self.form_sz)
        self.io_config_gui.build_gui()
        self.hs["IOConfig tab"].children = [
            self.io_config_gui.hs["IOOptn form"],
            self.io_config_gui.hs["IOConfig form"],
        ]
        self.hs["IOConfig tab"].set_title(0, "IO Option")
        self.hs["IOConfig tab"].set_title(1, "Struct h5 IO")

        try:
            f = open(self.io_data_struc_tomo_cfg_file, "r")
            self.io_tomo_cfg = json.load(f)
            set_io_config(self.io_config_gui,
                          self.io_tomo_cfg,
                          cfg_type="tomo")
        except:
            self.io_tomo_cfg = IO_TOMO_CFG_DEFAULT
            set_io_config(self.io_config_gui,
                          IO_TOMO_CFG_DEFAULT,
                          cfg_type="tomo")
        try:
            f = open(self.io_data_struc_xanes2D_cfg_file, "r")
            self.io_xanes2D_cfg = json.load(f)
            set_io_config(self.io_config_gui,
                          self.io_xanes2D_cfg,
                          cfg_type="xanes2D")
        except:
            self.io_xanes2D_cfg = IO_XANES2D_CFG_DEFAULT
            set_io_config(self.io_config_gui,
                          IO_XANES2D_CFG_DEFAULT,
                          cfg_type="xanes2D")
        try:
            f = open(self.io_data_struc_xanes3D_cfg_file, "r")
            self.io_xanes3D_cfg = json.load(f)
            set_io_config(self.io_config_gui,
                          self.io_xanes3D_cfg,
                          cfg_type="xanes3D")
        except:
            self.io_xanes3D_cfg = IO_XANES3D_CFG_DEFAULT
            set_io_config(self.io_config_gui,
                          IO_XANES3D_CFG_DEFAULT,
                          cfg_type="xanes3D")
        self.io_config_gui.boxes_logics()

        #################################################################################################################
        #                                                                                                               #
        #                                                    TOMO RECON                                                 #
        #                                                                                                               #
        #################################################################################################################
        self.tomo_recon_gui = trg.tomo_recon_gui(self, form_sz=self.form_sz)
        self.tomo_recon_gui.build_gui()

        self.hs["TomoRecon tab"].children = [
            self.tomo_recon_gui.hs["Config&Input form"],
            self.tomo_recon_gui.hs["Filter&Recon tab"],
        ]
        self.hs["TomoRecon tab"].set_title(0, "Config")
        self.hs["TomoRecon tab"].set_title(1, "Recon")

        #################################################################################################################
        #                                                                                                               #
        #                                                     2D XANES                                                  #
        #                                                                                                               #
        #################################################################################################################
        self.xanes2D_gui = x2drg.xanes2D_tools_gui(self, form_sz=self.form_sz)
        self.xanes2D_gui.build_gui()

        self.hs["XANES2D tab"].children = [
            self.xanes2D_gui.hs["Config&Input form"],
            self.xanes2D_gui.hs["RegSetting form"],
            self.xanes2D_gui.hs["Reg&Rev form"],
            self.xanes2D_gui.hs["Fitting form"],
            self.xanes2D_gui.hs["Analysis form"],
        ]
        self.hs["XANES2D tab"].set_title(0, "Data Config")
        self.hs["XANES2D tab"].set_title(1, "Reg Config")
        self.hs["XANES2D tab"].set_title(2, "Reg Review")
        self.hs["XANES2D tab"].set_title(3, "Fitting")
        self.hs["XANES2D tab"].set_title(4, "Analysis")

        #################################################################################################################
        #                                                                                                               #
        #                                                     3D XANES                                                  #
        #                                                                                                               #
        #################################################################################################################
        self.xanes3D_gui = x3drg.xanes3D_tools_gui(self, form_sz=self.form_sz)
        self.xanes3D_gui.build_gui()
        self.hs["XANES3D tab"].children = [
            self.xanes3D_gui.hs["Config&Input form"],
            self.xanes3D_gui.hs["RegSetting form"],
            self.xanes3D_gui.hs["Reg&Rev form"],
            self.xanes3D_gui.hs["Fitting form"],
            self.xanes3D_gui.hs["Analysis form"],
        ]
        self.hs["XANES3D tab"].set_title(0, "Data Config")
        self.hs["XANES3D tab"].set_title(1, "Reg Config")
        self.hs["XANES3D tab"].set_title(2, "Reg Review")
        self.hs["XANES3D tab"].set_title(3, "Fitting")
        self.hs["XANES3D tab"].set_title(4, "Analysis")
        
        #################################################################################################################
        #                                                                                                               #
        #                                                       MISC                                                    #
        #                                                                                                               #
        #################################################################################################################
        self.misc_gui = misc.misc_gui(self, form_sz=self.form_sz)
        self.misc_gui.build_gui()
        self.hs["MISC tab"].children = [self.misc_gui.hs['GenImgAlign form'], self.misc_gui.hs['ConvertData form']]
        self.hs["MISC tab"].set_title(0, "General Img Align")
        self.hs["MISC tab"].set_title(1, "Convert Data")
        

        display(self.hs["MainGUI form"])
