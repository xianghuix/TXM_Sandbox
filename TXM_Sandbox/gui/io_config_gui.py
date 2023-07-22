#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:45:03 2020

@author: xiao
"""
import json

from ipywidgets import widgets, GridspecLayout, dlink

from .gui_components import (get_handles, create_widget, enable_disable_boxes, save_io_config)


class io_config_gui():
    def __init__(self, parent_h, form_sz=[650, 740]):
        self.gui_name = 'io_config'
        self.form_sz = form_sz
        self.global_h = parent_h
        self.hs = {}

    def boxes_logics(self):
        if self.hs['IOOptnH5 chbx'].value:
            boxes = ['IOUserRdr box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            boxes = ['IOTomoConfigData box',
                     'IOTomoConfigInfo box',
                     'IOXANES2DConfigData box',
                     'IOXANES2DConfigInfo box',
                     'IOXANES3DConfigData box',
                     'IOXANES3DConfigInfo box',
                     'IOCfm box']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
        else:
            boxes = ['IOTomoConfigData box',
                     'IOTomoConfigInfo box',
                     'IOXANES2DConfigData box',
                     'IOXANES2DConfigInfo box',
                     'IOXANES3DConfigData box',
                     'IOXANES3DConfigInfo box',
                     'IOCfm box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            boxes = ['IOUserRdr box']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1) 

    def build_gui(self):
        ## ## ## reader option -- start
        self.hs['IOOptn form'] = create_widget('VBox', {'border':'3px solid #FFCC00', 'width': '100%', 'height': '100%'})
        
        ## ## ## ## reader option -- start
        self.hs['IOOptn box'] = create_widget('VBox', {'border':'3px solid #8855AA', 'width': 'auto'})
        self.hs['IOOptnH5 chbx'] = create_widget('Checkbox', {'width': '20%'},
                                                              **{'description':'Struct h5', 'value':True, 'disabled':False,
                                                                 'description_tooltip':'Check this if the data is saved in a h5 file with a structure of separated datasets for data, flat, dark, energy, and angle etc.', 'indent':False})

        self.hs['IOOptnH5 chbx'].observe(self.io_optn_chbx_chg, names='value')
        self.hs['IOOptn box'].children = [self.hs['IOOptnH5 chbx']]
        ## ## ## ## reader option -- end
        
        ## ## ## ## user reader load -- start
        self.hs['IOUserSpec box'] = create_widget('VBox', {'border':'3px solid #8855AA', 
                                                  'width': 'auto', 'height': f'{0.21*(self.form_sz[0]-136)}px'})
        
        grid_user_reader_hs = GridspecLayout(3, 100,
                                    layout = {"border":"3px solid #FFCC00",
                                              "width":"100%",
                                              "height":'100%',
                                              "align_items":"flex-start",
                                              "justify_items":"flex-start"})
        self.hs['IOUserRdr box'] = grid_user_reader_hs
        grid_user_reader_hs[0, :66] = create_widget('Text', {'width': '95%'}, **{'disabled':True, 'value':'load tomo data reader'})
        self.hs['IOSpecTomoRdr text'] = grid_user_reader_hs[0, :66]
        grid_user_reader_hs[0, 68:83] = create_widget('SelectFilesButton', {'width': '95%'}, **{'option':'askopenfilename','open_filetypes':(('python files', '*.py'),), 'text_h':self.hs['IOSpecTomoRdr text']})
        self.hs['IOSpecTomoRdr btn'] = grid_user_reader_hs[0, 68:83]
        self.hs['IOSpecTomoRdr btn'].description = 'Tomo Reader' 
        grid_user_reader_hs[1, :66] = create_widget('Text', {'width': '95%'}, **{'disabled':True, 'value':'load xanes2D data reader'})
        self.hs['IOSpecXANES2DRdr text'] = grid_user_reader_hs[1, :66]
        grid_user_reader_hs[1, 68:83] = create_widget('SelectFilesButton', {'width': '95%'}, **{'option':'askopenfilename','open_filetypes':(('python files', '*.py'),), 'text_h':self.hs['IOSpecTomoRdr text']})
        self.hs['IOSpecXANES2DRdr btn'] = grid_user_reader_hs[1, 68:83]
        self.hs['IOSpecXANES2DRdr btn'].description = 'XANES2D Reader' 
        grid_user_reader_hs[2, :66] = create_widget('Text', {'width': '95%'}, **{'disabled':True, 'value':'load xanes3D data reader'})
        self.hs['IOSpecXANES3DRdr text'] = grid_user_reader_hs[2, :66]
        grid_user_reader_hs[2, 68:83] = create_widget('SelectFilesButton', {'width': '95%'}, **{'option':'askopenfilename','open_filetypes':(('python files', '*.py'),), 'text_h':self.hs['IOSpecTomoRdr text']})
        self.hs['IOSpecXANES3DRdr btn'] = grid_user_reader_hs[2, 68:83]
        self.hs['IOSpecXANES3DRdr btn'].description = 'XANES3D Reader' 
        
        self.hs['IOSpecTomoRdr btn'].on_click(self.io_spec_tomo_rdr_btn_clk)
        self.hs['IOSpecXANES2DRdr btn'].on_click(self.io_spec_xanes2D_rdr_btn_clk)
        self.hs['IOSpecXANES3DRdr btn'].on_click(self.io_spec_xanes3D_rdr_btn_clk)
        self.hs['IOUserRdr box'].children = [self.hs['IOSpecTomoRdr text'],
                                             self.hs['IOSpecTomoRdr btn'],
                                             self.hs['IOSpecXANES2DRdr text'],
                                             self.hs['IOSpecXANES2DRdr btn'],
                                             self.hs['IOSpecXANES3DRdr text'],
                                             self.hs['IOSpecXANES3DRdr btn']]
        
        self.hs['IOUserSpec box'].children = [self.hs['IOUserRdr box']]
        ## ## ## ## user reader load -- end
        
        ## ## ## default file config -- start
        self.hs['FnPatn box'] = create_widget('VBox', {'border':'3px solid #FFCC00', 'width': '100%', 'height': f'{0.35*(self.form_sz[0]-136)}px'})
        
        ## ## ## ## default file patterns -- start
        self.hs['FnDefPatn box'] = create_widget('VBox', {'border':'3px solid #8855AA', 'width': '100%', 'height': '100%'})
        
        grid_fn_pattern_hs = GridspecLayout(5, 100,
                                            layout = {"border":"3px solid #FFCC00",
                                                      "width": "100%",
                                                      "height": "100%",
                                                      "align_items":"flex-start",
                                                      "justify_items":"flex-start"})
        self.hs['FnDefEdit box'] = grid_fn_pattern_hs
        
        grid_fn_pattern_hs[0, 38:70] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; align: center; background-color:rgb(135,206,250);">' + 'Default Filename Patterns' + '</span>')
        self.hs['FnDefPatn label'] = grid_fn_pattern_hs[0, 38:70]
        grid_fn_pattern_hs[1, :50] = create_widget('Text', {'width': '95%'}, **{'description':'tomo raw fn', 'value':'fly_scan_id_{0}.h5', 'disabled':False, 'description_tooltip':'default tomo raw data file name pattern'})
        self.hs['FnTomoDefRawPatn text'] = grid_fn_pattern_hs[1, :50]
        grid_fn_pattern_hs[1, 50:] = create_widget('Text', {'width': '95%'}, **{'description':'xanes2D raw fn', 'value':'xanes_scan2_id_{0}.h5', 'disabled':False, 'description_tooltip':'default xanes2D raw data file name pattern'})
        self.hs['FnXANES2DDefRawPatn text'] = grid_fn_pattern_hs[1, 50:]
        grid_fn_pattern_hs[2, :50] = create_widget('Text', {'width': '95%'}, **{'description':'xanes3D raw tomo fn', 'value':'fly_scan_id_{0}.h5', 'disabled':False, 'description_tooltip':'default XANES3D raw tomo data file name pattern'})
        self.hs['FnXANES3DDefRawPatn text'] = grid_fn_pattern_hs[2, :50]
        grid_fn_pattern_hs[2, 50:] = create_widget('Text', {'width': '95%'}, **{'description':'xanes3D tomo recon dir', 'value':'recon_fly_scan_id_{0}', 'disabled':False, 'description_tooltip':'default XANES3D tomo recon directory name pattern'})
        self.hs['FnXANES3DDefReconDirPatn text'] = grid_fn_pattern_hs[2, 50:]
        grid_fn_pattern_hs[3, :50] = create_widget('Text', {'width': '95%'}, **{'description':'xanes3D tomo recon fn', 'value':'fly_scan_id_{0}.h5', 'disabled':False, 'description_tooltip':'default XANES3D tomo recon file name pattern'})
        self.hs['FnXANES3DDefReconFnPatn text'] = grid_fn_pattern_hs[3, :50]
        grid_fn_pattern_hs[4, 44:60] = create_widget('Button', {'width': '95%'}, **{'description':'Confirm', 'disabled':False})
        self.hs['FnDefPatnCfm btn'] = grid_fn_pattern_hs[4, 44:60]
        self.hs['FnDefPatnCfm btn'].style.button_color = 'darkviolet'
        self.hs['FnDefPatnCfm btn'].on_click(self.io_fn_def_patn_cfm_btn_clk)
        
        self.hs['FnDefEdit box'].children = [self.hs['FnDefPatn label'],
                                             self.hs['FnTomoDefRawPatn text'],
                                             self.hs['FnXANES2DDefRawPatn text'],
                                             self.hs['FnXANES3DDefRawPatn text'],
                                             self.hs['FnXANES3DDefReconDirPatn text'],
                                             self.hs['FnXANES3DDefReconFnPatn text'],
                                             self.hs['FnDefPatnCfm btn']]
        
        self.hs['FnDefPatn box'].children = [self.hs['FnDefEdit box']]
        ## ## ## ## default file patterns -- end
        
        self.hs['FnPatn box'].children = [self.hs['FnDefPatn box']]
        ## ## ## default file config -- end

        self.hs['IOOptn form'].children = [self.hs['IOOptn box'],
                                           self.hs['IOUserSpec box'],
                                           self.hs['FnPatn box']]
        ## ## ## reader option -- end
        
        ## ## ## structured h5 -- start
        self.hs['IOConfig form'] = create_widget('VBox', {'border':'3px solid #FFCC00', 'width': '100%', 'height': '100%'})
        
        ## ## ## ## tomo io config -- start        
        self.hs['IOTomoConfig box'] = create_widget('VBox', {'border':'3px solid #8855AA', 'width': '100%', 'height':f'{0.31*(self.form_sz[0]-136)}px'})
        
        grid_tomo_data_hs = GridspecLayout(2, 100,
                                    layout = {"border":"3px solid #FFCC00",
                                              "width": "100%",
                                              "height": '40%',
                                              "align_items":"flex-start",
                                              "justify_items":"flex-start"})
        self.hs['IOTomoConfigData box'] = grid_tomo_data_hs
        grid_tomo_data_hs[0, 37:70] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; align: center; background-color:rgb(135,206,250);">' + 'Tomo File Data Structure' + '</span>')
        self.hs['IOTomoConfigData label'] = grid_tomo_data_hs[0, 37:70]
        grid_tomo_data_hs[1, 0:25] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'data path', 'value':'/img_tomo', 'disabled':False, 'description_tooltip':'path to sample image data in h5 file', 'indent':False})
        self.hs['IOTomoConfigDataImg text'] = grid_tomo_data_hs[1, 0:25]
        grid_tomo_data_hs[1, 25:50] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'flat path', 'value':'/img_bkg', 'disabled':False, 'description_tooltip':'path to flat (reference) image data in h5 file', 'indent':False})
        self.hs['IOTomoConfigDataFlat text'] = grid_tomo_data_hs[1, 25:50]
        grid_tomo_data_hs[1, 50:75] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'dark path', 'value':'/img_dark', 'disabled':False, 'description_tooltip':'path to dark image data in h5 file', 'indent':False})
        self.hs['IOTomoConfigDataDark text'] = grid_tomo_data_hs[1, 50:75]
        grid_tomo_data_hs[1, 75:100] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'theta path', 'value':'/angle', 'disabled':False, 'description_tooltip':'path to theta in h5 file', 'indent':False})
        self.hs['IOTomoConfigDataTheta text'] = grid_tomo_data_hs[1, 75:100]
        
        self.hs['IOTomoConfigData box'].children = [self.hs['IOTomoConfigData label'],
                                                    self.hs['IOTomoConfigDataImg text'],
                                                    self.hs['IOTomoConfigDataFlat text'],
                                                    self.hs['IOTomoConfigDataDark text'],
                                                    self.hs['IOTomoConfigDataTheta text']]
        
        grid_tomo_info_hs = GridspecLayout(3, 100,
                                    layout = {"border":"3px solid #FFCC00",
                                              "width": "100%",
                                              "height": "60%",
                                              "align_items":"flex-start",
                                              "justify_items":"flex-start"})
        self.hs['IOTomoConfigInfo box'] = grid_tomo_info_hs
        grid_tomo_info_hs[0, 38:70] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; align: center; background-color:rgb(135,206,250);">' + 'Config Tomo Data Info' + '</span>')
        self.hs['IOTomoConfigInfo label'] = grid_tomo_info_hs[0, 38:70]
        grid_tomo_info_hs[1, 0:25] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info0 path', 'value':'/img_tomo', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOTomoConfigInfo0 text'] = grid_tomo_info_hs[1, 0:25]
        grid_tomo_info_hs[1, 25:50] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info1 path', 'value':'/angle', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOTomoConfigInfo1 text'] = grid_tomo_info_hs[1, 25:50]
        grid_tomo_info_hs[1, 50:75] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info2 path', 'value':'/Magnification', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOTomoConfigInfo2 text'] = grid_tomo_info_hs[1, 50:75]
        grid_tomo_info_hs[1, 75:] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info3 path', 'value':'/Pixel Size', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOTomoConfigInfo3 text'] = grid_tomo_info_hs[1, 75:]
        
        grid_tomo_info_hs[2, 0:25] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info4 path', 'value':'/X_eng', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOTomoConfigInfo4 text'] = grid_tomo_info_hs[2, 0:25]
        grid_tomo_info_hs[2, 25:50] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info5 path', 'value':'/note', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOTomoConfigInfo5 text'] = grid_tomo_info_hs[2, 25:50]
        grid_tomo_info_hs[2, 50:75] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info6 path', 'value':'/scan_time', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOTomoConfigInfo6 text'] = grid_tomo_info_hs[2, 50:75]
        grid_tomo_info_hs[2, 75:] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info7 path', 'value':'', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOTomoConfigInfo7 text'] = grid_tomo_info_hs[2, 75:]
        
        self.hs['IOTomoConfigInfo box'].children = [self.hs['IOTomoConfigInfo label'],
                                                    self.hs['IOTomoConfigInfo0 text'],
                                                    self.hs['IOTomoConfigInfo1 text'],
                                                    self.hs['IOTomoConfigInfo2 text'],
                                                    self.hs['IOTomoConfigInfo3 text'],
                                                    self.hs['IOTomoConfigInfo4 text'],
                                                    self.hs['IOTomoConfigInfo5 text'],
                                                    self.hs['IOTomoConfigInfo6 text'],
                                                    self.hs['IOTomoConfigInfo7 text']]
        
        self.hs['IOTomoConfig box'].children = [self.hs['IOTomoConfigData box'],
                                                self.hs['IOTomoConfigInfo box']]
        ## ## ## ## tomo io config -- end
        
        ## ## ## ## xanes2D io config -- start 
        self.hs['IOXANES2DConfig box'] = create_widget('VBox', {'border':'3px solid #8855AA', 'width': '100%', 'height':f'{0.31*(self.form_sz[0]-136)}px'})
        
        grid_xanes2D_data_hs = GridspecLayout(2, 100,
                                    layout = {"border":"3px solid #FFCC00",
                                              "width": "100%",
                                              "height": "40%",
                                              "align_items":"flex-start",
                                              "justify_items":"flex-start"})
        self.hs['IOXANES2DConfigData box'] = grid_xanes2D_data_hs
        grid_xanes2D_data_hs[0, 35:70] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; align: center; background-color:rgb(135,206,250);">' + 'XANES2D File Data Structure' + '</span>')
        self.hs['IOXANES2DConfigData label'] = grid_xanes2D_data_hs[0, 35:70]
        grid_xanes2D_data_hs[1, 0:25] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'data path', 'value':'/img_xanes', 'disabled':False, 'description_tooltip':'path to xanes image data in h5 file', 'indent':False})
        self.hs['IOXANES2DConfigDataImg text'] = grid_xanes2D_data_hs[1, 0:25]
        grid_xanes2D_data_hs[1, 25:50] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'flat path', 'value':'/img_bkg', 'disabled':False, 'description_tooltip':'path to flat (reference) image data in h5 file', 'indent':False})
        self.hs['IOXANES2DConfigDataFlat text'] = grid_xanes2D_data_hs[1, 25:50]
        grid_xanes2D_data_hs[1, 50:75] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'dark path', 'value':'/img_dark', 'disabled':False, 'description_tooltip':'path to dark image data in h5 file', 'indent':False})
        self.hs['IOXANES2DConfigDataDark text'] = grid_xanes2D_data_hs[1, 50:75]
        grid_xanes2D_data_hs[1, 75:100] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'eng path', 'value':'/X_eng', 'disabled':False, 'description_tooltip':'path to x-ray energy in h5 file', 'indent':False})
        self.hs['IOXANES2DConfigDataEng text'] = grid_xanes2D_data_hs[1, 75:100]
        
        self.hs['IOXANES2DConfigData box'].children = [self.hs['IOXANES2DConfigData label'],
                                                       self.hs['IOXANES2DConfigDataImg text'],
                                                       self.hs['IOXANES2DConfigDataFlat text'],
                                                       self.hs['IOXANES2DConfigDataDark text'],
                                                       self.hs['IOXANES2DConfigDataEng text']]
        
        grid_xanes2D_info_hs = GridspecLayout(3, 100,
                                    layout = {"border":"3px solid #FFCC00",
                                              "width": "100%",
                                              "height": "60%",
                                              "align_items":"flex-start",
                                              "justify_items":"flex-start"})
        self.hs['IOXANES2DConfigInfo box'] = grid_xanes2D_info_hs
        grid_xanes2D_info_hs[0, 37:70] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; align: center; background-color:rgb(135,206,250);">' + 'Config XANES2D Data Info' + '</span>')
        self.hs['IOXANES2DConfigInfo label'] = grid_xanes2D_info_hs[0, 37:70]
        grid_xanes2D_info_hs[1, 0:25] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info0 path', 'value':'/img_xanes', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOXANES2DConfigInfo0 text'] = grid_xanes2D_info_hs[1, 0:25]
        grid_xanes2D_info_hs[1, 25:50] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info1 path', 'value':'X_eng', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOXANES2DConfigInfo1 text'] = grid_xanes2D_info_hs[1, 25:50]
        grid_xanes2D_info_hs[1, 50:75] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info2 path', 'value':'/Magnification', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOXANES2DConfigInfo2 text'] = grid_xanes2D_info_hs[1, 50:75]
        grid_xanes2D_info_hs[1, 75:] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info3 path', 'value':'/Pixel Size', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOXANES2DConfigInfo3 text'] = grid_xanes2D_info_hs[1, 75:]
        
        grid_xanes2D_info_hs[2, 0:25] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info4 path', 'value':'/note', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOXANES2DConfigInfo4 text'] = grid_xanes2D_info_hs[2, 0:25]
        grid_xanes2D_info_hs[2, 25:50] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info5 path', 'value':'/scan_time', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOXANES2DConfigInfo5 text'] = grid_xanes2D_info_hs[2, 25:50]
        grid_xanes2D_info_hs[2, 50:75] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info6 path', 'value':'', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOXANES2DConfigInfo6 text'] = grid_xanes2D_info_hs[2, 50:75]
        grid_xanes2D_info_hs[2, 75:] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info7 path', 'value':'', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOXANES2DConfigInfo7 text'] = grid_xanes2D_info_hs[2, 75:]
        
        self.hs['IOXANES2DConfigInfo box'].children = [self.hs['IOXANES2DConfigInfo label'],
                                                       self.hs['IOXANES2DConfigInfo0 text'],
                                                       self.hs['IOXANES2DConfigInfo1 text'],
                                                       self.hs['IOXANES2DConfigInfo2 text'],
                                                       self.hs['IOXANES2DConfigInfo3 text'],
                                                       self.hs['IOXANES2DConfigInfo4 text'],
                                                       self.hs['IOXANES2DConfigInfo5 text'],
                                                       self.hs['IOXANES2DConfigInfo6 text'],
                                                       self.hs['IOXANES2DConfigInfo7 text'],]
        
        self.hs['IOXANES2DConfig box'].children = [self.hs['IOXANES2DConfigData box'],
                                                   self.hs['IOXANES2DConfigInfo box']]
        ## ## ## ## xanes2D io config -- end
        
        ## ## ## ## xanes3D io config -- start 
        self.hs['IOXANES3DConfig box'] = create_widget('VBox', {'border':'3px solid #8855AA', 'width': '100%', 'height':f'{0.31*(self.form_sz[0]-136)}px'})
        
        grid_xanes3D_data_hs = GridspecLayout(2, 100,
                                    layout = {"border":"3px solid #FFCC00",
                                              "width": "100%",
                                              "height": "40%",
                                              "align_items":"flex-start",
                                              "justify_items":"flex-start"})
        self.hs['IOXANES3DConfigData box'] = grid_xanes3D_data_hs
        grid_xanes3D_data_hs[0, 35:70] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; align: center; background-color:rgb(135,206,250);">' + 'XANES3D File Data Structure' + '</span>')
        self.hs['IOXANES3DConfigData label'] = grid_xanes3D_data_hs[0, 35:70]
        grid_xanes3D_data_hs[1, 0:25] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'data path', 'value':'/img_tomo', 'disabled':False, 'description_tooltip':'path to sample image data in h5 file', 'indent':False})
        self.hs['IOXANES3DConfigDataImg text'] = grid_xanes3D_data_hs[1, 0:25]
        grid_xanes3D_data_hs[1, 25:50] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'flat path', 'value':'/img_bkg', 'disabled':False, 'description_tooltip':'path to flat (reference) image data in h5 file', 'indent':False})
        self.hs['IOXANES3DConfigDataFlat text'] = grid_xanes3D_data_hs[1, 25:50]
        grid_xanes3D_data_hs[1, 50:75] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'dark path', 'value':'/img_dark', 'disabled':False, 'description_tooltip':'path to dark image data in h5 file', 'indent':False})
        self.hs['IOXANES3DConfigDataDark text'] = grid_xanes3D_data_hs[1, 50:75]
        grid_xanes3D_data_hs[1, 75:100] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'eng path', 'value':'/X_eng', 'disabled':False, 'description_tooltip':'path to x-ray energy in h5 file', 'indent':False})
        self.hs['IOXANES3DConfigDataEng text'] = grid_xanes3D_data_hs[1, 75:100]
        
        self.hs['IOXANES3DConfigData box'].children = [self.hs['IOXANES3DConfigData label'],
                                                       self.hs['IOXANES3DConfigDataImg text'],
                                                       self.hs['IOXANES3DConfigDataFlat text'],
                                                       self.hs['IOXANES3DConfigDataDark text'],
                                                       self.hs['IOXANES3DConfigDataEng text']]
        
        grid_xanes3D_info_hs = GridspecLayout(3, 100,
                                    layout = {"border":"3px solid #FFCC00",
                                              "width": "100%",
                                              "height": "60%",
                                              "align_items":"flex-start",
                                              "justify_items":"flex-start"})
        self.hs['IOXANES3DConfigInfo box'] = grid_xanes3D_info_hs
        grid_xanes3D_info_hs[0, 37:70] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; align: center; background-color:rgb(135,206,250);">' + 'Config XANES3D Data Info' + '</span>')
        self.hs['IOXANES3DConfigInfo label'] = grid_xanes3D_info_hs[0, 37:70]
        grid_xanes3D_info_hs[1, 0:25] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info0 path', 'value':'/img_tomo', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOXANES3DConfigInfo0 text'] = grid_xanes3D_info_hs[1, 0:25]
        grid_xanes3D_info_hs[1, 25:50] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info1 path', 'value':'/angle', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOXANES3DConfigInfo1 text'] = grid_xanes3D_info_hs[1, 25:50]
        grid_xanes3D_info_hs[1, 50:75] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info2 path', 'value':'/Magnification', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOXANES3DConfigInfo2 text'] = grid_xanes3D_info_hs[1, 50:75]
        grid_xanes3D_info_hs[1, 75:] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info3 path', 'value':'/Pixel Size', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOXANES3DConfigInfo3 text'] = grid_xanes3D_info_hs[1, 75:]
        
        grid_xanes3D_info_hs[2, 0:25] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info4 path', 'value':'/X_eng', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOXANES3DConfigInfo4 text'] = grid_xanes3D_info_hs[2, 0:25]
        grid_xanes3D_info_hs[2, 25:50] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info5 path', 'value':'/note', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOXANES3DConfigInfo5 text'] = grid_xanes3D_info_hs[2, 25:50]
        grid_xanes3D_info_hs[2, 50:75] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info6 path', 'value':'/scan_time', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOXANES3DConfigInfo6 text'] = grid_xanes3D_info_hs[2, 50:75]
        grid_xanes3D_info_hs[2, 75:] = create_widget('Text', {'width': '95%', 'height': '95%'}, **{'description':'info7 path', 'value':'', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['IOXANES3DConfigInfo7 text'] = grid_xanes3D_info_hs[2, 75:]
        
        self.hs['IOXANES3DConfigInfo box'].children = [self.hs['IOXANES3DConfigInfo label'],
                                                       self.hs['IOXANES3DConfigInfo0 text'],
                                                       self.hs['IOXANES3DConfigInfo1 text'],
                                                       self.hs['IOXANES3DConfigInfo2 text'],
                                                       self.hs['IOXANES3DConfigInfo3 text'],
                                                       self.hs['IOXANES3DConfigInfo4 text'],
                                                       self.hs['IOXANES3DConfigInfo5 text'],
                                                       self.hs['IOXANES3DConfigInfo6 text'],
                                                       self.hs['IOXANES3DConfigInfo7 text']]
        
        self.hs['IOXANES3DConfig box'].children = [self.hs['IOXANES3DConfigData box'],
                                                   self.hs['IOXANES3DConfigInfo box']]
        ## ## ## ## xanes3D io config -- end
        
        ## ## ## ## confirm -- start
        grid_confirm_hs = GridspecLayout(1, 100,
                                    layout = {"border":"3px solid #FFCC00",
                                              "width": "100%",
                                              "height":f'{0.06*(self.form_sz[0]-136)}px',
                                              "align_items":"flex-start",
                                              "justify_items":"flex-start"})
        self.hs['IOCfm box'] = grid_confirm_hs
        grid_confirm_hs[0, 44:60] = create_widget('Button', {"width": "95%", "height": "95%"}, **{'description':'Confirm', 'disabled':False})
        self.hs['IOCfm btn'] = grid_confirm_hs[0, 44:60]
        self.hs['IOCfm btn'].style.button_color = 'darkviolet'
        
        self.hs['IOCfm btn'].on_click(self.io_cfm_btn_clk)
        self.hs['IOCfm box'].children = [self.hs['IOCfm btn']]
        ## ## ## ## confirm -- end
        
        self.hs['IOConfig form'].children = [self.hs['IOTomoConfig box'],
                                             self.hs['IOXANES2DConfig box'],
                                             self.hs['IOXANES3DConfig box'],
                                             self.hs['IOCfm box']]
        ## ## ## structured h5 -- end

    def io_optn_chbx_chg(self, a):
        self.boxes_logics()

    def io_cfm_btn_clk(self, a):
        save_io_config(self)
        with open(self.global_h.io_data_struc_tomo_cfg_file, 'r') as f:
            self.global_h.io_tomo_cfg = json.load(f)
            self.global_h.tomo_recon_gui.tomo_raw_fn_temp = self.global_h.io_tomo_cfg['tomo_raw_fn_template']
        with open(self.global_h.io_data_struc_xanes2D_cfg_file, 'r') as f:
            self.global_h.io_xanes2D_cfg = json.load(f)
            self.global_h.xanes2D_gui.xanes2D_raw_fn_temp = self.global_h.io_xanes2D_cfg['xanes2D_raw_fn_template']
        with open(self.global_h.io_data_struc_xanes3D_cfg_file, 'r') as f:
            self.global_h.io_xanes3D_cfg = json.load(f)
            self.global_h.xanes3D_gui.xanes3D_raw_fn_temp = self.global_h.io_xanes3D_cfg['tomo_raw_fn_template']
            self.global_h.xanes3D_gui.xanes3D_recon_dir_temp = self.global_h.io_xanes3D_cfg['xanes3D_recon_dir_template']
            self.global_h.xanes3D_gui.xanes3D_recon_fn_temp = self.global_h.io_xanes3D_cfg['xanes3D_recon_fn_template']

    def io_spec_tomo_rdr_btn_clk(self, a):
        if len(a.files[0]) != 0:
            self.hs['IOSpecTomoRdr text'].value = a.files[0]        
            save_io_config(self)
            with open(self.global_h.io_data_struc_tomo_cfg_file, 'r') as f:
                self.global_h.io_tomo_cfg = json.load(f)

    def io_spec_xanes2D_rdr_btn_clk(self, a):
        if len(a.files[0]) != 0:
            self.hs['IOSpecXANES2DRdr text'].value = a.files[0]
            save_io_config(self)
            with open(self.global_h.io_data_struc_xanes2D_cfg_file, 'r') as f:
                self.global_h.io_xanes2D_cfg = json.load(f)

    def io_spec_xanes3D_rdr_btn_clk(self, a):
        if len(a.files[0]) != 0:
            self.hs['IOSpecXANES3DRdr text'].value = a.files[0]
            save_io_config(self)
            with open(self.global_h.io_data_struc_xanes3D_cfg_file, 'r') as f:
                self.global_h.io_xanes3D_cfg = json.load(f)

    def io_fn_def_patn_cfm_btn_clk(self, a):
        save_io_config(self)
        with open(self.global_h.io_data_struc_tomo_cfg_file, 'r') as f:
            self.global_h.io_tomo_cfg = json.load(f)
            self.global_h.tomo_recon_gui.tomo_raw_fn_temp = self.global_h.io_tomo_cfg['tomo_raw_fn_template']
        with open(self.global_h.io_data_struc_xanes2D_cfg_file, 'r') as f:
            self.global_h.io_xanes2D_cfg = json.load(f)
            self.global_h.xanes2D_gui.xanes2D_raw_fn_temp = self.global_h.io_xanes2D_cfg['xanes2D_raw_fn_template']
        with open(self.global_h.io_data_struc_xanes3D_cfg_file, 'r') as f:
            self.global_h.io_xanes3D_cfg = json.load(f)
            self.global_h.xanes3D_gui.xanes3D_raw_fn_temp = self.global_h.io_xanes3D_cfg['tomo_raw_fn_template']
            self.global_h.xanes3D_gui.xanes3D_recon_dir_temp = self.global_h.io_xanes3D_cfg['xanes3D_recon_dir_template']
            self.global_h.xanes3D_gui.xanes3D_recon_fn_temp = self.global_h.io_xanes3D_cfg['xanes3D_recon_fn_template']
        