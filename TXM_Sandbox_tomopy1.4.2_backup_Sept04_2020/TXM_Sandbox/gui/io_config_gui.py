#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:45:03 2020

@author: xiao
"""
import json

from ipywidgets import widgets, GridspecLayout, dlink

from .gui_components import (get_handles, create_widget, 
                             enable_disable_boxes, save_io_config)

class io_config_gui():
    def __init__(self, parent_h, form_sz=[650, 740]):
        self.gui_name = 'io_config'
        self.form_sz = form_sz
        self.global_h = parent_h
        self.hs = {}
    
    def boxes_logics(self):
        if self.global_h.use_struc_h5_reader:
            boxes = ['L[0][3][0][1][0]_io_user_reader_box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            boxes = ['L[0][3][1][0][0]_io_tomo_data_box',
                     'L[0][3][1][0][1]_io_tomo_info_box'
                     'L[0][3][1][1][0]_io_xanes2D_data_box',
                     'L[0][3][1][1][1]_io_xanes2D_info_box',
                     'L[0][3][1][2][0]_io_xanes3D_data_box',
                     'L[0][3][1][2][1]_io_xanes3D_info_box',
                     'L[0][3][1][3]_io_confirm_box']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
        else:
            boxes = ['L[0][3][1][0][0]_io_tomo_data_box',
                     'L[0][3][1][0][1]_io_tomo_info_box'
                     'L[0][3][1][1][0]_io_xanes2D_data_box',
                     'L[0][3][1][1][1]_io_xanes2D_info_box',
                     'L[0][3][1][2][0]_io_xanes3D_data_box',
                     'L[0][3][1][2][1]_io_xanes3D_info_box',
                     'L[0][3][1][3]_io_confirm_box']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            boxes = ['L[0][3][0][1][0]_io_user_reader_box']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1) 
    
    def build_gui(self):
        ## ## ## reader option -- start
        self.hs['L[0][3][0]_io_option_form'] = create_widget('VBox', {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-86}px', 'height':f'{self.form_sz[0]-136}px'})
        
        ## ## ## ## reader option -- start
        self.hs['L[0][3][0][0]_io_option_box'] = create_widget('VBox', {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.07*(self.form_sz[0]-136)}px'})
        self.hs['L[0][3][0][0][0]_io_option_checkbox'] = create_widget('Checkbox', {'width':f'{0.23*(self.form_sz[1]-86)}px', 'height':'90%'}, **{'description':'Struct h5', 'value':True, 'disabled':False, 'description_tooltip':'Check this if the data is saved in a h5 file with a structure of separated datasets for data, flat, dark, energy, and angle etc.', 'indent':False})
        self.hs['L[0][3][0][0][0]_io_option_checkbox'].observe(self.L0_3_0_0_0_io_option_checkbox_change, names='value')
        self.hs['L[0][3][0][0]_io_option_box'].children = get_handles(self.hs, 'L[0][3][0][0]_io_option_box', -1)
        ## ## ## ## reader option -- end
        
        ## ## ## ## user reader load -- start
        self.hs['L[0][3][0][1]_io_user_spec_box'] = create_widget('VBox', {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.23*(self.form_sz[0]-136)}px'})
        
        grid_user_reader_hs = GridspecLayout(3, 100,
                                    layout = {"border":"3px solid #FFCC00",
                                              "height":f'{0.23*(self.form_sz[0]-136)}px',
                                              "align_items":"flex-start",
                                              "justify_items":"flex-start"})
        self.hs['L[0][3][0][1][0]_io_user_reader_box'] = grid_user_reader_hs
        grid_user_reader_hs[0, :66] = create_widget('Text', {'width':f'{0.66*(self.form_sz[1]-86)}px', "height":f'{0.06*(self.form_sz[0]-136)}px'}, **{'disabled':True, 'value':'load tomo data reader'})
        self.hs['L[0][3][0][1][0][0]_io_spec_tomo_reader_text'] = grid_user_reader_hs[0, :66]
        grid_user_reader_hs[0, 68:83] = create_widget('SelectFilesButton', {'width':f'{0.15*(self.form_sz[1]-86)}px', "height":f'{0.043*(self.form_sz[0]-136)}px'}, **{'option':'askopenfilename','open_filetypes':(('python files', '*.py'),), 'text_h':self.hs['L[0][3][0][1][0][0]_io_spec_tomo_reader_text']})
        self.hs['L[0][3][0][1][0][1]_io_spec_tomo_reader_button'] = grid_user_reader_hs[0, 68:83]
        self.hs['L[0][3][0][1][0][1]_io_spec_tomo_reader_button'].description = 'Tomo Reader' 
        grid_user_reader_hs[1, :66] = create_widget('Text', {'width':f'{0.66*(self.form_sz[1]-86)}px', "height":f'{0.06*(self.form_sz[0]-136)}px'}, **{'disabled':True, 'value':'load xanes2D data reader'})
        self.hs['L[0][3][0][1][0][2]_io_spec_xanes2D_reader_text'] = grid_user_reader_hs[1, :66]
        grid_user_reader_hs[1, 68:83] = create_widget('SelectFilesButton', {'width':f'{0.15*(self.form_sz[1]-86)}px', "height":f'{0.043*(self.form_sz[0]-136)}px'}, **{'option':'askopenfilename','open_filetypes':(('python files', '*.py'),), 'text_h':self.hs['L[0][3][0][1][0][0]_io_spec_tomo_reader_text']})
        self.hs['L[0][3][0][1][0][3]_io_spec_xanes2D_reader_button'] = grid_user_reader_hs[1, 68:83]
        self.hs['L[0][3][0][1][0][3]_io_spec_xanes2D_reader_button'].description = 'XANES2D Reader' 
        grid_user_reader_hs[2, :66] = create_widget('Text', {'width':f'{0.66*(self.form_sz[1]-86)}px', "height":f'{0.06*(self.form_sz[0]-136)}px'}, **{'disabled':True, 'value':'load xanes3D data reader'})
        self.hs['L[0][3][0][1][0][4]_io_spec_xanes3D_reader_text'] = grid_user_reader_hs[2, :66]
        grid_user_reader_hs[2, 68:83] = create_widget('SelectFilesButton', {'width':f'{0.15*(self.form_sz[1]-86)}px', "height":f'{0.043*(self.form_sz[0]-136)}px'}, **{'option':'askopenfilename','open_filetypes':(('python files', '*.py'),), 'text_h':self.hs['L[0][3][0][1][0][0]_io_spec_tomo_reader_text']})
        self.hs['L[0][3][0][1][0][5]_io_spec_xanes3D_reader_button'] = grid_user_reader_hs[2, 68:83]
        self.hs['L[0][3][0][1][0][5]_io_spec_xanes3D_reader_button'].description = 'XANES3D Reader' 
        
        self.hs['L[0][3][0][1][0][1]_io_spec_tomo_reader_button'].on_click(self.L0_3_0_1_0_1_io_spec_tomo_reader_button_click)
        self.hs['L[0][3][0][1][0][3]_io_spec_xanes2D_reader_button'].on_click(self.L0_3_0_1_0_3_io_spec_xanes2D_reader_button_click)
        self.hs['L[0][3][0][1][0][5]_io_spec_xanes3D_reader_button'].on_click(self.L0_3_0_1_0_5_io_spec_xanes3D_reader_button_click)
        self.hs['L[0][3][0][1][0]_io_user_reader_box'].children = get_handles(self.hs, 'L[0][3][0][1][0]_io_user_reader_box', -1)
        
        self.hs['L[0][3][0][1]_io_user_spec_box'].children = get_handles(self.hs, 'L[0][3][0][1]_io_user_spec_box', -1)
        ## ## ## ## user reader load -- end
        
        self.hs['L[0][3][0]_io_option_form'].children = get_handles(self.hs, 'L[0][3][0]_io_option_form', -1)
        ## ## ## reader option -- end
        
        ## ## ## default file config -- start
        self.hs['L[0][3][2]_fn_pattern_form'] = create_widget('VBox', {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-86}px', 'height':f'{self.form_sz[0]-136}px'})
        
        ## ## ## ## default file patterns -- start
        self.hs['L[0][3][2][0]_fn_def_patt_box'] = create_widget('VBox', {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.31*(self.form_sz[0]-136)}px'})
        
        grid_fn_pattern_hs = GridspecLayout(6, 100,
                                            layout = {"border":"3px solid #FFCC00",
                                                      "height":f'{0.36*(self.form_sz[0]-136)}px',
                                                      "align_items":"flex-start",
                                                      "justify_items":"flex-start"})
        self.hs['L[0][3][2][0][0]_fn_def_edit_box'] = grid_fn_pattern_hs
        
        grid_fn_pattern_hs[0, 38:70] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; align: center; background-color:rgb(135,206,250);">' + 'Default Filename Patterns' + '</span>')
        self.hs['L[0][3][2][0][0][0]_fn_def_patt_html'] = grid_fn_pattern_hs[0, 38:70]
        grid_fn_pattern_hs[1, :50] = create_widget('Text', {'width':f'{0.45*(self.form_sz[1]-86)}px'}, **{'description':'tomo raw fn', 'value':'fly_scan_id_{}.h5', 'disabled':False, 'description_tooltip':'default tomo raw data file name pattern'})
        self.hs['L[0][3][2][0][0][1]_fn_tomo_def_raw_patt_text'] = grid_fn_pattern_hs[1, :50]
        grid_fn_pattern_hs[1, 50:] = create_widget('Text', {'width':f'{0.45*(self.form_sz[1]-86)}px'}, **{'description':'xanes2D raw fn', 'value':'xanes_scan2_id_{}.h5', 'disabled':False, 'description_tooltip':'default xanes2D raw data file name pattern'})
        self.hs['L[0][3][2][0][0][2]_fn_xanes2D_def_raw_patt_text'] = grid_fn_pattern_hs[1, 50:]
        grid_fn_pattern_hs[2, :50] = create_widget('Text', {'width':f'{0.45*(self.form_sz[1]-86)}px'}, **{'description':'xanes3D raw tomo fn', 'value':'fly_scan_id_{}.h5', 'disabled':False, 'description_tooltip':'default raw tomo data file name pattern in xanes3D'})
        self.hs['L[0][3][2][0][0][3]_fn_xanes3D_def_tomo_raw_patt_text'] = grid_fn_pattern_hs[2, :50]
        grid_fn_pattern_hs[2, 50:] = create_widget('Text', {'width':f'{0.45*(self.form_sz[1]-86)}px'}, **{'description':'xanes3D tomo recon dir', 'value':'recon_fly_scan_id_{}', 'disabled':False, 'description_tooltip':'default tomo recon directory name pattern in xanes3D'})
        self.hs['L[0][3][2][0][0][4]_fn_xanes3D_def_tomo_recon_dir_patt_text'] = grid_fn_pattern_hs[2, 50:]
        grid_fn_pattern_hs[3, :50] = create_widget('Text', {'width':f'{0.45*(self.form_sz[1]-86)}px'}, **{'description':'xanes3D raw tomo fn', 'value':'fly_scan_id_{}.h5', 'disabled':False, 'description_tooltip':'default tomo recon file name pattern in xanes3D'})
        self.hs['L[0][3][2][0][0][5]_fn_xanes3D_def_tomo_recon_fn_patt_text'] = grid_fn_pattern_hs[3, :50]
        grid_fn_pattern_hs[5, 44:60] = create_widget('Button', {'width':f'{0.15*(self.form_sz[1]-86)}px', "height":f'{0.045*(self.form_sz[0]-136)}px'}, **{'description':'Confirm', 'disabled':False})
        self.hs['L[0][3][2][0][0][6]_fn_def_patt_confirm_button'] = grid_fn_pattern_hs[5, 44:60]
        self.hs['L[0][3][2][0][0][6]_fn_def_patt_confirm_button'].style.button_color = 'darkviolet'
        self.hs['L[0][3][2][0][0][6]_fn_def_patt_confirm_button'].on_click(self.L0_3_2_0_0_6_fn_def_patt_confirm_button_click)
        
        self.hs['L[0][3][2][0][0]_fn_def_edit_box'].children = get_handles(self.hs, 'L[0][3][2][0][0]_fn_def_edit_box', -1)
        
        self.hs['L[0][3][2][0]_fn_def_patt_box'].children = get_handles(self.hs, 'L[0][3][2][0]_file_default_pattern_box', -1)
        ## ## ## ## default file patterns -- end
        
        self.hs['L[0][3][2]_fn_pattern_form'].children = get_handles(self.hs, 'L[0][3][2]_file_pattern_form', -1)
        ## ## ## default file config -- end
        
        ## ## ## structured h5 -- start
        self.hs['L[0][3][1]_io_config_form'] = create_widget('VBox', {'border':'3px solid #FFCC00', 'width':f'{self.form_sz[1]-86}px', 'height':f'{self.form_sz[0]-136}px'})
        
        ## ## ## ## tomo io config -- start        
        self.hs['L[0][3][1][0]_io_tomo_config_box'] = create_widget('VBox', {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.31*(self.form_sz[0]-136)}px'})
        
        grid_tomo_data_hs = GridspecLayout(2, 100,
                                    layout = {"border":"3px solid #FFCC00",
                                              "height":f'{0.12*(self.form_sz[0]-136)}px',
                                              "align_items":"flex-start",
                                              "justify_items":"flex-start"})
        self.hs['L[0][3][1][0][0]_io_tomo_data_box'] = grid_tomo_data_hs
        grid_tomo_data_hs[0, 37:70] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; align: center; background-color:rgb(135,206,250);">' + 'Tomo File Data Structure' + '</span>')
        self.hs['L[0][3][1][0][0][4]_io_tomo_data_config_html'] = grid_tomo_data_hs[0, 37:70]
        grid_tomo_data_hs[1, 0:25] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'data path', 'value':'/img_tomo', 'disabled':False, 'description_tooltip':'path to sample image data in h5 file', 'indent':False})
        self.hs['L[0][3][1][0][0][0]_io_tomo_img_text'] = grid_tomo_data_hs[1, 0:25]
        grid_tomo_data_hs[1, 25:50] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'flat path', 'value':'/img_bkg', 'disabled':False, 'description_tooltip':'path to flat (reference) image data in h5 file', 'indent':False})
        self.hs['L[0][3][1][0][0][1]_io_tomo_flat_text'] = grid_tomo_data_hs[1, 25:50]
        grid_tomo_data_hs[1, 50:75] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'dark path', 'value':'/img_dark', 'disabled':False, 'description_tooltip':'path to dark image data in h5 file', 'indent':False})
        self.hs['L[0][3][1][0][0][2]_io_tomo_dark_text'] = grid_tomo_data_hs[1, 50:75]
        grid_tomo_data_hs[1, 75:100] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'theta path', 'value':'/angle', 'disabled':False, 'description_tooltip':'path to theta in h5 file', 'indent':False})
        self.hs['L[0][3][1][0][0][3]_io_tomo_theta_text'] = grid_tomo_data_hs[1, 75:100]
        
        self.hs['L[0][3][1][0][0]_io_tomo_data_box'].children = get_handles(self.hs, 'L[0][3][1][0][0]_io_tomo_data_box', -1)
        
        grid_tomo_info_hs = GridspecLayout(3, 100,
                                    layout = {"border":"3px solid #FFCC00",
                                              "height":f'{0.18*(self.form_sz[0]-136)}px',
                                              "align_items":"flex-start",
                                              "justify_items":"flex-start"})
        self.hs['L[0][3][1][0][1]_io_tomo_info_box'] = grid_tomo_info_hs
        grid_tomo_info_hs[0, 38:70] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; align: center; background-color:rgb(135,206,250);">' + 'Config Tomo Data Info' + '</span>')
        self.hs['L[0][3][1][0][1][8]_io_tomo_info_config_html'] = grid_tomo_info_hs[0, 38:70]
        grid_tomo_info_hs[1, 0:25] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info0 path', 'value':'/img_tomo', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][0][1][0]_io_tomo_info0_text'] = grid_tomo_info_hs[1, 0:25]
        grid_tomo_info_hs[1, 25:50] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info1 path', 'value':'/angle', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][0][1][1]_io_tomo_info1_text'] = grid_tomo_info_hs[1, 25:50]
        grid_tomo_info_hs[1, 50:75] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info2 path', 'value':'/Magnification', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][0][1][2]_io_tomo_info2_text'] = grid_tomo_info_hs[1, 50:75]
        grid_tomo_info_hs[1, 75:] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info3 path', 'value':'/Pixel Size', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][0][1][3]_io_tomo_info3_text'] = grid_tomo_info_hs[1, 75:]
        
        grid_tomo_info_hs[2, 0:25] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info4 path', 'value':'/X_eng', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][0][1][4]_io_tomo_info4_text'] = grid_tomo_info_hs[2, 0:25]
        grid_tomo_info_hs[2, 25:50] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info5 path', 'value':'/note', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][0][1][5]_io_tomo_info5_text'] = grid_tomo_info_hs[2, 25:50]
        grid_tomo_info_hs[2, 50:75] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info6 path', 'value':'/scan_time', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][0][1][6]_io_tomo_info6_text'] = grid_tomo_info_hs[2, 50:75]
        grid_tomo_info_hs[2, 75:] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info7 path', 'value':'', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][0][1][7]_io_tomo_info7_text'] = grid_tomo_info_hs[2, 75:]
        
        self.hs['L[0][3][1][0][1]_io_tomo_info_box'].children = get_handles(self.hs, 'L[0][3][1][0][1]_io_tomo_info_box', -1)
        
        self.hs['L[0][3][1][0]_io_tomo_config_box'].children = get_handles(self.hs, 'L[0][3][1][0]_io_tomo_config_box', -1)
        ## ## ## ## tomo io config -- end
        
        ## ## ## ## xanes2D io config -- start 
        self.hs['L[0][3][1][1]_io_xanes2D_config_box'] = create_widget('VBox', {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.31*(self.form_sz[0]-136)}px'})
        
        grid_xanes2D_data_hs = GridspecLayout(2, 100,
                                    layout = {"border":"3px solid #FFCC00",
                                              "height":f'{0.12*(self.form_sz[0]-136)}px',
                                              "align_items":"flex-start",
                                              "justify_items":"flex-start"})
        self.hs['L[0][3][1][1][0]_io_xanes2D_data_box'] = grid_xanes2D_data_hs
        grid_xanes2D_data_hs[0, 35:70] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; align: center; background-color:rgb(135,206,250);">' + 'XANES2D File Data Structure' + '</span>')
        self.hs['L[0][3][1][1][0][4]_io_xanes2D_data_config_html'] = grid_xanes2D_data_hs[0, 35:70]
        grid_xanes2D_data_hs[1, 0:25] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'data path', 'value':'/img_xanes', 'disabled':False, 'description_tooltip':'path to xanes image data in h5 file', 'indent':False})
        self.hs['L[0][3][1][1][0][0]_io_xanes2D_img_text'] = grid_xanes2D_data_hs[1, 0:25]
        grid_xanes2D_data_hs[1, 25:50] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'flat path', 'value':'/img_bkg', 'disabled':False, 'description_tooltip':'path to flat (reference) image data in h5 file', 'indent':False})
        self.hs['L[0][3][1][1][0][1]_io_xanes2D_flat_text'] = grid_xanes2D_data_hs[1, 25:50]
        grid_xanes2D_data_hs[1, 50:75] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'dark path', 'value':'/img_dark', 'disabled':False, 'description_tooltip':'path to dark image data in h5 file', 'indent':False})
        self.hs['L[0][3][1][1][0][2]_io_xanes2D_dark_text'] = grid_xanes2D_data_hs[1, 50:75]
        grid_xanes2D_data_hs[1, 75:100] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'eng path', 'value':'/X_eng', 'disabled':False, 'description_tooltip':'path to x-ray energy in h5 file', 'indent':False})
        self.hs['L[0][3][1][1][0][3]_io_xanes2D_eng_text'] = grid_xanes2D_data_hs[1, 75:100]
        
        self.hs['L[0][3][1][1][0]_io_xanes2D_data_box'].children = get_handles(self.hs, 'L[0][3][1][1][0]_io_xanes2D_data_box', -1)
        
        grid_xanes2D_info_hs = GridspecLayout(3, 100,
                                    layout = {"border":"3px solid #FFCC00",
                                              "height":f'{0.18*(self.form_sz[0]-136)}px',
                                              "align_items":"flex-start",
                                              "justify_items":"flex-start"})
        self.hs['L[0][3][1][1][1]_io_xanes2D_info_box'] = grid_xanes2D_info_hs
        grid_xanes2D_info_hs[0, 37:70] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; align: center; background-color:rgb(135,206,250);">' + 'Config XANES2D Data Info' + '</span>')
        self.hs['L[0][3][1][1][1][8]_io_xanes2D_info_config_html'] = grid_xanes2D_info_hs[0, 37:70]
        grid_xanes2D_info_hs[1, 0:25] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info0 path', 'value':'/img_xanes', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][1][1][0]_io_xanes2D_info0_text'] = grid_xanes2D_info_hs[1, 0:25]
        grid_xanes2D_info_hs[1, 25:50] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info1 path', 'value':'/Magnification', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][1][1][1]_io_xanes2D_info1_text'] = grid_xanes2D_info_hs[1, 25:50]
        grid_xanes2D_info_hs[1, 50:75] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info2 path', 'value':'/Pixel Size', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][1][1][2]_io_xanes2D_info2_text'] = grid_xanes2D_info_hs[1, 50:75]
        grid_xanes2D_info_hs[1, 75:] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info3 path', 'value':'X_eng', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][1][1][3]_io_xanes2D_info3_text'] = grid_xanes2D_info_hs[1, 75:]
        
        grid_xanes2D_info_hs[2, 0:25] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info4 path', 'value':'/note', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][1][1][4]_io_xanes2D_info4_text'] = grid_xanes2D_info_hs[2, 0:25]
        grid_xanes2D_info_hs[2, 25:50] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info5 path', 'value':'/scan_time', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][1][1][5]_io_xanes2D_info5_text'] = grid_xanes2D_info_hs[2, 25:50]
        grid_xanes2D_info_hs[2, 50:75] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info6 path', 'value':'', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][1][1][6]_io_xanes2D_info6_text'] = grid_xanes2D_info_hs[2, 50:75]
        grid_xanes2D_info_hs[2, 75:] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info7 path', 'value':'', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][1][1][7]_io_xanes2D_info7_text'] = grid_xanes2D_info_hs[2, 75:]
        
        self.hs['L[0][3][1][1][1]_io_xanes2D_info_box'].children = get_handles(self.hs, 'L[0][3][1][1][1]_io_xanes2D_info_box', -1)   
        
        self.hs['L[0][3][1][1]_io_xanes2D_config_box'].children = get_handles(self.hs, 'L[0][3][1][1]_io_xanes2D_config_box', -1)
        ## ## ## ## xanes2D io config -- end
        
        ## ## ## ## xanes3D io config -- start 
        self.hs['L[0][3][1][2]_io_xanes3D_config_box'] = create_widget('VBox', {'border':'3px solid #8855AA', 'width':f'{self.form_sz[1]-92}px', 'height':f'{0.31*(self.form_sz[0]-136)}px'})
        
        grid_xanes3D_data_hs = GridspecLayout(2, 100,
                                    layout = {"border":"3px solid #FFCC00",
                                              "height":f'{0.12*(self.form_sz[0]-136)}px',
                                              "align_items":"flex-start",
                                              "justify_items":"flex-start"})
        self.hs['L[0][3][1][2][0]_io_xanes3D_data_box'] = grid_xanes3D_data_hs
        grid_xanes3D_data_hs[0, 35:70] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; align: center; background-color:rgb(135,206,250);">' + 'XANES3D File Data Structure' + '</span>')
        self.hs['L[0][3][1][2][0][4]_io_xanes3D_data_config_html'] = grid_xanes3D_data_hs[0, 35:70]
        grid_xanes3D_data_hs[1, 0:25] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'data path', 'value':'/img_tomo', 'disabled':False, 'description_tooltip':'path to sample image data in h5 file', 'indent':False})
        self.hs['L[0][3][1][2][0][0]_io_xanes3D_img_text'] = grid_xanes3D_data_hs[1, 0:25]
        grid_xanes3D_data_hs[1, 25:50] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'flat path', 'value':'/img_bkg', 'disabled':False, 'description_tooltip':'path to flat (reference) image data in h5 file', 'indent':False})
        self.hs['L[0][3][1][2][0][1]_io_xanes3D_flat_text'] = grid_xanes3D_data_hs[1, 25:50]
        grid_xanes3D_data_hs[1, 50:75] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'dark path', 'value':'/img_dark', 'disabled':False, 'description_tooltip':'path to dark image data in h5 file', 'indent':False})
        self.hs['L[0][3][1][2][0][2]_io_xanes3D_dark_text'] = grid_xanes3D_data_hs[1, 50:75]
        grid_xanes3D_data_hs[1, 75:100] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'eng path', 'value':'/X_eng', 'disabled':False, 'description_tooltip':'path to x-ray energy in h5 file', 'indent':False})
        self.hs['L[0][3][1][2][0][3]_io_xanes3D_eng_text'] = grid_xanes3D_data_hs[1, 75:100]
        
        self.hs['L[0][3][1][2][0]_io_xanes3D_data_box'].children = get_handles(self.hs, 'L[0][3][1][2][0]_io_xanes3D_data_box', -1)
        
        grid_xanes3D_info_hs = GridspecLayout(3, 100,
                                    layout = {"border":"3px solid #FFCC00",
                                              "height":f'{0.18*(self.form_sz[0]-136)}px',
                                              "align_items":"flex-start",
                                              "justify_items":"flex-start"})
        self.hs['L[0][3][1][2][1]_io_xanes3D_info_box'] = grid_xanes3D_info_hs
        grid_xanes3D_info_hs[0, 37:70] = widgets.HTML('<span style="color:red; font-size: 150%; font-weight: bold; align: center; background-color:rgb(135,206,250);">' + 'Config XANES3D Data Info' + '</span>')
        self.hs['L[0][3][1][2][1][8]_io_xanes3D_info_config_html'] = grid_xanes3D_info_hs[0, 37:70]
        grid_xanes3D_info_hs[1, 0:25] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info0 path', 'value':'/img_tomo', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][2][1][0]_io_xanes3D_info0_text'] = grid_xanes3D_info_hs[1, 0:25]
        grid_xanes3D_info_hs[1, 25:50] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info1 path', 'value':'/angle', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][2][1][1]_io_xanes3D_info1_text'] = grid_xanes3D_info_hs[1, 25:50]
        grid_xanes3D_info_hs[1, 50:75] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info2 path', 'value':'/Magnification', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][2][1][2]_io_xanes3D_info2_text'] = grid_xanes3D_info_hs[1, 50:75]
        grid_xanes3D_info_hs[1, 75:] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info3 path', 'value':'/Pixel Size', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][2][1][3]_io_xanes3D_info3_text'] = grid_xanes3D_info_hs[1, 75:]
        
        grid_xanes3D_info_hs[2, 0:25] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info4 path', 'value':'/X_eng', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][2][1][4]_io_xanes3D_info4_text'] = grid_xanes3D_info_hs[2, 0:25]
        grid_xanes3D_info_hs[2, 25:50] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info5 path', 'value':'/note', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][2][1][5]_io_xanes3D_info5_text'] = grid_xanes3D_info_hs[2, 25:50]
        grid_xanes3D_info_hs[2, 50:75] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info6 path', 'value':'/scan_time', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][2][1][6]_io_xanes3D_info6_text'] = grid_xanes3D_info_hs[2, 50:75]
        grid_xanes3D_info_hs[2, 75:] = create_widget('Text', {'width':f'{0.23*(self.form_sz[1]-86)}px'}, **{'description':'info7 path', 'value':'', 'disabled':False, 'description_tooltip':'path to a field in h5 file which information will be display in "Data Info"', 'indent':False})
        self.hs['L[0][3][1][2][1][7]_io_xanes3D_info7_text'] = grid_xanes3D_info_hs[2, 75:]
        
        self.hs['L[0][3][1][2][1]_io_xanes3D_info_box'].children = get_handles(self.hs, 'L[0][3][1][2][1]_io_xanes3D_info_box', -1)
        
        self.hs['L[0][3][1][2]_io_xanes3D_config_box'].children = get_handles(self.hs, 'L[0][3][1][2]_io_xanes3D_config_box', -1)
        ## ## ## ## xanes3D io config -- end
        
        ## ## ## ## confirm -- start
        grid_confirm_hs = GridspecLayout(1, 100,
                                    layout = {"border":"3px solid #FFCC00",
                                              "height":f'{0.06*(self.form_sz[0]-136)}px',
                                              "align_items":"flex-start",
                                              "justify_items":"flex-start"})
        self.hs['L[0][3][1][3]_io_confirm_box'] = grid_confirm_hs
        grid_confirm_hs[0, 44:60] = create_widget('Button', {'width':f'{0.15*(self.form_sz[1]-86)}px', "height":f'{0.045*(self.form_sz[0]-136)}px'}, **{'description':'Confirm', 'disabled':False})
        self.hs['L[0][3][1][3][0]_io_confirm_button'] = grid_confirm_hs[0, 44:60]
        self.hs['L[0][3][1][3][0]_io_confirm_button'].style.button_color = 'darkviolet'
        
        self.hs['L[0][3][1][3][0]_io_confirm_button'].on_click(self.L0_3_1_3_0_io_confirm_button_click)
        self.hs['L[0][3][1][3]_io_confirm_box'].children = get_handles(self.hs, 'L[0][3][1][3]_io_confirm_box', -1)
        ## ## ## ## confirm -- end
        
        self.hs['L[0][3][1]_io_config_form'].children = get_handles(self.hs, 'L[0][3][1]_io_config_form', -1)
        ## ## ## structured h5 -- end
        
    def L0_3_1_3_0_io_confirm_button_click(self, a):
        save_io_config(self)
        with open(self.global_h.io_data_struc_tomo_cfg_file, 'r') as f:
            self.global_h.io_tomo_cfg = json.load(f)
            self.global_h.tomo_recon_gui.tomo_raw_fn_temp = self.global_h.io_tomo_cfg['structured_h5_reader']['tomo_raw_fn_template']
        with open(self.global_h.io_data_struc_xanes2D_cfg_file, 'r') as f:
            self.global_h.io_xanes2D_cfg = json.load(f)
            self.global_h.xanes2D_gui.xanes2D_raw_fn_temp = self.global_h.io_xanes2D_cfg['structured_h5_reader']['xanes2D_raw_fn_template']
        with open(self.global_h.io_data_struc_xanes3D_cfg_file, 'r') as f:
            self.global_h.io_xanes3D_cfg = json.load(f)
            self.global_h.xanes3D_gui.xanes3D_raw_fn_temp = self.global_h.io_xanes3D_cfg['structured_h5_reader']['tomo_raw_fn_template']
            self.global_h.xanes3D_gui.xanes3D_recon_dir_temp = self.global_h.io_xanes3D_cfg['structured_h5_reader']['xanes3D_recon_dir_template']
            self.global_h.xanes3D_gui.xanes3D_recon_fn_temp = self.global_h.io_xanes3D_cfg['structured_h5_reader']['xanes3D_recon_fn_template']
            
    def L0_3_0_0_0_io_option_checkbox_change(self, a):
        if a['owner'].value:
            self.global_h.use_struc_h5_reader = True
        else:
            self.global_h.use_struc_h5_reader = False   
        self.boxes_logics()
    
    def L0_3_0_1_0_1_io_spec_tomo_reader_button_click(self, a):
        if len(a.files[0]) != 0:
            self.hs['L[0][3][0][1][0][0]_io_spec_tomo_reader_text'].value = a.files[0]        
            save_io_config(self)
            with open(self.global_h.io_data_struc_tomo_cfg_file, 'r') as f:
                self.global_h.io_tomo_cfg = json.load(f)
        
    
    def L0_3_0_1_0_3_io_spec_xanes2D_reader_button_click(self, a):
        if len(a.files[0]) != 0:
            self.hs['L[0][3][0][1][0][2]_io_spec_xanes2D_reader_text'].value = a.files[0]
            save_io_config(self)
            with open(self.global_h.io_data_struc_xanes2D_cfg_file, 'r') as f:
                self.global_h.io_xanes2D_cfg = json.load(f)
    
    def L0_3_0_1_0_5_io_spec_xanes3D_reader_button_click(self, a):
        if len(a.files[0]) != 0:
            self.hs['L[0][3][0][1][0][4]_io_spec_xanes3D_reader_text'].value = a.files[0]
            save_io_config(self)
            with open(self.global_h.io_data_struc_xanes3D_cfg_file, 'r') as f:
                self.global_h.io_xanes3D_cfg = json.load(f)
    
    def L0_3_2_0_0_6_fn_def_patt_confirm_button_click(self, a):
        save_io_config(self)
        with open(self.global_h.io_data_struc_tomo_cfg_file, 'r') as f:
            self.global_h.io_tomo_cfg = json.load(f)
            self.global_h.tomo_recon_gui.tomo_raw_fn_temp = self.global_h.io_tomo_cfg['structured_h5_reader']['tomo_raw_fn_template']
        with open(self.global_h.io_data_struc_xanes2D_cfg_file, 'r') as f:
            self.global_h.io_xanes2D_cfg = json.load(f)
            self.global_h.xanes2D_gui.xanes2D_raw_fn_temp = self.global_h.io_xanes2D_cfg['structured_h5_reader']['xanes2D_raw_fn_template']
        with open(self.global_h.io_data_struc_xanes3D_cfg_file, 'r') as f:
            self.global_h.io_xanes3D_cfg = json.load(f)
            self.global_h.xanes3D_gui.xanes3D_raw_fn_temp = self.global_h.io_xanes3D_cfg['structured_h5_reader']['tomo_raw_fn_template']
            self.global_h.xanes3D_gui.xanes3D_recon_dir_temp = self.global_h.io_xanes3D_cfg['structured_h5_reader']['xanes3D_recon_dir_template']
            self.global_h.xanes3D_gui.xanes3D_recon_fn_temp = self.global_h.io_xanes3D_cfg['structured_h5_reader']['xanes3D_recon_fn_template']
        