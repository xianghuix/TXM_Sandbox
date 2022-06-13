#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 23:27:42 2020

@author: xiao
"""

IO_TOMO_CFG_DEFAULT = {'structured_h5_reader':{
                       'io_data_structure':{'data_path':'/img_tomo',
                                            'flat_path':'/img_bkg',
                                            'dark_path':'/img_dark',
                                            'theta_path':'/angle'},
                       'io_data_info':{'item00_path':'/img_tomo',
                                       'item01_path':'/angle',
                                       'item02_path':'/Magnification',
                                       'item03_path':'/Pixel Size',
                                       'item04_path':'/X_eng',
                                       'item05_path':'/note',
                                       'item06_path':'/scan_time',
                                       'item07_path':''}
                                               },
                       'customized_reader':{
                          'user_tomo_reader':''   
                                              },
                       'tomo_raw_fn_template':'fly_scan_id_{}.h5'
                      }


IO_XANES2D_CFG_DEFAULT = {'structured_h5_reader':{
                          'io_data_structure':{'data_path':'/img_xanes',
                                               'flat_path':'/img_bkg',
                                               'dark_path':'/img_dark',
                                               'eng_path':'/X_eng'},
                          'io_data_info':{'item00_path':'/img_xanes',
                                          'item01_path':'/Magnification',
                                          'item02_path':'/Pixel Size',
                                          'item03_path':'/X_eng',
                                          'item04_path':'/note',
                                          'item05_path':'/scan_time',
                                          'item06_path':'',
                                          'item07_path':''}
                                                  },
                          'customized_reader':{
                          'user_xanes2D_reader':''    
                                              },
                          'xanes2D_raw_fn_template':'xanes_scan2_id_{}.h5'
                         }


IO_XANES3D_CFG_DEFAULT = {'structured_h5_reader':{
                          'io_data_structure':{'data_path':'/img_tomo',
                                               'flat_path':'/img_bkg',
                                               'dark_path':'/img_dark',
                                               'eng_path':'/X_eng'},
                          'io_data_info':{'item00_path':'/img_tomo',
                                          'item01_path':'/angle',
                                          'item02_path':'/Magnification',
                                          'item03_path':'/Pixel Size',
                                          'item04_path':'/X_eng',
                                          'item05_path':'/note',
                                          'item06_path':'/scan_time',
                                          'item07_path':''}
                                                  },
                          'customized_reader':{
                          'user_xanes3D_reader':''    
                                              },
                          'tomo_raw_fn_template':'fly_scan_id_{}.h5',
                          'xanes3D_recon_dir_template':'recon_fly_scan_id_{}',
                          'xanes3D_recon_fn_template':'recon_fly_scan_id_{0}_{1}.tiff'
                         }