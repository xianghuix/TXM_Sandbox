# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:26:06 2015

@author: xhxiao
"""

# from tomopy.io.reader import *
# import numpy as np
# import os.path
# from numpy.testing import assert_allclose
import tomopy
from tomopy.recon.rotation import write_center
from tomopy.recon.algorithm import recon
from scipy.ndimage import zoom
import os, h5py, glob
import numpy as np
import time
import skimage.restoration as skr
import scipy.ndimage.filters as snf
from scipy.ndimage import filters
import dxchange
#from pystackreg import StackReg
from pathlib import Path
import tomo_recon_tools as trt
import gc, sys
import tifffile





if __name__ == '__main__':
############################### user input section -- Start ###############################
##### This is the only section which you need to edit
    config = {'file_config':{},
              'alg_config':{},
              'recon_config':{},
              'filter_config':{}}

    config['file_config']['data_files'] = ''
    config['file_config']['data_center_path'] = ''
    config['file_config']['out_files'] = ''
    config['file_config']['debug_dir'] = ''
    config['file_config']['flat_name'] = ''
    config['file_config']['dark_name'] = ''
    config['file_config']['fake_flat'] = ''
    config['file_config']['fake_dark'] = ''
    config['file_config']['fake_flat_val'] = ''
    config['file_config']['fake_dark_val'] = ''
    config['file_config']['smooth_flat']['use'] = ''
    config['file_config']['smooth_flat']['sigma'] = ''
    config['file_config']['down_sampling']['use'] = ''
    config['file_config']['down_sampling']['level'] = ''

    config['alg_config']['algorithm'] = ''
    config['alg_config']['alg_recon_filter'] = ''

    config['recon_config']['sli_start'] = ''
    config['recon_config']['sli_end'] = ''
    config['recon_config']['cen_shift'] = ''
    config['recon_config']['cen_shift_wz'] = ''
    config['recon_config']['center'] = ''
    config['recon_config']['chunk_sz'] = ''
    config['recon_config']['margin_slices'] = ''
    config['recon_config']['mask'] = ''
    config['recon_config']['mask_ratio'] = ''
    config['recon_config']['wedge'] = ''
    config['recon_config']['wedge_thres'] = ''
    config['recon_config']['wedge_ref_fn'] = ''
    config['recon_config']['wedge_block_at'] = ''
    config['recon_config']['logging'] = ''
    config['recon_config']['debug'] = ''
    config['recon_config']['debug_dir'] = ''

    config['filter_config']['zinger_filter']['use']['use'] = ''
    config['filter_config']['zinger_filter']['params']['zinger_level'] = ''



    raw_data_top_dir = "/NSLS2/xf18id1/users/2020Q1/ENYUAN_Proposal_305409/"
    #raw_data_top_dir = "/NSLS2/xf18id1/users/2019Q3/XIANGHUI_XIAO_Proposal_305570"
    # keep [] even you only do reconstruction of one data set; put all data set Exp indices that you want to reconstruct in [];
    # if you want to reconstruction all data sets in a range, e.g. with indices from 1 to 10, you can use scan_idx = np.arrange(1,10,1);
    # if you want to reconstruction one every two, you can use scan_idx = np.arange(1,10,2)
#    scan_idx = [24567, 24568, 24569, 24570, 24571, 24572, 24573, 24574, 24575, 24576, 24577, 24578]
#    scan_idx = [24626, 24627, 24628, 24629, 24630, 24631, 24632, 24633, 24634, 24635, 24636, 24637, 24638, 24639, 24640, 24641, 24642]
    scan_idx = [29671]
    #center_list = [1326.0]
#    center_list = [1370.00, 1377.00, 1375.00, 1366.00, 1361.50, 1384.50, 1390.50, 1377.50, 1379.50, 1373.50, 1373.50, 1379.50, 1366.00, 1370.50, 1362.00, 1378.00, 1379.00]
    center_list = [642.5] #663 is 30059
    #center_list = [1376.5, 1336, 1344]                  # If you find center values for the list of data sets defined in scan_idx above, you put
                                            # these center values here in []. The number of values here in [] must be
                                            # equal to number of items in scan_idx = [], and orders in these two [] must match.

    flat_name = None
    dark_name = None
    fake_flat = False
    fake_dark = False
    fake_flat_val = 15e4
    fake_dark_val = 100.

    # if recon_top_dir = None
    #   save the recon in a folder in the sample directory level as the raw data file
    # else:
    #   save the recon into a folder in the specified directory
    recon_top_dir = None #'/media/KarenUSB/BNL_FXI_Dec05_2019'


    # manualCenter = True will turn on trial finding center routine. With this option, you can not only find center position but also
    # test filter combination and parameters quickly. When manualCenter = True, you may need to adjust parameter in the below if statements.
    # manualCenter = False will reconstruct whole volume in a data set with the center value calculated with automatic center finding
    # routine. It may output wrong center whose accuracy may vary from data set to data set.
    if len(sys.argv) == 1:
        manualCenter = True
    elif len(sys.argv) == 2:
        manualCenter = False
    elif len(sys.argv) == 3:
        manualCenter = False
        scan_idx, center_list = read_center(sys.argv[2])
        print(manualCenter)
    #manualCenter = False
    logging = True
    algorithm = "gridrec"
    filter_name = "shepp"

    # configure for manual finding center
    if manualCenter == True:
        center_shift = -80  #-10
        center_shift_w = 80   # 50
        sliceStart = 500 #1580

        #sliceStart = 55       #1580
        sliceEnd = sliceStart + 20    # 20
        #data_center_path = os.path.join(os.path.abspath(os.path.join(raw_data_top_dir,'..')),'data_center')
        data_center_path = os.path.join(raw_data_top_dir,'data_center')

        #print(f'\n\n\n{data_center_path}')
#        data_center_path = "/media/XIAOUSB1/BNL_FXI_June2019/data_center"

    # configure for volume reconstruction
    offset = 100         # set to None if you want to reconstruct the whole volume
    numRecSlices = 800    # set to None if you want to reconstruct the whole volume
    chunk_size = 200        # this number is determined by available RAM in your computer; 300 is good for a computer with at least 128GB RAM
    margin_slices = 30      # leave this fixed
    zinger = False
    zinger_level = 500      # You may need to change this is you see some artifacts in reconstructed slice images
    mask = True             # You set 'mask' to be True if you like mask out four corners in the reconsructed slice images
    mask_ratio = 1          # Ratio of the mask's diameter in pixels to slice image width

#    center_list = [30478:1313.5, 30480:1294.5, 30949:1247.5]
    # define wedge data processing parameters
    BlankAt = 90
    first_missing_end = None
    second_missing_start = None
    missing_start = None
    missing_end = None
    bad_angs = None

    if manualCenter:
        sli_start = sliceStart
        sli_end = sliceEnd
    else:
        if offset is None:
            sli_start = 0
        else:
            sli_start = offset
        if numRecSlices is not None:
            sli_end = offset + numRecSlices
        else:
            sli_end = numRecSlices

    thres = 500
    wedge = True
    fn = "/NSLS2/xf18id1/users/2020Q1/ENYUAN_Proposal_305409/fly_scan_id_29721.h5"
    if wedge:
        bad_angs = trt.get_dim_angle_range_slice_range(fn, sli_start, sli_end=sli_end, thres=thres, block_view_at=BlankAt)

    if wedge == True:
        if BlankAt == 0:
            first_missing_end = 100  # 180
            second_missing_start = 700  #1360
            missing_start = None
            missing_end = None
        elif BlankAt == 90:
            missing_start = 340
            missing_end = 390
            first_missing_end = None
            second_missing_start = None

    wedgeParams = {'first_missing_end':first_missing_end,
                    'second_missing_start':second_missing_start,
                    'missing_start':missing_start,
                    'missing_end':missing_end,
                    'BlankAt':BlankAt,
                    'bad_angs':bad_angs}

    # define parameters for various filters
    denoise_filter = {'use_est_sigma': True,
                      'filter_name': 'wiener',
                      'psf_reset_flag': False,
                      'filter_params': {'wiener': {'psf': 2,
                                  'balance': 0.3,
                                  'reg': None,
                                  'is_real': True,
                                  'clip': True},

                                  'unsupervised_wiener': {'psf': 2,
                                  'balance': 0.3,
                                  'reg': None,
                                  'is_real': True,#True
                                  'clip': True},#True

                                  'denoise_nl_means': {'patch_size': 5,
                                  'patch_distance': 7,
                                  'h': 0.1,
                                  'multichannel': False,
                                  'fast_mode': True,
                                  'sigma': 0.05},

                                  'denoise_tv_bregman': {'weight': 1.0,
                                  'max_iter': 100,
                                  'eps': 0.001,
                                  'isotropic': True},

                                  'denoise_tv_chambolle': {'weight': 0.1,
                                  'eps': 0.0002,
                                  'max_iter': 200,
                                  'multichannel': False},

                                  'denoise_bilateral': {'win_size': None,
                                  'sigma_color': None,
                                  'sigma_spatial': 1,
                                  'bins': 10000,
                                  'mode': 'constant',
                                  'cval': 0,
                                  'multichannel': False},

                                  'denoise_wavelet': {'sigma': 1,
                                  'wavelet': 'db1',
                                  'mode': 'soft',
                                  'wavelet_levels': 3,
                                  'multichannel': False,
                                  'convert2ycbcr': False,
                                  'method': 'BayesShrink'}
                                  }
                        }

    downsample_Params = {'level': (0.5, 0.5, 0.5)}

    retrieve_phaseParams = {'filter': 'paganin',
                            'pixel_size': 0.65e-4,
                            'dist': 15,
                            'energy': 35.0,
                            'alpha': 1e-2,
                            'pad':True}

    remove_stripe_tiParams = {'nblock': 0,
                              'alpha': 5}

    remove_stripe_fwParams = {'level': 9,
                              'wname': 'db5',
                              'sigma': 2,
                              'pad':True}


    remove_stripe_sfParams = {'size': 31}

    remove_stripe_voParams = {'snr':3,
                              'la_size': 81,
                              'sm_size': 21}

    normalize_bgParams = {'air': 30}

    remove_cuppingParams = {'cc': 0.0}


### define filter combination
    use_stripe_removal_fw = {'use':'no',
                             'order':1}

    use_stripe_removal_ti = {'use':'no',
                             'order':None}

    use_stripe_removal_sf = {'use':'no',
                             'order':1}

    use_stripe_removal_vo = {'use':'yes',
                             'order':1} #yes

    use_normalize_bg =      {'use':'no',
                             'order':3}

    use_retrieve_phase =    {'use':'no',
                             'order':2}

    use_denoise_filter =    {'use':'no',
                             'order':2}

    use_remove_cupping =    {'use':'no',
                             'order':3}

    use_downsample =        {'use':'no'}





############################### user input section -- End ###############################

    for jj in range(len(scan_idx)):
        if manualCenter == True:
            if len(scan_idx) == 1:
                data_files, output_files = getFiles(raw_data_top_dir, scan_idx[0])
                if recon_top_dir is not None:
                    output_files = os.path.join(recon_top_dir, output_files.split('/')[-2], output_files.split('/')[-1])
                if data_files == 0:
                    print ('!!!!! Error !!!!!')
                    print ('There is no given file in the path. Reconstructon is aborted.')
                    exit()

                dim = dataInfo(data_files, showInfor=True)
                if dim == 0:
                    print ('!!!!! Error !!!!!')
                    print ('Cannot read the given data file. Reconstruction is aborted.')
                    exit()

                print('Finding center manually')

                if wedge is True:
                    if BlankAt == 0 and (first_missing_end == None or second_missing_start == None):
                        print('You need to define first_missing_end and second_missing_start.')
                        exit()
                    elif BlankAt == 90 and (missing_start == None or missing_end == None):
                        print('You need to define missing_start and missing_end.')
                        exit()

                loopEngineParams = {'ExplicitParams':
                                        {'algorithm': algorithm,
                                         'filter_name': filter_name,
                                         'sliceStart': sliceStart,
                                         'sliceEnd': sliceEnd,
                                         'zinger': zinger,
                                         'zinger_level': zinger_level,
                                         'mask': mask,
                                         'mask_ratio': mask_ratio,
                                         'logging': logging},

                                    'fileParams':
                                        {'data_files': data_files,
                                         'data_center_path': data_center_path,
                                         'flat_name': flat_name,
                                         'dark_name': dark_name,
                                         'fake_flat': fake_flat,
                                         'fake_dark': fake_dark,
                                         'fake_flat_val': fake_flat_val,
                                         'fake_dark_val': fake_dark_val},
                                    'wedgeData':
                                        {'wedge': wedge,
                                         'wedgeParams': wedgeParams},

                                    'denoise_filter':
                                        {'use': use_denoise_filter,
                                         'params': denoise_filter},

                                    'remove_stripe_fwFilter':
                                        {'use': use_stripe_removal_fw,
                                         'params': remove_stripe_fwParams},

                                    'retrieve_phaseFilter':
                                        {'use': use_retrieve_phase,
                                         'params': retrieve_phaseParams},

                                    'remove_stripe_tiFilter':
                                        {'use': use_stripe_removal_ti,
                                         'params': remove_stripe_tiParams},

                                    'remove_stripe_sfFilter':
                                        {'use': use_stripe_removal_sf,
                                         'params': remove_stripe_sfParams},

                                    'remove_stripe_voFilter':
                                        {'use': use_stripe_removal_vo,
                                         'params': remove_stripe_voParams},

                                    'normalize_bgFilter':
                                        {'use': use_normalize_bg,
                                         'params': normalize_bgParams},

                                    'remove_cuppingFilter':
                                        {'use': use_remove_cupping,
                                         'params': remove_cuppingParams},

                                    'downsample_Filter':
                                        {'use': use_downsample,
                                         'params': downsample_Params}
                                    }

                manualFindCenter(center_shift, center_shift_w, **loopEngineParams)
            else:
                print ('!!!!! Error !!!!!')
                print ('Finding center manually requires only one data set. Please either change manualCenter to False or provide only one data set.')

        else:
            print('You are doing volume reconstructions...')
            data_files, output_files = getFiles(raw_data_top_dir, scan_idx[jj])
            if recon_top_dir is not None:
                    output_files = os.path.join(recon_top_dir, output_files.split('/')[-2], output_files.split('/')[-1])
            if data_files == 0:
                print ('!!!!! Error !!!!!')
                print ('There is no given file in the path. Reconstructon is aborted.')
                exit()
            print(data_files)
            print(data_files[0])
            dim = dataInfo(data_files, showInfor=True) ### earlier it was data_files[0] within brackets, didn't work
            if dim == 0:
                print ('!!!!! Error !!!!!')
                print ('Cannot read the given data file. Reconstruction is aborted.')
                exit()

            loopEngineParams = {'ExplicitParams':
                                                {'algorithm': algorithm,
                                                 'filter_name': filter_name,
                                                 'center':center_list[jj],
                                                 'zinger':zinger,
                                                 'zinger_level':zinger_level,
                                                 'offset':offset,
                                                 'chunk_size':chunk_size,
                                                 'numRecSlices':numRecSlices,
                                                 'margin_slices':margin_slices,
                                                 'mask':mask,
                                                 'mask_ratio':mask_ratio,
                                                 'logging': logging},

                                    'fileParams':
                                                {'data_files': data_files,
                                                 'output_files': output_files,
                                                 'flat_name': flat_name,
                                                 'dark_name': dark_name,
                                                 'fake_flat': fake_flat,
                                                 'fake_dark': fake_dark,
                                                 'fake_flat_val': fake_flat_val,
                                                 'fake_dark_val': fake_dark_val},

                                     'wedgeData':
                                                {'wedge':wedge,
                                                 'wedgeParams':wedgeParams},

                          'denoise_filter':
                                                {'use': use_denoise_filter,
                                                 'params': denoise_filter},

                        'remove_stripe_fwFilter':
                                                {'use':use_stripe_removal_fw,
                                                 'params':remove_stripe_fwParams},

                          'retrieve_phaseFilter':
                                                {'use':use_retrieve_phase,
                                                 'params':retrieve_phaseParams},

                        'remove_stripe_tiFilter':
                                                {'use':use_stripe_removal_ti,
                                                 'params':remove_stripe_tiParams},

                        'remove_stripe_sfFilter':
                                                {'use':use_stripe_removal_sf,
                                                 'params':remove_stripe_sfParams},
                        'remove_stripe_voFilter':
                                                {'use': use_stripe_removal_vo,
                                                 'params': remove_stripe_voParams},

                            'normalize_bgFilter':
                                                {'use':use_normalize_bg,
                                                 'params':normalize_bgParams},

                          'remove_cuppingFilter':
                                                {'use': use_remove_cupping,
                                                 'params': remove_cuppingParams},

                             'downsample_Filter':
                                                {'use': use_downsample,
                                                 'params': downsample_Params}
                              }

            reconEngine(**loopEngineParams)













