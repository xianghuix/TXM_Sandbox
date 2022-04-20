#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:38:41 2020

@author: xiao
"""

import h5py, tifffile
import os, sys, glob, gc, time, shutil
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom

import skimage.restoration as skr
import scipy.ndimage.filters as snf
from scipy.ndimage import filters
import dxchange, tomopy

self.tomo_recon_param_dict["file_params"] = {"raw_data_top_dir":self.tomo_raw_data_top_dir,
                                             "data_center_dir":self.tomo_data_center_path,
                                             "recon_top_dir":self.tomo_recon_top_dir,
                                             "debug_top_dir":self.tomo_debug_top_dir,
                                             "cen_list_file":self.tomo_cen_list_file,
                                             "alt_flat_file":self.tomo_alt_flat_file,
                                             "alt_dark_file":self.tomo_alt_dark_file,
                                             "wedge_ang_auto_det_ref_fn":self.tomo_wedge_ang_auto_det_ref_fn}

self.tomo_recon_param_dict["recon_config"] = {"recon_type":self.tomo_recon_type,
                                              "use_debug":self.tomo_use_debug,
                                              "use_alt_flat":self.tomo_use_alt_flat,
                                              "use_alt_dark":self.tomo_use_alt_dark,
                                              "use_fake_flat":self.tomo_use_fake_flat,
                                              "use_fake_dark":self.tomo_use_fake_dark,
                                              "use_rm_zinger":self.tomo_use_rm_zinger,
                                              "use_mask":self.tomo_use_mask,
                                              "use_wedge_ang_auto_det":self.tomo_use_wedge_ang_auto_det,
                                              "is_wedge":self.tomo_is_wedge}
self.tomo_recon_param_dict["flt_params"] = {"filters":self.tomo_right_filter_dict}
self.tomo_recon_param_dict["data_params"] = {"scan_id":self.tomo_scan_id,
                                             "rot_cen":self.tomo_rot_cen,
                                             "cen_win_s":self.tomo_cen_win_s,
                                             "cen_win_w":self.tomo_cen_win_w,
                                             "fake_flat_val":self.tomo_fake_flat_val,
                                             "fake_dark_val":self.tomo_fake_dark_val,
                                             "sli_s":self.tomo_sli_s,
                                             "sli_e":self.tomo_sli_e,
                                             "chunk_sz":self.tomo_chunk_sz,
                                             "margin":self.tomo_margin,
                                             "zinger_val":self.tomo_zinger_val,
                                             "mask_ratio":self.tomo_mask_ratio,
                                             "wedge_blankat":self.tomo_wedge_blankat,
                                             "wedge_missing_s":self.tomo_wedge_missing_s,
                                             "wedge_missing_e":self.tomo_wedge_missing_e,
                                             "wedge_ang_auto_det_thres":self.tomo_wedge_ang_auto_det_thres}
self.tomo_recon_param_dict["alg_params"] = {self.tomo_selected_alg:self.alg_param_dict}


options=["phase retrieval",
"down sample data",
"flatting bkg",
"remove cupping",
"stripe_removal: vo",
"stripe_removal: ti",
"stripe_removal: sf",
"stripe_removal: fw",
"denoise: wiener",
"denoise: unsupervised_wiener",
"denoise: denoise_nl_means",
"denoise: denoise_tv_bregman",
"denoise: denoise_tv_chambolle",
"denoise: denoise_bilateral",
"denoise: denoise_wavelet"],


filterList = ["phase retrieval",
              "down sample data",
              "flatting bkg",
              "remove cupping"
              "stripe_removal: vo",
              "stripe_removal: ti",
              "stripe_removal: sf",
              "stripe_removal: fw",
              "denoise: wiener",
              "denoise: unsupervised_wiener",
              "denoise: denoise_nl_means",
              "denoise: denoise_tv_bregman",
              "denoise: denoise_tv_chambolle",
              "denoise: denoise_bilateral",
              "denoise: denoise_wavelet"]

def alignt_proj(data, data_ref = None, **kwargs):
    pass

def data_down_sampling(data, levels):
    return zoom(data, levels)

def data_info(fn, showInfor=False):
    f = h5py.File(fn,"r")
    try:
        arr = f["img_tomo"]
        dim = arr.shape
        return dim
    except:
        return 0

def data_reader(fn, sli_start=0, sli_end=20,
                flat_name=None, dark_name=None,
                fake_flat=False, fake_dark=False,
                fake_flat_val=1e4, fake_dark_val=100,
                ds_use=False, ds_level=1.0):
    """
    input:
        fn: str
            full file path to the h5 raw data file
        sli_start: int, optional
            starting slice index
        sli_end: int, optional
            if sli_end=None, the sli_end=slice.max;
        flat_name: str, optional
            full path to the file for alternative flat images
        dark_name: str, optional
            full path to the file for alternative dark images
        fake_flat: boolean, optional
            if fake_flat=True, a uniform flat image is generated with its value
            equal to fake_flat_val
        fake_dark: boolean, optional
            if fake_dark=True, a uniform dark image is generated with its value
            equal to fake_dark_val
        ds_use: boolean, optional
            if ds_use=True, the projection images is read and downsampled by
            ds_level; if ds_use=False, data of original size is read; default
            is False
        ds_level: float, optional
            downsampling level; 0.5 corresponds to a downsampling factor 2

    return: tuple of ndarray, or None
        return data, white, dark, theta if succeeds; otherwise None
    """
    if flat_name == None:
        flat_name = fn
    if dark_name == None:
        dark_name = fn

    if ds_use:
        f = h5py.File(fn,"r")
        try:
            arr = f["img_tomo"]
        except:
           return None
        data = data_down_sampling(arr[:,sli_start:sli_end,:], [1, ds_level, ds_level])
        f.close()

        if fake_flat:
            white = data_down_sampling(fake_flat_val*np.ones([8, data.shape[1], data.shape[2]]), [1, ds_level, ds_level])
        else:
            f = h5py.File(flat_name,"r")
            try:
                arr = f["img_bkg"]
            except:
                return None
            white = data_down_sampling(arr[:,sli_start:sli_end,:], [1, ds_level, ds_level])
            f.close()


        if fake_dark:
            dark = data_down_sampling(fake_dark_val*np.ones([8, data.shape[1], data.shape[2]]), [1, ds_level, ds_level])
        else:
            f = h5py.File(dark_name,"r")
            try:
                arr = f["img_dark"]
            except:
                return None
            dark = data_down_sampling(arr[1:9,sli_start:sli_end,:], [1, ds_level, ds_level])
            f.close()

        f = h5py.File(fn,"r")
        try:
            arr = f["angle"]
        except:
            return None
        theta = arr[:]
        f.close()
    else:
        f = h5py.File(fn,"r")
        try:
            arr = f["img_tomo"]
        except:
            return None
        data = arr[:,sli_start:sli_end,:]
        f.close()

        if fake_flat:
            white = fake_flat_val*np.ones([8, data.shape[1], data.shape[2]])
        else:
            f = h5py.File(flat_name,"r")
            try:
                arr = f["img_bkg"]
            except:
                return None
            white = arr[:,sli_start:sli_end,:]
            f.close()

        if fake_dark:
            dark = fake_dark_val*np.ones([8, data.shape[1], data.shape[2]])
        else:
            f = h5py.File(dark_name,"r")
            try:
                arr = f["img_dark"]
            except:
                return None
            dark = arr[:,sli_start:sli_end,:]
            f.close()

        f = h5py.File(fn,"r")
        try:
            arr = f["angle"]
        except:
            return None
        theta = arr[:]
        f.close()

    gc.collect()
    return data, white, dark, theta

def filter_container(data,**kwargs):
    """
       kwargs: kwargs using format of filternameParams. For instance, to use filter
               remove_stripe_sf, you need to provide a kwarg
               remove_stripe_sfParams = {'use':'yes','size':31}
               By default, this routine assume five filters
               1. retrieve_phase
               2. remove_stripe_fw
               3. remove_stripe_ti
               4. remove_stripe_sf
               5. normalize_bg

               in the __main__ function below, this functions uses all five filters.
               You can set 'use':'no' in a filter kwargs to disable that filter. You
               can also include more filters in this function in the same format per
               your purposes.
    """
    use = sort_filters_order(**kwargs)
    if use == 0:
        pass
    elif use is None:
        return None
    else:
        for ii in sorted(use.keys()):
            data = run_filter(data,use[ii],**kwargs)
            data[data<0] = 1
        logDo = if_log(**kwargs)
        if logDo  == 'yes':
            data = tomopy.prep.normalize.minus_log(data)
        if kwargs["remove cupping"]['use']['use'] == 'yes':
            data -= kwargs["remove cupping"]['params']['cc']
        return data

def get_dim_angle_range_single_slice(sino, thres=500., block_view_at=90):
    """
    input:
        sino: ndarray
            the sinogram of a slice that has missing angles
        thres: float, optional
            the threshold below wich the signal is regarded as bas
        block_view_at: int, optional
            it indicates the missing angle range; it can only take two values: 0, or 90.
            if it is 0, the missing angles are in the beginning and end of 180 scans;
            if it is 90, the missing angles are in the middle of 180 scans.
    return:
        ndarray, the bad angle list
    """
    bad_ang = []
    for idx, count in enumerate(sino.mean(axis=1)):
        if count<thres:
            bad_ang.append(idx)
    return bad_ang

def get_dim_angle_range_slice_range(fn, sli_start, sli_end=None, dataset_path='/img_tomo', thres=500, block_view_at=90):
    """
    inputs:
        fn: str,
            full file path to the h5 raw data file
        sli_start: int
            starting slice index
        chunk_sz: int, optional
            the slice range that has missing angle range same as that at a representative slice;
            if chunk_sz=None, the same missing angle range at sli_start will be applied to the
            entire slice rangle; otherwise, the missing angle range will be computed chunk by chunk
        sli_end: int, optional
            if sli_end=None, the sli_end=slice.max;
        dataset_path: str, optional
            the path to the tomo dataset in h5 file structure; default is '/img_tomo'
        thres: float, optional
            count level below which is regarded as bad; default is 500
        blcok_view_at: int
            it indicates the missing angle range; it can only take two values: 0, or 90.
            if it is 0, the missing angles are in the beginning and end of 180 scans;
            if it is 90, the missing angles are in the middle of 180 scans.

    return:
        dictionary,
        the bad angle list for each slice in the slice range [sli_start, sli_end]
    """
    sli_start = int(sli_start)
    try:
        f = h5py.File(fn, 'r')
    except:
        print('the provided dataset path does not exist.')
        return None

    img_shape = f[dataset_path].shape
    if sli_end is None:
        sli_end = img_shape[1]
    elif sli_end>img_shape[1]:
        print('sli_end exceeds the maximum allowed range...')
        return None
    else:
        sli_end = int(sli_end)

    bad_angs = {}
    for ii in range(sli_start, sli_end):
        bad_angs[ii] = get_dim_angle_range_single_slice(f[dataset_path][:, ii, :], thres=thres, block_view_at=block_view_at)
    f.close()
    return bad_angs

def get_dim_angle_range_by_chunck(fn, sli_start, chunk_sz=None, sli_end=None, dataset_path='/img_tomo', thres=500, block_view_at=90):
    """
    inputs:
        fn: str,
            full file path to the h5 raw data file
        sli_start: int
            starting slice index
        chunk_sz: int, optional
            the slice range that has missing angle range same as that at a representative slice;
            if chunk_sz=None, the same missing angle range at sli_start will be applied to the
            entire slice rangle; otherwise, the missing angle range will be computed chunk by chunk
        sli_end: int, optional
            if sli_end=None, the sli_end=slice.max;
        dataset_path: str, optional
            the path to the tomo dataset in h5 file structure; default is '/img_tomo'
        thres: float, optional
            count level below which is regarded as bad; default is 500
        blcok_view_at: int
            it indicates the missing angle range; it can only take two values: 0, or 90.
            if it is 0, the missing angles are in the beginning and end of 180 scans;
            if it is 90, the missing angles are in the middle of 180 scans.

    return:
        interpolated missing angle range for each slice in the slice range [sli_start, sli_end]
    """
    pass
    # slic_start = int(sli_start)
    # chunk_sz = int(chunk_sz)
    # f = h5py.File(fn, 'r')
    # img_shape = f[dataset_path].shape
    # if sli_end is None:
    #     num_chunk = int(np.ceil((img_shape[0]-sli_start-1)/chunk_sz))
    # elif sli_end>img_shape[0]-1:
    #     print('sli_end exceeds the maximum allowed range...')
    #     return None
    # else:
    #     num_chunk = int(np.ceil((sli_end-sli_start)/chunk_sz))
    # node_id = [sli_start+i*chunk_sz for i in num_chunk]
    # bad_ang_nodes = {}
    # for ii in node_id:
    #     bad_ang_nodes[ii] = get_dim_angle_range(f[dataset_path][:, ii, :], thres=thres, block_view_at=block_view_at)
    # f.close()
    # return bad_ang_nodes

def get_files(raw_data_top_dir, scan_id):
    data_files = glob.glob(os.path.join(raw_data_top_dir, '*{}.h5'.format(scan_id)))

    if data_files is []:
        return None
    output_files = []
    for fn in data_files:
        output_files.append(os.path.join(raw_data_top_dir,
                                         'recon_'+os.path.basename(fn).split(".")[-2],
                                         'recon_'+os.path.basename(fn).split(".")[-2]))
    return data_files, output_files

def if_log(**kwargs):
    pr_use = kwargs["phase retrieval"]['use']['use']
    pr_flt = kwargs["phase retrieval"]['params']['filter']
    if pr_use == 'yes':
        if pr_flt == 'paganino':
            logDo = 'yes'
        elif pr_flt == 'paganin':
            logDo = 'yes'
        elif pr_flt == 'bronnikov':
            logDo = 'no'
        elif pr_flt == 'fba':
            logDo = 'no'
    else:
        logDo = 'yes'
    return logDo

def interpolate_dim_angle_range(bad_ang_nodes):
    pass
    # interp_bad_proj_id = []
    # for i1, i2 in zip(sorted(bad_ang_nodes.keys())[:-1], sorted(bad_ang_nodes.keys())[1:]):
    #     bad_ang_nodes[i1]+int((bad_ang_nodes[i2] - bad_ang_nodes[i1])*np.arange(i2-i1)/(i2-i1))

def read_center(fn):
    f = open(fn, 'r')
    idx_center = f.readlines()
    idx_list = []
    center_list = []
    for ii in idx_center:
        if ii.split():
            idx_list.append(int(ii.split()[0]))
            center_list.append(np.float(ii.split()[1]))
    return idx_list, center_list

def run_engine(**kwargs):
    recon_type = kwargs['recon_config']['type']
    if recon_type == 'trial':
        file_fn = kwargs['file_config']['data_files']
        file_data_center_path = kwargs['fileParams']['data_center_path']
        file_debug_dir = kwargs['fileParams']['debug_dir']
        file_flat_name = kwargs['file_config']['flat_name']
        file_dark_name = kwargs['file_config']['dark_name']
        file_fake_flat = kwargs['file_config']['fake_flat']
        file_fake_dark = kwargs['file_config']['fake_dark']
        file_fake_flat_val = kwargs['file_config']['fake_flat_val']
        file_fake_dark_val = kwargs['file_config']['fake_dark_val']
        file_smooth_flat = kwargs['file_config']['smooth_flat']['use']
        file_smooth_flat_sigma = kwargs['file_config']['smooth_flat']['sigma']
        file_ds_use =  kwargs['file_config']['down_sampling']['use']
        if file_ds_use.upper() == 'YES':
            file_ds_level = kwargs['file_config']['down_sampling']['level']
        else:
            file_ds_level = None

        alg_alg = kwargs['alg_config']['algorithm']
        alg_recon_filter = kwargs['alg_config']['alg_recon_filter']

        rec_sli_start = kwargs['recon_config']['sli_start']
        rec_sli_end = kwargs['recon_config']['sli_end']
        rec_cen_shift = kwargs['recon_config']['cen_shift']
        rec_cen_shift_wz = kwargs['recon_config']['cen_shift_wz']
        rec_mask = kwargs['recon_config']['mask']
        rec_mask_ratio = kwargs['recon_config']['mask_ratio']
        rec_wedge = kwargs['recon_config']['wedge']
        rec_wedge_thres = kwargs['recon_config']['wedge_thres']
        rec_wedge_ref_fn = kwargs['recon_config']['wedge_ref_fn']
        rec_wedge_block_at = kwargs['recon_config']['wedge_block_at']
        rec_logging = kwargs['recon_config']['logging']
        rec_debug = kwargs['recon_config']['debug']
        rec_debug_dir = kwargs['recon_config']['debug_dir']

        flt_zinger = kwargs['filter_config']['zinger_filter']['use']['use']
        flt_zinger_level = kwargs['filter_config']['zinger_filter']['params']['zinger_level']

        data,white,dark,theta = data_reader(file_fn,sli_start=rec_sli_start,sli_end=rec_sli_end,
                                            flat_name=file_flat_name, dark_name=file_dark_name,
                                            fake_flat=file_fake_flat, fake_dark=file_fake_dark,
                                            fake_flat_val=file_fake_flat_val, fake_dark_val=file_fake_dark_val,
                                            ds_use=file_ds_use, ds_level=file_ds_level)
        theta = theta * np.pi/180.0
        dim = data.shape
        numProj = dim[0]
        numSlices = dim[1]
        widthImg = dim[2]

        if flt_zinger:
            data[:] = tomopy.misc.corr.remove_outlier(data, flt_zinger_level, size=15, axis=0)[:]
            white[:] = tomopy.misc.corr.remove_outlier(white, flt_zinger_level, size=15, axis=0)[:]

        if file_smooth_flat:
            white[:] = filters.gaussian_filter(white, sigma=file_smooth_flat_sigma)[:]
        data[:] = tomopy.prep.normalize.normalize(data, white, dark)[:]

        if file_debug_dir is None:
            file_debug_dir = os.path.join(os.path.dirname(file_fn), 'debug')

        if rec_debug:
            save_debug(file_debug_dir, 'norm_data.tiff', data)

        data = filter_container(data,**kwargs)
        if rec_debug:
            save_debug(file_debug_dir, 'filtered_data.tiff', data)

        if file_data_center_path is None:
            overwrite_dir(os.path.join(os.path.dirname(file_fn), 'data_center'))
            # file_data_center_path = os.path.join(os.path.dirname(file_fn), 'data_center')
            # if os.path.exists(file_data_center_path):
            #     shutil.rmtree(file_data_center_path)
            #     os.makedirs(file_data_center_path)
            # else:
            #     os.makedirs(file_data_center_path)
        else:
            overwrite_dir(file_data_center_path)

        if rec_wedge:
            bad_angs = get_dim_angle_range_slice_range(rec_wedge_ref_fn,
                                                       rec_sli_start, sli_end=rec_sli_end,
                                                       thres=rec_wedge_thres,
                                                       block_view_at=rec_wedge_block_at)
            data = sort_wedge(data, bad_angs, rec_sli_start, rec_sli_end)

        tomopy.write_center(data[:,int(numSlices/2)-1:int(numSlices/2)+1,:], theta, dpath=file_data_center_path,
                     cen_range=(data.shape[2]/2+rec_cen_shift,data.shape[2]/2+rec_cen_shift+rec_cen_shift_wz,0.5),
                     mask=rec_mask, ratio=rec_mask_ratio, algorithm=alg_alg, filter_name=alg_recon_filter)

        if rec_logging is True:
            fout = os.path.join(os.path.dirname(file_fn), ''.join(os.path.basename(file_fn).split('.')[:-1]) +\
                                '_finding_cneter_log.txt')
            with open(fout, "w") as fo:
                for k, v in kwargs.items():
                    fo.write(str(k) + ': ' + str(v) + '\n\n')
    elif recon_type == 'volume':
        state = 1
        file_fn = kwargs['file_config']['data_files']
        file_output_file = kwargs['file_config']['out_files']
        file_flat_name = kwargs['file_config']['flat_name']
        file_dark_name = kwargs['file_config']['dark_name']
        file_fake_flat = kwargs['file_config']['fake_flat']
        file_fake_dark = kwargs['file_config']['fake_dark']
        file_fake_flat_val = kwargs['file_config']['fake_flat_val']
        file_fake_dark_val = kwargs['file_config']['fake_dark_val']
        file_smooth_flat = kwargs['file_config']['smooth_flat']['use']
        file_smooth_flat_sigma = kwargs['file_config']['smooth_flat']['sigma']
        file_ds_use =  kwargs['file_config']['down_sampling']['use']
        if file_ds_use.upper() == 'YES':
            file_ds_level = kwargs['file_config']['down_sampling']['level']
        else:
            file_ds_level = None

        alg = kwargs['alg_config']['algorithm']
        alg_recon_filter = kwargs['alg_config']['alg_recon_filter']

        rec_sli_start = kwargs['recon_config']['sli_start']
        rec_sli_end = kwargs['recon_config']['sli_end']
        rec_center = kwargs['recon_config']['center']
        rec_chunk_sz = kwargs['recon_config']['chunk_sz']
        rec_margin = kwargs['recon_config']['margin_slices']
        rec_mask = kwargs['recon_config']['mask']
        rec_mask_ratio = kwargs['recon_config']['mask_ratio']
        rec_wedge = kwargs['recon_config']['wedge']
        rec_wedge_thres = kwargs['recon_config']['wedge_thres']
        rec_wedge_ref_fn = kwargs['recon_config']['wedge_ref_fn']
        rec_wedge_block_at = kwargs['recon_config']['wedge_block_at']
        rec_logging = kwargs['recon_config']['logging']
        rec_debug = kwargs['recon_config']['debug']
        rec_debug_dir = kwargs['recon_config']['debug_dir']

        flt_zinger = kwargs['filter_config']['zinger']
        flt_zinger_level = kwargs['filter_config']['zinger_level']

        if rec_wedge:
            bad_angs = get_dim_angle_range_slice_range(rec_wedge_ref_fn,
                                                           rec_sli_start, sli_end=rec_sli_end,
                                                           thres=rec_wedge_thres,
                                                           block_view_at=rec_wedge_block_at)

        dim = data_info(file_fn)
        numSlices = dim[1]
        widthImg = dim[2]

        if rec_sli_start is None:
            rec_sli_start = 0
        if rec_sli_end is None:
            rec_sli_end = dim[1]

        if rec_chunk_sz >= (rec_sli_end-rec_sli_start):
            rec_chunk_sz = (rec_sli_end-rec_sli_start)
            num_chunk = 1
        else:
            num_chunk = np.int((rec_sli_end-rec_sli_start)/(rec_chunk_sz-rec_margin)) + 1

        for ii in range(num_chunk):
            try:
                if ii == 0:
                    sli_start = rec_sli_start
                    sli_end = rec_sli_start + rec_chunk_sz
                else:
                    sli_start = rec_sli_start + ii*(rec_chunk_sz-rec_margin)
                    sli_end = sli_start + rec_chunk_sz
                    if sli_end > rec_sli_end:
                        sli_end = rec_sli_end
                    if sli_end > dim[1]:
                        sli_end = dim[1]

                if (sli_end - sli_start) <= rec_margin:
                    break

                data, white, dark, theta = data_reader(file_fn, sli_start=sli_start, sli_end=sli_end,
                                                              flat_name=file_flat_name, dark_name=file_dark_name,
                                                              fake_flat=file_fake_flat, fake_dark=file_fake_dark,
                                                              fake_flat_val=file_fake_flat_val, fake_dark_val=file_fake_dark_val,
                                                              ds_use=file_ds_use, ds_level=file_ds_level)
                theta= theta*np.pi/180

                if flt_zinger == True:
                    data = tomopy.misc.corr.remove_outlier(data, flt_zinger_level, size=15, axis=0)
                    white = tomopy.misc.corr.remove_outlier(white, flt_zinger_level, size=15, axis=0)

                if file_smooth_flat:
                    white[:] = filters.gaussian_filter(white, sigma=file_smooth_flat_sigma)[:]
                data = tomopy.prep.normalize.normalize(data, white, dark)

                data = filter_container(data,**kwargs)

                if ii == 0 and ((rec_center is False) or (rec_center is None)):
                    rec_center = tomopy.find_center_vo(data)

                if rec_wedge:
                    data = sort_wedge(data, bad_angs, rec_sli_start, rec_sli_end)

                data_recon = tomopy.recon(data, theta, center=rec_center,
                                          algorithm=alg, filter_name=alg_recon_filter)

                if rec_mask:
                    data_recon = tomopy.circ_mask(data_recon, 0, ratio=rec_mask_ratio)

                dxchange.writer.write_tiff_stack(data_recon[int(rec_margin/2):(sli_end-sli_start-int(rec_margin/2)),:,:],
                                                            axis=0,
                                                            fname=file_output_file,
                                                            start=sli_start+int(rec_margin/2),
                                                            overwrite=True)

                del(data)
                del(white)
                del(dark)
                del(theta)
                del(data_recon)
                gc.collect()
                print ('chunk ',ii, ' reconstruction is saved')
                print (time.asctime())
            except:
                state = 0
        if state == 1:
            print ('Reconstruction finishes!')
            if rec_logging is True:
                fout = os.path.join(Path(file_output_file).parents[0], ''.join(os.path.basename(file_fn).split('.')[:-1]) +\
                                    '_recon_log.txt')
                fo = open(fout, "w")
                for k, v in kwargs.items():
                    fo.write(str(k) + ': ' + str(v) + '\n\n')
                fo.close()
        else:
            print ('Reconstruction is terminated due to data file error.')


def run_filter(data,fltname,**kwargs):
    if fltname == 'denoise_filter':
        params = kwargs[fltname]['params']['filter_params'][kwargs[fltname]['params']['filter_name']]
        if kwargs[fltname]['params']['use_est_sigma'] is True:
            if kwargs[fltname]['params']['filter_name'] in ['denoise_nl_means', 'denoise_wavelet']:
                sigma = skr.estimate_sigma(data[0])
            if kwargs[fltname]['params']['filter_name'] == 'denoise_nl_means':
                params['sigma'] = sigma
                params['h'] = 0.9 * sigma
            elif kwargs[fltname]['params']['filter_name'] == 'denoise_wavelet':
                params['sigma'] = sigma
        if kwargs[fltname]['params']['filter_name'] in ['wiener', 'unsupervised_wiener']:
            if kwargs[fltname]['params']['psf_reset_flag'] is False:
                psfw = params['psf']
                params['psf'] = np.ones([psfw, psfw])/(psfw**2)
                kwargs[fltname]['params']['psf_reset_flag'] = True
        if kwargs[fltname]['params']['filter_name'] == 'wiener':
            for ii in range(data.shape[0]):
                data[ii] = skr.wiener(data[ii], **params)
        elif kwargs[fltname]['params']['filter_name'] == 'unsupervised_wiener':
            for ii in range(data.shape[0]):
                data[ii], _ = skr.unsupervised_wiener(data[ii], **params)
        elif kwargs[fltname]['params']['filter_name'] == 'denoise_nl_means':
            for ii in range(data.shape[0]):
                data[ii] = skr.denoise_nl_means(data[ii], **params)
        elif kwargs[fltname]['params']['filter_name'] == 'denoise_tv_bregman':
            for ii in range(data.shape[0]):
                data[ii] = skr.denoise_tv_bregman(data[ii], **params)
        elif kwargs[fltname]['params']['filter_name'] == 'denoise_tv_chambolle':
            for ii in range(data.shape[0]):
                data[ii] = skr.denoise_tv_chambolle(data[ii], **params)
        elif kwargs[fltname]['params']['filter_name'] == 'denoise_bilateral':
            for ii in range(data.shape[0]):
                data[ii] = skr.denoise_bilateral(data[ii], **params)
        elif kwargs[fltname]['params']['filter_name'] == 'denoise_wavelet':
            for ii in range(data.shape[0]):
                data[ii] = skr.denoise_wavelet(data[ii], **params)
    elif fltname == "stripe_removal: fw":
        params = kwargs[fltname]['params']
        data = tomopy.prep.stripe.remove_stripe_fw(data,**params)
    elif fltname == "stripe_removal: ti":
        params = kwargs[fltname]['params']
        data = tomopy.prep.stripe.remove_stripe_ti(data,**params)
    elif fltname == "stripe_removal: sf":
        params = kwargs[fltname]['params']
        data = tomopy.prep.stripe.remove_stripe_sf(data,**params)
    elif fltname == "stripe_removal: vo":
        params = kwargs[fltname]['params']
        data = tomopy.prep.stripe.remove_all_stripe(data,**params)
    elif fltname == "phase retrieval":
        params = kwargs[fltname]['params']
        if params['filter'] == 'paganino':
            data = tomopy.prep.phase.retrieve_phase(data,**params)
        elif params['filter'] == 'paganin':
            data = tomopy.prep.phase.retrieve_phase(data,**params)
        elif params['filter'] == 'bronnikov':
            data = 1 - data
            data = tomopy.prep.phase.retrieve_phase(data,**params)
        elif params['filter'] == 'fba':
            data = (1 - data)/2
            data = tomopy.prep.phase.retrieve_phase(data,**params)
        else:
            return None
    elif fltname == "flatting bkg":
        params = kwargs[fltname]['params']
        data = tomopy.prep.normalize.normalize_bg(data,**params)
    elif fltname == "remove cupping":
        pass
    else:
        return None
    return data

def save_debug(debug_dir, debug_fn, data):
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    tifffile.imsave(os.path.join(debug_dir, debug_fn), data)

def sort_filters_order(**kwarg):
    use = {}
    cnt = 0
    for flt in filterList:
        if kwarg[flt]['use']['use'] != 'no':
            use[kwarg[flt]['use']['order']] = flt
            cnt += 1
    if len(use) == cnt:
        if cnt == 0:
            return 0
        else:
            return use
    else:
        return None

def sort_wedge(data, bad_angs, sli_start, sli_end):
    for ii in range(sli_start, sli_end):
        data[bad_angs[ii], ii-sli_start, :] = 0
    return data

def overwrite_dir(path):
    if os.path.isdir(path):
        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            os.makedirs(path)
    else:
        if os.path.exists(os.path.dirname(path)):
            shutil.rmtree(os.path.dirname(path))
            os.makedirs(os.path.dirname(path))
        else:
            os.makedirs(os.path.dirname(path))


