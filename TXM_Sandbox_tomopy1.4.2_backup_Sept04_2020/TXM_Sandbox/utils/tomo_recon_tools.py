#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:38:41 2020

@author: xiao
"""


import os, glob, gc, time, shutil, numpy as np

from pathlib import Path
import h5py, tifffile, json
from scipy.ndimage import zoom
import skimage.restoration as skr
import dxchange, tomopy
import tomopy.util.mproc as mproc

from ..utils.io import (data_reader, tomo_h5_reader,
                        data_info, tomo_h5_info)

IF_LOG = True

TOMO_RECON_PARAM_DICT = {}
TOMO_RECON_PARAM_DICT["file_params"] = {"raw_data_top_dir":None,
                                        "data_center_dir":None,
                                        "recon_top_dir":None,
                                        "debug_top_dir":None,
                                        "cen_list_file":None,
                                        "alt_flat_file":None,
                                        "alt_dark_file":None,
                                        "wedge_ang_auto_det_ref_fn":None}
TOMO_RECON_PARAM_DICT["recon_config"] = {"recon_type":'Trial Center',
                                        "use_debug":False,
                                        "use_alt_flat":False,
                                        "use_alt_dark":False,
                                        "use_fake_flat":False,
                                        "use_fake_dark":False,
                                        "use_rm_zinger":False,
                                        "use_mask":True,
                                        "use_wedge_ang_auto_det":False,
                                        "is_wedge":False}
TOMO_RECON_PARAM_DICT["flt_params"] = {"filters":{}}
TOMO_RECON_PARAM_DICT["data_params"] = {"scan_id":0,
                                        "downsample":1,
                                        "rot_cen":1280,
                                        "cen_win_s":1240,
                                        "cen_win_w":8,
                                        "fake_flat_val":1e4,
                                        "fake_dark_val":1e2,
                                        "sli_s":1280,
                                        "sli_e":1300,
                                        "chunk_sz":200,
                                        "margin":15,
                                        "zinger_val":500,
                                        "mask_ratio":1,
                                        # "wedge_blankat":90,
                                        "wedge_missing_s":500,
                                        "wedge_missing_e":600,
                                        "wedge_ang_auto_det_thres":500}
TOMO_RECON_PARAM_DICT["alg_params"] = {}

FILTERLIST = ["phase retrieval",
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
              "denoise: denoise_wavelet"]


def alignt_proj(data, data_ref = None, **kwargs):
    pass

def data_down_sampling(data, levels):
    return zoom(data, levels)

def get_dim_ang_range_single_sli(sino, thres=500.):
    """
    input:
        sino: ndarray
            the sinogram of a slice that has missing angles
        thres: float, optional
            the threshold below wich the signal is regarded as bas
    return:
        ndarray, the bad angle list
    """
    bad_ang = []
    for idx, count in enumerate(sino.mean(axis=1)):
        if count<thres:
            bad_ang.append(idx)
    return bad_ang

def get_dim_ang_range_range_sli(dim_info, reader, 
                                fn, sli_start, cfg, sli_end=None, 
                                col_start=None, col_end=None, thres=500):
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
    dataset_path = cfg['structured_h5_reader']['io_data_structure']['data_path'] 
    sli_start = int(sli_start)
    
    # img_shape = f[dataset_path].shape
    img_shape = dim_info(fn, dtype='data', cfg=cfg)
    if sli_end is None:
        sli_end = img_shape[1]
    elif sli_end>img_shape[1]:
        print('sli_end exceeds the maximum allowed range...')
        return None
    else:
        sli_end = int(sli_end)
        
    if col_start is None:
        col_start = 0
    else:
        col_start = int(col_start)
    
    if col_end is None:
        col_end = img_shape[2]
    else:
        col_end = int(col_end)
        
    bad_angs = {}
    
    try:
        f = h5py.File(fn, 'r')
    except:
        print('the provided dataset path does not exist.')
        return None
    
    for ii in range(sli_start, sli_end):
        bad_angs[ii] = get_dim_ang_range_single_sli(f[dataset_path][:, ii, col_start:col_end], thres=thres)
    f.close()
    return bad_angs

def get_file(raw_data_top_dir, scan_id, cfg):
    # data_file = glob.glob(os.path.join(raw_data_top_dir, '*{}.h5'.format(scan_id)))
    print(cfg)
    print(cfg['tomo_raw_fn_template'].format(scan_id))
    data_file = glob.glob(os.path.join(raw_data_top_dir, cfg['tomo_raw_fn_template'].format(scan_id)))

    if data_file is []:
        return None
    output_file = os.path.join(raw_data_top_dir,
                               'recon_'+os.path.basename(data_file[0]).split(".")[-2],
                               'recon_'+os.path.basename(data_file[0]).split(".")[-2])
    return data_file[0], output_file

def if_log(flt_dict):
    use_log = True
    for key in flt_dict.keys():
        if 'phase retrieval' == flt_dict[key]['filter_name']:
            if 'bronnikov' == flt_dict[key]['params']['filter']:
                use_log = False
                return use_log
    return use_log

def normalize(arr, flat, dark, cutoff=None, ncore=None, out=None):
    """
    Normalize raw projection data using the flat and dark field projections.

    Parameters
    ----------
    arr : ndarray
        3D stack of projections.
    flat : ndarray
        3D flat field data.
    dark : ndarray
        3D dark field data.
    cutoff : float, optional
        Permitted maximum vaue for the normalized data.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    out : ndarray, optional
        Output array for result. If same as arr,
        process will be done in-place.

    Returns
    -------
    ndarray
        Normalized 3D tomographic data.
    """
    flat = np.mean(flat, axis=0, dtype=np.float32)
    dark = np.mean(dark, axis=0, dtype=np.float32)

    with mproc.set_numexpr_threads(ncore):
        denom = (flat-dark).astype(np.float32)
        denom[:] = np.where(denom<1, 1, denom)[:]
        out = (arr-dark).astype(np.float32)
        out[:] = np.where(out<1, 1, out)[:]
        out[:] = (out/denom)[:]
        if cutoff is not None:
            cutoff = np.float32(cutoff)
            out[:] = np.where(out>cutoff, cutoff, out)[:]
    return out

def read_center(fn):
    if os.path.basename(fn).split('.')[-1] == 'json':
        with open(fn, 'r') as f:
            tem = json.load(f)
            idx_list = []
            center_list = []
            for ii in tem.items():
                idx_list.append(ii["data_params"]["scan_id"])
                center_list.append(ii["data_params"]["rot_cen"])           
    else:
        f = open(fn, 'r')
        idx_center = f.readlines()
        idx_list = []
        center_list = []
        for ii in idx_center:
            if ii.split():
                idx_list.append(int(ii.split()[0]))
                center_list.append(np.float(ii.split()[1]))
    return idx_list, center_list

def read_config(fn):
    with open(fn, 'r') as f:
        tem = json.load(f)
        idx_list = []
        center_list = []
        for ii in tem.items():
            idx_list.append(ii["data_params"]["scan_id"])
            center_list.append(ii["data_params"]["rot_cen"]) 

def read_data(reader, fn, cfg, sli_start=0, sli_end=20,
              col_start=None, col_end=None,
              flat_name=None, dark_name=None,
              use_fake_flat=False, use_fake_dark=False,
              fake_flat_val=1e4, fake_dark_val=100,
              ds_use=False, ds_level=1.0, mean_axis=None):
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
        use_fake_flat: boolean, optional
            if fake_flat=True, a uniform flat image is generated with its value
            equal to fake_flat_val
        use_fake_dark: boolean, optional
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
    
    if mean_axis is None:
        if ds_use:
            data = data_down_sampling(reader(fn, dtype='data', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg), [1, ds_level, ds_level]).astype(np.float32)
    
            if use_fake_flat:
                white = data_down_sampling(fake_flat_val*np.ones([8, data.shape[1], data.shape[2]]), [1, ds_level, ds_level]).astype(np.float32)
            else:
                white = data_down_sampling(reader(fn, dtype='flat', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg), [1, ds_level, ds_level]).astype(np.float32)
    
            if use_fake_dark:
                dark = data_down_sampling(fake_dark_val*np.ones([8, data.shape[1], data.shape[2]]), [1, ds_level, ds_level]).astype(np.float32)
            else:
                dark = data_down_sampling(reader(fn, dtype='dark', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg), [1, ds_level, ds_level]).astype(np.float32)    
            theta = reader(fn, dtype='theta', sli=[None], cfg=cfg).astype(np.float32)
        else:
            data = reader(fn, dtype='data', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg).astype(np.float32)
    
            if use_fake_flat:
                white = fake_flat_val*np.ones([8, data.shape[1], data.shape[2]], dtype=np.float32)
            else:
                white = reader(fn, dtype='flat', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg).astype(np.float32)
    
            if use_fake_dark:
                dark = fake_dark_val*np.ones([8, data.shape[1], data.shape[2]], dtype=np.float32)
            else:
                dark = reader(fn, dtype='dark', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg).astype(np.float32)    
            theta = reader(fn, dtype='theta', sli=[None], cfg=cfg).astype(np.float32)
    else:
        if ds_use:
            tem = data_down_sampling(reader(fn, dtype='data', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg), [1, ds_level, ds_level]).astype(np.float32)  
            if use_fake_flat:
                white = data_down_sampling(fake_flat_val*np.ones([tem.shape[1], tem.shape[2]]), [ds_level, ds_level]).astype(np.float32)
            else:
                white = data_down_sampling(reader(fn, dtype='flat', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg), [1, ds_level, ds_level]).mean(axis=0).astype(np.float32)
    
            if use_fake_dark:
                dark = data_down_sampling(fake_dark_val*np.ones([tem.shape[1], tem.shape[2]]), [ds_level, ds_level]).astype(np.float32)
            else:
                dark = data_down_sampling(reader(fn, dtype='dark', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg), [1, ds_level, ds_level]).mean(axis=0).astype(np.float32)
            data = ((tem-dark).astype(np.float32)/(white- dark).astype(np.float32)).mean(axis=mean_axis).astype(np.float32)  
            theta = reader(fn, dtype='theta', sli=[None], cfg=cfg).astype(np.float32)             
        else:
            tem = reader(fn, dtype='data', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg).astype(np.float32)
            if use_fake_flat:
                white = fake_flat_val*np.ones([tem.shape[1], tem.shape[2]], dtype=np.float32)
            else:
                white = reader(fn, dtype='flat', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg).mean(axis=0).astype(np.float32)
    
            if use_fake_dark:
                dark = fake_dark_val*np.ones([tem.shape[1], tem.shape[2]], dtype=np.float32)
            else:
                dark = reader(fn, dtype='dark', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg).mean(axis=0).astype(np.float32)
            data = ((tem - dark).astype(np.float32)/(white - dark).astype(np.float32)).mean(axis=mean_axis).astype(np.float32) 
            theta = reader(fn, dtype='theta', sli=[None], cfg=cfg).astype(np.float32)
    gc.collect()
    return data, white, dark, theta

def retrieve_phase(data, pixel_size=1e-4, dist=50, energy=20,
                   alpha=1e-3, pad=True, filter='paganin'):
    data[:] = tomopy.prep.phase.retrieve_phase(data, pixel_size=pixel_size,
                                               dist=dist, energy=energy,
                                               alpha=alpha, pad=pad)[:]
    return data

def run_engine(**kwargs):
    """
    kwargs: dictionary
        This is the reconstruction configuration dictionary tomo_recon_param_dict
        input from tomo_recon_gui
    """
    file_raw_data_top_dir = kwargs['file_params']['raw_data_top_dir']
    file_data_cen_dir = kwargs['file_params']['data_center_dir']
    file_recon_top_dir = kwargs['file_params']['recon_top_dir']
    file_debug_top_dir = kwargs['file_params']['debug_top_dir']
    file_alt_flat_fn = kwargs['file_params']['alt_flat_file']
    file_alt_dark_fn = kwargs['file_params']['alt_dark_file']
    file_cen_list_fn = kwargs['file_params']['cen_list_file']
    file_wedge_ang_auto_det_ref_fn = kwargs['file_params']['wedge_ang_auto_det_ref_fn']
    file_cfg = kwargs['file_params']['io_confg']
    reader = kwargs['file_params']['reader']
    dim_info = kwargs['file_params']['info_reader']

    data_scan_id = kwargs["data_params"]['scan_id']
    data_ds_level = kwargs["data_params"]['downsample']
    data_rot_cen = kwargs["data_params"]['rot_cen']
    data_cen_win_s = kwargs["data_params"]['cen_win_s']
    data_cen_win_w = kwargs["data_params"]['cen_win_w']
    data_sli_s = kwargs["data_params"]['sli_s']
    data_sli_e = kwargs["data_params"]['sli_e']
    data_col_s = kwargs["data_params"]['col_s']
    data_col_e = kwargs["data_params"]['col_e']
    data_fake_flat_val = kwargs["data_params"]['fake_flat_val']
    data_fake_dark_val = kwargs["data_params"]['fake_dark_val']
    data_chunk_sz = kwargs["data_params"]['chunk_sz']
    data_margin = kwargs["data_params"]['margin']
    data_zinger_val = kwargs["data_params"]['zinger_val']
    data_mask_ratio = kwargs["data_params"]['mask_ratio']
    # data_wedge_blankat = kwargs["data_params"]['wedge_blankat']
    data_wedge_missing_s = kwargs["data_params"]['wedge_missing_s']
    data_wedge_missing_e = kwargs["data_params"]['wedge_missing_e']
    data_wedge_ang_auto_det_thres = kwargs["data_params"]['wedge_ang_auto_det_thres']

    rec_type = kwargs['recon_config']['recon_type']
    if data_ds_level == 1:
        rec_use_ds = False
    else:
        rec_use_ds = True
    rec_use_debug = kwargs['recon_config']['use_debug']
    rec_use_alt_flat = kwargs['recon_config']['use_alt_flat']
    rec_use_alt_dark = kwargs['recon_config']['use_alt_dark']
    rec_use_fake_flat = kwargs['recon_config']['use_fake_flat']
    rec_use_fake_dark = kwargs['recon_config']['use_fake_dark']
    rec_use_rm_zinger = kwargs['recon_config']['use_rm_zinger']
    rec_use_mask = kwargs['recon_config']['use_mask']
    rec_use_wedge_ang_auto_det = kwargs['recon_config']['use_wedge_ang_auto_det']
    rec_is_wedge = kwargs['recon_config']['is_wedge']

    flt_param_dict = kwargs["flt_params"]

    alg_param_dict = kwargs["alg_params"]

    if rec_type == 'Trial Center':
        file_raw_fn, file_recon_template = get_file(file_raw_data_top_dir, data_scan_id, file_cfg)

        data, white, dark, theta = read_data(reader, file_raw_fn, file_cfg, 
                                             sli_start=data_sli_s,sli_end=data_sli_s+20,
                                             col_start=data_col_s, col_end=data_col_e,
                                             flat_name=file_alt_flat_fn, dark_name=file_alt_dark_fn,
                                             use_fake_flat=rec_use_fake_flat, use_fake_dark=rec_use_fake_dark,
                                             fake_flat_val=data_fake_flat_val, fake_dark_val=data_fake_dark_val,
                                             ds_use=rec_use_ds, ds_level=data_ds_level)
        theta = theta * np.pi/180.0
        dim = data.shape

        if rec_use_rm_zinger:
            data[:] = tomopy.misc.corr.remove_outlier(data, data_zinger_val, size=15, axis=0)[:]
            white[:] = tomopy.misc.corr.remove_outlier(white, data_zinger_val, size=15, axis=0)[:]

        data[:] = normalize(data, white, dark)[:]

        if rec_use_debug:
            save_debug(file_debug_top_dir, 'norm_data.tiff', data)

        for idx in sorted(flt_param_dict.keys()):
            data[:] = run_filter(data, flt_param_dict[idx])[:]

        if if_log(flt_param_dict):
            data[:] = tomopy.prep.normalize.minus_log(data)[:]
            print('doing log')

        if rec_use_debug:
            save_debug(file_debug_top_dir, 'filtered_data.tiff', data)

        overwrite_dir(file_data_cen_dir)

        if rec_is_wedge:
            if rec_use_wedge_ang_auto_det:
                bad_angs = get_dim_ang_range_range_sli(dim_info, reader, 
                                                       file_wedge_ang_auto_det_ref_fn,
                                                       data_sli_s, file_cfg, 
                                                       sli_end=data_sli_s+20,
                                                       col_start=data_col_s, col_end=data_col_e,
                                                       thres=data_wedge_ang_auto_det_thres)
            else:
                bad_angs = np.arange(data_wedge_missing_s, data_wedge_missing_e)
            data[:] = sort_wedge(data, bad_angs, data_sli_s, data_sli_s+20)[:]

        tomopy.write_center(data[:,int(dim[1]/2)-1:int(dim[1]/2)+1,:], theta, dpath=file_data_cen_dir,
                            cen_range=(data_cen_win_s, data_cen_win_s+data_cen_win_w, 0.5),
                            mask=rec_use_mask, ratio=data_mask_ratio,
                            algorithm='gridrec',
                            filter_name='parzen')
        rec_use_logging = True
        if rec_use_logging:
            fout = os.path.join(os.path.dirname(file_raw_fn), ''.join(os.path.basename(file_raw_fn).split('.')[:-1]) +\
                                '_finding_cneter_log.txt')
            with open(fout, "w") as fo:
                for k, v in kwargs.items():
                    fo.write(str(k) + ': ' + str(v) + '\n\n')
        return 0
    else:
        state = 1
        if rec_type == "Vol Recon: Single":
            if not isinstance(data_scan_id, list):
                idx_list = [data_scan_id]
            else:
                idx_list = data_scan_id
            if not isinstance(data_rot_cen, list):
                cen_list = [data_rot_cen]
            else:
                cen_list = data_rot_cen
            print('single')
        elif rec_type == "Vol Recon: Multi":
            idx_list, cen_list = read_center(file_cen_list_fn)
            print('multi')

        print(idx_list, cen_list)
        if rec_is_wedge:
            if rec_use_wedge_ang_auto_det:
                bad_angs = get_dim_ang_range_range_sli(dim_info, reader, file_wedge_ang_auto_det_ref_fn,
                                                       data_sli_s, file_cfg, 
                                                       sli_end=data_sli_e,
                                                       col_start=data_col_s, col_end=data_col_e,
                                                       thres=data_wedge_ang_auto_det_thres)
            else:
                bad_angs = np.arange(data_wedge_missing_s, data_wedge_missing_e)

        for idx, cen in zip(idx_list, cen_list):
            print(idx, cen)
            file_raw_fn, file_recon_template = get_file(file_raw_data_top_dir, idx, file_cfg)
            # dim = data_info(file_raw_fn)
            dim = dim_info(file_raw_fn, dtype='data', cfg=file_cfg)

            if data_chunk_sz >= (data_sli_e - data_sli_s):
                data_chunk_sz = (data_sli_e - data_sli_s)
                num_chunk = 1
            else:
                num_chunk = np.int((data_sli_e - data_sli_s)/(data_chunk_sz - 2*data_margin)) + 1

            for ii in range(num_chunk):
                try:
                    if ii == 0:
                        sli_start = data_sli_s
                        sli_end = data_sli_s + data_chunk_sz
                    else:
                        sli_start = data_sli_s + ii*(data_chunk_sz - 2*data_margin)
                        sli_end = sli_start + data_chunk_sz
                        if sli_end > data_sli_e:
                            sli_end = data_sli_e
                        if sli_end > dim[1]:
                            sli_end = dim[1]

                    if (sli_end - sli_start) <= data_margin:
                        print('skip')
                        break
                    else:
                        data, white, dark, theta = read_data(reader, file_raw_fn, file_cfg, 
                                                             sli_start=sli_start, sli_end=sli_end,
                                                             col_start=data_col_s, col_end=data_col_e,
                                                             flat_name=file_alt_flat_fn, dark_name=file_alt_dark_fn,
                                                             use_fake_flat=rec_use_fake_flat, use_fake_dark=rec_use_fake_dark,
                                                             fake_flat_val=data_fake_flat_val, fake_dark_val=data_fake_dark_val,
                                                             ds_use=rec_use_ds, ds_level=data_ds_level)
    
                        theta= theta*np.pi/180
                        if rec_use_rm_zinger:
                            data[:] = tomopy.misc.corr.remove_outlier(data, data_zinger_val, size=15, axis=0)[:]
                            white[:] = tomopy.misc.corr.remove_outlier(white, data_zinger_val, size=15, axis=0)[:]
    
                        data[:] = tomopy.prep.normalize.normalize(data, white, dark)[:]
    
                        for fp_key in sorted(flt_param_dict.keys()):
                            data[:] = run_filter(data, flt_param_dict[fp_key])[:]
    
                        if if_log(flt_param_dict):
                            data[:] = tomopy.prep.normalize.minus_log(data)[:]
    
                        if rec_is_wedge:
                            data[:] = sort_wedge(data, bad_angs, sli_start, sli_end)[:]
    
                        data_recon = tomopy.recon(data, theta, center=cen,
                                                  algorithm=alg_param_dict['algorithm'],
                                                  **(translate_params(alg_param_dict['params'])))
                        if rec_use_mask:
                            data_recon = tomopy.circ_mask(data_recon, 0, ratio=data_mask_ratio)
                        dxchange.writer.write_tiff_stack(data_recon[int(data_margin):(sli_end - sli_start - int(data_margin)),:,:],
                                                         axis=0, fname=file_recon_template,
                                                         start=sli_start + int(data_margin),
                                                         overwrite=True)
                        del(data)
                        del(white)
                        del(dark)
                        del(theta)
                        del(data_recon)
                        gc.collect()
                        print ('chunk ',ii, ' reconstruction is saved')
                        print (time.asctime())
                except Exception as e:
                    state = 0
                    print(type(e))
                    print(e.args)
        if state == 1:
            print ('Reconstruction finishes!')
            rec_logging = True
            if rec_logging is True:
                fout = os.path.join(Path(file_recon_template).parents[0], ''.join(os.path.basename(file_raw_fn).split('.')[:-1]) +\
                                    '_recon_log.txt')
                fo = open(fout, "w")
                for k, v in kwargs.items():
                    fo.write(str(k) + ': ' + str(v) + '\n\n')
                fo.close()
            return 0
        else:
            print ('Reconstruction is terminated due to data file error.')
            return -1


def run_filter(data, flt):
    flt_name = flt['filter_name']
    params = translate_params(flt['params'])
    print(flt_name, params)
    if flt_name == "denoise: wiener":
        psfw = int(params['psf'])
        params['psf'] = np.ones([psfw, psfw])/(psfw**2)
        for ii in range(data.shape[0]):
            data[ii] = skr.wiener(data[ii], params['psf'],
                                  params['balance'], reg=params['reg'],
                                  is_real=params['is_real'], clip=params['clip'])[:]
    elif flt_name == "denoise: unsupervised_wiener":
        psfw = int(params['psf'])
        params['psf'] = np.ones([psfw, psfw])/(psfw**2)
        for ii in range(data.shape[0]):
                data[ii], _ = skr.unsupervised_wiener(data[ii], params['psf'],
                                                      reg=params['reg'], user_params=params['user_params'],
                                                      is_real=params['is_real'], clip=params['clip'])[:]
    elif flt_name  == "denoise: denoise_nl_means":
        for ii in range(data.shape[0]):
                data[ii] = skr.denoise_nl_means(data[ii], patch_size=params['patch_size'],
                                                patch_distance=params['patch_distance'],
                                                h=params['h'], multichannel=params['multichannel'],
                                                fast_mode=params['fast_mode'], sigma=params['sigma'],
                                                preserve_range=None)[:]
    elif flt_name  == "denoise: denoise_tv_bregman":
        for ii in range(data.shape[0]):
            data[ii] = skr.denoise_tv_bregman(data[ii], params['weight'],
                                              max_iter=params['max_iter'], eps=params['eps'],
                                              isotropic=params['isotropic'],
                                              multichannel=params['multichannel'])[:]
    elif flt_name  == "denoise: denoise_tv_chambolle":
        for ii in range(data.shape[0]):
            data[ii] = skr.denoise_tv_chambolle(data[ii], params['weight'],
                                                n_iter_max=params['n_iter_max'], eps=params['eps'],
                                                multichannel=params['multichannel'])[:]
    elif flt_name  == "denoise: denoise_bilateral":
        for ii in range(data.shape[0]):
            data[ii] = skr.denoise_bilateral(data[ii], win_size=params['win_size'],
                                            sigma_color=params['sigma_color'], sigma_spatial=params['sigma_spatial'],
                                            bins=params['bins'], mode=params['mode'],
                                            cval=params['cval'], multichannel=params['multichannel'])[:]
    elif flt_name  == "denoise: denoise_wavelet":
        for ii in range(data.shape[0]):
            data[ii] = skr.denoise_wavelet(data[ii], sigma=params['sigma'],
                                           wavelet=params['wavelet'], mode=params['mode'],
                                           wavelet_levels=params['wavelet_levels'], multichannel=params['multichannel'],
                                           convert2ycbcr=params['convert2ycbcr'], method=params['method'],
                                           rescale_sigma=params['rescale_sigma'])[:]
    elif flt_name  == "flatting bkg":
        data[:] = tomopy.prep.normalize.normalize_bg(data, air=params['air'])[:]
    elif flt_name  == "remove cupping":
        data -= params['cc']
    elif flt_name  == "stripe_removal: vo":
        for key in params.keys():
            if key in ["la_size", "sm_size"]:
                params[key] = int(params[key])
        data[:] = tomopy.prep.stripe.remove_all_stripe(data, **params)[:]
    elif flt_name  == "stripe_removal: ti":
        data[:] = tomopy.prep.stripe.remove_all_stripe(data, **params)[:]
    elif flt_name  == "stripe_removal: sf":
        data[:] = tomopy.prep.stripe.remove_all_stripe(data, **params)[:]
    elif flt_name  == "stripe_removal: fw":
        data[:] = tomopy.prep.stripe.remove_all_stripe(data, **params)[:]
    elif flt_name  == "phase retrieval":
        data[:] = retrieve_phase(data, **params)[:]
        # data[:] = tomopy.prep.phase.retrieve_phase(data, **params)[:]
    return data

def save_debug(debug_dir, debug_fn, data):
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    tifffile.imsave(os.path.join(debug_dir, debug_fn), data.astype(np.float32))

def sort_wedge(data, bad_angs, sli_start, sli_end):
    for ii in range(sli_start, sli_end):
        data[bad_angs[ii], ii-sli_start, :] = 0
    return data

def translate_params(params):
    for key, param in params.items():
        if param == 'None':
            params[key] = None
        elif param == 'True':
            params[key] = True
        elif param == 'False':
            params[key] = False
    return params

def overwrite_dir(path):
    if os.path.isdir(path):
        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            os.makedirs(path)
        return 0
    else:
        return -1


