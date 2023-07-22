#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:38:41 2020

@author: xiao
"""

import os, glob, gc, time, shutil, numpy as np

from pathlib import Path
import h5py, tifffile, json
from scipy.ndimage import zoom, gaussian_filter as gf, median_filter as median
import skimage.restoration as skr
from skimage.transform import rescale
import tomopy
from .io import (data_reader, tomo_h5_reader,
                 data_info, tomo_h5_info)

IF_LOG = True

TOMO_RECON_PARAM_DICT = {"file_params": {"raw_data_top_dir": None,
                                         "data_center_dir": None,
                                         "recon_top_dir": None,
                                         "debug_top_dir": None,
                                         "cen_list_file": None,
                                         "alt_flat_file": None,
                                         "alt_dark_file": None,
                                         "wedge_ang_auto_det_ref_fn": None},
                         "recon_config": {"recon_type": 'Trial Cent',
                                          "use_debug": False,
                                          "use_alt_flat": False,
                                          "use_alt_dark": False,
                                          "use_fake_flat": False,
                                          "use_fake_dark": False,
                                          "use_rm_zinger": False,
                                          "use_mask": True,
                                          "use_wedge_ang_auto_det": False,
                                          "is_wedge": False}, "flt_params": {"filters": {}},
                         "data_params": {"scan_id": 0,
                                         "downsample": 1,
                                         "rot_cen": 1280,
                                         "cen_win_s": 1240,
                                         "cen_win_w": 8,
                                         "fake_flat_val": 1e4,
                                         "fake_dark_val": 1e2,
                                         "sli_s": 1280,
                                         "sli_e": 1300,
                                         "chunk_sz": 200,
                                         "margin": 15,
                                         "zinger_val": 500,
                                         "mask_ratio": 1,
                                         # "wedge_blankat":90,
                                         "wedge_missing_s": 500,
                                         "wedge_missing_e": 600,
                                         "wedge_ang_auto_det_thres": 0.1}, "alg_params": {}}

FILTERLIST = ["phase retrieval",
              "flatting bkg",
              "remove cupping",
              "stripe_removal: vo",
              "stripe_removal: ti",
              "stripe_removal: sf",
              "stripe_removal: fw",
              "denoise: median",
              "denoise: wiener",
              "denoise: unsupervised_wiener",
              "denoise: denoise_nl_means",
              "denoise: denoise_tv_bregman",
              "denoise: denoise_tv_chambolle",
              "denoise: denoise_bilateral",
              "denoise: denoise_wavelet"]


def align_proj(data, data_ref=None, **kwargs):
    pass


def data_down_sampling(data, levels):
    if np.any((np.array(levels) - 1) > 0):
        return rescale(data, 1 / np.array(levels), mode='edge', anti_aliasing=True)
    else:
        return rescale(data, 1 / np.array(levels), mode='edge', anti_aliasing=False)


def get_dim_ang_range_range_sli(dim_info, reader, fn, cfg,
                                sli_start=0, sli_end=None,
                                col_start=None, col_end=None,
                                ds_use=False, ds_level=1.0, thres=0.1):
    """
    read_data(reader, fn, cfg, sli_start=0, sli_end=20,
              col_start=None, col_end=None,
              flat_name=None, dark_name=None,
              use_fake_flat=False, use_fake_dark=False,
              fake_flat_val=1e4, fake_dark_val=100,
              ds_use=False, ds_level=1.0, mean_axis=None)

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

    img_shape = dim_info(fn, dtype='data', cfg=cfg["file_params"]["io_confg"])
    if sli_end is None:
        sli_end = img_shape[1]
    elif sli_end > img_shape[1]:
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

    tem, _, _, _ = read_data(reader, fn, cfg,
                             sli_start=sli_start, sli_end=sli_end,
                             col_start=col_start, col_end=col_end,
                             flat_name=None, dark_name=None,
                             use_fake_flat=False, use_fake_dark=False,
                             fake_flat_val=1e4, fake_dark_val=100,
                             ds_use=ds_use, ds_level=ds_level, mean_axis=2)
    bad_angs = tem < thres
    return bad_angs


def get_file(raw_data_top_dir, scan_id, cfg, recon_top_dir=None):
    data_file = glob.glob(os.path.join(raw_data_top_dir, cfg['tomo_raw_fn_template'].format(scan_id)))

    if data_file is []:
        return None

    if recon_top_dir is None:
        output_file = os.path.join(raw_data_top_dir,
                                   'recon_' + os.path.basename(data_file[0]).split(".")[-2],
                                   'recon_' + os.path.basename(data_file[0]).split(".")[-2] + "_{0}.tiff")
    else:
        output_file = os.path.join(recon_top_dir,
                                   'recon_' + os.path.basename(data_file[0]).split(".")[-2],
                                   'recon_' + os.path.basename(data_file[0]).split(".")[-2] + "_{0}.tiff")
    if (not os.path.exists(os.path.dirname(output_file))) and (recon_top_dir is not None):
        os.makedirs(os.path.dirname(output_file), mode=0o777)
    return data_file[0], output_file


def if_log(flt_dict):
    if sorted(flt_dict.keys())[0]:
        for key in sorted(flt_dict.keys()):
            if 'phase retrieval' == flt_dict[key]['filter_name']:
                if 'bronnikov' == flt_dict[key]['params']['filter']:
                    return False
        return True
    else:
        return True


def normalize(arr, flat, dark, fake_flat_roi=None, cutoff=None, ncore=None, out=None):
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
    if fake_flat_roi is None:
        flat = np.mean(flat, axis=0, dtype=np.float32)
    dark = np.mean(dark, axis=0, dtype=np.float32)
    with tomopy.util.mproc.set_numexpr_threads(ncore):
        denom = (flat - dark).astype(np.float32)
        out = (arr - dark).astype(np.float32)
        out[:] = (out / denom)[:]
        out[np.isnan(out)] = 1
        out[np.isinf(out)] = 1
        out[out <= 0] = 1
        if cutoff is not None:
            cutoff = np.float32(cutoff)
            out[:] = np.where(out > cutoff, cutoff, out)[:]
    return out


def read_data(reader, fn, cfg, sli_start=0, sli_end=20,
              col_start=None, col_end=None,
              flat_name=None, dark_name=None,
              use_fake_flat=False, use_fake_dark=False,
              fake_flat_val=1e4, fake_dark_val=100,
              fake_flat_roi=None,
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
    if flat_name is None:
        flat_name = fn
    if dark_name is None:
        dark_name = fn
       
    data_dim = cfg["file_params"]["info_reader"](
        fn, dtype="data", cfg=cfg["file_params"]["io_confg"]
    )

    theta = reader(fn, dtype='theta', sli=[None], cfg=cfg["file_params"]["io_confg"]).astype(np.float32)
    idx = rm_redundant(theta)
    theta = theta[idx]
    if data_dim[0] > theta.shape[0]:
        idx = np.concatenate((idx, np.zeros(data_dim[0] - theta.shape[0], dtype=bool)))

    if mean_axis is None:
        if ds_use:
            data = data_down_sampling(reader(fn, dtype='data',
                                             sli=[None, [sli_start, sli_end], [col_start, col_end]],
                                             cfg=cfg["file_params"]["io_confg"]), [1, ds_level, ds_level]).astype(np.float32)[idx]

            if use_fake_flat:
                if fake_flat_roi is None:
                    white = data_down_sampling(fake_flat_val * np.ones([8, data.shape[1], data.shape[2]]),
                                               [1, ds_level, ds_level]).astype(np.float32)
                else:
                    white = data_down_sampling((data[:, fake_flat_roi[0]:fake_flat_roi[1],
                                                fake_flat_roi[2]:fake_flat_roi[3]].mean(axis=(1, 2), keepdims=True) *
                                                np.ones([1, data.shape[1], data.shape[2]])),
                                               [1, ds_level, ds_level]).astype(np.float32)
            else:
                white = data_down_sampling(
                    reader(flat_name, dtype='flat', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg),
                    [1, ds_level, ds_level]).astype(np.float32)

            if use_fake_dark:
                dark = data_down_sampling(fake_dark_val * np.ones([8, data.shape[1], data.shape[2]]),
                                          [1, ds_level, ds_level]).astype(np.float32)
            else:
                dark = data_down_sampling(
                    reader(dark_name, dtype='dark', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg["file_params"]["io_confg"]),
                    [1, ds_level, ds_level]).astype(np.float32)
        else:
            data = reader(fn, dtype='data', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg["file_params"]["io_confg"]).astype(
                np.float32)[idx]

            if use_fake_flat:
                if fake_flat_roi is None:
                    white = fake_flat_val * np.ones([8, data.shape[1], data.shape[2]], dtype=np.float32)
                else:
                    white = data[:, fake_flat_roi[0]:fake_flat_roi[1],
                                 fake_flat_roi[2]:fake_flat_roi[3]].mean(axis=(1, 2), keepdims=True) * \
                            np.ones([1, data.shape[1], data.shape[2]], dtype=np.float32)
            else:
                white = reader(flat_name, dtype='flat', sli=[None, [sli_start, sli_end], [col_start, col_end]],
                               cfg=cfg["file_params"]["io_confg"]).astype(np.float32)

            if use_fake_dark:
                dark = fake_dark_val * np.ones([8, data.shape[1], data.shape[2]], dtype=np.float32)
            else:
                dark = reader(dark_name, dtype='dark', sli=[None, [sli_start, sli_end], [col_start, col_end]],
                              cfg=cfg["file_params"]["io_confg"]).astype(np.float32)
    else:
        if ds_use:
            data = data_down_sampling(
                reader(fn, dtype='data', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg["file_params"]["io_confg"]),
                [1, ds_level, ds_level]).astype(np.float32)[idx]
            if use_fake_flat:
                white = data_down_sampling(fake_flat_val * np.ones([data.shape[1], data.shape[2]]),
                                           [ds_level, ds_level]).astype(np.float32)
            else:
                white = data_down_sampling(
                    reader(flat_name, dtype='flat', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg["file_params"]["io_confg"]),
                    [1, ds_level, ds_level]).mean(axis=0).astype(np.float32)

            if use_fake_dark:
                dark = data_down_sampling(fake_dark_val * np.ones([data.shape[1], data.shape[2]]),
                                          [ds_level, ds_level]).astype(np.float32)
            else:
                dark = data_down_sampling(
                    reader(dark_name, dtype='dark', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg["file_params"]["io_confg"]),
                    [1, ds_level, ds_level]).mean(axis=0).astype(np.float32)
            data[:] = (data - dark[np.newaxis, :]) / (white[np.newaxis, :] - dark[np.newaxis, :])[:]
            data[np.isinf(data)] = 0
            data[np.isnan(data)] = 0
            data = data.mean(axis=mean_axis).astype(np.float32)
        else:
            data = reader(fn, dtype='data', sli=[None, [sli_start, sli_end], [col_start, col_end]], cfg=cfg["file_params"]["io_confg"]).astype(
                np.float32)[idx]
            if use_fake_flat:
                white = fake_flat_val * np.ones([data.shape[1], data.shape[2]], dtype=np.float32)
            else:
                white = reader(flat_name, dtype='flat', sli=[None, [sli_start, sli_end], [col_start, col_end]],
                               cfg=cfg["file_params"]["io_confg"]).mean(axis=0).astype(np.float32)

            if use_fake_dark:
                dark = fake_dark_val * np.ones([data.shape[1], data.shape[2]], dtype=np.float32)
            else:
                dark = reader(dark_name, dtype='dark', sli=[None, [sli_start, sli_end], [col_start, col_end]],
                              cfg=cfg["file_params"]["io_confg"]).mean(axis=0).astype(np.float32)
            data[:] = (data - dark[np.newaxis, :]) / (white[np.newaxis, :] - dark[np.newaxis, :])[:]
            data[np.isinf(data)] = 0
            data[np.isnan(data)] = 0
            data = data.mean(axis=mean_axis).astype(np.float32)
    gc.collect()
    return data, white, dark, theta


def retrieve_phase(data, pixel_size=1e-4, dist=50, energy=20,
                   alpha=1e-3, pad=True, filter='paganin'):
    if filter == 'paganin':
        data[:] = tomopy.prep.phase.retrieve_phase(data, pixel_size=pixel_size,
                                                   dist=dist, energy=energy,
                                                   alpha=alpha, pad=pad)[:]
    elif filter == 'bronnikov':
        data[:] = (1 - data)[:]
        data[:] = tomopy.prep.phase.retrieve_phase(data, pixel_size=pixel_size,
                                                   dist=dist, energy=energy,
                                                   alpha=alpha, pad=pad)[:]
    return data


def rm_redundant(ang):
    dang = np.diff(ang, prepend=0)
    idx = (dang > 0.01)
    if np.argmax(idx) > 0:
        idx[np.argmax(idx) - 1] = True
    return idx


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
    file_wedge_ang_auto_det_ref_fn = kwargs['file_params']['wedge_ang_auto_det_ref_fn']
    file_cfg = kwargs['file_params']['io_confg']
    reader = kwargs['file_params']['reader']
    dim_info = kwargs['file_params']['info_reader']
    use_struc_h5_reader = kwargs['file_params']['use_struc_h5_reader']

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
    fake_flat_roi = kwargs["data_params"]["fake_flat_roi"]
    data_chunk_sz = kwargs["data_params"]['chunk_sz']
    data_margin = kwargs["data_params"]['margin']
    data_flat_blur_kernel = kwargs["data_params"]["blur_kernel"]
    data_zinger_val = kwargs["data_params"]['zinger_val']
    data_mask_ratio = kwargs["data_params"]['mask_ratio']
    data_wedge_missing_s = kwargs["data_params"]['wedge_missing_s']
    data_wedge_missing_e = kwargs["data_params"]['wedge_missing_e']
    data_wedge_col_s = kwargs["data_params"]["wedge_col_s"]
    data_wedge_col_e = kwargs["data_params"]["wedge_col_e"]
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
    rec_use_blur_flat = kwargs["recon_config"]["use_flat_blur"]
    rec_use_rm_zinger = kwargs['recon_config']['use_rm_zinger']
    rec_use_mask = kwargs['recon_config']['use_mask']
    rec_use_wedge_ang_auto_det = kwargs['recon_config']['use_wedge_ang_auto_det']
    is_wedge = kwargs['recon_config']['is_wedge']

    flt_param_dict = kwargs["flt_params"]
    alg_param_dict = kwargs["alg_params"]

    if not rec_use_alt_flat:
        file_alt_flat_fn = None
    if not rec_use_alt_dark:
        file_alt_dark_fn = None

    if rec_type == 'Trial Cent':
        file_raw_fn, file_recon_template = get_file(file_raw_data_top_dir, data_scan_id, file_cfg)

        data, white, dark, theta = read_data(reader, file_raw_fn, kwargs,
                                             sli_start=data_sli_s, sli_end=data_sli_s + 20,
                                             col_start=data_col_s, col_end=data_col_e,
                                             flat_name=file_alt_flat_fn, dark_name=file_alt_dark_fn,
                                             use_fake_flat=rec_use_fake_flat, use_fake_dark=rec_use_fake_dark,
                                             fake_flat_val=data_fake_flat_val, fake_dark_val=data_fake_dark_val,
                                             ds_use=rec_use_ds, ds_level=data_ds_level)
        theta = theta * np.pi / 180.0
        dim = data.shape

        if rec_use_rm_zinger:
            data[:] = tomopy.misc.corr.remove_outlier(data, data_zinger_val, size=15, axis=0)[:]
            white[:] = tomopy.misc.corr.remove_outlier(white, data_zinger_val, size=15, axis=0)[:]

        if rec_use_blur_flat:
            white[:] = gf(white, data_flat_blur_kernel)[:]
        data[:] = normalize(data, white, dark)[:]

        if is_wedge:
            if rec_use_wedge_ang_auto_det:
                print('wedge_ref_file: ', file_wedge_ang_auto_det_ref_fn)
                bad_angs = get_dim_ang_range_range_sli(dim_info,
                                                       reader,
                                                       file_wedge_ang_auto_det_ref_fn,
                                                       kwargs,
                                                       sli_start=data_sli_s,
                                                       sli_end=data_sli_s + 20,
                                                       col_start=data_wedge_col_s, col_end=data_wedge_col_e,
                                                       ds_use=rec_use_ds, ds_level=data_ds_level,
                                                       thres=data_wedge_ang_auto_det_thres)
            else:
                bad_angs = np.zeros([data.shape[0], data.shape[1]], dtype=bool)
                bad_angs[data_wedge_missing_s:data_wedge_missing_e, :] = True
            data[:] = sort_wedge(data, bad_angs, 0, 20, padval=1)[:]
            if rec_use_debug:
                save_debug(file_debug_top_dir, '1-wedge_data.tiff', data)

        if rec_use_debug:
            save_debug(file_debug_top_dir, '2-norm_data.tiff', data)

        if 0 != len(flt_param_dict.keys()):
            for idx in sorted(flt_param_dict.keys()):
                data[:] = run_filter(data, flt_param_dict[idx])[:]
                if rec_use_debug:
                    save_debug(file_debug_top_dir,
                               '3-filter_name' + str(flt_param_dict[idx].keys()) + '_filtered_data.tiff', data)

        if if_log(flt_param_dict):
            data[:] = tomopy.prep.normalize.minus_log(data)[:]
            print('doing log')
            if "remove cupping" in flt_param_dict.keys():
                params = translate_params(flt_param_dict["remove cupping"]['params'])
                data -= params['cc']
                print("running remove cupping")

        if is_wedge:
            data[:] = sort_wedge(data, bad_angs, 0, 20, padval=0)[:]

        if rec_use_debug:
            save_debug(file_debug_top_dir, '4-log_data.tiff', data)

        overwrite_dir(file_data_cen_dir)

        write_center(data[:, int(dim[1] / 2) - 1:int(dim[1] / 2) + 1, :],
                     theta, dpath=file_data_cen_dir,
                     cen_range=(data_cen_win_s, data_cen_win_s + data_cen_win_w, 0.5),
                     mask=rec_use_mask, ratio=data_mask_ratio,
                     algorithm=(alg_param_dict['algorithm']
                                if (alg_param_dict['algorithm'] != 'astra') else tomopy.astra),
                     **(translate_params(alg_param_dict['params'])))
        rec_use_logging = True
        if rec_use_logging:
            fout = os.path.join(os.path.dirname(file_raw_fn), ''.join(os.path.basename(file_raw_fn).split('.')[:-1]) + \
                                '_finding_cneter_log.txt')
            with open(fout, "w") as fo:
                for k, v in kwargs.items():
                    fo.write(str(k) + ': ' + str(v) + '\n\n')
        return 0
    else:
        state = 1
        if is_wedge:
            if rec_use_wedge_ang_auto_det:
                bad_angs = get_dim_ang_range_range_sli(dim_info, reader, file_wedge_ang_auto_det_ref_fn,
                                                       kwargs, sli_start=0,
                                                       sli_end=None, col_start=data_wedge_col_s,
                                                       col_end=data_wedge_col_e,
                                                       ds_use=rec_use_ds, ds_level=data_ds_level,
                                                       thres=data_wedge_ang_auto_det_thres)
            else:
                bad_angs = np.zeros([data.shape[0], data.shape[1]])
                bad_angs[data_wedge_missing_s:data_wedge_missing_e, :] = 1

        file_raw_fn, file_recon_template = get_file(file_raw_data_top_dir, data_scan_id, file_cfg,
                                                    recon_top_dir=file_recon_top_dir)
        dim = dim_info(file_raw_fn, dtype='data', cfg=file_cfg)

        if data_chunk_sz >= (data_sli_e - data_sli_s):
            data_chunk_sz = (data_sli_e - data_sli_s)
            num_chunk = 1
        else:
            num_chunk = int((data_sli_e - data_sli_s) / (data_chunk_sz - 2 * data_margin)) + 1

        for ii in range(num_chunk):
            try:
                if ii == 0:
                    sli_start = data_sli_s
                    sli_end = data_sli_s + data_chunk_sz
                else:
                    sli_start = data_sli_s + ii * (data_chunk_sz - 2 * data_margin)
                    sli_end = sli_start + data_chunk_sz
                    if sli_end > data_sli_e:
                        sli_end = data_sli_e
                    if sli_end > dim[1]:
                        sli_end = dim[1]

                if (sli_end - sli_start) <= data_margin:
                    print('skip')
                    break
                else:
                    data, white, dark, theta = read_data(reader,
                                                         file_raw_fn, kwargs,
                                                         sli_start=sli_start, sli_end=sli_end,
                                                         col_start=data_col_s, col_end=data_col_e,
                                                         flat_name=file_alt_flat_fn, dark_name=file_alt_dark_fn,
                                                         use_fake_flat=rec_use_fake_flat,
                                                         use_fake_dark=rec_use_fake_dark,
                                                         fake_flat_val=data_fake_flat_val,
                                                         fake_dark_val=data_fake_dark_val,
                                                         fake_flat_roi=fake_flat_roi,
                                                         ds_use=rec_use_ds, ds_level=data_ds_level)

                    if is_wedge:
                        data[:] = sort_wedge(data, bad_angs, sli_start, sli_end, padval=1)[:]

                    theta = theta * np.pi / 180
                    if rec_use_rm_zinger:
                        data[:] = tomopy.misc.corr.remove_outlier(data, data_zinger_val, size=15, axis=0)[:]
                        white[:] = tomopy.misc.corr.remove_outlier(white, data_zinger_val, size=15, axis=0)[:]

                    if rec_use_blur_flat:
                        white[:] = gf(white, data_flat_blur_kernel)[:]
                    data[:] = normalize(data, white, dark)[:]

                    if 0 != len(flt_param_dict.keys()):
                        for fp_key in sorted(flt_param_dict.keys()):
                            data[:] = run_filter(data, flt_param_dict[fp_key])[:]

                    if if_log(flt_param_dict):
                        data[:] = tomopy.prep.normalize.minus_log(data)[:]
                        print('doing log')
                        if "remove cupping" in flt_param_dict.keys():
                            params = translate_params(flt_param_dict["remove cupping"]['params'])
                            data -= params['cc']
                            print("running remove cupping")

                    if is_wedge:
                        data[:] = sort_wedge(data, bad_angs, sli_start, sli_end, padval=0)[:]

                    data_recon = tomopy.recon(data, theta, center=data_rot_cen,
                                              algorithm=(alg_param_dict['algorithm']
                                                         if (alg_param_dict['algorithm'] != 'astra') else tomopy.astra),
                                              **(translate_params(alg_param_dict['params'])))
                    if rec_use_mask:
                        data_recon = tomopy.circ_mask(data_recon, 0, ratio=data_mask_ratio)

                    write_tiff_stack(data_recon[int(data_margin):(sli_end - sli_start - int(data_margin)), :, :],
                                     axis=0, fnt=file_recon_template,
                                     start=sli_start + int(data_margin),
                                     overwrite=True)
                    del (data)
                    del (white)
                    del (dark)
                    del (theta)
                    del (data_recon)
                    gc.collect()
                    print('chunk ', ii, ' reconstruction is saved')
                    print(time.asctime())
            except Exception as e:
                state = 0
                print(type(e))
                print(e.args)
        if state == 1:
            print('Reconstruction finishes!')
            rec_logging = True
            if rec_logging is True:
                fout = Path(file_recon_template).parents[0]/ \
                       (''.join(os.path.basename(file_raw_fn).split('.')[:-1]) + '_recon_log.txt')
                fo = open(fout, "w")
                for k, v in kwargs.items():
                    fo.write(str(k) + ': ' + str(v) + '\n\n')
                fo.close()
            return 0
        else:
            print('Reconstruction is terminated due to error.')
            return -1


def run_filter(data, flt):
    flt_name = flt['filter_name']
    params = translate_params(flt['params'])
    print('running', flt_name)
    if flt_name == "denoise: wiener":
        psfw = int(params['psf'])
        params['psf'] = np.ones([psfw, psfw]) / (psfw ** 2)
        for ii in range(data.shape[0]):
            data[ii] = skr.wiener(data[ii], params['psf'],
                                  params['balance'], reg=params['reg'],
                                  is_real=params['is_real'], clip=params['clip'])[:]
    elif flt_name == "denoise: median":
        data[:] = median(data, size=(int(params['size angle']),
                                     int(params['size y']),
                                     int(params['size x'])))[:]
    elif flt_name == "denoise: unsupervised_wiener":
        psfw = int(params['psf'])
        params['psf'] = np.ones([psfw, psfw]) / (psfw ** 2)
        for ii in range(data.shape[0]):
            data[ii], _ = skr.unsupervised_wiener(data[ii], params['psf'],
                                                  reg=params['reg'], user_params=params['user_params'],
                                                  is_real=params['is_real'], clip=params['clip'])[:]
    elif flt_name == "denoise: denoise_nl_means":
        for ii in range(data.shape[0]):
            data[ii] = skr.denoise_nl_means(data[ii], **params, preserve_range=None)[:]
    elif flt_name == "denoise: denoise_tv_bregman":
        for ii in range(data.shape[0]):
            data[ii] = skr.denoise_tv_bregman(data[ii], params['weight'],
                                              max_iter=params['max_iter'], eps=params['eps'],
                                              isotropic=params['isotropic'],
                                              multichannel=params['multichannel'])[:]
    elif flt_name == "denoise: denoise_tv_chambolle":
        for ii in range(data.shape[0]):
            data[ii] = skr.denoise_tv_chambolle(data[ii], params['weight'],
                                                n_iter_max=params['n_iter_max'], eps=params['eps'],
                                                multichannel=params['multichannel'])[:]
    elif flt_name == "denoise: denoise_bilateral":
        for ii in range(data.shape[0]):
            data[ii] = skr.denoise_bilateral(data[ii], **params)[:]
    elif flt_name == "denoise: denoise_wavelet":
        for ii in range(data.shape[0]):
            data[ii] = skr.denoise_wavelet(data[ii], **params)[:]
    elif flt_name == "flatting bkg":
        data[:] = tomopy.prep.normalize.normalize_bg(data, air=params['air'])[:]
    elif flt_name == "stripe_removal: vo":
        for key in params.keys():
            if key in ["la_size", "sm_size"]:
                params[key] = int(params[key])
        data[:] = tomopy.prep.stripe.remove_all_stripe(data, **params)[:]
    elif flt_name == "stripe_removal: ti":
        data[:] = tomopy.prep.stripe.remove_stripe_ti(data, **params)[:]
    elif flt_name == "stripe_removal: sf":
        params['size'] = int(params['size'])
        data[:] = tomopy.prep.stripe.remove_stripe_sf(data, **params)[:]
    elif flt_name == "stripe_removal: fw":
        params['level'] = int(params['level'])
        data[:] = tomopy.prep.stripe.remove_stripe_fw(data, **params)[:]
    elif flt_name == "phase retrieval":
        data[:] = retrieve_phase(data, **params)[:]
    return data


def save_debug(debug_dir, debug_fn, data):
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    tifffile.imsave(os.path.join(debug_dir, debug_fn), data.astype(np.float32))


def sort_wedge(data, bad_angs, sli_start, sli_end, padval=0):
    bad_angs = match_size(data, bad_angs)
    for ii in range(sli_start, sli_end):
        data[bad_angs[:, ii], ii - sli_start, :] = padval
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


def match_size(data, bad_angs):
    if bad_angs.shape[0] < data.shape[0]:
        bad_angs = np.pad(bad_angs, [[0, data.shape[0] - bad_angs.shape[0]],
                                     [0, 0]], mode='edge')
    elif bad_angs.shape[0] > data.shape[0]:
        bad_angs = bad_angs[:data.shape[0], ...]
    return bad_angs


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


def write_center(tomo, theta, dpath=Path('tmp/center'), cen_range=None, ind=None,
                 mask=False, ratio=1., sinogram_order=False, algorithm='gridrec',
                 filter_name='parzen', **kwargs):
    """
    Save images reconstructed with a range of rotation centers.

    Helps finding the rotation center manually by visual inspection of
    images reconstructed with a set of different centers.The output
    images are put into a specified folder and are named by the
    center position corresponding to the image.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    theta : array
        Projection angles in radian.
    dpath : str, optional
        Folder name to save output images.
    cen_range : list, optional
        [start, end, step] Range of center values.
    ind : int, optional
        Index of the slice to be used for reconstruction.
    mask : bool, optional
        If ``True``, apply a circular mask to the reconstructed image to
        limit the analysis into a circular region.
    ratio : float, optional
        The ratio of the radius of the circular mask to the edge of the
        reconstructed image.
    sinogram_order: bool, optional
        Determins whether data is a stack of sinograms (True, y-axis first axis)
        or a stack of radiographs (False, theta first axis).
    algorithm : {str, function}
        One of the following string values.

        'art'
            Algebraic reconstruction technique :cite:`Kak:98`.
        'bart'
            Block algebraic reconstruction technique.
        'fbp'
            Filtered back-projection algorithm.
        'gridrec'
            Fourier grid reconstruction algorithm :cite:`Dowd:99`,
            :cite:`Rivers:06`.
        'mlem'
            Maximum-likelihood expectation maximization algorithm
            :cite:`Dempster:77`.
        'osem'
            Ordered-subset expectation maximization algorithm
            :cite:`Hudson:94`.
        'ospml_hybrid'
            Ordered-subset penalized maximum likelihood algorithm with
            weighted linear and quadratic penalties.
        'ospml_quad'
            Ordered-subset penalized maximum likelihood algorithm with
            quadratic penalties.
        'pml_hybrid'
            Penalized maximum likelihood algorithm with weighted linear
            and quadratic penalties :cite:`Chang:04`.
        'pml_quad'
            Penalized maximum likelihood algorithm with quadratic penalty.
        'sirt'
            Simultaneous algebraic reconstruction technique.
        'tv'
            Total Variation reconstruction technique
            :cite:`Chambolle:11`.
        'grad'
            Gradient descent method with a constant step size
        'tikh'
            Tikhonov regularization with identity Tikhonov matrix.            


    filter_name : str, optional
        Name of the filter for analytic reconstruction.

        'none'
            No filter.
        'shepp'
            Shepp-Logan filter (default).
        'cosine'
            Cosine filter.
        'hann'
            Cosine filter.
        'hamming'
            Hamming filter.
        'ramlak'
            Ram-Lak filter.
        'parzen'
            Parzen filter.
        'butterworth'
            Butterworth filter.
        'custom'
            A numpy array of size `next_power_of_2(num_detector_columns)/2`
            specifying a custom filter in Fourier domain. The first element
            of the filter should be the zero-frequency component.
        'custom2d'
            A numpy array of size `num_projections*next_power_of_2(num_detector_columns)/2`
            specifying a custom angle-dependent filter in Fourier domain. The first element
            of each filter should be the zero-frequency component.
    """
    tomo = tomopy.util.dtype.as_float32(tomo)
    theta = tomopy.util.dtype.as_float32(theta)

    if sinogram_order:
        dy, dt, dx = tomo.shape
    else:
        dt, dy, dx = tomo.shape
    if ind is None:
        ind = dy // 2
    if cen_range is None:
        center = np.arange(dx / 2 - 5, dx / 2 + 5, 0.5)
    else:
        center = np.arange(*cen_range)

    stack = tomopy.util.dtype.empty_shared_array((len(center), dt, dx))

    for m in range(center.size):
        if sinogram_order:
            stack[m] = tomo[ind]
        else:
            stack[m] = tomo[:, ind, :]

    # Reconstruct the same slice with a range of centers.
    rec = tomopy.recon(stack,
                       theta,
                       center=center,
                       sinogram_order=True,
                       algorithm=algorithm,
                       nchunk=1, **kwargs)

    # Apply circular mask.
    if mask is True:
        rec = tomopy.circ_mask(rec, axis=0)

    # Save images to a temporary folder.
    dpath = os.path.abspath(dpath)
    if not os.path.exists(dpath):
        os.makedirs(dpath)

    for m in range(len(center)):
        tomopy.util.misc.write_tiff(data=rec[m], fname=dpath, digit='{0:.2f}'.format(center[m]))


def write_tiff_stack(img_stack, axis=0, fnt=None, start=0, overwrite=True):
    if axis == 0:
        for ii in range(img_stack.shape[axis]):
            tifffile.imsave(fnt.format(str(start + ii).zfill(5)), img_stack[ii].astype(np.float32))
    elif axis == 1:
        for ii in range(img_stack.shape[axis]):
            tifffile.imsave(fnt.format(str(start + ii).zfill(5)), img_stack[:, ii, :].astype(np.float32))
    elif axis == 2:
        for ii in range(img_stack.shape[axis]):
            tifffile.imsave(fnt.format(str(start + ii).zfill(5)), img_stack[:, :, ii].astype(np.float32))
