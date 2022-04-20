#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:18:13 2020

@author: xiao
"""
import time

from scipy.ndimage import (gaussian_filter, median_filter,
                           binary_erosion, binary_dilation)
from skimage.morphology import disk
import numpy as np

import astra
import tomopy
import tifffile
import h5py


def radr(fn, dtype, path=None, dim=None):
    if dtype == 'prj':
        with h5py.File(fn, 'r') as f:
            if dim is None:
                tem = -np.log((f['/img_tomo'][:] - f['/img_dark_avg'][:]) /
                              (f['/img_bkg_avg'][:] - f['/img_dark_avg'][:]))
                tem[np.isinf(tem)] = 1
                tem[np.isnan(tem)] = 1
            else:
                tem = -np.log((f['/img_tomo'][dim] - f['/img_dark_avg'][dim]) /
                              (f['/img_bkg_avg'][dim] - f['/img_dark_avg'][dim]))
                tem[np.isinf(tem)] = 1
                tem[np.isnan(tem)] = 1
        return tem
    if dtype == 'obj':
        if dim is None:
            return 0
        else:
            return np.zeros(dim)
    if dtype == 'theta':
        """
        return:
            theta: ndarray, projection angles
            dai:   ndarray, data angle indices
            mai:   ndarray, missing angle indices
        """
        if dim is None:
            with h5py.File(fn, 'r') as f:
                if path is None:
                    tem = f['/angle'][:]*np.pi/180
                else:
                    tem = f[path][:]*np.pi/180
            step = np.partition(np.diff(tem), 5)[4]
            print(f"{step = }")
            was = np.argmax(np.diff(tem))
            print(f"{was = }")
            wan = int((tem[was+1] - tem[was])//step)
            print(f"{wan = }")
            return (np.concatenate((tem[:was], np.linspace(tem[was], tem[was+1], wan, endpoint=False), tem[was+1:])),
                    np.concatenate((np.arange(was), np.arange(
                        was+wan, tem.shape[0]+(wan-1)))),
                    np.arange(was, was+wan))
        else:
            with h5py.File(fn, 'r') as f:
                if path is None:
                    tem = f['/angle'][:]*np.pi/180
                else:
                    tem = f[path][:]*np.pi/180
            print(dim[0], dim[1])
            return (tem,
                    np.concatenate(
                        (np.arange(dim[0]), np.arange(dim[1], tem.shape[0]))),
                    np.arange(dim[0], dim[1]))
    if dtype == 'obj_supp':
        with h5py.File(fn, 'r') as f:
            if dim is None:
                if path is None:
                    return (f['/obj_supp'][:] > 0).astype(np.int8)
                else:
                    return (f[path][:] > 0).astype(np.int8)
            else:
                if path is None:
                    return (f['/obj_supp'][dim] > 0).astype(np.int8)
                else:
                    return (f[path][dim] > 0).astype(np.int8)
    if dtype == 'sin_supp':
        with h5py.File(fn, 'r') as f:
            if dim is None:
                if path is None:
                    return (f['/sin_supp'][:] > 0).astype(np.int8)
                else:
                    return (f[path][:] > 0).astype(np.int8)
            else:
                if path is None:
                    return (f['/sin_supp'][dim] > 0).astype(np.int8)
                else:
                    return (f[path][dim] > 0).astype(np.int8)


def cfgr(pars):
    cfg = {'alg': {}, 'dat': {}, 'itr': {}}
    vol_geom = astra.create_vol_geom(pars['rec_dim'])
    prj_geom_f = astra.create_proj_geom('parallel', 1.0, pars['sin_dim'][1],
                                        pars['theta'])
    prj_geom_f_cor = astra.functions.geom_postalignment(
        prj_geom_f, pars['cen'])
    prj_geom_s = astra.create_proj_geom('parallel', 1.0, pars['sin_dim'][1],
                                        pars['theta'][pars['dai']])
    prj_geom_s_cor = astra.functions.geom_postalignment(
        prj_geom_s, pars['cen'])

    cfg['dat']['sin_f_id'] = astra.data2d.create('-sino', prj_geom_f_cor, 0)
    cfg['dat']['sin_s_id'] = astra.data2d.create('-sino', prj_geom_s_cor, 0)
    cfg['dat']['rec_id'] = astra.data2d.create('-vol', vol_geom, 0)
    cfg['dat']['rec_supp_id'] = astra.data2d.create(
        '-vol', vol_geom, pars['rec_supp'])
    cfg['dat']['rec_supp_rel'] = pars['rec_supp_rel']
    cfg['dat']['rec_thres'] = pars['rec_thres']
    cfg['dat']['mai'] = pars['mai']

    if pars['prj_id'] is None:
        cfg['dat']['prj_id'] = astra.create_projector(
            'cuda', prj_geom_f_cor, vol_geom)
    else:
        cfg['dat']['prj_id'] = pars['prj_id']
    if pars['pjtr'] is None:
        cfg['dat']['pjtr'] = astra.OpTomo(cfg['dat']['prj_id'])
    else:
        cfg['dat']['pjtr'] = pars['pjtr']

    cfg['dat']['update_sin'] = pars['update_sin']
    cfg['dat']['sin_f_supp_id'] = astra.data2d.create(
        '-sino', prj_geom_f_cor, pars['sin_supp'])
    cfg['dat']['sin_s_supp_id'] = astra.data2d.create('-sino', prj_geom_s_cor,
                                                      pars['sin_supp'][pars['dai']])
    cfg['dat']['sin_supp_rel'] = pars['sin_supp_rel']
    cfg['dat']['sin_thres'] = pars['sin_thres']
    cfg['dat']['dai'] = pars['dai']

    cfg['alg']['s_cfg'] = pars['s_cfg']
    cfg['alg']['f_cfg'] = pars['f_cfg']

    cfg['itr']['tot_num'] = pars['tot_num']
    cfg['itr']['sub_s_num'] = pars['sub_s_num']
    cfg['itr']['sub_f_num1'] = pars['sub_f_num1']
    cfg['itr']['sub_f_num2'] = pars['sub_f_num2']

    return cfg

    # alg_s_cfg = astra.astra_dict('SIRT_CUDA')
    # alg_s_cfg['ProjectionDataId'] = cfg['sin_s_id']
    # alg_s_cfg['ReconstructionDataId'] = cfg['rec_id']
    # # alg_s_cfg['option'] = {}
    # #alg_s_cfg['option']['SinogramMaskId'] = sin_s_supp_id
    # # alg_s_cfg['option']['ReconstructionMaskId'] = rec_supp_id
    # # alg_s_cfg['option']['MaxConstraint'] = 2e-2
    # #alg_s_cfg['option']['MinConstraint'] = -1e-3
    # # alg_s_id = astra.algorithm.create(alg_s_cfg)

    # alg_f_cfg = astra.astra_dict('SIRT_CUDA')
    # alg_f_cfg['ProjectionDataId'] = cfg['sin_f_id']
    # alg_f_cfg['ReconstructionDataId'] = cfg['rec_id']
    # # alg_f_cfg['option'] = {}
    # #alg_f_cfg['option']['SinogramMaskId'] = sin_f_supp_id
    # #alg_f_cfg['option']['ReconstructionMaskId'] = rec_supp_id
    # #alg_f_cfg['option']['MaxConstraint'] = 2e-2
    # # alg_f_cfg['option']['MinConstraint'] = -1e-3
    # # alg_f_id = astra.algorithm.create(alg_f_cfg)


def rntr(prj, cfgs):
    """
    The reconstruction routine is based on the itr module. The key step is the
    update of the support in the volume (reconstruction) space.

    Parameters
    ----------
    prj  : ndarray, Na X Ny X Nx  array; Na is number of angles, Ny is number
           of slices, Nx is image width in pixel
    cfgs : list of configuration dictionaries. The dictionaries and the orders
           define the reconstruction routine

    Returns
    -------
    recon: ndarray, Ny X Nr X Nr, reconstruction of projection data prj with
           missing angle; Nr is the recon image width defined in cfg['dat']['rec_id']

    """
    recon = np.zeros([prj.shape[1],
                      astra.data2d.get_geometry(cfgs[0]['dat']['rec_id'])[
        'GridRowCount'],
        astra.data2d.get_geometry(cfgs[0]['dat']['rec_id'])['GridColCount']])
    tr = np.zeros([astra.data2d.get_geometry(cfgs[0]['dat']['rec_id'])['GridRowCount'],
                   astra.data2d.get_geometry(cfgs[0]['dat']['rec_id'])['GridColCount']])
    if len(prj.shape) == 3:
        for ii in range(prj.shape[1]):
            prj[:, ii, :] -= ((prj[:, ii, :].sum(axis=1)-np.partition(
                list(set(prj[:, ii, :].sum(axis=1))), 2)[1]+5)/prj.shape[2])[:, np.newaxis]
            prj[:, ii, :][cfgs[0]['dat']['mai'], :] = 0
            prj[:, ii, :][prj[:, ii, :] < 0] = 0

            sf = astra.data2d.get_shared(cfgs[0]['dat']['sin_f_id'])
            sf[:] = prj[:, ii, :]
            ss = astra.data2d.get_shared(cfgs[0]['dat']['sin_s_id'])
            ss[:] = prj[:, ii, :][cfgs[0]['dat']['dai']]

            astra.data2d.get_shared(cfgs[0]['dat']['rec_id'])[:] = tr[:]
            for jj in range(len(cfgs)):
                itr(cfgs[jj])
            recon[ii] = astra.data2d.get(cfgs[jj]['dat']['rec_id'])
    return recon


def extr(oid, out=None):
    pass


def itr(cfg):
    """
    This is the iteration module in wedge data recostruction. The module  is
    composed of  two alternating recon steps: one is the recon with wedge data,
    and aother is the recon with updaed data in the wedge angle range from the
    the first one. In the second step, a median filter is applied to the updated
    data (in sinogram space) to reduce outliers due to inaccurate boundary
    definition. The data residents in the shared space associated to the astra
    ids, so no return is needed.

    Parameters
    ----------
    cfg : dictionary
        The configuration dictionary is composed of three sub-dictionaries: one
        is cfg['alg'] that contains the reconstruction algorithm configurations;
        the second is cfg['dat'] that contains data object ids; and the third
        is cfg['itr'] contains the iteration configuration. The detail structure
        is:
            cfg['alg']:
                's': algorithm configuration with wedge data
                'f': algorithm configuration with wedge data and the update data
                     in the missing angle range
            cfg['dat']:
                'sin_s_id':
                'sin_f_id':
                'rec_id':
                'rec_supp_id':
                'pjtr':
                'mai':
                'rec_supp_rel':
                'rec_thres':
                'update_sin':
                if cfg['dat']['update_sin'] is True:
                    'prj_id':
                    'sin_f_supp_id':
                    'sin_s_supp_id':
                    'dai':
                    'sin_supp_rel':
                    'sin_thres':
            cfg['itr']:
                'tot_num':
                'sub_s_num':
                'sub_f_num1':
                'sub_f_num2':

    Returns
    -------
    None.

    """
    alg_s_id = astra.algorithm.create(cfg['alg']['s_cfg'])
    alg_f_id = astra.algorithm.create(cfg['alg']['f_cfg'])
    ts = np.zeros([astra.data2d.get_geometry(alg_f_id)['Vectors'].shape[0],
                   astra.data2d.get_geometry(alg_f_id)['DetectorCount']],
                  dtype=np.float32)

    for ii in range(cfg['itr']['tot_num']):
        ss = astra.data2d.get_shared(cfg['dat']['sin_s_id'])
        median_filter(ss, [1, 3], output=ss)
        astra.algorithm.run(alg_s_id, cfg['itr']['sub_s_num'])

        v = astra.data2d.get_shared(cfg['dat']['rec_id'])
        sf = astra.data2d.get_shared(cfg['dat']['sin_f_id'])
        ts[:] = cfg['dat']['pjtr'].FP(v)[:]
        sf[cfg['dat']['mai']] = ts[cfg['dat']['mai']]
        astra.algorithm.run(alg_f_id, cfg['itr']['sub_f_num1'])
        median_filter(sf, [1, 3], output=sf)
        astra.algorithm.run(alg_f_id, cfg['itr']['sub_f_num2'])

    # cfg['dat']['rec'][:] = astra.data2d.get(cfg['dat']['rec_id'])[:]
    # rs = astra.data2d.get_shared(cfg['dat']['rec_supp_id'])
    astra.data2d.get_shared(cfg['dat']['rec_supp_id'])[:] = (gaussian_filter(astra.data2d.get(cfg['dat']['rec_id']),
                                                                             cfg['dat']['rec_supp_rel']) > cfg['dat']['rec_thres']).astype(np.int8)[:]
    if cfg['dat']['update_sin']:
        sinogram_id, sino = astra.create_sino(
            astra.data2d.get(cfg['dat']['rec_id']), cfg['dat']['prj_id'])
        ssf = astra.data2d.get_shared(cfg['dat']['sin_f_supp_id'])
        ssf[:] = (sino > cfg['dat']['sin_thres']).astype(np.int8)[:]
        ssf[cfg['dat']['dai']] = (
            cfg['dat']['prj'][cfg['dat']['dai'], :] > 0).astype(np.int8)[:]
        ssf[:] = binary_dilation(ssf, disk(cfg['dat']['sin_supp_rel']))[:]
        # sss = astra.data2d.get_shared(cfg['dat']['sin_s_supp_id'])
        astra.data2d.get_shared(cfg['dat']['sin_s_supp_id'])[
            :] = ssf[[cfg['dat']['dai']]]
        astra.data2d.delete(sinogram_id)
    astra.data2d.delete(alg_s_id)
    astra.data2d.delete(alg_f_id)
