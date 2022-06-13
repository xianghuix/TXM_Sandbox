#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 08:56:30 2020

@author: xiao
"""

import os, functools, inspect

from pathlib import Path
import h5py, tifffile
import numpy as np

from ..dicts.config_dict import (IO_TOMO_CFG_DEFAULT,
                                 IO_XANES2D_CFG_DEFAULT,
                                 IO_XANES3D_CFG_DEFAULT)


def sli_interpreter(sli_tuple):
    sli = []
    for s in sli_tuple:
        if isinstance(s, list):
            sli.append(slice(s[0], s[1]))
        else:
            sli.append(slice(s))
    return tuple(sli)


def tomo_h5_reader(fn, dtype='data', sli=[[None, None]], cfg=IO_TOMO_CFG_DEFAULT):
    """
    Inputs:
        fnt: string
             file name template to raw h5 data files
        cfg: dictionary; optional
             dictionary with 4 items 'data_path', 'flat_path', 'dark_path', and
             'eng_path' under item 'io_data_structure'; paths point to the
             datasets locations in h5 files
        dtype: string; optional
             specifying which one of the above four items will be read
        sli: list; optional
             a 4-element list; each element is again a list indicating the
             start and end of the slicing on that dimension. The four dimensions
             are [eng, angle, y, x] for dtype='data', or three dimensions of
             [eng, y, x] for dtype='flat' and 'dark', or one dimension of [eng]
             for dtype='eng'
    Returns:
        ndarray
    """
    sli = sli_interpreter(sli)
    with h5py.File(fn, 'r') as f:
        if dtype == 'data':
            # print(sli)
            return np.squeeze(f[cfg['structured_h5_reader']['io_data_structure']['data_path']])[sli]
        elif dtype == 'flat':
            return np.squeeze(f[cfg['structured_h5_reader']['io_data_structure']['flat_path']])[sli]
        elif dtype == 'dark':
            return np.squeeze(f[cfg['structured_h5_reader']['io_data_structure']['dark_path']])[sli]
        elif dtype == 'theta':
            return np.squeeze(f[cfg['structured_h5_reader']['io_data_structure']['theta_path']])[sli[0]]


def tomo_h5_info(fn, dtype='data', cfg=IO_XANES2D_CFG_DEFAULT):
    with h5py.File(fn,"r") as f:
        if dtype == 'data':
            try:
                arr = np.squeeze(f[cfg['structured_h5_reader']['io_data_structure']['data_path']])
                dim = arr.shape
                return dim
            except:
                return 0
        elif dtype == 'flat':
            try:
                arr = np.squeeze(f[cfg['structured_h5_reader']['io_data_structure']['flat_path']])
                dim = arr.shape
                return dim
            except:
                return 0
        elif dtype == 'dark':
            try:
                arr = np.squeeze(f[cfg['structured_h5_reader']['io_data_structure']['dark_path']])
                dim = arr.shape
                return dim
            except:
                return 0
        elif dtype == 'theta':
            try:
                arr = np.squeeze(f[cfg['structured_h5_reader']['io_data_structure']['theta_path']])
                dim = arr.shape
                return dim
            except:
                return 0


def xanes2D_h5_reader(fn, dtype='data', sli=[[None, None]], cfg=IO_XANES2D_CFG_DEFAULT):
    sli = sli_interpreter(sli)
    with h5py.File(fn, 'r') as f:
        if dtype == 'data':
            return np.squeeze(f[cfg['structured_h5_reader']['io_data_structure']['data_path']])[sli]
        elif dtype == 'flat':
            return np.squeeze(f[cfg['structured_h5_reader']['io_data_structure']['flat_path']])[sli]
        elif dtype == 'dark':
            return np.squeeze(f[cfg['structured_h5_reader']['io_data_structure']['dark_path']])[sli]
        elif dtype == 'eng':
            return np.squeeze(f[cfg['structured_h5_reader']['io_data_structure']['eng_path']])[sli]


def xanes3D_h5_reader(fn, ids, dtype='data', sli=[[None, None]], cfg=IO_XANES3D_CFG_DEFAULT):
    """
    Inputs:
        fn:  string
             file name template to raw h5 data files
        cfg: dictionary; optional
             dictionary with 4 items 'data_path', 'flat_path', 'dark_path', and
             'eng_path' under item 'io_data_structure'; paths point to the
             datasets locations in h5 files
        dtype: string; optional
             specifying which one of the above four items will be read
        sli: list; optional
             a 4-element list; each element is again a list indicating the
             start and end of the slicing on that dimension. The four dimensions
             are [eng, angle, y, x] for dtype='data', or three dimensions of
             [eng, y, x] for dtype='flat' and 'dark', or one dimension of [eng]
             for dtype='eng'
    Returns:
        ndarray
    """
    sli = sli_interpreter(sli)
    if dtype == 'data':
        data = []
        for ii in ids:
            fn0 = fn.format(ii)
            with h5py.File(fn0, 'r') as f:
                data.append(np.squeeze(f[cfg['structured_h5_reader']
                              ['io_data_structure']['data_path']])[sli])
        return np.array(data)
    elif dtype == 'flat':
        data = []
        for ii in ids:
            fn0 = fn.format(ii)
            with h5py.File(fn0, 'r') as f:
                data.append(np.squeeze(f[cfg['structured_h5_reader']
                              ['io_data_structure']['flat_path']])[sli])
        return np.array(data)
    elif dtype == 'dark':
        data = []
        for ii in ids:
            fn0 = fn.format(ii)
            with h5py.File(fn0, 'r') as f:
                data.append(np.squeeze(f[cfg['structured_h5_reader']
                              ['io_data_structure']['dark_path']])[sli])
        return np.array(data)
    elif dtype == 'eng':
        eng = []
        for ii in ids:
            fn0 = fn.format(ii)
            with h5py.File(fn0, 'r') as f:
                eng.append(np.squeeze(f[cfg['structured_h5_reader']
                             ['io_data_structure']['eng_path']])[()])
        return np.array(eng)


def data_reader(func):
    @functools.wraps(func)
    def wrapper_data_reader(fn, *args, **kwargs):
        return func(fn, *args, **kwargs)
    return wrapper_data_reader


def data_info(func):
    @functools.wraps(func)
    def wrapper_data_info(fn, dtype='data', *args, **kwargs):
        if not (set(('fn', 'dtype')) <= set(inspect.getfullargspec(func).args)):
            raise TypeError(f'{func.__name__} missing one or more arguments in ("fn", "dtype")')
            return None
        return func(fn, dtype=dtype, *args, **kwargs)
    return wrapper_data_info


def read_xanes2D_wl_fit(fn):
    with h5py.File(fn, 'r') as f:
        return f['/processed_XANES2D/proc_spectrum/whiteline_pos_fit'][:]
