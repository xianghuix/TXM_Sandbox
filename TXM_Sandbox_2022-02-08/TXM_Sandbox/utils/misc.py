#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 17:48:06 2021

@author: xiao
"""
import time, numpy as np
import functools

import h5py

def msgit(wd=100, fill='-'):
    def decorator_msgit(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            print((f'{func.__name__} starts at {time.asctime()}'.center(wd, fill)))
            rlt = func(*args, **kwargs)
            print((f'{func.__name__} finishes at {time.asctime()}'.center(wd, fill)))
            return rlt
        return inner
    return decorator_msgit


def str2bool(str):
    if str.upper() == 'TRUE':
        return True
    elif str.upper() == 'FALSE':
        return False
    else:
        return 'UNKNOWN'


def save_fxi_tomo(fs, imgs, fo=None, ang=None, mag=1, eng=1, note='',
                  pxl=1, scantime=0, logged=True):
    """
    Save tomography image data into an h5 file in FXI architecture.

    Inputs:
    ------
        fs: str; file name of h5 file into which images are saved
        imgs: ndarray; tomo image data to be saved
        fo: str; optional; file name of an original h5 file from which meta
            data can be extracted
        ang: ndarray; optional; angles at which images are taken

    Return:
    ------
        None
    """
    with h5py.File(fs, 'w') as f:
        for ii in f:
            del ii

        if logged:
            f.create_dataset('img_tomo', data=np.exp(-imgs.astype(np.float32)),
                             dtype=np.float32)
        else:
            f.create_dataset('img_tomo', data=imgs.astype(np.float32),
                             dtype=np.float32)
        f.create_dataset('img_dark', data=np.zeros([2, *imgs.shape[1:]]),
                         dtype=np.float32)
        f.create_dataset('img_bkg', data=np.ones([2, *imgs.shape[1:]]),
                         dtype=np.float32)
        if fo is not None:
            with h5py.File(fo, 'r') as foo:
                f.create_dataset('angle', data=foo['angle'][:],
                                 dtype=np.float32)
                f.create_dataset('Magnification', data=foo['Magnification'][()],
                                 dtype=np.float32)
                f.create_dataset('X_eng', data=foo['X_eng'][()],
                                 dtype=np.float32)
                f.create_dataset('Pixel Size', data=foo['Pixel Size'][()])
                f.create_dataset('scan_time', data=foo['scan_time'][()],
                                 dtype=np.float32)
                f.create_dataset('note', data=str(foo['note'][()]))
        elif ang is not None:
            f.create_dataset('angle', data=ang, dtype=np.float32)
            f.create_dataset('Magnification', data=mag, dtype=np.float32)
            f.create_dataset('X_eng', data=eng, dtype=np.float32)
            f.create_dataset('Pixel Size', data=str(pxl))
            f.create_dataset('scan_time', data=scantime, dtype=np.float32)
            f.create_dataset('note', data=str('aligned normalized data'))
        else:
            print('Either an original h5 filename or an angle array is needed.')
