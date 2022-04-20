#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:18:19 2019

@author: xiao
"""
import tifffile, h5py
import os, shutil, sys
import numpy as np

from copy import deepcopy

def fit_poly1d(x, y, order):
    """
    x:      ndarray in shape (1,), independent variable values
    y:      ndarray in shape (1,), dependent variable values at x
    order:  int, polynomial order

    return: callable polynomial function with fitting coefficients of polynomial
    """
    return np.poly1d(np.polyfit(x, y, order))

def fit_poly2d(x, y, order):
    """
    x:      ndarray in shape (n,), independent variable values
    y:      ndarray in shape (n, k), multiple dependent variable value arries at x
    order:  int, polynomial order

    return: ndarray in shape (order+1, y.shape[1]), fitting coefficients of polynomial
    """
    return np.polyfit(x, y, order)

def fit_polynd(x, y, order):
    """
    x:      ndarray in shape (n,), independent variable values
    y:      ndarray in shape (n, ...), multiple dependent variable value arries at x
    order:  int, polynomial order

    return: ndarray in shape (order+1, y.shape[1:]), fitting coefficients of polynomial
    """
    s = list(y.shape)
#    print('x', x.shape, 'y', y.shape)
    s[0] = order + 1
    return (np.polyfit(x, y.reshape([y.shape[0], -1]), order)).reshape(s)

def index_of(arr, e):
    """
    finding the element in arr that has value closes to e
    return: the indices of these elements in arr; return has shape of arr.shape[1:]
    """
    return np.argmin(abs(arr - e), axis=0)

def index_lookup(us_idx, ref_list, us_ratio=10):
    """
    Find the table value with its upsampled index
    upsample table's index and find the interpolated value in the table that
    corresponds to the upsampled index idx

    Parameters
    ----------
    us_idx : int
        upsampled table index.
    ref_list : 1-D array like
        list that will be looked up.
    us_ratio : int, optional
        index's upsampling rator. The default is 100.

    Returns
    -------
    the table's value referred by the upsampled index idx

    """
    ref_list = np.array(ref_list)
    # print('table', ref_list.shape)
    idx_ds = 1.*us_idx/us_ratio
    # print('idx_ds', idx_ds.shape)
    idx = np.int32(np.floor(idx_ds))
    idx[:] = np.where(idx<0, 0, idx)[:]
    idx[:] = np.where(idx>=ref_list.shape[0], ref_list.shape[0]-1, idx)[:]
    xl = ref_list[idx]
    # print('xl', xl.shape)
    idx[:] = np.int32(np.ceil(idx_ds))[:]
    idx[:] = np.where(idx<0, 0, idx)[:]
    idx[:] = np.where(idx>=ref_list.shape[0], ref_list.shape[0]-1, idx)[:]
    xu = ref_list[idx]
    # print('xu', xu.shape)
    x = xl + (xu-xl)*(idx_ds-np.floor(idx_ds))
    # print('x', x.shape)
    return x

