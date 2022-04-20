#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 12:45:50 2019

@author: xiao
"""

import numpy as np

"""
    This clase include all xanes spectra filters, which are used as pre-processing
    to xanes raw spectra;
"""
def cal_pre_edge_fit(eng, pre_edge_fit_coef):
    """
    inputs:
        eng: 1D array-like; full energy list of the spectrum
        pre_edge_fit_coef: array-like; pixel-wise pre_edge fit coef
    ouputs:
        pre_edge_fit: array-like; pixel-wise line profile over eng
    """
    if len(pre_edge_fit_coef.shape) == 3:
        pre_edge_fit = (pre_edge_fit_coef * \
                        (np.vstack((eng, np.ones(eng.shape[0]))).T)\
                            [:, :, np.newaxis, np.newaxis]).sum(axis=1)
    elif len(pre_edge_fit_coef.shape) == 4:
        pre_edge_fit = (pre_edge_fit_coef * \
                        (np.vstack((eng, np.ones(eng.shape[0]))).T)\
                            [:, :, np.newaxis, np.newaxis, np.newaxis]).sum(axis=1)
    return np.squeeze(pre_edge_fit)

def cal_post_edge_fit(eng, post_edge_fit_coef):
    """
    inputs:
        eng: 1D array-like; full energy list of the spectrum
        post_edge_fit_coef: array-like; pixel-wise post_edge fit coef
    ouputs:
        post_edge_fit: array-like; pixel-wise line profile over eng
    """
    if len(post_edge_fit_coef.shape) == 3:
        post_edge_fit = (post_edge_fit_coef * \
                         (np.vstack((eng, np.ones(eng.shape[0]))).T)\
                             [:, :, np.newaxis, np.newaxis]).sum(axis=1)
    elif len(post_edge_fit_coef.shape) == 4:
        post_edge_fit = (post_edge_fit_coef * \
                         (np.vstack((eng, np.ones(eng.shape[0]))).T)\
                             [:, :, np.newaxis, np.newaxis, np.newaxis]).sum(axis=1)
    return np.squeeze(post_edge_fit)
        
def edge_jump_filter(edge_eng, pre_edge_fit_coef, post_edge_fit_coef, pre_edge_sd, edge_jump_thres):
    """
    inputs:
        calculate pre-edge signal standard deviation; this is used to compare with
        edge jump. If the noise (standard deviation) is too high compared to the edge
        jump, the corresponding pixels will be marked as False.
    
        edge_jump, pre_edge_sd and post_edge_sd dimensions are all equal to spectrum
        dimension size - 1 (spectrum.ndim - 1)

    returns: 
        mask: array-like; mask in shape of spectrum.shape
    """
    mask = ((cal_post_edge_fit(edge_eng, post_edge_fit_coef) -
             cal_pre_edge_fit(edge_eng, pre_edge_fit_coef)) 
            > edge_jump_thres*pre_edge_sd).astype(np.int8)
    return mask

def fitted_edges_filter(eng, pre_edge_fit_coef, post_edge_fit_coef, pre_edge_sd, pre_edge_thres):
    """
    inputs:
        pre_edge_fit and post_edge_fit are both linear functions of energy 'eng'.
        pre_edge_fit and post_edge_fit are in shape [2].append(list(spectrum.shape[1:]))
        pre_edge_sd is in shape spectrum.shape[1:]
    returns:
        mask: array-like; mask in shape of spectrum.shape
    """
    mask = np.any((cal_post_edge_fit(eng, post_edge_fit_coef) -
                   cal_pre_edge_fit(eng, pre_edge_fit_coef)) 
                  > pre_edge_thres*pre_edge_sd, axis=0).astype(np.int8)
    return mask











