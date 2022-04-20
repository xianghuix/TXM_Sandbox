#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 12:45:50 2019

@author: xiao
"""

import numpy as np
import xanes_math as xm
import xanes_analysis as xa

"""
    This clase include all xanes spectra filters, which are used as pre-processing
    to xanes raw spectra;

"""

# def edge_jump_filter(edge_jump, pre_edge_sd, threshold):
#     """
#     calculate pre-edge signal standard deviation; this is used to compare with
#     edge jump. If the noise (standard deviation) is too high compared to the edge
#     jump, the corresponding pixels will be marked as False.

#     edge_jump, pre_edge_sd and post_edge_sd dimensions are all equal to spectrum
#     dimension size - 1 (spectrum.ndim - 1)

#     return: a matrix of boolean in shape of [threshold].append(list(spectrum.shape[1:]))
#     """
#     return edge_jump > pre_edge_sd*threshold

def edge_jump_filter(edge_eng, pre_edge_fit_coef, post_edge_fit_coef, pre_edge_sd, edge_jump_thres):
    """
    calculate pre-edge signal standard deviation; this is used to compare with
    edge jump. If the noise (standard deviation) is too high compared to the edge
    jump, the corresponding pixels will be marked as False.

    edge_jump, pre_edge_sd and post_edge_sd dimensions are all equal to spectrum
    dimension size - 1 (spectrum.ndim - 1)

    return: a matrix of boolean in shape of [threshold].append(list(spectrum.shape[1:]))
    """
    for ii in range(1, pre_edge_fit_coef.ndim):
        eng = np.array([edge_eng, 1])[:, np.newaxis]
    mask = (((post_edge_fit_coef * eng).sum(axis=0) -
             (pre_edge_fit_coef * eng).sum(axis=0)) > edge_jump_thres).astype(np.int8)
    # if len(pre_edge_sd.shape) == 1:
    #     mask = (((post_edge_fit_coef * np.array([edge_eng, 1])[:, :, np.newaxis]).sum(axis=0) -
    #            (pre_edge_fit_coef * np.array([edge_eng, 1])[:, :, np.newaxis]).sum(axis=0)) > edge_jump_thres).astype(np.int8)
    # elif len(pre_edge_sd.shape) == 2:
    #     mask = (((post_edge_fit_coef * np.array([edge_eng, 1])[:, :, np.newaxis, np.newaxis]).sum(axis=0) -\
    #            (pre_edge_fit_coef * np.array([edge_eng, 1])[:, :, np.newaxis, np.newaxis]).sum(axis=0)) > edge_jump_thres).astype(np.int8)
    return mask

def fitted_edges_filter(eng, pre_edge_fit_coef, post_edge_fit_coef, pre_edge_sd, pre_edge_thres):
    """
    pre_edge_fit and post_edge_fit are both linear functions of energy 'eng'.
    pre_edge_fit and post_edge_fit are in shape [2].append(list(spectrum.shape[1:]))
    pre_edge_sd is in shape spectrum.shape[1:]
    return: mask in shape spectrum.shape
    """
    for ii in range(1, pre_edge_fit_coef.ndim):
        eng = eng[:, np.newaxis]
    mask = np.any(((post_edge_fit_coef * eng).sum(axis=0) -
                   (pre_edge_fit_coef * eng).sum(axis=0)) > pre_edge_thres, axis=0).astype(np.int8)

    # print('pre_edge_fit.shape', pre_edge_fit.shape)
    # print('np.polyval(pre_edge_fit, eng).shape',
    #       np.polyval(pre_edge_fit, eng).shape)

    # mask = np.any((np.polyval(post_edge_fit, eng)-np.polyval(pre_edge_fit, eng)) >
    #               (pre_edge_sd*pre_edge_thres), axis=0).astype(np.int8)

    # print('fitted_edge_filter.shape:', mask.shape)
    # print('pre_edge_sd.shape:', pre_edge_sd.shape)
    return mask











