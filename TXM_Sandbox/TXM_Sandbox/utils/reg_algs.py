#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from functools import partial
from copy import deepcopy
import numpy as np
import tifffile
from scipy.optimize import (differential_evolution as deo,
                            dual_annealing as dao,
                            shgo, brute, fmin)
from scipy.ndimage import (zoom, shift, #rotate,
                           binary_erosion, fourier_shift,
                           gaussian_filter as gf)
from imutils import rotate

from skimage.registration import phase_cross_correlation
from pystackreg import StackReg

from .misc import msgit

#
# def op_comb2(x, ref, tgt, mask, gfk, Ns=40, roi=None, cft='tv', itr=3):
#     """ tv of the difference map between ref and transformed tgt
#     INPUTS:
#        x: transformation parameters;
#           x[0] and x[1]: x and y translations
#           x[2]: rotaiton angle
#           x[3]: scaling factor
#        ref: reference image to be compared to
#        tgt: the image to be transformed to be compared to ref
#        gfk: gaussian_filter kernel
#        cft: cost function type; choose from ['tv', 'de']
#     RETURNS:
#         TV: total variance of the difference map between ref and the
#         transformed tgt
#     """
#     rlt = []
#     if x[-1][0] == x[-1][1] == 1:
#         for ii in range(itr):
#             rlt.append(brute(rigid_transform_comb, x[:3],
#                              Ns=Ns,
#                              args=(ref, tgt, mask, gfk, roi, cft),
#                              workers=-1,
#                              finish=None))
#             print(f"{ii}: {rlt[ii]}")
#             x = [(rlt[ii][0] - (x[0][1] - x[0][0]) / 4, rlt[ii][0] + (x[0][1] - x[0][0]) / 4),
#                  (rlt[ii][1] - (x[1][1] - x[1][0]) / 4, rlt[ii][1] + (x[1][1] - x[1][0]) / 4),
#                  (rlt[ii][2] - (x[2][1] - x[2][0]) / 4, rlt[ii][2] + (x[2][1] - x[2][0]) / 4)]
#     else:
#         for ii in range(itr):
#             rlt.append(brute(transform_comb, x,
#                              Ns=Ns,
#                              args=(ref, tgt, mask, gfk, roi, cft),
#                              workers=-1,
#                              finish=None))
#             print(f"{ii}: {rlt[ii]}")
#             x = [(rlt[ii][0] - (x[0][1] - x[0][0]) / 4, rlt[ii][0] + (x[0][1] - x[0][0]) / 4),
#                  (rlt[ii][1] - (x[1][1] - x[1][0]) / 4, rlt[ii][1] + (x[1][1] - x[1][0]) / 4),
#                  (rlt[ii][2] - (x[2][1] - x[2][0]) / 4, rlt[ii][2] + (x[2][1] - x[2][0]) / 4),
#                  (rlt[ii][3] - (x[3][1] - x[3][0]) / 4, rlt[ii][3] + (x[3][1] - x[3][0]) / 4)]
#     return rlt
#
#
# def op_comb(x, ref, tgt, mask, gfk, popsize=27, Ns=40, roi=None, cft='tv'):
#     """ tv of the difference map between ref and transformed tgt
#     INPUTS:
#        x: transformation parameters;
#           x[0] and x[1]: x and y translations
#           x[2]: rotaiton angle
#           x[3]: scaling factor
#        ref: reference image to be compared to
#        tgt: the image to be transformed to be compared to ref
#        gfk: gaussian_filter kernel
#        cft: cost function type; choose from ['tv', 'de']
#     RETURNS:
#         TV: total variance of the difference map between ref and the
#         transformed tgt
#     """
#     if (x[-1][0] == x[-1][1] == 1) and (x[-2][0] == x[-2][1] == 0):
#         rlt1 = deo(tran, x[:-2],
#                    args=(ref, tgt, mask, gfk, roi, cft),
#                    workers=-1, mutation=(0.5, 1.5), popsize=popsize)
#         print('deo is done')
#         rlt2 = brute(tran,
#                      [(rlt1.x[0] - 1, rlt1.x[0] + 1),
#                       (rlt1.x[1] - 1, rlt1.x[1] + 1)],
#                      Ns=Ns,
#                      args=(ref, tgt, mask, gfk, roi, cft),
#                      workers=-1,
#                      finish=None)
#     elif (x[-1][0] == x[-1][1] == 1):
#         rlt1 = deo(rigid_transform_comb, x[:3],
#                    args=(ref, tgt, mask, gfk, roi, cft),
#                    workers=-1, mutation=(0.5, 1.5), popsize=popsize)
#         print('deo is done')
#         rlt2 = brute(rigid_transform_comb,
#                      [(rlt1.x[0] - 1, rlt1.x[0] + 1),
#                       (rlt1.x[1] - 1, rlt1.x[1] + 1),
#                       (rlt1.x[2] - 1, rlt1.x[2] + 1)],
#                      Ns=Ns,
#                      args=(ref, tgt, mask, gfk, roi, cft),
#                      workers=-1,
#                      finish=None)
#     else:
#         rlt1 = deo(transform_comb, x,
#                    args=(ref, tgt, mask, gfk, roi, cft),
#                    workers=-1, popsize=popsize)
#         print('deo is done')
#         rlt2 = brute(transform_comb,
#                      [(rlt1.x[0] - 1, rlt1.x[0] + 1), (rlt1.x[1] - 1, rlt1.x[1] + 1),
#                       (rlt1.x[2] - 1, rlt1.x[2] + 1), (rlt1.x[3] - 0.2, rlt1.x[3] + 0.2)],
#                      Ns=Ns,
#                      args=(ref, tgt, mask, gfk, roi, cft),
#                      workers=-1,
#                      finish=None)
#     return rlt1, rlt2
#
#
# def op_comb_deos(x, ref, tgt, mask, gfk, Ns=40, roi=None, cft='tv'):
#     """ tv of the difference map between ref and transformed tgt
#     INPUTS:
#        x: transformation parameters;
#           x[0] and x[1]: x and y translations
#           x[2]: rotaiton angle
#           x[3]: scaling factor
#        ref: reference image to be compared to
#        tgt: the image to be transformed to be compared to ref
#        gfk: gaussian_filter kernel
#        cft: cost function type; choose from ['tv', 'de']
#     RETURNS:
#         TV: total variance of the difference map between ref and the
#         transformed tgt
#     """
#     if (x[-1][0] == x[-1][1] == 1) and (x[-2][0] == x[-2][1] == 0):
#         rlt1 = deo(tran, x[:-2],
#                    args=(ref, tgt, mask, gfk, roi, cft),
#                    workers=-1, mutation=(0.5, 1.5))
#         print('deo is done')
#         rlt2 = deo(tran,
#                    [(rlt1.x[0] - 1, rlt1.x[0] + 1),
#                     (rlt1.x[1] - 1, rlt1.x[1] + 1)],
#                    args=(ref, tgt, mask, gfk, roi, cft),
#                    workers=-1,
#                    mutation=(0.5, 1.5))
#     elif (x[-1][0] == x[-1][1] == 1):
#         rlt1 = deo(rigid_transform_comb, x[:3],
#                    args=(ref, tgt, mask, gfk, roi, cft),
#                    workers=-1, mutation=(0.5, 1.5))
#         print('deo is done')
#         rlt2 = deo(rigid_transform_comb,
#                    [(rlt1.x[0] - 1, rlt1.x[0] + 1),
#                     (rlt1.x[1] - 1, rlt1.x[1] + 1),
#                     (rlt1.x[2] - 1, rlt1.x[2] + 1)],
#                    args=(ref, tgt, mask, gfk, roi, cft),
#                    workers=-1,
#                    mutation=(0.5, 1.5))
#     else:
#         rlt1 = deo(transform_comb, x,
#                    args=(ref, tgt, mask, gfk, roi, cft),
#                    workers=-1)
#         print('deo is done')
#         rlt2 = deo(transform_comb,
#                    [(rlt1.x[0] - 1, rlt1.x[0] + 1), (rlt1.x[1] - 1, rlt1.x[1] + 1),
#                     (rlt1.x[2] - 1, rlt1.x[2] + 1), (rlt1.x[3] - 0.2, rlt1.x[3] + 0.2)],
#                    args=(ref, tgt, mask, gfk, roi, cft),
#                    workers=-1,
#                    mutation=(0.5, 1.5))
#     return rlt1, rlt2
#
#
# def mrtv_deo(conf, tgt, src, sub_conf=None, gfk=1, ps=None, roi=None, cft='tv'):
#     """
#     Provide a unified interace for using multi-resolution TV-based registration
#     algorithm with different configuration options.
#
#     Inputs:
#     ______
#         conf: dict; multi-resolution TV minimization configuration at
#               resolution higher than a single pixel; it includes items:
#               'levs': int
#               'wz': list with length equal to dimension
#         tgt: 2D target image to be registed to
#         src: 2D source image to be transformed
#         sub_conf: dict; optional; sub-pixel TV minimization configuration; it includes
#                   items:
#                   'use': boolean; if conduct sub-pixel registration on the end
#                   'type': ['ana', 'fit']; which sub-pixel routine to use
#                   'sp_wz': int; for 'fit' option; the number of TV points to be
#                            used in fitting
#                   'sp_us': int; for 'ana' option; up-sampling factor
#         ps: optonal; pre-defined shift; None or a 2-element tuple
#         gfk: optional; Gaussian filter kernel width
#         cft: string, optional; cost function type; choose between 'tv' for TV and 'de'
#              for differential energy (sum of squared difference)
#
#     Outputs:
#     _______
#         tv_pxl: dict; TV at each search point under levels as the keywords
#         tv_pxl_id: the id of the minimum TV in the flatted tv_pxl for each level
#         shift: ndarray; shift at each level of resolution
#         tot_shift: 2-element tuple; overall displacement between the pair of images
#     """
#     """
#     single process version of mrtv_reg2. Parallelization is implemented
#     on the upper level on which the alignments of pairs of images are
#     parallelized
#     """
#     if ps is not None:
#         if np.array(ps).shape[0] == 2:
#             imgs[1] = np.real(np.fft.ifftn(
#                 fourier_shift(np.fft.fftn(imgs[1]), ps)))
#         elif np.array(ps).shape[0] == 3:
#             imgs[1] = np.real(np.fft.ifftn(
#                 fourier_shift(np.fft.fftn(rotate(imgs[1], ps[2], order=1, reshape=False)), ps[:2])))
#         elif np.array(ps).shape[0] == 4:
#             imgs[1] = np.real(np.fft.ifftn(
#                 fourier_shift(np.fft.fftn(rotate(zoom(imgs[1], ps[3]), ps[2], order=1, reshape=False)), ps[:2])))
#
#     levs = conf['levs']
#     w = np.array(conf['wz'])
#     step = {}
#     tv_pxl = {}
#     tv_pxl_id = np.zeros(levs, dtype=np.int16)
#     shift = np.zeros([levs + 1, np.array(w).shape[0]], dtype=np.float32)
#
#     for ii in range(levs):
#         step[levs - 1 - ii] = 0.5 ** ii
#
#     if ((np.array(tgt.shape) * 0.5 ** (levs - 2) - w) <= 0).any():
#         return -1
#
#     if np.array(w).shape[0] == 2:
#         for ii in range(levs):
#             s = step[ii]
#             f = zoom(tgt, s)
#             m = zoom(src, s)
#             mk = np.zeros(m.shape, dtype=np.int8)
#             mk[int(w[0] * 2 ** (ii - 1)):-int(w[0] * 2 ** (ii - 1)),
#             int(w[1] * 2 ** (ii - 1)):-int(w[1] * 2 ** (ii - 1))] = 1
#
#             sh = shift[ii] * 2
#
#             print(f"{sh.shape=}, {w.shape=}")
#             rlt = deo(tran,
#                       [(sh[0] - w[0] / 2, sh[0] + w[0] / 2),
#                        (sh[1] - w[1] / 2, sh[1] + w[1] / 2)],
#                       args=(f, m, mk, gfk, roi, cft),
#                       workers=-1, mutation=(0.5, 1.5))
#
#             tv_pxl[ii] = rlt.fun
#             shift[ii] = rlt.x
#
#             if ii == levs - 1:
#                 while (shift[ii][0] <= -w[0] / 2 or
#                        shift[ii][0] >= w[0] / 2 or
#                        shift[ii][1] <= -w[1] / 2 or
#                        shift[ii][1] >= w[1] / 2):
#                     if shift[ii][0] <= -w[0] / 2 or shift[ii][0] >= w[0] / 2:
#                         sh[0] = shift[ii][0]
#                     if shift[ii][1] <= -w[1] / 2 or shift[ii][1] >= w[1] / 2:
#                         sh[1] = shift[ii][1]
#                     rlt = brute(tran,
#                                 [(sh[0] - 1, sh[0] + 1),
#                                  (sh[1] - 1, sh[1] + 1)],
#                                 args=(f, m, mk, gfk, roi, cft),
#                                 Ns=10, workers=-1, finish=None)
#
#                     tv_pxl[ii] = rlt[1]
#                     shift[ii] = rlt[0]
#
#         return tv_pxl, shift
#
#
# def mrtv_deo2(conf, tgt, src, sub_conf=None, gfk=1, ps=None, roi=None, subpxl_res=0.2, cft='tv'):
#     """
#     Provide a unified interace for using multi-resolution TV-based registration
#     algorithm with different configuration options.
#
#     Inputs:
#     ______
#         conf: dict; multi-resolution TV minimization configuration at
#               resolution higher than a single pixel; it includes items:
#               'levs': int
#               'wz': list with length equal to dimension
#         tgt: 2D target image to be registed to
#         src: 2D source image to be transformed
#         sub_conf: dict; optional; sub-pixel TV minimization configuration; it includes
#                   items:
#                   'use': boolean; if conduct sub-pixel registration on the end
#                   'type': ['ana', 'fit']; which sub-pixel routine to use
#                   'sp_wz': int; for 'fit' option; the number of TV points to be
#                            used in fitting
#                   'sp_us': int; for 'ana' option; up-sampling factor
#         ps: optonal; pre-defined shift; None or a 2-element tuple
#         gfk: optional; Gaussian filter kernel width
#         cft: string, optional; cost function type; choose between 'tv' for TV and 'de'
#              for differential energy (sum of squared difference)
#
#     Outputs:
#     _______
#         tv_pxl: dict; TV at each search point under levels as the keywords
#         tv_pxl_id: the id of the minimum TV in the flatted tv_pxl for each level
#         shift: ndarray; shift at each level of resolution
#         tot_shift: 2-element tuple; overall displacement between the pair of images
#     """
#     """
#     single process version of mrtv_reg2. Parallelization is implemented
#     on the upper level on which the alignments of pairs of images are
#     parallelized
#     """
#     if ps is not None:
#         if np.array(ps).shape[0] == 2:
#             src = np.real(np.fft.ifftn(
#                 fourier_shift(np.fft.fftn(src), ps)))
#         elif np.array(ps).shape[0] == 3:
#             src = np.real(np.fft.ifftn(
#                 fourier_shift(np.fft.fftn(rotate(src, ps[2], order=1, reshape=False)), ps[:2])))
#         elif np.array(ps).shape[0] == 4:
#             src = np.real(np.fft.ifftn(
#                 fourier_shift(np.fft.fftn(rotate(zoom(src, ps[3]), ps[2], order=1, reshape=False)), ps[:2])))
#
#     w = np.array(conf['wz'])
#     tv_pxl = {}
#     shift = np.zeros([2, np.array(w).shape[0]], dtype=np.float32)
#     sub_wz = 10 * subpxl_res / 2
#
#     if np.array(w).shape[0] == 2:
#         mk = np.zeros(src.shape, dtype=np.int8)
#         mk[int(w[0] / 2):-int(w[0] / 2),
#         int(w[1] / 2):-int(w[1] / 2)] = 1
#
#         print(f"{shift.shape=}, {w.shape=}")
#         rlt = deo(tran,
#                   [(shift[0, 0] - w[0] / 2, shift[0, 0] + w[0] / 2),
#                    (shift[0, 1] - w[1] / 2, shift[0, 1] + w[1] / 2)],
#                   args=(tgt, src, mk, gfk, roi, cft),
#                   workers=-1, mutation=(0.5, 1.5), popsize=25, tol=1e-3)
#
#         tv_pxl[0] = rlt.fun
#         shift[0] = rlt.x
#         print(f"{shift[0]=}")
#         print('deo is done')
#         # print(f"{[(shift[0, 0]-1, shift[0, 0]+1), (shift[0, 1]-1, shift[0, 1]+1)]}")
#
#         rlt = brute(tran,
#                     [(shift[0, 0] - sub_wz, shift[0, 0] + sub_wz),
#                      (shift[0, 1] - sub_wz, shift[0, 1] + sub_wz)],
#                     args=(tgt, src, mk, gfk, roi, cft),
#                     Ns=10, workers=-1, finish=None, full_output=1)
#         tv_pxl[1] = rlt[1]
#         shift[1] = rlt[0]
#         print(f"{shift[1]=}")
#         print('pre brute is done')
#
#         sh = shift[0]
#         while (shift[1, 0] <= sh[0] - sub_wz or
#                shift[1, 0] >= sh[0] + sub_wz or
#                shift[1, 1] <= sh[1] - sub_wz or
#                shift[1, 1] >= sh[1] + sub_wz):
#             if shift[1, 0] <= sh[0] - sub_wz or shift[1, 0] >= sh[0] + sub_wz:
#                 sh[0] = shift[1, 0]
#             if shift[1, 1] <= sh[1] - sub_wz or shift[1, 1] >= sh[1] + sub_wz:
#                 sh[1] = shift[1, 1]
#
#             rlt = brute(tran,
#                         [(sh[0] - sub_wz, sh[0] + sub_wz),
#                          (sh[1] - sub_wz, sh[1] + sub_wz)],
#                         args=(tgt, src, mk, gfk, roi, cft),
#                         Ns=10, workers=-1, finish=None, full_output=1)
#
#             print(f"post brute {sh=} \t {rlt[0]=}")
#             tv_pxl[1] = rlt[1]
#             shift[1] = rlt[0]
#
#         return tv_pxl, shift
#
#
# def transform_comb(x, ref, tgt, mask, gfk, roi=None, cft='tv'):
#     """ tv of the difference map between ref and transformed tgt
#     INPUTS:
#        x: transformation parameters;
#           x[0] and x[1]: x and y translations
#           x[2]: rotaiton angle
#           x[3]: scaling factor
#        ref: reference image to be compared to
#        tgt: the image to be transformed to be compared to ref
#        gfk: gaussian_filter kernel
#        cft: cost function type; choose from ['tv', 'de']
#     RETURNS:
#         TV: total variance of the difference map between ref and the
#         transformed tgt
#     """
#     tgt = zoom(gf(tgt, gfk), x[3])
#     if roi is None:
#         roi = np.s_[int((tgt.shape[0] - ref.shape[0]) / 2):int((tgt.shape[0] - ref.shape[0]) / 2) + ref.shape[0],
#               int((tgt.shape[1] - ref.shape[1]) / 2):int((tgt.shape[1] - ref.shape[1]) / 2) + ref.shape[1]]
#     return cost_norm(ref, shift(rotate(tgt, x[2], order=1, reshape=False), x[:2])[roi], gfk=gfk, type=cft)
#
#
# def rigid_transform_deo(x, ref, tgt, mask, gfk, itr=2, roi=None, cft='tv', order='i'):
#     """ tv of the difference map between ref and transformed tgt
#     INPUTS:
#        x: transformation parameters;
#           x[0] and x[1]: x and y translations
#           x[2]: rotaiton angle
#        ref: reference image to be compared to
#        tgt: the image to be transformed to be compared to ref
#        gfk: gaussian_filter kernel
#        cft: cost function type; choose from ['tv', 'de']
#     RETURNS:
#         TV: total variance of the difference map between ref and the
#         transformed tgt
#     """
#     rlt = {}
#     tgt = gf(tgt, gfk)
#     if roi is None:
#         roi = np.s_[int((tgt.shape[0] - ref.shape[0]) / 2):int((tgt.shape[0] - ref.shape[0]) / 2) + ref.shape[0],
#               int((tgt.shape[1] - ref.shape[1]) / 2):int((tgt.shape[1] - ref.shape[1]) / 2) + ref.shape[1]]
#     for ii in range(itr):
#         rlt[ii] = {}
#         rr = deo(rota, [(x[-1][0], x[-1][1])], args=(tgt, src, mask, gfk, roi, cft, order),
#                  workers=-1, mutation=(0.5, 1.5), popsize=25, tol=1e-3)
#         src[:] = rotate(src, rr.x[0], order=1, reshape=False)[:]
#         rt = deo(tran, list(x[:2]), args=(tgt, src, mask, gfk, roi, cft, order),
#                  workers=-1, mutation=(0.5, 1.5), popsize=25, tol=1e-3)
#         src[:] = shift(src, rt.x)[:]
#         rlt[ii]['rota'] = rr.x
#         rlt[ii]['tran'] = rt.x
#         print(ii)
#     return rlt
#
#
# def rigid_transform_comb(x, ref, tgt, mask, gfk, roi=None, cft='tv', order='i'):
#     """ tv of the difference map between ref and transformed tgt
#     INPUTS:
#        x: transformation parameters;
#           x[0] and x[1]: x and y translations
#           x[2]: rotaiton angle
#        ref: reference image to be compared to
#        tgt: the image to be transformed to be compared to ref
#        gfk: gaussian_filter kernel
#        cft: cost function type; choose from ['tv', 'de']
#     RETURNS:
#         TV: total variance of the difference map between ref and the
#         transformed tgt
#     """
#     tgt = gf(tgt, gfk)
#     if roi is None:
#         roi = np.s_[int((tgt.shape[0] - ref.shape[0]) / 2):int((tgt.shape[0] - ref.shape[0]) / 2) + ref.shape[0],
#               int((tgt.shape[1] - ref.shape[1]) / 2):int((tgt.shape[1] - ref.shape[1]) / 2) + ref.shape[1]]
#     return cost_norm(ref, shift(rotate(tgt, x[2], order=1, reshape=False), x[:2])[roi], gfk=gfk, type=cft, order=order)


# def tran(x, ref, tgt, mask, gfk, roi=None, cft='tv'):
#     """ tv of the difference map between ref and transformed tgt
#     INPUTS:
#        x: transformation parameters;
#           x[0] and x[1]: x and y translations
#           x[2]: rotaiton angle
#        ref: reference image to be compared to
#        tgt: the image to be transformed to be compared to ref
#        gfk: gaussian_filter kernel
#        cft: cost function type; choose from ['tv', 'de']
#     RETURNS:
#         TV: total variance of the difference map between ref and the
#         transformed tgt
#     """
#     tgt = gf(tgt, gfk)
#     if roi is None:
#         roi = np.s_[int((tgt.shape[0]-ref.shape[0])/2):int((tgt.shape[0]-ref.shape[0])/2)+ref.shape[0],
#                     int((tgt.shape[1]-ref.shape[1])/2):int((tgt.shape[1]-ref.shape[1])/2)+ref.shape[1]]
#     return cost_norm(ref, shift(tgt, x[:])[roi], gfk=gfk, type=cft)
#
#
# def ad_deo(bounds, ref, tgt, mask, gfk, roi, itr=3, cft='tv'):
#     rlt = {}
#     if bounds[3][0] == bounds[3][1] == 1:
#         for ii in range(itr):
#             rlt[ii] = {}
#             rlt[ii]['rota'] = deo(rota, [bounds[2]], args=(ref, tgt, mask, gfk, roi, cft),
#                                   workers=-1)
#             tgt = rotate(tgt, rlt[ii]['rota'].x[0], order=1, reshape=False)
#             rlt[ii]['tran'] = deo(tran, bounds[:2], args=(ref, tgt, mask, gfk, roi, cft),
#                                   workers=-1)
#             tgt = shift(tgt, rlt[ii]['tran'].x)
#     else:
#         for ii in range(itr):
#             rlt[ii] = {}
#             rlt[ii]['rota'] = deo(rota, [bounds[2]], args=(ref, tgt, mask, gfk, roi, cft),
#                                   workers=-1)
#             tgt = rotate(tgt, rlt[ii]['rota'].x[0], order=1, reshape=False)
#             rlt[ii]['tran'] = deo(tran, bounds[:2], args=(ref, tgt, mask, gfk, roi, cft),
#                                   workers=-1)
#             tgt = shift(tgt, rlt[ii]['tran'].x)
#             rlt[ii]['zoom'] = deo(zoom, [bounds[3]], args=(ref, tgt, mask, gfk, roi, cft),
#                                   workers=-1)
#             tgt = zoom(tgt, rlt[ii]['zoom'].x[0])
#     return rlt, tgt
#
#
# def edge(x, tgt, src):
#     return np.abs(tgt - shift(src, x)).sum()


# def cost_norm(tgt, src, mask=1, gfk=3, type='tv', order='l1'):
#     """ calculate cost function normalized to the number of effective pixels
#     INPUTS:
#         tgt: reference image
#         src: target image
#         mask: optional; define effective pixels
#         gfk: Gaussian filter kernel width
#         type: optional
#             'tv': total variance of the difference map between ref and tgt
#             'de': difference energy between ref and tgt in L2 form
#         order: optional; choose between 'l1', 'ao', 'ai'
#             'l1': L1-norm
#             'ao': absolute difference between the absolute Gaussian filtered images along each dimension
#             'am': absolute difference between the absolute Gaussian filtered images on mixed dimension order
#     """
#     pnts = mask.sum()
#     # if pnts > tgt.shape[0]*tgt.shape[1]:
#     #     mask = np.ones(tgt.shape)
#     if type == 'tv':
#         if order == 'l1':
#             diff_img = gf(tgt, gfk) - gf(src, gfk)
#             return (((np.abs(np.diff(diff_img, axis=0, prepend=1)) + np.abs(np.diff(diff_img, axis=1, prepend=1)))) * mask).sum()/pnts
#         elif order == 'ao':
#             diff_img = (np.abs(np.abs(np.diff(gf(tgt, gfk), axis=0, prepend=1)) - np.abs(np.diff(gf(src, gfk), axis=0, prepend=1))) + \
#                         np.abs(np.abs(np.diff(gf(tgt, gfk), axis=1, prepend=1)) - np.abs(np.diff(gf(src, gfk), axis=1, prepend=1))))
#             return (diff_img * mask).sum()/pnts
#         elif order == 'am':
#             diff_img = (np.abs(np.abs(np.diff(gf(tgt, gfk), axis=0, prepend=1)) - np.abs(np.diff(gf(src, gfk), axis=1, prepend=1))) + \
#                         np.abs(np.abs(np.diff(gf(tgt, gfk), axis=1, prepend=1)) - np.abs(np.diff(gf(src, gfk), axis=0, prepend=1))))
#             return (diff_img * mask).sum()/pnts
#     elif type == 'de':
#         return (((tgt - src)**2)*mask).sum()/pnts

# def cost_norm(tgt, src, mask=1, gfk=3, type='tv', order='i'):
#     """
#     INPUTS:
#         tgt: reference image
#         src: target image
#         type: optional
#             'tv': total variance of the difference map between ref and tgt
#             'de': difference energy between ref and tgt in L2 form
#     """
#     if not isinstance(mask, np.ndarray):
#         mask = (src != 0)
#         for ii in range(6):
#             mask[:] = binary_erosion(mask)[:]
#     pnts = mask.sum()
#     # if pnts > tgt.shape[0] * tgt.shape[1]:
#     #     mask = np.ones(tgt.shape)
#     if type == 'tv':
#         if order == 'i':
#             diff_img = gf(tgt, gfk) - gf(src, gfk)
#             return (((np.abs(np.diff(diff_img, axis=0, prepend=1)) + np.abs(
#                 np.diff(diff_img, axis=1, prepend=1)))) * mask).sum() / pnts
#         elif order == 'ao':
#             diff_img = (np.abs(
#                 np.abs(np.diff(gf(tgt, gfk), axis=0, prepend=1)) - np.abs(np.diff(gf(src, gfk), axis=0, prepend=1))) + \
#                         np.abs(np.abs(np.diff(gf(tgt, gfk), axis=1, prepend=1)) - np.abs(
#                             np.diff(gf(src, gfk), axis=1, prepend=1))))
#             return (diff_img * mask).sum() / pnts
#         elif order == 'ai':
#             diff_img = (np.abs(
#                 np.abs(np.diff(gf(tgt, gfk), axis=0, prepend=1)) - np.abs(np.diff(gf(src, gfk), axis=1, prepend=1))) + \
#                         np.abs(np.abs(np.diff(gf(tgt, gfk), axis=1, prepend=1)) - np.abs(
#                             np.diff(gf(src, gfk), axis=0, prepend=1))))
#             return (diff_img * mask).sum() / pnts
#     elif type == 'de':
#         return (((tgt - src) ** 2) * mask).sum() / pnts
#         # return (((ref - tgt) ** 2)).sum()
#
#
# def cost_norm_v2(tgt, src, mask=1, type='tv', order='i'):
#     """
#     INPUTS:
#         tgt: reference image
#         src: target image
#         type: optional
#             'tv': total variance of the difference map between ref and tgt
#             'de': difference energy between ref and tgt in L2 form
#     """
#     if isinstance(mask, np.ndarray):
#         pnts = mask.sum()
#     else:
#         pnts = tgt.shape[0] * tgt.shape[1]
#     if type == 'tv':
#         if order == 'i':
#             diff_img = tgt - src
#             return (((np.abs(np.diff(diff_img, axis=0, prepend=1)) + np.abs(
#                 np.diff(diff_img, axis=1, prepend=1)))) * mask).sum() / pnts
#         elif order == 'ao':
#             diff_img = (np.abs(
#                 np.abs(np.diff(tgt, axis=0, prepend=1)) - np.abs(np.diff(src, axis=0, prepend=1))) + \
#                         np.abs(
#                 np.abs(np.diff(tgt, axis=1, prepend=1)) - np.abs(np.diff(src, axis=1, prepend=1))))
#             return (diff_img * mask).sum() / pnts
#         elif order == 'ai':
#             diff_img = (np.abs(
#                 np.abs(np.diff(tgt, axis=0, prepend=1)) - np.abs(np.diff(src, axis=1, prepend=1))) + \
#                         np.abs(
#                 np.abs(np.diff(tgt, axis=1, prepend=1)) - np.abs(np.diff(src, axis=0, prepend=1))))
#             return (diff_img * mask).sum() / pnts
#     elif type == 'de':
#         return (((tgt - src) ** 2) * mask).sum() / pnts


# def cost_abs(tgt, src, mask=1, gfk=3, type='tv', order='i'):
#     """
#     INPUTS:
#         tgt: reference image
#         src: target image
#         type: optional
#             'tv': total variance of the difference map between ref and tgt
#             'de': difference energy between ref and tgt in L2 form
#     """
#     if type == 'tv':
#         if order == 'i':
#             diff_img = gf(tgt, gfk) - gf(src, gfk)
#             return (((np.abs(np.diff(diff_img, axis=0, prepend=1)) +
#                       np.abs(np.diff(diff_img, axis=1, prepend=1)))) * mask).sum()
#         elif order == 'ao':
#             diff_img = (np.abs(
#                 np.abs(np.diff(gf(tgt, gfk), axis=0, prepend=1)) - np.abs(np.diff(gf(src, gfk), axis=0, prepend=1))) + \
#                        np.abs(np.abs(np.diff(gf(tgt, gfk), axis=1, prepend=1)) - np.abs(
#                               np.diff(gf(src, gfk), axis=1, prepend=1))))
#             return (diff_img * mask).sum()
#         elif order == 'ai':
#             diff_img = (np.abs(
#                 np.abs(np.diff(gf(tgt, gfk), axis=0, prepend=1)) - np.abs(np.diff(gf(src, gfk), axis=1, prepend=1))) + \
#                        np.abs(np.abs(np.diff(gf(tgt, gfk), axis=1, prepend=1)) - np.abs(
#                               np.diff(gf(src, gfk), axis=0, prepend=1))))
#             return (diff_img * mask).sum()
#     elif type == 'de':
#         return (((tgt - src) ** 2) * mask).sum()
#
#
# def cost_abs_v2(tgt, src, mask=1, type='tv', order='i'):
#     """
#     INPUTS:
#         tgt: reference image
#         src: target image
#         type: optional
#             'tv': total variance of the difference map between ref and tgt
#             'de': difference energy between ref and tgt in L2 form
#     """
#     if type == 'tv':
#         if order == 'i':
#             diff_img = tgt - src
#             return (((np.abs(np.diff(diff_img, axis=0, prepend=1)) +
#                       np.abs(np.diff(diff_img, axis=1, prepend=1)))) * mask).sum()
#         elif order == 'ao':
#             diff_img = (np.abs(
#                 np.abs(np.diff(tgt, axis=0, prepend=1)) - np.abs(np.diff(src, axis=0, prepend=1))) + \
#                         np.abs(
#                 np.abs(np.diff(tgt, axis=1, prepend=1)) - np.abs(np.diff(src, axis=1, prepend=1))))
#             return (diff_img * mask).sum()
#         elif order == 'ai':
#             diff_img = (np.abs(
#                 np.abs(np.diff(tgt, axis=0, prepend=1)) - np.abs(np.diff(src, axis=1, prepend=1))) + \
#                         np.abs(
#                 np.abs(np.diff(tgt, axis=1, prepend=1)) - np.abs(np.diff(src, axis=0, prepend=1))))
#             return (diff_img * mask).sum()
#     elif type == 'de':
#         return (((tgt - src) ** 2) * mask).sum()


def cost_norm(tgt, src, mask=1, gfk=3, type='tv', order='i', filt=True):
    """
    INPUTS:
        tgt: reference image
        src: target image
        type: optional
            'tv': total variance of the difference map between ref and tgt
            'de': difference energy between ref and tgt in L2 form
    """
    if not isinstance(mask, np.ndarray):
        mask = (src != 0)
        mask[:] = binary_erosion(mask, iterations=3)[:]
    pnts = mask.sum()

    if filt:
        if type == 'tv':
            if order == 'i':
                diff_img = gf(tgt, gfk) - gf(src, gfk)
                return (((np.abs(np.diff(diff_img, axis=0, prepend=1)) + np.abs(
                    np.diff(diff_img, axis=1, prepend=1)))) * mask).sum() / pnts
            elif order == 'ao':
                diff_img = (np.abs(
                    np.abs(np.diff(gf(tgt, gfk), axis=0, prepend=1)) - np.abs(np.diff(gf(src, gfk), axis=0, prepend=1))) + \
                            np.abs(np.abs(np.diff(gf(tgt, gfk), axis=1, prepend=1)) - np.abs(
                                np.diff(gf(src, gfk), axis=1, prepend=1))))
                return (diff_img * mask).sum() / pnts
            elif order == 'ai':
                diff_img = (np.abs(
                    np.abs(np.diff(gf(tgt, gfk), axis=0, prepend=1)) - np.abs(np.diff(gf(src, gfk), axis=1, prepend=1))) + \
                            np.abs(np.abs(np.diff(gf(tgt, gfk), axis=1, prepend=1)) - np.abs(
                                np.diff(gf(src, gfk), axis=0, prepend=1))))
                return (diff_img * mask).sum() / pnts
        elif type == 'de':
            return (((tgt - src) ** 2) * mask).sum() / pnts
    else:
        if type == 'tv':
            if order == 'i':
                diff_img = tgt - src
                return (((np.abs(np.diff(diff_img, axis=0, prepend=1)) + np.abs(
                    np.diff(diff_img, axis=1, prepend=1)))) * mask).sum() / pnts
            elif order == 'ao':
                diff_img = (np.abs(
                    np.abs(np.diff(tgt, axis=0, prepend=1)) - np.abs(np.diff(src, axis=0, prepend=1))) + \
                            np.abs(
                                np.abs(np.diff(tgt, axis=1, prepend=1)) - np.abs(np.diff(src, axis=1, prepend=1))))
                return (diff_img * mask).sum() / pnts
            elif order == 'ai':
                diff_img = (np.abs(
                    np.abs(np.diff(tgt, axis=0, prepend=1)) - np.abs(np.diff(src, axis=1, prepend=1))) + \
                            np.abs(
                                np.abs(np.diff(tgt, axis=1, prepend=1)) - np.abs(np.diff(src, axis=0, prepend=1))))
                return (diff_img * mask).sum() / pnts
        elif type == 'de':
            return (((tgt - src) ** 2) * mask).sum() / pnts


def cost_abs(tgt, src, mask=1, gfk=3, type='tv', order='i', filt=True):
    """
    INPUTS:
        tgt: reference image
        src: target image
        type: optional
            'tv': total variance of the difference map between ref and tgt
            'de': difference energy between ref and tgt in L2 form
    """
    if filt:
        if type == 'tv':
            if order == 'i':
                diff_img = gf(tgt, gfk) - gf(src, gfk)
                return (((np.abs(np.diff(diff_img, axis=0, prepend=1)) +
                          np.abs(np.diff(diff_img, axis=1, prepend=1)))) * mask).sum()
            elif order == 'ao':
                diff_img = (np.abs(
                    np.abs(np.diff(gf(tgt, gfk), axis=0, prepend=1)) - np.abs(np.diff(gf(src, gfk), axis=0, prepend=1))) + \
                            np.abs(np.abs(np.diff(gf(tgt, gfk), axis=1, prepend=1)) - np.abs(
                                np.diff(gf(src, gfk), axis=1, prepend=1))))
                return (diff_img * mask).sum()
            elif order == 'ai':
                diff_img = (np.abs(
                    np.abs(np.diff(gf(tgt, gfk), axis=0, prepend=1)) - np.abs(np.diff(gf(src, gfk), axis=1, prepend=1))) + \
                            np.abs(np.abs(np.diff(gf(tgt, gfk), axis=1, prepend=1)) - np.abs(
                                np.diff(gf(src, gfk), axis=0, prepend=1))))
                return (diff_img * mask).sum()
        elif type == 'de':
            return (((tgt - src) ** 2) * mask).sum()
    else:
        if type == 'tv':
            if order == 'i':
                diff_img = tgt - src
                return (((np.abs(np.diff(diff_img, axis=0, prepend=1)) +
                          np.abs(np.diff(diff_img, axis=1, prepend=1)))) * mask).sum()
            elif order == 'ao':
                diff_img = (np.abs(
                    np.abs(np.diff(tgt, axis=0, prepend=1)) - np.abs(np.diff(src, axis=0, prepend=1))) + \
                            np.abs(
                                np.abs(np.diff(tgt, axis=1, prepend=1)) - np.abs(np.diff(src, axis=1, prepend=1))))
                return (diff_img * mask).sum()
            elif order == 'ai':
                diff_img = (np.abs(
                    np.abs(np.diff(tgt, axis=0, prepend=1)) - np.abs(np.diff(src, axis=1, prepend=1))) + \
                            np.abs(
                                np.abs(np.diff(tgt, axis=1, prepend=1)) - np.abs(np.diff(src, axis=0, prepend=1))))
                return (diff_img * mask).sum()
        elif type == 'de':
            return (((tgt - src) ** 2) * mask).sum()


def tran(x, tgt, src, mask, gfk, roi=None, cft='tv', order='i'):
    """ tv of the difference map between ref and transformed tgt
    INPUTS:
       x: transformation parameters;
          x[0] and x[1]: x and y translations
       ref: reference image to be compared to
       tgt: the image to be transformed to be compared to ref
       gfk: gaussian_filter kernel
    RETURNS:
        TV: total variance of the difference map between ref and the
        transformed tgt
    """
    if roi is None:
        roi = np.s_[int((src.shape[0] - tgt.shape[0]) / 2):int((src.shape[0] - tgt.shape[0]) / 2) + tgt.shape[0],
              int((src.shape[1] - tgt.shape[1]) / 2):int((src.shape[1] - tgt.shape[1]) / 2) + tgt.shape[1]]
    return cost_norm(tgt, shift(src, x[:])[roi], mask=mask, gfk=gfk, type=cft, order=order)


def rota(x, tgt, src, mask, gfk, roi=None, cft='tv', order='i'):
    """ tv of the difference map between ref and transformed tgt
    INPUTS:
       x: transformation parameters;
          x[0]: rotaiton angle
       ref: reference image to be compared to
       tgt: the image to be transformed to be compared to ref
       gfk: gaussian_filter kernel
    RETURNS:
        TV: total variance of the difference map between ref and the
        transformed tgt
    """
    if roi is None:
        roi = np.s_[int((src.shape[0] - tgt.shape[0]) / 2):int((src.shape[0] - tgt.shape[0]) / 2) + tgt.shape[0],
              int((src.shape[1] - tgt.shape[1]) / 2):int((src.shape[1] - tgt.shape[1]) / 2) + tgt.shape[1]]
    # return cost_norm(tgt, rotate(src, x[0], order=1, reshape=False)[roi], mask=mask, gfk=gfk, type=cft, order=order)
    return cost_norm(tgt, rotate(src, x[0])[roi], mask=mask, gfk=gfk, type=cft, order=order)


def zoomu(x, ref, tgt, mask, gfk, roi=None, cft='tv', order='i'):
    """ tv of the difference map between ref and transformed tgt
    INPUTS:
       x: transformation parameters;
          x[0]: scaling factor
       ref: reference image to be compared to
       tgt: the image to be transformed to be compared to ref
       gfk: gaussian_filter kernel
    RETURNS:
        TV: total variance of the difference map between ref and the
        transformed tgt
    """
    tgt = zoom(tgt, x[0])
    if roi is None:
        roi = np.s_[int((tgt.shape[0] - ref.shape[0]) / 2):int((tgt.shape[0] - ref.shape[0]) / 2) + ref.shape[0],
              int((tgt.shape[1] - ref.shape[1]) / 2):int((tgt.shape[1] - ref.shape[1]) / 2) + ref.shape[1]]
    return cost_norm(ref, tgt[roi], gfk=gfk, type=cft, order=order)


def set_tran_quad(t):
    tm = np.eye(3)
    tm[0, 2] = t[0]
    tm[1, 2] = t[1]
    return tm


def set_rota_quad(r):
    rm = np.eye(3)
    rm[0, 0] = rm[1, 1] = np.cos(r)
    rm[0, 1] = -np.sin(r)
    rm[1, 0] = np.sin(r)
    return rm


def cal_ang_rad(c, s):
    if s >= 0:
        return np.arccos(c)
    elif s < 0:
        return 2 * np.pi - np.arccos(c)


def rad_2_deg(rad):
    return 180 * rad / np.pi


def deg_2_rad(deg):
    return deg * np.pi / 180


def rig_transform(src, param, dtype='mrtv'):
    """
    param: rigid-body transform paprameters
    dtype: string, optional; choose between 'pc', 'sr', 'mrtv'
    """
    if dtype == 'mrtv':
        # return shift(rotate(src, rad_2_deg(cal_ang_rad(param[0, 0], param[1, 0])), order=1, reshape=False), param[:2, 2])
        return shift(rotate(src, rad_2_deg(cal_ang_rad(param[0, 0], param[1, 0]))), param[:2, 2])
    elif dtype == 'pc':
        # return shift(rotate(src, -rad_2_deg(param[2]), order=1, reshape=False), [-param[1], -param[0]])
        return shift(rotate(src, -rad_2_deg(param[2])), [-param[1], -param[0]])
    elif dtype == 'sr':
        sr = StackReg(StackReg.RIGID_BODY)
        return sr.transform(src, param)


def mrtv_rigid_reg(tgt, src, pxl_conf, sub_conf, angr, itr=6, ps=None, mask=1, roi=None, gfk=1.5, popsize=27,
                   order='i'):
    s = deepcopy(src)
    m = np.eye(3)
    order = order

    tp, tpi, ss, mrtv_shift = mrtv_reg(pxl_conf, sub_conf, ps=ps, kernel=gfk, imgs=[tgt, s])
    m = np.matmul(set_tran_quad(mrtv_shift), m)
    print("0:\t", mrtv_shift, '\n')
    s[:] = shift(s, mrtv_shift)

    rlt = deo(rota, [list(angr)], args=(tgt, s, mask, gfk, roi, 'tv', order), workers=-1, popsize=popsize)
    m = np.matmul(set_rota_quad(np.pi * rlt.x[-1] / 180), m)
    print("0:\t", rlt.x, "\n")
    # s[:] = rotate(s, rlt.x[-1], order=1, reshape=False)[:]
    s[:] = rotate(s, rlt.x[-1])[:]

    for ii in range(itr):
        tp, tpi, ss, mrtv_shift = mrtv_reg(pxl_conf, sub_conf, ps=None, kernel=gfk, imgs=[tgt, s])
        m = np.matmul(set_tran_quad(mrtv_shift), m)
        print(f"{ii + 1}:\t", mrtv_shift, '\n')
        s[:] = shift(s, mrtv_shift)
        rlt = deo(rota, [(-10, 10)], args=(tgt, s, mask, gfk, roi, 'tv', order), workers=-1, popsize=popsize)
        m = np.matmul(set_rota_quad(np.pi * rlt.x[-1] / 180), m)
        print(f"{ii + 1}:\t", rlt.x, "\n")
        # s[:] = rotate(s, rlt.x[-1], order=1, reshape=False)[:]
        s[:] = rotate(s, rlt.x[-1])[:]
    return m, s


def mrtv_rigid_reg2(tgt, src, pxl_conf, sub_conf, angr, itr=6, ps=None, mask=1, roi=None, gfk=1.5, popsize=27,
                    order='i'):
    s = deepcopy(src)
    m = np.eye(3)
    order = order

    gm = (s != 0).astype(np.int8)
    tp, tpi, ss, mrtv_shift = mrtv_reg_v2(pxl_conf, sub_conf, ps=ps, gm=gm, gfk=gfk, imgs=[tgt, s])
    m = np.matmul(set_tran_quad(mrtv_shift), m)
    print("0:\t", mrtv_shift, '\n')
    s[:] = shift(s, np.array(mrtv_shift))

    rlt = deo(rota, [list(angr)], args=(tgt, s, mask, gfk, roi, 'tv', order), workers=-1, popsize=popsize)
    m = np.matmul(set_rota_quad(np.pi * rlt.x[-1] / 180), m)
    print("0:\t", rlt.x, "\n")
    # s[:] = rotate(s, rlt.x[-1], order=1, reshape=False)[:]
    s[:] = rotate(s, rlt.x[-1])[:]

    for ii in range(itr):
        gm = (s != 0).astype(np.int8)
        tp, tpi, ss, mrtv_shift = mrtv_reg_v2(pxl_conf, sub_conf, ps=None, gm=gm, gfk=gfk, imgs=[tgt, s])
        m = np.matmul(set_tran_quad(mrtv_shift), m)
        print(f"{ii + 1}:\t", mrtv_shift, '\n')
        s[:] = shift(s, np.array(mrtv_shift))

        rlt = deo(rota, [(-10, 10)], args=(tgt, s, mask, gfk, roi, 'tv', order), workers=-1, popsize=popsize)
        m = np.matmul(set_rota_quad(np.pi * rlt.x[-1] / 180), m)
        print(f"{ii + 1}:\t", rlt.x, "\n")
        # s[:] = rotate(s, rlt.x[-1], order=1, reshape=False)[:]
        s[:] = rotate(s, rlt.x[-1])[:]
    return m, s


def mrtv_rigid_reg4(tgt, src, pxl_conf, sub_conf, angr, itr=6, ps=None, mask=1, roi=None, gfk=1.5, popsize=27,
                    order='i', filt=True, norm=True):
    fpath = "/run/media/xiao/Data/data/tmp/gm_{}.tiff"
    s = deepcopy(src)
    m = np.eye(3)
    order = order

    # gm = ((np.abs(s) > 1e-3)*(np.abs(tgt) > 1e-3)).astype(np.int8)
    # tifffile.imsave(fpath.format(str(-1).zfill(2)), gm)
    # tp, tpi, ss, mrtv_shift = mrtv_reg_v4(pxl_conf, sub_conf, ps=ps, gfk=gfk,
    #                                       gm=gm, imgs=[s, tgt], filt=filt, norm=False)
    # m = np.matmul(set_tran_quad(-np.array(mrtv_shift)), m)
    # print("0:\t", mrtv_shift, '\n')
    # s[:] = shift(s, -np.array(mrtv_shift))

    rlt = deo(rota, [list(angr)], args=(tgt, s, mask, gfk, roi, 'tv', order), workers=-1, popsize=popsize)
    m = np.matmul(set_rota_quad(np.pi * rlt.x[-1] / 180), m)
    print("0:\t", rlt.x, "\n")
    # s[:] = rotate(s, rlt.x[-1], order=1, reshape=False)[:]
    s[:] = rotate(s, rlt.x[-1])[:]

    for ii in range(itr):
        gm = ((np.abs(s) > 1e-3)*(np.abs(tgt) > 1e-3)).astype(np.int8)
        tifffile.imsave(fpath.format(str(2 * ii).zfill(2)), gm)
        tp, tpi, ss, mrtv_shift = mrtv_reg_v4(pxl_conf, sub_conf, ps=None, gfk=gfk,
                                              gm=gm, imgs=[s, tgt], filt=filt, norm=norm)
        m = np.matmul(set_tran_quad(-np.array(mrtv_shift)), m)
        print(f"{ii + 1}:\t", mrtv_shift, '\n')
        s[:] = shift(s, -np.array(mrtv_shift))
        tifffile.imsave(fpath.format(str(2 * ii + 1).zfill(2)), (np.abs(s) > 1e-3).astype(np.int8))

        rlt = deo(rota, [(-10, 10)], args=(tgt, s, mask, gfk, roi, 'tv', order), workers=-1, popsize=popsize)
        m = np.matmul(set_rota_quad(np.pi * rlt.x[-1] / 180), m)
        print(f"{ii + 1}:\t", rlt.x, "\n")
        # s[:] = rotate(s, rlt.x[-1], order=1, reshape=False)[:]
        s[:] = rotate(s, rlt.x[-1])[:]
        # tifffile.imsave(fpath.format(str(3 * ii + 2).zfill(2)), (np.abs(s) > 1e-3).astype(np.int8))
    return m, s


def mrtv_rigid_reg5(tgt, src, pxl_conf, sub_conf, angr, itr=6, ps=None, mask=1, roi=None, gfk=1.5, popsize=27,
                    order='i'):
    fpath = "/run/media/xiao/Data/data/tmp/gm_{}.tiff"
    s = deepcopy(src)
    m = np.eye(3)
    order = order

    gm = (np.abs(s) > 1e-3).astype(np.int8)
    tp, tpi, ss, mrtv_shift = mrtv_reg_v4(pxl_conf, sub_conf, ps=ps, gm=gm, gfk=gfk, imgs=[s, tgt])
    m = np.matmul(set_tran_quad(-np.array(mrtv_shift)), m)
    print("0:\t", mrtv_shift, '\n')
    s[:] = shift(s, -np.array(mrtv_shift))

    gm = (np.abs(s) > 1e-3).astype(np.int8)
    rlt = deo(rota, [[-angr[1], -angr[0]]], args=(s, tgt, gm, gfk, roi, 'tv', order), workers=-1, popsize=popsize)
    m = np.matmul(set_rota_quad(-np.pi * rlt.x[-1] / 180), m)
    print("0:\t", rlt.x, "\n")
    # s[:] = rotate(s, -rlt.x[-1], order=1, reshape=False)[:]
    s[:] = rotate(s, -rlt.x[-1])[:]

    for ii in range(itr):
        gm = (np.abs(s) > 1e-3).astype(np.int8)
        tifffile.imsave(fpath.format(str(2*ii).zfill(2)), gm)
        tp, tpi, ss, mrtv_shift = mrtv_reg_v4(pxl_conf, sub_conf, ps=None, gm=gm, gfk=gfk, imgs=[s, tgt])
        m = np.matmul(set_tran_quad(-np.array(mrtv_shift)), m)
        print(f"{ii + 1}:\t", mrtv_shift, '\n')
        s[:] = shift(s, -np.array(mrtv_shift))

        gm = (np.abs(s) > 1e-3).astype(np.int8)
        tifffile.imsave(fpath.format(str(2*ii+1).zfill(2)), gm)
        rlt = deo(rota, [(-10, 10)], args=(s, tgt, gm, gfk, roi, 'tv', order), workers=-1, popsize=popsize)
        m = np.matmul(set_rota_quad(-np.pi * rlt.x[-1] / 180), m)
        print(f"{ii + 1}:\t", rlt.x, "\n")
        # s[:] = rotate(s, -rlt.x[-1], order=1, reshape=False)[:]
        s[:] = rotate(s, -rlt.x[-1])[:]
    return m, s


def tv_l1_pixel(fixed_img, img, mask, shift, filt=True, kernel=3):
    if filt:
        diff_img = gf(fixed_img, kernel) - \
            gf(np.roll(img, shift, axis=[0, 1]), kernel)
        return ((np.abs(np.diff(diff_img, axis=0, prepend=1))+np.abs(np.diff(diff_img, axis=1, prepend=1)))*np.roll(mask, shift, axis=[0, 1])).sum()
    else:
        diff_img = fixed_img - np.roll(img, shift, axis=[0, 1])
        return ((np.abs(np.diff(diff_img, axis=0, prepend=1))+np.abs(np.diff(diff_img, axis=1, prepend=1)))*np.roll(mask, shift, axis=[0, 1])).sum()


def tv_l1_pixel_v2(tgt, src, mask, gfk, shift, type='tv', order='l1'):
    """
    INPUTS:
        tgt: the reference image
        src: the image to be transformed to match tgt
    """
    s = np.roll(src, shift, axis=[0, 1])
    m = np.roll(mask, shift, axis=[0, 1])
    return cost_norm(tgt, s, mask=m, gfk=gfk, type='tv', order='l1')


def tv_l1_pixel_v3(tgt, src, mask, shift):
    """
    INPUTS:
        tgt: the reference image
        src: the image to be transformed to match tgt
    """
    diff_img = tgt-np.roll(src, shift, axis=[0, 1])
    return ((np.abs(np.diff(diff_img, axis=0, prepend=1))+np.abs(np.diff(diff_img, axis=1, prepend=1)))*mask).sum()


def tv_l1_pixel_v4(fixed_img, img, mask, shift, filt=True, kernel=3, norm=True):
    if norm:
        return cost_norm(fixed_img, np.roll(img, shift, axis=[0, 1]), mask=mask, gfk=kernel, type='tv', order='i', filt=filt)
        # diff_img = gf(fixed_img, kernel) - \
        #     gf(np.roll(img, shift, axis=[0, 1]), kernel)
        # return ((np.abs(np.diff(diff_img, axis=0, prepend=1))+np.abs(np.diff(diff_img, axis=1, prepend=1)))*mask).sum()
    else:
        return cost_abs(fixed_img, np.roll(img, shift, axis=[0, 1]), mask=mask, gfk=kernel, type='tv', order='i', filt=filt)
        # diff_img = fixed_img - np.roll(img, shift, axis=[0, 1])
        # return ((np.abs(np.diff(diff_img, axis=0, prepend=1))+np.abs(np.diff(diff_img, axis=1, prepend=1)))*mask).sum()


def tv_l2_pixel(fixed_img, img, mask, shift, norm=True):
    if norm:
        diff_img = fixed_img - np.roll(img, shift, axis=[0, 1])
        diff_img /= np.sqrt((diff_img**2).sum())
    else:
        diff_img = fixed_img - np.roll(img, shift, axis=[0, 1])
    return (((np.diff(diff_img, axis=0, prepend=1))**2 +
             (np.diff(diff_img, axis=1, prepend=1))**2)*mask).sum()


def tv_l1_subpixel(fixed_img, img, mask, shift, filt=True):
    if filt:
        diff_img = (gf(fixed_img, 3) -
                    np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(gf(img, 3)), shift))))
        return ((np.abs(np.diff(diff_img, axis=0, prepend=1)) +
                 np.abs(np.diff(diff_img, axis=1, prepend=1)))*mask).sum()
    else:
        diff_img = fixed_img - \
            np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)))
        return ((np.abs(np.diff(diff_img, axis=0, prepend=1)) +
                 np.abs(np.diff(diff_img, axis=1, prepend=1)))*mask).sum()


def tv_l1_subpixel_fit(x0, x1, tvn):
    coef = np.polyfit(x0, tvn[0], 2)
    tvx = x1[np.argmin(np.polyval(coef, x1))]
    coef = np.polyfit(x0, tvn[1], 2)
    tvy = x1[np.argmin(np.polyval(coef, x1))]
    return [tvx, tvy]


def tv_l1_subpixel_ana(x3, x21, tv3):
    a = (tv3[0][0]+tv3[0][2]-2*tv3[0][1])/2
    b = (tv3[0][2]-tv3[0][0])/2
    c = tv3[0][1]
    tvy = x21[np.argmin(a*x21**2 + b*x21 + c)]
    a = (tv3[1][0]+tv3[1][2]-2*tv3[1][1])/2
    b = (tv3[1][2]-tv3[1][0])/2
    c = tv3[1][1]
    tvx = x21[np.argmin(a*x21**2 + b*x21 + c)]
    return [tvy, tvx]


def tv_l2_subpixel(fixed_img, img, mask, shift, norm=True):
    if norm:
        diff_img = fixed_img - \
            np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)))
        diff_img /= np.sqrt((diff_img**2).sum())
    else:
        diff_img = fixed_img - \
            np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)))
    return (((np.diff(diff_img, axis=0, prepend=1))**2 +
             (np.diff(diff_img, axis=1, prepend=1))**2)*mask).sum()


def mrtv_reg1(fixed_img, img, levs=4, wz=10, sp_wz=20, sp_step=0.2):
    wz = int(wz)
    sch_config = {}
    sch_config[levs-1] = {'wz': sp_wz, 'step': sp_step}
    tv1_pxl = {}
    tv1_pxl_id = np.zeros(levs, dtype=np.int16)
    shift = np.zeros([levs, 2], dtype=np.float32)
    rlt = []

    for ii in range(levs-1):
        sch_config[levs-2-ii] = {'wz': wz, 'step': int(2**ii)}

    border_wz = sch_config[0]['wz']*sch_config[0]['step']
    if ((np.array(fixed_img.shape) - border_wz) <= 0).any():
        return -1

    mask = np.zeros(fixed_img.shape)
    mask[int(border_wz/2):-int(border_wz/2),
         int(border_wz/2):-int(border_wz/2)] = 1

    for ii in range(levs-1):
        w = sch_config[ii]['wz']
        step = sch_config[ii]['step']
        with mp.get_context('spawn').Pool(N_CPU) as pool:
            rlt = pool.map(partial(tv_l1_pixel, fixed_img, img, mask),
                           [[int(step*(jj//w-int(w/2))+shift[:ii, 0].sum()),
                             int(step*(jj % w-int(w/2))+shift[:ii, 1].sum())]
                            for jj in np.int32(np.arange(w**2))])
        pool.close()
        pool.join()

        tem = np.ndarray([w, w], dtype=np.float32)
        for kk in range(w**2):
            tem[kk//w, kk % w] = rlt[kk]
        del(rlt)
        gc.collect()

        tv1_pxl[ii] = np.array(tem)
        tv1_pxl_id[ii] = tv1_pxl[ii].argmin()
        shift[ii, 0] = step*(tv1_pxl_id[ii]//w-int(w/2))
        shift[ii, 1] = step*(tv1_pxl_id[ii] % w-int(w/2))

    w = sch_config[levs-1]['wz']
    step = sch_config[levs-1]['step']
    with mp.get_context('spawn').Pool(N_CPU) as pool:
        rlt = pool.map(partial(tv_l1_subpixel, fixed_img, img, mask),
                       [[step*(jj//w-int(w/2))+shift[:(levs-1), 0].sum(),
                         step*(jj % w-int(w/2))+shift[:(levs-1), 1].sum()]
                        for jj in np.int32(np.arange(w**2))])
    pool.close()
    pool.join()

    tem = np.ndarray([w, w], dtype=np.float32)
    for kk in range(w**2):
        tem[kk//w, kk % w] = rlt[kk]
    del(rlt)
    gc.collect()

    tv1_pxl[levs-1] = np.array(tem)
    tv1_pxl_id[levs-1] = tv1_pxl[levs-1].argmin()
    shift[levs-1, 0] = step*(tv1_pxl_id[levs-1]//w-int(w/2))
    shift[levs-1, 1] = step*(tv1_pxl_id[levs-1] % w-int(w/2))

    print(time.asctime())
    return tv1_pxl, tv1_pxl_id, shift, shift.sum(axis=0)


def mrtv_reg2(fixed_img, img, levs=4, wz=10, sp_wz=20, sp_step=0.2, ps=None):
    if ps is not None:
        img[:] = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), ps)))[:]
    wz = int(wz)
    sch_config = {}
    sch_config[levs-1] = {'wz': sp_wz, 'step': sp_step}
    tv1_pxl = {}
    tv1_pxl_id = np.zeros(levs, dtype=np.int16)
    shift = np.zeros([levs, 2], dtype=np.float32)
    rlt = []

    for ii in range(levs-1):
        sch_config[levs-2-ii] = {'wz': 6, 'step': 0.5**ii}
    sch_config[0] = {'wz': wz, 'step': 0.5**(levs-2)}

    if ((np.array(fixed_img.shape)*0.5**(levs-2) - wz) <= 0).any():
        return -1

    for ii in range(levs-1):
        w = sch_config[ii]['wz']
        step = sch_config[ii]['step']
        f = zoom(fixed_img, step)
        m = zoom(img, step)
        mk = np.zeros(m.shape, dtype=np.int8)
        mk[int(wz*2**(ii-1)):-int(wz*2**(ii-1)),
           int(wz*2**(ii-1)):-int(wz*2**(ii-1))] = 1

        s = np.array([0, 0])
        for kk in range(ii):
            s = s + shift[kk]*2**(ii-kk)
        with mp.get_context('spawn').Pool(N_CPU) as pool:
            rlt = pool.map(partial(tv_l1_pixel, f, m, mk),
                           [[int((jj//w-int(w/2))+s[0]),
                             int((jj % w-int(w/2))+s[1])]
                            for jj in np.int32(np.arange(w**2))])
        pool.close()
        pool.join()

        tem = np.ndarray([w, w], dtype=np.float32)
        for kk in range(w**2):
            tem[kk//w, kk % w] = rlt[kk]
        del(rlt)
        gc.collect()

        tv1_pxl[ii] = np.array(tem)
        tv1_pxl_id[ii] = tv1_pxl[ii].argmin()
        shift[ii, 0] = (tv1_pxl_id[ii]//w-int(w/2))
        shift[ii, 1] = (tv1_pxl_id[ii] % w-int(w/2))

    mk = np.zeros(fixed_img.shape)
    mk[int(wz*2**(levs-2)):-int(wz*2**(levs-2)),
       int(wz*2**(levs-2)):-int(wz*2**(levs-2))] = 1
    w = sch_config[levs-1]['wz']
    step = sch_config[levs-1]['step']
    s = s + shift[levs-2]
    with mp.get_context('spawn').Pool(N_CPU) as pool:
        rlt = pool.map(partial(tv_l1_subpixel, fixed_img, img, mk),
                       [[step*(jj//w-int(w/2))+s[0],
                         step*(jj % w-int(w/2))+s[1]]
                        for jj in np.int32(np.arange(w**2))])
    pool.close()
    pool.join()

    tem = np.ndarray([w, w], dtype=np.float32)
    for kk in range(w**2):
        tem[kk//w, kk % w] = rlt[kk]
    del(rlt)
    gc.collect()

    tv1_pxl[levs-1] = np.array(tem)
    tv1_pxl_id[levs-1] = tv1_pxl[levs-1].argmin()
    shift[levs-1, 0] = step*(tv1_pxl_id[levs-1]//w-int(w/2))
    shift[levs-1, 1] = step*(tv1_pxl_id[levs-1] % w-int(w/2))

    print(time.asctime())
    return tv1_pxl, tv1_pxl_id, shift, s+shift[levs-1]


@msgit(wd=100, fill='-')
def mrtv_reg3(levs=7, wz=10, sp_wz=8, sp_step=0.5, ps=None, imgs=None):
    """
    single process version of mrtv_reg2. Parallelization is implemented
    on the upper level on which the alignments of pairs of images are
    parallelized
    """
    if ps is not None:
        imgs[1] = np.real(np.fft.ifftn(
            fourier_shift(np.fft.fftn(imgs[1]), ps)))

    sch_config = {}
    sch_config[levs-1] = {'wz': sp_wz, 'step': sp_step}
    tv1_pxl = {}
    tv1_pxl_id = np.zeros(levs, dtype=np.int16)
    shift = np.zeros([levs, 2], dtype=np.float32)

    for ii in range(levs-1):
        sch_config[levs-2-ii] = {'wz': 6, 'step': 0.5**ii}
    sch_config[0] = {'wz': wz, 'step': 0.5**(levs-2)}

    if ((np.array(imgs[0].shape)*0.5**(levs-2) - wz) <= 0).any():
        return -1

    for ii in range(levs-1):
        w = sch_config[ii]['wz']
        step = sch_config[ii]['step']
        f = zoom(imgs[0], step)
        m = zoom(imgs[1], step)
        mk = np.zeros(m.shape, dtype=np.int8)
        mk[int(wz*2**(ii-1)):-int(wz*2**(ii-1)),
           int(wz*2**(ii-1)):-int(wz*2**(ii-1))] = 1

        s = np.array([0, 0], dtype=np.int32)
        for jj in range(ii):
            s = np.int_(s + shift[jj]*2**(ii-jj))

        tem = np.ndarray([w, w], dtype=np.float32)
        for jj in range(w):
            for kk in range(w):
                tem[jj, kk] = tv_l1_pixel(f, m, mk,
                                          [jj-int(w/2)+s[0], kk-int(w/2)+s[1]])

        tv1_pxl[ii] = np.array(tem)
        tv1_pxl_id[ii] = tv1_pxl[ii].argmin()
        shift[ii, 0] = (tv1_pxl_id[ii]//w-int(w/2))
        shift[ii, 1] = (tv1_pxl_id[ii] % w-int(w/2))

    mk = np.zeros(imgs[0].shape)
    mk[int(wz*2**(levs-2)):-int(wz*2**(levs-2)),
       int(wz*2**(levs-2)):-int(wz*2**(levs-2))] = 1
    w = sch_config[levs-1]['wz']
    step = sch_config[levs-1]['step']
    s = s + shift[levs-2]

    tem = np.ndarray([w, w], dtype=np.float32)
    for jj in range(w):
        for kk in range(w):
            tem[jj, kk] = tv_l1_subpixel(imgs[0], imgs[1], mk,
                                         [jj-int(w/2)+s[0], kk-int(w/2)+s[1]])

    tv1_pxl[levs-1] = np.array(tem)
    tv1_pxl_id[levs-1] = tv1_pxl[levs-1].argmin()
    shift[levs-1, 0] = step*(tv1_pxl_id[levs-1]//w-int(w/2))
    shift[levs-1, 1] = step*(tv1_pxl_id[levs-1] % w-int(w/2))

    print(time.asctime())
    return tv1_pxl, tv1_pxl_id, shift, s+shift[levs-1]


@msgit(wd=100, fill='-')
def mrtv_reg4(levs=7, wz=10, sp_wz=8, sp_step=0.5, ps=None, imgs=None):
    """
    Single process version of mrtv_reg2. Parallelization is implemented
    on the upper level on which the alignments of pairs of images are
    parallelized

    INPUTS:
        levs: int
    """
    if ps is not None:
        imgs[1] = np.real(np.fft.ifftn(
            fourier_shift(np.fft.fftn(imgs[1]), ps)))

    sch_config = {}
    sch_config[levs-1] = {'wz': sp_wz, 'step': sp_step}
    tv1_pxl = {}
    tv1_pxl_id = np.zeros(levs, dtype=np.int16)
    shift = np.zeros([levs, 2], dtype=np.float32)

    for ii in range(levs-1):
        sch_config[levs-2-ii] = {'wz': wz, 'step': 0.5**ii}
    sch_config[0] = {'wz': wz, 'step': 0.5**(levs-2)}

    if ((np.array(imgs[0].shape)*0.5**(levs-2) - wz) <= 0).any():
        return -1

    for ii in range(levs-1):
        w = sch_config[ii]['wz']
        step = sch_config[ii]['step']
        f = zoom(imgs[0], step)
        m = zoom(imgs[1], step)
        mk = np.zeros(m.shape, dtype=np.int8)
        mk[int(wz*2**(ii-1)):-int(wz*2**(ii-1)),
           int(wz*2**(ii-1)):-int(wz*2**(ii-1))] = 1

        s = np.array([0, 0], dtype=np.int32)
        for jj in range(ii):
            s = np.int_(s + shift[jj]*2**(ii-jj))

        tem = np.ndarray([w, w], dtype=np.float32)
        for jj in range(w):
            for kk in range(w):
                tem[jj, kk] = tv_l1_pixel(f, m, mk,
                                          [jj-int(w/2)+s[0], kk-int(w/2)+s[1]])

        tv1_pxl[ii] = np.array(tem)
        tv1_pxl_id[ii] = tv1_pxl[ii].argmin()
        shift[ii, 0] = (tv1_pxl_id[ii]//w-int(w/2))
        shift[ii, 1] = (tv1_pxl_id[ii] % w-int(w/2))

    s = s + shift[levs-2]

    if sp_wz > (int(wz/2)-1):
        sp_wz = (int(wz/2)-1)

    idx = np.int_(shift[levs-2] + int(w/2))
    print('shift:', shift)
    print('idx:', idx)
    shift[levs-1] = tv_l1_subpixel_fit(np.linspace(-sp_wz, sp_wz, 2*sp_wz+1),
                                       np.linspace(-sp_wz*10, sp_wz *
                                                   10, 2*sp_wz*10+1)/10,
                                       [tv1_pxl[levs-2].reshape([wz, wz])[(idx[0]-sp_wz):(idx[0]+sp_wz+1), idx[1]],
                                        tv1_pxl[levs-2].reshape([wz, wz])[idx[0], (idx[1]-sp_wz):(idx[1]+sp_wz+1)]])

    print(time.asctime())
    return tv1_pxl, tv1_pxl_id, shift, s+shift[levs-1]


@msgit(wd=100, fill='-')
def mrtv_reg5(levs=7, wz=10, sp_wz=1, sp_step=0.5, lsw=10, ps=None, imgs=None):
    """
    single process version of mrtv_reg2. Parallelization is implemented
    on the upper level on which the alignments of pairs of images are
    parallelized
    """
    if ps is not None:
        imgs[1] = np.real(np.fft.ifftn(
            fourier_shift(np.fft.fftn(imgs[1]), ps)))

    sch_config = {}
    sch_config[levs-1] = {'wz': sp_wz, 'step': sp_step}
    tv1_pxl = {}
    tv1_pxl_id = np.zeros(levs, dtype=np.int16)
    shift = np.zeros([levs, 2], dtype=np.float32)

    for ii in range(levs-1):
        sch_config[levs-2-ii] = {'wz': wz, 'step': 0.5**ii}
    sch_config[0] = {'wz': wz, 'step': 0.5**(levs-2)}

    if ((np.array(imgs[0].shape)*0.5**(levs-2) - wz) <= 0).any():
        return -1

    for ii in range(levs-1):
        w = sch_config[ii]['wz']
        step = sch_config[ii]['step']
        f = zoom(imgs[0], step)
        m = zoom(imgs[1], step)
        mk = np.zeros(m.shape, dtype=np.int8)
        mk[int(wz*2**(ii-1)):-int(wz*2**(ii-1)),
           int(wz*2**(ii-1)):-int(wz*2**(ii-1))] = 1

        s = np.array([0, 0], dtype=np.int32)
        for jj in range(ii):
            s = np.int_(s + shift[jj]*2**(ii-jj))

        if ii == levs-2:
            tv_lv = []
            for kk in range(2*lsw+1):
                tv_lv.append(tv_l1_pixel(f, m, 1, [kk-lsw+s[0], s[1]]))
            s[0] += (np.array(tv_lv).argmin() - lsw)

            tv_lh = []
            for kk in range(2*lsw+1):
                tv_lh.append(tv_l1_pixel(f, m, 1, [s[0], kk-lsw+s[1]]))
            s[1] += (np.array(tv_lh).argmin() - lsw)

        tem = np.ndarray([w, w], dtype=np.float32)
        for jj in range(w):
            for kk in range(w):
                tem[jj, kk] = tv_l1_pixel(f, m, mk,
                                          [jj-int(w/2)+s[0], kk-int(w/2)+s[1]])

        tv1_pxl[ii] = np.array(tem)
        tv1_pxl_id[ii] = tv1_pxl[ii].argmin()
        shift[ii, 0] = (tv1_pxl_id[ii]//w-int(w/2))
        shift[ii, 1] = (tv1_pxl_id[ii] % w-int(w/2))

    print('s before:', s)
    s = s + shift[levs-2]

    if sp_wz > (int(wz/2)-1):
        sp_wz = (int(wz/2)-1)

    idx = np.int_(shift[levs-2] + int(w/2))
    print('s after:', s)
    print('shift:', shift)
    print('idx:', idx)

    us = int(1/sp_step)
    shift[levs-1] = tv_l1_subpixel_ana(np.linspace(-1, 1, 3),
                                       np.linspace(-us, us, 2*us+1)/us,
                                       [tv1_pxl[levs-2].reshape([wz, wz])[(idx[0]-sp_wz):(idx[0]+sp_wz+1), idx[1]],
                                        tv1_pxl[levs-2].reshape([wz, wz])[idx[0], (idx[1]-sp_wz):(idx[1]+sp_wz+1)]])
    return tv1_pxl, tv1_pxl_id, shift, s+shift[levs-1]


# @msgit(wd=100, fill='-')
# def mrtv_reg(pxl_conf, sub_conf, ps, imgs=None):
#     """
#     Provide a unified interace for using multi-resolution TV-based registration
#     algorithm with different configuration options.

#     Inputs:
#     ______
#         pxl_conf: dict; multi-resolution TV minimization configuration at
#                   resolution higher than a single pixel; it includes items:
#                   'type': 'area', 'line', 'al', 'la'
#                   'levs': int
#                   'wz': int
#                   'lsw': int
#                       levs=7, wz=10, sp_wz=1, sp_step=0.5, lsw=10, ps=None, imgs=None
#         sub_conf: dict; sub-pixel TV minimization configuration; it includes
#                   items:
#                   'use': boolean; if conduct sub-pixel registration on the end
#                   'type': ['ana', 'fit']; which sub-pixel routine to use
#                   'sp_wz': int; for 'fit' option; the number of TV points to be
#                            used in fitting
#                   'sp_us': int; for 'ana' option; up-sampling factor
#         ps: optonal; None or a 2-element tuple
#         imgs: the pair of the images to be registered

#     Outputs:
#     _______
#         tv_pxl: dict; TV at each search point under levels as the keywords
#         tv_pxl_id: the id of the minimum TV in the flatted tv_pxl for each level
#         shift: ndarray; shift at each level of resolution
#         tot_shift: 2-element tuple; overall displacement between the pair of images
#     """
#     """
#     single process version of mrtv_reg2. Parallelization is implemented
#     on the upper level on which the alignments of pairs of images are
#     parallelized
#     """
#     # print(pxl_conf)
#     # print(sub_conf)
#     # print(ps)
#     if ps is not None:
#         imgs[1] = np.real(np.fft.ifftn(
#             fourier_shift(np.fft.fftn(imgs[1]), ps)))

#     levs = pxl_conf['levs']
#     w = pxl_conf['wz']
#     step = {}
#     tv_pxl = {}
#     tv_pxl_id = np.zeros(levs, dtype=np.int16)
#     shift = np.zeros([levs+1, 2], dtype=np.float32)

#     for ii in range(levs):
#         step[levs-1-ii] = 0.5**ii

#     if ((np.array(imgs[0].shape)*0.5**(levs-2) - w) <= 0).any():
#         return -1

#     if pxl_conf['type'] == 'area':
#         for ii in range(levs):
#             s = step[ii]
#             f = zoom(imgs[0], s)
#             m = zoom(imgs[1], s)
#             mk = np.zeros(m.shape, dtype=np.int8)
#             mk[int(w*2**(ii-1)):-int(w*2**(ii-1)),
#                int(w*2**(ii-1)):-int(w*2**(ii-1))] = 1

#             sh = np.array([0, 0])
#             for jj in range(ii):
#                 sh = np.int_(sh + shift[jj]*2**(ii-jj))

#             if ii == levs-1:
#                 tem = np.ndarray([2*w, 2*w], dtype=np.float32)
#                 for jj in range(2*w):
#                     for kk in range(2*w):
#                         tem[jj, kk] = tv_l1_pixel(f, m, mk,
#                                                   [jj-w+sh[0],
#                                                    kk-w+sh[1]])

#                 tv_pxl[ii] = np.array(tem)
#                 tv_pxl_id[ii] = tv_pxl[ii].argmin()
#                 shift[ii, 0] = (tv_pxl_id[ii]//(2*w)-w)
#                 shift[ii, 1] = (tv_pxl_id[ii] % (2*w)-w)
#             else:
#                 tem = np.ndarray([w, w], dtype=np.float32)
#                 for jj in range(w):
#                     for kk in range(w):
#                         tem[jj, kk] = tv_l1_pixel(f, m, mk,
#                                                   [jj-int(w/2)+sh[0],
#                                                    kk-int(w/2)+sh[1]])

#                 tv_pxl[ii] = np.array(tem)
#                 tv_pxl_id[ii] = tv_pxl[ii].argmin()
#                 shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
#                 shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))

#             # tem = np.ndarray([w, w], dtype=np.float32)
#             # for jj in range(w):
#             #     for kk in range(w):
#             #         tem[jj, kk] = tv_l1_pixel(f, m, mk,
#             #                                   [jj-int(w/2)+sh[0],
#             #                                    kk-int(w/2)+sh[1]])

#             # tv_pxl[ii] = np.array(tem)
#             # tv_pxl_id[ii] = tv_pxl[ii].argmin()
#             # shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
#             # shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
#     elif pxl_conf['type'] == 'line':
#         lsw = pxl_conf['lsw']
#         shift = np.zeros([3, 2])
#         tv_pxl_id = np.zeros(2, dtype=np.int16)

#         tv_v = []
#         mk = np.zeros(imgs[0].shape, dtype=np.int8)
#         mk[int(lsw/2):-int(lsw/2), :] = 1
#         for ii in range(lsw):
#             tv_v.append(tv_l1_pixel(imgs[0], imgs[1], mk,
#                                     [ii-int(lsw/2), 0]))
#         shift[0, 0] = (np.array(tv_v).argmin() - int(lsw/2))
#         tv_h = []
#         mk = np.zeros(imgs[0].shape, dtype=np.int8)
#         mk[:, int(lsw/2):-int(lsw/2)] = 1
#         for ii in range(lsw):
#             tv_h.append(tv_l1_pixel(imgs[0], imgs[1], mk,
#                                     [0, ii-int(lsw/2)]))
#         shift[0, 1] = (np.array(tv_h).argmin() - int(lsw/2))
#         tv_pxl[0] = np.vstack(np.array(tv_v), np.array(tv_h))
#         tv_pxl_id[0] = 0
#         sh = shift
#     elif pxl_conf['type'] == 'al':
#         lsw = pxl_conf['lsw']
#         for ii in range(levs):
#             s = step[ii]
#             f = zoom(imgs[0], s)
#             m = zoom(imgs[1], s)
#             mk = np.zeros(m.shape, dtype=np.int8)
#             mk[int(w*2**(ii-1)):-int(w*2**(ii-1)),
#                int(w*2**(ii-1)):-int(w*2**(ii-1))] = 1

#             sh = np.array([0, 0])
#             for jj in range(ii):
#                 sh = np.int_(sh + shift[jj]*2**(ii-jj))

#             if ii == levs-1:
#                 tv_v = []
#                 mk = np.zeros(imgs[0].shape, dtype=np.int8)
#                 mk[int(lsw/2):-int(lsw/2), :] = 1
#                 for kk in range(lsw):
#                     tv_v.append(tv_l1_pixel(f, m, mk,
#                                             [kk-int(lsw/2)+sh[0], sh[1]]))
#                 sh[0] += (np.array(tv_v).argmin() - int(lsw/2))

#                 tv_h = []
#                 mk = np.zeros(imgs[0].shape, dtype=np.int8)
#                 mk[:, int(lsw/2):-int(lsw/2)] = 1
#                 for kk in range(lsw):
#                     tv_h.append(tv_l1_pixel(f, m, mk,
#                                             [sh[0], kk-int(lsw/2)+sh[1]]))
#                 sh[1] += (np.array(tv_h).argmin() - int(lsw/2))

#                 tem = np.ndarray([2*w, 2*w], dtype=np.float32)
#                 for jj in range(2*w):
#                     for kk in range(2*w):
#                         tem[jj, kk] = tv_l1_pixel(f, m, mk,
#                                                   [jj-w+sh[0],
#                                                    kk-w+sh[1]])

#                 tv_pxl[ii] = np.array(tem)
#                 tv_pxl_id[ii] = tv_pxl[ii].argmin()
#                 shift[ii, 0] = (tv_pxl_id[ii]//(2*w)-w)
#                 shift[ii, 1] = (tv_pxl_id[ii] % (2*w)-w)
#             else:
#                 tem = np.ndarray([w, w], dtype=np.float32)
#                 for jj in range(w):
#                     for kk in range(w):
#                         tem[jj, kk] = tv_l1_pixel(f, m, mk,
#                                                   [jj-int(w/2)+sh[0],
#                                                    kk-int(w/2)+sh[1]])

#                 tv_pxl[ii] = np.array(tem)
#                 tv_pxl_id[ii] = tv_pxl[ii].argmin()
#                 shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
#                 shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
#         sh = sh + shift[levs-1]
#     elif pxl_conf['type'] == 'la':
#         # TODO
#         pass

#     if sub_conf['use']:
#         # idx = np.int_(shift[levs-2] + int(w/2))
#         idx = np.int_(shift[levs-1] + w)
#         if sub_conf['type'] == 'fit':
#             sw = int(sub_conf['sp_wz'])
#             shift[levs] = tv_l1_subpixel_fit(np.linspace(-sw, sw, 2*sw+1),
#                                              np.linspace(-10*sw,
#                                                          10*sw, 20*sw+1),
#                                              [tv_pxl[levs-1].reshape(2*w, 2*w)
#                                               [(idx[0]-sw):(idx[0]+sw+1), idx[1]],
#                                               tv_pxl[levs-1].reshape(2*w, 2*w)
#                                               [idx[0], (idx[1]-sw):(idx[1]+sw+1)]])
#         elif sub_conf['type'] == 'ana':
#             us = int(sub_conf['sp_us'])
#             shift[levs] = tv_l1_subpixel_ana(np.linspace(-1, 1, 3),
#                                              np.linspace(-us, us, 2*us+1)/us,
#                                              [tv_pxl[levs-1].reshape(2*w, 2*w)
#                                               [(idx[0]-1):(idx[0]+2), idx[1]],
#                                               tv_pxl[levs-1].reshape(2*w, 2*w)
#                                               [idx[0], (idx[1]-1):(idx[1]+2)]])
#     else:
#         shift[levs] = [0, 0]
#     return tv_pxl, tv_pxl_id, shift, sh+shift[levs]


def mrtv_reg(pxl_conf, sub_conf, ps=None, kernel=3, imgs=None):
    """
    Provide a unified interace for using multi-resolution TV-based registration
    algorithm with different configuration options.

    Inputs:
    ______
        pxl_conf: dict; multi-resolution TV minimization configuration at
                  resolution higher than a single pixel; it includes items:
                  'type': 'area', 'line', 'al', 'la'
                  'levs': int
                  'wz': int
                  'lsw': int
        sub_conf: dict; sub-pixel TV minimization configuration; it includes
                  items:
                  'use': boolean; if conduct sub-pixel registration on the end
                  'type': ['ana', 'fit']; which sub-pixel routine to use
                  'sp_wz': int; for 'fit' option; the number of TV points to be
                           used in fitting
                  'sp_us': int; for 'ana' option; up-sampling factor
        ps: optonal; None or a 2-element tuple
        imgs: the pair of the images to be registered

    Outputs:
    _______
        tv_pxl: dict; TV at each search point under levels as the keywords
        tv_pxl_id: the id of the minimum TV in the flatted tv_pxl for each level
        shift: ndarray; shift at each level of resolution
        tot_shift: 2-element tuple; overall displacement between the pair of images
    """
    """
    single process version of mrtv_reg2. Parallelization is implemented
    on the upper level on which the alignments of pairs of images are
    parallelized
    """
    if ps is not None:
        imgs[1] = np.real(np.fft.ifftn(
            fourier_shift(np.fft.fftn(imgs[1]), ps)))

    levs = pxl_conf['levs']
    w = pxl_conf['wz']
    step = {}
    tv_pxl = {}
    tv_pxl_id = np.zeros(levs, dtype=np.int16)
    shift = np.zeros([levs+1, 2], dtype=np.float32)

    for ii in range(levs):
        step[levs-1-ii] = 0.5**ii

    if ((np.array(imgs[0].shape)*0.5**(levs-2) - w) <= 0).any():
        return -1

    if pxl_conf['type'] == 'area':
        for ii in range(levs):
            s = step[ii]
            f = zoom(imgs[0], s)
            m = zoom(imgs[1], s)
            mk = np.zeros(m.shape, dtype=np.int8)
            mk[int(w*2**(ii-1)):-int(w*2**(ii-1)),
               int(w*2**(ii-1)):-int(w*2**(ii-1))] = 1
            mk *= binary_erosion((m!=0)).astype(np.int8)

            sh = np.array([0, 0])
            for jj in range(ii):
                sh = np.int_(sh + shift[jj]*2**(ii-jj))

            tem = np.ndarray([w, w], dtype=np.float32)
            for jj in range(w):
                for kk in range(w):
                    tem[jj, kk] = tv_l1_pixel(f, m, mk,
                                              [jj-int(w/2)+sh[0],
                                               kk-int(w/2)+sh[1]],
                                              kernel=kernel)

            tv_pxl[ii] = np.array(tem)
            tv_pxl_id[ii] = tv_pxl[ii].argmin()
            shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
            shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
            if ii == levs - 1:
                idx = np.int_(shift[levs-1] + int(w/2))
                while idx[0] == 0 or idx[0] == w-1 or idx[1] == 0 or idx[1] == w-1:
                    if idx[0] == 0:
                        sh[0] -= int(w/2)
                    elif idx[0] == w-1:
                        sh[0] += int(w/2)
                    if idx[1] == 0:
                        sh[1] -= int(w/2)
                    elif idx[1] == w-1:
                        sh[1] += int(w/2)
                    for jj in range(w):
                        for kk in range(w):
                            tem[jj, kk] = tv_l1_pixel(f, m, mk,
                                                      [jj-int(w/2)+sh[0],
                                                       kk-int(w/2)+sh[1]],
                                                      kernel=kernel)

                    tv_pxl[ii] = np.array(tem)
                    tv_pxl_id[ii] = tv_pxl[ii].argmin()
                    shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
                    shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
                    idx = np.int_(shift[levs-1] + int(w/2))
        sh = sh + shift[levs-1]
    elif pxl_conf['type'] == 'line':
        levs = 1
        lsw = pxl_conf['lsw']
        shift = np.zeros([2, 2])
        tv_pxl_id = np.zeros(2, dtype=np.int16)

        tv_v = []
        mk = np.zeros(imgs[0].shape, dtype=np.int8)
        mk[int(lsw/2):-int(lsw/2), :] = 1
        for ii in range(lsw):
            tv_v.append(tv_l1_pixel(imgs[0], imgs[1], mk,
                                    [ii-int(lsw/2), 0],
                                    kernel=kernel))
        shift[0, 0] = (np.array(tv_v).argmin() - int(lsw/2))
        tv_h = []
        mk = np.zeros(imgs[0].shape, dtype=np.int8)
        mk[:, int(lsw/2):-int(lsw/2)] = 1
        for ii in range(lsw):
            tv_h.append(tv_l1_pixel(imgs[0], imgs[1], mk,
                                    [0, ii-int(lsw/2)],
                                    kernel=kernel))
        shift[0, 1] = (np.array(tv_h).argmin() - int(lsw/2))
        tv_pxl[0] = np.stack([np.array(tv_v), np.array(tv_h)], axis=0)
        tv_pxl_id[0] = 0
        sh = shift
    elif pxl_conf['type'] == 'al':
        lsw = pxl_conf['lsw']
        for ii in range(levs):
            s = step[ii]
            f = zoom(imgs[0], s)
            m = zoom(imgs[1], s)
            mk = np.zeros(m.shape, dtype=np.int8)
            mk[int(w*2**(ii-1)):-int(w*2**(ii-1)),
               int(w*2**(ii-1)):-int(w*2**(ii-1))] = 1

            sh = np.array([0, 0])
            for jj in range(ii):
                sh = np.int_(sh + shift[jj]*2**(ii-jj))

            if ii == levs-1:
                tv_v = []
                mk = np.zeros(imgs[0].shape, dtype=np.int8)
                mk[int(lsw/2):-int(lsw/2), :] = 1
                for kk in range(lsw):
                    tv_v.append(tv_l1_pixel(f, m, mk,
                                            [kk-int(lsw/2)+sh[0], sh[1]],
                                            kernel=kernel))
                sh[0] += (np.array(tv_v).argmin() - int(lsw/2))

                tv_h = []
                mk = np.zeros(imgs[0].shape, dtype=np.int8)
                mk[:, int(lsw/2):-int(lsw/2)] = 1
                for kk in range(lsw):
                    tv_h.append(tv_l1_pixel(f, m, mk,
                                            [sh[0], kk-int(lsw/2)+sh[1]],
                                            kernel=kernel))
                sh[1] += (np.array(tv_h).argmin() - int(lsw/2))

            tem = np.ndarray([w, w], dtype=np.float32)
            for jj in range(w):
                for kk in range(w):
                    tem[jj, kk] = tv_l1_pixel(f, m, mk,
                                              [jj-int(w/2)+sh[0],
                                               kk-int(w/2)+sh[1]],
                                              kernel=kernel)

            tv_pxl[ii] = np.array(tem)
            tv_pxl_id[ii] = tv_pxl[ii].argmin()
            shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
            shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))

            if ii == levs - 1:
                idx = np.int_(shift[levs-1] + int(w/2))
                while idx[0] == 0 or idx[0] == w-1 or idx[1] == 0 or idx[1] == w-1:
                    if idx[0] == 0:
                        sh[0] -= int(w/2)
                    elif idx[0] == w-1:
                        sh[0] += int(w/2)
                    if idx[1] == 0:
                        sh[1] -= int(w/2)
                    elif idx[1] == w-1:
                        sh[1] += int(w/2)
                    for jj in range(w):
                        for kk in range(w):
                            tem[jj, kk] = tv_l1_pixel(f, m, mk,
                                                      [jj-int(w/2)+sh[0],
                                                       kk-int(w/2)+sh[1]],
                                                      kernel=kernel)

                    tv_pxl[ii] = np.array(tem)
                    tv_pxl_id[ii] = tv_pxl[ii].argmin()
                    shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
                    shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
                    idx = np.int_(shift[levs-1] + int(w/2))
        sh = sh + shift[levs-1]
    elif pxl_conf['type'] == 'la':
        # TODO
        pass

    if sub_conf['use']:
        idx = np.int_(shift[levs-1] + int(w/2))
        if sub_conf['type'] == 'fit':
            sw = int(sub_conf['sp_wz'])
            shift[levs] = tv_l1_subpixel_fit(np.linspace(-sw, sw, 2*sw+1),
                                             np.linspace(-10*sw,
                                                         10*sw, 20*sw+1),
                                             [tv_pxl[levs-1].reshape(w, w)
                                              [(idx[0]-sw):(idx[0]+sw+1), idx[1]],
                                              tv_pxl[levs-1].reshape(w, w)
                                              [idx[0], (idx[1]-sw):(idx[1]+sw+1)]])
        elif sub_conf['type'] == 'ana':
            us = int(sub_conf['sp_us'])
            shift[levs] = tv_l1_subpixel_ana(np.linspace(-1, 1, 3),
                                             np.linspace(-us, us, 2*us+1)/us,
                                             [tv_pxl[levs-1].reshape(w, w)
                                              [(idx[0]-1):(idx[0]+2), idx[1]],
                                              tv_pxl[levs-1].reshape(w, w)
                                              [idx[0], (idx[1]-1):(idx[1]+2)]])
    else:
        shift[levs] = [0, 0]
    return tv_pxl, tv_pxl_id, shift, sh+shift[levs]


def mrtv_reg_v2(pxl_conf, sub_conf, ps=None, gfk=3, gm=1, imgs=None):
    """
    Provide a unified interace for using multi-resolution TV-based registration
    algorithm with different configuration options. Using normalized TVDM on pixels

    Inputs:
    ______
        pxl_conf: dict; multi-resolution TV minimization configuration at
                  resolution higher than a single pixel; it includes items:
                  'type': 'area', 'line', 'al', 'la'
                  'levs': int
                  'wz': int
                  'lsw': int
        sub_conf: dict; sub-pixel TV minimization configuration; it includes
                  items:
                  'use': boolean; if conduct sub-pixel registration on the end
                  'type': ['ana', 'fit']; which sub-pixel routine to use
                  'sp_wz': int; for 'fit' option; the number of TV points to be
                           used in fitting
                  'sp_us': int; for 'ana' option; up-sampling factor
        ps: optonal; None or a 2-element tuple
        gm: optional; global mask
        imgs: the pair of the images to be registered

    Outputs:
    _______
        tv_pxl: dict; TV at each search point under levels as the keywords
        tv_pxl_id: the id of the minimum TV in the flatted tv_pxl for each level
        shift: ndarray; shift at each level of resolution
        tot_shift: 2-element tuple; overall displacement between the pair of images
    """
    """
    single process version of mrtv_reg2. Parallelization is implemented
    on the upper level on which the alignments of pairs of images are
    parallelized
    """
    if ps is not None:
        imgs[1] = np.real(np.fft.ifftn(
            fourier_shift(np.fft.fftn(imgs[1]), ps)))

    levs = pxl_conf['levs']
    w = pxl_conf['wz']
    step = {}
    tv_pxl = {}
    tv_pxl_id = np.zeros(levs, dtype=np.int16)
    shift = np.zeros([levs+1, 2], dtype=np.float32)

    for ii in range(levs):
        step[levs-1-ii] = 0.5**ii

    if ((np.array(imgs[0].shape)*0.5**(levs-2) - w) <= 0).any():
        return -1

    if pxl_conf['type'] == 'area':
        for ii in range(levs):
            s = step[ii]
            f = zoom(imgs[0], s)
            m = zoom(imgs[1], s)
            gmz = zoom(gm, s)
            mk = np.zeros(m.shape, dtype=np.int8)
            mk[int(w*2**(ii-1)):-int(w*2**(ii-1)),
               int(w*2**(ii-1)):-int(w*2**(ii-1))] = 1
            mk *= gmz

            sh = np.array([0, 0])
            for jj in range(ii):
                sh = np.int_(sh + shift[jj]*2**(ii-jj))

            tem = np.ndarray([w, w], dtype=np.float32)
            for jj in range(w):
                for kk in range(w):
                    tem[jj, kk] = tv_l1_pixel_v2(f, m, mk, gfk, [jj-int(w/2)+sh[0], kk-int(w/2)+sh[1]])

            tv_pxl[ii] = np.array(tem)
            tv_pxl_id[ii] = tv_pxl[ii].argmin()
            shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
            shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
            if ii == levs - 1:
                idx = np.int_(shift[levs-1] + int(w/2))
                while idx[0] == 0 or idx[0] == w-1 or idx[1] == 0 or idx[1] == w-1:
                    if idx[0] == 0:
                        sh[0] -= int(w/2)
                    elif idx[0] == w-1:
                        sh[0] += int(w/2)
                    if idx[1] == 0:
                        sh[1] -= int(w/2)
                    elif idx[1] == w-1:
                        sh[1] += int(w/2)
                    for jj in range(w):
                        for kk in range(w):
                            tem[jj, kk] = tv_l1_pixel_v2(f, m, mk, gfk, [jj-int(w/2)+sh[0], kk-int(w/2)+sh[1]])

                    tv_pxl[ii] = np.array(tem)
                    tv_pxl_id[ii] = tv_pxl[ii].argmin()
                    shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
                    shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
                    idx = np.int_(shift[levs-1] + int(w/2))
        sh = sh + shift[levs-1]
    elif pxl_conf['type'] == 'line':
        levs = 1
        lsw = pxl_conf['lsw']
        shift = np.zeros([2, 2])
        tv_pxl_id = np.zeros(2, dtype=np.int16)

        tv_v = []
        gmz = zoom(gm, s)
        mk = np.zeros(te.shape, dtype=np.int8)
        mk[int(lsw/2):-int(lsw/2), :] = 1
        mk *= gmz
        for ii in range(lsw):
            tv_v.append(tv_l1_pixel_v2(f, m, mk, gfk, [ii-int(lsw/2), 0]))
        shift[0, 0] = (np.array(tv_v).argmin() - int(lsw/2))
        tv_h = []
        mk = np.zeros(te.shape, dtype=np.int8)
        mk[:, int(lsw/2):-int(lsw/2)] = 1
        mk *= gmz
        for ii in range(lsw):
            tv_h.append(tv_l1_pixel_v2(f, m, mk, gfk, [0, ii-int(lsw/2)]))
        shift[0, 1] = (np.array(tv_h).argmin() - int(lsw/2))
        tv_pxl[0] = np.stack([np.array(tv_v), np.array(tv_h)], axis=0)
        tv_pxl_id[0] = 0
        sh = shift
    elif pxl_conf['type'] == 'al':
        lsw = pxl_conf['lsw']
        for ii in range(levs):
            s = step[ii]
            f = zoom(te, s)
            m = zoom(se, s)
            gmz = zoom(gm, s)
            mk = np.zeros(m.shape, dtype=np.int8)
            mk[int(w*2**(ii-1)):-int(w*2**(ii-1)),
               int(w*2**(ii-1)):-int(w*2**(ii-1))] = 1
            mk *= gmz

            sh = np.array([0, 0])
            for jj in range(ii):
                sh = np.int_(sh + shift[jj]*2**(ii-jj))

            if ii == levs-1:
                tv_v = []
                mk = np.zeros(te.shape, dtype=np.int8)
                mk[int(lsw/2):-int(lsw/2), :] = 1
                mk *= gmz
                for kk in range(lsw):
                    tv_v.append(tv_l1_pixel_v2(f, m, mk, gfk, [kk-int(lsw/2)+sh[0], sh[1]]))
                sh[0] += (np.array(tv_v).argmin() - int(lsw/2))

                tv_h = []
                mk = np.zeros(te.shape, dtype=np.int8)
                mk[:, int(lsw/2):-int(lsw/2)] = 1
                m *= gmz
                for kk in range(lsw):
                    tv_h.append(tv_l1_pixel_v2(f, m, mk, gfk, [sh[0], kk-int(lsw/2)+sh[1]]))
                sh[1] += (np.array(tv_h).argmin() - int(lsw/2))

            tem = np.ndarray([w, w], dtype=np.float32)
            for jj in range(w):
                for kk in range(w):
                    tem[jj, kk] = tv_l1_pixel_v2(f, m, mk, gfk, [jj-int(w/2)+sh[0], kk-int(w/2)+sh[1]])

            tv_pxl[ii] = np.array(tem)
            tv_pxl_id[ii] = tv_pxl[ii].argmin()
            shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
            shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))

            if ii == levs - 1:
                idx = np.int_(shift[levs-1] + int(w/2))
                while idx[0] == 0 or idx[0] == w-1 or idx[1] == 0 or idx[1] == w-1:
                    if idx[0] == 0:
                        sh[0] -= int(w/2)
                    elif idx[0] == w-1:
                        sh[0] += int(w/2)
                    if idx[1] == 0:
                        sh[1] -= int(w/2)
                    elif idx[1] == w-1:
                        sh[1] += int(w/2)
                    for jj in range(w):
                        for kk in range(w):
                            tem[jj, kk] = tv_l1_pixel_v2(f, m, mk, gfk, [jj-int(w/2)+sh[0], kk-int(w/2)+sh[1]])

                    tv_pxl[ii] = np.array(tem)
                    tv_pxl_id[ii] = tv_pxl[ii].argmin()
                    shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
                    shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
                    idx = np.int_(shift[levs-1] + int(w/2))
        sh = sh + shift[levs-1]
    elif pxl_conf['type'] == 'la':
        # TODO
        pass

    if sub_conf['use']:
        idx = np.int_(shift[levs-1] + int(w/2))
        if sub_conf['type'] == 'fit':
            sw = int(sub_conf['sp_wz'])
            shift[levs] = tv_l1_subpixel_fit(np.linspace(-sw, sw, 2*sw+1),
                                             np.linspace(-10*sw, 10*sw, 20*sw+1),
                                             [tv_pxl[levs-1].reshape(w, w)
                                              [(idx[0]-sw):(idx[0]+sw+1), idx[1]],
                                              tv_pxl[levs-1].reshape(w, w)
                                              [idx[0], (idx[1]-sw):(idx[1]+sw+1)]])
        elif sub_conf['type'] == 'ana':
            us = int(sub_conf['sp_us'])
            shift[levs] = tv_l1_subpixel_ana(np.linspace(-1, 1, 3),
                                             np.linspace(-us, us, 2*us+1)/us,
                                             [tv_pxl[levs-1].reshape(w, w)
                                              [(idx[0]-1):(idx[0]+2), idx[1]],
                                              tv_pxl[levs-1].reshape(w, w)
                                              [idx[0], (idx[1]-1):(idx[1]+2)]])
    else:
        shift[levs] = [0, 0]
    return tv_pxl, tv_pxl_id, shift, -np.array(sh+shift[levs])


def mrtv_reg_v3(pxl_conf, sub_conf, ps=None, kernel=3, imgs=None):
    """
    Provide a unified interace for using multi-resolution TV-based registration
    algorithm with different configuration options.

    Inputs:
    ______
        pxl_conf: dict; multi-resolution TV minimization configuration at
                  resolution higher than a single pixel; it includes items:
                  'type': 'area', 'line', 'al', 'la'
                  'levs': int
                  'wz': int
                  'lsw': int
        sub_conf: dict; sub-pixel TV minimization configuration; it includes
                  items:
                  'use': boolean; if conduct sub-pixel registration on the end
                  'type': ['ana', 'fit']; which sub-pixel routine to use
                  'sp_wz': int; for 'fit' option; the number of TV points to be
                           used in fitting
                  'sp_us': int; for 'ana' option; up-sampling factor
        ps: optonal; None or a 2-element tuple
        imgs: the pair of the images to be registered

    Outputs:
    _______
        tv_pxl: dict; TV at each search point under levels as the keywords
        tv_pxl_id: the id of the minimum TV in the flatted tv_pxl for each level
        shift: ndarray; shift at each level of resolution
        tot_shift: 2-element tuple; overall displacement between the pair of images
    """
    """
    single process version of mrtv_reg2. Parallelization is implemented
    on the upper level on which the alignments of pairs of images are
    parallelized
    """
    if ps is not None:
        imgs[1] = np.real(np.fft.ifftn(
            fourier_shift(np.fft.fftn(imgs[1]), ps)))

    levs = pxl_conf['levs']
    w = pxl_conf['wz']
    step = {}
    tv_pxl = {}
    tv_pxl_id = np.zeros(levs, dtype=np.int16)
    shift = np.zeros([levs+1, 2], dtype=np.float32)

    for ii in range(levs):
        step[levs-1-ii] = 0.5**ii

    if ((np.array(imgs[0].shape)*0.5**(levs-2) - w) <= 0).any():
        return -1

    if pxl_conf['type'] == 'area':
        for ii in range(levs):
            s = step[ii]
            f = gf(zoom(imgs[0], s), kernel)
            m = gf(zoom(imgs[1], s), kernel)
            mk = np.zeros(m.shape, dtype=np.int8)
            mk[int(w*2**(ii-1)):-int(w*2**(ii-1)),
               int(w*2**(ii-1)):-int(w*2**(ii-1))] = 1

            sh = np.array([0, 0])
            for jj in range(ii):
                sh = np.int_(sh + shift[jj]*2**(ii-jj))

            tem = np.ndarray([w, w], dtype=np.float32)
            for jj in range(w):
                for kk in range(w):
                    tem[jj, kk] = tv_l1_pixel_v3(f, m, mk, [jj-int(w/2)+sh[0], kk-int(w/2)+sh[1]])

            tv_pxl[ii] = np.array(tem)
            tv_pxl_id[ii] = tv_pxl[ii].argmin()
            shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
            shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
            if ii == levs - 1:
                idx = np.int_(shift[levs-1] + int(w/2))
                while idx[0] == 0 or idx[0] == w-1 or idx[1] == 0 or idx[1] == w-1:
                    if idx[0] == 0:
                        sh[0] -= int(w/2)
                    elif idx[0] == w-1:
                        sh[0] += int(w/2)
                    if idx[1] == 0:
                        sh[1] -= int(w/2)
                    elif idx[1] == w-1:
                        sh[1] += int(w/2)
                    for jj in range(w):
                        for kk in range(w):
                            tem[jj, kk] = tv_l1_pixel_v3(f, m, mk, [jj-int(w/2)+sh[0], kk-int(w/2)+sh[1]])

                    tv_pxl[ii] = np.array(tem)
                    tv_pxl_id[ii] = tv_pxl[ii].argmin()
                    shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
                    shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
                    idx = np.int_(shift[levs-1] + int(w/2))
        sh = sh + shift[levs-1]
    elif pxl_conf['type'] == 'line':
        f = gf(imgs[0], kernel)
        m = gf(imgs[1], kernel)
        levs = 1
        lsw = pxl_conf['lsw']
        shift = np.zeros([2, 2])
        tv_pxl_id = np.zeros(2, dtype=np.int16)

        tv_v = []
        mk = np.zeros(imgs[0].shape, dtype=np.int8)
        mk[int(lsw/2):-int(lsw/2), :] = 1
        for ii in range(lsw):
            tv_v.append(tv_l1_pixel_v3(f, m, mk, [ii-int(lsw/2), 0]))
        shift[0, 0] = (np.array(tv_v).argmin() - int(lsw/2))
        tv_h = []
        mk = np.zeros(imgs[0].shape, dtype=np.int8)
        mk[:, int(lsw/2):-int(lsw/2)] = 1
        for ii in range(lsw):
            tv_h.append(tv_l1_pixel_v3(imgs[0], imgs[1], mk, [0, ii-int(lsw/2)]))
        shift[0, 1] = (np.array(tv_h).argmin() - int(lsw/2))
        tv_pxl[0] = np.stack([np.array(tv_v), np.array(tv_h)], axis=0)
        tv_pxl_id[0] = 0
        sh = shift
    elif pxl_conf['type'] == 'al':
        lsw = pxl_conf['lsw']
        for ii in range(levs):
            s = step[ii]
            f = gf(zoom(imgs[0], s), kernel)
            m = gf(zoom(imgs[1], s), kernel)
            mk = np.zeros(m.shape, dtype=np.int8)
            mk[int(w*2**(ii-1)):-int(w*2**(ii-1)),
               int(w*2**(ii-1)):-int(w*2**(ii-1))] = 1

            sh = np.array([0, 0])
            for jj in range(ii):
                sh = np.int_(sh + shift[jj]*2**(ii-jj))

            if ii == levs-1:
                tv_v = []
                mk = np.zeros(imgs[0].shape, dtype=np.int8)
                mk[int(lsw/2):-int(lsw/2), :] = 1
                for kk in range(lsw):
                    tv_v.append(tv_l1_pixel_v3(f, m, mk, [kk-int(lsw/2)+sh[0], sh[1]]))
                sh[0] += (np.array(tv_v).argmin() - int(lsw/2))

                tv_h = []
                mk = np.zeros(imgs[0].shape, dtype=np.int8)
                mk[:, int(lsw/2):-int(lsw/2)] = 1
                for kk in range(lsw):
                    tv_h.append(tv_l1_pixel_v3(f, m, mk, [sh[0], kk-int(lsw/2)+sh[1]]))
                sh[1] += (np.array(tv_h).argmin() - int(lsw/2))

            tem = np.ndarray([w, w], dtype=np.float32)
            for jj in range(w):
                for kk in range(w):
                    tem[jj, kk] = tv_l1_pixel_v3(f, m, mk, [jj-int(w/2)+sh[0], kk-int(w/2)+sh[1]])

            tv_pxl[ii] = np.array(tem)
            tv_pxl_id[ii] = tv_pxl[ii].argmin()
            shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
            shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))

            if ii == levs - 1:
                idx = np.int_(shift[levs-1] + int(w/2))
                while idx[0] == 0 or idx[0] == w-1 or idx[1] == 0 or idx[1] == w-1:
                    if idx[0] == 0:
                        sh[0] -= int(w/2)
                    elif idx[0] == w-1:
                        sh[0] += int(w/2)
                    if idx[1] == 0:
                        sh[1] -= int(w/2)
                    elif idx[1] == w-1:
                        sh[1] += int(w/2)
                    for jj in range(w):
                        for kk in range(w):
                            tem[jj, kk] = tv_l1_pixel_v3(f, m, mk, [jj-int(w/2)+sh[0],
                                                       kk-int(w/2)+sh[1]])

                    tv_pxl[ii] = np.array(tem)
                    tv_pxl_id[ii] = tv_pxl[ii].argmin()
                    shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
                    shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
                    idx = np.int_(shift[levs-1] + int(w/2))
        sh = sh + shift[levs-1]
    elif pxl_conf['type'] == 'la':
        # TODO
        pass

    if sub_conf['use']:
        idx = np.int_(shift[levs-1] + int(w/2))
        if sub_conf['type'] == 'fit':
            sw = int(sub_conf['sp_wz'])
            shift[levs] = tv_l1_subpixel_fit(np.linspace(-sw, sw, 2*sw+1),
                                             np.linspace(-10*sw,
                                                         10*sw, 20*sw+1),
                                             [tv_pxl[levs-1].reshape(w, w)
                                              [(idx[0]-sw):(idx[0]+sw+1), idx[1]],
                                              tv_pxl[levs-1].reshape(w, w)
                                              [idx[0], (idx[1]-sw):(idx[1]+sw+1)]])
        elif sub_conf['type'] == 'ana':
            us = int(sub_conf['sp_us'])
            shift[levs] = tv_l1_subpixel_ana(np.linspace(-1, 1, 3),
                                             np.linspace(-us, us, 2*us+1)/us,
                                             [tv_pxl[levs-1].reshape(w, w)
                                              [(idx[0]-1):(idx[0]+2), idx[1]],
                                              tv_pxl[levs-1].reshape(w, w)
                                              [idx[0], (idx[1]-1):(idx[1]+2)]])
    else:
        shift[levs] = [0, 0]
    return tv_pxl, tv_pxl_id, shift, sh+shift[levs]


def mrtv_reg_v4(pxl_conf, sub_conf, ps=None, gfk=3, gm=1, imgs=None, filt=True, norm=True):
    """
    Provide a unified interace for using multi-resolution TV-based registration
    algorithm with different configuration options.

    Inputs:
    ______
        pxl_conf: dict; multi-resolution TV minimization configuration at
                  resolution higher than a single pixel; it includes items:
                  'type': 'area', 'line', 'al', 'la'
                  'levs': int
                  'wz': int
                  'lsw': int
        sub_conf: dict; sub-pixel TV minimization configuration; it includes
                  items:
                  'use': boolean; if conduct sub-pixel registration on the end
                  'type': ['ana', 'fit']; which sub-pixel routine to use
                  'sp_wz': int; for 'fit' option; the number of TV points to be
                           used in fitting
                  'sp_us': int; for 'ana' option; up-sampling factor
        ps: optonal; None or a 2-element tuple
        imgs: the pair of the images to be registered

    Outputs:
    _______
        tv_pxl: dict; TV at each search point under levels as the keywords
        tv_pxl_id: the id of the minimum TV in the flatted tv_pxl for each level
        shift: ndarray; shift at each level of resolution
        tot_shift: 2-element tuple; overall displacement between the pair of images
    """
    """
    single process version of mrtv_reg2. Parallelization is implemented
    on the upper level on which the alignments of pairs of images are
    parallelized
    """
    if ps is not None:
        imgs[1] = np.real(np.fft.ifftn(
            fourier_shift(np.fft.fftn(imgs[1]), ps)))

    levs = pxl_conf['levs']
    w = pxl_conf['wz']
    step = {}
    tv_pxl = {}
    tv_pxl_id = np.zeros(levs, dtype=np.int16)
    shift = np.zeros([levs+1, 2], dtype=np.float32)

    for ii in range(levs):
        step[levs-1-ii] = 0.5**ii

    if ((np.array(imgs[0].shape)*0.5**(levs-2) - w) <= 0).any():
        return -1

    if not isinstance(gm, np.ndarray):
        gm = np.ndarray(imgs[0].shape, dtype=np.int8)

    if pxl_conf['type'] == 'area':
        for ii in range(levs):
            s = step[ii]
            f = zoom(imgs[0], s)
            m = zoom(imgs[1], s)
            gmz = zoom(gm, s)
            mk = np.zeros(m.shape, dtype=np.int8)
            mk[int(w*2**(ii-1)):-int(w*2**(ii-1)),
               int(w*2**(ii-1)):-int(w*2**(ii-1))] = 1
            mk *= gmz

            sh = np.array([0, 0])
            for jj in range(ii):
                sh = np.int_(sh + shift[jj]*2**(ii-jj))

            tem = np.ndarray([w, w], dtype=np.float32)
            for jj in range(w):
                for kk in range(w):
                    tem[jj, kk] = tv_l1_pixel_v4(f, m, mk,
                                                 [jj-int(w/2)+sh[0],
                                                 kk-int(w/2)+sh[1]],
                                                 kernel=gfk, filt=filt,
                                                 norm=norm)

            tv_pxl[ii] = np.array(tem)
            tv_pxl_id[ii] = tv_pxl[ii].argmin()
            shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
            shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
            if ii == levs - 1:
                idx = np.int_(shift[levs-1] + int(w/2))
                while idx[0] == 0 or idx[0] == w-1 or idx[1] == 0 or idx[1] == w-1:
                    if idx[0] == 0:
                        sh[0] -= int(w/2)
                    elif idx[0] == w-1:
                        sh[0] += int(w/2)
                    if idx[1] == 0:
                        sh[1] -= int(w/2)
                    elif idx[1] == w-1:
                        sh[1] += int(w/2)
                    for jj in range(w):
                        for kk in range(w):
                            tem[jj, kk] = tv_l1_pixel_v4(f, m, mk,
                                                         [jj-int(w/2)+sh[0],
                                                          kk-int(w/2)+sh[1]],
                                                         kernel=gfk, filt=filt,
                                                         norm=norm)

                    tv_pxl[ii] = np.array(tem)
                    tv_pxl_id[ii] = tv_pxl[ii].argmin()
                    shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
                    shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
                    idx = np.int_(shift[levs-1] + int(w/2))
        sh = sh + shift[levs-1]
    elif pxl_conf['type'] == 'line':
        levs = 1
        lsw = pxl_conf['lsw']
        shift = np.zeros([2, 2])
        tv_pxl_id = np.zeros(2, dtype=np.int16)

        tv_v = []
        mk = np.zeros(imgs[0].shape, dtype=np.int8)
        mk[int(lsw/2):-int(lsw/2), :] = 1
        mk *= gm
        for ii in range(lsw):
            tv_v.append(tv_l1_pixel_v4(imgs[0], imgs[1], mk,
                                       [ii-int(lsw/2), 0],
                                       kernel=gfk, filt=filt,
                                       norm=norm))
        shift[0, 0] = (np.array(tv_v).argmin() - int(lsw/2))
        tv_h = []
        mk = np.zeros(imgs[0].shape, dtype=np.int8)
        mk[:, int(lsw/2):-int(lsw/2)] = 1
        mk *= gm
        for ii in range(lsw):
            tv_h.append(tv_l1_pixel_v4(imgs[0], imgs[1], mk,
                                       [0, ii-int(lsw/2)],
                                       kernel=gfk, filt=filt,
                                       norm=norm))
        shift[0, 1] = (np.array(tv_h).argmin() - int(lsw/2))
        tv_pxl[0] = np.stack([np.array(tv_v), np.array(tv_h)], axis=0)
        tv_pxl_id[0] = 0
        sh = shift
    elif pxl_conf['type'] == 'al':
        lsw = pxl_conf['lsw']
        for ii in range(levs):
            s = step[ii]
            f = zoom(imgs[0], s)
            m = zoom(imgs[1], s)
            gmz = zoom(gm, s)

            sh = np.array([0, 0])
            for jj in range(ii):
                sh = np.int_(sh + shift[jj]*2**(ii-jj))

            if ii == levs-1:
                tv_v = []
                mk = np.zeros(imgs[0].shape, dtype=np.int8)
                mk[int(lsw/2):-int(lsw/2), :] = 1
                mk *= gmz
                for kk in range(lsw):
                    tv_v.append(tv_l1_pixel_v4(f, m, mk,
                                               [kk-int(lsw/2)+sh[0], sh[1]],
                                               kernel=gfk, filt=filt,
                                               norm=norm))
                sh[0] += (np.array(tv_v).argmin() - int(lsw/2))

                tv_h = []
                mk = np.zeros(imgs[0].shape, dtype=np.int8)
                mk[:, int(lsw/2):-int(lsw/2)] = 1
                mk *= gmz
                for kk in range(lsw):
                    tv_h.append(tv_l1_pixel_v4(f, m, mk,
                                               [sh[0], kk-int(lsw/2)+sh[1]],
                                               kernel=gfk, filt=filt,
                                               norm=norm))
                sh[1] += (np.array(tv_h).argmin() - int(lsw/2))

            tem = np.ndarray([w, w], dtype=np.float32)
            mk = np.zeros(m.shape, dtype=np.int8)
            mk[int(w * 2 ** (ii - 1)):-int(w * 2 ** (ii - 1)),
            int(w * 2 ** (ii - 1)):-int(w * 2 ** (ii - 1))] = 1
            mk *= gmz
            for jj in range(w):
                for kk in range(w):
                    tem[jj, kk] = tv_l1_pixel_v4(f, m, mk,
                                                 [jj-int(w/2)+sh[0],
                                                  kk-int(w/2)+sh[1]],
                                                 kernel=gfk, filt=filt,
                                                 norm=norm)

            tv_pxl[ii] = np.array(tem)
            tv_pxl_id[ii] = tv_pxl[ii].argmin()
            shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
            shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))

            if ii == levs - 1:
                idx = np.int_(shift[levs-1] + int(w/2))
                while idx[0] == 0 or idx[0] == w-1 or idx[1] == 0 or idx[1] == w-1:
                    if idx[0] == 0:
                        sh[0] -= int(w/2)
                    elif idx[0] == w-1:
                        sh[0] += int(w/2)
                    if idx[1] == 0:
                        sh[1] -= int(w/2)
                    elif idx[1] == w-1:
                        sh[1] += int(w/2)
                    for jj in range(w):
                        for kk in range(w):
                            tem[jj, kk] = tv_l1_pixel_v4(f, m, mk,
                                                         [jj-int(w/2)+sh[0],
                                                          kk-int(w/2)+sh[1]],
                                                         kernel=gfk, filt=filt,
                                                         norm=norm)

                    tv_pxl[ii] = np.array(tem)
                    tv_pxl_id[ii] = tv_pxl[ii].argmin()
                    shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
                    shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
                    idx = np.int_(shift[levs-1] + int(w/2))
        sh = sh + shift[levs-1]
    elif pxl_conf['type'] == 'la':
        # TODO
        pass

    if sub_conf['use']:
        idx = np.int_(shift[levs-1] + int(w/2))
        if sub_conf['type'] == 'fit':
            sw = int(sub_conf['sp_wz'])
            shift[levs] = tv_l1_subpixel_fit(np.linspace(-sw, sw, 2*sw+1),
                                             np.linspace(-10*sw,
                                                         10*sw, 20*sw+1),
                                             [tv_pxl[levs-1].reshape(w, w)
                                              [(idx[0]-sw):(idx[0]+sw+1), idx[1]],
                                              tv_pxl[levs-1].reshape(w, w)
                                              [idx[0], (idx[1]-sw):(idx[1]+sw+1)]])
        elif sub_conf['type'] == 'ana':
            us = int(sub_conf['sp_us'])
            shift[levs] = tv_l1_subpixel_ana(np.linspace(-1, 1, 3),
                                             np.linspace(-us, us, 2*us+1)/us,
                                             [tv_pxl[levs-1].reshape(w, w)
                                              [(idx[0]-1):(idx[0]+2), idx[1]],
                                              tv_pxl[levs-1].reshape(w, w)
                                              [idx[0], (idx[1]-1):(idx[1]+2)]])
    else:
        shift[levs] = [0, 0]
    return tv_pxl, tv_pxl_id, shift, sh+shift[levs]


def mrtv_mpc_combo_reg(fixed_img, img, us=100, reference_mask=None,
                       overlap_ratio=0.3, levs=4, wz=10, sp_wz=20,
                       sp_step=0.2):
    shift = np.zeros([levs+1, 2])
    if reference_mask is not None:
        shift[0] = phase_cross_correlation(fixed_img, img, upsample_factor=us,
                                           reference_mask=reference_mask,
                                           overlap_ratio=overlap_ratio)
    else:
        shift[0], _, _ = phase_cross_correlation(fixed_img, img,
                                                 upsample_factor=us,
                                                 overlap_ratio=overlap_ratio)

    tv1_pxl, tv1_pxl_id, shift[1:], ss = mrtv_reg2(fixed_img, img, levs=levs,
                                                   wz=wz, sp_wz=sp_wz,
                                                   sp_step=sp_step, ps=shift[0])

    print(time.asctime())
    return tv1_pxl, tv1_pxl_id, shift, shift[0]+ss


@msgit(wd=100, fill='-')
def mrtv_ls_combo_reg(ls_w=100, levs=2, wz=10, sp_wz=8, sp_step=0.5, kernel=3, imgs=None):
    """
    line search combined with two-level multi-resolution search
    """
    shift = np.zeros([3, 2])
    tv1_pxl_id = np.zeros(2, dtype=np.int16)
    tv1_pxl = {}
    tv_v = []
    for ii in range(ls_w):
        tv_v.append(tv_l1_pixel(imgs[0], imgs[1], 1, [2*ii-ls_w, 0],
                                kernel=kernel))
    shift[0, 0] = 2*(np.array(tv_v).argmin() - int(ls_w/2))
    # shift[0, 0] = (np.array(tv_v).argmin() - int(ls_w/2))

    tv_h = []
    for ii in range(ls_w):
        tv_h.append(tv_l1_pixel(imgs[0], imgs[1], 1, [0, 2*ii-ls_w],
                                kernel=kernel))
    shift[0, 1] = 2*(np.array(tv_h).argmin() - int(ls_w/2))
    # shift[0, 1] = (np.array(tv_h).argmin() - int(ls_w/2))

    tv1_pxl, tv1_pxl_id, shift[1:], ss = mrtv_reg3(levs=levs, wz=wz, sp_wz=sp_wz,
                                                   sp_step=sp_step, imgs=imgs,
                                                   ps=shift[0])

    print(time.asctime())
    return tv1_pxl, tv1_pxl_id, shift, shift[0]+ss


@msgit(wd=100, fill='-')
def mrtv_ls_combo_reg2(ls_w=100, levs=2, wz=10, sp_wz=8, sp_step=0.5, kernel=3, imgs=None):
    """
    line search combined with two-level multi-resolution search
    """
    shift = np.zeros([3, 2])
    tv1_pxl_id = np.zeros(2, dtype=np.int16)
    tv1_pxl = {}
    tv_v = []
    for ii in range(ls_w):
        tv_v.append(tv_l1_pixel(imgs[0], imgs[1], 1, [ii-int(ls_w/2), 0],
                                kernel=kernel))
    shift[0, 0] = (np.array(tv_v).argmin() - int(ls_w/2))

    tv_h = []
    for ii in range(ls_w):
        tv_h.append(tv_l1_pixel(imgs[0], imgs[1], 1, [0, ii-int(ls_w/2)],
                                kernel=kernel))
    shift[0, 1] = (np.array(tv_h).argmin() - int(ls_w/2))

    tv1_pxl, tv1_pxl_id, shift[1:], ss = mrtv_reg4(levs=levs, wz=wz, sp_wz=sp_wz,
                                                   sp_step=sp_step, imgs=imgs,
                                                   ps=shift[0])

    print(time.asctime())
    return tv1_pxl, tv1_pxl_id, shift, shift[0]+ss


def shift_img(img_shift):
    img = img_shift[0]
    shift = img_shift[1]
    return np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)))


def sr_transform_img(sr, data):
    return sr.transform(data[0], tmat=data[1])


def mp_shift_img(imgs, shifts, stype='trans', sr=None, axis=0):
    if stype.upper() == 'SR':
        if axis == 0:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(sr_transform_img, sr),
                               [[imgs[ii, ...], shifts[ii]] for ii
                                in range(imgs.shape[0])])
            pool.close()
            pool.join()
            return np.array(rlt).astype(np.float32)
        elif axis == 1:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(sr_transform_img, sr),
                               [[imgs[:, ii, :], shifts[ii]] for ii
                                in range(imgs.shape[1])])
            pool.close()
            pool.join()
            return np.swapaxes(np.array(rlt).astype(np.float32), 0, 1)
        elif axis == 2:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(sr_transform_img, sr),
                               [[imgs[:, :, ii], shifts[ii]] for ii
                                in range(imgs.shape[2])])
            pool.close()
            pool.join()
            return np.swapaxes(np.array(rlt).astype(np.float32), 0, 2)
    else:
        if axis == 0:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(shift_img,
                               [[imgs[ii, ...], shifts[ii]] for ii
                                in range(imgs.shape[0])])
            pool.close()
            pool.join()
            return np.array(rlt).astype(np.float32)
        elif axis == 1:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(shift_img,
                               [[imgs[:, ii, :], shifts[ii]] for ii
                                in range(imgs.shape[1])])
            pool.close()
            pool.join()
            return np.swapaxes(np.array(rlt).astype(np.float32), 0, 1)
        elif axis == 2:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(shift_img,
                               [[imgs[:, :, ii], shifts[ii]] for ii
                                in range(imgs.shape[2])])
            pool.close()
            pool.join()
            return np.swapaxes(np.array(rlt).astype(np.float32), 0, 2)


def pc_reg(upsample_factor, imgs):
    shift, _, _ = phase_cross_correlation(
        imgs[0], imgs[1], upsample_factor=upsample_factor)
    return shift


def mpc_reg(reference_mask, overlap_ratio, imgs):
    shift = phase_cross_correlation(imgs[0], imgs[1], reference_mask=reference_mask,
                                    overlap_ratio=overlap_ratio)
    return shift


def sr_reg(sr, imgs):
    shift = sr.register(imgs[0], imgs[1])
    return shift


def reg_stack(imgs, method, *args, axis=0, filter=None, **kwargs):
    shift = []
    if method.upper() == 'MRTV_REG':
        if filter is None:
            if axis == 0:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[imgs[ii], imgs[ii+1]] for ii
                                    in range(imgs.shape[0]-1)])
            elif axis == 1:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[imgs[:, ii, :], imgs[:, ii+1, :]] for ii
                                    in range(imgs.shape[1]-1)])
            elif axis == 2:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[imgs[:, :, ii], imgs[:, :, ii+1]] for ii
                                    in range(imgs.shape[2]-1)])
            pool.close()
            pool.join()
            shift = np.vstack(np.array(rlt, dtype=object)[:, -1])
        else:
            flt_func = filter['func']
            flt_parm = filter['params']
            if axis == 0:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[flt_func(imgs[ii], *flt_parm), flt_func(imgs[ii+1], *flt_parm)] for ii
                                    in range(imgs.shape[0]-1)])
            elif axis == 1:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[flt_parm(imgs[:, ii, :], *flt_parm), flt_func(imgs[:, ii+1, :], *flt_parm)] for ii
                                    in range(imgs.shape[1]-1)])
            elif axis == 2:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[flt_parm(imgs[:, :, ii], *flt_parm), flt_func(imgs[:, :, ii+1], *flt_parm)] for ii
                                    in range(imgs.shape[2]-1)])
            pool.close()
            pool.join()
            shift = np.vstack(np.array(rlt, dtype=object)[:, -1])
    elif method.upper() == 'PC':
        if axis == 0:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(pc_reg, *args, **kwargs),
                               [[imgs[ii], imgs[ii+1]] for ii
                                in range(imgs.shape[0]-1)])
        elif axis == 1:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(pc_reg, *args, **kwargs),
                               [[imgs[:, ii, :], imgs[:, ii+1, :]] for ii
                                in range(imgs.shape[1]-1)])
        elif axis == 2:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(pc_reg, *args, **kwargs),
                               [[imgs[:, :, ii], imgs[:, :, ii+1]] for ii
                                in range(imgs.shape[2]-1)])
        pool.close()
        pool.join()
        shift = np.array(rlt)
    elif method.upper() == 'MPC':
        if axis == 0:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(mpc_reg, *args, **kwargs),
                               [[imgs[ii], imgs[ii+1]] for ii
                                in range(imgs.shape[0]-1)])
        elif axis == 1:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(mpc_reg, *args, **kwargs),
                               [[imgs[:, ii, :], imgs[:, ii+1, :]] for ii
                                in range(imgs.shape[1]-1)])
        elif axis == 2:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(mpc_reg, *args, **kwargs),
                               [[imgs[:, :, ii], imgs[:, :, ii+1]] for ii
                                in range(imgs.shape[2]-1)])
        pool.close()
        pool.join()
        shift = np.array(rlt)
    elif method.upper() == 'SR':
        if axis == 0:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(sr_reg, *args, **kwargs),
                               [[imgs[ii], imgs[ii+1]] for ii
                                in range(imgs.shape[0]-1)])
        elif axis == 1:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(sr_reg, *args, **kwargs),
                               [[imgs[:, ii, :], imgs[:, ii+1, :]] for ii
                                in range(imgs.shape[1]-1)])
        elif axis == 2:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(sr_reg, *args, **kwargs),
                               [[imgs[:, :, ii], imgs[:, :, ii+1]] for ii
                                in range(imgs.shape[2]-1)])
        pool.close()
        pool.join()
        shift = np.array(rlt)
    return shift


def reg_stacks(refs, movs, method, *args, axis=0, filter=None, **kwargs):
    shift = []
    if method.upper() == 'MRTV_REG':
        if filter is None:
            if axis == 0:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[refs[ii], movs[ii]] for ii
                                    in range(refs.shape[0])])
            elif axis == 1:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[refs[:, ii, :], movs[:, ii, :]] for ii
                                    in range(refs.shape[1])])
            elif axis == 2:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[refs[:, :, ii], movs[:, :, ii]] for ii
                                    in range(refs.shape[2])])
        else:
            flt_func = filter['func']
            flt_parm = filter['params']
            if axis == 0:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[flt_func(refs[ii], *flt_parm), flt_func(movs[ii], *flt_parm)] for ii
                                    in range(refs.shape[0])])
            elif axis == 1:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[flt_func(refs[:, ii, :], *flt_parm), flt_func(movs[:, ii, :], *flt_parm)] for ii
                                    in range(refs.shape[1])])
            elif axis == 2:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[flt_func(refs[:, :, ii], *flt_parm), flt_func(movs[:, :, ii], *flt_parm)] for ii
                                    in range(refs.shape[2])])
        pool.close()
        pool.join()
        shift = np.vstack(np.array(rlt, dtype=object)[:, -1])
    elif method.upper() == 'SR':
        if axis == 0:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(sr_reg, *args, **kwargs),
                               [[refs[ii], movs[ii]] for ii
                                in range(refs.shape[0])])
        elif axis == 1:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(sr_reg, *args, **kwargs),
                               [[refs[:, ii, :], movs[:, ii, :]] for ii
                                in range(refs.shape[1])])
        elif axis == 2:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(sr_reg, *args, **kwargs),
                               [[refs[:, :, ii], movs[:, :, ii]] for ii
                                in range(refs.shape[2])])
        pool.close()
        pool.join()
        shift = np.array(rlt)
    elif method.upper() == 'PC':
        if axis == 0:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(pc_reg, *args, **kwargs),
                               [[refs[ii], movs[ii]] for ii
                                in range(refs.shape[0])])
        elif axis == 1:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(pc_reg, *args, **kwargs),
                               [[refs[:, ii, :], movs[:, ii, :]] for ii
                                in range(refs.shape[1])])
        elif axis == 2:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(pc_reg, *args, **kwargs),
                               [[refs[:, :, ii], movs[:, :, ii]] for ii
                                in range(refs.shape[2])])
        pool.close()
        pool.join()
        shift = np.array(rlt)
    elif method.upper() == 'MPC':
        if axis == 0:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(mpc_reg, *args, **kwargs),
                               [[refs[ii], movs[ii]] for ii
                                in range(refs.shape[0])])
        elif axis == 1:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(mpc_reg, *args, **kwargs),
                               [[refs[:, ii, :], movs[:, ii, :]] for ii
                                in range(refs.shape[1])])
        elif axis == 2:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(mpc_reg, *args, **kwargs),
                               [[refs[:, :, ii], movs[:, :, ii]] for ii
                                in range(refs.shape[2])])
        pool.close()
        pool.join()
        shift = np.array(rlt)
    return shift


