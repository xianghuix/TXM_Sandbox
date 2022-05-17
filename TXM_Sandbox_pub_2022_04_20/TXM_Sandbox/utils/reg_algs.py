#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from functools import partial
import numpy as np
from scipy.ndimage import (zoom, shift,
                           binary_erosion, fourier_shift,
                           gaussian_filter as gf)
from skimage.registration import phase_cross_correlation
from .misc import msgit


def tv_l1_pixel(fixed_img, img, mask, shift, filt=True, kernel=3):
    if filt:
        diff_img = gf(fixed_img, kernel) - \
            gf(np.roll(img, shift, axis=[0, 1]), kernel)
        return ((np.abs(np.diff(diff_img, axis=0, prepend=1))+np.abs(np.diff(diff_img, axis=1, prepend=1)))*np.roll(mask, shift, axis=[0, 1])).sum()
    else:
        diff_img = fixed_img - np.roll(img, shift, axis=[0, 1])
        return ((np.abs(np.diff(diff_img, axis=0, prepend=1))+np.abs(np.diff(diff_img, axis=1, prepend=1)))*np.roll(mask, shift, axis=[0, 1])).sum()


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


@msgit(wd=100, fill='-')
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

    tv_h = []
    for ii in range(ls_w):
        tv_h.append(tv_l1_pixel(imgs[0], imgs[1], 1, [0, 2*ii-ls_w],
                                kernel=kernel))
    shift[0, 1] = 2*(np.array(tv_h).argmin() - int(ls_w/2))

    tv1_pxl, tv1_pxl_id, shift[1:], ss = mrtv_reg3(levs=levs, wz=wz, sp_wz=sp_wz,
                                                   sp_step=sp_step, imgs=imgs,
                                                   ps=shift[0])

    print(time.asctime())
    return tv1_pxl, tv1_pxl_id, shift, shift[0]+ss


def shift_img(img_shift):
    img = img_shift[0]
    shift = img_shift[1]
    return np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)))
