#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:09:13 2019

@author: xiao
"""

#import faulthandler; faulthandler.enable()
import os, gc, psutil
import multiprocess as mp
from functools import partial

from pystackreg import StackReg
import tifffile, h5py
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
import numpy as np
from silx.io.dictdump import dicttoh5, h5todict

from .reg_algs import mrtv_mpc_combo_reg, mrtv_reg, mrtv_ls_combo_reg, shift_img
from .io import tiff_vol_reader

N_CPU = os.cpu_count()-1
__all__ = ['regtools']


class regtools():
    def __init__(self, dtype='2D_XANES', **kwargs):
        if dtype not in ['2D_XANES', '3D_XANES']:
            print("Wrong input data_type ...")

        self.si = None
        self.mse = None
        self.nrmse = None
        self.error = None
        self.shift = None
        self.fit_curve = None
        self.fit_curve_diff = None
        self.possible_match = None

        self.data_type = dtype.upper()
        self.fixed = None
        self.img = None
        self.eng_list = None
        self.eng_dict = {}
        self.img_ids = None
        self.img_ids_dict = {}
        self.mask = None
        self.use_mask = False
        self.overlap_ratio = 0.3
        self.save_path = os.curdir
        self.savefn = os.path.join(self.save_path, 'regisstration_results.h5')
        self.xanes3D_trial_reg_fn = os.path.join(self.save_path, 'xanes3D_trial_reg_results.h5')
        self.raw_data_info = {}
        self.data_pnts = None
        self.roi = None

        self.chunk_sz = 7
        self.anchor = 0
        self.chunks = {}
        self.num_chunk = None
        self.alignment_pair_list = []
        self.anchor_chunk = None
        self.use_chunk = False
        self.alg_mrtv_level = 5
        self.alg_mrtv_width  = 10
        self.alg_mrtv_sp_kernel = 3
        self.alg_mrtv_sp_wz = 8

        self.xanes3D_recon_path_template = None
        self.xanes3D_recon_fixed_sli = None
        self.xanes3D_sli_search_half_range = None
        self.xanes3D_recon_file_id_s = None
        self.xanes3D_recon_file_id_e = None
        self.xanes3D_raw_h5_top_dir = None

        self.xanes2D_raw_filename = None
        self.xanes2D_eng_start = None
        self.xanes2D_eng_end = None

        if 'method' in kwargs:
            self.method = kwargs['method']
        else:
            self.method = 'MPC'
            self.overlap_ratio = 0.3
            print('Registration method is set to default phase-correlation \
                  method.')

        if self.method.upper() == 'SR':
            if 'mode' in kwargs:
                self.mode = kwargs['mode'].upper()
            else:
                self.mode = 'TRANSLATION'
                print('"mode" for stackreg method is set to default \
                      "TRANSLATION".')
        else:
            self.mode = ''
        self.ref_mode = 'neighbor'

    def set_analysis_type(self, dtype='3D_XANES'):
        """
        Parameters
        ----------
        dtype : string, optional
            type of analysis in ['2D_XANES', '3D_XANES']. The default is '3D_XANES'.

        Returns
        -------
        None.
        """
        self.data_type = dtype.upper()

    def set_chunk_sz(self, chunk_sz):
        self.chunk_sz = chunk_sz

    def set_roi(self, roi):
        self.roi = roi

    def set_eng_list(self, eng_list):
        self.eng_list = eng_list

    def set_xanes3D_raw_h5_top_dir(self, raw_h5_top_dir):
        self.xanes3D_raw_h5_top_dir = raw_h5_top_dir

    def set_raw_data_info(self, **kwargs):
        self.raw_data_info = kwargs

    def set_xanes3D_recon_path_template(self, path_template):
        self.xanes3D_recon_path_template = path_template

    def set_xanes2D_raw_filename(self, filename):
        self.xanes2D_raw_filename = filename

    def set_xanes2D_tmp_filename(self, filename):
        self.xanes2D_tmp_filename = filename

    def set_xanes3D_tmp_filename(self, filename):
        self.xanes3D_tmp_filename = filename

    def set_xanes3D_scan_ids(self, avail_recon_ids):
        self.img_ids = avail_recon_ids[self.img_id_s:self.img_id_e+1]

    def set_indices(self, img_id_s, img_id_e, fixed_img_id):
        self.img_id_s = img_id_s
        self.img_id_e = img_id_e
        self.fixed_img_id = fixed_img_id

    def set_reg_options(self, use_mask=True, mask_thres=0,
                        use_chunk=True, chunk_sz=7,
                        use_smooth_img=False, smooth_sigma=0,
                        mrtv_level=5, mrtv_width=10,
                        mrtv_sp_wz=8, mrtv_sp_kernel=3):
        """
        the current use_anchor setting is not contraversial. all the registrations
        use some anchors. It is not anchor but chunk setting making differences.
        The code shoudl be modified to have use_chunk and remove use_anchor

        Parameters
        ----------
        use_mask : TYPE, optional
            DESCRIPTION. The default is True.
        mask_thres : TYPE, optional
            DESCRIPTION. The default is 0.
        use_chunk : TYPE, optional
            DESCRIPTION. The default is True.
        anchor_id : TYPE, optional
            DESCRIPTION. The default is 0.
        use_smooth_img : TYPE, optional
            DESCRIPTION. The default is False.
        smooth_sigma : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        self.use_mask = use_mask
        self.mask_thres = mask_thres
        self.use_chunk = use_chunk
        self.chunk_sz = chunk_sz
        self.use_smooth_img = use_smooth_img
        self.img_smooth_sigma = smooth_sigma
        self.alg_mrtv_level = mrtv_level
        self.alg_mrtv_width  = mrtv_width
        self.alg_mrtv_sp_wz = mrtv_sp_wz
        self.alg_mrtv_sp_kernel = mrtv_sp_kernel

    def set_method(self, method):
        self.method = method.upper()

    def set_ref_mode(self, ref_mode):
        self.ref_mode = ref_mode.upper()

    def set_img_data(self, moving):
        self.img = moving

    def set_mask(self, mask):
        self.mask = mask

    def set_saving(self, save_path, fn=None):
        if save_path is not None:
            self.save_path = save_path
        if fn is None:
            fn = 'regisstration_results.h5'
        self.savefn = os.path.join(self.save_path, fn)
        print('1. The registration results will be saved in {:s}'
              .format(self.savefn))

    def read_xanes2D_tmp_file(self, mode='reg'):
        with h5py.File(self.xanes2D_tmp_filename, 'r') as f:
            if mode == 'reg':
                self.img = f['xanes2D_img'][self.img_id_s:self.img_id_e, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
            elif mode == 'align':
                self.img = f['xanes2D_img'][self.img_id_s:self.img_id_e, :]
            self.mask = f['xanes2D_reg_mask'][:]
            if len(self.mask.shape) == 1:
                self.mask = None
            self.eng_list = f['analysis_eng_list'][:]

    def read_xanes3D_tmp_file(self):
        with h5py.File(self.xanes3D_tmp_filename, 'r') as f:
            self.mask = f['xanes3D_reg_mask'][:]
            if len(self.mask.shape) == 1:
                self.mask = None
            self.eng_list = f['analysis_eng_list'][:]

    def compose_dicts(self):
        if self.data_type == '3D_XANES':
            if self.fixed_img_id in range(self.img_id_s, self.img_id_e):
                self.anchor = self.fixed_img_id-self.img_id_s
                self.data_pnts = self.img_id_e - self.img_id_s
                print(self.img_id_s, self.img_id_e)
                print(len(self.img_ids))
                print(self.img_ids)
                print(self.fixed_img_id-self.img_id_s)
                self.anchor_id = self.img_ids[self.fixed_img_id-self.img_id_s]
                cnt = 0
                for ii in self.img_ids:
                    self.eng_dict[str(cnt).zfill(3)] = self.eng_list[cnt]
                    self.img_ids_dict[str(cnt).zfill(3)] = ii
                    cnt += 1
            else:
                print('fixed_img_id is outside of [img_id_s, img_id_e].')
        elif self.data_type == '2D_XANES':
            if self.fixed_img_id in range(self.img_id_s, self.img_id_e):
                self.anchor = self.fixed_img_id - self.img_id_s
                self.data_pnts = self.img_id_e - self.img_id_s
                self.img_ids = np.arange(self.img_id_s, self.img_id_e)
                self.anchor_id = self.fixed_img_id
                cnt = 0
                for ii in range(self.img_id_s, self.img_id_e):
                    self.eng_dict[str(cnt).zfill(3)] = self.eng_list[cnt]
                    self.img_ids_dict[str(cnt).zfill(3)] = ii
                    cnt += 1
            else:
                print('fixed_img_id is outside of [img_id_s, img_id_e].')

    def _set_chunks(self):
        """
        self.data_pnts: relative number defined as self.img_id_e - self.img_id_s + 1
        self.anchor: relative number defined as self.fixed_img_id - self.img_id_s

        self.chunks: generated variable; starting and ending idx of each chunk.
                     the idx are relative idx from self.img_id_s
        Returns
        -------
        None.

        """
        if self.use_chunk:
            right_num_chunk = int(np.ceil((self.data_pnts -
                                           self.anchor) / self.chunk_sz))
            left_num_chunk = int(np.ceil(self.anchor / self.chunk_sz))
            num_chunk = left_num_chunk + right_num_chunk
            self.num_chunk = num_chunk
            self.left_num_chunk = left_num_chunk - 1
            self.anchor_chunk = left_num_chunk - 1
            for ii in range(left_num_chunk-1):
                self.chunks[left_num_chunk-1-ii] = {'chunk_s':\
                    self.anchor - self.chunk_sz * (ii + 1) + 1}
                self.chunks[left_num_chunk-1-ii]['chunk_e'] =\
                    self.anchor - self.chunk_sz * ii

            if (self.anchor % self.chunk_sz) != 0:
                self.chunks[0] = {'chunk_s': 0}
                self.chunks[0]['chunk_e'] = int(self.anchor %
                                                self.chunk_sz)
            else:
                self.chunks[0] = {'chunk_s': 0}
                self.chunks[0]['chunk_e'] = self.chunk_sz

            for ii in range(left_num_chunk, num_chunk-1):
                self.chunks[ii] = {'chunk_s': self.anchor +\
                    self.chunk_sz * (ii - left_num_chunk) + 1}
                self.chunks[ii]['chunk_e'] = self.anchor +\
                    self.chunk_sz * (ii - left_num_chunk + 1)

            if ((self.data_pnts - self.anchor) % self.chunk_sz) != 1:
                self.chunks[num_chunk-1] = {'chunk_s':\
                    self.chunks[num_chunk-2]['chunk_e'] + 1}
                self.chunks[num_chunk-1]['chunk_e'] =\
                    self.data_pnts - 1
            else:
                self.chunks[num_chunk-1] = {'chunk_s':\
                    self.data_pnts - 1}
                self.chunks[num_chunk-1]['chunk_e'] =\
                    self.data_pnts - 1
        else:
            self.chunks[0] = {'chunk_s': 0}
            self.chunks[0]['chunk_e'] = self.data_pnts - 1

    def _alignment_scheduler(self, dtype='2D_XANES'):
        """
        2D XANES: [chunk_sz, img_s,     img_e,     ref_mode, fixed_img]
        3D XANES: [chunk_sz, scan_id_s, scan_id_e, ref_mode, fixed_scan_id]

        According to chunk_sz and ref_mode, we should make a list of pairs for
        comparison. imgs/scan_id_s and img_e/scan_id_e are used to determine
        the bounds to the numbers in the pairs. fixed_img/fixed_scan_id are the
        image/scan used as the anchor in the aligned image/recon sequence.
        These two numbers also affect the list fabrication.

        ref_mode: 'single', 'neighbor', 'average'
            'single': the last images in two consecutive chunks will be
                      compared and aligned. The fixed_img/fixed_scan_id are
                      anchored as the last image in its chunk. The chunks are
                      propagated to the right and left. So, the list of pairs
                      should look like
                      [[fixed_img, left_neighbor_chunk_last_img],
                       [left_neighbor_chunk_last_img, its_left_neighbor],
                       ...,
                       [fixed_img, right_neighbor_chunk_last_img],
                       [right_neighbor_chunk_last_img, its_right_neighbor],
                       ...,
                       [each_pair_in_each_chunk_with_last_img]
                      ]
            'neighbor': the neighbor images in two consecutive chunks will be
                      compared and aligned. The fixed_img/fixed_scan_id are
                      anchored as the last image in its chunk. The chunks are
                      propagated to the right and left. So, the list of pairs
                      should look like
                      [[fixed_img, first_img_in_same_chunk],
                       [first_img_in_same_chunk, its_left_neighbor],
                       ...,
                       [fixed_img, first_img_in_right_neighbor],
                       [first_img_in_right_neighbor, last_img_in_its_chunk],
                       ...,
                       [each_pair_in_each_chunk_with_last_img]
                      ]
        self.alignment_pair_list: generated variable with this function. It defines
                                  pairs of imgs for shift calculation. the idx of
                                  each pair are relative to self.img_id_s
        """
        self._set_chunks()
        self.alignment_pair_list = []

        if self.use_chunk:
            if self.ref_mode.upper() == 'SINGLE':
                # inter-chunk alignment pair
                for ii in range(self.left_num_chunk):
                    self.alignment_pair_list.append([self.chunks[self.left_num_chunk-ii]['chunk_e'],
                                                self.chunks[self.left_num_chunk-ii-1]['chunk_e']])
                self.alignment_pair_list.append([self.anchor_chunk,
                                            self.anchor_chunk])
                print(self.left_num_chunk, self.num_chunk)
                for ii in range(self.left_num_chunk+1, self.num_chunk):
                    self.alignment_pair_list.append([self.chunks[ii-1]['chunk_e'],
                                                self.chunks[ii]['chunk_e']])
                # intra-chunk alignment pair
                for ii in range(self.num_chunk):
                    for jj in range(self.chunks[ii]['chunk_s'],
                                    self.chunks[ii]['chunk_e']+1):
                        self.alignment_pair_list.append([self.chunks[ii]['chunk_e'], jj])

                tem = []
                for ii in self.alignment_pair_list:
                    if ii[0] == ii[1]:
                        tem.append(ii)
                for ii in tem:
                    self.alignment_pair_list.remove(ii)
                self.alignment_pair_list.append([self.anchor, self.anchor])
            elif self.ref_mode.upper() == 'NEIGHBOR':
                # inter-chunk alignment pair
                for ii in range(self.left_num_chunk):
                    self.alignment_pair_list.append([self.chunks[self.left_num_chunk-ii]['chunk_e'],
                                                self.chunks[self.left_num_chunk-ii]['chunk_s']])
                    self.alignment_pair_list.append([self.chunks[self.left_num_chunk-ii]['chunk_s'],
                                                self.chunks[self.left_num_chunk-ii-1]['chunk_e']])
                self.alignment_pair_list.append([self.chunks[self.anchor_chunk]['chunk_e'],
                                            self.chunks[self.anchor_chunk]['chunk_e']+1])
                for ii in range(self.left_num_chunk+1, self.num_chunk-1):
                    self.alignment_pair_list.append([self.chunks[ii]['chunk_s'],
                                                self.chunks[ii]['chunk_e']])
                    self.alignment_pair_list.append([self.chunks[ii]['chunk_e'],
                                                self.chunks[ii+1]['chunk_s']])
                self.alignment_pair_list.append([self.chunks[self.num_chunk-1]['chunk_s'],
                                            self.chunks[self.num_chunk-1]['chunk_e']])
                # inter-chunk alignment pair
                for ii in range(self.num_chunk):
                    for jj in range(self.chunks[ii]['chunk_s'],
                                    self.chunks[ii]['chunk_e']+1):
                        self.alignment_pair_list.append([self.chunks[ii]['chunk_e'], jj])

                tem = []
                for ii in self.alignment_pair_list:
                    if ii[0] == ii[1]:
                        tem.append(ii)
                for ii in tem:
                    self.alignment_pair_list.remove(ii)
                self.alignment_pair_list.append([self.anchor, self.anchor])
        else:
            for ii in range(self.data_pnts-1):
                self.alignment_pair_list.append([ii, ii+1])
            self.alignment_pair_list.append([self.anchor, self.anchor])

    def _sort_absolute_shift(self, trialfn, shift_dict=None, optional_shift_dict=None):
        """
        self.shift_chain_dict: generated variables with this function. the idx
                               each img is associated with a chain of other img
                               idx with which the img shift is uniquely defined
                               relative to the anchor img. all idx are relative
                               to self.img_id_s

        Parameters
        ----------
        trialfn : TYPE
            DESCRIPTION.
        shift_dict : TYPE, optional
            DESCRIPTION. The default is None.
        optional_shift_dict : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if self.data_type.upper() == "3D_XANES":
            f = h5py.File(trialfn, 'r')
            method = f['/trial_registration/trial_reg_parameters/reg_method'][()]

            self.shift_chain_dict = {}
            for ii in range(len(self.alignment_pair_list)-1, -1, -1):
                self.shift_chain_dict[self.alignment_pair_list[ii][1]] = [self.alignment_pair_list[ii][0]]
                jj = ii - 1
                while jj>=0:
                    if self.shift_chain_dict[self.alignment_pair_list[ii][1]][-1] == self.alignment_pair_list[jj][1]:
                        self.shift_chain_dict[self.alignment_pair_list[ii][1]].append(self.alignment_pair_list[jj][0])
                    jj -= 1
            abs_shift_dict = {}
            if method.upper() == "SR":
                for key, item in self.shift_chain_dict.items():
                    item.insert(0, key)
                    shift = np.identity(3)
                    slioff = 0
                    for ii in range(len(item)-1):
                        idx = self.alignment_pair_list.index([item[ii+1], item[ii]])
                        shift = np.matmul(shift, np.array(shift_dict[str(idx)][1]))
                        slioff += int(optional_shift_dict[str(idx)][0])
                    abs_shift_dict[str(key).zfill(3)] = {'in_sli_shift':shift, 'out_sli_shift':slioff}
            else:
                for key, item in self.shift_chain_dict.items():
                    item.insert(0, key)
                    shift = 0.
                    slioff = 0
                    for ii in range(len(item)-1):
                        idx = self.alignment_pair_list.index([item[ii+1], item[ii]])
                        shift += np.array(shift_dict[str(idx)][1:])
                        slioff += int(shift_dict[str(idx)][0])
                    abs_shift_dict[str(key).zfill(3)] = {'in_sli_shift':shift, 'out_sli_shift':slioff}
            f.close()
        elif self.data_type.upper() == "2D_XANES":
            f = h5py.File(trialfn, 'r')
            method = f['/trial_registration/trial_reg_parameters/reg_method'][()]
            self.shift_chain_dict = {}
            for ii in range(len(self.alignment_pair_list)-1, -1, -1):
                self.shift_chain_dict[self.alignment_pair_list[ii][1]] = [self.alignment_pair_list[ii][0]]
                jj = ii - 1
                while jj>=0:
                    if self.shift_chain_dict[self.alignment_pair_list[ii][1]][-1] == self.alignment_pair_list[jj][1]:
                        self.shift_chain_dict[self.alignment_pair_list[ii][1]].append(self.alignment_pair_list[jj][0])
                    jj -= 1
            abs_shift_dict = {}
            if method.upper() == "SR":
                for key, item in self.shift_chain_dict.items():
                    item.insert(0, key)
                    shift = np.identity(3)
                    for ii in range(len(item)-1):
                        idx = self.alignment_pair_list.index([item[ii+1], item[ii]])
                        shift = np.matmul(shift, np.array(shift_dict[str(idx)]))
                    abs_shift_dict[str(key).zfill(3)] = {'in_sli_shift':shift}
            else:
                for key, item in self.shift_chain_dict.items():
                    item.insert(0, key)
                    print(key, item)
                    shift = 0.
                    for ii in range(len(item)-1):
                        idx = self.alignment_pair_list.index([item[ii+1], item[ii]])
                        shift += np.array(shift_dict[str(idx)])
                    abs_shift_dict[str(key).zfill(3)] = {'in_sli_shift':shift}
            f.close()
        self.abs_shift_dict = abs_shift_dict

    def _chunking(self, dim, mem_lim=None):
        img_sz = (dim[1]*dim[2]*4)
        if mem_lim is None:
            mem = psutil.virtual_memory()
            mem_lim = mem.available/3.
        mem_lim = (mem_lim//img_sz)*img_sz
        while (dim[0]*img_sz%mem_lim)*mem_lim/img_sz < N_CPU:
            mem_lim += N_CPU*img_sz
        num_img_in_batch = np.round(mem_lim/img_sz/N_CPU)*N_CPU
        num_batch = int(np.ceil(dim[0]/num_img_in_batch))
        bdi = []
        chunk = int(np.round(num_img_in_batch/N_CPU))
        for ii in range(num_batch):
            if ii < num_batch-1:
                for jj in range(N_CPU):
                    bdi.append(int(ii*num_img_in_batch + jj*chunk))
            else:
                chunk = int(np.ceil((dim[0] - ii*num_img_in_batch)/N_CPU))
                for jj in range(N_CPU+1):
                    bdi.append(int(ii*num_img_in_batch + jj*chunk))
                bdi[-1] = min(dim[0], bdi[-1])
        return bdi, num_batch

    def reg_xanes2D_chunk(self, overlap_ratio=0.3):
        """
        chunk_sz: int, number of image in one chunk for alignment; each chunk
                  use the last image in that chunk as reference
        method:   str
                  'PC':   skimage.feature.register_translation
                  'MPC':  skimage.feature.masked_register_translation
                  'SR':   pystackreg.StackReg
        overlap_ratio: float, overlap_ratio for method == 'MPC'
        ref_mode: str, control how inter-chunk alignment is done
                  'average': the average of each chunk after intra-chunk
                             re-alignment is used for inter-chunk alignment
                  'single':  the last image in each chunk is used in
                             inter-chunk alignment

        imgs in self.img are registered relative to anchor img. self.img
        is the sub stack with self.img_id_s as its first image, and self.img_id_e
        as the last.
        """
        self.overlap_ratio = overlap_ratio
        self._alignment_scheduler(dtype='2D_XANES')

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, mode=777)
            print(f'2. The registration results will be saved in {self.savefn}')

        f = h5py.File(self.savefn, 'a')
        if 'trial_registration' not in f:
            g0 = f.create_group('trial_registration')
        else:
            del f['trial_registration']
            g0 = f.create_group('trial_registration')

        g1 = g0.create_group('trial_reg_results')
        g2 = g0.create_group('trial_reg_parameters')

        g2.create_dataset('reg_method', data=str(self.method.upper()))
        g2.create_dataset('reg_ref_mode', data=str(self.ref_mode.upper()))
        g2.create_dataset('use_smooth_img', data=str(self.use_smooth_img))
        g2.create_dataset('img_smooth_sigma', data=self.img_smooth_sigma)

        g2.create_dataset('alignment_pairs', data=self.alignment_pair_list)
        g2.create_dataset('scan_ids', data=self.img_ids)
        g2.create_dataset('use_chunk', data=str(self.use_chunk))
        g2.create_dataset('chunk_sz', data=self.chunk_sz)
        g2.create_dataset('fixed_scan_id', data=self.anchor_id)
        g2.create_dataset('slice_roi', data=self.roi)
        g2.create_dataset('eng_list', data=self.eng_list)
        dicttoh5(self.eng_dict, self.savefn, mode='a',
                 overwrite_data=True,
                 h5path='/trial_registration/trial_reg_parameters/eng_dict')
        dicttoh5(self.img_ids_dict, self.savefn, mode='a',
                 overwrite_data=True,
                 h5path='/trial_registration/trial_reg_parameters/scan_ids_dict')
        g2.create_dataset('use_mask', data=str(self.use_mask))
        g2.create_dataset('mask_thres', data=self.mask_thres)
        if self.use_mask:
            g2.create_dataset('mask', data=self.mask)

        g3 = g0.create_group('data_directory_info')
        for key, val in self.raw_data_info.items():
            g3.create_dataset(key, data=val)

        shifted_image = np.ndarray(self.img.shape)

        if self.img.ndim != 3:
                print('XANES2D image stack is required. Please set XANES2D \
                      image stack first.')
        else:
            if self.method.upper() in {'PC', 'MPC', 'MRTV', 'LS+MRTV', 'MPC+MRTV'}:
                self.shift = np.ndarray([len(self.alignment_pair_list), 2])
            else:
                self.shift = np.ndarray([len(self.alignment_pair_list), 3, 3])

            self.error = np.ndarray(len(self.alignment_pair_list))
            self.si = np.ndarray(len(self.alignment_pair_list))
            self.mse = np.ndarray(len(self.alignment_pair_list))
            self.nrmse = np.ndarray(len(self.alignment_pair_list))

            if self.method.upper() == 'PC':
                print('We are using "phase correlation" method for registration.')
                for ii in range(len(self.alignment_pair_list)):
                    self.shift[ii], self.error[ii], _ = phase_cross_correlation(
                            self.img[self.alignment_pair_list[ii][0]],
                            self.img[self.alignment_pair_list[ii][1]], upsample_factor=100)
                    shifted_image[ii] = np.real(np.fft.ifftn(fourier_shift(
                            np.fft.fftn(self.img[self.alignment_pair_list[ii][1]]), self.shift[ii])))[:]
                    g11 = g1.create_group(str(ii).zfill(3))
                    g11.create_dataset('shift'+str(ii).zfill(3),
                                       data=self.shift[ii])
                    g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                       data=shifted_image[ii])
                    g11.create_dataset('trial_reg_fixed'+str(ii).zfill(3),
                                       data=self.img[self.alignment_pair_list[ii][0]])
            elif self.method.upper() == 'MPC':
                print('We are using "masked phase correlation" method for registration.')
                for ii in range(len(self.alignment_pair_list)):
                    self.shift[ii] = phase_cross_correlation(self.img[self.alignment_pair_list[ii][0]],
                                                             self.img[self.alignment_pair_list[ii][1]],
                                                             reference_mask=self.mask,
                                                             overlap_ratio=self.overlap_ratio)
                    shifted_image[ii] = np.real(np.fft.ifftn(fourier_shift(
                            np.fft.fftn(self.img[self.alignment_pair_list[ii][1]]), self.shift[ii])))[:]
                    g11 = g1.create_group(str(ii).zfill(3))
                    g11.create_dataset('shift'+str(ii).zfill(3),
                                       data=self.shift[ii])
                    g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                       data=shifted_image[ii])
                    g11.create_dataset('trial_reg_fixed'+str(ii).zfill(3),
                                       data=self.img[self.alignment_pair_list[ii][0]])
            elif self.method.upper() == 'SR':
                print('We are using "stack registration" method for registration.')
                if self.mode.upper() == 'TRANSLATION':
                    sr = StackReg(StackReg.TRANSLATION)
                elif  self.mode.upper() == 'RIGID_BODY':
                    sr = StackReg(StackReg.RIGID_BODY)
                elif  self.mode.upper() == 'SCALED_ROTATION':
                    sr = StackReg(StackReg.SCALED_ROTATION)
                elif  self.mode.upper() == 'AFFINE':
                    sr = StackReg(StackReg.AFFINE)
                elif  self.mode.upper() == 'BILINEAR':
                    sr = StackReg(StackReg.BILINEAR)

                if self.mask is not None:
                    for ii in range(len(self.alignment_pair_list)):
                        self.shift[ii] = sr.register(self.img[self.alignment_pair_list[ii][0]]*self.mask,
                                                     self.img[self.alignment_pair_list[ii][1]]*self.mask)
                        shifted_image[ii] = sr.transform(self.img[self.alignment_pair_list[ii][1]],
                                                                               tmat=self.shift[ii])[:]
                        g11 = g1.create_group(str(ii).zfill(3))
                        g11.create_dataset('shift'+str(ii).zfill(3),
                                           data=self.shift[ii])
                        g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                           data=shifted_image[ii])
                        g11.create_dataset('trial_reg_fixed'+str(ii).zfill(3),
                                           data=self.img[self.alignment_pair_list[ii][0]])
                else:
                    for ii in range(len(self.alignment_pair_list)):
                        self.shift[ii] = sr.register(self.img[self.alignment_pair_list[ii][0]],
                                                     self.img[self.alignment_pair_list[ii][1]])
                        shifted_image[ii] = sr.transform(self.img[self.alignment_pair_list[ii][1]],
                                                                               tmat=self.shift[ii])[:]
                        g11 = g1.create_group(str(ii).zfill(3))
                        g11.create_dataset('shift'+str(ii).zfill(3),
                                           data=self.shift[ii])
                        g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                           data=shifted_image[ii])
                        g11.create_dataset('trial_reg_fixed'+str(ii).zfill(3),
                                           data=self.img[self.alignment_pair_list[ii][0]])
            elif self.method.upper() == 'MRTV':
                print('We are using "multi-resolution total variation" method for registration.')
                print(self.alg_mrtv_sp_wz, self.alg_mrtv_sp_kernel)
                pxl_conf = {'type': 'area',
                            'levs': self.alg_mrtv_level,
                            'wz': self.alg_mrtv_width,
                            'lsw': 10}
                sub_conf = {'use': True,
                            'type': 'ana',
                            'sp_wz': self.alg_mrtv_sp_wz,
                            'sp_us': 10}
                with mp.get_context('spawn').Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, pxl_conf, sub_conf, None, self.alg_mrtv_sp_kernel),
                                   [[self.img[self.alignment_pair_list[ii][0]],
                                    self.img[self.alignment_pair_list[ii][1]]]
                                   for ii in range(len(self.alignment_pair_list))])
                pool.close()
                pool.join()

                for ii in range(len(rlt)):
                    self.shift[ii] = rlt[ii][3]
                del(rlt)
                gc.collect()

                for ii in range(len(self.alignment_pair_list)):
                    shifted_image[ii] = np.real(np.fft.ifftn(fourier_shift(
                            np.fft.fftn(self.img[self.alignment_pair_list[ii][1]]), self.shift[ii])))[:]
                    g11 = g1.create_group(str(ii).zfill(3))
                    g11.create_dataset('shift'+str(ii).zfill(3),
                                        data=self.shift[ii])
                    g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                        data=shifted_image[ii])
                    g11.create_dataset('trial_reg_fixed'+str(ii).zfill(3),
                                        data=self.img[self.alignment_pair_list[ii][0]])
            elif self.method.upper() == 'LS+MRTV':
                print('We are using "line search and multi-resolution total variation" method for registration.')
                with mp.get_context('spawn').Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_ls_combo_reg, self.alg_mrtv_width, 2, 10,
                                           self.alg_mrtv_sp_wz, self.alg_mrtv_sp_wz),
                                   [[self.img[self.alignment_pair_list[ii][0]],
                                     self.img[self.alignment_pair_list[ii][1]]]
                                    for ii in range(len(self.alignment_pair_list))])
                pool.close()
                pool.join()

                for ii in range(len(rlt)):
                    self.shift[ii] = rlt[ii][3]
                del(rlt)
                gc.collect()

                for ii in range(len(self.alignment_pair_list)):
                    shifted_image[ii] = np.real(np.fft.ifftn(fourier_shift(
                            np.fft.fftn(self.img[self.alignment_pair_list[ii][1]]), self.shift[ii])))[:]
                    g11 = g1.create_group(str(ii).zfill(3))
                    g11.create_dataset('shift'+str(ii).zfill(3),
                                        data=self.shift[ii])
                    g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                        data=shifted_image[ii])
                    g11.create_dataset('trial_reg_fixed'+str(ii).zfill(3),
                                        data=self.img[self.alignment_pair_list[ii][0]])
            elif self.method.upper() == 'MPC+MRTV':
                print('We are using combo of "masked phase correlation" and "multi-resolution total variation" method for registration.')
                for ii in range(len(self.alignment_pair_list)):
                    _, _, _, self.shift[ii] = mrtv_mpc_combo_reg(self.img[self.alignment_pair_list[ii][0]],
                                                                 self.img[self.alignment_pair_list[ii][1]],
                                                                 reference_mask=self.mask,
                                                                 overlap_ratio=self.overlap_ratio,
                                                                 levs=self.alg_mrtv_level,
                                                                 wz=self.alg_mrtv_width,
                                                                 sp_wz=self.alg_mrtv_sp_wz,
                                                                 sp_step=self.alg_mrtv_sp_wz)
                    shifted_image[ii] = np.real(np.fft.ifftn(fourier_shift(
                            np.fft.fftn(self.img[self.alignment_pair_list[ii][1]]), self.shift[ii])))[:]
                    g11 = g1.create_group(str(ii).zfill(3))
                    g11.create_dataset('shift'+str(ii).zfill(3),
                                        data=self.shift[ii])
                    g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                        data=shifted_image[ii])
                    g11.create_dataset('trial_reg_fixed'+str(ii).zfill(3),
                                        data=self.img[self.alignment_pair_list[ii][0]])
        f.close()
        print('Done!')

    def apply_xanes2D_chunk_shift(self, optional_shift_dict, trialfn=None, savefn=None):
        """
        trialfn:    string; optional
                    trial registration filename
        savefn:     string; optional
                    filename to the file in which the shifted volume to be saved
        optional_shift_dict: dict; optional
                    user input shifts for specified scan ids. This is useful to
                    correct individual pairs that cannot be aligned with others
                    with the same registration method
        """
        if savefn is None:
            savefn = self.savefn
        if trialfn is None:
            trialfn = self.savefn

        if savefn == trialfn:
            with h5py.File(savefn, 'a') as f:
                if 'registration_results' not in f:
                    g0 = f.create_group('registration_results')
                else:
                    del f['registration_results']
                    g0 = f.create_group('registration_results')

                g1 = g0.create_group('reg_parameters')
                self.alignment_pair_list = f['/trial_registration/trial_reg_parameters/alignment_pairs'][:].tolist()
                g1.create_dataset('alignment_pairs', data=self.alignment_pair_list)
                self.img_ids = f['/trial_registration/trial_reg_parameters/scan_ids'][:]
                g1.create_dataset('scan_ids', data=self.img_ids)
                g1.create_dataset('slice_roi', data=self.roi)
                self.anchor_id = f['/trial_registration/trial_reg_parameters/fixed_scan_id'][()]
                g1.create_dataset('fixed_scan_id', data=self.anchor_id)
                self.chunk_sz = f['/trial_registration/trial_reg_parameters/chunk_sz'][()]
                g1.create_dataset('chunk_sz', data=self.chunk_sz)
                self.method = f['/trial_registration/trial_reg_parameters/reg_method'][()].decode('ascii')
                g1.create_dataset('reg_method', data=self.method)
                self.ref_mode = f['/trial_registration/trial_reg_parameters/reg_ref_mode'][()].decode('ascii')
                g1.create_dataset('reg_ref_mode', data=str(self.ref_mode.upper()))
                self.img_ids_dict = h5todict(savefn, path='/trial_registration/trial_reg_parameters/scan_ids_dict')
                self.eng_dict = h5todict(savefn, path='/trial_registration/trial_reg_parameters/eng_dict')
                dicttoh5(self.eng_dict, savefn, mode='a',
                     overwrite_data=True,
                     h5path='/registration_results/reg_parameters/eng_dict')
                dicttoh5(self.img_ids_dict, savefn, mode='a',
                     overwrite_data=True,
                     h5path='/registration_results/reg_parameters/scan_ids_dict')

                dicttoh5(optional_shift_dict, savefn, mode='a', overwrite_data=True,
                         h5path='/registration_results/reg_parameters/user_determined_shift/relative_shift')

                self._sort_absolute_shift(trialfn, shift_dict=optional_shift_dict)

                shift = {}
                for key, item in self.abs_shift_dict.items():
                    shift[key] = item['in_sli_shift']

                dicttoh5(shift, savefn, mode='a',
                         overwrite_data=True,
                         h5path='/registration_results/reg_parameters/user_determined_shift/absolute_shift')

                g2 = g0.create_group('reg_results')
                g21 = g2.create_dataset('registered_xanes2D',
                                        shape=(len(self.img_ids_dict),
                                               self.roi[1]-self.roi[0],
                                               self.roi[3]-self.roi[2]))
                g22 = g2.create_dataset('eng_list', shape=(len(self.img_ids_dict),))

                cnt1 = 0
                for key in sorted(self.abs_shift_dict.keys()):
                    shift = self.abs_shift_dict[key]['in_sli_shift']
                    self._translate_single_img(self.img[int(key)], shift, self.method)
                    g21[cnt1] = self.img[int(key)][self.roi[0]:self.roi[1],
                                                   self.roi[2]:self.roi[3]]
                    g22[cnt1] = self.eng_dict[key]
                    cnt1 += 1

    def reg_xanes3D_chunk(self):
        """
        This function will align 3D XANES reconstructions chunk by chunk. Each
        3D dataset in each chunk will be aligned to the last dataset in the
        same chunk. Different chunks will be aligned in three different
        manners: 'single', 'neighbor', and 'average'.
        One way to do it is to make a checking order list according to the
        user input. The list is composed of a sequence of pairs. The alignment
        will be applied on each pair. A scheduler is therefore needed for this
        purpose.

        """
        fn = self.xanes3D_recon_path_template.format(self.img_ids[self.anchor],
                                                      str(self.xanes3D_recon_fixed_sli).zfill(5))
        self.fixed = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        img = np.ndarray([2*self.xanes3D_sli_search_half_range,
                          self.roi[1]-self.roi[0],
                          self.roi[3]-self.roi[2]])
        self._alignment_scheduler(dtype='3D_XANES')

        sli_s = self.xanes3D_recon_fixed_sli-self.xanes3D_sli_search_half_range
        sli_e = self.xanes3D_recon_fixed_sli+self.xanes3D_sli_search_half_range

        f = h5py.File(self.savefn, 'a')
        if 'trial_registration' not in f:
            g0 = f.create_group('trial_registration')
        else:
            del f['trial_registration']
            g0 = f.create_group('trial_registration')


        g1 = g0.create_group('trial_reg_results')

        g2 = g0.create_group('trial_reg_parameters')
        g2.create_dataset('reg_method', data=str(self.method.upper()))
        g2.create_dataset('reg_ref_mode', data=str(self.ref_mode.upper()))
        g2.create_dataset('use_smooth_img', data=str(self.use_smooth_img))
        g2.create_dataset('img_smooth_sigma', data=self.img_smooth_sigma)

        g2.create_dataset('alignment_pairs', data=self.alignment_pair_list)
        g2.create_dataset('scan_ids', data=self.img_ids)
        g2.create_dataset('use_chunk', data=str(self.use_chunk))
        g2.create_dataset('chunk_sz', data=self.chunk_sz)
        g2.create_dataset('fixed_scan_id', data=self.anchor_id)
        g2.create_dataset('slice_roi', data=self.roi)
        g2.create_dataset('fixed_slice', data=self.xanes3D_recon_fixed_sli)
        g2.create_dataset('sli_search_half_range',
                          data=self.xanes3D_sli_search_half_range)
        g2.create_dataset('eng_list', data=self.eng_list)
        g2.create_dataset('use_mask', data=self.use_mask)
        if self.use_mask:
            g2.create_dataset('mask', data=self.mask)
            g2.create_dataset('mask_thres', data=self.mask_thres)
        else:
            g2.create_dataset('mask', data=str(self.mask))
            g2.create_dataset('mask_thres', data=self.mask_thres)

        dicttoh5(self.eng_dict, self.savefn, mode='a',
                 overwrite_data=True,
                 h5path='/trial_registration/trial_reg_parameters/eng_dict')
        dicttoh5(self.img_ids_dict, self.savefn, mode='a',
                 overwrite_data=True,
                 h5path='/trial_registration/trial_reg_parameters/scan_ids_dict')

        g3 = g0.create_group('data_directory_info')
        tem = ''
        for ii in self.xanes3D_recon_path_template.split('/')[:-2]:
            tem = os.path.join(tem, ii)
        g3.create_dataset('raw_h5_top_dir', data=self.xanes3D_raw_h5_top_dir)
        g3.create_dataset('recon_top_dir', data=tem)
        g3.create_dataset('recon_path_template', data=self.xanes3D_recon_path_template)
        for key, val in self.raw_data_info.items():
            try:
                g3.create_dataset(key, data=val)
            except:
                pass

        if self.method.upper() in {'PC', 'MPC', 'MRTV', 'LS+MRTV', 'MPC+MRTV'}:
            self.shift = np.ndarray([len(self.alignment_pair_list),
                                     2*self.xanes3D_sli_search_half_range,
                                     2])
        else:
            self.shift = np.ndarray([len(self.alignment_pair_list),
                                     2*self.xanes3D_sli_search_half_range,
                                     3,
                                     3])
        self.error = np.ndarray([len(self.alignment_pair_list),
                                    2*self.xanes3D_sli_search_half_range])
        self.si = np.ndarray(len(self.alignment_pair_list))
        self.mse = np.ndarray(len(self.alignment_pair_list))
        self.nrmse = np.ndarray(len(self.alignment_pair_list))

        if self.method.upper() == 'PC':
            print('We are using "phase correlation" method for registration.')
            for ii in range(len(self.alignment_pair_list)):
                fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][0]).zfill(3)],
                                                             str(self.xanes3D_recon_fixed_sli).zfill(5))
                self.fixed[:] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                jj_id = 0
                for jj in range(sli_s, sli_e):
                    fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][1]).zfill(3)], str(jj).zfill(5))
                    img[jj_id] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

                    self.shift[ii, jj_id], self.error[ii, jj_id], _ = phase_cross_correlation(self.fixed,
                                                                                           img[jj_id], upsample_factor=100)
                    img[jj_id] = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img[jj_id]),
                                                                    self.shift[ii, jj_id])))[:]

                    jj_id += 1
                g11 = g1.create_group(str(ii).zfill(3))
                g11.create_dataset('shift'+str(ii).zfill(3),
                                   data=self.shift[ii])
                g11.create_dataset('error'+str(ii).zfill(3),
                                   data=self.error[ii])
                g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                   data=img)
                g11.create_dataset('trial_fixed_img'+str(ii).zfill(3),
                                   data=self.fixed)
        elif self.method.upper() == 'MPC':
            print('We are using "masked phase correlation" method for registration.')
            for ii in range(len(self.alignment_pair_list)):
                fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][0]).zfill(3)],
                                                             str(self.xanes3D_recon_fixed_sli).zfill(5))
                self.fixed[:] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                jj_id = 0
                for jj in range(sli_s, sli_e):
                    fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][1]).zfill(3)], str(jj).zfill(5))
                    img[jj_id] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

                    self.shift[ii, jj_id] = phase_cross_correlation(self.fixed, img[jj_id],
                                                                    reference_mask=self.mask, overlap_ratio=self.overlap_ratio)
                    img[jj_id] = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img[jj_id]),
                                                                    self.shift[ii, jj_id])))[:]

                    jj_id += 1
                g11 = g1.create_group(str(ii).zfill(3))
                g11.create_dataset('shift'+str(ii).zfill(3),
                                   data=self.shift[ii])
                g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                   data=img)
                g11.create_dataset('trial_fixed_img'+str(ii).zfill(3),
                                   data=self.fixed)
        elif self.method.upper() == 'SR':
            print('We are using "stack registration" method for registration.')
            if self.mode.upper() == 'TRANSLATION':
                sr = StackReg(StackReg.TRANSLATION)
            elif  self.mode.upper() == 'RIGID_BODY':
                sr = StackReg(StackReg.RIGID_BODY)
            elif  self.mode.upper() == 'SCALED_ROTATION':
                sr = StackReg(StackReg.SCALED_ROTATION)
            elif  self.mode.upper() == 'AFFINE':
                sr = StackReg(StackReg.AFFINE)
            elif  self.mode.upper() == 'BILINEAR':
                sr = StackReg(StackReg.BILINEAR)

            if self.mask is not None:
                for ii in range(len(self.alignment_pair_list)):
                    fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][0]).zfill(3)],
                                                                 str(self.xanes3D_recon_fixed_sli).zfill(5))
                    self.fixed[:] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                    jj_id = 0
                    for jj in range(sli_s, sli_e):
                        fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][1]).zfill(3)], str(jj).zfill(5))
                        img[jj_id] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

                        self.shift[ii, jj_id] = sr.register(self.fixed*self.mask, img[jj_id]*self.mask)
                        img[jj_id] = sr.transform(img[jj_id], self.shift[ii, jj_id])[:]

                        jj_id += 1
                    g11 = g1.create_group(str(ii).zfill(3))
                    g11.create_dataset('shift'+str(ii).zfill(3),
                                       data=self.shift[ii, :].astype(np.float32))
                    g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                       data=img)
                    g11.create_dataset('trial_fixed_img'+str(ii).zfill(3),
                                       data=self.fixed)
            else:
                for ii in range(len(self.alignment_pair_list)):
                    fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][0]).zfill(3)],
                                                                 str(self.xanes3D_recon_fixed_sli).zfill(5))
                    self.fixed[:] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                    jj_id = 0
                    for jj in range(sli_s, sli_e):
                        fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][1]).zfill(3)], str(jj).zfill(5))
                        img[jj_id] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

                        self.shift[ii, jj_id] = sr.register(self.fixed, img[jj_id])
                        img[jj_id] = sr.transform(img[jj_id], self.shift[ii, jj_id])[:]

                        jj_id += 1
                    g11 = g1.create_group(str(ii).zfill(3))
                    g11.create_dataset('shift'+str(ii).zfill(3),
                                       data=self.shift[ii, :].astype(np.float32))
                    g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                       data=img)
                    g11.create_dataset('trial_fixed_img'+str(ii).zfill(3),
                                       data=self.fixed)
        elif self.method.upper() == 'MRTV':
            print('We are using "multi-resolution total variation" method for registration.')
            sli_s = self.xanes3D_recon_fixed_sli-self.xanes3D_sli_search_half_range
            sli_e = self.xanes3D_recon_fixed_sli+self.xanes3D_sli_search_half_range
            pxl_conf = dict(type='area', levs=self.alg_mrtv_level, wz=self.alg_mrtv_width, lsw=10)
            sub_conf = dict(use=True, type='ana', sp_wz=self.alg_mrtv_sp_wz, sp_us=10)
            for ii in range(len(self.alignment_pair_list)):
                fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][0]).zfill(3)],
                                                             str(self.xanes3D_recon_fixed_sli).zfill(5))
                self.fixed[:] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

                jj_id = 0
                for jj in range(sli_s, sli_e):
                    fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][1]).zfill(3)], str(jj).zfill(5))
                    img[jj_id] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                    jj_id += 1

                with mp.get_context('spawn').Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, pxl_conf, sub_conf, None, self.alg_mrtv_sp_kernel),
                                   [[self.fixed, img[jj]] for jj in range(sli_e-sli_s)])
                pool.close()
                pool.join()

                tvl = []
                for jj in range(len(rlt)):
                    self.shift[ii, jj] = rlt[jj][3]
                    tvl.append(rlt[jj][0][self.alg_mrtv_level-1].flatten()[rlt[jj][1][-1]])
                tv = np.array(tvl).argmin()
                del(rlt)
                gc.collect()

                with mp.get_context('spawn').Pool(N_CPU) as pool:
                    rlt = pool.map(shift_img,
                                   [[img[jj], self.shift[ii, jj]] for jj in range(sli_e-sli_s)])
                pool.close()
                pool.join()

                for jj in range(len(rlt)):
                    img[jj] = rlt[jj]
                del(rlt)
                gc.collect()

                g11 = g1.create_group(str(ii).zfill(3))
                g11.create_dataset('mrtv_best_shift_id'+str(ii).zfill(3),
                                   data=tv)
                g11.create_dataset('tv'+str(ii).zfill(3),
                                   data=tvl)
                g11.create_dataset('shift'+str(ii).zfill(3),
                                   data=self.shift[ii])
                g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                   data=img)
                g11.create_dataset('trial_fixed_img'+str(ii).zfill(3),
                                   data=self.fixed)
        elif self.method.upper() == 'LS+MRTV':
            print('We are using "multi-resolution total variation" method for registration.')
            sli_s = self.xanes3D_recon_fixed_sli-self.xanes3D_sli_search_half_range
            sli_e = self.xanes3D_recon_fixed_sli+self.xanes3D_sli_search_half_range
            for ii in range(len(self.alignment_pair_list)):
                fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][0]).zfill(3)],
                                                             str(self.xanes3D_recon_fixed_sli).zfill(5))
                self.fixed[:] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

                jj_id = 0
                for jj in range(sli_s, sli_e):
                    fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][1]).zfill(3)], str(jj).zfill(5))
                    img[jj_id] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                    jj_id += 1

                with mp.get_context('spawn').Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_ls_combo_reg, self.alg_mrtv_width, 2, 10,
                                           self.alg_mrtv_sp_wz, self.alg_mrtv_sp_wz),
                                   [[self.fixed, img[jj]] for jj in range(sli_e-sli_s)])
                pool.close()
                pool.join()

                tvl = []
                for jj in range(len(rlt)):
                    self.shift[ii, jj] = rlt[jj][3]
                    tvl.append(rlt[jj][0][self.alg_mrtv_level-1].flatten()[rlt[jj][1][-1]])
                tv = np.array(tvl).argmin()
                del(rlt)
                gc.collect()

                with mp.get_context('spawn').Pool(N_CPU) as pool:
                    rlt = pool.map(shift_img,
                                   [[img[jj], self.shift[ii, jj]] for jj in range(sli_e-sli_s)])
                pool.close()
                pool.join()

                for jj in range(len(rlt)):
                    img[jj] = rlt[jj]
                del(rlt)
                gc.collect()

                g11 = g1.create_group(str(ii).zfill(3))
                g11.create_dataset('mrtv_best_shift_id'+str(ii).zfill(3),
                                   data=tv)
                g11.create_dataset('tv'+str(ii).zfill(3),
                                   data=tvl)
                g11.create_dataset('shift'+str(ii).zfill(3),
                                   data=self.shift[ii])
                g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                   data=img)
                g11.create_dataset('trial_fixed_img'+str(ii).zfill(3),
                                   data=self.fixed)
        elif self.method.upper() == 'MPC+MRTV':
            print('We are using combo of "masked phase correlation" and "masked phase correlation" method for registration.')
            for ii in range(len(self.alignment_pair_list)):
                fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][0]).zfill(3)],
                                                             str(self.xanes3D_recon_fixed_sli).zfill(5))
                self.fixed[:] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                jj_id = 0
                for jj in range(sli_s, sli_e):
                    fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][1]).zfill(3)], str(jj).zfill(5))
                    img[jj_id] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                    _, _, _, self.shift[ii, jj_id] = mrtv_mpc_combo_reg(self.fixed, img[jj_id],
                                                                        reference_mask=self.mask,
                                                                        overlap_ratio=self.overlap_ratio,
                                                                        levs=self.alg_mrtv_level,
                                                                        wz=self.alg_mrtv_width,
                                                                        sp_wz=self.alg_mrtv_sp_wz,
                                                                        sp_step=self.alg_mrtv_sp_wz)
                    img[jj_id] = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img[jj_id]),
                                                                    self.shift[ii, jj_id])))[:]

                    jj_id += 1
                g11 = g1.create_group(str(ii).zfill(3))
                g11.create_dataset('shift'+str(ii).zfill(3),
                                   data=self.shift[ii])
                g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                   data=img)
                g11.create_dataset('trial_fixed_img'+str(ii).zfill(3),
                                   data=self.fixed)
        f.close()

    def apply_xanes3D_chunk_shift(self, shift_dict, sli_s, sli_e,
                                  trialfn=None, savefn=None,
                                  optional_shift_dict=None, mem_lim=None):
        """
        shift_dict: disctionary;
                    user-defined shift list based on visual inspections of
                    trial registration results; it is configured as
            {'id1':    shift_id1,
             ...
             'idn':    shift_idn}
                  idn: xanes3D_trial_reg_results.h5['reg_results/xxx']
            shift_idn: used specified id in
                       xanes3D_trial_reg_results.h5['reg_results/shiftxxx']
        sli_s:      int
                    starting slice id of the volume to be shifted
        sli_e:      int
                    ending slice id of the volume to be shifted
        trialfn:    string; optional
                    trial registration filename
        savefn:     string; optional
                    filename to the file in which the shifted volume to be saved
        optional_shift_dict: dict; optional
                    user input shifts for specified scan ids. This is useful to
                    correct individual pairs that cannot be aligned with others
                    with the same registration method
        """
        if savefn is None:
            savefn = self.savefn
        if trialfn is None:
            trialfn = self.savefn

        if savefn == trialfn:
            with h5py.File(savefn, 'a') as f:
                if 'registration_results' not in f:
                    g0 = f.create_group('registration_results')
                else:
                    del f['registration_results']
                    g0 = f.create_group('registration_results')

                g1 = g0.create_group('reg_parameters')
                self.alignment_pair_list = f['/trial_registration/trial_reg_parameters/alignment_pairs'][:].tolist()
                g1.create_dataset('alignment_pairs', data=self.alignment_pair_list)
                self.img_ids = f['/trial_registration/trial_reg_parameters/scan_ids'][:]
                g1.create_dataset('scan_ids', data=self.img_ids)
                g1.create_dataset('slice_roi', data=self.roi)
                self.anchor_id = f['/trial_registration/trial_reg_parameters/fixed_scan_id'][()]
                g1.create_dataset('fixed_scan_id', data=self.anchor_id)
                self.chunk_sz = f['/trial_registration/trial_reg_parameters/chunk_sz'][()]
                g1.create_dataset('chunk_sz', data=self.chunk_sz)
                self.method = f['/trial_registration/trial_reg_parameters/reg_method'][()].decode("utf-8")
                g1.create_dataset('reg_method', data=self.method)
                self.ref_mode = f['/trial_registration/trial_reg_parameters/reg_ref_mode'][()].decode("utf-8")
                g1.create_dataset('reg_ref_mode', data=str(self.ref_mode.upper()))
                self.xanes3D_sli_search_half_range = f['/trial_registration/trial_reg_parameters/sli_search_half_range'][()]
                self.img_ids_dict = h5todict(savefn, path='/trial_registration/trial_reg_parameters/scan_ids_dict')
                self.eng_dict = h5todict(savefn, path='/trial_registration/trial_reg_parameters/eng_dict')
                dicttoh5(self.eng_dict, savefn, mode='a',
                     overwrite_data=True,
                     h5path='/registration_results/reg_parameters/eng_dict')
                dicttoh5(self.img_ids_dict, savefn, mode='a',
                     overwrite_data=True,
                     h5path='/registration_results/reg_parameters/scan_ids_dict')

                self._sort_absolute_shift(trialfn, shift_dict=shift_dict, optional_shift_dict=optional_shift_dict)
                dicttoh5(self.abs_shift_dict, savefn, mode='a',
                         overwrite_data=True,
                         h5path='/registration_results/reg_parameters/user_determined_shift/user_input_shift_lookup')

                shift = {}
                slioff = {}
                for key in sorted(self.abs_shift_dict.keys()):
                    shift[str(key).zfill(3)] = self.abs_shift_dict[key]['in_sli_shift']
                    slioff[str(key).zfill(3)] = self.abs_shift_dict[key]['out_sli_shift']

                dicttoh5(shift, savefn, mode='a',
                         overwrite_data=True,
                         h5path='/registration_results/reg_parameters/user_determined_shift/absolute_in_slice_shift')
                dicttoh5(slioff, savefn, mode='a',
                         overwrite_data=True,
                         h5path='/registration_results/reg_parameters/user_determined_shift/absolute_out_slice_shift')

                g2 = g0.create_group('reg_results')
                g21 = g2.create_dataset('registered_xanes3D',
                                        shape=(len(self.img_ids_dict),
                                               sli_e-sli_s+1,
                                               self.roi[1]-self.roi[0],
                                               self.roi[3]-self.roi[2]))
                g22 = g2.create_dataset('eng_list', shape=(len(self.img_ids_dict),))

                img = np.ndarray([self.roi[1]-self.roi[0], self.roi[3]-self.roi[2]])
                cnt1 = 0
                for key in sorted(self.abs_shift_dict.keys()):
                    shift = self.abs_shift_dict[key]['in_sli_shift']
                    slioff = self.abs_shift_dict[key]['out_sli_shift']
                    scan_id = self.img_ids_dict[key]
                    cnt2 = 0
                    print(self.method)
                    if self.method == 'SR':
                        yshift_int = int(shift[0, 2])
                        xshift_int = int(shift[1, 2])
                        shift[0, 2] -= yshift_int
                        shift[1, 2] -= xshift_int
                    elif self.method in ['PC', 'MPC', 'MRTV', 'MPC+MRTV']:
                        yshift_int = int(shift[0])
                        xshift_int = int(shift[1])
                        shift[0] -= yshift_int
                        shift[1] -= xshift_int

                    # for ii in range(int(sli_s+slioff), int(sli_e+slioff)):
                    #     # print(4)
                    #     fn = self.xanes3D_recon_path_template.format(scan_id,
                    #                                                  str(ii).zfill(5))
                    #     img[:] = tifffile.imread(fn)[self.roi[0]-yshift_int:self.roi[1]-yshift_int,
                    #                                  self.roi[2]-xshift_int:self.roi[3]-xshift_int]
                    #
                    #     self._translate_single_img(img, shift, self.method)
                    #     g21[cnt1, cnt2] = img[:]
                    #     cnt2 += 1
                    # g22[cnt1] = self.eng_dict[key]
                    # cnt1 += 1
                    bdi, num_batch = self._chunking([sli_e-sli_s+1, self.roi[1]-self.roi[0], self.roi[3]-self.roi[2]],
                                                    mem_lim=mem_lim)
                    for i in range(num_batch):
                        img = tiff_vol_reader(self.xanes3D_recon_path_template, scan_id,
                                              [sli_s+slioff+bdi[i*N_CPU], sli_s+slioff+bdi[(i+1)*N_CPU],
                                               self.roi[0]-yshift_int, self.roi[1]-yshift_int,
                                               self.roi[2]-xshift_int, self.roi[3]-xshift_int])
                        print(f"{img.shape=}")
                        with mp.Pool(N_CPU) as pool:
                            rlt = pool.map(partial(self._translate_vol_img, shift, self.method), [
                                           img[bdi[i*N_CPU+ii-1]:bdi[i*N_CPU+ii], ...] for ii in range(N_CPU)])
                        pool.close()
                        pool.join()
                        rlt = np.array(rlt, dtype=object)
                        print(f"{g21[cnt1, bdi[i*N_CPU]:bdi[(i+1)*N_CPU], ...].shape=}, {img.shape=}")
                        g21[cnt1, bdi[i*N_CPU]:bdi[(i+1)*N_CPU], ...] = img[:]
                    g22[cnt1] = self.eng_dict[key]
                    cnt1 += 1

    def save_reg_result(self, dtype='2D_XANES', data=None):
        if dtype.upper() == '2D_XANES':
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path, mode=777)
            print(f'3. The registration results will be saved in {self.savefn}')

            f = h5py.File(self.savefn, 'a')
            if 'trial_registration' not in f:
                g1 = f.create_group('trial_registration')
            else:
                g1 = f['trial_registration']

            if 'method' not in g1:
                dset = g1.create_dataset('method', data=str(self.method))
                dset.attrs['method'] = str(self.method)
            else:
                del g1['method']
                dset = g1.create_dataset('method', data=str(self.method))
                dset.attrs['method'] = str(self.method)

            if 'mode' not in g1:
                dset = g1.create_dataset('mode', data=str(self.mode))
                dset.attrs['mode'] = str(self.mode)
            else:
                del g1['mode']
                dset = g1.create_dataset('mode', data=str(self.mode))
                dset.attrs['mode'] = str(self.mode)

            if 'ref_mode' not in g1:
                dset = g1.create_dataset('ref_mode', data=str(self.ref_mode))
                dset.attrs['ref_mode'] = str(self.ref_mode)
            else:
                del g1['ref_mode']
                dset = g1.create_dataset('ref_mode', data=str(self.ref_mode))
                dset.attrs['ref_mode'] = str(self.ref_mode)

            if 'registered_image' not in g1:
                if data is None:
                    g1.create_dataset('registered_image', data=self.img)
                else:
                    g1.create_dataset('registered_image', data=data)
            else:
                del g1['registered_image']
                if data is None:
                    g1.create_dataset('registered_image', data=self.img)
                else:
                    g1.create_dataset('registered_image', data=data)

            if 'shift' not in g1:
                g1.create_dataset('shift', data=self.shift)
            else:
                del g1['shift']
                g1.create_dataset('shift', data=self.shift)

            if 'ssim' not in g1:
                g1.create_dataset('ssim', data=self.si)
            else:
                del g1['ssim']
                g1.create_dataset('ssim', data=self.si)

            if 'mse' not in g1:
                g1.create_dataset('mse', data=self.mse)
            else:
                del g1['mse']
                g1.create_dataset('mse', data=self.mse)

            if 'nrmse' not in g1:
                g1.create_dataset('nrmse', data=self.nrmse)
            else:
                del g1['nrmse']
                g1.create_dataset('nrmse', data=self.nrmse)

            if 'raw_data_info' not in f:
                g2 = f.create_group('raw_data_info')
            else:
                g2 = f['raw_data_info']

            for key, val in self.raw_data_info.items():
                if key not in g2:
                    g2.create_dataset(key, data=val)
                else:
                    del g2[key]
                    g2.create_dataset(key, data=val)
            f.close()
        elif dtype.upper() == '3D_XANES':
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path, mode=777)
            print(f'4. The registration results will be saved in {self.savefn}')

            f = h5py.File(self.savefn, 'a')
            if 'registration' not in f:
                g1 = f.create_group('registration')
            else:
                g1 = f['registration']

            if 'method' not in g1:
                dset = g1.create_dataset('method', data=str(self.method))
                dset.attrs['method'] = str(self.method)
            else:
                del g1['method']
                dset = g1.create_dataset('method', data=str(self.method))
                dset.attrs['method'] = str(self.method)

            if 'registered_image' not in g1:
                if data is None:
                    g1.create_dataset('registered_image', data=self.img)
                else:
                    g1.create_dataset('registered_image', data=data)
            else:
                del g1['registered_image']
                if data is None:
                    g1.create_dataset('registered_image', data=self.img)
                else:
                    g1.create_dataset('registered_image', data=data)

            if 'residual_image' not in g1:
                if data is None:
                    g1.create_dataset('residual_image',
                                      data=np.float32(self.fixed)-self.img)
                else:
                    g1.create_dataset('residual_image',
                                      data=np.float32(self.fixed)-data)
            else:
                del g1['residual_image']
                if data is None:
                    g1.create_dataset('residual_image',
                                      data=np.float32(self.fixed)-self.img)
                else:
                    g1.create_dataset('residual_image',
                                      data=np.float32(self.fixed)-data)

            if 'shift' not in g1:
                g1.create_dataset('shift', data=self.shift)
            else:
                del g1['shift']
                g1.create_dataset('shift', data=self.shift)

            if 'error' not in g1:
                g1.create_dataset('error', data=self.error)
            else:
                del g1['error']
                g1.create_dataset('error', data=self.error)

            if 'ssim' not in g1:
                g1.create_dataset('ssim', data=self.si)
            else:
                del g1['ssim']
                g1.create_dataset('ssim', data=self.si)

            if 'mse' not in g1:
                g1.create_dataset('mse', data=self.mse)
            else:
                del g1['mse']
                g1.create_dataset('mse', data=self.mse)

            if 'nrmse' not in g1:
                g1.create_dataset('nrmse', data=self.nrmse)
            else:
                del g1['nrmse']
                g1.create_dataset('nrmse', data=self.nrmse)

            if 'raw_data_info' not in f:
                g2 = f.create_group('raw_data_info')
            else:
                g2 = f['raw_data_info']

            for key, val in self.raw_data_info.items():
                if key not in g2:
                    g2.create_dataset(key, data=val)
                else:
                    del g2[key]
                    g2.create_dataset(key, data=val)
            f.close()
        else:
            print("'dtype' can only be '2D_XANES' or '3D_XANES'. Quit!")

    def _translate_single_img(self, img, shift, method):
        if method.upper() in ['PC', 'MPC', 'MRTV', 'MPC+MRTV']:
            img[:] = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)))[:]
        elif method == 'SR':
            if self.mode.upper() == 'TRANSLATION':
                sr = StackReg(StackReg.TRANSLATION)
            elif  self.mode.upper() == 'RIGID_BODY':
                sr = StackReg(StackReg.RIGID_BODY)
            elif  self.mode.upper() == 'SCALED_ROTATION':
                sr = StackReg(StackReg.SCALED_ROTATION)
            elif  self.mode.upper() == 'AFFINE':
                sr = StackReg(StackReg.AFFINE)
            elif  self.mode.upper() == 'BILINEAR':
                sr = StackReg(StackReg.BILINEAR)
            img[:] = sr.transform(img, tmat = shift)[:]
        else:
            print('Nonrecognized method. Quit!')
            # exit()

    def _translate_vol_img(self, shift, method, img):
        for ii in range(img.shape[0]):
            self._translate_single_img(img[ii], shift, method)

    def xanes3D_shell_slicing(self, fn):
        pass
