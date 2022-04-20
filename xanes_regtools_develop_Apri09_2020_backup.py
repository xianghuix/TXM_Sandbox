#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:09:13 2019

@author: xiao
"""

#import faulthandler; faulthandler.enable()
from pystackreg import StackReg
import tifffile, os, h5py
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from skimage.measure import compare_nrmse as nrmse
from skimage.feature import register_translation, masked_register_translation
from scipy.ndimage import fourier_shift
import numpy as np
from xanes_math import fit_poly1d
from copy import deepcopy
from silx.io.dictdump import dicttoh5, h5todict

__all__ = ['regtools']


class regtools():
    def __init__(self, dtype='2D_XANES', **kwargs):
        if dtype not in ['2D_XANES', '3D_XANES']:
            print("Wrong input data_type ...")
            # exit()

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
        self.moving = None
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
        self.use_anchor = False

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
            self.method = 'mpc'
            self.overlap_ratio = 0.3
            print('Registration method is set to default phase-correlation \
                  method.')

        if self.method == 'sr':
            if 'mode' in kwargs:
                self.mode = kwargs['mode']
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

    def set_anchor(self, anchor):
        self.anchor = anchor

    def set_xanes3D_raw_h5_top_dir(self, raw_h5_top_dir):
        self.xanes3D_raw_h5_top_dir = raw_h5_top_dir

    def set_xanes3D_raw_data_info(self, **kwargs):
        self.raw_data_info = kwargs

    def set_xanes2D_raw_filename(self, filename):
        self.xanes2D_raw_filename = filename

    def set_cal_anchor(self, img_id_s, img_id_e, fixed_img_id):
        """
        Parameters
        ----------
        img_id_s : integer
            if dtype = '3D_XANES', img_id_s is the starting scan id;
            if dtype = '2D_XANES', img_id_s is the starting image index.
        img_id_e : integer
            if dtype = '3D_XANES', img_id_e is the ending scan id;
            if dtype = '2D_XANES', img_id_e is the ending image index.
        fixed_img_id : integer
            if dtype = '3D_XANES', fixed_img_id is the anchor scan id;
            if dtype = '2D_XANES', fixed_img_id is the anchor image index.
        dtype : string, optional
            type of analysis in ['2D_XANES', '3D_XANES']. The default is '3D_XANES'.

        Returns
        -------
        None.
        """
        self.img_id_s = img_id_s
        self.img_id_e = img_id_e
        self.fixed_img_id = fixed_img_id

    def use_anchor(self):
        self.use_anchor = True

    # def set_fixed_data(self, fixed):
    #     self.fixed = fixed

    def set_method(self, method):
        self.method = method

    def set_ref_mode(self, ref_mode):
        self.ref_mode = ref_mode

    # def set_moving_data(self, moving):
    #     self.moving = moving

    # def set_data(self, fixed, moving):
        # self.fixed = fixed
        # self.moving = moving

    def set_xanes3D_data_pnts(self, data_pnts):
        self.data_pnts = data_pnts

    def set_xanes3D_recon_path_template(self, path_template):
        self.xanes3D_recon_path_template = path_template

    def set_mask(self, mask):
        self.mask = mask

    def set_saving(self, save_path, fn=None):
        if save_path is not None:
            self.save_path = save_path
        if fn is None:
            fn = 'regisstration_results.h5'
        self.savefn = os.path.join(self.save_path, fn)
        print('The registration results will be saved in {:s}'
              .format(self.savefn))

    def compose_dicts(self):
        if self.data_type == '3D_XANES':
            if self.fixed_img_id in range(self.img_id_s, self.img_id_e+1):
                self.anchor = self.fixed_img_id - self.img_id_s
                self.data_pnts = self.img_id_e - self.img_id_s + 1
                self.img_ids = np.arange(self.img_id_s, self.img_id_e+1)
                self.anchor_id = self.fixed_img_id
                self.use_anchor = True
                self.eng_list = []
                if self.xanes3D_raw_h5_top_dir is not None:
                    print(self.xanes3D_raw_h5_top_dir)
                    cnt = 0
                    for ii in range(self.img_id_s, self.img_id_e+1):
                        fn = os.path.join(self.xanes3D_raw_h5_top_dir, 'fly_scan_id_{}.h5'.format(ii))
                        f = h5py.File(fn, 'r')
                        self.eng_list.append(f['X_eng'][()])
                        f.close()
                        self.eng_dict[str(cnt).zfill(3)] = self.eng_list[-1]
                        self.img_ids_dict[str(cnt).zfill(3)] = self.img_ids[cnt]
                        cnt += 1
                else:
                    print('xanes3D raw h5 top dir is not defined.')
                    # exit()
            else:
                print('fixed_img_id is outside of [img_id_s, img_id_e]. use_anchor\
                       is not supported.')
                self.use_anchor = False
        else:
            if self.fixed_img_id in range(self.img_id_s, self.img_id_e+1):
                self.anchor = self.fixed_img_id - self.img_id_s
                self.data_pnts = self.img_id_e - self.img_id_s + 1
                self.img_ids = np.arange(self.img_id_s, self.img_id_e+1)
                self.anchor_id = self.fixed_img_id
                self.use_anchor = True
                if self.xanes2D_raw_filename is not None:
                    print(self.xanes2D_raw_filename)
                    f = h5py.File(self.xanes2D_raw_filename, 'r')
                    self.eng_list = f['/X_eng'][self.img_id_s:self.img_id_e+1]
                    f.close()
                cnt = 0
                for ii in range(self.img_id_s, self.img_id_e+1):
                    self.eng_dict[int(cnt).zfill(3)] = self.eng_list[ii-self.img_id_s]
                    self.img_ids_dict[str(cnt).zfill(3)] = ii
                    cnt += 1
                else:
                    print('xanes3D raw h5 top dir is not defined.')
                    # exit()
            else:
                print('fixed_img_id is outside of [img_id_s, img_id_e]. use_anchor\
                       is not supported.')
                self.use_anchor = False

    def _chunking(self):
        if self.use_anchor:
            right_num_chunk = int(np.ceil((self.data_pnts -
                                           self.anchor) / self.chunk_sz))
            left_num_chunk = int(np.ceil(self.anchor / self.chunk_sz))
            num_chunk = left_num_chunk + right_num_chunk
            self.num_chunk = num_chunk
            # number of chunks before the anchor chunk
            self.left_num_chunk = left_num_chunk - 1
            # the index of anchor chunk
            self.anchor_chunk = left_num_chunk - 1
            # chunks = np.ndarray(num_chunk, dtype=np.int)
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
                self.chunks[0]['chunk_e'] = self.chunk_sz-1

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
            num_chunk = int(np.ceil(self.data_pnts/self.chunk_sz))
            for ii in range(num_chunk-1):
                self.chunks[ii] = {'chunk_s': ii*self.chunk_sz}
                self.chunks[ii]['chunk_e'] = (ii+1)*self.chunk_sz - 1
            self.chunks[num_chunk-1] = {'chunk_s':\
                self.chunks[num_chunk-2]['chunk_e']+1}
            self.chunks[num_chunk-1]['chunk_e'] = self.data_pnts - 1

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
        """
        self._chunking()
        self.alignment_pair_list = []

        if self.use_anchor:
            if self.ref_mode.upper() == 'SINGLE':
                # inter-chunk alignment pair
                for ii in range(self.left_num_chunk):
                    self.alignment_pair_list.append([self.chunks[self.left_num_chunk-ii]['chunk_e'],
                                                self.chunks[self.left_num_chunk-ii-1]['chunk_e']])
                self.alignment_pair_list.append([self.anchor_chunk,
                                            self.anchor_chunk])
                for ii in range(self.left_num_chunk+1, self.num_chunk):
                    self.alignment_pair_list.append([self.chunks[ii-1]['chunk_e'],
                                                self.chunks[ii]['chunk_e']])
                # intra-chunk alignment pair
                for ii in range(self.num_chunk):
                    for jj in range(self.chunks[ii]['chunk_s'],
                                    self.chunks[ii]['chunk_e']+1):
                        self.alignment_pair_list.append([self.chunks[ii]['chunk_e'], jj])

                for ii in self.alignment_pair_list:
                    if ii[0] == ii[1]:
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

                for ii in self.alignment_pair_list:
                    if ii[0] == ii[1]:
                        self.alignment_pair_list.remove(ii)
                self.alignment_pair_list.append([self.anchor, self.anchor])

    def _sort_absolute_shift(self, trialfn, shift_dict=None, optional_shift_dict=None):
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
                    shift = 1
                    slioff = 0
                    for ii in range(len(item)-1):
                        idx = self.alignment_pair_list.index([item[ii+1], item[ii]])
                        if optional_shift_dict is None:
                            h5path = '/trial_registration/trial_reg_results/'+str(idx).zfill(3)+'/shift'+str(idx).zfill(3)
                            shift *= f[h5path][shift_dict[str(idx)]]
                            slioff += (shift_dict[str(idx)]-self.xanes3D_sli_search_half_range)
                        else:
                            if str(idx) in optional_shift_dict.keys():
                                shift *= optional_shift_dict[str(idx)][1:]
                                slioff += (optional_shift_dict[str(idx)][0]-self.xanes3D_sli_search_half_range)
                            else:
                                h5path = '/trial_registration/trial_reg_results/'+str(idx).zfill(3)+'/shift'+str(idx).zfill(3)
                                shift *= f[h5path][shift_dict[str(idx)]]
                                slioff += (shift_dict[str(idx)]-self.xanes3D_sli_search_half_range)
                    # abs_shift_dict[str(key).zfill(3)] = [shift, slioff]
                    abs_shift_dict[str(key).zfill(3)] = {'in_sli_shift':shift, 'out_sli_shift':slioff}
            else:
                for key, item in self.shift_chain_dict.items():
                    item.insert(0, key)
                    print(key, item)
                    shift = 0
                    slioff = 0
                    for ii in range(len(item)-1):
                        idx = self.alignment_pair_list.index([item[ii+1], item[ii]])
                        if optional_shift_dict is None:
                            h5path = '/trial_registration/trial_reg_results/'+str(idx).zfill(3)+'/shift'+str(idx).zfill(3)
                            shift += f[h5path][shift_dict[str(idx)]]
                            slioff += (shift_dict[str(idx)]-self.xanes3D_sli_search_half_range)
                        else:
                            if str(idx) in optional_shift_dict.keys():
                                shift += optional_shift_dict[str(idx)][1:]
                                slioff += (optional_shift_dict[str(idx)][0]-self.xanes3D_sli_search_half_range)
                            else:
                                h5path = '/trial_registration/trial_reg_results/'+str(idx).zfill(3)+'/shift'+str(idx).zfill(3)
                                shift += f[h5path][shift_dict[str(idx)]]
                                slioff += (shift_dict[str(idx)]-self.xanes3D_sli_search_half_range)
                    # abs_shift_dict[str(key).zfill(3)] = [shift, slioff]
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
                    shift = 1
                    for ii in range(len(item)-1):
                        idx = self.alignment_pair_list.index([item[ii+1], item[ii]])
                        if optional_shift_dict is None:
                            h5path = '/trial_registration/trial_reg_results/shifts'
                            shift *= f[h5path][idx]
                        else:
                            if str(idx) in optional_shift_dict.keys():
                                shift *= optional_shift_dict[str(idx)]
                            else:
                                h5path = '/trial_registration/trial_reg_results/shifts'
                                shift *= f[h5path][idx]
                    # abs_shift_dict[str(key).zfill(3)] = shift
                    abs_shift_dict[str(key).zfill(3)] = {'in_sli_shift':shift}
            else:
                for key, item in self.shift_chain_dict.items():
                    item.insert(0, key)
                    print(key, item)
                    shift = 0
                    for ii in range(len(item)-1):
                        idx = self.alignment_pair_list.index([item[ii+1], item[ii]])
                        if optional_shift_dict is None:
                            h5path = '/trial_registration/trial_reg_results/shifts'
                            shift += f[h5path][idx]
                        else:
                            if str(idx) in optional_shift_dict.keys():
                                shift += optional_shift_dict[str(idx)]
                            else:
                                h5path = '/trial_registration/trial_reg_results/shifts'
                                shift += f[h5path][idx]
                    # abs_shift_dict[str(key).zfill(3)] = shift
                    abs_shift_dict[str(key).zfill(3)] = {'in_sli_shift':shift}
            f.close()
        self.abs_shift_dict = abs_shift_dict

    def _reg_xanes2D_chunk(self):
        """
        chunk_sz: int, number of image in one chunk for alignment; each chunk
                  use the last image in that chunk as reference
        method:   str
                  'pc':   skimage.feature.register_translation
                  'mpc':  skimage.feature.masked_register_translation
                  'sr':   pystackreg.StackReg
        overlap_ratio: float, overlap_ratio for method == 'mpc'
        ref_mode: str, control how inter-chunk alignment is done
                  'average': the average of each chunk after intra-chunk
                             re-alignment is used for inter-chunk alignment
                  'single':  the last image in each chunk is used in
                             inter-chunk alignment
        """
        self._alignment_scheduler(dtype='2D_XANES')

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, mode=777)
            print(f'The registration results will be saved in {self.savefn}')

        f = h5py.File(self.savefn, 'a')
        if 'trial_registration' not in f:
            g0 = f.create_group('trial_registration')
        else:
            del f['trial_registration']
            g0 = f['trial_registration']

        g1 = g0.create_group('trial_reg_results')
        g2 = g0.create_group('trial_reg_parameters')

        g2.create_dataset('reg_method', data=str(self.method.upper()))
        g2.create_dataset('reg_ref_mode', data=str(self.ref_mode.upper()))

        g2.create_dataset('alignment_pairs', data=self.alignment_pair_list)
        g2.create_dataset('scan_ids', data=self.img_ids)
        g2.create_dataset('fixed_scan_id', data=self.anchor_id)
        g2.create_dataset('slice_roi', data=self.roi)
        g2.create_dataset('eng_list', data=self.eng_list)
        dicttoh5(self.eng_dict, self.savefn, mode='a',
                 overwrite_data=True,
                 h5path='/trial_registration/trial_reg_parameters/eng_dict')
        dicttoh5(self.img_ids_dict, self.savefn, mode='a',
                 overwrite_data=True,
                 h5path='/trial_registration/trial_reg_parameters/scan_ids_dict')
        g2.create_dataset('chunk_sz', data=self.chunk_sz)
        g2.create_dataset('use_mask', data=self.use_mask)
        if self.use_mask:
            g2.create_dataset('mask', data=self.mask)

        g3 = g0.create_group('data_directory_info')
        for key, val in self.raw_data_info.items():
            g3.create_dataset(key, data=val)

        shifted_image = np.ndarray(self.moving.shape)

        if self.moving.ndim != 3:
                print('XANES2D image stack is required. Please set XANES2D \
                      image stack first.')
                # exit()
        else:
            if self.method.upper() in {'PC', 'MPC'}:
                self.shift = np.ndarray([len(self.alignment_pair_list), 2])
            else:
                self.shift = np.ndarray([len(self.alignment_pair_list), 3, 3])

            self.error = np.ndarray(len(self.alignment_pair_list))
            self.si = np.ndarray(len(self.alignment_pair_list))
            self.mse = np.ndarray(len(self.alignment_pair_list))
            self.nrmse = np.ndarray(len(self.alignment_pair_list))

            if self.method.upper() == 'PC':
                print('We are using "phase correlation" method for registration.')
                if self.mask is not None:
                    for ii in range(len(self.alignment_pair_list)):
                        self.shift[ii], self.error[ii], _ = register_translation(
                                self.moving[self.alignment_pair_list[ii][0]]*self.mask[0],
                                self.moving[self.alignment_pair_list[ii][1]]*self.mask[0], 100)
                        shifted_image[ii] = np.real(np.fft.ifftn(fourier_shift(
                                np.fft.fftn(self.moving[self.alignment_pair_list[ii][1]]), self.shift[ii])))[:]
                        print(ii, self.shift[ii])
                else:
                    for ii in range(len(self.alignment_pair_list)):
                        self.shift[ii], self.error[ii], _ = register_translation(
                                self.moving[self.alignment_pair_list[ii][0]],
                                self.moving[self.alignment_pair_list[ii][1]], 100)
                        shifted_image[ii] = np.real(np.fft.ifftn(fourier_shift(
                                np.fft.fftn(self.moving[self.alignment_pair_list[ii][1]]), self.shift[ii])))[:]
                        print(ii, self.shift[ii])

            elif self.method.upper() == 'MPC':
                print('We are using "masked phase correlation" method for registration.')
                print(len(self.alignment_pair_list))
                for ii in range(len(self.alignment_pair_list)):
                    self.shift[ii] = masked_register_translation(self.moving[self.alignment_pair_list[ii][0]],
                                                                 self.moving[self.alignment_pair_list[ii][1]], self.mask[0],
                                                                 overlap_ratio=self.overlap_ratio)
                    shifted_image[ii] = np.real(np.fft.ifftn(fourier_shift(
                            np.fft.fftn(self.moving[self.alignment_pair_list[ii][1]]), self.shift[ii])))[:]
                    print(ii, self.shift[ii])

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
                        self.shift[ii] = sr.register(self.moving[self.alignment_pair_list[ii][0]]*self.mask[0],
                                                     self.moving[self.alignment_pair_list[ii][1]]*self.mask[0])
                        shifted_image[ii] = sr.transform(self.moving[self.alignment_pair_list[ii][1]],
                                                                               tmat=self.shift[ii])[:]
                        print(ii, self.shift[ii])
                else:
                    for ii in range(len(self.alignment_pair_list)):
                        self.shift[ii] = sr.register(self.moving[self.alignment_pair_list[ii][0]],
                                                     self.moving[self.alignment_pair_list[ii][1]])
                        shifted_image[ii] = sr.transform(self.moving[self.alignment_pair_list[ii][1]],
                                                                               tmat=self.shift[ii])[:]
                        print(ii, self.shift[ii])
        g1.create_dataset('trial_reg_img', data=np.float32(self.fixed-shifted_image))
        g1.create_dataset('shifts', data=np.float32(self.shift))
        f.close()

    def reg_xanes2D_chunk(self, overlap_ratio=0.3, method=None,
                          ref_mode=None, is_raw=False):
        """

        """
        if not is_raw:
            f = h5py.File(self.xanes2D_raw_filename, 'r')
            self.fixed = f['img_xanes'][self.anchor_id, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
            f.close()
        else:
            f = h5py.File(self.xanes2D_raw_filename, 'r')
            self.fixed = ((f['img_xanes'][self.anchor_id, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] -
                          f['img_dark'][self.anchor_id, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]) /
                         (f['img_flat'][self.anchor_id, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] -
                          f['img_dark'][self.anchor_id, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]))
            self.fixed[np.isnan(self.fixed)] = 0
            self.fixed[np.isinf(self.fixed)] = 0
            f.close()
        self.moving = np.ndarray([self.img_id_e-self.img_id_s,
                                  self.roi[1]-self.roi[0],
                                  self.roi[3]-self.roi[2]])
        if ref_mode is None:
            ref_mode = 'neighbor'
        elif ref_mode.upper() not in ['SINGLE', 'NEIGHBOR', 'AVERAGE']:
            print('"ref_mode" can only be "single", "neighbor", or \
                  "average". Quit!')
            # exit()

        if method is None:
            ref_mode = 'mpc'
        elif method.upper() not in ['PC', 'MPC', 'SR']:
            print('"method" can only be "pc", "mpc", or "sr". Quit!')
            # exit()

        self.method = method
        self.ref_mode = ref_mode
        self.overlap_ratio = 0.3
        self._reg_xanes2D_chunk()
        # if save is True:
        #     self.save_reg_result(dtype='2D_XANES')
        # print(np.where(self.si.max()), np.where(self.error.min()))
        print('Done!')

    def apply_xanes2D_chunk_shift(self, trialfn=None, savefn=None, optional_shift_dict=None):
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
            f = h5py.File(savefn, 'a')
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
            self.roi = f['/trial_registration/trial_reg_parameters/slice_roi'][:]
            g1.create_dataset('slice_roi', data=self.roi)
            self.anchor_id = f['/trial_registration/trial_reg_parameters/fixed_scan_id'][()]
            g1.create_dataset('fixed_scan_id', data=self.anchor_id)
            self.chunk_sz = f['/trial_registration/trial_reg_parameters/chunk_sz'][()]
            g1.create_dataset('chunk_sz', data=self.chunk_sz)
            self.method = f['/trial_registration/trial_reg_parameters/reg_method'][()]
            g1.create_dataset('reg_method', data=self.method)
            self.ref_mode = f['/trial_registration/trial_reg_parameters/reg_ref_mode'][()]
            g1.create_dataset('reg_ref_mode', data=str(self.ref_mode.upper()))
            self.img_ids_dict = h5todict(savefn, path='/trial_registration/trial_reg_parameters/scan_ids_dict')
            self.eng_dict = h5todict(savefn, path='/trial_registration/trial_reg_parameters/eng_dict')
            dicttoh5(self.eng_dict, savefn, mode='a',
                 overwrite_data=True,
                 h5path='/registration_results/reg_parameters/eng_dict')
            dicttoh5(self.img_ids_dict, savefn, mode='a',
                 overwrite_data=True,
                 h5path='/registration_results/reg_parameters/scan_ids_dict')

            if optional_shift_dict is not None:
                dicttoh5(optional_shift_dict, savefn, mode='a', overwrite_data=True,
                         h5path='/registration_results/reg_parameters/user_determined_shift/optional_shift')
            else:
                g1.create_dataset('optional_shift', data=str('None'))

            self._sort_absolute_shift(trialfn, shift_dict=None, optional_shift_dict=optional_shift_dict)

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

            img = np.ndarray([self.roi[1]-self.roi[0], self.roi[3]-self.roi[2]])

            cnt1 = 0
            for key in sorted(self.abs_shift_dict.keys()):
                shift = self.abs_shift_dict[key]['in_sli_shift']
                if self.method == 'SR':
                    yshift_int = int(shift[0, 2])
                    xshift_int = int(shift[1, 2])
                    shift[0, 2] -= yshift_int
                    shift[1, 2] -= xshift_int
                elif self.method in ['PC', 'MPC']:
                    yshift_int = int(shift[0])
                    xshift_int = int(shift[1])
                    shift[0] -= yshift_int
                    shift[1] -= xshift_int

                self._translate_single_img(self.moving[int(key)][self.roi[0]-yshift_int:self.roi[1]-yshift_int,
                                                                 self.roi[2]-xshift_int:self.roi[3]-xshift_int],
                                           shift, self.method)
                print(shift)
                g21[cnt1] = img[:]
                g22[cnt1] = self.eng_dic[key]
                cnt1 += 1
            f.close()

    def _reg_xanes3D_chunk(self):
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
        self._alignment_scheduler(dtype='3D_XANES')

        f = h5py.File(self.savefn, 'a')
        if 'trial_registration' not in f:
            g0 = f.create_group('trial_registration')
        else:
            del f['trial_registration']
            g0 = f.create_group('trial_registration')


        g1 = g0.create_group('trial_reg_results')

        g2 = g0.create_group('trial_reg_parameters')
        g2.create_dataset('alignment_pairs', data=self.alignment_pair_list)
        g2.create_dataset('scan_ids', data=self.img_ids)
        g2.create_dataset('fixed_scan_id', data=self.anchor_id)
        g2.create_dataset('slice_roi', data=self.roi)
        g2.create_dataset('fixed_slice', data=self.xanes3D_recon_fixed_sli)
        g2.create_dataset('sli_search_half_range',
                          data=self.xanes3D_sli_search_half_range)
        g2.create_dataset('eng_list', data=self.eng_list)
        g2.create_dataset('chunk_sz', data=self.chunk_sz)
        g2.create_dataset('use_mask', data=self.use_mask)
        if self.use_mask:
            g2.create_dataset('mask', data=self.mask)
        g2.create_dataset('reg_method', data=str(self.method.upper()))
        g2.create_dataset('reg_ref_mode', data=str(self.ref_mode.upper()))
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

        if self.method.upper() in {'PC', 'MPC'}:
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

        print(self.method.upper())
        if self.method.upper() == 'PC':
            print('We are using "phase correlation" method for registration.')
            if self.mask is not None:
                for ii in range(len(self.alignment_pair_list)):
                    fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][0]).zfill(3)],
                                                                 str(self.xanes3D_recon_fixed_sli).zfill(5))
                    self.fixed[:] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                    jj_id = 0
                    for jj in range(self.xanes3D_recon_fixed_sli-self.xanes3D_sli_search_half_range,
                                    self.xanes3D_recon_fixed_sli+self.xanes3D_sli_search_half_range):
                        fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][1]).zfill(3)], str(jj).zfill(5))
                        self.moving[jj_id] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

                        self.shift[ii, jj_id], self.error[ii, jj_id], _ = register_translation(self.fixed*self.mask,
                                                                                 self.moving[jj_id]*self.mask, 100)
                        self.moving[jj_id] = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(self.moving[jj_id]),
                                                                                         self.shift[ii, jj_id])))[:]
                        print(self.shift[ii, jj_id])
                        jj_id += 1
                    g11 = g1.create_group(str(ii).zfill(3))
                    g11.create_dataset('shift'+str(ii).zfill(3),
                                       data=self.shift[ii])
                    g11.create_dataset('error'+str(ii).zfill(3),
                                       data=self.error[ii])
                    g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                       data=self.moving-self.fixed)
            else:
                for ii in range(len(self.alignment_pair_list)):
                    fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][0]).zfill(3)],
                                                                 str(self.xanes3D_recon_fixed_sli).zfill(5))
                    self.fixed[:] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                    jj_id = 0
                    for jj in range(self.xanes3D_recon_fixed_sli-self.xanes3D_sli_search_half_range,
                                    self.xanes3D_recon_fixed_sli+self.xanes3D_sli_search_half_range):
                        fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][1]).zfill(3)], str(jj).zfill(5))
                        self.moving[jj_id] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

                        self.shift[ii, jj_id], self.error[ii, jj_id], _ = register_translation(self.fixed,
                                                                                     self.moving[jj_id], 100)
                        self.moving[jj_id] = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(self.moving[jj_id]),
                                                                                         self.shift[ii, jj_id])))[:]
                        print(self.shift[ii, jj_id])
                        jj_id += 1
                    g11 = g1.create_group(str(ii).zfill(3))
                    g11.create_dataset('shift'+str(ii).zfill(3),
                                       data=self.shift[ii])
                    g11.create_dataset('error'+str(ii).zfill(3),
                                       data=self.error[ii])
                    g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                       data=self.moving-self.fixed)

        elif self.method.upper() == 'MPC':
            print('We are using "masked phase correlation" method for registration.')
            if self.mask is not None:
                for ii in range(len(self.alignment_pair_list)):
                    fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][0]).zfill(3)],
                                                                 str(self.xanes3D_recon_fixed_sli).zfill(5))
                    self.fixed[:] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                    jj_id = 0
                    for jj in range(self.xanes3D_recon_fixed_sli-self.xanes3D_sli_search_half_range,
                                    self.xanes3D_recon_fixed_sli+self.xanes3D_sli_search_half_range):
                        fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][1]).zfill(3)], str(jj).zfill(5))
                        self.moving[jj_id] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

                        self.shift[ii, jj_id] = masked_register_translation(self.fixed, self.moving[jj_id],
                                                                         self.mask, overlap_ratio=self.overlap_ratio)
                        self.moving[jj_id] = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(self.moving[jj_id]),
                                                                                         self.shift[ii, jj_id])))[:]
                        print(self.shift[ii, jj_id])
                        jj_id += 1
                    g11 = g1.create_group(str(ii).zfill(3))
                    g11.create_dataset('shift'+str(ii).zfill(3),
                                       data=self.shift[ii])
                    g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                       data=self.moving-self.fixed)
            else:
                print('Please provide a mask to use mpc method.')

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
                    for jj in range(self.xanes3D_recon_fixed_sli-self.xanes3D_sli_search_half_range,
                                    self.xanes3D_recon_fixed_sli+self.xanes3D_sli_search_half_range):
                        fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][1]).zfill(3)], str(jj).zfill(5))
                        self.moving[jj_id] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

                        self.shift[ii, jj_id] = sr.register(self.fixed*self.mask, self.moving[jj_id]*self.mask)
                        self.moving[jj_id] = sr.transform(self.moving[jj_id], self.shift[ii, jj_id])[:]
                        print(ii, self.shift[ii, jj_id])
                        jj_id += 1
                    g11 = g1.create_group(str(ii).zfill(3))
                    g11.create_dataset('shift'+str(ii).zfill(3),
                                       data=self.shift[ii, :].astype(np.float32))
                    g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                       data=self.moving-self.fixed)

            else:
                for ii in range(len(self.alignment_pair_list)):
                    fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][0]).zfill(3)],
                                                                 str(self.xanes3D_recon_fixed_sli).zfill(5))
                    self.fixed[:] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                    jj_id = 0
                    for jj in range(self.xanes3D_recon_fixed_sli-self.xanes3D_sli_search_half_range,
                                    self.xanes3D_recon_fixed_sli+self.xanes3D_sli_search_half_range):
                        fn = self.xanes3D_recon_path_template.format(self.img_ids_dict[str(self.alignment_pair_list[ii][1]).zfill(3)], str(jj).zfill(5))
                        self.moving[jj_id] = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

                        self.shift[ii, jj_id] = sr.register(self.fixed, self.moving[jj_id])
                        self.moving[jj_id] = sr.transform(self.moving[jj_id], self.shift[ii, jj_id])[:]
                        print(ii, self.shift[ii, jj_id])
                        jj_id += 1
                    g11 = g1.create_group(str(ii).zfill(3))
                    g11.create_dataset('shift'+str(ii).zfill(3),
                                       data=self.shift[ii, :].astype(np.float32))
                    g11.create_dataset('trial_reg_img'+str(ii).zfill(3),
                                       data=self.moving-self.fixed)
        f.close()

    def reg_xanes3D_chunk(self):
        fn = self.xanes3D_recon_path_template.format(self.fixed_img_id,
                                                     str(self.xanes3D_recon_fixed_sli).zfill(5))
        self.fixed = tifffile.imread(fn)[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        self.moving = np.ndarray([2*self.xanes3D_sli_search_half_range,
                                  self.roi[1]-self.roi[0],
                                  self.roi[3]-self.roi[2]])
        self._reg_xanes3D_chunk()

    def apply_xanes3D_chunk_shift(self, shift_dict, sli_s, sli_e,
                                  trialfn=None, savefn=None, optional_shift_dict=None):
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
            f = h5py.File(savefn, 'a')
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
            self.roi = f['/trial_registration/trial_reg_parameters/slice_roi'][:]
            g1.create_dataset('slice_roi', data=self.roi)
            self.anchor_id = f['/trial_registration/trial_reg_parameters/fixed_scan_id'][()]
            g1.create_dataset('fixed_scan_id', data=self.anchor_id)
            self.chunk_sz = f['/trial_registration/trial_reg_parameters/chunk_sz'][()]
            g1.create_dataset('chunk_sz', data=self.chunk_sz)
            self.method = f['/trial_registration/trial_reg_parameters/reg_method'][()]
            g1.create_dataset('reg_method', data=self.method)
            self.ref_mode = f['/trial_registration/trial_reg_parameters/reg_ref_mode'][()]
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
                                           sli_e-sli_s,
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
                if self.method == 'SR':
                    yshift_int = int(shift[0, 2])
                    xshift_int = int(shift[1, 2])
                    shift[0, 2] -= yshift_int
                    shift[1, 2] -= xshift_int
                elif self.method in ['PC', 'MPC']:
                    yshift_int = int(shift[0])
                    xshift_int = int(shift[1])
                    shift[0] -= yshift_int
                    shift[1] -= xshift_int

                for ii in range(sli_s+slioff, sli_e+slioff):
                    fn = self.xanes3D_recon_path_template.format(scan_id,
                                                                 str(ii).zfill(5))
                    img[:] = tifffile.imread(fn)[self.roi[0]-yshift_int:self.roi[1]-yshift_int,
                                                 self.roi[2]-xshift_int:self.roi[3]-xshift_int]

                    self._translate_single_img(img, shift, self.method)
                    print(shift)
                    g21[cnt1, cnt2] = img[:]
                    cnt2 += 1

                g22[cnt1] = self.eng_dict[key]
                cnt1 += 1
            f.close()

    def save_reg_result(self, dtype='2D_XANES', data=None):
        if dtype.upper() == '2D_XANES':
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path, mode=777)
            print(f'The registration results will be saved in {self.savefn}')

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
                    g1.create_dataset('registered_image', data=self.moving)
                else:
                    g1.create_dataset('registered_image', data=data)
            else:
                del g1['registered_image']
                if data is None:
                    g1.create_dataset('registered_image', data=self.moving)
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
            print(f'The registration results will be saved in {self.savefn}')

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
                    g1.create_dataset('registered_image', data=self.moving)
                else:
                    g1.create_dataset('registered_image', data=data)
            else:
                del g1['registered_image']
                if data is None:
                    g1.create_dataset('registered_image', data=self.moving)
                else:
                    g1.create_dataset('registered_image', data=data)

            if 'residual_image' not in g1:
                if data is None:
                    g1.create_dataset('residual_image',
                                      data=np.float32(self.fixed)-self.moving)
                else:
                    g1.create_dataset('residual_image',
                                      data=np.float32(self.fixed)-data)
            else:
                del g1['residual_image']
                if data is None:
                    g1.create_dataset('residual_image',
                                      data=np.float32(self.fixed)-self.moving)
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

    # def fit_metrics(self, *argv, order=5, plot=True, save=True):
    #     metric_dict = {'si': self.si, 'mse': self.mse, 'nrmse': self.nrmse}
    #     p = []
    #     pf = []
    #     possible_match = []
    #     cnt = 0
    #     for metric in argv:
    #         try:
    #             x = np.arange(metric_dict[metric].shape[0])
    #         except:
    #             print('The metric to be fitted does not exist. Quit!')
    #             exit()
    #         p.append(fit_poly1d(x, metric_dict[metric], order))
    #         pf.append(np.abs(np.diff(p[cnt](x))))
    #         possible_match.append(pf[cnt].argmin())
    #         cnt += 1

    #     self.fit_curve = p
    #     self.fit_curve_diff = pf
    #     self.possible_match = possible_match

    #     if save:
    #         if os.path.exists(self.save_path):
    #             print(f'The fitting results will be saved in {self.savefn}')
    #             f = h5py.File(self.savefn, 'a')
    #             if not 'registration' in f:
    #                 g = f.create_group('registration')
    #             else:
    #                 g = f['registration']
    #             cnt = 0
    #             for metric in argv:
    #                 if not 'fitting to '+metric in g:
    #                     g.create_dataset('fitting to '+metric, data=p[cnt])
    #                 else:
    #                     del g['fitting to '+metric]
    #                     g.create_dataset('fitting to '+metric, data=p[cnt])

    #                 if not 'differential of the fitting to '+metric in g:
    #                     g.create_dataset('differential of the fitting to '+metric, data=pf[cnt])
    #                 else:
    #                     del g['differential of the fitting to '+metric]
    #                     g.create_dataset('differential of the fitting to '+metric, data=pf[cnt])

    #                 if not 'possible match based on the fitting to '+metric+' at' in g:
    #                     g.create_dataset('possible match based on the fitting to '+metric+' at', data=possible_match[cnt])
    #                 else:
    #                     del g['possible match based on the fitting to '+metric+' at']
    #                     g.create_dataset('possible match based on the fitting to '+metric+' at', data=possible_match[cnt])
    #                 cnt += 1
    #             f.close()
    #         else:
    #             os.makedirs(self.save_path, mode=777)
    #             print(f'The fitting results will be saved in {self.savefn}')
    #             f = h5py.File(self.savefn, 'a')
    #             cnt = 0
    #             for metric in argv:
    #                 if not 'fitting to '+metric in g:
    #                     g.create_dataset('fitting to '+metric, data=p[cnt])
    #                 else:
    #                     del g['fitting to '+metric]
    #                     g.create_dataset('fitting to '+metric, data=p[cnt])

    #                 if not 'differential of the fitting to '+metric in g:
    #                     g.create_dataset('differential of the fitting to '+metric, data=pf[cnt])
    #                 else:
    #                     del g['differential of the fitting to '+metric]
    #                     g.create_dataset('differential of the fitting to '+metric, data=pf[cnt])

    #                 if not 'possible match based on the fitting to '+metric+' at' in g:
    #                     g.create_dataset('possible match based on the fitting to '+metric+' at', data=possible_match[cnt])
    #                 else:
    #                     del g['possible match based on the fitting to '+metric+' at']
    #                     g.create_dataset('possible match based on the fitting to '+metric+' at', data=possible_match[cnt])
    #                 cnt += 1
    #             f.close()
    #     if plot:
    #         plt.ioff()
    #         n = len(argv)
    #         fig, ax = plt.subplots(n, 2, sharex=True)
    #         cnt = 0
    #         for metric in argv:
    #             ax[cnt, 0].plot(x, p[cnt](x), metric_dict[metric], '+')
    #             ax[cnt, 0].set_title(metric)
    #             ax[cnt, 1].plot(x[:-1], pf[cnt], np.abs(np.diff(metric_dict[metric])), '+')
    #             ax[cnt, 1].set_title('differentiation of ' + metric)
    #             print(f'extrem of {metric} is at {pf[cnt].argmin()}')
    #             cnt += 1
    #         plt.show()

    def _translate_single_img(self, img, shift, method):
        if method.upper() in ['PC', 'MPC']:
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

    # def translate_moving_data(self):
    #     if self.method.upper() in {'PC', 'MPC'}:
    #         for ii in range(len(self.alignment_pair_list)):
    #             self.moving[self.alignment_pair_list[ii][1]] = np.real(np.fft.ifftn(fourier_shift(
    #                     np.fft.fftn(self.moving[self.alignment_pair_list[ii][1]]), self.shift[ii])))[:]
    #             print(ii, self.shift[ii])
    #     else:
    #         if self.mode.upper() == 'TRANSLATION':
    #             sr = StackReg(StackReg.TRANSLATION)
    #         elif  self.mode.upper() == 'RIGID_BODY':
    #             sr = StackReg(StackReg.RIGID_BODY)
    #         elif  self.mode.upper() == 'SCALED_ROTATION':
    #             sr = StackReg(StackReg.SCALED_ROTATION)
    #         elif  self.mode.upper() == 'AFFINE':
    #             sr = StackReg(StackReg.AFFINE)
    #         elif  self.mode.upper() == 'BILINEAR':
    #             sr = StackReg(StackReg.BILINEAR)
    #         for ii in range(len(self.alignment_pair_list)):
    #             self.moving[self.alignment_pair_list[ii][1]] = sr.transform(self.moving[self.alignment_pair_list[ii][1]],
    #                                                                    tmat=self.shift[ii])[:]
    #             print(ii, self.shift[ii])

    # def translate_2D_stack(self, img_stack):
    #     if self.method.upper() in {'PC', 'MPC'}:
    #         for ii in range(len(self.alignment_pair_list)):
    #             img_stack[self.alignment_pair_list[ii][1]] = np.real(np.fft.ifftn(fourier_shift(
    #                     np.fft.fftn(img_stack[self.alignment_pair_list[ii][1]]), self.shift[ii])))[:]
    #             print(ii, self.shift[ii])
    #     else:
    #         if self.mode.upper() == 'TRANSLATION':
    #             sr = StackReg(StackReg.TRANSLATION)
    #         elif  self.mode.upper() == 'RIGID_BODY':
    #             sr = StackReg(StackReg.RIGID_BODY)
    #         elif  self.mode.upper() == 'SCALED_ROTATION':
    #             sr = StackReg(StackReg.SCALED_ROTATION)
    #         elif  self.mode.upper() == 'AFFINE':
    #             sr = StackReg(StackReg.AFFINE)
    #         elif  self.mode.upper() == 'BILINEAR':
    #             sr = StackReg(StackReg.BILINEAR)
    #         for ii in range(len(self.alignment_pair_list)):
    #             img_stack[self.alignment_pair_list[ii][1]] = sr.transform(img_stack[self.alignment_pair_list[ii][1]],
    #                                                                  tmat=self.shift[ii])[:]
    #             print(ii, self.shift[ii])



    def xanes3D_shell_slicing(self, fn):
        pass





