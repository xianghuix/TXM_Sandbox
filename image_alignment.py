#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 22:10:43 2020

@author: xiao
"""
import os, gc, time
import numpy as np
from scipy.ndimage import fourier_shift
from collections import OrderedDict

from pystackreg import StackReg
from skimage.registration import phase_cross_correlation
import multiprocessing as mp
import astra, tomopy

ALG_PARAM_DICT = OrderedDict({"gridrec":{0: ["filter_name", ["parzen", 'shepp', 'cosine', 'hann', 'hamming', 'ramlak', 'butterworth', 'none'], "filter_name: filter that is used in frequency space"]},
                              "sirt":{3: ["num_gridx", 1280, "num_gridx: number of the reconstructed slice image along x direction"],
                                      4: ["num_gridy", 1280, "num_gridy: number of the reconstructed slice image along y direction"],
                                      5: ["num_inter", 10, "num_inter: number of reconstruction iterations"]},
                              "tv":{3: ["num_gridx", 1280, "num_gridx: number of the reconstructed slice image along x direction"],
                                    4: ["num_gridy", 1280, "num_gridy: number of the reconstructed slice image along y direction"],
                                    5: ["num_inter", 10, "num_inter: number of reconstruction iterations"],
                                    6: ["reg_par", 0.1, "reg_par: relaxation factor in tv regulation"]},
                              "mlem":{3: ["num_gridx", 1280, "num_gridx: number of the reconstructed slice image along x direction"],
                                      4: ["num_gridy", 1280, "num_gridy: number of the reconstructed slice image along y direction"],
                                      5: ["num_inter", 10, "num_inter: number of reconstruction iterations"]},
                              "astra":{0: ["method", ["EM_CUDA"], "method: astra reconstruction methods"],
                                       1: ["proj_type", ["cuda"], "proj_type: projection calculation options used in astra"],
                                       2: ["extra_options", ["MinConstraint"], "extra_options: extra constraints used in the reconstructions. you need to set p03 for a MinConstraint level"],
                                       3: ["extra_options_param", -0.1, "extra_options_param: parameter used together with extra_options"],
                                       4: ["num_inter", 50, "num_inter: number of reconstruction iterations"]}})

class align_images():
    def __init__(self, data_order='astra'):
        """
        corr_cfg: dictionary; correlator configuration; element 'correlator' has
                  options 'pc', 'sr', 'sift', 'combo'
        proj_cfg: dictionary; projector configuration; element 'projector' has
                  options 'tomopy' and 'astra'
        rec_alg: dictionary; reconstruction algorithm chosen from 'tomopy_gridrec',
                 'tomopy_tv', 'tomopy_mlem', 'astra_sirt', and 'astra_cgls'; the
                 configuration 'cfg' for each algorithm has options
                 {"gridrec":{"filter_name": ["parzen", 'shepp', 'cosine', 'hann', 'hamming', 'ramlak', 'butterworth', 'none']},
                  "sirt":{"num_gridx": 1280, "num_gridx: number of the reconstructed slice image along x direction",
                          "num_gridy": 1280, "num_gridy: number of the reconstructed slice image along y direction",
                          "num_iter: 10, "num_iter: number of reconstruction iterations"},
                  "tv":{"num_gridx": 1280, "num_gridx: number of the reconstructed slice image along x direction",
                        "num_gridy": 1280, "num_gridy: number of the reconstructed slice image along y direction",
                        "num_iter": 10, "num_iter: number of reconstruction iterations",
                        "reg_par": 0.1, "reg_par: relaxation factor in tv regulation"},
                  "mlem":{"num_gridx": 1280, "num_gridx: number of the reconstructed slice image along x direction",
                          "num_gridy": 1280, "num_gridy: number of the reconstructed slice image along y direction",
                          "num_iter": 10, "num_iter: number of reconstruction iterations"},
                  "astra":{"method": ["EM_CUDA"], "method: astra reconstruction methods",
                           "proj_type": ["cuda"], "proj_type: projection calculation options used in astra",
                           "extra_options": ["MinConstraint"], "extra_options: extra constraints used in the reconstructions. you need to set p03 for a MinConstraint level",
                           "extra_options_param": -0.1, "extra_options_param: parameter used together with extra_options",
                           "num_iter": 50, "num_iter: number of reconstruction iterations"}}

        Returns
        -------
        None.

        """
        self.data_order = data_order
        self.ref_data = None
        self.mov_data = None
        self.angs = None
        self.data_dim = None
        self.mask_thres = None
        self.shift = []
        self.rec_alg = {'alg_name':'tomopy_gridrec', 'cfg':{'filter_name':'parzen'}}
        self.corr_cfg = {0:{'cor_name':'pc', 'sub_itr':1, 'cfg':{'mask_thres':None}},
                         1:{'cor_name':'sr', 'sub_itr':1, 'cfg':{'mode':'RIGID_BODY'}}}
        self.proj_cfg = {'prj_name':'astra', 'cfg':{'proj_geom':None, 'vol_geom':None}}
    
    def set_data(self, mov, angs):
        self.mov_data = mov
        self.angs = angs
    
    def set_rec_alg(self, rec_name, **kwargs):
        self.rec_alg['alg_name'] = rec_name
        self.rec_alg['cfg'] = kwargs
        
    def get_data_info(self):
        self.data_dim = self.mov_data.shape
    
    def _register(self, ref, mov, correlator, cfg):
        """modify self.shift between mov and ref images"""
        if correlator == 'pc':
            if cfg == None:
                cfg = {}
                cfg['reference_mask'] = None
                cfg['moving_mask'] = None
            else:
                thres = cfg
                cfg = {}
                cfg['reference_mask'] = (ref>thres)
                cfg['moving_mask'] = (mov>thres)
            shift, _ = phase_cross_correlation(ref, mov, upsample_factor=100, 
                                                  **cfg)                
            ref = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(mov), shift)))
            return shift, ref
        elif correlator == 'sr':
            if cfg.upper() == 'TRANSLATION':
                sr = StackReg(StackReg.TRANSLATION)
            elif cfg.upper() == 'RIGID_BODY':
                sr = StackReg(StackReg.RIGID_BODY)
            elif cfg.upper() == 'SCALED_ROTATION':
                sr = StackReg(StackReg.SCALED_ROTATION)
            elif cfg.upper() == 'AFFINE':
                sr = StackReg(StackReg.AFFINE)
            elif cfg.upper() == 'BILINEAR':
                sr = StackReg(StackReg.BILINEAR)
            shift = sr.register(ref, mov)
            ref = sr.transform(mov, tmat=shift)
            return shift, ref
        elif correlator == 'sift':
            pass
        elif correlator == 'combo':
            pass
        
            
    def register(self, correlator, cfg):
        """modify self.shift between mov and ref images"""
        print(f"    registration starts at {time.asctime()}")
        n_cpu = os.cpu_count()         
        if self.data_order == 'astra':
            print(0)
            with mp.Pool(n_cpu-1) as pool:
                rlt = pool.starmap(self._register, [(self.ref_data[:, ii, :], self.mov_data[:, ii, :], correlator, cfg) for ii in np.int32(np.arange(self.data_dim[1]))])
            print(1)
            pool.join()
            pool.close()
            print(2)
            for jj in range(self.data_dim[1]):
                # self.shift[jj] = rlt[jj][0]
                self.ref_data[:, jj, :] = rlt[jj][1][:, :]
            del(rlt)
            gc.collect()
        elif self.data_order == 'tomopy':
            with mp.Pool(n_cpu-1) as pool:
                rlt = pool.starmap(self._register, [(self.ref_data[ii, :, :], self.mov_data[ii, :, :], correlator, cfg) for ii in np.int32(np.arange(self.data_dim[0]))])
            pool.join()
            pool.close()
            for jj in range(self.data_dim[0]):
                self.shift[jj] = rlt[jj][0]
                self.ref_data[jj, :, :] = rlt[jj][1][:, :]
            del(rlt)
            gc.collect()
        print(f"    registration finishes at {time.asctime()}")
    
    def projector(self, vol):
        """project recon vol into projection images; modify self.ref_data"""
        projector = self.proj_cfg['prj_name']
        if projector == 'astra':
            fp_id, mov = astra.create_sino3d_gpu(vol, self.proj_geom, self.vol_geom)
            astra.data3d.delete(fp_id)
            astra.functions.clear()
            if self.data_order == 'tomopy':
                self.ref_data = np.swapaxes(mov, 0, 1)
            elif self.data_order == 'astra':
                self.ref_data = mov
        elif projector == 'tomopy':
            pass
    
    def recon(self):
        """recon self.ref_data"""
        if self.rec_alg['alg_name'] == 'tomopy_gridrec':
            if self.data_order == 'tomopy':
                vol = tomopy.recon(self.mov_data, self.angs, center=self.cen, algorithm='gridrec', filter_name='parzen')
            elif self.data_order == 'astra':
                vol = tomopy.recon(np.swapaxes(self.mov_data, 0, 1), self.angs, center=self.cen, algorithm='gridrec', filter_name='parzen')
        elif self.rec_alg['alg_name'] == 'tomopy_mlem':
            if self.data_order == 'tomopy':
                vol = tomopy.recon(self.mov_data, self.angs, center=self.cen, algorithm='mlem', **self.rec_alg['cfg'])
            elif self.data_order == 'astra':
                vol = tomopy.recon(np.swapaxes(self.mov_data), self.angs, center=self.cen, algorithm='mlem', **self.rec_alg['cfg'])
        elif self.rec_alg['alg_name'] == 'astra_cgls':
            recon_id = astra.data3d.create('-vol', self.vol_geom, data=0)
            if self.data_order == 'astra':
                proj_id = astra.data3d.create('-sino', self.proj_geom, self.ref_data)
            if self.data_order == 'tomopy':
                proj_id = astra.data3d.create('-sino', self.proj_geom, np.swapaxes(self.ref_data, 0 ,1))
            
            alg_cfg = astra.astra_dict('CGLS3D_CUDA')
            alg_cfg['ProjectionDataId'] = proj_id
            alg_cfg['ReconstructionDataId'] = recon_id
            algorithm_id = astra.algorithm.create(alg_cfg)
            astra.algorithm.run(algorithm_id, 50)     
            vol = astra.data3d.get(recon_id)
            
            astra.algorithm.delete(algorithm_id)
            astra.data3d.delete(recon_id)
            astra.data3d.delete(proj_id)
            astra.functions.clear()
            print(f"    astra recons finishes at {time.asctime()}")   
        elif self.rec_alg['alg_name'] == 'astra_sirt':
            recon_id = astra.data3d.create('-vol', self.vol_geom, data=0)
            if self.data_order == 'astra':
                proj_id = astra.data3d.create('-sino', self.proj_geom, self.ref_data)
            if self.data_order == 'tomopy':
                proj_id = astra.data3d.create('-sino', self.proj_geom, np.swapaxes(self.ref_data, 0 ,1))
            
            alg_cfg = astra.astra_dict('SIRT3D_CUDA')
            alg_cfg['ProjectionDataId'] = proj_id
            alg_cfg['ReconstructionDataId'] = recon_id
            algorithm_id = astra.algorithm.create(alg_cfg)
            astra.algorithm.run(algorithm_id, 50)     
            vol = astra.data3d.get(recon_id)
            
            astra.algorithm.delete(algorithm_id)
            astra.data3d.delete(recon_id)
            astra.data3d.delete(proj_id)
            astra.functions.clear()
            print(f"    astra recons finishes at {time.asctime()}")  
        return vol
    
        if self.rec_alg == 'tomopy_gridrec':
            pass
        elif self.rec_alg == 'tomopy_mlem':
            pass
        elif self.rec_alg == 'astra_cgls':
            pass
        elif self.rec_alg == 'astra_sirt':
            pass 
    def align_stack(self, itr):
        """
        This is the routine to align tomography dataset by repeating reconstruction,
        re-projection, and correlation between the original images and re-projected
        images

        Parameters
        ----------
        itr : int
            number of iterations of alignment based on re-projection.

        Returns
        -------
        None.

        """
        self.get_data_info()
        if 'astra' in self.rec_alg['alg_name']:
            if self.data_order == 'astra':
                self.proj_geom = astra.creators.create_proj_geom('parallel3d', 1., 1., self.data_dim[0], self.data_dim[2], self.angs)
                self.vol_geom = astra.creators.create_vol_geom(self.data_dim[2], self.data_dim[2], self.data_dim[0])
            elif self.data_order == 'tomopy':
                self.proj_geom = astra.creators.create_proj_geom('parallel3d', 1., 1., self.data_dim[1], self.data_dim[2], self.angs)
                self.vol_geom = astra.creators.create_vol_geom(self.data_dim[2], self.data_dim[2], self.data_dim[1])
        for ii in range(itr):
            for key in sorted(self.corr_cfg.keys()):
                correlator = self.corr_cfg[key]['cor_name']
                sub_itr = self.corr_cfg[key]['sub_itr']
                if correlator == 'pc':
                    cfg = self.corr_cfg[key]['cfg']['mask_thres']
                elif correlator == 'sr':
                    cfg = self.corr_cfg[key]['cfg']['mode']
                elif correlator == 'sift':
                    pass
                elif correlator == 'combo':
                    pass 
                for jj in range(sub_itr):
                    print(ii, key, jj)
                    vol = self.recon()
                    self.projector(vol)
                    self.register(correlator, cfg)
    
    def align_stacks(self):
        """
        This is the routine for aligning two different sets of images

        Returns
        -------
        None.

        """
        pass