#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 17:40:00 2019

@author: xiao
"""

import numpy as np
import skimage.restoration as skr
from copy import deepcopy
from scipy.ndimage import gaussian_filter

"""
    General image filters for xanes data analysis will be implemented here
"""
def smooth_flat(flat, sigma):
    flat[:] = gaussian_filter(flat, sigma)[:]
    
    
def normalize_raw(imgs, flat, dark):
    imgs[:] = (imgs - dark)/(flat - dark)[:]
    imgs[imgs<0] = 1
    imgs[np.isnan(imgs)] = 1
    imgs[np.isinf(imgs)] = 1


class denoise_filters():
    def __init__(self, img, flt, **kwargs):
        self.filter_dict = {'wiener': skr.wiener,
                            'denoise_nl_means': skr.denoise_nl_means,
                            'denoise_tv_chambolle': skr.denoise_tv_chambolle,
                            'denoise_bilateral': skr.denoise_bilateral,
                            'denoise_wavelet': skr.denoise_wavelet}
        self.img = img
        self.flt = flt
        if len(kwargs) != 0:
            self.params = kwargs
        else:    
            self.params = None
        print(self.params)    
            
    def set_img(self, img):
        self.img = img

    def set_filter(self, flt, **kwargs):
        self.flt = flt
        self.params = kwargs 
        
    def apply_filter(self, inplace=True):
        if self.flt is None:
            print('There is no filter configured. Please configure a filter first...')
            exit()
        else:
            s = self.img.shape
            if inplace:
                if len(s) == 2:
                    self.img[:] = self.filter_dict[self.flt](self.img, self.params)
                elif len(s) == 3:
                    for ii in range(s[0]):
                        self.img[ii, :] = self.filter_dict[self.flt](self.img[ii, :], **self.params)
                        print(ii)
                return self.img        
            else:
                img_temp = deepcopy(self.img)
                if len(s) == 2:
                    img_temp[:] = self.filter_dict[self.flt](img_temp, self.params)
                elif len(s) == 3:
                    for ii in range(s[0]):
                        img_temp[ii, :] = self.filter_dict[self.flt](img_temp[ii, :], self.params)
                return img_temp        
        