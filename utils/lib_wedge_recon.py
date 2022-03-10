#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 09:04:35 2020

@author: xiao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:18:13 2020

@author: xiao
"""
import time

from scipy.ndimage import (gaussian_filter, median_filter, 
                           binary_erosion, binary_dilation)
from skimage.morphology import disk
import numpy as np

import astra, tomopy
import tifffile, h5py

def readr(fn, dtype, path=None, dim=None):
    if dtype == 'prj':
        with h5py.File(fn, 'r') as f:
            if dim is None:
                tem = -np.log((f['/img_tomo'][:] - f['/img_dark_avg'][:])/(f['/img_bkg_avg'][:] - f['/img_dark_avg'][:]))
                tem[np.isinf(tem)] = 1
                tem[np.isnan(tem)] = 1
            else:
                tem = -np.log((f['/img_tomo'][dim] - f['/img_dark_avg'][dim])/(f['/img_bkg_avg'][dim] - f['/img_dark_avg'][dim]))
                tem[np.isinf(tem)] = 1
                tem[np.isnan(tem)] = 1
        return tem
    if dtype == 'obj':
        if dim is None:
            return 0
        else:
            return np.zeros(dim)
    if dtype == 'theta':
        if dim is None:
            with h5py.File(fn, 'r') as f:
                if path is None:            
                    tem = f['/angle'][:]*np.pi/180
                else:
                    tem = f[path][:]*np.pi/180
            step = np.partition(np.diff(tem), 5)[4]
            print(f"{step = }")
            was = np.argmax(np.diff(tem))
            print(f"{was = }")
            wan = int((tem[was+1] - tem[was])//step)
            print(f"{wan = }")
            return (np.concatenate((tem[:was], np.linspace(tem[was], tem[was+1], wan, endpoint=False), tem[was+1:])),
                    was, was+wan-1)
        else:
            with h5py.File(fn, 'r') as f:
                if path is None:
                    tem = f['/angle'][:]*np.pi/180
                else:
                    tem = f[path][:]*np.pi/180
            return tem, dim[0], dim[1]
    if dtype == 'obj_supp':
        with h5py.File(fn, 'r') as f:
            if dim is None:
                if path is None:
                    return (f['/obj_supp'][:]>0).astype(np.int8)
                else:
                    return (f[path][:]>0).astype(np.int8)
            else:
                if path is None:
                    return (f['/obj_supp'][dim]>0).astype(np.int8)
                else:
                    return (f[path][dim]>0).astype(np.int8)
    if dtype == 'sin_supp':
        with h5py.File(fn, 'r') as f:
            if dim is None:
                if path is None:
                    return (f['/sin_supp'][:]>0).astype(np.int8)
                else:
                    return (f[path][:]>0).astype(np.int8)
            else:
                if path is None:
                    return (f['/sin_supp'][dim]>0).astype(np.int8)
                else:
                    return (f[path][dim]>0).astype(np.int8)
                
def pjtr(pjtr, cfg):
    pass

def rntr(alg_cfg):
    pass

def extr(oid, out=None):
    pass

def itr(cfg):
    ts = np.zeros([astra.data2d.get_geometry(cfg['alg']['f'])['Vectors'].shape[0], 
                   astra.data2d.get_geometry(cfg['alg']['f'])['DetectorCount']], 
                   dtype=np.float32)
    alg_s_id = astra.algorithm.create(cfg['alg']['s'])
    alg_f_id = astra.algorithm.create(cfg['alg']['f'])
    
    for ii in range(cfg['itr']['tot_num']):
        ss = astra.data2d.get_shared(cfg['dat']['sin_s_id'])    
        median_filter(ss, [1, 3], output=ss)
        astra.algorithm.run(cfg['alg']['s'], cfg['itr']['sub_s_num'])

        v = astra.data2d.get_shared(cfg['dat']['rec_id'])
        sf = astra.data2d.get_shared(cfg['dat']['sin_f_id']) 
        ts[:] = cfg['dat']['pjtr'].FP(v)[:]
        sf[cfg['dat']['mar'][0]:cfg['dat']['mar'][1]+1] = ts[cfg['dat']['mar'][0]:cfg['dat']['mar'][1]+1]
        astra.algorithm.run(cfg['alg']['f'], cfg['itr']['sub_f_num'])           
        median_filter(sf, [1, 3], output=sf)
        astra.algorithm.run(cfg['alg']['f'], cfg['itr']['sub_f_num'])  

    cfg['dat']['rec'][:] = astra.data2d.get(cfg['dat']['rec_id'])[:]
    rs = astra.data2d.get_shared(cfg['dat']['rec_supp_id']) 
    rs[:] = (gaussian_filter(cfg['dat']['rec'], 
                             cfg['alg']['rec_supp_rel'])>cfg['alg']['rec_thres']).astype(np.int8)[:]
    if cfg['alg']['update_sin']:
        sinogram_id, sino = astra.create_sino(cfg['dat']['rec'], cfg['dat']['prj_id'])
        ssf = astra.data2d.get_shared(cfg['dat']['sin_f_supp_id']) 
        ssf[:] = (sino>cfg['alg']['sin_thres']).astype(np.int8)[:]
        ssf[cfg['dat']['ang_idx']] = (cfg['dat']['prj'][cfg['dat']['ang_idx'], :]>0).astype(np.int8)[:]
        ssf[:] = binary_dilation(ssf, disk(cfg['alg']['sin_supp_rel']))[:]
        sss = astra.data2d.get_shared(cfg['dat']['sin_s_supp_id']) 
        sss[:] = ssf[[cfg['dat']['ang_idx']]]
        astra.data2d.delete(sinogram_id)
    astra.data2d.delete(alg_s_id)
    astra.data2d.delete(alg_f_id)


