#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 18:53:26 2020

@author: xiao
"""
import os
import numpy as np

import astra, tomopy
import tifffile, h5py
import matplotlib.pylab as mpp

def reader(fn, dtype, path=None, dim=None):
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
            step = np.partition(np.diff(tem), 5)
            was = np.argmax(np.diff(tem))
            wan = int((tem[was+1] - tem[was])//step)
            return (np.concatenate(tem[:was], np.linspace(tem[was], tem[was+1], wan, endpoint=False)[:], tem[was+1:]),
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
                    return f['/obj_supp'][:]>0
                else:
                    return f[path][:]>0
            else:
                if path is None:
                    return f['/obj_supp'][dim]>0
                else:
                    return f[path][dim]>0
            
def bprjr(prj, cfg):
    """
    "back projector": prj -> obj
    Inputs:
        alg: str, algorithm name in ['gridrec', 'art', 'fbp', 'bart', 'sirt', 'tv',
                                     'mlem', 'osem', 'ospml_hybrid', 'ospml_quad', 
                                     'pml_hybrid', 'pml_quad', 'grad',
                                     'astra_3D', 'astra_2D']
        data: 'img' and 'theta' for alg in ['gridrec', 'art', 'fbp', 'bart', 'sirt', 'tv',
                                     'mlem', 'osem', 'ospml_hybrid', 'ospml_quad', 
                                     'pml_hybrid', 'pml_quad', 'grad']
              'img' and 'num_iters' for alg in ['astra_2D']
              'num_iters' for alg in ['astra_3D']
              
    """
    alg = cfg['name']
    if alg in ['gridrec', 'art', 'fbp', 'bart', 'sirt', 'tv', 'mlem', 'osem', 
               'ospml_hybrid', 'ospml_quad', 'pml_hybrid', 'pml_quad', 'grad']:
        # print(cfg['cfg'])
        mod_obj = tomopy.recon(prj, cfg['theta'], **cfg['cfg'])
    elif alg in ['astra.3D']:
        algorithm_id = astra.algorithm.create(cfg['cfg']['alg_cfg'])
        astra.algorithm.run(algorithm_id, cfg['cfg']['num_iters'])
        mod_obj = astra.data3d.get(cfg['cfg']['alg_cfg']['ReconstructionDataId'])
        astra.algorithm.delete(algorithm_id)
        astra.data3d.delete(cfg['cfg']['alg_cfg']['ReconstructionDataId'])
        astra.data3d.delete(cfg['cfg']['alg_cfg']['ProjectionDataId'])
        astra.functions.clear()
    elif alg in ['astra.2D']:
        mod_obj = []
        proj_geom = astra.data2d.get_geometry(cfg['cfg']['alg_cfg']['ProjectionDataId'])
        for ii in range(prj.shape[0]):
            cfg['cfg']['alg_cfg']['ProjectionDataId'] = astra.data2d.create('-sino', proj_geom, prj[ii])
            algorithm_id = astra.algorithm.create(cfg['cfg']['alg_cfg'])
            astra.algorithm.run(algorithm_id, cfg['cfg']['num_iters'])
            mod_obj.append(astra.data2d.get(cfg['cfg']['alg_cfg']['ReconstructionDataId']))
        astra.algorithm.delete(algorithm_id)
        astra.data3d.delete(cfg['cfg']['alg_cfg']['ReconstructionDataId'])
        astra.data3d.delete(cfg['cfg']['alg_cfg']['ProjectionDataId'])
        astra.functions.clear()
    return np.array(mod_obj)

def fprjr(obj, cfg):
    """
    "forward projector": obj -> prj
    inputs:
        obj: ndarray, 3D array of the object
        projector: str, type of projector in ['tomopy', 'astra']
        *args: positional arg theta for projector 'tomopy', and 'proj_geom', and
               'vol_geom' for projector 'astra'
    """
    if cfg['name'] == 'tomopy':
        proj = tomopy.project(obj, **cfg['cfg'])
    elif cfg['name'] == 'astra':
        fp_id, proj = astra.create_sino3d_gpu(obj, cfg['cfg'])
    return proj

def HIOEngine(inp):
    """
    The convergence is guarantted by the support constraint in obj space and 
    modulus constraint in the projection space. The support constraint in obj
    space uses HIO update scheme.
    Inputs:
        inp: dictionary
            inp['data']: 'prj', 'obj'
            inp['cnts']: 'use_pos_cnt', 'obj_supp'
            inp['prjr']: ['fwd']: 'name', 'cfg' (obj -> prj, 'theta' is defined in 'cfg')
                         ['bac']: 'name', 'cfg' (prj -> obj, 'ang_idx' is define in 'cfg', inp[])
            inp['HIO']: 'beta_obj'
    support constrain is set in slice space; magnitude constrain is set in sinogram space. in the terminology,
    sinogram space = proj space, slice space = obj space

    repeat:
        obj = tomopy.recon(mod_prj)
        prj = tomopy.project(obj)
        mod_prj = Prm(prj)
        mod_obj = tomopy.recon(mod_prj)
        Prm(obj)
            obj = mod_obj                     in support
            obj = obj - beta*mod_obj          not in support
    """
    # forward project (obj - prj), and apply modulus constraint in projection space
    mod_proj = fprjr(inp['data']['obj'], inp['prjr']['fwd'])
    # print(f"{inp['data']['obj'].shape = }, \n{inp['data']['prj'].shape = }, \n{mod_proj.shape = }", '\n',
    #       inp['data']['ang_idx'].shape, inp['prjr']['fwd']['cfg']['theta'].shape)
    mod_proj[inp['data']['ang_idx']] = inp['data']['prj'][inp['data']['ang_idx']]

    if inp['cnts']['use_pos_cnt']:
        mod_proj[mod_proj < 0] = 0  
        
    # back project (prj -> obj), and apply support constraint in object space
    mod_obj = bprjr(mod_proj, inp['prjr']['bac'])
    # print(f"{mod_obj.shape = }, \n{inp['cnts']['obj_supp'].shape = }")
    inp['data']['obj'][:] = (mod_obj*inp['cnts']['obj_supp'] + 
                             (inp['data']['obj'] - inp['HIO']['beta_obj']*mod_obj) * 
                             (1 - inp['cnts']['obj_supp']))[:]
    if inp['cnts']['use_pos_cnt']:
        inp['data']['obj'][inp['data']['obj']<0] = 0

    return inp['data']['obj'], mod_proj

def wedge_recon(cfg):
    """
    This is a routine for reconstructing tomography dataset that has a missing
    angle range. The initial inputs of 'proj_modu' and 'theta', however, cover
    the entire 180 deg angle range. The data values in 'proj_modu' in the missing
    angle range are initialized to 0's. These values will be gradually updated 
    in the following iterations.
    
    Inputs:
        cfg: dictionary
            cfg['iter']: dictionary, iteration scheme
                        ['num_updates'], ['update_step'], ['num_HIO_iter']
            cfg['inps']: dictionary, inputs to HIOEngine
                        ['data']: ['prj'], ['obj'], 
                                  ['was'] (wedge angle start), 
                                  ['wae'] (wedge angle end)
                        ['cnts']: ['use_pos_cnt'], ['obj_supp']
                        ['prjr']: ['fwd']: ['name'], ['cfg'] (obj -> prj, 'theta' is defined in 'cfg')
                                  ['bac']: ['name'], ['cfg'] (prj -> obj, inp[])
                        ['HIO']: ['beta_obj']
            cfg['rcfg']: dictionary, debug setting
                        ['rec']: boolean, save intermediate results
                        ['path']: str, where to save intermediate results
    
    Returns:
    """
    if cfg['rcfg']['rec']:
        if not os.path.exists(cfg['rcfg']['path']):
            os.makedirs(cfg['rcfg']['path'], mode=0o777)
            
    cnt = 0
    for ii in range(cfg['iter']['num_updates']):
        p11_s = (cfg['inps']['data']['was'] + cnt*cfg['iter']['update_step'])
        p11_e = (cfg['inps']['data']['was'] + (cnt+1)*cfg['iter']['update_step'])
        # p12_s = (cfg['inps']['data']['was'] + (cnt-1)*cfg['iter']['update_step'])
        # p12_e = (cfg['inps']['data']['was'] + cnt*cfg['iter']['update_step'])
        
        p21_s = (cfg['inps']['data']['wae'] - (cnt+1)*cfg['iter']['update_step'])
        p21_e = (cfg['inps']['data']['wae'] - cnt*cfg['iter']['update_step'])
        # p22_s = (cfg['inps']['data']['wae'] - cnt*cfg['iter']['update_step'])
        # p22_e = (cfg['inps']['data']['wae'] - (cnt-1)*cfg['iter']['update_step'])
        print(f'working on angles with indices between {p11_s} and {p11_e}, and {p21_s} and {p21_e}')
        cfg['inps']['data']['ang_idx'] = np.concatenate((np.arange(p11_s), np.arange(p21_e+1, cfg['inps']['data']['prj'].shape[0])))
                
        for jj in range(cfg['iter']['num_HIO_iter']):
            obj, proj = HIOEngine(cfg['inps'])
            print(f'\t update iteration: {ii}; HIO iteration: {jj}')

        cfg['inps']['data']['prj'][p11_s:p11_e, :] = proj[p11_s:p11_e, :]
        cfg['inps']['data']['prj'][p21_s:p21_e, :] = proj[p21_s:p21_e, :]

        if cfg['rcfg']['rec']:
            fn = os.path.join(cfg['rcfg']['path'], f"recon_obj_iter_{str(ii).zfill(3)}.tif")
            tifffile.imsave(fn, obj.astype(np.float32))
            fn = os.path.join(cfg['rcfg']['path'], f"updated_sino_{str(ii).zfill(3)}.tif")
            tifffile.imsave(fn, proj.astype(np.float32))
            fn = os.path.join(cfg['rcfg']['path'], f"updated_prj_iter_{str(ii).zfill(3)}.tif")
            tifffile.imsave(fn, cfg['inps']['data']['prj'].astype(np.float32))
            
        # if p11_s < cfg['inps']['data']['wae']:
        if p11_e < p21_s:
            cnt += 1
        else:
            break
    return obj
    
if __name__ == '__main__':
    """
    cfg: dictionary
         cfg['iter']: dictionary, iteration scheme
                     ['num_updates'], ['update_step'], ['num_HIO_iter']
         cfg['inps']: dictionary, inputs to HIOEngine
                     ['data']: ['prj'], ['obj'], 
                               ['was'] (wedge angle start), 
                               ['wae'] (wedge angle end)
                     ['cnts']: ['use_pos_cnt'], ['obj_supp']
                     ['prjr']: ['fwd']: ['name'], ['cfg'] (obj -> prj, 'theta' is defined in 'cfg')
                               ['bac']: ['name'], ['cfg'] (prj -> obj, inp[])
                     ['HIO']: ['beta_obj']
         cfg['rcfg']: dictionary, debug setting
                     ['rec']: boolean, save intermediate results
                     ['path']: str, where to save intermediate results
    """ 
    prj_dim = np.s_[:, 1080:1081, 930:1520]
    obj_dim = [1, 415, 415]
    theta_dim = [122, 222]
    
    prj_fn = '/run/media/xiao/Data/data/Tiankai/fly_scan2_id_66885.h5' 
    theta_fn = '/run/media/xiao/Data/data/Tiankai/fly_scan2_id_66885.h5' 
    obj_fn = '' 
    obj_supp_fn = '/run/media/xiao/Data/data/Tiankai/66885_1080_mask_415.h5' 

    theta, was, wae = reader(theta_fn, 'theta', dim=theta_dim)   

    rec_save_fn = '/run/media/xiao/Data/data/Tiankai/test_rec/final_rec.tiff'               
    
    # config
    cfg = {'iter':{}, 
           'inps':{'data':{}, 
                   'cnts':{}, 
                   'prjr':{'fwd':{'cfg':{}}, 
                           'bac':{'cfg':{}}}, 
                   'HIO':{}}, 
           'rcfg':{}}
    
    cfg['iter']['num_updates'] = 10
    cfg['iter']['update_step'] = 10
    cfg['iter']['num_HIO_iter'] = 10
    
    cfg['inps']['data']['prj'] = reader(prj_fn, 'prj', dim=prj_dim)
    cfg['inps']['data']['obj'] = reader(obj_fn, 'obj', dim=obj_dim)
    cfg['inps']['data']['was'] = was
    cfg['inps']['data']['wae'] = wae
    cfg['inps']['data']['ang_idx'] = np.concatenate((np.arange(was), np.arange(wae+1, cfg['inps']['data']['prj'].shape[0])))
    
    cfg['inps']['cnts']['use_pos_cnt'] = True
    cfg['inps']['cnts']['obj_supp'] = reader(obj_supp_fn, 'obj_supp', dim=None)
    
    cfg['inps']['prjr']['fwd']['name'] = 'tomopy'
    cfg['inps']['prjr']['fwd']['cfg']['theta'] = theta
    
    cfg['inps']['prjr']['bac']['name'] = 'mlem'
    cfg['inps']['prjr']['bac']['theta'] = theta
    cfg['inps']['prjr']['bac']['cfg']['algorithm'] = cfg['inps']['prjr']['bac']['name']
    cfg['inps']['prjr']['bac']['cfg']['center'] = 303
    cfg['inps']['prjr']['bac']['cfg']['num_gridx'] = obj_dim[2] 
    cfg['inps']['prjr']['bac']['cfg']['num_gridy'] = obj_dim[1] 
    # cfg['inps']['prjr']['bac']['cfg']['reg_par'] = 0.1 
    
    cfg['inps']['HIO']['beta_obj'] = 0.7
    
    cfg['rcfg']['rec'] = True
    cfg['rcfg']['path'] = '/run/media/xiao/Data/data/Tiankai/test_rec'
    
    # recon
    cfg['inps']['data']['obj'] = wedge_recon(cfg)
    
    tifffile.imsave(os.path.join(cfg['rcfg']['path'], 'final_rec.tiff'), 
                    cfg['inps']['data']['obj'].astype(np.float32))
    
    # imshow
    mpp.imshow(cfg['inps']['data']['obj'][0, :])
    mpp.title('rec_wedge_sirt_no_patch')
    mpp.show()
    
    
    # data = np.zeros([1, 128, 128])
    # data[0, 14:114, 14:114] = tomopy.lena(size=100)[:]
    # # data[:] = tomopy.shepp2d(size=128)[:]
    # obj = np.zeros([1, 128, 128])
    # num_angles = 180
    # obj_supp_dialation = 1
    # proj_supp_dialation = 1
    # wedge_start = 50
    # wedge_end = 130
    # propagate_step = 0
    # test_dir = 'test44_lena_combined'

    # proj, theta = genProjData(data, num_angles=num_angles)
    # print('shepp proj data', proj.shape)
    # obj_supp = genObjSupp(data, dilation=obj_supp_dialation)
    # print('shepp obj supp', obj_supp.shape)
    # proj_supp = genProjSupp(proj, wedge_start, wedge_end, dilation=proj_supp_dialation)
    # proj_modu = np.ndarray(proj.shape)
    # proj_modu[:] = proj[:]
    # proj_modu[wedge_start:wedge_end, :] = 0
    
    # obj_recon = tomopy.recon(proj, theta, algorithm='gridrec')
    # tifffile.imsave('/home/xiao/tmp/temp.tif', obj_recon)
    # # if algorithm_config['algorithm'] == 'gridrec':
    # #     obj_recon = tomopy.recon(proj, theta, algorithm='gridrec')
    # # else:
    # #     obj_recon = tomopy.recon(proj, theta, num_iter=num_inner_iters, **algorithm_config)
    # # ----------- generate phantom data -- end

    # # ----------- iterative recon -- start
    # mpp.figure(100)
    # mpp.imshow(obj_recon[0, :])
    # mpp.title('initial_obj')

    # mpp.figure(200)
    # mpp.imshow(proj[:, 0, :])
    # mpp.title('init_full_sino')

    # mpp.figure(300)
    # mpp.imshow(proj_modu[:, 0, :])
    # mpp.title('init_wedge_sino')
    # # ++++++++++++ generate data -- end





    # # algorithm_config = {'algorithm': 'sirt',
    # #                     'num_gridx': 128,
    # #                     'num_gridy': 128,
    # #                     'init_recon': None}
    # #
    # # algorithm_config = {'algorithm': 'tv',
    # #                     'num_gridx': 128,
    # #                     'num_gridy': 128,
    # #                     'reg_par': 0.3,
    # #                     'init_recon': None, }
    # #
    # # params = {'data_params': {'wedge_start': wedge_start,
    # #                           'wedge_end': wedge_end,
    # #                           'proj_modu': proj_modu,
    # #                           'obj': obj,
    # #                           'num_angles': num_angles},
    # #           'algorithm_params': algorithm_config,
    # #           'recon_params': {'use_support_constraints': True,
    # #                            'use_positivity_constraint': True,
    # #                            'num_outer_iters': 20,
    # #                            'num_inner_iters': 30,
    # #                            'num_iters_in_HIO': 20,
    # #                            'propagate_step': propagate_step},
    # #           'HIO_params': {'obj_supp': obj_supp,
    # #                          'beta_obj': 1.,
    # #                          'beta_proj': 1.,
    # #                          'use_proj_supp': None,
    # #                          'proj_supp': proj_supp},
    # #           'record_params': {'record_freq': 1,
    # #                             'record_dir': '/media/Disk2/data/WedgedDataReconTest/' + test_dir + '/tv1'}}
    # #
    # # obj = new_2recon_universial(**params)

    # mpp.figure(500)
    # mpp.imshow(obj[0, :])
    # mpp.title('rec_wedge_sirt_no_patch')

    # cnt = 2
    # for i in range(2):
    #     # algorithm_config = {'algorithm': 'sirt',
    #     #                     'num_gridx': 128,
    #     #                     'num_gridy': 128,
    #     #                     'init_recon': obj}
    #     algorithm_config = {'algorithm': 'grad',
    #                         'num_gridx': 128,
    #                         'num_gridy': 128,
    #                         'reg_par': 0.1,
    #                         'init_recon': obj, }

    #     params = {'data_params': {'wedge_start': wedge_start,
    #                               'wedge_end': wedge_end,
    #                               'proj_modu': proj_modu,
    #                               'obj': obj,
    #                               'num_angles': num_angles},
    #               'algorithm_params': algorithm_config,
    #               'recon_params': {'use_support_constraints': True,
    #                                'use_positivity_constraint': True,
    #                                'num_outer_iters': 20,
    #                                'num_inner_iters': 30,
    #                                'num_iters_in_HIO': 20,
    #                                'propagate_step': propagate_step},
    #               'HIO_params': {'obj_supp': obj_supp,
    #                              'beta_obj': 1.,
    #                              'beta_proj': 1.,
    #                              'use_proj_supp': None,
    #                              'proj_supp': proj_supp},
    #               'record_params': {'record_freq': 1,
    #                                 'record_dir': '/media/Disk2/data/WedgedDataReconTest/' + test_dir + '/' +
    #                                 algorithm_config['algorithm'] + str(cnt)}}

    #     obj = new_2recon_universial(**params)
    #     cnt += 1

    #     mpp.figure((cnt + 3) * 100)
    #     mpp.imshow(obj[0, :])
    #     mpp.title('rec_wedge_sirt_no_patch')

    #     # algorithm_config = {'algorithm': 'tv',
    #     #                     'num_gridx': 128,
    #     #                     'num_gridy': 128,
    #     #                     'reg_par': 0.3,
    #     #                     'init_recon': obj, }
    #     # algorithm_config = {'algorithm': 'grad',
    #     #                     'num_gridx': 128,
    #     #                     'num_gridy': 128,
    #     #                     'reg_par': 0.1,
    #     #                     'init_recon': obj, }
    #     algorithm_config = {'algorithm': 'mlem',
    #                         'num_gridx': 128,
    #                         'num_gridy': 128,
    #                         'init_recon': obj, }
    #     # algorithm_config = {'algorithm': 'bart',
    #     #                     'num_gridx': 128,
    #     #                     'num_gridy': 128,
    #     #                     'num_block': 5,
    #     #                     'init_recon': obj, }

    #     params = {'data_params': {'wedge_start': wedge_start,
    #                               'wedge_end': wedge_end,
    #                               'proj_modu': proj_modu,
    #                               'obj': obj,
    #                               'num_angles': num_angles},
    #               'algorithm_params': algorithm_config,
    #               'recon_params': {'use_support_constraints': True,
    #                                'use_positivity_constraint': True,
    #                                'num_outer_iters': 20,
    #                                'num_inner_iters': 30,
    #                                'num_iters_in_HIO': 20,
    #                                'propagate_step': propagate_step},
    #               'HIO_params': {'obj_supp': obj_supp,
    #                              'beta_obj': 1.,
    #                              'beta_proj': 1.,
    #                              'use_proj_supp': None,
    #                              'proj_supp': proj_supp},
    #               'record_params': {'record_freq': 1,
    #                                 'record_dir': '/media/Disk2/data/WedgedDataReconTest/' + test_dir + '/' +
    #                                 algorithm_config['algorithm'] + str(cnt)}}

    #     obj = new_2recon_universial(**params)
    #     cnt += 1

    #     mpp.figure((cnt + 3) * 100)
    #     mpp.imshow(obj[0, :])
    #     mpp.title('rec_wedge_sirt_no_patch')

    # mpp.show()