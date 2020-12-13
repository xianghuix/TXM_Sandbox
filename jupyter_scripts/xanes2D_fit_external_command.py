import os, h5py
import numpy as np
import TXM_Sandbox.TXM_Sandbox.utils.xanes_math as xm
import TXM_Sandbox.TXM_Sandbox.utils.xanes_analysis as xa
from copy import deepcopy

with h5py.File('/run/media/xiao/Data/data/2D_xanes/2D_trial_reg_multipos_2D_xanes_scan2_id_61201_repeat_00_pos_01_2020-10-05-18-45-25.h5', 'r+') as f:
    imgs = f['/registration_results/reg_results/registered_xanes2D'][:]
    xanes2D_analysis_eng_list = f['/processed_XANES2D/proc_parameters/eng_list'][:]
    xanes2D_analysis_edge_eng = f['/processed_XANES2D/proc_parameters/edge_eng'][()]
    xanes2D_analysis_pre_edge_e = f['/processed_XANES2D/proc_parameters/pre_edge_e'][()]
    xanes2D_analysis_post_edge_s = f['/processed_XANES2D/proc_parameters/post_edge_s'][()]
    xanes2D_analysis_edge_jump_thres = f['/processed_XANES2D/proc_parameters/edge_jump_threshold'][()]
    xanes2D_analysis_edge_offset_thres = f['/processed_XANES2D/proc_parameters/edge_offset_threshold'][()]
    xanes2D_analysis_use_mask = f['/processed_XANES2D/proc_parameters/use_mask'][()]
    xanes2D_analysis_type = f['/processed_XANES2D/proc_parameters/analysis_type'][()]
    xanes2D_analysis_data_shape = f['/processed_XANES2D/proc_parameters/data_shape'][:]
    xanes2D_analysis_data_shape = imgs.shape
    xanes2D_analysis_edge_0p5_fit_s = f['/processed_XANES2D/proc_parameters/edge_0p5_fit_s'][()]
    xanes2D_analysis_edge_0p5_fit_e = f['/processed_XANES2D/proc_parameters/edge_0p5_fit_e'][()]
    xanes2D_analysis_wl_fit_eng_s = f['/processed_XANES2D/proc_parameters/wl_fit_eng_s'][()]
    xanes2D_analysis_wl_fit_eng_e = f['/processed_XANES2D/proc_parameters/wl_fit_eng_e'][()]
    xanes2D_analysis_use_flt_spec = f['/processed_XANES2D/proc_parameters/flt_spec'][()]
    xanes2D_analysis_edge_fit_order = f['/processed_XANES2D/proc_parameters/pre_post_edge_norm_fit_order'][()]
    xana = xa.xanes_analysis(imgs, xanes2D_analysis_eng_list, xanes2D_analysis_edge_eng, pre_ee=xanes2D_analysis_pre_edge_e, post_es=xanes2D_analysis_post_edge_s, edge_jump_threshold=xanes2D_analysis_edge_jump_thres, pre_edge_threshold=xanes2D_analysis_edge_offset_thres)
    if '/processed_XANES2D/proc_spectrum' in f:
        del f['/processed_XANES2D/proc_spectrum']
        g12 = f.create_group('/processed_XANES2D/proc_spectrum')
    else:
        g12 = f.create_group('/processed_XANES2D/proc_spectrum')

    if xanes2D_analysis_type == 'wl':
        _g12 = {}
        for jj in {'centroid_of_eng', 'post_edge_mean', 'normalized_spectrum', 'whiteline_peak_height_direct', 'weighted_eng', 'edge_pos_fit', 'edge0.5_pos_direct', 'pre_edge_sd', 'weighted_attenuation', 'whiteline_pos_fit', 'whiteline_pos_direct', 'edge_offset_filter', 'edge_jump_filter', 'edge0.5_pos_fit', 'centroid_of_eng_relative_to_wl', 'edge_pos_direct', 'pre_edge_mean', 'post_edge_sd'}:
            if jj in ['pre_edge_fit_coef', 'post_edge_fit_coef', 'whiteline_fit_coef']:
                _g12[jj] = g12.create_dataset(jj, shape=(list(len([1.0, 7729.0, 1.0])) + list(xanes2D_analysis_data_shape[1:])), dtype=np.float32)
            elif jj == 'edge_fit_coef':
                _g12[jj] = g12.create_dataset(jj, shape=(list(len([])) + list(xanes2D_analysis_data_shape[1:])), dtype=np.float32)
            else:
                _g12[jj] = g12.create_dataset(jj, shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)
        xana.fit_whiteline(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, 'scipy', **{'model': 'lorentzian', 'fvars': [1.0, 7729.0, 1.0], 'bnds': None, 'jac': '3-point', 'method': 'trf', 'ftol': 1e-07, 'xtol': 1e-07, 'gtol': 1e-07, 'ufac': 50})
        if True:
            xana.fit_edge(xanes2D_analysis_edge_0p5_fit_s, xanes2D_analysis_edge_0p5_fit_e, 'numpy', **{'order': 3, 'ufac': 50, 'flt_spec': False})
        if True:
            xana.find_edge()
        xana.calc_whiteline_direct(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e)
        for jj in {'centroid_of_eng', 'post_edge_mean', 'normalized_spectrum', 'whiteline_peak_height_direct', 'weighted_eng', 'edge_pos_fit', 'edge0.5_pos_direct', 'pre_edge_sd', 'weighted_attenuation', 'whiteline_pos_fit', 'whiteline_pos_direct', 'edge_offset_filter', 'edge_jump_filter', 'edge0.5_pos_fit', 'centroid_of_eng_relative_to_wl', 'edge_pos_direct', 'pre_edge_mean', 'post_edge_sd'}:
            if jj == 'whiteline_pos_fit':
                _g12[jj][:] = np.float32(xana.wl_pos_fit)[:]
            if jj == 'whiteline_pos_direct':
                _g12[jj][:] = np.float32(xana.wl_pos_direct)[:]
            if jj == 'edge_pos_fit':
                _g12[jj][:] = np.float32(xana.edge_pos_fit)[:]
            if jj == 'edge_pos_direct':
                _g12[jj][:] = np.float32(xana.edge_pos_direct)[:]
            if jj == 'edge_fit_coef':
                _g12[jj][:] = np.float32(xana.edge_fit_coef)[:]
            if jj == 'whiteline_fit_coef':
                _g12[jj][:] = np.float32(xana.wl_fit_coef)[:]

    elif xanes2D_analysis_type == 'full':
        _g12 = {}
        for jj in {'centroid_of_eng', 'post_edge_mean', 'normalized_spectrum', 'whiteline_peak_height_direct', 'weighted_eng', 'edge_pos_fit', 'edge0.5_pos_direct', 'pre_edge_sd', 'weighted_attenuation', 'whiteline_pos_fit', 'whiteline_pos_direct', 'edge_offset_filter', 'edge_jump_filter', 'edge0.5_pos_fit', 'centroid_of_eng_relative_to_wl', 'edge_pos_direct', 'pre_edge_mean', 'post_edge_sd'}:
            if jj in ['pre_edge_fit_coef', 'post_edge_fit_coef', 'whiteline_fit_coef']:
                _g12[jj] = g12.create_dataset(jj, shape=(list(len([1.0, 7729.0, 1.0])) + list(xanes2D_analysis_data_shape[1:])), dtype=np.float32)
            elif jj == 'edge_fit_coef':
                _g12[jj] = g12.create_dataset(jj, shape=(list(len([])) + list(xanes2D_analysis_data_shape[1:])), dtype=np.float32)
            elif jj == 'normalized_spectrum':
                _g12[jj] = g12.create_dataset(jj, shape=(xanes2D_analysis_data_shape), dtype=np.float32)
            else:
                _g12[jj] = g12.create_dataset(jj, shape=(xanes2D_analysis_data_shape[1:]), dtype=np.float32)
        xana.fit_pre_edge()
        xana.fit_post_edge()
        xana.cal_edge_jump_map()
        xana.cal_pre_edge_sd()
        xana.cal_post_edge_sd()
        xana.cal_pre_edge_mean()
        xana.cal_post_edge_mean()
        xana.create_edge_jump_filter(xanes2D_analysis_edge_jump_thres)
        xana.create_fitted_edge_filter(xanes2D_analysis_edge_offset_thres)
        xana.normalize_xanes(xanes2D_analysis_edge_eng, order=xanes2D_analysis_edge_fit_order, save_pre_post=True)
        if ('edge0.5_pos_fit' in {'centroid_of_eng', 'post_edge_mean', 'normalized_spectrum', 'whiteline_peak_height_direct', 'weighted_eng', 'edge_pos_fit', 'edge0.5_pos_direct', 'pre_edge_sd', 'weighted_attenuation', 'whiteline_pos_fit', 'whiteline_pos_direct', 'edge_offset_filter', 'edge_jump_filter', 'edge0.5_pos_fit', 'centroid_of_eng_relative_to_wl', 'edge_pos_direct', 'pre_edge_mean', 'post_edge_sd'}) and ('edge_pos_fit' in {'centroid_of_eng', 'post_edge_mean', 'normalized_spectrum', 'whiteline_peak_height_direct', 'weighted_eng', 'edge_pos_fit', 'edge0.5_pos_direct', 'pre_edge_sd', 'weighted_attenuation', 'whiteline_pos_fit', 'whiteline_pos_direct', 'edge_offset_filter', 'edge_jump_filter', 'edge0.5_pos_fit', 'centroid_of_eng_relative_to_wl', 'edge_pos_direct', 'pre_edge_mean', 'post_edge_sd'}):
            tem = deepcopy({'order': 3, 'ufac': 50, 'flt_spec': False})
            tem['cal_deriv'] = True
            xana.fit_edge_0p5(xanes2D_analysis_edge_0p5_fit_s, xanes2D_analysis_edge_0p5_fit_e, 'numpy', **tem)
        elif ('edge0.5_pos_fit' in {'centroid_of_eng', 'post_edge_mean', 'normalized_spectrum', 'whiteline_peak_height_direct', 'weighted_eng', 'edge_pos_fit', 'edge0.5_pos_direct', 'pre_edge_sd', 'weighted_attenuation', 'whiteline_pos_fit', 'whiteline_pos_direct', 'edge_offset_filter', 'edge_jump_filter', 'edge0.5_pos_fit', 'centroid_of_eng_relative_to_wl', 'edge_pos_direct', 'pre_edge_mean', 'post_edge_sd'}) and not ('edge_pos_fit' in {'centroid_of_eng', 'post_edge_mean', 'normalized_spectrum', 'whiteline_peak_height_direct', 'weighted_eng', 'edge_pos_fit', 'edge0.5_pos_direct', 'pre_edge_sd', 'weighted_attenuation', 'whiteline_pos_fit', 'whiteline_pos_direct', 'edge_offset_filter', 'edge_jump_filter', 'edge0.5_pos_fit', 'centroid_of_eng_relative_to_wl', 'edge_pos_direct', 'pre_edge_mean', 'post_edge_sd'}):
            xana.fit_edge(xanes2D_analysis_edge_0p5_fit_s, xanes2D_analysis_edge_0p5_fit_e, 'numpy', **{'order': 3, 'ufac': 50, 'flt_spec': False})
        elif not ('edge0.5_pos_fit' in {'centroid_of_eng', 'post_edge_mean', 'normalized_spectrum', 'whiteline_peak_height_direct', 'weighted_eng', 'edge_pos_fit', 'edge0.5_pos_direct', 'pre_edge_sd', 'weighted_attenuation', 'whiteline_pos_fit', 'whiteline_pos_direct', 'edge_offset_filter', 'edge_jump_filter', 'edge0.5_pos_fit', 'centroid_of_eng_relative_to_wl', 'edge_pos_direct', 'pre_edge_mean', 'post_edge_sd'}) and ('edge_pos_fit' in {'centroid_of_eng', 'post_edge_mean', 'normalized_spectrum', 'whiteline_peak_height_direct', 'weighted_eng', 'edge_pos_fit', 'edge0.5_pos_direct', 'pre_edge_sd', 'weighted_attenuation', 'whiteline_pos_fit', 'whiteline_pos_direct', 'edge_offset_filter', 'edge_jump_filter', 'edge0.5_pos_fit', 'centroid_of_eng_relative_to_wl', 'edge_pos_direct', 'pre_edge_mean', 'post_edge_sd'}):
            tem = deepcopy({'order': 3, 'ufac': 50, 'flt_spec': False})
            tem['cal_deriv'] = False
            xana.fit_edge_0p5(xanes2D_analysis_edge_0p5_fit_s, xanes2D_analysis_edge_0p5_fit_e, 'numpy', **tem)
        if 'edge_pos_direct' in {'centroid_of_eng', 'post_edge_mean', 'normalized_spectrum', 'whiteline_peak_height_direct', 'weighted_eng', 'edge_pos_fit', 'edge0.5_pos_direct', 'pre_edge_sd', 'weighted_attenuation', 'whiteline_pos_fit', 'whiteline_pos_direct', 'edge_offset_filter', 'edge_jump_filter', 'edge0.5_pos_fit', 'centroid_of_eng_relative_to_wl', 'edge_pos_direct', 'pre_edge_mean', 'post_edge_sd'}:
            xana.find_edge()
        if 'edge0.5_pos_direct' in {'centroid_of_eng', 'post_edge_mean', 'normalized_spectrum', 'whiteline_peak_height_direct', 'weighted_eng', 'edge_pos_fit', 'edge0.5_pos_direct', 'pre_edge_sd', 'weighted_attenuation', 'whiteline_pos_fit', 'whiteline_pos_direct', 'edge_offset_filter', 'edge_jump_filter', 'edge0.5_pos_fit', 'centroid_of_eng_relative_to_wl', 'edge_pos_direct', 'pre_edge_mean', 'post_edge_sd'}:
            xana.calc_edge_0p5_direct(xanes2D_analysis_edge_0p5_fit_s, xanes2D_analysis_edge_0p5_fit_e)
        if 'whiteline_pos_fit' in {'centroid_of_eng', 'post_edge_mean', 'normalized_spectrum', 'whiteline_peak_height_direct', 'weighted_eng', 'edge_pos_fit', 'edge0.5_pos_direct', 'pre_edge_sd', 'weighted_attenuation', 'whiteline_pos_fit', 'whiteline_pos_direct', 'edge_offset_filter', 'edge_jump_filter', 'edge0.5_pos_fit', 'centroid_of_eng_relative_to_wl', 'edge_pos_direct', 'pre_edge_mean', 'post_edge_sd'}:
            xana.fit_whiteline(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e, 'scipy', **{'model': 'lorentzian', 'fvars': [1.0, 7729.0, 1.0], 'bnds': None, 'jac': '3-point', 'method': 'trf', 'ftol': 1e-07, 'xtol': 1e-07, 'gtol': 1e-07, 'ufac': 50})
        if 'whiteline_pos_direct' in {'centroid_of_eng', 'post_edge_mean', 'normalized_spectrum', 'whiteline_peak_height_direct', 'weighted_eng', 'edge_pos_fit', 'edge0.5_pos_direct', 'pre_edge_sd', 'weighted_attenuation', 'whiteline_pos_fit', 'whiteline_pos_direct', 'edge_offset_filter', 'edge_jump_filter', 'edge0.5_pos_fit', 'centroid_of_eng_relative_to_wl', 'edge_pos_direct', 'pre_edge_mean', 'post_edge_sd'}:
            xana.calc_whiteline_direct(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e)
        if 'whiteline_peak_height_direct' in {'centroid_of_eng', 'post_edge_mean', 'normalized_spectrum', 'whiteline_peak_height_direct', 'weighted_eng', 'edge_pos_fit', 'edge0.5_pos_direct', 'pre_edge_sd', 'weighted_attenuation', 'whiteline_pos_fit', 'whiteline_pos_direct', 'edge_offset_filter', 'edge_jump_filter', 'edge0.5_pos_fit', 'centroid_of_eng_relative_to_wl', 'edge_pos_direct', 'pre_edge_mean', 'post_edge_sd'}:
            xana.calc_direct_whiteline_peak_height(xanes2D_analysis_wl_fit_eng_s, xanes2D_analysis_wl_fit_eng_e)
        if ('centroid_of_eng' or 'centroid_of_eng_relative_to_wl' or 'weighted_attenuation' or 'weighted_eng') in {'centroid_of_eng', 'post_edge_mean', 'normalized_spectrum', 'whiteline_peak_height_direct', 'weighted_eng', 'edge_pos_fit', 'edge0.5_pos_direct', 'pre_edge_sd', 'weighted_attenuation', 'whiteline_pos_fit', 'whiteline_pos_direct', 'edge_offset_filter', 'edge_jump_filter', 'edge0.5_pos_fit', 'centroid_of_eng_relative_to_wl', 'edge_pos_direct', 'pre_edge_mean', 'post_edge_sd'}:
            xana.calc_weighted_eng(xanes2D_analysis_pre_edge_e)
        for jj in {'centroid_of_eng', 'post_edge_mean', 'normalized_spectrum', 'whiteline_peak_height_direct', 'weighted_eng', 'edge_pos_fit', 'edge0.5_pos_direct', 'pre_edge_sd', 'weighted_attenuation', 'whiteline_pos_fit', 'whiteline_pos_direct', 'edge_offset_filter', 'edge_jump_filter', 'edge0.5_pos_fit', 'centroid_of_eng_relative_to_wl', 'edge_pos_direct', 'pre_edge_mean', 'post_edge_sd'}:
            if jj == 'whiteline_pos_fit':
                _g12[jj][:] = np.float32(xana.wl_pos_fit)[:]
            if jj == 'whiteline_pos_direct':
                _g12[jj][:] = np.float32(xana.wl_pos_direct)[:]
            if jj == 'whiteline_peak_height_direct':
                _g12[jj][:] = np.float32(xana.direct_wl_ph)[:]
            if jj == 'centroid_of_eng':
                _g12[jj][:] = np.float32(xana.centroid_of_eng)[:]
            if jj == 'centroid_of_eng_relative_to_wl':
                _g12[jj][:] = np.float32(xana.centroid_of_eng_rel_wl)[:]
            if jj == 'weighted_attenuation':
                _g12[jj][:] = np.float32(xana.weighted_atten)[:]
            if jj == 'weighted_eng':
                _g12[jj][:] = np.float32(xana.weighted_eng)[:]
            if jj == 'edge0.5_pos_fit':
                _g12[jj][:] = np.float32(xana.edge_pos_0p5_fit)[:]
            if jj == 'edge_pos_fit':
                _g12[jj][:] = np.float32(xana.edge_pos_fit)[:]
            if jj == 'edge_pos_direct':
                _g12[jj][:] = np.float32(xana.edge_pos_direct)[:]
            if jj == 'edge_jump_filter':
                _g12[jj][:] = np.float32(xana.edge_jump_mask)[:]
            if jj == 'edge_offset_filter':
                _g12[jj][:] = np.float32(xana.fitted_edge_mask)[:]
            if jj == 'pre_edge_sd':
                _g12[jj][:] = np.float32(xana.pre_edge_sd_map)[:]
            if jj == 'post_edge_sd':
                _g12[jj][:] = np.float32(xana.post_edge_sd_map)[:]
            if jj == 'pre_edge_mean':
                _g12[jj][:] = np.float32(xana.pre_edge_mean_map)[:]
            if jj == 'post_edge_mean':
                _g12[jj][:] = np.float32(xana.post_edge_mean_map)[:]
            if jj  == 'pre_edge_fit_coef':
                _g12[jj][:] = np.float32(xana.pre_edge_fit_coef)[:]
            if jj == 'post_edge_fit_coef':
                _g12[jj][:] = np.float32(xana.post_edge_fit_coef)[:]
            if jj == 'edge_fit_coef':
                _g12[jj][:] = np.float32(xana.edge_fit_coef)[:]
            if jj == 'whiteline_fit_coef':
                _g12[jj][:] = np.float32(xana.wl_fit_coef)[:]
            if jj == 'normalized_spectrum':
                _g12[jj][:] = np.float32(xana.normalized_spectrum)[:]
print('xanes2D analysis is done!')

