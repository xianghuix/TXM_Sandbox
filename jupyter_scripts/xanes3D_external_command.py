import os, h5py
import numpy as np
import xanes_math as xm
import xanes_analysis as xa
from copy import deepcopy

with h5py.File('/media/xiao_usb/3D_xanes/3D_trial_reg_scan_id_56600-56609_2020-06-21-15-29-52.h5', 'r+') as f:
    imgs = f['/registration_results/reg_results/registered_xanes3D'][:, 0, :, :]
    xanes3D_analysis_eng_list = f['/processed_XANES3D/proc_parameters/eng_list'][:]
    xanes3D_analysis_edge_eng = f['/processed_XANES3D/proc_parameters/edge_eng'][()]
    xanes3D_analysis_pre_edge_e = f['/processed_XANES3D/proc_parameters/pre_edge_e'][()]
    xanes3D_analysis_post_edge_s = f['/processed_XANES3D/proc_parameters/post_edge_s'][()]
    xanes3D_analysis_edge_jump_thres = f['/processed_XANES3D/proc_parameters/edge_jump_threshold'][()]
    xanes3D_analysis_edge_offset_thres = f['/processed_XANES3D/proc_parameters/edge_offset_threshold'][()]
    xanes3D_analysis_use_mask = f['/processed_XANES3D/proc_parameters/use_mask'][()]
    xanes3D_analysis_type = f['/processed_XANES3D/proc_parameters/analysis_type'][()]
    xanes3D_analysis_data_shape = f['/processed_XANES3D/proc_parameters/data_shape'][:]
    xanes3D_analysis_edge_0p5_fit_s = f['/processed_XANES3D/proc_parameters/edge_0p5_fit_s'][()]
    xanes3D_analysis_edge_0p5_fit_e = f['/processed_XANES3D/proc_parameters/edge_0p5_fit_e'][()]
    xanes3D_analysis_wl_fit_eng_s = f['/processed_XANES3D/proc_parameters/wl_fit_eng_s'][()]
    xanes3D_analysis_wl_fit_eng_e = f['/processed_XANES3D/proc_parameters/wl_fit_eng_e'][()]
    xanes3D_analysis_edge_fit_order = f['/processed_XANES3D/proc_parameters/pre_post_edge_norm_fit_order'][()]
    xanes3D_analysis_use_flt_spec = f['/processed_XANES3D/proc_parameters/flt_spec'][()]
    xana = xa.xanes_analysis(imgs, xanes3D_analysis_eng_list, xanes3D_analysis_edge_eng, pre_ee=xanes3D_analysis_pre_edge_e, post_es=xanes3D_analysis_post_edge_s, edge_jump_threshold=xanes3D_analysis_edge_jump_thres, pre_edge_threshold=xanes3D_analysis_edge_offset_thres)
    if '/processed_XANES3D/proc_spectrum' in f:
        del f['/processed_XANES3D/proc_spectrum']
        g12 = f.create_group('/processed_XANES3D/proc_spectrum')
    else:
        g12 = f.create_group('/processed_XANES3D/proc_spectrum')

    if xanes3D_analysis_type == 'wl':
        _g12 = {}
        for jj in {'whiteline_pos_fit', 'whiteline_pos_direct'}:
            if jj in ['pre_edge_fit_coef', 'post_edge_fit_coef', 'whiteline_fit_coef']:
                _g12[jj] = g12.create_dataset(jj, shape=(list(len([1.0, 8350.499007209824, 1.0])) + list(xanes3D_analysis_data_shape[1:])), dtype=np.float32)
            elif jj == 'edge_fit_coef':
                _g12[jj] = g12.create_dataset(jj, shape=(list(len([])) + list(xanes3D_analysis_data_shape[1:])), dtype=np.float32)
            else:
                _g12[jj] = g12.create_dataset(jj, shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)
        for ii in range(xanes3D_analysis_data_shape[1]):
            imgs[:] = f['/registration_results/reg_results/registered_xanes3D'][:, ii, :, :]
            xana.spectrum[:] = imgs[:]
            if True:
                xana.fit_whiteline(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, 'scipy', **{'model': 'lorentzian', 'fvars': [1.0, 8350.499007209824, 1.0], 'bnds': None, 'jac': '3-point', 'method': 'trf', 'ftol': 1e-07, 'xtol': 1e-07, 'gtol': 1e-07, 'ufac': 50})
            if False:
                xana.fit_edge(xanes3D_analysis_edge_0p5_fit_s, xanes3D_analysis_edge_0p5_fit_e, 'numpy', **{})
            if False:
                xana.find_edge(xanes3D_analysis_edge_0p5_fit_s, xanes3D_analysis_edge_0p5_fit_e)
            xana.calc_whiteline_direct(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e)
            for jj in {'whiteline_pos_fit', 'whiteline_pos_direct'}:
                if jj == 'whiteline_pos_fit':
                    _g12[jj][ii] = np.float32(xana.wl_pos_fit)[:]
                if jj == 'whiteline_pos_direct':
                    _g12[jj][ii] = np.float32(xana.wl_pos_direct)[:]
                if jj == 'edge_pos_fit':
                    _g12[jj][ii] = np.float32(xana.edge_pos_fit)[:]
                if jj == 'edge_pos_direct':
                    _g12[jj][ii] = np.float32(xana.edge_pos_direct)[:]
                if jj == 'edge_fit_coef':
                    _g12[jj][ii] = np.float32(xana.edge_fit_coef)[:]
                if jj == 'whiteline_fit_coef':
                    _g12[jj][ii] = np.float32(xana.wl_fit_coef)[:]
            print(ii)
    elif xanes3D_analysis_type == 'full':
        _g12 = {}
        for jj in {'whiteline_pos_fit', 'whiteline_pos_direct'}:
            if jj in ['pre_edge_fit_coef', 'post_edge_fit_coef', 'whiteline_fit_coef']:
                _g12[jj] = g12.create_dataset(jj, shape=(list(len([1.0, 8350.499007209824, 1.0])) + list(xanes3D_analysis_data_shape[1:])), dtype=np.float32)
            elif jj == 'edge_fit_coef':
                _g12[jj] = g12.create_dataset(jj, shape=(list(len([])) + list(xanes3D_analysis_data_shape[1:])), dtype=np.float32)
            elif jj == 'normalized_spectrum':
                _g12[jj] = g12.create_dataset(jj, shape=(xanes3D_analysis_data_shape), dtype=np.float32)
            else:
                _g12[jj] = g12.create_dataset(jj, shape=(xanes3D_analysis_data_shape[1:]), dtype=np.float32)
        for ii in range(xanes3D_analysis_data_shape[1]):
            imgs[:] = f['/registration_results/reg_results/registered_xanes3D'][:, ii, :, :]
            xana.spectrum[:] = imgs[:]
            xana.fit_pre_edge()
            xana.fit_post_edge()
            xana.cal_edge_jump_map()
            xana.cal_pre_edge_sd()
            xana.cal_post_edge_sd()
            xana.cal_pre_edge_mean()
            xana.cal_post_edge_mean()
            xana.create_edge_jump_filter(xanes3D_analysis_edge_jump_thres)
            xana.create_fitted_edge_filter(xanes3D_analysis_edge_offset_thres)
            xana.normalize_xanes(xanes3D_analysis_edge_eng, order=xanes3D_analysis_edge_fit_order, save_pre_post=True)
            if ('edge0.5_pos_fit' in {'whiteline_pos_fit', 'whiteline_pos_direct'}) and ('edge_pos_fit' in {'whiteline_pos_fit', 'whiteline_pos_direct'}):
                tem = deepcopy({})
                tem['cal_deriv'] = True
                xana.fit_edge_0p5(xanes3D_analysis_edge_0p5_fit_s, xanes3D_analysis_edge_0p5_fit_e, 'numpy', **tem)
            elif ('edge0.5_pos_fit' in {'whiteline_pos_fit', 'whiteline_pos_direct'}) and not ('edge_pos_fit' in {'whiteline_pos_fit', 'whiteline_pos_direct'}):
                xana.fit_edge(xanes3D_analysis_edge_0p5_fit_s, xanes3D_analysis_edge_0p5_fit_e, 'numpy', **{})
            elif not ('edge0.5_pos_fit' in {'whiteline_pos_fit', 'whiteline_pos_direct'}) and ('edge_pos_fit' in {'whiteline_pos_fit', 'whiteline_pos_direct'}):
                tem = deepcopy({})
                tem['cal_deriv'] = False
                xana.fit_edge_0p5(xanes3D_analysis_edge_0p5_fit_s, xanes3D_analysis_edge_0p5_fit_e, 'numpy', **tem)
            if 'edge_pos_direct' in {'whiteline_pos_fit', 'whiteline_pos_direct'}:
                xana.find_edge()
            if 'edge0.5_pos_direct' in {'whiteline_pos_fit', 'whiteline_pos_direct'}:
                xana.calc_edge_0p5_direct(xanes3D_analysis_edge_0p5_fit_s, xanes3D_analysis_edge_0p5_fit_e)
            if 'whiteline_pos_fit' in {'whiteline_pos_fit', 'whiteline_pos_direct'}:
                xana.fit_whiteline(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e, 'scipy', **{'model': 'lorentzian', 'fvars': [1.0, 8350.499007209824, 1.0], 'bnds': None, 'jac': '3-point', 'method': 'trf', 'ftol': 1e-07, 'xtol': 1e-07, 'gtol': 1e-07, 'ufac': 50})
            if 'whiteline_pos_direct' in {'whiteline_pos_fit', 'whiteline_pos_direct'}:
                xana.calc_whiteline_direct(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e)
            if 'whiteline_peak_height_direct' in {'whiteline_pos_fit', 'whiteline_pos_direct'}:
                xana.calc_direct_whiteline_peak_height(xanes3D_analysis_wl_fit_eng_s, xanes3D_analysis_wl_fit_eng_e)
            if ('centroid_of_eng' or 'centroid_of_eng_relative_to_wl' or 'weighted_attenuation' or 'weighted_eng') in {'whiteline_pos_fit', 'whiteline_pos_direct'}:
                xana.calc_weighted_eng(xanes3D_analysis_pre_edge_e)

            for jj in {'whiteline_pos_fit', 'whiteline_pos_direct'}:
                if jj == 'whiteline_pos_fit':
                    _g12[jj][ii] = np.float32(xana.wl_pos_fit)[:]
                if jj == 'whiteline_pos_direct':
                    _g12[jj][ii] = np.float32(xana.wl_pos_direct)[:]
                if jj == 'whiteline_peak_height_direct':
                    _g12[jj][ii] = np.float32(xana.direct_wl_ph)[:]
                if jj == 'centroid_of_eng':
                    _g12[jj][ii] = np.float32(xana.centroid_of_eng)[:]
                if jj == 'centroid_of_eng_relative_to_wl':
                    _g12[jj][ii] = np.float32(xana.centroid_of_eng_rel_wl)[:]
                if jj == 'weighted_attenuation':
                    _g12[jj][ii] = np.float32(xana.weighted_atten)[:]
                if jj == 'weighted_eng':
                    _g12[jj][ii] = np.float32(xana.weighted_eng)[:]
                if jj == 'edge0.5_pos_fit':
                    _g12[jj][ii] = np.float32(xana.edge_pos_0p5_fit)[:]
                if (jj == 'edge_pos_fit') and False:
                    _g12[jj][ii] = np.float32(xana.edge_pos_fit)[:]
                if jj == 'edge_pos_direct':
                    _g12[jj][ii] = np.float32(xana.edge_pos_direct)[:]
                if jj == 'edge_jump_filter':
                    _g12[jj][ii] = np.float32(xana.edge_jump_mask)[:]
                if jj == 'edge_offset_filter':
                    _g12[jj][ii] = np.float32(xana.fitted_edge_mask)[:]
                if jj == 'pre_edge_sd':
                    _g12[jj][ii] = np.float32(xana.pre_edge_sd_map)[:]
                if jj == 'post_edge_sd':
                    _g12[jj][ii] = np.float32(xana.post_edge_sd_map)[:]
                if jj == 'pre_edge_mean':
                    _g12[jj][ii] = np.float32(xana.pre_edge_mean_map)[:]
                if jj == 'post_edge_mean':
                    _g12[jj][ii] = np.float32(xana.post_edge_mean_map)[:]
                if jj  == 'pre_edge_fit_coef':
                    _g12[jj][ii] = np.float32(xana.pre_edge_fit_coef)[:]
                if jj == 'post_edge_fit_coef':
                    _g12[jj][ii] = np.float32(xana.post_edge_fit_coef)[:]
                if jj == 'edge_fit_coef':
                    _g12[jj][ii] = np.float32(xana.edge_fit_coef)[:]
                if jj == 'whiteline_fit_coef':
                    _g12[jj][ii] = np.float32(xana.wl_fit_coef)[:]
                if jj == 'normalized_spectrum':
                    _g12[jj][ii] = np.float32(xana.normalized_spectrum)[:]
            print(ii)
print('xanes3D analysis is done!')
