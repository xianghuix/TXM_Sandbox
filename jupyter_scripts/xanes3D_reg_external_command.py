import os
from TXM_Sandbox.TXM_Sandbox.utils import xanes_regtools as xr
reg = xr.regtools(dtype='3D_XANES', method='MRTV', mode='TRANSLATION')
from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()
    reg.set_raw_data_info(**{'raw_h5_top_dir': '/run/media/xiao/Data/data/3D_xanes', 'recon_top_dir': '/run/media/xiao/Data/data/3D_xanes'})
    reg.set_method('MRTV')
    reg.set_ref_mode('single')
    reg.set_xanes3D_tmp_filename('/home/xiao/software/anaconda3/user_packages/TXM_Sandbox/TXM_Sandbox/tmp/xanes3D_tmp.h5')
    reg.read_xanes3D_tmp_file()
    reg.set_xanes3D_raw_h5_top_dir('/run/media/xiao/Data/data/3D_xanes')
    reg.set_indices(56600, 56610, 56605)
    reg.set_reg_options(use_mask=False, mask_thres=0.0,                     use_chunk=True, chunk_sz=7,                     use_smooth_img=False, smooth_sigma=0,                     mrtv_level=6, mrtv_width=10,                      mrtv_sp_wz=8, mrtv_sp_step=0.5)
    reg.set_roi([541, 1006, 337, 846, 474, 637])
    reg.set_xanes3D_recon_path_template('/run/media/xiao/Data/data/3D_xanes/recon_fly_scan_id_{0}/recon_fly_scan_id_{0}_{1}.tiff')
    reg.set_saving(os.path.dirname('/run/media/xiao/Data/data/3D_xanes/3D_trial_reg_scan_id_56600-56609_2020-10-27-10-15-53.h5'),                      fn=os.path.basename('/run/media/xiao/Data/data/3D_xanes/3D_trial_reg_scan_id_56600-56609_2020-10-27-10-15-53.h5'))
    reg.xanes3D_sli_search_half_range = 20
    reg.xanes3D_recon_fixed_sli = 552
    reg.compose_dicts()
    reg.reg_xanes3D_chunk()
