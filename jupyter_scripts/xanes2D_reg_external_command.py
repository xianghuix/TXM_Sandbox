import os
from TXM_Sandbox.TXM_Sandbox.utils import xanes_regtools as xr
from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()
    reg = xr.regtools(dtype='2D_XANES', method='LS+MRTV', mode='TRANSLATION')
    reg.set_xanes2D_raw_filename('/run/media/xiao/Data/data/Weijiang_2020Q1/data/multipos_2D_xanes_scan2_id_61201_repeat_00_pos_00.h5')
    reg.set_raw_data_info(**{'raw_h5_filename': '/run/media/xiao/Data/data/Weijiang_2020Q1/data/multipos_2D_xanes_scan2_id_61201_repeat_00_pos_00.h5', 'config_filename': '/run/media/xiao/Data/data/Weijiang_2020Q1/data/2D_trial_reg_multipos_2D_xanes_scan2_id_61201_repeat_00_pos_00_config_2020-11-03-10-09-25.json'})
    reg.set_method('LS+MRTV')
    reg.set_ref_mode('single')
    reg.set_roi([100, 980, 100, 1180])
    reg.set_indices(0, 101, 51)
    reg.set_xanes2D_tmp_filename('/home/xiao/software/anaconda3/user_packages/TXM_Sandbox/TXM_Sandbox/tmp/xanes2D_tmp.h5')
    reg.read_xanes2D_tmp_file(mode='reg')
    reg.set_reg_options(use_mask=False, mask_thres=0.0,                     use_chunk=True, chunk_sz=7,                     use_smooth_img=False, smooth_sigma=5.0,                     mrtv_level=5, mrtv_width=100,                      mrtv_sp_wz=8, mrtv_sp_step=0.5)
    reg.set_saving(os.path.dirname('/run/media/xiao/Data/data/Weijiang_2020Q1/data/2D_trial_reg_multipos_2D_xanes_scan2_id_61201_repeat_00_pos_00_2020-11-03-09-36-51.h5'),                      fn=os.path.basename('/run/media/xiao/Data/data/Weijiang_2020Q1/data/2D_trial_reg_multipos_2D_xanes_scan2_id_61201_repeat_00_pos_00_2020-11-03-09-36-51.h5'))
    reg.compose_dicts()
    reg.reg_xanes2D_chunk()
