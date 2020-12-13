import TXM_Sandbox.TXM_Sandbox.utils.xanes_regtools as xr
reg = xr.regtools(dtype='3D_XANES', mode='TRANSLATION')
reg.set_xanes3D_recon_path_template('/run/media/xiao/Data/data/3D_xanes/recon_fly_scan_id_{0}/recon_fly_scan_id_{0}_{1}.tiff')
reg.set_roi([541, 1006, 337, 846, 474, 637])
reg.apply_xanes3D_chunk_shift({'1': (-7.0, 8.5, 0.5), '0': (3.0, -18.0, 1.0), '2': (-6.0, 10.0, 1.0), '3': (-4.0, 10.0, 0.0), '4': (-2.0, 6.0, 1.0), '5': (-2.0, -0.5, 0.5), '6': (-2.0, 12.0, -1.5), '7': (-2.0, 14.5, -2.0), '8': (-1.0, 7.5, 1.0), '9': (0.0, 0.0, 0.0)}, 474, 637, trialfn='/run/media/xiao/Data/data/3D_xanes/3D_trial_reg_scan_id_56600-56609_2020-10-27-10-15-53.h5', savefn='/run/media/xiao/Data/data/3D_xanes/3D_trial_reg_scan_id_56600-56609_2020-10-27-10-15-53.h5')
