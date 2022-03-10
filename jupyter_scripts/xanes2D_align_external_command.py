import TXM_Sandbox.TXM_Sandbox.utils.xanes_regtools as xr
reg = xr.regtools(dtype='2D_XANES', method='MPC', mode='TRANSLATION')
reg.set_roi([100, 980, 100, 1180])
reg.set_indices(0, 101, 51)
reg.set_xanes2D_tmp_filename('/home/xiao/software/anaconda3/user_packages/TXM_Sandbox/TXM_Sandbox/tmp/xanes2D_tmp.h5')
reg.read_xanes2D_tmp_file(mode='align')
reg.apply_xanes2D_chunk_shift({'0': (-0.5, -1.0), '1': (-1.5, 0.0), '2': (0.0, 2.0), '3': (-9.0, 11.0), '4': (1.5, -1.0), '5': (0.0, 0.0), '6': (-3.0, 1.0), '7': (1.0, 1.0), '8': (-1.0, 6.0), '9': (21.5, 2.5), '10': (0.5, 2.0), '11': (1.0, 2.5), '12': (-4.0, 10.0), '13': (-8.0, 16.0), '14': (-0.5, -5.5), '15': (0.0, -1.0), '16': (-2.0, 4.0), '17': (-2.0, 5.0), '18': (-1.5, 0.0), '19': (-1.5, 0.0), '20': (-2.0, 1.0), '21': (-1.0, 0.0), '22': (0.0, 4.0), '23': (0.0, 0.0), '24': (0.0, -1.5), '25': (0.0, -1.0), '26': (0.0, 0.0), '27': (0.0, 4.0), '28': (2.0, -1.0), '29': (1.0, 2.0), '30': (1.0, 0.0), '31': (1.0, -1.0), '32': (1.0, 0.0), '33': (0.0, -1.0), '34': (0.0, -1.0), '35': (0.0, 2.0), '36': (0.0, 2.0), '37': (0.0, -2.0), '38': (-1.0, 2.0), '39': (0.0, -2.0), '40': (1.0, 1.0), '41': (0.0, 1.0), '42': (0.0, 5.0), '43': (0.0, 3.0), '44': (1.0, 2.5), '45': (0.0, 1.0), '46': (0.0, 1.0), '47': (0.0, 4.0), '48': (0.0, 1.0), '49': (0.0, 4.0), '50': (0.0, 4.0), '51': (0.0, 4.0), '52': (0.0, 2.0), '53': (0.0, 3.0), '54': (0.0, -1.0), '55': (0.0, 0.0), '56': (0.0, -1.0), '57': (0.0, 1.0), '58': (-1.0, -1.0), '59': (-1.0, -1.0), '60': (0.0, 3.0), '61': (0.0, -1.0), '62': (0.0, 4.0), '63': (0.0, 0.0), '64': (0.0, -2.0), '65': (1.0, -5.0), '66': (0.0, -4.5), '67': (0.0, -2.0), '68': (0.0, -4.0), '69': (0.0, -3.0), '70': (-23.0, -6.0), '71': (-22.0, -2.0), '72': (-21.0, -6.0), '73': (-21.0, -6.0), '74': (-20.5, -6.0), '75': (-20.0, -6.0), '76': (0.0, -1.0), '77': (0.0, -1.5), '78': (0.0, 0.0), '79': (0.0, -1.0), '80': (0.0, 3.0), '81': (-0.5, -1.0), '82': (-1.5, -2.5), '83': (-1.0, -2.0), '84': (-1.0, -1.0), '85': (-1.0, -1.0), '86': (-1.0, 3.0), '87': (0.0, -1.0), '88': (4.0, -10.0), '89': (4.0, -10.0), '90': (2.0, -9.0), '91': (4.0, -10.0), '92': (1.0, -7.0), '93': (0.0, 0.0), '94': (7.0, -16.0), '95': (6.0, -13.0), '96': (5.0, -16.0), '97': (3.0, -21.0), '98': (0.0, -19.0), '99': (-3.0, -7.0), '100': (0.0, 0.0)},                      trialfn='/run/media/xiao/Data/data/Weijiang_2020Q1/data/2D_trial_reg_multipos_2D_xanes_scan2_id_61201_repeat_00_pos_00_2020-11-02-20-48-50.h5',                      savefn='/run/media/xiao/Data/data/Weijiang_2020Q1/data/2D_trial_reg_multipos_2D_xanes_scan2_id_61201_repeat_00_pos_00_2020-11-02-20-48-50.h5')