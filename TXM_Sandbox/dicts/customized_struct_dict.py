XANES_PEAK_LINE_SHAPES = [
    'lorentzian', 'gaussian', 'voigt', 'pvoigt', 'moffat', 'pearson7',
    'breit_wigner', 'damped_oscillator', 'dho', 'lognormal', 'students_t',
    'expgaussian', 'donaich', 'skewed_gaussian', 'skewed_voigt', 'step',
    'rectangle', 'parabolic', 'sine', 'expsine', 'split_lorentzian'
]

XANES_STEP_LINE_SHAPES = ['logistic', 'exponential', 'powerlaw', 'linear']

XANES_PEAK_FIT_PARAM_DICT = {
    "parabolic": {
        0: ["a", 1, "a: ampflitude in parabolic function"],
        1: ["b", 0, "b: center of parabolic function"],
        2: ["c", 1, "c: standard deviation of parabolic function"]
    },
    "gaussian": {
        0: ["amp", 1, "amp: ampflitude in gaussian function"],
        1: ["cen", 0, "cen: center of gaussian function"],
        2: ["sig", 1, "sig: standard deviation of gaussian function"]
    },
    "lorentzian": {
        0: ["amp", 1, "amp: ampflitude in lorentzian function"],
        1: ["cen", 0, "cen: center of lorentzian function"],
        2: ["sig", 1, "sig: standard deviation of lorentzian function"]
    },
    "damped_oscillator": {
        0: ["amp", 1, "amp: ampflitude in damped_oscillator function"],
        1: ["cen", 0, "cen: center of damped_oscillator function"],
        2: ["sig", 1, "sig: standard deviation of damped_oscillator function"]
    },
    "lognormal": {
        0: ["amp", 1, "amp: ampflitude in lognormal function"],
        1: ["cen", 0, "cen: center of lognormal function"],
        2: ["sig", 1, "sig: standard deviation of lognormal function"]
    },
    "students_t": {
        0: ["amp", 1, "amp: ampflitude in students_t function"],
        1: ["cen", 0, "cen: center of students_t function"],
        2: ["sig", 1, "sig: standard deviation of students_t function"]
    },
    "sine": {
        0: ["amp", 1, "amp: ampflitude in sine function"],
        1: ["frq", 1, "frq: freqency in sine function"],
        2: ["shft", 0, "shft: shift in sine function"]
    },
    "voigt": {
        0: ["amp", 1, "amp: ampflitude in voigt function"],
        1: ["cen", 0, "cen: center of voigt function"],
        2: ["sig", 1, "sig: standard voigt of gaussian function"],
        3: ["gamma", 0, "gamma: "]
    },
    "split_lorentzian": {
        0: ["amp", 1, "amp: ampflitude in split_lorentzian function"],
        1: ["cen", 0, "cen: center of split_lorentzian function"],
        2: ["sig", 1, "sig: standard deviation of split_lorentzian function"],
        3: [
            "sigr", 1,
            "sigr: standard deviation of the right-hand side half in split_lorentzian function"
        ]
    },
    "pvoigt": {
        0: ["amp", 1, "amp: ampflitude in pvoigt function"],
        1: ["cen", 0, "cen: center of pvoigt function"],
        2: ["sig", 1, "sig: standard pvoigt of gaussian function"],
        3: ["frac", 0, "frac: "]
    },
    "moffat": {
        0: ["amp", 1, "amp: ampflitude in moffat function"],
        1: ["cen", 0, "cen: center of moffat function"],
        2: ["sig", 1, "sig: standard moffat of gaussian function"],
        3: ["beta", 0, "beta: "]
    },
    "pearson7": {
        0: ["amp", 1, "amp: ampflitude in pearson7 function"],
        1: ["cen", 0, "cen: center of pearson7 function"],
        2: ["sig", 1, "sig: standard pearson7 of gaussian function"],
        3: ["expo", 0, "expo: "]
    },
    "breit_wigner": {
        0: ["amp", 1, "amp: ampflitude in breit_wigner function"],
        1: ["cen", 0, "cen: center of breit_wigner function"],
        2: ["sig", 1, "sig: standard breit_wigner of gaussian function"],
        3: ["q", 0, "q: "]
    },
    "dho": {
        0: ["amp", 1, "amp: ampflitude in dho function"],
        1: ["cen", 0, "cen: center of dho function"],
        2: ["sig", 1, "sig: standard dho of gaussian function"],
        3: ["gama", 1, "gama: "]
    },
    "expgaussian": {
        0: ["amp", 1, "amp: ampflitude in expgaussian function"],
        1: ["cen", 0, "cen: center of expgaussian function"],
        2: ["sig", 1, "sig: standard expgaussian of gaussian function"],
        3: ["gama", 1, "gama: "]
    },
    "donaich": {
        0: ["amp", 1, "amp: ampflitude in donaich function"],
        1: ["cen", 0, "cen: center of donaich function"],
        2: ["sig", 1, "sig: standard donaich of gaussian function"],
        3: ["gama", 0, "gama: "]
    },
    "skewed_gaussian": {
        0: ["amp", 1, "amp: ampflitude in skewed_gaussian function"],
        1: ["cen", 0, "cen: center of skewed_gaussian function"],
        2: ["sig", 1, "sig: standard skewed_gaussian of gaussian function"],
        3: ["gama", 0, "gama: "]
    },
    "expsine": {
        0: ["amp", 1, "amp: ampflitude in expsine function"],
        1: ["frq", 1, "frq:  width of expsine function"],
        2: ["shft", 0, "shft: center of gaussian function"],
        3: ["dec", 0, "dec: exponential decay factor"]
    },
    "step": {
        0: ["amp", 1, "amp: ampflitude in step function"],
        1: ["cen", 0, "cen: center of step function"],
        2: ["sig", 1, "sig: standard step of gaussian function"],
        5: ["form", "linear", "form: "]
    },
    "skewed_voigt": {
        0: ["amp", 1, "amp: ampflitude in skewed_voigt function"],
        1: ["cen", 0, "cen: center of skewed_voigt function"],
        2: ["sig", 1, "sig: standard skewed_voigt of gaussian function"],
        3: ["gamma", 0, "gamma: "],
        4: ["skew", 0, "skew: "]
    },
    "rectangle": {
        0: ["amp", 1, "amp: ampflitude in rectangle function"],
        1: ["cen1", 0, "cen1: center of rectangle function"],
        2: ["sig1", 1, "sig1: standard deviation of rectangle function"],
        3: ["cen2", 0, "cen2: center of rectangle function"],
        4: ["sig2", 1, "sig2: standard deviation of rectangle function"],
        5: ["form", "linear", "form: "]
    },
}

XANES_PEAK_FIT_PARAM_BND_DICT = {
    "parabolic": {
        0: ["a", [-1e3, 1e3], "a: ampflitude in parabolic function"],
        1: ["b", [-1e3, 1e3], "b: center of parabolic function"],
        2: ["c", [-1e5, 1e5], "c: standard deviation of parabolic function"]
    },
    "gaussian": {
        0: ["amp", [-10, 10], "amp: ampflitude in gaussian function"],
        1: ["cen", [-2, 2], "cen: center of gaussian function"],
        2: ["sig", [0, 1e3], "sig: standard deviation of gaussian function"]
    },
    "lorentzian": {
        0: ["amp", [-10, 10], "amp: ampflitude in lorentzian function"],
        1: ["cen", [-2, 2], "cen: center of lorentzian function"],
        2: ["sig", [0, 1e3], "sig: standard deviation of lorentzian function"]
    },
    "damped_oscillator": {
        0: ["amp", [-10, 10], "amp: ampflitude in damped_oscillator function"],
        1: ["cen", [-2, 2], "cen: center of damped_oscillator function"],
        2: [
            "sig", [0, 1e3],
            "sig: standard deviation of damped_oscillator function"
        ]
    },
    "lognormal": {
        0: ["amp", [-10, 10], "amp: ampflitude in lognormal function"],
        1: ["cen", [-2, 2], "cen: center of lognormal function"],
        2: ["sig", [0, 1e3], "sig: standard deviation of lognormal function"]
    },
    "students_t": {
        0: ["amp", [-10, 10], "amp: ampflitude in students_t function"],
        1: ["cen", [-2, 2], "cen: center of students_t function"],
        2: ["sig", [0, 1e3], "sig: standard deviation of students_t function"]
    },
    "sine": {
        0: ["amp", [-10, 10], "amp: ampflitude in sine function"],
        1: ["frq", [0, 1], "frq: freqency in sine function"],
        2: ["shft", [0, 1], "shft: shift in sine function"]
    },
    "voigt": {
        0: ["amp", [-10, 10], "amp: ampflitude in voigt function"],
        1: ["cen", [-2, 2], "cen: center of voigt function"],
        2: ["sig", [0, 1e3], "sig: standard voigt of gaussian function"],
        3: ["gamma", [0, 1e3], "gamma: "]
    },
    "split_lorentzian": {
        0: ["amp", [-10, 10], "amp: ampflitude in split_lorentzian function"],
        1: ["cen", [-2, 2], "cen: center of split_lorentzian function"],
        2: [
            "sig", [0, 1e3],
            "sig: standard deviation of split_lorentzian function"
        ],
        3: [
            "sigr", [0, 1e3],
            "sigr: standard deviation of the right-hand side half in split_lorentzian function"
        ]
    },
    "pvoigt": {
        0: ["amp", [-10, 10], "amp: ampflitude in pvoigt function"],
        1: ["cen", [-2, 2], "cen: center of pvoigt function"],
        2: ["sig", [0, 1e3], "sig: standard pvoigt of gaussian function"],
        3: ["frac", [0, 1], "frac: "]
    },
    "moffat": {
        0: ["amp", [-10, 10], "amp: ampflitude in moffat function"],
        1: ["cen", [-2, 2], "cen: center of moffat function"],
        2: ["sig", [0, 1e3], "sig: standard moffat of gaussian function"],
        3: ["beta", [-1e3, 1e3], "beta: "]
    },
    "pearson7": {
        0: ["amp", [-10, 10], "amp: ampflitude in pearson7 function"],
        1: ["cen", [-2, 2], "cen: center of pearson7 function"],
        2: ["sig", [0, 1e3], "sig: standard pearson7 of gaussian function"],
        3: ["expo", [-1e2, 1e2], "expo: "]
    },
    "breit_wigner": {
        0: ["amp", [-10, 10], "amp: ampflitude in breit_wigner function"],
        1: ["cen", [-2, 2], "cen: center of breit_wigner function"],
        2:
        ["sig", [0, 1e3], "sig: standard breit_wigner of gaussian function"],
        3: ["q", [-10, 10], "q: "]
    },
    "dho": {
        0: ["amp", [-10, 10], "amp: ampflitude in dho function"],
        1: ["cen", [-2, 2], "cen: center of dho function"],
        2: ["sig", [0, 1e3], "sig: standard dho of gaussian function"],
        3: ["gama", [-10, 10], "gama: "]
    },
    "expgaussian": {
        0: ["amp", [-10, 10], "amp: ampflitude in expgaussian function"],
        1: ["cen", [-2, 2], "cen: center of expgaussian function"],
        2: ["sig", [0, 1e3], "sig: standard expgaussian of gaussian function"],
        3: ["gama", [-10, 10], "gama: "]
    },
    "donaich": {
        0: ["amp", [-10, 10], "amp: ampflitude in donaich function"],
        1: ["cen", [-2, 2], "cen: center of donaich function"],
        2: ["sig", [0, 1e3], "sig: standard donaich of gaussian function"],
        3: ["gama", [-10, 10], "gama: "]
    },
    "skewed_gaussian": {
        0: ["amp", [-10, 10], "amp: ampflitude in skewed_gaussian function"],
        1: ["cen", [-2, 2], "cen: center of skewed_gaussian function"],
        2: [
            "sig", [0, 1e3],
            "sig: standard skewed_gaussian of gaussian function"
        ],
        3: ["gama", 0, "gama: "]
    },
    "expsine": {
        0: ["amp", [-10, 10], "amp: ampflitude in expsine function"],
        1: ["frq", [0, 1], "frq: center of expsine function"],
        2: ["shft", [0, 1], "shft: standard expsine of gaussian function"],
        3: ["dec", [-10, 10], "dec: "]
    },
    "step": {
        0: ["amp", [-10, 10], "amp: ampflitude in step function"],
        1: ["cen", [-2, 2], "cen: center of step function"],
        2: ["sig", [0, 1e3], "sig: standard step of gaussian function"]
    },
    "skewed_voigt": {
        0: ["amp", [-10, 10], "amp: ampflitude in skewed_voigt function"],
        1: ["cen", [-2, 2], "cen: center of skewed_voigt function"],
        2:
        ["sig", [0, 1e3], "sig: standard skewed_voigt of gaussian function"],
        3: ["gamma", [0, 1e-3], "gamma: "],
        4: ["skew", [-10, 10], "skew: "]
    },
    "rectangle": {
        0: ["amp", [-10, 10], "amp: ampflitude in rectangle function"],
        1: ["cen1", [-2, 2], "cen1: center of rectangle function"],
        2:
        ["sig1", [0, 1e3], "sig1: standard deviation of rectangle function"],
        3: ["cen2", [-2, 2], "cen2: center of rectangle function"],
        4:
        ["sig2", [0, 1e3], "sig2: standard deviation of rectangle function"]
    }
}

XANES_EDGE_LINE_SHAPES = [
    'lorentzian', 'split_lorentzian', 'voigt', 'pvoigt', 'skewed_voigt',
    'gaussian', 'skewed_gaussian', 'expgaussian', 'sine', 'expsine'
]

XANES_EDGE_FIT_PARAM_DICT = {
    "gaussian": {
        0: ["amp", 1, "amp: ampflitude in gaussian function"],
        1: ["cen", 0, "cen: center of gaussian function"],
        2: ["sig", 1, "sig: standard deviation of gaussian function"]
    },
    "lorentzian": {
        0: ["amp", 1, "amp: ampflitude in lorentzian function"],
        1: ["cen", 0, "cen: center of lorentzian function"],
        2: ["sig", 1, "sig: standard deviation of lorentzian function"]
    },
    "sine": {
        0: ["amp", 1, "amp: ampflitude in sine function"],
        1: ["frq", 1, "frq: freqency in sine function"],
        2: ["shft", 0, "shft: shift in sine function"]
    },
    "voigt": {
        0: ["amp", 1, "amp: ampflitude in voigt function"],
        1: ["cen", 0, "cen: center of voigt function"],
        2: ["sig", 1, "sig: standard voigt of gaussian function"],
        3: ["gamma", 0, "gamma: "]
    },
    "split_lorentzian": {
        0: ["amp", 1, "amp: ampflitude in split_lorentzian function"],
        1: ["cen", 0, "cen: center of split_lorentzian function"],
        2: ["sig", 1, "sig: standard deviation of split_lorentzian function"],
        3: [
            "sigr", 1,
            "sigr: standard deviation of the right-hand side half in split_lorentzian function"
        ]
    },
    "pvoigt": {
        0: ["amp", 1, "amp: ampflitude in pvoigt function"],
        1: ["cen", 0, "cen: center of pvoigt function"],
        2: ["sig", 1, "sig: standard pvoigt of gaussian function"],
        3: ["frac", 0, "frac: "]
    },
    "expgaussian": {
        0: ["amp", 1, "amp: ampflitude in expgaussian function"],
        1: ["cen", 0, "cen: center of expgaussian function"],
        2: ["sig", 1, "sig: standard expgaussian of gaussian function"],
        3: ["gama", 1, "gama: "]
    },
    "skewed_gaussian": {
        0: ["amp", 1, "amp: ampflitude in skewed_gaussian function"],
        1: ["cen", 0, "cen: center of skewed_gaussian function"],
        2: ["sig", 1, "sig: standard skewed_gaussian of gaussian function"],
        3: ["gama", 0, "gama: "]
    },
    "expsine": {
        0: ["amp", 1, "amp: ampflitude in expsine function"],
        1: ["frq", 1, "frq:  width of expsine function"],
        2: ["shft", 0, "shft: center of gaussian function"],
        3: ["dec", 0, "dec: exponential decay factor"]
    },
    "skewed_voigt": {
        0: ["amp", 1, "amp: ampflitude in skewed_voigt function"],
        1: ["cen", 0, "cen: center of skewed_voigt function"],
        2: ["sig", 1, "sig: standard skewed_voigt of gaussian function"],
        3: ["gamma", 0, "gamma: "],
        4: ["skew", 0, "skew: "]
    }
}

XANES_EDGE_FIT_PARAM_BND_DICT = {
    "gaussian": {
        0: ["amp", [-10, 10], "amp: ampflitude in gaussian function"],
        1: ["cen", [0, 1], "cen: center of gaussian function"],
        2: ["sig", [0, 1e3], "sig: standard deviation of gaussian function"]
    },
    "lorentzian": {
        0: ["amp", [-10, 10], "amp: ampflitude in lorentzian function"],
        1: ["cen", [0, 1], "cen: center of lorentzian function"],
        2: ["sig", [0, 1e3], "sig: standard deviation of lorentzian function"]
    },
    "sine": {
        0: ["amp", [-10, 10], "amp: ampflitude in sine function"],
        1: ["frq", [0, 1], "frq: freqency in sine function"],
        2: ["shft", [0, 1], "shft: shift in sine function"]
    },
    "voigt": {
        0: ["amp", [-10, 10], "amp: ampflitude in voigt function"],
        1: ["cen", [0, 1], "cen: center of voigt function"],
        2: ["sig", [0, 1e3], "sig: standard voigt of gaussian function"],
        3: ["gamma", [0, 1e3], "gamma: "]
    },
    "split_lorentzian": {
        0: ["amp", [-10, 10], "amp: ampflitude in split_lorentzian function"],
        1: ["cen", [0, 1], "cen: center of split_lorentzian function"],
        2: [
            "sig", [0, 1e3],
            "sig: standard deviation of split_lorentzian function"
        ],
        3: [
            "sigr", [0, 1e3],
            "sigr: standard deviation of the right-hand side half in split_lorentzian function"
        ]
    },
    "pvoigt": {
        0: ["amp", [-10, 10], "amp: ampflitude in pvoigt function"],
        1: ["cen", [0, 1], "cen: center of pvoigt function"],
        2: ["sig", [0, 1e3], "sig: standard pvoigt of gaussian function"],
        3: ["frac", [0, 1], "frac: "]
    },
    "expgaussian": {
        0: ["amp", [-10, 10], "amp: ampflitude in expgaussian function"],
        1: ["cen", [0, 1], "cen: center of expgaussian function"],
        2: ["sig", [0, 1e3], "sig: standard expgaussian of gaussian function"],
        3: ["gama", [-10, 10], "gama: "]
    },
    "skewed_gaussian": {
        0: ["amp", [-10, 10], "amp: ampflitude in skewed_gaussian function"],
        1: ["cen", [0, 1], "cen: center of skewed_gaussian function"],
        2: [
            "sig", [0, 1e3],
            "sig: standard skewed_gaussian of gaussian function"
        ],
        3: ["gama", 0, "gama: "]
    },
    "expsine": {
        0: ["amp", [-10, 10], "amp: ampflitude in expsine function"],
        1: ["frq", [0, 1], "frq: center of expsine function"],
        2: ["shft", [0, 1], "shft: standard expsine of gaussian function"],
        3: ["dec", [-10, 10], "dec: "]
    },
    "skewed_voigt": {
        0: ["amp", [-10, 10], "amp: ampflitude in skewed_voigt function"],
        1: ["cen", [0, 1], "cen: center of skewed_voigt function"],
        2:
        ["sig", [0, 1e3], "sig: standard skewed_voigt of gaussian function"],
        3: ["gamma", [0, 1e-3], "gamma: "],
        4: ["skew", [-10, 10], "skew: "]
    }
}

# XANES_FULL_SAVE_ITEM_OPTIONS = [
#     'norm_spec', 'wl_pos_fit', 'wl_fit_err', 'wl_pos_dir',
#     'wl_peak_height_dir', 'centroid_of_eng', 'centroid_of_eng_relative_to_wl',
#     'weighted_attenuation', 'weighted_eng', 'edge50_pos_fit', 'edge50_pos_dir',
#     'edge_pos_fit', 'edge_fit_err', 'edge_pos_dir', 'edge_jump_filter',
#     'edge_offset_filter', 'pre_edge_sd', 'pre_edge_mean', 'post_edge_sd',
#     'post_edge_mean', 'pre_edge_fit_coef', 'post_edge_fit_coef', 'wl_fit_coef',
#     'edge_fit_coef', 'lcf_fit', 'lcf_fit_err'
# ]

XANES_FULL_SAVE_ITEM_OPTIONS = [
    'norm_spec', 'wl_pos_fit', 'wl_fit_err', 'wl_pos_dir',
    'wl_peak_height_dir', 'centroid_of_eng', 'centroid_of_eng_relative_to_wl',
    'weighted_attenuation', 'weighted_eng', 'edge50_pos_fit', 'edge50_pos_dir',
    'edge_pos_fit', 'edge_fit_err', 'edge_pos_dir', 'pre_edge_sd',
    'pre_edge_mean', 'post_edge_sd', 'post_edge_mean', 'pre_edge_fit_coef',
    'post_edge_fit_coef', 'wl_fit_coef', 'edge_fit_coef', 'lcf_fit',
    'lcf_fit_err'
]

XANES_FULL_SAVE_DEFAULT = [
    '', 'norm_spec', 'wl_pos_fit', 'wl_fit_err', 'centroid_of_eng',
    'centroid_of_eng_relative_to_wl', 'weighted_attenuation', 'weighted_eng',
    'edge50_pos_fit', 'edge_pos_fit', 'edge_fit_err', 'pre_edge_sd',
    'pre_edge_mean', 'post_edge_sd', 'post_edge_mean'
]

XANES_WL_SAVE_ITEM_OPTIONS = [
    'wl_fit_coef',
    'wl_fit_err',
    'wl_pos_dir',
    'wl_pos_fit',
    'centroid_of_eng',
    'centroid_of_eng_relative_to_wl',
    'weighted_attenuation',
    'weighted_eng',
]

XANES_WL_SAVE_DEFAULT = [
    '',
    'wl_fit_err',
    'wl_pos_fit',
    'weighted_attenuation',
]

TOMO_H5_ITEM_DICT = {
    'img_bkg': {
        'description':
        'illumination beam reference images for normalizing sample images',
        'dtype': 'np.uint16'
    },
    'img_dark': {
        'description':
        'no-beam reference images for removing background noises in sample images',
        'dtype': 'np.uint16'
    },
    'img_tomo': {
        'description': 'sample images taken at different rotation angles',
        'dtype': 'np.uint16'
    },
    'angle': {
        'description': 'angles at which sample images are taken',
        'dtype': 'np.float64'
    }
}

XANES2D_ANA_ITEM_DICT = {
    'mask': {
        'description': 'a mask for isolating sample area from the background',
        'path': '/processed_XANES2D/gen_masks/{0}/{0}',
        'dtype': 'np.int8'
    },
    'registered_xanes2D': {
        'description': 'aligned 2D spectra image; it is a 3D data array',
        'path': '/registration_results/reg_results/registered_xanes2D',
        'dtype': 'np.float32'
    },
    'eng_list': {
        'description':
        'X-ray energy points at which 2D XANES images are taken',
        'path': '/registration_results/reg_results/eng_list',
        'dtype': 'np.float32'
    },
    'centroid_of_eng': {
        'description':
        'centroid of energy; can be used for making image masks to select just sample regions',
        'path': '/processed_XANES2D/proc_spectrum/centroid_of_eng',
        'dtype': 'np.float32'
    },
    'centroid_of_eng_relative_to_wl': {
        'description':
        'centroid of energy in a different way; can be used for making image masks to select just sample regions',
        'path':
        '/processed_XANES2D/proc_spectrum/centroid_of_eng_relative_to_wl',
        'dtype': 'np.float32'
    },
    'weighted_attenuation': {
        'description':
        'average attenuation over all energy points; it is the best for making image masks to select just sample regions by setting a global threshold',
        'path': '/processed_XANES2D/proc_spectrum/weighted_attenuation',
        'dtype': 'np.float32'
    },
    'weighted_eng': {
        'description':
        'averaged energy over all energy points; can be used for making image masks to select just sample regions',
        'path': '/processed_XANES2D/proc_spectrum/weighted_eng',
        'dtype': 'np.float32'
    },
    'whiteline_fit_err': {
        'description': 'whiteline fitting error',
        'path': '/processed_XANES2D/proc_spectrum/whiteline_fit_err',
        'dtype': 'np.float32'
    },
    'wl_fit_err': {
        'description': 'whiteline fitting error',
        'path': '/processed_XANES2D/proc_spectrum/wl_fit_err',
        'dtype': 'np.float32'
    },
    'whiteline_pos_fit': {
        'description': 'fitted whiteline positions',
        'path': '/processed_XANES2D/proc_spectrum/whiteline_pos_fit',
        'dtype': 'np.float32'
    },
    'wl_pos_fit': {
        'description': 'fitted whiteline positions',
        'path': '/processed_XANES2D/proc_spectrum/wl_pos_fit',
        'dtype': 'np.float32'
    },
    'wl_fit_coef': {
        'description': 'coefficients obtained from whiteline fitting',
        'path': '/processed_XANES2D/proc_spectrum/wl_fit_coef',
        'dtype': 'np.float32'
    },
    'wl_pos_dir': {
        'description':
        'whiteline positions measured directly from the experimental data; its precision may be strongly affected by the measurement quality and sampling rate of the energy points',
        'path': '/processed_XANES2D/proc_spectrum/wl_pos_dir',
        'dtype': 'np.float32'
    },
    'whiteline_pos_direct': {
        'description':
        'whiteline positions measured directly from the experimental data; its precision may be strongly affected by the measurement quality and sampling rate of the energy points',
        'path': '/processed_XANES2D/proc_spectrum/whiteline_pos_direct',
        'dtype': 'np.float32'
    },
    'norm_spec': {
        'description':
        'normalized spectra; it takes standard XANES spectrum normalization procedure, subtracting the background before the pre-edge then normalized by the fitted straight line in the post-edge range',
        'path': '/processed_XANES2D/proc_spectrum/norm_spec',
        'dtype': 'np.float32'
    },
    'normalized_spectrum': {
        'description':
        'normalized spectra; it takes standard XANES spectrum normalization procedure, subtracting the background before the pre-edge then normalized by the fitted straight line in the post-edge range',
        'path': '/processed_XANES2D/proc_spectrum/normalized_spectrum',
        'dtype': 'np.float32'
    },
    'wl_peak_height_dir': {
        'description':
        'the whiteline peak height measured directly at the energy sampling point where the x-ray attenuation is strongest; the result may be strongly affected by the measurement quality so prone to the noises',
        'path': '/processed_XANES2D/proc_spectrum/wl_peak_height_dir',
        'dtype': 'np.float32'
    },
    'whiteline_peak_height_direct': {
        'description':
        'the whiteline peak height measured directly at the energy sampling point where the x-ray attenuation is strongest; the result may be strongly affected by the measurement quality so prone to the noises',
        'path':
        '/processed_XANES2D/proc_spectrum/whiteline_peak_height_direct',
        'dtype': 'np.float32'
    },
    'edge50_pos_fit': {
        'description':
        'the energy position where the X-ray attenuation is 50% of that at the whiteline peak position; the calculation is based on the fitted whiteline peak and fitted X-ray absorption edge',
        'path': '/processed_XANES2D/proc_spectrum/edge50_pos_fit',
        'dtype': 'np.float32'
    },
    'edge0.5_pos_fit': {
        'description':
        'the energy position where the X-ray attenuation is 50% of that at the whiteline peak position; the calculation is based on the fitted whiteline peak and fitted X-ray absorption edge',
        'path': '/processed_XANES2D/proc_spectrum/edge0.5_pos_fit',
        'dtype': 'np.float32'
    },
    'edge50_pos_dir': {
        'description':
        'the energy position where the X-ray attenuation is 50% of that at the whiteline peak position; the calculation is based on the direct measurements of the whiteline and X-ray absorption edge; it may be strongly affected the noises',
        'path': '/processed_XANES2D/proc_spectrum/edge50_pos_dir',
        'dtype': 'np.float32'
    },
    'edge0.5_pos_dir': {
        'description':
        'the energy position where the X-ray attenuation is 50% of that at the whiteline peak position; the calculation is based on the direct measurements of the whiteline and X-ray absorption edge; it may be strongly affected the noises',
        'path': '/processed_XANES2D/proc_spectrum/edge0.5_pos_dir',
        'dtype': 'np.float32'
    },
    'edge0.5_pos_direct': {
        'description':
        'the energy position where the X-ray attenuation is 50% of that at the whiteline peak position; the calculation is based on the direct measurements of the whiteline and X-ray absorption edge; it may be strongly affected the noises',
        'path': '/processed_XANES2D/proc_spectrum/edge0.5_pos_direct',
        'dtype': 'np.float32'
    },
    'edge_pos_fit': {
        'description':
        'the x-ray absorption edge position defined as the maximum in the derivative of x-ray absorption edge; the calculation is based on the fitted X-ray absorption edge',
        'path': '/processed_XANES2D/proc_spectrum/edge_pos_fit',
        'dtype': 'np.float32'
    },
    'edge_fit_err': {
        'description': 'the fitting error in fitted x-ray absorption edge map',
        'path': '/processed_XANES2D/proc_spectrum/edge_fit_err',
        'dtype': 'np.float32'
    },
    'edge_fit_coef': {
        'description':
        'the fitting coefficients of the fitted x-ray absorption edge map',
        'path': '/processed_XANES2D/proc_spectrum/edge_fit_coef',
        'dtype': 'np.float32'
    },
    'edge_pos_dir': {
        'description':
        'the edge postions at each pixel calculated directly based on measured x-ray absorption spectra',
        'path': '/processed_XANES2D/proc_spectrum/edge_pos_dir',
        'dtype': 'np.float32'
    },
    'edge_pos_direct': {
        'description':
        'the edge postions at each pixel calculated directly based on measured x-ray absorption spectra',
        'path': '/processed_XANES2D/proc_spectrum/edge_pos_dir',
        'dtype': 'np.float32'
    },
    'edge_jump_filter': {
        'description':
        'a map over all pixels that measures the jump from the background section to the post-edge and normalized to the standard deviation between the measured values in the background section; it can be used as mask filter to select only sample regions',
        'path': '/processed_XANES2D/proc_spectrum/edge_jump_filter',
        'dtype': 'np.float32'
    },
    'edge_offset_filter': {
        'description':
        'it measures how parallel between the background section and the post-edge section in a spectrum; if they cross each other in the measurement energy range, it may indicate the measurement quality is bad; this filter provides such measurement at each pixel; it can be used as mask filter to select only sample regions',
        'path': '/processed_XANES2D/proc_spectrum/edge_offset_filter',
        'dtype': 'np.float32'
    },
    'pre_edge_sd': {
        'description':
        'standard deviation map over all pixels in the background region',
        'path': '/processed_XANES2D/proc_spectrum/pre_edge_sd',
        'dtype': 'np.float32'
    },
    'pre_edge_mean': {
        'description': 'the mean map over all pixels in the background region',
        'path': '/processed_XANES2D/proc_spectrum/pre_edge_mean',
        'dtype': 'np.float32'
    },
    'post_edge_sd': {
        'description':
        'standard deviation map over all pixels in the post-edge region',
        'path': '/processed_XANES2D/proc_spectrum/post_edge_sd',
        'dtype': 'np.float32'
    },
    'post_edge_mean': {
        'description': 'the mean map over all pixels in the post-edge region',
        'path': '/processed_XANES2D/proc_spectrum/post_edge_mean',
        'dtype': 'np.float32'
    },
    'pre_edge_fit_coef': {
        'description':
        'the fitting coefficients of the fitted background section',
        'path': '/processed_XANES2D/proc_spectrum/pre_edge_fit_coef',
        'dtype': 'np.float32'
    },
    'post_edge_fit_coef': {
        'description':
        'the fitting coefficients of the fitted post-edge section',
        'path': '/processed_XANES2D/proc_spectrum/post_edge_fit_coef',
        'dtype': 'np.float32'
    },
    'lcf_fit': {
        'description':
        'the fitting coefficients of the linear combination fitting of the spectra at all pixels',
        'path': '/processed_XANES2D/proc_spectrum/lcf_fit',
        'dtype': 'np.float32'
    },
    'lcf_fit_err': {
        'description':
        'the fitting errors of the linear combination fitting of the spectra at all pixels',
        'path': '/processed_XANES2D/proc_spectrum/lcf_fit_err',
        'dtype': 'np.float32'
    }
}

XANES3D_ANA_ITEM_DICT = {
    'mask': {
        'description': 'a mask for isolating sample area from the background',
        'path': '/processed_XANES3D/gen_masks/{0}/{0}',
        'dtype': 'np.int8'
    },
    'registered_xanes3D': {
        'description': 'aligned 3D spectra image; it is a 4D data array',
        'path': '/registration_results/reg_results/registered_xanes3D',
        'dtype': 'np.float32'
    },
    'eng_list': {
        'description':
        'X-ray energy points at which tomography scans are taken',
        'path': '/registration_results/reg_results/eng_list',
        'dtype': 'np.float32'
    },
    'centroid_of_eng': {
        'description':
        'centroid of energy; can be used for making image masks to select just sample regions',
        'path': '/processed_XANES3D/proc_spectrum/centroid_of_eng',
        'dtype': 'np.float32'
    },
    'centroid_of_eng_relative_to_wl': {
        'description':
        'centroid of energy in a different way; can be used for making image masks to select just sample regions',
        'path':
        '/processed_XANES3D/proc_spectrum/centroid_of_eng_relative_to_wl',
        'dtype': 'np.float32'
    },
    'weighted_attenuation': {
        'description':
        'average attenuation over all energy points; it is the best for making image masks to select just sample regions by setting a global threshold',
        'path': '/processed_XANES3D/proc_spectrum/weighted_attenuation',
        'dtype': 'np.float32'
    },
    'weighted_eng': {
        'description':
        'averaged energy over all energy points; can be used for making image masks to select just sample regions',
        'path': '/processed_XANES3D/proc_spectrum/weighted_eng',
        'dtype': 'np.float32'
    },
    'whiteline_fit_err': {
        'description': 'whiteline fitting error',
        'path': '/processed_XANES3D/proc_spectrum/whiteline_fit_err',
        'dtype': 'np.float32'
    },
    'wl_fit_err': {
        'description': 'whiteline fitting error',
        'path': '/processed_XANES3D/proc_spectrum/wl_fit_err',
        'dtype': 'np.float32'
    },
    'whiteline_pos_fit': {
        'description': 'fitted whiteline positions',
        'path': '/processed_XANES3D/proc_spectrum/whiteline_pos_fit',
        'dtype': 'np.float32'
    },
    'wl_pos_fit': {
        'description': 'fitted whiteline positions',
        'path': '/processed_XANES3D/proc_spectrum/wl_pos_fit',
        'dtype': 'np.float32'
    },
    'wl_fit_coef': {
        'description': 'coefficients obtained from whiteline fitting',
        'path': '/processed_XANES3D/proc_spectrum/wl_fit_coef',
        'dtype': 'np.float32'
    },
    'wl_pos_dir': {
        'description':
        'whiteline positions measured directly from the experimental data; its precision may be strongly affected by the measurement quality and sampling rate of the energy points',
        'path': '/processed_XANES3D/proc_spectrum/wl_pos_dir',
        'dtype': 'np.float32'
    },
    'whiteline_pos_direct': {
        'description':
        'whiteline positions measured directly from the experimental data; its precision may be strongly affected by the measurement quality and sampling rate of the energy points',
        'path': '/processed_XANES3D/proc_spectrum/whiteline_pos_direct',
        'dtype': 'np.float32'
    },
    'norm_spec': {
        'description':
        'normalized spectra; it takes standard XANES spectrum normalization procedure, subtracting the background before the pre-edge then normalized by the fitted straight line in the post-edge range',
        'path': '/processed_XANES3D/proc_spectrum/norm_spec',
        'dtype': 'np.float32'
    },
    'normalized_spectrum': {
        'description':
        'normalized spectra; it takes standard XANES spectrum normalization procedure, subtracting the background before the pre-edge then normalized by the fitted straight line in the post-edge range',
        'path': '/processed_XANES3D/proc_spectrum/normalized_spectrum',
        'dtype': 'np.float32'
    },
    'wl_peak_height_dir': {
        'description':
        'the whiteline peak height measured directly at the energy sampling point where the x-ray attenuation is strongest; the result may be strongly affected by the measurement quality so prone to the noises',
        'path': '/processed_XANES3D/proc_spectrum/wl_peak_height_dir',
        'dtype': 'np.float32'
    },
    'whiteline_peak_height_direct': {
        'description':
        'the whiteline peak height measured directly at the energy sampling point where the x-ray attenuation is strongest; the result may be strongly affected by the measurement quality so prone to the noises',
        'path':
        '/processed_XANES3D/proc_spectrum/whiteline_peak_height_direct',
        'dtype': 'np.float32'
    },
    'edge50_pos_fit': {
        'description':
        'the energy position where the X-ray attenuation is 50% of that at the whiteline peak position; the calculation is based on the fitted whiteline peak and fitted X-ray absorption edge',
        'path': '/processed_XANES3D/proc_spectrum/edge50_pos_fit',
        'dtype': 'np.float32'
    },
    'edge0.5_pos_fit': {
        'description':
        'the energy position where the X-ray attenuation is 50% of that at the whiteline peak position; the calculation is based on the fitted whiteline peak and fitted X-ray absorption edge',
        'path': '/processed_XANES3D/proc_spectrum/edge0.5_pos_fit',
        'dtype': 'np.float32'
    },
    'edge50_pos_dir': {
        'description':
        'the energy position where the X-ray attenuation is 50% of that at the whiteline peak position; the calculation is based on the direct measurements of the whiteline and X-ray absorption edge; it may be strongly affected the noises',
        'path': '/processed_XANES3D/proc_spectrum/edge50_pos_dir',
        'dtype': 'np.float32'
    },
    'edge0.5_pos_dir': {
        'description':
        'the energy position where the X-ray attenuation is 50% of that at the whiteline peak position; the calculation is based on the direct measurements of the whiteline and X-ray absorption edge; it may be strongly affected the noises',
        'path': '/processed_XANES3D/proc_spectrum/edge0.5_pos_dir',
        'dtype': 'np.float32'
    },
    'edge0.5_pos_direct': {
        'description':
        'the energy position where the X-ray attenuation is 50% of that at the whiteline peak position; the calculation is based on the direct measurements of the whiteline and X-ray absorption edge; it may be strongly affected the noises',
        'path': '/processed_XANES3D/proc_spectrum/edge0.5_pos_direct',
        'dtype': 'np.float32'
    },
    'edge_pos_fit': {
        'description':
        'the x-ray absorption edge position defined as the maximum in the derivative of x-ray absorption edge; the calculation is based on the fitted X-ray absorption edge',
        'path': '/processed_XANES3D/proc_spectrum/edge_pos_fit',
        'dtype': 'np.float32'
    },
    'edge_fit_err': {
        'description': 'the fitting error in fitted x-ray absorption edge map',
        'path': '/processed_XANES3D/proc_spectrum/edge_fit_err',
        'dtype': 'np.float32'
    },
    'edge_fit_coef': {
        'description':
        'the fitting coefficients of the fitted x-ray absorption edge map',
        'path': '/processed_XANES3D/proc_spectrum/edge_fit_coef',
        'dtype': 'np.float32'
    },
    'edge_pos_dir': {
        'description':
        'the edge postions at each pixel calculated directly based on measured x-ray absorption spectra',
        'path': '/processed_XANES3D/proc_spectrum/edge_pos_dir',
        'dtype': 'np.float32'
    },
    'edge_pos_direct': {
        'description':
        'the edge postions at each pixel calculated directly based on measured x-ray absorption spectra',
        'path': '/processed_XANES3D/proc_spectrum/edge_pos_direct',
        'dtype': 'np.float32'
    },
    'edge_jump_filter': {
        'description':
        'a map over all pixels that measures the jump from the background section to the post-edge and normalized to the standard deviation between the measured values in the background section; it can be used as mask filter to select only sample regions',
        'path': '/processed_XANES3D/proc_spectrum/edge_jump_filter',
        'dtype': 'np.float32'
    },
    'edge_offset_filter': {
        'description':
        'it measures how parallel between the background section and the post-edge section in a spectrum; if they cross each other in the measurement energy range, it may indicate the measurement quality is bad; this filter provides such measurement at each pixel; it can be used as mask filter to select only sample regions',
        'path': '/processed_XANES3D/proc_spectrum/edge_offset_filter',
        'dtype': 'np.float32'
    },
    'pre_edge_sd': {
        'description':
        'standard deviation map over all pixels in the background region',
        'path': '/processed_XANES3D/proc_spectrum/pre_edge_sd',
        'dtype': 'np.float32'
    },
    'pre_edge_mean': {
        'description': 'the mean map over all pixels in the background region',
        'path': '/processed_XANES3D/proc_spectrum/pre_edge_mean',
        'dtype': 'np.float32'
    },
    'post_edge_sd': {
        'description':
        'standard deviation map over all pixels in the post-edge region',
        'path': '/processed_XANES3D/proc_spectrum/post_edge_sd',
        'dtype': 'np.float32'
    },
    'post_edge_mean': {
        'description': 'the mean map over all pixels in the post-edge region',
        'path': '/processed_XANES3D/proc_spectrum/post_edge_mean',
        'dtype': 'np.float32'
    },
    'pre_edge_fit_coef': {
        'description':
        'the fitting coefficients of the fitted background section',
        'path': '/processed_XANES3D/proc_spectrum/pre_edge_fit_coef',
        'dtype': 'np.float32'
    },
    'post_edge_fit_coef': {
        'description':
        'the fitting coefficients of the fitted post-edge section',
        'path': '/processed_XANES3D/proc_spectrum/post_edge_fit_coef',
        'dtype': 'np.float32'
    },
    'lcf_fit': {
        'description':
        'the fitting coefficients of the linear combination fitting of the spectra at all pixels',
        'path': '/processed_XANES3D/proc_spectrum/lcf_fit',
        'dtype': 'np.float32'
    },
    'lcf_fit_err': {
        'description':
        'the fitting errors of the linear combination fitting of the spectra at all pixels',
        'path': '/processed_XANES3D/proc_spectrum/lcf_fit_err',
        'dtype': 'np.float32'
    }
}
