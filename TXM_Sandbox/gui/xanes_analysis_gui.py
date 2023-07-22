#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 22:58:16 2020

@author: xiao
"""

from ipywidgets import widgets
from IPython.display import Javascript, display
from copy import deepcopy
import os, h5py, json
from silx.io.dictdump import dicttoh5
import numpy as np
from dask import delayed
import dask.array as da
import napari

from .gui_components import (SelectFilesButton, NumpyArrayEncoder, get_handles,
                             enable_disable_boxes, gen_external_py_script,
                             fiji_viewer_off, fiji_viewer_on,
                             update_json_content)
from ..dicts.xanes_analysis_dict import XANES_ANA_METHOD
from ..utils.io import h5_lazy_reader
from ..utils import xanes_post_analysis as xpa


class xanes_analysis_gui():
    MASK_OP_PAR_DICT = {
        'Threshold': {
            'description':
            'Apply a threshold on the selected image to convert it into a binary image',
            'url': 'https://www.google.com',
            'flt': xpa.threshold,
            'default_pars': {
                'AnaSnglMaskOpPar08 txt': {
                    'var': 'lower',
                    'val': 0,
                    'tooltip': 'lower threshold'
                },
                'AnaSnglMaskOpPar09 txt': {
                    'var': 'upper',
                    'val': 1e30,
                    'tooltip': 'upper threshold'
                }
            }
        },
        'Gaussian': {
            'description':
            'Apply a Gaussian filter on the selected image to smooth the image',
            'url':
            'https://scikit-image.org/docs/0.19.x/api/skimage.filters.html#skimage.filters.gaussian',
            'flt': xpa.gaussian,
            'default_pars': {
                'AnaSnglMaskOpPar08 txt': {
                    'var': 'sigma',
                    'val': 3,
                    'tooltip': 'Gausssian blurring filter kernal size'
                }
            }
        },
        'Median': {
            'description':
            'Apply a median filter on the selected image to smooth the image',
            'url':
            'https://scikit-image.org/docs/0.19.x/api/skimage.filters.html#skimage.filters.median',
            'flt': xpa.median,
            'default_pars': {
                'AnaSnglMaskOpPar08 txt': {
                    'var': 'footprint',
                    'val': 3,
                    'tooltip': 'median filter kernal size'
                }
            }
        },
        'Dilation': {
            'description':
            'Return grayscale morphological dilation of the selected image',
            'url':
            'https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.dilation',
            'flt': xpa.dilation,
            'default_pars': {
                'AnaSnglMaskOpPar08 txt': {
                    'var': 'footprint',
                    'val': 2,
                    'tooltip':
                    'footprint: int, radius of the dilation disk kernal'
                },
                'AnaSnglMaskOpPar09 txt': {
                    'var': 'iter',
                    'val': 1,
                    'tooltip': 'dilation iterations'
                }
            }
        },
        'Erosion': {
            'description':
            'Return grayscale morphological erosion of the selected image',
            'url':
            'https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.erosion',
            'flt': xpa.erosion,
            'default_pars': {
                'AnaSnglMaskOpPar08 txt': {
                    'var': 'footprint',
                    'val': 2,
                    'tooltip':
                    'footprint: int, radius of the erosion disk kernal'
                },
                'AnaSnglMaskOpPar09 txt': {
                    'var': 'iter',
                    'val': 1,
                    'tooltip': 'erosion iterations'
                }
            }
        },
        'Opening': {
            'description':
            'Return grayscale morphological opening of an image',
            'url':
            'https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.opening',
            'flt': xpa.opening,
            'default_pars': {
                'AnaSnglMaskOpPar08 txt': {
                    'var': 'footprint',
                    'val': 2,
                    'tooltip':
                    'footprint: int, radius of the opening disk kernal'
                }
            }
        },
        'Closing': {
            'description':
            'Return grayscale morphological closing of an image',
            'url':
            'https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.closing',
            'flt': xpa.closing,
            'default_pars': {
                'AnaSnglMaskOpPar08 txt': {
                    'var': 'footprint',
                    'val': 2,
                    'tooltip':
                    'footprint: int, radius of the closing disk kernal'
                }
            }
        },
        'Area Opening': {
            'description':
            'Perform an area opening of a grayscale image; removes all dark structures of an image with a surface smaller than area_threshold',
            'url':
            'https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.area_opening',
            'flt': xpa.area_opening,
            'default_pars': {
                'AnaSnglMaskOpPar08 txt': {
                    'var': 'area_threshold',
                    'val': 64,
                    'tooltip': 'The size parameter (number of pixels)'
                }
            }
        },
        'Area Closing': {
            'description':
            'Perform an area closing of a grayscale image; removes all dark structures of an image with a surface smaller than area_threshold',
            'url':
            'https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.area_closing',
            'flt': xpa.area_closing,
            'default_pars': {
                'AnaSnglMaskOpPar08 txt': {
                    'var': 'area_threshold',
                    'val': 64,
                    'tooltip': 'The size parameter (number of pixels)'
                }
            }
        },
        'Diameter Opening': {
            'description':
            'Removes all bright structures of an image with maximal extension smaller than diameter_threshold',
            'url':
            'https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.diameter_opening',
            'flt': xpa.diameter_opening,
            'default_pars': {
                'AnaSnglMaskOpPar08 txt': {
                    'var':
                    'diameter_threshold',
                    'val':
                    8,
                    'tooltip':
                    'diameter_threshold: unsigned int; The maximal extension parameter (number of pixels)'
                }
            }
        },
        'Diameter Closing': {
            'description':
            'Removes all dark structures of an image with maximal extension smaller than diameter_threshold',
            'url':
            'https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.diameter_closing',
            'flt': xpa.diameter_closing,
            'default_pars': {
                'AnaSnglMaskOpPar08 txt': {
                    'var':
                    'diameter_threshold',
                    'val':
                    8,
                    'tooltip':
                    'diameter_threshold: unsigned int; The maximal extension parameter (number of pixels)'
                }
            }
        },
        'Bin Dilation': {
            'description':
            'Returns the same result as grayscale dilation but performs faster for binary images',
            'url':
            'https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_dilation',
            'flt': xpa.binary_dilation,
            'default_pars': {
                'AnaSnglMaskOpPar08 txt': {
                    'var': 'footprint',
                    'val': 2,
                    'tooltip':
                    'footprint: int, radius of the dilation disk kernal'
                },
                'AnaSnglMaskOpPar09 txt': {
                    'var': 'iter',
                    'val': 1,
                    'tooltip': 'dilation iterations'
                }
            }
        },
        'Bin Erosion': {
            'description':
            'Returns the same result as grayscale erosion but performs faster for binary images',
            'url':
            'https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_erosion',
            'flt': xpa.binary_erosion,
            'default_pars': {
                'AnaSnglMaskOpPar08 txt': {
                    'var': 'footprint',
                    'val': 2,
                    'tooltip':
                    'footprint: int, radius of the erosion disk kernal'
                },
                'AnaSnglMaskOpPar09 txt': {
                    'var': 'iter',
                    'val': 1,
                    'tooltip': 'dilation iterations'
                }
            }
        },
        'Bin Opening': {
            'description':
            'returns the same result as grayscale opening but performs faster for binary images',
            'url':
            'https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_opening',
            'flt': xpa.binary_opening,
            'default_pars': {
                'AnaSnglMaskOpPar08 txt': {
                    'var': 'footprint',
                    'val': 2,
                    'tooltip':
                    'footprint: int, radius of the opening disk kernal'
                }
            }
        },
        'Bin Closing': {
            'description':
            'returns the same result as grayscale closing but performs faster for binary images',
            'url':
            'https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_closing',
            'flt': xpa.binary_closing,
            'default_pars': {
                'AnaSnglMaskOpPar08 txt': {
                    'var': 'footprint',
                    'val': 2,
                    'tooltip':
                    'footprint: int, radius of the closing disk kernal'
                }
            }
        },
        'Bin Rm Small Holes': {
            'description':
            'Remove contiguous holes (smaller values) smaller than the specified size',
            'url':
            'https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.remove_small_holes',
            'flt': xpa.remove_small_holes,
            'default_pars': {
                'AnaSnglMaskOpPar08 txt': {
                    'var':
                    'area_threshold',
                    'val':
                    64,
                    'tooltip':
                    'area_threshold: int, The maximum area, in pixels, of a contiguous hole that will be filled'
                }
            }
        },
        'Bin Rm Small Objs': {
            'description':
            'Remove contiguous objects (higher values) smaller than the specified size',
            'url':
            'https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.remove_small_objects',
            'flt': xpa.remove_small_objects,
            'default_pars': {
                'AnaSnglMaskOpPar08 txt': {
                    'var':
                    'min_size',
                    'val':
                    64,
                    'tooltip':
                    'min_size: int, The maximum area, in pixels, of a contiguous hole that will be filled'
                }
            }
        }
    }

    SET_OP_DICT = {
        '': {
            'description': 'identity operation',
            'op_symb': '',
            'flt': xpa.set_identity
        },
        'Union (+)': {
            'description': 'Union operation between two masks',
            'op_symb': ' + ',
            'flt': xpa.set_union
        },
        'Intersection (x)': {
            'description': 'Intersection between two masks',
            'op_symb': ' x ',
            'flt': xpa.set_intersection
        },
        'Differnece (-)': {
            'description': 'Difference between two masks',
            'op_symb': ' - ',
            'flt': xpa.set_difference
        },
        'Complement (I-A)': {
            'description': 'Complement of a mask',
            'op_symb': '1 - A',
            'flt': xpa.set_complement
        }
    }

    """ 
    operation sequence for making a single mask
    self.ana_mask_sngl_op = {
        key: int <operation order> {
            key: str <flt_name: filter's name>,
            key: dictionary <filter function parameters>
        },
    }

    singled mask collection dictionary structure:
    self.ana_mask_sngl = {
        key: <mask name> {
            key: str <var_name: variable name in string>,
            key: dictionary <op: operation sequence> {
                key: int <operation order> {
                    key: str <flt_name: filter's name>,
                    key: dictionary <filter function parameters>
                }
            }
        },
    }

    operation sequence for making a combined mask
    self.ana_mask_comb_op = {
        key: int <operation order> {
            key: str <op: operation name>,
            key: str <op_symb: operation symbol>,
            key: str <B name: name of the second argument in the binary function>,
            key: str <B short name: short name of the second argument in the binary function>
        }, 
    
    }

    self.ana_mask_comb = {
        key: str <mask name>: {
            key: dictionary <op: operation sequence>: {
                key: int <operation order>: {
                    key: str <op: operation name>,
                    key: str <op_symb: operation symbol>,
                    key: str <B name: name of the second argument in the binary function>,
                    key: str <B short name: short name of the second argument in the binary function>
                } 
            }
        },
    }

    self.ana_mask = {
        key: delayed dask.array <mask: mask>,
        key: dict <source: inherit from single and combined mask operations>
    }

    self.ana_data = {
        key: delayed dask.array <data>,
    }
    """

    def __init__(self, parent_h, form_sz=[650, 740]):
        self.parent_h = parent_h
        self.hs = {}
        self.form_sz = form_sz

        self.ana_fn_seled = False
        self.ana_data_existed = False
        self.ana_data_cnfged = False
        self.ana_mask_sngl_cnfged = False
        self.ana_mask_cnfged = False
        self.ana_ana_data_cnfged = False
        self.ana_proc_done = False
        self.ana_mask_sngl_op_seled = False
        self.ana_mask_comb_op_seled = False

        self.ana_fn = ""
        self.ana_proc_spec_path_in_h5 = None
        self.ana_proc_data_list = [""]

        self.ana_data = {}
        self.ana_spec_list = []
        self.viewer_type = 'napari'
        self.viewer = None

        self.ana_data = {}
        self.ana_spec = {}
        self.ana_mask = {}
        self.ana_mask_sngl_op = {}
        self.ana_mask_sngl = {}
        self.ana_mask_comb_op = {}
        self.ana_mask_comb = {}
        self.ana_ana_spec = None
        self.ana_ana_mask = None

    def lock_message_text_boxes(self):
        boxes = [
            'AnaDataFn txt', 'AnaDatasetInfo txt', 'AnaVarsEq sel',
            'AnaVarName sel', 'AnaCombMaskVarInfo txt'
        ]
        enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)

    def build_gui(self):
        self.hs['Ana form'] = widgets.Tab()
        self.hs['AnaData tab'] = widgets.VBox()
        self.hs['AnaMask tab'] = widgets.VBox()
        self.hs['AnaConf tab'] = widgets.VBox()
        self.hs['AnaDisp tab'] = widgets.VBox()
        layout = {
            'border': '3px solid #FFCC00',
            'width': '100%',
            'height': '100%'
        }
        self.hs['AnaData tab'].layout = layout
        self.hs['AnaMask tab'].layout = layout
        self.hs['AnaConf tab'].layout = layout
        self.hs['AnaDisp tab'].layout = layout
        self.hs['Ana form'].children = [
            self.hs['AnaData tab'], self.hs['AnaMask tab'],
            self.hs['AnaConf tab'], self.hs['AnaDisp tab']
        ]
        self.hs['Ana form'].set_title(0, 'Data Conf')
        self.hs['Ana form'].set_title(1, 'Mask Make')
        self.hs['Ana form'].set_title(2, 'Ana Conf')
        self.hs['Ana form'].set_title(3, 'Ana Disp')

        ## ## ## ## ## data setup -- start
        self.hs['AnaDataSetup box'] = widgets.VBox(layout={
            'width': '100%',
            'height': '100%'
        })

        ## ## ## ## ## ## config parameters -- start
        data_setup_GridSpecLayout = widgets.GridspecLayout(17,
                                                           100,
                                                           layout={
                                                               "width": '100%',
                                                               "height": '100%'
                                                           })

        data_setup_GridSpecLayout[0, :80] = widgets.Text(
            value='choose your data file ...',
            layout={
                'width': '100%',
                'height': '100%'
            },
            disabled=True)
        self.hs['AnaDataFn txt'] = data_setup_GridSpecLayout[0, :80]

        data_setup_GridSpecLayout[0, 82:98] = SelectFilesButton(
            option='askopenfilename',
            layout={
                'width': '100%',
                'height': '100%'
            },
            open_filetypes=(('hdf5 files', ['*.hdf', '*.h5']), ))
        self.hs['AnaDataFn btn'] = data_setup_GridSpecLayout[0, 82:98]
        if (self.parent_h.global_h.cwd is None) or (not os.path.exists(
                os.path.dirname(self.parent_h.global_h.cwd))):
            self.hs[
                'AnaDataFn btn'].initialdir = self.parent_h.global_h.script_dir
        else:
            self.hs['AnaDataFn btn'].initialdir = self.parent_h.global_h.cwd

        data_setup_GridSpecLayout[2:11, :29] = widgets.Select(
            options=[],
            value=None,
            description='Avai Data:',
            disabled=False,
            layout={
                'width': '100%',
                'height': '100%'
            },
            tooltip='Available datasets in the chosen file')
        self.hs['AnaAvaiDatasets sel'] = data_setup_GridSpecLayout[2:11, :29]

        data_setup_GridSpecLayout[5, 30:37] = widgets.Button(
            description='=>',
            tooltip='Add Dataset',
            layout={
                'width': '90%',
                'height': '100%'
            },
            disabled=True)
        self.hs['AnaSelDatasetAdd btn'] = data_setup_GridSpecLayout[5, 30:37]
        self.hs['AnaSelDatasetAdd btn'].style.button_color = 'darkviolet'

        data_setup_GridSpecLayout[7, 30:37] = widgets.Button(
            description='<=',
            tooltip='Remove Dataset',
            layout={
                'width': '90%',
                'height': '100%'
            },
            disabled=True)
        self.hs['AnaSelDatasetRm btn'] = data_setup_GridSpecLayout[7, 30:37]
        self.hs['AnaSelDatasetRm btn'].style.button_color = 'darkviolet'

        data_setup_GridSpecLayout[2:11, 37:66] = widgets.Select(
            options=[],
            value=None,
            description='Sel Vars:',
            disabled=True,
            layout={
                'width': '100%',
                'height': '100%'
            },
            tooltip='Selected Varibles')
        self.hs['AnaSelVars sel'] = data_setup_GridSpecLayout[2:11, 37:66]

        data_setup_GridSpecLayout[2:11, 69:72] = widgets.Select(options=[],
                                                                value=None,
                                                                disabled=True,
                                                                layout={
                                                                    'width':
                                                                    '100%',
                                                                    'height':
                                                                    '100%'
                                                                })
        self.hs['AnaVarsEq sel'] = data_setup_GridSpecLayout[2:11, 69:72]

        data_setup_GridSpecLayout[2:11, 73:99] = widgets.Select(
            options=[],
            value=None,
            description='Sav Vars:',
            disabled=True,
            layout={
                'width': '100%',
                'height': '100%'
            },
            tooltip='Saved Varibles')
        self.hs['AnaVarName sel'] = data_setup_GridSpecLayout[2:11, 73:99]

        data_setup_GridSpecLayout[12:16, :37] = widgets.Textarea(
            value='shape: \nmin: \nmax: \nmean: ',
            description='Data Info',
            disabled=True,
            layout={
                'width': '100%',
                'height': '100%'
            })
        self.hs['AnaDatasetInfo txt'] = data_setup_GridSpecLayout[12:16, :37]

        data_setup_GridSpecLayout[12:17,
                                  38:80] = widgets.Output(layout={
                                      'border': '1px solid #FFCC11',
                                      'width': '100%',
                                      'height': '90%'
                                  })
        self.hs['AnaDataHist plt'] = data_setup_GridSpecLayout[12:17, 38:80]

        self.hs['AnaDataPrev box'] = widgets.VBox(layout={
            'width': 'auto',
            'height': 'auto'
        },
                                                  disabled=True)

        data_setup_GridSpecLayout[13:15, 82:98] = widgets.RadioButtons(
            options=['Fiji', 'napari'],
            value='Fiji',
            layout={
                'width': 'auto',
                'height': 'auto'
            },
            disabled=True)
        self.hs['AnaSelDatasetPrevOpt rad'] = data_setup_GridSpecLayout[13:15,
                                                                        82:98]

        data_setup_GridSpecLayout[15, 82:98] = widgets.Button(
            description='Preview',
            layout={
                'width': 'auto',
                'height': 'auto'
            },
            disabled=True)
        self.hs['AnaSelDatasetPrev btn'] = data_setup_GridSpecLayout[15, 82:98]
        self.hs['AnaSelDatasetPrev btn'].style.button_color = 'darkviolet'

        self.hs['AnaDataPrev box'].children = [
            self.hs['AnaSelDatasetPrev btn'],
            self.hs['AnaSelDatasetPrevOpt rad']
        ]

        self.hs['AnaDataFn btn'].on_click(self.AnaDataFn_btn_clk)
        self.hs['AnaAvaiDatasets sel'].observe(self.AnaAvaiDatasets_sel_chg,
                                               names='value')
        self.hs['AnaSelDatasetAdd btn'].on_click(self.AnaSelDatasetAdd_btn_clk)
        self.hs['AnaSelDatasetRm btn'].on_click(self.AnaSelDatasetRm_btn_clk)
        self.hs['AnaSelVars sel'].observe(self.AnaSelVars_sel_chg,
                                          names='value')
        self.hs['AnaSelDatasetPrev btn'].on_click(
            self.AnaSelDatasetPrev_btn_clk)
        self.hs['AnaSelDatasetPrevOpt rad'].observe(
            self.AnaSelDatasetPrevOpt_rad_chg, names='value')
        self.hs['AnaDataSetup box'].children = [data_setup_GridSpecLayout]

        self.hs['AnaData tab'].children = [self.hs['AnaDataSetup box']]
        ## ## ## ## ## data setup -- end

        ## ## ## ## ## preparing masks -- start
        self.hs['AnaMask acc'] = widgets.Accordion(
            titles=('Make A Single Mask', 'Combine Masks'),
            layout={
                'width': 'auto',
                'height': 'auto'
            })

        ## ## ## ## ## ## define single mask -- start
        AnaSnglMask_GridspecLayout = widgets.GridspecLayout(
            17,
            100,
            layout={
                'width': '100%',
                'height': f'{0.68 * (self.form_sz[0] - 136)}px'
            })

        AnaSnglMask_GridspecLayout[:, :20] = widgets.VBox(layout={
            'width':
            '100%',
            'height':
            '100%',
            'border':
            '1px solid #FFCC11'
        },
                                                          disabled=True)
        self.hs['AnaSnglMaskVarName box'] = AnaSnglMask_GridspecLayout[:, :20]
        self.hs['AnaSnglMaskVarName lbl'] = widgets.Label('Sel Var')
        self.hs['AnaSnglMaskVarName drpn'] = widgets.Dropdown(
            tooltip='Select a Varible to Make a Mask Based on It.',
            options=self.hs['AnaVarName sel'].options,
            layout={
                'width': '90%',
                'height': 'auto',
                'left': '5%'
            },
            disabled=True)
        self.hs['AnaSnglMaskSpecName lbl'] = widgets.Label('Sel Spec')
        self.hs['AnaSnglMaskSpecName drpn'] = widgets.Dropdown(
            tooltip='Select a Spec on Which the Mask Will Be Applied.',
            options=self.ana_spec_list,
            layout={
                'width': '90%',
                'height': 'auto',
                'left': '5%'
            },
            disabled=True)
        self.hs['AnaSnglMaskVarName box'].children = [
            self.hs['AnaSnglMaskVarName lbl'],
            self.hs['AnaSnglMaskVarName drpn'],
            self.hs['AnaSnglMaskSpecName lbl'],
            self.hs['AnaSnglMaskSpecName drpn']
        ]

        AnaSnglMask_GridspecLayout[:, 21:41] = widgets.VBox(layout={
            'width':
            '100%',
            'height':
            '100%',
            'border':
            '1px solid #FFCC11'
        },
                                                            disabled=True)
        self.hs['AnaSnglMaskOp box'] = AnaSnglMask_GridspecLayout[:, 21:41]
        self.hs['AnaSnglMaskOp lbl'] = widgets.Label('Sel Op')
        self.hs['AnaSnglMaskOp drpn'] = widgets.Dropdown(
            tooltip=xanes_analysis_gui.MASK_OP_PAR_DICT['Threshold']
            ['description'],
            options=(
                'Threshold',
                'Gaussian',
                'Median',
                'Dilation',
                'Erosion',
                'Opening',
                'Closing',
                'Area Opening',
                'Area Closing',
                'Diameter_Opening',
                'Diameter_Closing',
                'Bin Rm Small Holes',
                'Bin Rm Small Objs',
                'Bin Dilation',
                'Bin Erosion',
                'Bin Opening',
                'Bin Closing',
            ),
            layout={
                'width': '90%',
                'height': 'auto',
                'left': '5%'
            },
            disabled=True)

        for ii in range(4):
            self.hs[f'AnaSnglMaskOpPar{str(ii).zfill(2)} box'] = widgets.HBox(
                layout={
                    'width': '100%',
                    'height': f'{0.06 * (self.form_sz[0] - 136)}px'
                })
            children = []
            for jj in range(2):
                self.hs[
                    f'AnaSnglMaskOpPar{str(ii*2+jj).zfill(2)} drpn'] = widgets.Dropdown(
                        layout={
                            'width': '44%',
                            'height': 'auto',
                            'left': '5%'
                        },
                        tooltip=f'p{str(ii*2+jj).zfill(2)}',
                        disabled=True)
                children.append(
                    self.hs[f'AnaSnglMaskOpPar{str(ii*2+jj).zfill(2)} drpn'])
            self.hs[
                f'AnaSnglMaskOpPar{str(ii).zfill(2)} box'].children = children
        for ii in range(4, 8):
            self.hs[f'AnaSnglMaskOpPar{str(ii).zfill(2)} box'] = widgets.HBox(
                layout={
                    'width': '100%',
                    'height': f'{0.06 * (self.form_sz[0] - 136)}px'
                })
            children = []
            for jj in range(2):
                self.hs[
                    f'AnaSnglMaskOpPar{str(ii*2+jj).zfill(2)} txt'] = widgets.FloatText(
                        layout={
                            'width': '44%',
                            'height': '80%',
                            'left': '5%'
                        },
                        tooltip=f'p{str(ii*2+jj).zfill(2)}',
                        disabled=True)
                children.append(
                    self.hs[f'AnaSnglMaskOpPar{str(ii*2+jj).zfill(2)} txt'])
            self.hs[
                f'AnaSnglMaskOpPar{str(ii).zfill(2)} box'].children = children

        self.hs[
            f'AnaSnglMaskOpPar08 txt'].value = xanes_analysis_gui.MASK_OP_PAR_DICT[
                'Threshold']['default_pars']['AnaSnglMaskOpPar08 txt']['val']
        self.hs[
            f'AnaSnglMaskOpPar08 txt'].tooltip = xanes_analysis_gui.MASK_OP_PAR_DICT[
                'Threshold']['default_pars']['AnaSnglMaskOpPar08 txt'][
                    'tooltip']
        self.hs[
            f'AnaSnglMaskOpPar09 txt'].value = xanes_analysis_gui.MASK_OP_PAR_DICT[
                'Threshold']['default_pars']['AnaSnglMaskOpPar09 txt']['val']
        self.hs[
            f'AnaSnglMaskOpPar09 txt'].tooltip = xanes_analysis_gui.MASK_OP_PAR_DICT[
                'Threshold']['default_pars']['AnaSnglMaskOpPar09 txt'][
                    'tooltip']

        self.hs['AnaSnglMaskOpHelp out'] = widgets.Output(layout={
            'width': 'auto',
            'height': '0.1%'
        })
        self.hs['AnaSnglMaskOpHelp btn'] = widgets.Button(
            description='Help',
            tooltip='https://www.google.com',
            layout={
                'width': '60%',
                'height': 'auto',
                'top': '2%',
                'left': '20%'
            },
            disabled=True)
        self.hs['AnaSnglMaskOpHelp btn'].style.button_color = 'darkviolet'

        self.hs['AnaSnglMaskOp box'].children = [
            self.hs['AnaSnglMaskOp lbl'], self.hs['AnaSnglMaskOp drpn'], *[
                self.hs[f'AnaSnglMaskOpPar{str(ii).zfill(2)} box']
                for ii in range(8)
            ], self.hs['AnaSnglMaskOpHelp out'],
            self.hs['AnaSnglMaskOpHelp btn']
        ]

        AnaSnglMask_GridspecLayout[7, 42:49] = widgets.Button(
            description='=>',
            tooltip='Add an Operation',
            layout={
                'width': '90%',
                'height': '100%'
            },
            disabled=True)
        self.hs['AnaSnglMaskOpAdd btn'] = AnaSnglMask_GridspecLayout[7, 42:49]
        self.hs['AnaSnglMaskOpAdd btn'].style.button_color = 'darkviolet'

        AnaSnglMask_GridspecLayout[9, 42:49] = widgets.Button(
            description='<=',
            tooltip='Remove an Operation',
            layout={
                'width': '90%',
                'height': '100%'
            },
            disabled=True)
        self.hs['AnaSnglMaskOpRm btn'] = AnaSnglMask_GridspecLayout[9, 42:49]
        self.hs['AnaSnglMaskOpRm btn'].style.button_color = 'darkviolet'

        AnaSnglMask_GridspecLayout[:, 50:70] = widgets.VBox(layout={
            'width':
            '100%',
            'height':
            '100%',
            'border':
            '1px solid #FFCC11'
        },
                                                            disabled=True)
        self.hs['AnaSnglMaskOpSeq box'] = AnaSnglMask_GridspecLayout[:, 50:70]
        self.hs['AnaSnglMaskOpSeq lbl'] = widgets.Label('Op Seq')
        self.hs['AnaSnglMaskOpSeq sel'] = widgets.Select(layout={
            'width': '90%',
            'height': '70%',
            'left': '5%'
        },
                                                         disabled=True)
        self.hs['AnaSnglMaskOpSeqZSli txt'] = widgets.BoundedIntText(
            tooltip='Z Slice #',
            min=0,
            max=1,
            step=1,
            layout={
                'width': '60%',
                'height': 'auto',
                'left': '20%'
            },
            disabled=True)
        self.hs['AnaSnglMaskOpSeqPrev btn'] = widgets.Button(
            description='Preview',
            layout={
                'width': '60%',
                'height': 'auto',
                'left': '20%'
            },
            disabled=True)
        self.hs['AnaSnglMaskOpSeqPrev btn'].style.button_color = 'darkviolet'
        self.hs['AnaSnglMaskOpSeq box'].children = [
            self.hs['AnaSnglMaskOpSeq lbl'], self.hs['AnaSnglMaskOpSeq sel'],
            self.hs['AnaSnglMaskOpSeqZSli txt'],
            self.hs['AnaSnglMaskOpSeqPrev btn']
        ]

        AnaSnglMask_GridspecLayout[7, 71:78] = widgets.Button(
            description='=>',
            tooltip='Make New Mask',
            layout={
                'width': '90%',
                'height': '100%'
            },
            disabled=True)
        self.hs['AnaSnglMaskAddName btn'] = AnaSnglMask_GridspecLayout[7,
                                                                       71:78]
        self.hs['AnaSnglMaskAddName btn'].style.button_color = 'darkviolet'

        AnaSnglMask_GridspecLayout[9, 71:78] = widgets.Button(
            description='<=',
            tooltip='Remove a Mask',
            layout={
                'width': '90%',
                'height': '100%'
            },
            disabled=True)
        self.hs['AnaSnglMaskRmName btn'] = AnaSnglMask_GridspecLayout[9, 71:78]
        self.hs['AnaSnglMaskRmName btn'].style.button_color = 'darkviolet'

        AnaSnglMask_GridspecLayout[:, 79:99] = widgets.VBox(layout={
            'width':
            '100%',
            'height':
            '100%',
            'border':
            '1px solid #FFCC11'
        },
                                                            disabled=True)
        self.hs['AnaSnglMaskName box'] = AnaSnglMask_GridspecLayout[:, 79:99]
        self.hs['AnaSnglMaskName lbl'] = widgets.Label('Mask Names')
        self.hs['AnaSnglMaskName sel'] = widgets.Select(layout={
            'width': '90%',
            'height': '70%',
            'left': '5%'
        },
                                                        disabled=True)
        self.hs['AnaSnglMaskNamePH txt'] = widgets.Text(layout={
            'width': '60%',
            'height': 'auto',
            'left': '20%',
            'visibility': 'hidden'
        },
                                                        disabled=True)
        self.hs['AnaSnglMaskNameCfm btn'] = widgets.Button(
            description='Confirm',
            layout={
                'width': '60%',
                'height': 'auto',
                'left': '20%'
            },
            disabled=True)
        self.hs['AnaSnglMaskNameCfm btn'].style.button_color = 'darkviolet'

        self.hs['AnaSnglMaskName box'].children = [
            self.hs['AnaSnglMaskName lbl'], self.hs['AnaSnglMaskName sel'],
            self.hs['AnaSnglMaskNamePH txt'], self.hs['AnaSnglMaskNameCfm btn']
        ]

        self.hs['AnaSnglMaskVarName drpn'].observe(
            self.AnaSnglMaskVarName_drpn_chg, names='value')
        self.hs['AnaSnglMaskOp drpn'].observe(self.AnaSnglMaskOp_drpn_chg,
                                              names='value')
        self.hs['AnaSnglMaskSpecName drpn'].observe(
            self.AnaSnglMaskSpecName_drpn_chg, names='value')
        self.hs['AnaSnglMaskOpHelp btn'].on_click(
            self.AnaSnglMaskOpHelp_btn_clk)
        self.hs['AnaSnglMaskOpRm btn'].on_click(self.AnaSnglMaskOpRm_btn_clk)
        self.hs['AnaSnglMaskOpAdd btn'].on_click(self.AnaSnglMaskOpAdd_btn_clk)
        self.hs['AnaSnglMaskOpSeqZSli txt'].observe(
            self.AnaSnglMaskOpSeqZSli_txt_chg, names='value')
        self.hs['AnaSnglMaskOpSeqPrev btn'].on_click(
            self.AnaSnglMaskOpSeqPrev_btn_clk)
        self.hs['AnaSnglMaskAddName btn'].on_click(
            self.AnaSnglMaskAddName_btn_clk)
        self.hs['AnaSnglMaskRmName btn'].on_click(
            self.AnaSnglMaskRmName_btn_clk)
        self.hs['AnaSnglMaskNameCfm btn'].on_click(
            self.AnaSnglMaskNameCfm_btn_clk)
        ## ## ## ## ## ## define single mask -- end

        ## ## ## ## ## ## combine multiple masks -- start
        AnaCombMask_GridspecLayout = widgets.GridspecLayout(
            17,
            100,
            layout={
                'width': '100%',
                'height': f'{0.68 * (self.form_sz[0] - 136)}px'
            })

        AnaCombMask_GridspecLayout[:, :20] = widgets.VBox(layout={
            'width':
            '100%',
            'height':
            '100%',
            'border':
            '1px solid #FFCC11'
        },
                                                          disabled=True)
        self.hs['AnaCombMaskVar box'] = AnaCombMask_GridspecLayout[:, :20]
        self.hs['AnaCombMaskVar lbl'] = widgets.Label('Sel Mask')
        self.hs['AnaCombMaskVar drpn'] = widgets.Dropdown(
            tooltip='Select a Mask to Make a Combined Mask.',
            options=self.hs['AnaSnglMaskName sel'].options,
            layout={
                'width': '90%',
                'height': 'auto',
                'left': '5%'
            },
            disabled=True)
        self.hs['AnaCombMaskVarInfo txt'] = widgets.Textarea(layout={
            'width': '90%',
            'height': '60%',
            'left': '5%'
        },
                                                             disabled=True)
        self.hs['AnaCombMaskZSli txt'] = widgets.BoundedIntText(
            tooltip='Z Slice #',
            min=0,
            max=1,
            step=1,
            layout={
                'width': '60%',
                'height': '6%',
                'left': '20%'
            },
            disabled=True)
        self.hs['AnaCombMaskPrev btn'] = widgets.Button(description='Preview',
                                                        layout={
                                                            'width': '60%',
                                                            'height': '6%',
                                                            'left': '20%',
                                                            'top': '2%'
                                                        },
                                                        disabled=True)
        self.hs['AnaCombMaskPrev btn'].style.button_color = 'darkviolet'
        self.hs['AnaCombMaskVar box'].children = [
            self.hs['AnaCombMaskVar lbl'], self.hs['AnaCombMaskVar drpn'],
            self.hs['AnaCombMaskVarInfo txt'], self.hs['AnaCombMaskZSli txt'],
            self.hs['AnaCombMaskPrev btn']
        ]

        AnaCombMask_GridspecLayout[7, 21:28] = widgets.Button(
            description='=>',
            tooltip='Add a Mask',
            layout={
                'width': '90%',
                'height': '100%'
            },
            disabled=True)
        self.hs['AnaCombMaskVarAdd btn'] = AnaCombMask_GridspecLayout[7, 21:28]
        self.hs['AnaCombMaskVarAdd btn'].style.button_color = 'darkviolet'

        AnaCombMask_GridspecLayout[9, 21:28] = widgets.Button(
            description='<=',
            tooltip='Remove a Mask',
            layout={
                'width': '90%',
                'height': '100%'
            },
            disabled=True)
        self.hs['AnaCombMaskVarRm btn'] = AnaCombMask_GridspecLayout[9, 21:28]
        self.hs['AnaCombMaskVarRm btn'].style.button_color = 'darkviolet'

        AnaCombMask_GridspecLayout[:, 29:49] = widgets.VBox(layout={
            'width':
            '100%',
            'height':
            '100%',
            'border':
            '1px solid #FFCC11'
        },
                                                            disabled=True)
        self.hs['AnaCombMaskList box'] = AnaCombMask_GridspecLayout[:, 29:49]
        self.hs['AnaCombMasks lbl'] = widgets.Label('Mask List')
        self.hs['AnaCombMaskOp drpn'] = widgets.Dropdown(
            tooltip=
            'Select an Operation on the Selected Single Mask to Make a Combined Mask.',
            options=['', 'Complement (I-A)'],
            value='',
            layout={
                'width': '90%',
                'height': 'auto',
                'left': '5%'
            },
            disabled=True)
        self.hs['AnaCombMaskOpStps sel'] = widgets.Select(layout={
            'width': '90%',
            'height': '60%',
            'left': '5%'
        },
                                                          disabled=True)
        self.hs['AnaCombMasksPH txt'] = widgets.BoundedIntText(
            tooltip='',
            min=0,
            max=1,
            step=1,
            layout={
                'width': '60%',
                'height': '6%',
                'left': '20%',
                'visibility': 'hidden'
            },
            disabled=True)
        self.hs['AnaCombMasksPrev btn'] = widgets.Button(description='Preview',
                                                         layout={
                                                             'width': '60%',
                                                             'height': '6%',
                                                             'left': '20%',
                                                             'top': '2%'
                                                         },
                                                         disabled=True)
        self.hs['AnaCombMasksPrev btn'].style.button_color = 'darkviolet'

        self.hs['AnaCombMaskVar drpn'].observe(self.AnaCombMaskVar_drpn_chg,
                                               names='value')
        self.hs['AnaCombMaskZSli txt'].observe(self.AnaCombMaskZSli_txt_chg,
                                               names='value')
        self.hs['AnaCombMaskPrev btn'].on_click(self.AnaCombMaskPrev_btn_clk)
        self.hs['AnaCombMaskVarAdd btn'].on_click(
            self.AnaCombMaskVarAdd_btn_clk)
        self.hs['AnaCombMaskVarRm btn'].on_click(self.AnaCombMaskVarRm_btn_clk)
        self.hs['AnaCombMaskOp drpn'].observe(self.AnaCombMaskOp_drpn_chg,
                                              names='value')
        self.hs['AnaCombMasksPrev btn'].on_click(self.AnaCombMasksPrev_btn_clk)

        self.hs['AnaCombMaskList box'].children = [
            self.hs['AnaCombMasks lbl'], self.hs['AnaCombMaskOp drpn'],
            self.hs['AnaCombMaskOpStps sel'], self.hs['AnaCombMasksPH txt'],
            self.hs['AnaCombMasksPrev btn']
        ]

        AnaCombMask_GridspecLayout[7, 50:57] = widgets.Button(
            description='=>',
            tooltip='Make New Mask',
            layout={
                'width': '90%',
                'height': '100%'
            },
            disabled=True)
        self.hs['AnaCombMaskAddName btn'] = AnaCombMask_GridspecLayout[7,
                                                                       50:57]
        self.hs['AnaCombMaskAddName btn'].style.button_color = 'darkviolet'

        AnaCombMask_GridspecLayout[9, 50:57] = widgets.Button(
            description='<=',
            tooltip='Remove a Mask',
            layout={
                'width': '90%',
                'height': '100%'
            },
            disabled=True)
        self.hs['AnaCombMaskRmName btn'] = AnaCombMask_GridspecLayout[9, 50:57]
        self.hs['AnaCombMaskRmName btn'].style.button_color = 'darkviolet'

        AnaCombMask_GridspecLayout[:, 58:78] = widgets.VBox(layout={
            'width':
            '100%',
            'height':
            '100%',
            'border':
            '1px solid #FFCC11'
        },
                                                            disabled=True)
        self.hs['AnaCombMaskName box'] = AnaCombMask_GridspecLayout[:, 58:78]
        self.hs['AnaCombMaskName lbl'] = widgets.Label('Mask Names')
        self.hs['AnaCombMaskNamePH drpn'] = widgets.Dropdown(layout={
            'width':
            '90%',
            'height':
            'auto',
            'left':
            '5%',
            'visibility':
            'hidden'
        },
                                                             disabled=True)
        self.hs['AnaCombMaskName sel'] = widgets.Select(layout={
            'width': '90%',
            'height': '60%',
            'left': '5%'
        },
                                                        disabled=True)
        self.hs['AnaCombMaskNamePH txt'] = widgets.Text(layout={
            'width': '60%',
            'height': 'auto',
            'left': '20%',
            'visibility': 'hidden'
        },
                                                        disabled=True)
        self.hs['AnaCombMaskNameCfm btn'] = widgets.Button(
            description='Confirm',
            layout={
                'width': '60%',
                'height': '6%',
                'left': '20%',
                'top': '2%'
            },
            disabled=True)
        self.hs['AnaCombMaskNameCfm btn'].style.button_color = 'darkviolet'

        self.hs['AnaCombMaskName box'].children = [
            self.hs['AnaCombMaskName lbl'], self.hs['AnaCombMaskNamePH drpn'],
            self.hs['AnaCombMaskName sel'], self.hs['AnaCombMaskNamePH txt'],
            self.hs['AnaCombMaskNameCfm btn']
        ]

        self.hs['AnaCombMaskAddName btn'].on_click(
            self.AnaCombMaskAddName_btn_clk)
        self.hs['AnaCombMaskRmName btn'].on_click(
            self.AnaCombMaskRmName_btn_clk)
        self.hs['AnaCombMaskName sel'].observe(self.AnaCombMaskName_sel_chg,
                                               names='value')
        self.hs['AnaCombMaskNameCfm btn'].on_click(
            self.AnaCombMaskNameCfm_btn_clk)

        AnaCombMask_GridspecLayout[8, 83:95] = widgets.Button(
            description='Save Masks',
            layout={
                'width': '100%',
                'height': '100%'
            },
            tooltip=
            'Caution: this will OVERWRITE the generated masks with same names as ones in the current mask list that already exist in the selected h5 file.',
            disabled=True)
        self.hs['AnaMaskSav btn'] = AnaCombMask_GridspecLayout[8, 83:95]
        self.hs['AnaMaskSav btn'].style.button_color = 'darkviolet'

        self.hs['AnaMaskSav btn'].on_click(self.AnaMaskSav_btn_clk)
        ## ## ## ## ## ## combine multiple masks -- end

        self.hs['AnaMask acc'].children = [
            AnaSnglMask_GridspecLayout, AnaCombMask_GridspecLayout
        ]

        self.hs['AnaMask acc'].set_title(0, 'Make A Single Mask')
        self.hs['AnaMask acc'].set_title(1, 'Combine Masks')
        self.hs['AnaMask acc'].selected_index = None

        self.hs['AnaMask tab'].children = [self.hs['AnaMask acc']]
        ## ## ## ## ## preparing masks --end

        ## ## ## ## ##  config analysis -- start
        # self.hs['AnaConfig box'] = widgets.HBox()
        self.hs['AnaAna acc'] = widgets.Accordion(
            titles=['Config Data for Analysis', 'Config Analysis'],
            layout={
                'width': 'auto',
                'height': 'auto'
            })

        ## ## ## ## ## ## config data for analysis -- start
        self.AnaConfAnaData_GridSpecLayout = widgets.GridspecLayout(
            17,
            100,
            layout={
                'width': "100%",
                "height": f'{0.68 * (self.form_sz[0] - 136)}px'
            })

        self.AnaConfAnaData_GridSpecLayout[0, 20:45] = widgets.Dropdown(
            description='ana spec',
            tooltip='choose the spectrum data for further analysis',
            options=[],
            disabled=True,
            layout={'width': '95%'})
        self.hs['AnaAnaSpec drpdn'] = self.AnaConfAnaData_GridSpecLayout[0,
                                                                         20:45]
        self.AnaConfAnaData_GridSpecLayout[1:5, 20:45] = widgets.Textarea(
            disabled=True, layout={
                'width': '100%',
                'height': '100%'
            })
        self.hs['AnaAnaSpecInfo txt'] = self.AnaConfAnaData_GridSpecLayout[
            1:5, 20:45]
        self.AnaConfAnaData_GridSpecLayout[0, 55:80] = widgets.Dropdown(
            description='ana mask',
            tooltip=
            'choose a mask for defining a region of interest for further analysis',
            options=['None'],
            disabled=True,
            layout={'width': '95%'})
        self.hs['AnaAnaMask drpdn'] = self.AnaConfAnaData_GridSpecLayout[0,
                                                                         55:80]
        self.AnaConfAnaData_GridSpecLayout[1:5, 55:80] = widgets.Textarea(
            disabled=True, layout={
                'width': '100%',
                'height': '100%'
            })
        self.hs['AnaAnaMaskInfo txt'] = self.AnaConfAnaData_GridSpecLayout[
            1:5, 55:80]
        self.AnaConfAnaData_GridSpecLayout[6, 44:56] = widgets.Button(
            description='Confirm',
            disabled=True,
            layout={
                'width': '100%',
                'height': '100%'
            })
        self.hs['AnaAnaDataCfm btn'] = self.AnaConfAnaData_GridSpecLayout[
            6, 44:56]
        self.hs['AnaAnaDataCfm btn'].style.button_color = 'darkviolet'

        self.hs['AnaAnaSpec drpdn'].observe(self.AnaAnaSpec_drpdn_chg,
                                            names='value')
        self.hs['AnaAnaMask drpdn'].observe(self.AnaAnaMask_drpdn_chg,
                                            names='value')
        self.hs['AnaAnaDataCfm btn'].on_click(self.AnaAnaDataCfm_btn_clk)
        ## ## ## ## ## ## config data for analysis -- end

        ## ## ## ## ## ## config analysis -- start
        self.AnaConfAna_GridSpecLayout = widgets.GridspecLayout(
            17,
            100,
            layout={
                'width': '100%',
                'height': f'{0.68 * (self.form_sz[0] - 136)}px'
            })
        self.AnaConfAna_GridSpecLayout[0, 0:25] = widgets.Dropdown(
            description='ana type',
            tooltip='choose analysis to apply to the selected data',
            options=[
                'Preproc', 'Decomp', 'Classif', 'Regres', 'Cluster', 'Neighbor'
            ],
            value='Neighbor',
            disabled=True,
            layout={'width': '95%'})
        self.hs['AnaType drpdn'] = self.AnaConfAna_GridSpecLayout[0, 0:25]
        self.AnaConfAna_GridSpecLayout[0, 25:50] = widgets.Dropdown(
            description='ana method',
            tooltip='choose analysis to apply to the selected data',
            options=['KDE', 'PCA', 'KNN'],
            value='KDE',
            disabled=True,
            layout={'width': '95%'})
        self.hs['AnaMethod drpdn'] = self.AnaConfAna_GridSpecLayout[0, 25:50]

        for ii in range(2):
            for jj in range(4):
                self.AnaConfAna_GridSpecLayout[ii+1, jj*25:(jj+1)*25] = \
                    widgets.Dropdown(description = f'p{str(ii*4+jj).zfill(2)}',
                                     value = 'linear',
                                     options = ['linear'],
                                     tooltip = f"analysis function variable {str(ii*4+jj).zfill(2)}",
                                     layout = {'width': "95%"},
                                     disabled=True)
                self.hs[f'AnaConfigPars{ii*4+jj}'] = \
                    self.AnaConfAna_GridSpecLayout[ii+1, jj*25:(jj+1)*25]

        for ii in range(2, 4):
            for jj in range(4):
                self.AnaConfAna_GridSpecLayout[ii+1, jj*25:(jj+1)*25] = \
                    widgets.BoundedFloatText(description = f'p{str(ii*4+jj).zfill(2)}',
                                             value = 0,
                                             min = -1e5,
                                             max = 1e5,
                                             tooltip = f"analysis function variable {str(ii*4+jj).zfill(2)}",
                                             layout = {'width': "95%"},
                                             disabled=True)
                self.hs[f'AnaConfigPars{ii*4+jj}'] = \
                    self.AnaConfAna_GridSpecLayout[ii+1, jj*25:(jj+1)*25]

        self.AnaConfAna_GridSpecLayout[6, 5:95] = \
            widgets.IntProgress(value=0,
                                min=0,
                                max=100,
                                step=1,
                                description='Completing:',
                                bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                orientation='horizontal',
                                indent=False,
                                layout={'width':'95%', 'height':'95%'})
        self.hs['AnaConfigPrgr bar'] = \
            self.AnaConfAna_GridSpecLayout[6, 5:95]

        self.AnaConfAna_GridSpecLayout[7, 45:55] = \
            widgets.Button(description='Compute',
                           description_tip='Perform the analysis',
                           disabled=True)
        self.hs['AnaConfigCmpt btn'] = \
            self.AnaConfAna_GridSpecLayout[7, 45:55]
        self.hs['AnaConfigCmpt btn'].style.button_color = 'darkviolet'

        self.hs['AnaType drpdn'].observe(self.AnaType_drpdn_chg, names='value')
        self.hs['AnaMethod drpdn'].observe(self.AnaMeth_drpdn_chg,
                                           names='value')
        self.hs['AnaConfigPars0'].observe(self.AnaCnfgP0_chg, names='value')
        self.hs['AnaConfigPars1'].observe(self.AnaCnfgP1_chg, names='value')
        self.hs['AnaConfigPars2'].observe(self.AnaCnfgP2_chg, names='value')
        self.hs['AnaConfigPars3'].observe(self.AnaCnfgP3_chg, names='value')
        self.hs['AnaConfigPars4'].observe(self.AnaCnfgP4_chg, names='value')
        self.hs['AnaConfigPars5'].observe(self.AnaCnfgP5_chg, names='value')
        self.hs['AnaConfigPars6'].observe(self.AnaCnfgP6_chg, names='value')
        self.hs['AnaConfigPars7'].observe(self.AnaCnfgP7_chg, names='value')
        self.hs['AnaConfigCmpt btn'].on_click(self.AnaCnfgCmpt_btn_clk)
        self.hs['AnaAna acc'].children = [
            self.AnaConfAnaData_GridSpecLayout, self.AnaConfAna_GridSpecLayout
        ]
        self.hs['AnaAna acc'].set_title(0, 'Config Data for Analysis')
        self.hs['AnaAna acc'].set_title(1, 'Config Analysis')
        self.hs['AnaAna acc'].selected_index = None
        ## ## ## ## ## ## config analysis -- end

        # self.hs['AnaConf tab'].children = [self.hs['AnaConfig box']]
        self.hs['AnaConf tab'].children = [self.hs['AnaAna acc']]
        ## ## ## ## ## config analysis -- end

        ## ## ## ## ##  result display -- start
        self.hs['AnaDataDisp box'] = widgets.HBox()
        ## ## ## ## ##  result display -- end

    def boxes_logic(self):

        def component_logic():
            if (self.ana_data_cnfged & (not self.ana_mask_sngl_op_seled)):
                boxes = [
                    'AnaSnglMaskOpSeq box', 'AnaSnglMaskAddName btn',
                    'AnaSnglMaskRmName btn', 'AnaSnglMaskName box'
                ]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            elif (self.ana_data_cnfged & self.ana_mask_sngl_op_seled):
                boxes = [
                    'AnaSnglMaskOpSeq box', 'AnaSnglMaskAddName btn',
                    'AnaSnglMaskRmName btn', 'AnaSnglMaskName box'
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            if ((self.ana_data_cnfged & self.ana_mask_sngl_op_seled) &
                (len(self.hs['AnaSnglMaskName sel'].options) == 0)):
                boxes = ['AnaSnglMaskNameCfm btn']
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            elif ((self.ana_data_cnfged & self.ana_mask_sngl_op_seled) &
                  (len(self.hs['AnaSnglMaskName sel'].options) > 0)):
                boxes = ['AnaSnglMaskNameCfm btn']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            if (self.ana_mask_sngl_cnfged &
                (len(self.hs["AnaCombMaskOpStps sel"].options) == 0)):
                boxes = [
                    'AnaCombMasksPrev btn', 'AnaCombMaskAddName btn',
                    'AnaCombMaskRmName btn', 'AnaCombMaskName box'
                ]
                enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
                boxes = ['AnaMaskSav btn']
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            elif (self.ana_mask_sngl_cnfged &
                  (len(self.hs["AnaCombMaskOpStps sel"].options) > 0)):
                boxes = [
                    'AnaCombMasksPrev btn', 'AnaCombMaskAddName btn',
                    'AnaCombMaskRmName btn'
                ]
                enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
                if len(self.hs['AnaCombMaskName sel'].options) == 0:
                    boxes = ['AnaCombMaskName box']
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=True,
                                         level=-1)
                else:
                    boxes = ['AnaCombMaskName box']
                    enable_disable_boxes(self.hs,
                                         boxes,
                                         disabled=False,
                                         level=-1)
            if self.ana_ana_data_cnfged:
                for ii in self.AnaConfAna_GridSpecLayout.children:
                    ii.disabled = False
            else:
                for ii in self.AnaConfAna_GridSpecLayout.children:
                    ii.disabled = True

        if ((not self.ana_fn_seled) or (not self.ana_data_existed)):
            boxes = [
                'AnaData tab', 'AnaMask tab', 'AnaConf tab', 'AnaDisp tab'
            ]
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            boxes = ['AnaDataFn btn']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
        elif ((self.ana_fn_seled & self.ana_data_existed) &
              (not self.ana_data_cnfged)):
            boxes = ['AnaMask tab', 'AnaDisp tab']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            boxes = ['AnaData tab', 'AnaConf tab']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
        elif ((self.ana_fn_seled & self.ana_data_existed) &
              (self.ana_data_cnfged & (not self.ana_mask_sngl_cnfged))):
            boxes = ['AnaMask tab', 'AnaDisp tab']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            boxes = [
                'AnaData tab', 'AnaSnglMaskVarName box', 'AnaSnglMaskOp box',
                'AnaSnglMaskOpAdd btn', 'AnaSnglMaskOpRm btn', 'AnaConf tab'
            ]
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            self.set_single_mask_op(reset=False)
        elif ((self.ana_fn_seled & self.ana_data_existed) &
              (self.ana_data_cnfged & self.ana_mask_sngl_cnfged) &
              (not self.ana_mask_cnfged)):
            boxes = ['AnaDisp tab']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            boxes = ['AnaData tab', 'AnaMask tab', 'AnaConf tab']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
            self.set_single_mask_op(reset=False)
        elif ((self.ana_fn_seled & self.ana_data_existed) &
              (self.ana_data_cnfged & self.ana_mask_cnfged) &
              (not self.ana_proc_done)):
            boxes = ['AnaDisp tab']
            enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
            boxes = ['AnaData tab', 'AnaMask tab', 'AnaConf tab']
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
        elif ((self.ana_fn_seled & self.ana_data_existed) &
              (self.ana_data_cnfged & self.ana_mask_cnfged) &
              (self.ana_proc_done)):
            boxes = [
                'AnaData tab', 'AnaMask tab', 'AnaConf tab', 'AnaDisp tab'
            ]
            enable_disable_boxes(self.hs, boxes, disabled=False, level=-1)
        component_logic()
        self.lock_message_text_boxes()

    def set_single_mask_op(self, reset=False):
        boxes = [f'AnaSnglMaskOpPar{str(ii).zfill(2)} box' for ii in range(8)]
        enable_disable_boxes(self.hs, boxes, disabled=True, level=-1)
        op = self.hs['AnaSnglMaskOp drpn'].value
        self.hs[
            'AnaSnglMaskOp drpn'].tooltip = xanes_analysis_gui.MASK_OP_PAR_DICT[
                op]['description']
        for key in xanes_analysis_gui.MASK_OP_PAR_DICT[op][
                'default_pars'].keys():
            if reset:
                self.hs[key].value = xanes_analysis_gui.MASK_OP_PAR_DICT[op][
                    'default_pars'][key]['val']
                self.hs[key].tooltip = xanes_analysis_gui.MASK_OP_PAR_DICT[op][
                    'default_pars'][key]['tooltip']
            self.hs[key].disabled = False

    def prep_sngl_mask(self, mk_name, idx=0, confirmed=True):
        """ 
        According to the selected fitting data and a sequence of operations to make a mask. Including all such masks in 
        singled mask collection dictionary structure:
        self.ana_mask_sngl = {key: <mask name>
                                  {key: str <var_name: variable name in string>,
                                   key: dictionary <op: operation sequence>,
                                        {key: int <operation order>
                                              {key: str <flt_name: filter's name>,
                                               key: dictionary <pars: filter function parameters>}}}}
        """
        if confirmed:
            mask = self.ana_data[self.ana_mask_sngl[mk_name]['var_name']]
            op = self.ana_mask_sngl[mk_name]['op']
            spec = self.ana_spec[self.hs['AnaSnglMaskSpecName drpn'].value]
            info = ''
            for k0 in op.keys():
                info = info + op[k0]['flt_name'] + ':' + str(
                    [f'{k1}: {i1},'
                     for k1, i1 in op[k0]['pars'].items()]) + '\n'
            if ((len(mask.shape) == 3) & (idx != -1)):
                idx = self.hs['AnaSnglMaskOpSeqZSli txt'].value
                mask = mask[idx]
                spec = spec[idx]
            for key in sorted(op.keys()):
                flt = xanes_analysis_gui.MASK_OP_PAR_DICT[op[key]
                                                          ['flt_name']]['flt']
                mask = flt(mask, **op[key]['pars'])
        else:
            shp = self.ana_data[self.hs['AnaSnglMaskVarName drpn'].value].shape
            mask = self.ana_data[self.hs['AnaSnglMaskVarName drpn'].value]
            spec = self.ana_data[self.hs['AnaSnglMaskSpecName drpn'].value]
            if ((len(shp) == 3) & (idx != -1)):
                idx = self.hs['AnaSnglMaskOpSeqZSli txt'].value
                mask = mask[idx]
                spec = spec[idx]
            for key in sorted(self.ana_mask_sngl_op.keys()):
                flt = xanes_analysis_gui.MASK_OP_PAR_DICT[
                    self.ana_mask_sngl_op[key]['flt_name']]['flt']
                mask = flt(mask, **self.ana_mask_sngl_op[key]['pars'])
        return mask, spec

    def prep_comb_mask(self, mk_name, idx=0, confirmed=True):
        """ 
        combined mask collection dictionary structure:
        self.ana_mask_sngl = {
            key: <mask name> {
                key: str <var_name: variable name in string>,
                key: dictionary <op: operation sequence> {
                    key: int <operation order> {
                        key: str <flt_name: filter's name>,
                        key: dictionary <filter function parameters>
                    }
                }
            }
        }
        """
        """
        self.ana_mask_comb_op = {
            key: int <operation order> {
                key: str <op: operation name>,
                key: str <op_symb: operation symbol>,
                key: str <B name: name of the second argument in the binary function>,
                key: str <B short name: short name of the second argument in the binary function>
            } 
        
        }
        """
        """
        SET_OP_DICT = {
            '': {
            'description': 'identity operation',
            'op_symb': '',
            'flt': xpa.set_identity
            },
            'Union (+)': {
                'description': 'Union operation between two masks',
                'op_symb': ' + ',
                'flt': xpa.set_union
            },
            'Intersection (x)': {
                'description': 'Intersection between two masks',
                'op_symb': ' x ',
                'flt': xpa.set_intersection
            },
            'Differnece (-)': {
                'description': 'Difference between two masks',
                'op_symb': ' - ',
                'flt': xpa.set_difference
            },
            'Complement (I-A)': {
                'description': 'Complement of a mask',
                'op_symb': '1 - A',
                'flt': xpa.set_complement
            }
        }
        """
        if confirmed:
            op = self.ana_mask_comb[mk_name]["op"]
            for key in sorted(op.keys()):
                B_name = op[key]["B name"]
                if key == '0':
                    mask = self.ana_mask[B_name]['mask']
                elif op[key]["op"] == "Complement (I-A)":
                    flt = xanes_analysis_gui.SET_OP_DICT[op[key]["op"]]["flt"]
                    mask = flt(mask)
                    # mask = xpa.set_complement(mask)
                else:
                    flt = xanes_analysis_gui.SET_OP_DICT[op[key]["op"]]["flt"]
                    mask = flt(mask, self.ana_mask[B_name]['mask'])
            return mask.astype(np.int8)
        else:
            for key in sorted(self.ana_mask_comb_op.keys()):
                B_name = self.ana_mask_comb_op[key]["B name"]
                if key == '0':
                    mask = self.ana_mask[B_name]['mask']
                elif self.ana_mask_comb_op[key]["op"] == "Complement (I-A)":
                    flt = xanes_analysis_gui.SET_OP_DICT[
                        self.ana_mask_comb_op[key]["op"]]["flt"]
                    mask = flt(mask)
                else:
                    flt = xanes_analysis_gui.SET_OP_DICT[
                        self.ana_mask_comb_op[key]["op"]]["flt"]
                    mask = flt(mask, self.ana_mask[B_name]['mask'])
            return mask.astype(np.int8)

    def AnaDataFn_btn_clk(self, a):
        if len(a.files[0]) != 0:
            self.ana_fn_seled = True
            self.ana_fn = os.path.abspath(a.files[0])
            self.ana_data_existed = False
            update_json_content(
                self.parent_h.global_h.GUI_cfg_file,
                {'cwd': os.path.dirname(os.path.abspath(a.files[0]))})
            self.parent_h.global_h.cwd = os.path.dirname(
                os.path.abspath(a.files[0]))
            with h5py.File(self.ana_fn, 'r') as f:
                self.ana_proc_data_list = []
                for key in f.keys():
                    if 'processed_XANES' in key:
                        self.ana_proc_spec_path_in_h5 = '/' + key + '/proc_spectrum'
                        self.ana_proc_data_list = list(
                            f[self.ana_proc_spec_path_in_h5].keys())
                        self.ana_data_existed = True
            if self.ana_data_existed:
                self.ana_spec_list = []
                for key in self.ana_proc_data_list:
                    if key in [
                            'wl_pos_fit', 'wl_pos_dir', 'edge50_pos_fit',
                            'edge50_pos_dir', 'edge_pos_fit', 'edge_pos_dir'
                    ]:
                        self.ana_spec_list.append(key)
                        self.ana_spec[key] = h5_lazy_reader(
                            self.ana_fn,
                            os.path.join(self.ana_proc_spec_path_in_h5, key),
                            np.s_[:])
                for key in self.ana_proc_data_list:
                    self.ana_data[key] = h5_lazy_reader(
                        self.ana_fn,
                        os.path.join(self.ana_proc_spec_path_in_h5, key),
                        np.s_[:])
                self.hs['AnaDataFn txt'].value = self.ana_fn
            else:
                self.ana_proc_data_list = []
                self.ana_spec_list = []
            self.hs['AnaAvaiDatasets sel'].options = self.ana_proc_data_list
            self.hs['AnaSnglMaskSpecName drpn'].options = self.ana_spec_list
            self.hs['AnaAnaSpec drpdn'].options = self.ana_spec_list
            if len(self.ana_spec_list) != 0:
                self.hs['AnaSnglMaskSpecName drpn'].index = 0
                self.hs['AnaAnaSpec drpdn'].index = 0
            self.hs['AnaAnaMask drpdn'].options = ['None']
        else:
            self.ana_fn_seled = False
            self.ana_proc_data_list = []
            self.hs['AnaAvaiDatasets sel'].options = self.ana_proc_data_list
            self.hs['AnaAvaiDatasets sel'].value = self.ana_proc_data_list[0]
        self.boxes_logic()

    def AnaAvaiDatasets_sel_chg(self, a):
        with h5py.File(self.ana_fn, 'r') as f:
            try:
                shape = f[self.ana_proc_spec_path_in_h5 + '/' +
                          self.hs['AnaAvaiDatasets sel'].value].shape
            except:
                shape = None
            try:
                dmin = np.min(f[self.ana_proc_spec_path_in_h5 + '/' +
                                self.hs['AnaAvaiDatasets sel'].value])
            except:
                dmin = None
            try:
                dmax = np.max(f[self.ana_proc_spec_path_in_h5 + '/' +
                                self.hs['AnaAvaiDatasets sel'].value])
            except:
                dmax = None
            try:
                mean = np.mean(f[self.ana_proc_spec_path_in_h5 + '/' +
                                 self.hs['AnaAvaiDatasets sel'].value])
            except:
                mean = None

            try:
                with self.hs['AnaDataHist plt']:
                    # plot histogram of the selected dataset in this canvas with plt.hist()
                    pass
            except:
                pass
        self.hs['AnaDatasetInfo txt'].value = 'shape: ' + str(shape) + '\n' +\
                                              'min: ' + str(dmin) + '\n' +\
                                              'max: ' +str(dmax) + '\n' +\
                                              'mean: ' +str(mean)
        self.boxes_logic()

    def AnaSelDatasetAdd_btn_clk(self, a):
        seled_ds = list(self.hs['AnaSelVars sel'].options)
        new_ds = self.hs['AnaAvaiDatasets sel'].value
        if new_ds not in seled_ds:
            seled_ds.append(new_ds)
            eqn = ['=' for ii in range(len(seled_ds))]

            self.hs['AnaVarName sel'].options = seled_ds
            self.hs['AnaVarsEq sel'].options = eqn
            self.hs['AnaSelVars sel'].options = seled_ds
            self.hs['AnaSelVars sel'].value = new_ds
            self.hs['AnaSnglMaskVarName drpn'].options = self.hs[
                'AnaVarName sel'].options
            self.hs['AnaSnglMaskVarName drpn'].value = self.hs[
                'AnaVarName sel'].value
        if len(self.hs['AnaSelVars sel'].options) != 0:
            self.ana_data_cnfged = True
        else:
            self.ana_data_cnfged = False
        self.boxes_logic()

    def AnaSelDatasetRm_btn_clk(self, a):
        seled_ds = list(self.hs['AnaSelVars sel'].options)
        vars_nm = list(self.hs['AnaVarName sel'].options)
        eqn = list(self.hs['AnaVarsEq sel'].options)
        idx = seled_ds.index(self.hs['AnaSelVars sel'].value)

        self.ana_data.pop(seled_ds[idx])
        seled_ds.pop(idx)
        vars_nm.pop(idx)
        eqn.pop(idx)

        self.hs['AnaVarName sel'].options = vars_nm
        self.hs['AnaVarsEq sel'].options = eqn
        self.hs['AnaSelVars sel'].options = seled_ds
        self.hs['AnaSnglMaskVarName drpn'].options = self.hs[
            'AnaVarName sel'].options
        self.hs['AnaSnglMaskVarName drpn'].value = self.hs[
            'AnaVarName sel'].value
        if len(self.hs['AnaSelVars sel'].options) != 0:
            self.ana_data_cnfged = True
        else:
            self.ana_data_cnfged = False
        self.boxes_logic()

    def AnaSelVars_sel_chg(self, a):
        if self.hs['AnaSelVars sel'].value is not None:
            seled_ds = list(self.hs['AnaSelVars sel'].options)
            idx = seled_ds.index(self.hs['AnaSelVars sel'].value)
            self.hs['AnaVarName sel'].index = idx
            self.hs['AnaVarsEq sel'].index = idx
            self.ana_data_cnfged = True
            self.ana_mask_sngl_op_seled = False
        else:
            self.ana_data_cnfged = False
            self.ana_mask_sngl_op_seled = False
        self.boxes_logic()

    def AnaSelDatasetPrev_btn_clk(self, a):
        if self.viewer_type == 'napari':
            self.viewer = napari.current_viewer()
            if self.viewer is not None:
                self.viewer.close()
            self.viewer = napari.Viewer()
            data = self.ana_data[self.hs['AnaSelVars sel'].value].compute()
            self.viewer.add_image(data)
            napari.run()
        else:
            fiji_viewer_off(self.parent_h.global_h,
                            gui_h=self,
                            viewer_name='xanes_pp_data_prev_viewer')
            fiji_viewer_on(self.parent_h.global_h,
                           gui_h=self,
                           viewer_name='xanes_pp_data_prev_viewer')
        self.boxes_logic()

    def AnaSelDatasetPrevOpt_rad_chg(self, a):
        self.viewer_type = a['owner'].value

    def AnaSnglMaskVarName_drpn_chg(self, a):
        self.ana_mask_sngl_op = {}
        self.hs['AnaSnglMaskOpSeq sel'].options = []
        self.hs['AnaSnglMaskOpSeqPrev btn'].disabled = True
        self.hs['AnaSnglMaskOpSeqZSli txt'].disabled = True
        s = self.ana_data[self.hs['AnaSnglMaskVarName drpn'].value].shape
        self.hs['AnaSnglMaskOpSeqZSli txt'].value = 0
        if len(s) == 3:
            self.hs['AnaSnglMaskOpSeqZSli txt'].max = s[0]
        self.ana_mask_sngl_op_seled = False
        self.boxes_logic()

    def AnaSnglMaskSpecName_drpn_chg(self, a):
        self.ana_spec_name = a['owner'].value
        self.boxes_logic()

    def AnaSnglMaskOp_drpn_chg(self, a):
        self.set_single_mask_op(reset=True)

    def AnaSnglMaskOpHelp_btn_clk(self, a):
        op = self.hs['AnaSnglMaskOp drpn'].value
        url = xanes_analysis_gui.MASK_OP_PAR_DICT[op]['url']
        with self.hs['AnaSnglMaskOpHelp out']:
            display(Javascript(f'window.open("{url}");'))

    def AnaSnglMaskOpAdd_btn_clk(self, a):
        idx = str(len(self.ana_mask_sngl_op.keys()))
        op = self.hs['AnaSnglMaskOp drpn'].value
        self.ana_mask_sngl_op[idx] = {}
        self.ana_mask_sngl_op[idx]['flt_name'] = op
        self.ana_mask_sngl_op[idx]['pars'] = {}
        for key in xanes_analysis_gui.MASK_OP_PAR_DICT[op][
                'default_pars'].keys():
            self.ana_mask_sngl_op[idx]['pars'][
                xanes_analysis_gui.MASK_OP_PAR_DICT[op]['default_pars'][key]
                ['var']] = self.hs[key].value
        op_list = list(self.hs['AnaSnglMaskOpSeq sel'].options)
        op_list.append(op)
        self.hs['AnaSnglMaskOpSeq sel'].options = op_list
        self.hs['AnaSnglMaskOpSeqPrev btn'].disabled = False
        s = self.ana_data[self.hs['AnaSnglMaskVarName drpn'].value].shape
        if len(s) == 2:
            self.hs['AnaSnglMaskOpSeqZSli txt'].disabled = True
        else:
            self.hs['AnaSnglMaskOpSeqZSli txt'].disabled = False
        self.ana_mask_sngl_op_seled = True
        self.boxes_logic()

    def AnaSnglMaskOpRm_btn_clk(self, a):
        rm_op = self.hs['AnaSnglMaskOpSeq sel'].value
        idx = self.hs['AnaSnglMaskOpSeq sel'].index
        if rm_op is not None:
            op_list = list(self.hs['AnaSnglMaskOpSeq sel'].options)
            op_list.remove(rm_op)
            self.hs['AnaSnglMaskOpSeq sel'].options = op_list
            self.ana_mask_sngl_op.pop(str(idx))
            op = {}
            for ii, key in enumerate(sorted(self.ana_mask_sngl_op.keys())):
                op[ii] = self.ana_mask_sngl_op[key]
            self.ana_mask_sngl_op = op
            self.ana_mask_sngl_op_seled = True
        if len(self.hs['AnaSnglMaskOpSeq sel'].options) == 0:
            self.hs['AnaSnglMaskOpSeqPrev btn'].disabled = True
            self.hs['AnaSnglMaskOpSeqZSli txt'].disabled = True
            self.ana_mask_sngl_op_seled = False
        self.boxes_logic()

    def AnaSnglMaskOpSeqZSli_txt_chg(self, a):
        pass

    def AnaSnglMaskOpSeqPrev_btn_clk(self, a):
        mk_name = self.hs['AnaSnglMaskVarName drpn'].value
        idx = self.hs['AnaSnglMaskOpSeqZSli txt']
        mask, spec = self.prep_sngl_mask(mk_name, idx=idx, confirmed=False)
        fiji_viewer_off(self.parent_h.global_h, gui_h=self, viewer_name='all')
        fiji_viewer_on(self.parent_h.global_h,
                       gui_h=self,
                       viewer_name='xanes_pp_mask_prev_viewer',
                       data=mask,
                       win_name='xanes_pp_mask')
        fiji_viewer_on(self.parent_h.global_h,
                       gui_h=self,
                       viewer_name='xanes_pp_mask_prev_viewer',
                       data=spec,
                       win_name='xanes_pp_spec')
        fiji_viewer_on(self.parent_h.global_h,
                       gui_h=self,
                       viewer_name='xanes_pp_mask_prev_viewer',
                       data=delayed(spec * mask),
                       win_name='masked_xanes_pp_spec')

    def AnaSnglMaskAddName_btn_clk(self, a):
        n = len(self.ana_mask_sngl.keys())
        var = self.hs['AnaSnglMaskVarName drpn'].value
        mk = f'mk{str(n).zfill(2)}_' + var
        self.ana_mask_sngl[mk] = {}
        self.ana_mask_sngl[mk]['var_name'] = var
        self.ana_mask_sngl[mk]['op'] = self.ana_mask_sngl_op

        mk_list = list(self.hs['AnaSnglMaskName sel'].options)
        mk_list.append(mk)
        self.hs['AnaSnglMaskName sel'].options = mk_list
        self.boxes_logic()

    def AnaSnglMaskRmName_btn_clk(self, a):
        if ((len(self.hs['AnaCombMaskName sel'].options) == 0) and (len(self.hs['AnaCombMaskOpStps sel'].options))):
            mk_list = list(self.hs['AnaSnglMaskName sel'].options)
            mk = self.hs['AnaSnglMaskName sel'].value
            if mk:
                idx = int(mk.split('_')[0].strip('mk'))
                # if len(mk_list) != 0:
                self.ana_mask_sngl.pop(mk)
                mk_list.pop(idx)
                if mk in self.ana_mask.keys():
                    self.ana_mask.pop(mk)
                if mk_list:
                    for order, mask_name in enumerate(mk_list):
                        mk_list[order] = mask_name.replace(
                            mask_name.split('_')[0], f'mk{str(order).zfill(2)}')
                        tem = self.ana_mask_sngl[mask_name]
                        self.ana_mask_sngl.pop(mask_name)
                        self.ana_mask_sngl[mk_list[order]] = tem
                    self.hs['AnaSnglMaskName sel'].options = mk_list
                else:
                    self.hs['AnaSnglMaskName sel'].options = []
                    self.ana_mask_sngl_cnfged = False
                self.hs['AnaCombMaskVar drpn'].options = self.hs[
                    'AnaSnglMaskName sel'].options
        self.boxes_logic()

    def AnaSnglMaskNameCfm_btn_clk(self, a):
        # TODO: decide if apply compute and dataset structure in h5
        """
        self.ana_mask_sngl = {key: <mask name>
                                  {key: str <var_name: variable name in string>,
                                   key: dictionary <op: operation sequence>,
                                        {key: int <operation order>
                                              {key: str <flt_name: filter's name>,
                                               key: dictionary <pars: filter function parameters>}}}}
        """
        fiji_viewer_off(self.parent_h.global_h, gui_h=self, viewer_name='all')
        if len(self.hs['AnaSnglMaskName sel'].options) != 0:
            self.hs['AnaCombMaskVar drpn'].options = self.hs[
                'AnaSnglMaskName sel'].options
            self.hs['AnaCombMaskVar drpn'].index = 0
            for key in self.hs['AnaSnglMaskName sel'].options:
                mask, _ = self.prep_sngl_mask(key, idx=-1, confirmed=True)
                self.ana_mask[key] = {
                    'mask': mask,
                    'source': {
                        'var_name': self.ana_mask_sngl[key]['var_name'],
                        'op': self.ana_mask_sngl[key]['op']
                    }
                }
            self.ana_mask_sngl_cnfged = True
        else:
            self.ana_mask_sngl_cnfged = False
        if len(self.ana_mask.keys()) == 0:
            self.hs['AnaAnaMask drpdn'].options = ['None']
        else:
            self.hs['AnaAnaMask drpdn'].options = list(
                self.ana_mask.keys()) + ['None']
            self.hs['AnaAnaMask drpdn'].value = 'None'
        self.boxes_logic()

    def AnaCombMaskVar_drpn_chg(self, a):
        info = ''
        if a['owner'].value is not None:
            op = self.ana_mask_sngl[a['owner'].value]['op']
            for k0 in op.keys():
                info = info + op[k0]['flt_name'] + ':' + str(
                    [f'{k1}: {i1},'
                     for k1, i1 in op[k0]['pars'].items()]) + '\n'
        self.hs['AnaCombMaskVarInfo txt'].value = info

    def AnaCombMaskPrev_btn_clk(self, a):
        idx = self.hs['AnaCombMaskZSli txt'].value
        mk_name = self.hs["AnaCombMaskVar drpn"].value
        if mk_name in self.ana_mask.keys():
            mask = self.ana_mask[mk_name]['mask']
            if len(mask.shape) == 3:
                mask = mask[idx]
        elif mk_name in self.ana_mask_comb.keys():
            # TODO: add newly added combined mask into the list so need to update mask generator
            # mask = self.prep_comb_mask(mk_name, idx=idx, confirmed=False)
            pass
        spec = self.ana_spec[self.ana_spec_name]
        fiji_viewer_off(self.parent_h.global_h, gui_h=self, viewer_name='all')
        fiji_viewer_on(self.parent_h.global_h,
                       gui_h=self,
                       viewer_name='xanes_pp_mask_prev_viewer',
                       data=mask,
                       win_name='xanes_pp_mask')
        fiji_viewer_on(self.parent_h.global_h,
                       gui_h=self,
                       viewer_name='xanes_pp_mask_prev_viewer',
                       data=spec,
                       win_name='xanes_pp_spec')
        fiji_viewer_on(self.parent_h.global_h,
                       gui_h=self,
                       viewer_name='xanes_pp_mask_prev_viewer',
                       data=delayed(spec * mask),
                       win_name='masked_xanes_pp_spec')

    def AnaCombMaskOp_drpn_chg(self, a):
        pass

    def AnaCombMaskZSli_txt_chg(self, a):
        pass

    def AnaCombMaskVarAdd_btn_clk(self, a):
        """
        SET_OP_DICT = {
        'Union (+)': {
            'description': 'Union operation between two masks',
            'op_symb': ' + ',
            'flt': xpa.set_union
        },
        'Intersection (x)': {
            'description': 'Intersection between two masks',
            'op_symb': ' x ',
            'flt': xpa.set_intersection
        },
        'Differnece (-)': {
            'description': 'Difference between two masks',
            'op_symb': ' - ',
            'flt': xpa.set_difference
        },
        'Complement (I-A)': {
            'description': 'Complement of a mask',
            'op_symb': '1 - A',
            'flt': xpa.set_complement
        }
    }
        """
        mk = self.hs['AnaCombMaskVar drpn'].value
        mk_short = mk.split('_')[0]
        op = self.hs['AnaCombMaskOp drpn'].value
        op_symb = xanes_analysis_gui.SET_OP_DICT[
            self.hs['AnaCombMaskOp drpn'].value]['op_symb']
        info = list(self.hs["AnaCombMaskOpStps sel"].options)
        idx = str(len(info))
        if idx == '0':
            self.hs['AnaCombMaskOp drpn'].options = [
                'Union (+)', 'Intersection (x)', 'Differnece (-)',
                'Complement (I-A)'
            ]

        self.ana_mask_comb_op[idx] = {}
        # if idx == '0':
        #     self.ana_mask_comb_op[idx]["op"] = "identity"
        #     self.ana_mask_comb_op[idx]["op_symb"] = "1x"
        #     self.ana_mask_comb_op[idx]["B name"] = mk
        #     self.ana_mask_comb_op[idx]["B short name"] = mk_short
        #     info = [
        #         f"{mk_short}",
        #     ]
        # else:
        #     self.ana_mask_comb_op[idx]["op"] = op
        #     self.ana_mask_comb_op[idx]["op_symb"] = op_symb
        #     self.ana_mask_comb_op[idx]["B name"] = mk
        #     self.ana_mask_comb_op[idx]["B short name"] = mk_short

        #     info = list(self.hs["AnaCombMaskOpStps sel"].options)
        #     if op == "Complement (I-A)":
        #         info[0] = "[" + info[0]
        #         info.append(f"].complement")
        #     else:
        #         info.append(f"{op_symb} {mk_short}")

        self.ana_mask_comb_op[idx]["op"] = op
        self.ana_mask_comb_op[idx]["op_symb"] = op_symb
        self.ana_mask_comb_op[idx]["B name"] = mk
        self.ana_mask_comb_op[idx]["B short name"] = mk_short
        info = list(self.hs["AnaCombMaskOpStps sel"].options)
        if op == "Complement (I-A)":
            info[0] = "[" + info[0]
            info.append(f"].complement")
        else:
            info.append(f"{op_symb} {mk_short}")

        self.hs["AnaCombMaskOpStps sel"].options = info
        self.ana_mask_comb_op_seled = True
        self.boxes_logic()

    def AnaCombMaskVarRm_btn_clk(self, a):
        rm_op = self.hs["AnaCombMaskOpStps sel"].value
        idx = self.hs["AnaCombMaskOpStps sel"].index
        if ((idx is not None) & (idx != 0)):
            op_list = list(self.hs["AnaCombMaskOpStps sel"].options)
            op_list.remove(rm_op)
            self.hs["AnaCombMaskOpStps sel"].options = op_list
            self.ana_mask_comb_op.pop(str(idx))
            op = {}
            for new_idx, old_idx in enumerate(
                    sorted(self.ana_mask_comb_op.keys())):
                op[str(new_idx)] = self.ana_mask_comb_op[old_idx]
            self.ana_mask_comb_op = op
        elif (
            (len(self.hs["AnaCombMaskOpStps sel"].options) != 1) & (idx == 0)):
            print("Cannot remove the first mask.")
        else:
            op_list = list(self.hs["AnaCombMaskOpStps sel"].options)
            op_list.remove(rm_op)
            self.hs["AnaCombMaskOpStps sel"].options = op_list
            self.ana_mask_comb_op.pop(str(idx))
        if len(self.hs["AnaCombMaskOpStps sel"].options) == 0:
            self.hs['AnaCombMaskOp drpn'].options = ['', 'Complement (I-A)']
            self.hs['AnaCombMaskOp drpn'].value = ''
            self.ana_mask_comb_op_seled = False
        self.boxes_logic()

    def AnaCombMasksPrev_btn_clk(self, a):
        idx = self.hs['AnaCombMaskZSli txt'].value
        mk_name = self.hs["AnaCombMaskVar drpn"].value
        mask = self.prep_comb_mask(mk_name, idx=idx, confirmed=False)
        spec = self.ana_spec[self.ana_spec_name]
        fiji_viewer_off(self.parent_h.global_h, gui_h=self, viewer_name='all')
        fiji_viewer_on(self.parent_h.global_h,
                       gui_h=self,
                       viewer_name='xanes_pp_mask_prev_viewer',
                       data=mask,
                       win_name='xanes_pp_mask')
        fiji_viewer_on(self.parent_h.global_h,
                       gui_h=self,
                       viewer_name='xanes_pp_mask_prev_viewer',
                       data=spec,
                       win_name='xanes_pp_spec')
        fiji_viewer_on(self.parent_h.global_h,
                       gui_h=self,
                       viewer_name='xanes_pp_mask_prev_viewer',
                       data=delayed(spec * mask),
                       win_name='masked_xanes_pp_spec')

    def AnaCombMaskAddName_btn_clk(self, a):
        n = len(self.ana_mask_comb.keys())
        mk = f'comb_mk{str(n).zfill(2)}'
        self.ana_mask_comb[mk] = {}
        self.ana_mask_comb[mk]['op'] = self.ana_mask_comb_op

        mk_list = list(self.hs['AnaCombMaskName sel'].options)
        mk_list.append(mk)
        self.hs['AnaCombMaskName sel'].options = mk_list
        self.boxes_logic()

    def AnaCombMaskRmName_btn_clk(self, a):
        mk_list = list(self.hs['AnaCombMaskName sel'].options)
        mk = self.hs['AnaCombMaskName sel'].value
        if mk:
            idx = int(mk.strip('comb_mk'))
            if len(mk_list) != 0:
                self.ana_mask_comb.pop(mk)
                mk_list.pop(idx)
                if mk in self.ana_mask.keys():
                    self.ana_mask.pop(mk)
            self.hs['AnaCombMaskName sel'].options = mk_list
            if mk_list:
                for order, mask_name in enumerate(mk_list):
                    mk_list[order] = mask_name.replace(
                        mask_name.split('_')[-1], f'mk{str(order).zfill(2)}')
                    tem = self.ana_mask_comb[mask_name]
                    self.ana_mask_comb.pop(mask_name)
                    self.ana_mask_comb[mk_list[order]] = tem
                self.hs['AnaCombMaskName sel'].options = mk_list
            else:
                self.hs['AnaCombMaskName sel'].options = []
                self.ana_mask_comb_cnfged = False
        self.boxes_logic()

    def AnaCombMaskName_sel_chg(self, a):
        # show combination information as tooltip
        pass

    def AnaCombMaskNameCfm_btn_clk(self, a):
        # TODO: decide if apply compute and dataset structure in h5
        """
        self.ana_mask_comb = {
            key: str <mask name>: {
                key: dictionary <op: operation sequence>: {
                    key: int <operation order>: {
                        key: str <op: operation name>,
                        key: str <op_symb: operation symbol>,
                        key: str <B name: name of the second argument in the binary function>,
                        key: str <B short name: short name of the second argument in the binary function>
                    } 
                }
            },
        }
        """
        fiji_viewer_off(self.parent_h.global_h, gui_h=self, viewer_name='all')
        if len(self.hs['AnaCombMaskOpStps sel'].options) != 0:
            for key in self.hs['AnaCombMaskName sel'].options:
                mask = self.prep_comb_mask(key, idx=-1, confirmed=True)
                self.ana_mask[key] = {
                    'mask': mask,
                    'source': self.ana_mask_comb[key]['op']
                }
            self.ana_mask_comb_cnfged = True
        else:
            self.ana_mask_comb_cnfged = False
        if len(self.ana_mask.keys()) == 0:
            self.hs['AnaAnaMask drpdn'].options = ['None']
        else:
            self.hs['AnaAnaMask drpdn'].options = list(
                self.ana_mask.keys()) + ['None']
            self.hs['AnaAnaMask drpdn'].value = 'None'
        self.boxes_logic()

    def AnaMaskSav_btn_clk(self, a):
        with h5py.File(self.ana_fn, 'r+') as f:
            if self.parent_h.gui_name == 'xanes2D':
                if 'gen_masks' in f['processed_XANES2D'].keys():
                    del f['processed_XANES2D/gen_masks']
                gm0 = f['processed_XANES2D'].create_group('gen_masks')
                for mask_name in self.ana_mask.keys():
                    if mask_name in gm0.keys():
                        del gm0[mask_name]
                    gm1 = gm0.create_group(mask_name)
                    gm1.create_dataset(
                        mask_name,
                        data=self.ana_mask[mask_name]['mask'].compute().astype(
                            np.int8),
                        dtype=np.int8)
                    dicttoh5(
                        self.ana_mask[mask_name]['source'],
                        f,
                        overwrite_data=True,
                        h5path=
                        f'/processed_XANES2D/gen_masks/{mask_name}/source')
            elif self.parent_h.gui_name == 'xanes3D':
                if 'gen_masks' not in f['processed_XANES3D'].keys():
                    gm0 = f['processed_XANES3D'].create_group('gen_masks')
                else:
                    gm0 = f['processed_XANES3D/gen_masks']
                for mask_name in self.ana_mask.keys():
                    if mask_name in gm0.keys():
                        del gm0[mask_name]
                    gm1 = gm0.create_group(mask_name)
                    gm1.create_dataset(
                        mask_name,
                        data=self.ana_mask[mask_name]['mask'].compute().astype(
                            np.int8),
                        dtype=np.int8)
                    dicttoh5(
                        self.ana_mask[mask_name]['source'],
                        f,
                        overwrite_data=True,
                        h5path=
                        f'/processed_XANES3D/gen_masks/{mask_name}/source')
        self.boxes_logic()

    def AnaAnaSpec_drpdn_chg(self, a):
        spec_nm = self.hs['AnaAnaSpec drpdn'].value
        if ((spec_nm is None) or (spec_nm == 'None')):
            self.hs['AnaAnaSpecInfo txt'].value = ''
        else:
            self.hs[
                'AnaAnaSpecInfo txt'].value = f"Shape: {self.ana_spec[spec_nm].shape}\ndtype: {self.ana_spec[spec_nm].dtype}"
        self.ana_ana_data_cnfged = False
        self.boxes_logic()

    def AnaAnaMask_drpdn_chg(self, a):
        mask_nm = self.hs['AnaAnaMask drpdn'].value
        if ((mask_nm is None) or (mask_nm == 'None')):
            self.hs['AnaAnaMaskInfo txt'].value = ''
        else:
            self.hs[
                'AnaAnaMaskInfo txt'].value = f"Shape: {self.ana_mask[mask_nm]['mask'].shape}\ndtype: {self.ana_mask[mask_nm]['mask'].dtype}"
        self.ana_ana_data_cnfged = False
        self.boxes_logic()

    def AnaAnaDataCfm_btn_clk(self, a):
        self.ana_ana_spec = self.ana_spec[self.hs['AnaAnaSpec drpdn'].value]
        if self.hs['AnaAnaMask drpdn'].value == 'None':
            self.ana_ana_mask = 1
        else:
            self.ana_ana_mask = self.ana_mask[
                self.hs['AnaAnaMask drpdn'].value]['mask']
        self.ana_ana_data_cnfged = True
        self.boxes_logic()

    def AnaType_drpdn_chg(self, a):
        pass

    def AnaMeth_drpdn_chg(self, a):
        pass

    def AnaCnfgP0_chg(self, a):
        pass

    def AnaCnfgP1_chg(self, a):
        pass

    def AnaCnfgP2_chg(self, a):
        pass

    def AnaCnfgP3_chg(self, a):
        pass

    def AnaCnfgP4_chg(self, a):
        pass

    def AnaCnfgP5_chg(self, a):
        pass

    def AnaCnfgP6_chg(self, a):
        pass

    def AnaCnfgP7_chg(self, a):
        pass

    def AnaCnfgCmpt_btn_clk(self, a):
        pass
