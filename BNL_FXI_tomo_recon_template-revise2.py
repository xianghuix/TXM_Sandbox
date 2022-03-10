# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:26:06 2015

@author: xhxiao
"""

# from tomopy.io.reader import *
# import numpy as np
# import os.path
# from numpy.testing import assert_allclose
import tomopy
from tomopy.recon.rotation import write_center
from tomopy.recon.algorithm import recon
from scipy.ndimage import zoom
import os, h5py, glob
import numpy as np
import time
import skimage.restoration as skr
import scipy.ndimage.filters as snf
from scipy.ndimage import filters
import dxchange
#from pystackreg import StackReg
from pathlib import Path
import tomo_recon_tools as trt
import gc, sys
import tifffile


#def correctProjAlignment(data, data_ref = None, **kwargs):
#    """
#    :param data: ndarray
#                 projection image stack with misalignments
#    :param mode: string, optional
#                 one of TRANSLATION, RIGID_BODY, SCALED_ROTATION, AFFINE, BILINEAR
#
#    :return: ndarray
#             corrected projection image stack
#    """
#    mode = kwargs['stackReg']['params']['mode']
#
#    if mode.upper() == 'TRANSLATION':
#        sr = StackReg(StackReg.TRANSLATION)
#    elif  mode.upper() == 'RIGID_BODY':
#        sr = StackReg(StackReg.RIGID_BODY)
#    elif  mode.upper() == 'SCALED_ROTATION':
#        sr = StackReg(StackReg.SCALED_ROTATION)
#    elif  mode.upper() == 'AFFINE':
#        sr = StackReg(StackReg.AFFINE)
#    elif  mode.upper() == 'BILINEAR':
#        sr = StackReg(StackReg.BILINEAR)

#    dim = data.shape
#    row_shift = np.ndarray(dim[0])
#    col_shift = np.ndarray(dim[0])
#    row_shift[0] = 0.
#    col_shift[0] = 0.
#    if data_ref is None:
#        # data[:] = sr.register_transform_stack(data, reference='previous')[:]
#        for ii in range(dim[0]-1):
#            tmat = sr.register(data[ii], data[ii+1])
#            data[ii+1, :] = sr.transform(data[ii+1], tmat = tmat)[:]
#            row_shift[ii+1] = -tmat[1, 2]
#            col_shift[ii+1] = -tmat[0, 2]
#    else:
#        for ii in range(dim[0]):
#            tmat = sr.register(data_ref[ii], data[ii])
#            data[ii, :] = sr.transform(data[ii], tmat = tmat)[:]
#            row_shift[ii+1] = -tmat[1, 2]
#            col_shift[ii+1] = -tmat[0, 2]
#
#    return data, row_shift, col_shift

def read_center(fn):
    f = open(fn, 'r')
    idx_center = f.readlines()
    idx_list = []
    center_list = []
    for ii in idx_center:
        if ii.split():
            idx_list.append(int(ii.split()[0]))
            center_list.append(np.float(ii.split()[1]))
    return idx_list, center_list

def sortWedge(data, sliceStart, sliceEnd, **kwargs):
    first_missing_end = kwargs['wedgeData']['wedgeParams']['first_missing_end']
    second_missing_start = kwargs['wedgeData']['wedgeParams']['second_missing_start']
    missing_start = kwargs['wedgeData']['wedgeParams']['missing_start']
    missing_end = kwargs['wedgeData']['wedgeParams']['missing_end']
    BlankAt = kwargs['wedgeData']['wedgeParams']['BlankAt']
    bad_angs = kwargs['wedgeData']['wedgeParams']['bad_angs']

    if bad_angs is None:
        if BlankAt == 90:
            data[missing_start:missing_end,:] = 0
        elif BlankAt == 0:
            data[:first_missing_end,:] = 0
            data[second_missing_start:,:] = 0
    else:
        for ii in range(sliceStart, sliceEnd):
            data[bad_angs[ii], ii-sliceStart, :] = 0
    tifffile.imsave('/NSLS2/xf18id1/users/2020Q1/ENYUAN_Proposal_305409/wedge.tiff', data)
    return data


def ds(data, levels):
    data = zoom(data, levels)
    # axis = list(axis)
    # for ii in axis:
    #     data[:] = downsample(data, level=level, axis=ii)[:]

def dataInfo(filename, showInfor=False):

    print (filename)
    f = h5py.File(filename,"r")
    try:
        arr = f["img_tomo"]
        print ('!!!!! Infor !!!!!')
        dim = arr.shape
        if showInfor == True:
            print ('Data dimension is [Theta:Y:X] = [', dim[0],':', dim[1],':', dim[2],']')
        arr = f["img_bkg"]
        if arr.shape[0] == 1:
            print ('!!!!! Infor !!!!! ')
            print ('There is no white images in this file.')
        arr = f["img_dark"]
        if arr.shape[0] == 1:
            print ('!!!!! Infor !!!!! ')
            print ('There is no dark images in this file.')
        return dim
    except:
        print ('!!!!! Error !!!!!')
        print ('Dataset \'img_tomo\' does not exist in the given file.')
        return 0



def dataStandardReader(filename, sliceStart=0, sliceEnd=20, flat_name=None, dark_name=None,
                       fake_flat=False, fake_dark=False, fake_flat_val=1e4, fake_dark_val=100,
                       ds_levels=None):
    if ds_levels is None:
        ds_use = False
    else:
        ds_use = True

    if flat_name == None:
        flat_name = filename
    if dark_name == None:
        dark_name = filename

    if ds_use:
        f = h5py.File(flat_name,"r")
        try:
            arr = f["img_bkg"]
        except:
            print ('!!!!!Error!!!!!')
            print ('There is no flat images in the file! Please provide an alternative file with flat images (using argument \'flat_name=some_name\'). Quit.')
            exit()
        white = ds(arr[1:9,sliceStart:sliceEnd,:], ds_levels)
        f.close()
        if fake_flat:
            white = ds(fake_flat_val*np.ones([8, white.shape[1], white.shape[2]]), ds_levels)

        f = h5py.File(dark_name,"r")
        try:
            arr = f["img_dark"]
        except:
            print ('!!!!!Error!!!!!')
            print ('There is no dark images in the file! Please provide an alternative file with dark images (using argument \'dark_name=some_name\'). Quit.')
            exit()
        dark = ds(arr[1:9,sliceStart:sliceEnd,:], ds_levels)
        f.close()
        if fake_dark:
            dark = ds(fake_dark_val*np.ones([8, dark.shape[1], dark.shape[2]]), ds_levels)

        f = h5py.File(filename,"r")
        try:
            arr = f["img_tomo"]
        except:
            print ('!!!!!Error!!!!!')
            print('There is no img_tomo in the file! Please provide an alternative file with dark images (using argument \'dark_name=some_name\'). Quit.')
            exit()
        data = ds(arr[:,sliceStart:sliceEnd,:], ds_levels)
        f.close()

        f = h5py.File(filename,"r")
        try:
            arr = f["angle"]
        except:
            print ('!!!!!Error!!!!!')
            print('There is no angle list in the given file. Quit.')
            exit()
        theta = arr[:]
        f.close()
    else:
        f = h5py.File(flat_name,"r")
        try:
            arr = f["img_bkg"]
        except:
            print ('!!!!!Error!!!!!')
            print ('There is no flat images in the file! Please provide an alternative file with flat images (using argument \'flat_name=some_name\'). Quit.')
            exit()
        white = arr[1:9,sliceStart:sliceEnd,:]
        f.close()
        if fake_flat:
            white = fake_flat_val*np.ones([8, white.shape[1], white.shape[2]])

        f = h5py.File(dark_name,"r")
        try:
            arr = f["img_dark"]
        except:
            print ('!!!!!Error!!!!!')
            print ('There is no dark images in the file! Please provide an alternative file with dark images (using argument \'dark_name=some_name\'). Quit.')
            exit()
        dark = arr[1:9,sliceStart:sliceEnd,:]
        f.close()
        if fake_dark:
            dark = fake_dark_val*np.ones([8, dark.shape[1], dark.shape[2]])

        f = h5py.File(filename,"r")
        try:
            arr = f["img_tomo"]
        except:
            print ('!!!!!Error!!!!!')
            print('There is no img_tomo in the file! Please provide an alternative file with dark images (using argument \'dark_name=some_name\'). Quit.')
            exit()
        data = arr[:,sliceStart:sliceEnd,:]
        f.close()

        f = h5py.File(filename,"r")
        try:
            arr = f["angle"]
        except:
            print ('!!!!!Error!!!!!')
            print('There is no angle list in the given file. Quit.')
            exit()
        theta = arr[:]
        f.close()

    gc.collect()
    print ('Data is read successfully')
    return data,white,dark,theta




def getFiles(raw_data_top_dir, scan_id):
    # data_top_dir = raw_data_top_dir + '/'
    data_top_dir = raw_data_top_dir
    missing_scan_id = False
    print(data_top_dir, scan_id)
    filenames = glob.glob(os.path.join(data_top_dir, '*{0:d}.h5'.format(scan_id)))

    if filenames is []:
        print ('!!!!!Error!!!!! Data file with scan_id {0:d} does not exist in the given path.'.format(scan_id))
        print ('Please check your input scan ids. Quit.')
        exit()
        missing_scan_id = True
    data_files = filenames
    print(filenames)
    # output_files = data_top_dir+'/recon_'+os.path.basename(filenames[0]).split(".")[-2]+'/recon_'+os.path.basename(filenames[0]).split(".")[-2])
    output_files = os.path.join(data_top_dir, 'recon_'+os.path.basename(filenames[0]).split(".")[-2], 'recon_'+os.path.basename(filenames[0]).split(".")[-2])

    print(data_files, output_files)
    return data_files[0], output_files


def sortFiltersOrder(**kwarg):
    seq = []
    use = []
    filterList = ['denoise_filter',
                  'remove_stripe_fwFilter',
                  'remove_stripe_tiFilter',
                  'remove_stripe_sfFilter',
                  'remove_stripe_voFilter',
                  'retrieve_phaseFilter',
                  'normalize_bgFilter',
                  'remove_cuppingFilter']
    if kwarg['denoise_filter']['use']['use'] is 'no':
        use.append(0)
        seq.append(-1)
    else:
        use.append(1)
        seq.append(kwarg['denoise_filter']['use']['order'])

    if kwarg['remove_stripe_fwFilter']['use']['use'] is 'no':
        use.append(0)
        seq.append(-1)
    else:
        use.append(1)
        seq.append(kwarg['remove_stripe_fwFilter']['use']['order'])

    if kwarg['remove_stripe_tiFilter']['use']['use'] is 'no':
        use.append(0)
        seq.append(-1)
    else:
        use.append(1)
        seq.append(kwarg['remove_stripe_tiFilter']['use']['order'])

    if kwarg['remove_stripe_sfFilter']['use']['use'] is 'no':
        use.append(0)
        seq.append(-1)
    else:
        use.append(1)
        seq.append(kwarg['remove_stripe_sfFilter']['use']['order'])

    if kwarg['remove_stripe_voFilter']['use']['use'] is 'no':
        use.append(0)
        seq.append(-1)
    else:
        use.append(1)
        seq.append(kwarg['remove_stripe_voFilter']['use']['order'])

    if kwarg['retrieve_phaseFilter']['use']['use'] is 'no':
        use.append(0)
        seq.append(-1)
    else:
        use.append(1)
        seq.append(kwarg['retrieve_phaseFilter']['use']['order'])

    if kwarg['normalize_bgFilter']['use']['use'] is 'no':
        use.append(0)
        seq.append(-1)
    else:
        use.append(1)
        seq.append(kwarg['normalize_bgFilter']['use']['order'])

    if kwarg['remove_cuppingFilter']['use']['use'] is 'no':
        use.append(0)
        seq.append(-1)
    else:
        use.append(1)
        seq.append(kwarg['remove_cuppingFilter']['use']['order'])

    if sum(use) != max(seq):
        if sum(use)!=0 and max(seq)!=-1:
            print ('!!!!! Info !!!!!')
            print ('Filter ordering is wrong. Aborted.')
            print (use,seq)
            print ('sum(use)=',sum(use),'; max(seq)=',max(seq))
            exit()
    elif np.sum(np.multiply(seq,use)) != int((1+max(seq))*np.sum(use)/2.0):
        print ('!!!!! Info !!!!!')
        print ('Filter ordering is wrong. Aborted')
        print (use,seq)
        print ('np.sum(np.multiply(seq*use))=',np.sum(np.multiply(seq,use)),'; int((1+max(seq))*np.sum(use)/2.0)=',int((1+max(seq))*np.sum(use)/2.0))
        exit()

    return filterList,seq


def runFilter(data,fltname,**kwargs):
    if fltname is 'denoise_filter':
        params = kwargs['denoise_filter']['params']['filter_params'][kwargs['denoise_filter']['params']['filter_name']]
        if kwargs['denoise_filter']['params']['use_est_sigma'] is True:
            if kwargs['denoise_filter']['params']['filter_name'] in ['denoise_nl_means', 'denoise_wavelet']:
                sigma = skr.estimate_sigma(data[0])
            if kwargs['denoise_filter']['params']['filter_name'] == 'denoise_nl_means':
                params['sigma'] = sigma
                params['h'] = 0.9 * sigma
            elif kwargs['denoise_filter']['params']['filter_name'] == 'denoise_wavelet':
                params['sigma'] = sigma
        if kwargs['denoise_filter']['params']['filter_name'] in ['wiener', 'unsupervised_wiener']:
            if kwargs['denoise_filter']['params']['psf_reset_flag'] is False:
                psfw = params['psf']
                params['psf'] = np.ones([psfw, psfw])/(psfw**2)
                kwargs['denoise_filter']['params']['psf_reset_flag'] = True

        if kwargs['denoise_filter']['params']['filter_name'] == 'wiener':
            #print('test without using wiener')
            for ii in range(data.shape[0]):
                data[ii] = skr.wiener(data[ii], **params)

        elif kwargs['denoise_filter']['params']['filter_name'] == 'unsupervised_wiener':
            for ii in range(data.shape[0]):
                # print(skr.unsupervised_wiener(data[ii], **params).shape)
                data[ii], _ = skr.unsupervised_wiener(data[ii], **params)
        elif kwargs['denoise_filter']['params']['filter_name'] == 'denoise_nl_means':
            for ii in range(data.shape[0]):
                data[ii] = skr.denoise_nl_means(data[ii], **params)
        elif kwargs['denoise_filter']['params']['filter_name'] == 'denoise_tv_bregman':
            for ii in range(data.shape[0]):
                data[ii] = skr.denoise_tv_bregman(data[ii], **params)
        elif kwargs['denoise_filter']['params']['filter_name'] == 'denoise_tv_chambolle':
            for ii in range(data.shape[0]):
                data[ii] = skr.denoise_tv_chambolle(data[ii], **params)
        elif kwargs['denoise_filter']['params']['filter_name'] == 'denoise_bilateral':
            for ii in range(data.shape[0]):
                data[ii] = skr.denoise_bilateral(data[ii], **params)
        elif kwargs['denoise_filter']['params']['filter_name'] == 'denoise_wavelet':
            for ii in range(data.shape[0]):
                data[ii] = skr.denoise_wavelet(data[ii], **params)
        print ('wiener denoise is done')

    elif fltname is 'remove_stripe_fwFilter':
        params = kwargs['remove_stripe_fwFilter']['params']
        data = tomopy.prep.stripe.remove_stripe_fw(data,**params)
        print ('remove_stripe_fw is done')
#             tomopy.write_tiff(data,fname='/local/data/2017_10/Jinfang/debug/after_remove_stripe_fw.tiff')
    elif fltname is 'remove_stripe_tiFilter':
        params = kwargs['remove_stripe_tiFilter']['params']
        data = tomopy.prep.stripe.remove_stripe_ti(data,**params)
        print ('remove_stripe_ti is done')
    elif fltname is 'remove_stripe_sfFilter':
        params = kwargs['remove_stripe_sfFilter']['params']
        data = tomopy.prep.stripe.remove_stripe_sf(data,**params)
        print ('remove_stripe_sf is done')
    elif fltname is 'remove_stripe_voFilter':
        params = kwargs['remove_stripe_voFilter']['params']
        data = tomopy.prep.stripe.remove_all_stripe(data,**params)
        print ('remove_stripe_vo is done')
    elif fltname is 'retrieve_phaseFilter':
        params = kwargs['retrieve_phaseFilter']['params']
        print (params)
        if params['filter'] is 'paganino':
            print ('paganino is here')
            #del params['pr_filteryes']
            data = tomopy.prep.phase.retrieve_phase(data,**params)
        elif params['filter'] is 'paganin':
            print ('paganin is here')
            #del params['pr_filter']
            data = tomopy.prep.phase.retrieve_phase(data,**params)
        elif params['filter'] == 'bronnikov':
#            data = data - 1
            print ('bronnikov is here')
            #del params['pr_filter']
            data = 1 - data
            data = tomopy.prep.phase.retrieve_phase(data,**params)
        elif params['filter'] == 'fba':
            print ('fba is here')
            #del params['pr_filter']
#            data[:] = (data[:] - 1)/2
            data = (1 - data)/2
            data = tomopy.prep.phase.retrieve_phase(data,**params)
        else:
            print ('wrong phase retrieval option.')
            exit()
        print ('retrieve_phase is done...')
#             tomopy.write_tiff(data,fname='/local/data/2017_10/Jinfang/debug/after_retrieve_phase.tiff')
    elif fltname is 'normalize_bgFilter':
        params = kwargs['normalize_bgFilter']['params']
        data = tomopy.prep.normalize.normalize_bg(data,**params)
        print ('normalize_bg is done')
    elif fltname is 'remove_cuppingFilter':
        print ('remove_cuppingFilter is done')
    else:
        print ('!!!!! Info !!!!!')
        print ('Filter ',fltname, ' currently is not implemented in the script. Aborted.')
        exit()

    return data

def ifLog(**kwargs):
    pr_use = kwargs['retrieve_phaseFilter']['use']['use']
    pr_flt = kwargs['retrieve_phaseFilter']['params']['filter']
    if pr_use is 'yes':
        if pr_flt is 'paganino':
            logDo = 'yes'
        elif pr_flt is 'paganin':
            logDo = 'yes'
        elif pr_flt is 'bronnikov':
            logDo = 'no'
        elif pr_flt is 'fba':
            logDo = 'no'
    else:
        logDo = 'yes'

    return logDo


def generalFilterContainer(data,**kwargs):
    """
       kwargs: kwargs using format of filternameParams. For instance, to use filter
               remove_stripe_sf, you need to provide a kwarg
               remove_stripe_sfParams = {'use':'yes','size':31}
               By default, this routine assume five filters
               1. retrieve_phase
               2. remove_stripe_fw
               3. remove_stripe_ti
               4. remove_stripe_sf
               5. normalize_bg

               in the __main__ function below, this functions uses all five filters.
               You can set 'use':'no' in a filter kwargs to disable that filter. You
               can also include more filters in this function in the same format per
               your purposes.
    """
    print ('start filtering')
#    tomopy.write_tiff(data,fname='/local/data/2017_10/Jinfang/debug/data.tiff')

    filterList,seq = sortFiltersOrder(**kwargs)
#    print('filter seq = {}'.format(seq))

    if max(seq)==-1:
        pass
    else:
        for ii in range(max(seq)):
            data = runFilter(data,filterList[seq.index(ii+1)],**kwargs)
            data[data<0] = 1

    logDo = ifLog(**kwargs)
    print ('minus_log: ',logDo)
    if logDo  == 'yes':
        data = tomopy.prep.normalize.minus_log(data)

    if kwargs['remove_cuppingFilter']['use']['use'] is 'yes':
        print(kwargs['remove_cuppingFilter']['params']['cc'])
        data -= kwargs['remove_cuppingFilter']['params']['cc']

    return data




def manualFindCenter(center_shift,center_shift_w,**kwargs):
    algorithm = kwargs['ExplicitParams']['algorithm']
    filter_name = kwargs['ExplicitParams']['filter_name']
    sliceStart = kwargs['ExplicitParams']['sliceStart']
    sliceEnd = kwargs['ExplicitParams']['sliceEnd']
    zinger = kwargs['ExplicitParams']['zinger']
    zinger_level = kwargs['ExplicitParams']['zinger_level']
    mask = kwargs['ExplicitParams']['mask']
    mask_ratio = kwargs['ExplicitParams']['mask_ratio']
    wedge = kwargs['wedgeData']['wedge']
    logging = kwargs['ExplicitParams']['logging']

    filename = kwargs['fileParams']['data_files']
    data_center_path = kwargs['fileParams']['data_center_path']
    flat_name = kwargs['fileParams']['flat_name']
    dark_name = kwargs['fileParams']['dark_name']
    fake_flat = kwargs['fileParams']['fake_flat']
    fake_dark = kwargs['fileParams']['fake_dark']
    fake_flat_val = kwargs['fileParams']['fake_flat_val']
    fake_dark_val = kwargs['fileParams']['fake_dark_val']

    ds_use = kwargs['downsample_Filter']['use']['use']
    if ds_use.upper() == 'YES':
        ds_levels = kwargs['downsample_Filter']['params']['level']
    else:
        ds_levels = None

    data,white,dark,theta = dataStandardReader(filename,sliceStart=sliceStart,sliceEnd=sliceEnd,
                                               flat_name=flat_name, dark_name=dark_name,
                                               fake_flat=fake_flat, fake_dark=fake_dark,
                                               fake_flat_val=fake_flat_val, fake_dark_val=fake_dark_val,
                                               ds_levels=ds_levels)
    theta = theta * np.pi/180.0
    dim = data.shape
    numProj = dim[0]
    numSlices = dim[1]
    widthImg = dim[2]
    print (dim)
    print (theta.shape)
#    theta = np.linspace(0,np.pi,num=numProj+1)
    print ('data is read')

#    # remove zingers (pixels with abnormal counts)
    if zinger != False or zinger != None:
        data = tomopy.misc.corr.remove_outlier(data,zinger_level,size=15,axis=0)
        white = tomopy.misc.corr.remove_outlier(white,zinger_level,size=15,axis=0)
        print  ('remove outlier is done')

    # normalize projection images; for now you need to do below two operations in sequence
    white = filters.gaussian_filter(white,sigma=10)
    data = tomopy.prep.normalize.normalize(data,white,dark)
    print ('normalization is done')
#    tifffile.imsave('/NSLS2/xf18id1/users/2019Q2/debug/norm_data.tiff',data)

#    data = filters.gaussian_laplace(data,5)
    data = generalFilterContainer(data,**kwargs)
    if data_center_path==None:
        data_center_path = '~/tomopy_data_center'

#    tifffile.imsave('/NSLS2/xf18id1/users/2019Q2/debug/after_filter.tiff',data)

    if wedge is True:
        print ('Projection images are in less than 180 degree range.')
        data = sortWedge(data, sliceStart, sliceEnd, **kwargs)
        write_center(data[:,int(numSlices/2)-1:int(numSlices/2)+1,:], theta, dpath=data_center_path,
                     cen_range=(data.shape[2]/2+center_shift,data.shape[2]/2+center_shift+center_shift_w,0.5),
                     mask = mask, ratio = mask_ratio, algorithm = algorithm, filter_name = filter_name)
    elif wedge is False:
        print ('wedge = None')
        print (data_center_path)
        write_center(data[:,int(numSlices/2)-1:int(numSlices/2)+1,:], theta, dpath=data_center_path,
                 cen_range=(data.shape[2]/2+center_shift,data.shape[2]/2+center_shift+center_shift_w,0.5),
                 mask = mask, ratio = mask_ratio, algorithm = algorithm, filter_name = filter_name)
    else:
        print ('Wrong wedge setting. Aborted')
        exit()

    if logging is True:
        fout = os.path.join(os.path.dirname(filename), ''.join(os.path.basename(filename).split('.')[:-1]) +\
                            '_finding_cneter_log.txt')
        fo = open(fout, "w")
        for k, v in kwargs.items():
            fo.write(str(k) + ': ' + str(v) + '\n\n')
        fo.close()





def reconEngine(**kwargs):
    state = 1

    filename = kwargs['fileParams']['data_files']
    output_file = kwargs['fileParams']['output_files']
    print('output_file:', output_file)
    flat_name = kwargs['fileParams']['flat_name']
    dark_name = kwargs['fileParams']['dark_name']
    fake_flat = kwargs['fileParams']['fake_flat']
    fake_dark = kwargs['fileParams']['fake_dark']
    fake_flat_val = kwargs['fileParams']['fake_flat_val']
    fake_dark_val = kwargs['fileParams']['fake_dark_val']

    algorithm = kwargs['ExplicitParams']['algorithm']
    filter_name = kwargs['ExplicitParams']['filter_name']
    center = kwargs['ExplicitParams']['center']
    zinger = kwargs['ExplicitParams']['zinger']
    zinger_level = kwargs['ExplicitParams']['zinger_level']
    offset = kwargs['ExplicitParams']['offset']
    chunk_size = kwargs['ExplicitParams']['chunk_size']
    numRecSlices = kwargs['ExplicitParams']['numRecSlices']
    margin_slices = kwargs['ExplicitParams']['margin_slices']
    mask = kwargs['ExplicitParams']['mask']
    mask_ratio = kwargs['ExplicitParams']['mask_ratio']
    wedge = kwargs['wedgeData']['wedge']
    logging = kwargs['ExplicitParams']['logging']

    ds_use =  kwargs['downsample_Filter']['use']['use']
    if ds_use.upper() == 'YES':
        ds_levels = kwargs['downsample_Filter']['params']['level']
    else:
        ds_levels = None

    dim = dataInfo(filename)
    numSlices = dim[1]
    widthImg = dim[2]

    if offset == None:
        offset = 0
    if numRecSlices == None:
        numRecSlices = dim[1]

    if chunk_size > numRecSlices:
        chunk_size = numRecSlices
    num_chunk = np.int(numRecSlices/(chunk_size-margin_slices)) + 1
    if numRecSlices == chunk_size:
        num_chunk = 1

    for ii in range(num_chunk):
        print ('chunk ',ii, ' reconstruction starts')
        print (time.asctime())

        if ii == 0:
            sliceStart = offset + ii*chunk_size
            sliceEnd = offset + (ii+1)*chunk_size
        else:
            sliceStart = offset + ii*(chunk_size-margin_slices)
            sliceEnd = sliceStart + chunk_size
            if sliceEnd > (offset+numRecSlices):
                sliceEnd = offset+numRecSlices
            if sliceEnd > numSlices:
                sliceEnd = numSlices

        if (sliceEnd - sliceStart) <= margin_slices:
            print ('Reconstruction finishes!')
            break

        data,white,dark,theta = dataStandardReader(filename,sliceStart=sliceStart,sliceEnd=sliceEnd,
                                                   flat_name=flat_name, dark_name=dark_name,
                                                   fake_flat=fake_flat, fake_dark=fake_dark,
                                                   fake_flat_val=fake_flat_val, fake_dark_val=fake_dark_val,
                                                   ds_levels=ds_levels)
        theta= theta*np.pi/180

        data_size = data.shape
#        theta = np.linspace(0,np.pi,num=data_size[0]+1)
        print ('data is read')

    #    # remove zingers (pixels with abnormal counts)
        if zinger == True:
            data = tomopy.misc.corr.remove_outlier(data,zinger_level,size=15,axis=0)
            white = tomopy.misc.corr.remove_outlier(white,zinger_level,size=15,axis=0)
            print  ('remove outlier is done')

        # normalize projection images; for now you need to do below two operations in sequence
        data = tomopy.prep.normalize.normalize(data,white,dark)
        print ('normalization is done')

        data = generalFilterContainer(data,**kwargs)

        if ii == 0 and (center == False or center == None):
            center = tomopy.find_center_vo(data)

        if wedge is True:
            print ('Projection images are in less than 180 degree range.')
            data = sortWedge(data, sliceStart, sliceEnd, **kwargs)
        elif wedge is False:
            pass
        else:
            print ('Wrong wedge setting. Aborted')
            exit()

        # tomo reconstruction
        data_recon = recon(data,theta,center=center,
                           algorithm = algorithm, filter_name = filter_name)
        print ('reconstruction is done')

        if mask == True:
            data_recon = tomopy.circ_mask(data_recon, 0, ratio=mask_ratio)

        # save reconstructions
        dxchange.writer.write_tiff_stack(data_recon[np.int(margin_slices/2):(sliceEnd-sliceStart-np.int(margin_slices/2)),:,:],
                                                     axis = 0,
                                                     fname = output_file,
                                                     start = sliceStart+np.int(margin_slices/2),
                                                     overwrite = True)

        del(data)
        del(white)
        del(dark)
        del(theta)
        del(data_recon)
        gc.collect()
        print ('chunk ',ii, ' reconstruction is saved')
        print (time.asctime())


    if state == 1:
        print ('Reconstruction finishes!')
        if logging is True:
            fout = os.path.join(os.path.dirname(filename), ''.join(os.path.basename(filename).split('.')[:-1]) +\
                                    '_finding_cneter_log.txt')
            fout = os.path.join(Path(output_file).parents[0], ''.join(os.path.basename(filename).split('.')[:-1]) +\
                                '_recon_log.txt')
            fo = open(fout, "w")
            for k, v in kwargs.items():
                fo.write(str(k) + ': ' + str(v) + '\n\n')
            fo.close()
    else:
        print ('Reconstruction is terminated due to data file error.')





if __name__ == '__main__':
############################### user input section -- Start ###############################
##### This is the only section which you need to edit
    # This is the path in which ExpXXX locate.
    # raw_data_top_dir = "/media/XIAOUSB2/BNL_FXI_June_2019"

    raw_data_top_dir = "/NSLS2/xf18id1/users/2020Q1/ENYUAN_Proposal_305409/"
    #raw_data_top_dir = "/NSLS2/xf18id1/users/2019Q3/XIANGHUI_XIAO_Proposal_305570"
    # keep [] even you only do reconstruction of one data set; put all data set Exp indices that you want to reconstruct in [];
    # if you want to reconstruction all data sets in a range, e.g. with indices from 1 to 10, you can use scan_idx = np.arrange(1,10,1);
    # if you want to reconstruction one every two, you can use scan_idx = np.arange(1,10,2)
#    scan_idx = [24567, 24568, 24569, 24570, 24571, 24572, 24573, 24574, 24575, 24576, 24577, 24578]
#    scan_idx = [24626, 24627, 24628, 24629, 24630, 24631, 24632, 24633, 24634, 24635, 24636, 24637, 24638, 24639, 24640, 24641, 24642]
    scan_idx = [29671]
    #center_list = [1326.0]
#    center_list = [1370.00, 1377.00, 1375.00, 1366.00, 1361.50, 1384.50, 1390.50, 1377.50, 1379.50, 1373.50, 1373.50, 1379.50, 1366.00, 1370.50, 1362.00, 1378.00, 1379.00]
    center_list = [642.5] #663 is 30059
    #center_list = [1376.5, 1336, 1344]                  # If you find center values for the list of data sets defined in scan_idx above, you put
                                            # these center values here in []. The number of values here in [] must be
                                            # equal to number of items in scan_idx = [], and orders in these two [] must match.

    flat_name = None
    dark_name = None
    fake_flat = False
    fake_dark = False
    fake_flat_val = 15e4
    fake_dark_val = 100.

    # if recon_top_dir = None
    #   save the recon in a folder in the sample directory level as the raw data file
    # else:
    #   save the recon into a folder in the specified directory
    recon_top_dir = None #'/media/KarenUSB/BNL_FXI_Dec05_2019'


    # manualCenter = True will turn on trial finding center routine. With this option, you can not only find center position but also
    # test filter combination and parameters quickly. When manualCenter = True, you may need to adjust parameter in the below if statements.
    # manualCenter = False will reconstruct whole volume in a data set with the center value calculated with automatic center finding
    # routine. It may output wrong center whose accuracy may vary from data set to data set.
    if len(sys.argv) == 1:
        manualCenter = True
    elif len(sys.argv) == 2:
        manualCenter = False
    elif len(sys.argv) == 3:
        manualCenter = False
        scan_idx, center_list = read_center(sys.argv[2])
        print(manualCenter)
    #manualCenter = False
    logging = True
    algorithm = "gridrec"
    filter_name = "shepp"

    # configure for manual finding center
    if manualCenter == True:
        center_shift = -80  #-10
        center_shift_w = 80   # 50
        sliceStart = 500 #1580

        #sliceStart = 55       #1580
        sliceEnd = sliceStart + 20    # 20
        #data_center_path = os.path.join(os.path.abspath(os.path.join(raw_data_top_dir,'..')),'data_center')
        data_center_path = os.path.join(raw_data_top_dir,'data_center')

        #print(f'\n\n\n{data_center_path}')
#        data_center_path = "/media/XIAOUSB1/BNL_FXI_June2019/data_center"

    # configure for volume reconstruction
    offset = 100         # set to None if you want to reconstruct the whole volume
    numRecSlices = 800    # set to None if you want to reconstruct the whole volume
    chunk_size = 200        # this number is determined by available RAM in your computer; 300 is good for a computer with at least 128GB RAM
    margin_slices = 30      # leave this fixed
    zinger = False
    zinger_level = 500      # You may need to change this is you see some artifacts in reconstructed slice images
    mask = True             # You set 'mask' to be True if you like mask out four corners in the reconsructed slice images
    mask_ratio = 1          # Ratio of the mask's diameter in pixels to slice image width

#    center_list = [30478:1313.5, 30480:1294.5, 30949:1247.5]
    # define wedge data processing parameters
    BlankAt = 90
    first_missing_end = None
    second_missing_start = None
    missing_start = None
    missing_end = None
    bad_angs = None

    if manualCenter:
        sli_start = sliceStart
        sli_end = sliceEnd
    else:
        if offset is None:
            sli_start = 0
        else:
            sli_start = offset
        if numRecSlices is not None:
            sli_end = offset + numRecSlices
        else:
            sli_end = numRecSlices

    thres = 500
    wedge = True
    fn = "/NSLS2/xf18id1/users/2020Q1/ENYUAN_Proposal_305409/fly_scan_id_29721.h5"
    if wedge:
        bad_angs = trt.get_dim_angle_range_slice_range(fn, sli_start, sli_end=sli_end, thres=thres, block_view_at=BlankAt)

    if wedge == True:
        if BlankAt == 0:
            first_missing_end = 100  # 180
            second_missing_start = 700  #1360
            missing_start = None
            missing_end = None
        elif BlankAt == 90:
            missing_start = 340
            missing_end = 390
            first_missing_end = None
            second_missing_start = None

    wedgeParams = {'first_missing_end':first_missing_end,
                    'second_missing_start':second_missing_start,
                    'missing_start':missing_start,
                    'missing_end':missing_end,
                    'BlankAt':BlankAt,
                    'bad_angs':bad_angs}

    # define parameters for various filters
    denoise_filter = {'use_est_sigma': True,
                      'filter_name': 'wiener',
                      'psf_reset_flag': False,
                      'filter_params': {'wiener': {'psf': 2,
                                  'balance': 0.3,
                                  'reg': None,
                                  'is_real': True,
                                  'clip': True},

                                  'unsupervised_wiener': {'psf': 2,
                                  'balance': 0.3,
                                  'reg': None,
                                  'is_real': True,#True
                                  'clip': True},#True

                                  'denoise_nl_means': {'patch_size': 5,
                                  'patch_distance': 7,
                                  'h': 0.1,
                                  'multichannel': False,
                                  'fast_mode': True,
                                  'sigma': 0.05},

                                  'denoise_tv_bregman': {'weight': 1.0,
                                  'max_iter': 100,
                                  'eps': 0.001,
                                  'isotropic': True},

                                  'denoise_tv_chambolle': {'weight': 0.1,
                                  'eps': 0.0002,
                                  'max_iter': 200,
                                  'multichannel': False},

                                  'denoise_bilateral': {'win_size': None,
                                  'sigma_color': None,
                                  'sigma_spatial': 1,
                                  'bins': 10000,
                                  'mode': 'constant',
                                  'cval': 0,
                                  'multichannel': False},

                                  'denoise_wavelet': {'sigma': 1,
                                  'wavelet': 'db1',
                                  'mode': 'soft',
                                  'wavelet_levels': 3,
                                  'multichannel': False,
                                  'convert2ycbcr': False,
                                  'method': 'BayesShrink'}
                                  }
                        }

    downsample_Params = {'level': (0.5, 0.5, 0.5)}

    retrieve_phaseParams = {'filter': 'paganin',
                            'pixel_size': 0.65e-4,
                            'dist': 15,
                            'energy': 35.0,
                            'alpha': 1e-2,
                            'pad':True}

    remove_stripe_tiParams = {'nblock': 0,
                              'alpha': 5}

    remove_stripe_fwParams = {'level': 9,
                              'wname': 'db5',
                              'sigma': 2,
                              'pad':True}


    remove_stripe_sfParams = {'size': 31}

    remove_stripe_voParams = {'snr':3,
                              'la_size': 81,
                              'sm_size': 21}

    normalize_bgParams = {'air': 30}

    remove_cuppingParams = {'cc': 0.0}


### define filter combination
    use_stripe_removal_fw = {'use':'no',
                             'order':1}

    use_stripe_removal_ti = {'use':'no',
                             'order':None}

    use_stripe_removal_sf = {'use':'no',
                             'order':1}

    use_stripe_removal_vo = {'use':'yes',
                             'order':1} #yes

    use_normalize_bg =      {'use':'no',
                             'order':3}

    use_retrieve_phase =    {'use':'no',
                             'order':2}

    use_denoise_filter =    {'use':'no',
                             'order':2}

    use_remove_cupping =    {'use':'no',
                             'order':3}

    use_downsample =        {'use':'no'}





############################### user input section -- End ###############################

    for jj in range(len(scan_idx)):
        if manualCenter == True:
            if len(scan_idx) == 1:
                data_files, output_files = getFiles(raw_data_top_dir, scan_idx[0])
                if recon_top_dir is not None:
                    output_files = os.path.join(recon_top_dir, output_files.split('/')[-2], output_files.split('/')[-1])
                if data_files == 0:
                    print ('!!!!! Error !!!!!')
                    print ('There is no given file in the path. Reconstructon is aborted.')
                    exit()

                dim = dataInfo(data_files, showInfor=True)
                if dim == 0:
                    print ('!!!!! Error !!!!!')
                    print ('Cannot read the given data file. Reconstruction is aborted.')
                    exit()

                print('Finding center manually')

                if wedge is True:
                    if BlankAt == 0 and (first_missing_end == None or second_missing_start == None):
                        print('You need to define first_missing_end and second_missing_start.')
                        exit()
                    elif BlankAt == 90 and (missing_start == None or missing_end == None):
                        print('You need to define missing_start and missing_end.')
                        exit()

                loopEngineParams = {'ExplicitParams':
                                        {'algorithm': algorithm,
                                         'filter_name': filter_name,
                                         'sliceStart': sliceStart,
                                         'sliceEnd': sliceEnd,
                                         'zinger': zinger,
                                         'zinger_level': zinger_level,
                                         'mask': mask,
                                         'mask_ratio': mask_ratio,
                                         'logging': logging},

                                    'fileParams':
                                        {'data_files': data_files,
                                         'data_center_path': data_center_path,
                                         'flat_name': flat_name,
                                         'dark_name': dark_name,
                                         'fake_flat': fake_flat,
                                         'fake_dark': fake_dark,
                                         'fake_flat_val': fake_flat_val,
                                         'fake_dark_val': fake_dark_val},
                                    'wedgeData':
                                        {'wedge': wedge,
                                         'wedgeParams': wedgeParams},

                                    'denoise_filter':
                                        {'use': use_denoise_filter,
                                         'params': denoise_filter},

                                    'remove_stripe_fwFilter':
                                        {'use': use_stripe_removal_fw,
                                         'params': remove_stripe_fwParams},

                                    'retrieve_phaseFilter':
                                        {'use': use_retrieve_phase,
                                         'params': retrieve_phaseParams},

                                    'remove_stripe_tiFilter':
                                        {'use': use_stripe_removal_ti,
                                         'params': remove_stripe_tiParams},

                                    'remove_stripe_sfFilter':
                                        {'use': use_stripe_removal_sf,
                                         'params': remove_stripe_sfParams},

                                    'remove_stripe_voFilter':
                                        {'use': use_stripe_removal_vo,
                                         'params': remove_stripe_voParams},

                                    'normalize_bgFilter':
                                        {'use': use_normalize_bg,
                                         'params': normalize_bgParams},

                                    'remove_cuppingFilter':
                                        {'use': use_remove_cupping,
                                         'params': remove_cuppingParams},

                                    'downsample_Filter':
                                        {'use': use_downsample,
                                         'params': downsample_Params}
                                    }

                manualFindCenter(center_shift, center_shift_w, **loopEngineParams)
            else:
                print ('!!!!! Error !!!!!')
                print ('Finding center manually requires only one data set. Please either change manualCenter to False or provide only one data set.')

        else:
            print('You are doing volume reconstructions...')
            data_files, output_files = getFiles(raw_data_top_dir, scan_idx[jj])
            if recon_top_dir is not None:
                    output_files = os.path.join(recon_top_dir, output_files.split('/')[-2], output_files.split('/')[-1])
            if data_files == 0:
                print ('!!!!! Error !!!!!')
                print ('There is no given file in the path. Reconstructon is aborted.')
                exit()
            print(data_files)
            print(data_files[0])
            dim = dataInfo(data_files, showInfor=True) ### earlier it was data_files[0] within brackets, didn't work
            if dim == 0:
                print ('!!!!! Error !!!!!')
                print ('Cannot read the given data file. Reconstruction is aborted.')
                exit()

            loopEngineParams = {'ExplicitParams':
                                                {'algorithm': algorithm,
                                                 'filter_name': filter_name,
                                                 'center':center_list[jj],
                                                 'zinger':zinger,
                                                 'zinger_level':zinger_level,
                                                 'offset':offset,
                                                 'chunk_size':chunk_size,
                                                 'numRecSlices':numRecSlices,
                                                 'margin_slices':margin_slices,
                                                 'mask':mask,
                                                 'mask_ratio':mask_ratio,
                                                 'logging': logging},

                                    'fileParams':
                                                {'data_files': data_files,
                                                 'output_files': output_files,
                                                 'flat_name': flat_name,
                                                 'dark_name': dark_name,
                                                 'fake_flat': fake_flat,
                                                 'fake_dark': fake_dark,
                                                 'fake_flat_val': fake_flat_val,
                                                 'fake_dark_val': fake_dark_val},

                                     'wedgeData':
                                                {'wedge':wedge,
                                                 'wedgeParams':wedgeParams},

                          'denoise_filter':
                                                {'use': use_denoise_filter,
                                                 'params': denoise_filter},

                        'remove_stripe_fwFilter':
                                                {'use':use_stripe_removal_fw,
                                                 'params':remove_stripe_fwParams},

                          'retrieve_phaseFilter':
                                                {'use':use_retrieve_phase,
                                                 'params':retrieve_phaseParams},

                        'remove_stripe_tiFilter':
                                                {'use':use_stripe_removal_ti,
                                                 'params':remove_stripe_tiParams},

                        'remove_stripe_sfFilter':
                                                {'use':use_stripe_removal_sf,
                                                 'params':remove_stripe_sfParams},
                        'remove_stripe_voFilter':
                                                {'use': use_stripe_removal_vo,
                                                 'params': remove_stripe_voParams},

                            'normalize_bgFilter':
                                                {'use':use_normalize_bg,
                                                 'params':normalize_bgParams},

                          'remove_cuppingFilter':
                                                {'use': use_remove_cupping,
                                                 'params': remove_cuppingParams},

                             'downsample_Filter':
                                                {'use': use_downsample,
                                                 'params': downsample_Params}
                              }

            reconEngine(**loopEngineParams)













