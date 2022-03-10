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
from tomopy import minus_log
import os, h5py, glob, fnmatch
import numpy as np
from scipy import misc
import time
import skimage.restoration as skr
import skimage.morphology as skm
import tifffile
import matplotlib.pylab as mpp

def gegenerateData(data, ind_range):
    return data[range(ind_range)]

def saveFile(data, filename):
    tifffile.imsave(data, filename)


def genSheppPhantom(sli=None):
    data = tomopy.shepp3d()

    if sli is not None:
        return data[sli[0]:sli[1], :, :]
    else:
        return data


def genSheppProjData(sli=None, num_angles=180):
    data = tomopy.shepp3d()
    theta = tomopy.angles(num_angles)
    data = tomopy.project(data, theta)

    if sli is not None:
        return data[:, sli[0]:sli[1], :], theta
    else:
        return data, theta


def genSheppObjSupp(sli=None, dilation=None):
    if dilation is None:
        data = genSheppPhantom(sli=sli)
        supp = (data != 0)
    else:
        data = genSheppPhantom(sli=sli)
        supp = (data != 0)
        for ii in range(supp.shape[0]):
            supp[ii, :] = skm.binary_dilation(supp[ii, :], selem=skm.disk(dilation))
    return supp


def genSheppProjSupp(wedgeStart, wedgeEnd, sli=None, dilation=None):
    if dilation is None:
        data, theta = genSheppProjData(sli=sli)
        # data, theta = genWedgeProjData(data, wedgeStart, wedgeEnd, reduce=False)
        supp = (data > 0)
    else:
        data, theta = genSheppProjData(sli=sli)
        # data, theta = genWedgeProjData(data, wedgeStart, wedgeEnd, reduce=False)
        supp = (data > 0)
        for ii in range(supp.shape[0]):
            supp[ii, :] = skm.binary_dilation(supp[ii, :], selem=skm.disk(dilation))
    return supp


def genSheppPhantom(sli=None):
    data = tomopy.shepp3d()

    if sli is not None:
        return data[sli[0]:sli[1], :, :]
    else:
        return data


def genProjData(obj, num_angles=180):
    theta = tomopy.angles(num_angles)
    data = tomopy.project(obj, theta)

    return data, theta


def genObjSupp(obj, dilation=None):
    if dilation is None:
        supp = (obj != 0)
    else:
        supp = (obj != 0)
        for ii in range(supp.shape[0]):
            supp[ii, :] = skm.binary_dilation(supp[ii, :], selem=skm.disk(dilation))
    return supp


def genProjSupp(proj, wedgeStart, wedgeEnd, dilation=None):
    if dilation is None:
        supp = (proj > 0)
        supp[wedgeStart:wedgeEnd, :] = 0
    else:
        supp = (proj > 0)
        for ii in range(supp.shape[0]):
            supp[ii, :] = skm.binary_dilation(supp[ii, :], selem=skm.disk(dilation))
        supp[wedgeStart:wedgeEnd, :] = 0
    return supp


def genWedgeProjData(full_proj, wedgeStart, wedgeEnd, reduce=False, sli=None):
    theta = np.linspace(0, np.pi, num=full_proj.shape[0] + 1)
    #    print(full_proj.shape)
    if reduce:
        wedge_data = np.delete(full_proj, np.arange(wedgeStart, wedgeEnd), axis=0)
        wedge_theta = np.delete(theta, np.arange(wedgeStart, wedgeEnd), axis=0)
        if sli:
            return wedge_data[:, sli[0]:sli[1], :], wedge_theta
        else:
            return wedge_data, wedge_theta
    else:
        full_proj[wedgeStart:wedgeEnd, :] = 0
        if sli:
            return full_proj[:, sli[0]:sli[1], :], theta
        else:
            return full_proj, theta


def patchWedgeProjData(full_proj, wedgeStart, wedgeEnd, avg_width, roll_dist=0):
    missing_width = wedgeEnd - wedgeStart
    cos_phi = np.cos((missing_width + avg_width) * np.pi / full_proj.shape[0])
    theta = np.linspace(0, np.pi, num=full_proj.shape[0] + 1)
    for ii in range(wedgeStart, wedgeEnd):
        full_proj[ii, :, :] = np.roll(
            np.mean(np.roll(full_proj[(ii - missing_width - avg_width):(ii - missing_width), :, :], roll_dist, axis=2),
                    axis=0) + \
            np.mean(np.roll(full_proj[(ii + missing_width):(ii + missing_width + avg_width), :, :], roll_dist, axis=2),
                    axis=0), -1 * roll_dist) / 2.0 / cos_phi
    return full_proj, theta


def readProjData(proj_data, theta, sli=None):
    if sli is not None:
        return proj_data[:, sli[0]:sli[1], :], theta[sli[0]:sli[1]]
    else:
        return proj_data, theta


def reconWedgeProjData(proj, theta, wedgeStart, wedgeEnd,
                       avg_width=5, roll_dist=0, algorithm='sirt',
                       num_iter=3, num_gridx=128, num_gridy=128, patch=True, init_recon=None):
    if patch:
        proj, theta = patchWedgeProjData(proj, wedgeStart, wedgeEnd, avg_width, roll_dist=roll_dist)

        if algorithm == 'sirt':
            print('sirt_patch:', proj.shape, theta.shape)
            rec_obj_data = tomopy.recon(proj, theta, algorithm=algorithm,
                                        num_gridx=num_gridx, num_gridy=num_gridy, num_iter=num_iter,
                                        init_recon=init_recon)
        else:
            print('gridrec_patch:', proj.shape, theta.shape)
            rec_obj_data = tomopy.recon(proj, theta, algorithm='gridrec')
    else:
        if algorithm == 'sirt':
            print('sirt_no_patch:', proj.shape, theta.shape)
            rec_obj_data = tomopy.recon(proj, theta, algorithm=algorithm,
                                        num_gridx=num_gridx, num_gridy=num_gridy, num_iter=num_iter,
                                        init_recon=init_recon)
        else:
            print('gridrec_no_patch:', proj.shape, theta.shape)
            rec_obj_data = tomopy.recon(proj, theta, algorithm='gridrec')

    return rec_obj_data


# def HIOEngine(obj, theta, obj_supp, proj_modu, proj_old, wedgeStart=None, wedgeEnd=None,
#              betao=0.7, betap=0.7, wedge_supp=None, proj_supp=None):
#    '''
#    repeat:
#        obj = tomopy.recon(proj)
#        proj = tomopy.project(obj)
#        mod_proj = Prm(proj)
#        mod_obj = tomopy.recon(mod_proj)
#        Pr-1m(obj)
#            obj = mod_obj                     in support
#            obj = obj - beta*mod_obj          not in support
#    '''
#    proj_new = tomopy.project(obj,theta)
#    obj_supp = obj_supp.astype(np.int8)
#    mod_proj = np.ndarray(proj_new.shape)
#
#    mpp.figure(0)
#    mpp.imshow(proj_modu[:,2,:])
#    mpp.figure(1)
#    mpp.imshow(obj[2,:])
#    mpp.figure(2)
#    mpp.imshow(proj_new[:,2,:])
#
#    if proj_supp is None:
#        mod_proj[:] = proj_new[:]
#        mod_proj[:wedgeStart,:] = proj_modu[:wedgeStart,:]
#        mod_proj[wedgeEnd:,:] = proj_modu[wedgeEnd:,:]
#    else:
#        proj_supp = proj_supp.astype(np.int8)
#        proj_comp_supp = 1 - proj_supp
#
##        mpp.figure(1)
##        mpp.imshow(proj_supp[:,2,:])
##        mpp.figure(2)
##        mpp.imshow(proj_new[:,2,:])
##        mpp.figure(3)
##        mpp.imshow(proj_comp_supp[:,2,:])
##        mpp.figure(4)
##        mpp.imshow(proj_modu[:,2,:])
#
#        mod_proj = proj_modu * proj_supp + proj_new * wedge_supp \
#                 + proj_old * proj_comp_supp - betap * proj_new * proj_comp_supp
#
##        mpp.figure(5)
##        mpp.imshow(mod_proj[:,2,:])
#
#    mpp.figure(3)
#    mpp.imshow(mod_proj[:,2,:])
#
#    mod_obj = reconWedgeProjData(mod_proj,theta, wedgeStart, wedgeEnd,
#                                 avg_width = 5, roll_dist=0, algorithm='sirt',
#                                 num_iter=1, num_gridx=128, num_gridy=128,patch=True)
#
#    mpp.figure(4)
#    mpp.imshow(mod_obj[2,:])
#
##    obj = mod_obj*obj_supp + obj*(1-obj_supp) - betao*mod_obj*(1-obj_supp)
#    obj = mod_obj*obj_supp
#
#    mpp.figure(5)
#    mpp.imshow(obj[2,:])
#    mpp.show()
#
#    return obj,mod_proj
#
#

# def reconInterSheppProjDataWPatch(wedgeStart,wedgeEnd,sli=None,
#                                  avg_width = 5, roll_dist=0,
#                                  num_iter=3, num_gridx=128, num_gridy=128,
#                                  wedgeStart_r=None, wedgeEnd_r=None,update_freq=None):
#
#    proj,theta = genSheppProjData(sli=sli)
#    print('shepp proj data',proj.shape)
#    obj_supp = genSheppObjSupp(sli=sli,dilation=5)
#    print('shepp obj supp',obj_supp.shape)
##    proj_supp = genSheppProjSupp(wedgeStart,wedgeEnd,sli=sli,dilation=5)
#    proj_supp = genSheppProjSupp(0,0,sli=sli,dilation=5)
#    wedge_supp = np.zeros(proj.shape)
#    wedge_supp[wedgeStart:wedgeEnd,:] = 1
#    wedge_supp = wedge_supp * proj_supp
#    proj_modu = np.ndarray(proj.shape)
#    proj_modu[:] = proj[:]
#    proj_modu[wedgeStart:wedgeEnd,:] = 0
#    theta = np.linspace(0,np.pi,num=proj.shape[0])
#
#    obj = genSheppPhantom(sli=sli)
#
#    if wedgeStart_r is None:
#        wedgeStart_r= wedgeStart
#    if wedgeEnd_r is None:
#        wedgeEnd_r  = wedgeEnd
#    missing_width_r = wedgeEnd_r - wedgeStart_r
#
#    cos_phi_r = np.cos((missing_width_r+avg_width)*np.pi/180)
#
#    patched_proj = patchWedgeProjData(proj,wedgeStart,wedgeEnd,
#                                      avg_width,roll_dist=roll_dist)
#
#    obj = tomopy.recon(patched_proj[:,:,:], theta, algorithm='sirt',
#                       num_gridx=num_gridx,num_gridy=num_gridy,num_iter=3)
#
#    proj[:] = patched_proj[:]
#
##    mpp.figure(1)
##    mpp.imshow(obj_supp[2,:])
##    mpp.figure(2)
##    mpp.imshow(obj[2,:])
#
##    mpp.figure(3)
##    mpp.imshow(proj_modu[:,2,:])
##    mpp.show()
#
#    for jj in range(num_iter):
#        obj,proj = HIOEngine(obj, theta, obj_supp, proj_modu, proj, wedgeStart=wedgeStart, wedgeEnd=wedgeEnd,
#                  betao=0.7, betap=0.7, proj_supp=proj_supp,wedge_supp=wedge_supp)
#        obj = reconWedgeProjData(proj,theta, wedgeStart, wedgeEnd,
#                        avg_width = 5, roll_dist=0, algorithm='sirt',
#                        num_iter=5, num_gridx=128, num_gridy=128,patch=False)
#        proj = tomopy.project(obj,theta)
#        print(jj)
#
#    return obj


# def HIOEngine(obj, theta, obj_supp, proj_modu, proj_old, wedgeStart=None, wedgeEnd=None,
#               betao=0.7, betap=0.7, wedge_supp=None, proj_supp=None):
#     '''
#     repeat:
#         obj = tomopy.recon(proj)
#         proj = tomopy.project(obj)
#         mod_proj = Prm(proj)
#         mod_obj = tomopy.recon(mod_proj)
#         Pr-1m(obj)
#             obj = mod_obj                     in support
#             obj = obj - beta*mod_obj          not in support
#     '''
#     proj_new = tomopy.project(obj,theta)
#     obj_supp = obj_supp.astype(np.int8)
#     mod_proj = np.ndarray(proj_new.shape)
#
#     mpp.figure(0)
#     mpp.imshow(proj_modu[:,2,:])
#     mpp.figure(1)
#     mpp.imshow(obj[2,:])
#     mpp.figure(2)
#     mpp.imshow(proj_new[:,2,:])
#
#     if proj_supp is None:
#         mod_proj[:] = proj_new[:]
#         mod_proj[:wedgeStart,:] = proj_modu[:wedgeStart,:]
#         mod_proj[wedgeEnd:,:] = proj_modu[wedgeEnd:,:]
#     else:
#         proj_supp = proj_supp.astype(np.int8)
#         proj_comp_supp = 1 - proj_supp
#
# #        mpp.figure(1)
# #        mpp.imshow(proj_supp[:,2,:])
# #        mpp.figure(2)
# #        mpp.imshow(proj_new[:,2,:])
# #        mpp.figure(3)
# #        mpp.imshow(proj_comp_supp[:,2,:])
# #        mpp.figure(4)
# #        mpp.imshow(proj_modu[:,2,:])
#
#         mod_proj = proj_modu * proj_supp + proj_new * wedge_supp \
#                  + proj_old * proj_comp_supp - betap * proj_new * proj_comp_supp
#
# #        mpp.figure(5)
# #        mpp.imshow(mod_proj[:,2,:])
#
#     mpp.figure(3)
#     mpp.imshow(mod_proj[:,2,:])
#
#     mod_obj = reconWedgeProjData(mod_proj,theta, wedgeStart, wedgeEnd,
#                                  avg_width = 5, roll_dist=0, algorithm='sirt',
#                                  num_iter=1, num_gridx=128, num_gridy=128,patch=True)
#
#     mpp.figure(4)
#     mpp.imshow(mod_obj[2,:])
#
# #    obj = mod_obj*obj_supp + obj*(1-obj_supp) - betao*mod_obj*(1-obj_supp)
#     obj = mod_obj*obj_supp
#
#     mpp.figure(5)
#     mpp.imshow(obj[2,:])
#     mpp.show()
#
#     return obj,mod_proj


def HIOEngine(obj, theta, obj_supp, proj_modu, proj_old, wedgeStart=None, wedgeEnd=None,
              betao=0.7, betap=0.7, wedge_supp=None, proj_supp=None):
    """
    support constrain is set in slice space; magnitude constrain is set in sinogram space. in the terminology,
    sinogram space = proj space, slice space = obj space

    repeat:
        obj = tomopy.recon(proj)
        proj = tomopy.project(obj)
        mod_proj = Prm(proj)
        mod_obj = tomopy.recon(mod_proj)
        Pr-1m(obj)
            obj = mod_obj                     in support
            obj = obj - beta*mod_obj          not in support
    """
    proj_new = tomopy.project(obj, theta)
    obj_supp = obj_supp.astype(np.int8)
    mod_proj = np.ndarray(proj_new.shape)

    # mpp.figure(0)
    # mpp.imshow(proj_modu[:, 2, :])
    # mpp.figure(1)
    # mpp.imshow(obj[2, :])
    # mpp.figure(2)
    # mpp.imshow(proj_new[:, 2, :])

    if proj_supp is None:
        mod_proj[:] = proj_new[:]
        mod_proj[:wedgeStart, :] = proj_modu[:wedgeStart, :]
        mod_proj[wedgeEnd:, :] = proj_modu[wedgeEnd:, :]
    else:
        proj_supp = proj_supp.astype(np.int8)
        proj_comp_supp = 1 - proj_supp

        #        mpp.figure(1)
        #        mpp.imshow(proj_supp[:,2,:])
        #        mpp.figure(2)
        #        mpp.imshow(proj_new[:,2,:])
        #        mpp.figure(3)
        #        mpp.imshow(proj_comp_supp[:,2,:])
        #        mpp.figure(4)
        #        mpp.imshow(proj_modu[:,2,:])

        # mod_proj[:] = proj_modu * proj_supp + proj_new * wedge_supp \
        #            + proj_old * proj_comp_supp - betap * proj_new * proj_comp_supp

        # mod_proj[:] = proj_modu * proj_supp + (proj_old - betap * proj_new) * proj_comp_supp
        # mod_proj[(mod_proj * proj_supp) < 0] *= -1

        mod_proj[:] = proj_modu * proj_supp

    #        mpp.figure(5)
    #        mpp.imshow(mod_proj[:,2,:])

    # mpp.figure(3)
    # mpp.imshow(mod_proj[:, 2, :])

    # mod_obj = reconWedgeProjData(mod_proj, theta, wedgeStart, wedgeEnd,
    #                              avg_width=5, roll_dist=0, algorithm='sirt',
    #                              num_iter=1, num_gridx=128, num_gridy=128, patch=True)

    mod_obj = tomopy.recon(mod_proj, theta, algorithm='sirt',
                           num_gridx=128, num_gridy=128, num_iter=10,
                           init_recon=None)

    # mpp.figure(4)
    # mpp.imshow(mod_obj[2, :])

    obj[:] = mod_obj * obj_supp + (obj - betao * mod_obj) * (1 - obj_supp)
    obj[(obj * (1 - obj_supp)) < 0] *= -1

    # mpp.figure(5)
    # mpp.imshow(obj[2, :])
    # mpp.show()

    return obj, mod_proj


#
#
# def reconInterSheppProjDataWPatch(wedgeStart,wedgeEnd,sli=None,
#                                   avg_width = 5, roll_dist=0,
#                                   num_iter=3, num_gridx=128, num_gridy=128,
#                                   wedgeStart_r=None, wedgeEnd_r=None,update_freq=None):
#
#     proj, theta = genSheppProjData(sli=sli)
#     print('shepp proj data',proj.shape)
#     obj_supp = genSheppObjSupp(sli=sli, dilation=5)
#     print('shepp obj supp',obj_supp.shape)
# #    proj_supp = genSheppProjSupp(wedgeStart,wedgeEnd,sli=sli,dilation=5)
#     proj_supp = genSheppProjSupp(0, 0, sli=sli, dilation=5)
#     wedge_supp = np.zeros(proj.shape)
#     wedge_supp[wedgeStart:wedgeEnd,:] = 1
#     wedge_supp = wedge_supp * proj_supp
#     proj_modu = np.ndarray(proj.shape)
#     proj_modu[:] = proj[:]
#     proj_modu[wedgeStart:wedgeEnd, :] = 0
#     theta = np.linspace(0, np.pi, num=proj.shape[0])
#
#     obj = genSheppPhantom(sli=sli)
#
#     if wedgeStart_r is None:
#         wedgeStart_r= wedgeStart
#     if wedgeEnd_r is None:
#         wedgeEnd_r  = wedgeEnd
#     missing_width_r = wedgeEnd_r - wedgeStart_r
#
#     cos_phi_r = np.cos((missing_width_r+avg_width)*np.pi/180)
#
#     patched_proj = patchWedgeProjData(proj, wedgeStart, wedgeEnd,
#                                       avg_width, roll_dist=roll_dist)
#
#     obj = tomopy.recon(patched_proj[:, :, :], theta, algorithm='sirt',
#                        num_gridx=num_gridx, num_gridy=num_gridy, num_iter=3)
#
#     proj[:] = patched_proj[:]
#
# #    mpp.figure(1)
# #    mpp.imshow(obj_supp[2,:])
# #    mpp.figure(2)
# #    mpp.imshow(obj[2,:])
#
# #    mpp.figure(3)
# #    mpp.imshow(proj_modu[:,2,:])
# #    mpp.show()
#
#     for jj in range(num_iter):
#         obj, proj = HIOEngine(obj, theta, obj_supp, proj_modu, proj, wedgeStart=wedgeStart, wedgeEnd=wedgeEnd,
#                   betao=0.7, betap=0.7, proj_supp=proj_supp, wedge_supp=wedge_supp)
#         obj = reconWedgeProjData(proj, theta, wedgeStart, wedgeEnd,
#                         avg_width = 5, roll_dist=0, algorithm='sirt',
#                         num_iter=5, num_gridx=128, num_gridy=128, patch=False)
#         proj = tomopy.project(obj, theta)
#         print(jj)
#
#     return obj

####### sirt: reconstruct phantom with 50deg missing wedge dataset with patching -- start
# mask = (proj_data > 0).as_type(np.int8)
#
# wedgeStart= 70
# wedgeEnd  = 110
# missing_width = wedgeEnd - wedgeStart
#
# mask[wedgeStart:wedgeEnd,:,:] = False
#
# wedgeStart_r= 85
# wedgeEnd_r  = 95
# missing_width_r = wedgeEnd_r - wedgeStart_r
# freq = 5.0
#
# avg_width = 1
# roll_dist = 0
# cos_phi = np.cos((missing_width+avg_width)*np.pi/180)
# cos_phi_r = np.cos((missing_width_r+avg_width)*np.pi/180)
#
# print proj_data.shape
# patched_proj_data = np.ndarray(proj_data.shape)
# patched_proj_data[:] = proj_data[:]
# print patched_proj_data.shape
# for ii in range(wedgeStart,wedgeEnd):
#    patched_proj_data[ii,:,:] = np.roll(np.mean(np.roll(proj_data[(ii - missing_width - avg_width):(ii - missing_width),:,:],roll_dist,axis=2),axis=0) + \
#                           np.mean(np.roll(proj_data[(ii + missing_width):(ii + missing_width + avg_width),:,:],roll_dist,axis=2),axis=0),-1*roll_dist)/2.0/cos_phi
#
##mpp.figure(1)
##mpp.imshow(proj_data[:,5,:])
##mpp.figure(2)
##mpp.imshow(patched_proj_data[:,5,:])
##mpp.show()
#
# for jj in range(20):
#    temp_obj_data = tomopy.recon(patched_proj_data[:,:,:], theta, algorithm='sirt',num_gridx=128,num_gridy=128,num_iter=3)
#    temp_patched_proj_data = tomopy.project(temp_obj_data, theta)
##    mpp.figure(1)
##    mpp.imshow(patched_proj_data[:,5,:])
##    mpp.figure(2)
##    mpp.imshow(temp_patched_proj_data[:,5,:])
#
#
#    for ii in range(wedgeStart,wedgeEnd):
#        temp_patched_proj_data[ii,:,:] = ((np.roll(np.mean(np.roll(temp_patched_proj_data[(ii - missing_width - avg_width):(ii - missing_width),:,:],roll_dist,axis=2),axis=0) + \
#                               np.mean(np.roll(temp_patched_proj_data[(ii + missing_width):(ii + missing_width + avg_width),:,:],roll_dist,axis=2),axis=0),-1*roll_dist)/2.0/cos_phi)
#                               + temp_patched_proj_data[ii,:,:])/2.0
#
##    if int(np.round(jj/freq)) < len(theta)/2-wedgeStart:
##        wedgeStart_r = wedgeStart + int(np.round(jj/freq))
##    else:
##        wedgeStart_r = len(theta)/2 - 1
##    if int(np.round(jj/freq)) < wedgeEnd - len(theta)/2:
##        wedgeEnd_r   = wedgeEnd - int(np.round(jj/10.0))
##    else:
##        wedgeEnd_r = len(theta)/2 + 1
##
##    missing_width_r = wedgeEnd_r - wedgeStart_r
###    for ii in range(wedgeStart_r,wedgeEnd_r):
###        temp_patched_proj_data[ii,:,:] = (((np.roll(np.mean(np.roll(temp_patched_proj_data[(ii - missing_width_r - avg_width):(ii - missing_width_r),:,:],roll_dist,axis=2),axis=0) + \
###                               np.mean(np.roll(temp_patched_proj_data[(ii + missing_width_r):(ii + missing_width_r + avg_width),:,:],roll_dist,axis=2),axis=0),-1*roll_dist)/2.0/cos_phi_r))
###                               + temp_patched_proj_data[ii,:,:])/2.0
##    for ii in range(wedgeStart_r,wedgeEnd_r):
##        temp_patched_proj_data[ii,:,:] = ((np.roll(np.mean(np.roll(temp_patched_proj_data[(ii - missing_width_r - avg_width):(ii - missing_width_r),:,:],roll_dist,axis=2),axis=0) + \
##                               np.mean(np.roll(temp_patched_proj_data[(ii + missing_width_r):(ii + missing_width_r + avg_width),:,:],roll_dist,axis=2),axis=0),-1*roll_dist)/2.0/cos_phi_r))
#
#
#    patched_proj_data = temp_patched_proj_data
##    patched_data[:wedgeStart,:] = data[:wedgeStart,:]
##    patched_data[wedgeEnd:,:] = data[wedgeEnd:,:]
#    patched_proj_data[:wedgeStart,:] = proj_data[:wedgeStart,:]
#    patched_proj_data[wedgeEnd:,:] = proj_data[wedgeEnd:,:]
#
#    patched_proj_data[1-mask] = (1-mask)
#
##    mpp.figure(3)
##    mpp.imshow(patched_proj_data[:,5,:])
##    mpp.show()
#
#    print jj
##    temp_obj_data = tomopy.recon(patched_proj_data[:,5:7,:], theta, algorithm='gridrec')
#
#
#
# tomopy.write_tiff(temp_obj_data,'/local/data/xhxiao/2017_08/Commissioning/sirt_test/combined_patch_wedge_dataset_recon'+'_'+str(missing_width)+'deg')
# mpp.figure(3)
# mpp.imshow(patched_proj_data[:,5,:])
# mpp.show()
####### sirt: reconstruct phantom with 50deg missing wedge dataset with patching -- end


def reconInterSheppProjDataWPatch(wedgeStart, wedgeEnd, sli=None,
                                  avg_width=5, roll_dist=0,
                                  num_iter=3, num_gridx=128, num_gridy=128,
                                  wedgeStart_r=None, wedgeEnd_r=None, update_freq=None):
    proj, theta = genSheppProjData(sli=sli)
    print('shepp proj data', proj.shape)
    obj_supp = genSheppObjSupp(sli=sli, dilation=None)
    print('shepp obj supp', obj_supp.shape)
    #    proj_supp = genSheppProjSupp(wedgeStart,wedgeEnd,sli=sli,dilation=5)
    proj_supp = genSheppProjSupp(0, 0, sli=sli, dilation=None)
    wedge_supp = np.zeros(proj.shape)
    wedge_supp[wedgeStart:wedgeEnd, :] = 1
    wedge_supp = wedge_supp * proj_supp
    proj_modu = np.ndarray(proj.shape)
    proj_modu[:] = proj[:]
    proj_modu[wedgeStart:wedgeEnd, :] = 0
    theta = np.linspace(0, np.pi, num=proj.shape[0])

    obj = tomopy.recon(proj_modu[:, :, :], theta, algorithm='sirt',
                       num_gridx=num_gridx, num_gridy=num_gridy, num_iter=3)

    mpp.figure(100)
    #    mpp.imshow(obj_supp[2,:])
    #    mpp.figure(2)
    mpp.imshow(obj[2, :])
    mpp.title('initial_obj')

    mpp.figure(200)
    mpp.imshow(proj[:, 2, :])
    mpp.title('init_full_sino')

    mpp.figure(300)
    mpp.imshow(proj_modu[:, 2, :])
    mpp.title('init_wedge_sino')
    #    mpp.show()

    progress_step = max(int((wedgeEnd - wedgeStart) / num_iter), 1)
    for jj in range(num_iter):
        # obj, proj = HIOEngine(obj, theta, obj_supp, proj_modu, proj, wedgeStart=wedgeStart, wedgeEnd=wedgeEnd,
        #                       betao=1.5, betap=2., proj_supp=proj_supp, wedge_supp=wedge_supp)

        print('reconstructed proj shape:', proj.shape)
        print(wedgeStart + jj * progress_step, wedgeStart + (jj + 1) * progress_step)
        print('p11 shape: ', proj[(wedgeStart + jj * progress_step):(wedgeStart + (jj + 1) * progress_step), :].shape)
        print('p12 shape: ', proj[(wedgeStart + (jj - 1) * progress_step):(wedgeStart + jj * progress_step), :].shape)
        print('p21 shape: ', proj[(wedgeEnd - (jj + 1) * progress_step):(wedgeEnd - jj * progress_step), :].shape)
        print('p22 shape: ', proj[(wedgeEnd - jj * progress_step):(wedgeEnd - (jj - 1) * progress_step), :].shape)
        proj[wedgeStart + jj * progress_step:wedgeStart + (jj + 1) * progress_step, :] = \
            proj[wedgeStart + (jj - 1) * progress_step:wedgeStart + jj * progress_step, :]
        proj[wedgeEnd - (jj + 1) * progress_step:wedgeEnd - jj * progress_step, :] = \
            proj[wedgeEnd - jj * progress_step:wedgeEnd - (jj - 1) * progress_step, :]

        obj, proj = HIOEngine(obj, theta, obj_supp, proj_modu, proj, wedgeStart=wedgeStart, wedgeEnd=wedgeEnd,
                              betao=2., betap=1.5, proj_supp=None, wedge_supp=None)

        obj = reconWedgeProjData(proj, theta, wedgeStart, wedgeEnd,
                                 avg_width=5, roll_dist=0, algorithm='sirt',
                                 num_iter=50, num_gridx=128, num_gridy=128, patch=False)
        proj = tomopy.project(obj, theta)
        print(jj)

    mpp.figure(400)
    mpp.imshow(proj[:, 2, :])
    mpp.title('final_sino')

    return obj


def new_HIOEngine(obj, theta, obj_supp, proj_modu, wedgeStart=None, wedgeEnd=None,
                  beta_obj=0.7, beta_proj=0.7, num_gridx=128, num_gridy=128, num_iters=10, proj_supp=None):
    """
    support constrain is set in slice space; magnitude constrain is set in sinogram space. in the terminology,
    sinogram space = proj space, slice space = obj space

    repeat:
        obj = tomopy.recon(proj)
        proj = tomopy.project(obj)
        mod_proj = Prm(proj)
        mod_obj = tomopy.recon(mod_proj)
        Pr-1m(obj)
            obj = mod_obj                     in support
            obj = obj - beta*mod_obj          not in support
    """
    if wedgeStart is None:
        wedgeStart = 0
    if wedgeEnd is None:
        wedgeEnd = 0

    proj_new = tomopy.project(obj, theta)
    obj_supp = obj_supp.astype(np.int8)
    mod_proj = np.ndarray(proj_new.shape)

    if proj_supp is None:
        mod_proj[:] = proj_new[:]
        mod_proj[:wedgeStart, :] = proj_modu[:wedgeStart, :]
        mod_proj[wedgeEnd:, :] = proj_modu[wedgeEnd:, :]
    else:
        proj_supp = proj_supp.astype(np.int8)
        proj_comp_supp = 1 - proj_supp

        mod_proj[:] = proj_modu * proj_supp

    mod_obj = tomopy.recon(mod_proj, theta, algorithm='sirt',
                           num_gridx=num_gridx, num_gridy=num_gridy, num_iter=num_iters,
                           init_recon=None)

    obj[:] = mod_obj * obj_supp + (obj - beta_obj * mod_obj) * (1 - obj_supp)
    obj[(obj * (1 - obj_supp)) < 0] *= 0

    return obj, mod_proj


def new_recon_shepplogan_w_constraint(*args, **kwargs):
    """
    This is a unified routine for reconstructing shepp-logan phantom with either full or wedged projection data; support
    constraints can be optionally applied.
    :param num_iter:
    :param update_freq:
    :param kwargs:
    :return:
    """
    # ----------- extract input parameters -- start
    print(kwargs)
    wedge_start = kwargs['data_params']['wedge_start']
    wedge_end = kwargs['data_params']['wedge_end']
    sli = kwargs['data_params']['sli']
    num_angles = kwargs['data_params']['num_angles']
    num_gridx = kwargs['data_params']['num_gridx']
    num_gridy = kwargs['data_params']['num_gridy']

    use_constraints = kwargs['recon_params']['use_constraints']
    algorithm = kwargs['recon_params']['algorithm']
    num_outer_iters = kwargs['recon_params']['num_outer_iters']
    num_inner_iters = kwargs['recon_params']['num_inner_iters']
    num_iters_in_HIO = kwargs['recon_params']['num_iters_in_HIO']
    propagate_step = kwargs['recon_params']['propagate_step']

    obj_supp_dialation = kwargs['HIO_params']['obj_supp_dialation']
    beta_obj = kwargs['HIO_params']['beta_obj']
    beta_proj = kwargs['HIO_params']['beta_proj']
    use_proj_supp = kwargs['HIO_params']['use_proj_supp']
    proj_supp_dialation = kwargs['HIO_params']['proj_supp_dialation']

    record_freq = kwargs['record_params']['record_freq']
    record_dir = kwargs['record_params']['record_dir']
    # ----------- extract input parameters -- end

    # ----------- generate phantom data -- start
    proj, theta = genSheppProjData(sli=sli, num_angles=num_angles)
    print('shepp proj data', proj.shape)
    obj_supp = genSheppObjSupp(sli=sli, dilation=obj_supp_dialation)
    print('shepp obj supp', obj_supp.shape)
    proj_supp = genSheppProjSupp(0, 0, sli=sli, dilation=proj_supp_dialation)
    proj_modu = np.ndarray(proj.shape)
    proj_modu[:] = proj[:]
    proj_modu[wedge_start:wedge_end, :] = 0
    theta = np.linspace(0, np.pi, num=proj.shape[0])

    obj = tomopy.recon(proj_modu[:, :, :], theta, algorithm=algorithm,
                       num_gridx=num_gridx, num_gridy=num_gridy, num_iter=num_inner_iters)
    # ----------- generate phantom data -- end

    # ----------- iterative recon -- start
    mpp.figure(100)
    mpp.imshow(obj[2, :])
    mpp.title('initial_obj')

    mpp.figure(200)
    mpp.imshow(proj[:, 2, :])
    mpp.title('init_full_sino')

    mpp.figure(300)
    mpp.imshow(proj_modu[:, 2, :])
    mpp.title('init_wedge_sino')

    if use_constraints is not True:
        obj = tomopy.recon(proj_modu[:, :, :], theta, algorithm=algorithm,
                           num_gridx=num_gridx, num_gridy=num_gridy, num_iter=num_inner_iters * num_outer_iters)
    else:
        cnt = 0
        for jj in range(num_outer_iters):
            # obj, proj = HIOEngine(obj, theta, obj_supp, proj_modu, proj, wedgeStart=wedgeStart, wedgeEnd=wedgeEnd,
            #                       betao=1.5, betap=2., proj_supp=proj_supp, wedge_supp=wedge_supp)

            print('reconstructed proj shape:', proj.shape)
            print(wedge_start + cnt * propagate_step, wedge_start + (cnt + 1) * propagate_step)
            p11_s = (wedge_start + cnt * propagate_step)
            p11_e = (wedge_start + (cnt + 1) * propagate_step)
            p12_s = (wedge_start + (cnt - 1) * propagate_step)
            p12_e = (wedge_start + cnt * propagate_step)
            p21_s = (wedge_end - (cnt + 1) * propagate_step)
            p21_e = (wedge_end - cnt * propagate_step)
            p22_s = (wedge_end - cnt * propagate_step)
            p22_e = (wedge_end - (cnt - 1) * propagate_step)
            print('p11 shape: ', proj[p11_s:p11_e, :].shape)
            print('p12 shape: ', proj[p12_s:p12_e, :].shape)
            print('p21 shape: ', proj[p21_s:p21_e, :].shape)
            print('p22 shape: ', proj[p22_s:p22_e, :].shape)

            proj[p11_s:p11_e, :] = proj[p12_s:p12_e, :]
            proj[p21_s:p21_e, :] = proj[p22_s:p22_e, :]

            if p12_e < wedge_end:
                cnt += 1
            else:
                cnt = 0

            if use_proj_supp is None:
                obj, proj = new_HIOEngine(obj, theta, obj_supp, proj_modu, wedgeStart=wedge_start, wedgeEnd=wedge_end,
                                          beta_obj=beta_obj, beta_proj=beta_proj, proj_supp=None,
                                          num_iters=num_iters_in_HIO)
            else:
                obj, proj = new_HIOEngine(obj, theta, obj_supp, proj_modu, wedgeStart=wedge_start, wedgeEnd=wedge_end,
                                          beta_obj=beta_obj, beta_proj=beta_proj, proj_supp=proj_supp,
                                          num_iters=num_iters_in_HIO)

            obj = reconWedgeProjData(proj, theta, wedge_start, wedge_end, algorithm=algorithm,
                                     num_iter=num_inner_iters, num_gridx=num_gridx, num_gridy=num_gridx, patch=False)

            proj = tomopy.project(obj, theta)
            print(jj)

            if record_freq is not None:
                if not os.path.exists(record_dir):
                    os.makedirs(record_dir)
                if jj % record_freq == 0:
                    fn = os.path.join(record_dir, "recon_obj_" + str(jj) + ".tif")
                    tifffile.imsave(fn, obj)
                    fn = os.path.join(record_dir, "updated_sino_" + str(jj) + ".tif")
                    tifffile.imsave(fn, proj)

                fout = os.path.join(record_dir, 'log.txt')
                fo = open(fout, "w")
                for k, v in kwargs.items():
                    fo.write(str(k) + ': ' + str(v) + '\n\n')
                fo.close()

    mpp.figure(400)
    mpp.imshow(proj[:, 2, :])
    mpp.title('final_sino')

    return obj


def new_2reconWedgeProjData(proj, theta, wedgeStart, wedgeEnd, avg_width, roll_dist,
                            num_iter=3, patch=True, **algorithm_config):
    if patch:
        proj, theta = patchWedgeProjData(proj, wedgeStart, wedgeEnd, avg_width, roll_dist=roll_dist)

        if algorithm_config['algorithm'] == 'gridrec':
            print('gridrec_patch:', proj.shape, theta.shape)
            rec_obj_data = tomopy.recon(proj, theta, algorithm='gridrec')
        else:
            print(algorithm_config['algorithm'] + '_patch:', proj.shape, theta.shape)
            rec_obj_data = tomopy.recon(proj, theta, num_iter=num_iter, **algorithm_config)
    else:
        if algorithm_config['algorithm'] == 'gridrec':
            print('gridrec_no_patch:', proj.shape, theta.shape)
            rec_obj_data = tomopy.recon(proj, theta, algorithm='gridrec')
        else:
            print(algorithm_config['algorithm'] + '_no_patch:', proj.shape, theta.shape)
            rec_obj_data = tomopy.recon(proj, theta, num_iter=num_iter, **algorithm_config)

    return rec_obj_data


def new_2HIOEngine(obj, theta, obj_supp, proj_modu, wedgeStart = None, wedgeEnd = None, beta_obj = 0.7,
                   beta_proj = 0.7, num_iters = 10, proj_supp = None, positivity_constraint = True, **kwargs):
    """
    support constrain is set in slice space; magnitude constrain is set in sinogram space. in the terminology,
    sinogram space = proj space, slice space = obj space

    repeat:
        obj = tomopy.recon(proj)
        proj = tomopy.project(obj)
        mod_proj = Prm(proj)
        mod_obj = tomopy.recon(mod_proj)
        Pr-1m(obj)
            obj = mod_obj                     in support
            obj = obj - beta*mod_obj          not in support
    """
    if wedgeStart is None:
        wedgeStart = 0
    if wedgeEnd is None:
        wedgeEnd = 0

    proj_new = tomopy.project(obj, theta)
    # obj_supp = obj_supp.astype(np.int8)
    mod_proj = np.ndarray(proj_new.shape)

    if proj_supp is None:
        mod_proj[:] = proj_new[:]
        mod_proj[:wedgeStart, :] = proj_modu[:wedgeStart, :]
        mod_proj[wedgeEnd:, :] = proj_modu[wedgeEnd:, :]
    else:
        # proj_supp = proj_supp.astype(np.int8)
        proj_comp_supp = 1 - proj_supp

        mod_proj[:] = proj_modu * proj_supp

    if positivity_constraint:
        mod_proj[mod_proj < 0] = 0

    if kwargs['algorithm'] == 'gridrec':
        mod_obj = tomopy.recon(mod_proj, theta, algorithm='gridrec')
    elif kwargs['algorithm'] in ['art', 'fbp', 'bart', 'sirt', 'tv']:
        mod_obj = tomopy.recon(mod_proj, theta, num_iter=num_iters, **kwargs)
    elif kwargs['algorithm'] in ['mlem', 'osem', 'ospml_hybrid', 'ospml_quad', 'pml_hybrid', 'pml_quad', 'grad']:
        ind_range = np.concatenate((np.arange(0, wedgeStart), np.arange(wedgeEnd, mod_proj.shape[0])))
        mod_obj = tomopy.recon(mod_proj[ind_range, :], theta[ind_range], num_iter=num_iters, **kwargs)

    obj[:] = mod_obj * obj_supp + (obj - beta_obj * mod_obj) * (1 - obj_supp)
    if positivity_constraint:
        obj[obj<0] = 0
    # obj[(obj * (1 - obj_supp)) < 0] *= 0

    return obj, mod_proj


def new_2recon_shepplogan(*args, **kwargs):
    """
    This is a unified routine for reconstructing shepp-logan phantom with either full or wedged projection data; support
    constraints can be optionally applied.
    :param num_iter:
    :param update_freq:
    :param kwargs:
    :return:
    """
    # ----------- extract input parameters -- start
    print(kwargs)
    wedge_start = kwargs['data_params']['wedge_start']
    wedge_end = kwargs['data_params']['wedge_end']
    sli = kwargs['data_params']['sli']
    num_angles = kwargs['data_params']['num_angles']

    # algorithm = kwargs['algorithm_params']['algorithm']
    # num_gridx = kwargs['algorithm_params']['num_gridx']
    # num_gridy = kwargs['algorithm_params']['num_gridy']
    # reg_par = kwargs['algorithm_params']['reg_par']
    algorithm_config = kwargs['algorithm_params']

    use_constraints = kwargs['recon_params']['use_constraints']
    num_outer_iters = kwargs['recon_params']['num_outer_iters']
    num_inner_iters = kwargs['recon_params']['num_inner_iters']
    num_iters_in_HIO = kwargs['recon_params']['num_iters_in_HIO']
    propagate_step = kwargs['recon_params']['propagate_step']

    obj_supp_dialation = kwargs['HIO_params']['obj_supp_dialation']
    beta_obj = kwargs['HIO_params']['beta_obj']
    beta_proj = kwargs['HIO_params']['beta_proj']
    use_proj_supp = kwargs['HIO_params']['use_proj_supp']
    proj_supp_dialation = kwargs['HIO_params']['proj_supp_dialation']

    record_freq = kwargs['record_params']['record_freq']
    record_dir = kwargs['record_params']['record_dir']
    # ----------- extract input parameters -- end

    # ----------- generate phantom data -- start
    proj, theta = genSheppProjData(sli=sli, num_angles=num_angles)
    print('shepp proj data', proj.shape)
    obj_supp = genSheppObjSupp(sli=sli, dilation=obj_supp_dialation)
    print('shepp obj supp', obj_supp.shape)
    proj_supp = genSheppProjSupp(0, 0, sli=sli, dilation=proj_supp_dialation)
    proj_modu = np.ndarray(proj.shape)
    proj_modu[:] = proj[:]
    proj_modu[wedge_start:wedge_end, :] = 0
    # theta = np.linspace(0, np.pi, num = proj.shape[0])

    if algorithm_config['algorithm'] == 'gridrec':
        obj = tomopy.recon(proj_modu, theta, algorithm='gridrec')
    else:
        obj = tomopy.recon(proj_modu, theta, num_iter=num_inner_iters, **algorithm_config)
    # ----------- generate phantom data -- end

    # ----------- iterative recon -- start
    mpp.figure(100)
    mpp.imshow(obj[5, :])
    mpp.title('initial_obj')

    mpp.figure(200)
    mpp.imshow(proj[:, 5, :])
    mpp.title('init_full_sino')

    mpp.figure(300)
    mpp.imshow(proj_modu[:, 5, :])
    mpp.title('init_wedge_sino')

    if record_freq is not None:
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)

        fn = os.path.join(record_dir, "proj_modu.tif")
        tifffile.imsave(fn, np.float32(proj_modu))
        fn = os.path.join(record_dir, "proj_full.tif")
        tifffile.imsave(fn, np.float32(proj))
        fn = os.path.join(record_dir, "obj_supp.tif")
        tifffile.imsave(fn, np.float32(obj_supp))
        fn = os.path.join(record_dir, "obj_org.tif")
        tifffile.imsave(fn, np.float32(tomopy.shepp3d()[sli[0]:sli[1], :]))

    if use_constraints is not True:
        obj = tomopy.recon(proj_modu[:, :, :], theta, num_iter=num_inner_iters * num_outer_iters, **algorithm_config)
        proj = tomopy.project(obj, theta)

        if record_freq is not None:
            fn = os.path.join(record_dir, "recon_obj_final.tif")
            tifffile.imsave(fn, obj)
            fn = os.path.join(record_dir, "recon_proj_final.tif")
            tifffile.imsave(fn, proj)

            fout = os.path.join(record_dir, 'log.txt')
            fo = open(fout, "w")
            for k, v in kwargs.items():
                fo.write(str(k) + ': ' + str(v) + '\n\n')
            fo.close()
    else:
        cnt = 0
        for jj in range(num_outer_iters):
            print('reconstructed proj shape:', proj.shape)
            print(wedge_start + cnt * propagate_step, wedge_start + (cnt + 1) * propagate_step)
            p11_s = (wedge_start + cnt * propagate_step)
            p11_e = (wedge_start + (cnt + 1) * propagate_step)
            p12_s = (wedge_start + (cnt - 1) * propagate_step)
            p12_e = (wedge_start + cnt * propagate_step)
            p21_s = (wedge_end - (cnt + 1) * propagate_step)
            p21_e = (wedge_end - cnt * propagate_step)
            p22_s = (wedge_end - cnt * propagate_step)
            p22_e = (wedge_end - (cnt - 1) * propagate_step)
            print('p11 shape: ', proj[p11_s:p11_e, :].shape)
            print('p12 shape: ', proj[p12_s:p12_e, :].shape)
            print('p21 shape: ', proj[p21_s:p21_e, :].shape)
            print('p22 shape: ', proj[p22_s:p22_e, :].shape)

            proj[p11_s:p11_e, :] = proj[p12_s:p12_e, :]
            proj[p21_s:p21_e, :] = proj[p22_s:p22_e, :]

            if p12_e < wedge_end:
                cnt += 1
            else:
                cnt = 0

            if use_proj_supp is None:
                obj, proj = new_2HIOEngine(obj, theta, obj_supp, proj_modu, wedgeStart=wedge_start, wedgeEnd=wedge_end,
                                           beta_obj=beta_obj, beta_proj=beta_proj, proj_supp=None,
                                           num_iters=num_iters_in_HIO, **algorithm_config)
            else:
                obj, proj = new_2HIOEngine(obj, theta, obj_supp, proj_modu, wedgeStart=wedge_start, wedgeEnd=wedge_end,
                                           beta_obj=beta_obj, beta_proj=beta_proj, proj_supp=proj_supp,
                                           num_iters=num_iters_in_HIO, **algorithm_config)

            obj = new_2reconWedgeProjData(proj, theta, wedge_start, wedge_end, num_iter=num_inner_iters, patch=False,
                                          **algorithm_config)

            proj = tomopy.project(obj, theta)
            print(jj)

            if record_freq is not None:
                if jj % record_freq == 0:
                    fn = os.path.join(record_dir, "recon_obj_" + str(jj) + ".tif")
                    tifffile.imsave(fn, obj)
                    fn = os.path.join(record_dir, "updated_sino_" + str(jj) + ".tif")
                    tifffile.imsave(fn, proj)

                fout = os.path.join(record_dir, 'log.txt')
                fo = open(fout, "w")
                for k, v in kwargs.items():
                    fo.write(str(k) + ': ' + str(v) + '\n\n')
                fo.close()

    mpp.figure(400)
    mpp.imshow(proj[:, 5, :])
    mpp.title('final_sino')

    return obj


def new_2recon_universial(*args, **kwargs):
    """
    This is a unified routine for reconstructing shepp-logan phantom with either full or wedged projection data; support
    constraints can be optionally applied.
    :param num_iter:
    :param update_freq:
    :param kwargs:
    :return:
    """
    # ----------- extract input parameters -- start
    print(kwargs)
    wedge_start = kwargs['data_params']['wedge_start']
    wedge_end = kwargs['data_params']['wedge_end']
    obj = kwargs['data_params']['obj']
    proj_modu = kwargs['data_params']['proj_modu']
    num_angles = kwargs['data_params']['num_angles']

    algorithm_config = kwargs['algorithm_params']

    use_support_constraints = kwargs['recon_params']['use_support_constraints']
    use_positivity_constraint = kwargs['recon_params']['use_positivity_constraint']
    num_outer_iters = kwargs['recon_params']['num_outer_iters']
    num_inner_iters = kwargs['recon_params']['num_inner_iters']
    num_iters_in_HIO = kwargs['recon_params']['num_iters_in_HIO']
    propagate_step = kwargs['recon_params']['propagate_step']

    obj_supp = kwargs['HIO_params']['obj_supp']
    beta_obj = kwargs['HIO_params']['beta_obj']
    beta_proj = kwargs['HIO_params']['beta_proj']
    use_proj_supp = kwargs['HIO_params']['use_proj_supp']
    proj_supp = kwargs['HIO_params']['proj_supp']

    record_freq = kwargs['record_params']['record_freq']
    record_dir = kwargs['record_params']['record_dir']
    # ----------- extract input parameters -- end

    # # ----------- generate phantom data -- start
    # proj, theta = genProjData(obj_org, num_angles=num_angles)
    # print('shepp proj data', proj.shape)
    # obj_supp = genObjSupp(obj_org, dilation=obj_supp_dialation)
    # print('shepp obj supp', obj_supp.shape)
    # proj_supp = genProjSupp(proj, wedge_start, wedge_end, dilation=proj_supp_dialation)
    # proj_modu = np.ndarray(proj.shape)
    # proj_modu[:] = proj[:]
    # proj_modu[wedge_start:wedge_end, :] = 0
    # # theta = np.linspace(0, np.pi, num = proj.shape[0])
    #
    # if algorithm_config['algorithm'] == 'gridrec':
    #     obj_org = tomopy.recon(proj, theta, algorithm='gridrec')
    # else:
    #     obj_org = tomopy.recon(proj, theta, num_iter=num_inner_iters, **algorithm_config)
    # # ----------- generate phantom data -- end
    #
    # # ----------- iterative recon -- start
    # mpp.figure(100)
    # mpp.imshow(obj_org[0, :])
    # mpp.title('initial_obj')
    #
    # mpp.figure(200)
    # mpp.imshow(proj[:, 0, :])
    # mpp.title('init_full_sino')
    #
    # mpp.figure(300)
    # mpp.imshow(proj_modu[:, 0, :])
    # mpp.title('init_wedge_sino')

    proj, theta = genProjData(obj, num_angles=num_angles)
    if record_freq is not None:
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)

        fn = os.path.join(record_dir, "proj_modu.tif")
        tifffile.imsave(fn, np.float32(proj_modu))
        fn = os.path.join(record_dir, "proj_inter.tif")
        tifffile.imsave(fn, np.float32(proj))
        fn = os.path.join(record_dir, "obj_supp.tif")
        tifffile.imsave(fn, np.float32(obj_supp))
        # fn = os.path.join(record_dir, "obj_org.tif")
        # tifffile.imsave(fn, np.float32(obj_org))

    if use_support_constraints is not True:
        obj = tomopy.recon(proj_modu[:, :, :], theta, num_iter=num_inner_iters * num_outer_iters, **algorithm_config)
        proj = tomopy.project(obj, theta)

        if record_freq is not None:
            fn = os.path.join(record_dir, "recon_obj_final.tif")
            tifffile.imsave(fn, obj)
            fn = os.path.join(record_dir, "recon_proj_final.tif")
            tifffile.imsave(fn, proj)

            fout = os.path.join(record_dir, 'log.txt')
            fo = open(fout, "w")
            for k, v in kwargs.items():
                fo.write(str(k) + ': ' + str(v) + '\n\n')
            fo.close()
    else:
        cnt = 0
        for jj in range(num_outer_iters):
            print('reconstructed proj shape:', proj.shape)
            print(wedge_start + cnt * propagate_step, wedge_start + (cnt + 1) * propagate_step)
            p11_s = (wedge_start + cnt * propagate_step)
            p11_e = (wedge_start + (cnt + 1) * propagate_step)
            p12_s = (wedge_start + (cnt - 1) * propagate_step)
            p12_e = (wedge_start + cnt * propagate_step)
            p21_s = (wedge_end - (cnt + 1) * propagate_step)
            p21_e = (wedge_end - cnt * propagate_step)
            p22_s = (wedge_end - cnt * propagate_step)
            p22_e = (wedge_end - (cnt - 1) * propagate_step)
            print('p11 shape: ', proj[p11_s:p11_e, :].shape)
            print('p12 shape: ', proj[p12_s:p12_e, :].shape)
            print('p21 shape: ', proj[p21_s:p21_e, :].shape)
            print('p22 shape: ', proj[p22_s:p22_e, :].shape)

            proj[p11_s:p11_e, :] = proj[p12_s:p12_e, :]
            proj[p21_s:p21_e, :] = proj[p22_s:p22_e, :]
            # obj = tomopy.recon(proj, theta, num_iter=10, **algorithm_config)

            if p12_e < wedge_end:
                cnt += 1
            else:
                cnt = 0

            if use_proj_supp is None:
                obj, proj = new_2HIOEngine(obj, theta, obj_supp, proj_modu, wedgeStart=wedge_start, wedgeEnd=wedge_end,
                                           beta_obj=beta_obj, beta_proj=beta_proj, proj_supp=None,
                                           num_iters=num_iters_in_HIO, positivity_constraint=use_positivity_constraint,
                                           **algorithm_config)
            else:
                obj, proj = new_2HIOEngine(obj, theta, obj_supp, proj_modu, wedgeStart=wedge_start, wedgeEnd=wedge_end,
                                           beta_obj=beta_obj, beta_proj=beta_proj, proj_supp=proj_supp,
                                           num_iters=num_iters_in_HIO, positivity_constraint=use_positivity_constraint,
                                           **algorithm_config)

            # obj = new_2reconWedgeProjData(proj, theta, wedge_start, wedge_end, num_iter=num_inner_iters, patch=False,
            #                               **algorithm_config)
            #
            # proj = tomopy.project(obj, theta)
            print(jj)

            if record_freq is not None:
                if jj % record_freq == 0:
                    fn = os.path.join(record_dir, "recon_obj_" + str(jj) + ".tif")
                    tifffile.imsave(fn, obj)
                    fn = os.path.join(record_dir, "updated_sino_" + str(jj) + ".tif")
                    tifffile.imsave(fn, proj)

                fout = os.path.join(record_dir, 'log.txt')
                fo = open(fout, "w")
                for k, v in kwargs.items():
                    fo.write(str(k) + ': ' + str(v) + '\n\n')
                fo.close()

    mpp.figure(400)
    mpp.imshow(proj[:, 0, :])
    mpp.title('final_sino')

    return obj


if __name__ == '__main__':
    #     wedgeStart = 60
    #     wedgeEnd = 120
    #     obj = reconInterSheppProjDataWPatch(wedgeStart, wedgeEnd, sli=[60,70],
    #                        avg_width = 5, roll_dist=0,
    #                        num_iter=80, num_gridx=128, num_gridy=128,
    #                        wedgeStart_r=None, wedgeEnd_r=None,update_freq=None)
    #
    #     # proj_full, theta = genSheppProjData(sli=[60,70])
    #     # rec_full = reconWedgeProjData(proj_full, theta, wedgeStart, wedgeEnd,
    #     #                     avg_width = 5, roll_dist=0, algorithm='sirt',
    #     #                     num_iter=10, num_gridx=184, num_gridy=184, patch=False)
    #     #
    #     # mpp.figure(1)
    #     # mpp.imshow(obj[2, :])
    #     # mpp.title('rec_full')
    #     mpp.figure(2)
    #     mpp.imshow(obj[2, :])
    #     mpp.title('rec_wedge_sirt_no_patch')
    #
    #     # mpp.show()
    #
    #
    #
    #     proj_full, theta = genSheppProjData(sli=[60,70])
    # # #    proj,theta = genWedgeProjData(proj,wedgeStart,wedgeEnd,reduce=False,sli=None)
    # #     rec_full = reconWedgeProjData(proj_full, theta, wedgeStart, wedgeEnd,
    # #                         avg_width = 5, roll_dist=0, algorithm='sirt',
    # #                         num_iter=10, num_gridx=184, num_gridy=184, patch=False)
    # #
    #     proj_patch, theta = genWedgeProjData(proj_full, wedgeStart, wedgeEnd, reduce=True, sli=None)
    #     rec_wedge_sirt_no_patch = reconWedgeProjData(proj_patch, theta, wedgeStart, wedgeEnd,
    #                         avg_width = 5, roll_dist=0, algorithm='sirt',
    #                         num_iter=2000, num_gridx=184, num_gridy=184, patch=False, init_recon=None)
    #     proj_final = tomopy.project(rec_wedge_sirt_no_patch, np.linspace(0, np.pi, num=181))
    #     # rec_wedge_gridrec_no_patch = reconWedgeProjData(proj_patch,theta, wedgeStart, wedgeEnd,
    #     #                     avg_width = 5, roll_dist=0, algorithm='gridrec',patch=False)
    #     # rec_wedge_gridrec_patch = reconWedgeProjData(proj_full,theta, wedgeStart, wedgeEnd,
    #     #                     avg_width = 5, roll_dist=0, algorithm='gridrec',patch=True)
    #
    #     # rec_wedge_sirt_no_patch.shape)
    #     # print(rec_wedge_gridrec_no_patch.shape)
    #     # print(rec_wedge_gridrec_patch.shape)
    #
    #     # mpp.figure(1)
    #     # mpp.imshow(rec_full[2,:])
    #     # mpp.title('rec_full')
    #     mpp.figure(2000)
    #     mpp.imshow(rec_wedge_sirt_no_patch[2, :])
    #     mpp.title('rec_wedge_sirt_no_patch')
    #     mpp.figure(3000)
    #     mpp.imshow(proj_full[:, 2, :])
    #     mpp.title('init_full_sino')
    #     mpp.figure(4000)
    #     mpp.imshow(proj_final[:, 2, :])
    #     mpp.title('final_sino')
    #     # mpp.figure(3)
    #     # mpp.imshow(rec_wedge_gridrec_no_patch[2,:])
    #     # mpp.title('rec_wedge_gridrec_no_patch')
    #     # mpp.figure(4)
    #     # mpp.imshow(rec_wedge_gridrec_patch[2,:])
    #     # mpp.title('rec_wedge_gridrec_patch')
    #
    #     # mpp.figure(6)
    #     # mpp.imshow((rec_wedge_gridrec_no_patch[2,:]-rec_full[2,:])/rec_full[2,:])
    #     # mpp.title('rec_wedge_gridrec_no_patch-rec_full')
    #     # mpp.figure(7)
    #     # mpp.imshow((rec_wedge_gridrec_patch[2,:]-rec_full[2,:])/rec_full[2,:])
    #     # mpp.title('rec_wedge_gridrec_patch-rec_full')
    #     mpp.show()

    # # ------------------ new recon function 1
    # params = {'data_params': {'wedge_start': 50,
    #                           'wedge_end': 130,
    #                           'sli': [60, 70],
    #                           'num_angles': 180,
    #                           'num_gridx': 128,
    #                           'num_gridy': 128},
    #           'recon_params': {'use_constraints': True,
    #                            'algorithm': 'sirt',
    #                            'num_outer_iters': 80,
    #                            'num_inner_iters': 50,
    #                            'num_iters_in_HIO': 10,
    #                            'propagate_step': 4},
    #           'HIO_params': {'obj_supp_dialation': 0,
    #                          'beta_obj': 1.,
    #                          'beta_proj': 1.,
    #                          'use_proj_supp': None,
    #                          'proj_supp_dialation': 0},
    #           'record_params': {'record_freq': 1,
    #                             'record_dir': '/media/Disk2/data/WedgedDataReconTest/test3'}}
    #
    # obj = new_recon_shepplogan_w_constraint(**params)
    #
    # mpp.figure(2)
    # mpp.imshow(obj[2, :])
    # mpp.title('rec_wedge_sirt_no_patch')
    # mpp.show()

    # # ------------------ new recon function 2
    # # algorithm_config = {'algorithm': 'sirt',
    # #                     'num_gridx': 128,
    # #                     'num_gridy': 128,
    # #                     'init_recon': None}
    #
    # algorithm_config = {'algorithm': 'tv',
    #                     'num_gridx': 128,
    #                     'num_gridy': 128,
    #                     'reg_par': 0.6,
    #                     'init_recon': None,}
    #
    # params = {'data_params': {'wedge_start': 50,
    #                           'wedge_end': 130,
    #                           'sli': [60, 70],
    #                           'num_angles': 180},
    #           'algorithm_params': algorithm_config,
    #           'recon_params': {'use_constraints': True,
    #                            'num_outer_iters': 20,
    #                            'num_inner_iters': 20,
    #                            'num_iters_in_HIO': 20,
    #                            'propagate_step': 4},
    #           'HIO_params': {'obj_supp_dialation': 1,
    #                          'beta_obj': 1.,
    #                          'beta_proj': 1.,
    #                          'use_proj_supp': None,
    #                          'proj_supp_dialation': 0},
    #           'record_params': {'record_freq': 1,
    #                             'record_dir': '/media/Disk2/data/WedgedDataReconTest/test20_combined/tv1'}}
    #
    # obj = new_2recon_shepplogan(**params)
    #
    # mpp.figure(500)
    # mpp.imshow(obj[5, :])
    # mpp.title('rec_wedge_sirt_no_patch')
    #
    # cnt = 2
    # for i in range(2):
    #     algorithm_config = {'algorithm': 'sirt',
    #                         'num_gridx': 128,
    #                         'num_gridy': 128,
    #                         'init_recon': obj}
    #
    #     params = {'data_params': {'wedge_start': 50,
    #                               'wedge_end': 130,
    #                               'sli': [60, 70],
    #                               'num_angles': 180},
    #               'algorithm_params': algorithm_config,
    #               'recon_params': {'use_constraints': True,
    #                                'num_outer_iters': 40,
    #                                'num_inner_iters': 50,
    #                                'num_iters_in_HIO': 20,
    #                                'propagate_step': 4},
    #               'HIO_params': {'obj_supp_dialation': 1,
    #                              'beta_obj': 1.,
    #                              'beta_proj': 1.,
    #                              'use_proj_supp': None,
    #                              'proj_supp_dialation': 0},
    #               'record_params': {'record_freq': 1,
    #                                 'record_dir': '/media/Disk2/data/WedgedDataReconTest/test20_combined/sirt' + str(cnt)}}
    #
    #     obj = new_2recon_shepplogan(**params)
    #     cnt += 1
    #
    #     mpp.figure((cnt + 3) * 100)
    #     mpp.imshow(obj[5, :])
    #     mpp.title('rec_wedge_sirt_no_patch')
    #
    #     algorithm_config = {'algorithm': 'tv',
    #                         'num_gridx': 128,
    #                         'num_gridy': 128,
    #                         'reg_par': 0.6,
    #                         'init_recon': obj, }
    #
    #     params = {'data_params': {'wedge_start': 50,
    #                               'wedge_end': 130,
    #                               'sli': [60, 70],
    #                               'num_angles': 180},
    #               'algorithm_params': algorithm_config,
    #               'recon_params': {'use_constraints': True,
    #                                'num_outer_iters': 40,
    #                                'num_inner_iters': 50,
    #                                'num_iters_in_HIO': 20,
    #                                'propagate_step': 4},
    #               'HIO_params': {'obj_supp_dialation': 1,
    #                              'beta_obj': 1.,
    #                              'beta_proj': 1.,
    #                              'use_proj_supp': None,
    #                              'proj_supp_dialation': 0},
    #               'record_params': {'record_freq': 1,
    #                                 'record_dir': '/media/Disk2/data/WedgedDataReconTest/test20_combined/tv' + str(cnt)}}
    #
    #     obj = new_2recon_shepplogan(**params)
    #     cnt += 1
    #
    #     mpp.figure((cnt + 3) * 100)
    #     mpp.imshow(obj[5, :])
    #     mpp.title('rec_wedge_sirt_no_patch')
    #
    # mpp.show()

    # ------------------ new recon function 2 for universial input data
    # ++++++++++++ generate data -- start
    data = np.zeros([1, 128, 128])
    data[0, 14:114, 14:114] = tomopy.lena(size=100)[:]
    # data[:] = tomopy.shepp2d(size=128)[:]
    obj = np.zeros([1, 128, 128])
    num_angles = 180
    obj_supp_dialation = 1
    proj_supp_dialation = 1
    wedge_start = 50
    wedge_end = 130
    propagate_step = 0
    test_dir = 'test44_lena_combined'

    proj, theta = genProjData(data, num_angles=num_angles)
    print('shepp proj data', proj.shape)
    obj_supp = genObjSupp(data, dilation=obj_supp_dialation)
    print('shepp obj supp', obj_supp.shape)
    proj_supp = genProjSupp(proj, wedge_start, wedge_end, dilation=proj_supp_dialation)
    proj_modu = np.ndarray(proj.shape)
    proj_modu[:] = proj[:]
    proj_modu[wedge_start:wedge_end, :] = 0
    # theta = np.linspace(0, np.pi, num = proj.shape[0])

    # proj = np.delete(proj, np.arange(wedge_start, wedge_end), axis=0)
    # theta = np.delete(theta, np.arange(wedge_start, wedge_end), axis=0)
    # print(proj.shape, theta.shape)
    # obj_recon = tomopy.recon(proj, theta, algorithm='pml_quad', num_gridx=128, num_gridy=128, reg_par=0.1, num_iter=20)
    obj_recon = tomopy.recon(proj, theta, algorithm='gridrec')
    tifffile.imsave('/home/xiao/tmp/temp.tif', obj_recon)
    # if algorithm_config['algorithm'] == 'gridrec':
    #     obj_recon = tomopy.recon(proj, theta, algorithm='gridrec')
    # else:
    #     obj_recon = tomopy.recon(proj, theta, num_iter=num_inner_iters, **algorithm_config)
    # ----------- generate phantom data -- end

    # ----------- iterative recon -- start
    mpp.figure(100)
    mpp.imshow(obj_recon[0, :])
    mpp.title('initial_obj')

    mpp.figure(200)
    mpp.imshow(proj[:, 0, :])
    mpp.title('init_full_sino')

    mpp.figure(300)
    mpp.imshow(proj_modu[:, 0, :])
    mpp.title('init_wedge_sino')
    # ++++++++++++ generate data -- end





    # algorithm_config = {'algorithm': 'sirt',
    #                     'num_gridx': 128,
    #                     'num_gridy': 128,
    #                     'init_recon': None}
    #
    # algorithm_config = {'algorithm': 'tv',
    #                     'num_gridx': 128,
    #                     'num_gridy': 128,
    #                     'reg_par': 0.3,
    #                     'init_recon': None, }
    #
    # params = {'data_params': {'wedge_start': wedge_start,
    #                           'wedge_end': wedge_end,
    #                           'proj_modu': proj_modu,
    #                           'obj': obj,
    #                           'num_angles': num_angles},
    #           'algorithm_params': algorithm_config,
    #           'recon_params': {'use_support_constraints': True,
    #                            'use_positivity_constraint': True,
    #                            'num_outer_iters': 20,
    #                            'num_inner_iters': 30,
    #                            'num_iters_in_HIO': 20,
    #                            'propagate_step': propagate_step},
    #           'HIO_params': {'obj_supp': obj_supp,
    #                          'beta_obj': 1.,
    #                          'beta_proj': 1.,
    #                          'use_proj_supp': None,
    #                          'proj_supp': proj_supp},
    #           'record_params': {'record_freq': 1,
    #                             'record_dir': '/media/Disk2/data/WedgedDataReconTest/' + test_dir + '/tv1'}}
    #
    # obj = new_2recon_universial(**params)

    mpp.figure(500)
    mpp.imshow(obj[0, :])
    mpp.title('rec_wedge_sirt_no_patch')

    cnt = 2
    for i in range(2):
        # algorithm_config = {'algorithm': 'sirt',
        #                     'num_gridx': 128,
        #                     'num_gridy': 128,
        #                     'init_recon': obj}
        algorithm_config = {'algorithm': 'grad',
                            'num_gridx': 128,
                            'num_gridy': 128,
                            'reg_par': 0.1,
                            'init_recon': obj, }

        params = {'data_params': {'wedge_start': wedge_start,
                                  'wedge_end': wedge_end,
                                  'proj_modu': proj_modu,
                                  'obj': obj,
                                  'num_angles': num_angles},
                  'algorithm_params': algorithm_config,
                  'recon_params': {'use_support_constraints': True,
                                   'use_positivity_constraint': True,
                                   'num_outer_iters': 20,
                                   'num_inner_iters': 30,
                                   'num_iters_in_HIO': 20,
                                   'propagate_step': propagate_step},
                  'HIO_params': {'obj_supp': obj_supp,
                                 'beta_obj': 1.,
                                 'beta_proj': 1.,
                                 'use_proj_supp': None,
                                 'proj_supp': proj_supp},
                  'record_params': {'record_freq': 1,
                                    'record_dir': '/media/Disk2/data/WedgedDataReconTest/' + test_dir + '/' +
                                    algorithm_config['algorithm'] + str(cnt)}}

        obj = new_2recon_universial(**params)
        cnt += 1

        mpp.figure((cnt + 3) * 100)
        mpp.imshow(obj[0, :])
        mpp.title('rec_wedge_sirt_no_patch')

        # algorithm_config = {'algorithm': 'tv',
        #                     'num_gridx': 128,
        #                     'num_gridy': 128,
        #                     'reg_par': 0.3,
        #                     'init_recon': obj, }
        # algorithm_config = {'algorithm': 'grad',
        #                     'num_gridx': 128,
        #                     'num_gridy': 128,
        #                     'reg_par': 0.1,
        #                     'init_recon': obj, }
        algorithm_config = {'algorithm': 'mlem',
                            'num_gridx': 128,
                            'num_gridy': 128,
                            'init_recon': obj, }
        # algorithm_config = {'algorithm': 'bart',
        #                     'num_gridx': 128,
        #                     'num_gridy': 128,
        #                     'num_block': 5,
        #                     'init_recon': obj, }

        params = {'data_params': {'wedge_start': wedge_start,
                                  'wedge_end': wedge_end,
                                  'proj_modu': proj_modu,
                                  'obj': obj,
                                  'num_angles': num_angles},
                  'algorithm_params': algorithm_config,
                  'recon_params': {'use_support_constraints': True,
                                   'use_positivity_constraint': True,
                                   'num_outer_iters': 20,
                                   'num_inner_iters': 30,
                                   'num_iters_in_HIO': 20,
                                   'propagate_step': propagate_step},
                  'HIO_params': {'obj_supp': obj_supp,
                                 'beta_obj': 1.,
                                 'beta_proj': 1.,
                                 'use_proj_supp': None,
                                 'proj_supp': proj_supp},
                  'record_params': {'record_freq': 1,
                                    'record_dir': '/media/Disk2/data/WedgedDataReconTest/' + test_dir + '/' +
                                    algorithm_config['algorithm'] + str(cnt)}}

        obj = new_2recon_universial(**params)
        cnt += 1

        mpp.figure((cnt + 3) * 100)
        mpp.imshow(obj[0, :])
        mpp.title('rec_wedge_sirt_no_patch')

    mpp.show()





