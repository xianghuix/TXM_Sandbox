#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:18:19 2019

@author: xiao
"""
import time, os, gc
import multiprocess as mp

import numpy as np
from scipy.ndimage import (zoom, fourier_shift, median_filter as mf,
                           gaussian_filter as gf)
from scipy.optimize import least_squares as lsq
from functools import partial

from skimage.registration import phase_cross_correlation
from pystackreg import StackReg

from .lineshapes import (gaussian, lorentzian, voigt, pvoigt, moffat, pearson7,
                         breit_wigner, damped_oscillator, dho, logistic, lognormal,
                         students_t, expgaussian, donaich, skewed_gaussian,
                         skewed_voigt, step, rectangle, exponential, powerlaw, 
                         linear, parabolic, sine, expsine, split_lorentzian)
# from larch.math.fitpeak import fit_peak


functions = {'gaussian':gaussian, 'lorentzian':lorentzian, 'voigt':voigt, 
             'pvoigt':pvoigt, 'moffat':moffat, 'pearson7':pearson7,
             'breit_wigner':breit_wigner, 'damped_oscillator':damped_oscillator,
             'dho':dho, 'logistic':logistic, 'lognormal':lognormal,
             'students_t':students_t, 'expgaussian':expgaussian, 
             'donaich':donaich, 'skewed_gaussian':skewed_gaussian,
             'skewed_voigt':skewed_voigt, 'step':step, 'rectangle':rectangle, 
             'exponential':exponential, 'powerlaw':powerlaw, 
             'linear':linear, 'parabolic':parabolic, 'sine':sine, 
             'expsine':expsine, 'split_lorentzian':split_lorentzian}

def fit_poly1d(x, y, order):
    """
    inputs:
        x:      ndarray in shape (1,), independent variable values
        y:      ndarray in shape (1,), dependent variable values at x
        order:  int, polynomial order

    returns: 
        callable polynomial function with fitting coefficients of polynomial
    """
    return np.poly1d(np.polyfit(x, y, order))

def fit_poly2d(x, y, order):
    """
    inputs:
        x:      ndarray in shape (n,), independent variable values
        y:      ndarray in shape (n, k), multiple dependent variable value arries at x
        order:  int, polynomial order

    return: 
        ndarray: fitting coefficients of polynomial in shape (order+1, y.shape[1]) 
    """
    return np.polyfit(x, y, order)

def fit_polynd(x, y, order):
    """
    inputs:
        x:      ndarray in shape (n,), independent variable values
        y:      ndarray in shape (n, ...), multiple dependent variable value arries at x
        order:  int, polynomial order

    return: 
        ndarray: fitting coefficients of polynomial in shape (order+1, y.shape[1:])
    """
    s = list(y.shape)
    s[0] = order + 1
    print('fit_polynd is done')
    return (np.polyfit(x, y.reshape([y.shape[0], -1]), order).astype(np.float32)).reshape(s)

def index_of(arr, e):
    """
    finding the element in arr that has value closes to e
    return: the indices of these elements in arr; return has shape of arr.shape[1:]
    """
    return np.argmin(abs(arr - e), axis=0)

def index_lookup(us_idx, eng_list, ufac=10):
    """
    Find the table value with its upsampled index
    upsample table's index and find the interpolated value in the table that
    corresponds to the upsampled index idx

    Parameters
    ----------
    us_idx : int
        upsampled table index.
    ref_list : 1-D array like
        list that will be looked up.
    ufac : int, optional
        index's upsampling rator. The default is 100.

    Returns
    -------
    the table's value referred by the upsampled index idx

    """
    x = np.squeeze(eng_list[np.int_(np.floor(us_idx))]) + \
        np.squeeze(eng_list[np.int_(np.ceil(us_idx))]-eng_list[np.int_(np.floor(us_idx))])*\
            np.squeeze(us_idx - np.floor(us_idx))/np.squeeze(np.ceil(us_idx) - np.floor(us_idx))
    return np.array(x).astype(np.float32)

"""
below routines adopted from xraylarch and lmfit

lmfit defines Parameter and Model classes

lmfit/lineshapes
     |
     V
lmfit/Model - > Models              Mininizer
                    |                 |
                    v-> ModelResult <-v
                            |
                            v
scipy.minimize    ->    fitting_tools
np.linalg.minizer ->         


For linear combination fitting, scipy.linalg.lstsq is used to find the weight
factors for a set of reference curves to fit a sample curve       
"""
# def fit_peak_larch(eng, spec, model, background=None, form=None, step=None,
#                    negative=False, use_gamma=False):
#     """
#     This is a parallelization of fit_peak function in Larch. The computation
#     overhead and memory footpriint are both high
    
#     fit_peak(x, y, model, dy=None, background=None, form=None, step=None,
#              negative=False, use_gamma=False)
#     arguments:
#     ---------
#     x           array of values at which to calculate model
#     y           array of values for model to try to match
#     dy          array of values for uncertainty in y data to be matched.
#     model       name of model to use.  One of (case insensitive)
#                      'linear', 'quadratic', 'step', 'rectangle',
#                       'gaussian', 'lorentzian', 'voigt', 'exponential'
#     background  name of background model to use. One of (case insensitive)
#                      None, 'constant', 'linear', or 'quadratic'
#                 this is ignored when model is 'linear' or 'quadratic'
#     form        name of form to use for 'step' and 'rectangle' models.
#                 One of (case insensitive):
#                     'linear', 'erf', or 'atan'
#     negative    True/False for whether peak or steps are expected to go down.
#     use_gamma   True/False for whether to use separate gamma parameter for
#                 voigt model.
#     output:
#     -------
#     Group with fit parameters, and more...
#     """ 
#     def fit_peak_internal(eng, spec, model, background, form, step,
#                           negative, use_gamma):
#         fit_peak(eng, spec, model, background=background, form=form, step=step,
#                  negative=negative, use_gamma=use_gamma)
#     dim = spec.shape    
#     y = spec.reshpae([dim[0], -1])
#     n_cpu = os.cpu_count()
#     ############ multiprocessing
#     with mp.Pool(n_cpu-1) as pool:
#         rlt = pool.starmap(fit_peak_internal, [(eng, y[:, ii], model, background, form, step, negative, use_gamma) for ii in np.int32(np.arange(dim[1]*dim[2]))])
#     pool.join()
#     pool.close()
#     return rlt

def fit_peak_scipy(eng, spec, model, fvars, bnds=None,
                   ftol = 1e-7, xtol = 1e-7, gtol = 1e-7,
                   jac = '3-point', method = 'trf'):
    """
    INPUTS:
        eng: 1-D array-like in shape (n,); energy point list around peak position
        
        spec: float2D, 3D, or 3D np.array in form [spec_dim, space_dim]
        
        model: lineshape name defined below
        
        fvars: list; fitting variables of model function
        
        bnds: bounding range of fitting variables
        
        ftol: Tolerance for termination by the change of the cost function
        
        xtol: Tolerance for termination by the change of the independent variables
        
        gtol: Tolerance for termination by the norm of the gradient
        jac: {‘2-point’, ‘3-point’, ‘cs’, callable}, optional; Method of 
        computing the Jacobian matrix (an m-by-n matrix, where element (i, j) 
        is the partial derivative of f[i] with respect to x[j]). The keywords 
        select a finite difference scheme for numerical estimation. The scheme 
        ‘3-point’ is more accurate, but requires twice as many operations as 
        ‘2-point’ (default). The scheme ‘cs’ uses complex steps, and while 
        potentially the most accurate, it is applicable only when fun correctly 
        handles complex inputs and can be analytically continued to the complex 
        plane. Method ‘lm’ always uses the ‘2-point’ scheme. If callable, it is 
        used as jac(x, *args, **kwargs) and should return a good approximation 
        (or the exact value) for the Jacobian as an array_like (np.atleast_2d 
        is applied), a sparse matrix or a scipy.sparse.linalg.LinearOperator.

        method: {‘trf’, ‘dogbox’, ‘lm’}, optional
                Algorithm to perform minimization.
    OUTPUTS:
        x: ndarray, shape (n,)
            Solution found.
        cost: float
            Value of the cost function at the solution.
        fun: ndarray, shape (m,)
            Vector of residuals at the solution.
        jac: ndarray, sparse matrix or LinearOperator, shape (m, n)
            Modified Jacobian matrix at the solution, in the sense that J^T J 
            is a Gauss-Newton approximation of the Hessian of the cost function. 
            The type is the same as the one used by the algorithm.
        grad: ndarray, shape (m,)
            Gradient of the cost function at the solution.
        optimality: float
            First-order optimality measure. In unconstrained problems, it is 
            always the uniform norm of the gradient. In constrained problems, 
            it is the quantity which was compared with gtol during iterations.
        active_mask: ndarray of int, shape (n,)
            Each component shows whether a corresponding constraint is active 
            (that is, whether a variable is at the bound):
            0 : a constraint is not active.
            -1 : a lower bound is active.
            1 : an upper bound is active.
            Might be somewhat arbitrary for ‘trf’ method as it generates a 
            sequence of strictly feasible iterates and active_mask is 
            determined within a tolerance threshold.
        nfev: int
            Number of function evaluations done. Methods ‘trf’ and ‘dogbox’ do 
            not count function calls for numerical Jacobian approximation, as 
            opposed to ‘lm’ method.
        njev: int or None
            Number of Jacobian evaluations done. If numerical Jacobian 
            approximation is used in ‘lm’ method, it is set to None.
        status: int
            The reason for algorithm termination:
            -1 : improper input parameters status returned from MINPACK.
            0 : the maximum number of function evaluations is exceeded.
            1 : gtol termination condition is satisfied.
            2 : ftol termination condition is satisfied.
            3 : xtol termination condition is satisfied.
            4 : Both ftol and xtol termination conditions are satisfied.
        message: str
            Verbal description of the termination reason.
        success: bool
            True if one of the convergence criteria is satisfied (status > 0).

        
    lineshpaes are inherited from lmfit.
    
    2-parameter functions:
        {'exponential', 'powerlaw', 'linear', 'parabolic'}
    exponential(x, amplitude=1, decay=1)
    powerlaw(x, amplitude=1, exponent=1.0)
    linear(x, slope=1.0, intercept=0.0)
    parabolic(x, a=0.0, b=0.0, c=0.0)
    
    3-parameter functions:
        {'gaussian', 'lorentzian', 'damped_oscillator',
         'logistic', 'lognormal', 'students_t', 'sine'}
    gaussian(x, amplitude=1.0, center=0.0, sigma=1.0)
    lorentzian(x, amplitude=1.0, center=0.0, sigma=1.0)
    damped_oscillator(x, amplitude=1.0, center=1., sigma=0.1)
    logistic(x, amplitude=1., center=0., sigma=1.)
    lognormal(x, amplitude=1.0, center=0., sigma=1)
    students_t(x, amplitude=1.0, center=0.0, sigma=1.0)
    sine(x, amplitude=1.0, frequency=1.0, shift=0.0)
    
    4-parameter functions:
        {'split_lorentzian', 'voigt', 'pvoigt', 'moffat',
         'pearson7', 'breit_wigner', 'dho', 'expgaussian',
         'donaich', 'skewed_gaussian', 'expsine', 'step'}
    split_lorentzian(x, amplitude=1.0, center=0.0, sigma=1.0, sigma_r=1.0)
    voigt(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=None)
    pvoigt(x, amplitude=1.0, center=0.0, sigma=1.0, fraction=0.5)
    moffat(x, amplitude=1, center=0., sigma=1, beta=1.)
    pearson7(x, amplitude=1.0, center=0.0, sigma=1.0, expon=1.0)
    breit_wigner(x, amplitude=1.0, center=0.0, sigma=1.0, q=1.0)
    dho(x, amplitude=1., center=0., sigma=1., gamma=1.0)
    expgaussian(x, amplitude=1, center=0, sigma=1.0, gamma=1.0)
    donaich(x, amplitude=1.0, center=0, sigma=1.0, gamma=0.0)
    skewed_gaussian(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=0.0)
    expsine(x, amplitude=1.0, frequency=1.0, shift=0.0, decay=0.0)
    step(x, amplitude=1.0, center=0.0, sigma=1.0, form='linear')
    
    5-parameter functions:
        {'skewed_voigt'}
    skewed_voigt(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=None, skew=0.0)
    
    6-parameter functions:
        {'rectangle'}
    rectangle(x, amplitude=1.0, center1=0.0, sigma1=1.0,
              center2=1.0, sigma2=1.0, form='linear')
    """
    func = functions[model]
    if len(fvars) == 2:
        def _f(fvars, func, x, y):
            return (func(x, fvars[0], fvars[1]) - y)
    elif len(fvars) == 3:
        def _f(fvars, func, x, y):
            return (func(x, fvars[0], fvars[1], fvars[2]) - y)
    elif len(fvars) == 4:
        def _f(fvars, func, x, y):
            return (func(x, fvars[0], fvars[1], fvars[2], fvars[3]) - y)
    elif len(fvars) == 5:
        def _f(fvars, func, x, y):
            return (func(x, fvars[0], fvars[1], fvars[2], fvars[3], fvars[4]) - y)
    elif len(fvars) == 6:
        def _f(fvars, func, x, y):
            return (func(x, fvars[0], fvars[1], fvars[2], fvars[3], fvars[4], fvars[5]) - y)
    else:
        return None
       
    def _my_lsq(f, fvars, func, x, y, jac, bnds, method, ftol, xtol, gtol):
        return lsq(f, fvars, jac=jac, bounds=bnds, method=method, ftol=ftol, xtol=xtol, gtol=gtol, args=(func, x, y)).x

    dim = spec.shape    
    if bnds is None:
        bnds = ((-np.inf, eng[0], 0.01), (np.inf, eng[-1], 100))
    
    print(time.asctime())    
    n_cpu = os.cpu_count()
    with mp.Pool(n_cpu - 1) as pool:
        rlt = pool.starmap(_my_lsq, [(_f, fvars, func, eng, spec[:, ii], jac, bnds, method, ftol, xtol, gtol) for ii in np.int32(np.arange(dim[1]))])
    pool.close()
    pool.join()    
    print(time.asctime())
    print('fit_peak_scipy is done')
    return np.array(rlt).astype(np.float32)

def find_edge_0p5_map_direct(spec):
    """
    inputs: 
        peak_fit_coef: array-like; pixel-wise peak fitting coefficients
        eng: array-like; dense energy point list around the peak
    returns:
        ndarray: pixel-wise edge=0.5 position map
    """
    try:
        # a = np.squeeze(spec[np.argmin(np.abs(spec-0.5), axis=0)]).astype(np.float32) 
        a = np.take_along_axis(spec, np.expand_dims(np.argmin(np.abs(spec-0.5), axis=0), axis=0), axis=0)
        print('find_fit_edge_0p5_map_poly is done')
        return a
    except:
        print('Something wrong in find_edge_0p5_map_direct')
        return -1

def find_deriv_peak_map_poly(peak_fit_coef, idx):
    """
    inputs: 
        peak_fit_coef: array-like; pixel-wise peak fitting coefficients
        eng: array-like; dense energy point list around the peak
    returns:
        ndarray: pixel-wise peak position map
    """
    try:
        order = peak_fit_coef.shape[0]
        a = 0
        for ii in range(order):
            a += peak_fit_coef[ii]*(idx**(order-ii-1)) 
        b = np.take_along_axis(idx, np.expand_dims(np.argmax(np.gradient(a, axis=0)/
                              np.gradient(idx, axis=0), axis=0), axis=0), axis=0)
        print('find_deriv_peak_map_poly is done')
        return b .astype(np.float32)       
    except:
        print('Something wrong in find_deriv_peak_map_poly')
        return -1    

def find_fit_edge_0p5_map_poly(edge_fit_coef, idx):
    """
    inputs: 
        peak_fit_coef: array-like; pixel-wise peak fitting coefficients
        eng: array-like; dense energy point list around the peak
    returns:
        ndarray: pixel-wise peak position map
    """
    try:
        order = edge_fit_coef.shape[0]
        a = 0
        for ii in range(order):
            a += edge_fit_coef[ii]*(idx**(order-ii-1)) 
        b = np.take_along_axis(idx, np.expand_dims(np.argmin(np.abs(a-0.5), axis=0), axis=0), axis=0)
        print('find_fit_edge_0p5_map_poly is done')        
        return b.astype(np.float32)       
    except:
        print('Something wrong in find_fit_edge_0p5_map_poly')
        return -1

def find_fit_peak_map_poly(peak_fit_coef, idx):
    try:
        order = peak_fit_coef.shape[0]
        a = 0
        for ii in range(order):
            a += peak_fit_coef[ii]*(idx**(order-ii-1)) 
        b = np.take_along_axis(idx, np.expand_dims(np.argmax(a, axis=0), axis=0), axis=0)
        print('find_fit_peak_map_poly is done')
        return b.astype(np.float32)
    except:
        print('Something wroing in find_fit_peak_map_poly')
        return -1    

def find_deriv_peak_map_scipy(model, eng, peak_fit_coef):
    """
    inputs: 
        model: string; line shape function name in 'functions' 
        eng: array-like; dense energy point list around the peak
        peak_fit_coef: array-like; pixel-wise peak fitting coefficients
    returns:
        ndarray: pixel-wise peak position map
    """
    func = functions[model]
    fvars = peak_fit_coef[0]
    if len(fvars) == 2:
        def _max_grad(func, x, fvars):
            return x[np.argmax(np.gradient(func(x, fvars[0], fvars[1]))/
                               np.gradient(eng))]
    elif len(fvars) == 3:
        def _max_grad(func, x, fvars):
            return x[np.argmax(np.gradient(func(x, fvars[0], fvars[1], fvars[2]))/
                               np.gradient(eng))]
    elif len(fvars) == 4:
        def _max_grad(func, x, fvars):
            return x[np.argmax(np.gradient(func(x, fvars[0], fvars[1], fvars[2], fvars[3]))/
                               np.gradient(eng))]
    elif len(fvars) == 5:
        def _max_grad(func, x, fvars):
            return x[np.argmax(np.gradient(func(x, fvars[0], fvars[1], fvars[2], fvars[3], fvars[4]))/
                               np.gradient(eng))]
    elif len(fvars) == 6:
        def _max_grad(func, x, fvars):
            return x[np.argmax(np.gradient(func(x, fvars[0], fvars[1], fvars[2], fvars[3], fvars[4], fvars[5]))/
                               np.gradient(eng))]
    else:
        return None

    n_cpu = os.cpu_count()
    print(time.asctime())
    with mp.Pool(n_cpu-1) as pool:
        derivs = pool.starmap(_max_grad, [(func, eng, peak_fit_coef[ii]) for ii in np.int32(np.arange(len(peak_fit_coef)))])
    pool.close()
    pool.join()
    print(time.asctime())
    print('find_deriv_peak_map_scipy is done')
    return np.array(derivs).astype(np.float32)

def find_fit_edge_0p5_map_scipy(model, peak_fit_coef, eng):
    """
    inputs: 
        model: string; line shape function name in 'functions' 
        eng: array-like; dense energy point list around the peak
        peak_fit_coef: array-like; pixel-wise peak fitting coefficients
    returns:
        ndarray: pixel-wise peak position map
    """
    func = functions[model]
    fvars = peak_fit_coef[0]
    if len(fvars) == 2:
        def _max_grad(func, x, fvars):
            return x[np.argmin(np.abs(func(x, fvars[0], fvars[1])-0.5))]
    elif len(fvars) == 3:
        def _max_grad(func, x, fvars):
            return x[np.argmin(np.abs(func(x, fvars[0], fvars[1], fvars[2])-0.5))]
    elif len(fvars) == 4:
        def _max_grad(func, x, fvars):
            return x[np.argmin(np.abs(func(x, fvars[0], fvars[1], fvars[2], fvars[3])-0.5))]
    elif len(fvars) == 5:
        def _max_grad(func, x, fvars):
            return x[np.argmin(np.abs(func(x, fvars[0], fvars[1], fvars[2], fvars[3], fvars[4])-0.5))]
    elif len(fvars) == 6:
        def _max_grad(func, x, fvars):
            return x[np.argmin(np.abs(func(x, fvars[0], fvars[1], fvars[2], fvars[3], fvars[4], fvars[5])-0.5))]
    else:
        return None

    n_cpu = os.cpu_count()
    print(time.asctime())
    with mp.Pool(n_cpu-1) as pool:
        edge_0p5 = pool.starmap(_max_grad, [(func, eng, peak_fit_coef[ii]) for ii in np.int32(np.arange(len(peak_fit_coef)))])
    pool.close()
    pool.join()
    print(time.asctime())
    print('find_fit_edge_0p5_map_scipy is done')
    return np.array(edge_0p5).astype(np.float32)

def find_fit_peak_map_scipy(model, eng, peak_fit_coef):
    """
    inputs: 
        model: string; line shape function name in 'functions' 
        eng: array-like; dense energy point list around the peak
        peak_fit_coef: array-like; pixel-wise peak fitting coefficients
    returns:
        ndarray: pixel-wise peak position map
    """
    func = functions[model]
    fvars = peak_fit_coef[0]
    if len(fvars) == 2:
        def _find_peak(func, x, fvars):
            return x[np.argmax(func(x, fvars[0], fvars[1]))]
            # return np.max(func(x, fvars[0], fvars[1]))
    elif len(fvars) == 3:
        def _find_peak(func, x, fvars):
            return x[np.argmax(func(x, fvars[0], fvars[1], fvars[2]))]
            # return np.max(func(x, fvars[0], fvars[1], fvars[2]))
    elif len(fvars) == 4:
        def _find_peak(func, x, fvars):
            return x[np.argmax(func(x, fvars[0], fvars[1], fvars[2], fvars[3]))]
            # return np.max(func(x, fvars[0], fvars[1], fvars[2], fvars[3]))
    elif len(fvars) == 5:
        def _find_peak(func, x, fvars):
            return x[np.argmax(func(x, fvars[0], fvars[1], fvars[2], fvars[3], fvars[4]))]
            # return np.max(func(x, fvars[0], fvars[1], fvars[2], fvars[3], fvars[4]))
    elif len(fvars) == 6:
        def _find_peak(func, x, fvars):
            return x[np.argmax(func(x, fvars[0], fvars[1], fvars[2], fvars[3], fvars[4], fvars[5]))]
            # return np.max(func(x, fvars[0], fvars[1], fvars[2], fvars[3], fvars[4], fvars[5]))
    else:
        return None

    n_cpu = os.cpu_count()
    print(time.asctime())
    with mp.Pool(n_cpu-1) as pool:
        peaks = pool.starmap(_find_peak, [(func, eng, peak_fit_coef[ii]) for ii in np.int32(np.arange(len(peak_fit_coef)))])
    pool.close()
    pool.join()
    print(time.asctime())
    print('find_fit_peak_map_scipy is done')
    return np.array(peaks).astype(np.float32)

# def tv_l1_pixel(fixed_img, img, mask, shift, norm=True):
#     if norm:
#         diff_img = fixed_img - np.roll(img, shift, axis=[0, 1])
#         diff_img /= np.sqrt((diff_img**2).sum())
#     else:
#         diff_img = fixed_img - np.roll(img, shift, axis=[0, 1])
#     return ((np.abs(np.diff(diff_img, axis=0, prepend=1))+np.abs(np.diff(diff_img, axis=1, prepend=1)))*mask).sum()
def tv_l1_pixel(fixed_img, img, mask, shift, filt=True):
    if filt:
        diff_img = gf(fixed_img, 3) - gf(np.roll(img, shift, axis=[0, 1]), 3)
        # diff_img /= np.sqrt((diff_img**2).sum())
        return ((np.abs(np.diff(diff_img, axis=0, prepend=1))+np.abs(np.diff(diff_img, axis=1, prepend=1)))*mask).sum()
    else:
        diff_img = fixed_img - np.roll(img, shift, axis=[0, 1])
        return ((np.abs(np.diff(diff_img, axis=0, prepend=1))+np.abs(np.diff(diff_img, axis=1, prepend=1)))*mask).sum()

def tv_l2_pixel(fixed_img, img, mask, shift, norm=True):
    if norm:
        diff_img = fixed_img - np.roll(img, shift, axis=[0, 1])
        diff_img /= np.sqrt((diff_img**2).sum())
    else:
        diff_img = fixed_img - np.roll(img, shift, axis=[0, 1])        
    return (((np.diff(diff_img, axis=0, prepend=1))**2+(np.diff(diff_img, axis=1, prepend=1))**2)*mask).sum() 

# def tv_l1_subpixel(fixed_img, img, mask, shift, norm=True):
#     if norm:
#         diff_img = fixed_img - np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)))
#         diff_img /= np.sqrt((diff_img**2).sum())
#     else:
#         diff_img = fixed_img - np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)))
#     return ((np.abs(np.diff(diff_img, axis=0, prepend=1))+np.abs(np.diff(diff_img, axis=1, prepend=1)))*mask).sum()
def tv_l1_subpixel(fixed_img, img, mask, shift, filt=True):
    if filt:
        diff_img = gf(fixed_img, 3) - gf(np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift))), 3)
        # diff_img /= np.sqrt((diff_img**2).sum())
        return ((np.abs(np.diff(diff_img, axis=0, prepend=1))+np.abs(np.diff(diff_img, axis=1, prepend=1)))*mask).sum()
    else:
        diff_img = fixed_img - np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)))
        return ((np.abs(np.diff(diff_img, axis=0, prepend=1))+np.abs(np.diff(diff_img, axis=1, prepend=1)))*mask).sum()

def tv_l2_subpixel(fixed_img, img, mask, shift, norm=True):
    if norm:
        diff_img = fixed_img - np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)))
        diff_img /= np.sqrt((diff_img**2).sum())
    else:
        diff_img = fixed_img - np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)))
    return (((np.diff(diff_img, axis=0, prepend=1))**2+(np.diff(diff_img, axis=1, prepend=1))**2)*mask).sum()

def mrtv_reg1(fixed_img, img, levs=4, wz=10, sp_wz=20, sp_step=0.2):
    sch_config = {}
    sch_config[levs-1] = {'wz':sp_wz, 'step':sp_step}
    tv1_pxl = {}
    tv1_pxl_id = np.zeros(levs, dtype=np.int16)
    shift = np.zeros([levs, 2], dtype=np.float32)
    rlt = []
    
    for ii in range(levs-1):
        sch_config[levs-2-ii] = {'wz':wz, 'step':int(2**ii)}
    
    border_wz = sch_config[0]['wz']*sch_config[0]['step']
    if ((np.array(fixed_img.shape) - border_wz) <= 0).any():
        return -1
    
    mask = np.zeros(fixed_img.shape)
    mask[int(border_wz/2):-int(border_wz/2), int(border_wz/2):-int(border_wz/2)] = 1
        
    n_cpu = os.cpu_count()    
    for ii in range(levs-1): 
        w = sch_config[ii]['wz']
        step = sch_config[ii]['step']
        with mp.get_context('spawn').Pool(int(n_cpu-1)) as pool:
            rlt = pool.map(partial(tv_l1_pixel, fixed_img, img, mask), 
                           [[int(step*(jj//w-int(w/2))+shift[:ii, 0].sum()), 
                             int(step*(jj%w-int(w/2))+shift[:ii, 1].sum())] for jj in np.int32(np.arange(w**2))])
        pool.close()
        pool.join()

        tem = np.ndarray([w, w], dtype=np.float32)
        for kk in range(w**2):
            tem[kk//w, kk%w] = rlt[kk]
        del(rlt)
        gc.collect()
        
        tv1_pxl[ii] = np.array(tem)
        tv1_pxl_id[ii] = tv1_pxl[ii].argmin()
        shift[ii, 0] = step*(tv1_pxl_id[ii]//w-int(w/2))
        shift[ii, 1] = step*(tv1_pxl_id[ii]%w-int(w/2))
    
    w = sch_config[levs-1]['wz']
    step = sch_config[levs-1]['step']
    with mp.get_context('spawn').Pool(int(n_cpu-1)) as pool:
        rlt = pool.map(partial(tv_l1_subpixel, fixed_img, img, mask), 
                       [[step*(jj//w-int(w/2))+shift[:(levs-1), 0].sum(), 
                         step*(jj%w-int(w/2))+shift[:(levs-1), 1].sum()] for jj in np.int32(np.arange(w**2))])
    pool.close() 
    pool.join()
    
    tem = np.ndarray([w, w], dtype=np.float32)
    for kk in range(w**2):
        tem[kk//w, kk%w] = rlt[kk]
    del(rlt)
    gc.collect()
    
    tv1_pxl[levs-1] = np.array(tem)
    tv1_pxl_id[levs-1] = tv1_pxl[levs-1].argmin()
    shift[levs-1, 0] = step*(tv1_pxl_id[levs-1]//w-int(w/2))
    shift[levs-1, 1] = step*(tv1_pxl_id[levs-1]%w-int(w/2))
    
    print(time.asctime())     
    return tv1_pxl, tv1_pxl_id, shift, shift.sum(axis=0)

# def mrtv_reg2(fixed_img, img, levs=4, wz=10, sp_wz=20, sp_step=0.2, ps=None):
#     if ps is not None:
#         img[:] = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), ps)))[:]
#     wz = int(wz)
#     sch_config = {}
#     sch_config[levs-1] = {'wz':sp_wz, 'step':sp_step}
#     tv1_pxl = {}
#     tv1_pxl_id = np.zeros(levs, dtype=np.int16)
#     shift = np.zeros([levs, 2], dtype=np.float32)
#     rlt = []
    
#     for ii in range(levs-1):
#         sch_config[levs-2-ii] = {'wz':wz, 'step':0.5**ii}

#     if ((np.array(fixed_img.shape)*0.5**(levs-2) - wz) <= 0).any():
#         return -1
    
#     n_cpu = os.cpu_count()
#     for ii in range(levs-1): 
#         w = sch_config[ii]['wz']
#         step = sch_config[ii]['step']
#         f = zoom(fixed_img, step)
#         m = zoom(img, step)
#         mk = np.zeros(m.shape, dtype=np.int8)
#         # mk[mk.shape[0]//4:-mk.shape[0]//4, mk.shape[1]//4:-mk.shape[1]//4] = 1
#         mk[int(wz*2**(ii-1)):-int(wz*2**(ii-1)), int(wz*2**(ii-1)):-int(wz*2**(ii-1))] = 1
#         # with mp.get_context('spawn').Pool(int(n_cpu-1)) as pool:
#         #     rlt = pool.map(partial(tv_l1_pixel, f, m, mk), 
#         #                    [[int((jj//w-int(w/2))/step+shift[:ii, 0].sum()), 
#         #                      int((jj%w-int(w/2))/step+shift[:ii, 1].sum())] for jj in np.int32(np.arange(w**2))])
        
#         s = np.array([0, 0])
#         for kk in range(ii):
#             s = s + shift[kk]*2**(ii-kk)
#         with mp.get_context('spawn').Pool(int(n_cpu-1)) as pool:
#             rlt = pool.map(partial(tv_l1_pixel, f, m, mk), 
#                            [[int((jj//w-int(w/2))+s[0]), 
#                              int((jj%w-int(w/2))+s[1])] for jj in np.int32(np.arange(w**2))])
#         pool.close()
#         pool.join()

#         tem = np.ndarray([w, w], dtype=np.float32)
#         for kk in range(w**2):
#             tem[kk//w, kk%w] = rlt[kk]
#         del(rlt)
#         gc.collect()
        
#         tv1_pxl[ii] = np.array(tem)
#         tv1_pxl_id[ii] = tv1_pxl[ii].argmax()
#         # shift[ii, 0] = (tv1_pxl_id[ii]//w-int(w/2))/step
#         # shift[ii, 1] = (tv1_pxl_id[ii]%w-int(w/2))/step
#         shift[ii, 0] = (tv1_pxl_id[ii]//w-int(w/2))
#         shift[ii, 1] = (tv1_pxl_id[ii]%w-int(w/2))

#     mk = np.zeros(fixed_img.shape)
#     # mk[int(sp_wz*sp_step):-max(int(sp_wz*sp_step), 1), int(sp_wz*sp_step):-max(int(sp_wz*sp_step), 1)] = 1   
#     mk[int(wz*2**(levs-2)):-int(wz*2**(levs-2)), int(wz*2**(levs-2)):-int(wz*2**(levs-2))] = 1
#     w = sch_config[levs-1]['wz']
#     step = sch_config[levs-1]['step']
#     # s= [0, 0]
#     # for kk in range(levs-1):
#     #     s += shift[kk]*2**(levs-2-kk)
#     s = s + shift[levs-2]
#     with mp.get_context('spawn').Pool(int(n_cpu-1)) as pool:
#         rlt = pool.map(partial(tv_l1_subpixel, fixed_img, img, mk), 
#                        [[step*(jj//w-int(w/2))+s[0], 
#                          step*(jj%w-int(w/2))+s[1]] for jj in np.int32(np.arange(w**2))])
#     pool.close() 
#     pool.join()
    
#     tem = np.ndarray([w, w], dtype=np.float32)
#     for kk in range(w**2):
#         tem[kk//w, kk%w] = rlt[kk]
#     del(rlt)
#     gc.collect()
    
#     tv1_pxl[levs-1] = np.array(tem)
#     tv1_pxl_id[levs-1] = tv1_pxl[levs-1].argmax()
#     shift[levs-1, 0] = step*(tv1_pxl_id[levs-1]//w-int(w/2))
#     shift[levs-1, 1] = step*(tv1_pxl_id[levs-1]%w-int(w/2))
    
#     print(time.asctime())     
#     return tv1_pxl, tv1_pxl_id, shift, s+shift[levs-1]

def mrtv_reg2(fixed_img, img, levs=4, wz=10, sp_wz=20, sp_step=0.2, ps=None):
    if ps is not None:
        img[:] = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), ps)))[:]
    wz = int(wz)
    sch_config = {}
    sch_config[levs-1] = {'wz':sp_wz, 'step':sp_step}
    tv1_pxl = {}
    tv1_pxl_id = np.zeros(levs, dtype=np.int16)
    shift = np.zeros([levs, 2], dtype=np.float32)
    rlt = []
    
    for ii in range(levs-1):
        sch_config[levs-2-ii] = {'wz':wz, 'step':0.5**ii}

    if ((np.array(fixed_img.shape)*0.5**(levs-2) - wz) <= 0).any():
        return -1
    
    n_cpu = os.cpu_count()
    for ii in range(levs-1): 
        w = sch_config[ii]['wz']
        step = sch_config[ii]['step']
        f = zoom(fixed_img, step)
        m = zoom(img, step)
        mk = np.zeros(m.shape, dtype=np.int8)
        # mk[mk.shape[0]//4:-mk.shape[0]//4, mk.shape[1]//4:-mk.shape[1]//4] = 1
        mk[int(wz*2**(ii-1)):-int(wz*2**(ii-1)), int(wz*2**(ii-1)):-int(wz*2**(ii-1))] = 1
        # with mp.get_context('spawn').Pool(int(n_cpu-1)) as pool:
        #     rlt = pool.map(partial(tv_l1_pixel, f, m, mk), 
        #                    [[int((jj//w-int(w/2))/step+shift[:ii, 0].sum()), 
        #                      int((jj%w-int(w/2))/step+shift[:ii, 1].sum())] for jj in np.int32(np.arange(w**2))])
        
        s = np.array([0, 0])
        for kk in range(ii):
            s = s + shift[kk]*2**(ii-kk)
        with mp.get_context('spawn').Pool(int(n_cpu-1)) as pool:
            rlt = pool.map(partial(tv_l1_pixel, f, m, mk), 
                           [[int((jj//w-int(w/2))+s[0]), 
                             int((jj%w-int(w/2))+s[1])] for jj in np.int32(np.arange(w**2))])
        pool.close()
        pool.join()

        tem = np.ndarray([w, w], dtype=np.float32)
        for kk in range(w**2):
            tem[kk//w, kk%w] = rlt[kk]
        del(rlt)
        gc.collect()
        
        tv1_pxl[ii] = np.array(tem)
        tv1_pxl_id[ii] = tv1_pxl[ii].argmin()
        # shift[ii, 0] = (tv1_pxl_id[ii]//w-int(w/2))/step
        # shift[ii, 1] = (tv1_pxl_id[ii]%w-int(w/2))/step
        shift[ii, 0] = (tv1_pxl_id[ii]//w-int(w/2))
        shift[ii, 1] = (tv1_pxl_id[ii]%w-int(w/2))

    mk = np.zeros(fixed_img.shape)
    # mk[int(sp_wz*sp_step):-max(int(sp_wz*sp_step), 1), int(sp_wz*sp_step):-max(int(sp_wz*sp_step), 1)] = 1   
    mk[int(wz*2**(levs-2)):-int(wz*2**(levs-2)), int(wz*2**(levs-2)):-int(wz*2**(levs-2))] = 1
    w = sch_config[levs-1]['wz']
    step = sch_config[levs-1]['step']
    # s= [0, 0]
    # for kk in range(levs-1):
    #     s += shift[kk]*2**(levs-2-kk)
    s = s + shift[levs-2]
    with mp.get_context('spawn').Pool(int(n_cpu-1)) as pool:
        rlt = pool.map(partial(tv_l1_subpixel, fixed_img, img, mk), 
                       [[step*(jj//w-int(w/2))+s[0], 
                         step*(jj%w-int(w/2))+s[1]] for jj in np.int32(np.arange(w**2))])
    pool.close() 
    pool.join()
    
    tem = np.ndarray([w, w], dtype=np.float32)
    for kk in range(w**2):
        tem[kk//w, kk%w] = rlt[kk]
    del(rlt)
    gc.collect()
    
    tv1_pxl[levs-1] = np.array(tem)
    tv1_pxl_id[levs-1] = tv1_pxl[levs-1].argmin()
    shift[levs-1, 0] = step*(tv1_pxl_id[levs-1]//w-int(w/2))
    shift[levs-1, 1] = step*(tv1_pxl_id[levs-1]%w-int(w/2))
    
    print(time.asctime())     
    return tv1_pxl, tv1_pxl_id, shift, s+shift[levs-1]
    
def mrtv_mpc_combo_reg(fixed_img, img, us=100, reference_mask=None, overlap_ratio=0.3, 
                       levs=4, wz=10, sp_wz=20, sp_step=0.2):   
    shift = np.zeros([levs+1, 2])
    if reference_mask is not None:
        shift[0] = phase_cross_correlation(fixed_img, img, upsample_factor=us, reference_mask=reference_mask, overlap_ratio=overlap_ratio)
    else:
        shift[0], _, _ = phase_cross_correlation(fixed_img, img, upsample_factor=us, overlap_ratio=overlap_ratio)
    
    # mk = np.zeros(fixed_img.shape)
    # mk[mk.shape[0]//4:-mk.shape[0]//4, mk.shape[1]//4:-mk.shape[1]//4] = 1
    # n_cpu = os.cpu_count()
    # with mp.get_context('spawn').Pool(int(n_cpu-1)) as pool:
    #     rlt = pool.map(partial(tv_l1_subpixel, fixed_img, img, mk), 
    #                    [[sp_step*(jj//sp_wz-int(sp_wz/2))+shift[0].sum(), 
    #                      sp_step*(jj%sp_wz-int(sp_wz/2))+shift[1].sum()] for jj in np.int32(np.arange(sp_wz**2))])
    # pool.close() 
    # pool.join()

    # tem = np.ndarray([sp_wz, sp_wz], dtype=np.float32)
    # for kk in range(sp_wz**2):
    #     tem[kk//sp_wz, kk%sp_wz] = rlt[kk]
    # del(rlt)
    # gc.collect()
    
    # tv1_pxl = np.array(tem)
    # tv1_pxl_id = tv1_pxl.argmin()
    # shift[1, 0] = sp_step*(tv1_pxl_id//sp_wz-int(sp_wz/2))
    # shift[1, 1] = sp_step*(tv1_pxl_id%sp_wz-int(sp_wz/2))
    
    tv1_pxl, tv1_pxl_id, shift[1:], ss = mrtv_reg2(fixed_img, img, levs=levs, wz=wz, sp_wz=sp_wz, sp_step=sp_step, ps=shift[0])

    print(time.asctime())     
    return tv1_pxl, tv1_pxl_id, shift, shift[0]+ss

def _pc(fixed_img, upsample_factor, img):
    shift, _, _ = phase_cross_correlation(fixed_img, img, upsample_factor=upsample_factor)
    img[:] = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)))[:]
    return shift, img

def mp_pc(fixed_img, upsample_factor, img_stack):
    n_cpu = os.cpu_count()
    with mp.Pool(n_cpu-1) as pool:
        rlt = pool.map(partial(_pc, fixed_img, upsample_factor), 
                       [img_stack[ii] for ii in range(img_stack.shape[0])])
    pool.close() 
    pool.join()

    # shift = []
    # for kk in range(img_stack.shape[0]):
    #     img_stack[kk//sp_wz, kk%sp_wz] = rlt[kk]
    # del(rlt)
    # gc.collect()
    return list(rlt[0]), np.array(rlt[1])

def _mpc(fixed_img, reference_mask, overlap_ratio, img):
    shift = phase_cross_correlation(fixed_img, img, reference_mask=reference_mask, 
                                          overlap_ratio=overlap_ratio)
    img[:] = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)))[:]
    return shift, img

def mp_mpc(fixed_img, reference_mask, overlap_ratio, img_stack):
    n_cpu = os.cpu_count()
    with mp.Pool(n_cpu-1) as pool:
        rlt = pool.map(partial(_mpc, fixed_img, reference_mask, overlap_ratio), 
                       [img_stack[ii] for ii in range(img_stack.shape[0])])
    pool.close() 
    pool.join()
    return list(rlt[0]), np.array(rlt[1])
        
def _sr(sr, fixed_img, mask, img):
    shift = sr.register(fixed_img*mask, img*mask)
    img[:] = sr.transform(img, shift)[:]
    return shift, img

def mp_sr(mode, fixed_img, mask, img_stack):
    if mode.upper() == 'TRANSLATION':
        sr = StackReg(StackReg.TRANSLATION)
    elif  mode.upper() == 'RIGID_BODY':
        sr = StackReg(StackReg.RIGID_BODY)
    elif  mode.upper() == 'SCALED_ROTATION':
        sr = StackReg(StackReg.SCALED_ROTATION)
    elif  mode.upper() == 'AFFINE':
        sr = StackReg(StackReg.AFFINE)
    elif  mode.upper() == 'BILINEAR':
        sr = StackReg(StackReg.BILINEAR)
                    
    n_cpu = os.cpu_count()
    with mp.Pool(n_cpu-1) as pool:
        rlt = pool.map(partial(_sr, sr, fixed_img, mask), 
                       [img_stack[ii] for ii in range(img_stack.shape[0])])
    pool.close() 
    pool.join()
    return list(rlt[0]), np.array(rlt[1])