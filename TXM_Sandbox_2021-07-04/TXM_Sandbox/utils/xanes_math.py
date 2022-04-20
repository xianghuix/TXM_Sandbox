#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:18:19 2019

@author: xiao
"""
import time
import os
import gc
import multiprocess as mp

import numpy as np
from scipy.ndimage import (zoom, fourier_shift, median_filter as mf, shift as sshift,
                           gaussian_filter as gf)
from scipy.optimize import least_squares as lsq, lsq_linear, minimize, Bounds, LinearConstraint
from functools import partial

from skimage.registration import phase_cross_correlation
from sklearn.neighbors import KernelDensity
from pystackreg import StackReg
from itertools import chain

from .lineshapes import (gaussian, lorentzian, voigt, pvoigt, moffat, pearson7,
                         breit_wigner, damped_oscillator, dho, logistic, lognormal,
                         students_t, expgaussian, donaich, skewed_gaussian,
                         skewed_voigt, step, rectangle, exponential, powerlaw,
                         linear, parabolic, sine, expsine, split_lorentzian, polynd)

from .misc import msgit


functions = {'gaussian': gaussian, 'lorentzian': lorentzian, 'voigt': voigt,
             'pvoigt': pvoigt, 'moffat': moffat, 'pearson7': pearson7,
             'breit_wigner': breit_wigner, 'damped_oscillator': damped_oscillator,
             'dho': dho, 'logistic': logistic, 'lognormal': lognormal,
             'students_t': students_t, 'expgaussian': expgaussian,
             'donaich': donaich, 'skewed_gaussian': skewed_gaussian,
             'skewed_voigt': skewed_voigt, 'step': step, 'rectangle': rectangle,
             'exponential': exponential, 'powerlaw': powerlaw,
             'linear': linear, 'parabolic': parabolic, 'sine': sine,
             'expsine': expsine, 'split_lorentzian': split_lorentzian,
             'polynd': polynd}

N_CPU = os.cpu_count()-1


def index_of(arr, e):
    """
    finding the element in arr that has value closes to e

    Parameters
    ----------
    arr : 1D or 2D ndarray of size [NxM]
        if arr is 1D it is a spectrum of N points
        if arr is 2D, its column is the spectrum dimension (N) and row is the number of spectra dimension (M)
        search is performed on each column.
    e : float or 1D ndarray of size M; float corresponds to M=1
        if e is 1D, its size equals to arr.shape[1]
        target value to be compared.

    Returns
    -------
    list
        the indices of the elements closest to e in arr; has size of arr.shape[1]

    """
    if len(arr.shape) == 1:
        return np.argmin(abs(arr - e), axis=0)
    elif len(arr.shape) == 2:
        return np.argmin(abs(arr - e[np.newaxis, :]), axis=0)


def lookup(x, y, y0):
    """
    Find corresponding x according to y0 relative in y

    Parameters
    ----------
    x : 1D array-like of size N
        corresponding energy points in the spectra
    y : 2D array-like of size NxM
        its column dimension is the spectrum dimension (N) and row dimension the sampling dimension (M)
    y0 : 1D array-like of size M
        specific values for each spectrum in y.

    Returns
    -------
    float
        x value corresponding to y0.

    """
    return x[:, np.newaxis][index_of(y, y0)]


@msgit(wd=100, fill='-')
def eval_polynd(p, x, reshape=None):
    bdi = _chunking(p.shape[1])
    with mp.Pool(N_CPU) as pool:
        rlt = pool.map(partial(_polyval, x), [
                       p[:, bdi[ii]:bdi[ii+1]] for ii in range(N_CPU)])
    pool.close()
    pool.join()
    if reshape is None:
        return np.vstack(list(chain(*rlt))).T
    else:
        return np.vstack(list(chain(*rlt))).T.reshape(reshape)


def _polyval(x0, p0):
    rlt = [np.polyval(p0[:, ii], x0) for ii in range(p0.shape[1])]
    return rlt


@msgit(wd=100, fill='-')
def fit_curv_polynd(x, y, order):
    """
    inputs:
        x:      ndarray in shape (n,), independent variable values
        y:      ndarray in shape (n, ...), multiple dependent variable value arries at x
        order:  int, polynomial order

    return:
        ndarray: fitting coefficients of polynomial in shape (order+1, y.shape[1:])
    """
    y = y.reshape(y.shape[0], -1)
    bdi = _chunking(y.shape[1])
    with mp.Pool(N_CPU) as pool:
        rlt = pool.map(partial(_polyfit, x, order, None, True), [
                       y[:, bdi[ii]:bdi[ii+1]] for ii in range(N_CPU)])
    pool.close()
    pool.join()

    rlt = np.array(rlt, dtype=object)
    print(rlt.shape)
    return [np.concatenate(rlt[:, 0], axis=1), np.concatenate(rlt[:, 1], axis=0), None, None]
    # return [np.vstack(rlt[:, 0]).T, np.vstack(rlt[:, 1])]


def _polyfit(x0, order, rcond, full, y0):
    rlt = np.polyfit(x0, y0, order, rcond=rcond, full=full)
    return (rlt[0], rlt[1]/np.sum(y0**2))


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


@msgit(wd=100, fill='-')
def fit_curv_scipy(x, y, model, fvars, bnds=None,
                   ftol=1e-7, xtol=1e-7, gtol=1e-7,
                   jac='3-point', method='trf'):
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
        def _f(fvar, fun, x0, y0):
            return (fun(x0, fvar[0], fvar[1]) - y0)
    elif len(fvars) == 3:
        def _f(fvar, fun, x0, y0):
            return (fun(x0, fvar[0], fvar[1], fvar[2]) - y0)
    elif len(fvars) == 4:
        def _f(fvar, fun, x0, y0):
            return (fun(x0, fvar[0], fvar[1], fvar[2], fvar[3]) - y0)
    elif len(fvars) == 5:
        def _f(fvar, fun, x0, y0):
            return (fun(x0, fvar[0], fvar[1], fvar[2], fvar[3], fvar[4]) - y0)
    elif len(fvars) == 6:
        def _f(fvar, fun, x0, y0):
            return (fun(x0, fvar[0], fvar[1], fvar[2], fvar[3], fvar[4], fvar[5]) - y0)
    else:
        return None

    dim = y.reshape([y.shape[0], -1]).shape[1]
    if bnds is None:
        bnds = ((-np.inf, x[0], 0.01), (np.inf, x[-1], 100))

    with mp.Pool(N_CPU) as pool:
        rlt = pool.map(partial(_lsq, _f, fvars, func, x, jac, bnds, method, ftol, xtol, gtol), [
                       y.reshape([y.shape[0], -1])[:, ii] for ii in range(dim)])
    pool.close()
    pool.join()
    rlt = np.array(rlt, dtype=object)
    return [np.vstack(rlt[:, 0]).T, np.vstack(rlt[:, 1]), np.vstack(rlt[:, 2]), np.vstack(rlt[:, 3])]


def _lsq(f, fvar, fun, x0, jac, bnds, method, ftol, xtol, gtol, y0):
    rlt = lsq(f, fvar, jac=jac, bounds=bnds, method=method,
              ftol=ftol, xtol=xtol, gtol=gtol, args=(fun, x0, y0))
    return (rlt.x.astype(np.float32), rlt.cost/np.sum(y0**2), rlt.status, np.int8(rlt.success))


@msgit(wd=100, fill='-')
def find_raw_val(spec, val=0.5):
    """
    inputs:
        peak_fit_coef: array-like; pixel-wise peak fitting coefficients
        eng: array-like; dense energy point list around the peak
    returns:
        ndarray: pixel-wise edge=0.5 position map
    """
    return spec[np.argmin(np.abs(spec-val), axis=0)]


# @msgit(wd=100, fill='-')
# def find_deriv_peak_poly(p, x):
#     """
#     inputs:
#         :param p: array-like; spec array
#         :param x: array-like; energy point list around the peak
#     returns:
#         ndarray: pixel-wise peak position map
#     """
#     x_grad = np.gradient(x, axis=0)
#     bdi = _chunking(p.shape[1])
#     with mp.Pool(N_CPU) as pool:
#         rlt = pool.map(partial(_max_poly_grad, x, x_grad), [
#                        p[:, bdi[ii]:bdi[ii+1]] for ii in range(N_CPU)])
#     pool.close()
#     pool.join()
#     return np.vstack(list(chain(*rlt))).astype(np.float32)
#
#
# def _max_poly_grad(x0, xg, coef):
#     rlt0 = []
#     for ii in range(coef.shape[1]):
#         c = np.polyval(coef[:, ii], x0)
#         rlt0.append([x0[np.argmax(np.gradient(c, axis=0)/xg, axis=0)]])
#     return rlt0


# @msgit(wd=100, fill='-')
# def find_fit_val_poly(p, x, v=0.5):
#     """
#     Find the indices to 'x' where the N polynomials have values closest to the
#     target value 'v'
#
#     Parameters
#     ----------
#     p : MxN array; polynomial coefficients; M: polynomial order + 1, N: number
#         of fitting points.
#     x : locations where polynomial values are calculated.
#     v : float, optional
#         target value. The default is 0.5.
#
#     Returns
#     -------
#     array of size N
#         the indices to 'x' where the N polynomials have values closest to the
#         target value 'v'.
#
#     """
#     bdi = _chunking(p.shape[1])
#     with mp.Pool(N_CPU) as pool:
#         rlt = pool.map(partial(_find_poly_v, x, v), [
#                        p[:, bdi[ii]:bdi[ii+1]] for ii in range(N_CPU)])
#     pool.close()
#     pool.join()
#     return np.vstack(list(chain(*rlt))).astype(np.float32)
#
#
# def _find_poly_v(x0, v0, coef):
#     rlt0 = []
#     for ii in range(coef.shape[1]):
#         cur = np.polyval(coef[:, ii], x0)
#         rlt0.append(x0[np.argmin(np.abs(cur-v0), axis=0)])
#     return rlt0


# @msgit(wd=100, fill='-')
# def find_fit_peak_poly(p, x):
#     """
#     :param p: array-like; spec array
#     :param x: array-like; energy point list around the peak
#     :return:
#     """
#     bdi = _chunking(p.shape[1])
#     with mp.Pool(N_CPU) as pool:
#         rlt = pool.map(partial(_find_poly_peak, x), [
#                        p[:, bdi[ii]:bdi[ii+1]] for ii in range(N_CPU)])
#     pool.close()
#     pool.join()
#     return np.vstack(list(chain(*rlt))).astype(np.float32)
#
#
# def _find_poly_peak(x0, coef):
#     rlt0 = []
#     for ii in range(coef.shape[1]):
#         c = np.polyval(coef[:, ii], x0)
#         rlt0.append(x0[np.argmax(c)])
#     return rlt0


# @msgit(wd=100, fill='-')
# def find_deriv_peak_scipy(model, p, x):
#     """
#     inputs:
#         model: string; line shape function name in 'functions'
#         p: array-like; spec array
#         x: array-like; energy point list around the peak
#     returns:
#         ndarray: pixel-wise peak position map
#     """
#     func = functions[model]
#     bdi = _chunking(p.shape[1])
#     with mp.Pool(N_CPU) as pool:
#         rlt = pool.map(partial(_max_fun_grad, func, x), [
#                        p[:, bdi[ii]:bdi[ii+1]] for ii in range(N_CPU)])
#     pool.close()
#     pool.join()
#     return np.vstack(list(chain(*rlt))).astype(np.float32)


@msgit(wd=100, fill='-')
def find_deriv_peak(model, p, x):
    """
    inputs:
        model: string; line shape function name in 'functions'
        p: array-like; spec array
        x: array-like; energy point list around the peak
    returns:
        ndarray: pixel-wise peak position map
    """
    func = functions[model]
    bdi = _chunking(p.shape[1])
    with mp.Pool(N_CPU) as pool:
        rlt = pool.map(partial(_max_fun_grad, func, x), [
                       p[:, bdi[ii]:bdi[ii+1]] for ii in range(N_CPU)])
    pool.close()
    pool.join()
    return np.vstack(list(chain(*rlt))).astype(np.float32)


# def _max_fun_grad(f, x0, fvars):
#     rlt0 = []
#     x_grad = np.gradient(x0)
#
#     for ii in range(fvars.shape[1]):
#         rlt0.append(x0[np.argmax(np.gradient(f(x0, *fvars[:, ii]))/x_grad)])
#     return rlt0


def _max_fun_grad(f, x0, fvars):
    rlt0 = []
    x_grad = np.gradient(x0)
    for ii in range(fvars.shape[1]):
        dmu = np.gradient(f(x0, *fvars[:, ii])) / x_grad
        nmin = max(3, int(len(dmu) * 0.05))
        maxdmu = max(dmu[nmin:-nmin])
        high_deriv_pts = np.where(dmu > maxdmu * 0.1)[0]
        idmu_max, dmu_max = 0, 0
        for i in high_deriv_pts:
            if i < nmin or i > len(x0) - nmin:
                continue
            if (dmu[i] > dmu_max and
                    (i + 1 in high_deriv_pts) and
                    (i - 1 in high_deriv_pts)):
                idmu_max, dmu_max = i, dmu[i]
        rlt0.append(x0[idmu_max])
    return rlt0


# @msgit(wd=100, fill='-')
# def find_fit_val_scipy(model, p, x, v=0.5):
#     """
#     inputs:
#         model: string; line shape function name in 'functions'
#         p: array-like; spec array
#         x: array-like; energy point list around the peak
#     returns:
#         ndarray: pixel-wise peak position map
#     """
#     func = functions[model]
#     bdi = _chunking(p.shape[1])
#     with mp.Pool(N_CPU) as pool:
#         rlt = pool.map(partial(_find_fun_v, func, x, v), [
#                        p[:, bdi[ii]:bdi[ii+1]] for ii in range(N_CPU)])
#     pool.close()
#     pool.join()
#     return np.vstack(list(chain(*rlt))).astype(np.float32)


@msgit(wd=100, fill='-')
def find_fit_val(model, p, x, v=0.5):
    """
    inputs:
        model: string; line shape function name in 'functions'
        p: array-like; spec array
        x: array-like; energy point list around the peak
    returns:
        ndarray: pixel-wise peak position map
    """
    func = functions[model]
    bdi = _chunking(p.shape[1])
    with mp.Pool(N_CPU) as pool:
        rlt = pool.map(partial(_find_fun_v, func, x, v), [
                       p[:, bdi[ii]:bdi[ii+1]] for ii in range(N_CPU)])
    pool.close()
    pool.join()
    return np.vstack(list(chain(*rlt))).astype(np.float32)


def _find_fun_v(f, x0, v0, fvars):
    rlt0 = []
    for ii in range(fvars.shape[1]):
        rlt0.append(x0[np.argmin(np.abs(f(x0, *fvars[:, ii])-v0))])
    return rlt0


# @msgit(wd=100, fill='-')
# def find_fit_peak_scipy(model, p, x):
#     """
#     inputs:
#         model: string; line shape function name in 'functions'
#         p: array-like; spec array
#         x: array-like; energy point list around the peak
#     returns:
#         ndarray: pixel-wise peak position map
#     """
#     func = functions[model]
#     bdi = _chunking(p.shape[1])
#     with mp.Pool(N_CPU) as pool:
#         rlt = pool.map(partial(_find_fun_peak, func, x), [
#                        p[:, bdi[ii]:bdi[ii+1]] for ii in range(N_CPU)])
#     pool.close()
#     pool.join()
#     return np.vstack(list(chain(*rlt))).astype(np.float32)


@msgit(wd=100, fill='-')
def find_fit_peak(model, p, x):
    """
    inputs:
        model: string; line shape function name in 'functions'
        p: array-like; spec array
        x: array-like; energy point list around the peak
    returns:
        ndarray: pixel-wise peak position map
    """
    func = functions[model]
    bdi = _chunking(p.shape[1])
    with mp.Pool(N_CPU) as pool:
        rlt = pool.map(partial(_find_fun_peak, func, x), [
                       p[:, bdi[ii]:bdi[ii+1]] for ii in range(N_CPU)])
    pool.close()
    pool.join()
    return np.vstack(list(chain(*rlt))).astype(np.float32)


def _find_fun_peak(f, x0, fvars):
    rlt0 = []
    for ii in range(fvars.shape[1]):
        rlt0.append(x0[np.argmax(f(x0, *fvars[:, ii]))])
    return rlt0


@msgit(wd=100, fill='-')
def find_50_peak(model_e, x_e, p_e, model_p, x_p, p_p, ftype='both'):
    """ find energy where model_e's value is 50% of the maximum of model_p
    inputs:
        model_e: string; line shape function name in 'functions' for edge fitting
        x_e: array-like; energy point list for edge fitting
        model_p: string; line shape function name in 'functions' for peak fitting
        x_p: array-like; energy point list for peak fitting
        p: [2 x N] array-like; p[0]: edge fitting function arguments
                               p[1]: peak fitting function arguments
    returns:
        ndarray: pixel-wise peak position map
    """
    if ftype == 'both':
        func_e = functions[model_e]
        func_p = functions[model_p]
        bdi = _chunking(len(p_e[0]))
        # print(f'{x_e.shape=}, {x_p.shape=}')
        # print(f'{p_e.shape=}, {p_p.shape=}')
        # print(f'{p_e[:, 0]=}')
        # print(f'{model_e=}, {model_p=}')
        with mp.Pool(N_CPU) as pool:
            rlt = pool.map(partial(_find_50_peak_fit_both, func_e, x_e, func_p, x_p), [
                           [p_e[:, bdi[ii]:bdi[ii+1]], p_p[:, bdi[ii]:bdi[ii+1]]] for ii in range(N_CPU)])
        pool.close()
        pool.join()
        return np.vstack(list(chain(*rlt))).astype(np.float32)
    elif ftype == 'wl':
        # x_e: coordinate where (model_e, p_e) to be interpolated
        # model_e: independent variable (energy points in a measured spectrum)
        # p_e: dependent variable (chi at each energy point in a measured spectrum)
        func_p = functions[model_p]
        bdi = _chunking(len(p_e[0]))
        with mp.Pool(N_CPU) as pool:
            rlt = pool.map(partial(_find_50_peak_fit_wl, model_e, x_e, func_p, x_p), [
                           [p_e[:, bdi[ii]:bdi[ii + 1]], p_p[:, bdi[ii]:bdi[ii + 1]]] for ii in range(N_CPU)])
        pool.close()
        pool.join()
        return np.vstack(list(chain(*rlt))).astype(np.float32)
    elif ftype == 'edge':
        # x_p: coordinate where (model_e, p_e) to be interpolated
        # model_p: independent variable (energy points in a measured spectrum)
        # p_p: dependent variable (chi at each energy point in a measured spectrum)
        func_e = functions[model_e]
        bdi = _chunking(len(p_e[0]))
        with mp.Pool(N_CPU) as pool:
            rlt = pool.map(partial(_find_50_peak_fit_edge, func_e, x_e, model_p, x_p), [
                           [p_e[:, bdi[ii]:bdi[ii + 1]], p_p[:, bdi[ii]:bdi[ii + 1]]] for ii in range(N_CPU)])
        pool.close()
        pool.join()
        return np.vstack(list(chain(*rlt))).astype(np.float32)
    elif ftype == 'none':
        # model_e: coordinate where (model_e, p_e) to be interpolated
        # x_e: independent variable (energy points in a measured spectrum)
        # p_e: dependent variable (chi at each energy point in a measured spectrum)
        # model_p: coordinate where (model_e, p_e) to be interpolated
        # x_p: independent variable (energy points in a measured spectrum)
        # p_p: dependent variable (chi at each energy point in a measured spectrum)
        bdi = _chunking(len(p_e[0]))
        with mp.Pool(N_CPU) as pool:
            rlt = pool.map(partial(_find_50_peak_fit_none, model_e, x_e, model_p, x_p), [
                           [p_e[:, bdi[ii]:bdi[ii + 1]], p_p[:, bdi[ii]:bdi[ii + 1]]] for ii in range(N_CPU)])
        pool.close()
        pool.join()
        return np.vstack(list(chain(*rlt))).astype(np.float32)


def _find_50_peak_fit_both(f0, x0, f1, x1, fvars):
    rlt0 = []
    for ii in range(fvars[0].shape[1]):
        # print(f'{fvars[0][:, ii]=}, {x0.shape=}, {fvars[1][:, ii]=}, {x1.shape=}')
        rlt0.append(x0[np.argmin(np.abs(f0(x0, *fvars[0][:, ii]) - 0.5*np.max(f1(x1, *fvars[1][:, ii]))))])
    return rlt0


def _find_50_peak_fit_wl(x, x0, f1, x1, fvars):
    rlt0 = []
    # print(f"{x.shape=}, {x0.shape=}, {fvars[0].shape=}, {x1.shape=}, {fvars[1].shape=}")
    for ii in range(fvars[0].shape[1]):
        rlt0.append(x[np.argmin(np.abs(np.interp(x, x0, fvars[0][:, ii]) - 0.5*np.max(f1(x1, *fvars[1][:, ii]))))])
    return rlt0


def _find_50_peak_fit_edge(f0, x0, y, y0, fvars):
    rlt0 = []
    for ii in range(fvars[0].shape[1]):
        rlt0.append(x0[np.argmin(np.abs(f0(x0, *fvars[0][:, ii]) - 0.5*np.max(np.interp(y, y0, fvars[1][:, ii]))))])
    return rlt0


def _find_50_peak_fit_none(x, x0, y, y0, fvars):
    rlt0 = []
    for ii in range(fvars[0].shape[1]):
        rlt0.append(x[np.argmin(np.abs(np.interp(x, x0, fvars[0][:, ii]) - 0.5*np.max(np.interp(y, y0, fvars[1][:, ii]))))])
    return rlt0


@msgit(wd=100, fill='-')
def lcf(ref, spec, constr=True, tol=1e-7):
    """

    :param ref:
    :param spec:
    :param const:
    :param kwargs:
    :return:
    """
    dim = spec.reshape([spec.shape[0], -1]).shape[1]
    # print(ref.shape, spec.shape, dim)
    if constr:
        bnds = Bounds(np.zeros(ref.shape[1]), np.ones(ref.shape[1]))
        eq_constr = [LinearConstraint(np.ones([1, ref.shape[1]]), [1], [1])]
        with mp.Pool(N_CPU) as pool:
            rlt = pool.map(partial(_lcf_constr_minimizer, ref, bnds, eq_constr, tol), [
                spec.reshape([spec.shape[0], -1])[:, ii] for ii in range(dim)])
        pool.close()
        pool.join()
    else:
        bnds = Bounds(np.zeros(ref.shape[1]), np.ones(ref.shape[1]))
        with mp.Pool(N_CPU) as pool:
            rlt = pool.map(partial(_lcf_unconstr_minimizer, ref, bnds, tol), [
                spec.reshape([spec.shape[0], -1])[:, ii] for ii in range(dim)])
        pool.close()
        pool.join()
    rlt = np.array(rlt, dtype=object)
    return [np.vstack(rlt[:, 0]).T, np.vstack(rlt[:, 1]), np.vstack(rlt[:, 2]), np.vstack(rlt[:, 3])]


def _lcf_unconstr_minimizer(ref, bnds, tol, spec):
    rlt = lsq_linear(ref, spec, bounds=bnds, tol=tol)
    return (rlt.x.astype(np.float32), rlt.fun/np.sum(spec**2), rlt.status, np.int8(rlt.success))


def _lcf_constr_minimizer(ref, bnds, constr, tol, spec):
    rlt = minimize(_lin_fun, 0.5*np.ones(ref.shape[1]), args=(ref, spec),
                   method='trust-constr', bounds=bnds, constraints=constr, tol=tol)
    return (rlt.x.astype(np.float32), rlt.fun/np.sum(spec**2), rlt.status, np.int8(rlt.success))


def _lin_fun(x, A, b):
    return ((np.matmul(A, x) - b)**2).sum()


def tv_l1_pixel(fixed_img, img, mask, shift, filt=True, kernel=3):
    if filt:
        diff_img = gf(fixed_img, kernel) - \
            gf(np.roll(img, shift, axis=[0, 1]), kernel)
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
    return (((np.diff(diff_img, axis=0, prepend=1))**2 +
             (np.diff(diff_img, axis=1, prepend=1))**2)*mask).sum()


def tv_l1_subpixel(fixed_img, img, mask, shift, filt=True):
    if filt:
        diff_img = (gf(fixed_img, 3) -
                    np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(gf(img, 3)), shift))))
        return ((np.abs(np.diff(diff_img, axis=0, prepend=1)) +
                 np.abs(np.diff(diff_img, axis=1, prepend=1)))*mask).sum()
    else:
        diff_img = fixed_img - \
            np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)))
        return ((np.abs(np.diff(diff_img, axis=0, prepend=1)) +
                 np.abs(np.diff(diff_img, axis=1, prepend=1)))*mask).sum()


def tv_l1_subpixel_fit(x0, x1, tvn):
    coef = np.polyfit(x0, tvn[0], 2)
    tvx = x1[np.argmin(np.polyval(coef, x1))]
    coef = np.polyfit(x0, tvn[1], 2)
    tvy = x1[np.argmin(np.polyval(coef, x1))]
    return [tvx, tvy]


def tv_l1_subpixel_ana(x3, x21, tv3):
    a = (tv3[0][0]+tv3[0][2]-2*tv3[0][1])/2
    b = (tv3[0][2]-tv3[0][0])/2
    c = tv3[0][1]
    tvy = x21[np.argmin(a*x21**2 + b*x21 + c)]
    a = (tv3[1][0]+tv3[1][2]-2*tv3[1][1])/2
    b = (tv3[1][2]-tv3[1][0])/2
    c = tv3[1][1]
    tvx = x21[np.argmin(a*x21**2 + b*x21 + c)]
    return [tvy, tvx]


def tv_l2_subpixel(fixed_img, img, mask, shift, norm=True):
    if norm:
        diff_img = fixed_img - \
            np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)))
        diff_img /= np.sqrt((diff_img**2).sum())
    else:
        diff_img = fixed_img - \
            np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)))
    return (((np.diff(diff_img, axis=0, prepend=1))**2 +
             (np.diff(diff_img, axis=1, prepend=1))**2)*mask).sum()


def mrtv_reg1(fixed_img, img, levs=4, wz=10, sp_wz=20, sp_step=0.2):
    wz = int(wz)
    sch_config = {}
    sch_config[levs-1] = {'wz': sp_wz, 'step': sp_step}
    tv1_pxl = {}
    tv1_pxl_id = np.zeros(levs, dtype=np.int16)
    shift = np.zeros([levs, 2], dtype=np.float32)
    rlt = []

    for ii in range(levs-1):
        sch_config[levs-2-ii] = {'wz': wz, 'step': int(2**ii)}

    border_wz = sch_config[0]['wz']*sch_config[0]['step']
    if ((np.array(fixed_img.shape) - border_wz) <= 0).any():
        return -1

    mask = np.zeros(fixed_img.shape)
    mask[int(border_wz/2):-int(border_wz/2),
         int(border_wz/2):-int(border_wz/2)] = 1

    for ii in range(levs-1):
        w = sch_config[ii]['wz']
        step = sch_config[ii]['step']
        with mp.get_context('spawn').Pool(N_CPU) as pool:
            rlt = pool.map(partial(tv_l1_pixel, fixed_img, img, mask),
                           [[int(step*(jj//w-int(w/2))+shift[:ii, 0].sum()),
                             int(step*(jj % w-int(w/2))+shift[:ii, 1].sum())]
                            for jj in np.int32(np.arange(w**2))])
        pool.close()
        pool.join()

        tem = np.ndarray([w, w], dtype=np.float32)
        for kk in range(w**2):
            tem[kk//w, kk % w] = rlt[kk]
        del(rlt)
        gc.collect()

        tv1_pxl[ii] = np.array(tem)
        tv1_pxl_id[ii] = tv1_pxl[ii].argmin()
        shift[ii, 0] = step*(tv1_pxl_id[ii]//w-int(w/2))
        shift[ii, 1] = step*(tv1_pxl_id[ii] % w-int(w/2))

    w = sch_config[levs-1]['wz']
    step = sch_config[levs-1]['step']
    with mp.get_context('spawn').Pool(N_CPU) as pool:
        rlt = pool.map(partial(tv_l1_subpixel, fixed_img, img, mask),
                       [[step*(jj//w-int(w/2))+shift[:(levs-1), 0].sum(),
                         step*(jj % w-int(w/2))+shift[:(levs-1), 1].sum()]
                        for jj in np.int32(np.arange(w**2))])
    pool.close()
    pool.join()

    tem = np.ndarray([w, w], dtype=np.float32)
    for kk in range(w**2):
        tem[kk//w, kk % w] = rlt[kk]
    del(rlt)
    gc.collect()

    tv1_pxl[levs-1] = np.array(tem)
    tv1_pxl_id[levs-1] = tv1_pxl[levs-1].argmin()
    shift[levs-1, 0] = step*(tv1_pxl_id[levs-1]//w-int(w/2))
    shift[levs-1, 1] = step*(tv1_pxl_id[levs-1] % w-int(w/2))

    print(time.asctime())
    return tv1_pxl, tv1_pxl_id, shift, shift.sum(axis=0)


def mrtv_reg2(fixed_img, img, levs=4, wz=10, sp_wz=20, sp_step=0.2, ps=None):
    if ps is not None:
        img[:] = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), ps)))[:]
    wz = int(wz)
    sch_config = {}
    sch_config[levs-1] = {'wz': sp_wz, 'step': sp_step}
    tv1_pxl = {}
    tv1_pxl_id = np.zeros(levs, dtype=np.int16)
    shift = np.zeros([levs, 2], dtype=np.float32)
    rlt = []

    for ii in range(levs-1):
        sch_config[levs-2-ii] = {'wz': 6, 'step': 0.5**ii}
    sch_config[0] = {'wz': wz, 'step': 0.5**(levs-2)}

    if ((np.array(fixed_img.shape)*0.5**(levs-2) - wz) <= 0).any():
        return -1

    for ii in range(levs-1):
        w = sch_config[ii]['wz']
        step = sch_config[ii]['step']
        f = zoom(fixed_img, step)
        m = zoom(img, step)
        mk = np.zeros(m.shape, dtype=np.int8)
        mk[int(wz*2**(ii-1)):-int(wz*2**(ii-1)),
           int(wz*2**(ii-1)):-int(wz*2**(ii-1))] = 1

        s = np.array([0, 0])
        for kk in range(ii):
            s = s + shift[kk]*2**(ii-kk)
        with mp.get_context('spawn').Pool(N_CPU) as pool:
            rlt = pool.map(partial(tv_l1_pixel, f, m, mk),
                           [[int((jj//w-int(w/2))+s[0]),
                             int((jj % w-int(w/2))+s[1])]
                            for jj in np.int32(np.arange(w**2))])
        pool.close()
        pool.join()

        tem = np.ndarray([w, w], dtype=np.float32)
        for kk in range(w**2):
            tem[kk//w, kk % w] = rlt[kk]
        del(rlt)
        gc.collect()

        tv1_pxl[ii] = np.array(tem)
        tv1_pxl_id[ii] = tv1_pxl[ii].argmin()
        shift[ii, 0] = (tv1_pxl_id[ii]//w-int(w/2))
        shift[ii, 1] = (tv1_pxl_id[ii] % w-int(w/2))

    mk = np.zeros(fixed_img.shape)
    mk[int(wz*2**(levs-2)):-int(wz*2**(levs-2)),
       int(wz*2**(levs-2)):-int(wz*2**(levs-2))] = 1
    w = sch_config[levs-1]['wz']
    step = sch_config[levs-1]['step']
    s = s + shift[levs-2]
    with mp.get_context('spawn').Pool(N_CPU) as pool:
        rlt = pool.map(partial(tv_l1_subpixel, fixed_img, img, mk),
                       [[step*(jj//w-int(w/2))+s[0],
                         step*(jj % w-int(w/2))+s[1]]
                        for jj in np.int32(np.arange(w**2))])
    pool.close()
    pool.join()

    tem = np.ndarray([w, w], dtype=np.float32)
    for kk in range(w**2):
        tem[kk//w, kk % w] = rlt[kk]
    del(rlt)
    gc.collect()

    tv1_pxl[levs-1] = np.array(tem)
    tv1_pxl_id[levs-1] = tv1_pxl[levs-1].argmin()
    shift[levs-1, 0] = step*(tv1_pxl_id[levs-1]//w-int(w/2))
    shift[levs-1, 1] = step*(tv1_pxl_id[levs-1] % w-int(w/2))

    print(time.asctime())
    return tv1_pxl, tv1_pxl_id, shift, s+shift[levs-1]


@msgit(wd=100, fill='-')
def mrtv_reg3(levs=7, wz=10, sp_wz=8, sp_step=0.5, ps=None, imgs=None):
    """
    single process version of mrtv_reg2. Parallelization is implemented
    on the upper level on which the alignments of pairs of images are
    parallelized
    """
    if ps is not None:
        imgs[1] = np.real(np.fft.ifftn(
            fourier_shift(np.fft.fftn(imgs[1]), ps)))

    sch_config = {}
    sch_config[levs-1] = {'wz': sp_wz, 'step': sp_step}
    tv1_pxl = {}
    tv1_pxl_id = np.zeros(levs, dtype=np.int16)
    shift = np.zeros([levs, 2], dtype=np.float32)

    for ii in range(levs-1):
        sch_config[levs-2-ii] = {'wz': 6, 'step': 0.5**ii}
    sch_config[0] = {'wz': wz, 'step': 0.5**(levs-2)}

    if ((np.array(imgs[0].shape)*0.5**(levs-2) - wz) <= 0).any():
        return -1

    for ii in range(levs-1):
        w = sch_config[ii]['wz']
        step = sch_config[ii]['step']
        f = zoom(imgs[0], step)
        m = zoom(imgs[1], step)
        mk = np.zeros(m.shape, dtype=np.int8)
        mk[int(wz*2**(ii-1)):-int(wz*2**(ii-1)),
           int(wz*2**(ii-1)):-int(wz*2**(ii-1))] = 1

        s = np.array([0, 0], dtype=np.int32)
        for jj in range(ii):
            s = np.int_(s + shift[jj]*2**(ii-jj))

        tem = np.ndarray([w, w], dtype=np.float32)
        for jj in range(w):
            for kk in range(w):
                tem[jj, kk] = tv_l1_pixel(f, m, mk,
                                          [jj-int(w/2)+s[0], kk-int(w/2)+s[1]])

        tv1_pxl[ii] = np.array(tem)
        tv1_pxl_id[ii] = tv1_pxl[ii].argmin()
        shift[ii, 0] = (tv1_pxl_id[ii]//w-int(w/2))
        shift[ii, 1] = (tv1_pxl_id[ii] % w-int(w/2))

    mk = np.zeros(imgs[0].shape)
    mk[int(wz*2**(levs-2)):-int(wz*2**(levs-2)),
       int(wz*2**(levs-2)):-int(wz*2**(levs-2))] = 1
    w = sch_config[levs-1]['wz']
    step = sch_config[levs-1]['step']
    s = s + shift[levs-2]

    tem = np.ndarray([w, w], dtype=np.float32)
    for jj in range(w):
        for kk in range(w):
            tem[jj, kk] = tv_l1_subpixel(imgs[0], imgs[1], mk,
                                         [jj-int(w/2)+s[0], kk-int(w/2)+s[1]])

    tv1_pxl[levs-1] = np.array(tem)
    tv1_pxl_id[levs-1] = tv1_pxl[levs-1].argmin()
    shift[levs-1, 0] = step*(tv1_pxl_id[levs-1]//w-int(w/2))
    shift[levs-1, 1] = step*(tv1_pxl_id[levs-1] % w-int(w/2))

    print(time.asctime())
    return tv1_pxl, tv1_pxl_id, shift, s+shift[levs-1]


@msgit(wd=100, fill='-')
def mrtv_reg4(levs=7, wz=10, sp_wz=8, sp_step=0.5, ps=None, imgs=None):
    """
    Single process version of mrtv_reg2. Parallelization is implemented
    on the upper level on which the alignments of pairs of images are
    parallelized

    INPUTS:
        levs: int
    """
    if ps is not None:
        imgs[1] = np.real(np.fft.ifftn(
            fourier_shift(np.fft.fftn(imgs[1]), ps)))

    sch_config = {}
    sch_config[levs-1] = {'wz': sp_wz, 'step': sp_step}
    tv1_pxl = {}
    tv1_pxl_id = np.zeros(levs, dtype=np.int16)
    shift = np.zeros([levs, 2], dtype=np.float32)

    for ii in range(levs-1):
        sch_config[levs-2-ii] = {'wz': wz, 'step': 0.5**ii}
    sch_config[0] = {'wz': wz, 'step': 0.5**(levs-2)}

    if ((np.array(imgs[0].shape)*0.5**(levs-2) - wz) <= 0).any():
        return -1

    for ii in range(levs-1):
        w = sch_config[ii]['wz']
        step = sch_config[ii]['step']
        f = zoom(imgs[0], step)
        m = zoom(imgs[1], step)
        mk = np.zeros(m.shape, dtype=np.int8)
        mk[int(wz*2**(ii-1)):-int(wz*2**(ii-1)),
           int(wz*2**(ii-1)):-int(wz*2**(ii-1))] = 1

        s = np.array([0, 0], dtype=np.int32)
        for jj in range(ii):
            s = np.int_(s + shift[jj]*2**(ii-jj))

        tem = np.ndarray([w, w], dtype=np.float32)
        for jj in range(w):
            for kk in range(w):
                tem[jj, kk] = tv_l1_pixel(f, m, mk,
                                          [jj-int(w/2)+s[0], kk-int(w/2)+s[1]])

        tv1_pxl[ii] = np.array(tem)
        tv1_pxl_id[ii] = tv1_pxl[ii].argmin()
        shift[ii, 0] = (tv1_pxl_id[ii]//w-int(w/2))
        shift[ii, 1] = (tv1_pxl_id[ii] % w-int(w/2))

    s = s + shift[levs-2]

    if sp_wz > (int(wz/2)-1):
        sp_wz = (int(wz/2)-1)

    idx = np.int_(shift[levs-2] + int(w/2))
    print('shift:', shift)
    print('idx:', idx)
    shift[levs-1] = tv_l1_subpixel_fit(np.linspace(-sp_wz, sp_wz, 2*sp_wz+1),
                                       np.linspace(-sp_wz*10, sp_wz *
                                                   10, 2*sp_wz*10+1)/10,
                                       [tv1_pxl[levs-2].reshape([wz, wz])[(idx[0]-sp_wz):(idx[0]+sp_wz+1), idx[1]],
                                        tv1_pxl[levs-2].reshape([wz, wz])[idx[0], (idx[1]-sp_wz):(idx[1]+sp_wz+1)]])

    print(time.asctime())
    return tv1_pxl, tv1_pxl_id, shift, s+shift[levs-1]


@msgit(wd=100, fill='-')
def mrtv_reg5(levs=7, wz=10, sp_wz=1, sp_step=0.5, lsw=10, ps=None, imgs=None):
    """
    single process version of mrtv_reg2. Parallelization is implemented
    on the upper level on which the alignments of pairs of images are
    parallelized
    """
    if ps is not None:
        imgs[1] = np.real(np.fft.ifftn(
            fourier_shift(np.fft.fftn(imgs[1]), ps)))

    sch_config = {}
    sch_config[levs-1] = {'wz': sp_wz, 'step': sp_step}
    tv1_pxl = {}
    tv1_pxl_id = np.zeros(levs, dtype=np.int16)
    shift = np.zeros([levs, 2], dtype=np.float32)

    for ii in range(levs-1):
        sch_config[levs-2-ii] = {'wz': wz, 'step': 0.5**ii}
    sch_config[0] = {'wz': wz, 'step': 0.5**(levs-2)}

    if ((np.array(imgs[0].shape)*0.5**(levs-2) - wz) <= 0).any():
        return -1

    for ii in range(levs-1):
        w = sch_config[ii]['wz']
        step = sch_config[ii]['step']
        f = zoom(imgs[0], step)
        m = zoom(imgs[1], step)
        mk = np.zeros(m.shape, dtype=np.int8)
        mk[int(wz*2**(ii-1)):-int(wz*2**(ii-1)),
           int(wz*2**(ii-1)):-int(wz*2**(ii-1))] = 1

        s = np.array([0, 0], dtype=np.int32)
        for jj in range(ii):
            s = np.int_(s + shift[jj]*2**(ii-jj))

        if ii == levs-2:
            tv_lv = []
            for kk in range(2*lsw+1):
                tv_lv.append(tv_l1_pixel(f, m, 1, [kk-lsw+s[0], s[1]]))
            s[0] += (np.array(tv_lv).argmin() - lsw)

            tv_lh = []
            for kk in range(2*lsw+1):
                tv_lh.append(tv_l1_pixel(f, m, 1, [s[0], kk-lsw+s[1]]))
            s[1] += (np.array(tv_lh).argmin() - lsw)

        tem = np.ndarray([w, w], dtype=np.float32)
        for jj in range(w):
            for kk in range(w):
                tem[jj, kk] = tv_l1_pixel(f, m, mk,
                                          [jj-int(w/2)+s[0], kk-int(w/2)+s[1]])

        tv1_pxl[ii] = np.array(tem)
        tv1_pxl_id[ii] = tv1_pxl[ii].argmin()
        shift[ii, 0] = (tv1_pxl_id[ii]//w-int(w/2))
        shift[ii, 1] = (tv1_pxl_id[ii] % w-int(w/2))

    print('s before:', s)
    s = s + shift[levs-2]

    if sp_wz > (int(wz/2)-1):
        sp_wz = (int(wz/2)-1)

    idx = np.int_(shift[levs-2] + int(w/2))
    print('s after:', s)
    print('shift:', shift)
    print('idx:', idx)

    us = int(1/sp_step)
    shift[levs-1] = tv_l1_subpixel_ana(np.linspace(-1, 1, 3),
                                       np.linspace(-us, us, 2*us+1)/us,
                                       [tv1_pxl[levs-2].reshape([wz, wz])[(idx[0]-sp_wz):(idx[0]+sp_wz+1), idx[1]],
                                        tv1_pxl[levs-2].reshape([wz, wz])[idx[0], (idx[1]-sp_wz):(idx[1]+sp_wz+1)]])
    return tv1_pxl, tv1_pxl_id, shift, s+shift[levs-1]


# @msgit(wd=100, fill='-')
# def mrtv_reg(pxl_conf, sub_conf, ps, imgs=None):
#     """
#     Provide a unified interace for using multi-resolution TV-based registration
#     algorithm with different configuration options.

#     Inputs:
#     ______
#         pxl_conf: dict; multi-resolution TV minimization configuration at
#                   resolution higher than a single pixel; it includes items:
#                   'type': 'area', 'line', 'al', 'la'
#                   'levs': int
#                   'wz': int
#                   'lsw': int
#                       levs=7, wz=10, sp_wz=1, sp_step=0.5, lsw=10, ps=None, imgs=None
#         sub_conf: dict; sub-pixel TV minimization configuration; it includes
#                   items:
#                   'use': boolean; if conduct sub-pixel registration on the end
#                   'type': ['ana', 'fit']; which sub-pixel routine to use
#                   'sp_wz': int; for 'fit' option; the number of TV points to be
#                            used in fitting
#                   'sp_us': int; for 'ana' option; up-sampling factor
#         ps: optonal; None or a 2-element tuple
#         imgs: the pair of the images to be registered

#     Outputs:
#     _______
#         tv_pxl: dict; TV at each search point under levels as the keywords
#         tv_pxl_id: the id of the minimum TV in the flatted tv_pxl for each level
#         shift: ndarray; shift at each level of resolution
#         tot_shift: 2-element tuple; overall displacement between the pair of images
#     """
#     """
#     single process version of mrtv_reg2. Parallelization is implemented
#     on the upper level on which the alignments of pairs of images are
#     parallelized
#     """
#     # print(pxl_conf)
#     # print(sub_conf)
#     # print(ps)
#     if ps is not None:
#         imgs[1] = np.real(np.fft.ifftn(
#             fourier_shift(np.fft.fftn(imgs[1]), ps)))

#     levs = pxl_conf['levs']
#     w = pxl_conf['wz']
#     step = {}
#     tv_pxl = {}
#     tv_pxl_id = np.zeros(levs, dtype=np.int16)
#     shift = np.zeros([levs+1, 2], dtype=np.float32)

#     for ii in range(levs):
#         step[levs-1-ii] = 0.5**ii

#     if ((np.array(imgs[0].shape)*0.5**(levs-2) - w) <= 0).any():
#         return -1

#     if pxl_conf['type'] == 'area':
#         for ii in range(levs):
#             s = step[ii]
#             f = zoom(imgs[0], s)
#             m = zoom(imgs[1], s)
#             mk = np.zeros(m.shape, dtype=np.int8)
#             mk[int(w*2**(ii-1)):-int(w*2**(ii-1)),
#                int(w*2**(ii-1)):-int(w*2**(ii-1))] = 1

#             sh = np.array([0, 0])
#             for jj in range(ii):
#                 sh = np.int_(sh + shift[jj]*2**(ii-jj))

#             if ii == levs-1:
#                 tem = np.ndarray([2*w, 2*w], dtype=np.float32)
#                 for jj in range(2*w):
#                     for kk in range(2*w):
#                         tem[jj, kk] = tv_l1_pixel(f, m, mk,
#                                                   [jj-w+sh[0],
#                                                    kk-w+sh[1]])

#                 tv_pxl[ii] = np.array(tem)
#                 tv_pxl_id[ii] = tv_pxl[ii].argmin()
#                 shift[ii, 0] = (tv_pxl_id[ii]//(2*w)-w)
#                 shift[ii, 1] = (tv_pxl_id[ii] % (2*w)-w)
#             else:
#                 tem = np.ndarray([w, w], dtype=np.float32)
#                 for jj in range(w):
#                     for kk in range(w):
#                         tem[jj, kk] = tv_l1_pixel(f, m, mk,
#                                                   [jj-int(w/2)+sh[0],
#                                                    kk-int(w/2)+sh[1]])

#                 tv_pxl[ii] = np.array(tem)
#                 tv_pxl_id[ii] = tv_pxl[ii].argmin()
#                 shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
#                 shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))

#             # tem = np.ndarray([w, w], dtype=np.float32)
#             # for jj in range(w):
#             #     for kk in range(w):
#             #         tem[jj, kk] = tv_l1_pixel(f, m, mk,
#             #                                   [jj-int(w/2)+sh[0],
#             #                                    kk-int(w/2)+sh[1]])

#             # tv_pxl[ii] = np.array(tem)
#             # tv_pxl_id[ii] = tv_pxl[ii].argmin()
#             # shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
#             # shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
#     elif pxl_conf['type'] == 'line':
#         lsw = pxl_conf['lsw']
#         shift = np.zeros([3, 2])
#         tv_pxl_id = np.zeros(2, dtype=np.int16)

#         tv_v = []
#         mk = np.zeros(imgs[0].shape, dtype=np.int8)
#         mk[int(lsw/2):-int(lsw/2), :] = 1
#         for ii in range(lsw):
#             tv_v.append(tv_l1_pixel(imgs[0], imgs[1], mk,
#                                     [ii-int(lsw/2), 0]))
#         shift[0, 0] = (np.array(tv_v).argmin() - int(lsw/2))
#         tv_h = []
#         mk = np.zeros(imgs[0].shape, dtype=np.int8)
#         mk[:, int(lsw/2):-int(lsw/2)] = 1
#         for ii in range(lsw):
#             tv_h.append(tv_l1_pixel(imgs[0], imgs[1], mk,
#                                     [0, ii-int(lsw/2)]))
#         shift[0, 1] = (np.array(tv_h).argmin() - int(lsw/2))
#         tv_pxl[0] = np.vstack(np.array(tv_v), np.array(tv_h))
#         tv_pxl_id[0] = 0
#         sh = shift
#     elif pxl_conf['type'] == 'al':
#         lsw = pxl_conf['lsw']
#         for ii in range(levs):
#             s = step[ii]
#             f = zoom(imgs[0], s)
#             m = zoom(imgs[1], s)
#             mk = np.zeros(m.shape, dtype=np.int8)
#             mk[int(w*2**(ii-1)):-int(w*2**(ii-1)),
#                int(w*2**(ii-1)):-int(w*2**(ii-1))] = 1

#             sh = np.array([0, 0])
#             for jj in range(ii):
#                 sh = np.int_(sh + shift[jj]*2**(ii-jj))

#             if ii == levs-1:
#                 tv_v = []
#                 mk = np.zeros(imgs[0].shape, dtype=np.int8)
#                 mk[int(lsw/2):-int(lsw/2), :] = 1
#                 for kk in range(lsw):
#                     tv_v.append(tv_l1_pixel(f, m, mk,
#                                             [kk-int(lsw/2)+sh[0], sh[1]]))
#                 sh[0] += (np.array(tv_v).argmin() - int(lsw/2))

#                 tv_h = []
#                 mk = np.zeros(imgs[0].shape, dtype=np.int8)
#                 mk[:, int(lsw/2):-int(lsw/2)] = 1
#                 for kk in range(lsw):
#                     tv_h.append(tv_l1_pixel(f, m, mk,
#                                             [sh[0], kk-int(lsw/2)+sh[1]]))
#                 sh[1] += (np.array(tv_h).argmin() - int(lsw/2))

#                 tem = np.ndarray([2*w, 2*w], dtype=np.float32)
#                 for jj in range(2*w):
#                     for kk in range(2*w):
#                         tem[jj, kk] = tv_l1_pixel(f, m, mk,
#                                                   [jj-w+sh[0],
#                                                    kk-w+sh[1]])

#                 tv_pxl[ii] = np.array(tem)
#                 tv_pxl_id[ii] = tv_pxl[ii].argmin()
#                 shift[ii, 0] = (tv_pxl_id[ii]//(2*w)-w)
#                 shift[ii, 1] = (tv_pxl_id[ii] % (2*w)-w)
#             else:
#                 tem = np.ndarray([w, w], dtype=np.float32)
#                 for jj in range(w):
#                     for kk in range(w):
#                         tem[jj, kk] = tv_l1_pixel(f, m, mk,
#                                                   [jj-int(w/2)+sh[0],
#                                                    kk-int(w/2)+sh[1]])

#                 tv_pxl[ii] = np.array(tem)
#                 tv_pxl_id[ii] = tv_pxl[ii].argmin()
#                 shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
#                 shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
#         sh = sh + shift[levs-1]
#     elif pxl_conf['type'] == 'la':
#         # TODO
#         pass

#     if sub_conf['use']:
#         # idx = np.int_(shift[levs-2] + int(w/2))
#         idx = np.int_(shift[levs-1] + w)
#         if sub_conf['type'] == 'fit':
#             sw = int(sub_conf['sp_wz'])
#             shift[levs] = tv_l1_subpixel_fit(np.linspace(-sw, sw, 2*sw+1),
#                                              np.linspace(-10*sw,
#                                                          10*sw, 20*sw+1),
#                                              [tv_pxl[levs-1].reshape(2*w, 2*w)
#                                               [(idx[0]-sw):(idx[0]+sw+1), idx[1]],
#                                               tv_pxl[levs-1].reshape(2*w, 2*w)
#                                               [idx[0], (idx[1]-sw):(idx[1]+sw+1)]])
#         elif sub_conf['type'] == 'ana':
#             us = int(sub_conf['sp_us'])
#             shift[levs] = tv_l1_subpixel_ana(np.linspace(-1, 1, 3),
#                                              np.linspace(-us, us, 2*us+1)/us,
#                                              [tv_pxl[levs-1].reshape(2*w, 2*w)
#                                               [(idx[0]-1):(idx[0]+2), idx[1]],
#                                               tv_pxl[levs-1].reshape(2*w, 2*w)
#                                               [idx[0], (idx[1]-1):(idx[1]+2)]])
#     else:
#         shift[levs] = [0, 0]
#     return tv_pxl, tv_pxl_id, shift, sh+shift[levs]


def mrtv_reg(pxl_conf, sub_conf, ps=None, kernel=3, imgs=None):
    """
    Provide a unified interace for using multi-resolution TV-based registration
    algorithm with different configuration options.

    Inputs:
    ______
        pxl_conf: dict; multi-resolution TV minimization configuration at
                  resolution higher than a single pixel; it includes items:
                  'type': 'area', 'line', 'al', 'la'
                  'levs': int
                  'wz': int
                  'lsw': int
        sub_conf: dict; sub-pixel TV minimization configuration; it includes
                  items:
                  'use': boolean; if conduct sub-pixel registration on the end
                  'type': ['ana', 'fit']; which sub-pixel routine to use
                  'sp_wz': int; for 'fit' option; the number of TV points to be
                           used in fitting
                  'sp_us': int; for 'ana' option; up-sampling factor
        ps: optonal; None or a 2-element tuple
        imgs: the pair of the images to be registered

    Outputs:
    _______
        tv_pxl: dict; TV at each search point under levels as the keywords
        tv_pxl_id: the id of the minimum TV in the flatted tv_pxl for each level
        shift: ndarray; shift at each level of resolution
        tot_shift: 2-element tuple; overall displacement between the pair of images
    """
    """
    single process version of mrtv_reg2. Parallelization is implemented
    on the upper level on which the alignments of pairs of images are
    parallelized
    """
    if ps is not None:
        imgs[1] = np.real(np.fft.ifftn(
            fourier_shift(np.fft.fftn(imgs[1]), ps)))

    levs = pxl_conf['levs']
    w = pxl_conf['wz']
    step = {}
    tv_pxl = {}
    tv_pxl_id = np.zeros(levs, dtype=np.int16)
    shift = np.zeros([levs+1, 2], dtype=np.float32)

    for ii in range(levs):
        step[levs-1-ii] = 0.5**ii

    if ((np.array(imgs[0].shape)*0.5**(levs-2) - w) <= 0).any():
        return -1

    if pxl_conf['type'] == 'area':
        for ii in range(levs):
            s = step[ii]
            f = zoom(imgs[0], s)
            m = zoom(imgs[1], s)
            mk = np.zeros(m.shape, dtype=np.int8)
            mk[int(w*2**(ii-1)):-int(w*2**(ii-1)),
               int(w*2**(ii-1)):-int(w*2**(ii-1))] = 1

            sh = np.array([0, 0])
            for jj in range(ii):
                sh = np.int_(sh + shift[jj]*2**(ii-jj))

            tem = np.ndarray([w, w], dtype=np.float32)
            for jj in range(w):
                for kk in range(w):
                    tem[jj, kk] = tv_l1_pixel(f, m, mk,
                                              [jj-int(w/2)+sh[0],
                                               kk-int(w/2)+sh[1]],
                                              kernel=kernel)

            tv_pxl[ii] = np.array(tem)
            tv_pxl_id[ii] = tv_pxl[ii].argmin()
            shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
            shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
            if ii == levs - 1:
                idx = np.int_(shift[levs-1] + int(w/2))
                while idx[0] == 0 or idx[0] == w-1 or idx[1] == 0 or idx[1] == w-1:
                    if idx[0] == 0:
                        sh[0] -= int(w/2)
                    elif idx[0] == w-1:
                        sh[0] += int(w/2)
                    if idx[1] == 0:
                        sh[1] -= int(w/2)
                    elif idx[1] == w-1:
                        sh[1] += int(w/2)
                    for jj in range(w):
                        for kk in range(w):
                            tem[jj, kk] = tv_l1_pixel(f, m, mk,
                                                      [jj-int(w/2)+sh[0],
                                                       kk-int(w/2)+sh[1]],
                                                      kernel=kernel)

                    tv_pxl[ii] = np.array(tem)
                    tv_pxl_id[ii] = tv_pxl[ii].argmin()
                    shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
                    shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
                    idx = np.int_(shift[levs-1] + int(w/2))
        sh = sh + shift[levs-1]
    elif pxl_conf['type'] == 'line':
        levs = 1
        lsw = pxl_conf['lsw']
        shift = np.zeros([2, 2])
        tv_pxl_id = np.zeros(2, dtype=np.int16)

        tv_v = []
        mk = np.zeros(imgs[0].shape, dtype=np.int8)
        mk[int(lsw/2):-int(lsw/2), :] = 1
        for ii in range(lsw):
            tv_v.append(tv_l1_pixel(imgs[0], imgs[1], mk,
                                    [ii-int(lsw/2), 0],
                                    kernel=kernel))
        shift[0, 0] = (np.array(tv_v).argmin() - int(lsw/2))
        tv_h = []
        mk = np.zeros(imgs[0].shape, dtype=np.int8)
        mk[:, int(lsw/2):-int(lsw/2)] = 1
        for ii in range(lsw):
            tv_h.append(tv_l1_pixel(imgs[0], imgs[1], mk,
                                    [0, ii-int(lsw/2)],
                                    kernel=kernel))
        shift[0, 1] = (np.array(tv_h).argmin() - int(lsw/2))
        tv_pxl[0] = np.stack([np.array(tv_v), np.array(tv_h)], axis=0)
        tv_pxl_id[0] = 0
        sh = shift
    elif pxl_conf['type'] == 'al':
        lsw = pxl_conf['lsw']
        for ii in range(levs):
            s = step[ii]
            f = zoom(imgs[0], s)
            m = zoom(imgs[1], s)
            mk = np.zeros(m.shape, dtype=np.int8)
            mk[int(w*2**(ii-1)):-int(w*2**(ii-1)),
               int(w*2**(ii-1)):-int(w*2**(ii-1))] = 1

            sh = np.array([0, 0])
            for jj in range(ii):
                sh = np.int_(sh + shift[jj]*2**(ii-jj))

            if ii == levs-1:
                tv_v = []
                mk = np.zeros(imgs[0].shape, dtype=np.int8)
                mk[int(lsw/2):-int(lsw/2), :] = 1
                for kk in range(lsw):
                    tv_v.append(tv_l1_pixel(f, m, mk,
                                            [kk-int(lsw/2)+sh[0], sh[1]],
                                            kernel=kernel))
                sh[0] += (np.array(tv_v).argmin() - int(lsw/2))

                tv_h = []
                mk = np.zeros(imgs[0].shape, dtype=np.int8)
                mk[:, int(lsw/2):-int(lsw/2)] = 1
                for kk in range(lsw):
                    tv_h.append(tv_l1_pixel(f, m, mk,
                                            [sh[0], kk-int(lsw/2)+sh[1]],
                                            kernel=kernel))
                sh[1] += (np.array(tv_h).argmin() - int(lsw/2))

            tem = np.ndarray([w, w], dtype=np.float32)
            for jj in range(w):
                for kk in range(w):
                    tem[jj, kk] = tv_l1_pixel(f, m, mk,
                                              [jj-int(w/2)+sh[0],
                                               kk-int(w/2)+sh[1]],
                                              kernel=kernel)

            tv_pxl[ii] = np.array(tem)
            tv_pxl_id[ii] = tv_pxl[ii].argmin()
            shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
            shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))

            if ii == levs - 1:
                idx = np.int_(shift[levs-1] + int(w/2))
                while idx[0] == 0 or idx[0] == w-1 or idx[1] == 0 or idx[1] == w-1:
                    if idx[0] == 0:
                        sh[0] -= int(w/2)
                    elif idx[0] == w-1:
                        sh[0] += int(w/2)
                    if idx[1] == 0:
                        sh[1] -= int(w/2)
                    elif idx[1] == w-1:
                        sh[1] += int(w/2)
                    for jj in range(w):
                        for kk in range(w):
                            tem[jj, kk] = tv_l1_pixel(f, m, mk,
                                                      [jj-int(w/2)+sh[0],
                                                       kk-int(w/2)+sh[1]],
                                                      kernel=kernel)

                    tv_pxl[ii] = np.array(tem)
                    tv_pxl_id[ii] = tv_pxl[ii].argmin()
                    shift[ii, 0] = (tv_pxl_id[ii]//w-int(w/2))
                    shift[ii, 1] = (tv_pxl_id[ii] % w-int(w/2))
                    idx = np.int_(shift[levs-1] + int(w/2))
        sh = sh + shift[levs-1]
    elif pxl_conf['type'] == 'la':
        # TODO
        pass

    if sub_conf['use']:
        idx = np.int_(shift[levs-1] + int(w/2))
        if sub_conf['type'] == 'fit':
            sw = int(sub_conf['sp_wz'])
            shift[levs] = tv_l1_subpixel_fit(np.linspace(-sw, sw, 2*sw+1),
                                             np.linspace(-10*sw,
                                                         10*sw, 20*sw+1),
                                             [tv_pxl[levs-1].reshape(w, w)
                                              [(idx[0]-sw):(idx[0]+sw+1), idx[1]],
                                              tv_pxl[levs-1].reshape(w, w)
                                              [idx[0], (idx[1]-sw):(idx[1]+sw+1)]])
        elif sub_conf['type'] == 'ana':
            us = int(sub_conf['sp_us'])
            shift[levs] = tv_l1_subpixel_ana(np.linspace(-1, 1, 3),
                                             np.linspace(-us, us, 2*us+1)/us,
                                             [tv_pxl[levs-1].reshape(w, w)
                                              [(idx[0]-1):(idx[0]+2), idx[1]],
                                              tv_pxl[levs-1].reshape(w, w)
                                              [idx[0], (idx[1]-1):(idx[1]+2)]])
    else:
        shift[levs] = [0, 0]
    return tv_pxl, tv_pxl_id, shift, sh+shift[levs]


def mrtv_mpc_combo_reg(fixed_img, img, us=100, reference_mask=None,
                       overlap_ratio=0.3, levs=4, wz=10, sp_wz=20,
                       sp_step=0.2):
    shift = np.zeros([levs+1, 2])
    if reference_mask is not None:
        shift[0] = phase_cross_correlation(fixed_img, img, upsample_factor=us,
                                           reference_mask=reference_mask,
                                           overlap_ratio=overlap_ratio)
    else:
        shift[0], _, _ = phase_cross_correlation(fixed_img, img,
                                                 upsample_factor=us,
                                                 overlap_ratio=overlap_ratio)

    tv1_pxl, tv1_pxl_id, shift[1:], ss = mrtv_reg2(fixed_img, img, levs=levs,
                                                   wz=wz, sp_wz=sp_wz,
                                                   sp_step=sp_step, ps=shift[0])

    print(time.asctime())
    return tv1_pxl, tv1_pxl_id, shift, shift[0]+ss


@msgit(wd=100, fill='-')
def mrtv_ls_combo_reg(ls_w=100, levs=2, wz=10, sp_wz=8, sp_step=0.5, kernel=3, imgs=None):
    """
    line search combined with two-level multi-resolution search
    """
    shift = np.zeros([3, 2])
    tv1_pxl_id = np.zeros(2, dtype=np.int16)
    tv1_pxl = {}
    tv_v = []
    for ii in range(ls_w):
        tv_v.append(tv_l1_pixel(imgs[0], imgs[1], 1, [2*ii-ls_w, 0],
                                kernel=kernel))
    shift[0, 0] = 2*(np.array(tv_v).argmin() - int(ls_w/2))
    # shift[0, 0] = (np.array(tv_v).argmin() - int(ls_w/2))

    tv_h = []
    for ii in range(ls_w):
        tv_h.append(tv_l1_pixel(imgs[0], imgs[1], 1, [0, 2*ii-ls_w],
                                kernel=kernel))
    shift[0, 1] = 2*(np.array(tv_h).argmin() - int(ls_w/2))
    # shift[0, 1] = (np.array(tv_h).argmin() - int(ls_w/2))

    tv1_pxl, tv1_pxl_id, shift[1:], ss = mrtv_reg3(levs=levs, wz=wz, sp_wz=sp_wz,
                                                   sp_step=sp_step, imgs=imgs,
                                                   ps=shift[0])

    print(time.asctime())
    return tv1_pxl, tv1_pxl_id, shift, shift[0]+ss


@msgit(wd=100, fill='-')
def mrtv_ls_combo_reg2(ls_w=100, levs=2, wz=10, sp_wz=8, sp_step=0.5, kernel=3, imgs=None):
    """
    line search combined with two-level multi-resolution search
    """
    shift = np.zeros([3, 2])
    tv1_pxl_id = np.zeros(2, dtype=np.int16)
    tv1_pxl = {}
    tv_v = []
    for ii in range(ls_w):
        tv_v.append(tv_l1_pixel(imgs[0], imgs[1], 1, [ii-int(ls_w/2), 0],
                                kernel=kernel))
    shift[0, 0] = (np.array(tv_v).argmin() - int(ls_w/2))

    tv_h = []
    for ii in range(ls_w):
        tv_h.append(tv_l1_pixel(imgs[0], imgs[1], 1, [0, ii-int(ls_w/2)],
                                kernel=kernel))
    shift[0, 1] = (np.array(tv_h).argmin() - int(ls_w/2))

    tv1_pxl, tv1_pxl_id, shift[1:], ss = mrtv_reg4(levs=levs, wz=wz, sp_wz=sp_wz,
                                                   sp_step=sp_step, imgs=imgs,
                                                   ps=shift[0])

    print(time.asctime())
    return tv1_pxl, tv1_pxl_id, shift, shift[0]+ss


def shift_img(img_shift):
    img = img_shift[0]
    shift = img_shift[1]
    return np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)))


def sr_transform_img(sr, data):
    return sr.transform(data[0], tmat=data[1])


def mp_shift_img(imgs, shifts, stype='trans', sr=None, axis=0):
    if stype.upper() == 'SR':
        if axis == 0:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(sr_transform_img, sr),
                               [[imgs[ii, ...], shifts[ii]] for ii
                                in range(imgs.shape[0])])
            pool.close()
            pool.join()
            return np.array(rlt).astype(np.float32)
        elif axis == 1:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(sr_transform_img, sr),
                               [[imgs[:, ii, :], shifts[ii]] for ii
                                in range(imgs.shape[1])])
            pool.close()
            pool.join()
            return np.swapaxes(np.array(rlt).astype(np.float32), 0, 1)
        elif axis == 2:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(sr_transform_img, sr),
                               [[imgs[:, :, ii], shifts[ii]] for ii
                                in range(imgs.shape[2])])
            pool.close()
            pool.join()
            return np.swapaxes(np.array(rlt).astype(np.float32), 0, 2)
    else:
        if axis == 0:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(shift_img,
                               [[imgs[ii, ...], shifts[ii]] for ii
                                in range(imgs.shape[0])])
            pool.close()
            pool.join()
            return np.array(rlt).astype(np.float32)
        elif axis == 1:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(shift_img,
                               [[imgs[:, ii, :], shifts[ii]] for ii
                                in range(imgs.shape[1])])
            pool.close()
            pool.join()
            return np.swapaxes(np.array(rlt).astype(np.float32), 0, 1)
        elif axis == 2:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(shift_img,
                               [[imgs[:, :, ii], shifts[ii]] for ii
                                in range(imgs.shape[2])])
            pool.close()
            pool.join()
            return np.swapaxes(np.array(rlt).astype(np.float32), 0, 2)


def pc_reg(upsample_factor, imgs):
    shift, _, _ = phase_cross_correlation(
        imgs[0], imgs[1], upsample_factor=upsample_factor)
    return shift


def mpc_reg(reference_mask, overlap_ratio, imgs):
    shift = phase_cross_correlation(imgs[0], imgs[1], reference_mask=reference_mask,
                                    overlap_ratio=overlap_ratio)
    return shift


def sr_reg(sr, imgs):
    shift = sr.register(imgs[0], imgs[1])
    return shift


def _chunking(dim):
    bdi = []
    chunk = int(np.ceil(dim/N_CPU))
    for ii in range(N_CPU+1):
        bdi.append(ii*chunk)
    bdi[-1] = min(dim, ii*chunk)
    return bdi


def reg_stack(imgs, method, *args, axis=0, filter=None, **kwargs):
    shift = []
    if method.upper() == 'MRTV_REG':
        if filter is None:
            if axis == 0:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[imgs[ii], imgs[ii+1]] for ii
                                    in range(imgs.shape[0]-1)])
            elif axis == 1:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[imgs[:, ii, :], imgs[:, ii+1, :]] for ii
                                    in range(imgs.shape[1]-1)])
            elif axis == 2:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[imgs[:, :, ii], imgs[:, :, ii+1]] for ii
                                    in range(imgs.shape[2]-1)])
            pool.close()
            pool.join()
            shift = np.vstack(np.array(rlt, dtype=object)[:, -1])
        else:
            flt_func = filter['func']
            flt_parm = filter['params']
            if axis == 0:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[flt_func(imgs[ii], *flt_parm), flt_func(imgs[ii+1], *flt_parm)] for ii
                                    in range(imgs.shape[0]-1)])
            elif axis == 1:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[flt_parm(imgs[:, ii, :], *flt_parm), flt_func(imgs[:, ii+1, :], *flt_parm)] for ii
                                    in range(imgs.shape[1]-1)])
            elif axis == 2:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[flt_parm(imgs[:, :, ii], *flt_parm), flt_func(imgs[:, :, ii+1], *flt_parm)] for ii
                                    in range(imgs.shape[2]-1)])
            pool.close()
            pool.join()
            shift = np.vstack(np.array(rlt, dtype=object)[:, -1])
    elif method.upper() == 'PC':
        if axis == 0:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(pc_reg, *args, **kwargs),
                               [[imgs[ii], imgs[ii+1]] for ii
                                in range(imgs.shape[0]-1)])
        elif axis == 1:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(pc_reg, *args, **kwargs),
                               [[imgs[:, ii, :], imgs[:, ii+1, :]] for ii
                                in range(imgs.shape[1]-1)])
        elif axis == 2:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(pc_reg, *args, **kwargs),
                               [[imgs[:, :, ii], imgs[:, :, ii+1]] for ii
                                in range(imgs.shape[2]-1)])
        pool.close()
        pool.join()
        shift = np.array(rlt)
    elif method.upper() == 'MPC':
        if axis == 0:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(mpc_reg, *args, **kwargs),
                               [[imgs[ii], imgs[ii+1]] for ii
                                in range(imgs.shape[0]-1)])
        elif axis == 1:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(mpc_reg, *args, **kwargs),
                               [[imgs[:, ii, :], imgs[:, ii+1, :]] for ii
                                in range(imgs.shape[1]-1)])
        elif axis == 2:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(mpc_reg, *args, **kwargs),
                               [[imgs[:, :, ii], imgs[:, :, ii+1]] for ii
                                in range(imgs.shape[2]-1)])
        pool.close()
        pool.join()
        shift = np.array(rlt)
    elif method.upper() == 'SR':
        if axis == 0:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(sr_reg, *args, **kwargs),
                               [[imgs[ii], imgs[ii+1]] for ii
                                in range(imgs.shape[0]-1)])
        elif axis == 1:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(sr_reg, *args, **kwargs),
                               [[imgs[:, ii, :], imgs[:, ii+1, :]] for ii
                                in range(imgs.shape[1]-1)])
        elif axis == 2:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(sr_reg, *args, **kwargs),
                               [[imgs[:, :, ii], imgs[:, :, ii+1]] for ii
                                in range(imgs.shape[2]-1)])
        pool.close()
        pool.join()
        shift = np.array(rlt)
    return shift


def reg_stacks(refs, movs, method, *args, axis=0, filter=None, **kwargs):
    shift = []
    if method.upper() == 'MRTV_REG':
        if filter is None:
            if axis == 0:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[refs[ii], movs[ii]] for ii
                                    in range(refs.shape[0])])
            elif axis == 1:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[refs[:, ii, :], movs[:, ii, :]] for ii
                                    in range(refs.shape[1])])
            elif axis == 2:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[refs[:, :, ii], movs[:, :, ii]] for ii
                                    in range(refs.shape[2])])
        else:
            flt_func = filter['func']
            flt_parm = filter['params']
            if axis == 0:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[flt_func(refs[ii], *flt_parm), flt_func(movs[ii], *flt_parm)] for ii
                                    in range(refs.shape[0])])
            elif axis == 1:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[flt_func(refs[:, ii, :], *flt_parm), flt_func(movs[:, ii, :], *flt_parm)] for ii
                                    in range(refs.shape[1])])
            elif axis == 2:
                with mp.Pool(N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, *args, **kwargs),
                                   [[flt_func(refs[:, :, ii], *flt_parm), flt_func(movs[:, :, ii], *flt_parm)] for ii
                                    in range(refs.shape[2])])
        pool.close()
        pool.join()
        shift = np.vstack(np.array(rlt, dtype=object)[:, -1])
    elif method.upper() == 'SR':
        if axis == 0:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(sr_reg, *args, **kwargs),
                               [[refs[ii], movs[ii]] for ii
                                in range(refs.shape[0])])
        elif axis == 1:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(sr_reg, *args, **kwargs),
                               [[refs[:, ii, :], movs[:, ii, :]] for ii
                                in range(refs.shape[1])])
        elif axis == 2:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(sr_reg, *args, **kwargs),
                               [[refs[:, :, ii], movs[:, :, ii]] for ii
                                in range(refs.shape[2])])
        pool.close()
        pool.join()
        shift = np.array(rlt)
    elif method.upper() == 'PC':
        if axis == 0:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(pc_reg, *args, **kwargs),
                               [[refs[ii], movs[ii]] for ii
                                in range(refs.shape[0])])
        elif axis == 1:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(pc_reg, *args, **kwargs),
                               [[refs[:, ii, :], movs[:, ii, :]] for ii
                                in range(refs.shape[1])])
        elif axis == 2:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(pc_reg, *args, **kwargs),
                               [[refs[:, :, ii], movs[:, :, ii]] for ii
                                in range(refs.shape[2])])
        pool.close()
        pool.join()
        shift = np.array(rlt)
    elif method.upper() == 'MPC':
        if axis == 0:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(mpc_reg, *args, **kwargs),
                               [[refs[ii], movs[ii]] for ii
                                in range(refs.shape[0])])
        elif axis == 1:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(mpc_reg, *args, **kwargs),
                               [[refs[:, ii, :], movs[:, ii, :]] for ii
                                in range(refs.shape[1])])
        elif axis == 2:
            with mp.Pool(N_CPU) as pool:
                rlt = pool.map(partial(mpc_reg, *args, **kwargs),
                               [[refs[:, :, ii], movs[:, :, ii]] for ii
                                in range(refs.shape[2])])
        pool.close()
        pool.join()
        shift = np.array(rlt)
    return shift


def cal_kde(masked_img):
    imgv = masked_img[(masked_img != masked_img.min()) &
                      (masked_img != masked_img.max())].flatten()
    val = np.unique(imgv)

    kde = KernelDensity(bandwidth=np.abs(val[1:]-val[:-1]).min()/1.5,
                        algorithm="kd_tree", kernel='gaussian',
                        metric='euclidean', atol=1e-5, rtol=1e-5,
                        breadth_first=True, leaf_size=40, metric_params=None)\
        .fit(imgv[:, None])
    x_grid = np.linspace(val.min(), val.max(), 500)
    pdf = np.exp(kde.score_samples(x_grid[:, None]))

    return x_grid, pdf, val, imgv


def sav_kde(fn, x_grid, pdf):
    with open(fn, 'w') as f:
        for x, y in zip(x_grid, pdf):
            f.write(str(x)+'\t'+str(y)+'\n')


def cal_fwhm(x_grid, pdf, margin=10):
    x_grid = x_grid[margin:-margin]
    pdf = pdf[margin:-margin]

    pidx = np.argmax(pdf)
    dmax = pdf[pidx]
    peng = x_grid[pidx]

    lidx = np.argmin(np.abs(pdf[:pidx] - dmax/2.))
    lval = x_grid[lidx]

    ridx = np.argmin(np.abs(pdf[pidx:] - dmax/2.)) + pidx
    rval = x_grid[ridx]

    e_cen = np.sum(pdf*x_grid)/np.sum(pdf)
    return pidx, dmax, peng, lidx, lval, ridx, rval, e_cen
