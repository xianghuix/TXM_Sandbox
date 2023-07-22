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
                           gaussian_filter as gf, binary_erosion)
from scipy.optimize import least_squares as lsq, lsq_linear, minimize, Bounds, LinearConstraint
from functools import partial

from skimage.registration import phase_cross_correlation
from skimage.filters.rank import threshold, median
from skimage.morphology import square
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


def _chunking(dim):
    bdi = []
    chunk = int(np.ceil(dim/N_CPU))
    for ii in range(N_CPU+1):
        bdi.append(ii*chunk)
    bdi[-1] = min(dim, ii*chunk)
    return bdi


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
    print(f"{x.shape=}")
    print(f"{y.shape=}")
    print(f"{rlt.shape=}")
    print(f"{len(rlt[:, 3])=}")
    print(f"{np.vstack(rlt[:, 0]).astype(np.float32).shape=}")
    print(f"{np.vstack(rlt[:, 1]).astype(np.float32).shape=}")
    print(f"{np.vstack(rlt[:, 2]).astype(np.float32).shape=}")
    print(f"{np.vstack(rlt[:, 3]).astype(np.float32).shape=}")
    return [np.vstack(rlt[:, 0]).T, np.vstack(rlt[:, 1]), 
            np.vstack(rlt[:, 2]), np.vstack(rlt[:, 3])]


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


@msgit(wd=100, fill='-')
def find_fit_peak(model, p, x):
    """
    inputs:
        model: string; line shape function name in 'functions'
        p: NxK array-like; spec array; N equals the number of model function arguments; K is the number of data point
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
    rlt = np.array(rlt, dtype=object)
    return (np.vstack(list(chain(*rlt[:, 0]))).astype(np.float32), 
            np.vstack(list(chain(*rlt[:, 1]))).astype(np.float32))


def _find_fun_peak(f, x0, fvars):
    rlt0 = []
    rlt1 = []
    for ii in range(fvars.shape[1]):
        xmax = x0[np.argmax(f(x0, *fvars[:, ii]))]
        rlt0.append(xmax)
        rlt1.append(f(xmax, *fvars[:, ii]))
    return (rlt0, rlt1)


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
        with mp.Pool(N_CPU) as pool:
            rlt = pool.map(partial(_find_50_peak_fit_both, func_e, x_e, func_p, x_p), [
                           [p_e[:, bdi[ii]:bdi[ii+1]], p_p[:, bdi[ii]:bdi[ii+1]]] for ii in range(N_CPU)])
        pool.close()
        pool.join()
        return np.vstack(list(chain(*rlt))).astype(np.float32)
    elif ftype == 'wl':
        func_p = functions[model_p]
        bdi = _chunking(len(p_e[0]))
        with mp.Pool(N_CPU) as pool:
            rlt = pool.map(partial(_find_50_peak_fit_wl, model_e, x_e, func_p, x_p), [
                           [p_e[:, bdi[ii]:bdi[ii + 1]], p_p[:, bdi[ii]:bdi[ii + 1]]] for ii in range(N_CPU)])
        pool.close()
        pool.join()
        return np.vstack(list(chain(*rlt))).astype(np.float32)
    elif ftype == 'edge':
        func_e = functions[model_e]
        bdi = _chunking(len(p_e[0]))
        with mp.Pool(N_CPU) as pool:
            rlt = pool.map(partial(_find_50_peak_fit_edge, func_e, x_e, model_p, x_p), [
                           [p_e[:, bdi[ii]:bdi[ii + 1]], p_p[:, bdi[ii]:bdi[ii + 1]]] for ii in range(N_CPU)])
        pool.close()
        pool.join()
        return np.vstack(list(chain(*rlt))).astype(np.float32)
    elif ftype == 'none':
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
        rlt0.append(x0[np.argmin(np.abs(f0(x0, *fvars[0][:, ii]) - 0.5*np.max(f1(x1, *fvars[1][:, ii]))))])
    return rlt0


def _find_50_peak_fit_wl(x, x0, f1, x1, fvars):
    rlt0 = []
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
