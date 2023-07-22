#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 08:56:30 2020

@author: xiao
"""

import functools
import numbers

import skimage.morphology as skm
import skimage.filters as skf
import numpy as np
from dask import delayed
from dask.array import from_delayed


def make_da_decorator(dtype=np.int8):

    def inner(func):

        def wrapper(img, *args, **kwargs):
            shp = img.shape
            return from_delayed(delayed(func(img, *args, **kwargs)),
                                shape=shp,
                                dtype=dtype)

        return wrapper

    return inner


@make_da_decorator(np.int8)
def threshold(image, lower=0, upper=1):
    return np.int8((image > lower) & (image < upper))


@make_da_decorator(np.float32)
def gaussian(image, sigma=1, **kwargs):
    return skf.gaussian(image, sigma=sigma, **kwargs).astype(np.float32)


@make_da_decorator(np.float32)
def median(image, footprint=2, **kwargs):
    return skf.median(image, footprint=footprint,
                          **kwargs).astype(np.float32)


@make_da_decorator(np.float32)
def dilation(image, footprint=2, iter=1, **kwargs):
    dtp = image.dtype
    if isinstance(footprint, numbers.Number):
        footprint = skm.disk(footprint)
    for _ in range(int(iter)):
        image[:] = skm.dilation(image, footprint=footprint, **kwargs)[:]
    return image.astype(np.float32)


@make_da_decorator(np.float32)
def erosion(image, footprint=2, iter=1, **kwargs):
    if isinstance(footprint, numbers.Number):
        footprint = skm.disk(footprint)
    for _ in range(int(iter)):
        image[:] = skm.erosion(image, footprint=footprint, **kwargs)[:]
    return image.astype(np.float32)


@make_da_decorator(np.float32)
def opening(image, footprint=2, **kwargs):
    if isinstance(footprint, numbers.Number):
        footprint = skm.disk(footprint)
    return skm.opening(image, footprint=footprint, **kwargs).astype(np.float32)


@make_da_decorator(np.float32)
def closing(image, footprint=2, **kwargs):
    if isinstance(footprint, numbers.Number):
        footprint = skm.disk(footprint)
    return skm.closing(image, footprint=footprint, **kwargs).astype(np.float32)


@make_da_decorator(np.float32)
def area_closing(image, area_threshold=64, **kwargs):
    if isinstance(area_threshold, numbers.Number):
        area_threshold = skm.disk(area_threshold)
    return skm.area_closing(image, area_threshold=area_threshold, **kwargs).astype(np.float32)


@make_da_decorator(np.float32)
def area_opening(image, area_threshold=64, **kwargs):
    if isinstance(area_threshold, numbers.Number):
        area_threshold = skm.disk(area_threshold)
    return skm.area_opening(image, area_threshold=area_threshold, **kwargs).astype(np.float32)


@make_da_decorator(np.float32)
def diameter_closing(image, diameter_threshold=8, **kwargs):
    if isinstance(diameter_threshold, numbers.Number):
        diameter_threshold = skm.disk(diameter_threshold)
    return skm.area_closing(image, diameter_threshold=diameter_threshold, **kwargs).astype(np.float32)


@make_da_decorator(np.float32)
def diameter_opening(image, diameter_threshold=8, **kwargs):
    if isinstance(diameter_threshold, numbers.Number):
        diameter_threshold = skm.disk(diameter_threshold)
    return skm.area_opening(image, diameter_threshold=diameter_threshold, **kwargs).astype(np.float32)


@make_da_decorator(np.int8)
def binary_opening(image, footprint=2, **kwargs):
    if isinstance(footprint, numbers.Number):
        footprint = skm.disk(footprint)
    return skm.binary_opening(image, footprint=footprint,
                              **kwargs).astype(np.int8)


@make_da_decorator(np.int8)
def binary_closing(image, footprint=2, **kwargs):
    if isinstance(footprint, numbers.Number):
        footprint = skm.disk(footprint)
    return skm.binary_closing(image, footprint=footprint,
                              **kwargs).astype(np.int8)


@make_da_decorator(np.int8)
def binary_dilation(image, footprint=2, iter=1, **kwargs):
    if isinstance(footprint, numbers.Number):
        footprint = skm.disk(footprint)
    for _ in range(int(iter)):
        image[:] = skm.binary_dilation(image, footprint=footprint, **kwargs)[:]
    return image.astype(np.int8)


@make_da_decorator(np.int8)
def binary_erosion(image, footprint=2, iter=1, **kwargs):
    if isinstance(footprint, numbers.Number):
        footprint = skm.disk(footprint)
    for _ in range(int(iter)):
        image[:] = skm.binary_erosion(image, footprint=footprint, **kwargs)[:]
    return image.astype(np.int8)


@make_da_decorator(np.float32)
def remove_small_holes(image, area_threshold=64, **kwargs):
    if isinstance(area_threshold, numbers.Number):
        area_threshold = skm.disk(area_threshold)
    return skm.remove_small_holes(image, area_threshold=area_threshold, **kwargs).astype(np.float32)


@make_da_decorator(np.float32)
def remove_small_objects(image, min_size=64, **kwargs):
    if isinstance(min_size, numbers.Number):
        min_size = skm.disk(min_size)
    return skm.remove_small_objects(image, min_size=min_size, **kwargs).astype(np.float32)


@make_da_decorator(np.int8)
def set_union(A, B):
    return np.int8(((A + B) > 0))


@make_da_decorator(np.int8)
def set_intersection(A, B):
    return np.int8(((A + B) == 2))


@make_da_decorator(np.int8)
def set_difference(A, B):
    return np.int8(((A - B) > 0))


@make_da_decorator(np.int8)
def set_complement(B):
    return np.int8(((1 - B) > 0))


@make_da_decorator(np.int8)
def set_identity(B):
    return B


"""
SET_OP_DICT = {
    'Union (+)': ' + ', 
    'Intersection (x)': ' x ', 
    'Differnece (-)': ' - ', 
    'Complement (I-A)': '.Complement'
}
self.ana_mask_comb_op = {}
self.ana_mask_comb = {}
"""
(
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
)
