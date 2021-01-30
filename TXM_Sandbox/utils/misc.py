#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 17:48:06 2021

@author: xiao
"""
import time
import functools

def msgit(wd=100, fill='-'):
    def decorator_msgit(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            print((f'{func.__name__} starts at {time.asctime()}'.center(wd, fill)))
            rlt = func(*args, **kwargs)
            print((f'{func.__name__} finishes at {time.asctime()}'.center(wd, fill)))
            return rlt
        return inner
    return decorator_msgit
    