#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 22:20:14 2020

@author: xiao
"""

from setuptools import setup, find_packages

setup(name='TXM_Sandbox',
      version='0.1.5',
      description='Integrated Spectro-Imaging Analysis Toolbox',
      url='https://github.com/xianghuix/TXM_Sandbox',
      author='Xianghui Xiao',
      author_email='xianghuix@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'python>=3.8',
            'pyimagej',
            'nodejs',
            'xarray',
            'tomopy',
            'h5py',
            'tifffile',
            'imutils',
            'opencv',
            'openpyxl',
            'pytables',
            'pandas',
            'jupyter',
            'jupyterlab',
            'multiprocess',
            'scikit-image',
            'scikit-learn',
            'pystackreg',
            'silx',
            'napari'
      ],
      zip_safe=False,
      include_package_data=True,
      package_data={
          "":["LICENSE", "README.md", "TXM_Sandbox/tmp/readme.txt", "TXM_Sandbox/config/*.json", "TXM_GUI2.ipynb"]})
