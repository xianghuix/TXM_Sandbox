#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:53:10 2020

@author: xiao
"""

import os, subprocess

path = ''
fn_temp = '{0}/recon_fly_scan_id_{1}/recon_fly_scan_id_{1}_.tiff'


for scan_id in range(29774, 29874):
    os.chdir(os.path.dirname(fn_temp.format(path, scan_id)))
    if not os.path.exists(os.path.join(os.path.dirname(fn_temp.format(path, scan_id)), 'sub_dir')):
        os.makedirs(os.path.join(os.path.dirname(fn_temp.format(path, scan_id)), 'sub_dir'))
    curdir = os.path.abspath(os.path.curdir)
    newdir = os.path.join(curdir, "sub_dir")
    fn = fn_temp.format(path, scan_id)
    # for idx in range(215, 385):
    #     subprocess.run(["mv", os.path.join(curdir, "recon_fly_scan_id_{0}_{1}.tiff".format(scan_id, str(idx).zfill(5))), newdir])
    # for idx in range(585, 755):
    #     subprocess.run(["mv", os.path.join(curdir, "recon_fly_scan_id_{0}_{1}.tiff".format(scan_id, str(idx).zfill(5))), newdir])
    # for idx in range(770, 885):
    #     subprocess.run(["mv", os.path.join(curdir, "recon_fly_scan_id_{0}_{1}.tiff".format(scan_id, str(idx).zfill(5))), newdir])

    for idx in range(215, 385):
        subprocess.run(["mv", os.path.join(newdir, "recon_fly_scan_id_{0}_{1}.tiff".format(scan_id, str(idx).zfill(5))), curdir])