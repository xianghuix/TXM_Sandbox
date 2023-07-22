#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:56:17 2020

@author: xiao
"""
from .gen_algn_gui import gen_algn_gui
from .conv_data_gui import conv_data_gui


class misc_gui():

    def __init__(self, parent_h, form_sz=[650, 740]):
        self.gui_name = 'misc'
        self.form_sz = form_sz
        self.global_h = parent_h
        self.hs = {}

    def build_gui(self):
        self.gen_algn_gui = gen_algn_gui(
            self.global_h, self.form_sz)
        self.gen_algn_gui.build_gui()
        self.hs['GenImgAlign form'] = self.gen_algn_gui.hs['GenImgAlign form']

        self.conv_data_gui = conv_data_gui(self.global_h, self.form_sz)
        self.conv_data_gui.build_gui()
        self.hs['ConvertData form'] = self.conv_data_gui.hs['ConvertData form']
