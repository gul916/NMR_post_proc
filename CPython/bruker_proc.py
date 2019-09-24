#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:46:07 2019

@author: guillaume
"""

import matplotlib.pyplot as plt
import nmrglue as ng

data_dir = '/home/guillaume/Data_NMR/300WB_3.2/data/Guillaume/nmr/20140522-TEOS-MTEOS-50-50-4MQ-500/2013/pdata/1'
dic, data = ng.bruker.read(data_dir)

data = ng.bruker.remove_digital_filter(dic, data)
data = ng.proc_base.sp(data, off=0.5, end=1.0, pow=1.0)
data = ng.proc_base.zf_auto(data)
data = ng.proc_base.zf_double(data, 2)
data = ng.proc_base.fft_norm(data)
data = ng.proc_base.rev(data)
data = ng.proc_autophase.autops(data, 'acme')

udic = ng.bruker.guess_udic(dic, data)
uc = ng.fileiobase.uc_from_udic(udic)
ppm_scale = uc.ppm_scale()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ppm_scale, data.real)
ax.invert_xaxis()

ng.bruker.write_pdata(data_dir, dic, data, overwrite=True)