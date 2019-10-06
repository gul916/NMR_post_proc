#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:50:07 2019

@author: guillaume
"""

import matplotlib.pyplot as plt
import numpy as np
import CPMG_proc2 as proc

def T2_figure(dic):
    """T2 relaxaton fit figure"""
    fitEcho = dic['CPMG']['fitEcho']
    maxEcho = dic['CPMG']['maxEcho']
    timeEcho = dic['CPMG']['timeEcho']
    ms_scale = dic['CPMG']['ms_scale']
    
    timeFit = np.linspace(ms_scale[0], ms_scale[-1], 100)
    valFit = proc.fit_func(fitEcho[0], timeFit)
    plt.scatter(timeEcho, maxEcho)
    plt.plot(timeFit, valFit)

plt.figure()
nexp = 1000
result = np.empty((nexp, 3))
for i in range(nexp):
    dic, FIDref, FIDraw = proc.data_import()
    FIDshift = proc.shift_FID(dic, FIDraw)
    dic, FIDapod = proc.echo_apod(dic, FIDshift, method='exp')
    dic, FIDapod2 = proc.global_apod(dic, FIDapod)
    T2_figure(dic)

plt.gca().set_title('Intensity of echoes')
plt.gca().set_xlabel('Time (ms)')
plt.gca().set_ylabel('Intensity')
plt.tight_layout(rect=(0,0,1,0.95))
plt.show()
