#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:50:07 2019

@author: guillaume
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import CPMG_proc2 as proc

def findMaxEcho(dic, FIDapod):
    firstDec = dic['CPMG']['firstDec']
    nbHalfEcho = dic['CPMG']['nbHalfEcho']
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    ms_scale = dic['CPMG']['ms_scale']
    
    maxEcho = np.zeros(int(np.ceil(nbHalfEcho / 2)))
    timeEcho = np.zeros(int(np.ceil(nbHalfEcho / 2)))
    if firstDec == True:
        firstEcho = 0
    else:
        firstEcho = 1
    for i in range(firstEcho, nbHalfEcho + firstEcho, 2):
        maxSlice = slice(i * nbPtHalfEcho, (i+1) * nbPtHalfEcho)
        maxIndex = i * nbPtHalfEcho
        maxEcho[i//2] = max(abs(FIDapod[maxSlice]))             # max of echo
        timeEcho[i//2] = ms_scale[maxIndex]                 # time of echo
    maxEcho = maxEcho / max(maxEcho)                        # normalization
    return timeEcho, maxEcho

def findT2(timeEcho, maxEcho):
    # fitting function and residual calculation
    def fit_func(p, x):
        M0, noise, T2 = p
        fit = M0 * (1-noise) * np.exp(-np.array(x) / T2)
        return fit
    def residuals(p, x, y):
        err = y - fit_func(p, x)
        return err
    p0 = [1.0, maxEcho[-1], timeEcho[timeEcho.size // 2]]   # initial guess
    # fit the trajectory using leastsq (fmin, etc can also be used)
    result = sp.optimize.leastsq(residuals, p0, args=(timeEcho, maxEcho))
    timeFit = np.linspace(timeEcho[0], timeEcho[-1], 100)
    valFit = fit_func(result[0], timeFit)
    plt.scatter(timeEcho, maxEcho)
    plt.plot(timeFit, valFit)
    return result

plt.figure()
nexp = 100
result = np.empty((nexp, 3))
for i in range(nexp):
    dic, FIDraw = proc.data_import()
    FIDshift = proc.shift_FID(dic, FIDraw)
    FIDapod = proc.echo_apod(dic, FIDshift)
    timeEcho, maxEcho = findMaxEcho(dic, FIDapod)
    result[i][0], result[i][1], result[i][2] = findT2(timeEcho, maxEcho)[0]
plt.figure()
plt.plot(FIDapod.real)