#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@authors: Pierre-Aymeric GILLES & Guillaume LAURENT
"""

# Python libraries
import matplotlib.pyplot as plt
import nmrglue as ng
import numpy as np
import scipy as sp
import sys
# User defined libraries
import CPMG_gen
import denoise_nmr
import postproc

def data_import():
    if len(sys.argv) == 1:
        dic, FIDraw = CPMG_gen.main()
    else:
        raise NotImplementedError('Additional arguments are not yet supported')
    return dic, FIDraw

def shift_FID(dic, old):
    new = old.copy()
    nbPtShift = dic['CPMG']['nbPtShift']
    if nbPtShift < 0:                                       # left shift
        new = ng.proc_base.ls(nbPtShift)
    elif nbPtShift > 0:                                     # right shift
        new = ng.proc_base.rs(new, nbPtShift)
    return new

def echo_apod(dic, old):
    new = old.copy()
    new = new[:dic['CPMG']['nbPtSignal']]
    desc = dic['CPMG']['firstDec']
    # apodization of each half echo
    for i in range(dic['CPMG']['nbHalfEcho']):
        ptHalfEcho = slice(
            i*dic['CPMG']['nbPtHalfEcho'], (i+1)*dic['CPMG']['nbPtHalfEcho'])
        new[ptHalfEcho] = ng.proc_base.sp(
            new[ptHalfEcho], off=0, end=0.5, pow=1.0, rev=desc)
        desc = not(desc)
    return new

def findMaxEcho(dic, old):
    new = old.copy()
    nbHalfEcho = dic['CPMG']['nbHalfEcho']
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    ms_scale = dic['CPMG']['ms_scale']
    maxEcho = np.zeros(int(np.ceil(nbHalfEcho / 2)))
    timeEcho = np.zeros(int(np.ceil(nbHalfEcho / 2)))
    if dic['CPMG']['firstDec'] == True:
        firstEcho = 0
    else:
        firstEcho = 1
    for i in range(firstEcho, nbHalfEcho + firstEcho, 2):
        maxSlice = slice(i * nbPtHalfEcho, (i+1) * nbPtHalfEcho)
        maxIndex = i * nbPtHalfEcho
        maxEcho[i//2] = max(abs(new[maxSlice]))             # max of echo
        timeEcho[i//2] = ms_scale[maxIndex]                 # time of echo
    maxEcho = maxEcho / max(maxEcho)                        # normalization
    return timeEcho, maxEcho

def findT2(timeEcho, maxEcho):
    # fitting function and residual calculation
    def fit_func(p, x):
        M0, T2, noise = p
        fit = M0 * np.exp(-np.array(x) / T2) + noise
        return fit
    def residuals(p, x, y):
        err = y - fit_func(p, x)
        return err
    p0 = [1.0, timeEcho[timeEcho.size // 2], maxEcho[-1]]   # initial guess
    # fit the trajectory using leastsq (fmin, etc can also be used)
    result = sp.optimize.leastsq(residuals, p0, args=(timeEcho, maxEcho))
    timeFit = np.linspace(timeEcho[0], timeEcho[-1], 100)
    valFit = fit_func(result[0], timeFit)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(timeEcho, maxEcho)
    ax.plot(timeFit, valFit)
    ax.set_title('Intensity of echoes, T2 = {:8.2f} ms'.format(result[0][1]))
    return result

def global_apod(dic, old):
    new = old.copy()
    timeEcho, maxEcho = findMaxEcho(dic, new)
    T2 = findT2(timeEcho, maxEcho)[0][1]                    # in milliseconds
    maxSignal = np.argwhere(dic['CPMG']['ms_scale'] > 1.26*T2)[0][0]
    print(maxSignal)
    lb = (1e3 / (np.pi * T2))                               # in Hz
    lb *= (dic['CPMG']['DW2'])                              # in points
    new = ng.proc_base.em(new[:maxSignal], lb)
    return new

def trunc(dic, old):
    new = old.copy()
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    if dic['CPMG']['firstDec'] == True:
        new = new[:nbPtHalfEcho]
    else:
        new = new[nbPtHalfEcho:2*nbPtHalfEcho]
    return new

def plot_function(dic, FIDshift, FIDapod2, FIDden, FIDtrunc, k_thres):
    # Zero-filling, Fourier transform and phasing
    SPCshift = postproc.postproc_data(dic, FIDshift, False)
    SPCapod2 = postproc.postproc_data(dic, FIDapod2, False)
    SPCden = postproc.postproc_data(dic, FIDden, False)
    SPCtrunc = postproc.postproc_data(dic, FIDtrunc, False)
    
    # Scaling
    acquiT = dic['CPMG']['AQ']
    halfEcho = dic['CPMG']['halfEcho']
    ms_scale = dic['CPMG']['ms_scale']
    Hz_scale = dic['CPMG']['Hz_scale']
    vert_scale_FID = max(abs(FIDshift)) * 1.1
    vert_scale_SPC = max(abs(SPCshift)) * 1.1
    
    # Plotting
    plt.ion()                               # interactive mode on
    fig1 = plt.figure()
    fig1.suptitle('CPMG NMR signal processing - FID', fontsize=16)
    
    # FID figure
    ax1_1 = fig1.add_subplot(411)
    ax1_1.set_title('Shifted FID')
    ax1_1.plot(ms_scale, FIDshift.real)
    ax1_1.set_xlim([-halfEcho * 1e3, (acquiT + halfEcho)*1e3])
    ax1_1.set_ylim([-vert_scale_FID, vert_scale_FID])
    
    ax1_2 = fig1.add_subplot(412)
    ax1_2.set_title('Echo + global apodised FID')
    ax1_2.plot(ms_scale[:FIDapod2.size], FIDapod2.real)
    ax1_2.set_xlim([-halfEcho * 1e3, (acquiT + halfEcho)*1e3])
    ax1_2.set_ylim([-vert_scale_FID, vert_scale_FID])
    
    ax1_3 = fig1.add_subplot(413)
    ax1_3.set_title('Denoised FID, k = {:d}'.format(k_thres))
    ax1_3.plot(ms_scale[:FIDden.size], FIDden.real)
    ax1_3.set_xlim([-halfEcho * 1e3, (acquiT + halfEcho)*1e3])
    ax1_3.set_ylim([-vert_scale_FID, vert_scale_FID])
    
    ax1_4 = fig1.add_subplot(414)
    ax1_4.set_title('Denoised and truncated FID, k = {:d}'.format(k_thres))
    ax1_4.plot(ms_scale[:FIDtrunc.size], FIDtrunc.real)
    ax1_4.set_xlim([-halfEcho * 1e3, (acquiT + halfEcho)*1e3])
    ax1_4.set_ylim([-vert_scale_FID, vert_scale_FID])

    # SPC figure
    fig2 = plt.figure()
    fig2.suptitle('CPMG NMR signal processing - SPC', fontsize=16)
    
    ax2_1 = fig2.add_subplot(411)
    ax2_1.set_title('Shifted SPC')
    ax2_1.plot(Hz_scale, SPCshift.real)
    ax2_1.invert_xaxis()
    ax2_1.set_ylim([-vert_scale_SPC * 0.1, vert_scale_SPC])
    
    ax2_2 = fig2.add_subplot(412)
    ax2_2.set_title('Echo + global apodised SPC')
    ax2_2.plot(Hz_scale, SPCapod2.real)
    ax2_2.invert_xaxis()
    ax2_2.set_ylim([-vert_scale_SPC * 0.1, vert_scale_SPC])
    
    ax2_3 = fig2.add_subplot(413)
    ax2_3.set_title('Denoised SPC, k = {:d}'.format(k_thres))
    ax2_3.plot(Hz_scale, SPCden.real)
    ax2_3.invert_xaxis()
    ax2_3.set_ylim([-vert_scale_SPC * 0.1, vert_scale_SPC])
    
    ax2_4 = fig2.add_subplot(414)
    ax2_4.set_title('Denoised and truncated SPC, k = {:d}'.format(k_thres))
    ax2_4.plot(Hz_scale, SPCtrunc.real)
    ax2_4.invert_xaxis()
    
    # Display figures
    fig1.tight_layout(rect=(0,0,1,0.95))    # Avoid superpositions on figures
    fig2.tight_layout(rect=(0,0,1,0.95))
    fig1.show()
    fig2.show()

def main():
    dic, FIDraw = data_import()
    FIDshift = shift_FID(dic, FIDraw)
    FIDapod = echo_apod(dic, FIDshift)
    FIDapod2 = global_apod(dic, FIDapod)
    FIDden, k_thres = denoise_nmr.denoise(FIDapod)
    FIDtrunc = trunc(dic, FIDden)
    plot_function(dic, FIDshift, FIDapod2, FIDden, FIDtrunc, k_thres)
    return dic, FIDraw

#%%----------------------------------------------------------------------------
### When this file is executed directly
###----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    input('\nPress enter key to exit')      # wait before closing figures