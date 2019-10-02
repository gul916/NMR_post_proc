#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@authors: Pierre-Aymeric GILLES & Guillaume LAURENT
"""

# Python libraries
import matplotlib.pyplot as plt
import nmrglue as ng
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
        new[ptHalfEcho] = (ng.proc_base.sp(
            new[ptHalfEcho], off=0, end=0.5, pow=1.0, rev=desc))
        desc = not(desc)
    return new

def denoise_CPMG(old):
    new = old.copy()
    new, k_thres = denoise_nmr.denoise(new)
    return new, k_thres

def trunc(dic, old):
    new = old.copy()
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    if dic['CPMG']['firstDec'] == True:
        new = new[:nbPtHalfEcho]
    else:
        new = new[nbPtHalfEcho:2*nbPtHalfEcho]
    return new

def plot_function(dic, FIDraw, FIDapod, FIDden, FIDtrunc, k_thres):
    # Zero-filling, Fourier transform and phasing
    SPCraw = postproc.postproc_data(dic, FIDraw, False)
    SPCapod = postproc.postproc_data(dic, FIDapod, False)
    SPCden = postproc.postproc_data(dic, FIDden, False)
    SPCtrunc = postproc.postproc_data(dic, FIDtrunc, False)
    
    # Scaling
    acquiT = dic['CPMG']['AQ']
    halfEcho = dic['CPMG']['halfEcho']
    ms_scale = dic['CPMG']['ms_scale']
    Hz_scale = dic['CPMG']['Hz_scale']
    vert_scale_FID = abs(max(FIDraw.real)) * 1.1
    vert_scale_SPC = abs(max(SPCraw.real)) * 1.1
    
    # Plotting
    plt.ion()                               # interactive mode on
    fig1 = plt.figure()
    fig1.suptitle('CPMG NMR signal processing - FID', fontsize=16)
    
    # FID figure
    ax1_1 = fig1.add_subplot(411)
    ax1_1.set_title('Raw FID')
    ax1_1.plot(ms_scale, FIDraw.real)
    ax1_1.set_xlim([-halfEcho * 1e3, (acquiT + halfEcho)*1e3])
    ax1_1.set_ylim([-vert_scale_FID, vert_scale_FID])
    
    ax1_2 = fig1.add_subplot(412)
    ax1_2.set_title('Shift + echo apodised FID')
    ax1_2.plot(ms_scale[:FIDapod.size], FIDapod.real)
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
    ax2_1.set_title('Raw SPC')
    ax2_1.plot(Hz_scale, SPCraw[::-1].real)
    ax2_1.invert_xaxis()
    ax2_1.set_ylim([-vert_scale_SPC, vert_scale_SPC])
    
    ax2_2 = fig2.add_subplot(412)
    ax2_2.set_title('Shift + echo apodised SPC')
    ax2_2.plot(Hz_scale, SPCapod[::-1].real)
    ax2_2.invert_xaxis()
    ax2_2.set_ylim([-vert_scale_SPC, vert_scale_SPC])
    
    ax2_3 = fig2.add_subplot(413)
    ax2_3.set_title('Denoised SPC, k = {:d}'.format(k_thres))
    ax2_3.plot(Hz_scale, SPCden[::-1].real)
    ax2_3.invert_xaxis()
    ax2_3.set_ylim([-vert_scale_SPC, vert_scale_SPC])
    
    ax2_4 = fig2.add_subplot(414)
    ax2_4.set_title('Denoised and truncated SPC, k = {:d}'.format(k_thres))
    ax2_4.plot(Hz_scale, SPCtrunc[::-1].real)
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
    FIDden, k_thres = denoise_CPMG(FIDapod)
    FIDtrunc = trunc(dic, FIDden)
    plot_function(dic, FIDraw, FIDapod, FIDden, FIDtrunc, k_thres)

#%%----------------------------------------------------------------------------
### When this file is executed directly
###----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    input('\nPress enter key to exit')      # wait before closing figures