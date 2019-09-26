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
        FIDraw = CPMG_gen.main()
    else:
        raise NotImplementedError('Additional arguments are not yet supported')
    return FIDraw

def shift_FID(old):
    new = old.data.copy()
    if old.nbPtDeadTime < 0:        # left shift
        new = ng.proc_base.ls(new, old.nbPtDeadTime)
    elif old.nbPtDeadTime > 0:      # right shift
        new = ng.proc_base.rs(new, old.nbPtDeadTime)
    new = postproc.Signal(
        new, old.dw, old.de,
        old.firstDec, old.nbEcho, old.fullEcho)
    return new

def echo_apod(old):
    new = old.data.copy()
    desc = old.firstDec
    # apodization of each half echo, according to slope
    for i in range(old.nbHalfEcho):
        halfEcho = slice(i*old.nbPtHalfEcho, (i+1)*old.nbPtHalfEcho)
        new[halfEcho] = (ng.proc_base.sp(
            new[halfEcho], off=0, end=0.5, pow=1.0, rev=desc))
        desc = not(desc)
    end = slice(old.nbHalfEcho * old.nbPtHalfEcho, old.td2 + 1)
    new[end] = 0 + 1j*0
    new = postproc.Signal(
        new, old.dw, old.de,
        old.firstDec, old.nbEcho, old.fullEcho)
    return new

def denoise_CPMG(old):
    new = old.data.copy()
    new, k_thres = denoise_nmr.denoise(new, 0, 7.5)
    new = postproc.Signal(
        new, old.dw, old.de,
        old.firstDec, old.nbEcho, old.fullEcho)
    return new, k_thres

def plot_function(FIDraw, FIDshift, FIDapod, FIDden, k_thres):
    # Zero-filling, Fourier transform and phasing
    SPCraw = postproc.spc(FIDraw)
    SPCshift = postproc.spc(FIDshift)
    SPCapod = postproc.spc(FIDapod)
    SPCden = postproc.spc(FIDden)
    SPCraw.data = ng.proc_autophase.autops(SPCraw.data, 'acme')
    SPCshift.data = ng.proc_autophase.autops(SPCshift.data, 'acme')
    SPCapod.data = ng.proc_autophase.autops(SPCapod.data, 'acme')
    SPCden.data = ng.proc_autophase.autops(SPCden.data, 'acme')
    vert_scale_FID = abs(max(FIDraw.data.real)) * 1.1
    vert_scale_SPC = abs(max(SPCraw.data.real)) * 1.1
    
    # Plotting
    plt.ion()                               # interactive mode on
    fig1 = plt.figure()
    fig1.suptitle('CPMG NMR signal processing - FID', fontsize=16)
    
    # FID figure
    ax1_1 = fig1.add_subplot(411)
    ax1_1.set_title('Raw FID')
    ax1_1.plot(FIDraw.ms_scale, FIDraw.data.real)
    ax1_1.set_xlim(
        [-FIDraw.halfEcho * 1e3, (FIDraw.acquiT+FIDraw.halfEcho)*1e3])
    ax1_1.set_ylim([-vert_scale_FID, vert_scale_FID])
    
    ax1_2 = fig1.add_subplot(412)
    ax1_2.set_title('Shifted FID')
    ax1_2.plot(FIDshift.ms_scale, FIDshift.data.real)
    ax1_2.set_xlim(
        [-FIDshift.halfEcho * 1e3, (FIDshift.acquiT+FIDshift.halfEcho)*1e3])
    ax1_2.set_ylim([-vert_scale_FID, vert_scale_FID])
    
    ax1_3 = fig1.add_subplot(413)
    ax1_3.set_title('Echo apodised FID')
    ax1_3.plot(FIDapod.ms_scale, FIDapod.data.real)
    ax1_3.set_xlim(
        [-FIDapod.halfEcho * 1e3, (FIDapod.acquiT+FIDapod.halfEcho)*1e3])
    ax1_3.set_ylim([-vert_scale_FID, vert_scale_FID])
    
    ax1_4 = fig1.add_subplot(414)
    ax1_4.set_title('Denoised FID')
    ax1_4.plot(FIDden.ms_scale, FIDden.data.real)
    ax1_4.set_xlim(
        [-FIDden.halfEcho * 1e3, (FIDden.acquiT+FIDden.halfEcho)*1e3])
    ax1_4.set_ylim([-vert_scale_FID, vert_scale_FID])
    
    # SPC figure
    fig2 = plt.figure()
    fig2.suptitle('CPMG NMR signal processing - SPC', fontsize=16)
    
    ax2_1 = fig2.add_subplot(411)
    ax2_1.set_title('Raw SPC')
    ax2_1.plot(SPCraw.ppm_scale, SPCraw.data.real)
    ax2_1.invert_xaxis()
    ax2_1.set_ylim([-vert_scale_SPC*0.1, vert_scale_SPC])
    
    ax2_2 = fig2.add_subplot(412)
    ax2_2.set_title('Shifted SPC')
    ax2_2.plot(SPCshift.ppm_scale, SPCshift.data.real)
    ax2_2.invert_xaxis()
    ax2_2.set_ylim([-vert_scale_SPC*0.1, vert_scale_SPC])

    ax2_3 = fig2.add_subplot(413)
    ax2_3.set_title('Echo apodised SPC')
    ax2_3.plot(SPCapod.ppm_scale, SPCapod.data.real)
    ax2_3.invert_xaxis()
    ax2_3.set_ylim([-vert_scale_SPC*0.1, vert_scale_SPC])
    
    ax2_4 = fig2.add_subplot(414)
    ax2_4.set_title('Denoised SPC')
    ax2_4.plot(SPCden.ppm_scale, SPCden.data.real)
    ax2_4.invert_xaxis()
    ax2_4.set_ylim([-vert_scale_SPC*0.1, vert_scale_SPC])
    
    # Display figures
    fig1.tight_layout(rect=(0,0,1,0.95))    # Avoid superpositions on figures
    fig2.tight_layout(rect=(0,0,1,0.95))
    fig1.show()
    fig2.show()

def main():
    FIDraw = data_import()
    FIDshift = shift_FID(FIDraw)
    FIDapod = echo_apod(FIDshift)
    FIDden, k_thres = denoise_CPMG(FIDapod)
    plot_function(FIDraw, FIDshift, FIDapod, FIDden, k_thres)

#%%----------------------------------------------------------------------------
### When this file is executed directly
###----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    input('\nPress enter key to exit')      # wait before closing figures