#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@authors: Pierre-Aymeric GILLES & Guillaume LAURENT
"""

# Python libraries
import matplotlib.pyplot as plt
import nmrglue as ng
import numpy as np
import sys
# User defined libraries
import CPMG_gen
import postproc

def data_import():
    if len(sys.argv) == 1:
        rawFID = CPMG_gen.main()
    else:
        raise NotImplementedError('Additional arguments are not yet supported')
    return rawFID

def echo_apod(rawFID):
    desc = rawFID.firstDec
    apodFID = rawFID.data.copy()
    for i in range(rawFID.nbHalfEcho):
        halfEcho = slice(i*rawFID.nbPtHalfEcho, (i+1)*rawFID.nbPtHalfEcho)
        apodFID[halfEcho] = (ng.proc_base.sp(
            apodFID[halfEcho], off=0, end=0.5, pow=1.0, rev=desc))
        desc = not(desc)
    end = slice(rawFID.nbHalfEcho * rawFID.nbPtHalfEcho, rawFID.td2 + 1)
    apodFID[end] = 0 + 1j*0
    apodFID = postproc.Signal(
        apodFID, rawFID.dw, rawFID.de,
        rawFID.firstDec, rawFID.nbEcho, rawFID.fullEcho)
    return apodFID

def plot_function(rawFID, apodFID):
    # Zero-filling, Fourier transform and phasing
    rawSPC = postproc.spc(rawFID)
    rawSPC.data = ng.proc_autophase.autops(rawSPC.data, 'acme')
    apodSPC = postproc.spc(apodFID)
    apodSPC.data = ng.proc_autophase.autops(apodSPC.data, 'acme')
    vert_scale_FID = abs(max(rawFID.data.real)) * 1.1
    vert_scale_SPC = abs(max(rawSPC.data.real)) * 1.1
    
    # Plotting
    plt.ion()                               # interactive mode on
    fig1 = plt.figure()
    fig1.suptitle('CPMG NMR signal processing - FID', fontsize=16)
    
    ax1_1 = fig1.add_subplot(411)
    ax1_1.set_title('Raw FID')
    ax1_1.plot(rawFID.ms_scale, rawFID.data.real)
    ax1_1.set_xlim(
        [-rawFID.halfEcho * 1e3, (rawFID.acquiT+rawFID.halfEcho)*1e3])
    ax1_1.set_ylim([-vert_scale_FID, vert_scale_FID])
    
    ax1_2 = fig1.add_subplot(412)
    ax1_2.set_title('Echo apodised FID')
    ax1_2.plot(apodFID.ms_scale, apodFID.data.real)
    ax1_2.set_xlim(
        [-apodFID.halfEcho * 1e3, (apodFID.acquiT+apodFID.halfEcho)*1e3])
    ax1_2.set_ylim([-vert_scale_FID, vert_scale_FID])
    
    fig2 = plt.figure()
    fig2.suptitle('CPMG NMR signal processing - SPC', fontsize=16)
    
    ax2_1 = fig2.add_subplot(411)
    ax2_1.set_title('Raw SPC')
    ax2_1.plot(rawSPC.ppm_scale, rawSPC.data.real)
    ax2_1.invert_xaxis()
    ax2_1.set_ylim([-vert_scale_SPC*0.1, vert_scale_SPC])
    
    ax2_2 = fig2.add_subplot(412)
    ax2_2.set_title('Echo apodised SPC')
    ax2_2.plot(apodSPC.ppm_scale, apodSPC.data.real)
    ax2_2.invert_xaxis()
    ax2_2.set_ylim([-vert_scale_SPC*0.1, vert_scale_SPC])

    # Display figures
    fig1.tight_layout(rect=(0,0,1,0.95))    # Avoid superpositions on figures
    fig2.tight_layout(rect=(0,0,1,0.95))
    fig1.show()
    fig2.show()

def main():
    rawFID = data_import()
    apodFID = echo_apod(rawFID)
    plot_function(rawFID, apodFID)

#%%----------------------------------------------------------------------------
### When this file is executed directly
###----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    input('\nPress enter key to exit')      # wait before closing figures