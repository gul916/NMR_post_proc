#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 10:58:54 2018
@author: guillaume
"""

import matplotlib.pyplot as plt
import nmrglue as ng
import numpy as np
import os.path
import sys

# Customize figures
plt.rc('font', family='Times New Roman', size=16)       # default text
plt.rc('mathtext', default='regular')                   # default equations

def data_import(data_dir):
    dic, data = ng.bruker.read_pdata(data_dir)        # import data
    if data.ndim != 1:
        raise NotImplementedError \
            ("SNR on", data.ndim, "dimensions is not yet supported.")
    udic = ng.bruker.guess_udic(dic, data)    # convert to universal dictionary
    uc = ng.fileiobase.uc_from_udic(udic)
    ppm_scale = uc.ppm_scale()
    plt.ion()                                   # to avoid stop when plotting
    fig, ax = plt.subplots()
    ax.plot(ppm_scale, data.real)
    ax.get_yaxis().set_visible(False)
#    ax.get_xaxis().set_visible(False)
    ax.axis([ppm_scale[0], ppm_scale[-1], \
             np.min(data.real)*1.1, np.max(data.real)*1.1])
    plt.pause(0.1)                                      # to update display
    return ppm_scale, data

def psnr_limit(ppm_scale, data):
    while True:
        signal_lim = np.zeros(2)
        signal_lim[0] = input('Enter left signal limit (in ppm): ')
        signal_lim[1] = input('Enter right signal limit (in ppm): ')
        if (ppm_scale[0] >= signal_lim[0] >= ppm_scale[-1]) \
            and (ppm_scale[0] >= signal_lim[1] >= ppm_scale[-1]) \
            and (signal_lim[0] > signal_lim[1]):
            break
        else:
            print('\nError: Invalid signal region. Please try again.')
    ax = plt.gca()
    ax.axvline(x=signal_lim[0], color='g', linestyle=':', linewidth=2)
    ax.axvline(x=signal_lim[1], color='g', linestyle=':', linewidth=2)
    ax.text(np.mean(signal_lim), np.max(data)*1.03, 'signal', \
        color='g', ha='center')
    plt.pause(0.1)                                      # to update display
    
    while True:
        noise_lim = np.zeros(2)
        noise_lim[0] = input('Enter left noise limit (in ppm): ')
        noise_lim[1] = input('Enter right noise limit (in ppm): ')
        if (ppm_scale[0] >= noise_lim[0] >= ppm_scale[-1]) \
            and (ppm_scale[0] >= noise_lim[1] >= ppm_scale[-1]) \
            and (noise_lim[0] > noise_lim[1]):
            break
        else:
            print('\nError: Invalid noise region. Please try again.')
    ax.axvline(x=noise_lim[0], color='r', linestyle=':', linewidth=2)
    ax.axvline(x=noise_lim[1], color='r', linestyle=':', linewidth=2)
    ax.text(np.mean(noise_lim), np.max(data)*1.03, 'noise', \
        color='r', ha='center')
    plt.pause(0.1)                                      # to update display
    
    for i in range(2):
        signal_lim[i] = np.floor((ppm_scale[0]-signal_lim[i]) \
              * ppm_scale.size / (ppm_scale[0]-ppm_scale[-1]))
        noise_lim[i] = np.floor((ppm_scale[0]-noise_lim[i]) \
              * ppm_scale.size / (ppm_scale[0]-ppm_scale[-1]))
    signal_lim = signal_lim.astype(int)
    noise_lim = noise_lim.astype(int)
    return signal_lim, noise_lim

def signal_noise(ppm_scale, data, signal_lim, noise_lim):
    signal = data[signal_lim[0]:signal_lim[1]].real
    noise = data[noise_lim[0]:noise_lim[1]].real
    noise_mean = np.mean(noise)
    noise_rms = np.std(noise, ddof=1)
    noise_max = (np.max(noise) - np.min(noise)) / 2
    signal_rms = np.std(signal, ddof=1)
    signal_max = np.max(signal - noise_mean)
    snnr = signal_rms / noise_rms
    snr = snnr**2 - 1
    psnr_max = signal_max / noise_max
    psnr_rms = signal_max / (noise_rms *2)
    Lc = noise_mean + noise_rms * 1.64                  # critical limit
    Ld = noise_mean + noise_rms * 3.29                  # detection limit
    Lq = noise_mean + noise_rms * 10                    # quantification limit
    txt_xpos = ppm_scale[-1] - 0.01 * (ppm_scale[0]-ppm_scale[-1])
    max_y = np.max([np.max(data.real)*1.1, Lq*1.1])
    
    ax = plt.gca()
    plt.title(r'${PSNR}_{rms} = $' + '{:.1f}, '.format(psnr_rms) \
        + r'${PSNR}_{max} = $' + '{:.1f}, '.format(psnr_max) \
        + r'$SNR = $' + '{:.1f}'.format(snr))
#    plt.title(r'${PSNR}_{rms} = $' + '{:.1f}'.format(psnr_rms))
    ax.axhline(y=Lc, color='tab:red', linestyle=':', linewidth=2)
    ax.axhline(y=Ld, color='tab:orange', linestyle=':', linewidth=2)
    ax.axhline(y=Lq, color='tab:green', linestyle=':', linewidth=2)
    ax.text(txt_xpos, Lc+0.01*np.max(data), r'$L_c$', color='tab:red')
    ax.text(txt_xpos, Ld+0.01*np.max(data), r'$L_d$', color='tab:orange')
    ax.text(txt_xpos, Lq+0.01*np.max(data), r'$L_q$', color='tab:green')
    ax.axis([ppm_scale[0], ppm_scale[-1], np.min(data.real)*1.1, max_y])
    plt.tight_layout()                              # to avoid superpositions
    plt.pause(0.1)                                  # to update display
    plt.ioff()                                      # to avoid figure closing
    plt.show()                                      # to allow zooming

def main():
    print('\n\n\n---------------------------------------------')    # line jump
    try:
        # Data directory has to be provided as an argument
        if (len(sys.argv) >= 2) and (os.path.isdir(sys.argv[1])):
            data_dir = str(sys.argv[1])
        else:
            raise ValueError \
                ('Please provide SPC data directory as first argument')
        
        ppm_scale, data = data_import(data_dir)
        signal_lim, noise_lim = psnr_limit(ppm_scale, data)
        signal_noise(ppm_scale, data, signal_lim, noise_lim)
        
    except ValueError as err:                           # arguments
        print('Error:', err)
        for i in range(1, len(sys.argv)):
            print('Argument', i, '=', sys.argv[i])
    print('---------------------------------------------')          # line jump

#%%----------------------------------------------------------------------------
### IF PROGRAM IS DIRECTLY EXECUTED
###----------------------------------------------------------------------------
if __name__ == '__main__':
    main()