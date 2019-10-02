#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@authors: Pierre-Aymeric GILLES & Guillaume LAURENT
"""

# Python libraries
import matplotlib.pyplot as plt
import numpy as np
# User defined library
import postproc

###----------------------------------------------------------------------------
### PARAMETERS
###----------------------------------------------------------------------------

# Asked to the user
firstDec = True
fullEcho = 10e-3
nbEcho = 4

# From Topspin interface
td = 1000                       # nb of real points + nb of imag points
dw = 50e-6                      # dwell time between two points
de = 100e-6                     # dead time before signal acquisition
#de = 0

# Simulation of noise
mean = 0
std = 0.3

# 1st frequency
amp1 = 1                        # amplitude
nu1 = 1750                      # frequency in Hz
t21 = 30e-3                     # true T2 relaxation time
t21star = 1e-3                  # apparent T2 relaxation time
sigma1 = np.sqrt(2*np.log(2))*t21star       # for Gaussian shape
gl1 = 1                         # 0: Lorentzian shape, 1: Gaussian shape

# 2nd frequency
amp2 = 1                        # amplitude
nu2 = -3000                     # frequency in Hz
t22 = 100e-3                     # true T2 relaxation time
t22star = 1e-3                  # apparent T2 relaxation time
sigma2 = np.sqrt(2*np.log(2))*t22star;      # for Gaussian shape
gl2 = 1                         # 0: Lorentzian shape, 1: Gaussian shape

# Calculated
halfEcho = fullEcho / 2
nbHalfEcho = nbEcho * 2
if firstDec == True:
    nbHalfEcho += 1

dw2 = dw * 2        # 1 real + 1 imag points are needed to have a complex point
td2 = td // 2                   # nb of complex points
acquiT = (td2 - 1) * dw2        # total acquisition duration, starts at 0
nbPtHalfEcho = int(halfEcho / dw2)
nbPtSignal = nbPtHalfEcho * nbHalfEcho
dureeSignal = (nbPtSignal -1) * dw2     # signal duration, starts at 0
nbPtLast = td2 - nbPtSignal     # Number of points after last echo
nbPtShift = int(de / dw2)       # Number of point during dead time

if (gl1 < 0) or (gl1 > 1):
    raise ValueError('gl1 must be between 0 and 1')
if (gl2 < 0) or (gl2 > 1):
    raise ValueError('gl2 must be between 0 and 1')

#%%---------------------------------------------------------------------------
### SYNTHESE DE SIGNAL RMN
###----------------------------------------------------------------------------
def signal_generation():
    desc = firstDec
    Aref = np.array([])

    # tracé de la courbe par les demi echos
    for i in range (0, nbHalfEcho):
        deb = i*halfEcho
        fin = (i+1)*halfEcho-dw2
        timei = np.linspace(deb,fin,nbPtHalfEcho)
        if desc == True:
            tzero = i*halfEcho
        else:
            tzero = (i+1)*halfEcho
        
        # Gaussian-Lorentzian broadening (t2star)
        # with exponential relaxation (t2)
        yi1 = (
            amp1 * np.exp(1j*2*np.pi*nu1*(timei-tzero))
            * ((1-gl1)*np.exp(-abs(timei-tzero)/t21star)
            + (gl1)*np.exp(-(timei-tzero)**2/(2*sigma1**2)))
            * np.exp(-(timei)/t21))
        yi2 = (
            amp2 * np.exp(1j*2*np.pi*nu2*(timei-tzero))
            * ((1-gl2)*np.exp(-abs(timei-tzero)/t22star)
            + (gl2)*np.exp(-(timei-tzero)**2/(2*sigma2**2)))
            * np.exp(-(timei)/t22))
        yi = yi1 + yi2
        Aref = np.concatenate((Aref, yi))
        desc = not(desc)
    
    # Final points
    end = np.zeros(nbPtLast, dtype=np.complex)
    Aref = np.concatenate((Aref,end))

    # Suppression of points during dead time 
    end = np.zeros(nbPtShift, dtype=np.complex)
    Adead = np.concatenate((Aref[nbPtShift:],end))

    # Adding noise
    noise = (
        np.random.normal(mean, std, td2)
        + 1j*np.random.normal(mean, std, td2))
    Anoisy = Adead + noise
    
    # Convert to a dictionary containing all parameters
    dic = postproc.CPMG_pseudo_dic(Aref, dw2)
    dic = postproc.CPMG_dic(
        dic, Aref, fullEcho, nbEcho, firstDec, nbPtShift)
    return dic, Aref, Adead, Anoisy

def plot_function(dic, Aref, Adead, Anoisy):
    # keep only a half echo for ArefSPC and normalization
    if firstDec == True:
        ArefSPC = Aref[:nbPtHalfEcho] * nbHalfEcho
    else:
        ArefSPC = Aref[nbPtHalfEcho:2*nbPtHalfEcho] * nbHalfEcho
    # Zero-filling and Fourier transform
    ArefSPC = postproc.postproc_data(dic, ArefSPC, False)
    AnoisySPC = postproc.postproc_data(dic, Anoisy, False)
    ms_scale = dic['CPMG']['ms_scale']
    Hz_scale = dic['CPMG']['Hz_scale']
    
    # Plotting
    plt.ion()                           # interactive mode on
    fig1 = plt.figure()
    fig1.suptitle('CPMG NMR signal synthesis', fontsize=16)
    vert_scale = abs(max(Aref.real)) * 1.1
    
    # Reference FID display
    ax1 = fig1.add_subplot(411)
    ax1.set_title('Reference FID')
    ax1.plot(ms_scale, Aref.real)
    ax1.plot(ms_scale, Aref.imag)
    ax1.set_xlim([-halfEcho * 1e3, (acquiT+halfEcho)*1e3])
    ax1.set_ylim([-vert_scale, vert_scale])
    
    # FID display after dead time suppression
    ax2 = fig1.add_subplot(412)
    ax2.set_title('FID after dead time suppression')
    ax2.plot(ms_scale, Adead.real)
    ax2.plot(ms_scale, Adead.imag)
    ax2.set_xlim([-halfEcho * 1e3, (acquiT+halfEcho)*1e3])
    ax2.set_ylim([-vert_scale, vert_scale])
    
    # FID display after dead time suppression and noise addition
    ax3 = fig1.add_subplot(413)
    ax3.set_title('FID after addition of noise')
    ax3.plot(ms_scale, Anoisy.real)
    ax3.plot(ms_scale, Anoisy.imag)
    ax3.set_xlim([-halfEcho * 1e3, (acquiT+halfEcho)*1e3])
    ax3.set_ylim([-vert_scale, vert_scale])
    
    # Spectra display
    ax4 = fig1.add_subplot(414)
    ax4.set_title('Noisy SPC and reference SPC')
    ax4.plot(Hz_scale, AnoisySPC[::-1].real)
    ax4.plot(Hz_scale, ArefSPC[::-1].real)
    ax4.invert_xaxis()
    
    # Avoid superpositions on figure
    fig1.tight_layout(rect=(0,0,1,0.95))
    fig1.show()                 # Display figure
    
def main():
    dic, Aref, Adead, Anoisy = signal_generation()
    plot_function(dic, Aref, Adead, Anoisy)
    return dic, Anoisy

#%%----------------------------------------------------------------------------
### When this file is executed directly
###----------------------------------------------------------------------------

if __name__ == "__main__":
    dic, Anoisy = main()
    input('\nPress enter key to exit') # wait before closing figure
