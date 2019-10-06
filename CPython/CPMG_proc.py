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

#%%----------------------------------------------------------------------------
### Importation of data
###----------------------------------------------------------------------------
def data_import():
    if len(sys.argv) == 1:
        dic, FIDref, FIDraw = CPMG_gen.main()
    else:
        raise NotImplementedError('Additional arguments are not yet supported')
    return dic, FIDref, FIDraw

def shift_FID(dic, data):
    ndata = data[:]                                 # avoid data corruption
    nbPtShift = dic['CPMG']['nbPtShift']
    if nbPtShift < 0:                                       # left shift
        ndata = ng.proc_base.ls(ndata, nbPtShift)
    elif nbPtShift > 0:                                     # right shift
        ndata = ng.proc_base.rs(ndata, nbPtShift)
    return ndata

#%%----------------------------------------------------------------------------
### Apodisation
###----------------------------------------------------------------------------
def echo_apod(dic, data, method):
    # Apodisation for each half echo
    ndata = data[:dic['CPMG']['nbPtSignal']]        # discard useless points
    desc = dic['CPMG']['firstDec']
    if method == 'cos':
        dic['CPMG']['apodEcho'] = 'cos'
        apod = np.ones(dic['CPMG']['nbPtSignal']+1) # one additional point
        for i in range(dic['CPMG']['nbHalfEcho']):
            ptHalfEcho = slice(
                i*dic['CPMG']['nbPtHalfEcho'],
                (i+1)*dic['CPMG']['nbPtHalfEcho']+1)
            apod[ptHalfEcho] = ng.proc_base.sp(
                apod[ptHalfEcho], off=0.5, end=1, pow=1.0, rev=(not desc))
            desc = not(desc)
        apod = apod[:-1]            # discard last point used for calculation
    elif method == 'exp':
        T2 = dic['CPMG']['halfEcho']                        # in seconds
        lb = (1 / (np.pi * T2))                             # in Hz
        dic['CPMG']['apodEcho'] = 'LB = {:s} Hz'.format(str(int(lb)))
        lb *= (dic['CPMG']['DW2'])                          # in points
        apod = np.ones(dic['CPMG']['nbPtSignal'])
        for i in range(dic['CPMG']['nbHalfEcho']):
            ptHalfEcho = slice(
                i*dic['CPMG']['nbPtHalfEcho'],
                (i+1)*dic['CPMG']['nbPtHalfEcho'])
            apod[ptHalfEcho] = ng.proc_base.em(
                apod[ptHalfEcho], lb, rev=(not desc))
            desc = not(desc)
    else:
        raise NotImplementedError('Unkown method for echo apodisation')
    ndata = ndata * apod
    return dic, ndata

def findMaxEcho(dic, data):
    ndata = data[:]                                 # avoid data corruption
    nbHalfEcho = dic['CPMG']['nbHalfEcho']
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    firstDec = dic['CPMG']['firstDec']
    ms_scale = dic['CPMG']['ms_scale']
    maxEcho = np.zeros(int(np.ceil(nbHalfEcho / 2)))
    timeEcho = np.zeros(int(np.ceil(nbHalfEcho / 2)))
    for i in range(int(not(firstDec)), nbHalfEcho, 2):
        maxSlice = slice(i * nbPtHalfEcho, (i+1) * nbPtHalfEcho)
        maxIndex = i * nbPtHalfEcho
        maxEcho[i//2] = max(abs(ndata[maxSlice]))           # max of echo
        timeEcho[i//2] = ms_scale[maxIndex]                 # time of echo
    maxEcho = maxEcho / max(maxEcho)                        # normalization
    return timeEcho, maxEcho

def fit_func(p, x):
    M0, T2, noise = p
    fit = M0 * np.exp(-np.array(x) / T2) + noise
    return fit

def findT2(timeEcho, maxEcho):
    # fitting function and residual calculation
    def residuals(p, x, y):
        err = y - fit_func(p, x)
        return err
    p0 = [1.0, timeEcho[timeEcho.size // 2], maxEcho[-1]]   # initial guess
    # fit the trajectory using leastsq (fmin, etc can also be used)
    result = sp.optimize.leastsq(residuals, p0, args=(timeEcho, maxEcho))
    return result

def global_apod(dic, data):
    ndata = data[:]                                 # avoid data corruption
    dureeSignal = 1e3*dic['CPMG']['dureeSignal']
    firstDec = dic['CPMG']['firstDec']
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    firstMin = int(firstDec) * nbPtHalfEcho
    timeEcho, maxEcho = findMaxEcho(dic, ndata)
    fitEcho = findT2(timeEcho, maxEcho)
    T2 = fitEcho[0][1]                                      # in milliseconds
    if 3*T2 > dureeSignal:
        maxSignal = dic['CPMG']['nbPtSignal']
        T2 = min(T2, dureeSignal)
    else:
        maxSignal = np.argwhere(dic['CPMG']['ms_scale'] > 3*T2)[0][0]
    if (maxSignal - firstMin) % (2*nbPtHalfEcho) != 0:
        maxSignal = (
            ((maxSignal - firstMin) // (2*nbPtHalfEcho) + 1)
            *2*nbPtHalfEcho + firstMin)           # round to next full echo
    lb_Hz = (1e3 / (np.pi * T2))                            # in Hz
    lb = lb_Hz * (dic['CPMG']['DW2'])                       # in points
    apod = np.ones(maxSignal)
    apod = ng.proc_base.em(apod, lb)
    ndata = ndata[:maxSignal] * apod                # discard useless points
    nbEchoApod = int((maxSignal - firstMin) // (2*nbPtHalfEcho))
    dic['CPMG']['fitEcho'] = fitEcho
    dic['CPMG']['maxEcho'] = maxEcho
    dic['CPMG']['timeEcho'] = timeEcho
    dic['CPMG']['apodFull'] = 'LB = {:s} Hz'.format(str(int(lb_Hz)))
    dic['CPMG']['nbEchoApod'] = nbEchoApod
    return dic, ndata

#%%----------------------------------------------------------------------------
### Final processing
###----------------------------------------------------------------------------
def trunc(dic, data):
    ndata = data[:]                                 # avoid data corruption
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    firstDec = dic['CPMG']['firstDec']
    firstTop = int(not firstDec)*nbPtHalfEcho
    ndata = ndata[firstTop: firstTop+nbPtHalfEcho]
    return ndata

def echo_sep(dic, data):
    firstDec = dic['CPMG']['firstDec']
    nbEchoApod = dic['CPMG']['nbEchoApod']
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    ndata = np.zeros(
        (nbEchoApod+int(firstDec), nbPtHalfEcho*2), dtype='complex128')
    if firstDec == True:
        ndata[0, nbPtHalfEcho:] = data[:nbPtHalfEcho]
    for row in range(nbEchoApod):
        ndata[row+int(firstDec), :] = (data[
            (2*row+int(firstDec))*nbPtHalfEcho:
            (2*row+int(firstDec)+2)*nbPtHalfEcho])
    ndata2 = np.zeros((nbEchoApod+1, nbPtHalfEcho), dtype='complex128')
    for row in range(nbEchoApod+int(firstDec)):
        ndata2[row, :] = ndata[row, nbPtHalfEcho:]
        ndata2[row, 1:] += (ndata[row, nbPtHalfEcho-1:0:-1].real
            -1j*ndata[row, nbPtHalfEcho-1:0:-1].imag)
    ndata2[0,:] *= int(firstDec) + 1
    return ndata2

def echo_sum(dic, data):
    row, col = data.shape
    ndata = np.zeros(col, dtype='complex128')
    for i in range(row):
        ndata[:] += data[i, :]
    return ndata

#%%----------------------------------------------------------------------------
### Plotting
###----------------------------------------------------------------------------
def FID_figure(dic, A, B, C, D, k_thres):
    #Time domain figure
    acquiT = dic['CPMG']['AQ']
    halfEcho = dic['CPMG']['halfEcho']
    ms_scale = dic['CPMG']['ms_scale']
    vert_scale_FID = max(abs(A)) * 1.1
    fig1 = plt.figure()
    fig1.suptitle('CPMG NMR signal processing - FID', fontsize=16)
    
    ax1_1 = fig1.add_subplot(411)
    ax1_1.set_title('Raw FID')
    ax1_1.plot(ms_scale, A.real)
    ax1_1.set_xlim([-halfEcho * 1e3, (acquiT + halfEcho)*1e3])
    ax1_1.set_ylim([-vert_scale_FID, vert_scale_FID])
    
    ax1_2 = fig1.add_subplot(412)
    ax1_2.set_title(
        'Apodized FID, {:s} and {:s}, {:d} echoes'
        .format(dic['CPMG']['apodEcho'], dic['CPMG']['apodFull'],
        dic['CPMG']['nbEchoApod']))
    ax1_2.plot(ms_scale[:B.size], B.real)
    ax1_2.set_xlim([-halfEcho * 1e3, (acquiT + halfEcho)*1e3])
    ax1_2.set_ylim([-vert_scale_FID, vert_scale_FID])
    
    ax1_3 = fig1.add_subplot(413)
    ax1_3.set_title('Denoised FID, k = {:d}'.format(k_thres))
    ax1_3.plot(ms_scale[:C.size], C.real)
    ax1_3.set_xlim([-halfEcho * 1e3, (acquiT + halfEcho)*1e3])
    ax1_3.set_ylim([-vert_scale_FID, vert_scale_FID])
    
    ax1_4 = fig1.add_subplot(414)
    ax1_4.set_title('Denoised and truncated FID, k = {:d}'.format(k_thres))
    ax1_4.set_xlabel('Time (ms)')
    ax1_4.plot(ms_scale[:D.size], D.real)
    ax1_4.set_xlim([-halfEcho * 1e3, (acquiT + halfEcho)*1e3])
    ax1_4.set_ylim([-vert_scale_FID, vert_scale_FID])
    
    fig1.tight_layout(rect=(0,0,1,0.95))            # Avoid superpositions
    fig1.show()

def SPC_figure(dic, A, B, C, D, k_thres):
    #Frequency domain figure
    # Zero-filling, and Fourier transform
    SPC_A = postproc.postproc_data(dic, A, False)
    SPC_B = postproc.postproc_data(dic, B, False)
    SPC_C = postproc.postproc_data(dic, C, False)
    SPC_D = postproc.postproc_data(dic, D, False)
    Hz_scale = dic['CPMG']['Hz_scale']
    fig2 = plt.figure()
    fig2.suptitle('CPMG NMR signal processing - SPC', fontsize=16)
    
    ax2_1 = fig2.add_subplot(411)
    ax2_1.set_title('Raw SPC')
    ax2_1.plot(Hz_scale, SPC_A.real)
    ax2_1.invert_xaxis()
    
    ax2_2 = fig2.add_subplot(412)
    ax2_2.set_title(
        'Apodized SPC, {:s} and {:s}, {:d} echoes'
        .format(dic['CPMG']['apodEcho'], dic['CPMG']['apodFull'],
        dic['CPMG']['nbEchoApod']))
    ax2_2.plot(Hz_scale, SPC_B.real)
    ax2_2.invert_xaxis()
    
    ax2_3 = fig2.add_subplot(413)
    ax2_3.set_title('Denoised SPC, k = {:d}'.format(k_thres))
    ax2_3.plot(Hz_scale, SPC_C.real)
    ax2_3.invert_xaxis()
    
    ax2_4 = fig2.add_subplot(414)
    ax2_4.set_title('Denoised and truncated SPC, k = {:d}'.format(k_thres))
    ax2_4.set_xlabel('Frequency (Hz)')
    ax2_4.plot(Hz_scale, SPC_D.real)
    ax2_4.invert_xaxis()
    
    fig2.tight_layout(rect=(0,0,1,0.95))            # Avoid superpositions
    fig2.show()

def T2_figure(dic):
    #T2 relaxaton fit figure
    fitEcho = dic['CPMG']['fitEcho']
    maxEcho = dic['CPMG']['maxEcho']
    timeEcho = dic['CPMG']['timeEcho']
    ms_scale = dic['CPMG']['ms_scale']
    
    fig3 = plt.figure()
    ax3_1 = fig3.add_subplot(111)
    ax3_1.set_title(
        'Intensity of echoes, T2 = {:8.2f} ms'.format(fitEcho[0][1]))
    timeFit = np.linspace(ms_scale[0], ms_scale[-1], 100)
    valFit = fit_func(fitEcho[0], timeFit)
    ax3_1.scatter(timeEcho, maxEcho)
    ax3_1.plot(timeFit, valFit)
    ax3_1.set_xlabel('Time (ms)')
    ax3_1.set_ylabel('Intensity')
    
    fig3.tight_layout(rect=(0,0,1,0.95))
    fig3.show()

def mat_figure(dic, E):
    # Separated echos Figure
    row, col = E.shape
    ms_scale = dic['CPMG']['ms_scale']
    Hz_scale = dic['CPMG']['Hz_scale']
    vert_scale_FID = max(abs(E[0,:])) * 1.1
    vert_scale_SPC = max(abs(postproc.postproc_data(dic, E[0, :], False)))*1.1
    
    fig4 = plt.figure()
    ax4_1 = fig4.add_subplot(121)
    ax4_1.set_title('Separated echoes FID')
    ax4_1.set_xlabel('Time (ms)')
    ax4_1.set_ylabel('Intensity')
    ax4_2 = fig4.add_subplot(122)
    ax4_2.set_title('Separated echoes SPC')
    ax4_2.set_xlabel('Frequency (Hz)')
    ax4_2.set_ylabel('Intensity')
    for i in range(row):
        ax4_1.plot(ms_scale[:E[i,:].size], E[i, :].real + i*vert_scale_FID)
        ax4_1.plot(ms_scale[:E[i,:].size], E[i, :].imag + i*vert_scale_FID)
        ax4_2.plot(
            Hz_scale, postproc.postproc_data(dic, E[i, :], False).real
            + i*vert_scale_SPC)
        ax4_2.plot(
            Hz_scale, postproc.postproc_data(dic, E[i, :], False).imag
            + i*vert_scale_SPC)
    
    fig4.tight_layout(rect=(0,0,1,0.95))
    fig4.show()

def comp_figure(dic, B, D, F, G, k_thres):
    # Comparison figure
    # Zero-filling, and Fourier transform
    nbEchoApod = dic['CPMG']['nbEchoApod']
    SPC_B = postproc.postproc_data(dic, B, False)
    SPC_D = postproc.postproc_data(dic, D, False)
    SPC_F = postproc.postproc_data(dic, F, False)
    SPC_G = postproc.postproc_data(dic, trunc(dic, G), False)
    Hz_scale = dic['CPMG']['Hz_scale']
    fig5 = plt.figure()
    fig5.suptitle('CPMG NMR signal processing - comparison', fontsize=16)
    
    ax5_1 = fig5.add_subplot(411)
    ax5_1.set_title(
        'Processed SPC, {:s} and {:s}'
        .format(dic['CPMG']['apodEcho'], dic['CPMG']['apodFull']))
    ax5_1.plot(Hz_scale, SPC_B.real / max(abs(SPC_B)))
    ax5_1.invert_xaxis()

    ax5_2 = fig5.add_subplot(412)
    ax5_2.set_title(
        'Summed SPC, {:d} echoes'.format(nbEchoApod))
    ax5_2.plot(Hz_scale, SPC_F.real / max(abs(SPC_F)))
    ax5_2.invert_xaxis()

    ax5_3 = fig5.add_subplot(413)
    ax5_3.set_title('Denoised SPC, k = {:d}'.format(k_thres))
    ax5_3.plot(Hz_scale, SPC_D.real / max(abs(SPC_D)))
    ax5_3.invert_xaxis()

    ax5_4 = fig5.add_subplot(414)
    ax5_4.set_title('Reference SPC')
    ax5_4.plot(Hz_scale, SPC_G.real / max(abs(SPC_G)))
    ax5_4.invert_xaxis()
    ax5_4.set_xlabel('Frequency (Hz)')

    fig5.tight_layout(rect=(0,0,1,0.95))            # Avoid superpositions
    fig5.show()

def plot_function(dic, A, B, C, D, E, F, G, k_thres):
    """Plotting"""
    plt.ion()                                       # interactive mode on
#    FID_figure(dic, A, B, C, D, k_thres)
#    SPC_figure(dic, A, B, C, D, k_thres)
#    T2_figure(dic)
#    mat_figure(dic, E)
    comp_figure(dic, B, D, F, G, k_thres)

#%%----------------------------------------------------------------------------
### When this file is executed directly
###----------------------------------------------------------------------------
def main():
    dic, FIDref, FIDraw = data_import()
    FIDshift = shift_FID(dic, FIDraw)
    dic, FIDapod = echo_apod(dic, FIDshift, method='exp')
    dic, FIDapod2 = global_apod(dic, FIDapod)
    FIDden, k_thres = denoise_nmr.denoise(
        FIDapod2, k_thres='auto', max_err=5)
    FIDtrunc = trunc(dic, FIDden)
    FIDmat = echo_sep(dic, FIDapod2)
    FIDsum = echo_sum(dic, FIDmat)
    plot_function(
        dic, FIDraw, FIDapod2, FIDden, FIDtrunc, 
        FIDmat, FIDsum, FIDref, k_thres)
    return dic, FIDref, FIDraw

if __name__ == "__main__":
    main()
    input('\nPress enter key to exit')      # wait before closing figures
