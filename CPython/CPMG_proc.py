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
### Importation and exportation of data
###----------------------------------------------------------------------------
def data_import():
    """Import or create data"""
    if len(sys.argv) == 1:                                  # create a noisy FID
        dic, FIDref, FIDraw = CPMG_gen.main()
    elif len(sys.argv) == 6:                                # import data
        data_dir = sys.argv[1]
        fullEcho = sys.argv[2]
        nbEcho = sys.argv[3]
        firstDec = sys.argv[4]
        nbPtShift = sys.argv[5]
        dic, FIDraw = postproc.import_data(data_dir)
        dic = postproc.CPMG_dic(
            dic, FIDraw, fullEcho, nbEcho, firstDec, nbPtShift)
        FIDref = FIDraw[:]
    else:
        raise NotImplementedError(
            'Arguments should be data, fullEcho, nbEcho, firstDec, nbPtShift')
    return dic, FIDref, FIDraw

def data_export(dic, data):
    """Export data"""
    if len(sys.argv) > 1:                                  # save spectrum
        data_dir = sys.argv[1]
        postproc.export_data(dic, data, data_dir)

def shift_FID(dic, data):
    """Correct dead time and echo delay"""
    ndata = data[:]                                 # avoid data corruption
    nbPtShift = dic['CPMG']['nbPtShift']
    firstDec = dic['CPMG']['firstDec']
    dw2 = dic['CPMG']['DW2']
    td2 = dic['CPMG']['TD2']
    nbEcho = dic['CPMG']['nbEcho']
    halfEcho = dic['CPMG']['halfEcho']
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    # Correct dead time
    if nbPtShift < 0:                                       # left shift
        ndata = ng.proc_base.ls(ndata, nbPtShift)
        dic['CPMG']['nbPtShift'] = 0                        # update dictionary
    elif nbPtShift > 0:                                     # right shift
        # Backward linear prediction increases errors
        # ndata = ng.proc_lp.lp(ndata, nbPtShift, mode='b', append='before')
        ndata = ng.proc_base.rs(ndata, nbPtShift)
    # Correct echo delay
    rest = 0.0
    sumShift = 0
    ndata2 = np.zeros(td2, dtype='complex128')
    if firstDec == True:                                    # first echo
        ndata2[:nbPtHalfEcho] = ndata[:nbPtHalfEcho]
    else:
        ndata2[:2*nbPtHalfEcho] = ndata[:2*nbPtHalfEcho]
    for i in range(int(not(firstDec)), nbEcho):             # following echoes
        sliceNdata2 = slice(
            (int(firstDec) + 2*i) * nbPtHalfEcho - sumShift,
            (int(firstDec) + 2*(i+1)) * nbPtHalfEcho - sumShift)
        rest += 2 * (halfEcho / dw2 - nbPtHalfEcho)
        if rest >= 1:                                       # discard 1 point
            sliceNdata = slice(
                (int(firstDec) + 2*i) * nbPtHalfEcho + 1,
                (int(firstDec) + 2*(i+1)) * nbPtHalfEcho + 1)
            rest -= 1.0
            sumShift += 1
        else:                                               # copy data
            sliceNdata = slice(
                (int(firstDec) + 2*i) * nbPtHalfEcho,
                (int(firstDec) + 2*(i+1)) * nbPtHalfEcho)
        ndata2[sliceNdata2] = ndata[sliceNdata]
    if sumShift // (nbPtHalfEcho * 2) >= 1:         # update dictionnary
        dic['CPMG']['nbEcho'] -= sumShift // (nbPtHalfEcho * 2)
        dic['CPMG']['nbEchoApod'] = dic['CPMG']['nbEcho']
        dic['CPMG']['nbHalfEcho'] -= 2 * sumShift // (nbPtHalfEcho * 2)
        dic['CPMG']['nbPtSignal'] = nbPtHalfEcho * dic['CPMG']['nbHalfEcho']
    dic['CPMG']['halfEcho'] = nbPtHalfEcho * dw2
    dic['CPMG']['fullEcho'] = halfEcho * 2
    return dic, ndata2

#%%----------------------------------------------------------------------------
### Apodisation
###----------------------------------------------------------------------------
def echo_apod(dic, data, method):
    """Apodisation for each half echo"""
    ndata = data[:dic['CPMG']['nbPtSignal']]        # discard useless points
    desc = dic['CPMG']['firstDec']
    apod = np.ones(dic['CPMG']['nbPtSignal']+1)     # one additional point
    if method == 'cos':
        dic['CPMG']['apodEcho'] = 'cos'
    elif method == 'exp':
        T2 = dic['CPMG']['halfEcho']                        # in seconds
        lb = (1 / (np.pi * T2))                             # in Hz
        dic['CPMG']['apodEcho'] = 'LB = {:s} Hz'.format(str(int(lb)))
        lb *= (dic['CPMG']['DW2'])                          # in points
    else:
        raise NotImplementedError('Unkown method for echo apodisation')
    for i in range(dic['CPMG']['nbHalfEcho']):
        ptHalfEcho = slice(
            i*dic['CPMG']['nbPtHalfEcho'],
            (i+1)*dic['CPMG']['nbPtHalfEcho']+1)
        if method == 'cos':
            apod[ptHalfEcho] = ng.proc_base.sp(
                apod[ptHalfEcho], off=0.5, end=1, pow=1.0, rev=(not desc))
        elif method == 'exp':
            apod[ptHalfEcho] = ng.proc_base.em(
                apod[ptHalfEcho], lb, rev=(not desc))
        apod[(i+1)*dic['CPMG']['nbPtHalfEcho']] = 1
        desc = not(desc)
    apod = apod[:-1]                # discard last point used for calculation
    ndata = ndata * apod
    return dic, ndata

def findMaxEcho(dic, data):
    """Find maximum of each echo"""
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
    """Relaxation function"""
    M0, T2, noise = p
    fit = M0 * np.exp(-np.array(x) / T2) + noise
    return fit

def findT2(dic, data):
    """Fitting function and residuals calculation"""
    def residuals(p, x, y):
        err = y - fit_func(p, x)
        return err
    dureeSignal = 1e3*dic['CPMG']['dureeSignal']
    firstDec = dic['CPMG']['firstDec']
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    firstMin = int(firstDec) * nbPtHalfEcho
    timeEcho, maxEcho = findMaxEcho(dic, data)
    # fit the trajectory using leastsq
    p0 = [1.0, timeEcho[timeEcho.size // 2], maxEcho[-1]]   # initial guess
    fitEcho = sp.optimize.leastsq(residuals, p0, args=(timeEcho, maxEcho))
    # Find end of signal
    T2 = fitEcho[0][1]                                      # in milliseconds
    if 3*T2 > dureeSignal:
        nbPtSignal = dic['CPMG']['nbPtSignal']
        T2 = min(T2, dureeSignal)
    else:
        nbPtSignal = np.argwhere(dic['CPMG']['ms_scale'] > 3*T2)[0][0]
    if (nbPtSignal - firstMin) % (2*nbPtHalfEcho) != 0:
        nbPtSignal = (
            ((nbPtSignal - firstMin) // (2*nbPtHalfEcho) + 1)
            *2*nbPtHalfEcho + firstMin)           # round to next full echo
    dic['CPMG']['timeEcho'] = timeEcho
    dic['CPMG']['maxEcho'] = maxEcho
    dic['CPMG']['fitEcho'] = fitEcho
    dic['CPMG']['T2'] = T2
    dic['CPMG']['nbPtSignal'] = nbPtSignal
    return dic

def global_apod(dic, data):
    """Global apodisation"""
    ndata = data[:]                                 # avoid data corruption
    nbPtSignal = dic['CPMG']['nbPtSignal']
    firstDec = dic['CPMG']['firstDec']
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    firstMin = int(firstDec) * nbPtHalfEcho
    lb_Hz = (1e3 / (np.pi * dic['CPMG']['T2']))             # in Hz
    lb = lb_Hz * (dic['CPMG']['DW2'])                       # in points
    apod = np.ones(nbPtSignal)
    apod = ng.proc_base.em(apod, lb)
    ndata = ndata[:nbPtSignal] * apod                # discard useless points
    nbEchoApod = int((nbPtSignal - firstMin) // (2*nbPtHalfEcho))
    dic['CPMG']['apodFull'] = 'LB = {:s} Hz'.format(str(int(lb_Hz)))
    dic['CPMG']['nbEchoApod'] = nbEchoApod
    return dic, ndata

#%%----------------------------------------------------------------------------
### Final processing
###----------------------------------------------------------------------------
def echo_sep(dic, data):
    """Separation of echoes into a matrix"""
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
    return ndata

def mat_sum(dic, data):
    """Folding and sum of echoes from matrix to 1D"""
    firstDec = dic['CPMG']['firstDec']
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    row, col = data.shape
    # Echoes folding
    ndata = np.zeros(
        (row, nbPtHalfEcho), dtype='complex128')
    for i in range(row):
        ndata[i, :] = data[i, nbPtHalfEcho:]
        ndata[i, 1:] += (data[i, nbPtHalfEcho-1:0:-1].real
            -1j*data[i, nbPtHalfEcho-1:0:-1].imag)
    ndata[0,:] *= int(firstDec) + 1
    # Echoes sum
    ndata2 = np.zeros(nbPtHalfEcho, dtype='complex128')
    for i in range(row):
        ndata2[:] += ndata[i, :]
    return ndata2

def fid_sum(dic, data):
    """Decrease number of echoes"""
    firstDec = dic['CPMG']['firstDec']
    nbEcho = dic['CPMG']['nbEchoApod']
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    if firstDec == True:                        # keep intact first decrease
        firstSum = 1
    else:
        firstSum = 2
    maxNbEcho2 = 10                                 # maximum number of echoes
    nbEchoSum = round((nbEcho - firstSum//2) / maxNbEcho2)  # summed echoes
    nbEcho2 = (nbEcho - firstSum//2) // nbEchoSum           # new nb of echoes
    rest = (nbEcho - firstSum//2) % nbEchoSum               # last echoes
    ndata = np.zeros(
        (firstSum + 2*nbEcho2) * nbPtHalfEcho, dtype='complex128')
    # First echo
    sliceData = slice(0, firstSum*nbPtHalfEcho)
    sliceNdata = sliceData
    ndata[sliceNdata] = data[sliceData]                     # new data
    # Following echoes
    for i in range(nbEcho2-1):
        sumEcho = np.zeros(2*nbPtHalfEcho, dtype='complex128')
        for k in range(i*nbEchoSum, (i+1)*nbEchoSum):
            sliceData = slice(
                (firstSum + 2*k) * nbPtHalfEcho,
                (firstSum + 2*k + 2) * nbPtHalfEcho)
            sumEcho += data[sliceData]                      # old data
        sliceNdata = slice(
            (firstSum + 2*i) * nbPtHalfEcho,
            (firstSum + 2*i + 2) * nbPtHalfEcho)
        ndata[sliceNdata] = sumEcho / nbEchoSum             # new data
    # Last echo
    for i in range(nbEcho2-1, nbEcho2):
        sumEcho = np.zeros(2*nbPtHalfEcho, dtype='complex128')
        for k in range(i*nbEchoSum, nbEcho-firstSum//2):
            sliceData = slice(
                (firstSum + 2*k) * nbPtHalfEcho,
                (firstSum + 2*k + 2) * nbPtHalfEcho)
            sumEcho += data[sliceData]                      # old data
        sliceNdata = slice(
            (firstSum + 2*i) * nbPtHalfEcho,
            (firstSum + 2*i + 2) * nbPtHalfEcho)
        ndata[sliceNdata] = sumEcho / (nbEchoSum + rest)    # new data
    # Update dictionary
    dic['CPMG']['nbEchoDen'] = nbEcho2 + firstSum//2
    return dic, ndata

def trunc(dic, data):
    """Truncation of first half echo"""
    ndata = data[:]                                 # avoid data corruption
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    firstDec = dic['CPMG']['firstDec']
    firstTop = int(not firstDec)*nbPtHalfEcho
    ndata = ndata[firstTop: firstTop+nbPtHalfEcho]
    return ndata

#%%----------------------------------------------------------------------------
### Plotting
###----------------------------------------------------------------------------
def FID_figure(dic, A, B, C, D, k_thres):
    """Time domain figure"""
    acquiT = dic['CPMG']['AQ']
    halfEcho = dic['CPMG']['halfEcho']
    ms_scale = dic['CPMG']['ms_scale']
    nbEchoApod = dic['CPMG']['nbEchoApod']
    nbEchoDen = dic['CPMG']['nbEchoDen']
    vert_scale_FID = max(abs(A)) * 1.1
    fig1 = plt.figure()
    fig1.suptitle('CPMG NMR signal processing - FID', fontsize=16)
    
    ax1_1 = fig1.add_subplot(411)
    ax1_1.set_title('Noisy FID, {:d} echoes'.format(dic['CPMG']['nbEcho']))
    ax1_1.plot(ms_scale, A.real)
    ax1_1.set_xlim([-halfEcho * 1e3, (acquiT + halfEcho)*1e3])
    ax1_1.set_ylim([-vert_scale_FID, vert_scale_FID])
    
    ax1_2 = fig1.add_subplot(412)
    ax1_2.set_title(
        'Apodised FID, {:s} and {:s}, {:d} echoes'
        .format(dic['CPMG']['apodEcho'], dic['CPMG']['apodFull'], nbEchoApod))
    ax1_2.plot(ms_scale[:B.size], B.real)
    ax1_2.set_xlim([-halfEcho * 1e3, (acquiT + halfEcho)*1e3])
    ax1_2.set_ylim([-vert_scale_FID, vert_scale_FID])
    
    ax1_3 = fig1.add_subplot(413)
    ax1_3.set_title(
        'Apodised and denoised FID, k = {:d}, {:d} echoes'
        .format(k_thres, nbEchoDen))
    ax1_3.plot(ms_scale[:C.size], C.real)
    ax1_3.set_xlim([-halfEcho * 1e3, (acquiT + halfEcho)*1e3])
    ax1_3.set_ylim([-vert_scale_FID, vert_scale_FID])
    
    ax1_4 = fig1.add_subplot(414)
    ax1_4.set_title(
        'Apodised, denoised and truncated FID, k = {:d}, {:d} echoes'
        .format(k_thres, nbEchoDen))
    ax1_4.set_xlabel('Time (ms)')
    ax1_4.plot(ms_scale[:D.size], D.real)
    ax1_4.set_xlim([-halfEcho * 1e3, (acquiT + halfEcho)*1e3])
    ax1_4.set_ylim([-vert_scale_FID, vert_scale_FID])
    
    fig1.tight_layout(rect=(0,0,1,0.95))            # Avoid superpositions

def SPC_figure(dic, A, B, C, D, k_thres):
    """Frequency domain figure"""
    # Zero-filling, and Fourier transform
    nbEchoApod = dic['CPMG']['nbEchoApod']
    nbEchoDen = dic['CPMG']['nbEchoDen']
    nbPtShift = dic['CPMG']['nbPtShift']
    Hz_scale = dic['CPMG']['Hz_scale']
    SPC_A = postproc.postproc_data(dic, A, False)
    SPC_B = postproc.postproc_data(dic, B, False)
    SPC_C = postproc.postproc_data(dic, C, False)
    SPC_D = postproc.postproc_data(dic, D, False)
#    SPC_A = postproc.postproc_data(dic, A[nbPtShift:], False)
#    SPC_B = postproc.postproc_data(dic, B[nbPtShift:], False)
#    SPC_C = postproc.postproc_data(dic, C[nbPtShift:], False)
#    SPC_D = postproc.postproc_data(dic, D[nbPtShift:], False)
    SPC_A = ng.proc_base.ps(SPC_A, p0=45.0*nbPtShift, p1=-360*nbPtShift)
#    SPC_B = ng.proc_base.ps(SPC_B, p0=45.0*nbPtShift, p1=-360*nbPtShift)
#    SPC_C = ng.proc_base.ps(SPC_C, p0=45.0*nbPtShift, p1=-360*nbPtShift)
#    SPC_D = ng.proc_base.ps(SPC_D, p0=45.0*nbPtShift, p1=-360*nbPtShift)
    fig2 = plt.figure()
    fig2.suptitle('CPMG NMR signal processing - SPC', fontsize=16)
    
    ax2_1 = fig2.add_subplot(411)
    ax2_1.set_title('Noisy SPC, {:d} echoes'.format(dic['CPMG']['nbEcho']))
    ax2_1.plot(Hz_scale, SPC_A.real)
    ax2_1.invert_xaxis()
    
    ax2_2 = fig2.add_subplot(412)
    ax2_2.set_title(
        'Apodised SPC, {:s} and {:s}, {:d} echoes'
        .format(dic['CPMG']['apodEcho'], dic['CPMG']['apodFull'], nbEchoApod))
    ax2_2.plot(Hz_scale, SPC_B.real)
    ax2_2.invert_xaxis()
    
    ax2_3 = fig2.add_subplot(413)
    ax2_3.set_title(
        'Apodised and denoised SPC, k = {:d}, {:d} echoes'
        .format(k_thres, nbEchoDen))
    ax2_3.plot(Hz_scale, SPC_C.real)
    ax2_3.invert_xaxis()
    
    ax2_4 = fig2.add_subplot(414)
    ax2_4.set_title(
        'Apodised, denoised and truncated SPC, k = {:d}, {:d} echoes'
        .format(k_thres, nbEchoDen))
    ax2_4.set_xlabel('Frequency (Hz)')
    ax2_4.plot(Hz_scale, SPC_D.real)
    ax2_4.invert_xaxis()
    
    fig2.tight_layout(rect=(0,0,1,0.95))            # Avoid superpositions

def comp_figure(dic, B, D, F, G, k_thres):
    """Comparison figure"""
    # Zero-filling, and Fourier transform
    nbEchoApod = dic['CPMG']['nbEchoApod']
    nbEchoDen = dic['CPMG']['nbEchoDen']
    nbPtShift = dic['CPMG']['nbPtShift']
    Hz_scale = dic['CPMG']['Hz_scale']
    SPC_B = postproc.postproc_data(dic, B, False)
    SPC_D = postproc.postproc_data(dic, D, False)
    SPC_F = postproc.postproc_data(dic, F, False)
    SPC_G = postproc.postproc_data(dic, trunc(dic, G), False)
#    SPC_B = postproc.postproc_data(dic, B[nbPtShift:], False)
#    SPC_D = postproc.postproc_data(dic, D[nbPtShift:], False)
#    SPC_F = postproc.postproc_data(dic, F[nbPtShift:], False)
#    SPC_G = postproc.postproc_data(dic, trunc(dic, G), False)
#    SPC_B = ng.proc_base.ps(SPC_B, p0=45.0*nbPtShift, p1=-360*nbPtShift)
#    SPC_D = ng.proc_base.ps(SPC_D, p0=45.0*nbPtShift, p1=-360*nbPtShift)
#    SPC_F = ng.proc_base.ps(SPC_F, p0=45.0*nbPtShift, p1=-360*nbPtShift)
    fig3 = plt.figure()
    fig3.suptitle('CPMG NMR signal processing - comparison', fontsize=16)
    
    ax3_1 = fig3.add_subplot(411)
    ax3_1.set_title(
        'Spikelets method, {:s} and {:s}, {:d} echoes'
        .format(dic['CPMG']['apodEcho'], dic['CPMG']['apodFull'], nbEchoApod))
    ax3_1.plot(Hz_scale, SPC_B.real / max(abs(SPC_B)))
    ax3_1.invert_xaxis()

    ax3_2 = fig3.add_subplot(412)
    ax3_2.set_title(
        'Weighted sum method, {:s} and {:s}, {:d} echoes'
        .format(dic['CPMG']['apodEcho'], dic['CPMG']['apodFull'], nbEchoApod))
    ax3_2.plot(Hz_scale, SPC_F.real / max(abs(SPC_F)))
    ax3_2.invert_xaxis()

    ax3_3 = fig3.add_subplot(413)
    ax3_3.set_title(
        'Denoising method, {:s} and {:s}, {:d} echoes, k = {:d}'
        .format(dic['CPMG']['apodEcho'], dic['CPMG']['apodFull'], nbEchoDen,
        k_thres))
    ax3_3.plot(Hz_scale, SPC_D.real / max(abs(SPC_D)))
    ax3_3.invert_xaxis()

    ax3_4 = fig3.add_subplot(414)
    ax3_4.set_title('Reference SPC')
    ax3_4.plot(Hz_scale, SPC_G.real / max(abs(SPC_G)))
    ax3_4.invert_xaxis()
    ax3_4.set_xlabel('Frequency (Hz)')

    fig3.tight_layout(rect=(0,0,1,0.95))            # Avoid superpositions

def echoes_figure(dic, E):
    """Details figure"""
    fitEcho = dic['CPMG']['fitEcho']
    maxEcho = dic['CPMG']['maxEcho']
    timeEcho = dic['CPMG']['timeEcho']
    ms_scale = dic['CPMG']['ms_scale']
    nbEchoApod = dic['CPMG']['nbEchoApod']
    row, col = E.shape
    fig4 = plt.figure()
    fig4.suptitle('CPMG NMR signal processing - Echoes', fontsize=16)

    ax4_1 = fig4.add_subplot(211)
    ax4_1.set_title('Separated FID, {:d} echoes'.format(nbEchoApod))
    ax4_1.set_xlabel('Time (ms)')
    ax4_1.set_ylabel('Intensity')
    for i in range(0, row, max(2, int(row/10))):
        ax4_1.plot(ms_scale[:E[i,:].size], E[i, :].real)
    ax4_1.axvline(
        x=ms_scale[int(E[0,:].size/2)], color='k', linestyle=':', linewidth=2)

    ax4_2 = fig4.add_subplot(212)
    ax4_2.set_title(
        'Intensity of echoes, T2 = {:8.2f} ms'.format(fitEcho[0][1]))
    timeFit = np.linspace(ms_scale[0], ms_scale[-1], 100)
    valFit = fit_func(fitEcho[0], timeFit)
    ax4_2.scatter(timeEcho, maxEcho)
    ax4_2.plot(timeFit, valFit)
    ax4_2.set_xlabel('Time (ms)')
    ax4_2.set_ylabel('Intensity')
    
    fig4.tight_layout(rect=(0,0,1,0.95))

def plot_function(dic, A, B, C, D, E, F, G, k_thres):
    """Plotting"""
    plt.ion()                                   # to avoid stop when plotting
#    echoes_figure(dic, E)
    FID_figure(dic, A, B, C, D, k_thres)
    SPC_figure(dic, A, B, C, D, k_thres)
#    comp_figure(dic, B, D, F, G, k_thres)
    plt.ioff()                                  # to avoid figure closing
    plt.show()                                  # to allow zooming

#%%----------------------------------------------------------------------------
### When this file is executed directly
###----------------------------------------------------------------------------
def main():
    """Min CPMG processing function"""
    dic, FIDref, FIDraw = data_import()                     # importation
    dic, FIDshift = shift_FID(dic, FIDraw)                  # dead time
    dic, FIDapod = echo_apod(dic, FIDshift, method='exp')   # echoes apod
    dic = findT2(dic, FIDapod)                              # relaxation
    dic, FIDapod2 = global_apod(dic, FIDapod)               # global apod
    FIDmat = echo_sep(dic, FIDapod2)                        # echoes separation
    FIDsum2 = mat_sum(dic, FIDmat)                          # echoes sum

    dic, FIDsum = fid_sum(dic, FIDapod2)                    # decrease nbEchoes
    FIDden, k_thres = denoise_nmr.denoise(
        FIDsum, k_thres='auto', max_err=5)                  # denoising
    FIDtrunc = trunc(dic, FIDden)                           # truncation
    plot_function(
        dic, FIDraw, FIDapod2, FIDden, FIDtrunc, 
        FIDmat, FIDsum2, FIDref, k_thres)                   # plotting
#    data_export(dic, FIDtrunc, FIDsum, FIDapod2)            # saving

if __name__ == "__main__":
    main()
    