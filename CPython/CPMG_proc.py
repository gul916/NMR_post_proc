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
        FIDref = None
    else:
        raise NotImplementedError(
            'Arguments should be data, fullEcho, nbEcho, firstDec, nbPtShift')
    return dic, FIDref, FIDraw

def data_export(dic, E, F, H):
    """Export data"""
    if len(sys.argv) > 1:                                   # save spectrum
        firstDec = dic['CPMG']['firstDec']
        nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
        firstMax = int(not(firstDec)) * nbPtHalfEcho
        nextMin = (int(not(firstDec)) + 1) * nbPtHalfEcho
        # First echo shift
        SPC_Eref = E[firstMax:nextMin]                  # reference spectrum
        SPC_E = E[firstMax:]                            # spikelets method
        SPC_F = F[:]                                    # sum method
        SPC_H = H[:]                                    # denoising method
        # Zero-filling, and Fourier transform
        SPC_Eref = postproc.postproc_data(dic, SPC_Eref, False)
        SPC_E = postproc.postproc_data(dic, SPC_E, False)
        SPC_F = postproc.postproc_data(dic, SPC_F, False)
        SPC_H = postproc.postproc_data(dic, SPC_H, False)
        # Writing title
        data_dir = sys.argv[1][:-1]
        with open(data_dir+'1/title', 'w+') as file:
            file.write('Reference spectrum')
        with open(data_dir+'2/title', 'w+') as file:
            file.write('Spikelets method')
        with open(data_dir+'3/title', 'w+') as file:
            file.write('Weighted sum method')
        with open(data_dir+'4/title', 'w+') as file:
            file.write('Denoising method')
        # Writing data
        postproc.export_data(dic, SPC_Eref, data_dir + '1')
        postproc.export_data(dic, SPC_E, data_dir + '2')
        postproc.export_data(dic, SPC_F, data_dir + '3')
        postproc.export_data(dic, SPC_H, data_dir + '4')

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
    dic['CPMG']['apodEchoPoints'] = apod
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
        nbPtApod = dic['CPMG']['nbPtSignal']
        T2 = min(T2, dureeSignal)
    else:
        nbPtApod = np.argwhere(dic['CPMG']['ms_scale'] > 3*T2)[0][0]
    if (nbPtApod - firstMin) % (2*nbPtHalfEcho) != 0:
        nbPtApod = (
            ((nbPtApod - firstMin) // (2*nbPtHalfEcho) + 1)
            *2*nbPtHalfEcho + firstMin)           # round to next full echo
    dic['CPMG']['timeEcho'] = timeEcho
    dic['CPMG']['maxEcho'] = maxEcho
    dic['CPMG']['fitEcho'] = fitEcho
    dic['CPMG']['T2'] = T2
    dic['CPMG']['nbPtApod'] = nbPtApod
    return dic

def global_apod(dic, data):
    """Global apodisation"""
    ndata = data[:]                                 # avoid data corruption
    nbPtApod = dic['CPMG']['nbPtApod']
    firstDec = dic['CPMG']['firstDec']
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    firstMin = int(firstDec) * nbPtHalfEcho
    lb_Hz = (1e3 / (np.pi * dic['CPMG']['T2']))             # in Hz
    lb = lb_Hz * (dic['CPMG']['DW2'])                       # in points
    apod = np.ones(nbPtApod)
    apod = ng.proc_base.em(apod, lb)
    ndata = ndata[:nbPtApod] * apod                # discard useless points
    nbEchoApod = int((nbPtApod - firstMin) // (2*nbPtHalfEcho))
    dic['CPMG']['apodFull'] = 'LB = {:s} Hz'.format(str(int(lb_Hz)))
    dic['CPMG']['apodFullPoints'] = apod
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

def fid_sum(dic, data, firstIntact=False):
    """Decrease number of echoes"""
    # TODO: could probably be simplified
    firstDec = dic['CPMG']['firstDec']
    nbEcho = dic['CPMG']['nbEcho']
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    firstEcho = int(firstDec)                               # first half echo
    firstSum = int(not(firstDec)) + 1                       # shift of echoes
    maxNbEcho2 = 25                             # maximum number of echoes
    if firstIntact == True:                     # keep intact first decrease
        nbEchoSum = max(
                1, round((nbEcho - firstSum//2) / maxNbEcho2))  # summed echoes
        nbEcho2 = (nbEcho - firstSum//2) // nbEchoSum       # new nb of echoes
        rest = (nbEcho - firstSum//2) % nbEchoSum           # last echoes
        ndata = np.zeros(
            (firstSum + 2*nbEcho2) * nbPtHalfEcho, dtype='complex128')
        # First echo
        sliceData = slice(0, firstSum*nbPtHalfEcho)
        sliceNdata = sliceData
        ndata[sliceNdata] = data[sliceData]                 # new data
        # Following echoes
        for i in range(nbEcho2-1):
            sumEcho = np.zeros(2*nbPtHalfEcho, dtype='complex128')
            for k in range(i*nbEchoSum, (i+1)*nbEchoSum):
                sliceData = slice(
                    (firstSum + 2*k) * nbPtHalfEcho,
                    (firstSum + 2*k + 2) * nbPtHalfEcho)
                sumEcho += data[sliceData]                  # old data
            sliceNdata = slice(
                (firstSum + 2*i) * nbPtHalfEcho,
                (firstSum + 2*i + 2) * nbPtHalfEcho)
#            ndata[sliceNdata] = sumEcho / nbEchoSum # Increases discontinuity
            ndata[sliceNdata] = sumEcho                     # new data
        # Last echo
        for i in range(nbEcho2-1, nbEcho2):
            sumEcho = np.zeros(2*nbPtHalfEcho, dtype='complex128')
            for k in range(i*nbEchoSum, nbEcho-firstSum//2):
                sliceData = slice(
                    (firstSum + 2*k) * nbPtHalfEcho,
                    (firstSum + 2*k + 2) * nbPtHalfEcho)
                sumEcho += data[sliceData]                  # old data
            sliceNdata = slice(
                (firstSum + 2*i) * nbPtHalfEcho,
                (firstSum + 2*i + 2) * nbPtHalfEcho)
#            ndata[sliceNdata] = sumEcho / nbEchoSum # Increases discontinuity
            ndata[sliceNdata] = sumEcho                     # new data
            # Update dictionary
        dic['CPMG']['nbEchoDen'] = nbEcho2 + firstSum//2
    else:                                       # Average first decrease
        nbEchoSum = max(
                1, round((nbEcho + firstEcho) / maxNbEcho2))# summed echoes
        nbEcho2 = (nbEcho + firstEcho) // nbEchoSum         # new nb of echoes
        rest = (nbEcho + firstEcho) % nbEchoSum             # last echoes
        ndata = np.zeros(2 * nbEcho2 * nbPtHalfEcho, dtype='complex128')
        # First echo
        for i in range(0, 1):
            sliceData = slice(0, firstSum*nbPtHalfEcho)
            sliceNdata = slice(firstEcho*nbPtHalfEcho, 2*nbPtHalfEcho)
            sumEcho = np.zeros(2*nbPtHalfEcho, dtype='complex128')
            sumEcho[sliceNdata] += data[sliceData]          # old data
            for k in range(i*nbEchoSum, (i+1)*nbEchoSum-1):
                sliceData = slice(
                    (firstSum + 2*k) * nbPtHalfEcho,
                    (firstSum + 2*k + 2) * nbPtHalfEcho)
                sumEcho += data[sliceData]                  # old data
            sliceNdata = slice(2*i * nbPtHalfEcho, (2*i + 2) * nbPtHalfEcho)
            ndata[sliceNdata] = sumEcho / nbEchoSum         # new data
        # Following echoes
        for i in range(1, nbEcho2-1):
            sumEcho = np.zeros(2*nbPtHalfEcho, dtype='complex128')
            for k in range(i*nbEchoSum-1, (i+1)*nbEchoSum-1):
                sliceData = slice(
                    (firstSum + 2*k) * nbPtHalfEcho,
                    (firstSum + 2*k + 2) * nbPtHalfEcho)
                sumEcho += data[sliceData]                  # old data
            sliceNdata = slice(2*i * nbPtHalfEcho, (2*i + 2) * nbPtHalfEcho)
            ndata[sliceNdata] = sumEcho / nbEchoSum         # new data
        # Last echo
        for i in range(nbEcho2-1, nbEcho2):
            sumEcho = np.zeros(2*nbPtHalfEcho, dtype='complex128')
            for k in range(i*nbEchoSum-1, nbEcho-firstSum//2):
                sliceData = slice(
                    (firstSum + 2*k) * nbPtHalfEcho,
                    (firstSum + 2*k + 2) * nbPtHalfEcho)
                sumEcho += data[sliceData]                  # old data
            sliceNdata = slice(2*i * nbPtHalfEcho, (2*i + 2) * nbPtHalfEcho)
            ndata[sliceNdata] = sumEcho / (nbEchoSum + rest)# new data
        # Update dictionary
        dic['CPMG']['nbEchoDen'] = nbEcho2
        dic['CPMG']['firstDecDen'] = False
    return dic, ndata

def trunc(dic, data):
    """Truncation of first half echo"""
    ndata = data[:]                                 # avoid data corruption
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    if 'firstDecDen' in dic['CPMG']:
        firstDec = dic['CPMG']['firstDecDen']
    else:
        firstDec = dic['CPMG']['firstDec']
    firstTop = int(not firstDec)*nbPtHalfEcho
    ndata = ndata[firstTop: firstTop+nbPtHalfEcho]
    return ndata

#%%----------------------------------------------------------------------------
### Plotting
###----------------------------------------------------------------------------
def echoes_figure(dic, D):
    """Echoes figure"""
    fitEcho = dic['CPMG']['fitEcho']
    maxEcho = dic['CPMG']['maxEcho']
    timeEcho = dic['CPMG']['timeEcho']
    ms_scale = dic['CPMG']['ms_scale']
    nbEcho = dic['CPMG']['nbEcho']
    Dmat = echo_sep(dic, D)
    row, col = Dmat.shape
    fig = plt.figure()
    fig.suptitle('CPMG NMR signal processing - echoes', fontsize=16)

    ax1 = fig.add_subplot(211)
    ax1.set_title('Separated FID, {:d} echoes'.format(nbEcho))
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Intensity')
    for i in range(0, row, max(2, int(row/10))):
        ax1.plot(ms_scale[:Dmat[i,:].size], Dmat[i, :].real)
    ax1.axvline(
        x=ms_scale[int(Dmat[0,:].size/2)],
        color='k', linestyle=':', linewidth=2)

    ax2 = fig.add_subplot(212)
    ax2.set_title(
        'Intensity of echoes, T2 = {:8.2f} ms'.format(fitEcho[0][1]))
    timeFit = np.linspace(ms_scale[0], ms_scale[-1], 100)
    valFit = fit_func(fitEcho[0], timeFit)
    ax2.scatter(timeEcho, maxEcho)
    ax2.plot(timeFit, valFit)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Intensity')
    
    fig.tight_layout(rect=(0, 0, 1, 0.95))      # Avoid superpositions

def apod_figure(dic, B, D, E):
    """Time domain figure"""
    acquiT = dic['CPMG']['AQ']
    apodEcho = dic['CPMG']['apodEcho']
    apodEchoPoints = dic['CPMG']['apodEchoPoints']
    apodFull = dic['CPMG']['apodFull']
    apodFullPoints = dic['CPMG']['apodFullPoints']
    halfEcho = dic['CPMG']['halfEcho']
    ms_scale = dic['CPMG']['ms_scale']
    nbEcho = dic['CPMG']['nbEcho']
    nbEchoApod = dic['CPMG']['nbEchoApod']
    # Normalisation
    B /= max(B.real)
    D /= max(D.real)
    E /= max(E.real)
    # Figure
    fig = plt.figure()
    fig.suptitle('CPMG NMR signal processing - apodisation', fontsize=16)
    
    ax1 = fig.add_subplot(311)
    ax1.set_title('Noisy FID, {:d} echoes'.format(nbEcho))
    ax1.plot(ms_scale, B.real)
    ax1.set_xlim([-halfEcho * 1e3, (acquiT + halfEcho)*1e3])
    
    ax2 = fig.add_subplot(312)
    ax2.set_title(
        'Apodised FID, {:s}, {:d} echoes'.format(apodEcho, nbEcho))
    ax2.plot(ms_scale[:D.size], D.real)
    ax2.plot(ms_scale[:apodEchoPoints.size], apodEchoPoints.real)
    ax2.set_xlim([-halfEcho * 1e3, (acquiT + halfEcho)*1e3])
    
    ax3 = fig.add_subplot(313)
    ax3.set_title(
        'Apodised FID, {:s} and {:s}, {:d} echoes'
        .format(apodEcho, apodFull, nbEchoApod))
    ax3.plot(ms_scale[:E.size], E.real)
    ax3.plot(ms_scale[:apodFullPoints.size], apodFullPoints.real)
    ax3.set_xlim([-halfEcho * 1e3, (acquiT + halfEcho)*1e3])
    ax3.set_xlabel('Time (ms)')

    fig.tight_layout(rect=(0, 0, 1, 0.95))      # Avoid superpositions

def sum_figure(dic, E, F, A, C):
    """Weighted sum figure"""
    firstDec = dic['CPMG']['firstDec']
    apodEcho = dic['CPMG']['apodEcho']
    apodFull = dic['CPMG']['apodFull']
    nbEchoApod = dic['CPMG']['nbEchoApod']
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    Hz_scale = dic['CPMG']['Hz_scale']
    firstMax = int(not(firstDec)) * nbPtHalfEcho
    nextMin = (int(not(firstDec)) + 1) * nbPtHalfEcho
    # Reference spectrum
    if A is None:
        A = C
    # First echo shift
    SPC_E = E[firstMax:]
    SPC_F = F[:]
    SPC_A = A[firstMax:nextMin]
    # Zero-filling, and Fourier transform
    SPC_E = postproc.postproc_data(dic, SPC_E, False)[::-1]
    SPC_F = postproc.postproc_data(dic, SPC_F, False)[::-1]
    SPC_A = postproc.postproc_data(dic, SPC_A, False)[::-1]
    # Normalisation
    SPC_E /= max(SPC_E.real)
    SPC_F /= max(SPC_F.real)
    SPC_A /= max(SPC_A.real)

    fig = plt.figure()
    fig.suptitle('CPMG NMR signal processing - standard methods', fontsize=16)
    
    ax1 = fig.add_subplot(311)
    ax1.set_title(
        'Spikelets method, {:s} and {:s}, {:d} echoes'
        .format(apodEcho, apodFull, nbEchoApod))
    ax1.plot(Hz_scale, SPC_E.real)
    ax1.invert_xaxis()

    ax2 = fig.add_subplot(312)
    ax2.set_title(
        'Weighted sum method, {:s} and {:s}, {:d} echoes'
        .format(apodEcho, apodFull, nbEchoApod))
    ax2.plot(Hz_scale, SPC_F.real)
    ax2.invert_xaxis()

    ax3 = fig.add_subplot(313)
    ax3.set_title('Reference SPC')
    ax3.plot(Hz_scale, SPC_A.real)
    ax3.invert_xaxis()
    ax3.set_xlabel('Frequency (Hz)')

    fig.tight_layout(rect=(0, 0, 1, 0.95))      # Avoid superpositions

def den_figure(dic, G, H, A, C, k_thres):
    """Denoising figure"""
    firstDec = dic['CPMG']['firstDec']
    nbEchoDen = dic['CPMG']['nbEchoDen']
    nbPtHalfEcho = dic['CPMG']['nbPtHalfEcho']
    Hz_scale = dic['CPMG']['Hz_scale']
    firstMax = int(not(firstDec)) * nbPtHalfEcho
    nextMin = (int(not(firstDec)) + 1) * nbPtHalfEcho
    if 'firstDecDen' in dic['CPMG']:            # first echo averaged
        firstMaxDen = nbPtHalfEcho
    else:
        firstMaxDen = firstMax
    # Reference spectrum
    if A is None:
        A = C
    # First echo shift
    SPC_G = G[firstMaxDen:]
    SPC_H = H[:]
    SPC_A = A[firstMax:nextMin]
    # Zero-filling, and Fourier transform
    SPC_G = postproc.postproc_data(dic, SPC_G, False)[::-1]
    SPC_H = postproc.postproc_data(dic, SPC_H, False)[::-1]
    SPC_A = postproc.postproc_data(dic, SPC_A, False)[::-1]
    # Normalisation
    SPC_G /= max(SPC_G.real)
    SPC_H /= max(SPC_H.real)
    SPC_A /= max(SPC_A.real)

    fig = plt.figure()
    fig.suptitle('CPMG NMR signal processing - denoising', fontsize=16)
    
    ax1 = fig.add_subplot(311)
    ax1.set_title('Apodised and denoised SPC, k = {:d}, {:d} echoes'
        .format(k_thres, nbEchoDen))
    ax1.plot(Hz_scale, SPC_G.real)
    ax1.invert_xaxis()
    
    ax2 = fig.add_subplot(312)
    ax2.set_title(
        'Apodised, denoised and truncated SPC, k = {:d}, {:d} echoes'
        .format(k_thres, nbEchoDen))
    ax2.plot(Hz_scale, SPC_H.real)
    ax2.invert_xaxis()
    
    ax3 = fig.add_subplot(313)
    ax3.set_title('Reference SPC')
    ax3.plot(Hz_scale, SPC_A.real)
    ax3.invert_xaxis()
    ax3.set_xlabel('Frequency (Hz)')

    fig.tight_layout(rect=(0, 0, 1, 0.95))      # Avoid superpositions

def plot_function(dic, A, B, C, D, E, F, G, H, k_thres):
    """Plotting"""
    plt.ion()                                   # to avoid stop when plotting
    echoes_figure(dic, D)
    apod_figure(dic, B, D, E)
    sum_figure(dic, E, F, A, C)
    den_figure(dic, G, H, A, C, k_thres)
    plt.ioff()                                  # to avoid figure closing
    plt.show()                                  # to allow zooming

#%%----------------------------------------------------------------------------
### When this file is executed directly
###----------------------------------------------------------------------------
def main():
    """Main CPMG processing function"""
    dic, FIDref, FIDraw = data_import()                     # importation
    dic, FIDshift = shift_FID(dic, FIDraw)                  # dead time
    # Spikelets and weighted sum methods
    dic, FIDapod = echo_apod(dic, FIDshift, method='exp')   # echoes apod
    dic = findT2(dic, FIDapod)                              # relaxation
    dic, FIDapod2 = global_apod(dic, FIDapod)               # global apod
    FIDmat = echo_sep(dic, FIDapod2)                        # echoes separation
    FIDmatSum = mat_sum(dic, FIDmat)                        # echoes sum
    # Denoising method
    dic, FIDsum = fid_sum(dic, FIDapod, firstIntact=True)   # decrease nbEchoes
    FIDden, k_thres = denoise_nmr.denoise(
        FIDsum, k_thres='auto', max_err=5)                  # denoising
    FIDtrunc = trunc(dic, FIDden)                           # truncation
    # Plotting
    plot_function(
        dic, FIDref, FIDraw, FIDshift, FIDapod, FIDapod2, FIDmatSum,
        FIDden, FIDtrunc, k_thres)
    data_export(dic, FIDapod2, FIDmatSum, FIDtrunc)         # saving

if __name__ == "__main__":
    main()