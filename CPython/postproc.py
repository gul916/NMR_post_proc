#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@authors: Pierre-Aymeric GILLES & Guillaume LAURENT
"""

# Python libraries
import nmrglue as ng
import numpy as np
import time
import warnings

def CPMG_pseudo_dic(data, dw2):
    td2 = data.size                                     # complex points TD / 2
    dic = {}
    dic['acqus'] = {}
    dic['acqus']['SW_h'] = 1 / dw2
    dic['procs'] = {}
    dic['procs']['SI'] = ng.proc_base.largest_power_of_2(td2*4)
    return dic

def CPMG_dic(dic, data, fullEcho=1e-3, nbEcho=0, firstDec=True, nbPtShift=0):
    td2 = data.size                                     # complex points TD / 2
    si = dic['procs']['SI']                             # complex points SI
    dw2 = 1 / dic['acqus']['SW_h']                  # complex dwell time DW * 2
    acquiT = dw2 * (td2-1)                              # acquisition time
    halfEcho = fullEcho / 2
    nbHalfEcho = nbEcho * 2                             # number of half echoes
    if firstDec == True:
        nbHalfEcho += 1
    nbPtHalfEcho = int(halfEcho / dw2)                  # points per half echo
    nbPtSignal = nbPtHalfEcho * nbHalfEcho              # points with signal
    dureeSignal = (nbPtSignal -1) * dw2
    nbPtLast = td2 - nbPtSignal                         # unused point
    
    # Verifications
    if fullEcho <= 0:
        raise ValueError('fullEcho must be > 0')
    if round((halfEcho / dw2) % 1, 2) not in (0.0, 1.0):
        warnings.warn('HalfEcho is supposed to be a multiple of 2*dw')
    if (not isinstance(nbEcho, int)) and (not nbEcho >= 0):
        raise ValueError('nbEcho must be of type integer and > 0')
    if not isinstance(firstDec, bool):
        raise ValueError('firstDec must be of type boolean')
    if (dureeSignal > acquiT):
        raise ValueError('Too many echoes during acquisition time')
    if not isinstance(nbPtShift, int):
        raise ValueError('nbPtShift must be of type integer')
    
    # Universal dictionary
    dic['CPMG'] = {}
    dic['CPMG']['TD2'] = td2                            # complex points TD / 2
    dic['CPMG']['DW2'] = dw2                        # complex dwell time DW * 2
    dic['CPMG']['AQ'] = acquiT                          # acquisition time
    dic['CPMG']['firstDec'] = firstDec                  # first decrease
    dic['CPMG']['fullEcho'] = fullEcho                  # full echo delay
    dic['CPMG']['halfEcho'] = halfEcho                  # half echo delay
    dic['CPMG']['nbEcho'] = nbEcho                      # number of echoes
    dic['CPMG']['nbHalfEcho'] = nbHalfEcho              # number of half echoes
    dic['CPMG']['nbPtShift'] = nbPtShift                # points to shift
    dic['CPMG']['nbPtHalfEcho'] = nbPtHalfEcho          # points per half echo
    dic['CPMG']['nbPtSignal'] = nbPtSignal              # points with signal
    dic['CPMG']['nbPtLast'] = nbPtLast                  # unused points
    dic['CPMG']['ms_scale'] = np.linspace(0, acquiT, td2) * 1e3     # time scale
    dic['CPMG']['Hz_scale'] = np.linspace(-1/(2*dw2), 1/(2*dw2), si)
    return dic

#%%----------------------------------------------------------------------------
### IMPORT AND EXPORT
###----------------------------------------------------------------------------
def import_data(data_dir):
    # Import data
    dic, data = ng.bruker.read(data_dir)
    data = ng.bruker.remove_digital_filter(dic, data)   # digital filtering
    return dic, data

def export_data(dic, data, data_dir):
    # Write data
    # Data should have exatly the original processed size (bug #109)
    scaling = 8
    if data.ndim == 1:
        ng.bruker.write_pdata(
            data_dir, dic, data.real*scaling,
            scale_data=True, bin_file='1r', overwrite=True)
        ng.bruker.write_pdata(
            data_dir, dic, data.imag*scaling,
            scale_data=True, bin_file='1i', overwrite=True)
    elif data.ndim == 2:
        datarr = data[::2,:].real
        datari = data[1::2,:].real
        datair = data[::2,:].imag
        dataii = data[1::2,:].imag
        ng.bruker.write_pdata(
            data_dir, dic, datarr*scaling,
            scale_data=True, bin_file='2rr', overwrite=True)
        ng.bruker.write_pdata(
            data_dir, dic, datari*scaling,
            scale_data=True, bin_file='2ri', overwrite=True)
        ng.bruker.write_pdata(
            data_dir, dic, datair*scaling,
            scale_data=True, bin_file='2ir', overwrite=True)
        ng.bruker.write_pdata(
            data_dir, dic, dataii*scaling,
            scale_data=True, bin_file='2ii', overwrite=True)
    else:
        raise NotImplementedError(
            "Data of", data.ndim, "dimensions are not yet supported.")

#%%----------------------------------------------------------------------------
### PROCESSING
###----------------------------------------------------------------------------
def preproc_data(data):
    # Preprocessing
    # Direct dimension
    data = ng.proc_base.sp(data, off=0.5, end=1.0, pow=1.0)     # apodization
    # Indirect dimension processing
    if data.ndim == 2:
        data = ng.proc_base.tp_hyper(data)          # hypercomplex transpose
        data = ng.proc_base.sp(data, off=0.5, end=1.0, pow=1.0) # apodization
        data = ng.proc_base.tp_hyper(data)          # hypercomplex transpose
    elif data.ndim > 2:
        raise NotImplementedError(
            "Data of", data.ndim, "dimensions are not yet supported.")
    return data

def postproc_data(dic, data, autophase=True):
    # Postprocessing
    # Direct dimension
    data = ng.proc_base.zf_size(data, dic['procs']['SI'])   # zero-filling
    if data.ndim == 1:
        data[0] /= 2                                        # normalization
    elif data.ndim == 2:
        data[:, 0] /= 2                                     # normalization
    else:
        raise NotImplementedError(
            "Data of", data.ndim, "dimensions are not yet supported.")
    data = ng.proc_base.fft_norm(data)                      # FFT with norm
    data = ng.proc_base.rev(data)                           # revert spectrum
    if autophase == True:
        print("Autophasing:")
        t_0 = time.time()
        data = ng.proc_autophase.autops(data, 'acme')       # autophasing
        t_1 = time.time()
        print('Autophasing time:             {:8.2f} s\n'.format(t_1 - t_0))
    if data.ndim == 2:
        # Indirect dimension
        data = ng.proc_base.tp_hyper(data)          # hypercomplex transpose
        data = ng.proc_base.zf_size(data, dic['proc2s']['SI'])  # zero-filling
        data[:, 0] /= 2                                     # normalization
        data = ng.proc_base.fft_norm(data)                  # FFT with norm
        if dic['acqu2s']['FnMODE'] == 4:                    # STATES
            pass
        elif dic['acqu2s']['FnMODE'] == 5:                  # STATES-TPPI
            data = np.fft.fftshift(data, axes=-1)           # swap spectrum
        data = ng.proc_base.rev(data)                       # revert spectrum
        data = ng.proc_base.tp_hyper(data)          # hypercomplex transpose
    return data
