#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@authors: Pierre-Aymeric GILLES & Guillaume LAURENT
"""

# Python libraries
import nmrglue as ng
import numpy as np
import warnings

class Signal:
    """
    Class storing a signal's characteristics
    
    data: data set (array)
    dw: dwell time between two aquisition points (float)
    de: dead time at beginning of acquisition (float)
    firstDec: presence of first decreasing (boolean)
    nbEcho: number of echoes (integer)
    fullEcho: full echo delay (float)
    
    TO DO: use NMRglue universal dictionary
    """

    def __init__(
            self, newdata, newdw, newde, 
            newfirstDec, newnbEcho, newfullEcho):
        self.data = newdata
        self.dw = newdw
        self.de = newde
        self.firstDec = newfirstDec
        self.nbEcho = newnbEcho
        self.fullEcho = newfullEcho
        
    @property
    def data(self):
        return self.__data
    @data.setter
    def data(self, val):
        if not isinstance(val, np.ndarray):
            raise ValueError('data must be an array')
        self.__data = val
        self.td2 = self.data.size
        self.__td = self.td2 * 2
    
    @property
    def dw(self):
        return self.__dw
    @dw.setter
    def dw(self, val):
        if val <= 0:
            raise ValueError('dw must be > 0')
        self.__dw = val
        self.__dw2 = 2 * self.__dw
        self.acquiT = (self.td2 - 1) * self.__dw2
        self.ms_scale = np.linspace(0, self.acquiT, self.td2) * 1e3
        self.ppm_scale = np.linspace(
                -1/(2*self.__dw2), 1/(2*self.__dw2), self.td2)
    
    @property
    def de(self):
        return self.__de
    @de.setter
    def de(self, val):
        if val < 0:
            raise ValueError('de must be >= 0')
        self.__de = val
        if round((self.__de / self.__dw2) % 1, 2) not in (0.0, 1.0):
            warnings.warn('de is supposed to be a multiple of 2*dw')
        self.__nbPtDeadTime = int(self.__de / self.__dw2)
            
    
    @property
    def firstDec(self):
        return self.__firstDec
    @firstDec.setter
    def firstDec(self, val):
        if not isinstance(val, bool):
            raise ValueError('firstDec must be of type boolean')
        self.__firstDec = val

    @property
    def nbEcho(self):
        return self.__nbEcho
    @nbEcho.setter
    def nbEcho(self, val):
        if not isinstance(val, int):
            raise ValueError('nbEcho must be of type int')
        if val < 0:
            raise ValueError('nbEcho must be > 0')
        self.__nbEcho = val
        self.nbHalfEcho = self.__nbEcho * 2
        if self.__firstDec == True:
            self.nbHalfEcho += 1

    @property
    def fullEcho(self):
        return self.__fullEcho
    @fullEcho.setter
    def fullEcho(self, val):
        if val <= 0:
            raise ValueError('fullEcho must be > 0')
        self.__fullEcho = val
        self.halfEcho = self.__fullEcho / 2
        if round((self.halfEcho / self.__dw2) % 1, 2) not in (0.0, 1.0):
            warnings.warn('HalfEcho is supposed to be a multiple of 2*dw')
        self.nbPtHalfEcho = int(self.halfEcho / self.__dw2)
        self.__nbPtSignal = self.nbPtHalfEcho * self.nbHalfEcho
        self.__dureeSignal = (self.__nbPtSignal -1) * self.__dw2
        if (self.__dureeSignal > self.acquiT):
            raise ValueError('Too many echoes during acquisition time')
        self.__missingPts = self.td2 - self.__nbPtSignal

#%%----------------------------------------------------------------------------
### PROCESSING
###----------------------------------------------------------------------------
def apod_cos(data_fid):
    """
    Apply cosine apodisation
    Usage:  data_apod = apod(data)
    Input:  data_fid    data to process (array)
    Output: data_apod   data apodized (array)
    """
    # 1D data set or direct dimension of nD dataset
    data_apod = ng.proc_base.sp(data_fid, off=0.5, end=1.0, pow=1.0)
    # 2D data set
    if data_fid.ndim == 2:                              # 2D data set
        data_apod = ng.proc_base.tp_hyper(data_apod)    # hypercomplex transpose
        data_apod = ng.proc_base.sp(data_apod, off=0.5, end=1.0, pow=1.0)
        data_apod = ng.proc_base.tp_hyper(data_apod)    # hypercomplex transpose
    # higher dimensions data set
    elif data_fid.ndim > 2:
        raise NotImplementedError(
            'Data of dimensions higher than 2 are not supported')
    return data_apod

def spc(data_fid):
    """
    FFT of FID with normalization and zero-filling
    Usage:  data_spc = spc(data_fid)
    Input:  data_fid     data to transform (signal class)
    Output: data_spc     transformed data (signal class)
    """
    # 1D data set or direct dimension of nD dataset
    data_spc = data_fid.data                            # import data
    data_spc = ng.proc_base.zf_auto(data_spc)           # zero-filling 2^n
    data_spc = ng.proc_base.zf_double(data_spc, 2)      # zero-filling *4
    data_spc[0] /=  2                                   # normalization
    data_spc = ng.proc_base.fft_norm(data_spc)          # FFT with norm
    # 2D data set
    if data_spc.ndim == 2:
        data_spc = ng.proc_base.tp_hyper(data_spc)      # hypercomplex transpose
        data_spc = ng.proc_base.zf_auto(data_spc)       # zero-filling 2^n
        data_spc = ng.proc_base.zf_double(data_spc, 2)  # zero-filling *4
        data_spc[:, 0] /= 2                             # normalization
        data_spc = ng.proc_base.fft_norm(data_spc)      # FFT with norm
        data_spc = ng.proc_base.tp_hyper(data_spc)      # hypercomplex transpose
    # higher dimensions data set
    elif data_spc.ndim > 2:
        raise NotImplementedError(
            'Data of dimensions higher than 2 are not supported')
    data_spc = Signal(
        data_spc, data_fid.dw, data_fid.de,
        data_fid.firstDec, data_fid.nbEcho, data_fid.fullEcho)
    return data_spc

def phase(data_spc, dic):
    """
    Automatic spectrum phasing
    Usage:  data_spc = spc(dic, data_fid)
    Input:  data_spc     spectrum
            dic          data parameters (dictionary)
    Output: data_spc1    phased data
    """
    data_spc1 = data_spc.copy()                          # avoid data corruption
    # 1D data set or direct dimension of nD dataset
    data_spc1 = ng.proc_base.ps(data_spc1, \
        dic['procs']['PHC0'], dic['procs']['PHC1'], True)       # phasing
    # 2D data set
    if data_spc1.ndim == 2:                             # 2D data set
        data_spc1 = ng.proc_base.tp_hyper(data_spc1)    # hypercomplex transpose
        data_spc1 = ng.proc_base.ps(data_spc1, \
            dic['proc2s']['PHC0'], dic['proc2s']['PHC1'], True) # phasing
        data_spc1 = ng.proc_base.tp_hyper(data_spc1)    # hypercomplex transpose
    # higher dimensions data set
    elif data_spc1.ndim > 2:
        raise NotImplementedError(
            'Data of dimensions higher than 2 are not supported')
    return data_spc1