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
    """

    def __init__(self):
        self._fullEcho = 0
        self._halfEcho = 0
        self._nbEcho = 0
        self._nbHalfEcho = 0
        self._firstDec = False

        self._dw = 0
        self._dw2 = 0
        self._td = 0
        self._td2 = 0
        self._acquiT = 0
        self._de = 0

        self._dureeSignal = 0
        self._nbPtHalfEcho = 0
        self._nbPtSignal = 0
        self._missingPts = 0
        self._nbPtDeadTime = 0

        self._userInitialised = False
        self._topspinInitialised = False
        self._data = np.array([])


    def _get_fullEcho(self):
        return self._fullEcho
    def _set_fullEcho(self,newfullEcho):
        if newfullEcho <= 0:
            raise ValueError("fullEcho must be > 0")
        self._fullEcho = newfullEcho
        self._halfEcho = self._fullEcho / 2
    fullEcho = property(_get_fullEcho,_set_fullEcho)


    def _get_halfEcho(self):
        return self._halfEcho
    halfEcho = property(_get_halfEcho)


    def _get_nbEcho(self):
        return self._nbEcho
    def _set_nbEcho(self,newnbEcho):
        if not isinstance(newnbEcho, int):
            raise ValueError("nbEcho must be of type int")
        if newnbEcho <= 0:
            raise ValueError("nbEcho must be > 0")
        self._nbEcho = newnbEcho
        self._nbHalfEcho = self._nbEcho * 2
        if self._firstDec == True:
            self._nbHalfEcho += 1
    nbEcho = property(_get_nbEcho,_set_nbEcho)


    def _get_nbHalfEcho(self):
        return self._nbHalfEcho
    nbHalfEcho = property(_get_nbHalfEcho)


    def _get_firstDec(self):
        return self._firstDec
    def _set_firstDec(self,newfirstDec):
        if not isinstance(newfirstDec, bool):
            raise ValueError("firstDec must be of type boolean")
        if self._firstDec != newfirstDec:
            if self._firstDec == False:
                self._nbHalfEcho += 1
            else:
                self._nbHalfEcho -= 1
            self._firstDec = newfirstDec
    firstDec = property(_get_firstDec,_set_firstDec)


    def _get_dw(self):
        return self._dw
    def _set_dw(self,newdw):
        if newdw <= 0:
            raise ValueError("dw must be > 0")
        self._dw = newdw
        self._dw2 = 2 * self._dw
    dw = property(_get_dw,_set_dw)


    def _get_dw2(self):
        return self._dw2
    dw2 = property(_get_dw2)

    
    def _get_nbPt(self):
        return self._td
    def _set_nbPt(self,newnbPt):
        if not isinstance(newnbPt, int):
            raise ValueError("td must be of type int")
        if newnbPt <= 0:
            raise ValueError("dw must be > 0")
        self._td = newnbPt
        self._td2 = int(self._td / 2)
    td = property(_get_nbPt,_set_nbPt)
    
    
    def _get_td2(self):
        return self._td2
    td2 = property(_get_td2)


    def _get_acquiT(self):
        return self._acquiT
    acquiT = property(_get_acquiT)
    def set_acquiT(self):
        self._acquiT = (self._td2 -1)*self._dw2


    def _get_de(self):
        return self._de
    def _set_de(self,newde):
        self._de = newde
    de = property(_get_de,_set_de)


    def _get_dureeSignal(self):
        return self._dureeSignal
    dureeSignal = property(_get_dureeSignal)
    def set_dureeSignal(self):
        self._dureeSignal = self._nbHalfEcho * self._halfEcho
        if (self._dureeSignal > self._acquiT):
            raise ValueError("Too many echoes during acquisition time")


    def _get_nbPtHalfEcho(self):
        return self._nbPtHalfEcho
    nbPtHalfEcho = property(_get_nbPtHalfEcho)
    def set_nbPtHalfEcho(self):
        self._nbPtHalfEcho = int(self._halfEcho / self._dw2)
        if round((self._halfEcho / self._dw2) % 1, 2) not in (0.0, 1.0):
            warnings.warn("HalfEcho is supposed to be a multiple of 2*dw")


    def _get_nbPtSignal(self):
        return self._nbPtSignal
    nbPtSignal = property(_get_nbPtSignal)
    def set_nbPtSignal(self):
        self._nbPtSignal = self._nbPtHalfEcho * self._nbHalfEcho


    def _get_missingPts(self):
        return self._missingPts
    missingPts = property(_get_missingPts)
    def set_missingPts(self):
        self._missingPts = self._td - self._nbPtSignal


    def _get_nbPtDeadTime(self):
        return self._nbPtDeadTime
    nbPtDeadTime = property(_get_nbPtDeadTime)
    def set_nbPtDeadTime(self):
        self._nbPtDeadTime = int(self._de / self._dw2)
        if round((self._de / self._dw2) % 1, 2) not in (0.0, 1.0):
            warnings.warn("de is supposed to be a multiple of 2*dw")


    def setValues_topspin(self,newnbPt,newdw,newde):
        try:
            self.td = newnbPt
            self.dw = newdw
            self.de = newde
            self.set_acquiT()
            self.set_nbPtDeadTime()
        except ValueError as err:
            print("function setValues_topspin() returns error :")
            print("  ",err.args[0])
            print()
        else:
            self._topspinInitialised = True


    def setValues_CPMG(self,newfirstDec,newfullEcho,newnbEcho):
        try:
            if not self._topspinInitialised:
                raise ValueError("You must set topsin values with setValues_topspin() first !")
            self.firstDec = newfirstDec
            self.fullEcho = newfullEcho
            self.nbEcho = newnbEcho
            self.set_dureeSignal()
            self.set_nbPtHalfEcho()
            self.set_nbPtSignal()
            self.set_missingPts()
        except ValueError as err:
            print("function setValues_CPMG() returns error :")
            print("  ",err.args[0])
            print()
        else:
            self._userInitialised = True


    def _get_data(self):
        return self._data
    data = property(_get_data)
    def setData(self,newdata):
        self._data = newdata


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
        raise NotImplementedError('Data of dimensions higher than 2 are not supported')
    return data_apod


#def spc(dic, data_fid):
def spc(data_fid):
    """
    FFT of FID with normalization and zero-filling
    Usage:  data_spc = spc(data_fid)
    Input:  data_fid     data to transform
    Output: data_spc     transformed data
    """
    data_spc = data_fid[:]                              # avoid data corruption
    # 1D data set or direct dimension of nD dataset
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
        raise NotImplementedError('Data of dimensions higher than 2 are not supported')
    return data_spc


def phase(data_spc, dic):
    """
    Automatic spectrum phasing
    Usage:  data_spc = spc(dic, data_fid)
    Input:  data_spc     spectrum
            dic          data parameters (dictionary)
    Output: data_spc1    phased data
    """
    data_spc1 = data_spc[:]                             # avoid data corruption
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
        raise NotImplementedError('Data of dimensions higher than 2 are not supported')
    return data_spc1