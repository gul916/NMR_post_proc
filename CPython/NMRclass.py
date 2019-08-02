#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@authors: Pierre-Aymeric GILLES & Guillaume LAURENT
"""

# Python libraries
import numpy as np


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
        if (Decimal(self._halfEcho) % Decimal(self._dw2)) != Decimal('0.00'):
            # modulo doesn't work with decimal numbers (precision)
            # print("Warning : HalfEcho is supposed to be a multiple of 2*dw")
            pass


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
        if (Decimal(self._de) % Decimal(self._dw2)) != Decimal('0.00'):
            print("Warning : de is supposed to be a multiple of 2*dw")


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
    