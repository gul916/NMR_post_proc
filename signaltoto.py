#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
		self._nbPt = 0
		self._aquiT = 0
		self._de = 0

		self._dureeT = 0
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
		return self._nbPt
	def _set_nbPt(self,newnbPt):
		if not isinstance(newnbPt, int):
			raise ValueError("nbPt must be of type int")
		if newnbPt <= 0:
			raise ValueError("dw must be > 0")
		self._nbPt = newnbPt
	nbPt = property(_get_nbPt,_set_nbPt)
	

	def _get_aquiT(self):
		return self._aquiT
	aquiT = property(_get_aquiT)
	def set_aquiT(self):
		self._aquiT = (self._nbPt -1)*self._dw2


	def _get_de(self):
		return self._de
	def _set_de(self,newde):
		self._de = newde
	de = property(_get_de,_set_de)


	def _get_dureeT(self):
		return self._dureeT
	dureeT = property(_get_dureeT)
	def set_dureeT(self):
		self._dureeT = self._aquiT + self._de


	def _get_dureeSignal(self):
		return self._dureeSignal
	dureeSignal = property(_get_dureeSignal)
	def set_dureeSignal(self):
		self._dureeSignal = self._nbHalfEcho * self._halfEcho


	def _get_nbPtHalfEcho(self):
		return self._nbPtHalfEcho
	nbPtHalfEcho = property(_get_nbPtHalfEcho)
	def set_nbPtHalfEcho(self):
		self._nbPtHalfEcho = int(self._halfEcho / self._dw2)
		if (self._halfEcho % self._dw2) != 0:
			print("Warning : fullEcho is supposed to be a multiple of dw2")


	def _get_nbPtSignal(self):
		return self._nbPtSignal
	nbPtSignal = property(_get_nbPtSignal)
	def set_nbPtSignal(self):
		self._nbPtSignal = self._nbPtHalfEcho * self._nbHalfEcho


	def _get_missingPts(self):
		return self._missingPts
	missingPts = property(_get_missingPts)
	def set_missingPts(self):
		self._missingPts = self._nbPt - self._nbPtSignal


	def _get_nbPtDeadTime(self):
		return self._nbPtDeadTime
	nbPtDeadTime = property(_get_nbPtDeadTime)
	def set_nbPtDeadTime(self):
		self._nbPtDeadTime = int(self._de / self._dw2)
		if (self._de % self._dw2) != 0:
			print("Warning : de is supposed to be a multiple of dw2")



	def setValues_user(self,newfirstDec,newfullEcho,newnbEcho):
		try:
			self.firstDec = newfirstDec
			self.fullEcho = newfullEcho
			self.nbEcho = newnbEcho
			self.set_dureeSignal()
		except ValueError as err:
			print("function setValues_user() returns error :")
			print("  ",err.args[0])
			print()
		else:
			self._userInitialised = True

	def setValues_topspin(self,newdw,newnbPt,newde):
		try:
			self.dw = newdw
			self.nbPt = newnbPt
			self.de = newde
			self.set_aquiT()
			self.set_dureeT()
			self.set_nbPtDeadTime()
		except ValueError as err:
			print("function setValues_topspin() returns error :")
			print("  ",err.args[0])
			print()
		else:
			self._topspinInitialised = True

	def checkInitialisation(self):
		if not self._topspinInitialised:
			raise ValueError("You must set topsin values with setValues_topspin() first !")
		elif not self._userInitialised:
			raise ValueError("You must set user values with setValues_user() first !")

	
	def _get_data(self):
		return self._data
	data = property(_get_data)
	def setData(self,newdata):
		try:
			self.checkInitialisation()
			self.set_nbPtHalfEcho()
			self.set_nbPtSignal()
			self.set_missingPts()
		except ValueError as err:
			print("function setData() returns error :")
			print("  ",err.args[0])
			print()
		else:
			self._data = newdata
	