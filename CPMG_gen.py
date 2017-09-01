#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@authors: Pierre-Aymeric GILLES & Guillaume LAURENT
"""

# Python libraries
import matplotlib.pyplot as plt
import numpy as np
# User defined library
import NMRclass as sig



###----------------------------------------------------------------------------
### PARAMETERS
###----------------------------------------------------------------------------

# Asked to the user
firstDec = True
fullEcho = 10e-3
nbEcho = 20					# 38 for less than 8k points, 76 for more

# From Topspin interface
td = 32768					# nb of real points + nb of imag points
dw = 24e-6					# dwell time between two points
de = 96e-6					# dead time before signal acquisition
# de = 0

# Simulation of noise
mean = 0
std = 0.1

# 1st frequency
amp1 = 1						# amplitude
nu1 = 1750					# frequency in Hz
t21 = 500e-3				# true relaxation time
t21star = 1e-3				# apparent relaxation time
sigma1 = np.sqrt(2*np.log(2))*t21star		# for Gaussian shape
gl1 = 0						# 0: Lorentzian shape, 1: Gaussian shape

# 2nd frequency
amp2 = 1						# amplitude
nu2 = -2500					# frequency in Hz
t22 = 100e-3				# true relaxation time
t22star = 0.5e-3			# apparent relaxation time
sigma2 = np.sqrt(2*np.log(2))*t22star;		# for Gaussian shape
gl2 = 0						# 0: Lorentzian shape, 1: Gaussian shape

# Calculated
halfEcho = fullEcho / 2
nbHalfEcho = int(nbEcho * 2)
if firstDec == True:
	nbHalfEcho += int(1)

dw2 = 2*dw					# 1 real + 1 imag points are needed to have a complex point
td2 = int(td/2)				# nb of complex points
acquiT = (td2-1)*dw2			# total acquisition time, starts at 0
dureeSignal = nbHalfEcho * halfEcho
nbPtHalfEcho = int(halfEcho / dw2)
nbPtSignal = int(nbPtHalfEcho * nbHalfEcho)
missingPts = int(td2-nbPtSignal)		# Number of points after last echo
nbPtDeadTime = int(de / dw2)	# Number of point during dead time

if (gl1 < 0) or (gl1 > 1):
	raise ValueError("gl1 must be between 0 and 1")
if (gl2 < 0) or (gl2 > 1):
	raise ValueError("gl2 must be between 0 and 1")

# Axes
timeT = np.linspace(0,acquiT,td2)
nbPtFreq = int(td2*4)			# zero filling
freq = np.linspace(-1/(2*dw2), 1/(2*dw2), nbPtFreq)



# ###----------------------------------------------------------------------------
# ### AFFICHAGE DES VALEURS DES PARAMETRES (RETOUR UTILISATEUR)
# ###----------------------------------------------------------------------------

# print("\n------------------------------------------------------------------------")
# print('File : signal_generation')
# print("------------------------------------------------------------------------\n")
# print("\nSYNTHESE DES VALEURS :")
# print("\n Valeurs demandées à l'utilisateur :\n")
# print("\tfirstDec =", firstDec)
# print("\tfullEcho =", fullEcho)
# print("\thalfEcho =", halfEcho, "(déduit de full echo)")
# print("\tnbEcho =", nbEcho)
# print("\tnbHalfEcho =", nbHalfEcho, "(calculée : dépend de 1ere decroissance ou non)")

# print("\n Valeurs passées en paramètres :\n")
# print("\tdw =", dw)
# print("\tnbPt =", td)
# print("\taquiT =", acquiT)
# print("\tde =", de)

# print("\n Valeurs calculées :\n")
# print("\tdureeSignal =", dureeSignal)
# print("\tnbPtHalfEcho =", nbPtHalfEcho)
# #print("\tnbPtSignal_via_dureeSignal =", nbPtSignal_via_dureeSignal)
# #print("\tnbPtSignal_via_nbPtHalfEcho =", nbPtSignal)
# print("\tnbPtSignal =", nbPtSignal)
# print("\tmissingPts =", missingPts)
# print("\tnbPtDeadTime =", nbPtDeadTime)


# print("\nSpecified SVD method :", SVD_method)


#%%---------------------------------------------------------------------------
### SYNTHESE DE SIGNAL RMN
###----------------------------------------------------------------------------

def signal_generation():
	desc = firstDec
	Aref = np.array([])

	print("\n------------------------------------------------------------------------")
	# print("\n 1er point de chaque demi echo à la creation : ")
	# tracé de la courbe par les demi echos
	
	for i in range (0, nbHalfEcho):
		deb = i*halfEcho
		fin = (i+1)*halfEcho-dw2
		timei = np.linspace(deb,fin,nbPtHalfEcho)
		if(desc==True):
			tzero = i*halfEcho
		else:
			tzero = (i+1)*halfEcho
		
		# Gaussian-Lorentzian broadening (t2star) with exponential relaxation (t2)
		yi1 = amp1 * np.exp(1j*2*np.pi*nu1*(timei[:]-tzero)) \
			* ((1-gl1)*np.exp(-abs(timei[:]-tzero)/t21star) \
			+ (gl1)*np.exp(-(timei[:]-tzero)**2/(2*sigma1**2))) \
			* np.exp(-(timei[:])/t21)
		yi2 = amp2 * np.exp(1j*2*np.pi*nu2*(timei[:]-tzero)) \
			* ((1-gl2)*np.exp(-abs(timei[:]-tzero)/t22star) \
			+ (gl2)*np.exp(-(timei[:]-tzero)**2/(2*sigma2**2))) \
			* np.exp(-(timei[:])/t21)
		yi = yi1 + yi2
		# print("\t1er elem du demi echo", i ," =", yi[0])

		Aref = np.concatenate((Aref[:],yi[:]))
		desc = not(desc)

	#print("\tAref.size =",Aref.size)
	end = np.zeros(missingPts, dtype=np.complex)
	Aref = np.concatenate((Aref[:],end[:]))
	#print("\tAref.size =",Aref.size)

	A = Aref.copy()		# Avoid reference signal corruption



	# Suppression of points during dead time 
	end = np.zeros(nbPtDeadTime, dtype=np.complex)
	A = np.concatenate((A[nbPtDeadTime:],end[:]))
	Adead = A.copy()



	# Adding noise
	noise = np.random.normal(mean, std, td2) + 1j*np.random.normal(mean, std, td2)
	A+=noise



	# Saving data to Signal class
	generatedSignal = sig.Signal()
	generatedSignal.setValues_topspin(td,dw,de)
	generatedSignal.setValues_user(firstDec,fullEcho,nbEcho)
	generatedSignal.setData(A)



	#%% Figures
	# Spectra calculation
	ArefSPC = Aref.copy()					# Avoid reference FID corruption
	ASPC = A.copy()						# Avoid noisy FID corruption

	if firstDec == True:
		ArefSPC = ArefSPC[:nbPtHalfEcho] * nbHalfEcho
	else:
		ArefSPC = ArefSPC[nbPtHalfEcho:2*nbPtHalfEcho] * nbHalfEcho
	
	ArefSPC[0] *= 0.5						# FFT artefact correction
	ASPC[0] *= 0.5
	ArefSPC = np.fft.fftshift(np.fft.fft(ArefSPC[:], nbPtFreq))
	ASPC = np.fft.fftshift(np.fft.fft(ASPC[:], nbPtFreq))	# FFT with zero filling

	# Plotting
	plt.ion()					# interactive mode on
	fig1 = plt.figure()
	fig1.suptitle("CPMG NMR signal synthesis", fontsize=16)

	# Reference signal display
	ax1 = fig1.add_subplot(411)
	ax1.set_title("Reference FID")
	ax1.plot(timeT[:],Aref[:].real)
	ax1.plot(timeT[:],Aref[:].imag)
	ax1.set_xlim([-halfEcho, acquiT+halfEcho])

	# Signal display after dead time suppression
	ax2 = fig1.add_subplot(412)
	ax2.set_title("FID after dead time suppression")
	ax2.plot(timeT[:],Adead[:].real)
	ax2.plot(timeT[:],Adead[:].imag)
	ax2.set_xlim([-halfEcho, acquiT+halfEcho])

	# Signal display after dead time suppression and noise addition
	ax3 = fig1.add_subplot(413)
	ax3.set_title("FID with added noise")
	ax3.plot(timeT[:],A[:].real)
	ax3.plot(timeT[:],A[:].imag)
	ax3.set_xlim([-halfEcho, acquiT+halfEcho])

	# Spectra display
	ax4 = fig1.add_subplot(414)
	ax4.set_title("Noisy SPC and reference SPC")
	ax4.invert_xaxis()
	ax4.plot(freq[:], ASPC.real)
	ax4.plot(freq[:], ArefSPC.real)

	fig1.tight_layout(rect=[0, 0, 1, 0.95])		# Avoid superpositions on display
	fig1.show()								# Display figure



	return generatedSignal


#%%----------------------------------------------------------------------------
### When this file is executed directly
###----------------------------------------------------------------------------

if __name__ == "__main__":
	s1 = signal_generation()
#	print('s1.dw : ', s1.dw)
#	print('s1.data : ', s1.data)
	
#	np.savetxt('CPMG_FID.csv', np.transpose([s1.data[:].real, s1.data[:].imag]),\
#			delimiter='\t', header='sep=\t', comments='')

	###Values used to save CPMG_FID.csv, please don't overwrite this file.
	#fullEcho = 10e-3
	#nbEcho = 20				# 38 for less than 8k points, 76 for more
	#firstDec = True
	#
	#dw = 24e-6					# dwell time between two points
	#td = 32768					# nb of real points + nb of imag points
	#de = 96e-6					# dead time before signal acquisition
	#
	#mean = 0
	#std = 0.1
	#
	#t21 = 500e-3
	#t21star = 1e-3
	#nu1 = 1750
	#
	#t22 = 100e-3
	#t22star = 0.5e-3
	#nu2 = -2500
	#
	
	
	
	input('\nPress enter key to exit') # wait before closing terminal