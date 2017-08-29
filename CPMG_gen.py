#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@authors: Pierre-Aymeric GILLES & Guillaume LAURENT
"""

# Python libraries
import matplotlib.pyplot as plt
import numpy as np
# User defined libraries
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
t21 = 500e-3
t21star = 1e-3
nu1 = 1750

# 2nd frequency
t22 = 100e-3
t22star = 0.5e-3
nu2 = -2500

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
			yi1 = np.exp(1j*2*np.pi*nu1*(timei[:]-deb)) \
				* np.exp(-(timei[:]-deb)/t21star) * np.exp(-(timei[:])/t21)
			yi2 = np.exp(1j*2*np.pi*nu2*(timei[:]-deb)) \
				* np.exp(-(timei[:]-deb)/t22star) * np.exp(-(timei[:])/t22)
	#		yi2 = np.zeros(timei.size, dtype='complex')
		else:
			yi1 = np.exp(1j*2*np.pi*nu1*(-(fin+dw2)+timei[:])) \
				* np.exp((-(fin+dw2)+timei[:])/t21star) * np.exp(-(timei[:])/t21)
			yi2 = np.exp(1j*2*np.pi*nu2*(-(fin+dw2)+timei[:])) \
				* np.exp((-(fin+dw2)+timei[:])/t22star) * np.exp(-(timei[:])/t22)
	#		yi2 = np.zeros(timei.size, dtype='complex')
		yi = yi1 + yi2
		desc = not(desc)

		# print("\t1er elem du demi echo", i ," =", yi[0])

		Aref = np.concatenate((Aref[:],yi[:]))

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
	ArefSpc = Aref.copy()					# Avoid reference FID corruption
	ASpc = A.copy()						# Avoid noisy FID corruption

	if firstDec == True:
		ArefSpc = ArefSpc[:nbPtHalfEcho] * nbHalfEcho
	else:
		ArefSpc = ArefSpc[nbPtHalfEcho:2*nbPtHalfEcho] * nbHalfEcho
	
	ArefSpc[0] *= 0.5						# FFT artefact correction
	ASpc[0] *= 0.5
	ArefSpc = np.fft.fftshift(np.fft.fft(ArefSpc[:], nbPtFreq))
	ASpc = np.fft.fftshift(np.fft.fft(ASpc[:], nbPtFreq))	# FFT with zero filling

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
	ax4.plot(freq[:], ASpc.real)
	ax4.plot(freq[:], ArefSpc.real)

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
	
	
	
	input('\nPress enter key to exit') # have the graphs stay displayed even when launched from linux terminal