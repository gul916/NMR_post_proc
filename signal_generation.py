#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File started from CPMG_PAG_2017-08-11
@authors: Pierre-Aymeric GILLES & Guillaume LAURENT
"""


###----------------------------------------------------------------------------
### PARAMETERS
###----------------------------------------------------------------------------

import math
import matplotlib.pyplot as plt
import numpy as np
import signaltoto as sig

# demandés à l'utilisateur :
fullEcho = 10e-3
nbEcho = 20					# 38 for less than 8k points, 76 for more
firstDec = True

halfEcho = fullEcho / 2
nbHalfEcho = (nbEcho * 2)
if firstDec == True:
	nbHalfEcho += 1

# paramètres :
dw = 24e-6					# temps entre 2 prises de points
dw2 = 2*dw
nbPt = 16384				# nb de pts complexes  ( nbPt == td/2 )
aquiT = (nbPt-1)*dw2		# temps acquisition total : aquiT = (nbPt-1)*dw2
de = 96e-6					# temps de non-acquisition au début
# de = 0

# noise generation 
mean = 0
std = 0.3

# calculés :
dureeT = aquiT + de
dureeSignal = nbHalfEcho * halfEcho
#nbPtSignal_via_dureeSignal = np.rint(1 + (dureeSignal / (dw2)))
nbPtHalfEcho = int(halfEcho / dw2)	# avec arrondi
#nbPtHalfEcho = 1 + (halfEcho / (dw2))		# sans arrondi ici
nbPtSignal = nbPtHalfEcho * nbHalfEcho
#nbPtHalfEcho_via_nbPtSignal = nbPtSignal/(2*nbEcho+1)
missingPts = nbPt-nbPtSignal
nbPtDeadTime = int(de / dw2)	# nb de pts à 0 au début


# 1st frequency
t21 = 500e-3
t21star = 1e-3
nu1 = 1750

# 2nd frequency
t22 = 100e-3
t22star = 0.5e-3
nu2 = -2500



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
# print("\tnbPt =", nbPt)
# print("\taquiT =", aquiT)
# print("\tde =", de)

# print("\n Valeurs calculées :\n")
# print("\tdureeSignal =", dureeSignal)
# print("\tduree totale (dureeT) =", dureeT)
# print("\tnbPtHalfEcho =", nbPtHalfEcho)
# #print("\tnbPtSignal_via_dureeSignal =", nbPtSignal_via_dureeSignal)
# #print("\tnbPtSignal_via_nbPtHalfEcho =", nbPtSignal)
# print("\tnbPtSignal =", nbPtSignal)
# print("\tmissingPts =", missingPts)
# print("\tnbPtDeadTime =", nbPtDeadTime)


# print("\nSpecified SVD method :", SVD_method)


#%%----------------------------------------------------------------------------
### SYNTHESE DE SIGNAL RMN
###----------------------------------------------------------------------------

def signalCreation():
	desc = firstDec
	A = np.array([])

	print("\n------------------------------------------------------------------------")
	# print("\n 1er point de chaque demi echo à la creation : ")
	# tracé de la courbe par les demi echos
	for i in range (0, nbHalfEcho):

		deb = i*halfEcho
		fin = (i+1)*halfEcho-dw2

		timei = np.linspace(deb,fin,nbPtHalfEcho)

		if(desc==True):
			yi1 = np.exp(1j*2*np.pi*nu1*(timei[:]-deb)) \
				*np.exp(-(timei[:]-deb)/t21star) * np.exp(-(timei[:])/t21)
			yi2 = np.exp(1j*2*np.pi*nu2*(timei[:]-deb)) \
				*np.exp(-(timei[:]-deb)/t22star) * np.exp(-(timei[:])/t22)
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

		A = np.concatenate((A[:],yi[:]))

	#print("\tA.size =",A.size)
	end = np.zeros((missingPts,), dtype=np.complex)
	A = np.concatenate((A[:],end[:]))
	#print("\tA.size =",A.size)


	# affichage de la matrice A (signal entier) avant ajout du bruit
	timeT = np.linspace(0,dureeT,nbPt)

	plt.ion()					# interactive mode on
	fig1 = plt.figure()
	fig1.suptitle("CPMG NMR signal synthesis", fontsize=16)

	ax1 = fig1.add_subplot(411)
	ax1.set_title("FID without noise")
	ax1.plot(timeT[:],A[:].real)

	nbPtFreq = nbPt*4
	freq = np.linspace(-1/(2*dw2), 1/(2*dw2), nbPtFreq)
	ax2 = fig1.add_subplot(412)
	ax2.set_title("SPC without noise")
	ax2.plot(freq[:], np.fft.fftshift(np.fft.fft(A[:], nbPtFreq)).real)
	ax2.plot(freq[:], np.fft.fftshift(np.fft.fft(A[:nbPtHalfEcho]*nbEcho, nbPtFreq)).real)


	# noise generation 
	num_samples = nbPt
	noise = np.random.normal(mean, std, size=num_samples)

	# ajout du bruit au signal
	A+=noise


	## affichage de la matrice A (signal entier) après ajout du bruit
	#timeT = np.linspace(0,dureeT,nbPt)
	#ax3 = fig1.add_subplot(413)
	#ax3.set_title("FID with noise and dead time")
	#ax3.plot(timeT[:],A[:].real)


	# print("\n 1er point de chaque demi echo dans la matrice A (avec bruit) : ")
	# for i in range (0, nbHalfEcho):
	# 	pt = i*nbPtHalfEcho
	# 	print("\t1er elem du demi echo", i ," (point", pt, ") =", A[pt])


	# suppression de points pour prendre en compte le dead time

	A = A[nbPtDeadTime:]


	# affichage de la matrice A (signal entier) après prise en compte du dead time
	timeT = np.linspace(0,dureeT,nbPt-nbPtDeadTime)
	ax3 = fig1.add_subplot(413)
	ax3.set_title("FID with noise and dead time")
	ax3.plot(timeT[:],A[:].real)


	# affichage du spectre
	nbPtFreq = nbPt*4
	freq = np.linspace(-1/(2*dw2), 1/(2*dw2), nbPtFreq)
	ax4 = fig1.add_subplot(414)
	ax4.set_title("SPC with noise and dead time")
	ax4.plot(freq[:], np.fft.fftshift(np.fft.fft(A[:], nbPtFreq)).real)
	ax4.plot(freq[:], np.fft.fftshift(np.fft.fft(A[:nbPtHalfEcho]*nbEcho, nbPtFreq)).real)
	fig1.tight_layout(rect=[0, 0, 1, 0.95])			# Avoid superpositions on display
	fig1.show()					# Display figure

	generatedSignal = sig.Signal()
	generatedSignal.setValues_user(firstDec,fullEcho,nbEcho)
	generatedSignal.setValues_topspin(dw,nbPt,de)
	generatedSignal.setData(A)

	return generatedSignal


if __name__ == "__main__":
	s1 = signalCreation()
	print('s1.dw : ', s1.dw)
	print('s1.data : ', s1.data)
	input('\nPress enter key to exit') # have the graphs stay displayed even when launched from linux terminal