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

import signal_generation as sigG

import svd_sfa as svd
svd.init()

A = sigG.signalCreation().data
print(A)

# demandés à l'utilisateur :
firstDec = True
fullEcho = 10e-3
halfEcho = fullEcho / 2
nbEcho = 20					# 38 for less than 8k points, 76 for more
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


lb = 3/(np.pi*halfEcho)		# line broadening (Herz)

# calculés :
dureeT = aquiT + de
dureeSignal = nbHalfEcho * halfEcho
#nbPtSignal_via_dureeSignal = np.rint(1 + (dureeSignal / (dw2)))
nbPtHalfEcho = int(halfEcho / dw2)	# avec arrondi
#nbPtHalfEcho = 1 + (halfEcho / (dw2))		# sans arrondi ici
nbPtSignal = nbPtHalfEcho * nbHalfEcho
#nbPtHalfEcho_via_nbPtSignal = nbPtSignal/(2*nbEcho+1)
missingPts = nbPt-nbPtHalfEcho*nbHalfEcho
nbPtDeadTime = int(de / dw2)	# nb de pts à 0 au début

# Singular Value Decomposition
# 0 : no SVD applied
# Method 1: on full 1D with echoes --> very long
# Method 2: on full 2D of stacked echoes --> very fast
# Method 3: on separated echoes --> fast
SVD_method = 2

#if nbPtSignal <= 8192:
#	SVD_method = 1
#else:
#	SVD_method = 2
#thres = 16


###----------------------------------------------------------------------------
### AFFICHAGE DES VALEURS DES PARAMETRES (RETOUR UTILISATEUR)
###----------------------------------------------------------------------------

print("\n------------------------------------------------------------------------")
print('File : cpmg.py')
print("------------------------------------------------------------------------\n")
print("\nSYNTHESE DES VALEURS :")
print("\n Valeurs demandées à l'utilisateur :\n")
print("\tfirstDec =", firstDec)
print("\tfullEcho =", fullEcho)
print("\thalfEcho =", halfEcho, "(déduit de full echo)")
print("\tnbEcho =", nbEcho)
print("\tnbHalfEcho =", nbHalfEcho, "(calculée : dépend de 1ere decroissance ou non)")

print("\n Valeurs passées en paramètres :\n")
print("\tdw =", dw)
print("\tnbPt =", nbPt)
print("\taquiT =", aquiT)
print("\tde =", de)

print("\n Valeurs calculées :\n")
print("\tdureeSignal =", dureeSignal)
print("\tduree totale (dureeT) =", dureeT)
print("\tnbPtHalfEcho =", nbPtHalfEcho)
#print("\tnbPtSignal_via_dureeSignal =", nbPtSignal_via_dureeSignal)
#print("\tnbPtSignal_via_nbPtHalfEcho =", nbPtSignal)
print("\tnbPtSignal =", nbPtSignal)
print("\tmissingPts =", missingPts)
print("\tnbPtDeadTime =", nbPtDeadTime)


print("\nSpecified SVD method :", SVD_method)



#%%----------------------------------------------------------------------------
### Exploitation du signal
###----------------------------------------------------------------------------

timeT = np.linspace(0,dureeT,nbPt-nbPtDeadTime)
fig2 = plt.figure()
fig2.suptitle("CPMG NMR signal processing", fontsize=16)
ax1 = fig2.add_subplot(411)
ax1.set_title("Raw FID !!! Different scales !!!")
ax1.plot(timeT[:],A[:].real)

# ajout de points à zero pour compenser le dead time
zerosToAdd = np.zeros((nbPtDeadTime,), dtype=np.complex)
print("\nzerosToAdd.size =", zerosToAdd.size)
A = np.concatenate((zerosToAdd[:],A[:]))
print("A.size =", A.size)


# on supprime les points en trop
echos1D = A[0:nbPtSignal]
timeT = np.linspace(0, dureeSignal-dw2, nbPtSignal)
print("echos1D.size =",echos1D.size)


# preprocessing
print("preprocessing...")
desc = firstDec

for i in range (0, nbHalfEcho):
	if (desc==True):
		timei = np.linspace(0,halfEcho-dw2,nbPtHalfEcho)
	else:
		timei = np.linspace(-halfEcho,-dw2,nbPtHalfEcho)
	echos1D[i*nbPtHalfEcho:(i+1)*nbPtHalfEcho] *= np.exp(-abs(timei[:])*np.pi*lb)
	desc = not(desc)

ax2 = fig2.add_subplot(412)
ax2.set_title("FID after preprocessing")
ax2.plot(timeT[:],echos1D[:].real)


# Singular Value Decompostion (SVD) on Toeplitz matrix
if (SVD_method == 1):
	echos1D = svd.svd(echos1D,nbHalfEcho,nbPtHalfEcho,SVD_method)
	
	ax3 = fig2.add_subplot(413)
	ax3.set_title("FID after SVD on Toeplitz matrix")
	ax3.plot(timeT[:], echos1D[:].real)
	
	ax4 = fig2.add_subplot(414)
	ax4.set_title("SPC after SVD on Toeplitz matrix")
	ax4.plot(freq[:], np.fft.fftshift(np.fft.fft(echos1D[:], nbPtFreq)).real)


fig2.tight_layout(rect=[0, 0, 1, 0.95])			# Avoid superpositions on display
fig2.show()					# Display figure


#%%
# separation des echos
# si 1ere decroissance : on inclut un demi echo de 0 devant 
if firstDec:
	A[0:nbPtHalfEcho] *= 2		# !!! A mettre juste avant la FFT
	firstHalfEcho = np.zeros((nbPtHalfEcho,), dtype=np.complex)
	echos1D = np.concatenate((firstHalfEcho[:],echos1D[:]))
print("echos1D.size =", echos1D.size)


# separation après avoir determiné le nb de pts des echos et le nb d'echos
nbPtFullEcho = 2*nbPtHalfEcho
nbFullEchoTotal = int((nbHalfEcho+1)/2) 
# print("\n 1er elem de chaque demi echo à la separation (reshape) des echos")
echos2D = echos1D.reshape(nbFullEchoTotal,nbPtFullEcho)


# affichage des echos separés
timeFullEcho = np.linspace(0,fullEcho-dw2,nbPtFullEcho)
fig3 = plt.figure()
fig3.suptitle("Processing of separated echoes", fontsize=16)
ax1 = fig3.add_subplot(411)
ax1.set_title("FID after echoes separation")

for i in range (0, nbFullEchoTotal):
	#for j in range (0, nbPtFullEcho):
		#echos2D[i][j]+=2*i
	ax1.plot(timeFullEcho[:],(echos2D[i][0:nbPtFullEcho]).real)

	# print("\t1er elem du demi echo", 2*i ," =", echos2D[i][0])
	# print("\t1er elem du demi echo", 2*i+1 ," =", echos2D[i][nbPtHalfEcho])


'''
# somme des echos (temporelle) et affichage 
# Rq : pas utilisé pour la suite -> juste pour voir ce que donne la somme temporelle
sommeTempo = np.zeros((nbPtFullEcho,nbFullEchoTotal), dtype=np.complex)
for i in range (0, nbFullEchoTotal):
	for j in range (0, nbPtFullEcho):
		sommeTempo[j] += echos2D[i][j]
plt.figure()
plt.plot(timeFullEcho[:],sommeTempo[0:nbPtFullEcho].real)
plt.show() # affiche la figure a l'ecran
'''


# Singular Value Decompostion (SVD) on echo matrix
if (SVD_method == 2):
	echos2D = svd.svd(echos2D,nbHalfEcho,nbPtHalfEcho,SVD_method)

	ax2 = fig3.add_subplot(412)
	ax2.set_title("FID after SVD on echoes matrix")
	for i in range (0, nbFullEchoTotal):
		ax2.plot(timeFullEcho[:],(echos2D[i][0:nbPtFullEcho]).real)



# Singular Value Decompostion (SVD) on Toeplitz matrix of each echo
if (SVD_method == 3):
	echos2D = svd.svd(echos2D,nbHalfEcho,nbPtHalfEcho,SVD_method)

	ax2 = fig3.add_subplot(412)
	ax2.set_title("FID after SVD on Toeplitz matrix of echoes")
	for i in range (0, nbFullEchoTotal):
		ax2.plot(timeFullEcho[:],(echos2D[i][:]).real)
	

# prediction lineaire


# somme ponderee -> calcul de Imax
Imax = np.empty([nbFullEchoTotal])
for i in range (0, nbFullEchoTotal):
	#Imax = np.amax((echos2D[i][:]).real)
	#echos2D[i][0:nbPtFullEcho]*=Imax
	Imax[i] = np.amax((echos2D[i,:]).real)


# correction T2


# fftshift => inversion des halfEcho 2 à 2
echosFFTSHIFT = np.fft.fftshift(echos2D[0:nbFullEchoTotal][0:nbPtFullEcho],axes=1)
echosFFTSHIFT[0][0]*=0.5		# permet de corriger l'artefact due à la FFT
echosFFT = np.fft.fftshift(np.fft.fft(echosFFTSHIFT[:,:],axis=1),axes=1)

ax3 = fig3.add_subplot(413)
ax3.set_title("SPC after SVD")
for i in range (0, nbFullEchoTotal):
	ax3.plot(timeFullEcho[:],(echosFFT[i][0:nbPtFullEcho]).real)


# ponderation par Imax
for i in range (0, nbFullEchoTotal):
	echosFFT[i][0:nbPtFullEcho]*=Imax[i]

# affichage du spectre de la 1ere decroissance pour comparaison avec la somme
timeFullEcho = np.linspace(0,fullEcho-dw2,nbPtFullEcho)
#plt.plot(timeFullEcho[:],echosFFT[0][0:nbPtFullEcho].real)


# somme des echos (spectrale)
sommeSpect = np.zeros((nbPtFullEcho*nbFullEchoTotal), dtype=np.complex)
for i in range (0, nbFullEchoTotal):
	for j in range (0, nbPtFullEcho):
		sommeSpect[j] += echosFFT[i][j]
ax4 = fig3.add_subplot(414)
ax4.set_title("SPC after Imax ponderation and sum")
ax4.plot(timeFullEcho[:],sommeSpect[0:nbPtFullEcho].real)
fig3.tight_layout(rect=[0, 0, 1, 0.95])			# Avoid superpositions on display
fig3.show()					# Display figure


#print("\n------------------------------------------------------------------------\n\n")

input('\nPress enter key to exit') # have the graphs stay displayed even when launched from linux terminal