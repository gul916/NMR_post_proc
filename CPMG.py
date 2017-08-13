#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File started from CPMG_PAG_2017-08-11
@authors: Pierre-Aymeric GILLES & Guillaume LAURENT
"""

###----------------------------------------------------------------------------
### PARAMETERS
###----------------------------------------------------------------------------

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import dot, linalg
from time import time

# demandés à l'utilisateur :
firstDec = True
fullEcho = 10e-3
halfEcho = fullEcho / 2
nbEcho = 76
nbHalfEcho = (nbEcho * 2) 
if firstDec == True:
	nbHalfEcho += 1

# paramètres :
dw = 24e-6			# temps entre 2 prises de points
dw2 = 2*dw
nbPt = 16384			# nb de pts complexes  ( nbPt == td/2 )
aquiT = (nbPt-1)*dw2	# temps acquisition total : aquiT = (nbPt-1)*dw2
de = 96e-6				# temps de non-acquisition au début
lb = 5/(np.pi*halfEcho)			# line broadening (Herz)

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
missingPts = nbPt-nbPtHalfEcho*nbHalfEcho
nbPtDeadTime = int(de / dw2)	# nb de pts à 0 au début

# Singular Value Decomposition
# Method 1: on full 1D with echoes --> very long
# Method 2: on full 2D of stacked echoes --> very fast
# Method 3: on separated echoes --> fast
SVD_method = 3
#if nbPtSignal <= 8192:
#	SVD_method = 1
#else:
#	SVD_method = 2
thres = 16

# 1st frequency
t21 = 500e-3
t21star = 1e-3
nu1 = 1750

# 2nd frequency
t22 = 100e-3
t22star = 0.5e-3
nu2 = -2500



###----------------------------------------------------------------------------
### AFFICHAGE DES VALEURS DES PARAMETRES (RETOUR UTILISATEUR)
###----------------------------------------------------------------------------

print("\n------------------------------------------------------------------------")
print("\nSYNTHESE DES VALEURS :")
print("\nValeurs demandées à l'utilisateur :")
print("\tfirstDec =", firstDec)
print("\tfullEcho =", fullEcho)
print("\thalfEcho =", halfEcho, "(déduit de full echo)")
print("\tnbEcho =", nbEcho)
print("\tnbHalfEcho =", nbHalfEcho, "(calculée : dépend de 1ere decroissance ou non)")

print("\nValeurs passées en paramètres :")
print("\tdw =", dw)
print("\tnbPt =", nbPt)
print("\taquiT =", aquiT)
print("\tde =", de)

print("\nValeurs calculées :")
print("\tdureeSignal =", dureeSignal)
print("\tduree totale (dureeT) =", dureeT)
print("\tnbPtHalfEcho =", nbPtHalfEcho)
#print("\tnbPtSignal_via_dureeSignal =", nbPtSignal_via_dureeSignal)
#print("\tnbPtSignal_via_nbPtHalfEcho =", nbPtSignal)
print("\tnbPtSignal =", nbPtSignal)
print("\tmissingPts =", missingPts)
print("\tnbPtDeadTime =", nbPtDeadTime)



#%%----------------------------------------------------------------------------
### SYNTHESE DE SIGNAL RMN
###----------------------------------------------------------------------------


desc = firstDec
A = np.array([])

print("\n------------------------------------------------------------------------")
print("\n 1er point de chaque demi echo à la creation : ")
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
		yi = yi1 + yi2
	else:
		yi1 = np.exp(1j*2*np.pi*nu1*(-(fin+dw2)+timei[:])) \
			* np.exp((-(fin+dw2)+timei[:])/t21star) * np.exp(-(timei[:])/t21)
		yi2 = np.exp(1j*2*np.pi*nu2*(-(fin+dw2)+timei[:])) \
			* np.exp((-(fin+dw2)+timei[:])/t22star) * np.exp(-(timei[:])/t22)
#		yi2 = np.zeros(timei.size, dtype='complex')
		yi = yi1 + yi2
	desc = not(desc)

	print("\t1er elem du demi echo", i ," =", yi[0])

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


print("\n 1er point de chaque demi echo dans la matrice A (avec bruit) : ")
for i in range (0, nbHalfEcho):
	pt = i*nbPtHalfEcho
	print("\t1er elem du demi echo", i ," (point", pt, ") =", A[pt])

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
fig1.show()					# affiche la figure a l'ecran





#%%----------------------------------------------------------------------------
### Exploitation du signal
###----------------------------------------------------------------------------

fig2 = plt.figure()
fig2.suptitle("CPMG NMR signal processing", fontsize=16)
ax1 = fig2.add_subplot(411)
ax1.set_title("Raw FID !!! Different scales !!!")
ax1.plot(timeT[:],A[:].real)

# ajout de points à zero pour compenser le dead time
zerosToAdd = np.zeros((nbPtDeadTime,), dtype=np.complex)
print("\tzerosToAdd.size =", zerosToAdd.size)
A = np.concatenate((zerosToAdd[:],A[:]))
print("\tA.size =", A.size)


# on supprime les points en trop
echos1D = A[0:nbPtSignal]
timeT = np.linspace(0, dureeSignal-dw2, nbPtSignal)
print("\techos1D.size =",echos1D.size)


# preprocessing
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
	row = math.ceil(nbPtSignal / 2)
	col = nbPtSignal - row + 1
	echos1D = echos1D.astype('complex64')		# decrease SVD computation time

	# decomposition
	print("SVD on Toeplitz matrix in progress. Please be patient.")
	t_0 = time()
	T = linalg.toeplitz(echos1D[row-1::-1], echos1D[row-1::1])
	U, s, Vh = linalg.svd(T[:][:], full_matrices=False)
	S = linalg.diagsvd(s[:], row, col)
	t_1 = time()
	
	# reconstruction
	T_rec = dot(U[:,:thres], dot(S[:thres,:thres], Vh[:thres,:]))
	echos1D_rec = np.empty([nbPtSignal],dtype='complex64')
	for i in range (0, nbPtSignal):
		echos1D_rec[i] = np.mean(np.diag(T_rec[:][:],i-row+1))
	t_2 = time()
	
	print("Decomposition time:\t\t{0:8.2f}s".format(t_1 - t_0))
	print("Reconstruction time:\t\t{0:8.2f}s".format(t_2 - t_1))
	
	ax3 = fig2.add_subplot(413)
	ax3.set_title("FID after SVD on Toeplitz matrix")
	ax3.plot(timeT[:], echos1D_rec[:].real)
	
	ax4 = fig2.add_subplot(414)
	ax4.set_title("SPC after SVD on Toeplitz matrix")
	ax4.plot(freq[:], np.fft.fftshift(np.fft.fft(echos1D_rec[:], nbPtFreq)).real)
	
	echos1D = echos1D_rec[:].astype('complex')	# back to double precision

fig2.show()


#%%
# separation des echos
# si 1ere decroissance : on inclut un demi echo de 0 devant 
if firstDec:
	A[0:nbPtHalfEcho] *= 2		# !!! A mettre juste avant la FFT
	firstHalfEcho = np.zeros((nbPtHalfEcho,), dtype=np.complex)
	echos1D = np.concatenate((firstHalfEcho[:],echos1D[:]))
print("\techos1D.size =", echos1D.size)


# separation après avoir determiné le nb de pts des echos et le nb d'echos
nbPtFullEcho = 2*nbPtHalfEcho
nbFullEchoTotal = int((nbHalfEcho+1)/2) 
print("\n 1er elem de chaque demi echo à la separation (reshape) des echos")
echos2D = echos1D.reshape(nbFullEchoTotal,nbPtFullEcho)


# affichage des echos separés
timeFullEcho = np.linspace(0,fullEcho-dw2,nbPtFullEcho)
fig3 = plt.figure()
ax1 = fig3.add_subplot(411)
ax1.set_title("FID after echoes separation")

for i in range (0, nbFullEchoTotal):
	#for j in range (0, nbPtFullEcho):
		#echos2D[i][j]+=2*i
	ax1.plot(timeFullEcho[:],(echos2D[i][0:nbPtFullEcho]).real)

	print("\t1er elem du demi echo", 2*i ," =", echos2D[i][0])
	print("\t1er elem du demi echo", 2*i+1 ," =", echos2D[i][nbPtHalfEcho])


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
	row, col = echos2D.shape
	echos2D = echos2D.astype('complex64')		# decrease SVD computation time
	
	# decomposition
	print("SVD on echoes matrix in progress. Please be patient.")
	t_0 = time()
	U, s, Vh = linalg.svd(echos2D[:][:], full_matrices=False)
	S = linalg.diagsvd(s, row, col)
	t_1 = time()
	
	# reconstruction
	echos2D_rec = dot(U[:,:thres], dot(S[:thres,:thres], Vh[:thres,:]))
	t_2 = time()
	
	print("Decomposition time:\t\t{0:8.2f}s".format(t_1 - t_0))
	print("Reconstruction time:\t\t{0:8.2f}s".format(t_2 - t_1))
	
	echos2D = echos2D_rec[:][:].astype('complex')	# back to double precision
	
	ax2 = fig3.add_subplot(412)
	ax2.set_title("FID after SVD on echoes matrix")
	for i in range (0, nbFullEchoTotal):
		ax2.plot(timeFullEcho[:],(echos2D[i][0:nbPtFullEcho]).real)


# Singular Value Decompostion (SVD) on Toeplitz matrix of each echo
if (SVD_method == 3):
	row = math.ceil(nbPtFullEcho / 2)
	col = nbPtFullEcho - row + 1
	echos2D = echos2D.astype('complex64')		# decrease SVD computation time
	echos2D_rec = np.empty([nbFullEchoTotal, nbPtFullEcho],dtype='complex64')
	
	print("SVD on Toeplitz matrix of echoes in progress. Please be patient.")
	t_0 = time()
	
	for i in range (0, nbFullEchoTotal):
		# decomposition
		T = linalg.toeplitz(echos2D[i][row-1::-1], echos2D[i][row-1::1])
		U, s, Vh = linalg.svd(T[:][:], full_matrices=False)
		S = linalg.diagsvd(s[:], row, col)
		
		# reconstruction
		T_rec = dot(U[:,:thres], dot(S[:thres,:thres], Vh[:thres,:]))
		for j in range (0, nbPtFullEcho):
			echos2D_rec[i][j] = np.mean(np.diag(T_rec[:][:],j-row+1))
	
	t_2 = time()
	print("Decomposition + Reconstruction time:\t\t{0:8.2f}s".format(t_2 - t_0))
	
	ax2 = fig3.add_subplot(412)
	ax2.set_title("FID after SVD on Toeplitz matrix of echoes")
	for i in range (0, nbFullEchoTotal):
		ax2.plot(timeFullEcho[:],(echos2D_rec[i][:]).real)
	
	echos2D = echos2D_rec[:][:].astype('complex')	# back to double precision

# prediction lineaire


# somme ponderee -> calcul de Imax
Imax = np.empty([nbFullEchoTotal])
for i in range (0, nbFullEchoTotal):
	#Imax = np.amax((echos2D[i][:]).real)
	#echos2D[i][0:nbPtFullEcho]*=Imax
	Imax[i] = np.amax((echos2D[i][:]).real)


# correction T2


# fftshift => inversion des halfEcho 2 à 2
echosFFTSHIFT = np.fft.fftshift(echos2D[0:nbFullEchoTotal][0:nbPtFullEcho],axes=1)
echosFFTSHIFT[0][0]*=0.5		# permet de corriger l'artefact due à la FFT
echosFFT = np.fft.fftshift(np.fft.fft(echosFFTSHIFT[:][:],axis=1),axes=1)

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
fig3.show()			# affiche la figure a l'ecran


print("\n------------------------------------------------------------------------\n\n")
