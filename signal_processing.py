#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@authors: Pierre-Aymeric GILLES & Guillaume LAURENT
"""

# Python libraries
import matplotlib.pyplot as plt
import numpy as np
# User defined libraries
import signalTopspin as sig
import svd_sfa as svd
svd.svd_init()



def signal_processing(processedSig):

	#%%-------------------------------------------------------------------------
	### PARAMETERS
	###-------------------------------------------------------------------------

	# Singular Value Decomposition
	# 0 : no SVD applied
	# Method 1: on full 1D with echoes --> very long
	# Method 2: on full 2D of stacked echoes --> very fast
	# Method 3: on separated echoes --> fast

#	if nbPtSignal <= 8192:
#		SVD_method = 1
#	else:
#		SVD_method = 2
#	thres = 16
	
	SVD_method = 2

	# Asked to the user
	firstDec = processedSig.firstDec
	fullEcho = processedSig.fullEcho
	nbEcho = processedSig.nbEcho

	# From Topspin interface
	td = processedSig.td					# nb de pts complexes  ( td == td/2 )
	dw = processedSig.dw					# Dwell time between two points
	de = processedSig.de					# temps de non-acquisition au début

	# Calculated
	halfEcho = processedSig.halfEcho
	nbHalfEcho = processedSig.nbHalfEcho
	nbFullEchoTotal = int((nbHalfEcho+1)/2)
	td2 = processedSig.td2					# nb de pts complexes  ( td == td/2 )
	dw2 = processedSig.dw2
	acquiT = processedSig.acquiT	# temps acquisition total : acquiT = (td-1)*dw2

	lb = 3/(np.pi*halfEcho)				# line broadening (Herz)
	dureeSignal = processedSig.dureeSignal
	nbPtHalfEcho = processedSig.nbPtHalfEcho
	nbPtFullEcho = 2*nbPtHalfEcho
	nbPtSignal = processedSig.nbPtSignal
	missingPts = processedSig.missingPts
	nbPtDeadTime = processedSig.nbPtDeadTime

	# Axes
	timeT = np.linspace(0,acquiT,td2)
	timeFullEcho = timeT[:nbPtFullEcho]
	nbPtFreq = int(td2*4)			# zero filling
	freq = np.linspace(-1/(2*dw2), 1/(2*dw2), nbPtFreq)
	freqFullEcho = np.linspace(-1/(2*dw2), 1/(2*dw2), nbPtFullEcho)


	###----------------------------------------------------------------------------
	### AFFICHAGE DES VALEURS DES PARAMETRES (RETOUR UTILISATEUR)
	###----------------------------------------------------------------------------

	print("\n------------------------------------------------------------------------")
	print('File : signal_processing.py')
	print("------------------------------------------------------------------------\n")
	print("\nSYNTHESE DES VALEURS :")
	print("\n Valeurs demandées à l'utilisateur :\n")
	print("\tfirstDec =", firstDec)
	print("\tfullEcho = {:8.2e}".format(fullEcho))
	print("\thalfEcho = {:8.2e}".format(halfEcho))
	print("\tnbEcho =", nbEcho)
	print("\tnbHalfEcho =", nbHalfEcho, "(calculée : dépend de 1ere decroissance ou non)")

	print("\n Valeurs passées en paramètres :\n")
	print("\tdw = {:8.2e}".format(dw))
	print("\tnbPt = ", td)
	print("\tnbPt complex=", td2)
	print("\tacquiT = {:8.2e}".format(acquiT))
	print("\tde = {:8.2e}".format(de))

	print("\n Valeurs calculées :\n")
	print("\tdureeSignal = {:8.2e}".format(dureeSignal))
	#print("\tnbPtSignal_via_dureeSignal =", nbPtSignal_via_dureeSignal)
	#print("\tnbPtSignal_via_nbPtHalfEcho =", nbPtSignal)
	print("\tnbPtSignal =", nbPtSignal)
	print("\tmissingPts =", missingPts)
	print("\tnbPtDeadTime =", nbPtDeadTime)


	print("\nSpecified SVD method :", SVD_method)



	#%%----------------------------------------------------------------------------
	### Preprocessing
	###----------------------------------------------------------------------------

	A = processedSig.data
	Araw = A.copy()

	# ajout de points à zero pour compenser le dead time
	zerosToAdd = np.zeros((nbPtDeadTime), dtype=np.complex)
	A = np.concatenate((zerosToAdd[:],A[:td2-nbPtDeadTime]))
#	print("A.size =", A.size)

	# on supprime les points en trop
	echos1D = A[:nbPtSignal]
#	print("echos1D.size =",echos1D.size)

	# preprocessing of echoes
	print("preprocessing...")
	desc = firstDec

	for i in range (0, nbHalfEcho):
		if (desc==True):
			timei = np.linspace(0,halfEcho-dw2,nbPtHalfEcho)
		else:
			timei = np.linspace(-halfEcho,-dw2,nbPtHalfEcho)
		echos1D[i*nbPtHalfEcho:(i+1)*nbPtHalfEcho] *= np.exp(-abs(timei[:])*np.pi*lb)
		desc = not(desc)
	
	# For plotting
	echos1Dpreproc = echos1D.copy()


	#%%----------------------------------------------------------------------------
	# SVD and echoes separation
	###----------------------------------------------------------------------------

	# Singular Value Decompostion (SVD) on Toeplitz matrix
	if (SVD_method == 1):
		echos1D = svd.svd(echos1D,SVD_method)



	# separation des echos
	# si 1ere decroissance : on inclut un demi echo de 0 devant 
	if firstDec:
		firstHalfEcho = np.zeros((nbPtHalfEcho), dtype=np.complex)
		echos1D = np.concatenate((firstHalfEcho[:],echos1D[:]))
#	print("echos1D.size =", echos1D.size)

	# separation
#	print("\n 1er elem de chaque demi echo à la separation (reshape) des echos")
	echos2D = echos1D.reshape(nbFullEchoTotal,nbPtFullEcho)



	# Singular Value Decompostion (SVD) on echo matrix
	if (SVD_method == 2):
		echos2D = svd.svd(echos2D,SVD_method)

	# Singular Value Decompostion (SVD) on Toeplitz matrix of each echo
	if (SVD_method == 3):
		echos2D = svd.svd(echos2D,SVD_method)
	
	# For plotting
	echos2Dsvd = echos2D.copy()


	#%%----------------------------------------------------------------------------
	# Echoes ponderation
	###----------------------------------------------------------------------------

	# prediction lineaire



#	# somme des echos (temporelle) et affichage 
#	# Rq : pas utilisé pour la suite -> juste pour voir ce que donne la somme temporelle
#	sommeTempo = np.zeros((nbPtFullEcho,nbFullEchoTotal), dtype=np.complex)
#	for i in range (0, nbFullEchoTotal):
#		for j in range (0, nbPtFullEcho):
#			sommeTempo[j] += echos2D[i, j]
#	plt.figure()
#	plt.plot(timeT[:nbPtFullEcho],sommeTempo[:nbPtFullEcho].real)
#	plt.show() # affiche la figure a l'ecran



	# somme ponderee -> calcul de Imax
	Imax = np.empty([nbFullEchoTotal])
	for i in range (0, nbFullEchoTotal):
		#Imax = np.amax((echos2D[i, :]).real)
		#echos2D[i, 0:nbPtFullEcho]*=Imax
		Imax[i] = max(np.absolute(echos2D[i,:]))



#	# fftshift => inversion des halfEcho 2 à 2
#	echosFFTSHIFT = np.fft.fftshift(echos2D[0:nbFullEchoTotal, 0:nbPtFullEcho],axes=1)
#	echosFFTSHIFT[0, 0]*=0.5		# permet de corriger l'artefact due à la FFT
#	echosFFT = np.fft.fftshift(np.fft.fft(echosFFTSHIFT[:,:],axis=1),axes=1)



	# mesure et correction de T2



	# somme pondérée
	for i in range (0, nbFullEchoTotal):
		echos2D[i, :]*= Imax[i]



	# somme des echos (spectrale)
	sommeFID = np.zeros((nbPtFullEcho), dtype=np.complex)
	for i in range (0, nbPtFullEcho):
		sommeFID[i] = sum(echos2D[:,i])



	#%% Spectra calculation
	ArawSPC = Araw.copy()
	ArawSPC[0] *= 0.5					# FFT artefact correction
	ArawSPC = np.fft.fftshift(np.fft.fft(ArawSPC[:], nbPtFreq))

	echos1DpreprocSPC = echos1Dpreproc.copy()
	echos1DpreprocSPC[0] *= 0.5	# FFT artefact correction
	echos1DpreprocSPC = np.fft.fftshift(np.fft.fft(echos1DpreprocSPC[:], nbPtFreq))
	
	echos2DsvdSPC = echos2Dsvd.copy()
	echos2DsvdSPC = np.fft.fftshift(echos2DsvdSPC, axes=1) 	# to avoid phasing
	echos2DsvdSPC[:,0] *= 0.5		# FFT artefact correction
	echos2DsvdSPC = np.fft.fftshift(np.fft.fft(echos2DsvdSPC[:,:], \
		axis=1), axes=1)				# no zero-filling on full echo

	sommeFIDSPC = sommeFID.copy()
	sommeFIDSPC = np.fft.fftshift(sommeFIDSPC) 		# to avoid phasing
	sommeFIDSPC *= 0.5				# FFT artefact correction
	sommeFIDSPC = np.fft.fftshift(np.fft.fft(sommeFIDSPC[:]))
											# no zero-filling on full echo



	#%% Figures
	plt.ion()							# interactive mode on
	
	# Temporal data (FID)
	fig1 = plt.figure()
	fig1.suptitle("CPMG NMR signal processing", fontsize=16)
	
	ax1 = fig1.add_subplot(411)
	ax1.set_title("Raw FID")
	ax1.set_xlim([-acquiT*0.05, acquiT*1.05])
	ax1.plot(timeT[:],Araw[:].real)
	ax1.plot(timeT[:],Araw[:].imag)
	
	ax2 = fig1.add_subplot(412)
	ax2.set_title("FID after preprocessing")
	ax2.set_xlim([-acquiT*0.05, acquiT*1.05])
	ax2.plot(timeT[:nbPtSignal],echos1Dpreproc[:].real)
	ax2.plot(timeT[:nbPtSignal],echos1Dpreproc[:].imag)
	
	ax3 = fig1.add_subplot(413)
	ax3.set_title("FID after SVD and echo separation")
	for i in range (0, nbFullEchoTotal,5):
		ax3.plot(timeFullEcho[:],(echos2Dsvd[i,:]).real)
#		print("\t1er elem du demi echo", 2*i ," =", echos2D[i, 0])
#		print("\t1er elem du demi echo", 2*i+1 ," =", echos2D[i, nbPtHalfEcho])
	
	ax4 = fig1.add_subplot(414)
	ax4.set_title("FID after ponderated sum")
	ax4.plot(timeFullEcho[:],sommeFID[:].real)
#	ax4.plot(timeFullEcho[:],sommeFID[:].imag)
	
	fig1.tight_layout(rect=[0, 0, 1, 0.95])			# Avoid superpositions on display
	fig1.show()					# Display figure



	# Spectral data (SPC)
	fig2 = plt.figure()
	fig2.suptitle("CPMG NMR signal processing", fontsize=16)
	
	ax1 = fig2.add_subplot(411)
	ax1.set_title("Raw SPC")
	ax1.plot(freq[:],ArawSPC[:].real)
#	ax1.plot(freq[:],Araw[:].imag)
	ax1.invert_xaxis()
	
	ax2 = fig2.add_subplot(412)
	ax2.set_title("SCP after preprocessing")
	ax2.plot(freq[:],echos1DpreprocSPC[:].real)
#	ax2.plot(freq[:],echos1Dpreproc[:].imag)
	ax2.invert_xaxis()
	
	ax3 = fig2.add_subplot(413)
	ax3.set_title("SPC after SVD and echo separation")
	for i in range (0, nbFullEchoTotal,5):
		ax3.plot(freqFullEcho[:],(echos2DsvdSPC[i,:]).real)
#		print("\t1er elem du demi echo", 2*i ," =", echos2D[i, 0])
#		print("\t1er elem du demi echo", 2*i+1 ," =", echos2D[i, nbPtHalfEcho])
	ax3.invert_xaxis()
	
	ax4 = fig2.add_subplot(414)
	ax4.set_title("SPC after ponderated sum")
	ax4.plot(freqFullEcho[:],sommeFIDSPC[:].real)
#	ax4.plot(freqFullEcho[:],sommeFID[:].imag)
	ax4.invert_xaxis()
	
	fig2.tight_layout(rect=[0, 0, 1, 0.95])			# Avoid superpositions on display
	fig2.show()					# Display figure



	return sommeFID

#%%----------------------------------------------------------------------------
### When this file is executed directly
###----------------------------------------------------------------------------

if __name__ == "__main__":
	import sys
	
#	print('len(sys.argv)', len(sys.argv))

	if len(sys.argv) == 1:
		A_data = np.genfromtxt('./CPMG_FID.csv',delimiter='\t', skip_header=1)
		A_data = A_data[:,0] + 1j*A_data[:,1]
	
		A_firstDec = True
		A_fullEcho = 10e-3
		A_nbEcho = 20				# 38 for less than 8k points, 76 for more
		
		A_td = 32768				# nb of real points + nb of imag points
		A_dw = 24e-6				# dwell time between two points
		A_de = 96e-6				# dead time before signal acquisition
		
		# Saving data to Signal class
		rawFID = sig.Signal()
		rawFID.setValues_topspin(A_td,A_dw,A_de)
		rawFID.setValues_user(A_firstDec,A_fullEcho,A_nbEcho)
		rawFID.setData(A_data)
	else:
		print("Additional arguments are not yet supported")
	
	processed = signal_processing(rawFID)



	input('\nPress enter key to exit') # have the graphs stay displayed even when launched from linux terminal