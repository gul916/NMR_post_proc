#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@authors: Pierre-Aymeric GILLES & Guillaume LAURENT
"""

# Python libraries
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from time import time



###----------------------------------------------------------------------------
### PARAMETERS
###----------------------------------------------------------------------------

# svd_tools_resolution_override  =>  default value of override for init()
# change it to change the default value of override in init()
## 0 : default choice (skcuda if available / scipy if not)
## 1 : force arrayfire use
## 2 : force skcuda use
## 3 : force scipy use
svd_tools_resolution_override = 0

# svdMethod :#
# 0 : no Singular Value Decomposion (SVD)
# 1 : SVD applied on 1D data converted to Toeplitz matrix --> very long
# 2 : SVD applied on 2D data --> very fast
# 3 : SVD applied on slices of 2D data converted to Toeplitz matrix --> fast
svdMethod = 1

#Initialisation
arrayfireOK = False
skcudaOK = False
scipyOK = False
initCalled = False

modulesCheck = [svd_tools_resolution_override,arrayfireOK,skcudaOK,scipyOK,initCalled]



###----------------------------------------------------------------------------
### SVD tools importation
###----------------------------------------------------------------------------

def import_arrayfire():
	arrayfireLoad = False
	try:
		print('\nLoading module arrayfire ...')
		global af
		import arrayfire as af
	except ImportError:
		print('Error while loading module : arrayfire')
	else:
		arrayfireLoad = True
		print('Module arrayfire loaded successfully')
	return arrayfireLoad



def import_skcuda():
	skcudaLoad = False
	try:
		print('\nLoading module skcuda ...')
		global gpuarray, culinalg, cudriver
		import pycuda.autoinit			# needed
		import pycuda.driver as cudriver
		import pycuda.gpuarray as gpuarray
		import skcuda.linalg as culinalg
	except ImportError:
		print('Error while loading module : skcuda')
	else:
		skcudaLoad = True
		print('Module skcuda loaded successfully')
	return skcudaLoad



def import_scipy():
	scipyLoad = False
	try:
		print('\nLoading module scipy ...')
		global dot, linalg
		from scipy import dot, linalg
	except ImportError:
		print('Error while loading module : scipy')
	else:
		scipyLoad = True
		print('Module scipy loaded successfully')
	return scipyLoad



###----------------------------------------------------------------------------
### SVD tools resolution
###----------------------------------------------------------------------------

def svd_init(override=svd_tools_resolution_override):

	arrayfireOK = False
	skcudaOK = False
	scipyOK = False

	print("\n------------------------------------------------------------------------")
	print('File : svd_sfa.py')
	print("------------------------------------------------------------------------\n")

	print('Establishing modules to import / Importing')

	if override == 0:
		# default choice

		arrayfireOK = import_arrayfire()

		if (not arrayfireOK):
			skcudaOK = import_skcuda()

			if (not skcudaOK):
				scipyOK = import_scipy()

	elif override == 1:
		print('\noverride == 1  =>')
		print('  Forcing load of module arrayfire')
		arrayfireOK = import_arrayfire()

	elif override == 2:
		print('\noverride == 2  =>')
		print('  Forcing load of module skcuda')
		skcudaOK = import_skcuda()

	elif override == 3:
		print('\noverride == 3  =>')
		print('  Forcing load of module scipy')
		scipyOK = import_scipy()

	else:
		print("\nInvalid value specified for override (", \
			override,"). Correct values : 0 1 2 3")

	initCalled = True

	modulesCheck[0] = override
	modulesCheck[1] = arrayfireOK
	modulesCheck[2] = skcudaOK
	modulesCheck[3] = scipyOK
	modulesCheck[4] = initCalled



def svd_tools_resolution():

	override = modulesCheck[0]
	arrayfireOK = modulesCheck[1]
	skcudaOK = modulesCheck[2]
	scipyOK = modulesCheck[3]
	initCalled = modulesCheck[4]

	print ("\nSVD tools resolution :")
	# print("override : ",override)
	# print("arrayfireOK : ",arrayfireOK)
	# print("skcudaOK : ",skcudaOK)
	# print("scipyOK : ",scipyOK)
	if not initCalled:
		raise ImportError("svd_init() function not called")
	if override == 0:
		if arrayfireOK:
			choice = 'arrayfire'
			print("Using methods from module : arrayfire")
		elif skcudaOK:
			choice = 'skcuda'
			print("Using methods from module : skcuda")
		elif scipyOK:
			choice = 'scipy'
			print("Using methods from module : scipy")
		else:
			raise ImportError('No SVD module available.')
	elif override == 1:
		if arrayfireOK:
			choice = 'arrayfire'
			print("Using methods from module : arrayfire")
		else:
			raise ImportError('Selected SVD module (arrayfire) not found.')
	elif override == 2:
		if skcudaOK:
			choice = 'skcuda'
			print("Using methods from module : skcuda")
		else:
			raise ImportError('Selected SVD module (skcuda) not found.')
	elif override == 3:
		if scipyOK:
			choice = 'scipy'
			print("Using methods from module : scipy")
		else:
			raise ImportError('Selected SVD module (scipy) not found.')
	else:
		errorMsg = "Specified override value incorrect ("
		errorMsg += str(override)
		errorMsg += "). Correct values : 0 1 2 3"
		raise ImportError(errorMsg)
		

	return choice



def svd_preliminary_operations(setSVDTools):
	if setSVDTools == 'arrayfire':
		#	if (row*col < 2048*2048):
		#		af.set_backend('cpu')
		#	else:
		#		af.set_backend('unified')		# Bug :Anormally slow
		#	# see https://github.com/arrayfire/arrayfire-python/issues/134
		af.set_backend('cpu')
		print("\nUsing arrayfire version :")
		print(af.info_str())
	elif setSVDTools == 'skcuda':
		culinalg.init()



###----------------------------------------------------------------------------
### svd_decomposition
###----------------------------------------------------------------------------

def svd_decomposition(mat, choice):
	'''
	U, s_gpu, s_cpu, Vh = svd_decomposition(mat, choice)
	
	U: left matrix
	s_gpu: singular values in 1D array
	s_cpu: copy of s to be used on cpu
	Vh: hermitian transpose of right matrix
	choice: svd frontend used
	'''
	if choice=='arrayfire':
		af.device_gc()		# clear memory
		mat_gpu = af.to_array(mat[:,:])
		U, s_gpu, Vh = af.svd(mat_gpu[:,:])
		s_cpu = np.array(s_gpu)

	elif choice=='skcuda':
		cudriver.pagelocked_empty()		# clear memory
		mat_gpu = gpuarray.to_gpu(mat[:,:])
		U, s_gpu, Vh = culinalg.svd(mat_gpu[:,:])
		s_cpu = s_gpu.get()

	elif choice=='scipy':
		U, s_cpu, Vh = linalg.svd(mat[:,:], full_matrices=False)
		s_gpu = s_cpu	# scipy not using gpu => s_cpu and s_gpu are one and the same

	else:
		print("Unknown svd tools")

	return U, s_gpu, s_cpu, Vh



###----------------------------------------------------------------------------
### THRESHOLDING METHODS
###----------------------------------------------------------------------------

## Indicator function IND
def indMethod(s,m,n):

	# preallocating
	ev = np.zeros(n)
	df = np.zeros(n)
	rev = np.zeros(n)
	sev = np.zeros(n-1)
	sdf = np.zeros(n-1)
	re = np.zeros(n-1)
	ind = np.zeros(n-1)

	for j in range (0, n):
		ev[j] = s[j]**2
		df[j] = (m-j)*(n-j)
		rev[j] = ev[j] / df[j]

	for k in range (0, n-1):
		sev[k] = np.sum(ev[k+1:n])
		sdf[k] = np.sum(df[k+1:n])

	for i in range (0, n-1):
		re[i] = np.sqrt(sev[i] / (m * (n-i-1)))		# see eq. 4.44
		ind[i] = re[i] / (n-i-1)**2					# see eq. 4.63

	nval = np.argmin(ind)
#	print('\nnval\n', nval)
#	print('\nind[nval]\n', ind[nval])
#	print('\nind[nval-2:nval+3]\n', ind[nval-2:nval+3])
	
	null = np.array([None])
	re = np.concatenate((re[:],null[:]))
	ind = np.concatenate((ind[:],null[:]))

	t = np.zeros((n,6))

	for j in range (0, n):
		t[j,0] = j
		t[j,1] = ev[j]
		t[j,2] = re[j]
		t[j,3] = ind[j]
		t[j,4] = rev[j]

	return (nval,sdf,ev,sev,t)



## Significance Level SL
def slMethod(s,m,n,max_err):
	nval,sdf,ev,sev,t = indMethod(s,m,n)
	for j in range (0, n-1):
		f = (sdf[j] * ev[j]) / ((m-j) * (n-j) * sev[j])
		# convert f (see eq. 4.83) into percent significance level
		if j < n:
			tt = np.sqrt(f)
			df = n - j - 1
			a = tt / np.sqrt(df)
			b = df / (df + tt * tt)
			im = df - 2
			jm = df - 2 * np.fix(df / 2)
			ss = 1
			cc = 1
			ks = int(2 + jm)
			fk = ks
			if (im - 2) >= 0:
				for k in range(ks,(im+2),2):
					cc = cc * b * (fk-1) / fk
					ss = ss + cc
					fk = fk + 2
			if (df - 1) > 0:
				c1 = .5 + (a * b * ss + np.arctan(a)) * .31831
			else:
				c1 = .5 + np.arctan(a) * .31831
			if jm <= 0:
				c1 = .5 + .5 *a * np.sqrt(b) * ss
		s1 = 100 * (1 - c1)
		s1 = 2 * s1
		if s1 < 1e-2:
			s1 = 0
		t[j, 5] = s1
	t[n-1, 5] = None

	nval = np.argmax((t[:n-1,5]) > max_err) - 1
#	print('\nnval\n', nval)
#	print('\nt[nval,5]\n', t[nval,5])
#	print('\nt[nval-2:nval+3,5]\n', t[nval-2:nval+3,5])

	return nval



###----------------------------------------------------------------------------
### svd_reconstruction
###----------------------------------------------------------------------------

def svd_reconstruction(U, s_gpu, Vh, thres, choice):
	'''
	mat_rec = svd_reconstruction(U, s_gpu, Vh, thres, choice)
	
	mat_rec: reconstructed matrix
	U: left matrix
	s_gpu: copy of s to be used on cpu
	Vh: hermitian transpose of right matrix
	thres: number of singular values keeped
	choice: svd frontend used
	'''
	thres += 1				# to include last point
	if choice=='arrayfire':
		S_gpu = af.diag(s_gpu[:thres], 0, False).as_type(af.Dtype.c32)
		mat_rec_gpu = af.matmul(U[:,:thres], \
			af.matmul(S_gpu[:thres,:thres], Vh[:thres,:]))
		mat_rec = np.array(mat_rec_gpu[:,:])
		af.device_gc()		# clear memory

	elif choice=='skcuda':
		S_gpu = gpuarray.zeros((thres,thres), np.complex64)
		S_gpu = culinalg.diag(s_gpu[:thres]).astype(np.complex64)
		mat_rec_gpu = culinalg.dot(U[:,:thres], \
			culinalg.dot(S_gpu[:,:], Vh[:thres,:]))
		mat_rec = mat_rec_gpu.get()
		cudriver.pagelocked_empty()		# clear memory

	elif choice=='scipy':
		S = linalg.diagsvd(s_gpu[:thres], thres, thres)
		mat_rec = dot(U[:,:thres], dot(S[:thres,:thres], Vh[:thres,:]))

	else:
		print("Unknown svd tools")

	return mat_rec



###----------------------------------------------------------------------------
### SVD THRESHOLD
###----------------------------------------------------------------------------

def svd_thres(data,svdTools,thresMethod='SL',max_err=5):
	'''
	svd_autoThres 	significant factor analysis - a program designed
					to help determine the number of significant factors in a matrix.
	
	Factor Analysis in Chemistry, Third Edition, p387-389
	Edmund R. Malinowki
	'''

	# data : data matrix
	# thresMethod : 
		# IND : Indicator Function
		# SL : Significance Level
	# max_err : probability of being noise for SL
	# denData : denoised data matrix

	denData = data
	nval = None

	# transpose if needed
	[n,m] = np.shape(data)
	# print("m :",m)
	# print("n :",n)
	
	transp = 0
	
	if(m<n):
		#print("data transpose...")
		data = data.transpose()
		transp = 1
		[n,m] = np.shape(data)
		# print("m :",m)
		# print("n :",n)

	# svd decomposition
	u, sgpu, scpu, v = svd_decomposition(data,svdTools)

	# thresholding
	if (thresMethod == 'IND'):
		nval = indMethod(scpu,m,n)[0]
	else:
		nval = slMethod(scpu,m,n,max_err)
#	print('\nnval\n', nval)

	if nval < 0:
		raise ValueError

	# reconstruction
	denData = svd_reconstruction(u,sgpu,v,nval,svdTools)

	# transpose back if needed
	if transp == 1:
		denData = denData.transpose()

	# [m2,n2] = np.shape(denData)
	# print("m2 :",m2)
	# print("n2 :",n2)

	return denData, nval



###----------------------------------------------------------------------------
### MAIN SVD METHOD
###----------------------------------------------------------------------------

# svdMethod :#
# 0 : no Singular Value Decomposion (SVD)
# 1 : SVD applied on 1D data converted to Toeplitz matrix --> very long
# 2 : SVD applied on 2D data --> very fast
# 3 : SVD applied on slices of 2D data converted to Toeplitz matrix --> fast

def svd(data,svdMethod=0,thresMethod='SL',max_err=5):
	# default values
	## returned if for some reason svd isn't performed
	processedData = data
	
	if svdMethod == 0:
		print("\nSVD method = ",svdMethod," SVD not performed.")
	elif svdMethod not in [1,2,3]:
		print("\nInvalid SVD method specified (",svdMethod,"). SVD not performed.")
	elif (data.ndim == 1) and (svdMethod in [2,3]):
		print("\nSVD method ",svdMethod," can't be applied to 1D data. SVD not performed.")
	elif (data.ndim == 2) and (svdMethod == 1):
		print("\nSVD method ",svdMethod," can't be applied to 2D data. SVD not performed.")
	elif (data.ndim > 2):
		print("\nSVD not yet tested on high dimensionnal data. SVD not performed.")
	else:
		try:
			print("\n------------------------------------------------------------------------")
			svdTools = svd_tools_resolution()
		except ImportError as err:
			print ("\nsvd_tools_resolution returns error :")
			print("  ",err.args[0])
			print("\nSVD unavailable. Resuming signal processing.")
		else:
			svd_preliminary_operations(svdTools)
			print()
			print("thresMethod :",thresMethod)
			print("max_err :",max_err)
			print("svdMethod :",svdMethod)
			print("\n------------------------------------------------------------------------\n")
			
			try:
				t_0 = time()
				data64 = data.astype('complex64')		# decrease SVD computation time

				if svdMethod == 1:
					# Singular Value Decompostion (SVD) on Toeplitz matrix
					nbPtSignal = data64.size
					row = math.ceil(nbPtSignal / 2)
#					col = nbPtSignal - row + 1

					mat = linalg.toeplitz(data64[row-1::-1], data64[row-1::1])
					data_rec = np.empty([nbPtSignal],dtype='complex64')
					
					print("SVD on 1D data converted to Toeplitz matrix in progress.")
					print("Please be patient.")
					mat_rec, thres = svd_thres(mat,svdTools,thresMethod,max_err)
					for i in range (0, nbPtSignal):
						data_rec[i] = np.mean(np.diag(mat_rec[:,:],i-row+1))

					print("number of singular values = ",thres+1)
					
				elif svdMethod == 2:
					# Singular Value Decompostion (SVD) on echo matrix
					print("SVD on 2D data in progress. Please be patient.")

					data_rec, thres = svd_thres(data64,svdTools,thresMethod,max_err)

					print("number of singular values = ",thres+1)

				elif svdMethod == 3:
					# Singular Value Decompostion (SVD) on Toeplitz matrix of each echo
					print("SVD on slices of 2D data converted to Toeplitz matrix in progress.")
					print("Please be patient.")

					nslices, nbPtSlice = data64.shape
					row = math.ceil(nbPtSlice / 2)
#					col = nbPtSlice - row + 1

					data_rec = np.empty([nslices, nbPtSlice],dtype='complex64')
					
					for i in range (0, nslices):
						mat = linalg.toeplitz(data64[i,row-1::-1], data64[i,row-1::1])
						mat_rec, thres = svd_thres(mat,svdTools,thresMethod,max_err)
						print("number of singular values = ",thres+1)
						for j in range (0, nbPtSlice):
							data_rec[i,j] = np.mean(np.diag(mat_rec[:,:],j-row+1))
				
				processedData = data_rec.astype('complex128')	# back to double precision
				t_1 = time()
				print("Decomposition + Reconstruction time:\t\t{0:8.2f}s".format(t_1 - t_0))
				print("\n------------------------------------------------------------------------\n")
			
			except ValueError:	# nval < 0 dans svd_thres
				print("No singular value detected, aborting SVD")
				print("\n------------------------------------------------------------------------\n")
				processedData = data

	return processedData



###----------------------------------------------------------------------------
### When this file is executed directly
###----------------------------------------------------------------------------

if __name__ == "__main__":
	
	svd_init(svd_tools_resolution_override)

	A = np.genfromtxt('./CPMG_FID.csv',delimiter='\t', skip_header=1)
	nbPtHalfEcho=104
	nbHalfEcho=41
	nbFullEchoTotal = int((nbHalfEcho+1)/2)
	
	Amax = int((nbHalfEcho+1) * nbPtHalfEcho)
	nbPtFreq = int(Amax * 4)
	
	
	
	# Raw data
	A = A[:Amax,0] + 1j*A[:Amax,1]
	ASPC = A[:nbPtHalfEcho]
	ASPC[0] *= 0.5				# FFT artefact correction
	ASPC = np.fft.fftshift(np.fft.fft(ASPC[:], nbPtFreq))

	plt.ion()					# interactive mode on
	fig1 = plt.figure()
	fig1.suptitle("SVD processing", fontsize=16)

	ax1 = fig1.add_subplot(411)
	ax1.set_title("Raw FID")
	ax1.plot(A[:].real)
	
	fig2 = plt.figure()
	fig2.suptitle("SVD processing", fontsize=16)

	ax1 = fig2.add_subplot(411)
	ax1.set_title("Raw SPC")
	ax1.invert_xaxis()
	ax1.plot(ASPC[:].real)
	
	
	
	# 1 : SVD applied on 1D data converted to Toeplitz matrix --> very long
	svdMethod = 1
	newA = svd(A,svdMethod)
	
	newASPC = newA[:nbPtHalfEcho]
	newASPC[0] *= 0.5				# FFT artefact correction
	newASPC = np.fft.fftshift(np.fft.fft(newASPC[:], nbPtFreq))

	ax2 = fig1.add_subplot(412)
	ax2.set_title("Denoised FID with method 1")
	ax2.plot(newA[:].real)
	
	ax2 = fig2.add_subplot(412)
	ax2.set_title("Denoised SPC with method 1")
	ax2.invert_xaxis()
	ax2.plot(newASPC[:].real)
	
	
	
	# Conversion to 2D
	firstHalfEcho = np.zeros((nbPtHalfEcho), dtype=np.complex)
#	firstHalfEcho = A[nbPtHalfEcho:0:-1].real -1j*A[nbPtHalfEcho:0:-1].imag
	A = np.concatenate((firstHalfEcho[:],A[:Amax-nbPtHalfEcho]))
	A = A.reshape(nbFullEchoTotal, 2*nbPtHalfEcho)
	
	
	
	# 2 : SVD applied on 2D data --> very fast
	svdMethod = 2
	newA = svd(A,svdMethod)
	
	newASPC = newA[:,nbPtHalfEcho:2*nbPtHalfEcho]
	newASPC[:,0] *= 0.5				# FFT artefact correction
	newASPC = np.fft.fftshift(np.fft.fft(newASPC[:,:], nbPtFreq))

	ax3 = fig1.add_subplot(413)
	ax3.set_title("Denoised FID with method 2")
	for i in range (0,nbFullEchoTotal,5):
		ax3.plot(newA[i,:].real)
	
	ax3 = fig2.add_subplot(413)
	ax3.set_title("Denoised SPC with method 2")
	ax3.invert_xaxis()
	for i in range (0,nbFullEchoTotal,5):
		ax3.plot(newASPC[i,:].real)
	
	
	
	# 3 : SVD applied on slices of 2D data converted to Toeplitz matrix --> fast
	svdMethod = 3
	newA = svd(A,svdMethod)
	
	newASPC = newA[:,nbPtHalfEcho:2*nbPtHalfEcho]
	newASPC[:,0] *= 0.5				# FFT artefact correction
	newASPC = np.fft.fftshift(np.fft.fft(newASPC[:,:], nbPtFreq))

	ax4 = fig1.add_subplot(414)
	ax4.set_title("Denoised FID with method 3")
	for i in range (0,nbFullEchoTotal,5):
		ax4.plot(newA[i,:].real)
	
	ax4 = fig2.add_subplot(414)
	ax4.set_title("Denoised SPC with method 3")
	ax4.invert_xaxis()
	for i in range (0,nbFullEchoTotal,5):
		ax4.plot(newASPC[i,:].real)



		# Display figures
	fig1.tight_layout(rect=[0, 0, 1, 0.95])		# Avoid superpositions on display
	fig1.show()												# Display figure
	
	fig2.tight_layout(rect=[0, 0, 1, 0.95])		# Avoid superpositions on display
	fig2.show()												# Display figure



	input('\nPress enter key to exit') # have the graphs stay displayed even when launched from linux terminal