#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import csv
from scipy import linalg
from time import time


# svd_tools_resolution_override  =>  default value of override for init()
# change it to change the default value of override in init()
## 0 : default choice (skcuda if available / scipy if not)
## 1 : force arrayfire use
## 2 : force skcuda use
## 3 : force scipy use
svd_tools_resolution_override = 0

arrayfireOK = False
skcudaOK = False
scipyOK = False
initCalled = False

modulesCheck = [svd_tools_resolution_override,arrayfireOK,skcudaOK,scipyOK,initCalled]


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
		import pycuda.autoinit
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


def init(override=svd_tools_resolution_override):

	arrayfireOK = False
	skcudaOK = False
	scipyOK = False

	print("\n------------------------------------------------------------------------")
	print('File : svd_sfa.py')
	print("------------------------------------------------------------------------\n")

	print('Establishing modules to import / Importing')

	if override == 1:
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

	elif override == 0:
		# default choice

		arrayfireOK = import_arrayfire()

		if (not arrayfireOK):
			skcudaOK = import_skcuda()

			if (not skcudaOK):
				scipyOK = import_scipy()

	else:
		print("\nInvalid value specified for override ("\
			,override,"). Correct values : 0 1 2 3")

	initCalled = True

	modulesCheck[0] = override
	modulesCheck[1] = arrayfireOK
	modulesCheck[2] = skcudaOK
	modulesCheck[3] = scipyOK
	modulesCheck[4] = initCalled



###----------------------------------------------------------------------------
### SVD METHODS (WIP)
###----------------------------------------------------------------------------


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
		raise ImportError("init() function not called")
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
	if override == 1:
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
		af.info()
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


	nval = np.argmin(ind)+1

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

	# disp(['IND function indicates ', int2str(nval), ' significant factors.'])
	# disp(['The real error (RE) is +/-', num2str(re(nval)), '.'])


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
		t[j][5] = s1
	t[n-1][5] = None

	nval = (np.nonzero((t[:,5]) < max_err))[0]
	if (nval.size==0):
		nval = 0
	else:
		nval = nval[-1]+1
	
	return nval

	# disp(['SL function indicates ', int2str(nval), ' significant factors.'])
	# if nval ~= 0
	# disp(['The real error (RE) is +/-', num2str(re(nval)), '.'])


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
		#if (thresMethod != 'SL'):
		#	print("Invalid threshold method specified ! Using default method (SL)")
		nval = slMethod(scpu,m,n,max_err)

	# try:
	# 	if nval <= 3:
	# 		raise ValueError
	# except ValueError:
	# 	print("No singular value detected, aborting SVD")
	# 	return data, nval

	if nval <= 0:
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


# svdMethod :
#
## 1 : Singular Value Decompostion (SVD) on Toeplitz matrix
##		=> on full 1D with echoes --> very long
## 2 : Singular Value Decompostion (SVD) on echo matrix
##		=> on full 2D of stacked echoes --> very fast
## 3 : Singular Value Decompostion (SVD) on Toeplitz matrix of each echo
##		=> on separated echoes --> fast
#

def svd(data,nbHalfEcho,nbPtHalfEcho,svdMethod,thresMethod='SL',max_err=5):
	# default values
	## returned if for some reason svd isn't performed
	processedData = data
	if svdMethod not in [1,2,3]:
		print("\nInvalid SVD method specified (",svdMethod,"). SVD not performed.")
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
			
			#print('data[0,0] = ',data[0])
			try:

				t_0 = time()
				data64 = data.astype('complex64')		# decrease SVD computation time

				if svdMethod == 1:
					# Singular Value Decompostion (SVD) on Toeplitz matrix
					print("SVD on Toeplitz matrix in progress. Please be patient.")

					nbPtSignal = nbPtHalfEcho * nbHalfEcho
					row = math.ceil(nbPtSignal / 2)
					col = nbPtSignal - row + 1

					data_rec = np.empty([nbPtSignal],dtype='complex64')

					mat = linalg.toeplitz(data64[row-1::-1], data64[row-1::1])
					mat_rec, thres = svd_thres(mat,svdTools,thresMethod,max_err)
					for i in range (0, nbPtSignal):
						data_rec[i] = np.mean(np.diag(mat_rec[:,:],i-row+1))

					print("thres = ",thres)
					

				elif svdMethod == 2:
					# Singular Value Decompostion (SVD) on echo matrix
					print("SVD on echo matrix in progress. Please be patient.")

					data_rec, thres = svd_thres(data64,svdTools,thresMethod,max_err)

					print("thres = ",thres)


				elif svdMethod == 3:
					# Singular Value Decompostion (SVD) on Toeplitz matrix of each echo
					print("SVD on Toeplitz matrix of echoes in progress. Please be patient.")

					nbPtFullEcho = 2*nbPtHalfEcho
					nbFullEchoTotal = int((nbHalfEcho+1)/2)
					row = math.ceil(nbPtFullEcho / 2)
					col = nbPtFullEcho - row + 1

					data_rec = np.empty([nbFullEchoTotal, nbPtFullEcho],dtype='complex64')
					
					for i in range (0, nbFullEchoTotal):
						mat = linalg.toeplitz(data64[i,row-1::-1], data64[i,row-1::1])
						mat_rec, thres = svd_thres(mat,svdTools,thresMethod,max_err)
						print("thres = ",thres)
						for j in range (0, nbPtFullEcho):
							data_rec[i,j] = np.mean(np.diag(mat_rec[:,:],j-row+1))

				
				processedData = data_rec[:][:].astype('complex128')	# back to double precision
				t_2 = time()
				print("Decomposition + Reconstruction time:\t\t{0:8.2f}s".format(t_2 - t_0))
				print("\n------------------------------------------------------------------------\n")
			
			except ValueError:	# nval <= 0 dans svd_thres
				print("No singular value detected, aborting SVD")
				print("\n------------------------------------------------------------------------\n")
				processedData = data

			#print('processedData[0,0] = ',processedData[0])
	
	return processedData




###----------------------------------------------------------------------------
### 							TEST ZONE
### 		Beware for the test zone is messy and full of bugs
###----------------------------------------------------------------------------

if __name__ == "__main__":

	#A = np.genfromtxt('/home/pagilles/fichiersTaf/CPMG/code/NMR_post_proc-master/toeplitz.csv',delimiter=',',dtype='complex128')
	#print(A)

	#svd_thres(A,'SL')

	#newA, threshold = svd(A,1)
	#print("threshold value : ", threshold)
	#input('\nPress Any Key To Exit') # have the graphs stay displayed even when launched from linux terminal

	print("\n------------------------------------------------------------------------\n")
	print('arrayfireOK :',arrayfireOK)
	print('skcudaOK :',skcudaOK)
	print('scipyOK :',scipyOK)
	print('svd_tools_resolution_override :',svd_tools_resolution_override)
	print('modulesCheck :',modulesCheck)
	
	print()
	init()
	print('svd_tools_resolution_override :',svd_tools_resolution_override)
	print('modulesCheck :',modulesCheck)
	
	print()
	init(2)
	print('svd_tools_resolution_override :',svd_tools_resolution_override)
	print('modulesCheck :',modulesCheck)

	print()
	init(3)
	print('svd_tools_resolution_override :',svd_tools_resolution_override)
	print('modulesCheck :',modulesCheck)
	print("\n------------------------------------------------------------------------\n")

