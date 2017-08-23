#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import csv
from time import time

arrayfireOK = False
skcudaOK = False
scipyOK = False

# svd_tools_resolution_override  =>  
## 0 : default choice (skcuda if available / scipy if not)
## 1 : force arrayfire use
## 2 : force skcuda use
## 3 : force scipy use
svd_tools_resolution_override = 0

print("\n------------------------------------------------------------------------")
print('File : svd_sfa.py')
print("------------------------------------------------------------------------\n")

print('Establishing modules to import / Importing')

def import_arrayfire():
	arrayfireLoad = False
	try:
		print('\nLoading module arrayfire ...')
		global af
		import arrayfire as af
	except ModuleNotFoundError:
		print('Module arrayfire not found')
	else:
		arrayfireLoad = True
		print('Module arrayfire loaded successfully')
	return arrayfireLoad


def import_skcuda():
	skcudaLoad = False
	try:
		print('\nLoading module skcuda ...')
		global sk
		import skcuda as sk
	except ModuleNotFoundError:
		print('Module skcuda not found')
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
	except ModuleNotFoundError:
		print('Module scipy not found')
	else:
		scipyLoad = True
		print('Module scipy loaded successfully')
	return scipyLoad


if svd_tools_resolution_override == 1:
	print('\nsvd_tools_resolution_override == 1  =>')
	print('  Forcing load of module arrayfire')
	arrayfireOK = import_arrayfire()

elif svd_tools_resolution_override == 2:
	print('\nsvd_tools_resolution_override == 2  =>')
	print('  Forcing load of module skcuda')
	skcudaOK = import_skcuda()

elif svd_tools_resolution_override == 3:
	print('\nsvd_tools_resolution_override == 3  =>')
	print('  Forcing load of module scipy')
	scipyOK = import_scipy()

else:

	if svd_tools_resolution_override != 0:
		print("\nInvalid value specified for svd_tools_resolution_override ("\
			,svd_tools_resolution_override,").")
		print("Using default choice (0) :")

	# arrayfire module loading cancelled in default choice for now
	# due to errors in arrayfire methods

	#arrayfireOK = import_arrayfire()
	if (not arrayfireOK):
		skcudaOK = import_skcuda()

	if (not arrayfireOK)and(not skcudaOK):
		scipyOK = import_scipy()


###----------------------------------------------------------------------------
### SVD METHODS (WIP)
###----------------------------------------------------------------------------


def svd_tools_resolution():
	print ("\nSVD tools resolution :")
	if svd_tools_resolution_override == 1:
		if arrayfireOK:
			choice = 'arrayfire'
			print("Using methods from module : arrayfire")
		else:
			raise ImportError('Selected SVD module (arrayfire) not found.')
	elif svd_tools_resolution_override == 2:
		if skcudaOK:
			choice = 'skcuda'
			print("Using methods from module : skcuda")
		else:
			raise ImportError('Selected SVD module (skcuda) not found.')
	elif svd_tools_resolution_override == 3:
		if scipyOK:
			choice = 'scipy'
			print("Using methods from module : scipy")
		else:
			raise ImportError('Selected SVD module (scipy) not found.')
	else:
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


###----------------------------------------------------------------------------
### svd_decomposition
###----------------------------------------------------------------------------

def svd_decomposition(mat, choice):
	#if choice=='skcuda':

	if choice=='arrayfire':
		mat_af = af.to_array(mat[:,:])
		U, s, Vh = af.svd_inplace(mat_af[:,:])

	elif choice=='scipy':
		U, s, Vh = linalg.svd(mat[:,:], full_matrices=False)

	else:
		print("Unknown svd tools")

	return U, s, Vh



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

	global t
	t = np.zeros((n,6))

	for j in range (0, n):
	    t[j][0] = j
	    t[j][1] = ev[j]
	    t[j][2] = re[j]
	    t[j][3] = ind[j]
	    t[j][4] = rev[j]

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

def svd_reconstruction(U, s, Vh, thres, choice):
	#if choice=='skcuda':

	if choice=='arrayfire':
		S = af.diag(s[:], 0, False).as_type(af.Dtype.c32)
		mat_rec_af = af.matmul(af.matmul(U[:,:thres], S[:thres,:thres]), \
		Vh[:thres,:])
		mat_rec = np.array(mat_rec_af[:,:])

	elif choice=='scipy':
		S = linalg.diagsvd(s[:thres], thres, thres)
		mat_rec = dot(U[:,:thres], dot(S[:thres,:thres], Vh[:thres,:]))

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

	# transpose if needed
	[m,n] = np.shape(data)
	# print("m :",m)
	# print("n :",n)
	
	transp = 0
	
	if(m<n):
		#print("data transpose...")
		data = data.transpose()
		transp = 1
		[m,n] = np.shape(data)
		# print("m :",m)
		# print("n :",n)

	# svd decomposition
	u, s, v = svd_decomposition(data,svdTools)	

	# thresholding
	nval = None
	if (thresMethod == 'SL'):
		nval = slMethod(s,m,n,max_err)
	elif (thresMethod == 'IND'):
		nval = indMethod(s,m,n)[0]
	else :
		print("Invalid threshold method specified ! Using default method (SL)")
		nval = slMethod(s,m,n,max_err)

	# reconstruction
	denData = svd_reconstruction(u,s,v,nval,svdTools)

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
	thres = None
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
			if svdMethod == 1:
				# Singular Value Decompostion (SVD) on Toeplitz matrix
				print("SVD on Toeplitz matrix in progress. Please be patient.")
				t_0 = time()

				nbPtSignal = nbPtHalfEcho * nbHalfEcho
				row = math.ceil(nbPtSignal / 2)
				col = nbPtSignal - row + 1

				data = data.astype('complex64')		# decrease SVD computation time
				data_rec = np.empty([nbPtSignal],dtype='complex64')

				mat = linalg.toeplitz(data[row-1::-1], data[row-1::1])
				mat_rec, thres = svd_thres(mat,svdTools,thresMethod,max_err)
				for i in range (0, nbPtSignal):
					data_rec[i] = np.mean(np.diag(mat_rec[:,:],i-row+1))

				processedData = data_rec[:].astype('complex128')	# back to double precision

				t_2 = time()
				print("thres = ",thres)
				print("Decomposition + Reconstruction time:\t\t{0:8.2f}s".format(t_2 - t_0))

			elif svdMethod == 2:
				# Singular Value Decompostion (SVD) on echo matrix
				print("SVD on echo matrix in progress. Please be patient.")
				t_0 = time()

				mat = data.astype('complex64')		# decrease SVD computation time

				mat_rec, thres = svd_thres(mat,svdTools,thresMethod,max_err)

				processedData = mat_rec[:,:].astype('complex128')	# back to double precision

				t_2 = time()
				print("thres = ",thres)
				print("Decomposition + Reconstruction time:\t\t{0:8.2f}s".format(t_2 - t_0))
				
			elif svdMethod == 3:
				# Singular Value Decompostion (SVD) on Toeplitz matrix of each echo
				print("SVD on Toeplitz matrix of echoes in progress. Please be patient.")
				t_0 = time()

				nbPtFullEcho = 2*nbPtHalfEcho
				nbFullEchoTotal = int((nbHalfEcho+1)/2)
				row = math.ceil(nbPtFullEcho / 2)
				col = nbPtFullEcho - row + 1

				data = data.astype('complex64')		# decrease SVD computation time
				data_rec = np.empty([nbFullEchoTotal, nbPtFullEcho],dtype='complex64')
				
				for i in range (0, nbFullEchoTotal):
					mat = linalg.toeplitz(data[i,row-1::-1], data[i,row-1::1])
					mat_rec, thres = svd_thres(mat,svdTools,thresMethod,max_err)
					for j in range (0, nbPtFullEcho):
						data_rec[i,j] = np.mean(np.diag(mat_rec[:,:],j-row+1))

				processedData = data_rec[:,:].astype('complex128')	# back to double precision
				
				t_2 = time()
				print("thres = ",thres)
				print("Decomposition + Reconstruction time:\t\t{0:8.2f}s".format(t_2 - t_0))

	return processedData, thres




###----------------------------------------------------------------------------
### 							TEST ZONE
### 		Beware for the test zone is messy and full of bugs
###----------------------------------------------------------------------------

if __name__ == "__main__":

	A = np.genfromtxt('/home/pagilles/fichiersTaf/CPMG/code/NMR_post_proc-master/toeplitz.csv',delimiter=',',dtype='complex128')
	#print(A)

	#svd_thres(A,'SL')

	newA, threshold = svd(A,1)
	print("threshold value : ", threshold)
	input('\nPress Any Key To Exit') # have the graphs stay displayed even when launched from linux terminal