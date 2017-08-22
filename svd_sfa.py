#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import csv

# Please create a function

afOK = False
cudaOK = False
scipyOK = False

# svd_tools_resolution_override  =>  
## 0 : default choice (skcuda if available / scipy if not)
## 1 : force arrayfire use
## 2 : force skcuda use
## 3 : force scipy use
svd_tools_resolution_override = 2

print('File : svd_sfa.py')

if svd_tools_resolution_override == 1:
	try:
		print('\nsvd_tools_resolution_override == 1  =>')
		print('  Forcing load of module arrayfire ...')
		import arrayfire as af
		af.set_backend('cpu')
		afOK = True
		print('Module arrayfire loaded successfully')
	except ModuleNotFoundError:
		print('Module arrayfire not found')

elif svd_tools_resolution_override == 2:
	try:
		print('\nsvd_tools_resolution_override == 2  =>')
		print('  Forcing load of module skcuda ...')
		import pycuda.autoinit
		import pycuda.gpuarray as gpuarray
		import skcuda.linalg as culinalg
		culinalg.init()
		cudaOK = True
		print('Module skcuda loaded successfully')
	except ModuleNotFoundError:
		print('Module skcuda not found')

elif svd_tools_resolution_override == 3:
	try:
		print('\nsvd_tools_resolution_override == 3  =>')
		print('  Forcing load of module scipy ...')
		from scipy import dot, linalg
		scipyOK = True
		print('Module scipy loaded successfully')
	except ModuleNotFoundError:
		print('Module scipy not found')

else:

	if svd_tools_resolution_override != 0:
		print("\nInvalid value specified for svd_tools_resolution_override ("\
			,svd_tools_resolution_override,").")
		print("Using default choice (0) :")

	# default choice
	
	try:
		print('\nLoading module arrayfire ...')
		import arrayfire as af
		afOK = True
		print('Module arrayfire loaded successfully')
	except ModuleNotFoundError:
		print('Module arrayfire not found')
	
	if (not afOK):
		try:
			print('\nLoading module skcuda ...')
			import pycuda.autoinit
			import pycuda.gpuarray as gpuarray
			import skcuda.linalg as culinalg
			culinalg.init()
			cudaOK = True
			print('Module skcuda loaded successfully')
		except ModuleNotFoundError:
			print('Module skcuda not found')

	elif (not cudaOK):
		try:
			print('\nLoading module scipy ...')
			from scipy import dot, linalg
			scipyOK = True
			print('Module scipy loaded successfully')
		except ModuleNotFoundError:
			print('Module scipy not found')




def svd_thres(data,thresMethod='SL',max_err=5):
	'''
	denData, nval = svd_thres(data,thresMethod='SL',max_err=5)

	data : data matrix
	thresMethod : 
		IND : Indicator Function (based on values difference)
		SL : Significance Level (probability of beeing noise)
	max_err : probability of being noise for SL
	denData : denoised data matrix
	'''

	try:
		svdTools = svd_tools_resolution()
	except ImportError as err:
		print ("\nsvd_tools_resolution returns error :")
		print("  ",err.args[0])
		print("\nSVD unavailable. Resuming signal processing.")
		nval = None
		return data, nval
	else:
		print("\n------------------------------------------------------------------------\n")
		print("thresMethod :",thresMethod)
		print("max_err :",max_err)
		print()

		# transpose if needed
		#global m,n
		[n,m] = np.shape(data)
		print("m :",m)
		print("n :",n)
		
		transp = 0
		
		if(m<n):
			print("data transpose...")
			data = data.transpose()
			transp = 1
			[n,m] = np.shape(data)
			print("m :",m)
			print("n :",n)

		print()

		# svd
		print("svd...")
		
		print("svdTools :",svdTools)
		u, sgpu, scpu, v = svd_decomposition(data, svdTools)
		
		#global s
		#u, s, v = linalg.svd(data, full_matrices=False)
		print("u.shape :",u.shape)
		print("v.shape :",v.shape)
		print("scpu.shape :",scpu.shape)
		print(scpu)
		

		print("\n------------------------------------------------------------------------\n")

		# thresholding
		nval = None
		if (thresMethod == 'IND'):
			nval = indMethod(scpu,m,n)[0]
		elif (thresMethod == 'SL'):
			nval = slMethod(scpu,m,n,max_err)
		else :
			print("Invalid threshold method specified !")
		
		print("nval :",nval)

		print("\n------------------------------------------------------------------------\n")


		# reconstruction
		
		denData = svd_reconstruction(u,sgpu,v,nval,svdTools)

		# transpose
		if transp == 1:
		    denData = denData.transpose()

		[n2,m2] = np.shape(denData)
		print("m2 :",m2)
		print("n2 :",n2)

		return denData, nval
	


###----------------------------------------------------------------------------
### SVD METHODS (WIP)
###----------------------------------------------------------------------------


def svd_tools_resolution():
	'''
	if afOK:
		choice = 'af'
	elif cudaOK:
		choice = 'cuda'
	elif scipyOK:
		choice = 'scipy'
	else:
		raise ImportError('Aucun module svd disponible.')
	'''
	if svd_tools_resolution_override == 1:
		if afOK:
			choice = 'af'
		else:
			raise ImportError('Selected SVD module (arrayfire) not found.')
	elif svd_tools_resolution_override == 2:
		if cudaOK:
			choice = 'cuda'
		else:
			raise ImportError('Selected SVD module (skcuda) not found.')
	elif svd_tools_resolution_override == 3:
		if scipyOK:
			choice = 'scipy'
		else:
			raise ImportError('Selected SVD module (scipy) not found.')
	else:
		if afOK:
			choice = 'af'
		elif cudaOK:
			choice = 'cuda'
		elif scipyOK:
			choice = 'scipy'
		else:
			raise ImportError('No SVD module available.')

	return choice


def svd_decomposition(mat, choice):
	'''
	U, s, s_cpu, Vh = svd_decomposition(mat, choice)
	
	U: left matrix
	s: singular values in 1D array
	s_cpu: copy of s to be used on cpu
	Vh: hermitian transpose of right matrix
	choice: svd frontend used
	'''
	if choice=='af':
		mat_gpu = af.to_array(mat[:,:])
		U, s_gpu, Vh = af.svd_inplace(mat_gpu[:,:])
		s_cpu = np.array(s_gpu)

	elif choice=='cuda':
		mat_gpu = gpuarray.to_gpu(mat[:,:])
		U, s_gpu, Vh = culinalg.svd(mat_gpu[:,:])
		s_cpu = s_gpu.get()

	elif choice=='scipy':
		U, s_cpu, Vh = linalg.svd(mat[:,:], full_matrices=False)
		s_gpu = s_cpu[:]
	else:
		print("Unknown svd tools")

	return U, s_gpu, s_cpu, Vh


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
	if choice=='af':
		S_gpu = af.diag(s_gpu[:thres], 0, False).as_type(af.Dtype.c32)
		mat_rec_gpu = af.matmul(U[:,:thres], 
			af.matmul(S_gpu[:thres,:thres], Vh[:thres,:]))
		mat_rec = np.array(mat_rec_gpu[:,:])

	elif choice=='cuda':
		S_gpu = gpuarray.zeros((thres,thres), np.complex64)
		S_gpu = culinalg.diag(s_gpu[:thres]).astype(np.complex64)
		mat_rec_gpu = culinalg.dot(U[:,:thres], \
			culinalg.dot(S_gpu[:,:], Vh[:thres,:]))
		mat_rec = mat_rec_gpu.get()

	elif choice=='scipy':
		S = linalg.diagsvd(s_gpu[:thres], thres, thres)
		mat_rec = dot(U[:,:thres], dot(S[:thres,:thres], Vh[:thres,:]))

	else:
		print("Unknown svd tools")

	return mat_rec


###----------------------------------------------------------------------------
### THRESHOLDING METHODS
###----------------------------------------------------------------------------
#significant factor analysis to help to determine
#	the number of significant factors in a matrix.
#
#Factor Analysis in Chemistry, Third Edition, p387-389
#Edmund R. Malinowki


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
		t[j,5] = s1
	t[n-1,5] = None

	nval = (np.nonzero((t[:,5]) < max_err))[0]
	if (nval.size==0):
		nval = 0
	else:
		nval = nval[-1]+1
	
	return nval

	# disp(['SL function indicates ', int2str(nval), ' significant factors.'])
	# if nval ~= 0
	# disp(['The real error (RE) is +/-', num2str(re(nval)), '.'])

def svd_preliminary_operations():
	if svd_tools_resolution() == 'af':	
		#	if (row*col < 2048*2048):
		#		af.set_backend('cpu')
		#	else:
		#		af.set_backend('unified')		# Bug :Anormally slow
		#	# see https://github.com/arrayfire/arrayfire-python/issues/134
		af.set_backend('cpu')
		af.info()


#A = np.genfromtxt('/home/pagilles/fichiersTaf/CPMG/code/NMR_post_proc-master/toeplitz.csv',delimiter=',',dtype='complex')
#print(A)

#svd_thres(A,'SL')