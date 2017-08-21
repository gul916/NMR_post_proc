#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import csv

afOK = False
cudaOK = False
scipyOK = False

# svd_tools_resolution_override  =>  
## 0 : default choice (skcuda if available / scipy if not)
## 1 : force arrayfire use
## 2 : force skcuda use
## 3 : force scipy use
svd_tools_resolution_override = 0

print('File : svd_sfa.py')

if svd_tools_resolution_override == 1:
	try:
		print('\nsvd_tools_resolution_override == 1  =>')
		print('  Forcing load of module arrayfire ...')
		import arrayfire as af
		afOK = True
		print('Module arrayfire loaded successfully')
	except ModuleNotFoundError:
		print('Module arrayfire not found')

elif svd_tools_resolution_override == 2:
	try:
		print('\nsvd_tools_resolution_override == 2  =>')
		print('  Forcing load of module skcuda ...')
		import skcuda as sk
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

	# arrayfire module loading cancelled in default choice for now
	# due to errors in af methods
	'''
	try:
		print('\nLoading module arrayfire ...')
		import arrayfire as af
		afOK = True
		print('Module arrayfire loaded successfully')
	except ModuleNotFoundError:
		print('Module arrayfire not found')
	'''
	if (not afOK):
		try:
			print('\nLoading module skcuda ...')
			import skcuda as sk
			cudaOK = True
			print('Module skcuda loaded successfully')
		except ModuleNotFoundError:
			print('Module skcuda not found')

	if (not afOK)and(not cudaOK):
		try:
			print('\nLoading module scipy ...')
			from scipy import dot, linalg
			scipyOK = True
			print('Module scipy loaded successfully')
		except ModuleNotFoundError:
			print('Module scipy not found')




def svd_thres(data,thresMethod='SL',max_err=5):
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
		[m,n] = np.shape(data)
		print("m :",m)
		print("n :",n)
		
		transp = 0
		
		if(m<n):
			print("data transpose...")
			data = data.transpose()
			transp = 1
			[m,n] = np.shape(data)
			print("m :",m)
			print("n :",n)

		print()

		# svd
		print("svd...")
		
		print("svdTools :",svdTools)
		u, s, v = svd_decomposition(data,svdTools)
		
		#global s
		#u, s, v = linalg.svd(data, full_matrices=False)
		print("u.shape :",u.shape)
		print("v.shape :",v.shape)
		print("s.shape :",s.shape)
		print(s)
		

		print("\n------------------------------------------------------------------------\n")

		# thresholding
		nval = None
		if (thresMethod == 'SL'):
			nval = slMethod(s,m,n,max_err)
		elif (thresMethod == 'IND'):
			nval = indMethod(s,m,n)[0]
		else :
			print("Invalid threshold method specified !")
		
		print("nval :",nval)

		print("\n------------------------------------------------------------------------\n")


		# reconstruction
		
		denData = svd_reconstruction(u,s,v,n,nval,svdTools)

		# transpose
		if transp == 1:
		    denData = denData.transpose()

		[m2,n2] = np.shape(denData)
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
	#if choice=='cuda':

	if choice=='af':
		mat_af = af.to_array(mat[:,:])
		U, s, Vh = af.svd_inplace(mat_af[:,:])

	elif choice=='scipy':
		U, s, Vh = linalg.svd(mat[:,:], full_matrices=False)

	else:
		print("Unknown svd tools")

	return U, s, Vh


def svd_reconstruction(U, s, Vh, n, thres, choice):
	#if choice=='cuda':

	if choice=='af':
		S = af.diag(s[:], 0, False).as_type(af.Dtype.c32)
		mat_rec_af = af.matmul(af.matmul(U[:,:thres], S[:thres,:thres]), \
		Vh[:thres,:])
		mat_rec = np.array(mat_rec_af[:,:])

	elif choice=='scipy':
		S = linalg.diagsvd(s[:], n, n)
		mat_rec = dot(U[:,:thres], dot(S[:thres,:thres], Vh[:thres,:]))

	else:
		print("Unknown svd tools")

	return mat_rec


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