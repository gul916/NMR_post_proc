#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import csv
from scipy import linalg

'''
try
	import ...
	define
except
	
'''

def svd_thres(data,method='SL',max_err=5):
	'''
	svd_autoThres 	significant factor analysis - a program designed
	    			to help determine the number of significant factors in a matrix.
	
	Factor Analysis in Chemistry, Third Edition, p387-389
	Edmund R. Malinowki
	'''

	# data : data matrix
	# method : 
		# IND : Indicator Function
		# SL : Significance Level
	# max_err : probability of being noise for SL

	# denData_IND : denoised data matrix with IND values
	# denData_SL : denoised data matrix with SL values

	print("\n------------------------------------------------------------------------\n")
	print("method :",method)
	print("max_err :",max_err)
	print()

	# transpose if needed
	global m,n
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
	svdMethod = svd_method_resolution()
	'''
	global s
	u, s, v = linalg.svd(data, full_matrices=False)
	print("u.shape :",u.shape)
	print("v.shape :",v.shape)
	print("s.shape :",s.shape)
	print(s)
	'''

	print("\n------------------------------------------------------------------------\n")

	# thresholding
	nval = None
	if method == 'SL':
		nval = slMethod(max_err)
	elif method == 'IND':
		nval = indMethod()[0]
	else :
		print("Invalid method specified !")
	
	print("nval :",nval)

	print("\n------------------------------------------------------------------------\n")


	print("\ndiagsvd...")
	s = linalg.diagsvd(s[:], n, n)
	print("s.shape :",s.shape)
	print(s)

	'''
	# matrix reconstruction
	d_den = u(:,1:nval)*s(1:nval,1:nval)*v(:,1:nval)'			# using IND
	d_den2 = u(:,1:nval)*s(1:nval,1:nval)*v(:,1:nval)'		# using SL

	# transpose
	if transp == 1
	    d_den = d_den'
	    d_den2 = d_den2'
	'''


###----------------------------------------------------------------------------
### SVD METHODS (WIP)
###----------------------------------------------------------------------------

def svd_method_resolution():
	# try except => determine svd method to use
	# choice = 'cuda'/'af'/'scipy'
	return choice


def svd_decomposition(data, choice):
	if choice=='cuda':

	elif choice=='af':

	elif choice=='scipy':

	else:
		print("Unknown svd method")

	return u,s,v

def svd_reconstruction(u, s, v, choice):
	if choice=='cuda':

	elif choice=='af':

	elif choice=='scipy':

	else:
		print("Unknown svd method")

	return newData


###----------------------------------------------------------------------------
### THRESHOLDING METHODS
###----------------------------------------------------------------------------

## Indicator function IND
def indMethod():

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
def slMethod(max_err):
	nval,sdf,ev,sev,t = indMethod()
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


A = np.genfromtxt('/home/pagilles/fichiersTaf/CPMG/code/NMR_post_proc-master/toeplitz.csv',delimiter=',',dtype='complex')
#print(A)

svd_thres(A,'SL')