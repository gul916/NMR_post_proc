#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import csv
from scipy import linalg

def svd_autoThres(data,method='SL',max_err=5):
	
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

	print("svd...")
	u, s, v = linalg.svd(data, full_matrices=False)
	print("u.shape :",u.shape)
	print("v.shape :",v.shape)
	print("s.shape :",s.shape)
	print(s)

	

	print("\n------------------------------------------------------------------------\n")
	
	
	## Indicator function IND
	# preallocating

	ev = np.zeros(n)

	df = np.zeros(n)
	rev = np.zeros(n)
	sev = np.zeros(n-1)
	sdf = np.zeros(n-1)
	re = np.zeros(n-1)
	ind = np.zeros(n-1)

	#for j = 1:n
	for j in range (0, n):
	    ev[j] = s[j]**2
	    df[j] = (m-j)*(n-j)
	    rev[j] = ev[j] / df[j]
	    #print(j)
	#print("ev :",ev)
	#print("df :",df)
	#print("rev :",rev)
	
	#for k = 1:n-1
	for k in range (0, n-1):
	    sev[k] = np.sum(ev[k+1:n])
	    sdf[k] = np.sum(df[k+1:n])
	#print("sev :",sev)
	#print("sdf :",sdf)
	
	#for i = 1:n-1
	for i in range (0, n-1):
	    re[i] = np.sqrt(sev[i] / (m * (n-i-1)))		# see eq. 4.44
	    ind[i] = re[i] / (n-i-1)**2					# see eq. 4.63
	#print("re :",re)
	#print("ind :",ind)



	#nval = np.amin(ind)
	nval = np.argmin(ind)+1
	
	null = np.array([None])
	re = np.concatenate((re[:],null[:]))
	ind = np.concatenate((ind[:],null[:]))

	t = np.zeros((n,6))
	#for j = 1:n
	for j in range (0, n):
	    t[j][0] = j
	    t[j][1] = ev[j]
	    t[j][2] = re[j]
	    t[j][3] = ind[j]
	    t[j][4] = rev[j]
	#print("t :\n",t)
	print("nval :",nval)

	# disp(['IND function indicates ', int2str(nval), ' significant factors.'])
	# disp(['The real error (RE) is +/-', num2str(re(nval)), '.'])
	

	print("\n------------------------------------------------------------------------\n")

	## Significance Level SL

	#for j = 1:n-1
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
			#print("ks : ",ks)
			#print("im : ",im)
			fk = ks
			if (im - 2) >= 0:
				#for k = ks:2:im
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
		#print("s1 : ",s1)
		t[j][5] = s1
	t[n-1][5] = None
	#print("t :\n",t)
	#print("t[:][5] :\n",t[:,5])
	#print("t[:][5] < max_err:\n",np.nonzero((t[:,5])< max_err))
	
	#nval2 = find((t[:][5]) < max_err, 1, 'last')		# see eq. 4.83
	nval2 = (np.nonzero((t[:,5]) < max_err))[0]
	#print(nval2)
	#if isempty(nval2):
	if (nval2.size==0):
		nval2 = 0
	else:
		nval2 = nval2[-1]+1
	print("nval2 :",nval2)

	
	# disp(['SL function indicates ', int2str(nval2), ' significant factors.'])
	# if nval2 ~= 0
	    # disp(['The real error (RE) is +/-', num2str(re(nval2)), '.'])

	print("\n------------------------------------------------------------------------\n")

	
	print("\ndiagsvd...")
	s = linalg.diagsvd(s[:], n, n)
	print("s.shape :",s.shape)
	print(s)

	'''
	# matrix reconstruction
	d_den = u(:,1:nval)*s(1:nval,1:nval)*v(:,1:nval)'			# using IND
	d_den2 = u(:,1:nval2)*s(1:nval2,1:nval2)*v(:,1:nval2)'		# using SL

	# transpose
	if transp == 1
	    d_den = d_den'
	    d_den2 = d_den2'
	'''

#A = np.random.rand(3,5)
A = np.ones((3,5))
A*=-1
print(A)
#reader = csv.reader(open("/home/pagilles/fichiersTaf/CPMG/code/NMR_post_proc-master/toeplitz.csv", "rb"), delimiter=",")
#x = list(reader)
#A = numpy.array(x).astype('complex')
A = np.genfromtxt('/home/pagilles/fichiersTaf/CPMG/code/NMR_post_proc-master/toeplitz.csv',delimiter=',',dtype='complex')
#print(A)

svd_autoThres(A)