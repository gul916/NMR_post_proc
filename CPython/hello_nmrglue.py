#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import nmrglue as ng
import numpy as np
import sys


def test_nmrglue(data_dir):
	dic, data = ng.bruker.read_pdata(data_dir)		# import data
	udic = ng.bruker.guess_udic(dic, data)	# convert to universal dictionary
	
	if data.ndim == 1:
		uc = ng.fileiobase.uc_from_udic(udic)
		ppm_scale = uc.ppm_scale()
		
		plt.plot(ppm_scale, data.real)
		plt.gca().invert_xaxis()
	
	elif data.ndim == 2:
		uc_dim1 = ng.fileiobase.uc_from_udic(udic, dim=1)
		ppm_dim1 = uc_dim1.ppm_scale()
		
		uc_dim0 = ng.fileiobase.uc_from_udic(udic, dim=0)
		ppm_dim0 = uc_dim0.ppm_scale()
		
		lev0 = 0.05 * np.amax(data.real)
		toplev = 0.95 * np.amax(data.real)
		nlev = 15
		levels = np.linspace(lev0, toplev, nlev)
		
		plt.contour(ppm_dim1, ppm_dim0, data.real, levels)
		plt.gca().invert_yaxis()
		plt.gca().invert_xaxis()
	
	else:
		raise NotImplementedError("Data of", data.ndim, "dimensions are not yet supported.")
	
	plt.ioff()									# Interactive mode off
	plt.show()


#%%----------------------------------------------------------------------------
### When this file is executed directly
###----------------------------------------------------------------------------

if __name__ == "__main__":
	print()										# Line jump
	
	try:
		if len(sys.argv) == 1:
			raise NotImplementedError("Please enter the directory of the Bruker file.")
		elif len(sys.argv) == 2:
			data_dir = sys.argv[1]
			test_nmrglue(data_dir)
		elif len(sys.argv) >= 3:
			raise NotImplementedError("There should be only one argument.")
	
	except NotImplementedError as err:
		print("Error:", err)
		for i in range(0, len(sys.argv)):
			print("Argument", i, '=', sys.argv[i])
	except OSError as err:
		print("Error:", err)
	else:		# When no error occured
		print("NMRglue successfully tested")