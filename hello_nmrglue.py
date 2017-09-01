#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import nmrglue as ng
import sys

def test_nmrglue(direc):
	params, A = ng.bruker.read(direc)		# import data
	
	if A.ndim ==1:
		plt.plot(A.real)
	elif A.ndim ==2:
		plt.contour(A.real)
	else:
		raise NotImplementedError("Data of", A.ndim, "dimensions are not yet supported.")
	
	direc += "-copy"
	ng.bruker.write(direc, params, A)		# export data
	print("Data saved to", direc)



#%%----------------------------------------------------------------------------
### When this file is executed directly
###----------------------------------------------------------------------------

if __name__ == "__main__":
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
		print("\nNMRglue read and write successfully tested")
	
	
	
	input("Press enter key to exit") # wait before closing terminal