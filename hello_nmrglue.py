#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import nmrglue as ng
import sys

def load_data(direc):
	par, sig = ng.bruker.read(direc)
	return par, sig

def export_data(direc, par, sig):
	ng.bruker.write(direc, par, sig)

try:
	data_dir = sys.argv[1]
	if len(sys.argv) > 2:
		raise NotImplementedError("There should be only one argument.")

	params, signal = load_data(data_dir)
	plt.plot(signal.real)
	data_dir2 = data_dir + "-copie"
	export_data(data_dir2, params, signal)

except IndexError:
	print("Error: Please enter the directory of the Bruker file.")
except NotImplementedError as err:
	print("Error:", err)
	for i in range(1, len(sys.argv)):
		print("Argument", i, '=', sys.argv[i])
except OSError as err:
	print("Error:", err)
except:
	print("An unknown error occured")