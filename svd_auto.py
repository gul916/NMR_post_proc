# Jython for Topspin
# -*- coding: utf-8 -*-

from CPython_init import CPYTHON_BIN, CPYTHON_LIB
import subprocess


# Get current raw data
dataset = CURDATA()
def fullpath(dataset):
	dat=dataset[:]		# copy the original array
	if len(dat)==5:	# for topspin 2
		dat[3]="%s/data/%s/nmr" % (dat[3], dat[4])
	fulldata="%s/%s/%s" % (dat[3], dat[0], dat[1])
	return fulldata
fulldataPATH = fullpath(dataset)

# Confirmation dialog
if CONFIRM("Warning", "Overwrite current raw data?") == 0:
	EXIT()
else:
	confirmTS = 'True'

# Call to standard python
FILE = 'svd_auto.py'
script = CPYTHON_LIB + FILE
subprocess.call([CPYTHON_BIN] + [script] + [fulldataPATH] + [confirmTS])