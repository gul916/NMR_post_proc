# Jython  for Topspin
# -*- coding: utf-8 -*-

from CPython_init import CPYTHON_BIN, CPYTHON_LIB
import subprocess


FILE = 'hello_nmrglue.py'
script = CPYTHON_LIB + FILE
dataset = CURDATA()

def fullpath(dataset):
	dat=dataset[:]	# make a copy of the original array
	if len(dat)==5:	# for topspin 2
		dat[3]="%s/data/%s/nmr" % (dat[3], dat[4])
	fulldata="%s/%s/%s/pdata/%s" % (dat[3], dat[0], dat[1], dat[2])
	return fulldata

fulldataPATH = fullpath(dataset)

print("\n")			# line jump
subprocess.call([CPYTHON_BIN] + [script] + [fulldataPATH])