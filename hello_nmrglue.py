# Jython for Topspin
# -*- coding: utf-8 -*-

from CPython_init import CPYTHON_BIN, CPYTHON_LIB
import subprocess

def fullpath(dataset):
    dat=dataset[:]      # copy the original array
    if len(dat)==5:     # for topspin 2
        dat[3]="%s/data/%s/nmr" % (dat[3], dat[4])
    fulldata="%s/%s/%s/pdata/%s" % (dat[3], dat[0], dat[1], dat[2])
    return fulldata

# Get current processed data
dataset = CURDATA()
fulldataPATH = fullpath(dataset)

# Call to standard python
FILE = 'hello_nmrglue.py'
CPYTHON_FILE = CPYTHON_LIB + FILE
subprocess.call(CPYTHON_BIN + ' ' + CPYTHON_FILE + ' ' + fulldataPATH)