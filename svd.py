# Jython for Topspin
# -*- coding: utf-8 -*-

from CPython_init import CPYTHON_BIN, CPYTHON_LIB
import os.path
import subprocess

def fullpath(dataset):
    dat=dataset[:]      # copy the original array
    if len(dat)==5:     # for topspin 2
        dat[3]="%s/data/%s/nmr" % (dat[3], dat[4])
    fulldata="%s/%s/%s" % (dat[3], dat[0], dat[1])
    fulldata_proc="%s/%s/%s/pdata/%s" % (dat[3], dat[0], dat[1], dat[2])
    return fulldata, fulldata_proc

def dimension(fulldata):
    if os.path.exists(fulldata + '/acqu2s'):                    # check 2D file
        dim = 2
    else:
        dim = 1
    return dim

print('\n\n\n---------------------------------------------')    # line jump
print('svd.py started')

# Get raw data
fulldata, fulldata_proc = fullpath(CURDATA())
dim = dimension(fulldata)
print('\nOriginal data %s' %fulldata_proc)

# Analogic conversion
XCMD('convdta')
fulldata_new, fulldata_new_proc = fullpath(CURDATA())
if fulldata_new == fulldata:
    ERRMSG('Analogic conversion was not performed, SVD aborted.', 'SVD')
    EXIT()

# For phasing under python
if dim == 1:
    EFP()
else:
    XFB()

# Call to standard python
FILE = 'denoise_nmr.py'
CPYTHON_FILE = CPYTHON_LIB + FILE
SHOW_STATUS('SVD in progress, please be patient.')
subprocess.call(CPYTHON_BIN + ' ' + CPYTHON_FILE \
    + ' ' + fulldata_new_proc + ' ' + fulldata_new)

# Read processed data
PUTPAR('1 WDW', 'no')
if dim == 1:
    FP()
else:
    PUTPAR('2 WDW', 'no')
    XFB()

print('\nsvd.py finished')
print('---------------------------------------------')          # line jump