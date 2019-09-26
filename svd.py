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
# Induces negative components for each line
XCMD('convdta')
fulldata_new, fulldata_new_proc = fullpath(CURDATA())
if fulldata_new == fulldata:
    ERRMSG('Analogic conversion was not performed, SVD aborted.', 'SVD')
    EXIT()

# For phasing under python
PUTPAR('1 WDW', 'SINE')
PUTPAR('1 SSB', '2')
if dim == 1:
    EFP()
else:
    PUTPAR('2 WDW', 'SINE')
    PUTPAR('2 SSB', '2')
    XFB()

# SVD options
options = INPUT_DIALOG(
    'SVD options', '', ['k_thres = ', 'max_err = '], ['0', '7.5'],
    ['0: automatic thresholding\n>0: manual thresholding',
    'allowed error (5-10 %)\nirrelevant if k_thres > 0'], ['1', '1'])

# Call to standard python
FILE = 'denoise_nmr.py'
CPYTHON_FILE = CPYTHON_LIB + FILE
ARGUMENTS = fulldata_new_proc +' '+ fulldata_new \
    +' '+ options[0] +' '+ options[1]
SHOW_STATUS('SVD in progress, please be patient.')
subprocess.call(CPYTHON_BIN +' '+ CPYTHON_FILE +' '+ ARGUMENTS)

# Read processed data
PUTPAR('1 WDW', 'no')
if dim == 1:
    FP()
else:
    PUTPAR('2 WDW', 'no')
    XFB()

print('\nsvd.py finished')
print('---------------------------------------------')          # line jump