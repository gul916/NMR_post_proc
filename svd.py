# Jython for Topspin
# -*- coding: utf-8 -*-

from CPython_init import CPYTHON_BIN, CPYTHON_LIB
import subprocess

def fullpath(dataset):
    dat=dataset[:]      # copy the original array
    if len(dat)==5:     # for topspin 2
        dat[3]="%s/data/%s/nmr" % (dat[3], dat[4])
    fulldata="%s/%s/%s" % (dat[3], dat[0], dat[1])
    return fulldata

print('\n\n\n---------------------------------------------')    # line jump
print('svd.py started')

# Get raw data
fulldata = fullpath(CURDATA())
print('\nOriginal data %s' %fulldata)

# Analogic conversion
XCMD('convdta')
fulldata_new = fullpath(CURDATA())
if fulldata_new == fulldata:
    ERRMSG('Analogic conversion was not performed, SVD aborted.', 'SVD')
    EXIT()

# Call to standard python
FILE = 'denoise_nmr.py'
CPYTHON_FILE = CPYTHON_LIB + FILE
VIEWTEXT('SVD', text='SVD in progress. Please be patient.', modal=0)
subprocess.call(CPYTHON_BIN + ' ' + CPYTHON_FILE \
    + ' ' + fulldata_new + ' ' + fulldata_new)

VIEWTEXT('SVD', text='SVD finished.', modal=0)
print('\nsvd.py finished')
print('---------------------------------------------')    # line jump