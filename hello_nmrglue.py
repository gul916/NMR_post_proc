# Jython for Topspin
# -*- coding: utf-8 -*-

from CPython_init import CPYTHON_BIN, CPYTHON_LIB
from subprocess import Popen, PIPE

FILE = 'hello_nmrglue.py'
CPYTHON_FILE = CPYTHON_LIB + FILE

def fullpath(dataset):
    dat=dataset[:]          # copy the original array
    if len(dat)==5:         # for topspin 2
        dat[3]="%s/data/%s/nmr" % (dat[3], dat[4])
    fulldata="%s/%s/%s/pdata/%s" % (dat[3], dat[0], dat[1], dat[2])
    return fulldata

# Get raw data
dataset = CURDATA()
fulldata = fullpath(dataset)

# Copy data
dataset_new = dataset[:]    # copy the original array
dataset_new[1]= INPUT_DIALOG(
    'Copy dataset', '', ['new expno ='], [str(int(dataset[1])+100)])[0]
XCMD(str('wrpa ' + dataset_new[1]))
RE(dataset_new)

# Verification
fulldata_new = fullpath(CURDATA())
if fulldata_new == fulldata:
    ERRMSG('Copy was not performed, hello_nmrglue aborted.', 'hello_nmrglue')
    EXIT()

# Call to standard python
SHOW_STATUS('hello_nmrglue in progress')
p = Popen(
    [CPYTHON_BIN, CPYTHON_FILE, fulldata_new],
    stdin=PIPE, stdout=PIPE, stderr=PIPE)
output, err = p.communicate()

# Display result
RE(dataset_new)
VIEWTEXT(title='hello_nmrglue', header='Output of hello_nmrglue script',
     text=output+'\n'+err, modal=0)