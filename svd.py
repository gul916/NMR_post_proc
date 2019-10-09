# Jython for Topspin
# -*- coding: utf-8 -*-

from CPython_init import CPYTHON_BIN, CPYTHON_LIB, get_os_version
from subprocess import Popen, PIPE

FILE = 'denoise_nmr.py'
CPYTHON_FILE = CPYTHON_LIB + FILE

def fullpath(dataset):
    dat=dataset[:]          # copy the original array
    if len(dat)==5:         # for topspin 2
        dat[3]="%s/data/%s/nmr" % (dat[3], dat[4])
    fulldata="%s/%s/%s/pdata/%s" % (dat[3], dat[0], dat[1], dat[2])
    return fulldata

# SVD options
options = INPUT_DIALOG(
    'SVD options', '', ['k_thres = ', 'max_err = '], ['0', '7.5'],
    ['0: automatic thresholding\n>0: manual thresholding',
    'allowed error (5-10 %)\nirrelevant if k_thres > 0'], ['1', '1'])

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
COMMAND_LINE = [
    CPYTHON_BIN, CPYTHON_FILE, fulldata_new, options[0], options[1]]
if get_os_version().startswith('windows'):
    COMMAND_LINE = " ".join(str(elm) for elm in COMMAND_LINE)
SHOW_STATUS('SVD in progress, please be patient.')
p = Popen(COMMAND_LINE, stdin=PIPE, stdout=PIPE, stderr=PIPE)
output, err = p.communicate()

# Display result
RE(dataset_new)
VIEWTEXT(
    title='svd', header='Output of svd script',
    text=output+'\n'+err, modal=0)
