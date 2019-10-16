# Jython for Topspin
# -*- coding: utf-8 -*-

from CPython_init import CPYTHON_BIN, CPYTHON_LIB, get_os_version
from subprocess import Popen, PIPE

FILE = 'CPMG_proc.py'
CPYTHON_FILE = CPYTHON_LIB + FILE

def fullpath(dataset):
    dat=dataset[:]          # copy the original array
    if len(dat)==5:         # for topspin 2
        dat[3]="%s/data/%s/nmr" % (dat[3], dat[4])
    fulldata="%s/%s/%s/pdata/%s" % (dat[3], dat[0], dat[1], dat[2])
    return fulldata

# CPMG options
de = float(GETPAR('DE'))
dw = float(GETPAR('DW'))
options = INPUT_DIALOG(
    'CPMG options',
    'If you don\'t know these values, please calibrate them with cpmgcal',
    ['fullEcho = ', 'nbEcho = ', 'firstDec = ', 'nbPtShift ='],
    ['1e-3', '10', 'True', str(int(de/dw))],
    ['full echo duration (s)', 'number of echoes (integer)',
     'first decrease (True / False)',
     'number of points to shift (>0: right shift, <0: left shift)'],
    ['1', '1', '1', '1'])

# Get raw data
dataset = CURDATA()
fulldata = fullpath(dataset)

# Copy data
dataset_new = dataset[:]    # copy the original array
dataset_new[1]= INPUT_DIALOG(
    'Copy dataset', '', ['new expno ='], [str(int(dataset[1])+100)])[0]
XCMD(str('wrpa ' + dataset_new[1] + ' 1'))
RE(dataset_new)

# Verification
fulldata_new = fullpath(CURDATA())
if fulldata_new == fulldata:
    ERRMSG('Copy was not performed, cpmg aborted.', 'cpmg')
    EXIT()

# Update parameters
PUTPAR('PH_mod', 'pk')
EFP()
XCMD(str('wrp ' + '2'))
XCMD(str('wrp ' + '3'))
XCMD(str('wrp ' + '4'))

# Call to standard python
COMMAND_LINE = [
    CPYTHON_BIN, CPYTHON_FILE, fulldata_new,
    options[0], options[1], options[2], options[3]]
if get_os_version().startswith('windows'):
    COMMAND_LINE = " ".join(str(elm) for elm in COMMAND_LINE)
SHOW_STATUS('CPMG processing in progress, please be patient.')
p = Popen(COMMAND_LINE, stdin=PIPE, stdout=PIPE, stderr=PIPE)
output, err = p.communicate()

# Display result
XCMD(str('rep ' + '4'))
VIEWTEXT(title='cpmg', header='Output of cpmg script',
     text=output+'\n'+err, modal=0)
