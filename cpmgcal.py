# Jython for Topspin
# -*- coding: utf-8 -*-

from CPython_init import CPYTHON_BIN, CPYTHON_LIB, get_os_version
from subprocess import Popen, PIPE

FILE = 'CPMG_cal.py'
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
    'CPMG options', 'Use this program recursively to calibrate these values',
    ['fullEcho = ', 'nbEcho = ', 'firstDec = ', 'nbPtShift ='],
    ['1e-3', '10', 'True', str(int(de/dw))],
    ['full echo duration (s)', 'number of echoes (integer)',
     'first decrease (True / False)',
     'number of points to shift (>0: right shift, <0: left shift)'],
    ['1', '1', '1', '1'])

# Get raw data
dataset = CURDATA()
fulldata = fullpath(dataset)

# Call to standard python
COMMAND_LINE = [
    CPYTHON_BIN, CPYTHON_FILE, fulldata,
    options[0], options[1], options[2], options[3]]
if get_os_version().startswith('windows'):
    COMMAND_LINE = " ".join(str(elm) for elm in COMMAND_LINE)
SHOW_STATUS('CPMG calibration in progress, please be patient')
p = Popen(COMMAND_LINE, stdin=PIPE, stdout=PIPE, stderr=PIPE)
output, err = p.communicate()

# Display result
VIEWTEXT(title='cpmgcal', header='Output of cpmgcal script',
     text=output+'\n'+err, modal=0)
