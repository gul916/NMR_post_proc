# Jython for Topspin
# -*- coding: utf-8 -*-

from CPython_init import CPYTHON_BIN, CPYTHON_LIB
from subprocess import Popen, PIPE

FILE = 'snr.py'
CPYTHON_FILE = CPYTHON_LIB + FILE

def fullpath(dataset):
    dat=dataset[:]          # copy the original array
    if len(dat)==5:         # for topspin 2
        dat[3]="%s/data/%s/nmr" % (dat[3], dat[4])
    fulldata="%s/%s/%s/pdata/%s" % (dat[3], dat[0], dat[1], dat[2])
    return fulldata

# Get current processed data
dataset = CURDATA()
fulldata = fullpath(dataset)

# Get signal and noise regions
if CONFIRM('SNR regions',
           'Please define signal and noise regions with ".sino"') == 0:
    EXIT()
sig_lim_1 = GETPAR2('SIGF1')
sig_lim_2 = GETPAR2('SIGF2')
nois_lim_1 = GETPAR2('NOISF1')
nois_lim_2 = GETPAR2('NOISF2')

# Call to standard python
COMMAND_LINE = " ".join(str(elm) for elm in [
    CPYTHON_BIN, CPYTHON_FILE, fulldata,
    sig_lim_1, sig_lim_2, nois_lim_1, nois_lim_2])
SHOW_STATUS('snr in progress')
p = Popen(COMMAND_LINE, stdin=PIPE, stdout=PIPE, stderr=PIPE)
output, err = p.communicate()

# Display result
VIEWTEXT(title='snr', header='Output of snr script',
     text=output+'\n'+err, modal=0)