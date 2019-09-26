# Jython for Topspin
# -*- coding: utf-8 -*-

from CPython_init import CPYTHON_BIN, CPYTHON_LIB
from subprocess import Popen, PIPE

FILE = 'hello_numpy.py'
CPYTHON_FILE = CPYTHON_LIB + FILE

# Call to standard python
#subprocess.call(CPYTHON_BIN + ' ' + CPYTHON_FILE)
p = Popen([CPYTHON_BIN, CPYTHON_FILE], stdin=PIPE, stdout=PIPE, stderr=PIPE)
output, err = p.communicate()
rc = p.returncode

SHOW_STATUS('hello_numpy in progress')
# Display result
VIEWTEXT(title='hello_numpy', header='Output of hello_numpy script',
     text=output+'\n'+err, modal=0)