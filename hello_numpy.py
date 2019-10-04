# Jython for Topspin
# -*- coding: utf-8 -*-

from CPython_init import CPYTHON_BIN, CPYTHON_LIB, get_os_version
from subprocess import Popen, PIPE

FILE = 'hello_numpy.py'
CPYTHON_FILE = CPYTHON_LIB + FILE
COMMAND_LINE = [CPYTHON_BIN, CPYTHON_FILE]
if get_os_version().startswith('windows'):
    COMMAND_LINE = " ".join(str(elm) for elm in COMMAND_LINE)

# Call to standard python
SHOW_STATUS('hello_numpy in progress')
p = Popen(COMMAND_LINE, stdin=PIPE, stdout=PIPE, stderr=PIPE)
output, err = p.communicate()

# Display result
VIEWTEXT(
    title='hello_numpy', header='Output of hello_numpy script',
    text=output+'\n'+err, modal=0)
