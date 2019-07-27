# Jython for Topspin
# -*- coding: utf-8 -*-

from CPython_init import CPYTHON_BIN, CPYTHON_LIB
import subprocess

print('\n\n\n---------------------------------------------')    # line jump
print('hello_numpy.py started\n')

# Call to standard python
FILE = 'hello_numpy.py'
CPYTHON_FILE = CPYTHON_LIB + FILE
subprocess.call(CPYTHON_BIN + ' ' + CPYTHON_FILE)

print('\nhello_numpy.py finished')
print('---------------------------------------------')    # line jump