# Jython for Topspin
# -*- coding: utf-8 -*-

from CPython_init import CPYTHON_BIN, CPYTHON_LIB
import subprocess

# Call to standard python
FILE = 'hello_numpy.py'
CPYTHON_FILE = CPYTHON_LIB + FILE
subprocess.call(CPYTHON_BIN + ' ' + CPYTHON_FILE)
