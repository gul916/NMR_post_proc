# Jython  for Topspin
# -*- coding: utf-8 -*-

from CPython_init import CPYTHON_BIN, CPYTHON_LIB
import subprocess


FILE = 'hello_numpy.py'
script = CPYTHON_LIB + FILE

print("\n")			# line jump
subprocess.call([CPYTHON_BIN] + [script])
