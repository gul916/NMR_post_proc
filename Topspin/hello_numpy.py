# Jython  for Topspin

from CPython_init import CPYTHON_BIN, CPYTHON_LIB
import subprocess

FILE = 'hello_numpy.py'

script = CPYTHON_LIB + FILE
subprocess.call([CPYTHON_BIN]+[script])
