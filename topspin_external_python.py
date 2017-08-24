#import os
import subprocess

#CPYTHON=os.getenv('CPYTHON','NotDefined')
CPYTHON='/opt/anaconda3/bin/python'
script='/home/guillaume/Documents/PYTHON/hello_np.py'
subprocess.call([CPYTHON]+[script])
