import subprocess
from CPython_init import CPYTHON_LOCATION, CPYTHON_FILES_LOCATION

script = CPYTHON_FILES_LOCATION + "np.py"

subprocess.call([CPYTHON_LOCATION]+[script])