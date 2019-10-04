# Jython for Topspin
# -*- coding: utf-8 -*-

# Path to CPython executable
# Under Topspin 3.6.1, global PATH variable is overwritten
# and Anaconda PATH variable need to be reloaded

# Check operating system
import sys
def get_os_version():
    ver = sys.platform.lower()
    if ver.startswith('java'):
        import java.lang
        ver = java.lang.System.getProperty("os.name").lower()
    return ver

# CPYTHON_BIN: path to CPython executable
# NMR_POST_PROC: directory of NMR_post_proc
# CPYTHON_LIB: directory of CPython programs to execute
# Please use slash (/) rather than backslash (\)

# Windows
if get_os_version().startswith('windows'):
    CPYTHON_BIN = (
        'C:/Windows/System32/cmd.exe /C'
        + 'C:/ProgramData/Anaconda3/Scripts/activate.bat'
        + 'C:/ProgramData/Anaconda3 && python')
    NMR_POST_PROC = 'D:/Users/Guillaume/Documents/PYTHON/NMR_post_proc/'

# Linux
elif get_os_version().startswith('linux'):
    CPYTHON_BIN = '/opt/anaconda3/bin/python3'
    NMR_POST_PROC = '/home/guillaume/Documents/PYTHON/NMR_post_proc/'

# Other
else:
    ERRMSG('Operating system not supported.', 'CPython_init')
    EXIT()

CPYTHON_LIB = NMR_POST_PROC + 'CPython/'
