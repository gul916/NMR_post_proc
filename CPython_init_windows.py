# Jython for Topspin
# -*- coding: utf-8 -*-
# Please use slash (/) rather than backslash (\)

# Path to CPython executable
# Under Topspin 3.6.1, global PATH variable is overwritten
# and Anaconda PATH variable need to be reloaded
CPYTHON_BIN = 'C:/Windows/System32/cmd.exe /C \
C:/ProgramData/Anaconda3/Scripts/activate.bat C:/ProgramData/Anaconda3 \
&& python'

# Directory of NMR_post_proc
NMR_POST_PROC = 'D:/Users/Guillaume/Documents/Python/NMR_post_proc/'

# Directory of CPython programs to execute
CPYTHON_LIB = NMR_POST_PROC + 'CPython/'
