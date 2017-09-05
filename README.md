# NMR_post_proc
Post Processing of NMR Bruker data using Topspin software
Written by Guillaume LAURENT and Pierre-Aymeric GILLES in 2017


### Introduction
Two kinds of files are present in this repository :
- Jython files based on Java language
 --> in NMR_post_proc directory
- standard Python files based on C language (CPython)
 --> in CPython subdirectory

Jython Files provide an interface from Topspin to CPython files.

CPython files can be started directly with a Python 3 program.
Anaconda Python is suggested as it uses Intel MKL library
which is especially fast.



### Installation:
- Download NMR_post_proc.zip.
- Extract NMR_post_proc.zip into the desired directory.
- Modify CPython_init.py according to Anaconda and NMR_post_proc directories.
- Make a shortcut of CPython_init.py and move it into
$TOPSPIN/exp/stan/nmr/py/user where $TOPSPIN is Topspin directory.
- In Topspin preferences, locate "Directories/Manage source directories for edpul, edau, etc." and click on "Change". Select "Python Programs", and add NMR_post_proc directory.
- Restart Topspin.



### Test installation:
Enter hello_numpy in Topspin command line.
In Topspin terminal, you should see various libraries refering to
anaconda and MKL



### File modifications
When you modify CPython_init.py, remove CPython_init$py.class file
in $TOPSPIN/exp/stan/nmr/py/user and restart Topspin.

