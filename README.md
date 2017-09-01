# NMR_post_proc
Post Processing of NMR Bruker data

Two kinds of files are present :
- standard Python files based on C language (CPython)
 --> this directory, to be placed in CPython subdirectory of Topspin 
- Jython files based on Java language --> Topspin directory

CPython files can be started directly with a Python 3 program.
Anaconda is suggested as it is especially efficient and uses Intel MKL library.

Jython Files provide an interface from Topspin to CPython files.