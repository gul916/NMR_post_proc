### NMR_post_proc
Post Processing of NMR Bruker data using Topspin software
Written by Guillaume LAURENT and Pierre-Aymeric GILLES in 2017


### Introduction
Two kinds of files are present in this repository :
- Jython files based on Java language.
 --> They are in NMR_post_proc directory.
 --> They provide an interface from Topspin to standard Python files.
- standard Python files based on C language (CPython).
 --> They are in CPython subdirectory
 --> They can be started directly with a Python 3 program.


### Prerequisites
- Install Bruker Topspin
- Install Anaconda Python. It uses Intel MKL library which is especially fast.
- Install nmrglue : conda install -c fernandezc nmrglue 


### Installation under windows
- Download NMR_post_proc.zip.
- Extract NMR_post_proc.zip into the desired directory.
- Modify CPython_init_windows.py according to Anaconda and NMR_post_proc directories.
- Copy CPython_init_windows.py as CPython_init.py and move it into
'C:\Bruker\TopSpin3.5pl7\exp\stan\nmr\py\user'.
- In Topspin, click on Manage tab and preferences button, locate 'Manage source directories for edpul, edau, etc.' in Directories section and click on 'Change'. Select 'Python Programs', and add NMR_post_proc directory.
- Restart Topspin.


### Installation under linux
- Download NMR_post_proc.zip.
- Extract NMR_post_proc.zip into the desired directory.
- Modify CPython_init_linux.py according to Anaconda and NMR_post_proc directories.
- Copy CPython_init_linux.py as CPython_init.py and move it into
/opt/Bruker/TopSpin3.5pl7/exp/stan/nmr/py/user.
- In Topspin, click on Manage tab and preferences button, locate "Directories/Manage source directories for edpul, edau, etc." and click on "Change". Select "Python Programs", and add NMR_post_proc directory.
- Restart Topspin.


### CPython_init.py modifications
If you modify CPython_init.py, you need to remove CPython_init$py.class file in $TOPSPIN/exp/stan/nmr/py/user and to restart Topspin.


### Tests
- Enter hello_numpy in Topspin command line. In Topspin terminal, you should see various libraries refering to anaconda and MKL
- Open a processed 1D or 2D dataset. Enter hello_nmrglue in Topspin command line. You should see a figure of the corresponding spectrum. Furthermore, in Topspin terminal, you should see "NMRglue successfully tested"


### Acknowledgments
Julien TREBOSC is thanked for providing sample code in https://github.com/jtrebosc/JTutils , especially for the subprocess call.
