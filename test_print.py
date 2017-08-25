# -*- coding: utf-8 -*-

import subprocess
from CPython_init import CPYTHON_LOCATION, CPYTHON_FILES_LOCATION

script = CPYTHON_FILES_LOCATION + "print.py"


result=INPUT_DIALOG("CPMG PARAMETERS", 
  """ please provide the full echo duration FED (milliseconds),
   the number of echoes nbE
   and specify if there is a first decay (yes : 1 / no : 0)..""", 
     ["FED=","nbE=", "first decay : "])
(fullEcho,nbEcho,FD)=(result[0],result[1],result[2])

dataset=CURDATA()
print(dataset)
def fullpath(dataset):
	# function coded by Julien TRÉBOSC
	# original code : https://github.com/jtrebosc/JTutils/blob/master/TSpy/qcpmgadd.py
	dat=dataset[:] # make a copy because I don't want to modify the original array
	if len(dat)==5: # for topspin 2-
	        dat[3]="%s/data/%s/nmr" % (dat[3],dat[4])
	#fulldata="%s/%s/%s/pdata/%s/" % (dat[3],dat[0],dat[1],dat[2])
	fulldata="%s/%s/%s/" % (dat[3],dat[0],dat[1])
	return fulldata

fulldataPATH=fullpath(dataset)
print(fulldataPATH)

subprocess.call([CPYTHON_LOCATION]+[script]+[fulldataPATH]+[fullEcho]+[nbEcho]+[FD])