#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# User defined libraries
import CPMG_gen as sigG
import CPMG_proc as sigP

if __name__ == "__main__":
	sig1 = sigG.signal_generation()
	denData = sigP.signal_processing(sig1)
	
	input('\nPress enter key to exit') # have the graphs stay displayed even when launched from linux terminal