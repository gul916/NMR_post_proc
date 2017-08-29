#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == "__main__":
	import CPMG_gen as sigG
	sig1 = sigG.signal_generation()
	import CPMG_proc as sigP
	denData = sigP.signal_processing(sig1)
	
	input('\nPress enter key to exit') # have the graphs stay displayed even when launched from linux terminal