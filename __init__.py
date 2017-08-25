#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == "__main__":
	import signal_generation as sigG
	sig1 = sigG.signal_generation()
	import signal_processing as sigP
	denData = sigP.signal_processing(sig1)