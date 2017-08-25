#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nmrglue as ng

def load_data(path):
	parameters, signal = ng.bruker.read(path)
	return signal, parameters

def export_data(directory, parameters, data):
	ng.bruker.write(directory, parameters, data,overwrite=True)