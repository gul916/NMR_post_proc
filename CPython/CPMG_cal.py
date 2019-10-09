#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guillaume LAURENT
"""

# Python libraries
import matplotlib.pyplot as plt
# User defined libraries
import CPMG_proc

def plot_function(dic, data):
    plt.ion()                                   # to avoid stop when plotting
    CPMG_proc.echoes_figure(dic, data)          # plotting
    plt.ioff()                                  # to avoid figure closing
    plt.show()                                  # to allow zooming

def main():
    dic, _, FIDraw = CPMG_proc.data_import()                # importation
    FIDshift = CPMG_proc.shift_FID(dic, FIDraw)             # dead time
    FIDmat = CPMG_proc.echo_sep(dic, FIDshift)              # echoes separation
    dic = CPMG_proc.findT2(dic, FIDshift)                   # relaxation
    plot_function(dic, FIDmat)                              # plotting

if __name__ == "__main__":
    main()