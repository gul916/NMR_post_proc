#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guillaume LAURENT
"""

# Python libraries
import matplotlib.pyplot as plt
# User defined libraries
import CPMG_proc

def plot_function(dic, D):
    plt.ion()                                   # to avoid stop when plotting
    CPMG_proc.echoes_figure(dic, D)             # plotting
    plt.ioff()                                  # to avoid figure closing
    plt.show()                                  # to allow zooming

def main():
    dic, _, FIDraw = CPMG_proc.data_import()                # importation
    dic, FIDshift = CPMG_proc.shift_FID(dic, FIDraw)        # dead time
    dic, FIDapod = CPMG_proc.echo_apod(
        dic, FIDshift, method='exp')                        # echoes apod
    dic = CPMG_proc.findT2(dic, FIDapod)                    # relaxation
    plot_function(dic, FIDapod)                             # plotting

if __name__ == "__main__":
    main()