#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Python libraries
import matplotlib.pyplot as plt
import nmrglue as ng
import numpy as np
import sys
# User defined library
import postproc

def plotting(dic, data):
    fig = plt.figure()
    if data.ndim == 1:
        # Scaling
        udic = ng.bruker.guess_udic(dic, data)      # Universal dictionary
        uc = ng.fileiobase.uc_from_udic(udic)
        ppm_scale = uc.ppm_scale()
        # Plotting
        ax1 = fig.add_subplot(121)
        ax1.plot(ppm_scale, data.real)
        ax1.invert_xaxis()
        ax2 = fig.add_subplot(122)
        ax2.plot(ppm_scale, data.imag)
        ax2.invert_xaxis()
    if data.ndim == 2:
        # Scaling
        datarr = data[::2,:].real
        datari = data[1::2,:].real
        datair = data[::2,:].imag
        dataii = data[1::2,:].imag
        udic = ng.bruker.guess_udic(dic, datarr)    # Universal dictionary
        uc_dim0 = ng.fileiobase.uc_from_udic(udic, dim=0)
        uc_dim1 = ng.fileiobase.uc_from_udic(udic, dim=1)
        ppm_dim0 = uc_dim0.ppm_scale()
        ppm_dim1 = uc_dim1.ppm_scale()
        lev0 = 0.1 * np.amax(datarr)
        toplev = 0.9 * np.amax(datarr)
        nlev = 15
        levels = np.geomspace(lev0, toplev, nlev)
        levels = np.concatenate((-levels[::-1], levels))
        # Plotting
        ax1 = fig.add_subplot(221)
        ax1.contour(ppm_dim1, ppm_dim0, datarr, levels)
        ax1.invert_yaxis()
        ax1.invert_xaxis()
        ax2 = fig.add_subplot(222)
        ax2.contour(ppm_dim1, ppm_dim0, datari, levels)
        ax2.invert_yaxis()
        ax2.invert_xaxis()
        ax3 = fig.add_subplot(223)
        ax3.contour(ppm_dim1, ppm_dim0, datair, levels)
        ax3.invert_yaxis()
        ax3.invert_xaxis()
        ax4 = fig.add_subplot(224)
        ax4.contour(ppm_dim1, ppm_dim0, dataii, levels)
        ax4.invert_yaxis()
        ax4.invert_xaxis()
    plt.tight_layout()
    plt.show()

def main():
    try:
        if len(sys.argv) == 1:
            raise NotImplementedError(
                "Please enter the directory of the Bruker file.")
        elif len(sys.argv) == 2:
            data_dir = sys.argv[1]
        elif len(sys.argv) >= 3:
            raise NotImplementedError("There should be only one argument.")
        
        dic, data = postproc.import_data(data_dir)
        data = postproc.preproc_data(dic, data)
        data = postproc.postproc_data(dic, data)
        postproc.export_data(dic, data, data_dir)
        plotting(dic, data)
    
    except NotImplementedError as err:
        print("Error:", err)
        for i in range(0, len(sys.argv)):
            print("Argument", i, '=', sys.argv[i])
    except OSError as err:
        print("Error:", err)
    else:                                           # When no error occured
        print("NMRglue was successfully tested:")
        print("importation, digital filtering, fft, autophasing, and saving")

#%%----------------------------------------------------------------------------
### When this file is executed directly
###----------------------------------------------------------------------------
if __name__ == "__main__":
    main()