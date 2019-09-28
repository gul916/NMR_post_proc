#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import nmrglue as ng
import numpy as np
import sys


def plotting(dic, data):
    fig = plt.figure()
    plt.rc('image', cmap='viridis')                 # color map
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

def test_nmrglue(data_dir):
    # Import data
    dic, data = ng.bruker.read(data_dir)
    if data.ndim == 1:
        # 1D processing
        data = ng.bruker.remove_digital_filter(dic, data)   # digital filtering
        data = ng.proc_base.sp(data, off=0.5, end=1.0, pow=1.0) # apodization
        data = ng.proc_base.zf_size(data, dic['procs']['SI'])   # zero-filling
        data[0] /= 2                                        # normalization
        data = ng.proc_base.fft_norm(data)                  # FFT with norm
        data = ng.proc_base.rev(data)                       # revert spectrum
        data = ng.proc_autophase.autops(data, 'acme')       # autophasing
        # Write data
        # Data should have exatly the original processed size (bug #109)
        ng.bruker.write_pdata(
            data_dir, dic, data.real*8,
            scale_data=True, bin_file='1r', overwrite=True)
        ng.bruker.write_pdata(
            data_dir, dic, data.imag*8,
            scale_data=True, bin_file='1i', overwrite=True)
    elif data.ndim == 2:
        # Direct dimension processing
        data = ng.bruker.remove_digital_filter(dic, data)   # digital filtering
        data = ng.proc_base.sp(data, off=0.5, end=1.0, pow=1.0) # apodization
        data = ng.proc_base.zf_size(data, dic['procs']['SI'])   # zero-filling
        data[:, 0] /= 2                                     # normalization
        data = ng.proc_base.fft_norm(data)                  # FFT with norm
        data = ng.proc_base.rev(data)                       # revert spectrum
        data = ng.proc_autophase.autops(data, 'acme')       # autophasing
        # Indirect dimension processing
        data = ng.proc_base.tp_hyper(data)          # hypercomplex transpose
        data = ng.proc_base.sp(data, off=0.5, end=1.0, pow=1.0) # apodization
        data = ng.proc_base.zf_size(data, dic['proc2s']['SI'])  # zero-filling
        data[:, 0] /= 2                                     # normalization
        data = ng.proc_base.fft_norm(data)                  # FFT with norm
        if dic['acqu2s']['FnMODE'] == 4:                    # STATES
            pass
        elif dic['acqu2s']['FnMODE'] == 5:                  # STATES-TPPI
            data = np.fft.fftshift(data, axes=-1)           # swap spectrum
        data = ng.proc_base.rev(data)                       # revert spectrum
        data = ng.proc_base.tp_hyper(data)          # hypercomplex transpose
        datarr = data[::2,:].real
        datari = data[1::2,:].real
        datair = data[::2,:].imag
        dataii = data[1::2,:].imag
        # Write data
        # Data should have exatly the original processed size (bug #109)
        ng.bruker.write_pdata(
            data_dir, dic, datarr*8,
            scale_data=True, bin_file='2rr', overwrite=True)
        ng.bruker.write_pdata(
            data_dir, dic, datari*8,
            scale_data=True, bin_file='2ri', overwrite=True)
        ng.bruker.write_pdata(
            data_dir, dic, datair*8,
            scale_data=True, bin_file='2ir', overwrite=True)
        ng.bruker.write_pdata(
            data_dir, dic, dataii*8,
            scale_data=True, bin_file='2ii', overwrite=True)
    else:
        raise NotImplementedError(
            "Data of", data.ndim, "dimensions are not yet supported.")
    plotting(dic, data)


def main():
    try:
        if len(sys.argv) == 1:
            raise NotImplementedError(
                "Please enter the directory of the Bruker file.")
        elif len(sys.argv) == 2:
            data_dir = sys.argv[1]
            test_nmrglue(data_dir)
        elif len(sys.argv) >= 3:
            raise NotImplementedError("There should be only one argument.")
    
    except NotImplementedError as err:
        print("Error:", err)
        for i in range(0, len(sys.argv)):
            print("Argument", i, '=', sys.argv[i])
    except OSError as err:
        print("Error:", err)
    else:                                           # When no error occured
        print("\nNMRglue was successfully tested")

#%%----------------------------------------------------------------------------
### When this file is executed directly
###----------------------------------------------------------------------------

if __name__ == "__main__":
    main()