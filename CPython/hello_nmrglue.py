#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import nmrglue as ng
import numpy as np
import sys

def test_nmrglue(data_dir):
    # Import data
    dic, data = ng.bruker.read(data_dir)

    if data.ndim == 1:
        # 1D processing
        data = ng.bruker.remove_digital_filter(dic, data)   # digital filtering
        data = ng.proc_base.sp(data, off=0.5, end=1.0, pow=1.0) # apodization
        data = ng.proc_base.zf_auto(data)                   # zero-filling 2^n
        data = ng.proc_base.zf_double(data, 2)              # zero-filling *4
        data[0] /= 2                                        # normalization
        data = ng.proc_base.fft_norm(data)                  # FFT with norm
        data = ng.proc_base.rev(data)                       # revert spectrum
        data = ng.proc_autophase.autops(data, 'acme')       # autophasing
        # Scaling
        udic = ng.bruker.guess_udic(dic, data)          # Universal dictionary
        uc = ng.fileiobase.uc_from_udic(udic)
        ppm_scale = uc.ppm_scale()
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ppm_scale, data.real)
        ax.invert_xaxis()
        # Write data
        ng.bruker.write_pdata(data_dir, dic, data, overwrite=True)
    
    elif data.ndim == 2:
        # Direct dimension processing
        data = ng.bruker.remove_digital_filter(dic, data)   # digital filtering
        data = ng.proc_base.sp(data, off=0.5, end=1.0, pow=1.0) # apodization
        data = ng.proc_base.zf_auto(data)                   # zero-filling 2^n
        data = ng.proc_base.zf_double(data, 2)              # zero-filling *4
        data[:, 0] /= 2                                     # normalization
        data = ng.proc_base.fft_norm(data)                  # FFT with norm
        data = ng.proc_base.rev(data)                       # revert spectrum
        data = ng.proc_autophase.autops(data, 'acme')       # autophasing
        # Indirect dimension processing
        data = ng.proc_base.tp_hyper(data)              # hypercomplex transpose
        data = ng.proc_base.sp(data, off=0.5, end=1.0, pow=1.0) # apodization
        data = ng.proc_base.zf_auto(data)                   # zero-filling 2^n
        data = ng.proc_base.zf_double(data, 2)              # zero-filling *4
        data[:, 0] /= 2                                     # normalization
        data = ng.proc_base.fft_norm(data)                  # FFT with norm
        if dic['acqu2s']['FnMODE'] == 4:                    # STATES
            pass
        elif dic['acqu2s']['FnMODE'] == 5:                  # STATES-TPPI
            data = np.fft.fftshift(data, axes=-1)           # swap spectrum
        data = ng.proc_base.rev(data)                       # revert spectrum
        data = ng.proc_autophase.autops(data, 'acme')       # autophasing
        data = ng.proc_base.tp_hyper(data)              # hypercomplex transpose
        # Scaling
        udic = ng.bruker.guess_udic(dic, data)          # Universal dictionary
        uc_dim0 = ng.fileiobase.uc_from_udic(udic, dim=0)
        ppm_dim0 = uc_dim0.ppm_scale()
        uc_dim1 = ng.fileiobase.uc_from_udic(udic, dim=1)
        ppm_dim1 = uc_dim1.ppm_scale()
        lev0 = 0.1 * np.amax(data.real)
        toplev = 0.9 * np.amax(data.real)
        nlev = 15
        levels = np.geomspace(lev0, toplev, nlev)
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.contour(ppm_dim1, ppm_dim0, data.real, levels)
        ax.invert_yaxis()
        ax.invert_xaxis()
        # Write data
#        ng.bruker.write_pdata(data_dir, dic, data, overwrite=True)
        print('Saving 2D processed data is not working (bug reported)')
    else:
        raise NotImplementedError(
            "Data of", data.ndim, "dimensions are not yet supported.")
    
    plt.ioff()                                      # Interactive mode off
    plt.show()

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
        print("NMRglue successfully tested")

#%%----------------------------------------------------------------------------
### When this file is executed directly
###----------------------------------------------------------------------------

if __name__ == "__main__":
    main()