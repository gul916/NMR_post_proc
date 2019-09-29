#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Guillaume Laurent & Pierre-Aymeric Gilles
denoise_nmr.py
denoise() function provides denoising with Singular Value Decomposition (SVD)
and thresholding of a one- or two-dimensional NMR FID (time domain).

Needs nmrglue.

G. Laurent, P.-A. Gilles, W. Woelffel, V. Barret-Vivin, E. Gouillart, and C. Bonhomme,
« Denoising applied to spectroscopies – Part II: Decreasing computation time »,
Appl. Spectrosc. Rev., 2019, doi: 10.1080/05704928.2018.1559851
"""

# Python libraries
import matplotlib.pyplot as plt
import nmrglue as ng
import numpy as np
import os.path
import scipy.linalg as sp_linalg
import sys
# User defined libraries
# svd_auto.py needs scikit-cuda, pycuda, CUDA toolkit and CULA library
import postproc
import svd_auto

# Default values
k_thres = 'auto'
max_err = 'auto'

#%%----------------------------------------------------------------------------
### DENOISE DATA
###----------------------------------------------------------------------------
def precision_single(data):
    """
    Convert data to single precision
    Usage:  data, typ = precision_single(data)
    Input:  data        data with original precision (array)
    Output: data        data with single precision (array)
            typ         original precision (string)
    """
    typ = data.dtype                                    # data type
    if typ in ['float32', 'complex64']:                 # single precision
        pass
    elif typ == 'float64':                      # convert to single float
        data = data.astype('float32')
    elif typ == 'complex128':                   # convert to single complex
        data = data.astype('complex64')
    else:
        raise ValueError('Unsupported data type')
    return data, typ

def precision_original(data, typ):
    """
    Revert data to original precision
    Usage:  data = precision_original(data, typ)
    Input:  data        data with single precision (array)
            typ         original precision (string)
    Output: data        data with original precision
    """
    data = data.astype(typ)
    return data

def vector_toeplitz(data):
    """
    Convert one-dimensional data to Toeplitz matrix
    Usage:  mat = vector_toeplitz(data)
    Input:  data        1D data (array)
    Output: mat         2D matrix (array)
    """
    row = int(np.ceil(data.size / 2))                   # almost square matrix
    # col = data.size - row + 1
    mat = sp_linalg.toeplitz(data[row-1::-1], data[row-1::1])
    return mat

def toeplitz_vector(mat):
    """
    Convert Toeplitz matrix to one-dimensional data
    Usage:  data = toeplitz_vector(mat)
    Input:  mat         2D matrix (array)
    Output: data        1D data (array)
    """
    row, col = mat.shape
    points = row+col-1
    data = np.zeros(points, dtype=mat.dtype)
    for i in range (0, points):
        data[i] = np.mean(np.diag(mat[:,:],i-row+1))
    return data

def denoise(data, k_thres='auto', max_err='auto'):
    """
    Denoise one- or two-dimensional data using Singular Value Decomposition
    Usage:  data_den, k_thres = denoise(data)
            data_den, k_thres = denoise(data, k_thres=0, max_err=7.5)
    Input:  data        noisy data (array)
            k_thres     if 0, allows automatic thresholding
                        if > 0 and <= min(row, col), manual threshold (integer)
            max_err     error level for automatic thresholding (float)
                        from 5 to 10 %
    Output: data_den    denoised data (array)
            k_thres     number of values used for thresholding
    """
    # Single precision to decrease computation time
    data, typ = precision_single(data)
    # SVD Denoising with thresholding
    if data.ndim == 1:          # convert to Toeplitz matrix and denoise
        mat = vector_toeplitz(data)
        mat_den, k_thres = svd_auto.svd_auto(mat, k_thres, max_err)
        data_den = toeplitz_vector(mat_den)
    elif data.ndim == 2:                                # denoise directly
        data_den, k_thres = svd_auto.svd_auto(data, k_thres, max_err)
    # Revert to original precision
    data_den = precision_original(data_den, typ)
    return data_den, k_thres

#%%----------------------------------------------------------------------------
### PLOT DATA
###----------------------------------------------------------------------------
def plot_data(dic, data_apod, data_den, k_thres):
    """
    Plot noisy and denoised data
    Usage:  plot_data(dic, data_apod, data_den, plot_value)
    Input:  dic          data parameters (dictionary)
            data_apod    apodised noisy data (array)
            data_den     apodised denoised data (array)
            k_thres      number of values used for thresholding
    Output: data_den     FFT of data_den
    """
    fig = plt.figure()
    data_apod_spc = postproc.postproc_data(dic, data_apod)  # FFT and phasing
    data_den_spc = postproc.postproc_data(dic, data_den)
    if data_apod.ndim == 1:                             # 1D data set
        # FID scale
        udic = ng.bruker.guess_udic(dic, data_apod)     # universal dictionary
        x_scale_fid = ng.fileiobase.uc_from_udic(udic).ms_scale()
        min_y_fid, max_y_fid = [
            np.min(data_apod.real)*1.1, np.max(data_apod.real)*1.1]
        # SPC scale
        udic = ng.bruker.guess_udic(dic, data_apod_spc) # universal dictionary
        x_scale_spc = ng.fileiobase.uc_from_udic(udic).ppm_scale()
        min_y_spc, max_y_spc = [
            np.min(data_apod_spc.real)*1.1, np.max(data_apod_spc.real)*1.1]
        # Figure
        ax1 = fig.add_subplot(221)
        ax1.set_title('Noisy FID')
        ax1.set_xlabel('ms')
        ax1.plot(x_scale_fid, data_apod.real)
        ax1.axis([x_scale_fid[0], x_scale_fid[-1], min_y_fid, max_y_fid])
        ax2 = fig.add_subplot(222)
        ax2.set_title('Denoised FID, k = {:d}'.format(k_thres))
        ax2.set_xlabel('ms')
        ax2.plot(x_scale_fid, data_den.real)
        ax2.axis([x_scale_fid[0], x_scale_fid[-1], min_y_fid, max_y_fid])
        ax3 = fig.add_subplot(223)
        ax3.set_title('Noisy SPC')
        ax3.set_xlabel('ppm')
        ax3.plot(x_scale_spc, data_apod_spc.real)
        ax3.axis([x_scale_spc[0], x_scale_spc[-1], min_y_spc, max_y_spc])
        ax4 = fig.add_subplot(224)
        ax4.set_title('Denoised SPC, k = {:d}'.format(k_thres))
        ax4.set_xlabel('ppm')
        ax4.plot(x_scale_spc, data_den_spc.real)
        ax4.axis([x_scale_spc[0], x_scale_spc[-1], min_y_spc, max_y_spc])
    elif data_apod.ndim == 2:                           # 2D data set
        data_apod_real = data_apod[::2,:]   # remove interleaved imaginary part
        data_den_real = data_den[::2,:]
        data_spc_real = data_apod_spc[::2,:]
        data_den_spc_real = data_den_spc[::2,:]
        nlev = 15
        # FID scale
        udic = ng.bruker.guess_udic(dic, data_apod_real)# universal dictionary
        x_scale_fid = ng.fileiobase.uc_from_udic(udic, dim=1).ms_scale()
        y_scale_fid = ng.fileiobase.uc_from_udic(udic, dim=0).ms_scale()
        lev0_fid = 0.1 * np.amax(data_apod_real.real)
        toplev_fid = 0.9 * np.amax(data_apod_real.real)
        levels_fid = np.geomspace(lev0_fid, toplev_fid, nlev)
        levels_fid = np.concatenate((-levels_fid[-1::-1], levels_fid))
        # SPC scale
        udic = ng.bruker.guess_udic(dic, data_spc_real) # universal dictionary
        x_scale_spc = ng.fileiobase.uc_from_udic(udic, dim=1).ppm_scale()[::-1]
        y_scale_spc = ng.fileiobase.uc_from_udic(udic, dim=0).ppm_scale()[::-1]
        lev0_spc = 0.1 * np.amax(data_spc_real.real)
        toplev_spc = 0.9 * np.amax(data_spc_real.real)
        levels_spc = np.geomspace(lev0_spc, toplev_spc, nlev)
        levels_spc = np.concatenate((-levels_spc[-1::-1], levels_spc))
        # Figure
        ax1 = fig.add_subplot(221)
        ax1.set_title('Noisy FID')
        ax1.set_xlabel('ms')
        ax1.set_ylabel('ms')
        ax1.contour(x_scale_fid, y_scale_fid, data_apod_real.real, levels_fid)
        ax2 = fig.add_subplot(222)
        ax2.set_title('Denoised FID, k = {:d}'.format(k_thres))
        ax2.set_xlabel('ms')
        ax2.set_ylabel('ms')
        ax2.contour(x_scale_fid, y_scale_fid, data_den_real.real, levels_fid)
        ax3 = fig.add_subplot(223)
        ax3.set_title('Noisy SPC')
        ax3.set_xlabel('ppm')
        ax3.set_ylabel('ppm')
        ax3.contour(x_scale_spc, y_scale_spc, data_spc_real.real, levels_spc)
        ax3.invert_xaxis()
        ax3.invert_yaxis()
        ax4 = fig.add_subplot(224)
        ax4.set_title('Denoised SPC, k = {:d}'.format(k_thres))
        ax4.set_xlabel('ppm')
        ax4.set_ylabel('ppm')
        ax4.contour(
            x_scale_spc, y_scale_spc, data_den_spc_real.real, levels_spc)
        ax4.invert_xaxis()
        ax4.invert_yaxis()
    fig.tight_layout()
    plt.show()
    return data_den_spc

#%%----------------------------------------------------------------------------
### MAIN FUNCTION
###----------------------------------------------------------------------------
def denoise_io(data_dir, k_thres='auto', max_err='auto'):
    """
    Import, denoise, plot and export data
    Usage:  denoise_io(data_dir)
            denoise_io(data_dir, k_thres=0, max_err=7.5)
    Input:  data_dir     directory of data to denoise (string)
                         ex: c:/Users/nmr/29Si/1/pdata/1
            k_thres      if 0, allows automatic thresholding
                         if > 0 and <= min(row, col), manual threshold (integer)
            max_err      error level for automatic thresholding
                         from 5 to 10 % (float)
    Output: none
    """
    try:
        # Import data and apply apodisation
        dic, data = postproc.import_data(data_dir)
        data_apod = postproc.preproc_data(dic, data)
        
        # Denoise data, Fourier transform, and plot
        data_den, k_thres = denoise(data_apod, k_thres, max_err)
        data_den = plot_data(dic, data_apod, data_den, k_thres)
        
        # Export data
        postproc.export_data(dic, data_den, data_dir)
        
    except KeyError as err:                             # writing data
        print('Error: unable to write data. Missing key:', err)
    except ImportError as err:                          # libraries
        print('Error:', err)
    except NotImplementedError as err:                  # data dimensions
        print('Error:', err)
    except OSError as err:                              # file system access
        print('Error:', err)

#%%----------------------------------------------------------------------------
### IF PROGRAM IS DIRECTLY EXECUTED
###----------------------------------------------------------------------------
if __name__ == '__main__':
    # Print arguments
    try:
        # Data directory has to be provided as an argument
        if (len(sys.argv) >= 2) and (os.path.isdir(sys.argv[1])):
            data_dir = str(sys.argv[1])
        else:
            raise ValueError \
                ('Please provide data directory as first argument')
        # Manual or authomatic threshold
        if len(sys.argv) >= 3:
            k_thres = int(sys.argv[2])
        # Automatic threshold error level
        if len(sys.argv) >= 4:
            max_err = float(sys.argv[3])
        # Maximum number of arguments
        if len(sys.argv) >= 5:
            raise ValueError \
                ('Supports only 3 arguments')
        
        # Import, denoise, plot and export data
        denoise_io(data_dir, k_thres, max_err)
    
    except ValueError as err:                           # arguments
        print('Error:', err)
        for i in range(len(sys.argv)):
            print('Argument', i, '=', sys.argv[i])