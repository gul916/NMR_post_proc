#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Guillaume Laurent & Pierre-Aymeric Gilles
denoise_nmr.py
denoise() function provides denoising with Singular Value Decomposition (SVD)
and thresholding of a one- or two-dimensional NMR FID (time domain).

Needs nmrglue.

G. Laurent, P.-A. Gilles, W. Woelffel, V. Barret-Vivin, E. Gouillart, et C. Bonhomme,
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
# Needs file svd_auto.py with scikit-cuda, pycuda, CUDA toolkit and CULA library
import svd_auto

# Default values
k_thres = 0     # if 0, allows automatic thresholding
                # from 1 to min(row, col): manual threshold (integer)
max_err = 7.5   # error level for automatic thresholding, from 5 to 10 % (float)

#%%----------------------------------------------------------------------------
### IMPORT AND EXPORT DATA
###----------------------------------------------------------------------------
def import_data(data_dir, data_den_dir):
    """
    Import data
    Usage:  dic, data, data_den_dir = import_data(data_dir, data_den_dir)
    Input:  data_dir     data directory (string)
    Output: dic          data parameters (dictionary)
            data         imported data (array)
    """
    print('\nNoisy data {:s}'.format(data_dir))
    dic, data = ng.bruker.read(data_dir)                # import data
    if data.ndim >= 3:
        raise NotImplementedError \
            ('Data of {:d} dimensions is not yet supported'.format(data.ndim))
    data_den_dir = export_dir(data_dir, data_den_dir)
    return dic, data, data_den_dir

def export_dir(data_dir, data_den_dir):
    """
    Check if output directory exists
    Usage:  data_den_dir = export_dir(data_dir, data_den_dir)
    Input:  data_dir     original data directory (string)
            data_den_dir denoised data directory (string or None)
    Output: data_den_dir modified denoised data directory (string)
    """
    if data_den_dir == None:
        if data_dir[-1] in ['/', '\\']:
            data_den_dir = data_dir[:-1] + '_denoised/' # remove last character
        else:
            data_den_dir = data_dir[:] + '_denoised'
        while os.path.isdir(data_den_dir):
            print('\n{:s} directory exists'.format(data_den_dir))
            overwrite = input('Overwrite denoised data (y/n) ? ')
            if overwrite in ['Y', 'y']:
                overwrite = True
                break
            if overwrite in ['N', 'n']:
                raise ValueError ('Overwritten not allowed')
    return data_den_dir

def export_data(dic, data_den, data_den_dir):
    """
    Export data to file
    Usage:  export_data(data_dir, dic, data, overw)
    Input:  dic          data parameters (dictionary)
            data_den     data to export (array)
            data_den_dir data directory (string)
    Output: none
    """
    ng.bruker.write(data_den_dir, dic, data_den, overwrite=True)
    print('\nDenoised data saved to', data_den_dir)

#%%----------------------------------------------------------------------------
### PROCESSING
###----------------------------------------------------------------------------
def apod(data):
    """
    Apply cosine apodisation
    Usage:  data_apod = apod(data)
    Input:  data        data to process (array)
    Output: data_apod   data apodized (array)
    """
    data_apod = ng.proc_base.sp(data, off=0.5, end=1.0, pow=1.0)
    if data.ndim == 2:                                  # 2D data set
        data_apod = ng.proc_base.tp_hyper(data_apod)    # hypercomplex transpose
        data_apod = ng.proc_base.sp(data_apod, off=0.5, end=1.0, pow=1.0)
        data_apod = ng.proc_base.tp_hyper(data_apod)    # hypercomplex transpose
    return data_apod

def spc(dic, data_fid):
    """
    FFT of FID with normalization, zero-filling and phasing
    Usage: data_spc = spc(dic, data_fid)
    Input:  dic          data parameters (dictionary)
            data_fid     data to transform
    Output: data_spc     transformed data
    """
    data_fid1 = data_fid[:]                             # avoid corruption
    data_fid1 = ng.proc_base.zf_auto(data_fid1)         # zero-filling 2^n
    data_fid1 = ng.proc_base.zf_double(data_fid1, 2)    # zero-filling *4
    if data_fid1.ndim == 1:                             # 1D data set
        data_fid1[0] = data_fid1[0] / 2                 # normalization
        data_spc = ng.proc_base.fft_norm(data_fid1)     # FFT with norm
        data_spc = ng.proc_base.ps(data_spc, \
            dic['procs']['PHC0'], dic['procs']['PHC1'], True)   # phasing
    elif data_fid1.ndim == 2:                           # 2D data set
        # First dimension
        data_fid1[:, 0] = data_fid1[:, 0] / 2           # normalization
        data_spc = ng.proc_base.fft_norm(data_fid1)     # FFT with norm
        data_spc = ng.proc_base.ps(data_spc, \
            dic['procs']['PHC0'], dic['procs']['PHC1'], True)   # phasing
        # Second dimension
        data_spc = ng.proc_base.tp_hyper(data_spc)      # hypercomplex transpose
        data_spc = ng.proc_base.zf_auto(data_spc)       # zero-filling 2^n
        data_spc = ng.proc_base.zf_double(data_spc, 2)  # zero-filling *4
        data_spc[:, 0] = data_spc[:, 0] / 2             # normalization
        data_spc = ng.proc_base.fft_norm(data_spc)      # FFT with norm
        if dic['acqu2s']['FnMODE'] == 4:                # STATES
            pass
        elif dic['acqu2s']['FnMODE'] == 5:              # STATES-TPPI
            data_spc = np.fft.fftshift(data_spc, axes=-1)       # swap spectrum
        data_spc = ng.proc_base.ps(data_spc, \
            dic['proc2s']['PHC0'], dic['proc2s']['PHC1'], True) # phasing
        data_spc = ng.proc_base.tp_hyper(data_spc)      # hypercomplex transpose
    return data_spc

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

def denoise(data, k_thres, max_err):
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
    Output: none
    """
    fig = plt.figure()
    data_spc = spc(dic, data_apod)                      # FFT and phasing
    data_den_spc = spc(dic, data_den)
    if data_apod.ndim == 1:                             # 1D data set
        # FID scale
        udic = ng.bruker.guess_udic(dic, data_apod)     # universal dictionary
        x_scale_fid = ng.fileiobase.uc_from_udic(udic).ms_scale()
        min_y_fid, max_y_fid = [np.min(data_apod.real)*1.1, \
                                np.max(data_apod.real)*1.1]
        # SPC scale
        udic = ng.bruker.guess_udic(dic, data_spc)      # universal dictionary
        x_scale_spc = ng.fileiobase.uc_from_udic(udic).ppm_scale()
        min_y_spc, max_y_spc = [np.min(data_spc.real)*1.1, \
                                np.max(data_spc.real)*1.1]
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
        ax3.plot(x_scale_spc, data_spc.real)
        ax3.axis([x_scale_spc[0], x_scale_spc[-1], min_y_spc, max_y_spc])
        ax4 = fig.add_subplot(224)
        ax4.set_title('Denoised SPC, k = {:d}'.format(k_thres))
        ax4.set_xlabel('ppm')
        ax4.plot(x_scale_spc, data_den_spc.real)
        ax4.axis([x_scale_spc[0], x_scale_spc[-1], min_y_spc, max_y_spc])
    elif data_apod.ndim == 2:                           # 2D data set
        data_apod_real = data_apod[::2,:]   # remove interleaved imaginary part
        data_den_real = data_den[::2,:]
        data_spc_real = data_spc[::2,:]
        data_den_spc_real = data_den_spc[::2,:]
        nlev = 15
        colormap = 'viridis'                            # color map
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
        ax1.contour(x_scale_fid, y_scale_fid, data_apod_real.real, \
            levels_fid, cmap=colormap)
        ax2 = fig.add_subplot(222)
        ax2.set_title('Denoised FID, k = {:d}'.format(k_thres))
        ax2.set_xlabel('ms')
        ax2.set_ylabel('ms')
        ax2.contour(x_scale_fid, y_scale_fid, data_den_real.real, \
            levels_fid, cmap=colormap)
        ax3 = fig.add_subplot(223)
        ax3.set_title('Noisy SPC')
        ax3.set_xlabel('ppm')
        ax3.set_ylabel('ppm')
        ax3.contour(x_scale_spc, y_scale_spc, data_spc_real.real, \
            levels_spc, cmap=colormap)
        ax3.invert_xaxis()
        ax3.invert_yaxis()
        ax4 = fig.add_subplot(224)
        ax4.set_title('Denoised SPC, k = {:d}'.format(k_thres))
        ax4.set_xlabel('ppm')
        ax4.set_ylabel('ppm')
        ax4.contour(x_scale_spc, y_scale_spc, data_den_spc_real.real, \
            levels_spc, cmap=colormap)
        ax4.invert_xaxis()
        ax4.invert_yaxis()
    fig.tight_layout()
    plt.show()

#%%----------------------------------------------------------------------------
### MAIN FUNCTION
###----------------------------------------------------------------------------
def denoise_io(data_dir, data_den_dir, k_thres, max_err):
    """
    Import, denoise, plot and export data
    Usage:  denoise_io(data_dir)
            denoise_io(data_dir, k_thres=0, max_err=7.5)
    Input:  data_dir     directory of data to denoise (string)
                         ex: c:/Users/nmr/29Si/1
            data_den_dir directory of denoised data (string)
                         ex: c:/Users/nmr/29Si/1001
            k_thres      if 0, allows automatic thresholding
                         if > 0 and <= min(row, col), manual threshold (integer)
            max_err      error level for automatic thresholding
                         from 5 to 10 % (float)
    Output: none
    """
    try:
        # Import data and apply apodisation
        dic, data, data_den_dir = import_data(data_dir, data_den_dir)
        data_apod = apod(data)
        
        # Denoise data
        data_den, k_thres = denoise(data_apod, k_thres, max_err)
        
        # Plot original and denoised data
        plot_data(dic, data_apod, data_den, k_thres)
        
        # Export data
        export_data(dic, data_den, data_den_dir)
        
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
        # Directory for denoised data
        if (len(sys.argv) >= 3):
            data_den_dir = str(sys.argv[2])
        else:
            data_den_dir = None
        # Manual or authomatic threshold
        if len(sys.argv) >= 4:
            k_thres = int(sys.argv[3])
        # Automatic threshold error level
        if len(sys.argv) == 5:
            max_err = float(sys.argv[4])
        # Maximum number of arguments
        if len(sys.argv) > 5:
            raise ValueError \
                ('Supports only 4 arguments')
        
        # Import, denoise, plot and export data
        denoise_io(data_dir, data_den_dir, k_thres, max_err)
    
    except ValueError as err:                           # arguments
        print('Error:', err)
        for i in range(len(sys.argv)):
            print('Argument', i, '=', sys.argv[i])