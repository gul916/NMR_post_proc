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

def import_data(data_dir, data_den_dir):
    """
    Import data
    Usage:  dic, data = import_data(data_dir, overw)
    Input:  data_dir     data directory (string)
    Output: dic          data parameters (dictionary)
            data         imported data (array)
    """
    print('\nNoisy data {:s}\n'.format(data_dir))
    dic, data = ng.bruker.read(data_dir)                # import data
    data_den_dir = export_dir(data_dir, data_den_dir)
    return dic, data, data_den_dir

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
    else:
        raise NotImplementedError \
        ('Data of {:d} dimensions is not yet supported'.format(data.ndim))
    # Revert to original precision
    data_den = precision_original(data_den, typ)
    return data_den, k_thres

#%%----------------------------------------------------------------------------
### PLOT DATA
###----------------------------------------------------------------------------
def plot_data(data, data_den, k_thres):
    """
    Plot noisy and denoised data, if needed
    Usage:  plot_data(data, data_den, plot_value)
    Input:  data        noisy data (array)
            data_den    denoised data (array)
            k_thres     number of values used for thresholding
    Output: none
    """
    fig = plt.figure()
    if data.ndim == 1:                              # 1D data set
        data_spc = np.fft.fftshift(np.fft.fft(data))
        data_den_spc = np.fft.fftshift(np.fft.fft(data_den))
        min_x, max_x = [0, data.size-1]
        min_y_fid, max_y_fid = [np.min(data.real)*1.1, \
                                np.max(data.real)*1.1]
        min_y_spc, max_y_spc = [np.min(data_spc.real)*1.1, \
                                np.max(data_spc.real)*1.1]
        ax1 = fig.add_subplot(221)
        ax1.set_title('Noisy FID')
        ax1.plot(data.real)
        ax1.axis([min_x, max_x, min_y_fid, max_y_fid])
        ax2 = fig.add_subplot(222)
        ax2.set_title('Denoised FID, k = {:d}'.format(k_thres))
        ax2.plot(data_den.real)
        ax2.axis([min_x, max_x, min_y_fid, max_y_fid])
        ax3 = fig.add_subplot(223)
        ax3.set_title('Noisy SPC')
        ax3.plot(data_spc.real)
        ax3.axis([max_x, min_x, min_y_spc, max_y_spc])
        ax4 = fig.add_subplot(224)
        ax4.set_title('Denoised SPC, k = {:d}'.format(k_thres))
        ax4.plot(data_den_spc.real)
        ax4.axis([max_x, min_x, min_y_spc, max_y_spc])
    elif data.ndim == 2:                            # 2D data set
        data_spc = np.fft.fftshift(np.fft.fft \
            (np.fft.fft(data, axis=1), axis=0))
        data_den_spc = np.fft.fftshift(np.fft.fft \
            (np.fft.fft(data_den, axis=1), axis=0))
        lev0_fid = 0.1 * np.amax(data.real)
        toplev_fid = 0.9 * np.amax(data.real)
        lev0_spc = 0.1 * np.amax(data_spc.real)
        toplev_spc = 0.9 * np.amax(data_spc.real)
        nlev = 15
        levels_fid = np.geomspace(lev0_fid, toplev_fid, nlev)
        levels_spc = np.geomspace(lev0_spc, toplev_spc, nlev)
        ax1 = fig.add_subplot(221)
        ax1.set_title('Noisy FID')
        ax1.contour(data.real, levels_fid)
        ax2 = fig.add_subplot(222)
        ax2.set_title('Denoised FID, k = {:d}'.format(k_thres))
        ax2.contour(data_den.real, levels_fid)
        ax3 = fig.add_subplot(223)
        ax3.set_title('Noisy SPC')
        ax3.contour(data_spc.real, levels_spc)
        ax3.invert_xaxis()
        ax3.invert_yaxis()
        ax4 = fig.add_subplot(224)
        ax4.set_title('Denoised SPC, k = {:d}'.format(k_thres))
        ax4.contour(data_den_spc.real, levels_spc)
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
        # Import data with single precision to decrease computation time
        dic, data, data_den_dir = import_data(data_dir, data_den_dir)
        
        # Denoise data
        data_den, k_thres = denoise(data, k_thres, max_err)
        
        # Plot original and denoised data
        plot_data(data, data_den, k_thres)
        
        # Export data with original precision
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