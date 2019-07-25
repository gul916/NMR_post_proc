#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Guillaume Laurent & Pierre-Aymeric Gilles
denoise_nmr.py
denoise() function provides denoising with Singular Value Decomposition (SVD)
and thresholding of a one- or two-dimensional NMR FID (time domain).

Needs nmrglue to import and export Bruker data.

G. Laurent, P.-A. Gilles, W. Woelffel, V. Barret-Vivin, E. Gouillart, et C. Bonhomme,
« Denoising applied to spectroscopies – Part II: Decreasing computation time »,
Appl. Spectrosc. Rev., 2019, doi: 10.1080/05704928.2018.1559851
"""

# Python libraries
import matplotlib.pyplot as plt
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
def check_overwrite(data_dir):
    """
    Check if output directory exists
    Usage:  data_dir, overwrite = check_overwrite(data_dir)
    Input:  data_dir    data directory (string)
    Output: data_dir    modified data directory (string)
            overwrite   if true, overwrite denoised data (boolean)
    """
    overwrite = False
    if data_dir[-1] in ['/', '\\']:
        data_dir = data_dir[:-1]
    data_dir_den = data_dir + '_denoised'
    print('\n--------------------------------------------------')
    while os.path.isdir(data_dir_den):
        print('{:s} directory exists'.format(data_dir_den))
        overwrite = input('Overwrite denoised data (y/n) ? ')
        if overwrite in ['Y', 'y']:
            overwrite = True
            break
        if overwrite in ['N', 'n']:
            overwrite = False
            break
    else:
        overwrite = True
    return data_dir, overwrite

def import_data(data_dir):
    """
    Import data
    Usage:  dic, data = import_data(data_dir)
    Input:  data_dir    data directory (string)
    Output: dic         data parameters (dictionary)
            data        imported data (array)
    """
    global ng
    import nmrglue as ng                                # nmrglue
    dic, data = ng.bruker.read(data_dir)                # import data
    return dic, data

def export_data(data_dir, dic, data, overw):
    """
    Export data to file
    Usage:  export_data(data_dir, dic, data, overw)
    Input:  data_dir    data directory (string)
            dic         data parameters (dictionary)
            data        data to export (array)
            overw       if True, overwrite original data (boolean)
    Output: none
    """
    data_dir_den = data_dir + '_denoised'
    if overwrite == True:                               # export data
        ng.bruker.write(data_dir_den, dic, data, overwrite=overw)
        print('Data saved to', data_dir_den)
    else:
        print('\nData not saved, overwrite = {:s}'.format(str(overwrite)))

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

#%%----------------------------------------------------------------------------
### DENOISE DATA
###----------------------------------------------------------------------------
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

def denoise_mat(data, k_thres, max_err):
    """
    Denoise one- or two-dimensional data using Singular Value Decomposition
    Usage:  data_den, k_thres = denoise_mat(data)
    Input:  data        noisy data (array)
            k_thres     if 0, allows automatic thresholding
                        if > 0 and <= min(row, col), manual threshold (integer)
            max_err     error level for automatic thresholding (float)
                        from 5 to 10 %
    Output: data_den    denoised data (array)
            k_thres     number of values used for thresholding
    """
    if data.ndim == 1:          # convert to Toeplitz matrix and denoise
        mat = vector_toeplitz(data)
        mat_den, k_thres = svd_auto.svd_auto(mat, k_thres, max_err)
        data_den = toeplitz_vector(mat_den)
    elif data.ndim == 2:                                # denoise directly
        data_den, k_thres = svd_auto.svd_auto(data, k_thres, max_err)
    else:
        raise NotImplementedError \
        ('Data of {:d} dimensions is not yet supported'.format(data.ndim))
    return data_den, k_thres

#%%----------------------------------------------------------------------------
### PLOT DATA
###----------------------------------------------------------------------------
def plot_data(data, data_den, k_thres, plot_value):
    """
    Plot noisy and denoised data, if needed
    Usage:  plot_data(data, data_den, plot_value)
    Input:  data        noisy data (array)
            data_den    denoised data (array)
            k_thres     number of values used for thresholding
            plot_value  if True, plot noisy and denoised data (boolean)
    Output: none
    """
    if plot_value == True:
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
            lev0 = 0.01 * np.amax(data.real)
            toplev = 0.99 * np.amax(data.real)
            nlev = 15
            levels = np.linspace(lev0, toplev, nlev)
            ax1 = fig.add_subplot(221)
            ax1.contour(data.real, levels)
            ax1.set_title('Noisy FID')
            ax2 = fig.add_subplot(222)
            ax2.contour(data_den.real, levels)
            ax2.set_title('Denoised FID, k = {:d}'.format(k_thres))
            ax3 = fig.add_subplot(223)
            ax3.contour(abs(np.fft.fftshift(np.fft.fft(data))), levels)
            ax3.invert_xaxis()
            ax3.invert_yaxis()
            ax3.set_title('Noisy SPC')
            ax4 = fig.add_subplot(224)
            ax4.contour(abs(np.fft.fftshift(np.fft.fft(data_den))), levels)
            ax4.invert_xaxis()
            ax4.invert_yaxis()
            ax4.set_title('Denoised SPC, k = {:d}'.format(k_thres))
        fig.tight_layout()
        plt.show()

#%%----------------------------------------------------------------------------
### MAIN FUNCTION
###----------------------------------------------------------------------------
def denoise(data_dir, k_thres, max_err, overwrite=False, plot_value=False):
    """
    Import, denoise, plot and export data
    Usage:  denoise(data_dir)
            denoise(data_dir, plot_value=True, overwrite=True)
    Input:  data_dir    directory of data to denoise (string)
                        ex: c:/Users/nmr/29Si/1
            overwrite   if True, overwrite denoised data (boolean)
            k_thres     if 0, allows automatic thresholding
                        if > 0 and <= min(row, col), manual threshold (integer)
            max_err     error level for automatic thresholding
                        from 5 to 10 % (float)
            plot_value  if True, plot noisy and denoised data (boolean)
    Output: none
    """
    try:
        # Import data with single precision to decrease computation time
        dic, data = import_data(data_dir)
        data, typ = precision_single(data)
        
        # Denoise data
        data_den, k_thres = denoise_mat(data, k_thres, max_err)
        
        # Plot original and denoised data
        plot_data(data, data_den, k_thres, plot_value)
        
        # Export data with original precision
        data_den = precision_original(data_den, typ)
        export_data(data_dir, dic, data_den, overwrite)
                
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
        if len(sys.argv) == 4:
            max_err = float(sys.argv[3])
        # Maximum number of arguments
        if len(sys.argv) > 4:
            raise ValueError \
                ('Supports only 3 arguments: data_dir, k_thres and max_err')
        
        # Denoise and check if overwritten is allowed
        data_dir, overwrite = check_overwrite(data_dir)
        denoise(data_dir, k_thres, max_err, overwrite, plot_value=True)
    
    except ValueError as err:                           # arguments
        print('Error:', err)
        for i in range(1, len(sys.argv)):
            print('Argument', i, '=', sys.argv[i])
