#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Guillaume Laurent & Pierre-Aymeric Gilles
svd_auto.py
svd_auto() function provides Singular Value Decomposition (SVD)
with automatic low-rank approximation on processor (SciPy)
and on NVIDIA graphic card (scikit-cuda).

G. Laurent, P.-A. Gilles, W. Woelffel, V. Barret-Vivin, E. Gouillart, et C. Bonhomme,
« Denoising applied to spectroscopies – Part II: Decreasing computation time »,
Appl. Spectrosc. Rev., 2019, doi: 10.1080/05704928.2018.1559851
"""

import numpy as np
import time
# CPU library
import scipy.linalg as sp_linalg
# GPU libraries
try:
    import pycuda.autoinit                          # needed
    import pycuda.gpuarray as gpuarray
    import skcuda.linalg as cu_linalg
    cu_linalg.init()                                # needed
except:
    lib = 'scipy'

# Default value
cpu_gpu_lim = 4096  # Number of columns to switch denoising hardware.
                    # Under this value, processor is used.
                    # Starting with this value, graphic card is used
                    
                    # Run this file directly and adjust this value
                    # accordingly to your hardware

#%%----------------------------------------------------------------------------
### PRE-, POST-PROCESSING AND LIBRARY IMPORT
###----------------------------------------------------------------------------
def pre_svd(X):
    """
    Check and transpose X matrix if needed before SVD
    Usage:  X, transp, typ = pre_svd(X)
    Input:  X           matrix to decompose (array)
    Output: X           transposed matrix, if needed (array)
            transp      if True, transposition was applied (boolean)
    """
    if X.ndim != 2:                                     # wrong data dimensions
        raise NotImplementedError('Data should be a 2D matrix.')
    m, n = X.shape
    if(m < n):                                  # transpose if needed for GPU
        X = X.transpose()
        transp = True
    else:
        transp = False
    return X, transp

def lib_svd(X, lib):
    """
    Import libraries needed for SVD
    Usage:  lib, typ = lib_svd(X, lib)
    Input:  X           matrix to decompose (array)
            lib         library to use, 'auto', 'scipy' or 'skcuda' (string)
    Output: lib         selected library (string)
            typ         data type (string)
    """
    m, n = X.shape
    typ = X.dtype                                       # check data type
    if lib == 'auto':
        if min(m, n) < cpu_gpu_lim:                     # prefer CPU
            lib = 'scipy'
        else:                                           # prefer GPU
            lib = 'skcuda'
    if lib == 'scipy':                                  # use CPU
        print('\t{:5} x {:5} points - Scipy - {:8s}' \
            .format(m, n, str(np.dtype(typ))))
    elif lib == 'skcuda':                               # use GPU
        print('\t{:5} x {:5} points - skCuda - {:8s}' \
            .format(m, n, str(np.dtype(typ))))
    else:                                               # no library
        raise ImportError('Unknown SVD library')
    return lib, typ

def post_svd(X, Xk, transp, disp_err):
    """
    Transpose Xk matrix if needed after SVD
    Usage:  Xk = post_svd(X, Xk, transp, disp_err)
    Input:  X           original matrix (array)
            Xk          recovered matrix (array)
            transp      if True, transposition was applied (boolean)
            disp_err    if True, display relative error (boolean)
    Output: Xk          transposed recovered matrix, if needed (array)
    """
    if transp == True:
        X = X.transpose()
        Xk = Xk.transpose()
    if isinstance(Xk, np.ndarray) and (disp_err==True): # Only if array exists
        print('Maximum relative error:\t\t{0:8.2e}'\
            .format(np.max(np.abs(Xk-X)) / np.max(np.abs(X))))
        print('Root mean squared error:\t{0:8.2e}'\
            .format(np.sqrt(((Xk - X) ** 2).mean()).real))
        print('l2-norm:\t\t\t{0:8.2e}'\
            .format(np.linalg.norm(Xk-X)))
    return Xk

#%%----------------------------------------------------------------------------
### MATRIX DECOMPOSITION
###----------------------------------------------------------------------------
def svd_scipy(X, typ):
    """
    Singular Value Decomposition with scipy
    Usage:  U, S, Vt = svd_scipy(X, typ)
    Input:  X           matrix to decompose (array)
            typ         data type (string)
    Output: U           left unitary matrix (array)
            S           singular values (vector)
            Vt          transposed right unitary matrix (array)
    """
    X = X.astype(typ)
    U, S, Vt = sp_linalg.svd(X, full_matrices=False)
    t_1 = time.time()
    return U, S, Vt

def svd_skcuda(X, typ):
    """
    Singular Value Decomposition with scikit-cuda and CULA library
    Usage:  U_gpu, S, Vt_gpu = svd_skcuda(X, typ)
    Input:  X           matrix to decompose (array)
            typ         data type (string)
                        CULA free only support typ = float32 and complex64
    Output: U_gpu       left unitary matrix on GPU (array)
            S           singular values on CPU (vector)
            Vt_gpu      transposed right unitary matrix on GPU (array)
    """
    X = X.astype(typ)
    X_gpu = gpuarray.to_gpu(X)                          # send data to gpu
    U_gpu, S_gpu, Vt_gpu = cu_linalg.svd(
        X_gpu, jobu='S', jobvt='S', lib='cula')     # small U and Vt matrices
    S = S_gpu.get()
    X_gpu.gpudata.free()                                # release gpu memory
    S_gpu.gpudata.free()
    return U_gpu, S, Vt_gpu

def decomposition(X, lib, typ):
    """
    Singular Value Decomposition with scipy
    Usage:  U, S, Vt = decomposition(X, lib, typ)
    Input:  X           matrix to decompose (array)
            lib         library to use, 'scipy' or 'skcuda' (string)
            typ         data type (string)
    Output: U           left unitary matrix (array)
            S           singular values (vector)
            Vt          transposed right unitary matrix (array)
    """
    print('\tSVD in progress. Please be patient.')
    t_0 = time.time()
    if lib == 'scipy':                                  # using SciPy
        U, S, Vt = svd_scipy(X, typ)
    elif lib == 'skcuda':                               # using scikit-cuda
        U, S, Vt = svd_skcuda(X, typ)
    t_1 = time.time()
    print('Decomposition time:           {:8.2f} s'.format(t_1 - t_0))
    return U, S, Vt

#%%----------------------------------------------------------------------------
### THRESHOLDING
###----------------------------------------------------------------------------
def thres_ind(S, m, n):                                 # Indicator function
    """
    Automatic thresholding using INDicator function (IND)
    Factor Analysis in Chemistry, Third Edition, p387-389
    Edmund R. Malinowki
    Usage:  k_ind, params_ind = thres_ind(S, m, n)
    Input:  S           singular values (vector)
            m           number of rows (integer)
            n           number of columns (integer)
    Output: k_ind       indicator function IND (integer)
            params_ind  additional parameters (tuple)
    """
    # preallocating
    ev = np.zeros(n)
    df = np.zeros(n)
    rev = np.zeros(n)
    sev = np.zeros(n-1)
    sdf = np.zeros(n-1)
    re = np.zeros(n)
    ind = np.zeros(n)
    for j in range(n):
        ev[j] = S[j]**2
        df[j] = (m-j)*(n-j)
        rev[j] = ev[j] / df[j]
    for k in range(n-1):
        sev[k] = np.sum(ev[k+1:n])
        sdf[k] = np.sum(df[k+1:n])
    for i in range(n-1):
        re[i] = np.sqrt(sev[i] / (m * (n-i-1)))         # see eq. 4.44
        ind[i] = re[i] / (n-i-1)**2                     # see eq. 4.63
    re[-1] = np.nan
    ind[-1] = np.nan
    k_ind = np.argmin(ind[:-1]) + 1                 # to compensate start at 0
    params_ind = (ev, re, ind, rev, sdf, sev)           # group parameters
    print('IND thresholding:          {:8d} values'.format(k_ind))
    return k_ind, params_ind

def thres_sl(S, X, max_err):                            # Significant level
    """
    Automatic thresholding using Significant Level (SL)
    Factor Analysis in Chemistry, Third Edition, p387-389
    Edmund R. Malinowki
    Usage:  k_sl = thres_ind(X, S, max_err)
    Input:  S           singular values (vector)
            X           original matrix (array)
            max_err     maximum error level to discriminate signal and noise
    Output: k_sl        significant level SL (integer)
    """
    t_0 = time.time()
    m, n = X.shape
    if m < n:
        raise ValueError(
            '\nnumber of rows should be higher than number of colomns \
            for thresholding')
    k_ind, params_ind = thres_ind(S, m, n)
    ev, re, ind, rev, sdf, sev = params_ind[:]
    t = np.zeros((n,6))
    for j in range(n):
        t[j,0] = j
        t[j,1] = ev[j]
        t[j,2] = re[j]
        t[j,3] = ind[j]
        t[j,4] = rev[j]
    for j in range(n-1):
        f = (sdf[j] * ev[j]) / ((m-j) * (n-j) * sev[j])
        # convert f (see eq. 4.83) into percent significance level
        if j < n:
            tt = np.sqrt(f)
            df = n - j -1
            a = tt / np.sqrt(df)
            b = df / (df + tt**2)
            im = df - 2
            jm = df - 2 * int(np.fix(df / 2))
            ss = 1
            cc = 1
            ks = 2 + jm
            fk = ks
            if (im - 2) >= 0:
                for k in range(ks, im+2, 2):
                    cc = cc * b * (fk-1) / fk
                    ss = ss + cc
                    fk = fk + 2
            if (df - 1) > 0:
                c1 = .5 + (a * b * ss + np.arctan(a)) * .31831
            else:
                c1 = .5 + np.arctan(a) * .31831
            if jm <= 0:
                c1 = .5 + .5 *a * np.sqrt(b) * ss
        s1 = 100 * (1 - c1)
        s1 = 2 * s1
        if s1 < 1e-2:
            s1 = 0
        t[j, 5] = s1
    t[n-1, 5] = np.nan
    k_sl = np.argmin((t[:n-1,5]) < max_err)         # to compensate start at 0
    t_1 = time.time()
    print('SL thresholding at {:4.1f} %: {:8d} values' \
        .format(max_err, k_sl))
    if k_sl < 0:
        raise ValueError('No signal found after thresholding')
    print('Thresholding time:            {:8.2f} s'.format(t_1 - t_0))
    return k_sl

def threshold(S, X, k_thres, max_err):
    """
    Manual or automatic threshold
    Usage:  k_thres = threshold(S, X, k_thres, max_err)
    Input:  S           singular values (vector)
            X           original matrix (array)
            k_thres     original threshold (integer)
            max_err     maximum error level to discriminate signal and noise
    Output: k_thres     checked threshold (integer)
    """
    if k_thres == 0:                                    # automatic
        k_thres = thres_sl(S, X, max_err)
    elif (k_thres >= 1) and (k_thres <= min(X.shape)):  # manual
        print('Manual thresholding: \t\t{:8d} values' \
            .format(k_thres))
    else:
        raise ValueError('k_thres out of matrix limits')
    return k_thres

#%%----------------------------------------------------------------------------
### MATRIX RECONSTRUCTION
###----------------------------------------------------------------------------
def lowrank_scipy(U, S, Vt, k_thres, typ):
    """
    Low-rank approximation with scipy
    Usage:  Xk = lowrank_scipy(U, S, Vt, k_thres, typ)
    Input:  U           left unitary matrix
            S           singular values vector
            Vt          transposed right unitary matrix
            k_thres     threshold for low-rank approximation (integer)
            typ         data type (string)
    Output: Xk          recovered matrix
    """
    Smat = sp_linalg.diagsvd(S[:k_thres], k_thres, k_thres).astype(typ)
    Xk = np.dot (U[:,:k_thres], np.dot (Smat, Vt[:k_thres,:]))
    return Xk

def lowrank_skcuda(U_gpu, S, Vt_gpu, k_thres, typ):
    """
    Low-rank approximation with skcuda
    Usage:  Xk = lowrank_skcuda(U_gpu, S, Vt_gpu, k_thres, typ)
    Input:  U           left unitary matrix on GPU
            S           singular values vector on CPU
            Vt          transposed right unitary matrix on GPU
            k_thres     threshold for low-rank approximation (integer)
            typ         data type (string)
    Output: Xk          recovered matrix
    """
    S_gpu = gpuarray.to_gpu(S)                          # send data to gpu
    S_gpu = S_gpu.astype(typ)
    Xk_gpu = (cu_linalg.dot(U_gpu[:,:k_thres], \
        cu_linalg.dot_diag(S_gpu[:k_thres], Vt_gpu[:k_thres,:])))
    Xk = Xk_gpu.get()                                   # take data from gpu
    Xk_gpu.gpudata.free()                               # release gpu memory
    U_gpu.gpudata.free()
    S_gpu.gpudata.free()
    Vt_gpu.gpudata.free()
    return Xk

def lowrank(U, S, Vt, k_thres, lib, typ):
    """
    Low-rank approximation with library choice
    Usage:  Xk = lowrank(U, S, Vt, k_thres, lib, typ)
    Input:  U           left unitary matrix on GPU
            S           singular values vector on CPU
            Vt          transposed right unitary matrix on GPU
            k_thres     threshold for low-rank approximation (integer)
            lib         library to use, 'scipy' or 'skcuda' (string)
            typ         data type (string)
    Output: Xk          recovered matrix
    """
    t_0 = time.time()
    if lib == 'scipy':                                  # using SciPy
        Xk = lowrank_scipy(U, S, Vt, k_thres, typ)
    elif lib == 'skcuda':                               # using scikit-cuda
        Xk = lowrank_skcuda(U, S, Vt, k_thres, typ)
    t_1 = time.time()
    print('Resconstruction time:         {:8.2f} s\n'.format(t_1 - t_0))
    return Xk

#%%----------------------------------------------------------------------------
### MAIN FUNCTION
###----------------------------------------------------------------------------
def svd_auto(X, k_thres=0, max_err=7.5, lib='auto', disp_err=False):
    """
    Singular Value Decomposition (SVD) and low-rank approximation
    Usage:  Xk, k_thres = svd_auto(X)
            Xk, k_thres = svd_auto(X, max_err=7.5)
            Xk, k_thres = svd_auto(X, lib='scipy', k_thres=100)
    Input:  X           matrix to denoise (array)
            k_thres     if 0, allows automatic thresholding
                        if > 0 and <= min(row, col), manual threshold (integer)
            max_err     error level for automatic thresholding (float)
                        from 5 to 10 %
            lib         library to use, 'auto, 'scipy' or 'skcuda' (string)
            disp_err    if True, display relative error
                        between original and denoised matrix (boolean)
    Output: Xk          denoised matrix (array)
            k_thres     number of values used for thresholding
    """
    try:
        # Pre-processing
        X, transp = pre_svd(X)
        
        # Library import
        lib, typ = lib_svd(X, lib)
        
        # Matrix decomposition with library and type selection
        U, S, Vt = decomposition(X, lib, typ)
        
        # Thresholding
        k_thres = threshold(S, X, k_thres, max_err)
        
        # Matrix reconstruction
        Xk = lowrank(U, S, Vt, k_thres, lib, typ)
        
        # Post-processing
        Xk = post_svd(X, Xk, transp, disp_err)
        return Xk, k_thres
    
    # Errors management
    except ImportError as err:                          # libraries
        print('Error: {:s}'.format(str(err)))
    except MemoryError as err:                          # out of memory
        print('Error: {:s}'.format(str(err)))
        print('Please decrease data size')
    except NotImplementedError as err:                  # 2D array
        print('Error: {:s}'.format(str(err)))
    except ValueError as err:                       # threshold or precision
        print('Error: {:s}'.format(str(err)))

#%%----------------------------------------------------------------------------
### IF PROGRAM IS DIRECTLY EXECUTED
###----------------------------------------------------------------------------
if __name__ == '__main__':
    # Select matrix dimensions and threshold
    size_list = [1024, 2048, 4096, 8192]
    # Select data types
    # CULA free only support float32 and complex64
    type_list = ['float32', 'float64', 'complex64', 'complex128']
    benchmark = [['Points', 'Type', 'scipy (s)', 'skcuda (s)']]

    # Benchmarking
    for n in size_list:                                 # number of columns
        m = n + 1                                       # number of rows
        print('\n--------------------------------------------------')
        for typ in type_list:                           # data type
            X = np.arange(m*n).reshape(m, n).astype(typ)
            try:                        # using CPU
                t_cpu_0 = time.time()
                Xk, k_thres = svd_auto(
                    X, k_thres=n, lib='scipy', disp_err=True)
                t_cpu = time.time() - t_cpu_0
            except TypeError:           # avoid insignificant value
                t_cpu = float('nan')
            try:                        # using GPU
                t_gpu_0 = time.time()
                Xk, k_thres = svd_auto(
                    X, k_thres=n, lib='skcuda', disp_err=True)
                t_gpu = time.time() - t_gpu_0
            except TypeError:           # avoid insignificant value
                t_gpu = float('nan')
            benchmark.append([str(m)+' x '+str(n), typ, t_cpu, t_gpu])
    
    # Final result
    print('\n--------------------------------------------------')
    print('{:^15s}{:^12s}{:^8s}{:^8s}'.format(*benchmark[0]))
    for row in benchmark[1:]:
        print('{:^15s}{:<12s}{:>8.2f}{:>8.2f}'.format(*row))
