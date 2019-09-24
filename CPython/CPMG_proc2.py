#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@authors: Pierre-Aymeric GILLES & Guillaume LAURENT
"""

# Python libraries
import matplotlib.pyplot as plt
import numpy as np
# User defined libraries
import postproc

def plot_function(rawFID, rawSPC):
    nbPtFreq = rawSPC.size
    freq = np.linspace(-1/(2*rawFID.dw2), 1/(2*rawFID.dw2), nbPtFreq)
    
    plt.ion()                       # interactive mode on
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(freq, rawSPC.real)
    ax1.invert_xaxis()
    
    # Avoid superpositions on figure
    fig1.tight_layout(rect=(0,0,1,0.95))
    fig1.show()                     # Display figure

#%%----------------------------------------------------------------------------
### When this file is executed directly
###----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    
#    print('len(sys.argv)', len(sys.argv))

    if len(sys.argv) == 1:
        A_data = np.genfromtxt('./CPMG_FID.csv',delimiter='\t', skip_header=1)
        A_data = A_data[:,0] + 1j*A_data[:,1]
    
        A_firstDec = True
        A_fullEcho = 10e-3
        A_nbEcho = 20               # 38 for less than 8k points, 76 for more
        
        A_td = 32768                # nb of real points + nb of imag points
        A_dw = 24e-6                # dwell time between two points
        A_de = 96e-6                # dead time before signal acquisition
        
        # Saving data to Signal class
        rawFID = postproc.Signal()
        rawFID.setValues_topspin(A_td,A_dw,A_de)
        rawFID.setValues_CPMG(A_firstDec,A_fullEcho,A_nbEcho)
        rawFID.setData(A_data)
    else:
        print("Additional arguments are not yet supported")
    
    rawSPC = postproc.spc(rawFID.data)
    plot_function(rawFID, rawSPC)

    input('\nPress enter key to exit') # wait before closing terminal