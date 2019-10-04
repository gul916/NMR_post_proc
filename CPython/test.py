#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:10:58 2019

@author: guillaume
"""

def test_id(data):
    ndata = data[:]                                        # avoid data corruption
    for i in range(10):
        print('FID , id(data[100]) = {:}'.format(id(data[100])))
    for i in range(10):
        print('FID , id(ndata[100]) = {:}'.format(id(ndata[100])))
    return ndata

data = range(200,500)
ndata = test_id(data)