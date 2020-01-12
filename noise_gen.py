# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 22:32:11 2020

@author: Flo
"""

import noise
import numpy as np
import itertools


def data(n, d):
    assert d == 2 or d == 3
    
    shape = (n,)*d
    
    data = np.zeros(shape)
    
    for index in itertools.product(range(n), repeat = d):
        if d == 3:
            data[index] = noise.pnoise3(*np.array(index)/n)
        else:
            data[index] = noise.pnoise2(*np.array(index)/n)
    
    data -= np.min(data)
    data /= np.max(data)
    data *= 255
    
    return data.astype(np.uint8)
        