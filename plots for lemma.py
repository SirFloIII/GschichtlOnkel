# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 20:28:08 2020

@author: Flo
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1,1,101)
y = x**3 - 0.3*x

bx = 0.32
by = bx**3 - 0.3*bx

ax = -0.63


plt.plot((ax, bx), (by, by))
plt.plot(x, y)