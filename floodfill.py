# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 20:07:01 2019

@author: Flo
"""

import numpy as np
import PIL
import itertools
from matplotlib import pyplot as plt

import saddle_points

def find_critical(array, r = 2, tol = 0.001, rim = 0.001):
    
    d = len(array.shape)
    
    print("")
    crits = saddle_points.find_critical(array, r*2)
    
    criticals = []
    
    for kind, points in crits.items():
        
        for pos in zip(*points):
            
            criticals.append(Critical(pos, kind, array[pos]))
    
    #for all oberflächen-hyperebenen:
    for axis, j in itertools.product(range(d), [r - 1, -r]):
        if d == 1:
            if array[j] > array[j + np.sign(j)]:
                criticals.append(Critical((j%len(array),), "max", array[j]))
            elif array[j] < array[j + np.sign(j)]:
                criticals.append(Critical((j%len(array),), "min", array[j]))
            else:
                criticals.append(Critical((j%len(array),), "degen", array[j]))
            
        else:
            crits = find_critical(np.take(array, j, axis = axis), r = r, tol = tol, rim = rim)
            for c in crits:
                pos = (*c.pos[:axis], j%array.shape[axis], *c.pos[axis:])
                posInwards = (*c.pos[:axis], j%array.shape[axis] + np.sign(j), *c.pos[axis:])
                
                if c.kind == "max":
                    if array[posInwards] > c.value:
                        kind = "saddle"
                    else:
                        kind = "max"
                elif c.kind == "min":
                    if array[posInwards] < c.value:
                        kind = "saddle"
                    else:
                        kind = "min"
                else:
                    kind = c.kind
                
                l = [a for a in criticals if a.pos == pos]
                if l:
                    print(l, kind)
                    a = l.pop()
                    if a.kind == kind and kind != "saddle":
                        print("kept")
                        pass
                    else:
                        print("removed")
                        criticals.remove(a)
                else:
                    criticals.append(Critical(pos, kind, c.value))
            
            
    return criticals

class Critical:
    
    def __init__(self, pos, kind, value):
        
        self.pos = pos
        self.kind = kind
        self.value = value
        
    def __repr__(self):
        return f"C({self.pos}, {self.kind}, {self.value})"

class SlopeDecomposition:
    
    def __init__(self, image, r = 2):
        
        self.d = len(image.shape)
        self.regions = []
        
        self.image = image
        self.map = -np.ones(image.shape, dtype = np.int)
        
        for mi in itertools.product(*[range(n) for n in image.shape]):
            if any(np.array(mi) < r-1) or any(np.array(mi)-image.shape >= -r):
                self.map[mi] = -2
        
        self.criticals = sorted(find_critical(array, r), key = lambda x:-x.value) + [Critical(np.unravel_index(np.argmin(self.image), self.image.shape), "min", -float("inf"))]
        
        self.decompose()

    def decompose(self):

        for c in self.criticals:
            print(c.value)
            if c.kind == "max":
                i = len(self.regions)
                self.regions.append(SlopeRegion(i, c.pos))
                self.map[c.pos] = i
            elif c.kind == "saddle" or c.kind == "min":
                for r in self.regions:
                    if not r.closed:
                        while r.open_border:
                            p = r.open_border.pop()
                            for n in self.neighbors(p):
                                if self.image[n] >= c.value and self.map[n] == -1:
                                    r.open_border.append(n)
                                    self.map[n] = r.i
                                elif self.image[n] < c.value and p not in r.closed_border:
                                    r.closed_border.append(p)
                        r.open_border = r.closed_border
                        r.closed_border = []
                
                
                if c.kind == "saddle":
                    self.regions[self.map[c.pos]].closed = True
            
            else:
                print("good luck")
            
    def neighbors(self, p):
        return [tuple(np.array(p) + (np.array(range(self.d)) == d) * i) for d, i in itertools.product(range(self.d), [-1, 1])]
            
class SlopeRegion:

    def __init__(self, i, seed):
        
        self.i = i
        self.open_border = [seed]
        self.closed_border = []
        self.closed = False



col = {"max": "r",
       "min": "b",
       "saddle": "k",
       "degen": "0.5",#gray
       "valley": "g",
       "ridge": "m"}


im = PIL.Image.open("TestImage.png").convert()

array = np.array(im)[:,:,0]

#saddle_points.plot(array, 4)


s = SlopeDecomposition(array)

plt.contour(array)
plt.pcolormesh(s.map, shading = "gourand")
plt.scatter([c.pos[1] + 0.5 for c in s.criticals],
            [c.pos[0] + 0.5 for c in s.criticals],
            c = [col[c.kind] for c in s.criticals])

plt.scatter([c[1] + 0.5 for c in s.regions[1].open_border],
            [c[0] + 0.5 for c in s.regions[1].open_border],
            c = "g")


print("")
print(*s.criticals, sep = "\n")