# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 00:47:06 2019

@author: Flo
"""

import numpy as np
import PIL

import itertools


#indices: 0 1
#         2 3

def saddleCheck(corners):
    if corners.shape[1:] == (2,2) or corners.size == 4:
        corners.reshape((-1,4))
    return np.logical_or(np.logical_and(np.logical_and(corners[:,0] > corners[:,1], corners[:,0] > corners[:,2]),
                                        np.logical_and(corners[:,3] > corners[:,1], corners[:,3] > corners[:,2])),
                         np.logical_and(np.logical_and(corners[:,0] < corners[:,1], corners[:,0] < corners[:,2]),
                                        np.logical_and(corners[:,3] < corners[:,1], corners[:,3] < corners[:,2])))

    
def simpleGradients(n, m):
    corners = np.random.uniform(size = (m,4,1,1))
    isSaddle = saddleCheck(corners)
    
    a = np.linspace(1,0, num = n).reshape(1,n,1)
    b = np.linspace(1,0, num = n).reshape(1,1,n)
    
    images = a*b * corners[:,0] + (1-a)*b * corners[:,1] + a*(1-b) * corners[:,2] + (1-a)*(1-b) * corners[:,3]

    return images, isSaddle


def quadraticImage(n, m):

    middle = np.random.uniform(size = (m,2,1,1))
    i = np.random.uniform(low = -1, high = 1, size = (m,1,1))
    j = np.random.uniform(low = -1, high = 1, size = (m,1,1))
    k = np.random.uniform(low = -1, high = 1, size = (m,1,1))
    
    a = np.linspace(1,0, num = n).reshape(1,n,1)
    b = np.linspace(1,0, num = n).reshape(1,1,n)
        
    images = i*(middle[:,0] - a)**2 + 2*j*(middle[:,0] - a)*(middle[:,1] - b) + k*(middle[:,1] - b)**2
    
    isSaddle = (i*k < 0).reshape(m)
    
    return images, isSaddle

def centerImage(x):
    
    mi = np.min(x)
    ma = np.max(x)
    
    return (x-mi)/(ma-mi)

def randomJumpFunctionGenerator():
    
    jumpheight = np.random.uniform(low = 0, high = 0.4)
    jumpposition = np.random.uniform()
    
    return lambda x: x*(1-jumpheight) + jumpheight*(x>jumpposition)

def randomPowerFunctionGenerator():
    
    power = np.random.exponential()
    
    return lambda x: x**power

def compositeFunctions(a, b):
    """
    a many discontinuous functions
    b bool if power
    """
    
    funcs = ([centerImage] + 
             [randomPowerFunctionGenerator()]*bool(b) +
             [randomJumpFunctionGenerator() for x in range(a)])
            
    def f(x):
        for func in funcs:
            x = func(x)
        return x
    
    return f

#x = np.linspace(0,1, num = 500)
#y = compositeFunctions(3, True)(x)
#
#import matplotlib.pyplot as plt
#
#plt.plot(x,y)

n = 8 #sidelength of images
m = 100 #samples per (sampletype)

sampletypes = itertools.product([simpleGradients, quadraticImage],
                                range(4),
                                range(2))

totalImages = []
totalSaddles = []

for f, a, b in sampletypes:
    images, isSaddle = f(n, m)
    for i, image in enumerate(images):
        images[i] = compositeFunctions(a, b)(image)
    totalImages += list(images)
    totalSaddles += list(isSaddle)


with open("useless_file.csv", "w+") as file:
    for i, (image, saddle) in enumerate(zip(totalImages, totalSaddles)):
        myString = str(i) + "  "
        myString += str(image.flatten())[1:-1].replace("\n", "") + "  "
        myString += str(int(saddle)) + "\n"
        
        file.write(myString)



































