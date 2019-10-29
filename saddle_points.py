import numpy as np
from PIL import Image
from numba import jit
from scipy.special import binom
import matplotlib.pyplot as plt
##import matplotlib.ticker as ticker
##import matplotlib.patches as patches
##import matplotlib.animation as animation

#@jit(nopython=True)#, parallel=True)

##pic = Image.open("baum.png")
##data = np.array(pic)[200:400, 300:500,1]
##data = data/data.max()
##data = np.round(100*np.random.rand(100,100)).astype(np.int)
data = np.array(range(4*5*6)).reshape((4,5,6))

#threshold for determining critical points from linear regression
thresh = 0.0001


def is_increasing(vec):
    return np.all(np.diff(vec)>=0)

#binomial, returning integer
def bino(n, k):
    return int(binom(n,k))

#normalize an array along an axis
#@jit(nopython=True)
def normalize(a, order=2, axis=-1):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

#@jit(nopython=True)
def local_regression(a, s=2, deg=1, weights=None):
    """consider all hypercubes of length s in the uniform grid a, and perform regression with the values in a, up to the points at the upper edges, where regression can't be applied.

returns an array s-1 shorter than a in every dimension, but with one additional dimension of length a.ndim+1, which contains the regression coefficients."""
    assert s%2 == 0 and 2 <= s <= 36 #np.base_repr() works only up to 36
    assert weights is None or len(weights) == s//2
    dim = a.ndim
    
    #enumerate the corners in a s**dim hypercube with s-ary strings
    index_strings = [np.base_repr(n, s).zfill(dim) for n in range(s**dim)]
    index_array = np.array([[int(n) for n in str] for str in index_strings])

    #construct Vandermonde matrix
    V = np.ones((s**dim, bino(dim+deg, deg)), np.int)
    V[:, 1:dim+1] = index_array

    #enumerate the (mixed) products of all variables and
    #constant 1 up to the degree specified
    powers_strings = [np.base_repr(n, dim+1).zfill(deg)
                      for n in range((dim+1)**deg)]
    powers_list = [[int(n) for n in str] for str in powers_strings]
    powers_list = [p for p in powers_list if is_increasing(p)]

    #then calculate products and fill the matrix col by col, going right
    for counter, powers in enumerate(powers_list):
        V[:, counter] = np.prod([V[:, c] for c in powers], axis=0)
    print(V)

    if weights:
        #the min() expression calculates the discrete distance from the
        #outermost faces of the hypercube
        W = np.diag([weights[min(np.min(v), s-1-np.max(v))]
                     for v in index_array])
    else:
        W = np.eye(V.shape[0])
    
    #we need to solve V.T@V@x=V.T@y many times later,
    #so compute the inverse now once.
    R = np.linalg.inv(V.T @ W @ V) @ V.T @ W

    #the coefficients for regression will be stored in
    #an array s-1 shorter than data in every dimension, but
    #with an additional dimension to store all coefficients in.
    result_shape = tuple([k+1-s for k in data.shape]+[V.shape[1]])
    result = np.zeros(result_shape)

    #perform regression in all cubes with length s
    it = np.nditer(result[...,0], flags=['multi_index'])
    while not it.finished:
        #this computes the current hypercube indices
        hypercube = index_array + np.array(it.multi_index)
        #this performs the actual regression
        result[it.multi_index] = R @ a[tuple(hypercube.T)].flatten()
        it.iternext()

    return result

v=local_regression(data, 4, 2)


#this bool array is to save where regression suggests critical points
critical = np.zeros(v[...,0].shape, bool)
#if the others dominate the 0th coefficient, then the plane's horizontal.
critical = (abs(normalize(v)[..., 0])>1-thresh)
#now save the indices where that happens
crit_indices = np.argwhere(critical)


#write the coefficients of quadratic terms into a matrix
tri_u_indices = np.triu_indices(dim)

for counter, result in enumerate(quad_results):
    hessian = np.zeros((dim,dim))
    hessian[tri_u_indices] = result[dim+1:]
    hessian = (hessian+hessian.T)/2
    print(result[:dim+1], hessian, np.linalg.eig(hessian)[0])


plt.imshow(critical.astype(np.int))
plt.show()
