import numpy as np
import copy
from PIL import Image
from numba import jit, njit, types
from numba.typed import Dict
import matplotlib.pyplot as plt
##import matplotlib.ticker as ticker
##import matplotlib.patches as patches
##import matplotlib.animation as animation


pic = Image.open("kleineNarzisse.jpg")
##data = np.array(pic)[100:200, 150:300 ,1]
data = np.array(pic)[...,1]
##data = data/data.max()
#data = np.round(100*np.random.rand(100,100)).astype(np.int)
##data = np.array(range(4*5*6)).reshape((4,5,6))

#this is so we can convert back and forth in number bases up to base 36
#create an empty, typed dict for numba
int_from_char = Dict.empty(key_type=types.unicode_type,
                           value_type=types.int64)
#now fill the dict
for n in range(36):
    int_from_char[np.base_repr(n,36)] = n


#@njit(parallel=True, inline="always")
def is_increasing(vec):
    return np.all(np.diff(vec)>=0)


#binomial coefficient, returning integer
#@njit("int64(int64, int64)", parallel=True, inline="always")
def bino(n, k):
    return np.prod(np.arange(n-k+1, n+1, 1, np.int64)) //\
           np.prod(np.arange(1, k+1, 1, np.int64))


#normalize an array along an axis
#@njit(parallel=True, inline="always")
def normalize(a, order=2, axis=-1):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


#@jit(parallel=True)
def regression_matrix(dim, s=2, deg=1, weights=None):
    """
    Compute the matrix that solves multivariate regression in the cube {0,...,s-1}^dim


    Parameters:
    dim : int
    Number of variables in the multivariate regression.
    
    s : int
    The side length of hypercubes in which to perform local regression.

    deg : int
    Degree of regression function.
    
    weights : list, optional
    List of weights that the vertices get during regression, starting with the outermost layer, going inward. The list must thus have a length of s/2.

    Returns:
    R : array_like
    The matrix that solves the regression problem and yields the coefficients for 1,x,y,z,x²,xy,xz,y²,yz,z²,... when multiplied onto the given data indexed by index_array.

    index_array : array_like
    The matrix whose rows are the index tuples of points in the cube.
    
    """
    assert weights is None or len(weights) == np.ceil(s/2)
    
    #enumerate the corners in a s**dim hypercube with s-ary strings
    index_strings = [np.base_repr(n, s).zfill(dim) for n in range(s**dim)]
    index_array = np.array([[int_from_char[n] for n in str]
                            for str in index_strings])

    #construct Vandermonde matrix
    V = np.ones((s**dim, bino(dim+deg, deg)), np.int)

    #how over-/underdetermined are we?
    print("Regression will estimate", V.shape[1],
          "parameters from", V.shape[0], "data points.")
    factor = V.shape[0]/V.shape[1]
    if factor < 1:
        print("Warning: Underdetermined by a factor of",
              round(factor, 2), "!")
    else:
        print("Overdetermined by a factor of ", round(factor, 1))

    #write the linear part of the Vandermonde matrix
    V[:, 1:dim+1] = index_array

    #enumerate the (mixed) products of all variables and
    #constant 1 up to the degree specified
    powers_strings = [np.base_repr(n, dim+1).zfill(deg)
                      for n in range((dim+1)**deg)]
    powers_list = [[int_from_char[n] for n in str] for str in powers_strings]
    powers_list = [p for p in powers_list if is_increasing(np.array(p))]

    #then calculate products and fill the matrix col by col, going right
    for counter, powers in enumerate(powers_list):
        V[:, counter] = np.prod([V[:, c] for c in powers], axis=0)

    if weights is not None:
        #the min() expression calculates the discrete distance from the
        #outermost faces of the hypercube
        W = np.diag([weights[min(np.min(v), s-1-np.max(v))]
                     for v in index_array])
    else:
        W = np.eye(V.shape[0])
    
    #we need to solve V.T@W@V@x=V.T@W@y many times later,
    #so compute the inverse now once. @ is matrix multiplication.
    R = np.linalg.inv(V.T @ W @ V) @ V.T @ W

    print("Vandermonde solver done.")
    return R, index_array


#@njit(parallel=True)
def local_regression(a, neigh=2, deg=1, weights=None):
    """
    Consider all hypercubes of length s in the uniform grid a, and perform regression with the values in a, up to the points at the upper edges, where regression can't be applied.


    Parameters:
    a : array_like
    The data on which to perform multivariate regression.
    
    neigh : int
    The side length of hypercubes in which to perform local regression.
    
    deg : int
    Degree of regression function.

    weights : list, optional
    List of weights that the vertices get during regression, starting with the outermost layer, going inward. The list must thus have a length of s/2.

    Returns:
    result : array_like
    An array s-1 shorter than a in every dimension, but with one additional dimension of length a.ndim+1, which contains the regression coefficients.
    
    """

    #get the matrix that locally solves regression
    R, index_array = regression_matrix(a.ndim, neigh, deg, weights)
    
    #the coefficients for regression will be stored in
    #an array neigh-1 shorter than data in every dimension, but
    #with an additional dimension to store all coefficients in.
    result_shape = tuple( [k+1-neigh for k in data.shape] + [R.shape[0]] )
    result = np.zeros(result_shape)

    #perform regression in all cubes with length s
    it = np.nditer(result[...,0], flags=['multi_index'])
    while not it.finished:
        #this computes the current hypercube indices
        hypercube = index_array + np.array(it.multi_index)
        #this performs the actual regression
        result[it.multi_index] = R @ a[tuple(hypercube.T)].flatten()
        it.iternext()

    print("Regression done.")
    return result


#@njit(parallel=True)
def find_critical(a, neigh, tol, rim):
    """
    Find and classify critical points in the voxels of multidimensional data. Local quadratic regression is used.
    

    Parameters:
    a : array_like
    The data. It is understood to be the scalar value of a function sampled on a uniform grid.

    neigh : int
    Number of data points outside of the central voxel used for regression, along each dimension.

    tol : float
    Curvature is only considered positive if >tol and only negative if <-tol, and flat otherwise.

    rim : float
    The halo around the central voxel, within which critical points are registered.

    Returns:
    indices_of : dict
    Dictionary with the different types of critical points as keys ("min", "max", "degen", "ridge", "valley", "saddle") and the list of indices of such points as corresponding value.
    """
    
    n = a.ndim
    
    #local quadratic regression on the data
    coeff = local_regression(a, neigh, 2, (np.arange(neigh/2)+1)**2)

    #remember indices where critical points occur
    indices_of = {"min":[],
                  "max":[],
                  "degen":[],
                  "ridge":[],
                  "valley":[],
                  "saddle":[]}
    
    #for writing the coefficients of quadratic terms into a matrix
    tri_u_indices = np.triu_indices(n)

    it = np.nditer(coeff[...,0], flags=['multi_index'])
    while not it.finished:
        #the index of the vertex in the center cube with lowest indices
        center_idx = np.array(it.multi_index) + neigh/2-1
        vec = coeff[it.multi_index]
        linear_c = vec[1:1+n]
        hessian = np.zeros((n,n))
        hessian[tri_u_indices] = vec[n+1:]
        hessian = (hessian+hessian.T)/2
        
        #look at where the quadratic function has its critical point
##        try:
##            crit = np.linalg.solve(hessian, -linear_c/2)
##        except np.linalg.LinAlgError:
##            indices_of["degen"] += [center_idx]
##            crit = np.inf #so we skip the if block below
        crit = np.linalg.solve(hessian, -linear_c/2)
        
        #the critical point is only of interest if in the innermost cube
        if np.all(neigh/2-1<crit+rim) and np.all(crit<neigh/2+rim):
            eigvals = np.linalg.eig(hessian)[0]
            if   np.all(eigvals>tol):       indices_of["min"] += [center_idx]
            elif np.all(eigvals<-tol):      indices_of["max"] += [center_idx]
            elif np.all(np.abs(eigvals)<tol):indices_of["degen"] += [center_idx]
            elif np.all(eigvals<tol):       indices_of["ridge"] += [center_idx]
            elif np.all(eigvals>-tol):      indices_of["valley"] += [center_idx]
            else:                           indices_of["saddle"] += [center_idx]
            
        it.iternext()

    print("Found",
          len(indices_of["min"]), "minima,",
          len(indices_of["max"]), "maxima,",
          len(indices_of["degen"]), "degenerate points,",
          len(indices_of["ridge"]), "ridge points,",
          len(indices_of["valley"]), "valley points and",
          len(indices_of["saddle"]), "saddle points.")

    #convert list of float vectors [(x1,y1,z1), (x2,y2,z2), ...] to the
    #tuples of arrays integers ([x1,x2,...], [y1,y2,...], [z1,z2,...])
    for key in indices_of:
        indices_of[key] = tuple(np.array(indices_of[key], np.int).T)
        
    return indices_of

a=find_critical(data, 6, 0.0, 0.0)
#a={}

crits = copy.copy(data)/3 + 255/3
saddles = copy.copy(crits)
other = copy.copy(crits)
if a["max"]: crits[a["max"]] = 255
if a["min"]: crits[a["min"]] = 0
if a["saddle"]:saddles[a["saddle"]] = 255
if a["degen"]: other[a["degen"]] = 255
if a["valley"]: other[a["valley"]] = 0
if a["ridge"]: other[a["ridge"]] = 255

plt.subplot(2, 2, 1)
plt.imshow(data, cmap='gray', vmin=0, vmax=255)
plt.xlabel("original data")

plt.subplot(2, 2, 2)
plt.imshow(crits, cmap='gray', vmin=0, vmax=255)
plt.xlabel("minima and maxima")

plt.subplot(2, 2, 3)
plt.imshow(saddles, cmap='gray', vmin=0, vmax=255)
plt.xlabel("saddle points")

plt.subplot(2, 2, 4)
plt.imshow(other, cmap='gray', vmin=0, vmax=255)
plt.xlabel("other critical points")

#TODO: 3D plot of 2D data, marking critical points

plt.show()
