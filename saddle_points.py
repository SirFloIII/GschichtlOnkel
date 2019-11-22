import numpy as np
from copy import copy
from PIL import Image
#from numba import jit, njit, types
#from numba.typed import Dict
import matplotlib.pyplot as plt
from matplotlib import cm
##import matplotlib.ticker as ticker
##import matplotlib.patches as patches
##import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D



#this is so we can convert back and forth in number bases up to base 36
#create an empty, typed dict for numba
##int_from_char = Dict.empty(key_type=types.unicode_type,
##                           value_type=types.int64)
#now fill the dict
##for n in range(36):
##    int_from_char[np.base_repr(n,36)] = n
int_from_char = {np.base_repr(n,36) : n for n in range(36)}

#num_in_base = jit(np.base_repr, "unicode_type(int64, int64)")
num_in_base = np.base_repr

#@njit(parallel=True, inline="always", cache=True)
def is_increasing(vec):
    return np.all(np.diff(vec)>=0)


#binomial coefficient, returning integer
#@njit("int64(int64, int64)", parallel=True, inline="always", cache=True)
def bino(n, k):
    return np.prod(np.arange(n-k+1, n+1, 1, np.int64)) //\
           np.prod(np.arange(1, k+1, 1, np.int64))


#normalize an array along an axis
#@njit(parallel=True, inline="always", cache=True)
def normalize(a, order=2, axis=-1):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


#@jit("UniTuple(float64[:], 2)(int64, int64, int64, float64[:])",
#      parallel=True)
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
    index_strings = [num_in_base(n, s).zfill(dim) for n in range(s**dim)]
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
    powers_strings = [num_in_base(n, dim+1).zfill(deg)
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
    assert np.all( np.array(a.shape) >= neigh )#long enough in each dim

    #get the matrix that locally solves regression
    R, index_array = regression_matrix(a.ndim, neigh, deg, weights)

    #the coefficients for regression will be stored in
    #an array neigh-1 shorter than data in every dimension, but
    #with an additional dimension to store all coefficients in.
    result_shape = tuple( [k+1-neigh for k in a.shape] + [R.shape[0]] )
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
def find_critical(a, neigh, tol=1E-3, rim=1E-3):
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
    #with weights such that the innermost cube has same
    #weight as every one going outwards, like onion layers
    weights = np.arange(neigh//2, 0, -1)
    weights = 1/(weights**n-(weights-1)**n)
    coeff = local_regression(a, neigh, 2, weights)

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
        center_idx = np.array(it.multi_index, dtype=np.int) + neigh//2-1
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
##        print("cube at" ,np.array(it.multi_index, dtype=np.int),
##              "with center at", center_idx,
##              "has crit at:", it.multi_index+np.round(crit, 1))

        #the critical point is only of interest if in the innermost cube
        if np.all(neigh/2-1<crit+rim) and np.all(crit<neigh/2+rim):
            eigvals = np.linalg.eig(hessian)[0]
            #print(eigvals)
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
          len(indices_of["degen"]), "degenerate points,\n",
          len(indices_of["ridge"]), "ridge points,",
          len(indices_of["valley"]), "valley points and",
          len(indices_of["saddle"]), "saddle points.")

    #convert list of float vectors [(x1,y1,z1), (x2,y2,z2), ...] to the
    #tuples of arrays integers ([x1,x2,...], [y1,y2,...], [z1,z2,...])
    for key in indices_of:
        indices_of[key] = tuple(np.array(indices_of[key], np.int).T)

    return indices_of

#generate data
k=1/8
r = np.arange(-3, 3, k)
x, y = np.meshgrid(r, r)
f = -np.sin(x*y)+np.cos(x**2+y**2) + np.random.rand(x.shape[0], x.shape[1])/4

#print to CLI how many critical points we found
#d = find_critical(f, 6, tol=0, rim=0)
##for p in d:
##    if d[p]: print(p, d[p])

#3D plot
def plot(x,y,f, n, tol, rim):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.subplots_adjust(top = 1,
                        bottom = 0,
                        right = 1,
                        left = 0,
                        hspace = 0,
                        wspace = 0)
    plt.margins(x=0, y=0)

    ax.plot_wireframe(x,y,f,
                      colors="k",
                      linewidths=0.1)

    col = {"max": "r",
           "min": "b",
           "saddle": "k",
           "degen": "0.5",#gray
           "valley": "g",
           "ridge": "m"}
    
    d = find_critical(f, n, tol=tol, rim=rim)

    for p in d:
        if d[p]: ax.scatter(d[p][0]+k/2,
                            d[p][1]+k/2,
                            f[d[p]],
                            zdir="z",
                            zorder=1,
                            s=15,
                            c=col[p],
                            depthshade=False)

    plt.show()


if __name__ == "__main__":
    
    #plot(x, y, f, 6, 1E-3, 0)
    
    pic = Image.open("narzisse.jpg")
    data = np.array(pic)[100:250, 150:300, 1]
    ###data = np.array(pic)[...,1]
    plt.imshow(data)
    x = range(data.shape[0])
    y = range(data.shape[1])
    ##x=(np.arange(17)-8)/4
    x, y= np.meshgrid(x, x)
    ##d=x**2+y**2+z**2
    x = x.T
    y = y.T
    plot(x, y, data, 6, 1E-3, 0)
    
