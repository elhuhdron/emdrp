
import numpy as np
import scipy.sparse as sp

# argmax over csr sparse matrix, inspired from:
# http://stackoverflow.com/questions/30742572/argmax-of-each-row-or-column-in-scipy-sparse-matrix
# http://stackoverflow.com/questions/22124332/group-argmax-argmin-over-partitioning-indices-in-numpy
def csr_csc_argmax(X, axis=None):
    is_csr = isinstance(X, sp.csr_matrix)
    is_csc = isinstance(X, sp.csc_matrix)
    assert( is_csr or is_csc )
    assert( not axis or (is_csr and axis==1) or (is_csc and axis==0) )

    major_size = X.shape[0 if is_csr else 1]
    major_lengths = np.diff(X.indptr) # group_lengths
    major_not_empty = (major_lengths > 0)

    result = -np.ones(shape=(major_size,), dtype=X.indices.dtype)
    split_at = X.indptr[:-1][major_not_empty]
    maxima = np.zeros((major_size,), dtype=X.dtype)
    maxima[major_not_empty] = np.maximum.reduceat(X.data, split_at)
    all_argmax = np.flatnonzero(np.repeat(maxima, major_lengths) == X.data)
    result[major_not_empty] = X.indices[all_argmax[np.searchsorted(all_argmax, split_at)]]
    return result

# generic image data plotting routine
def showImgData(img, name="plot"):
    from matplotlib import pylab as pl
    #import matplotlib as plt
    interp_string = 'nearest'
    pl.figure(1);
    pl.imshow(img.transpose((1,0)),interpolation=interp_string);
    pl.title(name)
    pl.colorbar()
    pl.show()

# http://www.geeksforgeeks.org/backttracking-set-5-m-coloring-problem/

# A utility function to check if the current color assignment is safe for vertex v
def isSafe(v, graph, color, c):
    for i in graph.neighbors(v):
        if c == color[i]: return False
    return True

def graphColoringUtil(graph, m, color, v):
    # base case: If all vertices are assigned a color then
    if v == graph.number_of_nodes(): return True

    # consider vertex v and try different colors
    for c in range(1,m+1):
        # Check if assignment of color c to v is fine
        if isSafe(v, graph, color, c):
            color[v] = c;

            # recur to assign colors to rest of the vertices
            if graphColoringUtil(graph, m, color, v+1): return True

            # If assigning color c doesn't lead to a solution then remove it
            color[v] = 0;

    # If no color can be assigned to this vertex then return false
    return False

# This function solves the m Coloring problem using Backtracking.
# It mainly uses graphColoringUtil() to solve the problem. It returns
# false if the m colors cannot be assigned, otherwise return true and
# prints assignments of colors to all vertices. Please note that there
# may be more than one solutions, this function prints one of the
# feasible solutions
def optimal_color(graph, chromatic):
    # Initialize all color values as 0. This initialization is needed correct functioning of isSafe()
    color = dict((node_id, 0) for node_id in graph.nodes())

    solution_exists = graphColoringUtil(graph, chromatic, color, 0)
    return dict((n, c-1) for n,c in color.items()) if solution_exists else None

# testing
if __name__ == '__main__':
    #X = sp.identity(5, dtype='int8', format='csc')
    #X = sp.rand(900, 1200, density=0.01, format='csr', dtype=np.float32); X[400:410,:] = 0; X.eliminate_zeros()
    #X = sp.coo_matrix( np.array([[1,0,3,4],[0,7,6,5],[0,0,0,2]]) ).tocsr()
    #x = np.zeros((5,5)); x[3,3]=1; x[2,1]=1; X = sp.coo_matrix( x ).tocsc()
    x = np.zeros((5,5)); X = sp.coo_matrix( x ).tocsr()
    print(X)

    Xa = csr_csc_argmax(X)
    print(Xa); print(Xa.shape, Xa.dtype)
