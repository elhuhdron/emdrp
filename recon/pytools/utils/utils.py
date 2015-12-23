
import numpy as np

# generic image data plotting routine
def showImgData(img, name="plot"):
    from matplotlib import pylab as pl
    import matplotlib as plt
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

