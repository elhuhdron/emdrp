
# xxx - this script is incomplete and currently does nothing

#import argparse
#import time
import numpy as np
#import h5py
#from scipy import ndimage as nd
#from scipy import io as sio
#from scipy import interpolate
#from skimage import morphology as morph
#import networkx as nx

#from numpy import linalg as nla
from scipy import ndimage as nd
from scipy import linalg as sla

#from dpLoadh5 import dpLoadh5
from dpWriteh5 import dpWriteh5
#from typesh5 import emLabels, emProbabilities, emVoxelType
from typesh5 import emLabels
#from pyCext import binary_warping

somas_in='/home/watkinspv/Downloads/K0057_soma_annotation/out/K0057_D31_dsx12y12z4_somas.gipl'
data, hdr, info = dpWriteh5.gipl_read_volume(somas_in)
somas = data.reshape(hdr['sizes'][:3][::-1]).transpose(((2,1,0)))
sampling = hdr['scales'][:3]
print(data.shape, data.dtype, sampling)

sizes = emLabels.getSizes(somas); sizes = sizes[1:]; 
soma_valid_labels = np.transpose(np.nonzero(sizes > 0)) + 1
print( 'Number of soma labels is %d' % (soma_valid_labels.size,) )

# iterate over labels, fill each label within bounding box
svox_bnd = nd.measurements.find_objects(somas, max_label=sizes.size)
for j in soma_valid_labels:
    sel = somas[svox_bnd[j-1]] # binary select within bounding box
    
    # use svd to get the eigenvectors of the object
    pts = np.transpose(np.nonzero(sel)).astype(np.double)*sampling
    C = np.mean(pts, axis=0) # centroid
    Cpts = pts-C # center points around centroid
    U, S, Vt = sla.svd(pts,overwrite_a=False,full_matrices=False)

    # get diameter of object along the principal axis

    # make a plot of diameter along the principal axis
    

