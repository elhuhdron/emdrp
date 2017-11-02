
# convert itksnap annotated somas (with 2p overlays for cell type) to "node"-ized nrrd 

#import argparse
#import time
import numpy as np
#import h5py
from scipy import ndimage as nd
#from scipy import interpolate
#from skimage import morphology as morph
#import networkx as nx

from dpLoadh5 import dpLoadh5
from dpWriteh5 import dpWriteh5
#from typesh5 import emLabels, emProbabilities, emVoxelType
#from typesh5 import emLabels
from pyCext import binary_warping

overlay_in='/home/watkinspv/Downloads/K0057_soma_annotation/affine_2Pstack/K0057_D31_soma_seg_overlays_v3.gipl'
data, hdr, info = dpWriteh5.gipl_read_volume(overlay_in)
data = data.reshape(hdr['sizes'][:3][::-1]).transpose(((2,1,0)))
print(data.shape, data.dtype)

# warp all the way down to points for each object. xxx - could be prohibitive for larger volumes
bwlabels, diff, simpleLUT = binary_warping((data > 0).copy(order='C'), np.zeros(data.shape,dtype=np.bool), 
    borderval=False, slow=True, connectivity=3)

subs = np.nonzero(bwlabels)
#np.set_printoptions(threshold=np.nan); print(np.transpose(subs))

# get soma information back from original overlay
types = data[subs]

# convert subscripts into appropriate downsampling space
offset = np.array([1,1,0]) # try to deal with warping bias, has to do with shape cells were labeled with 9x9x8 rect
subs_out = (np.transpose(subs) + offset) * np.array([16,16,16]) // np.array([12,12,4])

# create new nrrd in different downsampling space
size_out = (1696, 1440, 640)
sel = (subs_out < size_out).all(1)

# save two version, one with just "node"-ized points, the other with more human visible diamonds
data_out = np.zeros(size_out, dtype=np.uint16)
data_out[[subs_out[sel,x] for x in range(3)]] = types[sel]
#print(sel.sum(), (data_out > 0).sum())
overlay_out='/home/watkinspv/Downloads/K0057_soma_annotation/out/K0057_D31_soma_seg_overlays_v3_dsx12y12z4.gipl'
dpLoadh5.gipl_write_volume(data_out.transpose((2,1,0)), np.array(size_out), overlay_out, hdr['scales'][:3])

sz = 33; hsz = sz//2;
strel_offs = np.zeros((sz,sz,sz),dtype=np.bool); strel_offs[hsz,hsz,hsz]=1
bwconn = nd.morphology.generate_binary_structure(dpLoadh5.ND, 1)
strel_offs = nd.binary_dilation(strel_offs, structure=bwconn, iterations=15);
strel_offs = np.transpose(np.nonzero(strel_offs)) - hsz
# can't figure out a better way to do this without loop, binary_dilation, but this does not retain the label
for pt,ctype in zip(subs_out[sel,:], types[sel]):
    pts = pt + strel_offs; sel2 = np.logical_and((pts < size_out).all(1), (pts >= 0).all(1))
    data_out[[pts[sel2,x] for x in range(3)]] = ctype
overlay_out='/home/watkinspv/Downloads/K0057_soma_annotation/out/K0057_D31_soma_seg_overlays_v3_diamond_dsx12y12z4.gipl'
dpLoadh5.gipl_write_volume(data_out.transpose((2,1,0)), np.array(size_out), overlay_out, hdr['scales'][:3])

