#!/usr/bin/env python
# use python3

import numpy as np
import time
import sys
import argparse
import scipy

from scipy import ndimage as nd
from dpFRAG import dpFRAG
import _dilation_Extension as dilation

#labeled chunks
chunk = [16,17,0]
#size = [1024,1024,480]
size = [384, 384, 384]
offset = [0,0,32]
has_ECS = True

#username = 'watkinspv'
username = 'patilra'

# Input supervoxel labels (hdf5)
labelfile           = '/Data/' + username + '/full_datasets/neon/mbfergus32all/huge_supervoxels.h5'
label_subgroups     = ['with_background','0.99999000']

# Input probability data (hdf5)
probfile            = '/Data/' + username + '/full_datasets/neon/mbfergus32all/huge_probs.h5'
#probfile = ''

# Input segmented labels (hdf5)
#gtfile              = '/Data/datasets/labels/gt/M0007_33_labels_briggmankl_watkinspv_39x35x7chunks_Forder.h5'
#gt_dataset          = 'labels'

# Input raw EM data
rawfile             = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5'
raw_dataset         = 'data_mag1'
#rawfile = ''

# Output agglomerated labels
outfile             = '/Data/' + username + '/tmp_agglo_out.h5'

# Input probability augmented data
probaugfile         = ''
#probaugfile         = '/Data/' + username + '/full_datasets/neon_sixfold/mbfergus32/huge_probs.h5'

# Input raw EM augmented data                                                                    
rawaugfile          = ''
#rawaugfile          = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder_aug.h5'

# output raw supervoxels (with empty labels removed)
rawout              = '/home/' + username + ('/Downloads/svox_%dx%dx%d.raw' % tuple(size))

feature_set = 'minimal'
progressBar = True
verbose = True

# use getFeatures=False to only get the RAG (without boundary voxels or features)
getFeatures = True

# must specify rawfile and probfile for computing features (and boundary voxels)
assert( not getFeatures or (rawfile and probfile) )

# instantiate frag and load data
frag = dpFRAG.makeTestingFRAG(labelfile, chunk, size, offset,
    [probfile, probaugfile], [rawfile, rawaugfile],
    raw_dataset, outfile, label_subgroups, ['testing','thr'],
    progressBar=progressBar, feature_set=feature_set, has_ECS=has_ECS,
    verbose=verbose)

# prepare data for dilation
vol_dim = 3
connectivity = 3
dilation_iter = 1
blockdim = 8
sample_supervoxels = frag.supervoxels
nsupervox = frag.nsupervox
structuring_element = nd.morphology.generate_binary_structure(vol_dim, connectivity)

time1 = 0
time2 = 0
svox_bnd = nd.measurements.find_objects(sample_supervoxels, nsupervox) 
 
for i in range(1,nsupervox):
    pbnd = tuple([slice(x.start-(2*dilation_iter),x.stop+(2*dilation_iter)) for x in svox_bnd[i-1]])  
    svox_cur = sample_supervoxels[pbnd]
    svox_sel = (svox_cur == i)
    svox_sel_out = np.zeros((svox_sel.shape), dtype=np.bool)
    t = time.time();
    dilation.dilate(svox_sel, structuring_element, svox_sel_out, dilation_iter, blockdim, np.asarray((svox_sel.shape), dtype=np.uint32)) 
    time1 += time.time() - t
    #svox_sel_out = svox_sel_out.astype(dtype=bool)
    nbrlbls = np.unique(svox_cur[svox_sel_out])
    # do not add edges to background or to the same supervoxel
    nbrlbls = nbrlbls[np.logical_and(nbrlbls != i, nbrlbls != 0)]
print('gpu done in %.4f s' % time1)


for j in range(1, nsupervox):
    #incase of dilation greater than 1 the inout binary mask changes. Hence 
    #required to reintialize the binary mask
    pbnd = tuple([slice(x.start-(2*dilation_iter),x.stop+(2*dilation_iter)) for x in svox_bnd[j-1]])
    svox_cur = sample_supervoxels[pbnd]
    svox_sel = (svox_cur == j)
    t = time.time()
    svox_sel_out_gndtrth = nd.morphology.binary_dilation(svox_sel, structure=structuring_element, iterations=dilation_iter)
    time2 += time.time() - t;
    n_lbls = np.unique(svox_cur[svox_sel_out_gndtrth])
    n_lbls = n_lbls[np.logical_and(n_lbls != j, n_lbls !=0)]
 
print('original done in %.4f s' % time2)

'''if((svox_sel_out == svox_sel_out_gndtrth).all()):
        print('pass')
    else:
        print('fail', i)'''
