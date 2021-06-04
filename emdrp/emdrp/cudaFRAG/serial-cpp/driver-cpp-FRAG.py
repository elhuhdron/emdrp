#!/usr/bin/env python
# use python3

import numpy as np
import time
import sys
import argparse
import scipy
from numpy.linalg import matrix_rank
#from dpLoadh5 import dpLoadh5
from dpFRAG import dpFRAG
import _FRAG_extension as FRAG_extension

#read the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--size_of_edges',nargs =1, type= np.int32, default = 30000, help = 'Enter the size of maximum edges that a supervoxel can have')
parser.add_argument('--do_cpu_rag', nargs =1, type = bool , default = False , help = 'Turn the python-serial creation of rag on/off')
parser.add_argument('--validate', nargs=1, type=bool , default = False , help = 'Perform Valiation')
parser.add_argument('--adjacencyMatrix', nargs=1, type=bool, default=False, help = 'Use adjacency Matrix in kernel or not')
args = parser.parse_args()
size_of_edges = args.size_of_edges
do_cpu_rag = args.do_cpu_rag
validate = args.validate
adjacencyMatrix = args.adjacencyMatrix

# labeled chunks
#chunk_range_beg     = 17,19,2, 17,23,1, 22,23,1, 22,18,1, 22,23,2, 19,22,2
chunk = [17,19,2]
size = [128,128,128]
offset = [0,0,0]
has_ECS = True

username = 'patilra'

# Input supervoxel labels (hdf5)
labelfile           = '/Data/' + username + '/full_datasets/neon_sixfold/mbfergus32/huge_supervoxels.h5'
label_subgroups     = ['with_background','0.99999000']

# Input probability data (hdf5)
probfile            = '/Data/' + username + '/full_datasets/neon_sixfold/mbfergus32/huge_probs.h5'

# Input segmented labels (hdf5)
gtfile              = '/Data/datasets/labels/gt/M0007_33_labels_briggmankl_watkinspv_39x35x7chunks_Forder.h5'
gt_dataset          = 'labels'

# Input raw EM data
rawfile             = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5'
raw_dataset         = 'data_mag1'

# Output agglomerated labels
outfile             = '/Data/' + username + '/tmp_agglo_out.h5'

# Input probability augmented data
probaugfile         = ''
#probaugfile         = '/Data/' + username + '/full_datasets/neon_sixfold/mbfergus32/huge_probs.h5'

# Input raw EM augmented data
rawaugfile          = ''
#rawaugfile          = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder_aug.h5'

feature_set = 'minimal'
progressBar = True
verbose = True

# use getFeatures=False to only get the RAG (wihtout boundary voxels or features)
getFeatures = False

# instantiate frag and load data
frag = dpFRAG.makeBothFRAG(labelfile, chunk, size, offset,
    [probfile, probaugfile], [rawfile, rawaugfile],
    raw_dataset, gtfile, outfile, label_subgroups, ['training','thr'],
    progressBar=progressBar, feature_set=feature_set, has_ECS=has_ECS,
    verbose=verbose)

# hack to save raveled indices of overlap in context of whole volume (including boundary)
# boundary size is saved in frag.eperim
frag.ovlp_attrs += ['ovlp_cur_dilate']

# create graph
generated_adjacency = np.zeros((frag.nsupervox,frag.nsupervox),dtype=np.int32)

# calculate the jump steps requied for 1X dilation to get neighborhood
neigh_sel_size = (2*frag.neighbor_perim)+1
dilate_array = np.zeros((neigh_sel_size,neigh_sel_size,neigh_sel_size))
dilate_array[neigh_sel_size//2,neigh_sel_size//2,neigh_sel_size//2] = 1;
binary_struct=scipy.ndimage.morphology.generate_binary_structure(frag.supervoxels.ndim, frag.connectivity)
neigh_sel = scipy.ndimage.morphology.binary_dilation(dilate_array, binary_struct, frag.neighbor_perim)
neigh_sel_indices = np.transpose(np.nonzero(neigh_sel))
neigh_sel_indices = neigh_sel_indices - (neigh_sel_size//2)

#calculate the jump steps required for 2X dilation to check for borders. Always look two times the actual dilation to
# calculate the borders
border_sel_size = (4*frag.neighbor_perim)+1
dilate_array1 = np.zeros((border_sel_size,border_sel_size,border_sel_size))
dilate_array1[border_sel_size//2,border_sel_size//2,border_sel_size//2] = 1;
binary_struct=scipy.ndimage.morphology.generate_binary_structure(frag.supervoxels.ndim, frag.connectivity)
neigh_sel = scipy.ndimage.morphology.binary_dilation(dilate_array1, binary_struct, 2*frag.neighbor_perim)
border_sel_indices = np.transpose(np.nonzero(neigh_sel))
border_sel_indices = border_sel_indices - (border_sel_size//2)
set_border = [tuple(a) for a in border_sel_indices]
set_neigh = [tuple(b) for b in neigh_sel_indices]
unique_indices = set(set_border) & set(set_neigh)
compliment_indices = set(set_border) - unique_indices
compliment_index = list(compliment_indices)

#calculate indices for 1X dilation
steps = np.array([(x[0]*frag.supervoxels.shape[1]*frag.supervoxels.shape[2] + x[1]*frag.supervoxels.shape[2] + x[2]) for x in neigh_sel_indices])
steps = steps[np.nonzero(steps)]

#calculate indices for 2X dilation
steps_border = np.array([(y[0]*frag.supervoxels.shape[1]*frag.supervoxels.shape[2] + y[1]*frag.supervoxels.shape[2] + y[2]) for y in compliment_index])
steps_border = steps_border[np.nonzero(steps_border)]

# build the rag
print('Cpp serial generation of rag'); t=time.time()
FRAG_extension.build_frag(frag.supervoxels, frag.nsupervox, frag.connectivity, np.int32(size_of_edges), generated_adjacency, bool(validate),steps, bool(adjacencyMatrix))
print('done in %.4f s'%  (time.time() - t))

if do_cpu_rag:
    frag.createFRAG(features=getFeatures)

# just to use same name for RAG networkx object as was in driver-cpu.py (from gala example.py)
    g_train = frag.FRAG

# save adjacency matrix
    print('Exporting adjacency matrix'); t=time.time()
    import networkx as nx
    am=nx.to_numpy_matrix(g_train).astype(np.int32)
    fn = 'tmp-adjacency-matrix-cpu.raw' 
    am.tofile(fn)
    print('\tdone in %.4f s' % (time.time() - t))

    if getFeatures:
      print('Outputting boundary voxels'); t=time.time()
      fout = open("tmp-boundary_pixel_indices-cpu.txt","w")
      edges = g_train.edges()
      edges.sort()
      for edge in edges:
          fout.write("(%d,%d), "%(edge[0],edge[1]))
      fout.close()   
      print('\tdone in %.4f s' % (time.time() - t))



if validate:
    file_name = 'tmp-adjacency-matrix-cpu.raw' 
    reference_adjacency = np.fromfile(file_name, dtype=np.int32).reshape((frag.nsupervox,frag.nsupervox),order='C')

    # make both adj matrices symmetric
    reference_adjacency = reference_adjacency + reference_adjacency.T
    sel = reference_adjacency > 0; reference_adjacency[sel] = 1
    generated_adjacency = generated_adjacency + generated_adjacency.T
    sel = generated_adjacency > 0; generated_adjacency[sel] = 1
    #test case to check for edges
    diff = generated_adjacency - reference_adjacency
    print(np.transpose(np.nonzero(diff[0,:])))
    false_positives = np.transpose(np.nonzero(diff == 1))
    false_negatives = np.transpose(np.nonzero(diff == -1))
    if false_negatives.shape == 0 and false_positives.shape == 0:
        print("Edges perfectly matching");
    else:
        print("Difference in edge in generated adjacency list Vs reference",false_positives)
        print("Difference in edge values in referece absent in generated adjacency",false_negatives)
        print("The referece adjacency edges", np.sum(reference_adjacency)//2)
        print("The generated adjacency edges", np.sum(generated_adjacency)//2)

