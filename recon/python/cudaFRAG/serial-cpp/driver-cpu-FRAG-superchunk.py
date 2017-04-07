#!/usr/bin/env python
# use python3

import numpy as np
import time
import sys
import argparse
import scipy

#from dpLoadh5 import dpLoadh5
from dpFRAG import dpFRAG
import _FRAG_extension as FRAG_extension

#read the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--size_of_edges',nargs =1, type= np.uint32, default = 30000, help = 'Enter the size of maximum edges that a supervoxel can have')
parser.add_argument('--do_cpu_rag', nargs =1, type = bool , default = False , help = 'Turn the python-serial creation of rag on/off')
parser.add_argument('--validate', nargs=1, type=bool , default = False , help = 'Perform Valiation')
parser.add_argument('--adjacencyMatrix', nargs=1, type=bool, default=False, help = 'Use adjacency Matrix in kernel or not')
parser.add_argument('--label_count', nargs=1, type=np.uint32, default=3000, help = 'The number of labels to be processed at a time on a gpu')
args = parser.parse_args()
size_of_edges = args.size_of_edges
do_cpu_rag = args.do_cpu_rag
validate = args.validate
adjacencyMatrix = args.adjacencyMatrix
label_count = args.label_count

# labeled chunks
chunk = [16,17,0]
#size = [1024,1024,480]
size = [512, 512, 480]
offset = [0,0,32]#32
has_ECS = True

#username = 'watkinspv'
username = 'patilra'

# Input supervoxel labels (hdf5)
labelfile           = '/Data/' + username + '/full_datasets/neon/mbfergus32all/huge_supervoxels.h5'
label_subgroups     = ['with_background','0.99999000']

# Input probability data (hdf5)
#probfile            = '/Data/' + username + '/full_datasets/neon/mbfergus32all/huge_probs.h5'
probfile = ''

# Input segmented labels (hdf5)
#gtfile              = '/Data/datasets/labels/gt/M0007_33_labels_briggmankl_watkinspv_39x35x7chunks_Forder.h5'
#gt_dataset          = 'labels'

# Input raw EM data
#rawfile             = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5'
raw_dataset         = 'data_mag1'
rawfile = ''

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
getFeatures = False

# must specify rawfile and probfile for computing features (and boundary voxels)
assert( not getFeatures or (rawfile and probfile) )

# instantiate frag and load data
frag = dpFRAG.makeTestingFRAG(labelfile, chunk, size, offset,
    [probfile, probaugfile], [rawfile, rawaugfile],
    raw_dataset, outfile, label_subgroups, ['testing','thr'],
    progressBar=progressBar, feature_set=feature_set, has_ECS=has_ECS,
    verbose=verbose)

# hack to save raveled indices of overlap in context of whole volume (including boundary)
# boundary size is saved in frag.eperim
frag.ovlp_attrs += ['ovlp_cur_dilate']

# get list of edges after creation of rag 
list_of_edges = np.zeros((np.uint32(size_of_edges),2), dtype=np.uint32)
count = np.zeros(2, dtype=np.uint32)
hybrid_adjacency = np.zeros((np.uint32(label_count)*frag.nsupervox) ,dtype = np.uint8)

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
FRAG_extension.build_frag(frag.supervoxels, frag.nsupervox, frag.connectivity, np.uint32(size_of_edges), list_of_edges, bool(validate),steps, bool(adjacencyMatrix), np.uint32(label_count), count, hybrid_adjacency)
print('done in %.4f s'%  (time.time() - t))



# create graph

if do_cpu_rag:
    frag.createFRAG(features=getFeatures)

    # just to use same name for RAG networkx object as was in driver-cpu.py (from gala example.py)
    g_test = frag.FRAG
    # save adjacency matrix
    print('Exporting adjacency matrix'); t=time.time()
    import networkx as nx
    nx.write_edgelist(g_test,"tmp-edge-list-cpu.txt",data=False)
    #am=nx.to_numpy_matrix(g_test)
    ##np.savetxt("tmp-adjacency_matrix-cpu.txt",am, fmt="%d", delimiter='')
    #am.tofile('tmp-adjacency-matrix-cpu-%dx%d-%s.raw' % (am.shape[0], am.shape[1], str(am.dtype)))
    print('\tdone in %.4f s' % (time.time() - t))

    # dump supervoxels
    #frag.supervoxels_noperim.transpose((2,1,0)).tofile(rawout)

    if getFeatures:
        print('Outputting boundary voxels'); t=time.time()
        fout = open("tmp-boundary_pixel_indices-cpu.txt","w")
        edges = g_test.edges()
        edges.sort()
        for edge in edges:
            fout.write("(%d, %d): "%(edge[0],edge[1]))
            #for b in g_test[edge[0]][edge[1]]['boundary']:
        #    fout.write("%d "%b)
            boundary_subs = np.transpose(np.nonzero(g_test[edge[0]][edge[1]]['ovlp_attrs']['ovlp_cur_dilate']))
            start_sub = np.array([x.start for x in g_test[edge[0]][edge[1]]['ovlp_attrs']['aobnd']])
            #global_subs_padded = boundary_subs + start_sub
            #global_inds = np.ravel_multi_index(global_subs_padded.T.reshape(3,-1), frag.supervoxels.shape)
            #for b in global_inds:
       #    fout.write("%d "%b)
            global_subs_unpadded = boundary_subs + start_sub - frag.eperim
            for b in range(global_subs_unpadded.shape[0]):
                fout.write("(%d,%d,%d) " % tuple(global_subs_unpadded[b,:].tolist()))
            fout.write("\n")
        fout.close()
        print('\tdone in %.4f s' % (time.time() - t))

if validate:
    file_name = 'tmp-edge-list-cpu.txt'
    reference_edges = np.fromfile(file_name, dtype=np.int32, sep=" ").reshape((-1,2), order='C')
    print(reference_edges)
    print(list_of_edges[np.nonzero(list_of_edges)].reshape((count[0],2)))
    ref = [tuple(a) for a in reference_edges]
    gen = [tuple(b) for b in list_of_edges]
    unique_indices = set(ref) & set(gen)
    compliment_indices = set(ref) - unique_indices
    compliment_index = list(compliment_indices)
    if(compliment_index == []):
      print("The edge list matches for this dataset.")
    else:
      print(compliment_index)
