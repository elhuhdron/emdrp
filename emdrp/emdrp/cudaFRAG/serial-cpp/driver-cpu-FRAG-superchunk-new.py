#!/usr/bin/env python
# use python3

import numpy as np
import time
#import sys
import argparse
import scipy
from scipy import ndimage as nd

#from dpLoadh5 import dpLoadh5
from dpFRAG import dpFRAG
import _FRAG_extension as FRAG_extension

# python wrapper for new RAG building routine
def build_frag_new(supervoxels, nsupervoxels, pad=True, nbhd=1, conn=3, steps=None, max_step=None, nalloc_rag=50,
                   nalloc_borders=1000):
    dtype=np.uint32; test=np.zeros((2,2),dtype=dtype)
    if type(supervoxels) != type(test):
        raise Exception( 'In build_frag, supervoxels is not *NumPy* array')
    if len(supervoxels.shape) != 3:
        raise Exception( 'In build_frag, supervoxels shape not 3 dimensional')
    if not supervoxels.flags.contiguous or np.isfortran(supervoxels):
        raise Exception( 'In build_frag, supervoxels not C-order contiguous')
    if supervoxels.dtype != dtype:
        raise Exception( 'In build_frag, supervoxels wrong datatype')
    if type(nsupervoxels) != type(1):
        raise Exception( 'In build_frag, nsupervoxels argument is not a integer')
    if type(nbhd) != type(1):
        raise Exception( 'In build_frag, nbhd argument is not a integer')
    if type(conn) != type(1):
        raise Exception( 'In build_frag, conn argument is not a integer')

    if pad:
        sz =  [x+nbhd*2 for x in supervoxels.shape]
        svox = np.zeros(sz, dtype=dtype); svox[nbhd:-nbhd,nbhd:-nbhd,nbhd:-nbhd] = supervoxels
    else:
        svox = supervoxels; sz = supervoxels.shape

    return_steps = (steps is None)
    if return_steps:
        bwconn = nd.morphology.generate_binary_structure(supervoxels.ndim, conn)

        neigh_sel_size = 2*nbhd + 1
        dilate_array = np.zeros((neigh_sel_size,neigh_sel_size,neigh_sel_size), dtype=bool)
        dilate_array[neigh_sel_size//2,neigh_sel_size//2,neigh_sel_size//2] = 1;
        neigh_sel = scipy.ndimage.morphology.binary_dilation(dilate_array, bwconn, nbhd)
        neigh_sel_indices = np.transpose(np.nonzero(neigh_sel)) - (neigh_sel_size//2)

        #calculate indices for 1X dilation - C-order!
        steps = np.array([(x[0]*sz[1]*sz[2] + x[1]*sz[2] + x[2]) for x in neigh_sel_indices], dtype=np.int32)

        # remove double checking of edges between voxels by removing symmetrical negative steps
        # NOTE: the cpp code is optimized assuming that steps is positive, so removing this requires cpp code mods.
        steps = steps[steps > 0]

        # steps range used in Cpp-code so that we don't look out-of-bounds
        min_step = steps.min(); max_step = steps.max()

    list_of_edges, list_of_borders = FRAG_extension.build_frag_new(svox, nsupervoxels, steps, min_step, max_step,
                                                                   nalloc_rag, nalloc_borders)

    if return_steps:
        return list_of_edges, list_of_borders, steps, max_step
    else:
        return list_of_edges, list_of_borders

#read the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--size_of_edges',nargs =1, type= np.uint32, default = 30000,
                    help = 'Enter the size of maximum edges that the supervoxels can have')
parser.add_argument('--do_cpu_rag', nargs =1, type = bool , default = False ,
                    help = 'Turn the python-serial creation of rag on/off')
parser.add_argument('--validate', nargs=1, type=bool , default = False ,
                    help = 'Perform Valiation')
parser.add_argument('--adjacencyMatrix', nargs=1, type=bool, default=False,
                    help = 'Use adjacency Matrix in kernel or not')
parser.add_argument('--label_count', nargs=1, type=np.uint32, default=30000,
                    help = 'The number of labels to be processed at a time on a gpu/cpu')
parser.add_argument('--size_of_borders', nargs=1, type=np.uint32, default=30000,
                    help= ' The number of borders for a single edge')
parser.add_argument('--no_dilation', nargs =1, type=bool , default=True,
                    help= 'Use the dilation method to calculate the borders or not')
parser.add_argument('--tmp_edge_size', nargs=1, type=np.int, default=100,
                    help='Temporary size for edgelist for each label on gpu')

args = parser.parse_args()
size_of_edges = np.uint32(args.size_of_edges)[0]
do_cpu_rag = args.do_cpu_rag
validate = args.validate
adjacencyMatrix = args.adjacencyMatrix
label_count = args.label_count
size_of_borders = np.uint32(args.size_of_borders)[0]
no_dilation = args.no_dilation
tmp_edge_size = args.tmp_edge_size

# labeled chunks
chunk = [16,17,0]
#size = [1024,1024,480]
size = [128, 128, 128]
#size = [384, 384, 384]
#size = [512, 512, 480]
offset = [0,0,32]#32
has_ECS = True

username = 'watkinspv'
#username = 'patilra'

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
#probaugfile         = ''
probaugfile         = '/Data/' + username + '/full_datasets/neon_sixfold/mbfergus32/huge_probs.h5'

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
    neighbor_only=True, pad_prob_svox_perim=True,
    verbose=verbose)

# hack to save raveled indices of overlap in context of whole volume (including boundary)
# boundary size is saved in frag.eperim
frag.ovlp_attrs += ['ovlp_cur_dilate']

# get list of edges after creation of rag
list_of_edges = np.zeros((size_of_edges,2), dtype=np.uint32)

#final count of the number of edgees and borders generated by the rag
count = np.zeros(2, dtype=np.uint32)

#change supervoxel count to uint64 so that to avoid overflow
hybrid_adjacency = np.zeros((np.uint32(label_count)*frag.nsupervox) ,dtype = np.uint8)

# calculate the jump steps requied for 1X dilation to get neighborhood
neigh_sel_size = (2*frag.neighbor_perim)+1
dilate_array = np.zeros((neigh_sel_size,neigh_sel_size,neigh_sel_size))
dilate_array[neigh_sel_size//2,neigh_sel_size//2,neigh_sel_size//2] = 1;
binary_struct=scipy.ndimage.morphology.generate_binary_structure(frag.supervoxels.ndim, frag.connectivity)
neigh_sel = scipy.ndimage.morphology.binary_dilation(dilate_array, binary_struct, frag.neighbor_perim)
neigh_sel_indices = np.transpose(np.nonzero(neigh_sel))
neigh_sel_indices = neigh_sel_indices - (neigh_sel_size//2)
#print(neigh_sel_indices.shape)

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
steps = np.array([(x[0]*frag.supervoxels.shape[1]*frag.supervoxels.shape[2] + x[1]*frag.supervoxels.shape[2] + x[2]) \
                  for x in neigh_sel_indices])

#calculate indices for 2X dilation
steps_border = np.array([(y[0]*frag.supervoxels.shape[1]*frag.supervoxels.shape[2] + y[1]*frag.supervoxels.shape[2] \
                          + y[2]) for y in compliment_index])
#steps_border = steps_border[np.nonzero(steps_border)]

# build the rag
print('\nCpp serial generation of rag'); t=time.time()
#print(frag.supervoxels.shape)
FRAG_extension.build_frag(frag.supervoxels, frag.nsupervox, frag.connectivity, np.uint32(size_of_edges), list_of_edges,
                          bool(validate),steps, bool(adjacencyMatrix), np.uint32(label_count), count, hybrid_adjacency)
print('done in %.4f s'%  (time.time() - t))

# new build the rag
print('\nCpp new serial generation of rag'); t=time.time()
list_of_edges_new,list_of_borders_new,steps_new,max_step = build_frag_new(frag.supervoxels, frag.nsupervox, pad=False,
                                                      conn=frag.connectivity, nbhd=frag.neighbor_perim)
print(len(list_of_borders_new),sum([x.size for x in list_of_borders_new]))

#fout = open("tmp-boundary_pixel_indices-serial-new.txt","w")
#edges = [tuple(x) for x in list_of_edges_new]
##edges.sort()
#for edge,i in zip(edges,range(list_of_edges_new.shape[0])):
#    fout.write("(%d, %d): "%(edge[0],edge[1]))
#    for b in list_of_borders_new[i]:
#      fout.write("%d "%b)
#    fout.write("\n")
#fout.close()

#print(list_of_edges_new)
print('\tdone in %.4f s'%  (time.time() - t))
print('Found %d edges' %  (list_of_edges_new.shape[0],))

#allocate space for frag border data structure
list_of_borders = np.zeros((count[0], size_of_borders), dtype=np.uint32)
tmp_edges = list_of_edges[0:count[0],:]
list_of_borders[:,0] = tmp_edges[:,0]
list_of_borders[:,1] = tmp_edges[:,1]
list_of_borders[:,2] = 3;

# tmp data structure for edges calculated for each label
'''tmp_lab_edges = np.zeros((frag.nsupervox,np.int(tmp_edge_size)), dtype = np.uint32)
tmp_lab_edges[:,0] = 2;
tmp_lab_edges[:,1] = np.arange(1,frag.nsupervox+1)
for i in range(0, count[0]):
    tmp_lab_edges[list_of_edges[i][0]-1,tmp_lab_edges[list_of_edges[i][0]-1,0]] = list_of_edges[i][1]
    tmp_lab_edges[list_of_edges[i][0]-1,0] += 1
'''

if no_dilation:
    print('\nCpp serial generation of borders for rag using nearest neigh');
    t=time.time()
    FRAG_extension.build_frag_borders_nearest_neigh(frag.supervoxels, frag.nsupervox, list_of_borders, count,
                                                    bool(validate),steps)
    print('border calculation done in %.4f s'% (time.time() - t))
else:
    print('\nCpp serial generation of borders for rag using dilation'); t=time.time()
    FRAG_extension.build_frag_borders(frag.supervoxels, frag.nsupervox, tmp_edges, list_of_borders, count,
                                      bool(validate), steps, steps_border  )
    print('border calculation done in %.4f s'% (time.time() - t))


# create graph

if do_cpu_rag:
    print('\nPython FRAG with features %s' % ('on' if getFeatures else 'off',));
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
            global_subs_padded = boundary_subs + start_sub
            global_inds = np.ravel_multi_index(global_subs_padded.T.reshape(3,-1), frag.supervoxels.shape)
            for b in global_inds:
              fout.write("%d "%b)
            #global_subs_unpadded = boundary_subs + start_sub - frag.eperim
            '''for b in range(global_subs_unpadded.shape[0]):
                fout.write("(%d,%d,%d) " % tuple(global_subs_unpadded[b,:].tolist()))'''
            fout.write("\n")
        fout.close()
        print('\tdone in %.4f s' % (time.time() - t))

if validate:
    file_name = 'tmp-edge-list-cpu.txt'
    reference_edges = np.fromfile(file_name, dtype=np.int32, sep=" ").reshape((-1,2), order='C')
    #list_of_edges[np.nonzero(list_of_edges)].reshape((count[0],2)) # ??
    ref = [tuple(a) for a in reference_edges]
    gen = [tuple(b) for b in list_of_edges if np.all(b != (0,0))]
    unique_indices = set(ref) & set(gen)
    false_negatives = set(ref) - unique_indices
    false_positives = set(gen) - unique_indices
    false_positives = list(false_positives)
    false_negatives = list(false_negatives)
    if(false_positives == [] and false_negatives == []):
      print("\nThe edge list matches for this dataset.")
    else:
      print(false_positives)
      print(false_negatives)

    fn = 'tmp-boundary_pixel_indices-cpu.txt'
    ref = []
    all_pass = True
    import re
    for line in open(fn, 'r'):
        ref.append(np.uint32(re.findall('\d+', line)))

    reference_borders = [x_r for x_r in ref]
    print(len(reference_borders),sum([x.size for x in reference_borders]))
    generated_borders = [np.concatenate((x_g[0:2], x_g[3:x_g[2]])) for x_g in list_of_borders]
    for i in range(0,count[0]):
          if(np.all(generated_borders[i] == reference_borders[i])):
              pass
          else:
              all_pass = False
              print(generated_borders[i])
              print(reference_borders[i])

    if all_pass:
        print("the borders match for this dataset")
    else:
        print("mismatch")
