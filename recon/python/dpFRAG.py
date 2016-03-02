#!/usr/bin/env python

# The MIT License (MIT)
# 
# Copyright (c) 2016 Paul Watkins, National Institutes of Health / NINDS
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Create a Featured Region Adjacency Graph (FRAG) for an oversegmentation (supervoxels) from inputs:
#   (1) supervoxels
#   (2) probabilities
#   (3) raw EM data
#   
# Output either:
#   (1) training set by providing ground truth labels. training set exported in scikit-learn data format.
#   (2) agglomerated labels by providing testing data set targets also in scikit-learn data format.
# 
# NOTE: instead of debugging / validating with original label inputs, use relabel-seq option to dpLoadh5.py.


#import os, sys
import argparse
import time
import numpy as np
from numpy import linalg as nla
from scipy import ndimage as nd
from scipy import linalg as sla
#from skimage import morphology as morph
from skimage.segmentation import relabel_sequential
import networkx as nx
from io import StringIO
from collections import OrderedDict

from dpLoadh5 import dpLoadh5
from dpWriteh5 import dpWriteh5
from typesh5 import emLabels, emProbabilities, emVoxelType
#from pyCext import binary_warping

try:
    from progressbar import ProgressBar, Percentage, Bar, ETA, RotatingMarker
    _dpFRAG__useProgressBar = True
except:
    _dpFRAG__useProgressBar = False                            

class dpFRAG(emLabels):

    TARGETS = {'no_merge':0, 'yes_merge':1}     # hard coded as false/true throughout code
    
    # different features setups / options... xxx - work on this
    log_size = True
    FEATURES = {'size_small':0, 'size_large':1, 'size_overlap':2, 
        'mean_grayscale':3, 'mean_prob_MEM':4, 'mean_prob_ICS':5, 
        'ang_cntr':6, 'dist_cntr_small':7, 'dist_cntr_large':8, 
        'size_ovlp_small':9, 'size_ovlp_large':10, 'labeled_ovlp':11,
        'conv_overlap':12, 'rad_std_ovlp':13, 'ang_std_ovlp':14, 
        'pca_angle0':15, 'pca_angle_small0':16, 'pca_angle_large0':17,
        'pca_angle1':18, 'pca_angle_small1':19, 'pca_angle_large1':20,
        'pca_angle2':21, 'pca_angle_small2':22, 'pca_angle_large2':23,
        }
        
    #FEATURES = {'size_overlap':0, 'mean_prob_MEM':1}
    FEATURES = OrderedDict(sorted(FEATURES.items(), key=lambda t: t[1]))
    FEATURES_NAMES = list(FEATURES.keys())

    # augmented inputs derived from probabilities and raw EM
    # xxx - revisit this is some way of make features as options, might need to move to .ini based init?
    augments = ['smooth', 'sharpen', 'edges', 
        'blur15', 'blur20', 'blur3', 'blur4', 'blur5', 'blur6', 
        'median', 'mean', 'min', 'max', 'var',
        'grad_mag', 'grad_dir', 'laplacian', 'large_hess', 'small_hess', 'hess_ori',
        'kuwahara', 'diff_blur',
        ]
    
    # number of pca eigenvector angles to measure
    npcaang = 3 
    
    # the probability types to use for features
    types = ['MEM','ICS']
    #types = ['MEM']

    # how much to dilate supervoxels in order to find neighbors.
    # xxx - never found it useful to include neighbors that are further out than adjacent voxels.
    neighbor_perim = 1

    # xxx - consistently found max connectivity the best for finding neighbors
    connectivity = 3

    # xxx - usefulness of this parameter is unclear
    ovlp_dilate = 0

    # for optimizations that prevent some feature variables from being recalculated
    #   if they did not change during an agglomeration.

    # these are items that are calculated during createFRAG for features.
    # some of them can be saved between agglomerations as an optimization for speed.
    # these are lists of local variables that are saved in some instances where all of the features can not be
    #   preserved, but some of the overlap calculations can still be preserved.
    svox_attrs = ['pbnd','svox_size','lsvox_size','svox_sel_out']
    sovlp_attrs = ['sel_size','lsel_size','C','V','angles','Cpts']
    ovlp_attrs = ['mean_probs','mean_grayscale','aobnd','ovlp_rmom','ovlp_amom','ovlp_conv','ovlp_labeled']

    def __init__(self, args):
        emLabels.__init__(self,args)
        
        # save the command line argument dict as a string
        out = StringIO(); print( vars(args), file=out )
        self.arg_str = out.getvalue(); out.close()

        if not self.data_type_out: self.data_type_out = self.data_type

        self.ntypes = len(self.types)
        self.nfeatures = len(self.FEATURES)
        self.bperim = 2*max([self.ovlp_dilate, dpFRAG.neighbor_perim])
        # external perimeter used to pad all volumes
        self.eperim = self.perim + self.bperim
        self.bwconn = nd.morphology.generate_binary_structure(dpLoadh5.ND, self.connectivity)
        self.outRAG = None
        self.naugments = len(self.augments)
        self._augments = ['_' + x for x in self.augments]

        assert( not self.trainout or self.gtfile )  # need ground truth to generate training data

        # print out all initialized variables in verbose mode
        #if self.dpFRAG_verbose: print('dpFRAG, verbose mode:\n'); print(vars(self))

    def loadData(self):
        if self.dpFRAG_verbose:
            print('Loading data'); t = time.time()
            
        # amount for padding around edges
        spad = tuple((np.ones((3,2),dtype=np.int32)*self.eperim[:,None]).tolist()); self.spad = spad

        # load the supervoxel label data
        self.readCubeToBuffers()
        if self.remove_ECS:
            self.data_cube[self.data_cube > self.data_attrs['types_nlabels'][0]] = 0
        relabel, fw, inv = relabel_sequential(self.data_cube)
        self.nsupervox = inv.size - 1; self.data_cube = np.zeros((0,))
        self.supervoxels_noperim = relabel
        self.supervoxels = np.lib.pad(relabel, self.spad, 'constant',
            constant_values=0).astype(self.data_type_out, copy=False)
        # supervoxel sizes are needed in advance for createFRAG (ignore background size).
        self.svox_sizes = emLabels.getSizesMax(self.supervoxels_noperim, self.nsupervox)[1:]
    
        # load the probability data
        if self.probfile:
            if not self.load_prob_perim: offset = self.offset; size = self.size
            else: offset = self.offset - self.eperim; size = self.size + 2*self.eperim
            
            self.probs = [None]*self.ntypes
            self.probs_aug = [[None]*self.ntypes for x in range(self.naugments)]
            for i in range(self.ntypes):
                loadh5 = dpLoadh5.readData(srcfile=self.probfile, dataset=self.types[i], chunk=self.chunk.tolist(), 
                    offset=offset.tolist(), size=size.tolist(), data_type=emProbabilities.PROBS_STR_DTYPE, 
                    verbose=self.dpLoadh5_verbose); data = loadh5.data_cube
                    
                if not self.load_prob_perim:
                    # pad data, xxx - what to pad with, zeros just easy, not clear any other method is better
                    self.probs[i] = np.lib.pad(data, spad, 'constant',constant_values=0)
                else:
                    self.probs[i] = data
                    
                '''
                for j in range(self.naugments):
                    loadh5 = dpLoadh5.readData(srcfile=self.probaugfile, dataset=self.types[i]+self.augments[j], 
                        chunk=self.chunk.tolist(), offset=offset.tolist(), size=size.tolist(), 
                        verbose=self.dpLoadh5_verbose); data = loadh5.data_cube

                    if not self.load_prob_perim:
                        # pad data, xxx - what to pad with, zeros just easy, not clear any other method is better
                        self.probs_aug[j][i] = np.lib.pad(data, spad, 'constant',constant_values=0)
                    else:
                        self.probs_aug[j][i] = data
                '''

        # load the raw em data
        if self.rawfile:
            if self.pad_raw_perim: offset = self.offset; size = self.size
            else: offset = self.offset - self.eperim; size = self.size + 2*self.eperim
        
            loadh5 = dpLoadh5.readData(srcfile=self.rawfile, dataset=self.raw_dataset, chunk=self.chunk.tolist(), 
                offset=offset.tolist(), size=size.tolist(), verbose=self.dpLoadh5_verbose); data = loadh5.data_cube
            
            if self.pad_raw_perim: 
                # pad data, xxx - what to pad with, zeros just easy, not clear any other method is better
                self.raw = np.lib.pad(data, spad, 'constant',constant_values=0)
            else:
                self.raw = data

            print(self._augments, self.naugments)
            '''
            self.raw_aug = [None]*self.naugments)
            for j in range(self.naugments):
                loadh5 = dpLoadh5.readData(srcfile=self.rawaugfile, dataset=self.raw_dataset+self.augments[j], 
                    chunk=self.chunk.tolist(), offset=offset.tolist(), size=size.tolist(), 
                    verbose=self.dpLoadh5_verbose); data = loadh5.data_cube

                if self.pad_raw_perim: 
                    # pad data, xxx - what to pad with, zeros just easy, not clear any other method is better
                    self.raw_aug[j] = np.lib.pad(data, spad, 'constant',constant_values=0)
                else:
                    self.raw_aug[j] = data
            '''

        # load the ground truth data
        if self.gtfile:
            loadh5 = emLabels.readLabels(srcfile=self.gtfile, chunk=self.chunk.tolist(), 
                offset=self.offset.tolist(), size=self.size.tolist(), verbose=self.dpLoadh5_verbose)
            if self.remove_ECS and self.gt_ECS_label != 0:
                if self.gt_ECS_label > 0:
                    loadh5.data_cube[loadh5.data_cube == self.gt_ECS_label] = 0
                else:
                    loadh5.data_cube[loadh5.data_cube == loadh5.data_cube.max()] = 0
            relabel, fw, inv = relabel_sequential(loadh5.data_cube); self.ngtlbl = inv.size - 1
            self.gt = np.lib.pad(relabel, spad, 'constant',constant_values=0)
        else:
            self.gt = None; self.ngtlbl = -1

        if self.dpFRAG_verbose:
            print('\tdone in %.4f s, %d supervoxels, %d gt labels' % (time.time() - t, self.nsupervox, self.ngtlbl))

    def createFRAG(self, features=True, update=False):
        # get bounding boxes for each supervoxel
        self.svox_bnd = nd.measurements.find_objects(self.supervoxels)

        # use update to only calculate features for nodes and neighbors in FRAG updated by agglomerate()
        make_outRAG = False
        if update and hasattr(self,'FRAG') and self.FRAG is not None:
            assert( self.nsupervox == self.FRAG.number_of_nodes() )
        else:
            # create emtpy FRAG, using networkx for now for graph
            self.FRAG = nx.Graph(); self.FRAG.add_nodes_from(range(1,self.nsupervox+1))

            if update:
                # keep another graph that contains the outlbls neighbors
                self.outRAG = nx.Graph(); self.outRAG.add_nodes_from(range(1,self.nsupervox+1))
                make_outRAG = True; update = False  # this is first pass, so acting like not update mode

        # other inits for the supervoxel iteration loop
        mean_probs = [None]*self.ntypes

        if self.dpFRAG_verbose:
            print('Creating FRAG'); t = time.time()
            useProgressBar = __useProgressBar and features and self.progress_bar
            if useProgressBar:
                running_size = 0
                #total_size = (self.supervoxels > 0).sum(dtype=np.int64)  # use svox volume
                total_size = self.nsupervox     # use number of supervoxels
                widgets = [RotatingMarker(), ' ', Percentage(), ' ', Bar(marker='='), ' ', ETA()]
                pbar = ProgressBar(widgets=widgets, maxval=total_size).start()

        # because of the way these features are calculated, with the first pass requiring the most computations,
        #   it's much faster to run in increasing order of supervoxel size.
        #svox_order = range(1,self.nsupervox+1)     # given order, much slower
        #svox_order = np.argsort(self.svox_sizes)[::-1] + 1    # order of decreasing supervoxel size, slowest
        svox_order = np.argsort(self.svox_sizes) + 1    # order of increasing supervoxel size, fastest
        assert( svox_order.size == self.nsupervox )     # supervoxel count/sizes not updated properly

        # iterate over supervoxels and get neighbors (edges) and add features for each neighbor
        for i in svox_order:
            # if this is a FRAG update, then skip if all the edges coming out of this node have features.
            # also can't have any features that are in first_pass.
            # NOTE: if there are no neighbors left for this supervoxel, no need to compute supervoxel attributes.
            #   In this special case the supervoxel is skipped because all([]) evaluates to True.
            if update:
                if all([('features' in x and not x['first_pass']) for x in self.FRAG[i].values()]): continue
                previ = self.nsupervox_prev + i

            # if the supervoxel itself has not changed but some neighbors have, load from node attributes.
            if update and 'svox_attrs' in self.FRAG.node[i]:
                n = self.FRAG.node[i]['svox_attrs']

                # this has to be redone because some of the neighbors inside of pbnd have changed
                svox_cur = self.supervoxels[n['pbnd']]
            else:
                # bound with perimeter around current supervoxel
                pbnd = tuple([slice(x.start-self.bperim,x.stop+self.bperim) for x in self.svox_bnd[i-1]])
                svox_cur = self.supervoxels[pbnd]
            
                # select the curent supervoxel and dilate it out by perim
                svox_sel = (svox_cur == i)
                #svox_size = svox_sel.sum(dtype=np.int64); assert( self.svox_sizes[i-1] == svox_size )
                svox_size = self.svox_sizes[i-1]    # optimization for speed with sizes kept updated
                lsvox_size = self.voxel_size_xform(svox_size)
                svox_sel_out = nd.morphology.binary_dilation(svox_sel, structure=self.bwconn, 
                    iterations=dpFRAG.neighbor_perim)

                # save the variables in svox_attrs to node attributes
                d = locals(); n = { k:d[k] for k in dpFRAG.svox_attrs }; self.FRAG.node[i]['svox_attrs'] = n

            # get the neighbors for this supervoxel
            nbrlbls = np.unique(svox_cur[n['svox_sel_out']])
            # do not add edges to background or to the same supervoxel
            nbrlbls = nbrlbls[np.logical_and(nbrlbls != i, nbrlbls != 0)]

            # udpate the progress bar based on the current supervoxel size divided by its number of neighbors.
            # add all remainders to last update for this supervoxel.
            #if self.dpFRAG_verbose and useProgressBar:
            #    if nbrlbls.size > 0:
            #        bar_update = n['svox_size'] // nbrlbls.size
            #        bar_final = n['svox_size'] - bar_update * (nbrlbls.size-1)
            #    else:
            #        running_size += n['svox_size']
                
            # add each corresponding neighbor label to the FRAG
            for j in nbrlbls:
                if update: prevj = self.nsupervox_prev + j
                
                if not features:
                    # only make RAG without features (like for agglomeration without fit)
                    if not self.FRAG.has_edge(i,j):
                        self.FRAG.add_edge(i,j); self.FRAG[i][j]['features'] = None
                    continue

                # add the edge if it does not exist, otherwise update any features for the current supervoxel
                has_edge = self.FRAG.has_edge(i,j); has_features = has_edge and 'features' in self.FRAG[i][j]
                if not has_edge or not has_features:
                    # if the edge is already there and the features are missing, this should only be in the mode
                    #   of updating features in the current FRAG.
                    if has_features: assert(update)
                    else: self.FRAG.add_edge(i,j)

                    # if this edge contains overlap info that means the FRAG was updated to preserve this overlap.
                    # this is the case if the overlap itself didn't change, but the neighboring supervoxels did not.
                    loadovlp = update and 'ovlp_attrs' in self.FRAG[i][j]
                    if loadovlp:
                        m = self.FRAG[i][j]['ovlp_attrs']; mo = m

                        # this has to be redone because some of the neighbors 
                        #   inside of the bounding box may have changed.
                        ovlp_svox_cur = self.supervoxels[m['aobnd']]
                    else:
                        # dilate the other supervoxel by the same perimeter amount within this bounding box.
                        # use the binary overlap between the two dilations as the boundary for these neighbors.
                        svox_ovlp = np.logical_and(n['svox_sel_out'], nd.morphology.binary_dilation((svox_cur == j), 
                            structure=self.bwconn, iterations=dpFRAG.neighbor_perim))
                            
                        # create another bounding box arond the overlap area. this is simply an optimization for the
                        #   mean features, as this will be a smaller cropped area. in addition, used to calculate
                        #   more completex features of the objects and the overlap in the local overlap volume.
                        obnd = nd.measurements.find_objects(svox_ovlp)[0]
                        # convert bounding box back to entire volume space.
                        aobnd = tuple([slice(x.start+y.start-self.perim[k],
                            x.stop+y.start+self.perim[k]) for x,y,k in zip(obnd,n['pbnd'],range(dpLoadh5.ND))])

                        # get the supervoxels within the overlap bounding box.                            
                        ovlp_svox_cur = self.supervoxels[aobnd]
                        
                        # convert the overlap to within the same bounding box.
                        ovlp_cur = np.zeros(ovlp_svox_cur.shape,dtype=np.bool)
                        ovlp_cur[tuple([slice(x,-x) for x in self.perim])] = svox_ovlp[obnd]
                        
                        # xxx - this is an alternative method that does not work as well. 
                        #   replace this to use old method along with outRAG so that overlap is recomputed if the
                        #   overlap is polluted by a non-adjacent neighbor. keeping other method here for reference:
                        ## IMPORTANT: consistently did not find benefit of including neighbors that are not directly
                        ##   adjacent. Avoid these from "polluting" the overlap, which means that the overlap can be
                        ##   conserved between agglomerations (i.e., the small non-neighbor issue). Do this by only
                        ##   counting the area that is also contained within the supervoxels as overlap.
                        ##ovlp_cur = np.logical_and(ovlp_cur, np.logical_or(ovlp_svox_cur==i, ovlp_svox_cur==j))
                        ##assert( ovlp_cur.sum(dtype=np.int64) > 0 ) # commented for speed, will error below anyways

                        # SIMPLEST FEATURES: calculate mean features in the overlapping area between the neighbors.

                        # optionally further dilate the overlap in order to increase averaging area for boundaries.
                        if self.ovlp_dilate > 0:
                            ovlp_cur_dilate = nd.morphology.binary_dilation(ovlp_cur, structure=self.bwconn, 
                                iterations=self.ovlp_dilate)
                        else:
                            ovlp_cur_dilate = ovlp_cur
                        
                        for k in range(self.ntypes):
                            mean_probs[k] = self.probs[k][aobnd][ovlp_cur_dilate].mean(dtype=np.double)
                        mean_grayscale = self.raw[aobnd][ovlp_cur_dilate].mean(dtype=np.double)

                        # MORE COMPLEX FEATURES: object attributes within the overlap bounding box.
                        mo = self.getOvlpAttrs(ovlp_cur, self.sampling, return_Cpts=True)

                        # percentage of voxels in the overlap area that are labeled (not background).
                        ovlp_labeled = (ovlp_svox_cur[ovlp_cur] > 0).sum(dtype=np.double)/mo['sel_size']

                        # radial standard deviation of the overlap from the centroid
                        ovlp_rmom = np.std(nla.norm(mo['Cpts'], axis=1))
                        # angular standard deviation of the overlap from the first principle component
                        ovlp_amom = np.std(np.arctan2(nla.norm(np.cross(mo['Cpts'],mo['V'][0,:]),axis=1), 
                            np.dot(mo['Cpts'],mo['V'][0,:])))
                        mo['Cpts'] = None  # no need for this potentially large item to persist

                        # simple "convexity" measure, compare size of overlap with that of overlap bounding box
                        ovlp_conv = mo['sel_size']/np.array([x.stop-x.start for x in obnd]).prod(dtype=np.double)
                        #assert(ovlp_conv <= 1)     # silly sanity check, HIASSERT

                        # save the variables in ovlp_attrs to edge attributes
                        d = locals(); m = { k:d[k] for k in dpFRAG.ovlp_attrs }
                        # concatenate sovlp_attrs and ovlp_attrs for the overlap
                        self.FRAG[i][j]['ovlp_attrs'] = {**m, **mo}

                    # more complicated optimization. if the overlap is preserved then calculations on one of the
                    #   supervoxels might also be preserved if it was not agglomerated in the previous iteration.
                    loadi = False; loadj = False
                    if update and 'sovlp_attrs' in self.FRAG[i][j]:
                        # avoid second pass to renumber nodes in agglomerate because of relabel
                        loadi = previ in self.FRAG[i][j]['sovlp_attrs']; loadj = prevj in self.FRAG[i][j]['sovlp_attrs']
                        assert( not loadi or not loadj )     # both sovlp_attrs should not happen
                        assert( (not loadi and not loadj) or loadovlp )     # sovlp_attrs without ovlp_attrs
                    else:
                        self.FRAG[i][j]['sovlp_attrs'] = {}
                    if loadi:
                        mi = self.FRAG[i][j]['sovlp_attrs'][previ]
                        self.FRAG[i][j]['sovlp_attrs'] = {i:mi}     # move to current supervoxel value
                    else:
                        mi = self.getOvlpAttrs(ovlp_svox_cur == i, self.sampling, Vother=mo['V'])
                        self.FRAG[i][j]['sovlp_attrs'][i] = mi
                    if loadj:
                        mj = self.FRAG[i][j]['sovlp_attrs'][prevj]
                        self.FRAG[i][j]['sovlp_attrs'] = {j:mj}     # move to current supervoxel value
                    else:
                        mj = self.getOvlpAttrs(ovlp_svox_cur == j, self.sampling, Vother=mo['V'])
                        self.FRAG[i][j]['sovlp_attrs'][j] = mj
                    
                    # calculate angles between corresponding supervoxel eigenvectors within overlap box
                    angles_ij = np.zeros((dpFRAG.npcaang,),np.double)
                    for k in range(dpFRAG.npcaang):
                        angles_ij[k] = np.arctan2(nla.norm(np.cross(mi['V'][k,:],mj['V'][k,:])), 
                            np.dot(mi['V'][k,:],mj['V'][k,:]))

                    # angle / distance of vectors from overlap centroid to object centroids
                    Vi = mi['C'] - mo['C']; Vj = mj['C'] - mo['C']
                    # https://newtonexcelbach.wordpress.com/2014/03/01/the-angle-between-two-vectors-python-version/
                    oanglec = np.arctan2(nla.norm(np.cross(Vi,Vj)), np.dot(Vi,Vj))
                    odistci = nla.norm(Vi); odistcj = nla.norm(Vj)

                    # xxx - this feature is expensive and seemed marginally useful at best.
                    ### total number of other labels in the overlap area
                    ##otherlbls = np.unique(ovlp_svox_cur[m['ovlp_cur']])
                    ### do not count background or the labels currently involved in this edge
                    ##otherlbls = otherlbls[np.logical_and(np.logical_and(otherlbls != i, otherlbls != j),otherlbls != 0)]

                    # set all the features except the size of the neighbor label for current object
                    F = self.FEATURES
                    f = {
                        F['size_overlap']:mo['lsel_size'],
                        F['mean_grayscale']:m['mean_grayscale'],
                        F['ang_cntr']:oanglec, 
                        F['rad_std_ovlp']:m['ovlp_rmom'],
                        F['ang_std_ovlp']:m['ovlp_amom'],
                        F['conv_overlap']:m['ovlp_conv'],
                        F['labeled_ovlp']:m['ovlp_labeled'],
                        #F['other_ovlp']:otherlbls.size,
                        # these features might need to be swapped depending on which object is larger
                        F['size_small']:n['lsvox_size'],
                        F['dist_cntr_small']:odistci, 
                        F['dist_cntr_large']:odistcj, 
                        F['size_ovlp_small']:mi['lsel_size'],
                        F['size_ovlp_large']:mj['lsel_size'],
                        }
                    for k in range(dpFRAG.npcaang):
                        f[F['pca_angle' + str(k)]] = angles_ij[k]
                        # these features might need to be swapped depending on which object is larger
                        f[F['pca_angle_small' + str(k)]] = mi['angles'][k]
                        f[F['pca_angle_large' + str(k)]] = mj['angles'][k]
                    for k in range(self.ntypes):
                        f[F['mean_prob_' + self.types[k]]] = m['mean_probs'][k]
                    self.FRAG[i][j]['features'] = f; self.FRAG[i][j]['first_pass'] = True
                else: # if edge already in graph
                    # if this is update mode and the edge is there and not marked as first pass, 
                    #   then this is an edge that does not need feature udpates at all (copied from previous FRAG).
                    if update:
                        if not self.FRAG[i][j]['first_pass']: continue
                    else:
                        assert( self.FRAG[i][j]['first_pass'] )     # meh, sth really wrong
                    
                    # update this edge for second pass (the other neighbor is now the current neighbor)
                    f = self.FRAG[i][j]['features']; F = self.FEATURES; self.FRAG[i][j]['first_pass'] = False

                    # features are stored with size sorted by large / small object, so set features accordingly.
                    if f[F['size_small']] <= n['lsvox_size']:
                        # set the size of the neighbor label for current object, current is larger.
                        f[F['size_large']] = n['lsvox_size']
                    else:
                        # set the size of the neighbor label for current object, current is smaller.
                        f[F['size_large']] = f[F['size_small']]; f[F['size_small']] = n['lsvox_size']
                            
                        # features that were set originally with i object as small, j is actually smaller, so swap
                        f[F['dist_cntr_small']], f[F['dist_cntr_large']] = \
                            f[F['dist_cntr_large']], f[F['dist_cntr_small']]
                        f[F['size_ovlp_small']], f[F['size_ovlp_large']] = \
                            f[F['size_ovlp_large']], f[F['size_ovlp_small']]
                        f[F['dist_cntr_small']], f[F['dist_cntr_large']] = \
                            f[F['dist_cntr_large']], f[F['dist_cntr_small']]
                        for k in range(dpFRAG.npcaang):
                            str_small = 'pca_angle_small' + str(k); str_large = 'pca_angle_large' + str(k)
                            f[F[str_small]], f[F[str_large]] = f[F[str_large]], f[F[str_small]]

                    # xxx - this sanity check was used during development of this function.
                    #   would need some major updates to verify overlap is the same in both directions...
                    # overlap features here should be the same as done before, can verify with asserts for debug
                    #if self.FRAG[i][j]['features']['overlap'] != size_ovlp:
                    #    print(i,j,size_ovlp,self.FRAG[i][j]['features']['overlap']); assert(False)
    
                # update progress bar based on size of all neighbors added so far
                #if self.dpFRAG_verbose and useProgressBar:
                #    running_size += (bar_update if j != nbrlbls[-1] else bar_final); pbar.update(running_size)

            if make_outRAG:
                # dilate out to the area that would overlap when the other supervoxel is dilated.
                # supervoxels in this area that are not adjacent neighbors these will not be included in the 
                #   FRAG but need to be taken into account in optimization to not recompute overlap (ovlp_attrs).
                svox_sel_outer = nd.morphology.binary_dilation(svox_sel_out, structure=self.bwconn, 
                    iterations=dpFRAG.neighbor_perim)
            
                # get all supervoxels that would overlap with this one with symmetric dilations (includes neighbors).
                outlbls = np.unique(svox_cur[svox_sel_outer])
                # again do not include background or current supervoxel
                outlbls = outlbls[np.logical_and(outlbls != i, outlbls != 0)]

                # add any other labels that are not neighbors but are in symmetric overlap region into a separate graph.
                # this is maintained because these need to be counted as neighbors (ovlp_attrs)
                for j in outlbls:
                    if not self.FRAG.has_edge(i,j): self.outRAG.add_edge(i,j)
            
            if self.dpFRAG_verbose and useProgressBar:
                running_size += 1; pbar.update(running_size)
                
        if self.dpFRAG_verbose: 
            if useProgressBar: pbar.finish()                        
            print('\n\tdone in %.4f s' % (time.time() - t))

    def createDataset(self, train=True):    
        if train:
            # count overlap with gt for each supervoxel. ignore background in gt.
            # xxx - could make this as an option or use some percentage parameters here.
            #   decided for now not much point as this is not useful if the supervoxels are "decent"
            # use overlap to map each supervoxel to the gt label with the greatest overlap
            supervox_to_gt = np.argmax(np.stack(nd.measurements.histogram(self.gt, 1, self.ngtlbl, self.ngtlbl, 
                labels=self.supervoxels, index=np.arange(1, self.nsupervox+1)),axis=1).T, axis=1)+1
        
            # optionally output the mapping from supervoxel to GT and any unmapped GT labels
            if self.mapout:
                # list of gt labels that are mapped to from a supervoxel
                mapped_gt = np.unique(supervox_to_gt)
                
                # get bounding boxes for each ground truth
                gt_bnd = nd.measurements.find_objects(self.gt)
            
                out = open(self.mapout, 'w')
                out.write('\nunmapped relabeled GT at bnd (x,y,z with index starting at 1):\n')
                for i in range(1,self.ngtlbl+1):
                    if i not in mapped_gt:
                        out.write('%d at ' % i)
                        print([slice(x.start-self.eperim[j]+1,x.stop+self.eperim[j]+1) for x,j in zip(gt_bnd[i-1], 
                            range(3))],file=out)
                out.write('\nSVOX (%d) => GT (%d) at SVOX bound (with index starting at 1):\n' % (self.nsupervox, 
                    self.ngtlbl))
                for i in range(1,self.nsupervox+1):
                    out.write('%d => %d at ' % (i, supervox_to_gt[i-1]))
                    print([slice(x.start-self.eperim[j]+1,x.stop+self.eperim[j]+1) for x,j in zip(self.svox_bnd[i-1], 
                        range(3))],file=out)
                out.close()

        # initialize the training set in scikit-learn format
        #dict_keys(['feature_names', 'DESCR', 'target_names', 'target', 'data'])
        ntargets = self.FRAG.number_of_edges()
        target = np.zeros((ntargets,), dtype=np.int64)
        fdata = np.zeros((ntargets,self.nfeatures), dtype=np.double)
        
        if self.dpFRAG_verbose:
            print('Creating training set from %d edges' % ntargets); t = time.time()
        
        # iterate the network edges and create a scikit-learn style training set
        for e,i in zip(self.FRAG.edges_iter(data=True),range(ntargets)):
            if train: 
                #if supervox_to_gt[e[0]-1] == supervox_to_gt[e[1]-1]:
                #    target[i] = self.TARGETS['yes_merge']
                #else:
                #    target[i] = self.TARGETS['no_merge']
                target[i] = (supervox_to_gt[e[0]-1] == supervox_to_gt[e[1]-1])
            if e[2]['features'] is not None:
                fdata[i,:] = np.array([e[2]['features'][x] for x in range(self.nfeatures)], dtype=np.double)

        #dict_keys(['feature_names', 'DESCR', 'target_names', 'target', 'data'])
        descr = 'Training data from dpFRAG.py with command line:\n' + self.arg_str
        data = {'feature_names':self.FEATURES, 'DESCR':descr, 'target_names':self.TARGETS, 'target':target, 
            'data':fdata}

        if self.dpFRAG_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))

        return data

    # single target agglomerate, for debug or creating "perfect" agglomeration from supervoxels based on GT.
    # this is like a loopback test for this script only (create FRAG using GT then agglomerate based on GT overlap).
    # also used by iterative supervoxel agglomeration method.
    # this method updates the FRAG (but not the FRAG edge features) based on the target agglomeration.
    def agglomerate(self, target):
        if self.keep_subgroups:
            # the agglomerated labels depend on the input labels, so keep that information in output subgroups
            if self.subgroups_out: 
                self.subgroups_out = self.subgroups + self.subgroups_out
            else:
                self.subgroups_out = self.subgroups
        
        ntargets = target.size
        assert( ntargets == self.FRAG.number_of_edges() ); # agglomerate input data must match FRAG edges

        if self.dpFRAG_verbose:
            print('Doing single agglomeration from supplied targets'); t = time.time()

        # single agglomerate with binary targets
        #self._incremental_agglomerate(target)

        # create agglomerated graph that only contains yes merge edges
        aggloG = nx.Graph(); aggloG.add_nodes_from(range(1,self.nsupervox+1))
        for e,i in zip(self.FRAG.edges_iter(),range(ntargets)):
            if target[i] and not aggloG.has_edge(e[0],e[1]): aggloG.add_edge(e[0],e[1])

        # get connected component nodes, create supervoxel mapping and update FRAG based on agglomerated components
        supervox_map = np.zeros((self.nsupervox+1,),dtype=self.data_type_out)
        compsG = nx.connected_components(aggloG); ncomps = 0
        svox_sizes = np.zeros((self.nsupervox,),dtype=np.int64)
        for nodes in compsG:
            # SINGLETON NODE: single supervoxel that is not undergoing any merging (agglomeration)
        
            # supervoxel mapping from previous to new agglomerated supervoxels, update supervoxel sizes
            npnodes = np.array(tuple(nodes),dtype=np.int64); svox_sizes[ncomps] = self.svox_sizes[npnodes-1].sum()
            ncomps += 1; supervox_map[npnodes] = ncomps
            
            # update the FRAG by creating a new agglomerated node and moving neighboring edges to this node.
            # make a new singleton node for if this node is not containing any agglomerations.
            newnode = ncomps+self.nsupervox; self.FRAG.add_node(newnode)
            if self.outRAG is not None: self.outRAG.add_node(newnode)

            if len(nodes) == 1:
                # SINGLETON COMPONENT NODE: single supervoxel that is not being merged with anything.
                # this is the only time that features may be copied over from old FRAG.
                # i.e., this supervoxel did not change.
                node = next(iter(nodes))    # the singleton node in old FRAG

                # if this is a singleton node (not agglomerated) then re-add the node attributes.
                # this saves time in recalculating them in createFRAG as this supervoxel itself has not changed.
                # NOTE: if there are no neighbors left for this supervoxel, no need to compute supervoxel attributes.
                #   In this special case the supervoxel is skipped in the createFRAG loop.
                #   So no need to save the supervoxel attributes in the node.
                if len(self.FRAG[node]) > 0: self.FRAG.node[newnode]['svox_attrs'] = self.FRAG.node[node]['svox_attrs']

                self._noagglo_iterate_neighbors(self.FRAG, node, newnode, copy_attrs=True)
                if self.outRAG is not None:
                    self._noagglo_iterate_neighbors(self.outRAG, node, newnode)
                    
            else:
                # MUTIPLE COMPONENT NODE: group of supervoxels being merged (agglomerated)
            
                # copy over overlap attributes for neighbors that border on only one of the nodes in the agglomerate.
                # in this case the overlap area has not changed. need two passes to check for this.
                neighbors = {}
            
                # each node in connected component (that compose the agglomerated newnode)
                for node in nodes:
                    self._agglo_iterate_neighbors(self.FRAG, node, nodes, neighbors, newnode)
                    if self.outRAG is not None: 
                        self._agglo_iterate_neighbors(self.outRAG, node, nodes, neighbors, newnode, remove=True)

                for node in nodes:
                    # for each neighbor of this node (edge in FRAG)
                    for neighbor,f in self.FRAG[node].items():
                        # copy over overlap attributes for neighbors that border on only one of agglomerated nodes.
                        if neighbor not in nodes and neighbors[neighbor] == 1:
                            if 'ovlp_attrs' in f: 
                                self.FRAG[neighbor][newnode]['ovlp_attrs'] = f['ovlp_attrs'] 
                            # copy the neighbor supervoxel overlap attributes over.
                            # they might be preservable in this case if the neighbor is a singleton node.
                            if 'sovlp_attrs' in f and neighbor in f['sovlp_attrs']:
                                self.FRAG[neighbor][newnode]['sovlp_attrs'] = {neighbor: f['sovlp_attrs'][neighbor]}
                                #assert( 'ovlp_attrs' in f ) # should not have sovlp_attrs without ovlp_attrs, HIASSERT
                    # after visiting, remove this old node that makes up part of the agglomerated node.
                    self.FRAG.remove_node(node)
                
        # sanity checks
        assert( ncomps == self.FRAG.number_of_nodes() )
        #assert( self.nsupervox+ncomps == max(self.FRAG.nodes()) )  # commented for speed, HIASSERT
        # relabel the FRAG starting at supervoxel 1
        self.FRAG = nx.relabel_nodes(self.FRAG, {x+self.nsupervox:x for x in range(1,ncomps+1)}, copy=False)
        
        # same steps for outRAG as for FRAG
        if self.outRAG is not None: 
            assert( ncomps == self.outRAG.number_of_nodes() )
            #assert( self.nsupervox+ncomps == max(self.outRAG.nodes()) )  # commented for speed, HIASSERT
            self.outRAG = nx.relabel_nodes(self.outRAG, {x+self.nsupervox:x for x in range(1,ncomps+1)}, copy=False)
                    
        # create/write the new supervoxels from the supervoxel_map containing mapping from old nodes to agglo nodes.
        self.data_cube = supervox_map[self.supervoxels_noperim]
        verbose = self.dpWriteh5_verbose; self.dpWriteh5_verbose = self.dpFRAG_verbose;
        self.data_attrs['types_nlabels'] = [ncomps]
        self.writeCube(); self.dpWriteh5_verbose = verbose

        # other self updates for the new supervoxels
        self.supervoxels_noperim = self.data_cube
        self.supervoxels = np.lib.pad(self.data_cube, self.spad, 'constant',
            constant_values=0).astype(self.data_type_out, copy=False)
        self.nsupervox_prev = self.nsupervox; self.nsupervox = ncomps; self.data_cube = np.zeros((0,))
        self.svox_sizes = svox_sizes[:ncomps]

        if self.dpFRAG_verbose:
            print('\tnsupervox',ncomps)
            print('\tdone in %.4f s' % (time.time() - t, ))

    # "macro" for iterating neighbors for singleton nodes (not being agglomerated) in FRAG and outRAG
    def _noagglo_iterate_neighbors(self, cRAG, node, newnode, copy_attrs=False):
        # for each neighbor of this node (edge in FRAG)
        for neighbor,f in cRAG[node].items():
            #assert( not cRAG.has_edge(neighbor, newnode) )     # sanity check for singleton node, HIASSERT
            cRAG.add_edge(neighbor, newnode)   # move this edge to newnode
            
            # copying features and attributes only happens for FRAG
            if copy_attrs:
                # both new nodes must be singleton components to not require any feature updates.
                # edges not requiring feature updates are marked by having features copied over from previous FRAG.
                # only copy the features if the other node was already visited and was also singleton (has features)
                #   or if the other node was not visited yet (neighbor node <= previous nsupervox).
                if neighbor <= self.nsupervox or 'features' in f:
                    features_copied = True
                    cRAG[neighbor][newnode]['features'] = f['features']
                    cRAG[neighbor][newnode]['first_pass'] = False
                else:
                    features_copied = False

                # copy over overlap attributes but only if features are not copied over to final new edge.
                if neighbor <= self.nsupervox or not features_copied:
                    # if the overlap attributes are present, then copy them over to the new edge.
                    if 'ovlp_attrs' in f: 
                        cRAG[neighbor][newnode]['ovlp_attrs'] = f['ovlp_attrs']
                    # copy the supervoxel overlap attributes over for this singleton node
                    if 'sovlp_attrs' in f and node in f['sovlp_attrs']:
                        cRAG[neighbor][newnode]['sovlp_attrs'] = {newnode: f['sovlp_attrs'][node]}
                        #assert( 'ovlp_attrs' in f )     # should not have sovlp_attrs without ovlp_attrs, HIASSERT

        # remove the old singleton node
        cRAG.remove_node(node)

    # "macro" for iterating neighbors for compound nodes (those being agglomerated) in FRAG and outRAG
    def _agglo_iterate_neighbors(self, cRAG, node, nodes, neighbors, newnode, remove=False):
        # for each neighbor of this node (edge in cRAG)
        for neighbor in cRAG[node].keys():
            # skip neighbors that are other nodes of this component.
            if neighbor not in nodes:
                # move this edge to newnode
                if not cRAG.has_edge(neighbor, newnode): cRAG.add_edge(neighbor, newnode)   
                # add or increment neighbor count
                if neighbor not in neighbors:
                    neighbors[neighbor] = 1
                else:
                    neighbors[neighbor] += 1
        if remove: cRAG.remove_node(node)

    # multiple probability thresholded agglomerate.
    # this method does NOT update the FRAG based on the target agglomeration.
    def threshold_agglomerate(self, probs, thresholds, threshold_subgroups=None):
        ntargets = probs.shape[0]; nthresholds = len(thresholds)
        assert( ntargets == self.FRAG.number_of_edges() ); # agglomerate input data must match FRAG edges
        if threshold_subgroups is None:
            threshold_subgroups = thresholds
        else:
            assert( nthresholds == len(threshold_subgroups) )   # output subgroups must match length of actual thrs

        self_subgroups_out = self.subgroups_out
        if self.keep_subgroups:
            # the agglomerated labels depend on the input labels, so keep that information in output subgroups
            if self.subgroups_out: 
                self.subgroups_out = self.subgroups + self.subgroups_out + ['thr']
            else:
                self.subgroups_out = self.subgroups + ['thr']
        else:
            self.subgroups_out += ['thr']
        verbose = self.dpWriteh5_verbose; self.dpWriteh5_verbose = False;

        if self.dpFRAG_verbose:
            useProgressBar = __useProgressBar and self.progress_bar
            print('Threshold agglomeration for thresholds %s' % (' '.join([str(x) for x in thresholds]),))
            t = time.time()
            if useProgressBar:
                widgets = [RotatingMarker(), ' ', Percentage(), ' ', Bar(marker='='), ' ', ETA()]
                pbar = ProgressBar(widgets=widgets, maxval=nthresholds).start()

        # create empty agglomerated graph
        aggloG = nx.Graph(); aggloG.add_nodes_from(range(1,self.nsupervox+1))
        
        # do incremental agglomerate with decreasing thresholds, store each one into output hdf5
        thresholds = np.sort(thresholds)[::-1]; threshold_subgroups = np.sort(threshold_subgroups)[::-1]
        for i in range(nthresholds):
            self.subgroups_out[-1] = '%.8f' % threshold_subgroups[i]
            
            #self._incremental_agglomerate(probs[:,1] > thresholds[i], aggloG)
            
            target = probs[:,1] > thresholds[i]
            # add edges in target if not already present in aggloG
            for e,j in zip(self.FRAG.edges_iter(),range(ntargets)):
                if target[j] and not aggloG.has_edge(e[0],e[1]): aggloG.add_edge(e[0],e[1])

            # get connected component nodes and create supervoxel mapping based on agglomerated components
            supervox_map = np.zeros((self.nsupervox+1,),dtype=self.data_type_out)
            compsG = nx.connected_components(aggloG); ncomps = 0
            for nodes in compsG:
                # supervoxel mapping from previous to new agglomerated supervoxels
                ncomps += 1; supervox_map[np.array(tuple(nodes),dtype=np.int64)] = ncomps
            # create the new supervoxels from the supervoxel_map containing mapping from agglo nodes to new nodes
            self.data_cube = supervox_map[self.supervoxels_noperim]

            if self.dpFRAG_verbose: 
                print('nsupervox',ncomps)
                if useProgressBar: pbar.update(i)
            #print(self.offset, self.size, self.chunk, self.data_type, self.data_type_out)
            self.data_attrs['types_nlabels'] = [ncomps]
            self.writeCube()
                
        if self.dpFRAG_verbose: 
            if useProgressBar: pbar.finish()                        
            print('\n\tdone in %.4f s' % (time.time() - t))
        
        self.subgroups_out = self_subgroups_out
        self.dpWriteh5_verbose = verbose

    def voxel_size_xform(self, size):
        return np.log10(size.astype(np.double)) if self.log_size else size.astype(np.double)

    # macro for computing attributes of supervoxels and overlap that are within the overlap bounding box.
    # these are used for the more complex features in createFRAG.
    # called for the two neighboring supervoxels and the overlap area.
    def getOvlpAttrs(self, sel, sampling, Vother=None, return_Cpts=False):
        # size of supervoxel or overlap selection
        sel_size = sel.sum(dtype=np.int64)
        lsel_size = self.voxel_size_xform(sel_size)

        # get the point coordinates within the overlap bounding box.
        # use the sampling resolution for the points if available.
        pts = np.transpose(np.nonzero(sel)).astype(np.double)*sampling

        # get the centroid and center points
        C = np.mean(pts, axis=0); Cpts = pts-C
                        
        # get the principal axes.
        V = dpFRAG.getOrthoAxes(Cpts, sampling)

        # only return points if needed
        if not return_Cpts: Cpts = None
                    
        # calculate angles between corresponding eigenvectors
        if Vother is None:
            angles = None
        else:
            angles = np.zeros((dpFRAG.npcaang,),np.double)
            for k in range(dpFRAG.npcaang):
                angles[k] = np.arctan2(nla.norm(np.cross(V[k,:],Vother[k,:])), np.dot(V[k,:],Vother[k,:]))

            # rigid transform rotation decomposed to Euler angles.
            # xxx - this method did not work as well for classification. look at this again in the future?
            #R, t = dpFRAG.rigid_transform_3D(V, Vother); angles = dpFRAG.decompose_rotation(R)

        # save the variables in dict to return
        d = locals(); return { k:d[k] for k in dpFRAG.sovlp_attrs }

    # this is just svd to get eigenvectors of points, but handling degenerate cases by forcing each voxel into 3d by
    #   expanding each point to 6 points at the center of each voxel face (based on voxel size, sampling).
    # NOTE IMPORTANT: from scipy, svd different from matlab:
    #   "The SVD is commonly written as a = U S V.H. The v returned by this function is V.H and u = U."
    # pts is Nx3
    # returns Vt, the eigenvectors along the rows (along axis 0, i.e., Nx3 where N=3 with full rank)
    @staticmethod
    def getOrthoAxes(pts, sampling):
        assert(pts.shape[1] == 3)     # coordinates along axis 1
        if pts.shape[0] > 3:
            U, S, Vt = sla.svd(pts,overwrite_a=False,full_matrices=False)
            convertPts = (Vt.shape[0] < 3)
        else:
            convertPts = True
        if convertPts:
            newpts = np.vstack((pts,pts,pts,pts,pts,pts)); npts = pts.shape[0]; cnt = 0
            for d in range(pts.shape[1]):
                for sign in [-1,1]:
                    newpts[cnt:cnt+npts,d] += sign*sampling[d]/2; cnt = cnt + npts
            U, S, Vt = sla.svd(newpts,overwrite_a=True,full_matrices=False)
            assert(Vt.shape[0] == 3)
        return Vt
    
    # modified from http://nghiaho.com/uploads/code/rigid_transform_3D.py_
    #       https://en.wikipedia.org/wiki/Kabsch_algorithm
    #       http://nghiaho.com/?page_id=671
    # Input: expects Nx3 matrix of points
    # t = 3x1 column vector
    # returns R = 3x3 rotation matrix
    @staticmethod
    def rigid_transform_3D(A, B):
        assert len(A) == len(B)
        cA = A.mean(axis=0, dtype=np.double); cB = B.mean(axis=0, dtype=np.double)
        AA = A - cA; BB = B - cB
        U, S, Vt = sla.svd(np.dot(AA.T,BB),overwrite_a=True,full_matrices=False); V = Vt.T

        R = np.dot(V,U.T)
        if nla.det(R) < 0: 
            V[:,2] = -V[:,2]; R = np.dot(V,U.T)
        t = -R*cA.T + cB.T
        return R, t

    # modified from http://nghiaho.com/uploads/code/rotation_matrix_demo.m    
    #       https://en.wikipedia.org/wiki/Euler_angles
    #       http://nghiaho.com/?page_id=846
    @staticmethod
    def decompose_rotation(R):
        A = np.zeros((3,),np.double)    # xyz
        A[0] = np.arctan2(R[2,1], R[2,2])
        A[1] = np.arctan2(-R[2,0], np.sqrt(R[2,1]*R[2,1] + R[2,2]*R[2,2]))
        A[2] = np.arctan2(R[1,0], R[0,0])
        return A
        
    @classmethod
    def makeTrainingFRAG(cls, labelfile, chunk, size, offset, probfile, rawfile, raw_dataset, gtfile, 
            subgroups=[], G=None, progressBar=False, verbose=False):
        parser = argparse.ArgumentParser(description='class:dpFRAG', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        dpFRAG.addArgs(parser); arg_str = ''

        arg_str += ' --srcfile ' + labelfile
        arg_str += ' --chunk %d %d %d ' % tuple(chunk)
        arg_str += ' --offset %d %d %d ' % tuple(offset)
        arg_str += ' --size %d %d %d ' % tuple(size)
        if subgroups: arg_str += ' --subgroups %s ' % ' '.join(subgroups)
        arg_str += ' --probfile ' + probfile
        arg_str += ' --rawfile ' + rawfile
        arg_str += ' --raw-dataset ' + raw_dataset
        arg_str += ' --gtfile ' + gtfile
        
        if verbose: arg_str += ' --dpFRAG-verbose '
        if progressBar: arg_str += ' --progress-bar '
        if verbose: print(arg_str)
        args = parser.parse_args(arg_str.split())
        frag = cls(args); 
        frag.loadData()
        frag.FRAG = G
        return frag

    @classmethod
    def makeTestingFRAG(cls, labelfile, chunk, size, offset, probfile, rawfile, raw_dataset, outfile, subgroups=[], 
            subgroups_out=[], G=None, progressBar=False, verbose=False):
        parser = argparse.ArgumentParser(description='class:dpFRAG', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        dpFRAG.addArgs(parser); arg_str = ''

        arg_str += ' --srcfile ' + labelfile
        arg_str += ' --chunk %d %d %d ' % tuple(chunk)
        arg_str += ' --offset %d %d %d ' % tuple(offset)
        arg_str += ' --size %d %d %d ' % tuple(size)
        arg_str += ' --probfile ' + probfile
        arg_str += ' --rawfile ' + rawfile
        arg_str += ' --raw-dataset ' + raw_dataset
        arg_str += ' --outfile ' + outfile
        if subgroups: arg_str += ' --subgroups %s ' % ' '.join(subgroups)
        if subgroups_out: arg_str += ' --subgroups-out %s ' % ' '.join(subgroups_out)
        
        if verbose: arg_str += ' --dpFRAG-verbose '
        if progressBar: arg_str += ' --progress-bar '
        if verbose: print(arg_str)
        args = parser.parse_args(arg_str.split())
        frag = cls(args); 
        frag.loadData()
        frag.FRAG = G
        return frag

    @classmethod
    def makeBothFRAG(cls, labelfile, chunk, size, offset, probfile, rawfile, raw_dataset, gtfile, outfile,
            subgroups=[], subgroups_out=None, G=None, progressBar=False, verbose=False):
        parser = argparse.ArgumentParser(description='class:dpFRAG', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        dpFRAG.addArgs(parser); arg_str = ''

        arg_str += ' --srcfile ' + labelfile
        arg_str += ' --chunk %d %d %d ' % tuple(chunk)
        arg_str += ' --offset %d %d %d ' % tuple(offset)
        arg_str += ' --size %d %d %d ' % tuple(size)
        arg_str += ' --probfile ' + probfile
        arg_str += ' --rawfile ' + rawfile
        arg_str += ' --raw-dataset ' + raw_dataset
        arg_str += ' --gtfile ' + gtfile
        arg_str += ' --outfile ' + outfile
        if subgroups: arg_str += ' --subgroups %s ' % ' '.join(subgroups)
        if subgroups_out: arg_str += ' --subgroups-out %s ' % ' '.join(subgroups_out)
        
        if verbose: arg_str += ' --dpFRAG-verbose '
        if progressBar: arg_str += ' --progress-bar '
        if verbose: print(arg_str)
        args = parser.parse_args(arg_str.split())
        frag = cls(args); 
        frag.loadData()
        frag.FRAG = G
        return frag


    @staticmethod
    def addArgs(p):
        dpWriteh5.addArgs(p)
        p.add_argument('--probfile', nargs=1, type=str, default='', help='Path/name of hdf5 probability (input) file')
        p.add_argument('--probaugfile', nargs=1, type=str, default='', 
            help='Path/name of hdf5 augmented probability (input) file')
        #p.add_argument('--types', nargs='+', type=str, default=['MEM','ICS','ECS'], 
        #    metavar='TYPE', help='Dataset names of the voxel types to use from the probabilities')
        p.add_argument('--rawfile', nargs=1, type=str, default='', help='Path/name of hdf5 raw EM (input) file')
        p.add_argument('--rawaugfile', nargs=1, type=str, default='', help='Path/name of hdf5 augmented raw EM file')
        p.add_argument('--raw-dataset', nargs=1, type=str, default='data', help='Name of the raw EM dataset to read')
        p.add_argument('--gtfile', nargs=1, type=str, default='', 
            help='Path/name of ground truth (GT) labels (create training data)')
        p.add_argument('--trainout', nargs=1, type=str, default='', help='Output file for dumping training data (dill)')
        p.add_argument('--testin', nargs=1, type=str, default='', help='Input file for loading testing data (dill)')
        p.add_argument('--perim', nargs=3, type=int, default=[16,16,8], choices=range(1,20),
            help='Size of bounding box around overlap for object features')
        p.add_argument('--remove-ECS', dest='remove_ECS', action='store_true', 
            help='Set to remove ECS supervoxels (set to 0)')
        p.add_argument('--gt-ECS-label', nargs=1, type=int, default=[1], 
            help='Which label is ECS in GT for remove-ECS (-1 is last label, 0 is none)')
        p.add_argument('--mapout', nargs=1, type=str, default='', help='Optional text dump of supervox to GT mapping')
        #p.add_argument('--ovlp-dilate', nargs=1, type=int, default=[1], choices=range(0,20),
        #    help='Amount to dilate overlap for calculating boundary features')
        #p.add_argument('--connectivity', nargs=1, type=int, default=[3], choices=[1,2,3],
        #    help='Connectivity for binary morphology operations')
        p.add_argument('--keep-subgroups', action='store_true', 
            help='Keep subgroups for labels in path for subgroups-out')
        p.add_argument('--progress-bar', action='store_true', help='Enable progress bar if available')
        p.add_argument('--pad-raw-perim', action='store_true', help='Pad perimeter of raw EM instead of loading')
        p.add_argument('--load-prob-perim', action='store_true', help='Load perimeter of probs instead of padding')
        p.add_argument('--dpFRAG-verbose', action='store_true', help='Debugging output for dpFRAG')

if __name__ == '__main__':
    import dill

    parser = argparse.ArgumentParser(description='Create Featured Region Adjacency Graph (FRAG) for supervoxels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpFRAG.addArgs(parser)
    args = parser.parse_args()
    
    frag = dpFRAG(args)
    frag.loadData()
    if frag.trainout and not frag.testin:
        frag.createFRAG()
        data = frag.createDataset()
        print('Dumping training data')
        with open(frag.trainout, 'wb') as f: dill.dump(data, f)
    elif frag.testin and not frag.trainout:
        frag.createFRAG(features=False)
        frag.createDataset(train=False)
        print('Loading testing data')
        with open(frag.testin, 'rb') as f: data = dill.load(f)
        frag.agglomerate(data['target'])
    else:
        assert(False)   # only specify training output or testing input on command line
        
