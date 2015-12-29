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
from scipy import ndimage as nd
from scipy import linalg as la
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

    TARGETS = {'no_merge':0, 'yes_merge':1}     # hard coded as true and false below
    
    # different features setups / options... xxx - work on this
    log_size = True
    FEATURES = {'size_small':0, 'size_large':1, 'size_overlap':2, 'mean_grayscale':3,
        'mean_prob_MEM':4, 'mean_prob_ICS':5, 'angle0':6, 'angle1':7, 'angle2':8}
    #FEATURES = {'size_overlap':0, 'mean_prob_MEM':1}
    FEATURES = OrderedDict(sorted(FEATURES.items(), key=lambda t: t[1]))
    FEATURES_NAMES = list(FEATURES.keys())
    
    # making this fixed for now, otherwise need "optional" features
    types = ['MEM','ICS']
    #types = ['MEM']

    # number of angles to measure between neighboring object orientations
    nangles = 3

    def __init__(self, args):
        emLabels.__init__(self,args)
        
        # save the command line argument dict as a string
        out = StringIO(); print( vars(args), file=out )
        self.arg_str = out.getvalue(); out.close()

        if not self.data_type_out: self.data_type_out = self.data_type

        self.ntypes = len(self.types)
        self.nfeatures = len(self.FEATURES)
        self.bperim = 2*self.perim
        # external perimeter used to pad all volumes
        self.eperim = self.operim + self.bperim
        self.bwconn = nd.morphology.generate_binary_structure(dpLoadh5.ND, self.connectivity)

        assert( not self.trainout or self.gtfile )  # need ground truth to generate training data

        # print out all initialized variables in verbose mode
        #if self.dpFRAG_verbose: print('dpFRAG, verbose mode:\n'); print(vars(self))

    def loadSupervoxels(self):
        # load the supervoxel label data
        self.readCubeToBuffers()
        if self.remove_ECS:
            self.data_cube[self.data_cube > self.data_attrs['types_nlabels'][0]] = 0
        relabel, fw, inv = relabel_sequential(self.data_cube)
        self.nsupervox = inv.size - 1; self.data_cube = np.zeros((0,))
        self.supervoxels = np.lib.pad(relabel, self.spad, 'constant',constant_values=0).astype(self.data_type_out, 
            copy=False)

    def loadData(self):
        if self.dpFRAG_verbose:
            print('Loading data'); t = time.time()
            
        # amount for zero padding around edges
        spad = tuple((np.ones((3,2),dtype=np.int32)*self.eperim[:,None]).tolist()); self.spad = spad

        self.loadSupervoxels()
    
        # load the probability data
        if self.probfile:
            self.probs = [None]*self.ntypes
            for i in range(self.ntypes):
                loadh5 = dpLoadh5.readData(srcfile=self.probfile, dataset=self.types[i], chunk=self.chunk.tolist(), 
                    offset=self.offset.tolist(), size=self.size.tolist(), data_type=emProbabilities.PROBS_STR_DTYPE, 
                    verbose=self.dpLoadh5_verbose)
                self.probs[i] = np.lib.pad(loadh5.data_cube, spad, 'constant',constant_values=0); del loadh5

        # load the raw em data
        if self.rawfile:
            loadh5 = dpLoadh5.readData(srcfile=self.rawfile, dataset=self.raw_dataset, chunk=self.chunk.tolist(), 
                offset=self.offset.tolist(), size=self.size.tolist(), verbose=self.dpLoadh5_verbose)
            self.raw = np.lib.pad(loadh5.data_cube, spad, 'constant',constant_values=0); del loadh5

        # load the ground truth data
        if self.gtfile:
            loadh5 = emLabels.readLabels(srcfile=self.gtfile, chunk=self.chunk.tolist(), 
                offset=self.offset.tolist(), size=self.size.tolist(), verbose=self.dpLoadh5_verbose)
            #loadh5 = dpLoadh5.readData(srcfile=self.gtfile, dataset=self.gt_dataset, chunk=self.chunk.tolist(), 
            #    offset=self.offset.tolist(), size=self.size.tolist(), verbose=self.dpLoadh5_verbose)
            if self.remove_ECS and self.gt_ECS_label != 0:
                if self.gt_ECS_label > 0:
                    loadh5.data_cube[loadh5.data_cube == self.gt_ECS_label] = 0
                else:
                    loadh5.data_cube[loadh5.data_cube == loadh5.data_cube.max()] = 0
            relabel, fw, inv = relabel_sequential(loadh5.data_cube); self.ngtlbl = inv.size - 1
            self.gt = np.lib.pad(relabel, spad, 'constant',constant_values=0); del loadh5
        else:
            self.gt = None; self.ngtlbl = -1

        if self.dpFRAG_verbose:
            print('\tdone in %.4f s, %d supervoxels, %d gt labels' % (time.time() - t, self.nsupervox, self.ngtlbl))

    def createFRAG(self, features=True):
        # get bounding boxes for each supervoxel
        self.svox_bnd = nd.measurements.find_objects(self.supervoxels)

        # create emtpy FRAG, using networkx for now for graph
        self.FRAG = nx.Graph(); self.FRAG.add_nodes_from(range(1,self.nsupervox+1))

        # other inits for the supervoxel iteration loop
        mean_probs = [None]*self.ntypes

        if self.dpFRAG_verbose:
            print('Creating FRAG'); t = time.time()
            if __useProgressBar:
                running_size = 0; total_size = (self.supervoxels > 0).sum(dtype=np.int64)
                widgets = [RotatingMarker(), ' ', Percentage(), ' ', Bar(marker='='), ' ', ETA()]
                pbar = ProgressBar(widgets=widgets, maxval=total_size).start()

        # iterate over supervoxels and get neighbors (edges) and add features for each neighbor
        for i in range(1,self.nsupervox+1):
            # bound with perimeter around current supervoxel
            pbnd = tuple([slice(x.start-self.bperim,x.stop+self.bperim) for x in self.svox_bnd[i-1]])
            svox_cur = self.supervoxels[pbnd]
            
            # select the curent supervoxel and dilate it out by perim
            svox_sel = (svox_cur == i); svox_size = svox_sel.sum(dtype=np.int64)
            lsvox_size = self.voxel_size_xform(svox_size)
            svox_sel_out = nd.morphology.binary_dilation(svox_sel, structure=self.bwconn, iterations=self.perim)
            nbrlbls = np.unique(svox_cur[svox_sel_out])

            # do not add edges to background or to the same supervoxel
            nbrlbls = nbrlbls[np.logical_and(nbrlbls != i, nbrlbls != 0)]
            
            # udpate the progress bar based on the current supervoxel size divided by its number of neighbors.
            # add all remainders to last update for this supervoxel.
            if self.dpFRAG_verbose and __useProgressBar:
                if nbrlbls.size > 0:
                    bar_update = svox_size // nbrlbls.size; bar_final = svox_size - bar_update * (nbrlbls.size-1)
                else:
                    running_size += svox_size
                
            # add each corresponding neighbor label to the FRAG
            for j in nbrlbls:
            
                if features:
                    # add the edge if it does not exist, otherwise update any features for the current supervoxel
                    if not self.FRAG.has_edge(i,j):
                        self.FRAG.add_edge(i,j)
                    
                        # dilate the other supervoxel by the same perimeter amount within this bounding box.
                        # use the binary overlap between the two dilations as the boundary for these neighbors.
                        svox_ovlp = np.logical_and(svox_sel_out, nd.morphology.binary_dilation((svox_cur == j), 
                            structure=self.bwconn, iterations=self.perim))
                    
                        # calculate average features in the overlap area between the neighbors
                        for k in range(self.ntypes):
                            mean_probs[k] = self.probs[k][pbnd][svox_ovlp].mean(dtype=np.double)
                        mean_grayscale = self.raw[pbnd][svox_ovlp].mean(dtype=np.double)
                    
                        # use another bounding box around the perimeter to get features of the neighboring objects that
                        #   are to be merged in a local region near the boundary between them.
                        obnd = nd.measurements.find_objects(svox_ovlp)
                        # convert back to volume space and then grab a new volume around the overlap area
                        obnd = tuple([slice(x.start+y.start-self.operim[k],x.stop+y.start+self.operim[k]) \
                            for x,y,k in zip(obnd[0],pbnd,range(dpLoadh5.ND))])
                        ovlp_svox_cur = self.supervoxels[obnd]
                        
                        # axes of the principal components
                        sampling = self.data_attrs['scale'] if 'scale' in self.data_attrs else None
                        #Vo = dpFRAG.getSelPCAAxes(svox_ovlp,sampling)  # xxx - did not find this useful
                        Vi = dpFRAG.getSelPCAAxes(ovlp_svox_cur == i,sampling)
                        Vj = dpFRAG.getSelPCAAxes(ovlp_svox_cur == j,sampling)
                        # angles between primary components, use some out of range default if pca fails (small object)
                        adef = (1.0+1.0/9)*np.pi; angles = np.zeros((3,),np.double)
                        for k in range(self.nangles):
                            try: angles[k] = np.arctan2(la.norm(np.cross(Vi[:,k],Vj[:,k])), np.dot(Vi[:,k],Vj[:,k]))
                            except ValueError: angles[k] = adef
                            
                        self.FRAG[i][j]['features'] = {
                            self.FEATURES['size_small']:lsvox_size,
                            self.FEATURES['size_overlap']:self.voxel_size_xform(svox_ovlp.sum(dtype=np.int64)),
                            self.FEATURES['mean_grayscale']:mean_grayscale,
                            self.FEATURES['angle0']:angles[0], 
                            self.FEATURES['angle1']:angles[1], 
                            self.FEATURES['angle2']:angles[2], 
                            }
                        for k in range(self.ntypes):
                            self.FRAG[i][j]['features'][self.FEATURES['mean_prob_' + self.types[k]]] = mean_probs[k]
                    else:
                        if self.FRAG[i][j]['features'][self.FEATURES['size_small']] <= lsvox_size:
                            self.FRAG[i][j]['features'][self.FEATURES['size_large']] = lsvox_size
                        else:
                            self.FRAG[i][j]['features'][self.FEATURES['size_large']] = \
                                self.FRAG[i][j]['features'][self.FEATURES['size_small']]
                            self.FRAG[i][j]['features'][self.FEATURES['size_small']] = lsvox_size
                                                    
                        # overlap features here should be the same as done before, can verify with asserts for debug
                        #if self.FRAG[i][j]['features']['overlap'] != size_ovlp:
                        #    print(i,j,size_ovlp,self.FRAG[i][j]['features']['overlap']); assert(False)
                elif not self.FRAG.has_edge(i,j):
                    # much faster if features are not needed (like for agglomeration without fit)
                    self.FRAG.add_edge(i,j); self.FRAG[i][j]['features'] = None
    
                # update progress bar based on size of all neighbors added so far
                if self.dpFRAG_verbose and __useProgressBar:
                    running_size += (bar_update if j != nbrlbls[-1] else bar_final); pbar.update(running_size)
                
        if self.dpFRAG_verbose: 
            if __useProgressBar: pbar.finish()                        
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

    # single target agglomerate, mostly for debug or creating "perfect" agglomeration from supervoxels based on GT.
    # this is like a loopback test for this script only (create FRAG using GT then agglomerate based on GT overlap).
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

        # create empty agglomerated graph
        aggloG = nx.Graph(); aggloG.add_nodes_from(range(1,self.nsupervox+1))

        # single agglomerate with binary targets
        self._incremental_agglomerate(target, aggloG)

        if self.dpFRAG_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))

        verbose = self.dpWriteh5_verbose; self.dpWriteh5_verbose = self.dpFRAG_verbose;
        self.writeCube()
        self.dpWriteh5_verbose = verbose


    # multiple probability thresholded agglomerate
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
            print('Threshold agglomeration for thresholds %s' % (' '.join([str(x) for x in thresholds]),))
            t = time.time()
            if __useProgressBar:
                widgets = [RotatingMarker(), ' ', Percentage(), ' ', Bar(marker='='), ' ', ETA()]
                pbar = ProgressBar(widgets=widgets, maxval=nthresholds).start()

        # create empty agglomerated graph
        aggloG = nx.Graph(); aggloG.add_nodes_from(range(1,self.nsupervox+1))
        
        # do incremental agglomerate with decreasing thresholds, store each one into output hdf5
        thresholds = np.sort(thresholds)[::-1]; threshold_subgroups = np.sort(threshold_subgroups)[::-1]
        for i in range(nthresholds):
            self.subgroups_out[-1] = '%.8f' % threshold_subgroups[i]
            self._incremental_agglomerate(probs[:,1] > thresholds[i], aggloG)

            if self.dpFRAG_verbose and __useProgressBar: pbar.update(i)
            #print(self.offset, self.size, self.chunk, self.data_type, self.data_type_out)
            self.writeCube()
                
        if self.dpFRAG_verbose: 
            if __useProgressBar: pbar.finish()                        
            print('\n\tdone in %.4f s' % (time.time() - t))
        
        self.subgroups_out = self_subgroups_out
        self.dpWriteh5_verbose = verbose

    def _incremental_agglomerate(self, target, aggloG):
        ntargets = target.size
        assert( ntargets == self.FRAG.number_of_edges() ) # agglomerate input data must match FRAG edges
        assert( self.nsupervox == aggloG.number_of_nodes() ) # agglomerate graph must have node for each supervoxel

        # add edges in target if not already present in aggloG
        for e,i in zip(self.FRAG.edges_iter(data=True),range(ntargets)):
            #if target[i] == self.TARGETS['yes_merge'] and not aggloG.has_edge(e[0],e[1]):
            if target[i] and not aggloG.has_edge(e[0],e[1]):
                aggloG.add_edge(e[0],e[1])

        # allocate current output agglomerated labels
        bp = self.eperim; supervoxels = self.supervoxels[bp[0]:-bp[0],bp[1]:-bp[1],bp[2]:-bp[2]]

        # get component subgraphs and create agglomerated labels based on subgraphs
        supervox_map = np.zeros((self.nsupervox+1,),dtype=self.data_type_out)
        compsG = nx.connected_components(aggloG); ncomps = 0
        for nodes in compsG:
            ncomps += 1; supervox_map[np.array(tuple(nodes), dtype=self.data_type_out)] = ncomps
        self.data_cube = supervox_map[supervoxels]
        print('graph comps',ncomps)

    def voxel_size_xform(self, size):
        return np.log10(size.astype(np.double)) if self.log_size else size.astype(np.double)
    
    @staticmethod
    def getSelPCAAxes(sel, sampling):
        inds = np.transpose(np.nonzero(sel)).astype(np.double)
        if sampling is not None: inds *= sampling
        # center, then pca with svd to get principal axes
        inds -= inds.mean(axis=0,dtype=np.double)
        U, S, V = la.svd(inds,overwrite_a=True,full_matrices=False)
        return V

    @classmethod
    def makeTrainingFRAG(cls, labelfile, chunk, size, offset, probfile, rawfile, raw_dataset, gtfile, 
            subgroups=[], G=None, verbose=False):
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
        if verbose: print(arg_str)
        args = parser.parse_args(arg_str.split())
        frag = cls(args); 
        frag.loadData()
        frag.FRAG = G
        return frag

    @classmethod
    def makeTestingFRAG(cls, labelfile, chunk, size, offset, probfile, rawfile, raw_dataset, outfile, subgroups=[], 
            subgroups_out=[], G=None, verbose=False):
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
        if verbose: print(arg_str)
        args = parser.parse_args(arg_str.split())
        frag = cls(args); 
        frag.loadData()
        frag.FRAG = G
        return frag

    @classmethod
    def makeBothFRAG(cls, labelfile, chunk, size, offset, probfile, rawfile, raw_dataset, gtfile, outfile,
            subgroups=[], subgroups_out=None, G=None, verbose=False):
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
        #p.add_argument('--types', nargs='+', type=str, default=['MEM','ICS','ECS'], 
        #    metavar='TYPE', help='Dataset names of the voxel types to use from the probabilities')
        p.add_argument('--rawfile', nargs=1, type=str, default='', help='Path/name of hdf5 raw EM (input) file')
        p.add_argument('--raw-dataset', nargs=1, type=str, default='data', help='Name of the raw EM dataset to read')
        p.add_argument('--gtfile', nargs=1, type=str, default='', 
            help='Path/name of ground truth (GT) labels (create training data)')
        p.add_argument('--trainout', nargs=1, type=str, default='', help='Output file for dumping training data (dill)')
        p.add_argument('--testin', nargs=1, type=str, default='', help='Input file for loading testing data (dill)')
        p.add_argument('--perim', nargs=1, type=int, default=[1], choices=range(1,20),
            help='Size of perimeter around supervoxels for calculating boundary features')
        p.add_argument('--operim', nargs=3, type=int, default=[12,12,6], choices=range(1,20),
            help='Size of bounding box around overlap for object features')
        p.add_argument('--remove-ECS', dest='remove_ECS', action='store_true', 
            help='Set to remove ECS supervoxels (set to 0)')
        p.add_argument('--gt-ECS-label', nargs=1, type=int, default=[1], 
            help='Which label is ECS in GT for remove-ECS (-1 is last label, 0 is none)')
        p.add_argument('--mapout', nargs=1, type=str, default='', help='Optional text dump of supervox to GT mapping')
        p.add_argument('--connectivity', nargs=1, type=int, default=[3], choices=[1,2,3],
            help='Connectivity for binary morphology operations')
        p.add_argument('--keep-subgroups', action='store_true', 
            help='Keep subgroups for labels in path for subgroups-out')
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
        
