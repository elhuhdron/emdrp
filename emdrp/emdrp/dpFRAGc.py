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
import networkx as nx
from io import StringIO
from collections import OrderedDict

from dpLoadh5 import dpLoadh5
from dpWriteh5 import dpWriteh5
from utils.typesh5 import emLabels, emProbabilities
from Kuwahara import Kuwahara

from pyCext import frag_with_borders


class dpFRAG(emLabels):

    TARGETS = {'no_merge':0, 'yes_merge':1}     # hard coded as false/true throughout code

    ############ parameters that were found to not be useful so made static

    # how much to dilate supervoxels in order to find neighbors.
    # xxx - never found it useful to include neighbors that are further out than adjacent voxels.
    neighbor_perim = 1

    # xxx - consistently found max connectivity the best for finding neighbors
    connectivity = 3

    # xxx - usefulness of this parameter is unclear
    ovlp_dilate = 0

    # this needs to be fixed for the C-code
    lbl_dtype = np.uint32

    ## xxx - enables "nearest neighbor" only method with neighbor_perim==1.
    ##   consistently found that having this == True leads to worse segmentations metrics for agglo.
    #neighbor_only = False

    ############ dictionary keys used for saving variables for optimizations

    # these are items that are calculated during createFRAG for features.
    # some of them can be saved between agglomerations as an optimization for speed.
    # these are lists of local variables that are saved in some instances where all of the features can not be
    #   preserved, but some of the overlap calculations can still be preserved.
    #svox_attrs = ['pbnd','svox_size','lsvox_size','svox_sel_out','ppbnd','perim_ovlp','svox_sel_out_perim']
    sovlp_attrs = ['sel_size','lsel_size','C','V','angles','Cpts']
    ovlp_attrs = ['mean_probs','mean_probs_aug','mean_probs_static_aug',
                  'mean_grayscale','mean_grayscale_aug','mean_grayscale_static_aug',
                  'aobnd','rad_std_ovlp','ang_std_ovlp','conv_overlap','labeled_ovlp']

    @staticmethod
    def make_features(feature_set, has_ECS):
        # old static class properties:
        #FEATURES = {'size_small':0, 'size_large':1, 'size_overlap':2,
        #    'mean_grayscale':3, 'mean_prob_MEM':4, 'mean_prob_ICS':5,
        #    'ang_cntr':6, 'dist_cntr_small':7, 'dist_cntr_large':8,
        #    'size_ovlp_small':9, 'size_ovlp_large':10, 'labeled_ovlp':11,
        #    'conv_overlap':12, 'rad_std_ovlp':13, 'ang_std_ovlp':14,
        #    'pca_angle0':15, 'pca_angle_small0':16, 'pca_angle_large0':17,
        #    'pca_angle1':18, 'pca_angle_small1':19, 'pca_angle_large1':20,
        #    'pca_angle2':21, 'pca_angle_small2':22, 'pca_angle_large2':23,
        #    }
        #FEATURES = OrderedDict(sorted(FEATURES.items(), key=lambda t: t[1]))
        #FEATURES_NAMES = list(FEATURES.keys())

        # the variables returned as a dict by this method
        features_vars = ['log_size', 'npcaang', 'prob_types', 'nprob_types', 'augments', 'naugments',
            'features_names', 'features', 'nfeatures', 'static_augments', 'nstatic_augments']

        ############ hyperparameters controlling features

        # wheter to use log for voxel sizes of objects
        log_size = True

        if feature_set == 'all':
            # 81 total features
            static_features = ['size_small', 'size_large', 'size_overlap',
                'mean_grayscale', 'ang_cntr', 'dist_cntr_small', 'dist_cntr_large',
                'size_ovlp_small', 'size_ovlp_large', 'labeled_ovlp',
                'conv_overlap', 'rad_std_ovlp', 'ang_std_ovlp']
            npcaang = 3
            prob_types = ['MEM','ICS']
            # all of the original augments that were computed for raw EM and MEM/ICS probs (23 augments)
            #            augments = ['smooth', 'sharpen', 'edges',
            #                'blur10', 'blur15', 'blur20', 'blur30', 'blur40', 'blur50', 'blur60',
            #                'median', 'mean', 'min', 'max', 'var',
            #                'grad_mag', 'grad_dir', 'laplacian', 'large_hess', 'small_hess', 'hess_ori',
            #                'kuwahara', 'diff_blur'
            #                ]
            # if all were included, total features would be 93 features
            # notes on augments found to be colinear:
            #   smooth == sharp, blurs == diff_blur, large_hess == small_hess
            #   laplacian <==> grad_mag BUT only with some combination of blurs and smooth/edges/kuwahara??
            # after removing colinears, 19 augments:
            augments = ['smooth', 'edges', 'kuwahara',
                'blur10', 'blur15', 'blur20', 'blur30', 'blur40', 'blur50', 'blur60',
                'median', 'mean', 'min', 'max', 'var',
                'grad_mag', 'grad_dir', 'large_hess', 'hess_ori',
                ]
            static_augments = []
        elif feature_set == 'standard':
            # 36 total features
            static_features = ['size_small', 'size_large', 'size_overlap',
                'mean_grayscale', 'ang_cntr', 'dist_cntr_small', 'dist_cntr_large',
                'size_ovlp_small', 'size_ovlp_large', 'labeled_ovlp',
                'conv_overlap', 'rad_std_ovlp', 'ang_std_ovlp']
            npcaang = 3
            prob_types = ['MEM','ICS']
            augments = ['kuwahara', 'blur30', 'blur40', 'max']
            static_augments = []
        elif feature_set == 'medium':
            # 23 total features (21 without ECS)
            static_features = ['size_small', 'size_large', 'size_overlap',
                'mean_grayscale', 'ang_cntr', 'dist_cntr_small', 'dist_cntr_large',
                'size_ovlp_small', 'size_ovlp_large', 'labeled_ovlp',
                'conv_overlap', 'rad_std_ovlp', 'ang_std_ovlp']
            npcaang = 0
            prob_types = ['MEM','ICS']
            augments = []
            static_augments = ['MEM_max', 'ICS_min',
                               '_kuwahara', '_blur30', '_blur40', '_max']
            if has_ECS:
                prob_types += ['ECS']
                static_augments += ['ECS_max']
        elif feature_set == 'reduced':
            # 21 total features
            static_features = ['size_small', 'size_large', 'size_overlap',
                'mean_grayscale', 'ang_cntr', 'dist_cntr_small', 'dist_cntr_large',
                'size_ovlp_small', 'size_ovlp_large', 'labeled_ovlp',
                'conv_overlap', 'rad_std_ovlp', 'ang_std_ovlp']
            npcaang = 0
            prob_types = ['MEM','ICS']
            augments = ['blur30', 'max']
            static_augments = []
        elif feature_set == 'small':
            # 15 total features, no augments
            static_features = ['size_small', 'size_large', 'size_overlap',
                'mean_grayscale', 'ang_cntr', 'dist_cntr_small', 'dist_cntr_large',
                'size_ovlp_small', 'size_ovlp_large', 'labeled_ovlp',
                'conv_overlap', 'rad_std_ovlp', 'ang_std_ovlp']
            npcaang = 0
            prob_types = ['MEM','ICS']
            augments = []
            static_augments = []
        elif feature_set == 'minimal':
            # 6 total features
            static_features = ['size_small', 'size_large', 'size_overlap', 'mean_grayscale']
            npcaang = 0
            prob_types = ['MEM', 'ICS']
            augments = []
            static_augments = []
        else:
            assert(False)   # bad feature set

        ############ generate features based on hyperparameters

        features_names = list(static_features)
        augments = ['_' + x for x in augments]
        naugments = len(augments); nprob_types = len(prob_types)
        nstatic_augments = len(static_augments)

        for k in range(nprob_types):
            features_names += ['mean_prob_' + prob_types[k]]

        for k in range(naugments):
            features_names += ['mean_grayscale' + augments[k]]
            for j in range(nprob_types):
                features_names += ['mean_prob_' + prob_types[j] + augments[k]]

        for k in range(nstatic_augments):
            if static_augments[k][0] == '_':
                features_names += ['mean_grayscale' + static_augments[k]]
            else:
                features_names += ['mean_prob_' + static_augments[k]]

        for k in range(npcaang):
            features_names += ['pca_angle' + str(k)]
            features_names += ['pca_angle_small' + str(k)]
            features_names += ['pca_angle_large' + str(k)]

        nfeatures = len(features_names)
        features = OrderedDict([(x,y) for x,y in zip(features_names,range(nfeatures))])

        d = locals(); return { k:d[k] for k in features_vars }

    @staticmethod
    def filter_data(data, ftype, sampling_ratio):
        ftype = ftype.lower()
        # xxx - expand this out with more options, parameterize blur / others?
        if ftype=='kuwahara':
            for i in range(data.shape[2]):
                data[:,:,i] = Kuwahara(data[:,:,i], 5)
        elif ftype=='blur30':
            data = nd.gaussian_filter(data, [3.0/x for x in sampling_ratio], mode='constant')
        elif ftype=='blur40':
            data = nd.gaussian_filter(data, [4.0/x for x in sampling_ratio], mode='constant')
        elif ftype=='max':
            data = nd.maximum_filter(data, [np.ceil(5.0/x) for x in sampling_ratio], mode='constant')
        else:
            assert(False)

        return data

    def __init__(self, args):
        emLabels.__init__(self,args)

        assert( self.neighbor_only ) # the C version only computes RAG using nearest neighbor method

        # save the command line argument dict as a string
        out = StringIO(); print( vars(args), file=out )
        self.arg_str = out.getvalue(); out.close()

        if not self.data_type_out: self.data_type_out = self.data_type

        # force to keep subgroups in output if the chunk subgroup mode is set.
        # this is if training cubes with context overlap each other so need to be stored in separate datasets.
        # NOTE: MUST do this here before most inits so that dataset properties are loaded properly
        # xxx - ugly, svox context was never helping, so instead only use this for probs
        self.chunk_subgroups_txt = 'chunk_x%04d_y%04d_z%04d' % tuple(self.chunk.tolist())
        #if self.chunk_subgroups:
        #    self.keep_subgroups = True
        #    self.subgroups = ['chunk_x%04d_y%04d_z%04d' % tuple(self.chunk.tolist())] + self.subgroups
        #    self.inith5() # MUST so that dataset properties are loaded

        # create features based on feature_set mode
        d = dpFRAG.make_features(self.feature_set, self.has_ECS)
        for k,v in d.items(): setattr(self,k,v)

        # added this to "automatically" adjust perim based on anisotropy.
        # use integer sampling ratio, round down.
        s = self.sperim
        self.perim = np.array([s, s, s],dtype=np.int32) // np.fix(self.sampling_ratio).astype(np.int32)

        self.bperim = 2*max([self.ovlp_dilate, self.neighbor_perim])
        # external perimeter used to pad all volumes
        self.eperim = self.perim + self.bperim

        # setup the subgroups out for writing hdf5 outputs
        if self.keep_subgroups: self.subgroups_out += self.subgroups
        # this is just a placeholder for the agglomeration parameter
        self.subgroups_out += ['aggloparam']

        # print out all initialized variables in verbose mode
        if self.dpFRAG_verbose: print('dpFRAG, verbose mode:\n'); print(vars(self))

        # other inits asserts that we don't need to see printed
        self.bwconn = nd.morphology.generate_binary_structure(dpLoadh5.ND, self.connectivity)

        assert( not self.trainout or self.gtfile )  # need ground truth to generate training data

    def loadData(self):
        if self.dpFRAG_verbose:
            print('Loading data'); t = time.time()

        # amount for padding around edges
        spad = tuple((np.ones((3,2),dtype=np.int32)*self.eperim[:,None]).tolist()); self.spad = spad

        # load the supervoxel label data
        self.readCubeToBuffers()
        assert((np.iinfo(self.data_cube.dtype).max > self.data_cube).all())

        # optionally remove ECS supervoxels entirely (set to background)
        if self.remove_ECS and self.has_ECS:
            self.data_cube[self.data_cube > self.data_attrs['types_nlabels'][0]] = 0
        relabel, sizes, mapping = emLabels.relabel_sequential(self.data_cube, return_mapping=True)
        self.nsupervox = sizes.size; self.data_cube = np.zeros((0,)); self.svox_sizes = sizes

        # "do not merge" list allows some supervoxels to not be added to the RAG and never merged.
        # this is as opposed to training them not be be merged which would not guarantee they are never merged.
        # keep the "do not merge" supervoxels at the end (highest valued supervoxel labels).
        self.nsupervox_merge = self.nsupervox; self.nsupervox_nomerge = 0

        # optionally never merge any supervoxels that were classified as ECS on input
        if self.no_agglo_ECS and self.has_ECS:
            # this assumes the relabel sequential does not change the order of the supervoxels,
            #   except removing empty labels (which is what it does).
            #self.isECS = (mapping > self.data_attrs['types_nlabels'][0])
            self.nsupervox_merge = np.searchsorted(mapping, self.data_attrs['types_nlabels'][0])
            self.nsupervox_nomerge = self.nsupervox - self.nsupervox_merge

        self.supervoxels_noperim = relabel.astype(self.lbl_dtype, copy=False)
        self.supervoxels = np.lib.pad(self.supervoxels_noperim, self.spad, 'constant', constant_values=0)
        self.supervoxels_zeroperim = self.supervoxels

        # load the probability data
        if self.probfile:
            if self.pad_prob_perim: offset = self.offset; size = self.size
            else: offset = self.offset - self.eperim; size = self.size + 2*self.eperim

            self.probs = [None]*self.nprob_types
            self.probs_aug = [[None]*self.nprob_types for x in range(self.naugments)]
            self.probs_static_aug = [None]*self.nstatic_augments
            for i in range(self.nprob_types):
                loadh5 = dpLoadh5.readData(srcfile=self.probfile, dataset=self.prob_types[i], chunk=self.chunk.tolist(),
                    offset=offset.tolist(), size=size.tolist(), data_type=emProbabilities.PROBS_STR_DTYPE,
                    subgroups=[self.chunk_subgroups_txt] if self.chunk_subgroups else [], verbose=self.dpLoadh5_verbose)
                data = loadh5.data_cube

                if self.pad_prob_perim:
                    # pad data, xxx - what to pad with, zeros just easy, not clear any other method is better
                    self.probs[i] = np.lib.pad(data, spad, 'constant',constant_values=0.5)
                else:
                    self.probs[i] = data

                self.probs[i][np.logical_not(np.isfinite(self.probs[i]))] = 0   # no NaNs/Infs

                for j in range(self.naugments):
                    loadh5 = dpLoadh5.readData(srcfile=self.probaugfile, dataset=self.prob_types[i]+self.augments[j],
                        chunk=self.chunk.tolist(), offset=offset.tolist(), size=size.tolist(),
                        subgroups=[self.chunk_subgroups_txt] if self.chunk_subgroups else [],
                        verbose=self.dpLoadh5_verbose)
                    data = loadh5.data_cube

                    if self.pad_prob_perim:
                        # pad data, xxx - what to pad with, zeros just easy, not clear any other method is better
                        self.probs_aug[j][i] = np.lib.pad(data, spad, 'constant',constant_values=0.5)
                    else:
                        self.probs_aug[j][i] = data

                    self.probs_aug[j][i][np.logical_not(np.isfinite(self.probs_aug[j][i]))] = 0     # no NaNs/Infs

            for j in range(self.nstatic_augments):
                if self.static_augments[j][0] != '_':
                    loadh5 = dpLoadh5.readData(srcfile=self.probaugfile, dataset=self.static_augments[j],
                        chunk=self.chunk.tolist(), offset=offset.tolist(), size=size.tolist(),
                        subgroups=[self.chunk_subgroups_txt] if self.chunk_subgroups else [],
                        verbose=self.dpLoadh5_verbose)
                    data = loadh5.data_cube

                    if self.pad_prob_perim:
                        # pad data, xxx - what to pad with, zeros just easy, not clear any other method is better
                        self.probs_static_aug[j] = np.lib.pad(data, spad, 'constant',constant_values=0.5)
                    else:
                        self.probs_static_aug[j] = data

                    self.probs_static_aug[j][np.logical_not(np.isfinite(self.probs_static_aug[j]))] = 0  # no NaNs/Infs

        # load the raw em data
        if self.rawfile:
            if self.pad_raw_perim: offset = self.offset; size = self.size
            else: offset = self.offset - self.eperim; size = self.size + 2*self.eperim

            loadh5 = dpLoadh5.readData(srcfile=self.rawfile, dataset=self.raw_dataset, chunk=self.chunk.tolist(),
                offset=offset.tolist(), size=size.tolist(), verbose=self.dpLoadh5_verbose); data = loadh5.data_cube

            if self.pad_raw_perim:
                # pad data, xxx - what to pad with, zeros just easy, not clear any other method is better
                self.raw = np.lib.pad(data, spad, 'constant',constant_values=128)
            else:
                self.raw = data

            self.raw[np.logical_not(np.isfinite(self.raw))] = 0     # no NaNs/Infs

            # added capability for raw augments to be calculated on-the-fly
            if not self.rawaugfile and (self.naugments > 0 or (self.nstatic_augments > 0 and (self.nstatic_augments \
                > sum([self.static_augments[x][0] != '_' for x in range(self.nstatic_augments)])))):

                fpad=16; offset = offset - fpad; size = size + 2*fpad
                loadh5 = dpLoadh5.readData(srcfile=self.rawfile, dataset=self.raw_dataset, chunk=self.chunk.tolist(),
                    offset=offset.tolist(), size=size.tolist(), verbose=self.dpLoadh5_verbose); rdata = loadh5.data_cube

                self.raw_aug = [None]*self.naugments
                for j in range(self.naugments):
                    if self.dpFRAG_verbose:
                        print('\tFiltering raw data with %s' % (self.augments[j],)); t = time.time()
                    data = dpFRAG.filter_data(rdata, self.augments[j],
                                              self.sampling_ratio)[fpad:-fpad,fpad:-fpad,fpad:-fpad]
                    if self.dpFRAG_verbose:
                        print('\t\tdone in %.4f s' % (time.time() - t,))

                    if self.pad_raw_perim:
                        # pad data, xxx - what to pad with, zeros just easy, not clear any other method is better
                        self.raw_aug[j] = np.lib.pad(data, spad, 'constant',constant_values=128)
                    else:
                        self.raw_aug[j] = data

                    self.raw_aug[j][np.logical_not(np.isfinite(self.raw_aug[j]))] = 0   # no NaNs/Infs

                self.raw_static_aug = [None]*self.nstatic_augments
                for j in range(self.nstatic_augments):
                    if self.static_augments[j][0] == '_':
                        if self.dpFRAG_verbose:
                            print('\tFiltering raw data with %s' % (self.static_augments[j],)); t = time.time()
                        data = dpFRAG.filter_data(rdata, self.static_augments[j][1:],
                                                  self.sampling_ratio)[fpad:-fpad,fpad:-fpad,fpad:-fpad]
                        if self.dpFRAG_verbose:
                            print('\t\tdone in %.4f s' % (time.time() - t,))

                        if self.pad_raw_perim:
                            # pad data, xxx - what to pad with, zeros just easy, not clear any other method is better
                            self.raw_static_aug[j] = np.lib.pad(data, spad, 'constant',constant_values=128)
                        else:
                            self.raw_static_aug[j] = data

                        self.raw_static_aug[j][np.logical_not(np.isfinite(self.raw_static_aug[j]))] = 0 # no NaNs/Infs
            # normal raw augment loading procedure (precalculated)
            else:
                self.raw_aug = [None]*self.naugments
                for j in range(self.naugments):
                    loadh5 = dpLoadh5.readData(srcfile=self.rawaugfile, dataset=self.raw_dataset+self.augments[j],
                        chunk=self.chunk.tolist(), offset=offset.tolist(), size=size.tolist(),
                        verbose=self.dpLoadh5_verbose); data = loadh5.data_cube

                    if self.pad_raw_perim:
                        # pad data, xxx - what to pad with, zeros just easy, not clear any other method is better
                        self.raw_aug[j] = np.lib.pad(data, spad, 'constant',constant_values=128)
                    else:
                        self.raw_aug[j] = data

                    self.raw_aug[j][np.logical_not(np.isfinite(self.raw_aug[j]))] = 0   # no NaNs/Infs

                self.raw_static_aug = [None]*self.nstatic_augments
                for j in range(self.nstatic_augments):
                    if self.static_augments[j][0] == '_':
                        loadh5 = dpLoadh5.readData(srcfile=self.rawaugfile, verbose=self.dpLoadh5_verbose,
                            dataset=self.raw_dataset+self.static_augments[j],
                            chunk=self.chunk.tolist(), offset=offset.tolist(), size=size.tolist())
                        data = loadh5.data_cube

                        if self.pad_raw_perim:
                            # pad data, xxx - what to pad with, zeros just easy, not clear any other method is better
                            self.raw_static_aug[j] = np.lib.pad(data, spad, 'constant',constant_values=128)
                        else:
                            self.raw_static_aug[j] = data

                        self.raw_static_aug[j][np.logical_not(np.isfinite(self.raw_static_aug[j]))] = 0 # no NaNs/Infs

        # load the ground truth data
        if self.gtfile:
            loadh5 = emLabels.readLabels(srcfile=self.gtfile, chunk=self.chunk.tolist(),
                offset=self.offset.tolist(), size=self.size.tolist(), verbose=self.dpLoadh5_verbose)
            if self.remove_ECS and self.gt_ECS_label != 0:
                if self.gt_ECS_label > 0:
                    loadh5.data_cube[loadh5.data_cube == self.gt_ECS_label] = 0
                else:
                    loadh5.data_cube[loadh5.data_cube == loadh5.data_cube.max()] = 0
            relabel, sizes = emLabels.relabel_sequential(loadh5.data_cube); self.ngtlbl = sizes.size
            self.gt = np.lib.pad(relabel, spad, 'constant',constant_values=0)
        else:
            self.gt = None; self.ngtlbl = -1

        if self.dpFRAG_verbose:
            print('\tdone in %.4f s, %d supervoxels, %d merge-able supervoxels, %d gt labels' % (time.time() - t,
                self.nsupervox, self.nsupervox_merge, self.ngtlbl))

    def createFRAG(self, features=True, update=False):
        if self.dpFRAG_verbose:
            print('Creating FRAG'); ttime = time.time()

        #if self.dpFRAG_verbose:
        #    print('\tFind objects'); t = time.time()
        ## get bounding boxes for each supervoxel
        #self.svox_bnd = nd.measurements.find_objects(self.supervoxels, self.nsupervox_merge)
        #if self.dpFRAG_verbose:
        #    print('\t\tdone in %.4f s' % (time.time() - t))

        # use optimized C-code to calculate the RAG and the border voxels
        if self.dpFRAG_verbose:
            print('\tCalculating RAG and border voxels'); t = time.time()
        if hasattr(self, 'steps'):
            list_of_edges, list_of_borders = frag_with_borders(self.supervoxels, self.nsupervox_merge, pad=False,
                steps=self.steps, min_step=self.min_step, max_step=self.max_step)
        else:
            list_of_edges, list_of_borders, self.steps, self.min_step, self.max_step = frag_with_borders(\
                self.supervoxels, self.nsupervox_merge, pad=False, nbhd=self.neighbor_perim, conn=self.connectivity)
        if self.dpFRAG_verbose:
            print('\t\tdone in %.4f s' % (time.time() - t))
        nedges = list_of_edges.shape[0];

        if update and hasattr(self,'FRAG') and self.FRAG is not None:
            assert( self.nsupervox_merge == self.FRAG.number_of_nodes() )

            ## HIASSERT, compare graphs
            #G = nx.Graph(); G.add_nodes_from(range(1,self.nsupervox_merge+1)); G.add_edges_from(list_of_edges)
            #assert( nx.difference(self.FRAG, G).number_of_edges() == 0 )
        else:
            # initialize FRAG based on RAG computed in C-code
            # do not even include "do not merge" supervoxels as nodes in the FRAG at all
            self.FRAG = nx.Graph(); self.FRAG.add_nodes_from(range(1,self.nsupervox_merge+1));

            # this is the first iteration, add the edges calculated from the C-code
            self.FRAG.add_edges_from(list_of_edges)

            # do this init the first time incase features is turned off
            for e in self.FRAG.edges_iter(data=True): e[2]['features'] = None

            update = False  # this is first iteration, so acting like not update mode

        # only make RAG without features (like for agglomeration without fit)
        if not features: return

        # other inits for the supervoxel iteration loop
        mean_probs = [None]*self.nprob_types
        mean_probs_aug = [[None]*self.nprob_types for x in range(self.naugments)]
        mean_probs_static_aug = [None]*self.nstatic_augments
        mean_grayscale_aug = [None]*self.naugments
        mean_grayscale_static_aug = [None]*self.nstatic_augments

        if self.dpFRAG_verbose:
            print('\tCalculating features for each RAG edge'); t = time.time()

        # iterate over all the edges in the RAG and compute features
        for e in range(nedges):
            # the labels involved in this edge
            i,j = list_of_edges[e,:]

            # if this is a FRAG update, then skip if all the edges coming out of this node have features.
            # NOTE: if there are no neighbors left for this supervoxel, no need to compute supervoxel attributes.
            #   In this special case the supervoxel is skipped because all([]) evaluates to True.
            if update:
                if all([('features' in x) for x in self.FRAG[i].values()]) or \
                        all([('features' in x) for x in self.FRAG[j].values()]):
                    continue
                previ = self.prev_max_node + i; prevj = self.prev_max_node + j
                loadovlp = 'ovlp_attrs' in self.FRAG[i][j]

            # not recalculating sizes every time is big optimization, keep updated during agglo
            svox_size = self.svox_sizes[i-1]
            lsvox_size = self.voxel_size_xform(svox_size)
            ljsvox_size = self.voxel_size_xform(self.svox_sizes[j-1])

            # get bounding box of the border voxels for this edge and add overlap perimeter
            border_voxels = np.transpose(np.unravel_index(list_of_borders[e], self.supervoxels.shape))
            bmin = border_voxels.min(axis=0); bmax = border_voxels.max(axis=0)+1
            aobnd = tuple([slice(x-z, y+z) for x,y,z in zip(bmin,bmax,self.perim)])

            # get the supervoxels within the overlap bounding box.
            ovlp_svox_cur = self.supervoxels[aobnd]

            # create mask for overlap within the overlap bounding box
            ovlp_cur = np.zeros(ovlp_svox_cur.shape,dtype=bool)
            ovlp_cur.flat[np.ravel_multi_index((border_voxels - bmin + self.perim).T, ovlp_svox_cur.shape)] = 1

            # SIMPLEST FEATURES: calculate mean features in the overlapping area between the neighbors.

            # optionally further dilate the overlap in order to increase averaging area for boundaries.
            if self.ovlp_dilate > 0:
                ovlp_cur_dilate = nd.morphology.binary_dilation(ovlp_cur, structure=self.bwconn,
                                                                iterations=self.ovlp_dilate)
            else:
                ovlp_cur_dilate = ovlp_cur

            for k in range(self.nprob_types):
                mean_probs[k] = self.probs[k][aobnd][ovlp_cur_dilate].mean(dtype=np.double)
            mean_grayscale = self.raw[aobnd][ovlp_cur_dilate].mean(dtype=np.double)

            # augmented raw data and probability mean boundary features
            for k in range(self.naugments):
                mean_grayscale_aug[k] = self.raw_aug[k][aobnd][ovlp_cur_dilate].mean(dtype=np.double)
                for x in range(self.nprob_types):
                    mean_probs_aug[k][x] = \
                        self.probs_aug[k][x][aobnd][ovlp_cur_dilate].mean(dtype=np.double)

            # static augmented raw data and probability mean boundary features
            for k in range(self.nstatic_augments):
                if self.static_augments[k][0] == '_':
                    mean_grayscale_static_aug[k] = \
                        self.raw_static_aug[k][aobnd][ovlp_cur_dilate].mean(dtype=np.double)
                else:
                    mean_probs_static_aug[k] = \
                        self.probs_static_aug[k][aobnd][ovlp_cur_dilate].mean(dtype=np.double)

            # MORE COMPLEX FEATURES: object attributes within the overlap bounding box.
            # xxx - missing some optimization here not doing ovlp pca for reduced set... punting
            mo = self.getOvlpAttrs(ovlp_cur, self.sampling, return_Cpts=True)

            if 'labeled_ovlp' in self.features_names:
                # percentage of voxels in the overlap area that are labeled (not background).
                labeled_ovlp = (ovlp_svox_cur[ovlp_cur] > 0).sum(dtype=np.double)/mo['sel_size']

            if 'rad_std_ovlp' in self.features_names:
                # radial standard deviation of the overlap from the centroid
                rad_std_ovlp = np.std(nla.norm(mo['Cpts'], axis=1))
            if 'ang_std_ovlp' in self.features_names:
                # angular standard deviation of the overlap from the first principle component
                ang_std_ovlp = np.std(np.arctan2(nla.norm(np.cross(mo['Cpts'],mo['V'][0,:]),axis=1),
                    np.dot(mo['Cpts'],mo['V'][0,:])))
            mo['Cpts'] = None  # no need for this potentially large item to persist

            if 'conv_overlap' in self.features_names:
                # simple "convexity" measure, compare size of overlap with that of overlap bounding box
                conv_overlap = mo['sel_size']/(bmin-bmax).prod(dtype=np.double)
                #assert(conv_overlap <= 1)     # silly sanity check, HIASSERT

            # save the variables in ovlp_attrs to edge attributes
            d = locals(); m = {}
            for k in self.ovlp_attrs:
                if k in d: m[k] = d[k]
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
            angles_ij = np.zeros((self.npcaang,),np.double)
            for k in range(self.npcaang):
                angles_ij[k] = np.arctan2(nla.norm(np.cross(mi['V'][k,:],mj['V'][k,:])),
                    np.dot(mi['V'][k,:],mj['V'][k,:]))

            # make ang_cntr and dist_cntr_* tied together in feature sets (all or none)
            if 'ang_cntr' in self.features_names:
                # angle / distance of vectors from overlap centroid to object centroids
                Vi = mi['C'] - mo['C']; Vj = mj['C'] - mo['C']
                # https://newtonexcelbach.wordpress.com/2014/03/01/the-angle-between-two-vectors-python-version/
                m['ang_cntr'] = np.arctan2(nla.norm(np.cross(Vi,Vj)), np.dot(Vi,Vj))
                m['dist_cntr_i'] = nla.norm(Vi); m['dist_cntr_j'] = nla.norm(Vj)

            # xxx - this feature is expensive and seemed marginally useful at best.
            ### total number of other labels in the overlap area
            ##otherlbls = np.unique(ovlp_svox_cur[m['ovlp_cur']])
            ### do not count background or the labels currently involved in this edge
            ##otherlbls = otherlbls[np.logical_and(np.logical_and(otherlbls != i, otherlbls != j),otherlbls != 0)]

            # set all the features for this edge
            F = self.features
            f = { F['size_overlap']:mo['lsel_size'] }
            for k in self.features_names:
                if k in m: f[F[k]] = m[k]

            if lsvox_size < ljsvox_size:
                f[F['size_small']], f[F['size_large']] = lsvox_size, ljsvox_size
                if 'size_ovlp_small' in self.features_names:
                    f[F['size_ovlp_small']], f[F['size_ovlp_large']] = mi['lsel_size'], mj['lsel_size']
                if 'ang_cntr' in self.features_names:
                    f[F['dist_cntr_small']], f[F['dist_cntr_large']] = m['dist_cntr_i'], m['dist_cntr_j']
                for k in range(self.npcaang):
                    f[F['pca_angle_small' + str(k)]] = mi['angles'][k]
                    f[F['pca_angle_large' + str(k)]] = mj['angles'][k]
            else:
                f[F['size_small']], f[F['size_large']] = ljsvox_size, lsvox_size
                if 'size_ovlp_small' in self.features_names:
                    f[F['size_ovlp_small']], f[F['size_ovlp_large']] = mj['lsel_size'], mi['lsel_size']
                if 'ang_cntr' in self.features_names:
                    f[F['dist_cntr_small']], f[F['dist_cntr_large']] = m['dist_cntr_j'], m['dist_cntr_i']
                for k in range(self.npcaang):
                    f[F['pca_angle_small' + str(k)]] = mj['angles'][k]
                    f[F['pca_angle_large' + str(k)]] = mi['angles'][k]

            for k in range(self.npcaang):
                f[F['pca_angle' + str(k)]] = angles_ij[k]
            for k in range(self.nprob_types):
                f[F['mean_prob_' + self.prob_types[k]]] = m['mean_probs'][k]
            for k in range(self.naugments):
                f[F['mean_grayscale' + self.augments[k]]] = m['mean_grayscale_aug'][k]
                for x in range(self.nprob_types):
                    f[F['mean_prob_' + self.prob_types[x] + self.augments[k]]] = m['mean_probs_aug'][k][x]
            for k in range(self.nstatic_augments):
                if self.static_augments[k][0] == '_':
                    f[F['mean_grayscale' + self.static_augments[k]]] = m['mean_grayscale_static_aug'][k]
                else:
                    f[F['mean_prob_' + self.static_augments[k]]] = m['mean_probs_static_aug'][k]

            self.FRAG[i][j]['features'] = f

        if self.dpFRAG_verbose:
            print('\t\tdone in %.4f s' % (time.time() - t))
            print('\tdone in %.4f s' % (time.time() - ttime))

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
                gt_bnd = nd.measurements.find_objects(self.gt, self.ngtlbl)

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
        data = {'feature_names':self.features, 'DESCR':descr, 'target_names':self.TARGETS, 'target':target,
            'data':fdata}

        if self.dpFRAG_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))

        return data

    # single target agglomerate, for debug or creating "perfect" agglomeration from supervoxels based on GT.
    # this is like a loopback test for this script only (create FRAG using GT then agglomerate based on GT overlap).
    # also used by iterative supervoxel agglomeration method.
    # this method updates the FRAG (but not the FRAG edge features) based on the target agglomeration.
    def agglomerate(self, target, doWrite=True):
        ntargets = target.size
        assert( ntargets == self.FRAG.number_of_edges() ); # agglomerate input data must match FRAG edges

        if self.dpFRAG_verbose:
            print('Doing single agglomeration from supplied targets'); t = time.time()

        # single agglomerate with binary targets
        #self._incremental_agglomerate(target)

        # create agglomerated graph that only contains yes merge edges
        aggloG = nx.Graph(); aggloG.add_nodes_from(range(1,self.nsupervox_merge+1))
        for e,i in zip(self.FRAG.edges_iter(),range(ntargets)):
            if target[i] and not aggloG.has_edge(e[0],e[1]): aggloG.add_edge(e[0],e[1])

        # get connected component nodes, create supervoxel mapping and update FRAG based on agglomerated components
        supervox_map = np.zeros((self.nsupervox+1,),dtype=self.lbl_dtype)
        compsG = nx.connected_components(aggloG); ncomps = 0
        svox_sizes = np.zeros((self.nsupervox,),dtype=np.int64)

        # this is used for re-creating the graph after agglo without creating a new graph.
        # appends nodes that are beyond current last node. typcially this was nsupervox, but added the "do not merge"
        #   mode for which these supervoxels are not added to the graph at all.
        # this property indicates the last max node of the graph (also used in optimization).
        self.prev_max_node = self.nsupervox_merge

        for nodes in compsG:
            # SINGLETON NODE: single supervoxel that is not undergoing any merging (agglomeration)

            # supervoxel mapping from previous to new agglomerated supervoxels, update supervoxel sizes
            npnodes = np.array(tuple(nodes),dtype=np.int64); svox_sizes[ncomps] = self.svox_sizes[npnodes-1].sum()
            ncomps += 1; supervox_map[npnodes] = ncomps

            # update the FRAG by creating a new agglomerated node and moving neighboring edges to this node.
            # make a new singleton node for if this node is not containing any agglomerations.
            newnode = ncomps+self.prev_max_node; self.FRAG.add_node(newnode)

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
                #if len(self.FRAG[node]) > 0: self.FRAG.node[newnode]['svox_attrs'] = self.FRAG.node[node]['svox_attrs']

                self._noagglo_iterate_neighbors(self.FRAG, node, newnode, copy_attrs=True)

            else:
                # MUTIPLE COMPONENT NODE: group of supervoxels being merged (agglomerated)

                # copy over overlap attributes for neighbors that border on only one of the nodes in the agglomerate.
                # in this case the overlap area has not changed. need two passes to check for this.
                neighbors = {}

                # each node in connected component (that compose the agglomerated newnode)
                for node in nodes:
                    self._agglo_iterate_neighbors(self.FRAG, node, nodes, neighbors, newnode)

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
        #assert( self.prev_max_node+ncomps == max(self.FRAG.nodes()) )  # commented for speed, HIASSERT
        # relabel the FRAG starting at supervoxel 1
        self.FRAG = nx.relabel_nodes(self.FRAG, {x+self.prev_max_node:x for x in range(1,ncomps+1)}, copy=False)

        # xxx - this block is identical in threshold_agglomerate()
        if self.nsupervox_nomerge > 0:
            assert( ncomps < self.nsupervox_merge ) # sanity check
            # add back any "do not merge" supervoxel labels
            supervox_map[self.nsupervox_merge+1:] = \
                np.arange(ncomps+1,ncomps+self.nsupervox_nomerge+1,dtype=self.lbl_dtype)
            #print(self.svox_sizes.shape, self.nsupervox_merge, self.nsupervox_nomerge, ncomps)
            svox_sizes[ncomps:ncomps+self.nsupervox_nomerge] = self.svox_sizes[self.nsupervox_merge:]

        # create the new supervoxels from the supervoxel_map containing mapping from old nodes to agglo nodes.
        self.supervoxels = supervox_map[self.supervoxels]
        p = self.eperim; self.supervoxels_noperim = self.supervoxels[p[0]:-p[0],p[1]:-p[1],p[2]:-p[2]]
        self.nsupervox_merge = ncomps; ncomps += self.nsupervox_nomerge
        self.nsupervox = ncomps; self.svox_sizes = svox_sizes[:ncomps]
        self.data_cube = self.supervoxels_noperim
        self.data_attrs['types_nlabels'] = [self.nsupervox]
        self.supervoxels_zeroperim = self.supervoxels

        # write out the agglomerated supervoxels (without perimeter context)
        verbose = self.dpWriteh5_verbose; self.dpWriteh5_verbose = self.dpFRAG_verbose;
        if doWrite:
            self.writeCube(); self.dpWriteh5_verbose = verbose
        self.data_cube = np.zeros((0,))


        if self.dpFRAG_verbose:
            print('\tnsupervox',self.data_attrs['types_nlabels'][0])
            print('\tdone in %.4f s' % (time.time() - t, ))

    # "macro" for iterating neighbors for singleton nodes (not being agglomerated) in FRAG
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
                #   or if the other node was not visited yet (neighbor node <= previous max node (nsupervox_merge)).
                if neighbor <= self.prev_max_node or 'features' in f:
                    features_copied = True
                    cRAG[neighbor][newnode]['features'] = f['features']
                else:
                    features_copied = False

                # copy over overlap attributes but only if features are not copied over to final new edge.
                if neighbor <= self.prev_max_node or not features_copied:
                    # if the overlap attributes are present, then copy them over to the new edge.
                    if 'ovlp_attrs' in f:
                        cRAG[neighbor][newnode]['ovlp_attrs'] = f['ovlp_attrs']
                    # copy the supervoxel overlap attributes over for this singleton node
                    if 'sovlp_attrs' in f and node in f['sovlp_attrs']:
                        cRAG[neighbor][newnode]['sovlp_attrs'] = {newnode: f['sovlp_attrs'][node]}
                        #assert( 'ovlp_attrs' in f )     # should not have sovlp_attrs without ovlp_attrs, HIASSERT

        # remove the old singleton node
        cRAG.remove_node(node)

    # "macro" for iterating neighbors for compound nodes (those being agglomerated) in FRAG
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

        verbose = self.dpWriteh5_verbose; self.dpWriteh5_verbose = False;

        if self.dpFRAG_verbose:
            print('Threshold agglomeration for thresholds %s' % (' '.join([str(x) for x in thresholds]),))
            t = time.time()

        # create empty agglomerated graph
        aggloG = nx.Graph(); aggloG.add_nodes_from(range(1,self.nsupervox_merge+1))

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
            supervox_map = np.zeros((self.nsupervox+1,),dtype=self.lbl_dtype)
            compsG = nx.connected_components(aggloG); ncomps = 0
            for nodes in compsG:
                # supervoxel mapping from previous to new agglomerated supervoxels
                ncomps += 1; supervox_map[np.array(tuple(nodes),dtype=np.int64)] = ncomps
            # xxx - this block is identical in agglomerate()
            if self.nsupervox_nomerge > 0:
                assert( ncomps < self.nsupervox_merge ) # sanity check
                # add back any "do not merge" supervoxel labels
                supervox_map[self.nsupervox_merge+1:] = \
                    np.arange(ncomps+1,ncomps+self.nsupervox_nomerge+1,dtype=self.lbl_dtype)
                ncomps += self.nsupervox_nomerge
            # create the new supervoxels from the supervoxel_map containing mapping from agglo nodes to new nodes
            self.data_cube = supervox_map[self.supervoxels_noperim]

            if self.dpFRAG_verbose:
                print('\tnsupervox',ncomps)
            self.data_attrs['types_nlabels'] = [ncomps]
            self.writeCube()

        if self.dpFRAG_verbose:
            print('\n\tdone in %.4f s' % (time.time() - t))

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
        C = np.mean(pts, axis=0) # centroid

        # xxx - this is hacky, but prevent pca from being called on svox areas in bounding box when
        #   the pca angles are not being calculated at all for the feature set.
        # xxx - maybe clean this up by just a single input bool that is "for overlap" or "for svox"
        if Vother is None or self.npcaang > 0:
            Cpts = pts-C # center points around centroid
            V = dpFRAG.getOrthoAxes(Cpts, sampling) # get the principal axes.
        else:
            V = None; Cpts = None

        # only return points if needed
        if not return_Cpts: Cpts = None

        # calculate angles between corresponding eigenvectors
        #if Vother is None:
        if Vother is None or self.npcaang == 0:
            angles = None
        else:
            angles = np.zeros((self.npcaang,),np.double)
            for k in range(self.npcaang):
                angles[k] = np.arctan2(nla.norm(np.cross(V[k,:],Vother[k,:])), np.dot(V[k,:],Vother[k,:]))

            # rigid transform rotation decomposed to Euler angles.
            # xxx - this method did not work as well for classification. look at this again in the future?
            #   later found a bug in the rigid body function, maybe try this again?
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
            V[:,-1] = -V[:,-1]; R = np.dot(V,U.T)
        t = -np.dot(R, cA.T) + cB.T
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
    def makeTrainingFRAG(cls, labelfile, chunk, size, offset, probfiles, rawfiles, raw_dataset, gtfile,
            subgroups=[], G=None, progressBar=False, feature_set=None, has_ECS=True, chunk_subgroups=False,
            neighbor_only=False, pad_prob_svox_perim=False, no_agglo_ECS=False, verbose=False):
        parser = argparse.ArgumentParser(description='class:dpFRAG',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        dpFRAG.addArgs(parser); arg_str = ''

        arg_str += ' --srcfile ' + labelfile
        arg_str += ' --chunk %d %d %d ' % tuple(chunk)
        arg_str += ' --offset %d %d %d ' % tuple(offset)
        arg_str += ' --size %d %d %d ' % tuple(size)
        if subgroups: arg_str += ' --subgroups %s ' % ' '.join(subgroups)
        if probfiles[0]: arg_str += ' --probfile ' + probfiles[0]
        if probfiles[1]: arg_str += ' --probaugfile ' + probfiles[1]
        if rawfiles[0]: arg_str += ' --rawfile ' + rawfiles[0]
        if rawfiles[1]: arg_str += ' --rawaugfile ' + rawfiles[1]
        arg_str += ' --raw-dataset ' + raw_dataset
        arg_str += ' --gtfile ' + gtfile
        if feature_set: arg_str += ' --feature-set ' + feature_set
        if not has_ECS: arg_str += ' --no-ECS '
        if chunk_subgroups: arg_str += ' --chunk-subgroups '
        if neighbor_only: arg_str += ' --neighbor-only '
        if pad_prob_svox_perim: arg_str += ' --pad-prob-perim '
        if no_agglo_ECS: arg_str += ' --no-agglo-ECS '

        if verbose: arg_str += ' --dpFRAG-verbose '
        if verbose: print(arg_str)
        args = parser.parse_args(arg_str.split())
        frag = cls(args);
        frag.loadData()
        frag.FRAG = G
        return frag

    @classmethod
    def makeTestingFRAG(cls, labelfile, chunk, size, offset, probfiles, rawfiles, raw_dataset, outfile=None,
            subgroups=[], subgroups_out=[], G=None, progressBar=False, feature_set=None, has_ECS=True,
            chunk_subgroups=False, neighbor_only=False, pad_prob_svox_perim=False, no_agglo_ECS=False, verbose=False):
        parser = argparse.ArgumentParser(description='class:dpFRAG',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        dpFRAG.addArgs(parser); arg_str = ''

        arg_str += ' --srcfile ' + labelfile
        arg_str += ' --chunk %d %d %d ' % tuple(chunk)
        arg_str += ' --offset %d %d %d ' % tuple(offset)
        arg_str += ' --size %d %d %d ' % tuple(size)
        if probfiles[0]: arg_str += ' --probfile ' + probfiles[0]
        if probfiles[1]: arg_str += ' --probaugfile ' + probfiles[1]
        if rawfiles[0]: arg_str += ' --rawfile ' + rawfiles[0]
        if rawfiles[1]: arg_str += ' --rawaugfile ' + rawfiles[1]
        arg_str += ' --raw-dataset ' + raw_dataset
        if outfile: arg_str += ' --outfile ' + outfile
        if subgroups: arg_str += ' --subgroups %s ' % ' '.join(subgroups)
        if subgroups_out: arg_str += ' --subgroups-out %s ' % ' '.join(subgroups_out)
        if feature_set: arg_str += ' --feature-set ' + feature_set
        if not has_ECS: arg_str += ' --no-ECS '
        if chunk_subgroups: arg_str += ' --chunk-subgroups '
        if neighbor_only: arg_str += ' --neighbor-only '
        if pad_prob_svox_perim: arg_str += ' --pad-prob-perim '
        if no_agglo_ECS: arg_str += ' --no-agglo-ECS '

        if verbose: arg_str += ' --dpFRAG-verbose '
        if verbose: print(arg_str)
        args = parser.parse_args(arg_str.split())
        frag = cls(args);
        frag.loadData()
        frag.FRAG = G
        return frag

    @classmethod
    def makeBothFRAG(cls, labelfile, chunk, size, offset, probfiles, rawfiles, raw_dataset, gtfile, outfile=None,
            subgroups=[], subgroups_out=None, G=None, progressBar=False, feature_set=None, has_ECS=True,
            neighbor_only=False, chunk_subgroups=False, pad_prob_svox_perim=False, no_agglo_ECS=False, verbose=False):
        parser = argparse.ArgumentParser(description='class:dpFRAG',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        dpFRAG.addArgs(parser); arg_str = ''

        arg_str += ' --srcfile ' + labelfile
        arg_str += ' --chunk %d %d %d ' % tuple(chunk)
        arg_str += ' --offset %d %d %d ' % tuple(offset)
        arg_str += ' --size %d %d %d ' % tuple(size)
        if probfiles[0]: arg_str += ' --probfile ' + probfiles[0]
        if probfiles[1]: arg_str += ' --probaugfile ' + probfiles[1]
        if rawfiles[0]: arg_str += ' --rawfile ' + rawfiles[0]
        if rawfiles[1]: arg_str += ' --rawaugfile ' + rawfiles[1]
        arg_str += ' --raw-dataset ' + raw_dataset
        arg_str += ' --gtfile ' + gtfile
        if outfile: arg_str += ' --outfile ' + outfile
        if subgroups: arg_str += ' --subgroups %s ' % ' '.join(subgroups)
        if subgroups_out: arg_str += ' --subgroups-out %s ' % ' '.join(subgroups_out)
        if feature_set: arg_str += ' --feature-set ' + feature_set
        if not has_ECS: arg_str += ' --no-ECS '
        if chunk_subgroups: arg_str += ' --chunk-subgroups '
        if neighbor_only: arg_str += ' --neighbor-only '
        if pad_prob_svox_perim: arg_str += ' --pad-prob-perim '
        if no_agglo_ECS: arg_str += ' --no-agglo-ECS '

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
        #p.add_argument('--perim', nargs=3, type=int, default=[16,16,8], metavar=('X', 'Y', 'Z'), #choices=range(1,20),
        #    help='Size of bounding box around overlap for object features')
        p.add_argument('--sperim', nargs=1, type=int, default=16, metavar=('S'),
            help='Size of one side of bounding box around overlap for object features (3D calculated using sampling)')
        p.add_argument('--remove-ECS', dest='remove_ECS', action='store_true',
            help='Set to remove ECS supervoxels (set to 0)')
        p.add_argument('--gt-ECS-label', nargs=1, type=int, default=[1],
            help='Which label is ECS in GT for remove-ECS (-1 is last label, 0 is none)')
        p.add_argument('--mapout', nargs=1, type=str, default='', help='Optional text dump of supervox to GT mapping')
        #p.add_argument('--ovlp-dilate', nargs=1, type=int, default=[1], choices=range(0,20),
        #    help='Amount to dilate overlap for calculating boundary features')
        #p.add_argument('--connectivity', nargs=1, type=int, default=[3], choices=[1,2,3],
        #    help='Connectivity for binary morphology operations')
        p.add_argument('--neighbor-only', dest='neighbor_only', action='store_true',
            help='Only use boundary voxels labeled with neighboring supervoxels (no background / non-neighbor voxels)')
        p.add_argument('--chunk-subgroups', action='store_true',
            help='This mode is for probs and labels that have overlapping context so need to be stored separately.')
        p.add_argument('--keep-subgroups', action='store_true',
            help='Keep subgroups for labels in path for subgroups-out')
        #p.add_argument('--progress-bar', action='store_true', help='Enable progress bar if available')
        p.add_argument('--feature-set', nargs=1, type=str, default='standard',
            choices=['all','standard','medium','reduced','small','minimal'],
            help='Option to control which features are calculated')
        p.add_argument('--no-ECS', dest='has_ECS', action='store_false',
                       help='Specify when data does not contain ECS classified voxels')
        # default is to load raw, probs and supervoxels in perimeter with the idea of decreasing boundary effects.
        p.add_argument('--pad-raw-perim', action='store_true', help='Pad perimeter of raw EM instead of loading')
        # using pad prob perim assumes that fill value for probs is 0 (not -1 or other)
        p.add_argument('--pad-prob-perim', action='store_true', help='Pad perimeter of probs instead of loading')
        #p.add_argument('--pad-svox-perim', action='store_true', help='Pad perimeter of supervoxels instead of loading')
        p.add_argument('--no-agglo-ECS', action='store_true', help='Do not agglomerate ECS supervoxels')
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
