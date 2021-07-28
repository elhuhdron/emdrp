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

# Top level for EM supervoxel classifier that learns merge / no merge classification based on some feature set.
# Featured Region Adjacency Graph (dpFRAG.py) creates the graph of neighboring supervoxels and their features.
#
# There is no iteration in this procedure, typically referred to as "flat learning" in the context of merging
#   supervoxels.
# After investigating flat learning, added a simple iterative procedure that agglomerates some percentage of the most
#   confident mergers and continues for set number of iterations.


import numpy as np
import time
import argparse
import os
#import sys
import importlib
from io import BytesIO

from configobj import ConfigObj, flatten_errors
from validate import Validator, ValidateError
import dill
from io import StringIO
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
from scipy import ndimage as nd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# xxx - how to make imports as optional? make plotting class?
from matplotlib import pylab as pl
from matplotlib import pyplot as plt
#import matplotlib as mpl
#from matplotlib import colors
#from cycler import cycler

from dpLoadh5 import dpLoadh5
# made dpFRAG import dynamic in init
#from dpFRAG import dpFRAG
#from dpFRAGc import dpFRAG
dpFRAG = []
from metrics import pixel_error_fscore
from utils import print_cpu_info_linux

class dpSupervoxelClassifier():

    # Constants
    LIST_ARGS = ['test_chunks', 'label_subgroups', 'label_subgroups_out', 'iterate_merge_perc',
        'thresholds', 'threshold_subgroups']

    def __init__(self, args):

        # save command line arguments from argparse, see definitions in main or run with --help
        for k, v in vars(args).items():
            # do not override any values that are already set as a method of allowing inherited classes to specify
            if hasattr(self,k): continue
            if type(v) is list and k not in self.LIST_ARGS:
                if len(v)==1:
                    setattr(self,k,v[0])  # save single element lists as first element
                elif type(v[0]) is int:   # convert the sizes and offsets to numpy arrays
                    setattr(self,k,np.array(v,dtype=np.int32))
                else:
                    setattr(self,k,v)   # store other list types as usual (floats)
            else:
                setattr(self,k,v)

        # save the command line argument dict as a string
        out = StringIO(); print( vars(args), file=out )
        self.arg_str = out.getvalue(); out.close()

        print('dpSupervoxelClassifier: config file ''%s''' % self.cfgfile)
        # retrieve / save options from ini files, see definitions dpSupervoxelClassifier.ini
        opts = dpSupervoxelClassifier.get_options(self.cfgfile)
        d = vars(self)
        for k, v in opts.items():
            # do not import if have a "True" (non-empty) value from command line
            if k in d and d[k]: continue
            if type(v) is list and k not in self.LIST_ARGS:
                if len(v)==1:
                    setattr(self,k,v[0])  # save single element lists as first element
                elif len(v)>0 and type(v[0]) is int:   # convert the sizes and offsets to numpy arrays
                    setattr(self,k,np.array(v,dtype=np.int32))
                else:
                    setattr(self,k,v)   # store other list types as usual (floats, empties)
            else:
                setattr(self,k,v)

        # save the command line argument dict as a string
        out = StringIO(); print( opts, file=out )
        self.ini_str = out.getvalue(); out.close()

        # Options / Inits

        # dynamic imports
        dpFRAGl = importlib.import_module('dpFRAGc') if self.useFRAGc else importlib.import_module('dpFRAG')
        globals().update({'dpFRAG':dpFRAGl.dpFRAG})

        # these are so standard cubeIter inputs can be used from command line to override from .ini
        if (self.chunk >= 0).all():
            self.chunk_range_beg = self.chunk
            self.chunk_range_end = []
            self.offset_list = self.offset
            self.size_list = self.size

        self.doplots = (self.show_plots or self.export_plots)

        # default is to have sklearn calculate the priors
        self.priors = None
        if self.merge_prior > 0: self.priors = np.array([1-self.merge_prior, self.merge_prior],dtype=np.double)

        # keep the list of merge percentages, use original to store current iteration value
        self.iterate_merge_perc_list = self.iterate_merge_perc

        # initialize for "chunkrange" or "chunklist" mode if these parameters are not empty.
        # this code was modified from that in the em data parser for cuda-convnets2.
        self.chunk_range_beg = self.chunk_range_beg.reshape(-1,3); self.use_chunk_range = False
        self.nchunk_list = self.chunk_range_beg.shape[0]
        self.nchunks = self.nchunk_list

        # offsets are either per chunk or per range, depending on above mode (whether chunk_range_end empty or not)
        self.offset_list = self.offset_list.reshape(-1,3)   # list must have multiple of 3 elements for volumes
        if self.offset_list.shape[0] > 1:
            assert( self.offset_list.shape[0] == self.nchunk_list )
        else:
            self.offset_list = np.ones_like(self.chunk_range_beg)*self.offset_list

        # sizes are either per chunk or per range, depending on above mode (whether chunk_range_end empty or not).
        self.size_list = self.size_list.reshape(-1,3)   # list must have multiple of 3 elements for volumes
        if self.size_list.shape[0] > 1:
            assert( self.size_list.shape[0] == self.nchunk_list )
        else:
            self.size_list = np.ones_like(self.chunk_range_beg)*self.size_list

        if len(self.chunk_range_end) > 0:
            # "chunkrange" mode, chunks are selected based on defined beginning and end of ranges in X,Y,Z
            # range is open ended (python-style, end is not included in range)
            self.chunk_range_end = self.chunk_range_end.reshape(-1,3);
            assert( self.chunk_range_end.shape[0] == self.nchunk_list )
            self.chunk_range_rng = self.chunk_range_end - self.chunk_range_beg
            assert( (self.chunk_range_rng >= 0).all() )     # some bad ranges

            # this allows for iterating in chunk_range mode with sizes greater than chunksize
            # just read the h5info for the rawfile to get the chunksize
            rawh5 = dpLoadh5.readInith5(self.rawfile, self.raw_dataset, [0,0,0], [0,0,0], [128,128,128], 'uint8')
            self.chunksize = rawh5.chunksize
            self.chunkscale = np.ones_like(self.size_list)
            # only allow this if the sizes are a multiple of chunksize
            if (self.size_list % self.chunksize == 0).all():
                scale = self.size_list // self.chunksize
                # only allow this if the ranges are a multiple of the sizes
                if (self.chunk_range_rng % scale == 0).all():
                    self.chunkscale = scale
            self.chunk_range_rng //= self.chunkscale

            self.chunk_range_size = self.chunk_range_rng.prod(axis=1)
            self.chunk_range_cumsize = np.concatenate((np.zeros((1,),dtype=self.chunk_range_size.dtype),
                self.chunk_range_size.cumsum()))
            self.chunk_range_nchunks = self.chunk_range_cumsize[-1]
            self.use_chunk_range = True; self.nchunks = self.chunk_range_nchunks

        # print out all initialized variables in verbose mode
        if self.dpSupervoxelClassifier_verbose:
            # print out info for chunklist / chunkrange modes so that input data is logged
            print(('Using %d ' % self.nchunk_list) + ('ranges' if self.use_chunk_range else 'chunks') + ':')
            fh = BytesIO()
            if self.use_chunk_range:
                np.savetxt(fh, np.concatenate((np.arange(self.nchunk_list).reshape((self.nchunk_list,1)),
                    self.chunk_range_beg, self.chunk_range_end, self.chunk_range_size.reshape((self.nchunk_list,1)),
                    self.size_list, self.offset_list), axis=1),
                    fmt='\t(%d) range %d %d %d to %d %d %d (%d chunks), size %d %d %d, offset %d %d %d',
                    delimiter='', newline='\n', header='', footer='', comments='')
            else:
                np.savetxt(fh, np.concatenate((np.arange(self.nchunk_list).reshape((self.nchunk_list,1)),
                    self.chunk_range_beg, self.size_list, self.offset_list), axis=1),
                    fmt='\t(%d) chunk %d %d %d, size %d %d %d, offset %d %d %d',
                    delimiter='', newline='\n', header='', footer='', comments='')
            cstr = fh.getvalue(); fh.close(); print(cstr.decode('UTF-8'))
            print('Test chunks: %s' % ' '.join([str(x) for x in self.test_chunks]))

            #print('dpSupervoxelClassifier, verbose mode:\n'); print(vars(self))
            print_cpu_info_linux() # for debugging runtime variance on biowulf

        # inits for iterative prior mode
        self.iterative_mode = (self.iterate_count > 0)
        self.iterative_frag = [None] * self.nchunks
        self.iterative_mode_count = 0
        if self.outfile:
            if len(self.iterate_save_ranges) > 0:
                assert(len(self.iterate_save_ranges) % 3 == 0) # must specify as python-style ranges
                n = len(self.iterate_save_ranges)//3
                self.iterate_save_mask = np.zeros((self.iterate_count,), dtype=bool)
                for i in range(n):
                    start, stop, step = self.iterate_save_ranges[3*i:3*i+3]
                    self.iterate_save_mask[range(start,stop,step)] = 1
                self.iterate_save_mask[stop:] = 1
            else:
                self.iterate_save_mask = np.ones((self.iterate_count,), dtype=bool)
        else:
            self.iterate_save_mask = np.zeros((self.iterate_count,), dtype=bool)

        if self.iterative_mode:
            # expand the list out to the number of iterations by repeating the last merge percentage
            tmp = np.zeros((self.iterate_count,), dtype=np.double)
            tmp[:len(self.iterate_merge_perc_list)] = self.iterate_merge_perc_list
            tmp[len(self.iterate_merge_perc_list):] = self.iterate_merge_perc_list[-1]
            self.iterate_merge_perc_list = tmp

        # xxx - clean this up at some point, basically hack for legacy metric comparison scripts
        if len(self.threshold_subgroups) == 0:
            if self.iterative_mode:
                self.threshold_subgroups = np.arange(1,self.iterate_count+1,dtype=np.double)
            else:
                self.threshold_subgroups = np.sort(self.thresholds)[::-1]
        else:
            self.threshold_subgroups = np.sort(self.threshold_subgroups)[::-1]
            if self.iterative_mode:
                assert( self.iterate_count == len(self.threshold_subgroups) )
            else:
                assert( len(self.thresholds) == len(self.threshold_subgroups) )

        # new mode where saving classifiers separately for iterative mode
        if self.classifierout and self.iterative_mode:
            [self.classifierout_path, tmp] = os.path.split(self.classifierout)
            [self.classifierout_name, self.classifierout_ext] = os.path.splitext(tmp)
            # make a subdirectory of the same name without ext and put all the iterative classifier dills into it
            self.classifierout_path = os.path.join(self.classifierout_path,self.classifierout_name)
            os.makedirs(self.classifierout_path, exist_ok=True)

        # for loading new mode where each classifier stored in separate dill file.
        # if classifierin is not a directory then use old mode where all classifiers stored in a single file.
        if self.classifierin and self.iterative_mode and not os.path.isfile(self.classifierin):
            [tmp, self.classifierin_name] = os.path.split(self.classifierin)

        # dpFRAG features are now selectable as sets
        #self.FEATURES, self.FEATURES_NAMES, self.nfeatures = dpFRAG.make_FEATURES()
        d = dpFRAG.make_features(self.feature_set, self.has_ECS)
        for k in ['features','features_names','nfeatures']: setattr(self,k,d[k])

        # xxx - make this more systematic for other libraries?
        #   typically using anaconda, for which this should work since built on mkl
        try:
            import mkl
            print('Setting mkl num_threads to %d' % (self.nthreads,))
            mkl.set_num_threads(self.nthreads)
        except ImportError:
            pass

    def train(self):

        if self.dpSupervoxelClassifier_verbose: print('\nTRAIN')

        if self.trainin and (not self.classifierin or self.doplots):
            if self.dpSupervoxelClassifier_verbose:
                print('Loading training data')
            with open(self.trainin, 'rb') as f: data = dill.load(f)
            target = data['target']; fdata = data['data']
            ntargets = target.size; nfeatures = fdata.shape[1]
            assert( nfeatures == self.nfeatures )

        elif not self.classifierin:
            #dict_keys(['feature_names', 'DESCR', 'target_names', 'target', 'data'])
            nalloc = self.nchunks*self.nalloc_per_chunk
            nfeatures = self.nfeatures
            target = np.zeros((nalloc,), dtype=np.int64)
            fdata = np.zeros((nalloc,nfeatures), dtype=np.double)

            # accumulate training data from all training chunks
            cnt_targets = 0; ntargets = np.zeros((self.nchunks,),dtype=np.int64)
            for chunk in range(self.nchunks):
                cchunk, chunk_list_index, chunk_range_index = self.get_chunk_inds(chunk)
                offset = self.offset_list[chunk_list_index,:]; size = self.size_list[chunk_list_index,:]

                if chunk_list_index in self.test_chunks: continue
                print('Appending training data for chunk %d,%d,%d' % tuple(cchunk.tolist()))

                if self.iterative_mode:
                    if self.iterative_frag[chunk] is None:
                        frag = dpFRAG.makeBothFRAG(self.labelfile, cchunk, size, offset,
                            [self.probfile, self.probaugfile], [self.rawfile, self.rawaugfile],
                            self.raw_dataset, self.gtfile, self.outfile, self.label_subgroups,
                            ['agglomeration_training'], progressBar=self.progress_bar, feature_set=self.feature_set,
                            has_ECS=self.has_ECS, chunk_subgroups=self.chunk_subgroups, no_agglo_ECS=self.no_agglo_ECS,
                            pad_prob_svox_perim = not self.prob_svox_context, neighbor_only=self.neighbor_only,
                            verbose=self.dpSupervoxelClassifier_verbose)
                        frag.isTraining = True; self.iterative_frag[chunk] = frag
                    else:
                        frag = self.iterative_frag[chunk]
                else:
                    frag = dpFRAG.makeTrainingFRAG(self.labelfile, cchunk, size, offset,
                        [self.probfile, self.probaugfile], [self.rawfile, self.rawaugfile],
                        self.raw_dataset, self.gtfile, self.label_subgroups, feature_set=self.feature_set,
                        has_ECS=self.has_ECS, chunk_subgroups=self.chunk_subgroups, neighbor_only=self.neighbor_only,
                        pad_prob_svox_perim = not self.prob_svox_context, progressBar=self.progress_bar,
                        no_agglo_ECS=self.no_agglo_ECS, verbose=self.dpSupervoxelClassifier_verbose)
                frag.createFRAG(update = self.iterative_mode)
                #frag.createFRAG(update = False)
                data = frag.createDataset()
                ntargets[chunk] = data['target'].shape[0]
                target[cnt_targets:cnt_targets+ntargets[chunk]] = data['target']
                fdata[cnt_targets:cnt_targets+ntargets[chunk],:] = data['data']
                cnt_targets += ntargets[chunk]
            target = target[:cnt_targets]; fdata = fdata[:cnt_targets,:]

            if self.trainout:
                #dict_keys(['feature_names', 'DESCR', 'target_names', 'target', 'data'])
                descr = 'Training data from dpFRAG.py with command line:\n' + self.arg_str
                descr = ('With ini file "%s":\n' % (self.cfgfile,)) + self.ini_str
                data['data'] = fdata; data['target'] = target; data['DESCR'] = descr
                with open(self.trainout, 'wb') as f: dill.dump(data, f)

        if not self.classifierin or self.doplots:
            # everyone wants to be norml
            sdata = scale(fdata)   # normalize for the classifiers
            # shuffle data for training as well
            target, sdata = shuffle(target, sdata)

        if self.classifierin:
            if self.dpSupervoxelClassifier_verbose:
                print('\nLoading classifier:'); t = time.time()

            with open(self.classifierin, 'rb') as f: d = dill.load(f)
            self.clf = d['classifier'];
        else:
            if self.dpSupervoxelClassifier_verbose:
                print('\nTraining classifier %s with %d examples and %d features:' % (self.classifier,
                    cnt_targets, nfeatures)); t = time.time()

            # train a classifier
            if self.classifier == 'lda':
                self.clf = LinearDiscriminantAnalysis(solver='svd', store_covariance=False, priors=self.priors)
                #self.clf = LinearDiscriminantAnalysis(solver='eigen', store_covariance=True, priors=self.priors)
            elif self.classifier == 'qda':
                self.clf = QuadraticDiscriminantAnalysis(priors=self.priors)
            elif self.classifier == 'rf':
                # the gala parameters
                #self.clf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=20,
                #    bootstrap=False, random_state=None)
                #self.clf = RandomForestClassifier(n_estimators=5*nfeatures,n_jobs=self.nthreads,max_depth=10)
                self.clf = RandomForestClassifier(n_estimators=256,n_jobs=self.nthreads,max_depth=16)
            elif self.classifier == 'svm':
                self.clf = SVC(kernel='rbf',probability=True,cache_size=2000)
            elif self.classifier == 'nb':
                self.clf = GaussianNB()
            elif self.classifier == 'kn':
                self.clf = KNeighborsClassifier(n_neighbors=10,n_jobs=self.nthreads)
            elif self.classifier == 'dc':
                self.clf = DecisionTreeClassifier(max_depth=10)
            elif self.classifier == 'ada':
                self.clf = AdaBoostClassifier()
            elif self.classifier == 'lr':
                self.clf = LogisticRegression(penalty='l2',dual=False,solver='sag',n_jobs=self.nthreads)
            else:
                assert(False)   # i never try anything, i just do it

            # train to the normalized data and merge or no merge targets
            self.clf.fit(sdata, target)

            if self.classifierout and not self.iterative_mode:
                with open(self.classifierout, 'wb') as f: dill.dump({'classifier':self.clf}, f)

        if self.dpSupervoxelClassifier_verbose:
            print('\tdone in %.4f s' % (time.time() - t))

        # do the agglomeration to use as input to next iteration for iterative mode
        thr = -1
        if self.iterative_mode:
            cnt_targets = 0
            for chunk in range(self.nchunks):
                if self.iterative_frag[chunk] is None or not self.iterative_frag[chunk].isTraining: continue

                # get the feature data for current training chunk only
                cdata = sdata[cnt_targets:cnt_targets+ntargets[chunk],:]
                cnt_targets += ntargets[chunk]

                # merge based on current classifier
                self.iterative_frag[chunk].subgroups_out[-1] = '%.8f' \
                    % self.threshold_subgroups[self.iterative_mode_count]
                clf_predict, thr = self.get_merge_predict_thr(cdata)
                self.iterative_frag[chunk].agglomerate(clf_predict,
                    doWrite=self.iterate_save_mask[self.iterative_mode_count])

                # make the next training iteration load from the current agglomerated supervoxels
                self.iterative_frag[chunk].srcfile = self.iterative_frag[chunk].outfile
                self.iterative_frag[chunk].subgroups = self.iterative_frag[chunk].subgroups_out

        if self.doplots:
            return self.createPlots(target,sdata,self.clf,self.export_plots,
                name=self.classifier + '_train_' + '_'.join([str(x) for x in self.test_chunks]) + \
                '_iter_' + str(self.iterative_mode_count), thr=thr, plot_features=self.plot_features)

    def test(self):
        if self.dpSupervoxelClassifier_verbose: print('\nTEST')

        for chunk in range(self.nchunks):
            cchunk, chunk_list_index, chunk_range_index = self.get_chunk_inds(chunk)

            if chunk_list_index not in self.test_chunks: continue
            data,sdata,thr = self._test(chunk, cchunk, chunk_list_index, chunk_range_index)

        # xxx - this only plots the LAST test chunk (currently this is only used for leave-one-out cross-validations)
        if self.doplots:
            assert(len(self.test_chunks) == 1)  # exporting plots/test data only works for one test chunk
            return self.createPlots(data['target'],sdata,self.clf,self.export_plots,
                name=self.classifier + '_test_' + '_'.join([str(x) for x in self.test_chunks]) + \
                '_iter_' + str(self.iterative_mode_count), thr=thr, plot_features=self.plot_features)

    def _test(self, ichunk, cchunk, chunk_list_index, chunk_range_index):
        print('Exporting testing data for chunk %d,%d,%d' % tuple(cchunk.tolist()))
        offset = self.offset_list[chunk_list_index,:]; size = self.size_list[chunk_list_index,:]

        FRAG = None
        if len(self.test_chunks) == 1 and self.testin:
            if self.dpSupervoxelClassifier_verbose:
                print('Loading testing data')
            with open(self.testin, 'rb') as f: data = dill.load(f)
            FRAG = data['FRAG']; data = data['data']

        frag = self.iterative_frag[ichunk] if self.iterative_mode else None
        subgroups_out = list(self.label_subgroups_out)

        if frag is None:
            if self.doplots:
                frag = dpFRAG.makeBothFRAG(self.labelfile, cchunk, size, offset,
                    [self.probfile, self.probaugfile], [self.rawfile, self.rawaugfile],
                    self.raw_dataset, self.gtfile, self.outfile, self.label_subgroups, subgroups_out,
                    G=FRAG, progressBar=self.progress_bar, feature_set=self.feature_set, has_ECS=self.has_ECS,
                    chunk_subgroups=self.chunk_subgroups, neighbor_only=self.neighbor_only,
                    pad_prob_svox_perim = not self.prob_svox_context, no_agglo_ECS=self.no_agglo_ECS,
                    verbose=self.dpSupervoxelClassifier_verbose)
            else:
                frag = dpFRAG.makeTestingFRAG(self.labelfile, cchunk, size, offset,
                    [self.probfile, self.probaugfile], [self.rawfile, self.rawaugfile],
                    self.raw_dataset, self.outfile, self.label_subgroups, subgroups_out, G=FRAG,
                    progressBar=self.progress_bar, feature_set=self.feature_set, has_ECS=self.has_ECS,
                    chunk_subgroups=self.chunk_subgroups, neighbor_only=self.neighbor_only,
                    pad_prob_svox_perim = not self.prob_svox_context, no_agglo_ECS=self.no_agglo_ECS,
                    verbose=self.dpSupervoxelClassifier_verbose)

        if self.iterative_mode and self.iterative_frag[ichunk] is None:
            frag.isTraining = False; self.iterative_frag[ichunk] = frag

            # write out the starting labels as as subgroup 0
            frag.subgroups_out[-1] = '%.8f' % 0.0
            frag.data_cube = frag.supervoxels_noperim
            verbose = frag.dpWriteh5_verbose; frag.dpWriteh5_verbose = frag.dpFRAG_verbose;
            #self.data_attrs['types_nlabels'] = [ncomps]
            frag.writeCube(); frag.dpWriteh5_verbose = verbose

        if not (len(self.test_chunks) == 1 and self.testin):
            frag.createFRAG(update = self.iterative_mode)
            #frag.createFRAG(update = False)
            data = frag.createDataset(train=self.doplots)

            if self.testout:
                if self.dpSupervoxelClassifier_verbose:
                    print('Dumping testing data')
                descr = 'Testing data from dpFRAG.py with command line:\n' + self.arg_str
                descr = ('With ini file "%s":\n' % (self.cfgfile,)) + self.ini_str
                data['DESCR'] = descr
                with open(self.testout, 'wb') as f: dill.dump({'data':data,'FRAG':frag.FRAG}, f)

        sdata = scale(data['data'])     # normalize for the classifiers

        thr = -1
        if self.iterative_mode:
            # merge based on current classifier
            frag.subgroups_out[-1] = '%.8f' % self.threshold_subgroups[self.iterative_mode_count]
            clf_predict, thr = self.get_merge_predict_thr(sdata)
            frag.agglomerate(clf_predict, doWrite=self.iterate_save_mask[self.iterative_mode_count])

            # make the next training iteration load from the current agglomerated supervoxels
            self.iterative_frag[ichunk].srcfile = self.iterative_frag[ichunk].outfile
            self.iterative_frag[ichunk].subgroups = self.iterative_frag[ichunk].subgroups_out
        else:
            try:
                # predict merge or not on testing cube and write outputs at specified probability thresholds
                frag.threshold_agglomerate(self.clf.predict_proba(sdata), self.thresholds, self.threshold_subgroups)
                # there's an issue here if there are no mergers left, would be better to just copy the current
                #   agglomeration, xxx - deal with this later. handled this explicitly in other locations.
                #except AttributeError:
            except:
                # if the classifier doesn't do probabilities just export single prediction
                frag.subgroups_out[-1] = ('single_' + self.classifier)
                frag.agglomerate(self.clf.predict(sdata))

        return data,sdata,thr

    def iterative_classify(self):
        assert(self.iterative_mode)

        if self.classifierin and os.path.isfile(self.classifierin):
            # this is a legacy mode where all classifiers are stored in same dill
            if self.dpSupervoxelClassifier_verbose:
                print('\nLoading iterative classifiers:'); t = time.time()

            with open(self.classifierin, 'rb') as f: d = dill.load(f)
            self.clfs = d['classifiers'];
            print('\tdone in %.4f s' % (time.time() - t))
        else:
            self.clfs = [None] * self.iterate_count

        # save the ROC-style metrics for later analysis / plotting
        train_metrics = [None] * self.iterate_count; test_metrics = [None] * self.iterate_count

        # each iteration loop trains on the current supervoxels and
        #   then performs a merge (test) based with a normal sklearn predict based on a small merge prior.
        self.trainin = ''; self.trainout = ''; self.testin = ''; self.testout = ''
        if self.test_only:
            print('Test-only iterative mode with merge prior %.4f' % (self.merge_prior,))
            for chunk in range(self.nchunks):
                cchunk, chunk_list_index, chunk_range_index = self.get_chunk_inds(chunk)
                #if chunk_list_index not in self.test_chunks: continue

                for i in range(self.iterate_count):
                    self.iterate_merge_perc = self.iterate_merge_perc_list[i]; self.iterative_mode_count = i
                    print('Iteration %d, merge perc %.4f' % (self.iterative_mode_count+1,self.iterate_merge_perc))

                    if self.clfs[i] is None and self.classifierin and not os.path.isfile(self.classifierin):
                        with open(os.path.join(self.classifierin,self.classifierin_name+'_'+\
                            str(self.iterative_mode_count)+'.dill'), 'rb') as f: d = dill.load(f)
                        self.clf = d['classifier']
                    else:
                        self.clf = self.clfs[i]

                    assert( self.clf is not None) # classifierin must be specified for test only
                    self._test(chunk, cchunk, chunk_list_index, chunk_range_index)

                # let the previous chunk get garbage collected to save memory.
                # that is essentially the purpose of this special mode (iteration loop as inner loop).
                self.iterative_frag[chunk] = None
        else:
            print('Normal iterative mode with merge prior %.4f' % (self.merge_prior,))
            for i in range(self.iterate_count):
                self.iterate_merge_perc = self.iterate_merge_perc_list[i]; self.iterative_mode_count = i
                print('Iteration %d, merge perc %.4f' % (self.iterative_mode_count+1,self.iterate_merge_perc))

                if self.clfs[i] is None and self.classifierin and not os.path.isfile(self.classifierin):
                    with open(os.path.join(self.classifierin,self.classifierin_name+'_'+\
                        str(self.iterative_mode_count)+'.dill'), 'rb') as f: d = dill.load(f)
                    self.clf = d['classifier']
                else:
                    self.clf = self.clfs[i]

                if self.clf is None:
                    train_metrics[i] = self.train()
                test_metrics[i] = self.test()

                if self.classifierout:
                    # new mode that stores all classifiers in separate dill files
                    fn = os.path.join(self.classifierout_path,self.classifierout_name+'_'+\
                        str(self.iterative_mode_count)+self.classifierout_ext)
                    with open(fn, 'wb') as f:
                        dill.dump({'classifier':self.clf,'train_metrics':train_metrics,'test_metrics':test_metrics},f)

    def get_merge_predict_thr(self,data):
        thr = -1; #ntargets = data.shape[0]
        # without merge percentage specified just use the classifier predict().
        clf_predict = self.clf.predict(data)
        # use the merge percentage to merge only this percentage of the most confident mergers.
        # do not use any potential mergers that are against the classifier predict().
        if self.iterate_merge_perc > 0:
            clf_probs = self.clf.predict_proba(data)
            # xxx - figure out how to handle no mergers left more gracefully, just stop at this iteration?
            assert(clf_probs.ndim > 1 and clf_probs.shape[1] > 1) # ran out of mergers
            # use merge percentage as percentage of estimated yes merges remaining
            nmerge = (clf_predict==1).sum(dtype=np.int64); clf_probs = clf_probs[:,1]
            if nmerge > 0:
                thr = np.sort(clf_probs)[::-1][int(nmerge*self.iterate_merge_perc)]
                clf_predict = np.logical_and(clf_predict, clf_probs >= thr)
        return clf_predict, thr

    def get_chunk_inds(self, chunk):
        if self.use_chunk_range:
            chunk_list_index = np.nonzero(chunk >= self.chunk_range_cumsize)[0][-1]
            chunk_range_index = chunk - self.chunk_range_cumsize[chunk_list_index]
            cchunk = np.unravel_index(chunk_range_index, self.chunk_range_rng[chunk_list_index,:]) \
                *self.chunkscale[chunk_list_index,:] + self.chunk_range_beg[chunk_list_index,:]
        else:
            chunk_list_index = chunk; cchunk = self.chunk_range_beg[chunk,:]; chunk_range_index = None
        return cchunk, chunk_list_index, chunk_range_index

    # this method assumes binary classification
    # xxx - plotting code got really messy as per usual, clean this up
    def createPlots(self,target,sdata,clf,export_path,plot_features=False,name='clf',figno=100,thr=-1):
        ntargets = target.size; nfeatures = sdata.shape[1]
        bins = [np.arange(-5,5.1,0.1)] * nfeatures

        pl.figure(figno); plt.clf();
        if plot_features:
            self.plotFeatures(target,sdata,export_path=None,show_plot=False,bins=bins,figno=figno)
            nx, ny = 100, 100
            for x in range(nfeatures):
                for y in range(x+1,nfeatures):
                    pl.subplot(nfeatures-1,nfeatures-1,x*(nfeatures-1)+y)

                    # class 0 and 1 : get boundaries between classes, depending on classifier
                    x_min, x_max = plt.gca().get_xlim()
                    y_min, y_max = plt.gca().get_ylim()
                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                                         np.linspace(y_min, y_max, ny))
                    Y = np.zeros((nx*ny,nfeatures)); Y[:,[y,x]] = np.c_[xx.ravel(), yy.ravel()]
                    ZZ = clf.predict(Y)
                    try:
                        # use predict_proba with target merge percentage method to plot boundaries (if available)
                        Z = clf.predict_proba(Y)
                        if thr > 0:
                            Z = np.logical_and(ZZ,(Z[:,1] > thr)).astype(np.double).reshape(xx.shape)
                        else:
                            Z = Z[:, 1].reshape(xx.shape)
                    except:
                        # otherwise just use regular predict to plot boundaries
                        Z = ZZ.astype(np.double).reshape(xx.shape)

                    if nfeatures==2:
                        # if there are only two features, overlay probabilities from above with the feature densities
                        img = 1-np.abs(Z-0.5);
                        pl.imshow(img,interpolation='nearest',extent=(x_min,x_max,y_min,y_max),
                            aspect=(y_max-y_min)/(x_max-x_min), origin='lower', alpha=0.3, cmap='gray',)

                    # overlay the boundaries between the features
                    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='w')

                    try:
                        # plot the class means if available in the classifier (LDA)
                        plt.scatter(clf.means_[0][y], clf.means_[0][x], s=2, color='r', edgecolors='w')
                        plt.scatter(clf.means_[1][y], clf.means_[1][x], s=2, color='g', edgecolors='w')
                    except AttributeError:
                        pass

        # calculate ROC / PR style metrics using classifier predict (with target merge percentage method) and print
        clf_probs = None; clf_preds = clf.predict(sdata)
        if thr > 0:
            clf_probs = clf.predict_proba(sdata)
            # avoid error if there are no predictions left that predict yes merge
            if clf_probs.ndim > 1 and clf_probs.shape[1] > 1:
                clf_preds = np.logical_and(clf_preds,(clf_probs[:,1] > thr))
        yesmerge = (target==1); notmerge = (target==0);
        nyes = yesmerge.sum(dtype=np.int64); nnot = notmerge.sum(dtype=np.int64)
        fScore, tpr_recall, precision, pixel_error, fpr, tp, tn, fp, fn = pixel_error_fscore( target.astype(bool),
            clf_preds.astype(bool) )
        print('p=%d, n=%d, tp=%d, tn=%d, fp=%d, fn=%d, rec=%.4f, prec=%.4f, fscore=%.4f' % (nyes,nnot,tp,tn,fp,fn,
            tpr_recall,precision,fScore))

        # some plots that only work with certain classifiers, skip for non-relevant classifiers

        pl.figure(figno+1); plt.clf()
        try:
            # plot if the classifier implements predict_proba
            axes = pl.subplot(1,2,1)
            pbins = np.arange(0,1.01,0.01); binw = (pbins[1]-pbins[0])/2; cbins = pbins[:-1]+binw
            if clf_probs is None: clf_probs = clf.predict_proba(sdata)
            tnhist,tmp = np.histogram(1-clf_probs[np.logical_and(notmerge,clf_preds==0),0],bins=pbins)
            tnhist = tnhist.astype(np.double)/nnot
            fnhist,tmp = np.histogram(1-clf_probs[np.logical_and(yesmerge,clf_preds==0),0],bins=pbins)
            fnhist = fnhist.astype(np.double)/nyes
            # avoid error if there are no predictions left that predict yes merge
            if clf_probs.ndim > 1 and clf_probs.shape[1] > 1:
                tphist,tmp = np.histogram(clf_probs[np.logical_and(yesmerge,clf_preds==1),1],bins=pbins)
                tphist = tphist.astype(np.double)/nyes
                fphist,tmp = np.histogram(clf_probs[np.logical_and(notmerge,clf_preds==1),1],bins=pbins)
                fphist = fphist.astype(np.double)/nnot
            else:
                tphist = np.zeros_like(tnhist); fphist = np.zeros_like(fnhist)
            pl.plot(cbins,tphist,'g'); pl.plot(cbins,fphist,'g--')
            pl.plot(cbins,tnhist,'r'); pl.plot(cbins,fnhist,'r--')
            plt.xlim([-0.05,1.05]); plt.ylim([-0.05,1.05])
            plt.xlabel('P(merge)'); plt.ylabel('density'); plt.title('tp/fn Y(%d), tn/fp N(%d)' % (nyes,nnot))

            # plot for LDA classifier only
            axes = pl.subplot(1,2,2)
            plt.xticks(range(nfeatures),rotation=45)
            #FEATURES, FEATURES_NAMES, nfeatures = dpFRAG.make_FEATURES()    # xxx - meh
            axes.set_xticklabels(self.features_names)
            pl.plot(clf.scalings_[:,0],'k')
            d = clf.decision_function(clf.means_)   # distance from means to the decision boundary
            pl.title('dist: not %.5f yes %.5f' % (d[0],d[1]))
        except (AttributeError, TypeError) as e:
            pass

        if export_path:
            figna = [x % (name,) for x in ['merge_features_%s.png', 'merge_probs_eigen_%s.png']]
            nfigna = len(figna)
            for f,i in zip(range(figno, figno+nfigna), range(nfigna)):
                pl.figure(f)
                figure = plt.gcf() # get current figure
                figure.set_size_inches(20, 20)
                plt.savefig(os.path.join(export_path,figna[i]), dpi=72)
        else:
            pl.show()

        # return the ROC style metrics
        metrics = {}
        for i in ('nyes', 'nnot', 'tp', 'tn', 'fp', 'fn', 'tpr_recall', 'precision', 'fScore', 'pixel_error'):
            metrics[i] = locals()[i]
        return metrics

    def plotFeatures(self,target,fdata,export_path,show_plot=True,bins=None,figno=100):
        nfeatures = fdata.shape[1]; #ntargets = target.size
        yesmerge = (target==1); notmerge = (target==0);
        nyes = yesmerge.sum(dtype=np.int64); nnot = notmerge.sum(dtype=np.int64)
        if not bins:
            bins = [np.arange(0,4.6,0.125), np.arange(0,6.15,0.15), np.arange(0.75,3.1,0.075), np.arange(0,257,8),
                np.arange(0,1.025,0.025), np.arange(0,1.025,0.025), np.arange(0,1.025,0.025)]
        nbins = [x.size-1 for x in bins]; print(nbins)
        binw = [(x[1]-x[0])/2 for x in bins]; cbins = [x[:-1]+y for x,y in zip(bins,binw)]
        pl.figure(figno)
        if show_plot: plt.clf()
        for x in range(nfeatures):
            for y in range(x+1,nfeatures):
                pl.subplot(nfeatures-1,nfeatures-1,x*(nfeatures-1)+y)

                img = np.zeros((nbins[x],nbins[y],3),dtype=np.double)
                img[:,:,1] = np.histogram2d(fdata[yesmerge, x], fdata[yesmerge, y], bins=(bins[x],bins[y]))[0]/nyes
                img[:,:,0] = np.histogram2d(fdata[notmerge, x], fdata[notmerge, y], bins=(bins[x],bins[y]))[0]/nnot
                sel = (img > 0); img[sel] = -np.log10(img[sel]); sel = (img > 0); img[sel] = 1-img[sel]/img.max()
                #sel = (img > 0.05);
                bnd = nd.measurements.find_objects(sel);
                #xlim = (bins[x][bnd[0][0].start]+binw[x]/10, bins[x][bnd[0][0].stop]-binw[x]/10)
                #ylim = (bins[y][bnd[0][1].start]+binw[y]/10, bins[y][bnd[0][1].stop]-binw[y]/10)
                if len(bnd) == 0:
                    xlim = [0,1]; ylim = [0,1]
                else:
                    xlim = (cbins[x][bnd[0][0].start], cbins[x][bnd[0][0].stop-1])
                    ylim = (cbins[y][bnd[0][1].start], cbins[y][bnd[0][1].stop-1])

                # imshow uses f-order so x/y are flipped
                pl.imshow(img,interpolation='nearest',extent=(cbins[y][0],cbins[y][-1],cbins[x][0],
                    cbins[x][-1]), aspect=(cbins[y][-1]-cbins[y][0])/(cbins[x][-1]-cbins[x][0]), origin='lower')
                plt.xlim(ylim); plt.ylim(xlim)

                if y==x+1:
                    #FEATURES, FEATURES_NAMES, nfeatures = dpFRAG.make_FEATURES()    # xxx - meh
                    plt.xlabel(self.features_names[y])
                    plt.ylabel(self.features_names[x])
                else:
                    plt.gca().axes.get_xaxis().set_visible(False)
                    plt.gca().axes.get_yaxis().set_visible(False)

        if show_plot:
            if export_path:
                plt.gcf().set_size_inches(20, 20)
                plt.savefig(os.path.join(export_path,'merge_features.png'), dpi=72)
            else:
                pl.show()

    @staticmethod
    def get_options(cfgfile):
        config = ConfigObj(cfgfile,
            configspec=os.path.join(os.path.dirname(os.path.realpath(__file__)),'dpSupervoxelClassifier.ini'))

        # Validator handles missing / type / range checking
        validator = Validator()
        results = config.validate(validator, preserve_errors=True)
        if results != True:
            for (section_list, key, err) in flatten_errors(config, results):
                if key is not None:
                    if not err:
                        print('The "%s" key is missing in the following section(s):%s ' \
                            % (key, ', '.join(section_list)))
                        raise ValidateError
                    else:
                        print('The "%s" key in the section(s) "%s" failed validation' \
                            % (key, ', '.join(section_list)))
                        raise err
                elif section_list:
                    print('The following section(s) was missing:%s ' % ', '.join(section_list))
                    raise ValidateError

        return config

    @staticmethod
    def addArgs(p):
        p.add_argument('--cfgfile', nargs=1, type=str, default='', help='Path/name of ini config file')
        p.add_argument('--trainin', nargs=1, type=str, default='', help='Input file for loading training data (dill)')
        p.add_argument('--classifier', nargs=1, type=str, default='lda', help='Which sklearn classifier to use')
        p.add_argument('--classifierin', nargs=1, type=str, default='',
            help='Input file for loading trained classifier(s) (dill)')
        p.add_argument('--classifierout', nargs=1, type=str, default='',
            help='Output file for saving trained classifier(s) (dill)')
        p.add_argument('--testin', nargs=1, type=str, default='', help='Input file for loading testing data (dill)')
        p.add_argument('--test-chunks', nargs='*', type=int, default=[],
            metavar='CHUNKS', help='Chunks to use for test (override from .ini)')
        p.add_argument('--show-plots', action='store_true', help='Show various plots')
        p.add_argument('--export-plots', nargs=1, type=str, default='',
            help='Export various plots to this path (default no plots)')
        p.add_argument('--plot-features', action='store_true', help='If plotting, whether to include feature plots')
        p.add_argument('--outfile', nargs=1, type=str, default='', help='Override output file for agglomerations')
        p.add_argument('--labelfile', nargs=1, type=str, default='', help='Override input label (supervoxel) file')
        p.add_argument('--probfile', nargs=1, type=str, default='', help='Override input prob file')
        p.add_argument('--probaugfile', nargs=1, type=str, default='', help='Override input prob augment file')
        p.add_argument('--progress-bar', action='store_true', help='Enable progress bar if available')
        p.add_argument('--feature-set', nargs=1, type=str, default='',
            help='Option to control which FRAG features are calculated (override from .ini)')
        # these are so standard cubeIter inputs can be used from command line to override from .ini
        p.add_argument('--chunk', nargs=3, type=int, default=[-1,-1,-1], metavar=('X', 'Y', 'Z'),
            help='Corner chunk to parse out of hdf5')
        p.add_argument('--offset', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
            help='Offset in chunk to read')
        p.add_argument('--size', nargs=3, type=int, default=[256,256,128], metavar=('X', 'Y', 'Z'),
            help='Size in voxels to read')
        p.add_argument('--nthreads', nargs=1, type=int, default=[8],
            help='Number of parallel threads to set for scipy and scikit-learn')
        p.add_argument('--chunk-subgroups', action='store_true',
            help='This mode is for probs and labels that have overlapping context so need to be stored separately.')
        p.add_argument('--neighbor-only', dest='neighbor_only', action='store_true',
            help='Only use boundary voxels labeled with neighboring supervoxels (no background / non-neighbor voxels)')
        p.add_argument('--prob-svox-context', dest='prob_svox_context', action='store_true',
            help='Use context for loading probs and supervoxels along edge faces')
        p.add_argument('--no-agglo-ECS', action='store_true', help='Do not agglomerate ECS supervoxels')
        p.add_argument('--useFRAGc', action='store_true', help='Use the C-optimized version of FRAG')

        p.add_argument('--dpSupervoxelClassifier-verbose', action='store_true',
            help='Debugging output for dpSupervoxelClassifier')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flattened Supervoxel Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpSupervoxelClassifier.addArgs(parser)
    args = parser.parse_args()

    svoxClass = dpSupervoxelClassifier(args)
    if svoxClass.iterative_mode:
        svoxClass.iterative_classify()
    else:
        svoxClass.train()
        svoxClass.test()
