#!/usr/bin/env python
'''
Top level for EM supervoxel classifier that learns merge / no merge classification based on some feature set.
Featured Region Adjacency Graph (dpFRAG.py) creates the graph of neighboring supervoxels and their features.

There is no iteration in this procedure, typically referred to as "flat learning" in the context of merging superpixels.

Created pwatkins, Dec 7, 2015

Example invocations:

./dpSupervoxelClassifier.py --cfgfile config/svox_M0007_huge_egmini.ini --test-chunks 0 --dpSupervoxelClassifier-verbose 

./dpSupervoxelClassifier.py --cfgfile config/svox_M0007_huge.ini --test-chunks 0 --dpSupervoxelClassifier-verbose 

./dpSupervoxelClassifier.py --cfgfile config/svox_M0007_huge.ini --test-chunks 0 --dpSupervoxelClassifier-verbose --trainin out_train.dill

'''

import h5py
import numpy as np
import time
import argparse
import os
#import sys
from io import BytesIO

from configobj import ConfigObj, flatten_errors
from validate import Validator, ValidateError
import dill
from io import StringIO
from sklearn.preprocessing import scale
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
import matplotlib as mpl
from matplotlib import colors

from dpFRAG import dpFRAG
from metrics import pixel_error_fscore

class dpSupervoxelClassifier():

    # Constants
    LIST_ARGS = ['test_chunks', 'label_subgroups', 'label_subgroups_out']
    n_jobs = 8
    #n_jobs = 1
    
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
        if len(self.threshold_subgroups) == 0:
            self.threshold_subgroups = self.thresholds
        else:
            assert( len(self.thresholds) == len(self.threshold_subgroups) )

        self.doplots = (self.show_plots or self.export_plots)

        # initialize for "chunkrange" or "chunklist" mode if these parameters are not empty
        self.chunk_range_beg = self.chunk_range_beg.reshape(-1,3); self.use_chunk_range = False
        self.nchunk_list = self.chunk_range_beg.shape[0]
        self.nchunks = self.nchunk_list

        if len(self.chunk_range_end) > 0:
            # "chunkrange" mode, chunks are selected based on defined beginning and end of ranges in X,Y,Z
            # range is open ended (python-style, end is not included in range)
            self.chunk_range_end = self.chunk_range_end.reshape(-1,3);
            assert( self.chunk_range_end.shape[0] == self.nchunk_list )
            self.chunk_range_rng = self.chunk_range_end - self.chunk_range_beg
            assert( (self.chunk_range_rng >= 0).all() )     # some bad ranges
            self.chunk_range_size = self.chunk_range_rng.prod(axis=1)
            self.chunk_range_cumsize = np.concatenate((np.zeros((1,),dtype=self.chunk_range_size.dtype), 
                self.chunk_range_size.cumsum()))
            self.chunk_range_nchunks = self.chunk_range_cumsize[-1]
            self.use_chunk_range = True; self.nchunks = self.use_chunk_range

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

    def train(self):

        if self.dpSupervoxelClassifier_verbose: 
            print('\nTRAIN')

        if self.trainin and (not self.classifierin or self.doplots):
            if self.dpSupervoxelClassifier_verbose: 
                print('Loading training data')
            with open(self.trainin, 'rb') as f: data = dill.load(f)
            target = data['target']; fdata = data['data']
            ntargets = target.size; nfeatures = fdata.shape[1]
            assert( nfeatures == len(dpFRAG.FEATURES) ) # xxx - FRAG features currently static
            
        elif not self.classifierin: 
            #dict_keys(['feature_names', 'DESCR', 'target_names', 'target', 'data'])
            nalloc = self.nchunks*self.nalloc_per_chunk
            ntargets = 0; nfeatures = len(dpFRAG.FEATURES)
            target = np.zeros((nalloc,), dtype=np.int64)
            fdata = np.zeros((nalloc,nfeatures), dtype=np.double)
        
            # accumulate training data from all training chunks
            for chunk in range(self.nchunks):
                cchunk, chunk_list_index, chunk_range_index = self.get_chunk_inds(chunk)
                offset = self.offset_list[chunk_list_index,:]; size = self.size_list[chunk_list_index,:]
    
                if chunk_list_index in self.test_chunks: continue
                print('Appending training data for chunk %d,%d,%d' % tuple(cchunk.tolist()))
                
                frag = dpFRAG.makeTrainingFRAG(self.labelfile, cchunk, size, offset, self.probfile, self.rawfile, 
                    self.raw_dataset, self.gtfile, self.label_subgroups, 
                    verbose=self.dpSupervoxelClassifier_verbose)
                frag.createFRAG(); data = frag.createDataset()
                ctargets = data['target'].shape[0]
                target[ntargets:ntargets+ctargets] = data['target']
                fdata[ntargets:ntargets+ctargets,:] = data['data']
                ntargets += ctargets
            target = target[:ntargets]; fdata = fdata[:ntargets,:]
    
            if self.trainout:
                #dict_keys(['feature_names', 'DESCR', 'target_names', 'target', 'data'])
                descr = 'Training data from dpFRAG.py with command line:\n' + self.arg_str
                descr = ('With ini file "%s":\n' % (self.cfgfile,)) + self.ini_str
                data['data'] = fdata; data['target'] = target; data['DESCR'] = descr
                with open(self.trainout, 'wb') as f: dill.dump(data, f)

        if not self.classifierin or self.doplots:
            # everyone wants to be norml
            sdata = scale(fdata)   # normalize for the classifiers

        if self.classifierin:
            if self.dpSupervoxelClassifier_verbose: 
                print('\nLoading classifier:'); t = time.time()
                
            with open(self.classifierin, 'rb') as f: d = dill.load(f)
            self.clf = d['classifier']; 
        else:        
            if self.dpSupervoxelClassifier_verbose: 
                print('\nTraining classifier %s with %d examples and %d features:' % (self.classifier, 
                    ntargets, nfeatures)); t = time.time()

            # train a classifier
            if self.classifier == 'lda':
                self.clf = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
            elif self.classifier == 'qda':
                self.clf = QuadraticDiscriminantAnalysis()
            elif self.classifier == 'rf':
                # the gala parameters
                #self.clf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=20,
                #    bootstrap=False, random_state=None)
                self.clf = RandomForestClassifier(n_estimators=5*nfeatures,n_jobs=self.n_jobs,max_depth=10)
            elif self.classifier == 'svm':
                self.clf = svm.SVC(kernel='rbf',probability=True,cache_size=2000)
            elif self.classifier == 'nb':
                self.clf = GaussianNB()
            elif self.classifier == 'kn':
                self.clf = KNeighborsClassifier(n_neighbors=10,n_jobs=self.n_jobs)
            elif self.classifier == 'dc':
                self.clf = DecisionTreeClassifier(max_depth=10)
            elif self.classifier == 'ada':
                self.clf = AdaBoostClassifier()
            elif self.classifier == 'lr':
                self.clf = LogisticRegression(penalty='l2',dual=False,solver='sag',n_jobs=self.n_jobs)
            else:
                assert(False)   # i never try anything, i just do it

            # train to the normalized data and merge or no merge targets
            self.clf.fit(sdata, target)

            if self.classifierout:
                with open(self.classifierout, 'wb') as f: dill.dump({'classifier':self.clf}, f)

        if self.dpSupervoxelClassifier_verbose: 
            print('\n\tdone in %.4f s' % (time.time() - t))

        if self.doplots: 
            dpSupervoxelClassifier.plotClfFeatures(target,sdata,self.clf,self.export_plots,
                name=self.classifier + '_train_' + ' '.join([str(x) for x in self.test_chunks]))
            #dpSupervoxelClassifier.plotFeatures(target,fdata,self.export_plots)

    def test(self):
        if self.dpSupervoxelClassifier_verbose: 
            print('\nTEST')

        for chunk in range(self.nchunks):
            cchunk, chunk_list_index, chunk_range_index = self.get_chunk_inds(chunk)
            offset = self.offset_list[chunk_list_index,:]; size = self.size_list[chunk_list_index,:]

            if chunk_list_index not in self.test_chunks: continue
            print('Exporting testing data for chunk %d,%d,%d' % tuple(cchunk.tolist()))

            FRAG = None
            if len(self.test_chunks) == 1 and self.testin:
                if self.dpSupervoxelClassifier_verbose: 
                    print('Loading testing data')
                with open(self.testin, 'rb') as f: data = dill.load(f)
                FRAG = data['FRAG']; data = data['data']

            if self.doplots:                
                frag = dpFRAG.makeBothFRAG(self.labelfile, cchunk, size, offset, self.probfile, self.rawfile, 
                    self.raw_dataset, self.gtfile, self.outfile, self.label_subgroups, self.label_subgroups_out, 
                    G=FRAG, verbose=self.dpSupervoxelClassifier_verbose)
            else:
                frag = dpFRAG.makeTestingFRAG(self.labelfile, cchunk, size, offset, self.probfile, self.rawfile, 
                    self.raw_dataset, self.outfile, self.label_subgroups, self.label_subgroups_out, G=FRAG,
                    verbose=self.dpSupervoxelClassifier_verbose)

            if not (len(self.test_chunks) == 1 and self.testin):
                frag.createFRAG(); data = frag.createDataset(train=self.doplots)
                
                if self.dpSupervoxelClassifier_verbose: 
                    print('Dumping testing data')
                descr = 'Testing data from dpFRAG.py with command line:\n' + self.arg_str
                descr = ('With ini file "%s":\n' % (self.cfgfile,)) + self.ini_str
                data['DESCR'] = descr
                with open(self.testout, 'wb') as f: dill.dump({'data':data,'FRAG':frag.FRAG}, f)

            sdata = scale(data['data'])     # normalize for the classifiers

            if self.doplots: 
                dpSupervoxelClassifier.plotClfFeatures(data['target'],sdata,self.clf,self.export_plots,
                    name=self.classifier + '_test_' + ' '.join([str(x) for x in self.test_chunks]))
            try:
                # predict merge or not on testing cube and write outputs at specified probability thresholds
                frag.threshold_agglomerate(self.clf.predict_proba(sdata), self.thresholds, self.threshold_subgroups)
            except:
                # if the classifier doesn't do probabilities just export single prediction
                frag.subgroups_out += ['single_' + self.classifier]
                frag.agglomerate(self.clf.predict(sdata))
                
    def get_chunk_inds(self, chunk):
        if self.use_chunk_range:
            chunk_list_index = np.nonzero(chunk >= self.chunk_range_cumsize)[0][-1]
            chunk_range_index = chunk - self.chunk_range_cumsize[chunk_list_index]
            cchunk = np.unravel_index(chunk_range_index, self.chunk_range_rng[chunk_list_index,:]) \
                + self.chunk_range_beg[chunk_list_index,:]
        else:
            chunk_list_index = chunk; cchunk = self.chunk_range_beg[chunk,:]; chunk_range_index = None
        return cchunk, chunk_list_index, chunk_range_index

    @staticmethod
    def plotClfFeatures(target,sdata,clf,export_path,name='clf',figno=100):
        ntargets = target.size; nfeatures = sdata.shape[1]
        bins = [np.arange(-5,5.1,0.1)] * nfeatures
        dpSupervoxelClassifier.plotFeatures(target,sdata,export_path=None,show_plot=False,bins=bins,figno=figno)

        nx, ny = 100, 100
        for x in range(nfeatures):
            for y in range(x+1,nfeatures):
                pl.figure(figno); pl.subplot(nfeatures-1,nfeatures-1,x*(nfeatures-1)+y)

                # class 0 and 1 : areas
                x_min, x_max = plt.gca().get_xlim()
                y_min, y_max = plt.gca().get_ylim()
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                                     np.linspace(y_min, y_max, ny))
                Y = np.zeros((nx*ny,nfeatures)); Y[:,[y,x]] = np.c_[xx.ravel(), yy.ravel()]
                try:
                    Z = clf.predict_proba(Y); Z = Z[:, 1].reshape(xx.shape)
                except AttributeError:
                    Z = clf.predict(Y).astype(np.double).reshape(xx.shape)
                
                if nfeatures==2:
                    img = 1-np.abs(Z-0.5);
                    pl.imshow(img,interpolation='nearest',extent=(x_min,x_max,y_min,y_max), 
                        aspect=(y_max-y_min)/(x_max-x_min), origin='lower', alpha=0.3, cmap='gray',)
                    plt.contour(xx, yy, Z, [0.5], linewidths=1., colors='w')
                else:
                    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='w')

                try:
                    plt.scatter(clf.means_[0][y], clf.means_[0][x], s=2, color='r', edgecolors='w')
                    plt.scatter(clf.means_[1][y], clf.means_[1][x], s=2, color='g', edgecolors='w')
                except AttributeError:
                    pass

                try:
                    for i in range(clf.n_clusters):
                        plt.scatter(clf.cluster_centers_[i][y], clf.cluster_centers_[i][x], s=2, color='w')
                except AttributeError:
                    pass

        # this code assumes binary classification
        clf_preds = clf.predict(sdata)
        yesmerge = (target==1); notmerge = (target==0); 
        nyes = yesmerge.sum(dtype=np.int64); nnot = notmerge.sum(dtype=np.int64)
        try:
            pl.figure(figno+1)
            pbins = np.arange(0,1.01,0.01); binw = (pbins[1]-pbins[0])/2; cbins = pbins[:-1]+binw
            clf_probs = clf.predict_proba(sdata);
            tphist,tmp = np.histogram(clf_probs[np.logical_and(yesmerge,clf_preds==1),1],bins=pbins)
            tphist = tphist.astype(np.double)/nyes
            fphist,tmp = np.histogram(clf_probs[np.logical_and(yesmerge,clf_preds==0),0],bins=pbins)
            fphist = fphist.astype(np.double)/nyes
            tnhist,tmp = np.histogram(1-clf_probs[np.logical_and(notmerge,clf_preds==0),0],bins=pbins)
            tnhist = tnhist.astype(np.double)/nnot
            fnhist,tmp = np.histogram(1-clf_probs[np.logical_and(notmerge,clf_preds==1),1],bins=pbins)
            fnhist = fnhist.astype(np.double)/nnot
            pl.plot(cbins,tphist,'g'); pl.plot(cbins,fphist,'g--')
            pl.plot(cbins,tnhist,'r'); pl.plot(cbins,fnhist,'r--')
            #pl.plot(cbins,np.log10(tphist),'g'); pl.plot(cbins,np.log10(fphist),'g--')
            #pl.plot(cbins,np.log10(tnhist),'r'); pl.plot(cbins,np.log10(fnhist),'r--')
            plt.xlim([-0.05,1.05]); plt.ylim([-0.05,1.05])
            plt.xlabel('P(merge)')
        except AttributeError:
            pass

        fScore, tpr_recall, precision, pixel_error, tp,tn,fp,fn = pixel_error_fscore( target.astype(np.bool), 
            clf_preds.astype(np.bool) )
        print('p=%d, n=%d, tp=%d, tn=%d, fp=%d, fn=%d, rec=%.4f, prec=%.4f, fscore=%.4f' % (nyes,nnot,tp,tn,fp,fn,
            tpr_recall,precision,fScore))

        if export_path:
            figna = ['merge_features_%s.png' % (name,), 'merge_probs_%s.png' % (name,)]
            nfigna = len(figna)
            for f,i in zip(range(figno, figno+nfigna), range(nfigna)):
                pl.figure(f)
                figure = plt.gcf() # get current figure
                figure.set_size_inches(20, 20)
                plt.savefig(os.path.join(export_path,figna[i]), dpi=72)
        else:
            pl.show()

    @staticmethod
    def plotFeatures(target,fdata,export_path,show_plot=True,bins=None,figno=100):
        ntargets = target.size; nfeatures = fdata.shape[1]
        yesmerge = (target==1); notmerge = (target==0); 
        nyes = yesmerge.sum(dtype=np.int64); nnot = notmerge.sum(dtype=np.int64)
        if not bins:
            bins = [np.arange(0,4.6,0.125), np.arange(0,6.15,0.15), np.arange(0.75,3.1,0.075), np.arange(0,257,8), 
                np.arange(0,1.025,0.025), np.arange(0,1.025,0.025), np.arange(0,1.025,0.025)]
        nbins = [x.size-1 for x in bins]; print(nbins)
        binw = [(x[1]-x[0])/2 for x in bins]; cbins = [x[:-1]+y for x,y in zip(bins,binw)]
        for x in range(nfeatures):
            for y in range(x+1,nfeatures):
                pl.figure(figno); pl.subplot(nfeatures-1,nfeatures-1,x*(nfeatures-1)+y)

                img = np.zeros((nbins[x],nbins[y],3),dtype=np.double)
                img[:,:,1] = np.histogram2d(fdata[yesmerge, x], fdata[yesmerge, y], bins=(bins[x],bins[y]))[0]/nyes
                img[:,:,0] = np.histogram2d(fdata[notmerge, x], fdata[notmerge, y], bins=(bins[x],bins[y]))[0]/nnot
                sel = (img > 0); img[sel] = -np.log10(img[sel]); sel = (img > 0); img[sel] = 1-img[sel]/img.max()
                #sel = (img > 0.05); 
                bnd = nd.measurements.find_objects(sel); 
                #xlim = (bins[x][bnd[0][0].start]+binw[x]/10, bins[x][bnd[0][0].stop]-binw[x]/10)
                #ylim = (bins[y][bnd[0][1].start]+binw[y]/10, bins[y][bnd[0][1].stop]-binw[y]/10)
                xlim = (cbins[x][bnd[0][0].start], cbins[x][bnd[0][0].stop-1])
                ylim = (cbins[y][bnd[0][1].start], cbins[y][bnd[0][1].stop-1])
                
                # imshow uses f-order so x/y are flipped
                pl.imshow(img,interpolation='nearest',extent=(cbins[y][0],cbins[y][-1],cbins[x][0],
                    cbins[x][-1]), aspect=(cbins[y][-1]-cbins[y][0])/(cbins[x][-1]-cbins[x][0]), origin='lower')
                plt.xlim(ylim); plt.ylim(xlim)
                    
                if y==x+1: 
                    plt.xlabel(dpFRAG.FEATURES_NAMES[y])
                    plt.ylabel(dpFRAG.FEATURES_NAMES[x])
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
            help='Input file for loading trained classifier (dill)')
        p.add_argument('--testin', nargs=1, type=str, default='', help='Input file for loading testing data (dill)')
        p.add_argument('--test-chunks', nargs='*', type=int, default=[], 
            metavar='CHUNKS', help='Chunks to use for test (override from .ini)')
        p.add_argument('--show-plots', action='store_true', help='Show various plots')
        p.add_argument('--export-plots', nargs=1, type=str, default='', 
            help='Export various plots to this path (default no plots)')
        
        p.add_argument('--dpSupervoxelClassifier-verbose', action='store_true', 
            help='Debugging output for dpSupervoxelClassifier')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flattened Supervoxel Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpSupervoxelClassifier.addArgs(parser)
    args = parser.parse_args()
    
    svoxClass = dpSupervoxelClassifier(args)
    svoxClass.train()
    svoxClass.test()
    
