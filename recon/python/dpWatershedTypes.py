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

# Python object for reading EM voxel type probabilities and watershedding probabilities to create labels or supervoxels. 
#     This method was forked off of watershedEMprobs.py and labelEMComponents.py, dropping the affinity approach and
#     using the iterative threshold approach with "peak-detection", but instead on each foreground class separately.
#     This is normally ICS and ECS. Also assumes that probability hdf5 contains a single background label type. This is
#     normally the MEMbrane type.
# Supervoxels generated at each threshold iteration are saved to the output hdf5.
# Assumes that probabilities from multiple networks have been merged (averged) into a single probability for each class.
#     This can be done without any other frills using mergeEMprobs.py.


#import os, sys
import argparse
import time
import numpy as np
from scipy import ndimage as nd
from skimage import morphology as morph

from dpLoadh5 import dpLoadh5
from typesh5 import emLabels, emProbabilities, emVoxelType
from pyCext import binary_warping

class dpWatershedTypes(object):

    def __init__(self, args):
        # save command line arguments from argparse, see definitions in main or run with --help
        for k, v in vars(args).items(): 
            if type(v) is list and k not in ['ThrHi', 'ThrLo', 'fg_types_labels']: 
                # do not save items that are known to be lists (even if one element) as single elements
                if len(v)==1 and k not in ['fg_types', 'Tmins']:
                    setattr(self,k,v[0])  # save single element lists as first element
                elif type(v[0]) is int:   # convert the sizes and offsets to numpy arrays
                    setattr(self,k,np.array(v,dtype=np.int32))
                elif type(v[0]) is float:   # convert float arrays to numpy arrays
                    setattr(self,k,np.array(v,dtype=np.double))
                else:
                    setattr(self,k,v)   # store other list types as usual (floats)
            else:
                setattr(self,k,v)

        # initialize class properties
        self.nfg_types = len(self.fg_types); self.types = [self.bg_type] + self.fg_types
        self.ntypes = self.nfg_types + 1
        self.Ts = np.arange(self.ThrRng[0], self.ThrRng[1], self.ThrRng[2])
        if self.ThrLo: self.Ts = np.concatenate((np.array(self.ThrLo), self.Ts))
        if self.ThrHi: self.Ts = np.concatenate((self.Ts, np.array(self.ThrHi)))
        self.Ts = np.sort(self.Ts)  # just to be sure
        self.nthresh = self.Ts.size
        self.nTmin = self.Tmins.size
        if not self.fg_types_labels: self.fg_types_labels = [-1]* self.nfg_types
        assert( len(self.fg_types_labels) == self.nfg_types )

        # other input validations
        assert( (self.Ts > 0).all() and (self.Ts < 1).all() )
        assert( (self.Tmins > 1).all() )   # iterative cc's method needs a min threshold to "keep" small supervoxels
        assert( self.skimWatershed or self.connectivity in [1,3] )  # warping does not support 18-conn (no LUT)

        # xxx - intended skeletonizatino for GT objects, needs updating
        self.skeletonize = False

        # print out all initialized variables in verbose mode
        if self.dpWatershedTypes_verbose: print('dpWatershedTypes, verbose mode:\n'); print(vars(self))

    def watershed_cube(self):
        # load the probability data, allocate as array of volumes instead of 4D ndarray to maintain C-order volumes
        probs = [None]*self.ntypes; bwseeds = [None]*self.ntypes;
        if self.srclabels:
            # this path is typically not used in favor of the label checker for fully labeled 3d gt components.
            # but, some ground truth (for example, 2d ECS cases) was only labeled with voxel type,
            #   so this is used to create ground truth components from the voxel types.
            loadh5 = emLabels.readLabels(srcfile=self.srclabels, chunk=self.chunk.tolist(), offset=self.offset.tolist(), 
                size=self.size.tolist(), data_type='uint16', verbose=self.dpWatershedTypes_verbose)
            self.datasize = loadh5.datasize; self.chunksize = loadh5.chunksize; self.attrs = loadh5.data_attrs
            # pre-allocate for srclabels method, labeled areas are set to prob of 1 below
            for i in range(self.ntypes): probs[i] = np.zeros(self.size, dtype=emProbabilities.PROBS_DTYPE, order='C')
            if self.TminSrc < 2:
                # simple method with no "cleaning"
                for i in range(self.ntypes): probs[i][loadh5.data_cube==i] = 1
            else:
                # optionally "clean" labels by removing small bg and fg components for each foreground type
                fgbwlabels = np.zeros(self.size, dtype=np.bool)
                for i in range(self.nfg_types): 
                    # background connected components and threshold
                    comps, nlbls = nd.measurements.label(loadh5.data_cube!=i+1)
                    comps, sizes = emLabels.thresholdSizes(comps, minSize=self.TminSrc)
                    # foreground connected components and threshold
                    comps, nlbls = nd.measurements.label(comps==0)
                    comps, sizes = emLabels.thresholdSizes(comps, minSize=self.TminSrc)
                    # keep track of mask for all foreground types
                    bwlabels = (comps > 0); fgbwlabels = np.logical_or(fgbwlabels, bwlabels)
                    probs[i+1][bwlabels] = 1
                # set background type as all areas that are not in foreground types after "cleaning"
                probs[0][np.logical_not(fgbwlabels)] = 1
        else:
            for i in range(self.ntypes):
                loadh5 = dpLoadh5.readData(srcfile=self.probfile, dataset=self.types[i], chunk=self.chunk.tolist(), 
                    offset=self.offset.tolist(), size=self.size.tolist(), data_type=emProbabilities.PROBS_STR_DTYPE, 
                    verbose=self.dpWatershedTypes_verbose)
                self.datasize = loadh5.datasize; self.chunksize = loadh5.chunksize; self.attrs = loadh5.data_attrs
                probs[i] = loadh5.data_cube.copy(order='C'); del loadh5

        # save some of the parameters as attributes
        self.attrs['types'] = self.types; self.attrs['fg_types'] = self.fg_types
        self.attrs['fg_types_labels'] = self.fg_types_labels
        
        # save connnetivity structure and warping LUT because used on each iteration (for speed)
        self.bwconn = nd.morphology.generate_binary_structure(dpLoadh5.ND, self.connectivity)
        self.simpleLUT = None
        
        # background type was put first, so voxType of zero is background
        voxType = np.concatenate([x.reshape(x.shape + (1,)) for x in probs], axis=3).argmax(axis=3)  
        # write out the winning type for each voxel
        # save some params from this watershed run in the attributes
        d = self.attrs.copy(); d['thresholds'] = self.Ts; d['Tmins'] = self.Tmins
        emVoxelType.writeVoxType(outfile=self.outlabels, chunk=self.chunk.tolist(), 
            offset=self.offset.tolist(), size=self.size.tolist(), datasize=self.datasize.tolist(), 
            chunksize=self.chunksize.tolist(), verbose=self.dpWatershedTypes_verbose, attrs=d,
            data=voxType.astype(emVoxelType.VOXTYPE_DTYPE))

        # only allow a voxel to be included in the type of component that had max prob for that voxel.
        # do this by setting the non-winning probabilities to zero.
        for i in range(self.ntypes): probs[i][voxType != i] = 0;
        
        # create a type mask for each foreground type to select only current voxel type (winner-take-all from network)
        voxTypeSel = [None] * self.nfg_types; voxTypeNotSel =  [None] * self.nfg_types
        for i in range(self.nfg_types): 
            voxTypeSel[i] = (voxType == i+1); voxTypeNotSel[i] = np.logical_not(voxTypeSel[i])

        # iteratively apply thresholds, each time only keeping components that have fallen under Tmin.
        # at last iteration keep all remaining components.
        # do this separately for foreground types
        for k in range(self.nTmin):
            for i in range(self.ntypes): bwseeds[i] = np.zeros(self.size, dtype=np.bool, order='C')
            for i in range(self.nthresh):
                if self.dpWatershedTypes_verbose: 
                    print('creating supervoxels at threshold = %.8f with Tmin = %d' % (self.Ts[i], self.Tmins[k]))
                    t = time.time()
                types_labels = [None]*self.nfg_types; types_uclabels = [None]*self.nfg_types;
                if self.skeletonize: types_sklabels = [None]*self.nfg_types 
                types_nlabels = np.zeros((self.nfg_types,),dtype=np.int64)
                types_ucnlabels = np.zeros((self.nfg_types,),dtype=np.int64)
                for j in range(self.nfg_types):
                    # run connected components at this threshold on labels
                    labels, nlabels = nd.measurements.label(probs[j+1] > self.Ts[i], self.bwconn)

                    # merge the current thresholded components with the previous seeds to get current bwlabels
                    bwlabels = np.logical_or(labels, bwseeds[j])
                
                    # take the current components under threshold and merge with the seeds for the next iteration
                    if i < self.nthresh-1:
                        labels, sizes = emLabels.thresholdSizes(labels, minSize=-self.Tmins[k])
                        bwseeds[j] = np.logical_or(labels, bwseeds[j])

                    if self.skimWatershed:
                        # xxx - still trying to evaluate if there is any advantage to this more traditional watershed.
                        #   it does not leave a non-adjacency boundary and is about 1.5 times slower than bwmorph
                    
                        # run connected components on the thresholded labels merged with previous seeds
                        labels, nlabels = nd.measurements.label(bwlabels, self.bwconn)
                    
                        # run a true watershed based the current foreground probs using current components as markers
                        labels = morph.watershed(probs[j+1], labels, connectivity=self.bwconn, mask=voxTypeSel[j])
                        
                        # remove any adjacencies created during the watershed
                        # NOTE: currently this only removes 6-connectivity, no matter what specified connecitity is
                        # xxx - some method of removing adjacencies with arbitrary connectivity?
                        uclabels, ucnlabels = emLabels.remove_adjacencies(labels)
                    else:
                        if self.probWatershed and i>1:
                            # this is an alternative to the traditional watershed that warps out only based on stepping
                            #   back through the thresholds in reverse order. has advantages of non-connectivity using
                            #   the warping error and preliminarily shows better metric performance.
                            # about 2 times slower than regular warping method.
                        
                            # make an unconnected version of bwlabels by warping out but with mask only for this type
                            # everything above current threshold is already labeled, so only need to use gray thresholds
                            #    starting below the current threshold level.
                            bwlabels, diff, self.simpleLUT = binary_warping(bwlabels, np.ones(self.size,dtype=np.bool), 
                                mask=voxTypeSel[j], borderval=False, slow=True, simpleLUT=self.simpleLUT, 
                                connectivity=self.connectivity, gray=probs[j+1], 
                                grayThresholds=self.Ts[i-1::-1].astype(np.float32, order='C'))
                        else:
                            # make an unconnected version of bwlabels by warping out but with mask only for this type
                            bwlabels, diff, self.simpleLUT = binary_warping(bwlabels, np.ones(self.size,dtype=np.bool), 
                                mask=voxTypeSel[j], borderval=False, slow=True, simpleLUT=self.simpleLUT, 
                                connectivity=self.connectivity)

                        # run connected components on the thresholded labels merged with previous seeds (warped out)
                        uclabels, ucnlabels = nd.measurements.label(bwlabels, self.bwconn);
                        
                        # in this case the normal labels are the same as the unconnected labels because of warping
                        labels = uclabels; nlabels = ucnlabels;

                    # optionally make a skeletonized version of the unconnected labels
                    # xxx - revisit this, currently not being used for anything, started as a method to skeletonize GT
                    if self.skeletonize:
                        # method to skeletonize using max range endpoints only
                        sklabels, sknlabels = emLabels.ucskeletonize(uclabels, mask=voxTypeSel[j], 
                            sampling=self.attrs['scale'] if hasattr(self.attrs,'scale') else None)
                        assert( sknlabels == ucnlabels )
                    
                    # fill out these labels out so that they fill in remaining voxels based on voxType.
                    # this uses bwdist method for finding nearest neighbors, so connectivity can be violoated.
                    # this is mitigated by first filling out background using the warping transformation 
                    #   (or watershed) above, then this step is only to fill in remaining voxels for the 
                    #   current foreground voxType.
                    labels = emLabels.nearest_neighbor_fill(labels, mask=voxTypeNotSel[j], 
                        sampling=self.attrs['scale'] if hasattr(self.attrs,'scale') else None)

                    # save the components labels generated for this type
                    types_labels[j] = labels.astype(emLabels.LBLS_DTYPE, copy=False); 
                    types_uclabels[j] = uclabels.astype(emLabels.LBLS_DTYPE, copy=False); 
                    types_nlabels[j] = nlabels if self.fg_types_labels[j] < 0 else 1
                    types_ucnlabels[j] = ucnlabels if self.fg_types_labels[j] < 0 else 1
                    if self.skeletonize: types_sklabels[j] = sklabels.astype(emLabels.LBLS_DTYPE, copy=False)
                
                # merge the fg components labels. they can not overlap because voxel type is winner-take-all.
                nlabels = 0; ucnlabels = 0;
                labels = np.zeros(self.size, dtype=emLabels.LBLS_DTYPE); 
                uclabels = np.zeros(self.size, dtype=emLabels.LBLS_DTYPE); 
                if self.skeletonize: sklabels = np.zeros(self.size, dtype=emLabels.LBLS_DTYPE); 
                for j in range(self.nfg_types): 
                    sel = (types_labels[j] > 0); ucsel = (types_uclabels[j] > 0); 
                    if self.skeletonize: sksel = (types_sklabels[j] > 0);
                    if self.fg_types_labels[j] < 0:
                        labels[sel] += (types_labels[j][sel] + nlabels); 
                        uclabels[ucsel] += (types_uclabels[j][ucsel] + ucnlabels); 
                        if self.skeletonize: sklabels[sksel] += (types_sklabels[j][sksel] + ucnlabels); 
                        nlabels += types_nlabels[j]; ucnlabels += types_ucnlabels[j]; 
                    else:
                        labels[sel] = self.fg_types_labels[j]; 
                        uclabels[ucsel] = self.fg_types_labels[j]; 
                        if self.skeletonize: sklabels[sksel] = self.fg_types_labels[j] 
                        nlabels += 1; ucnlabels += 1;
            
                if self.dpWatershedTypes_verbose: 
                    print('\tdone in %.4f s' % (time.time() - t))

                # make a fully-filled out version using bwdist nearest foreground neighbor
                wlabels = emLabels.nearest_neighbor_fill(labels, mask=None, 
                    sampling=self.attrs['scale'] if hasattr(self.attrs,'scale') else None)

                # write out the results
                if self.nTmin == 1: subgroups = ['%.8f' % (self.Ts[i],)] 
                else: subgroups = ['%d' % (self.Tmins[k],), '%.8f' % (self.Ts[i],)]
                d = self.attrs.copy(); d['threshold'] = self.Ts[i]; 
                d['types_nlabels'] = types_nlabels; d['Tmin'] = self.Tmins[k]
                dpwrite = emLabels.writeLabels(outfile=self.outlabels, chunk=self.chunk.tolist(), 
                    offset=self.offset.tolist(), size=self.size.tolist(), datasize=self.datasize.tolist(), 
                    chunksize=self.chunksize.tolist(), data=labels, verbose=self.dpWatershedTypes_verbose, 
                    attrs=d, strbits=self.outlabelsbits, subgroups=['with_background']+subgroups )
                dpwrite = emLabels.writeLabels(outfile=self.outlabels, chunk=self.chunk.tolist(), 
                    offset=self.offset.tolist(), size=self.size.tolist(), datasize=self.datasize.tolist(), 
                    chunksize=self.chunksize.tolist(), data=wlabels, verbose=self.dpWatershedTypes_verbose, 
                    attrs=d, strbits=self.outlabelsbits, subgroups=['zero_background']+subgroups )
                d['type_nlabels'] = types_ucnlabels;
                dpwrite = emLabels.writeLabels(outfile=self.outlabels, chunk=self.chunk.tolist(), 
                    offset=self.offset.tolist(), size=self.size.tolist(), datasize=self.datasize.tolist(), 
                    chunksize=self.chunksize.tolist(), data=uclabels, verbose=self.dpWatershedTypes_verbose, 
                    attrs=d, strbits=self.outlabelsbits, subgroups=['no_adjacencies']+subgroups )
                if self.skeletonize: 
                    dpwrite = emLabels.writeLabels(outfile=self.outlabels, chunk=self.chunk.tolist(), 
                        offset=self.offset.tolist(), size=self.size.tolist(), datasize=self.datasize.tolist(), 
                        chunksize=self.chunksize.tolist(), data=sklabels, verbose=self.dpWatershedTypes_verbose, 
                        attrs=d, strbits=self.outlabelsbits, subgroups=['skeletonized']+subgroups )


    @staticmethod
    def addArgs(p):
        # adds arguments required for this object to specified ArgumentParser object
        p.add_argument('--probfile', nargs=1, type=str, default='.', help='Path/name of hdf5 probability (input) file')
        p.add_argument('--srclabels', nargs=1, type=str, default='', 
            help='Optional input that uses labels of types as probability inputs (instead of probfile)')
        p.add_argument('--TminSrc', nargs=1, type=int, default=[1], 
            help='Minimum component size for "cleaning" srclabels (unused without srclabels)')
        p.add_argument('--fg-types', nargs='+', type=str, default=['ICS','ECS'], 
            metavar='TYPE', help='Dataset names of the foreground voxel types in the hdf5')
        p.add_argument('--fg-types-labels', nargs='+', type=int, default=[], 
            metavar='LBL', help='Single label value to use for corresponding type')
        p.add_argument('--bg-type', nargs=1, type=str, default='MEM', 
            help='Dataset name of the background voxel type in the hdf5')
        p.add_argument('--chunk', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
            help='Corner chunk to parse out of hdf5')
        p.add_argument('--offset', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
            help='Offset in chunk to read')
        p.add_argument('--size', nargs=3, type=int, default=[256,256,128], metavar=('X', 'Y', 'Z'),
            help='Size in voxels to read')
        p.add_argument('--ThrRng', nargs=3, type=float, default=[0.95,0.999,0.01], metavar=('BEG', 'END', 'STP'),
            help='Python range (start, stop] by linear step for probability thresholds')
        p.add_argument('--ThrHi', nargs='*', type=float, default=[0.995, 0.999, 0.9995, 0.9999], 
            help='Extra thresholds for probs on high end')
        p.add_argument('--ThrLo', nargs='*', type=float, default=[], help='Extra thresholds for probs on low end')
        p.add_argument('--Tmins', nargs='+', type=int, default=[256], 
            help='Minimum component size threshold list (for "peak detection")')
        p.add_argument('--outlabels', nargs=1, type=str, default='', metavar='FILE', help='Supervoxels h5 output file')
        p.add_argument('--outlabelsbits', nargs=1, type=str, default=['32'], metavar=('BITS'),
            help='Number of bits for labels (always uint type)')
        #p.add_argument('--skeletonize', action='store_true', help='Create skeletonized version of labels')
        p.add_argument('--probWatershed', action='store_true', 
            help='Run actual watershed on probabilities to fill out labels')
        p.add_argument('--skimWatershed', action='store_true', 
            help='Run scikit-image watershed on probabilities to fill out labels')
        p.add_argument('--connectivity', nargs=1, type=int, default=[1], choices=[1,2,3],
            help='Connectivity for connected components (and watershed)')
        p.add_argument('--dpWatershedTypes-verbose', action='store_true', 
            help='Debugging output for dpWatershedTypes')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read EM voxel type probability data from h5 and create supervoxels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpWatershedTypes.addArgs(parser)
    args = parser.parse_args()
    
    ws = dpWatershedTypes(args)
    ws.watershed_cube()

