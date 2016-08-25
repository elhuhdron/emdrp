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
import h5py
from scipy import ndimage as nd
from scipy import interpolate
from skimage import morphology as morph
import networkx as nx

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
        self.nwarps = dpLoadh5.ND-1
        assert( len(self.warp_datasets) == self.nwarps )
        self.docrop = (self.cropborder > 0).any()
        self.size_crop = self.size - 2*self.cropborder; self.offset_crop = self.offset + self.cropborder
        assert( not self.docrop or self.method == 'overlap' )  # currently cropping only supported for overlap method
        assert( not self.warpfile or self.method == 'overlap' ) # warps only used for overlap method

        # other input validations
        assert( (self.Ts > 0).all() and (self.Ts < 1).all() )
        assert( (self.Tmins > 1).all() )   # iterative cc's method needs a min threshold to "keep" small supervoxels
        assert( self.method=='skim-ws' or self.connectivity in [1,3] )  # warping does not support 18-conn (no LUT)

        # xxx - intended skeletonizatino for GT objects, needs updating
        self.skeletonize = False

        # print out all initialized variables in verbose mode
        if self.dpWatershedTypes_verbose: print('dpWatershedTypes, verbose mode:\n'); print(vars(self))

    def watershed_cube(self):
        writeVerbose = False;
        #writeVerbose = self.dpWatershedTypes_verbose
        readVerbose = False;
        #readVerbose = self.dpWatershedTypes_verbose

        # load the probability data, allocate as array of volumes instead of 4D ndarray to maintain C-order volumes
        probs = [None]*self.ntypes; bwseeds = [None]*self.nfg_types
        if self.srclabels:
            # this code path is typically not used in favor of the label checker for fully labeled 3d gt components.
            # but, some ground truth (for example, 2d ECS cases) was only labeled with voxel type,
            #   so this is used to create ground truth components from the voxel types.
            loadh5 = emLabels.readLabels(srcfile=self.srclabels, chunk=self.chunk.tolist(), offset=self.offset.tolist(),
                size=self.size.tolist(), data_type='uint16', verbose=writeVerbose)
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
            # check if background is in the prob file
            hdf = h5py.File(self.probfile,'r'); has_bg = self.bg_type in hdf; hdf.close()
            for i in range(0 if has_bg else 1, self.ntypes):
                loadh5 = dpLoadh5.readData(srcfile=self.probfile, dataset=self.types[i], chunk=self.chunk.tolist(),
                    offset=self.offset.tolist(), size=self.size.tolist(), data_type=emProbabilities.PROBS_STR_DTYPE,
                    verbose=readVerbose)
                self.datasize = loadh5.datasize; self.chunksize = loadh5.chunksize; self.attrs = loadh5.data_attrs
                probs[i] = loadh5.data_cube; del loadh5
            # if background was not in hdf5 then create it as 1-sum(fg type probs)
            if not has_bg:
                probs[0] = np.ones_like(probs[1])
                for i in range(1,self.ntypes): probs[0] -= probs[i]
                #assert( (probs[0] >= 0).all() ) # comment for speed
                probs[0][probs[0] < 0] = 0 # rectify

        # save some of the parameters as attributes
        self.attrs['types'] = self.types; self.attrs['fg_types'] = self.fg_types
        self.attrs['fg_types_labels'] = self.fg_types_labels

        # save connnetivity structure and warping LUT because used on each iteration (for speed)
        self.bwconn = nd.morphology.generate_binary_structure(dpLoadh5.ND, self.connectivity)
        self.bwconn2d = self.bwconn[:,:,1]; self.simpleLUT = None

        # load the warpings if warping mode is enabled
        warps = None
        if self.warpfile:
            warps = [None]*self.nwarps
            for i in range(self.nwarps):
                loadh5 = dpLoadh5.readData(srcfile=self.warpfile, dataset=self.warp_datasets[i],
                    chunk=self.chunk.tolist(), offset=self.offset.tolist(), size=self.size.tolist(),
                    verbose=readVerbose)
                warps[i] = loadh5.data_cube; del loadh5

        # xxx - may need to revisit cropping, only intended to be used with warping method.
        if self.docrop: c = self.cropborder; s = self.size  # DO NOT use variables c or s below

        # optionally apply filters in attempt to fill small background (membrane) probability gaps.
        if self.close_bg > 0:
            # create structuring element
            n = 2*self.close_bg + 1; h = self.close_bg; strel = np.zeros((n,n,n),dtype=np.bool); strel[h,h,h]=1;
            strel = nd.binary_dilation(strel,iterations=self.close_bg)

            # xxx - this was the only thing tried here that helped some but didn't work well against the skeletons
            probs[0] = nd.grey_closing( probs[0], structure=strel )
            for i in range(self.nfg_types): probs[i+1] = nd.grey_opening( probs[i+1], structure=strel )
            # xxx - this gave worse results
            #probs[0] = nd.maximum_filter( probs[0], footprint=strel )
            # xxx - this had almost no effect
            #probs[0] = nd.grey_closing( probs[0], structure=strel )

        # argmax produces the winner-take-all assignment for each supervoxel.
        # background type was put first, so voxType of zero is background (membrane).
        voxType = np.concatenate([x.reshape(x.shape + (1,)) for x in probs], axis=3).argmax(axis=3)
        # write out the winning type for each voxel
        # save some params from this watershed run in the attributes
        d = self.attrs.copy(); d['thresholds'] = self.Ts; d['Tmins'] = self.Tmins
        data = voxType.astype(emVoxelType.VOXTYPE_DTYPE)
        if self.docrop: data = data[c[0]:s[0]-c[0],c[1]:s[1]-c[1],c[2]:s[2]-c[2]]
        emVoxelType.writeVoxType(outfile=self.outlabels, chunk=self.chunk.tolist(),
            offset=self.offset_crop.tolist(), size=self.size_crop.tolist(), datasize=self.datasize.tolist(),
            chunksize=self.chunksize.tolist(), verbose=writeVerbose, attrs=d,
            data=data)

        # only allow a voxel to be included in the type of component that had max prob for that voxel.
        # do this by setting the non-winning probabilities to zero.
        for i in range(self.ntypes): probs[i][voxType != i] = 0;

        # create a type mask for each foreground type to select only current voxel type (winner-take-all from network)
        voxTypeSel = [None] * self.nfg_types; voxTypeNotSel =  [None] * self.nfg_types
        for i in range(self.nfg_types):
            voxTypeSel[i] = (voxType == i+1)
            # create an inverted version, only used for complete fill not for warping (which requires C-contiguous),
            #   so apply crop here if cropping enabled
            voxTypeNotSel[i] = np.logical_not(voxTypeSel[i])
            if self.docrop: voxTypeNotSel[i] = voxTypeNotSel[i][c[0]:s[0]-c[0],c[1]:s[1]-c[1],c[2]:s[2]-c[2]]

        # need C-contiguous probabilities for binary_warping.
        for i in range(self.nfg_types):
            if not probs[i+1].flags.contiguous or np.isfortran(probs[i+1]):
                probs[i+1] = np.ascontiguousarray(probs[i+1])

        # iteratively apply thresholds, each time only keeping components that have fallen under size Tmin.
        # at last iteration keep all remaining components.
        # do this separately for foreground types.
        for k in range(self.nTmin):
            for i in range(self.nfg_types): bwseeds[i] = np.zeros(self.size, dtype=np.bool, order='C')
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

                    # this if/elif switch determines the main method for creating the labels.
                    # xxx - make cropping to be done in more efficient way, particular to avoid filling cropped areas
                    if self.method == 'overlap':
                        # definite advantage to this method over other methods, but cost is about 2-3 times slower.
                        # labels are linked per zslice using precalculated slice to slice warpings based on the probs.
                        labels, nlabels = self.label_overlap(bwlabels, voxTypeSel[j], warps)

                        # xxx - add switches to only optionally export the unconnected labels
                        #uclabels = labels; ucnlabels = nlabels;

                        # crop right after the labels are created and stay uncropped from here.
                        # xxx - labels will be wrong unless method implicitly handled the cropping during the labeling.
                        #   currently only the warping method is doing, don't need cropping for other methods anyways.
                        if self.docrop: labels = labels[c[0]:s[0]-c[0],c[1]:s[1]-c[1],c[2]:s[2]-c[2]]

                        # this method can not create true unconnected 3d labels, but should be unconnected in 2d.
                        # NOTE: currently this only removes 6-connectivity, no matter what specified connecitity is
                        # xxx - some method of removing adjacencies with arbitrary connectivity?
                        uclabels, ucnlabels = emLabels.remove_adjacencies(labels)
                    elif self.method == 'skim-ws':
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
                        if self.method == 'comps-ws' and i>1:
                            # this is an alternative to the traditional watershed that warps out only based on stepping
                            #   back through the thresholds in reverse order. has advantages of non-connectivity.
                            # may help slightly for small supervoxels but did not show much improved metrics in
                            #   terms of large-scale connectivity (against skeletons)
                            # about 4-5 times slower than regular warping method.

                            # make an unconnected version of bwlabels by warping out but with mask only for this type
                            # everything above current threshold is already labeled, so only need to use gray thresholds
                            #    starting below the current threshold level.
                            bwlabels, diff, self.simpleLUT = binary_warping(bwlabels, np.ones(self.size,dtype=np.bool),
                                mask=voxTypeSel[j], borderval=False, slow=True, simpleLUT=self.simpleLUT,
                                connectivity=self.connectivity, gray=probs[j+1],
                                grayThresholds=self.Ts[i-1::-1].astype(np.float32, order='C'))
                        else:
                            assert( self.method == 'comps' )     # bad method option
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
                labels = np.zeros(self.size_crop, dtype=emLabels.LBLS_DTYPE);
                uclabels = np.zeros(self.size_crop, dtype=emLabels.LBLS_DTYPE);
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
                    print('\tnlabels = %d' % (nlabels,))
                    #print('\tnlabels = %d %d' % (nlabels,labels.max())) # for debug only
                    #assert(nlabels == labels.max()) # sanity check for non-overlapping voxTypeSel, comment for speed
                    print('\tdone in %.4f s' % (time.time() - t,))

                # make a fully-filled out version using bwdist nearest foreground neighbor
                wlabels = emLabels.nearest_neighbor_fill(labels, mask=None,
                    sampling=self.attrs['scale'] if hasattr(self.attrs,'scale') else None)

                # write out the results
                if self.nTmin == 1: subgroups = ['%.8f' % (self.Ts[i],)]
                else: subgroups = ['%d' % (self.Tmins[k],), '%.8f' % (self.Ts[i],)]
                d = self.attrs.copy(); d['threshold'] = self.Ts[i];
                d['types_nlabels'] = types_nlabels; d['Tmin'] = self.Tmins[k]
                emLabels.writeLabels(outfile=self.outlabels, chunk=self.chunk.tolist(),
                    offset=self.offset_crop.tolist(), size=self.size_crop.tolist(), datasize=self.datasize.tolist(),
                    chunksize=self.chunksize.tolist(), data=labels, verbose=writeVerbose,
                    attrs=d, strbits=self.outlabelsbits, subgroups=['with_background']+subgroups )
                emLabels.writeLabels(outfile=self.outlabels, chunk=self.chunk.tolist(),
                    offset=self.offset_crop.tolist(), size=self.size_crop.tolist(), datasize=self.datasize.tolist(),
                    chunksize=self.chunksize.tolist(), data=wlabels, verbose=writeVerbose,
                    attrs=d, strbits=self.outlabelsbits, subgroups=['zero_background']+subgroups )
                d['type_nlabels'] = types_ucnlabels;
                emLabels.writeLabels(outfile=self.outlabels, chunk=self.chunk.tolist(),
                    offset=self.offset_crop.tolist(), size=self.size_crop.tolist(), datasize=self.datasize.tolist(),
                    chunksize=self.chunksize.tolist(), data=uclabels, verbose=writeVerbose,
                    attrs=d, strbits=self.outlabelsbits, subgroups=['no_adjacencies']+subgroups )
                if self.skeletonize:
                    emLabels.writeLabels(outfile=self.outlabels, chunk=self.chunk.tolist(),
                        offset=self.offset_crop.tolist(), size=self.size_crop.tolist(), datasize=self.datasize.tolist(),
                        chunksize=self.chunksize.tolist(), data=sklabels, verbose=writeVerbose,
                        attrs=d, strbits=self.outlabelsbits, subgroups=['skeletonized']+subgroups )

    # This labeling method connects zslices layer-by-layer. This can be done by simply overlapping the eroded labeled
    #   regoins or by overlapping by using warped labels (with warps generated externally by some optic flow method).
    def label_overlap(self, bwlabels, mask, warps=None):
        # this method operates slice by slice
        zlabels = np.zeros(self.size, dtype=np.int64)
        nzlabels = 0; prv_labels = None; connections = [None]*(self.size[2]-1)
        s = self.size; s2 = [s[0], s[1], 1]; c = self.cropborder
        if warps:
            x = np.arange(s[0], dtype=warps[0].dtype); y = np.arange(s[1], dtype=warps[0].dtype)
            X = np.meshgrid(x,y, indexing='ij')
        for z in range(self.size[2]):
            # get bwlabels and mask for the current zslice
            cur_bwlabels = bwlabels[:,:,z]; cur_mask = mask[:,:,z]

            # make an unconnected version of bwlabels by warping out but with mask only for this type
            bw = cur_bwlabels[:,:,None].copy(order='C'); msk = cur_mask[:,:,None].copy(order='C')
            bw, diff, self.simpleLUT = binary_warping(bw, np.ones(s2,dtype=np.bool),
                mask=msk, borderval=False, slow=True, simpleLUT=self.simpleLUT, connectivity=self.connectivity)
            cur_fill_bwlabels = bw[:,:,0]

            # run connected components on the thresholded labels merged with previous seeds (warped out)
            cur_fill_labels, cur_nlabels = nd.measurements.label(cur_fill_bwlabels, self.bwconn2d, output=np.int64)

            # make labels for this zslice unique and add to whole label cube
            sel = (cur_fill_labels > 0); cur_fill_labels[sel] += nzlabels; zlabels[:,:,z] = cur_fill_labels

            # get eroded labels by applying mask for original bwlabels
            cur_labels = cur_fill_labels.copy(); cur_labels[np.logical_not(cur_bwlabels)] = 0

            # warp the previous slice to this slice and connect them
            if z > 0:
                if warps:
                    # apply warping from previous label slice to current slice using nearest neighbor interpolation.
                    cur_warpsx = warps[0][:,:,z-1]; cur_warpsy = warps[1][:,:,z-1]
                    xi = X[0] + cur_warpsx; yi = X[1] + cur_warpsy
                    # remove warps that are out of the size bounds
                    #xi[xi < 0] = 0; xi[xi > s[0]-1] = s[0]-1; yi[yi < 0] = 0; yi[yi > s[1]-1] = s[1]-1
                    f = interpolate.RegularGridInterpolator((x,y), prv_labels, method='nearest',
                        bounds_error=False, fill_value=0)
                    prv_labels = f( np.vstack((xi.ravel(),yi.ravel())).T ).reshape(prv_labels.shape)

                # map the previous warped labels to the current labels based on pixel-by-pixel overlap of eroded labels.
                # only used the xy cropped area to do the linkage.
                prv_labels_crop = prv_labels; cur_labels_crop = cur_labels
                if self.docrop:
                    prv_labels_crop = prv_labels_crop[c[0]:s[0]-c[0],c[1]:s[1]-c[1]]
                    cur_labels_crop = cur_labels_crop[c[0]:s[0]-c[0],c[1]:s[1]-c[1]]
                tmp = dpWatershedTypes.unique_rows(\
                    np.ascontiguousarray( np.vstack((prv_labels_crop.ravel(),cur_labels_crop.ravel())).T ))
                # remove background connections (any rows with zeros)
                connections[z-1] = tmp[(tmp>0).all(axis=1),:]

            # loop updates for linking current slice to next
            prv_labels = cur_labels; nzlabels += cur_nlabels

        # run graph connected components on graph created from pairwise connections.
        # this graph represents labels that have been linked by the warping between zslices.
        G = nx.Graph(); G.add_edges_from(np.vstack(connections))
        compsG = nx.connected_components(G); nlabels = 0; mapping = np.zeros((nzlabels+1,), dtype=np.int64)
        for nodes in compsG:
            # create mapping from current per-zslice labels to linked labels across zslices
            nlabels += 1; mapping[np.array(tuple(nodes),dtype=np.int64)] = nlabels

        # create the final labels using the mapping built from the graph connected components
        return mapping[zlabels], nlabels

    # http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    @staticmethod
    def unique_rows(a):
        assert( a.flags.contiguous and not np.isfortran(a) )
        return np.unique(a.view(np.dtype((np.void, a.dtype.itemsize*a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1])

    # xxx - add switches to only optionally export the full watershedded labels and the unconnected labels
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
        p.add_argument('--method', nargs=1, type=str, default='comps', choices=['comps','comps-ws', 'skim-ws',
            'overlap'], help='Method to use for generating supervoxels')
        #p.add_argument('--skeletonize', action='store_true', help='Create skeletonized version of labels')
        p.add_argument('--connectivity', nargs=1, type=int, default=[1], choices=[1,2,3],
            help='Connectivity for connected components (and watershed)')
        p.add_argument('--warpfile', nargs=1, type=str, default='',
            help='hdf5 containing warps for optional warping mode')
        p.add_argument('--warp-datasets', nargs=2, type=str, default=['warpx','warpy'],
            help='Datasets for x/y warpings')
        p.add_argument('--cropborder', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
           help='Optionally crop down outputs before writing')
        p.add_argument('--close-bg', nargs=1, type=int, default=[0], choices=range(5),
            help='Diamond radius of structuring element to try to fill in background (membrane) gaps')
        p.add_argument('--dpWatershedTypes-verbose', action='store_true',
            help='Debugging output for dpWatershedTypes')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read EM voxel type probability data from h5 and create supervoxels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpWatershedTypes.addArgs(parser)
    args = parser.parse_args()

    ws = dpWatershedTypes(args)
    ws.watershed_cube()

