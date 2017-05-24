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

# Extends python hdf5 load / write class for specific datatypes.
# Major initial motivation for this was to have single location where label and probability data types are defined.
# Grew into extended classes with classfile methods for hdf5 data of specific types. These currently include:
#     (1) voxelType - uint8 representing the category for each voxel (BG, ICS, ECS, etc)
#     (2) label - unsigned integer containing ICS/ECS supervoxel labels
#     (3) probabilities - single floats containing network output probabilities (voxel type, affinity, etc)

import numpy as np
from scipy import ndimage as nd
import argparse
#import time
import networkx as nx
from utils import optimal_color
#import sys

from dpLoadh5 import dpLoadh5
from dpWriteh5 import dpWriteh5

class emVoxelType(dpWriteh5):
    VOXTYPE_DTYPE = np.uint8
    VOXTYPE_STR_DTYPE = 'uint8'
    VOXTYPE_DATASET = 'voxel_type'
    EMPTY_VOXTYPE = np.iinfo(VOXTYPE_DTYPE).max

    # dictionary lookups for voxel types
    # xxx - currently nothing is maintaining consistency between emVoxelType.TYPES define and
    #   type order from watershed. the order for types in watershed must be same as these defines.
    TYPES = {'BG':0, 'ICS':1, 'ECS':2}

    # maps special classes string to where it is assigned (increment on top of max ICS label)
    TYPES_LBL_INCR = {'ECS':1}

    def __init__(self, args):
        self.default_data_type = self.VOXTYPE_DTYPE
        #self.data_type = self.VOXTYPE_DTYPE
        self.dataset = self.VOXTYPE_DATASET
        self.fillvalue = self.EMPTY_VOXTYPE
        dpWriteh5.__init__(self,args)

    @classmethod
    def readVoxType(cls, srcfile, chunk, offset, size, verbose=False):
        parser = argparse.ArgumentParser(description='class:emVoxelType',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        dpWriteh5.addArgs(parser); arg_str = ''
        arg_str += ' --srcfile ' + srcfile
        arg_str += ' --chunk %d %d %d ' % tuple(chunk)
        arg_str += ' --offset %d %d %d ' % tuple(offset)
        arg_str += ' --size %d %d %d ' % tuple(size)
        if verbose: arg_str += ' --dpLoadh5-verbose '
        if verbose: print(arg_str)
        args = parser.parse_args(arg_str.split())
        loadh5 = cls(args); loadh5.readCubeToBuffers()
        return loadh5

    @classmethod
    def writeVoxType(cls, outfile, chunk, offset, size, datasize, chunksize, fillvalue=None, data=None, inraw='',
            outraw='', attrs={}, subgroups_out=[], verbose=False):
        assert( data is not None or inraw )
        parser = argparse.ArgumentParser(description='class:emVoxelType',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        dpWriteh5.addArgs(parser); arg_str = ''
        arg_str += ' --srcfile ' + outfile
        arg_str += ' --chunk %d %d %d ' % tuple(chunk)
        arg_str += ' --offset %d %d %d ' % tuple(offset)
        arg_str += ' --size %d %d %d ' % tuple(size)
        arg_str += ' --chunksize %d %d %d' % tuple(chunksize)
        arg_str += ' --datasize %d %d %d' % tuple(datasize)
        if fillvalue: arg_str += ' --fillvalue ' + str(fillvalue)
        if inraw: arg_str += ' --inraw ' + inraw
        if outraw: arg_str += ' --outraw ' + outraw
        if subgroups_out: arg_str += ' --subgroups-out ' + ' '.join(subgroups_out)
        if verbose: arg_str += ' --dpWriteh5-verbose '
        if verbose: print(arg_str)
        args = parser.parse_args(arg_str.split())
        writeh5 = cls(args); writeh5.data_attrs = attrs
        if inraw: writeh5.writeFromRaw()
        else: writeh5.writeCube(data)
        return writeh5

    @staticmethod
    def getVoxTypeCounts(supervoxel_type, dic):
        for k,v in emVoxelType.TYPES.items():
            dic['n' + k] = (supervoxel_type==v).sum(dtype=np.int64)


class emLabels(dpWriteh5):
    LBLS_DTYPE = np.uint32
    LBLS_STR_DTYPE = 'uint32'
    LBLS_DATASET = 'labels'
    EMPTY_LABEL = np.iinfo(LBLS_DTYPE).max

    def __init__(self, args):
        self.dataset = self.LBLS_DATASET
        self.default_data_type = self.LBLS_DTYPE   # added to constructors
        dpWriteh5.__init__(self,args)
        # reinitialize these for non-default data-type
        self.EMPTY_LABEL = np.iinfo(self.data_type).max
        self.fillvalue = self.EMPTY_LABEL

    @classmethod
    def readLabels(cls, srcfile, chunk, offset, size, data_type=None, subgroups=[], verbose=False):
        if not data_type: data_type = cls.LBLS_STR_DTYPE
        parser = argparse.ArgumentParser(description='class:emLabels',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        dpWriteh5.addArgs(parser); arg_str = ''
        arg_str += ' --srcfile ' + srcfile
        arg_str += ' --chunk %d %d %d ' % tuple(chunk)
        arg_str += ' --offset %d %d %d ' % tuple(offset)
        arg_str += ' --size %d %d %d ' % tuple(size)
        if data_type: arg_str += ' --data-type ' + data_type
        if subgroups: arg_str += ' --subgroups ' + ' '.join(subgroups)
        if verbose: arg_str += ' --dpLoadh5-verbose '
        if verbose: print(arg_str)
        args = parser.parse_args(arg_str.split())
        loadh5 = cls(args); loadh5.readCubeToBuffers()
        return loadh5

    @classmethod
    def writeLabels(cls, outfile, chunk, offset, size, datasize, chunksize, fillvalue=None, data=None, inraw='',
            strbits='32', outraw='', attrs={}, subgroups=[], verbose=False):
        assert( data is not None or inraw )
        parser = argparse.ArgumentParser(description='class:emProbabilities',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        dpWriteh5.addArgs(parser); arg_str = ''
        arg_str += ' --srcfile ' + outfile
        arg_str += ' --chunk %d %d %d ' % tuple(chunk)
        arg_str += ' --offset %d %d %d ' % tuple(offset)
        arg_str += ' --size %d %d %d ' % tuple(size)
        arg_str += ' --chunksize %d %d %d' % tuple(chunksize)
        arg_str += ' --datasize %d %d %d' % tuple(datasize)
        arg_str += ' --data-type %s ' % ('uint' + strbits)
        if subgroups: arg_str += ' --subgroups ' + ' '.join(subgroups)
        if fillvalue: arg_str += ' --fillvalue ' + str(fillvalue)
        if inraw: arg_str += ' --inraw ' + inraw
        if outraw: arg_str += ' --outraw ' + outraw
        #if verbose: arg_str += ' --dpWriteh5-verbose --dpLoadh5-verbose '
        if verbose: arg_str += ' --dpWriteh5-verbose '
        if verbose: print(arg_str)
        args = parser.parse_args(arg_str.split())
        writeh5 = cls(args); writeh5.data_attrs = attrs
        if inraw: writeh5.writeFromRaw()
        else: writeh5.writeCube(data)
        return writeh5

    # label manipulation routines
    # xxx - not a great reason that these were written as static methdods, maybe make as normal methods?
    #   would either modify labels in place or return a modified set of labels.

    # relabel sequential using thresholdSizes, just for a more clear name and if mappings are not needed
    @staticmethod
    def relabel_sequential(lbls, return_mapping=False):
        return emLabels.thresholdSizes(lbls, minSize=1, return_mapping=return_mapping)

    # calculate sizes for a label volume and remove labels below specified threshold, returns sizes for labels only
    @staticmethod
    def thresholdSizes(lbls, minSize=1, return_mapping=False):
        # xxx - all these methods should do something like this, would be better to move to normal class member methods
        assert( lbls.dtype.kind in 'ui' )

        sizes = emLabels.getSizes(lbls); sizes = sizes[1:];
        # negative minSize means only keep labels < minSize, positive means only keep labels >= minSize
        if minSize < 0: bgsel = (sizes >= -minSize)
        else: bgsel = (sizes < minSize)
        fgsel = np.logical_not(bgsel); sizes = sizes[fgsel]; fgcomps = np.cumsum(fgsel,dtype=emLabels.LBLS_DTYPE);
        fgcomps[bgsel] = 0; fgcomps = np.insert(fgcomps,0,0); L = fgcomps[lbls.flatten()].reshape(lbls.shape)
        if return_mapping:
            # return mapping from new supervoxels to old ones
            mapping = np.transpose(np.nonzero(fgsel)).reshape((-1,))
            return L, sizes, mapping
        else:
            return L, sizes

    # returns sizes for bg (first element) and for labels
    @staticmethod
    def getSizes(lbls, maxlbls=None):
        assert( lbls.dtype.kind in 'ui' )
        #assert( ~(lbls == np.iinfo(lbls.dtype).max).any() ) # HIASSERT
        #if maxlbls is None:
        #    maxlbls = lbls.max()
        # xxx - could not find an efficient method that uses maxlbls, would have to write C-function
        sizes = np.bincount(np.ravel(lbls)) # better
        ##sizes = nd.measurements.histogram(lbls, 0, maxlbls, maxlbls+1) # same as hist
        ##sizes,edges = np.histogram(lbls, bins=range(0,maxlbls+2), range=(0,maxlbls+1)) # slow
        ##sizes = nd.labeled_comprehension(1, lbls, np.arange(0,maxlbls+1,dtype=np.int64), np.sum, np.int64, 0) # worst
        return sizes

    # get type of each supervoxel by majority vote by summing votes per supervoxel
    @staticmethod
    def type_components(labels, voxel_type, nlabels, ntypes):
        from pyCext import type_components
        supervoxel_type = np.zeros((nlabels,), dtype=emVoxelType.VOXTYPE_DTYPE)
        voxel_out_type = np.zeros_like(voxel_type, dtype=emVoxelType.VOXTYPE_DTYPE)
        type_components(labels, voxel_type, supervoxel_type, voxel_out_type, ntypes)
        return supervoxel_type, voxel_out_type

    # remove connected adjacencies between labels, arbitrary connectivity
    @staticmethod
    def remove_adjacencies_nconn(labels, connectivity=1, bwconn=None):
        from pyCext import remove_adjacencies
        if bwconn is None:
            bwconn = nd.morphology.generate_binary_structure(dpLoadh5.ND, connectivity)
        return remove_adjacencies(labels, bwconn)

    # remove connected adjacencies between labels, 6-connectivity
    @staticmethod
    def remove_adjacencies_6conn(labels):
        bgmask = (labels > 0)    # to mask out labels that border on background to prevent erosion
        adjmask = np.zeros(labels.shape, dtype=np.bool)

        d = (np.diff(labels,axis=0) != 0)
        d = np.logical_and(np.logical_and(bgmask[:-1,:,:], d), np.logical_and(bgmask[1:,:,:], d))
        dc = np.zeros(labels.shape, dtype=np.bool); dc[:-1,:,:] = d; dc[1:,:,:] = np.logical_or(dc[1:,:,:], d)
        adjmask = np.logical_or(adjmask, dc); del d, dc

        d = (np.diff(labels,axis=1) != 0)
        d = np.logical_and(np.logical_and(bgmask[:,:-1,:], d), np.logical_and(bgmask[:,1:,:], d))
        dc = np.zeros(labels.shape, dtype=np.bool); dc[:,:-1,:] = d; dc[:,1:,:] = np.logical_or(dc[:,1:,:], d)
        adjmask = np.logical_or(adjmask, dc); del d, dc

        if labels.shape[2] > 1:
            d = (np.diff(labels,axis=2) != 0)
            d = np.logical_and(np.logical_and(bgmask[:,:,:-1], d), np.logical_and(bgmask[:,:,1:], d))
            dc = np.zeros(labels.shape, dtype=np.bool); dc[:,:,:-1] = d; dc[:,:,1:] = np.logical_or(dc[:,:,1:], d)
            adjmask = np.logical_or(adjmask, dc); del d, dc

        # not guaranteed to be left with the same number of components, so rerun
        L = labels.copy(); L[adjmask] = 0; L, nlabels = nd.measurements.label(L)
        return L.astype(labels.dtype), nlabels

    # xxx - ucskeletonize needs updating, was intended for skeletonizing GT, leaving code for later to clean up.
    #   need to add support for different connectivities.
    '''
    @staticmethod
    def ucskeletonize(uclabels, mask=None, sampling=None, nShrinkEndPoints=0):
        if mask is None: mask = np.ones(uclabels.shape, dtype=np.bool)
        if sampling is None:
            sampling = np.ones((3,),dtype=np.double)
        else:
            if sampling.size==2: sampling = np.concatenate((sampling, 1))

        # find anchor points for each object that are at the max and min of the dimension with the biggest range.
        amask = mask.copy(); ucnlabels= uclabels.max()
        for lbl in range(1,ucnlabels):
            inds = np.transpose(np.nonzero(uclabels == lbl)); pts = inds*sampling
            # what would really be good here would be to run something like pca to get the directions of variance
            #   and use the end points along those vectors. xxx - implement this
            # OR, any superior method of finding endpoints at anchor points
            vmax = pts.max(axis=0); vmin = pts.min(axis=0); ptrng = vmax - vmin
            if uclabels.shape[2] == 1: dmax = ptrng[:2].argmax()
            else: dmax = ptrng.argmax()

            # take any point at ends of largest range
            p1 = inds[pts[:,dmax] == vmax[dmax],:]; p2 = inds[pts[:,dmax] == vmin[dmax],:];
            #print(p1[0,:],p2[0,:],amask[p1[0,0],p1[0,1],p1[0,2]])
            amask[p1[0,0],p1[0,1],p1[0,2]] = False; amask[p2[0,0],p2[0,1],p2[0,2]] = False;

        # now warp all the way down but keeping two anchor points along max range dim.
        # this method assumes that original labels were unconnected.
        # xxx - add connectivity to this function and send it to binary_warping for loading simpleLUT
        bwlabels, diff = binary_warping((uclabels > 0).copy(order='C'),
            np.zeros(uclabels.shape,dtype=np.bool), mask=amask, borderval=False, slow=True)
        if nShrinkEndPoints > 0:
            # now warp down a few more iterations to move anchor endpoints away from the object borders
            bwlabels, diff = binary_warping(bwlabels.copy(order='C'), np.zeros(uclabels.shape,dtype=np.bool),
                mask=mask, borderval=False, slow=True, numiters=nShrinkEndPoints)
        # run connected components to get label values
        return nd.measurements.label(bwlabels)

        keeping this here for reference, maybe implement as another method?
                        ## "node"-ize instead of skeletonize
                        ## warp all the way down to points for each object.
                        #skbwlabels, diff = binary_warping(bwlabels.copy(order='C'),
                        #    np.zeros(self.size,dtype=np.bool), mask=cVoxTypeSel, borderval=False, slow=True)
                        ## now warp back up a few iterations to make circles or spheres around points
                        #skbwlabels, diff = binary_warping(skbwlabels.copy(order='C'),
                        #    np.ones(self.size,dtype=np.bool), mask=cVoxTypeSel, borderval=False, slow=True, numiters=2)
                        ## run connected components to get label values
                        #sklabels = np.zeros(self.size, dtype=emLabels.LBLS_DTYPE); sknlabels = 0;
                        #sknlabels = label_components(skbwlabels.astype(emLabels.LBLS_DTYPE), sklabels, sknlabels+1)
    '''

    @staticmethod
    def nearest_neighbor_fill(labels, mask=None, sampling=None):
        # fill in background labels so they have the value of the nearest non-background label
        wlabels = labels.copy(); bwlabels = (labels == 0);
        if bwlabels.sum(dtype=np.uint64) > 0:
            # use exact euclidean distance transform, also allows for optional voxel scale
            inds = nd.distance_transform_edt(bwlabels, return_indices=True, return_distances=False, sampling=sampling)
            # do not fill in voxels specified in mask
            if mask is not None: bwlabels[mask] = 0
            # xxx - what does it mean when distance transform is returning negative values?
            if ((mask is None) or bwlabels.sum(dtype=np.uint64) > 0) and (inds >= 0).all():
                nearest = np.ravel_multi_index(inds, bwlabels.shape)[bwlabels];
                wlabels[bwlabels] = labels.flatten()[nearest]
        return wlabels

    # xxx - likely modify this to use the FRAG code, without features.
    @staticmethod
    def color(labels, cmap, sampling=None, graySize=9, chromatic=None):

        # make a fully-filled out version using bwdist nearest foreground neighbor
        wlabels = emLabels.nearest_neighbor_fill(labels, mask=None, sampling=sampling); nlabels = wlabels.max();

        # select the objects to color based on component sizes
        sizes_obj = emLabels.getSizes(labels)[1:]; sel_obj = (sizes_obj >= graySize);
        ind_obj = np.transpose(np.nonzero(sel_obj)).reshape((-1,)); ncolor = ind_obj.size

        # create a graph for the nearest neighbors of each object for larger objects
        neighbors = np.zeros((ncolor, ncolor), dtype=labels.dtype)
        # using full connectivity not guaranteed to work, 4-color theorm only for borders, no diagonals (UT to NM)
        #struct = nd.morphology.generate_binary_structure(labels.ndim, labels.ndim)
        for n in range(ncolor):
            #nbrlbls = np.unique(wlabels[nd.morphology.binary_dilation(wlabels==(ind_obj[n]+1),structure=struct)])
            nbrlbls = np.unique(wlabels[nd.morphology.binary_dilation(wlabels==(ind_obj[n]+1))])
            sel = np.zeros((nlabels,),dtype=np.bool); sel[nbrlbls-1] = 1; neighbors[n,sel[sel_obj]] = 1
        neighbors.flatten()[::ncolor] = 0   # remove diagonal

        G = nx.Graph(neighbors)

        # try greedy first using all strategies, then try optimal (slow) if chromatic is specified
        #   meaning that we want an exact number of colors for coloring the non-neighboring labels.
        strategies = [
            'strategy_largest_first','strategy_smallest_last','strategy_independent_set',
            'strategy_connected_sequential','strategy_connected_sequential_dfs',
            'strategy_connected_sequential_bfs','strategy_saturation_largest_first',
        ]
        interchange = np.ones((len(strategies),),dtype=np.bool); interchange[[2,6]] = 0; minclrs = ncolor
        for i in range(len(strategies)):
            tmp = nx.coloring.greedy_color(G, strategy=eval('nx.coloring.'+strategies[i]),
                interchange=bool(interchange[i]))
            nclrs = max(tmp.values())+1;
            if nclrs < minclrs: Gclr = tmp; minclrs = nclrs
            if chromatic is not None and minclrs <= chromatic: break
        if chromatic is not None and minclrs > chromatic:
            Gclr = optimal_color(G,chromatic); minclrs = chromatic

        # create the initial colormap for each object which is specified as the first color in the specified RGB cmap.
        # objects with size less than graySize will have this color.
        cmap_obj = cmap[0,:]*np.ones((nlabels,3),dtype=cmap.dtype)
        for n in range(ncolor): cmap_obj[ind_obj[n],:] = cmap[Gclr[n]+1,:]

        # add black for background
        return minclrs, np.concatenate((np.zeros((1,3),dtype=cmap.dtype),cmap_obj),axis=0)


class emProbabilities(dpWriteh5):
    PROBS_DTYPE = np.single
    PROBS_STR_DTYPE = 'float32'
    PROBS_DATASET = 'probabilities'
    #EMPTY_PROB = -1.0
    EMPTY_PROB = 0.0

    def __init__(self, args):
        #self.data_type = self.PROBS_DTYPE
        self.default_data_type = self.PROBS_DTYPE
        self.fillvalue = self.EMPTY_PROB
        #self.dataset = self.PROBS_DATASET  # don't do this, set in classmethod
        dpWriteh5.__init__(self,args)

    @classmethod
    def readProbs(cls, srcfile, probName, chunk, offset, size, verbose=False):
        parser = argparse.ArgumentParser(description='class:emProbabilities',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        dpWriteh5.addArgs(parser); arg_str = ''
        arg_str += ' --srcfile ' + srcfile
        arg_str += ' --chunk %d %d %d ' % tuple(chunk)
        arg_str += ' --offset %d %d %d ' % tuple(offset)
        arg_str += ' --size %d %d %d ' % tuple(size)
        arg_str += ' --dataset ' + emProbabilities.PROBS_DATASET + str(probName)
        if verbose: arg_str += ' --dpLoadh5-verbose '
        if verbose: print(arg_str)
        args = parser.parse_args(arg_str.split())
        loadh5 = cls(args); loadh5.readCubeToBuffers()
        return loadh5

    @classmethod
    def writeProbs(cls, outfile, probName, chunk, offset, size, datasize, chunksize, fillvalue=None, data=None,
            inraw='', outraw='', attrs={}, verbose=False):
        assert( data is not None or inraw )
        parser = argparse.ArgumentParser(description='class:emProbabilities',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        dpWriteh5.addArgs(parser); arg_str = ''
        arg_str += ' --srcfile ' + outfile
        arg_str += ' --chunk %d %d %d ' % tuple(chunk)
        arg_str += ' --offset %d %d %d ' % tuple(offset)
        arg_str += ' --size %d %d %d ' % tuple(size)
        arg_str += ' --dataset ' + emProbabilities.PROBS_DATASET + str(probName)
        arg_str += ' --chunksize %d %d %d' % tuple(chunksize)
        arg_str += ' --datasize %d %d %d' % tuple(datasize)
        if fillvalue: arg_str += ' --fillvalue ' + str(fillvalue)
        if inraw: arg_str += ' --inraw ' + inraw
        if outraw: arg_str += ' --outraw ' + outraw
        #if verbose: arg_str += ' --dpWriteh5-verbose --dpLoadh5-verbose '
        if verbose: arg_str += ' --dpWriteh5-verbose '
        if verbose: print(arg_str)
        args = parser.parse_args(arg_str.split())
        writeh5 = cls(args); writeh5.data_attrs = attrs
        if inraw: writeh5.writeFromRaw()
        else: writeh5.writeCube(data)
        return writeh5


