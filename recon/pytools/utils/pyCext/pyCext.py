# C extensions for python using numpy for processing EM data.
# Structure based on:
#      http://wiki.scipy.org/Cookbook/C_Extensions/NumPy_arrays
#
# Created pwatkins 14 Apr 2015

import _pyCext
import numpy as np
import os
#import sys
import snappy
import zipfile

# run connected components, but using affinities to assess connectedness instead of pixel connectivity.
# this function is only for a weighted undirected affinity graph in 3d.
def label_affinities(affinities, labels, nextlabel, threshold):
    test=np.zeros((2,2)); testI=np.zeros((2,2),dtype=np.uint32)
    if type(affinities) != type(test):
        raise Exception( 'In label_affinities, affinities is not *NumPy* array')
    if len(affinities.shape) != 4:
        raise Exception( 'In label_affinities, affinities shape not 4 dimensional')
    if affinities.shape[3] != 3:
        raise Exception( 'In label_affinities, affinities not 3 dimensional')
    if not affinities.flags.contiguous or np.isfortran(affinities):
        raise Exception( 'In label_affinities, affinities not C-order contiguous')
    if affinities.dtype != np.single:
        raise Exception( 'In label_affinities, affinities not single floats')
    if type(labels) != type(testI):
        raise Exception( 'In label_affinities, labels is not *NumPy* array')
    if len(labels.shape) != 3:
        raise Exception( 'In label_affinities, labels is not 3 dimensional')
    if not labels.flags.contiguous or np.isfortran(labels):
        raise Exception( 'In label_affinities, labels not C-order contiguous')
    if labels.dtype != np.uint32:
        raise Exception( 'In label_affinities, labels not uint32')
    if type(threshold) != type(1.0):
        raise Exception( 'In label_affinities, threshold argument is not a float')
    if type(nextlabel) != type(1):
        raise Exception( 'In label_affinities, nextlabel argument is not a integer')
    if nextlabel < 1:
        raise Exception( 'In label_components, nextlabel argument is less than one')
        
    return _pyCext.label_affinities(affinities,labels,nextlabel,threshold)

# perform warping from one binary image to another. only connectivity of 1,3 (6,26) supported
# xxx - document all these optional parameters
def binary_warping(source, target, mask=None, gray=None, grayThresholds=None, borderval=False, numiters=-1, slow=False, 
        simpleLUT=None, connectivity=1, return_nonSimple=False):
    sz =  [x+2 for x in source.shape]   # need border for neighborhoods around the edge voxels
    dtype = np.bool; pdtype = np.float32
    test=np.zeros((2,2),dtype=dtype)
    
    if type(source) != type(test):
        raise Exception( 'In binary_warping, source is not *NumPy* array')
    if len(source.shape) != 3:
        raise Exception( 'In binary_warping, source is not 3 dimensional')
    if not source.flags.contiguous or np.isfortran(source):
        raise Exception( 'In binary_warping, source not C-order contiguous')
    if source.dtype != dtype:
        raise Exception( 'In binary_warping, source not correct data type')

    if type(target) != type(test):
        raise Exception( 'In binary_warping, target is not *NumPy* array')
    if len(target.shape) != 3:
        raise Exception( 'In binary_warping, target is not 3 dimensional')
    if not target.flags.contiguous or np.isfortran(target):
        raise Exception( 'In binary_warping, target not C-order contiguous')
    if target.dtype != dtype:
        raise Exception( 'In binary_warping, target not correct data type')
    if not np.array_equal(target.shape, source.shape):
        raise Exception( 'In binary_warping, target not same shape as source')

    if mask is None:
        mask = np.ones(source.shape, dtype=dtype)
    else:
        if type(mask) != type(test):
            raise Exception( 'In binary_warping, mask is not *NumPy* array')
        if len(mask.shape) != 3:
            raise Exception( 'In binary_warping, mask is not 3 dimensional')
        if not mask.flags.contiguous or np.isfortran(mask):
            raise Exception( 'In binary_warping, mask not C-order contiguous')
        if mask.dtype != dtype:
            raise Exception( 'In binary_warping, mask not correct data type')
        if not np.array_equal(mask.shape, source.shape):
            raise Exception( 'In binary_warping, mask not same shape as source')
            
    if gray is None:
        # over-ride gray Thresholds if gray data is not provided
        gray = np.ones((0,), dtype=pdtype); grayThresholds = np.ones((0,), dtype=pdtype); gry = gray
    else:
        if type(gray) != type(test):
            raise Exception( 'In binary_warping, gray is not *NumPy* array')
        if len(gray.shape) != 3:
            raise Exception( 'In binary_warping, gray is not 3 dimensional')
        if not gray.flags.contiguous or np.isfortran(gray):
            raise Exception( 'In binary_warping, gray not C-order contiguous')
        if gray.dtype != pdtype:
            raise Exception( 'In binary_warping, gray not correct data type')
        if not np.array_equal(gray.shape, source.shape):
            raise Exception( 'In binary_warping, gray not same shape as source')
    
        # xxx - grayThresholds need to be in decreasing order, assume caller does this instead of doing it here
        if type(grayThresholds) != type(test):
            raise Exception( 'In binary_warping, grayThresholds is not *NumPy* array')
        if grayThresholds.ndim != 1:
            raise Exception( 'In binary_warping, grayThresholds is not array')
        if not grayThresholds.flags.contiguous:
            raise Exception( 'In binary_warping, grayThresholds not contiguous')
        if grayThresholds.dtype != pdtype:
            raise Exception( 'In binary_warping, grayThresholds not correct data type')

    if type(borderval) != type(False):
        raise Exception( 'In binary_warping, borderval argument is not a boolean')
    if type(slow) != type(False):
        raise Exception( 'In binary_warping, slow argument is not a boolean')
    if type(numiters) != type(1):
        raise Exception( 'In binary_warping, numiters argument is not an integer')
    if numiters < 1:
        numiters = np.iinfo(np.int64).max

    # can pass in a previously loaded LUT (speed optimization for frequent/iterative calls).
    # connectivity parameter is only used if LUT is not provided as argument.
    if simpleLUT is None:
        assert(connectivity in [1,3]);  # 18-connectivity is not supported (no LUT, see matlab generation code)
        #assert(connectivity in [1,2,3]);
        
        # load the LUT from the current directory depending on connectivity
        if connectivity==1:
            fn = 'simpleLUT3d_6connFG_26connBG.raw'
        elif connectivity==3:
            fn = 'simpleLUT3d_26connFG_6connBG.raw'
        pfn = os.path.join(os.path.dirname(os.path.realpath(__file__)),fn)

        # LUT file expected in 2pass (snappy/zip, .sz.zip) compressed format
        zf = zipfile.ZipFile(pfn + '.sz.zip', mode='r'); simpleLUT = zf.read(fn + '.sz'); zf.close()
        simpleLUT = np.fromstring(snappy.uncompress(simpleLUT), dtype=np.uint8)
        assert( simpleLUT.size == 2**27 )

    # add the border to the source and target and mask and gray scale data (if supplied)
    src = np.ones(sz, dtype=dtype) * borderval; src[1:-1,1:-1,1:-1] = source
    tgt = np.ones(sz, dtype=dtype) * borderval; tgt[1:-1,1:-1,1:-1] = target
    msk = np.zeros(sz, dtype=dtype); msk[1:-1,1:-1,1:-1] = mask
    if gray.size > 0: gry = np.zeros(sz, dtype=pdtype); gry[1:-1,1:-1,1:-1] = gray
    
    # optionally return the type of the remaining non-simple points
    nonSimple = np.zeros(sz, dtype=np.uint8) if return_nonSimple else np.zeros((0,), dtype=np.uint8)
    
    diff = _pyCext.binary_warping(src, tgt, msk, simpleLUT, gry, grayThresholds, nonSimple, numiters, slow)
    #assert( (nonSimple > 0).sum(dtype=np.int64) == diff )  # should be true, not worth the time
    
    # the source is now the original source after warping towards the target
    if return_nonSimple:
        # xxx - this should probably be stored along with the lookup table... 
        # xxx - could also replace these with an ordered dict
        nonSimpleTypes = {
            'dic' : {'SIMPLE':0,
                'CREATE_OBJECT':1, 'DELETE_CAVITY':2, 'RESULT_MERGE':3,
                'CREATE_CAVITY':5, 'DELETE_OBJECT':6, 'RESULT_SPLIT':7,
                'DELETE_TUNNEL':4, 'CREATE_TUNNEL':8},
            'arr' : ['SIMPLE',
                'CREATE_OBJECT', 'DELETE_CAVITY', 'RESULT_MERGE',
                'CREATE_CAVITY', 'DELETE_OBJECT', 'RESULT_SPLIT',
                'DELETE_TUNNEL', 'CREATE_TUNNEL']
            }
        return src[1:-1,1:-1,1:-1], nonSimple[1:-1,1:-1,1:-1], nonSimpleTypes, diff, simpleLUT
    else:
        return src[1:-1,1:-1,1:-1], diff, simpleLUT

def merge_supervoxels(consensus_labels, data_cube, merged, nlabels):
    test = np.zeros((2,2), dtype = np.uint32)

    if type(consensus_labels) != type(test):
        raise Exception('In merge_supervoxels, consensus_labels is not a *NumPy* array')
    if len(consensus_labels.shape) != 3:
        raise Exception('In merge_supervoxels, consensus_labels is not 3 dimensional')
    if not consensus_labels.flags.contiguous or np.isfortran(consensus_labels):
        raise Exception('In merge_supervoxels, consensus_labels not contiguous or not C-order')
    if consensus_labels.dtype != np.uint32:
        raise Exception('In merge_supervoxels, consensus_labels not uint32')

    if type(data_cube) != type(test):
        raise Exception('In merge_supervoxels, data_cube is not a *NumPy* array')
    if len(data_cube.shape) != 3:
        raise Exception('In merge_supervoxels, data_cube is not 3 dimensional')
    if not data_cube.flags.contiguous or np.isfortran(data_cube):
        raise Exception('In merge_supervoxels, data_cube not contiguous or not C-order')
    if data_cube.dtype != np.uint32:
        raise Exception('In merge_supervoxels, data_cube not uint32')
    if not np.array_equal(consensus_labels.shape, data_cube.shape):
        raise Exception( 'In merge_supervoxels, consensus_labels not same shape as data_cube')

    if type(merged) != type(test):
        raise Exception('In merge_supervoxels, merged is not a *NumPy* array')
    if len(merged.shape) != 1:
        raise Exception('In merge_supervoxels, merged is not 1 dimensional')
    if not merged.flags.contiguous:
        raise Exception('In merge_supervoxels, merged not contiguous')
    if merged.dtype != np.int64:
        raise Exception('In merge_supervoxels, merged not int64')

    if type(nlabels) != int:
        raise Exception('In merge_supervoxels, nlabels is not an int')
    
    _pyCext.merge_supervoxels(consensus_labels, data_cube, merged, nlabels)

# assign type for each supervoxel using majority vote of contained voxels, output to supervoxel_type
def type_components(labels, voxel_type, supervoxel_type, voxel_out_type, num_types=2):
    test=np.zeros((2,2),dtype=np.uint32)
    if type(labels) != type(test):
        raise Exception( 'In type_components, labels is not *NumPy* array')
    if len(labels.shape) != 3:
        raise Exception( 'In type_components, labels is not 3 dimensional')
    if not labels.flags.contiguous or np.isfortran(labels):
        raise Exception( 'In type_components, labels not C-order contiguous')
    if labels.dtype != np.uint32:
        raise Exception( 'In type_components, labels not uint32')
    if type(voxel_type) != type(test):
        raise Exception( 'In type_components, voxel_type is not *NumPy* array')
    if len(voxel_type.shape) != 3:
        raise Exception( 'In type_components, voxel_type is not 3 dimensional')
    if not voxel_type.flags.contiguous or np.isfortran(voxel_type):
        raise Exception( 'In type_components, voxel_type not C-order contiguous')
    if voxel_type.dtype != np.uint8:
        raise Exception( 'In type_components, voxel_type not uint8')
    if type(supervoxel_type) != type(test):
        raise Exception( 'In type_components, supervoxel_type is not *NumPy* array')
    if len(supervoxel_type.shape) != 1:
        raise Exception( 'In type_components, supervoxel_type is not array')
    if not supervoxel_type.flags.contiguous:
        raise Exception( 'In type_components, supervoxel_type is not contiguous')
    if supervoxel_type.dtype != np.uint8:
        raise Exception( 'In type_components, supervoxel_type not uint8')
    if not np.array_equal(labels.shape, voxel_type.shape):
        raise Exception( 'In type_components, labels and voxel_type not same shape')
    if type(voxel_out_type) != type(test):
        raise Exception( 'In type_components, voxel_out_type is not *NumPy* array')
    if len(voxel_out_type.shape) != 3:
        raise Exception( 'In type_components, voxel_out_type is not 3 dimensional')
    if not voxel_out_type.flags.contiguous or np.isfortran(voxel_out_type):
        raise Exception( 'In type_components, voxel_out_type not C-order contiguous')
    if voxel_out_type.dtype != np.uint8:
        raise Exception( 'In type_components, voxel_out_type not uint8')
    # max operation is slow, assumer caller did this correctly... segfault'able tho if wrong
    #if supervoxel_type.size != labels.max():
    #    raise Exception( 'In type_components, supervoxel_type size not equal to number of supervoxels in labels')
    if type(num_types) != type(1):
        raise Exception( 'In type_components, num_types argument is not an integer')
    if num_types < 2:
        raise Exception( 'In type_components, num_types < 2, need at least two types')
        
    return _pyCext.type_components(labels, voxel_type, supervoxel_type, voxel_out_type, num_types)



