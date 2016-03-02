
import os, sys
import argparse
import time
import numpy as np
import numpy.ma as ma
import tifffile

from dpLoadh5 import dpLoadh5
from typesh5 import emLabels, emProbabilities, emVoxelType

params = {
    'segpaths' : [
        '/home/watkinspv/Data/convnet_out/cube_recons/newestECS/sixfold_threed_warp', 
        '/home/watkinspv/Data/convnet_out/cube_recons/newestECS/sixfold_threed/out/forwarp',
        #'/home/watkinspv/Data/convnet_out/cube_recons/newestECS/sixfold_threed/out/20151006',
        ],
    'segmentations' : [
        'huge_supervoxels.h5', 
        'huge_supervoxels.h5', 
        ],
    'gth5' : '/Data/datasets/labels/gt/M0007_33_labels_briggmankl_watkinspv_39x35x7chunks_Forder.h5',
    'chunks' : [[17,19,2], [17,23,1], [22,23,1], [22,18,1], [22,23,2], [19,22,2]],
    'size' : [128, 128, 128], 'offset' : [0, 0, 0],
    'outpath' : '/home/watkinspv/Downloads/out2',
    'outprefix' : 'huge_warp_compare_',
    'gt_ECS_label' : 1,
    }

globals().update(params)
nchunks = len(chunks); nsegs = len(segmentations)

metrics = {
    'caterror' : ma.array(np.zeros((nsegs,nchunks),dtype=np.double),mask=True),
    }                
globals().update(metrics)

# distinguishable colors lut
dlut = np.fromfile('/usr/local/Fiji.app/luts/distinguish2.lut',dtype=np.uint8).reshape([3,-1]).T
# different four/five colors options, 90's couch used for ECS paper:
#({'e9' '6d' '63' '7f' 'ca' '9f' 'f4' 'ba' '70' '85' 'c1' 'f5'})/255,[3 4])'; % sth else
#({'39' 'ff' '14' 'f3' 'f3' '15' '00' 'be' 'ff' 'ec' '13' '41'})/255,[3 4])'; % neons
# https://kuler.adobe.com/Theme-26-color-theme-3895203/
#({'04' '68' 'bf' '14' 'a6' '70' 'f2' 'bc' '1b' 'f2' '29' '29'})/255,[3 4])'; % 90's couch
# 90's couch plus one
c4lut = np.array([int(x,16) for x in ['99','99','99', '04','68','bf', '14','a6','70', 'f2','bc','1b', 
    'f2','29','29', 'da','cc','ab']],dtype=np.uint8).reshape((-1,3))

for j,chunk in zip(range(nchunks),chunks):

    # load ground truth and components from segmented labels file
    loadh5 = emLabels.readLabels(srcfile=gth5, chunk=chunk, offset=offset, size=size)
    gtComps = loadh5.data_cube; gtIsECS = (gtComps == gt_ECS_label);
    gtLbls = np.zeros(gtComps.shape, dtype=np.uint8)
    gtLbls[np.logical_and(gtComps > 0, np.logical_not(gtIsECS))] = 1; gtLbls[gtIsECS] = 2;

    fn1 = os.path.join(segpaths[0], segmentations[0])
    loadh5 = emVoxelType.readVoxType(srcfile=fn1, chunk=chunk, offset=offset, size=size)
    outLbls1 = loadh5.data_cube; attrs = loadh5.data_attrs
    
    fn2 = os.path.join(segpaths[1], segmentations[1])
    loadh5 = emVoxelType.readVoxType(srcfile=fn2, chunk=chunk, offset=offset, size=size)
    outLbls2 = loadh5.data_cube; attrs = loadh5.data_attrs

    chunk_str = 'x%04d_y%04d_z%04d' % tuple(chunk)
    print('exporting images for ' + chunk_str); t = time.time()

    # annotated categorization error figure
    cat = gtLbls.copy()
    cat[np.logical_and(gtLbls != outLbls1,gtLbls != outLbls2)] = 3
    cat[np.logical_and(gtLbls != outLbls1,gtLbls == outLbls2)] = 4
    cat[np.logical_and(gtLbls == outLbls1,gtLbls != outLbls2)] = 5
    cmap = (np.array([[1,1,1], [0,0,0], [0.5,0.5,0.5], [1,1,0], [1,0,0], [0,1,0]], 
        dtype=np.single)*255).astype(np.uint8)
    tifffile.imsave(os.path.join(outpath,outprefix+chunk_str+'.tif'),cmap[cat.transpose(2,1,0),:],compress=5)
    #(gtLbls!=outLbls).transpose(2,1,0).astype(np.uint8).tofile(os.path.join(outpath,
    #    ofn+og+'_cat_error_uint8.raw'))
    caterror[0,j] = (gtLbls != outLbls1).sum(dtype=np.int64) / gtLbls.size
    caterror[1,j] = (gtLbls != outLbls2).sum(dtype=np.int64) / gtLbls.size

    print('\tdone in %.4f s' % (time.time() - t))

print(caterror)

