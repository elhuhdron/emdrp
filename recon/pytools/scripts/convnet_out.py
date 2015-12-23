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

# Top level script for generating supervoxels / running metrics.
# Adapted from top level script for revision of ECS paper. 
# 
# Ideally this top-level script eventually morphs into some top-level control for EMDRP...


import os, sys
import argparse
import time
import numpy as np
import numpy.ma as ma
import tifffile
import dill
from shutil import copyfile
from gala import evaluate as ev
from scipy import stats
from scipy.stats import mstats

from dpLoadh5 import dpLoadh5
from metrics import warping_error, adapted_rand_error
from metrics import adapted_rand_error_resample_objects_points, adapted_rand_error_resample_objects
from typesh5 import emLabels, emProbabilities, emVoxelType

'''
# 2d tiles input parameters
nblocks = 9; ngroups = 4;
params = {
    'groups' : ['none', 'small', 'large', 'huge'],
    'groups_vals' : [0.6, 6, 11, 24],
    'groups_xlim' : [-3,27.5],
    'groups_xlbl' : 'Mean Amount of ECS (%)', 
    'blocks' : range(1,nblocks+1),
    'inpath' : '/home/watkinspv/Data/convnet_out/cube_recons/newestECS/runs',
    #'loadpath' : '/home/watkinspv/Data/convnet_out/cube_recons/newestECS/runs/out/20150918',
    'loadpath' : '/home/watkinspv/Data/convnet_out/cube_recons/newestECS/runs/out/test',
    'outpath' : '/home/watkinspv/Data/convnet_out/cube_recons/newestECS/runs/out',
    'size' : [640, 576, 1], 'offset' : [0, 0, 8], 'lblbits' : 16, 'chunks' : [[[0, 0, 0]]*nblocks]*ngroups, 
    'thrRng' : [0.6, 0.999, 0.01], 
    'thrHi' : [0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 0.999995, 0.999999],
    'thrLo' : [0.3, 0.4, 0.5],
    'Tmins' : [64],
    #'connectivity' : 3,
    #'probWatershed' : True,
    #'skimWatershed' : True,
    'global_mins' : True,
    'skeletonize' : False,
    'input_data' : 'batch_input_data.h5',
    'output_data' : 'batch_output_data.h5',
    #'save_file' : 'process_convnet_out_newestECS.allresamps100p_1000.dill',
    'save_file' : 'process_convnet_out_newestECS.test.dill',
    'gt_name' : '_gt.h5',
    'out_name' : '_supervoxels.h5',
    'run_watershed' : False,
    'sel_watershed' : [],
    'run_gt_watershed' : False,
    'run_metrics' : False,
    #'nReSamples' : 1000,
    'nReSamples' : 0,
    'sel_metrics' : [],
    'make_plots' : True,
    'plot_setlims' : True,
    'save_plots' : False,
    'figno' : 1000,
    'export_images' : False,
    }
'''

'''
# 3d chunks input parameters
nblocks = 6; ngroups = 2;
params = {
    'groups' : ['none', 'huge'],
    'groups_vals' : [0.6, 24],
    'groups_xlim' : [-3,27.5],
    'groups_xlbl' : 'Mean Amount of ECS (%)', 
    'blocks' : range(1,nblocks+1),
    'inpath' : '/home/watkinspv/Data/convnet_out/cube_recons/newestECS/sixfold_threed',
    'loadpath' : '/home/watkinspv/Data/convnet_out/cube_recons/newestECS/sixfold_threed/out/test',
    'outpath' : '/home/watkinspv/Data/convnet_out/cube_recons/newestECS/sixfold_threed/out',
    'size' : [128, 128, 128], 'offset' : [0, 0, 0], 'lblbits' : 16, 
    'chunks' : [ 
        [[16,17,4], [18,15,3], [13,15,3], [13,20,3], [18,20,3], [18,20,4]],     # none ECS
        [[19,22,2], [17,19,2], [17,23,1], [22,23,1], [22,18,1], [22,23,2]],     # huge ECS
        ], 
    'thrRng' : [0.6, 0.999, 0.01], 
    'thrHi' : [0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 0.999995, 0.999999],
    'thrLo' : [0.3, 0.4, 0.5],
    #'Tmins' : [16, 32, 64, 128],
    'Tmins' : [64],
    'global_mins' : True,
    'group_single' : True,
    # merge all
    'merge_probs' : ['_xyz_0_probs.h5', '_xzy_0_probs.h5', '_zyx_0_probs.h5', '_xyz_1_probs.h5', '_xyz_2_probs.h5'], 
    'merge_weightings' : [1.0,0.5,0.5,1.0,1.0],    
    'merge_orderings' : ['xyz','xzy','zyx','xyz','xyz'],   
    'gt_labels' : [
        'M0027_11_labels_briggmankl_watkinspv_33x37x7chunks_Forder.h5',
        'M0007_33_labels_briggmankl_watkinspv_39x35x7chunks_Forder.h5',
        ],
    'gt_ECS_label' : 1,
    'out_name_probs' : '_probs.h5',
    'out_name' : '_supervoxels.h5',
    'save_file' : 'process_convnet_out_newestECS.test.dill',
    'run_merge' : False,
    'run_watershed' : False,
    'sel_watershed' : [],
    'export_raw' : False,
    'raw_data' : [
        '/Data/big_datasets/M0027_11_33x37x7chunks_Forder.h5',
        '/Data/big_datasets/M0007_33_39x35x7chunks_Forder.h5',
        ],
    'run_metrics' : False,
    'sel_metrics' : [],
    'make_plots' : True,
    'plot_setlims' : True,
    'save_plots' : False,
    'figno' : 1000,
    'export_images' : False,
    }
'''


'''
# supervoxels over whole area
nblocks = 1; ngroups = 2;
params = {
    'groups' : ['none', 'huge'],
    'groups_vals' : [0.6, 24],
    'groups_xlim' : [-3,27.5],
    'groups_xlbl' : 'Mean Amount of ECS (%)', 
    'blocks' : range(1,nblocks+1),
    #'inpath' : '/home/watkinspv/Data/convnet_out/cube_recons/newestECS/sixfold_threed',
    'inpath' : '/Data/pwatkins/full_datasets/newestECSall',
    'loadpath' : '/home/watkinspv/Data/convnet_out/cube_recons/newestECS/sixfold_threed/out',
    'outpath' : '/Data/pwatkins/full_datasets/newestECSall',
    'size' : [1024, 1024, 480], 'offset' : [0, 0, 32], 'lblbits' : 32, 
    #'size' : [64, 64, 64], 'offset' : [0, 0, 32], 'lblbits' : 32, # for test
    #'chunks' : [ 
    #    [[16,17,4], [18,15,3], [13,15,3], [13,20,3], [18,20,3], [18,20,4]],     # none ECS
    #    [[19,22,2], [17,19,2], [17,23,1], [22,23,1], [22,18,1], [22,23,2]],     # huge ECS
    #    ], 
    'chunks' : [ 
        [[12,14,2], ],     # corner none ECS
        [[16,17,0], ],     # corner huge ECS
        ], 
    #'thrRng' : [0.95, 0.999, 0.01], 
    #'thrHi' : [0.995, 0.999, 0.9995, 0.9999, 0.99995],
    'thrRng' : [0.3, 0.999, 0.1], 
    'thrHi' : [0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 0.999995, 0.999999],
    'thrLo' : [],
    #'Tmins' : [8, 16, 32, 64, 128, 256],
    'Tmins' : [256],
    'group_single' : True,
    # merge all
    'merge_probs' : ['_all_xyz_0_probs.h5', '_all_xzy_0_probs.h5', '_all_zyx_0_probs.h5', '_all_xyz_1_probs.h5', 
        '_all_xyz_2_probs.h5'], 
    'merge_weightings' : [1.0,0.5,0.5,1.0,1.0],    
    'merge_orderings' : ['xyz','xzy','zyx','xyz','xyz'],   
    'gt_labels' : [
        'M0027_11_labels_briggmankl_watkinspv_33x37x7chunks_Forder.h5',
        'M0007_33_labels_briggmankl_watkinspv_39x35x7chunks_Forder.h5',
        ],
    'gt_ECS_label' : 1,
    'out_name_probs' : '_probs.h5',
    'out_name' : '_supervoxels.h5',
    'save_file' : 'process_convnet_out_newestECS.dill',
    'run_merge' : False,
    'run_watershed' : True,
    'sel_watershed' : [],
    'export_raw' : False,
    'raw_data' : [
        '/Data/big_datasets/M0027_11_33x37x7chunks_Forder.h5',
        '/Data/big_datasets/M0007_33_39x35x7chunks_Forder.h5',
        ],
    'run_metrics' : False,
    'sel_metrics' : [],
    'make_plots' : False,
    'plot_setlims' : False,
    'save_plots' : False,
    'figno' : 1000,
    'export_images' : False,
    }
'''

'''
# 3d chunks input parameters, compare merging probs with multiple xyz versus ortho dirs
nblocks = 6; ngroups = 3;
params = {
    'groups' : ['none', 'none', 'none'],
    'groups_vals' : [2,4,6],
    'groups_xlim' : [-2,8],
    'groups_xlbl' : 'prob merge groups', 
    'blocks' : range(1,nblocks+1),
    'inpath' : '/home/watkinspv/Data/convnet_out/cube_recons/newestECS/sixfold_threed',
    'loadpath' : '/home/watkinspv/Data/convnet_out/cube_recons/newestECS/sixfold_threed/out/20150817_mergecompare',
    'outpath' : '/home/watkinspv/Data/convnet_out/cube_recons/newestECS/sixfold_threed/out',
    'size' : [128, 128, 128], 'offset' : [0, 0, 0], 'lblbits' : 16, 
    'chunks' : [ [[16,17,4], [18,15,3], [13,15,3], [13,20,3], [18,20,3], [18,20,4]],    # none ECS
        [[16,17,4], [18,15,3], [13,15,3], [13,20,3], [18,20,3], [18,20,4]],    # none ECS        
        [[16,17,4], [18,15,3], [13,15,3], [13,20,3], [18,20,3], [18,20,4]],    # none ECS        
        ], 
    'thrRng' : [0.5, 1.0, 0.01], 
    'thrHi' : [0.995, 0.999, 0.9995, 0.9999],
    'thrLo' : [],
    'group_single' : True,
    'merge_probs' : [
        ['_xyz_0_probs.h5', '_xzy_0_probs.h5', '_zyx_0_probs.h5', '_xyz_1_probs.h5', '_xyz_2_probs.h5'], # merge all
        ['_xyz_0_probs.h5', '_xzy_0_probs.h5', '_zyx_0_probs.h5'],  # merge orthos
        ['_xyz_0_probs.h5', '_xyz_1_probs.h5', '_xyz_2_probs.h5'],  # merge xy's only
        ],
    'merge_weightings' : [
        [1.0,0.25,0.25,1.0,1.0],    # merge all
        #[1.0,0.25,0.25],  # merge orthos, xxx - ran it this way the first time by accident
        [1.0,0.25,0.25],  # merge orthos
        [1.0,1.0,1.0],  # merge xy's only
        ],
    'merge_orderings' : [
        ['xyz','xzy','zyx','xyz','xyz'],    # merge all
        ['xyz','xzy','zyx'],    # merge orthos
        ['xyz','xyz','xyz'],    # merge xy's only
        ],
    'gt_labels' : ['M0027_11_labels_briggmankl_watkinspv_33x37x7chunks_Forder.h5',
        'M0027_11_labels_briggmankl_watkinspv_33x37x7chunks_Forder.h5',
        'M0027_11_labels_briggmankl_watkinspv_33x37x7chunks_Forder.h5'],
    'gt_ECS_label' : 1,
    'out_name_probs' : ['_all5_probs.h5', '_ortho3_probs.h5', '_xyz3_probs.h5'],
    'out_name' : ['_all5_supervoxels.h5', '_ortho3_supervoxels.h5', '_xyz3_supervoxels.h5',],
    'out_group' : ['_all5', '_ortho3', '_xyz3'],
    'save_file' : 'process_convnet_out_newestECS_mergecompare.dill',
    'run_merge' : False,
    'run_watershed' : False,
    'sel_watershed' : [],
    'run_metrics' : False,
    'sel_metrics' : [],
    'make_plots' : False,
    'plot_setlims' : False,
    'save_plots' : False,
    'figno' : 1000,
    'export_images' : True,
    }
'''

#'''
# 3d compare gala
nblocks = 1; ngroups = 3;
params = {
    'groups' : ['huge', 'huge_gala', 'huge_flatagglo_lda_9f'],
    'groups_vals' : [5, 10, 15],
    #'groups_vals' : [5, 10],
    'groups_xlim' : [0,20],
    'groups_xlbl' : 'groups', 
    'blocks' : range(1,nblocks+1),
    'inpath' : '/home/watkinspv/Data/convnet_out/cube_recons/newestECS/sixfold_threed',
    'loadpath' : '/Data/pwatkins/full_datasets/newestECSall/20151001',
    'outpath' : '/Data/pwatkins/full_datasets/newestECSall/20151001',
    'size' : [128, 128, 128], 'offset' : [0, 0, 0], 'lblbitsgt' : 16, 'lblbits' : 32,  
    #'size' : [64,64,64], 'offset' : [32,32,32], 'lblbitsgt' : 16, 'lblbits' : 32, 
    #'chunks' : [ 
    #    [[17,19,2], [17,23,1], [22,23,1], [22,18,1], [22,23,2], [19,22,2]],     
    #    [[17,19,2], [17,23,1], [22,23,1], [22,18,1], [22,23,2], [19,22,2]],     
    #    [[17,19,2], [17,23,1], [22,23,1], [22,18,1], [22,23,2], [19,22,2]],     
    #    ], 
    'chunks' : [ 
        [[17,19,2]],     # super
        [[17,19,2]],     # lda 2f
        [[17,19,2]],     # rf 2f
        ], 
    'thrRng' : [0.3, 0.999, 0.1], 
    'thrHi' : [0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 0.999995, 0.999999],
    'thrLo' : [],
    #'Tmins' : [64],
    'Tmins' : [256],
    'global_mins' : True,
    'group_single' : True,
    # merge all
    'merge_probs' : ['_xyz_0_probs.h5', '_xzy_0_probs.h5', '_zyx_0_probs.h5', '_xyz_1_probs.h5', '_xyz_2_probs.h5'], 
    'merge_weightings' : [1.0,0.5,0.5,1.0,1.0],    
    'merge_orderings' : ['xyz','xzy','zyx','xyz','xyz'],   
    'gt_labels' : [
        'M0007_33_labels_briggmankl_watkinspv_39x35x7chunks_Forder.h5',
        'M0007_33_labels_briggmankl_watkinspv_39x35x7chunks_Forder.h5',
        'M0007_33_labels_briggmankl_watkinspv_39x35x7chunks_Forder.h5',
        ],
    'gt_ECS_label' : 1,
    'out_name_probs' : '_probs.h5',
    'out_name' : '_supervoxels.h5',
    'save_file' : 'process_convnet_out_newestECS_gala.dill',
    'run_merge' : False,
    'run_watershed' : False,
    'sel_watershed' : [],
    'export_raw' : False,
    'raw_data' : [
        '/Data/big_datasets/M0007_33_39x35x7chunks_Forder.h5',
        '/Data/big_datasets/M0007_33_39x35x7chunks_Forder.h5',
        '/Data/big_datasets/M0007_33_39x35x7chunks_Forder.h5',
        ],
    'run_metrics' : True,
    'sel_metrics' : [],
    'make_plots' : True,
    'plot_setlims' : False,
    'save_plots' : False,
    'figno' : 1000,
    'export_images' : False,
    }
#'''




# inits based on input parameters
globals().update(params)
assert( ngroups == len(groups) ); assert( nblocks == len(blocks)); ntotal = ngroups*nblocks;
thrHiStr = ' '.join(['%.8f' % (x,) for x in thrHi]) + ' '
thrLoStr = ' '.join(['%.8f' % (x,) for x in thrLo]) + ' '
lbldtype = 'uint%d' % lblbits
if 'lblbitsgt' not in globals(): lblbitsgt = lblbits
lbldtypegt = 'uint%d' % lblbitsgt
assert( ngroups == len(groups_vals) )
run_gt_watershed = 'run_gt_watershed' in globals() and run_gt_watershed
is_merge = 'merge_probs' in globals()
run_merge = is_merge and run_merge
load_gt_watershed = 'gt_labels' not in globals()
group_single = 'group_single' in globals() and group_single     # if there is a single h5 for each group
group_merge = 'merge_probs' in globals() and not isinstance(merge_probs[0], str)
assert( not group_merge or len(out_name_probs) == ngroups )
assert( isinstance(out_name, str) or len(out_name) == ngroups )
if 'out_group' not in globals(): out_group = ''
export_raw = 'export_raw' in globals() and export_raw
if 'Tmins' not in globals(): Tmins = []
if Tmins: TminsStr = ' '.join(['%d' % (x,) for x in Tmins]) + ' '
skeletonize = 'skeletonize' in globals() and skeletonize
global_mins = 'global_mins' in globals() and global_mins
if 'nReSamples' not in globals(): nReSamples = 0
probWatershed = 'probWatershed' in globals() and probWatershed
skimWatershed = 'skimWatershed' in globals() and skimWatershed
if 'connectivity' not in globals(): connectivity = 1



# loop to optionally run all ground truth and network output "watersheds" to create ICS/ECS components
if run_gt_watershed or run_watershed:
    print('iterating over ECS data to run watersheds')
for i in range(ngroups):
    for j in range(nblocks):
        fn = groups[i] + ('' if group_single else str(blocks[j]))
        op = '' if not is_merge else (out_name_probs if not group_merge else out_name_probs[i])
        on = out_name if isinstance(out_name, str) else out_name[i]
        og = out_group if isinstance(out_group, str) else out_group[i]
        ofn = groups[i] + str(blocks[j])

        if run_gt_watershed:
            # "watershed" ground truth labels, i.e., create ground truth components
            fpin = os.path.join(inpath, fn, input_data)
            fpout = os.path.join(outpath, fn + gt_name)
            if os.path.isfile(fpin):
                torun = ('dpWatershedTypes.py --srclabels %s ' % fpin) +\
                    ('--chunk %d %d %d --offset %d %d %d --size %d %d %d ' % tuple(chunks[i][j] + offset + size)) +\
                    ('--outlabels %s ' % fpout) +\
                    ('--ThrRng 0.5 0.6 0.1 --TminSrc 20 --ThrHi  --outlabelsbits %d ' % lblbits)
                if skeletonize: torun += '--skeletonize '
                print('\n' + torun); t = time.time()
                os.system(torun)                
                print('\tdone in %.4f s' % (time.time() - t))

        # merge network probability outputs by averaging
        if run_merge:
            mp = merge_probs if not group_merge else merge_probs[i]
            mo = merge_orderings if not group_merge else merge_orderings[i]
            mw = merge_weightings if not group_merge else merge_weightings[i]
            fpins = [fn + x for x in mp]
            fpout = os.path.join(outpath, fn + op)
            # if you're wondering what's wrong here, try turning off the select!
            #print(fpins, fpout)
            if all([os.path.isfile(os.path.join(inpath,x)) for x in fpins]) and \
                (not sel_watershed or (i*nblocks + j + 1) in sel_watershed):
                if len(fpins) > 1:
                    torun = ('dpMergeProbs.py --srcpath %s --outprobs %s ' % (inpath,fpout)) +\
                        ('--chunk %d %d %d --offset %d %d %d --size %d %d %d ' % tuple(chunks[i][j] + offset + size)) +\
                        ('--srcfiles %s --dim-orderings %s ' % (' '.join(fpins), ' '.join(mo))) +\
                        ('--weightings %s ' % (' '.join([str(x) for x in mw])))
                    print('\n' + torun); t = time.time()
                    os.system(torun)                
                    print('\tdone in %.4f s' % (time.time() - t))
                else:
                    copyfile(os.path.join(inpath,fpins[0]),fpout)
    
        if export_raw:
            fpin = os.path.join(loadpath, fn + op)
            if os.path.isfile(fpin):
                for t in ['ICS','ECS','MEM']:
                    fpout = os.path.join(outpath,ofn+og+t+'.raw')
                    torun = ('dpLoadh5.py --srcfile %s ' % (fpin)) +\
                        ('--chunk %d %d %d --offset %d %d %d --size %d %d %d ' % tuple(chunks[i][j] + offset + size)) +\
                        ('--dataset %s --data-type float32 --outraw %s ' % (t,fpout))
                    print('\n' + torun); t = time.time()
                    os.system(torun)                
                    print('\tdone in %.4f s' % (time.time() - t))
                
            fpin = raw_data[i]
            if os.path.isfile(fpin):
                fpout = os.path.join(outpath,ofn+og+'_data.nrrd')
                torun = ('dpLoadh5.py --srcfile %s ' % (fpin)) +\
                    ('--chunk %d %d %d --offset %d %d %d --size %d %d %d ' % tuple(chunks[i][j] + offset + size)) +\
                    ('--dataset data_mag1 --outraw %s ' % (fpout))
                print('\n' + torun); t = time.time()
                os.system(torun)                
                print('\tdone in %.4f s' % (time.time() - t))
    
        # watershed network outputs at multiple thresholds
        fpin = os.path.join(inpath, fn, output_data) if not is_merge else os.path.join(outpath, fn + op)
        fpout = os.path.join(outpath, fn + on)
        #print(fpin, fpout, is_merge, inpath, fn, group_single, groups[i])
        if run_watershed and os.path.isfile(fpin) and \
            (not sel_watershed or (i*nblocks + j + 1) in sel_watershed):
            torun = ('dpWatershedTypes.py --probfile %s ' % fpin) +\
                ('--chunk %d %d %d --offset %d %d %d --size %d %d %d ' % tuple(chunks[i][j] + offset + size)) +\
                ('--outlabels %s --outlabelsbits %d ' % (fpout, lblbits)) +\
                ('--ThrRng %.5f %.5f %.5f ' % tuple(thrRng)) + '--ThrHi ' + thrHiStr + '--ThrLo ' + thrLoStr +\
                ('--connectivity %d ' % (connectivity,))
            if Tmins: torun += ('--Tmins ' + TminsStr)
            if probWatershed: torun += '--probWatershed '
            if skimWatershed: torun += '--skimWatershed '
            print('\n' + torun); t = time.time()
            os.system(torun)                
            print('\tdone in %.4f s' % (time.time() - t))


def load_gt_cat(fpgt, fpout, i, j):
    gtSkelComps = None; skeletonize = True
    if load_gt_watershed:
        # load ground truth labels and components (watershedded version generated above)
        loadh5 = emVoxelType.readVoxType(srcfile=fpgt, chunk=chunks[i][j], offset=offset, size=size)
        gtLbls = loadh5.data_cube
        loadh5 = emLabels.readLabels(srcfile=fpgt, chunk=chunks[i][j], offset=offset, size=size, 
            data_type=lbldtypegt, subgroups=['with_background', '%.8f' % 0.5])
        gtComps = loadh5.data_cube; gtComps_types_nlabels = loadh5.data_attrs['types_nlabels']
        # remove the ECS components for calculating metrics
        gtComps[gtComps > gtComps_types_nlabels[0]] = 0
        try:
            loadh5 = emLabels.readLabels(srcfile=fpgt, chunk=chunks[i][j], offset=offset, size=size, 
                data_type=lbldtypegt, subgroups=['skeletonized', '%.8f' % 0.5])
        except:
            skeletonize = False
        if skeletonize:
            gtSkelComps = loadh5.data_cube;
            gtSkelComps[gtSkelComps > gtComps_types_nlabels[0]] = 0
            ## remove "nodes" that are not exactly the expected size, xxx - make this node stuff an option???
            #gtSkelComps, sizes = emLabels.thresholdSizes(gtSkelComps, minSize=13)
            #gtSkelComps, sizes = emLabels.thresholdSizes(gtSkelComps, minSize=-14)
    else:
        # load ground truth and components from segmented labels file
        loadh5 = emLabels.readLabels(srcfile=fpgt, chunk=chunks[i][j], offset=offset, size=size, 
            data_type=lbldtypegt)
        gtComps = loadh5.data_cube; gtIsECS = (gtComps == gt_ECS_label);
        gtLbls = np.zeros(gtComps.shape, dtype=np.uint8)
        gtLbls[np.logical_and(gtComps > 0, np.logical_not(gtIsECS))] = 1; gtLbls[gtIsECS] = 2;
        # remove the ECS components for calculating metrics
        gtComps[gtIsECS] = 0; n = gtComps.max(); gtComps[gtComps == n] = gt_ECS_label; 
        gtComps_types_nlabels = [n-1, 1]    # do not have number of ECS objects from GT
        
    # load network output max prob categories (voxelTypes, "labels")
    loadh5 = emVoxelType.readVoxType(srcfile=fpout, chunk=chunks[i][j], offset=offset, size=size)
    outLbls = loadh5.data_cube; attrs = loadh5.data_attrs
                
    return gtLbls, gtComps, gtComps_types_nlabels, gtSkelComps, outLbls, attrs


def load_prp_param(k):
    outComps = None; outComps_types_nlabels = None; outCompsFull = None; outCompsNoAdj = None;

    # current params are [threshold, Tmin] in F-order unravel
    kp = np.unravel_index(k, szparams, order='F')
    if nTmin == 1: subgroups = ['%.8f' % (thresholds[k],)] 
    else: subgroups = ['%d' % (Tmins[kp[1]],), '%.8f' % (thresholds[kp[0]],)]

    # if one of these does not load then usually this means something was really bad about the 
    #   network output (xxx - investigate why watershed doesn't handle this gracefully).
    # just skip these ones and leave them as no data points
    try:
        # load components at this threshold, normal, fully watershedded and without label adjacencies
        loadh5 = emLabels.readLabels(srcfile=fpout, chunk=chunks[i][j], offset=offset, size=size, 
            data_type=lbldtype, subgroups=['with_background'] + subgroups)
        outComps = loadh5.data_cube; outComps_types_nlabels = loadh5.data_attrs['types_nlabels']
        loadh5 = emLabels.readLabels(srcfile=fpout, chunk=chunks[i][j], offset=offset, size=size, 
            data_type=lbldtype, subgroups=['zero_background'] + subgroups)
        outCompsFull = loadh5.data_cube; outCompsFull_types_nlabels = loadh5.data_attrs['types_nlabels']
        loadh5 = emLabels.readLabels(srcfile=fpout, chunk=chunks[i][j], offset=offset, size=size, 
            data_type=lbldtype, subgroups=['no_adjacencies'] + subgroups)
        outCompsNoAdj = loadh5.data_cube
        
        # remove the ECS components for calculating metrics for non-fully-watershedded versions
        outComps[outComps > outComps_types_nlabels[0]] = 0
        outCompsNoAdj[outCompsNoAdj > outComps_types_nlabels[0]] = 0
    except:
        # hack to use this script to do gala comparisons
        loadh5 = emLabels.readLabels(srcfile=fpout, chunk=chunks[i][j], offset=offset, size=size, 
            data_type=lbldtype, subgroups=['with_background'] + subgroups)
        outComps = loadh5.data_cube; outComps_types_nlabels = loadh5.data_attrs['types_nlabels']
        outCompsFull = loadh5.data_cube; outCompsFull_types_nlabels = loadh5.data_attrs['types_nlabels']
        outCompsNoAdj = loadh5.data_cube

    return subgroups, outComps, outComps_types_nlabels, outCompsFull, outCompsNoAdj
    

def calcGTRands(gtCompsR):
    ris = np.zeros((ngroups,nblocks),dtype=np.double)
    for i in range(ngroups):
        for j in range(nblocks):
            fn = groups[i] + ('' if group_single else str(blocks[j]))
            on = out_name if isinstance(out_name, str) else out_name[i]

            fpgt = os.path.join(loadpath, fn + gt_name) if load_gt_watershed else os.path.join(inpath, gt_labels[i])
            fpout = os.path.join(loadpath, fn + on)
            #if not (os.path.isfile(fpgt) and os.path.isfile(fpout) and \
            #    (not sel_metrics or (i*nblocks + j + 1) in sel_metrics)): continue
                
            gtLbls, gtComps, gtComps_types_nlabels, gtSkelComps, outLbls, attrs = load_gt_cat(fpgt, fpout, i, j)
            are,prec,rec,ris[i,j] = adapted_rand_error( gtCompsR, gtComps, nogtbg=True, getRI=True )
    return ris


if run_metrics:
    print('iterating over watersheds to calculate metics')

    if sel_metrics and os.path.isfile(os.path.join(outpath,save_file)):
        # xxx - parameters could be different, maybe don't care here?
        with open(os.path.join(outpath,save_file), 'rb') as f: d = dill.load(f)
        metrics = d['metrics']; globals().update(metrics)
    else:
        metrics = None

    # saves a bit of time not reloading the LUT every time for the warping error
    simpleLUT = None

    for i in range(ngroups):
        for j in range(nblocks):
            fn = groups[i] + ('' if group_single else str(blocks[j]))
            on = out_name if isinstance(out_name, str) else out_name[i]

            fpgt = os.path.join(loadpath, fn + gt_name) if load_gt_watershed else os.path.join(inpath, gt_labels[i])
            fpout = os.path.join(loadpath, fn + on)
            if not (os.path.isfile(fpgt) and os.path.isfile(fpout) and \
                (not sel_metrics or (i*nblocks + j + 1) in sel_metrics)): continue
                
            gtLbls, gtComps, gtComps_types_nlabels, gtSkelComps, outLbls, attrs = load_gt_cat(fpgt, fpout, i, j)

            print('calculating metrics for ' + fn + on + (' chunk %d %d %d' % tuple(chunks[i][j]))); t = time.time()
            nthresh = len(attrs['thresholds']); nTmin = len(attrs['Tmins'])
            szparams = np.array([nthresh, nTmin], dtype=np.uint32); nparams = szparams.prod()
            if not metrics:
                metrics = {
                    'thresholds' : attrs['thresholds'], 'Tmins' : attrs['Tmins'],
                    'cat_error' : ma.array(np.zeros((ngroups,nblocks),dtype=np.double),mask=True),
                    'are_gala' : ma.array(np.zeros((ngroups,nblocks,nparams),dtype=np.double),mask=True),
                    'are_precrec_gala' : ma.array(np.zeros((ngroups,nblocks,nparams,2),dtype=np.double),mask=True),
                    'split_vi_gala' : ma.array(np.zeros((ngroups,nblocks,nparams,2),dtype=np.double),mask=True),
                    'are_skel' : ma.array(np.zeros((ngroups,nblocks,nparams),dtype=np.double),mask=True),
                    'are_precrec_skel' : ma.array(np.zeros((ngroups,nblocks,nparams,2),dtype=np.double),mask=True),
                    'are_resamp' : ma.array(np.zeros((ngroups,nblocks,nparams),dtype=np.double),mask=True),
                    'are_precrec_resamp' : ma.array(np.zeros((ngroups,nblocks,nparams,2),dtype=np.double),mask=True),
                    'ares_resamp' : ma.array(np.zeros((ngroups,nblocks,nparams,nReSamples),dtype=np.double),mask=True),
                    'precs_resamp' : ma.array(np.zeros((ngroups,nblocks,nparams,nReSamples),dtype=np.double),mask=True),
                    'recs_resamp' : ma.array(np.zeros((ngroups,nblocks,nparams,nReSamples),dtype=np.double),mask=True),
                    'wrp_err' : ma.array(np.zeros((ngroups,nblocks,nparams),dtype=np.double),mask=True),
                    'wrp_sltmrg' : ma.array(np.zeros((ngroups,nblocks,nparams,2),dtype=np.int64),mask=True),
                    'nlabels_out' : ma.array(np.zeros((ngroups,nblocks,nparams,2),dtype=np.int64),mask=True),
                    'nlabels' : ma.array(np.zeros((ngroups,nblocks),dtype=np.int64),mask=True),
                    'voxel_sizes' : [[None]*nblocks]*ngroups,
                    'voxel_sizes_median' : ma.array(np.zeros((ngroups,nblocks),dtype=np.int64),mask=True),
                    'voxel_sizes_groups' : [np.zeros((0,),dtype=np.int64)]*ngroups,
                    'nlabels_skel' : ma.array(np.zeros((ngroups,nblocks),dtype=np.int64),mask=True),
                    'voxel_sizes_skel' : [[None]*nblocks]*ngroups,
                    'voxel_sizes_skel_median' : ma.array(np.zeros((ngroups,nblocks),dtype=np.int64),mask=True),
                    'voxel_sizes_skel_groups' : [np.zeros((0,),dtype=np.int64)]*ngroups,
                    #'voxel_sizes_out' : ma.array(np.zeros((ngroups,nblocks,nparams,2),dtype=np.int64),mask=True),
                    'rand_index_gts' : ma.array(np.zeros((ngroups,nblocks,ngroups,nblocks),dtype=np.double),mask=True),
                    'ari_gtnorm' : ma.array(np.zeros((ngroups,nblocks,nparams),dtype=np.double),mask=True),
                    }                
                globals().update(metrics)
            else:
                assert( all([x==y for x,y in zip(metrics['thresholds'],attrs['thresholds'])]) )
                assert( all([x==y for x,y in zip(metrics['Tmins'],attrs['Tmins'])]) )
                    
            # calculate the categorization error
            cat_error[i,j] = (gtLbls != outLbls).sum(dtype=np.int64) / float(outLbls.size)
                
            # save the number of ground truth labels and voxel sizes
            nlabels[i,j] = gtComps_types_nlabels[0]
            voxel_sizes[i][j] = emLabels.getSizes(gtComps)[1:]
            voxel_sizes_median[i,j] = np.median(voxel_sizes[i][j])
            voxel_sizes_groups[i] = np.concatenate((voxel_sizes[i][j], voxel_sizes_groups[i]))
            if skeletonize:
                voxel_sizes_skel[i][j] = emLabels.getSizes(gtSkelComps)[1:]
                nlabels_skel[i,j] = voxel_sizes_skel[i][j].size
                voxel_sizes_skel_median[i,j] = np.median(voxel_sizes_skel[i][j])
                voxel_sizes_skel_groups[i] = np.concatenate((voxel_sizes_skel[i][j], voxel_sizes_skel_groups[i]))

            # calculate the rand error versus all other ground truths
            rand_index_gts[i,j,:,:] = calcGTRands(gtComps)

            for k in range(nparams):
                subgroups, outComps, outComps_types_nlabels, outCompsFull, outCompsNoAdj = load_prp_param(k)
                if outComps is None: continue

                # calculate the ISBI2013 rand error (gala, excludes gt background) using the full out components
                are, prec, rec = ev.adapted_rand_error(outCompsFull, gtComps, all_stats=True)
                are_gala[i,j,k] = are; are_precrec_gala[i,j,k,:] = np.array([prec,rec])
                    
                # calculate the split variation of information (gala) using full out components
                split_vi_gala[i,j,k,:] = ev.split_vi(outCompsFull, gtComps)

                # calculate the adapted rand index using the expected value by from comparing different GTs
                sel = np.ones((ngroups,nblocks),dtype=np.bool); sel[i,j] = False; eri = rand_index_gts[i,j,sel].mean()
                are,prec,rec,ri,ari_gtnorm[i,j,k] = adapted_rand_error( gtComps, outComps, nogtbg=True, getRI=True,
                    eri=eri )

                # xxx - clean this up, maybe remove entirely if not going to use it
                if skeletonize:
                    # calculate rand error at gt skeletonization points only (similar to skeleton nodes)
                    #   using full out components to avoid nodes falling into background
                        
                    # rand error with skels
                    are, prec, rec = adapted_rand_error(gtSkelComps, outCompsFull, nogtbg=True)
                    are_skel[i,j,k] = are; are_precrec_skel[i,j,k,:] = np.array([prec,rec])
                       
                # calculate rand error using a sampling method that tries to fairly compare different datasets by 
                #   sampling the same number of points and same number of objects from each block.
                # nObjects and nPoints are chosen as min number of objects and pixels per obj over all tiles / objs
                if nReSamples > 0:
                    ares_resamp[i,j,k,:], precs_resamp[i,j,k,:], recs_resamp[i,j,k,:] = \
                        adapted_rand_error_resample_objects_points(gtComps, outCompsFull, nObjects=71, nPoints=20, 
                        nSamples=nReSamples, getDistros=True, nThreads=10)
                        
                '''
                # calculate our warping metric using the out with adjacencies removed
                # NOTE: gt components were created using connected components, so adjacencies are already removed
                # OR gt components were generated with adjacencies pre-removed with label checker (for 3d gt)
                wrp, nsplits, nmerges, nonsimple, simpleLUT = warping_error(gtComps.astype(np.bool,order='C'), 
                    outCompsNoAdj.astype(np.bool,order='C'), doComps=True, simpleLUT=simpleLUT)
                wrp_err[i,j,k] = wrp; wrp_sltmrg[i,j,k,:] = np.array([nmerges,nsplits])
                '''
                    
                '''
                # hack to save warping outputs
                og = out_group if isinstance(out_group, str) else out_group[i]
                ofn = groups[i] + str(blocks[j]) + '_'.join(subgroups)
                # xxx - left off here, save subgroups as part of file name
                dlut = np.fromfile('/usr/local/Fiji.app/luts/distinguish2.lut',dtype=np.uint8).reshape([3,-1]).T
                tmp = nonsimple; sel = (tmp > 0); tmp[sel] = tmp[sel] % (dlut.shape[0]-1)
                tifffile.imsave(os.path.join(outpath,ofn+og+'_wrp.tif'), dlut[tmp.transpose(2,1,0),:], compress=5)
                '''
                    
                # store number of ICS/ECS labels for normal and fully watershedded and ICS labels for no adajacency
                nlabels_out[i,j,k,:] = outComps_types_nlabels
            print('\tdone in %.4f s' % (time.time() - t))

    # save all metric data using dill
    with open(os.path.join(outpath,save_file), 'wb') as f: dill.dump({'metrics':metrics, 'params':params}, f)

elif make_plots or export_images:
    # xxx - parameters could be different, maybe don't care here?
    with open(os.path.join(outpath,save_file), 'rb') as f: d = dill.load(f)
    globals().update(d); globals().update(metrics)



#print(nlabels)



if make_plots or export_images:
    print(','.join(['%.8f' % x for x in thresholds]))
    # some inits based on saved metrics
    nthresh = len(thresholds); nTmin = len(Tmins)
    szparams = np.array([nthresh, nTmin], dtype=np.uint32); nparams = szparams.prod()

    # some overall metrics based on those calculated
    #dist_vi_gala = np.linalg.norm(split_vi_gala, axis=3)
    dist_vi_gala = split_vi_gala.sum(axis=3)
    wrp_sltmrg_obj = wrp_sltmrg / nlabels.astype(np.double)[:,:,None,None]
    #dist_wrp_sltmrg_obj = np.linalg.norm(wrp_sltmrg_obj, axis=3)
    dist_wrp_sltmrg_obj = wrp_sltmrg_obj.sum(axis=3)

    # change ari to (1 - ari)
    #ari_gtnorm = 1-ari_gtnorm

    # only calculate resamplings stuff if it was run with non-zero number of resamples
    nReSamples = ares_resamp.shape[3] if 'ares_resamp' in globals() else 0

    if nReSamples > 0:
        # make resamps as medians of resamp distros
        are_resamp = np.median( ares_resamp, axis=3 )
        are_prec_rec_resamp = np.concatenate( (np.median(precs_resamp, axis=3, keepdims=True), 
            np.median(recs_resamp, axis=3, keepdims=True)), axis=3 )

    # shape of all scalar metrics is (ngroups,nblocks,nparams)
    if global_mins:
        i = range(ngroups)
        # use a single threshold for mins for each metric (instead of min metric threshold for each block)
        amin_are_gala = np.median(are_gala,axis=1).argmin(axis=1)
        min_are_gala = are_gala[i,:,amin_are_gala]
        amin_are_skel = np.median(are_skel,axis=1).argmin(axis=1)
        min_are_skel = are_skel[i,:,amin_are_skel]
        amin_are_resamp = np.median(are_resamp,axis=1).argmin(axis=1)
        min_are_resamp = are_resamp[i,:,amin_are_resamp]
        #amin_ari_gtnorm = np.median(ari_gtnorm,axis=1).argmin(axis=1)
        #min_ari_gtnorm = ari_gtnorm[i,:,amin_ari_gtnorm]
        amin_dist_vi_gala = np.median(dist_vi_gala,axis=1).argmin(axis=1)
        min_dist_vi_gala = dist_vi_gala[i,:,amin_dist_vi_gala]
        amin_dist_wrp_sltmrg_obj = np.median(dist_wrp_sltmrg_obj,axis=1).argmin(axis=1)
        #amin_dist_wrp_sltmrg_obj = np.mean(dist_wrp_sltmrg_obj,axis=1).argmin(axis=1)
        min_dist_wrp_sltmrg_obj = dist_wrp_sltmrg_obj[i,:,amin_dist_wrp_sltmrg_obj]
        amin_wrp_err = np.median(wrp_err,axis=1).argmin(axis=1)
        min_wrp_err = wrp_err[i,:,amin_wrp_err]
        
        if nReSamples > 0:
            # take all resamp samples but only at mins
            #'ares_resamp' : ma.array(np.zeros((ngroups,nblocks,nparams,nReSamples),dtype=np.double),mask=True),
            min_ares_resamp = ares_resamp[range(ngroups),:,amin_are_resamp,:].reshape((ngroups, nblocks*nReSamples))
    else:
        # mins across thresholds
        min_are_gala = are_gala.min(axis=2)
        amin_are_gala = are_gala.argmin(axis=2)
        min_are_skel = are_skel.min(axis=2); 
        amin_are_skel = are_skel.argmin(axis=2)
        min_are_resamp = are_resamp.min(axis=2); 
        amin_are_resamp = are_resamp.argmin(axis=2)
        #min_ari_gtnorm = ari_gtnorm.min(axis=2); 
        #amin_ari_gtnorm = ari_gtnorm.argmin(axis=2)
        min_dist_vi_gala = dist_vi_gala.min(axis=2)
        amin_dist_vi_gala = dist_vi_gala.argmin(axis=2)
        min_dist_wrp_sltmrg_obj = dist_wrp_sltmrg_obj.min(axis=2)
        amin_dist_wrp_sltmrg_obj = dist_wrp_sltmrg_obj.argmin(axis=2)
        min_wrp_err = wrp_err.min(axis=2)
        amin_wrp_err = wrp_err.argmin(axis=2)
        
        # xxx - fix however to select min_ares_resamp here

    # repmat of percentage ecs for blocks
    groups_vals_rep = np.array(groups_vals, dtype=np.double).reshape((ngroups,1)).repeat(nblocks,axis=1)



def scatter_err_plots(errs, strs, ylims, plsize, dostats, groups_vals_rep_loc=None, jitter=0.25):
    if groups_vals_rep_loc is None: groups_vals_rep_loc = groups_vals_rep
    for i in range(len(errs)):
        if plot_setlims and ylims: ax = pl.subplot( plsize[0],plsize[1],i+1, adjustable='box', 
            aspect=(groups_xlim[1]-groups_xlim[0])/(ylims[i][1]-ylims[i][0]) )
        else: ax = pl.subplot(plsize[0],plsize[1],i+1)
        x = groups_vals_rep_loc.reshape((-1,));
        if jitter > 0: x = x + np.random.randn(x.size) * jitter;
        if errs[i].shape[1] > 10:
            plt.scatter(x,errs[i].reshape((-1,)),marker=u'.',c=u'k',s=1)
        else:
            plt.scatter(x,errs[i].reshape((-1,)),marker=u'o',c=u'k',s=20)
        plt.scatter(groups_vals,np.median(errs[i],axis=1),c=u'r', marker=u'+',s=100, linewidths=2)
        # Hide the right and top spines
        ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left'); ax.xaxis.set_ticks_position('bottom')        
        if plot_setlims and ylims: plt.xlim(groups_xlim); plt.ylim(ylims[i])
        pl.xlabel(groups_xlbl); pl.ylabel(strs[i])
        ax.set_xticks(groups_vals); #ax.set_xticklabels(['10', '100', '1000', '10000', '100000'])
        
    
        if dostats[i]:
            # overall differences between groups
            H,p = mstats.kruskalwallis(*[errs[i][j,:] for j in range(ngroups)])
    
            # pairwise comparisons - check matlab code for multiple comparisons, decided to drop this
            pvals = []; Uvals = []; pstr = ''; cnt=0
            #for j in range(ngroups):
            for j in range(1):
                for k in range(j+1,ngroups):
                    U,pval = mstats.mannwhitneyu(errs[i][j,:],errs[i][k,:], use_continuity=False)
                    #U,pval = stats.ranksums(errs[i][j,:],errs[i][k,:])
                    pvals.append(pval); Uvals.append(U); pstr += '%dto%d %g %g ' % (j,k,pvals[-1],Uvals[-1]); cnt += 1
                    if (cnt%3)==0: pstr += '\n'; 
            plt.title('med=%s\np=(%g,%g)\n%s' % (' '.join(['%g' % (x,) for x in np.median(errs[i],axis=1)]),p,H,pstr))
        else:
            plt.title('med=%s' % (' '.join(['%g' % (x,) for x in np.median(errs[i],axis=1)]),))

if make_plots:
    print('generating plots')
    
    from matplotlib import pylab as pl
    from matplotlib import pyplot as plt
    import matplotlib as mpl

    baseno = figno; figno -= 1
    fignames = ['gt_sizes', 'gt_scatters', 'error_scatters', 'error_curves', 'resamp_scatter', 'gt_RIs']
    #figmask = [True, False, True, True, False, False]
    #figmask = [True, False, False, False, False, False]
    figmask = [False, False, True, True, False, False]

    # none, small, large, huge
    # blue, green, yellow, red
    clrs = np.array([[0.28235294, 0.23921569, 0.54509804], [0.33333333, 0.41960784, 0.18431373], [1, 0.54901961, 0], 
        [0.54509804, 0, 0]]).T

    figno+=1
    if figmask[figno-baseno]:
        pl.figure(figno);

        #sizes = [voxel_sizes_groups, voxel_sizes_skel_groups]
        #strs = ['GT voxels per object (log10)', 'skel GT voxels per object (log10)']
        #plsize = [1,2]
        
        sizes = [voxel_sizes_groups]
        strs = ['GT voxels per object']
        plsize = [1,1]

        #dx = 0.1; xrng = [1,5]; x = np.arange(xrng[0], xrng[1], dx); cx = np.arange(xrng[0]+dx/2, xrng[1]-dx/2, dx);
        dx = 0.1; xrng = [1,6.5]; x = np.arange(xrng[0], xrng[1], dx); cx = np.arange(xrng[0]+dx/2, xrng[1]-dx/2, dx);
        for i in range(len(sizes)):
            ax = pl.subplot(plsize[0],plsize[1],i+1)
            
            for j in range(ngroups):
                # xxx - any purpose of range=(x.min(),x.max() ???
                hx,edges = np.histogram(np.log10(sizes[i][j]), bins=x)
                plt.plot(np.power(10,cx), np.cumsum(hx)/hx.sum(dtype=np.double), color=clrs[:,j], 
                    label=str(groups_vals[j]))
            pl.ylabel('cdf'); pl.xlabel(strs[i])
            plt.xlim(np.power(10,xrng)); plt.ylim([-0.05, 1.05])
            ax.set_xscale('log')
            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
            #ax.set_xticks([10, 100, 1e3, 1e4, 1e5]); ax.set_xticklabels(['10', '100', '1000', '10000', '100000'])
            ax.set_xticks([10, 100, 1e3, 1e4, 1e5, 1e6]); ax.set_xticklabels(['10', '100', '1000', '10000', '100000', '1000000'])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            #plt.legend(loc='lower right')
            
            pvals = []; Dvals = []; pstr = ''; cnt=0
            #for j in range(ngroups):
            for j in range(1):
                for k in range(j+1,ngroups):
                    D,pval = stats.ks_2samp(sizes[i][j], sizes[i][k])
                    pvals.append(pval); Dvals.append(D); pstr += '%dto%d %g %g ' % (j,k,pvals[-1],Dvals[-1]); cnt += 1
                    if (cnt%3)==0: pstr += '\n'; 
    
            plt.title('med=%s\n%s' % \
                (' '.join(['%g' % (n,) for n in [np.median(sizes[i][k]) for k in range(ngroups)]]),pstr))

    figno+=1
    if figmask[figno-baseno]:
        pl.figure(figno);
        
        #errs = [nlabels, nlabels_skel, voxel_sizes_median, voxel_sizes_skel_median]
        #strs = ['GT Number of Objects', 'Skel GT Number of Objects', 
        #    'GT Object Voxel Size Medians', 'Skel GT Object Voxel Size Medians']
        #ylims = [];
        #plsize = [2,2]
        #dostats = [True, False, True, False]
        
        errs = [nlabels, voxel_sizes_median]
        strs = ['GT Number of Objects', 'GT Object Voxel Size Medians']
        ylims = [];
        plsize = [1,2]
        dostats = [True, True]
        
        scatter_err_plots(errs, strs, ylims, plsize, dostats)
        
    figno+=1
    if figmask[figno-baseno]:
        pl.figure(figno);

        '''
        errs = [cat_error, min_are_gala, min_dist_vi_gala, min_dist_wrp_sltmrg_obj]
        strs = ['Categorization Error', 'Adapted Rand Error', 'Variation of Information', '(Splits + Merges) / Object']
        #errs = [cat_error, min_are_gala, min_ari_gtnorm, min_dist_wrp_sltmrg_obj]
        #strs = ['Categorization Error', 'Adapted Rand Error', '1 - Adjusted Rand Index', '(Splits + Merges) / Object']
        #errs = [cat_error, min_are_resamp, min_are_gala, min_dist_wrp_sltmrg_obj]
        #strs = ['Categorization Error', 'Resampled Adapted Rand Error', 'Adapted Rand Error', 
        #    '(Splits + Merges) / Object']
        #errs = [cat_error, min_wrp_err, min_dist_vi_gala, min_dist_wrp_sltmrg_obj]
        #strs = ['Categorization Error', 'Warping Error', 'Variation of Information', '(Splits + Merges) / Object']
        #ylims = [[0,0.11], [0,0.08], [0,0.34], [0,1]];  # for 2d 
        ylims = [[0,0.16], [0,0.5], [0,2.5], [0,20]];  # for 3d
        plsize = [2,2]
        #dostats = [True, True, True, True]
        dostats = [False, False, False, False]
        '''
                                
        '''
        errs = [cat_error, min_are_resamp, min_dist_wrp_sltmrg_obj]
        strs = ['Categorization Error', 'Resampled Adapted Rand Error', '(Splits + Merges) / Object']
        ylims = [[0,0.12], [0,0.225], [0,1]];  # for 2d 
        plsize = [2,2]
        dostats = [True, True, True]
        '''

        errs = [min_are_gala, min_dist_vi_gala]
        strs = ['Adapted Rand Error', 'Variation of Information']
        ylims = [[0,0.12], [0,0.225]];  # for 2d 
        plsize = [1,2]
        dostats = [False, False, False, False]
        
        scatter_err_plots(errs, strs, ylims, plsize, dostats)

    figno+=1
    if figmask[figno-baseno]:
        pl.figure(figno); 

        '''
        errs = [1-are_precrec_gala, split_vi_gala, wrp_sltmrg_obj, ]
        amins = [amin_are_gala, amin_dist_vi_gala, amin_dist_wrp_sltmrg_obj, ]
        #errs = [1-are_precrec_resamp, split_vi_gala, wrp_sltmrg_obj, ]
        #amins = [amin_are_resamp, amin_dist_vi_gala, amin_dist_wrp_sltmrg_obj, ]
        xstrs = ['1-Recall', 'False Splits (bits)', 'Splits / Object', ]
        ystrs = ['1-Precision', 'False Merges (bits)', 'Merges / Object', ]
        lims = [[0,0.2], [0,0.4], [0,0.8], ];   # for 2d
        #lims = [[0,0.5], [0,3], [0,15], ];     # for 3d
        titles = ['Adapted Rand', 'Split Variation of Information', 'Warping Error', ]
        plsize = [2,2]
        plot_mean = False
        '''
        
        '''
        errs = [wrp_sltmrg_obj, ]
        amins = [amin_dist_wrp_sltmrg_obj, ]
        xstrs = ['Splits / Object', ]
        ystrs = ['Merges / Object', ]
        lims = [[0,0.6], ];   # for 2d
        titles = ['Warping Error', ]
        plsize = [1,1]
        plot_mean = True
        '''

        errs = [1-are_precrec_gala, split_vi_gala, ]
        amins = [amin_are_gala, amin_dist_vi_gala,  ]
        xstrs = ['1-Recall', 'False Splits (bits)', ]
        ystrs = ['1-Precision', 'False Merges (bits)', ]
        lims = [[-0.05,1.0], [-0.5,4], ];     # for gala
        titles = ['Adapted Rand', 'Split Variation of Information',  ]
        plsize = [1,2]
        plot_mean = True
        
        for i in range(len(errs)):
            if plot_setlims: ax = pl.subplot(plsize[0],plsize[1],i+1, adjustable='box', aspect=1)
            else: ax = pl.subplot(plsize[0],plsize[1],i+1)
            for j in range(ngroups):
                # 'wrp_sltmrg' : ma.array(np.zeros((ngroups,nblocks,nparams,[splits,merges])
                if plot_mean:
                    u = np.mean(errs[i][j,:,:,:], axis=0)
                    # divide by sqrt(nblocks) to get standard error of the mean
                    s = np.std(errs[i][j,:,:,:], axis=0) / np.sqrt(cat_error[j,:].count(axis=0))
                else:
                    u = np.median(errs[i][j,:,:,:], axis=0)
                    s = np.median(np.abs(errs[i][j,:,:,:] - u), axis=0) # median absolute deviation
                plt.plot(u[:,1], u[:,0], color=clrs[:,j], marker='x')
                plt.plot(u[:,1]-s[:,1], u[:,0]-s[:,0], color=clrs[:,j], linestyle='dashed')
                plt.plot(u[:,1]+s[:,1], u[:,0]+s[:,0], color=clrs[:,j], linestyle='dashed')
                # plot point at mean of min dist points across blocks
                if global_mins:
                    m = errs[i][j,:,amins[i][j],:]
                else:
                    m = np.zeros((nblocks,2),dtype=np.double)
                    for k in range(nblocks): m[k,:] = errs[i][j,k,amins[i][j,k],:]
                if plot_mean:
                    mu = np.mean(m,axis=0); 
                    ##su = np.std(m,axis=0) / np.sqrt(cat_error[j,:].count(axis=0))   # divide ngroups for SEM
                else:
                    mu = np.median(m,axis=0); 
                plt.scatter(mu[1],mu[0],color=clrs[:,j],s=20)
                ##plt.scatter(mu[0]-su[0],mu[1]-su[1],color=clrs[:,j],s=20,marker='x')
                ##plt.scatter(mu[0]+su[0],mu[1]+su[1],color=clrs[:,j],s=20,marker='x')
            # Hide the right and top spines
            ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
            # Only show ticks on the left and bottom spines
            ax.yaxis.set_ticks_position('left'); ax.xaxis.set_ticks_position('bottom')        
            if plot_setlims: plt.xlim(lims[i]); plt.ylim(lims[i]); 
            pl.xlabel(xstrs[i]); pl.ylabel(ystrs[i])
            plt.title(titles[i])

    figno+=1
    if figmask[figno-baseno]:
        pl.figure(figno); 
        #min_ares_resamp = ares_resamp[range(ngroups),:,amin_are_resamp,:].reshape((ngroups, nblocks*nReSamples))
        
        errs = [min_ares_resamp]
        strs = ['ReSamp ARE']
        ylims = [];   
        plsize = [1,1]
        dostats = [True]
        gv = np.array(groups_vals, dtype=np.double).reshape((ngroups,1)).repeat(nblocks*nReSamples,axis=1)
        
        scatter_err_plots(errs, strs, ylims, plsize, dostats, groups_vals_rep_loc=gv, jitter=0.8)

    figno+=1
    if figmask[figno-baseno]:
        pl.figure(figno); 
        #'rand_index_gts' : ma.array(np.zeros((ngroups,nblocks,ngroups,nblocks),dtype=np.double),mask=True),
        pl.imshow(rand_index_gts.reshape((ntotal,ntotal)),interpolation='nearest')
        pl.ylabel('GT (controls BG)'); pl.xlabel('to GT')
        pl.colorbar()


    if save_plots:
        print('saving plots')
        for f,i in zip(range(baseno, figno+1), range(figno-baseno+1)):
            print('exporting ',f,i,fignames[i])
            pl.figure(f)
            figure = plt.gcf() # get current figure
            figure.set_size_inches(20, 20)
            plt.savefig(os.path.join(outpath, fignames[i] + '.png'), dpi=72)
            plt.savefig(os.path.join(outpath, fignames[i] + '.eps'))
    else:
        #http://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python 
        #print('#1 Backend:',plt.get_backend())
        #figManager = plt.get_current_fig_manager()
        #figManager.window.showMaximized()
        pl.show()
 

 

if export_images:
    #amin_out = np.zeros((ngroups,nblocks),dtype=np.int64)
    #min_out = np.zeros((ngroups,nblocks),dtype=np.double)
    #amin_str = 'zero'
    #amin_out = amin_are_resamp
    #min_out = min_are_resamp
    #amin_str = 'AREresamp'
    amin_out = amin_dist_wrp_sltmrg_obj
    min_out = min_dist_wrp_sltmrg_obj
    amin_str = 'wrpsltmrgobj'
    

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

    print('iterating over watersheds to export images')
    for i in range(ngroups):
        for j in range(nblocks):
            fn = groups[i] + ('' if group_single else str(blocks[j]))
            on = out_name if isinstance(out_name, str) else out_name[i]
            og = out_group if isinstance(out_group, str) else out_group[i]
            ofn = groups[i] + str(blocks[j])

            fpgt = os.path.join(loadpath, fn + gt_name) if load_gt_watershed else os.path.join(inpath, gt_labels[i])
            fpout = os.path.join(loadpath, fn + on)
            if os.path.isfile(fpgt) and os.path.isfile(fpout):
                gtLbls, gtComps, gtComps_types_nlabels, gtSkelComps, outLbls, attrs = load_gt_cat(fpgt, fpout, i, j)
                
                k = amin_out[i] if global_mins else amin_out[i,j]
                subgroups, outComps, outComps_types_nlabels, outCompsFull, outCompsNoAdj = load_prp_param(k)
                
                print('exporting images for ' + ofn + on + (' chunk %d %d %d' % tuple(chunks[i][j]))); t = time.time()

                # annotated categorization error figure
                cat = gtLbls.copy(); cat[gtLbls != outLbls] = 3;
                cmap = (np.array([[0,0,0], [1,1,1], [0.5,0.5,0.5], [1,0,0]], dtype=np.single)*255).astype(np.uint8)
                tifffile.imsave(os.path.join(outpath,ofn+og+'_annotated_cat_error.tif'),cmap[cat.transpose(2,1,0),:],
                    compress=5)
                #(gtLbls!=outLbls).transpose(2,1,0).astype(np.uint8).tofile(os.path.join(outpath,
                #    ofn+og+'_cat_error_uint8.raw'))

                # gt labels
                cmap = (np.array([[0,0,0], [1,1,1], [0.5,0.5,0.5]], dtype=np.single)*255).astype(np.uint8)
                tifffile.imsave(os.path.join(outpath,ofn+og+'_gtLbls.tif'),cmap[gtLbls.transpose(2,1,0),:], compress=5)

                # gt components
                tmp = gtComps; sel = (tmp > 0); tmp[sel] = tmp[sel] % (dlut.shape[0]-1)
                tifffile.imsave(os.path.join(outpath,ofn+og+'_gtComps.tif'), 
                    dlut[tmp.transpose(2,1,0),:], compress=5)
                # four coloring: xxx - add option
                #clut = emLabels.color(gtComps, c4lut, graySize=9, chromatic=4,
                #    sampling=attrs['scale'] if hasattr(attrs,'scale') else None)
                #tifffile.imsave(os.path.join(outpath,ofn+og+'_gtComps.tif'), 
                #    clut[gtComps.transpose(2,1,0),:], compress=5)
                
                if skeletonize:
                    # gt skeletonized components
                    tmp = gtSkelComps; sel = (tmp > 0); tmp[sel] = tmp[sel] % (dlut.shape[0]-1)
                    tifffile.imsave(os.path.join(outpath,ofn+og+'_gtSkComps.tif'), 
                        dlut[tmp.transpose(2,1,0),:], compress=5)

                # out labels
                cmap = (np.array([[0,0,0], [1,1,1], [0.5,0.5,0.5]], dtype=np.single)*255).astype(np.uint8)
                tifffile.imsave(os.path.join(outpath,ofn+og+('_outLbls_cat%.6f.tif' % cat_error[i,j])),
                    cmap[outLbls.transpose(2,1,0),:],compress=5)

                # components at amin
                tmp = outComps; sel = (tmp > 0); tmp[sel] = tmp[sel] % (dlut.shape[0]-1)
                tifffile.imsave(os.path.join(outpath,ofn+og+'_outComps_min%s_%s_%.6f.tif' % (amin_str,
                    '_'.join(subgroups), min_out[i,j])), dlut[tmp.transpose(2,1,0),:], compress=5)
                # four coloring: xxx - add option
                #nclrs, clut = emLabels.color(outComps, c4lut, graySize=9, chromatic=4,
                #    sampling=attrs['scale'] if hasattr(attrs,'scale') else None)
                #tifffile.imsave(os.path.join(outpath,ofn+og+'_outComps_min%s_%s_%.6f_%dcolors.tif' % (amin_str,
                #    '_'.join(subgroups), min_out[i,j], nclrs)), clut[outComps.transpose(2,1,0),:], compress=5)
                

