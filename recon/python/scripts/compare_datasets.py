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

# Script for comparing different segmentations of the different datasets.


import os, sys
import argparse
import time
import dill
from cycler import cycler
import numpy as np
import numpy.ma as ma
from gala import evaluate as ev

from metrics import adapted_rand_error

from dpLoadh5 import dpLoadh5
#from metrics import warping_error, adapted_rand_error
from typesh5 import emLabels, emProbabilities, emVoxelType

params = {
    'gt_ECS_label' : 1,
    #
    'segpaths' : [
        '/home/watkinspv/Data/agglo', 
        '/Data/datasets/labels/supervoxels/sixfold_threed_20151006',
        '/Data/datasets/labels/supervoxels/sixfold_threed_20151006',
        #'/home/watkinspv/Data/agglo', 
        ],
    'segmentations' : [
        'new_huge_agglo_perfect_supervoxels.h5', 
        'huge_supervoxels.h5', 
        #'none_agglo_perfect_supervoxels.h5', 
        'none_supervoxels.h5', 
        #'huge_flatagglo_lda_23f_35iter_test_supervoxels.h5',
        #'huge_flatagglo_lda_25f_50iter_test_supervoxels.h5',
        #'huge_flatagglo_lda_24f_50iter_test_supervoxels.h5',
        #'huge_flatagglo_lda_27f_50iter_test_supervoxels.h5',
        #'huge_flatagglo_lda_27fn_50iter_test_supervoxels.h5',
        ],
    'seglbls' : [
        'perfect huge',
        'watershed huge',
        #'perfect none',
        'watershed none',
        #'lda_23f',
        #'lda_25f',
        #'lda_24f',
        #'lda_27f',
        #'lda_27fn',
        ],
    'subgroups' : [
        ['with_background',],
        ['with_background',],
        #['with_background',],
        ['with_background',],
        #['agglomeration',],
        ],
    'segparams' : [
        np.array([0.9995]),
        np.array([0.3, 0.4, 0.5, 0.6, 0.61, 0.62, 0.63, 0.64,
            0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72,
            0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,
            0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88,
            0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96,
            0.97, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999,
            0.99995, 0.99999, 0.999995, 0.999999,
            ]),
        #np.array([0.9995]),
        np.array([0.3, 0.4, 0.5, 0.6, 0.61, 0.62, 0.63, 0.64,
            0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72,
            0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,
            0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88,
            0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96,
            0.97, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999,
            0.99995, 0.99999, 0.999995, 0.999999,
            ]),
        #np.array([0.30000000,0.40000000,0.50000000,0.60000000,0.70000000,0.80000000,0.90000000,0.95000000, 0.97500000,
        #    0.99000000,0.99500000,0.99900000,0.99950000,0.99990000,0.99995000,0.99999000,0.99999500,0.99999900]),
        #np.arange(1,35,dtype=np.double),
        #np.arange(1,50,dtype=np.double),
        #np.arange(1,50,dtype=np.double),
        #np.arange(1,50,dtype=np.double),
        ],
    #
    'size' : [128, 128, 128], 'offset' : [0, 0, 0],
    'gth5' : [
        '/Data/datasets/labels/gt/M0007_33_labels_briggmankl_watkinspv_39x35x7chunks_Forder.h5',
        '/Data/datasets/labels/gt/M0007_33_labels_briggmankl_watkinspv_39x35x7chunks_Forder.h5',
        '/Data/datasets/labels/gt/M0027_11_labels_briggmankl_watkinspv_33x37x7chunks_Forder.h5',
        ],
    #'chunks' : [[17,19,2], [17,23,1], [22,23,1], [22,18,1], [22,23,2], [19,22,2]],
    #'chunks' : [[17,19,2], [17,23,1], [22,23,1]],
    #'chunks' : [[17,23,1],],
    'chunks' : [
        [[17,19,2], [17,23,1], [22,23,1], [22,18,1], [22,23,2], [19,22,2]],
        [[17,19,2], [17,23,1], [22,23,1], [22,18,1], [22,23,2], [19,22,2]],
        [[16,17,4], [13,20,3], [13,15,3], [18,15,3], [18,20,3], [18,20,4]],
        ],
    'figno' : 5000,
    'plot_only':True,
    'outpath' : '.',
    'save_file' : 'out.dill',
    'do_plots':True,
    'save_plots':False,
    }

globals().update(params)
nchunks = len(chunks[0]); nsegs = len(segmentations)
assert(nsegs == len(subgroups) and nsegs == len(segparams))
nparams = [len(segparams[i]) for i in range(nsegs)]
mparams = max(nparams)
groups_vals = np.arange(1,nsegs+1,dtype=np.double)
groups_vals_rep = groups_vals[:,None].repeat(nchunks,axis=1)
groups_xlim = [0.5,nsegs+0.5]



if not plot_only:

    metrics = {
        'are_gala' : ma.array(np.zeros((nsegs,nchunks,mparams),dtype=np.double),mask=True),
        'are_precrec_gala' : ma.array(np.zeros((nsegs,nchunks,mparams,2),dtype=np.double),mask=True),
        'split_vi_gala' : ma.array(np.zeros((nsegs,nchunks,mparams,2),dtype=np.double),mask=True),
        #'nlabels_skel' : ma.array(np.zeros((nsegs,nchunks),dtype=np.int64),mask=True),
        #'voxel_sizes_skel' : [[None]*nchunks]*nsegs,
        }                
    globals().update(metrics)


    for i,seg,segp in zip(range(nsegs), segmentations, segpaths):

        for j,chunk in zip(range(nchunks),chunks[i]):
            # xxx - potential savings here for same chunks, currently was not worth the effort relative to load time
            # load ground truth and components from segmented labels file
            loadh5 = emLabels.readLabels(srcfile=gth5[i], chunk=chunk, offset=offset, size=size)
            gtComps = loadh5.data_cube; gtIsECS = (gtComps == gt_ECS_label);
            gtComps[gtIsECS] = 0; n = gtComps.max(); gtComps[gtComps == n] = gt_ECS_label; gtnlabels = n-1
        
            fps = os.path.join(segp, seg)
            print('calculating metrics for ' + seg + (' chunk %d %d %d' % tuple(chunk))); t = time.time()
        
            for k,prm in zip(range(nparams[i]),segparams[i]):
                loadh5 = emLabels.readLabels(srcfile=fps, chunk=chunk, offset=offset, size=size, 
                    subgroups=subgroups[i] + ['%.8f' % (prm,)])
                segComps = loadh5.data_cube

                # calculate the ISBI2013 rand error (gala, excludes gt background) using the full out components
                are, prec, rec = ev.adapted_rand_error(segComps, gtComps, all_stats=True)
                #are, prec, rec = adapted_rand_error( gtComps, segComps, nogtbg=True)
                are_gala[i,j,k] = are; are_precrec_gala[i,j,k,:] = np.array([prec,rec])
    
                # calculate the split variation of information (gala) using full out components
                split_vi_gala[i,j,k,:] = ev.split_vi(segComps, gtComps)
    
            print('\tdone in %.4f s' % (time.time() - t))
            
    # save all metric data using dill
    with open(os.path.join(outpath,save_file), 'wb') as f: dill.dump({'metrics':metrics, 'params':params}, f)
else:
    with open(os.path.join(outpath,save_file), 'rb') as f: d = dill.load(f)
    #globals().update(d); globals().update(metrics)
    globals().update(d['metrics'])




# calculations based on parameters
vi_gala = split_vi_gala.sum(axis=3)

pseg = 1
print('mean VI')
print(vi_gala[pseg,:,:].mean(axis=0))
print('mean ARE')
print(are_gala[pseg,:,:].mean(axis=0))
print('mean ARE 1-rec')
print(1-are_precrec_gala[pseg,:,:,1].mean(axis=0))

# mins across params
min_are_gala = are_gala.min(axis=2)
amin_are_gala = are_gala.argmin(axis=2)
min_vi_gala = vi_gala.min(axis=2)
amin_vi_gala = vi_gala.argmin(axis=2)

print('min are across param')
print(min_are_gala)

def scatter_err_plots(errs, strs, ylims, plsize, dostats, groups_vals_rep_loc=None, jitter=0, doLines=True):
    if groups_vals_rep_loc is None: groups_vals_rep_loc = groups_vals_rep
    for i in range(len(errs)):
        if plot_setlims and ylims: ax = pl.subplot( plsize[0],plsize[1],i+1, adjustable='box', 
            aspect=(groups_xlim[1]-groups_xlim[0])/(ylims[i][1]-ylims[i][0]) )
        else: ax = pl.subplot(plsize[0],plsize[1],i+1)
        x = groups_vals_rep_loc.reshape((-1,));
        if jitter > 0: x = x + np.random.randn(x.size) * jitter;
        if doLines:
            #scalarMap = mpl.cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=nsegs), cmap=plt.get_cmap('Set1'))
            #colorVal = [scalarMap.to_rgba(x) for x in range(nsegs)]
            #ax.set_prop_cycle(cycler('color', colorVal))
            plt.plot(groups_vals,errs[i],'k',marker='o',zorder=1)
        else:
            if errs[i].shape[1] > 10:
                plt.scatter(x,errs[i].reshape((-1,)),marker=u'.',c=u'k',s=1)
            else:
                plt.scatter(x,errs[i].reshape((-1,)),marker=u'o',c=u'k',s=20)
        plt.scatter(groups_vals,np.median(errs[i],axis=1),c=u'r', marker=u'+',s=100, linewidths=2, zorder=2)
        # Hide the right and top spines
        ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left'); ax.xaxis.set_ticks_position('bottom')        
        if plot_setlims and ylims: plt.xlim(groups_xlim); plt.ylim(ylims[i])
        #pl.xlabel(groups_xlbl); 
        pl.ylabel(strs[i])
        plt.xticks(groups_vals,rotation=45); ax.set_xticklabels(seglbls)
        #ax.set_xticks(groups_vals); #ax.set_xticklabels(['10', '100', '1000', '10000', '100000'])
        
        plt.title('med=%s' % (' '.join(['%g' % (x,) for x in np.median(errs[i],axis=1)]),))

if do_plots:
    from matplotlib import pylab as pl
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    from matplotlib import colors

    # blue, green, yellow, red
    clrs = np.array([[0.28235294, 0.23921569, 0.54509804], [0.33333333, 0.41960784, 0.18431373], [1, 0.54901961, 0], 
        [0.54509804, 0, 0]]).T
    
    baseno=figno
    pl.figure(figno);
        
    errs = [min_are_gala, min_vi_gala]
    strs = ['Min Adapted Rand Error', 'Min Variation of Information']
    ylims = [[-0.01,0.12], [-0.1,1.4], ]
    plsize = [1,2]
    dostats = [False, False]
    plot_setlims = False

    scatter_err_plots(errs, strs, ylims, plsize, dostats)

    figno = figno+1; pl.figure(figno);

    errs = [1-are_precrec_gala, split_vi_gala, ]
    amins = [amin_are_gala, amin_vi_gala,  ]
    xstrs = ['1-Recall', 'False Splits (bits)', ]
    ystrs = ['1-Precision', 'False Merges (bits)', ]
    lims = [[[-0.05, 0.6], [-0.012, 0.12]], [[-0.5, 5], [-0.15, 1.5]] ]
    titles = ['Adapted Rand', 'Split Variation of Information',  ]
    plsize = [1,2]
    plot_mean = True
    plot_setlims = False
        
    for i in range(len(errs)):
        if plot_setlims: 
            ax = pl.subplot( plsize[0],plsize[1],i+1, adjustable='box', 
                aspect=(lims[i][0][1]-lims[i][0][0])/(lims[i][1][1]-lims[i][1][0]) )        
            #ax = pl.subplot(plsize[0],plsize[1],i+1, adjustable='box', aspect=1)
        else: 
            ax = pl.subplot(plsize[0],plsize[1],i+1)
        for j in range(nsegs):
            # 'wrp_sltmrg' : ma.array(np.zeros((ngroups,nblocks,nparams,[splits,merges])
            if plot_mean:
                u = np.mean(errs[i][j,:,:,:], axis=0)
                # divide by sqrt(nblocks) to get standard error of the mean
                s = np.std(errs[i][j,:,:,:], axis=0) / np.sqrt(nchunks)
            else:
                u = np.median(errs[i][j,:,:,:], axis=0)
                s = np.median(np.abs(errs[i][j,:,:,:] - u), axis=0) # median absolute deviation
            plt.plot(u[:,1], u[:,0], color=clrs[:,j], marker='x')
            plt.plot(u[:,1]-s[:,1], u[:,0]-s[:,0], color=clrs[:,j], linestyle='dashed')
            plt.plot(u[:,1]+s[:,1], u[:,0]+s[:,0], color=clrs[:,j], linestyle='dashed')
            # plot point at mean of min dist points across blocks
            m = np.zeros((nchunks,2),dtype=np.double)
            for k in range(nchunks): m[k,:] = errs[i][j,k,amins[i][j,k],:]
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
        if plot_setlims: plt.xlim(lims[i][0]); plt.ylim(lims[i][1]); 
        pl.xlabel(xstrs[i]); pl.ylabel(ystrs[i])
        plt.title(titles[i])

    if save_plots:
        print('saving plots')
        for f,i in zip(range(baseno, figno), range(figno-baseno)):
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

