
# quick script for plotting loglog test versus train error for trained convnets.

import os, sys
import numpy as np
from matplotlib import pylab as pl
import matplotlib as plt
import time
from math import sqrt, ceil, floor
sys.path.insert(0, '/home/watkinspv/workspace_eclipse/ctome_server/cuda-convnet2')
from python_util.gpumodel import IGPUModel
import scipy.ndimage.filters as filters
from scipy import interpolate

def get_errors(cp, show_cost, cost_idx, norm=False, smooth=[0,0], interp_test=True):
    ms = cp['model_state']
    layers = ms['layers']
    op = cp['op'].options
    
    if show_cost not in ms['train_outputs'][0][0]:
        raise ShowNetError("Cost function with name '%s' not defined by given convnet." % show_cost)
    train_errors = [eval(layers[show_cost]['outputFilter'])(o[0][show_cost], o[1])[cost_idx] \
        / (layers['labels']['outputs'] if norm else 1) for o in ms['train_outputs']]
    test_errors = [eval(layers[show_cost]['outputFilter'])(o[0][show_cost], o[1])[cost_idx] \
        / (layers['labels']['outputs'] if norm else 1) for o in ms['test_outputs']]

    #if smooth_test_errors:
    #    test_errors = [sum(test_errors[max(0,i-len(op['test_batch_range'].value)):i])/\
    #        (i-max(0,i-len(op['test_batch_range'].value))) for i in xrange(1,len(test_errors)+1)]
    if smooth[0]:
        te = np.array(test_errors,dtype=np.double)
        Wte = np.ones(smooth[0], dtype=np.double) / smooth[0]
        te = filters.convolve(te, Wte, mode='reflect', cval=0.0, origin=0)
        test_errors = te.tolist()
    if smooth[1]:
        tr = np.array(train_errors,dtype=np.double)
        Wtr = np.ones(smooth[1], dtype=np.double) / smooth[1]
        tr = filters.convolve(tr, Wtr, mode='reflect', cval=0.0, origin=0)
        train_errors = tr.tolist()

    if interp_test:
        x = np.arange(0, len(test_errors))
        xi = np.linspace(0, len(test_errors)-1, len(train_errors))
        te = np.array(test_errors,dtype=np.double)
        f = interpolate.interp1d(x, te, kind='linear')    
        test_errors = f(xi).tolist()

    if not interp_test:
        test_errors = np.row_stack(test_errors)
        test_errors = np.tile(test_errors, (1, op['testing_freq'].value))
        test_errors = list(test_errors.flatten())
        test_errors += [test_errors[-1]] * max(0,len(train_errors) - len(test_errors))
        test_errors = test_errors[:len(train_errors)]
    
    numbatches = len(op['train_batch_range'].value)
    return numbatches, train_errors, test_errors

def plot_errors(numbatches, train_errors, test_errors):
    numepochs = len(train_errors) / float(numbatches)
    pl.figure(1)
    x = range(0, len(train_errors))
    pl.plot(x, train_errors, 'k-', label='Training set')
    pl.plot(x, test_errors, 'r-', label='Test set')
    pl.legend()
    ticklocs = range(numbatches, len(train_errors) - len(train_errors) % numbatches + 1, numbatches)
    epoch_label_gran = int(ceil(numepochs / 20.)) 
    epoch_label_gran = int(ceil(float(epoch_label_gran) / 10) * 10) if numepochs >= 10 else epoch_label_gran 
    ticklabels = map(lambda x: str((x[1] / numbatches)) if x[0] % epoch_label_gran == epoch_label_gran-1 else '', enumerate(ticklocs))

    pl.xticks(ticklocs, ticklabels)
    pl.xlabel('Epoch')
    #pl.title('%s[%d]' % (show_cost, cost_idx))
    pl.show()

if __name__ == '__main__':
    dolog = True
    show_cost = 'logprob'
    #show_cost = 'sqdiff'
    #ncosts = 1
    ncosts = 2
    basefig = 10
    smooth = [5, 31]   # good for supervised
    #smooth = [0, 31]  # good for autoencoders
    #smooth = [0, 0]
    dointerp = True
    donames = False

    avg_groups = ['warp','nowarp']
    navg_groups = len(avg_groups)
    
    if not donames:
        load_convnets = sys.argv[1:]    # just the paths
    else:
        load_convnets = sys.argv[1::2]; legend_names = sys.argv[2::2]    # the paths and a shorter legend name
    
    print 'Loading convnets ' + ', '.join(load_convnets)
    
    for cost_idx in range(ncosts):
        pl.figure(cost_idx + basefig)
        if navg_groups > 1: allte = None; alltr = None
        for cv,i in zip(load_convnets, range(len(load_convnets))):
            cp = IGPUModel.load_checkpoint(cv)
            numbatches, train_errors, test_errors = get_errors(cp, show_cost,cost_idx, norm=cost_idx == 0,
                smooth=smooth, interp_test=dointerp)
            #plot_errors(numbatches, train_errors, test_errors)
            tr = np.array(train_errors, dtype=np.double); te = np.array(test_errors, dtype=np.double)
            if navg_groups > 1:
                if allte is None:
                    allte = np.zeros(list(te.shape)+[len(load_convnets)], dtype=np.double)
                    alltr = np.zeros(list(tr.shape)+[len(load_convnets)], dtype=np.double)
                allte[:,i] = te; alltr[:,i] = tr
            else:
                if dolog: tr = np.log10(tr); te = np.log10(te)
                pl.plot(tr, te, label=os.path.basename(legend_names[i] if donames else os.path.normpath(cv)))
        if navg_groups > 1:
            groups = np.arange(len(load_convnets),dtype=np.int64)
            groups = groups.reshape([navg_groups,len(load_convnets)//navg_groups])
            for i in range(navg_groups):
                tr, te = alltr[:,groups[i,:]].mean(axis=1), allte[:,groups[i,:]].mean(axis=1)
                if dolog: tr = np.log10(tr); te = np.log10(te); 
                pl.plot(tr, te, label=avg_groups[i])
                print allte[:,groups[i,:]].min(axis=0)
                print allte[-1,groups[i,:]]
        pl.legend(loc=4)
        #pl.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.)
        #pl.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        if dolog:
            pl.xlabel('log10 train error'); pl.ylabel('log10 test error')
        else:
            pl.xlabel('train error'); pl.ylabel('test error')
        # xxx - hack job to get correct name in there... better method? not really stored anywhere in convnet
        #   does not work for sqdiff or other cost layers, create some kind of lookup for each cost layer?
        pl.title('%s[%d] %s' % (show_cost, cost_idx, 'logprob' if cost_idx==0 else 'caterror'))
    pl.show()
    

