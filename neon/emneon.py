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

import time
import os
import sys
#import shutil
#import tempfile
#import signal

from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator
from math import ceil, sqrt
import numpy as np

from neon.backends import gen_backend
from neon.layers import GeneralizedCost
from neon.optimizers import GradientDescentMomentum, MultiOptimizer
from neon.optimizers import Schedule, PowerSchedule
from neon.transforms import CrossEntropyBinary, CrossEntropyMulti
from neon.models import Model
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser, extract_valid_args

from data.emdata import EMDataIterator, RandomEMDataIterator
from arch.emarch import EMModelArchitecture
from callbacks.emcallbacks import EMEpochCallback
#from optimizers.emoptimizers import DiscreteTauExpSchedule, TauExpSchedule
from transforms.emcost import EMMetric

# xxx - this did not help, see comment in write_output block
## http://softwareramblings.com/2008/06/running-functions-as-threads-in-python.html
## for computations for writing outputs to happen in parallel
#import threading
#class FuncThread(threading.Thread):
#    def __init__(self, target, *args):
#        self._target = target
#        self._args = args
#        threading.Thread.__init__(self)
# 
#    def run(self):
#        self._target(*self._args)

# http://stackoverflow.com/questions/9258602/elegant-pythonic-cumsum
def cumsum(it):
    total = 0
    for x in it:
        total += x
        yield total
        
# add custom command line arguments on top of standard neon arguments
parser = NeonArgparser(__doc__)
# extra arguments controlling model and learning
parser.add_argument('--model_arch', type=str, default='fergus', help='Specify convnet model architecture from arch/')
#parser.add_argument('--rate_decay', type=float, default=0.0, 
#                    help='Learning schedule rate decay time constant (in epochs)')
#parser.add_argument('--rate_freq', type=int, default=0, 
#                    help='Batch frequency to update rate decay (< 1 is once per EM epoch (training macrobatches))')
parser.add_argument('--rate_step', type=float, default=1.0, help='Learning schedule rate step (in emneon epochs)')
parser.add_argument('--epoch_dstep', nargs='*', type=int, default=[], 
                    help='Learning schedule neon delta epochs to adjust rate (use instead of rate_step)')
parser.add_argument('--rate_change', type=float, default=0.5, 
                    help='Learning schedule rate change (occurs each rate_step)')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
parser.add_argument('--rate_init', nargs=2, type=float, default=[0.001, 0.002], 
                    help='Initial learning rates [weight, bias]')
parser.add_argument('--momentum', nargs=2, type=float, default=[0.9, 0.9], 
                    help='Gradient descent learning momentums [weight, bias]')
parser.add_argument('--neon_progress', action="store_true",
                    help='Use neon builtin progress display instead of emneon display')
parser.add_argument('--save_best_path', type=str, default=None, help='Specify save path for best model so far (train)')

# extra arguments controlling data
parser.add_argument('--data_config', type=str, default=None, help='Specify em data configuration ini file')
parser.add_argument('--write_output', type=str, default='', help='File to to write outputs for test batches')
parser.add_argument('--train_range', nargs=2, type=int, default=[1,200], help='emcc2-style training batch range')
parser.add_argument('--test_range', nargs=2, type=int, default=[200001,200002], 
                    help='emcc2-style testing batch range, concatenated into single "neon epoch" (macrobatch)')
parser.add_argument('--image_in_size', type=int, default=None, help='Specify input image size, override .ini')
parser.add_argument('--chunk_skip_list', nargs='*', type=int, default=[], 
                    help='Skip these random EM chunks, usually for test, override .ini')
parser.add_argument('--dim_ordering', type=str, default='xyz', 
                    help='Which reslice ordering for EM provider, override .ini')
parser.add_argument('--nbebuf', type=int, default=2, 
                    help='How many backend buffers to use, 1 for no double buffering (saves gpu memory, slower)')
parser.add_argument('--plot_weight_layer', type=int, default=-1, 
                    help='Plot weights for specified layer (must specify model_file)')
parser.add_argument('--plot_norm_per_filter', action="store_true",
                    help='With plotting weights, normalize each filter over range')
parser.add_argument('--plot_combine_chans', action="store_true",
                    help='With plotting weights, make plot that combines first three channels into colors')
parser.add_argument('--plot_log', action="store_true", help='Plot weights on log scale')
parser.add_argument('--plot_save_path', type=str, default='', help='Path to save weight plots instead of displaying')

# parse the command line arguments (generates the backend)
args = parser.parse_args(gen_be=False)
print('emneon / neon options:'); print(args)

# setup backend
be_args = extract_valid_args(args, gen_backend)
# mutiple gpus accessing the cache dir for autotuning winograd was causing crashes / reboots
#be_args['cache_dir'] = tempfile.mkdtemp()  # create temp dir
be_args['deterministic'] = None  # xxx - why was this set?
be = gen_backend(**be_args)

# xxx - this doesn't work, interrupt is caught by neon for saving the model which then raises KeyboardInterrupt
#def signal_handler(signal, frame):
#    #print('You pressed Ctrl+C!')
#    shutil.rmtree(be_args['cache_dir'])  # delete directory
#signal.signal(signal.SIGINT, signal_handler)

# this function modified from cuda-convnets2 shownet.py
def make_filter_fig(filters, filter_start, fignum, _title, num_filters, combine_chans, FILTERS_PER_ROW=None,
                    plot_border=0.0):
    MAX_ROWS = 24
    filter_chans = 3 if combine_chans else filters.shape[0]
    filter_plot_chans = 1 if combine_chans else filters.shape[0]
    if FILTERS_PER_ROW is None: FILTERS_PER_ROW = min([filters.shape[2]*filter_plot_chans, 16])
    MAX_FILTERS = FILTERS_PER_ROW * MAX_ROWS
    num_colors = filters.shape[0]
    f_per_row = int(ceil(FILTERS_PER_ROW / float(1 if combine_chans else num_colors)))
    filter_end = min(filter_start+MAX_FILTERS, num_filters)
    filter_rows = int(ceil(float(filter_end - filter_start) / f_per_row))

    filters = filters[:filter_chans,:,:]  # only plot 3 channels if combine_chans
    #filter_pixels = filters.shape[1]
    filter_size = int(sqrt(filters.shape[1]))
    fig = pl.figure(fignum)
    fig.text(.5, .95, '%s %dx%d %d chans filters %d-%d' % (_title, filter_size, filter_size, filter_chans, 
                                                           filter_start, filter_end-1), horizontalalignment='center',
                                                           size='xx-large') 
    num_filters = filter_end - filter_start
    if not combine_chans:
        bigpic = np.ones((filter_size * filter_rows + filter_rows + 1, 
                          filter_size*num_colors * f_per_row + f_per_row + 1), dtype=np.single)*plot_border
    else:
        bigpic = np.ones((3, filter_size * filter_rows + filter_rows + 1, 
                          filter_size * f_per_row + f_per_row + 1), dtype=np.single)*plot_border

    for m in xrange(filter_start,filter_end ):
        filter = filters[:,:,m]
        y, x = (m - filter_start) / f_per_row, (m - filter_start) % f_per_row
        if not combine_chans:
            for c in xrange(num_colors):
                filter_pic = filter[c,:].reshape((filter_size,filter_size))
                bigpic[1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                       1 + (1 + filter_size*num_colors) * x + filter_size*c:1 + (1 + filter_size*num_colors) * x + \
                       filter_size*(c+1)] = filter_pic
        else:
            filter_pic = filter.reshape((3, filter_size,filter_size))
            bigpic[:,
                   1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                   1 + (1 + filter_size) * x:1 + (1 + filter_size) * x + filter_size] = filter_pic
            
    pl.xticks([])
    pl.yticks([])
    if not combine_chans:
        pl.imshow(bigpic, cmap=pl.cm.gray, interpolation='nearest')
    else:
        bigpic = bigpic.swapaxes(0,2).swapaxes(0,1)
        pl.imshow(bigpic, interpolation='nearest')        

try:
    # xxx - this is not clean, how to fix?

    # other command line checks
    assert( not args.data_config or os.path.isfile(args.data_config) ) # EMDataParser config file not found
    assert( not args.write_output or args.model_file )  # also specify model file for write_output
    
    if args.model_file:
        print('Loading model file ''%s''' % (args.model_file,))
        model = Model(args.model_file)
    
    if not args.write_output and args.plot_weight_layer < 0:
        # training mode, can start from existing model file or start a new model based on specified architecture    
        
        if args.data_config:
            # initialize the em data parser
            train = EMDataIterator(args.data_config, chunk_skip_list=args.chunk_skip_list, 
                                   dim_ordering=args.dim_ordering, batch_range=args.train_range, name='train', 
                                   NBUF=args.nbebuf, image_in_size=args.image_in_size)
            # test batches need to be concatenated into a single "neon epoch" for built-in neon eval_set to work 
            test = EMDataIterator(args.data_config, chunk_skip_list=args.chunk_skip_list, 
                                  dim_ordering=args.dim_ordering, batch_range=args.test_range, name='test', 
                                  isTest=True, concatenate_batches=True, NBUF=args.nbebuf,
                                  image_in_size=args.image_in_size) if args.callback_args['eval_freq'] else None
        else:
            # make dummy random data just for testing model inits
            train = RandomEMDataIterator(name='train')
            test = RandomEMDataIterator(name='test')
    
        if not args.model_file:
            # create the model based on the architecture specified via command line
            arch = EMModelArchitecture.init_model_arch(args.model_arch, train.parser.nclass, 
                                                       not train.parser.independent_labels)
            model = Model(layers=arch.layers)
    
        assert( train.nmacrobatches > 0 )    # no training batches specified and not in write_output mode
        macro_epoch = model.epoch_index//train.nmacrobatches+1
        macro_batch = model.epoch_index%train.nmacrobatches+1
        if args.data_config and macro_batch > train.batch_range[0]:
            print('Model loaded at model epoch %d, setting to training batch %d' % (model.epoch_index,macro_batch,))
            train.reset_batchnum(macro_batch)
        
        # print out epoch and batch as they were in cuda-convnets2, starting at 1
        print('Training from epoch %d to %d with %d/%d training/testing batches per epoch, %d examples/batch' \
            % (macro_epoch, args.epochs, train.nmacrobatches, test.nmacrobatches if test else 0, 
               train.parser.num_cases_per_batch))
    
        # configure optimizers and weight update schedules
        num_epochs = args.epochs*train.nmacrobatches  # for emneon, an epoch is now a batch, train_range is an epoch

        # old method using exp schedules specified with tau and epoch freq
        ## rate update frequency less than one means update twice per EM epoch (full set of training macrobatches)
        #if args.rate_freq < 1: args.rate_freq = train.nmacrobatches
        #if args.rate_decay > 0:
        #    if args.rate_freq > 1:
        #        weight_sched = DiscreteTauExpSchedule(args.rate_decay * train.nmacrobatches,num_epochs, args.rate_freq)
        #    else:
        #        weight_sched = TauExpSchedule(args.rate_decay * train.nmacrobatches, num_epochs)
        #else:
        #    weight_sched = Schedule()

        # simpler method directly from neon Schedule(), specify step and change on command line
        if len(args.epoch_dstep) > 0:
            epoch_step = list(cumsum(args.epoch_dstep))
            print('Adjusting learning rate by %.4f at %s' % (args.rate_change, ','.join([str(x) for x in epoch_step])))
            weight_sched = Schedule(step_config=epoch_step, change=args.rate_change)
        else:
            weight_sched = PowerSchedule(step_config=int(args.rate_step*train.nmacrobatches), change=args.rate_change)
        
        opt_gdm = GradientDescentMomentum(args.rate_init[0], args.momentum[0], wdecay=args.weight_decay, 
                                          schedule=weight_sched, stochastic_round=args.rounding)
        opt_biases = GradientDescentMomentum(args.rate_init[1], args.momentum[1], 
                                             schedule=weight_sched, stochastic_round=args.rounding)
        opt_fixed = GradientDescentMomentum(0.0, 1.0, wdecay=0.0)
        opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases, 'DOG': opt_fixed})
    
        # configure cost and test metrics
        cost = GeneralizedCost(costfunc=(CrossEntropyBinary() \
            if train.parser.independent_labels else CrossEntropyMulti()))
        metric = EMMetric(oshape=test.parser.oshape, use_softmax=not train.parser.independent_labels) if test else None
    
        # configure callbacks
        if not args.neon_progress: 
            args.callback_args['progress_bar'] = False
        callbacks = Callbacks(model, eval_set=test, metric=metric, **args.callback_args)
        if not args.neon_progress: 
            callbacks.add_callback(EMEpochCallback(args.callback_args['eval_freq'],train.nmacrobatches),insert_pos=None)
        # xxx - thought of making this an option but not clear that it slows anything down?
        #callbacks.add_hist_callback() # xxx - not clear what information this conveys
        if args.save_best_path:
            callbacks.add_save_best_state_callback(args.save_best_path)
        
        model.fit(train, optimizer=opt, num_epochs=num_epochs, cost=cost, callbacks=callbacks)
        print('Model training complete for %d epochs!' % (args.epochs,))
        #test.stop(); train.stop()
    
    elif args.write_output:
        # write_output mode, must have model loaded
            
        if args.data_config:
            test = EMDataIterator(args.data_config, write_output=args.write_output,
                                  chunk_skip_list=args.chunk_skip_list, dim_ordering=args.dim_ordering,
                                  batch_range=args.test_range, name='test', isTest=True, concatenate_batches=False,
                                  NBUF=args.nbebuf, image_in_size=args.image_in_size)
        else:
            # make dummy random data just for testing model inits
            test = RandomEMDataIterator(name='outputs')
            
        print('Model output (forward prop) for %d testing batches, %d examples/batch' % (test.nmacrobatches,
            test.parser.num_cases_per_batch)); 
        feature_path = ''; #out_thread = None
        for i in range(test.nmacrobatches):
            batchnum = test.batch_range[0]+i
            last_batch = i==test.nmacrobatches-1
            sys.stdout.write('%d ... ' % (batchnum,)); t = time.time()
    
            outputs = model.get_outputs(test)

            # serial version            
            test.parser.checkOutputCubes(feature_path, batchnum, last_batch, outputs=outputs)

            # xxx - could not get any improvment here, possibly overhead of the thread is longer than the
            #   computations for formatting the outputs (the h5 write itself is parallelized in the parser)            
            ## parallel version
            #if out_thread: out_thread.join()
            #out_thread = FuncThread(test.parser.checkOutputCubes, feature_path, batchnum, last_batch, outputs)
            #out_thread.start()
            #if last_batch: out_thread.join()

            sys.stdout.write('\tdone in %.2f s\n' % (time.time() - t,)); sys.stdout.flush()
            
        if args.data_config:
            print('Model output complete for %d test batches!' % (test.nmacrobatches,))
        else:
            print('WARNING: DummyEMDataParser does not actually write outputs, forward prop only')
        #test.stop();

    elif args.plot_weight_layer >= 0:
        # plot filters mode, must have model loaded
        assert(args.model_file) # specify model to plot filters for
        
        from matplotlib import pylab as pl
        from matplotlib import pyplot as plt
        #from matplotlib import cm

        # get the specified layer from the loaded model and copy the weights out
        layer = model.layers.layers[args.plot_weight_layer]; filters = layer.W.get()
        
        # format to plot with cuda-convents2 shownet.py code        
        fsize = reduce(operator.mul, layer.fshape, 1)
        filters = filters.reshape([filters.size/fsize, layer.fshape[0]**2, layer.fshape[2]])

        baseno = 1000; figno = baseno
        print('Plotting weights at layer %d ''%s'' %s' % (args.plot_weight_layer,layer.name, 
                                                          'log scale' if args.plot_log else ''))
        if args.plot_norm_per_filter:
            filters -= filters.min(axis=1,keepdims=True)
            filters /= filters.max(axis=1,keepdims=True)
        else:
            filters -= filters.min()
            filters /= filters.max()
        if args.plot_log:
            filters = np.log10(filters+filters[filters!=0].min())            
            if args.plot_norm_per_filter:
                filters -= filters.min(axis=1,keepdims=True)
                filters /= filters.max(axis=1,keepdims=True)
            else:
                filters -= filters.min()
                filters /= filters.max()
        filter_start = 0; num_filters = filters.shape[2]
        make_filter_fig(filters, filter_start, figno, 'Layer %s' % layer.name, num_filters, args.plot_combine_chans)
        figno += 1

        if args.plot_save_path:
            print('Saving plots')
            fignames = ['layer_%d_weights' % args.plot_weight_layer]
            for f,i in zip(range(baseno, figno), range(figno-baseno)):
                print('Exporting ',f,i,fignames[i])
                pl.figure(f)
                figure = plt.gcf() # get current figure
                figure.set_size_inches(20, 20)
                plt.savefig(os.path.join(args.plot_save_path, fignames[i] + '.png'), dpi=72)
                plt.savefig(os.path.join(args.plot_save_path, fignames[i] + '.eps'))
        else:
            pl.show()
    
    #shutil.rmtree(be_args['cache_dir'])  # delete directory

except KeyboardInterrupt:
    # xxx - this is not clean, how to fix?
    print('Killed with Ctrl-C, cleaning up')

    #    # xxx - consider going back to normal threads instead of daemon threads in emdata?
    #    #   this would prevent the "multiple process" situation
    #    try:
    #        shutil.rmtree(be_args['cache_dir'])  # delete directory
    #    except:
    #        pass
    
