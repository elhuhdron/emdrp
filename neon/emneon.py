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
import shutil
import tempfile
#import cPickle as myPickle
import pickle as myPickle

from neon.layers import GeneralizedCost
from neon.optimizers import GradientDescentMomentum, MultiOptimizer
from neon.transforms import CrossEntropyBinary, CrossEntropyMulti
from neon.models import Model
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

from data.emdata import EMDataIterator, RandomEMDataIterator
from arch.emarch import EMModelArchitecture
from callbacks.emcallbacks import EMEpochCallback
from optimizers.emoptimizers import DiscreteTauExpSchedule, TauExpSchedule
from transforms.emcost import EMMetric

# stole from cuda-convnets2 python_util/util.py
def pickle(filename, data):
    fo = filename
    if type(filename) == str:
        fo = open(filename, "w")
    
    myPickle.dump(data, fo, protocol=myPickle.HIGHEST_PROTOCOL)
    fo.close()

# add custom command line arguments on top of standard neon arguments
parser = NeonArgparser(__doc__)
# extra arguments controlling model and learning
parser.add_argument('--model_arch', type=str, default='fergus', help='Specify convnet model architecture from arch/')
parser.add_argument('--rate_decay', type=float, default=3.0, 
                    help='Learning schedule rate decay time constant (in epochs)')
parser.add_argument('--rate_freq', type=int, default=0, 
                    help='Batch frequency to update rate decay (< 1 is 2 times per EM epoch (training macrobatches))')
parser.add_argument('--weight_decay', type=float, default=0.02, help='Weight decay')
parser.add_argument('--rate_init', nargs=2, type=float, default=[0.0001, 0.0002], 
                    help='Initial learning rates [weight, bias]')
parser.add_argument('--momentum', nargs=2, type=float, default=[0.9, 0.9], 
                    help='Gradient descent learning momentums [weight, bias]')
parser.add_argument('--neon_progress', action="store_true",
                    help='Use neon builtin progress display instead of emneon display')
# extra arguments controlling data
parser.add_argument('--data_config', type=str, default=None, help='Specify em data configuration ini file')
parser.add_argument('--write_output', type=str, default='', help='File to to write outputs for test batches')
parser.add_argument('--train_range', nargs=2, type=int, default=[1,200], help='emcc2-style training batch range')
parser.add_argument('--test_range', nargs=2, type=int, default=[200001,200002], 
                    help='emcc2-style testing batch range, concatenated into single "neon epoch" (macrobatch)')
parser.add_argument('--chunk_skip_list', nargs='*', type=int, default=[], 
                    help='Skip these random EM chunks, usually for test, override .ini')
parser.add_argument('--dim_ordering', type=str, default='xyz', 
                    help='Which reslice ordering for EM provider, override .ini')
parser.add_argument('--nbebuf', type=int, default=2, 
                    help='How many backend buffers to use, 1 for no double buffering (saves gpu memory, slower)')

# parse the command line arguments (generates the backend)
args = parser.parse_args()
print('emneon / neon options:'); print(args)

# other command line checks
assert( not args.data_config or os.path.isfile(args.data_config) ) # EMDataParser config file not found
assert( not args.write_output or args.model_file )  # also specify model file for write_output

if args.model_file:
    print('Loading model file ''%s''' % (args.model_file,))
    model = Model(args.model_file)

if not args.write_output:
    # training mode, can start from existing model file or start a new model based on specified architecture    
    
    if args.data_config:
        # initialize the em data parser
        train = EMDataIterator(args.data_config, chunk_skip_list=args.chunk_skip_list, dim_ordering=args.dim_ordering,
                               batch_range=args.train_range, name='train', NBUF=args.nbebuf)
        # test batches need to be concatenated into a single "neon epoch" for built-in neon eval_set to work correctly
        test = EMDataIterator(args.data_config, chunk_skip_list=args.chunk_skip_list, dim_ordering=args.dim_ordering,
                              batch_range=args.test_range, name='test', isTest=True, NBUF=args.nbebuf)
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
        % (macro_epoch, args.epochs, train.nmacrobatches, test.nmacrobatches, train.parser.num_cases_per_batch))

    # configure optimizers and weight update schedules
    num_epochs = args.epochs*train.nmacrobatches  # for emneon, an epoch is now a batch, train_range is an epoch
    # rate update frequency less than one means update twice per EM epoch (full set of training macrobatches)
    if args.rate_freq < 1: args.rate_freq = train.nmacrobatches//2
    if args.rate_freq > 1:
        weight_sched = DiscreteTauExpSchedule(args.rate_decay * train.nmacrobatches, num_epochs, args.rate_freq)
    else:
        weight_sched = TauExpSchedule(args.rate_decay * train.nmacrobatches, num_epochs)
    opt_gdm = GradientDescentMomentum(args.rate_init[0], args.momentum[0], wdecay=args.weight_decay, 
                                      schedule=weight_sched, stochastic_round=args.rounding)
    opt_biases = GradientDescentMomentum(args.rate_init[1], args.momentum[1], 
                                         schedule=weight_sched, stochastic_round=args.rounding)
    opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})

    # configure cost and test metrics
    cost = GeneralizedCost(costfunc=(CrossEntropyBinary() if train.parser.independent_labels else CrossEntropyMulti()))
    metric = EMMetric(oshape=test.parser.oshape, use_softmax=not train.parser.independent_labels)

    # configure callbacks
    if not args.neon_progress: args.callback_args['progress_bar'] = False
    if not args.callback_args['eval_freq']: args.callback_args['eval_freq'] = train.nmacrobatches
    callbacks = Callbacks(model, eval_set=test, metric=metric, **args.callback_args)
    if not args.neon_progress: 
        callbacks.add_callback(EMEpochCallback(args.callback_args['eval_freq'],train.nmacrobatches), insert_pos=None)
    # xxx - thought of making this an option but not clear that it slows anything down?
    callbacks.add_hist_callback()
    
    model.fit(train, optimizer=opt, num_epochs=num_epochs, cost=cost, callbacks=callbacks)
    print('Model training complete for %d epochs!' % (args.epochs,))
    #test.stop(); train.stop()

else:
    # write_output mode, must have model loaded
        
    if args.data_config:
        test = EMDataIterator(args.data_config, write_output=args.write_output,
                              chunk_skip_list=args.chunk_skip_list, dim_ordering=args.dim_ordering,
                              batch_range=args.test_range, name='test', isTest=True, NBUF=args.nbebuf)
    else:
        # make dummy random data just for testing model inits
        test = RandomEMDataIterator(name='outputs')
        
    print('Model output (forward prop) for %d testing batches, %d examples/batch' % (test.nmacrobatches,
        test.parser.num_cases_per_batch)); 
    feature_path = tempfile.mkdtemp()  # create temp dir
    for i in range(test.nmacrobatches):
        batchnum = test.batch_range[0]+i
        sys.stdout.write('%d ... ' % (batchnum,)); t = time.time()

        # xxx - for now just write pickled batches to a temp folder, just as feature writer in cuda-convnets2 does
        outputs = model.get_outputs(test)
        path_out = os.path.join(feature_path, 'data_batch_%d' % (batchnum,))
        pickle(path_out, {'data': outputs})
        
        test.parser.checkOutputCubes(feature_path, batchnum, i==test.nmacrobatches-1)
        sys.stdout.write('\tdone in %.2f s\n' % (time.time() - t,))
        sys.stdout.flush()
    shutil.rmtree(feature_path)  # delete directory
        
    if args.data_config:
        print('Model output complete for %d test batches!' % (test.nmacrobatches,))
    else:
        print('WARNING: DummyEMDataParser does not actually write outputs, forward prop only')
    #test.stop();
    
