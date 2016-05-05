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

from neon.layers import GeneralizedCost
from neon.optimizers import GradientDescentMomentum, MultiOptimizer #, ExpSchedule
from neon.transforms import CrossEntropyBinary, CrossEntropyMulti
from neon.transforms import LogLoss # TopKMisclassification, Misclassification
from neon.models import Model
from neon.callbacks.callbacks import Callbacks #, MetricCallback
from neon.util.argparser import NeonArgparser

from data.emdata import EMDataIterator, RandomEMDataIterator
from arch.emarch import EMModelArchitecture
from callbacks.emcallbacks import EMEpochCallback
from optimizers.emoptimizers import DiscreteTauExpSchedule, TauExpSchedule

# add custom command line arguments on top of standard neon arguments
parser = NeonArgparser(__doc__)
# extra arguments controlling model and learning
parser.add_argument('--model_arch', type=str, default='fergus', help='Specify convnet model architecture from arch/')
parser.add_argument('--rate_decay', type=float, default=5.0, 
                    help='Learning schedule rate decay time constant (in epochs)')
parser.add_argument('--rate_freq', type=int, default=40, 
                    help='Batch frequency to update rate decay (<= 1 means per batch)')
parser.add_argument('--weight_decay', type=float, default=0.02, help='Weight decay')
parser.add_argument('--rate_init', nargs=2, type=float, default=[0.001, 0.002], 
                    help='Initial learning rates [weight, bias]')
parser.add_argument('--momentum', nargs=2, type=float, default=[0.9, 0.9], 
                    help='Gradient descent learning momentums [weight, bias]')
parser.add_argument('--neon_progress', action="store_true",
                    help='Use neon builtin progress display instead of emneon display')
# extra arguments controlling data
parser.add_argument('--data_config', type=str, default=None, help='Specify em data configuration ini file')
parser.add_argument('--write_output', type=str, default='', help='File to to write outputs for test batches')
parser.add_argument('--train_range', nargs=2, type=int, default=[1,200], help='emcc2-style training batch range')
parser.add_argument('--test_range', nargs=2, type=int, default=[200001,200002], help='emcc2-style testing batch range')
parser.add_argument('--chunk_skip_list', nargs='*', type=int, default=[], 
                    help='Skip these random EM chunks, usually for test, override .ini')
parser.add_argument('--dim-ordering', type=str, default='xyz', 
                    help='Which reslice ordering for EM provider, override .ini')

# parse the command line arguments (generates the backend)
args = parser.parse_args()
print('emneon / neon options:'); print(args)

if args.data_config:
    # initialize the em data parser
    train = EMDataIterator(args.data_config, chunk_skip_list=args.chunk_skip_list, dim_ordering=args.dim_ordering,
                           batch_range=args.train_range, name='train')
    test = EMDataIterator(args.data_config, chunk_skip_list=args.chunk_skip_list, dim_ordering=args.dim_ordering,
                          batch_range=args.test_range, name='test')
else:
    # make dummy random data just for testing model inits
    train = RandomEMDataIterator(name='train')
    test = RandomEMDataIterator(name='test')

if args.model_file:
    print('Loading model file ''%s''' % (args.model_file,))
    model = Model(args.model_file)
else:
    # create the model based on the architecture specified via command line
    arch = EMModelArchitecture.init_model_arch(args.model_arch, train.nclass, 
                                               not train.parser.independent_labels)
    model = Model(layers=arch.layers)

if train.nmacrobatches > 0:
    macro_epoch = model.epoch_index//train.nmacrobatches+1
    macro_batch = model.epoch_index%train.nmacrobatches+1
    if args.data_config and macro_batch > train.batch_range[0]:
        print('Model loaded at model epoch %d, setting to training batch %d' % (model.epoch_index,macro_batch,))
        train.set_batchnum(macro_batch)
    
    # print out epoch and batch as they were in cuda-convnets2, starting at 1
    print('Training from epoch %d to %d with %d/%d training/testing batches per epoch, %d examples/batch' \
        % (macro_epoch, args.epochs, train.nmacrobatches, test.nmacrobatches, train.parser.num_cases_per_batch))

    # configure optimizers and weight update schedules
    num_epochs = args.epochs*train.nmacrobatches  # for emneon, an epoch is now a batch, train_range is an epoch
    if args.rate_freq > 1:
        weight_sched = DiscreteTauExpSchedule(args.rate_decay * train.nmacrobatches, num_epochs, args.rate_freq)
    else:
        weight_sched = TauExpSchedule(args.rate_decay * train.nmacrobatches, num_epochs)
    opt_gdm = GradientDescentMomentum(args.rate_init[0], args.momentum[0], wdecay=args.weight_decay, 
                                      schedule=weight_sched, stochastic_round=args.rounding)
    opt_biases = GradientDescentMomentum(args.rate_init[1], args.momentum[1], 
                                         schedule=weight_sched, stochastic_round=args.rounding)
    opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})

    # configure callbacks
    if not args.neon_progress: args.callback_args['progress_bar'] = False
    if not args.callback_args['eval_freq']: args.callback_args['eval_freq'] = train.nmacrobatches
    callbacks = Callbacks(model, eval_set=test, metric=LogLoss(), **args.callback_args)
    if not args.neon_progress: 
        callbacks.add_callback(EMEpochCallback(args.callback_args['eval_freq'],train.nmacrobatches), insert_pos=None)
    
    # configure cost and train model, xxx - add option to use multinomial for softmax with not independent_labels
    cost = GeneralizedCost(costfunc=(CrossEntropyBinary() if train.parser.independent_labels else CrossEntropyMulti()))
    model.fit(train, optimizer=opt, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

# optionally write outputs from the trained model for the testing batches
if args.write_output:
    del test, train
    if args.data_config:
        assert(False) # xxx - unimplemented, fix me
        test = EMDataIterator(args.data_config)
    else:
        # make dummy random data just for testing model inits
        test = RandomEMDataIterator(name='outputs')
    print('Forward prop for %d testing batches, %d examples/batch' % (test.nmacrobatches,
        test.parser.num_cases_per_batch)); t = time.time()
    outputs = model.get_outputs(test)
    print('\tdone in %.4f s' % (time.time() - t,))
    if args.data_config:
        pass
    else:
        print('WARNING: DummyEMDataParser does not actually write outputs, forward prop only')
