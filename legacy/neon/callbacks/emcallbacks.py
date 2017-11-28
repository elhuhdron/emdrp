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
import sys

from neon.callbacks.callbacks import Callback

class EMEpochCallback(Callback):
    """
    Callback for custom printout of EM information for each neon "epoch"
      which is hijacked as an EM macrobatch.

    """
    def __init__(self, test_epoch_freq, nmacrobatches):
        self.epoch_freq = 1
        super(EMEpochCallback, self).__init__(epoch_freq=self.epoch_freq)
        self.test_epoch_freq = test_epoch_freq
        self.nmacrobatches = nmacrobatches
        #self.cur_minibatch_index = 0
        self.mini_batches_per_epoch = None
        self.running_epoch = 0 

    # always print out the model for the log
    def on_train_begin(self, callback_data, model, epochs):
        print(model)
    
    def on_minibatch_begin(self, callback_data, model, epoch, minibatch):
        self.cur_minibatch_index = minibatch
        
    def on_epoch_begin(self, callback_data, model, epoch):
        self.epoch_start = time.time()
        macro_epoch = epoch//self.nmacrobatches
        # print out epoch and batch as they were in cuda-convnets2, starting at 1
        if epoch % self.nmacrobatches == 0:
            sys.stdout.write('========= Begin epoch %d =========\n' % (macro_epoch+1,))
        batch = epoch % self.nmacrobatches
        if (epoch + 1) % self.epoch_freq == 0:
            if self.test_epoch_freq and (epoch + 1) % self.test_epoch_freq == 0:
                sys.stdout.write('%d.%d (test)... ' % (macro_epoch+1,batch+1,))
            else:
                sys.stdout.write('%d.%d ... ' % (macro_epoch+1,batch+1,))
        sys.stdout.flush()

    def on_epoch_end(self, callback_data, model, epoch):
        t = time.time() - self.epoch_start
        # this is because could not find where neon can load h5 data if training is continuing
        epoch = self.running_epoch; self.running_epoch += 1
        if (epoch + 1) % self.epoch_freq == 0:
            sys.stdout.write('done in %.2f s ' % t)

            if self.mini_batches_per_epoch is None: 
                self.mini_batches_per_epoch = self.cur_minibatch_index+1
            loss = callback_data['cost']['train']\
                [epoch*self.mini_batches_per_epoch:(epoch+1)*self.mini_batches_per_epoch].mean()
            progress_string = " [%s %.5f]" % (model.cost.costfunc.__class__.__name__, loss)
            sys.stdout.write(progress_string)

            if self.test_epoch_freq and (epoch + 1) % self.test_epoch_freq == 0:
                assert( 'metrics' in callback_data )    # EM epoch callback requires metric callbacks
                sys.stdout.write('\n\t')
                for met in callback_data['metrics'].keys():
                    sys.stdout.write(' [%s %.5f]' % (met, callback_data['metrics'][met][epoch//self.test_epoch_freq]))

            sys.stdout.write('\n')
            sys.stdout.flush()
