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

import numpy as np
from neon.transforms.cost import Metric

class EMMetric(Metric):

    """
    Compute the EM specific metrics
    """

    def __init__(self, oshape=None, use_softmax=False):
        if not oshape:
            oshape = [1,2]
        elif len(oshape)==2:
            self.oshape = list(oshape)
        else:
            self.oshape = [oshape[0]*oshape[1], oshape[2]]
        self.nclass = self.oshape[1]
        self.use_softmax = use_softmax
        
        #        self.predictions = self.be.iobuf(1)
        #        self.targets = self.be.iobuf(1)

        self.class_error = self.be.iobuf(1)  # Contains per record metric
        self.log_prob = self.be.iobuf(1)  # Contains per record metric

        self.log_name = 'CrossEntropyMulti' if use_softmax else 'CrossEntropyBinary'

        self.metric_names = ['ClassificationError', self.log_name]

    def __call__(self, y, t, calcrange=slice(0, None)):
        """
        Compute the accuracy metric

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            numpy ary : Returns the metrics in numpy array,
                        [ClassificationError CrossEntropy]
        """
        # xxx -  really want to do something like this where argmax is only over the classes, but
        #   neon won't do an argmax with more than 2 dimensions:
        # ValueError: Operations that are not simple elementwise are only currently supported in 2 dimensions.
        #        # calculates mean class error (over nclass) over all output pixels (noutputs)
        #        self.predictions[:] = self.be.argmax(y.reshape(self.oshape+[-1]), axis=1)
        #        self.targets[:] = self.be.argmax(t.reshape(self.oshape+[-1]), axis=1)
        #        self.class_error[:] = self.be.not_equal(self.predictions, self.targets).mean(axis=0)

        # instead just sum all correct or not-correct as if all outputs were completely independent
        #self.class_accuracy[:] = self.be.mean((y > 0.5) * t + (y <= 0.5) * (1 - t), axis=0)  
        self.class_error[:] = self.be.mean((y <= 0.5) * t + (y > 0.5) * (1 - t), axis=0)

        # calculates CrossEntropy (Multi or Binary depending on use_softmax) summed over all outputs
        log_tgt = - self.be.safelog(y) * t
        if self.use_softmax:
            self.log_prob[:] = self.be.sum(log_tgt, axis=0)
        else:
            self.log_prob[:] = self.be.sum(log_tgt - self.be.safelog(1 - y) * (1 - t), axis=0)

        return np.array((self.class_error.get()[:, calcrange].mean(),
                         self.log_prob.get()[:, calcrange].mean()))
                         