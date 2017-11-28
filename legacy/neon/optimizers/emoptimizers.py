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

from math import expm1
from neon.optimizers import ExpSchedule

class TauExpSchedule(ExpSchedule):
    """
    Exponential learning rate schedule.

    Arguments:
        tau (float): exponential decay time constant
    """
    def __init__(self, tau, nepochs):
        # set decay so that true exponential decay and neon "exponential" decay match after nepochs
        super(TauExpSchedule, self).__init__(expm1(nepochs/float(tau))/nepochs)
        self.tau = tau; self.nepochs = nepochs

def round_to(n, precision):
    correction = 0.5 if n >= 0 else -0.5
    return int( n/precision+correction ) * precision

class DiscreteTauExpSchedule(TauExpSchedule):
    """
    Discrete Exponential learning rate schedule based using tau with target after nepochs.

    Arguments:
        tau (float): exponential decay time constant
        nepochs (int): number of total epochs to calculate final target rate
        epoch_freq (int): discretization in epochs for calculating rate
    """
    def __init__(self, tau, nepochs, epoch_freq):
        super(DiscreteTauExpSchedule, self).__init__(tau, nepochs)
        self.epoch_freq = int(epoch_freq)

    def get_learning_rate(self, learning_rate, epoch):
        return ExpSchedule.get_learning_rate(self, learning_rate, round_to(epoch,self.epoch_freq))
