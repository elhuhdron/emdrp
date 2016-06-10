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

from neon.layers.layer import Convolution
from neon.initializers.initializer import Initializer
import math
import numpy as np

class DOGInit(Initializer):
    """
    A class for initializing parameter tensors with a single value.

    Args:
        val (float, optional): The value to assign to all tensor elements
    """
    def __init__(self, DOG, name="DOGInit"):
        super(DOGInit, self).__init__(name=name)
        self.DOG = DOG

    def fill(self, param):
        shape = self.DOG.fshape[0]
        b = -(shape//2); e = shape//2 + 1 if shape%2 == 1 else shape//2
        x,y = np.meshgrid(np.arange(b,e, dtype=np.double), np.arange(b,e, dtype=np.double))
        out = np.zeros(param.shape, param.dtype)
        # https://en.wikipedia.org/wiki/Difference_of_Gaussians
        for n in range(self.DOG.nchans):
            sigma_small2 = 2*self.DOG.sigma[n]**2
            sigma_large2 = 2*self.DOG.KD[n]**2*self.DOG.sigma[n]**2
            exponent_small = (x**2 + y**2) / sigma_small2
            exponent_large = (x**2 + y**2) / sigma_large2
            amplitude_small = 1.0/sigma_small2/np.pi
            amplitude_large = 1.0/sigma_large2/np.pi
            z = amplitude_small*np.exp(-exponent_small) - amplitude_large*np.exp(-exponent_large)
            out[:,n] = z.reshape(-1)
        param[:] = out
        
class DOG(Convolution):

    """
    Difference of Gaussians (DOG) layer implementation.

    Arguments:
        nchans (int): number of DOG channels
        sigmas (tuple of floats): initial standard deviation for DOG kernel
        K: intial large sigma ratio for DOG kernel
        name (str, optional): layer name. Defaults to "DOGLayer"
    """

    # constructor and initialize buffers
    def __init__(self, nchans, sigma, KD, name=None):
        self.nchans = nchans
        assert(len(sigma)==self.nchans)
        assert(len(KD)==self.nchans)
        self.sigma = sigma
        self.KD = KD
        shape = max([2*int(math.ceil(2*s*k))+1 for s,k in zip(sigma,KD)])
        if shape < 3: shape = 3
        #padding = shape//2
        padding = 0  # crop off edges
        super(DOG, self).__init__((shape,shape,self.nchans), strides=1, padding=padding, 
              init=DOGInit(self), bsum=False, name=name)

    def __str__(self):
        spatial_dim = len(self.in_shape[1:])
        spatial_str = "%d x (" + "x".join(("%d",) * spatial_dim) + ")"
        padstr_str = ",".join(("%d",) * spatial_dim)
        padstr_dim = ([] if spatial_dim == 2 else ['d']) + ['h', 'w']

        pad_tuple = tuple(self.convparams[k] for k in ['pad_' + d for d in padstr_dim])
        str_tuple = tuple(self.convparams[k] for k in ['str_' + d for d in padstr_dim])

        fmt_tuple = (self.name,) + self.in_shape + self.out_shape + pad_tuple + str_tuple
        fmt_string = "DOG Layer '%s': " + \
                     spatial_str + " inputs, " + spatial_str + " outputs, " + \
                     padstr_str + " padding, " + padstr_str + " stride, " + \
                     ','.join([str(x) for x in self.sigma]) + ' sigmas, ' + \
                     ','.join([str(x) for x in self.KD]) + ' ratios'

        return ((fmt_string % fmt_tuple))
