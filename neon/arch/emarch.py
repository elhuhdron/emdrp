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

from neon.initializers import Constant, Gaussian, Uniform
from neon.layers import Conv, Dropout, Pooling, Affine, LRN
from neon.transforms import Rectlin, Logistic, Softmax, Identity, Explin

class EMModelArchitecture(object):
    def __init__(self, noutputs, use_softmax):
        self.noutputs = noutputs
        self.use_softmax = use_softmax

    @property
    def layers(self):
        raise NotImplemented()

    @staticmethod
    def init_model_arch(name, noutputs, use_softmax):
        # instantiate the model with class given by name string
        return globals()[name](noutputs, use_softmax)

class fergus(EMModelArchitecture):
    def __init__(self, noutputs, use_softmax=False):
        super(fergus, self).__init__(noutputs, use_softmax)

    @property
    def layers(self):
        return [
            Conv((7, 7, 96), init=Gaussian(scale=0.01), bias=Constant(0), activation=Rectlin(), 
                 padding=3, strides=1),
            LRN(31, ascale=0.001, bpower=0.75),
            Pooling(3, strides=2),
            Conv((5, 5, 256), init=Gaussian(scale=0.01), bias=Constant(0), activation=Rectlin(), 
                 padding=2, strides=1),
            LRN(31, ascale=0.001, bpower=0.75),
            Pooling(3, strides=2),
            Conv((3, 3, 384), init=Gaussian(scale=0.03), bias=Constant(0), activation=Rectlin(), 
                 padding=1, strides=1),
            Conv((3, 3, 384), init=Gaussian(scale=0.03), bias=Constant(0), activation=Rectlin(), 
                 padding=1, strides=1),
            Conv((3, 3, 256), init=Gaussian(scale=0.03), bias=Constant(0), activation=Rectlin(), 
                 padding=1, strides=1),
            Pooling(3, strides=2),
            Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(0), activation=Identity()),
            Dropout(keep=0.5),
            Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(0), activation=Identity()),
            Dropout(keep=0.5),
            Affine(nout=self.noutputs, init=Gaussian(scale=0.01), bias=Constant(0), 
                   activation=Softmax() if self.use_softmax else Logistic(shortcut=True))
        ]

class fergus_bn(EMModelArchitecture):
    def __init__(self, noutputs, use_softmax=False, bn_first_layer=False):
        super(fergus_bn, self).__init__(noutputs, use_softmax)
        self.bn_first_layer = bn_first_layer

    @property
    def layers(self):
        bn = True
        return [
            Conv((7, 7, 96), init=Gaussian(scale=0.01), activation=Rectlin(), batch_norm=bn, 
                    padding=3, strides=1)\
                if self.bn_first_layer else\
                Conv((7, 7, 96), init=Gaussian(scale=0.01), bias=Constant(0), activation=Rectlin(), 
                    padding=3, strides=1),
            Pooling(3, strides=2),
            Conv((5, 5, 256), init=Gaussian(scale=0.01), activation=Rectlin(), batch_norm=bn, 
                 padding=2, strides=1),
            Pooling(3, strides=2),
            Conv((3, 3, 384), init=Gaussian(scale=0.03), activation=Rectlin(), batch_norm=bn, 
                 padding=1, strides=1),
            Conv((3, 3, 384), init=Gaussian(scale=0.03), activation=Rectlin(), batch_norm=bn, 
                 padding=1, strides=1),
            Conv((3, 3, 256), init=Gaussian(scale=0.03), activation=Rectlin(), batch_norm=bn, 
                 padding=1, strides=1),
            Pooling(3, strides=2),
            Affine(nout=4096, init=Gaussian(scale=0.01), activation=Rectlin(), batch_norm=bn),
            Dropout(keep=0.5),
            Affine(nout=4096, init=Gaussian(scale=0.01), activation=Rectlin(), batch_norm=bn),
            Dropout(keep=0.5),
            Affine(nout=self.noutputs, init=Gaussian(scale=0.01), 
                   activation=Softmax() if self.use_softmax else Logistic(shortcut=True))
        ]

class fergus_bn1(fergus_bn):
    def __init__(self, noutputs, use_softmax=False):
        super(fergus_bn1, self).__init__(noutputs, use_softmax, bn_first_layer=True)

class fergus_explin_bn(EMModelArchitecture):
    def __init__(self, noutputs, use_softmax=False, bn_first_layer=False):
        super(fergus_explin_bn, self).__init__(noutputs, use_softmax)
        self.bn_first_layer = bn_first_layer

    @property
    def layers(self):
        bn = True
        return [
            Conv((7, 7, 96), init=Gaussian(scale=0.01), activation=Explin(), batch_norm=bn, 
                    padding=3, strides=1)\
                if self.bn_first_layer else\
                Conv((7, 7, 96), init=Gaussian(scale=0.01), bias=Constant(0), activation=Explin(), 
                    padding=3, strides=1),
            Pooling(3, strides=2),
            Conv((5, 5, 256), init=Gaussian(scale=0.01), activation=Explin(), batch_norm=bn, 
                 padding=2, strides=1),
            Pooling(3, strides=2),
            Conv((3, 3, 384), init=Gaussian(scale=0.03), activation=Explin(), batch_norm=bn, 
                 padding=1, strides=1),
            Conv((3, 3, 384), init=Gaussian(scale=0.03), activation=Explin(), batch_norm=bn, 
                 padding=1, strides=1),
            Conv((3, 3, 256), init=Gaussian(scale=0.03), activation=Explin(), batch_norm=bn, 
                 padding=1, strides=1),
            Pooling(3, strides=2),
            Affine(nout=4096, init=Gaussian(scale=0.01), activation=Explin(), batch_norm=bn),
            Dropout(keep=0.5),
            Affine(nout=4096, init=Gaussian(scale=0.01), activation=Explin(), batch_norm=bn),
            Dropout(keep=0.5),
            Affine(nout=self.noutputs, init=Gaussian(scale=0.01), 
                   activation=Softmax() if self.use_softmax else Logistic(shortcut=True))
        ]

class fergus_explin_bn1(fergus_explin_bn):
    def __init__(self, noutputs, use_softmax=False):
        super(fergus_explin_bn1, self).__init__(noutputs, use_softmax, bn_first_layer=True)

class sfergus_explin_bn(EMModelArchitecture):
    def __init__(self, noutputs, use_softmax=False, bn_first_layer=False):
        super(sfergus_explin_bn, self).__init__(noutputs, use_softmax)
        self.bn_first_layer = bn_first_layer

    @property
    def layers(self):
        bn = True
        return [
            Conv((7, 7, 96), init=Gaussian(scale=0.01), activation=Explin(), batch_norm=bn, 
                    padding=3, strides=1)\
                if self.bn_first_layer else\
                Conv((7, 7, 96), init=Gaussian(scale=0.01), bias=Constant(0), activation=Explin(), 
                    padding=3, strides=1),
            Pooling(3, strides=2),
            Conv((5, 5, 384), init=Gaussian(scale=0.01), activation=Explin(), batch_norm=bn, 
                 padding=2, strides=1),
            Pooling(3, strides=2),
            Conv((3, 3, 512), init=Gaussian(scale=0.03), activation=Explin(), batch_norm=bn, 
                 padding=1, strides=1),
            Conv((3, 3, 512), init=Gaussian(scale=0.03), activation=Explin(), batch_norm=bn, 
                 padding=1, strides=1),
            Conv((3, 3, 384), init=Gaussian(scale=0.03), activation=Explin(), batch_norm=bn, 
                 padding=1, strides=1),
            Pooling(3, strides=2),
            Affine(nout=6144, init=Gaussian(scale=0.01), activation=Explin(), batch_norm=bn),
            Dropout(keep=0.5),
            Affine(nout=6144, init=Gaussian(scale=0.01), activation=Explin(), batch_norm=bn),
            Dropout(keep=0.5),
            Affine(nout=self.noutputs, init=Gaussian(scale=0.01), 
                   activation=Softmax() if self.use_softmax else Logistic(shortcut=True))
        ]

class cifar10(EMModelArchitecture):
    def __init__(self, noutputs, use_softmax=False):
        super(cifar10, self).__init__(noutputs, use_softmax)

    @property
    def layers(self):
        init_uni = Uniform(low=-0.1, high=0.1)
        bn = True
        return [
            Conv((5, 5, 16), init=init_uni, activation=Rectlin(), batch_norm=bn),
            Pooling((2, 2)),
            Conv((5, 5, 32), init=init_uni, activation=Rectlin(), batch_norm=bn),
            Pooling((2, 2)),
            Affine(nout=500, init=init_uni, activation=Rectlin(), batch_norm=bn),
            Affine(nout=self.noutputs, init=init_uni, 
                   activation=Softmax() if self.use_softmax else Logistic(shortcut=True))
        ]

class conv11_7(EMModelArchitecture):
    def __init__(self, noutputs, use_softmax=False):
        super(conv11_7, self).__init__(noutputs, use_softmax)

    @property
    def layers(self):
        #init_uni = Uniform(low=-0.1, high=0.1)
        init_uni = Gaussian(scale=0.01)
        bn = True
        # 2 conv layers, 1 local layers
        return [
            Conv((11, 11, 96), init=init_uni, activation=Rectlin(), batch_norm=bn),
            Pooling((3,3), strides=2),
            Conv((7, 7, 256), init=init_uni, activation=Rectlin(), batch_norm=bn),
            Pooling((3,3), strides=2),
            Affine(nout=1024, init=init_uni, activation=Rectlin(), batch_norm=bn),
            Dropout(keep=0.5),
            Affine(nout=1024, init=init_uni, activation=Rectlin(), batch_norm=bn),
            Dropout(keep=0.5),
            Affine(nout=self.noutputs, init=init_uni, 
                   activation=Softmax() if self.use_softmax else Logistic(shortcut=True))
        ]
