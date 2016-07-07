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

from neon.initializers import Constant, Gaussian, Uniform, Kaiming
from neon.layers import Conv, Dropout, Pooling, Affine, LRN
from neon.layers import Activation, MergeSum, SkipNode, BatchNorm
from neon.transforms import Rectlin, Logistic, Softmax, Identity, Explin
from layers.emlayers import DOG

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
            Pooling(3, strides=2, padding=1),
            Conv((5, 5, 256), init=Gaussian(scale=0.01), bias=Constant(0), activation=Rectlin(), 
                 padding=2, strides=1),
            LRN(31, ascale=0.001, bpower=0.75),
            Pooling(3, strides=2, padding=1),
            Conv((3, 3, 384), init=Gaussian(scale=0.03), bias=Constant(0), activation=Rectlin(), 
                 padding=1, strides=1),
            Conv((3, 3, 384), init=Gaussian(scale=0.03), bias=Constant(0), activation=Rectlin(), 
                 padding=1, strides=1),
            Conv((3, 3, 256), init=Gaussian(scale=0.03), bias=Constant(0), activation=Rectlin(), 
                 padding=1, strides=1),
            Pooling(3, strides=2, padding=1),
            Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(0), activation=Identity()),
            Dropout(keep=0.5),
            Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(0), activation=Identity()),
            Dropout(keep=0.5),
            Affine(nout=self.noutputs, init=Gaussian(scale=0.01), bias=Constant(0), 
                   activation=Softmax() if self.use_softmax else Logistic(shortcut=True))
        ]

class nfergus(EMModelArchitecture):
    def __init__(self, noutputs, use_softmax=False, bn_first_layer=False):
        super(nfergus, self).__init__(noutputs, use_softmax)
        self.bn_first_layer = bn_first_layer

    @property
    def layers(self):
        bn = True
        return [
            Conv((7, 7, 96), init=Kaiming(), activation=Explin(), batch_norm=bn, 
                    padding=3, strides=1)\
                if self.bn_first_layer else\
                Conv((7, 7, 96), init=Kaiming(), bias=Constant(0), activation=Explin(), 
                    padding=3, strides=1),
            Pooling(3, strides=2, padding=1),
            Conv((5, 5, 256), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=2, strides=1),
            Pooling(3, strides=2, padding=1),
            Conv((3, 3, 384), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=1),
            Conv((3, 3, 384), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=1),
            Conv((3, 3, 384), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=1),
            Pooling(3, strides=2, padding=1, op='avg'),
            Affine(nout=self.noutputs, init=Kaiming(), activation=Explin(), batch_norm=bn),
            Affine(nout=self.noutputs, init=Kaiming(), activation=Explin(), batch_norm=bn),
            Affine(nout=self.noutputs, init=Kaiming(), 
                   activation=Softmax() if self.use_softmax else Logistic(shortcut=True))
        ]

class mfergus(EMModelArchitecture):
    def __init__(self, noutputs, use_softmax=False, bn_first_layer=False):
        super(mfergus, self).__init__(noutputs, use_softmax)
        self.bn_first_layer = bn_first_layer

    @property
    def layers(self):
        bn = True
        return [
            Conv((7, 7, 96), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=3, strides=1)\
                if self.bn_first_layer else\
                Conv((7, 7, 96), init=Kaiming(), bias=Constant(0), activation=Explin(), padding=3, strides=1),
            Pooling(3, strides=2, padding=1),
            Conv((7, 7, 128), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=3, strides=1),
            Pooling(3, strides=2, padding=1),
            Conv((5, 5, 256), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=2, strides=1),
            Pooling(3, strides=2, padding=1),
            Conv((3, 3, 384), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=1),
            Conv((3, 3, 384), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=1),
            Conv((3, 3, 384), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=1),
            Pooling(3, strides=2, padding=1, op='avg'),
            Affine(nout=self.noutputs, init=Kaiming(), activation=Explin(), batch_norm=bn),
            Affine(nout=self.noutputs, init=Kaiming(), activation=Explin(), batch_norm=bn),
            Affine(nout=self.noutputs, init=Kaiming(), 
                   activation=Softmax() if self.use_softmax else Logistic(shortcut=True))
        ]

class kaiming(EMModelArchitecture):
    def __init__(self, noutputs, use_softmax=False):
        super(kaiming, self).__init__(noutputs, use_softmax)

    @property
    def layers(self):
        bn = True
        return [
            # input 128
            Conv((7, 7, 64), init=Kaiming(), bias=Constant(0), activation=Explin(), padding=3, strides=1),
            Pooling(3, strides=2, padding=1),
            # 64
            Conv((3, 3, 64), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=1),
            Conv((3, 3, 64), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=1),
            # 32
            Conv((3, 3, 256), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=2),
            Conv((3, 3, 256), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=1),
            Conv((3, 3, 256), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=1),
            Conv((3, 3, 256), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=1),
            # 16
            Conv((3, 3, 512), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=2),
            Conv((3, 3, 512), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=1),
            Conv((3, 3, 512), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=1),
            Conv((3, 3, 512), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=1),
            # 8
            Conv((3, 3, 1024), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=2),
            Conv((3, 3, 1024), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=1),
            Conv((3, 3, 1024), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=1),
            Conv((3, 3, 1024), init=Kaiming(), activation=Explin(), batch_norm=bn, padding=1, strides=1),
            # 1
            Pooling('all', op='avg'),
            Affine(nout=self.noutputs, init=Kaiming(), 
                   activation=Softmax() if self.use_softmax else Logistic(shortcut=True))
        ]

class dkaiming(EMModelArchitecture):
    def __init__(self, noutputs, use_softmax=False):
        super(dkaiming, self).__init__(noutputs, use_softmax)

    @property
    def layers(self):
        bn = True
        return [
            # input 224
            Conv((7, 7, 64), init=Kaiming(), bias=Constant(0), activation=Rectlin(), padding=3, strides=2),
            Pooling(3, strides=2, padding=1),
            # 56
            Conv((3, 3, 64), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=1, strides=1),
            Conv((3, 3, 64), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=1, strides=1),
            Conv((3, 3, 64), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=1, strides=1),
            Conv((3, 3, 64), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=1, strides=1),
            # 28
            Conv((3, 3, 128), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=1, strides=2),
            Conv((3, 3, 128), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=1, strides=1),
            Conv((3, 3, 128), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=1, strides=1),
            Conv((3, 3, 128), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=1, strides=1),
            # 14
            Conv((3, 3, 256), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=1, strides=2),
            Conv((3, 3, 256), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=1, strides=1),
            Conv((3, 3, 256), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=1, strides=1),
            Conv((3, 3, 256), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=1, strides=1),
            # 7
            Conv((1, 1, 512), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=0, strides=2),
            Conv((3, 3, 512), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=1, strides=1),
            Conv((1, 1, 2048), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=0, strides=1),
            Conv((1, 1, 512), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=0, strides=1),
            Conv((3, 3, 512), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=1, strides=1),
            Conv((1, 1, 2048), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=0, strides=1),
            # 1
            Pooling('all', op='avg'),
            Affine(nout=self.noutputs, init=Kaiming(), 
                   activation=Softmax() if self.use_softmax else Logistic(shortcut=True))
        ]

# xxx - in neon v1.5.2 requires way too much memory
class dpkaiming(EMModelArchitecture):
    def __init__(self, noutputs, use_softmax=False):
        super(dpkaiming, self).__init__(noutputs, use_softmax)

    @property
    def layers(self):
        return [
            # input 224
            Conv((7, 7, 64), init=Kaiming(), bias=Constant(0), activation=Rectlin(), padding=3, strides=2),
            Pooling(3, strides=2, padding=1),
            # 56
            MergeSum([
                SkipNode(), [
                    BatchNorm(), Activation(Rectlin()),
                    Conv((3, 3, 64), init=Kaiming(), activation=Rectlin(), batch_norm=True, padding=1, strides=1),
                    Conv((3, 3, 64), init=Kaiming(), activation=None, batch_norm=False, padding=1, strides=1),
                ] ]),
            MergeSum([
                SkipNode(), [
                    BatchNorm(), Activation(Rectlin()),
                    Conv((3, 3, 64), init=Kaiming(), activation=Rectlin(), batch_norm=True, padding=1, strides=1),
                    Conv((3, 3, 64), init=Kaiming(), activation=None, batch_norm=False, padding=1, strides=1),
                ] ]),
            #Conv((3, 3, 64), init=Kaiming(), activation=Rectlin(), batch_norm=True, padding=1, strides=1),
            #Conv((3, 3, 64), init=Kaiming(), activation=Rectlin(), batch_norm=True, padding=1, strides=1),
            #Conv((3, 3, 64), init=Kaiming(), activation=Rectlin(), batch_norm=True, padding=1, strides=1),
            #Conv((3, 3, 64), init=Kaiming(), activation=Rectlin(), batch_norm=True, padding=1, strides=1),
            # 28
            #BatchNorm(), Activation(Rectlin()),
            #MergeSum([
            #    Conv((3, 3, 128), init=Kaiming(), activation=None, batch_norm=False, padding=1, strides=1),
            #    [
            #        Conv((3, 3, 128), init=Kaiming(), activation=Rectlin(), batch_norm=True, padding=1, strides=2),
            #        Conv((3, 3, 128), init=Kaiming(), activation=None, batch_norm=False, padding=1, strides=1),
            #    ] ]),
            #MergeSum([
            #    SkipNode(), [
            #        BatchNorm(), Activation(Rectlin()),
            #        Conv((3, 3, 128), init=Kaiming(), activation=Rectlin(), batch_norm=True, padding=1, strides=1),
            #        Conv((3, 3, 128), init=Kaiming(), activation=None, batch_norm=False, padding=1, strides=1),
            #    ] ]),
            Conv((3, 3, 128), init=Kaiming(), activation=Rectlin(), batch_norm=True, padding=1, strides=2),
            Conv((3, 3, 128), init=Kaiming(), activation=Rectlin(), batch_norm=True, padding=1, strides=1),
            Conv((3, 3, 128), init=Kaiming(), activation=Rectlin(), batch_norm=True, padding=1, strides=1),
            Conv((3, 3, 128), init=Kaiming(), activation=Rectlin(), batch_norm=True, padding=1, strides=1),
            # 14
            Conv((3, 3, 256), init=Kaiming(), activation=Rectlin(), batch_norm=True, padding=1, strides=2),
            Conv((3, 3, 256), init=Kaiming(), activation=Rectlin(), batch_norm=True, padding=1, strides=1),
            Conv((3, 3, 256), init=Kaiming(), activation=Rectlin(), batch_norm=True, padding=1, strides=1),
            Conv((3, 3, 256), init=Kaiming(), activation=Rectlin(), batch_norm=True, padding=1, strides=1),
            # 7
            #Conv((1, 1, 512), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=0, strides=2),
            #Conv((3, 3, 512), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=1, strides=1),
            #Conv((1, 1, 2048), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=0, strides=1),
            #Conv((1, 1, 512), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=0, strides=1),
            #Conv((3, 3, 512), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=1, strides=1),
            #Conv((1, 1, 2048), init=Kaiming(), activation=Rectlin(), batch_norm=bn, padding=0, strides=1),
            # 1
            Pooling('all', op='avg'),
            Affine(nout=self.noutputs, init=Kaiming(), 
                   activation=Softmax() if self.use_softmax else Logistic(shortcut=True))
        ]

class cifar10(EMModelArchitecture):
    def __init__(self, noutputs, use_softmax=False):
        super(cifar10, self).__init__(noutputs, use_softmax)

    @property
    def layers(self):
        init_uni = Uniform(low=-0.1, high=0.1)
        bn = False
        return [
            Conv((5, 5, 16), init=init_uni, activation=Rectlin(), batch_norm=bn),
            Pooling((2, 2)),
            Conv((5, 5, 32), init=init_uni, activation=Rectlin(), batch_norm=bn),
            Pooling((2, 2)),
            Affine(nout=500, init=init_uni, activation=Rectlin(), batch_norm=bn),
            Affine(nout=self.noutputs, init=init_uni, 
                   activation=Softmax() if self.use_softmax else Logistic(shortcut=True))
        ]

class DOG_cifar10(EMModelArchitecture):
    def __init__(self, noutputs, use_softmax=False):
        super(DOG_cifar10, self).__init__(noutputs, use_softmax)

    @property
    def layers(self):
        init_uni = Uniform(low=-0.1, high=0.1)
        bn = False
        return [
            DOG((5.0, 4.0, 3.0, 1.6), 1.8),
            Conv((5, 5, 16), init=init_uni, activation=Rectlin(), batch_norm=bn),
            Pooling((2, 2)),
            Conv((5, 5, 32), init=init_uni, activation=Rectlin(), batch_norm=bn),
            Pooling((2, 2)),
            Affine(nout=500, init=init_uni, activation=Rectlin(), batch_norm=bn),
            Affine(nout=self.noutputs, init=init_uni, 
                   activation=Softmax() if self.use_softmax else Logistic(shortcut=True))
        ]
