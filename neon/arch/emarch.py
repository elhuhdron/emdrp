
from neon.initializers import Constant, Gaussian, Uniform
from neon.layers import Conv, Dropout, Pooling, Affine, LRN
from neon.transforms import Rectlin, Identity, Logistic, Softmax

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

        #    @property
        #    def layers(self):
        #        # 5 conv layers, 2 local layers, logistic outputs
        #        return [
        #            Conv((7, 7, 96), init=Gaussian(scale=0.01), bias=Constant(0),
        #                 activation=Rectlin(), padding=3, strides=1),
        #            #LRN(31, alpha=0.001, beta=0.75),
        #            Pooling(3, strides=2),
        #            Conv((5, 5, 256), init=Gaussian(scale=0.01), bias=Constant(1),
        #                 activation=Rectlin(), padding=2),
        #            #LRN(31, alpha=0.001, beta=0.75),
        #            Pooling(3, strides=2),
        #            Conv((3, 3, 384), init=Gaussian(scale=0.03), bias=Constant(0),
        #                 activation=Rectlin(), padding=1),
        #            Conv((3, 3, 384), init=Gaussian(scale=0.03), bias=Constant(1),
        #                 activation=Rectlin(), padding=1),
        #            Conv((3, 3, 256), init=Gaussian(scale=0.03), bias=Constant(1),
        #                 activation=Rectlin(), padding=1),
        #            Pooling(3, strides=2),
        #            Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin()),
        #            Dropout(keep=0.5),
        #            Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin()),
        #            Dropout(keep=0.5),
        #            Affine(nout=self.noutputs, init=Gaussian(scale=0.01), bias=Constant(-7), 
        #                   activation=Softmax() if self.use_softmax else Logistic())
        #        ]

    @property
    def layers(self):
        # 5 conv layers, 2 local layers, logistic outputs
        return [
            Conv((7, 7, 96), init=Gaussian(scale=0.0001), bias=Constant(0), activation=Rectlin(), 
                 padding=3, strides=1),
            LRN(31, ascale=0.001, bpower=0.75),
            Pooling(3, strides=2),
            Conv((5, 5, 256), init=Gaussian(scale=0.01), bias=Constant(0), activation=Rectlin(), 
                 padding=2, strides=1),
            LRN(31, ascale=0.001, bpower=0.75),
            Pooling(3, strides=2),
            Conv((3, 3, 384), init=Gaussian(scale=0.01), bias=Constant(0), activation=Rectlin(), 
                 padding=1, strides=1),
            Conv((3, 3, 384), init=Gaussian(scale=0.01), bias=Constant(0), activation=Rectlin(), 
                 padding=1, strides=1),
            Conv((3, 3, 256), init=Gaussian(scale=0.01), bias=Constant(0), activation=Rectlin(), 
                 padding=1, strides=1),
            Pooling(3, strides=2),
            Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(0), activation=Identity()),
            Dropout(keep=0.5),
            Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(0), activation=Identity()),
            Dropout(keep=0.5),
            Affine(nout=self.noutputs, init=Gaussian(scale=0.01), bias=Constant(0), 
                   activation=Softmax() if self.use_softmax else Logistic())
        ]

class cifar10(EMModelArchitecture):
    def __init__(self, noutputs, use_softmax=False):
        super(cifar10, self).__init__(noutputs, use_softmax)

    @property
    def layers(self):
        init_uni = Uniform(low=-0.1, high=0.1)
        bn = False
        # 2 conv layers, 2 local layers, logistic outputs
        return [
            Conv((5, 5, 16), init=init_uni, activation=Rectlin(), batch_norm=bn),
            Pooling((2, 2)),
            Conv((5, 5, 32), init=init_uni, activation=Rectlin(), batch_norm=bn),
            Pooling((2, 2)),
            Affine(nout=500, init=init_uni, activation=Rectlin(), batch_norm=bn),
            Affine(nout=self.noutputs, init=Gaussian(scale=0.01), bias=Constant(0), 
                   activation=Softmax() if self.use_softmax else Logistic())
        ]
