
import threading
import numpy as np
from neon.data.dataiterator import ArrayIterator
#from neon.data.dataiterator import NervanaDataIterator
from parseEMdata import EMDataParser

class DummyEMDataParser():
    def __init__(self, nexamples, image_in_size, nchan, image_out_size, nclass, independent_labels):
        # to be consistent with regular em parser names
        self.num_cases_per_batch = nexamples
        self.image_size = image_in_size
        self.nzslices = nchan
        self.independent_labels = independent_labels
        self.no_labels = False  # xxx - revisit this for testing autoencoder setup
        if independent_labels:
            self.image_out_size = image_out_size
            self.nIndepLabels = nclass
        else:
            # make consistent with EM provider, only one pixel out for not independent labels
            #   and need at least two outputs.
            self.image_out_size = 1
            self.nIndepLabels = nclass if nclass > 1 else 2

        self.pixels_per_image = self.nzslices*self.image_size**2
        self.pixels_per_out_image = self.image_out_size**2
        self.noutputs = self.nIndepLabels*self.image_out_size**2

class NervanaEMDataIterator(ArrayIterator):

    def __init__(self, X=None, y=None, name=None):

        # ArrayIterator parameters that are derived from EM parser
        self.make_onehot = not self._parser.independent_labels and not self._parser.no_labels
        lshape = (self._parser.nzslices,self._parser.image_size,self._parser.image_size)
        nclass = self._parser.noutputs if self._parser.independent_labels else self._parser.nIndepLabels
        nexamples = self._parser.num_cases_per_batch
        oshape = (nexamples, self._parser.noutputs)
        if X is None:
            X = np.random.randint(256,size=(nexamples, self._parser.pixels_per_image)).astype(np.float32)
        if y is None:
            if self._parser.independent_labels:
                y = np.random.randint(2,size=oshape).astype(np.float32)
            else:
                y = np.random.randint(self._parser.noutputs,size=oshape).astype(np.int32)
        #print(X.dtype,y.dtype,X.shape,y.shape,X.flags,y.flags)
        
        super(NervanaEMDataIterator, self).__init__(X, y, nclass=nclass, lshape=lshape, 
            make_onehot=self.make_onehot, name=name)

    @property
    def parser(self): 
        return self._parser
        
    @property
    def nmacrobatches(self):
        raise NotImplemented()



#class NervanaEMDataIterator(NervanaDataIterator):
#    """
#    This generic class defines an interface to iterate over minibatches of
#    data that has been preloaded into memory in the form of numpy arrays.
#    This is a customized version of ArrayIterator for loading batches of EM data.
#    Main difference is that X,y inputs are transposed so that nexamples is second dimension.
#    This also means data is not transposed when copied into minibatches (assign instead of transpose_gen)
#    """
#    
#    def __init__(self, name=None):
#        """
#        Implements loading of given data into backend tensor objects. If the
#        backend is specific to an accelarator device, the data is copied over
#        to that device. Random data generated based on EM parser config.
#        """
#        super(NervanaEMDataIterator, self).__init__(name=name)
#
#        # ArrayIterator parameters that are derived from EM parser
#        make_onehot = not self._parser.independent_labels and not self._parser.no_labels
#        lshape = (self._parser.nzslices,self._parser.image_size,self._parser.image_size)
#        nclass = self._parser.noutputs if self._parser.independent_labels else self._parser.nIndepLabels
#        nexamples = self._parser.num_cases_per_batch
#        X = np.random.randint(256,size=(self._parser.pixels_per_image, nexamples)).astype(np.float32)
#        oshape = (self._parser.noutputs, nexamples)
#        if self._parser.independent_labels:
#            y = np.random.randint(2,size=oshape).astype(np.int32)
#        else:
#            y = np.random.randint(self._parser.noutputs,size=oshape).astype(np.int32)
#
#        # Treat singletons like list so that iteration follows same syntax
#        X = X if isinstance(X, list) else [X]
#        #self.ndata = len(X[0])
#        self.ndata = X[0].shape[1]  # self.ndata is nexamples!
#        assert self.ndata >= self.be.bsz
#        self.start = 0
#        self.nclass = nclass
#        self.ybuf = None
#
#        # store shape of the input data
#        self.shape = [x.shape[1] if lshape is None else lshape for x in X]
#        if len(self.shape) == 1:
#            self.shape = self.shape[0]
#            self.lshape = lshape
#
#        # Helpers to make dataset, minibatch, unpacking function for transpose and onehot
#        def dev_copy(_in, _out, _islice, _oslice):
#            _out[:,_oslice] = _in[:,_islice]
#            
#        def copy_gen(z):
#            return (self.be.array(z), self.be.iobuf(z.shape[0]), dev_copy)
#
#        def onehot_gen(z):
#            return (self.be.array(z, dtype=np.int32), self.be.iobuf(nclass),
#                    lambda _in, _out, _islice, _oslice: self.be.onehot(_in[:,_islice], axis=0, out=_out[:,_oslice]))
#
#        self.Xdev, self.Xbuf, self.unpack_func = zip(*[copy_gen(x) for x in X])
#
#        # Shallow copies for appending, iterating
#        self.dbuf, self.hbuf = list(self.Xdev), list(self.Xbuf)
#        self.unpack_func = list(self.unpack_func)
#
#        self.ydev, self.ybuf, yfunc = onehot_gen(y) if make_onehot else copy_gen(y)
#        self.dbuf.append(self.ydev)
#        self.hbuf.append(self.ybuf)
#        self.unpack_func.append(yfunc)
#
#    @property
#    def nbatches(self):
#        return -((self.start - self.ndata) // self.be.bsz)
#
#    def reset(self):
#        """
#        For resetting the starting index of this dataset back to zero.
#        Relevant for when one wants to call repeated evaluations on the dataset
#        but don't want to wrap around for the last uneven minibatch
#        Not necessary when ndata is divisible by batch size
#        """
#        self.start = 0
#
#    def __iter__(self):
#        """
#        Defines a generator that can be used to iterate over this dataset.
#        Override for EM data so that unpack_func can be just an assignment.
#
#        Yields:
#            tuple: The next minibatch which includes both features and labels.
#        """
#        for i1 in range(self.start, self.ndata, self.be.bsz):
#            bsz = min(self.be.bsz, self.ndata - i1)
#            oslice1, islice1 = slice(0, bsz), slice(i1, i1 + bsz)
#            oslice2, islice2 = None, None
#            if self.be.bsz > bsz:
#                oslice2, islice2 = slice(bsz, None), slice(0, self.be.bsz - bsz)
#                self.start = self.be.bsz - bsz
#
#            for buf, dev, unpack_func in zip(self.hbuf, self.dbuf, self.unpack_func):
#                unpack_func(dev, buf, islice1, oslice1)
#                if oslice2:
#                    unpack_func(dev, buf, islice2, oslice2)
#
#            inputs = self.Xbuf[0] if len(self.Xbuf) == 1 else self.Xbuf
#            targets = self.ybuf if self.ybuf else inputs
#            yield (inputs, targets)
#
#    @property
#    def parser(self): 
#        return self._parser
#        
#    @property
#    def nmacrobatches(self):
#        raise NotImplemented()



class RandomEMDataIterator(NervanaEMDataIterator):
    def __init__(self, nexamples=256, image_in_size=64, nchan=1, image_out_size=16, nclass=3,
                 independent_labels=True, name=None):
        self._parser = DummyEMDataParser(nexamples, image_in_size, nchan, image_out_size, nclass, independent_labels)
        super(RandomEMDataIterator, self).__init__(name=name)

    @property
    def nmacrobatches(self):
        return 1

class EMDataIterator(NervanaEMDataIterator):
    def __init__(self, cfg_file, write_outputs=False, append_features=False, chunk_skip_list=[], dim_ordering='', 
                 batch_range=[1,10], one_batch=True, name='emdata'):

        # instantiate the actual em data parser, code shared with cuda-convnets2 em data parser
        self._parser = EMDataParser(cfg_file, write_outputs=write_outputs, append_features=append_features, 
                                   chunk_skip_list=chunk_skip_list, dim_ordering=dim_ordering)
        # parser relies on having initBatches called right away, xxx - could revisit this?
        self._parser.initBatches()

        # batches are numbered starting at 1 and inclusive of end of range
        self.batch_range = batch_range; self.batchnum = batch_range[0]
        self.nextdata = [None] * (2 + self._parser.naug_data)
        # immediately start loading the first batch, must be called here for __iter__ to work properly
        self.make_onehot = not self._parser.independent_labels and not self._parser.no_labels
        self.get_next_EMbatch(async=not one_batch)

        # mostly for debug, just stay on the first batch pre-loaded into ArrayIterator
        self.one_batch = one_batch
        if one_batch:
            print('WARNING: EMDataIterator one_batch mode enabled!')
            super(EMDataIterator, self).__init__(X=self.nextdata[:-1], y=self.nextdata[-1], name=name)
        else:
            # iterator initilizes random batches but will be overwritten with first batch in __iter__
            super(EMDataIterator, self).__init__(name=name)

    @property
    def nmacrobatches(self):
        return self.batch_range[1] - self.batch_range[0] + 1

    def get_next_EMbatch(self, async=True):
        if async:
            # starts a thread that loads the next EM batch in parallel with model running
            self.thread = threading.Thread(target=self._get_next_EMbatch); self.thread.start()
        else:
            self.thread = None
            self._get_next_EMbatch()
            
    def _get_next_EMbatch(self):
        nextdata = self._parser.getBatch(self.batchnum)

        # need to manipulate data and labels returned by EM parser to be congruent with neon
        assert( len(nextdata) == len(self.nextdata) )
        # re-arrange so that labels are last
        nextdata = [nextdata[i] for i in ([0] + range(2,self._parser.naug_data+2) + [1])]
        # order from EM data parser is tranpose of neon data, so switch nexamples (num_cases_per_batch) to first dim
        for i in range(len(nextdata)-1):
            self.nextdata[i] = nextdata[i].T.copy(order='C')
        # convert labels that are not onehot (independent_labels) to int
        if self.make_onehot:
            self.nextdata[-1] = nextdata[-1].T.astype(np.int32, order='C')
        else:
            self.nextdata[-1] = nextdata[-1].T.copy(order='C')

        # advance to next batch, roll around at end of batch range
        self.batchnum += 1
        if self.batchnum > self.batch_range[1]: self.batchnum = self.batch_range[0]

    def set_batchnum(self, batchnum):
        # wait until the current batch is loaded if loading asynchronously
        if self.thread: self.thread.join()
        self.batchnum = batchnum
        if self.batchnum < self.batch_range[0] or self.batchnum > self.batch_range[1]: 
            self.batchnum = self.batch_range[0]
        self.get_next_EMbatch()

    def __iter__(self):
        if not self.one_batch:
            # wait until the next batch is loaded if loading asynchronously
            if self.thread: self.thread.join()
            for i in range(len(self.nextdata)):
                self.dbuf[i].set(self.nextdata[i])
            self.get_next_EMbatch()   # start next batch loading

        return super(EMDataIterator, self).__iter__()
