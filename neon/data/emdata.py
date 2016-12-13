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

import threading
from threading import Thread
import numpy as np
import os

from neon.data.dataiterator import ArrayIterator
from neon.backends.nervanagpu import NervanaGPU
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
        self.nclass = self.noutputs if self.independent_labels else self.nIndepLabels
        self.oshape = (self.image_out_size,self.image_out_size,self.nIndepLabels)

class NervanaEMDataIterator(ArrayIterator):

    def __init__(self, X=None, y=None, nexamples=None, parser=None, name=None):
        # ArrayIterator parameters that are derived from EM parser
        if parser: self.parser = parser
        else: assert(hasattr(self,'parser'))
        if not nexamples: nexamples = self.parser.num_cases_per_batch
        self.make_onehot = not self.parser.independent_labels and not self.parser.no_labels
        lshape = (self.parser.nzslices,self.parser.image_size,self.parser.image_size)
        osize = (nexamples, self.parser.noutputs)
        if X is None:
            X = np.random.randint(256,size=(nexamples, self.parser.pixels_per_image)).astype(np.float32)
        if y is None and not self.parser.no_labels:
            if self.parser.independent_labels:
                y = np.random.randint(2,size=osize).astype(np.float32)
            else:
                y = np.random.randint(self.parser.noutputs,size=osize).astype(np.int32)
        #print(X.dtype,y.dtype,X.shape,y.shape,X.flags,y.flags)
        
        super(NervanaEMDataIterator, self).__init__(X, y, nclass=self.parser.nclass, lshape=lshape, 
            make_onehot=self.make_onehot, name=name)

    @property
    def nmacrobatches(self):
        raise NotImplemented()

    # implement this if specifically need to stop threads or cleanup
    def stop(self):
        pass

class RandomEMDataIterator(NervanaEMDataIterator):
    def __init__(self, nexamples=256, image_in_size=64, nchan=1, image_out_size=16, nclass=3,
                 independent_labels=False, name=None):
        self.parser = DummyEMDataParser(nexamples, image_in_size, nchan, image_out_size, nclass, independent_labels)
        super(RandomEMDataIterator, self).__init__(name=name)

    @property
    def nmacrobatches(self):
        return 1

class EMDataIterator(NervanaEMDataIterator, Thread):

    def __init__(self, cfg_file, write_output=None, chunk_skip_list=[], dim_ordering='', batch_range=[1,10], 
                 name='emdata', isTest=False, concatenate_batches=False, NBUF=2, image_in_size=None):
        Thread.__init__(self)
        self.name = name

        # mostly intended for double buffering (NBUF==2) so that data can be pushed to card simultaneous with training.
        # single buffer (NUF==1) fetches next EM batch in parallel but waits until __iter__ to push to backend buffer.
        # more buffers should work (NBUF > 2) but takes more gpu memory and likely no speed improvement
        assert( NBUF > 0 )
        self.NBUF = NBUF

        # batches are numbered starting at 1 and inclusive of end of range.
        # this needs to be done first so that nmacrobatches property works.
        self.batch_range = batch_range; self.batchnum = batch_range[0]

        # previously parser was agnostic to test or train, but needed it for allowing single ini in chunk_list_all mode
        self.isTest = isTest

        # if the output an hdf file name, then this is a single whole-dataset hdf5 file.
        # xxx - initializations for writing output features could be cleaned up.
        write_outputs = (write_output is not None); append_features = False
        if write_outputs:
            fn, ext = os.path.splitext(write_output); ext = ext.lower()
            # .conf indicates to write knossos-style outputs
            append_features = (ext == '.h5' or ext == '.hdf5' or ext == '.conf')
            write_outputs = not append_features
        # instantiate the actual em data parser, code shared with cuda-convnets2 em data parser
        self.parser = EMDataParser(cfg_file, write_outputs=write_outputs, append_features=append_features, 
                                    chunk_skip_list=chunk_skip_list, dim_ordering=dim_ordering, isTest=self.isTest,
                                    image_in_size=image_in_size)
        if write_outputs or append_features:
            # force some properties if in mode for writing outputs.
            # xxx - this is not clean, needs some rethinking on how write_outputs modes are initialized
            self.parser.outpath = write_output
            self.parser.no_label_lookup = True
            self.parser.append_features_knossos = append_features and (ext == '.conf')
            if self.parser.append_features_knossos: self.parser.outpath = os.path.dirname(fn)
        # parser relies on having initBatches called right away, xxx - could revisit this?
        self.parser.initBatches()

        # no need for special code to concatenate if there is only one macrobatch anyways
        self.concatenate_batches = concatenate_batches and (self.nmacrobatches > 1)

        self.nexamples = self.parser.num_cases_per_batch
        if self.concatenate_batches: self.nexamples *= self.nmacrobatches

        # locks and events for synchronizing data loading thread.
        self.init_event = threading.Event()
        if self.NBUF > 1:
            self.lbuf_lock = threading.Lock(); self.cbuf_lock = threading.Lock()
            self.lbuf_event = threading.Event(); self.cbuf_event = threading.Event()
        else:
            self.push_event = threading.Event(); self.push_done_event = threading.Event()

        # set pycuda driver for gpu backend
        # xxx - this is a bit hacky, is there a better way to do this?
        if type(self.be) == NervanaGPU:
            import pycuda.driver as drv
            self.drv = drv
            #self.stream = self.drv.Stream() # xxx - for other synchonize method??? see below
        else:
            self.drv = None
            
        # start the thread and wait for initialization to complete.
        # initialization of backend memory has to occur within the thread.
        self.daemon = True  # so that stop event is not necessary to terminate threads when process completes.
        self.start()
        self.init_event.wait()

    def run(self):
        # this allows the current running thread to push data to the gpu memory buffer.
        # ArrayIterator constructor has to be called within this thread also,
        #   so that the memory is allocated in this context.
        # xxx - cleanup call to self.ctx.detach() ???
        if self.drv is not None:
            self.ctx = self.drv.Device(self.be.device_id).make_context()

        # iterator initilizes random batches but will be overwritten with first batch in __iter__
        super(EMDataIterator, self).__init__(name=self.name, nexamples=self.nexamples)

        # setup multiple buffers (two should be sufficient?).
        # this allows data to be copied to the backend (gpu) memory while the previous macrobatch is running.
        self.iter_buf = [None]*self.NBUF; self.iter_buf[0] = self; self.cbuf = 0; self.lbuf = 0
        for i in range(1,self.NBUF):
            self.iter_buf[i] = NervanaEMDataIterator(name=self.name + str(i), nexamples=self.nexamples,
                parser=self.parser)

        # cpu buffers for storing batches from EM parser before they are written to gpu.
        self.num_data = 1 + self.parser.naug_data
        self.num_labels = 0 if self.parser.no_labels else 1
        self.num_data_labels = self.num_data + self.num_labels
        self.nextdata = [None] * self.num_data_labels
        if self.concatenate_batches:
            # http://stackoverflow.com/questions/2397141/how-to-initialize-a-two-dimensional-array-in-python
            # http://stackoverflow.com/questions/10668341/create-3d-array-using-python
            self.allnextdata = [[None for i in range(self.nmacrobatches)] for j in range(self.num_data_labels)]

        # run loop for loading data continues as long as process is running.
        self.init_event.set()  # initialization completed
        while True:
            # load the next set of batches into system memory
            self._get_EMbatches()

            if self.NBUF > 1:
                # immediately push the data into the current lbuf
                self._push_be_buffer()
                
                # advance the load buffer pointer
                self.lbuf_lock.acquire()
                self.lbuf = (self.lbuf + 1) % self.NBUF
                self.lbuf_event.set()
                self.lbuf_lock.release()
                
                # wait until the next load buffer is free
                self.cbuf_lock.acquire()
                wait = ((self.cbuf - 1) % self.NBUF == self.lbuf)
                self.cbuf_event.clear()
                self.cbuf_lock.release()
                if wait: self.cbuf_event.wait()
            else:
                # wait until backend is ready to push next data.
                self.push_event.wait()
                # push data to backend and then signal push done
                self.push_event.clear()
                self._push_be_buffer()
                self.push_done_event.set()

    def reset_batchnum(self, batchnum):
        # xxx - purpose of this is to start training a model at the batch where it left off.
        #   this is pretty minor in the grand scheme of training, and a pain to implement here.
        pass

    def _get_EMbatches(self):
        if self.concatenate_batches:
            # fetch all the batches into system memory at once
            for n in range(self.nmacrobatches):
                self._get_next_EMbatch()
                for i in range(self.num_data_labels):
                    self.allnextdata[i][n] = self.nextdata[i]
            self.pushdata = [np.concatenate(self.allnextdata[i], axis=0) for i in range(self.num_data_labels)]
        else:
            # featch single batch into system memory
            self._get_next_EMbatch()
            self.pushdata = self.nextdata

    def _push_be_buffer(self):
        # push batch onto backend buffer
        for i in range(self.num_data_labels):
            self.iter_buf[self.lbuf].dbuf[i].set(self.pushdata[i])

        if self.drv is not None:
            # xxx - does it matter which synchronize method is used here???
            #end = self.drv.Event()
            #end.record(self.stream)
            #end.synchronize()
            self.ctx.synchronize()
            
    def _get_next_EMbatch(self):
        p = self.parser
        nextdata = p.getBatch(self.batchnum)

        # need to manipulate data and labels returned by EM parser to be congruent with neon
        assert( len(nextdata) == self.num_data_labels )
        # re-arrange so that labels are last
        if self.num_labels > 0:
            nextdata = [nextdata[i] for i in ([0] + range(2,p.naug_data+2) + [1])]
        # order from EM data parser is tranpose of neon data, so switch nexamples (num_cases_per_batch) to first dim
        for i in range(self.num_data):
            # image dimensions and pixels / examples dimensions are transposed relative to cc2 input
            #self.nextdata[i] = nextdata[i].reshape((p.nzslices, p.image_size, p.image_size, p.num_cases_per_batch)).\
            #    transpose((3,0,2,1)).reshape((p.num_cases_per_batch, p.pixels_per_image)).copy(order='C')
            # xxx - decided above was a poor choice, transpose should not matter as long as input/ouput are in same
            #   orientation relative to each other. swap the image and samples dimensions only
            self.nextdata[i] = nextdata[i].T.copy(order='C')
                
        if self.num_labels > 0:
            # convert labels that are not onehot (independent_labels) to int
            if self.make_onehot:
                self.nextdata[-1] = nextdata[-1].T.astype(np.int32, order='C')
            else:
                self.nextdata[-1] = nextdata[-1].T.copy(order='C')

        # advance to next batch, roll around at end of batch range
        self.batchnum += 1
        if self.batchnum > self.batch_range[1]: self.batchnum = self.batch_range[0]

    @property
    def nmacrobatches(self):
        return self.batch_range[1] - self.batch_range[0] + 1

    def __iter__(self):
        if self.NBUF > 1:
            # wait until the next current buffer is available
            self.lbuf_lock.acquire()
            wait = (self.cbuf == self.lbuf)
            self.lbuf_event.clear()
            self.lbuf_lock.release()
            if wait: self.lbuf_event.wait()
        else:
            # signal to push data to backend and wait until push done
            self.push_event.set()
            self.push_done_event.wait()
            self.push_done_event.clear()

        # generate next batch from current buffer
        _iter = super(NervanaEMDataIterator, self.iter_buf[self.cbuf]).__iter__()

        if self.NBUF > 1:
            # advance current buffer pointer
            self.cbuf_lock.acquire()
            self.cbuf = (self.cbuf + 1) % self.NBUF
            self.cbuf_event.set()
            self.cbuf_lock.release()

        return _iter
        
