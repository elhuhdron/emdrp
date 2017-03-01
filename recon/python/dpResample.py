#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2017 Paul Watkins, National Institutes of Health / NINDS
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

# Method for doing simple factor of 2 downsampling/upsampling.
# Upsampling simply repeats pixels. Downsampling can just decimate without transformation or do "pixel averaging"

import numpy as np
#import h5py
import argparse
import time
#import glob
#import os

#from dpLoadh5 import dpLoadh5
from dpWriteh5 import dpWriteh5
#from typesh5 import emProbabilities, emVoxelType
from dpCubeIter import dpCubeIter

class dpResample(dpWriteh5):

    def __init__(self, args):
        #self.LIST_ARGS += ['train_offsets', 'prob_types']
        dpWriteh5.__init__(self,args)

        self.cubeIter = dpCubeIter.cubeIterGen(self.volume_range_beg,self.volume_range_end,self.overlap,self.cube_size,
                    left_remainder_size=self.left_remainder_size, right_remainder_size=self.right_remainder_size,
                    chunksize=self.chunksize, leave_edge=self.leave_edge)

        self.resample_dims = self.resample_dims.astype(np.bool)
        self.nresample_dims = self.resample_dims.sum(dtype=np.uint8)
        assert( self.nresample_dims > 0 )   # no resample dims specified
        self.slices = 2**self.nresample_dims

        # xxx - probably a way to do this programatically, but easier to read as enumerated.
        #   also this script is intended to only ever down/up sampled by factors of 2.
        if (self.resample_dims == np.array([1,0,0])).all():
            self.slices = [np.s_[::2,:,:], np.s_[1::2,:,:]]
        elif (self.resample_dims == np.array([0,1,0])).all():
            self.slices = [np.s_[:,::2,:], np.s_[:,1::2,:]]
        elif (self.resample_dims == np.array([0,0,1])).all():
            self.slices = [np.s_[:,:,::2], np.s_[:,:,1::2]]
        elif (self.resample_dims == np.array([1,1,0])).all():
            self.slices = [np.s_[::2,::2,:], np.s_[1::2,::2,:], np.s_[::2,1::2,:], np.s_[1::2,1::2,:]]
        elif (self.resample_dims == np.array([1,0,1])).all():
            self.slices = [np.s_[::2,:,::2], np.s_[1::2,:,::2], np.s_[::2,:,1::2], np.s_[1::2,:,1::2]]
        elif (self.resample_dims == np.array([0,1,1])).all():
            self.slices = [np.s_[:,::2,::2], np.s_[:,1::2,::2], np.s_[:,::2,1::2], np.s_[:,1::2,1::2]]
        elif self.resample_dims.all():
            self.slices = [np.s_[::2,::2,::2], np.s_[1::2,::2,::2], np.s_[::2,1::2,::2], np.s_[::2,::2,1::2],
                           np.s_[1::2,1::2,::2], np.s_[1::2,::2,1::2], np.s_[::2,1::2,1::2], np.s_[1::2,1::2,1::2]]
        assert( len(self.slices) == self.nslices ) # sanity check

        # print out all initialized variables in verbose mode
        if self.dpResample_verbose:
            print('dpResample, verbose mode:\n'); print(vars(self))

    def iterResample(self):
        for self.volume_info,n in zip(self.cubeIter, range(self.cubeIter.volume_size)):
            _, self.size, self.chunk, self.offset, _, _, _, _ = self.volume_info
            self.inith5()

            if self.dpResample_verbose:
                print('Resample chunk %d %d %d, size %d %d %d, offset %d %d %d' % tuple(self.chunk.tolist() + \
                    self.size.tolist() + self.offset.tolist())); t = time.time()
            self.readCubeToBuffers()

            if self.upsample:
                # update the scale and compute new chunk/size/offset
                self.data_attrs['scale'][self.resample_dims] /= 2
                new_chunk = self.chunk.copy()
                new_chunk[self.resample_dims]= new_chunk[self.resample_dims]*2
                new_size = self.size.copy()
                new_size[self.resample_dims]= new_size[self.resample_dims]*2
                new_offset = self.offset.copy()
                new_offset[self.resample_dims]= new_offset[self.resample_dims]*2

                new_data = np.zeros(new_size,dtype=self.data_type)
                for i in range(self.nslices):
                    new_data[self.slices[i]] = self.data_cube
            else:
                # update the scale and compute new chunk/size/offset
                self.data_attrs['scale'][self.resample_dims] *= 2
                new_chunk = self.chunk.copy()
                new_chunk[self.resample_dims]= new_chunk[self.resample_dims]//2
                new_size = self.size.copy()
                new_size[self.resample_dims]= new_size[self.resample_dims]//2
                new_offset = self.offset.copy()
                new_offset[self.resample_dims]= new_offset[self.resample_dims]//2

                if self.pixel_averaging:
                    new_data = np.zeros(new_size,dtype=np.double)
                    for i in range(self.nslices):
                        new_data += self.data_cube[self.slices[i]]
                    new_data = (new_data / self.nslices).astype(self.data_type)
                else:
                    new_data = self.data_cube[self.slices[0]]

            self.data_cube = new_data
            self.size, self.chunk, self.offset = new_size, new_chunk, new_offset
            self.inith5(); self.writeCube()
            if self.dpResample_verbose:
                print('\t\tdone in %.4f s' % (time.time() - t,))

    @staticmethod
    def addArgs(p):
        # adds arguments required for this object to specified ArgumentParser object
        dpWriteh5.addArgs(p)
        dpCubeIter.addArgs(p)
        p.add_argument('--upsample', action='store_true', help='Upsample mode (default downsampling)')
        p.add_argument('--pixel-averaging', action='store_true', help='Use pixel averaging method for downsampling')
        p.add_argument('--resample-dims', nargs=3, type=int, default=[1,1,1], metavar=('X', 'Y', 'Z'),
            help='Boolean specifying which dimensions to resample')
        p.add_argument('--dpResample-verbose', action='store_true', help='Debugging output for dpResample')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple factor of 2 up/down sampling script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpResample.addArgs(parser)
    args = parser.parse_args()

    resamp = dpResample(args)
    resamp.iterResample()
