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

# Method for doing simple divisible integer factor downsampling/upsampling.
# Upsampling simply repeats pixels (intended for upsampling labels only, not for interpolation).
# Downsampling can just decimate without transformation or do "pixel mixing".

import numpy as np
#import h5py
import argparse
import time
#import glob
#import os

from dpLoadh5 import dpLoadh5
from dpWriteh5 import dpWriteh5
#from utils.typesh5 import emProbabilities, emVoxelType
from dpCubeIter import dpCubeIter

class dpResample(dpWriteh5):

    def __init__(self, args):
        self.LIST_ARGS += dpCubeIter.LIST_ARGS
        dpWriteh5.__init__(self,args)

        # xxx - also semi-unclean, would fix along with cleaner in/out method
        self.dataset_in = self.dataset
        self.datasize_in = self.datasize

        self.resample_dims = self.resample_dims.astype(bool)
        self.nresample_dims = self.resample_dims.sum(dtype=np.uint8)
        assert( self.nresample_dims > 0 )   # no resample dims specified
        self.nslices = self.factor**self.nresample_dims

        # print out all initialized variables in verbose mode
        if self.dpResample_verbose:
            print('dpResample, verbose mode:\n'); print(vars(self))

        ## xxx - probably a way to do this programatically, but easier to read as enumerated.
        ##   this code commented out was only for downsampling by factor of 2
        #if (self.resample_dims == np.array([1,0,0])).all():
        #    self.slices = [np.s_[::2,:,:], np.s_[1::2,:,:]]
        #elif (self.resample_dims == np.array([0,1,0])).all():
        #    self.slices = [np.s_[:,::2,:], np.s_[:,1::2,:]]
        #elif (self.resample_dims == np.array([0,0,1])).all():
        #    self.slices = [np.s_[:,:,::2], np.s_[:,:,1::2]]
        #elif (self.resample_dims == np.array([1,1,0])).all():
        #    self.slices = [np.s_[::2,::2,:], np.s_[1::2,::2,:], np.s_[::2,1::2,:], np.s_[1::2,1::2,:]]
        #elif (self.resample_dims == np.array([1,0,1])).all():
        #    self.slices = [np.s_[::2,:,::2], np.s_[1::2,:,::2], np.s_[::2,:,1::2], np.s_[1::2,:,1::2]]
        #elif (self.resample_dims == np.array([0,1,1])).all():
        #    self.slices = [np.s_[:,::2,::2], np.s_[:,1::2,::2], np.s_[:,::2,1::2], np.s_[:,1::2,1::2]]
        #elif self.resample_dims.all():
        #    self.slices = [np.s_[::2,::2,::2], np.s_[1::2,::2,::2], np.s_[::2,1::2,::2], np.s_[::2,::2,1::2],
        #                   np.s_[1::2,1::2,::2], np.s_[1::2,::2,1::2], np.s_[::2,1::2,1::2], np.s_[1::2,1::2,1::2]]
        #assert( len(self.slices) == self.nslices ) # sanity check

        # programmatic for factor, but still not for dimensions, again didn't seem worth it, always 3d
        self.slices = [None]*self.nslices; f = self.factor
        if (self.resample_dims == np.array([1,0,0])).all():
            for i in range(f):
                self.slices[i] = np.s_[i::f,:,:]
        elif (self.resample_dims == np.array([0,1,0])).all():
            for i in range(f):
                self.slices[i] = np.s_[:,i::f,:]
        elif (self.resample_dims == np.array([0,0,1])).all():
            for i in range(f):
                self.slices[i] = np.s_[:,:,i::f]
        elif (self.resample_dims == np.array([1,1,0])).all():
            for i in range(f):
                for j in range(f):
                    self.slices[i*f + j] = np.s_[i::f,j::f,:]
        elif (self.resample_dims == np.array([1,0,1])).all():
            for i in range(f):
                for j in range(f):
                    self.slices[i*f + j] = np.s_[i::f,:,j::f]
        elif (self.resample_dims == np.array([0,1,1])).all():
            for i in range(f):
                for j in range(f):
                    self.slices[i*f + j] = np.s_[:,i::f,j::f]
        elif self.resample_dims.all():
            ff = f*f
            for i in range(f):
                for j in range(f):
                    for k in range(f):
                        self.slices[i*ff + j*f + k] = np.s_[i::f,j::f,k::f]

    def iterResample(self):
        assert( (self.cube_size[self.resample_dims] % self.factor == 0).all() ) # xxx - this probably could be fixed

        # xxx - ahhhhhh, this has to be fixed somehow
        if self.chunksize is not None and (self.chunksize < 0).all(): self.chunksize = self.use_chunksize
        self.cubeIter = dpCubeIter.cubeIterGen(self.volume_range_beg,self.volume_range_end,self.overlap,self.cube_size,
                    left_remainder_size=self.left_remainder_size, right_remainder_size=self.right_remainder_size,
                    chunksize=self.chunksize, leave_edge=self.leave_edge)

        for self.volume_info,n in zip(self.cubeIter, range(self.cubeIter.volume_size)):
            _, self.size, self.chunk, self.offset, _, _, _, _, _ = self.volume_info
            self.singleResample()

    def singleResample(self):
        self.dataset = self.dataset_in
        self.datasize = self.datasize_in
        self.inith5()
        assert( (self.size[self.resample_dims] % self.factor == 0).all() )

        if self.dpResample_verbose:
            print('Resample chunk %d %d %d, size %d %d %d, offset %d %d %d' % tuple(self.chunk.tolist() + \
                self.size.tolist() + self.offset.tolist())); t = time.time()
        self.readCubeToBuffers()

        new_attrs = self.data_attrs
        # changed this to be added when raw hdf5 is created
        if 'factor' not in new_attrs:
            new_attrs['factor'] = np.ones((dpLoadh5.ND,),dtype=np.double)
        new_chunk = self.chunk.copy()
        new_size = self.size.copy()
        new_offset = self.offset.copy()
        new_datasize = self.datasize.copy()

        f = self.factor
        if self.upsample:
            # update the scale and compute new chunk/size/offset
            if 'boundary' in new_attrs:
                new_attrs['boundary'][self.resample_dims] *= f
                new_attrs['nchunks'][self.resample_dims] *= f
            if 'scale' in new_attrs:
                new_attrs['scale'][self.resample_dims] /= f
            # this attribute is saved as downsample factor
            new_attrs['factor'][self.resample_dims] /= f
            new_chunk[self.resample_dims] *= f
            new_size[self.resample_dims] *= f
            new_offset[self.resample_dims] *= f
            new_datasize[self.resample_dims] *= f

            new_data = np.zeros(new_size,dtype=self.data_type)
            for i in range(self.nslices):
                new_data[self.slices[i]] = self.data_cube
        else:
            # update the scale and compute new chunk/size/offset
            if 'boundary' in new_attrs:
                new_attrs['boundary'][self.resample_dims] //= f
                new_attrs['nchunks'][self.resample_dims] = \
                    np.ceil(new_attrs['nchunks'][self.resample_dims] / f).astype(np.int32)
            if 'scale' in new_attrs:
                new_attrs['scale'][self.resample_dims] *= f
            # this attribute is saved as downsample factor
            new_attrs['factor'][self.resample_dims] *= f
            new_chunk[self.resample_dims] //= f
            new_size[self.resample_dims] //= f
            new_offset[self.resample_dims] //= f
            new_datasize[self.resample_dims] //= f

            # update offset for non-divisible chunks
            rmd_chunks = (self.chunk % f != 0)
            sel = (self.resample_dims & rmd_chunks)
            new_offset[sel] += self.chunksize[sel]//f

            if self.downsample_op == 'none':
                new_data = self.data_cube[self.slices[0]]
            elif self.downsample_op == 'labels':
                # same as none except zero if any voxels are zero
                new_data = np.zeros(np.concatenate([new_size, [self.nslices]]),dtype=np.double)
                for i in range(self.nslices):
                    new_data[:,:,:,i] = self.data_cube[self.slices[i]]
                sel = np.any(new_data==0, axis=3)
                new_data = self.data_cube[self.slices[0]]
                new_data[sel] = 0
            elif self.downsample_op == 'mean':
                new_data = np.zeros(new_size,dtype=np.double)
                for i in range(self.nslices):
                    new_data += self.data_cube[self.slices[i]]
                new_data = (new_data / self.nslices).astype(self.data_type)
            elif self.downsample_op == 'median':
                new_data = np.zeros(np.concatenate([new_size, [self.nslices]]),dtype=np.double)
                for i in range(self.nslices):
                    new_data[:,:,:,i] = self.data_cube[self.slices[i]]
                new_data = np.median(new_data, axis=3).astype(self.data_type)

        self.size, self.chunk, self.offset = new_size, new_chunk, new_offset
        if self.dpResample_verbose:
            print('\tdone in %.4f s' % (time.time() - t,))
            print('\twrite to chunk %d %d %d, size %d %d %d, offset %d %d %d' % tuple(self.chunk.tolist() + \
                self.size.tolist() + self.offset.tolist())); t = time.time()
        self.inith5()
        self.data_cube = new_data
        self.data_attrs = new_attrs
        self.datasize = new_datasize
        self.writeCube()
        if self.dpResample_verbose:
            print('\t\tdone in %.4f s' % (time.time() - t,))

    @staticmethod
    def addArgs(p):
        # adds arguments required for this object to specified ArgumentParser object
        dpWriteh5.addArgs(p)
        dpCubeIter.addArgs(p)
        p.add_argument('--upsample', action='store_true', help='Upsample mode (default downsampling)')
        p.add_argument('--downsample-op', nargs=1, type=str, default=['none'], metavar='OP',
                       choices=['none','labels','mean','median'],
                       help='Specify which operation to use for downsampling method')
        p.add_argument('--factor', nargs=1, type=int, default=[2], metavar=('F'),
                       help='Integer factor to resample, must divide size of resampled dims')
        p.add_argument('--resample-dims', nargs=3, type=int, default=[1,1,1], metavar=('X', 'Y', 'Z'),
            help='Boolean specifying which dimensions to resample')
        p.add_argument('--dpResample-verbose', action='store_true', help='Debugging output for dpResample')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple divisible integer factor up/down sampling script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpResample.addArgs(parser)
    args = parser.parse_args()

    resamp = dpResample(args)
    if (resamp.cube_size < 1).any():
        resamp.singleResample()
    else:
        resamp.iterResample()
