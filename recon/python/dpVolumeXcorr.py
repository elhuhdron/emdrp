#!/usr/bin/env python

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

# Python translation of volume_xcorr.m, based on code from kb for doing xcorr in fourier domain
# original codes from kb: Analyze_test_vs_train_xcorr_001.m, normxcorr2_kb.m

import numpy as np
#import h5py
import argparse
import time
import glob
import os

from dpLoadh5 import dpLoadh5
from dpWriteh5 import dpWriteh5
from typesh5 import emProbabilities, emVoxelType

class dpVolumeXcorr(dpWriteh5):

    use_scipy_fft = 0; use_numpy_fft = 1; use_pyfftw_fft = 2
    use_fft = use_pyfftw_fft

    def __init__(self, args):
        self.LIST_ARGS += ['train_offsets', 'prob_types']
        dpWriteh5.__init__(self,args)

        self.nprob_types = len(self.prob_types)

        self.train_chunks = self.train_chunks.reshape((-1,3))
        self.ntrain_chunks = self.train_chunks.shape[0]
        if len(self.train_offsets) == 0:
            self.train_offsets = np.zeros_like(self.train_chunks)
        else:
            self.train_offsets = np.array(self.train_chunks).reshape((-1,3))
            assert( self.ntrain_chunks == self.train_offsets.shape[0] )
        if (self.train_size < 1).all():
            self.train_size = np.array(self.chunksize)

        assert( (self.size[:2] % self.test_size == 0).all() )
        self.ntest = self.size[:2] // self.test_size
        self.nztrain = self.train_size[2] * self.ntrain_chunks

        # print out all initialized variables in verbose mode
        if self.dpVolumeXcorr_verbose:
            print('dpVolumeXcorr, verbose mode:\n'); print(vars(self))

    def loadData(self):
        if self.dpVolumeXcorr_verbose:
            print('Loading data'); t = time.time()

        # load the raw (test) data
        self.readCubeToBuffers()
        self.test_data = self.data_cube
        self.data_cube = np.zeros((0,))

        # load the raw (trained) data
        self.train_data = [None]*self.ntrain_chunks
        for i in range(self.ntrain_chunks):
            loadh5 = dpLoadh5.readData(srcfile=self.srcfile, dataset=self.dataset,chunk=self.train_chunks[i,:].tolist(),
                offset=self.train_offsets[i,:].tolist(), size=self.train_size.tolist(),
                verbose=self.dpLoadh5_verbose); self.train_data[i] = loadh5.data_cube
        # unroll training cubes
        self.train_data = np.dstack(self.train_data)
        self.ntrain_data = self.train_data.shape[2]

        if self.probfile:
            # load the probability data
            self.probs = [None]*self.nprob_types
            for i in range(self.nprob_types):
                loadh5 = dpLoadh5.readData(srcfile=self.probfile, dataset=self.prob_types[i], chunk=self.chunk.tolist(),
                    offset=self.offset.tolist(), size=self.size.tolist(), data_type=emProbabilities.PROBS_STR_DTYPE,
                    verbose=self.dpLoadh5_verbose); self.probs[i] = loadh5.data_cube

        # load the voxel type data
        voxType = emVoxelType.readVoxType(srcfile=self.typefile, chunk=self.chunk.tolist(),
            offset=self.offset.tolist(), size=self.size.tolist()); self.voxel_type = voxType.data_cube

        if self.dpVolumeXcorr_verbose:
            print('\tdone in %.4f s, %d train images' % (time.time() - t, self.ntrain_data))

    def xcorr(self):
        from scipy import fftpack as scipy_fft
        from numpy import fft as numpy_fft
        import pyfftw

        # allocate the outputs
        szC = np.hstack((self.ntest, self.size[2]))
        #szP = np.hstack((szC, self.nprob_types))
        Cout = np.zeros(szC, dtype=np.double)
        Pcout = [None]*self.nprob_types
        Poutm = [None]*self.nprob_types
        Pouts = [None]*self.nprob_types
        for i in range(self.nprob_types):
            Pcout[i] = np.zeros(szC, dtype=np.double)
            if self.probfile:
                Poutm[i] = np.zeros(szC, dtype=np.double)
                Pouts[i] = np.zeros(szC, dtype=np.double)

        # kevin's precompute fft loop
        if self.dpVolumeXcorr_verbose:
            print('precompute fft loop'); t = time.time()
        m,n = self.test_size; mn = m*n
        A_size = self.train_size[:2]
        T_size = self.test_size
        outsize = A_size + T_size - 1
        szO = np.hstack((outsize, self.ntrain_data))

        local_sum_A = np.zeros(szO, dtype=np.double)
        denom_A = np.zeros(szO, dtype=np.double)
        Fb = np.zeros(szO, dtype=np.complex64)

        if self.use_fft == dpVolumeXcorr.use_pyfftw_fft:
            fftA = pyfftw.builders.fft2(pyfftw.empty_aligned(A_size, dtype='complex64'),
                                        s=outsize, threads=self.nthreads)
            # input array is resized to output, so take a slice for the input array assignment
            _A = fftA.input_array; _A[:] = 0 + 0j; A = _A[:A_size[0],:A_size[1]]
        else:
            A = np.empty(A_size, dtype=np.single)

        for traincount in range(self.ntrain_data):
            Ad = self.train_data[:,:,traincount].astype(np.double)
            local_sum_A[:,:,traincount] = dpVolumeXcorr.local_sum(Ad,m,n)
            local_sum_A2 = dpVolumeXcorr.local_sum(Ad*Ad,m,n)
            diff_local_sums = ( local_sum_A2 - (local_sum_A[:,:,traincount]**2)/mn )
            denom_A[:,:,traincount] = np.sqrt( np.maximum(diff_local_sums, 0) )
            A[:,:] = self.train_data[:,:,traincount]
            if self.use_fft == dpVolumeXcorr.use_scipy_fft:
                Fb[:,:,traincount] = scipy_fft.fft2(A,shape=outsize)
            elif self.use_fft == dpVolumeXcorr.use_numpy_fft:
                Fb[:,:,traincount] = numpy_fft.fft2(A,s=outsize)
            elif self.use_fft == dpVolumeXcorr.use_pyfftw_fft:
                Fb[:,:,traincount] = fftA()
        #from scipy.io import savemat
        #savemat('tmp'  + str(self.use_fft),{'Fb':Fb})

        if self.dpVolumeXcorr_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))
            print('running %d x %d test images for %d slices' % (self.ntest[0],self.ntest[1],self.size[2]))

        # compute correlations over all test_data tiles using precomputed train_data ffts
        Fa = np.zeros(outsize, dtype=np.complex64)
        C = np.zeros((self.ntrain_data),dtype=np.double)

        if self.use_fft == dpVolumeXcorr.use_pyfftw_fft:
            fftT = pyfftw.builders.fft2(pyfftw.empty_aligned(T_size, dtype='complex64'), s=outsize,
                                        threads=self.nthreads, planner_effort='FFTW_EXHAUSTIVE')
            # input array is resized to output, so take a slice for the input array assignment
            _T = fftT.input_array; _T[:] = 0 + 0j; T = _T[:T_size[0],:T_size[1]]

            fftF = pyfftw.builders.ifft2(pyfftw.empty_aligned(outsize, dtype='complex64'),
                                         threads=self.nthreads, planner_effort='FFTW_EXHAUSTIVE')
            F = fftF.input_array
        else:
            T = np.empty(T_size, dtype=np.single)
            F = np.empty(outsize, dtype=np.complex64)

        for x in range(self.ntest[0]):
            xrng = slice(x*self.test_size[0],(x+1)*self.test_size[0])
            for y in range(self.ntest[1]):
                yrng = slice(y*self.test_size[1],(y+1)*self.test_size[1])
                if self.dpVolumeXcorr_verbose:
                    print('processing cube x %d/%d y %d/%d' % (x,self.ntest[0],y,self.ntest[1])); touter = time.time()
                for z in range(self.size[2]):
                    Td = self.test_data[xrng,yrng,z].astype(np.double)

                    # moved these outside inner loop, not dependent on training data
                    denom_T = np.sqrt(mn-1)*Td.std(dtype=np.double)
                    Tdsum_mn = Td.sum(dtype=np.double)/mn

                    # xxx - major optimization difference with matlab code, moved this outside of inner loop
                    T[:,:] = np.rot90(self.test_data[xrng,yrng,z], k=2)
                    if self.use_fft == dpVolumeXcorr.use_scipy_fft:
                        Fa[:,:] = scipy_fft.fft2(T,shape=outsize)
                    elif self.use_fft == dpVolumeXcorr.use_numpy_fft:
                        Fa[:,:] = numpy_fft.fft2(T,s=outsize)
                    elif self.use_fft == dpVolumeXcorr.use_pyfftw_fft:
                        Fa[:,:] = fftT()

                    # kevin's xcorr inner loop
                    for traincount in range(self.ntrain_data):
                        # translated from kb's code (taken from matlab central???), unrolled function into loop
                        # Modified 01/15/2016 by KB to use precomputed training cube FFTs, etc...
                        #NORMXCORR2 Normalized two-dimensional cross-correlation.
                        #   C = NORMXCORR2(TEMPLATE,A) computes the normalized cross-correlation of
                        #   matrices TEMPLATE and A. The matrix A must be larger than the matrix
                        #   TEMPLATE for the normalization to be meaningful. The values of TEMPLATE
                        #   cannot all be the same. The resulting matrix C contains correlation
                        #   coefficients and its values may range from -1.0 to 1.0.

                        #   We normalize the cross correlation to get correlation coefficients using the
                        #   definition of Haralick and Shapiro, Volume II (p. 317), generalized to
                        #   two-dimensions.
                        #
                        #   Lewis explicitly defines the normalized cross-correlation in two-dimensions
                        #   in this paper (equation 2):
                        #
                        #      "Fast Normalized Cross-Correlation", by J. P. Lewis, Industrial Light & Magic.
                        #      http://www.idiom.com/~zilla/Papers/nvisionInterface/nip.html
                        #
                        #   Our technical reference document on NORMXCORR2 shows how to get from
                        #   equation 2 of the Lewis paper to the code below.
                        # function C = normxcorr2_kb(T,A,local_sum_A,denom_A,Fb)
                        #RESULT = normxcorr2_kb(single(thistestimg),single(A),local_sum_A(:,:,traincount),...
                        #    denom_A(:,:,traincount),Fb(:,:,traincount));
                        F[:,:] = Fa*Fb[:,:,traincount]
                        if self.use_fft == dpVolumeXcorr.use_scipy_fft:
                            xcorr_TA = np.real(scipy_fft.ifft2(F))
                        elif self.use_fft == dpVolumeXcorr.use_numpy_fft:
                            xcorr_TA = np.real(numpy_fft.ifft2(F))
                        elif self.use_fft == dpVolumeXcorr.use_pyfftw_fft:
                            xcorr_TA = np.real(fftF())

                        # xxx - major optimization difference with matlab code, moved this outside of inner loop
                        #T[:,:] = np.rot90(self.test_data[xrng,yrng,z], k=2)
                        #if self.use_fft == dpVolumeXcorr.use_scipy_fft:
                        #    Fa[:,:] = scipy_fft.fft2(T,shape=outsize)
                        #elif self.use_fft == dpVolumeXcorr.use_numpy_fft:
                        #    Fa[:,:] = numpy_fft.fft2(T,s=outsize)
                        #elif self.use_fft == dpVolumeXcorr.use_pyfftw_fft:
                        #    Fa[:,:] = fftT()

                        denom = denom_T*denom_A[:,:,traincount]
                        numerator = (xcorr_TA - local_sum_A[:,:,traincount]*Tdsum_mn)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            Ci = numerator / denom
                        # end of original normxcorr2_kb function call in matlab code

                        Ci[np.isinf(Ci)] = 0; # when denom==0
                        # xxx - difference with matlab code? ignore nans (0/0 potentially possible)
                        C[traincount] = np.nanmax(Ci)

                    # take as the correlation the max correlation between all the training images and this test tile
                    Cout[x,y,z] = C.max();
                    #print(Cout[x,y,z])

                    # also record probability and winner take all stats for each test tile
                    thistesttypes = self.voxel_type[xrng,yrng,z]
                    for i in range(self.nprob_types):
                        winner_sel = (thistesttypes==i)
                        Pcout[i][x,y,z] = winner_sel.sum(dtype=np.double)/mn
                        if self.probfile and Pcout[i][x,y,z] > 0:
                            thistestprobs = self.probs[i][xrng,yrng,z]
                            winner_probs = thistestprobs[winner_sel]
                            # xxx - what measures to calculate here?
                            Poutm[i][x,y,z] = winner_probs.mean()
                            Pouts[i][x,y,z] = winner_probs.std()

                if self.dpVolumeXcorr_verbose:
                    print('\tdone in %.4f s' % (time.time() - touter, ))

        if self.savefile:
            if self.dpVolumeXcorr_verbose:
                print('saving output data'); t = time.time()

            np.savez(self.savefile, Cout=Cout, Pcout=Pcout, Poutm=Poutm, Pouts=Pouts,
                     prob_types=self.prob_types, chunk=self.chunk, offset=self.offset, size=self.size)

            if self.dpVolumeXcorr_verbose:
                print('\tdone in %.4f s' % (time.time() - t, ))

    def doplots(self):
        from matplotlib import pylab as pl
        #import matplotlib as plt

        o = np.load(self.loadfile)
        self.nprob_types = len(o['Pcout'])
        self.ntest = o['Cout'].shape[:2]

        # the plot is dead, long live the plot, huzzah!
        #m = plt.markers
        clrs = ['r','g','b','m','c','y','k']; #markers = ['x','+',m.CARETLEFT,m.CARETRIGHT,m.CARETUP,m.CARETDOWN,'*']

        if len(o['Poutm']) > 0 and o['Poutm'][0] is not None:
            pl.figure(1)
            pl.subplot(1,2,1)
            for i in range(self.nprob_types):
                pl.scatter(o['Cout'], o['Poutm'][i], s=8, c=clrs[i], alpha=0.5)
            pl.legend(o['prob_types'])
            pl.ylabel('mean winning prob')
            pl.xlabel('correlation')
            pl.subplot(1,2,2)
            for i in range(self.nprob_types):
                pl.scatter(o['Cout'], o['Pouts'][i], s=8, c=clrs[i], alpha=0.5)
            pl.legend(o['prob_types'])
            pl.ylabel('std winning prob')
            pl.xlabel('correlation')

        pl.show()

    def concatenate(self):
        loadfiles = glob.glob(os.path.join(self.loadfiles_path, '*.npz')); nFiles = len(loadfiles);
        if self.dpVolumeXcorr_verbose:
            print('Concatenating over %d load files' % (nFiles,))

        for i in range(nFiles):
            o = np.load(loadfiles[i])
            assert( (o['offset'] == 0).all() ) # did not see a reason to deal with non-chunk alignment
            assert( (o['size'] % self.chunksize == 0).all() ) # did not see a reason to deal with non-chunk alignment
            if i==0:
                size = o['size']; ntest = np.array(o['Cout'].shape)
                test_size = size // ntest  # already asserted divisible when xcorr run
                assert( (self.reduce_size % test_size == 0).all() )
                reduce_ntest = self.reduce_size // test_size
                # xxx - thought of making this general, but essentially z-direction is always per slice
                #   so reduce z locally per test block concatenation and then do final reduction afterwards
                concat_ntest = ntest
                if self.reduce_size[2] < size[2]:
                    assert( ntest[2] % self.reduce_size[2]  == 0 )
                    concat_ntest[2] = ntest[2] // self.reduce_size[2]
                    concat_zreduce = self.reduce_size[2]
                else:
                    assert( self.reduce_size[2] % size[2] == 0 )
                    concat_ntest[2] = size[2]
                    concat_zreduce = size[2]
                concat_zreduce_rng = range(0,size[2],concat_zreduce)

                nprob_types = len(o['prob_types'])

                # allocate concatenated outputs (assigned after per-block z reduction)
                concat_size = self.concat_nchunks * concat_ntest
                concat_Cout = np.zeros(concat_size, dtype=np.double)
                concat_Pcout = [np.zeros(concat_size, dtype=np.double) for i in range(nprob_types)]

                #print(size,test_size,ntest,self.reduce_size,concat_size,concat_ntest)
            else:
                assert( (size == o['size']).all() and (ntest == o['Cout']).all() )
                assert( len(o['prob_types']) == nprob_types )
            chunk = o['chunk']

            # assign with the z-accumulation
            inds = (chunk - self.concat_chunk)*self.chunksize//size
            concat_Cout[inds[0]:inds[0]+concat_ntest[0], inds[1]:inds[1]+concat_ntest[1],
                        inds[2]:inds[2]+concat_ntest[2]] = np.add.reduceat(o['Cout'], concat_zreduce_rng,
                            axis=2)/concat_zreduce
            for i in range(nprob_types):
                concat_Pcout[i][inds[0]:inds[0]+concat_ntest[0], inds[1]:inds[1]+concat_ntest[1],
                            inds[2]:inds[2]+concat_ntest[2]] = np.add.reduceat(o['Pcout'][i], concat_zreduce_rng,
                                axis=2)/concat_zreduce

        # now do the final reduction over the whole concatenated volume.
        # have to do z-first as to avoid the unequal average of averages scenario.
        Cout = concat_Cout; Pcout = concat_Pcout
        for j in range(dpLoadh5.ND)[::-1]:
            if j==2:
                n = self.reduce_size[j] // size[j]
            else:
                n = reduce_ntest[j]
            if n > 1:
                Cout = np.add.reduceat(Cout, range(0,concat_size[j],n), axis=j)/n
                for i in range(nprob_types):
                    Pcout[i] = np.add.reduceat(Pcout[i], range(0,concat_size[j],n), axis=j)/n
        print(Cout, Pcout[0])

        if self.savefile:
            if self.dpVolumeXcorr_verbose:
                print('saving output data'); t = time.time()

            np.savez(self.savefile, Cout=Cout, Pcout=Pcout,
                     prob_types=o['prob_types'], concat_chunk=self.concat_chunk, concat_nchunks=self.concat_nchunks,
                     reduce_size=self.reduce_size, ntest=ntest, size=size)

            if self.dpVolumeXcorr_verbose:
                print('\tdone in %.4f s' % (time.time() - t, ))


    # translated from kb's code (taken from matlab central???)
    # Another numerics backstop. If any of the coefficients are outside the
    # range [-1 1], the numerics are unstable to small variance in A or T. In
    # these cases, set C to zero to reflect undefined 0/0 condition.
    # C( ( abs(C) - 1 ) > sqrt(eps(1)) ) = 0;
    # toc(t1)
    #-------------------------------
    # Function  local_sum
    #
    #function local_sum_A = local_sum(A,m,n)
    @staticmethod
    def local_sum(A,m,n):
        # We thank Eli Horn for providing this code, used with his permission,
        # to speed up the calculation of local sums. The algorithm depends on
        # precomputing running sums as described in "Fast Normalized
        # Cross-Correlation", by J. P. Lewis, Industrial Light & Magic.
        # http://www.idiom.com/~zilla/Papers/nvisionInterface/nip.html

        B = np.lib.pad(A, ((m,m),(n,n)), 'constant', constant_values=0)
        s = np.cumsum(B, axis=0)
        c = s[m:-1,:]-s[:-m-1,:]
        s = np.cumsum(c, axis=1)
        return s[:,n:-1]-s[:,:-n-1]

    @staticmethod
    def addArgs(p):
        # adds arguments required for this object to specified ArgumentParser object
        dpWriteh5.addArgs(p)
        p.add_argument('--probfile', nargs=1, type=str, default='', help='Path/name of hdf5 probability (input) file')
        p.add_argument('--prob_types', nargs='+', type=str, default=['MEM'],
            metavar='TYPE', help='Dataset names of the voxel types to use from the probabilities')
        p.add_argument('--typefile', nargs=1, type=str, default='', help='Path/name of hdf5 with voxel types')
        p.add_argument('--train-chunks', nargs='+', type=int, default=[0,0,0], metavar='X_Y_Z',
            help='Training chunks (specify x0 y0 z0 x1 y1 ...) ')
        p.add_argument('--train-offsets', nargs='*', type=int, default=[], metavar='X_Y_Z',
            help='Training offsets (specify x0 y0 z0 x1 y1 ..., default all zero) ')
        p.add_argument('--train-size', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
            help='Size of training chunks (default chunksize)')
        p.add_argument('--test-size', nargs=2, type=int, default=[64, 64], metavar='S',
            help='Size of the sliding correlation windows')
        p.add_argument('--savefile', nargs=1, type=str, default='', help='Path/name npz file to save outputs to')
        p.add_argument('--loadfile', nargs=1, type=str, default='', help='Load previous run saved in npz for plotting')
        p.add_argument('--nthreads', nargs=1, type=int, default=[8], help='Number of threads to use in fftw')

        # arguments for concatenate mode (concatenate runs over dpCubeIter volumes and reduce down to specified size)
        p.add_argument('--loadfiles-path', nargs=1, type=str, default='', help='Path to saved runs to concatenate')
        p.add_argument('--concat-chunk', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
            help='Starting chunk alignment for concatenate')
        p.add_argument('--concat-nchunks', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
            help='Total area for concatenation (chunks)')
        p.add_argument('--reduce-size', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
            help='Block size to reduce down to after concantenation (voxels)')

        p.add_argument('--dpVolumeXcorr-verbose', action='store_true', help='Debugging output for dpVolumeXcorr')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Fourier domain raw data correlations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpVolumeXcorr.addArgs(parser)
    args = parser.parse_args()

    vxcorr = dpVolumeXcorr(args)
    if vxcorr.loadfiles_path:
        vxcorr.concatenate()
    elif vxcorr.loadfile:
        vxcorr.doplots()
    else:
        vxcorr.loadData()
        vxcorr.xcorr()
