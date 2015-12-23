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

# Python object for reading EM probabilities (affinities or voxel type probabilities) and merging probabilities from
#    multiple network runs with a weighted average, including runs in different reslice directions.


import os
import argparse
import time
import numpy as np

from dpLoadh5 import dpLoadh5
from dpWriteh5 import dpWriteh5
from typesh5 import emProbabilities

class dpMergeProbs(object):

    def __init__(self, args):
        # save command line arguments from argparse, see definitions in main or run with --help
        for k, v in vars(args).items(): 
            if type(v) is list: 
                # do not save items that are known to be lists (even if one element) as single elements
                if len(v)==1 and k not in ['srcfiles', 'affins', 'types', 'weightings', 'dim_orderings']:
                    setattr(self,k,v[0])  # save single element lists as first element
                elif type(v[0]) is int:   # convert the sizes and offsets to numpy arrays
                    setattr(self,k,np.array(v,dtype=np.int32))
                elif type(v[0]) is float:   # convert float arrays to numpy arrays
                    setattr(self,k,np.array(v,dtype=np.double))
                else:
                    setattr(self,k,v)   # store other list types as usual (floats)
            else:
                setattr(self,k,v)

        self.nsrcfiles = len(self.srcfiles)
        assert( self.nsrcfiles > 1 )    # no need to run merge on a single network output
        # empty affins indicates that inputs are voxel probabilities instead of affinities
        if not self.affins[0]: self.affins = []
        self.ntypes = len(self.types); self.naffins = len(self.affins) 
        #print(self.naffins)
        if self.naffins > 0:
            self.datasets = [[None for x in range(self.naffins)] for x in range(self.ntypes)] 
            for i in range(self.ntypes):
                for j in range(self.naffins):
                    self.datasets[i][j] = self.types[i] + self.affins[j]
        else:
            self.datasets = self.types
        assert( len(self.dim_orderings) == self.nsrcfiles )
        if (self.weightings < 0).all():
            # default all equal weightings
            if self.naffins > 0:
                self.weightings = np.ones((self.nsrcfiles, self.naffins), dtype=np.double)
            else:
                self.weightings = np.ones((self.nsrcfiles, 1), dtype=np.double)
        else:
            self.weightings = self.weightings.reshape((self.nsrcfiles,-1))
            assert( self.naffins == 0 or self.weightings.shape[1] == self.naffins )
            assert( self.naffins > 0 or self.weightings.shape[1] == 1 )
        self.srcpath = os.path.expandvars(os.path.expanduser(self.srcpath))

        # print out all initialized variables in verbose mode
        if self.dpMergeProbs_verbose: print('dpMergeProbs, verbose mode:\n'); print(vars(self))

    def merge_probs(self):
        if self.naffins > 0:
            affins = self.merge_affins_from_convnet_out(); szaffins = affins.shape
        else:
            probs = self.merge_probs_from_convnet_out()
        
        if self.naffins > 0:
            for i in range(dpLoadh5.ND):
                for j in range(self.ntypes):
                    dpWriteh5.writeData(outfile=self.outprobs, dataset=self.types[j]+'_DIM'+str(i), 
                        chunk=self.chunk.tolist(), offset=self.offset.tolist(), size=self.size.tolist(), 
                        datasize=self.datasize.tolist(), chunksize=self.chunksize.tolist(), attrs=self.attrs, 
                        data_type=emProbabilities.PROBS_STR_DTYPE, data=affins[:,:,:,i,j], 
                        verbose=self.dpMergeProbs_verbose)
        else: 
            for j in range(self.ntypes):
                print(self.types[j])
                dpWriteh5.writeData(outfile=self.outprobs, dataset=self.types[j], 
                    chunk=self.chunk.tolist(), offset=self.offset.tolist(), size=self.size.tolist(), 
                    datasize=self.datasize.tolist(), chunksize=self.chunksize.tolist(), attrs=self.attrs, 
                    data_type=emProbabilities.PROBS_STR_DTYPE, data=probs[:,:,:,j], 
                    verbose=self.dpMergeProbs_verbose)
                

    # this function takes the orthogonal reslice affinity outputs from trained convnets, also with multiple types
    #   (i.e., ICS, ECS, MEM, etc) and creates a single 3d affinity graph using the weighted average
    def merge_affins_from_convnet_out(self):
        szaffins = np.append(np.append(self.size,dpLoadh5.ND),self.ntypes)
        affins = np.zeros(szaffins, dtype=emProbabilities.PROBS_DTYPE, order='C')
        
        for k in range(self.ntypes):
            # xxx - for now use the same weightings for all types
            sum_weightings = np.zeros((dpLoadh5.ND,),dtype=emProbabilities.PROBS_DTYPE)
            for i in range(self.nsrcfiles):
                for j in range(self.naffins):
                    cz = dpLoadh5.RESLICES[self.dim_orderings[i]];  # current zreslice_order
                    cd = cz[j]  # current dimension being loaded in 3d
                    loadh5 = dpLoadh5.readData(srcfile=os.path.join(self.srcpath,self.srcfiles[i]), 
                        dataset=self.datasets[k][j], chunk=self.chunk[cz].tolist(), offset=self.offset[cz].tolist(), 
                        size=self.size[cz].tolist(), data_type=emProbabilities.PROBS_STR_DTYPE, 
                        verbose=self.dpMergeProbs_verbose)
                    self.datasize = loadh5.datasize[cz]; self.chunksize = loadh5.chunksize[cz]
                    self.attrs = loadh5.data_attrs
                    #print(self.weightings[i,j], cd)
                    affins[:,:,:,cd,k] += loadh5.data_cube.transpose(cz)*self.weightings[i,j]
                    sum_weightings[cd] += self.weightings[i,j]
                    del loadh5
            sum_weightings[sum_weightings == 0] = 1 # do not divide by zero because of any dimensions not contributing
            affins[:,:,:,:,k] /= sum_weightings
        return affins

    # this function merges outputs that are the probability of voxel id (given in types). 
    def merge_probs_from_convnet_out(self):
        szprobs = np.append(self.size,self.ntypes)
        probs = np.zeros(szprobs, dtype=emProbabilities.PROBS_DTYPE, order='C')
        
        for k in range(self.ntypes):
            for i in range(self.nsrcfiles):
                cz = dpLoadh5.RESLICES[self.dim_orderings[i]];  # current zreslice_order
                loadh5 = dpLoadh5.readData(srcfile=os.path.join(self.srcpath,self.srcfiles[i]), 
                    dataset=self.datasets[k], chunk=self.chunk[cz].tolist(), offset=self.offset[cz].tolist(), 
                    size=self.size[cz].tolist(), data_type=emProbabilities.PROBS_STR_DTYPE, 
                    verbose=self.dpMergeProbs_verbose)
                self.datasize = loadh5.datasize[cz]; self.chunksize = loadh5.chunksize[cz]
                self.attrs = loadh5.data_attrs
                probs[:,:,:,k] += loadh5.data_cube.transpose(cz)*self.weightings[i]; del loadh5
            probs[:,:,:,k] /= self.weightings.sum()
        return probs

    @staticmethod
    def addArgs(p):
        # adds arguments required for this object to specified ArgumentParser object
        p.add_argument('--srcpath', nargs=1, type=str, default='.', help='Location to hdf5 input files')
        p.add_argument('--srcfiles', nargs='+', type=str, default=['tmp.h5'], metavar='FILE',
            help='Input hdf5 files containing affinity probabilities')
        p.add_argument('--affins', nargs='+', type=str, default=[''], 
            metavar='NAME', help='Name of the affinities for each type (suffix for dataset). Use "" for voxel probs')
        #p.add_argument('--affins', nargs='+', type=str, default=['_DIM0POS', '_DIM1POS'], 
        #    metavar='NAME', help='Name of the affinities for each type (suffix for dataset). Use "" for voxel probs')
        p.add_argument('--types', nargs='+', type=str, default=['ICS','ECS','MEM'], 
            metavar='TYPE', help='Names of the voxel types (prefix for dataset for affinities, all combinations)')
        p.add_argument('--weightings', nargs='+', type=float, default=[-1.0,-1.0], metavar='W', 
            help='Weightings for probabilities specified in each srcfile (un-reordered), default to equal weightings')
        p.add_argument('--dim-orderings', nargs='+', type=str, default='xyz', choices=('xyz','xzy','zyx'),
            metavar='ORD', help='Specify the order to reslice the dimensions into (last one becomes new z)')
        p.add_argument('--chunk', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
            help='Corner chunk to parse out of hdf5')
        p.add_argument('--offset', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
            help='Offset in chunk to read')
        p.add_argument('--size', nargs=3, type=int, default=[256,256,128], metavar=('X', 'Y', 'Z'),
            help='Size in voxels to read')
            
        p.add_argument('--outprobs', nargs=1, type=str, default='', metavar='FILE', 
            help='Voxel type probs h5 output file')
        p.add_argument('--dpMergeProbs-verbose', action='store_true', 
            help='Debugging output for dpMergeProbs')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(\
        description='Read EM affinity probability data from h5 inputs and merge using weighted average',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpMergeProbs.addArgs(parser)
    args = parser.parse_args()
    
    mrg = dpMergeProbs(args)
    mrg.merge_probs()

