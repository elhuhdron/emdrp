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

# Extends emProbs class for merging voxel_type probabilities there were written as knossos-style
#   into hdf5 output containing merged probs (not intended for affinities or for hdf5 inputs, use dpMergeProbs).

#import h5py
import numpy as np
import argparse
import time
import os

#from dpLoadh5 import dpLoadh5
from dpWriteh5 import dpWriteh5
from utils.typesh5 import emProbabilities

class dpAggProbs(emProbabilities):

    def __init__(self, args):
        self.LIST_ARGS += ['weightings', 'agg_ops_types', 'dim_orderings']
        emProbabilities.__init__(self,args)

        self.ntypes = len(self.types)
        if len(self.agg_ops_types) == 0:
            # default just means
            self.agg_ops_types = [['mean'] for i in range(self.ntypes)]
        else:
            agg_ops_types = [[] for i in range(self.ntypes)]; i = 0
            for op in self.agg_ops_types:
                if op == '0': i += 1; continue
                assert( i < self.ntypes )
                agg_ops_types[i] += [op]
            for i in range(self.ntypes):
                # if op for any types are empty, default mean
                if len(agg_ops_types[i]) == 0:
                    agg_ops_types[i] += ['mean']
            self.agg_ops_types = agg_ops_types
        self.nops = [len(x) for x in self.agg_ops_types]

        if len(self.weightings) == 0:
            # default all equal weightings
            self.weightings = np.ones((self.nmerge,), dtype=np.double)
        else:
            self.weightings = np.array(self.weightings, dtype=np.double)
            assert( self.weightings.size == self.nmerge )

        if len(self.dim_orderings) == 0:
            # default all xyz
            self.dim_orderings = ['xyz' for i in range(self.nmerge)]
        else:
            assert( len(self.dim_orderings) == self.nmerge )

        self.inrawpath = os.path.expandvars(os.path.expanduser(self.inrawpath))

        # print out all initialized variables in verbose mode
        if self.dpAggProbs_verbose: print('dpAggProbs, verbose mode:\n'); print(vars(self))
        assert( os.path.isdir(self.inrawpath) )

    def aggregate(self):
        if self.dpAggProbs_verbose:
            print('dpAggProbs: Aggregating')
            t = time.time()

        # this is an optimization to only selection portion of cube to write in order to support overlaps
        ovlp_sel = (self.overlap != 0); ovlp_size = self.size.copy()
        use_ovlp = ovlp_sel.any()
        if use_ovlp:
            ovlp_neg = (self.overlap < 0); ovlp_pos = (self.overlap > 0)
            ovlp_size[ovlp_sel] = self.overlap[ovlp_sel]; ovlp_size[ovlp_neg] = -ovlp_size[ovlp_neg]
            ovlp_offset = self.offset.copy()
            ovlp_offset[ovlp_neg] += (self.size[ovlp_neg] - ovlp_size[ovlp_neg])
            chunk_sel = np.ones(self.size, dtype=bool)
            str_pslcs = ['self.overlap[j]:,:,:', ':,self.overlap[j]:,:', ':,:,self.overlap[j]:']
            str_nslcs = [':self.overlap[j],:,:', ':,:self.overlap[j],:', ':,:,:self.overlap[j]']
            for j in range(dpWriteh5.ND):
                if ovlp_pos[j]:
                    exec('chunk_sel[' + str_pslcs[j] + '] = 0')
                elif ovlp_neg[j]:
                    exec('chunk_sel[' + str_nslcs[j] + '] = 0')
            orig_size = self.size; orig_offset = self.offset

        for j in range(self.ntypes):
            cprobs = np.zeros(np.append(ovlp_size, self.nmerge), dtype=emProbabilities.PROBS_DTYPE, order='C')
            fn = self.types[j].upper()
            for i in range(self.nmerge):
                # load the raw files
                self.inraw = os.path.join( self.inrawpath, fn + str(i) + '.f32' )
                if use_ovlp:
                    self.loadFromRaw(); cprobs[:,:,:,i] = self.data_cube[chunk_sel].reshape(ovlp_size)
                else:
                    self.loadFromRaw(); cprobs[:,:,:,i] = self.data_cube

            for k in range(self.nops[j]):
                strop = self.agg_ops_types[j][k].lower()
                if strop == 'mean':
                    self.data_cube = (cprobs*self.weightings).sum(axis=3) / self.weightings.sum()
                elif strop == 'min':
                    self.data_cube = cprobs.min(axis=3)
                elif strop == 'max':
                    self.data_cube = cprobs.min(axis=3)
                elif strop == 'std':
                    self.data_cube = cprobs.std(axis=3)
                else:
                    assert(False) # bad op

                self.dataset_out = self.types[j] + ('' if strop=='mean' else '_' + strop)
                if use_ovlp:
                    self.size = ovlp_size; self.offset = ovlp_offset; self.inith5()
                self.writeCube()
                if use_ovlp:
                    self.size = orig_size; self.offset = orig_offset; self.inith5()

        if self.dpAggProbs_verbose:
            print('\tdone in %.4f s' % (time.time() - t))

    @staticmethod
    def addArgs(p):
        # adds arguments required for this object to specified ArgumentParser object
        dpWriteh5.addArgs(p)

        p.add_argument('--inrawpath', nargs=1, type=str, default='', metavar='PATH', help='Raw inputs path')

        # pertaining to voxel types
        p.add_argument('--types', nargs='+', type=str, default=['ICS','ECS','MEM'],
            metavar='TYPE', help='Names of the voxel types (prefix for raw file)')
        p.add_argument('--agg-ops-types', nargs='*', type=str, default=[], metavar='OP',
            help='Specify which operations to be done for each type, 0 in list delimits types')

        # pertaining to probs to be merged
        p.add_argument('--nmerge', nargs=1, type=int, default=[1],
            metavar='N', help='Number of network probability output rawfiles to merge')
        p.add_argument('--weightings', nargs='*', type=float, default=[], metavar='W',
            help='Weightings for probabilities specified in each rawfile (un-reordered), default to equal weightings')
        p.add_argument('--dim-orderings', nargs='*', type=str, default=[], choices=('xyz','xzy','zyx'),
            metavar='ORD', help='Specify the reslice ordering of the rawfile inputs (default all xyz)')

        # added this feature to optimize overlap support
        p.add_argument('--overlap', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
            help='Select out portion of raw cube to support overlaps')

        p.add_argument('--dpAggProbs-verbose', action='store_true', help='Debugging output for dpAggProbs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(\
        description='Aggregate multiple network probability outputs in knossos-style raw format into a single hdf5',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpAggProbs.addArgs(parser)
    args = parser.parse_args()

    aggProbs = dpAggProbs(args)
    aggProbs.aggregate()
