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

# Script / command line tool for stiching together volumes that were watershedded separately
#   based on overlapping regions of neighboring volumes.

import numpy as np
import scipy.sparse as sparse
import time
import argparse
import os
#import sys
#import itertools
from io import StringIO

from dpCubeIter import dpCubeIter
from typesh5 import emLabels
from dpWriteh5 import dpWriteh5
from dpLoadh5 import dpLoadh5
from utils import csr_csc_argmax

class dpCubeStitcher(emLabels):

    # Constants
    LIST_ARGS = dpLoadh5.LIST_ARGS + ['fileflags', 'filepaths', 'fileprefixes', 'volume_range_beg', 'volume_range_end',
                                      'overlap', 'left_remainder_size', 'right_remainder_size', 'use_chunksize',
                                      'leave_edge']

    def __init__(self, args):
        emLabels.__init__(self,args)

        # save the command line argument dict as a string
        out = StringIO(); print( vars(args), file=out )
        self.arg_str = out.getvalue(); out.close()

        # xxx - meh, need to fix this
        if not self.data_type_out: self.data_type_out = self.data_type

        assert( len(self.fileprefixes) == 1 and len(self.filepaths) == 1 )   # prefix / path for h5 label inputs only

        # print out all initialized variables in verbose mode
        if self.dpCubeStitcher_verbose: print('dpCubeStitcher, verbose mode:\n'); print(vars(self))

    def stitch(self):
        ncomps = 0
        cubeIter = dpCubeIter.cubeIterGen(self.volume_range_beg, self.volume_range_end, self.overlap, self.cube_size,
                    left_remainder_size=self.left_remainder_size, right_remainder_size=self.right_remainder_size,
                    chunksize=self.chunksize, leave_edge=self.leave_edge)
        total_ncomps = 0
        for volume_info in cubeIter:
            self.size, self.chunk, self.offset, suffix, is_left_border, is_right_border = volume_info
            self.inith5()

            if self.dpCubeStitcher_verbose:
                print('Stitching chunk %d %d %d, size %d %d %d, offset %d %d %d' % tuple(self.chunk.tolist() + \
                    self.size.tolist() + self.offset.tolist())); t = time.time()

            srcfile = os.path.join(self.filepaths[0], self.fileprefixes[0] + suffix + '.h5')
            loadh5 = emLabels.readLabels(srcfile=srcfile, chunk=self.chunk.tolist(), subgroups=self.subgroups,
                offset=self.offset.tolist(), size=self.size.tolist(), verbose=self.dpLoadh5_verbose)

            cur_data = loadh5.data_cube.astype(self.data_type_out)
            cur_ncomps = loadh5.data_attrs['types_nlabels'].sum()
            # xxx - make these as option? need if each volume being read is a portion of a larger labeled volume
            #cur_data = emLabels.relabel_sequential(cur_data)
            #cur_ncomps = cur_data.max()

            # keep track of how many comps we'd have without overlap merge
            total_ncomps += cur_ncomps

            if is_left_border.all():
                # "left-most" volume is starting volume
                self.data_attrs = loadh5.data_attrs; self.datasize = loadh5.datasize
                assert( (self.chunksize == loadh5.chunksize).all() )
                ncomps = cur_ncomps
                self.data_cube = cur_data
            else:
                # read the same volume out of the stitched output to get overlapping areas
                self.readCubeToBuffers()

                # create the overlap select
                sel_ovlp = np.zeros(self.size, dtype=np.bool)
                if not is_left_border[0]: sel_ovlp[:self.overlap[0],:,:] = 1
                if not is_left_border[1]: sel_ovlp[:,:self.overlap[1],:] = 1
                if not is_left_border[2]: sel_ovlp[:,:,:self.overlap[2]] = 1
                if not is_right_border[0]: sel_ovlp[-self.overlap[0]:,:,:] = 0
                if not is_right_border[1]: sel_ovlp[:,-self.overlap[1]:,:] = 0
                if not is_right_border[2]: sel_ovlp[:,:,-self.overlap[2]:] = 0

                # get the voxel-wise overlap between the new cube and previous cubes in the overlapping area
                prev_lbls = self.data_cube.astype(self.data_type_out)[sel_ovlp]; cur_lbls = cur_data[sel_ovlp]

                # do not count any background overlaps
                sel_nz = np.logical_and(prev_lbls != 0, cur_lbls != 0)
                prev_lbls = prev_lbls[sel_nz]; cur_lbls = cur_lbls[sel_nz]

                if prev_lbls.size == 0:
                    print('\tNO overlap detected' % (time.time() - t, ncomps))
                    cur_data[cur_data > 0] += ncomps
                else:
                    cx = sparse.csr_matrix((np.ones(prev_lbls.size, dtype=np.int64),
                                            (cur_lbls.flatten(), prev_lbls.flatten())),
                                           shape=(cur_ncomps+1, ncomps+1))
                    max_ovlp = csr_csc_argmax(cx)
                    assert( not (max_ovlp==0).any() and max_ovlp[0]==-1) # background overlaps should have been removed

                    # relabel any supervoxels that are to be merged based on the overlap.
                    # relabel any supervoxels that are not merged as new labels (starting at ncomps+1)
                    mapping = np.zeros((cur_ncomps+1,), dtype=self.data_type_out)
                    sel = (max_ovlp > 0); nsel = (max_ovlp < 0); nsel[0] = False
                    cur_ncomps = nsel.sum(dtype=np.int64);
                    mapping[sel] = max_ovlp[sel]; mapping[nsel] = np.arange(1,cur_ncomps+1) + ncomps
                    cur_data = mapping[cur_data]

                # update current data and ncomps for write
                ncomps += cur_ncomps
                self.data_cube = cur_data

            if self.dpCubeStitcher_verbose:
                print('\tdone in %.4f s, ncomps = %d' % (time.time() - t, ncomps))

            self.data_attrs['types_nlabels'] = [ncomps]
            self.data_attrs['no_overlap_nlabels'] = [total_ncomps]
            self.writeCube()

        if self.dpCubeStitcher_verbose:
            print('Final ncomps = %d, total ncomps = %d' % (ncomps, total_ncomps))

    @staticmethod
    def addArgs(p):
        dpWriteh5.addArgs(p)
        dpCubeIter.addArgs(p)
        #p.add_argument('--cfgfile', nargs=1, type=str, default='', help='Path/name of ini config file')

        p.add_argument('--dpCubeStitcher-verbose', action='store_true',
            help='Debugging output for dpCubeStitcher')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Overlap-based adjacent volume "stitcher" (supervoxel merger)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpCubeStitcher.addArgs(parser)
    args = parser.parse_args()

    stitcher = dpCubeStitcher(args)
    stitcher.stitch()

