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

# Script / command line tool for merging supervoxels into single labels that
#   were manually merged using the knossos standalone tool (annotation file).

#import numpy as np
import time
import argparse
import os
from io import StringIO

from dpCubeIter import dpCubeIter
from typesh5 import emLabels
from dpWriteh5 import dpWriteh5
from dpLoadh5 import dpLoadh5

class dpLabelMerger(emLabels):

    # Constants
    LIST_ARGS = dpLoadh5.LIST_ARGS + dpCubeIter.LIST_ARGS

    def __init__(self, args):
        emLabels.__init__(self,args)

        # save the command line argument dict as a string
        out = StringIO(); print( vars(args), file=out )
        self.arg_str = out.getvalue(); out.close()

        # xxx - meh, need to fix this
        if not self.data_type_out: self.data_type_out = self.data_type

        assert( len(self.fileprefixes) == 1 and len(self.filepaths) == 1 )   # prefix / path for h5 label inputs only

        # print out all initialized variables in verbose mode
        if self.dpLabelMerger_verbose: print('dpLabelMerger, verbose mode:\n'); print(vars(self))

    def doMerging(self):
        # xxx - ahhhhhh
        if self.chunksize is not None and (self.chunksize < 0).all(): self.chunksize = self.use_chunksize
        self.cubeIter = dpCubeIter.cubeIterGen(self.volume_range_beg,self.volume_range_end,self.overlap,self.cube_size,
                    left_remainder_size=self.left_remainder_size, right_remainder_size=self.right_remainder_size,
                    chunksize=self.chunksize, leave_edge=self.leave_edge)

        for cur_cube_info in self:
            cur_data, cur_attrs, cur_ncomps, n = cur_cube_info
            _, _, _, _, _, _, is_left_border, is_right_border, _ = self.volume_info
            
        #if self.dpLabelMerger_verbose:
        #    print('First pass, final ncomps = %d, total ncomps = %d' % (ncomps, total_ncomps))

    def __iter__(self):
        for self.volume_info,n in zip(self.cubeIter, range(self.cubeIter.volume_size)):
            _, self.size, self.chunk, self.offset, suffixes, _, _, _, _ = self.volume_info
            self.inith5()

            if self.dpLabelMerger_verbose:
                print('Loading chunk %d %d %d, size %d %d %d, offset %d %d %d' % tuple(self.chunk.tolist() + \
                    self.size.tolist() + self.offset.tolist())); t = time.time()

            srcfile = os.path.join(self.filepaths[0], self.fileprefixes[0] + suffixes[0] + '.h5') if self.first_pass \
                else self.srcfile
            loadh5 = emLabels.readLabels(srcfile=srcfile, chunk=self.chunk.tolist(), subgroups=self.subgroups,
                offset=self.offset.tolist(), size=self.size.tolist(), verbose=self.dpLoadh5_verbose)
            assert( (self.chunksize == loadh5.chunksize).all() )

            cur_data = loadh5.data_cube.astype(self.data_type_out)
            cur_ncomps = loadh5.data_attrs['types_nlabels'].sum()
            cur_attrs = loadh5.data_attrs; cur_attrs['datasize'] = loadh5.datasize

            # xxx - make these as option? need if each volume being read is a portion of a larger labeled volume
            #cur_data = emLabels.relabel_sequential(cur_data)
            #cur_ncomps = cur_data.max()

            if self.dpLabelMerger_verbose:
                print('\tdone in %.4f s, ncomps = %d' % (time.time() - t, cur_ncomps))

            yield cur_data, cur_attrs, cur_ncomps, n




    @staticmethod
    def addArgs(p):
        dpWriteh5.addArgs(p)
        dpCubeIter.addArgs(p)
        #p.add_argument('--concatenate_only', action='store_true', help='Just concatenate volumes, no stitching')
        #p.add_argument('--two_pass', action='store_true', help='Use two pass method')
        #p.add_argument('--two_pass_load', nargs=1, type=str, default='', help='Raw file to load first pass')
        #p.add_argument('--two_pass_save', nargs=1, type=str, default='', help='Raw file to export first pass')

        p.add_argument('--dpLabelMerger-verbose', action='store_true',
            help='Debugging output for dpLabelMerger')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Merge supervoxels from knossos annotation file over superchunked volume')
    dpLabelMerger.addArgs(parser)
    args = parser.parse_args()

    merger = dpLabelMerger(args)
    merger.doMerging()

