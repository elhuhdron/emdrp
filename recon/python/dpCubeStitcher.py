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
import networkx as nx

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
        self.cubeIter = dpCubeIter.cubeIterGen(self.volume_range_beg,self.volume_range_end,self.overlap,self.cube_size,
                    left_remainder_size=self.left_remainder_size, right_remainder_size=self.right_remainder_size,
                    chunksize=self.chunksize, leave_edge=self.leave_edge)

        if self.two_pass:
            assert( not self.concatenate_only ) # silly
            # implement a hacky save/load mostly for debugging purposes for two-pass
            if self.two_pass_load:
                connections = np.fromfile(self.two_pass_load,dtype=np.int64).reshape((-1,2))
            else:
                connections = self.stitch_first_pass(do_stitching=False)
            ncomps, total_ncomps = connections[-1,:]
            if self.two_pass_save:
                connections.tofile(self.two_pass_save)

            self.stitch_second_pass(connections[:-1,:], total_ncomps)
        else:
            self.stitch_first_pass(do_stitching=not self.concatenate_only)

    def __iter__(self):
        for self.volume_info,n in zip(self.cubeIter, range(self.cubeIter.volume_size)):
            _, self.size, self.chunk, self.offset, suffix, _, _, _ = self.volume_info
            self.inith5()

            if self.dpCubeStitcher_verbose:
                print('Loading chunk %d %d %d, size %d %d %d, offset %d %d %d' % tuple(self.chunk.tolist() + \
                    self.size.tolist() + self.offset.tolist())); t = time.time()

            srcfile = os.path.join(self.filepaths[0], self.fileprefixes[0] + suffix + '.h5') if self.first_pass \
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

            if self.dpCubeStitcher_verbose:
                print('\tdone in %.4f s, ncomps = %d' % (time.time() - t, cur_ncomps))

            yield cur_data, cur_attrs, cur_ncomps, n

    # the one pass stitch only merges each "next cube" supervoxel with the single largest overlapping previously written
    #   supervoxel, which prevents the need to run connected components. it also does not allow for a supervoxel to
    #   stitch together two supervoxels that come in on different cube faces, thus forcing splits in these cases.
    def stitch_first_pass(self, do_stitching=True):
        nfaces = 1 if do_stitching else 3; ncomps = 0; total_ncomps = 0;
        connections = [np.zeros((0,2),dtype=np.int64)]*(self.cubeIter.volume_size*nfaces + 1)

        self.first_pass = True
        for cur_cube_info in self:
            cur_data, cur_attrs, cur_ncomps, n = cur_cube_info
            _, _, _, _, _, _, is_left_border, is_right_border = self.volume_info
            total_ncomps += cur_ncomps

            if is_left_border.all():
                if self.dpCubeStitcher_verbose:
                    print('\tfirst volume, first pass'); t = time.time()

                # "left-most" volume is starting volume
                use_data_attrs = cur_attrs.copy()
                self.datasize = use_data_attrs['datasize']; del use_data_attrs['datasize']
            else:
                if self.dpCubeStitcher_verbose:
                    print('\tstitching first pass'); t = time.time()

                # read the same volume out of the stitched output to get overlapping areas
                self.readCubeToBuffers()
                prev_data = self.data_cube.astype(self.data_type_out)

                # create the overlap select
                asel_ovlp = np.zeros(self.size, dtype=np.int8);
                if not is_left_border[2]: asel_ovlp[:,:,:self.overlap[2]] = 3
                if not is_left_border[1]: asel_ovlp[:,:self.overlap[1],:] = 2
                if not is_left_border[0]: asel_ovlp[:self.overlap[0],:,:] = 1
                if not is_right_border[0]: asel_ovlp[-self.overlap[0]:,:,:] = 0
                if not is_right_border[1]: asel_ovlp[:,-self.overlap[1]:,:] = 0
                if not is_right_border[2]: asel_ovlp[:,:,-self.overlap[2]:] = 0

                # for the one-pass approach, each supervoxel can only be merged with a single other supervoxel.
                # for the two-pass approach, any other scheme can be used, currently do it as "face-wise".
                no_overlap = [False]*nfaces
                for face in range(nfaces):
                    nm = n*nfaces + face    # just hand unroll connections, just stacked at end so does not matter
                    sel_ovlp = (asel_ovlp > 0) if nfaces == 1 else (asel_ovlp == face+1)

                    # get the voxel-wise overlap between the new cube and previous cubes in the overlapping area
                    prev_lbls = prev_data[sel_ovlp]; cur_lbls = cur_data[sel_ovlp]

                    # do not count any background overlaps
                    sel_nz = np.logical_and(prev_lbls != 0, cur_lbls != 0)
                    prev_lbls = prev_lbls[sel_nz]; cur_lbls = cur_lbls[sel_nz]

                    # skip if there is no overlap
                    if prev_lbls.size == 0:
                        no_overlap[face] = True; continue

                    no_overlap[face] = False
                    cx = sparse.csr_matrix((np.ones(prev_lbls.size, dtype=np.int64),
                                            (cur_lbls.flatten(), prev_lbls.flatten())),
                                           shape=(cur_ncomps+1, ncomps+1))
                    max_ovlp = csr_csc_argmax(cx)
                    # background overlaps should have been removed
                    assert( not (max_ovlp==0).any() and max_ovlp[0]==-1)

                    if do_stitching:
                        # relabel any supervoxels that are to be merged based on the overlap.
                        # relabel any supervoxels that are not merged as new labels (starting at ncomps+1)
                        mapping = np.zeros((cur_ncomps+1,), dtype=self.data_type_out)
                        sel = (max_ovlp > 0); nsel = (max_ovlp < 0); nsel[0] = False
                        cur_ncomps = nsel.sum(dtype=np.int64);
                        mapping[sel] = max_ovlp[sel]; mapping[nsel] = np.arange(1,cur_ncomps+1) + ncomps
                        cur_data = mapping[cur_data]
                    else:
                        # if we are not remapping, this is for two pass. Keep a graph of components to be linked
                        sel = (max_ovlp > 0); # first element in sel is False, as asserted above (background)
                        connections[nm] = np.column_stack((np.arange(ncomps, ncomps+cur_ncomps+1)[sel],
                                                           max_ovlp[sel]))

                # update current data for write
                no_overlap = all(no_overlap);
                if no_overlap: print('\tNO overlap detected')
                if not do_stitching or no_overlap: cur_data[cur_data > 0] += ncomps
            ncomps += cur_ncomps

            # remove the left offset for the write, saves time since offsets cross chunking boundaries
            sel_novlp = np.ones(self.size, dtype=np.bool)
            if not is_left_border[0]:
                sel_novlp[:self.overlap[0],:,:] = 0; self.offset[0] += self.overlap[0]; self.size[0] -= self.overlap[0]
            if not is_left_border[1]:
                sel_novlp[:,:self.overlap[1],:] = 0; self.offset[1] += self.overlap[1]; self.size[1] -= self.overlap[1]
            if not is_left_border[2]:
                sel_novlp[:,:,:self.overlap[2]] = 0; self.offset[2] += self.overlap[2]; self.size[2] -= self.overlap[2]
            self.inith5(); self.data_cube = cur_data[sel_novlp].reshape(self.size)

            self.data_attrs = use_data_attrs
            self.data_attrs['types_nlabels'] = [ncomps]
            self.data_attrs['no_overlap_nlabels'] = [total_ncomps]
            self.writeCube()
            if self.dpCubeStitcher_verbose:
                print('\tdone in %.4f s, ncomps = %d, total = %d' % (time.time() - t, ncomps, total_ncomps))

        if self.dpCubeStitcher_verbose:
            print('First pass, final ncomps = %d, total ncomps = %d' % (ncomps, total_ncomps))

        # append ncomps to connections, do this for hacky serialization instead of just returning it
        connections[-1] = np.array([ncomps, total_ncomps],dtype=np.int64).reshape(1,2)
        return np.vstack(connections)

    def stitch_second_pass(self, connections, total_ncomps):
        # run graph connected components and create mapping from old supervoxels to stitched supervoxels
        G = nx.Graph(); G.add_edges_from(connections)
        compsG = nx.connected_components(G); ncomps = 0; mapping = np.zeros((total_ncomps+1,), dtype=np.int64)
        for nodes in compsG:
            # create mapping from current per-zslice labels to linked labels across zslices
            ncomps += 1; mapping[np.array(tuple(nodes),dtype=np.int64)] = ncomps

        # any supervoxels that were not stitched are reassigned to values after stitched ones
        nsel = (mapping == 0); nsel[0] = False # first slot is background, don't remap
        not_remapped = np.cumsum(nsel,dtype=np.int64) + ncomps
        mapping[nsel] = not_remapped[nsel]; ncomps = not_remapped[-1]; del not_remapped, nsel

        if self.dpCubeStitcher_verbose:
            print('Second pass, stitching results in %d comps down from %d total' % (ncomps, total_ncomps))

        # turn the overlap off for faster iteration over the volumes
        # xxx - need to fix this if we're trying to keep the right border overlaps (for another round of stitching)
        self.overlap = np.zeros((3,),dtype=np.int32)
        self.cubeIter = dpCubeIter.cubeIterGen(self.volume_range_beg,self.volume_range_end,self.overlap,self.cube_size,
                    left_remainder_size=self.left_remainder_size, right_remainder_size=self.right_remainder_size,
                    chunksize=self.chunksize, leave_edge=self.leave_edge)

        # second pass to reread supervoxels written on the first pass and write back with remapping
        self.first_pass = False
        for cur_cube_info in self:
            cur_data, cur_attrs, cur_ncomps, n = cur_cube_info
            _, _, _, _, _, _, is_left_border, is_right_border = self.volume_info

            if self.dpCubeStitcher_verbose:
                print('\tremapping supervoxels, second pass'); t = time.time()

            self.data_cube = mapping[cur_data]
            # do not move these outside loop because data_atrrs gets reset by inith5 call
            self.data_attrs['types_nlabels'] = [ncomps]
            self.data_attrs['no_overlap_nlabels'] = [total_ncomps]
            self.writeCube()
            if self.dpCubeStitcher_verbose:
                #print('\tdone in %.4f s, ncomps = %d, total = %d' % (time.time() - t, ncomps, total_ncomps))
                print('\tdone in %.4f s' % (time.time() - t, ))

        if self.dpCubeStitcher_verbose:
            print('Second pass, stitching results in %d comps down from %d total' % (ncomps, total_ncomps))

    @staticmethod
    def addArgs(p):
        dpWriteh5.addArgs(p)
        dpCubeIter.addArgs(p)
        p.add_argument('--concatenate_only', action='store_true', help='Just concatenate volumes, no stitching')
        p.add_argument('--two_pass', action='store_true', help='Use two pass method')
        p.add_argument('--two_pass_load', nargs=1, type=str, default='', help='Raw file to load first pass')
        p.add_argument('--two_pass_save', nargs=1, type=str, default='', help='Raw file to export first pass')

        p.add_argument('--dpCubeStitcher-verbose', action='store_true',
            help='Debugging output for dpCubeStitcher')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Overlap-based adjacent volume "stitcher" (supervoxel merger)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpCubeStitcher.addArgs(parser)
    args = parser.parse_args()

    stitcher = dpCubeStitcher(args)
    stitcher.stitch()

