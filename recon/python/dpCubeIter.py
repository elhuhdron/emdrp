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

# Generator class for creating chunk/size/offset/name information for hdf5 files
#   containing blocks of supervoxels that overlap at the edges between blocks.
# This is the basis for "stitching" together blocks using an overlap method.

import argparse
import os
import numpy as np

class dpCubeIter(object):

    LIST_ARGS = ['fileflags', 'filepaths', 'fileprefixes', 'filepostfixes', 'filemodulators']

    #def __init__(self, inprefix, volume_range_beg, volume_range_end, overlap,
    #             cube_size=[1,1,1], left_remainder_size=[0,0,0], right_remainder_size=[0,0,0],
    #             chunksize=[128,128,128], leave_edge=False):
    #    # str - prefix for the name of the file
    #    self.inprefix = inprefix
    #    # (3,) int - beginning and end of ranges in chunks specified python-style
    #    self.volume_range_beg = np.array(volume_range_beg, dtype=np.int64)
    #    self.volume_range_end = np.array(volume_range_end, dtype=np.int64)
    #    # (3,) int - how much overlap in each direction in voxels
    #    self.overlap = np.array(overlap, dtype=np.int64)
    #    # (3,) int - size of each cube being stitched in chunks
    #    self.cube_size = np.array(cube_size, dtype=np.int64)
    #    # (3,) int - size of remainder edges on "left" and "right" sides for unaligned stitching in voxels
    #    self.left_remainder_size = np.array(left_remainder_size, dtype=np.int64)
    #    self.right_remainder_size = np.array(right_remainder_size, dtype=np.int64)
    #    # (3,) int - chunksize in voxels
    #    self.chunksize = np.array(chunksize, dtype=np.int64)
    #    # bool - whether to leave the overlap on the right edges
    #    self.leave_edge = bool(leave_edge)
    def __init__(self, args):
        # save command line arguments from argparse, see definitions in main or run with --help
        for k, v in vars(args).items():
            # do not override any values that are already set as a method of allowing inherited classes to specify
            if hasattr(self,k): continue
            if type(v) is list and k not in self.LIST_ARGS:
                if len(v)==1:
                    setattr(self,k,v[0])  # save single element lists as first element
                elif type(v[0]) is int:   # convert the sizes and offsets to numpy arrays
                    setattr(self,k,np.array(v,dtype=np.int32))
                else:
                    setattr(self,k,v)   # store other list types as usual (floats)
            else:
                setattr(self,k,v)

        # other inits
        self.chunksize = self.use_chunksize
        self.cube_size_voxels = self.cube_size * self.chunksize
        self.left_remainder = self.left_remainder_size > 0; self.right_remainder = self.right_remainder_size > 0

        self.volume_range = self.volume_range_end - self.volume_range_beg
        assert( (self.volume_range % self.cube_size == 0).all() )
        self.volume_step = self.volume_range // self.cube_size

        self.volume_step += self.left_remainder; self.volume_step += self.right_remainder
        self.volume_size = np.prod(self.volume_step)

        # modulators default to all ones
        self.nflags = len(self.fileflags)
        # this is for the python interface mode (does not use the argument flag / file name creation stuff)
        if self.nflags == 0: self.nflags = 1
        if len(self.filemodulators) == 0:
            self.filemodulators = np.ones((self.nflags,3),dtype=np.uint32)
        else:
            self.filemodulators = np.array(self.filemodulators,dtype=np.uint32).reshape((-1,3))
            assert(self.filemodulators.shape[0] == self.nflags)
            assert( (self.cube_size % self.filemodulators == 0).all() )

    def __iter__(self):
        for cur_index in range(self.volume_size):
            # the current volume indices, including the right and left remainders
            cur_volume = np.array(np.unravel_index(cur_index, self.volume_step), dtype=np.int64)

            # need special cases to handle the remainders
            is_left_border = cur_volume == 0; is_right_border = cur_volume == (self.volume_step-1)
            #is_not_left_border = np.logical_not(is_left_border); is_not_right_border = np.logical_not(is_right_border)
            is_left_remainder = np.logical_and(is_left_border,self.left_remainder)
            is_right_remainder = np.logical_and(is_right_border,self.right_remainder)
            is_not_left_remainder = np.logical_not(is_left_remainder)
            #is_not_right_remainder = np.logical_not(is_right_remainder)
            assert( not (np.logical_and(is_left_remainder, is_right_remainder)).any() ) # bad use case

            # left and right remainders are offset from the start of the previous and last chunks respectfully
            cur_volume[is_not_left_remainder] -= self.left_remainder[is_not_left_remainder]
            cur_chunk = cur_volume * self.cube_size + self.volume_range_beg
            cur_chunk[is_left_remainder] -= self.cube_size[is_left_remainder]

            left_offset = self.overlap.copy(); left_offset[is_left_border] = 0
            right_offset = self.overlap.copy();
            if not self.leave_edge: right_offset[is_right_border] = 0

            # default size is adding left and right offsets
            size = self.cube_size_voxels + left_offset + right_offset

            # special cases for remainder blocks
            size[is_left_remainder] = self.left_remainder_size[is_left_remainder] + right_offset[is_left_remainder]
            size[is_right_remainder] = self.right_remainder_size[is_right_remainder] + left_offset[is_right_remainder]
            left_offset = -left_offset # default left offset is set negative as returned offset
            # left offset for left remainder block is from the left side of previous cube
            left_offset[is_left_remainder] = \
                self.cube_size_voxels[is_left_remainder] - self.left_remainder_size[is_left_remainder]

            # modified to allow for "modulators" which allows for chunk descriptors that only change at multiples of
            #   cube_size. allows for cubeiter to create command lines containing arguments with different cube_sizes
            suffixes = [None] * self.nflags; affixes = [None] * self.nflags
            for j in range(self.nflags):
                fm = self.filemodulators[j,:]
                if (fm==1).all():
                    mcur_chunk = cur_chunk
                else:
                    mcur_chunk = (cur_volume // fm)*fm * self.cube_size + self.volume_range_beg

                # create the name suffixes, path affixes
                suffixes[j] = ''; affixes[j] = ''
                for s,i in zip(['x','y','z'], range(3)):
                    r = 'l' if is_left_remainder[i] else ('r' if is_right_remainder[i] else '')
                    suffixes[j] += ('_%s%04d' % (s + r, mcur_chunk[i]))
                    affixes[j] = os.path.join(affixes[j], ('%s%04d' % (s, mcur_chunk[i])))
                affixes[j] += os.path.sep

            yield cur_volume, size, cur_chunk, left_offset, suffixes, affixes, is_left_border, is_right_border

    def flagsToString(self, flags, paths, prefixes, postfixes, suffixes, affixes):
        argstr = ' '
        for flag, path, prefix, postfix, suffix, affix in zip(flags, paths, prefixes, postfixes, suffixes, affixes):
            if flag != '0':
                argstr += '--' + flag + ' '
            # xxx - better names?
            # affix is the optional knossos-style path (i.e., x0001/y0002/z0005)
            # prefix is the specified file name without an extension or path
            # suffix is the optional knossos-style addition to the filename (i.e., _x0001_y0002_z0005)
            # postfix is the file extension
            name = affix + prefix + suffix + postfix
            if path != '0':
                name = os.path.join(path,name)
            argstr += name + ' '
        return argstr

    def printCmds(self):
        if self.cmdfile:
            with open(self.cmdfile, 'r') as myfile:
                cmd = myfile.read().split('\n'); cmd = [x for x in cmd if x]
        else:
            cmd = [self.cmd]
        ncmd = len(cmd)

        cnt = 0; empty_strs = ['' for i in range(self.nflags)]
        for volume_info in self:
            _, size, cur_chunk, left_offset, suffixes, affixes, is_left_border, is_right_border = volume_info
            ccmd = cmd[0] if ncmd == 1 else cmd[cnt]

            str_volume = (' --size %d %d %d ' % tuple(size.tolist())) + \
                (' --chunk %d %d %d ' % tuple(cur_chunk.tolist())) + \
                (' --offset %d %d %d ' % tuple(left_offset.tolist()))
            str_inputs = self.flagsToString(self.fileflags, self.filepaths, self.fileprefixes, self.filepostfixes,
                                            suffixes if self.use_suffix else empty_strs, 
                                            affixes if self.affix_path else empty_strs)
            print(ccmd + (''if self.no_volume_flags else str_volume) + str_inputs)

            cnt += 1

    @classmethod
    def cubeIterGen(cls, volume_range_beg, volume_range_end, overlap, cube_size,
                    left_remainder_size=None, right_remainder_size=None, chunksize=None, leave_edge=None):
        parser = argparse.ArgumentParser(description='cubeIterGen:dpCubeIter',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        dpCubeIter.addArgs(parser); arg_str = ''
        arg_str += ' --volume_range_beg %d %d %d ' % tuple(volume_range_beg)
        arg_str += ' --volume_range_end %d %d %d ' % tuple(volume_range_end)
        arg_str += ' --overlap %d %d %d ' % tuple(overlap)
        arg_str += ' --cube_size %d %d %d ' % tuple(cube_size)
        if left_remainder_size is not None: arg_str += ' --left_remainder_size %d %d %d ' % tuple(left_remainder_size)
        if right_remainder_size is not None: arg_str += '--right_remainder_size %d %d %d ' % tuple(right_remainder_size)
        if chunksize is not None: arg_str += ' --use-chunksize %d %d %d ' % tuple(chunksize)
        if leave_edge: arg_str += ' --leave_edge '
        args = parser.parse_args(arg_str.split())
        return cls(args)

    @staticmethod
    def addArgs(p):
        # adds arguments required for this object to specified ArgumentParser object
        p.add_argument('--cmdfile', nargs=1, type=str, default='',
                       help='Full name and path of text file containing command')
        p.add_argument('--cmd', nargs=1, type=str, default='', help='Specify command on command line as string')
        p.add_argument('--fileflags', nargs='*', type=str, default=[],
                       help='in/out files command line switches (0 for none)')
        p.add_argument('--filepaths', nargs='*', type=str, default=[], help='in/out files paths (0 for none)')
        p.add_argument('--fileprefixes', nargs='*', type=str, default=[], help='in/out files filename prefixes')
        p.add_argument('--filepostfixes', nargs='*', type=str, default=[], help='in/out files filename postfixes')
        p.add_argument('--filemodulators', nargs='*', type=int, default=[],
                       help='Allows for supervolumes at multiples of cube_size (x0 y0 z0  x1 y1 z1 ...)')
        p.add_argument('--volume_range_beg', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
            help='Starting range in chunks for total volume')
        p.add_argument('--volume_range_end', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
            help='Ending range in chunks for total volume (python style)')
        p.add_argument('--overlap', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
            help='Amount of overlap in each direction')
        p.add_argument('--cube_size', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
            help='Size in chunks of iterate volume (superchunk)')
        p.add_argument('--left_remainder_size', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
            help='Size in voxels of "left" remainder volumes')
        p.add_argument('--right_remainder_size', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
            help='Size in voxels of "right" remainder volumes')
        p.add_argument('--use-chunksize', nargs=3, type=int, default=[128,128,128], metavar=('X', 'Y', 'Z'),
                       help='Size of chunks in voxels')
        p.add_argument('--leave_edge', action='store_true', help='Whether to leave right-most overlap or not')
        p.add_argument('--affix_path', action='store_true',
                       help='Whether to add chunk suffix as path prefix (knossos-style)')
        p.add_argument('--no-suffix', action='store_false', dest='use_suffix',
                       help='Do not append knossos suffix to file name (iterate single file)')
        p.add_argument('--no_volume_flags', action='store_true',
                       help='Do not include chunk, size and offset flags in output')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate command lines for parallelized cube processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpCubeIter.addArgs(parser)
    args = parser.parse_args()

    ci = dpCubeIter(args)
    ci.printCmds()
