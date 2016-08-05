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
#import time
import argparse
#import os
#import sys
from io import StringIO

from typesh5 import emLabels, emProbabilities

class dpCubeStitcher(emLabels):

    # Constants
    LIST_ARGS = ['test_chunks']

    def __init__(self, args):
        emLabels.__init__(self,args)

        # save the command line argument dict as a string
        out = StringIO(); print( vars(args), file=out )
        self.arg_str = out.getvalue(); out.close()

        if not self.data_type_out: self.data_type_out = self.data_type




        # print out all initialized variables in verbose mode
        #if self.dpFRAG_verbose: print('dpFRAG, verbose mode:\n'); print(vars(self))

    @staticmethod
    def addArgs(p):
        #p.add_argument('--cfgfile', nargs=1, type=str, default='', help='Path/name of ini config file')

        p.add_argument('--dpCubeStitcher-verbose', action='store_true',
            help='Debugging output for dpCubeStitcher')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Overlap-based adjacent volume "stitcher" (supervoxel merger)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpCubeStitcher.addArgs(parser)
    args = parser.parse_args()

    stitcher = dpCubeStitcher(args)

