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

# Extends emLabels class for performing "label cleaning".

import h5py
import numpy as np
import argparse
import time
import os

from scipy import ndimage as nd

from dpLoadh5 import dpLoadh5
from dpWriteh5 import dpWriteh5
from typesh5 import emLabels, emProbabilities, emVoxelType

class dpCleanLabels(emLabels):

    def __init__(self, args):
        emLabels.__init__(self,args)

        self.bwconn = nd.morphology.generate_binary_structure(dpLoadh5.ND, self.connectivity)

    def clean(self):
        selbg = (self.data_cube == 0)   # used in more than one of the cleaning steps
            
        if self.minsize > 0:
            if self.dpCleanLabels_verbose:
                print('Scrubbing labels with minsize %d' % (self.minsize,)); t = time.time()
                
            self.data_cube, sizes = emLabels.thresholdSizes(self.data_cube, minSize=self.minsize)
            self.data_cube = emLabels.nearest_neighbor_fill(self.data_cube, mask=selbg, 
                sampling=self.data_attrs['scale'])

            if self.dpCleanLabels_verbose:
                print('\tdone in %.4f s' % (time.time() - t))

        if self.cavity_fill:
            if self.dpCleanLabels_verbose:
                print('Removing cavities using conn %d' % (self.connectivity,)); t = time.time()
                
            labels = np.ones([x + 2 for x in self.data_cube.shape], dtype=np.bool)
            labels[1:-1,1:-1,1:-1] = selbg
            # don't connect the top and bottom xy planes
            labels[1:-1,1:-1,0] = 0; labels[1:-1,1:-1,-1] = 0
            labels, nlabels = nd.measurements.label(labels, self.bwconn)
            msk = np.logical_and((labels[1:-1,1:-1,1:-1] != labels[0,0,0]), selbg); del labels
            self.data_cube[msk] = 0; selbg[msk] = 0
            self.data_cube = emLabels.nearest_neighbor_fill(self.data_cube, mask=selbg, 
                sampling=self.data_attrs['scale'])

            if self.dpCleanLabels_verbose:
                print('\tdone in %.4f s' % (time.time() - t))

        # this step is always last, as writes new voxel_type depending on the cleaning that was done
        if self.write_voxel_type:
            if self.dpCleanLabels_verbose:
                print('Rewriting voxel type pixel data'); t = time.time()
                
            voxType = emVoxelType.readVoxType(srcfile=self.srcfile, chunk=self.chunk.tolist(), 
                offset=self.offset.tolist(), size=self.size.tolist())
            voxel_type = voxType.data_cube.copy(order='C')
        
            labels = self.data_cube.copy(order='C')
            nlabels = labels.max(); ntypes = len(voxType.data_attrs['types'])
            supervoxel_type, voxel_type = emLabels.type_components(labels, voxel_type, nlabels, ntypes)
            # xxx - reorder labels so that types are grouped... decided no need for now
            types_nlabels = [(supervoxel_type == x).sum(dtype=np.int64) for x in range(1,ntypes)]
            assert( sum(types_nlabels) == nlabels )
            self.data_attrs['types_nlabels'] = [nlabels]
            self.data_attrs['unordered_types_nlabels'] = types_nlabels

            d = voxType.data_attrs.copy(); #d['types_nlabels'] = 
            emVoxelType.writeVoxType(outfile=self.outfile, chunk=self.chunk.tolist(), 
                offset=self.offset.tolist(), size=self.size.tolist(), datasize=voxType.datasize.tolist(), 
                chunksize=voxType.chunksize.tolist(), data=voxel_type.astype(emVoxelType.VOXTYPE_DTYPE), attrs=d)

            if self.dpCleanLabels_verbose:
                print('\tdone in %.4f s' % (time.time() - t))

    @staticmethod
    def addArgs(p):
        # adds arguments required for this object to specified ArgumentParser object
        dpWriteh5.addArgs(p)
        p.add_argument('--minsize', nargs=1, type=int, default=[-1], metavar=('size'),
            help='Minimum label size in voxels to keep')
        p.add_argument('--cavity-fill', action='store_true', help='Remove all BG not connected to cube faces')
        p.add_argument('--connectivity', nargs=1, type=int, default=[1], choices=[1,2,3],
            help='Connectivity (where applicable)')
        p.add_argument('--write-voxel-type', action='store_true', help='Recompute voxel type using cleaned labels')
        p.add_argument('--dpCleanLabels-verbose', action='store_true', help='Debugging output for dpCleanLabels')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write labels hdf5 file after some manipulations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpCleanLabels.addArgs(parser)
    args = parser.parse_args()
    
    cleanLbls = dpCleanLabels(args)
    cleanLbls.readCubeToBuffers()
    cleanLbls.clean()
    cleanLbls.writeCube()
    
