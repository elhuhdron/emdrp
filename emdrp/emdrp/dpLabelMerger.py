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
import zipfile
import glob
import numpy as np
from scipy import ndimage as nd
import scipy.ndimage.filters as filters

from dpCubeIter import dpCubeIter
from utils.typesh5 import emLabels
from dpWriteh5 import dpWriteh5
from dpLoadh5 import dpLoadh5

class dpLabelMerger(emLabels):

    # Constants
    LIST_ARGS = dpLoadh5.LIST_ARGS + dpCubeIter.LIST_ARGS + ['segmentation_values']

    # type to use for all processing operations
    PDTYPE = np.double

    def __init__(self, args):
        emLabels.__init__(self,args)

        # save the command line argument dict as a string
        out = StringIO(); print( vars(args), file=out )
        self.arg_str = out.getvalue(); out.close()

        # xxx - meh, need to fix this
        if not self.data_type_out: self.data_type_out = self.data_type

        assert( len(self.fileprefixes) == 1 and len(self.filepaths) == 1 )   # prefix / path for h5 label inputs only

        self.segmentation_levels = len(self.segmentation_values)

        # print out all initialized variables in verbose mode
        if self.dpLabelMerger_verbose: print('dpLabelMerger, verbose mode:\n'); print(vars(self))

        # copied this out of dpResample.py, but all dims resampled
        self.nresample_dims = 3
        self.nslices = self.dsfactor**self.nresample_dims
        self.slices = [None]*self.nslices; f = self.dsfactor; ff = f*f
        for i in range(f):
            for j in range(f):
                for k in range(f):
                    self.slices[i*ff + j*f + k] = np.s_[i::f,j::f,k::f]

        assert(self.contour_lvl >= 0 and self.contour_lvl < 1) # bad choice 


    def doMerging(self):

        volume_init = False
        cur_volume = np.zeros((4,), dtype=np.uint32)
        W = np.ones(self.smooth, dtype=self.PDTYPE) / self.smooth.prod() # smoothing kernel
        for s in range(self.segmentation_levels):
            self.cubeIter = dpCubeIter.cubeIterGen(self.volume_range_beg,self.volume_range_end,self.overlap,
                    self.cube_size, chunksize=self.chunksize, left_remainder_size=self.left_remainder_size,
                    right_remainder_size=self.right_remainder_size, leave_edge=self.leave_edge)
            self.subgroups[-1] = self.segmentation_values[s]
            cur_volume[3] = s

            for self.volume_info,n in zip(self.cubeIter, range(self.cubeIter.volume_size)):
                cur_volume[:3], self.size, self.chunk, self.offset, suffixes, _, _, _, _ = self.volume_info
                self.srcfile = os.path.join(self.filepaths[0], self.fileprefixes[0] + suffixes[0] + '.h5')
                self.inith5()

                # only load superchunks that contain some object supervoxels
                ind = np.ravel_multi_index(cur_volume, self.volume_step_seg)
                if len(self.sc_to_objs[ind]) < 1: continue

                if self.dpLabelMerger_verbose:
                    print('Merge in chunk %d %d %d, seglevel %d' % tuple(self.chunk.tolist() + [s])); t = time.time()
                self.readCubeToBuffers()
                cube = self.data_cube; cur_ncomps = self.data_attrs['types_nlabels'].sum()

                # xxx - writing to an hdf5 file in chunks or as a single volume from memory does not necessarily
                #   need to be tied to dsfactor==1, can add another command-line option for this.
                if not volume_init:
                    if self.dsfactor > 1:
                        volume_init=True; f = self.dsfactor
                        new_attrs = self.data_attrs
                        # changed this to be added when raw hdf5 is created
                        if 'factor' not in new_attrs:
                            new_attrs['factor'] = np.ones((dpLoadh5.ND,),dtype=np.double)
                        new_datasize = self.datasize.copy()
                        if 'boundary' in new_attrs: # proxy for whether attrs is there at all
                            # update the scale and compute new chunk/size/offset
                            new_attrs['scale'] *= f; new_attrs['boundary'] //= f
                            new_attrs['nchunks'] = np.ceil(new_attrs['nchunks'] / f).astype(np.int32)
                        # this attribute is saved as downsample factor
                        new_attrs['factor'] *= f; new_datasize //= f

                        new_data = np.zeros(new_datasize, dtype=self.data_type_out)
                    else:
                        # initialize by just writing a small chunk of zeros
                        self.inith5()
                        self.data_attrs['types_nlabels'] = [self.nobjects]
                        self.fillvalue = 0 # non-zero fill value not useful for merged "neurons"
                        # xxx - this probably should be cleaned up, see comments in dpWriteh5.py
                        orig_dataset = self.dataset; orig_subgroups = self.subgroups; orig_offset = self.offset
                        self.writeCube(data=np.zeros((32,32,32), dtype=self.data_type_out))
                        # reopen the dataset and write to it dynamically below
                        dset, group, h5file = self.createh5(self.outfile)
                        # xxx - this probably should be cleaned up, see comments in dpWriteh5.py
                        self.dataset = orig_dataset; self.subgroups = orig_subgroups; self.offset = orig_offset

                # much of this code copied from the label mesher, extract supervoxel and smooth
                # Pad data with zeros so that meshes are closed on the edges
                sizes = np.array(cube.shape); r = self.smooth.max() + 1; sz = sizes + 2*r;
                dataPad = np.zeros(sz, dtype=self.data_type); dataPad[r:sz[0]-r, r:sz[1]-r, r:sz[2]-r] = cube

                # get bounding boxes for all supervoxels in this volume
                svox_bnd = nd.measurements.find_objects(dataPad, cur_ncomps)

                for cobj in self.sc_to_objs[ind]:
                    #self.mergelists[cobj] = {'ids':allids[:,0], 'scids':allids[:,1:5], 'inds':inds}
                    cinds = np.nonzero(ind == self.mergelists[cobj]['inds'])[0]
                    for j in cinds:
                        cid = self.mergelists[cobj]['ids'][j]
                        cur_bnd = svox_bnd[cid-1]
                        imin = np.array([x.start for x in cur_bnd]); imax = np.array([x.stop-1 for x in cur_bnd])

                        # min and max coordinates of this seed within zero padded cube
                        pmin = imin - r; pmax = imax + r;
                        # min coordinates of this seed relative to original (non-padded cube)
                        mins = pmin - r; rngs = pmax - pmin + 1

                        crpdpls = (dataPad[pmin[0]:pmax[0]+1,pmin[1]:pmax[1]+1,
                                           pmin[2]:pmax[2]+1] == cid).astype(self.PDTYPE)

                        if W.size==0 or (W==1).all():
                            crpdplsSm = crpdpls
                        else:
                            crpdplsSm = filters.convolve(crpdpls, W, mode='reflect', cval=0.0, origin=0)
                        # if smoothing results in nothing above contour level, use original without smoothing
                        if (crpdplsSm > self.contour_lvl).any():
                            del crpdpls; crpdpls = crpdplsSm
                        del crpdplsSm
                        # save bounds relative to entire dataset
                        bounds_beg = mins + self.dataset_index
                        #bounds_end = mins + rngs - 1 + self.dataset_index;
                        bounds_end = mins + rngs + self.dataset_index; # exclusive end, python-style

                        if self.dsfactor > 1:
                            # downsample the smoothed supervoxel and assign it in the new downsampled volume
                            b = bounds_beg.copy(); b //= f
                            # stupid integer arithmetic, need to add 1 if it's not a multiple of the ds factor
                            e = b + (bounds_end-bounds_beg)//f + ((bounds_end-bounds_beg)%f != 0)
                            new_data[b[0]:e[0],b[1]:e[1],b[2]:e[2]][crpdpls[self.slices[0]] > self.contour_lvl] = cobj
                        else:
                            # write non-downsampled directly to h5 output file
                            b = bounds_beg; e = b + (bounds_end-bounds_beg)
                            # this is hard-coded to write the dataset in F-order (normal convention).
                            tmp = np.transpose(dset[b[2]:e[2],b[1]:e[1],b[0]:e[0]], (2,1,0))
                            tmp[crpdpls > self.contour_lvl] = cobj
                            dset[b[2]:e[2],b[1]:e[1],b[0]:e[0]] = np.transpose(tmp, (2,1,0))

                del self.data_cube # xxx - have to reallocate since view changes remove C-order contiguous
                if self.dpLabelMerger_verbose:
                    print('\tdone in %.4f s' % (time.time() - t, ))

        if self.dsfactor > 1:
            self.size = new_datasize; self.chunk[:] = 0; self.offset[:] = 0
            if self.dpLabelMerger_verbose:
                print('Writing out full downsampled dataset'); t = time.time()
            self.inith5()
            self.data_cube = new_data
            self.data_attrs = new_attrs
            self.data_attrs['types_nlabels'] = [self.nobjects]
            self.datasize = new_datasize
            self.fillvalue = 0 # non-zero fill value not useful for merged "neurons"
            self.writeCube()
            if self.dpLabelMerger_verbose:
                print('\tdone in %.4f s' % (time.time() - t, ))
        else:
            h5file.close()


    # first pass over annotation files creates a mapping from superchunks to objects.
    # this allows second pass to only have to load each superchunk only once, instead of potentially having to reload
    #   for different objects (as in a single pass).
    def enumerateAnnotationFiles(self):
        # xxx - ahhhhhh
        if self.chunksize is not None and (self.chunksize < 0).all(): self.chunksize = self.use_chunksize
        self.cubeIter = dpCubeIter.cubeIterGen(self.volume_range_beg,self.volume_range_end,self.overlap,self.cube_size,
                    left_remainder_size=self.left_remainder_size, right_remainder_size=self.right_remainder_size,
                    chunksize=self.chunksize, leave_edge=self.leave_edge)

        if self.dpLabelMerger_verbose:
            print('First pass, loading annotation files, creating lookups'); t = time.time()

        # this is the the cubeIter step (number of cube sizes in the volume) with number of seg levels appended
        # xxx - number of segmentation levels could potentially be computed by globing the labels, didn't seem worth it
        self.volume_step_seg = np.zeros((4,),dtype=np.uint32)
        self.volume_step_seg[:3] = self.cubeIter.volume_step; self.volume_step_seg[3] = self.segmentation_levels
        self.volume_size_seg = np.prod(self.volume_step_seg)

        loadfiles = glob.glob(self.annotation_file_glob); nFiles = len(loadfiles);
        assert( nFiles > 0 ) # empty glob for knossos annotation files
        self.mergelists = {}; self.nobjects = 0; self.sc_to_objs = [set() for x in range(self.volume_size_seg)];
        for j in range(nFiles):
            # get the mergelist out of the zipped knossos annotation file
            zf = zipfile.ZipFile(loadfiles[j], mode='r');
            inmerge = zf.read('mergelist.txt'); #inskel = zf.read('annotation.xml');
            zf.close()
            # read the merge list
            merge_list = inmerge.decode("utf-8").split('\n'); nlines = len(merge_list); n = nlines // 4

            for i in range(n):
                # Object_ID, ToDo_flag, Immutability_flag, [Supervoxel_ID, SCx, SCy, SCz, SClevel]* '\n'
                curmergeline = merge_list[i*4].split(' '); #cobj = int(curmergeline[0]) + self.nobjects
                # object ID in annotation list can be missing ids and out of order, so just use the order in the file
                cobj = i + self.nobjects + 1
                allids = np.array(curmergeline[3::],dtype=np.uint32).reshape((-1,5))
                scids = allids[:,1:5].copy()
                scids[:,:3] = (scids[:,:3]-self.cubeIter.volume_range_beg)//self.cubeIter.cube_size
                inds = np.ravel_multi_index(scids.T, self.volume_step_seg); uinds = np.unique(inds)
                # add this object to the mapping for all superchunks / seg levels that it contains in its mergelist
                for k in uinds:
                    self.sc_to_objs[k].add(cobj)
                # keep a hash of all the objects and their constituent supervoxels ids / superchunks / seg levels
                self.mergelists[cobj] = {'ids':allids[:,0], 'scids':allids[:,1:5], 'inds':inds}
            self.nobjects = len(self.mergelists.keys())

        if self.dpLabelMerger_verbose:
            print('\tdone in %.4f s, total objects to merge = %d' % (time.time() - t, self.nobjects))

    @staticmethod
    def addArgs(p):
        dpWriteh5.addArgs(p)
        dpCubeIter.addArgs(p)
        p.add_argument('--annotation-file-glob', nargs=1, type=str, default='',
                       help='Glob for a list of input annotation files from knossos')
        p.add_argument('--labels-path', nargs=1, type=str, default='',
                       help='Input path for superchunked supervoxel label files (to merge)')
        p.add_argument('--dsfactor', nargs=1, type=int, default=[1], metavar=('F'),
                       help='Downsample factor, mode to write out a single label file')
        #p.add_argument('--segmentation-levels', nargs=1, type=int, default=[4], metavar=('NLVLS'),
        #               help='Number of available segmentation levels')
        p.add_argument('--segmentation-values', nargs='+', type=str, default=[],
            help='Mapping from segmentation levels to parameter value in hdf5')
        p.add_argument('--smooth', nargs=3, type=int, default=[7,7,7], metavar=('X', 'Y', 'Z'),
            help='Size of smoothing kernel (zeros for none)')
        p.add_argument('--contour-lvl', nargs=1, type=float, default=[0.2], metavar=('LVL'),
            help='Level [0,1] to use to binarize after smoothing applied')

        p.add_argument('--dpLabelMerger-verbose', action='store_true',
            help='Debugging output for dpLabelMerger')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Merge supervoxels from knossos annotation file over superchunked volume')
    dpLabelMerger.addArgs(parser)
    args = parser.parse_args()

    merger = dpLabelMerger(args)
    merger.enumerateAnnotationFiles()
    merger.doMerging()
