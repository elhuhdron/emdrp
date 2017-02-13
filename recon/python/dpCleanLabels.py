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

# see label_checker.sh for usage for cleaning single annotator GT labels from itksnap

#import h5py
import numpy as np
import argparse
import time
#import os

from scipy import ndimage as nd

from dpLoadh5 import dpLoadh5
from dpWriteh5 import dpWriteh5
from typesh5 import emLabels, emVoxelType

class dpCleanLabels(emLabels):

    def __init__(self, args):
        emLabels.__init__(self,args)

        self.fgbwconn = nd.morphology.generate_binary_structure(dpLoadh5.ND, self.fg_connectivity)
        self.bgbwconn = nd.morphology.generate_binary_structure(dpLoadh5.ND, self.bg_connectivity)

    def clean(self):

        # read voxel types first, allows for cavity_fill and get_svox_type to be both specified
        if self.get_svox_type or self.write_voxel_type:
            if self.dpCleanLabels_verbose:
                print('Reading supervoxel types'); t = time.time()

            voxType = emVoxelType.readVoxType(srcfile=self.srcfile, chunk=self.chunk.tolist(),
                offset=self.offset.tolist(), size=self.size.tolist())
            voxel_type = voxType.data_cube.copy(order='C')
            ntypes = len(voxType.data_attrs['types'])

            if self.dpCleanLabels_verbose:
                print('\tdone in %.4f s' % (time.time() - t))

        # minpath overlay creation is intended to improve proofreading speed by highlighting connected path
        if self.minpath > 0:
            import skfmm
            if self.minpath_skel:
                from pyCext import binary_warping

            # xxx - allow for multiple minpaths with different labels?
            selmin = (self.data_cube == self.minpath)
            pts, npts = nd.measurements.label(selmin, self.fgbwconn)

            if self.dpCleanLabels_verbose:
                print('Finding shortest paths for all pairwise combinations of %d points' % (npts,)); t = time.time()

            labels = self.data_cube
            sel_ECS, ECS_label = self.getECS(labels); labels[sel_ECS] = 0
            selbg = (labels == 0)
            assert( self.minpath != ECS_label ) # can't have minpath and ECS label defined the same

            # create paths for all pair-wise combinations of points
            paths = np.zeros(self.size, dtype=np.uint8)
            for i in range(npts):
                for j in range(i+1,npts):
                    s1 = (pts==i+1); m = np.ones(self.size, dtype=np.double); m[s1] = 0
                    d1 = skfmm.distance(np.ma.masked_array(nd.distance_transform_edt(m, return_indices=False,
                              return_distances=True, sampling=self.data_attrs['scale']), selbg))

                    s2 = (pts==j+1); m = np.ones(self.size, dtype=np.double); m[s2] = 0
                    d2 = skfmm.distance(np.ma.masked_array(nd.distance_transform_edt(m, return_indices=False,
                              return_distances=True, sampling=self.data_attrs['scale']), selbg))

                    # xxx - need something like imregionalmin in 3d, could not quickly find an easy solution
                    #   parameterize the min select percentage?
                    d = d1+d2; bwlabels = ((d.data < 1.01*d.min()) & ~d.mask); bwlabels[s1 | s2] = 1
                    if self.minpath_skel:
                        # optionally skeletonize keeping original minpath points as anchors
                        bwlabels, diff, simpleLUT = binary_warping(bwlabels.copy(order='C'),
                            np.zeros(self.size,dtype=np.bool), mask=(~selmin).copy(order='C'), borderval=False,
                            slow=True, connectivity=self.fg_connectivity)
                        # fill back out slightly so more easily viewed in itksnap
                        bwlabels, diff, simpleLUT = binary_warping(bwlabels.copy(order='C'),
                            np.ones(self.size,dtype=np.bool), borderval=False, slow=True, simpleLUT=simpleLUT,
                            connectivity=self.fg_connectivity, numiters=1)
                    paths[bwlabels] = 1
            self.data_cube = paths

            if self.dpCleanLabels_verbose:
                print('\tdone in %.4f s' % (time.time() - t))

        # smoothing operates on each label one at a time
        if self.smooth:
            if self.dpCleanLabels_verbose:
                print('Smoothing labels object by object'); t = time.time()

            # threshold sizes to remove empty labels
            self.data_cube, sizes = emLabels.thresholdSizes(self.data_cube, minSize=1)

            # xxx - fix old comments from matlab meshing code, fix this

            # xxx - local parameters, expose if find any need to change these
            rad = 5;                # amount to pad (need greater than one for method 3 because of smoothing
            contour_level = 0.5;   # binary threshold for calculating surface mesh
            smooth_size = [3, 3, 3];

            #emptyLabel = 65535; % should define this in attribs?
            sizes = np.array(self.data_cube.shape); sz = sizes + 2*rad;
            image_with_zeros = np.zeros(sz, dtype=self.data_cube.dtype); # create zeros 3 dimensional array
            image_with_zeros[rad:-rad,rad:-rad,rad:-rad] = self.data_cube  # embed label array into zeros array

            image_with_brd = np.lib.pad(self.data_cube,((rad,rad), (rad,rad), (rad,rad)),'edge');
            nSeeds = self.data_cube.max()

            # do not smooth ECS labels
            sel_ECS, ECS_label = self.getECS(image_with_brd)
            if self.dpCleanLabels_verbose and ECS_label:
                print('\tignoring ECS label %d' % (ECS_label,))

            # get bounding boxes for each supervoxel in zero padded label volume
            svox_bnd = nd.measurements.find_objects(image_with_zeros)

            # iterate over labels
            nSeeds = self.data_cube.max(); lbls = np.zeros(sz, dtype=self.data_cube.dtype)
            assert( nSeeds == len(svox_bnd) )
            for j in range(nSeeds):
                if ECS_label and j+1 == ECS_label: continue

                #if self.dpCleanLabels_verbose:
                #    print('Smoothing label %d / %d' % (j+1,nSeeds)); t = time.time()

                pbnd = tuple([slice(x.start-rad,x.stop+rad) for x in svox_bnd[j]])
                Lcrp = (image_with_brd[pbnd] == j+1).astype(np.double)

                Lfilt = nd.filters.uniform_filter(Lcrp, size=smooth_size, mode='constant')
                # incase smoothing below contour level, use without smoothing
                if not (Lfilt > contour_level).any(): Lfilt = Lcrp

                # assign smoothed output for current label
                lbls[pbnd][Lfilt > contour_level] = j+1

            # put ECS labels back
            if ECS_label: lbls[sel_ECS] = ECS_label

            if self.dpCleanLabels_verbose:
                print('\tdone in %.4f s' % (time.time() - t))

            self.data_cube = lbls[rad:-rad,rad:-rad,rad:-rad]

        if self.remove_adjacencies:
            labels = self.data_cube.astype(np.uint32, copy=True, order='C')
            sel_ECS, ECS_label = self.getECS(labels); labels[sel_ECS] = 0

            if self.dpCleanLabels_verbose:
                print('Removing adjacencies with conn %d%s' % (self.fg_connectivity,
                    ', ignoring ECS label %d' % (ECS_label,) if ECS_label else ''))
                t = time.time()

            self.data_cube = emLabels.remove_adjacencies_nconn(labels, bwconn=self.fgbwconn)
            if ECS_label: self.data_cube[sel_ECS] = ECS_label

            if self.dpCleanLabels_verbose:
                print('\tdone in %.4f s' % (time.time() - t))

        if self.minsize > 0:
            labels = self.data_cube
            sel_ECS, ECS_label = self.getECS(labels); labels[sel_ECS] = 0

            if self.dpCleanLabels_verbose:
                print('Scrubbing labels with minsize %d%s' % (self.minsize,
                    ', ignoring ECS label %d' % (ECS_label,) if ECS_label else ''))
                print('\tnlabels = %d, before re-label' % (labels.max(),))
                t = time.time()

            selbg = np.logical_and((labels == 0), np.logical_not(sel_ECS))
            labels, sizes = emLabels.thresholdSizes(labels, minSize=self.minsize)
            if self.minsize_fill:
                if self.dpCleanLabels_verbose:
                    print('Nearest neighbor fill scrubbed labels')
                labels = emLabels.nearest_neighbor_fill(labels, mask=selbg, sampling=self.data_attrs['scale'])

            nlabels = sizes.size
            labels, nlabels = self.setECS(labels, sel_ECS, ECS_label, nlabels)
            self.data_cube = labels
            # allow this to work before self.get_svox_type or self.write_voxel_type
            self.data_attrs['types_nlabels'] = [nlabels]

            if self.dpCleanLabels_verbose:
                print('\tnlabels = %d after re-label' % (nlabels,))
                print('\tdone in %.4f s' % (time.time() - t))

        if self.cavity_fill:
            if self.dpCleanLabels_verbose:
                print('Removing cavities using conn %d' % (self.bg_connectivity,)); t = time.time()

            selbg = (self.data_cube == 0)
            if self.dpCleanLabels_verbose:
                print('\tnumber bg vox before = %d' % (selbg.sum(dtype=np.int64),))
            labels = np.ones([x + 2 for x in self.data_cube.shape], dtype=np.bool)
            labels[1:-1,1:-1,1:-1] = selbg
            # don't connect the top and bottom xy planes
            labels[1:-1,1:-1,0] = 0; labels[1:-1,1:-1,-1] = 0
            labels, nlabels = nd.measurements.label(labels, self.bgbwconn)
            msk = np.logical_and((labels[1:-1,1:-1,1:-1] != labels[0,0,0]), selbg); del labels
            self.data_cube[msk] = 0; selbg[msk] = 0
            self.data_cube = emLabels.nearest_neighbor_fill(self.data_cube, mask=selbg,
                sampling=self.data_attrs['scale'])

            if self.dpCleanLabels_verbose:
                print('\tnumber bg vox after = %d' % ((self.data_cube==0).sum(dtype=np.int64),))
                print('\tdone in %.4f s' % (time.time() - t))

            if self.get_svox_type or self.write_voxel_type:
                if self.dpCleanLabels_verbose:
                    print('\tRemoving cavities from voxel type'); t = time.time()
                if ntypes-1 > 1:
                    voxel_type = emLabels.nearest_neighbor_fill(voxel_type, mask=selbg,
                                                                sampling=self.data_attrs['scale'])
                else:
                    voxel_type[msk] = 1
                print('\t\tdone in %.4f s' % (time.time() - t))

            del msk, selbg

        if self.relabel:
            labels = self.data_cube
            sel_ECS, ECS_label = self.getECS(labels); labels[sel_ECS] = 0

            if self.dpCleanLabels_verbose:
                print('Relabeling fg components with conn %d%s' % (self.fg_connectivity,
                    ', ignoring ECS label %d' % (ECS_label,) if ECS_label else ''))
                print('\tnlabels = %d, max = %d, before re-label' % (len(np.unique(labels)), labels.max()))
                t = time.time()

            labels, nlabels = nd.measurements.label(labels, self.fgbwconn)

            labels, nlabels = self.setECS(labels, sel_ECS, ECS_label, nlabels)
            self.data_cube = labels

            if self.dpCleanLabels_verbose:
                print('\tnlabels = %d after re-label' % (nlabels,))
                print('\tdone in %.4f s' % (time.time() - t))

        # this step is always last, as writes new voxel_type depending on the cleaning that was done
        if self.get_svox_type or self.write_voxel_type:
            if self.dpCleanLabels_verbose:
                print('Recomputing supervoxel types and re-ordering labels'); t = time.time()

            # moved this as first step to allow other steps to modify voxel_type
            #voxType = emVoxelType.readVoxType(srcfile=self.srcfile, chunk=self.chunk.tolist(),
            #    offset=self.offset.tolist(), size=self.size.tolist())
            #voxel_type = voxType.data_cube.copy(order='C')
            #ntypes = len(voxType.data_attrs['types'])

            labels = self.data_cube.copy(order='C')
            #nlabels = labels.max(); assert(nlabels == self.data_attrs['types_nlabels'][0])
            nlabels = sum(self.data_attrs['types_nlabels'])
            supervoxel_type, voxel_type = emLabels.type_components(labels, voxel_type, nlabels, ntypes)
            assert( supervoxel_type.size == nlabels )
            # reorder labels so that supervoxels are grouped by / in order of supervoxel type
            remap = np.zeros((nlabels+1,), dtype=self.data_cube.dtype)
            remap[np.argsort(supervoxel_type)+1] = np.arange(1,nlabels+1,dtype=self.data_cube.dtype)
            self.data_cube = remap[self.data_cube]
            types_nlabels = [(supervoxel_type==x).sum(dtype=np.int64) for x in range(1,ntypes)]
            assert( sum(types_nlabels) == nlabels ) # indicates voxel type does not match supervoxels
            self.data_attrs['types_nlabels'] = types_nlabels

            if self.write_voxel_type:
                if self.dpCleanLabels_verbose:
                    print('Rewriting voxel type pixel data based on supervoxel types')
                d = voxType.data_attrs.copy(); #d['types_nlabels'] =
                emVoxelType.writeVoxType(outfile=self.outfile, chunk=self.chunk.tolist(),
                    offset=self.offset.tolist(), size=self.size.tolist(), datasize=voxType.datasize.tolist(),
                    chunksize=voxType.chunksize.tolist(), data=voxel_type.astype(emVoxelType.VOXTYPE_DTYPE), attrs=d)

            if self.dpCleanLabels_verbose:
                print('\tdone in %.4f s' % (time.time() - t))

    def getECS(self, labels):
        if self.ECS_label > 0:
            sel_ECS = (labels == self.ECS_label); ECS_label = self.ECS_label
        elif self.ECS_label < 0:
            ECS_label = labels.max(); sel_ECS = (labels == ECS_label)
        else:
            sel_ECS = np.zeros(labels.shape, dtype=np.bool); ECS_label = None
        return sel_ECS, ECS_label

    def setECS(self, labels, sel_ECS, ECS_label, nlabels):
        # special re-assignment needed for ECS_label after re-labeling
        if self.ECS_label > 0:
            sel = labels >= ECS_label; labels[sel] += ECS_label; labels[sel_ECS] = ECS_label; nlabels += 1
        elif self.ECS_label < 0:
            ECS_label = labels.max()+1; labels[sel_ECS] = ECS_label; nlabels += 1

        if self.min_label > 1:
            assert( self.ECS_label==0 or self.ECS_label==1 ) # xxx - did not fix other ECS labeling schemes
            selfg = np.logical_and((labels > 0), np.logical_not(sel_ECS))
            labels[selfg] += (self.min_label - 1); nlabels += (self.min_label - 1)

        return labels, nlabels

    @staticmethod
    def addArgs(p):
        # adds arguments required for this object to specified ArgumentParser object
        dpWriteh5.addArgs(p)

        # possible actions, suggest running one at a time since no easy way to specify the order
        # 3d smoothing of labels (done per label)
        p.add_argument('--smooth', action='store_true', help='Perform 3d smoothing on labels')
        # remove components smaller than size (using voxel counts only)
        p.add_argument('--minsize', nargs=1, type=int, default=[-1], metavar=('size'),
            help='Minimum label size in voxels to keep')
        p.add_argument('--minsize_fill', action='store_true',
                       help='Whether to nearest neighbor fill labels scrubbed with minsize')
        # remove adjacencies
        p.add_argument('--remove_adjacencies', action='store_true',
                       help='Perform 3d adjacency removal using fg-connectivity')
        # remove cavities
        p.add_argument('--cavity-fill', action='store_true', help='Remove all BG not connected to cube faces')
        # rerun labeling (connected components)
        p.add_argument('--relabel', action='store_true', help='Re-label components (run connected components)')
        # recompute voxel type based on majority winner for each supervoxel
        p.add_argument('--get-svox-type', action='store_true', help='Recompute supervoxel type using majority method')
        p.add_argument('--write-voxel-type', action='store_true',
            help='Perform get-svox-type and also write out voxel-type based on supervoxels')
        # make overlay that traces minpath between particular label value
        p.add_argument('--minpath', nargs=1, type=int, default=[-1], metavar=('label'),
            help='Calculate min foreground path between specified label (default off)')
        p.add_argument('--minpath-skel', action='store_true',
                       help='Whether to skeletonize min foreground path (for use with minpath)')

        # used to set min label value for both relabel and minsize
        p.add_argument('--min-label', nargs=1, type=int, default=[1], metavar=('min'),
                       help='First label after relabel if relabeling (also minsize)')

        # other options
        p.add_argument('--fg-connectivity', nargs=1, type=int, default=[1], choices=[1,2,3],
            help='Connectivity for foreground (where applicable)')
        p.add_argument('--bg-connectivity', nargs=1, type=int, default=[1], choices=[1,2,3],
            help='Connectivity for background (where applicable)')
        p.add_argument('--ECS-label', nargs=1, type=int, default=[1], metavar=('size'),
            help='Specify which label is ECS (== 0 means none, < 0 means max label)')

        p.add_argument('--dpCleanLabels-verbose', action='store_true', help='Debugging output for dpCleanLabels')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write labels hdf5 file after some manipulations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpCleanLabels.addArgs(parser)
    args = parser.parse_args()

    cleanLbls = dpCleanLabels(args)
    if cleanLbls.inraw:
        cleanLbls.loadFromRaw()
    else:
        cleanLbls.readCubeToBuffers()
    cleanLbls.clean()
    cleanLbls.writeCube()
