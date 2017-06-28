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

# "Cleaning" is used both for:
#       (1) cleaning up manually annotated ground truth (GT) labels (typically from single labeler)
#       (2) cleaning up automated labels (supervoxels) as a last step before meshing.
#       (3) cleaning up and calculating paths for "proof-reading" or "boot-strapping" 
#             xxx - this method did not work super well, so has been abandoned for now
# xxx - better quantification of the value of these steps, (2) has been difficult to measure but (1) could be measured
#   cleaned supervoxels are at least aesthetically much more pleasing than before cleaning

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
        if self.get_svox_type or self.write_voxel_type or self.apply_bg_mask:
            if self.dpCleanLabels_verbose:
                print('Reading supervoxel types'); t = time.time()

            voxType = emVoxelType.readVoxType(srcfile=self.srcfile, chunk=self.chunk.tolist(),
                offset=self.offset.tolist(), size=self.size.tolist())
            voxel_type = voxType.data_cube.copy(order='C')
            ntypes = len(voxType.data_attrs['types'])

            if self.dpCleanLabels_verbose:
                print('\tdone in %.4f s' % (time.time() - t))

        # minpath overlay creation is intended to improve proofreading speed by highlighting connected paths
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
                    d = d1+d2; bwlabels = ((d.data < self.minpath_perc*d.min()) & ~d.mask); bwlabels[s1 | s2] = 1
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

            # exposed smoothing kernel size and contour level as parameters
            smooth_size = self.smooth_size
            contour_level = self.contour_lvl
            # calculate padding based on smoothing kernel size
            rad = int(1.5*smooth_size.max())

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
            labels, nlabels = self.minsize_scrub(labels, self.minsize, self.minsize_fill)
            self.data_cube = labels
            # allow this to work before self.get_svox_type or self.write_voxel_type
            self.data_attrs['types_nlabels'] = [nlabels]

        # NOTE: cavity_fill not intended to work with ECS labeled with single value (ECS components are fine)
        if self.cavity_fill:
            self.data_cube, selbg, msk = self.cavity_fill_voxels(self.data_cube)

            # this prevents any supervoxels as being classified as "membrane".
            # many scripts assume that membrane is labeled as background (label 0).
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

        # NOTE: cavity_fill not intended to work with ECS labeled with single value (ECS components are fine)
        # This method is not foolproof, in one shot it removes all labels below a specified size and then reruns
        #   cavity fill. If a label was not filled then it puts it back. This will not remove any labels that are
        #   in a cavity but connected to background via a bunch of other labels smaller than specified size.
        if self.cavity_fill_minsize > 1:
            if self.dpCleanLabels_verbose:
                print('Removing labels < %d in cavities' % (self.cavity_fill_minsize,))
            labels_orig = self.data_cube
            data, nlabels = self.minsize_scrub(labels_orig, self.cavity_fill_minsize, False, tab=True, no_remap=True)

            # do a normal cavity fill after labels smaller than cavity_fill_minsize are removed
            labels, selbg, msk = self.cavity_fill_voxels(data, tab=True); del msk, selbg
                
            if self.dpCleanLabels_verbose:
                print('\tReplacing non-cavity labels')
            sel_not_fill = np.logical_and(labels_orig > 0, labels == 0)
            labels[sel_not_fill] = labels_orig[sel_not_fill]
            del labels_orig, sel_not_fill
            # remove any zero labels (that were removed as cavities)
            self.data_cube, nlabels = self.minsize_scrub(labels, 1, False, tab=True); del labels
            # allow this to work before self.get_svox_type or self.write_voxel_type
            self.data_attrs['types_nlabels'] = [nlabels]

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
            # allow this to work before self.get_svox_type or self.write_voxel_type
            self.data_attrs['types_nlabels'] = [nlabels]

            if self.dpCleanLabels_verbose:
                print('\tnlabels = %d after re-label' % (nlabels,))
                print('\tdone in %.4f s' % (time.time() - t))

        # this step re-writes the original background (membrane) mask back to the supervoxels.
        # this is useful if agglomeration was done using the completely watershedded supervoxels.
        if self.apply_bg_mask:
            if self.dpCleanLabels_verbose:
                print('Applying background (membrane) mask to supervoxels'); t = time.time()
            sel = (voxel_type == 0)
            self.data_cube[sel] = 0
            if self.dpCleanLabels_verbose:
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

        # this step should not be mixed with other steps
        if self.replace_ECS:
            assert( len(self.data_attrs['types_nlabels']) == 2 )
            sel_ECS = (self.data_cube > self.data_attrs['types_nlabels'][0]);
            sel_ICS = np.logical_and(self.data_cube > 0, self.data_cube <= self.data_attrs['types_nlabels'][0])
            self.data_cube[sel_ICS] += self.min_label
            self.data_cube[sel_ECS] = self.ECS_label
            # xxx - probably shouldn't be using this anyways?
            self.data_attrs['types_nlabels'] = self.data_attrs['types_nlabels'][0]

    def minsize_scrub(self, labels, minsize, minsize_fill, tab=False, no_remap=False):
        tabc = '\t' if tab else ''
        sel_ECS, ECS_label = self.getECS(labels); labels[sel_ECS] = 0

        if self.dpCleanLabels_verbose:
            print('%sScrubbing labels with minsize %d%s' % (tabc, minsize,
                ', ignoring ECS label %d' % (ECS_label,) if ECS_label else ''))
            print('%s\tnlabels = %d, before re-label' % (tabc, labels.max(),))
            t = time.time()

        selbg = np.logical_and((labels == 0), np.logical_not(sel_ECS))
        labels, sizes = emLabels.thresholdSizes(labels, minSize=minsize, no_remap=no_remap)
        if minsize_fill:
            if self.dpCleanLabels_verbose:
                print('%s\tNearest neighbor fill scrubbed labels' % (tabc,))
            labels = emLabels.nearest_neighbor_fill(labels, mask=selbg, sampling=self.data_attrs['scale'])

        nlabels = sizes.size
        labels, nlabels = self.setECS(labels, sel_ECS, ECS_label, nlabels)

        if self.dpCleanLabels_verbose:
            print('%s\tnlabels = %d after re-label' % (tabc, nlabels,))
            print('%s\tdone in %.4f s' % (tabc, time.time() - t))

        return labels, nlabels

    def cavity_fill_voxels(self, data, tab=False):
        tabc = '\t' if tab else ''
        if self.dpCleanLabels_verbose:
            print('%sRemoving cavities using conn %d' % (tabc, self.bg_connectivity,)); t = time.time()
            
        selbg = (data == 0)
        if self.dpCleanLabels_verbose:
            print('%s\tnumber bg vox before = %d' % (tabc, selbg.sum(dtype=np.int64),))
        labels = np.ones([x + 2 for x in data.shape], dtype=np.bool)
        labels[1:-1,1:-1,1:-1] = selbg
        # don't connect the top and bottom xy planes
        labels[1:-1,1:-1,0] = 0; labels[1:-1,1:-1,-1] = 0
        labels, nlabels = nd.measurements.label(labels, self.bgbwconn)
        msk = np.logical_and((labels[1:-1,1:-1,1:-1] != labels[0,0,0]), selbg); del labels
        #data[msk] = 0; # xxx - had this originally, seems redundant, delete this after verified
        selbg[msk] = 0
        filled = emLabels.nearest_neighbor_fill(data, mask=selbg, sampling=self.data_attrs['scale'])

        if self.dpCleanLabels_verbose:
            print('%s\tnumber bg vox after = %d' % (tabc, (filled==0).sum(dtype=np.int64),))
            print('%s\tdone in %.4f s' % (tabc, time.time() - t))
        
        return filled, selbg, msk

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
        p.add_argument('--smooth-size', nargs=3, type=int, default=[3,3,3], metavar=('X', 'Y', 'Z'),
            help='Size of smoothing kernel')
        p.add_argument('--contour-lvl', nargs=1, type=float, default=[0.25], metavar=('LVL'),
            help='Level [0,1] to use to create mesh isocontours')
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
        # remove any labels less than specified size that are within cavities
        p.add_argument('--cavity-fill-minsize', nargs=1, type=int, default=[1], metavar=('size'),
            help='Minimum label size to replace labels in cavities (force cavity fill)')
        # rerun labeling (connected components)
        p.add_argument('--relabel', action='store_true', help='Re-label components (run connected components)')
        # write background (membrane mask) using the voxel type
        p.add_argument('--apply-bg-mask', action='store_true', 
                       help='Write voxel-type background (membrane) mask to supervoxels')
        # recompute voxel type based on majority winner for each supervoxel
        p.add_argument('--get-svox-type', action='store_true', help='Recompute supervoxel type using majority method')
        p.add_argument('--write-voxel-type', action='store_true',
            help='Perform get-svox-type and also write out voxel-type based on supervoxels')
        p.add_argument('--replace-ECS', action='store_true', help='Replace all ECS supervoxels with ECS-label')
        # make overlay that traces minpath between particular label value
        p.add_argument('--minpath', nargs=1, type=int, default=[-1], metavar=('label'),
            help='Calculate min foreground path between specified label (default off)')
        p.add_argument('--minpath-perc', nargs=1, type=float, default=[1.01], metavar=('perc'),
            help='Percentage away from bwmin to label in minpath overlay')
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
