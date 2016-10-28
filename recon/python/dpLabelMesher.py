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

# Python object for reading and EM label data and doing surface meshing for labrainth "Task 3" and "Task 5".
# Forked off from above mentioned frontend version called lbSegmentToMeshLblEM.py 27 Oct 2016
# Moved out database read/write code and modified for pre-meshing for Knossos surface rendering of supervoxels.
#
# Some references for vtk:
# https://pyscience.wordpress.com/2014/09/06/numpy-to-vtk-converting-your-numpy-arrays-to-vtk-arrays-and-files/
# http://vtk.1045678.n5.nabble.com/vtk-to-numpy-how-to-get-a-vtk-array-td1244891.html
# http://www.bu.edu/tech/support/research/training-consulting/online-tutorials/vtk/
# https://pyscience.wordpress.com/2014/09/11/surface-extraction-creating-a-mesh-from-pixel-data-using-python-and-vtk/

import os
import numpy as np
import h5py
import argparse
import time

from scipy import ndimage as nd
import scipy.ndimage.filters as filters
import vtk
from vtk.util import numpy_support as nps

#from dpLoadh5 import dpLoadh5
from dpWriteh5 import dpWriteh5
from typesh5 import emLabels
from utils import showImgData

class dpLabelMesher(emLabels):

    # type to use for all processing operations
    PDTYPE = np.double
    #DTYPE = np.uint16
    DTYPE_MAX = 65535
    RAD = 5  # for padding
    doplots = False  # for debug

    # moved seed_range back to command line
    #seed_range = [-1,-1]    # for normal use
    #seed_range = [-1,10]    # for debug

    SEED_DTYPE = np.uint32
    #VERTEX_DTYPE = np.float32
    #FACE_DTYPE = np.uint32
    VERTEX_DTYPE = np.float16   # use half precision floating point to save space
    FACE_DTYPE = np.uint16      # assumes there are less than 64k vertices for each supervoxel

    def __init__(self, args):
        emLabels.__init__(self,args)

        #self.data_type = self.DTYPE
        self.do_smooth = self.smooth.all()

        self.mesh_outfile_stl = ''
        if self.mesh_outfile:
            # vtk does not seem to like home ~/ or env vars in path
            self.mesh_outfile = os.path.expandvars(os.path.expanduser(self.mesh_outfile))

            ext = os.path.splitext(self.mesh_outfile)[1][1:]
            if ext == 'stl':
                self.mesh_outfile_stl = self.mesh_outfile
            else:
                assert( ext == 'h5' ) # only stl or h5 outputs supported

        # print out all initialized variables in verbose mode
        if self.dpLabelMesher_verbose: print('dpLabelMesher, verbose mode:\n'); print(vars(self))

    def procData(self):
        # double negative so that the name doesn't collide with the same option that is off by default in dpLoadh5
        #if not self.no_legacy_transpose: cube = self.data_cube.transpose((1,0,2))
        # changed this to off by default and use same flag from dpLoadh5
        if self.legacy_transpose: cube = self.data_cube.transpose((1,0,2))
        else: cube = self.data_cube

        # Pad data with zeros so that meshes are closed on the edges
        sizes = np.array(cube.shape); r = self.RAD; sz = sizes + 2*r;
        dataPad = np.zeros(sz, dtype=self.data_type); dataPad[r:sz[0]-r, r:sz[1]-r, r:sz[2]-r] = cube

        # old method
        #        # get all unique seeds in the cube
        #        self.seeds = np.unique(cube)
        #        # remove the background label (label 0)
        #        if self.seeds.size > 0 and self.seeds[0] == 0: self.seeds = self.seeds[1:]

        # get sizes first with hist (prevents sums in meshing loop)
        #self.nVoxels = emLabels.getSizes(cube)[1:]
        self.nVoxels = emLabels.getSizesMax(cube, sum(self.data_attrs['types_nlabels']))[1:]
        self.seeds = np.arange(1, self.nVoxels.size+1, dtype=np.int64); self.seeds = self.seeds[self.nVoxels>0]

        assert( self.seeds.size > 0 )   # error, no labels
        n = self.seeds.size; #self.nVoxels = np.zeros((n,), dtype=np.int64)
        assert( n == self.seeds[-1] or not self.mesh_outfile_stl )   # for consistency with stl file, no empty labels

        # intended for debug, only process a subset of the seeds
        if self.seed_range[0] < 1 or self.seed_range[0] > n: self.seed_range[0] = 0
        if self.seed_range[1] < 1 or self.seed_range[1] < 0: self.seed_range[1] = n

        # other inits
        if self.do_smooth: W = np.ones(self.smooth, dtype=self.PDTYPE) / self.smooth.prod()

        # allocate outputs
        self.faces = n * [None]; self.vertices = n * [None]; self.mins = n * [None]; self.rngs = n * [None]
        self.bounds_beg = n * [None]; self.bounds_end = n * [None]
        self.nFaces = np.zeros((n,), dtype=np.uint64); self.nVertices = np.zeros((n,), dtype=np.uint64);
        if self.doplots or self.mesh_outfile_stl: self.allPolyData = vtk.vtkAppendPolyData()

        # get bounding boxes for each supervoxel
        svox_bnd = nd.measurements.find_objects(dataPad)

        if self.dpLabelMesher_verbose: tloop = time.time()
        for i in range(self.seed_range[0], self.seed_range[1]):
            if self.dpLabelMesher_verbose:
                print('seed : %d is %d / %d' % (self.seeds[i],i+1,self.seed_range[1])); t = time.time()

            # old method
            #            # select the labels
            #            #bwdpls = (dataPad == self.seeds[i]);
            #            #self.nVoxels[i] = bwdpls.sum();
            #            if self.dpLabelMesher_verbose: print('\tnVoxels = %d' % self.nVoxels[i])
            #
            #            # get the voxel coordinates relative to padded and non-padded cube
            #            idpls = np.argwhere(bwdpls)
            #            # bounding box within zero padded cube
            #            imin = idpls.min(axis=0); imax = idpls.max(axis=0)

            cur_bnd = svox_bnd[self.seeds[i]-1]
            imin = np.array([x.start for x in cur_bnd]); imax = np.array([x.stop-1 for x in cur_bnd])
            # min and max coordinates of this seed within zero padded cube
            pmin = imin - r; pmax = imax + r;
            # min coordinates of this seed relative to original (non-padded cube)
            self.mins[i] = pmin - r; self.rngs[i] = pmax - pmin + 1

            # old method
            # crop out the bounding box plus the padding, then optionally smooth
            #crpdpls = bwdpls[pmin[0]:pmax[0]+1,pmin[1]:pmax[1]+1,pmin[2]:pmax[2]+1].astype(self.PDTYPE)
            # crop out the bounding box then binarize this seed within bounding box
            crpdpls = (dataPad[cur_bnd] == self.seeds[i]).astype(self.PDTYPE)
            if self.do_smooth:
                crpdplsSm = filters.convolve(crpdpls, W, mode='reflect', cval=0.0, origin=0)
                # if smoothing results in nothing above contour level, use original without smoothing
                if (crpdplsSm > self.contour_lvl).any():
                    del crpdpls; crpdpls = crpdplsSm; del crpdplsSm
            if self.doplots: showImgData(np.squeeze(crpdpls[:,:,crpdpls.shape[2]/2]),'slice')

            # vtkImageImport is used to create image data from memory in vtk
            # http://wiki.scipy.org/Cookbook/vtkVolumeRendering
            dataImporter = vtk.vtkImageImport()
            # The preaviusly created array is converted to a byte string (not string, see np docs) and imported.
            data_string = crpdpls.transpose((2,1,0)).tostring();
            dataImporter.CopyImportVoidPointer(data_string, len(data_string))
            # Set the type of the newly imported data
            #dataImporter.SetDataScalarTypeToUnsignedChar()
            #dataImporter.SetDataScalarTypeToUnsignedShort()
            dataImporter.SetDataScalarTypeToDouble()
            # Because the data that is imported only contains an intensity value (i.e. not RGB), the importer
            # must be told this is the case.
            dataImporter.SetNumberOfScalarComponents(1)
            # Data extent is the extent of the actual buffer, whole extent is ???
            # Use extents that are relative to non-padded cube
            beg = self.mins[i]; end = np.array(crpdpls.shape) - 1 + beg
            dataImporter.SetDataExtent(beg[0], end[0], beg[1], end[1], beg[2], end[2])
            dataImporter.SetWholeExtent(beg[0], end[0], beg[1], end[1], beg[2], end[2])

            # save bounds relative to entire dataset
            self.bounds_beg[i] = beg + self.dataset_index; self.bounds_end[i] = end + self.dataset_index;

            # use vtk for isosurface contours and surface mesh reduction
            iso = vtk.vtkContourFilter()
            iso.SetInputConnection(dataImporter.GetOutputPort())
            iso.SetComputeNormals(0)
            iso.SetValue(0, self.contour_lvl)
            if self.decimatePro:
                deci = vtk.vtkDecimatePro()
                rf = 1-self.reduce_frac; deci.SetTargetReduction(rf); df = 0.01
                deci.SplittingOn(); deci.PreserveTopologyOff(); deci.BoundaryVertexDeletionOn()
                if self.min_faces > 0: updates = range(100)
                else: updates = ['deci.BoundaryVertexDeletionOff()','deci.PreserveTopologyOn()','0']
            else:
                deci = vtk.vtkQuadricClustering()
                #deci.SetDivisionOrigin(0.0,0.0,0.0); deci.SetDivisionSpacing(self.reduce_spacing)
                nb = self.reduce_nbins; deci.SetNumberOfDivisions(nb,nb,nb); deci.AutoAdjustNumberOfDivisionsOff()
                updates = ['deci.AutoAdjustNumberOfDivisionsOn()','0']

            # thought of adding checking for closed surfaces, http://comments.gmane.org/gmane.comp.lib.vtk.user/47957
            # this did not work, for low reduce_frac, many open edges remain even for large objects

            # not clear that triangle filter does anything, contour filter already makes triangulated meshes?
            # send polygonal mesh from isosurface to triangle filter to convert to triangular mesh
            #tri = vtk.vtkTriangleFilter(); tri.SetInputConnection(iso.GetOutputPort());
            #deci.SetInputConnection(tri.GetOutputPort())

            deci.SetInputConnection(iso.GetOutputPort())
            # xxx - this is kindof a cheap trick, if we reduce down "too much", then rerun to preserve more
            for update in updates:
                deci.Update()

                # http://forrestbao.blogspot.com/2012/06/vtk-polygons-and-other-cells-as.html
                # http://stackoverflow.com/questions/6684306/how-can-i-read-a-vtk-file-into-a-python-datastructure
                dOut = deci.GetOutput()
                # xxx - points seem to be single instead of inputted type, probably depends on vtk version:
                #   http://public.kitware.com/pipermail/vtkusers/2010-April/059413.html
                self.vertices[i] = nps.vtk_to_numpy(dOut.GetPoints().GetData())
                if self.center_origin:
                    self.vertices[i][:,0] -= sizes[0]/2; self.vertices[i][:,1] -= sizes[1]/2
                    self.vertices[i][:,2] = sizes[2]/2 - self.vertices[i][:,2]
                self.faces[i] = nps.vtk_to_numpy(dOut.GetPolys().GetData()).reshape((-1,4))[:,1:]
                if self.flip_faces: self.faces[i] = self.faces[i][:,::-1]
                self.nVertices[i] = self.vertices[i].shape[0]
                self.nFaces[i] = self.faces[i].shape[0]
                if self.dpLabelMesher_verbose: print('\t%d vertices, %d faces' % (self.nVertices[i], self.nFaces[i]))
                if self.min_faces > 0:
                    if self.nFaces[i] >= self.min_faces: break
                    rf -= df; deci.SetTargetReduction(rf)
                else:
                    if self.nVertices[i] > 2 and self.nFaces[i] > 0: break
                    eval(update)
            assert( self.nVertices[i] > 2 and self.nFaces[i] > 0 )  # there has to be at least one face

            if self.doplots:
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(deci.GetOutputPort())
                dpLabelMesher.vtkShow(mapper)

            # append the current surface to vtk object with all the surfaces
            if self.doplots or self.mesh_outfile_stl:
                self.allPolyData.AddInputConnection(deci.GetOutputPort())
            if self.doplots:
                connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
                connectivityFilter.SetInputConnection(self.allPolyData.GetOutputPort())
                connectivityFilter.SetExtractionModeToAllRegions()
                connectivityFilter.ColorRegionsOn()
                connectivityFilter.Update()
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(connectivityFilter.GetOutputPort())
                mapper.SetScalarRange(connectivityFilter.GetOutput().GetPointData().GetArray("RegionId").GetRange())
                dpLabelMesher.vtkShow(mapper)

            if self.dpLabelMesher_verbose: print('\tdone in %.3f s' % (time.time() - t,))
        if self.dpLabelMesher_verbose: print('Total ellapsed time meshing %.3f s' % (time.time() - tloop,))

    def writeMeshOutfile(self):
        if not self.mesh_outfile: return

        if self.mesh_outfile_stl:
            if self.dpLabelMesher_verbose: print('Writing output to %s' % self.mesh_outfile_stl)
            connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
            connectivityFilter.SetInputConnection(self.allPolyData.GetOutputPort())
            connectivityFilter.SetExtractionModeToAllRegions()
            connectivityFilter.ColorRegionsOn()
            connectivityFilter.Update()
            writer = vtk.vtkSTLWriter()
            #writer = vtk.vtkPolyDataWriter()
            writer.SetInputConnection(connectivityFilter.GetOutputPort())
            writer.SetFileTypeToBinary()
            writer.SetFileName(self.mesh_outfile_stl)
            writer.Write()
        else:
            if self.dpLabelMesher_verbose: print('Writing output to %s' % self.mesh_outfile)
            h5file = h5py.File(self.mesh_outfile, 'w')
            dataset_root = 'meshes'
            for i in range(self.seed_range[0], self.seed_range[1]):
                #self.nVoxels; self.faces; self.vertices; self.bounds_beg; self.bounds_end
                dsetpath = dataset_root + '/' + str(self.seeds[i])
                h5file.create_dataset(dsetpath + '/faces', data=self.faces[i], dtype=self.FACE_DTYPE,
                                      compression='gzip',compression_opts=self.HDF5_CLVL,shuffle=True,fletcher32=True)
            h5file.close()

    @classmethod
    def labelMesher(cls, srcfile, dataset, chunk, offset, size, reduce_frac, verbose=False):
        parser = argparse.ArgumentParser(description='class:dpLabelMesher',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        dpLabelMesher.addArgs(parser); arg_str = ''
        arg_str += ' --srcfile ' + srcfile
        arg_str += ' --chunk %d %d %d ' % tuple(chunk)
        arg_str += ' --offset %d %d %d ' % tuple(offset)
        arg_str += ' --size %d %d %d ' % tuple(size)
        arg_str += ' --reduce-frac %f ' % reduce_frac
        arg_str += ' --reduce-frac %f ' % reduce_frac
        arg_str += ' --dataset ' + dataset
        if verbose: arg_str += ' --dpLabelMesher-verbose '
        if verbose: print(arg_str)
        args = parser.parse_args(arg_str.split())
        stm = cls(args)
        return stm

    @staticmethod
    def vtkShow(mapper):
        # open a window and display the data specified by mapper
        # need an actor and a renderer to display data
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(1.0, 1.0, 1.0)

        # optionally setup the camera (xxx - and lighting?)
        #camera = renderer.MakeCamera()
        #camera.SetPosition(-500.0, 245.5, 122.0)
        #camera.SetFocalPoint(301.0, 245.5, 122.0)
        #camera.SetViewAngle(30.0)
        #camera.SetRoll(-90.0)
        #renderer.SetActiveCamera(camera)

        # setup the renderer window / interactor and run
        renderWin = vtk.vtkRenderWindow()
        renderWin.AddRenderer(renderer)
        renderInteractor = vtk.vtkRenderWindowInteractor()
        renderInteractor.SetRenderWindow(renderWin)
        renderWin.SetSize(800, 800)
        renderInteractor.Initialize()
        renderWin.Render()
        renderInteractor.Start()

    @staticmethod
    def addArgs(p):
        # adds arguments required for this object to specified ArgumentParser object
        dpWriteh5.addArgs(p)
        p.add_argument('--mesh-outfile', nargs=1, type=str, default='', help='Output label mesh file (stl)')
        p.add_argument('--reduce-frac', nargs=1, type=float, default=[0.2], metavar=('PERC'),
            help='Reduce fraction for reducing meshes (decimate pro)')
        #p.add_argument('--reduce-spacing', nargs=3, type=float, default=[10.0, 10.0, 5.0], metavar=('SPC'),
        #    help='Voxel spacing to use for mesh decimation (quadric clustering)')
        p.add_argument('--reduce-nbins', nargs=1, type=int, default=[10], metavar=('NBINS'),
            help='Number of bins to use for mesh decimation (quadric clustering)')
        p.add_argument('--min-faces', nargs=1, type=int, default=[6], metavar=('NFACES'),
            help='Minimum number of faces for each mesh (decimate pro, <= 0 to use update method)')
        p.add_argument('--smooth', nargs=3, type=int, default=[3,3,3], metavar=('X', 'Y', 'Z'),
            help='Size of smoothing kernel (zeros for none)')
        p.add_argument('--contour-lvl', nargs=1, type=float, default=[0.25], metavar=('LVL'),
            help='Level [0,1] to use to create mesh isocontours')
        p.add_argument('--center-origin', action='store_true', dest='center_origin',
            help='Do the weird "origin centering" transformation')
        #        p.add_argument('--no-flip-faces', action='store_false', dest='flip_faces',
        #            help='Do change the face order coming from vtk')
        p.add_argument('--flip-faces', action='store_true', dest='flip_faces',
            help='Change the face order coming from vtk')
        p.add_argument('--seed-range', nargs=2, type=int, default=[-1,-1], metavar=('BEG', 'END'),
            help='Subset of seeds to process (< 0 for beg/end)')
        p.add_argument('--no-decimatePro', action='store_false', dest='decimatePro',
            help='Do not use decimate pro from vtk for meshing (use quadric clustering instead)')
        #p.add_argument('--decimatePro', action='store_true', dest='decimatePro',
        #    help='Use decimate pro from vtk for meshing (default quadric clustering)')
        p.add_argument('--dpLabelMesher-verbose', action='store_true', help='Debugging output for dpLabelMesher')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create surface meshes from hdf5 label data for Knossos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpLabelMesher.addArgs(parser)
    args = parser.parse_args()

    seg2mesh = dpLabelMesher(args)
    seg2mesh.readCubeToBuffers()
    seg2mesh.procData()
    seg2mesh.writeMeshOutfile()

