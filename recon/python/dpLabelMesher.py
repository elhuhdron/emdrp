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
import glob
import zipfile
import re
from scipy import ndimage as nd
import scipy.ndimage.filters as filters
import vtk
from vtk.util import numpy_support as nps
from skimage.measure import mesh_surface_area

#from dpLoadh5 import dpLoadh5
from dpWriteh5 import dpWriteh5
from typesh5 import emLabels
from utils import showImgData, knossos_read_nml

class dpLabelMesher(emLabels):

    # type to use for all processing operations
    PDTYPE = np.double
    #RAD = 5  # for padding, calculate this based on smoothing kernel size

    #print_every = 500 # modulo for print update
    #dataset_root = 'meshes'

    VERTEX_DTYPE = np.uint16    # decided to use fixed precision to represent vertices, sets max vertex value
    VERTEX_BPLACES = 0          # binary place to fix vertices to (precision depends on if set_voxel_scale)
    FACE_DTYPE = np.uint32      # sets max number of vertices
    BOUNDS_DTYPE = np.uint32    # the bounding box types (sets max x/y/z coordinates)

    max_nvertices = np.iinfo(FACE_DTYPE).max
    max_vertex = 2**(np.iinfo(VERTEX_DTYPE).bits - VERTEX_BPLACES)-1
    max_bounds = 2**(np.iinfo(BOUNDS_DTYPE).bits - VERTEX_BPLACES)-1
    vertex_divisor = 2**VERTEX_BPLACES

    save_params = ['reduce_frac', 'decimatePro', 'reduce_nbins', 'min_faces', 'smooth', 'contour_lvl',
                   'set_voxel_scale', 'scale', 'vertex_divisor', 'nlabels']

    def __init__(self, args):
        self.LIST_ARGS += ['mesh_infiles', 'merge_objects', 'skeletons']
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

        self.nlabels = sum(self.data_attrs['types_nlabels'])

        # print out all initialized variables in verbose mode
        #if self.dpLabelMesher_verbose: print('dpLabelMesher, verbose mode:\n'); print(vars(self))

        # colormap for annotation-file mode (viewing merged meshes / skeletons)
        if self.lut_file:
            self.cmap = np.fromfile(self.lut_file, dtype=np.uint8).reshape((3,-1)).T.astype(np.double)/255
        else:
            # a crappy default
            self.cmap = np.array([ 0.0000, 0.0000, 1.0000,
                                   1.0000, 0.0000, 0.0000,
                                   0.0000, 1.0000, 0.0000,
                                   0.0000, 1.0000, 1.0000,
                                   1.0000, 1.0000, 0.0000,
                                   1.0000, 0.0000, 1.0000,
                                   ],dtype=np.double).reshape((-1,3))

        # xxx - using vtkColorMap or vtkLookupTable with appendPolyData did not work, using multiple actors instead
        #self.colorMap = vtk.vtkColorTransferFunction()
        #self.colorMap.SetColorSpaceToRGB()
        ##cmap = np.array([ 0,0,0,1,
        ##                  1,1,0,0,
        ##                  2,0,1,1,
        ##                  ],dtype=np.double).reshape((-1,4))
        ##self.colorMap.FillFromDataPointer(cmap.shape[0], cmap.tostring())
        #self.colorMap.AddRGBPoint(0.0, 0.0, 0.0, 1.0)
        #self.colorMap.AddRGBPoint(1.0, 1.0, 0.0, 0.0)
        #self.colorMap.AddRGBPoint(2.0, 0.0, 1.0, 0.0)
        #self.colorMap.Build()

        #self.colorMap = vtk.vtkLookupTable()
        #self.colorMap.SetNumberOfTableValues(256)
        #self.colorMap.SetTableValue(0, 0.0000, 0.0000, 1.0000, 1)
        #self.colorMap.SetTableValue(1, 1.0000, 0.0000, 0.0000, 1)
        #self.colorMap.SetTableValue(2, 0.0000, 1.0000, 0.0000, 1)
        #self.colorMap.Build()

    def procData(self):
        # use same flag from dpLoadh5 (only used for raw out manipulation in dpLoadh5)
        if self.legacy_transpose: cube = self.data_cube.transpose((1,0,2))
        else: cube = self.data_cube

        # for easy saving of scale as attribute in hdf5 output
        self.scale = self.data_attrs['scale']

        # get sizes first with hist (prevents sums in meshing loop)
        if self.dpLabelMesher_verbose:
            print('Getting supervoxel sizes using %d max labels' % (self.nlabels,)); t = time.time()
        self.nVoxels = emLabels.getSizes(cube, maxlbls=self.nlabels)[1:]
        if self.dpLabelMesher_verbose:
            print('\tdone in %.3f s' % (time.time() - t,))
        self.seeds = np.arange(1, self.nVoxels.size+1, dtype=np.int64); self.seeds = self.seeds[self.nVoxels>0]
        #print(np.argmax(self.nVoxels))

        r = self.smooth.max() + 1
        if self.dpLabelMesher_verbose:
            print('Padding data with %d zero border' % (r,)); t = time.time()
        # Pad data with zeros so that meshes are closed on the edges
        sizes = np.array(cube.shape); sz = sizes + 2*r;
        dataPad = np.zeros(sz, dtype=self.data_type); dataPad[r:sz[0]-r, r:sz[1]-r, r:sz[2]-r] = cube
        del self.data_cube, cube
        if self.dpLabelMesher_verbose:
            print('\tdone in %.3f s' % (time.time() - t,))

        assert( self.seeds.size > 0 )   # error, no labels
        n = self.seeds.size; #self.nVoxels = np.zeros((n,), dtype=np.int64)
        #assert( n == self.seeds[-1] or not self.mesh_outfile_stl )   # for consistency with stl file, no empty labels

        # intended for debug, only process a subset of the seeds
        if self.seed_range[0] < 1 or self.seed_range[0] > n: self.seed_range[0] = 0
        if self.seed_range[1] < 1 or self.seed_range[1] < 0: self.seed_range[1] = n

        # threw me off in debug twice, if the supervoxels are contiguous then have the seed_range mean actual seed
        if n == self.seeds[-1] and self.seed_range[0] > 0: self.seed_range[0] -= 1

        # other inits
        if self.do_smooth: W = np.ones(self.smooth, dtype=self.PDTYPE) / self.smooth.prod()

        # allocate outputs
        self.faces = n * [None]; self.vertices = n * [None]; self.mins = n * [None]; self.rngs = n * [None]
        self.bounds_beg = n * [None]; self.bounds_end = n * [None]
        self.nFaces = np.zeros((n,), dtype=np.uint64); self.nVertices = np.zeros((n,), dtype=np.uint64);
        if self.doplots or self.mesh_outfile_stl: self.allPolyData = vtk.vtkAppendPolyData()
        # compute and store surface area, number is based on whatever units are stored in the vertices
        self.surface_area = np.zeros((n,), dtype=np.double)

        # get bounding boxes for each supervoxel
        if self.dpLabelMesher_verbose:
            print('Getting supervoxel bounding boxes'); t = time.time()
        svox_bnd = nd.measurements.find_objects(dataPad, self.seeds[self.seed_range[1]-1])
        if self.dpLabelMesher_verbose:
            print('\tdone in %.3f s' % (time.time() - t,))

        if self.dpLabelMesher_verbose:
            tloop = time.time(); t = time.time()
            print('Running meshing on %d seeds' % (self.seed_range[1]-self.seed_range[0],))
            print('seed : %d is %d / %d' % (self.seeds[self.seed_range[0]],self.seed_range[0],self.seed_range[1]))
        for i in range(self.seed_range[0], self.seed_range[1]):

            if self.dpLabelMesher_verbose and (i % self.print_every == 0) and i > self.seed_range[0]:
                print('\tdone in %.3f s' % (time.time() - t,)); t = time.time()
                print('seed : %d is %d / %d' % (self.seeds[i],i+1,self.seed_range[1]))

            cur_bnd = svox_bnd[self.seeds[i]-1]
            imin = np.array([x.start for x in cur_bnd]); imax = np.array([x.stop-1 for x in cur_bnd])

            # min and max coordinates of this seed within zero padded cube
            pmin = imin - r; pmax = imax + r;
            # min coordinates of this seed relative to original (non-padded cube)
            self.mins[i] = pmin - r; self.rngs[i] = pmax - pmin + 1

            crpdpls = (dataPad[pmin[0]:pmax[0]+1,pmin[1]:pmax[1]+1,
                               pmin[2]:pmax[2]+1] == self.seeds[i]).astype(self.PDTYPE)

            if self.do_smooth:
                crpdplsSm = filters.convolve(crpdpls, W, mode='reflect', cval=0.0, origin=0)
                # if smoothing results in nothing above contour level, use original without smoothing
                if (crpdplsSm > self.contour_lvl).any():
                    del crpdpls; crpdpls = crpdplsSm
                del crpdplsSm
            if self.doplots: showImgData(np.squeeze(crpdpls[:,:,crpdpls.shape[2]/2]),'slice')
            self.crpdpls = crpdpls
            # save bounds relative to entire dataset
            self.bounds_beg[i] = self.mins[i] + self.dataset_index
            # xxx - this is an inclusive end!!! but, no one is currently using it (not viewer here or knossos)
            self.bounds_end[i] = self.mins[i] + self.rngs[i] - 1 + self.dataset_index;

            #call the vtk Pipeline
            vertices, faces = self.vtkMesh(self.mins[i],self.rngs[i])
            self.vertices[i] = vertices
            self.faces[i] = faces
            # store vertices and faces for future reference
            self.nVertices[i] = self.vertices[i].shape[0]
            self.nFaces[i] = self.faces[i].shape[0]
            
            # calculate surface area based on units in vertices
            self.surface_area[i] = mesh_surface_area(vertices, faces)

        if self.dpLabelMesher_verbose: print('Total ellapsed time meshing %.3f s' % (time.time() - tloop,))

    def vtkMesh(self,min_coord,min_range):# pass the cropped bounding box start and range of the box
        # vtkImageImport is used to create image data from memory in vtk
        # http://wiki.scipy.org/Cookbook/vtkVolumeRendering

        dataImporter = vtk.vtkImageImport()

        # The preaviusly created array is converted to a byte string (not string, see np docs) and imported.
        data_string = self.crpdpls.transpose((2,1,0)).tostring();
        dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        del data_string
        # Set the type of the newly imported data
        #dataImporter.SetDataScalarTypeToUnsignedChar()
        #dataImporter.SetDataScalarTypeToUnsignedShort()
        dataImporter.SetDataScalarTypeToDouble()
        # Because the data that is imported only contains an intensity value (i.e. not RGB), the importer
        # must be told this is the case.
        dataImporter.SetNumberOfScalarComponents(1)

        if self.set_voxel_scale:
            # Have to set the voxel anisotropy here, as there does not seem an easy way once the poly is created.
            dataImporter.SetDataSpacing(self.data_attrs['scale'])

        # Data extent is the extent of the actual buffer, whole extent is ???
        # Use extents that are relative to non-padded cube
        if(self.annotation_file_mesh):
            beg = min_coord - self.dataset_index
            end = beg + min_range - 1
        else:
            beg = min_coord ;  end = min_range + min_coord - 1;
            #print(beg,end)

        dataImporter.SetDataExtent(beg[0], end[0], beg[1], end[1], beg[2], end[2])
        dataImporter.SetWholeExtent(beg[0], end[0], beg[1], end[1], beg[2], end[2])

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
            vertices = nps.vtk_to_numpy(dOut.GetPoints().GetData())
            faces = nps.vtk_to_numpy(dOut.GetPolys().GetData()).reshape((-1,4))[:,1:]
            nVertices = vertices.shape[0]
            nFaces = faces.shape[0]
            #if self.dpLabelMesher_verbose :
            #    print('\t%d vertices, %d faces' % (nVertices, nFaces))
            if self.min_faces > 0:
                if nFaces >= self.min_faces: break
                rf -= df; deci.SetTargetReduction(rf)
            else:
                if nVertices > 2 and nFaces > 0: break
                eval(update)
        assert( nVertices > 2 and nFaces > 0 )  # there has to be at least one face

        if self.doplots:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(deci.GetOutputPort())
            dpLabelMesher.vtkShow(mapper=mapper)

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
            mapper.SetScalarRange(connectivityFilter.GetOutput().GetPointData().GetArray("RegionId").GetRange())
            dpLabelMesher.vtkShow(mapper=mapper)

        return vertices,faces

    def writeMeshOutfile(self):
        if not self.mesh_outfile: return

        if self.mesh_outfile_stl:
            #if self.dpLabelMesher_verbose: print('Writing output to %s' % self.mesh_outfile_stl)
            '''connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
            connectivityFilter.SetInputConnection(self.allPolyData.GetOutputPort())
            connectivityFilter.SetExtractionModeToAllRegions()
            connectivityFilter.ColorRegionsOn()
            connectivityFilter.Update()
            writer = vtk.vtkSTLWriter()
            #writer = vtk.vtkPolyDataWriter()
            writer.SetInputConnection(connectivityFilter.GetOutputPort())
            writer.SetFileTypeToBinary()
            writer.SetFileName(self.mesh_outfile_stl)
            writer.Write()'''
            print('write_output')
            plyWriter = vtk.vtkPLYWriter()
            plyWriter.SetFileName('1.ply')
            plyWriter.SetInputConnection(self.allPolyData.GetOutputPort())
            plyWriter.Write()


        else:
            if self.dpLabelMesher_verbose:
                print('Writing output to %s' % self.mesh_outfile); t = time.time()

            ## do some checking on the stored types
            #max_nvertices = np.iinfo(self.FACE_DTYPE).max
            #max_vertex = 2**(np.iinfo(self.VERTEX_DTYPE).bits - self.VERTEX_BPLACES)-1
            #max_bounds = 2**(np.iinfo(self.BOUNDS_DTYPE).bits - self.VERTEX_BPLACES)-1
            #bplace = 2**self.VERTEX_BPLACES

            h5file = h5py.File(self.mesh_outfile, 'w')
            for i in range(self.seed_range[0], self.seed_range[1]):
                # need to scale the bounds if the spacing has been set
                mins = self.mins[i]; beg = self.bounds_beg[i]; end = self.bounds_end[i];
                if self.set_voxel_scale:
                    scale = self.data_attrs['scale']; mins = mins*scale; beg = beg*scale; end = end*scale

                # the vertices were calculated relative to the whole area being meshed because of:
                #   dataImporter.SetDataExtent(beg[0], end[0], beg[1], end[1], beg[2], end[2])
                #   dataImporter.SetWholeExtent(beg[0], end[0], beg[1], end[1], beg[2], end[2])
                # wanted this so that for debug in this script the meshes can be rendered in the same axes by vtk.
                # make the vertices relative to the bounding box here before writing to the hdf5 output.
                #vertices = np.round(self.vertices[i] - mins, decimals=self.VERTEX_DROUND)
                # decided to use a fixed point for the vertex coordinates
                vertices = np.fix((self.vertices[i] - mins)*self.vertex_divisor)
                str_seed = ('%08d' % self.seeds[i])
                self.writeData(h5file, beg, end, str_seed, self.faces[i], vertices, self.nVoxels[i], 
                               surface_area=self.surface_area[i])

            self.nlabels = self.seeds[self.seed_range[1]-1]
            self.writeMeta(h5file)
            h5file.close()
            if self.dpLabelMesher_verbose:
                print('\tdone in %.3f s' % (time.time() - t,))

    def writeData(self, h5file, beg, end, str_seed, faces, vertices, nVoxels, surface_area=None):
        nVertices = vertices.shape[0]; nFaces = faces.shape[0]

        # do some checking on the stored types
        if nVertices > self.max_nvertices:
            print('Supervoxel %d (%d voxels) %d vertices' % (str_seed, nVoxels, nVertices))
            assert(False)
        if vertices.max() > self.max_vertex:
            print('Supervoxel %d max vertex %d' % (str_seed, nVoxels, vertices.max()))
            assert(False)
        if beg.max() > self.max_bounds:
            print('Supervoxel %d max beg bound %d' % (str_seed, nVoxels, beg.max()))
            assert(False)
        if end.max() > self.max_bounds:
            print('Supervoxel %d max end bound %d' % (str_seed, nVoxels, end.max()))
            assert(False)

        #self.nVoxels; self.faces; self.vertices; self.bounds_beg; self.bounds_end
        #str_seed = ('%08d' % seed)
        dsetpath = self.dataset_root + '/' + str_seed
        # only enable compression for larger supervoxels
        if nVertices > 128:
            h5file.create_dataset(dsetpath + '/vertices', data=vertices, dtype=self.VERTEX_DTYPE,
                compression='gzip',compression_opts=self.HDF5_CLVL,shuffle=True,fletcher32=True)
        else:
            h5file.create_dataset(dsetpath + '/vertices', data=vertices, dtype=self.VERTEX_DTYPE)
        if nFaces > 256:
            h5file.create_dataset(dsetpath + '/faces', data=faces, dtype=self.FACE_DTYPE,
                compression='gzip',compression_opts=self.HDF5_CLVL,shuffle=True,fletcher32=True)
        else:
            h5file.create_dataset(dsetpath + '/faces', data=faces, dtype=self.FACE_DTYPE)
        dset = h5file[self.dataset_root][str_seed]['vertices']
        dset.attrs.create('nVoxels',nVoxels)
        if surface_area is not None: dset.attrs.create('surface_area',surface_area)
        beg = np.array([int(x) for x in beg*self.vertex_divisor], dtype=self.BOUNDS_DTYPE)
        end = np.array([int(x) for x in end*self.vertex_divisor], dtype=self.BOUNDS_DTYPE)
        dset.attrs.create('bounds_beg',beg); dset.attrs.create('bounds_end',end)

    def writeMeta(self, h5file):
        # use seed 0 (0 is always background) to store global attributes
        str_seed = ('%08d' % 0); dsetpath = self.dataset_root + '/' + str_seed
        #if 'faces' in h5file[dataset_root][str_seed]: del h5file[dataset_root][str_seed]['faces']
        h5file.create_dataset(dsetpath + '/faces', data=np.zeros((0,1),dtype=self.FACE_DTYPE),dtype=self.FACE_DTYPE)
        dset = h5file[self.dataset_root][str_seed]['faces']
        for v in self.save_params: dset.attrs.create(v, getattr(self, v))

    # for merging separate mesh files into a single hdf5 (assumes meshes are for non-overlapping volumes)
    def mergeMeshInfiles(self):
        mesh_infiles = glob.glob(os.path.join(self.merge_mesh_path, '*.h5')); nFiles = len(mesh_infiles);
        if self.dpLabelMesher_verbose: print('Merging %d mesh files' % (nFiles,))

        # use seed 0 (0 is always background) to store global attributes
        h5outfile = h5py.File(self.mesh_outfile, 'w');
        self.writeMeta(h5outfile)
        dsetout_root = h5outfile[self.dataset_root]

        # open all the hdf5 files to be merged
        h5files = nFiles*[None]; dset_roots = nFiles*[None]
        for i, mesh_infile in zip(range(nFiles), mesh_infiles):
            h5files[i] = h5py.File(mesh_infile, 'r'); dset_roots[i] = h5files[i][self.dataset_root]

        if self.dpLabelMesher_verbose:
            tloop = time.time(); t = time.time()
        for seed in range(1,self.nlabels+1):
            if self.dpLabelMesher_verbose and (seed-1) % self.print_every == 0 and (seed-1) > 0:
                print('seed : %d / %d' % (seed,self.nlabels))
            str_seed = ('%08d' % seed)
            for i in range(nFiles):
                #h5file = h5py.File(mesh_infile, 'r'); dset_root = h5file[self.dataset_root]
                dset_root = dset_roots[i]
                if str_seed in dset_root and 'vertices' in dset_root[str_seed]:
                    vertices = np.empty_like(dset_root[str_seed]['vertices'])
                    faces = np.empty_like(dset_root[str_seed]['faces'])
                    dset_root[str_seed]['vertices'].read_direct(vertices)
                    dset_root[str_seed]['faces'].read_direct(faces)
                    if str_seed in dsetout_root and 'vertices' in dsetout_root[str_seed]:
                        cvertices = np.empty_like(dsetout_root[str_seed]['vertices'])
                        cfaces = np.empty_like(dsetout_root[str_seed]['faces'])
                        dsetout_root[str_seed]['vertices'].read_direct(cvertices)
                        dsetout_root[str_seed]['faces'].read_direct(cfaces)

                        # concatenate the meshes
                        faces += cvertices.shape[0]; faces = np.vstack((cfaces, faces))
                        nVoxels = dset_root[str_seed]['vertices'].attrs['nVoxels'] + \
                            dsetout_root[str_seed]['vertices'].attrs['nVoxels']
                        beg = np.min(np.stack((dset_root[str_seed]['vertices'].attrs['bounds_beg'],
                                                dsetout_root[str_seed]['vertices'].attrs['bounds_beg']), axis=1),axis=1)
                        end = np.max(np.stack((dset_root[str_seed]['vertices'].attrs['bounds_end'],
                                                dsetout_root[str_seed]['vertices'].attrs['bounds_end']), axis=1),axis=1)
                        vertices += (dset_root[str_seed]['vertices'].attrs['bounds_beg'] - beg)
                        cvertices += (dsetout_root[str_seed]['vertices'].attrs['bounds_beg'] - beg)
                        vertices = np.vstack((cvertices, vertices))

                        del dsetout_root[str_seed]
                        self.writeData(h5outfile, beg, end, str_seed, faces, vertices, nVoxels)
                    else:
                        self.writeData(h5outfile, dset_root[str_seed]['vertices'].attrs['bounds_beg'],
                                       dset_root[str_seed]['vertices'].attrs['bounds_end'], str_seed,
                                       faces, vertices, dset_root[str_seed]['vertices'].attrs['nVoxels'])
                #h5file.close()
            if self.dpLabelMesher_verbose and (seed-1) % self.print_every == 0 and (seed-1) > 0:
                print('\tdone in %.3f s' % (time.time() - t,)); t = time.time()

        # close all the hdf5 files
        for i in range(nFiles): h5files[i].close()
        h5outfile.close()

        if self.dpLabelMesher_verbose: print('Total ellapsed time merging %.3f s' % (time.time() - tloop,))

    # these routines are for reading previously generated mesh hdf5 files and plotting/writing out statistics
    def readMeshInfiles(self):
        from matplotlib import pylab as pl
        import matplotlib as plt

        nFiles = len(self.mesh_infiles);
        mesh_info = [None]*nFiles; reduce_fracs = np.zeros((nFiles,))
        file_sizes = np.zeros((nFiles,),dtype=np.uint64)
        nlabels = np.zeros((nFiles,), dtype=np.uint64)
        nVertices = np.zeros((nFiles,), dtype=np.uint64)
        nFaces = np.zeros((nFiles,), dtype=np.uint64)
        comp_bytes = np.zeros((nFiles,),dtype=np.uint64)
        bin_bytes = np.zeros((nFiles,),dtype=np.uint64)
        for i, infile in zip(range(nFiles), self.mesh_infiles):
            mesh_info[i] = self.readMeshInfile(infile)
            reduce_fracs[i] = mesh_info[i]['reduce_frac']
            file_sizes[i] = os.path.getsize(infile)
            nVertices[i] = mesh_info[i]['nVertices'].sum(dtype=np.uint64)
            nFaces[i] = mesh_info[i]['nFaces'].sum(dtype=np.uint64)
            bin_bytes[i] += (3*nVertices[i]*mesh_info[i]['dtypeVertices'].itemsize + \
                3*nFaces[i]*mesh_info[i]['dtypeFaces'].itemsize)
            comp_bytes[i] += (mesh_info[i]['storageSizeVertices'].sum(dtype=np.uint64) + \
                mesh_info[i]['storageSizeFaces'].sum(dtype=np.uint64))
            nlabels[i] = mesh_info[i]['nlabels']

        # print info to console
        scale = np.array(2**20,dtype=np.double); scale_str = 'MB'
        for i, infile in zip(range(nFiles), self.mesh_infiles):
            print('For reduce frac %g:' % (reduce_fracs[i],))
            print('\tvertices/faces %d/%d size uncompressed %g %s, compressed %g %s' % (nVertices[i], nFaces[i],
                bin_bytes[i]/scale,scale_str,comp_bytes[i]/scale,scale_str))
            print('\tratio %g, files size %g %s, svox %d, overhead / svox %g %s' % (comp_bytes/bin_bytes,
                file_sizes[i]/scale,scale_str, nlabels[i], (file_sizes[i] - comp_bytes[i])/nlabels[i]/scale, scale_str))

        # the plot is dead, long live the plot, huzzah!
        pl.figure(1);
        pl.subplot(1,2,1)
        pl.plot(reduce_fracs,comp_bytes/scale,'x--')
        pl.plot(reduce_fracs,bin_bytes/scale,'x--')
        pl.legend(['compressed', 'uncompressed'])
        pl.ylabel('meshing size (%s)' % (scale_str,))
        pl.xlabel('reduce fraction')
        pl.title('%d supervoxels' % (nlabels[0], ))

        pl.subplot(1,2,2);
        m = plt.markers
        clrs = ['r','g','b','m','c','y','k']; markers = ['x','+',m.CARETLEFT,m.CARETRIGHT,m.CARETUP,m.CARETDOWN,'*']
        for i,frac in zip(range(nFiles),reduce_fracs):
            size = 3*mesh_info[i]['nVertices']*mesh_info[i]['dtypeVertices'].itemsize+\
                3*mesh_info[i]['nFaces']*mesh_info[i]['dtypeFaces'].itemsize
            pl.scatter(np.log10(mesh_info[i]['nVoxels']), np.log10(size), s=8, c=clrs[i], alpha=0.5, marker=markers[i])
        pl.xlabel('voxels (log count)'); pl.ylabel('meshing size (log bytes)'); pl.title('binary size')
        pl.legend(reduce_fracs)

        pl.show()

    def readMeshInfile(self, mesh_infile):
        h5file = h5py.File(mesh_infile, 'r')
        dset_root = h5file[self.dataset_root]

        # read meta-data in seed 0
        str_seed = ('%08d' % 0)
        nlabels = h5file[self.dataset_root][str_seed]['faces'].attrs['nlabels']
        #vertex_divisor = h5file[self.dataset_root][str_seed]['faces'].attrs['vertex_divisor']
        reduce_frac = h5file[self.dataset_root][str_seed]['faces'].attrs['reduce_frac']

        nVoxels = -np.ones((nlabels+1,),dtype=np.int64)
        nFaces = -np.ones((nlabels+1,),dtype=np.int64)
        nVertices = -np.ones((nlabels+1,),dtype=np.int64)
        storageSizeVertices = -np.ones((nlabels+1,),dtype=np.int64)
        storageSizeFaces = -np.ones((nlabels+1,),dtype=np.int64)
        for k in dset_root:
            seed = int(k)
            if seed < 1: continue
            nVertices[seed] = dset_root[k]['vertices'].shape[0]
            nFaces[seed] = dset_root[k]['faces'].shape[0]
            nVoxels[seed] = dset_root[k]['vertices'].attrs['nVoxels']
            storageSizeVertices[seed] = dset_root[k]['vertices'].id.get_storage_size()
            storageSizeFaces[seed] = dset_root[k]['faces'].id.get_storage_size()

            dtypeVertices = dset_root[k]['vertices'].dtype
            dtypeFaces = dset_root[k]['faces'].dtype
        h5file.close()

        sel = (nVoxels > 0)
        nVoxels = nVoxels[sel]; nFaces = nFaces[sel]; nVertices = nVertices[sel]
        storageSizeVertices = storageSizeVertices[sel]; storageSizeFaces = storageSizeFaces[sel]

        return_vars = ['nVoxels', 'nFaces', 'nVertices', 'storageSizeVertices', 'storageSizeFaces',
                       'dtypeVertices', 'dtypeFaces', 'nlabels', 'reduce_frac']
        return_dict = {}; cur = locals()
        for v in return_vars: return_dict[v] = cur[v]
        return return_dict

    def showMergeMesh(self):

        # for viewing subsets of the objects/skeletons in the annotation file.
        # use object / skeleton < 0 to disable viewing objects / skeletons entirely.
        nobjs = len(self.merge_objects); nskels = len(self.skeletons)

        # incase there is no meshing file and just using this to view skeletons
        if nobjs > 0 and self.merge_objects[0] < 0:
            h5file = None
        else:
            # xxx - add support for multiple mesh files somehow (need knossos support likely with supercube ids)
            h5file = h5py.File(self.mesh_infiles[0], 'r'); dset_root = h5file[self.dataset_root]

            # read meta-data in seed 0
            str_seed = ('%08d' % 0)
            #nlabels = h5file[self.dataset_root][str_seed]['faces'].attrs['nlabels']
            vertex_divisor = dset_root[str_seed]['faces'].attrs['vertex_divisor']
            #reduce_frac = dset_root[str_seed]['faces'].attrs['reduce_frac']

            # xxx - need to fix this not getting copied during merge
            # make skeleton rendering consistent with h5 file
            #self.set_voxel_scale = dset_root[str_seed]['faces'].attrs['set_voxel_scale']

        # this loop continues until ctrl-C, automatically updates to next annotation file
        loop = True
        while loop:
            loop = not self.show_all
            if not self.show_all:
                # get the nml skeleton file and mergelist out of the zipped knossos annotation file
                zf = zipfile.ZipFile(self.annotation_file, mode='r');
                inmerge = zf.read('mergelist.txt'); inskel = zf.read('annotation.xml');
                zf.close()
                # read the merge list
                merge_list = inmerge.decode("utf-8").split('\n'); nlines = len(merge_list); n = nlines // 4
    
                # read the skeletons
                info, meta, commentsString = knossos_read_nml(krk_contents=inskel.decode("utf-8")); m = len(info)
            else:
                n = len(dset_root) - 1; m = 0

            # allocate renderer for this pass
            renderer = vtk.vtkRenderer()

            # reallocate everything for the meshes
            self.faces = n * [None]; self.vertices = n * [None]
            self.allPolyData = n * [None]; self.allMappers = n * [None]; self.allActors = n * [None]
            self.nFaces = np.zeros((n,), dtype=np.uint64); self.nVertices = np.zeros((n,), dtype=np.uint64);
            self.nSVoxels = np.zeros((n,), dtype=np.uint64); self.nVoxels = np.zeros((n,), dtype=np.uint64)

            # iterated over objects to be meshed, render all the meshes in the mergelist for each object
            obj_cnt = 0; obj_sel = np.zeros((n,),dtype=np.bool)
            for i in range(n):
                self.allPolyData[i] = vtk.vtkAppendPolyData()

                if not self.show_all:
                    # Object_ID, ToDo_flag, Immutability_flag, Supervoxel_IDs, '\n'
                    tomerge = merge_list[i*4].split(' ')
                    cobj = int(tomerge[0]); tomerge = tomerge[3:]
                else:
                    cobj = i+1; tomerge = np.array([cobj])
                    
                if nobjs > 0 and cobj not in self.merge_objects: continue
                obj_cnt += 1; obj_sel[i] = 1; cmap_cnt = (obj_cnt - 1) % self.cmap.shape[0]

                nsvox = len(tomerge); self.nSVoxels[i] = nsvox
                self.faces[i] = nsvox * [None]; self.vertices[i] = nsvox * [None]
                for j in range(nsvox):
                    str_seed = ('%08d' % int(tomerge[j]))
                    cvertices = np.empty_like(dset_root[str_seed]['vertices'])
                    cfaces = np.empty_like(dset_root[str_seed]['faces'])
                    dset_root[str_seed]['vertices'].read_direct(cvertices)
                    dset_root[str_seed]['faces'].read_direct(cfaces)
                    nvertices = cvertices.shape[0]; nfaces = cfaces.shape[0]

                    # vertices are stored as fixed-point
                    cvertices = cvertices.astype(np.double) / vertex_divisor
                    cvertices += dset_root[str_seed]['vertices'].attrs['bounds_beg']
                    # vtk needs unstructured grid preceded with number of points in each cell
                    cfaces = np.hstack((3*np.ones((nfaces, 1),dtype=cfaces.dtype), cfaces))

                    # need to keep references around apparently to avoid segfault
                     # https://github.com/Kitware/VTK/blob/master/Wrapping/Python/vtk/util/numpy_support.py
                    self.vertices[i][j] = cvertices; self.faces[i][j] = cfaces

                    # just for printing to console for each object
                    self.nFaces[i] += nfaces; self.nVertices[i] += nvertices
                    self.nVoxels[i] += dset_root[str_seed]['vertices'].attrs['nVoxels']

                    # create and append poly data
                    # http://www.vtk.org/Wiki/VTK/Examples/Python/GeometricObjects/Display/Polygon
                    points = vtk.vtkPoints(); points.SetData(nps.numpy_to_vtk(cvertices))

                    # http://stackoverflow.com/questions/20146421/how-to-convert-a-mesh-to-vtk-format/20146620#20146620
                    cells = vtk.vtkCellArray()
                    cells.SetCells(nfaces, nps.numpy_to_vtk(cfaces, array_type=vtk.vtkIdTypeArray().GetDataType()))

                    polyData = vtk.vtkPolyData(); polyData.SetPoints(points); polyData.SetPolys(cells)

                    # use appendpolydata to render multiple supervoxels per object
                    self.allPolyData[i].AddInputData(polyData)

                self.allMappers[i] = vtk.vtkPolyDataMapper()
                self.allMappers[i].SetInputConnection(self.allPolyData[i].GetOutputPort())
                ##mapper.SetLookupTable(self.colorMap)  # xxx - couldn't get this to work
                self.allActors[i] = vtk.vtkActor()
                self.allActors[i].SetMapper(self.allMappers[i])
                self.allActors[i].GetProperty().SetColor(self.cmap[cmap_cnt,0],self.cmap[cmap_cnt,1],
                              self.cmap[cmap_cnt,2])
                self.allActors[i].GetProperty().SetOpacity(self.opacity)
                renderer.AddActor(self.allActors[i])

            # reallocate everything for the skeletons
            self.skel_faces = m * [None]; self.skel_vertices = m * [None]
            self.skel_polyData = m * [None]; self.skel_allMappers = m * [None]; self.skel_allActors = m * [None]
            self.skel_nFaces = np.zeros((m,), dtype=np.uint64); self.skel_nVertices = np.zeros((m,), dtype=np.uint64);
            self.skel_lblStrings = m * [None]; self.skel_lblFilter = m * [None]
            self.skel_lblMappers = m * [None]; self.skel_lblActors = m * [None]

            # iterate over skeletons to be rendered
            skel_cnt = 0; skel_sel = np.zeros((m,),dtype=np.bool)
            for i in range(m):
                if nskels > 0 and info[i]['thingID'] not in self.skeletons: continue
                skel_cnt += 1; skel_sel[i] = 1; cmap_cnt = (skel_cnt - 1) % self.cmap.shape[0]

                cvertices = info[i]['nodes'][:,:3].copy(order='C'); cfaces = info[i]['edges']
                nvertices = cvertices.shape[0]; nfaces = cfaces.shape[0]

                # vertices are stored in nml as dataset voxel coordinates
                if self.set_voxel_scale: cvertices = cvertices * self.data_attrs['scale']

                # vtk needs unstructured grid preceded with number of points in each cell
                cfaces = np.hstack((2*np.ones((nfaces, 1),dtype=cfaces.dtype), cfaces))

                # need to keep references around apparently to avoid segfault
                # https://github.com/Kitware/VTK/blob/master/Wrapping/Python/vtk/util/numpy_support.py
                self.skel_vertices[i] = cvertices; self.skel_faces[i] = cfaces

                # just for printing to console for each skeleton
                self.skel_nFaces[i] += nfaces; self.skel_nVertices[i] += nvertices

                # create and append poly data
                # http://www.vtk.org/Wiki/VTK/Examples/Python/GeometricObjects/Display/Polygon
                points = vtk.vtkPoints(); points.SetData(nps.numpy_to_vtk(cvertices))

                # http://stackoverflow.com/questions/20146421/how-to-convert-a-mesh-to-vtk-format/20146620#20146620
                cells = vtk.vtkCellArray()
                cells.SetCells(nfaces, nps.numpy_to_vtk(cfaces, array_type=vtk.vtkIdTypeArray().GetDataType()))

                # vtkPolyData also works for lines, just have to use setlines instead of setpolys
                polyData = vtk.vtkPolyData(); polyData.SetPoints(points); polyData.SetLines(cells)

                self.skel_polyData[i] = polyData
                self.skel_allMappers[i] = vtk.vtkPolyDataMapper()
                self.skel_allMappers[i].SetInputData(self.skel_polyData[i])
                self.skel_allMappers[i].ScalarVisibilityOff()
                self.skel_allActors[i] = vtk.vtkActor()
                self.skel_allActors[i].SetMapper(self.skel_allMappers[i])
                self.skel_allActors[i].GetProperty().SetColor(self.cmap[cmap_cnt,0],self.cmap[cmap_cnt,1],
                    self.cmap[cmap_cnt,2])
                renderer.AddActor(self.skel_allActors[i])

                if self.show_node_ids:
                    # for adding node id labels to skeleton nodes
                    # http://www.vtk.org/Wiki/VTK/Examples/Cxx/Visualization/LabelPlacementMapper
                    # xxx - couldn't find a potentially more efficient way to convert to vtkStringArray
                    #   neither setting the labels as an integer array or attempting to convert to string worked:
                    #labels = nps.numpy_to_vtk(info[i]['nodes'][:,3].copy(order='C'))
                    #labels = nps.numpy_to_vtk(info[i]['nodes'][:,3].copy(order='C'),
                    #                          array_type=vtk.vtkStringArray().GetDataType());
                    labels = vtk.vtkStringArray(); labels.SetNumberOfValues(nvertices)
                    for j in range(nvertices):
                        labels.SetValue(j, str(info[i]['nodes'][j,3]))
                    labels.SetName('NodeIDs')
                    polyData.GetPointData().AddArray(labels)

                    # Generate the label hierarchy.
                    pointSetToLabelHierarchyFilter = vtk.vtkPointSetToLabelHierarchy()
                    pointSetToLabelHierarchyFilter.SetInputData(polyData)
                    pointSetToLabelHierarchyFilter.SetLabelArrayName('NodeIDs')
                    pointSetToLabelHierarchyFilter.Update()
                    # Create a mapper and actor for the labels.
                    labelMapper = vtk.vtkLabelPlacementMapper()
                    labelMapper.SetInputConnection(pointSetToLabelHierarchyFilter.GetOutputPort())
                    labelActor = vtk.vtkActor2D()
                    labelActor.SetMapper(labelMapper)

                    self.skel_lblStrings[i] = labels; self.skel_lblFilter[i] = pointSetToLabelHierarchyFilter
                    self.skel_lblMappers[i] = labelMapper; self.skel_lblActors[i] = labelActor
                    renderer.AddActor(self.skel_lblActors[i])

            print('For %d objects' % (n if nobjs==0 else obj_cnt,))
            print('\tnSuperVoxels %s' % (np.array_str(self.nSVoxels[obj_sel])[1:-1]))
            print('\tnVertices %s' % (np.array_str(self.nVertices[obj_sel])[1:-1]))
            print('\tnFaces %s' % (np.array_str(self.nFaces[obj_sel])[1:-1]))
            print('\tnVoxels %s' % (np.array_str(self.nVoxels[obj_sel])[1:-1]))
            print('\tworst case cube nVertices %s' % (np.array_str(8*self.nVoxels[obj_sel])[1:-1]))
            print('\tworst case cube nFaces %s' % (np.array_str(6*self.nVoxels[obj_sel])[1:-1]))
            print('For %d skeletons' % (m if nskels==0 else skel_cnt,))
            print('\tnNodes %s' % (np.array_str(self.skel_nVertices[skel_sel])[1:-1]))
            print('\tnEdges %s' % (np.array_str(self.skel_nFaces[skel_sel])[1:-1]))
            dpLabelMesher.vtkShow(renderer=renderer)

            parts = re.match('(.+?)\.(\d+?)\.(.+)', self.annotation_file).groups()
            if len(parts) == 3:
                try:
                    self.annotation_file = '%s.%03d.%s' % (parts[0],int(parts[1])+1,parts[2])
                except ValueError:
                    pass
            fn = input('Enter next annotation file [%s]: ' % self.annotation_file).strip()
            if len(fn) > 0: self.annotation_file = fn

        if h5file is not None: h5file.close()

    def mergeMesh(self):

        nobjs = len(self.merge_objects); #nskels = len(self.skeletons)

        # incase there is no meshing file and just using this to view skeletons
        if nobjs > 0 and self.merge_objects[0] < 0:
            h5file = None
        else:
        # xxx - add support for multiple mesh files somehow (need knossos support likely with supercube ids)
            h5file = h5py.File(self.mesh_infiles[0], 'r'); dset_root = h5file[self.dataset_root]

        # read meta-data in seed 0
        #str_seed = ('%08d' % 0)
        #nlabels = h5file[self.dataset_root][str_seed]['faces'].attrs['nlabels']
        #vertex_divisor = dset_root[str_seed]['faces'].attrs['vertex_divisor']

        # get the nml skeleton file and mergelist out of the zipped knossos annotation fil
        zf = zipfile.ZipFile(self.annotation_file, mode='r');
        inmerge = zf.read('mergelist.txt'); #inskel = zf.read('annotation.xml');
        zf.close()

        # read the merge list
        merge_list = inmerge.decode("utf-8").split('\n'); nlines = len(merge_list); n = nlines // 4

        # allocate renderer for this pass
        renderer = vtk.vtkRenderer()

        # reallocate everything for the meshes
        self.faces = n * [None]; self.vertices = n * [None]
        self.allPolyData = n * [None]; self.allMappers = n * [None]; self.allActors = n * [None]
        self.nFaces = np.zeros((n,), dtype=np.uint64); self.nVertices = np.zeros((n,), dtype=np.uint64);
        self.nSVoxels = np.zeros((n,), dtype=np.uint64);

        obj_cnt = 0; obj_sel = np.zeros((n,),dtype=np.bool)

        if self.write_ply:
           plyWriter = vtk.vtkPLYWriter()
           plyWriter.SetFileName('1.ply')
        elif self.write_hdf5:
           h5filewrite = h5py.File(self.mesh_outfile, 'w')


        for i in range(n):

           # Object_ID, ToDo_flag, Immutability_flag, Supervoxel_IDs, '\n'
           tomerge = merge_list[i*4].split(' ')
           cobj = int(tomerge[0])
           if nobjs > 0 and cobj not in self.merge_objects: continue
           obj_cnt += 1; obj_sel[i] = 1
           tomerge = tomerge[3:]
           self.scale = self.data_attrs['scale']

           nsvox = len(tomerge);
           bound_beg = nsvox * [None];
           bound_end = nsvox * [None];
           self.faces[i] = nsvox * [None]; self.vertices[i] = nsvox *[None]
           self.allPolyData[i] = vtk.vtkAppendPolyData()
           self.allMappers[i] = vtk.vtkPolyDataMapper()

           for j in range(nsvox):
              tr_seed = ('%08d' % int(tomerge[j]))
              bound_beg[j] = np.array(dset_root[tr_seed]['vertices'].attrs['bounds_beg'],dtype=np.float64)
              bound_end[j] = np.array(dset_root[tr_seed]['vertices'].attrs['bounds_end'],dtype=np.float64)
              #save the bounds in nm scale
              beg = bound_beg[j]
              end = bound_end[j]
              ## convert to right scale to calculate appropriate chunk and offset
              bound_beg[j] = bound_beg[j]/self.vertex_divisor
              bound_end[j] = bound_end[j]/self.vertex_divisor
              bound_beg[j] = bound_beg[j]/self.scale
              bound_end[j] = bound_end[j]/self.scale
              bound_beg[j] = np.round(bound_beg[j]).astype(np.uint32)
              bound_end[j] = np.round(bound_end[j]).astype(np.uint32)
              self.chunk = bound_beg[j]//self.chunksize
              self.offset = bound_beg[j]%self.chunksize
              self.size = bound_end[j] - bound_beg[j]

              #get the corresponding datacube according to bounds
              self.inith5()
              self.readCubeToBuffers();

              #get the data cube
              if self.legacy_transpose: cube = self.data_cube.transpose((1,0,2))
              else: cube = self.data_cube


              # other inits to smooth and binarize the data
              if self.do_smooth: W = np.ones(self.smooth, dtype=self.PDTYPE) / self.smooth.prod()
              bin_labels = (cube == int(tr_seed)).astype(self.PDTYPE)

              if self.do_smooth:
                crpdplsSm = filters.convolve(bin_labels, W, mode='reflect', cval=0.0, origin=0)
                # if smoothing results in nothing above contour level, use original without smoothing
                if (crpdplsSm > self.contour_lvl).any():
                    del bin_labels; bin_labels = crpdplsSm; del crpdplsSm
                #if self.doplots: showImgData(np.squeeze(crpdpls[:,:,crpdpls.shape[2]/2]),'slice')'''
              self.crpdpls = bin_labels
              #mesh the supervoxel to obtain vertices and faces
              verts,face = self.vtkMesh(bound_beg[j],self.size)

              self.vertices[i][j] = verts
              self.faces[i][j] = face

              nfaces = face.shape[0]
              #nvertices = verts.shape[0]


              #self.nFaces[i] += nfaces
              #self.nVertices[i] += nvertices

              self.nVoxels = j
              if self.write_hdf5:
                 self.writeData(h5filewrite, beg, end, tr_seed,face,verts,self.nVoxels)

              #add the upper bound of the cube to get appropriate context of the entire mesh
              verts = verts + beg



              #vtk needs unstructures grid preceded with number of points in each cell
              face = np.hstack((3*np.ones((nfaces, 1),dtype=face.dtype), face))

              self.vertices[i][j] = verts
              self.faces[i][j] = face

              polyData = vtk.vtkPolyData()
              points = vtk.vtkPoints(); points.SetData(nps.numpy_to_vtk(verts))

              # http://stackoverflow.com/questions/20146421/how-to-convert-a-mesh-to-vtk-format/20146620#20146620
              cells = vtk.vtkCellArray()
              cells.SetCells(nfaces, nps.numpy_to_vtk(face, array_type=vtk.vtkIdTypeArray().GetDataType()))

              polyData.SetPoints(points); polyData.SetPolys(cells)

              self.allPolyData[i].AddInputData(polyData)

           #vtk pipeline
           if self.write_ply:
             plyWriter.SetInputConnection(self.allPolyData[i].GetOutputPort())
             plyWriter.Write()
           elif self.write_hdf5:
             self.writeMeta(h5filewrite)
             h5filewrite.close()

           self.allMappers[i].SetInputConnection(self.allPolyData[i].GetOutputPort())
           self.allActors[i] = vtk.vtkActor()
           self.allActors[i].SetMapper(self.allMappers[i])
           self.allActors[i].GetProperty().SetColor(self.cmap[obj_cnt-1,0],self.cmap[obj_cnt-1,1],self.cmap[obj_cnt-1,2])
           self.allActors[i].GetProperty().SetOpacity(self.opacity)
           renderer.AddActor(self.allActors[i])
           dpLabelMesher.vtkShow(renderer=renderer)


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
        arg_str += ' --dataset ' + dataset
        if verbose: arg_str += ' --dpLabelMesher-verbose '
        #if verbose: print(arg_str)
        args = parser.parse_args(arg_str.split())
        stm = cls(args)
        return stm

    @staticmethod
    def vtkShow(mapper=None, renderer=None):

        if renderer is None:
            # open a window and display the data specified by mapper
            # need an actor and a renderer to display data
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            renderer = vtk.vtkRenderer()
            renderer.AddActor(actor)

        #renderer.SetBackground(1.0, 1.0, 1.0)
        renderer.SetBackground(0.0, 0.0, 0.0)

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
        renderInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
        renderInteractor.SetRenderWindow(renderWin)
        renderWin.SetSize(800, 800)
        renderInteractor.Initialize()
        renderWin.Render()
        renderInteractor.Start()

    @staticmethod
    def addArgs(p):
        # adds arguments required for this object to specified ArgumentParser object
        dpWriteh5.addArgs(p)
        p.add_argument('--mesh-outfile', nargs=1, type=str, default='', help='Output label mesh file')
        #p.add_argument('--mesh-outfile-stl', action='store_true', help='Output mesh ply file')
        p.add_argument('--mesh-infiles', nargs='*', type=str, default='',
                       help='Input label mesh file (calculate stats / show plots only)')
        p.add_argument('--merge-mesh-path', nargs=1, type=str, default='',
                       help='Input path for mesh files to be merged')
        p.add_argument('--annotation-file', nargs=1, type=str, default='',
                       help='Input annotation file from knossos (show merged meshes)')
        p.add_argument('--annotation-file-mesh', action='store_true',
                       help='Mesh only supervoxels from the knossos annotation file')
        p.add_argument('--show-all', action='store_true', help='Show all meshes in mesh_infiles')
        p.add_argument('--write_hdf5', action='store_true',
                       help='Write separate mesh file for annotated meshes')
        p.add_argument('--dataset-root', nargs=1, type=str, default='meshes', help='Top level for hdf5 outfile')
        p.add_argument('--write_ply', action='store_true',
                       help='Write separate ply file for annoatated meshes')
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
        p.add_argument('--seed-range', nargs=2, type=int, default=[-1,-1], metavar=('BEG', 'END'),
            help='Subset of seeds to process (< 0 for beg/end)')
        p.add_argument('--no-decimatePro', action='store_false', dest='decimatePro',
            help='Do not use decimate pro from vtk for meshing (use quadric clustering instead)')
        #p.add_argument('--decimatePro', action='store_true', dest='decimatePro',
        #    help='Use decimate pro from vtk for meshing (default quadric clustering)')
        p.add_argument('--set-voxel-scale', action='store_true', dest='set_voxel_scale',
            help='Use the voxel scale to set the data spacing to vtk (vertices in nm)')
        p.add_argument('--doplots', action='store_true', help='Debugging plotting enabled for each supervoxel')
        p.add_argument('--print-every', nargs=1, type=int, default=[500], metavar=('ITER'),
                       help='Modulo for print update')
        p.add_argument('--merge-objects', nargs='*', type=int, default=[], metavar=('OBJS'),
                       help='Which objects to display (for annotation-file mode)')
        p.add_argument('--skeletons', nargs='*', type=int, default=[], metavar=('SKELS'),
                       help='Which skeletons to display (for annotation-file mode)')
        p.add_argument('--lut-file', nargs=1, type=str, default='', help='Specify colormap (for annotation-file mode')
        p.add_argument('--show-node-ids', action='store_true', help='Show node id strings (for annotation-file mode)')
        p.add_argument('--opacity', nargs=1, type=float, default=[1.0], help='Mesh opacity (for annotation-file mode)')
        p.add_argument('--dpLabelMesher-verbose', action='store_true', help='Debugging output for dpLabelMesher')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create surface meshes from hdf5 label data for Knossos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpLabelMesher.addArgs(parser)
    args = parser.parse_args()

    seg2mesh = dpLabelMesher(args)
    if seg2mesh.annotation_file or seg2mesh.show_all:
        if seg2mesh.annotation_file_mesh:
            # create or recreate meshes but only for supervoxels in merge list from knossos annotation file
            seg2mesh.mergeMesh()
        else:
            # visualize meshes from knossos annotation file
            seg2mesh.showMergeMesh()
    elif seg2mesh.merge_mesh_path:
        # merge meshes from multiple mesh files (necessary if superchunks have been stitched)
        seg2mesh.mergeMeshInfiles()
    elif len(seg2mesh.mesh_infiles) > 0:
        # print out combined stats for mesh files over multiple superchunks      
        seg2mesh.readMeshInfiles()
    else:
        # standard mode, mesh all supervoxels in a single labeled volume (superchunk in one hdf5 label file)
        seg2mesh.readCubeToBuffers()
        seg2mesh.procData()
        seg2mesh.writeMeshOutfile()
