#!/usr/bin/env python

'''
Python object for reading labrainth data stored in large hdf5 files based on cube index.
Code is based on hdf5 parsing for EM data backend for cuda convnets (parseEMdata.py)
Original shared code was forked for use with backend only as toolset for manipulating hdf5 after voxel classifier.
pwatkins, created Jan 7, 2015
    forked Dec 1, 2015

Example invokations:

dpLoadh5.py --srcfile ~/Data/mouse_retina/k0725_27cubes/k0725_mag1_8x8x5chunks_128x128x128chunksize_Forder.h5 --dataset data --chunk 2 2 0 --dpLoadh5-verbose

dpLoadh5.py --srcfile /Data/pwatkins/full_datasets/newestECSall/20151001/huge_supervoxels.h5 --data-type uint32 --dataset labels --chunk 17 19 2 --subgroups with_background 0.99950000 --outraw '/Data/pwatkins/tmp/supervoxels.nrrd' --dpLoadh5-verbose

dpLoadh5.py --srcfile /Data/big_datasets/M0007_33_39x35x7chunks_Forder.h5 --size 128 128 128 --dataset data_mag1 --chunk 17 19 2 --outraw ~/Downloads/outraw.nrrd --dpLoadh5-verbose

'''

import h5py
import numpy as np
import argparse
import time
import os
from skimage.segmentation import relabel_sequential

class dpLoadh5(object):

    ND = 3      # this representation is soley meant for 3d volumes
    # this reslicing order is chosen so that same indices can be used for order and for un-order
    RESLICES = {
        'zyx' : [2,1,0],    # zyx, z is x after reslice
        'xzy' : [0,2,1],    # xzy, z is y after reslice
        'xyz' : [0,1,2],    # xyz, z is z after reslice
    }

    def __init__(self, args):
        # save command line arguments from argparse, see definitions in main or run with --help
        for k, v in vars(args).items(): 
            # do not override any values that are already set as a method of allowing inherited classes to specify
            if hasattr(self,k): continue
            if type(v) is list and k not in ['subgroups','subgroups_out']: 
                if len(v)==1:
                    setattr(self,k,v[0])  # save single element lists as first element
                elif type(v[0]) is int:   # convert the sizes and offsets to numpy arrays
                    setattr(self,k,np.array(v,dtype=np.int32))
                else:
                    setattr(self,k,v)   # store other list types as usual (floats)
            else:
                setattr(self,k,v)
                
        # method of forcing data-type from inherited classes (xxx - yeah, yeah, yeah)
        if not hasattr(self,'default_data_type'): 
            self.default_data_type = self.data_type if self.data_type else np.uint8
                
        self.inith5()

    def inith5(self):

        # Options / Inits
        
        # old throwbacks, kept just incase, not doing these in this context so not revealed as options
        self.origin_chunk_inds = False; self.image_size = 0; self.nzslices = 1
        
        # use the string input to decide on the numpy data type to load into
        if self.data_type and isinstance(self.data_type, str): self.data_type = eval('np.' + self.data_type)

        # read attributes from the hdf5 first, possibly need for inits
        self.isFile = False; self.isDataset = False
        if os.path.isfile(self.srcfile):
            hdf = h5py.File(self.srcfile,'r');
            self.isFile = True; self.data_attrs = {}
            self.dset, self.group, self.dsetpath = self.getDataset(hdf)
            if self.dset:
                self.isDataset = True
                for name,value in self.dset.attrs.items(): self.data_attrs[name] = value
                self.data_attrs['chunks'] = self.dset.chunks    # xxx - where is this used again?
                self.datasize = np.array(self.dset.shape); self.chunksize = np.array(self.dset.chunks)
                if not self.hdf5_Corder: 
                    self.datasize = self.datasize[::-1]; self.chunksize = self.chunksize[::-1]
                if not self.data_type: self.data_type = self.dset.dtype
            elif not self.data_type:
                self.data_type = self.default_data_type
                    
            # xxx - this probably should be refactored at some point.
            # maybe a single dataset in all hdf5 files that contain the "global" information?
            # did this for now to just maintain compatibility.
            # purpose is that if loading from some subgroup, the attributes like scale for the global dataset are used.
            if 'scale' not in self.data_attrs:
                dset_found = None
                dsets = ['data', 'data_mag1', 'labels', 'voxel_type', 'probabilities', 'ICS']
                for dset in dsets: 
                    if dset in hdf: dset_found = dset; break
                if dset_found is not None:
                    for name,value in hdf[dset_found].attrs.items():
                        if name not in self.data_attrs: self.data_attrs[name] = value
                    
            hdf.close()
        elif not self.data_type:
            self.data_type = self.default_data_type

        # optionally use chunk size from attributes to get actual read size
        if self.size_in_chunks: self.size *= self.data_attrs['chunks']
        if self.size_from_hdf5: self.size = self.data_attrs['read_size']

        # the manner in which the zreslice is defined, define sort from data -> re-order and from re-order -> data.
        # only 3 options because data can be automatically augmented to transpose the first two dims (in each "z-slice")
        # these orders were chosen because the "unsort" is the same as the "sort" indexing, so re-order->data not needed
        self.zreslice_dim_ordering = self.RESLICES[self.dim_ordering]

        # immediately re-order any arguments that need it because of reslice. this prevents from having to do this on 
        #   command line, which ended up being annoying.
        # originally reading the hdf5 was done using arguments that were re-ordered on command line, so those needed
        #   during read are un-re-ordered (back to normal order) in readCubeToBuffers.
        self.size = self.size[self.zreslice_dim_ordering]   # size more intuitive re-order, un-re-order during load

        # inits that depend on re-ordering        
        self.ntotal_zslice = self.size[2] + self.nzslices - 1
        self.data_slice_size = (self.size[0] + self.image_size, self.size[1] + self.image_size, self.ntotal_zslice)

        # print out all initialized variables in verbose mode
        if self.dpLoadh5_verbose: print('dpLoadh5, verbose mode:\n'); print(vars(self))

    # added this to allow things to be read/written to subgroups in the hdf5 easily
    def getDataset(self, h5file):
        dset = h5file; dsetpath = ''; allgroups = self.subgroups + [self.dataset]; group = dset
        for i in range(len(allgroups)):
            group = dset
            if dset and allgroups[i] in dset: 
                dset = dset[allgroups[i]]
            else:
                dset = None
            dsetpath += ('/' + allgroups[i])
        return dset, group, dsetpath

    def readCubeToBuffers(self):
        if self.dpLoadh5_verbose: 
            print('dpLoadh5: Buffering data to memory')
            t = time.time()

        # xxx - might think of a better way to "reslice" the dimensions later, for now, here's the method:
        # read_direct requires the same size for the numpy array as in the hdf5 file. so if we're re-ordering the dims:
        #   (1) re-order the sizes to allocate here as if in original xyz order. 
        #   (2) re-order the dims and sizes used in the *slices_from_indices functions into original xyz order. 
        #       chunk indices are not changed.
        #   (3) at the end of this function re-order the data into the specified dim ordering
        #   (4) the rest of the packager is then blind to the reslice dimension ordering
        # NOTE ORIGINAL: chunk indices should be given in original hdf5 ordering.
        #   all other command line arguments should be given in the re-ordered ordering.
        #   the C/F order re-ordering needs to be done nested inside the reslice re-ordering
        # NEW NOTE: had the re-ordering of command line inputs for reslice done automatically, meaning all inputs on 
        #   command line should be given in original ordering, but they are re-ordered in re-slice order in init, so
        #   un-re-order here to go back to original ordering again (minimal overhead, done to reduce debug time).
        data_size = list(self.data_slice_size[i] for i in self.zreslice_dim_ordering)
        size = self.size[self.zreslice_dim_ordering]
        if self.dpLoadh5_verbose: print('data slice size ' + str(self.data_slice_size) + ' data size ' + str(data_size))
            
        # ulimately everything is accessed as C-order, but support loading from F-order hdf5 inputs.
        # h5py requires that for read_direct data must be C order and contiguous. this means F-order must be dealt with 
        #   "manually" here. for F-order the cube will be in C-order, but shaped like F-order, and then the view 
        #   transposed back to C-order so that it's transparent in the rest of the code.
        if self.hdf5_Corder: 
            self.data_cube = np.zeros(data_size, dtype=self.data_type, order='C')
        else: 
            self.data_cube = np.zeros(data_size[::-1], dtype=self.data_type, order='C')

        # slice out the data hdf
        hdf = h5py.File(self.srcfile,'r'); self.dset, self.group, self.dsetpath = self.getDataset(hdf)
        assert( self.dset )     # fail here if dataset does not exist in subgroups path
        ind = self.get_hdf_index_from_chunk_index(self.dset, self.chunk, self.offset)
        #print(ind, self.dset.shape)
        slc,slcd = self.get_data_slices_from_indices(ind, size, data_size)
        self.dset.read_direct(self.data_cube, slc, slcd)
        hdf.close()

        # the C/F order re-ordering needs to be done nested inside the reslice re-ordering
        if not self.hdf5_Corder: 
            self.data_cube = self.data_cube.transpose(2,1,0)

        # zreslice re-ordering, so data is in re-sliced order view outside of this function           
        self.data_cube = self.data_cube.transpose(self.zreslice_dim_ordering)
        if self.dpLoadh5_verbose:
            self.data_cube_min = self.data_cube.min(); self.data_cube_max = self.data_cube.max();
            print('\tloaded in %.4f s' % (time.time() - t))
            print('\tafter re-ordering data cube shape ' + str(self.data_cube.shape))
            print('\tmin ' + str(self.data_cube_min) + ' max ' + str(self.data_cube_max))
            #if self.data_type == np.uint16 or self.data_type == np.uint32:
            #    print('\tnunique ' + str(len(np.unique(self.data_cube))))

    def get_hdf_index_from_chunk_index(self, hdf_dataset, chunk_index, offset):
        if hdf_dataset:
            datasize = np.array(hdf_dataset.shape, dtype=np.int64)
            chunksize =  np.array(hdf_dataset.chunks, dtype=np.int64)
        else:
            datasize = self.datasize; chunksize = self.chunksize
        nchunks = datasize/chunksize
        #print(nchunks, chunksize, datasize)
        if self.hdf5_Corder: ci = chunk_index
        else: ci = chunk_index[::-1]
        # chunk index is either given as origin-centered, or zero-based relative to corner
        if self.origin_chunk_inds: ci = (ci + nchunks/2 + nchunks%2 - 1) # origin-centered chunk index
        # always return the indices into the hdf5 in C-order
        if self.hdf5_Corder: return ci*chunksize + offset
        else: return (ci*chunksize)[::-1] + offset

    # xxx - add asserts to check that data select is inbounds in hdf5, currently not a graceful error
    def get_data_slices_from_indices(self, ind, size, dsize):
        xysel = self.zreslice_dim_ordering[0:2]; zsel = self.zreslice_dim_ordering[2]
        beg = ind; end = ind + size
        beg[xysel] = beg[xysel] - self.image_size/2; beg[zsel] = beg[zsel] - self.nzslices//2
        end[xysel] = end[xysel] + self.image_size/2; end[zsel] = end[zsel] + self.nzslices//2
        return self.get_slices_from_limits(beg,end,dsize)
        
    def get_slices_from_limits(self, beg, end, size):
        zsel = self.zreslice_dim_ordering[2]
        begd = np.zeros_like(size); endd = size;
        begd[zsel], endd[zsel] = 0, self.ntotal_zslice
        if self.hdf5_Corder: 
            slc = np.s_[beg[0]:end[0],beg[1]:end[1],beg[2]:end[2]]
            slcd = np.s_[begd[0]:endd[0],begd[1]:endd[1],begd[2]:endd[2]]
        else:
            slc = np.s_[beg[2]:end[2],beg[1]:end[1],beg[0]:end[0]]
            slcd = np.s_[begd[2]:endd[2],begd[1]:endd[1],begd[0]:endd[0]]
        return slc,slcd

    def writeRaw(self):
        if not self.outraw: return
        # the transpose of the first two dims is to be consistent with Kevin's legacy matlab scripts that swap them
        if self.dpLoadh5_verbose: 
            print('Writing raw output to "%s"' % self.outraw); t = time.time()

        # optional manipulations to write raw file
        data = self.data_cube
        islabels = (self.data_type == np.uint16 or self.data_type == np.uint32 or self.data_type == np.uint64)
        doGray = bool(self.dtypeGray)
        if doGray:
            dtypeGray = eval('np.' + self.dtypeGray) if isinstance(self.dtypeGray, str) else self.dtypeGray
        doZeropad = (self.zeropadraw > 0).any()
        doLUTmod = self.nColorsLUTraw and islabels
        doRelabel = self.relabel_seq and islabels
        if doZeropad:
            s = tuple(self.zeropadraw.reshape((3,-1)).tolist())
            data = np.lib.pad(data, s, 'constant',constant_values=tuple(np.zeros((3,2)).tolist()))
        else:
            if doGray or doLUTmod: data = np.copy(data)
        if doGray: 
            data -= data.min(); data /= data.max(); data = (data*np.iinfo(dtypeGray).max).astype(dtypeGray)
        if doLUTmod: data = data % self.nColorsLUTraw
        if self.relabel_seq and islabels:
            data, fw, inv = relabel_sequential(data); data = data.astype(self.data_type)

        # the transpose of the first two dims is to be consistent with Kevin's legacy matlab scripts that swap them
        shape = data.shape
        if self.legacy_transpose: data = data.transpose((2,0,1)); shape = tuple(np.array(shape)[[1,0,2]].tolist())
        else: data = data.transpose((2,1,0))
        
        if os.path.splitext(self.outraw)[1][1:] == 'nrrd':
            # stole this from pynrrd (which wasn't working by itself, gave up on it)
            _TYPEMAP_NUMPY2NRRD = {'i1': 'int8','u1': 'uint8','i2': 'int16','u2': 'uint16','i4': 'int32','u4': 'uint32',
                'i8': 'int64','u8': 'uint64','f4': 'float','f8': 'double','V': 'block'}
            hdr = \
                'NRRD0004\r\n' \
                'type: %s\r\n' % _TYPEMAP_NUMPY2NRRD[data.dtype.str[1:]] + \
                'dimension: 3\r\n' + \
                'space: left-posterior-superior\r\n' + \
                'endian: little\r\n' + \
                'sizes: %d %d %d\r\n' % shape + \
                'space directions: (%.8f,0,0) (0,%.8f,0) (0,0,%.8f)\r\n' % tuple(self.data_attrs['scale']) + \
                'kinds: domain domain domain\r\n' + \
                'space origin: (0,0,0)\r\n' \
                'encoding: raw\r\n\n'
            if self.dpLoadh5_verbose: print(hdr)
            fh = open(self.outraw, 'w'); fh.write(hdr); fh.close()
            fh = open(self.outraw, 'ab'); data.tofile(fh); fh.close()
        else:
            fh = open(self.outraw, 'wb'); data.tofile(fh); fh.close()
            
        if self.dpLoadh5_verbose: 
            print('\tdone in %.4f s' % (time.time() - t))
                    
    @classmethod
    def readData(cls, srcfile, dataset, chunk, offset, size, data_type='', subgroups=[], verbose=False):
        loadh5 = cls.readInith5(srcfile, dataset, chunk, offset, size, data_type, subgroups, verbose)
        loadh5.readCubeToBuffers()
        return loadh5

    @classmethod
    def readInith5(cls, srcfile, dataset, chunk, offset, size, data_type, subgroups=[], verbose=False):
        parser = argparse.ArgumentParser(description='class:dpLoadh5', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        dpLoadh5.addArgs(parser); arg_str = ''
        arg_str += ' --srcfile ' + srcfile
        if data_type: arg_str += ' --data-type ' + data_type
        arg_str += ' --chunk %d %d %d ' % tuple(chunk)
        arg_str += ' --offset %d %d %d ' % tuple(offset)
        arg_str += ' --size %d %d %d ' % tuple(size)
        arg_str += ' --dataset ' + dataset
        if subgroups: arg_str += ' --subgroups ' + ' '.join(subgroups)
        if verbose: arg_str += ' --dpLoadh5-verbose '
        if verbose: print(arg_str)
        args = parser.parse_args(arg_str.split())
        return cls(args)


    @staticmethod
    def addArgs(p):
        # adds arguments required for this object to specified ArgumentParser object
        p.add_argument('--srcfile', nargs=1, type=str, default='tmp.h5', help='Input file (hdf5)')
        p.add_argument('--dataset', nargs=1, type=str, default='data', help='Name of the dataset to read')
        p.add_argument('--subgroups', nargs='*', type=str, default=[], metavar=('GRPS'),
            help='List of groups to identify subgroup for the dataset (empty for top level)')
        p.add_argument('--data-type', nargs=1, type=str, default='', metavar='DTYPE',
            help='numpy type to read into (default from dataset)')
        p.add_argument('--chunk', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
            help='Corner chunk to parse out of hdf5')
        p.add_argument('--offset', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
            help='Offset in chunk to read')
        p.add_argument('--size', nargs=3, type=int, default=[256,256,128], metavar=('X', 'Y', 'Z'),
            help='Size in voxels to read')
        p.add_argument('--dim-ordering', nargs=1, type=str, default='xyz', choices=('xyz','xzy','zyx'),
            metavar='ORD', help='Specify the order to reslice the dimensions into (last one becomes new z)')
        p.add_argument('--hdf5-Corder', dest='hdf5_Corder', action='store_true', 
            help='Specify hdf5 file is in C-order')
        p.add_argument('--legacy-transpose', dest='legacy_transpose', action='store_true', 
            help='Specify transpose of x/y if writing raw')
        p.add_argument('--outraw', nargs=1, type=str, default='', metavar='FILE', 
            help='Optional raw or nrrd output file')
        p.add_argument('--nColorsLUTraw', nargs=1, type=int, default=[0], metavar=('NCLRS'),
            help='Specify non-zero number of colors for uint data (apply modulo, raw output)')
        p.add_argument('--dtypeGray', nargs=1, type=str, default=[0], metavar=('DTYPE'),
            help='Specify data type for converting to grayscale (raw output)')
        p.add_argument('--zeropadraw', nargs=6, type=int, default=[0,0,0,0,0,0], 
            metavar=('Xb', 'Xa', 'Yb', 'Ya', 'Zb', 'Za'), help='Size in voxels to zero pad (b=before, a=after)')
        p.add_argument('--relabel-seq', dest='relabel_seq', action='store_true', 
            help='Relabel sequentially (labels only)')
        p.add_argument('--size-in-chunks', action='store_true', help='Size is specified in chunks, not voxels')
        p.add_argument('--size-from-hdf5', action='store_true', help='Size is specified by attribute in hdf5')
        p.add_argument('--dpLoadh5-verbose', action='store_true', help='Debugging output for dpLoadh5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read hdf5 input cubes for labrainth',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpLoadh5.addArgs(parser)
    args = parser.parse_args()
    
    loadh5 = dpLoadh5(args)
    loadh5.readCubeToBuffers()
    loadh5.writeRaw()

