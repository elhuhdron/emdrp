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

# Extends python hdf5 load class to write (new or append) hdf5 data in a subset of a whole dataset.

import h5py
import numpy as np
import re
import argparse
import time
import os
from emdrp.dpLoadh5 import dpLoadh5

class dpWriteh5(dpLoadh5):

    HDF5_CLVL = 5           # compression level in hdf5

    def __init__(self, args):
        dpLoadh5.__init__(self, args)

        # Options / Inits
        if not self.outfile: self.outfile = self.srcfile

    def writeCube(self, data=None, outfile=None):
        # do not move this to init, won't work with typesh5.py
        # xxx - this class hierarchy maybe should be revisited someday.... to die
        if not self.data_type_out: self.data_type_out = self.data_type
        if isinstance(self.data_type_out, str): self.data_type_out = eval('np.' + self.data_type_out)
        if not self.fillvalue: self.fillvalue = '0'
        if isinstance(self.fillvalue, str):
            self.fillvalue = np.asscalar(np.fromstring(self.fillvalue, dtype=self.data_type_out, sep=' '))

        if data is None:
            data = self.data_cube.astype(self.data_type_out)
        else:
            #assert(data.dtype == self.data_type)    # xxx - probably revisit this, this was original
            # xxx - is there a problem with fillvalue now?
            data = data.astype(self.data_type_out)
            # xxx - writeRaw will still write with the type of this object, the out type is only for hdf5
            #   this option is mostly for frontend compatibility, revisit this again if this is needed for backend
            # xxx - re-added this, something more comprehensive probably needs to be done about this... meh
            self.data_cube = data; self.data_type = self.data_type_out
        if outfile is None: outfile = self.outfile if self.outfile else self.srcfile
        self.writeRaw()
        if not outfile: return

        # xxx - this probably should be cleaned up, allows for basically "copying" a dataset to another hdf5
        #   with this tool. make this more explicit in how classes are defined?
        if self.dataset_out: self.dataset = self.dataset_out
        if len(self.subgroups_out)==0 or self.subgroups_out[0] is not None: self.subgroups = self.subgroups_out
        if self.offset_out[0] is not None: self.offset = self.offset_out

        dset, group, h5file = self.createh5(outfile)
        if self.dpWriteh5_verbose:
            print('dpWriteh5: Writing hdf5')
            t = time.time()
        # always write outputs in F-order
        ind = self.get_hdf_index_from_chunk_index(dset, self.chunk, self.offset)
        ind = ind[self.zreslice_dim_ordering][::-1] # re-order for specified ordering, then to F-order
        d = data.transpose((2,1,0));
        #print(ind, d.shape, dset.shape, d.max(), d.min(), dset.dtype, d.dtype)
        dset[ind[0]:ind[0]+d.shape[0],ind[1]:ind[1]+d.shape[1],ind[2]:ind[2]+d.shape[2]] = d

        # optionally add a list of chunk Regions of Interest specified in text file
        if self.inroi:
            rois=np.loadtxt(self.inroi,dtype=np.int64).reshape((-1,3,3))
            self.data_attrs['roi_chunks'] = rois[:,0,:].reshape((-1,3))
            self.data_attrs['roi_sizes'] = rois[:,1,:].reshape((-1,3))
            self.data_attrs['roi_offsets'] = rois[:,2,:].reshape((-1,3))

        # write attributes
        for name,value in self.data_attrs.items():
            if name in dset.attrs: del dset.attrs[name]
            newname = self.dataset + '_' + name
            if newname in group: del group[newname]
            # xxx - this is arbitrary, but don't want to exceed 64k hdf5 header limit
            if isinstance(value, np.ndarray) and value.size > 100:
                group.create_dataset(newname, data=value, compression='gzip',
                    compression_opts=self.HDF5_CLVL, shuffle=True, fletcher32=True)
            else:
                #http://stackoverflow.com/questions/23220513/storing-a-list-of-strings-to-a-hdf5-dataset-from-python
                if isinstance(value, str):
                    value = value.encode("ascii", "ignore")
                elif type(value) is list and isinstance(value[0], str):
                    value = [n.encode("ascii", "ignore") for n in value]
                dset.attrs.create(name,value)

        h5file.close()
        if self.dpWriteh5_verbose:
            print('\tdone in %.4f s' % (time.time() - t))

    def createh5(self, outfile):
        h5file = h5py.File(outfile, 'r+' if os.path.isfile(outfile) else 'w')
        dset, group, dsetpath = self.getDataset(h5file)
        if not dset:
            self.createh5dataset(h5file, dsetpath)
            dset, group, dsetpath = self.getDataset(h5file)
            assert( dset )  # dataset not created? this is bad
        return dset, group, h5file

    def createh5dataset(self, h5file, dsetpath):
        if self.dpWriteh5_verbose:
            print('dpWriteh5: Creating hdf5 dataset')
            t = time.time()
        # create an output prob hdf5 file (likely for a larger dataset, this is how outputs are "chunked")
        # get the shape and chunk size from the data hdf5. if this file is in F-order, re-order to C-order
        shape = self.datasize; chunks = self.chunksize
        # do not re-order for F-order here, should have already been re-ordered in dpLoadh5
        #if not self.hdf5_Corder:
        #    shape = shape[::-1]; chunks = chunks[::-1]
        # now re-order the dims based on the specified re-ordering and then re-order back to F-order
        shape = shape[self.zreslice_dim_ordering]; chunks = chunks[self.zreslice_dim_ordering]
        shape = shape[::-1]; chunks = tuple(chunks[::-1])
        h5file.create_dataset(dsetpath, shape=shape, dtype=self.data_type_out, compression='gzip',
            compression_opts=self.HDF5_CLVL, shuffle=True, fletcher32=True, fillvalue=self.fillvalue, chunks=chunks)
        if self.dpWriteh5_verbose:
            print('\tdone in %.4f s' % (time.time() - t))

    def writeFromRaw(self):
        self.loadFromRaw()
        #if self.dpWriteh5_verbose: print(self.data_cube.min(), self.data_cube.max(), self.data_cube.shape)
        # xxx - total hacks, keep commented
        #self.data_attrs['dimOrdering'] = [1,2,3]
        self.writeCube()

    def loadFromRaw(self):
        # xxx - this is duplicated in writeCube(), couldn't move it to init because of types.h5
        #   code needs some refactoring that allows for a single base class and input and output types.
        if not self.data_type_out: self.data_type_out = self.data_type
        if isinstance(self.data_type_out, str): self.data_type_out = eval('np.' + self.data_type_out)

        ext = os.path.splitext(self.inraw)[1][1:]
        use_const = False
        if not ext:
            # kludgy support writing constants (for use as a mask, etc)
            try:
                const = float(self.inraw)
                use_const = True
            except ValueError:
                pass

        if use_const:
            print('Initializing with constant value %d' % (const,))
            data = const * np.ones(self.size, dtype=self.data_type_out)
        elif ext == 'nrrd':
            # stole this from pynrrd (which wasn't working by itself, gave up on it)
            # xxx - new version is available as of early 2016, try migrating to it
            _TYPEMAP_NRRD2NUMPY = {'signed char': 'i1', 'int8': 'i1', 'int8_t': 'i1', 'uchar': 'u1',
                'unsigned char': 'u1', 'uint8': 'u1', 'uint8_t': 'u1', 'short': 'i2', 'short int': 'i2',
                'signed short': 'i2', 'signed short int': 'i2', 'int16': 'i2', 'int16_t': 'i2', 'ushort': 'u2',
                'unsigned short': 'u2', 'unsigned short int': 'u2', 'uint16': 'u2', 'uint16_t': 'u2', 'int': 'i4',
                'signed int': 'i4', 'int32': 'i4', 'int32_t': 'i4', 'uint': 'u4', 'unsigned int': 'u4', 'uint32': 'u4',
                'uint32_t': 'u4', 'longlong': 'i8', 'long long': 'i8', 'long long int': 'i8', 'signed long long': 'i8',
                'signed long long int': 'i8', 'int64': 'i8', 'int64_t': 'i8', 'ulonglong': 'u8',
                'unsigned long long': 'u8', 'unsigned long long int': 'u8', 'uint64': 'u8', 'uint64_t': 'u8',
                'float': 'f4', 'double': 'f8', 'block': 'V'
                }
            with open(self.inraw,'rb') as nrrdfile:
                headerSize = 0; hdr = {'type':self.data_type_out, 'endian':'little'}
                for raw_line in iter(nrrdfile):
                    headerSize += len(raw_line)
                    raw_line = raw_line.decode('ascii')
                    # Trailing whitespace ignored per the NRRD spec
                    line = raw_line.rstrip()
                    # Single blank line separates the header from the data
                    if line == '': break

                    # xxx - very basic header elements
                    reline = line.lstrip()
                    m = re.search(r'type\:\s+(.+)', reline)
                    if m is not None: hdr['type'] = _TYPEMAP_NRRD2NUMPY[m.group(1).strip()]
                    m = re.search(r'endian\:\s+(\w+)', reline)
                    if m is not None:
                        endian = m.group(1).lower()
                        if endian in ['litle','big']:
                            hdr['type'] = ('<' if endian == 'little' else '>') + hdr['type']
                    #space directions: (13.20000000,0,0) (0,13.20000000,0) (0,0,30.00000000)
                    #space directions: (13.199999999999999,0,0) (0,13.199999999999999,0) (0,0,30)
                    m = re.search(r'space directions\:'
                                   '\s+\((\d*\.\d+|\d+),0,0\) \(0,(\d*\.\d+|\d+),0\) \(0,0,(\d*\.\d+|\d+)\)', reline)
                    if m is not None and 'scale' not in self.data_attrs:
                        self.data_attrs['scale'] = np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))])

                nrrdfile.seek(headerSize)
                # xxx - fix this to get data type and endianess from the header, pynrrd still sucks too much
                #data = np.fromfile(nrrdfile,dtype=self.data_type_out)
                #data = np.fromfile(nrrdfile,dtype=self.data_type_out).byteswap(True)    # meh, imagej
                # addded very basic header elements above just to get the type and endianess correctly
                print('nrrd data type for numpy: ' + hdr['type'])
                data = np.fromfile(nrrdfile,dtype=np.dtype(hdr['type']))
            # pynrrd is super slow and does some kind of view changing for some reason
            #import nrrd
            #data, hdr = nrrd.read(self.inraw)
        elif ext == 'gipl':
            data, hdr, info = dpWriteh5.gipl_read_volume(self.inraw)
            if 'scale' not in self.data_attrs:
                self.data_attrs['scale'] = hdr['scales'][:3]
        else:
            if self.inraw_bigendian:
                data = np.fromfile(self.inraw,dtype=self.data_type_out).byteswap(True)
            else:
                data = np.fromfile(self.inraw,dtype=self.data_type_out)

        # xxx - hacky command line over-ride for scale
        if all([x > 0 for x in self.scale]): self.data_attrs['scale'] = self.scale

        # xxx - this always assumes raw file is in F-order, add something here for C-order if we need it
        #self.data_cube = data.astype(self.data_type_out).reshape(self.size[::-1]).transpose((2,1,0))
        # add support for reslice reordering of raw inputs
        zord = self.zreslice_dim_ordering; size = self.size[zord]; tord = [2,1,0]
        self.data_cube = data.astype(self.data_type_out).reshape(size[::-1]).transpose([tord[i] for i in zord])

    # xxx - move this as a utility or a GIPL class?
    # translated from matlab toolbox http://www.mathworks.com/matlabcentral/fileexchange/16407-gipl-toolbox

    @staticmethod
    def gipl_read_header(fname):
        hdr, info = dpLoadh5.gipl_generate_header()
        fh = open(fname, 'rb')

        # add the file size and name to the info struct
        fh.seek(0, os.SEEK_END); info['filesize'] = fh.tell(); fh.seek(0); info['filename'] = fname

        # read binary header with correct order / data types, gipl format is big-endian!!!
        for field in info['hdr_fields']:
            hdr[field] = np.fromfile(fh, dtype=hdr[field].dtype, count=hdr[field].size).byteswap(True)
            #print('\t',field,'\tsize',hdr[field].size,'\ttell ',fh.tell())
        assert( fh.tell() == info['hdr_size_bytes'] )
        assert( hdr['magic_number'] == info['magic_number'] )
        fh.close()

        return hdr, info

    @staticmethod
    def gipl_read_volume(fname):
        hdr, info = dpWriteh5.gipl_read_header(fname)

        dtype = info['numpy_types'][hdr['image_type'][0]]
        datasize = hdr['sizes'].prod(dtype=np.int64)

        # read data, gipl format is big-endian!!!
        fh = open(fname, 'rb')
        fh.seek(info['hdr_size_bytes'])
        V = np.fromfile(fh, dtype=dtype, count=datasize).byteswap(True).reshape((hdr['sizes'][:3]))

        return V, hdr, info

    @classmethod
    def writeData(cls, outfile, dataset, chunk, offset, size, data_type, datasize, chunksize, fillvalue=None, data=None,
            inraw='', outraw='', attrs={}, subgroups_out=[], verbose=False):
        assert( data is not None or inraw )
        parser = argparse.ArgumentParser(description='class:dpWriteh5',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        dpWriteh5.addArgs(parser); arg_str = ''
        arg_str += ' --srcfile ' + outfile
        arg_str += ' --data-type ' + data_type
        arg_str += ' --chunk %d %d %d ' % tuple(chunk)
        arg_str += ' --offset %d %d %d ' % tuple(offset)
        arg_str += ' --size %d %d %d ' % tuple(size)
        arg_str += ' --dataset ' + dataset
        arg_str += ' --chunksize %d %d %d' % tuple(chunksize)
        arg_str += ' --datasize %d %d %d' % tuple(datasize)
        if fillvalue: arg_str += ' --fillvalue ' + str(fillvalue)
        if inraw: arg_str += ' --inraw ' + inraw
        if outraw: arg_str += ' --outraw ' + outraw
        if subgroups_out: arg_str += ' --subgroups-out ' + ' '.join(subgroups_out)
        if verbose: arg_str += ' --dpWriteh5-verbose '
        if verbose: print(arg_str)
        args = parser.parse_args(arg_str.split())
        writeh5 = cls(args); writeh5.data_attrs = attrs
        if inraw: writeh5.writeFromRaw()
        else: writeh5.writeCube(data)
        return writeh5


    @staticmethod
    def addArgs(p):
        # adds arguments required for this object to specified ArgumentParser object
        dpLoadh5.addArgs(p)
        p.add_argument('--outfile', nargs=1, type=str, default='',
            help='Output file (allows dataset copy), default: srcfile')
        p.add_argument('--chunksize', nargs=3, type=int, default=[-1,-1,-1], metavar=('X', 'Y', 'Z'),
            help='Chunk size to use for new hdf5')
        p.add_argument('--datasize', nargs=3, type=int, default=[-1,-1,-1], metavar=('X', 'Y', 'Z'),
            help='Total size of the hdf5 dataset')
        p.add_argument('--fillvalue', nargs=1, type=str, default=[''], metavar=('FILL'),
            help='Fill value for empty (default 0)')
        p.add_argument('--inraw', nargs=1, type=str, default='', metavar='FILE', help='Raw input file')
        p.add_argument('--inraw-bigendian', action='store_true', help='Raw input is big endian format')
        p.add_argument('--scale', nargs=3, type=float, default=[0.0,0.0,0.0], metavar=('X', 'Y', 'Z'),
            help='Override scale (use only with inraw and without srcfile')
        p.add_argument('--inroi', nargs=1, type=str, default='',
                       help='text file with list of ROIs as chunks, sizes, offsets (in that order)')
        p.add_argument('--dataset-out', nargs=1, type=str, default='',
            help='Name of the dataset to write: default: dataset')
        p.add_argument('--subgroups-out', nargs='*', type=str, default=[None], metavar=('GRPS'),
            help='List of groups to identify subgroup for the output dataset (empty for top level), default:subgroups')
        p.add_argument('--data-type-out', nargs=1, type=str, default='', metavar='DTYPE',
            help='numpy type to write out as')
        p.add_argument('--offset-out', nargs=3, type=int, default=[None,None,None], metavar=('X', 'Y', 'Z'),
            help='Hacky way to shift datasets over during "copy"')
        p.add_argument('--dpWriteh5-verbose', action='store_true', help='Debugging output for dpWriteh5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write (create if no file) hdf5 file at specified location',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpWriteh5.addArgs(parser)
    args = parser.parse_args()

    writeh5 = dpWriteh5(args)
    if writeh5.outfile and not writeh5.inraw:
        writeh5.readCubeToBuffers()
        writeh5.writeCube()
    else:
        writeh5.writeFromRaw()

