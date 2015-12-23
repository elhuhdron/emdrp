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
import argparse
import time
import os
from dpLoadh5 import dpLoadh5

class dpWriteh5(dpLoadh5):

    HDF5_CLVL = 5           # compression level in hdf5

    def __init__(self, args):
        dpLoadh5.__init__(self, args)

        # Options / Inits
        if not self.fillvalue: self.fillvalue = '0'
        if isinstance(self.fillvalue, str): 
            self.fillvalue = np.asscalar(np.fromstring(self.fillvalue, dtype=self.data_type, sep=' '))


    def writeCube(self, data=None, outfile=None):
        # do not move this to init, won't work with typesh5.py
        # xxx - this class hierarchy maybe should be revisited someday.... to die
        if not self.data_type_out: self.data_type_out = self.data_type
        if isinstance(self.data_type_out, str): self.data_type_out = eval('np.' + self.data_type_out)
        
        if data is None: 
            data = self.data_cube
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
        if hasattr(self, 'data_attrs'):
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
        if self.dpWriteh5_verbose: print(self.data_cube.min(), self.data_cube.max(), self.data_cube.shape)
        self.writeCube()
        
    def loadFromRaw(self):
        # xxx - this always assumes raw file is in F-order, add something here for C-order if we need it
        if os.path.splitext(self.inraw)[1][1:] == 'nrrd':
            # stole this from pynrrd (which wasn't working by itself, gave up on it)
            with open(self.inraw,'rb') as nrrdfile:
                headerSize = 0
                for raw_line in iter(nrrdfile):
                    headerSize += len(raw_line)
                    raw_line = raw_line.decode('ascii')
                    # Trailing whitespace ignored per the NRRD spec
                    line = raw_line.rstrip()
                    # Single blank line separates the header from the data
                    if line == '': break
                nrrdfile.seek(headerSize)
                # for uint16 here because currently this is just for itksnap and that's the only way it writes (?)
                # xxx - get this from the header? what if not consistent with data-size instantiated for this object?
                data = np.fromfile(nrrdfile,dtype=np.uint16).reshape(self.size[::-1]).transpose((2,1,0)).\
                    astype(self.data_type)
        else:
            data = np.fromfile(self.inraw,dtype=self.data_type).reshape(self.size[::-1]).transpose((2,1,0))
        self.data_cube = data

    @classmethod
    def writeData(cls, outfile, dataset, chunk, offset, size, data_type, datasize, chunksize, fillvalue=None, data=None, 
            inraw='', outraw='', attrs={}, verbose=False):
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
        p.add_argument('--chunksize', nargs=3, type=int, default=[128,128,128], metavar=('X', 'Y', 'Z'),
            help='Chunk size to use for new hdf5')
        p.add_argument('--datasize', nargs=3, type=int, default=[1024,1024,256], metavar=('X', 'Y', 'Z'),
            help='Total size of the hdf5 dataset')
        p.add_argument('--fillvalue', nargs=1, type=str, default=[''], metavar=('FILL'),
            help='Fill value for empty (default 0)')
        p.add_argument('--inraw', nargs=1, type=str, default='', metavar='FILE', help='Raw input file')
        p.add_argument('--dataset-out', nargs=1, type=str, default='', 
            help='Name of the dataset to write: default: dataset')
        p.add_argument('--subgroups-out', nargs='*', type=str, default=[None], metavar=('GRPS'),
            help='List of groups to identify subgroup for the output dataset (empty for top level), default:subgroups')
        p.add_argument('--data-type-out', nargs=1, type=str, default='', metavar='DTYPE',
            help='numpy type to write out as')
        p.add_argument('--dpWriteh5-verbose', action='store_true', help='Debugging output for dpWriteh5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write (create if no file) hdf5 file at specified location',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpWriteh5.addArgs(parser)
    args = parser.parse_args()
    
    writeh5 = dpWriteh5(args)
    if writeh5.outfile:
        writeh5.readCubeToBuffers()
        writeh5.writeCube()
    else:
        writeh5.writeFromRaw()
    
