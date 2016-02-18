#!/usr/bin/env python
import h5py
import sys

# adapted from:
#https://confluence.slac.stanford.edu/display/PSDM/How+to+access+HDF5+data+from+Python#HowtoaccessHDF5datafromPython-Example3:Printentirefile/groupstructureusingrecursivemethod
 
def print_hdf5_file_structure(file_name, print_attrs=False) :
    """Prints the HDF5 file structure"""
    fh = h5py.File(file_name, 'r') # open read-only
    print_hdf5_item_structure(fh, print_attrs=print_attrs)
    fh.close()
 
def print_hdf5_item_structure(g, offset='    ', print_attrs=False) :
    """Prints the input file/group/dataset (g) name and begin iterations on its content"""
    if   isinstance(g,h5py.File) :
        print(g.file, '(File)', g.name)
 
    elif isinstance(g,h5py.Dataset) :
        print('(Dataset)', g.name, '    len =', g.shape, ' dtype=', g.dtype)
 
    elif isinstance(g,h5py.Group) :
        print('(Group)', g.name)
 
    else :
        print('WARNING: UNKNOWN ITEM IN HDF5 FILE', g.name)
        sys.exit ( "EXECUTION IS TERMINATED" )

    if print_attrs:
        for name,value in g.attrs.items(): 
            print(offset, '%s = %s' % (name, str(value)))
 
    if isinstance(g, h5py.File) or isinstance(g, h5py.Group) :
        #for key,val in dict(g).iteritems() :
        for key,val in iter(sorted(dict(g).items())):
            subg = val
            print(offset, key, end=" ") #,"   ", subg.name #, val, subg.len(), type(subg),
            print_hdf5_item_structure(subg, offset + '    ', print_attrs)
 
if __name__ == "__main__" :
    print_attrs = False
    if len(sys.argv[1:]) > 1:
        print_attrs = (sys.argv[1:][1].lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh'])
    print_hdf5_file_structure(sys.argv[1:][0], print_attrs)

