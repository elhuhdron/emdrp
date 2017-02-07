
import numpy as np
import h5py
import sys

def chunk_mask_to_raw(h5file, outfile, chunk_mask_name):
    h5file = h5py.File(h5file, 'r')

    chunk_mask = np.empty_like(h5file[chunk_mask_name])
    h5file[chunk_mask_name].read_direct(chunk_mask)

    s = chunk_mask.shape
    fn = outfile + ('_%s_%dx%dx%d' % (str(chunk_mask.dtype),s[2],s[1],s[0]) + '.raw')
    chunk_mask.tofile(fn)

if __name__ == "__main__" :
    chunk_mask_to_raw(*sys.argv[1:])
