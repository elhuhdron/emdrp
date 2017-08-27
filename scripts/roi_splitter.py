#!/usr/bin/env python
# Script for taking ROIs.txt file that lists manually labelled regions and splitting into regions of specified size.



#import re
#import sys
#import glob
#import os
#import shutil
import numpy as np
import argparse

p = argparse.ArgumentParser(description='Split ROIs into pieces of specified size',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--inroi', nargs=1, type=str, default='',
               help='text file with list of ROIs as chunks, sizes, offsets (in that order)')
p.add_argument('--size', nargs=3, type=int, default=[128,128,128], metavar=('X', 'Y', 'Z'),
               help='Size to split rois into')
p.add_argument('--offset', nargs=3, type=int, default=[0,0,0], metavar=('X', 'Y', 'Z'),
               help='Further offset the offsets (for including context)')
args = p.parse_args()

size=np.array(args.size,dtype=np.int64)
offset=np.array(args.offset,dtype=np.int64)

rois=np.loadtxt(args.inroi[0],dtype=np.int64).reshape((-1,3,3))
roi_chunks = rois[:,0,:].reshape((-1,3))
roi_sizes = rois[:,1,:].reshape((-1,3))
roi_offsets = rois[:,2,:].reshape((-1,3))

assert( (roi_sizes % size == 0).all() )

str_chunks=''; str_offsets=''; tnn=0
for i in range(roi_chunks.shape[0]):
    n = roi_sizes[i] // size; nn = np.prod(n); tnn += nn
    for j in range(nn):
        ci = np.array(np.unravel_index(j, n), dtype=np.int64)
        coffset = roi_offsets[i,:] + ci * size + offset
        
        if j==0 and i==0:
            str_chunks += ('%d,%d,%d' % tuple(roi_chunks[i,:].tolist()))
            str_offsets += ('%d,%d,%d' % tuple(coffset.tolist()))
        else:
            str_chunks += (', %d,%d,%d' % tuple(roi_chunks[i,:].tolist()))
            str_offsets += (', %d,%d,%d' % tuple(coffset.tolist()))

print('total pieces: %d' % (tnn,))
print('chunks:'); print(str_chunks)
print('offsets:'); print(str_offsets)
