
# Used this hacked up script to rename threshold "segmentation levels".
# Had to do this because had the threshold order backwards in volume_*.sh files for cleaning and meshing.

import h5py
import sys

fn = sys.argv[1:][2]

h5file = h5py.File(fn, 'r+')

inlvl = sys.argv[1:][0]
outlvl = sys.argv[1:][1]

#print(inlvl, outlvl, list(h5file.keys()))
h5file[outlvl] = h5file[inlvl]
del h5file[inlvl]
h5file.close()

