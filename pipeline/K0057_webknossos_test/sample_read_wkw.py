
# to export raw nrrd (using emdrp toolset):
#dpLoadh5.py --srcfile /mnt/cne/from_externals/K0057_D31/K0057_D31.h5 --chunk 90 78 15 --size 192 192 32 --offset 96 96 96 --dataset data_mag1 --outraw ~/Downloads/K0057_webknossos_test/K0057_D31_x90o96_y78o96_z15o96.nrrd --dpL

# script to read the webknossos formatted label volume and export as nrrd.

# pip install wkw
import wkw
# pip install pynrrd
import nrrd
import numpy as np
import os

# top level path to data files
dn = '/home/pwatkins/Downloads/K0057_webknossos_test'

# location of the nrrd raw data file
raw_fn = os.path.join(dn, 'K0057_D31_x90o96_y78o96_z15o96.nrrd')

# location of the stored wkw labels (top dir)
wkw_labels = os.path.join(dn, 'data/1')

# filename where to store the nrrd formatted label data
out_labels = os.path.join(dn, 'K0057_D31_x90o96_y78o96_z15o96_labels.nrrd')

# location of the labels within the dataset
labeled_coordinate = [11616, 10080, 2016]
labeled_size = [192, 192, 32]

dataset = wkw.Dataset.open(wkw_labels)
data = dataset.read(labeled_coordinate, labeled_size)
print(data.shape)

# xxx - what is the first singleton dimension?
data = np.squeeze(data)

rawdata, header = nrrd.read(raw_fn)
print(rawdata.shape, rawdata.dtype, data.shape, data.dtype)
print(header)

header['type'] = 'uint32'
nrrd.write(out_labels, data, header)
