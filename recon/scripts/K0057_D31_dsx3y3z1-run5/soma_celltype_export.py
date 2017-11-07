
# simple volume and surface area export to mat file for soma data 

#import argparse
#import time
import numpy as np
import h5py
#from scipy import ndimage as nd
from scipy import io as sio
#from scipy import interpolate
#from skimage import morphology as morph
#import networkx as nx

from dpLoadh5 import dpLoadh5
from dpWriteh5 import dpWriteh5
#from typesh5 import emLabels, emProbabilities, emVoxelType
from typesh5 import emLabels
#from pyCext import binary_warping

overlay_in='/home/watkinspv/Downloads/K0057_soma_annotation/out/K0057_D31_soma_seg_overlays_v3_dsx12y12z4.gipl'
data, hdr, info = dpWriteh5.gipl_read_volume(overlay_in)
cell_centers_types = data.reshape(hdr['sizes'][:3][::-1]).transpose(((2,1,0)))
print(data.shape, data.dtype)

somas_in='/home/watkinspv/Downloads/K0057_soma_annotation/out/K0057_D31_dsx12y12z4_somas_cut.gipl'
data, hdr, info = dpWriteh5.gipl_read_volume(somas_in)
somas = data.reshape(hdr['sizes'][:3][::-1]).transpose(((2,1,0)))
print(data.shape, data.dtype)

sizes = emLabels.getSizes(somas); sizes = sizes[1:]; 
soma_valid_labels = np.transpose(np.nonzero(sizes > 0)) + 1
print( 'Number of soma labels is %d' % (soma_valid_labels.size,) )

# get the soma label and cell type for each soma center
sel = (cell_centers_types > 0); soma_labels = somas[sel]; soma_types = cell_centers_types[sel]; del cell_centers_types
inds = np.transpose(np.nonzero(sel)); del sel
# remove unlabeled somas
sel = (soma_labels > 0); soma_labels = soma_labels[sel]; soma_types = soma_types[sel]; inds = inds[sel,:]
nLabels = soma_labels.size

# get the volume and surface area data from the mesh file
mesh_in='/home/watkinspv/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean-cut.0.mesh.h5'
h5file = h5py.File(mesh_in, 'r'); dset_root = h5file['0']
soma_volumes = np.zeros((nLabels,),dtype=np.double); soma_surface_areas = np.zeros((nLabels,),dtype=np.double);
str_seed = ('%08d' % 0)
scale = dset_root[str_seed]['faces'].attrs['scale']; voxel_volume = scale.prod()
for i in range(nLabels):
    str_seed = ('%08d' % soma_labels[i])
    soma_volumes[i] = dset_root[str_seed]['vertices'].attrs['nVoxels'] * voxel_volume
    soma_surface_areas[i] = dset_root[str_seed]['vertices'].attrs['surface_area']
h5file.close()

mat_out='/home/watkinspv/Downloads/K0057_soma_annotation/out/somas_cut.mat'
sio.savemat(mat_out, {'soma_labels':soma_labels, 'soma_types':soma_types, 'soma_center_inds':inds,
                      'soma_volumes':soma_volumes, 'soma_surface_areas':soma_surface_areas,
                      'soma_valid_labels':soma_valid_labels})

# export another volume with each soma labeled with its cell type
soma_types = np.insert(soma_types,0,0); somas = soma_types[somas]
somas_out='/home/watkinspv/Downloads/K0057_soma_annotation/out/K0057_D31_dsx12y12z4_somas_celltypes_cut.gipl'
dpLoadh5.gipl_write_volume(somas.transpose((2,1,0)), np.array(somas.shape), somas_out, hdr['scales'][:3])
