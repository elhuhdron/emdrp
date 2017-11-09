
# take cutting planes generated in matlab ellipse fitting code and perform cuts on seg file

#import argparse
#import time
import numpy as np
#import h5py
#from scipy import ndimage as nd
#from scipy import io as sio
#from scipy import interpolate
#from skimage import morphology as morph
#import networkx as nx

#from numpy import linalg as nla
from scipy import ndimage as nd
#from scipy import linalg as sla

from scipy import io as sio

from dpLoadh5 import dpLoadh5
from dpWriteh5 import dpWriteh5
#from typesh5 import emLabels, emProbabilities, emVoxelType
from typesh5 import emLabels
#from pyCext import binary_warping

somas_in='/home/watkinspv/Downloads/K0057_soma_annotation/out/K0057_D31_dsx12y12z4_somas_clean_cut.gipl'
data, hdr, info = dpWriteh5.gipl_read_volume(somas_in)
somas = data.reshape(hdr['sizes'][:3][::-1]).transpose(((2,1,0)))
sampling = hdr['scales'][:3]
print(somas.shape, somas.dtype, sampling)

sizes = emLabels.getSizes(somas); sizes = sizes[1:]; 
soma_valid_labels = (np.transpose(np.nonzero(sizes > 0)) + 1).reshape(-1).tolist()
print( 'Number of soma labels is %d' % (len(soma_valid_labels),) )

#mat_in='/home/watkinspv/Downloads/K0057_soma_annotation/out/soma_cuts.mat'
mat_in='/home/watkinspv/Downloads/K0057_soma_annotation/out/somas_cut_fits.mat'
d = sio.loadmat(mat_in)

apply_cuts=False
apply_ellipsoids=True

# iterate over labels, fill each label within bounding box
svox_bnd = nd.measurements.find_objects(somas, max_label=sizes.size)
cut_somas = np.zeros_like(somas)
for j in soma_valid_labels:

    if apply_cuts:
        sel = (somas[svox_bnd[j-1]] == j) # binary select within bounding box
        pts = np.transpose(np.nonzero(sel)).astype(np.double)*sampling # pts is nx3
        npts = pts.shape[0]
    
        # remove points outside cutting planes loaded from mat file
        #(sum(bsxfun(@times,pts,n),2) + d(1) > 0) & (sum(bsxfun(@times,pts,n),2) + d(2) < 0);
        sel_pts = np.ones((npts,), np.bool)
        if not np.isinf(d['cut_d'][j-1,0]):
            sel_pts = np.logical_and(sel_pts, (pts*d['cut_n'][j-1,:]).sum(1) + d['cut_d'][j-1,0] > 0)
            assert(sel_pts.sum() < npts)
        if not np.isinf(d['cut_d'][j-1,1]):
            sel_pts = np.logical_and(sel_pts, (pts*d['cut_n'][j-1,:]).sum(1) + d['cut_d'][j-1,1] < 0)

    if apply_ellipsoids:
        sel = np.ones(somas[svox_bnd[j-1]].shape, dtype=np.bool) # binary select of all bounding box
        pts = np.transpose(np.nonzero(sel)).astype(np.double)*sampling # pts is nx3

        R = d['fit_R'][j-1,:,:].reshape([3,3]); C = d['fit_C'][j-1,:].reshape([3,1])
        rpts = (np.dot(R.T, pts.T - C) + C).T # eigenvector rotation matrix around center from matlab
        v = d['fit_v'][j-1,:].reshape([10]); x = rpts[:,0]; y = rpts[:,1]; z = rpts[:,2]
        sel_pts = (v[0] *x*x +   v[1] * y*y + v[2] * z*z + \
            2*v[3] *x*y + 2*v[4]*x*z + 2*v[5] * y*z + \
            2*v[6] *x    + 2*v[7]*y    + 2*v[8] * z >= -v[9])

    pts = np.round(pts[sel_pts,:]/sampling).astype(np.int64)
    sel_out = np.zeros_like(sel); sel_out[[pts[:,x] for x in range(3)]] = 1
    print('For label %d count %d to %d' % (j, sel.sum(), sel_out.sum()))
    cut_somas[svox_bnd[j-1]][sel_out] = j

        
#somas_out='/home/watkinspv/Downloads/K0057_soma_annotation/out/K0057_D31_dsx12y12z4_somas_clean_cut.gipl'
somas_out='/home/watkinspv/Downloads/K0057_soma_annotation/out/K0057_D31_dsx12y12z4_somas_clean_cut_ellipses.gipl'
dpLoadh5.gipl_write_volume(cut_somas.transpose((2,1,0)), np.array(cut_somas.shape), somas_out, hdr['scales'][:3])
