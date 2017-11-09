
#import argparse
import numpy as np
import h5py
#from scipy import ndimage as nd
#from scipy import io as sio

#from dpLoadh5 import dpLoadh5
#from dpWriteh5 import dpWriteh5
#from typesh5 import emLabels

mesh_in='/home/watkinspv/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean-cut.0.mesh.h5'

nsurfpts = 100;
doplots = True;

# get the volume and surface area data from the mesh file
h5file = h5py.File(mesh_in, 'r'); dset_root = h5file['0']
str_seed = ('%08d' % 0)
scale = dset_root[str_seed]['faces'].attrs['scale']; voxel_volume = scale.prod()
nseeds = len(dset_root)-1
for i in range(nseeds):
    str_seed = ('%08d' % (i+1,))
    #soma_volumes[i] = dset_root[str_seed]['vertices'].attrs['nVoxels'] * voxel_volume
    #soma_surface_areas[i] = dset_root[str_seed]['vertices'].attrs['surface_area']

    vertices = np.empty_like(dset_root[str_seed]['vertices'])
    faces = np.empty_like(dset_root[str_seed]['faces'])
    dset_root[str_seed]['vertices'].read_direct(vertices)
    dset_root[str_seed]['faces'].read_direct(faces)
    pts = vertices.astype(np.double, copy=True); npts = pts.shape[0]

    #C = mean(pts,1); Cpts = bsxfun(@minus,pts,C);
    #
    #  % svd on centered points to get principal axes.
    #  % in matlab V is returned normal (not transposed), so "eigenvectors" are along columns,
    #  %   i.e. V(:,1) V(:,2) V(:,3)
    #  [~,S,V] = svd(Cpts,0);
    #  % the std of the points along the eigenvectors
    #  s = sqrt(diag(S).^2/(npts-1));
    #  
    #  % rotate the points to align on cartesian axes
    #  rpts = bsxfun(@plus,(V'*bsxfun(@minus,pts',C')),C')';
    #
    #  if doplots
    #    plot_pts_fit(rpts, [], C, 4^(1/3)*s, nsteps);
    #    pause
    #  end
    #  
    #  fprintf(1,'seed %d of %d in %.4f s\n',seed,nseeds,(now-t)*86400);
    #end

h5file.close()

