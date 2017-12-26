
% downsample densely skeletonized k0725 cube.
% this cube was initially intended as seeds for contouring, using for efpl validation also.

skelin = '/Data/datasets/skeletons/k0725_contourcube_cube_8to9_9to10_3.055.nml';
skelout = '~/Downloads/k0725_contourcube_cube_x4_y4o64_z3_dsx2y2z1.055.nml';

p = struct;
p.ds_ratio = [2 2 1];
p.use_radii = false;
p.experiment = 'k0725';

tic; [minnodes, rngnodes] = knossos_downsample(skelin,skelout,p); toc
