
p = struct;
p.nmlin = '/mnt/cne/pwatkins/from_externals/ECS_paper/skeletons/M0007_33_dense_skels.152.interp.nml';
%p.nmlin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.nml';
p.chunk = [16 17 0];
%p.chunk = [12 14 2];
p.chunksize = 128;

% orig crop off top 32
% p.nchunks = [8 8 4];
% p.offset = [0 0 32];
% p.crop_conncomp = 1;

% GAN 2 slices missing at 150-1 (zero based)
p.size = [1024 1024 18];
p.offset = [0 0 142];
p.crop_conncomp = 1;

% % GAN 4 slices missing at 200-3 (zero based)
% p.size = [1024 1024 20];
% p.offset = [0 0 192];
% p.crop_conncomp = 1;

%p.nmlout = '~/Downloads/Bahar_GAN_tmp/M0007_33_dense_skels.152.interp.crop142-160.nml';
p.nmlout = '~/Downloads/Bahar_GAN_tmp/M0007_33_dense_skels.152.interp.crop192-212.nml';

tic; o = knossos_clean_crop(p); toc
