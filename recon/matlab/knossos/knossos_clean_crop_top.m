
p = struct;
%p.nmlin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.nml';
p.nmlin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.nml';
%p.offset_ind = [16 17 0];
p.offset_ind = [12 14 2];
p.raw_size = 128;
p.ncubes_raw = [8 8 4];
p.offset = [0 0 32];

p.nmlout = '~/Downloads/out2.nml';

tic; o = knossos_clean_crop(p); toc

