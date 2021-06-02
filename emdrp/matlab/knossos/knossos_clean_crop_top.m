
p = struct;
%p.nmlin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.nml';
p.nmlin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.nml';
%p.chunk = [16 17 0];
p.chunk = [12 14 2];
p.nchunks = [8 8 4];
p.offset = [0 0 32];
p.chunksize = 128;
p.nmlout = '~/Downloads/out2.nml';
p.crop_conncomp = 1;

tic; o = knossos_clean_crop(p); toc

