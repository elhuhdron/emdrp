
skelout = '/Data/datasets/skeletons/skeleton-kara-mod.054.nml';
giplout = '~/Downloads/out.gipl';

chunk = [8 9 3];
dim_order = [1 2 3];
raw_size = 128;

p = struct;
p.raw_size = raw_size;
p.offset_ind = chunk;
p.ncubes_raw = [6 6 3];
p.dim_order = dim_order;
p.isotopic_voxels = false;
p.strel_offs = [0 0 0];
%p.strel_offs = [0 0 0; -1 0 0; 1 0 0; 0 -1 0; 0 1 0; 1 1 0; 1 -1 0; -1 1 0; -1 -1 0];
%p.strel_offs = [0 0 0; -1 0 0; 1 0 0; 0 -1 0; 0 1 0; 1 1 0; 1 -1 0; -1 1 0; -1 -1 0; -2 0 0; 2 0 0; 0 -2 0; 0 2 0];
p.do_plots = true;

tic; rngnodes = knossos_nodes_to_gipl(skelout, giplout, p); toc
