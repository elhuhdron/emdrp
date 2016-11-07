

skelout = '~/Downloads/out.nml';
giplout = '~/Downloads/out.gipl';

% skelin = 'M0025_02_skelseeds.023.nml';
% chunk = [14 14 4];
% dim_order = [2 1 3];
skelin = '/Data/datasets/skeletons/M0007_33_MLseeds.024.nml';
chunk = [19 20 1];
dim_order = [1 2 3];

raw_size = 128;

p = struct;
p.remove_branch_edges = false;
p.remove_inflection_edges = false;
p.interp_dim = 3;
p.rngdiff = [1 2];
p.write_new_nml = true;
p.min_nodes = 2;
p.interp_dim_rng = ([0 1]+chunk(p.interp_dim))*raw_size; p.interp_dim_rng(1) = p.interp_dim_rng(1)+1;
p.extrap_max = 2;
p.extrap_do_line = true;

tic; [minnodes, rngnodes] = knossos_interpolate(skelin,skelout,p); toc

p = struct;
p.raw_size = raw_size;
p.offset_ind = chunk;
p.ncubes_raw = [2 2 1];
p.dim_order = dim_order;
p.isotopic_voxels = false;
p.strel_offs = [0 0 0];
%p.strel_offs = [0 0 0; -1 0 0; 1 0 0; 0 -1 0; 0 1 0; 1 1 0; 1 -1 0; -1 1 0; -1 -1 0];
%p.strel_offs = [0 0 0; -1 0 0; 1 0 0; 0 -1 0; 0 1 0; 1 1 0; 1 -1 0; -1 1 0; -1 -1 0; -2 0 0; 2 0 0; 0 -2 0; 0 2 0];
p.do_plots = true;

tic; rngnodes = knossos_nodes_to_gipl(skelout, giplout, p); toc

