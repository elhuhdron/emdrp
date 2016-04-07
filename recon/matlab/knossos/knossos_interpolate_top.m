

%skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.nml';
skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.nml';

skelout = '/Users/pwatkins/Downloads/out.nml';

p = struct;
p.remove_branch_edges = false;
p.remove_inflection_edges = false;
p.interp_dim = 3;
p.rngdiff = [0 inf];
p.write_new_nml = true;
p.min_nodes = 2;
p.interp_dim_rng = [inf -1];  % do not extrapolate
p.extrap_max = 2;
p.extrap_do_line = true;

tic; [minnodes, rngnodes] = knossos_interpolate(skelin,skelout,p); toc

