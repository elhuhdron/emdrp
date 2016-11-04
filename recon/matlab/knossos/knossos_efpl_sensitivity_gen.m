
% generate label inputs for sensitivity test for skeleton based metrics.
% this is done by only modifying labels at each node location 
%   and writing to input h5 file for analysis with knossos_efpl.

p = struct;
p.nmlin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.nml';
%p.nmlin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.nml';
p.offset_ind = [15 16 0];
%p.offset_ind = [11 13 2];
p.raw_size = 128;
%p.ncubes_raw = [8 8 4];
p.ncubes_raw = [10 10 4];
p.dim_order = [1 2 3];
p.isotopic_voxels = false;
%p.strel_offs = [0 0 0];
%p.strel_offs = [0 0 0; -1 0 0; 1 0 0; 0 -1 0; 0 1 0; 1 1 0; 1 -1 0; -1 1 0; -1 -1 0];
p.strel_offs = [0 0 0; -1 0 0; 1 0 0; 0 -1 0; 0 1 0; 1 1 0; 1 -1 0; -1 1 0; -1 -1 0; -2 0 0; 2 0 0; 0 -2 0; 0 2 0];
p.dtype_str = 'uint16';

% p.merge_percs = [0 0.01 0.025 0.05 0.075 0.1 0.2];
% p.split_percs = [0 0.05 0.1 0.15 0.25 0.5 0.75];
% p.merge_percs = [0 0.01 0.1];
% p.split_percs = [0 0.05 0.2];

% generate "realistic" split merger curves.
p.params_meshed = true;
alphax=logspace(-2,0,9); alphax=[0.0001 0.001 0.004 alphax];
%alphax=[0.0001 0.001];
splitx=[0 0.0001 0.001 0.01 0.03 0.06 0.1:0.1:0.2 0.4:0.2:1];
% order in nodes_to_gipl: params = {p.merge_percs p.split_percs};
[alpha, split]=ndgrid(alphax,splitx); 
merge=alpha.*(alpha+1)./(split+alpha)-alpha;
% remove duplicates
tmp=round([split(:) merge(:)],8); [~,n]=unique(tmp,'rows','stable'); 
sel = true(size(tmp,1),1); sel(n) = false;
split(sel) = nan; merge(sel) = nan;
p.split_percs = split;
p.merge_percs = merge;

p.hdf5lbls = '/Data/datasets/labels/gt/M0007_33_labels_briggmankl_watkinspv_39x35x7chunks_Forder.h5';

for i = 1:10
  fn = sprintf('/home/watkinspv/Downloads/tmp%d.h5', i)
  p.hdf5out = fn;
  tic; o = knossos_simulate_errors(p); toc
end
