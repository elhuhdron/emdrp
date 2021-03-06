
% generate label inputs for sensitivity test for skeleton based metrics.
% this is done by only modifying labels at each node location 
%   and writing to input h5 file for analysis with knossos_efpl.

p = struct;
%p.nmlin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.crop.nml';
p.nmlin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.crop.nml';
%p.offset_ind = [15 16 0];
%p.offset_ind = [11 13 2];
%p.ncubes_raw = [10 10 4];
%p.offset_ind = [16 17 0];
p.offset_ind = [12 14 2];
%p.ncubes_raw = [10 10 4];
p.ncubes_raw = [8 8 4];
p.dim_order = [1 2 3];
p.raw_size = 128;
p.isotopic_voxels = false;
%p.strel_offs = [0 0 0];
%p.strel_offs = [0 0 0; -1 0 0; 1 0 0; 0 -1 0; 0 1 0; 1 1 0; 1 -1 0; -1 1 0; -1 -1 0];
p.strel_offs = [0 0 0; -1 0 0; 1 0 0; 0 -1 0; 0 1 0; 1 1 0; 1 -1 0; -1 1 0; -1 -1 0; -2 0 0; 2 0 0; 0 -2 0; 0 2 0];
p.dtype_str = 'uint16';

p.params_meshed = false;
p.merge_percs = 0:0.02:0.2;
p.split_percs = 0:0.08:0.8;

% % generate "realistic" split merger curves.
% p.params_meshed = true;
% alphax=logspace(-2,0,9); alphax=[0.0001 0.001 0.004 alphax];
% %alphax=[0.0001 0.001];
% splitx=[0 0.0001 0.001 0.01 0.03 0.06 0.1:0.1:0.2 0.4:0.2:1];
% % order in nodes_to_gipl: params = {p.merge_percs p.split_percs};
% [alpha, split]=ndgrid(alphax,splitx); 
% merge=alpha.*(alpha+1)./(split+alpha)-alpha;
% % remove duplicates
% tmp=round([split(:) merge(:)],8); [~,n]=unique(tmp,'rows','stable'); 
% sel = true(size(tmp,1),1); sel(n) = false;
% split(sel) = nan; merge(sel) = nan;
% p.split_percs = split;
% p.merge_percs = merge;

%p.hdf5lbls = '/Data/datasets/labels/gt/M0007_33_labels_briggmankl_watkinspv_39x35x7chunks_Forder.h5';
p.hdf5lbls = '/Data/datasets/labels/gt/M0027_11_labels_briggmankl_watkinspv_33x37x7chunks_Forder.h5';

for i = 1:11
  %p.hdf5out = sprintf('/Data/watkinspv/sensitivity/M0007/tmp%d.h5', i);
  p.hdf5out = sprintf('/Data/watkinspv/sensitivity/M0027/tmp%d.h5', i);
  display(p.hdf5out);
  tic; o = knossos_simulate_errors(p); toc
end
