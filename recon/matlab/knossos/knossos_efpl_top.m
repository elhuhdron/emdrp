% The MIT License (MIT)
% 
% Copyright (c) 2016 Paul Watkins, National Institutes of Health / NINDS
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

% Top level script for calling knossos_efpl.m to calculate path lengths for different datasets.

pdata = struct;  % input parameters depending on dataset

% START - main dataset formats / locations , do not delete

% % with almost no ECS
% i = 1;
% pdata(i).datah5 = '/Data/datasets/raw/M0027_11_33x37x7chunks_Forder.h5';
% % corner chunk
% pdata(i).chunk = [12 14 2];
% % % ground truth
% % pdata(i).lblsh5 = '/home/watkinspv/Data/M0027_11/M0027_11_labels_briggmankl_watkinspv_33x37x7chunks_Forder.h5';
% % % labeled chunks
% % pdata(i).chunk = [16 17 4];
% % pdata(i).chunk = [13 20 3];
% % pdata(i).chunk = [13 15 3];
% % pdata(i).chunk = [18 15 3];
% % pdata(i).chunk = [18 20 3];
% % pdata(i).chunk = [18 20 4];
% pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.nml';
% % supervoxels, all thresholds and watershed types
% pdata(i).lblsh5 = '/Data/datasets/labels/supervoxels/newestECSall_20151001/none_supervoxels.h5';
% %pdata(i).probh5 = '/Data/datasets/labels/supervoxels/newestECSall_20151001/none_probs.h5';
% pdata(i).name = 'none';
% pdata(i).subgroups = {'with_background'};
% pdata(i).segparam_attr = 'thresholds';
% pdata(i).nlabels_attr = 'types_nlabels';
% 
% % with ~20% ECS
% i = 2;
% pdata(i).datah5 = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5';
% % corner chunk
% pdata(i).chunk = [16 17 0];
% % % ground truth
% % pdata(i).lblsh5 = '/home/watkinspv/Data/M0007_33/M0007_33_labels_briggmankl_39x35x7chunks_Forder.h5';
% % % labeled chunks
% % pdata(i).chunk = [19 22 2];
% % pdata(i).chunk = [17,19,2];
% % pdata(i).chunk = [17,23,1];
% % pdata(i).chunk = [22,23,1];
% % pdata(i).chunk = [22,18,1];
% % pdata(i).chunk = [22,23,2];
% % pdata(i).chunk = [19,22,2];
% pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.nml';
% % supervoxels, all thresholds and watershed types
% pdata(i).lblsh5 = '/Data/datasets/labels/supervoxels/newestECSall_20151001/huge_supervoxels.h5';
% %pdata(i).probh5 = '/Data/datasets/labels/supervoxels/newestECSall_20151001/huge_probs.h5';
% pdata(i).name = 'huge';
% pdata(i).subgroups = {'with_background'};
% pdata(i).segparam_attr = 'thresholds';
% pdata(i).nlabels_attr = 'types_nlabels';

% % with almost no ECS
% i = 1;
% pdata(i).datah5 = '/Data/datasets/raw/M0027_11_33x37x7chunks_Forder.h5';
% pdata(i).chunk = [12 14 2];
% %pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.nml';
% pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.crop.nml';
% %pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.interp.nml';
% pdata(i).lblsh5 = '/Data/datasets/labels/supervoxels/newestECSall_20151001/none_supervoxels.h5';
% pdata(i).probh5 = '/Data/datasets/labels/supervoxels/newestECSall_20151001/none_probs.h5';
% pdata(i).name = 'none';
% pdata(i).subgroups = {'with_background'};
% pdata(i).segparam_attr = 'thresholds';
% pdata(i).nlabels_attr = 'types_nlabels';
% 
% % with ~20% ECS
% i = 2;
% pdata(i).datah5 = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5';
% pdata(i).chunk = [16 17 0];
% %pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.nml';
% pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.crop.nml';
% %pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.interp.nml';
% pdata(i).lblsh5 = '/Data/datasets/labels/supervoxels/newestECSall_20151001/huge_supervoxels.h5';
% pdata(i).probh5 = '/Data/datasets/labels/supervoxels/newestECSall_20151001/huge_probs.h5';
% pdata(i).name = 'huge';
% pdata(i).subgroups = {'with_background'};
% pdata(i).segparam_attr = 'thresholds';
% pdata(i).nlabels_attr = 'types_nlabels';

% % k0725
% i = 1;
% pdata(i).datah5 = '/Data/datasets/raw/k0725.h5';
% pdata(i).chunk = [8 9 3];
% pdata(i).skelin = '/Data/datasets/skeletons/skeleton-kara-mod.054.interp.nml';
% pdata(i).lblsh5 = '/Data/watkinspv/full_datasets/neon/vgg3pool_k0725/k0725_supervoxels.h5';
% %pdata(i).probh5 = '/Data/datasets/labels/supervoxels/newestECSall_20151001/huge_probs.h5';
% pdata(i).name = 'k0725';
% pdata(i).subgroups = {'with_background'};
% pdata(i).segparam_attr = 'thresholds';
% pdata(i).nlabels_attr = 'types_nlabels';

% % with almost no ECS, agglomeration
% i = 1;
% pdata(i).datah5 = '/Data/datasets/raw/M0027_11_33x37x7chunks_Forder.h5';
% % corner chunk
% pdata(i).chunk = [12 14 2];
% pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.nml';
% % supervoxels, all thresholds and watershed types
% pdata(i).lblsh5 = '/Data/watkinspv/agglo/none_aggloall_rf_75iter2p_reduced_supervoxels_fixed.h5';
% pdata(i).name = 'none_agglo';
% pdata(i).subgroups = {'agglomeration'};
% pdata(i).segparam_attr = '';
% pdata(i).segparams = 1:75;
% pdata(i).nlabels_attr = 'types_nlabels';

% % with ~20% ECS, agglomeration
% i = 1;
% pdata(i).datah5 = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5';
% % corner chunk
% pdata(i).chunk = [16 17 0];
% pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.interp.nml';
% % supervoxels, all thresholds and watershed types
% pdata(i).lblsh5 = '/Data/watkinspv/full_datasets/neon/mbfergus32all/ovlp8/agglo/huge_agglo_twopass_fixed.h5';
% pdata(i).name = 'huge_mbf32_2pagg8';
% pdata(i).subgroups = {'agglomeration'};
% pdata(i).segparam_attr = '';
% pdata(i).segparams = 1:75;
% pdata(i).nlabels_attr = 'types_nlabels';

% % k0725 agglomeration
% i = 1;
% pdata(i).datah5 = '/Data/datasets/raw/k0725.h5';
% pdata(i).chunk = [8 9 3];
% pdata(i).skelin = '/Data/datasets/skeletons/skeleton-kara-mod.054.interp.nml';
% pdata(i).lblsh5 = '/Data/watkinspv/full_datasets/neon/vgg3pool_k0725/k0725_supervoxels.h5';
% % supervoxels, all thresholds and watershed types
% pdata(i).lblsh5 = '/Data/watkinspv/agglo/k0725_vgg3pool_aggloall_rf_75iter2p_medium_supervoxels_fixed.h5';
% pdata(i).name = 'k0725 agglo';
% pdata(i).subgroups = {'agglomeration'};
% pdata(i).segparam_attr = '';
% pdata(i).segparams = 1:75;
% pdata(i).nlabels_attr = 'types_nlabels';

% % with almost no ECS
% i = 3;
% pdata(i).datah5 = '/Data/datasets/raw/M0027_11_33x37x7chunks_Forder.h5';
% pdata(i).probh5 = '/Data/watkinspv/full_datasets/newestECSall_xyzonly/none_probs.h5';
% pdata(i).chunk = [12 14 2];
% %pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.nml';
% pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.crop.nml';
% %pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.interp.nml';
% %pdata(i).lblsh5 = '/Data/datasets/labels/supervoxels/newestECSall_20151001/none_supervoxels.h5';
% pdata(i).lblsh5 = '/Data/watkinspv/full_datasets/newestECSall_xyzonly/none_supervoxels.h5';
% pdata(i).name = 'none xyz';
% pdata(i).subgroups = {'with_background'};
% pdata(i).segparam_attr = 'thresholds';
% pdata(i).nlabels_attr = 'types_nlabels';
% 
% % with ~20% ECS
% i = 4;
% pdata(i).datah5 = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5';
% pdata(i).probh5 = '/Data/watkinspv/full_datasets/newestECSall_xyzonly/huge_probs.h5';
% pdata(i).chunk = [16 17 0];
% %pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.nml';
% pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.crop.nml';
% %pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.interp.nml';
% %pdata(i).lblsh5 = '/Data/datasets/labels/supervoxels/newestECSall_20151001/huge_supervoxels.h5';
% pdata(i).lblsh5 = '/Data/watkinspv/full_datasets/newestECSall_xyzonly/huge_supervoxels.h5';
% pdata(i).name = 'huge xyz';
% pdata(i).subgroups = {'with_background'};
% pdata(i).segparam_attr = 'thresholds';
% pdata(i).nlabels_attr = 'types_nlabels';

% END - main dataset formats / locations , do not delete





% % with ~20% ECS
% i = 1;
% pdata(i).datah5 = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5';
% pdata(i).chunk = [16 17 0];
% %pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.nml';
% pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.interp.nml';
% pdata(i).lblsh5 = '/Data/watkinspv/full_datasets/neon/mbfergus32all/huge_supervoxels.h5';
% pdata(i).probh5 = '/Data/watkinspv/full_datasets/neon/mbfergus32all/huge_probs.h5';
% pdata(i).name = 'huge_mbf32';
% pdata(i).subgroups = {'with_background'};
% pdata(i).segparam_attr = 'thresholds';
% pdata(i).nlabels_attr = 'types_nlabels';
% 
% % with almost no ECS
% i = 2;
% pdata(i).datah5 = '/Data/datasets/raw/M0027_11_33x37x7chunks_Forder.h5';
% pdata(i).chunk = [12 14 2];
% %pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.nml';
% pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.interp.nml';
% pdata(i).lblsh5 = '/Data/watkinspv/full_datasets/neon/mbfergus32all/none_supervoxels.h5';
% pdata(i).probh5 = '/Data/watkinspv/full_datasets/neon/mbfergus32all/none_probs.h5';
% pdata(i).name = 'none_mbf32';
% pdata(i).subgroups = {'with_background'};
% %pdata(i).segparam_attr = 'thresholds';
% pdata(i).segparam_attr = '';
% pdata(i).segparams = [0.5 0.6 0.7 0.8 0.9 0.95 0.99 0.995 0.999 0.99925 0.9995 0.99975 0.9999 0.99999000];
% pdata(i).nlabels_attr = 'types_nlabels';




% % for kevin's talk 20160915
% % with ~20% ECS
% i = 1;
% pdata(i).datah5 = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5';
% pdata(i).chunk = [16 17 0];
% %pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.nml';
% pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.interp.nml';
% pdata(i).lblsh5 = '/Data/watkinspv/full_datasets/neon/mbfergus32all/huge_supervoxels.h5';
% %pdata(i).probh5 = '/Data/watkinspv/full_datasets/neon/mbfergus32all/huge_probs.h5';
% pdata(i).name = 'huge_mbf32';
% pdata(i).subgroups = {'with_background'};
% pdata(i).segparam_attr = 'thresholds';
% pdata(i).nlabels_attr = 'types_nlabels';
% 
% % with ~20% ECS, agglomeration
% i = 2;
% pdata(i).datah5 = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5';
% % corner chunk
% pdata(i).chunk = [16 17 0];
% pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.interp.nml';
% % supervoxels, all thresholds and watershed types
% pdata(i).lblsh5 = '/Data/watkinspv/agglo/huge_vgg4pool64_aggloall_rf_75iter2p_small_supervoxels_fixed.h5';
% pdata(i).name = 'huge_vgg4_agglo';
% pdata(i).subgroups = {'agglomeration'};
% pdata(i).segparam_attr = '';
% pdata(i).segparams = 1:75;
% pdata(i).nlabels_attr = 'types_nlabels';
% 
% % with ~20% ECS
% i = 3;
% pdata(i).datah5 = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5';
% pdata(i).chunk = [16 17 0];
% %pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.nml';
% pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.interp.nml';
% pdata(i).lblsh5 = '/Data/datasets/labels/supervoxels/newestECSall_20151001/huge_supervoxels.h5';
% %pdata(i).probh5 = '/Data/datasets/labels/supervoxels/newestECSall_20151001/huge_probs.h5';
% pdata(i).name = 'huge';
% pdata(i).subgroups = {'with_background'};
% pdata(i).segparam_attr = 'thresholds';
% pdata(i).nlabels_attr = 'types_nlabels';
% 
% % with ~20% ECS
% i = 4;
% pdata(i).datah5 = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5';
% pdata(i).probh5 = '/Data/watkinspv/full_datasets/newestECSall_xyzonly/huge_probs.h5';
% pdata(i).chunk = [16 17 0];
% %pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.nml';
% pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.interp.nml';
% %pdata(i).lblsh5 = '/Data/datasets/labels/supervoxels/newestECSall_20151001/huge_supervoxels.h5';
% pdata(i).lblsh5 = '/Data/watkinspv/full_datasets/newestECSall_xyzonly/huge_supervoxels.h5';
% pdata(i).name = 'huge xyz';
% pdata(i).subgroups = {'with_background'};
% pdata(i).segparam_attr = 'thresholds';
% pdata(i).nlabels_attr = 'types_nlabels';




% % sensitivity generate "realistic" split merger curves.
% alphax=logspace(-2,0,9); alphax=[0.0001 0.001 0.004 alphax];
% %alphax=[0.0001 0.001];
% splitx=[0 0.0001 0.001 0.01 0.03 0.06 0.1:0.1:0.2 0.4:0.2:1];
% % order in nodes_to_gipl: params = {p.merge_percs p.split_percs p.remove_percs};
% [alpha, split]=ndgrid(alphax,splitx); 
% merge=alpha.*(alpha+1)./(split+alpha)-alpha;
% nruns = 11;
% for x = 1:nruns
%   strb = sprintf('huge%d',x);
%   for y = 1:length(alphax)
%     % with ~20% ECS
%     i = length(alphax)*(x-1) + y;
%     pdata(i).datah5 = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5';
%     pdata(i).chunk = [16 17 0];
%     pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.nml';
%     %pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.interp.nml';
%     pdata(i).lblsh5 = sprintf('/Data/watkinspv/sensitivity/M0007/tmp%d.h5',x);
%     pdata(i).name = [strb sprintf(' %g',alphax(y))];
%     pdata(i).subgroups = {'perc_merge_split'};
%     pdata(i).segparam_attr = '';
%     pdata(i).segparams = {round(merge(y,:),8) round(split(y,:),8)};
%     pdata(i).nlabels_attr = '';
%   end
% end

% % sensitivity normal split merger tiling.
% merge_percs = 0:0.02:0.2;
% split_percs = 0:0.08:0.8;
% [merge, split]=ndgrid(merge_percs,split_percs); 
% nruns = 11;
% for x = 1:nruns
%   strb = sprintf('huge%d',x);
%   %strb = sprintf('none%d',x);
% 
%   for y = 1:length(merge_percs)
%     i = length(merge_percs)*(x-1) + y;
% 
%     pdata(i).datah5 = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5';
%     %pdata(i).datah5 = '/Data/datasets/raw/M0027_11_33x37x7chunks_Forder.h5';
% 
%     pdata(i).chunk = [16 17 0];
%     %pdata(i).chunk = [12 14 2];
%     
%     pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.nml';
%     %pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.crop.nml';
%     %pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.interp.nml';
%     %pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.nml';
%     %pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.crop.nml';
%     %pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.interp.nml';
% 
%     pdata(i).lblsh5 = sprintf('/Data/watkinspv/sensitivity/M0007/tmp%d.h5',x);
%     %pdata(i).lblsh5 = sprintf('/Data/watkinspv/sensitivity/M0027/tmp%d.h5',x);
% 
%     pdata(i).name = [strb sprintf(' %g',merge_percs(y))];
%     pdata(i).subgroups = {'perc_merge_split'};
%     pdata(i).segparam_attr = '';
%     pdata(i).segparams = {round(merge(y,:),8) round(split(y,:),8)};
%     pdata(i).nlabels_attr = '';
%   end
% end

% K0057 agglomeration somas clean
i = 1;
pdata(i).datah5 = '/Data/datasets/raw/K0057_D31_dsx3y3z1.h5';
%pdata(i).chunk = [8 9 3];
% beg and end for superchunked labels (soma mode) are inclusive, matlab-style
pdata(i).chunk = [2 8 1];
pdata(i).skelin = '/Data/datasets/skeletons/K0057-D31-somas.365.xml';
pdata(i).lblsh5 = '/Data/watkinspv/full_datasets/neon/mfergus32all_K0057_ds3_run2/clean';
pdata(i).name = 'K0057 clean';
pdata(i).subgroups = {'agglomeration'};
pdata(i).segparam_attr = '';
pdata(i).segparams = 39:48;
pdata(i).nlabels_attr = 'types_nlabels';

% K0057 agglomeration somas agglo
i = 2;
pdata(i).datah5 = '/Data/datasets/raw/K0057_D31_dsx3y3z1.h5';
%pdata(i).chunk = [8 9 3];
% beg and end for superchunked labels (soma mode) are inclusive, matlab-style
pdata(i).chunk = [2 8 1];
pdata(i).skelin = '/Data/datasets/skeletons/K0057-D31-somas.365.xml';
pdata(i).lblsh5 = '/Data/watkinspv/full_datasets/neon/mfergus32all_K0057_ds3_run2/agglo';
pdata(i).name = 'K0057 clean';
pdata(i).subgroups = {'agglomeration'};
pdata(i).segparam_attr = '';
pdata(i).segparams = 1:48;
pdata(i).nlabels_attr = 'types_nlabels';





p = struct;  % input parameters independent of dataset

p.knossos_base = [1 1 1];   % knossos starts at 1, verified
%p.knossos_base = [0 0 0];  % knossos starts at 0 (NO!!! verified)
p.matlab_base = [1 1 1];  % matlab starts at 1 !!!
p.empty_label = uint32(2^32-1);
p.load_data = false;
p.load_probs = [];
%p.load_probs = {'MEM', 'ICS', 'ECS'};
%p.load_probs = {'MEM'};
p.nalloc = 1e6; % for confusion matrix and for stacks
p.tol = 1e-5; % for assert sanity checks

% true preserves the total path length, false only counts error-free edges in path length
p.count_half_error_edges = true;
% cutoff for binarizing confusion matrix, need nodes >= this value to be considered overlapping with skel
p.m_ij_threshold = 1;
% number of passes to make over edges for identifying whether an edge is an error or not
% up to four passes over edges are defined as:
%   (1) splits only (2) mergers only (3) split or merger errors (4) split and merger errors
p.npasses_edges = 3;

p.jackknife_resample = false;
p.bernoulli_n_resample = 206;   % 95% of 217 (nskels is 220, 217 for two none/huge)
p.n_resample = 0; % use zero for no resampling
p.p_resample = 0;
% p.n_resample = 1000; 
% p.p_resample = 0.01;

% set to < 1 for subsampling sensitivity tests
p.skel_subsample_perc = 1;
%p.skel_subsample_perc = 0.2;

% feature to estimate neurite diameters at error free edges
p.estimate_diameters = false;

% usually set these two to true for interpolation, but false for normal
% set this to true to remove non-ICS nodes from polluting the rand error
p.remove_MEM_ECS_nodes = false;
% set this to true to remove nodes falling into MEM areas from counting as merged nodes
p.remove_MEM_merged_nodes = false;



p.skeleton_mode = false;
if p.skeleton_mode
  p.nchunks = [8 8 4];
  %p.nchunks = [6 6 3];
  p.offset = [0 0 32];
  %p.offset = [0 0 0];
  p.min_edges = 1;  % only include skeletons with at least this many edges
else
  % new feature that counts split mergers for single nodes that were annotated in soma (cell body) centers.
  % counts over whole large area that might be split between multiple superchunk label files.
  p.nchunks = [48 30 18];
  p.supernchunks = [6 6 6];
  p.offset = [0 0 0];
  p.max_nodes = 1;  % only count somas that have this number of nodes or less (always 1???)
  p.node_radius = 50;
  
  % xxx - this should have been written to the downsampled hdf5 as an attribute, fix this when fixed in hdf5
  p.ds_ratio = [3 3 1];
end

% these could be defined per pdata blocks, but did not see a good reason for this.
% have to do separate runs if the dataset names are different.
%p.dataset_data = 'data_mag1';
p.dataset_data = 'data_mag_x3y3z1';
p.dataset_lbls = 'labels';

% optional outputs for debug / validation
p.rawout = false;
p.outpath = '/Data/pwatkins/tmp/knout';
p.outdata = 'outdata.gipl';
p.outlbls = 'outlbls.gipl';
p.outprobs = 'outprobs.raw';
p.nmlout = false;




% run error free path length for each dataset
%o = struct;  % meh
o = cell(1,length(pdata));
for i = 1:length(pdata)
  fprintf(1,'\nRunning efpl for "%s"\n\n',pdata(i).name);
  o{i} = knossos_efpl(p,pdata(i));
end

% save the results
%save('/home/watkinspv/Data/efpl/efpl_interp_k0725_agglo','p','pdata','o');
%save('/home/watkinspv/Data/efpl/efpl_huge_sensitivity_crop_big_sample0p2.mat','p','pdata','o');
%save('/home/watkinspv/Data/efpl/efpl_paper_crop_diameters.mat','p','pdata','o');
save('/home/watkinspv/Downloads/tmp.mat','p','pdata','o');
