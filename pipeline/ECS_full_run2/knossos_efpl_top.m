
% Top level script for calling knossos_efpl.m to calculate path lengths for ECS datasets.

pdata = struct;  % input parameters depending on dataset

i = 1;
pdata(i).datah5 = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5';
pdata(i).chunk = [16 17 0];
%pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.nml';
pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.crop.interp.nml';
pdata(i).lblsh5 = '/Data/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/M0007_supervoxels.h5';
pdata(i).name = 'M0007_wtsh';
pdata(i).subgroups = {'with_background'};
pdata(i).segparam_attr = 'thresholds';
pdata(i).nlabels_attr = 'types_nlabels';

i = 2;
pdata(i).datah5 = '/Data/datasets/raw/M0027_11_33x37x7chunks_Forder.h5';
pdata(i).chunk = [12 14 2];
%pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.nml';
pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.crop.interp.nml';
pdata(i).lblsh5 = '/Data/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/M0027_supervoxels.h5';
pdata(i).name = 'M0027_wtsh';
pdata(i).subgroups = {'with_background'};
pdata(i).segparam_attr = 'thresholds';
pdata(i).nlabels_attr = 'types_nlabels';

i = 3;
pdata(i).datah5 = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5';
pdata(i).chunk = [16 17 0];
%pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.nml';
pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.crop.interp.nml';
pdata(i).lblsh5 = '/Data/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/M0007_supervoxels_agglo.h5';
pdata(i).name = 'M0007_agglo';
pdata(i).subgroups = {'agglomeration'};
pdata(i).segparam_attr = '';
pdata(i).segparams = 0:2:100;
pdata(i).nlabels_attr = 'types_nlabels';

i = 4;
pdata(i).datah5 = '/Data/datasets/raw/M0027_11_33x37x7chunks_Forder.h5';
pdata(i).chunk = [12 14 2];
%pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.nml';
pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.crop.interp.nml';
pdata(i).lblsh5 = '/Data/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/M0027_supervoxels_agglo.h5';
pdata(i).name = 'M0027_agglo';
pdata(i).subgroups = {'agglomeration'};
pdata(i).segparam_attr = '';
pdata(i).segparams = 0:2:100;
pdata(i).nlabels_attr = 'types_nlabels';



p = struct;  % input parameters independent of dataset

p.knossos_base = [1 1 1];   % knossos starts at 1, verified
p.matlab_base = [1 1 1];  % matlab starts at 1 !!!
p.empty_label = uint32(2^32-1);  % type needs to match labels
p.load_data = false;
p.load_probs = [];
p.tol = 1e-5; % for assert sanity checks
p.ds_ratio = [1 1 1]; % need to define here if not defined data h5

% true preserves the total path length, false only counts error-free edges in path length
p.count_half_error_edges = true;
% cutoff for binarizing confusion matrix, need nodes >= this value to be considered overlapping with skel
p.m_ij_threshold = 1;
% number of passes to make over edges for identifying whether an edge is an error or not
% up to four passes over edges are defined as:
%   (1) splits only (2) mergers only (3) split or merger errors (4) split and merger errors
p.npasses_edges = 3;

p.jackknife_resample = false;
p.bernoulli_n_resample = 0;
p.n_resample = 0; % use zero for no resampling
p.p_resample = 0;

% set to < 1 for subsampling sensitivity tests
p.skel_subsample_perc = 1;

% feature to estimate neurite diameters at error free edges
p.estimate_diameters = false;

% usually set these two to true for interpolation, but false for normal
% set this to true to remove non-ICS nodes from polluting the rand error
p.remove_MEM_ECS_nodes = true;
% set this to true to remove nodes falling into MEM areas from counting as merged nodes
p.remove_MEM_merged_nodes = true;

% skeleton_mode true is normal efpl mode, false is "soma-mode"
p.skeleton_mode = true;

% legacy mode size in chunks and offset as "skip"
p.nchunks = [8 8 4];
p.offset = [0 0 32];

p.min_edges = 1;  % only include skeletons with at least this many edges
p.nalloc = 1e6; % for confusion matrix and for stacks

% these could be defined per pdata blocks, but did not see a good reason for this.
% have to do separate runs if the dataset names are different.
p.dataset_data = 'data_mag1';
p.dataset_lbls = 'labels';

% optional outputs for debug / validation
p.rawout = false;
p.outpath = '/home/watkinspv/Downloads';
p.outdata = 'outdata.gipl';
p.outlbls = 'outlbls.gipl';
p.outprobs = 'outprobs.raw';
p.nmlout = false;



% run error free path length for each dataset
o = cell(1,length(pdata));
for i = 1:length(pdata)
  fprintf(1,'\nRunning efpl for "%s"\n\n',pdata(i).name);
  o{i} = knossos_efpl(p,pdata(i));
end

% save the results
save('~/Data/efpl/vgg3pool64_ECS_full_run2_crop_interp','p','pdata','o');
