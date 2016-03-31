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
% pdata(i).skelin = '/home/watkinspv/Data/M0027_11/M0027_11_dense_skels.186.nml';
% % supervoxels, all thresholds and watershed types
% %pdata(i).lblsh5 = '/Data/pwatkins/full_datasets/newestECSall/20150903/none_supervoxels.h5';
% pdata(i).lblsh5 = '/Data/pwatkins/full_datasets/newestECSall/20151001/none_supervoxels.h5';
% %pdata(i).probh5 = '/Data/pwatkins/full_datasets/newestECSall/none_probs.h5';
% pdata(i).name = 'none';
% pdata(i).subgroups = {'with_background'};
% pdata(i).segparam_attr = 'thresholds';
% pdata(i).nlabels_attr = 'types_nlabels';

% % with almost no ECS, agglomeration
% i = 2;
% pdata(i).datah5 = '/Data/datasets/raw/M0027_11_33x37x7chunks_Forder.h5';
% % corner chunk
% pdata(i).chunk = [12 14 2];
% pdata(i).skelin = '/home/watkinspv/Data/M0027_11/M0027_11_dense_skels.186.nml';
% % supervoxels, all thresholds and watershed types
% pdata(i).lblsh5 = '/Data/pwatkins/agglo/none_aggloall_supervoxels_relabel.h5';
% pdata(i).name = 'none_agglo';
% pdata(i).subgroups = {'agglomeration'};
% pdata(i).segparam_attr = '';
% pdata(i).nlabels_attr = '';
% pdata(i).segparams = 1:50;

% with ~20% ECS
i = 1;
pdata(i).datah5 = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5';
% corner chunk
pdata(i).chunk = [16 17 0];
% % ground truth
% pdata(i).lblsh5 = '/home/watkinspv/Data/M0007_33/M0007_33_labels_briggmankl_39x35x7chunks_Forder.h5';
% % labeled chunks
% pdata(i).chunk = [19 22 2];
% pdata(i).chunk = [17,19,2];
% pdata(i).chunk = [17,23,1];
% pdata(i).chunk = [22,23,1];
% pdata(i).chunk = [22,18,1];
% pdata(i).chunk = [22,23,2];
% pdata(i).chunk = [19,22,2];
pdata(i).skelin = '/home/watkinspv/Data/M0007_33/M0007_33_dense_skels.152.nml';
% supervoxels, all thresholds and watershed types
pdata(i).lblsh5 = '/Data/pwatkins/full_datasets/newestECSall/20151001/huge_supervoxels.h5';
%pdata(i).probh5 = '/Data/pwatkins/full_datasets/newestECSall/huge_probs.h5';
pdata(i).name = 'huge';
pdata(i).subgroups = {'with_background'};
pdata(i).segparam_attr = 'thresholds';
pdata(i).nlabels_attr = 'types_nlabels';

% with ~20% ECS, agglomeration
i = 2;
pdata(i).datah5 = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5';
% corner chunk
pdata(i).chunk = [16 17 0];
pdata(i).skelin = '/home/watkinspv/Data/M0007_33/M0007_33_dense_skels.152.nml';
% supervoxels, all thresholds and watershed types
pdata(i).lblsh5 = '/Data/pwatkins/agglo/huge_aggloall_rf60_supervoxels.h5';
pdata(i).name = 'huge_agglo';
pdata(i).subgroups = {'agglomeration'};
pdata(i).segparam_attr = '';
pdata(i).nlabels_attr = '';
pdata(i).segparams = 1:50;



p = struct;  % input parameters independent of dataset

p.knossos_base = [1 1 1];   % knossos starts at 1
%p.knossos_base = [0 0 0];  % knossos starts at 0 (pretty sure no)
p.matlab_base = [1 1 1];  % matlab starts at 1 !!!
p.empty_label = uint32(2^32-1);
p.min_edges = 1;  % only include skeletons with at least this many edges
p.load_data = false;
p.load_probs = [];
%p.load_probs = {'MEM', 'ICS', 'ECS'};
p.nalloc = 1e5; % for confusion matrix and for stacks
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

p.nchunks = [8 8 4];
p.offset = [0 0 32];
p.dataset_data = 'data_mag1';
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
save('/home/watkinspv/Data/agglo/efpl_huge_rf60.mat','p','pdata','o');
