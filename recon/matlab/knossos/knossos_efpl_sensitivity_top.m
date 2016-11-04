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
% i = 2;
% pdata(i).datah5 = '/Data/datasets/raw/M0027_11_33x37x7chunks_Forder.h5';
% pdata(i).chunk = [12 14 2];
% %pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.nml';
% pdata(i).skelin = '/Data/datasets/skeletons/M0027_11_dense_skels.186.interp.nml';
% pdata(i).lblsh5 = '/Data/watkinspv/full_datasets/neon/mbfergus16all_ds2/none_supervoxels.h5';
% pdata(i).name = 'none_mbf16ds';
% pdata(i).subgroups = {'with_background'};
% pdata(i).segparam_attr = 'thresholds';
% %pdata(i).segparam_attr = '';
% %pdata(i).segparams = [0.5 0.6 0.7 0.8 0.9 0.95 0.99 0.995 0.999 0.99925 0.9995 0.99975 0.9999 0.99999000];
% pdata(i).nlabels_attr = 'types_nlabels';

% xxx - probably just merge this back into knossos_efpl_ws_top.m

% generate "realistic" split merger curves.
p.params_meshed = true;
alphax=logspace(-2,0,9); alphax=[0.0001 0.001 0.004 alphax];
%alphax=[0.0001 0.001];
splitx=[0 0.0001 0.001 0.01 0.03 0.06 0.1:0.1:0.2 0.4:0.2:1];
% order in nodes_to_gipl: params = {p.merge_percs p.split_percs p.remove_percs};
[alpha, split]=ndgrid(alphax,splitx); 
merge=alpha.*(alpha+1)./(split+alpha)-alpha;

nruns = 11;
for x = 1:nruns
  strb = sprintf('huge%d',x);
  for y = 1:length(alphax)
    % with ~20% ECS
    i = length(alphax)*(x-1) + y;
    pdata(i).datah5 = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5';
    pdata(i).chunk = [16 17 0];
    pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.nml';
    %pdata(i).skelin = '/Data/datasets/skeletons/M0007_33_dense_skels.152.interp.nml';
    pdata(i).lblsh5 = sprintf('/Data/watkinspv/sensitivity/M0007/tmp%d.h5',x);
    pdata(i).name = [strb sprintf(' %g',alphax(y))];
    pdata(i).subgroups = {'perc_merge_split'};
    pdata(i).segparam_attr = '';
    pdata(i).segparams = {round(merge(y,:),8) round(split(y,:),8)};
    pdata(i).nlabels_attr = '';
  end
end




p = struct;  % input parameters independent of dataset

p.knossos_base = [1 1 1];   % knossos starts at 1
%p.knossos_base = [0 0 0];  % knossos starts at 0 (pretty sure no)
p.matlab_base = [1 1 1];  % matlab starts at 1 !!!
p.empty_label = uint32(2^32-1);
p.min_edges = 1;  % only include skeletons with at least this many edges
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

% usually set these two to true for interpolation, but false for normal
% set this to true to remove non-ICS nodes from polluting the rand error
p.remove_MEM_ECS_nodes = false;
% set this to true to remove nodes falling into MEM areas from counting as merged nodes
p.remove_MEM_merged_nodes = false;



p.nchunks = [8 8 4];
%p.offset = [0 0 32];
p.offset = [0 0 0];
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
save('/home/watkinspv/Data/efpl/efpl_sensitivity_alpha_big.mat','p','pdata','o');
