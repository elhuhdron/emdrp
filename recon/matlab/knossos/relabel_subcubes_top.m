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

% with almost no ECS, agglomeration
i = 1;
pdata(i).datah5 = '/Data/datasets/raw/M0027_11_33x37x7chunks_Forder.h5';
% corner chunk
pdata(i).chunk = [12 14 2];
pdata(i).skelin = '/home/watkinspv/Data/M0027_11/M0027_11_dense_skels.186.nml';
% supervoxels, all thresholds and watershed types
pdata(i).lblsh5 = '/Data/pwatkins/agglo/none_aggloall_supervoxels.h5';
pdata(i).name = 'none_agglo';
pdata(i).subgroup = 'agglomeration';
pdata(i).numsegs = 50;
pdata(i).nlabels_attr = 'types_nlabels';
pdata(i).outlbls = '/Data/pwatkins/agglo/none_aggloall_supervoxels_relabel.h5';
pdata(i).prm_start = 1;

% % with ~20% ECS, agglomeration
% i = 2;
% pdata(i).datah5 = '/Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5';
% % corner chunk
% pdata(i).chunk = [16 17 0];
% pdata(i).skelin = '/home/watkinspv/Data/M0007_33/M0007_33_dense_skels.152.nml';
% % supervoxels, all thresholds and watershed types
% pdata(i).lblsh5 = '/Data/datasets/labels/supervoxels/M0007_33_20151001_newestECSall/huge_aggloall_supervoxels.h5';
% pdata(i).name = 'huge_agglo';
% pdata(i).subgroup = 'agglomeration';
% pdata(i).numsegs = 18;
% pdata(i).nlabels_from_attrs = false;



p = struct;  % input parameters independent of dataset

p.knossos_base = [1 1 1];   % knossos starts at 1
%p.knossos_base = [0 0 0];  % knossos starts at 0 (pretty sure no)
p.matlab_base = [1 1 1];  % matlab starts at 1 !!!
p.empty_label = uint32(2^32-1);
p.min_edges = 1;  % only include skeletons with at least this many edges

p.nchunks = [8 8 4];
p.offset = [0 0 32];
p.szsubchunks = [1 1 1];
p.dataset_data = 'data_mag1';
p.dataset_lbls = 'labels';

% optional outputs for debug / validation
p.outlbls = 'outlbls.gipl';

% run relabel subcubes for each dataset
%o = struct;  % meh
o = cell(1,length(pdata));
for i = 1:length(pdata)
  fprintf(1,'\nRunning relabel for "%s"\n\n',pdata(i).name);
  o{i} = relabel_subcubes(p,pdata(i));
end

