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

function knossos_efpl_top_snakemake(output_file_path, lblsh5, h5_raw_data_path, skelin, chunk)

  [outpath, outfile, ~] = fileparts(output_file_path)
  output_file_path = fullfile(outpath, outfile)

  name = 'efpl_placeholder';
  skeleton_mode = true;


  pdata = struct;  % input parameters depending on dataset


  % K0057 watershed somas clean
  i = 1;
  pdata(i).datah5 = h5_raw_data_path;
  pdata(i).chunk = chunk;
  pdata(i).skelin = skelin;
  pdata(i).lblsh5 = lblsh5;
  pdata(i).name = name;
  pdata(i).subgroups = {'with_background'};
  pdata(i).segparam_attr = '';
  pdata(i).segparams = [0.7 0.8 0.9 0.95 0.99 0.995 0.999 0.99925 0.9995 0.99975 0.9999 0.99995 0.99999];
  pdata(i).nlabels_attr = 'types_nlabels';
  pdata(i).node_radius = 20;

  p = struct;  % input parameters independent of dataset

  p.knossos_base = [1 1 1];   % knossos starts at 1, verified
  %p.knossos_base = [0 0 0];  % knossos starts at 0 (NO!!! verified)
  p.matlab_base = [1 1 1];  % matlab starts at 1 !!!
  p.empty_label = uint32(2^32-1);  % type needs to match labels
  p.load_data = false;
  p.load_probs = [];
  %p.load_probs = {'MEM', 'ICS', 'ECS'};
  %p.load_probs = {'MEM'};
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
  p.remove_MEM_ECS_nodes = true;
  % set this to true to remove nodes falling into MEM areas from counting as merged nodes
  p.remove_MEM_merged_nodes = true;



  % xxx - this should have been written to the downsampled hdf5 as an attribute, fix this when fixed in hdf5
  %p.ds_ratio = [3 3 1];
  p.ds_ratio = [1 1 1];

  p.skeleton_mode = skeleton_mode;
  if p.skeleton_mode
    % legacy mode size in chunks and offset as "skip"
    p.nchunks = [8 8 4];
    p.offset = [0 0 0];
    
    % made it possible to specify size and offset normally
    %p.size = [992 992 464];
    %p.offset = [16 16 40];
    
    p.min_edges = 1;  % only include skeletons with at least this many edges
    p.nalloc = 1e6; % for confusion matrix and for stacks
  else
    % new feature that counts split mergers for single nodes that were annotated in soma (cell body) centers.
    % counts over whole large area that might be split between multiple superchunk label files.
    p.nchunks = [48 30 18];
    %p.nchunks = [12 12 12];
    p.supernchunks = [6 6 6];
    p.offset = [0 0 0];
    p.max_nodes = 1;  % only count somas that have this number of nodes or less (always 1???)
    %p.node_radius = 0;  % zero means only the label that is directly under the knossos node (only checks mergers)
    p.superchunk_labels_unique = false;
    p.nalloc = 1e7; % soma mode requires bigger stacks
    p.remove_MEM_merged_nodes = true; % absolutely need this on for this to make sense
  end

  % these could be defined per pdata blocks, but did not see a good reason for this.
  % have to do separate runs if the dataset names are different.
  p.dataset_data = 'data_mag1';
  p.dataset_lbls = 'labels';

  % optional outputs for debug / validation
  p.rawout = false;

  p.outpath = outpath;
  p.outdata = 'outdata.gipl';
  p.outlbls = 'outlbls.gipl';
  p.outprobs = 'outprobs.raw';
  p.nmlout = true;




  % run error free path length for each dataset
  %o = struct;  % meh
  o = cell(1,length(pdata));
  for i = 1:length(pdata)
    fprintf(1,'\nRunning efpl for "%s"\n\n',pdata(i).name);
    o{i} = knossos_efpl(p,pdata(i));
  end

  % save the results
  fprintf('Save output in: %s\n', output_file_path)
  save(output_file_path,'p','pdata','o');
end