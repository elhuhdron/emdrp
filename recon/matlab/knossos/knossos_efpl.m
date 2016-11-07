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

% function to calculate error free path length for labeled data versus
% skeletonized data. this script uses two methods that are related:
%
% first calculate number of splits and mergers using node-based confusion matrix.
% this is the simplest method that only requires visiting each edge (and 
%   connected nodes at that edge) and not in any particular order.
% calculate the rand error based on confusion matrix.
% during this pass also tally which edges are split and which nodes are merged.
%
% second, edges split / nodes merged info from the first pass and traverse edges 
%   of each tree along the connected paths. error free path length is calculated
%   separately for splits and mergers as total path length along edges before an
%   error occurs. optionally had halve path length to current and next efpls when
%   an error is encountered at an edge.
%
% thing is knossos-speak for skeleton

function o = knossos_efpl(p, pdata)
o = struct;

fprintf(1,'getting dataset info from h5 files\n');
inf = h5info(pdata.datah5);
ind = find(strcmp({inf.Datasets.Name},p.dataset_data)); % find the dataset in the info struct
o.chunksize = inf.Datasets(ind).ChunkSize;
o.datasize = inf.Datasets(ind).Dataspace(1).Size;
o.scale = h5readatt(pdata.datah5,['/' p.dataset_data],'scale')';

% allow for ndgrid parameter spaces, being used in combination with sensitivity analysis.
% for that mode, this output needs to be set as a cell array of the ndgrid of parameter space.
o.segparams = 1;
% either load the parameter space from the attributes of voxel_type, or they are passed in as parameters.
if ~isempty(pdata.segparam_attr)
  % get the parameter space from the labels h5 file
  o.thresholds = h5readatt(pdata.lblsh5,['/' 'voxel_type'],pdata.segparam_attr)';
else
  if iscell(pdata.segparams)
    o.thresholds = 1:length(pdata.segparams{1}(:));
    o.segparams = cellfun(@(x) x(:), pdata.segparams,'UniformOutput',false);
  else
    o.thresholds = pdata.segparams;
  end
end

% % to select only a subset of available parameters to analyze
% use_Tmins = isfield(p, 'use_Tmins');
% if use_Tmins, o.Tmins = o.Tmins(ismember(o.Tmins,p.use_Tmins)); end
% use_thresholds = isfield(p, 'use_thresholds');
% % mmmm, floating point
% if use_thresholds, o.thresholds = o.thresholds(ismember(round(1e6*o.thresholds),round(1e6*p.use_thresholds))); end

% other calculated values based on params
o.loadsize = p.nchunks.*o.chunksize - p.offset;
o.loadcorner = pdata.chunk.*o.chunksize + p.offset;
o.nthresholds = length(o.thresholds);
o.nparams = o.nthresholds;
assert( o.nparams > 1 );
assert( p.m_ij_threshold > 0 );

if p.load_data
  fprintf(1,'reading data\n'); t = now;
  Vdata = h5read(pdata.datah5,['/' p.dataset_data],o.loadcorner+p.matlab_base,o.loadsize);
  display(sprintf('\tdone in %.3f s',(now-t)*86400));
end

if ~isempty(p.load_probs)
  Vprobs = zeros([o.loadsize length(p.load_probs)],'single');
  for i = 1:length(p.load_probs)
    fprintf(1,'reading probabilities %s\n',p.load_probs{i}); t = now;
    Vprobs(:,:,:,i) = h5read(pdata.probh5,['/' p.load_probs{i}],o.loadcorner+p.matlab_base,o.loadsize);
    display(sprintf('\tdone in %.3f s',(now-t)*86400));
  end
end

if p.rawout
  % for debug, export volumes as gipl or raw
  fh = fopen(fullfile(p.outpath,p.outprobs), 'wb'); fwrite(fh,Vprobs(:),'float32'); fclose(fh);
  gipl_write_volume(Vdata,fullfile(p.outpath,p.outdata),o.scale);
end





fprintf(1,'reading nml file\n'); t = now;
[o.info, meta, ~] = knossos_read_nml(pdata.skelin);
display(sprintf('\tdone in %.3f s',(now-t)*86400));

% voxel scale from Knossos file (should match hdf5)
o.scaleK = [meta.scale.x meta.scale.y meta.scale.z];
assert( all((o.scale - o.scaleK) < p.tol) );

% convert to struct array for indexing, reorder by thingID
o.info = [o.info{:}]; [~,i] = sort([o.info.thingID]); o.info = o.info(i);

% get number of edges and nodes and total skeleton count (nThings) from nml data
o.nedges = cellfun('size',{o.info.edges},1); o.nnodes = cellfun('size',{o.info.nodes},1);
o.nThings = length(o.info); % total skeletons in nml file

% these are purely informative checks to get the bounds for the skeleton nodes within the dataset.
allnodes = vertcat(o.info.nodes);
maxx = max(allnodes(:,1)); minx = min(allnodes(:,1));
maxy = max(allnodes(:,2)); miny = min(allnodes(:,2));
maxz = max(allnodes(:,3)); minz = min(allnodes(:,3));
o.minnodes = [minx miny minz]; o.maxnodes = [maxx maxy maxz];





% read out labels at some threshold, this first step is just to get nodes/edges
%   within labeled supervoxel area (within volume and not empty label) and path lengths.
% the labeled supervoxel area and skeleton path lengths are not a function of threshold.
fprintf(1,'reading labels at thr %.8f\n', o.thresholds(1)); t = now;
if iscell(o.segparams)
  dset = sprintf('/%s/%s%s',strjoin(pdata.subgroups,'/'),...
    sprintf('%.8f/', cellfun(@(x) x(1), o.segparams)),p.dataset_lbls);
else
  dset = sprintf('/%s/%.8f/%s',strjoin(pdata.subgroups,'/'),o.thresholds(1),p.dataset_lbls);
  % xxx - this was for older matlabs? probably remove
  %dset = sprintf('/%s/%.8f/%s',strrep(reshape(char(pdata.subgroups)',1,[]), ' ','/'),o.thresholds(1),p.dataset_lbls);
end
Vlbls = h5read(pdata.lblsh5,dset,o.loadcorner+p.matlab_base,o.loadsize);
display(sprintf('\tdone in %.3f s',(now-t)*86400));

if p.rawout
  % for debug, export volumes as gipl or raw
  gipl_write_volume(Vlbls,fullfile(p.outpath,p.outlbls),o.scale);
end

fprintf(1,'iterating skeletons to get connected nodes and path lengths\n');
o.empty_things = (o.nedges < p.min_edges); o.path_length = zeros(1,o.nThings); o.edge_length = cell(1,o.nThings);
o.edges_use = cell(1,o.nThings); o.nodes_use = cell(1,o.nThings); 
o.path_length_use = zeros(1,o.nThings); o.edge_length_use = cell(1,o.nThings);
fprintf(1,'total %d things, not including %d with < %d edges, total included %d\n',o.nThings,sum(o.empty_things),...
  p.min_edges,sum(~o.empty_things));
for n=1:o.nThings
  if o.empty_things(n)
    %fprintf(1,'thingID %d has %d edges < %d min and %d nodes, not including\n',n,nedges(n),min_edges,nnodes(n));
    continue;
  end
  
  % iterate over edges
  o.edge_length{n} = zeros(1,o.nedges(n));
  o.nodes_use{n} = false(1,o.nnodes(n)); o.edges_use{n} = false(1,o.nedges(n)); 
  o.edge_length_use{n} = zeros(1,o.nedges(n));
  for e=1:o.nedges(n)
    n1 = o.info(n).edges(e,1); n2 = o.info(n).edges(e,2); % current nodes involved in this edge
    
    % update overall metrics for skeleton
    o.edge_length{n}(e) = sqrt(sum(((o.info(n).nodes(n1,1:3)-o.info(n).nodes(n2,1:3)).*o.scaleK).^2));
    o.path_length(n) = o.path_length(n) + o.edge_length{n}(e);
    
    % get the supervoxel label at both node points
    n1pt = round(o.info(n).nodes(n1,1:3) - p.knossos_base);
    n2pt = round(o.info(n).nodes(n2,1:3) - p.knossos_base);
    
    % skip this edge if either node is out of bounds of the supervoxel area
    n1subs = n1pt-o.loadcorner+p.matlab_base;
    if any(n1subs < 1) || any(n1subs > o.loadsize), continue; end
    tmp = num2cell(n1subs); n1lbl = Vlbls(sub2ind(o.loadsize,tmp{:}));
    n2subs = n2pt-o.loadcorner+p.matlab_base;
    if any(n2subs < 1) || any(n2subs > o.loadsize), continue; end
    tmp = num2cell(n2subs); n2lbl = Vlbls(sub2ind(o.loadsize,tmp{:}));
    
    % skip this edge if it it out of bounds of the supervoxel loaded area
    %   (unlabeled area within the loaded volume).
    if n1lbl == p.empty_label || n2lbl == p.empty_label, continue; end
    
    % this edge and these nodes are within the supervoxel labeled volume so include it
    o.edges_use{n}(e) = true; o.nodes_use{n}(n1) = true; o.nodes_use{n}(n2) = true;
    
    % update overall metrics for skeleton
    o.path_length_use(n) = o.path_length_use(n) + o.edge_length{n}(e);
    o.edge_length_use{n}(e) = o.edge_length{n}(e);
  end
end % first pass over skeletons, get path length and connected nodes inside labeled volume

o.nedges_use = cellfun(@sum,o.edges_use); nedges = sum(o.nedges_use);
o.empty_things_use = (o.nedges_use < p.min_edges); nskels = sum(~o.empty_things_use);
o.nnodes_use = cellfun(@sum,o.nodes_use); nnodes = sum(o.nnodes_use);

% sanity checks
assert( all( (o.path_length - cellfun(@sum,o.edge_length)) < p.tol ) );
assert( all( (o.path_length_use - cellfun(@sum,o.edge_length_use)) < p.tol ) );

fprintf(1,'excluding %d/%d edges, %d/%d nodes and %d/%d things because no edges or outside of labeled volume\n',...
  sum(o.nedges)-nedges,sum(o.nedges),sum(o.nnodes)-nnodes,sum(o.nnodes),...
  sum(~o.empty_things)-nskels,sum(~o.empty_things));



% optionally output the skeletons
if p.nmlout
  fprintf(1,'\texporting non-empty skeleton within labeled volume\n'); t = now;
  [~,o.skelname,~] = fileparts(pdata.skelin);

  jnk = struct; 
  [outThings, nOutNodes] = getOutThings(o);
  jnk.fn = fullfile(p.outpath, [o.skelname '_use.nml']);
  knossos_write_nml(jnk.fn,outThings,meta,{});
  display(sprintf('\t\tdone in %.3f s',(now-t)*86400));
end 





% second pass over all edges for all skeletons by walking along the paths
%   to get best and worst case error free path length distributions.
% the worst case metric is the sum of half lengths on either side of each node.
% use randomized error rate of 1 to get the worst-case distribution.
% the best case should be exactly the path lengths except for any skeletons that have passed outside the labeled volume
%   and then return back into the labeled volume and therefore have unconnected components. 
%   xxx - ignore unconnected skeletons for now and compute best case with randomized error rate of 0.
%     only a few skeletons where this happens, could deal with it more properly by running graph connected components.
fprintf(1,'\tcomputing best/worse case efpl (no errors / all errors)\n'); t = now;
[efpl, ~] = labelsWalkEdges(o,p, [], [], [], [0 1]); 
o.efpl_bestcase = efpl{1}; o.efpl_worstcase = efpl{2};
display(sprintf('\t\tdone in %.3f s',(now-t)*86400));
fprintf(1,'\t\tbest case count  = %d, worst case count = %d\n',length(efpl{1}),length(efpl{2}));






% iterate over single dimension of specified segmentation parameters, typically threshold.

% allocate outputs
o.nSMs = zeros(o.nparams,2); o.nSMs_segEM = zeros(o.nparams,2); 
o.nBGnodes = zeros(o.nparams,1); o.nECSnodes = zeros(o.nparams,1);
o.are = zeros(o.nparams,1); o.are_precrec = zeros(o.nparams,2);
o.ri = zeros(o.nparams,1); o.ari = zeros(o.nparams,1);
p.npasses_edges = p.npasses_edges;  % xxx - make this more clear somewhere as a top level define?
o.efpls = cell(o.nparams,p.npasses_edges); o.efpl_thing_ptrs = zeros(o.nparams,o.nThings,p.npasses_edges); 
o.error_free_edges = cell(o.nparams,p.npasses_edges);
o.eftpl = zeros(o.nparams,o.nThings,p.npasses_edges); o.error_rates = zeros(o.nparams,p.npasses_edges); 
o.types_nlabels = zeros(o.nparams,2);
o.are_CI = zeros(o.nparams,2); o.are_precrec_CI = zeros(o.nparams,2,2); 
o.nSMs_CI = zeros(o.nparams,2,2); o.nSMs_segEM_CI = zeros(o.nparams,2,2);
o.ri_CI = zeros(o.nparams,2); o.ari_CI = zeros(o.nparams,2);
o.error_rate_CI = zeros(o.nparams,p.npasses_edges,2); o.eftpl_CI = zeros(o.nparams,p.npasses_edges,2);

for prm=1:o.nparams
  
  thr = prm;
  fprintf(1,'\nreading labels at thr %.8f\n', o.thresholds(thr)); t = now;
  if iscell(o.segparams)
    dset = sprintf('/%s/%s%s',strjoin(pdata.subgroups,'/'),...
      sprintf('%.8f/', cellfun(@(x) x(thr), o.segparams)),p.dataset_lbls);
  else
    dset = sprintf('/%s/%.8f/%s',strjoin(pdata.subgroups,'/'),o.thresholds(thr),p.dataset_lbls);
  end
  clear Vlbls; Vlbls = h5read(pdata.lblsh5,dset,o.loadcorner+p.matlab_base,o.loadsize);
  
  if ~isempty(pdata.nlabels_attr)
    % get nlabels from attributes
    tmp = h5readatt(pdata.lblsh5,dset,pdata.nlabels_attr);
    assert( ~p.remove_MEM_ECS_nodes || length(tmp) > 1 );   % need labels sorted by supervoxel type for this to work
    o.types_nlabels(prm,1:length(tmp)) = tmp;
    nlabels = double(sum(o.types_nlabels(prm,:))); % do not remove ECS components
    %nlabels = double(nlabels(1)); Vlbls(Vlbls > nlabels) = 0;  % remove ECS components
  else
    % get nlabels with max, no easy way to get num ICS/ECS individually.
    % use this pathway for comparing against agglomeration before it has
    %   been resorted based on supervoxel_type (dpCleanLabels).
    nlabels = double(max(Vlbls(:))); o.types_nlabels(prm,:) = [nlabels 0];
    assert( ~p.remove_MEM_ECS_nodes );  % need labels sorted by supervoxel type for this to work
  end
  display(sprintf('\t\tdone in %.3f s, nlabels = %d',(now-t)*86400,nlabels));
  
  % first pass over all edges for all skeletons to get confusion matrix
  % instead of pixels, each tally in the confusion matrix is a node count
  fprintf(1,'\titerating edges within labeled volume to get confusion matrix\n');
  [edge_split, label_merged, nodes_to_labels, m_ij, m_ijl, ~] = labelsPassEdges(o,p,Vlbls,nnodes,nlabels,1:o.nThings);
  
  fprintf(1,'\tgetting simple split/merger estimates from confusion matrix\n');
  [nsplits, nmergers] = getSplitMerger(m_ij, m_ijl, p.remove_MEM_merged_nodes); o.nSMs(prm,:) = [nsplits, nmergers]; 
  [nsplits, nmergers] = getSplitMergerSegEM(m_ijl); o.nSMs_segEM(prm,:) = [nsplits, nmergers];

  % count number of nodes falling into background (membrane) and into ECS supervoxels
  o.nBGnodes(prm) = full(sum(m_ij(:,1)));
  % plus two because first supervoxel column is background in confusion matrix.
  o.nECSnodes(prm) = full(sum(sum(m_ij(:,o.types_nlabels(prm,1)+2:end),2),1));
  
  % optionally completely remove nodes falling into ECS and MEM supervoxels from
  %   confusion matrix.
  if p.remove_MEM_ECS_nodes
    m_ij = m_ij(:,2:o.types_nlabels(prm,1)+2);
  end
  
  fprintf(1,'\tcomputing rand error from confusion matrix\n');
  [are,prec,rec,ri,ari] = getRandErrors(m_ij, nnodes); 
  o.are_precrec(prm,:) = [rec prec]; o.are(prm) = are; o.ri(prm) = ri; o.ari(prm) = ari;
    

  
  % second pass over all edges for all skeletons along the actual skeleton paths
  %   to get errror free path lengths.
  fprintf(1,'\tcomputing split/merger efpl by traversing trees\n'); t = now;
  [efpl, efpl_thing_ptr] = labelsWalkEdges(o,p, edge_split, label_merged, nodes_to_labels, -1);
  o.efpls(prm,:) = efpl; o.efpl_thing_ptrs(prm,:,:) = efpl_thing_ptr; 
  display(sprintf('\t\tdone in %.3f s',(now-t)*86400));
  
  % third pass over all edges for all skeletons along to get edge error metrics.
  fprintf(1,'\tcomputing error free edges by iterating edges\n'); t = now;
  error_free_edges = labelsPassEdgesErrors(o,p,edge_split, label_merged, nodes_to_labels);
  o.error_free_edges(prm,:) = error_free_edges;
  % use the remaining error free edges to get error rates and error free total path length
  for i=1:p.npasses_edges
    o.error_rates(prm,i) = 1 - (sum(cellfun(@sum, error_free_edges{i})) / nedges);
    o.eftpl(prm,:,i) = cellfun(@(a,b) sum(a(b)), o.edge_length, error_free_edges{i});
  end
  display(sprintf('\t\tmean split/merger error rate %.4f %.4f',o.error_rates(prm,1),o.error_rates(prm,2)));
  display(sprintf('\t\tdone in %.3f s',(now-t)*86400));
  

  
  % optionally resample over the skeletons to get confidence intervals
  fprintf(1,'\tresample skeletons to get confidence intervals\n'); t = now;
  if p.jackknife_resample 
    cnt = min([nskels p.n_resample]);   % for the normal jackknife
  else
    % bernoulli resampling
    assert( p.n_resample == 0 || p.bernoulli_n_resample > 1 && p.bernoulli_n_resample <= nskels );
    cnt = p.n_resample; bre = (rand([cnt nskels]) < p.bernoulli_n_resample/nskels);
  end
  % for mapping from sequential non-empty things back to things, random indices in non-empty things
  seli = 1:o.nThings; seli = seli(~o.empty_things_use); 
  bt_are = zeros(cnt,1); bt_are_precrec = zeros(cnt,2); bt_nSMs = zeros(cnt,2); bt_nSMs_segEM = zeros(cnt,2);
  bt_ri = zeros(cnt,1); bt_ari = zeros(cnt,1); 
  bt_error_rates = zeros(cnt,p.npasses_edges); bt_eftpl = zeros(cnt,p.npasses_edges); 
  for bt=1:cnt
    if p.jackknife_resample
      % do jackknife by removing one sample at a time
      selre = true(1,nskels); selre(bt) = false;
    else
      selre = bre(bt,:);
    end
    % map back to thing numbering that contains empty things
    selre = seli(selre);

    % resample regular and logical confusion matrices
    bt_m_ij = m_ij(selre,:); bt_m_ijl = m_ijl(selre,:);
    bt_nnodes = full(sum(bt_m_ij(:))); bt_nskels = length(selre);

    % actual splits / mergers are stored as count, instead store percentages for CI based on selected nnodes
    [nsplits, nmergers] = getSplitMerger(bt_m_ij, bt_m_ijl); bt_nSMs(bt,:) = [nsplits, nmergers]/bt_nnodes;
    [nsplits, nmergers] = getSplitMergerSegEM(bt_m_ijl); bt_nSMs_segEM(bt,:) = [nsplits/bt_nnodes, nmergers/bt_nskels];
    
    [are,prec,rec,ri,ari] = getRandErrors(bt_m_ij, bt_nnodes);
    bt_are_precrec(bt,:) = [rec prec]; bt_are(bt) = are; bt_ri(bt) = ri; bt_ari(bt) = ari;
    %display(sprintf('\t\tdone in %.3f s',(now-t)*86400)); t = now;
    
    % resample error_free_edges only for no error condition (no split and no merger)
    bt_edge_length = o.edge_length(selre); bt_path_length = sum(o.path_length_use(selre));
    bt_nedges = sum(o.nedges_use(selre));
    for i=1:p.npasses_edges
      bt_error_free_edges = error_free_edges{i}(selre); 
      bt_error_rates(bt,i) = 1 - (sum(cellfun(@sum, bt_error_free_edges)) / bt_nedges);
      % store eftpl as percentage of total path length
      bt_eftpl(bt,i) = sum(cellfun(@(a,b) sum(a(b)), bt_edge_length, bt_error_free_edges)) / bt_path_length;
    end
  end
  display(sprintf('\t\tdone in %.3f s',(now-t)*86400));

  plo = floor(cnt * p.p_resample/2);
  if plo >= 1
    phi = cnt-plo+1;
    [~, i] = sort(bt_are);
    o.are_CI(prm,1) = bt_are(i(plo)); o.are_CI(prm,2) = bt_are(i(phi));
    o.are_precrec_CI(prm,1,1) = bt_are_precrec(i(plo),1); o.are_precrec_CI(prm,2,1) = bt_are_precrec(i(plo),2);
    o.are_precrec_CI(prm,1,2) = bt_are_precrec(i(phi),1); o.are_precrec_CI(prm,2,2) = bt_are_precrec(i(phi),2);
    [~, i] = sort(sum(bt_nSMs,2));
    o.nSMs_CI(prm,1,1) = bt_nSMs(i(plo),1); o.nSMs_CI(prm,2,1) = bt_nSMs(i(plo),2);
    o.nSMs_CI(prm,1,2) = bt_nSMs(i(phi),1); o.nSMs_CI(prm,2,2) = bt_nSMs(i(phi),2);
    [~, i] = sort(sum(bt_nSMs_segEM,2));
    o.nSMs_segEM_CI(prm,1,1) = bt_nSMs_segEM(i(plo),1); o.nSMs_segEM_CI(prm,2,1) = bt_nSMs_segEM(i(plo),2);
    o.nSMs_segEM_CI(prm,1,2) = bt_nSMs_segEM(i(phi),1); o.nSMs_segEM_CI(prm,2,2) = bt_nSMs_segEM(i(phi),2);
    [~, i] = sort(bt_ri);
    o.ri_CI(prm,1) = bt_ri(i(plo)); o.ri_CI(prm,2) = bt_ri(i(phi));
    [~, i] = sort(bt_ari);
    o.ari_CI(prm,1) = bt_ari(i(plo)); o.ari_CI(prm,2) = bt_ari(i(phi));
    for k=1:p.npasses_edges
      [~, i] = sort(bt_error_rates(:,k));
      o.error_rate_CI(prm,k,1) = bt_error_rates(i(plo),k); o.error_rate_CI(prm,k,2) = bt_error_rates(i(phi),k);
      [~, i] = sort(bt_eftpl(:,k));
      o.eftpl_CI(prm,k,1) = bt_eftpl(i(plo),k); o.eftpl_CI(prm,k,2) = bt_eftpl(i(phi),k);
    end
  else
    fprintf(1,'\tNO resample stats for n=%d, p=%g\n',p.n_resample,p.p_resample);
  end

  
  
  % optionally write out nml that contains all error_free_edges in a single skeleton
  %   in addition to skeleton that is in the labeled volume.
  if p.nmlout
    fprintf(1,'\texporting error free edges with non-empty skeleton\n'); t = now;
    
    nOutThings = length(outThings); allnodes = zeros(nnodes,4); nodecnt = 0; thingcnt = 0;
    n_ef_edges = cellfun(@sum, error_free_edges{3}); all_ef_edges = zeros(sum(n_ef_edges),2); edgecnt = 0;
    for n=1:o.nThings
      if o.empty_things_use(n), continue; end
      thingcnt = thingcnt + 1;
      
      cnt = size(outThings{thingcnt}.nodes,1); 
      nodes = outThings{thingcnt}.nodes; nodes(:,1) = nodes(:,1) + nOutNodes;
      allnodes(nodecnt+1:nodecnt+cnt,:) = nodes;
      
      sel = false(1,o.nnodes(n)); sel(o.nodes_use{n}) = true; icnodes = cumsum(sel);
      edges = o.info(n).edges(error_free_edges{3}{n},:); 
      all_ef_edges(edgecnt+1:edgecnt+n_ef_edges(n),:) = ...
        reshape(icnodes(edges(:)), [n_ef_edges(n), 2]) + nodecnt + nOutNodes;
      
      nodecnt = nodecnt + cnt; edgecnt = edgecnt + n_ef_edges(n);
    end % for each new thing
    assert( (nodecnt == nOutNodes) && (thingcnt == nOutThings) );
    
    jnk = struct; 
    jnk.coutThings = outThings;
    jnk.coutThings{nOutThings+1}.nodes = allnodes;
    jnk.coutThings{nOutThings+1}.edges = all_ef_edges;
    jnk.coutThings{nOutThings+1}.thingid = nOutThings+1;
    
    jnk.fn = fullfile(p.outpath, [o.skelname sprintf('_thr%.8f_use.nml',o.thresholds(thr))]);
    knossos_write_nml(jnk.fn,jnk.coutThings,meta,{});
    display(sprintf('\t\tdone in %.3f s',(now-t)*86400));
  end
  
end % for each watershed parameter

end % knossos_efpl





% single pass over the edges to get split edges, merged supervoxel labels, node to label mapping and confusion matrix.
% turned this into a function for resampling techniques. 
% thing_list / allThings was to specify resampling instead of iterating over all objects, not currently being used.
function [edge_split, label_merged, nodes_to_labels, m_ij, m_ijl, things_labels_cnt] = ...
  labelsPassEdges(o,p,Vlbls,nnodes,nlabels,thing_list)

  % if this is exactly one of each skeleton, add some optimizations and sanity checks
  allThings = (length(thing_list)==o.nThings) && all(1:o.nThings == thing_list);
  if allThings
    edge_split = cell(1,o.nThings); nodes_to_labels = cell(1,o.nThings);
    things_labels_cnt = 0; things_labels = zeros(nnodes,2);
  else
    edge_split = cell(1,p.nalloc); nodes_to_labels = cell(1,p.nalloc);
    things_labels_cnt = 0; things_labels = zeros(p.nalloc,2);
  end    

  cnt = 0;
  for n=thing_list
    % for re-sampling, count duplicate skeletons as if they were new skeletons in confusion matrix
    if o.empty_things_use(n), continue; end
    if allThings, cnt = n; else cnt = cnt + 1; end
    
    % iterate over edges
    edge_split{cnt} = false(1,o.nedges(n)); nodes_to_labels{cnt} = double(p.empty_label)*ones(o.nedges(n),2);
    for e=1:o.nedges(n)
      if ~o.edges_use{n}(e), continue; end
      n1 = o.info(n).edges(e,1); n2 = o.info(n).edges(e,2); % current nodes involved in this edge
      
      % get the supervoxel label at both node points
      n1pt = round(o.info(n).nodes(n1,1:3) - p.knossos_base);
      n2pt = round(o.info(n).nodes(n2,1:3) - p.knossos_base);
      
      % edge should have already been excluded on first iteration if out of bounds
      n1subs = n1pt-o.loadcorner+p.matlab_base;
      assert( ~(any(n1subs < 1) || any(n1subs > o.loadsize)) );
      tmp = num2cell(n1subs); n1lbl = Vlbls(sub2ind(o.loadsize,tmp{:}));
      n2subs = n2pt-o.loadcorner+p.matlab_base;
      assert( ~(any(n2subs < 1) || any(n2subs > o.loadsize)) );
      tmp = num2cell(n2subs); n2lbl = Vlbls(sub2ind(o.loadsize,tmp{:}));
      
      % edge should have already been excluded on first iteration if unlabeled
      assert( ~(n1lbl == p.empty_label || n2lbl == p.empty_label) );

      % add tally for the supervoxels that current thing's nodes are in. do not include the same node twice 
      % still include if labels are background, handle these situations by slicing confusion matrix below.
      assert( ~allThings || (things_labels_cnt <= nnodes) );
      if nodes_to_labels{cnt}(n1) == p.empty_label
        things_labels_cnt = things_labels_cnt+1;
        things_labels(things_labels_cnt,1) = cnt; things_labels(things_labels_cnt,2) = n1lbl;
      end
      if nodes_to_labels{cnt}(n2) == p.empty_label
        things_labels_cnt = things_labels_cnt+1;
        things_labels(things_labels_cnt,1) = cnt; things_labels(things_labels_cnt,2) = n2lbl;
      end
      
      % save the labels at the nodes so we don't have to look them up again.
      nodes_to_labels{cnt}(n1) = n1lbl; nodes_to_labels{cnt}(n2) = n2lbl;
      
      % count this edge as split if supervoxel labels are not the same
      % count every edge that contains a node in background as a split.
      edge_split{cnt}(e) = (n1lbl==0) || (n2lbl==0) || (n1lbl ~= n2lbl);
    end
  end
  
  % sanity check - make sure each used node was tallied once in the things to labels mapping.
  if allThings
    cnt = o.nThings;
    assert( things_labels_cnt == nnodes );
    node_count_hist = hist(things_labels(:,1), 1:cnt);
    assert( all(node_count_hist == o.nnodes_use) );
  else
    things_labels = things_labels(1:things_labels_cnt,:);
    edge_split = edge_split(1:cnt); nodes_to_labels = nodes_to_labels(1:cnt); 
  end
  
  % the full confusion matrix that counts duplicates and includes background
  % also overlap matrix, contingency or confusion matrix
  m_ij = sparse(things_labels(:,1), things_labels(:,2)+1, 1, cnt, nlabels+1);
  
  % the logical (binary) confusion matrix that does not count duplicates
  % use specified threshold for binarizing.
  m_ijl = (m_ij >= p.m_ij_threshold);
  
  % get boolean of supervoxel labels that contain mergers.
  % do not include background label, as these are counted as splits if either node is in background label.
  label_merged = full(sum(m_ijl,1) > 1); label_merged = label_merged(2:end);
end % labelsPassEdges





% walk trees using stacks to calculate error free path lengths separately for splits and for mergers.
% this calculates the efpl along the actual skeletons, not just based on the confusion matrix.
% turned this into a function for resampling techniques.
function [efpl, efpl_thing_ptr] = labelsWalkEdges(o,p,edge_split, label_merged, nodes_to_labels, rand_error_rate)

  % rand_error_rate is optional, but if specified contains all non-negative entries,
  %   then make a pass for each rand error rate entry.
  if ~exist('rand_error_rate','var'), rand_error_rate = -1; end
  npasses = length(rand_error_rate);
  % if rand_error_rate is not specified or has negative entries then make four passes:
  %   (1) splits only (2) mergers only (3) split or merger errors (3) split and merger errors
  if any(rand_error_rate < 0), npasses = p.npasses_edges; rand_error_rate = -ones(1,npasses); end

  efpl = repmat({zeros(p.nalloc,1)}, [1 npasses]); efpl_cnt = zeros(1,npasses);
  % keep track of where each thing starts on the efpl lists
  efpl_thing_ptr = zeros(o.nThings,npasses);
  
  randcnt = 1; rands = rand(1,p.nalloc);  % for rand_error_rate
  nalloc = 5000;   % local allocation max, just for node stacks
  % need to calculate split efpl and merger efpl separately
  for pass=1:npasses
    
    for n=1:o.nThings
      if o.empty_things_use(n), continue; end
      efpl_thing_ptr(n,pass) = efpl_cnt(pass)+1;

      % keep updated set of unvisited edges
      edges_unvisited = o.edges_use{n}; cur_edges = o.info(n).edges(edges_unvisited, :);
      % stack to return to nodes when an error has occurred along the current path.
      % second index is to store the half path length of the edge on which the error ocurred.
      next_nodes = zeros(nalloc,2); next_node_cnt = 0;
      % stack to return to branch points when encountered along the current path.
      % error free path length is accumulated as long as cur_nodes stack is not empty.
      cur_nodes = zeros(nalloc,1); cur_node_cnt = 0;
      while any(edges_unvisited)
        % find an end point in the remaining edges
        cur_nodes_hist = hist(cur_edges(:), 1:o.nnodes(n));
        end_node = find(cur_nodes_hist==1,1);
        
        % if there are still unvisited edges and none of them only appear
        % once in the edge list, then there must be a cycle. in that case
        % just take any node with edges that have not been visited.
        if isempty(end_node)
          end_node = find(cur_nodes_hist > 0, 1); assert( ~isempty(end_node) );
        end
        
        % push end node (or starting node) onto the next node stack with zero current path length
        next_node_cnt = next_node_cnt+1; next_nodes(next_node_cnt,:) = [end_node 0];
        
        % next node stack keeps track of nodes to continue on after an error has occurred
        while next_node_cnt > 0
          % pop the next node stack
          tmp = next_nodes(next_node_cnt,:); next_node_cnt = next_node_cnt - 1;
          cur_node = tmp(1); cur_efpl = tmp(2);

          % get the remaining edges out of this node
          [cur_node_edges,~] = find(cur_edges==cur_node);
          
          % if there are no remaining edges out of this node then 
          %   we've reached an end point without any more nodes after the last error.
          % save the half length of the previous edge that was stored on the next_nodes stack.
          if isempty(cur_node_edges)
            efpl_cnt(pass) = efpl_cnt(pass)+1; efpl{pass}(efpl_cnt(pass)) = cur_efpl; 
            continue;
          end
          
          % push the onto current node stack with current path length
          cur_node_cnt = cur_node_cnt+1; cur_nodes(cur_node_cnt) = cur_node;
          
          % error free path length accumulates while cur_nodes stack is not empty
          while cur_node_cnt > 0
            % pop the current node stack
            cur_node = cur_nodes(cur_node_cnt); cur_node_cnt = cur_node_cnt - 1;

            % get the remaining edges out of this node
            [cur_node_edges,~] = find(cur_edges==cur_node);

            % if there are no remaining edges out of this node, we've reached
            %   the end of this path, continue to any remaining branch points
            if isempty(cur_node_edges), continue; end
            
            % if this node has more than one edge remaining (branch), 
            %   then push it back to cur_node stack.
            if length(cur_node_edges) > 1
              cur_node_cnt = cur_node_cnt+1; cur_nodes(cur_node_cnt) = cur_node;
            end
            
            % take the first edge out of this node, get both nodes connected to this edge
            n1 = cur_edges(cur_node_edges(1),1); n2 = cur_edges(cur_node_edges(1),2);
            % get the original edge number, should only be one edge
            e = find(all(o.info(n).edges == repmat([n1 n2],[o.nedges(n) 1]),2));
            assert( length(e) == 1 );
            
            % figure out which is the other node connected to this edge
            if cur_node == n1
              other_node = n2;
            elseif cur_node == n2
              other_node = n1;
            else
              assert( false );
            end

            % if introducing randomized errors at edges, then over-ride actual error with randomized error
            if rand_error_rate(pass) >= 0
              error_occurred = (rands(randcnt) < rand_error_rate(pass)); randcnt = randcnt+1;
            else
              error_occurred = checkErrorAtEdge(p,n,n1,n2,e,pass, edge_split, label_merged, nodes_to_labels);
            end
            
            if error_occurred
              % optionally add half edge length at error edge
              % true preserves the total path length, false only counts error-free edges in path length
              if p.count_half_error_edges, cur_len = o.edge_length{n}(e)/2; else cur_len = 0; end
                
              % update current error free path length with half this edge length
              cur_efpl = cur_efpl + cur_len;
              % push the other node onto the next node stack with half this edge length
              next_node_cnt = next_node_cnt+1; next_nodes(next_node_cnt,:) = [other_node cur_len];
            else
              % update current error free path length with this edge length
              cur_efpl = cur_efpl + o.edge_length{n}(e);
              % push other node onto current node stack
              cur_node_cnt = cur_node_cnt+1; cur_nodes(cur_node_cnt) = other_node;
            end
            
            % tally that current edge has been visited, udpate current edges
            edges_unvisited(e) = false; cur_edges = o.info(n).edges(edges_unvisited, :);
          end % while cur node stack
          
          % add the accumulated error free path length after cur_nodes stack is empty
          efpl_cnt(pass) = efpl_cnt(pass)+1; efpl{pass}(efpl_cnt(pass)) = cur_efpl;
          
        end % while next node stack
      end % while unvisited edges

      % sanity check - verify that total efpl is equal to the path length for this thing
      assert( ~p.count_half_error_edges || ...
        (abs(sum(efpl{pass}(efpl_thing_ptr(n,pass):efpl_cnt(pass))) - o.path_length_use(n))) < p.tol );
    end % for each thing
    
    % prune down from allocated size to actual list of efpls
    efpl{pass} = efpl{pass}(1:efpl_cnt(pass));
  end % for each pass
end % labelsWalkEdges



function error_free_edges = labelsPassEdgesErrors(o,p,edge_split, label_merged, nodes_to_labels)
  % keep track of the edges that do not contain errors.
  error_free_edges = repmat({o.edges_use}, [1 p.npasses_edges]);
  for pass=1:p.npasses_edges

    % iterate over things
    for n=1:o.nThings
      if o.empty_things_use(n), continue; end
  
      % iterate over edges
      for e=1:o.nedges(n)
        if ~o.edges_use{n}(e), continue; end
        n1 = o.info(n).edges(e,1); n2 = o.info(n).edges(e,2); % current nodes involved in this edge
        error_free_edges{pass}{n}(e) = ~checkErrorAtEdge(p,n,n1,n2,e,pass, edge_split, label_merged, nodes_to_labels);
      end % for each edge_use
      
    end % for each thing_use
  end % for each pass
end % labelsPassEdgesErrors
      



% use information on merged labels and split edges from first pass through edges
%   to decide if a split or merger error has occurred at this edge or these nodes.
function error_occurred = checkErrorAtEdge(p,n,n1,n2,e,pass, edge_split, label_merged, nodes_to_labels)
  % a split error has occurred on this path if this edge is split
  split_error_occurred = edge_split{n}(e);
  
  % a merge error has occurred on this path if either node is involved in a merger.
  n1lbl = nodes_to_labels{n}(n1); n2lbl = nodes_to_labels{n}(n2);
  assert( ~(n1lbl == p.empty_label || n2lbl == p.empty_label) );
  % do not count a merger for nodes that fall into background areas, these are counted as splits
  merge_error_occurred = ( ((n1lbl > 0) && label_merged(n1lbl)) || ((n2lbl > 0) && label_merged(n2lbl)) );
  
  % up to four passes over edges are defined as:
  %   (1) splits only (2) mergers only (3) split or merger errors (4) split and merger errors
  if pass==1
    error_occurred = split_error_occurred;
  elseif pass==2
    error_occurred = merge_error_occurred;
  elseif pass==3
    error_occurred = (split_error_occurred | merge_error_occurred);
  elseif pass==4
    error_occurred = (split_error_occurred & merge_error_occurred);
  else
    assert( false );
  end
end




function [outThings, nOutNodes] = getOutThings(o)
  nskels = sum(~o.empty_things_use);
  outThings = cell(1,nskels); nOutThings = 0; nOutNodes = 0;
  for n=1:o.nThings
    if o.empty_things_use(n), continue; end

    % remove empty nodes and edges and write new things... meh
    nOutThings = nOutThings + 1;
    outThings{nOutThings}.nodes = [(1:o.nnodes_use(n))'+nOutNodes o.info(n).nodes(o.nodes_use{n},1:3)];
    sel = false(1,o.nnodes(n)); sel(o.nodes_use{n}) = true; icnodes = cumsum(sel);
    edges = o.info(n).edges(o.edges_use{n},:); edges = reshape(icnodes(edges(:)), [o.nedges_use(n), 2]);
    outThings{nOutThings}.edges = edges + nOutNodes;
    outThings{nOutThings}.thingid = nOutThings;
    nOutNodes = nOutNodes + o.nnodes_use(n);
  end % for each new thing
end % getOutThings





function [nsplits, nmergers] = getSplitMerger(m_ij, m_ijl, remove_MEM_merged_nodes)
  % tally up splits as the number supervoxels that are assiciated with more than one thing
  %   but do not count background label nodes. this can range from [0,nnodes]
  tmp = full(sum(m_ijl(:,2:end),2)); nsplits = sum(tmp(tmp > 1));

  % sum the number of nodes that are associated with supervoxel labels that contain nodes from more than one thing.
  % count multiple nodes in the background supervoxel label as mergers unless remove_MEM_merged_nodes specified.
  tmp = full(sum(m_ijl,1) > 1);
  if remove_MEM_merged_nodes, tmp(1) = false; end
  nmergers = full(sum(sum(m_ij(:,tmp),2),1));
end % getSplitMerger

% very similar to above method, see paper "SegEM: Efficient Image Analysis..."
% splits are still bounded by number of nodes, mergers are bounded by number of skeletons (things).
function [nsplits, nmergers] = getSplitMergerSegEM(m_ijl)
  tmp = full(sum(m_ijl(:,2:end),2)); nsplits = sum(tmp(tmp > 1) - 1);
  tmp = full(sum(m_ijl(:,2:end),1)); nmergers = sum(tmp(tmp > 1) - 1);
end 





function [are,prec,rec,ri,ari] = getRandErrors(m_ij, n)

  % sum over proposed labels in contingency matrix
  m_idot = full(sum(m_ij, 2));
  
  % sum over ground truth labels in contingency matrix
  m_dotj = full(sum(m_ij, 1));
  
  % calculate values equal to number of pairwise combinations
  Pk = sum(m_idot.*m_idot) - n; % 2x number of pairs together in ground truth
  Qk = sum(m_dotj.*m_dotj) - n; % 2x number of pairs together in proposed
  Tk = sum(full(sum(m_ij.*m_ij))) - n; % 2x number of pairs together in both
  
  %   % values used in original SNEMI3D metric, -n only relevant for small n?
  %   Pk = sum(m_idot.*m_idot); % ~ number of pairs together in ground truth
  %   Qk = sum(m_dotj.*m_dotj); % ~ number of pairs together in proposed
  %   Tk = sum(full(sum(m_ij.*m_ij))); % ~ number of pairs together in both
  
  % Rand index
  %ri = 1 - (0.5*(Pk + Qk) - Tk) / nchoosek(n,2);
  ri = 0;
  
  % Adjusted Rand index
  %eri = 1 - (Pk + Qk)/(n*(n-1)) + (2*Qk*Pk)/(n*n*(n-1)*(n-1));
  %ari = (ri - eri) / (1 - eri);
  ari = 0;
  
  % precision
  % pairs together in both out of pairs together in proposed
  prec = Tk / Qk;
  
  % recall
  % pairs together in both out of pairs together in ground truth
  rec = Tk / Pk;
  
  % F-score
  fScore = 2.0 * prec * rec / (prec + rec);
  
  % adapted rand error
  are = 1.0 - fScore;

end % getRandErrors
