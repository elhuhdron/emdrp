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

% xxx - this method did not work, replaced with iterating over skeletons
%   to either remove or introduce edges and then re-running graph components.
% simple conversion from nml skeleton to gipl labels.
% only fills in exact points gipl, plus a specified 2d structuring element.
% expanded this to also introduce random split, merger and remove skeleton perturbations.
function rngnodes = knossos_rasterize_nodes_errors(p)

% inputs just for the rasterizing
% p.raw_size
% p.offset_ind
% p.ncubes_raw
% p.dim_order
% p.strel_offs

% inputs that control perturbations to skeletons
if ~isfield(p,'merge_percs'), p.merge_percs = 0; end
if ~isfield(p,'split_percs'), p.split_percs = 0; end
if ~isfield(p, 'params_meshed'), p.params_meshed = false; end
% p.merge_rad

% read the nml file, script originally from Kevin
evalc('[info, meta] = KLEE_readKNOSSOS_v4(p.nmlin)'); % suppresss output
scale = [meta.scale.x meta.scale.y meta.scale.z];

% convert to struct array for indexing, reorder by thingID
info = [info{:}]; [~,i] = sort([info.thingID]); info = info(i);

% get number of edges and nodes and total skeleton count from nml data
nedges = cellfun('length',{info.edges}); nnodes = cellfun('size',{info.nodes},1);
if any((nedges==0)&(nnodes>0)) || any((nedges>0)&(nnodes==0))
  fprintf(1,'WARNING: thingID with nodes and no edges or vice versa\n');
  %error('thingID with nodes and no edges or vice versa');
end
sel = (nedges > 0); info = info(sel); nedges = nedges(sel); nnodes = nnodes(sel); % remove empty thingIDs
nnml_skel = length(info); % total skeletons expected from nml file

% this will only work if the skels traverse in all directions... so does not really work, good checks tho
allnodes = vertcat(info.nodes);
maxx = max(allnodes(:,1)); minx = min(allnodes(:,1));
maxy = max(allnodes(:,2)); miny = min(allnodes(:,2));
maxz = max(allnodes(:,3)); minz = min(allnodes(:,3));
minnodes = [minx miny minz]; maxnodes = [maxx maxy maxz];
rngnodes = maxnodes - minnodes;

% distance matrix is used for deciding on mergers
%node_dist = squareform(pdist(bsxfun(@times,allnodes(:,1:3),(scale/scale(1)))));
node_dist = squareform(pdist(allnodes(:,1:3)));

% inits
minv = p.offset_ind*p.raw_size;
total_size = round(p.ncubes_raw*p.raw_size);
cum_nnodes = [0 cumsum(nnodes)]; tnnodes = cum_nnodes(end);

% iterate over input parameters.
% params order is important for the calling function, didn't see a need for any order.
params = {p.merge_percs p.split_percs};
params_str = {'merge' 'split'};
if p.params_meshed
  nparams = size(params{1}); ntparams = numel(params{1});
  sparams = cellfun(@size,params,'UniformOutput',false);
  assert( all(cellfun(@(x) all(x==nparams), sparams)) );
else
  nparams = cellfun('length',params); ntparams = prod(nparams);
  assert( ~isempty(p.hdf5out) || ntparams == 1 );
end

for prm = 1:ntparams
  [x,y,z] = ind2sub(nparams, prm); inds = [x,y,z];
  if p.params_meshed
    merge_perc = p.merge_percs(x,y,z);
    split_perc = p.split_percs(x,y,z);
  else
    merge_perc = p.merge_percs(x);
    split_perc = p.split_percs(y);
  end
  if any(~isfinite([merge_perc split_perc])), continue; end
  fprintf(1,'%d of %d generating for merge %g split %g\n',prm,ntparams,merge_perc,split_perc);

  % inits
  labels_skel_raw = zeros(total_size,p.dtype_str); nskels = nnml_skel;
  node_assign = zeros(1,tnnodes); is_merged = false(1,tnnodes);  
  
  % determine the total number and "seed" nodes for merger objects
  %   % xxx - slight randomization of nmergers?
  %   %   nmergers = ((randi(7)-4)/100 + merge_perc); nmergers(nmergers < 0) = 0; nmergers(nmergers > 1) = 1;
  %   %   nmergers = fix((1-nmergers)*(nnml_skel-1)) + 1;
  %   nmergers = fix((1-merge_perc)*(nnml_skel-1)) + 1;
  %   merger_seeds = randperm(size(allnodes,1), nmergers);
  merger_seeds = []; nmergers = 0;
  % assign the merge seed nodes to their skeletons, calculate min distance from each node to the closest merger_seed
  merger_seeds_dist = zeros(nmergers, tnnodes);
  for i=1:nmergers;
    mrg_node=merger_seeds(i);
  	node_assign(mrg_node) = find(mrg_node > cum_nnodes,1,'last'); is_merged(mrg_node) = true;
    merger_seeds_dist(i,:) = node_dist(mrg_node,:);
  end

  % for introducing randomly merged/split nodes
  merge_node = (rand(1,tnnodes) < merge_perc);
  split_node = (rand(1,tnnodes) < split_perc);

  %node_order = 1:tnnodes;  
  node_order = randperm(tnnodes);
  %   % iterate over nodes in order from the closest nodes to the merger_seeds to the furthest
  %   [~, node_order] = sort(min(merger_seeds_dist,[],1));
  for inode=node_order
    if node_assign(inode) > 0; continue; end % already assigned (from a merger)
    n = find(inode > cum_nnodes,1,'last'); % which skeleton this node is in
    assert(nedges(n) > 0); % empty skeletons should have been removed above
    nn = inode - cum_nnodes(n); % index of node in current skeleton
    
    % one-based, this appears to be correct for nml data
    curnode = info(n).nodes(nn,1:3) - minv;
    
    % can't tolerate out of bounds
    if any(curnode < 1) || any(curnode > total_size), assert(false); end
    
    node_assign(inode) = n; % default is to label node with current skeleton
    if any(merger_seeds == inode)
      % this is a merger seed so leave it alone, assign to original skeleton
    elseif merge_node(nn)
      %       % find the closest merger seed node and merge to it
      %       [~, imin] = min(node_dist(inode,merger_seeds));
      %       mrg_node = merger_seeds(imin);
      %       assert(node_assign(mrg_node) > 0); % seed node should always have been assigned
      %       node_assign(inode) = node_assign(mrg_node);
      
      %       % find the closest already merged node and merge to it
      %       is_merged_nodes = find(is_merged);
      %       [~, imin] = min(node_dist(inode,is_merged_nodes));
      %       mrg_node = is_merged_nodes(imin);
      %       assert(mrg_node ~= inode); % bigly wrong
      %       assert(node_assign(mrg_node) > 0); % seed node should always have been assigned
      %       node_assign(inode) = node_assign(mrg_node); is_merged(inode) = true;
      
      % pick a random nearby node to merge to
      [~, imin] = sort(node_dist(inode,:)); imin = imin(2:p.merge_radN+1);
      mrg_node = imin(randi(p.merge_radN));
      if node_assign(mrg_node) > 0
        % node being merged to was already assigned, assign current node to that
        node_assign(inode) = node_assign(mrg_node);
      elseif split_node(nn) 
        % split / merger
        nskels = nskels + 1; node_assign(inode) = nskels;
        node_assign(mrg_node) = node_assign(inode);
      else
        node_assign(mrg_node) = node_assign(inode);
      end
    elseif split_node(nn)
      nskels = nskels + 1; node_assign(inode) = nskels;
    end
    
    % add point and any surrounding points defined by offsets
    for j=1:size(p.strel_offs,1)
      subs = curnode+p.strel_offs(j,:);
      if any(subs < 1) || any(subs > total_size), continue; end
      labels_skel_raw(subs(:,1),subs(:,2),subs(:,3)) = node_assign(inode);
    end
  end % for each node (over all nodes)
  
  %unique(node_assign)
  if ~all(p.dim_order == 1:3)
    labels_skel = permute(labels_skel_raw,p.dim_order);   % legacy matlab scripts can swap x/y
  else
    labels_skel = labels_skel_raw;
  end
  
  % create subgroups depending on which parameters are being iterated
  subgroup_str = 'perc'; subgroup_gstr = {};
  for si = 1:length(nparams)
    if nparams(si) > 1
      subgroup_str = [subgroup_str '_' params_str{si}];
      if p.params_meshed
        subgroup_gstr = [subgroup_gstr sprintf('%.8f',params{si}(x,y,z))];
      else
        subgroup_gstr = [subgroup_gstr sprintf('%.8f',params{si}(inds(si)))];
      end
    end
  end
  subgroups = [{subgroup_str} subgroup_gstr];
  
  try
    dset = ['/' strjoin(subgroups,'/'),'/labels']; h5info(p.hdf5out,dset);
    %fprintf(1,'\tWARNING: already in output file, skipping h5write\n');
    create_dataset = false;
  catch
    create_dataset = true;
  end
  
  % read the info from the "template" input file
  attrs = struct;
  s = h5info(p.hdf5lbls,'/labels');
  for i=1:length(s.Attributes)
    attrs.(s.Attributes(i).Name) = s.Attributes(i).Value;
  end
  attrs.scale = scale; % override scale with nml one
  
  if create_dataset
    % write data, xxx - how to map h5 datatype to matlab type?
    %     h5create(p.hdf5out,dset,s.Dataspace.Size,'ChunkSize',s.ChunkSize,'Datatype',s.Datatype.Type,...
    %       'Deflate',5,'Fletcher32',true,'Shuffle',true,'FillValue',s.FillValue);
    h5create(p.hdf5out,dset,s.Dataspace.Size,'ChunkSize',s.ChunkSize,'Datatype',p.dtype_str,...
      'Deflate',5,'Fletcher32',true,'Shuffle',true,'FillValue',intmax(p.dtype_str));
  end
  h5write(p.hdf5out,dset,labels_skel,minv+[1 1 1],size(labels_skel));
  
  % copy attributes
  write_fields = fields(attrs);
  for i=1:length(write_fields)
    h5writeatt(p.hdf5out,dset,write_fields{i},attrs.(write_fields{i}));
  end
  
end % for each parameter

