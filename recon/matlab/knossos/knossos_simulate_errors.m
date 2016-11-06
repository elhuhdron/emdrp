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

% Create new skeletons that have randomly introduced splits and mergers based on specified percentages.
% Convert skeletons to gipl labels and write to different datasets in single h5 output.
% Based on knossos_nodes_to_gipl.m which does:
%   simple conversion from nml skeleton to gipl labels.
%   only fills in exact points gipl, plus a specified 2d structuring element.
% Original attempt for this was one pass through skeletons randomly split / merge at each node.
%   This was not effective (xxx - kept for reference in knossos_rasterize_nodes_errors.m)
function o = knossos_simulate_errors(p)

% output struct
o = struct;

% inputs just for the rasterizing
% p.raw_size
% p.offset_ind
% p.ncubes_raw
% p.dim_order
% p.strel_offs
% p.merge_percs
% p.split_percs
% p.params_meshed

% read the nml file, script originally from Kevin
[info, meta] = KLEE_readKNOSSOS_v4(p.nmlin);
scale = [meta.scale.x meta.scale.y meta.scale.z];

% function removes empty skeletons and unconnected nodes and create logical edge_matrix.
% edge_matrix represents connections between all nodes (each skeleton is a component of this graph).
remove_empty_skeletons = true;
remove_unconnected_nodes = true; % xxx - how does this affect comparing against skeletons without them removed?
knossos_info_graph = knossos_info_to_edges(info, remove_empty_skeletons, remove_unconnected_nodes);
o.knossos_info_graph = knossos_info_graph;

%info = knossos_info_graph.info;
edge_matrix = knossos_info_graph.edge_matrix;
all_edges = knossos_info_graph.all_edges;
all_nodes = knossos_info_graph.all_nodes;
%nedges = knossos_info_graph.nedges;
%nnodes = knossos_info_graph.nnodes;
%cum_nnodes = knossos_info_graph.cum_nnodes;
%nnml_skel = knossos_info_graph.nnml_skel;
tnedges = knossos_info_graph.tnedges;
tnnodes = knossos_info_graph.tnnodes;

% % get the components of the edge matrix (the skeletons)
% ci = conncomp(graph(edge_matrix)); %sizes = hist(ci,1:double(max(ci)));

% get the bounding box of the nodes
maxx = max(all_nodes(:,1)); minx = min(all_nodes(:,1));
maxy = max(all_nodes(:,2)); miny = min(all_nodes(:,2));
maxz = max(all_nodes(:,3)); minz = min(all_nodes(:,3));
minnodes = [minx miny minz]; maxnodes = [maxx maxy maxz];
o.rngnodes = maxnodes - minnodes;

% % distance matrix between all node pairs
% distance_matrix = squareform(pdist(bsxfun(@times,all_nodes(:,1:3),(scale/scale(1)))));

% adjacent neighbor nodes (convex hull) from delaunay triangulation
T = delaunayn(all_nodes(:,1:3));
tetra_perm = nchoosek(1:4,2);
nalloc = 1e6; all_nbrs = zeros(nalloc,2); cnt = 0;
for i=1:size(tetra_perm,1)
  nbrs = T(:,tetra_perm(i,:));
  all_nbrs(cnt+1:size(nbrs,1)+cnt,:) = nbrs; cnt = cnt+size(nbrs,1);
end
all_nbrs = all_nbrs(1:cnt,:);
neighbor_matrix = logical(sparse(all_nbrs(:,1),all_nbrs(:,2),ones(cnt,1),tnnodes,tnnodes)) | ...
  logical(sparse(all_nbrs(:,2),all_nbrs(:,1),ones(cnt,1),tnnodes,tnnodes));
clear all_nbrs;

% inits
minv = p.offset_ind*p.raw_size;
total_size = round(p.ncubes_raw*p.raw_size);

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
  [x,y] = ind2sub(nparams, prm); inds = [x,y];
  if p.params_meshed
    merge_perc = p.merge_percs(x,y);
    split_perc = p.split_percs(x,y);
  else
    merge_perc = p.merge_percs(x);
    split_perc = p.split_percs(y);
  end
  if any(~isfinite([merge_perc split_perc])), continue; end
  fprintf(1,'%d of %d generating for merge %g split %g\n',prm,ntparams,merge_perc,split_perc); t = now;  
  
  % the manipulated graph of skeletons with errors added
  error_edge_matrix = edge_matrix;

  % iterate over edges and randomly remove edge (split)
  split_edges = find(rand(1,tnedges) < split_perc);
  for i=1:length(split_edges)
    error_edge_matrix(all_edges(split_edges(i),1), all_edges(split_edges(i),2)) = false;
    error_edge_matrix(all_edges(split_edges(i),2), all_edges(split_edges(i),1)) = false;
  end
  
  % iterate over nodes and add edge to neighboring node not in this skeleton (merger)
  merge_nodes = find(rand(1,tnnodes) < merge_perc);
  for i=1:length(merge_nodes)
    % add edge to a random neighbor that this node does not already have an edge to
    nbr = randsample(find(neighbor_matrix(merge_nodes(i),:) & ~error_edge_matrix(merge_nodes(i),:)), 1);
    error_edge_matrix(merge_nodes(i), nbr) = true;
    error_edge_matrix(nbr, merge_nodes(i)) = true;
  end

  % run connected graph components on the edge matrix we added random split/merger errors to
  ci = conncomp(graph(error_edge_matrix)); %sizes = hist(ci,1:double(max(ci)));

	% iterate over nodes and assign label just at node location
	labels_skel_raw = zeros(total_size,p.dtype_str);
  for i=1:tnnodes
    % one-based, this appears to be correct for nml data
    curnode = all_nodes(i,1:3) - minv;
    % can't tolerate out of bounds
    if any(curnode < 1) || any(curnode > total_size), assert(false); end
    
    % add point and any surrounding points defined by offsets
    for j=1:size(p.strel_offs,1)
      subs = curnode+p.strel_offs(j,:);
      if any(subs < 1) || any(subs > total_size), continue; end
      labels_skel_raw(subs(:,1),subs(:,2),subs(:,3)) = ci(i);
    end
  end % for each node
  
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
        subgroup_gstr = [subgroup_gstr sprintf('%.8f',params{si}(x,y))];
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

  display(sprintf('\tdone in %.3f s',(now-t)*86400));
end % for each parameter

