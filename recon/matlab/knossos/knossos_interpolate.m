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

% Create a new nml file that has exactly one node for each skeleton in the given dim direction.
function [minnodes, rngnodes] = knossos_interpolate(skelin,skelout,p)

%p.remove_branch_edges
%p.remove_inflection_edges
%p.interp_dim
%p.rngdiff
%p.min_nodes
%p.write_new_nml
%p.interp_dim_rng
%p.extrap_max
%p.extrap_do_line

% some constants
dimstr = {'x' 'y' 'z'}; dimistr = dimstr{p.interp_dim};
alloc_nodes = 1000; alloc_things = 200;
node_meta.color_r = -1;
node_meta.color_g = -1;
node_meta.color_b = -1;
node_meta.rad = 1.5;
node_meta.inVp = 0;
node_meta.inMag = 1;
node_meta.time = fix(now*24); % ???
useBGL = false;

% read the nml file, script from Kevin
evalc('[info, meta, ~] = KLEE_readKNOSSOS_v4(skelin)'); % suppresss output
scale = [meta.scale.x meta.scale.y meta.scale.z];

% convert to struct array for indexing, reorder by thingID
info = [info{:}]; [~,i] = sort([info.thingID]); info = info(i);

% get number of edges and nodes and total skeleton count from nml data
nedges = cellfun('length',{info.edges}); nnodes = cellfun('size',{info.nodes},1);
% don't do this, reports wrong thingIDs for check and will happen automatically in creating new nml file
% if any((nedges==0)&(nnodes>0)) || any((nedges>0)&(nnodes==0))
%   error('thingID with nodes and no edges or vice versa');
% end
%sel = (nedges > 0); info = info(sel); nedges = nedges(sel); nnodes = nnodes(sel); % remove empty thingIDs
nThings = length(info); % total skeletons in nml file

% this will only work for bounds if the skels traverse in all directions... so does not really work, good checks tho
allnodes = vertcat(info.nodes);
maxx = max(allnodes(:,1)); minx = min(allnodes(:,1));
maxy = max(allnodes(:,2)); miny = min(allnodes(:,2));
maxz = max(allnodes(:,3)); minz = min(allnodes(:,3));
minnodes = [minx miny minz]; maxnodes = [maxx maxy maxz];
rngnodes = maxnodes - minnodes;
%nextNodeID = max(allnodes(:,5)) + 1; % not using this, re-number the nodes

nbranches = zeros(1,nThings); ninflections = zeros(1,nThings); nsmallthings = zeros(1,nThings);
nNewNodes = 0; nNewThings = 0; newThings = cell(1,alloc_things); nextrap = zeros(1,nThings);
for n=1:nThings
  if nedges(n) == 0
    fprintf(1,'thingID %d has zero edges and %d nodes, will be removed in new nml\n',n,nnodes(n)); continue;
  end
  
  % pass 1 through edges, count edges at each node, decide which edges to remove
  edge_cnt = zeros(1,nnodes(n)); prev_diff_into_node = nan(1,nnodes(n)); remove_edge = false(1,nedges(n));
  for e=1:nedges(n)
    n1 = info(n).edges(e,1); n2 = info(n).edges(e,2); % current nodes involved in this edge
    if (edge_cnt(n1) > 1) || (edge_cnt(n2) > 1)
      nbranches(n) = nbranches(n) + 1; remove_edge(e) = remove_edge(e) | p.remove_branch_edges;
    end
    edge_cnt(n1) = edge_cnt(n1) + 1; edge_cnt(n2) = edge_cnt(n2) + 1;
    
    % the slope of this edge
    n1_to_n2_diff = info(n).nodes(n2,1:3) - info(n).nodes(n1,1:3);
    n1_to_n2_diff_dimi = n1_to_n2_diff(p.interp_dim); % diff of whatever dim we're interpolating
    n1_to_n2_diff_dimiabs = abs(n1_to_n2_diff_dimi);
    
    % flag 'error' cases
    if (n1_to_n2_diff_dimiabs < p.rngdiff(1)) || (n1_to_n2_diff_dimiabs > p.rngdiff(2))
      fprintf(1,'thingID %d edge from node %d (%d) to node %d (%d) %sdiff %d\n',n,n1,info(n).nodes(n1,5),n2,...
        info(n).nodes(n2,5),dimistr,n1_to_n2_diff_dimiabs);
    end
    
    % check for inflections in the interp dim so these can also be split
    % record diff going IN to each node, so compare against diff going OUT
    if isfinite(prev_diff_into_node(n1)) && (sign(prev_diff_into_node(n1)) ~= sign(n1_to_n2_diff_dimi))
      if ~remove_edge(e), ninflections(n) = ninflections(n) + 1; end
      remove_edge(e) = remove_edge(e) | p.remove_inflection_edges;
    elseif n1_to_n2_diff_dimiabs > 0
      prev_diff_into_node(n1) = -n1_to_n2_diff_dimi;
    end
    if isfinite(prev_diff_into_node(n2)) && (sign(prev_diff_into_node(n2)) ~= sign(-n1_to_n2_diff_dimi))
      if ~remove_edge(e), ninflections(n) = ninflections(n) + 1; end
      remove_edge(e) = remove_edge(e) | p.remove_inflection_edges;
    elseif n1_to_n2_diff_dimiabs > 0
      prev_diff_into_node(n2) = n1_to_n2_diff_dimi;
    end
  end % for each edge
  
  % skip any graph manipulations if we're creating a new nml file (check only mode)
  if ~p.write_new_nml, continue; end
  
  % pass 2 through edges, add non-removed edges/nodes and add interpolated nodes
  edges = zeros(alloc_nodes,2); node_pts = zeros(alloc_nodes,3);
  cnnodes = nnodes(n); cnedges = 0;
  for e=1:nedges(n)
    if remove_edge(e), continue; end
    
    n1 = info(n).edges(e,1); n2 = info(n).edges(e,2); % current nodes involved in this edge
    
    % the slope of this edge
    n1_to_n2_diff = info(n).nodes(n2,1:3) - info(n).nodes(n1,1:3);
    n1_to_n2_diff_dimi = n1_to_n2_diff(p.interp_dim); % diff of whatever dim we're interpolating
    n1_to_n2_diff_dimiabs = abs(n1_to_n2_diff_dimi);
    
    % check if nodes are on same z-plane
    if n1_to_n2_diff_dimiabs == 0
      % if one of these nodes is an endpoint, then remove this edge
      % xxx - this doesn't fix all conditions... only very "simple mistake" cases
      % if it's a singleton branch it will get removed anyways
      if (edge_cnt(n1) < 2) || (edge_cnt(n2) < 2), remove_edge(e) = true; continue; end
    end
    
    % add the points for the existing nodes, doesn't matter if already added (existing points don't change)
    node_pts(n1,:) = info(n).nodes(n1,1:3);
    node_pts(n2,:) = info(n).nodes(n2,1:3);
    
    if n1_to_n2_diff_dimiabs == 1
      % add back the existing edge
      cnedges = cnedges + 1; edges(cnedges,:) = [n1 n2];
    else
      % interpolate missing z-slices, add points for interpolated nodes
      vt = repmat((1:n1_to_n2_diff_dimiabs-1)'/n1_to_n2_diff_dimiabs,[1 3]); nt = size(vt,1);
      node_pts(cnnodes+1:cnnodes+nt,:) = round(repmat(info(n).nodes(n1,1:3),[nt 1]) + repmat(n1_to_n2_diff,[nt 1]).*vt);
      
      % add the new edges connecting through interpolated nodes
      cedges = repmat(1:nt,[1 2]) + cnnodes; cedges = reshape([n1 cedges n2],[nt+1 2]);
      edges(cnedges+1:cnedges+nt+1,:) = cedges;
      
      % increment the node count by interpolated nodes, increment edge count by new number of edges
      cnnodes = cnnodes + nt; cnedges = cnedges + nt+1;
    end
  end % for each existing edge

  % add extrapolated nodes by checking if nodes extend the desired range of the interp dim.
  % optionally do line-based extrapolation from previous node. if not, drop straight up or straight down in this dim.
  clim = [max(node_pts(1:cnnodes,p.interp_dim)); min(node_pts(1:cnnodes,p.interp_dim))]; cdir = [1 -1];
  nmissing = [p.interp_dim_rng(2) - clim(1) clim(2) - p.interp_dim_rng(1)];
  for i=1:length(clim)
    if (nmissing(i) <= 0) || (nmissing(i) > p.extrap_max), continue; end
    % iterate over all nodes at this limit
    inds = find(node_pts(:,p.interp_dim)==clim(i));
    for j=1:length(inds);
      from_node = inds(j); pt = node_pts(from_node,:);
      
      % optionally do line-based extrapolation, but only if there is a single previous node
      [xe,ye] = find(edges==from_node);
      if p.extrap_do_line && length(xe) == 1
        prev_pt = node_pts(edges(xe,mod(ye,2)+1),:);
      else
        % drop node straight up or down in this dim
        prev_pt = pt - [0 0 cdir(i)];
      end
      
      % extrapolate missing z-slices, add points for extrapolated nodes
      vt = repmat((1:nmissing(i))'/nmissing(i),[1 3]); nt = size(vt,1);
      node_pts(cnnodes+1:cnnodes+nt,:) = round(repmat(pt,[nt 1]) + repmat(pt - prev_pt,[nt 1]).*vt);
      
      % add the new edges connecting through interpolated nodes
      cedges = repmat(1:(nt-1),[1 2]) + cnnodes; cedges = reshape([from_node cedges cnnodes+nt],[nt 2]);
      edges(cnedges+1:cnedges+nt,:) = cedges;
      
      % increment the node count by interpolated nodes, increment edge count by new number of edges
      cnnodes = cnnodes + nmissing(i); cnedges = cnedges + nmissing(i);
      nextrap(n) = nextrap(n) + nmissing(i); % keep count
    end % for each node at the limit
  end % for each limit ("top" / "bottom" of interp dir)
  
  % create the sparse edge matrix (un-directed graph)
  edges = edges(1:cnedges,:); node_pts = node_pts(1:cnnodes,:);
  edge_matrix = sparse(edges(:,1), edges(:,2), true, cnnodes, cnnodes);
  edge_matrix = edge_matrix' | edge_matrix;
  
  % run graph connected components on the edge matrix to get the new things created from this skeleton
  if useBGL
    [ci, sizes] = components(edge_matrix); % matlabBGL
  else
    ci = conncomp(graph(edge_matrix)); sizes=hist(ci,1:double(max(ci))); 
  end
  assert(any(remove_edge) || length(sizes)==1);
  %if ~(any(remove_edge) || length(sizes)==1)
  %  fprintf(1,'thingID %d has %d components and no remove edges\n\t',n,length(sizes));
  %  fprintf(1,'%d ',sizes); fprintf(1,'\n');
  %end
  
  % remove graph components less than threshold size
  % xxx - need to rethink edge_matrix if the number of nodes gets very large
  bgsel = (sizes < p.min_nodes); nremove = sum(bgsel);
  if nremove > 0
    remove_nodes = find(bgsel(ci));    
    edge_matrix = full(edge_matrix);
    edge_matrix(remove_nodes,:) = []; edge_matrix(:,remove_nodes) = [];
    node_pts(remove_nodes,:) = [];
    if useBGL
      [ci, sizes] = components(sparse(edge_matrix));
    else
      ci = conncomp(graph(edge_matrix)); sizes=hist(ci,1:double(max(ci))); 
    end
    nsmallthings(n) = nsmallthings(n) + nremove;
  end
  edge_matrix = full(triu(edge_matrix));
  
  % set all new unique node IDs created from splitting this thing
  cnnodes = size(edge_matrix,1); cnodes = [(1:cnnodes)' + nNewNodes node_pts];

  % add all new things created from splitting this thing apart by removing edges
  for i=1:length(sizes)
    % recreate the edges for each component from the edge_matrix
    node_sel = (ci==i);
    cedge_matrix = edge_matrix;
    cedge_matrix(~node_sel,:) = false; % cedge_matrix(:,~node_sel) = false; % upper triangular
    [xe,ye] = find(cedge_matrix); edges = [xe ye];
    
    % set new unique node IDs, add new thing
    nNewThings = nNewThings + 1;
    newThings{nNewThings}.nodes = cnodes(node_sel,:);
    newThings{nNewThings}.edges = edges + nNewNodes;
    newThings{nNewThings}.thingid = nNewThings;
  end % for each new thing
  nNewNodes = nNewNodes + cnnodes;

end % for each object
newThings = newThings(1:nNewThings);
fprintf(1,'\norig %d thingIDs, got %d new thingIDs, %d branches, %d inflections, %d small things\n',...
  nThings,nNewThings,sum(nbranches),sum(ninflections),sum(nsmallthings));
fprintf(1,'extrapolated %d nodes\n',sum(nextrap));

if p.write_new_nml
  evalc('KnossosM_exportNML(skelout,newThings,meta,{},node_meta);'); % suppress output
end
