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

% Convert a info struct array read from a knossos nml file to an edge_matrix representing graph of all nodes.
% Optionally remove any skeletons without edges or nodes and remove any unconnected nodes.
function knossos_info_graph = knossos_info_to_edges(info, remove_empty_skeletons, remove_unconnected_nodes)

if ~exist('remove_empty_skeletons','var'), remove_empty_skeletons = true; end
if ~exist('remove_unconnected_nodes','var'), remove_unconnected_nodes = true; end
knossos_info_graph = struct;

% convert to struct array for indexing, reorder by thingID
info = [info{:}]; [~,i] = sort([info.thingID]); info = info(i);

% get number of edges and nodes and total skeleton count from nml data
nedges = cellfun('size',{info.edges},1); nnodes = cellfun('size',{info.nodes},1);

if remove_empty_skeletons
  % remove empty skeletons
  sel = ((nedges > 0) & (nnodes > 0)); info = info(sel); nedges = nedges(sel); nnodes = nnodes(sel);
  knossos_info_graph.empty_skeletons = find(~sel);
end % remove_empty_skeletons

cum_nnodes = [0 cumsum(nnodes)];
[edge_matrix, all_edges] = create_edge_matrix(info, cum_nnodes, nedges);

if remove_unconnected_nodes
  % remove any unconnected nodes
  ci = conncomp(graph(edge_matrix)); sizes = hist(ci,1:double(max(ci)));
  ind = find(sizes < 2); 
  if ~isempty(ind)
      uc = find(ci == ind);
      knossos_info_graph.unconnected_nodes = uc;
      
      % iterate nodes to be removed and remove them from the info struct.
      for i=1:length(uc)
          % map back to the skeleton number (might not be same as component number from conncomp)
          skel = find(uc(i) > cum_nnodes,1,'last'); uc_skel = uc(i) - cum_nnodes(skel);
          % remove the node from this skeleton, and decrement nodes for edges with nodes greater than removed node
          sel = true(1,size(info(skel).nodes,1)); sel(uc_skel) = false;
          info(skel).nodes = info(skel).nodes(sel,:);
          sel = (info(skel).edges > uc_skel);
          info(skel).edges(sel) = info(skel).edges(sel)-1;
      end
      
      % recalculate everything based on revised info struct array
      nedges = cellfun('size',{info.edges},1); nnodes = cellfun('size',{info.nodes},1);
      cum_nnodes = [0 cumsum(nnodes)];
      [edge_matrix, all_edges] = create_edge_matrix(info, cum_nnodes, nedges);
  end % if there are unconnected nodes
end % remove_unconnected_nodes

% populate output struct
nnml_skel = length(info); tnedges = sum(nedges); tnnodes = cum_nnodes(end);
all_nodes = vertcat(info.nodes);
knossos_info_graph.info = info;
knossos_info_graph.edge_matrix = edge_matrix;
knossos_info_graph.all_edges = all_edges;
knossos_info_graph.all_nodes = all_nodes;
knossos_info_graph.nedges = nedges;
knossos_info_graph.nnodes = nnodes;
knossos_info_graph.cum_nnodes = cum_nnodes;
knossos_info_graph.nnml_skel = nnml_skel;
knossos_info_graph.tnedges = tnedges;
knossos_info_graph.tnnodes = tnnodes;



function [edge_matrix, all_edges] = create_edge_matrix(info, cum_nnodes, nedges)

% create a graph (edge_matrix) representing the skeletons
nnml_skel = length(info); tnedges = sum(nedges); tnnodes = cum_nnodes(end);
all_edges = cell(1,nnml_skel);
for i = 1:nnml_skel
  all_edges{i} = info(i).edges + cum_nnodes(i);
end
all_edges = vertcat(all_edges{:});
% populate both upper and lower triangular portions of edge matrix
edge_matrix = logical(sparse(all_edges(:,1),all_edges(:,2),ones(tnedges,1),tnnodes,tnnodes)) | ...
  logical(sparse(all_edges(:,2),all_edges(:,1),ones(tnedges,1),tnnodes,tnnodes));
