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

% Cleans up skeletons using knossos_info_to_edges.m and then crops skeletons to a specified area.
% Runs graph components so that skeletons in cropped area are not split.
function o = knossos_clean_crop(p)

% output struct
o = struct;

% inputs 
% p.raw_size
% p.offset_ind
% p.ncubes_raw
% p.offset
% p.nmlin
% p.nmlout

% read the nml file, script originally from Kevin
[info, meta, ~, ~] = knossos_read_nml(p.nmlin);
%scale = [meta.scale.x meta.scale.y meta.scale.z];

% function removes empty skeletons and unconnected nodes and create logical edge_matrix.
% edge_matrix represents connections between all nodes (each skeleton is a component of this graph).
remove_empty_skeletons = true;
remove_unconnected_nodes = true; 
knossos_info_graph = knossos_info_to_edges(info, remove_empty_skeletons, remove_unconnected_nodes);
%o.knossos_info_graph = knossos_info_graph;

%info = knossos_info_graph.info;
edge_matrix = knossos_info_graph.edge_matrix;
%all_edges = knossos_info_graph.all_edges;
all_nodes = knossos_info_graph.all_nodes;
%nedges = knossos_info_graph.nedges;
%nnodes = knossos_info_graph.nnodes;
%cum_nnodes = knossos_info_graph.cum_nnodes;
%nnml_skel = knossos_info_graph.nnml_skel;
%tnedges = knossos_info_graph.tnedges;
%tnnodes = knossos_info_graph.tnnodes;

% get the specified cropped area
minv = p.offset_ind*p.raw_size + p.offset;
total_size = p.ncubes_raw*p.raw_size - p.offset;

% one-based, this appears to be correct for nml data
crop_nodes = bsxfun(@minus, all_nodes(:,1:3), minv);
remove_nodes = any(crop_nodes < 1, 2) | any(bsxfun(@minus,total_size,crop_nodes) < 0, 2);
keep_nodes = ~remove_nodes;

% remove the nodes from the edge_matrix
edge_matrix = edge_matrix(keep_nodes, keep_nodes);
% remove the nodes from all_nodes
all_nodes = all_nodes(keep_nodes, :);

% get the components of the edge matrix (the skeletons)
ci = conncomp(graph(edge_matrix)); sizes = hist(ci,1:double(max(ci)));

% remove any skeletons left with a single node (unconnected) after removing out of bounds nodes
remove_nodes = ismember(ci, find(sizes < 2));
keep_nodes = ~remove_nodes;
% remove the nodes from the edge_matrix
edge_matrix = edge_matrix(keep_nodes, keep_nodes);
% remove the nodes from all_nodes
all_nodes = all_nodes(keep_nodes, :);

% get the components of the edge matrix (the skeletons)
ci = conncomp(graph(edge_matrix)); %sizes = hist(ci,1:double(max(ci)));

% create a new array of node / edge info based on new components (should be splits only)
nThings = max(ci); outThings = cell(1,nThings); nnodes = 0; nedges = 0;
for n=1:nThings
    outThings{n}.thingid = n;
    
    sel_nodes = (ci == n); cnodes = sum(sel_nodes);
    outThings{n}.nodes = [(1:cnodes)'+nnodes all_nodes(sel_nodes, 1:3)];
    [i,j] = find(triu(edge_matrix(sel_nodes, sel_nodes)));
    outThings{n}.edges = [i j] + nnodes;
    nnodes = nnodes + cnodes; nedges = nedges + size(outThings{n}.edges,1);
end % for each new thing

% sanity checks
assert( nnodes == size(all_nodes,1) );
assert( nedges == nnz(edge_matrix)/2 );

knossos_write_nml(p.nmlout,outThings,meta,{},{});
