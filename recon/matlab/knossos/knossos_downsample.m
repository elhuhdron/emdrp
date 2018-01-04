% The MIT License (MIT)
% 
% Copyright (c) 2017 Paul Watkins, National Institutes of Health / NINDS
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

% Downsample an nml file and write out downsampled nml.
function [minnodes, rngnodes] = knossos_downsample(skelin,skelout,p)

%p.ds_ratio
%p.use_radii
%p.experiment

% some constants
node_meta.color_r = -1;
node_meta.color_g = -1;
node_meta.color_b = -1;
node_meta.rad = 1.5;
node_meta.inVp = 0;
node_meta.inMag = 1;
node_meta.time = fix(now*24); % ???

% read the nml file, script from Kevin
[info, meta, commentsString, branchpointsString] = knossos_read_nml(skelin);
scale = [meta.scale.x meta.scale.y meta.scale.z];
scale = scale .* p.ds_ratio;
meta.scale.x = scale(1); meta.scale.y = scale(2); meta.scale.z = scale(3);
meta.experiment.name = p.experiment;

% convert to struct array for indexing, reorder by thingID
info = [info{:}]; %[~,i] = sort([info.thingID]); info = info(i); % removed sorting so 1-1 match with originals

% get number of edges and nodes and total skeleton count from nml data
%nedges = cellfun('size',{info.edges},1); 
nnodes = cellfun('size',{info.nodes},1);
nThings = length(info); % total skeletons in nml file

% not using these, good checks on bounds though
allnodes = vertcat(info.nodes);
maxx = max(allnodes(:,1)); minx = min(allnodes(:,1));
maxy = max(allnodes(:,2)); miny = min(allnodes(:,2));
maxz = max(allnodes(:,3)); minz = min(allnodes(:,3));
minnodes = [minx miny minz]; maxnodes = [maxx maxy maxz];
rngnodes = maxnodes - minnodes;

newThings = cell(1,nThings); cnnodes = 0;
for n=1:nThings
  cnodes = (1:nnodes(n))' + cnnodes;
  pts = fix(bsxfun(@rdivide, info(n).nodes(:,1:3), p.ds_ratio));
  newThings{n}.nodes = [cnodes pts];
  if p.use_radii
     newThings{n}.nodes = [newThings{n}.nodes info(n).nodes(:,4)];
  end    
  newThings{n}.edges = info(n).edges + cnnodes;
  newThings{n}.thingid = info(n).thingID;
  newThings{n}.comment = info(n).comment;
  cnnodes = cnnodes + nnodes(n);
end

% xxx - there is not necessarily a mapping with the newly numbered nodes.
%   nml read needs to be modified to map the comments directly to nodes in each thing.
knossos_write_nml(skelout,newThings,meta,{},{},node_meta);
