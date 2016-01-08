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

% simple conversion from nml skeleton to gipl labels.
% only fills in exact points gipl, plus a specified 2d structuring element.
function rngnodes = knossos_nodes_to_gipl(nmlin, giplout, p)

% p.read_size
% p.raw_size
% p.offset_ind
% p.ncubes_raw
% p.dim_order
% p.isotopic_voxels
% p.strel_offs 

% read the nml file, script originally from Kevin
evalc('[info, meta] = KLEE_readKNOSSOS_v4(nmlin)'); % suppresss output
scale = [meta.scale.x meta.scale.y meta.scale.z];

% convert to struct array for indexing, reorder by thingID
info = [info{:}]; [~,i] = sort([info.thingID]); info = info(i);

% get number of edges and nodes and total skeleton count from nml data
nedges = cellfun('length',{info.edges}); nnodes = cellfun('size',{info.nodes},1);
if any((nedges==0)&(nnodes>0)) || any((nedges>0)&(nnodes==0))
  error('thingID with nodes and no edges or vice versa'); 
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

% inits
if p.isotopic_voxels, scalex = scale/scale(1); else scalex = [1 1 1]; end
scaleo = scale./scalex;
minv = p.offset_ind*p.raw_size;
total_size = round(p.ncubes_raw*p.raw_size.*scalex);
labels_skel_raw = zeros(total_size,'uint16');

for n=1:nnml_skel
  for nn=1:nnodes(n)
    % one-based, this appears to be correct for nml data
    curnode = info(n).nodes(nn,1:3) - minv;
    
    % for anisotropic voxels
    curnode = round(curnode.*scalex);
    
    % omit out of bounds
    if any(curnode < 1) || any(curnode > total_size)
      fprintf('node %d in thing %d out of bounds at %d,%d,%d\n',info(n).nodes(nn,5),n,curnode);
      continue; 
    end

    % add point and any surrounding points defined by offsets
    for j=1:size(p.strel_offs,1)
      subs = curnode+p.strel_offs(j,:);
      if any(subs < 1) || any(subs > total_size), continue; end
      labels_skel_raw(subs(:,1),subs(:,2),subs(:,3)) = n;
    end
  end % for each node
end % for each object

labels_skel = permute(labels_skel_raw,p.dim_order);   % legacy matlab scripts can swap x/y

gipl_write_volume(labels_skel,giplout,scaleo);

% plot skeletonized GT
figno = 2005;
bg_clrs = [0 0 0; 0.1 0.1 0.1; 0.2 0.2 0.2; 0.3 0.3 0.3; 0.4 0.4 0.4; 0.5 0.5 0.5; 0.8 0.8 0.8; 0.9 0.9 0.9; 1 1 1;...
  0.1 0 0; 0.2 0 0; 0 0.1 0; 0 0.2 0; 0 0 0.1; 0 0 0.2];
figure(figno); clf; %figno=figno+1;
n = max(labels_skel(:)); clrs = distinguishable_colors(n,bg_clrs);
set(gcf,'Color','white'); set(gca,'DataAspectRatio',1./scaleo); view(140,80)
set(gca,'xlim',[0 size(labels_skel,1)+1],'ylim',[0 size(labels_skel,2)+1] ,'zlim',[0 size(labels_skel,3)+1]);
for i=1:n
  [x,y,z]=ind2sub(size(labels_skel),find(labels_skel(:)==i));
  hold on; plot3(y,x,z,'square','Markersize',4,'MarkerFaceColor',clrs(i,:),'Color',clrs(i,:));
end
