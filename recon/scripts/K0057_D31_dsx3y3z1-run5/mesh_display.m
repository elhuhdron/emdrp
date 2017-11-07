
% xxx - change this to wherever you downloaded the file to
h5file = '~/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean.0.mesh.h5';

inf = h5info(h5file);
dataset_root = '0';
nseeds = length({inf.Groups(1).Groups.Name})-1;
cmap = hsv(32);
cmap = cmap(randperm(size(cmap,1)),:);

figure(1234); clf
for seed=1:nseeds
  seed_root = sprintf('/%s/%08d',dataset_root,seed);
  faces = h5read(h5file, [seed_root '/faces']);
  vertices = h5read(h5file, [seed_root '/vertices']);
  corner = h5readatt(h5file, [seed_root '/vertices'], 'bounds_beg');
  
  vertices = bsxfun(@plus, uint32(vertices), corner);
  h = trisurf(faces' + 1, vertices(1,:), vertices(2,:), vertices(3,:));
  set(h,'edgecolor','none','facecolor',cmap(mod(seed-1,size(cmap,1))+1,:),'facealpha',0.8);
  hold on
end

set(gca,'dataaspectratio',[1 1 1]);
l = light('Position',[-0.4 0.2 0.9],'Style','infinite');
lighting gouraud
