
function volume_diameter_fit

% xxx - change this to wherever you downloaded the file to
h5mesh = '~/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean.0.mesh.h5';
h5vol = '/home/watkinspv/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean.h5';
bounds_mat = '~/Downloads/K0057_soma_annotation/out/soma_bounds.mat';

ncuts = 50;
smoothdia = 500; % nm
dataset_root = '0';
doplots = true;
getbounds = false;

scale = h5readatt(h5vol,'/labels','scale')';
minnormal = 1e-4;
dx = scale/5;

if getbounds
  % read the whole volume and get bounding boxes
  Vlbls = h5read(h5vol, '/labels'); nseeds = double(max(Vlbls(:)));
  [label_mins, label_maxs] = find_objects_mex(Vlbls, uint32(nseeds)); clear Vlbls
  save(bounds_mat, 'label_mins', 'label_maxs', 'nseeds');
else
  load(bounds_mat);
end

info = h5info(h5mesh);
nseeds = length({info.Groups(1).Groups.Name})-1;

%cut_n = zeros(nseeds,3); cut_d = zeros(nseeds,2);
for seed=1:nseeds
  %t = now;
  
  % load the meshes
  seed_root = sprintf('/%s/%08d',dataset_root,seed);
  faces = h5read(h5mesh, [seed_root '/faces'])' + 1;
  vertices = double(h5read(h5mesh, [seed_root '/vertices'])'); nverts = size(vertices,1);
  bounds_beg = double(h5readatt(h5mesh, [seed_root '/vertices'], 'bounds_beg')');
  % vertex coordinates relative to entire dataset
  vertices = bsxfun(@plus, vertices, bounds_beg);

  % load the volume
  pmin = double(label_mins(:,seed)'); pmax = double(label_maxs(:,seed)');
  corner = double(pmin); rng = double(pmax - pmin + 1); srng = rng.*scale;
  Vlbls = h5read(h5vol, '/labels', corner, rng); bwcrp = (Vlbls == seed);

  % get the coordinates of the points for this label, relative to bounding box of the label
  [x,y,z] = ind2sub(rng,find(bwcrp(:)));
  vpts = bsxfun(@times,[x y z]-0.5,scale);

  % correct vertices to be relative to bounding box of label
  vertices = bsxfun(@minus, vertices, (corner-1).*scale);
  
  % what points to use for svd, mesh aligns any "processes" better along principal eigenaxis
  pts = vertices; C = mean(pts,1); Cpts = bsxfun(@minus,pts,C);
  
  % svd on centered points to get principal axes.
  % in matlab V is returned normal (not transposed), so "eigenvectors" are along columns,
  %   i.e. V(:,1) V(:,2) V(:,3)
  %[~,S,V] = svd(Cpts,0); s = sqrt(diag(S).^2/(npts-1));
  [~,~,V] = svd(Cpts,0);
  
  % rotate the points to align on cartesian axes
  rvertices = bsxfun(@plus,(V'*bsxfun(@minus,vertices',C')),C')';
  rpts = bsxfun(@plus,(V'*bsxfun(@minus,vpts',C')),C')';
  rC = mean(rpts,1);

  selmin = true(nverts,1);
  if ncuts > 0
    % march plane along principal eigen axis and measure diameter
    mind = min( rpts ); maxd = max( rpts ); step = ( maxd - mind ) / (ncuts+1); 
    cut_pts = repmat( rC, [ncuts 1] );
    cut_pts(:,1) = linspace( mind(1) + step(1), maxd(1) - step(1), ncuts );
    % rotate cutting plane points back to original frame
    rcut_pts = bsxfun(@plus,(V*bsxfun(@minus,cut_pts',C')),C')';
    % calculate the plane offsets using the cutting points
    normal = (V*[1;0;0])'; d = -sum(bsxfun(@times,normal,rcut_pts),2);

    diameters = nan(1,ncuts);
    for i=1:ncuts
        % create the plane orthgonal to the edge within cropped area
        assert(any(abs(normal)>minnormal));
        if normal(3) > minnormal || normal(3) < -minnormal
          [xx,yy] = ndgrid(0:dx(1):srng(1),0:dx(2):srng(2));
          zz = -(normal(1)*xx + normal(2)*yy + d(i))/normal(3);
        elseif normal(2) > minnormal || normal(2) < -minnormal
          [xx,zz] = ndgrid(0:dx(1):srng(1),0:dx(3):srng(3));
          yy = -(normal(1)*xx + normal(3)*zz + d(i))/normal(2);
        else
          [yy,zz] = ndgrid(0:dx(2):srng(2),0:dx(3):srng(3));
          xx = -(normal(2)*yy + normal(3)*zz + d(i))/normal(1);
        end

        % rasterize the plane
        plane_subs = round(bsxfun(@rdivide,[xx(:) yy(:) zz(:)],scale));
        % remove out of bounds
        plane_subs = plane_subs(~any(bsxfun(@gt, plane_subs, rng),2),:); 
        plane_subs = plane_subs(~any(plane_subs < 1,2),:);         
        plane_sel = false(rng); plane_sel(sub2ind(rng,plane_subs(:,1),plane_subs(:,2),plane_subs(:,3))) = true;

        % intersect rasterized plane with this label. take largest component for diameter
        bwp = regionprops(bwcrp & plane_sel,'basic'); [~,j] = max([bwp.Area]);
        if ~isempty(bwp)
          % get diameter as euclidean distance across the bounding box range
          diameters(i) = sqrt(sum((bwp(j).BoundingBox(4:6).*scale).^2));
        end
    end
    
    % smooth the diameters
    box = ceil(smoothdia/step(1)); if mod(box,2)==0, box=box+1; end 
    fdelay = (box-1)/2; xdiameters = (1:length(diameters))*step(1);
    sdiameters = nan(1,length(diameters)); dsdiameters = nan(1,length(diameters)); 
    tmp = filter(ones(1, box)/box, 1, diameters); 
    sdiameters(1:end-fdelay) = tmp(fdelay+1:end);
    tmp = filter(ones(1, box)/box, 1, diff(sdiameters)); 
    dsdiameters(1:end-fdelay-1) = abs(tmp(fdelay+1:end));

    % heuristics to find a good cutting plane, if any
    selbig = (sdiameters > max(sdiameters)/2);
    selsteep = (dsdiameters > max(dsdiameters(selbig))/2);
    selcut = (~selbig & ~selsteep & isfinite(sdiameters) & isfinite(dsdiameters));
    bwp = regionprops(selcut,'basic'); [~,j] = max([bwp.Area]);
    cuti = 0; cutright = false;
    if ~isempty(bwp)
      left = ceil(bwp(j).BoundingBox(1)); right = ceil(bwp(j).BoundingBox(1)) + bwp(j).BoundingBox(3);
      if left < box
        %cuti = right - 1;
        cuti = right;
        selmin = (rvertices(:,1) > cut_pts(cuti,1));
      elseif right > ncuts - box
        %cuti = left + 1; 
        cuti = left; 
        cutright = true;
        selmin = (rvertices(:,1) < cut_pts(cuti,1));
      end
    end
  end
  
  if doplots
    plot_pts_surf(faces, rvertices, selmin);
    
    figure(1235);clf
    [ax,hLine1,hLine2] = plotyy(xdiameters, sdiameters, xdiameters, dsdiameters); hold on
    if cuti > 0
      if cutright
        plot([xdiameters(cuti) xdiameters(cuti)], [0 max(sdiameters)], 'r--');
      else
        plot([xdiameters(cuti) xdiameters(cuti)], [0 max(sdiameters)], 'r');
      end
    end
    hLine1.Marker = '.'; hLine2.Marker = '.';
    xlabel('eigenaxis distance (nm)')
    set(ax(1),'xlim',[xdiameters(1) xdiameters(end)]);
    set(ax(2),'xlim',[xdiameters(1) xdiameters(end)]);
    
    pause
  end
end

  
function plot_pts_surf(faces, pts, sel)

if isempty(sel), sel = true(size(pts,1),1); end
x = pts(:,1); y = pts(:,2); z = pts(:,3);

figure(1234); clf
plot3( x(sel), y(sel), z(sel), '.b' ); hold on
plot3( x(~sel), y(~sel), z(~sel), '.r' ); hold on
h = trisurf(faces, x, y, z);
set(h,'edgecolor','none','facecolor','g','facealpha',0.8);
set( h, 'FaceColor', 'g', 'EdgeColor', 'none', 'facealpha', 0.5);
%view( -70, 40 );
view( 0, 90 );
axis vis3d equal; camlight; lighting phong;
xlabel('x'); ylabel('y'); zlabel('z');
