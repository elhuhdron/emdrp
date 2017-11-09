
function mesh_ellipsoid_fit2

% xxx - change this to wherever you downloaded the file to
h5file = '~/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean-cut.0.mesh.h5';
outfitfile = '~/Downloads/K0057_soma_annotation/out/somas_cut_fits.mat';

nsteps = 100;
doplots = true;
savefits = false;

info = h5info(h5file);
dataset_root = '0';
nseeds = length({info.Groups(1).Groups.Name})-1;

%fit_v = zeros(nseeds,10); fit_r =  zeros(nseeds,3); fit_R = zeros(nseeds,3,3); fit_C = zeros(nseeds,3);
for seed=1:nseeds
  t = now;
  seed_root = sprintf('/%s/%08d',dataset_root,seed);
  %faces = h5read(h5file, [seed_root '/faces']);
  vertices = h5read(h5file, [seed_root '/vertices']);
  pts = double(vertices'); npts = size(pts,1);
  C = mean(pts,1); Cpts = bsxfun(@minus,pts,C);

  % svd on centered points to get principal axes.
  % in matlab V is returned normal (not transposed), so "eigenvectors" are along columns,
  %   i.e. V(:,1) V(:,2) V(:,3)
  [~,S,V] = svd(Cpts,0);
  % the std of the points along the eigenvectors
  s = sqrt(diag(S).^2/(npts-1));
  
  % rotate the points to align on cartesian axes
  rpts = bsxfun(@plus,(V'*bsxfun(@minus,pts',C')),C')';

  if doplots
    plot_pts_fit(rpts, [], C, 4^(1/3)*s, nsteps);
    pause
  end
  
  fprintf(1,'seed %d of %d in %.4f s\n',seed,nseeds,(now-t)*86400);
end

if savefits
  save(outfitfile, 'fit_v', 'fit_R', 'fit_C', 'fit_r');
end  

function plot_pts_fit(pts, sel, center, radii, nsteps)

if isempty(sel), sel = true(size(pts,1),1); end
x = pts(:,1); y = pts(:,2); z = pts(:,3);

figure(1234); clf
plot3( x(sel), y(sel), z(sel), '.b' ); hold on
plot3( x(~sel), y(~sel), z(~sel), '.r' ); hold on
p = patch( ellipse_pts(pts,nsteps,center,radii) ); 
set( p, 'FaceColor', 'g', 'EdgeColor', 'none', 'facealpha', 0.5);
view( -70, 40 ); 
axis vis3d equal; camlight; lighting phong;
xlabel('x'); ylabel('y'); zlabel('z');

function fv = ellipse_pts(pts,nsteps,center,radii)
mind = min( pts ); maxd = max( pts );
step = ( maxd - mind ) / nsteps;
[ x, y, z ] = meshgrid( linspace( mind(1) - step(1), maxd(1) + step(1), nsteps ), linspace( mind(2) - step(2), ...
  maxd(2) + step(2), nsteps ), linspace( mind(3) - step(3), maxd(3) + step(3), nsteps ) );
xc = x - center(1); yc = y - center(2); zc = z - center(3); r2 = radii.^2;
Ellipsoid = xc.*xc/r2(1) + yc.*yc/r2(2) + zc.*zc/r2(3);
fv = isosurface( x, y, z, Ellipsoid, 1 );
