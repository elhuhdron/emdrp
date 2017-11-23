
function mesh_ellipsoid_fit

% xxx - change this to wherever you downloaded the file to
h5file = '~/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean-cut.0.mesh.h5';
outcutfile = '~/Downloads/K0057_soma_annotation/out/soma_cuts.mat';
outfitfile = '~/Downloads/K0057_soma_annotation/out/somas_cut_fits.mat';

ncuts = 0;
%ncuts = 20;
scale = 4; % fudge factor
nsteps = 100;
doplots = false;
savefits = true;

info = h5info(h5file);
dataset_root = '0';
nseeds = length({info.Groups(1).Groups.Name})-1;

cut_n = zeros(nseeds,3); cut_d = zeros(nseeds,2);
fit_v = zeros(nseeds,10); fit_r =  zeros(nseeds,3); fit_R = zeros(nseeds,3,3); fit_C = zeros(nseeds,3);
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
  %[~,S,V] = svd(Cpts,0); s = sqrt(diag(S).^2/(npts-1));
  [~,~,V] = svd(Cpts,0);
  
  % rotate the points to align on cartesian axes
  rpts = bsxfun(@plus,(V'*bsxfun(@minus,pts',C')),C')';
  rC = mean(rpts,1);

  if ncuts > 0
    % march cutting plane along principal eigen axis starting at either end into centroid
    mind = min( rpts ); maxd = max( rpts ); dmin = inf;
    stepmin = ( rC - mind ) / ncuts; stepmax = ( maxd - rC ) / ncuts;
    x = linspace( mind(1), rC(1) - stepmin(1), ncuts ); y = linspace( rC(1) + stepmax(1), maxd(1), ncuts );
    for i=1:ncuts
      for j=1:ncuts
        sel = (rpts(:,1) > x(i)) & (rpts(:,1) < y(j));
        [ ~,radii,~,~,~, d, ~ ] = best_ellipsoid_fit( rpts(sel,:), {}, nsteps );
        if d < dmin
          % expand the cut back out by some amount in both directions
          minx = x(i) - radii(1)/scale; maxx = y(j) + radii(1)/scale;
          dmin = d; %selmin = (rpts(:,1) > minx) & (rpts(:,1) < maxx);
        end
      end
    end
    
    % get the equation of the cutting plane in the original coordinate system
    cut_pts = [rC; rC]; cut_pts(1,1) = minx; cut_pts(2,1) = maxx;
    % rotate cutting plane points back to original frame
    cut_pts = bsxfun(@plus,(V*bsxfun(@minus,cut_pts',C')),C')';
    % calculate the plane offsets using the cutting points
    n = (V*[1;0;0])'; d = [-sum(n.*cut_pts(1,:)) -sum(n.*cut_pts(2,:))];
    cut_n(seed,:) = n; cut_d(seed,:) = d;
    
    % create point select in original coordinate frame as sanity check
    selmin = (sum(bsxfun(@times,pts,n),2) + d(1) > 0) & (sum(bsxfun(@times,pts,n),2) + d(2) < 0);
  else
    selmin = true(npts,1);
  end

  if doplots || savefits
    [ center, radii, evecs, v, ~, d, ~ ] = best_ellipsoid_fit( rpts(selmin, :), {}, nsteps );
    fit_v(seed,:) = v; fit_R(seed,:,:) = V; fit_C(seed,:) = C; fit_r(seed,:) = radii;
  end
  
  if doplots
    plot_pts_fit(rpts, selmin, center, radii, evecs, v, d, nsteps);
    pause
  end
  
  fprintf(1,'seed %d of %d in %.4f s\n',seed,nseeds,(now-t)*86400);
end

if ncuts > 0
  save(outcutfile, 'cut_n', 'cut_d');
end

if savefits
  save(outfitfile, 'fit_v', 'fit_R', 'fit_C', 'fit_r');
end  

function [ center, radii, evecs, v, chi2, dmin, ind ] = best_ellipsoid_fit( pts, fit_params, nsteps )
if isempty(fit_params), fit_params = {'' 'xy' 'xz' 'yz' 'xyz' '0' '0xy' '0xz' '0yz'}; end
dmin = inf; ind = 0; center = []; radii = []; evecs = []; v = []; chi2 = [];
for i=1:length(fit_params)
  [ ccenter, cradii, cevecs, cv, cchi2 ] = ellipsoid_fit( pts, fit_params{i} );
  %fv = ellipse_pts(pts,nsteps,cv); [~, d] = knnsearch(pts, fv.vertices);
  epts = ellipse_pts_sph(ccenter,cradii,nsteps); [~, d] = knnsearch(pts, epts);
  % what metric to use with nearest point distances?
  d = mean(d);
  % only take ellipsoids, other quadric sections have negative radii
  if all(cradii > 0) && d < dmin 
    dmin = d; ind = i; center = ccenter; radii = cradii; evecs = cevecs; v = cv; chi2 = cchi2;
  end
end

function plot_pts_fit(pts, sel, center, radii, evecs, v, dist, nsteps)

if isempty(sel), sel = true(size(pts,1),1); end
x = pts(:,1); y = pts(:,2); z = pts(:,3);

figure(1234); clf
plot3( x(sel), y(sel), z(sel), '.b' ); hold on
plot3( x(~sel), y(~sel), z(~sel), '.r' ); hold on
%draw fit
%epts = ellipse_pts_sph(center, radii, nsteps);
%plot3(epts(:,1),epts(:,2),epts(:,3),'.g');
p = patch( ellipse_pts(pts,50,v) ); set( p, 'FaceColor', 'g', 'EdgeColor', 'none', 'facealpha', 0.5);
view( -70, 40 ); 
%view( 2 ); 
axis vis3d equal; camlight; lighting phong;
title(sprintf('distance metric %g', sqrt( dist / size( pts, 1 ) )));
xlabel('x'); ylabel('y'); zlabel('z');

% fprintf( '\nEllipsoid center: %.5g %.5g %.5g\n', center );
% fprintf( 'Ellipsoid radii: %.5g %.5g %.5g\n', radii );
% fprintf( 'Ellipsoid evecs:\n' );
% fprintf( '%.5g %.5g %.5g\n%.5g %.5g %.5g\n%.5g %.5g %.5g\n', ...
%     evecs(1), evecs(2), evecs(3), evecs(4), evecs(5), evecs(6), evecs(7), evecs(8), evecs(9) );


function fv = ellipse_pts(pts,nsteps,v)
mind = min( pts ); maxd = max( pts );
step = ( maxd - mind ) / nsteps;
[ x, y, z ] = meshgrid( linspace( mind(1) - step(1), maxd(1) + step(1), nsteps ), linspace( mind(2) - step(2), ...
  maxd(2) + step(2), nsteps ), linspace( mind(3) - step(3), maxd(3) + step(3), nsteps ) );
Ellipsoid = v(1) *x.*x +   v(2) * y.*y + v(3) * z.*z + ...
  2*v(4) *x.*y + 2*v(5)*x.*z + 2*v(6) * y.*z + ...
  2*v(7) *x    + 2*v(8)*y    + 2*v(9) * z;
fv = isosurface( x, y, z, Ellipsoid, -v(10) );

function pts = ellipse_pts_sph(center, radii, nsteps)
[theta,phi] = meshgrid(linspace(-pi/2,pi/2,nsteps), linspace(-pi,pi,nsteps));
x=radii(1)*cos(theta).*cos(phi);
y=radii(2)*cos(theta).*sin(phi);
z=radii(3)*sin(theta);
pts = bsxfun(@plus, [x(:) y(:) z(:)], center');
