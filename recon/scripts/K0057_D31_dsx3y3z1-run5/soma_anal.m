
load('~/Downloads/K0057_soma_annotation/out/somas_cut.mat');
load('~/Downloads/K0057_soma_annotation/out/soma_fits.mat');
dsfactor=[12 12 4]./[16 16 16];

% print out "mergers" or discrepancies
[unique_soma_labels,ia,ic] = unique(soma_labels);
duplicate_labels = setdiff((1:length(soma_labels))', ia);
for i=1:length(duplicate_labels)
  lbl = soma_labels(duplicate_labels(i));
  fprintf(1,'In label %d:\n',lbl);
  sel = (soma_labels==lbl); inds = soma_center_inds(sel,:)+1; inds2 = fix(bsxfun(@times, double(inds), dsfactor))+1;
  for j=1:sum(sel(:))
    fprintf(1,'\t%d %d %d, mag16 %d %d %d\n', inds(j,1),inds(j,2),inds(j,3), inds2(j,1),inds2(j,2),inds2(j,3));
  end
end

% print out "missing" or discrepancies
missing_nodes = setdiff(double(soma_valid_labels), double(soma_labels));
if ~isempty(missing_nodes)
  fprintf(1,'Labels missing nodes: '); fprintf(1,'%d ',missing_nodes); fprintf(1,'\n');
end

fprintf(1,'Total %d unique soma labels\n', length(unique_soma_labels));

soma_surface_areas = soma_surface_areas/1e6;
soma_volumes = soma_volumes/1e9;
fit_r = fit_r/1000;
sphericity = pi^(1/3)*(6*soma_volumes).^(2/3)./soma_surface_areas;

bins = 0:30:1200; cbins = bins(2:end)-((bins(2)-bins(1))/2);
sbins = 0:0.05:1; csbins = sbins(2:end)-((sbins(2)-sbins(1))/2);
rbins = 0:0.5:15; crbins = rbins(2:end)-((rbins(2)-rbins(1))/2);
bins2 = 0:50:1200; cbins2 = bins2(2:end)-((bins2(2)-bins2(1))/2);
sbins2 = 0:0.025:1; csbins2 = sbins2(2:end)-((sbins2(2)-sbins2(1))/2);
types = {'CR','TH','CB','?'}; ntypes = length(types);
%types_clrs={'r','g','b','y'}; types_clrs_rgb=[1 0 0; 0 1 0; 0 0 1; 1 1 0];
types_clrs = reshape(hex2dec(...
  {'95' '19' '0c', '97' 'cc' '04', '08' '4e' 'ab', 'e3' 'b5' '05', '14' '14' '13'}...
  ),3,[])'/255;
vlim=[0 1100]; salim=[0 900]; sphlim=[0.4 0.95]; rlim=[0 15];

hvol = zeros(ntypes,length(bins)-1); hsa = zeros(ntypes,length(bins)-1); 
hsph = zeros(ntypes,length(sbins)-1); cnt_types = zeros(ntypes,1);
hvolsa = zeros(ntypes,length(bins2)-1,length(bins2)-1);
hvolsph = zeros(ntypes,length(bins2)-1,length(sbins2)-1);
hsasph = zeros(ntypes,length(bins2)-1,length(sbins2)-1);
hrad0 = zeros(ntypes,length(rbins)-1);
hrad1 = zeros(ntypes,length(rbins)-1);
hrad2 = zeros(ntypes,length(rbins)-1);
types_cnts_str = cell(1,ntypes);
for type=1:ntypes
  sel = (soma_types==type);
  hvol(type,:) = histcounts(soma_volumes(sel), bins);
  hsa(type,:) = histcounts(soma_surface_areas(sel), bins);
  hsph(type,:) = histcounts(sphericity(sel), sbins);
  cnt_types(type) = sum(sel(:));
  fprintf(1,'Total %d %s somas\n', cnt_types(type), types{type});
  types_cnts_str{type} = sprintf('%s = %d', types{type}, cnt_types(type));
  
  hvolsa(type,:,:) = histcounts2(soma_volumes(sel), soma_surface_areas(sel), bins2, bins2);
  hvolsph(type,:,:) = histcounts2(soma_volumes(sel), sphericity(sel), bins2, sbins2);
  hsasph(type,:,:) = histcounts2(soma_surface_areas(sel), sphericity(sel), bins2, sbins2);

  hrad0(type,:) = histcounts(fit_r(sel,1), rbins);
  hrad1(type,:) = histcounts(fit_r(sel,2), rbins);
  hrad2(type,:) = histcounts(fit_r(sel,3), rbins);
end
% cdfs
cvol = bsxfun(@rdivide, cumsum(hvol,2), cnt_types);
csa = bsxfun(@rdivide, cumsum(hsa,2), cnt_types);
csph = bsxfun(@rdivide, cumsum(hsph,2), cnt_types);
crad0 = bsxfun(@rdivide, cumsum(hrad0,2), cnt_types);
crad1 = bsxfun(@rdivide, cumsum(hrad1,2), cnt_types);
crad2 = bsxfun(@rdivide, cumsum(hrad2,2), cnt_types);

% normalize for pdfs, comment for counts
hvol = bsxfun(@rdivide, hvol, cnt_types);
hsa = bsxfun(@rdivide, hsa, cnt_types);
hsph = bsxfun(@rdivide, hsph, cnt_types);
hrad0 = bsxfun(@rdivide, hrad0, cnt_types);
hrad1 = bsxfun(@rdivide, hrad1, cnt_types);
hrad2 = bsxfun(@rdivide, hrad2, cnt_types);
dlbl = 'probability density';
%dlbl = 'count';

% normalize 2d pdfs
hvolsa = bsxfun(@rdivide, hvolsa, cnt_types);
hvolsph = bsxfun(@rdivide, hvolsph, cnt_types);
hsasph = bsxfun(@rdivide, hsasph, cnt_types);

figure(1234); clf
subplot(2,2,1);
set(gca, 'ColorOrder', types_clrs, 'NextPlot', 'replacechildren');
plot(cbins, hvol);
xlabel('volume (mm^3)'); legend(types_cnts_str); ylabel(dlbl)
set(gca,'plotboxaspectratio',[1 1 1]); set(gca,'xlim',vlim);
subplot(2,2,2);
set(gca, 'ColorOrder', types_clrs, 'NextPlot', 'replacechildren');
plot(cbins, hsa);
xlabel('surface area (mm^2)'); ylabel(dlbl)
set(gca,'plotboxaspectratio',[1 1 1]); set(gca,'xlim',salim);
subplot(2,2,3);
set(gca, 'ColorOrder', types_clrs, 'NextPlot', 'replacechildren');
plot(csbins, hsph);
xlabel('sphericity'); ylabel(dlbl)
set(gca,'plotboxaspectratio',[1 1 1]); set(gca,'xlim',sphlim);

figure(1235); clf
subplot(2,2,1);
set(gca, 'ColorOrder', types_clrs, 'NextPlot', 'replacechildren');
plot(cbins, cvol);
xlabel('volume (mm^3)'); legend(types_cnts_str,'location','southeast'); ylabel('cumulative density')
set(gca,'plotboxaspectratio',[1 1 1]); set(gca,'xlim',vlim);
subplot(2,2,2);
set(gca, 'ColorOrder', types_clrs, 'NextPlot', 'replacechildren');
plot(cbins, csa);
xlabel('surface area (mm^2)'); ylabel('cumulative density')
set(gca,'plotboxaspectratio',[1 1 1]); set(gca,'xlim',salim);
subplot(2,2,3);
set(gca, 'ColorOrder', types_clrs, 'NextPlot', 'replacechildren');
plot(csbins, csph);
xlabel('sphericity'); ylabel('cumulative density')
set(gca,'plotboxaspectratio',[1 1 1]); set(gca,'xlim',sphlim);

% figure(1236); clf
% for type=1:ntypes
%   sel = (soma_types==type);
%   scatter(soma_volumes(sel), sphericity(sel), 16, types_clrs(type,:)); hold on;
% end
% xlabel('volume (mm^3)'); ylabel('sphericity');

% figure(2235); clf; figure(2236); clf; figure(2237); clf
% for type=1:ntypes
%   figure(2235);
%   subplot(2,2,type);
%   imagesc(bins2,bins2,squeeze(hvolsa(type,:,:))'); colormap(1-gray); colorbar
%   set(gca,'ydir','normal'); xlabel('Vol'); ylabel('SA');
%   set(gca,'xlim',vlim,'ylim',salim);
%   title(types{type})
%   figure(2236);
%   subplot(2,2,type);
%   imagesc(bins2,sbins2,squeeze(hvolsph(type,:,:))'); colormap(1-gray); colorbar
%   set(gca,'ydir','normal'); xlabel('Vol'); ylabel('Sph');
%   set(gca,'xlim',vlim,'ylim',sphlim);
%   title(types{type})
%   figure(2237);
%   subplot(2,2,type);
%   imagesc(bins2,sbins2,squeeze(hsasph(type,:,:))'); colormap(1-gray); colorbar
%   set(gca,'ydir','normal'); xlabel('SA'); ylabel('Sph');
%   set(gca,'xlim',salim,'ylim',sphlim);
%   title(types{type})
% end

figure(2345); clf
for type=1:ntypes
  sel = (soma_types==type);
  scatter3(fit_r(sel,1), fit_r(sel,2), fit_r(sel,3), 16, types_clrs(type,:)); hold on;
end
xlabel('x rad (mm)'); ylabel('y rad (mm)'); zlabel('z rad (mm)'); 

figure(2346); clf
subplot(2,2,1);
set(gca, 'ColorOrder', types_clrs, 'NextPlot', 'replacechildren');
plot(crbins, hrad0);
xlabel('major r (mm)'); legend(types_cnts_str); ylabel(dlbl)
set(gca,'plotboxaspectratio',[1 1 1]); set(gca,'xlim',rlim);
subplot(2,2,2);
set(gca, 'ColorOrder', types_clrs, 'NextPlot', 'replacechildren');
plot(crbins, hrad1);
xlabel('middle r (mm)'); ylabel(dlbl)
set(gca,'plotboxaspectratio',[1 1 1]); set(gca,'xlim',rlim);
subplot(2,2,3);
set(gca, 'ColorOrder', types_clrs, 'NextPlot', 'replacechildren');
plot(crbins, hrad2);
xlabel('minor r (mm)'); ylabel(dlbl)
set(gca,'plotboxaspectratio',[1 1 1]); set(gca,'xlim',rlim);

figure(2347); clf
subplot(2,2,1);
set(gca, 'ColorOrder', types_clrs, 'NextPlot', 'replacechildren');
plot(crbins, crad0);
xlabel('major r (mm)'); legend(types_cnts_str,'location','southeast'); ylabel('cumulative density')
set(gca,'plotboxaspectratio',[1 1 1]); set(gca,'xlim',rlim);
subplot(2,2,2);
set(gca, 'ColorOrder', types_clrs, 'NextPlot', 'replacechildren');
plot(crbins, crad1);
xlabel('middle r (mm)'); ylabel('cumulative density')
set(gca,'plotboxaspectratio',[1 1 1]); set(gca,'xlim',rlim);
subplot(2,2,3);
set(gca, 'ColorOrder', types_clrs, 'NextPlot', 'replacechildren');
plot(crbins, crad2);
xlabel('minor r (mm)'); ylabel('cumulative density')
set(gca,'plotboxaspectratio',[1 1 1]); set(gca,'xlim',rlim);
