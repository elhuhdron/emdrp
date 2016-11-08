
% variables that are saved in knossos_efpl_plot output struct (plot meta)
%   save_vars = {
%     'ndatasets' 'params' 'path_lengths' 'internode_lengths' 'nskels' 'split_mergers' 'split_mergers_segEM'
%     'nBGnodes' 'nECSnodes' 'nnodes' 'nnodes_skel' 'names' 'split_mergers_CI' 'split_mergers_segEM_CI' 'nparams'
%     'split_er' 'split_er_CI' 'merge_fracnodes' 'split_fracnodes' 'are' 'are_CI' 'combined_eftpl' 'norm_params'
%     'nlabels'
%   };

% % xxx - save these and drive them down from sensitivity_gen
% % generate "realistic" split merger curves.
% alphax=logspace(-2,0,9); alphax=[0.0001 0.001 0.004 alphax];
% %alphax=[0.0001 0.001];
% split_percs=[0 0.0001 0.001 0.01 0.03 0.06 0.1:0.1:0.2 0.4:0.2:1];
% % order in nodes_to_gipl: params = {p.merge_percs p.split_percs p.remove_percs};
% [alpha, split]=ndgrid(alphax,split_percs); 
% merge=alpha.*(alpha+1)./(split+alpha)-alpha;

merge_percs = 0:0.02:0.2;
split_percs = 0:0.08:0.8;
[merge, split]=ndgrid(merge_percs,split_percs); 


% plot two sensitivity tests and then compare them below

% in_meta = {
%   '/home/watkinspv/Data/efpl/efpl_none_sensitivity_crop_big_meta.mat'
%   '/home/watkinspv/Data/efpl/efpl_huge_sensitivity_crop_big_meta.mat'
% };
% names = {'none' 'huge'};

% in_meta = {
%   '/home/watkinspv/Data/efpl/efpl_none_sensitivity_crop_big_meta.mat'
%   '/home/watkinspv/Data/efpl/efpl_none_sensitivity_crop_big_sample0p8_meta.mat'
% };
% names = {'none all' 'none 80%'};

in_meta = {
  '/home/watkinspv/Data/efpl/efpl_huge_sensitivity_crop_big_meta.mat'
  '/home/watkinspv/Data/efpl/efpl_huge_sensitivity_crop_big_sample0p8_meta.mat'
};
names = {'huge all' 'huge 80%'};

baseno = 11; figno = 0;
nruns = 11;
dolog = false;
nslices = length(merge_percs);
npoints = length(split_percs);
ndata = length(in_meta);
std_lim = 0.08;

eftpl_u = zeros(nslices, npoints, ndata); eftpl_s = zeros(nslices, npoints, ndata);
are_u = zeros(nslices, npoints, ndata); are_s = zeros(nslices, npoints, ndata);
for i=1:ndata
  
  load(in_meta{i});
  assert( po.nparams == npoints );
  
  X = reshape(po.combined_eftpl, [nslices, nruns, po.nparams]);
  eftpl_u(:,:,i) = reshape(nanmean(X,2), [nslices, po.nparams]);
  eftpl_s(:,:,i) = reshape(nanstd(X,[],2), [nslices, po.nparams]);
  if dolog, eftpl_u = log10(eftpl_u(:,:,i)); end
  
  X = reshape(po.are, [nslices, nruns, po.nparams]);
  are_u(:,:,i) = reshape(nanmean(X,2), [nslices, po.nparams]);
  are_s(:,:,i) = reshape(nanstd(X,[],2), [nslices, po.nparams]);
  if dolog, are_u = log10(are_u(:,:,i)); end

  % scatter plot, for reference, only useful for non-grid data
  % pointsize=16;
  % mergemax = 0.25; splitmax = 0.85;
  % subplot(2,2,1);
  % scatter(split(:),merge(:),pointsize,eftpl_u(:));colorbar
  % set(gca,'ylim',[-0.05 mergemax],'xlim',[-0.05 splitmax]);
  % set(gca,'plotboxaspectratio',[1 1 1]);

  figure(baseno+figno); figno = figno+1; clf
  
  subplot(2,2,1);
  imagesc(split_percs,merge_percs,eftpl_u(:,:,i)); colorbar
  set(gca,'plotboxaspectratio',[1 1 1],'ydir','normal','clim',[0 1]);
  xlabel('% splits'); ylabel('% merges'); title(['tefpl mean ' names{i}]);
  
  subplot(2,2,2);
  imagesc(split_percs,merge_percs,eftpl_s(:,:,i)); colorbar
  set(gca,'plotboxaspectratio',[1 1 1],'ydir','normal','clim',[0 std_lim]);
  xlabel('% splits'); ylabel('% merges'); title(['tefpl std ' names{i}]);
  
  subplot(2,2,3);
  imagesc(split_percs,merge_percs,are_u(:,:,i)); colorbar
  set(gca,'plotboxaspectratio',[1 1 1],'ydir','normal','clim',[0 1]);
  xlabel('% splits'); ylabel('% merges'); title(['are mean ' names{i}]);
  
  subplot(2,2,4);
  imagesc(split_percs,merge_percs,are_s(:,:,i)); colorbar
  set(gca,'plotboxaspectratio',[1 1 1],'ydir','normal','clim',[0 std_lim]);
  xlabel('% splits'); ylabel('% merges'); title(['are std ' names{i}]);

end

figure(baseno+figno); figno = figno+1; clf
colormap(gray);

subplot(2,2,1);
d = eftpl_u(:,:,1)-eftpl_u(:,:,2); m = max(abs(d(:)));
imagesc(split_percs,merge_percs,d); colorbar
set(gca,'plotboxaspectratio',[1 1 1],'ydir','normal','clim',[-m m]);
xlabel('% splits'); ylabel('% merges'); title(['tefpl mean ' names{1} '-' names{2}]);

subplot(2,2,2);
d = eftpl_s(:,:,1)-eftpl_s(:,:,2); m = max(abs(d(:)));
imagesc(split_percs,merge_percs,d); colorbar
set(gca,'plotboxaspectratio',[1 1 1],'ydir','normal','clim',[-m m]);
xlabel('% splits'); ylabel('% merges'); title(['tefpl std ' names{1} '-' names{2}]);

subplot(2,2,3);
d = are_u(:,:,1)-are_u(:,:,2); m = max(abs(d(:)));
imagesc(split_percs,merge_percs,d); colorbar
set(gca,'plotboxaspectratio',[1 1 1],'ydir','normal','clim',[-m m]);
xlabel('% splits'); ylabel('% merges'); title(['are mean ' names{1} '-' names{2}]);

subplot(2,2,4);
d = are_s(:,:,1)-are_s(:,:,2); m = max(abs(d(:)));
imagesc(split_percs,merge_percs,d); colorbar
set(gca,'plotboxaspectratio',[1 1 1],'ydir','normal','clim',[-m m]);
xlabel('% splits'); ylabel('% merges'); title(['are std ' names{1} '-' names{2}]);

