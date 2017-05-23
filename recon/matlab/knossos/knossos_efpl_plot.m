
function po = knossos_efpl_plot(pdata,o,p)

useColorOrder = ~verLessThan('matlab','8.4');

% remove really small internode distances that are just from dropping nodes right next to each other
minPL = 10^(p.plx(1)-p.dplx(1)/2);

po = struct;
ndatasets = length(pdata);
% use old name for parameter dimension, typically thresholds.
params = cell(1,ndatasets);
path_lengths = cell(1,ndatasets);       % actually "best case"
internode_lengths = cell(1,ndatasets);  % actually "worst case"
nskels = zeros(1,ndatasets);
split_mergers = cell(1,ndatasets);
split_mergers_segEM = cell(1,ndatasets);
nBGnodes = cell(1,ndatasets); nECSnodes = cell(1,ndatasets);
nnodes = zeros(1,ndatasets);
nnodes_skel = cell(1,ndatasets);
names = cell(1,ndatasets);

split_mergers_CI = cell(1,ndatasets);
split_mergers_segEM_CI = cell(1,ndatasets);

for i = 1:ndatasets
  use = ~o{i}.omit_things_use; nskels(i) = sum(use);
  
  if p.use_efpl_edges
    % new mode, with one efpl counted per each edge
    path_lengths{i} = [o{i}.efpl_edges_bestcase{:}] / 1000;
    internode_lengths{i} = [o{i}.efpl_edges_worstcase{:}] / 1000;
  else
    % old mode, with one efpl counted per each connected path
    path_lengths{i} = o{i}.efpl_bestcase / 1000;
    internode_lengths{i} = o{i}.efpl_worstcase / 1000;
  end
  
  % remove really small internode distances that are just from dropping nodes right next to each other
  path_lengths{i} = path_lengths{i}(path_lengths{i} >= minPL);
  internode_lengths{i} = internode_lengths{i}(internode_lengths{i} >= minPL);
  
  split_mergers{i} = o{i}.nSMs;
  split_mergers_segEM{i} = o{i}.nSMs_segEM;
  nBGnodes{i} = o{i}.nBGnodes; nECSnodes{i} = o{i}.nECSnodes;
  nnodes(i) = sum(o{i}.nnodes_use);
  nnodes_skel{i} = o{i}.nnodes_use(~o{i}.omit_things_use);
  
  split_mergers_CI{i} = o{i}.nSMs_CI;
  split_mergers_segEM_CI{i} = o{i}.nSMs_segEM_CI;
  
  params{i} = o{i}.thresholds;
  names{i} = pdata{i}.name;
end

% for single matrices over parameter dimension, use max length and fill shorter rows with NaN
nparams = max(cellfun(@length, params));

split_er = nan(ndatasets,nparams); merge_er = nan(ndatasets,nparams);
split_er_CI = nan(ndatasets,nparams,2); merge_er_CI = nan(ndatasets,nparams,2);
merge_fracnodes = nan(ndatasets,nparams); merge_fracnodes_CI = nan(ndatasets,nparams,2);
split_fracnodes = nan(ndatasets,nparams); split_fracnodes_CI = nan(ndatasets,nparams,2);
are = nan(ndatasets,nparams); are_precrec = nan(ndatasets,nparams,2);
are_CI = nan(ndatasets,nparams,2); are_precrec_CI = nan(ndatasets,nparams,2,2);
combined_eftpl = nan(ndatasets,nparams); combined_eftpl_CI = nan(ndatasets,nparams,2);
norm_params = nan(ndatasets,nparams);
nlabels = nan(ndatasets,nparams);
for k = 1:ndatasets
  nthr = length(params{k}); ind = 1:nthr;
  split_er(k,ind) = o{k}.error_rates(ind,1);
  merge_er(k,ind) = o{k}.error_rates(ind,2);
  split_er_CI(k,ind,:) = o{k}.error_rate_CI(ind,1,:);
  merge_er_CI(k,ind,:) = o{k}.error_rate_CI(ind,2,:);
  merge_fracnodes(k,ind) = o{k}.nSMs(ind,2)/nnodes(k);
  merge_fracnodes_CI(k,ind,:) = o{k}.nSMs_CI(ind,2,:);
  split_fracnodes(k,ind) = o{k}.nSMs(ind,1)/nnodes(k);
  split_fracnodes_CI(k,ind,:) = o{k}.nSMs_CI(ind,1,:);
  are(k,ind) = o{k}.are(ind); are_CI(k,ind,:) =  o{k}.are_CI(ind,:);
  % just convert precision recall to 1- here (so min is better)
  are_precrec(k,ind,:) = 1-o{k}.are_precrec(ind,:); are_precrec_CI(k,ind,:,:) = 1-o{k}.are_precrec_CI(ind,:,:);
  combined_eftpl(k,ind) = sum(o{k}.eftpl(ind,:,3),2)/sum(o{k}.path_length_use);
  combined_eftpl_CI(k,ind,:) = o{k}.eftpl_CI(ind,3,:);
  nlabels(k,ind) = sum(o{k}.types_nlabels(ind,:),2);
  if p.param_name
    norm_params(k,1:nthr) = 1:nthr;
  else
    norm_params(k,1:nthr) = linspace(0,1,nthr);
  end
end

% optionally just use sum for rand-error, like other errors
if p.are_sum
  are = squeeze(sum(are_precrec, 3));
end

% get histograms 
pl_hist = zeros(ndatasets,p.nplx); inpl_hist = zeros(ndatasets,p.nplx); node_hist = zeros(ndatasets,p.nndx);
pl_actual_median = zeros(ndatasets,1); inpl_actual_median = zeros(ndatasets,1);
pl_actual_iqr = zeros(ndatasets,2);
pl_hist_roc = zeros(ndatasets,p.nplxs); inpl_hist_roc = zeros(ndatasets,p.nplxs);
for i = 1:ndatasets
  pl_hist(i,:) = hist(log10(path_lengths{i}), p.plx);
  inpl_hist(i,:) = hist(log10(internode_lengths{i}), p.plx);
  pl_actual_median(i) = median(path_lengths{i});
  inpl_actual_median(i) = median(internode_lengths{i});
  pl_actual_iqr(i,:) = prctile(path_lengths{i},[37.5 62.5]);

  pl_hist_roc(i,:) = hist(log10(path_lengths{i}), p.plxs);
  inpl_hist_roc(i,:) = hist(log10(internode_lengths{i}), p.plxs);
  
  node_hist(i,:) = hist(nnodes_skel{i},p.ndx);
end
pl_pdf = pl_hist./repmat(sum(pl_hist,2),[1 p.nplx]);
inpl_pdf = inpl_hist./repmat(sum(inpl_hist,2),[1 p.nplx]);
pl_cdf = cumsum(pl_pdf,2); inpl_cdf = cumsum(inpl_pdf,2);

nd_pdf = node_hist./repmat(sum(node_hist,2),[1 p.nndx]); nd_cdf = cumsum(nd_pdf,2);

% % get actual median above
% [~,k] = min(abs(pl_cdf - 0.5),[],2); pl_median = p.plx(k);
% [~,k] = min(abs(pl_cdf_roc - 0.5),[],2); pl_median_roc = p.plxs(k);

pl_cdf_roc = cumsum(pl_hist_roc,2)./repmat(sum(pl_hist_roc,2),[1 p.nplxs]); 
inpl_cdf_roc = cumsum(inpl_hist_roc,2)./repmat(sum(inpl_hist_roc,2),[1 p.nplxs]); 


if p.plot_efpl_metrics
  max_auroc = zeros(1,ndatasets);
  for i = 1:ndatasets
    max_auroc(i) = sum(inpl_cdf_roc(i,1:end-1).*diff(pl_cdf_roc(i,:)));
  end
  
  % compute aurocs and medians for combined efpls, normalize by best and worst cases
  % compute aurocs for combined efpl distribution versus best case path length distribution
  combined_efpl_norm_med = nan(ndatasets,nparams);
  combined_efpl_med = nan(ndatasets,nparams);
  combined_efpl_norm_auroc = nan(ndatasets,nparams);
  for i = 1:ndatasets
    nthr = length(params{i}); ind = 1:nthr;
    for k=ind
      if p.use_efpl_edges
        % new mode, with one efpl counted per each edge
        cur_efpls = [o{i}.efpl_edges{k,3}{:}]; cur_efpls = cur_efpls(:)/1000;
      else
        % old mode, with one efpl counted per each connected path
        cur_efpls = o{i}.efpls{k,3}/1000;
      end
      % remove really small internode distances (nodes next to each other or zeros if not counting error half edges).
      cur_efpls = cur_efpls(cur_efpls >= minPL);
      
      combined_efpl_med(i,k) = median(cur_efpls);
      %   pl_actual_median(i) = median(path_lengths{i});
      %   inpl_actual_median(i) = median(internode_lengths{i});
      % make norm efpl med range:
      %   1 (median same as skeleton path lengths, best case, distribution)
      %   0 (median same as half path lengths)
      tmp = (combined_efpl_med(i,k) - inpl_actual_median(i)) / (pl_actual_median(i) - inpl_actual_median(i));
      %tmp = log10(tmp + 1e-10);
      %n = 2; logn = 10^-n; tmp(tmp < logn) = logn; tmp = (log10(tmp) + n) / n;
      combined_efpl_norm_med(i,k) = tmp;
      
      tmp = hist(log10(cur_efpls), p.plxs); tmp = cumsum(tmp)./sum(tmp);
      % make norm auroc range:
      %   1 (completely overlapping with skeleton path lengths, best case, distribution)
      %   0 (completely overlapping with half path lengths, worst case, normalize by worst case auroc)
      tmp = 1 - 2*(sum(tmp(1:end-1).*diff(pl_cdf_roc(i,:))) - 0.5) / max_auroc(i);
      %tmp = log10(tmp + 1e-10);
      %n = 2; logn = 10^-n; tmp(tmp < logn) = logn; tmp = (log10(tmp) + n) / n;
      combined_efpl_norm_auroc(i,k) = tmp;
    end
    
    if p.plot_efpl_metrics_log
      % normalize these metrics to [0 1] on a log scale, per dataset.
      % these metrics are too sensitive on a linear scale.
      % this is why they were not used in the ECS paper.
      tmp = combined_efpl_norm_med(i,:);
      n = min(tmp(tmp>0)); tmp(tmp < n) = n; logn = log10(n); tmp = -(log10(tmp) - logn) / logn;
      combined_efpl_norm_med(i,:) = tmp;
      tmp = combined_efpl_norm_auroc(i,:);
      n = min(tmp(tmp>0)); tmp(tmp < n) = n; logn = log10(n); tmp = -(log10(tmp) - logn) / logn;
      combined_efpl_norm_auroc(i,:) = tmp;
    end
  end
  
end % if plot efpl metrics

if p.plot_efpl_diameters
  % make 3 density plots, less than optimal threshold, at optimal threshold, greater than optimal threshold
  [~,mi] = max(combined_eftpl,[],2); % threshold on max eftpl
  error_free_diameters = cell(ndatasets,3); efpl_edges = cell(ndatasets,3);
  error_free_dia_edge_cnt = zeros(ndatasets,3); error_free_dia_total_cnt = zeros(ndatasets,3);
  error_free_diameter_ithr = mi;
  for k = 1:ndatasets
    nthr = length(params{k}); inds = {1:mi(k)-1 mi(k) mi(k)+1:nthr};
    for j=1:length(inds)
      for i=inds{j}
        tmp = [o{k}.error_free_diameters{i}]; tmp = vertcat(tmp{:});
        error_free_diameters{k,j} = [error_free_diameters{k,j} tmp(:,1)'];
        
        tmp = [o{k}.efpl_edges{i,3}];
        efpl_edges{k,j} = [efpl_edges{k,j} tmp{:}];
      end
      
      %assert( ~any(xor(isnan(error_free_diameters{k,j}), isnan(efpl_edges{k,j}))) ) % before error edges were recorded
      sel = ~isnan(error_free_diameters{k,j});
      assert( all(efpl_edges{k,j}(1,sel) == efpl_edges{k,j}(2,sel)) ); % after error edges were recorded
      error_free_diameters{k,j} = error_free_diameters{k,j}(sel); efpl_edges{k,j} = efpl_edges{k,j}(1,sel);
      error_free_dia_edge_cnt(k,j) = sum(sel); error_free_dia_total_cnt(k,j) = size(sel,2);
    end
  end
end % if plotting error free neurite diameters



ticksel = 1:p.dxticksel:nparams; if ticksel(end) ~= nparams, ticksel = [ticksel nparams]; end
baseno = p.baseno; figno = 0;

if ~isempty(p.save_plot_results)
  save_vars = {
    'ndatasets' 'params' 'path_lengths' 'internode_lengths' 'nskels' 'split_mergers' 'split_mergers_segEM' ...
    'nBGnodes' 'nECSnodes' 'nnodes' 'nnodes_skel' 'names' 'split_mergers_CI' 'split_mergers_segEM_CI' 'nparams' ...
    'split_er' 'split_er_CI' 'merge_fracnodes' 'split_fracnodes' 'are' 'are_CI' 'combined_eftpl' 'norm_params' ...
    'nlabels' ...
  };
  for i=1:length(save_vars)
    po.(save_vars{i}) = eval(save_vars{i});
  end
  
  save(p.save_plot_results,'p','po');
  return % save plot meta only
end



figure(baseno+figno); figno = figno+1; clf
subplot(1,2,1);
plot(p.ndx,nd_cdf);
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(median(nnodes_skel{1}),0.5,'x'); plot(median(nnodes_skel{2}),0.5,'x');
xlabel('nodes per skeleton');
ylabel('cdf');
set(gca,'ylim',[-0.05 1.05],'ytick',0:0.25:1);
set(gca,'plotboxaspectratio',[1 1 1]);
[~,pt] = kstest2(nnodes_skel{1},nnodes_skel{2});
title(sprintf('none: %d nodes median %d\nhuge: %d nodes median %d\nks2 p = %g',...
  sum(nnodes_skel{1}),median(nnodes_skel{1}),sum(nnodes_skel{2}),median(nnodes_skel{2}),pt));
legend(names)

subplot(1,2,2);
plot(p.plx,pl_cdf);
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(p.plx,inpl_cdf,'--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(log10(pl_actual_median(1)),0.5,'x'); plot(log10(pl_actual_median(2)),0.5,'x');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(log10(inpl_actual_median(1)),0.5,'x'); plot(log10(inpl_actual_median(2)),0.5,'x');
if length(path_lengths{1}) > 1
  [~,pt] = kstest2(log10(path_lengths{1}),log10(path_lengths{2}));
else
  pt = inf;
end
if length(internode_lengths{1}) > 1
  [~,pin] = kstest2(log10(internode_lengths{1}),log10(internode_lengths{2}));
else
  pin = inf;
end
xlabel('path length (log10 um)');
ylabel('cdf');
%set(gca,'xlim',[p.plx(1) p.plx(end)]);
set(gca,'xlim',[-2 p.plx(end)]);
set(gca,'ylim',[-0.05 1.05],'ytick',0:0.25:1);
set(gca,'plotboxaspectratio',[1 1 1]);
title(sprintf('none: %d skels, median %.2f (%.4f)\nhuge: %d skels, median %.2f (%.4f)\nks2 p = %g %g\niqr %g-%g %g-%g',...
  nskels(1),pl_actual_median(1),inpl_actual_median(1),nskels(2),pl_actual_median(2),inpl_actual_median(2),pt,pin,...
  pl_actual_iqr(1,1),pl_actual_iqr(1,2),pl_actual_iqr(2,1),pl_actual_iqr(2,2)));

% subplot(2,2,4);
% plot(pl_cdf_roc(1,:),inpl_cdf_roc(1,:));
% hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot(pl_cdf_roc(2,:),inpl_cdf_roc(2,:));
% xlabel('skels (best) cdf'); ylabel('internodes (worst) cdf');
% set(gca,'ylim',[-0.05 1.05],'xlim',[-0.05 1.05]);
% set(gca,'plotboxaspectratio',[1 1 1]);
% title(sprintf('AUROC = %.4f, %.4f',max_auroc(1),max_auroc(2)));








figure(baseno+figno); figno = figno+1; clf
subplot(2,2,1);
sumSM = split_er+merge_fracnodes;
[m,mi] = min(sumSM,[],2);
minSM = [[split_er(1,mi(1)); merge_fracnodes(1,mi(1))] ...
  [split_er(2,mi(2)); merge_fracnodes(2,mi(2))]];
plot(split_er', merge_fracnodes', '-o', 'markersize', 2);
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(minSM(1,1),minSM(2,1),'x');
hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(minSM(1,2),minSM(2,2),'x');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(squeeze(split_er_CI(1,:,1)),squeeze(merge_fracnodes_CI(1,:,1)),'--');
hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(squeeze(split_er_CI(2,:,1)),squeeze(merge_fracnodes_CI(2,:,1)),'--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(squeeze(split_er_CI(1,:,2)),squeeze(merge_fracnodes_CI(1,:,2)),'--');
hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(squeeze(split_er_CI(2,:,2)),squeeze(merge_fracnodes_CI(2,:,2)),'--');
set(gca,'plotboxaspectratio',[1 1 1]);
set(gca,'ylim',[-0.05 1.05],'xlim',[-0.05 1.05]);
xlabel('split edges'); ylabel('merged nodes');
title(sprintf('maxd=%g\n%g=%g+%g %g=%g+%g\n@thr=%g %g',abs(m(2)-m(1)),m(1),...
  minSM(1,1),minSM(2,1),m(2),minSM(1,2),minSM(2,2),params{1}(mi(1)),params{2}(mi(2))));
legend(names)

% %figure(baseno+figno); figno = figno+1; clf
% subplot(2,2,3);
% plot(norm_params',combined_eftpl');
% hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% % plot([ithr_minmergers ithr_minmergers]',repmat([-0.05;0.55],[1 ndatasets]),'--');
% hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot(norm_params',squeeze(combined_eftpl_CI(:,:,1))','--');
% hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot(norm_params',squeeze(combined_eftpl_CI(:,:,2))','--');
% set(gca,'plotboxaspectratio',[1 1 1]);
% ylabel('combined eftpl (%PL)');
% if p.param_name
%   set(gca,'xtick',ticksel,'xticklabel',params{1}(ticksel)); xlim([0.5 nparams+0.5])
%   xlabel(p.param_name)
% else
%   xlabel('norm parameter')
% end
% [m,mi] = max(combined_eftpl,[],2);
% set(gca,'ylim',[-0.025 0.8]); box off
% title(sprintf('maxd=%g\n%g %g\n@thr=%g %g',abs(m(2)-m(1)),m(1),m(2),...
%   params{1}(mi(1)),params{2}(mi(2))));

%figure(baseno+figno); figno = figno+1; clf
subplot(2,2,2);
plot(split_er',combined_eftpl', '-o', 'markersize', 2);
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot([ithr_minmergers ithr_minmergers]',repmat([-0.05;0.55],[1 ndatasets]),'--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(squeeze(split_er_CI(:,:,1))',squeeze(combined_eftpl_CI(:,:,1))','--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(squeeze(split_er_CI(1,:,2))',squeeze(combined_eftpl_CI(:,:,2))','--');
set(gca,'plotboxaspectratio',[1 1 1]);
ylabel('combined eftpl (%PL)');
xlabel('split edges'); set(gca,'xlim',[0 0.5]);
[m,mi] = max(combined_eftpl,[],2);
set(gca,'ylim',[-0.025 0.8]); box off
title(sprintf('maxd=%g\n%g %g\n@split edges=%g %g',abs(m(2)-m(1)),m(1),m(2),...
  split_er(1,mi(1)),split_er(2,mi(2))));

%figure(baseno+figno); figno = figno+1; clf
subplot(2,2,4);
plot(merge_fracnodes',combined_eftpl', '-o', 'markersize', 2);
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot([ithr_minmergers ithr_minmergers]',repmat([-0.05;0.55],[1 ndatasets]),'--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(squeeze(split_er_CI(:,:,1))',squeeze(combined_eftpl_CI(:,:,1))','--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(squeeze(split_er_CI(1,:,2))',squeeze(combined_eftpl_CI(:,:,2))','--');
set(gca,'plotboxaspectratio',[1 1 1]);
ylabel('combined eftpl (%PL)');
xlabel('merged nodes'); set(gca,'xlim',[0 0.5]);
[m,mi] = max(combined_eftpl,[],2);
set(gca,'ylim',[-0.025 0.8],'xlim',[-0.025 0.5]); box off
title(sprintf('maxd=%g\n%g %g\n@split edges=%g %g',abs(m(2)-m(1)),m(1),m(2),...
  merge_fracnodes(1,mi(1)),merge_fracnodes(2,mi(2))));

subplot(2,2,3);
sumSM = split_fracnodes+merge_fracnodes;
[m,mi] = min(sumSM,[],2);
minSM = [[split_fracnodes(1,mi(1)); merge_fracnodes(1,mi(1))] ...
  [split_fracnodes(2,mi(2)); merge_fracnodes(2,mi(2))]];
plot(split_fracnodes', merge_fracnodes', '-o', 'markersize', 2);
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(minSM(1,1),minSM(2,1),'x');
hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(minSM(1,2),minSM(2,2),'x');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(squeeze(split_er_CI(1,:,1)),squeeze(merge_fracnodes_CI(1,:,1)),'--');
hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(squeeze(split_er_CI(2,:,1)),squeeze(merge_fracnodes_CI(2,:,1)),'--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(squeeze(split_er_CI(1,:,2)),squeeze(merge_fracnodes_CI(1,:,2)),'--');
hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(squeeze(split_er_CI(2,:,2)),squeeze(merge_fracnodes_CI(2,:,2)),'--');
set(gca,'plotboxaspectratio',[1 1 1]);
%set(gca,'ylim',[-0.05 1.05],'xlim',[-0.05 1.05]);
xlabel('splits per node'); ylabel('merged nodes');
title(sprintf('maxd=%g\n%g=%g+%g %g=%g+%g\n@thr=%g %g\nnode radius=%d',abs(m(2)-m(1)),m(1),...
  minSM(1,1),minSM(2,1),m(2),minSM(1,2),minSM(2,2),params{1}(mi(1)),params{2}(mi(2)),p.node_radius));
%legend(names)




if p.plot_efpl_metrics
  
  figure(baseno+figno); figno = figno+1; clf
  subplot(1,2,1);
  plot(norm_params',combined_efpl_norm_med');
  hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
  % % plot([ithr_minmergers ithr_minmergers]',repmat([-0.05;0.55],[1 ndatasets]),'--');
  % hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
  % plot(norm_params',squeeze(combined_eftpl_CI(:,:,1))','--');
  % hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
  % plot(norm_params',squeeze(combined_eftpl_CI(:,:,2))','--');
  set(gca,'plotboxaspectratio',[1 1 1]);
  ylabel('normalized median efpl'); box off
  if p.param_name
    set(gca,'xtick',ticksel,'xticklabel',params{1}(ticksel)); xlim([0.5 nparams+0.5])
    xlabel(p.param_name)
  else
    xlabel('norm parameter')
  end
  [m,mi] = max(combined_efpl_norm_med,[],2);
  %set(gca,'ylim',[-0.00025 0.01]);
  %set(gca,'ylim',[-0.025 1]);
  %set(gca,'yscale','log');
  title(sprintf('maxd=%g\n%g %g\n@thr=%g %g',abs(m(2)-m(1)),m(1),m(2),...
    params{1}(mi(1)),params{2}(mi(2))));
  
  subplot(1,2,2);
  plot(norm_params',combined_efpl_norm_auroc');
  hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
  % % plot([ithr_minmergers ithr_minmergers]',repmat([-0.05;0.55],[1 ndatasets]),'--');
  % hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
  % plot(norm_params',squeeze(combined_eftpl_CI(:,:,1))','--');
  % hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
  % plot(norm_params',squeeze(combined_eftpl_CI(:,:,2))','--');
  set(gca,'plotboxaspectratio',[1 1 1]);
  ylabel('normalized auroc'); box off;
  if p.param_name
    set(gca,'xtick',ticksel,'xticklabel',params{1}(ticksel)); xlim([0.5 nparams+0.5])
    xlabel(p.param_name)
  else
    xlabel('norm parameter')
  end
  [m,mi] = max(combined_efpl_norm_auroc,[],2);
  %set(gca,'ylim',[-0.0025 0.1]);
  %set(gca,'ylim',[-0.025 1]);
  %set(gca,'yscale','log');
  title(sprintf('maxd=%g\n%g %g\n@thr=%g %g',abs(m(2)-m(1)),m(1),m(2),...
    params{1}(mi(1)),params{2}(mi(2))));
end




figure(baseno+figno); figno = figno+1; clf
subplot(2,2,1);
plot(squeeze(are_precrec(:,:,1))',squeeze(are_precrec(:,:,2))', '-o', 'markersize', 2);
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(squeeze(are_precrec_CI(:,:,1,1))',squeeze(are_precrec_CI(:,:,2,1))','--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(squeeze(are_precrec_CI(:,:,1,2))',squeeze(are_precrec_CI(:,:,2,2))','--');
set(gca,'ylim',[-0.05 1.05],'xlim',[-0.05 1.05]);
xlabel('1-recall'); ylabel('1-precision');
title('ARE precrec')
set(gca,'plotboxaspectratio',[1 1 1]);
legend(names)

%figure(baseno+figno); figno = figno+1; clf
subplot(2,2,3);
plot(norm_params',are','-o', 'markersize', 2);
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(norm_params',squeeze(are_CI(:,:,1))','--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(norm_params',squeeze(are_CI(:,:,2))','--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% if exist('ithr_minmergers_all','var')
%   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   plot([ithr_minmergers_all(:,i) ithr_minmergers_all(:,i)]',repmat([0.5;1.05],[1 ndatasets]),'--');
% end
set(gca,'plotboxaspectratio',[1 1 1]);
[m,mi] = min(are,[],2);
box off; title(sprintf('maxd=%g\n%g %g\nthr=%g %g',abs(m(2)-m(1)),m(1),m(2),...
 params{1}(mi(1)),params{2}(mi(2))));
if p.param_name
  set(gca,'xtick',ticksel,'xticklabel',params{1}(ticksel)); xlim([0.5 nparams+0.5])
  xlabel(p.param_name)
else
  xlabel('norm parameter')
end
if p.are_sum 
  set(gca,'ylim',[0.4 2.025]);
  ylabel('ARE, [0,2], 1-sum prec/rec');
else
  set(gca,'ylim',[0.4 1.025]);
  ylabel('ARE, [0,1], 1-Fscore');
end

subplot(2,2,2)
plot(split_er',are','-o', 'markersize', 2);
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot([ithr_minmergers ithr_minmergers]',repmat([-0.05;0.55],[1 ndatasets]),'--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot(squeeze(combined_eftpl_CI(:,:,1))','--');
% hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot(squeeze(combined_eftpl_CI(:,:,2))','--');
set(gca,'plotboxaspectratio',[1 1 1]); box off
xlabel('split edges'); set(gca,'xlim',[0 0.5]);
[m,mi] = max(combined_eftpl,[],2);
plot(split_er(1,mi(1)),are(1,mi(1)),'x');
plot(split_er(2,mi(2)),are(2,mi(2)),'x');
title(sprintf('max tefpl=%g %g\n@nlabels=%g %g',m(1),m(2),...
  nlabels(1,mi(1)),nlabels(2,mi(2))));
if p.are_sum 
  set(gca,'ylim',[0.4 2.025]);
  ylabel('ARE, [0,2], 1-sum prec/rec');
else
  set(gca,'ylim',[0.4 1.025]);
  ylabel('ARE, [0,1], 1-Fscore');
end

subplot(2,2,4)
plot(merge_fracnodes',are','-o', 'markersize', 2);
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot([ithr_minmergers ithr_minmergers]',repmat([-0.05;0.55],[1 ndatasets]),'--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot(squeeze(combined_eftpl_CI(:,:,1))','--');
% hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot(squeeze(combined_eftpl_CI(:,:,2))','--');
set(gca,'plotboxaspectratio',[1 1 1]); box off
xlabel('merged nodes'); set(gca,'xlim',[0 0.5]);
[m,mi] = max(combined_eftpl,[],2);
plot(merge_fracnodes(1,mi(1)),are(1,mi(1)),'x');
plot(merge_fracnodes(2,mi(2)),are(2,mi(2)),'x');
title(sprintf('max tefpl=%g %g\n@nlabels=%g %g',m(1),m(2),...
  nlabels(1,mi(1)),nlabels(2,mi(2))));
if p.are_sum 
  set(gca,'ylim',[0.4 2.025]);
  ylabel('ARE, [0,2], 1-sum prec/rec');
else
  set(gca,'ylim',[0.4 1.025]);
  ylabel('ARE, [0,1], 1-Fscore');
end







figure(baseno+figno); figno = figno+1; clf
dolog = false;
if dolog
  lnlabels = log10(nlabels); str = 'log10 nsupervoxels'; lim = [4.25 6];
else
  lnlabels = nlabels; str = 'nsupervoxels'; lim = [0 600000];
end
subplot(2,2,1)
plot(norm_params',lnlabels','-o', 'markersize', 2);
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot([ithr_minmergers ithr_minmergers]',repmat([-0.05;0.55],[1 ndatasets]),'--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot(squeeze(combined_eftpl_CI(:,:,1))','--');
% hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot(squeeze(combined_eftpl_CI(:,:,2))','--');
set(gca,'plotboxaspectratio',[1 1 1]); box off
ylabel(str);
if p.param_name
  set(gca,'xtick',ticksel,'xticklabel',params{1}(ticksel)); xlim([0.5 nparams+0.5])
  xlabel(p.param_name)
else
  xlabel('norm parameter')
end
% [m,mi] = max(combined_eftpl,[],2);
% set(gca,'ylim',[-0.025 0.75]); box off
% title(sprintf('maxd=%g\n%g %g\nthr=%g %g',abs(diff(m)),m(1),m(2),...
%   params{1}(mi(1)),params{2}(mi(2))));
legend(names)

subplot(2,2,2)
plot(split_er',lnlabels','-o', 'markersize', 2);
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot([ithr_minmergers ithr_minmergers]',repmat([-0.05;0.55],[1 ndatasets]),'--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot(squeeze(combined_eftpl_CI(:,:,1))','--');
% hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot(squeeze(combined_eftpl_CI(:,:,2))','--');
set(gca,'plotboxaspectratio',[1 1 1]); box off
ylabel(str); set(gca,'ylim',lim);
xlabel('split edges'); set(gca,'xlim',[0 0.5]);
[m,mi] = max(combined_eftpl,[],2);
plot(split_er(1,mi(1)),lnlabels(1,mi(1)),'x');
plot(split_er(2,mi(2)),lnlabels(2,mi(2)),'x');
title(sprintf('max tefpl=%g %g\n@nlabels=%g %g',m(1),m(2),...
  nlabels(1,mi(1)),nlabels(2,mi(2))));

subplot(2,2,4)
plot(merge_fracnodes',lnlabels','-o', 'markersize', 2);
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot([ithr_minmergers ithr_minmergers]',repmat([-0.05;0.55],[1 ndatasets]),'--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot(squeeze(combined_eftpl_CI(:,:,1))','--');
% hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot(squeeze(combined_eftpl_CI(:,:,2))','--');
set(gca,'plotboxaspectratio',[1 1 1]); box off
ylabel(str); set(gca,'ylim',lim);
xlabel('merged nodes'); set(gca,'xlim',[0 0.5]);
[m,mi] = max(combined_eftpl,[],2);
plot(merge_fracnodes(1,mi(1)),lnlabels(1,mi(1)),'x');
plot(merge_fracnodes(2,mi(2)),lnlabels(2,mi(2)),'x');
title(sprintf('max tefpl=%g %g\n@nlabels=%g %g',m(1),m(2),...
  nlabels(1,mi(1)),nlabels(2,mi(2))));

subplot(2,2,3)
plot(lnlabels',combined_eftpl', '-o', 'markersize', 2);
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot([ithr_minmergers ithr_minmergers]',repmat([-0.05;0.55],[1 ndatasets]),'--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot(squeeze(combined_eftpl_CI(:,:,1))','--');
% hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot(squeeze(combined_eftpl_CI(:,:,2))','--');
set(gca,'plotboxaspectratio',[1 1 1]); box off
xlabel(str); set(gca,'xlim',lim)
ylabel('tefpl'); set(gca,'ylim',[0 0.8]);
[m,mi] = max(combined_eftpl,[],2);
plot(lnlabels(1,mi(1)),combined_eftpl(1,mi(1)),'x');
plot(lnlabels(2,mi(2)),combined_eftpl(2,mi(2)),'x');
title(sprintf('max tefpl=%g %g\n@nlabels=%g %g',m(1),m(2),...
  nlabels(1,mi(1)),nlabels(2,mi(2))));



if p.plot_efpl_diameters
  figure(baseno+figno); figno = figno+1; clf
  nc=64; colormap([1 1 1; jet(nc)]);

  ebins_efpl = 2:0.1:5; bins_efpl = ebins_efpl(1:end-1) + (ebins_efpl(2)-ebins_efpl(1))/2;
  ebins_dia = 1.5:0.1:4.5; bins_dia = ebins_dia(1:end-1) + (ebins_dia(2)-ebins_dia(1))/2;
  
  % only plot the first few datasets
  nplot = 2;
  for i=1:min([nplot ndatasets])
    % make 3 density plots, less than optimal threshold, at optimal threshold, greater than optimal threshold
    %   error_free_diameters = cell(ndatasets,3); efpl_edges = cell(ndatasets,3);
    %   error_free_diameter_ithr = mi;
    strs = {sprintf('thr < %g, %s',params{i}(error_free_diameter_ithr(i)),names{i}) ...
      sprintf('thr == %g max eftpl',params{i}(error_free_diameter_ithr(i))) ...
      sprintf('thr > %g',params{i}(error_free_diameter_ithr(i)))};
    for j=1:3
      subplot(nplot,3,(i-1)*3 + j)
      N = histcounts2(log10(error_free_diameters{i,j}),log10(efpl_edges{i,j}),ebins_dia,ebins_efpl);
      maxN = max(N(:)); N = N/maxN*(nc-1); sel = (N > 0); N(sel) = fix(N(sel))+2; N(~sel) = 1;
      image(bins_dia,bins_efpl,N'); hc = colorbar;
      set(gca,'dataaspectratio',[1 1 1],'ydir','normal');
      xlabel('log10 diameters (nm)'); ylabel('log10 efpl (nm)');
      set(hc, 'yticklabel', [0 linspace(1,maxN,nc-1)/error_free_dia_total_cnt(i,j)*100], 'ytick',(0:8:nc)+1);
      title(hc,'% edges')
      title(strs{j});
    end
  end
end


if isempty(p.meta_param), return; end

% "meta-plots" of best metric values
figure(baseno+figno); figno = figno+1; clf
dolog = false;
if dolog
  lnlabels = log10(nlabels); str = 'log10 nsupervoxels'; lim = [4.25 6];
else
  lnlabels = nlabels; str = 'nsupervoxels'; lim = [0 600000];
end
dx = p.meta_param(2) - p.meta_param(1);

subplot(2,2,1)
[m,mi] = max(combined_eftpl,[],2);
for k=1:length(p.meta_groups)
  plot(p.meta_param,m(p.meta_groups{k})); hold on
end
set(gca,'plotboxaspectratio',[1 1 1]); box off
set(gca, 'xlim', [p.meta_param(1)-0.25*dx p.meta_param(end)+0.25*dx], ...
  'xtick', p.meta_param);
if ~isempty(p.meta_labels)
  set(gca, 'xticklabel', p.meta_labels);
end
if ~isempty(p.meta_labels)
  set(gca, 'xticklabel', p.meta_labels);
end
ylabel('max eftpl'); xlabel(p.meta_param_label);
legend(p.meta_groups_labels)

subplot(2,2,2)
[m,mi] = min(are,[],2);
for k=1:length(p.meta_groups)
  plot(p.meta_param,m(p.meta_groups{k})); hold on
end
set(gca,'plotboxaspectratio',[1 1 1]); box off
set(gca, 'xlim', [p.meta_param(1)-0.25*dx p.meta_param(end)+0.25*dx], ...
  'xtick', p.meta_param);
if ~isempty(p.meta_labels)
  set(gca, 'xticklabel', p.meta_labels);
end
if ~isempty(p.meta_labels)
  set(gca, 'xticklabel', p.meta_labels);
end
xlabel(p.meta_param_label);
if p.are_sum 
  ylabel('min ARE, 1-sum prec/rec');
else
  ylabel('min ARE, 1-Fscore');
end
legend(p.meta_groups_labels)

subplot(2,2,3)
[m,mi] = max(combined_eftpl,[],2);
for k=1:length(p.meta_groups)
  inds = sub2ind(size(lnlabels),p.meta_groups{k},mi(p.meta_groups{k})');
  plot(p.meta_param,lnlabels(inds)); hold on
end
set(gca,'plotboxaspectratio',[1 1 1]); box off
set(gca, 'xlim', [p.meta_param(1)-0.25*dx p.meta_param(end)+0.25*dx], ...
  'xtick', p.meta_param);
if ~isempty(p.meta_labels)
  set(gca, 'xticklabel', p.meta_labels);
end
if ~isempty(p.meta_labels)
  set(gca, 'xticklabel', p.meta_labels);
end
ylabel([str ' @ max eftpl']); xlabel(p.meta_param_label);
legend(p.meta_groups_labels)




% figure of type of edge errors for ordered edges
[m1,mi1] = max(combined_eftpl,[],2);
[m2,mi2] = min(are,[],2);
metrics = {m1 m2};
minds = {mi1 mi2}; 
mstrs = {'max eftpl', 'min are'};

for k=1:length(p.meta_groups)
  figure(baseno+figno); figno = figno+1; clf
  for x=1:length(metrics)
    mi = minds{x};
    clear error_edge_types
    group_len = length(p.meta_groups{k});
    for j=1:group_len
      g = p.meta_groups{k}(j);
      if j==1
        tmp = [o{g}.error_free_edges{mi(g),1}{:}];
        % not green is edge with split error, not red is edge with merger error
        error_edge_types = ones(group_len, length(tmp), 3);
      end
      error_edge_types(j,:,2) = [o{g}.error_free_edges{mi(g),1}{:}];
      error_edge_types(j,:,1) = [o{g}.error_free_edges{mi(g),2}{:}];
    end
    subplot(1,2,x)
    image(~error_edge_types)
    set(gca,'ytick', p.meta_param);
    if ~isempty(p.meta_labels)
      set(gca, 'yticklabel', p.meta_labels);
    end
    if ~isempty(p.meta_labels)
      set(gca, 'yticklabel', p.meta_labels);
    end
    xlabel('ordered edges'); ylabel(p.meta_param_label);
    title(sprintf('%s edges at %s\n(green split, red merger)', p.meta_groups_labels{k}, mstrs{x}))
  end
end
