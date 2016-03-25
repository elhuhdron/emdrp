
function po = knossos_efpl_plot(pdata,o,p)

useColorOrder = ~verLessThan('matlab','8.4');

po = struct;
ndatasets = length(pdata);
% use old name for parameter dimension, typically thresholds.
thresholds = cell(1,ndatasets);
path_lengths = cell(1,ndatasets);       % actually "best case"
internode_lengths = cell(1,ndatasets);  % actually "worst case"
nskels = zeros(1,ndatasets);
split_mergers = cell(1,ndatasets);
split_mergers_segEM = cell(1,ndatasets);
nBGnodes = cell(1,ndatasets); nECSnodes = cell(1,ndatasets);
nnodes = zeros(1,ndatasets);
nnodes_skel = cell(1,ndatasets);

split_mergers_CI = cell(1,ndatasets);
split_mergers_segEM_CI = cell(1,ndatasets);

for i = 1:ndatasets
  use = ~o{i}.empty_things_use; nskels(i) = sum(use);
  path_lengths{i} = o{i}.efpl_bestcase / 1000;
  internode_lengths{i} = o{i}.efpl_worstcase / 1000;
  
  % remove really small internode distances that are just from dropping nodes right next to each other
  minPL = 10^(p.plx(1)-p.dplx(1)/2);
  path_lengths{i} = path_lengths{i}(path_lengths{i} >= minPL);
  internode_lengths{i} = internode_lengths{i}(internode_lengths{i} >= minPL);
  
  split_mergers{i} = o{i}.nSMs;
  split_mergers_segEM{i} = o{i}.nSMs_segEM;
  nBGnodes{i} = o{i}.nBGnodes; nECSnodes{i} = o{i}.nECSnodes;
  nnodes(i) = sum(o{i}.nnodes_use);
  nnodes_skel{i} = o{i}.nnodes_use(~o{i}.empty_things_use);
  
  split_mergers_CI{i} = o{i}.nSMs_CI;
  split_mergers_segEM_CI{i} = o{i}.nSMs_segEM_CI;
  
  thresholds{i} = o{i}.thresholds;
end

% for single matrices over parameter dimension, use max length and fill shorter rows with NaN
nthresholds = max(cellfun(@length, thresholds));

split_er = nan(ndatasets,nthresholds); merge_er = nan(ndatasets,nthresholds);
split_er_CI = nan(ndatasets,nthresholds,2); merge_er_CI = nan(ndatasets,nthresholds,2);
merge_fracnodes = nan(ndatasets,nthresholds); merge_fracnodes_CI = nan(ndatasets,nthresholds,2);
split_fracnodes = nan(ndatasets,nthresholds); split_fracnodes_CI = nan(ndatasets,nthresholds,2);
are = nan(ndatasets,nthresholds); are_precrec = nan(ndatasets,nthresholds,2);
are_CI = nan(ndatasets,nthresholds,2); are_precrec_CI = nan(ndatasets,nthresholds,2,2);
combined_eftpl = nan(ndatasets,nthresholds); combined_eftpl_CI = nan(ndatasets,nthresholds,2);
    
for k = 1:ndatasets
  for j=1:length(o{k}.thresholds)
    ind=j; 
    split_er(k,j) = o{k}.error_rates(ind,1);
    merge_er(k,j) = o{k}.error_rates(ind,2);
    split_er_CI(k,j,:) = o{k}.error_rate_CI(ind,1,:);
    merge_er_CI(k,j,:) = o{k}.error_rate_CI(ind,2,:);
    merge_fracnodes(k,j) = o{k}.nSMs(ind,2)/nnodes(k);
    merge_fracnodes_CI(k,j,:) = o{k}.nSMs_CI(ind,2,:);
    split_fracnodes(k,j) = o{k}.nSMs(ind,1)/nnodes(k);
    split_fracnodes_CI(k,j,:) = o{k}.nSMs_CI(ind,1,:);
    are(k,j) = o{k}.are(ind); are_CI(k,j,:) =  o{k}.are_CI(ind,:);
    are_precrec(k,j,:) = o{k}.are_precrec(ind,:); are_precrec_CI(k,j,:,:) = o{k}.are_precrec_CI(ind,:,:);
    combined_eftpl(k,j) = sum(o{k}.eftpl(ind,:,3))/sum(o{k}.path_length_use);
    combined_eftpl_CI(k,j,:) = o{k}.eftpl_CI(ind,3,:);
  end
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

pl_cdf_roc = cumsum(pl_hist_roc,2)./repmat(sum(pl_hist_roc,2),[1 p.nplxs]); 
inpl_cdf_roc = cumsum(inpl_hist_roc,2)./repmat(sum(inpl_hist_roc,2),[1 p.nplxs]); 

nd_pdf = node_hist./repmat(sum(node_hist,2),[1 p.nndx]); nd_cdf = cumsum(nd_pdf,2);

% get actual median above
[~,k] = min(abs(pl_cdf - 0.5),[],2); pl_median = p.plx(k);
[~,k] = min(abs(pl_cdf_roc - 0.5),[],2); pl_median_roc = p.plxs(k);

max_auroc = zeros(1,ndatasets);
for i = 1:ndatasets
  max_auroc(i) = sum(inpl_cdf_roc(i,1:end-1).*diff(pl_cdf_roc(i,:)));
end

ticksel = 1:nthresholds;  % meh





baseno = 1000; figno = 0;

figure(baseno+figno); figno = figno+1; clf
subplot(1,2,1);
plot(p.ndx,nd_pdf);
xlabel('nodes per skeleton');
ylabel('pdf');
set(gca,'plotboxaspectratio',[1 1 1]);
subplot(1,2,2);
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

figure(baseno+figno); figno = figno+1; clf
subplot(1,2,1);
plot(p.plx,pl_pdf);
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(p.plx,inpl_pdf,'--');
xlabel('path length (log10 um)');
ylabel('pdf');
% title(sprintf('none (bl): mean %.2f, sem %.2f\nhuge (br): mean %.2f, sem %.2f',...
%   mean(log10(path_lengths{1})),std(log10(path_lengths{1}))/sqrt(nskels(1)),...
%   mean(log10(path_lengths{2})),std(log10(path_lengths{2}))/sqrt(nskels(2))));
set(gca,'xlim',[p.plx(1) p.plx(end)]);
set(gca,'plotboxaspectratio',[1 1 1]);
subplot(1,2,2);
plot(p.plx,pl_cdf);
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(p.plx,inpl_cdf,'--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(log10(pl_actual_median(1)),0.5,'x'); plot(log10(pl_actual_median(2)),0.5,'x');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(log10(inpl_actual_median(1)),0.5,'x'); plot(log10(inpl_actual_median(2)),0.5,'x');
[~,pt] = kstest2(log10(path_lengths{1}),log10(path_lengths{2}));
if length(internode_lengths{1}) > 1
  [~,pin] = kstest2(log10(internode_lengths{1}),log10(internode_lengths{2}));
else
  pin = inf;
end
xlabel('path length (log10 um)');
ylabel('cdf');
%set(gca,'xlim',[p.plx(1) p.plx(end)]);
set(gca,'xlim',[-1 p.plx(end)]);
set(gca,'ylim',[-0.05 1.05],'ytick',0:0.25:1);
set(gca,'plotboxaspectratio',[1 1 1]);
title(sprintf('none: %d skels, median %.2f (%.4f)\nhuge: %d skels, median %.2f (%.4f)\nks2 p = %g %g\niqr %g-%g %g-%g',...
  nskels(1),pl_actual_median(1),inpl_actual_median(1),nskels(2),pl_actual_median(2),inpl_actual_median(2),pt,pin,...
  pl_actual_iqr(1,1),pl_actual_iqr(1,2),pl_actual_iqr(2,1),pl_actual_iqr(2,2)));


figure(baseno+figno); figno = figno+1; clf
subplot(1,2,1);
plot(p.plxs,pl_cdf_roc);
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(p.plxs,inpl_cdf_roc,'--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(pl_median_roc(1),0.5,'x'); plot(pl_median_roc(2),0.5,'x');
%[~,p] = kstest2(log10(path_lengths{1}),log10(path_lengths{2}));
%[~,phn] = kstest2(log10(half_node_path_lengths{1}),log10(half_node_path_lengths{2}));
xlabel('path length (log10 um)');
ylabel('cdf');
set(gca,'xlim',[p.plxs(1) p.plxs(end)]);
set(gca,'ylim',[-0.05 1.05]);
set(gca,'plotboxaspectratio',[1 1 1]);
title(sprintf('none: %d skels, median %.2f (%.2f)\nhuge: %d skels, median %.2f (%.2f)',...
  nskels(1),10^pl_median_roc(1),pl_actual_median(1),nskels(2),10^pl_median_roc(2),pl_actual_median(2)));
subplot(1,2,2);
plot(pl_cdf_roc(1,:),inpl_cdf_roc(1,:));
hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(pl_cdf_roc(2,:),inpl_cdf_roc(2,:));
xlabel('skels (best) cdf'); ylabel('internodes (worst) cdf');
set(gca,'ylim',[-0.05 1.05],'xlim',[-0.05 1.05]);
set(gca,'plotboxaspectratio',[1 1 1]);
title(sprintf('AUROC = %.4f, %.4f',max_auroc(1),max_auroc(2)));








figure(baseno+figno); figno = figno+1; clf
sumSM = split_er+merge_fracnodes;
[m,mi] = min(sumSM,[],2);
minSM = [[split_er(1,mi(1)); merge_fracnodes(1,mi(1))] ...
  [split_er(2,mi(2)); merge_fracnodes(2,mi(2))]];
plot(split_er(1,:),merge_fracnodes(1,:),'--.');
hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(split_er(2,:),merge_fracnodes(2,:),'--.');
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
title(sprintf('maxd=%g\n%g=%g+%g %g=%g+%g\nthr=%g %g',abs(diff(m)),m(1),...
  minSM(1,1),minSM(2,1),m(2),minSM(1,2),minSM(2,2),thresholds{1}(mi(1)),thresholds{2}(mi(2))));





figure(baseno+figno); figno = figno+1; clf
plot(are');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(squeeze(are_CI(:,:,1))','--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(squeeze(are_CI(:,:,2))','--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end

% if exist('ithr_minmergers_all','var')
%   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   plot([ithr_minmergers_all(:,i) ithr_minmergers_all(:,i)]',repmat([0.5;1.05],[1 ndatasets]),'--');
% end
set(gca,'plotboxaspectratio',[1 1 1]);
[m,mi] = min(are',[],1);
title(sprintf('maxd=%g\n%g %g\nthr=%g %g',abs(diff(m)),m(1),m(2),...
 thresholds{1}(mi(1)),thresholds{2}(mi(2))));
if p.set_thresholds_axis
  set(gca,'xtick',ticksel,'xticklabel',thresholds{1}(ticksel)); xlim([0.5 nthresholds+0.5])
  xlabel('thresholds')
else
  xlabel('parameter')
end
set(gca,'ylim',[0.6 1.025]); box off;
ylabel('ARE, [0,1], 1-Fscore');




figure(baseno+figno); figno = figno+1; clf
plot(1-squeeze(are_precrec(:,:,1))',1-squeeze(are_precrec(:,:,2))');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(1-squeeze(are_precrec_CI(:,:,1,1))',1-squeeze(are_precrec_CI(:,:,2,1))','--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(1-squeeze(are_precrec_CI(:,:,1,2))',1-squeeze(are_precrec_CI(:,:,2,2))','--');
set(gca,'ylim',[-0.05 1.05],'xlim',[-0.05 1.05]);
xlabel('1-recall'); ylabel('1-precision');
title('ARE precrec')




figure(baseno+figno); figno = figno+1; clf
plot(combined_eftpl');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% plot([ithr_minmergers ithr_minmergers]',repmat([-0.05;0.55],[1 ndatasets]),'--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(squeeze(combined_eftpl_CI(:,:,1))','--');
hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
plot(squeeze(combined_eftpl_CI(:,:,2))','--');
set(gca,'plotboxaspectratio',[1 1 1]);
ylabel('combined eftpl (%PL)');
if p.set_thresholds_axis
  set(gca,'xtick',ticksel,'xticklabel',thresholds{1}(ticksel)); xlim([0.5 nthresholds+0.5])
  xlabel('thresholds')
else
  xlabel('parameter')
end
[m,mi] = max(combined_eftpl,[],2);
set(gca,'ylim',[-0.025 0.5]); box off
title(sprintf('maxd=%g\n%g %g\nthr=%g %g',abs(diff(m)),m(1),m(2),...
  thresholds{1}(mi(1)),thresholds{2}(mi(2))));

