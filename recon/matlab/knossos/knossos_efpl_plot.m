
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
are = cell(1,ndatasets); are_precrec = cell(1,ndatasets);
nBGnodes = cell(1,ndatasets); nECSnodes = cell(1,ndatasets);
nnodes = zeros(1,ndatasets);
nnodes_skel = cell(1,ndatasets);

are_CI = cell(1,ndatasets); are_precrec_CI = cell(1,ndatasets);
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
  are{i} = o{i}.are; are_precrec{i} = o{i}.are_precrec;
  nBGnodes{i} = o{i}.nBGnodes; nECSnodes{i} = o{i}.nECSnodes;
  nnodes(i) = sum(o{i}.nnodes_use);
  nnodes_skel{i} = o{i}.nnodes_use(~o{i}.empty_things_use);
  
  are_CI{i} = o{i}.are_CI; are_precrec_CI{i} = o{i}.are_precrec_CI;
  split_mergers_CI{i} = o{i}.nSMs_CI;
  split_mergers_segEM_CI{i} = o{i}.nSMs_segEM_CI;
  
  thresholds{i} = o{i}.thresholds;
end

% for single matrices over parameter dimension, use max length and fill shorter rows with NaN
nthresholds = max(cellfun(@sum, thresholds));

split_er = nan(ndatasets,nthresholds); merge_er = nan(ndatasets,nthresholds);
split_er_CI = nan(ndatasets,nthresholds,2); merge_er_CI = nan(ndatasets,nthresholds,2);
merge_fracnodes = nan(ndatasets,nthresholds); merge_fracnodes_CI = nan(ndatasets,nthresholds,2);
split_fracnodes = nan(ndatasets,nthresholds); split_fracnodes_CI = nan(ndatasets,nthresholds,2);
    
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
  end
end




% split_mergers = cat(3,split_mergers{:});
% split_mergers_segEM = cat(3,split_mergers_segEM{:});
% nBGnodes = [nBGnodes{:}]; nECSnodes = [nECSnodes{:}];
% are = [are{:}]; are_precrec = cat(3,are_precrec{:});
% 
% are_CI = cat(3,are_CI{:}); are_precrec_CI = cat(4,are_precrec_CI{:});
% split_mergers_CI = cat(4,split_mergers_CI{:});
% split_mergers_segEM_CI = cat(4,split_mergers_segEM_CI{:});

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











% if useTmin
%   figure(baseno+figno); figno = figno+1; clf
%   figure(baseno+figno); figno = figno+1; clf
%   for i=1:nthresholds
%     if nTmins < 2
%       ind = i;
%     else
%       % this is specific to how many parameter dimensions there are, currently just Tmin and threshold
%       ind = sub2ind([nthresholds nTmins],i,iTmin);
%     end
% 
%     figure(baseno+figno-1);
%     subplot(4,5,i);
%     [~, split_efpl_cdf, ~, merge_efpl_cdf, ~, combined_efpl_cdf, ~,~,~,~,~] = ...
%       getSplitMergeDistsAtIndAll(o, ind, ndatasets, p.plx, p.dplx);
%     %     plot(p.plx,pl_cdfc);
%     %     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     %     plot(p.plx,inpl_cdfc,'--');
%     %     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(p.plx,pl_cdf);
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(p.plx,inpl_cdf,'--');
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(p.plx,split_efpl_cdf,':');
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(p.plx,merge_efpl_cdf,'-.');
%     %     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     %     plot(p.plx,combined_efpl_cdf,'o');
%     xlabel('path length (log10 um)');
%     ylabel('cdf');
%     set(gca,'xlim',[p.plx(1) p.plx(end)]);
%     %set(gca,'xlim',[-5 5]);
%     set(gca,'ylim',[-0.05 1.05]);
%     set(gca,'plotboxaspectratio',[1 1 1]);
%     if nTmins < 2
%       title(sprintf('thr %.8f',thresholds(i)));
%     else
%       title(sprintf('thr %.8f, Tmin %d',thresholds(i),Tmins(iTmin)));
%     end
%     
%     figure(baseno+figno-2);
%     subplot(4,5,i);
%     [~, split_efpl_cdf, ~, merge_efpl_cdf, ~, ~, ~,~,~,~,~] = ...
%       getSplitMergeDistsAtIndAll(o, ind, ndatasets, p.plxs, p.dplxs);
%     
%     plot(pl_cdf_roc(1,:),split_efpl_cdf(1,:));
%     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(pl_cdf_roc(2,:),split_efpl_cdf(2,:));
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(pl_cdf_roc(1,:),merge_efpl_cdf(1,:),'-.');
%     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(pl_cdf_roc(2,:),merge_efpl_cdf(2,:),'-.');
%     xlabel('skels (best) cdf'); ylabel('current cdf');
%     
%     set(gca,'ylim',[-0.05 1.05],'xlim',[-0.05 1.05]);
%     set(gca,'plotboxaspectratio',[1 1 1]);
%     if nTmins < 2
%       title(sprintf('thr %.8f',thresholds(i)));
%     else
%       title(sprintf('thr %.8f, Tmin %d',thresholds(i),Tmins(iTmin)));
%     end
%   end
%   figure(baseno+figno-1);
%   axes('Position',[0 0 1 1],'Visible','off');
%   text(0.3, 0.975, sprintf('none(bl) huge(br) half edge (--) full skels (-) split (.) merge (-.)'))
%   figure(baseno+figno-2);
%   axes('Position',[0 0 1 1],'Visible','off');
%   text(0.3, 0.975, sprintf('none(bl) huge(br) split (-) merge (-.)'))
%   
% else
%   
%   figno = figno+1;
%   figure(baseno+figno); figno = figno+1; clf
%   figure(baseno+figno); figno = figno+1; clf
%   figure(baseno+figno); figno = figno+1; clf
%   figure(baseno+figno); figno = figno+1; clf
%   figure(baseno+figno); figno = figno+1; clf
%   figure(baseno+figno); figno = figno+1; clf
%   figure(baseno+figno); figno = figno+1; clf
%   figure(baseno+figno); figno = figno+1; clf
%   figure(baseno+figno); figno = figno+1; clf
%   figure(baseno+figno); figno = figno+1; clf
%   %ticksel = 1:3:nthresholds; nrc = [1 1];
%   ticksel = 1:nthresholds; nrc = [1 1];
%   
%   % xxx - all this could be combined with rand and computed originally above...
%   ithr_minmergers_all = zeros(ndatasets,nTmins);
%   for i=1:nTmins
%     
%     %split_emd = zeros(ndatasets,nthresholds); merge_emd = zeros(ndatasets,nthresholds);
%     combined_emd = zeros(ndatasets,nthresholds);
%     split_median = zeros(ndatasets,nthresholds); merge_median = zeros(ndatasets,nthresholds);
%     combined_median = zeros(ndatasets,nthresholds);
%     split_auroc = zeros(ndatasets,nthresholds); merge_auroc = zeros(ndatasets,nthresholds);
%     combined_auroc = zeros(ndatasets,nthresholds);
%     split_er = zeros(ndatasets,nthresholds); merge_er = zeros(ndatasets,nthresholds);
%     split_er_CI = zeros(ndatasets,nthresholds,2); merge_er_CI = zeros(ndatasets,nthresholds,2);
%     combined_er = zeros(ndatasets,nthresholds); combined_er_CI = zeros(ndatasets,nthresholds,2); 
%     split_eftpl = zeros(ndatasets,nthresholds); merge_eftpl = zeros(ndatasets,nthresholds);
%     combined_eftpl = zeros(ndatasets,nthresholds); 
%     combined_eftpl_CI = zeros(ndatasets,nthresholds,2); 
%     merge_fracnodes = zeros(ndatasets,nthresholds);
%     merge_fracnodes_CI = zeros(ndatasets,nthresholds,2);
%     split_fracnodes = zeros(ndatasets,nthresholds);
%     split_fracnodes_CI = zeros(ndatasets,nthresholds,2);
%     
%     for j=1:nthresholds
%       % this is specific to how many parameter dimensions there are, currently just Tmin and threshold
%       ind = sub2ind([nthresholds nTmins],j,i);
%       
%       %       [split_efpl_pdf, split_efpl_cdf, merge_efpl_pdf, merge_efpl_cdf, combined_efpl_pdf, combined_efpl_cdf, ...
%       %         smed, mmed, cmed,~,~] = ...
%       %         getSplitMergeDistsAtIndAll(o, ind, ndatasets, p.plx, p.dplx);
%       
%       %       for k = 1:ndatasets
%       %         %         split_emd(k,j) = emd(pl_pdfc(k,:), split_efpl_pdf(k,:));
%       %         %         merge_emd(k,j) = emd(pl_pdfc(k,:), merge_efpl_pdf(k,:));
%       %         %         combined_emd(k,j) = emd(pl_pdfc(k,:), combined_efpl_pdf(k,:));
%       %         split_emd(k,j) = emd(pl_pdf(k,:), split_efpl_pdf(k,:));
%       %         merge_emd(k,j) = emd(pl_pdf(k,:), merge_efpl_pdf(k,:));
%       %         combined_emd(k,j) = emd(pl_pdf(k,:), combined_efpl_pdf(k,:));
%       %       end
%       
%       %       [~,k] = min(abs(split_efpl_cdf - 0.5),[],2); split_median(:,j) = p.plx(k);
%       %       [~,k] = min(abs(merge_efpl_cdf - 0.5),[],2); merge_median(:,j) = p.plx(k);
%       %       [~,k] = min(abs(combined_efpl_cdf - 0.5),[],2); combined_median(:,j) = p.plx(k);
%       
%       [~, split_efpl_cdf, ~, merge_efpl_cdf, ~, combined_efpl_cdf, smed, mmed, cmed, ~,~] = ...
%         getSplitMergeDistsAtIndAll(o, ind, ndatasets, p.plxs, p.dplxs);
%       for k = 1:ndatasets
%         split_auroc(k,j) = sum(split_efpl_cdf(k,1:end-1).*diff(pl_cdf_roc(k,:)));
%         merge_auroc(k,j) = sum(merge_efpl_cdf(k,1:end-1).*diff(pl_cdf_roc(k,:)));
%         combined_auroc(k,j) = sum(combined_efpl_cdf(k,1:end-1).*diff(pl_cdf_roc(k,:)));
%         split_er(k,j) = o{k}.error_rates(ind,1);
%         merge_er(k,j) = o{k}.error_rates(ind,2);
%         split_er_CI(k,j,:) = o{k}.error_rate_CI(ind,1,:);
%         merge_er_CI(k,j,:) = o{k}.error_rate_CI(ind,2,:);
%         % error rates can average because the denominator is number of edges in both cases
%         %combined_er(k,j) = sum(o{k}.error_rates(ind,1:2))/2;
%         combined_er(k,j) = 1 - o{k}.error_rates(ind,3);
%         combined_er_CI(k,j,:) = 1 - o{k}.error_rate_CI(ind,3,:);
%         split_median(k,j) = smed(k)/pl_actual_median(k); 
%         merge_median(k,j) = mmed(k)/pl_actual_median(k); 
%         %combined_median(k,j) = cmed(k)/inpl_actual_median(k);
%         combined_median(k,j) = cmed(k)/pl_actual_median(k);
%         %combined_median(k,j) = cmed(k);
%         
%         merge_fracnodes(k,j) = o{k}.nSMs(ind,2)/nnodes(k);
%         merge_fracnodes_CI(k,j,:) = o{k}.nSMs_CI(ind,2,:);
%         split_fracnodes(k,j) = o{k}.nSMs(ind,1)/nnodes(k);
%         split_fracnodes_CI(k,j,:) = o{k}.nSMs_CI(ind,1,:);
%       end
%       
%       for k = 1:ndatasets
%         split_eftpl(k,j) = squeeze(sum(o{k}.eftpl(ind,:,1)))/1000;
%         merge_eftpl(k,j) = squeeze(sum(o{k}.eftpl(ind,:,2)))/1000;
%         combined_eftpl(k,j) = sum(o{k}.eftpl(ind,:,3))/sum(o{k}.path_length_use);
%         combined_eftpl_CI(k,j,:) = o{k}.eftpl_CI(ind,3,:);
%       end
%       
%     end
%     
%     dist_median = zeros(ndatasets,nthresholds);
%     for k = 1:ndatasets
%       dist_median(k,:) = sqrt((10.^split_median(k,:)-10.^pl_median(k)).^2 + ...
%         (10.^merge_median(k,:)-10.^pl_median(k)).^2);
%     end
%     [~,ithr_minmergers] = min(merge_er,[],2); ithr_minmergers_all(:,i) = ithr_minmergers;
%     %combined_auroc = (repmat(max_auroc',[1 nthresholds])-combined_auroc);
% 
%     figure(baseno+figno-10);
%     subplot(nrc(1),nrc(2),i);
%     plot(combined_median');
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot([ithr_minmergers ithr_minmergers]',repmat([0;2],[1 ndatasets]),'--');
%     set(gca,'plotboxaspectratio',[1 1 1]);
%     xlabel('thresholds'); ylabel('combined median (% HPL)');
%     set(gca,'xtick',ticksel,'xticklabel',thresholds(ticksel)); xlim([0.5 nthresholds+0.5])
%     [m,mi] = max(combined_median,[],2);
%     %set(gca,'ylim',[0.975 1.125]); box off
%     set(gca,'ylim',[0 0.2]); box off
%     title(sprintf('Tmin %d, maxd=%g\n%g %g\nthr=%g %g',Tmins(i),abs(diff(m)),m(1),m(2),...
%       thresholds(mi(1)),thresholds(mi(2))));
%     
%     figure(baseno+figno-9);
%     subplot(nrc(1),nrc(2),i);
%     plot(combined_auroc');
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot([ithr_minmergers ithr_minmergers]',repmat([0.015;0.06],[1 ndatasets]),'--');
%     set(gca,'plotboxaspectratio',[1 1 1]);
%     %set(gca,'ylim',[-0.05 0.55],'xlim',[-0.05 1.05]);
%     xlabel('thresholds'); ylabel('max auroc - combined auroc');
%     set(gca,'xtick',ticksel,'xticklabel',thresholds(ticksel));
%     [m,mi] = max(combined_auroc,[],2);
%     title(sprintf('Tmin %d, maxd=%g\n%g %g\nthr=%g %g',Tmins(i),abs(diff(m)),m(1),m(2),...
%       thresholds(mi(1)),thresholds(mi(2))));
% 
%     figure(baseno+figno-8);
%     subplot(nrc(1),nrc(2),i);
%     plot(combined_eftpl');
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot([ithr_minmergers ithr_minmergers]',repmat([-0.05;0.55],[1 ndatasets]),'--');
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(squeeze(combined_eftpl_CI(:,:,1))','--');
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(squeeze(combined_eftpl_CI(:,:,2))','--');
%     set(gca,'plotboxaspectratio',[1 1 1]);
%     xlabel('thresholds'); ylabel('combined eftpl (%PL)');
%     set(gca,'xtick',ticksel,'xticklabel',thresholds(ticksel)); xlim([0.5 nthresholds+0.5])
%     [m,mi] = max(combined_eftpl,[],2);
%     set(gca,'ylim',[-0.025 0.5]); box off
%     title(sprintf('Tmin %d, maxd=%g\n%g %g\nthr=%g %g',Tmins(i),abs(diff(m)),m(1),m(2),...
%       thresholds(mi(1)),thresholds(mi(2))));
%     
%     figure(baseno+figno-7);
%     subplot(nrc(1),nrc(2),i);
%     plot(combined_er');
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot([ithr_minmergers ithr_minmergers]',repmat([-0.05;0.55],[1 ndatasets]),'--');
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(squeeze(combined_er_CI(:,:,1))','--');
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(squeeze(combined_er_CI(:,:,2))','--');
%     set(gca,'plotboxaspectratio',[1 1 1]);
%     xlabel('thresholds'); ylabel('combined error rate');
%     set(gca,'xtick',ticksel,'xticklabel',thresholds(ticksel)); xlim([0.5 nthresholds+0.5])
%     [m,mi] = max(combined_er,[],2);
%     set(gca,'ylim',[-0.05 0.55]); box off
%     title(sprintf('Tmin %d, maxd=%g\n%g %g\nthr=%g %g',Tmins(i),abs(diff(m)),m(1),m(2),...
%       thresholds(mi(1)),thresholds(mi(2))));
% 
%     figure(baseno+figno-2);
%     subplot(nrc(1),nrc(2),i);
%     plot(split_er(1,:),merge_er(1,:));
%     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(split_er(2,:),merge_er(2,:));
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(squeeze(split_er_CI(1,:,1)),squeeze(merge_er_CI(1,:,1)),'--');
%     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(squeeze(split_er_CI(2,:,1)),squeeze(merge_er_CI(2,:,1)),'--');
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(squeeze(split_er_CI(1,:,2)),squeeze(merge_er_CI(1,:,2)),'--');
%     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(squeeze(split_er_CI(2,:,2)),squeeze(merge_er_CI(2,:,2)),'--');
%     set(gca,'plotboxaspectratio',[1 1 1]);
%     set(gca,'ylim',[-0.05 1.05],'xlim',[-0.05 1.05]);
%     xlabel('split error rate'); ylabel('merge error rate');
%     title(sprintf('Tmin %d',Tmins(i)));
% 
% %     figure(baseno+figno-2);
% %     subplot(nrc(1),nrc(2),i);
% %     plot(1-split_er(1,:),1-merge_er(1,:));
% %     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% %     plot(1-split_er(2,:),1-merge_er(2,:));
% %     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% %     plot(1-squeeze(split_er_CI(1,:,1)),1-squeeze(merge_er_CI(1,:,1)),'--');
% %     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% %     plot(1-squeeze(split_er_CI(2,:,1)),1-squeeze(merge_er_CI(2,:,1)),'--');
% %     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% %     plot(1-squeeze(split_er_CI(1,:,2)),1-squeeze(merge_er_CI(1,:,2)),'--');
% %     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% %     plot(1-squeeze(split_er_CI(2,:,2)),1-squeeze(merge_er_CI(2,:,2)),'--');
% %     set(gca,'plotboxaspectratio',[1 1 1]);
% %     set(gca,'ylim',[-0.05 1.05],'xlim',[-0.05 1.05]);
% %     xlabel('split error free rate'); ylabel('merge error free rate');
% %     title(sprintf('Tmin %d',Tmins(i)));
% 
% %     figure(baseno+figno-2);
% %     %subplot(nrc(1),nrc(2),i);
% %     subplot(1,2,1);
% %     plot(merge_er(1,:),merge_fracnodes(1,:),'-x');
% %     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% %     plot(merge_er(2,:),merge_fracnodes(2,:),'-x');
% %     set(gca,'plotboxaspectratio',[1 1 1]);
% %     set(gca,'ylim',[-0.05 1.05],'xlim',[-0.05 1.05]);
% %     xlabel('merge error edges'); ylabel('merge error rate nodes');
% %     subplot(1,2,2);
% %     plot(split_er(1,:),split_fracnodes(1,:),'-x');
% %     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% %     plot(split_er(2,:),split_fracnodes(2,:),'-x');
% %     set(gca,'plotboxaspectratio',[1 1 1]);
% %     set(gca,'ylim',[-0.05 1.05],'xlim',[-0.05 1.05]);
% %     xlabel('split error edges'); ylabel('split error rate nodes');
% 
%     
%     figure(baseno+figno-4);
%     subplot(nrc(1),nrc(2),i);
%     sumSM = split_er+merge_fracnodes;
%     [m,mi] = min(sumSM,[],2);
%     minSM = [[split_er(1,mi(1)); merge_fracnodes(1,mi(1))] ...
%       [split_er(2,mi(2)); merge_fracnodes(2,mi(2))]];
%     plot(split_er(1,:),merge_fracnodes(1,:));
%     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(split_er(2,:),merge_fracnodes(2,:));
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(minSM(1,1),minSM(2,1),'x');
%     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(minSM(1,2),minSM(2,2),'x');
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(squeeze(split_er_CI(1,:,1)),squeeze(merge_fracnodes_CI(1,:,1)),'--');
%     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(squeeze(split_er_CI(2,:,1)),squeeze(merge_fracnodes_CI(2,:,1)),'--');
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(squeeze(split_er_CI(1,:,2)),squeeze(merge_fracnodes_CI(1,:,2)),'--');
%     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(squeeze(split_er_CI(2,:,2)),squeeze(merge_fracnodes_CI(2,:,2)),'--');
%     set(gca,'plotboxaspectratio',[1 1 1]);
%     set(gca,'ylim',[-0.05 1.05],'xlim',[-0.05 1.05]);
%     xlabel('split edges'); ylabel('merged nodes');
%     title(sprintf('Tmin %d, maxd=%g\n%g=%g+%g %g=%g+%g\nthr=%g %g',Tmins(i),abs(diff(m)),m(1),...
%       minSM(1,1),minSM(2,1),m(2),minSM(1,2),minSM(2,2),thresholds(mi(1)),thresholds(mi(2))));
%   
%   
%     
% %     figure(baseno+figno-7);
% %     %subplot(nrc(1),nrc(2),i);
% %     subplot(2,3,1);
% %     plot(split_eftpl(1,:),merge_eftpl(1,:));
% %     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% %     plot(split_eftpl(2,:),merge_eftpl(2,:));
% %     set(gca,'plotboxaspectratio',[1 1 1]);
% %     %set(gca,'ylim',[-0.05 0.55],'xlim',[-0.05 1.05]);
% %     xlabel('split eftpl (um)'); ylabel('merge eftpl (um)');
% %     title(sprintf('Tmin %d',Tmins(i)));
% %     subplot(2,3,4);
% %     plot((split_eftpl+merge_eftpl)');
% %     set(gca,'plotboxaspectratio',[1 1 1]);
% %     %set(gca,'ylim',[-0.05 0.55],'xlim',[-0.05 1.05]);
% %     xlabel('thresholds'); ylabel('combined eftpl (um)');
% %     set(gca,'xtick',ticksel,'xticklabel',thresholds(ticksel));
% %     subplot(2,3,2);
% %     plot(split_er(1,:),merge_er(1,:));
% %     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% %     plot(split_er(2,:),merge_er(2,:));
% %     set(gca,'plotboxaspectratio',[1 1 1]);
% %     %set(gca,'ylim',[-0.05 0.55],'xlim',[-0.05 1.05]);
% %     xlabel('split error rate'); ylabel('merge error rate');
% %     subplot(2,3,5);
% %     plot(combined_er');
% %     set(gca,'plotboxaspectratio',[1 1 1]);
% %     %set(gca,'ylim',[-0.05 0.55],'xlim',[-0.05 1.05]);
% %     xlabel('thresholds'); ylabel('combined error rate');
% %     set(gca,'xtick',ticksel,'xticklabel',thresholds(ticksel));
% %     %     subplot(2,3,3);
% %     %     plot(split_auroc(1,:),merge_auroc(1,:));
% %     %     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% %     %     plot(split_auroc(2,:),merge_auroc(2,:));
% %     %     set(gca,'plotboxaspectratio',[1 1 1]);
% %     %     %set(gca,'ylim',[-0.05 0.55],'xlim',[-0.05 1.05]);
% %     %     xlabel('split auroc'); ylabel('merge auroc');
% %     %     subplot(2,3,6);
% %     %     plot((repmat(max_auroc',[1 nthresholds])-combined_auroc)');
% %     %     set(gca,'plotboxaspectratio',[1 1 1]);
% %     %     %set(gca,'ylim',[-0.05 0.55],'xlim',[-0.05 1.05]);
% %     %     xlabel('thresholds'); ylabel('max auroc - combined auroc');
% %     %     set(gca,'xtick',ticksel,'xticklabel',thresholds(ticksel));
% %     subplot(2,3,6);
% %     plot(combined_eftpl');
% %     set(gca,'plotboxaspectratio',[1 1 1]);
% %     %set(gca,'ylim',[-0.05 0.55],'xlim',[-0.05 1.05]);
% %     xlabel('thresholds'); ylabel('intersected eftpl (um)');
% %     set(gca,'xtick',ticksel,'xticklabel',thresholds(ticksel));
%     
%     figure(baseno+figno-6);
%     subplot(nrc(1),nrc(2),i);
%     plot(split_auroc(1,:),merge_auroc(1,:));
%     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(split_auroc(2,:),merge_auroc(2,:));
%     set(gca,'plotboxaspectratio',[1 1 1]);
%     %set(gca,'ylim',[-0.05 0.55],'xlim',[-0.05 1.05]);
%     xlabel('split auroc'); ylabel('merge auroc');
%     title(sprintf('Tmin %d',Tmins(i)));
%     
%     figure(baseno+figno-5);
%     subplot(nrc(1),nrc(2),i);
%     plot(log(split_er(1,:)),split_auroc(1,:),'x');
%     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(log(split_er(2,:)),split_auroc(2,:),'x');
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(log(merge_er(1,:)),merge_auroc(1,:),'o');
%     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(log(merge_er(2,:)),merge_auroc(2,:),'o');
%     set(gca,'plotboxaspectratio',[1 1 1]);
%     %set(gca,'ylim',[-0.05 0.55],'xlim',[-0.05 1.05]);
%     xlabel('error rates'); ylabel('auroc');
%     title(sprintf('Tmin %d',Tmins(i)));
%     
% %     figure(baseno+figno-4);
% %     subplot(nrc(1),nrc(2),i);
% %     plot(split_er(1,:),split_auroc(1,:),'x');
% %     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% %     plot(split_er(2,:),split_auroc(2,:),'x');
% %     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% %     plot(merge_er(1,:),merge_auroc(1,:),'o');
% %     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
% %     plot(merge_er(2,:),merge_auroc(2,:),'o');
% %     set(gca,'plotboxaspectratio',[1 1 1]);
% %     %set(gca,'ylim',[-0.05 0.55],'xlim',[-0.05 1.05]);
% %     xlabel('log error rates'); ylabel('auroc');
% %     title(sprintf('Tmin %d',Tmins(i)));
%     
%     figure(baseno+figno-3);
%     subplot(nrc(1),nrc(2),i);
%     plot(10.^split_median(1,:),10.^merge_median(1,:));
%     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(10.^split_median(2,:),10.^merge_median(2,:));
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(10^pl_median(1),10^pl_median(1),'x'); plot(10^pl_median(2),10^pl_median(2),'x');
%     xlabel('split median (um)'); ylabel('merge median (um)');
%     d = max(pl_median); set(gca,'xlim',[-5 10^d+5]); set(gca,'ylim',[-5 10^d+5]);
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     [d1,k1] = min(dist_median(1,:)); plot(10^split_median(1,k1),10^merge_median(1,k1),'o');
%     [d2,k2] = min(dist_median(2,:)); plot(10^split_median(2,k2),10^merge_median(2,k2),'o');
%     set(gca,'plotboxaspectratio',[1 1 1]);
%     title(sprintf('Tmin %d min dist:\nnone %.2f (%g) huge %.2f (%g)',Tmins(i),d1,thresholds(k1),d2,thresholds(k2)));
%     
%     figure(baseno+figno-1);
%     subplot(nrc(1),nrc(2),i);
%     plot(combined_emd(1,:));
%     hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot(combined_emd(2,:),'--');
%     xlabel('thresold'); ylabel('combined EMD');
%     %set(gca,'xlim',[-5 dset_max_emd]); set(gca,'ylim',[-5 dset_max_emd]);
%     %set(gca,'xtick',1:nthresholds,'xticklabel',thresholds);
%     set(gca,'plotboxaspectratio',[1 1 1]);
%     title(sprintf('Tmin %d',Tmins(i)));
%   end
%   figure(baseno+figno-3);
%   axes('Position',[0 0 1 1],'Visible','off');
%   text(0.3, 0.975, sprintf('none(bl) huge(br)'))
%   figure(baseno+figno-4);
%   axes('Position',[0 0 1 1],'Visible','off');
%   text(0.3, 0.975, sprintf('none(bl) huge(br) split (x) merge (o)'))
%   figure(baseno+figno-5);
%   axes('Position',[0 0 1 1],'Visible','off');
%   text(0.3, 0.975, sprintf('none(bl) huge(br) split (x) merge (o)'))
%   
% end
% 
% 
% figure(baseno+figno); figno = figno+1; clf
% figure(baseno+figno); figno = figno+1; clf
% figure(baseno+figno); figno = figno+1; clf
% figure(baseno+figno); figno = figno+1; clf
% figure(baseno+figno); figno = figno+1; clf
% figure(baseno+figno); figno = figno+1; clf
% %ticksel = 1:3:nthresholds; nrc = [2 3];
% ticksel = 1:nthresholds; nrc = [1 1];
% for i=1:nTmins
%   j = 1;
%   % this is specific to how many parameter dimensions there are, currently just Tmin and threshold
%   ind = sub2ind([nthresholds nTmins],j,i);
%   
%   figure(baseno+figno-1);
%   subplot(nrc(1),nrc(2),i);
%   plot(are(ind:ind+nthresholds-1,:));
%   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   plot(squeeze(are_CI(ind:ind+nthresholds-1,1,:)),'--');
%   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   plot(squeeze(are_CI(ind:ind+nthresholds-1,2,:)),'--');
%   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   %   plot(ri,'--');
%   %   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   %   plot(squeeze(ri_CI(:,1,:)),'--');
%   %   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   %   plot(squeeze(ri_CI(:,2,:)),'--');
%   %   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   %   plot(ari,'-.');
%   %   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   %   plot(squeeze(ari_CI(:,1,:)),'-.');
%   %   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   %   plot(squeeze(ari_CI(:,2,:)),'-.');
% 
%   if exist('ithr_minmergers_all','var')
%     hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%     plot([ithr_minmergers_all(:,i) ithr_minmergers_all(:,i)]',repmat([0.5;1.05],[1 ndatasets]),'--');
%   end
%   set(gca,'plotboxaspectratio',[1 1 1]);
%   [m,mi] = min(are(ind:ind+nthresholds-1,:),[],1);
%   title(sprintf('Tmin %d, maxd=%g\n%g %g\nthr=%g %g',Tmins(i),abs(diff(m)),m(1),m(2),...
%     thresholds(mi(1)),thresholds(mi(2))));
%   set(gca,'xtick',ticksel,'xticklabel',thresholds(ticksel)); xlim([0.5 nthresholds+0.5])
%   set(gca,'ylim',[0.6 1.025]); box off;
%   xlabel('threshold'); ylabel('ARE, [0,1], 1-Fscore');
%   
%   figure(baseno+figno-2);
%   subplot(nrc(1),nrc(2),i);
%   sumSM = squeeze(sum(split_mergers(ind:ind+nthresholds-1,:,:),2))./repmat(nnodes,[nthresholds 1]);
%   [m,mi] = min(sumSM,[],1); mi = mi-1+ind;
%   minSM = [[squeeze(split_mergers(mi(1),1,1))./nnodes(1); squeeze(split_mergers(mi(1),2,1))./nnodes(1)] ...
%     [squeeze(split_mergers(mi(2),1,2))./nnodes(2); squeeze(split_mergers(mi(2),2,2))./nnodes(2)]];
%   plot(squeeze(split_mergers(ind:ind+nthresholds-1,1,:))./repmat(nnodes,[nthresholds 1]),...
%     squeeze(split_mergers(ind:ind+nthresholds-1,2,:))./repmat(nnodes,[nthresholds 1]));
%   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   plot(minSM(1,1),minSM(2,1),'x');
%   hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   plot(minSM(1,2),minSM(2,2),'x');
%   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   plot(squeeze(split_mergers_CI(ind:ind+nthresholds-1,1,1,:)),...
%     squeeze(split_mergers_CI(ind:ind+nthresholds-1,2,1,:)),'--');
%   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   plot(squeeze(split_mergers_CI(ind:ind+nthresholds-1,1,2,:)),...
%     squeeze(split_mergers_CI(ind:ind+nthresholds-1,2,2,:)),'--');
%   title(sprintf('Tmin=%d',Tmins(i)));
%   set(gca,'plotboxaspectratio',[1 1 1]);
%   set(gca,'ylim',[-0.05 1.05],'xlim',[-0.05 1.05]);
%   xlabel('splits'); ylabel('mergers');
%   title(sprintf('Tmin %d, maxd=%g\n%g=%g+%g %g=%g+%g\nthr=%g %g',Tmins(i),abs(diff(m)),m(1),...
%     minSM(1,1),minSM(2,1),m(2),minSM(1,2),minSM(2,2),thresholds(mi(1)),thresholds(mi(2))));
%   
%   figure(baseno+figno-3);
%   subplot(nrc(1),nrc(2),i);
%   plot(nBGnodes(ind:ind+nthresholds-1,:));
%   title(sprintf('Tmin=%d',Tmins(i)));
%   %set(gca,'ylim',[0.5 1.05]);
%   xlabel('threshold'); ylabel('nBGnodes');
%   
%   figure(baseno+figno-4);
%   subplot(nrc(1),nrc(2),i);
%   plot(nECSnodes(ind:ind+nthresholds-1,:));
%   title(sprintf('Tmin=%d',Tmins(i)));
%   %set(gca,'ylim',[0.5 1.05]);
%   xlabel('threshold'); ylabel('nECSnodes');
%   
%   figure(baseno+figno-5);
%   subplot(nrc(1),nrc(2),i);
%   plot(squeeze(are_precrec(ind:ind+nthresholds-1,1,:)),squeeze(are_precrec(ind:ind+nthresholds-1,2,:)));
%   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   plot(squeeze(are_precrec_CI(ind:ind+nthresholds-1,1,1,:)),squeeze(are_precrec_CI(ind:ind+nthresholds-1,2,1,:)),'--');
%   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   plot(squeeze(are_precrec_CI(ind:ind+nthresholds-1,1,2,:)),squeeze(are_precrec_CI(ind:ind+nthresholds-1,2,2,:)),'--');
%   title(sprintf('Tmin=%d',Tmins(i)));
%   %set(gca,'ylim',[0.5 1.05]);
%   xlabel('recall'); ylabel('precision');
%   
%   figure(baseno+figno-6);
%   subplot(nrc(1),nrc(2),i);
%   sumSM = squeeze(sum(split_mergers_segEM(ind:ind+nthresholds-1,:,:)./...
%     repmat(permute([nnodes; nskels],[3 1 2]),[nthresholds 1 1]),2));
%   [m,mi] = min(sumSM,[],1); mi = mi-1+ind;
%   minSM = [[squeeze(split_mergers_segEM(mi(1),1,1))./nnodes(1); squeeze(split_mergers_segEM(mi(1),2,1))./nskels(1)] ...
%     [squeeze(split_mergers_segEM(mi(2),1,2))./nnodes(2); squeeze(split_mergers_segEM(mi(2),2,2))./nskels(2)]];
%   plot(squeeze(split_mergers_segEM(ind:ind+nthresholds-1,1,:))./repmat(nnodes,[nthresholds 1]),...
%     squeeze(split_mergers_segEM(ind:ind+nthresholds-1,2,:))./repmat(nskels,[nthresholds 1]));
%   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   plot(minSM(1,1),minSM(2,1),'x');
%   hold on; %if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   plot(minSM(1,2),minSM(2,2),'x');
%   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   plot(squeeze(split_mergers_segEM_CI(ind:ind+nthresholds-1,1,1,:)),...
%     squeeze(split_mergers_segEM_CI(ind:ind+nthresholds-1,2,1,:)),'--');
%   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', 1); end
%   plot(squeeze(split_mergers_segEM_CI(ind:ind+nthresholds-1,1,2,:)),...
%     squeeze(split_mergers_segEM_CI(ind:ind+nthresholds-1,2,2,:)),'--');
%   title(sprintf('Tmin=%d',Tmins(i)));
%   set(gca,'plotboxaspectratio',[1 1 1]);
%   set(gca,'ylim',[-0.05 1.05],'xlim',[-0.05 1.05]);
%   xlabel('segem splits'); ylabel('segem mergers');
%   title(sprintf('Tmin %d, maxd=%g\n%g=%g+%g %g=%g+%g\nthr=%g %g',Tmins(i),abs(diff(m)),m(1),...
%     minSM(1,1),minSM(2,1),m(2),minSM(1,2),minSM(2,2),thresholds(mi(1)),thresholds(mi(2))));
%   
% end
% figure(baseno+figno-1); axes('Position',[0 0 1 1],'Visible','off');
% text(0.3, 0.975, sprintf('none(bl) huge(br) are (-) ri(--) ari(-.)'))
% figure(baseno+figno-2);
% axes('Position',[0 0 1 1],'Visible','off');
% text(0.3, 0.975, sprintf('none(bl) n=%d huge(br) n=%d',nnodes))
% figure(baseno+figno-3);
% axes('Position',[0 0 1 1],'Visible','off');
% text(0.3, 0.975, sprintf('none(bl) huge(br)'))
% figure(baseno+figno-4);
% axes('Position',[0 0 1 1],'Visible','off');
% text(0.3, 0.975, sprintf('none(bl) huge(br)'))
% figure(baseno+figno-5);
% axes('Position',[0 0 1 1],'Visible','off');
% text(0.3, 0.975, sprintf('none(bl) huge(br)'))
% 
% 
% 
% figure(baseno+figno); figno = figno+1; clf
% 
% trs = [0.999 0.99]; tms = [256 256]; 
% cnt = zeros(1,ndatasets); med = zeros(1,ndatasets); tot = zeros(1,ndatasets);
% for k = 1:ndatasets
%   i = find(abs(thresholds - trs(k)) < 1e-6); assert( length(i)==1 );
%   j = find(Tmins==tms(k),1); assert( length(j)==1 );
%   ind = sub2ind([nthresholds nTmins],i,j);
%   [~, ~, ~, ~, cpdf, ccdf, ~,~,cmed, ccnt, ctot] = ...
%     getSplitMergeDistsAtIndAll(o, ind, ndatasets, p.plx, p.dplx);
% 
%   subplot(1,2,1);
%   hold on; plot(p.plx,cpdf(k,:));
%   xlabel('path length (log10 um)');
%   ylabel('pdf');
%   set(gca,'xlim',[p.plx(1) p.plx(end)]);
%   set(gca,'plotboxaspectratio',[1 1 1]);
%   subplot(1,2,2);
%   hold on; plot(p.plx,ccdf(k,:));
%   hold on; if useColorOrder, set(gca, 'ColorOrderIndex', k); end
%   plot(log10(cmed(k)),0.5,'x');
%   %[~,p] = kstest2(log10(path_lengths{1}),log10(path_lengths{2}));
%   %[~,phn] = kstest2(log10(half_node_path_lengths{1}),log10(half_node_path_lengths{2}));
%   xlabel('path length (log10 um)');
%   ylabel('cdf');
%   set(gca,'xlim',[p.plx(1) p.plx(end)]);
%   set(gca,'ylim',[-0.05 1.05]);
%   set(gca,'plotboxaspectratio',[1 1 1]);
%   cnt(k) = ccnt(k); med(k) = cmed(k); tot(k) = ctot(k)*1000/sum(o{k}.path_length_use);
% end
% title(sprintf('none: %d segments, %.4f of tpl, median %.4f\nhuge: %d segments, %.4f of tpl, median %.4f',...
%   cnt(1),tot(1),med(1),cnt(2),tot(2),med(2)));
% 
% 
% 
% function [split_efpl_pdf, split_efpl_cdf, merge_efpl_pdf, merge_efpl_cdf, combined_efpl_pdf, combined_efpl_cdf,...
%   split_median, merge_median, combined_median, combined_cnt, combined_tot] = ...
%   getSplitMergeDistsAtIndAll(o, ind, ndatasets, plx, dplx)
% 
% nplx = length(plx);
% split_efpl_hist = zeros(ndatasets,nplx); merge_efpl_hist = zeros(ndatasets,nplx);
% combined_efpl_hist = zeros(ndatasets,nplx);
% split_median = zeros(ndatasets,1); merge_median = zeros(ndatasets,1); combined_median = zeros(ndatasets,1);
% combined_cnt = zeros(ndatasets,1); combined_tot = zeros(ndatasets,1);
% for j = 1:ndatasets
%   % remove really small internode distances that are just from dropping nodes right next to each other
%   minPL = 10^(plx(1)-dplx(1)/2);
%   tmp = o{j}.efpls{ind,1}/1000; tmp = tmp(tmp >= minPL);
%   split_efpl_hist(j,:) = hist(log10(tmp), plx); split_median(j) = median(tmp);
%   tmp = o{j}.efpls{ind,2}/1000; tmp = tmp(tmp >= minPL);
%   merge_efpl_hist(j,:) = hist(log10(tmp), plx); merge_median(j) = median(tmp);
%   
%   tmp = o{j}.efpls{ind,3}/1000; tmp = tmp(tmp >= minPL);
%   combined_efpl_hist(j,:) = hist(log10(tmp), plx); 
%   combined_median(j) = median(tmp); combined_cnt(j) = length(tmp); combined_tot(j) = sum(tmp);
% end
% %combined_efpl_hist = split_efpl_hist + merge_efpl_hist;
% %split_efpl_hist(rand_pdf_sel) = split_efpl_hist(rand_pdf_sel) ./ rand_pdf(rand_pdf_sel);
% %merge_efpl_hist(rand_pdf_sel) = merge_efpl_hist(rand_pdf_sel) ./ rand_pdf(rand_pdf_sel);
% %combined_efpl_hist(rand_pdf_sel) = combined_efpl_hist(rand_pdf_sel) ./ rand_pdf(rand_pdf_sel);
% split_efpl_pdf = split_efpl_hist./repmat(sum(split_efpl_hist,2),[1 nplx]);
% merge_efpl_pdf = merge_efpl_hist./repmat(sum(merge_efpl_hist,2),[1 nplx]);
% combined_efpl_pdf = combined_efpl_hist./repmat(sum(combined_efpl_hist,2),[1 nplx]);
% split_efpl_cdf = cumsum(split_efpl_pdf,2); merge_efpl_cdf = cumsum(merge_efpl_pdf,2);
% combined_efpl_cdf = cumsum(combined_efpl_pdf,2);
%     
% 
