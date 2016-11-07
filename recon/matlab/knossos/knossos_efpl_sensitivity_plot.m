
%   are(k,ind) = o{k}.are(ind); are_CI(k,ind,:) =  o{k}.are_CI(ind,:);
%   % just convert precision recall to 1- here (so min is better)
%   are_precrec(k,ind,:) = 1-o{k}.are_precrec(ind,:); are_precrec_CI(k,ind,:,:) = 1-o{k}.are_precrec_CI(ind,:,:);
%   combined_eftpl(k,ind) = sum(o{k}.eftpl(ind,:,3),2)/sum(o{k}.path_length_use);

%   save_vars = {
%     'ndatasets' 'params' 'path_lengths' 'internode_lengths' 'nskels' 'split_mergers' 'split_mergers_segEM'
%     'nBGnodes' 'nECSnodes' 'nnodes' 'nnodes_skel' 'names' 'split_mergers_CI' 'split_mergers_segEM_CI' 'nparams'
%     'split_er' 'split_er_CI' 'merge_fracnodes' 'split_fracnodes' 'are' 'are_CI' 'combined_eftpl' 'norm_params'
%     'nlabels'
%   };

load('/home/watkinspv/Data/efpl/efpl_sensitivity_big_meta.mat');
%load('~/Downloads/efpl_sensitivity_big_meta.mat');

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


baseno = 1; figno = 0;

dx = 0.05; sms = [0:dx:(1-dx)] + dx/2;
[gx, gy] = ndgrid(sms, sms);

nruns = 11;
assert( po.nparams == length(split_percs) );
nslices = length(merge_percs);

figure(baseno+figno); figno = figno+1; clf
pointsize=16;
dolog = false;
mergemax = 0.25; splitmax = 0.85;

X = reshape(po.combined_eftpl, [nslices, nruns, po.nparams]);
X_u = reshape(nanmean(X,2), [nslices, po.nparams]);
X_s = reshape(nanstd(X,[],2), [nslices, po.nparams]);
if dolog, X_u = log10(X_u); end

subplot(2,2,1);
scatter(split(:),merge(:),pointsize,X_u(:));colorbar
set(gca,'ylim',[-0.05 mergemax],'xlim',[-0.05 splitmax]);
set(gca,'plotboxaspectratio',[1 1 1]);
xlabel('% splits'); ylabel('% merges'); title('tefpl mean');

subplot(2,2,2);
scatter(split(:),merge(:),pointsize,X_s(:));colorbar
set(gca,'ylim',[-0.05 mergemax],'xlim',[-0.05 splitmax]);
set(gca,'plotboxaspectratio',[1 1 1]);
xlabel('% splits'); ylabel('% merges'); title('tefpl std');

X = reshape(po.are, [nslices, nruns, po.nparams]);
X_u = reshape(nanmean(X,2), [nslices, po.nparams]);
X_s = reshape(nanstd(X,[],2), [nslices, po.nparams]);
if dolog, X_u = log10(X_u); end

subplot(2,2,3);
scatter(split(:),merge(:),pointsize,X_u(:));colorbar
set(gca,'ylim',[-0.05 mergemax],'xlim',[-0.05 splitmax]);
set(gca,'plotboxaspectratio',[1 1 1]);
xlabel('% splits'); ylabel('% merges'); title('are mean');

subplot(2,2,4);
scatter(split(:),merge(:),pointsize,X_s(:));colorbar
set(gca,'ylim',[-0.05 mergemax],'xlim',[-0.05 splitmax]);
set(gca,'plotboxaspectratio',[1 1 1]);
xlabel('% splits'); ylabel('% merges'); title('are std');




figure(baseno+figno); figno = figno+1; clf
dolog = true;

X = reshape(po.combined_eftpl, [nslices, nruns, po.nparams]);
X_u = reshape(nanmean(X,2), [nslices, po.nparams]);
X_s = reshape(nanstd(X,[],2), [nslices, po.nparams]);
if dolog, X_u = log10(X_u); end

subplot(2,2,1);
imagesc(split_percs,merge_percs,X_u); colorbar
set(gca,'plotboxaspectratio',[1 1 1],'ydir','normal');
xlabel('% splits'); ylabel('% merges'); title('tefpl mean');

subplot(2,2,2);
imagesc(split_percs,merge_percs,X_s); colorbar
set(gca,'plotboxaspectratio',[1 1 1],'ydir','normal');
xlabel('% splits'); ylabel('% merges'); title('tefpl std');

X = reshape(po.are, [nslices, nruns, po.nparams]);
X_u = reshape(nanmean(X,2), [nslices, po.nparams]);
X_s = reshape(nanstd(X,[],2), [nslices, po.nparams]);
if dolog, X_u = log10(X_u); end

subplot(2,2,3);
imagesc(split_percs,merge_percs,X_u); colorbar
set(gca,'plotboxaspectratio',[1 1 1],'ydir','normal');
xlabel('% splits'); ylabel('% merges'); title('are mean');

subplot(2,2,4);
imagesc(split_percs,merge_percs,X_s); colorbar
set(gca,'plotboxaspectratio',[1 1 1],'ydir','normal');
xlabel('% splits'); ylabel('% merges'); title('are std');
