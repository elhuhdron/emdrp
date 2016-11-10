
% load_files = {
%   '/home/watkinspv/Data/efpl/efpl_paper.mat'
%   '/home/watkinspv/Data/efpl/efpl_paper.mat'
%   '/home/watkinspv/Data/efpl/efpl_paper_and_xyz_crop.mat'
%   '/home/watkinspv/Data/efpl/efpl_paper_and_xyz_crop.mat'
% };
% load_indices = [1 2 1 2]; 

% load_files = {
%   '/home/watkinspv/Data/efpl/efpl_paper_interp_norandbg.mat'
%   '/home/watkinspv/Data/efpl/efpl_paper_interp_norandbg.mat'
%   '/home/watkinspv/Data/efpl/efpl_agglo_rf_75iter2p_reduced_interp_norandbg.mat'
%   '/home/watkinspv/Data/efpl/efpl_agglo_rf_75iter2p_reduced_interp_norandbg.mat'
% };
% load_indices = [1 2 1 2]; 

% load_files = {
%   '/home/watkinspv/Data/efpl/efpl_paper_interp_norandbg.mat'
%   %'/home/watkinspv/Data/efpl/efpl_huge_xyzonly_offset_interp_norandbg.mat'
%   '/home/watkinspv/Data/efpl/efpl_huge_xyzonly_offset_interp_norandbg.mat'
%   '/home/watkinspv/Data/efpl/efpl_huge_xyzonly_offset_interp_norandbg.mat'
% };
% load_indices = [2 2 3]; 

% % pre-new arch paper-like runs
% load_files = {
%   %'/home/watkinspv/Data/efpl/efpl_huge_interp_wsnew.mat' % ind 1
%   '/home/watkinspv/Data/efpl/efpl_huge_interp_fergus_wsnew.mat'
%   '/home/watkinspv/Data/efpl/efpl_huge_interp_kaiming2.mat'
%   %'/home/watkinspv/Data/efpl/efpl_huge_interp_vgg3.mat'
%   %'/home/watkinspv/Data/efpl/efpl_huge_interp_vgg3_2.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_vgg4_pferg.mat'
%   %'/home/watkinspv/Data/efpl/efpl_interp_vgg4_pferg.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_p2ferg.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_vgg5.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_vgg5.mat'
% };
% %load_indices = [2 2 1 1 1 2 1]; 
% load_indices = [2 2 1 1 1 2]; 

% % compare stitching
% load_files = {
%   '/home/watkinspv/Data/efpl/efpl_huge_interp_fergus_wsnew.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_nmferg_stitch.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_nmferg_stitch.mat'
% };
% load_indices = [2 1 2]; 

% load_files = {
%   '/home/watkinspv/Data/efpl/efpl_paper_interp_norandbg.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_xyz_norandbg.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_n3f16_oldthr.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_mnbf32.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_vgg3b64.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_mbf16_ds2.mat'  
% %   '/home/watkinspv/Data/efpl/efpl_paper_interp_norandbg.mat'
% %   '/home/watkinspv/Data/efpl/efpl_interp_xyz_norandbg.mat'
% %   '/home/watkinspv/Data/efpl/efpl_interp_n3f16_oldthr.mat'
% %   '/home/watkinspv/Data/efpl/efpl_interp_mnbf32.mat'
% %   '/home/watkinspv/Data/efpl/efpl_interp_vgg3b64.mat'
% %   '/home/watkinspv/Data/efpl/efpl_interp_mbf16_ds2.mat'  
% };
% load_indices = [2 2 1 1 1 1];
% %load_indices = [1 1 2 2 2 2];

% load_files = {
%   '/home/watkinspv/Data/efpl/efpl_paper_interp_norandbg.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_xyz_norandbg.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_n3f16.mat'
%   '/home/watkinspv/Data/efpl/efpl_paper_vgg4_pferg.mat'
%   '/home/watkinspv/Data/efpl/efpl_huge_interp_vgg3.mat'
%   '/home/watkinspv/Data/efpl/efpl_huge_interp_vgg3_2.mat'
%   '/home/watkinspv/Data/efpl/efpl_huge_interp_fergus_wsnew.mat'
% };
% load_indices = [2 2 1 1 1 1 1];

% % plot for kevin's talk
% load_files = {
%   '/home/watkinspv/Data/efpl/efpl_interp_kbtalk20160915.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_kbtalk20160915_new.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_kbtalk20160915.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_kbtalk20160915.mat'
% };
% load_indices = [1 1 3 4]; 

% % plot for kevin's talk with (agglo) stitching
% load_files = {
%   '/home/watkinspv/Data/efpl/efpl_interp_kbtalk20160915.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_kbtalk20160915_new.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_mbf32_agglo_stitch.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_mbf32_agglo_stitch.mat'
%   '/home/watkinspv/Data/efpl/efpl_interp_mbf32_stitch_ws.mat'  
%   '/home/watkinspv/Data/efpl/efpl_interp_mbf32_stitch_agglo.mat'
% };
% load_indices = [1 1 2 4 1 1]; 

% % compare increasing amounts of training data
% load_files = {
%   '/home/watkinspv/Data/efpl/efpl_paper_mbf32_huge_comp_train.mat'
%   '/home/watkinspv/Data/efpl/efpl_paper_mbf32_huge_comp_train.mat'
%   '/home/watkinspv/Data/efpl/efpl_paper_mbf32_huge_comp_train.mat'
%   '/home/watkinspv/Data/efpl/efpl_paper_mbf32_comp_train45.mat'
%   '/home/watkinspv/Data/efpl/efpl_paper_mbf32_comp_train45.mat'
%   '/home/watkinspv/Data/efpl/efpl_paper_mbf32_huge_comp_train.mat'
%   '/home/watkinspv/Data/efpl/efpl_paper_mbf32_none_comp_train.mat'
%   '/home/watkinspv/Data/efpl/efpl_paper_mbf32_none_comp_train.mat'
%   '/home/watkinspv/Data/efpl/efpl_paper_mbf32_none_comp_train.mat'
%   '/home/watkinspv/Data/efpl/efpl_paper_mbf32_comp_train45.mat'
%   '/home/watkinspv/Data/efpl/efpl_paper_mbf32_comp_train45.mat'
%   '/home/watkinspv/Data/efpl/efpl_paper_mbf32_none_comp_train.mat'
% };
% %load_indices = [2 3 4 1 2 1]; % huge
% %load_indices = [2 3 4 3 4 1]; % none
% load_indices = [2 3 4 1 2 1 2 3 4 3 4 1]; % all

% metric sensitivity
load_files = repmat({'/home/watkinspv/Data/efpl/efpl_huge_sensitivity_crop_big_sample0p2.mat'},[1 121]);
load_indices = 1:121; 

% load_files = {
%   '~/Documents/Data/em/efpl/efpl_nointerp_k0725.mat'
%   '~/Documents/Data/em/efpl/efpl_interp_k0725.mat'
%   '~/Documents/Data/em/efpl/efpl_interp_k0725_agglo.mat'
% };
% load_indices = [1 1 1];



% load from different saved .mat files generated by knossos_efpl
nload = length(load_indices); o = cell(1,nload); pdata = cell(1,nload); cur_load = '';
for i = 1:nload
  if ~strcmp(cur_load,load_files{i})
    X = load(load_files{i});
  end
  o{i} = X.o{load_indices(i)}; pdata{i} = X.pdata(load_indices(i));
  cur_load = load_files{i}; % avoid reloading the same mat file over and over if we don't need to
end





% plot paramters
pplot = struct;

% for normal pdfs
% fixed width bins
pplot.dplx = 0.0001; pplot.plx = -1.9+pplot.dplx/2:pplot.dplx:2.2-pplot.dplx/2; pplot.nplx = length(pplot.plx);
%pplot.dplx = 0.0005; pplot.plx = -1.9+pplot.dplx/2:pplot.dplx:2.2-pplot.dplx/2; pplot.nplx = length(pplot.plx);
%pplot.dplx = 0.05; pplot.plx = -1.9+pplot.dplx/2:pplot.dplx:2.2-pplot.dplx/2; pplot.nplx = length(pplot.plx);
%pplot.dplx = 0.075; pplot.plx = -1.925+pplot.dplx/2:pplot.dplx:2.2-pplot.dplx/2; pplot.nplx = length(pplot.plx);
%pplot.dplx = 0.1; pplot.plx = -1.9+pplot.dplx/2:pplot.dplx:2.2-pplot.dplx/2; pplot.nplx = length(pplot.plx);
% variable bins
%pplot.dplx = linspace(0.0005,0.05,163); pplot.plx = cumsum(pplot.dplx)-1.9; pplot.nplx = length(pplot.plx);
%pplot.dplx = linspace(0.0001,0.05,164); pplot.plx = cumsum(pplot.dplx)-1.9; pplot.nplx = length(pplot.plx);

% for roc analysis
pplot.dplxs = 0.0001; pplot.plxs = -1.9+pplot.dplxs/2:pplot.dplxs:2.2-pplot.dplxs/2; pplot.nplxs = length(pplot.plxs);

% for node histograms
%pplot.dndx = 1; pplot.ndx = 0+pplot.dndx/2:pplot.dndx:150-pplot.dndx/2; pplot.nndx = length(pplot.ndx);
pplot.dndx = 5; pplot.ndx = 0+pplot.dndx/2:pplot.dndx:2000-pplot.dndx/2; pplot.nndx = length(pplot.ndx);

%pplot.param_name = 'thresholds';
pplot.param_name = '';
pplot.dxticksel = 3;

% this parameter plots the are metric as the sum of 1 - prec/rec instead of typical 1 - f-score
pplot.are_sum = false;

% whether to return the intermediate variables used for plotting in output struct (empty for no)
%pplot.save_plot_results = '';
pplot.save_plot_results = '/home/watkinspv/Data/efpl/efpl_huge_sensitivity_crop_big_sample0p2_meta.mat';

pplot.baseno = 3000;

% for "meta-plots"

pplot.meta_param = 1:6;
pplot.meta_labels = {};
%pplot.meta_labels = {1 2 3 6};
pplot.meta_param_label = 'num training cubes';
pplot.meta_groups = {1:6 7:12};
%pplot.meta_groups_labels = {};
pplot.meta_groups_labels = {'huge', 'none'};

pplot.meta_param = []; % set to disable meta-plots
po = knossos_efpl_plot(pdata,o,pplot);

