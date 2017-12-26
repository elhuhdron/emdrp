
% gets appended to legend names for each load_file/index, leave empty for none
name_suffixes = {};

load_files = {
  '~/Data/efpl/vgg3pool64_k0725_ds2_run1_interp.mat'
  '~/Data/efpl/vgg3pool64_k0725_ds2_run1_interp.mat'
};
load_indices = [1 2];

% load from different saved .mat files generated by knossos_efpl
nload = length(load_indices); o = cell(1,nload); pdata = cell(1,nload); pefpl = cell(1,nload); cur_load = '';
for i = 1:nload
  if ~strcmp(cur_load,load_files{i})
    X = load(load_files{i});
  end
  o{i} = X.o{load_indices(i)}; pdata{i} = X.pdata(load_indices(i)); pefpl{i} = X.p;
  cur_load = load_files{i}; % avoid reloading the same mat file over and over if we don't need to

  if pefpl{i}.skeleton_mode
    pdata{i}.node_radius = -1;
  elseif isfield(pefpl{i},'node_radius')
    % old mode, did not have node radius in pdata
    pdata{i}.node_radius = pefpl{i}.node_radius;
  end
  
  if ~isempty(name_suffixes)
    % gets appended to legend names for each load_file/index
    pdata{i}.name_suffixes = name_suffixes{i};
  end
end



% plot paramters
pplot = struct;

% for normal pdfs
pplot.dplx = 0.0001; pplot.plx = -1.9+pplot.dplx/2:pplot.dplx:2.2-pplot.dplx/2; pplot.nplx = length(pplot.plx);

% for roc analysis
pplot.dplxs = 0.0001; pplot.plxs = -1.9+pplot.dplxs/2:pplot.dplxs:2.2-pplot.dplxs/2; pplot.nplxs = length(pplot.plxs);

% for node histograms
pplot.dndx = 5; pplot.ndx = 0+pplot.dndx/2:pplot.dndx:2000-pplot.dndx/2; pplot.nndx = length(pplot.ndx);

% if comparing single parameter space, set name here
pplot.param_name = '';
pplot.dxticksel = 3;

% use the efpl where each edge is counted once in distribution (vs per error free components)
pplot.use_efpl_edges = false;

% plot the efpls related metrics (were not used in paper)
pplot.plot_efpl_metrics = false;

% whether to normalize efpl median and auroc to log scales
pplot.plot_efpl_metrics_log = false;
  
% plot density of error free path length versus error free diameter scatters (were not used in paper)
pplot.plot_efpl_diameters = false;

% this parameter plots the are metric as the sum of 1 - prec/rec instead of typical 1 - f-score
pplot.are_sum = false;

% whether to return the intermediate variables used for plotting in output struct (empty for no)
pplot.save_plot_results = '';

pplot.baseno = 1000; % first figure number

pplot.meta_param = []; % set to disable meta-plots



po = knossos_efpl_plot(pdata,o,pplot);
