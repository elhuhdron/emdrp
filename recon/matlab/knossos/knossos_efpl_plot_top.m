
load('/Users/pwatkins/Downloads/efpl_paper.mat');
%load('/home/watkinspv/Data/agglo/efpl_huge_rf60.mat');
%load('/Data/pwatkins/full_datasets/newestECSall/20151001/efpl.mat');

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
pplot.dndx = 1; pplot.ndx = 0+pplot.dndx/2:pplot.dndx:150-pplot.dndx/2; pplot.nndx = length(pplot.ndx);

pplot.set_thresholds_axis = false;

po = knossos_efpl_plot(pdata,o,pplot);


