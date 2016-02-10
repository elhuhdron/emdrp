

%load('/home/watkinspv/Data/xcorr_out/xcorr.mat','p','pdata','o');
load('/home/watkinspv/Data/xcorr_out/xcorr_all.mat','p','pdata','o');

% X = load('/home/watkinspv/Data/xcorr_out/xcorr_test2.mat','o');
% for i = 1:length(pdata)
%   o{i}.Pout = X.o{i}.Pout;
% end

pplot = struct;
pplot.baseno = 2000;
pplot.zrngs = {1:32, 33:96, 97:160, 161:224, 225:288, 289:352, 353:416, 417:480};
%pplot.zrngs = {};

po = volume_xcorr_plot(pdata,p,o,pplot);
