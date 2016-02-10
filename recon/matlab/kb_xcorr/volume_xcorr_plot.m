
function po = volume_xcorr_plot(pdata,pin,o,p)

po = struct;

ndatasets = length(pdata);

% for i = 1:ndatasets
% 
% end 

Cout = cat(4,o{:}.Cout);
Pout = cat(5,o{:}.Poutm);
Pcout = cat(5,o{:}.Pcout);

% remove stuff without much membrane
Cout(squeeze(Pcout(:,:,:,1,:) < 0.2)) = NaN;

nzrng = length(p.zrngs); 
% % do averaging along the zdirection
% if nzrng > 0
%   sz = size(Cout); sz(3) = nzrng; Coutd = zeros(sz);
%   sz = size(Pout); sz(3) = nzrng; Poutd = zeros(sz);
%   sz = size(Pcout); sz(3) = nzrng; Pcoutd = zeros(sz);
%   for i = 1:nzrng
%     zrng = p.zrngs{i};
%     Coutd(:,:,i,:) = nanmean(Cout(:,:,zrng,:),3);
%     Poutd(:,:,i,:,:) = nanmean(Pout(:,:,zrng,:,:),3);
%     Pcoutd(:,:,i,:,:) = nanmean(Pcout(:,:,zrng,:,:),3);
%   end
%   Cout = Coutd; Pout = Poutd; Pcout = Pcoutd;
% end
% do count of "poor" slices along the zdirection
if nzrng > 0
  sz = size(Cout); sz(3) = nzrng; Coutcnt = zeros(sz);
  for i = 1:nzrng
    zrng = p.zrngs{i};
    Coutcnt(:,:,i,:) = sum((Cout(:,:,zrng,:) > 0.4) | ~isfinite(Cout(:,:,zrng,:)),3)/length(zrng);
  end
end


Coutd = reshape(Cout,[],ndatasets);
Poutd = reshape(Pout,[],o{1}.nprobs,ndatasets);

% remove correlations for the training cubes
Csel = (1-Coutd > 0.001);
Pout1 = Poutd(~Csel & isfinite(Coutd),:);  Poutd = Poutd(Csel,:);
Coutd = Coutd(Csel);

figno = 0;

figure(p.baseno+figno); figno = figno+1; clf
scatter(Coutd(:,1),Poutd(:,1,1),ones(size(Coutd(:,1))),'b.'); hold on;
%scatter(Coutd(:,1),Poutd(:,2,1),'g');
%scatter(Coutd(:,1),Poutd(:,3,1),'r');
xlabel('max xcorr'); ylabel('mean prob')
set(gca,'xlim',[0 1],'ylim',[0 1])
legend(pin.load_probs)

X = [ones(size(Coutd(:,1))) squeeze(Coutd(:,1))];
y = squeeze(Poutd(:,1,1));
[b,bint,r,rint,stats] = regress(y,X);
title(sprintf('slope=%g, r2=%g, p=%g',b(2),stats(1),stats(3)))

figure(p.baseno+figno); figno = figno+1; clf
X = Coutd(:,1); Y = Poutd(:,1,1);
%# bin centers (integers)
%xbins = floor(min(X)):1:ceil(max(X));
%ybins = floor(min(Y)):1:ceil(max(Y));
%dx = 0.005
dx = 0.01;
xbins = dx/2:dx:1; ybins = xbins;
xNumBins = numel(xbins); yNumBins = numel(ybins);

%# map X/Y values to bin indices
Xi = round( interp1(xbins, 1:xNumBins, X, 'linear', 'extrap') );
Yi = round( interp1(ybins, 1:yNumBins, Y, 'linear', 'extrap') );

%# limit indices to the range [1,numBins]
Xi = max( min(Xi,xNumBins), 1);
Yi = max( min(Yi,yNumBins), 1);

%# count number of elements in each bin
H = accumarray([Yi(:) Xi(:)], 1, [yNumBins xNumBins]);

%# plot 2D histogram
imagesc(xbins, ybins, log10(H)), axis on %# axis image
colormap gray; colorbar; set(gca,'ydir','normal','plotboxaspectratio',[1 1 1]);
%hold on, plot(X, Y, 'b.', 'MarkerSize',1), hold off


figure(p.baseno+figno); figno = figno+1; clf
subplot(1,2,1);
dx = 0.005; xb = dx/2:dx:1;
c = hist(Coutd,xb); c = cumsum(c)/sum(c);
xlabel('xcorr'); ylabel('cdf')
plot(xb,c);
subplot(1,2,2);
dx = 0.005; xb = dx/2:dx:1;
c = hist(Pout1(:,1),xb); c = cumsum(c)/sum(c);
plot(xb,c,'r'); hold on
c = hist(Poutd(:,1),xb); c = cumsum(c)/sum(c);
plot(xb,c,'b'); hold on
xlabel('mean mem prob'); ylabel('cdf'); legend('training','testing')

%[C,i] = sort(Cout(:));
%[x,y,z] = ind2sub(size(Cout),i(1:10));
[C,i] = sort(Coutcnt(:));
[x,y,z] = ind2sub(size(Coutcnt),i(1:10));
C(1:10)
[x,y,z]
bsxfun( @plus, [fix((x-1)/2) fix((y-1)/2) fix((z-1)/2)], pdata(1).chunk )
[mod(x-1,2), mod(y-1,2), mod(z-1,2)]

