% The MIT License (MIT)
% 
% Copyright (c) 2016 Paul Watkins, National Institutes of Health / NINDS
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

%trainingcube_dir = 'F:\M0007_33\cubes\raw';
trainingcube_dir = '/home/watkinspv/Data/xcorr_raw';

trainingcubes = {'M0007_33_raw_x0017_y0019_z0002.gipl',...
    'M0007_33_raw_x0017_y0023_z0001.gipl',...
    'M0007_33_raw_x0019_y0022_z0002.gipl',...
    'M0007_33_raw_x0022_y0018_z0001.gipl',...
    'M0007_33_raw_x0022_y0023_z0001.gipl',...
    'M0007_33_raw_x0022_y0023_z0002.gipl'};

%testcube_dir = 'F:\M0007_33\cubes\test';
testcube_dir = '/home/watkinspv/Data/xcorr_raw';

testcubes = {'huge_1_raw.gipl',...
             'huge_0_raw.gipl'};
         
trainX = 128;
trainY = 128;
trainZ = 128;

testX = 64;
testY = 64;
testZ = 64;

% load training cubes  
train_imgs = repmat(uint8(0),[trainX trainY trainZ*length(trainingcubes)]);
for count = 1:length(trainingcubes)  
    thistrain = gipl_read_volume(fullfile(trainingcube_dir,trainingcubes{count}));
    train_imgs(:,:,(count-1)*trainZ+1:(count-1)*trainZ+trainZ) = thistrain;
end
A_size = size(train_imgs(:,:,1));

% load test cubes        
test_imgs = repmat(uint8(0),[testX testY testZ length(testcubes)]);
for count = 1:length(testcubes)  
    thistest = gipl_read_volume(fullfile(testcube_dir,testcubes{count}));
    test_imgs(:,:,:,count) = thistest;
end
T_size = size(test_imgs(:,:,1));

% precompute training cube xcorr variables and fft
m = testY;
n = testX;
mn = m*n;
outsize = A_size + T_size - 1;
tic
%local_sum_A = []; denom_A = []; Fb = [];
local_sum_A = zeros([outsize size(train_imgs,3)]);
denom_A = zeros([outsize size(train_imgs,3)]);
Fb = zeros([outsize size(train_imgs,3)]);
for traincount = 1:size(train_imgs,3)
    A = single(train_imgs(:,:,traincount));
    local_sum_A(:,:,traincount) = local_sum(A,m,n);
    local_sum_A2 = local_sum(A.*A,m,n);
    diff_local_sums = ( local_sum_A2 - (local_sum_A(:,:,traincount).^2)/mn );
    denom_A(:,:,traincount) = sqrt( max(diff_local_sums,0) ); 
    Fb(:,:,traincount) = fft2(A,outsize(1),outsize(2));
end
toc

% augmentations
% 1 = none
% 2 = flipud
% 3 = fliplr
% 4 = fliplr, flipud
% 4 = rot90
% 5 = rotminus90
% 6 = flipud, rot90
% 7 = flipud, rotminus90
% 8 = fliplr, flipud, rot90

% figure;
% subplot(2,5,1)
% imagesc(thistestimg); colormap('gray'); axis square
% subplot(2,5,2)
% imagesc(flipud(thistestimg)); colormap('gray'); axis square
% subplot(2,5,3)
% imagesc(fliplr(thistestimg)); colormap('gray'); axis square
% subplot(2,5,4)
% imagesc(fliplr(flipud((thistestimg)))); colormap('gray'); axis square
% subplot(2,5,6)
% imagesc(rot90(thistestimg)); colormap('gray'); axis square
% subplot(2,5,7)
% imagesc(rot90(thistestimg,-1)); colormap('gray'); axis square
% subplot(2,5,8)
% imagesc(flipud(rot90(thistestimg))); colormap('gray'); axis square
% subplot(2,5,9)
% imagesc(flipud(rot90(thistestimg,-1))); colormap('gray'); axis square

n_augmentations = 8;
C = zeros(size(train_imgs,3),size(test_imgs,3),n_augmentations,length(testcubes));
for testcubecount = 1:length(testcubes)
    %for augmentcount = 7:8
    for augmentcount = 1:8
        tic
        for testcount = 1:size(test_imgs,3)
            %tic
            thistestimg = test_imgs(:,:,testcount,testcubecount);
            %       temp = train_imgs(:,:,testcount);
            %       thistestimg = temp(64-31:64+32,64-31:64+32);
            switch augmentcount
                case 1
                    
                case 2
                    thistestimg= flipud(thistestimg);
                case 3
                    thistestimg= fliplr(thistestimg);
                case 4
                    %thistestimg= fliplr(flipud(thistestimg));
                    thistestimg= rot90(thistestimg,2);
                case 5
                    thistestimg= rot90(thistestimg);
                case 6
                    thistestimg= rot90(thistestimg,-1);
                case 7
                    thistestimg= flipud(rot90(thistestimg));
                case 8
                    thistestimg= flipud(rot90(thistestimg,-1));
            end
            for traincount=1:size(train_imgs,3)
                A = single(train_imgs(:,:,1));
                RESULT = normxcorr2_kb(single(thistestimg),single(A),local_sum_A(:,:,traincount),...
                  denom_A(:,:,traincount),Fb(:,:,traincount));
                RESULT(isinf(RESULT)) = 0;
                C(traincount,testcount,augmentcount,testcubecount) = max(RESULT(:));
            end
            %toc
        end
        toc
    end
end
max(C(:,:,:,1),[],1)
    
thiscube = 1;
cube_1_max = max(max(squeeze(max(C(:,:,:,thiscube),[],1)),[],3),[],2);
[huge_1_hist,bincenters] = hist(cube_1_max(:),0:0.01:1);

thiscube = 2;
cube_0_max = max(max(squeeze(max(C(:,:,:,thiscube),[],1)),[],3),[],2);
[huge_0_hist,bincenters] = hist(cube_0_max(:),0:0.01:1);

figure; plot(bincenters,huge_1_hist,'k'); hold on; plot(bincenters,huge_0_hist,'r');
legend({'huge 1 raw.gipl','huge 0 raw.gipl'})
xlabel('norm xcorr')
ylabel('# slices')

figure; plot(cube_1_max,'.-k')
hold on; plot(cube_0_max,'.-r')
plot(1:testZ,repmat(median(cube_1_max),[testZ 1]),'--k')
plot(1:testZ,repmat(median(cube_0_max),[testZ 1]),'--r')
legend({'huge 1 raw.gipl','huge 0 raw.gipl'})
xlabel('slice #')
ylabel('norm xcorr')
