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

function o = volume_xcorr(p, pdata)
o = struct;

fprintf(1,'getting dataset info from h5 files\n');
inf = h5info(pdata.datah5);
ind = find(strcmp({inf.Datasets.Name},p.dataset_data)); % find the dataset in the info struct
o.chunksize = inf.Datasets(ind).ChunkSize;
o.datasize = inf.Datasets(ind).Dataspace(1).Size;
o.scale = h5readatt(pdata.datah5,['/' p.dataset_data],'scale')';

% other calculated values based on params
o.loadsize = p.nchunks.*o.chunksize - p.skip;
o.loadcorner = pdata.chunk.*o.chunksize + p.skip;

assert( all(mod(o.loadsize(1:2), p.test_size) == 0) )
o.ntest = o.loadsize(1:2) ./ p.test_size;

% load all the data, "offset" used to skip some data (xxx - bad name same as in efpl code)

fprintf(1,'reading data\n'); t = now;
Vdata = h5read(pdata.datah5,['/' p.dataset_data],o.loadcorner+p.matlab_base,o.loadsize);
display(sprintf('\tdone in %.3f s',(now-t)*86400));

Vprobs = zeros([o.loadsize length(p.load_probs)],'single');
o.nprobs = length(p.load_probs);
for i = 1:o.nprobs
  fprintf(1,'reading probabilities %s\n',p.load_probs{i}); t = now;
  Vprobs(:,:,:,i) = h5read(pdata.probh5,['/' p.load_probs{i}],o.loadcorner+p.matlab_base,o.loadsize);
  display(sprintf('\tdone in %.3f s',(now-t)*86400));
end

fprintf(1,'reading voxel types\n'); t = now;
Vtypes = h5read(pdata.lblsh5,'/voxel_type',o.loadcorner+p.matlab_base,o.loadsize);
display(sprintf('\t\tdone in %.3f s',(now-t)*86400));

% load the training cube data, use normal meaning for offset

o.train_loadsize = p.train_nchunks.*o.chunksize;
o.ntrain = size(pdata.train_chunks,1);
o.train_loadcorner = zeros(o.ntrain,3);
Vtrain_data = zeros([o.train_loadsize o.ntrain],'uint8');
fprintf(1,'loading training data\n'); t = now;
for n = 1:o.ntrain
  fprintf(1,'\treading training data %s %d %d %d\n', pdata.name, pdata.train_chunks(n,:));
  o.train_loadcorner(n,:) = pdata.train_chunks(n,:).*o.chunksize + p.train_offset;
  Vtrain_data(:,:,:,n) = h5read(pdata.datah5,['/' p.dataset_data],o.train_loadcorner(n,:)+p.matlab_base,...
    o.train_loadsize);
end
display(sprintf('\tdone in %.3f s',(now-t)*86400));

% unroll training cubes
o.nztrain = o.train_loadsize(3)*o.ntrain;
train_imgs = reshape(Vtrain_data,[o.train_loadsize(1:2) o.nztrain]);

% allocate the outputs
o.Cout = zeros([o.ntest o.loadsize(3)]);
o.Poutm = zeros([o.ntest o.loadsize(3) o.nprobs]);
o.Pouts = zeros([o.ntest o.loadsize(3) o.nprobs]);
o.Pcout = zeros([o.ntest o.loadsize(3) o.nprobs]);

m = p.test_size(1); n = p.test_size(2); mn = m*n;
if p.run_xcorr
  % kevin's precompute fft loop
  A_size = o.train_loadsize(1:2);
  T_size = p.test_size;
  outsize = A_size + T_size - 1;
  fprintf(1,'precompute fft loop\n'); t = now;
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
  display(sprintf('\tdone in %.3f s',(now-t)*86400));
end

fprintf(1,'running %d x %d test images for %d slices\n',o.ntest(1),o.ntest(2),o.loadsize(3));

for x = 1:o.ntest(1)
  for y = 1:o.ntest(2)
    fprintf(1,'processing cube x %d y %d\n',x,y); touter = now;
    for z = 1:o.loadsize(3)
      xrng = (x-1)*p.test_size(1)+1:x*p.test_size(1);
      yrng = (y-1)*p.test_size(2)+1:y*p.test_size(2);
      thistestimg = Vdata(xrng,yrng,z);

      if p.run_xcorr
        % kevin's xcorr inner loop
        %fprintf(1,'inner traincount loop\n'); tinner = now;
        C = zeros(1,o.nztrain);
        for traincount=1:o.nztrain
          A = single(train_imgs(:,:,1));
          RESULT = normxcorr2_kb(single(thistestimg),single(A),local_sum_A(:,:,traincount),...
            denom_A(:,:,traincount),Fb(:,:,traincount));
          RESULT(isinf(RESULT)) = 0;
          C(traincount) = max(RESULT(:));
        end
        %display(sprintf('\tdone in %.3f s',(now-tinner)*86400));
        
        % xxx - what to apply to train image results, mean? max? median? ...
        %o.Cout(x,y,z) = mean(C);
        o.Cout(x,y,z) = max(C);
      end
      
      % get the mean of the specified probabilities where that type is the winnner corresponding to test image
      thistesttypes = squeeze(Vtypes(xrng,yrng,z));
      % just a sanity check
      %t = reshape(squeeze(Vprobs(xrng,yrng,z,:)),[mn o.nprobs]); [~,j] = max(t,[],2);
      %assert( all(thistesttypes(:) == j-1) );
      for i = 1:o.nprobs
        thistestprobs = squeeze(Vprobs(xrng,yrng,z,i));
        winner_sel = thistesttypes==i-1;
        winner_probs = thistestprobs(winner_sel);
        % xxx - what measure to use here?
        o.Poutm(x,y,z,i) = mean(winner_probs);
        o.Pouts(x,y,z,i) = sqrt(sum((winner_probs-1).^2)/(length(winner_probs)-1));
        o.Pcout(x,y,z,i) = sum(winner_sel(:))/mn;
      end
      
    end
    display(sprintf('\tdone in %.3f s',(now-touter)*86400));
  end
end

