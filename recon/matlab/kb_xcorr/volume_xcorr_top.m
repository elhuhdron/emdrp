
pdata = struct;  % input parameters depending on dataset

% with ~20% ECS
pdata(1).datah5 = '/Data/big_datasets/M0007_33_39x35x7chunks_Forder.h5';
% corner chunk
pdata(1).chunk = [16 17 0];  % all
%pdata(1).chunk = [16 17 0]; % 1
%pdata(1).chunk = [20 17 0]; % 2
%pdata(1).chunk = [16 21 0]; % 3
%pdata(1).chunk = [20 21 0]; % 4
pdata(1).train_chunks = [19 22 2; 17,19,2; 17,23,1; 22,23,1; 22,18,1; 22,23,2];
pdata(1).lblsh5 = '/Data/pwatkins/full_datasets/newestECSall/20151001/huge_supervoxels.h5';
pdata(1).probh5 = '/Data/pwatkins/full_datasets/newestECSall/huge_probs.h5';
pdata(1).name = 'huge';

p = struct;  % input parameters independent of dataset

p.matlab_base = [1 1 1];  % matlab starts at 1 !!!
p.empty_label = uint32(2^32-1);
p.load_probs = {'MEM', 'ICS', 'ECS'};
%p.load_probs = {'MEM'};
p.nchunks = [8 8 4]; % for real
%p.nchunks = [1 1 1];  % for small test
%p.nchunks = [5 5 1];  % for medium test
%p.nchunks = [4 4 4]; % for real 4 processes
p.skip = [0 0 32];
p.test_size = [64 64];
p.dataset_data = 'data_mag1';
p.dataset_lbls = 'labels';
p.dataset_type = 'voxel_type';
p.run_xcorr = false;

p.train_nchunks = [1 1 1];
p.train_offset = [0 0 0];

% run xcorr for each dataset
o = cell(1,length(pdata));
for i = 1:length(pdata)
  fprintf(1,'\nRunning xcorr for "%s"\n\n',pdata(i).name);
  o{i} = volume_xcorr(p,pdata(i));
end

% save the results
save('/home/watkinspv/Data/xcorr_out/xcorr_all_ponly.mat','p','pdata','o');
