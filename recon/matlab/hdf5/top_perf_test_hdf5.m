
p = struct;

% total size of the volume (edge of entire data cube)
p.edge_size = 1024;

% cube sizes to test
p.cube_sizes = [64 128 256 512 1024];

% compression levels to test (zero for no compression)
% xxx - compression ignored for raw, no easy in memory zip compression for matlab
p.compressions = [0 5];

% need chunking to enable compression in hdf5
p.chunk_size = [128 128 128];

% data type of data to read / write
p.dtypes = {'uint8', 'uint32', 'single'}; 

% location to read / write test data
p.path = '/Data/watkinspv/test';

% number of times to repeat each test (for average speed)
p.nrepeats = 10;

% % for small test
% p.edge_size = 512;
% p.cube_sizes = [128 256 512];
% p.compressions = [0 5];
% p.dtypes = {'uint8'}; 
% p.nrepeats = 1;

o = perf_test_hdf5(p);

save('tmp_out.mat');
