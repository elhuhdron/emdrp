
clear

% parameters for merge (need to match those from runs)
nranges = 8;
range_nums = 1:8;
%range_nums = inf;  # to use all
%connFG = 6; connBG = 26;
connFG = 26; connBG = 6;

% essentially contants for 3d lookup table
nbhd = 3; npix = nbhd*nbhd*nbhd; n = 2^npix; c = ceil(nbhd/2);

% concatenate parallel runs into a single LUT
simpleLUTmerge = zeros(n,1,'uint8');

% the purpose of the range parameters is so that many instances of this can easily be run in parallel.
% then mat files can be merged to make a single lookup table.
assert( mod(n,nranges) == 0 );  % not dealing with remainders
range_size = n / nranges;
if any(~isfinite(range_nums)), range_nums = 1:nranges; end

for range_num=range_nums
  range_beg = (range_num-1)*range_size + 1; range_end = range_beg + range_size - 1;

  load(sprintf('tmp/tmp_out_simpleLUT3d_%dconnFG_%dconnBG_rng%dof%d.mat',connFG,connBG,range_num,nranges));
  simpleLUTmerge(range_beg:range_end) = simpleLUT(range_beg:range_end);
end
simpleLUT = simpleLUTmerge;

% save mat version
save(sprintf('tmp/simpleLUT3d_%dconnFG_%dconnBG.mat',connFG,connBG),'simpleLUT');

% save raw version (for python)
fh = fopen(sprintf('tmp/simpleLUT3d_%dconnFG_%dconnBG.raw',connFG,connBG),'wb');
fwrite(fh,simpleLUT(:),'uint8');
fclose(fh);
