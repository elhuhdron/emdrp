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

% Function to write EM data chunk to hd5 file
%
% fn_hdffile - path and name of hdf5 file to read
% dataset - string name of the dataset to read from the hdf5 file
% chunk_in - zero based index of the chunk to read from the hdf5 file (for example, [-1 0 1])

function write_hd5_chunk(V,fn_hdffile, dataset, chunk_ind, do_Corder, chunk_ind_centered)
if(~exist('do_Corder','var') || isempty(do_Corder)), do_Corder = true; end
if(~exist('chunk_ind_centered','var') || isempty(chunk_ind_centered)), chunk_ind_centered = true; end

inf = h5info(fn_hdffile);
ind = find(strcmp({inf.Datasets.Name},dataset)); % find the dataset in the info struct
if do_Corder
  chunksize = inf.Datasets(ind).ChunkSize([3 2 1]); 
  datasize = inf.Datasets(ind).Dataspace(1).Size([3 2 1]);
else
  chunksize = inf.Datasets(ind).ChunkSize;
  datasize = inf.Datasets(ind).Dataspace(1).Size;
end
sizeV = size(V); if ismatrix(V), sizeV = [sizeV 1]; end
if ~all(sizeV == chunksize), error('write_hd5_chunk: bad volume'); end

if chunk_ind_centered
  nchunks = datasize ./ chunksize;
  ind = 1 + (chunk_ind + ceil(nchunks/2) - 1).*chunksize; % index into the dataset using the chunksize
else
  ind = 1 + chunk_ind.*chunksize; % index into the dataset using the chunksize
end
if do_Corder
  h5write(fn_hdffile,['/' dataset],permute(V,[3 2 1]),ind([3 2 1]),chunksize([3 2 1]));
else
  h5write(fn_hdffile,['/' dataset],V,ind,chunksize);
end  

