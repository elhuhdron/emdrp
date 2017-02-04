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
% struct_in - matlab struct to write, supports int32, float and string fields
% fn_hdffile - path and name of hdf5 file to read
% hdf5path - string name of the path to write the struct to in the hdf file

function write_struct_to_hd5attr(struct_in, fn_hdffile, dataname)

write_fields = fields(struct_in);
for i=1:length(write_fields)
  h5writeatt(fn_hdffile,['/' dataname],write_fields{i},struct_in.(write_fields{i}));
end

