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

% Function to parse the directory structure written for Knossos for raw 3D EM data.
% Return the max number of chunks in each dimension, and also an 3d boolean matrix of populated chunks.
function [nchunks, chunk_lists] = parse_knossos_raw(inpath)

nchunks = [-1 -1 -1]; chunk_lists = [];

xdirs_str = findfiles(inpath,'x*',true); 
tmp = xdirs_str'; tmp = tmp(:)'; xdirs = regexp(tmp,'x(\d+)','tokens'); xdirs = str2double([xdirs{:}]); 
nxdirs = length(xdirs);
for x = 1:nxdirs
  xpath = fullfile(inpath,sprintf('x%04d',xdirs(x)));
  ydirs_str = findfiles(xpath,'y*',true);
  tmp = ydirs_str'; tmp = tmp(:)'; ydirs = regexp(tmp,'y(\d+)','tokens'); ydirs = str2double([ydirs{:}]); 
  nydirs = length(ydirs);
  tic; display(['Parsing xdir ' xpath]);
  
  for y = 1:nydirs
    ypath = fullfile(xpath,sprintf('y%04d',ydirs(y)));
    zdirs_str = findfiles(ypath,'z*',true);
    tmp = zdirs_str'; tmp = tmp(:)'; zdirs = regexp(tmp,'z(\d+)','tokens'); zdirs = str2double([zdirs{:}]);
    nzdirs = length(zdirs); 
    
    for z = 1:nzdirs
      zpath = fullfile(ypath,sprintf('z%04d',zdirs(z)));
      raw_str = findfiles(zpath,'*.raw',false);
      if size(raw_str,1) == 1
        cur_chunk = [xdirs(x) ydirs(y) zdirs(z)]; nchunks = max([nchunks;cur_chunk],[],1);
        % no easy way to know how big the hypercube is before parsing the dirs.
        chunk_lists(xdirs(x)+1, ydirs(y)+1, zdirs(z)+1) = true;
      elseif size(raw_str,1) > 1
        error('more than one raw file in dir ''%s''',zpath);
      end % if raw file exists in these subdirs
    end % for each zdir
  end % for each ydir
  toc
end % for each zdir
nchunks = nchunks + 1; % chunks are zero based

function flist = findfiles(fpath,fspec,isdir)
flist = '';
s = dir(fullfile(fpath,fspec));
if length(s)==1
  if (isdir && s.isdir) || (~isdir && ~s.isdir)
    flist = s.name;
  end
elseif ~isempty(s) > 0
  sel = [s.isdir]; if ~isdir, s = ~s; end
  s = s(sel); flist = char(s.name);
end

