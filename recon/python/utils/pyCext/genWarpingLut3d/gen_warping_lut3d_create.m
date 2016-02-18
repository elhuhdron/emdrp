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

% Function to create an F-order lookup table (LUT) for all pixel combinations of
%   a 3d neighborhood to assess whether the center pixel is a simple point or not.
%
% The basic algorithm is described by Theorem 9.1 in Kong and Rosenfeld (1989).
% This is esentially the most brute force check for a simple point, but since it's only run once to create the lookup
%   table, this is not a concern for efficiency. It is also highly parallelizable (each patch could be in parallel).
% Makes use of "Minkowski" toolbox from David Legland, see Legland, Kieu and Devaux (2007).
%   Toolbox was acquired from matlab central. Used to calculate 3d Euler number.
%
% NOTE: matlab warping error has not been updated to reflect this LUT creation.
%   Instead see old version called lut_3dwarping_error_create_simple_LUT.m
%   This code generates the LUT just for the python/C version of the binary warping. 
%   Kept this code in matlab because of Minkowski toolbox. 
%
% Example invocation:
%   gen_warping_lut3d_create(6, 26, 1, 1, 1000)
% 
% Use gen_warping_lut3d_top.sh to run multiple instances in parallel (without parallel computing toolbox).

function gen_warping_lut3d_create(connFG, connBG, nranges, range_num, print_every)

% connectivities must be 6,18 or 26 for 3d
assert( any(connFG == [6,18,26]) && any(connBG == [6,18,26]) );
% foreground and background connectivities must be different and one must be 6 (see Kong & Rosenfeld)
assert( connFG ~= connBG ); assert( (connFG==6) || (connBG==6) );

% xxx - imEuler3d does not support connectivity of 18
assert( (connFG ~= 18) && (connBG ~= 18) );

% these are essentially constants for 3d warping
nbhd = 3; npix = nbhd*nbhd*nbhd; n = 2^npix; c = ceil(nbhd/2);
nbhdFG = gen_warping_lut3d_get_nbhd(connFG); nbhdBG = gen_warping_lut3d_get_nbhd(connBG);

% selpix determines ordering for LUT, which end starts the numbering for the patch, just for debug.
%selpix = npix:-1:1; % last logical of batch is least significant bit of index
% MUST use first logical of batch is least significant bit of index for matlab/python warping code.
selpix = 1:npix; % first logical of batch is least significant bit of index

% simpleLUT is zero for simple points, non-simple points classified (see enum below).
simpleLUT = zeros(n,1,'uint8');

% the purpose of the range parameters is so that many instances of this can easily be run in parallel
% then mat files can be merged to make a single lookup table (use gen_warping_lut3d_merge.m).
assert( mod(n,nranges) == 0 );  % not dealing with remainders
range_size = n / nranges;
range_beg = (range_num-1)*range_size + 1; range_end = range_beg + range_size - 1;
% rng determines the order over which to iterate the patches. just for debug.
rng = range_beg:1:range_end;    % starting at all zeros (normal mode)
%rng = range_end:-1:range_beg;   % starting at all ones

% Create an enum with corresponding description of nonsimple point types.
point_type.SIMPLE_POINT = 0;    % simple points are zero in the LUT
% for flipping nonsimple background point
point_type.CREATE_OBJECT = 1;
point_type.DELETE_CAVITY = 2;
point_type.RESULT_MERGE = 3;
% for flipping nonsimple foreground point
point_type.CREATE_CAVITY = 5;
point_type.DELETE_OBJECT = 6;
point_type.RESULT_SPLIT = 7;
% tunnels can be created or destroyed by either flipping nonsimple foreground or background points
point_type.DELETE_TUNNEL = 4;
point_type.CREATE_TUNNEL = 8;
point_type.STR_NONSIMPLE = {...
  '0=>1 create object (missing object)','0=>1 delete cavity (extra cavity)',...
  '0=>1 merger (split objects)','delete tunnel (extra tunnel)',...
  '1=>0 create cavity (missing cavity)','1=>0 delete object (extra object)',...
  '1=>0 split (merged objects)','create tunnel (missing tunnel)'};

tloop = now;
for i=rng
  if isfinite(print_every) && mod(i,print_every)==0
    display(sprintf('i=%d / %d, time = %.2f s',i,range_end,(now-tloop)*86400)); tloop=now; 
  end
  
  % create the F-order patch based on the index
  patch = reshape(bitget(i-1,selpix,'uint32'),[nbhd nbhd nbhd]); %patch, pause
  isBGpoint = (patch(c,c,c) == 0);
  
  % This long if/elseif is basically Theorem 9.1 from Kong and Rosenfeld.
  % Need only one adjacent FG component and one adjacent BG component for simple point.
  % If these conditions hold, then the number of tunnels must stay the same.
  
  patchFG = patch; patchFG(c,c,c) = 0; lblFG = bwlabeln(patchFG,connFG);
  % note that checking for adjacency does not include the center pixel and uses only points defined by connectivity
  nadjacentFG = length(labels_unique_nonzeros(lblFG.*nbhdFG));
  if nadjacentFG == 0
    % there are no adjacent foreground components
    if isBGpoint
      simpleLUT(i) = point_type.CREATE_OBJECT;
      %point_type.STR_NONSIMPLE{simpleLUT(i)}, patch, pause
    else
      simpleLUT(i) = point_type.DELETE_OBJECT;
      %point_type.STR_NONSIMPLE{simpleLUT(i)}, patch, pause
    end
  elseif nadjacentFG > 1
    % there is more than one adjacent foreground component
    if isBGpoint
      simpleLUT(i) = point_type.RESULT_MERGE;
      %point_type.STR_NONSIMPLE{simpleLUT(i)}, patch, pause
    else
      simpleLUT(i) = point_type.RESULT_SPLIT;
      %point_type.STR_NONSIMPLE{simpleLUT(i)}, patch, pause
    end
  else % one adjacent foreground component
    patchBG = ~patch; patchBG(c,c,c) = 1; lblBG = bwlabeln(patchBG,connBG);
    % note that checking for adjacency does not include the center pixel and uses only points defined by connectivity
    nadjacentBG = length(labels_unique_nonzeros(lblBG.*nbhdBG));
    if nadjacentBG == 0
      % there are no adjacent background components
      if isBGpoint
        simpleLUT(i) = point_type.DELETE_CAVITY;
        %point_type.STR_NONSIMPLE{simpleLUT(i)}, patch, pause
      else
        simpleLUT(i) = point_type.CREATE_CAVITY;
        %point_type.STR_NONSIMPLE{simpleLUT(i)}, patch, pause
      end
    elseif nadjacentBG > 1
      % there is more than one adjacent background component
      assert(connBG~=26); % should not happen since center pixel connected to everything in this case
      if isBGpoint
        simpleLUT(i) = point_type.DELETE_TUNNEL;
        %point_type.STR_NONSIMPLE{simpleLUT(i)}, patch, pause
      else 
        simpleLUT(i) = point_type.CREATE_TUNNEL;
        %point_type.STR_NONSIMPLE{simpleLUT(i)}, patch, pause
      end
    else % one adjacent background component
      % this could be a simple point, still have to check for same number of tunnels.
      patch_flip = patch; patch_flip(c,c,c) = ~patch_flip(c,c,c);
      % imEuler3d only allows to specify foreground connectivity.
      % xxx - Kong and Rosenfeld has both connectivities as parameters to the Euler number.
      %   assuming here that FG==6 means BG==26 and vice versa for imEuler3d, imEuler3d does not support 18.
      euler_patch = imEuler3d(patch,connFG); euler_patch_flip = imEuler3d(patch_flip,connFG);
      if euler_patch < euler_patch_flip
        simpleLUT(i) = point_type.DELETE_TUNNEL;
        %point_type.STR_NONSIMPLE{simpleLUT(i)}, patch, pause
      elseif euler_patch > euler_patch_flip
        simpleLUT(i) = point_type.CREATE_TUNNEL;
        %point_type.STR_NONSIMPLE{simpleLUT(i)}, patch, pause
      else % since previous conditions were passed, now the same euler number means the same number of tunnels
        simpleLUT(i) = point_type.SIMPLE_POINT;
        %'Simple!', patch, pause
      end
    end % number of adjacent background components
  end % number of adjacent foreground components
  
end % for all possible patches in nbhd

save(sprintf('tmp/tmp_out_simpleLUT3d_%dconnFG_%dconnBG_rng%dof%d.mat',connFG,connBG,range_num,nranges),'simpleLUT');




function nbhd = gen_warping_lut3d_get_nbhd(conn)

% calling code assumes that center point in neighborhood is zero
if conn==6
  nbhd = cat(3, ...
    [0 0 0; 0 1 0; 0 0 0],...
    [0 1 0; 1 0 1; 0 1 0],...
    [0 0 0; 0 1 0; 0 0 0]);
elseif conn==18
  nbhd = cat(3, ...
    [0 1 0; 1 1 1; 0 1 0],...
    [1 1 1; 1 0 1; 1 1 1],...
    [0 1 0; 1 1 1; 0 1 0]);
elseif conn==26
  nbhd = ones(3,3,3); nbhd(2,2,2) = 0;
end

