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

% Function for creating EM data hdf5 file from Knossos raw format.
function make_hdf5_from_knossos_raw(p)

assert( ~p.do_chunk_select || length(p.mags)==1 ); % xxx - can't do select with multiple mags right now

tall = now;
knossos_conf = parse_knossos_conf(p.inpath, p.knossos_conf_fn);
rawtotal = prod(int64(p.rawsize));

matname = [p.dataset '_' data_name '_dirs_info.mat'];
if use_conf_size
  fprintf(1,'Using Knossos conf size %d,%d,%d\n',knossos_conf.boundary);
  if p.scale_conf_size
    knossos_conf.boundary = fix(knossos_conf.boundary / mag);
    fprintf(1,'Scaling Knossos conf size to %d,%d,%d\n',knossos_conf.boundary);
  end
  nchunks = ceil(double(knossos_conf.boundary) ./ p.rawsize);
  chunk_lists = true(nchunks);
  fprintf(1,'Ceil size size %d,%d,%d\n',nchunks.*p.chunksize);
elseif p.reparse_dirs || ~exist(matname,'file')
  % parse the directories to get the total number of chunks, or load previously stored .mat file with parsed info
  % xxx - this is a legacy mode, decided it's better to keep the span of the dataset consistent with knossos
  [nchunks, chunk_lists] = parse_knossos_raw(p.inpath);
  save(matname,'nchunks','chunk_lists');
else
  load(matname);
end

% this has to come after parsing the dirs, so we know the total number of chunks.
% force some parameters if not doing any chunk select (writing entire dataset to hdf5).
if ~p.do_chunk_select
  p.nchunks_sel = nchunks;
  p.chunk_sel_offset = [0 0 0];
  p.do_chunk_select_crop = true;
end

chunksizer = p.chunksize; p.chunksize = chunksize(p.dim_order);
nchunks_selr = p.nchunks_sel; p.nchunks_sel = nchunks_sel(p.dim_order);
if p.do_chunk_select_crop
  totalsize = p.chunksize.*p.nchunks_sel;
else
  totalsize = p.chunksize.*nchunks;
end
bytes_per_chunk = prod(p.chunksize);

% put anything to be written to the atributes section for the data into this struct.
% use double precision for floats and int32 for integers.
data_conf = struct;
data_conf.dataset = p.dataset;
data_conf.rawsize = int32(p.rawsize);
data_conf.isCorder = int32(p.do_Corder);
data_conf.isSubset = int32(p.do_chunk_select);
data_conf.chunkOffset = int32(p.chunk_sel_offset);
data_conf.dimOrdering = int32(p.dim_order);
data_conf.nchunks = int32(p.nchunks_sel);
% debated on this as downsample or upsample factor, upsampling from native not common, so use downsample factor.
data_conf.factor = double([mag mag mag]);

%   if do_Corder, str = 'Corder'; else str = 'Forder'; end
%   if do_chunk_select
%     hdffname = sprintf('%s_%dx%dx%dchunks_at%d-%d-%d_%s.h5',dataset,nchunks_selr,chunk_sel_offset,str);
%   else
%     hdffname = sprintf('%s_%dx%dx%dchunks_%s.h5',dataset,nchunks,str);
%   end
if p.do_chunk_select
  if p.do_chunk_select_crop
    hdffname = sprintf('%s_%dx%dx%dchunks_at_x%04d_y%04d_z%04d_crop.h5',p.dataset,nchunks_selr,p.chunk_sel_offset);
  else
    hdffname = sprintf('%s_%dx%dx%dchunks_at_x%04d_y%04d_z%04d.h5',p.dataset,nchunks_selr,p.chunk_sel_offset);
  end
else
  hdffname = [p.dataset '.h5'];
end
outfile = fullfile(p.outpath,hdffname);

if p.do_write
  % xxx - be carefule with this, not sure how to delete an existing dataset
  evalc('delete(outfile)'); % silently remove existing file
  if p.do_Corder
    h5create(outfile,['/' data_name],totalsize([3 2 1]),'ChunkSize',p.chunksize([3 2 1]),'Datatype','uint8',...
      'Deflate',5,'Fletcher32',true,'Shuffle',true,'FillValue',uint8(0));
  else
    h5create(outfile,['/' data_name],totalsize,'ChunkSize',p.chunksize,'Datatype','uint8','Deflate',5,...
      'Fletcher32',true,'Shuffle',true,'FillValue',uint8(0));
  end
end

for ix1 = 1:nchunks(1)
  t = now; block_disp = false;
  for ix2 = 1:nchunks(2)
    for ix3 = 1:nchunks(3)
      if chunk_lists(ix1,ix2,ix3)
        chunk_lists(ix1,ix2,ix3) = false; % set back to true below if this chunk is valid
        chunk_ind = [ix1 ix2 ix3]-1;
        chunk_ind_write = chunk_ind - p.chunk_sel_offset;
        if all(chunk_ind_write >= 0 & chunk_ind_write < nchunks_selr)
          if ~p.do_chunk_select_crop
            chunk_ind_write = chunk_ind_write + p.chunk_sel_offset;
          end
          if ~block_disp
            disp(['writing hdf mag' num2str(mag) ' data for all-yz chunks starting at chunk ',num2str(chunk_ind),...
              ' to all-yz chunks starting at chunk ', num2str(chunk_ind_write),' in ',hdffname]);
            block_disp = true;
          end
          fn = fullfile(p.inpath,sprintf('x%04d',chunk_ind(1)),sprintf('y%04d',chunk_ind(2)),...
            sprintf('z%04d',chunk_ind(3)),sprintf('%s_x%04d_y%04d_z%04d.raw',raw_prefix,chunk_ind));
          if exist(fn,'file')
            fh = fopen(fn); V = fread(fh,bytes_per_chunk,'uint8=>uint8'); fclose(fh);
            if length(V) == rawtotal
              chunk_lists(ix1,ix2,ix3) = true; % valid chunk
              V = reshape(V,chunksizer);
              if p.do_write
                write_hd5_chunk(permute(V,dim_order),outfile, data_name, chunk_ind_write(p.dim_order), ...
                  p.do_Corder, false);
              end
            end % if raw file is correct size
          end % if file exists
        end % if chunk selected for write
      end % if in chunk list
    end % for z chunk
  end % for y chunk
  if block_disp
    display(sprintf('\tdone in %.3f s',(now-t)*86400));
  end
end % for x chunk

% write all the meta data, knossos conf and data conf
if p.do_write
  write_struct_to_hd5attr(knossos_conf,outfile,data_name);
  write_struct_to_hd5attr(data_conf,outfile,data_name);
  
  % write out another dataset that contains a boolean mask of valid chunks (chunk_lists)
  cname = [data_name '_chunk_mask']; csize = size(chunk_lists);
  cdata = uint8(permute(chunk_lists,p.dim_order));
  if p.do_Corder
    h5create(outfile,['/' cname],csize([3 2 1]),'ChunkSize',csize,'Datatype','uint8',...
      'Deflate',5,'Fletcher32',true,'Shuffle',true,'FillValue',uint8(0));
    h5write(outfile,['/' cname],permute(cdata,[3 2 1]),[1 1 1],csize([3 2 1]));
  else
    h5create(outfile,['/' cname],csize,'ChunkSize',csize,'Datatype','uint8','Deflate',5,...
      'Fletcher32',true,'Shuffle',true,'FillValue',uint8(0));
    h5write(outfile,['/' cname],cdata,[1 1 1],csize);
  end
  
  h5disp(outfile)
end

display(sprintf('done writing %s in %.3f s',data_name,(now-tall)*86400));
