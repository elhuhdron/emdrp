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

% Top level script for creating EM data hdf5 file from Knossos raw format.

% START PARAMETERS

% Iterate over multiple magnifications and write to separate datasets in hdf5
%mags = [1 2 4 8 16];
mags = 1;

for mag = mags

  % Name of the dataset, stored in meta, used in filename
  %dataset = 'M0007_33';
  %dataset = 'k0725';
  dataset = 'K0057_D31';

  % Paths to root of Knossos raw data and path to where hdf5 should be written.
  %inpath = sprintf('/mnt/fs/common/ECS_paper/ECS_3d_analysis/M0007_33/cubes/M0007_33_mag%d',mag);
  %outpath = '/Data/big_datasets';
  %inpath = sprintf('/mnt/cdcu/common/110629_k0725/cubes/%s_mag%d',dataset,mag);
  inpath = sprintf('/run/media/watkinspv/My Passport/K0057_D31/cubes/K0057_D31_mag%d',mag);
  outpath = '/Data/watkinspv';

  % The raw size of the Knossos cubes
  rawsize = [128 128 128];
  
  % name of the Knossos configuration file with data to be added to hdf5 meta folder
  knossos_conf_fn = 'Knossos.conf';
  
  % The prefix of the raw file names (Knossos cubes)
  %raw_prefix = sprintf('M0007_33_mag%d',mag);
  %raw_prefix = sprintf('110629_k0725_mag%d',mag);
  raw_prefix = sprintf('K0057_D31_mag%d',mag);
  
  % Chunksize written to hdf5 file, typically same as the Knossos raw size
  chunksize = rawsize;
  
  % Whether to write in C order, typically false, needs to be false for frontend readable
  do_Corder = false;

  % Whether to use the size from Knossos conf instead of parsing the dirs.
  use_conf_size = true;

  % Whether to rerun the script for parsing to Knossos paths to find all populated raw files in the hypercube.
  reparse_dirs = false;

  % True to write hdf5 (false will only do a moch run)
  do_write = true;
  
  % Options for only writing a subset of the Knossos raw chunks to the hdf5 file.
  % This will result in chunk indices into the hdf5 file that start at (0,0,0) for the offset defined here.
  % Use false for writing the entire hypercube to the hdf5 file.
  do_chunk_select = false;
  assert( ~do_chunk_select || length(mags)==1 ); % xxx - can't do select with multiple mags right now
  % original 27 frontend "cubes"
  chunk_sel_offset = [8 9 3];
  nchunks_sel = [6 6 3];
  % if crop is on, dataset size is only the selected size, otherwise chunks written in context of whole dataset
  do_chunk_select_crop = false;
  
  % to add "context" cubes if selecting subset (over-ridden below if not)
  chunk_sel_offset = chunk_sel_offset - 1; nchunks_sel = nchunks_sel + 2;
  
  % Method for swapping the x and y directions, typically false (Knossos x direction corresponds to the first dimension)
  %dim_order = [2 1 3]; % to allow for "unnecessary" transpose, ONLY use this for k0725
  dim_order = [1 2 3]; % for all other datasets
  
  % name of the variable to write data into, convention here to write to root and specify without leading /
  data_name = sprintf('data_mag%d',mag);
  
  % END PARAMETERS

  tall = now;
  knossos_conf = parse_knossos_conf(inpath, knossos_conf_fn);
  rawtotal = prod(int64(rawsize));

  matname = [dataset '_' data_name '_dirs_info.mat'];
  if use_conf_size
    fprintf(1,'Using Knossos conf size %d,%d,%d\n',knossos_conf.boundary);
    %assert( all(mod(knossos_conf.boundary, rawsize)==0) );
    nchunks = ceil(double(knossos_conf.boundary) ./ rawsize);
    chunk_lists = true(nchunks);
    fprintf(1,'Ceil size size %d,%d,%d\n',nchunks.*chunksize);
  elseif reparse_dirs || ~exist(matname,'file')
    % parse the directories to get the total number of chunks, or load previously stored .mat file with parsed info
    [nchunks, chunk_lists] = parse_knossos_raw(inpath);
    save(matname,'nchunks','chunk_lists');
  else
    load(matname);
  end

  % this has to come after parsing the dirs, so we know the total number of chunks
  if ~do_chunk_select
    nchunks_sel = nchunks;
    chunk_sel_offset = [0 0 0];
    do_chunk_select_crop = true;
  end
  
  chunksizer = chunksize; chunksize = chunksize(dim_order);
  nchunks_selr = nchunks_sel; nchunks_sel = nchunks_sel(dim_order);
  if do_chunk_select_crop
    totalsize = chunksize.*nchunks_sel; 
  else
    totalsize = chunksize.*nchunks; 
  end    
  bytes_per_chunk = prod(chunksize);

  % put anything to be written to the atributes section for the data into this struct.
  % use double precision for floats and int32 for integers.
  data_conf = struct;
  data_conf.dataset = dataset;
  data_conf.rawsize = int32(rawsize);
  data_conf.isCorder = int32(do_Corder);
  data_conf.isSubset = int32(do_chunk_select);
  data_conf.chunkOffset = int32(chunk_sel_offset);
  data_conf.dimOrdering = int32(dim_order);
  data_conf.nchunks = int32(nchunks_sel);

  %   if do_Corder, str = 'Corder'; else str = 'Forder'; end
  %   if do_chunk_select
  %     hdffname = sprintf('%s_%dx%dx%dchunks_at%d-%d-%d_%s.h5',dataset,nchunks_selr,chunk_sel_offset,str);
  %   else
  %     hdffname = sprintf('%s_%dx%dx%dchunks_%s.h5',dataset,nchunks,str);
  %   end
  if do_chunk_select
    if do_chunk_select_crop    
      hdffname = sprintf('%s_%dx%dx%dchunks_at_x%04d_y%04d_z%04d_crop.h5',dataset,nchunks_selr,chunk_sel_offset);
    else
      hdffname = sprintf('%s_%dx%dx%dchunks_at_x%04d_y%04d_z%04d.h5',dataset,nchunks_selr,chunk_sel_offset);
    end
  else
    hdffname = [dataset '.h5'];
  end
  outfile = fullfile(outpath,hdffname);
  
  if do_write
    % xxx - be carefule with this, not sure how to delete an existing dataset
    evalc('delete(outfile)'); % silently remove existing file
    if do_Corder
      h5create(outfile,['/' data_name],totalsize([3 2 1]),'ChunkSize',chunksize([3 2 1]),'Datatype','uint8',...
        'Deflate',5,'Fletcher32',true,'Shuffle',true,'FillValue',uint8(0));
    else
      h5create(outfile,['/' data_name],totalsize,'ChunkSize',chunksize,'Datatype','uint8','Deflate',5,...
        'Fletcher32',true,'Shuffle',true,'FillValue',uint8(0));
    end
  end
  
  for ix1 = 1:nchunks(1)
    t = now;
    for ix2 = 1:nchunks(2)
      for ix3 = 1:nchunks(3)
        if chunk_lists(ix1,ix2,ix3)
          chunk_lists(ix1,ix2,ix3) = false; % set back to true below if this chunk is valid
          chunk_ind = [ix1 ix2 ix3]-1;
          chunk_ind_write = chunk_ind - chunk_sel_offset;
          if all(chunk_ind_write >= 0 & chunk_ind_write < nchunks_selr)
            if ~do_chunk_select_crop
              chunk_ind_write = chunk_ind_write + chunk_sel_offset;
            end
            if ix2==1 && ix3==1
              disp(['writing hdf mag' num2str(mag) ' data for all-yz chunks starting at chunk ',num2str(chunk_ind),...
                ' to all-yz chunks starting at chunk ', num2str(chunk_ind_write),' in ',hdffname]);
            end
            fn = fullfile(inpath,sprintf('x%04d',chunk_ind(1)),sprintf('y%04d',chunk_ind(2)),...
              sprintf('z%04d',chunk_ind(3)),sprintf('%s_x%04d_y%04d_z%04d.raw',raw_prefix,chunk_ind));
            if exist(fn,'file')
              fh = fopen(fn); V = fread(fh,bytes_per_chunk,'uint8=>uint8'); fclose(fh);
              if length(V) == rawtotal
                chunk_lists(ix1,ix2,ix3) = true; % valid chunk
                V = reshape(V,chunksizer);
                if do_write
                  write_hd5_chunk(permute(V,dim_order),outfile, data_name, chunk_ind_write(dim_order), ...
                    do_Corder, false); 
                end
              end % if raw file is correct size
            end % if file exists
          end % if chunk selected for write
        end % if in chunk list
      end % for z chunk
    end % for y chunk
    display(sprintf('\tdone in %.3f s',(now-t)*86400));
  end % for x chunk
  
  % write all the meta data, knossos conf and data conf
  if do_write
    write_struct_to_hd5attr(knossos_conf,outfile,data_name);
    write_struct_to_hd5attr(data_conf,outfile,data_name);
    
    % write out another dataset that contains a boolean mask of valid chunks (chunk_lists)
    cname = [data_name '_chunk_mask']; csize = size(chunk_lists); 
    cdata = uint8(permute(chunk_lists,dim_order)); 
    if do_Corder
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
end
