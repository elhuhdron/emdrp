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

% compare read/write speeds of raw data versus hdf5

function o = perf_test_hdf5(p)

o = struct;
total_size = p.edge_size*ones(1,3);

% xxx - could expand to other types...
file_types = {'raw' 'hdf5'};

o.write_times = zeros(length(file_types),numel(p.cube_sizes),numel(p.compressions),numel(p.dtypes),p.nrepeats);
o.read_times = zeros(length(file_types),numel(p.cube_sizes),numel(p.compressions),numel(p.dtypes),p.nrepeats);

for d = 1:length(p.dtypes)
  meh = zeros(1,1,p.dtypes{d});
  if isinteger(meh)
    data = randi(intmax(p.dtypes{d}),total_size,'like',meh);
  else
    data = rand(total_size,p.dtypes{d});
  end
  
  for i = 1:length(p.cube_sizes)
    for c = 1:length(p.compressions)

      for f = 1:length(file_types)
        % xxx - matlab compression method with zip is not a fair comparison as it can't compress from memory
        if strcmp(file_types{f},'raw') && p.compressions(c) > 0, continue; end
        
        display(sprintf('write/read %s dtype=%s cube size=%d compression=%d for %d repeats',p.dtypes{d},...
          file_types{f},p.cube_sizes(i),p.compressions(c),p.nrepeats));
        tloop = now;
        for r = 1:p.nrepeats
          [~, t] = iterate_cubes(f-1, true, data, p.dtypes{d}, total_size, p.path, p.cube_sizes(i)*ones(1,3), ...
            p.compressions(c), p.chunk_size);
          o.write_times(f,i,c,d,r) = t;
          
          [odata, t] = iterate_cubes(f-1, false, [], p.dtypes{d}, total_size, p.path, p.cube_sizes(i)*ones(1,3), ...
            [], []);
          o.read_times(f,i,c,d,r) = t;
          
          assert( all(data(:) == odata(:)) );
        end
        display(sprintf('\tdone %.3f s',(now-tloop)*86400));
      end

    end % for each compression level
  end % for each cube size
end % for each data type

end % perf_test_hdf5



function [odata, t] = iterate_cubes(hdf5, write, data, dtype, total_size, path, cube_size, compression, chunk_size)

assert( all(mod(total_size, cube_size) == 0) );
ncubes = total_size ./ cube_size;

fill = zeros(1,1,dtype);
%dtype_size = whos('fill'); dtype_size = dtype_size.bytes;

if write
  odata = [];
else
  odata = zeros(total_size,dtype);
end

if hdf5
  fn_hdf5 = fullfile(path,'out.h5');
  plist = 'H5P_DEFAULT';
  if write
    evalc('delete(fn_hdf5)'); % silently remove existing file
    t = now; % count hdf5 creation and handle opening in time
    if compression > 0
      h5create(fn_hdf5,'/data',total_size,'ChunkSize',chunk_size,'Datatype',dtype,...
        'Deflate',compression,'Fletcher32',true,'Shuffle',true,'FillValue',fill);
    else
      h5create(fn_hdf5,'/data',total_size,'Datatype',dtype,'FillValue',fill);
    end
    fid_hdf5 = H5F.open(fn_hdf5,'H5F_ACC_RDWR',plist);
  else
    t = now; % count hdf5 creation and handle opening in time
    fid_hdf5 = H5F.open(fn_hdf5);
  end
  dset_id = H5D.open(fid_hdf5,'/data');

  % taken from matlab help for H5.write / H5.read
  % assume fliplr is for F -> C order, ignore for symmetric cube_sizes
  h5_block = cube_size;
  mem_space_id = H5S.create_simple(3,h5_block,[]);
  file_space_id = H5D.get_space(dset_id);  
else
  root_path = fullfile(path,'rawcubes');
  if write
    evalc('[stat, mess, id]=rmdir(root_path,''s'')'); % silently remove existing raw cubes
  else
    read_conv = sprintf('%s=>%s',dtype,dtype);
    elements_per_cube = prod(cube_size);
  end
  t = now;
end

for ix1 = 1:ncubes(1)
  for ix2 = 1:ncubes(2)
    for ix3 = 1:ncubes(3)
      cube_ind = [ix1 ix2 ix3]-1;
      ind = cube_ind.*cube_size;
      if write
        cdata = data(ind(1)+1:ind(1)+cube_size(1),ind(2)+1:ind(2)+cube_size(2),ind(3)+1:ind(3)+cube_size(3));
      end
      
      if hdf5
        H5S.select_hyperslab(file_space_id,'H5S_SELECT_SET',ind,[],[],h5_block);
        if write
          H5D.write(dset_id,'H5ML_DEFAULT',mem_space_id,file_space_id,plist,cdata);
        else
          cdata = H5D.read(dset_id,'H5ML_DEFAULT',mem_space_id,file_space_id,plist);
        end
      else % if hdf5
        fp = fullfile(root_path,sprintf('x%04d',cube_ind(1)),sprintf('y%04d',cube_ind(2)),...
          sprintf('z%04d',cube_ind(3)));
        fn = fullfile(fp,sprintf('%s_x%04d_y%04d_z%04d.raw','out',cube_ind));
        if write
          mkdir(fp);
          fh = fopen(fn, 'wb'); fwrite(fh,cdata,dtype); fclose(fh);
          %if compression > 0
          %  zip([fn '.zip'],fn)
          %end
        else
          %if compression > 0
          %  unzip([fn '.zip'],fp)
          %end
          fh = fopen(fn); cdata = reshape(fread(fh,elements_per_cube,read_conv),cube_size); fclose(fh);
        end
      end % else if hdf5

      if ~write
        odata(ind(1)+1:ind(1)+cube_size(1),ind(2)+1:ind(2)+cube_size(2),ind(3)+1:ind(3)+cube_size(3)) = cdata;
      end
      
    end % for z cube
  end % for y cube
end % for x cube

if hdf5
  H5F.close(fid_hdf5);    
end

t = (now - t)*86400;
end % iterate_cubes
