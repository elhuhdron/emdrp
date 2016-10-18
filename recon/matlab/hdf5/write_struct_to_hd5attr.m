
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

