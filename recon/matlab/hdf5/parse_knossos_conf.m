
% Function to parse the Knossos config file for raw 3D EM data.
function conf = parse_knossos_conf(inpath, fn)

conf = struct;

% written as generic read for any data in conf file in format:
%   field name up to either ""-delimited string, integer, or float
%   semicolon to mark end of expression
% should return types appropriate to be written to hdf5 attributes.
fh = fopen(fullfile(inpath,fn));
tline = fgetl(fh); curstmt = '';
while ischar(tline)
  %if isempty(strtrim(tline)), continue; end
  curstmt = [curstmt tline];
  stmts = regexp(curstmt, ';', 'split');
  for i=1:(length(stmts)-1)
    stmt_flts = regexp(stmts{i},'(.*?)(\d+)\.(\d+)\s*$','tokens');
    stmt_ints = regexp(stmts{i},'(.*?)(\d+)\s*$','tokens');
    stmt_strs = regexp(stmts{i},'(.*?)"(.*)"\s*$','tokens');
    if ~isempty(stmt_strs)
      field_name = regexprep(strtrim(stmt_strs{1}{1}), '\W', '_');
      conf.(field_name) = stmt_strs{1}{2};
    elseif ~isempty(stmt_flts)
      % use double precision float type for floating point fields
      field_name = regexprep(strtrim(stmt_flts{1}{1}), '\W', '_');
      conf.(field_name) = str2double(stmt_flts{1}{2}) + str2double(stmt_flts{1}{3}) / 10^length(stmt_flts{1}{3});
    elseif ~isempty(stmt_ints)
      % use int32 type for integer fields
      field_name = regexprep(strtrim(stmt_ints{1}{1}), '\W', '_');
      conf.(field_name) = int32(str2double(stmt_ints{1}{2}));
    end
  end
  if length(stmts) > 1, curstmt = stmts{end}; end
  tline = fgetl(fh);
end
fclose(fh);

% some specific stuff, merge some xyz fields
merge_fields = {'boundary' 'scale'};
for i=1:length(merge_fields)
  xn = [merge_fields{i} '_x']; yn = [merge_fields{i} '_y']; zn = [merge_fields{i} '_z']; 
  if isfield(conf,xn) && isfield(conf,yn) && isfield(conf,zn)
    conf.(merge_fields{i}) = [conf.(xn) conf.(yn) conf.(zn)];
    conf = rmfield(conf,xn); conf = rmfield(conf,yn); conf = rmfield(conf,zn);
  end
end
