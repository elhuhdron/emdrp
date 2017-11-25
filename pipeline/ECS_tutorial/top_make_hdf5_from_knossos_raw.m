
p = struct;
datasets = {'M0027_11' 'M0007_33'};
chunk_sels = {[16 17 0] []};

for i=1:length(datasets)
  
  % Iterate over multiple magnifications and write to separate datasets in hdf5
  p.mag = 1;
  
  % Name of the dataset, stored in meta, used in filename
  p.dataset = datasets{i};
  
  % Paths to root of Knossos raw data and path to where hdf5 should be written.
  p.inpath = sprintf('~/Data/ECS_tutorial/%s', datasets{i});
  p.outpath = '~/Data/ECS_tutorial';
  
  % The raw size of the Knossos cubes
  p.rawsize = [128 128 128];
  
  % name of the Knossos configuration file with data to be added to hdf5 meta folder
  p.knossos_conf_fn = 'Knossos.conf';
  
  % The prefix of the raw file names (Knossos cubes)
  p.raw_prefix = sprintf('%s_mag%d',p.dataset,p.mag);
  
  % Chunksize written to hdf5 file, typically same as the Knossos raw size
  p.chunksize = rawsize;
  
  % Whether to write in C order, typically false, needs to be false for frontend readable
  p.do_Corder = false;
  
  % Whether to use the size from Knossos conf instead of parsing the dirs.
  p.use_conf_size = true;
  
  % Whether dataset size needs to be scaled by mag or not
  p.scale_conf_size = true;
  
  % Whether to rerun the script for parsing to Knossos paths to find all populated raw files in the hypercube.
  p.reparse_dirs = false;
  
  % True to write hdf5 (false will only do a moch run)
  p.do_write = true;
  
  % Options for only writing a subset of the Knossos raw chunks to the hdf5 file.
  % This will result in chunk indices into the hdf5 file that start at (0,0,0) for the offset defined here.
  % Use false for writing the entire hypercube to the hdf5 file.
  p.do_chunk_select = true;
  
  p.chunk_sel_offset = chunk_sels{i};
  p.nchunks_sel = [8 8 4];
  % if crop is on, dataset size is only the selected size, otherwise chunks written in context of whole dataset
  p.do_chunk_select_crop = false;
  
  % Method for swapping the x and y directions, typically false (Knossos x direction corresponds to the first dimension)
  p.dim_order = [1 2 3]; % for all other datasets
  
  % name of the variable to write data into, convention here to write to root and specify without leading /
  % data name will be appended with _mag%d for each mag written
  p.data_name = sprintf('data_mag%d',p.mag);
  
  make_hdf5_from_knossos_raw(p);
  
end
