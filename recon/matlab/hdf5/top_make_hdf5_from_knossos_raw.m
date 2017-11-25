
p = struct;

% Iterate over multiple magnifications and write to separate datasets in hdf5
%p.mags = [1 2 4 8 16];
%p.mags = 1;
p.mags = 16;

% Name of the dataset, stored in meta, used in filename
%p.dataset = 'M0027_11';
%p.dataset = 'k0725';
p.dataset = 'K0057_D31';

% Paths to root of Knossos raw data and path to where hdf5 should be written.
%p.inpath = sprintf('/mnt/cdcu/Common/ECS_paper/ECS_3d_analysis/M0027_11/cubes/M0027_11_mag%d',mag);
%p.outpath = '/Data/big_datasets';
%p.inpath = sprintf('/mnt/cdcu/common/110629_k0725/cubes/%s_mag%d',dataset,mag);
p.inpath = sprintf('/mnt/ext/K0057_D31/cubes/mag%d',mag);
p.outpath = '/Data_yello/watkinspv/Downloads';

% The raw size of the Knossos cubes
p.rawsize = [128 128 128];

% name of the Knossos configuration file with data to be added to hdf5 meta folder
p.knossos_conf_fn = 'knossos.conf';

% The prefix of the raw file names (Knossos cubes)
%p.raw_prefix = sprintf('M0027_11_mag%d',mag);
%p.raw_prefix = sprintf('110629_k0725_mag%d',mag);
p.raw_prefix = sprintf('K0057_D31_mag%d',mag);

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
p.do_chunk_select = false;

p.chunk_sel_offset = [12 14 2] - 1;
p.nchunks_sel = [8 8 4] + 2;
% if crop is on, dataset size is only the selected size, otherwise chunks written in context of whole dataset
p.do_chunk_select_crop = false;

% Method for swapping the x and y directions, typically false (Knossos x direction corresponds to the first dimension)
%p.dim_order = [2 1 3]; % to allow for "unnecessary" transpose, ONLY use this for k0725
p.dim_order = [1 2 3]; % for all other datasets

% name of the variable to write data into, convention here to write to root and specify without leading /
% data name will be appended with _mag%d for each mag written
p.data_name = 'data';

make_hdf5_from_knossos_raw(p);
