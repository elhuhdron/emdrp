
% to export raw nrrd (using emdrp toolset):
%dpLoadh5.py --srcfile /mnt/cne/from_externals/K0057_D31/K0057_D31.h5 --chunk 90 78 15 --size 192 192 32 --offset 96 96 96 --dataset data_mag1 --outraw ~/Downloads/K0057_webknossos_test/K0057_D31_x90o96_y78o96_z15o96.nrrd --dpL

% script to read the webknossos formatted label volume and export as raw.
% one could export as gipl or nrrd also (for example, with toolboxes from matlab central).

% top level path to data files
dn = '/home/pwatkins/Downloads/K0057_webknossos_test';
%dn = '.';

% location of the nrrd raw data file
raw_fn = fullfile(dn, 'K0057_D31_x90o96_y78o96_z15o96.nrrd');

% relative paths of the stored wkw labels (top dir).
wkw_labels = {'data/1', 'data/1'};

% filename where to store the nrrd formatted label data
out_labels = fullfile(dn, 'K0057_D31_x90o96_y78o96_z15o96_labels');

% location of the labels within the dataset
labeled_coordinate = [11616, 10080, 2016];
labeled_size = [192, 192, 32];

% which label values to make consensuses for for each labeled volume.
% a workflow would typically only have one object labeled per user labeled volume.
consensus_label_values = [2,3,4];

fn = fullfile(dn, wkw_labels{1});
bbox = [(labeled_coordinate+1)' (labeled_coordinate+labeled_size)'];
data = wkwLoadRoi(fn, bbox);
data_size = size(data);
data_type = class(data);

% write out raw file
out_labels = [out_labels sprintf('_%d_%d_%d_%s.raw',data_size(1),data_size(2),data_size(3),data_type)];
fid=fopen(out_labels,'w+');
cnt=fwrite(fid,data,data_type);
fclose(fid);
