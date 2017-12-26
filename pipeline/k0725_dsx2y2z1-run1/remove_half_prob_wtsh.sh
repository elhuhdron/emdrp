
# run on biowulf to remove first half of prob and wtsh for space

# due to space on biowulf, had to run half at a time
volume_range_beg='1 1 1'
volume_range_end='19 61 37'
#volume_range_beg='1 1 37'
#volume_range_end='19 61 73'

# remove half of prob and wtsh files
dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 6 6 6 --cmd "rm " --fileflags 0 --filepaths /data/CDCU/full_datasets/neon/vgg3pool64_k0725_ds2_run1/probs --fileprefixes k0725_dsx2y2z1_probs --filepostfixes .h5 --filepaths-affixes 0 --no_volume_flags > tmp_out_1.sh
dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 6 6 6 --cmd "rm " --fileflags 0 --filepaths /data/CDCU/full_datasets/neon/vgg3pool64_k0725_ds2_run1/wtsh --fileprefixes k0725_dsx2y2z1_supervoxels --filepostfixes .h5 --filepaths-affixes 0 --no_volume_flags > tmp_out_2.sh

# echo current remove to output file
dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 6 6 6 --cmd "echo " --fileflags 0 --filepaths 'removing from ' --fileprefixes '' --filepostfixes ' ' --filepaths-affixes 0 --no_volume_flags > tmp_out_0.sh

# put into single command, ; delimited
paste -d';' tmp_out_0.sh tmp_out_1.sh tmp_out_2.sh > tmp_out.sh

