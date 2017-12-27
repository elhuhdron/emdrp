
# run on biowulf in ~/gits/emdrp/recon/python
# then start swarm in subdir out_batches/20171221_k0725_run1/clean

# due to space on biowulf, had to run half at a time
volume_range_beg='1 1 1'
volume_range_end='19 61 37'
#volume_range_beg='1 1 37'
#volume_range_end='19 61 73'

# copy voxel type from watershed to agglomeration hdfs, needed for cleaning step
# because it's all IO, did not see a need for lscratch in this case
python -u dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 6 6 6 --cmd "python -u $HOME/gits/emdrp/recon/python/dpWriteh5.py --dataset voxel_type --dpW --dpL" --fileflags srcfile outfile --filepaths /data/CDCU/full_datasets/neon/vgg3pool64_k0725_ds2_run1/wtsh /data/CDCU/full_datasets/neon/vgg3pool64_k0725_ds2_run1/agglo --fileprefixes k0725_dsx2y2z1_supervoxels k0725_dsx2y2z1_supervoxels_agglo --filepostfixes .h5 .h5 > out_batches/20171221_k0725_run1/clean/20171221_k0725_run1_write.swarm

# run swarm on biowulf with:
#swarm -f 20171221_k0725_run1_write.swarm -t 1 -p 2 --partition quick --verbose 1

