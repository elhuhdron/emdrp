
# run on biowulf in ~/gits/emdrp/recon/python
# then start swarm in subdir/out_batches/20170727_K0057_run4/clean

# K0057 large write voxel type (needed before agglo clean, tmp_K0057_clean.sh)
# because it's all IO, did not see a need for lscratch in this case
python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 8 8 8 --cube_size 6 6 6 --cmd "python -u $HOME/gits/emdrp/recon/python/dpWriteh5.py --dataset voxel_type --dpW --dpL" --fileflags srcfile outfile --filepaths /data/CDCU/full_datasets/neon/mfergus32_K0057_ds3_run4/wtsh /data/CDCU/full_datasets/neon/mfergus32_K0057_ds3_run4/agglo --fileprefixes K0057_D31_dsx3y3z1_supervoxels K0057_D31_dsx3y3z1_supervoxels_agglo --filepostfixes .h5 .h5 > out_batches/20170727_K0057_run4/clean/20170727_K0057_run4_write.swarm

# run swarm on biowulf with:
#swarm -f 20170727_K0057_run4_write.swarm -t 1 -p 2 --partition quick --verbose 1

