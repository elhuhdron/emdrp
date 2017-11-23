
# run on biowulf in ~/gits/emdrp/recon/python
# then start swarm in subdir/out_batches/20170624_K0057_run3/wtsh

# xxx - for future runs, use multiple dpCubeIter to create copy commands to/from lscratch (see agglo script)
# K0057 large watershed
python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 24 24 24 --cube_size 6 6 6 --leave_edge --cmd "python -u $HOME/gits/emdrp/recon/python/dpWatershedTypes.py --ThrRng 0.5 0.999 0.1 --ThrHi 0.95 0.99 0.995 0.999 0.99925 0.9995 0.99975 0.9999 0.99995 0.99999 --dpW" --fileflags outlabels probfile --filepaths /data/CDCU/full_datasets/neon/mfergus32_K0057_ds3_run3/wtsh /data/CDCU/full_datasets/neon/mfergus32_K0057_ds3_run3/probs --fileprefixes K0057_D31_dsx3y3z1_supervoxels K0057_D31_dsx3y3z1_probs --filepostfixes .h5 .h5 > out_batches/20170624_K0057_run3/wtsh/20170624_K0057_run3.swarm

# run swarm on biowulf with:
#swarm -f 20170624_K0057_run3.swarm -g 48 -t 16 --time 28:00:00 --verbose 1

