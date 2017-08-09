
# run on biowulf in ~/gits/emdrp/recon/python
# then start swarm in subdir/out_batches/20170727_K0057_run4/wtsh

OUTD=out_batches/20170727_K0057_run4/wtsh

# K0057 large watershed
python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 8 8 8 --cube_size 6 6 6 --leave_edge --cmd "python -u $HOME/gits/emdrp/recon/python/dpWatershedTypes.py --ThrRng 0.5 0.999 0.1 --ThrHi 0.95 0.99 0.995 0.999 0.99925 0.9995 0.99975 0.9999 0.99995 0.99999 --dpW" --fileflags outlabels probfile --filepaths '/lscratch/$SLURM_JOBID' '/lscratch/$SLURM_JOBID' --fileprefixes K0057_D31_dsx3y3z1_supervoxels K0057_D31_dsx3y3z1_probs --filepostfixes .h5 .h5 > $OUTD/20170727_K0057_run4_1.swarm

# copy watershed and probs to lscratch
python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 8 8 8 --cube_size 6 6 6 --cmd "cp -fp" --fileflags 0 0 --filepaths /data/CDCU/full_datasets/neon/mfergus32_K0057_ds3_run4/probs '/lscratch/$SLURM_JOBID' --fileprefixes K0057_D31_dsx3y3z1_probs K0057_D31_dsx3y3z1_probs --filepostfixes .h5 .h5 --no_volume_flags > $OUTD/20170727_K0057_run4_0.swarm

# copy supervoxels from lscratch to data
python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 8 8 8 --cube_size 6 6 6 --cmd "cp -fp" --fileflags 0 0 --filepaths '/lscratch/$SLURM_JOBID' /data/CDCU/full_datasets/neon/mfergus32_K0057_ds3_run4/wtsh --fileprefixes K0057_D31_dsx3y3z1_supervoxels K0057_D31_dsx3y3z1_supervoxels --filepostfixes .h5 .h5 --no_volume_flags > $OUTD/20170727_K0057_run4_2.swarm

# put into single command, ; delimited
paste -d';' $OUTD/20170727_K0057_run4_0.swarm $OUTD/20170727_K0057_run4_1.swarm $OUTD/20170727_K0057_run4_2.swarm > $OUTD/20170727_K0057_run4.swarm

# run swarm on biowulf with:
#swarm -f 20170727_K0057_run4.swarm -g 48 -t 16 --time 28:00:00 --sbatch " --gres=lscratch:100 " --verbose 1

