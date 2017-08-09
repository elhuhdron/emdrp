
# run on biowulf in ~/gits/emdrp/recon/python
# then start swarm in subdir/out_batches/20170727_K0057_run4/agglo

OUTD=out_batches/20170727_K0057_run4/agglo

# K0057 large agglo
python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 8 8 8 --cube_size 6 6 6 --cmd "python -u $HOME/gits/emdrp/recon/python/dpSupervoxelClassifier.py --cfgfile $HOME/gits/emdrp/scripts/shell/K0057_D31_dsx3y3z1-run4/svox_K0057_ds3_iterate_export_parallel.ini --dpSupervoxelClassifier-verbose --classifierin /lscratch/\$SLURM_JOBID/K0057_D31_dsx3y3z1-run4-classifier --classifierout '' --classifier rf --feature-set medium --neighbor-only --no-agglo-ECS --nthreads 16" --fileflags labelfile outfile probfile probaugfile --filepaths '/lscratch/$SLURM_JOBID' '/lscratch/$SLURM_JOBID' '/lscratch/$SLURM_JOBID' '/lscratch/$SLURM_JOBID' --fileprefixes K0057_D31_dsx3y3z1_supervoxels K0057_D31_dsx3y3z1_supervoxels_agglo K0057_D31_dsx3y3z1_probs K0057_D31_dsx3y3z1_probs --filepostfixes .h5 .h5 .h5 .h5 --pre-cmd 'cp -fpR /data/CDCU/agglo/K0057_D31_dsx3y3z1-run4-classifier /lscratch/$SLURM_JOBID' > $OUTD/20170727_K0057_run4_2.swarm

# copy watershed and probs to lscratch
python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 8 8 8 --cube_size 6 6 6 --cmd "cp -fp" --fileflags 0 0 --filepaths /data/CDCU/full_datasets/neon/mfergus32_K0057_ds3_run4/probs '/lscratch/$SLURM_JOBID' --fileprefixes K0057_D31_dsx3y3z1_probs K0057_D31_dsx3y3z1_probs --filepostfixes .h5 .h5 --no_volume_flags > $OUTD/20170727_K0057_run4_0.swarm
python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 8 8 8 --cube_size 6 6 6 --cmd "cp -fp" --fileflags 0 0 --filepaths /data/CDCU/full_datasets/neon/mfergus32_K0057_ds3_run4/wtsh '/lscratch/$SLURM_JOBID' --fileprefixes K0057_D31_dsx3y3z1_supervoxels K0057_D31_dsx3y3z1_supervoxels --filepostfixes .h5 .h5 --no_volume_flags > $OUTD/20170727_K0057_run4_1.swarm

# copy agglomerated supervoxels from lscratch to data
python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 8 8 8 --cube_size 6 6 6 --cmd "cp -fp" --fileflags 0 0 --filepaths '/lscratch/$SLURM_JOBID' /data/CDCU/full_datasets/neon/mfergus32_K0057_ds3_run4/agglo --fileprefixes K0057_D31_dsx3y3z1_supervoxels_agglo K0057_D31_dsx3y3z1_supervoxels_agglo --filepostfixes .h5 .h5 --no_volume_flags > $OUTD/20170727_K0057_run4_3.swarm

# put into single command, ; delimited
paste -d';' $OUTD/20170727_K0057_run4_0.swarm $OUTD/20170727_K0057_run4_1.swarm $OUTD/20170727_K0057_run4_2.swarm $OUTD/20170727_K0057_run4_3.swarm > $OUTD/20170727_K0057_run4.swarm

# run swarm on biowulf with:
#swarm -f 20170727_K0057_run4.swarm -g 64 -t 16 -p 1 --sbatch " --gres=lscratch:100 " --time 28:00:00 --verbose 1
#swarm -f 20170727_K0057_run4.swarm -g 64 -t 16 -p 1 --sbatch " --gres=lscratch:100 " --time 96:00:00 --verbose 1

