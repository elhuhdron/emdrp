
# run on biowulf in ~/gits/emdrp/recon/python
# then start swarm in subdir out_batches/20171221_k0725_run1/agglo
# NOTE: no context on this run

OUTD=out_batches/20171221_k0725_run1/agglo

# due to space on biowulf, had to run half at a time
volume_range_beg='1 1 1'
volume_range_end='19 61 37'
#volume_range_beg='1 1 37'
#volume_range_end='19 61 73'

# k0725 large agglo
python -u dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 6 6 6 --cmd "python -u $HOME/gits/emdrp/recon/python/dpSupervoxelClassifier.py --cfgfile $HOME/gits/emdrp/pipeline/k0725_dsx2y2z1-run1/classifier_k0725_export.ini --dpSupervoxelClassifier-verbose --classifierin /lscratch/\$SLURM_JOBID/k0725_ds2_agglo_classifiers --classifierout '' --classifier rf --feature-set medium --neighbor-only --nthreads 16" --fileflags labelfile outfile probfile probaugfile --filepaths '/lscratch/$SLURM_JOBID' '/lscratch/$SLURM_JOBID' '/lscratch/$SLURM_JOBID' '/lscratch/$SLURM_JOBID' --fileprefixes k0725_dsx2y2z1_supervoxels k0725_dsx2y2z1_supervoxels_agglo k0725_dsx2y2z1_probs k0725_dsx2y2z1_probs --filepostfixes .h5 .h5 .h5 .h5 --pre-cmd 'cp -fpR /data/CDCU/agglo/k0725_ds2_agglo_classifiers /lscratch/$SLURM_JOBID' > $OUTD/20170902_k0725_run1_2.swarm

# copy probs to lscratch
python -u dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 6 6 6 --cmd "cp -fp" --fileflags 0 0 --filepaths /data/CDCU/full_datasets/neon/vgg3pool64_k0725_ds2_run1/probs '/lscratch/$SLURM_JOBID' --fileprefixes k0725_dsx2y2z1_probs k0725_dsx2y2z1_probs --filepostfixes .h5 .h5 --no_volume_flags > $OUTD/20170902_k0725_run1_0.swarm
# copy watershed to lscratch
python -u dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 6 6 6 --cmd "cp -fp" --fileflags 0 0 --filepaths /data/CDCU/full_datasets/neon/vgg3pool64_k0725_ds2_run1/wtsh '/lscratch/$SLURM_JOBID' --fileprefixes k0725_dsx2y2z1_supervoxels k0725_dsx2y2z1_supervoxels --filepostfixes .h5 .h5 --no_volume_flags > $OUTD/20170902_k0725_run1_1.swarm

# copy agglomerated supervoxels from lscratch to data
python -u dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 6 6 6 --cmd "cp -fp" --fileflags 0 0 --filepaths '/lscratch/$SLURM_JOBID' /data/CDCU/full_datasets/neon/vgg3pool64_k0725_ds2_run1/agglo --fileprefixes k0725_dsx2y2z1_supervoxels_agglo k0725_dsx2y2z1_supervoxels_agglo --filepostfixes .h5 .h5 --no_volume_flags > $OUTD/20170902_k0725_run1_3.swarm

# put into single command, ; delimited
paste -d';' $OUTD/20170902_k0725_run1_0.swarm $OUTD/20170902_k0725_run1_1.swarm $OUTD/20170902_k0725_run1_2.swarm $OUTD/20170902_k0725_run1_3.swarm > $OUTD/20170902_k0725_run1.swarm

# run swarm on biowulf with:
#swarm -f 20170902_k0725_run1.swarm -g 96 -t 16 -p 1 --sbatch " --gres=lscratch:100 " --time 72:00:00 --verbose 1

