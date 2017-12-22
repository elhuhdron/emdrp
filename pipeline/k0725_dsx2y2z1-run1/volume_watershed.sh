
# run on biowulf in ~/gits/emdrp/recon/python
# then start swarm in subdir out_batches/20171221_k0725_run1/wtsh
# NOTE: no context on this run

OUTD=out_batches/20171221_k0725_run1/wtsh

# k0725 large watershed
python -u dpCubeIter.py --volume_range_beg 1 1 1 --volume_range_end 19 61 73 --overlap 0 0 0 --cube_size 6 6 6 --cmd "python -u $HOME/gits/emdrp/recon/python/dpWatershedTypes.py --ThrRng 0.5 0.999 0.1 --ThrHi 0.95 0.99 0.995 0.999 0.99925 0.9995 0.99975 0.9999 0.99995 0.99999 --fg-types ICS --dpW" --fileflags outlabels probfile --filepaths '/lscratch/$SLURM_JOBID' '/lscratch/$SLURM_JOBID' --fileprefixes k0725_dsx2y2z1_supervoxels k0725_dsx2y2z1_probs --filepostfixes .h5 .h5 > $OUTD/20170902_k0725_run1_1.swarm

# copy probs to lscratch
python -u dpCubeIter.py --volume_range_beg 1 1 1 --volume_range_end 19 61 73 --overlap 0 0 0 --cube_size 6 6 6 --cmd "cp -fp" --fileflags 0 0 --filepaths /data/CDCU/full_datasets/neon/vgg3pool64_k0725_ds2_run1/probs '/lscratch/$SLURM_JOBID' --fileprefixes k0725_dsx2y2z1_probs k0725_dsx2y2z1_probs --filepostfixes .h5 .h5 --no_volume_flags > $OUTD/20170902_k0725_run1_0.swarm

# copy supervoxels from lscratch to data
python -u dpCubeIter.py --volume_range_beg 1 1 1 --volume_range_end 19 61 73 --overlap 0 0 0 --cube_size 6 6 6 --cmd "cp -fp" --fileflags 0 0 --filepaths '/lscratch/$SLURM_JOBID' /data/CDCU/full_datasets/neon/vgg3pool64_k0725_ds2_run1/wtsh --fileprefixes k0725_dsx2y2z1_supervoxels k0725_dsx2y2z1_supervoxels --filepostfixes .h5 .h5 --no_volume_flags > $OUTD/20170902_k0725_run1_2.swarm

# put into single command, ; delimited
paste -d';' $OUTD/20170902_k0725_run1_0.swarm $OUTD/20170902_k0725_run1_1.swarm $OUTD/20170902_k0725_run1_2.swarm > $OUTD/20170902_k0725_run1.swarm

# run swarm on biowulf with:
#swarm -f 20170902_k0725_run1.swarm -g 48 -t 16 --time 36:00:00 --sbatch " --gres=lscratch:100 " --verbose 1

