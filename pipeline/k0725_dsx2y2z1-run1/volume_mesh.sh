
# run on biowulf in ~/gits/emdrp/recon/python
# then start swarm in subdir out_batches/20171221_k0725_run1/mesh
# NOTE: copied the thresholds to the segmentation levels backwards, fix on next run.

OUTD=out_batches/20171221_k0725_run1/mesh

# due to space on biowulf, had to run half at a time
volume_range_beg='1 1 1'
volume_range_end='19 61 37'
#volume_range_beg='1 1 37'
#volume_range_end='19 61 73'

declare -a thrs=(0 6 14 30 74)
level=0
while [ $level -lt ${#thrs[@]} ]
do
    thr=`LANG=C printf '%.8f' ${thrs[$level]}`
    echo processing $thr

    python -u dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 6 6 6 --cmd "python -u $HOME/gits/emdrp/recon/python/dpLabelMesher.py --dataset labels --subgroups agglomeration ${thr}  --dpLabelMesher-verbose --set-voxel-scale --dataset-root $level --reduce-frac 0.02 --smooth 5 5 5" --fileflags mesh-outfile srcfile --filepaths '/lscratch/$SLURM_JOBID' '/lscratch/$SLURM_JOBID' --fileprefixes k0725_dsx2y2z1 k0725_dsx2y2z1_supervoxels_agglo_clean --filepostfixes .cleanA.$level.mesh.h5 .h5 > $OUTD/20170902_k0725_run1_level${level}_1.swarm

    # copy input supervoxels to lscratch
    python -u dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 6 6 6 --cmd "cp -fp" --fileflags 0 0 --filepaths /data/CDCU/full_datasets/neon/vgg3pool64_k0725_ds2_run1/clean '/lscratch/$SLURM_JOBID' --fileprefixes k0725_dsx2y2z1_supervoxels_agglo_clean k0725_dsx2y2z1_supervoxels_agglo_clean --filepostfixes .h5 .h5 --no_volume_flags > $OUTD/20170902_k0725_run1_level${level}_0.swarm

    # copy meshes from lscratch to data
    python -u dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 6 6 6 --cmd "cp -fp" --fileflags 0 0 --filepaths '/lscratch/$SLURM_JOBID' /data/CDCU/full_datasets/neon/vgg3pool64_k0725_ds2_run1/mesh --fileprefixes k0725_dsx2y2z1 k0725_dsx2y2z1 --filepostfixes .cleanA.$level.mesh.h5 .cleanA.$level.mesh.h5 --no_volume_flags > $OUTD/20170902_k0725_run1_level${level}_2.swarm

    # put into single command, ; delimited
    paste -d';' $OUTD/20170902_k0725_run1_level${level}_0.swarm $OUTD/20170902_k0725_run1_level${level}_1.swarm $OUTD/20170902_k0725_run1_level${level}_2.swarm > $OUTD/20170902_k0725_run1_level${level}.swarm

    level=`expr $level + 1`
done

# run swarm on biowulf with:
#swarm -f 20170902_k0725_run1_level0.swarm -g 32 -t 16 --time 4:00:00 --partition quick --sbatch " --gres=lscratch:100 " --verbose 1

