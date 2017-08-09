
# run on biowulf in ~/gits/emdrp/recon/python
# then start swarm in subdir/out_batches/20170727_K0057_run4/mesh_wtsh

OUTD=out_batches/20170727_K0057_run4/mesh_wtsh

declare -a thrs=("0.99900000" "0.99925000" "0.99950000" "0.99975000")
level=0
while [ $level -lt 3 ]
do
    echo processing ${thrs[$level]}

    # K0057 large mesh
    python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 0 0 0 --cube_size 6 6 6 --cmd "python -u $HOME/gits/emdrp/recon/python/dpLabelMesher.py --dataset labels --subgroups with_background ${thrs[$level]}  --dpLabelMesher-verbose --set-voxel-scale --dataset-root $level --reduce-frac 0.01 --smooth 7 7 5" --fileflags mesh-outfile srcfile --filepaths '/lscratch/$SLURM_JOBID' '/lscratch/$SLURM_JOBID' --fileprefixes K0057_D31_mag1 K0057_D31_dsx3y3z1_supervoxels_clean --filepostfixes .clean_wtshA.$level.mesh.h5 .h5 > $OUTD/20170727_K0057_run4_level${level}_1.swarm

    # copy watershed to lscratch
    python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 8 8 8 --cube_size 6 6 6 --cmd "cp -fp" --fileflags 0 0 --filepaths /data/CDCU/full_datasets/neon/mfergus32_K0057_ds3_run4/clean_wtsh '/lscratch/$SLURM_JOBID' --fileprefixes K0057_D31_dsx3y3z1_supervoxels_clean K0057_D31_dsx3y3z1_supervoxels_clean --filepostfixes .h5 .h5 --no_volume_flags > $OUTD/20170727_K0057_run4_level${level}_0.swarm

    # copy meshes from lscratch to data
    python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 8 8 8 --cube_size 6 6 6 --cmd "cp -fp" --fileflags 0 0 --filepaths '/lscratch/$SLURM_JOBID' /data/CDCU/full_datasets/neon/mfergus32_K0057_ds3_run4/mesh_wtsh --fileprefixes K0057_D31_mag1 K0057_D31_mag1 --filepostfixes .clean_wtshA.$level.mesh.h5 .clean_wtshA.$level.mesh.h5 --no_volume_flags > $OUTD/20170727_K0057_run4_level${level}_2.swarm

    # put into single command, ; delimited
    paste -d';' $OUTD/20170727_K0057_run4_level${level}_0.swarm $OUTD/20170727_K0057_run4_level${level}_1.swarm $OUTD/20170727_K0057_run4_level${level}_2.swarm > $OUTD/20170727_K0057_run4_level${level}.swarm

    level=`expr $level + 1`
done

# run swarm on biowulf with:
#swarm -f 20170727_K0057_run4_level0.swarm -g 32 -t 16 --time 24:00:00 --sbatch " --gres=lscratch:100 " --verbose 1

