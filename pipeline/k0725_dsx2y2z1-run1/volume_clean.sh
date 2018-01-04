
# run on biowulf in ~/gits/emdrp/recon/python
# script automatically submits jobs with dependencies since all cleaned outputs are written to same hdf5 file.
# NOTE: DO NOT use lscratch, as we want all the subgroups in the same hdf5 file.
# NOTE: copied the thresholds to the segmentation levels backwards, fix on next run.

OUTD=out_batches/20171221_k0725_run1/clean
declare -a thrs=(0 6 14 30 74)

# due to space on biowulf, had to run half at a time
volume_range_beg='1 1 1'
volume_range_end='19 61 37'
#volume_range_beg='1 1 37'
#volume_range_end='19 61 73'

count=0
iter=0
while [ $iter -lt ${#thrs[@]} ]
do
    thr=`LANG=C printf '%.8f' ${thrs[$iter]}`
    echo processing $thr

    python -u dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 6 6 6 --cmd "python -u $HOME/gits/emdrp/recon/python/dpCleanLabels.py --dataset labels --subgroups agglomeration $thr --get-svox-type --cavity-fill --minsize 28 --minsize_fill --cavity-fill-minsize 13824 --ECS-label 0 --dpWriteh5-verbose --dpCleanLabels-verbose" --fileflags srcfile outfile --filepaths /data/CDCU/full_datasets/neon/vgg3pool64_k0725_ds2_run1/agglo /data/CDCU/full_datasets/neon/vgg3pool64_k0725_ds2_run1/clean --fileprefixes k0725_dsx2y2z1_supervoxels_agglo k0725_dsx2y2z1_supervoxels_agglo_clean --filepostfixes .h5 .h5 > $OUTD/20171221_k0725_run1_cnt${count}.swarm

    # writing to same h5 outputs so submit each one with dependency on the last instead of running swarm manually.
    # this assumes last output from swarm is the submitted jobid.
    cd $OUTD
    if [[ ${count} -eq 0 ]]; then
        last_job=`swarm -f 20171221_k0725_run1_cnt${count}.swarm -g 32 -t 16 --partition quick --verbose 1 | tail -1`
    else
        last_job=`swarm -f 20171221_k0725_run1_cnt${count}.swarm -g 32 -t 16 --partition quick --verbose 1 --sbatch " --dependency=afterany:${last_job} " | tail -1`
    fi
    # trim off any whitespace
    last_job=`echo ${last_job} | xargs`
    echo swarm submitted for $last_job
    cd ../../..

    count=`expr $count + 1`
    iter=`expr $iter + 1`
done

