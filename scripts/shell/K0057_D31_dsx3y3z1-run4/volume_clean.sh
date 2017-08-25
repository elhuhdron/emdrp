
# run on biowulf in ~/gits/emdrp/recon/python
# script automatically submits jobs with dependencies since all cleaned outputs are written to same hdf5 file.
# NOTE: DO NOT use lscratch, as we want all the subgroups in the same hdf5 file.

OUTD=out_batches/20170727_K0057_run4/clean
no_fill_iter=0

count=0
iter=0
while [ $iter -le 24 ]
do
    agglo_iter=`printf '%.8f' $iter`
    echo processing $agglo_iter

    if [[ ${iter} -ge ${no_fill_iter} ]]; then
        # K0057 large agglo cleaning
        # NOTE: on next runs change minsize to 64 and change cavity-fill-minsize to 8192
        python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 8 8 8 --cube_size 6 6 6 --cmd "python -u $HOME/gits/emdrp/recon/python/dpCleanLabels.py --dataset labels --subgroups agglomeration $agglo_iter --get-svox-type --cavity-fill --minsize 27 --minsize_fill --cavity-fill-minsize 4096 --ECS-label 0 --dpWriteh5-verbose --dpCleanLabels-verbose" --fileflags srcfile outfile --filepaths /data/CDCU/full_datasets/neon/mfergus32_K0057_ds3_run4/agglo /data/CDCU/full_datasets/neon/mfergus32_K0057_ds3_run4/clean --fileprefixes K0057_D31_dsx3y3z1_supervoxels_agglo K0057_D31_dsx3y3z1_supervoxels_agglo_clean --filepostfixes .h5 .h5 > $OUTD/20170727_K0057_run4_cnt${count}.swarm
    else 
        # K0057 large agglo cleaning no minsize
        python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 8 8 8 --cube_size 6 6 6 --cmd "python -u $HOME/gits/emdrp/recon/python/dpCleanLabels.py --dataset labels --subgroups agglomeration $agglo_iter --get-svox-type --cavity-fill --ECS-label 0 --dpWriteh5-verbose --dpCleanLabels-verbose" --fileflags srcfile outfile --filepaths /data/CDCU/full_datasets/neon/mfergus32_K0057_ds3_run4/agglo /data/CDCU/full_datasets/neon/mfergus32_K0057_ds3_run4/clean --fileprefixes K0057_D31_dsx3y3z1_supervoxels_agglo K0057_D31_dsx3y3z1_supervoxels_agglo_clean --filepostfixes .h5 .h5 > $OUTD/20170727_K0057_run4_cnt${count}.swarm
    fi

    # writing to same h5 outputs so submit each one with dependency on the last instead of running swarm manually.
    # this assumes last output from swarm is the submitted jobid.
    cd $OUTD
    if [[ ${count} -eq 0 ]]; then
        last_job=`swarm -f 20170727_K0057_run4_cnt${count}.swarm -g 32 -t 16 --partition quick --verbose 1 | tail -1`
    else
        last_job=`swarm -f 20170727_K0057_run4_cnt${count}.swarm -g 32 -t 16 --partition quick --verbose 1 --sbatch " --dependency=afterany:${last_job} " | tail -1`
    fi
    # trim off any whitespace
    last_job=`echo ${last_job} | xargs`
    echo swarm submitted for $last_job
    cd ../../..

    count=`expr $count + 1`
    iter=`expr $iter + 2`
done

