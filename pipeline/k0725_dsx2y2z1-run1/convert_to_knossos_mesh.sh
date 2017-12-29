
# run on green or red

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

    dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 6 6 6 --cmd "cp " --fileflags 0 0 --filepaths /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_k0725_ds2_run1/mesh /mnt/ext/110629_k0725/cubes_dsx2y2z1/mag1 --fileprefixes k0725_dsx2y2z1 k0725_dsx2y2z1 --filepostfixes .cleanA.$level.mesh.h5 .cleanB.$level.mesh.h5 --filepaths-affixes 0 1 --no_volume_flags > tmp_out${level}_1.sh

    # echo current copy to output file
    dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 6 6 6 --cmd "echo " --fileflags 0 --filepaths 'copying to ' --fileprefixes '' --filepostfixes ' ' --filepaths-affixes 0 --no_volume_flags > tmp_out${level}_0.sh

    # put into single command, ; delimited
    paste -d';' tmp_out${level}_0.sh tmp_out${level}_1.sh > tmp_out${level}.sh

    #nohup sh tmp_out${level}.sh >& tmp_k0725_ds2_run1-cp-mesh${level}.txt &

    level=`expr $level + 1`
done

