
# NOTE: use with caution, hacked up script for moving stuff
# only needed this because mistakenly assigned wrong ordering from agglomeration iteration to segmentation levels.

declare -a volume_range_begs=('1 1 1' '1 1 19' '1 1 37' '1 1 55')
declare -a volume_range_ends=('19 61 19' '19 61 37' '19 61 55' '19 61 73')
declare -a thrs=(0 6 14 30 74)

iter=0
while [ $iter -lt ${#volume_range_begs[@]} ]
do
    volume_range_beg=${volume_range_begs[$iter]}
    volume_range_end=${volume_range_ends[$iter]}
    rm -rf tmp_out${iter}.sh

    level=0
    level_out=`expr ${#thrs[@]} - 1`
    while [ $level -lt ${#thrs[@]} ]
    do
        thr=`LANG=C printf '%.8f' ${thrs[$level]}`
        echo processing $thr $iter

        dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 6 6 6 --cmd "mv " --fileflags 0 0 --filepaths /mnt/ext2/110629_k0725/cubes_dsx2y2z1/mag1 /mnt/ext2/110629_k0725/cubes_dsx2y2z1/mag1 --fileprefixes k0725_mag1 k0725_mag1 --filepostfixes .cleanB.$level.mesh.h5 .cleanA.$level_out.mesh.h5 --filepaths-affixes 1 1 --no_volume_flags > tmp_out_1.sh
        # NOTE: this is mad slow, decided not worth it, just re-export
        #dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 1 1 1 --cmd "mv " --fileflags 0 0 --filepaths /mnt/ext2/110629_k0725/cubes_dsx2y2z1/mag1 /mnt/ext2/110629_k0725/cubes_dsx2y2z1/mag1 --fileprefixes k0725_mag1 k0725_mag1 --filepostfixes .cleanB.$level.seg.sz.zip .cleanA.$level_out.seg.sz.zip --filepaths-affixes 1 1 --no_volume_flags > tmp_out_1.sh
        # this was to fix the dataset_root being the wrong seg-level
        #dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 6 6 6 --cmd "python -u volume_rename_mesh_groups.py $level $level_out " --fileflags 0 --filepaths /mnt/ext2/110629_k0725/cubes_dsx2y2z1/mag1 --fileprefixes k0725_mag1 --filepostfixes .cleanA.$level_out.mesh.h5 --filepaths-affixes 1 --no_volume_flags > tmp_out_1.sh

        # echo current copy to output file
        dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 6 6 6 --cmd "echo " --fileflags 0 --filepaths 'moving ' --fileprefixes '' --filepostfixes ' ' --filepaths-affixes 0 --no_volume_flags > tmp_out_0.sh
        #dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 6 6 6 --cmd "echo " --fileflags 0 --filepaths 'renaming ' --fileprefixes '' --filepostfixes ' ' --filepaths-affixes 0 --no_volume_flags > tmp_out_0.sh

        # put into single command, ; delimited
        paste -d';' tmp_out_0.sh tmp_out_1.sh >> tmp_out${iter}.sh
        #paste -d';' tmp_out_0.sh tmp_out_1.sh tmp_out_2.sh >> tmp_out${iter}.sh

        level=`expr $level + 1`
        level_out=`expr $level_out - 1`
    done

    #nohup sh tmp_out${iter}.sh >& tmp_k0725_ds2_run1-cp-mesh${iter}.txt &
    iter=`expr $iter + 1`
done

