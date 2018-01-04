
# run on blue in ~/gits/emdrp/recon/python

# due to space on biowulf, had to run half at a time
volume_range_beg='1 1 1'
#volume_range_end='19 61 37'
#volume_range_beg='1 1 37'
volume_range_end='19 61 73'

#declare -a thrs=(0 6 14 30 74) # wrong!!
declare -a thrs=(74 30 14 6 0)
level=0
while [ $level -lt ${#thrs[@]} ]
do
    thr=`LANG=C printf '%.8f' ${thrs[$level]}`
    echo processing $thr

    dpCubeIter.py --volume_range_beg $volume_range_beg --volume_range_end $volume_range_end --overlap 0 0 0 --cube_size 1 1 1 --cmd "python -u $HOME/gits/emdrp/recon/python/dpLoadh5.py --dataset labels --subgroups agglomeration ${thr} --data-type uint64 --raw-compression --dpL" --fileflags outraw srcfile --filepaths /mnt/ext2/110629_k0725/cubes_dsx2y2z1/mag1 /Data_yello/watkinspv/full_datasets/neon/vgg3pool64_k0725_ds2_run1/clean --fileprefixes k0725_mag1 k0725_dsx2y2z1_supervoxels_agglo_clean --filepostfixes .cleanA.$level.seg .h5 --filepaths-affixes 1 0 --filemodulators 1 1 1  6 6 6 > tmp_out${level}.sh

    nohup sh tmp_out${level}.sh >& tmp_k0725_ds2-cp-seg${level}.txt &

    level=`expr $level + 1`
done

