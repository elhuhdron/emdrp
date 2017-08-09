
# run on blue in ~/gits/emdrp/recon/python

declare -a thrs=("0.99900000" "0.99925000" "0.99950000" "0.99975000")
level=0
while [ $level -lt 4 ]
do
    echo processing ${thrs[$level]}

    python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 0 0 0 --cube_size 1 1 1 --cmd "python -u $HOME/gits/emdrp/recon/python/dpLoadh5.py --dataset labels --subgroups with_background ${thrs[$level]} --data-type uint64 --raw-compression --dpL" --fileflags outraw srcfile --filepaths "'/run/media/watkinspv/My Passport/K0057_D31/cubes_dsx3y3z1/K0057_D31_mag1'" /Data_yello/watkinspv/full_datasets/neon/mfergus32_K0057_ds3_run4/clean_wtsh --fileprefixes K0057_D31_mag1 K0057_D31_dsx3y3z1_supervoxels_clean --filepostfixes .clean_wtshA.$level.seg .h5 --filepaths-affixes 1 0 --filemodulators 1 1 1  6 6 6 > tmp_out${level}.sh

    nohup sh tmp_out${level}.sh >& tmp_K0057_D31_dsx3y3z1-run4-cp-seg${level}.txt &

    level=`expr $level + 1`
done

