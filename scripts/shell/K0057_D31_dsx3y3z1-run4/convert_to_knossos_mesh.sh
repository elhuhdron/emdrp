
# run on blue in ~/gits/emdrp/recon/python

declare -a thrs=("0.99900000" "0.99925000" "0.99950000" "0.99975000")
level=0
while [ $level -lt 3 ]
do
    echo processing ${thrs[$level]}

    python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 0 0 0 --cube_size 6 6 6 --cmd "cp" --fileflags 0 0 --filepaths /mnt/syn2/watkinspv/full_datasets/neon/mfergus32_K0057_ds3_run4/mesh_wtsh /mnt/ext/K0057_D31/cubes_dsx3y3z1/K0057_D31_mag1 --fileprefixes K0057_D31_mag1 K0057_D31_mag1 --filepostfixes .clean_wtshA.$level.mesh.h5 .clean_wtshA.$level.mesh.h5 --filepaths-affixes 0 1 --no_volume_flags > tmp_out${level}_1.sh

    # echo current copy to output file
    python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 8 8 8 --cube_size 6 6 6 --cmd "echo " --fileflags 0 --filepaths 'copying to ' --fileprefixes '' --filepostfixes ' ' --filepaths-affixes 0 --no_volume_flags > tmp_out${level}_0.sh

    # put into single command, ; delimited
    paste -d';' tmp_out${level}_0.sh tmp_out${level}_1.sh > tmp_out${level}.sh

    nohup sh tmp_out${level}.sh >& tmp_K0057_D31_dsx3y3z1-run4-cp-mesh${level}.txt &

    level=`expr $level + 1`
done

