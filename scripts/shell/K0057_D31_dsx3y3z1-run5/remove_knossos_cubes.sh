
# remove default seg files
python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 0 0 0 --cube_size 1 1 1 --cmd "rm " --fileflags 0 --filepaths /mnt/ext/K0057_D31/cubes_dsx3y3z1/mag1 --fileprefixes K0057_D31_mag1 --filepostfixes .seg.sz.zip --filepaths-affixes 1 --no_volume_flags > tmp_out_1.sh
python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 0 0 0 --cube_size 1 1 1 --cmd "rm " --fileflags 0 --filepaths /mnt/ext/K0057_D31/cubes_dsx3y3z1/mag1 --fileprefixes K0057_D31_mag1 --filepostfixes .ECS_probs.raw --filepaths-affixes 1 --no_volume_flags > tmp_out_2.sh
python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 0 0 0 --cube_size 1 1 1 --cmd "rm " --fileflags 0 --filepaths /mnt/ext/K0057_D31/cubes_dsx3y3z1/mag1 --fileprefixes K0057_D31_mag1 --filepostfixes .ICS_probs.raw --filepaths-affixes 1 --no_volume_flags > tmp_out_3.sh
python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 0 0 0 --cube_size 1 1 1 --cmd "mv " --fileflags 0 0 --filepaths /mnt/ext/K0057_D31/cubes_dsx3y3z1/mag1 /mnt/ext/K0057_D31/cubes_dsx3y3z1/mag1 --fileprefixes K0057_D31_mag1 K0057_D31_mag1 --filepostfixes .MEM_probs.raw .MEM_probsA.raw --filepaths-affixes 1 1 --no_volume_flags > tmp_out_4.sh

# echo current remove to output file
python -u dpCubeIter.py --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 0 0 0 --cube_size 1 1 1 --cmd "echo " --fileflags 0 --filepaths 'removing from ' --fileprefixes '' --filepostfixes ' ' --filepaths-affixes 0 --no_volume_flags > tmp_out_0.sh

# put into single command, ; delimited
paste -d';' tmp_out_0.sh tmp_out_1.sh tmp_out_2.sh tmp_out_3.sh tmp_out_4.sh > tmp_out.sh

