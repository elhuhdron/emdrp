
# run on green in ~/gits/emdrp/recon/python

# xxx - for reference, move this to another script, copy raw data into K0057_D31_ds3/ knossos-style
#python -u dpCubeIter.py --volume_range_beg 0 0 0 --volume_range_end 53 45 20 --overlap 0 0 0 --cube_size 1 1 1 --cmd "python -u $HOME/gits/emdrp/recon/python/dpLoadh5.py --srcfile /mnt/syn/datasets/raw/K0057_D31_dsx3y3z1.h5 --dataset data_mag_x3y3z1 --dpL" --fileflags outraw --filepaths /mnt/cdcu/Common/K0057_D31/cubes_proc/dsx3y3z1/K0057_D31_mag1 --fileprefixes 'K0057_D31_mag1' --filepostfixes '.raw' --filepaths-affixes 1

# see aggregate probs scripts for original export locations
declare -a chunk_range_beg_x=("2" "14" "26" "38")
declare -a chunk_range_end_x=("14" "26" "38" "50")
declare -a synin=("syn" "syn2" "syn" "syn2")

#declare -a types=("ICS" "ECS" "MEM")
declare -a types=("MEM")
count=0
while [ $count -lt ${#types[@]} ]
do
    cnt=0
    while [ $cnt -lt ${#chunk_range_beg_x[@]} ]
    do
        echo processing ${types[$count]} xbeg ${chunk_range_beg_x[$cnt]}

        # copy probs for K0057_D31_ds3 knossos-style
        python -u dpCubeIter.py --volume_range_beg ${chunk_range_beg_x[$cnt]} 8 1 --volume_range_end ${chunk_range_end_x[$cnt]} 38 19 --overlap 0 0 0 --cube_size 1 1 1 --cmd "python -u $HOME/gits/emdrp/recon/python/dpLoadh5.py --dataset ${types[$count]} --dtypeGray uint8 --dpL" --fileflags outraw srcfile --filepaths /mnt/ext/K0057_D31/cubes_dsx3y3z1/mag1 /mnt/${synin[$cnt]}/watkinspv/full_datasets/neon/mfergus32_K0057_ds3_run5/probs --fileprefixes K0057_D31_mag1 K0057_D31_dsx3y3z1_probs --filepostfixes .${types[$count]}_probsB.raw .h5 --filepaths-affixes 1 0 --filemodulators 1 1 1  6 6 6 > tmp_out_${types[$count]}_${chunk_range_beg_x[$cnt]}_${synin[$cnt]}.sh
    
        nohup sh tmp_out_${types[$count]}_${chunk_range_beg_x[$cnt]}_${synin[$cnt]}.sh >& tmp_K0057_D31_dsx3y3z1-run5-cp-prob_${types[$count]}_${chunk_range_beg_x[$cnt]}_${synin[$cnt]}.txt &

        cnt=`expr $cnt + 1`
    done

    count=`expr $count + 1`
done

