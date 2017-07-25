
# run on gpu-clone in ~/gits/emdrp/neon3
#   this version just runs on one clone (still relatively fast)

declare -a card=("0" "1" "2" "3" "0" "1" "2" "3" "0" "1" "2" "3" "0" "1" "2" "3" "0" "1" "2" "3")
declare -a skip_lists=("0 1 2 3" "4 5 6 7" "8 9 10 11" "12 13")

iter=0
totalc=0
while [ $iter -lt ${#skip_lists[@]} ]
do
    atestc=(${skip_lists[$iter]}); itestc=${atestc[0]}; ntestc=${#atestc[@]}
    testc=$( IFS=' '; echo "${atestc[*]}" )

    icount=0
    count=$icount
    #while [ $count -lt `expr $icount + 1` ]
    while [ $count -lt 4 ]
    do
        echo "python -u ./emneon.py --data_config ~/gits/emdrp/scripts/shell/K0057_D31_dsx3y3z1-run4/EMdata-3class-32x32out-rand-ctx-K0057-dsx3y3z1.ini --model_file /mnt/syn2/watkinspv/convnet_out/neon_xfold/mfergus32_K0057_ds3_run4/K0057-dsx3y3z1_xyz_test${itestc}_${count}.prm --write_output /mnt/syn2/watkinspv/full_datasets/neon_xfold/mfergus32_K0057_ds3_run4/K0057-dsx3y3z1_xyz_${count}.h5 --image_in_size 128 --test_range 200001 20000$ntestc --nbebuf 1 -i ${card[$totalc]} --chunk_skip_list $testc" > run_tmp.sh
        #cat run_tmp.sh
        ./run_gpu_job.py -s run_tmp.sh
       
        count=`expr $count + 1`
        totalc=`expr $totalc + 1`
    done
    iter=`expr $iter + 1`
done
./run_gpu_job.py 

