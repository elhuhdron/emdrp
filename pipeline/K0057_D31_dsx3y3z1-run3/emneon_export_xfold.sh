
# run on gpu-clone in ~/gits/emdrp/neon3

declare -a card=("0" "1" "2" "3" "1" "3")
icount=0
count=$icount
while [ $count -lt `expr $icount + 1` ]
do
    iter=0
    while [ $iter -lt 6 ]
    do
        echo "python -u ./emneon.py --data_config data/config/EMdata-3class-32x32out-rand-ctx-K0057-dsx3y3z1.ini --model_file /Data/watkinspv/convnet_out/neon_sixfold/mfergus32_K0057_ds3_run3/K0057-dsx3y3z1_xyz_test${iter}_${count}.prm --write_output /Data/watkinspv/full_datasets/neon_sixfold/mfergus32_K0057_ds3_run3/K0057-dsx3y3z1_xyz_test${iter}_${count}.h5 --image_in_size 128 --test_range 200001 200001 --nbebuf 1 -i ${card[$iter]} --chunk_skip_list $iter" > run_tmp.sh
        #cat run_tmp.sh
        ./run_gpu_job.py -s run_tmp.sh
       
        iter=`expr $iter + 1`
    done

    count=`expr $count + 1`
done
./run_gpu_job.py 

