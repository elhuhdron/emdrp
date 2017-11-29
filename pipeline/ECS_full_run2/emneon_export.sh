
# run on single gpu-clone for each dataset in ~/gits/emdrp/neon3

declare -a card=("0" "1" "2" "3" "0" "1" "2" "3" "0" "1" "2" "3" "0" "1" "2" "3" "0" "1" "2" "3")
declare -a slices=('xyz' 'xyz' 'xyz' 'xyz')

# run for both datasets (depending on which machine running on, got lazy and not automatic)
dataset=M0007
#dataset=M0027

totalc=0
icount=0
count=$icount
while [ $count -lt ${#slices[@]} ]
do
    echo "python -u ./emneon.py --data_config ~/gits/emdrp/pipeline/ECS_full_run2/EMdata-3class-64x64out-export-ctx-${dataset}.ini --model_file /mnt/syn2/watkinspv/convnet_out/neon/vgg3pool64_ECS_full_run2/${dataset}_${slices[$count]}_${count}.prm --write_output /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/${dataset}_${slices[$count]}_${count}.h5 --test_range 200001 200400 --nbebuf 1 -i ${card[$totalc]}" > run_tmp.sh
    #cat run_tmp.sh
    ./run_gpu_job.py -s run_tmp.sh
       
    count=`expr $count + 1`
    totalc=`expr $totalc + 1`
done
./run_gpu_job.py 

