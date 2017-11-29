
# run on gpu-clone in ~/gits/emdrp/neon3
#   this version just runs on one clone (still relatively fast)

declare -a card=("0" "1" "2" "3" "0" "1" "2" "3" "0" "1" "2" "3" "0" "1" "2" "3" "0" "1" "2" "3")
declare -a skip_lists=("0" "1" "2" "3" "4" "5")
declare -a slices=('xyz' 'xyz' 'xyz' 'xyz')

# run for both datasets
dataset=M0007
#dataset=M0027

iter=0
totalc=0
while [ $iter -lt ${#skip_lists[@]} ]
do
    atestc=(${skip_lists[$iter]}); itestc=${atestc[0]}; ntestc=${#atestc[@]}
    testc=$( IFS=' '; echo "${atestc[*]}" )

    icount=0
    count=$icount
    while [ $count -lt ${#slices[@]} ]
    do
        echo "python -u ./emneon.py --data_config ~/gits/emdrp/pipeline/ECS_full_run2/EMdata-3class-64x64out-export-${dataset}.ini --model_file /mnt/syn2/watkinspv/convnet_out/neon_xfold/vgg3pool64_ECS_full_run2/${dataset}_${slices[$count]}_test${itestc}_${count}.prm --write_output /mnt/syn2/watkinspv/full_datasets/neon_xfold/vgg3pool64_ECS_full_run2/${dataset}_${slices[$count]}_test${itestc}_${count}.h5 --test_range 200001 200256 --nbebuf 1 -i ${card[$totalc]} --chunk_skip_list $testc" > run_tmp.sh
        #cat run_tmp.sh
        ./run_gpu_job.py -s run_tmp.sh
       
        count=`expr $count + 1`
        totalc=`expr $totalc + 1`
    done
    iter=`expr $iter + 1`
done
./run_gpu_job.py 

