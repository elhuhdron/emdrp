
# run on gpu-clone in ~/gits/emdrp/neon3

declare -a card=("0" "1" "2" "3" "0" "1" "2" "3" "0" "1" "2" "3" "0" "1" "2" "3" "0" "1" "2" "3")
declare -a skip_lists=("0" "1" "2" "3" "4" "5")
declare -a slices=('xyz' 'xyz' 'xyz' 'xyz')
# for both datasets on 4 machines
declare -a machine_skip_inds=('x' '0 1 2' '3 4 5' '0 1 2' '3 4 5')

# run for both datasets (depending on which machine running on, got lazy and not automatic)
dataset=M0007
#dataset=M0027

# map IP to machine index
declare -a lips=(1 2 65 129 193)
machine=$(ifconfig eno1 | grep 'inet ' | perl -nle'/\s*inet \d+\.\d+\.\d+\.(\d+)/ && print $1')
machine=($machine)
for i in ${!lips[@]}; do
   if [[ ${lips[$i]} = ${machine} ]]; then
       machine=${i}
   fi
done
cmachine_skip_inds=(${machine_skip_inds[$machine]})

iter=0
totalc=0
while [ $iter -lt ${#cmachine_skip_inds[@]} ]
do
    atestc=(${skip_lists[${cmachine_skip_inds[$iter]}]}); itestc=${atestc[0]}; ntestc=${#atestc[@]}
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

