
# run on gpu-clones 3-4 in ~/gits/emdrp/neon3

declare -a card=("0" "1" "2" "3" "0" "1" "2" "3" "0" "1" "2" "3" "0" "1" "2" "3" "0" "1" "2" "3")
declare -a skip_lists=("0" "1")
declare -a slices=('xyz' 'xyz' 'xzy' 'zyx')
declare -a repeats=('0' '1' '0' '0')
# for both datasets on 4 machines
declare -a machine_skip_inds=('x' 'x' '0 0 0 0' '1 1 1 1' 'x')

dataset=k0725_dsx2y2z1

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
while [ $iter -lt ${#cmachine_skip_inds[@]} ]
do
    atestc=(${skip_lists[${cmachine_skip_inds[$iter]}]}); itestc=${atestc[0]}; ntestc=${#atestc[@]}
    testc=$( IFS=' '; echo "${atestc[*]}" )

    count=$iter
    echo "python -u ./emneon.py --data_config ~/gits/emdrp/pipeline/ECS_full_run2/EMdata-3class-64x64out-rand-ctx-${dataset}.ini --model_file /mnt/syn2/watkinspv/convnet_out/neon_xfold/vgg3pool64_k0725_ds2_run1/${dataset}_${slices[$count]}_test${itestc}_${repeats[$count]}.prm --write_output /mnt/syn2/watkinspv/full_datasets/neon_xfold/vgg3pool64_k0725_ds2_run1/${dataset}_${slices[$count]}_test${itestc}_${repeats[$count]}.h5 --test_range 200001 200001 --nbebuf 1 -i ${card[$totalc]} --chunk_skip_list $testc --dim_ordering ${slices[$count]}" > run_tmp.sh
    #cat run_tmp.sh
    ./run_gpu_job.py -s run_tmp.sh
       
    iter=`expr $iter + 1`
done
./run_gpu_job.py 

