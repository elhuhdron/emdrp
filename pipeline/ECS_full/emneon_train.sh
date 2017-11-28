
# run on gpu-clone in ~/gits/emdrp/neon3
# queues one set of jobs each for training on all GT cubes (1) and leave-one-out cross-fold training (num GT cubes)

declare -a card=("0" "2" "0" "2" "1" "3" "1" "3")
declare -a skip_lists=("0" "1" "2" "3" "4" "5")
declare -a machine_skip_inds=('0 5 4 3 1 5 3' '1 0 5' '2 1 0 4 2 0 4' '3 2 1 5 3 1 5' '4 3 2 0 4 2')
cnt=0

# run for both datasets
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

echo "nohup time python -u ./emneon.py -e 1 --data_config ~/gits/emdrp/pipeline/ECS_full/EMdata-3class-64x64out-rand-${dataset}.ini --serialize 800 -s /home/$(whoami)/Data/convnet_out/test-model.prm -o /home/$(whoami)/Data/convnet_out/test-output.h5 --model_arch vgg3pool --train_range 100001 112800 --epoch_dstep 5600 4000 2400 --nbebuf 1 -i ${card[$cnt]} >& /home/$(whoami)/Data/convnet_out/test-emneon-out.txt &" > run_tmp.sh
# just run on all 5 machines
#if [ ${#cmachine_skip_inds[@]} -lt 5 ]; then
    ./run_gpu_job.py -s run_tmp.sh
    #cat run_tmp.sh
    cnt=`expr $cnt + 1`
#fi

iter=0
while [ $iter -lt ${#cmachine_skip_inds[@]} ]
do

    echo "nohup time python -u ./emneon.py -e 1 --data_config ~/gits/emdrp/pipeline/ECS_full/EMdata-3class-64x64out-rand-${dataset}.ini --serialize 800 -s /home/$(whoami)/Data/convnet_out/test-model.prm -o /home/$(whoami)/Data/convnet_out/test-output.h5 --model_arch vgg3pool --train_range 100001 112800 --epoch_dstep 5600 4000 2400 --nbebuf 1 -i ${card[$cnt]} --chunk_skip_list ${skip_lists[${cmachine_skip_inds[$iter]}]} --test_range 200001 200001 --eval 800 >& /home/$(whoami)/Data/convnet_out/test-emneon-out.txt &" > run_tmp.sh
    ./run_gpu_job.py -s run_tmp.sh
    #cat run_tmp.sh

    iter=`expr $iter + 1`
    cnt=`expr $cnt + 1`
done
./run_gpu_job.py

