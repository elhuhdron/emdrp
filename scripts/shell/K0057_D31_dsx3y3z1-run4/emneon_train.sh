
# run on gpu-clone in ~/gits/emdrp/neon3
# queues one set of jobs each for training on all GT cubes (1) and leave-one-out cross-fold training (num GT cubes)

declare -a card=("0" "1" "2" "3")
declare -a skip_lists=("0 1 2 3" "4 5 6 7" "8 9 10 11" "12 13")
declare -a machine_skip_inds=('0 1 2' '0 2 3' '1 2 3' '0 1 3' '0 1 2 3')
cnt=0

# accidentally mixed up order for this run
declare -a lips=(1 2 65 129 193)

# map IP to machine index
machine=$(ifconfig eno1 | grep 'inet ' | perl -nle'/\s*inet \d+\.\d+\.\d+\.(\d+)/ && print $1')
machine=($machine)
for i in ${!lips[@]}; do
   if [[ ${lips[$i]} = ${machine} ]]; then
       machine=${i}
   fi
done
cmachine_skip_inds=(${machine_skip_inds[$machine]})

echo "nohup time python -u ./emneon.py -e 1 --data_config data/tmp_config/EMdata-3class-32x32out-rand-K0057-dsx3y3z1.ini --serialize 200 -s /home/$(whoami)/Data/convnet_out/test-model.prm -o /home/$(whoami)/Data/convnet_out/test-output.h5 --model_arch mfergus --train_range 100001 104200 --epoch_dstep 1800 1300 800 200 -i ${card[$cnt]} >& /home/$(whoami)/Data/convnet_out/test-emneon-out.txt &" > run_tmp.sh
if [ ${#cmachine_skip_inds[@]} -lt 4 ]; then
    ./run_gpu_job.py -s run_tmp.sh
    #cat run_tmp.sh
    cnt=`expr $cnt + 1`
fi

iter=0
while [ $iter -lt ${#cmachine_skip_inds[@]} ]
do

    echo "nohup time python -u ./emneon.py -e 1 --data_config data/tmp_config/EMdata-3class-32x32out-rand-K0057-dsx3y3z1.ini --eval 200 --serialize 200 -s /home/$(whoami)/Data/convnet_out/test-model.prm -o /home/$(whoami)/Data/convnet_out/test-output.h5 --model_arch mfergus --train_range 100001 104200 --epoch_dstep 1800 1300 800 200 --test_range 200001 200001 --chunk_skip_list ${skip_lists[${cmachine_skip_inds[$iter]}]} -i ${card[$cnt]} >& /home/$(whoami)/Data/convnet_out/test-emneon-out.txt &" > run_tmp.sh
    ./run_gpu_job.py -s run_tmp.sh
    #cat run_tmp.sh

    iter=`expr $iter + 1`
    cnt=`expr $cnt + 1`
done
./run_gpu_job.py

