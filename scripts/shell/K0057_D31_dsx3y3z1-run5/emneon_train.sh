
# run on gpu-clone in ~/gits/emdrp/neon3
# queues one set of jobs each for training on all GT cubes (1) and leave-one-out cross-fold training (num GT cubes)

#declare -a sizes=('256 256 128' '256 256 128' '256 256 128' '128 256 128' '256 256 32' '256 256 32' '256 256 32')
#declare -a chunks=("6 23 2" "16 19 15" "4 35 2" "4 11 14" "24 14 8" "13 18 15" "10 11 18")
#chunk_range_beg     = 6,23,2, 6,23,2, 6,23,2, 6,23,2, 6,23,2, 6,23,2, 6,23,2, 6,23,2, 16,19,15, 16,19,15, 16,19,15, 16,19,15, 16,19,15, 16,19,15, 16,19,15, 16,19,15, 4,35,2, 4,35,2, 4,35,2, 4,35,2, 4,35,2, 4,35,2, 4,35,2, 4,35,2, 4,11,14, 4,11,14, 4,11,14, 4,11,14, 24,14,8, 24,14,8, 13,18,15, 13,18,15, 10,11,18, 10,11,18

declare -a card=("1" "2" "3" "0" "1" "3" "0")
declare -a skip_lists=("0 1 2 3 4 5 6 7" "8 9 10 11 12 13 14 15" "16 17 18 19 20 21 22 23" "24 25 26 27" "28 29" "30 31" "32 33")
declare -a machine_skip_inds=('0 5 3 1 6' '1 6 4 2 0' '2 0 5 3 1 4' '3 1 6 4 2 5' '4 2 0 5 3 6')
cnt=0

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

echo "nohup time python -u ./emneon.py -e 1 --data_config ~/gits/emdrp/scripts/shell/K0057_D31_dsx3y3z1-run5/EMdata-3class-32x32out-rand-K0057-dsx3y3z1.ini --serialize 100 -s /home/$(whoami)/Data/convnet_out/test-model.prm -o /home/$(whoami)/Data/convnet_out/test-output.h5 --model_arch mfergus --train_range 100001 107800 --epoch_dstep 3400 2400 1400 400 -i ${card[$cnt]} >& /home/$(whoami)/Data/convnet_out/test-emneon-out.txt &" > run_tmp.sh
# just run on all 5 machines
#if [ ${#cmachine_skip_inds[@]} -lt 5 ]; then
    ./run_gpu_job.py -s run_tmp.sh
    #cat run_tmp.sh
    cnt=`expr $cnt + 1`
#fi

iter=0
while [ $iter -lt ${#cmachine_skip_inds[@]} ]
do

    echo "nohup time python -u ./emneon.py -e 1 --data_config ~/gits/emdrp/scripts/shell/K0057_D31_dsx3y3z1-run5/EMdata-3class-32x32out-rand-K0057-dsx3y3z1.ini --eval 100 --serialize 100 -s /home/$(whoami)/Data/convnet_out/test-model.prm -o /home/$(whoami)/Data/convnet_out/test-output.h5 --model_arch mfergus --train_range 100001 107800 --epoch_dstep 3400 2400 1400 400 --test_range 200001 200001 --chunk_skip_list ${skip_lists[${cmachine_skip_inds[$iter]}]} -i ${card[$cnt]} >& /home/$(whoami)/Data/convnet_out/test-emneon-out.txt &" > run_tmp.sh
    ./run_gpu_job.py -s run_tmp.sh
    #cat run_tmp.sh

    iter=`expr $iter + 1`
    cnt=`expr $cnt + 1`
done
./run_gpu_job.py

