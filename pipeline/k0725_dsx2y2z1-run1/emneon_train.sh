
# run on gpu-clone in ~/gits/emdrp/neon3
# queues one set of jobs each for training on all GT cubes (1) and leave-one-out cross-fold training (num GT cubes)

declare -a card=("0" "1" "2" "3" "0" "1" "2" "3")
declare -a skip_lists=("0" "1")
declare -a machine_skip_inds=('-1' '-1 -1 0 0 1' '-1 -1 0 0 1' '-1 -1 0 1 1 1' '-1 0 0 1 1')
declare -a machine_orderings=('xyz' 'xyz xzy xyz zyx xyz' 'xyz zyx xyz zyx xzy' 'xyz zyx xzy xyz xzy zyx' 'xzy xyz xzy xyz zyx')
cnt=0

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
machine=1
cmachine_skip_inds=(${machine_skip_inds[$machine]})
cmachine_orderings=(${machine_orderings[$machine]})

# use -1 in the skip ind list to mean train on all
iter=0
while [ ${cmachine_skip_inds[$iter]} -lt 0 ]
do
     
     echo "nohup time python -u ./emneon.py -e 1 --data_config ~/gits/emdrp/pipeline/k0725_dsx2y2z1-run1/EMdata-3class-64x64out-rand-${dataset}.ini --serialize 800 -s /home/$(whoami)/Data/convnet_out/test-model.prm -o /home/$(whoami)/Data/convnet_out/test-output.h5 --model_arch vgg3pool --train_range 100001 112800 --epoch_dstep 5600 4000 2400 --nbebuf 1 -i ${card[$cnt]} --dim_ordering ${cmachine_orderings[$iter]} >& /home/$(whoami)/Data/convnet_out/test-emneon-out.txt &" > run_tmp.sh
    #./run_gpu_job.py -s run_tmp.sh
    cat run_tmp.sh
    iter=`expr $iter + 1`
    cnt=`expr $cnt + 1`
done

while [ $iter -lt ${#cmachine_skip_inds[@]} ]
do

    echo "nohup time python -u ./emneon.py -e 1 --data_config ~/gits/emdrp/pipeline/k0725_dsx2y2z1-run1/EMdata-3class-64x64out-rand-${dataset}.ini --serialize 800 -s /home/$(whoami)/Data/convnet_out/test-model.prm -o /home/$(whoami)/Data/convnet_out/test-output.h5 --model_arch vgg3pool --train_range 100001 112800 --epoch_dstep 5600 4000 2400 --nbebuf 1 -i ${card[$cnt]} --dim_ordering ${cmachine_orderings[$iter]} --chunk_skip_list ${skip_lists[${cmachine_skip_inds[$iter]}]} --test_range 200001 200001 --eval 800 >& /home/$(whoami)/Data/convnet_out/test-emneon-out.txt &" > run_tmp.sh
    #./run_gpu_job.py -s run_tmp.sh
    cat run_tmp.sh

    iter=`expr $iter + 1`
    cnt=`expr $cnt + 1`
done

./run_gpu_job.py

