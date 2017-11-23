
# run on gpu-clone in ~/gits/emdrp/neon3
# queues one set of jobs each for training on all GT cubes (1) and leave-one-out cross-fold training (num GT cubes)

declare -a card=("1" "0" "1" "2" "3" "0" "3")
cnt=0

echo "nohup time python -u ./emneon.py -e 1 --data_config data/config/EMdata-3class-32x32out-rand-K0057-dsx3y3z1.ini --image_in_size 128 --serialize 200 -s /home/$(whoami)/Data/convnet_out/test-model.prm -o /home/$(whoami)/Data/convnet_out/test-output.h5 --model_arch mfergus --train_range 100001 103200 --epoch_dstep 1400 1000 600 -i ${card[$cnt]} >& /home/$(whoami)/Data/convnet_out/test-emneon-out.txt &" > run_tmp.sh
./run_gpu_job.py -s run_tmp.sh
#cat run_tmp.sh

iter=0
while [ $iter -lt 6 ]
do
    cnt=`expr $cnt + 1`

    echo "nohup time python -u ./emneon.py -e 1 --data_config data/config/EMdata-3class-32x32out-rand-K0057-dsx3y3z1.ini --image_in_size 128 --eval 200 --serialize 200 -s /home/$(whoami)/Data/convnet_out/test-model.prm -o /home/$(whoami)/Data/convnet_out/test-output.h5 --model_arch mfergus --train_range 100001 103200 --epoch_dstep 1400 1000 600 --test_range 200001 200001 --chunk_skip_list $iter -i ${card[$cnt]} >& /home/$(whoami)/Data/convnet_out/test-emneon-out.txt &" > run_tmp.sh
    ./run_gpu_job.py -s run_tmp.sh
    #cat run_tmp.sh

    iter=`expr $iter + 1`
done
./run_gpu_job.py

