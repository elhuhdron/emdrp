
# run on gpu-clone in ~/gits/emdrp/neon3

declare -a card=("0" "1" "2" "3" "0" "1" "2" "3")
outdir=/Data/watkinspv/full_datasets/neon/mfergus32_K0057_ds3_run4/cubes
#--volume_range_beg 2 8 1 --volume_range_end 50 38 19
# exports have to be split up due to 1TB data drives on clones
# total size is 48x30x18 knossos cubes (without overlaps)
#   divided into 6x6x6 sized superchunks == 8x5x3 == 120 total superchunks
# "top" half
declare -a chunk_range_beg=(" 1, 7, 0" " 1,13, 0" " 1,19, 0" " 1,25, 0" " 1,31, 0")
declare -a chunk_range_end=("27,15,20" "27,21,20" "27,27,20" "27,33,20" "27,39,20")
# "bottom" half
#declare -a chunk_range_beg=("25, 7, 0" "25,13, 0" "25,19, 0" "25,25, 0" "25,31, 0")
#declare -a chunk_range_end=("51,15,20" "51,21,20" "51,27,20" "51,33,20" "51,39,20")

# map IP to machine index
declare -a lips=(1 2 65 129 193)
machine=$(ifconfig eno1 | grep 'inet ' | perl -nle'/\s*inet \d+\.\d+\.\d+\.(\d+)/ && print $1')
machine=($machine)
for i in ${!lips[@]}; do
   if [[ ${lips[$i]} = ${machine} ]]; then
       machine=${i}
   fi
done

#rm -rf $outdir # bad idea's coming
mkdir -p $outdir

iter=0
while [ $iter -lt 4 ]
do
    #printf '%.8f' $iter
    fn="$outdir/EMdata-3class-32x32out-export-K0057-dsx3y3z1-m$machine-i$iter.ini"
    echo machine $machine iter $iter card ${card[$iter]} $fn
    echo "python -u ./emneon.py --data_config $fn --model_file /mnt/syn2/watkinspv/convnet_out/neon/mfergus32_K0057_ds3_run4/K0057-dsx3y3z1_xyz_$iter.prm --write_output $outdir/knossos$iter.conf --image_in_size 128 --test_range 200001 204160 -i ${card[$iter]}" > run_tmp.sh

    cp -f ~/gits/emdrp/scripts/shell/K0057_D31_dsx3y3z1-run4/EMdata-3class-32x32out-export-K0057-dsx3y3z1.ini $fn
    sed -i -- "s/chunk_range_beg.*/chunk_range_beg = ${chunk_range_beg[$machine]}/g" $fn
    sed -i -- "s/chunk_range_end.*/chunk_range_end = ${chunk_range_end[$machine]}/g" $fn

    ./run_gpu_job.py -s run_tmp.sh
    #cat run_tmp.sh
    iter=`expr $iter + 1`
done
./run_gpu_job.py

