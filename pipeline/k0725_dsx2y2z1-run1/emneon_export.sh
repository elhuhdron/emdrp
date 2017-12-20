
# run on gpu-clone (only 1-5) in ~/gits/emdrp/neon3

declare -a card=("1" "2" "0" "3")
declare -a slices=('xyz' 'xyz' 'xzy' 'zyx')
declare -a repeats=('0' '1' '0' '0')
outdir=/Data/watkinspv/full_datasets/neon/vgg3pool64_k0725_ds2_run1/cubes
#--volume_range_beg 1 1 1 --volume_range_end 19 61 73
# exports have to be split up due to 1TB data drives on clones
# total size is 18x60x72 knossos cubes (without overlaps)
#   divided into 6x6x6 sized superchunks == 3x10x12 == 360 total superchunks
# METHOD 1 with 4 machines:
# divided by 1x2x4 for export where 2 is halves in y dim and 4 is across machines in z dim
#   == 18x30x18 (9720) chunks == 3x5x3 (45) superchunks per export per machine
# METHOD 2 with 5 machines:
# divided by 1x5x2 for export where 2 is halves in z dim and 5 is across machines in y dim
#   == 18x12x36 (7776) chunks == 3x2x6 (36) superchunks per export per machine

# NOTE: neglected to pad to knossos boundary when downsampling hdf5
#   Instead of re-doing the downsampling with padding, 
#     decided to just drop probability context for this run.
# WITH CONTEXT (fix nchunks):
## METHOD 1:
#nchunks=9720
## y "top" half 
#declare -a chunk_range_beg=("x" " 0, 0, 0" " 0, 0,18" " 0, 0,36" " 0, 0,54")
#declare -a chunk_range_end=("x" "20,32,20" "20,32,38" "20,32,56" "20,32,74")
## y "bottom" half 
#declare -a chunk_range_beg=("x" " 0,30, 0" " 0,30,18" " 0,30,36" " 0,30,54")
#declare -a chunk_range_end=("x" "20,62,20" "20,62,38" "20,62,56" "20,62,74")
## METHOD 2:
#nchunks=7776
## z "top" half 
#declare -a chunk_range_beg=(" 0, 0, 0" " 0,12, 0" " 0,24, 0" " 0,36, 0" " 0,48, 0")
#declare -a chunk_range_end=("20,14,38" "20,26,38" "20,38,38" "20,50,38" "20,62,38")
## z "bottom" half
#declare -a chunk_range_beg=(" 0, 0,36" " 0,12,36" " 0,24,36" " 0,36,36" " 0,48,36")
#declare -a chunk_range_end=("20,14,74" "20,26,74" "20,38,74" "20,50,74" "20,62,74")

# NO CONTEXT:
# METHOD 1:
nchunks=9720
# y "top" half 
declare -a chunk_range_beg=("x" " 1, 1, 1" " 1, 1,19" " 1, 1,37" " 1, 1,55")
declare -a chunk_range_end=("x" "19,31,19" "19,31,37" "19,31,55" "19,31,73")
# y "bottom" half 
declare -a chunk_range_beg=("x" " 1,31, 1" " 1,31,19" " 1,31,37" " 1,31,55")
declare -a chunk_range_end=("x" "19,61,19" "19,61,37" "19,61,55" "19,61,73")
# METHOD 2:
nchunks=7776
# z "top" half 
declare -a chunk_range_beg=(" 1, 1, 1" " 1,13, 1" " 1,25, 1" " 1,37, 1" " 1,49, 1")
declare -a chunk_range_end=("19,13,37" "19,25,37" "19,37,37" "19,49,37" "19,61,37")
# z "bottom" half
declare -a chunk_range_beg=(" 1, 1,37" " 1,13,37" " 1,25,37" " 1,37,37" " 1,49,37")
declare -a chunk_range_end=("19,13,73" "19,25,73" "19,37,73" "19,49,73" "19,61,73")



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
    fn="$outdir/EMdata-3class-64x64out-export-k0725_dsx2y2z1-m$machine-i$iter.ini"
    echo machine $machine iter $iter card ${card[$iter]} $fn
    echo "python -u ./emneon.py --data_config $fn --model_file /mnt/syn2/watkinspv/convnet_out/neon/vgg3pool64_k0725_ds2_run1/k0725_dsx2y2z1_${slices[$iter]}_${repeats[$iter]}.prm --write_output $outdir/knossos$iter.conf --test_range 200001 20${nchunks} -i ${card[$iter]}" > run_tmp.sh

    cp -f ~/gits/emdrp/pipeline/k0725_dsx2y2z1-run1/EMdata-3class-64x64out-export-k0725_dsx2y2z1.ini $fn
    sed -i -- "s/chunk_range_beg.*/chunk_range_beg = ${chunk_range_beg[$machine]}/g" $fn
    sed -i -- "s/chunk_range_end.*/chunk_range_end = ${chunk_range_end[$machine]}/g" $fn

    ./run_gpu_job.py -s run_tmp.sh
    #cat run_tmp.sh
    iter=`expr $iter + 1`
done
./run_gpu_job.py

