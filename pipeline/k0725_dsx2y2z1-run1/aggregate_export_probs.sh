
# run on gpu-clone 

# for this run, "prob pushes" were divided on two synolgies.
# 5 streams of gpu-clones 2-5 split half to syn and half to syn2.
# NOTE: could go higher streams but currently at limit for one stream per x-block (with separate top/bottom halves).

# this is just here for reference, should match emneon_export.sh
# NO CONTEXT:
# METHOD 1:
#nchunks=9720
# y "top" half 
#declare -a chunk_range_beg=("x" " 1, 1, 1" " 1, 1,19" " 1, 1,37" " 1, 1,55")
#declare -a chunk_range_end=("x" "19,31,19" "19,31,37" "19,31,55" "19,31,73")
# y "bottom" half 
#declare -a chunk_range_beg=("x" " 1,31, 1" " 1,31,19" " 1,31,37" " 1,31,55")
#declare -a chunk_range_end=("x" "19,61,19" "19,61,37" "19,61,55" "19,61,73")

# all xz, top and bottom halves are split in y only
declare -a chunk_range_beg_x=("x" " 1" " 1" " 1" " 1")
declare -a chunk_range_end_x=("x" "19" "19" "19" "19")
declare -a chunk_range_beg_z=("x" " 1" "19" "37" "55")
declare -a chunk_range_end_z=("x" "19" "37" "55" "73")

# five y-subranges 
# y "top" half
declare -a chunk_range_beg_y=(" 1" " 7" "13" "19" "25")
declare -a chunk_range_end_y=(" 7" "13" "19" "25" "31")
# y "bottom" half
#declare -a chunk_range_beg_y=("31" "37" "43" "49" "55")
#declare -a chunk_range_end_y=("37" "43" "49" "55" "61")

declare -a synout=("syn" "syn2" "syn" "syn2" "syn2")

# map IP to machine index
declare -a lips=(1 2 65 129 193)
machine=$(ifconfig eno1 | grep 'inet ' | perl -nle'/\s*inet \d+\.\d+\.\d+\.(\d+)/ && print $1')
machine=($machine)
for i in ${!lips[@]}; do
   if [[ ${lips[$i]} = ${machine} ]]; then
       machine=${i}
   fi
done

# NOTE: no overlap used in this run due to resampling rounding to knossos-cube-size issue (see data_resample.sh)

count=0
for chunky in "${chunk_range_beg_y[@]}";
do
    echo processing $chunky

    # NOTE: ordering here has to match parallel card and slices array in emneon_export.sh
    dpCubeIter.py --volume_range_beg ${chunk_range_beg_x[$machine]} $chunky ${chunk_range_beg_z[$machine]} --volume_range_end ${chunk_range_end_x[$machine]} ${chunk_range_end_y[$count]} ${chunk_range_end_z[$machine]} --overlap 0 0 0 --cube_size 1 1 1 --cmd "python -u $HOME/gits/emdrp/recon/python/dpAggProbs.py --srcfile /mnt/syn/datasets/raw/k0725_dsx2y2z1.h5 --dataset data_mag_x2y2z1 --data-type-out float32 --nmerge 4 --agg-ops-types mean min 0 mean max --types ICS MEM --dim-orderings xzy xyz xyz zyx --dpW --dpA" --fileflags inrawpath outfile --filepaths /Data/watkinspv/full_datasets/neon/vgg3pool64_k0725_ds2_run1/cubes /mnt/${synout[$count]}/watkinspv/full_datasets/neon/vgg3pool64_k0725_ds2_run1/probs --fileprefixes '' k0725_dsx2y2z1_probs --filepostfixes '' .h5 --filepaths-affixes 1 0 --filenames-suffixes 0 1 --filemodulators 1 1 1  6 6 6 > tmp_out${count}.sh

    nohup sh tmp_out${count}.sh >& tmp_k0725_dsx2y2z1-agg_probs${count}.txt &

    count=`expr $count + 1`
done

