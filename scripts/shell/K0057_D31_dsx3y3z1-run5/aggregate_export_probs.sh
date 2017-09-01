
# run on gpu-clone 

# for this run, "prob pushes" were divided on two synolgies.
# 4 streams of gpu-clones 0-4 split half to syn and half to syn2.
# NOTE: could go higher streams but currently at limit for one stream per x-block (with separate top/bottom halves).

# this is just here for reference, should match emneon_export.sh
# "top" half 
#declare -a chunk_range_beg=(" 1  7  0" " 1 13  0" " 1 19  0" " 1 25  0" " 1 31  0")
#declare -a chunk_range_end=("27 15 20" "27 21 20" "27 27 20" "27 33 20" "27 39 20")
# "bottom" half
#declare -a chunk_range_beg=("25  7  0" "25 13  0" "25 19  0" "25 25  0" "25 31  0")
#declare -a chunk_range_end=("51 15 20" "51 21 20" "51 27 20" "51 33 20" "51 39 20")

# all yz, top and bottom halves are split in x only
declare -a chunk_range_beg_yz=(" 7  0" "13  0" "19  0" "25  0" "31  0")
declare -a chunk_range_end_yz=("15 20" "21 20" "27 20" "33 20" "39 20")

# two x-subranges "top" half
#declare -a chunk_range_beg_x=("1"  "13")
#declare -a chunk_range_end_x=("15" "27")
# two x-subranges "bottom" half
#declare -a chunk_range_beg_x=("25" "37")
#declare -a chunk_range_end_x=("39" "51")
#declare -a synout=("syn" "syn")

# four x-subranges "top" half
declare -a chunk_range_beg_x=("1" "7"  "13" "19")
declare -a chunk_range_end_x=("9" "15" "21" "27")
# four x-subranges "bottom" half
#declare -a chunk_range_beg_x=("25" "31" "37" "43")
#declare -a chunk_range_end_x=("33" "39" "45" "51")
declare -a synout=("syn" "syn" "syn2" "syn2")

# map IP to machine index
declare -a lips=(1 2 65 129 193)
machine=$(ifconfig eno1 | grep 'inet ' | perl -nle'/\s*inet \d+\.\d+\.\d+\.(\d+)/ && print $1')
machine=($machine)
for i in ${!lips[@]}; do
   if [[ ${lips[$i]} = ${machine} ]]; then
       machine=${i}
   fi
done

# filemodulators-overlap is an optimization that is only really supported currently by dpAggProbs in order to slice
#   out the perimeter volumes so they only use the overlap part. this prevents us from have to average a whole
#   knossos sized cube of context in every perimeter direction.
# NOTE: the volume_range_beg and _end should STILL include the overlap volumes.

count=0
for chunkx in "${chunk_range_beg_x[@]}";
do
    echo processing $chunkx

    python -u dpCubeIter.py --volume_range_beg $chunkx ${chunk_range_beg_yz[$machine]} --volume_range_end ${chunk_range_end_x[$count]} ${chunk_range_end_yz[$machine]} --overlap 0 0 0 --cube_size 1 1 1 --cmd "python -u $HOME/gits/emdrp/recon/python/dpAggProbs.py --srcfile /mnt/syn/datasets/raw/K0057_D31_dsx3y3z1.h5 --dataset data_mag_x3y3z1 --data-type-out float32 --nmerge 4 --agg-ops-types mean min 0 mean max 0 mean max --types ICS ECS MEM --dpW --dpA" --fileflags inrawpath outfile --filepaths /Data/watkinspv/full_datasets/neon/mfergus32_K0057_ds3_run5/cubes /mnt/${synout[$count]}/watkinspv/full_datasets/neon/mfergus32_K0057_ds3_run5/probs --fileprefixes '' K0057_D31_dsx3y3z1_probs --filepostfixes '' .h5 --filepaths-affixes 1 0 --filenames-suffixes 0 1 --filemodulators 1 1 1  6 6 6 --filemodulators-overlap 24 24 24 > tmp_out${count}.sh

    nohup sh tmp_out${count}.sh >& tmp_K0057_D31_dsx3y3z1-run5-agg_probs${count}.txt &

    count=`expr $count + 1`
done

