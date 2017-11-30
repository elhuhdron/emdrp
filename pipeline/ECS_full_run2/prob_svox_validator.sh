
# run on red or machine with mounted synology

# run for both datasets (depending on which machine running on, got lazy and not automatic)
#dataset=M0007
dataset=M0027

# chunks for M0007 MUST be in same order as in EM parser ini file
#declare -a chunks=("17 19 2" "17 23 1" "22 23 1" "22 18 1" "22 23 2" "19 22 2")
# chunks for M0027 MUST be in same order as in EM parser ini file
declare -a chunks=("18 15 3" "13 15 3" "13 20 3" "18 20 3" "18 20 4" "16 17 4")

outpath=/home/watkinspv/Downloads/tmp_probs
ctx_offset=32
declare -a sizes=('128 128 128' '128 128 128' '128 128 128' '128 128 128' '128 128 128' '128 128 128')
declare -a offsets=("0 0 0" "0 0 0" "0 0 0" "0 0 0" "0 0 0" "0 0 0")

inprobs=/mnt/syn2/watkinspv/full_datasets/neon_xfold/vgg3pool64_ECS_full_run2/${dataset}

# only for validation
count=0
for chunk in "${chunks[@]}";
do
    # create the subgroup name
    cchunk=($chunk)
    sg=`printf 'x%d_y%d_z%d' ${cchunk[0]} ${cchunk[1]} ${cchunk[2]}`

    # subtract context offset from offset
    coffset=(${offsets[$count]}); csize=(${sizes[$count]}); cnt=0
    while [ $cnt -lt 3 ]
    do
        coffset[$cnt]=`expr ${coffset[$cnt]} - $ctx_offset`
        csize[$cnt]=`expr ${csize[$cnt]} + 2 \* $ctx_offset`
        cnt=`expr $cnt + 1`
    done
    # convert back to space delimited strings
    ofst=$( IFS=' '; echo "${coffset[*]}" ); sz=$( IFS=' '; echo "${csize[*]}" )

    dpLoadh5.py --srcfile ${inprobs}_supervoxels.h5 --chunk $chunk --offset ${offsets[$count]} --size ${sizes[$count]} --dataset labels --subgroups with_background 0.99999000 --outraw $outpath/${dataset}_${sg}_supervoxels.nrrd --dpL 
    dpLoadh5.py --srcfile ${inprobs}_probs.h5 --chunk $chunk --size $sz --offset $ofst --dataset MEM --outraw $outpath/${dataset}_${sg}_MEM.nrrd --dpL --subgroups chunk_$sg

    count=`expr $count + 1`
done
