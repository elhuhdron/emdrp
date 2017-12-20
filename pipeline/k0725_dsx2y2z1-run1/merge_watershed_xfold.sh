
# run on red in ~/gits/emdrp/recon/python
# NOTE: this is all done serially so that xfold probs and supervoxels can all be in same file.
#   this is only an issue if the training volumes plus context overlap with each other.
#   In this case the chunk-subgroup feature of the classifier can be used, and probs and supervoxels
#     are created in different datasets in the same hdf5.
#   For the ECS datasets they do overlap, only export with context for the probs (not supervoxels).
# save output logs and run entire script background with:
#   nohup sh run_merge_watershed_xfold.sh >& tmp_M0007_merge_watershed_xfold.txt &

ctx_offset=32
declare -a sizes=('128 128 128' '128 128 128')
# chunks for MUST be in same order as in EM parser ini file
declare -a chunks=("5 5 3" "5 6 3")
declare -a offsets=("0 64 0" "0 64 0")

dataset=k0725_dsx2y2z1

# export probs with context
count=0
for chunk in "${chunks[@]}";
do
    echo processing $chunk

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

    # create the subgroup name
    cchunk=($chunk)
    fn=`printf 'x%04d_y%04d_z%04d' ${cchunk[0]} ${cchunk[1]} ${cchunk[2]}`

    time python -u dpMergeProbs.py --srcpath /mnt/syn2/watkinspv/full_datasets/neon_xfold/vgg3pool64_k0725_ds2_run1 --srcfiles ${dataset}_xyz_test${count}_0.h5 ${dataset}_xyz_test${count}_1.h5 ${dataset}_xzy_test${count}_0.h5 ${dataset}_zyx_test${count}_0.h5 --dim-orderings xyz xyz xzy zyx --outprobs /mnt/syn2/watkinspv/full_datasets/neon_xfold/vgg3pool64_k0725_ds2_run1/${dataset}_probs.h5 --chunk $chunk --offset $ofst --size $sz --types ICS --ops mean min --dpM --subgroups-out chunk_$fn
    time python -u dpMergeProbs.py --srcpath /mnt/syn2/watkinspv/full_datasets/neon_xfold/vgg3pool64_k0725_ds2_run1 --srcfiles ${dataset}_xyz_test${count}_0.h5 ${dataset}_xyz_test${count}_1.h5 ${dataset}_xzy_test${count}_0.h5 ${dataset}_zyx_test${count}_0.h5 --dim-orderings xyz xyz xzy zyx --outprobs /mnt/syn2/watkinspv/full_datasets/neon_xfold/vgg3pool64_k0725_ds2_run1/${dataset}_probs.h5 --chunk $chunk --offset $ofst --size $sz --types OUT --ops mean max --dpM --subgroups-out chunk_$fn

    count=`expr $count + 1`
done

# do not watershed with context
count=0
for chunk in "${chunks[@]}";
do
    echo processing $chunk

    # create the subgroup name
    cchunk=($chunk)
    fn=`printf 'x%04d_y%04d_z%04d' ${cchunk[0]} ${cchunk[1]} ${cchunk[2]}`
   
    time python -u dpWatershedTypes.py --probfile /mnt/syn2/watkinspv/full_datasets/neon_xfold/vgg3pool64_k0725_ds2_run1/${dataset}_probs.h5 --chunk $chunk --offset ${offsets[$count]} --size ${sizes[$count]} --outlabels /mnt/syn2/watkinspv/full_datasets/neon_xfold/vgg3pool64_k0725_ds2_run1/${dataset}_supervoxels.h5 --ThrRng 0.5 0.999 0.1 --ThrHi 0.95 0.99 0.995 0.999 0.99925 0.9995 0.99975 0.9999 0.99995 0.99999 --dpW --subgroups chunk_$fn --fg-types ICS --bg-type OUT

    count=`expr $count + 1`
done

