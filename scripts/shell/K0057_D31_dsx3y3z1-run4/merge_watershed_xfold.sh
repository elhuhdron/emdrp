
# run on red in ~/gits/emdrp/recon/python
# NOTE: this is all done serially so that xfold probs and supervoxels can all be in same file.
#   this is only an issue if the training volumes plus context overlap with each other.
#   In this case the chunk-subgroup feature of the classifier can be used, and probs and supervoxels
#     are created in different datasets in the same hdf5.
# save output logs and run entire script background with:
#   nohup sh merge_watershed_xfold_K0057.sh >& K0057_D31_dsx3y3z1-run4-merge_watershed_xfold.txt &

ctx_offset=32
declare -a sizes=('256 256 128' '256 256 128' '256 256 128' '128 256 128')
declare -a chunks=("6 23 2" "16 19 15" "4 35 2" "4 11 14")
declare -a offsets=("0 0 32" "0 0 32" "96 96 96"  "96 64 112")

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

    time python -u dpMergeProbs.py --srcpath /mnt/syn2/watkinspv/full_datasets/neon_xfold/mfergus32_K0057_ds3_run4 --srcfiles K0057-dsx3y3z1_xyz_0.h5 K0057-dsx3y3z1_xyz_1.h5 K0057-dsx3y3z1_xyz_2.h5 K0057-dsx3y3z1_xyz_3.h5 --dim-orderings xyz xyz xyz xyz --outprobs /mnt/syn2/watkinspv/full_datasets/neon_xfold/mfergus32_K0057_ds3_run4/K0057-dsx3y3z1_probs.h5 --chunk $chunk --offset $ofst --size $sz --types ICS --ops mean min --dpM
    time python -u dpMergeProbs.py --srcpath /mnt/syn2/watkinspv/full_datasets/neon_xfold/mfergus32_K0057_ds3_run4 --srcfiles K0057-dsx3y3z1_xyz_0.h5 K0057-dsx3y3z1_xyz_1.h5 K0057-dsx3y3z1_xyz_2.h5 K0057-dsx3y3z1_xyz_3.h5 --dim-orderings xyz xyz xyz xyz --outprobs /mnt/syn2/watkinspv/full_datasets/neon_xfold/mfergus32_K0057_ds3_run4/K0057-dsx3y3z1_probs.h5 --chunk $chunk --offset $ofst --size $sz --types ECS MEM --ops mean max --dpM

    count=`expr $count + 1`
done

count=0
for chunk in "${chunks[@]}";
do
    echo processing $chunk
    context_offset
    
    time python -u dpWatershedTypes.py --probfile /mnt/syn2/watkinspv/full_datasets/neon_xfold/mfergus32_K0057_ds3_run4/K0057-dsx3y3z1_probs.h5 --chunk $chunk --offset $ofst --size $sz --outlabels /mnt/syn2/watkinspv/full_datasets/neon_xfold/mfergus32_K0057_ds3_run4/K0057-dsx3y3z1_supervoxels.h5 --ThrRng 0.5 0.999 0.1 --ThrHi 0.95 0.99 0.995 0.999 0.99925 0.9995 0.99975 0.9999 0.99995 0.99999 --dpW

    count=`expr $count + 1`
done

