
# run on red in ~/gits/emdrp/recon/python
# also outputs the probs and supervoxels for validation
# save output logs and run entire script background with:
#   nohup sh run_merge_watershed.sh >& tmp_merge_watershed.txt &

declare -a datasets=("M0007" "M0027")
declare -a chunks=("16 17 0" "12 14 2")
declare -a offsets=("-32 -32 0" "-32 -32 -32")
declare -a sizes=('1088 1088 544' '1088 1088 576')

outpath=/home/watkinspv/Downloads/tmp_probs

# probs with context
count=0
for chunk in "${chunks[@]}";
do
    echo processing ${datasets[$count]} $chunk
    time python -u dpMergeProbs.py --srcpath /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2 --srcfiles ${datasets[$count]}_xyz_0.h5 ${datasets[$count]}_xyz_1.h5 ${datasets[$count]}_xyz_2.h5 ${datasets[$count]}_xyz_3.h5 --dim-orderings xyz xyz xyz xyz --outprobs /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/${datasets[$count]}_probs.h5 --chunk $chunk --offset ${offsets[$count]} --size ${sizes[$count]} --types ICS --ops mean min --dpM
    time python -u dpMergeProbs.py --srcpath /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2 --srcfiles ${datasets[$count]}_xyz_0.h5 ${datasets[$count]}_xyz_1.h5 ${datasets[$count]}_xyz_2.h5 ${datasets[$count]}_xyz_3.h5 --dim-orderings xyz xyz xyz xyz --outprobs /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/${datasets[$count]}_probs.h5 --chunk $chunk --offset ${offsets[$count]} --size ${sizes[$count]} --types ECS MEM --ops mean max --dpM

    dpLoadh5.py --srcfile /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/${datasets[$count]}_probs.h5 --chunk $chunk --offset ${offsets[$count]} --size ${sizes[$count]} --dataset MEM --outraw $outpath/${datasets[$count]}_probs_MEM.nrrd --dpL 

    count=`expr $count + 1`
done

# supervoxels without context
count=0
for chunk in "${chunks[@]}";
do
    time python -u dpWatershedTypes.py --probfile /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/${datasets[$count]}_probs.h5 --chunk $chunk --offset 0 0 32 --size 1024 1024 480 --outlabels /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/${datasets[$count]}_supervoxels.h5 --ThrRng 0.5 0.999 0.1 --ThrHi 0.95 0.99 0.995 0.999 0.99925 0.9995 0.99975 0.9999 0.99995 0.99999 --dpW

    dpLoadh5.py --srcfile /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/${datasets[$count]}_supervoxels.h5 --chunk $chunk --offset 0 0 32 --size 1024 1024 480 --dataset labels --subgroups with_background 0.99999000 --outraw $outpath/${datasets[$count]}_supervoxels.nrrd --dpL

    count=`expr $count + 1`
done

