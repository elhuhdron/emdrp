
# run on red in ~/gits/emdrp/recon/python

dpWriteh5.py --dataset voxel_type --srcfile /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/M0007_supervoxels.h5 --outfile /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/M0007_supervoxels_agglo.h5 --chunk 16 17 0 --offset 0 0 32 --size 1024 1024 480 --dpW --dpL
dpWriteh5.py --dataset voxel_type --srcfile /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/M0027_supervoxels.h5 --outfile /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/M0027_supervoxels_agglo.h5 --chunk 12 14 2 --offset 0 0 32 --size 1024 1024 480 --dpW --dpL

count=0
iter=0
while [ $iter -le 100 ]
do
    agglo_iter=`printf '%.8f' $iter`
    echo processing $agglo_iter

    dpCleanLabels.py --dataset labels --subgroups agglomeration $agglo_iter --get-svox-type --ECS-label 0 --dpW --dpCl --srcfile /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/M0007_supervoxels_agglo.h5 --chunk 16 17 0 --offset 0 0 32 --size 1024 1024 480 --outfile /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/M0007_supervoxels_agglo.h5
    dpCleanLabels.py --dataset labels --subgroups agglomeration $agglo_iter --get-svox-type --ECS-label 0 --dpW --dpCl --srcfile /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/M0027_supervoxels_agglo.h5 --chunk 12 14 2 --offset 0 0 32 --size 1024 1024 480 --outfile /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/M0027_supervoxels_agglo.h5

    count=`expr $count + 1`
    iter=`expr $iter + 2`
done

