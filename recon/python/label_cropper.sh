
vsrcfile=/Data/watkinspv/full_datasets/neon/vgg3pool_k0725_26x40x40/k0725_supervoxels_x0013_y0013_z0005.h5
lsrcfile=/Data/watkinspv/full_datasets/neon/vgg3pool_k0725_26x40x40/agglo/k0725_vgg3pool_aggloall_rf_75iter2p_medium_filter_supervoxels_x0013_y0013_z0005.h5
outfile=/home/watkinspv/Downloads/tmp.h5
chunk='15 15 6'
size='768 768 256'

# crop out voxel type and labels
./dpWriteh5.py --srcfile $vsrcfile --dataset voxel_type --chunk $chunk --size $size --outfile $outfile --dpW --dpL
#./dpWriteh5.py --srcfile $lsrcfile --dataset labels --chunk $chunk --size $size --outfile $outfile --dpW --dpL
./dpWriteh5.py --srcfile $lsrcfile --dataset labels --subgroups agglomeration 48.00000000 --subgroups-out --chunk $chunk --size $size --outfile $outfile --dpW --dpL

# apply the relabel sequential (minsize==1) and rewrite voxel type
./dpCleanLabels.py --srcfile $outfile --dataset labels --chunk $chunk --size $size --write-voxel-type --minsize 1 --ECS-label 0 --dpWriteh5-verbose --dpCleanLabels-verbose --dpLoadh5-verbose
