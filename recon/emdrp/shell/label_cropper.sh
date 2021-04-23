
#vsrcfile=/Data/watkinspv/full_datasets/neon/vgg3pool_k0725_26x40x40/k0725_supervoxels_x0013_y0013_z0005.h5
#lsrcfile=/Data/watkinspv/full_datasets/neon/vgg3pool_k0725_26x40x40/agglo/k0725_vgg3pool_aggloall_rf_75iter2p_medium_filter_supervoxels_x0013_y0013_z0005.h5
#outfile=/home/watkinspv/Downloads/tmp.h5
#subgroups='agglomeration 48.00000000'
#chunk='15 15 6'
#size='768 768 256'
#offset='0 0 0'

vsrcfile=/Data_yello/watkinspv/full_datasets/neon/mfergus32all_K0057_ds3_run2/wtsh/K0057_D31_dsx3y3z1_supervoxels_x0002_y0032_z0001.h5
lsrcfile=/Data_yello/watkinspv/full_datasets/neon/mfergus32all_K0057_ds3_run2/wtsh/K0057_D31_dsx3y3z1_supervoxels_x0002_y0032_z0001.h5
outfile=/home/watkinspv/Downloads/tmp.h5
subgroups='with_background 0.50000000'
chunk='4 35 3'
size='256 256 128'
offset='96 96 0'
outraw=/home/watkinspv/Downloads/K0057_bootstrapping/K0057_D31_dsx3y3z1_x4o96_y35o96_z3_boot.nrrd

# crop out voxel type and labels
./dpWriteh5.py --srcfile $vsrcfile --dataset voxel_type --chunk $chunk --size $size --offset $offset --outfile $outfile --dpW --dpL
./dpWriteh5.py --srcfile $lsrcfile --dataset labels --subgroups $subgroups --subgroups-out --chunk $chunk --size $size --offset $offset --outfile $outfile --dpW --dpL

# apply the relabel sequential (minsize==1) and rewrite voxel type
./dpCleanLabels.py --srcfile $outfile --dataset labels --chunk $chunk --size $size --offset $offset --write-voxel-type --minsize 1 --ECS-label 0 --dpWriteh5-verbose --dpCleanLabels-verbose --dpLoadh5-verbose

# setup for proofing with ECS, use bigger minsize, do cavity fill
./dpCleanLabels.py --srcfile $outfile --dataset labels --chunk $chunk --size $size --offset $offset --minsize 128 --cavity-fill --cavity-fill-minsize 4096 --write-voxel-type --ECS-label 0 --dpCleanLabels-verbose
# replace ECS with single label and set minlabel (do this last)
./dpCleanLabels.py --srcfile $outfile --dataset labels --chunk $chunk --size $size --offset $offset --replace-ECS --ECS-label 1 --min-label 2 --dpWriteh5-verbose --dpCleanLabels-verbose --dpLoadh5-verbose
./dpLoadh5.py --srcfile $outfile --dataset labels --chunk $chunk --size $size --offset $offset --outraw $outraw --zeropadraw 64 64 64 64 16 16 --dpL
