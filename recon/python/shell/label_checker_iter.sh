
fnraw=/Data/datasets/raw/K0057_D31_dsx3y3z1.h5
dataset=data_mag_x3y3z1
fnpath=/home/watkinspv/Downloads/K0057_tracing
tmp=/home/watkinspv/Downloads/K0057_tracing/tmp
size='256 256 32'
ctx_size='384 384 64'
ctx_chunk='0 0 0'
ctx_offset='64 64 16'
minsize=9
smooth_size='5 5 5'

declare -a chunks=("6 23 3" "16 19 15" "24 14 8" "13 18 15" "4 35 3" "10 11 18")
declare -a offsets=("0 0 0" "0 0 32" "0 0 0" "0 0 64" "96 96 64" "32 96 48")
declare -a fns=("K0057_D31_dsx3y3z1_x6_y23_z3o0" "K0057_D31_dsx3y3z1_x16_y19_z15o32" "K0057_D31_dsx3y3z1_x24_y14_z8o0" "K0057_D31_dsx3y3z1_x13_y18_z15o64" "K0057_D31_dsx3y3z1_x4o96_y35o96_z3o64" "K0057_D31_dsx3y3z1_x10o32_y11o96_z18o48")
declare -a contour_levels=('0.4' '0.45' '0.42' '0.4' '0.4' '0.35')

count=0
for chunk in "${chunks[@]}";
do
    echo processing $chunk

    # load raw data and write out nrrd
    dpLoadh5.py --srcfile $fnraw --dataset $dataset --outraw $fnpath/${fns[$count]}_crop.nrrd --chunk $chunk --size $size --offset ${offsets[$count]}
    
    # save labels into temp hdf5 file and crop out middle labeled region into nrrd
    rm $tmp.h5
    dpWriteh5.py --inraw $fnpath/${fns[$count]}_labels.nrrd --chunksize 128 128 64 --datasize $ctx_size --size $ctx_size --chunk 0 0 0 --outfile $tmp.h5 --data-type-out uint16 --dataset labels
    dpLoadh5.py --srcfile $tmp.h5 --dataset labels --outraw $fnpath/${fns[$count]}_labels_crop.nrrd --chunk $ctx_chunk --offset $ctx_offset --size $size
    
    # main label cleaning steps, all steps are done in 3d (NOT per 2d zslice)
    
    # (1) smoothing, done per label
    dpCleanLabels.py --srcfile $tmp.h5 --chunk $ctx_chunk --offset $ctx_offset --size $size --smooth --smooth-size $smooth_size --contour-lvl ${contour_levels[$count]} --dpC
    
    # (2) remove adjacencies
    dpCleanLabels.py --srcfile $tmp.h5 --chunk $ctx_chunk --offset $ctx_offset --size $size --remove_adjacencies --fg-connectivity 3 --dpC
    
    # (3) connected components
    dpCleanLabels.py --srcfile $tmp.h5 --chunk $ctx_chunk --offset $ctx_offset --size $size --relabel --fg-connectivity 3 --dpC
    
    # (4) remove small components by voxel size
    dpCleanLabels.py --srcfile $tmp.h5 --chunk $ctx_chunk --offset $ctx_offset --size $size --minsize $minsize --dpC
    
    # (5) fill cavities
    dpCleanLabels.py --srcfile $tmp.h5 --chunk $ctx_chunk --offset $ctx_offset --size $size --outraw $fnpath/${fns[$count]}_labels_clean.nrrd --cavity-fill --dpC

    count=`expr $count + 1`
done

