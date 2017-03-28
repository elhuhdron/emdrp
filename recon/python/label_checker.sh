
#fnraw=/Data/datasets/raw/M0027_11_33x37x7chunks_Forder
#dataset=data_mag1
#tmp=/home/watkinspv/Downloads/tmp
#size='128 128 128'
#ctx_size='384 384 384'
#ctx_chunk='1 1 1'
#ctx_offset='0 0 0'
#minsize=28
#fn=/home/watkinspv/Downloads/ITKcube_groundtruth_x0016_y0017_z0004
#outraw=/home/watkinspv/Downloads/ITKcube_raw_x0016_y0017_z0004
#chunk='16 17 4'
#offset='0 0 0'

fnraw=/Data_yello/watkinspv/K0057_D31_dsx3y3z1.h5
dataset=data_mag_x3y3z1
tmp=/home/watkinspv/Downloads/K0057_tracing/tmp
size='256 256 32'
ctx_size='384 384 64'
ctx_chunk='0 0 0'
ctx_offset='64 64 16'
minsize=9
fn=/home/watkinspv/Downloads/K0057_tracing/K0057_D31_dsx3y3z1_x6_y23_z3o0_labels
outraw=/home/watkinspv/Downloads/K0057_tracing/K0057_D31_dsx3y3z1_x6_y23_z3o0_crop
chunk='6 23 3'
offset='0 0 0'

# load raw data and write out nrrd
dpLoadh5.py --srcfile $fnraw --dataset $dataset --outraw $outraw.nrrd --chunk $chunk --size $size --offset $offset

# save labels into temp hdf5 file and crop out middle labeled region into nrrd
rm $tmp.h5
dpWriteh5.py --inraw $fn.nrrd --chunksize 128 128 64 --datasize $ctx_size --size $ctx_size --chunk 0 0 0 --outfile $tmp.h5 --data-type-out uint16 --dataset labels
dpLoadh5.py --srcfile $tmp.h5 --dataset labels --outraw ${fn}_crop.nrrd --chunk $ctx_chunk --offset $ctx_offset --size $size

# main label cleaning steps, all steps are done in 3d (NOT per 2d zslice)

# (1) smoothing, done per label
dpCleanLabels.py --srcfile $tmp.h5 --chunk $ctx_chunk --offset $ctx_offset --size $size --smooth --dpC

# (2) remove adjacencies
dpCleanLabels.py --srcfile $tmp.h5 --chunk $ctx_chunk --offset $ctx_offset --size $size --remove_adjacencies --fg-connectivity 3 --dpC

# (3) connected components
dpCleanLabels.py --srcfile $tmp.h5 --chunk $ctx_chunk --offset $ctx_offset --size $size --relabel --fg-connectivity 3 --dpC

# (4) remove small components by voxel size
dpCleanLabels.py --srcfile $tmp.h5 --chunk $ctx_chunk --offset $ctx_offset --size $size --minsize $minsize --dpC

# (5) fill cavities
dpCleanLabels.py --srcfile $tmp.h5 --chunk $ctx_chunk --offset $ctx_offset --size $size --outraw ${fn}_clean.nrrd --cavity-fill --dpC
