
fn=/home/watkinspv/Downloads/ITKcube_groundtruth_x0016_y0017_z0004
tmp=/home/watkinspv/Downloads/tmp
fnraw=/Data/datasets/raw/M0027_11_33x37x7chunks_Forder
outraw=/home/watkinspv/Downloads/ITKcube_raw_x0016_y0017_z0004
chunk='16 17 4'
size='128 128 128'

# load raw data and write out nrrd
dpLoadh5.py --srcfile $fnraw.h5 --dataset data_mag1 --outraw $outraw.nrrd --chunk $chunk --size $size

# save labels into temp hdf5 file and crop out middle labeled region into nrrd
rm $tmp.h5
dpWriteh5.py --inraw $fn.gipl --chunksize 128 128 128 --datasize 384 384 384 --size 384 384 384 --chunk 0 0 0 --outfile $tmp.h5 --data-type-out uint16 --dataset labels
dpLoadh5.py --srcfile $tmp.h5 --dataset labels --outraw ${fn}_crop.nrrd --chunk 1 1 1 --size $size

# main label cleaning steps, all steps are done in 3d (NOT per 2d zslice)

# (1) smoothing, done per label
dpCleanLabels.py --srcfile $tmp.h5 --chunk 1 1 1 --size $size --smooth --dpC

# (2) remove adjacencies
dpCleanLabels.py --srcfile $tmp.h5 --chunk 1 1 1 --size $size --remove_adjacencies --fg-connectivity 3 --dpC

# (3) connected components
dpCleanLabels.py --srcfile $tmp.h5 --chunk 1 1 1 --size $size --relabel --fg-connectivity 3 --dpC

# (4) remove small components by voxel size
dpCleanLabels.py --srcfile $tmp.h5 --chunk 1 1 1 --size $size --minsize 28 --dpC

# (5) fill cavities
dpCleanLabels.py --srcfile $tmp.h5 --chunk 1 1 1 --size $size --outraw ${fn}_clean.nrrd --cavity-fill --dpC
