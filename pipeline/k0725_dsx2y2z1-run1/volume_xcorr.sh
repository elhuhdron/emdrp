
# xxx - this is only for reference, only every ran xcorr code once over an entire dataset.
#   this was done on the original k0725 run that was not downsampled.
#   this needs fixing for future use, simply here to document one successful run.
#   cubes to trace were identified with this method (k0725_contouring_round2).

# volume xcorr for k0725, original run that was not downsampled
# start shitty method

#python -u dpCubeIter.py --volume_range_beg 5 5 1 --volume_range_end 29 45 41 --overlap 0 0 0 --cube_size 8 8 4 --cmd "python -u $HOME/gits/emdrp/recon/python/dpVolumeXcorr.py --srcfile /data/CDCU/datasets/raw/k0725.h5 --dataset data_mag1 --train-chunks 10 11 3  10 12 3  11 11 3  11 12 3  10 13 3  10 14 3  11 13 3  11 14 3 --nthreads 8 --dpVolume" --fileflags typefile --filepaths /data/CDCU/full_datasets/neon/vgg3pool_k0725_26x40x40/clean --fileprefixes k0725_vgg3pool_aggloall48_rf_75iter2p_medium_filter_clean_supervoxels --filepostfixes .h5
#perl -ne 'print(($_)x4)' tmp.txt > tmp2.txt
#python -u dpCubeIter.py --volume_range_beg 5 5 1 --volume_range_end 29 45 41 --overlap 0 0 0 --cube_size 4 4 2 --cmdfile tmp2.txt --fileflags savefile --filepaths /data/CDCU/full_datasets/neon/vgg3pool_k0725_26x40x40/xcorr --fileprefixes k0725_volume_xcorr --filepostfixes .npz 
# end shitty method

# xxx - this is a better method?
#python -u dpCubeIter.py --volume_range_beg 5 5 1 --volume_range_end 29 45 41 --overlap 0 0 0 --cube_size 4 4 2 --cmd "python -u $HOME/gits/emdrp/recon/python/dpVolumeXcorr.py --srcfile /data/CDCU/datasets/raw/k0725.h5 --dataset data_mag1 --train-chunks 10 11 3  10 12 3  11 11 3  11 12 3  10 13 3  10 14 3  11 13 3  11 14 3 --nthreads 8 --dpVolume" --fileflags typefile savefile --filepaths /data/CDCU/full_datasets/neon/vgg3pool_k0725_26x40x40/clean /data/CDCU/full_datasets/neon/vgg3pool_k0725_26x40x40/xcorr --fileprefixes k0725_vgg3pool_aggloall48_rf_75iter2p_medium_filter_clean_supervoxels k0725_volume_xcorr --filepostfixes .h5 .npz --filemodulators 2 2 2 1 1 1

