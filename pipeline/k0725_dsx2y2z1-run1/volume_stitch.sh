
# xxx - this is only for reference, only every ran stitching code once over an entire dataset.
#   this was done on the original k0725 run that was not downsampled.
#   this needs fixing for future use, simply here to document one successful run.

# concatenate only mode only makes all supervoxels unique over entire dataset
#nohup time python -u dpCubeStitcher.py --volume_range_beg 5 5 1 --volume_range_end 29 45 41 --overlap 0 0 0 --cube_size 8 8 4 --srcfile /Data/watkinspv/full_datasets/neon/vgg3pool_k0725_26x40x40/k0725_vgg3pool_aggloall48_rf_75iter2p_medium_filter_clean_concat_supervoxels.h5 --filepaths /Data/watkinspv/full_datasets/neon/vgg3pool_k0725_26x40x40/clean --fileprefixes k0725_vgg3pool_aggloall48_rf_75iter2p_medium_filter_clean_supervoxels --filepostfixes .h5 --dpCube --concatenate_only >& out_stitch_concat_vgg3pool_k0725_26x40x40.txt &

# two-pass mode stitches together supervoxels using maximum overlap with a two pass method utilizing graph CC.
#nohup time python -u dpCubeStitcher.py --volume_range_beg 5 5 1 --volume_range_end 29 45 41 --overlap 8 8 4 --cube_size 8 8 4 --srcfile /Data/watkinspv/full_datasets/neon/vgg3pool_k0725_26x40x40/k0725_vgg3pool_aggloall48_rf_75iter2p_medium_filter_clean_twopass_supervoxels.h5 --filepaths /Data/watkinspv/full_datasets/neon/vgg3pool_k0725_26x40x40/clean --fileprefixes k0725_vgg3pool_aggloall48_rf_75iter2p_medium_filter_clean_supervoxels --filepostfixes .h5 --dpCube --two_pass >& out_stitch_twopass_vgg3pool_k0725_26x40x40.txt &

