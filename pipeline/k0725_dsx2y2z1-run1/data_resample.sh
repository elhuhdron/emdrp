
# resample to approximately isotropic voxels, run on green

# NOTE: not padding to make resampled dataset a multiple of knossos cube size created some issues.
#   does not allow for context or easy writing out of the entire dataset.
#   decided to ignore this for now and just ignore the remainder portion.
#   likely need better fix for this going forward (like force dataset size in downsample as multiple of cube size).
nohup dpResample.py --srcfile /mnt/syn/datasets/raw/k0725.h5 --dataset data_mag1 --volume_range_beg 0 0 0 --volume_range_end 36 120 72 --overlap 0 0 0 --cube_size 12 12 8 --dpRes --resample-dims 1 1 0 --factor 2 --outfile /mnt/syn/watkinspv/k0725_dsx2y2z1.h5 --dataset-out data_mag_x2y2z1 --downsample-op median --right_remainder_size 384 640 896 >& tmp_data_resample.txt &

