
# for reference, resample to approximately isotropic voxels, run on green
#python -u dpResample.py --srcfile /mnt/syn/datasets/raw/K0057_D31.h5 --dataset data_mag1 --volume_range_beg 0 0 0 --volume_range_end 150 135 20 --overlap 0 0 0 --cube_size 15 15 5 --dpRes --resample-dims 1 1 0 --factor 3 --outfile /mnt/syn/watkinspv/K0057_D31_dsx3y3z1.h5 --dataset-out data_mag_x3y3z1 --downsample-op median --right_remainder_size 1152 0 0

# further downsampling for somas
python -u dpResample.py --srcfile /mnt/syn/datasets/raw/K0057_D31_dsx3y3z1.h5 --dataset data_mag_x3y3z1 --volume_range_beg 0 0 0 --volume_range_end 50 40 20 --overlap 0 0 0 --cube_size 10 10 10 --dpRes --resample-dims 1 1 1 --factor 4 --outfile /mnt/syn/watkinspv/K0057_D31_dsx12y12z4.h5 --dataset-out data_mag_x12y12z4 --downsample-op mean --right_remainder_size 384 640 0

