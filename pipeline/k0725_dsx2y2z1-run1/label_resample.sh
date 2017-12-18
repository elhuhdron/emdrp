
# resample labels to approximately isotropic voxels, run on green
#   labels op should prevent any need for smoothing first
dpResample.py --srcfile /mnt/syn/datasets/labels/gt/k0725_labels_benahmedf_2frontendcubes.h5 --dataset labels --volume_range_beg 10 11 3 --volume_range_end 12 15 4 --overlap 0 0 0 --cube_size 2 4 1 --dpRes --resample-dims 1 1 0 --factor 2 --outfile /mnt/syn/watkinspv/k0725_labels_dsx2y2z1.h5 --dataset-out labels --downsample-op labels

