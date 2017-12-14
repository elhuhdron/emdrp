
# run on blue in ~/gits/emdrp/recon/python

dpCubeIter.py --volume_range_beg 16 17 0 --volume_range_end 24 25 4 --overlap 0 0 0 --cube_size 1 1 1 --cmd "python -u $HOME/gits/emdrp/recon/python/dpLoadh5.py --dataset labels --subgroups agglomeration 50.00000000  --data-type uint64 --raw-compression --srcfile /Data/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/M0007_supervoxels_agglo.h5 --dpL" --fileflags outraw --filepaths /mnt/ext/ECS_paper/cubes_M0007 --fileprefixes 'M0007_33_mag1' --filepostfixes .seg --filepaths-affixes 1 --filemodulators 1 1 1 > tmp_out_M0007_seg.sh
nohup sh tmp_out_M0007_seg.sh >& tmp_out_M0007_seg.txt &

dpCubeIter.py --volume_range_beg 12 14 2 --volume_range_end 20 22 6 --overlap 0 0 0 --cube_size 1 1 1 --cmd "python -u $HOME/gits/emdrp/recon/python/dpLoadh5.py --dataset labels --subgroups agglomeration 56.00000000  --data-type uint64 --raw-compression --srcfile /Data/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/M0027_supervoxels_agglo.h5 --dpL" --fileflags outraw --filepaths /mnt/ext/ECS_paper/cubes_M0027 --fileprefixes 'M0027_11_mag1' --filepostfixes .seg --filepaths-affixes 1 --filemodulators 1 1 1 > tmp_out_M0027_seg.sh
nohup sh tmp_out_M0027_seg.sh >& tmp_out_M0027_seg.txt &

