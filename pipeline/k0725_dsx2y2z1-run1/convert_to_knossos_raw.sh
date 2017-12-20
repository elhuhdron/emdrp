
# run on blue in ~/gits/emdrp/recon/python

dpCubeIter.py --volume_range_beg 0 0 0 --volume_range_end 19 62 79 --overlap 0 0 0 --cube_size 1 1 1 --cmd "python -u $HOME/gits/emdrp/recon/python/dpLoadh5.py --srcfile /Data/datasets/raw/k0725_dsx2y2z1.h5 --dataset data_mag_x2y2z1 --dpL" --fileflags outraw --filepaths /mnt/ext/110629_k0725/cubes_dsx2y2z1/mag1 --fileprefixes 'k0725_mag1' --filepostfixes '.raw' --filepaths-affixes 1 > tmp_out.sh
nohup sh tmp_out.sh >& tmp_out.txt &

