
# run on blue in ~/gits/emdrp/recon/python
#   can both run in background in parallel

dpCubeIter.py --volume_range_beg 0 0 0 --volume_range_end 39 35 7 --overlap 0 0 0 --cube_size 1 1 1 --cmd "python -u $HOME/gits/emdrp/recon/python/dpLoadh5.py --srcfile /Data/datasets/raw/M0007_33_39x35x7chunks_Forder.h5 --dataset data_mag1 --dpL" --fileflags outraw --filepaths /mnt/ext/ECS_paper/cubes_M0007 --fileprefixes 'M0007_33_mag1' --filepostfixes '.raw' --filepaths-affixes 1 > tmp_out_M0007.sh
nohup sh tmp_out_M0007.sh >& tmp_out_M0007.txt &

dpCubeIter.py --volume_range_beg 0 0 0 --volume_range_end 33 37 7 --overlap 0 0 0 --cube_size 1 1 1 --cmd "python -u $HOME/gits/emdrp/recon/python/dpLoadh5.py --srcfile /Data/datasets/raw/M0027_11_33x37x7chunks_Forder.h5 --dataset data_mag1 --dpL" --fileflags outraw --filepaths /mnt/ext/ECS_paper/cubes_M0027 --fileprefixes 'M0027_11_mag1' --filepostfixes '.raw' --filepaths-affixes 1 > tmp_out_M0027.sh
nohup sh tmp_out_M0027.sh >& tmp_out_M0027.txt &

