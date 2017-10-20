
# run on blue in recon/python against hand merged supervoxels that were generated with this run.
# merged labels are then viewable as a whole dataset in itksnap with a machine with 32G of memory.
# use data_resample.py to generated the same downsampling of the raw data.
rm /home/watkinspv/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4.h5
./dpLabelMerger.py --annotation-file-glob '/home/watkinspv/Downloads/K0057_soma_annotation/K0057-D31-somas.365_use.rad150.*-*.k.zip' --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 0 0 0 --cube_size 6 6 6 --dsfactor 4 --dpLabelM --filepaths /Data_yello/watkinspv/full_datasets/neon/mfergus32_K0057_ds3_run5/clean_wtsh --fileprefixes K0057_D31_dsx3y3z1_supervoxels_clean --filepostfixes .h5 --dataset labels --subgroups with_background 00000000 --segmentation-values 0.99900000 0.99925000 0.99950000 0.99975000 --subgroups-out --outfile /home/watkinspv/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4.h5 --smooth 9 9 9 --contour-lvl 0.20

#./dpCleanLabels.py --dataset labels --cavity-fill --ECS-label 0 --dpWriteh5-verbose --dpCleanLabels-verbose --srcfile /home/watkinspv/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4.h5 --outfile /home/watkinspv/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean.h5 --chunk 0 0 0 --size 1696 1440 640

