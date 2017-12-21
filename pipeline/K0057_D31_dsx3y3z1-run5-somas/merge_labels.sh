
# run on blue in recon/python against hand merged supervoxels that were generated with this run.
# merged labels are then viewable as a whole dataset in itksnap with a machine with 32G of memory.
# use data_resample.py to generated the same downsampling of the raw data.

# backup old labels 
mv /home/watkinspv/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4.h5 /home/watkinspv/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-old.h5
mv /home/watkinspv/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean.h5 /home/watkinspv/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean-old.h5

# run merging that is concurrent with smoothing and downsampling.
# this can take several hours for lots of merging over many superchunks.
#   optimized so that only superchunks that contain supervoxels to be merged will be loaded.
#   each superchunk containing a supervoxel to be merged will also only be loaded once.
# segmentation values must match the thresholds that were written to knossos formatted segmentation levels.
./dpLabelMerger.py --annotation-file-glob '/home/watkinspv/Downloads/K0057_soma_annotation/done/K0057-D31-somas.365_use.rad150.*-*.k.zip' --volume_range_beg 2 8 1 --volume_range_end 50 38 19 --overlap 0 0 0 --cube_size 6 6 6 --dsfactor 4 --dpLabelM --filepaths /Data_yello/watkinspv/full_datasets/neon/mfergus32_K0057_ds3_run5/clean_wtsh --fileprefixes K0057_D31_dsx3y3z1_supervoxels_clean --filepostfixes .h5 --dataset labels --subgroups with_background 00000000 --segmentation-values 0.99900000 0.99925000 0.99950000 0.99975000 --subgroups-out --outfile /home/watkinspv/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4.h5 --smooth 9 9 9 --contour-lvl 0.20

# run cleaning to fill in any cavities (fixes a lot of 'non-perfect' merging ommissions for small supervoxels)
./dpCleanLabels.py --dataset labels --cavity-fill --ECS-label 0 --dpWriteh5-verbose --dpCleanLabels-verbose --srcfile /home/watkinspv/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4.h5 --outfile /home/watkinspv/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean.h5 --chunk 0 0 0 --size 1696 1440 640 --labelwise

# export labels as nrrd file
dpLoadh5.py --srcfile ~/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean.h5 --dataset labels --chunk 0 0 0 --size 1696 1440 640 --outraw ~/Downloads/K0057_soma_annotation/out/K0057_D31_dsx12y12z4_somas.nrrd --dpL

