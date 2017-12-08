
# run on red in ~/gits/emdrp/recon/python
#   can both run in background in parallel

nohup python -u dpSupervoxelClassifier.py --cfgfile ~/gits/emdrp/pipeline/ECS_full_run2/classifier_M0007_export.ini --dpSupervoxelClassifier-verbose --classifierin /mnt/syn2/watkinspv/full_datasets/neon_xfold/vgg3pool64_ECS_full_run2/M0007_agglo_classifiers --classifier rf --outfile /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/M0007_supervoxels_agglo.h5 --classifierout '' --feature-set medium --neighbor-only --no-agglo-ECS --nthreads 16 --prob-svox-context >& tmp_watershed_agglo_export_M0007.txt &
nohup python -u dpSupervoxelClassifier.py --cfgfile ~/gits/emdrp/pipeline/ECS_full_run2/classifier_M0027_export.ini --dpSupervoxelClassifier-verbose --classifierin /mnt/syn2/watkinspv/full_datasets/neon_xfold/vgg3pool64_ECS_full_run2/M0027_agglo_classifiers --classifier rf --outfile /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_ECS_full_run2/M0027_supervoxels_agglo.h5 --classifierout '' --feature-set medium --neighbor-only --no-agglo-ECS --nthreads 16 --prob-svox-context >& tmp_watershed_agglo_export_M0027.txt &

