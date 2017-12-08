
# run on red in ~/gits/emdrp/recon/python
#   can both run in background in parallel

nohup python -u dpSupervoxelClassifier.py --cfgfile ~/gits/emdrp/pipeline/ECS_full_run2/classifier_M0007_train.ini --dpSupervoxelClassifier-verbose --classifierout /mnt/syn2/watkinspv/full_datasets/neon_xfold/vgg3pool64_ECS_full_run2/M0007_agglo_classifiers.dill --classifier rf --outfile '' --feature-set medium --neighbor-only --no-agglo-ECS --nthreads 16 --prob-svox-context --chunk-subgroups >& tmp_watershed_agglo_train_M0007.txt &
nohup python -u dpSupervoxelClassifier.py --cfgfile ~/gits/emdrp/pipeline/ECS_full_run2/classifier_M0027_train.ini --dpSupervoxelClassifier-verbose --classifierout /mnt/syn2/watkinspv/full_datasets/neon_xfold/vgg3pool64_ECS_full_run2/M0027_agglo_classifiers.dill --classifier rf --outfile '' --feature-set medium --neighbor-only --no-agglo-ECS --nthreads 16 --prob-svox-context --chunk-subgroups >& tmp_watershed_agglo_train_M0027.txt &

