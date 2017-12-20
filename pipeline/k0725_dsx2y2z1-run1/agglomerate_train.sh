
# run on red in ~/gits/emdrp/recon/python

nohup python -u dpSupervoxelClassifier.py --cfgfile ~/gits/emdrp/pipeline/k0725_dsx2y2z1-run1/classifier_k0725_train.ini --dpSupervoxelClassifier-verbose --classifierout /mnt/syn2/watkinspv/full_datasets/neon_xfold/vgg3pool64_k0725_ds2_run1/k0725_ds2_agglo_classifiers.dill --classifier rf --outfile '' --feature-set medium --neighbor-only --nthreads 16 --chunk-subgroups >& tmp_agglo_train_k0725.txt &

