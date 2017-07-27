
# run on red in ~/gits/emdrp/recon/python

nohup time python -u ./dpSupervoxelClassifier.py --cfgfile ~/gits/emdrp/scripts/shell/K0057_D31_dsx3y3z1-run4/svox_K0057_ds3_iterate.ini --dpSupervoxelClassifier-verbose --classifierout /mnt/syn2/watkinspv/full_datasets/neon_xfold/mfergus32_K0057_ds3_run4/K0057_D31_dsx3y3z1-run4-classifier.dill --classifier rf --outfile '' --feature-set medium --neighbor-only --no-agglo-ECS --nthreads 16 >& out_K0057_D31_dsx3y3z1-run4-classifier.txt &

