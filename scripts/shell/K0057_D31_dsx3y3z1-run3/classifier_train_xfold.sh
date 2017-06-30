
# run on red in ~/gits/emdrp/recon/python

# context did not work well on this run (possibly bug?)
#nohup time python -u ./dpSupervoxelClassifier.py --cfgfile config/svox_K0057_ds3_iterate.ini --dpSupervoxelClassifier-verbose --classifierout /Data/watkinspv/agglo/K0057_D31_dsx3y3z1-run3-classifier.dill --classifier rf --outfile '' --feature-set medium --neighbor-only --no-agglo-ECS --prob-svox-context --nthreads 16 >& out_K0057_D31_dsx3y3z1-run3-classifier.txt &
nohup time python -u ./dpSupervoxelClassifier.py --cfgfile config/svox_K0057_ds3_iterate.ini --dpSupervoxelClassifier-verbose --classifierout /Data/watkinspv/agglo/K0057_D31_dsx3y3z1-run3-classifier.dill --classifier rf --outfile '' --feature-set medium --neighbor-only --no-agglo-ECS --nthreads 16 >& out_K0057_D31_dsx3y3z1-run3-classifier.txt &

