
# run on green,red,infra,ultra
# modify appropriately

cd /mnt/syn2/watkinspv/full_datasets/neon/mfergus32_K0057_ds3_run3/probs
find . -name '*.h5' > files.txt
split -l 30 files.txt tmp.
mkdir 01 02
xargs -a tmp.aa mv -t 01
xargs -a tmp.ab mv -t 02
rm files.txt tmp.a?
cd
rsync -avh --progress -e ssh /mnt/syn2/watkinspv/full_datasets/neon/mfergus32_K0057_ds3_run3/probs/04 watkinspv@biowulf.nih.gov:/data/CDCU/full_datasets/neon/mfergus32_K0057_ds3_run3/probs/

