
# run on green,red,infra,ultra
# modify appropriately
# NOTE: next time just stick to 4 or 5 transfers, multiple xfers on one machine slows the overall transfer rate

d=`pwd`
cd /mnt/syn2/watkinspv/full_datasets/neon/mfergus32_K0057_ds3_run5/probs
find . -name '*.h5' > files.txt
split -l 20 files.txt tmp.
mkdir 01 02 03
xargs -a tmp.aa mv -t 01
xargs -a tmp.ab mv -t 02
xargs -a tmp.ac mv -t 02
rm files.txt tmp.a?
cd /mnt/syn/watkinspv/full_datasets/neon/mfergus32_K0057_ds3_run5/probs
find . -name '*.h5' > files.txt
split -l 20 files.txt tmp.
mkdir 04 05 06
xargs -a tmp.aa mv -t 04
xargs -a tmp.ab mv -t 05
xargs -a tmp.ac mv -t 06
rm files.txt tmp.a?
cd $d

#rsync -avh --progress -e ssh /mnt/syn2/watkinspv/full_datasets/neon/mfergus32_K0057_ds3_run5/probs/04 watkinspv@biowulf.nih.gov:/data/CDCU/full_datasets/neon/mfergus32_K0057_ds3_run5/probs/

