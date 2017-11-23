
# run on green,red,infra,ultra
# modify appropriately
# next time modify this to split into 3 groups on each nas
#   then use two rsyncs on the last machine to copy from the 3rd group on each

d=`pwd`
cd /mnt/syn2/watkinspv/full_datasets/neon/mfergus32_K0057_ds3_run4/probs
find . -name '*.h5' > files.txt
split -l 30 files.txt tmp.
mkdir 01 02
xargs -a tmp.aa mv -t 01
xargs -a tmp.ab mv -t 02
rm files.txt tmp.a?
cd /mnt/syn/watkinspv/full_datasets/neon/mfergus32_K0057_ds3_run4/probs
find . -name '*.h5' > files.txt
split -l 30 files.txt tmp.
mkdir 03 04
xargs -a tmp.aa mv -t 03
xargs -a tmp.ab mv -t 04
rm files.txt tmp.a?
cd $d

#rsync -avh --progress -e ssh /mnt/syn2/watkinspv/full_datasets/neon/mfergus32_K0057_ds3_run4/probs/04 watkinspv@biowulf.nih.gov:/data/CDCU/full_datasets/neon/mfergus32_K0057_ds3_run4/probs/

