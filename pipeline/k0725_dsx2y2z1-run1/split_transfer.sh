
# run on green,red,infra,ultra
# modify appropriately

d=`pwd`
cd /mnt/syn/watkinspv/full_datasets/neon/vgg3pool64_k0725_ds2_run1/probs
find . -name '*.h5' > files.txt
split -l 72 files.txt tmp.
mkdir 01 02
xargs -a tmp.aa mv -t 01
xargs -a tmp.ab mv -t 02
rm files.txt tmp.a?
cd /mnt/syn2/watkinspv/full_datasets/neon/vgg3pool64_k0725_ds2_run1/probs
find . -name '*.h5' > files.txt
split -l 72 files.txt tmp.
mkdir 03 04 05
xargs -a tmp.aa mv -t 03
xargs -a tmp.ab mv -t 04
xargs -a tmp.ac mv -t 05
rm files.txt tmp.a?
cd $d

#rsync -avh --progress -e ssh /mnt/syn/watkinspv/full_datasets/neon/vgg3pool64_k0725_ds2_run1/probs/01 watkinspv@biowulf.nih.gov:/data/CDCU/full_datasets/neon/vgg3pool64_k0725_ds2_run1/probs/

