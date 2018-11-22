
# need to activate anaconda python first!!!
# ac3init

# delete the old env
source deactivate
conda remove -y --name neon --all
conda info --envs

# to rebuild release
cd ~/gits
rm -rf neon
git clone https://github.com/NervanaSystems/neon.git
cd neon
git fetch origin; git reset --hard v2.6.0

## to build HEAD
#cd ~/gits/neon
#git pull

# to apply the patch
patch -p1 < ../emdrp/neon3/neon.patch

# NOTE: python version is important
#   (3.x version) needs to match that of neon release
# may want to clear caches:
#   conda clean -y -a
#   rm -rf ~/.cache/pip
conda create -y --name neon python=3.6 pip matplotlib virtualenv
source activate neon
make sysinstall -e VIS=true
cd ../emdrp/neon3/
pip install -r requirements.txt

# remove the neon cache
rm -rf ~/.cache/neon
# possibly clear pycuda cache?
#rm -rf ~/.cache/pycuda

# to make a patch with git
# git diff > patchfile

