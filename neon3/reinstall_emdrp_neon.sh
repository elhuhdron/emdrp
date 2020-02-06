
# this is for installing emdrp version of neon using anaconda python, have to paste into command line.
# xxx - running as a bash script does not work because conda activate
#   does not work from bash script for some reason.

# IMPORTANT: python version is important
#   (3.x version) needs to match that of neon release
#   this was python 3.6(.10) as of the discontinuation of neon (HEAD or v2.6.0)
# cuda version works up to:
#     nvcc: NVIDIA (R) Cuda compiler driver
#     Copyright (c) 2005-2018 NVIDIA Corporation
#     Built on Sat_Aug_25_21:08:01_CDT_2018
#     Cuda compilation tools, release 10.0, V10.0.130
# most likely more recent cuda versions do NOT work.

# # to clear caches
# conda clean -y -a
# rm -rf ~/.cache/pip

# both neon and emdrp clones need to be under this path
REPODIR=~/gits

# delete the old env
conda activate base
conda remove -y --name neon --all
#conda info --envs

# to rebuild release
cd ${REPODIR}
rm -rf neon

# old method, clone nervana repository and checkout release
#git clone https://github.com/NervanaSystems/neon.git
#cd neon
#git fetch origin; git reset --hard v2.6.0
# to apply the patch
#patch -p1 < ../emdrp/neon3/neon.patch
## to make a patch with git
## git diff > patchfile

# new method, checkout forked version
git clone https://github.com/elhuhdron/neon.git
cd neon

# build and install neon into a conda environment
conda create -y --name neon python=3.6 pip matplotlib virtualenv
conda activate neon
make sysinstall -e VIS=true
# build the docs, they are no longer hosted on any nervana (intel) site
make doc

# install the neon emdrp specific python requirements
cd ${REPODIR}/emdrp/neon3/
pip install -r requirements.txt

# remove the neon cache
rm -rf ~/.cache/neon
# remove the pycuda cache
rm -rf ~/.cache/pycuda

# try to run the mnist example with gpu
cd ${REPODIR}/neon/
python examples/mnist_mlp.py -b gpu -e 10
conda deactivate
