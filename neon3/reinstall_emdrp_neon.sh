
# this is for installing emdrp version of neon using anaconda python, have to paste into command line.
# currently these "instructions" are on top of Anaconda3-2020.02-Linux-x86_64

# important notes for v2.6.0/v2.6.1em (as of Intel discontinuation with only minor patches for emdrp):
# IMPORTANT: python version is important
#   (3.x version) needs to match that of neon release
#   this was python 3.6(.10) as of the discontinuation of neon (HEAD or v2.6.0)
# cuda version works up to:
#     nvcc: NVIDIA (R) Cuda compiler driver
#     Copyright (c) 2005-2018 NVIDIA Corporation
#     Built on Sat_Aug_25_21:08:01_CDT_2018
#     Cuda compilation tools, release 10.0, V10.0.130
# most likely more recent cuda versions do NOT work.

# notes for starting AFTER v2.6.1em:
# cuda version works up to:
#     nvcc: NVIDIA (R) Cuda compiler driver
#     Copyright (c) 2005-2018 NVIDIA Corporation
#     Built on Sat_Aug_25_21:08:01_CDT_2018
#     Cuda compilation tools, release 10.0, V10.0.130
# most likely more recent cuda versions do NOT work.
# This only applies for neon gpu backend (not for the new tfl backend).

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
# for the v2.6.0/v2.6.1em (as of Intel discontinuation):
#git fetch origin; git reset --hard v2.6.1em
cd neon

# build and install neon into a conda environment
# for the v2.6.0/v2.6.1em (as of Intel discontinuation):
#conda create -y --name neon python=3.6 pip matplotlib virtualenv
# for AFTER v2.6.0/v2.6.1em:
conda create -y --name neon anaconda python=3.8 virtualenv
conda activate neon
make sysinstall -e VIS=true
# build the docs, they are no longer hosted on any nervana (intel) site
make doc

# install the neon emdrp specific python requirements
cd ${REPODIR}/emdrp/neon3/
pip install -r requirements.txt

# to install tensorflow into a new conda environment:
#conda create -n tf-gpu tensorflow-gpu
#conda activate tf-gpu
# install tensorflow into the same environment as neon:
# NOTE: tensorflow will only work using the methods AFTER neon v2.6.0/v2.6.1em
#   That means it will not install correctly in the neon version as of Intel discontinuation.
conda install tensorflow-gpu

# remove the neon cache
rm -rf ~/.cache/neon
# remove the pycuda cache
rm -rf ~/.cache/pycuda

# try to run the mnist example with gpu
cd ${REPODIR}/neon/
python examples/mnist_mlp.py -b gpu -e 10
# and the random em data iterator
cd ${REPODIR}/emdrp/neon3
./emneon.py

conda deactivate
