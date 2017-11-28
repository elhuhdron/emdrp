# electron microscopy data reconstruction pipeline (emdrp)

Tools for volumetric segmentation / reconstruction of nervous tissue from serial electron microscopy data. Potential exists for application to other 3D imaging modalities. See high-level [documentation and introduction](doc/wiki/README.md).

## Publications

> Pallotto M, Watkins PV, Fubara B, Singer JH, Briggman KL. (2015)
> [Extracellular space preservation aids the connectomic analysis of neural circuits.](https://elifesciences.org/articles/08206)
> *Elife.* 2015 Dec 9;4. pii: e08206. doi: 10.7554/eLife.08206. PMID: 26650352

## Installation and Dependencies

python3 is required. anaconda install is recommended, with a few additional [requirements](doc/setup/python3_pip_requirements.txt).

Full usage of the pipeline requires matlab installation with following toolboxes:
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

The emdrp utilizes [neon](https://github.com/NervanaSystems/neon) as the convnet implementation for machine voxel classification. Sync to the current [supported release](neon3/neon_version.txt), apply a small [patch](neon3/neon.patch) from the path where neon was cloned and install per their instructions (Anaconda install method recommended). Finally install a few additional [requirements](neon3/requirements.txt) in the neon environment.

python C extensions were created for fast performance of some pipeline steps. Build these are built with a simple [Makefile](recon/python/utils/pyCext/Makefile) after modifying the appropriate paths to python and numpy install locations.

Currently the emdrp is more a collection of python, matlab and shell scripts than a toolbox or single install. Until this is remedied, the following need to be added to the respective paths (relative to emdrp clone path):

- PATH
  - `emdrp/recon/python`
- PYTHONPATH
  - `emdrp/recon/python`
  - `emdrp/recon/python/utils`
  - `emdrp/recon/python/utils/pyCext`
- matlab path
  - `emdrp/recon/matlab/hdf5`
  - `emdrp/recon/matlab/knossos`

## Tutorial / Example Run

Reset the repository to the [commit]() that works with the example.

Download [datasets](https://elifesciences.org/articles/08206/figures#data-sets) and training and testing data (Figure 3—source data 1 to 4) generated for the [ECS preservation paper](https://elifesciences.org/articles/08206).

All scripts for running through this tutorial are located at `pipeline/ECS_tutorial`. Many scripts will require changing paths to the location data files were downloaded to.

### Create hdf5 data files

The emdrp uses hdf5 as the container for all data. The first step is to create hdf5 files for the raw EM data using top-level matlab script `top_make_hdf5_from_knossos_raw.m`

Manually annotated training data also needs to be converted to hdf5 using scripts `label_maker*.sh` The emdrp does not support tiff stacks, so the downloaded label data should convert to either nrrd, gipl or raw formats ([fiji](https://fiji.sc/) recommended). Labels can be validated using `label_validator*.sh` scripts. The raw format exports from the emdrp data scripts are typically used in conjunction with [itksnap](http://www.itksnap.org/pmwiki/pmwiki.php) for viewing small volumes.

### Train convnets



## Legacy

### Modified cuda-convnets2

Sample run for training modified [cuda-convnet2](https://github.com/akrizhevsky/cuda-convnet2) for EM data:

```
python -u convnet.py --data-path=./emdrp-config/EMdata-3class-16x16out-ebal-huge-all-xyz.ini --save-path=../data --test-range=1-5 --train-range=1-200 --layer-def=./emdrp-config/layers-EM-3class-16x16out.cfg --layer-params=./emdrp-config/layer-params-EM-3class-16x16out.cfg --data-provider=emdata --test-freq=40 --epochs=10 --gpu=0
```
Works with cuda 7.5 and anaconda python2.7 plus additional conda [requirements](doc/setup/python2_conda_requirements.txt).
