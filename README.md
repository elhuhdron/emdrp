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

python C extensions were created for fast performance of some pipeline steps. Build these with a simple [Makefile](recon/python/utils/pyCext/Makefile) after modifying the appropriate paths to python and numpy install locations.

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

## Tutorial / Example Workflow

Reset the repository to the [commit]() that works with the example.
TODO: add commit or release after tutorial completed, for now reset to HEAD

Download [datasets](https://elifesciences.org/articles/08206/figures#data-sets) and training and testing data (Figure 3—source data 1 to 4) generated for the [ECS preservation paper](https://elifesciences.org/articles/08206).

Raw data for these test cases is two volumes of size 1024x1024x512 voxels with voxel resolution of 9.8x9.8x25 nm. The data are used for the 3D section of the ECS preservation paper; `M0027_11` is prepared using standard tissue preparation techniques for EM, while `M0007_33` preserves a large percentage of extracellular space.

All scripts for running through this tutorial are located at `pipeline/ECS_tutorial`. Many scripts will require changing paths depending on the location that data files were downloaded to or written to at each pipeline step.

### Create data containers

The emdrp uses hdf5 as the container for all data. The first step is to create hdf5 files for the raw EM data using top-level matlab script `top_make_hdf5_from_knossos_raw.m`

Manually annotated training data also needs to be converted to hdf5 using scripts `label_maker*.sh` The emdrp does not support tiff stacks, so the downloaded label data needs to be converted to either nrrd, gipl or raw formats ([fiji](https://fiji.sc/) recommended). Labels can be validated using `label_validator*.sh` scripts. The raw format exports from the emdrp data scripts are typically used in conjunction with [itksnap](http://www.itksnap.org/pmwiki/pmwiki.php) for viewing small volumes.

### Train convnets

To train against all training data with neon, activate the neon environment and run from the emdrp neon3 subdirectory (change paths appropriately):

```
python -u ./emneon.py -e 1 --data_config ~/gits/emdrp/pipeline/ECS_tutorial/EMdata-3class-64x64out-rand-M0007.ini --image_in_size 128 --serialize 800 -s ~/Data/ECS_tutorial/convnet_out/M0007_0.prm -o ~/Data/ECS_tutorial/convnet_out/M0007_0.h5 --model_arch vgg3pool --train_range 100001 112800 --epoch_dstep 5600 4000 2400 --nbebuf 1 -i 0

python -u ./emneon.py -e 1 --data_config ~/gits/emdrp/pipeline/ECS_tutorial/EMdata-3class-64x64out-rand-M0027.ini --image_in_size 128 --serialize 800 -s ~/Data/ECS_tutorial/convnet_out/M0027_0.prm -o ~/Data/ECS_tutorial/convnet_out/M0027_0.h5 --model_arch vgg3pool --train_range 100001 112800 --epoch_dstep 5600 4000 2400 --nbebuf 1 -i 0
```

Typically 4 independent convnets are trained on all training data. However, the agglomeration step of the pipeline trains much better against segmentations created from the test volumes of cross-validated convnets. For a small number of training volumes, a leave-one-volume-out cross-validation, follwed by training the agglomeration with the test volumes, has given the best agglomeration training results.

`emneon.py` contains a handy flag `--chunk-skip-list` for leave-n-out cross validations:

```
python -u ./emneon.py -e 1 --data_config ~/gits/emdrp/pipeline/ECS_tutorial/EMdata-3class-64x64out-rand-M0027.ini --image_in_size 128 --serialize 800 -s ~/Data/ECS_tutorial/convnet_out/M0027_test0_0.prm -o ~/Data/ECS_tutorial/convnet_out/M0027_test0_0.h5 --model_arch vgg3pool --train_range 100001 112800 --epoch_dstep 5600 4000 2400 --nbebuf 1 -i 0 --test_range 200001 200001 --chunk_skip_list 0 --eval 800
```

This is repeated for each training volume, resulting in a total of 28 trained convnets for each dataset: 4 each for the six leave-one-volume-out runs and for the train-on-all-volumes runs.

### Export probabilities

This step exports probability of voxel classification types from each trained convnet. To simplify scripts and preserve some amount of context, the entirety of the volumes is exported for each trained convnet (28 for each dataset). Context outside of the test cube is optionally used by the watershed and agglomeration steps. For example, for each trained convnet:

```
python -u ./emneon.py --data_config ~/gits/emdrp/pipeline/ECS_tutorial/EMdata-3class-64x64out-export-M0007.ini --model_file ~/Data/ECS_tutorial/convnet_out/M0007_0.prm --write_output ~/Data/ECS_tutorial/xfold/M0007_0_probs.h5 --test_range 200001 200256 -i 0
```

### Merge probabilities

Although any number of aggregation of the trained convnets could be used, empirically probability mean, min and max operations have given the best segmentation results. The means are used to generate segmentations in the watershed step and the means and maxes are used as training features in the agglomeration step.

For example, for a single cross-validation:
```
python -u dpMergeProbs.py --srcpath ~/Data/ECS_tutorial/xfold --srcfiles M0007_0_probs.h5 M0007_1_probs.h5 M0007_2_probs.h5 M0007_3_probs.h5 --dim-orderings xyz xyz xyz xyz --outprobs ~/Data/ECS_tutorial/xfold/M0007_probs.h5 --chunk 18 15 3 --size 128 128 128 --types ICS --ops mean min --dpM

python -u dpMergeProbs.py --srcpath ~/Data/ECS_tutorial/xfold --srcfiles M0007_0_probs.h5 M0007_1_probs.h5 M0007_2_probs.h5 M0007_3_probs.h5 --dim-orderings xyz xyz xyz xyz --outprobs ~/Data/ECS_tutorial/xfold/M0007_probs.h5 --chunk 18 15 3 --size 128 128 128 --types MEM ECS --ops mean max --dpM
```

### Watershed

This step which creates the initial segmentations is a custom automatically-seeded watershed algorithm. The algorithm automatically picks seed locations by preserving 3D regions that have fallen below a particular size with increasing thresholds on the mean probabilities. Three segmentations are created:
  1. with_background: voxel identity as predicted with winner-take-all probability from the convnet outputs are preserved
  2. no_adjacencies: supervoxels are flushed out but background is preserved to maintain non-adjacency between components
  3. zero_background: fully watershedded segmentation with no background remaining

For example, for a single cross-validation:
```
python -u dpWatershedTypes.py --probfile ~/Data/ECS_tutorial/xfold/M0007_probs.h5 --chunk 18 15 3 --offset 0 0 0 --size 128 128 128 --outlabels ~/Data/ECS_tutorial/xfold/M0007_supervoxels.h5 --ThrRng 0.5 0.999 0.1 --ThrHi 0.95 0.99 0.995 0.999 0.99925 0.9995 0.99975
0.9999 0.99995 0.99999 --dpW
```

### Agglomerate

The agglomeration step empirically gives the best results when trained on the test volumes from convnets trained using leave-one-out cross validation. To create a trained set of random forest classifiers with the emdrp agglomerator, modify paths appropriately in corresponding ini files, then run:

```
dpSupervoxelClassifier.py --cfgfile ~/gits/emdrp/pipeline/ECS_tutorial/classifier_M0007_train.ini --dpSupervoxelClassifier-verbose --classifierout ~/Data/ECS_tutorial/xfold/M0007_agglo_classifiers.dill --classifier rf --outfile '' --feature-set medium --neighbor-only --no-agglo-ECS --nthreads 16

dpSupervoxelClassifier.py --cfgfile ~/gits/emdrp/pipeline/ECS_tutorial/classifier_M0027_train.ini --dpSupervoxelClassifier-verbose --classifierout ~/Data/ECS_tutorial/xfold/M0027_agglo_classifiers.dill --classifier rf --outfile '' --feature-set medium --neighbor-only --no-agglo-ECS --nthreads 16
```

The trained classifiers are then used to export agglomerations iteratively for the entire volumes. Paths in the ini files should be changed as to point to probabilities and watershed inputs that were exported over the entire volume and using all the training data (not the cross-validations). This is a "test-only" or export mode for the trained classifiers:

```
dpSupervoxelClassifier.py --cfgfile ~/gits/emdrp/pipeline/ECS_tutorial/classifier_M0007_export.ini --dpSupervoxelClassifier-verbose --classifierin ~/Data/ECS_tutorial/xfold/M0007_agglo_classifiers --classifier rf --outfile ~/Data/ECS_tutorial/M0007_supervoxels_agglo.h5 --classifierout '' --feature-set medium --neighbor-only --no-agglo-ECS --nthreads 16

dpSupervoxelClassifier.py --cfgfile ~/gits/emdrp/pipeline/ECS_tutorial/classifier_M0027_export.ini --dpSupervoxelClassifier-verbose --classifierin ~/Data/ECS_tutorial/xfold/M0027_agglo_classifiers --classifier rf --outfile ~/Data/ECS_tutorial/M0007_supervoxels_agglo.h5 --classifierout '' --feature-set medium --neighbor-only --no-agglo-ECS --nthreads 16
```

A further step is necessary in order to calculate metrics for the agglomeration segmentation, that is to reclassify which supervoxels are ICS and which are ECS based on the maximum number of original voxels classifications within each agglomerated supervoxel. First the voxel types need to be copied from the watershed output and then another script that can perform several heuristical label "cleaning" steps re-sorts the supervoxels into ICS or ECS based on the winning contained type. For example:

```
dpWriteh5.py --dataset voxel_type --srcfile ~/Data/ECS_tutorial/M0007_supervoxels.h5 --outfile ~/Data/ECS_tutorial/M0007_supervoxels_agglo.h5 --chunk 16 17 0 --offset 0 0 32 --size 1024 1024 480 --dpW --dpL

dpCleanLabels.py --dataset labels --subgroups agglomeration 2.00000000 --get-svox-type --ECS-label 0 --dpW --dpCl --srcfile ~/Data/ECS_tutorial/M0007_supervoxels_agglo.h5 --chunk 16 17 0 --offset 0 0 32 --size 1024 1024 480 --outfile ~/Data/ECS_tutorial/M0007_supervoxels_agglo.h5
```

The `dpCleanLabels.py` step needs to be repeated for each agglomeration iteration (identified by second `--subgroup` argument) that is to be analyzed (script recommended).

### Skeleton metrics

The agglomerated segmentation is then compared against skeletonized ground truth to arrive at a meaningful metric for EM data, error free path length (EFPL). Loosely EFPL is the distance that one can travel along a neurite before encountering either a split or merger error. The EFPL metrics for the emdrp are calculated with a matlab function and using supervoxels generated at each iterative step of the agglomeration (or without the agglomeration, each threshold of the watershed). An example top-level for calculating the emdrp EFPL metrics is given in `knossos_efpl_top.m`

After generating the metrics, split-merger and total EFPL plots (amongst others) can be displayed with an example top-level plotting script, `knossos_efpl_plot_top.m`

