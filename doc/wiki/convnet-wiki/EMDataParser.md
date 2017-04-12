
# Introduction

TODO: this is old example with cuda-convet2, update for neon, add working example based on same data

EM data is parsed directly out of hdf5 files by a single class ``EMDataParser`` in module ``python_util/parseEMdata.py`` that can also operate standalone for debug. Most typically the network is being trained using supervised learning, so both raw EM images and label hdf5 containers must be given as inputs. The parser pulls out 2D slices based on parameters specified in an ``.ini`` file that is given via command line as the ``--data-path`` argument to ``convnet.py``. The ``--data-provider`` must be specified as ``emdata``. A python glue class ``EMDataProvider`` in module ``python_util/convEMdata.py`` extends the cuda-convnet2 data provider class ``LabeledDataProvider`` and interfaces with ``EMDataParser`` so that image training and testing batches for the convnet are created in real time (happening in parallel on CPU while training occurs on GPU). Creating the batches in real time prevents having to pickle the batches. Also, since the batches are created in parallel with training, this method is essentially just as fast as having the data parsed directly on GPU memory and prevents the complication of implementing data parsing in CUDA code. Documentation below refers to a single presented image to the convnet, either for testing or for training as a *sample image*. The ``.ini`` file input to the ``EMDataParser`` is what determines how many samples images are presented per batch; batch is used here synonymously with the definition in cuda-convnet2.

# Training Example

Both the ``.ini`` files for the ``EMDataParser`` and layer definition and parameter ``.cfg`` [files](LayerParams.md) for convnet architectures used with EM data are stored under ``EM-layers-config``.  Here is a sample invocation for training a convnet using EM data:

```
python -u convnet.py --data-path=./EM-layers-config/samples/EMdata-3class-32x32out-ebal-huge-all-xyz.ini --save-path=/home/$(whoami)/Data/convnet_out/newestECS --test-range=200001-200002 --train-range=1-400 --layer-def=./EM-layers-config/samples/layers-EM-3class-32x32out.cfg --layer-params=./EM-layers-config/samples/layer-params-EM-3class-32x32out.cfg --data-provider=emdata --test-freq=80 --epochs=10 --gpu=0
```

This invocation requires access to the hdf5 files containing the raw EM volume and label volume corresponding to dataset M0007_33, ``M0007_33_39x35x7chunks_Forder.h5`` and ``M0007_33_labels_briggmankl_watkinspv_39x35x7chunks_Forder.h5``. Inputs are always using label volumes that are parallel to the raw EM data volumes. Only a small subset of the total label volume is ever expected to have manually created ground truth labels, as is the case in this example.

This invocation should produce output similar to this:
```
EMDataParser: config file ./EM-layers-config/samples/EMdata-3class-32x32out-ebal-huge-all-xyz.ini
EMDataParser: Chunk mode with 6 chunks of size 128 128 128:
	(0) chunk 17 19 2, offset 0 0 0
	(1) chunk 17 23 1, offset 0 0 0
	(2) chunk 22 23 1, offset 0 0 0
	(3) chunk 22 18 1, offset 0 0 0
	(4) chunk 22 23 2, offset 0 0 0
	(5) chunk 19 22 2, offset 0 0 0

	chunk_list_rand 0, chunk_range_rand 6
EMDataParser: Buffering data and labels chunk 17,19,2 offset 0,0,0
EMDataParser: Creating rand label lookup for specified zslices
EMDataParser: Creating tiled indices (typically for test and writing outputs)
EMDataParser: Initializing batch data pre-process with whiten none epslion 0.4, scalar mean 160.0000, scalar std 1.0000, overall normalize 0, pixel normalize 0, case normalize 0, em standard filter 0 (0 0) idelta (0.5000 0.5000) iinterp 1 gamma 0.8000
	actual EM u=150.6906810998917 s=61.73439612540292
	no batch pre-processing required
Initialized data layer 'data', producing 16384 outputs
Initialized data layer 'labels', producing 3072 outputs
```
indicating that the EMDataParser has been initialized and will be used to provide EM data to the convnet. This output is followed by the normal cuda-convnet output showing initializing of the particular network architecture and then the information printed for each training batch.

# Overview

The parser operates in two basic modes:
- all EM data is parsed out of a single continuous location within the hdf5 file, from here on referred to as a data *chunk*
- EM data is loaded from a chunk for a single batch, then loads another chunk, not necessarily contiguous with the last, for the next batch ("chunk mode")

These modes of operation allow for a compromise between:
(1) quick parsing from a single contiguous volume loaded into CPU memory during a single batch
(2) the ability to train on labels from different portions of the dataset

Some essential concepts for the EM data parser:
- raw EM data and label data hdf5 volumes are expected to be parallel
- cuda-convnet2 is inherently a 2D image classifier, so EM data is always parsed out from 2D slices of the EM data volumes that are parallel with one of the three orthogonal data planes (arbitrary plane reslicing is not supported)
- input images are square with size specified as an ini parameter
- multiple pixels can be classified in parallel (for a single example image) for a square (size also from ini) centered on the input image
- There are three basic ways that training or testing examples are selected from a particular loaded chunk. These different methods of selecting examples are controlled by specifying batch numbers in different ranges:

|batch range   | name |description|
|:-------------|:-----|:----------|
| 1-10000      | label-balanced | randomized batches that use a pre-generated lookup table for each batch so that selections from different types of labels can be balanced |
| 100001-20000 | randomized | randomized batches that do not use a lookup table, selected randomly from any location with the chunk |
| 200001-max   | tiled | sequential locations within and across chunks, typically used for writing output probabilities |

- label data is always presented as "segmented labels" meaning that different neurons or pieces of neurons are labeled with unique integer values
- selection of label-balanced sample images and how the label inputs should be interpreted for training purposes are done independently
- sample images (and corresponding labels) are augmented using combinations of simple image transformations (transpose and reflection)
- tiled batches create square or cube shaped tiles that cover the all the voxels in a chunk
- the size of the tilings for tiled batches determines the number of sample images per batch (num_cases_per_batch in cuda-convnet2 code)
- when [writing features](TrainingExample.md) from the convnet, features from the output layer (typically logistic regression) are created for tiled batches and then reassembled from the tiles into a volume of probabilities parallel to the EM data volume inputs

TODO: more info on reslicing
TODO: more info on augmentation
TODO: more info on logstic regression output
TODO: more info on label types (voxel classification versus affinity graphs)
TODO: more info on output bayesian reweighting

# Reference

The majority of the functionality of the EM data parser is exposed by the settings read out of the ``.ini`` file. The ``.ini`` file is read using the [configobj](http://www.voidspace.org.uk/python/articles/configobj.shtml) python module, which allows for an ini specification file to specify some basic checking on the input parameters. The ini specification file for the EM data parser, (``parseEMdataspec.ini``), contains types, allowable values and detailed description of each parameter and its usage.

TODO: more detail on major parameters with illustrations / image samples
