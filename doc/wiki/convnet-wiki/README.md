
# Convolutional Neural Networks for EM data

## Documentation

Currently the EMDRP is using [Intel Nervana neon](https://github.com/NervanaSystems/neon) with a harness for processing EM data. This harness extends several aspects of the neon python classes for easy processing of EM data, including a separate [top-level script](../../../neon3/emneon.py) for running the convnets.

The main extension of neon is for loading and augmenting EM data and converting to neon format, and processing outputs to reconstruct probability maps. Additional information is available on:
- [parsing EM data](EMDataParser.md) for presenting image examples to the network and for extracting network output probabilities
- measuring [network performance](EMPerformance.md)

TODO: more detail here on how neon is extended and description of directory structure

### Other options

The original emdrp machine voxel classifier was using a [modified version](../../cuda-convnet2) of [cuda-convnet2](https://github.com/akrizhevsky/cuda-convnet2). The initial EM data parser implementations were thus written and still maintained as a single python class. Thus switching to other convnet implementations that do not easily support a python extension for importing data would be more difficult to port to.

Initial investigations into using [Google Tensorflow](https://github.com/tensorflow/tensorflow) indicated that TF is much slower and uses more GPU memory than other options. Additionally, initial attempts indicated that the hurdle of modifying TF for processing other data formats is much higher; i.e. requiring something more than a python module (aka cpp code).

Other options:
- caffe
- theano
- torch

TODO: fill this out with other options / advantages / tradeoffs, etc
