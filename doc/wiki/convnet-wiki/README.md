
# cuda convnet2 EM

## Documentation

A modified version of the [cuda-convnet2](https://code.google.com/p/cuda-convnet2/) is utilized for classifying EM raw data volumes.
TODO: this is dated, moved to neon, update this
Additionally, completely extra documentation has been added for:
- [parsing EM data](EMDataParser.md) for presenting image examples to the network and for extracting network output probabilities
- measuring [network performance](EMPerformance.md)

TODO: update wiki with added options and layers

TODO: add some description somewhere of the autoencoder architecture and usage

### Directory Structure

All EMDRP code for the convolutional network is under ``cuda-convnet2``.
The code is maintained with best attempts to have changes to cuda-convnet2 original files at the bottom of the files (new classes, etc) such that merges with any commits to the [github source](https://github.com/akrizhevsky/cuda-convnet2) can be more easily merged. For this reason, reformatting of the original source is not recommended. It is currently unclear how much longer cuda-convnet2 will be maintained on github; last commit in May 2015.

### Possible Replacements

- caffe
- torch
- tensorflow

TODO: fill this out with other options / advantages / tradeoffs, etc

TODO: document the cuda-convnet2 directory structure (Alex did not do this)

TODO: figure out why these pages are rendering differently than top level with Chrome plugin
