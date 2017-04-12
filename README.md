# electron microscopy data reconstruction pipeline (emdrp)

## Modified cuda-convnets2

Repository for code utilized in:

> Pallotto M, Watkins PV, Fubara B, Singer JH, Briggman KL. (2015)
> [Extracellular space preservation aids the connectomic analysis of neural circuits.](http://elifesciences.org/content/early/2015/12/09/eLife.08206)
> *Elife e08206.* Epub ahead of print

Sample run for training modified [cuda-convnet2](https://github.com/akrizhevsky/cuda-convnet2) for EM data:

```
python -u convnet.py --data-path=./emdrp-config/EMdata-3class-16x16out-ebal-huge-all-xyz.ini --save-path=../data --test-range=1-5 --train-range=1-200 --layer-def=./emdrp-config/layers-EM-3class-16x16out.cfg --layer-params=./emdrp-config/layer-params-EM-3class-16x16out.cfg --data-provider=emdata --test-freq=40 --epochs=10 --gpu=0
```
Works with cuda 7.5 and anaconda python2.3 plus additional conda [requirements](doc/setup/python2_conda_requirements.txt).

## Updated workflow

See full high level [documentation](doc/wiki/README.md).
