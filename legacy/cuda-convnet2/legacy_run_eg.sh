python -u convnet.py --data-path=./emdrp-config/EMdata-3class-16x16out-ebal-huge-all-xyz.ini --save-path=../data --test-range=1-5 --train-range=1-200 --layer-def=./emdrp-config/layers-EM-3class-16x16out.cfg --layer-params=./emdrp-config/layer-params-EM-3class-16x16out.cfg --data-provider=emdata --test-freq=40 --epochs=10 --gpu=0

