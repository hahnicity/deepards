#!/bin/bash

for network in 'cnn_linear' 'cnn_lstm'
do
    for base_network in 'densenet121' 'densenet161' 'densenet201'
    do
        for bs in 16
        do
            ts python train_ards_detector.py -p /fastdata/deepards/padded_breath_by_breath-nb100-kfold.pkl -dt padded_breath_by_breath -n $network --cuda -b ${bs} --load-pretrained ${base_network}-bm-limited-c50-b500-bs128.pth -e 15  --no-print-progress --kfolds 5 -exp with_densenet_pretrained_models_c50_b500
        done
    done
done
