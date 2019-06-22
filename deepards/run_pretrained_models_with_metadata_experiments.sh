#!/bin/bash

for network in 'cnn_lstm'
do
    for base_network in 'senet18' 'senet154' 'se_resnet18' 'se_reset50' 'se_resnext50_32x4d'
    do
        for bs in 16
        do
            ts python train_ards_detector.py -p /fastdata/deepards/padded_breath_by_breath-nb100-kfold.pkl -dt padded_breath_by_breath -n $network --cuda -b ${bs} --load-pretrained ${base_network}-bm-limited-c50-b500-bs128.pth -e 10  --no-print-progress --kfolds 5 -exp with_senet_pretrained_models_c50_b500
        done
    done
done
