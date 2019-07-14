#!/bin/bash

for network in 'cnn_transformer'
do
    for base_network in 'resnet18' 'senet18' 'se_resnet18' 'densenet18'
    do
        for bs in 16 32 64
        do
            ts python train_ards_detector.py -p /fastdata/deepards/padded_breath_by_breath-nb100-kfold.pkl -dt padded_breath_by_breath -n $network --cuda -b ${bs} -e 10  --no-print-progress --kfolds 5 -exp cnn_transformer_model_eval --base-network $base_network
        done
    done
done
