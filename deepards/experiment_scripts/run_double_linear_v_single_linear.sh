#!/bin/bash


for base_network in 'resnet18' 'se_resnet18' 'densenet18'
do
    for network in 'cnn_linear' 'cnn_double_linear'
    do
        for bs in 16
        do
            ts python train_ards_detector.py --train-from-pickle /fastdata/deepards/padded_breath_by_breath-nb100-kfold.pkl -dt padded_breath_by_breath -n $network --cuda -b ${bs} -e 8  --no-print-progress --kfolds 5 -exp double_linear_eval --base-network $base_network
        done
    done
done
