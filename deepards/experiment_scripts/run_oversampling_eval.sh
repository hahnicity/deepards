#!/bin/bash


for base_network in 'resnet18' 'densenet18'
do
    for network in 'cnn_linear' 'cnn_transformer'
    do
        for bs in 16
        do
            ts python train_ards_detector.py --train-from-pickle /fastdata/deepards/padded_breath_by_breath-nb100-kfold.pkl -dt padded_breath_by_breath -n $network --cuda -b ${bs} -e 10  --no-print-progress --kfolds 5 -exp oversampling_eval --base-network $base_network --oversample --reshuffle-oversample-per-epoch
        done
    done
done
