#!/bin/bash

ts python train_ards_detector.py --train-to-pickle /fastdata/deepards/padded_breath_by_breath-nb100-kfold.pkl -dt padded_breath_by_breath -n cnn_lstm --cuda -b 16 -e 5  --no-print-progress --kfolds 5 -exp double_linear_eval --base-network resnet18
ts python train_ards_detector.py --train-from-pickle /fastdata/deepards/padded_breath_by_breath-nb100-kfold.pkl -dt padded_breath_by_breath -n cnn_lstm_double_linear --cuda -b 16 -e 5  --no-print-progress --kfolds 5 -exp double_linear_eval --base-network resnet18

for network in 'cnn_lstm' 'cnn_lstm_double_linear'
do
    for base_network in 'se_resnet18' 'densenet18'
    do
        for bs in 16
        do
            ts python train_ards_detector.py --train-from-pickle /fastdata/deepards/padded_breath_by_breath-nb100-kfold.pkl -dt padded_breath_by_breath -n $network --cuda -b ${bs} -e 5  --no-print-progress --kfolds 5 -exp double_linear_eval --base-network $base_network
        done
    done
done
