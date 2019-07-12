#!/bin/bash

for base_network in 'resnet18'
do
    for bs in 8 16
    do
        for nb in 10 20 30 40
        do
            ts python train_ards_detector.py --train-to-pickle /fastdata/deepards/unpadded-centered-downsampled-nb${nb}-kfold.pkl -dt unpadded_centered_downsampled_sequences -n cnn_lstm -nb ${nb} --cuda -b ${bs} -e 10 --no-print-progress --kfolds 5 -dp /fastdata/ardsdetection --loader-threads 4 -exp unpadded_centered_downsampled_sub_batch_search
        done
    done
done
