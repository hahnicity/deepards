#!/bin/bash

for base_network in 'resnet18'
do
    for lc in 'all_breaths'
    do
        for lr in .001
        do
            for bs in 8 16 32
            do
                for nb in 20 40
                do
                    ts python train_ards_detector.py --train-from-pickle unpadded_downsampled_sequences-nb${nb}-kfold.pkl -dt unpadded_downsampled_sequences -n cnn_lstm -nb ${nb} --cuda -b ${bs} -e 15 --no-print-progress --kfolds 5 --loader-threads 4 -exp pretrained_unpadded_downsampled_sequences --load-pretrained autoencoder-pretrained-basic-ae-b64-nb${nb}-bs16.pth
                done
            done
        done
    done
done
