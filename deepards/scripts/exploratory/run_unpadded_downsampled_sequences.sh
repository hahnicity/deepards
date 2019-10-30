#!/bin/bash

for base_network in 'densenet18'
do
    for lc in 'all_breaths'
    do
        for lr in .001
        do
            for bs in 16
            do
                for nb in 10 20 30 40 50
                do
                    ts python train_ards_detector.py --train-to-pickle unpadded_downsampled_sequences-nb${nb}-kfold.pkl -dt unpadded_downsampled_sequences -n cnn_linear -nb ${nb} --cuda -b ${bs} -e 15 --no-print-progress --kfolds 5 --loader-threads 4 -exp non_pretrained_unpadded_downsampled_sequences_eval
                done
            done
        done
    done
done
