#!/bin/bash

for base_network in 'resnet18'
do
    for lc in 'all_breaths'
    do
        for lr in .001
        do
            for rip in 64
            do
                for bs in 16 32
                do
                    for nb in 20 40 60 80
                    do
                        ts python train_ards_detector.py --train-to-pickle unpadded_downsampled_sequences-nb${nb}-kfold.pkl -dt unpadded_downsampled_sequences -n cnn_lstm -nb ${nb} --cuda -rip ${rip} -b ${bs} -e 15 --no-print-progress --kfolds 5 -dp ~/workspace/datasets/ardsdetection_data --loader-threads 4 -exp unpadded_downsampled_sequences_sub_batch_search
                    done
                done
            done
        done
    done
done
