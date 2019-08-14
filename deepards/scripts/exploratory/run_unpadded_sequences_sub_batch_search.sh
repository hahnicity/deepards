#!/bin/bash

for base_network in 'resnet18'
do
    for lc in 'all_breaths'
    do
        for lr in .001
        do
            for rip in 64
            do
                for bs in 16
                do
                    for nb in 25 50 75 100 150 225 300
                    do
                        ts python train_ards_detector.py --train-to-pickle unpadded_sequences-nb${nb}-kfold.pkl -dt unpadded_sequences -n cnn_lstm -nb ${nb} --cuda -rip ${rip} -b ${bs} -e 10 --no-print-progress --kfolds 5 -dp /fastdata/ardsdetection --loader-threads 4 -exp unpadded_sequences_sub_batch_search2
                    done
                done
            done
        done
    done
done