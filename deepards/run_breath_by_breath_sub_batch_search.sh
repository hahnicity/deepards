#!/bin/bash

for base_network in 'resnet18'
do
    for lc in 'all_breaths'
    do
        for lr in .001
        do
            for rip in 8 16 64
            do
                for bs in 16 32
                do
                    for nb in 20 40 50 100 200 400
                    do
                        ts python train_ards_detector.py --train-to-pickle padded_breath_by_breath-nb${nb}-kfold.pkl -dt padded_breath_by_breath -n cnn_lstm -nb ${nb} --cuda -rip ${rip} -b ${bs} -e 15 --no-print-progress --kfolds 5 -dp ~/workspace/datasets/ardsdetection_data
                    done
                done
            done
        done
    done
done
