#!/bin/bash

for base_network in 'resnet18'
do
    for lc in 'all_breaths'
    do
        for lr in .001
        do
            for bs in 16
            do
                for beta in .25 .5 1 2 3 4 5
                do
                    ts python train_ards_detector.py -p /fastdata/deepards/padded_breath_by_breath-nb100-kfold.pkl -dt padded_breath_by_breath -n cnn_lstm_double_linear --cuda -b ${bs} -e 5 --no-print-progress --kfolds 5 -exp confidence_penalty3 -loss confidence --conf-beta $beta
                done
            done
        done
    done
done
