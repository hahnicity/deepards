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
                    for nb in 100 200 400 600
                    do
                        ts python train_ards_detector.py -p /fastdata/deepards/padded_breath_by_breath-nb${nb}-kfold.pkl -dt padded_breath_by_breath -n cnn_lstm -nb ${nb} --cuda -rip ${rip} -b ${bs} -e 15 --no-print-progress --kfolds 5 -dp /fastdata/ardsdetection --loader-threads 1 -exp breath_by_breath_sub_batch_search
                    done
                done
            done
        done
    done
done
