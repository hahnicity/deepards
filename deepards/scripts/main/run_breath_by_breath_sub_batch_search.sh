#!/bin/bash

for base_network in 'densenet18'
do
    for lc in 'all_breaths'
    do
        for lr in .001
        do
            for rip in 64
            do
                for bs in 16
                do
                    for nb in 50 100 150 200 300 400 500 600
                    do
                        ts python train_ards_detector.py --train-from-pickle /fastdata/deepards/padded_breath_by_breath-nb${nb}-kfold.pkl -dt padded_breath_by_breath -n cnn_linear_compr_to_rf -nb ${nb} --cuda -rip ${rip} -b ${bs} -e 15 --no-print-progress --kfolds 5 -dp /fastdata/ardsdetection --loader-threads 1 -exp breath_by_breath_sub_batch_search_main_experiment
                    done
                done
            done
        done
    done
done
