#!/bin/bash

for base_network in 'resnet18'
do
    for lc in 'all_breaths'
    do
        for lr in .001
        do
            for rip in 64
            do
                for bs in 8 16 32
                do
                    ts python train_ards_detector.py --train-to-pickle /fastdata/deepards/padded_breath_by_breath_with_flow_time_features-nb100-kfolds.pkl -dt  padded_breath_by_breath_with_flow_time_features -n cnn_lstm --cuda -rip ${rip} -b ${bs} -e 15 --no-print-progress --kfolds 5 -exp with_flow_time
                    ts python train_ards_detector.py -p /fastdata/deepards/padded_breath_by_breath_with_flow_time_features-nb100-kfolds.pkl -dt  padded_breath_by_breath_with_flow_time_features -n cnn_lstm --cuda -rip ${rip} -b ${bs} -e 15 --no-print-progress --kfolds 5 --bm-to-linear -exp with_flow_time_bm_to_linear
                done
            done
        done
    done
done
