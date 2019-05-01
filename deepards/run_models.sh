#!/bin/bash

for base_network in 'resnet18'
do
    for lc in 'all_breaths'
    do
        for lr in .001
        do
            for rip in 8 16 64
            do
                for bs in 256
                do
                    ts python train_ards_detector.py -p padded_breath_by_breath_with_limited_bm-nb100-train.pkl --test-from-pickle padded_breath_by_breath_with_limited_bm-nb100-test.pkl -dt padded_breath_by_breath_with_limited_bm -n cnn_regressor -rdc --cuda -rip ${rip} -b ${bs} --save-model resnet18-bm-limited-rip${rip}-rdc.pth -rdc -e 12
                    ts python train_ards_detector.py -p padded_breath_by_breath_with_limited_bm-nb100-train.pkl --test-from-pickle padded_breath_by_breath_with_limited_bm-nb100-test.pkl -dt padded_breath_by_breath_with_limited_bm -n cnn_regressor -rdc --cuda -rip ${rip} -b ${bs} --save-model resnet18-bm-limited-rip${rip}.pth -e 12
                done
            done
        done
    done
done
