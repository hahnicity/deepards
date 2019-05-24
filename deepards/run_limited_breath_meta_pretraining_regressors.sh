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
                    ts python train_ards_detector.py --test-from-pickle /fastdata/deepards/padded_breath_by_breath_with_limited_bm_new-bm-dataset-c50-b500-test.pkl --train-from-pickle /fastdata/deepards/padded_breath_by_breath_with_limited_bm_new-bm-dataset-c50-b500-train.pkl -dt padded_breath_by_breath_with_limited_bm_target -n cnn_regressor --cuda -rip ${rip} -b ${bs} -e 10 --no-print-progress --save-model resnet18-bm-limited-c50-b500-rip${rip}-bs${bs}.pth -dp /fastdata/new_bm_dataset_c50_b500
                done
            done
        done
    done
done
