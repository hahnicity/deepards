#!/bin/bash

for base_network in 'densenet121' 'densenet161' 'densenet201'
do
    for bs in 64 128 256
    do
        ts python train_ards_detector.py --test-from-pickle /fastdata/deepards/padded_breath_by_breath_with_limited_bm_new-bm-dataset-c50-b500-test.pkl --train-from-pickle /fastdata/deepards/padded_breath_by_breath_with_limited_bm_new-bm-dataset-c50-b500-train.pkl -dt padded_breath_by_breath_with_limited_bm_target -n cnn_regressor --cuda -b ${bs} -e 10 --no-print-progress --save-model ${base_network}-bm-limited-c50-b500-bs${bs}.pth
    done
done
