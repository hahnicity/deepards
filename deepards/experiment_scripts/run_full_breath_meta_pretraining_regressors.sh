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
                    ts python train_ards_detector.py -dp /fastdata/new_bm_dataset -dt padded_breath_by_breath_with_full_bm_target -n cnn_regressor --cuda -rip ${rip} -b ${bs} -e 20 --no-print-progress --save-model resnet18-bm-full-rip${rip}-bs${bs}.pth --train-from-pickle padded_breath_by_breath_with_full_bm_target-train.pkl --test-from-pickle padded_breath_by_breath_with_full_bm_target-test.pkl
                done
            done
        done
    done
done
