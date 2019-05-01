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
                    ts python train_ards_detector.py -p /fastdata/deepards/padded_breath_by_breath-nb100-kfold.pkl -dt padded_breath_by_breath -n cnn_lstm --cuda -rip ${rip} -b ${bs} --load-pretrained resnet18-bm-limited-rip${rip}-rdc.pth -e 15 -rdc --no-print-progress
                    ts python train_ards_detector.py -p /fastdata/deepards/padded_breath_by_breath-nb100-kfold.pkl -dt padded_breath_by_breath -n cnn_lstm --cuda -rip ${rip} -b ${bs} --load-pretrained resnet18-bm-limited-rip${rip}.pth -e 15 --no-print-progress
                done
            done
        done
    done
done
