#!/bin/bash

for base_network in 'resnet18'
do
    for lc in 'all_breaths'
    do
        for lr in .001
        do
            for rip in 64
            do
                for bs in 8 16 32 64
                do
                    ts python train_ards_detector.py -p /fastdata/deepards/padded_breath_by_breath-nb100-kfold.pkl -dt padded_breath_by_breath -n cnn_lstm --cuda -rip ${rip} -b ${bs} --load-pretrained resnet18-bm-limited-c50-b500-rip64-bs32.pth -e 15  --no-print-progress --kfolds 5 -exp with_pretrained_models_c50_b500
                done
            done
        done
    done
done
