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
                    for valpha in .5 1 2
                    do
                        ts python train_ards_detector.py -p padded_breath_by_breath-nb100-kfold.pkl -dt padded_breath_by_breath -n cnn_lstm --cuda -rip ${rip} -b ${bs} -wd ${wd} --load-pretrained resnet18-bm-limited-rip64-bs32.pth -e 15  --no-print-progress --kfolds 5 -exp with_pretrained_weight_decay -loss vacillating --valpha ${valpha}
                    done
                done
            done
        done
    done
done
