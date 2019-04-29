#!/bin/bash

for base_network in 'resnet18'
do
    for lc in 'all_breaths'
    do
        for lr in .001
        do
            for rip in 8 16 64
            do
                for bs in 16 32 64
                do
                    ts python train_ards_detector.py -dp /home/greg/workspace/datasets/ardsdetection_data/ -p stretched_breath_by_breath-nb100-kfold.pkl -nb 100 -e 10 --kfolds 5 --cuda --no-print-progress -dt unpadded_sequences --base-network ${base_network} -rip ${rip} -b ${bs} -lr ${lr} -lc ${lc}
                done
            done
        done
    done
done
