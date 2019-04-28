#!/bin/bash

for base_network in 'resnet18'
do
    for lc in 'all_breaths' 'last_breath'
    do
        for lr in .0001
        do
            for rip in 16 64
            do
                ts python train_ards_detector.py -dp /home/greg/workspace/datasets/ardsdetection_data/ -p all-data-unpadded-seq-nb75.pkl -nb 75 -e 15 --kfolds 5 --cuda --no-print-progress -dt unpadded_sequences --base-network ${base_network} -rip ${rip} -b 16 -lr ${lr} -lc ${lc}
            done
        done
    done
done
