#!/bin/bash

for network in 'cnn_transformer'
do
    for base_network in 'resnet18'
    do
        for bs in 16 32 64
        do
            # do a bit sparser search just for speed
            for n_blocks in 2 4 6 8 10
            do
                ts python train_ards_detector.py -p /fastdata/deepards/padded_breath_by_breath-nb100-kfold.pkl -dt padded_breath_by_breath -n $network --cuda -b ${bs} -e 8  --no-print-progress --kfolds 5 -exp cnn_transformer_block_eval --base-network $base_network --transformer-blocks $n_blocks
            done
        done
    done
done
