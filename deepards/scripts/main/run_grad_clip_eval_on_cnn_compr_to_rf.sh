#!/bin/bash


for base_network in 'densenet18'
do
    for network in 'cnn_linear_compr_to_rf'
    do
        for clip_val in .0001 .001 .01 .1 1
        do
            for lr in .001
            do
                ts python train_ards_detector.py --train-from-pickle /fastdata/deepards/padded_breath_by_breath-nb150-kfold.pkl --kfolds 5  --cuda --network ${network} -e 15 -nb 150 -lr ${lr} --clip-grad  --clip-val ${clip_val} -exp grad_clip_eval_with_cnn_compr_to_rf3
            done
        done
    done
done
