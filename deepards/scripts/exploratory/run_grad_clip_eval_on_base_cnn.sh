#!/bin/bash


for base_network in 'densenet18'
do
    for network in 'cnn_linear_compr_to_rf'
    do
        for clip_val in .00001 .0001 .001 .01 .1 1
        do
            for lr in .001 .0001
            do
                ts python train_ards_detector.py --train-from-pickle XXX --kfolds 5  --cuda --network ${network} -e 20 -nb XXX -lr ${lr} --clip-grad  --clip-val ${clip_val} -exp grad_clip_eval_with_base_network
            done
        done
    done
done
