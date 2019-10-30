#!/bin/bash


for base_network in 'densenet18'
do
    for network in 'cnn_to_nested_transformer'
    do
        for clip_val in .00001 .0001 .001 .01 .1 1
        do
            for lr in .001 .0001
            do
                ts python train_ards_detector.py --train-from-pickle /fastdata/deepards/unpadded_centered_sequences-nb20-prototrain-whole-patient.pkl --test-from-pickle /fastdata/deepards/unpadded_centered_sequences-nb20-prototest-whole-patient.pkl  --cuda --network ${network} -e 40 -nb 20  --load-base-network unpadded_centered_majority_vote_cnn_transformer_densenet18_e5.pth --freeze-base-network -lr ${lr} --clip-grad  --clip-val ${clip_val} -exp grad_clip_eval
            done
        done
    done
done
