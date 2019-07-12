#!/bin/bash

for base_network in 'resnet18'
do
    for bs in 16 32
    do
        for nb in 20 40 60 80
        do
            ts python train_ards_detector.py --train-from-pickle /fastdata/deepards/unpadded-centered-nb${nb}-kfold.pkl -dt unpadded_centered_sequences -n cnn_lstm -nb ${nb} --cuda -b ${bs} -e 10 --no-print-progress --kfolds 5 -dp /fastdata/ardsdetection --loader-threads 4 -exp unpadded_centered_sub_batch_search_with_pretraining --load-pretrained /home/greg/workspace/python/deepards/deepards/saved_models/${base_network}-bm-limited-c50-b500-bs128.pth
        done
    done
done
