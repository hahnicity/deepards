#!/bin/bash

for nb in 20 40
do
	for bs in 8 16 32
	do
		ts python train_ards_detector.py -dt unpadded_downsampled_autoencoder_sequences --no-drop-frames -nb ${nb} --train-to-pickle unpadded_downsampled_autoencoder_sequences-nb${nb}-no-drop-train.pkl --test-to-pickle unpadded_downsampled_autoencoder_sequences-nb${nb}-no-drop-test.pkl -n autoencoder --base-network unet -dp /fastdata/autoencoder_dataset_b64 --cuda -b ${bs} --no-print-progress --save-model autoencoder-pretrained-b64-nb${nb}-bs${bs}.pth --downsample-factor 4 -exp autoencoder_pretraining -e 15
	done
done
