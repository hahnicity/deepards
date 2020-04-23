"""
correlation
~~~~~~~~~~~~~

Want to look through our data and see if auto-correlation or cross-correlation will
tell us anything about the data involved.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader


def autocorrelate_by_pred(model_data):
    cols = {
        #'ards_pred': 'ARDS-pred',
        #'non_ards_pred': 'Non-ARDS-pred',
        'ards_true': 'ARDS-true',
        'non_ards_true': 'Non-ARDS-true',
    }
    for col, name in cols.items():
        auto = []
        for seq in model_data[col]:
            seq = seq.numpy()
            out = np.correlate(seq.ravel(), seq.ravel(), mode='valid')
            auto.append(out)
        sns.distplot(auto, hist=False, label=name)

    plt.legend()
    plt.show()

    cols = {
        'ards_pred': 'ARDS-pred',
        'non_ards_pred': 'Non-ARDS-pred',
        #'ards_true': 'ARDS-true',
        #'non_ards_true': 'Non-ARDS-true',
    }
    for col, name in cols.items():
        auto = []
        for seq in model_data[col]:
            seq = seq.numpy()
            out = np.correlate(seq.ravel(), seq.ravel(), mode='valid')
            auto.append(out)
        sns.distplot(auto, hist=False, label=name)

    plt.legend()
    plt.show()


def run_entire_model(test_dataset_path, model_path):
    test_dataset = pd.read_pickle(test_dataset_path)
    test_dataset.transforms = None
    model = torch.load(model_path)
    model_data = {'ards_true': [], 'non_ards_true': [], 'ards_pred': [], 'non_ards_pred': []}
    softmax = nn.Softmax()

    with torch.no_grad():
        # use 256 because its a big number or something like that
        for data in DataLoader(test_dataset, batch_size=256):
            seq = data[1]
            target = data[3].argmax(dim=1)
            cuda_seq = seq.float().cuda()
            output = model(cuda_seq, None)
            batch_predictions = softmax(output).argmax(dim=1)
            for i, pred in enumerate(batch_predictions):
                if pred == 0:
                    model_data['non_ards_pred'].append(seq[i])
                else:
                    model_data['ards_pred'].append(seq[i])

                if target[i] == 0:
                    model_data['non_ards_true'].append(seq[i])
                else:
                    model_data['ards_true'].append(seq[i])

    return model_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='test set data to load.')
    parser.add_argument('model', help='Path to saved pytorch .pth model file')
    args = parser.parse_args()

    model_data = run_entire_model(args.file, args.model)
    autocorrelate_by_pred(model_data)


if __name__ == "__main__":
    main()
