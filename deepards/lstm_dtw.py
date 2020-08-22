"""
lstm_dtw
~~~~~~~~

Compare LSTM with DTW
"""
import argparse
from pathlib import Path
import re

import pandas as pd
import torch
from torch.utils.data import DataLoader

from deepards.config import Configuration
from deepards.dataset import ARDSRawDataset
from deepards.train_ards_detector import build_parser, CNNLSTMModel


class LSTMDTW(object):
    def __init__(self, model1, model2, dataset):
        model_regex = re.compile(r'epoch(\d+)-fold(\d).pth')
        self.model1 = torch.load(model1)
        self.model2 = torch.load(model2)
        self.dataset = pd.read_pickle(dataset)
        self.dataset.transforms = None

        match1 = model_regex.search(model1)
        match2 = model_regex.search(model2)
        if not match1 or not match2:
            raise Exception('could not find epoch/fold match for the files provided. please make sure you are using kfold')
        self.epoch1, self.fold1 = map(int, match1.groups())
        self.epoch2, self.fold2 = map(int, match2.groups())

    def infer(self):
        test_dataset = ARDSRawDataset.make_test_dataset_if_kfold(self.dataset)
        test_dataset.set_kfold_indexes_for_fold(self.fold1)
        test_loader = DataLoader(test_dataset, 16, True, pin_memory=True)

        config_override = Path(__file__).parent.joinpath(
            'experiment_files/unpadded_20_len_sub_batch_cnn_lsm.yml'
        )
        args = build_parser().parse_args([])
        args.config_override = str(config_override)
        args = Configuration(args)
        cls = CNNLSTMModel(args)
        cls.run_test_epoch(self.epoch1, self.model1, test_dataset, test_loader, self.fold1)

        import IPython; IPython.embed()
        test_dataset.set_kfold_indexes_for_fold(self.fold2)
        test_loader = DataLoader(test_dataset, 16, True, pin_memory=True)
        cls.run_test_epoch(self.epoch2, self.model2, test_dataset, test_loader, self.fold2)
        import IPython; IPython.embed()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model1')
    parser.add_argument('model2')
    parser.add_argument('dataset', help='path to pickled dataset')
    args = parser.parse_args()

    cls = LSTMDTW(args.model1, args.model2, args.dataset)
    cls.infer()


if __name__ == "__main__":
    main()
