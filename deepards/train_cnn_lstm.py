from __future__ import print_function
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader

from deepards.metrics import DeepARDSResults
from deepards.models.resnet import resnet18
from deepards.models.torch_cnn_lstm_combo import CNNLSTMNetwork
from deepards.models.torch_cnn_linear_network import CNNLinearNetwork
from deepards.dataset import ARDSRawDataset


class TrainModel(object):
    def __init__(self, args):
        self.args = args
        self.cuda_wrapper = lambda x: x.cuda() if args.cuda else x
        self.model_cuda_wrapper = lambda x: nn.DataParallel(x).cuda() if args.cuda else x
        self.criterion = torch.nn.BCELoss()

        self.n_runs = self.args.kfolds if self.args.kfolds is not None else 1
        # Train and test both load from the same dataset in the case of kfold
        if self.n_runs > 1:
            self.args.test_to_pickle = None

    def calc_loss(self, outputs, target):
        if self.args.loss_calc == 'all_breaths' and self.args.network == 'cnn_lstm':
            if self.args.batch_size > 1:
                target = target.unsqueeze(1)
            return self.criterion(outputs, target.repeat((1, self.args.n_breaths_in_seq, 1)))
        elif self.args.loss_calc == 'last_breath' and self.args.network == 'cnn_lstm':
            return self.criterion(outputs[:, -1, :], target)
        else:
            return self.criterion(outputs, target)

    def run_train_epochs(self, model, train_loader, optimizer):
        n_loss = 0
        total_loss = 0
        with torch.enable_grad():
            for ep in range(self.args.epochs):
                print("\nrun epoch {}\n".format(ep+1))
                for idx, (obs_idx, patient, seq, target) in enumerate(train_loader):
                    model.zero_grad()
                    target_shape = target.numpy().shape
                    target = self.cuda_wrapper(target.float())
                    inputs = self.cuda_wrapper(Variable(seq.float()))
                    outputs = model(inputs)
                    loss = self.calc_loss(outputs, target)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # print individual loss and total loss
                    total_loss += loss.data
                    n_loss += 1
                    if not self.args.no_print_progress:
                        print("batch num: {}/{}, avg loss: {}\r".format(idx+1, len(train_loader), total_loss/n_loss), end="")

    def run_test_epoch(self, model, test_loader):
        preds = []
        pred_idx = []
        with torch.no_grad():
            for idx, (obs_idx, patient, seq, target) in enumerate(test_loader):
                inputs = self.cuda_wrapper(Variable(seq.float()))
                outputs = model(inputs)
                # get the last prediction in the LSTM chain. Just do this for now. Maybe
                # later we can have a slightly more sophisticated voting. Or we can just
                # skip all of that together and only backprop on the last item.
                if self.args.network == 'cnn_lstm':
                    outputs = outputs[:, -1, :]
                preds.extend(outputs.argmax(dim=1).cpu().tolist())
                pred_idx.extend(obs_idx.cpu().tolist())
        preds = pd.Series(preds, index=pred_idx)
        preds = preds.sort_index()
        return preds

    def get_base_datasets(self):
        # We are doing things this way by loading sequence information here so that
        # train and test datasets can access the same reference to the sequence array
        # stored in memory if we are using kfold. It is a bit awkward on the coding
        # side but it saves us memory.
        #
        # XXX in future this function should probably handle to_pickle as well. Either
        # that or we just have a separate function that handles pickling

        # for holdout and kfold
        if self.args.train_from_pickle:
            train_sequences = pd.read_pickle(self.args.train_from_pickle)
        # no pickle
        else:
            train_sequences = []

        kfold_num = None if self.args.kfolds is None else 0
        train_dataset = ARDSRawDataset(
            self.args.data_path,
            self.args.experiment_num,
            self.args.cohort_file,
            self.args.n_breaths_in_seq,
            all_sequences=train_sequences,
            to_pickle=self.args.train_to_pickle,
            kfold_num=kfold_num,
            total_kfolds=self.args.kfolds,
        )
        # for holdout
        if self.args.test_from_pickle and self.args.kfolds is None:
            test_sequences = pd.read_pickle(self.args.test_from_pickle)
        # for kfold
        elif self.args.kfolds is not None:
            test_sequences = train_dataset.all_sequences
        # holdout, no pickle, no kfolds
        else:
            test_sequences = []

        # I can't easily the train dataset as the test set because doing so would
        # involve changing internal propeties on the train set
        test_dataset = ARDSRawDataset(
            self.args.data_path,
            self.args.experiment_num,
            self.args.cohort_file,
            self.args.n_breaths_in_seq,
            all_sequences=test_sequences,
            to_pickle=self.args.test_to_pickle,
            train=False,
            kfold_num=kfold_num,
            total_kfolds=self.args.kfolds,
        )
        return train_dataset, test_dataset

    def get_splits(self):
        train_dataset, test_dataset = self.get_base_datasets()
        for i in range(self.n_runs):
            if self.args.kfolds is not None:
                print('--- Run Fold {} ---'.format(i+1))
                train_dataset.get_kfold_indexes_for_fold(i)
                test_dataset.get_kfold_indexes_for_fold(i)
            yield train_dataset, test_dataset

    def train_and_test(self):
        results = DeepARDSResults()
        for train_dataset, test_dataset in self.get_splits():
            if self.args.base_network == 'resnet18':
                base_network = resnet18(initial_planes=self.args.resnet_initial_planes)

            if self.args.network == 'cnn_lstm':
                model = self.model_cuda_wrapper(CNNLSTMNetwork(base_network))
            elif self.args.network == 'cnn_linear':
                model = self.model_cuda_wrapper(CNNLinearNetwork(base_network, self.args.n_breaths_in_seq))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
            self.run_train_epochs(model, train_loader, optimizer)
            test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=True)
            y_test = test_dataset.get_ground_truth_df()
            preds = self.run_test_epoch(model, test_loader)
            results.perform_patient_predictions(y_test, preds)
        results.aggregate_all_results()
        print(results.aggregate_stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-path', default='/fastdata/ardsdetection', help='Path to ARDS detection dataset')
    parser.add_argument('-en', '--experiment-num', type=int, default=1)
    parser.add_argument('-c', '--cohort-file', default='cohort-description.csv')
    parser.add_argument('-n', '--network', choices=['cnn_lstm', 'cnn_linear'], default='cnn_lstm')
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('-p', '--train-from-pickle')
    parser.add_argument('--train-to-pickle')
    parser.add_argument('--test-from-pickle')
    parser.add_argument('--test-to-pickle')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('--base-network', choices=['resnet18'], default='resnet18')
    parser.add_argument('--loss-calc', choices=['all_breaths', 'last_breath'], default='all_breaths')
    parser.add_argument('-nb', '--n-breaths-in-seq', type=int, default=20)
    parser.add_argument('--no-print-progress', action='store_true')
    parser.add_argument('--kfolds', type=int)
    parser.add_argument('--resnet-initial-planes', type=int, default=64)
    # XXX should probably be more explicit that we are using kfold or holdout in the future
    args = parser.parse_args()

    cls = TrainModel(args)
    cls.train_and_test()


if __name__ == "__main__":
    main()
