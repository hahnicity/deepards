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
from deepards.dataset import ARDSRawDataset


class TrainModel(object):
    def __init__(self, args):
        self.args = args
        self.cuda_wrapper = lambda x: x.cuda() if args.cuda else x
        self.criterion = torch.nn.BCELoss()
        self.n_runs = self.args.kfolds if self.args.kfolds is not None else 1
        # Train and test both load from the same dataset in the case of kfold
        if self.n_runs > 1:
            self.args.test_from_pickle = self.args.train_from_pickle
            self.args.test_to_pickle = None

    def run_train_epochs(self, model, train_loader, optimizer):
        n_loss = 0
        total_loss = 0
        with torch.enable_grad():
            for ep in range(self.args.epochs):
                print("\nrun epoch {}\n".format(ep+1))
                for idx, (obs_idx, patient, seq, target) in enumerate(train_loader):
                    hidden = model.init_hidden(seq.shape[0])
                    model.zero_grad()
                    target_shape = target.numpy().shape
                    target = self.cuda_wrapper(target.float())
                    inputs = self.cuda_wrapper(Variable(seq.float()))
                    outputs = model(inputs, hidden)
                    if self.args.loss_calc == 'all_breaths':
                        if self.args.batch_size > 1:
                            target = target.unsqueeze(1)
                        loss = self.criterion(outputs, target.repeat((1, self.args.n_breaths_in_seq, 1)))
                    elif self.args.loss_calc == 'last_breath':
                        loss = self.criterion(outputs[:, -1, :], target)
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
                hidden = model.init_hidden(seq.shape[0])
                inputs = self.cuda_wrapper(Variable(seq.float()))
                outputs = model(inputs, hidden)
                # get the last prediction in the LSTM chain. Just do this for now. Maybe
                # later we can have a slightly more sophisticated voting. Or we can just
                # skip all of that together and only backprop on the last item.
                preds.extend(outputs[:, -1, :].argmax(dim=1).cpu().tolist())
                pred_idx.extend(obs_idx.cpu().tolist())
        preds = pd.Series(preds, index=pred_idx)
        preds = preds.sort_index()
        return preds

    def train_and_test(self):
        results = DeepARDSResults()
        for i in range(self.n_runs):
            if self.n_runs > 1:
                print('--- Run Fold {} ---'.format(i+1))
            network_map = {'basic': CNNLSTMNetwork}
            base_network = {'resnet18': resnet18}[self.args.base_network]()
            model = self.cuda_wrapper(network_map[self.args.network](base_network))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            train_dataset = ARDSRawDataset(
                self.args.data_path,
                self.args.experiment_num,
                self.args.cohort_file,
                self.args.n_breaths_in_seq,
                to_pickle=self.args.train_to_pickle,
                from_pickle=self.args.train_from_pickle,
                kfold_num=i,
                total_kfolds=self.args.kfolds,
            )
            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
            self.run_train_epochs(model, train_loader, optimizer)
            test_dataset = ARDSRawDataset(
                self.args.data_path,
                self.args.experiment_num,
                self.args.cohort_file,
                self.args.n_breaths_in_seq,
                to_pickle=self.args.test_to_pickle,
                from_pickle=self.args.test_from_pickle,
                train=False,
                kfold_num=i,
                total_kfolds=self.args.kfolds,
            )
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
    parser.add_argument('-n', '--network', choices=['basic'], default='basic')
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('-p', '--train-from-pickle')
    parser.add_argument('--train-to-pickle')
    parser.add_argument('--test-from-pickle')
    parser.add_argument('--test-to-pickle')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('--base-network', choices=['resnet18'], default='resnet18')
    parser.add_argument('--loss-calc', choices=['all_breaths', 'last_breath'], required=True)
    parser.add_argument('-nb', '--n-breaths-in-seq', type=int, default=20)
    parser.add_argument('--no-print-progress', action='store_true')
    parser.add_argument('--kfolds', type=int)
    args = parser.parse_args()

    cls = TrainModel(args)
    cls.train_and_test()


if __name__ == "__main__":
    main()
