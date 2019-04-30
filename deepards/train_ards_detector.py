from __future__ import print_function
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader

from deepards.metrics import DeepARDSResults, Reporting
from deepards.models.resnet import resnet18, resnet50, resnet101, resnet152
from deepards.models.torch_cnn_lstm_combo import CNNLSTMNetwork
from deepards.models.torch_cnn_bm_regressor import CNNRegressor
from deepards.models.torch_cnn_linear_network import CNNLinearNetwork
from deepards.dataset import ARDSRawDataset


class TrainModel(object):
    def __init__(self, args):
        self.args = args
        self.cuda_wrapper = lambda x: x.cuda() if args.cuda else x
        self.model_cuda_wrapper = lambda x: nn.DataParallel(x).cuda() if args.cuda else x
        if self.args.network == 'cnn_regressor':
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = torch.nn.BCELoss()

        self.is_classification = self.args.network != 'cnn_regressor'
        self.n_runs = self.args.kfolds if self.args.kfolds is not None else 1
        # Train and test both load from the same dataset in the case of kfold
        if self.n_runs > 1:
            self.args.test_to_pickle = None

        if self.args.save_model and self.n_runs > 1:
            raise NotImplementedError('We currently do not support saving kfold models')

        self.results = DeepARDSResults('{}_base{}_e{}_nb{}_lc{}_rip{}_lvp{}_rfpt{}_optim{}_lr{}_bs{}'.format(
            self.args.network,
            self.args.base_network,
            self.args.epochs,
            self.args.n_sub_batches,
            self.args.loss_calc,
            self.args.resnet_initial_planes,
            self.args.lstm_vote_percent,
            self.args.resnet_first_pool_type,
            self.args.optimizer,
            self.args.learning_rate,
            self.args.batch_size,
        ))

    def calc_loss(self, outputs, target):
        if self.args.loss_calc == 'all_breaths' and self.args.network == 'cnn_lstm':
            if self.args.batch_size > 1:
                target = target.unsqueeze(1)
            return self.criterion(outputs, target.repeat((1, self.args.n_sub_batches, 1)))
        elif self.args.loss_calc == 'last_breath' and self.args.network == 'cnn_lstm':
            return self.criterion(outputs[:, -1, :], target)
        else:
            return self.criterion(outputs, target)

    def run_train_epoch(self, model, train_loader, optimizer, epoch_num, run_num):
        n_loss = 0
        total_loss = 0
        with torch.enable_grad():
            print("\nrun epoch {}\n".format(epoch_num))
            for idx, (obs_idx, seq, target) in enumerate(train_loader):
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
                self.results.update_loss(run_num, loss.data)
                n_loss += 1
                if not self.args.no_print_progress:
                    print("batch num: {}/{}, avg loss: {}\r".format(idx+1, len(train_loader), total_loss/n_loss), end="")
                if self.args.debug:
                    break

    def run_test_epoch(self, model, test_loader, run_num):
        preds = []
        pred_idx = []
        with torch.no_grad():
            for idx, (obs_idx, seq, target) in enumerate(test_loader):
                inputs = self.cuda_wrapper(Variable(seq.float()))
                outputs = model(inputs)
                batch_preds = self._process_test_batch_predictions(outputs)
                if self.is_classification:
                    pred_idx.extend(obs_idx.cpu().tolist())
                    preds.extend(batch_preds)
                self._record_test_batch_results(batch_preds, target, run_num)

        if self.is_classification:
            preds = pd.Series(preds, index=pred_idx)
            preds = preds.sort_index()
        return preds

    def _record_test_batch_results(self, batch_preds, target, run_num):
        if self.is_classification:
            accuracy = accuracy_score(target.argmax(dim=1).cpu().tolist(), batch_preds)
            self.results.update_accuracy(run_num, accuracy)
        else:
            target = target.view((-1, 6)).cpu().numpy()
            mae = mean_absolute_error(target, batch_preds)
            self.results.update_meter('test_mae', run_num, mae)
            self.results.print_meter_results('test_mae', run_num)

    def _process_test_batch_predictions(self, outputs):
        if self.args.network == 'cnn_lstm':
            batch_preds = [
                1 if int(outputs[batch_num].argmax(dim=1).sum()) >= self.args.lstm_vote_percent else 0
                for batch_num in range(outputs.shape[0])
            ]

        elif self.args.network == 'cnn_linear':
            batch_preds = outputs.argmax(dim=1).cpu().tolist()

        elif self.args.network == 'cnn_regressor':
            batch_preds = outputs.view((-1, 6)).cpu().numpy()
        return batch_preds

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
            self.args.n_sub_batches,
            dataset_type=self.args.dataset_type,
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
            self.args.n_sub_batches,
            dataset_type=self.args.dataset_type,
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
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.args.batch_size,
                    shuffle=True,
                    pin_memory=self.args.cuda,
                    num_workers=self.args.loader_threads,
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.args.batch_size,
                    shuffle=True,
                    pin_memory=self.args.cuda,
                    num_workers=self.args.loader_threads,
                )
            yield train_loader, test_loader

    def train_and_test(self):
        for run_num, (train_loader, test_loader) in enumerate(self.get_splits()):
            base_network = {'resnet18': resnet18, 'resnet50': resnet50, 'resnet101': resnet101, 'resnet152': resnet152}[self.args.base_network]
            base_network = base_network(
                initial_planes=self.args.resnet_initial_planes,
                first_pool_type=self.args.resnet_first_pool_type,
            )
            model = self._get_model(base_network)
            optimizer = self._get_optimizer(model)
            for epoch in range(self.args.epochs):
                self.run_train_epoch(model, train_loader, optimizer, epoch+1, run_num)
                self._perform_testing(epoch, model, test_loader, run_num)

        if self.args.save_model:
            torch.save(model, self.args.save_model)

        if self.n_runs > 1 and self.is_classification:
            self.results.aggregate_classification_results()
        else:
            self.results.reporting.save_all()

    def _perform_testing(self, epoch_num, model, test_loader, run_num):
        if not self.args.no_test_after_epochs or epoch_num == self.args.epochs - 1:
            preds = self.run_test_epoch(model, test_loader, run_num)
            if self.args.network in ['cnn_lstm', 'cnn_linear']:
                y_test = test_dataset.get_ground_truth_df()
                self.results.perform_patient_predictions(y_test, preds, run_num)

    def _get_model(self, base_network):
        if self.args.network == 'cnn_lstm':
            model = self.model_cuda_wrapper(CNNLSTMNetwork(base_network))
        elif self.args.network == 'cnn_linear':
            model = self.model_cuda_wrapper(CNNLinearNetwork(base_network, self.args.n_sub_batches))
        elif self.args.network == 'cnn_regressor':
            model = self.model_cuda_wrapper(CNNRegressor(base_network))
        return model

    def _get_optimizer(self, model):
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=0.0001, nesterov=True)
        return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-path', default='/fastdata/ardsdetection', help='Path to ARDS detection dataset')
    parser.add_argument('-en', '--experiment-num', type=int, default=1)
    parser.add_argument('-c', '--cohort-file', default='cohort-description.csv')
    parser.add_argument('-n', '--network', choices=['cnn_lstm', 'cnn_linear', 'cnn_regressor'], default='cnn_lstm')
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('-p', '--train-from-pickle')
    parser.add_argument('--train-to-pickle')
    parser.add_argument('--test-from-pickle')
    parser.add_argument('--test-to-pickle')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('--base-network', choices=['resnet18', 'resnet50', 'resnet101', 'resnet152'], default='resnet18')
    parser.add_argument('-lc', '--loss-calc', choices=['all_breaths', 'last_breath'], default='all_breaths')
    parser.add_argument('-nb', '--n-sub-batches', type=int, default=100, help=(
        "number of breath-subbatches for each breath frame. This has different "
        "meanings for different dataset types. For breath_by_breath this means the "
        "number of individual breaths in each breath frame. For unpadded_sequences "
        "this means the number of contiguous flow measurements in each frame."))
    parser.add_argument('--no-print-progress', action='store_true')
    parser.add_argument('--kfolds', type=int)
    parser.add_argument('-rip', '--resnet-initial-planes', type=int, default=64)
    parser.add_argument('-rfpt', '--resnet-first-pool-type', default='max', choices=['max', 'avg'])
    parser.add_argument('--lstm-vote-percent', default=70, type=int)
    parser.add_argument('--no-test-after-epochs', action='store_true')
    parser.add_argument('--debug', action='store_true', help='debug code and dont train')
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='sgd')
    parser.add_argument('-dt', '--dataset-type', choices=['padded_breath_by_breath', 'unpadded_sequences', 'spaced_padded_breath_by_breath', 'stretched_breath_by_breath', 'padded_breath_by_breath_with_bm'], default='padded_breath_by_breath')
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--loader-threads', type=int, default=4)
    parser.add_argument('--save-model', help='save the model to a specific file')
    # XXX should probably be more explicit that we are using kfold or holdout in the future
    args = parser.parse_args()

    cls = TrainModel(args)
    cls.train_and_test()


if __name__ == "__main__":
    main()
