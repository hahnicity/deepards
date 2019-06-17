from __future__ import print_function
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader

from deepards.dataset import ARDSRawDataset
from deepards.loss import VacillatingLoss
from deepards.metrics import DeepARDSResults, Reporting
from deepards.models.autoencoder_network import AutoencoderNetwork
from deepards.models.resnet import resnet18, resnet50, resnet101, resnet152
from deepards.models.torch_cnn_lstm_combo import CNNLSTMNetwork
from deepards.models.torch_cnn_bm_regressor import CNNRegressor
from deepards.models.torch_cnn_linear_network import CNNLinearNetwork
from deepards.models.torch_metadata_only_network import MetadataOnlyNetwork
from deepards.models.unet import UNet


class BaseTraining(object):
    base_networks = {
        'resnet18': resnet18,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
        'unet': UNet,
    }

    def __init__(self, args):
        self.args = args
        self.cuda_wrapper = lambda x: x.cuda() if args.cuda else x
        self.model_cuda_wrapper = lambda x: nn.DataParallel(x).cuda() if args.cuda else x
        self.is_classification = self.args.network not in ['autoencoder', 'cnn_regressor']

        if self.is_classification and self.args.loss_func == 'vacillating':
            self.criterion = VacillatingLoss(self.cuda_wrapper(torch.FloatTensor([self.args.valpha])))
        elif self.is_classification and self.args.loss_func == 'bce':
            self.criterion = torch.nn.BCELoss()
        else:
            self.criterion = torch.nn.MSELoss()

        if self.args.dataset_type == 'padded_breath_by_breath_with_limited_bm_target':
            self.n_bm_features = 3
        elif self.args.dataset_type == 'padded_breath_by_breath_with_full_bm_target':
            self.n_bm_features = 9

        if self.args.dataset_type == 'padded_breath_by_breath_with_flow_time_features':
            self.n_metadata_inputs = 9
        else:
            self.n_metadata_inputs = 0

        self.n_runs = self.args.kfolds if self.args.kfolds is not None else 1
        # Train and test both load from the same dataset in the case of kfold
        if self.n_runs > 1:
            self.args.test_to_pickle = None

        if self.args.save_model and self.n_runs > 1:
            raise NotImplementedError('We currently do not support saving kfold models')

        self.start_time = datetime.now().strftime('%s')
        self.results = DeepARDSResults(
            self.start_time,
            self.args.experiment_name,
            network=self.args.network,
            base_network=self.args.base_network,
            batch_size=self.args.batch_size,
            learning_rate=self.args.learning_rate,
            n_sub_batches=self.args.n_sub_batches,
            weight_decay=self.args.weight_decay,
            valpha=self.args.valpha,
        )
        print('Run start time: {}'.format(self.start_time))

    def run_train_epoch(self, model, train_loader, optimizer, epoch_num, fold_num):
        n_loss = 0
        total_loss = 0
        with torch.enable_grad():
            print("\nrun epoch {}\n".format(epoch_num))
            # In the future we can abstract away dataset specific outputs into separate
            # classes, but for now I'm not convinced that the refactor is worth it.
            for idx, (obs_idx, seq, metadata, target) in enumerate(train_loader):
                model.zero_grad()
                target_shape = target.numpy().shape
                target = self.cuda_wrapper(target.float())
                inputs = self.cuda_wrapper(Variable(seq.float()))
                metadata = self.cuda_wrapper(Variable(metadata.float()))
                outputs = model(inputs, metadata)
                loss = self.calc_loss(outputs, target, inputs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # print individual loss and total loss
                total_loss += loss.data
                self.results.update_loss(fold_num, loss.data)
                n_loss += 1
                # If the average loss jumps by > 100% then drop into debugger
                #if n_loss > 1 and (total_loss / n_loss) / ((total_loss-loss.data) / (n_loss-1)) > 1.5:
                #    import IPython; IPython.embed()
                if not self.args.no_print_progress:
                    print("batch num: {}/{}, avg loss: {}\r".format(idx+1, len(train_loader), total_loss/n_loss), end="")
                if self.args.debug:
                    break

    def get_base_datasets(self):
        # We are doing things this way by loading sequence information here so that
        # train and test datasets can access the same reference to the sequence array
        # stored in memory if we are using kfold. It is a bit awkward on the coding
        # side but it saves us memory.

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
            unpadded_downsample_factor=self.args.downsample_factor,
            drop_frame_if_frac_missing=self.args.no_drop_frames,
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

        # I can't easily use the train dataset as the test set because doing so would
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
            unpadded_downsample_factor=self.args.downsample_factor,
            drop_frame_if_frac_missing=self.args.no_drop_frames,
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
            yield train_dataset, train_loader, test_dataset, test_loader

    def train_and_test(self):
        for fold_num, (train_dataset, train_loader, test_dataset, test_loader) in enumerate(self.get_splits()):
            model = self.get_model()
            optimizer = self.get_optimizer(model)
            for epoch in range(self.args.epochs):
                self.run_train_epoch(model, train_loader, optimizer, epoch+1, fold_num)
                self.perform_testing(epoch, model, test_dataset, test_loader, fold_num)

        if self.args.save_model:
            torch.save(model, self.args.save_model)

        if self.is_classification:
            self.results.aggregate_classification_results()
        else:
            self.results.save_all()
        print('Run start time: {}'.format(self.start_time))

    def run_test_epoch(self, model, test_loader, fold_num):
        self.preds = []
        self.pred_idx = []
        self.epoch_targets = []
        with torch.no_grad():
            for idx, (obs_idx, seq, metadata, target) in enumerate(test_loader):
                inputs = self.cuda_wrapper(Variable(seq.float()))
                metadata = self.cuda_wrapper(Variable(metadata.float()))
                outputs = model(inputs, metadata)
                batch_preds = self._process_test_batch_results(outputs, target, inputs, fold_num)
                self.pred_idx.extend(self.transform_obs_idx(obs_idx, outputs).cpu().tolist())
                self.preds.extend(batch_preds)

        if self.is_classification:
            self.preds = pd.Series(self.preds, index=self.pred_idx)
            self.preds = self.preds.sort_index()
        else:
            self.results.print_meter_results('test_mae', fold_num)
        self._record_test_epoch_results(fold_num)

        # Never really makes sense to return a self.var unless its leaving the class..
        return self.preds

    def _record_test_epoch_results(self, fold_num):
        if self.is_classification:
            accuracy = accuracy_score(self.epoch_targets, self.preds)
            self.results.update_meter('epoch_test_accuracy', fold_num, accuracy)
        else:
            mae = mean_absolute_error(self.epoch_targets, self.preds)
            self.results.update_meter('epoch_test_mae', fold_num, mae)

    def get_base_network(self):
        base_network = self.base_networks[self.args.base_network]

        if self.args.load_pretrained:
            saved_model = torch.load(self.args.load_pretrained)
            if isinstance(saved_model, torch.nn.DataParallel):
                saved_model = saved_model.module
            base_network = saved_model.breath_block
            # XXX tmp debug
            base_network.n_out_filters = 512
        elif 'resnet' in self.args.base_network:
            base_network = base_network(
                initial_planes=self.args.initial_planes,
                first_pool_type=self.args.resnet_first_pool_type,
                double_conv_first=self.args.resnet_double_conv,
            )
        elif 'unet' in self.args.base_network:
            base_network = base_network(1)
        return base_network

    def get_optimizer(self, model):
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=self.args.weight_decay, nesterov=True)
        return optimizer

    def perform_testing(self, epoch_num, model, test_dataset, test_loader, fold_num):
        if not self.args.no_test_after_epochs or epoch_num == self.args.epochs - 1:
            preds = self.run_test_epoch(model, test_loader, fold_num)
            if self.is_classification:
                y_test = test_dataset.get_ground_truth_df()
                self.results.perform_patient_predictions(y_test, preds, fold_num, epoch_num)

    def get_model(self):
        base_network = self.get_base_network()
        return self.model_cuda_wrapper(self.get_network(base_network))

    def transform_obs_idx(self, obs_idx, outputs):
        return obs_idx


class ClassifierMixin(object):
    def record_testing_results(self, target, batch_preds, fold_num):
        accuracy = accuracy_score(target, batch_preds)
        self.results.update_accuracy(fold_num, accuracy)
        self.epoch_targets.extend(target.tolist())


class RegressorMixin(object):
    def record_testing_results(self, target, batch_preds, fold_num):
        mae = mean_absolute_error(target, batch_preds)
        self.results.update_meter('test_mae', fold_num, mae)
        r2 = r2_score(target, batch_preds)
        self.results.update_meter('r2', fold_num, r2)
        mse = mean_squared_error(target, batch_preds)
        self.results.update_meter('test_mse', fold_num, mse)
        self.epoch_targets.extend(target.tolist())


class CNNLSTMModel(BaseTraining, ClassifierMixin):
    def __init__(self, args):
        super(CNNLSTMModel, self).__init__(args)

    def calc_loss(self, outputs, target, inputs):
        if self.args.loss_calc == 'all_breaths':
            if self.args.batch_size > 1:
                target = target.unsqueeze(1)
            return self.criterion(outputs, target.repeat((1, self.args.n_sub_batches, 1)))
        elif self.args.loss_calc == 'last_breath':
            return self.criterion(outputs[:, -1, :], target)

    # One thing that is common in this function is that we process the outputs, process
    # the target, record testing results, and then return batch preds. It would be
    # so much easier if this all was just in one function
    def _process_test_batch_results(self, outputs, target, inputs, fold_num):
        batch_preds = outputs.argmax(dim=-1).cpu().view(-1)
        target = target.argmax(dim=1).cpu().reshape((outputs.shape[0], 1)).repeat((1, outputs.shape[1])).view(-1)
        self.record_testing_results(target, batch_preds, fold_num)
        return batch_preds.tolist()

    def get_network(self, base_network):
        return CNNLSTMNetwork(base_network, self.n_metadata_inputs, self.args.bm_to_linear, self.args.lstm_hidden_units)

    def transform_obs_idx(self, obs_idx, outputs):
        return obs_idx.reshape((outputs.shape[0], 1)).repeat((1, outputs.shape[1])).view(-1)


class CNNLinearModel(BaseTraining, ClassifierMixin):
    def __init__(self, args):
        super(CNNLinearModel, self).__init__(args)

    def calc_loss(self, outputs, target, inputs):
        return self.criterion(outputs, target)

    def _process_test_batch_results(self, outputs, target, inputs, fold_num):
        batch_preds = outputs.argmax(dim=-1).cpu()
        target = target.argmax(dim=1).cpu()
        self.record_testing_results(target, batch_preds, fold_num)
        return batch_preds.tolist()

    def get_network(self, base_network):
        return CNNLinearNetwork(base_network, self.args.n_sub_batches, self.n_metadata_inputs)


class MetadataOnlyModel(BaseTraining):
    def __init__(self, args):
        super(CNNMetadataModel, self).__init__(args)

    def calc_loss(self, outputs, target, inputs):
        return self.criterion(outputs, target)

    def _process_test_batch_results(self, outputs, target, inputs, fold_num):
        batch_preds = outputs.argmax(dim=-1).cpu()
        target = target.argmax(dim=1).cpu()
        self.record_testing_results(target, batch_preds, fold_num)
        return batch_preds.tolist()

    def get_network(self, _):
        return MetadataOnlyNetwork()


class CNNRegressorModel(BaseTraining):
    def __init__(self, args):
        super(CNNRegressorModel, self).__init__(args)

    def calc_loss(self, outputs, target, inputs):
        return self.criterion(outputs, target)

    def _process_test_batch_results(self, outputs, target, inputs, fold_num):
        batch_preds = outputs.cpu().numpy()
        target = target.cpu()
        self.record_testing_results(target, batch_preds, fold_num)
        return batch_preds.tolist()

    def get_network(self, base_network):
        return CNNRegressor(base_network, self.n_bm_features)


class AutoencoderModel(BaseTraining):
    def __init__(self, args):
        super(AutoencoderModel, self).__init__(args)

    def _process_test_batch_results(self, outputs, target, inputs, fold_num):
        batch_preds = outputs.cpu().numpy().squeeze(1)
        target = inputs.view((inputs.shape[0]*inputs.shape[1], inputs.shape[2], inputs.shape[3])).squeeze(1)
        self.record_testing_results(target, batch_preds, fold_num)
        return batch_preds.tolist()

    def calc_loss(self, outputs, target, inputs):
        return self.criterion(outputs, inputs.view((inputs.shape[0]*inputs.shape[1], inputs.shape[2], inputs.shape[3])))

    def get_network(self, base_network):
        return AutoencoderNetwork(base_network)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-path', default='/fastdata/ardsdetection', help='Path to ARDS detection dataset')
    parser.add_argument('-en', '--experiment-num', type=int, default=1)
    parser.add_argument('-c', '--cohort-file', default='cohort-description.csv')
    parser.add_argument('-n', '--network', choices=['cnn_lstm', 'cnn_linear', 'cnn_regressor', 'metadata_only', 'autoencoder'], default='cnn_lstm')
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('-p', '--train-from-pickle')
    parser.add_argument('--train-to-pickle')
    parser.add_argument('--test-from-pickle')
    parser.add_argument('--test-to-pickle')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('--base-network', choices=BaseTraining.base_networks, default='resnet18')
    parser.add_argument('-lc', '--loss-calc', choices=['all_breaths', 'last_breath'], default='all_breaths')
    parser.add_argument('-nb', '--n-sub-batches', type=int, default=100, help=(
        "number of breath-subbatches for each breath frame. This has different "
        "meanings for different dataset types. For breath_by_breath this means the "
        "number of individual breaths in each breath frame. For unpadded_sequences "
        "this means the number of contiguous flow measurements in each frame."))
    parser.add_argument('--no-print-progress', action='store_true')
    parser.add_argument('--kfolds', type=int)
    parser.add_argument('-rip', '--initial-planes', type=int, default=64)
    parser.add_argument('-rfpt', '--resnet-first-pool-type', default='max', choices=['max', 'avg'])
    parser.add_argument('--no-test-after-epochs', action='store_true')
    parser.add_argument('--debug', action='store_true', help='debug code and dont train')
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='sgd')
    parser.add_argument('-dt', '--dataset-type', choices=['padded_breath_by_breath', 'unpadded_sequences', 'unpadded_downsampled_sequences', 'spaced_padded_breath_by_breath', 'stretched_breath_by_breath', 'padded_breath_by_breath_with_full_bm_target', 'padded_breath_by_breath_with_limited_bm_target', 'padded_breath_by_breath_with_flow_time_features', 'unpadded_downsampled_autoencoder_sequences'], default='padded_breath_by_breath')
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--loader-threads', type=int, default=0, help='specify how many threads we should use to load data. Sometimes the threads fail to shutdown correctly though and this can cause memory errors. If this happens a restart works well')
    parser.add_argument('--save-model', help='save the model to a specific file')
    parser.add_argument('--load-pretrained', help='load breath block from a saved model')
    parser.add_argument('-rdc','--resnet-double-conv', action='store_true')
    parser.add_argument('--bm-to-linear', action='store_true')
    parser.add_argument('-exp', '--experiment-name')
    parser.add_argument('--downsample-factor', type=float, default=4.0)
    parser.add_argument('--no-drop-frames', action='store_false')
    parser.add_argument('-wd', '--weight-decay', type=float, default=.0001)
    parser.add_argument('-loss', '--loss-func', choices=['bce', 'vacillating'], default='bce', help='This option only works for classification. Choose the loss function you want to use for classification purposes: BCE or vacillating loss.')
    parser.add_argument('--valpha', type=float, default=float('Inf'), help='alpha value to use for vacillating loss. Lower alpha values mean vacillating loss will contribute less to overall loss of the system. Default value is inf')
    parser.add_argument('--lstm-hidden-units', type=int, default=512)
    args = parser.parse_args()

    network_map = {'cnn_lstm': CNNLSTMModel, 'cnn_linear': CNNLinearModel, 'cnn_regressor': CNNRegressorModel, 'metadata_only': MetadataOnlyModel, 'autoencoder': AutoencoderModel}
    cls = network_map[args.network](args)
    cls.train_and_test()


if __name__ == "__main__":
    main()
