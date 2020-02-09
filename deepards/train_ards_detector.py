from __future__ import print_function
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from deepards.dataset import ARDSRawDataset, SiameseNetworkDataset
from deepards.loss import ConfidencePenaltyLoss, FocalLoss, VacillatingLoss
from deepards.metrics import DeepARDSResults, Reporting
from deepards.models.autoencoder_cnn import AutoencoderCNN
from deepards.models.autoencoder_network import AutoencoderNetwork
from deepards.models.cnn_to_nested_layer import CNNToNestedLSTMNetwork, CNNToNestedRNNNetwork, CNNToNestedTransformerNetwork
from deepards.models.cnn_transformer import CNNTransformerNetwork
from deepards.models.densenet import densenet18, densenet121, densenet161, densenet169, densenet201
from deepards.models.lstm_only import LSTMOnlyNetwork
from deepards.models.resnet import resnet18, resnet50, resnet101, resnet152
from deepards.models.senet import senet18, senet154, se_resnet18, se_resnet50, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d
from deepards.models.siamese import SiameseARDSClassifier, SiameseCNNLinearNetwork, SiameseCNNLSTMNetwork, SiameseCNNTransformerNetwork
from deepards.models.torch_cnn_lstm_combo import CNNLSTMDoubleLinearNetwork, CNNLSTMNetwork
from deepards.models.torch_cnn_bm_regressor import CNNRegressor
from deepards.models.torch_cnn_linear_network import CNNDoubleLinearNetwork, CNNLinearComprToRF, CNNLinearNetwork, CNNSingleBreathLinearNetwork
from deepards.models.torch_metadata_only_network import MetadataOnlyNetwork
from deepards.models.unet import UNet
from deepards.models.vgg import vgg11_bn, vgg13_bn

torch.set_default_tensor_type('torch.FloatTensor')
base_networks = {
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'unet': UNet,
    'densenet18': densenet18,
    'densenet121': densenet121,
    'densenet161': densenet161,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'basic_cnn_ae': AutoencoderCNN,
    'senet18': senet18,
    'senet154': senet154,
    'se_resnet18': se_resnet18,
    'se_resnet50': se_resnet50,
    'se_resnet101': se_resnet101,
    'se_resnet152': se_resnet152,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'se_resnext101_32x4d': se_resnext101_32x4d,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
}


class BaseTraining(object):

    def __init__(self, args):
        self.args = args
        self.cuda_wrapper = lambda x: x.cuda() if args.cuda else x
        if self.args.debug:
            self.model_cuda_wrapper = lambda x: x.cuda() if args.cuda else x
        else:
            self.model_cuda_wrapper = lambda x: nn.DataParallel(x).cuda() if args.cuda else x
        self.set_loss_criterion()

        if self.args.dataset_type == 'padded_breath_by_breath_with_limited_bm_target':
            self.n_bm_features = 3
        if self.args.dataset_type == 'padded_breath_by_breath_with_experimental_bm_target':
            self.n_bm_features = 7
        elif self.args.dataset_type == 'padded_breath_by_breath_with_full_bm_target':
            self.n_bm_features = 9

        if self.args.dataset_type == 'padded_breath_by_breath_with_flow_time_features':
            self.n_metadata_inputs = 9
        else:
            self.n_metadata_inputs = 0

        if self.args.unshuffled and self.args.batch_size > 1:
            raise Exception('Currently we can only run unshuffled runs with a batch size of 1!')

        self.n_runs = self.args.kfolds if self.args.kfolds is not None else 1
        # Train and test both load from the same dataset in the case of kfold
        if self.n_runs > 1:
            self.args.test_to_pickle = None

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
            confidence_beta=self.args.conf_beta,
            dataset_type=self.args.dataset_type,
            grad_clip_val=self.args.clip_val,
        )
        print('Run start time: {}'.format(self.start_time))

    def run_train_epoch(self, model, train_loader, optimizer, epoch_num, fold_num):
        with torch.enable_grad():
            print("\nrun epoch {}\n".format(epoch_num))
            for idx, (obs_idx, seq, metadata, target) in enumerate(train_loader):
                model.zero_grad()
                if not self.args.batch_size == 1:
                    obs_idx, seq, metadata, target = self.clip_odd_batch_sizes(obs_idx, seq, metadata, target)
                if seq.shape[0] == 0:
                    continue
                target = self.cuda_wrapper(target.float())
                inputs = self.cuda_wrapper(Variable(seq.float()))
                metadata = self.cuda_wrapper(Variable(metadata.float()))
                outputs = model(inputs, metadata)
                self.handle_train_optimization(optimizer, outputs, target, inputs, fold_num, len(train_loader), idx, epoch_num, model)
                if self.args.stop_on_loss and self.results.get_meter('loss', fold_num).peek() > self.args.stop_thresh and epoch_num > self.args.stop_after_epoch:
                    print('stop on loss')
                    import IPython; IPython.embed()
                if self.args.debug:
                    break

    def handle_train_optimization(self, optimizer, outputs, target, inputs, fold_num, total_batches, batch_idx, epoch_num, model):
        loss = self.calc_loss(outputs, target, inputs)
        loss.backward()
        # Can also put grad clip here if you wanted too. Example:
        # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
        optimizer.step()
        optimizer.zero_grad()
        self.results.update_meter('loss_epoch_{}'.format(epoch_num), fold_num, loss)
        self.results.update_loss(fold_num, loss.data)

        # print individual loss and total loss
        if not self.args.no_print_progress:
            print("batch num: {}/{}, avg loss: {}\r".format(batch_idx, total_batches, self.results.get_meter('loss_epoch_{}'.format(epoch_num), fold_num), end=""))

    def get_base_datasets(self):
        kfold_num = None if self.args.kfolds is None else 0

        # for holdout and kfold
        if not self.args.train_from_pickle:
            train_dataset = ARDSRawDataset(
                self.args.data_path,
                self.args.experiment_num,
                self.args.cohort_file,
                self.args.n_sub_batches,
                dataset_type=self.args.dataset_type,
                all_sequences=[],
                to_pickle=self.args.train_to_pickle,
                kfold_num=kfold_num,
                total_kfolds=self.args.kfolds,
                unpadded_downsample_factor=self.args.downsample_factor,
                drop_frame_if_frac_missing=self.args.no_drop_frames,
                oversample_minority=self.args.oversample,
            )
        else:
            train_dataset = ARDSRawDataset.from_pickle(self.args.train_from_pickle, self.args.oversample)

        self.n_sub_batches = train_dataset.n_sub_batches
        if not self.args.test_from_pickle and (self.args.kfolds is not None):
            test_dataset = ARDSRawDataset.make_test_dataset_if_kfold(train_dataset)
        elif self.args.test_from_pickle:
            test_dataset = ARDSRawDataset.from_pickle(self.args.test_from_pickle)
        else:  # holdout, no pickle, no kfold
            # there is a really bizarre bug where my default arg is being overwritten by
            # the state of the train_dataset obj. I checked pointer references and there was
            # nothing. I might be able to do a deepcopy, but it might be easier to just supply
            # a blank list instead of relying on default
            test_dataset = ARDSRawDataset(
                self.args.data_path,
                self.args.experiment_num,
                self.args.cohort_file,
                self.args.n_sub_batches,
                all_sequences=[],
                dataset_type=self.args.dataset_type,
                to_pickle=self.args.test_to_pickle,
                train=False,
                unpadded_downsample_factor=self.args.downsample_factor,
                drop_frame_if_frac_missing=self.args.no_drop_frames,
            )

        return train_dataset, test_dataset

    def get_splits(self):
        train_dataset, test_dataset = self.get_base_datasets()
        for i in range(self.n_runs):
            if self.args.kfolds is not None:
                print('--- Run Fold {} ---'.format(i+1))
                train_dataset.set_kfold_indexes_for_fold(i)
                test_dataset.set_kfold_indexes_for_fold(i)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True if not self.args.unshuffled else False,
                pin_memory=self.args.cuda,
                num_workers=self.args.loader_threads,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.args.batch_size,
                shuffle=True if not self.args.unshuffled else False,
                pin_memory=self.args.cuda,
                num_workers=self.args.loader_threads,
            )
            yield train_dataset, train_loader, test_dataset, test_loader

    def train_and_test(self):
        for fold_num, (train_dataset, train_loader, test_dataset, test_loader) in enumerate(self.get_splits()):
            model = self.get_model()
            optimizer = self.get_optimizer(model)
            for epoch_num in range(self.args.epochs):
                self.run_train_epoch(model, train_loader, optimizer, epoch_num+1, fold_num)
                if self.args.reshuffle_oversample_per_epoch:
                    train_loader.dataset.set_oversampling_indices()

                if not self.args.no_test_after_epochs or epoch_num == self.args.epochs - 1:
                    self.run_test_epoch(epoch_num, model, test_dataset, test_loader, fold_num)

            if self.args.save_model:
                model_path = self.args.save_model.replace('.pth', '') + "-fold-{}.pth".format(fold_num) if self.n_runs > 1 else self.args.save_model
                torch.save(model, model_path)

        self.perform_post_modeling_actions()
        print('Run start time: {}'.format(self.start_time))

    def get_base_network(self):
        base_network = base_networks[self.args.base_network]

        if self.args.load_base_network:
            saved_model = torch.load(self.args.load_base_network)
            if isinstance(saved_model, torch.nn.DataParallel):
                saved_model = saved_model.module
            base_network = saved_model.breath_block
            self.results.hyperparams['base_network'] = base_network.network_name
        elif self.args.base_network in ['resnet18', 'resnet50', 'resnet101', 'resnet152']:
            base_network = base_network(
                initial_planes=self.args.initial_planes,
                first_pool_type=self.args.resnet_first_pool_type,
                double_conv_first=self.args.resnet_double_conv,
            )
        elif 'unet' in self.args.base_network:
            base_network = base_network(1)
        else:
            base_network = base_network()

        if self.args.freeze_base_network:
            for param in base_network.parameters():
                param.requires_grad = False
        return base_network

    def get_optimizer(self, model):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=self.args.learning_rate)
        elif self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=self.args.learning_rate, momentum=0.9, weight_decay=self.args.weight_decay, nesterov=True)
        return optimizer

    def run_test_epoch(self, epoch_num, model, test_dataset, test_loader, fold_num):
        self.preds = []
        self.pred_idx = []
        with torch.no_grad():
            for idx, (obs_idx, seq, metadata, target) in enumerate(test_loader):
                if not self.args.batch_size == 1:
                    obs_idx, seq, metadata, target = self.clip_odd_batch_sizes(obs_idx, seq, metadata, target)
                if seq.shape[0] == 0:
                    continue
                inputs = self.cuda_wrapper(Variable(seq.float()))
                metadata = self.cuda_wrapper(Variable(metadata.float()))
                target = self.cuda_wrapper(target.float())
                outputs = model(inputs, metadata)
                loss = self.calc_loss(outputs, target, inputs)
                self.results.update_meter('test_loss', fold_num, loss.data)
                self.results.update_epoch_meter('test_loss', epoch_num, loss.data)
                batch_preds = self._process_test_batch_results(outputs, target, inputs, fold_num)
                self.pred_idx.extend(self.transform_obs_idx(obs_idx, outputs).cpu().tolist())
                self.preds.extend(batch_preds)

        self.record_final_epoch_testing_results(fold_num, epoch_num, test_dataset)

    def get_model(self):
        if self.args.load_checkpoint:
            model = torch.load(self.args.load_checkpoint)
        else:
            base_network = self.get_base_network()
            model = self.model_cuda_wrapper(self.get_network(base_network))

        for p in model.parameters():
            if p.requires_grad and self.args.clip_grad:
                p.register_hook(lambda x: torch.clamp(x, -self.args.clip_val, self.args.clip_val))
        return model

    def transform_obs_idx(self, obs_idx, outputs):
        return obs_idx

    def clip_odd_batch_sizes(self, obs_idx, seq, metadata, target):
        """
        For some reason or the other DataParallel can fail on the LSTM
        if batch sizes are odd. So just for paranoia's sake clip odd part of
        a batch to make it even.
        """
        if seq.shape[0] % 2 == 1:
            new_dim = seq.shape[0] - 1
            obs_idx = obs_idx[:new_dim]
            seq = seq[:new_dim]
            target = target[:new_dim]
            metadata = metadata[:new_dim]
        return obs_idx, seq, metadata, target


class PatientClassifierMixin(object):
    def record_testing_results(self, target, batch_preds, fold_num):
        accuracy = accuracy_score(target, batch_preds)
        self.results.update_accuracy(fold_num, accuracy)

    def record_final_epoch_testing_results(self, fold_num, epoch_num, test_dataset):
        self.preds = pd.Series(self.preds, index=self.pred_idx)
        self.preds = self.preds.sort_index()
        y_test = test_dataset.get_ground_truth_df()
        self.results.perform_patient_predictions(y_test, self.preds, fold_num, epoch_num)

    def set_loss_criterion(self):
        if self.args.loss_func == 'vacillating':
            self.criterion = VacillatingLoss(self.cuda_wrapper(torch.FloatTensor([self.args.valpha])))
        elif self.args.loss_func == 'bce':
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif self.args.loss_func == 'confidence':
            self.criterion = ConfidencePenaltyLoss(self.args.conf_beta)
        elif self.args.loss_func == 'focal':
            self.criterion = FocalLoss(gamma=self.args.fl_gamma, alpha=self.args.fl_alpha)

    def perform_post_modeling_actions(self):
        self.results.aggregate_classification_results()
        self.results.save_all()


class WithTimeLayerClassifierMixin(object):
    def calc_loss(self, outputs, target, inputs):
        if self.args.batch_size > 1:
            target = target.unsqueeze(1)
        return self.criterion(outputs, target.repeat((1, outputs.shape[1], 1)))

    # One thing that is common in this function is that we process the outputs, process
    # the target, record testing results, and then return batch preds. It would be
    # so much easier if this all was just in one function
    def _process_test_batch_results(self, outputs, target, inputs, fold_num):
        batch_preds = outputs.argmax(dim=-1).cpu().view(-1)
        target = target.argmax(dim=1).cpu().reshape((outputs.shape[0], 1)).repeat((1, outputs.shape[1])).view(-1)
        self.record_testing_results(target, batch_preds, fold_num)
        return batch_preds.tolist()

    def transform_obs_idx(self, obs_idx, outputs):
        return obs_idx.reshape((outputs.shape[0], 1)).repeat((1, outputs.shape[1])).view(-1)


class SiameseMixin(object):
    def record_testing_results(self, batch_preds, batch_target, epoch_num, fold_num):
        accuracy = accuracy_score(batch_target, batch_preds)
        self.results.update_meter('accuracy', fold_num, accuracy)
        self.results.update_epoch_meter('accuracy', epoch_num, accuracy)

    def calc_loss(self, outputs_pos, outputs_neg):
        # We  want to do a loss on a read by read basis. You don't want to do
        # breath by breath
        target_pos = self.cuda_wrapper(torch.FloatTensor([[0, 1]])).repeat((outputs_pos.shape[0], 1))
        target_neg = self.cuda_wrapper(torch.FloatTensor([[1, 0]])).repeat((outputs_neg.shape[0], 1))
        loss_pos = self.criterion(outputs_pos, target_pos)
        loss_neg = self.criterion(outputs_neg, target_neg)
        return loss_pos + loss_neg

    def set_loss_criterion(self):
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def perform_post_modeling_actions(self):
        self.results.save_all()

    def get_base_datasets(self):
        # for holdout and pickle
        if self.args.train_from_pickle:
            train_dataset = SiameseNetworkDataset.from_pickle(self.args.train_from_pickle)
        else:  # for holdout, no pickle
            train_dataset = SiameseNetworkDataset(
                self.args.data_path,
                self.args.experiment_num,
                self.args.n_sub_batches,
                dataset_type=self.args.dataset_type,
                to_pickle=self.args.train_to_pickle,
                train=True,
            )
        self.n_sub_batches = train_dataset.n_sub_batches
        # for holdout and pickle
        if self.args.test_from_pickle:
            test_dataset = SiameseNetworkDataset.from_pickle(self.args.test_from_pickle)
        else:  # holdout, no pickle
            test_dataset = SiameseNetworkDataset(
                self.args.data_path,
                self.args.experiment_num,
                self.args.n_sub_batches,
                dataset_type=self.args.dataset_type,
                to_pickle=self.args.test_to_pickle,
                train=False,
            )
        test_dataset.scaling_factors = train_dataset.scaling_factors
        return train_dataset, test_dataset

    def run_train_epoch(self, model, train_loader, optimizer, epoch_num, fold_num):
        with torch.enable_grad():
            print("\nrun epoch {}\n".format(epoch_num))
            for batch_idx, (seq, pos_compr, neg_compr) in enumerate(train_loader):
                model.zero_grad()
                # XXX do we need this?
                #obs_idx, seq, metadata, target = self.clip_odd_batch_sizes(obs_idx, seq, metadata, target)
                #if seq.shape[0] == 0:
                #    continue
                seq = self.cuda_wrapper(Variable(seq.float()))
                pos_compr = self.cuda_wrapper(Variable(pos_compr.float()))
                neg_compr = self.cuda_wrapper(Variable(neg_compr.float()))
                outputs_pos = model(seq, pos_compr)
                outputs_neg = model(seq, neg_compr)
                loss = self.calc_loss(outputs_pos, outputs_neg)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.results.update_loss(fold_num, loss.data)

                # print individual loss and total loss
                if not self.args.no_print_progress:
                    print("batch num: {}/{}, avg loss: {}\r".format(batch_idx, len(train_loader), self.results.get_meter('loss', fold_num), end=""))

                if self.args.debug:
                    break

    def run_test_epoch(self, epoch_num, model, test_dataset, test_loader, fold_num):
        self.preds = []
        self.pred_idx = []
        with torch.no_grad():
            for batch_idx, (seq, pos_compr, neg_compr) in enumerate(test_loader):
                #obs_idx, seq, metadata, target = self.clip_odd_batch_sizes(obs_idx, seq, metadata, target)
                #if seq.shape[0] == 0:
                #    continue
                seq = self.cuda_wrapper(Variable(seq.float()))
                pos_compr = self.cuda_wrapper(Variable(pos_compr.float()))
                neg_compr = self.cuda_wrapper(Variable(neg_compr.float()))
                outputs_pos = model(seq, pos_compr)
                outputs_neg = model(seq, neg_compr)
                loss = self.calc_loss(outputs_pos, outputs_neg)
                self.results.update_meter('test_loss', fold_num, loss.data)
                self.results.update_epoch_meter('test_loss', epoch_num, loss.data)
                batch_preds, batch_target = self._process_test_batch_results(outputs_pos, outputs_neg)
                self.record_testing_results(batch_preds, batch_target, epoch_num, fold_num)
            self.results.print_meter_results('accuracy', fold_num)
            self.results.print_epoch_meter_results('accuracy', epoch_num)

    def _process_test_batch_results(self, outputs_pos, outputs_neg):
        target_pos = [1] * outputs_pos.shape[0]
        target_neg = [0] * outputs_pos.shape[0]
        cat = torch.cat([outputs_pos, outputs_neg], dim=0)
        preds = torch.argmax(cat, dim=1).cpu().numpy()
        return preds, target_pos + target_neg


class RegressorMixin(object):
    def record_testing_results(self, target, batch_preds, fold_num):
        mae = mean_absolute_error(target, batch_preds)
        self.results.update_meter('test_mae', fold_num, mae)
        r2 = r2_score(target, batch_preds)
        self.results.update_meter('r2', fold_num, r2)
        mse = mean_squared_error(target, batch_preds)
        self.results.update_meter('test_mse', fold_num, mse)

    def record_final_epoch_testing_results(self, fold_num, epoch_num, test_dataset):
        self.results.print_meter_results('test_mae', fold_num)

    def set_loss_criterion(self):
        self.criterion = torch.nn.MSELoss()

    def perform_post_modeling_actions(self):
        self.results.save_all()


class NestedMixin(object):
    def set_loss_criterion(self):
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def _process_test_batch_results(self, outputs, target, inputs, fold_num):
        batch_preds = outputs.argmax(dim=-1).cpu().view(-1)
        target = target.argmax(dim=1).cpu().reshape((outputs.shape[0], 1)).repeat((1, outputs.shape[1])).view(-1)
        # XXX find a way to make this function work later
        #self.record_testing_results(target, batch_preds, fold_num)
        return batch_preds.tolist()

    def get_base_datasets(self):
        kfold_num = None if self.args.kfolds is None else 0

        # for holdout and kfold
        if not self.args.train_from_pickle:
            train_dataset = ARDSRawDataset(
                self.args.data_path,
                self.args.experiment_num,
                self.args.cohort_file,
                self.args.n_sub_batches,
                all_sequences=[],
                dataset_type=self.args.dataset_type,
                to_pickle=self.args.train_to_pickle,
                kfold_num=kfold_num,
                total_kfolds=self.args.kfolds,
                unpadded_downsample_factor=self.args.downsample_factor,
                drop_frame_if_frac_missing=self.args.no_drop_frames,
                whole_patient_super_batch=True,
            )
        else:
            train_dataset = ARDSRawDataset.from_pickle(self.args.train_from_pickle, self.args.oversample)

        self.n_sub_batches = train_dataset.n_sub_batches
        if not self.args.test_from_pickle and self.args.kfolds is not None:
            test_dataset = ARDSRawDataset.make_test_dataset_if_kfold(train_dataset)
        elif self.args.test_from_pickle:
            test_dataset = ARDSRawDataset.from_pickle(self.args.test_from_pickle)
        else:  # holdout, no pickle, no kfold
            test_dataset = ARDSRawDataset(
                self.args.data_path,
                self.args.experiment_num,
                self.args.cohort_file,
                self.args.n_sub_batches,
                all_sequences=[],
                dataset_type=self.args.dataset_type,
                to_pickle=self.args.test_to_pickle,
                train=False,
                unpadded_downsample_factor=self.args.downsample_factor,
                drop_frame_if_frac_missing=self.args.no_drop_frames,
                whole_patient_super_batch=True,
            )

        return train_dataset, test_dataset

    def calc_loss(self, outputs, target, inputs):
        # instead of all_breaths / last_breath this is more like all_windows/last_window
        if self.args.loss_calc == 'all_breaths':
            return self.criterion(outputs, target.repeat((1, outputs.shape[1], 1)))
        elif self.args.loss_calc == 'last_breath':
            return self.criterion(outputs[:, -1], target)

    def record_final_epoch_testing_results(self, fold_num, epoch_num, test_dataset):
        # XXX add last_breath / all_breaths split
        self.preds = pd.Series(self.preds, index=self.pred_idx)
        self.preds = self.preds.sort_index()
        y_test = test_dataset.get_ground_truth_df()
        self.results.perform_patient_predictions(y_test, self.preds, fold_num, epoch_num)

    def transform_obs_idx(self, obs_idx, outputs):
        # XXX add last_breath / all_breaths split
        return obs_idx.reshape((outputs.shape[0], 1)).repeat((1, outputs.shape[1])).view(-1)

    def perform_post_modeling_actions(self):
        self.results.aggregate_classification_results()
        self.results.save_all()


class CNNToNestedRNNModel(NestedMixin, BaseTraining):
    def __init__(self, args):
        args.batch_size = 1
        super(CNNToNestedRNNModel, self).__init__(args)

    def get_network(self, base_network):
        return CNNToNestedRNNNetwork(base_network, self.args.n_sub_batches, self.args.cuda)


class CNNToNestedLSTMModel(NestedMixin, BaseTraining):
    def __init__(self, args):
        args.batch_size = 1
        super(CNNToNestedLSTMModel, self).__init__(args)

    def get_network(self, base_network):
        return CNNToNestedLSTMNetwork(base_network, self.args.n_sub_batches, self.args.cuda)


class CNNToNestedTransformerModel(NestedMixin, BaseTraining):
    def __init__(self, args):
        args.batch_size = 1
        super(CNNToNestedTransformerModel, self).__init__(args)

    def get_network(self, base_network):
        return CNNToNestedTransformerNetwork(base_network, self.args.n_sub_batches, self.args.cuda, self.args.transformer_blocks)


class CNNTransformerModel(WithTimeLayerClassifierMixin, BaseTraining, PatientClassifierMixin):
    def __init__(self, args):
        super(CNNTransformerModel, self).__init__(args)

    def get_network(self, base_network):
        return CNNTransformerNetwork(base_network, self.n_metadata_inputs, self.args.bm_to_linear, self.args.time_series_hidden_units, self.args.transformer_blocks)


class CNNLSTMModel(WithTimeLayerClassifierMixin, BaseTraining, PatientClassifierMixin):
    def __init__(self, args):
        super(CNNLSTMModel, self).__init__(args)

    def calc_loss(self, outputs, target, inputs):
        if self.args.loss_calc == 'all_breaths':
            if self.args.batch_size > 1:
                target = target.unsqueeze(1)
            return self.criterion(outputs, target.repeat((1, outputs.shape[1], 1)))
        elif self.args.loss_calc == 'last_breath':
            return self.criterion(outputs[:, -1, :], target)

    def get_network(self, base_network):
        return CNNLSTMNetwork(base_network, self.n_metadata_inputs, self.args.bm_to_linear, self.args.time_series_hidden_units)

    def run_train_epoch(self, model, train_loader, optimizer, epoch_num, fold_num):
        print("\nrun epoch {}\n".format(epoch_num))
        gt_df = train_loader.dataset.get_ground_truth_df()
        last_pt = None
        with torch.enable_grad():
            for idx, (obs_idx, seq, metadata, target) in enumerate(train_loader):
                model.zero_grad()
                obs_idx, seq, metadata, target = self.clip_odd_batch_sizes(obs_idx, seq, metadata, target)
                if seq.shape[0] == 0:
                    continue
                target = self.cuda_wrapper(target.float())
                inputs = self.cuda_wrapper(Variable(seq.float()))
                metadata = self.cuda_wrapper(Variable(metadata.float()))
                if not self.args.unshuffled:
                    outputs, _ = model(inputs, metadata, None)
                else:
                    cur_pt = gt_df.loc[obs_idx].patient.unique()[0]
                    if cur_pt != last_pt:
                        hx_cx = None
                    outputs, hx_cx = model(inputs, metadata, hx_cx)
                    hx_cx = (hx_cx[0].detach(), hx_cx[1].detach())
                    last_pt = cur_pt
                self.handle_train_optimization(optimizer, outputs, target, inputs, fold_num, len(train_loader), idx, epoch_num, model)

                if self.args.debug:
                    break

    def run_test_epoch(self, epoch_num, model, test_dataset, test_loader, fold_num):
        self.preds = []
        self.pred_idx = []
        gt_df = test_dataset.get_ground_truth_df()
        last_pt = None
        with torch.no_grad():
            for idx, (obs_idx, seq, metadata, target) in enumerate(test_loader):
                obs_idx, seq, metadata, target = self.clip_odd_batch_sizes(obs_idx, seq, metadata, target)
                if seq.shape[0] == 0:
                    continue
                inputs = self.cuda_wrapper(Variable(seq.float()))
                metadata = self.cuda_wrapper(Variable(metadata.float()))
                target = self.cuda_wrapper(target.float())
                if not self.args.unshuffled:
                    outputs, _ = model(inputs, metadata, None)
                else:
                    cur_pt = gt_df.loc[obs_idx].patient.unique()[0]
                    if cur_pt != last_pt:
                        hx_cx = None
                    outputs, hx_cx = model(inputs, metadata, hx_cx)
                    last_pt = cur_pt
                loss = self.calc_loss(outputs, target, inputs)
                self.results.update_meter('test_loss', fold_num, loss.data)
                self.results.update_epoch_meter('test_loss', epoch_num, loss.data)
                batch_preds = self._process_test_batch_results(outputs, target, inputs, fold_num)
                self.pred_idx.extend(self.transform_obs_idx(obs_idx, outputs).cpu().tolist())
                self.preds.extend(batch_preds)

        self.record_final_epoch_testing_results(fold_num, epoch_num, test_dataset)


class CNNLSTMDoubleLinearModel(BaseTraining, PatientClassifierMixin):
    def __init__(self, args):
        super(CNNLSTMDoubleLinearModel, self).__init__(args)

    def get_network(self, base_network):
        return CNNLSTMDoubleLinearNetwork(base_network, self.n_metadata_inputs, self.args.bm_to_linear, self.args.time_series_hidden_units, self.n_sub_batches)

    def calc_loss(self, outputs, target, inputs):
        return self.criterion(outputs, target)

    def _process_test_batch_results(self, outputs, target, inputs, fold_num):
        batch_preds = outputs.argmax(dim=-1).cpu().view(-1)
        target = target.argmax(dim=1).cpu().reshape((outputs.shape[0], 1)).view(-1)
        self.record_testing_results(target, batch_preds, fold_num)
        return batch_preds.tolist()

    def transform_obs_idx(self, obs_idx, outputs):
        return obs_idx.reshape((outputs.shape[0], 1)).view(-1)


class CNNLinearModel(BaseTraining, PatientClassifierMixin):
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


class CNNDoubleLinearModel(BaseTraining, PatientClassifierMixin):
    def __init__(self, args):
        super(CNNDoubleLinearModel, self).__init__(args)

    def calc_loss(self, outputs, target, inputs):
        return self.criterion(outputs, target)

    def _process_test_batch_results(self, outputs, target, inputs, fold_num):
        batch_preds = outputs.argmax(dim=-1).cpu()
        target = target.argmax(dim=1).cpu()
        self.record_testing_results(target, batch_preds, fold_num)
        return batch_preds.tolist()

    def get_network(self, base_network):
        return CNNDoubleLinearNetwork(base_network, self.args.n_sub_batches, self.n_metadata_inputs)


class CNNSingleBreathLinearModel(WithTimeLayerClassifierMixin, BaseTraining, PatientClassifierMixin):
    def __init__(self, args):
        super(CNNSingleBreathLinearModel, self).__init__(args)

    def get_network(self, base_network):
        return CNNSingleBreathLinearNetwork(base_network)


class CNNLinearComprToRFModel(WithTimeLayerClassifierMixin, BaseTraining, PatientClassifierMixin):
    def __init__(self, args):
        super(CNNLinearComprToRFModel, self).__init__(args)

    def calc_loss(self, outputs, target, inputs):
        return self.criterion(outputs, target)

    def _process_test_batch_results(self, outputs, target, inputs, fold_num):
        batch_preds = outputs.argmax(dim=-1).cpu().view(-1)
        target = target.argmax(dim=1).cpu().reshape((outputs.shape[0], 1)).view(-1)
        self.record_testing_results(target, batch_preds, fold_num)
        return batch_preds.tolist()

    def transform_obs_idx(self, obs_idx, outputs):
        return obs_idx

    def get_network(self, base_network):
        return CNNLinearComprToRF(base_network)


class MetadataOnlyModel(BaseTraining, PatientClassifierMixin):
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


class LSTMOnlyModel(WithTimeLayerClassifierMixin, BaseTraining, PatientClassifierMixin):
    def __init__(self, args):
        super(LSTMOnlyModel, self).__init__(args)

    def get_network(self, _):
        return LSTMOnlyNetwork(self.args.time_series_hidden_units)


class CNNRegressorModel(BaseTraining, RegressorMixin):
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
        try:
            self.n_bm_features
        except:
            raise Exception('You have specified cnn regressor without specifying which dataset you want to use. Do so with the -dt flag')
        return CNNRegressor(base_network, self.n_bm_features)


class AutoencoderModel(BaseTraining, RegressorMixin):
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


class SiameseCNNLSTMModel(SiameseMixin, BaseTraining):
    def __init__(self, args):
        super(SiameseCNNLSTMModel, self).__init__(args)

    def get_network(self, base_network):
        return SiameseCNNLSTMNetwork(base_network, self.args.time_series_hidden_units, self.n_sub_batches)


class SiameseCNNTransformerModel(SiameseMixin, BaseTraining):
    def __init__(self, args):
        super(SiameseCNNTransformerModel, self).__init__(args)

    def get_network(self, base_network):
        return SiameseCNNTransformerNetwork(base_network, self.args.time_series_hidden_units, self.n_sub_batches)


class SiameseCNNLinearModel(SiameseMixin, BaseTraining):
    def __init__(self, args):
        super(SiameseCNNLinearModel, self).__init__(args)

    def get_network(self, base_network):
        return SiameseCNNLinearNetwork(base_network, self.n_sub_batches)


class SiamesePretrainedModel(WithTimeLayerClassifierMixin, BaseTraining, PatientClassifierMixin):
    def __init__(self, args):
        super(SiamesePretrainedModel, self).__init__(args)

    def get_network(self, base_network):
        network = torch.load(self.args.load_siamese)
        if isinstance(network, nn.DataParallel):
            network = network.module
        return SiameseARDSClassifier(network)


# Is putting a global in code like this a great idea? probably not, but saves some hassle
# on outside scripting
network_map = {
    'cnn_lstm': CNNLSTMModel,
    'cnn_linear': CNNLinearModel,
    'cnn_regressor': CNNRegressorModel,
    'metadata_only': MetadataOnlyModel,
    'autoencoder': AutoencoderModel,
    'cnn_transformer': CNNTransformerModel,
    'siamese_cnn_linear': SiameseCNNLinearModel,
    'siamese_cnn_lstm': SiameseCNNLSTMModel,
    'siamese_cnn_transformer': SiameseCNNTransformerModel,
    'siamese_pretrained': SiamesePretrainedModel,
    'cnn_lstm_double_linear': CNNLSTMDoubleLinearModel,
    'cnn_double_linear': CNNDoubleLinearModel,
    'cnn_single_breath_linear': CNNSingleBreathLinearModel,
    'lstm_only': LSTMOnlyModel,
    'cnn_to_nested_rnn': CNNToNestedRNNModel,
    'cnn_to_nested_lstm': CNNToNestedLSTMModel,
    'cnn_to_nested_transformer': CNNToNestedTransformerModel,
    'cnn_linear_compr_to_rf': CNNLinearComprToRFModel,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-path', default='/fastdata/ardsdetection', help='Path to ARDS detection dataset')
    parser.add_argument('-en', '--experiment-num', type=int, default=1)
    parser.add_argument('-c', '--cohort-file', default='cohort-description.csv')
    parser.add_argument('-n', '--network', choices=list(network_map.keys()), default='cnn_lstm')
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('-p', '--train-from-pickle')
    parser.add_argument('--train-to-pickle')
    parser.add_argument('--test-from-pickle')
    parser.add_argument('--test-to-pickle')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('--base-network', choices=base_networks, default='resnet18')
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
    parser.add_argument('-dt', '--dataset-type', choices=[
        'padded_breath_by_breath',
        'unpadded_sequences',
        'unpadded_centered_sequences',
        'unpadded_downsampled_sequences',
        'unpadded_centered_downsampled_sequences',
        'spaced_padded_breath_by_breath',
        'stretched_breath_by_breath',
        'padded_breath_by_breath_with_full_bm_target',
        'padded_breath_by_breath_with_limited_bm_target',
        'padded_breath_by_breath_with_experimental_bm_target',
        'padded_breath_by_breath_with_flow_time_features',
        'unpadded_downsampled_autoencoder_sequences'
    ], default='padded_breath_by_breath')
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--loader-threads', type=int, default=0, help='specify how many threads we should use to load data. Sometimes the threads fail to shutdown correctly though and this can cause memory errors. If this happens a restart works well')
    parser.add_argument('--save-model', help='save the model to a specific file')
    parser.add_argument('--load-base-network', help='load base network only from a saved model')
    parser.add_argument('-rdc','--resnet-double-conv', action='store_true')
    parser.add_argument('--bm-to-linear', action='store_true')
    parser.add_argument('-exp', '--experiment-name')
    parser.add_argument('--downsample-factor', type=float, default=4.0)
    parser.add_argument('--no-drop-frames', action='store_false')
    parser.add_argument('-wd', '--weight-decay', type=float, default=.0001)
    parser.add_argument('-loss', '--loss-func', choices=['bce', 'vacillating', 'confidence', 'focal'], default='bce', help='This option only works for classification. Choose the loss function you want to use for classification purposes: BCE or vacillating loss.')
    parser.add_argument('--valpha', type=float, default=float('Inf'), help='alpha value to use for vacillating loss. Lower alpha values mean vacillating loss will contribute less to overall loss of the system. Default value is inf')
    parser.add_argument('--conf-beta', type=float, default=1.0, help='Modifier to the intensity of the confidence penalty')
    parser.add_argument('--time-series-hidden-units', type=int, default=512)
    parser.add_argument('--transformer-blocks', type=int, default=2)
    parser.add_argument('--unshuffled', action='store_true', help='dont shuffle data for lstm processing')
    parser.add_argument('--load-siamese', help='load a siamese network pretrained model')
    parser.add_argument('--fl-gamma', type=float, default=2.0)
    parser.add_argument('--fl-alpha', type=float, default=.9)
    parser.add_argument('--oversample', action='store_true')
    parser.add_argument('--reshuffle-oversample-per-epoch', action='store_true')
    parser.add_argument('--load-checkpoint')
    parser.add_argument('--freeze-base-network', action='store_true')
    parser.add_argument('--stop-on-loss', action='store_true')
    parser.add_argument('--stop-thresh', type=float, default=1.5)
    parser.add_argument('--stop-after-epoch', type=int, default=1)
    parser.add_argument('--clip-grad', action='store_true')
    parser.add_argument('--clip-val', type=float, default=5)
    args = parser.parse_args()

    # convenience code
    if args.load_siamese:
        args.network = 'siamese_pretrained'

    if args.fl_alpha > 1 or args.fl_alpha < 0:
        raise Exception('Focal loss alpha must be between 0 and 1')
    cls = network_map[args.network](args)
    cls.train_and_test()


if __name__ == "__main__":
    main()
