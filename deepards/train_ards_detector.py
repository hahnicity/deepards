from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import box_area
from torchvision.transforms import Compose

from deepards.augmentation import IEWindowWarping, IEWindowWarpingIEProgrammable,  NaiveWindowWarping
from deepards.config import Configuration
from deepards.dataset import ARDSRawDataset, ImgARDSDataset, SiameseNetworkDataset, two_dim_transforms
from deepards.loss import ConfidencePenaltyLoss, FocalLoss, VacillatingLoss
from deepards.metrics import DeepARDSResults, Reporting
from deepards.models.autoencoder_cnn import AutoencoderCNN
from deepards.models.autoencoder_network import AutoencoderNetwork
from deepards.models.cnn_to_nested_layer import CNNToNestedLSTMNetwork, CNNToNestedRNNNetwork, CNNToNestedTransformerNetwork
from deepards.models.cnn_transformer import CNNTransformerNetwork
from deepards.models.densenet import densenet18, densenet121, densenet161, densenet169, densenet201
from deepards.models.densenet2d import densenet18 as densenet18_2d
from deepards.models.densenet2x1d import densenet18 as densenet18_2x1d
from deepards.models.detection import FasterRCNNMain, RetinaNetMain
from deepards.models.lstm_only import DoubleLSTMNetwork, LSTMOnlyNetwork, LSTMOnlyWithPacking
from deepards.models.protopnet1d.model import construct_PPNet
from deepards.models.protopnet1d.ppnet_push import prototype_viz, push_prototypes
from deepards.models.protopnet2d.model import construct_PPNet as construct_PPNet2D
from deepards.models.protopnet2d.push import Pusher
from deepards.models.resnet import resnet18, resnet50, resnet101, resnet152
from deepards.models.senet import senet18, senet154, se_resnet18, se_resnet50, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d
from deepards.models.siamese import SiameseARDSClassifier, SiameseCNNLinearNetwork, SiameseCNNLSTMNetwork, SiameseCNNTransformerNetwork
from deepards.models.torch_cnn_bm_regressor import CNNRegressor
from deepards.models.torch_cnn_linear_network import CNNDoubleLinearNetwork, CNNLinearNetwork, CNNLinearNetwork2D, CNNLinearComprToRF, CNNLinearToMean, CNNSingleBreathLinearNetwork
from deepards.models.torch_cnn_lstm_combo import CNNLSTMDoubleLinearNetwork, CNNLSTMNetwork
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
    'densenet18_2d': densenet18_2d,
    'densenet18_2x1d': densenet18_2x1d,
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
saved_models_default_dir = os.path.join(os.path.dirname(__file__), 'saved_models')


def detection_collate_fn(batch):
    return tuple(zip(*batch))


class BaseTraining(object):
    clip_odd_batches = False

    def __init__(self, args):
        self.args = args

        if self.args.cuda:
            self.cuda_wrapper = lambda x: x.cuda()
        elif self.args.cuda_no_dp:
            single_gpu = torch.device('cuda:{}'.format(args.cuda_device))
            self.cuda_wrapper = lambda x: x.to(single_gpu)
        else:
            self.cuda_wrapper = lambda x: x

        if self.args.debug or self.args.cuda_no_dp:
            self.model_cuda_wrapper = lambda x: x.to(single_gpu) if self.args.cuda_no_dp else x
        else:
            self.model_cuda_wrapper = lambda x: nn.DataParallel(x).cuda() if self.args.cuda else x
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

        self.is_2d_dataset = "_2d" in self.args.network
        self.is_2x1d_dataset = "_2x1d" in self.args.network
        if self.is_2d_dataset and '_2d' not in self.args.base_network:
            self.args.base_network = self.args.base_network + '_2d'
        elif self.is_2x1d_dataset and '2x1d' not in self.args.base_network:
            self.args.base_network = self.args.base_network + '_2x1d'

        if self.args.network in ['retinanet_2d', 'retinanet_2x1d', 'faster_rcnn_2d']:
            self.bbox = True
        else:
            self.bbox = False

        if self.args.unshuffled and self.args.batch_size > 1:
            raise Exception('Currently we can only run unshuffled runs with a batch size of 1!')

        self.n_kfolds = self.args.kfolds if self.args.kfolds is not None else 1
        # Train and test both load from the same dataset in the case of kfold
        if self.n_kfolds > 1:
            self.args.test_to_pickle = None

        self.start_time = datetime.now().strftime('%s')
        self.results = DeepARDSResults(self.start_time, self.args.experiment_name, **self.args.__dict__)
        print('Run start time: {}'.format(self.start_time))

    def run_train_epoch(self, model, train_loader, optimizer, epoch_num, fold_num):
        print('train instances: {}'.format(len(train_loader)))
        with torch.enable_grad():
            print("\nrun epoch {}\n".format(epoch_num))
            model.train()
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
        if not self.args.no_print_progress or self.args.print_progress:
            print("batch num: {}/{}, avg loss: {}\r".format(batch_idx, total_batches, self.results.get_meter('loss_epoch_{}'.format(epoch_num), fold_num), end=""))

    def get_transforms(self):
        # XXX I should rly be handling this in the dataset cls
        if self.args.transforms is None:
            return

        transforms = []
        if 'ie_ww' in self.args.transforms:
            transforms.append(IEWindowWarping(.5, 2, self.args.transform_probability))
        if 'naive_ww' in self.args.transforms:
            transforms.append(NaiveWindowWarping(.5, 2, self.args.transform_probability))
        if 'ie_ww_i_or_e' in self.args.transforms:
            transforms.append(IEWindowWarpingIEProgrammable(.5, 2, self.args.transform_probability, self.args.use_i))
        return Compose(transforms)

    def get_base_datasets(self):
        kfold_num = None if self.args.kfolds is None else 0
        transforms = self.get_transforms()
        if 'oversample' in self.args.__dict__:
            oversample_minority = self.args.oversample
        else:
            oversample_minority = self.args.oversample_minority

        if not self.args.train_from_pickle:
            # NOTE: the datasets themselves are storing way too much state and causes us to
            # have to reinstantiate dataset objects in storage whenever our underlying code
            # changes, and we also cannot use more than 1 cuda device anymore. It would be
            # good in the future if we didn't go down this path of dataset object development.
            # For now I think I don't want to refactor when we are close to done with the project
            # and I can just parallelize runs across multiple GPUs.

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
                oversample_minority=oversample_minority,
                train_patient_fraction=self.args.train_pt_frac,
                transforms=transforms,
                holdout_set_type=self.args.holdout_set_type,
                drop_if_under_r2=self.args.drop_if_under_r2,
                drop_i_lim=self.args.drop_i_lim,
                drop_e_lim=self.args.drop_e_lim,
                truncate_e_lim=self.args.truncate_e_lim,
                undersample_factor=self.args.undersample_factor,
                undersample_std_factor=self.args.undersample_std_factor,
            )
        else:
            train_dataset = ARDSRawDataset.from_pickle(
                self.args.train_from_pickle, oversample_minority, self.args.train_pt_frac, transforms, self.args.undersample_factor, self.args.undersample_std_factor, self.args.oversample_all_factor
            )

        self.n_sub_batches = train_dataset.n_sub_batches
        if not self.args.test_from_pickle and (self.args.kfolds is not None):
            test_dataset = ARDSRawDataset.make_test_dataset_if_kfold(train_dataset)
        elif self.args.test_from_pickle:
            test_dataset = ARDSRawDataset.from_pickle(self.args.test_from_pickle, False, 1.0, None, -1, None, 1.0)
        else:  # holdout, no pickle, no kfold
            # there is a really bizarre bug where my default arg is being overwritten by
            # the state of the train_dataset obj. I checked pointer references and there was
            # nothing. I might be able to do a deepcopy, but it is easier to just supply
            # a blank list instead of relying on default for all_sequences var.
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
                holdout_set_type=self.args.holdout_set_type,
                train_patient_fraction=1.0,
                transforms=None,
                final_validation_set=self.args.final_validation,
                drop_i_lim=self.args.drop_i_lim,
                drop_e_lim=self.args.drop_e_lim,
                truncate_e_lim=self.args.truncate_e_lim,
                undersample_factor=-1,
                oversample_minority=False,
            )

        if self.is_2d_dataset or self.is_2x1d_dataset:
            train_dataset = ImgARDSDataset(train_dataset, self.args.two_dim_transforms, self.args.with_fft, self.args.only_fft, self.args.fft_real_only, self.bbox, self.args.row_mix)
            test_dataset = ImgARDSDataset(test_dataset, self.args.two_dim_transforms, self.args.with_fft, self.args.only_fft, self.args.fft_real_only, self.bbox, self.args.row_mix)

        return train_dataset, test_dataset

    def get_splits(self, fold_num):
        train_dataset, test_dataset = self.get_base_datasets()
        collate_fn = None if not self.bbox else detection_collate_fn
        if self.args.kfolds is not None:
            print('--- Run Fold {} ---'.format(fold_num+1))
            train_dataset.set_kfold_indexes_for_fold(fold_num)
            test_dataset.set_kfold_indexes_for_fold(fold_num)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True if not self.args.unshuffled else False,
            pin_memory=self.args.cuda,
            num_workers=self.args.loader_threads,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            pin_memory=self.args.cuda,
            num_workers=self.args.loader_threads,
            collate_fn=collate_fn,
        )
        return train_dataset, train_loader, test_dataset, test_loader

    def train_and_test(self):
        saved_models_dir = saved_models_default_dir if not self.args.saved_models_dir else self.args.saved_models_dir
        for fold_num in range(self.n_kfolds):
            if fold_num == 0:
                train_dataset, train_loader, test_dataset, test_loader = self.get_splits(fold_num)
            if self.args.only_fold and fold_num != self.args.only_fold:
                continue
            model = self.get_model()
            optimizer = self.get_optimizer(model)
            for epoch_num in range(1, self.args.epochs+1):
                if not self.args.no_train:
                    self.run_train_epoch(model, train_loader, optimizer, epoch_num, fold_num)
                if self.args.reshuffle_oversample_per_epoch:
                    train_loader.dataset.set_oversampling_indices()

                if not self.args.no_test_after_epochs or epoch_num == self.args.epochs - 1:
                    self.run_test_epoch(epoch_num, model, test_dataset, test_loader, fold_num)
                if self.args.save_model_per_epoch:
                    model_path = os.path.join(
                        saved_models_dir,
                        os.path.basename(
                            os.path.splitext(self.args.save_model)[0] +
                            '-epoch{}'.format(epoch_num) +
                            ('-fold{}.pth'.format(fold_num) if self.n_kfolds > 1 else '.pth')
                        )
                    )
                    torch.save(model, model_path)

                if self.args.reload_dataset_per_epoch:
                    print('reload dataset')
                    train_dataset, train_loader, test_dataset, test_loader = self.get_splits(fold_num)

            if self.args.save_model:
                model_path = os.path.join(
                    saved_models_dir,
                    os.path.basename(
                        os.path.splitext(self.args.save_model)[0] +
                        ('-fold{}.pth'.format(fold_num) if self.n_kfolds > 1 else '.pth')
                    )
                )
                torch.save(model, model_path)

        self.perform_post_modeling_actions()
        self.perform_plotting(test_dataset)
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
        else:  # this is becoming our defacto densenet logic
            base_network = base_network(
                with_fft=self.args.with_fft,
                block_kernel_size=self.args.block_kernel_size,
                only_fft=self.args.only_fft,
                fft_real_only=self.args.fft_real_only,
            )

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
            # alright...this is weird af. each time I enter the input into
            # the model the results change each time. This is bizarre because
            # the weights are set and theres no gradient involved. Its like they're
            # floating in some probabilistic range. The predictions do
            # seem to change between gradcam and here. Also the predictions can
            # change between runs. what the actual fuck. Are the weights changing
            # somehow? weights dont seem to be changing on first conv layer.
            # Nor does it seem to change on final layer. It seems the breath block
            # is changing between iterations somehow. Crap, could it have been
            # dropout this whole time? It was the batch statistics that was doing
            # it.
            #
            # So on inference time batch norm normalizes by the computed mean and variance.
            # However, I think the thing here is that we end up normalizing twice. Once
            # in the dataset class, and again in the batch norm. This shouldn't really
            # do anything, but we're normalizing to unit std not unit var. So thats the
            # difference here.
            model.eval()
            for idx, (obs_idx, seq, metadata, target) in enumerate(test_loader):
                if not self.args.batch_size == 1 and self.clip_odd_batches:
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

    def perform_plotting(self, test_dataset):
        # just using hard-coded argument for now.
        dtw_cache_dir = 'dtw_cache'
        if self.args.plot_dtw_with_disease or self.args.plot_pt_dtw_by_minute or self.args.perform_dtw_preprocessing:
            self.results.perform_dtw_preprocessing(test_dataset, dtw_cache_dir)

        if self.args.plot_pt_dtw_by_minute:
            self.results.plot_dtw_by_minute(self.args.plot_pt_dtw_by_minute, test_dataset, dtw_cache_dir)

        if self.args.plot_dtw_with_disease and not self.args.plot_tiled_disease_evol:
            self.results.perform_hourly_patient_plot_with_dtw(test_dataset, dtw_cache_dir)
        elif self.args.plot_untiled_disease_evol:
            self.results.perform_hourly_patient_plot()

        if self.args.plot_tiled_disease_evol:
            self.results.plot_tiled_disease_evol(test_dataset, dtw_cache_dir, self.args.plot_dtw_with_disease)


class PatientClassifierMixin(object):
    def record_testing_results(self, target, batch_preds, fold_num):
        accuracy = accuracy_score(target, batch_preds)
        self.results.update_accuracy(fold_num, accuracy)

    def record_final_epoch_testing_results(self, fold_num, epoch_num, test_dataset):
        self.preds = pd.Series(self.preds, index=self.pred_idx)
        self.preds = self.preds.sort_index()
        y_test = test_dataset.get_ground_truth_df()
        self.results.perform_patient_predictions(y_test, self.preds, fold_num, epoch_num)
        self.results.save_predictions_by_hour(y_test, self.preds, test_dataset.seq_hours, epoch_num, fold_num)

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


class PerBreathClassifierMixin(object):
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
            test_dataset = SiameseNetworkDataset.from_pickle(self.args.test_from_pickle, False, 1.0)
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
        print('train instances: {}'.format(len(train_loader)))
        with torch.enable_grad():
            print("\nrun epoch {}\n".format(epoch_num))
            model.train()
            for batch_idx, (seq, pos_compr, neg_compr) in enumerate(train_loader):
                model.zero_grad()
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
                if not self.args.no_print_progress or self.args.print_progress:
                    print("batch num: {}/{}, avg loss: {}\r".format(batch_idx, len(train_loader), self.results.get_meter('loss', fold_num), end=""))

                if self.args.debug:
                    break

    def run_test_epoch(self, epoch_num, model, test_dataset, test_loader, fold_num):
        with torch.no_grad():
            model.eval()
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
        return batch_preds.tolist()

    def get_base_datasets(self):
        kfold_num = None if self.args.kfolds is None else 0

        if 'oversample' in self.args.__dict__:
            oversample_minority = self.args.oversample
        else:
            oversample_minority = self.args.oversample_minority

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
                whole_patient_super_batch=True,
                train_patient_fraction=self.args.train_pt_frac,
                drop_i_lim=self.args.drop_i_lim,
                drop_e_lim=self.args.drop_e_lim,
                truncate_e_lim=self.args.truncate_e_lim,
                undersample_factor=self.args.undersample_factor,
                undersample_std_factor=self.args.undersample_std_factor,
                oversample_minority=oversample_minority,
            )
        else:
            train_dataset = ARDSRawDataset.from_pickle(self.args.train_from_pickle, oversample_minority, self.args.train_pt_frac, self.args.undersample_factor, self.args.undersample_std_factor, self.args.oversample_all_factor)

        self.n_sub_batches = train_dataset.n_sub_batches
        if not self.args.test_from_pickle and self.args.kfolds is not None:
            test_dataset = ARDSRawDataset.make_test_dataset_if_kfold(train_dataset)
        elif self.args.test_from_pickle:
            test_dataset = ARDSRawDataset.from_pickle(self.args.test_from_pickle, False, 1.0, 1.0)
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
                whole_patient_super_batch=True,
                holdout_set_type=self.args.holdout_set_type,
                train_patient_fraction=1.0,
                drop_i_lim=self.args.drop_i_lim,
                drop_e_lim=self.args.drop_e_lim,
                truncate_e_lim=self.args.truncate_e_lim,
                undersample_factor=-1,
                oversample_minority=False,
            )

        return train_dataset, test_dataset

    def calc_loss(self, outputs, target, inputs):
        # instead of all_breaths / last_breath this is more like all_windows/last_window
        if self.args.loss_calc == 'all_breaths':
            return self.criterion(outputs, target.repeat((1, outputs.shape[1], 1)))
        elif self.args.loss_calc == 'last_breath':
            return self.criterion(outputs[:, -1], target)

    def record_final_epoch_testing_results(self, fold_num, epoch_num, test_dataset):
        self.preds = pd.Series(self.preds, index=self.pred_idx)
        self.preds = self.preds.sort_index()
        y_test = test_dataset.get_ground_truth_df()
        self.results.perform_patient_predictions(y_test, self.preds, fold_num, epoch_num)
        self.save_predictions_by_hour(y_test, self.preds, test_dataset.seq_hours, epoch_num, fold_num)

    def transform_obs_idx(self, obs_idx, outputs):
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
    clip_odd_batches = True

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


class CNNTransformerModel(PerBreathClassifierMixin, BaseTraining, PatientClassifierMixin):
    def __init__(self, args):
        super(CNNTransformerModel, self).__init__(args)

    def get_network(self, base_network):
        return CNNTransformerNetwork(base_network, self.n_metadata_inputs, self.args.bm_to_linear, self.args.time_series_hidden_units, self.args.transformer_blocks)


class CNNLSTMModel(PerBreathClassifierMixin, BaseTraining, PatientClassifierMixin):
    clip_odd_batches = True

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
        print('train instances: {}'.format(len(train_loader)))
        gt_df = train_loader.dataset.get_ground_truth_df()
        last_pt = None
        with torch.enable_grad():
            model.train()
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
            model.eval()
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
    clip_odd_batches = True

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


class CNNLinearModel2D(BaseTraining, PatientClassifierMixin):
    def __init__(self, args):
        super().__init__(args)

    def calc_loss(self, outputs, target, inputs):
        return self.criterion(outputs, target)

    def _process_test_batch_results(self, outputs, target, inputs, fold_num):
        batch_preds = outputs.argmax(dim=-1).cpu()
        target = target.argmax(dim=1).cpu()
        self.record_testing_results(target, batch_preds, fold_num)
        return batch_preds.tolist()

    def get_network(self, base_network):
        return CNNLinearNetwork2D(base_network)


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


class CNNSingleBreathLinearModel(PerBreathClassifierMixin, BaseTraining, PatientClassifierMixin):
    def __init__(self, args):
        super(CNNSingleBreathLinearModel, self).__init__(args)

    def get_network(self, base_network):
        return CNNSingleBreathLinearNetwork(base_network)


class CNNLinearComprToRFModel(PerBreathClassifierMixin, BaseTraining, PatientClassifierMixin):
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


class CNNLinearToMeanModel(PerBreathClassifierMixin, BaseTraining, PatientClassifierMixin):
    def __init__(self, args):
        super(CNNLinearToMeanModel, self).__init__(args)

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
        return CNNLinearToMean(base_network)


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


class LSTMOnlyModel(BaseTraining, PatientClassifierMixin):
    clip_odd_batches = True

    def __init__(self, args):
        super(LSTMOnlyModel, self).__init__(args)

    def calc_loss(self, outputs, target, inputs):
        return self.criterion(outputs, target)

    def _process_test_batch_results(self, outputs, target, inputs, fold_num):
        batch_preds = outputs.argmax(dim=-1).cpu()
        target = target.argmax(dim=-1).cpu()
        self.record_testing_results(target, batch_preds, fold_num)
        return batch_preds.tolist()

    def get_network(self, _):
        return LSTMOnlyNetwork(self.args.time_series_hidden_units, self.args.n_sub_batches)


class LSTMOnlyWithPackingModel(BaseTraining, PatientClassifierMixin):
    clip_odd_batches = True

    def __init__(self, args):
        super(LSTMOnlyWithPackingModel, self).__init__(args)

    def calc_loss(self, outputs, target, inputs):
        return self.criterion(outputs, target)

    def _process_test_batch_results(self, outputs, target, inputs, fold_num):
        batch_preds = outputs.argmax(dim=-1).cpu()
        target = target.argmax(dim=-1).cpu()
        self.record_testing_results(target, batch_preds, fold_num)
        return batch_preds.tolist()

    def get_network(self, _):
        return LSTMOnlyWithPacking(self.args.time_series_hidden_units, self.args.n_sub_batches)


class DoubleLSTMModel(BaseTraining, PatientClassifierMixin):
    clip_odd_batches = True

    def __init__(self, args):
        super(DoubleLSTMModel, self).__init__(args)

    def calc_loss(self, outputs, target, inputs):
        return self.criterion(outputs, target)

    def _process_test_batch_results(self, outputs, target, inputs, fold_num):
        batch_preds = outputs.argmax(dim=-1).cpu()
        target = target.argmax(dim=-1).cpu()
        self.record_testing_results(target, batch_preds, fold_num)
        return batch_preds.tolist()

    def get_network(self, _):
        return DoubleLSTMNetwork(self.args.time_series_hidden_units, self.args.n_sub_batches)


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
    clip_odd_batches = True

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


class SiamesePretrainedModel(PerBreathClassifierMixin, BaseTraining, PatientClassifierMixin):
    def __init__(self, args):
        super(SiamesePretrainedModel, self).__init__(args)

    def get_network(self, base_network):
        network = torch.load(self.args.load_siamese)
        if isinstance(network, nn.DataParallel):
            network = network.module
        return SiameseARDSClassifier(network)


class ProtoPNetModel(object):

    def get_optimizer(self, model):
        if self.args.optimizer == 'adam':
            optim_cls = torch.optim.Adam
        elif self.args.optimizer == 'sgd':
            optim_cls = torch.optim.SGD
        joint_optimizer_specs = [{
            'params': model.breath_block.parameters(),
            'lr': self.args.learning_rate,
            'weight_decay': self.args.weight_decay,
        }, {
            'params': model.add_on_layers.parameters(),
            'lr': self.args.learning_rate,
            'weight_decay': self.args.weight_decay,
        }, {
            'params': model.prototype_vectors,
            'lr': self.args.learning_rate,
        }]
        joint_optimizer = optim_cls(joint_optimizer_specs)

        warm_optimizer_specs = [{
            'params': model.add_on_layers.parameters(),
            'lr': self.args.learning_rate,
            'weight_decay': self.args.weight_decay,
        }, {
            'params': model.prototype_vectors,
            'lr': self.args.learning_rate,
        }]
        warm_optimizer = optim_cls(warm_optimizer_specs)
        last_layer_optimizer_specs = [{
            'params': model.last_layer.parameters(),
            'lr': self.args.learning_rate,
            'weight_decay': self.args.weight_decay,
        }]
        last_layer_optimizer = optim_cls(last_layer_optimizer_specs)
        return [warm_optimizer, joint_optimizer, last_layer_optimizer]

    def calc_loss(self, model, cls_output, target, min_distances):
        # compute loss. change to BCE with logits because of the binary classification
        # problem we are dealing with.
        cls_loss = F.binary_cross_entropy(F.softmax(cls_output, dim=1), target)
        label = target.argmax(dim=1)
        # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
        # calculate cluster cost
        # just pick which prototypes are relevant for each observation
        prototypes_of_correct_class = self.cuda_wrapper(torch.t(model.prototype_class_identity[:,label]))
        inverted_distances, _ = torch.max((self.max_dist - min_distances) * prototypes_of_correct_class, dim=1)
        cluster_cost = torch.mean(self.max_dist - inverted_distances)

        # calculate separation cost
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        inverted_distances_to_nontarget_prototypes, _ = \
            torch.max((self.max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
        separation_cost = torch.mean(self.max_dist - inverted_distances_to_nontarget_prototypes)

        # calculate avg cluster cost
        avg_separation_cost = \
            torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
        avg_separation_cost = torch.mean(avg_separation_cost)

        if self.args.use_l1:
            l1_mask = 1 - self.cuda_wrapper(torch.t(model.prototype_class_identity))
            l1 = (model.last_layer.weight * l1_mask).norm(p=1)
        else:
            l1 = torch.zeros(1).norm(p=1)

        # XXX this is if we are not class_specific. What does this mean? do we
        # need class specific? I know sep cost doesnt come into effect in
        # non-class_specific
        #else:
        #    min_distance, _ = torch.min(min_distances, dim=1)
        #    cluster_cost = torch.mean(min_distance)
        #    l1 = model.last_layer.weight.norm(p=1)

        # compute gradient and do SGD step
        #
        # interesting. cluster and sep losses come down during run, but
        # cls_loss can remain pretty high. I wonder if removing sep loss
        # helps this at all
        #
        # sep seems to come down regardless. having no sep loss actually seems
        # to help a bit as I predicted. Although there will need to
        # be some real empirical experiment to ensure this. There's
        # also the need of adding grad clipping here because the gradients
        # can def. explode with some batches.
        #
        # It would be nice to investigate lowering cluster loss. Dont think
        # that its always a reasonable assumption that we have a prototype well
        # represented in a single read.
        loss = cls_loss + self.args.clust_lambda * cluster_cost + self.args.sep_lambda * separation_cost + 1e-4 * l1
        return loss, cls_loss, cluster_cost, separation_cost, l1

    def _process_test_batch_results(self, outputs, target, inputs, fold_num):
        batch_preds = outputs.argmax(dim=-1).cpu()
        target = target.argmax(dim=1).cpu()
        self.record_testing_results(target, batch_preds, fold_num)
        return batch_preds.tolist()

    def run_train_epoch(self, model, train_loader, optimizer, epoch_num, fold_num):
        # and then what they start doing is pushing prototypes. I dont know what this
        # means. It seems like the prototypes are pushed towards the nearest patch.
        # Yes the paper describes this as the method of how the prototypes are
        # overlaid with the img.
        print('train instances: {}'.format(len(train_loader)))
        if epoch_num <= self.args.n_warm_epochs:
            # choose only the warm optimizer
            optim = optimizer[0]
        else:
            # choose the joint optimizer
            optim = optimizer[1]

        with torch.enable_grad():
            print("\nrun epoch {}\n".format(epoch_num))
            total_batches = len(train_loader)
            for batch_idx, (obs_idx, seq, metadata, target) in enumerate(train_loader):
                inputs = self.cuda_wrapper(Variable(seq).float())
                target = self.cuda_wrapper(Variable(target).float())

                cls_output, min_distances = model(inputs, None)
                loss, cls_loss, cluster_cost, separation_cost, l1 = self.calc_loss(model, cls_output, target, min_distances)
                optim.zero_grad()
                loss.backward()
                optim.step()
                self.results.update_meter('cls_loss', fold_num, cls_loss.data)
                self.results.update_meter('clst_loss', fold_num, cluster_cost.data)
                self.results.update_meter('sep_loss', fold_num, separation_cost.data)
                self.results.update_meter('l1_loss', fold_num, l1.data)
                self.results.update_meter('loss_epoch_{}'.format(epoch_num), fold_num, loss)
                self.results.update_loss(fold_num, loss.data)

                # print individual loss and total loss
                if not self.args.no_print_progress or self.args.print_progress:
                    print("batch num: {}/{}, avg loss: {}, cls_loss: {}\r".format(
                        batch_idx,
                        total_batches,
                        self.results.get_meter('loss_epoch_{}'.format(epoch_num), fold_num),
                        self.results.get_meter('cls_loss', fold_num),
                        end=""
                    ))

                if self.args.debug:
                    break

        if epoch_num >= self.args.push_start_epoch and (epoch_num - self.args.push_start_epoch) % self.args.push_every_n == 0:
            # saving prototype push viz isnt that helpful on training because
            # prototypes are subject to change. its more helpful on test
            self.push_func(train_loader, model, epoch_num)
            optim = optimizer[2]
            with torch.enable_grad():
                for iter in range(self.args.n_push_iters):
                    print("\nrun iter {}\n".format(iter))
                    total_batches = len(train_loader)
                    for batch_idx, (obs_idx, seq, metadata, target) in enumerate(train_loader):
                        inputs = self.cuda_wrapper(Variable(seq).float())
                        target = self.cuda_wrapper(Variable(target).float())

                        output, min_distances = model(inputs, None)
                        loss, cls_loss, cluster_cost, separation_cost, l1 = self.calc_loss(model, output, target, min_distances)
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        self.results.update_meter('cls_loss', fold_num, cls_loss.data)
                        self.results.update_loss(fold_num, loss.data)
                        # print individual loss and total loss
                        if not self.args.no_print_progress or self.args.print_progress:
                            print("batch num: {}/{}, cls_loss: {}\r".format(
                                batch_idx,
                                total_batches,
                                self.results.get_meter('cls_loss', fold_num),
                                end=""
                            ))
                    if self.args.debug:
                        break

    def run_test_epoch(self, epoch_num, model, test_dataset, test_loader, fold_num):
        self.preds = []
        self.pred_idx = []
        with torch.no_grad():
            model.eval()
            for batch_idx, (obs_idx, seq, metadata, target) in enumerate(test_loader):
                inputs = self.cuda_wrapper(Variable(seq).float())
                target = self.cuda_wrapper(Variable(target).float())

                outputs, min_distances = model(inputs, None)
                loss, cls_loss, cluster_cost, separation_cost, l1 = self.calc_loss(
                    model, outputs, target, min_distances
                )
                outputs = F.softmax(outputs, dim=1)
                self.results.update_meter('test_loss', fold_num, loss.data)
                self.results.update_epoch_meter('test_loss', epoch_num, loss.data)
                batch_preds = self._process_test_batch_results(outputs, target, inputs, fold_num)
                self.pred_idx.extend(self.transform_obs_idx(obs_idx, outputs).cpu().tolist())
                self.preds.extend(batch_preds)

        if epoch_num >= self.args.viz_start_epoch and (epoch_num-self.args.viz_start_epoch) % self.args.viz_every_n == 0:
            if not self.is_2d_dataset and not self.is_2x1d_dataset:
                prototype_viz(
                    test_loader,
                    model,
                    root_dir_for_saving_prototypes=self.args.prototype_results_dir,
                    epoch_num=epoch_num,
                    cuda_wrapper=self.cuda_wrapper,
                    proto_seq_filename_prefix=self.args.prototype_fname_prefix,
                )
            else:
                self.pusher.prototype_viz(
                    test_loader,
                    model,
                    self.args.prototype_results_dir,
                    self.args.prototype_fname_prefix,
                    epoch_num,
                    fold_num,
                )

        self.record_final_epoch_testing_results(fold_num, epoch_num, test_dataset)


class ProtoPNet1DModel(ProtoPNetModel, BaseTraining, PatientClassifierMixin):
    def __init__(self, args):
        super().__init__(args)

    def get_network(self, base_network):
        model = construct_PPNet(
            base_network,
            sub_batch_size=self.args.n_sub_batches,
            prototype_shape=(self.args.n_prototypes*2, 128, 1),
            batch_size=self.args.batch_size,
            incorrect_strength=self.args.incorrect_strength,
            average_linear=self.args.average_linear_layer,
        )
        self.max_dist = (model.prototype_shape[1] * model.prototype_shape[2])
        return model

    def push_func(self, train_loader, model, epoch_num):
        push_prototypes(train_loader, model, self.cuda_wrapper)


class ProtoPNet2DModel(ProtoPNetModel, BaseTraining, PatientClassifierMixin):
    def __init__(self, args):
        super().__init__(args)
        self.pusher = Pusher(self.cuda_wrapper, self.args.experiment_name)

    def get_network(self, base_network):
        model = construct_PPNet2D(base_network, prototype_shape=(self.args.n_prototypes*2, 128, 1, 1))
        self.max_dist = (model.prototype_shape[1] * model.prototype_shape[2] * model.prototype_shape[3])
        return model

    def push_func(self, train_loader, model, epoch_num):
        self.pusher.push_orig(train_loader, model, epoch_num)


class DetectionModel(BaseTraining, PatientClassifierMixin):
    def run_train_epoch(self, model, train_loader, optimizer, epoch_num, fold_num):
        print('train instances: {}'.format(len(train_loader)))
        with torch.enable_grad():
            print("\nrun epoch {}\n".format(epoch_num))
            model.train()
            model.detector.training = True
            total_batches = len(train_loader)
            for batch_idx, (obs_idx, seq, _, bbox_target) in enumerate(train_loader):
                model.zero_grad()
                inputs = [self.cuda_wrapper(Variable(s.float())) for s in seq]
                label_target = self.get_label_target(bbox_target)
                for idx, dict in enumerate(bbox_target):
                    for k, v in dict.items():
                        bbox_target[idx][k] = self.cuda_wrapper(Variable(v))
                if epoch_num > self.args.multitask_epochs:
                    out = model.backbone_classify(inputs)
                    bbox_loss = {
                        'classification': torch.tensor(0, requires_grad=False),
                        'bbox_regression': torch.tensor(0, requires_grad=False),
                    }
                else:
                    bbox_loss, out = model.multitarget_classify(inputs, bbox_target)
                loss = self.calc_loss(bbox_loss, out, label_target)
                loss.backward()
                optimizer.step()
                self.results.update_meter('loss_epoch_{}'.format(epoch_num), fold_num, loss)
                self.results.update_meter('cls_epoch_{}'.format(epoch_num), fold_num, bbox_loss['classification'].data)
                self.results.update_meter('bbox_epoch_{}'.format(epoch_num), fold_num, bbox_loss['bbox_regression'].data)
                if not self.args.no_print_progress or self.args.print_progress:
                    print("batch num: {}/{}, avg loss: {}, cls_loss: {} reg_loss: {}\r".format(
                        batch_idx,
                        total_batches,
                        self.results.get_meter('loss_epoch_{}'.format(epoch_num), fold_num),
                        self.results.get_meter('cls_epoch_{}'.format(epoch_num), fold_num),
                        self.results.get_meter('bbox_epoch_{}'.format(epoch_num), fold_num),
                        end=""
                    ))
                if self.args.debug:
                    break

    def get_label_target(self, bbox_target):
        label_target = torch.zeros(len(bbox_target), 2)
        for idx, t in enumerate(bbox_target):
            # this code will not work if you do >3 bboxes
            cls = 1 if t['labels'].sum() > 1 else 0
            label_target[idx, cls] = 1
        return self.cuda_wrapper(label_target)

    def run_test_epoch(self, epoch_num, model, test_dataset, test_loader, fold_num):
        print('test instances: {}'.format(len(test_loader)))
        self.pred_idx = []
        self.preds = []
        with torch.no_grad():
            print("\nrun test epoch {}\n".format(epoch_num))
            model.eval()
            model.detector.training = False
            total_batches = len(test_loader)
            for batch_idx, (obs_idx, seq, _, target) in enumerate(test_loader):
                inputs = [self.cuda_wrapper(s.float()) for s in seq]
                outputs = model.backbone_classify(inputs)
                batch_preds = self._process_test_batch_results(outputs, target, fold_num)
                self.pred_idx.extend(self.transform_obs_idx(obs_idx, outputs))
                self.preds.extend(batch_preds)
        self.record_final_epoch_testing_results(fold_num, epoch_num, test_dataset)

    def _process_test_batch_results(self, outputs, target, fold_num):
        batch_preds = outputs.argmax(dim=-1).cpu()
        target = np.array([t.argmax() for t in target])
        self.record_testing_results(target, batch_preds, fold_num)
        return batch_preds.tolist()

    def _process_bbox_test_batch_results(self, outputs, target, fold_num):
        batch_preds = []
        for idx, item in enumerate(outputs):
            other_pred = box_area(item['boxes'][item['labels'] == 0]).sum()
            ards_pred = box_area(item['boxes'][item['labels'] == 1]).sum()
            if ards_pred >= other_pred:
                batch_preds.append(1)
            else:
                batch_preds.append(0)
        target = np.array([t.argmax() for t in target])
        batch_preds = np.array(batch_preds)
        self.record_testing_results(target, batch_preds, fold_num)
        return batch_preds.tolist()

    def transform_obs_idx(self, obs_idx, outputs):
        return list(obs_idx)


class RetinaNetBBoxModel(DetectionModel):
    def __init__(self, args):
        super().__init__(args)

    def calc_loss(self, bbox_loss, output, label_target):
        cls_loss = F.binary_cross_entropy_with_logits(output, label_target)
        return bbox_loss['classification'] + bbox_loss['bbox_regression'] + cls_loss

    def get_network(self, base_network):
        return RetinaNetMain(base_network)


# Is putting a global in code like this a great idea? probably not, but saves some hassle
# on outside scripting
network_map = {
    'cnn_lstm': CNNLSTMModel,
    'cnn_linear': CNNLinearModel,
    'cnn_linear_2d': CNNLinearModel2D,
    'cnn_linear_2x1d': CNNLinearModel2D,
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
    'lstm_only_with_packing': LSTMOnlyWithPackingModel,
    'double_lstm': DoubleLSTMModel,
    'cnn_to_nested_rnn': CNNToNestedRNNModel,
    'cnn_to_nested_lstm': CNNToNestedLSTMModel,
    'cnn_to_nested_transformer': CNNToNestedTransformerModel,
    'cnn_linear_compr_to_rf': CNNLinearComprToRFModel,
    'cnn_linear_to_mean': CNNLinearToMeanModel,
    'protopnet': ProtoPNet1DModel,
    'protopnet_2d': ProtoPNet2DModel,
    'retinanet_2d': RetinaNetBBoxModel,
    'retinanet_2x1d': RetinaNetBBoxModel,
}


def build_parser():
    parser = argparse.ArgumentParser()
    true_false_flag = lambda flag, help: parser.add_argument(
        flag, action='store_true', help=help, default=None
    )
    parser.add_argument('-co', '--config-override', help='path to yml file that overrides elements of defaults.yml')
    parser.add_argument('-dp', '--data-path', help='Path to ARDS detection dataset')
    parser.add_argument('-en', '--experiment-num', type=int)
    parser.add_argument('-c', '--cohort-file')
    parser.add_argument('-n', '--network', choices=list(network_map.keys()))
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-p', '--train-from-pickle')
    parser.add_argument('--train-to-pickle')
    parser.add_argument('--test-from-pickle')
    parser.add_argument('--test-to-pickle')
    true_false_flag('--cuda', '')
    true_false_flag('--cuda-no-dp', '')
    parser.add_argument('-b', '--batch-size', type=int)
    parser.add_argument('--base-network', choices=base_networks)
    parser.add_argument('-lc', '--loss-calc', choices=['all_breaths', 'last_breath'])
    parser.add_argument('-nb', '--n-sub-batches', type=int, help=(
        "number of breath-subbatches for each breath frame. This has different "
        "meanings for different dataset types. For breath_by_breath this means the "
        "number of individual breaths in each breath frame. For unpadded_sequences "
        "this means the number of contiguous flow measurements in each frame."))
    true_false_flag('--no-print-progress', '')
    parser.add_argument('--kfolds', type=int)
    parser.add_argument('-rip', '--initial-planes', type=int)
    parser.add_argument('-rfpt', '--resnet-first-pool-type', choices=['max', 'avg'])
    true_false_flag('--no-test-after-epochs', '')
    true_false_flag('--debug', 'debug code and dont train')
    parser.add_argument('--optimizer', choices=['adam', 'sgd'])
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
        'unpadded_downsampled_autoencoder_sequences',
        'unpadded_centered_with_bm',
    ])
    parser.add_argument('-lr', '--learning-rate', type=float)
    parser.add_argument('--loader-threads', type=int, help='specify how many threads we should use to load data. Sometimes the threads fail to shutdown correctly though and this can cause memory errors. If this happens a restart works well')
    parser.add_argument('--save-model', help='save the model to a specific file')
    true_false_flag('--save-model-per-epoch', 'save the model at the end of each epoch')
    parser.add_argument('--load-base-network', help='load base network only from a saved model')
    parser.add_argument('--load-checkpoint', help='load a checkpoint of the model for further training or inference')
    true_false_flag('--no-train', 'Dont train model, just evaluate for inference')
    true_false_flag('--resnet-double-conv', '')
    true_false_flag('--bm-to-linear', '')
    parser.add_argument('-exp', '--experiment-name')
    parser.add_argument('--downsample-factor', type=float)
    parser.add_argument('-wd', '--weight-decay', type=float)
    parser.add_argument('-loss', '--loss-func', choices=['bce', 'vacillating', 'confidence', 'focal'], help='This option only works for classification. Choose the loss function you want to use for classification purposes: BCE or vacillating loss.')
    parser.add_argument('--valpha', type=float, default=float('Inf'), help='alpha value to use for vacillating loss. Lower alpha values mean vacillating loss will contribute less to overall loss of the system. Default value is inf')
    parser.add_argument('--conf-beta', type=float, default=1.0, help='Modifier to the intensity of the confidence penalty')
    parser.add_argument('--time-series-hidden-units', type=int)
    parser.add_argument('--transformer-blocks', type=int)
    true_false_flag('--unshuffled', 'dont shuffle data for lstm processing')
    parser.add_argument('--load-siamese', help='load a siamese network pretrained model')
    parser.add_argument('--fl-gamma', type=float)
    parser.add_argument('--fl-alpha', type=float)
    true_false_flag('--oversample-minority', '')
    parser.add_argument('--oversample-all-factor', type=float, help='amount by which we should randomly oversample all observations in dataset. Can be helpful to run in conjunction with augmentation')
    parser.add_argument('-usf', '--undersample-factor', type=float)
    parser.add_argument('-usdf', '--undersample-std-factor', type=float)
    true_false_flag('--reshuffle-oversample-per-epoch', '')
    true_false_flag('--freeze-base-network', '')
    true_false_flag('--stop-on-loss', '')
    parser.add_argument('--stop-thresh', type=float)
    parser.add_argument('--stop-after-epoch', type=int)
    true_false_flag('--clip-grad', '')
    parser.add_argument('--clip-val', type=float)
    parser.add_argument('--holdout-set-type', help="""Choose which holdout set to use:
        main: regular train/test holdout split
        proto: holdout set that is used for prototyping
        random: a random split of data
        <custom named>: A customly named split that was made.
    """)
    true_false_flag('--final-validation', 'Ensure you dont use validation set, but rather testing set. argument that is only used for random holdout sets.')
    true_false_flag('--plot-untiled-disease-evol', 'Plot the our ARDS/non-ARDS predictions by hour')
    true_false_flag('--plot-tiled-disease-evol', 'Plot the our ARDS/non-ARDS predictions by hour but in tiled format grouped by TPs/TNs/FPs/FNs')
    true_false_flag('--plot-dtw-with-disease', 'Plot DTW with ARDS/non-ARDS predictions by hour')
    parser.add_argument('--plot-pt-dtw-by-minute', help='Plot DTW and predictions by minute for a single patient.')
    true_false_flag('--perform-dtw-preprocessing', 'perform DTW preprocessing actions even if we dont want to visualize DTW')
    parser.add_argument('--train-pt-frac', type=float, help='Fraction of random training patients to use')
    parser.add_argument('--cuda-device', type=int, help='number of cuda device you want to use')
    parser.add_argument('--transforms', choices=['ie_ww', 'naive_ww', 'ie_ww_i_or_e'], nargs='*', help="""
        Option to determine which kinds of data augmentations to use with our modeling

        naive_ww: Naive window warping done under standard Le Guennec 2016 instructions
        ie_ww: I/E window warping
        ie_ww_i_or_e: I/E window warping but with choice of using either I or E all the time
    """)
    parser.add_argument('-tp', '--transform-probability', type=float, help='Probability that a modifying transform will be activated for a sub-batch')
    true_false_flag('--use-i', 'Argument only used if ie_ww_i_or_e transform is triggered. If you set this arg on CLI then insp lim will only be used. Otherwise exp. lim is used.')
    parser.add_argument('-r2', '--drop-if-under-r2', type=float)
    true_false_flag('--drop-i-lim', '')
    true_false_flag('--drop-e-lim', '')
    parser.add_argument('--truncate-e-lim', type=float, help='Number of seconds of the E lim to keep. Everything afterwards is discarded. Should be done in a number divisible by 2', default=None)
    parser.add_argument('--only-fold', type=int, default=None, help='only run specific fold')
    parser.add_argument('--n-warm-epochs', type=int, help='number of warm epochs to run with protopnet')
    parser.add_argument('-pse', '--push-start-epoch', type=int, help='epoch to start the projection pushes')
    parser.add_argument('--push-every-n', type=int, help='starting at push start, push every n epochs')
    parser.add_argument('--n-push-iters', type=int, help='number of iterations to optimize last layer following push')
    parser.add_argument('--clust-lambda', type=float, help='multiplier of cluster loss for protopnet')
    parser.add_argument('--sep-lambda', type=float, help='multiplier of separation loss for protopnet')
    parser.add_argument('-vse', '--viz-start-epoch', type=int, help='epoch to start visualizing prototypes')
    parser.add_argument('--viz-every-n', type=int, help='visualize prototypes every n eepochs')
    parser.add_argument('--prototype-results-dir', help='directory to save prototype visualizations')
    parser.add_argument('--prototype-fname-prefix', help='prefix to save prototype visualization filenames')
    parser.add_argument('-np', '--n-prototypes', type=int, help='number of prototypes to use per class in our model')
    parser.add_argument('-ic', '--incorrect-strength', type=float, help='the incorrect strength value for the linear layer of protopnet')
    parser.add_argument('--saved-models-dir', help='directory to save models')
    true_false_flag('--average-linear-layer', 'instead of flattening the linear layer average all like features together. currently this only works for protopnet')
    true_false_flag('--use-l1', 'add an l1 regularization for ppnet')
    true_false_flag('--print-progress', 'print progress for batch losses. This will override --no-print-progress flag if set')
    parser.add_argument('-2dt', '--two-dim-transforms', nargs='*', choices=two_dim_transforms.keys())
    true_false_flag('--with-fft', 'add FFT transforms to a 2d dataset')
    true_false_flag('--only-fft', 'only use FFT when using a 2d dataset. if you use this with --with-fft then --with-fft will take precedence and you will have a 3 chan input')
    parser.add_argument('-bks', '--block-kernel-size', type=int, help='kernel size of the main dense block convolution')
    parser.add_argument('--multitask-epochs', type=int)
    true_false_flag('--row-mix', 'mix row segments together from patients of the same pathophysiology')
    true_false_flag('--fft-real-only', 'Only incorporate the real component of FFT')
    true_false_flag('--reload-dataset-per-epoch', 'Reload your dataset for each epoch. Is useful in cases like the bounding box transforms and row mixing, because data is saved on loading rather than being performed ad-hoc.')
    return parser


def main():
    args = build_parser().parse_args()
    args = Configuration(args)

    # convenience code
    if args.load_siamese:
        args.network = 'siamese_pretrained'

    if args.fl_alpha > 1 or args.fl_alpha < 0:
        raise Exception('Focal loss alpha must be between 0 and 1')

    if args.save_model_per_epoch and not args.save_model:
        raise Exception('Must specify a filename to save your model using --save-model')
    cls = network_map[args.network](args)
    cls.train_and_test()


if __name__ == "__main__":
    main()
