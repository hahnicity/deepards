"""
protopnet_analysis
~~~~~~~~~~~~~~~~~

Helper functions for analyzing the protopnet
"""
import argparse
import math
import os
import uuid

import IPython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import torch
from torch.utils.data import DataLoader

from deepards.dataset import ARDSRawDataset
from deepards.models.protopnet1d.ppnet_push import viz_single_prototype

cuda_wrapper = lambda x, is_cuda: x.cuda() if is_cuda else x


class ProtoPNetAnalysis(object):
    def __init__(self, model, x_train, x_test, is_cuda):
        """
        :param model:
        :param x_train:
        :param x_test:
        """
        self.x_train_ds = x_train
        self.x_test_ds = x_test
        self.model = cuda_wrapper(model, is_cuda)
        self.is_cuda = is_cuda
        # account for some source code changes
        try:
            self.model.breath_block = self.model.features
        except AttributeError:
            pass

        self.train_loader = DataLoader(x_train, batch_size=1, shuffle=False)
        self.test_loader = DataLoader(x_test, batch_size=1, shuffle=False)
        self.gather_data()

        self.mlp = self.make_mlp_classifier()
        self.train_preds = self.mlp.predict_proba(self.all_train_outputs)
        self.test_preds = self.mlp.predict_proba(self.all_test_outputs)

    def gather_data(self):
        self.all_train_outputs, self.all_train_dists, self.all_train_targets = self.translate_torch_dataset_to_numpy(self.train_loader)
        self.all_test_outputs, self.all_test_dists, self.all_test_targets = self.translate_torch_dataset_to_numpy(self.test_loader)
        self.train_gt = self.x_train_ds.get_ground_truth_df()
        self.test_gt = self.x_test_ds.get_ground_truth_df()
        self.train_features, self.test_features = self.make_feature_datasets()

    def translate_torch_dataset_to_numpy(self, loader):
        all_outputs = []
        all_dists = []
        all_targets = []

        with torch.no_grad():
            for _, seq, __, target in loader:
                inputs = cuda_wrapper(seq.float(), self.is_cuda)
                all_targets.append(target)
                for i in range(seq.shape[0]):
                    outputs, min_distances = self.model.seq_forward(inputs[i])
                    all_outputs.append(outputs.view(-1).cpu())
                    all_dists.append(min_distances.cpu())

        all_outputs = torch.stack(all_outputs, dim=0).numpy()
        all_dists = torch.stack(all_dists, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        return all_outputs, all_dists, all_targets

    def make_feature_datasets(self):
        # ensure that you name all the features in the model appropriately
        features = []
        n_prototypes = self.model.prototype_shape[0]
        for i in range(self.all_train_outputs.shape[1]):
            proto_n = i % n_prototypes
            # step function here that maps 0-7 -> 0, 8-15 -> 1 ...
            breath_n = int((i + n_prototypes - (i % n_prototypes))/n_prototypes - 1)
            # if we want, we can use subscripting, however I've also found
            # that it makes it more difficult to read
            features.append(u'prototype {},{}'.format(breath_n, proto_n))

        train = pd.DataFrame(self.all_train_outputs, columns=features, index=self.train_gt.index)
        test = pd.DataFrame(self.all_test_outputs, columns=features, index=self.test_gt.index)
        return train, test

    def make_mlp_classifier(self):
        """
        Setup a 1 layer sklearn MLP with softmax at the end, equivalent
        to our final linear layer in the torch network. We can use this
        classifier with SHAP/LIME methods.
        """
        mlp = MLPClassifier(hidden_layer_sizes=[], activation='identity')
        # Run fit to initialize all the variables sklearn keeps hidden.
        # This step has no impact on final weights
        mlp.fit(self.all_train_outputs[0:2], self.all_train_targets[0:2])

        torch_weights = self.model.last_layer.weight.detach().cpu().numpy().T
        # set mlp weights to the ones found in torch
        mlp.coefs_ = [torch_weights]
        # turn off bias
        mlp.intercepts_ = [np.array([0, 0])]
        mlp.out_activation_ = 'softmax'
        return mlp

    def visualize_specific_prototypes(self, sequence_data, y, prototypes_of_interest):
        with torch.no_grad():
            protoL_input, distances = self.model.push_forward(torch.FloatTensor(sequence_data).cuda().unsqueeze(0))

        for idx, proto in enumerate(prototypes_of_interest):
            breath_n, proto_n = proto.split(' ')[1].split(',')
            breath_n, proto_n = int(breath_n), int(proto_n)
            viz_single_prototype(self.model, sequence_data, y, protoL_input, distances, proto_n, breath_n, True, True, True)
            plt.show()

    def plot_random_proto_from_linear_with_topk(self, gt_patho, pred_patho, topk):
        """
        """
        gt_patho_n = {'ards': 1, 'non_ards': 0}[gt_patho]
        pred_patho_n = {'ards': 1, 'non_ards': 0}[pred_patho]
        test_pred_labels = self.test_preds.argmax(axis=1)
        match = self.test_gt[(self.test_gt.y == gt_patho_n) & (test_pred_labels == pred_patho_n)]
        idx = np.random.choice(match.index)
        if isinstance(self.test_gt.loc[idx], pd.DataFrame):
            iloc = np.argwhere(self.test_gt.index.get_loc(idx)).flatten()[0]
            features = self.test_features.loc[idx].iloc[0]
        else:
            iloc = self.test_gt.index.get_loc(idx)
            features = self.test_features.loc[idx]
        _, seq, __, ___ = self.x_test_ds[iloc]
        mlp_out = self.mlp.coefs_[0] * np.expand_dims(features, axis=1)
        # focused on predicted proto class
        protos = self.test_features.columns[mlp_out[:, pred_patho_n].argsort()[::-1][:topk]]
        breath_n, proto_n = np.random.choice(protos).split(' ')[1].split(',')
        breath_n, proto_n = int(breath_n), int(proto_n)
        proto_input, distances = self.model.push_forward(
            cuda_wrapper(torch.FloatTensor(seq), self.is_cuda).unsqueeze(0)
        )
        viz_single_prototype(self.model, seq, gt_patho_n, proto_input, distances, proto_n, breath_n, False, False, False)
        return idx, breath_n, proto_n

    def make_random_sequence_pane(self, dirname):
        """
        :param dirname:
        """
        items_per_frame = 16
        graph_id = uuid.uuid4()
        data_record = []
        patho_iter = ['ards'] * 8 + ['non_ards'] * 8
        np.random.shuffle(patho_iter)

        for i in range(items_per_frame):
            p = patho_iter[i]
            plt.subplot(int(math.sqrt(items_per_frame)), int(math.sqrt(items_per_frame)), i+1)
            # XXX just plot correctly predicted items for now
            seq_idx, breath_n, proto_n = self.plot_random_proto_from_linear_with_topk(p, p, 40)
            data_record.append([str(i+1), p, str(seq_idx), str(breath_n), str(proto_n)])

        title = "Random Prototype Viz"
        graph_filename = os.path.join(dirname, "sample.png")
        self._finalize_multi_plot_graph(title)
        plt.savefig(graph_filename, dpi=400)
        plt.close()
        with open(graph_filename.replace('png', 'txt'), 'w') as record:
            record.write('n, patho, gt_idx, breath_n, proto_n\n')
            for line in data_record:
                record.write(', '.join(line)+'\n')

    def _finalize_multi_plot_graph(self, title):
        # XXX
        fig = plt.gcf()
        sm = fig.axes[0].pcolormesh(np.random.random((0, 0)), vmin=0, vmax=255)
        fig.set_size_inches(20, 10)
        fig.subplots_adjust(right=.8)
        #cbar_ax = fig.add_axes((0.85, 0.15, 0.025, 0.7))
        #cbar = fig.colorbar(sm, cax=cbar_ax)
        #cbar.set_label("Intensity")
        plt.suptitle(title)


def load_ppnet_from_scratch(args, model):
    if args.kfold_idx:
        x_train = ARDSRawDataset.from_pickle(args.kfold_from_pickle, True, 1.0, None, -1, None)
        x_train.set_kfold_indexes_for_fold(args.kfold_idx)
        x_test = ARDSRawDataset.make_test_dataset_if_kfold(x_train)
    else:
        x_train = ARDSRawDataset.from_pickle(args.holdout_train_pickle, True, 1.0, None, -1, None)
        x_test = ARDSRawDataset.from_pickle(args.holdout_test_pickle, False, 1.0, None, -1, None)

    return ProtoPNetAnalysis(model, x_train, x_test, True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='saved model checkpoint')
    parser.add_argument('--kfold-from-pickle', default="/fastdata/deepards/unpadded_centered_with_bm-nb20-kfold.pkl", help='pickled dataset path for the train cohort')
    parser.add_argument('--kfold-idx', type=int, help='kfold index to use in our dataset. if unset this will assume we want to use the holdout set')
    parser.add_argument('--holdout-train-pickle', default='/fastdata/deepards/unpadded_centered_sequences-nb20-aim1_holdout_train_redo.pkl')
    parser.add_argument('--holdout-test-pickle', default='/fastdata/deepards/unpadded_centered_sequences-nb20-aim1_holdout_test_redo.pkl')
    parser.add_argument('-cls', '--analysis-class-from-pickle', help='read pickled version of the ProtoPNetAnalysis class')
    parser.add_argument('-tp', '--analysis-class-to-pickle', help='save pickled version of the ProtoPNetAnalysis class')
    args = parser.parse_args()

    model = torch.load(args.model)
    if args.analysis_class_from_pickle:
        ppnet_analysis = pd.read_pickle(args.analysis_class_from_pickle)
    else:
        ppnet_analysis = load_ppnet_from_scratch(args, model)
    if args.analysis_class_to_pickle:
        pd.to_pickle(ppnet_analysis, args.analysis_class_to_pickle)
    ppnet_analysis.make_random_sequence_pane('foo')


if __name__ == "__main__":
    main()
