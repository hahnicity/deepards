import argparse
import math
import os
import pickle
import random
import uuid

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

#importing the class for calculating the cam values
from deepards.dataset import ARDSRawDataset
from deepards.gradcam import FracTotalNormCam, MaxMinNormCam


def do_makedirs(dir):
    try:
        os.makedirs(dir)
    except OSError:
        pass


class PatientGradCam(object):
    def __init__(self, pretrained_model, data, target, results_base_dir):
        self.grad_cam = MaxMinNormCam(pretrained_model.cuda())
        self.data = data
        self.gt = self.data.get_ground_truth_df()
        self.ards, self.non_ards = self.get_ardsids_otherids()
        self.sub_batch_size = self.data.all_sequences[0][1].shape[0]
        self.breath_len = 224
        self.target = target
        self.results_base_dir = results_base_dir

    def get_ardsids_otherids(self):
        ards = self.gt[self.gt.y == 1].patient.unique()
        non_ards = self.gt[self.gt.y == 0].patient.unique()
        return ards, non_ards

    def get_target(self, ground_truth):
        if isinstance(self.target, int):
            return [self.target]
        elif self.target == 'ground_truth':
            return [ground_truth]
        elif self.target == 'both':
            return [0, 1]
        else:
            return [{'ards': 1, 'other': 0}[self.target]]

    def get_median_patient_camout(self, patient_id):
        if self.target == 'both':
            raise NotImplementedError('both mode currently doesnt support operations outside sampled_seqs')
        #initializing variables
        patient_idxs = self.gt[self.gt.patient == patient_id].index
        target = self.get_target(self.gt.loc[patient_idxs].y.iloc[0])
        mapping = {0: 'non_ards', 1: 'ards'}
        dirname = os.path.join(self.results_base_dir, 'gradcam_results', 'patient_medians', mapping[target])
        do_makedirs(dirname)

        # XXX in future make this configurable
        med_breath = np.empty((0, self.sub_batch_size, 1, self.breath_len))
        cam_outputs = np.empty((0,7))
        target_class = None
        for i in patient_idxs:
            breath_sequence = np.expand_dims(self.data[i][1], axis=0)
            med_breath = np.append(med_breath, breath_sequence, axis=0)

        med_breath = np.median(np.median(med_breath, axis=0), axis=0)
        med_breath = np.expand_dims(med_breath, axis=0)
        br = torch.FloatTensor(med_breath)[[0] * self.sub_batch_size].cuda()
        cam, model_output = self.grad_cam.generate_cam(br, target)
        cam_outputs = cv2.resize(cam, (1, self.breath_len))
        filename = os.path.join(dirname, patient_id + '_target-{}.png'.format(self.target))
        self.visualize_sequence(med_breath[0], cam_outputs, patient_id, target, model_output, target)
        plt.savefig(filename)
        plt.close()

    def get_average_patient_camout(self, patient_id):
        if self.target == 'both':
            raise NotImplementedError('both mode currently doesnt support operations outside sampled_seqs')
        #initializing variables
        patient_idxs = self.gt[self.gt.patient == patient_id].index
        ground_truth = self.gt.loc[patient_idxs].y.iloc[0]
        target = self.get_target(ground_truth)
        mapping = {0: 'non_ards', 1: 'ards'}
        dirname = os.path.join(self.results_base_dir, 'gradcam_results', 'patient_averages', mapping[target])
        do_makedirs(dirname)
        avg_breath = np.empty((0, self.breath_len))
        cam_outputs = np.empty((0,7))
        mean_out = np.zeros((1, 2))
        for i in patient_idxs:
            breath_sequence = data[i][1]
            #concadinating the 224 breath sequences to get the avg of the patient breath
            br1 = np.mean(breath_sequence, axis = 0)
            avg_breath = np.append(avg_breath, br1, axis = 0)
            #for gradcam values
            br = torch.FloatTensor(breath_sequence).cuda()
            cam, model_output = self.grad_cam.generate_cam(br, target)
            cam = np.expand_dims(cam, axis = 0)
            cam_outputs = np.append(cam_outputs, cam, axis = 0)
            mean_out += model_output
        mean_out = mean_out / len(patient_idxs)
        cam_outputs = np.mean(cam_outputs, axis = 0)
        cam_outputs = cv2.resize(cam_outputs,(1, self.breath_len))
        avg_breath = np.mean(avg_breath, axis = 0)
        filename = os.path.join(dirname, patient_id + '_target-{}.png'.format(self.target))
        self.visualize_sequence(avg_breath, cam_outputs, patient_id, ground_truth, mean_out, target)
        plt.savefig(filename)
        plt.clf()

    def get_sampled_patient_sequences_camout(self, patient_id):
        """
        Idea is that we iterate over patient batches and sample a single sequence
        from each batch. That sequence is fed into gradcam for a result
        """
        patient_idxs = self.gt[self.gt.patient == patient_id].index
        ground_truth = self.gt.loc[patient_idxs].y.iloc[0]

        for abs_idx in patient_idxs:
            rand_seq = random.choice(range(self.sub_batch_size))
            for target in self.get_target(ground_truth):
                mapping = {0: 'non_ards', 1: 'ards'}
                dirname = os.path.join(self.results_base_dir, 'gradcam_results', 'sampled_sequences', mapping[target], patient_id)
                do_makedirs(dirname)
                rel_idx = list(self.data.kfold_indexes).index(abs_idx)
                cam_outputs, br, model_output = self.get_single_sequence_grad_cam(rel_idx, rand_seq, target)
                filename = os.path.join(dirname, 'seq-{}-{}-target-{}.png'.format(abs_idx, rand_seq, self.target))
                self.visualize_sequence(br[0], cam_outputs, patient_id, ground_truth, model_output, target)
                plt.savefig(filename)
                plt.clf()

    def get_cam_by_hour(self, patient_id, hour_start, hour_end, n_sequences_per_hour):
        patient_idxs = self.gt[(self.gt.patient == patient_id) & (self.gt.hour >= hour_start) & (self.gt.hour < hour_end)].index
        if n_sequences_per_hour is not None:
            n_sequences_per_hour = n_sequences_per_hour if n_sequences_per_hour < len(patient_idxs) else len(patient_idxs)
            patient_idxs = np.random.choice(patient_idxs, size=n_sequences_per_hour, replace=False)
        ground_truth = self.gt.loc[patient_idxs].y.iloc[0]
        for abs_idx in patient_idxs:
            for seq_idx in range(self.sub_batch_size):
                for target in self.get_target(ground_truth):
                    mapping = {0: 'non_ards', 1: 'ards'}
                    dirname = os.path.join(self.results_base_dir, 'gradcam_results', 'hour_sequences', mapping[target], patient_id, str(hour_start))
                    do_makedirs(dirname)
                    rel_idx = list(self.data.kfold_indexes).index(abs_idx)
                    cam_outputs, br, model_output = self.get_single_sequence_grad_cam(rel_idx, seq_idx, target)
                    filename = os.path.join(dirname, 'seq-{}-{}-target-{}.pkl'.format(abs_idx, seq_idx, self.target))
                    self.visualize_sequence(br[0], cam_outputs, patient_id, ground_truth, model_output, target)
                    ax = plt.gca()
                    # save to pickle first, and then can convert to png if we
                    # want to later. pickle is much faster than png
                    pd.to_pickle(ax, filename, compression=None)
                    plt.clf()

    def get_full_read_patient_sequences(self, patient_id):
        patient_idxs = self.gt[self.gt.patient == patient_id].index
        ground_truth = self.gt.loc[patient_idxs].y.iloc[0]

        for abs_idx in patient_idxs:
            for target in self.get_target(ground_truth):
                mapping = {0: 'non_ards', 1: 'ards'}
                dirname = os.path.join(self.results_base_dir, 'gradcam_results', 'full_read', mapping[target], patient_id)
                do_makedirs(dirname)
                rel_idx = list(self.data.kfold_indexes).index(abs_idx)
                # XXX need to figure this out from here below
                cam_outputs, br, model_output = self.get_read_grad_cam(rel_idx, target)
                filename = os.path.join(dirname, 'seq-{}-target-{}.png'.format(abs_idx, self.target))
                self.visualize_read(br, cam_outputs, patient_id, ground_truth, filename, model_output, target)

    def plot_sequence(self, br, cam_outputs):
        br_len = len(br.ravel())
        img = br.reshape((br_len, 1))
        t = np.arange(0, br_len, 1)
        plt.scatter(t, img, c=cam_outputs, vmin = 0, vmax = 255)
        plt.plot(t, img)

    def visualize_sequence(self, br, cam_outputs, patient_id, c, model_output, cam_target):
        self.plot_sequence(br, cam_outputs)
        cbar  = plt.colorbar()
        cbar.set_label("cam_outputs", labelpad=-1)
        mapping = {0: 'Non-ARDS', 1: 'ARDS'}
        pred_prob = F.softmax(model_output, dim=1).cpu().detach().numpy().round(3)
        pred = np.argmax(pred_prob)
        plt.title('{}, ground truth: {}, pred: {}, prob: {}, cam target: {}'.format(patient_id, mapping[c], mapping[pred], pred_prob, mapping[cam_target]), fontsize=8)

    def visualize_read(self, br, cam_outputs, patient_id, c, filepath, model_output, cam_target):
        fig = plt.figure(figsize=(3*8, 3*4))
        fig.add_subplot(1, 1, 1)
        half_len = int(len(br.ravel()) / 2)
        self.plot_sequence(br.ravel()[:half_len], cam_outputs.ravel()[:half_len])
        cbar  = plt.colorbar()
        cbar.set_label("cam_outputs", labelpad=-1)
        mapping = {0: 'Non-ARDS', 1: 'ARDS'}
        pred_prob = F.softmax(model_output, dim=1).cpu().detach().numpy().round(3)
        pred = np.argmax(pred_prob)
        plt.title('{}, ground truth: {}, pred: {}, prob: {}, cam target: {}'.format(patient_id, mapping[c], mapping[pred], pred_prob, mapping[cam_target]))
        plt.tight_layout()
        plt.xlim(-1, half_len+1)
        #plt.show()
        plt.savefig(filepath)
        plt.close()

    def get_camout_for_breath(self, br, target):
        if len(br.shape) == 1:
            br = np.expand_dims(np.expand_dims(br, axis=0), axis=0)
        elif len(br.shape) == 2:
            br = np.expand_dims(br, axis=0)

        # make sure the breath is duplicated <batch_size> times
        br = torch.FloatTensor(br)[[0] * self.sub_batch_size].cuda()
        cam, model_output = self.grad_cam.generate_cam(br, target)
        cam_outputs = cv2.resize(cam, (1, self.breath_len))
        return cam_outputs, br.cpu().numpy(), model_output

    def get_camout_for_read(self, br, target):
        if len(br.shape) == 1:
            br = np.expand_dims(np.expand_dims(br, axis=0), axis=0)
        elif len(br.shape) == 2:
            br = np.expand_dims(br, axis=0)
        br = torch.FloatTensor(br).cuda()
        cam, model_output = self.grad_cam.generate_read_cam(br, target)
        cam_outputs = np.zeros((20, self.breath_len))
        for i, cam_line in enumerate(cam):
            cam_outputs[i] = cv2.resize(cam[i], (1, self.breath_len)).ravel()
        return cam_outputs, br.cpu().numpy(), model_output

    def get_single_sequence_grad_cam(self, seq_idx, batch_idx, target):
        item = self.data[seq_idx]
        br = np.expand_dims(item[1][batch_idx], axis=0)
        return self.get_camout_for_breath(br, target)

    def get_read_grad_cam(self, seq_idx, target):
        item = self.data[seq_idx]
        return self.get_camout_for_read(item[1].round(4), target)

    def _plot_single_random_sequence(self, patho):
        target = {'ards': 1, 'non_ards': 0}[patho]
        patho_idxs = self.gt[self.gt.y == target].index
        abs_idx = random.choice(patho_idxs)
        br_idx = random.randint(0, self.sub_batch_size-1)

        rel_idx = list(self.data.kfold_indexes).index(abs_idx)
        cam_outputs, br, model_output = self.get_single_sequence_grad_cam(rel_idx, br_idx, target)
        self.plot_sequence(br[0], cam_outputs)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.yticks(fontsize='x-small')
        return abs_idx, br_idx

    def _finalize_multi_plot_graph(self, title):
        fig = plt.gcf()
        sm = fig.axes[0].pcolormesh(np.random.random((0, 0)), vmin=0, vmax=255)
        fig.set_size_inches(20, 10)
        fig.subplots_adjust(right=.8)
        cbar_ax = fig.add_axes((0.85, 0.15, 0.025, 0.7))
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Intensity")
        plt.suptitle(title)

    def _make_titled_sequence_pane(self, patho, dirname):
        """
        :param patho: either ards, non_ards, or random
        """
        items_per_frame = 16
        graph_id = uuid.uuid4()
        data_record = []
        if patho == 'random':
            patho_iter = ['ards'] * 8 + ['non_ards'] * 8
            np.random.shuffle(patho_iter)
        else:
            patho_iter = [patho] * items_per_frame

        for i in range(items_per_frame):
            p = patho_iter[i]
            plt.subplot(int(math.sqrt(items_per_frame)), int(math.sqrt(items_per_frame)), i+1)
            s_i, b_i = self._plot_single_random_sequence(p)
            data_record.append([str(i+1), p, str(s_i), str(b_i)])

        title = "{} Grad-Cam".format({'random': 'Random', 'non_ards': 'Non-ARDS', 'ards': 'ARDS'}[patho])
        graph_filename = os.path.join(dirname, "{}-sample-{}.png".format(patho, graph_id))
        self._finalize_multi_plot_graph(title)
        plt.savefig(graph_filename, dpi=400)
        plt.close()
        with open(graph_filename.replace('png', 'txt'), 'w') as record:
            record.write('n, patho, sequence_idx, breath_idx\n')
            for line in data_record:
                record.write(', '.join(line)+'\n')

    def rand_sample(self, randomize_groups=False):
        if not randomize_groups:
            dirname = os.path.join(self.results_base_dir, 'gradcam_results', 'rand_sample', 'non_random')
            do_makedirs(dirname)
            for _ in range(3):
                self._make_titled_sequence_pane('ards', dirname)

            for _ in range(3):
                self._make_titled_sequence_pane('non_ards', dirname)
        else:
            dirname = os.path.join('gradcam_results', 'rand_sample', 'randomized')
            do_makedirs(dirname)
            for _ in range(6):
                self._make_titled_sequence_pane('random', dirname)

    def perform_dtw_clustering(self, patient_id):
        # this is against pep8. so sue me.
        from dtwco import dtw
        from deepards.mediods import KMedoids
        patient_idxs = self.gt[self.gt.patient == patient_id].index
        ground_truth = self.gt.loc[patient_idxs].y.iloc[0]
        # Can make this configurable in future if we want.
        sequence_thresh = .8
        seq_min_len = 5

        sequences = []
        for abs_idx in patient_idxs:
            for target in self.get_target(ground_truth):
                mapping = {0: 'non_ards', 1: 'ards'}
                dirname = os.path.join(self.results_base_dir, 'gradcam_results', 'dtw_clustering', mapping[target], patient_id)
                do_makedirs(dirname)
                rel_idx = list(self.data.kfold_indexes).index(abs_idx)
                # XXX need to figure this out from here below
                cam_outputs, br, model_output = self.get_read_grad_cam(rel_idx, target)
                filename = os.path.join(dirname, 'seq-{}-target-{}.png'.format(abs_idx, self.target))
                # find contiguous periods above threshold
                tmp = pd.DataFrame(cam_outputs)
                mask = tmp >= (sequence_thresh*255)
                tmp[mask] = 1
                tmp[~mask] = np.nan
                for i in range(self.sub_batch_size):
                    br_cam_row = tmp.loc[i]
                    group_sizes = br_cam_row.groupby(
                        (br_cam_row.shift(1) != br_cam_row).cumsum()
                    ).size()
                    final_rows = group_sizes[group_sizes >= seq_min_len]
                    for start_idx, length in final_rows.iteritems():
                        sequences.append(br[i, 0, start_idx:start_idx+length-1])

        distance_matrix = np.zeros((len(sequences), len(sequences)))
        for i, seq in enumerate(sequences):
            for j_offset, seq2 in enumerate(sequences[i+1:]):
                j_abs = i + j_offset + 1
                score = dtw(seq, seq2)
                distance_matrix[i][j_abs] = score
                distance_matrix[j_abs][i] = score

        # So I have to retrieve the distance between each item and its medoid index
        # and then i find the min item, and then sum
        distortions = []
        for nclusts in range(2, 21):
            km = KMedoids(nclusts, metric='precomputed')
            km.fit(distance_matrix)
            medoids = km.medoid_indices_
            dist = sum(np.min(distance_matrix[:, medoids], axis=1)) / len(distance_matrix)
            distortions.append(dist)
        plt.plot(distortions)
        plt.xticks(np.arange(0, 19), np.arange(2, 21))
        plt.title('patient: {} target: {}'.format(patient_id, self.target))
        plt.show()


def plot_grads(pt_grad_obj):
    grads = pt_grad_obj.grad_cam.grads
    preds = pt_grad_obj.grad_cam.preds
    grad_norms = np.array([torch.FloatTensor(grads[i]).norm() for i in range(len(grads))])
    outputs = np.array([F.softmax(k).argmax() for k in preds])
    ards_preds = grad_norms[outputs == 1]
    other_preds = grad_norms[outputs == 0]
    plt.hist(ards_preds, bins=20, label='ARDS', alpha=.5)
    plt.hist(other_preds, bins=20, label='Other', alpha=.5)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='path to the saved_model')
    parser.add_argument('-pdp', '--pickled-data-path', help = 'PATH to pickled data', required=True)
    parser.add_argument('--only-patient')
    parser.add_argument('--fold', type=int, required=True, help="K-fold we want our dataset to use. This should be set in accordance to the patients that would normally be in the training dataset during a kfold run.")
    parser.add_argument('--ops', choices=['averages', 'medians', 'sample_seqs', 'read_cam', 'rand_sample', 'dtw_clust'], required=True, help="""
    *averages* - computes an average breath for patient and performs gradcam
    *medians* - performs a median breath for patient and performs gradcam
    *sample_seqs* -
    *read_cam* - perform cam for an entire read
    *rand_sample* - randomly sample sequences from stratified split of patients. output sequences and cam in figure for later analysis.
    *dtw_clust* - perform dtw clustering to see if we can gain insight into specific patterns used.
    """)
    parser.add_argument('-shuf', '--shuffle-samples', action='store_true')
    parser.add_argument('--results-base-dir', default='/slowdata/deepards/')
    parser.add_argument(
        '--target',
        choices=['ards', 'other', 'ground_truth', 'both'],
        default='ground_truth',
        help='perform grad cam wrt a certain pathophysiology. If unset it defaults to the ground truth labeling for the data. Both will examine both ards and other. However, this only works with sample_seqs'
    )
    args = parser.parse_args()

    data = pd.read_pickle(args.pickled_data_path)
    data.random_kfold = False
    data.bootstrap = False
    data.kfold_patient_splits = dict()
    data = ARDSRawDataset.make_test_dataset_if_kfold(data)
    data.set_kfold_indexes_for_fold(args.fold)
    pretrained_model = torch.load(args.model_path)
    # ensure that model is on same cuda device that data will be on
    if not isinstance(pretrained_model, torch.nn.DataParallel):
        pretrained_model = pretrained_model.to(torch.cuda.current_device())
    else:
        pretrained_model = pretrained_model.module

    pt_grad = PatientGradCam(pretrained_model, data, args.target, args.results_base_dir)
    if not args.only_patient:
        patients = np.append(pt_grad.ards, pt_grad.non_ards)
    else:
        patients = [args.only_patient]

    if args.ops == 'medians':
        for patient_id in patients:
            pt_grad.get_median_patient_camout(patient_id)
    elif args.ops == 'sample_seqs':
        for patient_id in patients:
            pt_grad.get_sampled_patient_sequences_camout(patient_id)
    elif args.ops == 'averages':
        for patient_id in patients:
            pt_grad.get_average_patient_camout(patient_id)
    elif args.ops == 'rand_sample':
        pt_grad.rand_sample(args.shuffle_samples)
    elif args.ops == 'read_cam':
        for patient_id in patients:
            pt_grad.get_full_read_patient_sequences(patient_id)
    elif args.ops == 'dtw_clust':
        for patient_id in patients:
            pt_grad.perform_dtw_clustering(patient_id)
