from glob import glob
import math
import os
import re

import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from ventmap.raw_utils import extract_raw, read_processed_file

# We will mainline all the experimental work after publication. Probably rename it as well.
from algorithms.breath_meta import get_experimental_breath_meta
from algorithms.constants import EXPERIMENTAL_META_HEADER


class ARDSRawDataset(Dataset):

    def __init__(self,
                 data_path,
                 experiment_num,
                 cohort_file,
                 n_sub_batches,
                 dataset_type,
                 all_sequences=[],
                 to_pickle=None,
                 train=True,
                 kfold_num=None,
                 total_kfolds=None,
                 unpadded_downsample_factor=4.0,
                 drop_frame_if_frac_missing=True):
        """
        Dataset to generate sequences of data for ARDS Detection
        """
        self.all_sequences = all_sequences
        self.train = train
        self.kfold_num = kfold_num
        self.total_kfolds = total_kfolds
        self.vent_bn_frac_missing = .5
        self.frames_dropped = dict()
        self.n_sub_batches = n_sub_batches
        self.unpadded_downsample_factor = unpadded_downsample_factor
        self.drop_frame_if_frac_missing = drop_frame_if_frac_missing

        if len(all_sequences) > 0 and kfold_num is not None:
            self.get_kfold_indexes()
            return
        elif len(all_sequences) > 0:
            return

        self.cohort = pd.read_csv(cohort_file)
        if kfold_num is None:
            data_subdir = 'prototrain' if train else 'prototest'
        else:
            data_subdir = 'all_data'
        raw_dir = os.path.join(data_path, 'experiment{}'.format(experiment_num), data_subdir, 'raw')
        self.meta_dir = os.path.join(data_path, 'experiment{}'.format(experiment_num), data_subdir, 'meta')
        if not os.path.exists(raw_dir):
            raise Exception('No directory {} exists!'.format(raw_dir))
        self.raw_files = sorted(glob(os.path.join(raw_dir, '*/*.raw.npy')))
        self.processed_files = sorted(glob(os.path.join(raw_dir, '*/*.processed.npy')))
        self.meta_files = sorted(glob(os.path.join(self.meta_dir, '*/*.csv')))

        # So I calculated out the stats on the training set. Our breaths are a mean of 140.5
        flow_sum = 0
        n_obs = 0
        # XXX there is a problem with this and we are not respecting train sets versus
        # testing sets. Everything is just scaled to the same factor. I think that we
        # should change this but for now I'm going to let it ride for a bit.

        # use precomputed scaling factors. These may change so we may need to compute them again
        #
        # I recalculated this and factors changed after addition of 0723. I wonder if
        # standard scaling is the right way to do this, especially because there are
        # so many outliers.
        # new mu: -0.23932047816188615
        # new std: 25.40184587980919
        self.mu = -0.16998896167389502
        self.std = 25.332015720945343
        flow_time_features = [
            'mean_flow_from_pef',
            'inst_RR',
            'slope_minF_to_zero',
            'pef_+0.16_to_zero',
            'iTime',
            'eTime',
            'I:E ratio',
            'dyn_compliance',
            'tve:tvi ratio',
        ]
        # XXX this is kinda hard coded to 9 features even though in the future the #
        # features will probably change.
        self.flow_time_bm_mu = [
            -1.12003803e+01,  2.27065158e+01,  5.41515510e+01,  2.68864330e+01,
            8.81662707e-01,  1.98707801e+00,  5.14447986e-01,  3.08663952e-02,
            1.03526574e+00
        ]
        self.flow_time_bm_std = [
            4.96512973e+00, 6.28153415e+00, 9.68798546e+01, 2.14905835e+01,
            1.57385909e-01, 8.65758973e-01, 4.93673691e-01, 5.38365875e-02,
            5.44132642e-01
        ]
        # 224 seems reasonable because it would fit well with existing img systems.
        self.seq_len = 224
        if dataset_type == 'padded_breath_by_breath':
            self._get_breath_by_breath_dataset(self._pad_breath, self._pathophysiology_target)
        elif dataset_type == 'stretched_breath_by_breath':
            self._get_breath_by_breath_dataset(self._stretch_breath, self._pathophysiology_target)
        elif dataset_type == 'spaced_padded_breath_by_breath':
            self._get_breath_by_breath_dataset(self._perform_spaced_padding, self._pathophysiology_target)
        elif dataset_type == 'unpadded_sequences':
            self.get_unpadded_sequences_dataset(self._regular_unpadded_processing, self._pathophysiology_target)
        elif dataset_type == 'unpadded_centered_sequences':
            self.get_unpadded_sequences_dataset(self._unpadded_centered_processing, self._pathophysiology_target)
        elif dataset_type == 'unpadded_centered_downsampled_sequences':
            self.get_unpadded_sequences_dataset(self._downsampled_centered_processing, self._pathophysiology_target)
        elif dataset_type == 'unpadded_downsampled_sequences':
            self.get_unpadded_sequences_dataset(self._downsampled_unpadded_processing, self._pathophysiology_target)
        elif dataset_type == 'unpadded_downsampled_autoencoder_sequences':
            self.get_unpadded_sequences_dataset(self._downsampled_unpadded_processing, self._autoencoder_target)
        elif dataset_type == 'padded_breath_by_breath_with_full_bm_target':
            self._get_breath_by_breath_with_breath_meta_target(self._pad_breath, flow_time_features)
        elif dataset_type == 'padded_breath_by_breath_with_limited_bm_target':
            self._get_breath_by_breath_with_breath_meta_target(self._pad_breath, ['iTime', 'eTime', 'inst_RR'])
        elif dataset_type == 'padded_breath_by_breath_with_flow_time_features':
            self._get_breath_by_breath_with_flow_time_features(self._pad_breath, flow_time_features)
        else:
            raise Exception('Unknown dataset type: {}'.format(dataset_type))

        if to_pickle:
            pd.to_pickle(self.all_sequences, to_pickle)

        if kfold_num is not None:
            self.get_kfold_indexes()

    def _get_breath_by_breath_with_flow_time_features(self, process_breath_func, bm_features):
        last_patient = None
        try:
            ratio_indices = [bm_features.index(f) for f in ['I:E ratio', 'tve:tvi ratio']]
        except ValueError:
            ratio_indices = []
        indices = [EXPERIMENTAL_META_HEADER.index(feature) for feature in bm_features]

        for fidx, filename in enumerate(self.raw_files):
            gen = read_processed_file(filename, self.processed_files[fidx])
            patient_id = self._get_patient_id_from_file(filename)
            if patient_id != last_patient:
                batch_arr = []
                seq_vent_bns = []
                meta_arr = []
            last_patient = patient_id
            patient_row = self.cohort[self.cohort['Patient Unique Identifier'] == patient_id]
            patient_row = patient_row.iloc[0]
            patho = 1 if patient_row['Pathophysiology'] == 'ARDS' else 0

            matching_meta = os.path.join(self.meta_dir, patient_id, 'breath_meta_' + os.path.basename(filename).replace('.raw.npy', '.csv'))
            has_preprocessed_meta = False
            if matching_meta in self.meta_files:
                try:
                    processed_meta = pd.read_csv(matching_meta, header=None).values
                    has_preprocessed_meta = True
                except pd.errors.EmptyDataError:
                    pass

            for bidx, breath in enumerate(gen):
                # cutoff breaths if they have too few points. It is unlikely ML
                # will ever learn anything useful from them. 21 is chosen because the mean
                # number of unpadded obs we have in our train set is 138.78 and the std
                # is 38.23. So 21 is < mu - 3*std and is divisible with 7.
                if len(breath['flow']) < 21:
                    continue
                if has_preprocessed_meta:
                    try:
                        meta = processed_meta[bidx]
                    except IndexError:
                        meta = np.array(get_experimental_breath_meta(breath))
                    # sanity check
                    if int(meta[0]) != breath['rel_bn']:
                        meta = np.array(get_experimental_breath_meta(breath))
                else:
                    meta = np.array(get_experimental_breath_meta(breath))

                meta = meta[indices].astype(np.float)
                if np.any(np.isinf(meta) | np.isnan(meta)):
                    continue
                # clip any breaths with ratios > 100. These breaths have proven their
                # ability to totally blow up gradients. 100 was chosen because it is > 3*std
                # for the tve:tvi ratios we have
                if np.any(np.abs(meta[ratio_indices]) > 100):
                    continue

                meta = (meta - self.flow_time_bm_mu) / self.flow_time_bm_std
                flow = (np.array(breath['flow']) - self.mu) / self.std
                b_seq = process_breath_func(flow)
                batch_arr.append(b_seq)
                seq_vent_bns.append(breath['vent_bn'])
                meta_arr.append(meta)

                if len(batch_arr) == self.n_sub_batches:
                    if self._should_we_drop_frame(seq_vent_bns, patient_id):
                        batch_arr = []
                        seq_vent_bns = []
                        meta_arr = []
                        continue
                    target = np.zeros(2)
                    target[patho] = 1
                    self.all_sequences.append([patient_id, np.array(batch_arr).reshape((self.n_sub_batches, 1, self.seq_len)), np.array(meta_arr), target])
                    batch_arr = []
                    seq_vent_bns = []
                    meta_arr = []

    def _get_breath_by_breath_with_breath_meta_target(self, process_breath_func, bm_features):
        try:
            ratio_indices = [bm_features.index(f) for f in ['I:E ratio', 'tve:tvi ratio']]
        except ValueError:
            ratio_indices = []
        indices = [EXPERIMENTAL_META_HEADER.index(feature) for feature in bm_features]
        for fidx, filename in enumerate(self.raw_files):
            gen = read_processed_file(filename, self.processed_files[fidx])
            patient_id = self._get_patient_id_from_file(filename)

            matching_meta = os.path.join(self.meta_dir, patient_id, 'breath_meta_' + os.path.basename(filename).replace('.raw.npy', '.csv'))
            has_preprocessed_meta = False
            if matching_meta in self.meta_files:
                try:
                    processed_meta = pd.read_csv(matching_meta, header=None).values
                    has_preprocessed_meta = True
                except pd.errors.EmptyDataError:
                    pass

            for bidx, breath in enumerate(gen):
                # cutoff breaths if they have too few points. It is unlikely ML
                # will ever learn anything useful from them. 21 is chosen because the mean
                # number of unpadded obs we have in our train set is 138.78 and the std
                # is 38.23. So 21 is < mu - 3*std and is divisible with 7.
                if len(breath['flow']) < 21:
                    continue
                # XXX I should abstract this whole section where we extract a row of metadata
                if has_preprocessed_meta:
                    try:
                        meta = processed_meta[bidx]
                    except IndexError:
                        meta = np.array(get_experimental_breath_meta(breath))
                    # sanity check
                    if int(meta[0]) != breath['rel_bn']:
                        meta = np.array(get_experimental_breath_meta(breath))
                else:
                    meta = np.array(get_experimental_breath_meta(breath))

                meta = meta[indices].astype(np.float)
                if np.any(np.isinf(meta) | np.isnan(meta)):
                    continue
                # clip any breaths with ratios > 100. These breaths have proven their
                # ability to totally blow up gradients. 100 was chosen because it is > 3*std
                # for the tve:tvi ratios we have
                if np.any(np.abs(meta[ratio_indices]) > 100):
                    continue

                flow = (np.array(breath['flow']) - self.mu) / self.std
                b_seq = process_breath_func(flow)
                # no scaling of the breath meta is required here because we are just doing
                # regression
                self.all_sequences.append([patient_id, b_seq.reshape((1, self.seq_len)), meta])

    def _get_breath_by_breath_dataset(self, process_breath_func, target_func):
        """
        Process data for patient where each component of a sub-batch is a breath padded
        to a desired sequence length. Breaths are batched in accordance to how many
        breaths we want clustered together
        """
        last_patient = None
        for fidx, filename in enumerate(self.raw_files):
            gen = read_processed_file(filename, self.processed_files[fidx])
            patient_id = self._get_patient_id_from_file(filename)
            if patient_id != last_patient:
                batch_arr = []
                seq_vent_bns = []
            last_patient = patient_id
            target = target_func(patient_id)

            # XXX need to eventually add cutoffs based on vent time or Berlin time
            for bidx, breath in enumerate(gen):
                # cutoff breaths if they have too few points. It is unlikely ML
                # will ever learn anything useful from them. 21 is chosen because the mean
                # number of unpadded obs we have in our train set is 138.78 and the std
                # is 38.23. So 21 is < mu - 3*std and is divisible with 7.
                if len(breath['flow']) < 21:
                    continue
                flow = (np.array(breath['flow']) - self.mu) / self.std
                b_seq = process_breath_func(flow)
                batch_arr.append(b_seq)
                seq_vent_bns.append(breath['vent_bn'])

                if len(batch_arr) == self.n_sub_batches:
                    if self._should_we_drop_frame(seq_vent_bns, patient_id):
                        batch_arr = []
                        seq_vent_bns = []
                        continue
                    self.all_sequences.append([patient_id, np.array(batch_arr).reshape((self.n_sub_batches, 1, self.seq_len)), target])
                    batch_arr = []
                    seq_vent_bns = []

    def _autoencoder_target(self, _):
        return np.array([np.nan, np.nan])

    def _pathophysiology_target(self, patient_id):
        patient_row = self.cohort[self.cohort['Patient Unique Identifier'] == patient_id]
        patient_row = patient_row.iloc[0]
        patho = 1 if patient_row['Pathophysiology'] == 'ARDS' else 0
        target = np.zeros(2)
        target[patho] = 1
        return target

    def _pad_breath(self, flow):
        if self.seq_len - len(flow) >= 0:
            return np.pad(flow, (0, self.seq_len - len(flow)), 'constant')
        else:
            return flow[:self.seq_len]

    def _stretch_breath(self, flow):
        if len(flow) < self.seq_len:
            return resample(flow, self.seq_len)
        else:
            return flow[:self.seq_len]

    def _perform_spaced_padding(self, flow):
        if len(flow) < self.seq_len:
            spacing = len(flow) / float(self.seq_len)
            new_arr = np.zeros(self.seq_len)
            i = 0
            for j in range(self.seq_len):
                if j * spacing >= i:
                    new_arr[j] = flow[i]
                    i += 1
                elif j * spacing > len(flow) - 1:
                    break
            return new_arr
        else:
            return flow[:self.seq_len]

    def get_unpadded_sequences_dataset(self, processing_func, target_func):
        last_patient = None
        for fidx, filename in enumerate(self.raw_files):
            gen = read_processed_file(filename, self.processed_files[fidx])
            patient_id = self._get_patient_id_from_file(filename)
            if patient_id != last_patient:
                batch_arr = []
                breath_arr = []
                seq_vent_bns = []
            last_patient = patient_id
            target = target_func(patient_id)

            # XXX need to eventually add cutoffs based on vent time or Berlin time
            for bidx, breath in enumerate(gen):
                # cutoff breaths if they have too few points. It is unlikely ML
                # will ever learn anything useful from them. 21 is chosen because the mean
                # number of unpadded obs we have in our train set is 138.78 and the std
                # is 38.23. So 21 is < mu - 3*std and is divisible with 7.
                if len(breath['flow']) < 21:
                    continue
                seq_vent_bns.append(breath['vent_bn'])
                batch_arr, breath_arr = processing_func(breath['flow'], breath_arr, batch_arr)

                if len(batch_arr) == self.n_sub_batches:
                    if self._should_we_drop_frame(seq_vent_bns, patient_id):
                        # drop breath arr to be safe
                        breath_arr = []
                        batch_arr = []
                        seq_vent_bns = []
                        continue
                    self.all_sequences.append([patient_id, np.array(batch_arr).reshape((self.n_sub_batches, 1, self.seq_len)), target])
                    batch_arr = []
                    seq_vent_bns = []

    def _regular_unpadded_processing(self, flow, breath_arr, batch_arr):
        if (len(flow) + len(breath_arr)) < self.seq_len:
            breath_arr.extend(flow)
        else:
            remaining = self.seq_len - len(breath_arr)
            breath_arr.extend(flow[:remaining])
            batch_arr.append((np.array(breath_arr) - self.mu) / self.std)
            if len(flow[remaining:]) > self.seq_len:
                breath_arr = flow[remaining:remaining+self.seq_len]
            else:
                breath_arr = flow[remaining:]
        return batch_arr, breath_arr

    def _downsampled_unpadded_processing(self, flow, breath_arr, batch_arr):
        new_samples = int(math.ceil(len(flow) / float(self.unpadded_downsample_factor)))
        flow = list(resample(flow, new_samples))
        return self._regular_unpadded_processing(flow, breath_arr, batch_arr)

    def _unpadded_centered_processing(self, flow, breath_arr, batch_arr):
        if (len(flow) + len(breath_arr)) < self.seq_len:
            breath_arr.extend(flow)
        else:
            remaining = self.seq_len - len(breath_arr)
            breath_arr.extend(flow[:remaining])
            batch_arr.append((np.array(breath_arr) - self.mu) / self.std)
            breath_arr = []
        return batch_arr, breath_arr

    def _downsampled_centered_processing(self, flow, breath_arr, batch_arr):
        new_samples = int(math.ceil(len(flow) / float(self.unpadded_downsample_factor)))
        flow = list(resample(flow, new_samples))
        return self._unpadded_centered_processing(flow, breath_arr, batch_arr)

    def _get_patient_id_from_file(self, filename):
        match = re.search(r'(0\d{3}RPI\d{10})', filename)
        try:
            return match.groups()[0]
        except:
            raise ValueError('could not find patient id in file: {}'.format(filename))

    def _should_we_drop_frame(self, seq_vent_bns, patient_id):
        if not self.drop_frame_if_frac_missing:
            return False

        seq_vent_bns = np.array(seq_vent_bns)
        diffs = seq_vent_bns[:-1] + 1 - seq_vent_bns[1:]
        # do not include the stack if it is discontiguous to too large a degree
        bns_missing = sum(abs(diffs))
        missing_thresh = int(self.n_sub_batches * self.vent_bn_frac_missing)
        if bns_missing > missing_thresh:
            # last vent BN possible is 65536 (2^16) I'd like to recognize if this is occurring
            if not abs(bns_missing - (2 ** 16)) <= missing_thresh:
                if not patient_id in self.frames_dropped:
                    self.frames_dropped[patient_id] = 1
                else:
                    self.frames_dropped[patient_id] += 1
                return True
        return False

    def __getitem__(self, index):
        # This is a bit tricky because pytorch will just assume that indexing
        # goes from 0 ... __len__ - 1. But this is not the case for kfold. So you
        # have to get the pytorch index and then translate that to what it looks
        # like in your kfold
        if self.kfold_num is not None:
            index = self.kfold_indexes[index]
        seq = self.all_sequences[index]
        if len(seq) == 3:
            _, data, target = seq
            meta = np.nan
        elif len(seq) == 4:
            _, data, meta, target = seq
        return index, data, meta, target

    def __len__(self):
        if self.kfold_num is None:
            return len(self.all_sequences)
        else:
            return len(self.kfold_indexes)

    def get_ground_truth_df(self):
        if self.kfold_num is None:
            return self._get_all_sequence_ground_truth()
        else:
            return self._get_kfold_ground_truth()

    def _get_all_sequence_ground_truth(self):
        rows = []
        for seq in self.all_sequences:
            if len(seq) == 3:
                patient, _, target = seq
            elif len(seq) == 4:
                patient, _, __, target = seq
            rows.append([patient, np.argmax(target, axis=0)])
        return pd.DataFrame(rows, columns=['patient', 'y'])

    def _get_kfold_ground_truth(self):
        rows = []
        for idx in self.kfold_indexes:
            seq = self.all_sequences[idx]
            if len(seq) == 3:
                patient, _, target = seq
            elif len(seq) == 4:
                patient, _, __, target = seq
            rows.append([patient, np.argmax(target, axis=0)])
        return pd.DataFrame(rows, columns=['patient', 'y'], index=self.kfold_indexes)

    def get_kfold_indexes_for_fold(self, fold_num):
        self.kfold_num = fold_num
        self.get_kfold_indexes()

    def get_kfold_indexes(self):
        ground_truth = self._get_all_sequence_ground_truth()
        other_patients = ground_truth[ground_truth.y == 0].patient.unique()
        ards_patients = ground_truth[ground_truth.y == 1].patient.unique()
        all_patients = np.append(other_patients, ards_patients)
        patho = [0] * len(other_patients) + [1] * len(ards_patients)
        kfolds = StratifiedKFold(n_splits=self.total_kfolds)
        for split_num, (train_pt_idx, test_pt_idx) in enumerate(kfolds.split(all_patients, patho)):
            train_pts = all_patients[train_pt_idx]
            test_pts = all_patients[test_pt_idx]
            if split_num == self.kfold_num and self.train:
                self.kfold_indexes = ground_truth[ground_truth.patient.isin(train_pts)].index
                break
            elif split_num == self.kfold_num and not self.train:
                self.kfold_indexes = ground_truth[ground_truth.patient.isin(test_pts)].index
                break
