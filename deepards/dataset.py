from glob import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from ventmap.raw_utils import extract_raw, read_processed_file


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
                 total_kfolds=None):
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

        if len(all_sequences) > 0 and kfold_num is not None:
            self.get_kfold_indexes()
            return
        elif len(all_sequences) > 0:
            return

        self.cohort = pd.read_csv(cohort_file)
        if kfold_num is None:
            data_subdir = 'prototrain' if train else 'prototest'
        else:
            data_subdir = 'training'
        raw_dir = os.path.join(data_path, 'experiment{}'.format(experiment_num), data_subdir, 'raw')
        if not os.path.exists(raw_dir):
            raise Exception('No directory {} exists!'.format(raw_dir))
        self.raw_files = sorted(glob(os.path.join(raw_dir, '*/*.raw.npy')))
        self.processed_files = sorted(glob(os.path.join(raw_dir, '*/*.processed.npy')))

        # So I calculated out the stats on the training set. Our breaths are a mean of 140.5
        flow_sum = 0
        n_obs = 0
        # XXX there is a problem with this and we are not respecting train sets versus
        # testing sets. Everything is just scaled to the same factor. I think that we
        # should change this but for now I'm going to let it ride for a bit.

        # use precomputed scaling factors. These may change so we may need to compute them
        # again
        self.mu = -0.16998896167389502
        self.std = 25.332015720945343

        # 224 seems reasonable because it would fit well with existing img systems.
        self.seq_len = 224
        if dataset_type == 'breath_by_breath':
            self.get_breath_by_breath_data()
        elif dataset_type == 'unpadded_sequences':
            self.get_unpadded_sequences_dataset()

        if to_pickle:
            pd.to_pickle(self.all_sequences, to_pickle)

        if kfold_num is not None:
            self.get_kfold_indexes()

    def get_breath_by_breath_data(self):
        """
        Process data for patient where each component of a sub-batch is a breath padded
        to a desired sequence length. Breaths are batched in accordance to how many
        breaths we want clustered together
        """
        last_patient = None
        for fidx, filename in enumerate(self.raw_files):
            gen = list(read_processed_file(filename, self.processed_files[fidx]))
            patient_id = self._get_patient_id_from_file(filename)
            if patient_id != last_patient:
                batch_arr = []
                seq_vent_bns = []
            last_patient = patient_id
            patient_row = self.cohort[self.cohort['Patient Unique Identifier'] == patient_id]
            patient_row = patient_row.iloc[0]
            patho = 1 if patient_row['Pathophysiology'] == 'ARDS' else 0

            # XXX need to eventually add cutoffs based on vent time or Berlin time
            for bidx, breath in enumerate(gen):
                b_seq = np.pad((np.array(breath['flow']) - self.mu) / self.std, (0, self.seq_len - len(breath['flow'])), 'constant') if self.seq_len - len(breath['flow']) >= 0 else (np.array(breath['flow'][:self.seq_len]) - self.mu) / self.std
                batch_arr.append(b_seq)
                seq_vent_bns.append(breath['vent_bn'])

                if len(batch_arr) == self.n_sub_batches:
                    if self._should_we_drop_frame(seq_vent_bns, patient_id):
                        batch_arr = []
                        seq_vent_bns = []
                        continue
                    target = np.zeros(2)
                    target[patho] = 1
                    self.all_sequences.append([patient_id, np.array(batch_arr).reshape((self.n_sub_batches, 1, self.seq_len)), target])
                    batch_arr = []
                    seq_vent_bns = []

    def get_unpadded_sequences_dataset(self):
        # XXX should probably consolidate these two funcs
        last_patient = None
        for fidx, filename in enumerate(self.raw_files):
            gen = list(read_processed_file(filename, self.processed_files[fidx]))
            patient_id = self._get_patient_id_from_file(filename)
            if patient_id != last_patient:
                batch_arr = []
                breath_arr = []
                seq_vent_bns = []
            last_patient = patient_id
            patient_row = self.cohort[self.cohort['Patient Unique Identifier'] == patient_id]
            patient_row = patient_row.iloc[0]
            patho = 1 if patient_row['Pathophysiology'] == 'ARDS' else 0

            for bidx, breath in enumerate(gen):
                seq_vent_bns.append(breath['vent_bn'])
                if (len(breath['flow']) + len(breath_arr)) < self.seq_len:
                    breath_arr.extend(breath['flow'])
                else:
                    remaining = self.seq_len - len(breath_arr)
                    breath_arr.extend(breath['flow'][:remaining])
                    batch_arr.append((np.array(breath_arr) - self.mu) / self.std)
                    if len(breath['flow'][remaining:]) > self.seq_len:
                        breath_arr = breath['flow'][remaining:remaining+self.seq_len]
                    else:
                        breath_arr = breath['flow'][remaining:]

                if len(batch_arr) == self.n_sub_batches:
                    if self._should_we_drop_frame(seq_vent_bns, patient_id):
                        # drop breath arr to be safe
                        breath_arr = []
                        batch_arr = []
                        seq_vent_bns = []
                        continue
                    target = np.zeros(2)
                    target[patho] = 1
                    self.all_sequences.append([patient_id, np.array(batch_arr).reshape((self.n_sub_batches, 1, self.seq_len)), target])
                    batch_arr = []
                    seq_vent_bns = []

    def _get_patient_id_from_file(self, filename):
        match = re.search(r'(0\d{3}RPI\d{10})', filename)
        try:
            return match.groups()[0]
        except:
            raise ValueError('could not find patient id in file: {}'.format(filename))

    def _should_we_drop_frame(self, seq_vent_bns, patient_id):
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
        _, seq, target = self.all_sequences[index]
        return index, seq, target

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
        for patient, _, target in self.all_sequences:
            rows.append([patient, np.argmax(target, axis=0)])
        return pd.DataFrame(rows, columns=['patient', 'y'])

    def _get_kfold_ground_truth(self):
        rows = []
        for idx in self.kfold_indexes:
            patient, _, target = self.all_sequences[idx]
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
