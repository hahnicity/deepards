from glob import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from ventmap.raw_utils import extract_raw, read_processed_file


class ARDSRawDataset(Dataset):
    def __init__(self, data_path, experiment_num, cohort_file, n_breaths_in_seq, all_sequences=[], to_pickle=None, train=True, kfold_num=None, total_kfolds=None):
        """
        Dataset to generate sequences of data for ARDS Detection
        """
        self.all_sequences = all_sequences
        self.train = train
        self.kfold_num = kfold_num
        self.total_kfolds = total_kfolds
        self.vent_bn_frac_missing = .5
        self.frames_dropped = dict()

        if len(all_sequences) > 0 and kfold_num is not None:
            self.get_kfold_indexes()
            return
        elif len(all_sequences) > 0:
            return

        cohort = pd.read_csv(cohort_file)
        if kfold_num is None:
            # XXX this will change in the future
            data_subdir = 'prototrain' if train else 'prototest'
        else:
            data_subdir = 'training'
        raw_dir = os.path.join(data_path, 'experiment{}'.format(experiment_num), data_subdir, 'raw')
        if not os.path.exists(raw_dir):
            raise Exception('No directory {} exists!'.format(raw_dir))
        raw_files = sorted(glob(os.path.join(raw_dir, '*/*.raw.npy')))
        processed_files = sorted(glob(os.path.join(raw_dir, '*/*.processed.npy')))
        # So I calculated out the stats on the training set. Our breaths are a mean of 140.5
        # obs and the std is 46.82. So mu + 2 * std = 234.19. So I can go up to 224 or 256.
        # 224 seems reasonable because it would fit well with existing img systems.
        seq_len = 224
        last_patient = None
        for fidx, filename in enumerate(raw_files):
            gen = list(read_processed_file(filename, processed_files[fidx]))
            match = re.search(r'(0\d{3}RPI\d{10})', filename)
            try:
                patient_id = match.groups()[0]
            except:
                raise ValueError('could not find patient id in file: {}'.format(filename))
            if patient_id != last_patient:
                seq_arr = []
                n_seq = 0
                seq_vent_bns = []
            last_patient = patient_id
            patient_row = cohort[cohort['Patient Unique Identifier'] == patient_id]
            patient_row = patient_row.iloc[0]
            patho = 1 if patient_row['Pathophysiology'] == 'ARDS' else 0

            # XXX need to eventually add cutoffs based on vent time or Berlin time
            for bidx, breath in enumerate(gen):
                seq_arr.append(breath['flow'])
                seq_vent_bns.append(breath['vent_bn'])
                n_seq += 1

                if n_seq == n_breaths_in_seq:
                    seq_vent_bns = np.array(seq_vent_bns)
                    diffs = seq_vent_bns[:-1] + 1 - seq_vent_bns[1:]
                    # do not include the stack if it is discontiguous to too large a degree
                    bns_missing = sum(abs(diffs))
                    missing_thresh = int(n_breaths_in_seq * self.vent_bn_frac_missing)
                    if bns_missing > missing_thresh:
                        # last vent BN possible is 65536 (2^16) I'd like to recognize if this is occurring
                        if not abs(bns_missing - (2 ** 16)) <= missing_thresh:
                            if not patient_id in self.frames_dropped:
                                self.frames_dropped[patient_id] = 1
                            else:
                                self.frames_dropped[patient_id] += 1
                            seq_arr = []
                            n_seq = 0
                            seq_vent_bns = []
                            continue
                    target = np.zeros(2)
                    target[patho] = 1
                    self.all_sequences.append([patient_id, seq_arr, target])
                    seq_arr = []
                    n_seq = 0
                    seq_vent_bns = []

        # find mean scaling factor
        #
        # XXX there is a problem with this and we are not respecting train sets versus
        # testing sets. Everything is just scaled to the same factor. I think that we
        # should change this but for now I'm going to let it ride for a bit.
        flow_sum = 0
        n_obs = 0
        for seq in self.all_sequences:
            flow_sum += sum([sum(row) for row in seq[1]])
            n_obs += sum([len(row) for row in seq[1]])
        mu = flow_sum / n_obs

        # find std scaling factor
        std_sum = 0
        for seq in self.all_sequences:
            std_sum += sum([((np.array(row) - mu) ** 2).sum()  for row in seq[1]])
        std = np.sqrt(std_sum / (n_obs-1))

        # scale and pad and update array
        for idx, seq in enumerate(self.all_sequences):
            # you can also pad before performing ops if you just pad with mu. But
            # for now the point is kinda moot
            new_seq = np.array([
                np.pad((np.array(row) - mu) / std, (0, seq_len - len(row)), 'constant')
                if len(row) < seq_len else ((np.array(row) - mu) / std)[:seq_len]
                for row in seq[1]
            ]).reshape((n_breaths_in_seq, 1, seq_len))
            self.all_sequences[idx][1] = new_seq

        # the overall dimensions correspond num breaths in sequence, num channels, num obs
        if to_pickle:
            pd.to_pickle(self.all_sequences, to_pickle)

        if kfold_num is not None:
            self.get_kfold_indexes()

    def __getitem__(self, index):
        # This is a bit tricky because pytorch will just assume that indexing
        # goes from 0 ... __len__ - 1. But this is not the case for kfold. So you
        # have to get the pytorch index and then translate that to what it looks
        # like in your kfold
        if self.kfold_num is not None:
            index = self.kfold_indexes[index]
        patient, seq, target = self.all_sequences[index]
        return index, patient, seq, target

    def collate(self, batch):
        """
        Stub method in case we ever want to use this. Normally just takes
        a list of inputs from __getitem__ and converts to tensor but you
        can add additional custom logic here too.
        """
        pass

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
