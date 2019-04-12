from glob import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from ventmap.raw_utils import extract_raw, read_processed_file


class ARDSRawDataset(Dataset):
    def __init__(self, data_path, experiment_num, cohort_file, sequence_size, to_pickle=None, from_pickle=None, train=True, kfold_num=None, total_kfolds=None):
        """
        Dataset to generate sequences of data for ARDS Detection
        """
        self.all_sequences = []
        self.train = train
        self.kfold_num = kfold_num
        self.total_kfolds = total_kfolds

        if from_pickle and kfold_num is not None:
            self.all_sequences = pd.read_pickle(from_pickle)
            self.get_kfold_indexes()
            return
        elif from_pickle:
            self.all_sequences = pd.read_pickle(from_pickle)
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
                seq_arr = None
                n_seq = 0
            last_patient = patient_id
            patient_row = cohort[cohort['Patient Unique Identifier'] == patient_id]
            patient_row = patient_row.iloc[0]
            patho = 1 if patient_row['Pathophysiology'] == 'ARDS' else 0

            for bidx, breath in enumerate(gen):
                flow = breath['flow']
                if len(flow) < seq_len:
                    flow.extend([0] * (seq_len - len(flow)))
                elif len(flow) > seq_len:
                    flow = flow[:seq_len]
                # the dimensions correspond num breaths in sequence, num channels, num obs
                flow = np.array(flow).reshape((1, 1, seq_len))
                # XXX in future ensure vent bns relatively contiguous
                if seq_arr is None:
                    seq_arr = flow
                else:
                    seq_arr = np.append(seq_arr, flow, axis=0)

                n_seq += 1
                if n_seq == sequence_size:
                    target = np.zeros(2)
                    target[patho] = 1
                    self.all_sequences.append((patient_id, seq_arr, target))
                    seq_arr = None
                    n_seq = 0

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
        rows = []

        if self.kfold_num is None:
            for patient, _, target in self.all_sequences:
                rows.append([patient, np.argmax(target, axis=0)])
        else:
            for idx in self.kfold_indexes:
                patient, _, target = self.all_sequences[idx]
                rows.append([patient, np.argmax(target, axis=0)])

        return pd.DataFrame(rows, columns=['patient', 'y'])

    def get_kfold_indexes(self):
        ground_truth = self.get_ground_truth_df()
        patients = ground_truth.patient
        patho = ground_truth.y
        kfolds = StratifiedKFold(n_splits=self.total_kfolds)
        for split_num, (train_idx, test_idx) in enumerate(kfolds.split(patients, patho)):
            if split_num == self.kfold_num and self.train:
                self.kfold_indexes = train_idx
                break
            elif split_num == self.kfold_num and not self.train:
                self.kfold_indexes = test_idx
                break
