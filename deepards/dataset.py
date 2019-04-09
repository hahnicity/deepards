from glob import glob
import os
import re

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from ventmap.raw_utils import extract_raw, read_processed_file


class ARDSRawDataset(Dataset):
    def __init__(self, data_path, experiment_num, cohort_file, to_pickle=None, from_pickle=None, train=True):
        self.all_sequences = []
        if from_pickle:
            self.all_sequences = pd.read_pickle(from_pickle)
            return

        cohort = pd.read_csv(cohort_file)
        # XXX this will change in the future
        data_subdir = 'prototrain' if train else 'prototest'
        raw_dir = os.path.join(data_path, 'experiment{}'.format(experiment_num), data_subdir, 'raw')
        if not os.path.exists(raw_dir):
            raise Exception('No directory {} exists!'.format(raw_dir))
        raw_files = sorted(glob(os.path.join(raw_dir, '*/*.raw.npy')))
        processed_files = sorted(glob(os.path.join(raw_dir, '*/*.processed.npy')))
        # So I calculated out the stats on the training set. Our breaths are a mean of 140.5
        # obs and the std is 46.82. So mu + 2 * std = 234.19. So I can go up to 224 or 256.
        # 224 seems reasonable because it would fit well with existing img systems.
        seq_len = 224
        n_breaths_in_seq = 20

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

                if seq_arr.shape[0] == n_breaths_in_seq:
                    self.all_sequences.append((patient_id, seq_arr, target))

        if to_pickle:
            pd.to_pickle(self.all_sequences, to_pickle)

    def __getitem__(self, index):
        return self.all_sequences[index]

    def collate(self, batch):
        """
        Stub method in case we ever want to use this. Normally just takes
        a list of inputs from __getitem__ and converts to tensor but you
        can add additional custom logic here too.
        """
        pass

    def __len__(self):
        return len(self.all_sequences)
