from copy import copy
from glob import glob
import math
import os
import re

from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from ventmap import SAM
from ventmap.raw_utils import extract_raw, read_processed_file

# We will mainline all the experimental work after publication. Probably rename it as well.
from algorithms.breath_meta import get_experimental_breath_meta
from algorithms.constants import EXPERIMENTAL_META_HEADER
import deepards.correlation


class ARDSRawDataset(Dataset):
    # 224 seems reasonable because it would fit well with existing img systems.
    seq_len = 224

    def __init__(self,
                 data_path,
                 experiment_num,
                 cohort_file,
                 n_sub_batches,
                 dataset_type,
                 to_pickle=None,
                 all_sequences=[],
                 train=True,
                 kfold_num=None,
                 total_kfolds=None,
                 oversample_minority=False,
                 unpadded_downsample_factor=4.0,
                 whole_patient_super_batch=False,
                 holdout_set_type='main',
                 train_patient_fraction=1.0,
                 transforms=None,
                 final_validation_set=False,
                 drop_if_under_r2=0.0,
                 drop_i_lim=False,
                 drop_e_lim=False,
                 truncate_e_lim=None):
        """
        Dataset to generate sequences of data for ARDS Detection
        """
        self.train = train
        self.kfold_num = kfold_num
        self.all_sequences = all_sequences
        self.experiment_num = experiment_num
        self.seq_hours = dict()
        self.dataset_type = dataset_type
        self.total_kfolds = total_kfolds
        self.vent_bn_frac_missing = .5
        self.frames_dropped = dict()
        self.n_sub_batches = n_sub_batches if all_sequences == [] else all_sequences[0][1].shape[0]
        self.unpadded_downsample_factor = unpadded_downsample_factor
        self.cohort_file = cohort_file
        self.oversample = oversample_minority
        self.whole_patient_super_batch = whole_patient_super_batch
        self.train_patient_fraction = train_patient_fraction
        self.transforms = transforms
        self.drop_if_under_r2 = drop_if_under_r2
        if drop_i_lim and drop_e_lim:
            raise Exception('You cannot drop both I and E lims!')
        if truncate_e_lim and drop_e_lim:
            raise Exception('You cant truncate the E lim and drop it at the same time')
        if truncate_e_lim and round(truncate_e_lim % 0.02, 2) != 0.02:
            raise Exception('--truncate-e-lim must be given in increments divisible by 0.02!')
        self.drop_i_lim = drop_i_lim
        self.drop_e_lim = drop_e_lim
        self.truncate_e_lim = truncate_e_lim
        if self.drop_if_under_r2 and 'unpadded' not in dataset_type:
            raise Exception('Non-unpadded datasets are not supported currently with drop_if_under_r2')

        if self.drop_if_under_r2 and kfold_num is not None:
            raise Exception('kfold are not supported currently with drop_if_under_r2')
        self.auto = deepards.correlation.AutoCorrelation()

        if self.oversample and self.whole_patient_super_batch:
            raise Exception('currently oversampling with whole patient super batch is not supported')

        self.cohort = pd.read_csv(cohort_file)
        self.cohort = self.cohort.rename(columns={'Patient Unique Identifier': 'patient_id'})
        self.cohort['patient_id'] = self.cohort['patient_id'].astype(str)

        if kfold_num is not None:
            data_subdir = 'all_data'
        elif holdout_set_type == 'proto':
            data_subdir = 'prototrain' if train else 'prototest'
        elif holdout_set_type == 'main':
            data_subdir = 'training' if train else 'testing'
        elif holdout_set_type == 'random':
            if train:
                data_subdir = 'randomtrain'
            elif not train and not final_validation_set:
                data_subdir = 'randomval'
            else:
                data_subdir = 'randomtest'
        elif (holdout_set_type is not None) and (holdout_set_type not in ['main', 'proto', 'random']):
            if train:
                data_subdir = '{}train'.format(holdout_set_type)
            elif not train and not final_validation_set:
                data_subdir = '{}val'.format(holdout_set_type)
            else:
                data_subdir = '{}test'.format(holdout_set_type)
        else:
            raise Exception('You must choose to either use kfold or a holdout set!')

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
        if self.all_sequences != []:
            self.finalize_dataset_create(to_pickle, kfold_num)
            return

        raw_dir = os.path.join(data_path, 'experiment{}'.format(experiment_num), data_subdir, 'raw')
        self.meta_dir = os.path.join(data_path, 'experiment{}'.format(experiment_num), data_subdir, 'meta')
        if not os.path.exists(raw_dir):
            raise Exception('No directory {} exists!'.format(raw_dir))
        self.raw_files = sorted(glob(os.path.join(raw_dir, '*/*.raw.npy')))
        self.meta_files = sorted(glob(os.path.join(self.meta_dir, '*/*.csv')))

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
        elif dataset_type == 'padded_breath_by_breath_with_experimental_bm_target':
            self._get_breath_by_breath_with_breath_meta_target(self._pad_breath, ['iTime', 'eTime', 'inst_RR', 'mean_flow_from_pef', 'I:E ratio', 'tve:tvi ratio', 'dyn_compliance'])
        else:
            raise Exception('Unknown dataset type: {}'.format(dataset_type))
        self.finalize_dataset_create(to_pickle, kfold_num)

    def finalize_dataset_create(self, to_pickle, kfold_num):
        self.derive_scaling_factors()
        if to_pickle:
            pd.to_pickle(self, to_pickle)

        if kfold_num is not None:
            self.set_kfold_indexes_for_fold(kfold_num)

    def set_oversampling_indices(self):
        # Cannot oversample with testing set
        if not self.train:
            return

        if not self.oversample:
            return

        if self.total_kfolds:
            x = self.non_oversampled_kfold_indexes
            y = [self.all_sequences[idx][-2].argmax() for idx in x]
            ros = RandomOverSampler()
            x_resampled, y_resampled = ros.fit_resample(np.array(x).reshape(-1, 1), y)
            self.kfold_indexes = x_resampled.ravel()
        else:
            raise NotImplementedError('We havent implemented oversampling for holdout sets yet')

    def set_indices_by_patient(self):
        if self.train_patient_fraction == 1.0:
            return

        if self.total_kfolds:
            uniq_patients = set()
            for i in self.kfold_indexes:
                uniq_patients.add(self.all_sequences[i][0])
            patient_data = self.cohort[self.cohort.patient_id.isin(list(uniq_patients)) & (self.cohort.experiment_group == self.experiment_num)]
            ards_patients = patient_data[patient_data.Pathophysiology == 'ARDS'].patient_id
            other_patients = patient_data[patient_data.Pathophysiology != 'ARDS'].patient_id
            # divide by 2 because then we have equal numbers patients in each classification
            n_patients = int(math.floor(len(uniq_patients)*self.train_patient_fraction)) / 2

            to_select = set(np.random.choice(list(other_patients), size=n_patients, replace=False))
            to_select.update(set(np.random.choice(list(ards_patients), size=n_patients, replace=False)))
            tmp = []
            for i in self.kfold_indexes:
                if self.all_sequences[i][0] in to_select:
                    tmp.append(i)
            self.kfold_indexes = tmp
        else:
            raise NotImplementedError("We haven't implemented train patient fractions for holdout yet")

    def derive_scaling_factors(self):
        is_kfolds = self.total_kfolds is not None
        if is_kfolds:
            indices = [self.get_kfold_indexes_for_fold(kfold_num) for kfold_num in range(self.total_kfolds)]
        else:
            indices = [range(len(self.all_sequences))]

        if 'padded_breath_by_breath' in self.dataset_type:
            is_padded = True
        elif 'unpadded' in self.dataset_type:
            is_padded = False
        else:
            raise Exception('unsupported dataset type {} for scaling'.format(self.dataset_type))

        self.scaling_factors = {
            kfold_num if is_kfolds else None: self._get_scaling_factors_for_indices(idxs, is_padded)
            for kfold_num, idxs in enumerate(indices)
        }

    @classmethod
    def make_test_dataset_if_kfold(self, train_dataset):
        try:
            if train_dataset.drop_if_under_r2 > 0:
                raise Exception('drop if under r2 is not supported in kfold yet!')
        except AttributeError:  # in this case our dataset was created before this attr was used
            pass
        test_dataset = ARDSRawDataset(
            None,
            None,
            train_dataset.cohort_file,
            train_dataset.n_sub_batches,
            train_dataset.dataset_type,
            all_sequences=train_dataset.all_sequences,
            train=False,
            kfold_num=train_dataset.kfold_num,
            total_kfolds=train_dataset.total_kfolds,
            train_patient_fraction=1.0,
            transforms=None,
            oversample_minority=False,
            drop_if_under_r2=0.0,
        )
        return test_dataset

    @classmethod
    def from_pickle(cls, data_path, oversample_minority, train_patient_fraction, transforms):
        dataset = pd.read_pickle(data_path)
        if not isinstance(dataset, ARDSRawDataset):
            raise ValueError('The pickle file you have specified is out-of-date. Please re-process your dataset and save the new pickled dataset.')
        dataset.oversample = oversample_minority
        dataset.train_patient_fraction = train_patient_fraction
        # paranoia
        try:
            dataset.scaling_factors
        except AttributeError:
            dataset.derive_scaling_factors()
        dataset.transforms = transforms
        return dataset

    def set_kfold_indexes_for_fold(self, kfold_num):
        self.kfold_num = kfold_num
        self.kfold_indexes = self.get_kfold_indexes_for_fold(kfold_num)
        self.set_indices_by_patient()
        self.non_oversampled_kfold_indexes = copy(list(self.kfold_indexes))
        self.set_oversampling_indices()

    def get_kfold_indexes_for_fold(self, kfold_num):
        ground_truth = self._get_all_sequence_ground_truth()
        other_patients = ground_truth[ground_truth.y == 0].patient.unique()
        ards_patients = ground_truth[ground_truth.y == 1].patient.unique()
        all_patients = np.append(other_patients, ards_patients)
        patho = [0] * len(other_patients) + [1] * len(ards_patients)
        kfolds = StratifiedKFold(n_splits=self.total_kfolds)
        for split_num, (train_pt_idx, test_pt_idx) in enumerate(kfolds.split(all_patients, patho)):
            train_pts = all_patients[train_pt_idx]
            test_pts = all_patients[test_pt_idx]
            if split_num == kfold_num and self.train:
                return ground_truth[ground_truth.patient.isin(train_pts)].index
            elif split_num == kfold_num and not self.train:
                return ground_truth[ground_truth.patient.isin(test_pts)].index

    def _get_scaling_factors_for_indices(self, indices, is_padded):
        """
        Get mu and std for a specific set of indices
        """
        std_sum = 0
        mean_sum = 0
        obs_count = 0

        for idx in indices:
            obs = self.all_sequences[idx][1]
            if is_padded:  # clear off 0's used for padding
                obs = obs.ravel()
                obs = obs[obs != 0]
                obs_count += len(obs)
            else:
                obs_count += np.prod(obs.shape)
            mean_sum += obs.sum()
        mu = mean_sum / obs_count

        # calculate std
        for idx in indices:
            obs = self.all_sequences[idx][1]
            if is_padded:  # clear off 0's used for padding
                obs = obs.ravel()
                obs = obs[obs != 0]
            std_sum += ((obs - mu) ** 2).sum()
        std = np.sqrt(std_sum / obs_count)
        return mu, std

    def _get_breath_by_breath_with_flow_time_features(self, process_breath_func, bm_features):
        last_patient = None
        try:
            ratio_indices = [bm_features.index(f) for f in ['I:E ratio', 'tve:tvi ratio']]
        except ValueError:
            ratio_indices = []
        indices = [EXPERIMENTAL_META_HEADER.index(feature) for feature in bm_features]

        for fidx, filename in enumerate(self.raw_files):
            gen = read_processed_file(filename, filename.replace('.raw.npy', '.processed.npy'))
            patient_id = self._get_patient_id_from_file(filename)
            if patient_id != last_patient:
                batch_arr = []
                seq_vent_bns = []
                meta_arr = []
                batch_seq_hours = []
            last_patient = patient_id
            patient_row = self.cohort[self.cohort['patient_id'] == patient_id]
            patient_row = patient_row.iloc[0]
            patho = 1 if patient_row['Pathophysiology'] == 'ARDS' else 0
            start_time = self._get_patient_start_time(patient_id)

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
                breath_time = self.get_abs_bs_dt(breath)
                if breath_time < start_time or breath_time > start_time + pd.Timedelta(hours=24):
                    continue
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

                seq_hour = (breath_time - start_time).total_seconds() / 60 / 60
                meta = (meta - self.flow_time_bm_mu) / self.flow_time_bm_std
                flow = np.array(breath['flow'])
                b_seq = process_breath_func(flow)
                batch_arr.append(b_seq)
                seq_vent_bns.append(breath['vent_bn'])
                meta_arr.append(meta)
                batch_seq_hours.append(seq_hour)

                if len(batch_arr) == self.n_sub_batches:
                    if not self._should_we_drop_frame(None, seq_vent_bns, patient_id):
                        target = np.zeros(2)
                        target[patho] = 1
                        self.all_sequences.append([patient_id, np.array(batch_arr).reshape((self.n_sub_batches, 1, self.seq_len)), np.array(meta_arr), target, batch_seq_hours])
                    batch_arr = []
                    seq_vent_bns = []
                    meta_arr = []
                    batch_seq_hours = []

    def _get_breath_by_breath_with_breath_meta_target(self, process_breath_func, bm_features):
        try:
            ratio_indices = [bm_features.index(f) for f in ['I:E ratio', 'tve:tvi ratio']]
        except ValueError:
            ratio_indices = []
        indices = [EXPERIMENTAL_META_HEADER.index(feature) for feature in bm_features]
        for fidx, filename in enumerate(self.raw_files):
            gen = read_processed_file(filename, filename.replace('.raw.npy', '.processed.npy'))
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

                flow = np.array(breath['flow'])
                b_seq = process_breath_func(flow)
                # no scaling of the breath meta is required here because we are just doing
                # regression
                self.all_sequences.append([patient_id, b_seq.reshape((1, self.seq_len)), meta, [np.nan]])

    def _get_breath_by_breath_dataset(self, process_breath_func, target_func):
        """
        Process data for patient where each component of a sub-batch is a breath padded
        to a desired sequence length. Breaths are batched in accordance to how many
        breaths we want clustered together
        """
        last_patient = None
        super_batch_tmp_arr = []

        for fidx, filename in enumerate(self.raw_files):
            gen = read_processed_file(filename, filename.replace('.raw.npy', '.processed.npy'))
            patient_id = self._get_patient_id_from_file(filename)

            if patient_id != last_patient:
                batch_arr = []
                seq_vent_bns = []
                batch_seq_hours = []
                if self.whole_patient_super_batch and super_batch_tmp_arr != []:
                    self.all_sequences.append([last_patient, np.array(super_batch_tmp_arr), target])
                    super_batch_tmp_arr = []

            last_patient = patient_id
            target = target_func(patient_id)
            start_time = self._get_patient_start_time(patient_id)

            for bidx, breath in enumerate(gen):
                # cutoff breaths if they have too few points. It is unlikely ML
                # will ever learn anything useful from them. 21 is chosen because the mean
                # number of unpadded obs we have in our train set is 138.78 and the std
                # is 38.23. So 21 is < mu - 3*std and is divisible with 7.
                if len(breath['flow']) < 21:
                    continue

                breath_time = self.get_abs_bs_dt(breath)
                if breath_time < start_time:
                    continue
                elif breath_time > start_time + pd.Timedelta(hours=24):
                    break

                seq_hour = (breath_time - start_time).total_seconds() / 60 / 60
                flow = np.array(self.truncate_lim(breath['flow']))
                b_seq = process_breath_func(flow)
                batch_arr.append(b_seq)
                seq_vent_bns.append(breath['vent_bn'])
                batch_seq_hours.append(seq_hour)

                if len(batch_arr) == self.n_sub_batches:
                    if not self._should_we_drop_frame(None, seq_vent_bns, patient_id):
                        breath_window = np.array(batch_arr).reshape((self.n_sub_batches, 1, self.seq_len))
                        if self.whole_patient_super_batch:
                            super_batch_tmp_arr.append(breath_window)
                        else:
                            self.all_sequences.append([patient_id, breath_window, target, batch_seq_hours])
                    batch_arr = []
                    seq_vent_bns = []
                    batch_seq_hours = []

    def get_unpadded_sequences_dataset(self, processing_func, target_func):
        last_patient = None
        super_batch_tmp_arr = []
        for fidx, filename in enumerate(self.raw_files):
            gen = read_processed_file(filename, filename.replace('.raw.npy', '.processed.npy'))
            patient_id = self._get_patient_id_from_file(filename)

            if patient_id != last_patient:
                batch_arr = []
                breath_arr = []
                seq_vent_bns = []
                if self.whole_patient_super_batch and super_batch_tmp_arr != []:
                    self.all_sequences.append([last_patient, np.array(super_batch_tmp_arr), target, batch_seq_hours])
                    super_batch_tmp_arr = []
                batch_seq_hours = []

            last_patient = patient_id
            target = target_func(patient_id)
            start_time = self._get_patient_start_time(patient_id)

            for bidx, breath in enumerate(gen):
                # cutoff breaths if they have too few points. It is unlikely ML
                # will ever learn anything useful from them. 21 is chosen because the mean
                # number of unpadded obs we have in our train set is 138.78 and the std
                # is 38.23. So 21 is < mu - 3*std and is divisible with 7.
                if len(breath['flow']) < 21:
                    continue

                breath_time = self.get_abs_bs_dt(breath)
                if breath_time < start_time:
                    continue
                elif breath_time > start_time + pd.Timedelta(hours=24):
                    break

                seq_hour = (breath_time - start_time).total_seconds() / 60 / 60
                seq_vent_bns.append(breath['vent_bn'])
                flow = self.truncate_lim(breath['flow'])
                batch_arr, breath_arr, batch_seq_hours = processing_func(
                    flow, breath_arr, batch_arr, batch_seq_hours, seq_hour
                )

                if len(batch_arr) == self.n_sub_batches:
                    raw_data = np.array(batch_arr)
                    if self._should_we_drop_frame(raw_data.ravel(), seq_vent_bns, patient_id):
                        # drop breath arr to be safe
                        breath_arr = []
                        batch_arr = []
                        seq_vent_bns = []
                        batch_seq_hours = []
                        continue
                    breath_window = raw_data.reshape((self.n_sub_batches, 1, self.seq_len))
                    if self.whole_patient_super_batch:
                        super_batch_tmp_arr.append(breath_window)
                    else:
                        self.all_sequences.append([patient_id, breath_window, target, batch_seq_hours])
                    batch_arr = []
                    seq_vent_bns = []
                    batch_seq_hours = []

                if len(batch_arr) > 0 and breath_arr == []:
                    batch_seq_hours.append(seq_hour)

    def truncate_lim(self, flow):
        if self.truncate_e_lim or self.drop_i_lim or self.drop_e_lim:
            dt = 0.02
            rel_time_array = [(i+1) * dt for i in range(len(flow))]
            x0_indices_dict = SAM.find_x0s_multi_algorithms(flow, rel_time_array, dt)
            startpoint = 0
            endpoint = len(flow)

            iTime, x0_index = SAM.x0_heuristic(x0_indices_dict, rel_time_array)
            if self.truncate_e_lim is not None:
                # Keep only number of specified seconds of data
                pts_to_keep = int(math.ceil(self.truncate_e_lim / dt))
                endpoint = x0_index + pts_to_keep

            if self.drop_i_lim:
                startpoint = x0_index
            elif self.drop_e_lim:
                endpoint = x0_index

            flow = flow[startpoint:endpoint]

        return flow

    def _autoencoder_target(self, _):
        return np.array([np.nan, np.nan])

    def _pathophysiology_target(self, patient_id):
        patient_row = self.cohort[self.cohort['patient_id'] == patient_id]
        try:
            patient_row = patient_row.iloc[0]
        except:
            raise ValueError('Could not find patient {} in cohort file'.format(patient_id))
        patho = 1 if patient_row['Pathophysiology'] == 'ARDS' else 0
        target = np.zeros(2)
        target[patho] = 1
        return target

    def _get_patient_start_time(self, patient_id):
        patient_row = self.cohort[self.cohort['patient_id'] == patient_id]
        patient_row = patient_row.iloc[0]
        patho = 1 if patient_row['Pathophysiology'] == 'ARDS' else 0
        if patho == 1:
            start_time = pd.to_datetime(patient_row['Date when Berlin criteria first met (m/dd/yyy)'])
        else:
            start_time = pd.to_datetime(patient_row['vent_start_time'])

        if start_time is pd.NaT:
            raise Exception('Could not find valid start time for {}'.format(patient_id))
        return start_time

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

    def _regular_unpadded_processing(self, flow, breath_arr, batch_arr, batch_seq_hours, seq_hour):
        if (len(flow) + len(breath_arr)) < self.seq_len:
            breath_arr.extend(flow)
        else:
            remaining = self.seq_len - len(breath_arr)
            breath_arr.extend(flow[:remaining])
            batch_arr.append(np.array(breath_arr))
            batch_seq_hours.append(seq_hour)
            if len(flow[remaining:]) > self.seq_len:
                breath_arr = flow[remaining:remaining+self.seq_len]
            else:
                breath_arr = flow[remaining:]
        return batch_arr, breath_arr, batch_seq_hours

    def _downsampled_unpadded_processing(self, flow, breath_arr, batch_arr, batch_seq_hours, seq_hour):
        new_samples = int(math.ceil(len(flow) / float(self.unpadded_downsample_factor)))
        flow = list(resample(flow, new_samples))
        return self._regular_unpadded_processing(flow, breath_arr, batch_arr, batch_seq_hours, seq_hour)

    def _unpadded_centered_processing(self, flow, breath_arr, batch_arr, batch_seq_hours, seq_hour):
        if (len(flow) + len(breath_arr)) < self.seq_len:
            breath_arr.extend(flow)
        else:
            remaining = self.seq_len - len(breath_arr)
            breath_arr.extend(flow[:remaining])
            batch_arr.append(np.array(breath_arr))
            batch_seq_hours.append(seq_hour)
            breath_arr = []
        return batch_arr, breath_arr, batch_seq_hours

    def _downsampled_centered_processing(self, flow, breath_arr, batch_arr, batch_seq_hours, seq_hour):
        new_samples = int(math.ceil(len(flow) / float(self.unpadded_downsample_factor)))
        flow = list(resample(flow, new_samples))
        return self._unpadded_centered_processing(flow, breath_arr, batch_arr, batch_seq_hours, seq_hour)

    def _get_patient_id_from_file(self, filename):
        pt_id = filename.split('/')[-2]
        # sanity check to see if patient
        match = re.search(r'(0\d{3}RPI\d{10})', filename)
        if match:
            return match.groups()[0]
        try:
            # id is from anonymous dataset
            float(pt_id)
            return pt_id
        except:
            raise ValueError('could not find patient id in file: {}'.format(filename))

    def _should_we_drop_frame(self, seq, seq_vent_bns, patient_id):
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

        if seq is not None and self.drop_if_under_r2:
            r2 = self.auto.get_auto_corr_r2(seq)
            if r2 < self.drop_if_under_r2:
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
        if len(seq) == 4:
            _, data, target, seq_hours = seq
            meta = np.nan
        elif len(seq) == 5:
            _, data, meta, target, seq_hours = seq

        self.seq_hours[index] = seq_hours
        try:
            mu, std = self.scaling_factors[self.kfold_num]
        except AttributeError:
            raise AttributeError('Scaling factors not found for dataset. You must derive them using the `derive_scaling_factors` function.')

        # If we are using transforms then we can't subtract by mu because it might
        # mess up any transform we make. I don't think this will be a big deal for the data
        # because mu is so small anyhow.
        if self.transforms is not None:
            mu = 0
            data = self.transforms(data)

        if 'padded_breath_by_breath' in self.dataset_type:
            padding_mask = self._get_padding_mask(data, mu)
            data = (data - padding_mask) / std
        else:
            data = (data - mu) / std

        return index, data, meta, target

    def _get_padding_mask(self, data, mu):
        padding_mask = np.zeros(data.shape)
        np.put(padding_mask, np.where(data.ravel() != 0)[0], v=mu)
        return padding_mask

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
            if len(seq) == 4:
                patient, _, target, hrs = seq
            elif len(seq) == 5:
                patient, _, __, target, hrs = seq
            rows.append([patient, np.argmax(target, axis=0)])
        return pd.DataFrame(rows, columns=['patient', 'y'])

    def _get_kfold_ground_truth(self):
        rows = []
        for idx in self.kfold_indexes:
            seq = self.all_sequences[idx]
            if len(seq) == 4:
                patient, _, target, hrs = seq
            elif len(seq) == 5:
                patient, _, __, target, hrs = seq
            rows.append([patient, np.argmax(target, axis=0)])
        return pd.DataFrame(rows, columns=['patient', 'y'], index=self.kfold_indexes)

    def get_abs_bs_dt(self, breath):
        try:
            breath_time = pd.to_datetime(breath['abs_bs'], format='%Y-%m-%d %H-%M-%S.%f')
        except:
            breath_time = pd.to_datetime(breath['abs_bs'], format='%Y-%m-%d %H:%M:%S.%f')
        return breath_time


class SiameseNetworkDataset(ARDSRawDataset):
    def __init__(self,
                 data_path,
                 experiment_num,
                 n_sub_batches,
                 dataset_type,
                 all_sequences=[],
                 to_pickle=None,
                 train=True):

        self.total_kfolds = None
        self.all_sequences = all_sequences
        self.n_sub_batches = n_sub_batches if all_sequences == [] else all_sequences[0][1].shape[0]
        # XXX enable ability to split to main if wanted.
        data_subdir = 'prototrain' if train else 'prototest'
        raw_dir = os.path.join(data_path, 'experiment{}'.format(experiment_num), data_subdir, 'raw')
        self.dataset_type = dataset_type

        if not os.path.exists(raw_dir):
            raise Exception('No directory {} exists!'.format(raw_dir))
        self.raw_files = sorted(glob(os.path.join(raw_dir, '*/*.raw.npy')))

        if self.all_sequences == [] and dataset_type == 'padded_breath_by_breath':
            self._process_padded_breath_by_breath_sequences(self._pad_breath)
        elif self.all_sequences == [] and dataset_type == 'unpadded_sequences':
            self.get_unpadded_sequences_dataset(self._regular_unpadded_processing, self._pathophysiology_target)
        elif self.all_sequences == [] and dataset_type == 'unpadded_centered_sequences':
            self.get_unpadded_sequences_dataset(self._unpadded_centered_processing, self._pathophysiology_target)

        # remove any patients who only have 1 observation because we cannot find a
        # positive example to compare for them
        patient_ids = [patient_id for (patient_id, _) in self.all_sequences]
        pt_counts = pd.Series(patient_ids).value_counts()
        pts_to_drop = pt_counts[pt_counts == 1].index
        idxs_to_drop = [idx for idx, pt in enumerate(patient_ids) if pt in pts_to_drop]
        for offset, idx in enumerate(idxs_to_drop):
            self.all_sequences.pop(idx - offset)

        # we can use the patient mapping to both keep track of the positive
        # examples we've already used and the indexing for our reads.
        self.patient_mapping = {}
        for idx, (patient_id, _) in enumerate(self.all_sequences):
            if patient_id not in self.patient_mapping:
                # The structure kinda looks like
                # {
                #   patient: [[master list of idx], [available pos list of idx]],
                #   ...
                # }
                self.patient_mapping[patient_id] = [idx]
            else:
                self.patient_mapping[patient_id].append(idx)
        self.derive_scaling_factors()
        self.available_neg_idxs = range(len(self.all_sequences))

        if to_pickle:
            pd.to_pickle(self, to_pickle)

    def _process_padded_breath_by_breath_sequences(self, process_breath_func):
        last_patient = None

        for fidx, filename in enumerate(self.raw_files):
            gen = read_processed_file(filename, filename.replace('.raw.npy', '.processed.npy'))
            patient_id = self._get_patient_id_from_file(filename)
            if patient_id != last_patient:
                last_patient = patient_id
                batch_arr = []
                last_vent_bn = None

            for bidx, breath in enumerate(gen):
                ## determine if we should drop the cluster
                if last_vent_bn is None:
                    last_vent_bn = breath['vent_bn']
                elif breath['vent_bn'] - 50 > last_vent_bn:
                    batch_arr = []

                flow = np.array(breath['flow'])
                b_seq = process_breath_func(flow)
                batch_arr.append(b_seq)
                if len(batch_arr) == self.n_sub_batches:
                    self.all_sequences.append([
                        patient_id,
                        np.array(batch_arr).reshape((self.n_sub_batches, 1, self.seq_len))
                    ])
                    batch_arr = []
                last_vent_bn = breath['vent_bn']

    def get_unpadded_sequences_dataset(self, processing_func, target_func):
        last_patient = None

        for fidx, filename in enumerate(self.raw_files):
            gen = read_processed_file(filename, filename.replace('.raw.npy', '.processed.npy'))
            patient_id = self._get_patient_id_from_file(filename)
            if patient_id != last_patient:
                last_patient = patient_id
                batch_arr = []
                breath_arr = []
                last_vent_bn = None

            for bidx, breath in enumerate(gen):
                ## determine if we should drop the cluster
                if last_vent_bn is None:
                    last_vent_bn = breath['vent_bn']
                elif breath['vent_bn'] - 50 > last_vent_bn:
                    batch_arr = []
                    breath_arr = []

                batch_arr, breath_arr = processing_func(breath['flow'], breath_arr, batch_arr)

                if len(batch_arr) == self.n_sub_batches:
                    self.all_sequences.append([
                        patient_id,
                        np.array(batch_arr).reshape((self.n_sub_batches, 1, self.seq_len))
                    ])
                    batch_arr = []
                last_vent_bn = breath['vent_bn']

    def __getitem__(self, index):
        # Here the trick is to save memory so you want to sequentially go down the
        # list of reads but then dynamically generate positive/negative examples on
        # the fly.
        patient_id, seq = self.all_sequences[index]
        pt_avail_pos = self.patient_mapping[patient_id]
        pt_available_neg = list(set(self.available_neg_idxs).difference(set(self.patient_mapping[patient_id])))

        # get positive idx after current epoch
        rel_idx = pt_avail_pos.index(index)
        if rel_idx == len(pt_avail_pos) - 1:
            pos_idx = pt_avail_pos[rel_idx - 1]
        else:
            pos_idx = pt_avail_pos[rel_idx + 1]
        pos_compr = self.all_sequences[pos_idx][1]

        # get negative examples so that we can choose without replacement
        neg_idx = np.random.choice(pt_available_neg)
        neg_compr = self.all_sequences[neg_idx][1]
        mu, std = self.scaling_factors[None]  # this dataset is not meant to be run with kfold
        # XXX need to accomodate unpadded dataset types
        if 'padded_breath_by_breath' in self.dataset_type:
            seq_padding_mask = self._get_padding_mask(seq, mu)
            pos_compr_padding_mask = self._get_padding_mask(pos_compr, mu)
            neg_compr_padding_mask = self._get_padding_mask(neg_compr, mu)
            seq = (seq - seq_padding_mask) / std
            pos_compr = (pos_compr - pos_compr_padding_mask) / std
            neg_compr = (neg_compr - neg_compr_padding_mask) / std
        else:
            seq = (seq - mu) / std
            pos_compr = (pos_compr - mu) / std
            neg_compr = (neg_compr - mu) / std

        return seq, pos_compr, neg_compr

    def __len__(self):
        return len(self.all_sequences)

    @classmethod
    def from_pickle(self, data_path):
        dataset = pd.read_pickle(data_path)
        if not isinstance(dataset, SiameseNetworkDataset):
            raise ValueError('The pickle file you have specified is out-of-date. Please re-process your dataset and save the new pickled dataset.')
        return dataset
