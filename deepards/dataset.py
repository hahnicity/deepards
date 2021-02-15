from copy import copy
from glob import glob
import math
import os
from pathlib import Path
import re

from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt, resample
from sklearn.model_selection import StratifiedKFold
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ventmap import SAM
from ventmap.raw_utils import extract_raw, read_processed_file

# We will mainline all the experimental work after publication. Probably rename it as well.
from algorithms.breath_meta import get_experimental_breath_meta
from algorithms.constants import EXPERIMENTAL_META_HEADER
import deepards
try:
    from deepards.correlation import AutoCorrelation
except ImportError:  # if we dont have this installed we probably dont need it
    from mock import Mock
    AutoCorrelation = Mock()


class GenericHomogeneityUndersampler(object):
    def __init__(self, undersample_factor):
        saved_results_path = Path(__file__).parent.joinpath('dtw_cache/patient_score_map.pkl')
        self.score_map = pd.read_pickle(str(saved_results_path))
        if not 0 <= undersample_factor < 1:
            raise Exception('Must set an undersampling factor in [0, 1)')
        self.undersample_factor = undersample_factor

    def fit(self, x, gt):
        """
        :param x:
        :param gt: pandas dataframe of ground truth data
        """
        if len(x) !=  len(gt):
            raise Exception('You must pass in a ground truth that matches len of x')
        if not len(set(['patient', 'y']).intersection(gt.columns)) == 2:
            raise Exception('Must pass a ground truth df with patient and y columns')

        all_scores = []
        gt['dtw'] = np.nan
        # match dtw_scores with actual data
        for pt in gt.patient.unique():
            all_scores.extend(self.score_map[pt])
            scores = [0] + self.score_map[pt]
            gt.loc[gt.patient == pt, 'dtw'] = scores

        global_median = np.nanmedian(all_scores)
        global_std = np.std(all_scores)
        gt['usamp_factor'] = 1
        gt.loc[(gt.dtw <= global_median+global_std) &
               (gt.dtw >= global_median-global_std), 'usamp_factor'] = self.undersample_factor

    def fit_resample(self, x, gt):
        self.fit(x, gt)
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        usamps = gt.loc[gt.usamp_factor != 1]
        gt['sample'] = 1
        mask = np.random.rand(len(usamps)) >= self.undersample_factor
        gt.loc[usamps.loc[mask].index, 'sample'] = 0
        return x[gt['sample'].astype(bool)], gt[gt['sample'] == 1]


class PatientLevelHomogeneityUndersampler(GenericHomogeneityUndersampler):
    def __init__(self, undersample_factor, std_factor):
        super(PatientLevelHomogeneityUndersampler, self).__init__(undersample_factor)
        self.std_factor = std_factor

    def fit(self, x, gt):
        """
        :param x:
        :param gt: pandas dataframe of ground truth data
        """
        if len(x) !=  len(gt):
            raise Exception('You must pass in a ground truth that matches len of x')
        if not len(set(['patient', 'y']).intersection(gt.columns)) == 2:
            raise Exception('Must pass a ground truth df with patient and y columns')

        all_scores = []
        gt['dtw'] = np.nan
        # match dtw_scores with actual data
        for pt in gt.patient.unique():
            all_scores.extend(self.score_map[pt])
            scores = [0] + self.score_map[pt]
            mask = gt.patient == pt
            gt.loc[mask, 'dtw'] = scores
            gt.loc[mask, 'pt_dtw_median'] = np.median(scores)
            gt.loc[mask, 'pt_dtw_std'] = np.std(scores)

        global_std = np.std(all_scores)
        gt['usamp_factor'] = 1
        gt.loc[(gt.dtw <= gt.pt_dtw_median+(gt.pt_dtw_std*self.std_factor)) &
               (gt.dtw >= gt.pt_dtw_median-(gt.pt_dtw_std*self.std_factor)), 'usamp_factor'] = self.undersample_factor


def magnitude_warp(x, sigma=0.2, knot=4):
    """
    :param x: (batch, time steps, chans)
    """
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        x[i] = pat * warper

    return x


def time_warp(x, sigma=0.2, knot=4):
    """
    :param x: (batch, time steps, chans)
    """
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T

    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            x[i,:,dim] = torch.from_numpy(np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T)
    return x


def window_slice(x, reduce_ratio=0.9):
    """
    :param x: (batch, time steps, chans)
    """
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)

    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            x[i,:,dim] = torch.from_numpy(np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T)
    return x


def window_warp(x, window_ratio=0.1, scales=[0.5, 2.], by_row=False):
    """
    :param x: (batch, time steps, chans)
    """
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    #warp_scales = np.random.uniform(scales[0], scales[1], size=x.shape[2])
    warp_dim = 2 if by_row else 0
    warp_scales = np.random.choice(scales, size=x.shape[warp_dim])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)

    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)

    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            warp_dim = dim if by_row else i
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[warp_dim])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))
            x[i,:,dim] = torch.from_numpy(np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T)
    return x


class RandomWindowSlicing(torch.nn.Module):
    def __init__(self, p=.5, reduce_ratio=0.9):
        super().__init__()
        self.p = p
        self.reduce_ratio = reduce_ratio

    def forward(self, x):
        if self.p < np.random.random():
            return x
        ret = window_slice(torch.transpose(x, 1, 2), self.reduce_ratio)
        return torch.transpose(ret, 1, 2)


class RandomWindowWarping(torch.nn.Module):
    def __init__(self, p=.5, window_ratio=.25, scales=[.5, 2], by_row=False):
        super().__init__()
        self.p = p
        self.window_ratio = window_ratio
        self.scales = scales
        self.by_row = by_row

    def forward(self, x):
        if self.p < np.random.random():
            return x
        ret = window_warp(torch.transpose(x, 1, 2), self.window_ratio, self.scales, self.by_row)
        return torch.transpose(ret, 1, 2)


class RandomTimeWarp(torch.nn.Module):
    def __init__(self, p=.5, sigma=.2, knot=4):
        super().__init__()
        self.p = p
        self.sigma = sigma
        self.knot = knot

    def forward(self, x):
        if self.p < np.random.random():
            return x
        ret = time_warp(torch.transpose(x, 1, 2), self.sigma, self.knot)
        return torch.transpose(ret, 1, 2)


class RandomMagnitudeWarp(torch.nn.Module):
    def __init__(self, p=.5, sigma=.2, knot=4):
        super().__init__()
        self.p = p
        self.sigma = sigma
        self.knot = knot

    def forward(self, x):
        if self.p < np.random.random():
            return x
        ret = magnitude_warp(torch.transpose(x, 1, 2), self.sigma, self.knot)
        return torch.transpose(ret, 1, 2)


# Apparently I forgot there was an augmentation.py module. But I dunno how much
# I care about re-shuffling code right now
class RowShuffle(torch.nn.Module):
    def __init__(self, p=0.5):
        """
        Randomly shuffle all rows in a tensor
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p < np.random.random():
            return x
        chans, seq_size, _ = x.shape
        idxs = list(range(seq_size))
        np.random.shuffle(idxs)
        return x[:, idxs]


class RandomRowHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5, frac_rows=.25):
        """
        Randomly perform a horizontal flip of a fraction of rows in
        a tensor
        """
        super().__init__()
        self.p = p
        self.frac_rows = frac_rows

    def forward(self, x):
        if self.p < np.random.random():
            return x
        chans, seq_size, _ = x.shape
        idxs = list(range(seq_size))
        np.random.shuffle(idxs)
        idxs = idxs[:int(seq_size*self.frac_rows)]
        x[:, idxs] = torch.flip(x[:, idxs], dims=(-1,))
        return x


class RandomRowScale(torch.nn.Module):
    def __init__(self, p=0.5, frac_rows=.25, mag=(.8, 1.2)):
        super().__init__()
        self.p = p
        self.frac_rows = frac_rows
        self.mag = mag

    def forward(self, x):
        if self.p < np.random.random():
            return x
        chans, seq_size, _ = x.shape
        idxs = list(range(seq_size))
        np.random.shuffle(idxs)
        n_rows = int(seq_size*self.frac_rows)
        idxs = idxs[:n_rows]
        warp = np.random.uniform(low=self.mag[0], high=self.mag[1], size=n_rows)
        warp = np.expand_dims(warp, axis=1)
        x[:, idxs] = (x[:, idxs] * warp)
        return x


class PatchWindowWarp(torch.nn.Module):
    def __init__(self, p=0.5, frac_rows=0.5, mag=(0.5, 2), n_obs=(10, 56)):
        super().__init__()
        self.p = p
        self.frac_rows = frac_rows
        self.mag = mag
        self.n_obs = n_obs

    def forward(self, x):
        if self.p < np.random.random():
            return x
        chans, seq_size, _ = x.shape
        # pick a starting row randomly
        n_rows = int(seq_size*self.frac_rows)
        start_row = np.random.randint(0, seq_size*frac_rows)
        rows = x[:, start_row:start_row+n_rows]
        warp_ratio = np.random.uniform(low=self.mag[0], high=self.mag[1])
        obs = np.random.randint(self.n_obs[0], self.n_obs[1])
        start_idx = np.random.randint(0, seq_size-obs)

        # goal is to figure out how many new obs we are adding and then
        # figure out start end indices for both old and new sections
        #new_start =
        #new_end =

        # so now we pick a start obs and end obs and then warp by ratio
        rows[:, :, start_idx:start_idx+obs]
        resampd = resample(rows, warp_ratio*obs)



two_dim_transforms = {
    'row_shuffle': RowShuffle,
    'rand_erase': transforms.RandomErasing,
    'row_horiz_flip': RandomRowHorizontalFlip,
    'horiz_flip': transforms.RandomHorizontalFlip,
    'vert_flip': transforms.RandomVerticalFlip,
    'scale': RandomRowScale,
    'mag_warp': RandomMagnitudeWarp,
    'win_warp': RandomWindowWarping,
    'win_slice': RandomWindowSlicing,
    'time_warp': RandomTimeWarp,
}


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
                 truncate_e_lim=None,
                 undersample_factor=-1,
                 undersample_std_factor=0.2,
                 oversample_all_factor=1.0,
                 butter_filter=None,
                 add_fft=False,
                 only_fft=False,
                 fft_real_only=False,):
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
        self.oversample_minority = oversample_minority
        self.oversample_all_factor = oversample_all_factor
        self.undersample_factor = undersample_factor
        self.undersample_std_factor = undersample_std_factor
        self.whole_patient_super_batch = whole_patient_super_batch
        self.train_patient_fraction = train_patient_fraction
        self.transforms = transforms
        self.drop_if_under_r2 = drop_if_under_r2
        self.only_fft = only_fft
        self.add_fft = add_fft
        self.fft_real_only = fft_real_only
        if butter_filter is not None:
            b, a = butter(1, butter_filter)
            self.butter_filter = lambda x: filtfilt(b, a, x, axis=-1)
        else:
            self.butter_filter = None

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
        self.auto = AutoCorrelation()

        if self.oversample_minority and self.whole_patient_super_batch:
            raise Exception('currently oversampling with whole patient super batch is not supported')

        self.cohort = pd.read_csv(cohort_file)
        self.cohort = self.cohort.rename(columns={'Patient Unique Identifier': 'patient_id'})
        self.cohort['patient_id'] = self.cohort['patient_id'].astype(str)

        if kfold_num is not None:
            data_subdir = 'all_data'
        elif holdout_set_type == 'proto':
            data_subdir = 'prototrain' if train else 'prototest'
        elif holdout_set_type == 'main':
            data_subdir = 'aim1_70_30_training' if train else 'aim1_70_30_testing'
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
        elif dataset_type == 'unpadded_centered_with_bm':
            self.get_unpadded_sequences_dataset_with_bm_data(self._unpadded_centered_processing, self._pathophysiology_target)
        else:
            raise Exception('Unknown dataset type: {}'.format(dataset_type))
        self._perform_fft()
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

        if self.oversample_minority and not self.total_kfolds:
            raise NotImplementedError('We havent implemented oversampling for holdout sets yet')
        elif self.oversample_minority:
            x = copy(self.kfold_indexes)
            y = [self.all_sequences[idx][-2].argmax() for idx in x]
            ros = RandomOverSampler()
            x_resampled, y_resampled = ros.fit_resample(np.array(x).reshape(-1, 1), y)
            self.kfold_indexes = x_resampled.ravel()

        if self.oversample_all_factor > 1.0:
            x = copy(self.kfold_indexes)
            y = np.array([self.all_sequences[idx][-2].argmax() for idx in x])
            non_ards_n = int(len(y[y==0])*self.oversample_all_factor)
            ards_n = int(len(y[y==1])*self.oversample_all_factor)
            ros = RandomOverSampler(sampling_strategy={0: non_ards_n, 1: ards_n})
            x_resampled, y_resampled = ros.fit_resample(np.array(x).reshape(-1, 1), y)
            self.kfold_indexes = x_resampled.ravel()

    def set_undersampling_indices(self):
        if not self.train:
            return

        if self.undersample_factor == -1:
            return

        undersampler = PatientLevelHomogeneityUndersampler(self.undersample_factor, self.undersample_std_factor)
        x = copy(self.kfold_indexes)
        gt = self.get_ground_truth_df()
        self.kfold_indexes, _ = undersampler.fit_resample(x, gt)

    def handle_fractional_patient_dataset(self):
        """
        If we are using a partial training dataset, then this function selects
        data that we want to use.
        """
        if self.train_patient_fraction == 1.0:
            return

        if self.total_kfolds:
            # get unique patients in kfold
            uniq_patients = set()
            for i in self.kfold_indexes:
                uniq_patients.add(self.all_sequences[i][0])
            # get data on patients from cohort file
            patient_data = self.cohort[self.cohort.patient_id.isin(list(uniq_patients)) & (self.cohort.experiment_group == self.experiment_num)]
            # Get ARDS+OTHER patients
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
        if self.total_kfolds is not None:
            indices = {
                kfold_num: self.get_kfold_indexes_for_fold(kfold_num)
                for kfold_num in range(self.total_kfolds)
            }
        else:
            indices = {None: range(len(self.all_sequences))}

        if 'padded_breath_by_breath' in self.dataset_type:
            is_padded = True
        elif 'unpadded' in self.dataset_type:
            is_padded = False
        else:
            raise Exception('unsupported dataset type {} for scaling'.format(self.dataset_type))

        self.scaling_factors = {
            kfold_num: self._get_scaling_factors_for_indices(idxs, is_padded)
            for kfold_num, idxs in indices.items()
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
            undersample_factor=-1,
        )
        return test_dataset

    @classmethod
    def from_pickle(cls,
                    data_path,
                    oversample_minority,
                    train_patient_fraction,
                    transforms,
                    undersample_factor,
                    undersample_std_factor,
                    oversample_all_factor,
                    butter_filter,
                    add_fft,
                    only_fft,
                    fft_real_only,):
        dataset = pd.read_pickle(data_path)
        if not isinstance(dataset, ARDSRawDataset) and not isinstance(dataset, deepards.dataset.ARDSRawDataset):
            raise ValueError('The pickle file you have specified is out-of-date. Please re-process your dataset and save the new pickled dataset.')
        dataset.oversample_minority = oversample_minority
        dataset.train_patient_fraction = train_patient_fraction
        dataset.transforms = transforms
        dataset.undersample_factor = undersample_factor
        dataset.undersample_std_factor = undersample_std_factor
        dataset.oversample_all_factor = oversample_all_factor
        if butter_filter is not None:
            b, a = butter(1, butter_filter)
            dataset.butter_filter = lambda x: filtfilt(b, a, x, axis=-1)
        else:
            dataset.butter_filter = None
        try:
            dataset.add_fft
        except AttributeError:
            dataset.add_fft = False
            dataset.only_fft = False
        # xor
        if (add_fft or only_fft) and not (dataset.add_fft or dataset.only_fft):
            run_new_fft = True
        else:
            run_new_fft = False
        dataset.add_fft = add_fft
        dataset.only_fft = only_fft
        dataset.fft_real_only = fft_real_only
        if run_new_fft:
            dataset._perform_fft()
            dataset.derive_scaling_factors()
        return dataset

    def set_kfold_indexes_for_fold(self, kfold_num):
        self.kfold_num = kfold_num
        self.kfold_indexes = self.get_kfold_indexes_for_fold(kfold_num)
        self.handle_fractional_patient_dataset()
        # undersample before oversampling because undersampling was built without
        # assumption that we would ever oversample
        self.set_undersampling_indices()
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
        # XXX need to do is_padded.
        # XXX need to test this
        chans = self.all_sequences[0][1].shape[1]
        std_sum = np.array([0] * chans, dtype=np.float)
        mean_sum = np.array([0] * chans, dtype=np.float)
        obs_count = 0

        for idx in indices:
            obs = self.all_sequences[idx][1]
            obs_count += np.prod(obs.shape[0:2])
            mean_sum += obs.sum(axis=-1).sum(axis=0)
        mu = mean_sum / obs_count
        mu = mu.reshape(1, chans, 1).repeat(self.seq_len, axis=-1).repeat(self.n_sub_batches, axis=0)

        # calculate std
        for idx in indices:
            obs = self.all_sequences[idx][1]
            std_sum += ((obs - mu) ** 2).sum(axis=-1).sum(axis=0)
        std = np.sqrt(std_sum / obs_count)
        std = std.reshape(1, chans, 1).repeat(self.seq_len, axis=-1).repeat(self.n_sub_batches, axis=0)
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

    def get_unpadded_sequences_dataset_with_bm_data(self, processing_func, target_func):
        if self.whole_patient_super_batch:
            raise NotImplementedError('We havent implemented super batch with this data type')
        last_patient = None
        indices = [EXPERIMENTAL_META_HEADER.index(feature) for feature in [
            'mean_flow_from_pef',
            'inst_RR',
            'slope_minF_to_zero',
            'pef_+0.16_to_zero',
            'iTime',
            'eTime',
            'I:E ratio',
            'dyn_compliance',
            'tve:tvi ratio',
        ]]
        for fidx, filename in enumerate(self.raw_files):
            gen = read_processed_file(filename, filename.replace('.raw.npy', '.processed.npy'))
            patient_id = self._get_patient_id_from_file(filename)

            if patient_id != last_patient:
                batch_arr = []
                breath_arr = []
                seq_vent_bns = []
                breath_meta = []
                batch_seq_hours = []

            matching_meta = os.path.join(self.meta_dir, patient_id, 'breath_meta_' + os.path.basename(filename).replace('.raw.npy', '.csv'))
            has_preprocessed_meta = False
            if matching_meta in self.meta_files:
                try:
                    processed_meta = pd.read_csv(matching_meta, header=None).values
                    has_preprocessed_meta = True
                except pd.errors.EmptyDataError:
                    pass

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

                if has_preprocessed_meta:
                    try:
                        meta = processed_meta[bidx]
                    except IndexError:
                        meta = np.array(get_experimental_breath_meta(breath))
                    # sanity check
                    if int(meta[0]) != breath['rel_bn'] or len(EXPERIMENTAL_META_HEADER) != len(meta):
                        meta = np.array(get_experimental_breath_meta(breath))
                else:
                    meta = np.array(get_experimental_breath_meta(breath))
                    # XXX save information to file so it can be retrieved

                breath_meta.append(meta[indices])
                seq_hour = (breath_time - start_time).total_seconds() / 60 / 60
                seq_vent_bns.append(breath['vent_bn'])
                flow = self.truncate_lim(breath['flow'])
                batch_arr, breath_arr, batch_seq_hours = processing_func(
                    flow, breath_arr, batch_arr, batch_seq_hours, seq_hour
                )

                if len(batch_arr) == self.n_sub_batches:
                    raw_data = np.array(batch_arr)
                    breath_meta = np.array(breath_meta).astype(float)
                    if self._should_we_drop_frame(raw_data.ravel(), seq_vent_bns, patient_id):
                        # drop breath arr to be safe
                        breath_arr = []
                        batch_arr = []
                        seq_vent_bns = []
                        batch_seq_hours = []
                        breath_meta = []
                        continue
                    breath_window = raw_data.reshape((self.n_sub_batches, 1, self.seq_len))
                    self.all_sequences.append([
                        patient_id,
                        breath_window,
                        np.mean(breath_meta, axis=0),
                        np.nanmedian(breath_meta, axis=0),
                        target,
                        batch_seq_hours,
                    ])
                    batch_arr = []
                    seq_vent_bns = []
                    batch_seq_hours = []
                    breath_meta = []

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

    def _perform_fft(self):
        if not self.add_fft and not self.only_fft:
            return
        for idx, (pt, seq, _, __) in enumerate(self.all_sequences):
            trans = np.fft.fft(seq, axis=-1)
            fft_chans = [trans.real] if self.fft_real_only else [trans.real, trans.imag]
            if self.add_fft:
                new_seq = np.concatenate([seq]+fft_chans, axis=1)
            elif self.fft_only:
                new_seq = np.concatenate(fft_chans, axis=1)
            self.all_sequences[idx][1] = new_seq

    def __getitem__(self, index):
        # This is a bit tricky because pytorch will just assume that indexing
        # goes from 0 ... __len__ - 1. But this is not the case for kfold. So you
        # have to get the pytorch index and then translate that to what it looks
        # like in your kfold. So translate the relative indexing of pytorch to
        # absolute indexing of the dataset
        if self.kfold_num is not None:
            index = self.kfold_indexes[index]
        seq = self.all_sequences[index]
        if len(seq) == 4:
            _, data, target, seq_hours = seq
            meta = np.nan
        elif len(seq) == 5:
            _, data, meta, target, seq_hours = seq
        # uses mean and median breath meta
        elif len(seq) == 6:
            _, data, m, mm, target, seq_hours = seq
            meta = np.array([m, mm])

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

        if self.butter_filter is not None:
            data = self.butter_filter(data).copy()
        # this will return absolute index of data, the data, metadata, and target
        # by absolute index we mean the indexing in self.all_sequences
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
            # uses mean and median breath meta
            elif len(seq) == 6:
                patient, _, __, ___, target, hrs = seq
            rows.append([patient, np.argmax(target, axis=0), hrs[0]])
        return pd.DataFrame(rows, columns=['patient', 'y', 'hour'])

    def _get_kfold_ground_truth(self):
        rows = []
        for idx in self.kfold_indexes:
            seq = self.all_sequences[idx]
            if len(seq) == 4:
                patient, _, target, hrs = seq
            elif len(seq) == 5:
                patient, _, __, target, hrs = seq
            # uses mean and median breath meta
            elif len(seq) == 6:
                patient, _, __, ___, target, hrs = seq
            rows.append([patient, np.argmax(target, axis=0), hrs[0]])
        return pd.DataFrame(rows, columns=['patient', 'y', 'hour'], index=self.kfold_indexes)

    def get_abs_bs_dt(self, breath):
        if isinstance(breath['abs_bs'], bytes):
            abs_bs = breath['abs_bs'].decode('utf-8')
        else:
            abs_bs = breath['abs_bs']

        try:
            breath_time = pd.to_datetime(abs_bs, format='%Y-%m-%d %H-%M-%S.%f')
        except:
            breath_time = pd.to_datetime(abs_bs, format='%Y-%m-%d %H:%M:%S.%f')
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


class ImgARDSDataset(ARDSRawDataset):
    def __init__(self, raw_dataset_obj, extra_transforms, add_fft, fft_only, bbox, butter_filter=None):
        """
        Create an ARDS dataset composed of 2D images. Operates off an ARDSRawDataset
        object because it has done most of the hard work for us already.

        :param raw_dataset_obj: instance of ARDSRawDataset
        :param extra_transforms: list of 2d transforms you want to use
        :param add_fft: whether or not to add fft to the img
        :param fft_only: True/False only return bounding box data
        :param bbox: True/False create bounding box mixed dataset
        :param butter_filter: A Hz bandwidth cutoff where 0 < Hz < 1
        """
        self.raw = raw_dataset_obj
        self.all_sequences = []
        self.add_fft = add_fft
        self.fft_only = fft_only
        self.bbox = bbox
        self.total_kfolds = self.raw.total_kfolds
        if butter_filter is not None:
            b, a = butter(1, butter_filter)
            self.butter_filter = lambda x: filtfilt(b, a, x, axis=1)
        else:
            self.butter_filter = None
        try:
            self.oversample_minority = self.raw.oversample_minority
        except AttributeError:
            self.oversample_minority = self.raw.oversample
        try:
            self.oversample_all_factor = self.raw.oversample_all_factor
        except AttributeError:  # if this was unset then it was 1.0
            self.oversample_all_factor = 1
        self.seq_hours = dict()
        self.train = self.raw.train
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
        ] + [two_dim_transforms[trans]() for trans in extra_transforms]
        )
        self.test_transforms = transforms.Compose([transforms.ToTensor(),])
        self.dataset_type = self.raw.dataset_type
        if self.dataset_type == 'padded_breath_by_breath':
            raise NotImplementedError('padded dataset types not implemented yet!')
        self.make_dataset_from_raw()
        self.derive_scaling_factors()
        # this will work for train segments of the dataset, however if we want
        # to test on a whole img then it needs to be modified. Possible that we
        # can save items in two competing datasets. But for now lets just focus
        # on the simple bbox problem and then we can work on other items
        if self.bbox and self.train:
            self.make_bbox_dataset()

    def _append_to_mat(self, mat, new_data, seq_hours, new_seq_hours):
        len_win, chans, seq_size = new_data.shape
        existing_rows = sum([m.shape[0] for m in mat])

        if existing_rows + new_data.shape[0] <= seq_size:
            mat.append(new_data.reshape(len_win, seq_size))
            seq_hours.extend(new_seq_hours)
            return mat, [], []
        else:
            n_rows = seq_size - existing_rows
            mat.append(new_data[:n_rows].reshape(n_rows, seq_size))
            # its difficult to figure out exactly how many sequence hours to add.
            # in unpadded sequences there will be multiple breaths in each row.
            breaths_per_row = (len(new_seq_hours) / n_rows) if n_rows > 0 else 0
            n_hrs = int(n_rows*breaths_per_row)
            seq_hours.extend(new_seq_hours[:n_hrs])
            return mat, new_data[n_rows:], new_seq_hours[n_hrs:]

    def _finish_mat(self, pt, img, target, seq_hours):
        if len(img) == 0:
            return
        existing_rows = sum([m.shape[0] for m in img])
        seq_size = img[0].shape[1]
        remaining_rows = seq_size - existing_rows

        if remaining_rows != 0:
            img.append(np.zeros((remaining_rows, seq_size)))

        img = np.expand_dims(np.concatenate(img), axis=-1)
        if self.add_fft:
            trans = np.fft.fft(img, axis=1)
            img = np.concatenate([img, trans.real, trans.imag], axis=-1)
        elif self.fft_only:
            trans = np.fft.fft(img, axis=1)
            img = np.concatenate([trans.real, trans.imag], axis=-1)

        self.all_sequences.append([pt, img, target, seq_hours])

    def _get_scaling_factors_for_indices(self, indices):
        """
        Get mu and std for a specific set of indices
        """
        chans = self.all_sequences[0][1].shape[-1]
        std_sum = np.array([0] * chans, dtype=np.float)
        mean_sum = np.array([0] * chans, dtype=np.float)
        obs_count = 0

        for idx in indices:
            obs = self.all_sequences[idx][1]
            obs_count += np.prod(obs.shape[0:2])
            mean_sum += obs.sum(axis=0).sum(axis=0)
        mu = mean_sum / obs_count
        mu = mu.reshape(1, 1, chans).repeat(224, axis=0).repeat(224, axis=1)

        # calculate std
        for idx in indices:
            obs = self.all_sequences[idx][1]
            std_sum += ((obs - mu) ** 2).sum(axis=0).sum(axis=0)
        std = np.sqrt(std_sum / obs_count)
        std = std.reshape(1, 1, chans).repeat(224, axis=0).repeat(224, axis=1)
        # mu/std should be returned in a matrix of shape (224,224,chans) this will
        # allow us to easily standardize matrices of variable chan size
        return mu, std

    def derive_scaling_factors(self):
        if self.total_kfolds is not None:
            indices = {
                kfold_num: self.get_kfold_indexes_for_fold(kfold_num)
                for kfold_num in range(self.total_kfolds)
            }
        else:
            raise NotImplementedError('holdout is not supported yet for Img datasets')

        self.scaling_factors = {
            kfold_num: self._get_scaling_factors_for_indices(idxs)
            for kfold_num, idxs in indices.items()
        }

    def make_bbox_dataset(self):
        train_kfold_idxs = {
            kfold_num: self.get_kfold_indexes_for_fold(kfold_num)
            for kfold_num in range(self.total_kfolds)
        }
        test_kfold_idxs = dict()
        # have to perform additional processing on this to get the true kfold
        # indices. the current kfold indices that we have are actually train
        # indices for each kfold.
        for i in range(5):
            test_kfold_idxs[i] = train_kfold_idxs[(i+1)%4].difference(train_kfold_idxs[i]).copy()
        # now can perform reverse indexing
        reverse_indices = {
            i: kfold_num for kfold_num, idxs in test_kfold_idxs.items() for i in idxs
        }
        gt = self._get_all_sequence_ground_truth()
        last_pt = None
        # wait reverse indices isnt perfect because an index can be in multiple
        # folds. So its not like theres really a perfect 1-1 correlation here
        #
        for idx, (pt, data, target, seq_hours) in enumerate(self.all_sequences):
            # If I use data from another patient then I have to be sure they're not
            # in cross-over kfolds
            int_target = target.argmax()
            if last_pt != pt:
                pt_fold = reverse_indices[idx]
                fold_idxs = set(test_kfold_idxs[pt_fold])
                pt_idxs = gt[gt.patient == pt].index
                non_pt_fold_idxs = fold_idxs.difference(pt_idxs)
                mask = gt.loc[non_pt_fold_idxs, 'y'] != int_target
                avail_idxs = mask[mask].index
            new_data = data.copy()
            # randomly choose sequence and then choose n rows that is 1/4-1/3 of 224
            rand_seq_idx = np.random.choice(avail_idxs)
            seq_size = data.shape[0]
            n_rows = np.random.randint(seq_size//4, seq_size//3)
            # -1 for 0 offset. At minimum have 10 rows to start/end with.
            row_start = np.random.randint(10, seq_size-n_rows-1-10)
            seq_slice = self.all_sequences[rand_seq_idx][1][row_start:row_start+n_rows]
            row_end = row_start + n_rows
            new_data[row_start:row_end] = seq_slice
            chunks = [
                [0, row_start-1, int_target],
                [row_start, row_end-1, (int_target+1)%2],
                [row_end, seq_size, int_target]
            ]

            # So in the main retinanet work the bbox is structured like:
            #
            # [x1, y1, x2, y2, cls]
            #
            # The overall annotation structure is [Nx5] where N is the
            # number of annotations in a single img
            bboxes = []
            labels = []
            for rs, re, target in chunks:
                bboxes.append([0, rs, seq_size, re+1])
                labels.append(target)
            new_target = {
                'boxes': torch.FloatTensor(bboxes),
                'labels': torch.IntTensor(labels).type(torch.int64),
            }
            self.all_sequences[idx].insert(2, new_data)
            self.all_sequences[idx].insert(-2, new_target)
            last_pt = pt

    def make_dataset_from_raw(self):
        # this could probably be a recursive function but oh well.
        last_pt = None
        last_target = None
        sh = []
        mat = []
        # if the data doesnt have this format then there will be an error
        if len(self.raw.all_sequences[0]) != 4:
            raise NotImplementedError('datasets with breath metadata or other information havent been implemented yet!')

        for pt, data, target, seq_hours in self.raw.all_sequences:
            if last_pt != pt and mat != []:
                # this is pretty hacky
                sh = sh if len(sh) > 0 else [last_hour_obs]
                self._finish_mat(last_pt, mat, last_target, sh)
                mat, sh = [], []

            last_hour_obs = seq_hours[-1]
            mat, remainder, remainder_sh = self._append_to_mat(mat, data, sh, seq_hours)
            if len(remainder) > 0:
                mat = self._finish_mat(pt, mat, target, sh)
                mat, sh = [], []
                # _, __ because there will be no remainder
                mat, _, __ = self._append_to_mat(mat, remainder, sh, remainder_sh)
            last_pt = pt
            last_target = target

        self._finish_mat(pt, mat, last_target, sh)

    def set_kfold_indexes_for_fold(self, kfold_num):
        self.kfold_num = kfold_num
        self.kfold_indexes = self.get_kfold_indexes_for_fold(kfold_num)
        # only oversampling here
        self.set_oversampling_indices()

    def __getitem__(self, index):

        if self.kfold_num is not None:
            index = self.kfold_indexes[index]
        seq = self.all_sequences[index]
        if len(seq) == 4:
            _, data, target, seq_hours = seq
        elif len(seq) == 6:
            _, orig_data, data, target, one_class_target, seq_hours = seq
        meta = np.nan

        self.seq_hours[index] = seq_hours
        try:
            mu, std = self.scaling_factors[self.kfold_num]
        except AttributeError:
            raise AttributeError(
                'Scaling factors not found for dataset. You must derive them using '
                'the `derive_scaling_factors` function.'
            )
        # XXX fft logic must be incorporated into transforms too. This will be a
        # bit hard tho, because need to duplicate all ops on one chan into another
        # chan. And even then... your FFT op would probably be corrupted anyhow if
        # you were doing something like window slicing or window warping and you'd
        # have to re-compute the fft in order to get a proper value. So there are
        # probably a number of transforms that I will have to fail on if FFT is
        # asked for.

        data = (data - mu) / std
        if self.butter_filter is not None:
            data = self.butter_filter(data).copy()

        if self.train:
            data = self.train_transforms(data)
        else:
            data = self.test_transforms(data)

        # this will return absolute index of data, the data, metadata, and target
        # by absolute index we mean the indexing in self.all_sequences
        return index, data, meta, target

    def __len__(self):
        if self.kfold_num is None:
            return len(self.all_sequences)
        else:
            return len(self.kfold_indexes)

    @classmethod
    def from_pickle(self):
        # XXX do this in a bit when we have figured all the ins and outs
        raise NotImplementedError('cant get 2d dataset from pickle yet')


def rescale_to_img(img):
    dims = len(img.shape)
    minima = img.min(axis=0).min(axis=0)
    if dims == 2:
        maxima = (img - minima).max()
        return np.uint8(((img - minima) / maxima) * 255)
    elif dims == 3:
        chans = img.shape[-1]
        minima = minima.reshape(1, 1, chans).repeat(224, axis=0).repeat(224, axis=1)
        maxima = (img - minima).max(axis=0).max(axis=0)
        maxima = maxima.reshape(1, 1, chans).repeat(224, axis=0).repeat(224, axis=1)
        img = np.uint8(((img - minima) / maxima) * 255)
        if chans == 2:
            img = np.dstack((img, np.ones((224,224,1))))
        return img


def non_bbox_viz(a):
    from matplotlib import pyplot as plt
    d = ImgARDSDataset(a, [], add_fft=False, fft_only=True, bbox=False)
    d.set_kfold_indexes_for_fold(0)
    kfold_idx = d.kfold_idx
    idx, seq, meta, target = d[0]
    #distort = transforms.RandomPerspective(distortion_scale=.1, p=1)
    #aff = transforms.RandomAffine(25)
    # jittering doesnt work very well
    #jit = transforms.ColorJitter(brightness=0.00001)
    #trans = transforms.RandomErasing(p=1)
    #trans = RandomTimeWarp(p=1)
    #trans = RandomWindowWarping(p=1)
    #trans = RandomMagnitudeWarp(p=1)
    #trans = RandomRowScale(p=1)
    #trans = RandomWindowSlicing(p=1)
    trans = RowShuffle(p=1)
    fig, axes = plt.subplots(nrows=1, ncols=3)
    ax = plt.subplot(1, 3, 1)
    if seq.shape[0] == 1:
        shape_op = lambda x: x.reshape(224,224)
    else:
        # imshow needs things in (h,w,c) whereas torch likes (c,h,w)
        shape_op = lambda x: np.rollaxis(x, 0, 3)

    ax.imshow(rescale_to_img(d.all_sequences[idx][1][:, :, 0].reshape(224, 224)))
    ax.set_title('original')
    ax = plt.subplot(1, 3, 2)
    ax.imshow(rescale_to_img(shape_op(seq.numpy())))
    ax.set_title('dataset out')
    ax = plt.subplot(1, 3, 3)
    ax.imshow(rescale_to_img(shape_op(trans(seq.numpy()))))
    ax.set_title('trans modified')
    plt.show()


def bbox_viz(a):
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    a.train = True
    d = ImgARDSDataset(a, [], add_fft=False, fft_only=False, bbox=True)
    rel_idx = 0
    d.train = True
    d.set_kfold_indexes_for_fold(0)
    abs_idx = d.kfold_indexes[rel_idx]

    idx, seq, meta, target = d[rel_idx]
    d.train = False
    pt, orig_seq, _, __, ___, ____ = d.all_sequences[abs_idx]
    fig, axes = plt.subplots(nrows=1, ncols=3)
    fig.set_figheight(10)
    fig.set_figwidth(14)
    if seq.shape[0] == 1:
        shape_op = lambda x: x.reshape(224,224)
    else:
        # imshow needs things in (h,w,c) whereas torch likes (c,h,w)
        shape_op = lambda x: np.rollaxis(x, 0, 3)

    ax = plt.subplot(1, 3, 1)
    ax.imshow(shape_op(orig_seq))
    ax.set_title('original')
    ax = plt.subplot(1, 3, 2)
    ax.imshow(shape_op(seq))
    ax.set_title('bbox modified ')
    ax = plt.subplot(1, 3, 3)
    patho_map = {'0': 'Non-ARDS', '1': 'ARDS'}
    for idx, (x1, y1, x2, y2) in enumerate(target['boxes']):
        cls = target['labels'][idx].item()
        r = Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor='r', lw=.75)
        ax.add_patch(r)
        #ax.annotate(patho_map[str(cls)], (x1+((x2-x1)/2)-25, y1+((y2-y1)/2)), color='r', fontsize=16)
    ax.imshow(rescale_to_img(shape_op(seq.numpy())))
    ax.set_title('with annotations')
    plt.savefig('bbox-img.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    a = pd.read_pickle('/fastdata/deepards/unpadded_centered_sequences-nb20-kfold.pkl')
    bbox_viz(a)
