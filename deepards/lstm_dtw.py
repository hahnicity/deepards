"""
lstm_dtw
~~~~~~~~

Compare LSTM with DTW
"""
import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from deepards.config import Configuration
from deepards.dataset import ARDSRawDataset
from deepards.train_ards_detector import build_parser, LSTMOnlyModel


class LSTMDTW(object):
    def __init__(self, model1, model2, dataset):
        model_regex = re.compile(r'epoch(\d+)-fold(\d).pth')
        self.model1 = torch.load(model1)
        self.model2 = torch.load(model2)
        self.dataset = pd.read_pickle(dataset)
        self.dataset.oversample = False
        self.dataset.transforms = None
        self.dataset.undersample_factor = -1

        match1 = model_regex.search(model1)
        match2 = model_regex.search(model2)
        if not match1 or not match2:
            raise Exception('could not find epoch/fold match for the files provided. please make sure you are using kfold')
        self.epoch1, self.fold1 = map(int, match1.groups())
        self.epoch2, self.fold2 = map(int, match2.groups())
        self.score_map = pd.read_pickle(Path(__file__).parent.joinpath('dtw_cache/patient_score_map.pkl'))

    def infer(self):
        test_dataset = ARDSRawDataset.make_test_dataset_if_kfold(self.dataset)
        test_dataset.set_kfold_indexes_for_fold(self.fold1)
        test_loader = DataLoader(test_dataset, 16, True, pin_memory=True)

        config_override = Path(__file__).parent.joinpath(
            'experiment_files/lstm_only_experiment_benchmark.yml'
        )
        args = build_parser().parse_args([])
        args.config_override = str(config_override)
        args = Configuration(args)
        cls = LSTMOnlyModel(args)
        cls.run_test_epoch(self.epoch1, self.model1, test_dataset, test_loader, self.fold1)

        test_dataset.set_kfold_indexes_for_fold(self.fold2)
        test_loader = DataLoader(test_dataset, 16, True, pin_memory=True)
        cls.run_test_epoch(self.epoch2, self.model2, test_dataset, test_loader, self.fold2)
        # hypothesis #1. that patients with higher dtw scores perform more poorly
        # as epochs increases.
        #
        testing_results = cls.results.results
        for pt in testing_results.patient.unique():
            testing_results.loc[testing_results.patient == pt, 'dtw_mean'] = np.mean(self.score_map[pt])
            testing_results.loc[testing_results.patient == pt, 'dtw_median'] = np.median(self.score_map[pt])
            testing_results.loc[testing_results.patient == pt, 'dtw_std'] = np.std(self.score_map[pt])
        import IPython; IPython.embed()
        # - this doesnt seem correct because there is slightly less heterogeneity
        # in the patients that were predicted incorrectly. There is rougly equivalent
        # std. So is the problem not heterogeneity but rather homogeneity? and
        # if so, would homogeneity undersampling help?
        #
        # hypothesis #2. It's not the heterogeneity in the testing set, but rather
        # in the training set that is causing problems. So more heterogeneous
        # training sets would cause the model to perform worse.
        #
        # - av dtw for fold 0: 8163.49
        # - av dtw for fold 1: 8286.91
        # - av dtw for fold 2: 8034.77
        # - av dtw for fold 3: 8403.07
        # - av dtw for fold 4: 8237.86
        #
        # No difference in DTW training scores across pts.
        #
        # hypothesis #3. its specific patients looking like class instances of
        # other classes that is causing the problem here.
        #
        # - this isnt a hypothesis so much as it is stating the obvious. But
        # we'd have to take a deep look into the waveforms on the mispredicted
        # patients to figure out if they look similar to conflicting patho data.
        #
        # Here is what my misprediction stuff looks like:
        #
        #               patient patho epoch_num  dtw_median
        #               2   0139RPI1620160205     1         1    9555.790
        #               5   0147RPI1220160213     1         1    4278.140
        #               6   0148RPI0120160214     1         1    7476.360
        #               9   0157RPI0920160218     0         1    2228.400
        #               15  0194RPI0320160317     1         1   11604.215
        #               24  0145RPI1120160212     0         6    5201.730
        #               25  0147RPI1220160213     1         6    4278.140
        #               26  0148RPI0120160214     1         6    7476.360
        #               29  0157RPI0920160218     0         6    2228.400
        #               31  0163RPI0720160222     0         6    3784.030
        #               35  0194RPI0320160317     1         6   11604.215
        #               37  0225RPI2520160416     0         6    5585.630
        #
        # Well what helps? Augmentations. Maybe homogeneity undersampling? But these
        # are specific actions. Is there something I can link to heterogeneity?
        # I think the real question we want to answer is what role does heterogeneity
        # have to play in machine learning with time series data? And how to show
        # it has / doesn't have a role.
        #
        # hypothesis #4. Once again it comes down to individual breaths and reads.
        # If the read is higher/lower DTW score then there is more likelihood of
        # a misclassification.
        phf = cls.results.pred_to_hour_frame
        for pt in phf.patient.unique():
            try:
                phf.loc[phf.patient == pt, 'dtw'] = [0] + self.score_map[pt]
            except ValueError:
                phf.loc[phf.patient == pt, 'dtw'] = self.score_map[pt]
        plt.hist(phf[(phf.patient == '0145RPI1120160212') & (phf.y != phf.pred)]['dtw'].values, bins=100, alpha=0.5)
        # - judging by simple probability distribution of items that are misclassified
        # the distrbution for misclassified is same as correct wrt dtw. Then what
        # about the overtrained patients?
        # They are:
        # * 0145RPI1120160212 - here the neqs are more clustered around homogeneity
        # rather than heterogeneity.
        # * 0163RPI0720160222 - hard here to say. there's not enuf data, but once
        # again most of the mispredictions are centered around homogeneity.
        # * 0225RPI2520160416 - hard to say again, because mispredictions are so
        # pervasive across all testing. but once again its centered around
        # homogeneity. But theres enough around heterogeneous zones to mean theres
        # no real correlation.
        #
        # I think for this fold, the AUC is so bad because the model is just so
        # incorrectly confident for these patients.
        #
        # These results could merely be coincidental very easily given the majority
        # of data for these 3 patients is all homogeneous

        # hypothesis #5. there is something happening on the later epochs that
        # is contributing to overtraining.
        #
        # hypothesis #6. homogeneity undersampling with oversampling retains
        # competitive performance with normal runs with oversampling
        # - nope
        #
        # hypothesis #7: Having a map or an index of past DTW sequences and how they
        # compare to other sequences for the same patient is the best way to filter
        # sequences. Just looking ahead 1 sequence is not good enough, but may be
        # used as a start.

        dtws = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model1')
    parser.add_argument('model2')
    parser.add_argument('dataset', help='path to pickled dataset')
    args = parser.parse_args()

    cls = LSTMDTW(args.model1, args.model2, args.dataset)
    cls.infer()


if __name__ == "__main__":
    main()
