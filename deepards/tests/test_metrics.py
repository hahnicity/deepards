import os
import shutil

import line_profiler
import numpy as np
import pandas as pd

from deepards.metrics import DeepARDSResults, dtw_lib


class TestMetrics(object):
    def __init__(self):
        self.test_cls = DeepARDSResults(0, "testing", testing=True)
        fname = os.path.join(os.path.dirname(__file__), 'test_dataset.pkl')
        self.test_dataset = pd.read_pickle(fname)
        self.dtw_cache_dir = os.path.join(os.path.dirname(__file__), 'dtw_cache')

    def test_dtw_speed(self):
        mock_preds = pd.Series(np.random.choice([0, 1], size=len(self.test_dataset.all_sequences)))
        gt = self.test_dataset.get_ground_truth_df()
        hours = np.random.randint(0, 24, size=len(self.test_dataset.all_sequences)).reshape((len(self.test_dataset.all_sequences), 1))
        self.test_cls.save_predictions_by_hour(gt[['patient', 'y']], mock_preds, hours)
        self.test_cls.perform_dtw_preprocessing(self.test_dataset, self.dtw_cache_dir)

    def test_full_dataset_dtw_speed(self):
        # NOTE this test takes a long time. If you want to run it you are welcome tho
        return
        full_data_path = '/fastdata/deepards/unpadded_centered_sequences-nb20-test-holdout.pkl'
        full_dataset = pd.read_pickle(full_data_path)
        mock_preds = pd.Series(np.random.choice([0, 1], size=len(full_dataset.all_sequences)))
        gt = full_dataset.get_ground_truth_df()
        hours = [seq[-1] for seq in full_dataset.all_sequences]
        self.test_cls.save_predictions_by_hour(gt[['patient', 'y']], mock_preds, hours)
        self.test_cls.perform_dtw_preprocessing(full_dataset, self.dtw_cache_dir)

    def teardown(self):
        dirs = os.listdir(self.dtw_cache_dir)
        for d in dirs:
            path = os.path.join(self.dtw_cache_dir, d)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
