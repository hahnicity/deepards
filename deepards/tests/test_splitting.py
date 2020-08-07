import os
import shutil
import tempfile

import pandas as pd

from deepards.perform_data_splitting import Splitting


class TestSplitting(object):
    def setup(self):
        self.dirpath = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.dirpath, 'experiment1', 'all_data', 'raw')
        os.makedirs(self.raw_dir)
        os.makedirs(os.path.join(self.dirpath, 'experiment1', 'all_data', 'meta'))

    def test_make_limited_n_pts(self):
        fake_csv = [
            ['Patient Unique Identifier', 'Pathophysiology'],
            ['A', 'ARDS'],
            ['B', 'ARDS'],
            ['C', 'ARDS'],
            ['D', 'OTHER'],
            ['E', 'OTHER'],
            ['F', 'COPD'],
            ['AA', 'ARDS'],
            ['BA', 'ARDS'],
            ['CA', 'ARDS'],
            ['DA', 'OTHER'],
            ['EA', 'OTHER'],
            ['FA', 'COPD'],
        ]
        for pt, _ in fake_csv[1:]:
            os.mkdir(os.path.join(self.raw_dir, pt))
        csv_path = os.path.join(self.dirpath, 'fake.csv')
        pd.DataFrame(fake_csv[1:], columns=fake_csv[0]).to_csv(csv_path, index=False)
        splitter = Splitting(self.dirpath, csv_path)
        splitter.perform_random_split(None, None, None, 2, 2, 2)
        train_dirs = os.listdir(os.path.join(self.dirpath, 'experiment1', 'randomtrain', 'raw'))
        val_dirs = os.listdir(os.path.join(self.dirpath, 'experiment1', 'randomval', 'raw'))
        test_dirs = os.listdir(os.path.join(self.dirpath, 'experiment1', 'randomtest', 'raw'))
        assert len(train_dirs) == 2
        assert len(val_dirs) == 2
        assert len(test_dirs) == 2, len(test_dirs)

    def test_random_split(self):
        fake_csv = [
            ['Patient Unique Identifier', 'Pathophysiology'],
            ['A', 'ARDS'],
            ['B', 'ARDS'],
            ['C', 'ARDS'],
            ['D', 'OTHER'],
            ['E', 'OTHER'],
            ['F', 'COPD'],
        ]
        for pt, _ in fake_csv[1:]:
            os.mkdir(os.path.join(self.raw_dir, pt))
        csv_path = os.path.join(self.dirpath, 'fake.csv')
        pd.DataFrame(fake_csv[1:], columns=fake_csv[0]).to_csv(csv_path, index=False)
        splitter = Splitting(self.dirpath, csv_path)
        splitter.perform_random_split(1/3.0, 0, None, None, None, None)
        train_dirs = os.listdir(os.path.join(self.dirpath, 'experiment1', 'randomtrain', 'raw'))
        try:
            val_dirs = os.listdir(os.path.join(self.dirpath, 'experiment1', 'randomval', 'raw'))
            assert False, "a validation dir was created and this was not requested to happen"
        except:
            pass
        test_dirs = os.listdir(os.path.join(self.dirpath, 'experiment1', 'randomtest', 'raw'))
        assert len(train_dirs) == 4, len(train_dirs)
        assert len(test_dirs) == 2, len(test_dirs)

    def teardown(self):
        shutil.rmtree(self.dirpath)
