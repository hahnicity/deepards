import argparse
import math
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import yaml

patient_map = {
    'ards_train': ['0723RPI2120190416', '0015RPI0320150401', '0021RPI0420150513', '0026RPI1020150523', '0027RPI0620150525', '0093RPI0920151212', '0098RPI1420151218', '0099RPI0120151219', '0102RPI0120151225', '0120RPI1820160118', '0129RPI1620160126', '0147RPI1220160213', '0148RPI0120160214', '0149RPI1820160212', '0153RPI0720160217', '0194RPI0320160317', '0209RPI1920160408', '0224RPI3020160414', '0243RPI0720160512', '0245RPI1420160512', '0253RPI1220160606', '0260RPI2420160617', '0265RPI2920160622', '0266RPI1720160622', '0268RPI1220160624', '0271RPI1220160630', '0372RPI2220161211', '0381RPI2320161212', '0390RPI2220161230', '0412RPI5520170121', '0484RPI4220170630', '0506RPI3720170807', '0511RPI5220170831', '0514RPI5420170905', '0527RPI0420171028', '0546RPI5120171216', '0549RPI4420171213', '0551RPI0720180102', '0569RPI0420180116', '0640RPI2820180822'],
    'other_train': ['0033RPI0520150603', '0108RPI0120160101', '0111RPI1520160101', '0112RPI1620160105', '0124RPI1220160123', '0125RPI1120160123', '0133RPI0920160127', '0144RPI0920160212', '0145RPI1120160212', '0157RPI0920160218', '0163RPI0720160222', '0166RPI2220160227', '0170RPI2120160301', '0173RPI1920160303', '0257RPI1220160615', '0304RPI1620160829', '0306RPI3520160830', '0317RPI3220160910', '0336RPI3920161006', '0343RPI3920161016', '0347RPI4220161016', '0354RPI5820161029', '0356RPI2220161101', '0361RPI4620161115', '0365RPI5820161125', '0387RPI3920161224', '0398RPI4220170104', '0423RPI3220170205', '0434RPI4520170224', '0460RPI2220170518', '0463RPI3220170522', '0544RPI2420171204', '0545RPI0520171214', '0552RPI2520180101', '0585RPI2720180206', '0593RPI1920180226', '0624RPI0320180708', '0624RPI1920180702', '0625RPI2820180628', '0705RPI5020190318'],
    'ards_test': ['0127RPI0120160124', '0411RPI5820170119', '0261RPI1220160617',
       '0235RPI1320160426', '0160RPI1420160220', '0122RPI1320160120',
       '0251RPI1820160609', '0139RPI1620160205', '0357RPI3520161101',
       '0558RPI0820180104'],
    'other_test': ['0443RPI1620170319', '0410RPI4120170118', '0380RPI3920161212',
       '0745RPI1900000000', '0135RPI1420160203', '0231RPI1220160424',
       '0137RPI1920160202', '0315RPI2720160910', '0132RPI1720160127',
       '0225RPI2520160416'],
    'aim1_train': ['0271RPI1220160630',
         '0027RPI0620150525',
         '0625RPI2820180628',
         '0343RPI3920161016',
         '0209RPI1920160408',
         '0372RPI2220161211',
         '0194RPI0320160317',
         '0149RPI1820160212',
         '0245RPI1420160512',
         '0257RPI1220160615',
         '0357RPI3520161101',
         '0268RPI1220160624',
         '0260RPI2420160617',
         '0412RPI5520170121',
         '0365RPI5820161125',
         '0434RPI4520170224',
         '0387RPI3920161224',
         '0546RPI5120171216',
         '0527RPI0420171028',
         '0593RPI1920180226',
         '0253RPI1220160606',
         '0108RPI0120160101',
         '0170RPI2120160301',
         '0390RPI2220161230',
         '0133RPI0920160127',
         '0511RPI5220170831',
         '0111RPI1520160101',
         '0225RPI2520160416',
         '0304RPI1620160829',
         '0398RPI4220170104',
         '0033RPI0520150603',
         '0347RPI4220161016',
         '0231RPI1220160424',
         '0144RPI0920160212',
         '0315RPI2720160910',
         '0265RPI2920160622',
         '0544RPI2420171204',
         '0098RPI1420151218',
         '0261RPI1220160617',
         '0624RPI1920180702',
         '0460RPI2220170518',
         '0463RPI3220170522',
         '0624RPI0320180708',
         '0317RPI3220160910',
         '0251RPI1820160609',
         '0585RPI2720180206',
         '0166RPI2220160227',
         '0423RPI3220170205',
         '0153RPI0720160217',
         '0021RPI0420150513',
         '0551RPI0720180102',
         '0102RPI0120151225',
         '0361RPI4620161115',
         '0160RPI1420160220',
         '0127RPI0120160124',
         '0545RPI0520171214',
         '0235RPI1320160426',
         '0122RPI1320160120',
         '0139RPI1620160205',
         '0266RPI1720160622',
         '0129RPI1620160126',
         '0484RPI4220170630',
         '0506RPI3720170807',
         '0354RPI5820161029',
         '0093RPI0920151212',
         '0640RPI2820180822',
         '0356RPI2220161101',
         '0443RPI1620170319',
         '0124RPI1220160123',
         '0410RPI4120170118'],
    'aim1_test': ['0411RPI5820170119',
         '0224RPI3020160414',
         '0336RPI3920161006',
         '0147RPI1220160213',
         '0514RPI5420170905',
         '0099RPI0120151219',
         '0558RPI0820180104',
         '0552RPI2520180101',
         '0148RPI0120160214',
         '0243RPI0720160512',
         '0549RPI4420171213',
         '0163RPI0720160222',
         '0132RPI1720160127',
         '0026RPI1020150523',
         '0015RPI0320150401',
         '0380RPI3920161212',
         '0120RPI1820160118',
         '0137RPI1920160202',
         '0381RPI2320161212',
         '0112RPI1620160105',
         '0135RPI1420160203',
         '0723RPI2120190416',
         '0306RPI3520160830',
         '0173RPI1920160303',
         '0569RPI0420180116',
         '0125RPI1120160123',
         '0705RPI5020190318',
         '0745RPI1900000000',
         '0157RPI0920160218',
         '0145RPI1120160212'],
}


class Splitting(object):
    def __init__(self, dataset_path, cohort_file):
        self.dataset_path = dataset_path
        self.all_data_dir = os.path.join(self.dataset_path, 'experiment1/all_data')
        self.all_data_raw_dir = os.path.join(self.all_data_dir, 'raw')
        self.all_data_meta_dir = os.path.join(self.all_data_dir, 'meta')
        self.original_map = patient_map
        self.ards_pts = self.original_map['ards_train'] + self.original_map['ards_test']
        self.other_pts = self.original_map['other_train'] + self.original_map['other_test']
        if cohort_file:
            cohort_file = pd.read_csv(cohort_file)
            self.ards_pts = cohort_file[cohort_file['Pathophysiology'] == 'ARDS']['Patient Unique Identifier'].astype(str).tolist()
            self.other_pts = cohort_file[cohort_file['Pathophysiology'] != 'ARDS']['Patient Unique Identifier'].astype(str).tolist()

    def perform_preset_proto_split(self):
        train_pts = self.original_map['ards_train'] + self.original_map['other_train']
        test_pts = self.original_map['ards_test'] + self.original_map['other_test']
        self.create_split(train_pts, 'prototrain')
        self.create_split(test_pts, 'prototest')

    def perform_preset_aim1_split(self):
        self.create_split(self.original_map['aim1_train'], 'aim1_70_30_training')
        self.create_split(self.original_map['aim1_test'], 'aim1_70_30_testing')

    def perform_preset_file_split(self, file_path):
        with open(file_path) as preset_file:
            conf = yaml.load(preset_file, Loader=yaml.FullLoader)
        train_pts = conf['train']
        test_pts = conf['test']
        split_name = os.path.splitext(os.path.basename(file_path))[0]
        self.create_split(train_pts, split_name + 'train')
        self.create_split(test_pts, split_name + 'test')

    def perform_random_split(self, split_ratio, validation_ratio, out_dir_prefix, n_train, n_val, n_test):
        all_pts = self.ards_pts + self.other_pts
        if not n_train or not n_val or not n_test:
            n_test = int((len(all_pts) * split_ratio))
            n_val = int(math.ceil(n_test * validation_ratio))
            n_train = len(all_pts) - n_test
        other_test_pts = list(np.random.choice(self.other_pts, size=n_test/2, replace=False))
        ards_test_pts = list(np.random.choice(self.ards_pts, size=n_test/2, replace=False))
        test_pts = other_test_pts + ards_test_pts
        train_pts = list(np.random.choice(list(set(all_pts).difference(set(test_pts))), size=n_train, replace=False))
        dir_prefix = out_dir_prefix if out_dir_prefix is not None else 'random'

        self.create_split(train_pts, '{}train'.format(dir_prefix))
        if n_val > 0:
            remaining_pts = set(all_pts).difference(test_pts).difference(train_pts)
            remaining_other_pts = set(self.other_pts).intersection(remaining_pts)
            remaining_ards_pts = set(self.ards_pts).intersection(remaining_pts)
            ards_val_pts = np.random.choice(list(remaining_ards_pts), size=n_val/2, replace=False)
            other_val_pts = np.random.choice(list(remaining_other_pts), size=n_val/2, replace=False)
            val_pts = list(ards_val_pts) + list(other_val_pts)
            self.create_split(val_pts, '{}val'.format(dir_prefix))
        self.create_split(test_pts, '{}test'.format(dir_prefix))
        print('Performed random split for {} train patients, {} validation patients, {} test patients'.format(
            n_train,
            n_val,
            n_test
        ))

    def create_split(self, pts, main_dirname):
        dir = os.path.join(self.dataset_path, 'experiment1', main_dirname)
        try:
            shutil.rmtree(dir)
        except OSError:
            pass
        os.mkdir(dir)
        raw_dir = os.path.join(dir, 'raw')
        meta_dir = os.path.join(dir, 'meta')
        os.mkdir(raw_dir)
        os.mkdir(meta_dir)

        for pt in pts:
            proc = subprocess.Popen(['ln', '-s', os.path.join(self.all_data_raw_dir, pt), raw_dir])
            proc.communicate()
            proc = subprocess.Popen(['ln', '-s', os.path.join(self.all_data_meta_dir, pt), meta_dir])
            proc.communicate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset-path', required=True)
    parser.add_argument('-c', '--cohort-file')
    parser.add_argument('set_type', choices=['preset_proto', 'preset_aim1', 'random', 'preset_file'], help="""
        Split your data in a specific format:

        *preset_proto:* Utilize the proto train/test split. As it implies, used for prototyping purposes and shouldn't be used for result reporting
        *preset_aim1:* Use the preset holdout split we used for initial Aim 1 paper.
        *random:* Use a random split of the patients with a validation set.
    """)
    parser.add_argument('-sr', '--split-ratio', type=float, default=1/6.0, help='The ratio of dataset to use for test set (both validation and final testing)')
    parser.add_argument('-vr', '--validation-ratio', type=float, default=1/6.0, help='Ratio of the dataset to split into the validation set. Only used for the random split type. If you dont want a validation set, set this argument to 0')
    parser.add_argument('-o', '--out-dir', help='New directory to place train/test splits. Only used for random splits. If unset will just revert to default "random"')
    parser.add_argument('-f', '--preset-file', help='Path to file where we set our train/test splits')
    parser.add_argument('-ntr', '--n-train', type=int, help='number of train patients to select. only used on random splits')
    parser.add_argument('-nv', '--n-val', type=int, help='number of validation patients to select. only used on random splits')
    parser.add_argument('-nt', '--n-test', type=int, help='number of test patients to select. only used on random splits')
    args = parser.parse_args()

    splitter = Splitting(args.dataset_path, args.cohort_file)
    if args.set_type == 'preset_proto':
        splitter.perform_preset_proto_split()
    elif args.set_type == 'random':
        splitter.perform_random_split(args.split_ratio, args.validation_ratio, args.out_dir, args.n_train, args.n_val, args.n_test)
    elif args.set_type == 'preset_aim1':
        splitter.perform_preset_aim1_split()
    elif args.set_type == 'preset_file':
        if args.preset_file is None:
            raise Exception('If you are using preset_file split you must set --preset-file flag to a valid filepath')
        splitter.perform_preset_file_split(args.preset_file)


if __name__ == "__main__":
    main()
