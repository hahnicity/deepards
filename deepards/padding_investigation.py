import argparse

import matplotlib.pyplot as plt
import pandas as pd

from dataset import ARDSRawDataset


parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('kfold', type=int)
parser.add_argument('patient', help='patient in the test cohort wed like to evaluate')
args = parser.parse_args()

seqs = pd.read_pickle(args.dataset)
train_data = ARDSRawDataset(
    '', '', 'cohort-description.csv', 100, '',
    all_sequences=seqs, kfold_num=args.kfold, total_kfolds=5, train=True
)
test_data = ARDSRawDataset(
    '', '', 'cohort-description.csv', 100, '',
    all_sequences=seqs, kfold_num=args.kfold, total_kfolds=5, train=False
)
gt_train = train_data.get_ground_truth_df()
train_pts = gt_train.patient.unique()
gt_test = test_data.get_ground_truth_df()
pt_idx = gt_test[gt_test.patient == args.patient].index
train_data = [r[1] for r in seqs if r[0] in train_pts]
pt_data = [r[1] for r in seqs if r[0] == args.patient]

unpadded_train_seq_lens = [
    len(train_data[i][j].reshape(224)[train_data[i][j].reshape(224) != 0])
    for i in range(len(train_data))
    for j in range(train_data[i].shape[0])
]
unpadded_pt_seq_lens = [
    len(pt_data[i][j].reshape(224)[pt_data[i][j].reshape(224) != 0])
    for i in range(len(pt_data))
    for j in range(pt_data[i].shape[0])
]

plt.hist(unpadded_train_seq_lens, bins=100)
plt.title('Unpadded Train sequence lens')
plt.show()

plt.hist(unpadded_pt_seq_lens, bins=100)
plt.title('Unpadded patient {} sequence lens'.format(args.patient))
plt.show()
