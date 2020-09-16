"""
analyze_predictions
~~~~~~~~~~~~~~~~~~~

Analyze predictions made for a machine learning run by using breath metadata to
see if it gives any insight

So what is the basic idea here. We'll need to iterate on it of course but I think
first you want to look into how the deep learning classifier is able to use the breath
information more effectively compared to a regular RF. So RF is just operating on
features. Would t-SNE help?

I think it might be helpful to look at how the distributions for each individual
feature and prediction matches up in RF versus DL.

So even if you were able to display a split in the data similar to the training set,
you still would have to address causality to show the deep network is using the feature
in some way or the other. You can trivially address causality in regular NN because
the features are the input. However, it is much more difficult to do this in a deep
network.
"""
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp

feature_mapping = {
    0: 'mean_flow_from_pef',
    1: 'inst_RR',
    2: 'slope_minF_to_zero',
    3: 'pef_+0.16_to_zero',
    4: 'iTime',
    5: 'eTime',
    6: 'I:E ratio',
    7: 'dyn_compliance',
    8: 'tve:tvi ratio',
}


def main():
    parser = ArgumentParser()
    parser.add_argument('experiment_name')
    parser.add_argument('dataset')
    parser.add_argument('-e', '--epoch', type=int, default=4, help='epoch to analyze')
    parser.add_argument('--patho', choices=['ards', 'other'], required=True)
    args = parser.parse_args()

    dataset = pd.read_pickle(args.dataset)
    results_files = list(Path(__file__).parent.joinpath('results/').glob('*' + args.experiment_name + '*.pkl'))
    assert len(results_files) == 1
    results = pd.read_pickle(results_files[0])
    patho_n = {'other': 0, 'ards': 1}[args.patho]
    preds_df = results.all_pred_to_hour[(results.all_pred_to_hour.epoch == args.epoch) & (results.all_pred_to_hour.y == patho_n)]
    gt = dataset._get_all_sequence_ground_truth()

    # analyze mispredictions
    #
    # XXX should prolly group by pathophys
    mispred = preds_df[preds_df.y != preds_df.pred]
    mispred_idxs = []
    for pt, pt_df in mispred.groupby('patient'):
        # get sequence idx first
        idxs = gt[(gt.patient == pt) & (gt.hour.isin(pt_df.hour))].index
        mispred_idxs.extend(idxs)
    correct_idxs = gt.index.difference(mispred_idxs)

    # now can analyze
    #
    # consider analyzing by patient
    mispred_data = []
    correct_data = []
    for i in mispred_idxs:
        # -4 is mean and -3 is median
        mispred_data.append(dataset.all_sequences[i][-3])

    for i in correct_idxs:
        correct_data.append(dataset.all_sequences[i][-3])

    mispred_data = np.array(mispred_data)
    correct_data = np.array(correct_data)
    for i in range(0, 9):
        result = ks_2samp(correct_data[:, i], mispred_data[:, i])
        print("feature: {}, kstest: {}".format(feature_mapping[i], result.pvalue))
    # so for the first round we didnt see anything. just overlapping
    # distributions, which isnt really that surprising honestly given
    # that we lumped both pathos together.
    #
    # Running ks test without bootstrap tells us we can reject null hypothesis they
    # came from same distribution
    #
    # After separating by patho it seems that the mispredictions rougly follow
    # the distribution of data.
    for i in range(0, 9):
        boot = np.random.choice(mispred_data[:, i], size=len(correct_data), replace=True)
        mask_c = np.isnan(correct_data[:, i])
        mask_b = np.isnan(boot)
        # max must be greater than min problem is caused by nans in the array
        plt.hist(correct_data[:, i][~mask_c], bins=100, alpha=0.5, label='correct')
        plt.hist(boot[~mask_b], bins=100, alpha=0.5, label='mispred')
        plt.title(feature_mapping[i])
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
