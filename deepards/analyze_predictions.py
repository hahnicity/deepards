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
map = {'ards': 1, 'other': 0}
reverse_map = {0: 'other', 1: 'ards'}


def get_results_and_data(args):
    dataset = pd.read_pickle(args.dataset)
    results_files = list(Path(__file__).parent.joinpath('results/').glob(args.experiment_name + '*.pkl'))
    assert len(results_files) == 1
    results = pd.read_pickle(results_files[0])
    gt = dataset._get_all_sequence_ground_truth()
    return results, dataset, gt



def get_data_by_preds(dataset, preds, gt, remove_outliers=True):
    idxs = []
    for pt, pt_df in preds.groupby('patient'):
        # get sequence idx first
        eyes = gt[(gt.patient == pt) & (gt.hour.isin(pt_df.hour))].index
        idxs.extend(eyes)
    data = []
    for i in idxs:
        # -4 is mean and -3 is median
        data.append(dataset.all_sequences[i][-3])
    data = np.array(data)
    # clear nans
    mask_nan = np.any(np.isnan(data), axis=1)
    data = data[~mask_nan]

    if remove_outliers:
        std = data.std(axis=0).reshape([1, 9]).repeat(len(data), axis=0)
        mean = data.mean(axis=0).reshape([1, 9]).repeat(len(data), axis=0)
        min_thresh = mean - 3 * std
        max_thresh = mean + 3 * std
        mask = np.logical_and(data > min_thresh, data < max_thresh)
        mask = np.all(mask, axis=1)
        data = data[mask]
    return data


def plot_ards_and_other_cond_dist(args):
    results, dataset, gt = get_results_and_data(args)
    # analyze mispredictions
    patho_n = map[args.patho]
    preds_df = results.all_pred_to_hour[(results.all_pred_to_hour.epoch == args.epoch) & (results.all_pred_to_hour.y == patho_n)]
    mispred_data = get_data_by_preds(dataset, preds_df[preds_df.y != preds_df.pred], gt)
    correct_data = get_data_by_preds(dataset, preds_df[preds_df.y == preds_df.pred], gt)

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
        plt.subplot(3, 3, i+1)
        boot = np.random.choice(mispred_data[:, i], size=len(correct_data), replace=True)
        mask_c = np.isnan(correct_data[:, i])
        mask_b = np.isnan(boot)
        # max must be greater than min problem is caused by nans in the array
        plt.hist(correct_data[:, i][~mask_c], bins=100, alpha=0.5, label='correct pred')
        plt.hist(boot[~mask_b], bins=100, alpha=0.5, label='predicted {}'.format(reverse_map[(patho_n + 1) % 2]))
        plt.title(feature_mapping[i], fontsize=8)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.legend(fontsize=8)
    plt.suptitle('Conditional Distributions for {} Reads'.format(args.patho.upper()))
    plt.show()


def plot_tp_tn_and_patient(frame, dataset, gt, true_pos_data, true_neg_data, epoch_results, data_type):
    for pt, patient_df in frame.groupby('patient'):
        pt_data = get_data_by_preds(dataset, patient_df, gt)
        prob = epoch_results[epoch_results.patient == pt].iloc[0].pred_frac
        prob = round(prob, 4)
        fig = plt.figure(figsize=(3*8, 3*4))

        for i in range(0, 9):
            fig.add_subplot(3, 3, i+1)
            bootstrapped = np.random.choice(
                pt_data[:, i],
                size=int(min([len(true_pos_data), len(true_neg_data)])/2.0),
                replace=True
            )

            # max must be greater than min problem is caused by nans in the array
            plt.hist(true_pos_data[:, i], bins=100, alpha=0.7, label='true pos', color='c')
            plt.hist(true_neg_data[:, i], bins=100, alpha=0.5, label='true neg', color='orange')
            plt.hist(bootstrapped, bins=100, alpha=0.45, label='{} reads'.format(data_type), color='purple')
            plt.title(feature_mapping[i], fontsize=8)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            plt.legend(fontsize=8)
        plt.suptitle("{}, ground truth: {}, prediction: {},\nARDS pred prob: {}".format(
            pt,
            reverse_map[patient_df.iloc[0].y].upper(),
            reverse_map[patient_df.iloc[0].pred].upper(),
            prob,
        ), fontsize=18)
        plt.savefig("{}.png".format(pt))
        plt.close()


def misclassified_pt_plotting(args):
    results, dataset, gt = get_results_and_data(args)
    patho_n = map[args.patho]
    true_pos = results.all_pred_to_hour[
        (results.all_pred_to_hour.epoch == args.epoch) &
        (results.all_pred_to_hour.y == 1) &
        (results.all_pred_to_hour.pred == 1)
    ]
    true_neg = results.all_pred_to_hour[
        (results.all_pred_to_hour.epoch == args.epoch) &
        (results.all_pred_to_hour.y == 0) &
        (results.all_pred_to_hour.pred == 0)
    ]
    true_pos_data = get_data_by_preds(dataset, true_pos, gt)
    true_neg_data = get_data_by_preds(dataset, true_neg, gt)

    epoch_results = results.results[results.results.epoch_num == args.epoch]
    false_pos_pt = epoch_results[
        (epoch_results.patho == 0) &
        (epoch_results.prediction == 1)
    ].patient.unique()
    false_neg_pt = epoch_results[
        (epoch_results.patho == 1) &
        (epoch_results.prediction == 0)
    ].patient.unique()
    false_pos = results.all_pred_to_hour[
        (results.all_pred_to_hour.epoch == args.epoch) &
        (results.all_pred_to_hour.patient.isin(false_pos_pt)) &
        (results.all_pred_to_hour.pred == 1)
    ]
    false_neg = results.all_pred_to_hour[
        (results.all_pred_to_hour.epoch == args.epoch) &
        (results.all_pred_to_hour.patient.isin(false_neg_pt)) &
        (results.all_pred_to_hour.pred == 0)
    ]
    plot_tp_tn_and_patient(false_pos, dataset, gt, true_pos_data, true_neg_data, epoch_results, 'false pos')
    plot_tp_tn_and_patient(false_neg, dataset, gt, true_pos_data, true_neg_data, epoch_results, 'false neg')


def main():
    parser = ArgumentParser()
    parser.add_argument('experiment_name')
    parser.add_argument('dataset')
    parser.add_argument('-e', '--epoch', type=int, default=4, help='epoch to analyze')
    parser.add_argument('--patho', choices=['ards', 'other'], required=True)
    args = parser.parse_args()

    misclassified_pt_plotting(args)



if __name__ == "__main__":
    main()
