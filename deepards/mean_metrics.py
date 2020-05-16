import argparse
from glob import glob
import os
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import torch


def computeMetricsFromPatientResults(df, df_stats):
    epochs = df.epoch_num.unique()
    folds = df.fold_num.unique()
    #df_stats = pd.DataFrame(columns = ['fold','epoch','AUC', 'Accuracy', 'sensitivity', 'specificity', 'precision', 'f1'])
    for fold in folds:
        for epoch in epochs:
            sub_df = df.loc[(df['fold_num'] == fold) & (df['epoch_num'] == epoch)]
            y_pred = sub_df.prediction.tolist()
            y_true = sub_df.patho.tolist()
            y_scores = sub_df.pred_frac.tolist()
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            auc = roc_auc_score(y_true, y_scores)
            try:
                accuracy = round(((tp + tn)/float(tp + tn + fp + fn)),4)
            except ZeroDivisionError:
                accuracy = 0
            try:
                sensitivity = round((tp/float(tp + fn)),4)
            except ZeroDivisionError:
                sensitivity = 0
            try:
                specificity = round((tn/float(tn + fp)),4)
            except ZeroDivisionError:
                specificity = 0
            try:
                precision = round((tp/float(tp + fp)),4)
            except ZeroDivisionError:
                precision = 0
            try:
                f1 = f1 = round(2 *((precision * sensitivity) / float(precision + sensitivity)), 4)
            except ZeroDivisionError:
                f1 = 0
            row = {'fold': fold,'epoch': epoch,'AUC': auc, 'Accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision, 'f1': f1}
            df_stats = df_stats.append(row, ignore_index = True)

    return df_stats


def confidence_score(score, sample_size):
    return np.round((1.96 * np.sqrt(score * (1-score) / sample_size)), 3)


def get_metrics(start_times):
    df_patient_results_list = []
    for time in start_times:
        df = pd.read_pickle("results/{}_patient_results.pkl".format(time))
        df_patient_results_list.append(df)
    df_stats = pd.DataFrame(columns = ['fold','epoch','AUC', 'Accuracy', 'sensitivity', 'specificity', 'precision', 'f1'])
    for df in df_patient_results_list:
        df_stats = computeMetricsFromPatientResults(df, df_stats)

    mean_df_stats = df_stats.groupby(['fold', 'epoch'], as_index = False).mean().round(4)
    mean_df_stats = mean_df_stats.sort_values('AUC', ascending = False).drop_duplicates('fold')
    mean_df_stats = mean_df_stats.sort_values('fold')
    mean_df_stats[['fold', 'epoch']] = mean_df_stats[['fold', 'epoch']].astype(int)
    mean_df_stats = mean_df_stats.reset_index(drop = True)
    mean_df_stats.rename(columns = {'epoch' : 'max_epoch'}, inplace = True)
    return mean_df_stats, df_stats


def _do_fold_graphing(df_stats, metric):
    if len(df_stats.fold.unique()) > 1:
        for k, stats in df_stats.groupby('fold'):
            sns.lineplot(x='epoch', y=metric, data=stats, label='fold {}'.format(int(k)))
    sns.lineplot(x='epoch', y=metric, data=df_stats, label='aggregate_results')
    plt.xticks(np.arange(len(df_stats.epoch.unique())), sorted((df_stats.epoch.unique()+1).astype(int)))
    ax = plt.gca()
    ax.yaxis.set_minor_locator(MultipleLocator(.01))
    plt.legend(loc='lower left')
    plt.grid(axis='both')
    plt.show()


def do_fold_graphing(start_times):
    df_patient_results_list = []
    for time in start_times:
        df = pd.read_pickle("results/{}_patient_results.pkl".format(time))
        df_patient_results_list.append(df)

    df_stats = pd.DataFrame(columns = ['fold','epoch','AUC', 'Accuracy', 'sensitivity', 'specificity', 'precision', 'f1'])
    for df in df_patient_results_list:
        df_stats = computeMetricsFromPatientResults(df, df_stats)

    for metric in ['AUC', 'Accuracy', 'sensitivity', 'specificity']:
        _do_fold_graphing(df_stats, metric)


def get_hyperparams(start_time):
    # search for legacy format
    hyperparam_file = glob('results/*{}*.pth'.format(start_time))
    hyperparams = torch.load(hyperparam_file[0])
    if 'dataset_type' in hyperparams:
        tmp = hyperparams
    else:
        tmp = hyperparams['conf']
    return tmp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment-name', default='main_experiment')
    args = parser.parse_args()

    exp_results = []

    # XXX I think regex is the proper thing to do in the future but it requires a bit
    # more finesse than just using glob and have to work over multiple versions of the
    # file naming. For now can punt on it
    main_experiments = glob('results/{}_*'.format(args.experiment_name))
    unique_experiments = set(['_'.join(exp.split('_')[:-1]) for exp in main_experiments])
    for exp in sorted(unique_experiments):
        start_times = list(set([os.path.splitext(file_.split('_')[-1])[0] for file_ in glob(exp + '*')]))
        mean_df_stats, all_stats = get_metrics(start_times)
        hyperparams = get_hyperparams(start_times[0])
        # get hyperparameter file
        dataset_type = hyperparams['dataset_type']
        network_type = hyperparams['network']
        base_net = hyperparams['base_network']

        exp_results.append([dataset_type, network_type, base_net, mean_df_stats.AUC.mean()])
    exp_results = pd.DataFrame(exp_results, columns=['dataset_type', 'network', 'base_cnn', 'auc'])
    do_fold_graphing(start_times)
    # XXX need to add proper experimental results analysis in addition to the graphing
