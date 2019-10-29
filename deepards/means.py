from argparse import ArgumentParser
from glob import glob
import os

import pandas as pd
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

def getMeanMetrics(start_times):
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
    return mean_df_stats


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('search_glob')
    parser.add_argument('split_by')
    args = parser.parse_args()

    exp_results = []
    datasets = ['unpadded_centered_sequences', 'unpadded_sequences', 'padded_breath_by_breath', 'unpadded_downsampled_sequences']
    networks = ['cnn_lstm', 'cnn_single_breath_linear', 'cnn_transformer', 'lstm_only']
    base_networks = ['se_resnet18', 'resnet18', 'densenet18', 'vgg11']

    main_experiments = glob('results/{}*'.format(args.search_glob))
    split_dict = {}
    for file_ in main_experiments:
        exp = torch.load(file_)
        start_time = file_.split('_')[-1][:-4]
        val = exp[args.split_by]
        if val in split_dict:
            split_dict[val].append(start_time)
        else:
            split_dict[val] = [start_time]

    all_exp_data = []
    for exp in sorted(split_dict.keys()):
        start_times = split_dict[exp]
        mean_df_stats = getMeanMetrics(start_times)
        exp_results.append([exp, mean_df_stats.AUC.mean(), mean_df_stats.f1.mean()])
        all_exp_data.append(mean_df_stats)

    exp_results = pd.DataFrame(exp_results, columns=[args.split_by, 'auc', 'f1'])
    import IPython; IPython.embed()
