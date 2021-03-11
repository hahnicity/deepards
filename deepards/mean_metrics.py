import argparse
from glob import glob
import os
import re
from warnings import warn
import zipfile

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import torch
import yaml


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
        df = pd.read_pickle(os.path.join(os.path.dirname(__file__), "results", "{}_patient_results.pkl".format(time)))
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


def _graph_aggregate(df_stats, metric, label):
    sns.lineplot(x='epoch', y=metric, data=df_stats, label=label)
    plt.xticks(np.arange(len(df_stats.epoch.unique())), sorted((df_stats.epoch.unique()+1).astype(int)))
    ax = plt.gca()
    ax.yaxis.set_minor_locator(MultipleLocator(.01))
    plt.legend(loc='lower left')
    plt.grid(axis='both')


def _do_fold_graphing(df_stats, metric, label='aggregate results'):
    if len(df_stats.fold.unique()) > 1:
        for k, stats in df_stats.groupby('fold'):
            sns.lineplot(x='epoch', y=metric, data=stats, label='fold {}'.format(int(k)))
    _graph_aggregate(df_stats, metric, label)


def do_fold_graphing(start_times, only_aggregate):
    df_patient_results_list = []
    for time in start_times:
        df = pd.read_pickle("results/{}_patient_results.pkl".format(time))
        df_patient_results_list.append(df)

    df_stats = pd.DataFrame(columns = ['fold','epoch','AUC', 'Accuracy', 'sensitivity', 'specificity', 'precision', 'f1'])
    for df in df_patient_results_list:
        df_stats = computeMetricsFromPatientResults(df, df_stats)

    for metric in ['Accuracy', 'f1', 'sensitivity', 'specificity', 'AUC']:

        if not only_aggregate:
            _do_fold_graphing(df_stats, metric)
        else:
            _graph_aggregate(df_stats, metric, 'aggregate results')
        plt.show()


def get_hyperparams(start_time):
    # search for legacy format
    hyperparam_file = glob('results/*{}*.pth'.format(start_time))
    try:
        hyperparams = torch.load(hyperparam_file[0])
    except RuntimeError:
        print('torch saved hyperparams as zip!')
        with zipfile.ZipFile(hyperparam_file[0], 'r') as zip_ref:
            zip_ref.extractall('tmp_zip/')
            hyperparams = pd.read_pickle('tmp_zip/archive/data.pkl')
    if 'dataset_type' in hyperparams:
        tmp = hyperparams
    else:
        tmp = hyperparams['conf']
    return tmp


def get_experiment_id(experiment_file):
    # This function is basically just one large sanity check

    # v1
    if re.search('_(\d{10}).pth', experiment_file):
        return os.path.splitext(experiment_file)[0].split('_')[-1]
    # v2
    elif re.search(r'_(\w{8}\-\w{4}\-\w{4}\-\w{4}\-\w{12}).pth', experiment_file):
        return os.path.splitext(experiment_file)[0].split('_')[-1]
    # probably some kind of error between v1 and v2
    else:
        warn('File {} did not match any versioning spec'.format(experiment_file))


def find_matching_experiments(experiment_name):
    # first pass
    first_pass = glob(os.path.join(os.path.dirname(__file__), 'results/{}_*'.format(experiment_name)))
    experiment_ids = []
    for file in first_pass:
        if '{}_results'.format(experiment_name) in file:
            continue
        experiment_id = get_experiment_id(file)
        if not experiment_id:
            continue
        candidate = os.path.basename(file).replace('_'+experiment_id+'.pth', '')
        if candidate == experiment_name:
            experiment_ids.append(experiment_id)
    return experiment_ids


def analyze_similar_dissimilar_experiments(sim_dissim_file, expr_ids):

    with open(os.path.join(os.path.dirname(__file__), sim_dissim_file)) as sds:
        conf = yaml.load(sds, Loader=yaml.FullLoader)

    df_patient_results_list = []
    for id_ in expr_ids:
        df = pd.read_pickle("results/{}_patient_results.pkl".format(id_))
        df_patient_results_list.append(df)

    similar_pts = conf['similar']
    dissimilar_pts = conf['dissimilar']

    df_stats_similar = pd.DataFrame(columns = ['fold','epoch','AUC', 'Accuracy', 'sensitivity', 'specificity', 'precision', 'f1'])
    df_stats_dissimilar = pd.DataFrame(columns = ['fold','epoch','AUC', 'Accuracy', 'sensitivity', 'specificity', 'precision', 'f1'])
    for df in df_patient_results_list:
        similar_results = df[df.patient.isin(similar_pts)]
        dissimilar_results = df[df.patient.isin(dissimilar_pts)]
        df_stats_similar = computeMetricsFromPatientResults(similar_results, df_stats_similar)
        df_stats_dissimilar = computeMetricsFromPatientResults(dissimilar_results, df_stats_dissimilar)
    for metric in ['AUC', 'Accuracy']:
        _do_fold_graphing(df_stats_similar, metric, label='Similar pt {}'.format(metric))
        _do_fold_graphing(df_stats_dissimilar, metric, label='Dissimilar pt {}'.format(metric))
        ax = plt.gca()
        line1 = ax.lines[0].get_data()[1]
        line2 = ax.lines[1].get_data()[1]
        f1 = 2 * (line1*line2) / (line1+line2)
        plt.plot(f1, label='harmonic mean')
        max_x = np.argmax(f1)
        sim = round(line1[max_x], 2)
        dissim = round(line2[max_x], 2)
        plt.annotate('max sim: {}, dissim: {}'.format(sim, dissim), xy=(max_x, f1[max_x]))
        plt.legend()
        plt.show()


def one_to_many_shot_analysis(experiment_name, start_times):
    cmap = sns.color_palette('deep', as_cmap=True)

    model_list = []
    # this will have format:
    #
    # patient,model idx,n,patho,pred_frac,pred
    one_to_many_results = []
    hour_range_results = []
    # so 40 frames is approx \eq to 1hr of vent data so 960 frames if eq to 1 day
    #max_n = 960

    for time in start_times:
        try:
            # apparently I can't recover py2 results from py3. So I'd need to run
            # this script in py2 if my results were done in py2
            df = pd.read_pickle("results/{}_results_{}.pkl".format(experiment_name, time))
        except FileNotFoundError:
            raise FileNotFoundError('Unable to find all results record for experiment: {}, id: {}. Is this an old experiment?'.format(experiment_name, time))

        model_list.append(df)
    gb = model_list[0].all_pred_to_hour.groupby(['patient', 'epoch'])
    max_n = gb.agg('count').y.max()
    # need to average together last epoch across all foldsand then

    # what should my primary metric be?? AUC/Accuracy? why not both? probably should
    # average the results over multiple folds.
    minute_interval = 60
    xrange = list(range(0, 60*24, minute_interval))
    # there are 40.17 breath windows per hour. so the 964.3 = 40.17*24
    xticks_labels = [i for i in range(1, 25)]
    xticks_x = list(range(1, 60*24, 60))
    #x_ticks = np.arange(0, max_n+interval, interval)
    hour_interval = 20
    hour_interval_range = list(range(hour_interval, 25))

    for model_idx, results in enumerate(model_list):
        final_epoch = results.all_pred_to_hour.epoch.max()
        last_epoch_preds = results.all_pred_to_hour[results.all_pred_to_hour.epoch == final_epoch]
        if 'hour_x' not in last_epoch_preds.columns:
            hour_col = 'hour'
        else:
            hour_col = 'hour_x'
        last_epoch_preds = last_epoch_preds.sort_values(by=['patient', hour_col])
        for patient, pt_df in last_epoch_preds.groupby('patient'):
            last_epoch_preds.loc[pt_df.index, 'rel_hour'] = last_epoch_preds.loc[pt_df.index, hour_col] - last_epoch_preds.loc[pt_df.index, hour_col].iloc[0]

        for start in xrange:
            range_preds = last_epoch_preds[
                (last_epoch_preds['rel_hour'] <= (start+minute_interval)/60)
            ]
            for pt_id, pt_df in range_preds.groupby('patient'):
                one_to_many_results.append([
                    pt_id,
                    model_idx,
                    pt_df.fold.iloc[0],
                    (start+minute_interval)/60,
                    pt_df.iloc[0].y,
                    pt_df.pred.sum() / len(pt_df),
                    int((pt_df.pred.sum() / len(pt_df)) >= .5),
                ])

        for start in range(0, 24-hour_interval+1):
            range_preds = last_epoch_preds[
                (last_epoch_preds['rel_hour'] >= start) &
                (last_epoch_preds['rel_hour'] <= start+hour_interval)
            ]
            for pt_id, pt_df in range_preds.groupby('patient'):
                hour_range_results.append([
                    pt_id,
                    model_idx,
                    pt_df.fold.iloc[0],
                    start+hour_interval,
                    pt_df.iloc[0].y,
                    pt_df.pred.sum() / len(pt_df),
                    int((pt_df.pred.sum() / len(pt_df)) >= .5),
                ])

    one_to_many_results = pd.DataFrame(one_to_many_results, columns=['patient', 'model', 'fold', 'n', 'patho', 'pred_frac', 'pred'])
    hour_range_results = pd.DataFrame(hour_range_results, columns=['patient', 'model', 'fold', 'end_hour', 'patho', 'pred_frac', 'pred'])

    # this will be formatted like
    #
    # model_n,n,sen,spec,auc
    pred_stats = []
    hour_range_pred_stats = []
    #XXX ok slight problem with the hour method because looking at the model
    # from time of first prediction is not the same as looking at model from
    # the hour of the prediction.
    #
    # I think what chao is looking for is relative hour, not absolute hour
    # whereas I'm using absolute hour. I think absolute hour is useful in
    # medical context, but not so much from computational.

    for model_n, model_df in one_to_many_results.groupby(['model', 'fold']):
        for n, n_df in model_df.groupby('n'):
            # just do stats for ARDS right now
            tps = len(n_df[(n_df.patho == 1) & (n_df.pred == 1)])
            tns = len(n_df[(n_df.patho != 1) & (n_df.pred != 1)])
            fps = len(n_df[(n_df.patho != 1) & (n_df.pred == 1)])
            fns = len(n_df[(n_df.patho == 1) & (n_df.pred != 1)])

            if tns == 0 or tps == 0:
                import IPython; IPython.embed()
                continue
            sen = tps / float(tps+fns)
            spec = tns / float(tns+fps)
            acc = (tps + tns) / (tps+tns+fps+fns)
            auc = roc_auc_score(n_df.patho, n_df.pred_frac)
            pred_stats.append([n_df.model.iloc[0], n_df.fold.iloc[0], n, sen, spec, acc, auc])

    for model_n, model_df in hour_range_results.groupby(['model', 'fold']):
        for n, n_df in model_df.groupby('end_hour'):
            # just do stats for ARDS right now
            tps = len(n_df[(n_df.patho == 1) & (n_df.pred == 1)])
            tns = len(n_df[(n_df.patho != 1) & (n_df.pred != 1)])
            fps = len(n_df[(n_df.patho != 1) & (n_df.pred == 1)])
            fns = len(n_df[(n_df.patho == 1) & (n_df.pred != 1)])
            sen = tps / float(tps+fns)
            spec = tns / float(tns+fps)
            acc = (tps + tns) / (tps+tns+fps+fns)
            auc = roc_auc_score(n_df.patho, n_df.pred_frac)
            hour_range_pred_stats.append([n_df.model.iloc[0], n_df.fold.iloc[0], n, sen, spec, acc, auc])

    # XXX alright this code isnt matching my experimental results whatsoever. Which
    # doesnt make sense because it just doesnt. theres no way that these results
    # should be off the main experimental results as our number of obs approaches
    # the max number of obs for each patient.

    pred_stats = pd.DataFrame(pred_stats, columns=['model_n', 'fold', 'n', 'sen', 'spec', 'acc', 'auc'])
    hour_range_pred_stats = pd.DataFrame(hour_range_pred_stats, columns=['model_n', 'fold', 'end_hour', 'sen', 'spec', 'acc', 'auc'])

    templ = pred_stats[['n', 'model_n', 'sen', 'spec', 'auc', 'acc', 'fold']].groupby(['n', 'fold'])
    # first mean is mean by model, next mean is mean by fold
    mean_sen = templ.sen.mean().groupby('n').mean()
    mean_spec = templ.spec.mean().groupby('n').mean()
    mean_auc = templ.auc.mean().groupby('n').mean()
    mean_acc = templ.acc.mean().groupby('n').mean()

    plt.plot(xrange, mean_auc, label='AUC', color=cmap[2], marker='o', linestyle='dotted')
    plt.plot(xrange, mean_sen, label='Sensitivity', color=cmap[0], marker='^', linestyle='dashed')
    plt.plot(xrange, mean_acc, label='Accuracy', color=cmap[3], marker='s', linestyle='dotted')
    plt.plot(xrange, mean_spec, label='Specificity', color=cmap[1], marker='D', linestyle='dashed')
    plt.xticks(xticks_x, xticks_labels)
    plt.xlim([-10, xticks_x[-1]+10])
    yticks = np.arange(.5, 1.01, .05)
    plt.ylim((yticks[0]-0.05, yticks[-1]+.025))
    plt.yticks(yticks)
    plt.grid(axis='y', alpha=.7)
    plt.xlabel('Hour', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
    plt.show()
    #plt.savefig('../img/one_to_many_plotting.png', dpi=200)

    templ = hour_range_pred_stats[['end_hour', 'model_n', 'sen', 'spec', 'auc', 'acc', 'fold']].groupby(['end_hour', 'fold'])
    # first mean is mean by model, next mean is mean by fold
    mean_sen = templ.sen.mean().groupby('end_hour').mean()
    mean_spec = templ.spec.mean().groupby('end_hour').mean()
    mean_auc = templ.auc.mean().groupby('end_hour').mean()
    mean_acc = templ.acc.mean().groupby('end_hour').mean()

    plt.plot(hour_interval_range, mean_auc, label='AUC', color=cmap[2])
    plt.plot(hour_interval_range, mean_sen, label='Sensitivity', color=cmap[0])
    plt.plot(hour_interval_range, mean_acc, label='Accuracy', color=cmap[3])
    plt.plot(hour_interval_range, mean_spec, label='Specificity', color=cmap[1])
    #plt.xticks(xticks_x, xticks_labels)
    plt.xlim([hour_interval_range[0]-0.2, hour_interval_range[-1]+0.2])
    plt.grid(axis='y', alpha=.7)
    plt.xlabel('Hour', fontsize=12)
    plt.legend(fontsize=12)
    plt.show()
    #plt.savefig('../img/one_to_many_plotting.png', dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment-name', default='main_experiment')
    parser.add_argument('-sds', '--sim-dissim-file', help='If we are comparing between similar and dissimilar patients in the testing cohort, then supply a yaml file which specifies which patients are similar and which are dissimilar')
    parser.add_argument('--only-aggregate', action='store_true', help='only graph aggregate results')
    args = parser.parse_args()

    exp_results = []

    unique_experiments = find_matching_experiments(args.experiment_name)
    mean_df_stats, all_stats = get_metrics(unique_experiments)
    hyperparams = get_hyperparams(unique_experiments[0])
    # get hyperparameter file
    dataset_type = hyperparams['dataset_type']
    network_type = hyperparams['network']
    base_net = hyperparams['base_network']

    exp_results.append([dataset_type, network_type, base_net, mean_df_stats.AUC.mean()])

    exp_results = pd.DataFrame(exp_results, columns=['dataset_type', 'network', 'base_cnn', 'auc'])
    if args.sim_dissim_file:
        analyze_similar_dissimilar_experiments(args.sim_dissim_file, unique_experiments)
    else:
        do_fold_graphing(unique_experiments, args.only_aggregate)
        pass
    one_to_many_shot_analysis(args.experiment_name, unique_experiments)
