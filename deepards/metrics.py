from copy import copy
from math import ceil, sqrt
import os
import csv
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score
import torch

try:
    import deepards.dtw_lib as dtw_lib
except ImportError:
    from mock import Mock
    dtw_lib = Mock()

filename = './Data/data.csv'


def get_fns_idx(actual, predictions, label):
    pos = actual[actual == label]
    pos_loc = predictions.loc[pos.index]
    return pos_loc[pos_loc != label].index


def get_fns(actual, predictions, label):
    pos = actual[actual == label]
    pos_loc = predictions.loc[pos.index]
    return len(pos_loc[pos_loc != label])


def get_tns(actual, predictions, label):
    neg = actual[actual != label]
    neg_loc = predictions.loc[neg.index]
    return len(neg_loc[neg_loc != label])


def get_fps_idx(actual, predictions, label):
    neg = actual[actual != label]
    neg_loc = predictions.loc[neg.index]
    return neg_loc[neg_loc == label].index


def get_fps_full_rows(actual, predictions, label, filename):
    idx = get_fps_idx(actual, predictions, label)
    full_df = read_pickle(filename)
    return full_df.loc[idx]


def get_fps(actual, predictions, label):
    neg = actual[actual != label]
    neg_loc = predictions.loc[neg.index]
    return len(neg_loc[neg_loc == label])


def get_tps(actual, predictions, label):
    pos = actual[actual == label]
    pos_loc = predictions.loc[pos.index]
    return len(pos_loc[pos_loc == label])


def false_positive_rate(actual, predictions, label):
    fp = get_fps(actual, predictions, label)
    tn = get_tns(actual, predictions, label)
    if fp == 0 and tn == 0:
        return 0
    else:
        return round(float(fp) / (fp + tn), 4)


def specificity(actual, predictions, label):
    """
    Also known as the true negative rate
    """
    fp = get_fps(actual, predictions, label)
    tn = get_tns(actual, predictions, label)
    if fp == 0 and tn == 0:
        return 1
    else:
        return round(float(tn) / (tn + fp), 4)


def sensitivity(actual, predictions, label):
    """
    Also known as recall
    """
    tps = get_tps(actual, predictions, label)
    fns = get_fns(actual, predictions, label)
    if tps == 0 and fns == 0:
        return nan
    return tps / (tps+fns)


def janky_roc(y_true, preds):
    # false positive rate
    fpr = []
    # true positive rate
    tpr = []
    # Iterate thresholds from 0.0, 0.01, ... 1.0
    thresholds = np.arange(0.0, 1.0001, .001)
    # get number of positive and negative examples in the dataset
    P = sum(y_true)
    N = len(y_true) - P

    # iterate through all thresholds and determine fraction of true positives
    # and false positives found at this threshold
    for thresh in thresholds:
        FP=0
        TP=0
        for i in range(len(preds)):
            if (preds[i] > thresh):
                if y_true[i] == 1:
                    TP = TP + 1
                if y_true[i] == 0:
                    FP = FP + 1
        fpr.append(FP/float(N))
        tpr.append(TP/float(P))

    return tpr, fpr, thresholds


class Meter(object):
    """
    A little helper class which keeps track of statistics during an epoch.
    """
    def __init__(self, name, cumulative=False):
        self.cumulative = cumulative
        if type(name) == str:
            name = (name,)
        self.name = name
        self.values = torch.FloatTensor([])
        self._total = torch.zeros(len(self.name))
        self._last_value = torch.zeros(len(self.name))
        self._count = 0.0

    def peek(self):
        return self.values[-1]

    def update(self, data, n=1):
        self._count = self._count + n
        if isinstance(data, torch.autograd.Variable):
            self._last_value.copy_(data.data)
            self.values = torch.cat((self.values, data.data.cpu().view(1)), 0)
        elif isinstance(data, torch.Tensor):
            self._last_value.copy_(data)
            self.values = torch.cat((self.values, data.cpu().view(1)), 0)
        else:
            self._last_value = torch.FloatTensor([data])
            self.values = torch.cat((self.values, self._last_value), 0)
        self._total += self._last_value

    def value(self):
        if self.cumulative:
            return self._total
        else:
            return self._total / self._count

    def __repr__(self):
        return '\t'.join(['%s: %.5f (%.3f)' % (n, lv, v)
            for n, lv, v in zip(self.name, self._last_value, self.value())])


class SavedObjects(object):
    def __init__(self, file_dir, file_suffix):
        """
        Because saving objects after a network finishes training is tricky,
        we can just use this helper class to keep track of the objects we
        want to save. Afterwards save everything to file. Example:
            saved_objs = SavedObjects('/path/to/results', 'file_suffix')
            model = ResNet50()
            training_loss = []
            saved_objs.register(model, 'resnet50_weights', True)
            saved_objs.register(training_loss, 'train_loss', False)
            ... Do training stuff
            ... Do testing stuff
            saved_objs.save_all()
        """
        self.saved_objects = {}
        self.file_dir = file_dir
        self.file_suffix = file_suffix

    def register(self, obj, file_prefix, save_weights):
        """
        :param obj: object you want to save later
        :param file_prefix: prefix of file to save eg. "model_weights"
        :param save_weights: True if its a nn model and we only want to save weights.
                             False otherwise. We do this so we only save model weights
                             and not the entire model
        """
        self.saved_objects[file_prefix] = (obj, save_weights)

    def save(self, name, timestamp="", dir_override=None):
        obj, save_weights = self.saved_objects[name]
        joined = [name, self.file_suffix, timestamp] if timestamp else [name, self.file_suffix]
        filename = "_".join(joined) + ".pt"
        if not dir_override:
            filepath = os.path.join(self.file_dir, filename)
        else:
            filepath = os.path.join(dir_override, filename)

        if save_weights:
            torch.save(obj.state_dict(), filepath)
        else:
            torch.save(obj, filepath)

    def save_all(self, timestamp=""):
        for name in self.saved_objects:
            self.save(name, timestamp=timestamp)


class Reporting(SavedObjects):
    def __init__(self, file_dir, file_suffix):
        super(Reporting, self).__init__(file_dir, file_suffix)
        self.meters = {
        }
        for name, meter in self.meters.items():
            self.register(meter, name, False)

    def does_meter_exist(self, name):
        return name in self.meters

    def get_meter(self, name):
        """
        :param name: meter name
        """
        return self.meters[name]

    def new_meter(self, name):
        """
        Create a new quantitative meter that will be registered to be saved
        :param name: meter name
        """
        self.meters[name] = Meter(name)
        self.register(self.meters[name], name, False)

    def new_unsaved_meter(self, name):
        """
        Create a new meter that will not be saved
        :param name: meter name
        """
        self.meters[name] = Meter(name)

    def update(self, meter, val):
        self.meters[meter].update(val)


class DeepARDSResults(object):
    def __init__(self, start_time, experiment_name, **hyperparams):
        self.pathos = {0: 'OTHER', 1: 'ARDS'}

        results_cols = ["patient", "patho"]
        for n, patho in self.pathos.items():
            results_cols.extend([
                "{}_tps".format(patho), "{}_fps".format(patho),
                "{}_tns".format(patho), "{}_fns".format(patho),
                "{}_votes".format(patho),
            ])
        results_cols += ["prediction", 'pred_frac', 'epoch_num', 'fold_num']
        # self.results is meant to be a high level dataframe of aggregated statistics
        # from our model.
        self.results = pd.DataFrame([], columns=results_cols)
        reporting_suffix = 'deepards_start_{}'.format(start_time)
        self.results_dir = os.path.join(os.path.dirname(__file__), 'results/')
        self.reporting = Reporting(self.results_dir, reporting_suffix)
        self.hyperparams = hyperparams
        self.hyperparams['start_time'] = start_time
        self.uuid_name = uuid.uuid4()
        self.experiment_save_filename = "{}_{}.pth".format(experiment_name, self.uuid_name) if experiment_name else str(self.uuid_name) + ".pth"
        self.results_save_filename = "{}_results_{}.pkl".format(experiment_name, self.uuid_name) if experiment_name else str(self.uuid_name) + ".pth"
        self.all_pred_to_hour = pd.DataFrame([], columns=['pred', 'hour', 'patient', 'y', 'epoch', 'fold'])

    def aggregate_classification_results(self):
        """
        Aggregate final results for all patients into a friendly data frame
        """
        aggregate_stats = None
        for fold_num in self.results.fold_num.unique():
            for epoch_num in self.results.epoch_num.unique():
                if aggregate_stats is None:
                    aggregate_stats = self._aggregate_specific_results(self.results[(self.results.epoch_num == epoch_num) & (self.results.fold_num == fold_num)], fold_num, epoch_num)
                else:
                    aggregate_stats = aggregate_stats.append(self._aggregate_specific_results(self.results[
                        (self.results.epoch_num == epoch_num) &
                        (self.results.fold_num == fold_num)
                    ], fold_num, epoch_num))
        aggregate_stats.index = range(len(aggregate_stats))

        self._print_specific_results_report(aggregate_stats)
        self.results.to_pickle('results/{}_patient_results.pkl'.format(self.uuid_name))
        aggregate_stats.to_pickle('results/{}_aggregate_results.pkl'.format(self.uuid_name))
        self.save_maximals('results/{}_maximal_results.pkl'.format(self.uuid_name), aggregate_stats)

    def save_maximals(self, output_filename, aggregate_stats):
        maximals = None
        table = PrettyTable()
        table.field_names = ['Patho', 'Accuracy', 'Recall', 'Precision', 'AUC', 'F1', 'Fold', 'Epoch']

        for fold_num in aggregate_stats.fold_num.unique():
            fold_stats = aggregate_stats[aggregate_stats.fold_num == fold_num]
            max_auc_idx = fold_stats.auc.idxmax()
            epoch_max = aggregate_stats.loc[max_auc_idx].epoch_num
            epoch_maxes = fold_stats[fold_stats.epoch_num == epoch_max]
            if maximals is None:
                maximals = epoch_maxes
            else:
                maximals = maximals.append(epoch_maxes)

            for idx, row in epoch_maxes.iterrows():
                table.add_row([row.patho, row.accuracy, row.sensitivity, row.precision, row.auc, row.f1, row.fold_num, row.epoch_num])
        maximals.to_pickle(output_filename)
        print('---- Max Stats ----')
        print(table)

    def _aggregate_specific_results(self, patient_results, fold_num, epoch_num):
        aggregate_stats = []
        for n, patho in self.pathos.items():
            tps = float(len(patient_results[(patient_results.patho == n) & (patient_results.prediction == n)]))
            tns = float(len(patient_results[(patient_results.patho != n) & (patient_results.prediction != n)]))
            fps = float(len(patient_results[(patient_results.patho != n) & (patient_results.prediction == n)]))
            fns = float(len(patient_results[(patient_results.patho == n) & (patient_results.prediction != n)]))
            accuracy = round((tps+tns) / (tps+tns+fps+fns), 4)
            try:
                sensitivity = round(tps / (tps+fns), 4)
            except ZeroDivisionError:
                sensitivity = 0
            try:
                specificity = round(tns / (tns+fps), 4)
            except ZeroDivisionError:
                specificity = 0
            try:
                precision = round(tps / (tps+fps), 4)
            except ZeroDivisionError:  # Can happen when no predictions for cls are made
                precision = 0
            if len(self.pathos) > 2:
                auc = np.nan
            elif len(self.pathos) == 2:
                auc = round(roc_auc_score(patient_results.patho.tolist(), patient_results.pred_frac.tolist()), 4)

            try:
                f1 = round(2 * ((precision * sensitivity) / (precision + sensitivity)), 4)
            except ZeroDivisionError:
                f1 = 0
            aggregate_stats.append([patho, tps, tns, fps, fns, accuracy, sensitivity, specificity, precision, auc, f1, fold_num, epoch_num])

        return pd.DataFrame(
            aggregate_stats,
            columns=['patho', 'tps', 'tns', 'fps', 'fns', 'accuracy', 'sensitivity', 'specificity', 'precision', 'auc', 'f1', 'fold_num', 'epoch_num']
        )

    def _print_specific_results_report(self, stats_frame):
        table = PrettyTable()
        table.field_names = ['Patho', 'Accuracy', 'Recall', 'Precision', 'AUC', 'F1', 'Fold', 'Epoch']
        for idx, row in stats_frame.iterrows():
            table.add_row([row.patho, row.accuracy, row.sensitivity, row.precision, row.auc, row.f1, row.fold_num, row.epoch_num])
        print('---- Patient-level stats ----')
        print(table)

    def update_loss(self, fold_num, loss):
        self.update_meter('loss', fold_num, loss)

    def update_accuracy(self, fold_num, accuracy):
        self.update_meter('test_accuracy', fold_num, accuracy)

    def update_r2(self, fold_num, r2):
        self.update_meter('test_r2', fold_num, r2)

    def update_meter(self, metric_name, fold_num, val):
        meter_name = '{}_fold_{}'.format(metric_name, fold_num)
        if not self.reporting.does_meter_exist(meter_name):
            self.reporting.new_meter(meter_name)
        self.reporting.update(meter_name, val)

    def update_epoch_meter(self, metric_name, epoch_num, val):
        meter_name = '{}_epoch_{}'.format(metric_name, epoch_num)
        if not self.reporting.does_meter_exist(meter_name):
            self.reporting.new_meter(meter_name)
        self.reporting.update(meter_name, val)

    def get_meter(self, metric_name, fold_num):
        meter_name = '{}_fold_{}'.format(metric_name, fold_num)
        if not self.reporting.does_meter_exist(meter_name):
            self.reporting.new_meter(meter_name)
        return self.reporting.meters[meter_name]

    def print_meter_results(self, metric_name, fold_num):
        meter_name = '{}_fold_{}'.format(metric_name, fold_num)
        print(self.reporting.meters[meter_name])

    def print_epoch_meter_results(self, metric_name, epoch_num):
        meter_name = '{}_epoch_{}'.format(metric_name, epoch_num)
        print(self.reporting.meters[meter_name])

    def process_pred_to_hour_for_dtw(self, test_dataset):
        """
        So the whole point of this function is to ensure that we have a timestamp
        associated with every single breath. It may not have this in cases where
        we are performing sub-batch-level predictions.
        """
        copy_pred_to_hour = self.pred_to_hour_frame.copy()
        gt = test_dataset.get_ground_truth_df()
        idx_pt = gt.patient.unique()[0]
        pt_preds = len(self.pred_to_hour_frame[self.pred_to_hour_frame.patient == idx_pt])
        pt_gt = len(gt[gt.patient == idx_pt])

        # in this case we aare doing breath window preds and must duplicate our indexing
        if pt_gt == pt_preds:
            repeat_n = test_dataset.all_sequences[0][1].shape[0]
            # expand out predictions by the sub-batch size
            copy_pred_to_hour = copy_pred_to_hour.loc[copy_pred_to_hour.index.repeat(repeat_n)]

            hour_arr = [None] * len(copy_pred_to_hour)
            gt_index = gt.index
            for pt, pt_rows in gt.groupby('patient'):
                pt_gt = gt[gt.patient == pt]
                for idx in gt_index:
                    # -1 is the index where hours are located
                    hour_arr[idx*repeat_n:idx*repeat_n+repeat_n] = test_dataset.all_sequences[idx][-1]
            copy_pred_to_hour['hour'] = hour_arr

        return copy_pred_to_hour

    def perform_dtw_preprocessing(self, test_dataset, dtw_cache_dir):
        copy_pred_to_hour = self.process_pred_to_hour_for_dtw(test_dataset)

        for _, pt_rows in self.pred_to_hour_frame.groupby('patient'):
            pt = pt_rows.iloc[0].patient
            dtw_scores = dtw_lib.analyze_patient(pt, test_dataset, dtw_cache_dir, copy_pred_to_hour)
            copy_pred_to_hour.loc[copy_pred_to_hour.patient==pt, 'dtw'] = dtw_scores.sort_index().dtw

        copy_pred_to_hour.to_pickle(os.path.join(dtw_cache_dir, 'dtw_{}_nb{}_{}_predictions.pkl'.format(
            test_dataset.dataset_type,
            test_dataset.n_sub_batches,
            'kfold' if test_dataset.kfold_num else 'holdout',
        )))

    def perform_hourly_patient_plot_with_dtw(self, test_dataset, dtw_cache_dir):
        for _, pt_rows in self.pred_to_hour_frame.groupby('patient'):
            self.plot_disease_evolution(pt_rows)
            pt = pt_rows.iloc[0].patient
            # can provide None because we will just be pulling from cache
            dtw = dtw_lib.analyze_patient(pt, test_dataset, dtw_cache_dir, None)
            dtw = dtw.sort_values(by='hour')
            self.plot_dtw_patient_data(dtw, True, 2, True)
            plt.show()

    def perform_hourly_patient_plot(self):
        for i, rows in self.pred_to_hour_frame.groupby('patient'):
            self.plot_disease_evolution(rows)
            plt.show()

    def plot_dtw_patient_data(self, dtw_data, set_label, size, xy_visible, y_max=None):
        """
        Plot DTW for an individual patient

        :param pt_rows: Rows grouped by patient from the dataframe received from
                        self.results.get_all_hourly_preds
        """
        y_max = dtw_data.dtw.max() + 1 if not y_max else y_max
        ax2 = plt.gca().twinx()
        ax2.scatter(dtw_data.hour, dtw_data.dtw, s=size, label='DTW', c='#663a3e')
        ax2.set_ylim(0, y_max)
        if set_label:
            ax2.set_ylabel('DTW Score')
        if not xy_visible:
            ax2.set_yticks([])
            ax2.set_xticks([])

    def plot_dtw_by_minute(self, pt, test_dataset, dtw_cache_dir):
        dtw = dtw_lib.analyze_patient(pt, test_dataset, dtw_cache_dir, None)
        dtw = dtw.sort_values(by='hour')
        pt_data = self.pred_to_hour_frame[self.pred_to_hour_frame.patient == pt]
        for hour in range(24):
            if len(pt_data[(pt_data.hour >= hour) & (pt_data.hour < hour+1)]) == 0:
                continue
            self.plot_disease_evolution(pt_data, plot_by='minute', plot_hour=hour, plt_title='Plot by Minute {} hour: {}'.format(pt, hour+1), xlab='Minute')
            dtw_hr_data = dtw[(dtw.hour >= hour) & (dtw.hour < hour+1)]
            # denormalize minute back to 0-60
            dtw_hr_data['hour'] = (dtw_hr_data.hour - hour) * 60
            self.plot_dtw_patient_data(dtw_hr_data, True, 6, True, dtw['dtw'].max())
            plt.show()

    def plot_disease_evolution(self, pt_data, legend=True, fontsize=11, xylabel=True, xy_visible=True, plot_by='hour', plot_hour=None, plt_title=None, xlab="Hour"):
        # defaults, but we can parameterize them in future if we need
        cmap = ['#6c89b7', '#ff919c']
        plt.rcParams['legend.loc'] = 'upper right'
        time_units = {'hour': 24, 'minute': 60}[plot_by]

        bar_data = [[0, 0] for _ in range(time_units)]
        pt = pt_data.iloc[0].patient
        pt_data_in_bin = pt_data if not plot_hour else pt_data[(pt_data.hour >= plot_hour) & (pt_data.hour < plot_hour+1)]

        for interval  in range(time_units):
            lower_bound = plot_hour + (interval / 60.0) if plot_hour is not None else interval
            upper_bound = plot_hour + (interval / 60.0) + (1 / 60.0) if plot_hour is not None else interval+1
            bin_pt_data = pt_data[(pt_data.hour >= lower_bound) & (pt_data.hour < upper_bound)]
            bar_data[interval] = [1 - bin_pt_data.pred.sum() / float(len(bin_pt_data)), bin_pt_data.pred.sum() / float(len(bin_pt_data))]

        plots = []
        bottom = np.zeros(time_units)
        for n in [0, 1]:
            bar_fracs = np.array([bar_data[bin][n] for bin in range(0, time_units)])
            plots.append(plt.bar(range(0, time_units), bar_fracs, bottom=bottom, color=cmap[n]))
            bottom = bottom + bar_fracs

        plt.title("Patient {}".format(pt[:4]) if not plt_title else plt_title, fontsize=fontsize, pad=1)
        if xylabel:
            plt.ylabel('Fraction Predicted', fontsize=fontsize)
            plt.xlabel(xlab, fontsize=fontsize)
        plt.xlim(-.8, time_units - .02)
        if legend:
            all_votes = len(pt_data_in_bin)
            mapping = {
                'Non-ARDS_percent': round(1 - pt_data_in_bin.pred.sum() / float(all_votes), 3) * 100,
                'ARDS_percent': round(pt_data_in_bin.pred.sum() / float(all_votes), 3) * 100,
            }

            plt.legend([
                "{}: {}%".format(patho, mapping['{}_percent'.format(patho)]) for patho in ['Non-ARDS', 'ARDS']
            ], fontsize=fontsize)
        if not xy_visible:
            plt.yticks([])
            plt.xticks([])
        else:
            plt.yticks(np.arange(0, 1.01, .1))
            plt.xticks(range(0, time_units+1, 5), range(1, time_units+1, 5))

    def plot_tiled_disease_evol(self, test_dataset, dtw_cache_dir, plot_with_dtw):
        """
        Plot a tiled bar chart of patient predictions. Plot by TPs/TNs/FPs/FNs
        """
        tps, tns, fps, fns = [], [], [], []
        for i, rows in self.results.groupby('patient'):
            pt = rows.iloc[0].patient
            total_votes = rows[['OTHER_votes', 'ARDS_votes']].sum().sum()
            ards_votes = rows['ARDS_votes'].sum()
            ground_truth = rows.iloc[0].patho
            if ards_votes / float(total_votes) >= .5:
                pred = 1
            else:
                pred = 0

            if pred == 1 and ground_truth == 1:
                tps.append(pt)
            elif pred == 0 and ground_truth == 0:
                tns.append(pt)
            elif pred == 1 and ground_truth == 0:
                fps.append(pt)
            elif pred != 1 and ground_truth == 1:
                fns.append(pt)

        for arr, title in [
            (tps, 'ARDS True Pos'),
            (tns, 'ARDS True Neg'),
            (fps, 'ARDS False Pos'),
            (fns, 'ARDS False Neg'),
        ]:
            for idx, pt in enumerate(arr):
                layout = int(ceil(sqrt(len(arr))))
                plt.suptitle(title)
                pt_rows = self.pred_to_hour_frame[self.pred_to_hour_frame.patient == pt]
                plt.subplot(layout, layout, idx+1)
                self.plot_disease_evolution(pt_rows, legend=False, fontsize=6, xylabel=False, xy_visible=False)
                if plot_with_dtw:
                    dtw = dtw_lib.analyze_patient(pt, test_dataset, dtw_cache_dir, None)
                    dtw = dtw.sort_values(by='hour')
                    self.plot_dtw_patient_data(dtw, False, 0.05, False, y_max=100)
            plt.show()

    def perform_patient_predictions(self, y_test, predictions, fold_num, epoch_num):
        """
        After a group of patients is run through the model, record all necessary stats
        such as true positives, false positives, etc.

        :param y_test: Should be a pd.DataFrame instance with 2 columns patient, y
        :param predictions: Should be a pd.Series instance with all predictions made on a per-stack basis. Should be numerically indexed in 1-1 match with y_test
        :param fold_num: Which K-fold are we in?
        :param epoch_num: Which epoch number are we in
        """
        for pt in y_test.patient.unique():
            # get all rows for patient from y_test
            pt_rows = y_test[y_test.patient == pt]
            pt_idx = pt_rows.index
            patho_n = pt_rows.y.unique()[0]

            pt_actual = pt_rows.y
            # what was actually predicted for the patient?
            pt_pred = predictions.loc[pt_rows.index]

            pt_results = [pt, patho_n]

            for n, patho in self.pathos.items():
                pt_results.extend([
                    get_tps(pt_actual, pt_pred, n), get_fps(pt_actual, pt_pred, n),
                    get_tns(pt_actual, pt_pred, n), get_fns(pt_actual, pt_pred, n),
                    len(pt_pred[pt_pred == n])
                ])

            pred_frac = float(pt_results[6+5*1]) / sum([pt_results[6+5*j] for j in self.pathos.keys()])
            patho_pred = np.argmax([pt_results[6 + 5*k] for k in range(len(self.pathos))])
            pt_results.extend([patho_pred, pred_frac, epoch_num, fold_num])
            self.results.loc[len(self.results)] = pt_results

        chunked_results = self.results[(self.results.patient.isin(y_test.patient.unique())) & (self.results.epoch_num == epoch_num)]
        stats = self._aggregate_specific_results(chunked_results, fold_num, epoch_num)

        self.update_meter('test_auc', fold_num, stats.iloc[0].auc)
        self.update_meter('test_prec_other', fold_num, stats[stats.patho == 'OTHER'].iloc[0].precision)
        self.update_meter('test_prec_ards', fold_num, stats[stats.patho == 'ARDS'].iloc[0].precision)
        self.update_meter('test_sen_other', fold_num, stats[stats.patho == 'OTHER'].iloc[0].sensitivity)
        self.update_meter('test_sen_ards', fold_num, stats[stats.patho == 'ARDS'].iloc[0].sensitivity)
        self.update_meter('test_f1_other', fold_num, stats[stats.patho == 'OTHER'].iloc[0].f1)
        self.update_meter('test_f1_ards', fold_num, stats[stats.patho == 'ARDS'].iloc[0].f1)
        self.update_meter('test_patient_accuracy', fold_num, stats[stats.patho == 'ARDS'].iloc[0].accuracy)

        self._print_specific_results_report(stats)
        incorrect_pts = chunked_results[chunked_results.patho != chunked_results.prediction]
        table = PrettyTable()
        table.field_names = ['patient', 'actual', 'prediction'] + ['{} Votes'.format(patho) for patho in self.pathos.values()]

        for idx, row in incorrect_pts.iterrows():
            table.add_row([row.patient, row.patho, row.prediction] + [row['{}_votes'.format(patho)] for patho in self.pathos.values()])
        print('Misclassified Patients')
        print(table)

    def save_all(self):
        self.reporting.save_all()
        torch.save(self.hyperparams, os.path.join(self.results_dir, self.experiment_save_filename))
        pd.to_pickle(self, os.path.join(self.results_dir, self.results_save_filename))

    def save_predictions_by_hour(self, y_test, predictions, pred_hour, epoch_num, fold_num):
        """
        Save predictions that we make by the hour (after study inclusion) that they were made.

        :param y_test: y_test dataset. Can be retrieved by dataset.get_ground_truth_df()
        :param predictions: Predictions made indexed by their batch index
        :param pred_hour: The hour that each breath was processed. Can be retrieved via dataset.seq_hours
        """
        processed_pred_hour = np.zeros(len(pred_hour))
        self.pred_to_hour_frame = predictions.to_frame(name='pred')
        for idx in self.pred_to_hour_frame.index.unique():
            hrs = pred_hour[idx]
            pred = self.pred_to_hour_frame.loc[idx, 'pred']
            if isinstance(pred, int) or isinstance(pred, np.int64):
                self.pred_to_hour_frame.loc[idx, 'hour'] = hrs[0]
            else:
                self.pred_to_hour_frame.loc[idx, 'hour'] = hrs

        self.pred_to_hour_frame = self.pred_to_hour_frame.merge(y_test, left_index=True, right_index=True)
        tmp = self.pred_to_hour_frame.copy()
        tmp['epoch'] = epoch_num
        tmp['fold'] = fold_num
        self.all_pred_to_hour = self.all_pred_to_hour.append(tmp, ignore_index=True)
