from copy import copy
import os

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score
import torch


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

    def update(self, data, n=1):
        self._count = self._count + n
        if isinstance(data, torch.autograd.Variable):
            self._last_value.copy_(data.data)
            self.values = torch.cat((self.values, data.data.cpu().view(1)), 0)
        elif isinstance(data, torch.Tensor):
            self._last_value.copy_(data)
            self.values = torch.cat((self.values, data.cpu().view(1)), 0)
        else:
            self._last_value.fill_(data)
            self.values = torch.cat((self.values, torch.FloatTensor([data])), 0)
        self._total.add_(self._last_value)

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
        results_cols += ["prediction"]
        # self.results is meant to be a high level dataframe of aggregated statistics
        # from our model.
        self.results = pd.DataFrame([], columns=results_cols)
        reporting_suffix = 'deepards_start_{}'.format(start_time)
        self.results_dir = os.path.join(os.path.dirname(__file__), 'results/')
        self.reporting = Reporting(os.path.join(self.results_dir, reporting_suffix))
        self.hyperparams = hyperparams
        self.hyperparams['start_time'] = start_time
        self.experiment_save_filename = "{}_{}.pth".format(experiment_name, start_time) if experiment_name else start_time + ".pth"

    def aggregate_classification_results(self):
        """
        Aggregate final results for all patients into a friendly data frame
        """
        self.aggregate_stats = self._aggregate_specific_results(self.results)
        self._print_specific_results_report(self.aggregate_stats)
        self.reporting.save_all()

    def _aggregate_specific_results(self, patient_results):
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
                auc = round(roc_auc_score(patient_results.patho.tolist(), patient_results.prediction.tolist()), 4)
            try:
                f1 = 2 * ((precision * sensitivity) / (precision + sensitivity))
            except ZeroDivisionError:
                f1 = 0
            aggregate_stats.append([patho, tps, tns, fps, fns, accuracy, sensitivity, specificity, precision, auc, f1])

        return pd.DataFrame(
            aggregate_stats,
            columns=['patho', 'tps', 'tns', 'fps', 'fns', 'accuracy', 'sensitivity', 'specificity', 'precision', 'auc', 'f1']
        )

    def _print_specific_results_report(self, stats_frame):
        table = PrettyTable()
        table.field_names = ['Patho', 'Accuracy', 'Recall', 'Precision', 'AUC', 'F1']
        for idx, row in stats_frame.iterrows():
            table.add_row([row.patho, row.accuracy, row.sensitivity, row.precision, row.auc, row.f1])
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

    def print_meter_results(self, metric_name, fold_num):
        meter_name = '{}_fold_{}'.format(metric_name, fold_num)
        print(self.reporting.meters[meter_name])

    def perform_patient_predictions(self, y_test, predictions, fold_num):
        """
        After a group of patients is run through the model, record all necessary stats
        such as true positives, false positives, etc.

        :param y_test: Should be a pd.DataFrame instance with 2 columns patient, y
        :param predictions: Should be a pd.Series instance with all predictions made on a per-stack basis. Should be numerically indexed in 1-1 match with y_test
        :param fold_num: Which K-fold are we in?
        """
        fold_auc_meter = 'test_auc_fold_{}'.format(fold_num)
        fold_prec_other_meter = 'test_prec_other_fold_{}'.format(fold_num)
        fold_prec_ards_meter = 'test_prec_ards_fold_{}'.format(fold_num)
        fold_sen_other_meter = 'test_sen_other_fold_{}'.format(fold_num)
        fold_sen_ards_meter = 'test_sen_ards_fold_{}'.format(fold_num)
        if not self.reporting.does_meter_exist(fold_auc_meter):
            self.reporting.new_meter(fold_auc_meter)
            self.reporting.new_meter(fold_prec_other_meter)
            self.reporting.new_meter(fold_prec_ards_meter)
            self.reporting.new_meter(fold_sen_other_meter)
            self.reporting.new_meter(fold_sen_ards_meter)

        # XXX should add metrics for individual frames as well besides just patient results
        for pt in y_test.patient.unique():
            pt_rows = y_test[y_test.patient == pt]
            pt_idx = pt_rows.index
            patho_n = pt_rows.y.unique()[0]

            pt_actual = pt_rows.y
            pt_pred = predictions.loc[pt_rows.index]
            pt_results = [pt, patho_n]

            for n, patho in self.pathos.items():
                pt_results.extend([
                    get_tps(pt_actual, pt_pred, n), get_fps(pt_actual, pt_pred, n),
                    get_tns(pt_actual, pt_pred, n), get_fns(pt_actual, pt_pred, n),
                    len(pt_pred[pt_pred == n]),
                ])

            patho_pred = np.argmax([pt_results[6 + 5*k] for k in range(len(self.pathos))])
            pt_results.extend([patho_pred])
            # If patient results exist then overwrite them. This will occur in the case
            # of running test after each train epoch. Otherwise just add new row
            if len(self.results[self.results.patient == pt]) > 0:
                self.results.loc[self.results.patient == pt] = [pt_results]
            else:
                self.results.loc[len(self.results)] = pt_results

        chunked_results = self.results[self.results.patient.isin(y_test.patient.unique())]
        stats = self._aggregate_specific_results(chunked_results)

        self.reporting.update(fold_auc_meter, stats.iloc[0].auc)
        self.reporting.update(fold_prec_other_meter, stats[stats.patho == 'OTHER'].iloc[0].precision)
        self.reporting.update(fold_prec_ards_meter, stats[stats.patho == 'ARDS'].iloc[0].precision)
        self.reporting.update(fold_sen_other_meter, stats[stats.patho == 'OTHER'].iloc[0].sensitivity)
        self.reporting.update(fold_sen_ards_meter, stats[stats.patho == 'ARDS'].iloc[0].sensitivity)

        self._print_specific_results_report(stats)
        incorrect_pts = chunked_results[chunked_results.patho != chunked_results.prediction]
        patho_votes = ["{}_votes".format(k) for k in self.pathos.values()]
        for idx, row in incorrect_pts.iterrows():
            print("Patient {}: Prediction: {}, Actual: {}. Voting:\n{}".format(
                row.patient, row.prediction, row.patho, row[patho_votes]
            ))

    def save_all(self):
        self.reporting.save_all()
        torch.save(self.hyperparams, os.path.join(self.results_dir, self.experiment_save_filename))
