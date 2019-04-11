import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


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


class DeepARDSResults(object):
    def __init__(self, y_test):
        """
        :param y_test: DataFrame with columns patient, y
        """
        self.y_test = y_test
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

    def aggregate_all_results(self):
        """
        Aggregate final results for all patients into a friendly data frame
        """
        aggregate_stats = []
        for n, patho in self.pathos.items():
            tps = float(len(self.results[(self.results.patho == n) & (self.results.prediction == n)]))
            tns = float(len(self.results[(self.results.patho != n) & (self.results.prediction != n)]))
            fps = float(len(self.results[(self.results.patho != n) & (self.results.prediction == n)]))
            fns = float(len(self.results[(self.results.patho == n) & (self.results.prediction != n)]))
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
                auc = round(roc_auc_score(self.results.patho.tolist(), self.results.prediction.tolist()), 4)
            try:
                f1 = 2 * ((precision * sensitivity) / (precision + sensitivity))
            except ZeroDivisionError:
                f1 = 0
            aggregate_stats.append([patho, tps, tns, fps, fns, accuracy, sensitivity, specificity, precision, auc, f1])

        self.aggregate_stats = pd.DataFrame(
            aggregate_stats,
            columns=['patho', 'tps', 'tns', 'fps', 'fns', 'accuracy', 'sensitivity', 'specificity', 'precision', 'auc', 'f1']
        )

    def perform_patient_predictions(self, predictions):
        """
        After a group of patients is run through the model, record all necessary stats
        such as true positives, false positives, etc.

        :param predictions: Should be a pd.Series instance with all predictions made on a per-stack basis. Should be numerically indexed in 1-1 match with y_test
        """
        for pt in self.y_test.patient.unique():
            pt_rows = self.y_test[self.y_test.patient == pt]
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
            self.results.loc[len(self.results)] = pt_results
