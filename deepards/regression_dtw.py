import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def group_by_hour(dtw_and_preds):
    dtw_to_frac = []
    for _, pt_rows in dtw_and_preds.groupby('patient'):
        patho = pt_rows.iloc[0].y
        for hour in range(24):
            rows = pt_rows[(pt_rows.hour >= hour) & (pt_rows.hour < hour+1)]
            if len(rows) == 0:
                continue

            pred_frac = rows.pred.sum() / float(len(rows))
            dtw_to_frac.append([pred_frac, patho, rows.dtw.mean(), rows.dtw.max(), rows.dtw.min(), rows.dtw.std()])
    data = pd.DataFrame(dtw_to_frac, columns=['target', 'patho', 'mean', 'max', 'min', 'std'])
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data


def group_by_minute(dtw_and_preds):
    # first col is pred frac, next is avg dtw, next is max, next is min, next is std
    dtw_to_frac = []
    for _, pt_rows in dtw_and_preds.groupby('patient'):
        patho = pt_rows.iloc[0].y
        for hour in range(24):
            for minute in range(60):
                min_rows = pt_rows[
                    (pt_rows.hour >= hour + (minute / 60.0)) &
                    (pt_rows.hour < hour + (minute / 60.0) + (1 / 60.0))
                ]
                if len(min_rows) == 0:
                    continue

                # XXX difficult to do this by minute currently because with sub-batch-level
                # predictions minute level predictions tend to be homogeneous. There are some
                # breaths however that are not like this. There may be something wrong with
                # their grouping in sub-batches, like somehow a breath from a different
                # time ends up in an incorrect sub-batch
                pred_frac = min_rows.pred.sum() / float(len(min_rows))
                dtw_to_frac.append([pred_frac, patho, min_rows.dtw.mean(), min_rows.dtw.max(), min_rows.dtw.min(), min_rows.dtw.std()])
    data = pd.DataFrame(dtw_to_frac, columns=['target', 'patho', 'mean', 'max', 'min', 'std'])
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data


def obs_only(dtw_and_preds):
    dtw_and_preds = dtw_and_preds.rename(columns={'pred': 'target', 'y': 'patho'})
    dtw_and_preds = dtw_and_preds.drop(['patient', 'hour'], axis=1)
    dtw_and_preds = dtw_and_preds.replace([np.inf, -np.inf], np.nan).dropna()
    return dtw_and_preds


def class_targeting(data):
    features = list(set(data.columns).difference({'target'}))
    x2 = sm.add_constant(data[features])
    est = sm.OLS(data.target, x2)
    res = est.fit()
    print(res.summary())
    visualize_model(x2, data.target, res)


def misclassification_targeting(data):
    features = list(set(data.columns).difference({'patho', 'target'}))
    target = (data.target - data.patho).abs()
    x2 = sm.add_constant(data[features])
    est = sm.OLS(target, x2, hasconst=True)
    res = est.fit()
    print(res.summary())
    visualize_model(x2, target, res)


def visualize_model(data, regression_target, res):
    # XXX plot out the regression target and predictions because the regression
    # summary is pretty misleading. Also its not clear why r^2 is so low when
    # statsmodels calculates it. But when I calc it, its much higher, which is
    # more in line with what I'm actually seeing from my model. Yeah, I think
    # I was doing MSE. R^2 is a bit different in its calculation
    params = res.params
    var_names = params.index
    non_const_vars = var_names.difference(['const'])
    preds = (data[non_const_vars] * params[non_const_vars]).sum(axis=1) + res.params.const
    plt.scatter(x=range(len(preds)), y=preds, s=1, c='orange')
    plt.scatter(x=range(len(preds)), y=regression_target, s=1)
    print('MSE: {}'.format(((preds-regression_target) ** 2).sum() / len(preds)))
    print('Abs Error: {}'.format(abs(preds-regression_target).sum() / len(preds)))
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="File left by DTW experiments in the dtw cache directory. Should be named something like dtw*predictions.pkl")
    parser.add_argument('--regression-type', choices=['class_targeting', 'misclassification_targeting'], default='class_targeting')
    parser.add_argument('--feature-method', choices=['group_by_hour', 'group_by_minute', 'obs_only'], default='obs_only')
    args = parser.parse_args()

    df = pd.read_pickle(args.file)
    if args.feature_method == 'group_by_minute':
        data = group_by_minute(df)
    elif args.feature_method == 'group_by_hour':
        data = group_by_hour(df)
    elif args.feature_method == 'obs_only':
        data = obs_only(df)

    if args.regression_type == 'class_targeting':
        class_targeting(data)
    elif args.regression_type == 'misclassification_targeting':
        misclassification_targeting(data)


if __name__ == "__main__":
    main()
