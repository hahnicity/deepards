import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def find_number_breaths_in_delta(df, delta):
    """
    :param df: dataframe with dtw information
    :param delta: Parameter N for number of breaths in past N seconds we want to find
    """
    result = []
    second_loc = list(df.columns).index('second')
    for _, patient_frame in df.groupby('patient'):
        idx_last = 0
        cur_count = 0
        patient_frame.index = range(len(patient_frame))
        for idx_cur, row in patient_frame.iterrows():
            cur_count += 1
            for i in range(idx_last, idx_cur):
                if row.second - delta <= patient_frame.iloc[i, second_loc] <= row.second:
                    result.append(cur_count)
                    break
                cur_count -= 1
                idx_last += 1
            else:
                result.append(cur_count)
    return np.array(result).astype(np.int16)


def evaluate_dtw_stats_by_time_col(df, colname):
    dtw_arr = df.dtw.values
    results = []
    col_idx = list(df.columns).index(colname)
    for i in range(len(df)):
        breaths_past_delta = df.iloc[i, col_idx]
        dtw_slice = dtw_arr[i+1-breaths_past_delta:i+1]
        dtw_slice = dtw_slice[~np.isnan(dtw_slice)]
        if len(dtw_slice) == 0:
            results.append([np.nan] * 4)
        else:
            results.append([dtw_slice.mean(), dtw_slice.max(), dtw_slice.min(), dtw_slice.std()])
    results = np.array(results)
    idx_to_name = {0: 'dtw_mean', 1: 'dtw_max', 2: 'dtw_min', 3: 'dtw_std'}
    for idx, name in idx_to_name.items():
        df[name] = results[:, idx]
    return df


def group_by_time(dtw_and_preds, seconds):
    dtw_and_preds['second'] = dtw_and_preds.hour * 60 * 60
    dtw_and_preds['breaths_past_t'] = find_number_breaths_in_delta(dtw_and_preds, seconds)
    dtw_and_preds = evaluate_dtw_stats_by_time_col(dtw_and_preds, 'breaths_past_t')

    dtw_and_preds = dtw_and_preds.rename(columns={'y': 'actual'})
    dtw_and_preds = dtw_and_preds.drop(['second', 'breaths_past_t', 'hour', 'patient'], axis=1)
    dtw_and_preds = dtw_and_preds.replace([np.inf, -np.inf], np.nan).dropna()
    return dtw_and_preds


def group_by_batch(dtw_and_preds):
    batch_size = len(dtw_and_preds.loc[0])
    if batch_size <= 1:
        raise Exception('You cannot group by batch when you were not doing batch-level processing to begin with')

    results = []
    for uniq in dtw_and_preds.index.unique():
        slice = dtw_and_preds.loc[uniq, 'dtw']
        results.extend([[slice.mean(), slice.max(), slice.min(), slice.std()]]*batch_size)

    results = np.array(results)
    idx_to_name = {0: 'dtw_mean', 1: 'dtw_max', 2: 'dtw_min', 3: 'dtw_std'}
    for idx, name in idx_to_name.items():
        dtw_and_preds[name] = results[:, idx]
    dtw_and_preds = dtw_and_preds.drop(['hour', 'patient'], axis=1)
    dtw_and_preds = dtw_and_preds.rename(columns={'y': 'actual'})
    dtw_and_preds = dtw_and_preds.replace([np.inf, -np.inf], np.nan).dropna()
    return dtw_and_preds


def obs_only(dtw_and_preds):
    dtw_and_preds = dtw_and_preds.rename(columns={'y': 'actual'})
    dtw_and_preds = dtw_and_preds.drop(['patient', 'hour'], axis=1)
    dtw_and_preds = dtw_and_preds.replace([np.inf, -np.inf], np.nan).dropna()
    return dtw_and_preds


def class_targeting(data):
    features = list(set(data.columns).difference({'actual', 'pred'}))
    x2 = sm.add_constant(data[features])
    est = sm.OLS(data.actual, x2)
    res = est.fit()
    print(res.summary())
    visualize_model(x2, data.actual, res)


def misclassification_targeting(data):
    features = list(set(data.columns).difference({'pred', 'actual'}))
    target = (data.pred - data.actual).abs()
    x2 = sm.add_constant(data[features])
    est = sm.OLS(target, x2, hasconst=True)
    res = est.fit()
    print(res.summary())
    visualize_model(x2, target, res)


def visualize_model(data, regression_target, res):
    # plot out the regression target and predictions because the regression
    # summary is pretty misleading. Also its not clear why r^2 is so low when
    # statsmodels calculates it. But when I calc it, its much higher, which is
    # more in line with what I'm actually seeing from my model. Yeah, I think
    # I was doing MSE. R^2 is a bit different in its calculation
    #
    # r^2 is low because the model doesn't explain the variance of the data. If
    # r^2 was closer to 1 then it would imply that the model is tightly fitting
    # the variance of the data. A crap r^2 value like you have is pretty indicative that
    # the modeling is not working
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
    parser.add_argument('--feature-method', choices=['group_by_hour', 'group_by_minute', 'obs_only', 'group_by_batch'], default='obs_only')
    args = parser.parse_args()

    df = pd.read_pickle(args.file)
    if args.feature_method == 'group_by_minute':
        data = group_by_time(df, 60)
    elif args.feature_method == 'group_by_hour':
        data = group_by_time(df, 60*60)
    elif args.feature_method == 'obs_only':
        data = obs_only(df)
    elif args.feature_method == 'group_by_batch':
        data = group_by_batch(df)

    if args.regression_type == 'class_targeting':
        class_targeting(data)
    elif args.regression_type == 'misclassification_targeting':
        misclassification_targeting(data)


if __name__ == "__main__":
    main()
