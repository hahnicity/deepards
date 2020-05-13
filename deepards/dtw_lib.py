from datetime import datetime
from glob import glob
import multiprocessing
import os

from dtwco.warping.core import dtw
from matplotlib.dates import date2num
import numpy as np
import pandas as pd
from ventmap.constants import OUT_DATETIME_FORMAT
from ventmap.raw_utils import read_processed_file


def multi_proc_helper(dataset, pt, df_map, pts):

    i = pts.index(pt)
    other_pts = pts[i+1:]
    dict_ = {i: [] for i in other_pts}

    for iloc, seq_info in enumerate(df_map[pt].iterrows()):

        pt_seq_data = dataset.all_sequences[seq_info[0]][1].ravel()
        for other_pt in other_pts:
            try:
                other_seq_info = df_map[other_pt].iloc[iloc]
            except:
                continue
            other_seq_data = dataset.all_sequences[other_seq_info.name][1].ravel()
            dict_[other_pt].append(dtw(pt_seq_data, other_seq_data))
    return (pt, dict_)


def func_star(args):
    return multi_proc_helper(*args)


def find_patient_similarity(dataset, fold_num, threads):
    """
    Want to find similarity between patients rather than within patients

    Search space will be unreasonably large if we try to compare each sequence
    against all other sequences. An O(n) search on DTW will take about an hour,
    so O(n^2) will around 25000 hours. So goal here is just to compare 1st sequence
    for each patient to all other 1st sequences, 2nd sequence to all other 2nd etc.
    """
    # So even if I followed this speedier calc to completion it would still take approx 4.5
    # days to complete. This is fairly frustrating. But maybe its just something I should
    # learn to live with. I think I can speed it up using multiprocessing tho
    dataset.set_kfold_indexes_for_fold(fold_num)
    gt = dataset.get_ground_truth_df().sort_index()
    pts = list(gt.patient.unique())
    df_map = {}
    fold_num = fold_num

    for pt in pts:
        df_map[pt] = gt[gt.patient == pt]

    pool = multiprocessing.Pool(threads)
    results = pool.map(func_star, [(dataset, pt, df_map, pts) for pt in pts])
    pool.close()
    pool.join()
    dtw_scores = {k[0]: k[1] for k in results}
    pd.to_pickle(dtw_scores, 'dtw_cache/inter_patient_similarity-fold-{}.pkl'.format(fold_num))


def _find_per_breath_dtw_score(prev_flow_waves, breath):
    # compare the last n_breaths to current breath to compute DTW score
    score = 0
    for flow in prev_flow_waves:
        score += dtw(flow, breath)
    return score / len(prev_flow_waves)


def dtw_analyze(pt_data, n_breaths, rolling_av_len, pt_preds_by_hour):
    """
    :param pt_data: List of sequences for the particular patient we want to analyze
    :param n_breaths: number of breaths we want to look back in the window
    :param rolling_av_len: An additional rolling average to compute on top of the stats. Can be 1 if you don't want a rolling average
    :param pt_preds_by_hour: dataframe of patient predictions by hour
    """
    flow_waves = []
    dtw_scores = [np.nan] * n_breaths
    hrs = [np.nan] * n_breaths
    pt_obs_idxs = list(set(pt_preds_by_hour.index))
    df_idx = []

    for idx, seq in enumerate(pt_data):
        cur_obs_idx = pt_obs_idxs[idx]
        hours = pt_preds_by_hour.loc[cur_obs_idx].hour.to_list()

        for j, breath in enumerate(seq):
            current_seq_time = hours[j]

            df_idx.append(cur_obs_idx)
            if len(flow_waves) == (n_breaths+1):
                flow_waves.pop(0)

            breath = breath.ravel()
            if len(flow_waves) < (n_breaths):
                flow_waves.append(breath)
                continue

            dtw_scores.append(_find_per_breath_dtw_score(flow_waves, breath))
            hrs.append(current_seq_time)
            flow_waves.append(breath)

    rolling_av = np.convolve(dtw_scores, np.ones((rolling_av_len,))/rolling_av_len, mode='valid')
    return pd.DataFrame(np.array([np.append([np.nan]*(rolling_av_len-1), rolling_av), hrs]).T, columns=['dtw', 'hour'], index=df_idx)


def analyze_patient(patient_id, dataset, cache_dir, preds_by_hour):
    """
    :param patient_id: The patient id you want to analyze
    :param dataset: An instance of the ARDSRawDataset you want to analyze
    :param cache_dir: A path to the DTW cache directory
    :param preds_by_hour: dataframe of all predictions with the hour they were made
    """
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if not os.path.exists(os.path.join(cache_dir, patient_id)):
        os.mkdir(os.path.join(cache_dir, patient_id))

    n_breaths = 3
    rolling_len = 1
    if dataset.kfold_num:
        split_type = 'kfold'
    else:
        split_type = 'holdout'

    cache_file = "{}_n{}_rolling{}_{}_nb{}_{}.pkl".format(
        patient_id, n_breaths, rolling_len, dataset.dataset_type, dataset.n_sub_batches, split_type
    )
    cache_file_path = os.path.join(cache_dir, patient_id, cache_file)
    if os.path.exists(cache_file_path):
        return pd.read_pickle(cache_file_path)

    y_test = dataset.get_ground_truth_df()
    pt_obs_idx = y_test[y_test.patient == patient_id].index
    pt_data = [dataset[i][1] for i in pt_obs_idx]

    pt_preds_by_hour = preds_by_hour[preds_by_hour.patient == patient_id]
    dtw_scores = dtw_analyze(pt_data, n_breaths, rolling_len, pt_preds_by_hour)
    pd.to_pickle(dtw_scores, cache_file_path)
    return dtw_scores
