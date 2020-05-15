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

from deepards.mediods import KMedoids


# XXX So theres a few things that I need to do.
#
# 1. Need to find cluster of like data
# 2. need to find a cluster of patients who are totally dissimilar
# 3. compare the clusters strategies to each other.
#
# Not totally clear if mediods strat is necessary here. Maybe better would be knapsack. Or
# traveling salesman. Brute force it would be an n choose k problem, but this would take
# obscenely long even for a dataset of 80 patients.
#
# So I can probably use TSP to 1. minimize cluster distances. 2. maximize cluster distances
# But setup of TSP is slightly different than what we have. because we dont care about
# returning to origin or having all nodes. Do maybe then we would want to do KP? I think KP
# is best, because your values can be your distances, your weight can be number items you
# want
#
# Altho I guess k-mediod could be useful from perspective that you might want to see
# how combining different clusters together react. For now lets stick to the suggestion
#
# Eh poop, KP wasnt meant for graph algorithms. back to mediods
#
# You can visualize the distance matrix using triangulation btw.

def pick_dissimilar_pts(dist_data, main_dataset, n_pts):
    """
    Pick set of as maximally possible dissimilar patients

    Use a greedy algorithm because its probably good enough to get the result that we are
    looking for. If it doesn't work then I don't see much value in trying to get the most
    perfectly dissimilar set possible because if it doesn't work with greedy case then
    working with a more tuned case may not be generalizable to the problem

    :param dist_data: pd.DataFrame of distances retrieved from find_patient_similarity
    :param main_dataset: Instance of ARDSRawDataset that we originally used
    :param n_pts: number of patients to select
    """
    gt = main_dataset.get_ground_truth_df().sort_index()
    patho = gt.groupby('patient').y.first()
    patients = gt.patient.unique()

    candidate_sets = []
    arr = dist_data.values
    patho_to_select = int(n_pts / 2)
    lower_tri = dist_data.copy()

    for i, patient in enumerate(patients):
        for pt2 in patients[i+1:]:
            lower_tri.loc[patient, pt2] = 0

    # greedy search with each patient as a starting point
    for patient in patients:

        patient_patho = patho.loc[patient]
        picked = [patient]
        # pick max distance. alternate picking other/ards patients based on pathophys
        # of the initial patient
        for i in range(n_pts-1):
            patho_to_select = (patient_patho+(i+1)) % 2
            patho_cand = patho[patho == patho_to_select].index.difference(picked)
            picked.append(dist_data.loc[patho_cand, picked].sum(axis=1).argmax())

        candidate_sets.append([lower_tri[picked].sum().sum(), picked])

    return sorted(candidate_sets, key=lambda x: x[0])[-1][1]


def pick_similar_pts(dist_data, main_dataset, n_pts):
    """
    Find set of most similar patients we can while preserving for pathophysiology split

    :param dist_data: pd.DataFrame of distances retrieved from find_patient_similarity
    :param main_dataset: Instance of ARDSRawDataset that we originally used
    :param n_pts:
    """
    gt = main_dataset.get_ground_truth_df().sort_index()
    patho = gt.groupby('patient').y.first()

    arr = dist_data.values
    candidates = []
    patho_to_select = int(n_pts / 2)
    for val in range(1000, int(dist_data.max().max()+1000), 1000):
        for i in range(len(dist_data.values)):
            mask = arr[i] < val
            count = len(arr[i][mask])
            if count >= n_pts:
                pts = dist_data.columns[mask]
                counts = patho.loc[pts].value_counts()
                try:
                    cond1 = counts[0] >= patho_to_select
                except:
                    break
                try:
                    cond2 = counts[1] >= patho_to_select
                except:
                    break
                if cond1 and cond2:
                    candidates.append((count, pts, dist_data.columns[i]))

        if len(candidates) > 0:
            break

    best = []
    for count, pts, mediod in candidates:
        # perform greedy selection from mediod
        normals = patho.loc[pts][patho.loc[pts] == 0].index
        ards = patho.loc[pts][patho.loc[pts] == 1].index
        best_normals = list(dist_data.loc[mediod, normals].sort_values()[:patho_to_select].index)
        best_ards = list(dist_data.loc[mediod, ards].sort_values()[:patho_to_select].index)
        cost = dist_data.loc[mediod, best_normals+best_ards].sum()
        best.append((cost, best_normals+best_ards))

    best = sorted(best, key=lambda x: x[0])
    return best[0][1]


def mediod_process(dist_data, nclusts, main_dataset):
    """
    :param dist_data: pd.DataFrame of distances retrieved from find_patient_similarity
    :param nclusts: number of clusters for kmediods
    :param main_dataset: Instance of ARDSRawDataset that we originally used
    """
    gt = main_dataset.get_ground_truth_df().sort_index()
    patho = gt.groupby('patient').y.first().to_frame()
    km = KMediods(nclusts, metric='precomputed')
    km.fit(dist_data.values)
    predicted_clusts = km.predict(dist_data.values)
    # need to tie patho to predicted clusts. seems to be some pretty clear dispersion
    # between ARDS and non-ARDS clusters. At least just from looking at fold 0
    patho['clust'] = predicted_clusts
    return patho


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

    Multiprocessing helps the speed issues as well.
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

    # transform scores into a matrix
    data_dict = {row[0]: {row2[0]: 0 for row2 in results} for row in results}
    for row in results:
        pt = row[0]
        pt_results = row[1]
        for pt2_name in pt_results:
            pt2_results = pt_results[pt2_name]
            mean_ = np.mean(pt2_results)
            data_dict[pt][pt2_name] = mean_
            data_dict[pt2_name][pt] = mean_

    pd.to_pickle(pd.DataFrame(data_dict), 'dtw_cache/inter_patient_similarity-fold-{}.pkl'.format(fold_num))


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
