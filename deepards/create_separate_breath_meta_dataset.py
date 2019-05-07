import argparse
from glob import glob
import multiprocessing
import os
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from ventmap.breath_meta import get_file_breath_meta, META_HEADER


def collect_data(patient_id, data_dir, intermediate_results_dir, warn_file, no_intermediates, desired_cols):
    intermediate_file = os.path.join(intermediate_results_dir, patient_id) + '.pkl'
    if os.path.exists(intermediate_file) and not no_intermediates:
        return pd.read_pickle(intermediate_file)

    files = glob(os.path.join(data_dir, patient_id, "*.csv"))
    cols = desired_cols + ['patient', 'filename']
    if len(files) == 0:
        if os.path.exists(warn_file):
            file_mode = 'a'
        else:
            file_mode = 'w'
        with open(warn_file, file_mode) as f:
            f.write(patient_id + '\n')
        warn('Could not find any data for patient: {}'.format(patient_id))
        return pd.DataFrame([], columns=cols)

    print('Analyze breaths for patient {}'.format(patient_id))
    all_meta = []
    meta_idxs = [META_HEADER.index(col) for col in desired_cols]
    for file in files:
        meta = get_file_breath_meta(file)[1:]
        desired_meta = [[row[idx] for idx in meta_idxs] + [patient_id, file] for row in meta]
        all_meta.extend(desired_meta)

    results = pd.DataFrame(all_meta, columns=cols)
    results.to_pickle(intermediate_file)
    return results


def perform_clustering(df, nclust, breaths_per_clust):
    # clear out frame headers
    df = df[df.ventBN != 'ventBN']
    clustering = AgglomerativeClustering(nclust)
    # cutoff ventbn, patient, and filename
    results = clustering.fit_predict(df.values[:, 1:-2])
    df['cluster'] = results
    idx_choies = []
    for clust in df.cluster.unique():
        # randomly pick n breaths from cluster
        if len(df[df.cluster == clust]]) < breaths_per_clust:
            idxs = list(df[df.cluster == clust]].index)
        else:
            idxs = list(np.random.choice(df[df.cluster == clust]].index, size=breaths_per_clust, replace=False))
        idx_choies.extend(idxs)
    df = df.loc[idx_choices]
    return df


def func_star(args):
    return collect_data(*args)


def clustering_func_star(args):
    return perform_clustering(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-dir', help="Directory where we keep all patient data", required=True)
    parser.add_argument('-i', '--intermediate-results-dir', default='tmp_cohort_results')
    parser.add_argument('-t', '--threads', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('--debug', action='store_true', help='Only run with one thread. That way if a thread dies the stacktrace will be clear')
    parser.add_argument('-wf', '--warn-file', default='no_data_found_warnings.txt')
    parser.add_argument('--only-patient', help='only analyze certain patient')
    parser.add_argument('--no-intermediates', help='do not use intermediates for analysis. Redo all processing', action='store_true')
    parser.add_argument('--clusters', type=int, default=20)
    parser.add_argument('-bp', '--breaths-per-clust', type=int, default=1000)
    args = parser.parse_args()

    if not os.path.exists(args.intermediate_results_dir):
        os.mkdir(args.intermediate_results_dir)

    print('Analyze all breaths for patients')
    patient_ids = glob(os.path.join(args.data_dir, '0*RPI*'))
    patient_ids = [os.path.basename(id) for id in patient_ids if os.path.isdir(id)]
    desired_cols = ['ventBN', 'iTime', 'eTime', 'inst_RR', 'tve:tvi ratio', 'I:E ratio']
    all_runs = [(patient_id, args.data_dir, args.intermediate_results_dir, args.warn_file, args.no_intermediates, desired_cols) for patient_id in patient_ids]
    if args.only_patient:
        all_runs = filter(lambda x: x[0] == args.only_patient, all_runs)
    if not args.debug:
        pool = multiprocessing.Pool(args.threads)
        results = pool.map(func_star, all_runs)
        pool.close()
        pool.join()
    else:
        results = []
        for run in all_runs:
            results.append(func_star(run))

    runs = [[df, args.clusters, args.breaths_per_clust] for df in results]
    if not args.debug:
        pool = multiprocessing.Pool(args.threads)
        results = pool.map(clustering_func_star, runs)
        pool.close()
        pool.join()
    else:
        for run in runs:
            clustering_func_star(run)


if __name__ == "__main__":
    main()
