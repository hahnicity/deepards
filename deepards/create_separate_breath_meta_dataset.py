import argparse
from glob import glob
import multiprocessing
import os
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from ventmap.breath_meta import get_file_breath_meta, get_production_breath_meta, META_HEADER, write_breath_meta
from ventmap.raw_utils import process_breath_file, read_processed_file

from deepards.create_breath_meta_dataset_split import perform_breath_meta_splits


def collect_data(patient_id, data_dir, intermediate_results_dir, warn_file, no_intermediates, desired_cols, nclust, breaths_per_clust, out_dir):
    intermediate_file = os.path.join(intermediate_results_dir, patient_id) + '.pkl'
    if os.path.exists(intermediate_file) and not no_intermediates:
        df = pd.read_pickle(intermediate_file)
    else:
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
            return

        print('Analyze breaths for patient {}'.format(patient_id))
        all_meta = []
        meta_idxs = [META_HEADER.index(col) for col in desired_cols]
        for file in files:
            meta = get_file_breath_meta(open(file, 'rU'))[1:]
            desired_meta = [[row[idx] for idx in meta_idxs] + [patient_id, file] for row in meta]
            all_meta.extend(desired_meta)

        df = pd.DataFrame(all_meta, columns=cols)
        df.to_pickle(intermediate_file)

    df = df[df.ventBN != 'ventBN']
    clustering = KMeans(nclust)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    if len(df) < nclust:
        return

    df.index = range(len(df))
    # cutoff ventbn, patient, and filename
    array = df.values[:, 1:-2]
    results = clustering.fit_predict(array)
    df['cluster'] = results
    idx_choices = []
    for clust in df.cluster.unique():
        # randomly pick n breaths from cluster
        if len(df[df.cluster == clust]) < breaths_per_clust:
            idxs = list(df[df.cluster == clust].index)
        else:
            idxs = list(np.random.choice(df[df.cluster == clust].index, size=breaths_per_clust, replace=False))
        idx_choices.extend(idxs)
    df = df.loc[idx_choices]

    raw_dir = os.path.join(out_dir, 'experiment1', 'all_data', 'raw')
    try:
        os.makedirs(raw_dir)
    except OSError:
        pass
    raw_pt_dir = os.path.join(raw_dir, df.patient.unique()[0])
    try:
        os.mkdir(raw_pt_dir)
    except OSError:
        pass
    meta_dir = os.path.join(out_dir, 'experiment1', 'all_data', 'meta')
    try:
        os.mkdir(meta_dir)
    except OSError:
        pass
    meta_pt_dir = os.path.join(meta_dir, df.patient.unique()[0])
    try:
        os.mkdir(meta_pt_dir)
    except OSError:
        pass

    for filename in df.filename.unique():
        vent_bns = df[df.filename == filename].ventBN.tolist()
        output_filename = os.path.join(raw_pt_dir, os.path.splitext(os.path.basename(filename))[0])
        process_breath_file(open(filename), False, output_filename, spec_vent_bns=vent_bns)
        raw_file = output_filename + '.raw.npy'
        proc_file = output_filename + '.processed.npy'
        gen = read_processed_file(raw_file, proc_file)
        meta_output_filename = os.path.join(meta_pt_dir, os.path.splitext(os.path.basename(filename))[0]) + '.csv'
        metadata = [get_production_breath_meta(breath) for breath in gen]
        write_breath_meta([META_HEADER] + metadata, meta_output_filename)


def func_star(args):
    return collect_data(*args)


def run_parallel_func(func, run_args, threads, is_debug):
    if not is_debug:
        pool = multiprocessing.Pool(threads)
        pool.map(func, run_args)
        pool.close()
        pool.join()
    else:
        for run in run_args:
            func(run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-dir', help="Directory where we keep all patient data", required=True)
    parser.add_argument('-i', '--intermediate-results-dir', default='tmp_cohort_results')
    parser.add_argument('-t', '--threads', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('--debug', action='store_true', help='Only run with one thread. That way if a thread dies the stacktrace will be clear')
    parser.add_argument('-wf', '--warn-file', default='no_data_found_warnings.txt')
    parser.add_argument('--only-patient', help='only analyze certain patient')
    parser.add_argument('--no-intermediates', help='do not use intermediates for analysis. Redo all processing', action='store_true')
    parser.add_argument('--clusters', type=int, default=50)
    parser.add_argument('-bp', '--breaths-per-clust', type=int, default=100)
    parser.add_argument('-o', '--output-dir', default='new_bm_dataset')
    args = parser.parse_args()

    if not os.path.exists(args.intermediate_results_dir):
        os.mkdir(args.intermediate_results_dir)

    print('Analyze all breaths for patients')
    patient_ids = glob(os.path.join(args.data_dir, '0*RPI*'))
    patient_ids = [os.path.basename(id) for id in patient_ids if os.path.isdir(id)]
    # We use these 5 features because they are part of our established fetaure set, whereas
    # the other 4/9 are part of the experimental feature set.
    desired_cols = ['ventBN', 'iTime', 'eTime', 'inst_RR', 'tve:tvi ratio', 'I:E ratio']

    try:
        os.mkdir(args.output_dir)
    except OSError:
        pass

    all_runs = [(patient_id, args.data_dir, args.intermediate_results_dir, args.warn_file, args.no_intermediates, desired_cols, args.clusters, args.breaths_per_clust, args.output_dir) for patient_id in patient_ids]
    if args.only_patient:
        all_runs = filter(lambda x: x[0] == args.only_patient, all_runs)
    run_parallel_func(func_star, all_runs, args.threads, args.debug)
    perform_breath_meta_splits(args.output_dir)


if __name__ == "__main__":
    main()
