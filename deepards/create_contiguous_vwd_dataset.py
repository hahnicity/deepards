import argparse
from glob import glob
import multiprocessing
import os
from warnings import warn

import numpy as np
import pandas as pd
from ventmap.breath_meta import get_file_breath_meta, get_production_breath_meta, META_HEADER, write_breath_meta
from ventmap.raw_utils import process_breath_file, read_processed_file

from create_breath_meta_dataset_split import perform_breath_meta_splits


def collect_data(patient_id, data_dir, intermediate_results_dir, warn_file, no_intermediates, contiguous_breaths, time_between_clusters, out_dir, max_clusters):
    intermediate_file = os.path.join(intermediate_results_dir, patient_id) + '.pkl'
    if os.path.exists(intermediate_file) and not no_intermediates:
        df = pd.read_pickle(intermediate_file)
    else:
        files = glob(os.path.join(data_dir, patient_id, "*.csv"))
        desired_cols = ['BN', 'ventBN', 'abs_time_at_BS', 'iTime', 'eTime']
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
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    try:
        df.abs_time_at_BS = pd.to_datetime(df.abs_time_at_BS)
    except:
        df.abs_time_at_BS = pd.to_datetime(df.abs_time_at_BS, format='%Y-%m-%d %H-%M-%S.%f')
    df = df.sort_values(by=['filename', 'BN'])
    df.index = range(len(df))

    if len(df) == 0:
        return

    start_idx = 0
    time_at_start = df.loc[0].abs_time_at_BS
    end_idx = contiguous_breaths
    # Dont consider using a cluster if there is > 20 second gap between breaths
    cutoff_thresh = 20
    mask = df.abs_time_at_BS.shift(-1) - (df.abs_time_at_BS + pd.to_timedelta(df.iTime + df.eTime, unit='s')) > pd.to_timedelta(cutoff_thresh, unit='s')
    idx_choices = []

    # Choose first n breath chunk and continue by retrieving
    # chunks by the amount of time between clusters. There
    # is a corner case and that is make sure that there is 100%
    # coverage in the epoch. If not then just go onto the next
    # possible n breath cluster.
    for n in range(max_clusters):
        while True:
            if not len(mask.loc[start_idx:end_idx-1]) == contiguous_breaths:
                break
            elif not mask.loc[start_idx:end_idx-1].any():
                idx_choices.extend(list(range(start_idx, end_idx)))
                break
            else:
                start_idx += contiguous_breaths
                end_idx += contiguous_breaths

        if end_idx > len(df):
            break

        time_at_end = df.loc[end_idx].abs_time_at_BS
        time_next_clust = time_at_end + pd.to_timedelta(time_between_clusters, unit='m')
        candidate_times = df[df.abs_time_at_BS >= time_next_clust]
        if len(candidate_times) == 0:
            break
        start_idx = candidate_times.iloc[0].name
        end_idx = start_idx + contiguous_breaths


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

    for filename in df.filename.unique():
        vent_bns = df[df.filename == filename].ventBN.tolist()
        output_filename = os.path.join(raw_pt_dir, os.path.splitext(os.path.basename(filename))[0])
        process_breath_file(open(filename), False, output_filename, spec_vent_bns=vent_bns)


def func_star(args):
    try:
        return collect_data(*args)
    except:
        print('patient {} has a fatal error in their dataset processing. Probably time related'.format(args[0]))


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
    parser.add_argument('-cb', '--contiguous-breaths', type=int, default=110)
    parser.add_argument('-tb', '--time-between-clusters', type=int, default=30, help='Time between breath clusters should be listed in minutes')
    parser.add_argument('--max-clusters', type=int, default=300)
    parser.add_argument('-o', '--output-dir', default='contiguous_breath_dataset')
    args = parser.parse_args()

    if not os.path.exists(args.intermediate_results_dir):
        os.mkdir(args.intermediate_results_dir)

    print('Analyze all breaths for patients')
    patient_ids = glob(os.path.join(args.data_dir, '0*RPI*'))
    patient_ids = [os.path.basename(id) for id in patient_ids if os.path.isdir(id)]

    try:
        os.mkdir(args.output_dir)
    except OSError:
        pass

    all_runs = [(patient_id, args.data_dir, args.intermediate_results_dir, args.warn_file, args.no_intermediates, args.contiguous_breaths, args.time_between_clusters, args.output_dir, args.max_clusters) for patient_id in patient_ids]
    if args.only_patient:
        all_runs = filter(lambda x: x[0] == args.only_patient, all_runs)
    run_parallel_func(func_star, all_runs, args.threads, args.debug)
    perform_breath_meta_splits(args.output_dir)


if __name__ == "__main__":
    main()
