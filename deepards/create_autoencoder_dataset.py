import argparse
from glob import glob
import multiprocessing
import os
import shutil
import subprocess
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from ventmap.breath_meta import get_file_breath_meta, get_production_breath_meta, META_HEADER, write_breath_meta
from ventmap.raw_utils import process_breath_file, read_processed_file


def collect_data(patient_id, data_dir, intermediate_results_dir, warn_file, no_intermediates, breaths_per_file, out_dir):
    desired_cols = ['BN', 'ventBN']
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
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    if len(df) == 0:
        return

    idx_choices = []
    df.index = range(len(df))
    for filename in df.filename.unique():
        idx_choices.extend(df[df.filename == filename][:breaths_per_file].index)
    df = df.loc[idx_choices]

    raw_dir = os.path.join(out_dir, 'raw')
    try:
        os.mkdir(raw_dir)
    except OSError:
        pass
    raw_pt_dir = os.path.join(out_dir, 'raw', df.patient.unique()[0])
    try:
        os.mkdir(raw_pt_dir)
    except OSError:
        pass

    for filename in df.filename.unique():
        vent_bns = df[df.filename == filename].ventBN.tolist()
        output_filename = os.path.join(raw_pt_dir, os.path.splitext(os.path.basename(filename))[0])
        process_breath_file(open(filename), False, output_filename, spec_vent_bns=vent_bns)


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
    parser.add_argument('-bp', '--breaths-per-file', type=int, default=64)
    parser.add_argument('-o', '--output-dir', default='autoencoder_dataset')
    parser.add_argument('--split-only', action='store_true', help='only split the datasets dont do any analysis')
    parser.add_argument('--cohort-description', default='cohort-description.csv')
    parser.add_argument('-e', '--experiment-num', type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(args.intermediate_results_dir):
        os.mkdir(args.intermediate_results_dir)

    print('Analyze all breaths for patients')
    patient_ids = glob(os.path.join(args.data_dir, '0*RPI*'))
    patient_ids = [os.path.basename(id) for id in patient_ids if os.path.isdir(id)]
    # We use these 5 features because they are part of our established fetaure set, whereas
    # the other 4/9 are part of the experimental feature set.

    try:
        os.mkdir(args.output_dir)
    except OSError:
        pass

    if not args.split_only:
        all_runs = [(patient_id, args.data_dir, args.intermediate_results_dir, args.warn_file, args.no_intermediates, args.breaths_per_file, args.output_dir) for patient_id in patient_ids]
        if args.only_patient:
            all_runs = filter(lambda x: x[0] == args.only_patient, all_runs)
        run_parallel_func(func_star, all_runs, args.threads, args.debug)

    # perform dataset splits and setup dirs in manner same as ardsdetection dataset
    desc = pd.read_csv(args.cohort_description)
    cohort_patient_ids = desc[(desc['Potential Enrollment'] == 'Y') & (desc['experiment_group'] == args.experiment_num)]['Patient Unique Identifier']
    # this code just removes any patient who didnt have data from the cohort
    cohort_patient_ids = set(patient_ids).intersection(cohort_patient_ids)
    non_cohort_ids = set(patient_ids).difference(cohort_patient_ids)
    all_data_dir = os.path.join(args.output_dir, 'experiment{}'.format(args.experiment_num), 'all_data')
    try:
        os.makedirs(all_data_dir)
    except OSError:
        pass
    shutil.move(os.path.join(args.output_dir, 'raw'), all_data_dir)

    train_raw_dir = os.path.join(args.output_dir, 'experiment{}'.format(args.experiment_num), 'prototrain', 'raw')
    test_raw_dir = os.path.join(args.output_dir, 'experiment{}'.format(args.experiment_num), 'prototest', 'raw')
    try:
        os.makedirs(train_raw_dir)
    except OSError:
        pass
    try:
        os.makedirs(test_raw_dir)
    except OSError:
        pass

    for patient in non_cohort_ids:
        proc = subprocess.Popen(['ln', '-s', os.path.abspath(os.path.join(all_data_dir, 'raw', patient)), os.path.abspath(train_raw_dir)])

    for patient in cohort_patient_ids:
        proc = subprocess.Popen(['ln', '-s', os.path.abspath(os.path.join(all_data_dir, 'raw', patient)), os.path.abspath(test_raw_dir)])


if __name__ == "__main__":
    main()
