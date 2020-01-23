import argparse
from glob import glob
import os
import subprocess

import pandas as pd


def perform_breath_meta_splits(dataset_dir):
    cohort = pd.read_csv('cohort-description.csv')
    ards_cohort_patients = cohort[(cohort.experiment_group == 1) & (cohort['Potential Enrollment'] == 'Y')]['Patient Unique Identifier']

    all_data_raw_path = os.path.join(dataset_dir, 'experiment1', 'all_data', 'raw')
    all_data_meta_path = os.path.join(dataset_dir, 'experiment1', 'all_data', 'meta')
    training_raw_path = os.path.join(dataset_dir, 'experiment1', 'prototrain', 'raw')
    training_meta_path = os.path.join(dataset_dir, 'experiment1', 'prototrain', 'meta')
    testing_raw_path = os.path.join(dataset_dir, 'experiment1', 'prototest', 'raw')
    testing_meta_path = os.path.join(dataset_dir, 'experiment1', 'prototest', 'meta')

    try:
        os.makedirs(training_raw_path)
    except OSError:
        pass
    try:
        os.makedirs(training_meta_path)
    except OSError:
        pass
    try:
        os.makedirs(testing_raw_path)
    except OSError:
        pass
    try:
        os.makedirs(testing_meta_path)
    except OSError:
        pass

    all_patients = [os.path.basename(i) for i in glob(all_data_raw_path + '/*')]
    train_cohort = set(all_patients).difference(set(ards_cohort_patients))
    test_cohort = set(all_patients).intersection(set(ards_cohort_patients))

    for patient in train_cohort:
        proc = subprocess.Popen(['ln', '-s', all_data_raw_path + "/" + patient, training_raw_path])
        proc.communicate()
        proc = subprocess.Popen(['ln', '-s', all_data_meta_path + "/" + patient, training_meta_path])
        proc.communicate()

    for patient in test_cohort:
        proc = subprocess.Popen(['ln', '-s', all_data_raw_path + "/" + patient, testing_raw_path])
        proc.communicate()
        proc = subprocess.Popen(['ln', '-s', all_data_meta_path + "/" + patient, testing_meta_path])
        proc.communicate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir')
    args = parser.parse_args()

    perform_breath_meta_splits(args.dataset_dir)


if __name__ == '__main__':
    main()
