"""
Automated script to do all training/testing on the training with similar data and then
testing with similar and dissimilar data.
"""
import argparse
import subprocess

import numpy as np
import os
import pandas as pd
import yaml

from deepards.config import Configuration
from deepards.dtw_lib import pick_similar_pts, pick_dissimilar_pts
from deepards.perform_splitting import perform_preset_file_split
from deepards.train_ards_detector import build_parser, BaseTraining


def do_split(similarity_file, pickled_dataset, n):
    a = pd.read_pickle(pickled_dataset)
    sim = pd.read_pickle(similarity_file)
    a.total_kfolds = None; a.kfold_num = None; a.train=True

    train_sim_pts = pick_similar_pts(sim, a, 40, retrieve_n=10, mean_similarity_thresh=.7)[n][1]
    test_dissim_pts = pick_dissimilar_pts(sim, a, 6, exclude=train_sim_pts, retrieve_n=10, mean_similarity_thresh=.7)[n][1]
    # XXX
    test_sim_pts = pick_similar_pts(sim, a, 6, exclude=train_sim_pts+test_dissim_pts, retrieve_n=10, mean_similarity_thresh=.7)[n][1]
    gt = a.get_ground_truth_df().sort_index()
    patho = gt.groupby('patient').y.first()

    patho_dissim = patho.loc[test_dissim_pts]
    patho_sim = patho.loc[test_sim_pts]

    if np.random.rand() > .5:  # trim non-ards from dissim, ards from sim
        trim_dissim = np.random.choice(patho_dissim[patho_dissim == 0].index, 1)[0]
        trim_sim = np.random.choice(patho_sim[patho_sim == 1].index, 1)[0]
    else:  # trim ards from dissim, non-ards from sim
        trim_dissim = np.random.choice(patho_dissim[patho_dissim == 1].index, 1)[0]
        trim_sim = np.random.choice(patho_sim[patho_sim == 0].index, 1)[0]

    test_dissim_pts = list(set(test_dissim_pts).difference({trim_dissim}))
    test_sim_pts = list(set(test_sim_pts).difference({trim_sim}))

    return {
        'train': train_sim_pts,
        'test': test_sim_pts+test_dissim_pts,
        'similar': test_sim_pts,
        'dissimilar': test_dissim_pts,
    }


def make_base_config_file(split_name):
    return {
		'base_network': 'densenet18',
		'batch_size': 16,
		'clip_val': 0.01,
		'cuda_no_dp': True,
		'data_path': '/home/grehm/workspace/datasets/ardsdetection',
		'dataset_type': 'unpadded_centered_sequences',
		'epochs': 15,
		'final_validation': True,
		'holdout_set_type': split_name,
		'loader_threads': 0,
		'n_sub_batches': 20,
		'network': 'cnn_linear',
    }


def make_config_file_write_to_pickle(split_name):
    base = make_base_config_file(split_name)
    save_dir = os.path.join(os.path.dirname(__file__), '../../pickle_cache')
    updates = {
		'test_to_pickle': os.path.join(save_dir, 'unpadded_centered_sequences-nb20-test-holdout-{}.pkl'.format(split_name)),
		'train_to_pickle': os.path.join(save_dir, 'unpadded_centered_sequences-nb20-train-holdout-{}.pkl'.format(split_name)),
    }
    base.update(updates)
    return base


def make_config_file_read_from_pickle(split_name):
    base = make_base_config_file(split_name)
    save_dir = os.path.join(os.path.dirname(__file__), '../../pickle_cache')
    updates = {
		'test_from_pickle': os.path.join(save_dir, 'unpadded_centered_sequences-nb20-test-holdout-{}.pkl'.format(split_name)),
		'train_from_pickle': os.path.join(save_dir, 'unpadded_centered_sequences-nb20-train-holdout-{}.pkl'.format(split_name)),
    }
    base.update(updates)
    return base


parser = argparse.ArgumentParser()
parser.add_argument('similarity_file')
parser.add_argument('pickled_dataset', help='path to the pickled dataset. should have all patients from the entire dataset represented in it')
parser.add_argument('dataset_path', help='pathing to the main dataset')
parser.add_argument('--n-splits', type=int, default=10)
args = parser.parse_args()

for n in range(args.n_splits):
    # already ran the 0th split on lakota
    if n == 0:
        continue

    split_name = 'train_sim_test_sim_dissim_split_{}'.format(n)
    split_results = do_split(args.similarity_file, args.pickled_dataset, n)
    split_filepath = os.path.join(
        os.path.dirname(__file__), '../../data_split_files', '{}.yml'.format(split_name)
    )
    with open(split_filepath, 'w') as f:
        yaml.dump(split_results, f)
    perform_preset_file_split(args.dataset_path, split_filepath)
    experimental_config_path = os.path.join(
        os.path.dirname(__file__), '../../experiment_files', '{}.yml'.format(split_name)
    )
    with open(experimental_config_path, 'w') as f:
        yaml.dump(make_config_file_write_to_pickle(split_name), f)

    # pickle data file
    main_parser = build_parser()
    main_parser.config_override = experimental_config_path
    args = main_parser.parse_args()
    config = Configuration(args)
    cls = BaseTraining(config)
    cls.get_base_datasets()

    with open(experimental_config_path, 'w') as f:
        yaml.dump(make_config_file_read_from_pickle(split_name), f)

    # set directory back to deepards/deepards so that we can run non_pretraining experiments
    # script
    os.chdir(os.path.join(os.dirname(__file__), '../../'))
    experimental_config_path = os.path.join('experiment_files', '{}.yml'.format(split_name))
    # debug for now
    proc = subprocess.Popen(['python', 'scripts/main/run_non_pretraining_experiments.py', '-co', experimental_config_path, '--cuda-devices', '0+1+2+3', '--debug'])
    proc.communicate()
    # XXX run benchmarking process
