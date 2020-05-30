from pprint import pprint

import numpy as np
import pandas as pd

from deepards.dtw_lib import pick_similar_pts, pick_dissimilar_pts


def do_first_split_run():
    a = pd.read_pickle('/fastdata/deepards/unpadded_centered_sequences-nb20-kfold.pkl')
    sim = pd.read_pickle('dtw_cache/unpadded_centered_sequences-nb20-all-pts-random-seqs-dist.pkl')
    a.total_kfolds = None; a.kfold_num = None; a.train=True

    train_sim_pts = pick_similar_pts(sim, a, 40)
    test_dissim_pts = pick_dissimilar_pts(sim, a, 6, exclude=train_sim_pts)
    test_sim_pts = pick_similar_pts(sim, a, 6, exclude=train_sim_pts+test_dissim_pts)
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

    print('Train patients')
    for pt in sorted(train_sim_pts):
        print(" - {}".format(pt))

    print('Test patients')
    for pt in sorted(test_sim_pts+test_dissim_pts):
        if pt in test_sim_pts:
            print(" - {}  # similar cluster".format(pt))
        elif pt in test_dissim_pts:
            print(" - {}  # dissimilar cluster".format(pt))
        else:
            raise Exception('something went wrong')

    print('\nput these in your split file for record keeping\n')
    print('similar:')
    for pt in sorted(test_sim_pts):
        print(' - {}'.format(pt))

    print('dissimilar:')
    for pt in sorted(test_dissim_pts):
        print(' - {}'.format(pt))


def do_second_split_run():
    a = pd.read_pickle('/fastdata/deepards/unpadded_centered_sequences-nb20-kfold.pkl')
    sim = pd.read_pickle('dtw_cache/unpadded_centered_sequences-nb20-all-pts-random-seqs-dist.pkl')
    a.total_kfolds = None; a.kfold_num = None; a.train=True

    train_sim_pts = pick_similar_pts(sim, a, 40, retrieve_n=10, mean_similarity_thresh=.7)[1][1]
    test_dissim_pts = pick_dissimilar_pts(sim, a, 6, exclude=train_sim_pts, retrieve_n=10, mean_similarity_thresh=.7)[1][1]
    # XXX
    test_sim_pts = pick_similar_pts(sim, a, 6, exclude=train_sim_pts+test_dissim_pts, retrieve_n=10, mean_similarity_thresh=.7)[1][1]
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

    print('Train patients')
    for pt in sorted(train_sim_pts):
        print(" - {}".format(pt))

    print('Test patients')
    for pt in sorted(test_sim_pts+test_dissim_pts):
        if pt in test_sim_pts:
            print(" - {}  # similar cluster".format(pt))
        elif pt in test_dissim_pts:
            print(" - {}  # dissimilar cluster".format(pt))
        else:
            raise Exception('something went wrong')

    print('\nput these in your split file for record keeping\n')
    print('similar:')
    for pt in sorted(test_sim_pts):
        print(' - {}'.format(pt))

    print('dissimilar:')
    for pt in sorted(test_dissim_pts):
        print(' - {}'.format(pt))

do_second_split_run()
