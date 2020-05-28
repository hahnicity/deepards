from pprint import pprint

import numpy as np
import pandas as pd

from deepards.dtw_lib import pick_similar_pts, pick_dissimilar_pts


a = pd.read_pickle('/fastdata/deepards/unpadded_centered_sequences-nb20-kfold.pkl')
sim = pd.read_pickle('dtw_cache/unpadded_centered_sequences-nb20-all-pts-random-seqs-dist.pkl')
a.total_kfolds = None; a.kfold_num = None; a.train=True

train_sim_pts = pick_similar_pts(sim, a, 40)
test_dissim_pts = pick_dissimilar_pts(sim, a, 6, exclude=train_sim_pts)
test_sim_pts = pick_similar_pts(sim, a, 6, exclude=train_sim_pts+test_dissim_pts)
gt = a.get_ground_truth_df().sort_index()
patho = gt.groupby('patient').y.first()

s = patho.loc[test_dissim_pts+test_sim_pts]
non_ards_test = list(np.random.choice(s[s == 0].index, 5, replace=False))
ards_test = list(np.random.choice(s[s == 1].index, 5, replace=False))

print('Train patients')
for pt in sorted(train_sim_pts):
    print(" - {}".format(pt))

print('Test patients')
for pt in sorted(non_ards_test+ards_test):
    if pt in test_sim_pts:
        print(" - {}  # similar cluster".format(pt))
    elif pt in test_dissim_pts:
        print(" - {}  # dissimilar cluster".format(pt))
    else:
        raise Exception('something went wrong')
