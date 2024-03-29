import argparse
import random

from dtwco.warping.core import dtw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from deepards.patient_gradcam import PatientGradCam


manhattan_distance = lambda x, y: np.abs(x - y)
euclidean_distance = lambda x, y: np.sqrt((x - y) ** 2)

def dtw_example_plot(x, y):
	d, cost_matrix, path = dtw(x, y, dist_only=False)
	plt.imshow(cost_matrix.T, origin='lower', cmap='plasma', interpolation='nearest')
	plt.plot(path[0], path[1], 'w')


def plot_cam_sequence(x, br, cam_outputs):
    img = br.reshape((len(br), 1))
    plt.scatter(x, img, c=cam_outputs, vmin = 0, vmax = 255)


parser = argparse.ArgumentParser()
parser.add_argument('model_path', help='path to saved model')
parser.add_argument('-pdp', '--pickled-data-path', help='pickled data path', required=True)
parser.add_argument('--fold', type=int, required=True)
args = parser.parse_args()

model = torch.load(args.model_path)
patient_id = '0015RPI0320150401'
data = pd.read_pickle(args.pickled_data_path)
data.set_kfold_indexes_for_fold(args.fold)
data.transforms = None
grad_cam = PatientGradCam(model, data)

gt = data.get_ground_truth_df()
gt.index = range(len(gt))
patient_idxs = list(gt[gt.patient == patient_id].index.copy())
target = gt.loc[patient_idxs].y.iloc[0]
idx1 = patient_idxs.pop(random.choice(range(len(patient_idxs))))
idx2 = patient_idxs.pop(random.choice(range(len(patient_idxs))))
rand_inst1 = random.randint(0, 19)
rand_inst2 = random.randint(0, 19)

cam_out1, br1 = grad_cam.get_single_sequence_grad_cam(idx1, rand_inst1, target)
cam_out2, br2 = grad_cam.get_single_sequence_grad_cam(idx2, rand_inst2, target)
br1 = br1[0].ravel()
br2 = br2[0].ravel()
plt.subplot(2, 2, 1)
grad_cam.plot_sequence(br1, cam_out1)
plt.subplot(2, 2, 2)
grad_cam.plot_sequence(br2, cam_out2)
plt.subplot(2, 2, 3)
cam_out1 = cam_out1.ravel().astype(int)
cam_out2 = cam_out2.ravel().astype(int)
dtw_example_plot(br1, br2)
# DTW already performs matching between two sequences so theory is that we can match grad cam
# points. however, this doesn't take distance into account at all. So need to figure out
# some way to translate distances to gradcam units to determine if the regions are similar
# or not

d, cost_matrix, path = dtw(br1, br2, dist_only=False)
pathx = path[0].astype(int)
pathy = path[1].astype(int)
matches = {pathx[i]: pathy[i] for i in range(len(pathx))}
cam_dists = [abs(cam_out1[idx] - cam_out2[matches[idx]]) for idx in matches]
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.subplot(2, 2, 4)
plt.plot(cam_dists)
plt.show()

# do i want the cost matrix or the path?
dtw_distances = [cost_matrix[idx][matches[idx]] for idx in matches]
slopes = [np.true_divide((pathy[i] - pathy[i-1]), (pathx[i]-pathx[i-1])) for i in range(len(br1))]
subsequences = []
prev = None
for i, k in enumerate(slopes):
    if k == 1 and prev != 1:
        subsequences.append([i])
    elif k == 1:
        subsequences[-1].append(i)
    prev = k

# just using 5 as cutoff
subsequences = filter(lambda x: len(x) > 5, subsequences)
for seq in subsequences:
    plt.subplot(1, 3, 1)
    plt.plot(br1)
    plot_cam_sequence(seq, br1[seq], cam_out1[seq])
    plt.subplot(1, 3, 2)
    plt.plot(br2)
    # XXX this is kinda buggy. Not representing the right seq.
    plot_cam_sequence(seq, br2[seq], cam_out2[seq])
    cam_dist = np.array([abs(cam_out1[idx] - cam_out2[matches[idx]]) for idx in seq]).ravel()
    plt.subplot(1, 3, 3)
    plt.plot(cam_dist)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.show()

# maybe run the dtws, and just come up with a lower bound on the gradcam sum. Maybe
# its better if you just do a sample of them first and just plot a histogram
idx1 = patient_idxs.pop(random.choice(range(len(patient_idxs))))
all_cam_dists = []
sequence_pairs = []
for i, idx in enumerate(patient_idxs[:10]):
    rand_inst1 = random.randint(0, 19)
    cam_out1, br1 = grad_cam.get_single_sequence_grad_cam(idx, rand_inst1, target)
    br1 = br1[0].ravel()
    for jdx in patient_idxs[i+1:100]:
        for k in range(20):
            cam_out2, br2 = grad_cam.get_single_sequence_grad_cam(jdx, k, target)
            br2 = br2[0].ravel()
            d, cost_matrix, path = dtw(br1, br2, dist_only=False)
            pathx = path[0].astype(int)
            pathy = path[1].astype(int)
            matches = {pathx[n]: pathy[n] for n in range(len(pathx))}
            cam_dists = [abs(cam_out1[n] - cam_out2[matches[n]]) for n in matches]
            slopes = [np.true_divide((pathy[n] - pathy[n-1]), (pathx[n]-pathx[n-1])) for n in range(len(br1))]
            subsequences = []
            prev = None
            for n, slo in enumerate(slopes):
                if slo == 1 and prev != 1:
                    subsequences.append([n])
                elif slo == 1:
                    subsequences[-1].append(n)
                prev = slo

            # just using 5 as cutoff
            subsequences = filter(lambda x: len(x) > 5, subsequences)
            for seq in subsequences:
                cam_dist = sum([abs(cam_out1[n] - cam_out2[matches[n]]) for n in seq])
                all_cam_dists.append(cam_dist)
                if cam_dist <= 15 and sum(cam_out1[seq]) > 100:
                    sequence_pairs.append([seq, [matches[n] for n in seq], br1, br2])

import IPython; IPython.embed()
choices = [random.choice(sequence_pairs) for _ in range(10)]
for seq1, match2, br1, br2 in choices:
    cam_out1, _ = grad_cam.get_camout_for_breath(br1, target)
    cam_out2, _ = grad_cam.get_camout_for_breath(br2, target)
    plt.subplot(1, 3, 1)
    plt.plot(br1)
    plot_cam_sequence(seq1, br1[seq1], cam_out1[seq1])
    plt.subplot(1, 3, 2)
    plt.plot(br2)
    plot_cam_sequence(match2, br2[match2], cam_out2[match2])
    cam_dist = abs(cam_out1[seq1] - cam_out2[match2]).ravel()
    plt.subplot(1, 3, 3)
    plt.plot(cam_dist)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.show()
