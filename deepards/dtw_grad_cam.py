import argparse
import random

import matplotlib.pyplot as plt
import pandas as pd
import torch

from deepards.patient_gradcam import PatientGradCam


parser = argparse.ArgumentParser()
parser.add_argument('model_path', help='path to saved model')
parser.add_argument('-pdp', '--pickled-data-path', help='pickled data path', required=True)
parser.add_argument('--fold', type=int, required=True)
args = parser.parse_args()

model = torch.load(args.model_path)
patient_id = '0099RPI0120151219'
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
plt.subplot(1, 2, 1)
grad_cam.plot_sequence(br1[0], cam_out1)
plt.subplot(1, 2, 2)
grad_cam.plot_sequence(br2[0], cam_out2)
plt.show()
