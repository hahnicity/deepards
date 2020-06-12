import argparse
import math
import os
import pickle
import random

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

#importing the class for calculating the cam values
from gradcam import GradCam


def do_makedirs(dir):
    try:
        os.makedirs(dir)
    except OSError:
        pass


class PatientGradCam(object):
    def __init__(self, pretrained_model, data):
        self.grad_cam = GradCam(pretrained_model.cuda())
        self.data = data
        self.gt = self.data.get_ground_truth_df()
        self.ards, self.non_ards = self.get_ardsids_otherids()

    def get_ardsids_otherids(self):
        ards = self.gt[self.gt.y == 1].patient.unique()
        non_ards = self.gt[self.gt.y == 0].patient.unique()
        return ards, non_ards

    def get_median_patient_camout(self, patient_id):
        #initializing variables
        patient_idxs = self.gt[self.gt.patient == patient_id].index
        target = self.gt.loc[patient_idxs].y.iloc[0]
        mapping = {0: 'non_ards', 1: 'ards'}
        dirname = os.path.join('gradcam_results', 'patient_medians', mapping[target])
        do_makedirs(dirname)

        # XXX in future make this configurable
        batch_size = 20
        med_breath = np.empty((0, batch_size, 1, 224))
        cam_outputs = np.empty((0,7))
        target_class = None
        for i in patient_idxs:
            breath_sequence = np.expand_dims(self.data[i][1], axis=0)
            med_breath = np.append(med_breath, breath_sequence, axis=0)

        med_breath = np.median(np.median(med_breath, axis=0), axis=0)
        med_breath = np.expand_dims(med_breath, axis=0)
        br = torch.FloatTensor(med_breath)[[0] * batch_size].cuda()
        cam = self.grad_cam.generate_cam(br, target)
        cam_outputs = cv2.resize(cam, (1,224))
        filename = os.path.join(dirname, patient_id + '.png')
        self.visualize_sequence(med_breath[0], cam_outputs, patient_id, target, filename)

    def get_average_patient_camout(self, patient_id):
        #initializing variables
        patient_idxs = self.gt[self.gt.patient == patient_id].index
        target = self.gt.loc[patient_idxs].y.iloc[0]
        mapping = {0: 'non_ards', 1: 'ards'}
        dirname = os.path.join('gradcam_results', 'patient_averages', mapping[target])
        do_makedirs(dirname)

    	avg_breath = np.empty((0,224))
    	cam_outputs = np.empty((0,7))

        for i in patient_idxs:
            breath_sequence = data[i][1]
            #concadinating the 224 breath sequences to get the avg of the patient breath
            br1 = np.mean(breath_sequence, axis = 0)
            avg_breath = np.append(avg_breath, br1, axis = 0)
            #for gradcam values
            br = torch.FloatTensor(breath_sequence).cuda()
            cam = self.grad_cam.generate_cam(br, target)
            cam = np.expand_dims(cam, axis = 0)
            cam_outputs = np.append(cam_outputs, cam, axis = 0)
        cam_outputs = np.mean(cam_outputs, axis = 0)
        cam_outputs = cv2.resize(cam_outputs,(1,224))
        avg_breath = np.mean(avg_breath, axis = 0)
        filename = os.path.join(dirname, patient_id + '.png')
        self.visualize_sequence(avg_breath, cam_outputs, patient_id, target, filename)

    def get_sampled_patient_sequences_camout(self, patient_id):
        """
        Idea is that we iterate over patient batches and sample a single sequence
        from each batch. That sequence is fed into gradcam for a result
        """
        # XXX this line is failing us we need to adjust indexes to be a continuous range
        # instead of a potentially discontinuous one
        patient_idxs = self.gt[self.gt.patient == patient_id].index
        target = self.gt.loc[patient_idxs].y.iloc[0]
        batch_size = 20
        mapping = {0: 'non_ards', 1: 'ards'}
        dirname = os.path.join('gradcam_results', 'sampled_sequences', mapping[target], patient_id)
        do_makedirs(dirname)

        for i in patient_idxs:
            rand_seq = random.choice(range(batch_size))
            cam_outputs, br = self.get_single_sequence_grad_cam(i, rand_seq, target)
            filename = os.path.join(dirname, 'seq-{}-{}.png'.format(i, rand_seq))
            self.visualize_sequence(breath_sequence[0], cam_outputs, patient_id, target, filename)

    def plot_sequence(self, br, cam_outputs):
        img = br.reshape((224, 1))
        t = np.arange(0, 224, 1)
        plt.scatter(t, img, c=cam_outputs, vmin = 0, vmax = 255)
        plt.plot(t, img)

    def visualize_sequence(self, br, cam_outputs, patient_id, c, filepath):
        self.plot_sequence(br, cam_outputs)
        cbar  = plt.colorbar()
        cbar.set_label("cam_outputs", labelpad=-1)
        mapping = {0: 'Non-ARDS', 1: 'ARDS'}
        plt.title(patient_id + ' ' + mapping[c])
        plt.savefig(filepath)
        plt.close()

    def get_single_sequence_grad_cam(self, seq_idx, batch_idx, target):
        item = self.data[seq_idx]
        batch_size = item[1].shape[0]
        br = np.expand_dims(item[1][batch_idx], axis=0)
        br = torch.FloatTensor(br)[[0] * batch_size].cuda()
        cam = self.grad_cam.generate_cam(br, target)
        cam_outputs = cv2.resize(cam, (1,224))
        return cam_outputs, br.cpu().numpy()

    def _do_patho_rand_sample(self, patho, filename):
        items_per_frame = 16
        batch_size = 20
        idxs = len(self.data)
        counter = 0
        target = {'ards': 1, 'non_ards': 0}[patho]
        self.gt.index = range(len(self.gt))
        patho_idxs = self.gt[self.gt.y == target].index
        for j in range(items_per_frame):
            seq_idx = random.choice(patho_idxs)
            br_idx = random.randint(0, batch_size-1)
            plt.subplot(int(math.sqrt(items_per_frame)), int(math.sqrt(items_per_frame)), j+1)
            cam_outputs, br = self.get_single_sequence_grad_cam(seq_idx, br_idx, target)
            self.plot_sequence(br[0].cpu().numpy(), cam_outputs)
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            plt.yticks(fontsize='x-small')

        fig = plt.gcf()
        sm = fig.axes[0].pcolormesh(np.random.random((0, 0)), vmin=0, vmax=255)
        fig.set_size_inches(20, 10)
        fig.subplots_adjust(right=.8)
        cbar_ax = fig.add_axes((0.85, 0.15, 0.025, 0.7))
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Intensity")
        plt.suptitle("Random Sample {} Grad-Cam".format({'non_ards': 'Non-ARDS', 'ards': 'ARDS'}[patho]))
        plt.savefig(filename, dpi=400)
        plt.close()

    def rand_sample(self):
        dirname = os.path.join('gradcam_results', 'rand_sample')
        do_makedirs(dirname)
        for i in range(3):
            filename = os.path.join(dirname, "ards-rand-samp-{}.png".format(i))
            self._do_patho_rand_sample('ards', filename)

        for j in range(3):
            filename = os.path.join(dirname, "non-ards-rand-samp-{}.png".format(j))
            self._do_patho_rand_sample('non_ards', filename)

    def do_all_patient_cam_ops(self):
        """
        Convenience method for doing everything we can
        """
        for patient_id in np.append(self.ards, self.non_ards):
            self.get_median_patient_camout(patient_id)
            self.get_sampled_patient_sequences_camout(patient_id)
            self.get_average_patient_camout(patient_id)
        self.rand_sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='path to the saved_model')
    parser.add_argument('-pdp', '--pickled-data-path', help = 'PATH to pickled data', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--ops', choices=['all', 'averages', 'medians', 'sample_seqs', 'rand_sample'], required=True)
    args = parser.parse_args()

    data = pd.read_pickle(args.pickled_data_path)
    data.set_kfold_indexes_for_fold(args.fold)
    data.transforms = None
    pretrained_model = torch.load(args.model_path)
    # ensure that model is on same cuda device that data will be on
    if not isinstance(pretrained_model, torch.nn.DataParallel):
        pretrained_model = pretrained_model.to(torch.cuda.current_device())
    else:
        pretrained_model = pretrained_model.module

    pt_grad = PatientGradCam(pretrained_model, data)
    if args.ops == 'all':
        pt_grad.do_all_patient_cam_ops()
    elif args.ops == 'medians':
        for patient_id in np.append(pt_grad.ards, pt_grad.non_ards):
            pt_grad.get_median_patient_camout(patient_id)
    elif args.ops == 'sample_seqs':
        for patient_id in np.append(pt_grad.ards, pt_grad.non_ards):
            pt_grad.get_sampled_patient_sequences_camout(patient_id)
    elif args.ops == 'averages':
        for patient_id in np.append(pt_grad.ards, pt_grad.non_ards):
            pt_grad.get_average_patient_camout(patient_id)
    elif args.ops == 'rand_sample':
        pt_grad.rand_sample()
