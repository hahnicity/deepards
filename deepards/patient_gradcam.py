import argparse
import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
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
        self.grad_cam = GradCam(pretrained_model)
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
        patient_idxs = self.gt[self.gt.patient == patient_id].index
        target = self.gt.loc[patient_idxs].y.iloc[0]
        batch_size = 20
        mapping = {0: 'non_ards', 1: 'ards'}
        dirname = os.path.join('gradcam_results', 'sampled_sequences', mapping[target], patient_id)
        do_makedirs(dirname)

        for i in patient_idxs:
            rand_seq = random.choice(range(batch_size))
            breath_sequence = np.expand_dims(self.data[i][1][rand_seq], axis=0)
            br = torch.FloatTensor(breath_sequence)[[0] * batch_size].cuda()
            cam = self.grad_cam.generate_cam(br, target)
            cam_outputs = cv2.resize(cam, (1,224))
            filename = os.path.join(dirname, 'seq-{}-{}.png'.format(i, rand_seq))
            self.visualize_sequence(breath_sequence[0], cam_outputs, patient_id, target, filename)

    def visualize_sequence(self, br, cam_outputs, patient_id, c, filepath):
        img = br.reshape((224, 1))
        t = np.arange(0, 224, 1)
        plt.scatter(t, img, c=cam_outputs, vmin = 0, vmax = 255)
        plt.plot(t, img)
        cbar  = plt.colorbar()
        cbar.set_label("cam_outputs", labelpad=-1)
        mapping = {0: 'Non-ARDS', 1: 'ARDS'}
        plt.title(patient_id + ' ' + mapping[c])
        plt.savefig(filepath)
        plt.close()

    def do_all_patient_cam_ops(self):
        """
        Convenience method for doing everything we can
        """
        for patient_id in np.append(self.ards, self.non_ards):
            self.get_median_patient_camout(patient_id)
            self.get_sampled_patient_sequences_camout(patient_id)
            self.get_average_patient_camout(patient_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='path to the saved_model')
    parser.add_argument('-pdp', '--pickled-data-path', help = 'PATH to pickled data', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--ops', choices=['all', 'averages', 'medians', 'sample_seqs'], required=True)
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
