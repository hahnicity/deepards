import argparse
import pickle

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


def get_ardsids_otherids(data):
    gt = data.get_ground_truth_df()
    ards = gt[gt.y == 1].patient.unique()
    non_ards = gt[gt.y == 0].patient.unique()
    return ards, non_ards


def get_avgbre_camout(patient_id, data,pretrained_model):
    #initializing variables
    gt = data.get_ground_truth_df()
    patient_idxs = gt[gt.patient == patient_id].index

    avg_breath = np.empty((0,224))
    cam_outputs = np.empty((0,7))
    target_class = None
    for i in patient_idxs:
        breath_sequence = data[i][1]
        #concadinating the 224 breath sequences to get the avg of the patient breath
        br1 = np.mean(breath_sequence, axis = 0)
        avg_breath = np.append(avg_breath, br1, axis = 0)
        #for gradcam values
        br = torch.FloatTensor(breath_sequence).cuda()
        grad_cam = GradCam(pretrained_model)
        cam = grad_cam.generate_cam(br, target_class)
        cam = np.expand_dims(cam, axis = 0)
        cam_outputs = np.append(cam_outputs, cam, axis = 0)
    cam_outputs = np.mean(cam_outputs, axis = 0)
    cam_outputs = cv2.resize(cam_outputs,(1,224))
    avg_breath = np.mean(avg_breath, axis = 0)
    return avg_breath,cam_outputs


def visualize_sequence(br, gradcam,patient_id,c):
    #inizializing the image variable to 224 pickles with 0
    img = np.zeros((224,1))
    br = np.expand_dims(br, axis = 0)
    #since the br is in 1 x 224 format we are transfering it into 224 x 1 format
    for i in range(0,224):
        img[i][0] = br[0][i]
    #arranging the x-axis since the data is of 224
    dt = 1
    t = np.arange(0, 224, dt)

    #plotting the imageto scalar va
    plt.scatter(t, img,c=gradcam,vmin = 0 , vmax = 255)
    plt.plot(t,img)
    cbar  = plt.colorbar()
    cbar.set_label("gradcam", labelpad=-1)
    mapping = {0: 'Non-ARDS', 1: 'ARDS'}
    plt.title(patient_id + ' ' + mapping[c])
    if c == 1:
        filename = 'gradcam_results' + '/ards/' + patient_id + '.png'
    else:
        filename = 'gradcam_results' + '/non_ards/' + patient_id + '.png'
    plt.savefig(filename)
    plt.close()
    #plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='path to the saved_model')
    parser.add_argument('-pdp', '--pickled-data-path', help = 'PATH to pickled data', required=True)
    parser.add_argument('--fold', type=int, required=True)
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

    ards,nonards = get_ardsids_otherids(data)
    #for ards patients
    for i in range(len(ards)):
        print('ARDS patient {}'.format(i))
        patient_id = ards[i]
        avg_br,cam_out = get_avgbre_camout(patient_id,data,pretrained_model)
        visualize_sequence(avg_br,cam_out,patient_id, 1)

    #for non_ards patients
    for i in range(len(nonards)):
        print('Non-ARDS patient {}'.format(i))
        patient_id = nonards[i]
        avg_br,cam_out = get_avgbre_camout(patient_id,data,pretrained_model)
        visualize_sequence(avg_br,cam_out,patient_id, 0)
