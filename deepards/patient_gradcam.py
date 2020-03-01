import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import numpy as np
import argparse
import cv2

#importing the class for calculating the cam values
from gradcam import GradCam

def get_ardsids_otherids(data):
    ards = np.zeros(shape=(0))
    non_ards = np.zeros(shape=(0))
    for i in range(len(data)):

        if data[i][2][1] == 1:
            if data[i][0] not in ards:
                ards = np.append(ards,data[i][0])

        elif data[i][2][1] == 0:
            if data[i][0] not in non_ards:
                non_ards = np.append(non_ards,data[i][0])

    return ards, non_ards

def get_avgbre_camout(patient_id, data,pretrained_model):
    #initializing variables
    avg_breath = np.empty((0,224))
    cam_outputs = np.empty((0,7))
    target_class = None
    for i in range(len(data)):
        if patient_id == data[i][0]:
            breath_sequence = data[i][1]
            #concadinating the 224 breath sequences to get the avg of the patient breath
            br1 = np.mean(breath_sequence, axis = 0)
            avg_breath = np.append(avg_breath, br1, axis = 0)
            #for gradcam values
            br = torch.FloatTensor(breath_sequence).cuda()
            grad_cam = GradCam(pretrained_model)
            cam = grad_cam.generate_cam(br, target_class)
            cam = np.expand_dims(cam, axis = 0)
            #print(cam)
            cam_outputs = np.append(cam_outputs, cam, axis = 0)
    cam_outputs = np.mean(cam_outputs, axis = 0)   
    cam_outputs = cv2.resize(cam_outputs,(1,224))
    avg_breath = np.mean(avg_breath, axis = 0)
    #print(cam_outputs)
    #print(len(cam_outputs))
    #print(cam_outputs.shape)
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
    plt.title(patient_id)
    if c == 1:
        filename = 'gradcam_results' + '/ards/' + patient_id + '.png'
    else:
        filename = 'gradcam_results' + '/non_ards/' + patient_id + '.png'
    plt.savefig(filename)
    plt.close()
    #plt.show()


if __name__ == '__main__':

    #space for putting all the required parsers
    parser = argparse.ArgumentParser()
    #add parser for pickle data
    parser.add_argument('-pd', '--pickled-data', help = 'PATH to pickled data')
    args = parser.parse_args()
    #loading the data
    data_filename = args.pickled_data
    data = pickle.load( open( data_filename, "rb" ) )
    #print(len(data))
    #print(data[0])

    #initializing the variables

    #Getting the pretrained model
    PATH = 'densenet-linear-model--fold-3.pth'
    pretrained_model = torch.load(PATH)
    #0015RPI0320150401 : example for patient id
    patient_id = '0015RPI0320150401'
    
    #getting all ardsids and nonards ids:
    ards,nonards = get_ardsids_otherids(data)
    c = 1
    
    #for ards patients
    for i in range(len(ards)):
        print(i)
        patient_id = ards[i]
        c = 1
        #itterating through the data to get the sequences of all the breaths of the particular patients
        avg_br,cam_out = get_avgbre_camout(patient_id,data,pretrained_model)
    
        #for visualizing the sequence for a patient id
        visualize_sequence(avg_br,cam_out,patient_id,c)
    
    #for non_ards patients
    for i in range(len(nonards)):
        print(i)
        patient_id = nonards[i]
        c = 0
        #itterating through the data to get the sequences of all the breaths of the particular patients
        avg_br,cam_out = get_avgbre_camout(patient_id,data,pretrained_model)

        #for visualizing the sequence for a patient id
        visualize_sequence(avg_br,cam_out,patient_id,c)