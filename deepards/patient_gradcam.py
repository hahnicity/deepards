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


def visualize_sequence(br, gradcam):
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
    plt.show()


if __name__ == '__main__':

    #space for putting all the required parsers
    parser = argparse.ArgumentParser()
    #add parser for pickle data
    #add parser for patient id

    #initializing the variables
    
    #0015RPI0320150401 : example for patient id
    patient_id = '0015RPI0320150401'
    #Getting the pretrained model
    PATH = 'densenet-linear-model--fold-3.pth'
    #for avg breath original
    avg_breath = np.empty((0,224))
    pretrained_model = torch.load(PATH)
    target_class = None
    cam_outputs = np.empty((0,7))

    #loading the data

    data = pickle.load( open( "pickledata1.pkl", "rb" ) )
    #print(len(data))
    #print(data[0])

    #itterating through the data to get the sequences of all the breaths of the particular patients

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

    #for visualizing the sequence for a patient id
    visualize_sequence(avg_breath,cam_outputs)