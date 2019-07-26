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

#from gradcam import get_sequence

#args = parser.parse_args()
#taking the pickle file we need to extract data from
#pickle_file_name = args.pickled_data_path
#data = pickle.load( open( pickle_file_name, "rb" ) )

#taking the first 100 sequence of the data
#first_seq = data[0]
#for i in range(len(first_seq)):
    #print('Patient: {} Class: {}'.format(first_seq[i][0], first_seq[i][2]))

#print(len(first_seq))
#taking the first breath 
#br = first_seq[1][0]
#br = np.expand_dims(br, 0)

#print(br.shape)

def get_avg_breath(data, breath_class):
    avg_breath = np.empty((0,224))
    for i in range(100):
        breath_sequence = get_sequence(data, breath_class, i)
        breath_sequence = np.mean(breath_sequence, axis = 0)
        avg_breath = np.append(avg_breath, breath_sequence, axis = 0)
    
    #print(avg_breath.shape)
    avg_breath = np.mean(avg_breath, axis = 0)
    return avg_breath

def get_gradcam(gradcam_filename):
    gradcam = pickle.load(open(gradcam_filename,'rb'))
    return gradcam

def get_sequence(data, ards, c):
    #print(len(data))
    first_seq = None
    count = 0
    for i, d in enumerate(data):
        #print("test")
        if d[2][1] == int(ards):
            if count == c:
                first_seq = d
                break
            count = count + 1
            continue 
    #first_seq = data[1]
    #print(first_seq[2])
    br = first_seq[1]
    #br = np.expand_dims(br, 0)
    #br = torch.from_numpy(br)
    return br

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

parser = argparse.ArgumentParser()
parser.add_argument('-pd', '--pickled-data', help = 'PATH to pickled data')
parser.add_argument('-pg', '--pickled-gradcam', help = 'PATH to pickled gradcam')
parser.add_argument('-t', '--type', help = 'Others[0] or Ards[1]')

args = parser.parse_args()

gradcam_filename = args.pickled_gradcam
data_filename = args.pickled_data
breath_class = args.type
data = pickle.load( open( data_filename, "rb" ) )
br = get_avg_breath(data, breath_class)
gradcam = get_gradcam(gradcam_filename)

visualize_sequence(br, gradcam)

