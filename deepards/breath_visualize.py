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

parser = argparse.ArgumentParser()

parser.add_argument('-pdp', '--pickled-data-path', help = 'PATH to pickled data')
args = parser.parse_args()
#taking the pickle file we need to extract data from
pickle_file_name = args.pickled_data_path
data = pickle.load( open( pickle_file_name, "rb" ) )

#taking the first 100 sequence of the data
first_seq = data
for i in range(len(first_seq)):
    print('Patient: {} Class: {}'.format(first_seq[i][0], first_seq[i][2]))

#taking the first breath 
br = first_seq[0][0]

#arranging the x-axis since the data is of 224 
dt = 1
t = np.arange(0, 224, dt)

#inizializing the image variable to 224 pickles with 0
img = np.zeros((224,1))

#since the br is in 1 x 224 format we are transfering it into 224 x 1 format
for i in range(0,224):
    img[i][0] = br[0][i]


#plotting the image
plt.plot(t, img)
plt.show()
