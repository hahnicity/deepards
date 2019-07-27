import os
import pickle
import numpy as np
#print(os.path.join(os.path.dirname(__file__), 'results/{}_1'.format(2)


data = pickle.load( open( "pickledata1.pkl", "rb" ) )
#print(len(data))
#print(data[0])
#0015RPI0320150401 : example for patient id
for i in range(len(data)):
    print(type(data[i][0])) 
    print(data[i+1][0])
    break