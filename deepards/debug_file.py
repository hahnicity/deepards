import os
import pickle
import numpy as np
#print(os.path.join(os.path.dirname(__file__), 'results/{}_1'.format(2)


data = pickle.load( open( "data.pkl", "rb" ) )
print(len(data))