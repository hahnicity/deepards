import os
import pickle
import numpy as np
#print(os.path.join(os.path.dirname(__file__), 'results/{}_1'.format(2)


data = pickle.load( open( "pickledata1.pkl", "rb" ) )
print(len(data))
#print(data[0])
#0015RPI0320150401 : example for patient id
ards = np.zeros(shape=(0))
non_ards = np.zeros(shape=(0))
count = 0
for i in range(len(data)):
    
    if data[i][2][1] == 1:
        count = count + 1
        if data[i][0] not in ards:
            ards = np.append(ards,data[i][0])
    
    elif data[i][2][1] == 0:
        count = count + 1
        if data[i][0] not in non_ards:
            non_ards = np.append(non_ards,data[i][0])

#print(ards)
#print(non_ards)
print(count)
   