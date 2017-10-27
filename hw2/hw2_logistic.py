import pandas as pd
import numpy as np 
import pickle 
import csv, os, math
import sys

# read model
w = np.load('hw2_w.npy')
X = np.load('hw2_max.npy')


#real test data
test_data = pd.read_csv(sys.argv[5])
test_data = np.array(test_data)
test_data = np.delete(test_data,[1],axis=1)
test_data2 = np.concatenate((test_data,test_data**2),axis=1)
test_data = np.concatenate((test_data2,test_data**3),axis=1)
(t_data_num , feature_num) = test_data.shape
test_data = test_data / X


value = []

z = np.dot(test_data, w)
estimate = 1 / (1 + np.exp(-z))

for n in range(t_data_num):
    if estimate[n]>0.5 : value.append(1)
    else : value.append(0)

test_id=[]
for item in range(t_data_num):
    test_id.append(str(item+1))

df = pd.DataFrame({'id': test_id, 'label': value})
df.to_csv(sys.argv[6], index=False)
