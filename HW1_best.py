import pandas as pd
import numpy as np 
import pickle 
import csv, os, math
import sys

# read model
w = np.load('grad_2order.npy')
X_max = np.load('X_max.npy')

#real test data
t_data = pd.read_csv(sys.argv[1],encoding = 'big5',header = None)
t_data_matrix = np.array(t_data)
t_data_matrix = t_data_matrix[:,2:]
t_data_matrix[t_data_matrix == 'NR'] = 0

data = np.zeros((18*240,9))
for n in range(240):
    for j in range(18):
        data[18*n+j ,:] = t_data_matrix[18*n+j ,:]

data2 = data**2

value = []
t_vec = np.zeros((1,18*9*2+1))
for n in range(240):
        t_vec[0,0:18*9] = data[18*n:18*n+18,:].reshape(1,18*9)
        t_vec[0,18*9:18*9*2] = data2[18*n:18*n+18,:].reshape(1,18*9)
        t_vec[0,18*9*2] = 1
        t_vec = t_vec / X_max.reshape(1,18*9*2+1)
        estimate = np.dot(t_vec,w)
        value.append(estimate[0])

#value = value.reshape(240,1)

test_id=[]
for item in range(240):
    test_id.append('id_'+str(item))

df = pd.DataFrame({'id': test_id, 'value': value})
df.to_csv(sys.argv[2], index=False) 
