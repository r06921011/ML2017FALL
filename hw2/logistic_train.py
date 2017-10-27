import pandas as pd
import numpy as np 
import pickle 
import csv, os, math
import sys

# read train data
all_data = pd.read_csv(sys.argv[3])
all_data_matrix = np.array(all_data)
all_data_matrix = np.delete(all_data_matrix,[1],axis=1)
all_data_matrix2 = np.concatenate((all_data_matrix , all_data_matrix**2) ,axis=1)
all_data_matrix = np.concatenate((all_data_matrix2 , all_data_matrix**3) ,axis=1)
(data_num , feature_num) = all_data_matrix.shape
X = np.max(all_data_matrix , axis=0)
all_data_matrix = all_data_matrix / X


# read ground truth
Ground = pd.read_csv(sys.argv[4])
Ground = np.array(Ground)


#gradient
b ,b_lr = 0 ,0
w = np.zeros((feature_num))
w_lr = np.ones((feature_num))
lr = 1
iteration = 10000
lamda = 0.1

# iteration
L=0.0

for i in range(iteration):
    
    b_grad = 0
    w_grad = np.zeros((feature_num))   
    
    z = np.dot(all_data_matrix, w)
    estimate = 1 / (1 + np.exp(-z))

    for n in range(data_num):
        
        b_grad = b_grad - (Ground[n] - estimate[n])
        w_grad = w_grad - (Ground[n] - estimate[n]) * all_data_matrix[n,:]


    b_grad = b_grad + lamda * b
    w_grad = w_grad + lamda * w
    b_lr = b_lr + b_grad ** 2
    w_lr = w_lr + w_grad ** 2

    # Update parameters.
    b = b - lr/np.sqrt(b_lr) * b_grad
    w = w - lr/np.sqrt(w_lr) * w_grad


    if i % 100 == 0 :
        T = 0
        for n in range(data_num):
            if (estimate[n]>0.5  and Ground[n]==1): T=T+1
            if (estimate[n]<=0.5 and Ground[n]==0): T=T+1
        A = T / data_num
        print('i:',i,'accuracy:' , A)


# save model
np.save('hw2_w.npy',w)
np.save('hw2_max.npy',X)
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
