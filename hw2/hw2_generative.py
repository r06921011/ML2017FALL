import pandas as pd
import numpy as np 
import csv, os, math
import sys

# read train data
all_data = pd.read_csv(sys.argv[3])
all_data_matrix = np.array(all_data)
all_data_matrix = np.delete(all_data_matrix,[1],axis=1)
(data_num , feature_num) = all_data_matrix.shape
X = np.max(all_data_matrix , axis=0)
all_data_matrix = all_data_matrix / X

#real test data
test_data = pd.read_csv(sys.argv[5])
test_data = np.array(test_data)
test_data = np.delete(test_data,[1],axis=1)
(t_data_num , feature_num) = test_data.shape
test_data = test_data / X

# read ground truth
Ground = pd.read_csv(sys.argv[4])
Ground = np.array(Ground)
Ground = (Ground).ravel()

# Class1 and Class2
class1 = all_data_matrix[Ground==1,:]
class2 = all_data_matrix[Ground==0,:]
(class1_num,feature_num) = class1.shape
(class2_num,feature_num) = class2.shape

# define means and covariance
mean1 = np.mean(class1,axis=0).reshape((-1,1))
mean2 = np.mean(class2,axis=0).reshape((-1,1))
cov1 = np.cov(class1,rowvar=False)
cov2 = np.cov(class2,rowvar=False)
cov = (class1_num/data_num)*cov1+(class2_num/data_num)*cov2

#Gaussian distribution
z1 = np.dot(np.dot((mean1-mean2).T , np.linalg.pinv(cov)),test_data.T)
z2 = - 0.5*np.dot(np.dot((mean1).T , np.linalg.pinv(cov)),mean1)
z3 = + 0.5*np.dot(np.dot((mean2).T , np.linalg.pinv(cov)),mean2)
z4 = + np.log(class1_num/class2_num)
z = z1 + z2 + z3 + z4
estimate = 1 / (1 + np.exp(-z))


value = []

for n in range(t_data_num):
    if estimate[0,n]>0.5 : value.append(1)
    else : value.append(0)

test_id=[]
for item in range(t_data_num):
    test_id.append(str(item+1))

df = pd.DataFrame({'id': test_id, 'label': value})
df.to_csv(sys.argv[6], index=False) 
