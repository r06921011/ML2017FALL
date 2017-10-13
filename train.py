import pandas as pd
import numpy as np 
import pickle 
import csv, os, math

# read data
all_data = pd.read_csv('train1.csv')
all_data_matrix = np.array(all_data)
all_data_matrix = all_data_matrix[:,3:]
(all_data_row , all_data_col) = all_data_matrix.shape
all_data_matrix[all_data_matrix == 'NR'] = 0

# reshape data
data_reshape = np.zeros((18,480,12))
data_reshape2 = np.zeros((18,480,12))
for month in range(12):
    for n in range(18):
        for date in range(20):
            data_reshape[ n , 24*date:24*(date+1) , month] = all_data_matrix[ 18*20*month+18*date+n , 0:24]
#data**2
data_reshape2 = data_reshape**2


# test data
test_data = np.zeros((18,80,12))
for month in range(12):
    for j in range(18):
        test_data[ j ,  0:40 , month] = data_reshape[ j ,  0:40 ,month]
        test_data[ j ,  40:80 , month] = data_reshape2[ j ,  0:40 ,month]

# training data
train_data = np.zeros((18,440*2,12))
for month in range(12):
    for j in range(18):
        train_data[ j , 0:440 , month] = data_reshape[ j , 40:480 ,month]
        train_data[ j , 440:880 , month] = data_reshape2[ j , 40:480 ,month]

X = np.zeros((431*12,18*9*2+1))
Ground = np.zeros((431*12))
train_vec = np.zeros((1,18*9*2))
for month in range(12):        
        for n in range(431):        
            train_vec[0,0:18*9] = train_data[:,n:n+9,month].reshape(1,18*9)
            train_vec[0,18*9:18*9*2] = train_data[:,n+440:n+440+9,month].reshape(1,18*9)
            X[n+431*month,0:18*9*2] = train_vec
            X[n+431*month,18*9*2] = 1.0
            Ground[n+431*month] = train_data[9,n+9,month]

X_max = np.max(X , axis=0)
X = X / X_max


#validation
X_test = np.zeros((12*31,18*9*2+1))
Ground_test = np.zeros((31*12))
test_vec = np.zeros((1,18*9*2))
for month in range(12):        
        for n in range(31):        
            test_vec[0,0:18*9] = test_data[:,n:n+9,month].reshape(1,18*9)
            test_vec[0,18*9:18*9*2] = test_data[:,n+40:n+40+9,month].reshape(1,18*9)
            X_test[n+31*month,0:18*9*2] = test_vec
            X_test[n+31*month,18*9*2] = 1.0
            Ground_test[n+31*month] = test_data[9,n+9,month]

X_test = X_test / X_max

#gradient
w = np.zeros((18*9*2+1))
w_lr = np.ones((18*9*2+1))
lr = 1
iteration =30000
lamda = 0.1

# iteration
L=0.0

for i in range(iteration):
    
    
    w_grad = np.zeros((18*9*2+1))
    
    estimate = np.dot(X, w)
    error = (Ground - estimate)

    for n in range(431*12):
        w_grad = w_grad - 2.0 * error[n] * X[n,:]
    
    w_grad = w_grad + lamda * w
    w_lr = w_lr + w_grad ** 2

    # Update parameters.
    w = w - lr/np.sqrt(w_lr) * w_grad

    #error
    L = np.sqrt(np.sum((Ground - estimate) ** 2 , axis=0)/ (431*12)) 

    estimate = np.dot(X_test,w)
    V = np.sqrt(np.sum((Ground_test - estimate) ** 2 , axis=0)/ (31*12))

    if i%100 == 0 :
        print('i:',i,'train error:' , L ,'validation error', V)



# save model
np.save('grad_2order.npy',w)
np.save('X_max.npy',X_max)
# read model
w = np.load('grad_2order.npy')
X_max = np.load('X_max.npy')


'''

#real test data
t_data = pd.read_csv('test.csv')
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
df.to_csv('10_12.csv', index=False) 

'''