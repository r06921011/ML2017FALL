import csv
import sys
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Add, Dot, Flatten, Dropout, Merge
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras import backend as K

data = np.load(sys.argv[1])

label = []
for i in range(data.shape[0]):
    num = 0
    nz = np.count_nonzero(data[i,:])
    image = data[i,:].reshape(28,28)
    col = np.count_nonzero(image[0,:])
    row = np.count_nonzero(image[:,0])
    row1 = np.count_nonzero(image[:,27])

    if col < 2 and row < 2 and row1 < 2 and nz < 300 : label.append(0)
    else : label.append(1)
    if (100*i/data.shape[0]) % 10 == 0 : print('i=',100*i/data.shape[0])

def load_data(fileName):
    data = pd.read_csv(fileName, encoding='UTF-8')
    data = data.as_matrix()

    image1 = data[:,1]
    image2 = data[:,2]

    return image1, image2

image1, image2 = load_data(sys.argv[2])

data_num = image1.shape[0]
print('data_num',data_num)

predict = []
n = 0
for i in range(data_num):
	a = image1[i]
	b = image2[i]
	if label[a] == label[b]:
		predict.append(1)
		n = n+1
	if label[a] != label[b] : predict.append(0)

predict = np.asarray(predict).reshape(-1,1)

print('num_1:',n)

lab = 0
for i in range(140000):
	if label[i] == 0 : lab = lab+1
print('label_0:',lab)

# generate prediction file
id_col = np.array([str(i) for i in range(data_num)]).reshape(-1,1)
output = np.hstack((id_col, predict))
output_df = pd.DataFrame(data = output, columns = ['ID', 'Ans'])
output_df.to_csv(sys.argv[3], index = False)