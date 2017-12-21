import csv
import sys
import numpy as np
import pandas as pd
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Embedding, Add, Dot, Flatten, Dropout, Merge
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras import backend as K

def load_data(fileName):
    data = pd.read_csv(fileName, encoding='UTF-8')
    data = data.as_matrix()

    userID = data[:,1]-1
    num_u = np.max(userID)+1
    movieID = data[:,2]-1
    num_m = np.max(movieID)+1

    return userID, num_u, movieID, num_m

model = load_model('model_k64.h5')
userID, num_u, movieID, num_m = load_data(sys.argv[1])

predict = model.predict([userID, movieID])
del model


# generate prediction file
id_col = np.array([str(i+1) for i in range(predict.shape[0])]).reshape(-1,1)
output = np.hstack((id_col, predict))
output_df = pd.DataFrame(data = output, columns = ['TestDataID', 'Rating'])
output_df.to_csv(sys.argv[2], index = False)