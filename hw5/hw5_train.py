import csv
import sys
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
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
    rating = data[:,3]

    return userID, num_u, movieID, num_m, rating

userID, num_u, movieID, num_m, rating = load_data(sys.argv[1])

l = 64

user = Input(shape=(1,))
user_embed = Embedding(num_u, l)(user)
user_out = Dropout(0.5)(Flatten()(user_embed))
movie = Input(shape=(1,), name = 'movie')
movie_embed = Embedding(num_m, l)(movie)
movie_out = Dropout(0.5)(Flatten(name = 'movie_out')(movie_embed))

out_dot = Dot(axes=1)([user_out, movie_out])

user_bias = Embedding(num_u, 1)(user)
user_bias = (Flatten()(user_bias))
movie_bias = Embedding(num_m, 1)(movie)
movie_bias = (Flatten()(movie_bias))

output = Add()([out_dot, user_bias, movie_bias])

model = Model(inputs=[user, movie], outputs=output)

model.compile(loss='mse', optimizer='adam')


# Generate order of exploration of dataset
i = np.random.permutation(userID.shape[0])
userID = userID[i]
movieID = movieID[i]
rating = rating[i]

epochs = 50
batch_size = 1024
checkpoint = ModelCheckpoint('model.h5',monitor = 'val_loss',save_best_only = True)

hist = model.fit([userID, movieID], rating, 
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.1,
                callbacks=[checkpoint])
