import numpy as np
import pandas as pd
import sys
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

data = pd.read_csv(sys.argv[1], sep = ',' ,encoding = 'UTF-8')
data = data.as_matrix()
y_data = data[:,0]
x_data = list()
for i in range(len(y_data)):
    x = data[i][1].split()
    x_data.append(x)
x_data = np.array(x_data).astype(int)
x_data = x_data.reshape(-1,48,48,1)
x_data = x_data / 255
y_data = np_utils.to_categorical(y_data,7)


datagen2 = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
datagen2.fit(x_data)

datagen1 = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)
datagen1.fit(x_data)

datagen3 = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True)
datagen3.fit(x_data)

model = Sequential()

model.add(ZeroPadding2D((1,1),input_shape=(48,48,1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))


model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=3e-4), metrics=['accuracy'])

checkpoint = ModelCheckpoint('model.h5',monitor = 'val_acc',save_best_only = True)


model.fit_generator(datagen2.flow(x_data, y_data, batch_size=64), steps_per_epoch=len(x_data) / 64, epochs=10)
model.fit_generator(datagen1.flow(x_data, y_data, batch_size=64), steps_per_epoch=len(x_data) / 64, epochs=10)
model.fit_generator(datagen3.flow(x_data, y_data, batch_size=64), steps_per_epoch=len(x_data) / 64, epochs=10)
model.fit(x_data,y_data,batch_size = 128, epochs = 10,validation_split = 0.1)

model.fit_generator(datagen3.flow(x_data, y_data, batch_size=128), steps_per_epoch=len(x_data) / 128, epochs=5)
model.fit_generator(datagen2.flow(x_data, y_data, batch_size=128), steps_per_epoch=len(x_data) / 128, epochs=5)
model.fit_generator(datagen1.flow(x_data, y_data, batch_size=128), steps_per_epoch=len(x_data) / 128, epochs=10)
model.fit(x_data,y_data,batch_size = 512, epochs = 3,validation_split = 0.1,callbacks = [checkpoint])

