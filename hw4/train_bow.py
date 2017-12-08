import pandas as pd
import numpy as np
from keras.preprocessing import sequence, text
from keras.models import Sequential, load_model
from keras.layers.embeddings import Embedding
from keras.layers import GRU, Dense, Activation, Dropout, LSTM, Bidirectional
from keras.layers.convolutional import Convolution1D
import time
import sys

# parameters
max_word_idx = 3000
max_sequence_len = 30
num_v = 20000
embedding_vector_len = 128
threshold = 0.5

#read train data
raw_data = pd.read_csv(sys.argv[1], sep = '\+\+\+\$\+\+\+' ,encoding = 'UTF-8', engine = 'python')
raw_data = raw_data.as_matrix()
y_train = raw_data[:,0].tolist()
num = len(y_train)
x_train = raw_data[:,1].tolist()

#read test data
x_test = [line.rstrip('\n') for line in open('testing_data.txt', 'r', encoding='UTF-8')]
x_test = [line.split(',', 1)[1] for line in x_test]
del x_test[0]

#tokenized
t = text.Tokenizer(num_words = max_word_idx)
t.fit_on_texts(x_train + x_test)
x_train = t.texts_to_matrix(x_train,mode='count')
#x_unlabeled = t.texts_to_matrix(x_unlabeled,mode='count')
x_test = t.texts_to_matrix(x_test,mode='count')
x_test1 = t.texts_to_matrix(x_test1,mode='count')

x_train = np.asarray(x_train)
y_train = np.asarray(y_train).reshape(-1,1)
#x_unlabeled = np.asarray(x_unlabeled)
x_test = np.asarray(x_test)

#validation
x_v = x_train[0:num_v]
y_v = y_train[0:num_v]
x_train = x_train[num_v:]
y_train = y_train[num_v:]

# build model
model = Sequential()
model.add(Dense(1024,input_shape=(max_word_idx,),activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()


model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary)

model.fit(x_train, y_train, nb_epoch = 2, batch_size = 64, validation_data = (x_v, y_v))
model.save('model_bow.h5')

