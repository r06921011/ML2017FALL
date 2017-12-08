import pandas as pd
import numpy as np
from keras.preprocessing import sequence, text
from keras.models import Sequential, load_model
from keras.layers.embeddings import Embedding
from keras.layers import GRU, Dense, Activation, Dropout, LSTM, Bidirectional,SpatialDropout1D
from keras.layers.convolutional import Convolution1D
import sys

# parameters
max_word_idx = 20000
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
#x_unlabeled = [line.rstrip('\n') for line in open('training_nolabel.txt', 'r', encoding='UTF-8')]

#tokenized
t = text.Tokenizer(num_words = max_word_idx,filters='\t\n')
t.fit_on_texts(x_train + x_test)
np.save('t.npy',t)
x_train = t.texts_to_sequences(x_train)
#x_unlabeled = t.texts_to_sequences(x_unlabeled)
x_test = t.texts_to_sequences(x_test)


#preprocess
x_train = sequence.pad_sequences(x_train, maxlen = max_sequence_len)
#x_unlabeled = sequence.pad_sequences(x_unlabeled, maxlen = max_sequence_len)
x_test = sequence.pad_sequences(x_test, maxlen = max_sequence_len)

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
model.add(Embedding(max_word_idx, embedding_vector_len, input_length = max_sequence_len))
model.add(SpatialDropout1D(0.2))
model.add(GRU(64, activation='tanh',dropout=0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()


model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary)

model.fit(x_train, y_train, nb_epoch = 2, batch_size = 64, validation_data = (x_v, y_v))
model.save('model_best.h5')

#############################################################################################

#selftrain
no = []
la = []
for n in range(len(p_big)):
    if p_big[n] == 1 :
        no.append(x_test[n,:])
        la.append(1)
    if p_small[n] == 1 :
        no.append(x_test[n,:])
        la.append(0)

no = np.asarray(no)
la = np.asarray(la).reshape(len(la),1)
print('add_data',la.shape)

x_train = np.concatenate((x_train,no),axis=0)
y_train = np.concatenate((y_train,la),axis=0)


# training
model.fit(x_train, y_train, nb_epoch = 1, batch_size = 64, validation_data = (x_v, y_v))
model.save('model_best1.h5')
