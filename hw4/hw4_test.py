import pandas as pd
import numpy as np
import pickle
from keras.preprocessing import sequence, text
from keras.models import Sequential, load_model
from keras.layers.embeddings import Embedding
from keras.layers import GRU, Dense, Activation, Dropout, LSTM
from keras.layers.convolutional import Convolution1D
import sys

# parameters
max_word_idx = 20000
max_sequence_len = 30
num_v = 20000
embedding_vector_len = 128
threshold = 0.5

#read test data
x_test = [line.rstrip('\n') for line in open(sys.argv[1], 'r', encoding='UTF-8')]
x_test = [line.split(',', 1)[1] for line in x_test]
del x_test[0]

with open('t.pickle', 'rb') as h:
	t = pickle.load(h)

x_test = t.texts_to_sequences(x_test)
x_test = sequence.pad_sequences(x_test, maxlen = max_sequence_len)
x_test = np.asarray(x_test)

model = load_model('model_best.h5') 
predict = model.predict(x_test).reshape(-1,1)
predict = (predict > threshold).astype(int)

# generate prediction file
id_col = np.array([str(i) for i in range(predict.shape[0])]).reshape(-1,1)
output = np.hstack((id_col, predict))
output_df = pd.DataFrame(data = output, columns = ['id', 'label'])
output_df.to_csv(sys.argv[2], index = False)