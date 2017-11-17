import numpy as np
import pandas as pd
import sys
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

model = load_model('model_best.h5')
model.summary()

data = pd.read_csv(sys.argv[1], sep = ',', encoding = 'UTF-8')
data = data.as_matrix()
id = data[:,0]
test = list()
for i in range(len(id)):
	x = data[i][1].split()
	test.append(x)
test = np.array(test).astype(int)
test = test.reshape(-1,48,48,1)
test = test/255

result = model.predict(test)
result = result.argmax(axis=-1)

output = pd.DataFrame({'id': id,'label':result})
output.to_csv(sys.argv[2], index = False) 
