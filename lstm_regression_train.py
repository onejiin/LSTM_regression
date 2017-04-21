from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Convolution1D, MaxPooling1D
from keras.datasets import imdb
# TimeDistributed
from keras.layers import TimeDistributed, TimeDistributedDense, Flatten


import shutil
from random import shuffle
from function import *

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

def padding_data(data):
    data = sequence.pad_sequences(data, dtype='float', maxlen=maxlen)
    #data = data.reshape((len(data), maxlen, 1, 172, 224))
    data = data.reshape((len(data), maxlen, 172*224))

    return data


def padding_label(data):
    #data = sequence.pad_sequences(data, maxlen=maxlen) # heat map int type
    data = sequence.pad_sequences(data, dtype='float', maxlen=maxlen)
    #data = data.reshape((len(data), maxlen, 1*3))
    data = data.reshape((len(data), maxlen, 1*2))

    return data


def model_define():
    model = Sequential()
#    model.add(TimeDistributed(Convolution2D(nb_filter, 3, 3, border_mode="same"),
#                              input_shape=[lstm_output_size, 1, 172, 224]))
#    model.add(TimeDistributed(Activation("relu")))
#    model.add(TimeDistributed(MaxPooling2D(pool_size=pool_size)))
#    print (model.output_shape)

    model.add(LSTM(256, return_sequences=True,
                   input_shape=(lstm_output_size, 172*224)))
    print(model.output_shape)
    model.add(LSTM(128, return_sequences=True))
    print(model.output_shape)
    model.add(LSTM(32, return_sequences=True))
    print(model.output_shape)
    model.add(TimeDistributedDense(2))
    model.add(Activation('linear'))
    print(model.output_shape)

    return model


# --------------------------- DATA LOAD ------------------------------- #
READ_train = './folder_list_train.scp'
READ_test = './folder_list_test.scp'

f = open(READ_train)
lines = f.readlines()
tot_sequence = []
for line in lines:
    dirname = line.strip('\r\n')
    tot_sequence += Sequence_array(dirname)
f.close()
print('total train data', len(tot_sequence))

# shuffling total database,
shuffle(tot_sequence)
(X_train, y_train) = load_data(tot_sequence)
print('load train database...')

# model define
print('Build model...')
model = model_define()
#model.compile(loss='mean_squared_error', optimizer='adam')
model.compile(optimizer='rmsprop', loss='mse')

print ("Num. of iter. per 1 epoch", X_train.shape[0]/batch_size)

for epoch in range(100):
    print ("epoch is", epoch)

    cnt = 0
    for index in range(int(X_train.shape[0]/batch_size)):
        X_batch = X_train[index*batch_size:(index+1)*batch_size]
        y_batch = y_train[index*batch_size:(index+1)*batch_size]

        X_batch = padding_data(X_batch)
        y_batch = padding_label(y_batch)

        loss = model.train_on_batch(X_batch, y_batch)
        print ("batch %d loss : %f" % (cnt, loss))

        #[X_batch_padding, return_cnt] = Batch_input(batch_size, X_batch, cnt)
        #cnt = return_cnt
        cnt += 1

    #remain batch index
    X_batch = X_train[cnt * batch_size:]
    y_batch = y_train[cnt * batch_size:]
    X_batch = padding_data(X_batch)
    y_batch = padding_label(y_batch)

    loss = model.train_on_batch(X_batch, y_batch)
    print("batch %d loss : %f" % (cnt, loss))

    if epoch % 10 == 0:
        weight_name = str("weight") + str(epoch)
        model.save_weights(weight_name, True)

model.save_weights("weight", True)