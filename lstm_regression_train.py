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
import h5py

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
    model.add(TimeDistributed(Convolution2D(nb_filter, 3, 3, border_mode="same"),
                              input_shape=[lstm_output_size, 1, 172, 224]))
    model.add(TimeDistributed(Activation("relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=pool_size)))
    print (model.output_shape)

    #
    model.add(TimeDistributed(Convolution2D(1, 3, 3, border_mode="same")))
    model.add(TimeDistributed(Activation("relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=pool_size)))
    print(model.output_shape)
    #

    model.add(TimeDistributed(Flatten()))
    print(model.output_shape)

#    model.add(LSTM(256, return_sequences=True,
#                   input_shape=(lstm_output_size, 172*224)))
    model.add(LSTM(256, return_sequences=True,
                   input_shape=(lstm_output_size, (172/4)*(224/4))))
    print(model.output_shape)
    model.add(LSTM(128, return_sequences=True))
    print(model.output_shape)
    model.add(LSTM(32, return_sequences=True))
    print(model.output_shape)
    model.add(TimeDistributedDense(2))
    model.add(Activation('linear'))
    print(model.output_shape)

    return model


print('Build model...')
model = model_define()
#model.compile(loss='mean_squared_error', optimizer='adam')
model.compile(optimizer='rmsprop', loss='mse')

# --------------------------- Change HDF5 ------------------------------- #
READ_HDF5_train = './folder_list_small_set_HDF5.scp'

f = open(READ_HDF5_train)
lines = f.readlines()
tot_sequence = []

num_batch = 3
for epoch in range(100):
    print("epoch is", epoch)
    cnt = 0
    for line in lines:
        hdf5_name = line.strip('\r\n')
        file = h5py.File(hdf5_name, 'r')
        X_train = file['data'].value
        y_train = file['label'].value

        leng_batch = 0
        if len(X_train)%num_batch == 0:
            leng_batch = len(X_train) / num_batch
        else:
            leng_batch = (len(X_train) / num_batch) + 1

        for i in range(leng_batch):
            if (i+1)*num_batch > len(X_train):
                loss = model.train_on_batch(X_train[i * num_batch:len(X_train)],
                                            y_train[i * num_batch:len(X_train)])
            else:
                loss = model.train_on_batch(X_train[i * num_batch:(i + 1)*num_batch],
                                            y_train[i * num_batch:(i + 1)*num_batch])
            print("batch %d loss : %f" % (cnt, loss))
            cnt += 1

    if epoch % 10 == 0:
        weight_name = str("weight") + str(epoch)
        model.save_weights(weight_name, True)
