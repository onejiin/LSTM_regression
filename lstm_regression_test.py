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
from keras.models import load_model
from keras.models import model_from_yaml

import shutil
from random import shuffle
from function import *

import cv2, cv


def padding_data(data):
    data = sequence.pad_sequences(data, dtype='float', maxlen=maxlen)
    #data = data.reshape((len(data), maxlen, 1, 172, 224))
    data = data.reshape((len(data), maxlen, 172 * 224))

    return data


def padding_label(data):
    #data = sequence.pad_sequences(data, maxlen=maxlen) # heat map int type
    data = sequence.pad_sequences(data, dtype='float', maxlen=maxlen)
    #data = data.reshape((len(data), maxlen, 1*3))
    data = data.reshape((len(data), maxlen, 1 * 2))

    return data


def model_define():
    model = Sequential()

    model.add(LSTM(256, return_sequences=True,
                   input_shape=(lstm_output_size, 172 * 224)))
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
READ_test = '/srv/repository/Ddrive_database/Genesis_EyeFingerTagg/folder_list_test.scp'

f = open(READ_test)
lines = f.readlines()
tot_sequence = []
for line in lines:
    dirname = line.strip('\r\n')
    tot_sequence += Sequence_array(dirname)
#shuffle(tot_sequence)
(X_test, y_test) = load_data(tot_sequence)
f.close()
print ('test data load end')

X_test = padding_data(X_test)
y_test = padding_label(y_test)

model = model_define()
model.load_weights('./weight')

predictions = model.predict(X_test, batch_size=1, verbose=0)

print (predictions)
fw = open('./coord_predict.txt','w')

for i in range(predictions.shape[0]):
    for j in range(predictions.shape[1]):
        for k in range(predictions.shape[2]):
            str_out = str(predictions[i][j][k])
            fw.writelines(str_out)
            fw.writelines(' ')
        fw.writelines('\n')
fw.close()

fw_label = open('./coord_label.txt','w')
for i in range(y_test.shape[0]):
    for j in range(y_test.shape[1]):
        for k in range(y_test.shape[2]):
            str_out = str(y_test[i][j][k])
            fw_label.writelines(str_out)
            fw_label.writelines(' ')
        fw_label.writelines('\n')
fw_label.close()

'''
predictions_map = predictions.reshape((len(predictions), maxlen, 1, 43, 56))

cnt = 0
for i in range(len(predictions)):
    for j in range(maxlen):
        dest = np.zeros((43, 56, 1), dtype="uint8")
        dest = predictions_map[i, j, 0]*255
        max_value = np.amax(dest)
        for k in range(43):
            for p in range(56):
                #if dest[k,p] > 255:
                if dest[k, p] == max_value:
                    dest[k, p] = 255

        resize_dest = cv2.resize(dest, (224, 171))
        strr = "tmp" + str(cnt) + ".png"
        cnt += 1
        cv2.imshow('tmp', resize_dest)
        #cv2.waitKey(0)
        cv2.imwrite(strr, resize_dest)
'''