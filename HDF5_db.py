import os
import sys
import shutil
from random import shuffle
import re
import random

from HDF5_function import *

__author__ = 'wjung'

#READ = './folder_list_small_set.scp'
READ = './folder_list_train.scp'

save_dot_h5_file_name = '/srv/repository/HDF5/Sequence_50timestep_heatmap/train_DepFin'
save_gzip_dot_h5_file_name = '/srv/repository/HDF5/Sequence_50timestep_heatmap/train_DepFin_gzip'
save_list_dot_txt = '/srv/repository/HDF5/Sequence_50timestep_heatmap/train_DepFin_list'


def hdf5_batch(X_train, y_train, hdf_cnt):
    # Transform HDF5, (time step[50] * "20") = 1000 data -> 20 kinds sequences
    kind_sequences = 20
    cnt = 0
#    tot_len = len(tot_sequence) * maxlen

    # X_train shape = ([total data set] [maxlen:50] [172*224])
    # for i in range(len(X_train)):
    while 1:
        if ((cnt + 1) * kind_sequences) >= len(X_train):
            array_1000_depth = X_train[(cnt * kind_sequences):]
            array_1000_label = y_train[(cnt * kind_sequences):]

            dot_h5_file_name = save_dot_h5_file_name + str(hdf_cnt) + '.h5'
            # gzip_dot_h5_file_name = save_gzip_dot_h5_file_name + str(cnt_3000) + '.h5'
            list_dot_txt = save_list_dot_txt + str(hdf_cnt) + '.txt'
            write_HDF5_sequence(array_1000_depth, array_1000_label, maxlen, dot_h5_file_name, list_dot_txt)
            hdf_cnt += 1
            break

        else:
            array_1000_depth = X_train[(cnt * kind_sequences):((cnt + 1) * kind_sequences)]
            array_1000_label = y_train[(cnt * kind_sequences):((cnt + 1) * kind_sequences)]

            dot_h5_file_name = save_dot_h5_file_name + str(hdf_cnt) + '.h5'
            # gzip_dot_h5_file_name = save_gzip_dot_h5_file_name + str(cnt_3000) + '.h5'
            list_dot_txt = save_list_dot_txt + str(hdf_cnt) + '.txt'
            write_HDF5_sequence(array_1000_depth, array_1000_label, maxlen, dot_h5_file_name, list_dot_txt)

        cnt += 1
        hdf_cnt += 1

    return hdf_cnt


def main(argv):
    # make sequence set
    f = open(READ)
    lines = f.readlines()
    tot_sequence = []
    for line in lines:
        dirname = line.strip('\r\n')
        tot_sequence += Sequence_array(dirname)
    f.close()

    # shuffling and total database load
    shuffle(tot_sequence)
    (X_train, y_train) = load_data(tot_sequence)
    print('load train database...')

    hdf_cnt = 0
    last_index = 0

    for index in range(int(X_train.shape[0]/batch_size)):
        X_batch = X_train[index * batch_size:(index+1) * batch_size]
        y_batch = y_train[index * batch_size:(index+1) * batch_size]

        X_batch = padding_data(X_batch)
        y_batch = padding_label(y_batch)

        prev_cnt = hdf5_batch(X_batch, y_batch, hdf_cnt)

        hdf_cnt = prev_cnt
        last_index = index

        print ("train : "), index * batch_size, ("~"), (index+1) * batch_size

#    print ("hdf_cnt"), hdf_cnt
#    print int(X_train.shape[0]), (","), int((last_index+1) * batch_size)
    print ("train : "), (last_index+1) * batch_size, ("~"), X_train.shape[0]
    #remain batch index
    X_batch = X_train[(last_index+1) * batch_size:]
    y_batch = y_train[(last_index+1) * batch_size:]
    X_batch = padding_data(X_batch)
    y_batch = padding_label(y_batch)

    prev_cnt = hdf5_batch(X_batch, y_batch, hdf_cnt)

    print prev_cnt


if __name__ == '__main__':
    sys.path.append('config')
    sys.exit(main(sys.argv))
