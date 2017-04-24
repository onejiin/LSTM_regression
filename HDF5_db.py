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

save_dot_h5_file_name = '/srv/repository/HDF5/Sequence_50timestep_imageinput/train_DepFin'
save_gzip_dot_h5_file_name = '/srv/repository/HDF5/Sequence_50timestep_imageinput/train_DepFin_gzip'
save_list_dot_txt = '/srv/repository/HDF5/Sequence_50timestep_imageinput/train_DepFin_list'

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
    # zero padding and pruning for max sequence
    X_train = padding_data(X_train)
    y_train = padding_label(y_train)

    # Transform HDF5, (time step[50] * "20") = 1000 data -> 20 kinds sequences
    kind_sequences = 20
    cnt = 0
    tot_len = len(tot_sequence)*maxlen

    # X_train shape = ([total data set] [maxlen:50] [172*224])
    #for i in range(len(X_train)):
    while 1:
        if ((cnt+1)*kind_sequences) > len(X_train):
            array_1000_depth = X_train[(cnt * kind_sequences):tot_len-1]
            array_1000_label = y_train[(cnt * kind_sequences):tot_len-1]

            dot_h5_file_name = save_dot_h5_file_name + str(cnt) + '.h5'
            # gzip_dot_h5_file_name = save_gzip_dot_h5_file_name + str(cnt_3000) + '.h5'
            list_dot_txt = save_list_dot_txt + str(cnt) + '.txt'
            write_HDF5_sequence(array_1000_depth, array_1000_label, maxlen, dot_h5_file_name, list_dot_txt)
            break

        else:
            array_1000_depth = X_train[(cnt * kind_sequences):((cnt+1) * kind_sequences)]
            array_1000_label = y_train[(cnt * kind_sequences):((cnt+1) * kind_sequences)]

            dot_h5_file_name = save_dot_h5_file_name + str(cnt) + '.h5'
            #gzip_dot_h5_file_name = save_gzip_dot_h5_file_name + str(cnt_3000) + '.h5'
            list_dot_txt = save_list_dot_txt + str(cnt) + '.txt'
            write_HDF5_sequence(array_1000_depth, array_1000_label, maxlen, dot_h5_file_name, list_dot_txt)

        cnt += 1


if __name__ == '__main__':
    sys.path.append('config')
    sys.exit(main(sys.argv))
