from __future__ import print_function
import h5py
import numpy as np
from PIL import Image
import os
from keras.preprocessing import sequence

__author__ = 'wjung'

# ===================================== #
#       MAKE FOLDER SEQUENCE SET        #
# ===================================== #
def search(dirname, total_path):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        if os.path.isdir(full_filename):
            search(full_filename, total_path)
        else:
            ext = os.path.splitext(full_filename)[-1]
            if ext == '.mat':
                total_path.append(full_filename)
            #if ext == '.infra':
            #    total_path.append(full_filename)


def one_list_line(save_dir_name, value):
    str_list = 0
    if value < 10:
        #str_list = save_dir_name + '/image000' + str(value) + '.infra\n'
        str_list = save_dir_name + '/image000' + str(value) + '.mat\n'
    elif value < 100:
        #str_list = save_dir_name + '/image00' + str(value) + '.infra\n'
        str_list = save_dir_name + '/image00' + str(value) + '.mat\n'
    elif value < 1000:
        #str_list = save_dir_name + '/image0' + str(value) + '.infra\n'
        str_list = save_dir_name + '/image0' + str(value) + '.mat\n'
    elif value < 10000:
        #str_list = save_dir_name + '/image' + str(value) + '.infra\n'
        str_list = save_dir_name + '/image' + str(value) + '.mat\n'

    return str_list


def Sequence_array(dirname):
    total_path = []
    search(dirname, total_path)
    for_max_index = []

    for i in range(len(total_path)):
        index_num = total_path[i]
        save_dir_name = os.path.split(index_num)
        file_name = os.path.basename(index_num)
        file_name = os.path.splitext(file_name)
        for_max_index.append(int(file_name[0][5:]))

    tot_maxvalue = max(for_max_index)
    tot_minvalue = min(for_max_index)

    #remove min and max value
    del for_max_index[for_max_index.index(tot_minvalue)]
    str_list = one_list_line(save_dir_name[0], tot_minvalue)

    total_sequence_output = []
    one_sequence_output = [str_list]

    hp = same_loop(for_max_index, tot_minvalue, one_sequence_output, total_sequence_output, save_dir_name[0])

    return total_sequence_output


def same_loop(for_max_index, tot_minvalue, one_sequence_output, total_sequence_output, save_dir_name):

    index = 0
    all_loop_check = 0
    sequence_value = tot_minvalue + 1
    total_len = len(for_max_index)

    while 1:
        all_loop_check += 1

        # sequence find
        if sequence_value == for_max_index[index]:
            str_list = one_list_line(save_dir_name, sequence_value)
            one_sequence_output.append(str_list)
            del for_max_index[for_max_index.index(sequence_value)]
            index = 0
            all_loop_check = 0
            sequence_value += 1
        elif index == len(for_max_index) - 1:
            print("single data")
            #index += 1
        else:
            index += 1
            continue

        # next sequence
        if len(for_max_index) == 0:
            total_sequence_output.append(one_sequence_output)   # last sequence
            return 1
        elif sequence_value < min(for_max_index):
            total_sequence_output.append(one_sequence_output)
            tot_minvalue = min(for_max_index)
            del for_max_index[for_max_index.index(tot_minvalue)]
            str_list = one_list_line(save_dir_name, tot_minvalue)
            one_sequence_output = [str_list]
            hp = same_loop(for_max_index, tot_minvalue, one_sequence_output, total_sequence_output, save_dir_name)
            if hp == 1:
                return 1
# =================== End sequence set ==================== #


# ============================================================ #
#                         Binary image load                    #
# ============================================================ #
def load_data(tot_sequence_array):
    total_image = len(tot_sequence_array)

    X_data = []
    y_data = []
    for n in range(total_image):
        sequence = tot_sequence_array[n]
        # in sequence data load
        sequence_data_list = []
        sequence_label_list = []
        for single_img in range(len(sequence)):
            line = sequence[single_img]
            depth_name = line.rstrip('\n')
            [label_name, garb] = os.path.splitext(line)
            label_name += ".txt"
            # depth : [172x224], 0~1 scaling raw image
            # label : [43x56], 1channel heat map
            # label : [1x2], (x,y) coordinates
            #[depth, label] = loads(depth_name, label_name)
            [depth, label] = loads_coord(depth_name, label_name)

            sequence_data_list.append(depth)
            sequence_label_list.append(label)

        X_data.append(sequence_data_list)
        y_data.append(sequence_label_list)

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return (X_data, y_data)


def loads_coord(depth_name, label_name):
    # LABEL ====================================================
    f = open(label_name)
    lines = f.readlines()
    for txt_lines in lines:
        txt = txt_lines.split(' ')
    f.close()
    # ==========================================================
    # Data =====================================================
    # do not use 2channel (no not use depth)
    depth_bin = np.fromfile(depth_name, dtype=np.uint16)
    cols = depth_bin[0] #224
    #depth_bin[1]
    rows = depth_bin[2] + 1 #171
    #depth_bin[3]
    sizeof_int = depth_bin[4]
    #depth_bin[5]
    #max depth
    depth_max = np.max(depth_bin[6:])
    data = np.empty((rows, cols))
    #label = np.zeros((label_rows, label_cols))
    #label = np.zeros((1, 3))
    label = np.zeros((1, 2))
    # ==========================================================

    for i in range(rows):
        for j in range(cols):
            if i == 171:
                data[i, j] = 0
            else:
                data[i, j] = float(depth_bin[6 + (i*cols) + j]) / float(depth_max)  # zero mean

    #label[0, 0] = str(int(float(txt[0])))
    #label[0, 1] = str(int(float(txt[1])))
    #label[0, 2] = str(int(float(txt[2])))
    label[0, 0] = str((float(txt[0])))
    label[0, 1] = str((float(txt[1])))
    #label[0, 2] = str((float(txt[2])))

    return (data, label)
# ==================== End binary data load ==================== #

# ============================================================== #
#                         padding and pruning                    #
#                 from keras.preprocessing import sequence       #
# ============================================================== #
maxlen = 50     # time sequence

def padding_data(data):
    data = sequence.pad_sequences(data, dtype='float', maxlen=maxlen)
    data = data.reshape((len(data), maxlen, 1, 172, 224))
    #data = data.reshape((len(data), maxlen, 172*224))

    return data


def padding_label(data):
    #data = sequence.pad_sequences(data, maxlen=maxlen) # heat map int type
    data = sequence.pad_sequences(data, dtype='float', maxlen=maxlen)
    #data = data.reshape((len(data), maxlen, 1*3))
    data = data.reshape((len(data), maxlen, 1*2))

    return data
# ==================== End padding and pruning ==================== #


# ================================================================= #
#                             Write HDF5 DB                         #
# ================================================================= #

def write_HDF5_sequence(line_depth, line_label, time_step, dot_h5_file_name, list_dot_txt):
    with h5py.File(dot_h5_file_name, 'w') as hf:
        hf['data'] = line_depth
        hf['label'] = line_label

