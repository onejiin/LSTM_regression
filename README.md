**Reference:** https://github.com/keras-team/keras/blob/master/examples/imdb_cnn_lstm.py

LSTM regressor with Keras. It is built by image sequence estimation. 
First I extract feature using CNN and then apply to LSTM.
It uses HDF5 database, which consist of image('data') and label('label').
If you want to change your data and label, you can make HDF5 db in our code in HDF_db.py

## Usage step:

1. Make Database: Sequence image(file_name) & real value label(x,y) 
 1) open HDF5_db.py
   .  
   . File must be read in list. (dir *.* /b /s >> list.scp)
 2) Change parameter: it is **hardcoding**
   . save_dot_h5_file_name / save_gzip_dot_h5_file_name / save_list_dot_txt

2. Learning to Train: lstm_regression_train.py
 - preparing: training DB by list, / READ_HDF5_train = 'list_file_name'
 - Convolution filter, data input, and time-step size are really **hardcoding**

3. Model structure
 - [Input] 172x224x1 image -> [Feature extractor] (Conv-Relu-Pooling)*2 -> [Sequence] (172/4)x(224/4) to 2(x,y)
 
4. Test
