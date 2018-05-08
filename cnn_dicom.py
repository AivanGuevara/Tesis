#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 11:05:46 2018

@author: duilio
"""

# dicom_cnn.py
#from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
 
#from keras.datasets import mnist
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Convolution2D, Convolution3D, MaxPooling2D, MaxPooling3D
from keras.utils import np_utils
#from keras import backend as K
#
import extract_dicom as ext## ESTE LO COMENTO
# 
batch_size = 2
nb_classes = 2
nb_epoch = 3
# 
folder="/home/duilio/Escritorio/EGGS_DIANA/datas"
## number of convolutional filters to use
#nb_filters = 32
## size of pooling area for max pooling
#pool_size = (2, 2)
## convolution kernel size
#kernel_size = (3, 3)

# the data, shuffled and split between train and test sets

(X_train, y_train), (X_test, y_test) = ext.extract_dataset(folder)## ESTE LO COMENTO
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
 
# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)