#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 11:05:46 2018

@author: duilio
"""

# dicom_cnn.py
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Convolution3D, MaxPooling2D, MaxPooling3D
from keras.utils import np_utils
from keras import backend as K
from keras.utils.data_utils import Sequence
from keras.preprocessing.image import ImageDataGenerator

import extract_dicom as ext## ESTE LO COMENTO
# 
batch_size = 2
nb_classes = 2
nb_epoch = 3
# 
folder="/home/duilio/Escritorio/EGGS_DIANA/datas"
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets

(x_train, y_train), (x_test, y_test) = ext.extract_dataset(folder)## ESTE LO COMENTO
print('X_train shape:', x_train.shape)
print('X_test shape:', x_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
 
# Convert class vectors to binary class matrices.
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
#
#y_train = np_utils.to_categorical(y_train, num_classes)
#y_test = np_utils.to_categorical(y_test, num_classes)


#
#
#model = Sequential()
# 
#model.add(Convolution3D(32, 3, 3, 3, border_mode='same', input_shape=(3, 320, 320, 1)))
#model.add(Activation('relu'))
##model.add(Convolution3D(32, 3, 3, 3))
##model.add(Activation('relu'))
#model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#model.add(Dropout(0.25))
# 
##model.add(Convolution3D(64, 3, 3, 3, border_mode='same'))
##model.add(Activation('relu'))
##model.add(Convolution3D(64, 3, 3, 3))
##model.add(Activation('relu'))
##model.add(MaxPooling3D(pool_size=(2, 2, 2)))
##model.add(Dropout(0.25))
# 
#model.add(Flatten())
#model.add(Dense(512))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(nb_classes))
#model.add(Activation('softmax'))
# 
## Let's train the model using RMSprop
#model.compile(loss='categorical_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])
# 
#model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#          verbose=1, validation_data=(x_test, y_test))
# 
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])
