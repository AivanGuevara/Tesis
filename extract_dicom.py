#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 19:43:30 2018

@author: duilio
"""
#extract_dicom.py
import numpy as np
import pydicom
#import dicom
import os
 
def load_dicoms(directory):
 
    filelist = [ name for name in os.listdir(directory) if name.find(".dcm") > 0 ]
    filelist.sort()
    num_images = len(filelist)  
    counter = 0
 
    filenames = []
    data = np.empty((num_images, 512, 512, 1))   # Each DaTsan image contains 320 x 320 pixels  
 
    for filename in filelist:
        filenames.append(filename)
        image = pydicom.dcmread(os.path.join(directory, filename))  # filename is two directories down from where this is running
        rows = image.Rows
        assert rows == 512
        cols = image.Columns
        assert cols == 512
 
        for x in range(0, 512):
            for y in range(0, 512):
              data[counter][x][y] = image.pixel_array[x][y]
        counter = counter + 1
   
    save_cache(directory, data)
 
    return data

def load_cache_dicoms(directory):
    return np.load(os.path.join(directory, '.cache.npy'))
   
def save_cache(directory, data):
   
    np.save(os.path.join(directory, '.cache'), data)
 
def cache_is_present(directory):
   
    return ".cache.npy" in os.listdir(directory)

def extract_images(directory):
   
    print('Extracting', directory)
 
    if cache_is_present(directory):
        print('Cache found')
        return load_cache_dicoms(directory)
    else:
        print('No cache')
        return load_dicoms(directory)
 
#    #------PRUEBA DE EXTRAER
#def extract_images(filename):
#  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
#  print('Extracting', filename)
#  with gzip.open(filename) as bytestream:
#    magic = _read32(bytestream)
#    if magic != 2051:
#      raise ValueError(
#          'Invalid magic number %d in MNIST image file: %s' %
#          (magic, filename))
#    num_images = _read32(bytestream)
#    rows = _read32(bytestream)
#    cols = _read32(bytestream)
#    buf = bytestream.read(rows * cols * num_images)
#    data = numpy.frombuffer(buf, dtype=numpy.uint8)
#    data = data.reshape(num_images, rows, cols, 1)
#    return data
#    
    ##PRUEBA2 DE EXTRAER
def extract_dataset(folder, bounds = .2):
 
    dirs = os.listdir(folder)
    dirs = [ label for label in dirs if label[0] != '.' ]
    dirs.sort()
#    dirs = [ name for name in os.listdir(folder) if name.find(".dcm") > 0 ]
#    dirs.sort()
    # print labels
    print(dirs)

 
    data_train = []
    data_test = []
    label_train = []
    label_test = []
 
    index = 0
    for label in dirs:
 
        subdir = os.path.join(folder, label)
#        subdirs = [ name for name in os.listdir(folder) if name.find(".dcm") > 0 ]
        subdirs = [ label for label in os.listdir(subdir) if label[0] != '.' ]
        print('dirs1')
        subdir_index = 0
       
        for container in subdirs:
            print(bounds * len(subdirs))#esto me 0.4
            data = extract_images(os.path.join(subdir, container))
 
            if subdir_index > bounds * len(subdirs):
                new_train = []
 
                for img in data:
                    new_train.append(img)
                print('aca estoy')
                data_train.append(new_train)
                label_train.append(index)
            else:
                new_test = []
 
                for img in data:
                    new_test.append(img)
                print('aca no estoy')
                data_test.append(new_test)
                label_test.append(index)
 
            subdir_index += 1
 
        index += 1
 
    data_train = np.array(data_train)#este me da bien
    data_test = np.array(data_test)#este me da mal
    label_train = np.array(label_train)#este me da bien
    label_test = np.array(label_test)#este me da mal

    return (data_train, label_train), ( data_test, label_test )

folder="/home/duilio/Escritorio/EGGS_DIANA/datas"
(X_train, y_train), (X_test, y_test) = extract_dataset(folder)