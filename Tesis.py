# python Tesis.py -o /home/duilio/Documentos/Tesis_github/IMG -t bone


import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt

import pydicom
from PIL.Image import fromarray
import keras_metrics

import matplotlib.animation as animation

from keras.models import Model;
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2D, Reshape, Conv2DTranspose;
from keras.layers import add, concatenate;
from keras.layers.normalization import BatchNormalization;
from keras.optimizers import Adadelta, Adam;
from keras.layers.core import Activation;

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()


ap.add_argument("-o", "--output", required=True,
	help="path to output dataset")
ap.add_argument("-DICOM",
                    help = "Image DICOM",
                    action = 'store_true',
                    default = None)
ap.add_argument("-t", "--tissue", help="type of tissue")

args = vars(ap.parse_args())





def get_unet(inputs, n_classes):

    x = BatchNormalization()(inputs)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(n_classes, (1, 1), activation='linear')(conv9)
    
    return conv10

def load_dicoms(directory):
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(directory):
        for filename in fileList:
#             if ".dcm" in filename.lower():  # check whether the file's DICOM
            if "" in filename.lower(): 
                lstFilesDCM.append(os.path.join(dirName,filename))

# Get ref file
    RefDs = pydicom.dcmread(lstFilesDCM[1])
    ConstPixelDims = ( len(lstFilesDCM), int(RefDs.Rows), int(RefDs.Columns))

# Load spacing values (in mm)

    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    ArrayBNP = []
    
# loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
    # read the file
        ds = pydicom.dcmread(filenameDCM)
    # store the raw image data
        ArrayDicom[lstFilesDCM.index(filenameDCM), :, :] = ds.pixel_array  
        im = fromarray(ds.pixel_array)
        im = im.convert('I')
#         ArrayBNP[lstFilesDCM.index(filenameDCM), :, :] = im
        ArrayBNP.append(ds.pixel_array/255.0)
    return ArrayDicom, ArrayBNP, lstFilesDCM




#Parameters
INPUT_CHANNELS = 3
NUMBER_OF_CLASSES = 4
IMAGE_W = 512
IMAGE_H = 512
loss_name = "categorical_crossentropy"

def get_model():
    
    inputs = Input((IMAGE_H, IMAGE_W, INPUT_CHANNELS))
    
    base = get_unet(inputs, NUMBER_OF_CLASSES)
    
    # softmax
    reshape= Reshape((-1,NUMBER_OF_CLASSES))(base)
    act = Activation('softmax')(reshape)
    
    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=Adadelta(), loss='categorical_crossentropy', metrics=[keras_metrics.precision(), keras_metrics.recall()])
    
#     print(model.summary())
    
    return model

model = get_model()
model.load_weights('model_weights_3mask_improved'+loss_name+'.h5')







BMP_IMAGE_DIR = Path(args["output"])


print(BMP_IMAGE_DIR)
lista2 = []
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(BMP_IMAGE_DIR):
    for i in range(500):
        titulo='Imagen00' + str(i+1) + '.bmp'
        lstFilesDCM.append(os.path.join(dirName,titulo))
        img = cv2.imread(os.path.join(dirName,titulo))
        if img is not None:
            lista2.append(img[:,:])
img_vol= [lista2[i] for i in range(len(lista2)-1, -1, -1)]

img_vol = np.array(img_vol, dtype=np.float32)    
img_vol /= 255.0 # [0,1]

print('Images', img_vol.shape)

import matplotlib
BMP_IMAGE_DIR3 = os.path.join(BMP_IMAGE_DIR,'SEG')



ArrayBNP, Array = [],[]
size = 512,512,3
for i in range(len(img_vol)):
    print(i)
    hueso = np.zeros(size)
    img = img_vol[i]#Cambiar el valor aca

    y_pred= model.predict(img[None,...].astype(np.float32))
    y_pred= y_pred.reshape((IMAGE_H,IMAGE_W,NUMBER_OF_CLASSES))
    imagen_negra = np.zeros(size)
    if args["tissue"] == "conective":
        imagen_negra[:,:,0]= y_pred[:,:,0]
        ArrayBNP.append(imagen_negra)
    elif args["tissue"] == "bone":
        imagen_negra[:,:,1]= y_pred[:,:,1]
        ArrayBNP.append(imagen_negra)
        Array.append(y_pred)
        
        for p in range(img.shape[0]):
            for q in range(img.shape[1]):
                if y_pred[q,p,1] > 0.003:
                    hueso[q,p]= 1
    elif args["tissue"] == "muscule":
        imagen_negra[:,:,2]= y_pred[:,:,2]
        ArrayBNP.append(imagen_negra)
    else:
        imagen_negra[:,:,0]= y_pred[:,:,0]
        imagen_negra[:,:,1]= y_pred[:,:,1]
        imagen_negra[:,:,2]= y_pred[:,:,2]
        ArrayBNP.append(imagen_negra)
    

    BMP_IMAGE_DIR2 = os.path.join(BMP_IMAGE_DIR3,'Seg' + str(i))
    print(BMP_IMAGE_DIR2)
    #matplotlib.image.imsave(BMP_IMAGE_DIR2 + '.bmp', imagen_negra)
    matplotlib.image.imsave(BMP_IMAGE_DIR2 + '.bmp', hueso)

