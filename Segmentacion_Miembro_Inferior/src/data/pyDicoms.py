#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 06:58:28 2018

@author: duilio
"""

import sys
import os
import numpy
#from matplotlib import pyplot, cm
import stat
import numpy as np
import pydicom
import matplotlib.pyplot as plt

#directorio =  '/home/duilio/Escritorio/EGGS_DIANA' 
#lista_archivos = os.listdir(directorio ) #lista con los nombres de los archivos
#l = len(lista_archivos) #longitud de la carpeta , numero de imagenes 
#
#file_name = os.path.join(directorio, f)#entra al directorio 



PathDicom = "./EGGS_DIANA/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

# Get ref file
RefDs = pydicom.dcmread(lstFilesDCM[4])
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])
# The array is sized based on 'ConstPixelDims'
ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

# loop through all the DICOM files
for filenameDCM in lstFilesDCM:
    # read the file
    ds = pydicom.dcmread(filenameDCM)
    # store the raw image data
    ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array  

#pyplot.figure(dpi=300)
#pyplot.axes().set_aspect('equal', 'datalim')
#pyplot.set_cmap(pyplot.gray())
#pyplot.pcolormesh(x, y, numpy.flipud(ArrayDicom[:, :, 80]))


dataset = pydicom.dcmread(lstFilesDCM[2])
plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
plt.show()
