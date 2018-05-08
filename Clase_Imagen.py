#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:45:18 2018

@author: duilio
"""
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import os
import numpy

class Imagen:
    def __init__(self):
#    def __init__(self, partereal, parteimaginaria):
        self.columna = []
        self.filas = ()

    def info_dicom(directory, numero):

        lstFilesDCM = []  # create an empty list
        for dirName, subdirList, fileList in os.walk(directory):
            for filename in fileList:
                if ".dcm" in filename.lower():  # check whether the file's DICOM
                    lstFilesDCM.append(os.path.join(dirName,filename))
        RefDs = pydicom.dcmread(lstFilesDCM[numero])
        plt.imshow(RefDs.pixel_array, cmap=plt.cm.bone)
        plt.show()
        print ('numero de columnas',int(RefDs.Rows))
        print ('numero de filas',int(RefDs.Columns))
        print ('numero de imagenes', len(lstFilesDCM))
        
    def load_dicoms(directory):
#        filelist = [ name for name in os.listdir(directory) if name.find(".dcm") > 0 ]
#        filelist.sort()
#        num_images = len(filelist)  
#        counter = 0
#        filenames = []        
#        for filename in filelist:
#            filenames.append(filename)
#            image = pydicom.dcmread(os.path.join(directory, filename))  # filename is two directories down from where this is running
#            rows = image.Rows
#            assert rows == 512
#            cols = image.Columns
#            assert cols == 512
#            data = np.empty((num_images, rows, cols, 1))
#            for x in range(0, rows):
#                for y in range(0, cols):
#                    data[counter][x][y] = image.pixel_array[x][y]
#            counter = counter + 1
#   
#        return data
        #--------------------------------------------------------
        
        lstFilesDCM = []  # create an empty list
        for dirName, subdirList, fileList in os.walk(directory):
            for filename in fileList:
                if ".dcm" in filename.lower():  # check whether the file's DICOM
                    lstFilesDCM.append(os.path.join(dirName,filename))
        lstFilesDCM.sort()
        RefDs = pydicom.dcmread(lstFilesDCM[1])
        ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
        ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
        for filenameDCM in lstFilesDCM:
            ds = pydicom.dcmread(filenameDCM)
            ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array  
        return ArrayDicom
    
    def plot_imagenes(directory):
        filelist = [ name for name in os.listdir(directory) if name.find(".dcm") > 0 ]
        filelist.sort()
        num_images = len(filelist)  
        filenames = []        
        for filename in filelist:
            filenames.append(filename)
            image = pydicom.dcmread(os.path.join(directory, filename)) 
            plt.imshow(image.pixel_array, cmap=plt.cm.bone)
            plt.show()
        
        
#x = Imagen()
#data=Imagen.load_dicoms("./EGGS_DIANA/")
#datainf=Imagen.info_dicom("./EGGS_DIANA/",4)
#datainf=Imagen.plot_imagenes("./EGGS_DIANA/")