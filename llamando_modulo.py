#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 15:30:11 2018

@author: duilio
"""

#import Funciones_Imagenes as Imagen
#data=Imagen.load_dicoms("./EGGS_DIANA/")
#datainf=Imagen.info_dicom("./EGGS_DIANA/",4)
#datainf2=Imagen.plot_imagenes("./EGGS_DIANA/")

import Clase_Imagen
x = Imagen()
data=Imagen.load_dicoms("./EGGS_DIANA/")
datainf=Imagen.info_dicom("./EGGS_DIANA/",4)
datainf=Imagen.plot_imagenes("./EGGS_DIANA/")