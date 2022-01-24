
import tensorflow.compat.v1 as tf
from keras import backend as K
from keras.models import Model
from keras.callbacks import TensorBoard,ModelCheckpoint

import glob
import os

import cv2
from math import sin, cos, radians
import random
import numpy as np
from keras.models import Sequential,load_model,model_from_json
from keras.utils import to_categorical

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

#######################################################################

x_Data=[]
for filename in glob.glob('./no_tumor/*.jpg'):
    
    img=cv2.imread(filename,0)
    img=cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR) 
    img=np.array(img)    
    inputdata = np.reshape(img, (img.shape[0],img.shape[1]))
    inputdata=np.array(inputdata)
    x_Data.append(inputdata.tolist())
    x_Data=x_Data

x_Data = np.array(x_Data)
print(x_Data.shape)
x_Data = np.reshape(x_Data, (x_Data.shape[0],x_Data.shape[1],x_Data.shape[2],1))
x_Data = x_Data.astype('float32')
    
    
Data_datagen1 = ImageDataGenerator(rotation_range=90)
Data_datagen2 = ImageDataGenerator(vertical_flip=True)


Data_total1 = 0
Data_total2 = 0

DataCount=x_Data.shape[0]

Data_imageGen1 = Data_datagen1.flow(x_Data, batch_size=1, save_to_dir='./AllData',
    save_prefix="no", save_format="jpg")
Data_imageGen2 = Data_datagen2.flow(x_Data, batch_size=1, save_to_dir='./AllData',
    save_prefix="no", save_format="jpg")


for image in Data_imageGen1:
    
    Data_total1 += 1
    
    if Data_total1 == DataCount:
        
        break
    
for image in Data_imageGen2:
    
    Data_total2 += 1
    
    if Data_total2 == DataCount:
        break


		
